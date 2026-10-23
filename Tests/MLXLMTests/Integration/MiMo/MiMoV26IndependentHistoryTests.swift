import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM

@Suite(.tags(.integration), .serialized)
struct MiMoV26IndependentHistoryTests {
    @Test func pageStorageMatchesScalarPhysicalRingsAndChunkMasks() {
        Device.withDefaultDevice(.cpu) {
            for window in [nil, 8, 65] as [Int?] {
                let ordinary: any KVCache = if let window { RotatingKVCache(maxSize: window) } else { KVCacheSimple() }
                let paged = MiMoV26IndependentHistoryCache(modelIdentity: UUID(), requestIdentity: UUID(),
                    layerIndex: 0, maximumContext: 1024, heads: 2, keyWidth: 16, valueWidth: 8,
                    dtype: .bfloat16, window: window)
                #expect(paged.allocatedBytes == 0 && paged.pageCount == 0)
                for count in [1, 7, 1, 55, 1, 65, 1, 1, 3, 1] {
                    let start = ordinary.offset
                    let keys = values([1, 2, count, 16], salt: start)
                    let vals = values([1, 2, count, 8], salt: start + 29)
                    let aMask = ordinary.makeMask(n: count, windowSize: window, returnArray: true)
                    let bMask = paged.makeMask(n: count, windowSize: window, returnArray: true)
                    switch (aMask, bMask) {
                    case (.none, .none), (.causal, .causal): break
                    case (.array(let a), .array(let b)): equal(a, b)
                    default: Issue.record("page mask changed")
                    }
                    let a = ordinary.update(keys: keys, values: vals)
                    let b = paged.update(keys: keys, values: vals)
                    equal(a.0, b.0); equal(a.1, b.1)
                    #expect(ordinary.offset == paged.offset)
                    #expect(paged.pageCount == (b.0.dim(2) + 63) / 64)
                    #expect(paged.allocatedBytes == paged.pageCount * 64 * 2 * (16 + 8) * 2)
                }
                let copy = paged.copy()
                let prior = copy.state.map(bytes)
                _ = paged.update(keys: values([1, 2, 1, 16], salt: 901),
                    values: values([1, 2, 1, 8], salt: 902))
                eval(paged.innerState())
                #expect(copy.state.map(bytes) == prior)
            }
        }
    }

    @Test func packedNativeRowsMatchSingletonLogitsResidualsAndLiveKV() throws {
        try Device.withDefaultDevice(.cpu) {
            for tied in [false, true] {
                let target = try model(tied: tied)
                for width in [1, 2, 4, 8] {
                    let ordinary = (0..<width).map { _ in target.newCache() }
                    let paged = try (0..<width).map { _ in try target.makeIndependentHistoryCaches() }
                    for row in 0..<width {
                        let history = 1 + row * 19
                        var offset = 0
                        while offset < history {
                            let count = min([7, 64, 9][row % 3], history - offset)
                            let ids = tokens((0..<count).map { ($0 + offset + row) % 61 })
                            let a = try target.forward(inputIDs: ids, cache: ordinary[row], captureLayers: [0, 1])
                            let b = try target.forwardIndependentHistory(inputIDs: ids, caches: paged[row], captureLayers: [0, 1])
                            compare(a, b); compareCache(ordinary[row], paged[row])
                            offset += count
                        }
                    }
                    // Reorder, remove rows, and add a fresh request among survivors.
                    var order = Array((0..<width).reversed())
                    for step in 0..<7 {
                        let next = order.map { ($0 * 7 + step + 3) % 61 }
                        let actual = try target.decodeIndependentHistories(
                            inputIDs: MLXArray(next.map(Int32.init)).reshaped(order.count, 1),
                            caches: order.map { paged[$0] }, captureLayers: [0, 1])
                        for (index, row) in order.enumerated() {
                            let expected = try target.forward(inputIDs: tokens([next[index]]),
                                cache: ordinary[row], captureLayers: [0, 1])
                            compare(expected, actual[index]); compareCache(ordinary[row], paged[row])
                        }
                        if step == 2 && order.count > 1 { order.removeLast() }
                        order.reverse()
                    }
                    if width > 1 {
                        let newcomer = try target.makeIndependentHistoryCaches(), reference = target.newCache()
                        let survivors = Array(order.prefix(1))
                        let actual = try target.decodeIndependentHistories(inputIDs: tokens([5, 6]).reshaped(2, 1),
                            caches: [paged[survivors[0]], newcomer], captureLayers: [0, 1])
                        compare(try target.forward(inputIDs: tokens([5]), cache: ordinary[survivors[0]], captureLayers: [0, 1]), actual[0])
                        compare(try target.forward(inputIDs: tokens([6]), cache: reference, captureLayers: [0, 1]), actual[1])
                        compareCache(reference, newcomer)
                    }
                }
            }
        }
    }

    @Test func splitStageResidualKeepsOriginalNormBoundary() throws {
        try Device.withDefaultDevice(.cpu) {
            let full = try model(tied: false), first = try model(tied: false, layers: 0..<1), last = try model(tied: false, layers: 1..<2)
            let wholeCache = (0..<4).map { _ in full.newCache() }
            let aCache = try (0..<4).map { _ in try first.makeIndependentHistoryCaches() }
            let bCache = try (0..<4).map { _ in try last.makeIndependentHistoryCaches() }
            for row in 0..<4 {
                let ids = tokens((0..<(row * 7 + 1)).map { $0 % 61 })
                let expected = try full.forward(inputIDs: ids, cache: wholeCache[row], captureLayers: [0])
                let a = try first.forwardIndependentHistory(inputIDs: ids, caches: aCache[row], captureLayers: [0])
                let residual = try #require(a.layerFeatures[0])
                equal(try #require(expected.layerFeatures[0]), residual)
                let b = try last.forwardIndependentHistory(embeddings: residual, caches: bCache[row])
                equal(expected.logits, b.logits)
            }
            for step in 0..<4 {
                let ids = MLXArray((0..<4).map { Int32($0 + step + 17) }).reshaped(4, 1)
                let a = try first.decodeIndependentHistories(inputIDs: ids, caches: aCache, captureLayers: [0])
                let residual = try concatenated(a.map { try #require($0.layerFeatures[0]) }, axis: 0)
                let b = try last.decodeIndependentHistories(embeddings: residual, caches: bCache)
                for row in 0..<4 {
                    let expected = try full.forward(inputIDs: ids[row..<row + 1], cache: wholeCache[row], captureLayers: [0])
                    equal(try #require(expected.layerFeatures[0]), try #require(a[row].layerFeatures[0]))
                    equal(expected.logits, b[row].logits)
                    compareCache([wholeCache[row][0]], aCache[row]); compareCache([wholeCache[row][1]], bCache[row])
                }
            }
        }
    }

    @Test func invalidRowsRefuseBeforeAnyHistoryAdvance() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try model(tied: false), foreign = try model(tied: false)
            let a = try target.makeIndependentHistoryCaches(), b = try target.makeIndependentHistoryCaches()
            let f = try foreign.makeIndependentHistoryCaches()
            for rows in [[a, a], [a, f], [a, [b[1], b[0]]], [a, [b[0]]]] {
                #expect(throws: MiMoV26ExecutionError.self) {
                    try target.decodeIndependentHistories(inputIDs: tokens([3, 4]).reshaped(2, 1), caches: rows)
                }
                #expect(a.allSatisfy { $0.offset == 0 } && b.allSatisfy { $0.offset == 0 })
            }
            #expect(throws: MiMoV26ExecutionError.self) {
                try target.decodeIndependentHistories(inputIDs: tokens([3, 64]).reshaped(2, 1), caches: [a, b])
            }
            #expect(throws: MiMoV26ExecutionError.self) {
                try target.decodeIndependentHistories(embeddings: MLXArray.zeros([2, 1, 64], dtype: .float32), caches: [a, b])
            }
            #expect(a.allSatisfy { $0.offset == 0 } && b.allSatisfy { $0.offset == 0 })
            // Ordinary public forward must not silently admit the new storage.
            #expect(throws: MiMoV26ExecutionError.self) { try target.forward(inputIDs: tokens([3]), cache: a) }
        }
    }

    @Test func wrappedVerifyRollbackMatchesOriginalWindowAndNextForward() throws {
        try Device.withDefaultDevice(.cpu) {
            for keeping in 1...4 {
                let target = try model(tied: false)
                let ordinary = target.newCache(), pages = try target.makeIndependentHistoryCaches()
                // One-token appends wrap the sliding ring, then the verify
                // window's multi-token update makes its physical order temporal.
                for step in 0..<19 {
                    let ids = tokens([step % 61])
                    compare(try target.forward(inputIDs: ids, cache: ordinary),
                        try target.forwardIndependentHistory(inputIDs: ids, caches: pages))
                }
                let ids = tokens([23, 24, 25, 26])
                compare(try target.forward(inputIDs: ids, cache: ordinary),
                    try target.forwardIndependentHistory(inputIDs: ids, caches: pages))
                let removed = 4 - keeping
                for cache in pages { try cache.validateWindowRollback(by: removed) }
                for cache in pages { try cache.rollbackWindow(by: removed) }
                if removed > 0 {
                    for cache in ordinary {
                        if let rotating = cache as? RotatingKVCache {
                            var meta = rotating.metaState; let state = rotating.innerState()
                            let n = state[0].dim(2) - removed
                            rotating.state = state.map { $0[.ellipsis, ..<n, 0...] }
                            meta[3] = String(rotating.offset - removed); meta[4] = String(n)
                            rotating.metaState = meta
                        } else { #expect(cache.trim(removed) == removed) }
                    }
                    #expect(throws: MiMoV26ExecutionError.self) { try pages[0].rollbackWindow(by: 1) }
                }
                compareCache(ordinary, pages)
                for next in [[31], [32, 33, 34], [35]] {
                    compare(try target.forward(inputIDs: tokens(next), cache: ordinary),
                        try target.forwardIndependentHistory(inputIDs: tokens(next), caches: pages))
                    compareCache(ordinary, pages)
                }
            }
        }
    }

    @Test func checkedRestorePreservesWrappedPhysicalOrderAndOwnership() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = try model(tied: false)
            let source = try target.makeIndependentHistoryCaches()
            for step in 0..<19 {
                _ = try target.forwardIndependentHistory(inputIDs: tokens([step % 61]), caches: source)
            }
            let before = source.map { $0.state.map(bytes) }
            let destination = try target.makeIndependentHistoryCaches()
            for (left, right) in zip(source, destination) {
                let state = left.state
                let index = left.maxSize == nil ? left.offset : Int(left.metaState[4])!
                try right.validateRestore(keys: state[0], values: state[1], offset: left.offset, index: index)
            }
            for (left, right) in zip(source, destination) {
                let state = left.state
                let index = left.maxSize == nil ? left.offset : Int(left.metaState[4])!
                try right.restore(keys: state[0], values: state[1], offset: left.offset, index: index)
                #expect(right.metaState == left.metaState)
                #expect(right.allocatedBytes > 0)
                #expect(throws: MiMoV26ExecutionError.self) {
                    try right.restore(keys: state[0], values: state[1], offset: left.offset, index: index)
                }
            }
            compare(try target.forwardIndependentHistory(inputIDs: tokens([21]), caches: source),
                try target.forwardIndependentHistory(inputIDs: tokens([21]), caches: destination))
            // Updating the recipient must not mutate the original page handles.
            let sourceBeforeRecipientAdvance = source.map { $0.state.map(bytes) }
            _ = try target.forwardIndependentHistory(inputIDs: tokens([22]), caches: destination)
            #expect(source.map { $0.state.map(bytes) } == sourceBeforeRecipientAdvance)
            #expect(sourceBeforeRecipientAdvance != before)
            #expect(source.allSatisfy { $0.offset == 20 } && destination.allSatisfy { $0.offset == 21 })
            let fresh = try target.makeIndependentHistoryCaches()[0], state = source[0].state
            for (offset, index) in [(18, 19), (20, 19), (513, 513), (20, -1)] {
                #expect(throws: MiMoV26ExecutionError.self) {
                    try fresh.restore(keys: state[0], values: state[1], offset: offset, index: index)
                }
                #expect(fresh.offset == 0 && fresh.pageCount == 0)
            }
            #expect(throws: MiMoV26ExecutionError.self) {
                try fresh.restore(keys: state[0].asType(.float32), values: state[1], offset: 20, index: 20)
            }
        }
    }

    @Test func assistantPageCompactionAndDetachedCopiesKeepMeasuredRootsIndependent() throws {
        try Device.withDefaultDevice(.cpu) {
            let page = MiMoV26IndependentHistoryCache(modelIdentity: UUID(), requestIdentity: UUID(),
                layerIndex: 0, maximumContext: 512, heads: 2, keyWidth: 16, valueWidth: 8,
                dtype: .bfloat16, window: 65)
            let original = RotatingKVCache(maxSize: 65)
            for n in [67, 1, 1, 3, 64] {
                let k = values([1, 2, n, 16], salt: page.offset), v = values([1, 2, n, 8], salt: page.offset + 17)
                let a = original.update(keys: k, values: v), b = page.update(keys: k, values: v)
                equal(a.0, b.0); equal(a.1, b.1)
                page.compactRetainedHistory()
                let arrays = original.state
                if arrays[0].dim(2) > 65 {
                    original.state = arrays.map { $0[.ellipsis, (-65)..., 0...] }
                    var meta = original.metaState; meta[4] = "65"; original.metaState = meta
                }
                eval(page.innerState()); Stream.cpu.synchronize()
                equal(original.state[0], page.state[0]); equal(original.state[1], page.state[1])
                #expect(original.metaState == page.metaState)
                let copy = page.copy() as! MiMoV26IndependentHistoryCache
                copy.detachPages(); eval(copy.innerState()); Stream.cpu.synchronize()
                for array in copy.innerState() {
                    let info = try #require(try array.evaluatedBufferInfo())
                    #expect(info.isUnique && info.isRowContiguous && info.dataOffset == 0)
                    #expect(info.dataElements == array.size)
                }
            }
        }
    }

    private func model(tied: Bool, layers: Range<Int> = 0..<2) throws -> MiMoV26TextModel {
        let fields: [String: Any] = [
            "model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"],
            "hidden_size": 64, "intermediate_size": 128, "moe_intermediate_size": 64,
            "vocab_size": 64, "num_hidden_layers": layers.count, "max_position_embeddings": 512,
            "sliding_window_size": 8, "sliding_window": 8, "num_nextn_predict_layers": 3,
            "hybrid_layer_pattern": layers.map { $0 % 2 }, "moe_layer_freq": layers.map { $0 % 2 },
            "partial_rotary_factor": 0.5, "attention_value_scale": 0.707,
            "layernorm_epsilon": 0.000001, "attention_projection_layout": "split",
            "moe_router_dtype": "bfloat16", "hidden_act": "silu", "dtype": "bfloat16",
            "attention_bias": false, "tie_word_embeddings": tied, "attention_dropout": 0,
            "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_routed_experts": 4,
            "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
            "n_shared_experts": NSNull(), "routed_scaling_factor": NSNull(),
            "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 32, "v_head_dim": 16,
            "swa_num_attention_heads": 8, "swa_num_key_value_heads": 2, "swa_head_dim": 16,
            "swa_v_head_dim": 8, "rope_theta": 10000000, "swa_rope_theta": 10000,
            "add_full_attention_sink_bias": true, "add_swa_attention_sink_bias": true,
            "eos_token_id": 3, "pad_token_id": 0,
        ]
        let target = try MiMoV26TextModel(JSONDecoder().decode(MiMoV26Configuration.self,
            from: JSONSerialization.data(withJSONObject: fields)))
        let parameters = target.parameters().flattened().map { name, array in
            var globalName = name
            if name.hasPrefix("model.layers.") {
                var parts = name.split(separator: ".").map(String.init)
                parts[2] = String(Int(parts[2])! + layers.lowerBound); globalName = parts.joined(separator: ".")
            }
            let salt = globalName.utf8.reduce(0) { $0 + Int($1) }
            let data = (0..<array.size).map { Float(($0 * 13 + salt) % 101 - 50) / 256 + (name.contains("norm.weight") ? 1 : 0) }
            return (name, MLXArray(data, array.shape).asType(name.hasSuffix("e_score_correction_bias") ? .float32 : .bfloat16))
        }
        try target.update(parameters: .unflattened(parameters), verify: .all)
        // MiMo's layer array is intentionally immutable. Install into each
        // actual owning module, as the verified stage loader does, rather
        // than replacing the layer array through generic quantize(model:).
        var parents = Dictionary(uniqueKeysWithValues: target.namedModules())
        parents[""] = target
        for (path, leaf) in target.leafModules().flattened() {
            let expert = path.contains("switch_mlp")
            guard let packed = quantizeSingle(layer: leaf, groupSize: expert ? 32 : 64,
                bits: expert ? 4 : 8, mode: expert ? .mxfp4 : .affine) else { continue }
            var pieces = path.split(separator: ".").map(String.init); pieces.removeLast()
            while !pieces.isEmpty && parents[pieces.joined(separator: ".")] == nil { pieces.removeLast() }
            let parent = pieces.joined(separator: ".")
            let suffix = parent.isEmpty ? path : String(path.dropFirst(parent.count + 1))
            try parents[parent]!.update(modules: .unflattened([(suffix, packed)]), verify: [.noUnusedKeys])
        }
        if layers.lowerBound > 0 {
            try target.model.update(modules: .unflattened([("embed_tokens", MiMoIndependentResidualEmbedding())]),
                verify: [.noUnusedKeys])
        }
        target.freeze(); eval(target)
        return target
    }
    private func tokens(_ values: [Int]) -> MLXArray { MLXArray(values.map(Int32.init)).reshaped(1, values.count) }
    private func values(_ shape: [Int], salt: Int) -> MLXArray {
        MLXArray((0..<shape.reduce(1, *)).map { Float(($0 + salt) % 113 - 56) / 64 }, shape).asType(.bfloat16)
    }
    private func bytes(_ value: MLXArray) -> Data { eval(value); return value.asData(access: .copy).data }
    private func equal(_ a: MLXArray, _ b: MLXArray) {
        #expect(a.shape == b.shape); #expect(a.dtype == b.dtype); #expect(bytes(a) == bytes(b))
    }
    private func compare(_ a: MiMoV26TextOutput, _ b: MiMoV26TextOutput) {
        #expect(a.firstPosition == b.firstPosition); equal(a.logits, b.logits)
        equal(a.normalizedHiddenStates, b.normalizedHiddenStates)
        #expect(Set(a.layerFeatures.keys) == Set(b.layerFeatures.keys))
        for key in a.layerFeatures.keys { equal(a.layerFeatures[key]!, b.layerFeatures[key]!) }
    }
    private func compareCache(_ a: [any KVCache], _ b: [MiMoV26IndependentHistoryCache]) {
        #expect(a.count == b.count)
        for (left, right) in zip(a, b) {
            #expect(left.offset == right.offset)
            for (x, y) in zip(left.state, right.state) { equal(x, y) }
        }
    }
}

private final class MiMoIndependentResidualEmbedding: Embedding {
    init() { super.init(weight: MLXArray.zeros([1, 64], dtype: .bfloat16)) }
    override func callAsFunction(_ x: MLXArray) -> MLXArray { preconditionFailure("consumer requires residual") }
    override func asLinear(_ x: MLXArray) -> MLXArray { preconditionFailure("inactive embedding is not a head") }
}
