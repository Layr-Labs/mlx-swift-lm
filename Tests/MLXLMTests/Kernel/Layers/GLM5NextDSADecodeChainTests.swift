// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLLM

extension KernelTests {
    /// One serving DSA layer's decode at the TP-rank geometry (hidden 4096,
    /// 32 MLA heads, q_lora 1536, kv_lora 512, 32 index heads x 128, top 2048,
    /// affine 4-bit g64) on random weights and random history: the decode
    /// launch layout and the two-launch cache append against the original
    /// kernels and slice updates, bitwise, plus a one-layer timing
    /// (`GLM53_DSA_BENCH=1`). Not real-checkpoint evidence.
    @Suite(.serialized)
    struct GLM5NextDSADecodeChainTests {
        struct Fixture {
            let text: GLM5NextTextConfiguration
            let inventory: GLM5NextWeightInventory

            init() throws {
                let base = try GLM5NextInventoryFixture.configuration(layers: 4, q4: true)
                var root = try #require(
                    JSONSerialization.jsonObject(with: JSONEncoder().encode(base))
                        as? [String: Any])
                var text = try #require(root["text_config"] as? [String: Any])
                text.merge([
                    "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 32,
                    "qk_head_dim": 256, "qk_nope_head_dim": 256, "v_head_dim": 256,
                    "q_lora_rank": 1536, "kv_lora_rank": 512, "index_n_heads": 32,
                    "index_head_dim": 128, "index_topk": 2048, "max_position_embeddings": 131072,
                ]) { _, new in new }
                root["text_config"] = text
                var vision = try #require(root["vision_config"] as? [String: Any])
                vision["out_hidden_size"] = 4096
                root["vision_config"] = vision
                let configuration = try JSONDecoder().decode(
                    GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
                self.text = configuration.textConfig
                inventory = try GLM5NextWeightInventory.validate(
                    configuration: configuration,
                    tensors: GLM5NextWeightInventory.expectedTensors(
                        configuration: configuration, storage: .vontraQ4,
                        shard: "dsa.safetensors"),
                    storage: .vontraQ4)
            }

            func tensor(_ name: String) throws -> MLXArray {
                let header = try #require(inventory.entries[name]).tensor
                let salt = UInt64(name.utf8.reduce(0) { ($0 &* 31 &+ Int($1)) % 1_000_003 })
                let key = MLXRandom.key(salt)
                if header.dtype == "U32" {
                    // Uniform 32-bit words (int32 wraps to uint32), so 4-bit codes are uniform.
                    return MLXRandom.randInt(Int32.min ..< Int32.max, header.shape, key: key)
                        .asType(.uint32)
                }
                let unit = MLXRandom.uniform(-1 ..< 1, header.shape, key: key)
                let values: MLXArray
                // Zero-mean 4-bit weights: bias = -7.5 x the group scale.
                let scale = 0.004 + 0.002 * MLXRandom.uniform(
                    -1 ..< 1, header.shape,
                    key: MLXRandom.key(
                        UInt64(
                            name.replacingOccurrences(of: ".scales", with: "")
                                .replacingOccurrences(of: ".biases", with: "").utf8
                                .reduce(0) { ($0 &* 31 &+ Int($1)) % 1_000_003 }) &+ 17))
                if name.hasSuffix(".scales") {
                    values = scale
                } else if name.hasSuffix(".biases") {
                    values = -7.5 * scale
                } else if name.contains("norm.weight") {
                    values = 1 + 0.25 * unit
                } else if name.hasSuffix(".bias") {
                    values = 0.05 * unit
                } else {
                    values = 0.5 * unit
                }
                return header.dtype == "BF16" ? values.asType(.bfloat16) : values
            }

            func attention(policy: GLM5NextExecutionPolicy = .serving) throws
                -> GLM5NextDSAAttention
            {
                try GLM5NextDSAAttention(
                    configuration: text, sourcePrefix: "model.language_model.layers.3.self_attn",
                    inventory: inventory, tensor: tensor, releaseSourceTensors: { _ in },
                    executionPolicy: policy)
            }

            /// History of `count` tokens appended in 4096-token chunks, as a
            /// chunked prefill grows it (geometric capacity).
            func history(_ count: Int, seed: UInt64 = 7, keyOffset: Float = 0.5) throws
                -> GLM5NextDSACache
            {
                let cache = GLM5NextDSACache()
                var start = 0
                while start < count {
                    let n = min(4096, count - start)
                    let key = MLXRandom.key(seed &+ UInt64(start))
                    let k = MLXRandom.split(key: key, into: 3)
                    try cache.append(
                        latents: MLXRandom.normal([1, n, text.kvLoraRank], key: k[0]).asType(
                            .bfloat16),
                        // Offset keys: the synthetic index queries share a common
                        // direction, and zero-mean keys would give many exact-zero
                        // (ReLU) pool scores, whose boundary ties send the
                        // selector to its one-thread fallback.
                        keys: (keyOffset + MLXRandom.normal([1, n, text.indexHeadDim], key: k[1]))
                            .asType(.bfloat16),
                        gates: (2 * MLXRandom.normal([1, n, text.indexHeadDim], key: k[2]))
                            .asType(.bfloat16),
                        validity: MLXArray.ones([1, n], dtype: .bool))
                    eval(cache.innerState())
                    start += n
                }
                return cache
            }

            func hidden(rows: Int, step: Int) -> MLXArray {
                MLXRandom.normal([1, rows, text.hiddenSize], key: MLXRandom.key(UInt64(1000 + step)))
                    .asType(.bfloat16)
            }
        }

        /// GPU-busy seconds since the last reset, when the MLX core is built
        /// with the local timing hook `mlx_debug_gpu_busy_seconds` (union of
        /// command-buffer GPU intervals; see `agent-work/dsadecode/NOTES.md`).
        /// Without it the benchmark reports wall time only.
        static let gpuBusy: (@convention(c) (Int32) -> Double)? = {
            guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "mlx_debug_gpu_busy_seconds")
            else { return nil }
            return unsafeBitCast(symbol, to: (@convention(c) (Int32) -> Double).self)
        }()

        /// Runs `steps` decode steps as the TP stage does (a working copy whose
        /// destination keeps the old storage until after the fence), chained
        /// through the output so consecutive steps do not overlap. Returns
        /// GPU-busy and wall microseconds per step.
        static func timeChain(
            _ attention: GLM5NextDSAAttention, cache: GLM5NextDSACache, fixture: Fixture,
            rows: Int, steps: Int
        ) throws -> (gpu: Double, wall: Double) {
            var held: [[MLXArray]] = []
            var hidden = fixture.hidden(rows: rows, step: 0)
            let base = hidden
            let validity = MLXArray.ones([1, rows], dtype: .bool)
            eval(hidden, validity)
            var outputs: [MLXArray] = []
            for _ in 0 ..< steps {
                held.append(cache.innerState())
                let working = cache.copy() as! GLM5NextDSACache
                let out = try attention.forward(hidden, cache: working, validity: validity)
                cache.adopt(working)
                outputs.append(out)
                hidden = base + (out * 0).asType(.bfloat16)
            }
            Stream.gpu.synchronize()
            _ = gpuBusy?(1)
            let t0 = DispatchTime.now().uptimeNanoseconds
            eval(outputs + cache.innerState())
            Stream.gpu.synchronize()
            let wall = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e3
            let busy = (gpuBusy?(1) ?? 0) * 1e6
            withExtendedLifetime(held) {}
            return (busy / Double(steps), wall / Double(steps))
        }

        static func layout(_ text: Substring) -> GLM5NextSparseDecodeKernel.Layout? {
            let fields = text.split(separator: "/")
            let parts = fields.prefix(3).compactMap { Int($0) }
            guard parts.count == 3 else { return nil }
            return .init(
                poolTile: parts[0], headGroup: parts[1], selectionThreads: parts[2],
                prefetchedMerge: fields.count > 3 && fields[3].contains("p"),
                pairedLatentLoads: fields.count > 3 && fields[3].contains("v"))
        }

        /// `GLM53_DSA_BENCH=1`: one layer's decode step (GPU busy per step,
        /// chained steps, TP-stage cache handling) for each launch layout.
        @Test func decodeLayerTiming() throws {
            let environment = ProcessInfo.processInfo.environment
            guard environment["GLM53_DSA_BENCH"] == "1" else { return }
            let fixture = try Fixture()
            let attention = try fixture.attention()
            let histories = (environment["GLM53_DSA_HISTORY"] ?? "2048,4096,12288,32768")
                .split(separator: ",").compactMap { Int($0) }
            let layouts = (environment["GLM53_DSA_LAYOUTS"] ?? "32/32/256,8/8/1024/pv")
                .split(separator: ",").compactMap(Self.layout)
            for history in histories {
                for rows in [1, 3] {
                    var line = "GLM53_DSA_BENCH history=\(history) rows=\(rows)"
                    for layout in layouts {
                        attention.sparseDecodeLayout = layout
                        let cache = try fixture.history(history)
                        cache.usesAppendKernel = layout != .original
                        // Warm up; also grows capacity once, as the first decode does.
                        _ = try Self.timeChain(
                            attention, cache: cache, fixture: fixture, rows: rows, steps: 4)
                        var gpu: [Double] = []
                        var wall: [Double] = []
                        for _ in 0 ..< 5 {
                            let t = try Self.timeChain(
                                attention, cache: cache, fixture: fixture, rows: rows, steps: 16)
                            gpu.append(t.gpu)
                            wall.append(t.wall)
                        }
                        gpu.sort()
                        wall.sort()
                        line +=
                            " [\(layout.poolTile)/\(layout.headGroup)/\(layout.selectionThreads)\(layout.prefetchedMerge ? "/p" : "")\(layout.pairedLatentLoads ? "v" : "")] gpu_us=\(Self.gpuBusy == nil ? "n/a" : String(format: "%.1f", gpu[2])) wall_us=\(String(format: "%.1f", wall[2]))"
                    }
                    print(line)
                }
            }
            attention.sparseDecodeLayout = .serving
        }

        private static func bits(_ array: MLXArray) -> MLXArray {
            switch array.dtype {
            case .float32: array.view(dtype: .uint32)
            case .bfloat16, .float16: array.view(dtype: .uint16)
            default: array
            }
        }

        private static func differing(_ a: MLXArray, _ b: MLXArray) -> Int {
            guard a.shape == b.shape, a.dtype == b.dtype else { return -1 }
            return (bits(a) .!= bits(b)).asType(.int32).sum().item(Int.self)
        }


        /// The two-launch append against the slice updates and against plain
        /// concatenation, bitwise, from TP-like capacities at every history.
        @Test func appendKernelIsBitwiseToSliceUpdatesOnGPU() throws {
            let fixture = try Fixture()
            for history in [2048, 4096, 12288, 32768] {
                let kernel = try fixture.history(history)
                let slices = kernel.copy() as! GLM5NextDSACache
                slices.usesAppendKernel = false
                var reference = kernel.innerState()
                var diffs: [Int] = []
                for (step, rows) in [1, 3, 1, 1, 3, 2].enumerated() {
                    let key = MLXRandom.split(key: MLXRandom.key(UInt64(history + step)), into: 4)
                    let latents = MLXRandom.normal([1, rows, 512], key: key[0]).asType(.bfloat16)
                    let keys = MLXRandom.normal([1, rows, 128], key: key[1]).asType(.bfloat16)
                    let gates = MLXRandom.normal([1, rows, 128], key: key[2]).asType(.bfloat16)
                    let validity = MLXRandom.bernoulli(0.8, [1, rows], key: key[3])
                    for cache in [kernel, slices] {
                        try cache.append(latents: latents, keys: keys, gates: gates, validity: validity)
                    }
                    reference = zip(reference, [latents, keys, gates, validity]).map {
                        concatenated([$0, $1], axis: 1)
                    }
                    if step == 3 {
                        // A rewind: the next appends overwrite stale spare columns.
                        for cache in [kernel, slices] { try cache.truncate(to: history + 2) }
                        reference = reference.map { $0[0..., ..<(history + 2)] }
                    }
                    eval(kernel.innerState() + slices.innerState() + reference)
                    for ((a, b), r) in zip(zip(kernel.innerState(), slices.innerState()), reference) {
                        diffs.append(Self.differing(a, b))
                        diffs.append(Self.differing(a, r))
                    }
                }
                print("GLM53_DSA_APPEND_BITWISE history=\(history) capacity=\(kernel.capacity) differing=\(diffs.reduce(0, +)) checks=\(diffs.count)")
                #expect(diffs.allSatisfy { $0 == 0 }, "history \(history): \(diffs)")
            }
        }

        /// The serving launch layout against the original one, bitwise: each
        /// decode kernel on the layer's own inputs, and the layer's output and
        /// history over several chained decode steps.
        @Test func servingLayoutIsBitwiseToTheOriginalOnGPU() throws {
            let fixture = try Fixture()
            let attention = try fixture.attention()
            let c = fixture.text
            let cases: [(history: Int, keyOffset: Float)] = [
                (2048, 0.5), (4096, 0.5), (12288, 0.5), (32768, 0.5),
                // Zero-mean keys: many exact-zero pool scores, so ties straddle
                // the selection boundary and the selector's fallback runs.
                (12288, 0),
            ]
            for (history, keyOffset) in cases {
                for rows in [1, 3] {
                    let context = "history=\(history) rows=\(rows) keyOffset=\(keyOffset)"
                    // Kernel level, on the layer's index queries and history.
                    let cache = try fixture.history(history, keyOffset: keyOffset)
                    let hidden = fixture.hidden(rows: rows, step: 3)
                    let parts = attention.fusedInputs!(hidden)
                    let residual = attention.qNorm(parts[0])
                    let index = attention.indexer.project(
                        hidden, residual: residual, rawKeys: parts[2], weights: parts[3])
                    let validity = MLXArray.ones([1, rows], dtype: .bool)
                    var selections: [GLM5NextSparseAttention.Selection] = []
                    var scores: [MLXArray] = []
                    for layout in [GLM5NextSparseDecodeKernel.Layout.original, .serving] {
                        let pooled = try GLM5NextSparseDecodeKernel.poolScores(
                            indexQueries: index.queries, headWeights: index.weights,
                            indexKeys: cache.indexKeys!, poolGates: cache.poolGates!,
                            poolPositionBias: attention.indexer.positionBias,
                            keyValidity: cache.validity!, queryStart: history - rows,
                            poolingType: .bfloat16, layout: layout)
                        scores.append(pooled.scores)
                        selections.append(
                            try GLM5NextSparseAttention.selectDecodeTokens(
                                indexQueries: index.queries, indexKeys: cache.indexKeys!,
                                poolGates: cache.poolGates!,
                                poolPositionBias: attention.indexer.positionBias,
                                headWeights: index.weights, keyValidity: cache.validity!,
                                queryValidity: validity, queryStart: history - rows,
                                tokenBudget: c.indexTopK, strategy: .parallelThreshold,
                                poolingType: nil, layout: layout))
                    }
                    let queryLatents = MLXRandom.normal(
                        [1, c.attentionHeads, rows, c.kvLoraRank], key: MLXRandom.key(UInt64(history))
                    ).asType(.bfloat16)
                    let attended = try [GLM5NextSparseDecodeKernel.Layout.original, .serving].map {
                        try GLM5NextSparseDecodeKernel.attend(
                            queryLatents: queryLatents, latents: cache.latents!,
                            indices: selections[0].indices, scale: 1 / 16,
                            outputType: .bfloat16, layout: $0)
                    }
                    let kernelDiffs = [
                        Self.differing(scores[0], scores[1]),
                        Self.differing(selections[0].indices, selections[1].indices),
                        Self.differing(attended[0], attended[1]),
                    ]
                    print("GLM53_DSA_LAYOUT_BITWISE kernels \(context) differing=\(kernelDiffs)")
                    #expect(kernelDiffs == [0, 0, 0], "\(context)")

                    // Layer level: chained decode steps on two copies of one history.
                    let oracle = cache.copy() as! GLM5NextDSACache
                    oracle.usesAppendKernel = false
                    let caches = [oracle, cache]
                    var layerDiffs: [Int] = []
                    for step in 0 ..< 3 {
                        let x = fixture.hidden(rows: rows, step: 10 + step)
                        var outputs: [MLXArray] = []
                        for (layout, history) in zip(
                            [GLM5NextSparseDecodeKernel.Layout.original, .serving], caches)
                        {
                            attention.sparseDecodeLayout = layout
                            outputs.append(
                                try attention.forward(x, cache: history, validity: validity))
                            eval(outputs.last!)
                        }
                        layerDiffs.append(Self.differing(outputs[0], outputs[1]))
                        for (a, b) in zip(caches[0].innerState(), caches[1].innerState()) {
                            layerDiffs.append(Self.differing(a, b))
                        }
                    }
                    attention.sparseDecodeLayout = .serving
                    print("GLM53_DSA_LAYOUT_BITWISE layer \(context) differing=\(layerDiffs)")
                    #expect(layerDiffs.allSatisfy { $0 == 0 }, "\(context)")
                }
            }
        }
    }
}
