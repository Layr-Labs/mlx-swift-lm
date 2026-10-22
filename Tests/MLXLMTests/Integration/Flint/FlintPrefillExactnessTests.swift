// Exact-bytes gates for Flint paged prefill speedups. Every fast path is
// compared with the unmodified expression on identical inputs: native BF16
// activations and KV, affine INT4 g64 weights, Flint's real layer geometry and
// 2048-token chunks through the production PagedKVBackend. Flint geometry
// follows ThirdPartyNotices/Flint.md: Prince Canuma/mlx-vlm (MIT) and Recursive
// rec (Apache-2.0). Random weights only; no checkpoint or downloaded data.
import Cmlx
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Flint prefill exactness", .serialized)
struct FlintPrefillExactnessTests {
    /// Real Flint attention/MoE geometry on the fixture's three-layer hybrid
    /// stack: dense full NoPE layer, RoPE sliding (513) MoE layer, full NoPE
    /// MoE layer. `slidingLast` instead ends on the RoPE sliding layer
    /// (full, full NoPE, sliding).
    private func configuration(experts: Int, slidingLast: Bool = false, fullDepth: Bool = false)
        throws -> FlintConfiguration
    {
        let url = try #require(
            Bundle.module.url(forResource: "flint-oracle", withExtension: "json"))
        let fixture = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var config = try #require(fixture["config"] as? [String: Any])
        if slidingLast {
            config["layer_types"] = ["full_attention", "full_attention", "sliding_attention"]
            config["no_rope_layer_indices"] = [0, 1]
        }
        if fullDepth {
            // The released layout: 49 layers, dense 0-1, full NoPE attention
            // at these 14 layers and RoPE sliding attention everywhere else.
            let full = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
            config["num_hidden_layers"] = 49
            config["layer_types"] = (0 ..< 49).map {
                full.contains($0) ? "full_attention" : "sliding_attention"
            }
            config["no_rope_layer_indices"] = full
            config["mlp_only_layers"] = [0, 1]
        }
        let geometry: [String: Int] = [
            "hidden_size": 2048, "intermediate_size": 6144, "num_attention_heads": 32,
            "num_key_value_heads": 4, "head_dim": 128, "num_experts": experts,
            "num_experts_per_tok": 6, "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 1024, "vocab_size": 256,
        ]
        for (key, value) in geometry { config[key] = value }
        return try JSONDecoder().decode(
            FlintConfiguration.self, from: JSONSerialization.data(withJSONObject: config))
    }

    /// Deterministic random BF16 parameters (FP32 learned sinks, as shipped),
    /// then affine INT4 g64 everywhere except the router and embeddings.
    private func randomize(_ module: Module, seed: UInt64) throws {
        var weights = [String: MLXArray]()
        let entries = module.parameters().flattened().sorted { $0.0 < $1.0 }
        for (index, (name, value)) in entries.enumerated() {
            let noise = MLXRandom.normal(value.shape, key: MLXRandom.key(seed + UInt64(index)))
            if name.hasSuffix("softmax_offset") {
                weights[name] = noise
            } else if name.contains("norm") {
                weights[name] = (1 + noise * 0.05).asType(.bfloat16)
            } else if name.hasSuffix("expert_bias") {
                weights[name] = (noise * 0.01).asType(.bfloat16)
            } else {
                weights[name] = (noise / sqrt(Float(value.shape.last ?? 1))).asType(.bfloat16)
            }
            eval(weights[name]!)
        }
        try module.update(parameters: .unflattened(weights), verify: .all)
        // The router stays BF16, also when `module` is a bare FlintMoE.
        quantize(model: module, groupSize: 64, bits: 4) { path, _ in
            !path.hasSuffix("mlp.gate") && path != "gate" && !path.contains("embed_tokens")
        }
        eval(module)
    }

    private func expectExact(
        _ actual: MLXArray, _ expected: MLXArray, _ context: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        eval(actual, expected)
        let note = Comment(rawValue: context)
        #expect(actual.shape == expected.shape, note, sourceLocation: sourceLocation)
        #expect(actual.dtype == expected.dtype, note, sourceLocation: sourceLocation)
        // Native bytes: detects BF16 rounding changes and signed zeros.
        let same = actual.asData(access: .copy).data == expected.asData(access: .copy).data
        #expect(same, note, sourceLocation: sourceLocation)
    }

    // MARK: - MoE combine

    /// The kernel against the original MLX expression on adversarial values:
    /// wide exponent spread, exact cancellations, signed zeros and BF16
    /// rounding ties, at Flint's [2048 tokens, 6 slots, 2048 hidden] shape.
    @Test(arguments: [6, 8])
    func fusedCombineKernelMatchesMLXExpressionBytes(topK: Int) {
        let tokens = 2048
        let hidden = 2048
        let n = tokens * topK
        let key = MLXRandom.key(UInt64(4_100 + topK))
        let keys = MLXRandom.split(key: key, into: 6)
        let order = argSort(MLXRandom.uniform(low: 0, high: 1, [n], key: keys[0]))
        let inverse = argSort(order)
        let scale = MLX.pow(
            MLXArray(Float(2)),
            MLXRandom.randInt(low: -16, high: 16, [n, hidden], key: keys[1]).asType(.float32))
        var sorted = MLXRandom.normal([n, hidden], key: keys[2]) * scale
        let row = MLXArray(Int32(0) ..< Int32(n)).reshaped(n, 1)
        let column = MLXArray(Int32(0) ..< Int32(hidden)).reshaped(1, hidden)
        // Signed-zero rows, +-256 cancellation rows and BF16 tie values.
        sorted = MLX.where((row % 7) .== 0, MLXArray(Float(-0.0)), sorted)
        sorted = MLX.where(
            (row % 11) .== 3, MLX.where((column % 2) .== 0, MLXArray(Float(256)), -256), sorted)
        sorted = MLX.where((row % 13) .== 5, MLXArray(Float(1 + 1.0 / 256)), sorted)
        let sortedBF16 = sorted.asType(.bfloat16)
        var weights = MLXRandom.uniform(low: -1, high: 1, [1, tokens, topK], key: keys[3])
        let slot = MLXArray(Int32(0) ..< Int32(tokens * topK)).reshaped(1, tokens, topK)
        weights = MLX.where((slot % 17) .== 0, MLXArray(Float(-0.0)), weights)
        weights = MLX.where((slot % 19) .== 0, MLXArray(Float(0.5)), weights)
        var addend = MLXRandom.normal([1, tokens, hidden], key: keys[4])
        let position = MLXArray(Int32(0) ..< Int32(tokens * hidden)).reshaped(1, tokens, hidden)
        addend = MLX.where((position % 5) .== 0, MLXArray(Float(-0.0)), addend)
        addend = MLX.where((position % 9) .== 0, MLXArray(Float(0)), addend)
        let addendBF16 = addend.asType(.bfloat16)
        eval(sortedBF16, inverse, weights, addendBF16)

        #expect(
            flintFP32WeightedUnsortAddSupported(
                sortedOutputs: sortedBF16, inverseOrder: inverse, weights: weights,
                addend: addendBF16))
        let actual = flintFP32WeightedUnsortAdd(
            sortedOutputs: sortedBF16, inverseOrder: inverse, weights: weights,
            addend: addendBF16)
        let unsorted = sortedBF16[inverse].reshaped(1, tokens, topK, hidden)
        let expected =
            (unsorted.asType(.float32) * weights[.ellipsis, .newAxis]).sum(axis: -2)
            .asType(.bfloat16) + addendBF16
        expectExact(actual, expected, "kernel K=\(topK)")
    }

    /// Real FlintMoE (256 experts, top-6, INT4 g64) on a 2048-token chunk:
    /// the production call selects the fused route and matches the original
    /// unfused combine bit for bit; batch rows stay independent.
    @Test func flintMoEFusedCombineMatchesOriginalBytes() throws {
        let moe = FlintMoE(try configuration(experts: 256))
        try randomize(moe, seed: 8_800)
        let x = (MLXRandom.normal([2, 2048, 2048], key: MLXRandom.key(8_801)) * 0.5)
            .asType(.bfloat16)
        eval(x)
        for batch in [1, 2] {
            let input = x[0 ..< batch]
            let (indices, weights) = moe.route(input)
            let routed = moe.experts.flintFP32WeightedSumRoute(
                input, indices, weights: weights, adding: moe.shared(input))
            #expect(routed.fused, "the fused route must engage for a sorted BF16 chunk")
            moe.usesFusedExpertCombine = false
            let expected = moe(input)
            eval(expected)
            moe.usesFusedExpertCombine = true
            let actual = moe(input)
            expectExact(actual, expected, "FlintMoE batch \(batch)")
            expectExact(routed.output, expected, "routed batch \(batch)")
            if batch == 2 {
                let alone = moe(x[1 ..< 2])
                expectExact(actual[1 ..< 2], alone, "row 1 packed vs alone")
            }
        }
        // Decode and tiny chunks keep the original expression (no sort).
        let decode = x[0 ..< 1, 0 ..< 1]
        let (indices, weights) = moe.route(decode)
        #expect(
            !moe.experts.flintFP32WeightedSumRoute(
                decode, indices, weights: weights, adding: moe.shared(decode)
            ).fused)
    }

    // MARK: - Full model through the paged backend

    /// Toggles for the unmodified (`false`) and optimized (`true`) paths.
    struct Paths {
        var fusedExpertCombine = true
        var batchedQueryBlocks = true
        var finalPromptKVOnly = true
    }

    private func apply(_ paths: Paths, to model: FlintModel, caches: [PagedLayerCache]) {
        for layer in model.model.layers {
            (layer.mlp as? FlintMoE)?.usesFusedExpertCombine = paths.fusedExpertCombine
        }
        model.commitsFinalPromptKVOnly = paths.finalPromptKVOnly
        for cache in caches {
            cache.batchesUniformQueryBlocks = paths.batchedQueryBlocks
            // As CBv2LayerCacheBank configures every layer no sibling borrows.
            cache.setRetainsChunkForBorrowers(false)
        }
    }

    struct Run {
        var outputs: [MLXArray] = []
        var kv: [MLXArray] = []
        var batchedQueryBlockCalls = 0
        var kvOnlyPromptWrites = 0
        /// Primitive histogram of the first decode step's lazy graph.
        var decodeGraph: [String: Int] = [:]
    }

    /// Prefill `streams` (one per packed row) chunk by chunk through a fresh
    /// PagedKVBackend, then decode one token. Records every chunk's full
    /// hidden states, the decode logits and every row's retained K/V. With
    /// `requirements`, chunks go through the CBv2 prompt seam instead
    /// (`.evaluationOnly`, then `.lastPositionLogits` for the frontier chunk)
    /// and only the frontier logits are recorded: intermediate handles have
    /// unspecified values by contract.
    private func pagedRun(
        _ model: FlintModel, streams: [[Int32]], chunks: [Int], paths: Paths,
        requirements: Bool = false, decodeSteps: Int = 1
    ) throws -> Run {
        let kinds = model.cbv2LayerKinds
        let total = chunks.reduce(0, +)
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 512 << 20, dtype: .bfloat16, maxPrefillChunk: 2048,
                nominalMaxSequenceLength: 8192, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: .bfloat16, count: kinds.count)))
        let caches = backend.makeLayerCaches()
        var states: [[CBv2SequenceKV?]] = []
        defer {
            for cache in caches { cache.setRows([]) }
            for state in states { backend.release(state) }
        }
        for _ in streams {
            states.append(
                try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: total, maxLength: total + decodeSteps + 8))
        }
        for (layer, cache) in caches.enumerated() {
            cache.setRows(states.map { $0[layer]! })
        }
        apply(paths, to: model, caches: caches)
        var run = Run()
        var start = 0
        for count in chunks {
            let tokens = MLXArray(streams.flatMap { $0[start ..< (start + count)] })
                .reshaped(streams.count, count)
            let frontier = start + count == total
            let output =
                requirements
                ? model.cbv2Prefill(
                    tokens, inputEmbedding: nil, cache: caches,
                    requirement: frontier ? .lastPositionLogits : .evaluationOnly)
                : model.model(tokens, cache: caches)
            eval([output] + caches.flatMap { $0.innerState() })
            #expect(!backend.pool.writeValidation.isFaulted)
            if !requirements || frontier { run.outputs.append(output) }
            start += count
        }
        for step in 0 ..< decodeSteps {
            let next = MLXArray(streams.map { $0[total + step] }).reshaped(streams.count, 1)
            let logits = model(next, cache: caches)
            if step == 0 {
                run.decodeGraph = Self.graphPrimitives(
                    [logits] + caches.flatMap { $0.innerState() })
            }
            eval([logits] + caches.flatMap { $0.innerState() })
            run.outputs.append(logits)
        }
        for state in states {
            for entry in state {
                let row = try #require(entry as? PagedSequenceKV)
                let snapshot = row.snapshot()
                run.kv += [snapshot.keys, snapshot.values]
            }
        }
        eval(run.kv)
        run.batchedQueryBlockCalls = caches.reduce(0) { $0 + $1.batchedQueryBlockCallsForTesting }
        run.kvOnlyPromptWrites = caches.reduce(0) { $0 + $1.kvOnlyPromptWritesForTesting }
        return run
    }

    private func streams(rows: Int, length: Int) -> [[Int32]] {
        (0 ..< rows).map { (row: Int) -> [Int32] in
            let stride = 7 + 2 * row
            let offset = 3 * row + 1
            return (0 ..< length).map { (index: Int) -> Int32 in
                Int32((index * stride + offset) % 251)
            }
        }
    }

    /// Cold 2048 chunk, two warm 2048 chunks (sliding history wraps the
    /// window and the page ring), an odd 641-token tail, then decode.
    private let chunks = [2048, 2048, 2048, 641]

    private func expectSameRun(_ actual: Run, _ expected: Run, _ context: String) {
        #expect(actual.outputs.count == expected.outputs.count)
        for (index, (a, e)) in zip(actual.outputs, expected.outputs).enumerated() {
            expectExact(a, e, "\(context) output \(index)")
        }
        #expect(actual.kv.count == expected.kv.count)
        for (index, (a, e)) in zip(actual.kv, expected.kv).enumerated() {
            expectExact(a, e, "\(context) kv \(index)")
        }
    }

    /// Runs the unmodified and optimized paths packed (two rows) and with row
    /// 1 alone; every output and K/V byte must match between paths. A packed
    /// row must equal the same row alone wherever the unmodified code already
    /// has that property; a path change may never alter it. Whether the
    /// unmodified code has it depends on the GPU's kernel tiling (it holds on
    /// M5 for 2048-token chunks but not on CI's paravirtual GPU), so it is
    /// compared, not assumed.
    @discardableResult
    private func expectPathsExact(_ reference: Paths, _ optimized: Paths, _ context: String)
        throws -> (reference: Run, optimized: Run)
    {
        let model = FlintModel(try configuration(experts: 64))
        try randomize(model, seed: 9_100)
        let input = streams(rows: 2, length: chunks.reduce(0, +) + 1)
        let expected = try pagedRun(model, streams: input, chunks: chunks, paths: reference)
        let actual = try pagedRun(model, streams: input, chunks: chunks, paths: optimized)
        expectSameRun(actual, expected, "\(context) packed")
        let expectedAlone = try pagedRun(
            model, streams: [input[1]], chunks: chunks, paths: reference)
        let actualAlone = try pagedRun(
            model, streams: [input[1]], chunks: chunks, paths: optimized)
        expectSameRun(actualAlone, expectedAlone, "\(context) alone")
        for index in actual.outputs.indices {
            let bytes = { (a: MLXArray) in a.asData(access: .copy).data }
            let referenceIndependent =
                bytes(expected.outputs[index][1 ..< 2]) == bytes(expectedAlone.outputs[index])
            let actualIndependent =
                bytes(actual.outputs[index][1 ..< 2]) == bytes(actualAlone.outputs[index])
            #expect(referenceIndependent == actualIndependent, "\(context) output \(index)")
        }
        return (expected, actual)
    }

    /// One sliding layer per chunk: the cold chunk batches its 12 unclamped
    /// blocks (4 ragged ones stay single), each warm 2048 chunk all 16, the
    /// 641 tail its 5 full blocks; packed rows batch per row.
    @Test func pagedPrefillBatchedSlidingBlocksMatchPerBlockBytes() throws {
        let (reference, optimized) = try expectPathsExact(
            Paths(batchedQueryBlocks: false), Paths(), "batched sliding blocks")
        #expect(reference.batchedQueryBlockCalls == 0)
        #expect(optimized.batchedQueryBlockCalls == 2 * chunks.count)
    }

    /// `.evaluationOnly` chunks commit the final layer's K/V without its
    /// attention/MLP. Every row's retained K/V (all layers), the frontier
    /// logits and the next decode logits stay byte-identical, packed and
    /// alone, whether the final layer is full NoPE or RoPE sliding.
    @Test(arguments: [false, true])
    func evaluationOnlyChunksCommitFinalKVWithoutChangingBytes(slidingLast: Bool) throws {
        let model = FlintModel(try configuration(experts: 64, slidingLast: slidingLast))
        try randomize(model, seed: 9_300)
        let input = streams(rows: 2, length: chunks.reduce(0, +) + 1)
        for rows in [input, [input[1]]] {
            let expected = try pagedRun(
                model, streams: rows, chunks: chunks, paths: Paths(finalPromptKVOnly: false),
                requirements: true)
            let actual = try pagedRun(
                model, streams: rows, chunks: chunks, paths: Paths(), requirements: true)
            expectSameRun(actual, expected, "final KV only, \(rows.count) rows")
            #expect(expected.kvOnlyPromptWrites == 0)
            // Every chunk but the frontier one.
            #expect(actual.kvOnlyPromptWrites == chunks.count - 1)
        }
    }

    @Test func pagedPrefillFusedExpertCombineMatchesOriginalBytes() throws {
        try expectPathsExact(
            Paths(fusedExpertCombine: false), Paths(), "fused expert combine")
    }

    // MARK: - Decode fusions

    /// Primitive applications in the lazy graph behind `outputs`, by MLX
    /// primitive name. MLX's DOT export emits one node line per application
    /// (a multi-output primitive once; a compiled function's replayed
    /// primitive once per call). View-only primitives that never dispatch a
    /// kernel are dropped; what remains approximates the step's GPU dispatches
    /// (a few primitives may add a copy at evaluation).
    static func graphPrimitives(_ outputs: [MLXArray]) -> [String: Int] {
        let views: Set<String> = [
            "Reshape", "Slice", "Transpose", "Broadcast", "BroadcastAxes", "ExpandDims",
            "Squeeze", "StopGradient", "Depends", "Flatten", "Unflatten", "AsStrided",
        ]
        let namer = mlx_node_namer_new()
        defer { mlx_node_namer_free(namer) }
        let handles = outputs.map(\.ctx)
        let vector = handles.withUnsafeBufferPointer {
            mlx_vector_array_new_data($0.baseAddress, $0.count)
        }
        defer { mlx_vector_array_free(vector) }
        let path = NSTemporaryDirectory() + "flint-decode-graph-\(UUID().uuidString).dot"
        defer { try? FileManager.default.removeItem(atPath: path) }
        guard let file = fopen(path, "w") else { return [:] }
        mlx_export_to_dot(file, namer, vector)
        fclose(file)
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [:] }
        var counts = [String: Int]()
        for line in text.split(separator: "\n") {
            // `{ <primitive id> [label ="<Name>", shape=rectangle]; }`
            guard let label = line.range(of: " [label =\""), line.hasPrefix("{ ") else { continue }
            let rest = line[label.upperBound...]
            guard let end = rest.firstIndex(of: "\"") else { continue }
            let name = String(rest[..<end])
            if !views.contains(name) { counts[name, default: 0] += 1 }
        }
        return counts
    }

    private static func describe(_ graph: [String: Int]) -> String {
        "\(graph.values.reduce(0, +)) ["
            + graph.sorted { $0.key < $1.key }
            .map { "\($0.key) \($0.value)" }.joined(separator: ", ") + "]"
    }

    /// Every decode-fusion switch (the reference turns all off on an unfused
    /// model, i.e. the original expressions on separately stored weights).
    static func setDecodeFusions(_ model: FlintModel, _ enabled: Bool) {
        for layer in model.model.layers {
            layer.attention.usesFusedProjection = enabled
            if let moe = layer.mlp as? FlintMoE {
                moe.usesFusedRouter = enabled
                moe.usesFusedExpertCombine = enabled
                moe.shared.usesFusedProjection = enabled
                moe.experts.suppliesHostRowIndices = enabled
            } else if let dense = layer.mlp as? FlintMLP {
                dense.usesFusedProjection = enabled
            }
        }
    }

    /// The decode combine kernel (unsorted, slot-order rows) against the
    /// original MLX expression, with and without the folded residual add, on
    /// adversarial values at Flint's decode shapes (batch 1 and 3) and a
    /// 2048-token chunk.
    @Test(arguments: [6, 8])
    func decodeCombineKernelMatchesMLXExpressionBytes(topK: Int) {
        let hidden = 2048
        for (batch, length) in [(1, 1), (3, 1), (1, 2048)] {
            let tokens = batch * length
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(5_200 + topK + tokens)), into: 6)
            let scale = MLX.pow(
                MLXArray(Float(2)),
                MLXRandom.randInt(low: -16, high: 16, [tokens * topK, hidden], key: keys[0])
                    .asType(.float32))
            var rows = MLXRandom.normal([tokens * topK, hidden], key: keys[1]) * scale
            let row = MLXArray(Int32(0) ..< Int32(tokens * topK)).reshaped(-1, 1)
            let column = MLXArray(Int32(0) ..< Int32(hidden)).reshaped(1, hidden)
            rows = MLX.where((row % 7) .== 0, MLXArray(Float(-0.0)), rows)
            rows = MLX.where(
                (row % 5) .== 3, MLX.where((column % 2) .== 0, MLXArray(Float(256)), -256), rows)
            rows = MLX.where((column % 13) .== 5, MLXArray(Float(1 + 1.0 / 256)), rows)
            let experts = rows.asType(.bfloat16).reshaped(batch, length, topK, hidden)
            var weights = MLXRandom.uniform(low: -1, high: 1, [batch, length, topK], key: keys[2])
            let slot = MLXArray(Int32(0) ..< Int32(tokens * topK)).reshaped(batch, length, topK)
            weights = MLX.where((slot % 4) .== 0, MLXArray(Float(-0.0)), weights)
            weights = MLX.where((slot % 9) .== 1, MLXArray(Float(0.5)), weights)
            let position = MLXArray(Int32(0) ..< Int32(tokens * hidden)).reshaped(
                batch, length, hidden)
            var addend = MLXRandom.normal([batch, length, hidden], key: keys[3])
            addend = MLX.where((position % 5) .== 0, MLXArray(Float(-0.0)), addend)
            addend = MLX.where((position % 9) .== 0, MLXArray(Float(0)), addend)
            var residual = MLXRandom.normal([batch, length, hidden], key: keys[4]) * 64
            residual = MLX.where((position % 7) .== 2, MLXArray(Float(-0.0)), residual)
            let addendBF16 = addend.asType(.bfloat16)
            let residualBF16 = residual.asType(.bfloat16)
            eval(experts, weights, addendBF16, residualBF16)
            let combined =
                (experts.asType(.float32) * weights[.ellipsis, .newAxis]).sum(axis: -2)
                .asType(.bfloat16) + addendBF16
            let context = "K=\(topK) [\(batch), \(length)]"
            #expect(
                flintFP32WeightedAddSupported(
                    expertOutputs: experts, weights: weights, addend: addendBF16,
                    residual: residualBF16))
            expectExact(
                flintFP32WeightedAdd(expertOutputs: experts, weights: weights, addend: addendBF16),
                combined, context)
            expectExact(
                flintFP32WeightedAdd(
                    expertOutputs: experts, weights: weights, addend: addendBF16,
                    residual: residualBF16),
                residualBF16 + combined, "\(context) + residual")
        }
    }

    /// Routed experts: the concatenated gate/up bank against the split banks
    /// it was built from (256 experts, INT4 g64), on unsorted decode rows
    /// (gather_qmv), a sorted small chunk (gather_qmv) and a sorted 2048-token
    /// chunk (gather_qmm_rhs). BF16 takes MLX's NAX gather on M5; FP32 (TF32
    /// off) takes the legacy kernels every GPU runs, so both families are held.
    @Test func routedGateUpFusionMatchesSplitBanksBytes() throws {
        let moe = FlintMoE(try configuration(experts: 256))
        try randomize(moe, seed: 8_900)
        let split = moe.experts
        let fused = try #require(split.fusingQuantizedGateUp())
        #expect(fused.hasFusedGateUp && !split.hasFusedGateUp)
        #expect(fused.downProj === split.downProj)
        let x = (MLXRandom.normal([3, 2048, 2048], key: MLXRandom.key(8_901)) * 0.5)
            .asType(.bfloat16)
        eval(x)
        for (batch, length) in [(1, 1), (3, 1), (1, 40), (1, 2048)] {
            let input = x[0 ..< batch, 0 ..< length]
            let (indices, _) = moe.route(input)
            eval(indices)
            for dtype in [DType.bfloat16, .float32] {
                let typed = input.asType(dtype)
                expectExact(
                    fused(typed, indices), split(typed, indices),
                    "gate_up [\(batch), \(length)] \(dtype)")
            }
        }
    }

    /// q/k/v and dense/shared gate/up: row views of one concatenated matrix.
    /// Single rows (q/k/v) and rows below MLX's qmv crossover (gate/up) take
    /// one fused matmul; everything else, including 2048-token chunks, the
    /// original projections on the views. All are compared with the separate
    /// projections the model held before fusion.
    @Test func projectionFusionMatchesSeparateProjectionsBytes() throws {
        let config = try configuration(experts: 256)
        let attention = FlintAttention(config, index: 1)
        let shared = FlintMLP(hidden: 2048, intermediate: 1024)
        let dense = FlintMLP(hidden: 2048, intermediate: 6144)
        for (index, module) in [attention, shared, dense].enumerated() {
            try randomize(module, seed: UInt64(9_000 + 10 * index))
        }
        let x = (MLXRandom.normal([1, 2048, 2048], key: MLXRandom.key(9_050)) * 0.5)
            .asType(.bfloat16)
        let rowCounts = [1, 3, 5, 6, 12, 13, 2048]
        func inputs(_ rows: Int) -> [MLXArray] {
            // Decode-shaped [rows, 1, H] and a prompt-shaped [1, rows, H].
            [x[0, 0 ..< rows].reshaped(rows, 1, 2048), x[0..., 0 ..< rows]]
        }
        eval(x)
        var expected = [[MLXArray]]()
        for rows in rowCounts {
            for input in inputs(rows) {
                let (q, k, v) = attention.projectQKV(input)
                let outputs = [q, k, v, shared(input), dense(input)]
                eval(outputs)
                expected.append(outputs)
            }
        }
        #expect(attention.fusedQKV.value == nil && shared.fusedGateUp.value == nil)
        try attention.fuseProjections()
        try shared.fuseProjections()
        try dense.fuseProjections()
        let qkv = try #require(attention.fusedQKV.value)
        let sharedFused = try #require(shared.fusedGateUp.value)
        let denseFused = try #require(dense.fusedGateUp.value)
        #expect(qkv.widths == [4096, 512, 512] && qkv.qmvRows == 6)
        #expect(sharedFused.widths == [1024, 1024] && sharedFused.qmvRows == 13)
        #expect(denseFused.widths == [6144, 6144] && denseFused.qmvRows == 6)
        var index = 0
        for rows in rowCounts {
            for input in inputs(rows) {
                // Which calls take the single fused matmul.
                let members = [attention.qProj, attention.kProj, attention.vProj]
                #expect((qkv(input, members: members, maxRows: 1) != nil) == (rows == 1))
                #expect(
                    (sharedFused(input, members: [shared.gate, shared.up], maxRows: 12) != nil)
                        == (rows <= 12))
                #expect(
                    (denseFused(input, members: [dense.gate, dense.up], maxRows: 5) != nil)
                        == (rows <= 5))
                let (q, k, v) = attention.projectQKV(input)
                let actual = [q, k, v, shared(input), dense(input)]
                for (name, (a, e)) in zip(
                    ["q", "k", "v", "shared", "dense"], zip(actual, expected[index]))
                {
                    expectExact(a, e, "\(name) rows \(rows) shape \(input.shape)")
                }
                index += 1
            }
        }
        // Any parameter update drops the fusion (the views may be replaced).
        try attention.update(parameters: attention.parameters(), verify: [])
        #expect(attention.fusedQKV.value == nil)
    }

    /// Real FlintMoE (256 experts, top-6, INT4 g64): every decode fusion
    /// (compiled router links, fused shared/routed gate-up, the slot-order
    /// combine with the residual folded in) against the original expressions
    /// on the separately stored weights, at decode batch 1 and 3 and on
    /// 2048-token chunks (one and two rows).
    @Test func flintMoEDecodeFusionsMatchOriginalBytes() throws {
        let moe = FlintMoE(try configuration(experts: 256))
        try randomize(moe, seed: 8_950)
        let x = (MLXRandom.normal([2, 2048, 2048], key: MLXRandom.key(8_951)) * 0.5)
            .asType(.bfloat16)
        let h = (MLXRandom.normal([2, 2048, 2048], key: MLXRandom.key(8_952)) * 4)
            .asType(.bfloat16)
        eval(x, h)
        let shapes = [(1, 1), (3, 1), (1, 2048), (2, 2048)]
        func slice(_ a: MLXArray, _ shape: (Int, Int)) -> MLXArray {
            shape.1 == 1
                ? a[0, 0 ..< shape.0].reshaped(shape.0, 1, 2048) : a[0 ..< shape.0]
        }
        moe.usesFusedRouter = false
        moe.usesFusedExpertCombine = false
        moe.shared.usesFusedProjection = false
        moe.experts.suppliesHostRowIndices = false
        var expected = [(MLXArray, MLXArray, MLXArray, MLXArray)]()
        for shape in shapes {
            let input = slice(x, shape)
            let (indices, weights) = moe.route(input)
            let output = slice(h, shape) + moe(input)
            let plain = moe(input)
            eval(indices, weights, output, plain)
            expected.append((indices, weights, output, plain))
        }
        try moe.fuseProjections()
        #expect(moe.experts.hasFusedGateUp && moe.shared.fusedGateUp.value != nil)
        moe.usesFusedRouter = true
        moe.usesFusedExpertCombine = true
        moe.shared.usesFusedProjection = true
        #expect(moe.experts.suppliesHostRowIndices == false)
        moe.experts.suppliesHostRowIndices = true
        for (shape, reference) in zip(shapes, expected) {
            let input = slice(x, shape)
            let residual = slice(h, shape)
            let context = "FlintMoE \(shape)"
            let (indices, weights) = moe.route(input)
            expectExact(indices, reference.0, "\(context) router indices")
            expectExact(weights, reference.1, "\(context) router weights")
            expectExact(moe(input, residual: residual), reference.2, "\(context) + residual")
            expectExact(moe(input), reference.3, context)
            let selection = moe.experts.flintFP32WeightedSumSelection(
                input, indices, weights: weights, adding: moe.shared(input), residual: residual)
            #expect(
                selection.kernel == (shape.1 == 1 ? .slotOrderKernel : .sortedUnsortKernel),
                Comment(rawValue: context))
        }
    }

    /// Decode-step graph size at the released depth and layer layout (49
    /// layers: 2 dense, 35 RoPE sliding MoE, 12 full NoPE MoE). Graph topology
    /// does not depend on the expert count, so 8 experts keep this cheap; the
    /// other dimensions are real. The fused model's logits and K/V also match
    /// the unfused model's bytes at full depth.
    @Test func decodeGraphShrinksAtFullDepth() throws {
        let model = FlintModel(try configuration(experts: 8, fullDepth: true))
        try randomize(model, seed: 9_700)
        let input = streams(rows: 3, length: 64 + 2)
        Self.setDecodeFusions(model, false)
        var expected = [Run]()
        for rows in [1, 3] {
            expected.append(
                try pagedRun(
                    model, streams: Array(input.prefix(rows)), chunks: [64],
                    paths: Paths(fusedExpertCombine: false), decodeSteps: 2))
        }
        try model.fuseProjections()
        Self.setDecodeFusions(model, true)
        for (rows, reference) in zip([1, 3], expected) {
            let actual = try pagedRun(
                model, streams: Array(input.prefix(rows)), chunks: [64], paths: Paths(),
                decodeSteps: 2)
            expectSameRun(actual, reference, "full depth, \(rows) rows")
            print(
                "[flint decode graph] layers 49, rows \(rows): "
                    + "before \(Self.describe(reference.decodeGraph)); "
                    + "after \(Self.describe(actual.decodeGraph))")
            #expect(
                actual.decodeGraph.values.reduce(0, +)
                    < reference.decodeGraph.values.reduce(0, +))
        }
    }

    /// The whole decoder through the production PagedKVBackend at real
    /// geometry (hidden 2048, 256 experts top-6, INT4 g64): a 2048-token
    /// prompt chunk per row, then six decode steps at batch 1 and batch 3.
    /// Every chunk output, decode logit and retained K/V byte matches the
    /// original expressions on separately stored weights. Also reports the
    /// decode step's graph primitives before and after.
    @Test func pagedDecodeFusionsMatchUnfusedBytes() throws {
        let model = FlintModel(try configuration(experts: 256))
        try randomize(model, seed: 9_500)
        let input = streams(rows: 3, length: 2048 + 7)
        Self.setDecodeFusions(model, false)
        var expected = [Run]()
        for rows in [1, 3] {
            expected.append(
                try pagedRun(
                    model, streams: Array(input.prefix(rows)), chunks: [2048],
                    paths: Paths(fusedExpertCombine: false), decodeSteps: 6))
        }
        try model.fuseProjections()
        Self.setDecodeFusions(model, true)
        for (rows, reference) in zip([1, 3], expected) {
            let actual = try pagedRun(
                model, streams: Array(input.prefix(rows)), chunks: [2048], paths: Paths(),
                decodeSteps: 6)
            expectSameRun(actual, reference, "decode fusions, \(rows) rows")
            let before = reference.decodeGraph.values.reduce(0, +)
            let after = actual.decodeGraph.values.reduce(0, +)
            print(
                "[flint decode graph] layers \(model.model.layers.count), rows \(rows): "
                    + "before \(Self.describe(reference.decodeGraph)); "
                    + "after \(Self.describe(actual.decodeGraph))")
            #expect(after < before)
        }
    }
}
