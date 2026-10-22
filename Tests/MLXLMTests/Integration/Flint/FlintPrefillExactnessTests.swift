// Exact-bytes gates for Flint paged prefill speedups. Every fast path is
// compared with the unmodified expression on identical inputs: native BF16
// activations and KV, affine INT4 g64 weights, Flint's real layer geometry and
// 2048-token chunks through the production PagedKVBackend. Flint geometry
// follows ThirdPartyNotices/Flint.md: Prince Canuma/mlx-vlm (MIT) and Recursive
// rec (Apache-2.0). Random weights only; no checkpoint or downloaded data.
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
    /// MoE layer.
    private func configuration(experts: Int) throws -> FlintConfiguration {
        let url = try #require(
            Bundle.module.url(forResource: "flint-oracle", withExtension: "json"))
        let fixture = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var config = try #require(fixture["config"] as? [String: Any])
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
        quantize(model: module, groupSize: 64, bits: 4) { path, _ in
            !path.hasSuffix("mlp.gate") && !path.contains("embed_tokens")
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
    }

    private func apply(_ paths: Paths, to model: FlintModel, caches: [PagedLayerCache]) {
        for layer in model.model.layers {
            (layer.mlp as? FlintMoE)?.usesFusedExpertCombine = paths.fusedExpertCombine
        }
        for cache in caches { cache.batchesUniformQueryBlocks = paths.batchedQueryBlocks }
    }

    struct Run {
        var outputs: [MLXArray] = []
        var kv: [MLXArray] = []
        var batchedQueryBlockCalls = 0
    }

    /// Prefill `streams` (one per packed row) chunk by chunk through a fresh
    /// PagedKVBackend, then decode one token. Records every chunk's full
    /// hidden states, the decode logits and every row's retained K/V.
    private func pagedRun(
        _ model: FlintModel, streams: [[Int32]], chunks: [Int], paths: Paths
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
                    layerKinds: kinds, promptLength: total, maxLength: total + 8))
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
            let hidden = model.model(tokens, cache: caches)
            eval([hidden] + caches.flatMap { $0.innerState() })
            #expect(!backend.pool.writeValidation.isFaulted)
            run.outputs.append(hidden)
            start += count
        }
        let next = MLXArray(streams.map { $0[total] }).reshaped(streams.count, 1)
        let logits = model(next, cache: caches)
        eval([logits] + caches.flatMap { $0.innerState() })
        run.outputs.append(logits)
        for state in states {
            for entry in state {
                let row = try #require(entry as? PagedSequenceKV)
                let snapshot = row.snapshot()
                run.kv += [snapshot.keys, snapshot.values]
            }
        }
        eval(run.kv)
        run.batchedQueryBlockCalls = caches.reduce(0) { $0 + $1.batchedQueryBlockCallsForTesting }
        return run
    }

    private func streams(rows: Int, length: Int) -> [[Int32]] {
        (0 ..< rows).map { row in
            (0 ..< length).map { Int32(($0 * (7 + 2 * row) + 3 * row + 1) % 251) }
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
    /// has that property (all 2048-token chunks); a path change may never
    /// alter it.
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
            if index < 3 {
                #expect(actualIndependent, "\(context) 2048-chunk \(index) packed vs alone")
            }
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

    @Test func pagedPrefillFusedExpertCombineMatchesOriginalBytes() throws {
        try expectPathsExact(
            Paths(fusedExpertCombine: false), Paths(), "fused expert combine")
    }
}
