// Exact-byte regression for Flint's paged decode. Flint geometry follows
// ThirdPartyNotices/Flint.md: Prince Canuma/mlx-vlm (MIT) and Recursive rec
// (Apache-2.0). No checkpoint or downloaded data.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A tiny BF16 Flint with the production attention shape (head dim 128, GQA 8,
/// learned sinks, 513-token sliding window, RoPE only on sliding layers) runs
/// through native segmented paged caches. Every comparison is of native bytes.
@Suite(.serialized)
struct FlintPagedDecodeTests {
    /// Per-row prompt lengths: both wrapped windows (ring reuse) and a row whose
    /// sliding window has not filled (attend start 0) in the same batch.
    static let prompts = [1100, 300, 1043]
    static let decodeSteps = 40
    static let pageBytes = 2 * 16 * 128 * DType.bfloat16.size

    static func hostTensor(_ shape: [Int], seed: UInt64) -> MLXArray {
        var state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 1
        let values: [Float] = (0 ..< shape.reduce(1, *)).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int((state >> 33) & 0xffff) - 32_768) / 32_768
        }
        return MLXArray(values).reshaped(shape)
    }

    static func model() throws -> FlintModel {
        let config: [String: Any] = [
            "model_type": "flint", "hidden_size": 64, "intermediate_size": 96,
            "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 1,
            "head_dim": 128, "vocab_size": 64, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262_144, "rope_theta": 10_000.0,
            "layer_types": [
                "full_attention", "sliding_attention", "sliding_attention", "full_attention",
            ],
            "no_rope_layer_indices": [0, 3], "mlp_only_layers": [0], "sliding_window": 513,
            "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64, "tie_word_embeddings": false,
            "hidden_act": "silu", "router_score_function": "sigmoid",
            "softmax_type": "learnable", "norm_topk_prob": true, "attention_bias": false,
            "use_sliding_window": true, "decoder_sparse_step": 1,
        ]
        let configuration = try JSONDecoder().decode(
            FlintConfiguration.self, from: JSONSerialization.data(withJSONObject: config))
        let model = FlintModel(configuration)
        var weights = [String: MLXArray]()
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        for (index, (name, value)) in parameters.enumerated() {
            let noise = hostTensor(value.shape, seed: UInt64(index + 1))
            let scaled =
                name.contains("norm")
                ? 1 + noise * 0.05 : noise / sqrt(Float(value.shape.last ?? 1))
            weights[name] = scaled.asType(.bfloat16)
        }
        try model.installFlintWeights(weights)
        eval(model)
        return model
    }

    struct Trace: Equatable {
        var prefill: [Data] = []
        var decode: [Data] = []
        var kv: [Data] = []
        /// Most segment buckets any cached decode plan used (0 for reference).
        var maxBuckets = 0
    }

    struct Mode {
        /// Decode steps run under Flint's decode command-buffer limits.
        var decodeLimits = false
        /// Layer caches run segmented decode as before metadata reuse.
        var referenceCaches = false
    }

    /// Prefills each row alone (256-token chunks), then decodes the batch with
    /// a fixed token schedule so a mismatch cannot propagate through sampling.
    static func run(
        _ model: FlintModel, batch: Int, segmentBytes: Int, mode: Mode,
        forward: (MLXArray, [KVCache]) -> MLXArray
    ) throws -> Trace {
        let kinds = model.cbv2LayerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, dtype: .bfloat16, maxPrefillChunk: 256,
                nominalMaxSequenceLength: 2048, segmentSizeBytes: segmentBytes,
                layerDTypes: Array(repeating: .bfloat16, count: kinds.count)))
        let caches = backend.makeLayerCaches()
        for cache in caches { cache.segmentDecodeReference = mode.referenceCaches }
        let prompts = Array(Self.prompts.prefix(batch))
        var states: [[CBv2SequenceKV?]] = []
        defer {
            eval(caches.flatMap { $0.innerState() })
            for cache in caches { cache.setRows([]) }
            for state in states { backend.release(state) }
        }
        for length in prompts {
            states.append(
                try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: length,
                    maxLength: length + Self.decodeSteps + 8))
        }
        var trace = Trace()
        for (lane, length) in prompts.enumerated() {
            for (index, cache) in caches.enumerated() { cache.setRows([states[lane][index]!]) }
            let tokens = (0 ..< length).map { Int32(($0 * 7 + 3 + lane * 5) % 64) }
            for start in stride(from: 0, to: length, by: 256) {
                let chunk = MLXArray(Array(tokens[start ..< min(start + 256, length)]))
                let logits = forward(chunk.reshaped(1, -1), caches)
                eval([logits] + caches.flatMap { $0.innerState() })
                if start + 256 >= length {
                    trace.prefill.append(logits[0..., -1, 0...].asData(access: .copy).data)
                }
            }
        }
        for (index, cache) in caches.enumerated() { cache.setRows(states.map { $0[index]! }) }
        for step in 0 ..< Self.decodeSteps {
            let tokens = MLXArray((0 ..< batch).map { Int32((step * 11 + $0 * 17 + 1) % 64) })
            func decode() -> MLXArray {
                let logits = forward(tokens.reshaped(batch, 1), caches)
                eval([logits] + caches.flatMap { $0.innerState() })
                return logits
            }
            let logits =
                mode.decodeLimits
                ? GPU.withCommandBufferLimits(FlintDecodeCommandBuffers.limits) { decode() }
                : decode()
            #expect(!backend.pool.writeValidation.isFaulted)
            trace.decode.append(logits.asData(access: .copy).data)
            for cache in caches {
                let buckets = cache.segmentDispatchCache.prepared?.plan.buckets.count ?? 0
                trace.maxBuckets = max(trace.maxBuckets, buckets)
            }
        }
        for state in states {
            for row in state {
                let snapshot = try #require(row).snapshot()
                eval(snapshot.keys, snapshot.values)
                trace.kv.append(snapshot.keys.asData(access: .copy).data)
                trace.kv.append(snapshot.values.asData(access: .copy).data)
            }
        }
        return trace
    }

    static func expectSame(_ actual: Trace, _ expected: Trace, _ context: String) {
        let note = Comment(rawValue: context)
        #expect(actual.prefill.count == expected.prefill.count, note)
        #expect(actual.decode.count == Self.decodeSteps, note)
        for (index, (a, e)) in zip(actual.prefill, expected.prefill).enumerated() {
            #expect(a == e, Comment(rawValue: "\(context) prefill row \(index)"))
        }
        for (step, (a, e)) in zip(actual.decode, expected.decode).enumerated() {
            #expect(a == e, Comment(rawValue: "\(context) decode step \(step)"))
        }
        #expect(actual.kv == expected.kv, note)
    }

    /// Flint's forward exactly as it was before sinks and RoPE offsets were
    /// reused: each layer narrows its FP32 sinks to the query dtype and copies
    /// its own cache's offsets (`+ 0`) on every call.
    static func legacyForward(_ m: FlintModel, _ inputs: MLXArray, _ cache: [KVCache]) -> MLXArray {
        let c = m.configuration
        var h = m.model.embedding(inputs)
        for (index, layer) in m.model.layers.enumerated() {
            let a = layer.attention
            let x = layer.inputNorm(h)
            let (batch, length) = (x.dim(0), x.dim(1))
            var q = a.qNorm(a.qProj(x).reshaped(batch, length, c.attentionHeads, c.headDim))
                .transposed(0, 2, 1, 3)
            var k = a.kNorm(a.kProj(x).reshaped(batch, length, c.kvHeads, c.headDim))
                .transposed(0, 2, 1, 3)
            let v = a.vProj(x).reshaped(batch, length, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
            let scale = 1 / sqrt(Float(c.headDim))
            let attending = cache[index] as! any CBv2AttendingLayerCache
            if let rope = a.rope {
                let offsets = attending.positionOffsets + 0
                q = rope(q, offset: offsets)
                k = rope(k, offset: offsets)
            }
            let output = attending.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: a.sinks.asType(q.dtype))
            let attended = h + a.oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
            h = attended + layer.mlp(layer.postNorm(attended))
        }
        let hidden = m.model.norm(h).asType(.float32)
        return m.lmHead?(hidden) ?? m.model.embedding.asLinear(hidden)
    }

    @Test(arguments: [1, 3])
    func sharedSinksAndRopeOffsetsKeepExactBytes(batch: Int) throws {
        let model = try Self.model()
        let expected = try Self.run(
            model, batch: batch, segmentBytes: 1 << 20, mode: .init(),
            forward: { Self.legacyForward(model, $0, $1) })
        let actual = try Self.run(
            model, batch: batch, segmentBytes: 1 << 20, mode: .init(),
            forward: { model($0, cache: $1) })
        Self.expectSame(actual, expected, "batch \(batch)")
    }

    @Test func narrowedSinksAreBuiltOncePerParameterAndDtype() throws {
        let model = try Self.model()
        let parameterNames = model.parameters().flattened().map(\.0).sorted()
        let attention = model.model.layers[1].attention
        let bytes = { (array: MLXArray) in array.asData(access: .copy).data }
        let narrowed = attention.querySinks(.bfloat16)
        #expect(attention.querySinks(.bfloat16) === narrowed)
        #expect(narrowed.dtype == .bfloat16)
        #expect(bytes(narrowed) == bytes(attention.sinks.asType(.bfloat16)))
        let wide = attention.querySinks(.float32)
        #expect(wide !== narrowed && bytes(wide) == bytes(attention.sinks))
        // The cache is not a module parameter, even once populated.
        #expect(model.parameters().flattened().map(\.0).sorted() == parameterNames)
        // A parameter update replaces the source array and refreshes the cache.
        let replacement = (attention.sinks + 0.375).asType(.float32)
        try attention.update(
            parameters: ModuleParameters.unflattened(["softmax_offset": replacement]),
            verify: .noUnusedKeys)
        let refreshed = attention.querySinks(.bfloat16)
        #expect(refreshed !== narrowed)
        #expect(bytes(refreshed) == bytes(replacement.asType(.bfloat16)))
    }

    /// Every decode change at once against the unmodified path: the legacy
    /// forward, reference layer caches and MLX's default command buffers.
    @Test(arguments: [1, 3])
    func decodeMatchesTheUnmodifiedPathByteForByte(batch: Int) throws {
        let model = try Self.model()
        for segmentBytes in [1 << 20, Self.pageBytes * 3] {
            let expected = try Self.run(
                model, batch: batch, segmentBytes: segmentBytes,
                mode: .init(referenceCaches: true),
                forward: { Self.legacyForward(model, $0, $1) })
            let actual = try Self.run(
                model, batch: batch, segmentBytes: segmentBytes, mode: .init(decodeLimits: true),
                forward: { model($0, cache: $1) })
            Self.expectSame(actual, expected, "batch \(batch), segment \(segmentBytes)")
            #expect(expected.maxBuckets == 0)
            if segmentBytes < 1 << 20 {
                // Small segments split a dispatch into several buckets, so the
                // first allocates the merge scratch and later ones write into it.
                #expect(actual.maxBuckets > 1)
            }
        }
    }

    @Test func decodeLimitsAreDeclaredThroughTheSteppableAdapter() throws {
        let model = try Self.model()
        #expect(model.decodeCommandBufferLimits == FlintDecodeCommandBuffers.limits)
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        #expect(adapter.decodeCommandBufferLimits == FlintDecodeCommandBuffers.limits)
        #expect(FlintDecodeCommandBuffers.limits.operations == 100)
        #expect(FlintDecodeCommandBuffers.limits.megabytes == 1 << 20)
    }

    @Test(arguments: [1, 3])
    func decodeCommandBufferLimitsKeepExactBytes(batch: Int) throws {
        let model = try Self.model()
        for segmentBytes in [1 << 20, Self.pageBytes * 3] {
            let expected = try Self.run(
                model, batch: batch, segmentBytes: segmentBytes, mode: .init(),
                forward: { model($0, cache: $1) })
            let actual = try Self.run(
                model, batch: batch, segmentBytes: segmentBytes, mode: .init(decodeLimits: true),
                forward: { model($0, cache: $1) })
            Self.expectSame(actual, expected, "batch \(batch), segment \(segmentBytes)")
        }
    }
}
