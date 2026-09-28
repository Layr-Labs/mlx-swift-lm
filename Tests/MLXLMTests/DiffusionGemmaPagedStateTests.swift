import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@testable import MLXLLM

/// Page-backed storage followed by the UNCHANGED native SDPA math. These tests
/// do not qualify the numerically different scalar paged decode kernel.
@Suite("DiffusionGemma page-backed native state", .serialized)
struct DiffusionGemmaPagedStateTests {
    func configuration() throws -> DiffusionGemmaTextConfiguration {
        let url = try #require(
            Bundle.module.url(forResource: "diffusiongemma-text-oracle", withExtension: "json"))
        let root = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var fields = try #require(root["model_config"] as? [String: Any])
        fields["head_dim"] = 256
        fields["global_head_dim"] = 512
        fields["num_attention_heads"] = 16
        fields["num_key_value_heads"] = 8
        fields["num_global_key_value_heads"] = 2
        fields["sliding_window"] = 32
        fields["max_position_embeddings"] = 512
        return try JSONDecoder().decode(
            DiffusionGemmaTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: fields))
    }
    func backend(_ config: DiffusionGemmaTextConfiguration, dtype: DType, chunk: Int = 32) throws
        -> PagedKVBackend
    {
        try .init(
            layerKinds: config.diffusionPagedLayerKinds,
            config: .init(
                capacityBytes: 64 << 20, dtype: dtype, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: 128, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: dtype, count: config.layerCount)))
    }
    private func exact(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        eval(a, b)
        #expect(a.shape == b.shape && a.dtype == b.dtype, "\(label) geometry")
        #expect(
            a.asArray(Float.self).map(\.bitPattern) == b.asArray(Float.self).map(\.bitPattern),
            "\(label) raw bits")
    }
    private func exactCache(_ a: DiffusionGemmaRequestCache, _ b: DiffusionGemmaRequestCache) {
        #expect(a.position == b.position && a.windowOrder == b.windowOrder)
        for (left, right) in zip(a.snapshots(), b.snapshots()) {
            exact(left.keys, right.keys, "keys")
            exact(left.values, right.values, "values")
        }
    }
    private func identity() throws -> DiffusionGemmaPrefixIdentity {
        try .init(
            tenantScope: "fixture", artifact: "generated-fixture", template: "test",
            media: "text-only",
            numericalProfile: "native-sdpa", epoch: "one")
    }

    @Test func preparedVisualBlocksPreserveBidirectionalAttentionAcrossPages() throws {
        let config = try configuration()
        let model = DiffusionGemmaTextDecoder(config)
        let scalars = DiffusionGemmaEncoderTextParameters(layerCount: config.layerCount)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        scalars.update(parameters: scalars.parameters().mapValues { $0.asType(.bfloat16) })
        eval(model, scalars)
        let pool = try backend(config, dtype: .bfloat16)
        try compareVisualState(model: model, scalars: scalars, pool: pool)
        #expect(pool.bytesInUse == 0 && pool.bytesReserved == 0 && pool.bytesWired == 0)
    }

    private func compareVisualState(
        model: DiffusionGemmaTextDecoder,
        scalars: DiffusionGemmaEncoderTextParameters, pool: PagedKVBackend
    ) throws {
        try MLX.withError { errors in
            let cold = try DiffusionGemmaRequestCache(
                configuration: model.configuration,
                expectedPromptLength: 65, maximumSequenceLength: 96)
            let paged = try DiffusionGemmaRequestCache(
                configuration: model.configuration,
                expectedPromptLength: 65, maximumSequenceLength: 96, pagedBackend: pool)
            for index in 0 ..< 2 {
                let tokens = MLXArray((0 ..< 32).map { Int32(($0 + index * 32) % 100 + 2) })
                    .reshaped(1, 32)
                // A whole visual block spans physical page boundaries. Its
                // prepared embeddings are immutable source values, not a claim
                // that this generated fixture ran the vision tower.
                let values = MLXArray(
                    (0 ..< 32 * model.configuration.hiddenSize).map {
                        Float(($0 * 13 + index) % 71) / 71 - 0.5
                    }
                ).reshaped(1, 32, model.configuration.hiddenSize).asType(.bfloat16)
                let blocks = MLXArray(
                    (0 ..< 32).map { Int32((8 ..< 28).contains($0) ? index : -1) }
                ).reshaped(1, 32)
                let a = try model.encode(
                    tokenIds: tokens, cache: cold, encoderParameters: scalars,
                    preparedEmbeddings: values, visualBlockIds: blocks)
                let b = try model.encode(
                    tokenIds: tokens, cache: paged, encoderParameters: scalars,
                    preparedEmbeddings: values, visualBlockIds: blocks)
                try errors.check()
                exact(a, b, "visual encoder")
                exactCache(cold, paged)
                try errors.check()
            }
            let tail = MLXArray([Int32(73)]).reshaped(1, 1)
            exact(
                try model.encode(tokenIds: tail, cache: cold, encoderParameters: scalars),
                try model.encode(tokenIds: tail, cache: paged, encoderParameters: scalars),
                "visual text tail")
            let canvas = MLXArray([Int32(2), 3, 5, 7]).reshaped(1, 4)
            exact(
                try model.denoise(canvasIds: canvas, cache: cold),
                try model.denoise(canvasIds: canvas, cache: paged), "visual denoise")
            try errors.check()
        }
    }

    @Test(arguments: ["fp32", "bf16"])
    func promptCanvasWrapRestoreAndTeardownAreExact(_ precision: String) throws {
        let config = try configuration()
        let dtype: DType = precision == "fp32" ? .float32 : .bfloat16
        let model = DiffusionGemmaTextDecoder(config)
        let scalars = DiffusionGemmaEncoderTextParameters(layerCount: config.layerCount)
        model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
        scalars.update(parameters: scalars.parameters().mapValues { $0.asType(dtype) })
        eval(model, scalars)
        let pool = try backend(config, dtype: dtype)
        for chunks in [[16, 16, 1, 1, 31, 1, 1, 1], [3, 29, 7, 25, 1, 16]] {
            try exercise(model: model, scalars: scalars, pool: pool, chunks: chunks)
            #expect(
                pool.bytesInUse == 0 && pool.bytesReserved == 0 && pool.bytesWired == 0,
                "All row pages and segmented backing retire between requests")
        }
    }

    private func exercise(
        model: DiffusionGemmaTextDecoder, scalars: DiffusionGemmaEncoderTextParameters,
        pool: PagedKVBackend, chunks: [Int]
    ) throws {
        try MLX.withError { errors in
            let length = chunks.reduce(0, +)
            let cold = try DiffusionGemmaRequestCache(
                configuration: model.configuration, expectedPromptLength: length,
                maximumSequenceLength: 128)
            let paged = try DiffusionGemmaRequestCache(
                configuration: model.configuration, expectedPromptLength: length,
                maximumSequenceLength: 128, pagedBackend: pool)
            #expect(paged.usesPagedStorage && !cold.usesPagedStorage && pool.bytesReserved > 0)
            var offset = 0
            for chunk in chunks {
                let tokens = MLXArray((offset ..< offset + chunk).map { Int32($0 % 100 + 2) })
                    .reshaped(1, chunk)
                let a = try model.encode(tokenIds: tokens, cache: cold, encoderParameters: scalars)
                let b = try model.encode(tokenIds: tokens, cache: paged, encoderParameters: scalars)
                try errors.check()
                exact(a, b, "encoder")
                exactCache(cold, paged)
                try errors.check()
                offset += chunk
            }
            let before = paged.stateArrays().map { $0.asArray(Float.self).map(\.bitPattern) }
            for width in [4, 32, 64] {
                let canvas = MLXArray((0 ..< width).map { Int32($0 % 100 + 2) }).reshaped(1, width)
                let a = try model.denoise(canvasIds: canvas, cache: cold)
                let b = try model.denoise(canvasIds: canvas, cache: paged)
                try errors.check()
                exact(a, b, "denoise")
                try errors.check()
                exact(
                    try model.denoise(canvasIds: canvas, cache: cold, selfConditioningLogits: a),
                    try model.denoise(canvasIds: canvas, cache: paged, selfConditioningLogits: a),
                    "conditioned canvas")
            }
            #expect(
                paged.position == length
                    && before
                        == paged.stateArrays().map { $0.asArray(Float.self).map(\.bitPattern) }
            )
            let checkpoint = try model.checkpoint(cache: paged, identity: identity(), compact: true)
            let tokenIds = MLXArray((0 ..< length + 1).map { Int32($0 % 100 + 2) }).reshaped(
                1, length + 1)
            let restored = try model.restorePrefix(
                checkpoint, identity: identity(), promptTokenIds: tokenIds,
                maximumSequenceLength: 128, pagedBackend: pool)
            exactCache(paged, restored)
            let last = tokenIds[0..., (length)...]
            exact(
                try model.encode(tokenIds: last, cache: cold, encoderParameters: scalars),
                try model.encode(tokenIds: last, cache: restored, encoderParameters: scalars),
                "restored singleton")
            try errors.check()
        }
    }

    @Test func rejectsOversizedChunksAndMismatchedGeometryWithoutMutatingState() throws {
        let config = try configuration()
        let model = DiffusionGemmaTextDecoder(config)
        let scalars = DiffusionGemmaEncoderTextParameters(layerCount: config.layerCount)
        let pool = try backend(config, dtype: .float32, chunk: 16)
        do {
            let cache = try DiffusionGemmaRequestCache(
                configuration: config, expectedPromptLength: 32, maximumSequenceLength: 64,
                pagedBackend: pool)
            #expect(throws: DiffusionGemmaModelError.self) {
                try model.encode(
                    tokenIds: MLXArray(Array(repeating: Int32(2), count: 17)).reshaped(1, 17),
                    cache: cache, encoderParameters: scalars)
            }
            #expect(cache.position == 0)
        }
        #expect(pool.bytesReserved == 0 && pool.bytesInUse == 0)
        var wrong = config.diffusionPagedLayerKinds
        wrong[0].kvHeads = 4
        let mismatched = try PagedKVBackend(
            layerKinds: wrong,
            config: .init(
                capacityBytes: 64 << 20, dtype: .float32, maxPrefillChunk: 16,
                segmentSizeBytes: 1 << 20))
        #expect(throws: DiffusionGemmaModelError.self) {
            try DiffusionGemmaRequestCache(
                configuration: config, expectedPromptLength: 16, maximumSequenceLength: 64,
                pagedBackend: mismatched)
        }
        #expect(mismatched.bytesReserved == 0 && mismatched.bytesInUse == 0)
        let wrongType = try backend(config, dtype: .bfloat16)
        let invalid = try DiffusionGemmaRequestCache(
            configuration: config, expectedPromptLength: 4,
            maximumSequenceLength: 64, pagedBackend: wrongType)
        #expect(throws: DiffusionGemmaModelError.self) {
            try model.encode(
                tokenIds: MLXArray([Int32(2), 3]).reshaped(1, 2), cache: invalid,
                encoderParameters: scalars)
        }
        #expect(wrongType.bytesInUse == 0 && wrongType.bytesReserved == 0 && invalid.position == 0)
        #expect(throws: DiffusionGemmaModelError.self) {
            try model.encode(
                tokenIds: MLXArray([Int32(2)]).reshaped(1, 1), cache: invalid,
                encoderParameters: scalars)
        }
    }

    @Test func interleavedNativeSessionsCancelOneAndPreserveTheSurvivor() throws {
        let config = try configuration()
        let fields = try JSONSerialization.jsonObject(with: JSONEncoder().encode(config))
        let root: [String: Any] = [
            "model_type": "diffusion_gemma", "text_config": fields,
            "canvas_length": 4, "tie_word_embeddings": true, "eos_token_id": [1],
        ]
        let model = try DiffusionGemma(
            JSONDecoder().decode(
                DiffusionGemmaConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root)))
        eval(model)
        let pool = try backend(config, dtype: .float32)
        let recipe = try DiffusionGemmaGenerationConfiguration(maxNewTokens: 12, eosTokenIds: [1])
        let a = MLXArray((0 ..< 67).map { Int32($0 % 100 + 2) }).reshaped(1, 67)
        let b = MLXArray((0 ..< 81).map { Int32(($0 * 7) % 100 + 2) }).reshaped(1, 81)
        let baseline = try model.generateNative(
            promptTokenIds: b, generation: recipe, seed: 341, prefillChunkSize: 32)
        let cancelled = try DiffusionGemmaGenerationSession(
            model: model, promptTokenIds: a, generation: recipe,
            seed: 87, prefillChunkSize: 32, pagedBackend: pool)
        let survivor = try DiffusionGemmaGenerationSession(
            model: model, promptTokenIds: b, generation: recipe,
            seed: 341, prefillChunkSize: 32, pagedBackend: pool)
        _ = try cancelled.advance()
        _ = try survivor.advance()
        _ = try cancelled.advance()
        _ = try survivor.advance()
        let beforeCancel = pool.bytesReserved
        cancelled.cancel()
        #expect(cancelled.phase == .cancelled && cancelled.retainedStateBytes == 0)
        #expect(pool.bytesReserved > 0 && pool.bytesReserved < beforeCancel)
        while survivor.phase != .finished { _ = try survivor.advance() }
        let result = try #require(survivor.result)
        #expect(
            result.tokenIds == baseline.tokenIds && result.denoisingSteps == baseline.denoisingSteps
        )
        #expect(
            result.generatedTokenCount == baseline.generatedTokenCount
                && result.finishReason == baseline.finishReason)
        #expect(pool.bytesInUse == 0 && pool.bytesReserved == 0 && pool.bytesWired == 0)
        let readmitted = try model.generateNative(
            promptTokenIds: b, generation: recipe, seed: 341,
            prefillChunkSize: 32, pagedBackend: pool)
        #expect(
            readmitted.tokenIds == baseline.tokenIds
                && readmitted.denoisingSteps == baseline.denoisingSteps)
        #expect(pool.bytesInUse == 0 && pool.bytesReserved == 0 && pool.bytesWired == 0)
        let faulted = try DiffusionGemmaGenerationSession(
            model: model, promptTokenIds: b, generation: recipe,
            seed: 341, prefillChunkSize: 32, pagedBackend: pool, prefixIdentity: identity(),
            onEncodedBoundary: { _, _ in
                let invalid = MLXArray.ones([2, 3]) + MLXArray.ones([4, 5])
                eval(invalid)
            })
        #expect(throws: MLX.MLXError.self) { try faulted.advance() }
        #expect(
            faulted.phase == .failed && faulted.generatedTokenCount == 0
                && faulted.retainedStateBytes == 0)
        #expect(pool.bytesInUse == 0 && pool.bytesReserved == 0 && pool.bytesWired == 0)
        let afterFailure = try model.generateNative(
            promptTokenIds: b, generation: recipe, seed: 341,
            prefillChunkSize: 32, pagedBackend: pool)
        #expect(
            afterFailure.tokenIds == baseline.tokenIds
                && afterFailure.denoisingSteps == baseline.denoisingSteps)
        #expect(pool.bytesInUse == 0 && pool.bytesReserved == 0 && pool.bytesWired == 0)
    }
}
