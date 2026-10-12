import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// Actual dense/MoE wrapper projections, Q/K norms and three-axis M-RoPE.
/// Short histories remain native; the shared packed primitive has separate
/// long-history numerical tests. No public-model answer-quality claim here.
@Suite("Qwen3-VL paged wrapper qualification", .serialized)
struct Qwen3VLPagedQuantizedTests {
    private func model(moe: Bool) throws -> Qwen3VL {
        var text: [String: Any] = [
            "model_type": moe ? "qwen3_vl_moe_text" : "qwen3_vl_text",
            "hidden_size": 128, "intermediate_size": 96, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "max_position_embeddings": 256, "vocab_size": 64, "rope_theta": 10000,
            "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
            "rope_scaling": [
                "type": "mrope", "mrope_interleaved": true, "mrope_section": [8, 8, 16],
            ],
        ]
        if moe {
            text.merge([
                "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
                "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": true,
            ]) { _, new in new }
        }
        let values: [String: Any] = [
            "model_type": moe ? "qwen3_vl_moe" : "qwen3_vl", "image_token_id": 60,
            "video_token_id": 61, "vision_start_token_id": 57, "vision_end_token_id": 58,
            "vision_token_id": 59, "text_config": text,
            "vision_config": [
                "model_type": "qwen3_vl", "depth": 1, "hidden_size": 32,
                "intermediate_size": 48, "out_hidden_size": 128, "num_heads": 4,
                "patch_size": 4, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "num_position_embeddings": 16, "deepstack_visual_indexes": [0],
            ],
        ]
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
        let model = Qwen3VL(config)
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        let weights = parameters.enumerated().compactMap { index, pair -> (String, MLXArray)? in
            let (name, array) = pair
            guard array.dtype.isFloatingPoint else { return nil }
            let input = MLXArray(0 ..< array.size).asType(.float32).reshaped(array.shape)
            let wave = sin(input * 0.017 + Float(index) * 0.031)
            let value = array.ndim <= 1 ? wave * 0.01 + 1 : wave / Float(array.dim(-1)).squareRoot()
            return (name, value.asType(.float32))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
        return model
    }

    @Test(arguments: [false, true], [false, true])
    func actualWrapperTextAndPositionedCausalEmbeddingMatchNative(moe: Bool, embeddings: Bool)
        throws
    {
        let model = try model(moe: moe)
        let kinds = model.cbv2LayerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 16 << 20, dtype: .float32, maxPrefillChunk: 16,
                nominalMaxSequenceLength: 64, maxBufferLength: 16 << 20,
                segmentSizeBytes: 64 << 10, quantization: .init()))
        let state = try backend.makeSequenceState(layerKinds: kinds, promptLength: 8, maxLength: 64)
        defer { backend.release(state) }
        let paged = backend.makeLayerCaches()
        let native = kinds.enumerated().map {
            CBv2LayerCache(layerIndex: $0.offset, kind: $0.element)
        }
        for index in kinds.indices {
            paged[index].setRows([state[index]!])
            native[index].setRows([
                CBv2FullSequenceKV(
                    promptLength: 8, maxLength: 64, kvHeads: kinds[index].kvHeads,
                    headDim: kinds[index].headDim)
            ])
        }
        var start = 0
        for count in [5, 2, 1] {
            let ids = MLXArray((start ..< start + count).map { Int32(3 + $0) }, [1, count])
            var positionValues: [Int32] = []
            positionValues.reserveCapacity(3 * count)
            for axis in 0 ..< 3 {
                for position in start ..< start + count {
                    let coordinate: Int
                    switch axis {
                    case 0: coordinate = position
                    case 1: coordinate = 2 * position
                    default: coordinate = position / 2
                    }
                    positionValues.append(Int32(coordinate))
                }
            }
            let positions = MLXArray(positionValues, [3, 1, count])
            let expected: MLXArray
            let actual: MLXArray
            if embeddings {
                let embedding = model.scaledInputEmbeddings(ids) + 0.003
                expected = model.embeddingForward(
                    ids, inputEmbedding: embedding,
                    cache: native.map { $0 as KVCache }, positionIds: positions)
                actual = model.embeddingForward(
                    ids, inputEmbedding: embedding,
                    cache: paged.map { $0 as KVCache }, positionIds: positions)
            } else {
                expected = model.cbv2Forward(
                    ids, cache: native.map { $0 as KVCache }, positionIds: positions)
                actual = model.cbv2Forward(
                    ids, cache: paged.map { $0 as KVCache }, positionIds: positions)
            }
            eval([expected, actual] + paged.flatMap { $0.innerState() })
            #expect(abs(expected - actual).max().item(Float.self) < 0.0003)
            #expect(!backend.pool.writeValidation.isFaulted)
            StreamOrDevice.default.stream.synchronize()
            try backend.pool.finishQuantizedStorageStep()
            start += count
        }
        #expect(model.cbv2Capabilities.supportsPagedKV)
        #expect(model.supportsCausalVisionPrefill)
        #expect(!model.supportsVisionSpanPrefill)
        #expect(!model.cbv2Capabilities.supportsPrefixReuse)
        #expect(!model.cbv2Capabilities.supportsCompiledDecode)
        #expect(!model.cbv2Capabilities.supportsPackedPrefill)
        #expect(!model.cbv2Capabilities.supportsMTP)
    }
}
