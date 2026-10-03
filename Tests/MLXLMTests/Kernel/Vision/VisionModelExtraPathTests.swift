import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

extension KernelTests {

    /// Model paths of Mistral3, Pixtral, LFM2VL, Gemma3, Idefics3 and
    /// FastVLM that the forward-pass tests with one image do not run:
    /// text-only prefill, more than one image, configuration variants,
    /// `sanitize(weights:)` key rules, and the output of each processor as
    /// the model input.
    ///
    /// The models are tiny (hidden size 32, 1 or 2 layers) with seeded
    /// random weights.
    @Suite
    struct VisionModelExtraPathTests {

        typealias ScriptTokenizer = VisionProcessorTests.ScriptTokenizer

        // Tolerance of the float32 comparisons between a chunked prefill
        // with a cache and one full forward pass: the paths differ in the
        // order of the attention sums; differences are near 1e-6 for logits
        // of size 1 to 5.
        static let tolerance: Float = 1e-4

        // MARK: - Helpers

        static func merged(_ base: [String: Any], _ overrides: [String: Any]) -> [String: Any] {
            base.merging(overrides) { _, new in new }
        }

        /// Builds a model from a configuration dictionary and randomizes it.
        static func build<C: Decodable, M: Module>(
            _ type: C.Type, _ configuration: [String: Any], seed: UInt64 = 1,
            _ make: (C) -> M
        ) throws -> M {
            let model = make(try SyntheticModel.configuration(type, configuration))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func random(_ shape: [Int], seed: UInt64) -> MLXArray {
            let x = MLXRandom.normal(shape, key: MLXRandom.key(seed))
            eval(x)
            return x
        }

        /// Runs `prepare` with a new cache and returns the logits.
        static func run(_ model: any LanguageModel, _ input: LMInput, windowSize: Int? = nil)
            throws -> MLXArray
        {
            let cache = model.newCache(parameters: nil)
            guard
                case .logits(let output) = try model.prepare(
                    input, cache: cache, windowSize: windowSize)
            else {
                Issue.record("prepare must return logits")
                return MLXArray.zeros([1])
            }
            eval(output.logits)
            return output.logits
        }

        /// Runs `prepare` on `prompt` with optional pixels.
        static func prefill(
            _ model: any LanguageModel, _ prompt: [Int], pixels: MLXArray? = nil,
            frames: [THW]? = nil, mask: MLXArray? = nil, windowSize: Int? = nil
        ) throws -> MLXArray {
            let tokens = SyntheticModel.batch([prompt])
            let input = LMInput(
                text: .init(
                    tokens: tokens, mask: mask ?? MLXArray.ones(tokens.shape, dtype: .int32)),
                image: pixels.map { LMInput.ProcessedImage(pixels: $0, frames: frames) })
            return try run(model, input, windowSize: windowSize)
        }

        /// One forward pass over `prompt` with a new cache.
        static func full(_ model: any LanguageModel, _ prompt: [Int]) -> MLXArray {
            let output = model(
                SyntheticModel.batch([prompt]), cache: model.newCache(parameters: nil))
            eval(output)
            return output
        }

        static func isAllFinite(_ x: MLXArray) -> Bool {
            isFinite(x).all().item(Bool.self)
        }

        static func keys(_ model: Module) -> Set<String> {
            Set(SyntheticModel.flatParameters(model).keys)
        }

        /// Checks that a chunked text-only prefill gives the logits of a
        /// full forward pass at the last position.
        static func checkTextOnlyPrefill(
            _ model: any LanguageModel, _ name: String, windowSize: Int = 2,
            prompt: [Int] = [5, 7, 9, 11, 13, 15],
            sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            let chunked = try prefill(model, prompt, windowSize: windowSize)
            let reference = full(model, prompt)
            let difference = SyntheticModel.maxAbsDifference(
                chunked[0..., -1], reference[0..., -1])
            #expect(
                difference <= tolerance, "\(name): text-only prefill differs by \(difference)",
                sourceLocation: sourceLocation)
        }

        /// Runs `processor` on a white and on a black `width` x `height`
        /// image, and `model` on each output. Returns the two logits and the
        /// processor output of the white image.
        static func endToEnd(
            _ processor: any UserInputProcessor, _ model: any LanguageModel, width: Int,
            height: Int, prompt: String = "hi"
        ) async throws -> (white: MLXArray, black: MLXArray, input: LMInput) {
            func image(_ color: CIColor) -> UserInput.Image {
                .ciImage(
                    CIImage(color: color).cropped(
                        to: CGRect(x: 0, y: 0, width: width, height: height)))
            }
            let white = try await processor.prepare(
                input: UserInput(prompt: prompt, images: [image(.white)]))
            let black = try await processor.prepare(
                input: UserInput(prompt: prompt, images: [image(.black)]))
            return try (run(model, white), run(model, black), white)
        }

        static func checkEndToEnd(
            _ name: String, _ result: (white: MLXArray, black: MLXArray, input: LMInput),
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            #expect(
                isAllFinite(result.white), "\(name): processor output gives finite logits",
                sourceLocation: sourceLocation)
            #expect(
                SyntheticModel.maxAbsDifference(result.white[0..., -1], result.black[0..., -1])
                    > 1e-3, "\(name): the processed image must change the logits",
                sourceLocation: sourceLocation)
        }

        // MARK: - Configurations (from the forward-pass tests)

        static var pixtralVision: [String: Any] {
            [
                "model_type": "pixtral", "hidden_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 4, "intermediate_size": 48, "patch_size": 2,
                "image_size": 8, "head_dim": 8,
            ]
        }

        static func mistral3(text: [String: Any] = [:], top: [String: Any] = [:])
            -> [String: Any]
        {
            let textConfig = merged(
                [
                    "model_type": "ministral3", "hidden_size": 32, "num_hidden_layers": 2,
                    "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": 64,
                    "rope_parameters": ["rope_type": "default", "rope_theta": 10000]
                        as [String: Any],
                ], text)
            return merged(
                [
                    "model_type": "mistral3", "image_token_index": 60, "vocab_size": 64,
                    "spatial_merge_size": 2, "text_config": textConfig,
                    "vision_config": pixtralVision,
                ], top)
        }

        static func pixtral(text: [String: Any] = [:], top: [String: Any] = [:]) -> [String: Any] {
            let textConfig = merged(
                [
                    "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                    "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": 64, "rope_theta": 10000,
                ], text)
            return merged(
                [
                    "model_type": "pixtral", "image_token_index": 60, "vocab_size": 64,
                    "text_config": textConfig, "vision_config": pixtralVision,
                ], top)
        }

        static func lfm2vl(
            text: [String: Any] = [:], vision: [String: Any] = [:], top: [String: Any] = [:]
        ) -> [String: Any] {
            let textConfig = merged(
                [
                    "model_type": "lfm2", "hidden_size": 32, "num_hidden_layers": 2,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64,
                    "block_ff_dim": 48, "block_auto_adjust_ff_dim": false,
                    "layer_types": ["conv", "full_attention"],
                ], text)
            let visionConfig = merged(
                [
                    "model_type": "siglip2_vision_model", "hidden_size": 32,
                    "intermediate_size": 48, "num_hidden_layers": 1, "num_attention_heads": 4,
                    "patch_size": 2, "num_patches": 16,
                ], vision)
            return merged(
                [
                    "model_type": "lfm2_vl", "downsample_factor": 2, "image_token_id": 60,
                    "projector_hidden_size": 32, "text_config": textConfig,
                    "vision_config": visionConfig,
                ], top)
        }

        static func idefics3(
            text: [String: Any] = [:], vision: [String: Any] = [:], top: [String: Any] = [:]
        ) -> [String: Any] {
            let textConfig = merged(
                [
                    "model_type": "llama", "hidden_size": 32, "intermediate_size": 48,
                    "num_attention_heads": 4, "rms_norm_eps": 1e-6, "vocab_size": 64,
                    "num_key_value_heads": 2, "rope_theta": 10000, "num_hidden_layers": 2,
                ], text)
            let visionConfig = merged(
                [
                    "model_type": "idefics3_vision", "hidden_size": 32, "num_attention_heads": 4,
                    "patch_size": 2, "image_size": 8, "num_hidden_layers": 1,
                    "intermediate_size": 48,
                ], vision)
            return merged(
                [
                    "model_type": "idefics3", "vocab_size": 64, "scale_factor": 2,
                    "image_token_id": 60, "text_config": textConfig,
                    "vision_config": visionConfig,
                ], top)
        }

        static func gemma3(text: [String: Any] = [:], top: [String: Any] = [:]) -> [String: Any] {
            let textConfig = merged(
                [
                    "model_type": "gemma3_text", "hidden_size": 32, "num_hidden_layers": 6,
                    "intermediate_size": 48, "sliding_window": 4, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 16, "query_pre_attn_scalar": 16,
                ], text)
            return merged(
                [
                    "model_type": "gemma3", "mm_tokens_per_image": 4, "text_config": textConfig,
                    "vision_config": [
                        "model_type": "siglip_vision_model", "num_hidden_layers": 1,
                        "hidden_size": 32, "intermediate_size": 48, "num_attention_heads": 4,
                        "patch_size": 2, "image_size": 8,
                    ] as [String: Any],
                ], top)
        }

        static func fastVLM(vision: [String: Any] = [:], top: [String: Any] = [:])
            -> [String: Any]
        {
            let visionConfig = merged(
                [
                    "cls_ratio": 2.0, "down_patch_size": 3, "down_stride": 2,
                    "downsamples": [true, true], "embed_dims": [16, 32], "hidden_size": 32,
                    "image_size": 32, "intermediate_size": 64, "layers": [1, 1],
                    "layer_scale_init_value": 1e-5, "mlp_ratios": [2, 2], "num_classes": 0,
                    "patch_size": 16, "pos_embs_shapes": [NSNull(), NSNull()],
                    "projection_dim": 32, "repmixer_kernel_size": 3,
                    "token_mixers": ["repmixer", "repmixer"],
                ], vision)
            return merged(
                [
                    "model_type": "fastvlm", "hidden_size": 32, "num_hidden_layers": 2,
                    "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "vocab_size": 64, "eos_token_id": 1, "mm_projector_type": "mlp2x_gelu",
                    "mm_hidden_size": 64, "tokenizer_model_max_length": 256,
                    "tokenizer_padding_side": "right", "image_token_index": -200,
                    "vision_config": visionConfig,
                ], top)
        }

        /// Builds FastVLM. BatchNorm must use its running statistics, and a
        /// running variance must be positive.
        static func buildFastVLM(_ configuration: [String: Any], seed: UInt64 = 1) throws
            -> FastVLM
        {
            let model = try build(
                FastVLMConfiguration.self, configuration, seed: seed, FastVLM.init)
            model.train(false)
            var variances: [(String, MLXArray)] = []
            for (key, value) in SyntheticModel.flatParameters(model)
            where key.hasSuffix("running_var") {
                variances.append((key, abs(value) + 1))
            }
            model.update(parameters: ModuleParameters.unflattened(variances))
            eval(model)
            return model
        }

        // MARK: - Mistral3

        @Test func mistral3TextOnlyPrefillMatchesAFullPass() throws {
            let model = try Self.build(
                Mistral3VLMConfiguration.self, Self.mistral3(), Mistral3VLM.init)
            try Self.checkTextOnlyPrefill(model, "Mistral3")
        }

        /// Without frames, the model uses the configured image size (8 x 8),
        /// so the result is the same as with the frame 8 x 8.
        @Test func mistral3UsesTheConfiguredImageSizeWithoutFrames() throws {
            let model = try Self.build(
                Mistral3VLMConfiguration.self, Self.mistral3(), Mistral3VLM.init)
            let prompt = [5, 7, 60, 60, 60, 60, 9, 11, 13]
            let pixels = Self.random([1, 3, 8, 8], seed: 1)
            let withFrames = try Self.prefill(model, prompt, pixels: pixels, frames: [THW(1, 8, 8)])
            let withoutFrames = try Self.prefill(model, prompt, pixels: pixels)
            #expect(
                SyntheticModel.maxAbsDifference(withFrames, withoutFrames) == 0,
                "Mistral3: frames nil uses the configured image size")
        }

        /// Sliding attention layers, Llama 4 attention scaling, tied
        /// embeddings, projector bias, `image_token_id` and a non-negative
        /// vision feature layer.
        @Test func mistral3VariantConfiguration() throws {
            var configuration = Self.mistral3(
                text: [
                    "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 16,
                    "tie_word_embeddings": true,
                    "rope_parameters": [
                        "rope_type": "default", "rope_theta": 10000,
                        "llama_4_scaling_beta": 0.1, "original_max_position_embeddings": 4,
                    ] as [String: Any],
                ],
                top: [
                    "multimodal_projector_bias": true, "vision_feature_layer": 0,
                    "image_token_id": 60,
                ])
            configuration.removeValue(forKey: "image_token_index")
            let model = try Self.build(
                Mistral3VLMConfiguration.self, configuration, Mistral3VLM.init)

            #expect(model.config.imageTokenIndex == 60, "Mistral3: image_token_id is read")
            let keys = Self.keys(model)
            #expect(
                !keys.contains("language_model.lm_head.weight"), "Mistral3: tied, no lm_head")
            #expect(
                keys.contains("multi_modal_projector.linear_1.bias"), "Mistral3: projector bias")
            #expect(model.loraLayers.count == 2, "Mistral3: one LoRA layer per text layer")

            let cache = model.newCache(parameters: nil)
            #expect(cache.count == 2, "Mistral3: one cache per layer")
            #expect(cache.first is RotatingKVCache, "Mistral3: sliding layer cache")
            #expect(cache.last is KVCacheSimple, "Mistral3: full attention layer cache")
            let limited = model.newCache(parameters: GenerateParameters(maxKVSize: 8))
            #expect(limited.last is RotatingKVCache, "Mistral3: maxKVSize gives a rotating cache")

            let logits = try Self.prefill(
                model, [5, 7, 60, 60, 60, 60, 9, 11, 13],
                pixels: Self.random([1, 3, 8, 8], seed: 1),
                frames: [THW(1, 8, 8)])
            #expect(logits.dim(-1) == 64, "Mistral3: variant logits vocabulary")
            #expect(Self.isAllFinite(logits), "Mistral3: variant logits are finite")
        }

        @Test func mistral3SanitizeRenamesKeysAndAppliesWeightScales() throws {
            let model = Mistral3VLM(
                try SyntheticModel.configuration(Mistral3VLMConfiguration.self, Self.mistral3()))
            let a = MLXArray.zeros([2])
            let layer = "language_model.model.layers.1.self_attn"
            let weights: [String: MLXArray] = [
                "vision_tower.transformer.layers.0.attention.q_proj.weight": a,
                "vision_tower.patch_conv.weight": a,
                "vision_tower.extra.weight": a,
                "model.vision_encoder.ln_pre.weight": a,
                "model.vision_encoder.extra.weight": a,
                "model.language_model.layers.0.mlp.up_proj.weight": a,
                "lm_head.weight": a,
                "model.vision_projection.linear_1.weight": a,
                "model.language_model.layers.0.self_attn.rotary_emb.inv_freq": a,
                "\(layer).q_proj.weight": MLXArray.ones([2, 2]),
                "\(layer).q_proj.weight_scale_inv": MLXArray([3, 3, 3, 3] as [Float]).reshaped(
                    2, 2),
                "\(layer).k_proj.activation_scale": a,
            ]
            let sanitized = model.sanitize(weights: weights)
            #expect(
                Set(sanitized.keys) == [
                    "vision_tower.vision_model.transformer.layers.0.attention.q_proj.weight",
                    "vision_tower.vision_model.patch_conv.weight",
                    "vision_tower.extra.weight",
                    "vision_tower.vision_model.ln_pre.weight",
                    "model.vision_encoder.extra.weight",
                    "language_model.model.layers.0.mlp.up_proj.weight",
                    "language_model.lm_head.weight",
                    "multi_modal_projector.linear_1.weight",
                    "\(layer).q_proj.weight",
                ], "Mistral3: sanitized keys")
            // 1 * 3 is exact in float32.
            #expect(
                sanitized["\(layer).q_proj.weight"]?.asArray(Float.self) == [3, 3, 3, 3],
                "Mistral3: weight times weight_scale_inv")
        }

        // MARK: - Pixtral

        @Test func pixtralTextOnlyPrefillMatchesAFullPass() throws {
            let model = try Self.build(PixtralConfiguration.self, Self.pixtral(), PixtralVLM.init)
            try Self.checkTextOnlyPrefill(model, "Pixtral")
        }

        /// QK norm, linear RoPE scaling, tied embeddings, `image_token_id`
        /// and a non-negative vision feature layer.
        @Test func pixtralVariantConfiguration() throws {
            var configuration = Self.pixtral(
                text: [
                    "use_qk_norm": true, "tie_word_embeddings": true,
                    "rope_scaling": ["type": "linear", "factor": 2.0] as [String: Any],
                ],
                top: ["vision_feature_layer": 0, "image_token_id": 60])
            configuration.removeValue(forKey: "image_token_index")
            let model = try Self.build(PixtralConfiguration.self, configuration, PixtralVLM.init)

            #expect(model.config.imageTokenIndex == 60, "Pixtral: image_token_id is read")
            let keys = Self.keys(model)
            #expect(
                keys.contains("language_model.model.layers.0.self_attn.q_norm.weight"),
                "Pixtral: QK norm")
            #expect(!keys.contains("language_model.lm_head.weight"), "Pixtral: tied, no lm_head")
            #expect(model.loraLayers.count == 2, "Pixtral: one LoRA layer per text layer")
            let limited = model.newCache(parameters: GenerateParameters(maxKVSize: 8))
            #expect(
                limited.allSatisfy { $0 is RotatingKVCache },
                "Pixtral: maxKVSize gives rotating caches")

            let logits = try Self.prefill(
                model, [5, 7, 60, 60, 60, 60, 9, 11, 13],
                pixels: Self.random([1, 3, 4, 4], seed: 1))
            #expect(logits.dim(-1) == 64, "Pixtral: variant logits vocabulary")
            #expect(Self.isAllFinite(logits), "Pixtral: variant logits are finite")
        }

        @Test func pixtralVisionHelpers() throws {
            #expect(
                PixtralVision.checkArrayShape(MLXArray.zeros([32, 2, 2, 3])),
                "Pixtral: MLX convolution layout")
            #expect(
                !PixtralVision.checkArrayShape(MLXArray.zeros([32, 3, 2, 2])),
                "Pixtral: PyTorch convolution layout")
            #expect(!PixtralVision.checkArrayShape(MLXArray.zeros([2, 2])), "Pixtral: not 4-D")

            #expect(
                PixtralVision.positionIdsInMeshgrid(patchHeight: 2, patchWidth: 3, maxWidth: 4)
                    .asArray(Int32.self) == [0, 1, 2, 4, 5, 6], "Pixtral: meshgrid position IDs")

            let mask = PixtralVision.generateBlockAttentionMask(
                patchCounts: [2, 1], batchSize: 2, dtype: .float32)
            #expect(mask.shape == [2, 1, 3, 3], "Pixtral: block mask shape")
            let n: Float = -1e9
            #expect(
                mask[1, 0].asArray(Float.self) == [0, 0, n, 0, 0, n, n, n, 0],
                "Pixtral: block mask values")

            // cos 1 and sin 0 keep q and k; cos 0 and sin 1 give
            // rotate_half(q) = [-q2, q1]. Both are exact.
            let q = MLXArray([1, 2, 3, 4] as [Float]).reshaped(1, 1, 1, 4)
            let k = MLXArray([5, 6, 7, 8] as [Float]).reshaped(1, 1, 1, 4)
            let (q0, k0) = PixtralVision.applyRotaryPosEmb(
                q: q, k: k, cos: MLXArray.ones([1, 4]), sin: MLXArray.zeros([1, 4]))
            #expect(SyntheticModel.maxAbsDifference(q0, q) == 0, "Pixtral: identity rotation q")
            #expect(SyntheticModel.maxAbsDifference(k0, k) == 0, "Pixtral: identity rotation k")
            let (q1, _) = PixtralVision.applyRotaryPosEmb(
                q: q, k: k, cos: MLXArray.zeros([1, 4]), sin: MLXArray.ones([1, 4]))
            #expect(
                q1.asArray(Float.self) == [-3, -4, 1, 2], "Pixtral: quarter-turn rotation of q")

            let vision = try SyntheticModel.configuration(
                PixtralVisionConfiguration.self, Self.pixtralVision)
            let rotary = PixtralVision.RotaryEmbedding(vision)
            // (8 / 2)^2 = 16 positions, head dimension 8.
            #expect(rotary.invFreq.shape == [16, 8], "Pixtral: inverse frequency shape")
            #expect(
                SyntheticModel.maxAbs(rotary.invFreq[0]) == 0, "Pixtral: position 0 has angle 0")
            let (cos, sin) = rotary(MLXArray.zeros([1, 2, 8]), positionIds: MLXArray([0, 5]))
            #expect(cos.shape == [2, 8] && sin.shape == [2, 8], "Pixtral: cos and sin shapes")
            #expect(cos[0].asArray(Float.self) == Array(repeating: 1, count: 8), "Pixtral: cos(0)")

            let sanitized = PixtralVision.VisionModel(vision).sanitize(weights: [
                "vision_model.position_ids": MLXArray.zeros([4]),
                "vision_model.patch_conv.weight": MLXArray.zeros([32, 3, 2, 2]),
                "other.patch_embedding.weight": MLXArray.zeros([32, 2, 2, 3]),
                "vision_model.ln_pre.weight": MLXArray.zeros([32]),
            ])
            #expect(
                sanitized["vision_model.position_ids"] == nil, "Pixtral: position_ids dropped")
            #expect(
                sanitized["vision_model.patch_conv.weight"]?.shape == [32, 2, 2, 3],
                "Pixtral: PyTorch convolution transposed")
            #expect(
                sanitized["other.patch_embedding.weight"]?.shape == [32, 2, 2, 3],
                "Pixtral: MLX convolution kept")
            #expect(sanitized["vision_model.ln_pre.weight"] != nil, "Pixtral: other keys kept")
        }

        @Test func pixtralSanitizeRenamesKeys() throws {
            let model = PixtralVLM(
                try SyntheticModel.configuration(PixtralConfiguration.self, Self.pixtral()))
            let a = MLXArray.zeros([2])
            let sanitized = model.sanitize(weights: [
                "vision_tower.ln_pre.weight": a,
                "vision_tower.extra.weight": a,
                "model.vision_encoder.transformer.layers.0.feed_forward.up_proj.weight": a,
                "model.vision_encoder.extra.weight": a,
                "model.language_model.norm.weight": a,
                "lm_head.weight": a,
                "model.vision_projection.linear_2.bias": a,
                "model.language_model.layers.1.self_attn.rotary_emb.inv_freq": a,
            ])
            #expect(
                Set(sanitized.keys) == [
                    "vision_tower.vision_model.ln_pre.weight",
                    "vision_tower.extra.weight",
                    "vision_tower.vision_model.transformer.layers.0.feed_forward.up_proj.weight",
                    "model.vision_encoder.extra.weight",
                    "language_model.model.norm.weight",
                    "language_model.lm_head.weight",
                    "multi_modal_projector.linear_2.bias",
                ], "Pixtral: sanitized keys")
        }

        // MARK: - LFM2VL

        /// Without frames, the model takes a square grid from the number of
        /// patches (16 gives 4 x 4), the same as the frame 4 x 4.
        @Test func lfm2vlInfersASquareGridWithoutFrames() throws {
            let model = try Self.build(LFM2VLConfiguration.self, Self.lfm2vl(), LFM2VL.init)
            let prompt = [5, 7, 60, 60, 60, 60, 9, 11, 13]
            let pixels = Self.random([1, 16, 12], seed: 1)
            let withFrames = try Self.prefill(model, prompt, pixels: pixels, frames: [THW(1, 4, 4)])
            let withoutFrames = try Self.prefill(model, prompt, pixels: pixels)
            #expect(
                SyntheticModel.maxAbsDifference(withFrames, withoutFrames) == 0,
                "LFM2VL: frames nil infers a square grid")
        }

        /// Two images with 4 x 4 and 2 x 2 patches: the mask of the second
        /// image is padded, and the images give 4 + 1 image tokens.
        @Test func lfm2vlRunsTwoImagesOfDifferentSizes() throws {
            let model = try Self.build(LFM2VLConfiguration.self, Self.lfm2vl(), LFM2VL.init)
            let prompt = [5, 7, 60, 60, 60, 60, 9, 60, 11]
            let frames = [THW(1, 4, 4), THW(1, 2, 2)]
            let first = Self.random([1, 16, 12], seed: 1)
            let a = try Self.prefill(
                model, prompt,
                pixels: concatenated([first, Self.random([1, 16, 12], seed: 2)], axis: 0),
                frames: frames)
            let b = try Self.prefill(
                model, prompt,
                pixels: concatenated([first, Self.random([1, 16, 12], seed: 3)], axis: 0),
                frames: frames)
            #expect(Self.isAllFinite(a), "LFM2VL: two-image logits are finite")
            #expect(
                SyntheticModel.maxAbsDifference(a[0..., -1], b[0..., -1]) > 1e-3,
                "LFM2VL: the second image must change the logits")
        }

        /// A 3 x 3 patch grid is padded to 4 x 4 before the 2 x 2 pixel
        /// unshuffle, which gives 4 image tokens.
        @Test func lfm2vlPadsAnOddGridBeforeThePixelUnshuffle() throws {
            let model = try Self.build(LFM2VLConfiguration.self, Self.lfm2vl(), LFM2VL.init)
            let logits = try Self.prefill(
                model, [5, 7, 60, 60, 60, 60, 9], pixels: Self.random([1, 9, 12], seed: 1),
                frames: [THW(1, 3, 3)])
            #expect(logits.dim(-1) == 64, "LFM2VL: odd grid logits vocabulary")
            #expect(Self.isAllFinite(logits), "LFM2VL: odd grid logits are finite")
        }

        @Test func lfm2vlVisionFeatureLayerSetsTheEncoderDepth() throws {
            func depth(_ featureLayer: Int) throws -> Int {
                let model = LFM2VL(
                    try SyntheticModel.configuration(
                        LFM2VLConfiguration.self,
                        Self.lfm2vl(
                            vision: ["num_hidden_layers": 3],
                            top: ["vision_feature_layer": featureLayer])))
                let keys = Self.keys(model)
                return (0 ..< 4).filter {
                    keys.contains("vision_tower.encoder.layers.\($0).layer_norm1.weight")
                }.count
            }
            #expect(try depth(-1) == 3, "LFM2VL: -1 uses all layers")
            #expect(try depth(-2) == 2, "LFM2VL: -2 drops the last layer")
            #expect(try depth(0) == 1, "LFM2VL: 0 keeps one layer")
            #expect(try depth(5) == 3, "LFM2VL: an out-of-range layer uses all layers")
        }

        /// No pixel unshuffle, a projector without layer norm and bias, the
        /// automatic feed-forward size, and `full_attn_idxs`.
        @Test func lfm2vlVariantConfiguration() throws {
            let model = try Self.build(
                LFM2VLConfiguration.self,
                Self.lfm2vl(
                    text: [
                        "layer_types": NSNull(), "full_attn_idxs": [1],
                        "block_auto_adjust_ff_dim": true, "block_ffn_dim_multiplier": 1.5,
                        "block_multiple_of": 20,
                    ],
                    top: [
                        "downsample_factor": 1, "projector_use_layernorm": false,
                        "projector_bias": false,
                    ]), LFM2VL.init)
            let parameters = SyntheticModel.flatParameters(model)
            #expect(
                parameters["multi_modal_projector.layer_norm.weight"] == nil,
                "LFM2VL: no projector layer norm")
            #expect(
                parameters["multi_modal_projector.linear_1.bias"] == nil,
                "LFM2VL: no projector bias")
            #expect(
                parameters["multi_modal_projector.linear_1.weight"]?.shape == [32, 32],
                "LFM2VL: projector input without unshuffle")
            // 2 * 48 / 3 = 32, 1.5 * 32 = 48, rounded up to a multiple of 20: 60.
            #expect(
                parameters["language_model.model.layers.0.feed_forward.w1.weight"]?.shape
                    == [60, 32], "LFM2VL: automatic feed-forward size")
            #expect(
                parameters["language_model.model.layers.0.conv.conv.weight"] != nil,
                "LFM2VL: layer 0 is a convolution layer")
            #expect(
                parameters["language_model.model.layers.1.self_attn.q_proj.weight"] != nil,
                "LFM2VL: layer 1 is an attention layer")
            #expect(model.kvHeads == [0, 2], "LFM2VL: KV heads per layer")
            #expect(model.loraLayers.count == 2, "LFM2VL: one LoRA layer per text layer")
            let cache = model.newCache(parameters: nil)
            #expect(cache.first is MambaCache, "LFM2VL: convolution layer cache")
            #expect(cache.last is KVCacheSimple, "LFM2VL: attention layer cache")

            // Without the unshuffle, 2 x 2 patches give 4 image tokens.
            let logits = try Self.prefill(
                model, [5, 7, 60, 60, 60, 60, 9], pixels: Self.random([1, 4, 12], seed: 1),
                frames: [THW(1, 2, 2)])
            #expect(Self.isAllFinite(logits), "LFM2VL: variant logits are finite")
        }

        @Test func lfm2vlSanitizeRenamesKeysAndTransposesConvolutions() throws {
            let model = LFM2VL(
                try SyntheticModel.configuration(LFM2VLConfiguration.self, Self.lfm2vl()))
            let a = MLXArray.zeros([2])
            let sanitized = model.sanitize(weights: [
                "model.vision_tower.vision_encoder.layers.0.mlp.fc1.weight": a,
                "model.vision_tower.vision_embeddings.patch_embedding.weight": a,
                "model.vision_tower.vision_post_layernorm.weight": a,
                "model.multi_modal_projector.linear_1.weight": a,
                "model.language_model.layers.0.conv.conv.weight": MLXArray.zeros([32, 1, 3]),
                "language_model.model.layers.2.conv.conv.weight": MLXArray.zeros([32, 3, 1]),
            ])
            #expect(
                Set(sanitized.keys) == [
                    "vision_tower.encoder.layers.0.mlp.fc1.weight",
                    "vision_tower.embeddings.patch_embedding.weight",
                    "vision_tower.post_layernorm.weight",
                    "multi_modal_projector.linear_1.weight",
                    "language_model.model.layers.0.conv.conv.weight",
                    "language_model.model.layers.2.conv.conv.weight",
                ], "LFM2VL: sanitized keys")
            #expect(
                sanitized["language_model.model.layers.0.conv.conv.weight"]?.shape == [32, 3, 1],
                "LFM2VL: PyTorch convolution transposed")
            #expect(
                sanitized["language_model.model.layers.2.conv.conv.weight"]?.shape == [32, 3, 1],
                "LFM2VL: MLX convolution kept")
        }

        // MARK: - Gemma3

        /// A 4-token prompt fits in the sliding window of 4, so the rotating
        /// caches do not drop entries.
        @Test func gemma3TextOnlyPrefillMatchesAFullPass() throws {
            let model = try Self.build(Gemma3Configuration.self, Self.gemma3(), Gemma3.init)
            try Self.checkTextOnlyPrefill(model, "Gemma3", windowSize: 1, prompt: [5, 7, 9, 11])
            #expect(model.loraLayers.count == 6, "Gemma3: one LoRA layer per text layer")
        }

        @Test func gemma3SoftcapsTheFinalLogits() throws {
            let raw = try Self.build(Gemma3Configuration.self, Self.gemma3(), Gemma3.init)
            let capped = try Self.build(
                Gemma3Configuration.self,
                Self.gemma3(text: ["final_logit_softcapping": 0.5]), Gemma3.init)
            let prompt = [5, 7, 9]
            let a = Self.full(raw, prompt)
            let b = Self.full(capped, prompt)
            #expect(SyntheticModel.maxAbs(b) <= 0.5, "Gemma3: capped logits are at most 0.5")
            // The same seed gives the same weights. Tolerance 1e-5: float32
            // tanh of the same logits.
            let cap: Float = 0.5
            let expected = MLX.tanh(a / cap) * cap
            let difference = SyntheticModel.maxAbsDifference(b, expected)
            #expect(difference <= 1e-5, "Gemma3: softcap differs by \(difference)")
        }

        @Test func gemma3RunsTwoImages() throws {
            let model = try Self.build(Gemma3Configuration.self, Self.gemma3(), Gemma3.init)
            let i = 262_144
            let prompt = [5, 7, i, i, i, i, 9, i, i, i, i, 11]
            let first = Self.random([1, 3, 8, 8], seed: 1)
            let a = try Self.prefill(
                model, prompt,
                pixels: concatenated([first, Self.random([1, 3, 8, 8], seed: 2)], axis: 0))
            let b = try Self.prefill(
                model, prompt,
                pixels: concatenated([first, Self.random([1, 3, 8, 8], seed: 3)], axis: 0))
            #expect(Self.isAllFinite(a), "Gemma3: two-image logits are finite")
            #expect(
                SyntheticModel.maxAbsDifference(a[0..., -1], b[0..., -1]) > 1e-3,
                "Gemma3: the second image must change the logits")
        }

        @Test func gemma3SanitizeTiesTransposesAndFilters() throws {
            let model = Gemma3(
                try SyntheticModel.configuration(Gemma3Configuration.self, Self.gemma3()))
            let embed = MLXArray([1, 2, 3, 4] as [Float]).reshaped(2, 2)
            let u = MLXArray.zeros([1], dtype: .uint32)
            let s = MLXArray.zeros([1])
            let sanitized = model.sanitize(weights: [
                "language_model.model.embed_tokens.weight": embed,
                "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": s,
                "vision_tower.vision_model.embeddings.patch_embedding.weight":
                    MLXArray.zeros([32, 3, 2, 2]),
                "other.patch_embedding.weight": MLXArray.zeros([32, 2, 2, 3]),
                "vision_tower.q": s, "vision_tower.q.scales": s, "vision_tower.q.biases": s,
                "vision_tower.q.weight": u,
            ])
            #expect(
                sanitized["language_model.lm_head.weight"]?.asArray(Float.self) == [1, 2, 3, 4],
                "Gemma3: lm_head tied to the embeddings")
            #expect(
                sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"] == nil,
                "Gemma3: rotary frequencies dropped")
            #expect(
                sanitized["vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape
                    == [32, 2, 2, 3], "Gemma3: PyTorch convolution transposed")
            #expect(
                sanitized["other.patch_embedding.weight"]?.shape == [32, 2, 2, 3],
                "Gemma3: MLX convolution kept")
            #expect(sanitized["vision_tower.q"] != nil, "Gemma3: quantized vision key kept")

            let own = MLXArray([5, 6, 7, 8] as [Float]).reshaped(2, 2)
            let explicit = model.sanitize(weights: [
                "language_model.model.embed_tokens.weight": embed,
                "language_model.lm_head.weight": own,
            ])
            #expect(
                explicit["language_model.lm_head.weight"]?.asArray(Float.self) == [5, 6, 7, 8],
                "Gemma3: an lm_head in the checkpoint is kept")
        }

        /// A quantized `lm_head` in the checkpoint makes `sanitize` quantize
        /// the `lm_head` of the model with the configured group size.
        @Test func gemma3SanitizeQuantizesTheLanguageModelHead() throws {
            let model = Gemma3(
                try SyntheticModel.configuration(
                    Gemma3Configuration.self,
                    Self.gemma3(top: [
                        "quantization": ["group_size": 32, "bits": 4] as [String: Any]
                    ])))
            let s = MLXArray.zeros([1])
            let sanitized = model.sanitize(weights: [
                "language_model.lm_head.weight": MLXArray.zeros([1], dtype: .uint32),
                "language_model.lm_head.scales": s, "language_model.lm_head.biases": s,
                "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": s,
            ])
            #expect(
                sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"] == nil,
                "Gemma3: rotary frequencies dropped from a quantized checkpoint")
            let parameters = SyntheticModel.flatParameters(model)
            #expect(
                parameters["language_model.lm_head.scales"] != nil,
                "Gemma3: lm_head is quantized")
            #expect(
                parameters["language_model.lm_head.weight"]?.dtype == .uint32,
                "Gemma3: quantized lm_head weight dtype")
            #expect(
                parameters["language_model.model.embed_tokens.scales"] == nil,
                "Gemma3: embeddings without scales stay unquantized")
        }

        // MARK: - Idefics3

        @Test func idefics3TextOnlyPrefillMatchesAFullPass() throws {
            let model = try Self.build(Idefics3Configuration.self, Self.idefics3(), Idefics3.init)
            try Self.checkTextOnlyPrefill(model, "Idefics3")
            #expect(model.loraLayers.count == 2, "Idefics3: one LoRA layer per text layer")
        }

        @Test func idefics3RunsTwoImages() throws {
            let model = try Self.build(Idefics3Configuration.self, Self.idefics3(), Idefics3.init)
            let prompt = [5, 60, 60, 60, 60, 7, 60, 60, 60, 60, 9]
            let first = Self.random([1, 8, 8, 3], seed: 1)
            let a = try Self.prefill(
                model, prompt,
                pixels: concatenated([first, Self.random([1, 8, 8, 3], seed: 2)], axis: 0))
            let b = try Self.prefill(
                model, prompt,
                pixels: concatenated([first, Self.random([1, 8, 8, 3], seed: 3)], axis: 0))
            #expect(Self.isAllFinite(a), "Idefics3: two-image logits are finite")
            #expect(
                SyntheticModel.maxAbsDifference(a[0..., -1], b[0..., -1]) > 1e-3,
                "Idefics3: the second image must change the logits")
        }

        @Test func idefics3TiedEmbeddings() throws {
            let model = try Self.build(
                Idefics3Configuration.self, Self.idefics3(text: ["tie_word_embeddings": true]),
                Idefics3.init)
            #expect(
                !Self.keys(model).contains("language_model.lm_head.weight"),
                "Idefics3: tied, no lm_head")
            let logits = Self.full(model, [5, 7, 9])
            #expect(logits.shape == [1, 3, 64], "Idefics3: tied logits shape")
            #expect(Self.isAllFinite(logits), "Idefics3: tied logits are finite")
        }

        @Test func idefics3SanitizeRenamesKeys() throws {
            let model = Idefics3(
                try SyntheticModel.configuration(Idefics3Configuration.self, Self.idefics3()))
            let a = MLXArray.zeros([2])
            let sanitized = model.sanitize(weights: [
                "model.text_model.layers.0.mlp.up_proj.weight": a,
                "model.text_model.layers.0.self_attn.rotary_emb.inv_freq": a,
                "model.vision_model.post_layernorm.weight": a,
                "model.connector.modality_projection.proj.weight": a,
                "lm_head.weight": a,
                "other.weight": a,
            ])
            #expect(
                Set(sanitized.keys) == [
                    "language_model.layers.0.mlp.up_proj.weight",
                    "vision_model.post_layernorm.weight",
                    "connector.modality_projection.proj.weight",
                    "language_model.lm_head.weight",
                    "other.weight",
                ], "Idefics3: sanitized keys")
        }

        // MARK: - FastVLM

        /// The model takes the tokens between the first and the last 1 of
        /// the mask, and puts the image at token -200, or first when there is
        /// no -200.
        @Test func fastVLMMaskAndImagePosition() throws {
            let model = try Self.buildFastVLM(Self.fastVLM())
            let pixels = Self.random([1, 3, 32, 32], seed: 1)
            let prompt = [5, -200, 9, 11]
            let reference = try Self.prefill(model, prompt, pixels: pixels)

            let noMask = try Self.run(
                model,
                LMInput(
                    text: .init(tokens: SyntheticModel.batch([prompt]), mask: nil),
                    image: .init(pixels: pixels)))
            #expect(
                SyntheticModel.maxAbsDifference(reference, noMask) == 0,
                "FastVLM: no mask is the same as a full mask")

            let padded = try Self.prefill(
                model, [0, 0] + prompt, pixels: pixels,
                mask: MLXArray([0, 0, 1, 1, 1, 1] as [Int32]).reshaped(1, 6))
            #expect(
                SyntheticModel.maxAbsDifference(reference, padded) == 0,
                "FastVLM: left padding is removed")

            let withoutToken = try Self.prefill(model, [5, 9, 11], pixels: pixels)
            let imageFirst = try Self.prefill(model, [-200, 5, 9, 11], pixels: pixels)
            #expect(
                SyntheticModel.maxAbsDifference(withoutToken, imageFirst) == 0,
                "FastVLM: without -200 the image goes first")
            #expect(model.loraLayers.count == 2, "FastVLM: one LoRA layer per text layer")
        }

        /// Attention token mixer, conditional position encoding and a
        /// 3-layer projector.
        @Test func fastVLMAttentionStageAndDeepProjector() throws {
            let model = try Self.buildFastVLM(
                Self.fastVLM(
                    vision: [
                        "token_mixers": ["repmixer", "attention"],
                        "pos_embs_shapes": [NSNull(), [3, 3]] as [Any],
                    ],
                    top: ["mm_projector_type": "mlp3x_gelu"]))
            let keys = Self.keys(model)
            #expect(
                keys.filter { $0.hasPrefix("mm_projector.") && $0.hasSuffix(".weight") }.count
                    == 3, "FastVLM: 3 projector linear layers")
            #expect(
                keys.contains { $0.hasSuffix("token_mixer.qkv.weight") },
                "FastVLM: attention token mixer")
            #expect(keys.contains { $0.contains("lkb_reparam") }, "FastVLM: downsampling stage")

            let prompt = [5, -200, 9, 11]
            let a = try Self.prefill(model, prompt, pixels: Self.random([1, 3, 32, 32], seed: 1))
            let b = try Self.prefill(model, prompt, pixels: Self.random([1, 3, 32, 32], seed: 2))
            #expect(Self.isAllFinite(a), "FastVLM: attention variant logits are finite")
            #expect(
                SyntheticModel.maxAbsDifference(a[0..., -1], b[0..., -1]) > 1e-3,
                "FastVLM: attention variant, the image must change the logits")
        }

        /// Equal stage widths without downsampling, and a linear projector.
        @Test func fastVLMWithoutDownsamplingAndLinearProjector() throws {
            let model = try Self.buildFastVLM(
                Self.fastVLM(
                    vision: ["embed_dims": [16, 16], "downsamples": [false, false]],
                    top: ["mm_projector_type": "linear", "mm_hidden_size": 32]))
            let keys = Self.keys(model)
            #expect(
                keys.filter { $0.hasPrefix("mm_projector.") && $0.hasSuffix(".weight") }.count
                    == 1, "FastVLM: one projector linear layer")
            #expect(!keys.contains { $0.contains("lkb_reparam") }, "FastVLM: no downsampling stage")
            let logits = try Self.prefill(
                model, [5, -200, 9, 11], pixels: Self.random([1, 3, 32, 32], seed: 1))
            #expect(Self.isAllFinite(logits), "FastVLM: linear projector logits are finite")
        }

        // MARK: - Processor output as model input

        @Test func mistral3ProcessorOutputRunsThroughTheModel() async throws {
            let model = try Self.build(
                Mistral3VLMConfiguration.self,
                Self.mistral3(text: ["vocab_size": 1200], top: ["vocab_size": 1200]),
                Mistral3VLM.init)
            let processor = try VisionProcessorTests.mistral3(
                longestEdge: 8,
                tokenizer: ScriptTokenizer(specials: ["[IMG]": 60], imageMarker: "[IMG]"))
            // 16 x 16 becomes 8 x 8: 4 x 4 patches and 2 x 2 image tokens.
            let result = try await Self.endToEnd(processor, model, width: 16, height: 16)
            #expect(
                VisionProcessorTests.tokens(result.input).filter { $0 == 60 }.count == 4,
                "Mistral3: 4 image tokens")
            Self.checkEndToEnd("Mistral3", result)
        }

        @Test func pixtralProcessorOutputRunsThroughTheModel() async throws {
            let model = try Self.build(
                PixtralConfiguration.self,
                Self.pixtral(text: ["vocab_size": 1200], top: ["vocab_size": 1200]),
                PixtralVLM.init)
            let processor = try VisionProcessorTests.pixtral(
                longestEdge: 8,
                tokenizer: ScriptTokenizer(specials: ["[IMG]": 60], encodeAddsBOS: true))
            // 16 x 12 becomes 8 x 6: 4 x 3 = 12 image tokens.
            let result = try await Self.endToEnd(processor, model, width: 16, height: 12)
            #expect(
                VisionProcessorTests.tokens(result.input).filter { $0 == 60 }.count == 12,
                "Pixtral: 12 image tokens")
            Self.checkEndToEnd("Pixtral", result)
        }

        @Test func lfm2vlProcessorOutputRunsThroughTheModel() async throws {
            let model = try Self.build(
                LFM2VLConfiguration.self,
                Self.lfm2vl(text: ["vocab_size": 1200], top: ["image_token_id": 396]),
                LFM2VL.init)
            let processor = try VisionProcessorTests.lfm2vl(
                tokenizer: ScriptTokenizer(specials: ["<image>": 396]))
            // 16 x 8 is 2 x 1 tiles: 4 x 8 patches and 2 x 4 image tokens.
            let result = try await Self.endToEnd(processor, model, width: 16, height: 8)
            #expect(
                VisionProcessorTests.tokens(result.input).filter { $0 == 396 }.count == 8,
                "LFM2VL: 8 image tokens")
            Self.checkEndToEnd("LFM2VL", result)
        }

        @Test func gemma3ProcessorOutputRunsThroughTheModel() async throws {
            let model = try Self.build(Gemma3Configuration.self, Self.gemma3(), Gemma3.init)
            let processor = try VisionProcessorTests.gemma3(
                tokenizer: ScriptTokenizer(
                    specials: ["<start_of_image>": 255_999], imageMarker: "<start_of_image>"))
            let result = try await Self.endToEnd(processor, model, width: 12, height: 8)
            #expect(
                VisionProcessorTests.tokens(result.input).filter { $0 == 262_144 }.count == 4,
                "Gemma3: 4 image tokens")
            Self.checkEndToEnd("Gemma3", result)
        }

        @Test func fastVLMProcessorOutputRunsThroughTheModel() async throws {
            let model = try Self.buildFastVLM(Self.fastVLM(top: ["vocab_size": 1200]))
            let processor = try VisionProcessorTests.fastVLM(
                tokenizer: ScriptTokenizer(specials: ["<image>": 50]), crop: 32)
            let result = try await Self.endToEnd(processor, model, width: 24, height: 16)
            #expect(
                VisionProcessorTests.tokens(result.input).filter { $0 == -200 }.count == 1,
                "FastVLM: one -200 token")
            Self.checkEndToEnd("FastVLM", result)
        }

        /// `Idefics3Processor` gives one image token. The model puts the
        /// image features at groups of `features per image` image tokens
        /// (16 here), so it does not use the image of the processor output.
        @Test func idefics3ProcessorOutputRunsThroughTheModel() async throws {
            // Image size 384 (the size of the processor) with patch size 48:
            // 8 x 8 patches and 16 image tokens after the 2 x 2 shuffle.
            let model = try Self.build(
                Idefics3Configuration.self,
                Self.idefics3(
                    text: ["vocab_size": 49160], vision: ["patch_size": 48, "image_size": 384],
                    top: ["vocab_size": 49160, "image_token_id": 49153]), Idefics3.init)
            let processor = try VisionProcessorTests.idefics3(tokenizer: ScriptTokenizer())
            let result = try await Self.endToEnd(processor, model, width: 20, height: 10)
            #expect(
                Self.isAllFinite(result.white), "Idefics3: processor output gives finite logits")

            // Control: 16 image tokens with the same pixels use the image.
            let tokens = VisionProcessorTests.tokens(result.input)
            let expanded = tokens.flatMap {
                $0 == 49153 ? Array(repeating: 49153, count: 16) : [$0]
            }
            let whitePixels = try #require(result.input.image?.pixels)
            let white = try Self.prefill(model, expanded, pixels: whitePixels)
            let black = try Self.prefill(model, expanded, pixels: -whitePixels)
            #expect(
                SyntheticModel.maxAbsDifference(white[0..., -1], black[0..., -1]) > 1e-3,
                "Idefics3: 16 image tokens use the image")

            let difference = SyntheticModel.maxAbsDifference(
                result.white[0..., -1], result.black[0..., -1])
            // A synchronous function, so that the synchronous
            // `withKnownIssue` is used.
            func check() {
                withKnownIssue(
                    """
                    Idefics3Processor inserts one image token (Idefics3.swift:881-882) and \
                    ignores image_seq_len. The model puts the features at groups of \
                    features-per-image tokens (Idefics3.swift:704-710), so with one token it \
                    drops the image.
                    """
                ) {
                    #expect(
                        difference > 1e-3, "Idefics3: the image of the processor output is used")
                } matching: {
                    $0.isFailedExpectation(["the image of the processor output is used"])
                }
            }
            check()
        }
    }
}
