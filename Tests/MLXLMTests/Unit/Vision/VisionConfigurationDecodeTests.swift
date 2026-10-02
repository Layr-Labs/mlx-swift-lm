import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

extension UnitTests {

    /// Decode tests of the model and processor configurations of Mistral3,
    /// Pixtral, LFM2VL, Gemma3, Idefics3 and FastVLM: default values,
    /// alternative keys and optional fields. No MLX arrays.
    @Suite
    struct VisionConfigurationDecodeTests {

        static func list(_ t: (CGFloat, CGFloat, CGFloat)) -> [CGFloat] {
            [t.0, t.1, t.2]
        }

        static func decode<T: Decodable>(_ type: T.Type, _ json: [String: Any]) throws -> T {
            try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: json))
        }

        static var mistralText: [String: Any] {
            [
                "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 48, "num_attention_heads": 4, "rms_norm_eps": 1e-6,
                "vocab_size": 64,
            ]
        }

        /// A Pixtral vision configuration without the optional keys.
        static var pixtralVision: [String: Any] {
            [
                "model_type": "pixtral", "hidden_size": 64, "num_hidden_layers": 1,
                "num_attention_heads": 4, "intermediate_size": 48, "patch_size": 2,
                "image_size": 8,
            ]
        }

        // MARK: - Mistral3

        @Test func mistral3ConfigurationDefaultsAndImageTokenKeys() throws {
            func configuration(_ extra: [String: Any]) throws -> Mistral3VLMConfiguration {
                try Self.decode(
                    Mistral3VLMConfiguration.self,
                    ([
                        "model_type": "mistral3", "text_config": Self.mistralText,
                        "vision_config": Self.pixtralVision,
                    ] as [String: Any]).merging(extra) { _, new in new })
            }
            let defaults = try configuration([:])
            #expect(defaults.imageTokenIndex == 10, "Mistral3: default image token")
            #expect(defaults.ignoreIndex == -100, "Mistral3: default ignore index")
            #expect(defaults.visionFeatureSelectStrategy == "full", "Mistral3: default strategy")
            #expect(defaults.visionFeatureLayer == -1, "Mistral3: default feature layer")
            #expect(defaults.vocabSize == 32000, "Mistral3: default vocabulary size")
            #expect(defaults.spatialMergeSize == 2, "Mistral3: default merge size")
            #expect(!defaults.multimodalProjectorBias, "Mistral3: default projector bias")
            #expect(defaults.eosTokenId == nil, "Mistral3: default EOS")

            #expect(
                try configuration(["image_token_id": 77]).imageTokenIndex == 77,
                "Mistral3: image_token_id")
            #expect(
                try configuration(["image_token_index": 5, "image_token_id": 77]).imageTokenIndex
                    == 5, "Mistral3: image_token_index wins")
            #expect(
                try configuration(["eos_token_id": [1, 2]]).eosTokenId == [1, 2],
                "Mistral3: EOS list")

            let text = defaults.textConfig
            #expect(text.headDim == nil, "Mistral3 text: no head_dim")
            #expect(text.maxPositionEmbeddings == nil, "Mistral3 text: no max positions")
            #expect(text.numKeyValueHeads == 4, "Mistral3 text: KV heads default to heads")
            #expect(text.ropeTheta == 1_000_000_000, "Mistral3 text: default RoPE theta")
            #expect(text.ropeParameters == nil, "Mistral3 text: no RoPE parameters")
            #expect(!text.ropeTraditional, "Mistral3 text: default RoPE traditional")
            #expect(text.ropeScaling == nil, "Mistral3 text: no RoPE scaling")
            #expect(!text.tieWordEmbeddings, "Mistral3 text: default tie")
            #expect(text.layerTypes == nil, "Mistral3 text: no layer types")
            #expect(text.slidingWindow == nil, "Mistral3 text: no sliding window")
            #expect(!text.useQkNorm, "Mistral3 text: default QK norm")

            let explicit = try Self.decode(
                Mistral3VLMTextConfiguration.self,
                Self.mistralText.merging([
                    "head_dim": 8, "num_key_value_heads": 2, "sliding_window": 16,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "rope_parameters": ["rope_type": "default", "rope_theta": 10000]
                        as [String: Any],
                    "use_qk_norm": true, "tie_word_embeddings": true,
                ]) { _, new in new })
            #expect(explicit.headDim == 8, "Mistral3 text: head_dim")
            #expect(explicit.numKeyValueHeads == 2, "Mistral3 text: KV heads")
            #expect(explicit.slidingWindow == 16, "Mistral3 text: sliding window")
            #expect(
                explicit.layerTypes == ["sliding_attention", "full_attention"],
                "Mistral3 text: layer types")
            #expect(
                explicit.ropeParameters?["rope_theta"]?.asFloat() == 10000,
                "Mistral3 text: RoPE theta parameter")
            #expect(explicit.useQkNorm && explicit.tieWordEmbeddings, "Mistral3 text: flags")
        }

        @Test func mistral3ProcessorConfigurationOptionalFields() throws {
            let c = try Self.decode(
                Mistral3VLMProcessorConfiguration.self,
                [
                    "image_processor": [
                        "image_mean": [0.1, 0.2, 0.3], "image_std": [0.4, 0.5, 0.6],
                        "size": ["width": 4, "height": 6] as [String: Any], "patch_size": 2,
                        "do_normalize": true, "rescale_factor": 0.5,
                    ] as [String: Any],
                    "image_token": "[IMG]", "image_break_token": "[IMG_BREAK]",
                    "image_end_token": "[IMG_END]", "patch_size": 2,
                ])
            #expect(c.imageBreakToken == "[IMG_BREAK]", "Mistral3 processor: break token")
            #expect(c.imageEndToken == "[IMG_END]", "Mistral3 processor: end token")
            #expect(c.spatialMergeSize == nil, "Mistral3 processor: no merge size")
            let size = c.imageProcessor.size
            #expect(
                size.width == 4 && size.height == 6 && size.longestEdge == nil,
                "Mistral3 processor: size")
            #expect(c.imageProcessor.doNormalize == true, "Mistral3 processor: do_normalize")
            #expect(c.imageProcessor.doResize == nil, "Mistral3 processor: no do_resize")
            #expect(c.imageProcessor.rescaleFactor == 0.5, "Mistral3 processor: rescale factor")
            #expect(
                Self.list(c.imageProcessor.imageMeanTuple) == [0.1, 0.2, 0.3],
                "Mistral3 processor: mean tuple")
            #expect(
                Self.list(c.imageProcessor.imageStdTuple) == [0.4, 0.5, 0.6],
                "Mistral3 processor: std tuple")
        }

        // MARK: - Pixtral

        @Test func pixtralConfigurationDefaults() throws {
            let c = try Self.decode(
                PixtralConfiguration.self,
                [
                    "model_type": "pixtral", "image_token_id": 9,
                    "text_config": Self.mistralText, "vision_config": Self.pixtralVision,
                ])
            #expect(c.imageTokenIndex == 9, "Pixtral: image_token_id")
            #expect(c.ignoreIndex == -100, "Pixtral: default ignore index")
            #expect(c.visionFeatureSelectStrategy == "full", "Pixtral: default strategy")
            #expect(c.visionFeatureLayer == -1, "Pixtral: default feature layer")
            #expect(c.vocabSize == 32000, "Pixtral: default vocabulary size")

            let text = c.textConfig
            #expect(text.headDim == 8, "Pixtral text: head_dim defaults to hidden / heads")
            #expect(text.numKeyValueHeads == 4, "Pixtral text: KV heads default to heads")
            #expect(text.ropeTheta == 1_000_000_000, "Pixtral text: default RoPE theta")
            #expect(text.maxPositionEmbeddings == nil, "Pixtral text: no max positions")
            #expect(text.ropeScaling == nil, "Pixtral text: no RoPE scaling")
            #expect(
                !text.ropeTraditional && !text.tieWordEmbeddings && !text.useQkNorm,
                "Pixtral text: default flags")

            let vision = c.visionConfig
            #expect(vision.numChannels == 3, "Pixtral vision: default channels")
            #expect(vision.rmsNormEps == 1e-5, "Pixtral vision: default epsilon")
            #expect(vision.headDim == 16, "Pixtral vision: head_dim defaults to hidden / heads")
            #expect(vision.ropeTheta == 10000, "Pixtral vision: default RoPE theta")

            let neither = try Self.decode(
                PixtralConfiguration.self,
                [
                    "model_type": "pixtral", "text_config": Self.mistralText,
                    "vision_config": Self.pixtralVision,
                ])
            #expect(neither.imageTokenIndex == 10, "Pixtral: default image token")
        }

        @Test func pixtralProcessorConfigurationWithoutOptionalFields() throws {
            let c = try Self.decode(
                PixtralProcessorConfiguration.self,
                [
                    "image_processor": [
                        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                        "size": ["longest_edge": 1540] as [String: Any], "patch_size": 16,
                    ] as [String: Any],
                    "image_token": "[IMG]", "patch_size": 16,
                ])
            #expect(c.imageToken == "[IMG]", "Pixtral processor: image token")
            #expect(c.imageBreakToken == nil, "Pixtral processor: no break token")
            #expect(c.imageEndToken == nil, "Pixtral processor: no end token")
            #expect(c.imageProcessor.size.longestEdge == 1540, "Pixtral processor: longest edge")
            #expect(c.imageProcessor.doRescale == nil, "Pixtral processor: no do_rescale")
        }

        // MARK: - LFM2VL

        static var lfm2Text: [String: Any] {
            [
                "model_type": "lfm2", "hidden_size": 32, "num_hidden_layers": 4,
                "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64,
            ]
        }

        static var lfm2Vision: [String: Any] {
            [
                "model_type": "siglip2_vision_model", "hidden_size": 32,
                "intermediate_size": 48, "num_hidden_layers": 1, "num_attention_heads": 4,
            ]
        }

        @Test func lfm2vlConfigurationDefaults() throws {
            let c = try Self.decode(
                LFM2VLConfiguration.self,
                [
                    "model_type": "lfm2_vl", "text_config": Self.lfm2Text,
                    "vision_config": Self.lfm2Vision,
                ])
            #expect(c.downsampleFactor == 2, "LFM2VL: default downsample factor")
            #expect(c.imageTokenIndex == 396, "LFM2VL: default image token")
            #expect(c.projectorBias, "LFM2VL: default projector bias")
            #expect(c.projectorHiddenSize == 2560, "LFM2VL: default projector size")
            #expect(c.projectorUseLayernorm, "LFM2VL: default projector layer norm")
            #expect(c.visionFeatureLayer == -1, "LFM2VL: default feature layer")
            #expect(c.doImageSplitting, "LFM2VL: default image splitting")
            #expect(c.maxImageTokens == 256, "LFM2VL: default max image tokens")
            #expect(c.maxNumPatches == 1024, "LFM2VL: default max patches")
            #expect(c.minImageTokens == 64, "LFM2VL: default min image tokens")
            #expect(c.minTiles == 2, "LFM2VL: default min tiles")
            #expect(!c.useThumbnail, "LFM2VL: default thumbnail")

            let text = c.textConfiguration
            #expect(text.normEps == 1e-5, "LFM2VL text: default epsilon")
            #expect(!text.convBias, "LFM2VL text: default convolution bias")
            #expect(text.convLCache == 3, "LFM2VL text: default convolution cache")
            #expect(text.blockDim == 32, "LFM2VL text: block_dim defaults to hidden size")
            #expect(text.blockFFDim == 32, "LFM2VL text: block_ff_dim defaults to hidden size")
            #expect(text.blockMultipleOf == 256, "LFM2VL text: default multiple")
            #expect(text.blockFFNDimMultiplier == 1, "LFM2VL text: default multiplier")
            #expect(text.blockAutoAdjustFFDim, "LFM2VL text: default auto adjust")
            #expect(text.ropeTheta == 1_000_000, "LFM2VL text: default RoPE theta")
            #expect(text.fullAttnIdxs == [0, 1, 2, 3], "LFM2VL text: all layers attend")

            let vision = c.visionConfiguration
            #expect(vision.numChannels == 3, "LFM2VL vision: default channels")
            #expect(vision.imageSize == 224, "LFM2VL vision: default image size")
            #expect(vision.patchSize == 16, "LFM2VL vision: default patch size")
            #expect(vision.numPatches == 256, "LFM2VL vision: default patches")
            #expect(vision.layerNormEps == 1e-6, "LFM2VL vision: default epsilon")
        }

        @Test func lfm2vlAttentionLayersFromIndexesOrLayerTypes() throws {
            func indexes(_ extra: [String: Any]) throws -> [Int] {
                try Self.decode(
                    LFM2VLConfiguration.TextConfiguration.self,
                    Self.lfm2Text.merging(extra) { _, new in new }
                ).fullAttnIdxs
            }
            let types = ["conv", "full_attention", "conv", "full_attention"]
            #expect(try indexes(["full_attn_idxs": [2]]) == [2], "LFM2VL: full_attn_idxs")
            #expect(try indexes(["layer_types": types]) == [1, 3], "LFM2VL: layer_types")
            #expect(
                try indexes(["full_attn_idxs": [0], "layer_types": types]) == [0],
                "LFM2VL: full_attn_idxs wins over layer_types")
        }

        @Test func lfm2vlProcessorConfigurationDefaults() throws {
            let defaults = try Self.decode(LFM2VLProcessorConfiguration.self, [:])
            #expect(defaults.imageMean == [0.5, 0.5, 0.5], "LFM2VL processor: default mean")
            #expect(defaults.imageStd == [0.5, 0.5, 0.5], "LFM2VL processor: default std")
            #expect(defaults.tileSize == 512, "LFM2VL processor: default tile size")
            #expect(defaults.encoderPatchSize == 16, "LFM2VL processor: default patch size")
            #expect(defaults.maxTiles == 10, "LFM2VL processor: default max tiles")
            #expect(defaults.downsampleFactor == 2, "LFM2VL processor: default downsample")

            let explicit = try Self.decode(
                LFM2VLProcessorConfiguration.self,
                [
                    "image_mean": [0.1, 0.2, 0.3], "image_std": [0.4, 0.5, 0.6],
                    "tile_size": 8, "encoder_patch_size": 2, "max_tiles": 3,
                    "downsample_factor": 1,
                ])
            #expect(
                Self.list(explicit.imageMeanTuple) == [0.1, 0.2, 0.3],
                "LFM2VL processor: mean tuple")
            #expect(
                Self.list(explicit.imageStdTuple) == [0.4, 0.5, 0.6], "LFM2VL processor: std tuple")
            #expect(
                explicit.tileSize == 8 && explicit.encoderPatchSize == 2
                    && explicit.maxTiles == 3 && explicit.downsampleFactor == 1,
                "LFM2VL processor: explicit sizes")
        }

        // MARK: - Gemma3

        static var gemma3: [String: Any] {
            [
                "model_type": "gemma3", "mm_tokens_per_image": 4,
                "text_config": [
                    "model_type": "gemma3_text", "hidden_size": 32, "num_hidden_layers": 6,
                    "intermediate_size": 48, "sliding_window": 4,
                ] as [String: Any],
                "vision_config": [
                    "model_type": "siglip_vision_model", "num_hidden_layers": 1,
                    "hidden_size": 32, "intermediate_size": 48, "num_attention_heads": 4,
                    "patch_size": 2, "image_size": 8,
                ] as [String: Any],
            ]
        }

        @Test func gemma3ConfigurationDefaultsAndOverrides() throws {
            let defaults = try Self.decode(Gemma3Configuration.self, Self.gemma3)
            #expect(defaults.vocabularySize == 262_208, "Gemma3: default vocabulary size")
            #expect(defaults.padTokenId == 0, "Gemma3: default pad token")
            #expect(defaults.hiddenSize == 32, "Gemma3: hidden size of the text model")
            #expect(defaults.quantization == nil, "Gemma3: no quantization")
            let text = defaults.textConfiguration
            #expect(text.attentionHeads == 8, "Gemma3 text: default heads")
            #expect(text.kvHeads == 4, "Gemma3 text: default KV heads")
            #expect(text.headDim == 256, "Gemma3 text: default head_dim")
            #expect(text.queryPreAttnScalar == 256, "Gemma3 text: default query scalar")
            #expect(text.finalLogitSoftcapping == nil, "Gemma3 text: no softcap")
            #expect(defaults.visionConfiguration.numChannels == 3, "Gemma3 vision: channels")

            let explicit = try Self.decode(
                Gemma3Configuration.self,
                Self.gemma3.merging([
                    "vocab_size": 1000, "pad_token_id": 3,
                    "quantization": ["group_size": 32, "bits": 4] as [String: Any],
                ]) { _, new in new })
            #expect(explicit.vocabularySize == 1000, "Gemma3: vocab_size")
            #expect(explicit.padTokenId == 3, "Gemma3: pad_token_id")
            #expect(explicit.quantization?.groupSize == 32, "Gemma3: quantization group size")
            #expect(explicit.quantization?.bits == 4, "Gemma3: quantization bits")
        }

        @Test func gemma3ProcessorConfigurationReadsTheHeight() throws {
            let c = try Self.decode(
                Gemma3ProcessorConfiguration.self,
                [
                    "processor_class": "Gemma3Processor",
                    "image_processor_type": "Gemma3ImageProcessor", "do_normalize": true,
                    "do_rescale": true, "do_resize": true, "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5], "image_seq_length": 256, "resample": 2,
                    "rescale_factor": 0.5, "size": ["height": 896, "width": 900] as [String: Any],
                    "do_pan_and_scan": false,
                ])
            #expect(c.imageSize == 896, "Gemma3 processor: image size is the height")
            #expect(c.imageTokenId == 262_144, "Gemma3 processor: image token")
            #expect(c.imageSeqLength == 256, "Gemma3 processor: sequence length")
            #expect(c.doPanAndScan == false, "Gemma3 processor: do_pan_and_scan")
            #expect(c.panAndScanMaxNumCrops == nil, "Gemma3 processor: no crop count")
            #expect(c.doConvertRgb == nil, "Gemma3 processor: no do_convert_rgb")
            #expect(Self.list(c.imageMeanTuple) == [0.5, 0.5, 0.5], "Gemma3 processor: mean tuple")
            #expect(Self.list(c.imageStdTuple) == [0.5, 0.5, 0.5], "Gemma3 processor: std tuple")
        }

        // MARK: - Idefics3

        static func idefics3(_ extra: [String: Any] = [:], text: [String: Any] = [:])
            -> [String: Any]
        {
            let textConfig: [String: Any] = [
                "model_type": "llama", "hidden_size": 32, "intermediate_size": 48,
                "num_attention_heads": 4, "rms_norm_eps": 1e-6, "vocab_size": 64,
                "num_key_value_heads": 2, "rope_theta": 10000,
            ]
            let configuration: [String: Any] = [
                "model_type": "idefics3",
                "text_config": textConfig.merging(text) { _, new in new },
                "vision_config": [
                    "model_type": "idefics3_vision", "hidden_size": 32,
                    "num_attention_heads": 4, "patch_size": 2, "image_size": 8,
                ] as [String: Any],
            ]
            return configuration.merging(extra) { _, new in new }
        }

        @Test func idefics3ConfigurationDefaults() throws {
            let c = try Self.decode(Idefics3Configuration.self, Self.idefics3())
            #expect(c.ignoreIndex == -100, "Idefics3: default ignore index")
            #expect(c.vocabSize == 128_259, "Idefics3: default vocabulary size")
            #expect(c.scaleFactor == 2, "Idefics3: default scale factor")
            #expect(c.imageTokenId == 49153, "Idefics3: default image token")
            #expect(c.imageTokenIndex == 49153, "Idefics3: image token index defaults to the ID")
            #expect(c.textConfig.numHiddenLayers == 32, "Idefics3 text: default layers")
            #expect(!c.textConfig.ropeTraditional, "Idefics3 text: default RoPE traditional")
            #expect(!c.textConfig.tieWordEmbeddings, "Idefics3 text: default tie")
            #expect(c.visionConfig.numHiddenLayers == 12, "Idefics3 vision: default layers")
            #expect(c.visionConfig.intermediateSize == 3072, "Idefics3 vision: default MLP size")
            #expect(c.visionConfig.numChannels == 3, "Idefics3 vision: default channels")
            #expect(c.visionConfig.layerNormEps == 1e-6, "Idefics3 vision: default epsilon")

            let explicit = try Self.decode(
                Idefics3Configuration.self,
                Self.idefics3(
                    [
                        "ignore_index": -1, "vocab_size": 70, "scale_factor": 3,
                        "image_token_id": 7, "image_token_index": 5,
                    ], text: ["num_hidden_layers": 2, "tie_word_embeddings": true]))
            #expect(explicit.ignoreIndex == -1, "Idefics3: ignore_index")
            #expect(explicit.vocabSize == 70, "Idefics3: vocab_size")
            #expect(explicit.scaleFactor == 3, "Idefics3: scale_factor")
            #expect(explicit.imageTokenId == 7, "Idefics3: image_token_id")
            #expect(explicit.imageTokenIndex == 5, "Idefics3: image_token_index")
            #expect(explicit.textConfig.numHiddenLayers == 2, "Idefics3 text: layers")
            #expect(explicit.textConfig.tieWordEmbeddings, "Idefics3 text: tie")
        }

        @Test func idefics3ProcessorConfiguration() throws {
            let c = try Self.decode(
                Idefics3ProcessorConfiguration.self,
                [
                    "image_mean": [0.1, 0.2, 0.3], "image_std": [0.4, 0.5, 0.6],
                    "size": ["longest_edge": 384] as [String: Any],
                ])
            #expect(c.size.longestEdge == 384, "Idefics3 processor: longest edge")
            #expect(c.imageSequenceLength == nil, "Idefics3 processor: no image_seq_len")
            #expect(
                Self.list(c.imageMeanTuple) == [0.1, 0.2, 0.3], "Idefics3 processor: mean tuple")
            #expect(Self.list(c.imageStdTuple) == [0.4, 0.5, 0.6], "Idefics3 processor: std tuple")
        }

        // MARK: - FastVLM

        static func fastVLM(_ extra: [String: Any] = [:]) -> [String: Any] {
            let configuration: [String: Any] = [
                "model_type": "fastvlm", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                "vocab_size": 64, "eos_token_id": 1, "mm_projector_type": "mlp2x_gelu",
                "mm_hidden_size": 64, "tokenizer_model_max_length": 256,
                "tokenizer_padding_side": "right",
                "vision_config": [
                    "cls_ratio": 2.0, "down_patch_size": 3, "down_stride": 2,
                    "downsamples": [true, true], "embed_dims": [16, 32], "hidden_size": 32,
                    "image_size": 32, "intermediate_size": 64, "layers": [1, 1],
                    "layer_scale_init_value": 1e-5, "mlp_ratios": [2, 2], "num_classes": 0,
                    "patch_size": 16, "pos_embs_shapes": [NSNull(), [7, 7]] as [Any],
                    "projection_dim": 32, "repmixer_kernel_size": 3,
                    "token_mixers": ["repmixer", "attention"],
                ] as [String: Any],
            ]
            return configuration.merging(extra) { _, new in new }
        }

        @Test func fastVLMConfigurationReadsTheTopLevelFields() throws {
            let c = try Self.decode(FastVLMConfiguration.self, Self.fastVLM())
            let base = c.baseConfiguration
            #expect(base.imageTokenIndex == -200, "FastVLM: default image token")
            #expect(base.modelType == "fastvlm", "FastVLM: model type")
            #expect(base.eosTokenId == 1, "FastVLM: EOS")
            #expect(base.multimodalProjectorType == "mlp2x_gelu", "FastVLM: projector type")
            #expect(base.multimodalProjectorHiddenSize == 64, "FastVLM: projector input size")
            #expect(base.tokenizerModelMaxLangth == 256, "FastVLM: max length")
            #expect(base.tokenizerPaddingSide == "right", "FastVLM: padding side")
            #expect(c.textConfiguration.hiddenSize == 32, "FastVLM: text hidden size")
            #expect(c.textConfiguration.kvHeads == 2, "FastVLM: text KV heads")
            #expect(c.textConfiguration.tieWordEmbeddings, "FastVLM: default tie")
            let vision = c.visionConfiguration
            #expect(vision.posEmbedShapes == [nil, [7, 7]], "FastVLM vision: position shapes")
            #expect(vision.tokenMixers == ["repmixer", "attention"], "FastVLM vision: mixers")
            #expect(vision.downSamples == [true, true], "FastVLM vision: downsamples")
            #expect(vision.classHeadRatio == 2, "FastVLM vision: class head ratio")

            let explicit = try Self.decode(
                FastVLMConfiguration.self, Self.fastVLM(["image_token_index": 7]))
            #expect(explicit.baseConfiguration.imageTokenIndex == 7, "FastVLM: image_token_index")
        }

        @Test func fastVLMProcessorConfiguration() throws {
            let c = try Self.decode(
                FastVLMProcessorConfiguration.self,
                [
                    "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
                    "crop_size": ["width": 16, "height": 8] as [String: Any],
                ])
            #expect(c.cropSize.cgSize == CGSize(width: 16, height: 8), "FastVLM processor: crop")
            #expect(Self.list(c.imageMeanTuple) == [0, 0, 0], "FastVLM processor: mean tuple")
            #expect(Self.list(c.imageStdTuple) == [1, 1, 1], "FastVLM processor: std tuple")
        }

        /// The FastVLM generator puts the images before the text and drops
        /// only an empty system message.
        @Test func fastVLMMessageGenerator() {
            let image = UserInput.Image.url(URL(fileURLWithPath: "/nonexistent/image.png"))
            let generator = FastVLMMessageGenerator()
            let messages = generator.generate(messages: [
                .system(""), .user("Hi", images: [image]),
            ])
            #expect(messages.count == 1, "FastVLM generator: empty system dropped")
            #expect(messages.first?["role"] as? String == "user", "FastVLM generator: role")
            #expect(
                messages.first?["content"] as? [[String: String]] == [
                    ["type": "image"], ["type": "text", "text": "Hi"],
                ], "FastVLM generator: image before text")

            let withSystem = generator.generate(messages: [.system("S"), .user("Hi")])
            #expect(
                withSystem.map { $0["role"] as? String } == ["system", "user"],
                "FastVLM generator: system with text kept")
        }
    }
}
