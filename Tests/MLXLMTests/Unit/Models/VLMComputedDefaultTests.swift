import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

extension UnitTests {
    /// Some VLM configurations store an optional value (`_ropeTheta`) and
    /// apply the default in a computed property (`ropeTheta`). The stored
    /// fields of the minimal JSON show only `nil` for these values, so these
    /// tests check the computed properties. Each expected default is read from
    /// the production source.
    @Suite
    struct VLMComputedDefaultTests {

        private func minimal<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
            let json = try #require(
                VLMConfigurationDecodingTests.cases.first { $0.name == name }
            ).json
            return try JSONDecoder.json5().decode(T.self, from: Data(json.utf8))
        }

        @Test
        func lfm2VLDefaults() throws {
            let config = try minimal(
                "LFM2VL: LFM2VLConfiguration", as: LFM2VLConfiguration.self)
            #expect(config.downsampleFactor == 2)
            #expect(config.imageTokenIndex == 396)
            #expect(config.projectorBias == true)
            #expect(config.projectorHiddenSize == 2560)
            #expect(config.projectorUseLayernorm == true)
            #expect(config.visionFeatureLayer == -1)
            #expect(config.doImageSplitting == true)
            #expect(config.maxImageTokens == 256)
            #expect(config.maxNumPatches == 1024)
            #expect(config.minImageTokens == 64)
            #expect(config.minTiles == 2)
            #expect(config.useThumbnail == false)

            let text = config.textConfiguration
            #expect(text.normEps == 1e-5)
            #expect(text.convBias == false)
            #expect(text.convLCache == 3)
            #expect(text.blockDim == text.hiddenSize)
            #expect(text.blockFFDim == text.hiddenSize)
            #expect(text.blockMultipleOf == 256)
            #expect(text.blockFFNDimMultiplier == 1.0)
            #expect(text.blockAutoAdjustFFDim == true)
            // Without full_attn_idxs and layer_types, every layer is full attention.
            #expect(text.fullAttnIdxs == Array(0 ..< text.hiddenLayers))
            #expect(text.ropeTheta == 1_000_000)

            let vision = config.visionConfiguration
            #expect(vision.numChannels == 3)
            #expect(vision.imageSize == 224)
            #expect(vision.patchSize == 16)
            #expect(vision.numPatches == 256)
            #expect(vision.layerNormEps == 1e-6)
        }

        @Test
        func lfm2VLProcessorDefaults() throws {
            let config = try minimal(
                "LFM2VL: LFM2VLProcessorConfiguration", as: LFM2VLProcessorConfiguration.self)
            #expect(config.imageMean == [0.5, 0.5, 0.5])
            #expect(config.imageStd == [0.5, 0.5, 0.5])
            #expect(config.tileSize == 512)
            #expect(config.encoderPatchSize == 16)
            #expect(config.maxTiles == 10)
            #expect(config.downsampleFactor == 2)
        }

        @Test
        func qwen3VLDefaults() throws {
            let config = try minimal(
                "Qwen3VL: Qwen3VLConfiguration", as: Qwen3VLConfiguration.self)
            #expect(config.ignoreIndex == -100)
            #expect(config.imageTokenId == 151_655)
            #expect(config.videoTokenId == 151_656)
            #expect(config.imageTokenIndex == 151_655)
            #expect(config.videoTokenIndex == 151_656)
            #expect(config.visionStartTokenId == 151_652)
            #expect(config.visionEndTokenId == 151_653)
            #expect(config.visionTokenId == 151_654)
            #expect(config.vocabSize == config.textConfiguration.vocabSize)
            #expect(config.eosTokenId == nil)

            let text = config.textConfiguration
            #expect(text.numKeyValueHeads == text.numAttentionHeads)
            #expect(text.ropeTheta == 1_000_000)
            #expect(text.rmsNormEps == 1e-6)
            #expect(text.ropeScaling == nil)
            #expect(text.normTopKProb == true)
            #expect(text.numExperts == 0)
            #expect(text.numExpertsPerTok == 0)
            #expect(text.decoderSparseStep == 1)
            #expect(text.mlpOnlyLayers == [])
            #expect(text.moeIntermediateSize == text.intermediateSize)
            #expect(text.tieWordEmbeddings == true)
            #expect(text.attentionBias == false)
            #expect(text.hiddenAct == "silu")

            let vision = config.visionConfiguration
            #expect(vision.inChannels == 3)
            #expect(vision.hiddenAct == "gelu")
            #expect(vision.deepstackVisualIndexes == [])
        }

        @Test
        func mistral3VLMDefaults() throws {
            let config = try minimal(
                "Mistral3: Mistral3VLMConfiguration", as: Mistral3VLMConfiguration.self)
            #expect(config.ignoreIndex == -100)
            #expect(config.imageTokenIndex == 10)
            #expect(config.visionFeatureSelectStrategy == "full")
            #expect(config.visionFeatureLayer == -1)
            #expect(config.vocabSize == 32000)
            #expect(config.spatialMergeSize == 2)
            #expect(config.multimodalProjectorBias == false)
            #expect(config.eosTokenId == nil)

            let text = config.textConfig
            #expect(text.headDim == nil)
            #expect(text.maxPositionEmbeddings == nil)
            #expect(text.numKeyValueHeads == text.numAttentionHeads)
            #expect(text.ropeTheta == 1_000_000_000)
            #expect(text.ropeParameters == nil)
            #expect(text.ropeTraditional == false)
            #expect(text.ropeScaling == nil)
            #expect(text.tieWordEmbeddings == false)
            #expect(text.layerTypes == nil)
            #expect(text.slidingWindow == nil)
            #expect(text.useQkNorm == false)

            let vision = config.visionConfig
            #expect(vision.numChannels == 3)
            #expect(vision.rmsNormEps == 1e-5)
            #expect(vision.headDim == vision.hiddenSize / vision.numAttentionHeads)
            #expect(vision.ropeTheta == 10_000)
        }

        @Test
        func pixtralDefaults() throws {
            let config = try minimal(
                "Pixtral: PixtralConfiguration", as: PixtralConfiguration.self)
            #expect(config.ignoreIndex == -100)
            #expect(config.imageTokenIndex == 10)
            #expect(config.visionFeatureSelectStrategy == "full")
            #expect(config.visionFeatureLayer == -1)
            #expect(config.vocabSize == 32000)

            let text = config.textConfig
            #expect(text.headDim == text.hiddenSize / text.numAttentionHeads)
            #expect(text.maxPositionEmbeddings == nil)
            #expect(text.numKeyValueHeads == text.numAttentionHeads)
            #expect(text.ropeTheta == 1_000_000_000)
            #expect(text.ropeTraditional == false)
            #expect(text.ropeScaling == nil)
            #expect(text.tieWordEmbeddings == false)
            #expect(text.useQkNorm == false)
        }

        @Test
        func qwen35VLMDefaults() throws {
            let config = try minimal(
                "Qwen35: Qwen35Configuration", as: MLXVLM.Qwen35Configuration.self)
            #expect(config.ignoreIndex == -100)
            #expect(config.imageTokenId == 248_056)
            #expect(config.videoTokenId == 248_057)
            #expect(config.imageTokenIndex == 248_056)
            #expect(config.videoTokenIndex == 248_057)
            #expect(config.visionStartTokenId == 248_045)
            #expect(config.visionEndTokenId == 248_046)
            #expect(config.vocabSize == config.textConfiguration.vocabularySize)
            #expect(config.eosTokenId == nil)
        }

        @Test
        func qwen4ExpVLMDefaults() throws {
            let config = try minimal(
                "Qwen4Exp: Qwen4ExpVLMConfiguration", as: Qwen4ExpVLMConfiguration.self)
            #expect(config.imageTokenId == 248_056)
            #expect(config.videoTokenId == 248_057)
            #expect(config.imageTokenIndex == 248_056)
            #expect(config.videoTokenIndex == 248_057)
            #expect(config.visionStartTokenId == 248_053)
            #expect(config.visionEndTokenId == 248_054)
            #expect(config.servesVision == false)
        }

        @Test
        func gemma3VLMDefaults() throws {
            let config = try minimal(
                "Gemma3: Gemma3Configuration", as: MLXVLM.Gemma3Configuration.self)
            #expect(config.vocabularySize == 262_208)
            #expect(config.padTokenId == 0)
            #expect(config.hiddenSize == config.textConfiguration.hiddenSize)
            let text = config.textConfiguration
            #expect(text.attentionHeads == 8)
            #expect(text.kvHeads == 4)
            #expect(text.headDim == 256)
            #expect(text.queryPreAttnScalar == 256)
        }

        @Test
        func glmOcrDefaults() throws {
            let config = try minimal(
                "GlmOcr: GlmOcrConfiguration", as: GlmOcrConfiguration.self)
            let base = config.baseConfiguration
            #expect(base.vocabularySize == 59392)
            #expect(base.imageTokenId == 59280)
            #expect(base.videoTokenId == 59281)
            #expect(base.imageStartTokenId == 59256)
            #expect(base.hiddenSize == 1536)
            let text = config.textConfiguration
            #expect(text.rmsNormEps == 1e-5)
            #expect(text.tieWordEmbeddings == false)
            #expect(text.ropeParameters.partialRotaryFactor == 1.0)
            #expect(text.ropeParameters.ropeTheta == 10_000)
            #expect(text.ropeTheta == 10_000)
            let vision = config.visionConfiguration
            #expect(vision.inChannels == 3)
            #expect(vision.rmsNormEps == 1e-5)
        }
    }
}
