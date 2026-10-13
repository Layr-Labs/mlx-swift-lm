import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 DFlash2 draft model surface")
struct Qwen38DFlash2DraftModelSurfaceTests {
    @Test("draft loader is bound to the pinned configuration type")
    func loaderSurfaceCompiles() {
        let loader: (URL) throws -> DFlash2Configuration = DFlash2DraftModel.loadConfiguration
        _ = loader
    }

    @Test("BF16 checkpoint is converted to the measured affine W4 G64 runtime")
    func runtimeQuantizationContract() {
        #expect(DFlash2DraftModel.runtimeQuantization.groupSize == 64)
        #expect(DFlash2DraftModel.runtimeQuantization.bits == 4)
        #expect(DFlash2DraftModel.runtimeQuantization.mode == .affine)
    }

    @Test("codebook remap follows the pinned checkpoint names")
    func codebookRemapNames() throws {
        #expect(
            try DFlash2DraftModel.remappedWeightName("candidate_selector.predecessor_codebook")
                == "candidate_selector.predecessor_codebook.weight")
        #expect(
            try DFlash2DraftModel.remappedWeightName("candidate_selector.successor_codebook")
                == "candidate_selector.successor_codebook.weight")
        #expect(
            try DFlash2DraftModel.remappedWeightName("layers.0.self_attn.q_proj.weight")
                == "layers.0.self_attn.q_proj.weight")
    }

    @Test("draft runtime surface exposes projected context, cache, and trained selection")
    func runtimeSurfaceCompiles() {
        let cacheFactory: (DFlash2DraftModel) -> [DFlash2ContextKVCache] = {
            $0.makeCache()
        }
        let projector: (DFlash2DraftModel, MLXArray) -> MLXArray = {
            $0.projectTargetFeatures($1)
        }
        let proposal:
            (
                DFlash2DraftModel, MLXArray, MLXArray, MLXArray, Float, Bool
            ) -> DFlash2DraftProposal = {
                $0.selectProposal(
                    draftHidden: $1,
                    logits: $2,
                    anchorIDs: $3,
                    temperature: $4,
                    captureQ: $5)
            }
        let advance:
            (
                DFlash2DraftModel, MLXArray, [DFlash2ContextKVCache]
            ) -> Void = {
                $0.advanceProjectedContextCache(draftContext: $1, cache: $2)
            }
        _ = cacheFactory
        _ = projector
        _ = proposal
        _ = advance
    }
}
