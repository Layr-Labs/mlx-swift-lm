import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Qwen 3.8 DFlash2 target capture plan")
struct Qwen38DFlash2TargetCapturePlanTests {
    @Test("fixed segments capture post-block features without per-layer membership checks")
    func fixedSegments() {
        #expect(Qwen35DFlashCapturePlan.layerIDs == [5, 19, 33, 47, 61])
        #expect(Qwen35DFlashCapturePlan.exclusiveSegmentEnds == [6, 20, 34, 48, 62, 64])
        #expect(Qwen35DFlashCapturePlan.concatenatedWidth(hiddenSize: 5_120) == 25_600)
    }

    @Test("text model exposes one direct logits plus feature forward")
    func capturedForwardSurfaceCompiles() {
        func requireSurface(
            _ model: Qwen35TextModel,
            input: LMInput.Text,
            cache: [any KVCache]
        ) -> Qwen35DFlashForwardResult {
            model.dflashForward(input: input, cache: cache, nConfirmed: 0)
        }

        _ = requireSurface

        func requirePrefillSurface(
            _ model: Qwen35TextModel,
            input: LMInput.Text,
            cache: [any KVCache]
        ) -> Qwen35DFlashForwardResult {
            model.dflashPrefillChunk(input: input, cache: cache)
        }
        _ = requirePrefillSurface

        func requireTargetOnlyPrefillSurface(
            _ model: Qwen35TextModel,
            input: LMInput.Text,
            cache: [any KVCache]
        ) -> MLXArray {
            model.dflashTargetOnlyPrefillChunk(input: input, cache: cache)
        }
        _ = requireTargetOnlyPrefillSurface
    }

    @Test("target exposes the embedding and LM head used by the trained drafter")
    func targetDraftBoundaryCompiles() {
        let textEmbedding: (Qwen35TextModel, MLXArray) -> MLXArray = {
            $0.dflashInputEmbedding($1)
        }
        let textLogits: (Qwen35TextModel, MLXArray) -> MLXArray = {
            $0.dflashLogits($1)
        }
        let wrapperEmbedding: (Qwen35Model, MLXArray) -> MLXArray = {
            $0.dflashInputEmbedding($1)
        }
        _ = textEmbedding
        _ = textLogits
        _ = wrapperEmbedding
    }
}
