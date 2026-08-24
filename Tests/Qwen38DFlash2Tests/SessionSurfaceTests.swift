import MLX
import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 DFlash2 speculative session")
struct DFlash2SessionSurfaceTests {
    @Test("session owns target and draft caches for the full generation")
    func surfaceCompiles() {
        let factory:
            (
                any DFlash2QwenTarget, DFlash2DraftModel, Int
            ) -> DFlash2Session = {
                DFlash2Session(target: $0, draft: $1, promptLength: $2)
            }
        let prefill: (DFlash2Session, MLXArray) -> MLXArray = {
            $0.prefill(promptTokens: $1)
        }
        let cycle: (DFlash2Session) -> DFlash2CycleResult = { $0.step() }
        let warmCycle: (DFlash2Session, Int) -> DFlash2CycleResult = {
            $0.warmStep(physicalWidth: $1)
        }
        _ = factory
        _ = prefill
        _ = cycle
        _ = warmCycle
    }
}
