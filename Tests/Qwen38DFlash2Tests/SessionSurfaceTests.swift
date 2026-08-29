import MLX
import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 DFlash2 speculative session")
struct DFlash2SessionSurfaceTests {
    @Test("step width is selected and validated once at construction")
    func stepBoundaryContract() {
        #expect(DFlash2WidthPolicy.adaptive.isValid)
        #expect(DFlash2WidthPolicy.fixed(1).isValid)
        #expect(DFlash2WidthPolicy.fixed(8).isValid)
        #expect(!DFlash2WidthPolicy.fixed(0).isValid)
        #expect(!DFlash2WidthPolicy.fixed(9).isValid)
        #expect(DFlash2WidthPolicy.adaptive.resolve(adaptiveWidth: 6) == 6)
        #expect(DFlash2WidthPolicy.fixed(8).resolve(adaptiveWidth: 6) == 8)
    }

    @Test("next-draft prefetch does not force target-cache rollback first")
    func prefetchEvaluationPlan() {
        #expect(
            !DFlash2CycleEvaluationPlan.explicitlyMaterializesTargetCache(
                prefetchingNextDraft: true))
        #expect(
            DFlash2CycleEvaluationPlan.explicitlyMaterializesTargetCache(
                prefetchingNextDraft: false))
    }

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
        let cycle: (DFlash2Session, Int) -> DFlash2CycleResult = {
            $0.step(remainingOutputTokens: $1)
        }
        let warmCycle: (DFlash2Session, Int) -> DFlash2CycleResult = {
            $0.warmStep(physicalWidth: $1)
        }
        let diagnosticCycle: (DFlash2Session) -> DFlash2CycleDiagnosticResult = {
            $0.diagnosticStep()
        }
        _ = factory
        _ = prefill
        _ = cycle
        _ = warmCycle
        _ = diagnosticCycle
    }
}
