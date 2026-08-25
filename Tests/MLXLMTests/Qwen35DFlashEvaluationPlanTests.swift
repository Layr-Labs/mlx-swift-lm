import MLXLLM
import Testing

@Suite("Qwen 3.8 DFlash evaluation ladder")
struct Qwen35DFlashEvaluationPlanTests {
    @Test("decode and prefill construction plans are fixed")
    func fixedRungs() {
        #expect(Qwen35DFlashEvaluationPlan.decodeExclusiveEnds == [1, 2, 10, 20, 30, 40, 50, 58])
        #expect(Qwen35DFlashEvaluationPlan.prefillExclusiveEnds(stride: 3) == [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63])
    }
}
