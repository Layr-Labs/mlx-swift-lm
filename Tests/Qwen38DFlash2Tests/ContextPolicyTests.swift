import Testing

@testable import Qwen38DFlash2

@Suite("PR 335 final DFlash2 context policy")
struct DFlash2ContextPolicyTests {
    @Test("16K and longer binds the fixed physical-M8 route")
    func longContextIsFixedM8() {
        var policy = DFlash2ContextPolicy(promptLength: 16_384)
        #expect(policy.nextPhysicalWidth == 8)
        policy.record(blockLength: 8, acceptedDraftTokens: 0)
        #expect(policy.nextPhysicalWidth == 8)
    }

    @Test("short context starts at row-11 plus row-15 EMA width")
    func shortContextStartsAdaptive() {
        let policy = DFlash2ContextPolicy(promptLength: 1_024)
        #expect(policy.nextPhysicalWidth == 5)
    }

    @Test("full acceptance expands the next physical width")
    func fullAcceptanceExpands() {
        var policy = DFlash2ContextPolicy(promptLength: 1_024)
        policy.record(blockLength: 5, acceptedDraftTokens: 4)
        #expect(policy.nextPhysicalWidth == 6)
    }

    @Test("repeated first-position rejection can select the direct M1 route")
    func rejectionContractsToM1() {
        var policy = DFlash2ContextPolicy(promptLength: 1_024)
        for _ in 0 ..< 16 {
            policy.record(
                blockLength: policy.nextPhysicalWidth,
                acceptedDraftTokens: 0)
        }
        #expect(policy.nextPhysicalWidth == 1)
    }
}
