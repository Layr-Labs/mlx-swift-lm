import Testing

@testable import Qwen38DFlash2

@Suite("DFlash2 context cache plan")
struct DFlash2ContextCachePlanTests {
    @Test("empty long prompt keeps the sink and newest window")
    func initialLongPrompt() {
        let plan = DFlash2ContextWindowPlan(sinkSize: 64, windowSize: 2048)
        #expect(
            plan.spans(cacheLength: 0, inputLength: 16_384) == [
                0 ..< 64,
                14_336 ..< 16_384,
            ])
    }

    @Test("incremental commits append only the newest window")
    func incrementalCommit() {
        let plan = DFlash2ContextWindowPlan(sinkSize: 64, windowSize: 2048)
        #expect(plan.spans(cacheLength: 2112, inputLength: 8) == [0 ..< 8])
        #expect(plan.spans(cacheLength: 2112, inputLength: 4096) == [2048 ..< 4096])
    }

    @Test("prefill follows the pinned 2048-token cadence")
    func prefillRanges() {
        #expect(
            dflash2PrefillRanges(tokenCount: 5000) == [
                0 ..< 2048,
                2048 ..< 4096,
                4096 ..< 5000,
            ])
    }
}
