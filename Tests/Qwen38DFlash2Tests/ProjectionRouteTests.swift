import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 projection route table")
struct Qwen38ProjectionRouteTests {
    @Test("retained width and shape island is exact")
    func retainedRouteTable() {
        #expect(qwen38ProjectionRoute(width: 4, k: 5_120, n: 17_408) == .m4KConstSplit)
        #expect(qwen38ProjectionRoute(width: 5, k: 6_144, n: 5_120) == .m5Exact)
        #expect(
            qwen38ProjectionRoute(width: 6, k: 5_120, n: 10_240)
                == .m6(kParts: 1, barrierFree: true))
        #expect(
            qwen38ProjectionRoute(width: 6, k: 5_120, n: 17_408)
                == .m6(kParts: 1, barrierFree: true))
        #expect(
            qwen38ProjectionRoute(width: 6, k: 17_408, n: 5_120)
                == .m6(kParts: 2, barrierFree: false))
        #expect(
            qwen38ProjectionRoute(width: 7, k: 6_144, n: 5_120)
                == .m8NAX(simdgroups: 8))
        #expect(
            qwen38ProjectionRoute(width: 7, k: 5_120, n: 6_144)
                == .m8NAX(simdgroups: 8))
        #expect(
            qwen38ProjectionRoute(width: 8, k: 5_120, n: 1_024)
                == .m8NAX(simdgroups: 8))
        #expect(
            qwen38ProjectionRoute(width: 8, k: 5_120, n: 10_240)
                == .m8NAX(simdgroups: 8))
        #expect(
            qwen38ProjectionRoute(width: 8, k: 5_120, n: 17_408)
                == .m8NAX(simdgroups: 8))

        #expect(qwen38ProjectionRoute(width: 7, k: 5_120, n: 1_024) == .stock)
        #expect(qwen38ProjectionRoute(width: 8, k: 6_144, n: 5_120) == .stock)
        #expect(qwen38ProjectionRoute(width: 3, k: 5_120, n: 17_408) == .stock)
    }

    @Test("draft route matches the final PR receipt's stock production widths")
    func draftRouteTable() {
        #expect(qwen38DraftProjectionRoute(width: 4, k: 5_120, n: 17_408) == .stock)
        #expect(qwen38DraftProjectionRoute(width: 2, k: 5_120, n: 17_408) == .stock)
        #expect(qwen38DraftProjectionRoute(width: 3, k: 5_120, n: 17_408) == .stock)
        #expect(qwen38DraftProjectionRoute(width: 5, k: 5_120, n: 17_408) == .stock)
        #expect(qwen38DraftProjectionRoute(width: 6, k: 5_120, n: 17_408) == .stock)
        #expect(qwen38DraftProjectionRoute(width: 7, k: 5_120, n: 17_408) == .stock)
        #expect(qwen38DraftProjectionRoute(width: 8, k: 5_120, n: 17_408) == .stock)
        // M16 is construction-only warmup for the draft, never a production
        // block in the pinned width-8 policy.
        #expect(qwen38DraftProjectionRoute(width: 16, k: 5_120, n: 17_408) == .m16NAX)
        #expect(qwen38DraftProjectionRoute(width: 16, k: 5_121, n: 17_408) == .stock)
        #expect(qwen38DraftProjectionRoute(width: 16, k: 5_120, n: 17_404) == .stock)
    }
}
