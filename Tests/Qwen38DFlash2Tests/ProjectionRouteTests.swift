import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 projection route table")
struct Qwen38ProjectionRouteTests {
    @Test("production installation requires the complete pinned module set")
    func productionInstallContract() throws {
        let complete = Qwen38ProjectionInstallReport(
            installed: 232,
            preservedFusedGDNInputs: 192,
            stockQuantized: 73)
        try validateQwen38ProductionProjectionInstall(complete)

        for incomplete in [
            Qwen38ProjectionInstallReport(
                installed: 231,
                preservedFusedGDNInputs: 192,
                stockQuantized: 73),
            Qwen38ProjectionInstallReport(
                installed: 232,
                preservedFusedGDNInputs: 191,
                stockQuantized: 73),
            Qwen38ProjectionInstallReport(
                installed: 232,
                preservedFusedGDNInputs: 192,
                stockQuantized: 72),
        ] {
            #expect(throws: Qwen38ProjectionInstallError.self) {
                try validateQwen38ProductionProjectionInstall(incomplete)
            }
        }
    }

    @Test("production installation inventory is path exact")
    func productionInstallInventory() {
        let inventory = qwen38ProductionProjectionInventory()
        #expect(inventory.optimized.count == 232)
        #expect(inventory.preserved.count == 192)
        #expect(inventory.stock.count == 73)
        #expect(inventory.optimized.contains("model.layers.3.self_attn.q_proj"))
        #expect(inventory.preserved.contains("model.layers.0.linear_attn.in_proj_qkv"))
        #expect(inventory.stock.contains("model.layers.56.mlp.down_proj"))
        #expect(inventory.stock.contains("lm_head"))
        #expect(!inventory.optimized.contains("model.layers.0.linear_attn.out_proj"))
    }

    @Test("draft quantization inventory covers every exact linear")
    func draftInstallInventory() {
        let inventory = qwen38DraftProjectionInventory()
        #expect(inventory.count == 47)
        #expect(inventory.contains("fc"))
        #expect(inventory.contains("layers.0.self_attn.q_proj"))
        #expect(inventory.contains("layers.4.mlp_conv.kernel_projection"))
        #expect(inventory.contains("candidate_selector.hidden_projection"))
    }

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
