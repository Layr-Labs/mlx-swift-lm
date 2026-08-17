import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2AttentionExecutionPolicyTests: XCTestCase {
    private func qualification(
        automatic: Bool
    ) -> CBv2AttentionExecutionQualification {
        CBv2AttentionExecutionQualification(
            architecture: .qwenLike,
            automaticOptimization: automatic
                ? .forcedFusedD256BF16FullAttentionPrefill : nil)
    }

    private func kind(
        qualification: CBv2AttentionExecutionQualification? = nil
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full,
            headDim: 256,
            kvHeads: 2,
            queryHeads: 4,
            attentionExecutionQualification: qualification)
    }

    private func route(
        control: CBv2AttentionExecutionControl,
        kind: CBv2LayerKind,
        queryLength: Int = 256,
        queryDType: DType = .bfloat16,
        attentionSoftcap: Float? = nil,
        hasSpanMask: Bool = false,
        serializesQueries: Bool = false
    ) -> CBv2AttentionExecutionRoute {
        CBv2AttentionExecutionPolicy(control: control).route(
            kind: kind,
            queryLength: queryLength,
            queryDType: queryDType,
            attentionSoftcap: attentionSoftcap,
            hasSpanMask: hasSpanMask,
            serializesQueries: serializesQueries)
    }

    func testDefaultAndInvalidControlsFailClosedToFallback() {
        XCTAssertEqual(CBv2AttentionExecutionControl.parse(nil), .fallback)
        XCTAssertEqual(CBv2AttentionExecutionControl.parse(""), .fallback)
        XCTAssertEqual(CBv2AttentionExecutionControl.parse("unknown"), .fallback)
        XCTAssertEqual(CBv2AttentionExecutionControl.parse(" FUSED "), .fused)

        XCTAssertEqual(
            route(
                control: .fallback,
                kind: kind(qualification: qualification(automatic: true))),
            .fallback)
    }

    func testFusedAndAutoHaveDistinctQualificationContracts() {
        let unknown = kind()
        let architectureOnly = kind(qualification: qualification(automatic: false))
        let automaticallyQualified = kind(qualification: qualification(automatic: true))

        XCTAssertEqual(route(control: .fused, kind: unknown), .fallback)
        XCTAssertEqual(route(control: .auto, kind: unknown), .fallback)

        XCTAssertEqual(route(control: .fused, kind: architectureOnly), .forcedFused)
        XCTAssertEqual(route(control: .auto, kind: architectureOnly), .fallback)

        XCTAssertEqual(route(control: .fused, kind: automaticallyQualified), .forcedFused)
        XCTAssertEqual(route(control: .auto, kind: automaticallyQualified), .forcedFused)
    }

    func testForcedFusedRequiresExactQwenD256BF16FullPrefill() {
        let eligible = kind(qualification: qualification(automatic: true))
        XCTAssertEqual(route(control: .fused, kind: eligible), .forcedFused)

        var wrongHeadDimension = eligible
        wrongHeadDimension.headDim = 128
        XCTAssertEqual(route(control: .fused, kind: wrongHeadDimension), .fallback)

        var sliding = eligible
        sliding.attention = .slidingWindow(4096)
        XCTAssertEqual(route(control: .fused, kind: sliding), .fallback)

        var bidirectional = eligible
        bidirectional.isBidirectional = true
        XCTAssertEqual(route(control: .fused, kind: bidirectional), .fallback)

        var withSinks = eligible
        withSinks.hasSinks = true
        XCTAssertEqual(route(control: .fused, kind: withSinks), .fallback)

        var borrowing = eligible
        borrowing.sharesKVWithLayer = 0
        XCTAssertEqual(route(control: .fused, kind: borrowing), .fallback)

        XCTAssertEqual(
            route(control: .fused, kind: eligible, queryDType: .float16),
            .fallback)
        XCTAssertEqual(
            route(control: .fused, kind: eligible, attentionSoftcap: 30),
            .fallback)
        XCTAssertEqual(
            route(control: .fused, kind: eligible, hasSpanMask: true),
            .fallback)
    }

    func testQueryBlockingIsBypassedOnlyForForcedFusedRoute() {
        let policy = CBv2AttentionExecutionPolicy(control: .fused)
        let fallbackRoute = route(control: .fused, kind: kind())
        let fusedRoute = route(
            control: .fused,
            kind: kind(qualification: qualification(automatic: false)))

        XCTAssertTrue(
            policy.shouldBlockQueries(
                queryLength: 256, blockSize: 128, route: fallbackRoute))
        XCTAssertFalse(
            policy.shouldBlockQueries(
                queryLength: 256, blockSize: 128, route: fusedRoute))
        XCTAssertFalse(
            policy.shouldBlockQueries(
                queryLength: 128, blockSize: 128, route: fallbackRoute))
        XCTAssertFalse(
            policy.shouldBlockQueries(
                queryLength: 256, blockSize: 0, route: fallbackRoute))
    }

    func testDecodeAndMTPRectanglesNeverSelectForcedFused() {
        let eligible = kind(qualification: qualification(automatic: true))

        XCTAssertEqual(
            route(control: .fused, kind: eligible, queryLength: 1),
            .fallback)
        XCTAssertEqual(
            route(
                control: .fused,
                kind: eligible,
                queryLength: 4,
                serializesQueries: true),
            .fallback)
    }

    func testEligibleRoutePreservesContiguousAndPagedCausalMasks() {
        let eligible = kind(qualification: qualification(automatic: true))
        XCTAssertEqual(route(control: .fused, kind: eligible), .forcedFused)

        let contiguous = CBv2AttentionV1.maskMode(L: 2, kL: 4, window: nil)
        guard case .causal = contiguous else {
            return XCTFail("eligible contiguous prefill must retain the symbolic causal mask")
        }

        XCTAssertEqual(PagedLayerCache.prefillMaskContract, .absoluteArray)
    }
}
