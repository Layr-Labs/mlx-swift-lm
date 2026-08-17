import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2AttentionExecutionPolicyTests: XCTestCase {
    private let hardwareQualification =
        CBv2AttentionHardwareQualification
        .qwenLikeD256Hq16Hkv2GQA8BF16FullAttentionPrefill

    private func qualification() -> CBv2AttentionExecutionQualification {
        CBv2AttentionExecutionQualification(architecture: .qwenLike)
    }

    private func kind(
        qualification: CBv2AttentionExecutionQualification? = nil
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full,
            headDim: 256,
            kvHeads: 2,
            queryHeads: 16,
            attentionExecutionQualification: qualification)
    }

    private func route(
        control: CBv2AttentionExecutionControl,
        hardwareQualified: Bool = false,
        kind: CBv2LayerKind,
        queryLength: Int = 256,
        queryDType: DType = .bfloat16,
        attentionSoftcap: Float? = nil,
        hasSpanMask: Bool = false,
        serializesQueries: Bool = false
    ) -> CBv2AttentionExecutionRoute {
        CBv2AttentionExecutionPolicy(
            control: control,
            hardwareQualification: hardwareQualified ? hardwareQualification : nil
        ).contiguousRoute(
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
            route(control: .fallback, kind: kind(qualification: qualification())),
            .fallback)
    }

    func testAutoRequiresArchitectureAndExternalHardwareQualification() {
        let unknownArchitecture = kind()
        let qwenLike = kind(qualification: qualification())

        XCTAssertEqual(
            route(
                control: .auto,
                hardwareQualified: true,
                kind: unknownArchitecture),
            .fallback)
        XCTAssertEqual(route(control: .auto, kind: qwenLike), .fallback)
        XCTAssertEqual(route(control: .fused, kind: qwenLike), .forcedFused)
        XCTAssertEqual(
            route(control: .auto, hardwareQualified: true, kind: qwenLike),
            .forcedFused)
    }

    func testForcedFusedRequiresExactQwenD256Hq16Hkv2GQA8BF16FullPrefill() {
        let eligible = kind(qualification: qualification())
        XCTAssertEqual(route(control: .fused, kind: eligible), .forcedFused)

        var wrongHeadDimension = eligible
        wrongHeadDimension.headDim = 128
        XCTAssertEqual(route(control: .fused, kind: wrongHeadDimension), .fallback)

        var wrongQueryHeads = eligible
        wrongQueryHeads.queryHeads = 4
        XCTAssertEqual(route(control: .fused, kind: wrongQueryHeads), .fallback)

        var wrongKVHeads = eligible
        wrongKVHeads.kvHeads = 1
        XCTAssertEqual(route(control: .fused, kind: wrongKVHeads), .fallback)

        var wrongGQA = eligible
        wrongGQA.kvHeads = 4
        XCTAssertEqual(route(control: .fused, kind: wrongGQA), .fallback)

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
            kind: kind(qualification: qualification()))

        XCTAssertTrue(
            policy.shouldBlockQueries(
                queryLength: 256, blockSize: 128, route: fallbackRoute))
        XCTAssertFalse(
            policy.shouldBlockQueries(
                queryLength: 256, blockSize: 128, route: fusedRoute))
        XCTAssertFalse(
            policy.shouldBlockQueries(
                queryLength: 128, blockSize: 128, route: fallbackRoute))
    }

    func testDecodeAndMTPRectanglesNeverSelectForcedFused() {
        let eligible = kind(qualification: qualification())

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
}
