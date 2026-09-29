import XCTest
@testable import MLXLMCommon

final class MiMoV26PrefillPolicyTests: XCTestCase {
    func testUnsetAndExplicitControls() {
        XCTAssertTrue(MiMoV26PrefillPolicy.isEnabled(nil))
        for value in ["1", "true", "yes", "on", "auto", " TRUE "] {
            XCTAssertTrue(MiMoV26PrefillPolicy.isEnabled(value), value)
        }
        for value in ["0", "false", "off", "no", "", "garbage"] {
            XCTAssertFalse(MiMoV26PrefillPolicy.isEnabled(value), value)
        }
    }

    func testWiderCandidatesRequireExplicitDefaultRequestAndNAX() {
        func widths(_ requested: Bool, _ nax: Bool, _ memory: UInt64 = 256 << 30,
                    _ maximum: Int? = 8192, _ original: Int = 2048) -> [Int] {
            MiMoV26PrefillPolicy.widerWidths(requested: requested, naxAvailable: nax,
                physicalMemoryBytes: memory, maximumTokens: maximum, originalWidth: original)
        }
        XCTAssertEqual(widths(true, true), [8192, 4096])
        XCTAssertEqual(widths(false, true), []) // explicit SDK scheduler
        XCTAssertEqual(widths(true, false), []) // M3/no NAX
        XCTAssertEqual(widths(true, true, 64 << 30), [])
        XCTAssertEqual(widths(true, true, 256 << 30, nil), []) // unsupported model
        XCTAssertEqual(widths(true, true, 256 << 30, 4096), [4096]) // no oversized gather
        XCTAssertEqual(widths(true, true, 256 << 30, 8192, 4096), [8192])
        XCTAssertEqual(widths(true, true, 256 << 30, 8192, 8192), [])
    }

    func testMTPRepricingPreservesEveryOtherFixedCharge() {
        XCTAssertEqual(MiMoV26PrefillPolicy.replacingMTPCharge(
            current: 1000, previous: 300, replacement: 800), 1500)
        XCTAssertEqual(MiMoV26PrefillPolicy.replacingMTPCharge(
            current: 1000, previous: 300, replacement: 300), 1000)
        XCTAssertNil(MiMoV26PrefillPolicy.replacingMTPCharge(
            current: 1000, previous: 300, replacement: 299))
        XCTAssertNil(MiMoV26PrefillPolicy.replacingMTPCharge(
            current: 100, previous: 300, replacement: 800))
        XCTAssertNil(MiMoV26PrefillPolicy.replacingMTPCharge(
            current: Int.max, previous: 1, replacement: 2))
    }

    func testCapacityRefusalPreservesOriginalMTPAndCallerCharges() throws {
        // 300 MTP + 700 caller/target; a wide candidate changes only its copy.
        let original = AdmissionV2.Config(fixedBytesPerRequest: 1000)
        var wide = original
        wide.fixedBytesPerRequest = try XCTUnwrap(MiMoV26PrefillPolicy.replacingMTPCharge(
            current: original.fixedBytesPerRequest, previous: 300, replacement: 800))
        XCTAssertNil(MiMoV26PrefillPolicy.addingGroupedCharge(
            to: wide, scratchBytes: 500, capacityBytes: 2000)) // equality refuses
        XCTAssertNil(MiMoV26PrefillPolicy.addingGroupedCharge(
            to: wide, scratchBytes: Int.max, capacityBytes: Int.max))
        XCTAssertEqual(original.fixedBytesPerRequest, 1000)
        let fallback = try XCTUnwrap(MiMoV26PrefillPolicy.addingGroupedCharge(
            to: original, scratchBytes: 100, capacityBytes: 2000))
        XCTAssertEqual(fallback.fixedBytesPerRequest, 1100)
        XCTAssertEqual(original.fixedBytesPerRequest, 1000)
    }
}
