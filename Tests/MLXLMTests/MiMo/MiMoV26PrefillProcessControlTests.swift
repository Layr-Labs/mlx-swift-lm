import Foundation
import XCTest
@testable import MLXLMCommon

final class MiMoV26PrefillProcessControlTests: XCTestCase {
    func testInjectedControlTruthTableUsesSuppliedLatchedSnapshot() throws {
        for attention in [false, true] {
            for grouped in [false, true] {
                try MiMoV26PrefillPolicy.validateInjectedControls(environment: [:],
                    latchedAttention: attention, latchedGrouping: grouped)
                for (key, actual) in [
                    (MiMoV26PrefillPolicy.attentionEnvironmentKey, attention),
                    (MiMoV26PrefillPolicy.groupedEnvironmentKey, grouped)] {
                    for value in ["0", "1", "off", "on", "auto", "malformed"] {
                        let requested = MiMoV26PrefillPolicy.isEnabled(value)
                        if requested == actual {
                            try MiMoV26PrefillPolicy.validateInjectedControls(environment: [key: value],
                                latchedAttention: attention, latchedGrouping: grouped)
                        } else {
                            XCTAssertThrowsError(try MiMoV26PrefillPolicy.validateInjectedControls(
                                environment: [key: value], latchedAttention: attention,
                                latchedGrouping: grouped)) { error in
                                XCTAssertEqual(error as? MiMoV26PrefillPolicy.ProcessControlMismatch,
                                    .incompatible(environmentKey: key, requested: requested, latched: actual))
                            }
                        }
                    }
                }
            }
        }
    }

    func testPublicValidatorUsesActualDispatcherConstants() throws {
        XCTAssertEqual(MiMoV26PrefillPolicy.latchedAttentionEnabled, MiMoV26NAXAttention.requested)
        XCTAssertEqual(MiMoV26PrefillPolicy.latchedGroupingEnabled, MiMoV26BlockBatchAttention.requested)
        try MiMoV26PrefillPolicy.validateProcessControls(environment: [:])
        try MiMoV26PrefillPolicy.validateProcessControls(environment: [
            MiMoV26PrefillPolicy.attentionEnvironmentKey: MiMoV26NAXAttention.requested ? "1" : "0",
            MiMoV26PrefillPolicy.groupedEnvironmentKey: MiMoV26BlockBatchAttention.requested ? "1" : "0"])
    }

    func testFreshProcessStartupZeroRemainsActualRollback() throws {
        // Run this selector in its own process with both original controls=0.
        // Never setenv after static initialization to simulate startup rollback.
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment[MiMoV26PrefillPolicy.attentionEnvironmentKey] == "0"
            && environment[MiMoV26PrefillPolicy.groupedEnvironmentKey] == "0",
            "requires a separate process started with both prefill kernel controls=0")
        XCTAssertFalse(MiMoV26NAXAttention.requested)
        XCTAssertFalse(MiMoV26BlockBatchAttention.requested)
        try MiMoV26PrefillPolicy.validateProcessControls(environment: [
            MiMoV26PrefillPolicy.attentionEnvironmentKey: "0",
            MiMoV26PrefillPolicy.groupedEnvironmentKey: "0"])
        XCTAssertThrowsError(try MiMoV26PrefillPolicy.validateProcessControls(environment: [
            MiMoV26PrefillPolicy.attentionEnvironmentKey: "1"]))
    }
}
