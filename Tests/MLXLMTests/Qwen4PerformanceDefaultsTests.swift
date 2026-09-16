import XCTest

@testable import MLXLLM

final class Qwen4PerformanceDefaultsTests: XCTestCase {
    func testAbsentFlagsEnableOnlyQualifiedFullKVAndLayerProfile() {
        XCTAssertTrue(Qwen4ExpParallelQSA.fullKVEnabled(environment: [:]))
        XCTAssertTrue(Qwen4ExpLayerSubmission.enabled(environment: [:]))
        XCTAssertFalse(Qwen4ExpParallelQSA.enabled(environment: [:]))
        XCTAssertFalse(Qwen4ExpParallelQSA.enabled(environment: [Qwen4ExpParallelQSA.fullKVFlag: "1"]))
        XCTAssertTrue(Qwen4ExpParallelQSA.enabled(environment: [Qwen4ExpParallelQSA.flag: "1"]))
    }

    func testExplicitOptOutAndExistingStrictFlagParsingRemainAvailable() {
        for value in ["0", "false", "off", "no", "", "true", "yes", "invalid"] {
            XCTAssertFalse(Qwen4ExpParallelQSA.fullKVEnabled(
                environment: [Qwen4ExpParallelQSA.fullKVFlag: value]))
            XCTAssertFalse(Qwen4ExpLayerSubmission.enabled(
                environment: [Qwen4ExpLayerSubmission.flag: value]))
        }
        XCTAssertTrue(Qwen4ExpParallelQSA.fullKVEnabled(
            environment: [Qwen4ExpParallelQSA.fullKVFlag: "1"]))
        XCTAssertTrue(Qwen4ExpLayerSubmission.enabled(
            environment: [Qwen4ExpLayerSubmission.flag: "1"]))
        XCTAssertFalse(Qwen4ExpParallelQSA.fullKVEnabled(environment: [
            Qwen4ExpParallelQSA.flag: "1", Qwen4ExpParallelQSA.fullKVFlag: "0"]))
    }

    func testAbsentValuePartitionDefaultIs32ForFullKVOnly() {
        for fallback in [1, 2, 4] {
            XCTAssertEqual(Qwen4ExpParallelQSA.valuePartitions(
                requested: nil, fallback: fallback, compactKV: false, environment: [:]), 32)
            XCTAssertEqual(Qwen4ExpParallelQSA.valuePartitions(
                requested: nil, fallback: fallback, compactKV: true, environment: [:]), fallback)
        }
    }

    func testPartitionOverridesAndInvalidFallbackRemainUnchanged() {
        let flag = Qwen4ExpParallelQSA.valuePartitionsFlag
        for compactKV in [false, true] {
            for partitions in [1, 2, 4, 8, 16, 32] {
                XCTAssertEqual(Qwen4ExpParallelQSA.valuePartitions(
                    requested: nil, fallback: 2, compactKV: compactKV,
                    environment: [flag: String(partitions)]), partitions)
                XCTAssertEqual(Qwen4ExpParallelQSA.valuePartitions(
                    requested: partitions, fallback: 2, compactKV: compactKV,
                    environment: [flag: "16"]), partitions)
            }
            for invalid in ["", "0", "3", "64", "invalid"] {
                XCTAssertEqual(Qwen4ExpParallelQSA.valuePartitions(
                    requested: nil, fallback: 2, compactKV: compactKV,
                    environment: [flag: invalid]), 2)
            }
            XCTAssertEqual(Qwen4ExpParallelQSA.valuePartitions(
                requested: 3, fallback: 2, compactKV: compactKV, environment: [flag: "32"]), 2)
        }
    }
}
