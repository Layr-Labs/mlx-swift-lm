import Foundation
import XCTest
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

@testable import MLXLLM
@testable import MLXLMCommon

/// Run this filter in an isolated RELEASE test process. Debug builds retain
/// explicit diagnostic routes for engineering and cannot prove this gate.
final class Qwen4ReleasePolicyTests: XCTestCase {
    func testReleaseIgnoresKnownNonExactDiagnosticOverrides() throws {
        #if DEBUG
            throw XCTSkip("Release-only policy gate; run swift test -c release --filter Qwen4ReleasePolicyTests")
        #else
            let names = ["DARKBLOOM_QWEN4_QMV_FORCE_FAST", "DARKBLOOM_QWEN4_QMV_ANY_N",
                         "DARKBLOOM_QWEN4_QMV_LOG_SHAPES"]
            let saved = ProcessInfo.processInfo.environment
            defer {
                for name in names {
                    if let value = saved[name] { setenv(name, value, 1) }
                    else { unsetenv(name) }
                }
                Qwen4ExpEnvironment.refresh()
            }
            setenv(names[0], "1", 1)
            setenv(names[1], "0", 1)
            setenv(names[2], "1", 1)
            Qwen4ExpEnvironment.refresh()
            XCTAssertFalse(Qwen4ExpAffineQMV.logShapes)
            XCTAssertEqual(Qwen4ExpAffineQMV.packsPerThread(inputDim: 512, outputDim: 4, bits: 4), 1)
            XCTAssertTrue(Qwen4ExpAffineQMV.matchesDecodeGeometry(
                tokens: 1, inputDim: 512, outputDim: 4, bits: 4, groupSize: 64))
            setenv(names[0], "k", 1)
            Qwen4ExpEnvironment.refresh()
            XCTAssertEqual(Qwen4ExpAffineQMV.packsPerThread(inputDim: 320, outputDim: 8, bits: 4), 1)
            // Valid stock-fast geometry remains enabled; this is not a
            // blanket disable of the optimized exact kernel.
            XCTAssertEqual(Qwen4ExpAffineQMV.packsPerThread(inputDim: 512, outputDim: 8, bits: 4), 2)
            XCTAssertFalse(Qwen4ExpGatheredQSA.decodeGatherEnabled(
                environment: [Qwen4ExpGatheredQSA.decodeGatherEnvFlag: "1"]))
        #endif
    }
}
