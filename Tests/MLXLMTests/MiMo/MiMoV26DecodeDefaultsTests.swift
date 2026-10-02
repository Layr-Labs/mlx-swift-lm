// Copyright © 2026 Eigen Labs.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class MiMoV26DecodeDefaultsTests: XCTestCase {
    func testShortForwardKernelsDefaultOnWithRollbackValues() {
        for key in MiMoV26DecodeDefaults.environmentKeys {
            XCTAssertTrue(MiMoV26DecodeDefaults.isEnabled(key, environment: [:]), key)
            for value in ["1", "true", "yes", "on", ""] {
                XCTAssertTrue(
                    MiMoV26DecodeDefaults.isEnabled(key, environment: [key: value]),
                    "\(key)=\(value)")
            }
            for value in ["0", "false", "no", "off", " OFF ", "False", "\tno\n"] {
                XCTAssertFalse(
                    MiMoV26DecodeDefaults.isEnabled(key, environment: [key: value]),
                    "\(key)=\(value)")
            }
        }
        XCTAssertEqual(
            MiMoV26DecodeDefaults.environmentKeys,
            [
                "DARKBLOOM_MIMO_FUSED_DECODE_NORMS", "DARKBLOOM_MIMO_DECODE_EXPERTS",
                "DARKBLOOM_MIMO_DECODE_ROUTER_GEMV",
            ])
    }

    func testExactRectangularVerifySwitchesDefaultOnWithRollbackValues() {
        XCTAssertEqual(
            MiMoV26DecodeDefaults.verifyEnvironmentKeys,
            ["DARKBLOOM_MIMO_RECTANGULAR_SCALAR_DENSE", "DARKBLOOM_MIMO_ROW_EXACT_PROJECTION"])
        let key = MiMoV26DecodeDefaults.scalarDenseVerifyKey
        XCTAssertTrue(MiMoV26RectangularDense.enabled(environment: [:]))
        XCTAssertTrue(MiMoV26RectangularDense.enabled(environment: [key: "1"]))
        for value in ["0", "false", "no", "off", " Off "] {
            XCTAssertFalse(MiMoV26RectangularDense.enabled(environment: [key: value]), value)
        }
    }

    func testFusedNormsStayOnWithTheScalarDenseVerifier() {
        let norms = MiMoV26DecodeDefaults.fusedNormsKey
        let scalarDense = "DARKBLOOM_MIMO_RECTANGULAR_SCALAR_DENSE"
        XCTAssertTrue(MiMoV26TextBackbone.fusedDecodeNormsEnabled(environment: [:]))
        XCTAssertTrue(MiMoV26TextBackbone.fusedDecodeNormsEnabled(environment: [scalarDense: "1"]))
        XCTAssertFalse(
            MiMoV26TextBackbone.fusedDecodeNormsEnabled(environment: [scalarDense: "1", norms: "0"])
        )
    }
}
