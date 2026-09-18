import XCTest

final class Qwen4RealArtifactProfileTests: XCTestCase {
    func testDefaultRetainsOriginalTargetAndUnknownProfilesFailClosed() throws {
        XCTAssertEqual(try Qwen4RealArtifactProfile.requested(environment: [:]), .affineQ4)
        XCTAssertEqual(try Qwen4RealArtifactProfile.requested(environment: [
            Qwen4RealArtifactProfile.environmentKey: "selected-q4"
        ]), .selectedQ4)
        for value in ["", "selected", "SELECTED-Q4", "qwen3_5"] {
            XCTAssertThrowsError(try Qwen4RealArtifactProfile.requested(environment: [
                Qwen4RealArtifactProfile.environmentKey: value
            ]))
        }
    }

    func testProfilesCannotQualifyEachOthersArtifacts() {
        for expected in Qwen4RealArtifactProfile.allCases {
            for actual in Qwen4RealArtifactProfile.allCases {
                XCTAssertEqual(expected.matches(configSHA256: actual.configSHA256,
                    indexSHA256: actual.indexSHA256, tensorCount: actual.tensorCount,
                    shardCount: actual.shardCount, mtpTensorCount: actual.mtpTensorCount),
                    expected == actual)
            }
        }
    }

    func testEachIdentityComponentIsMandatory() {
        for profile in Qwen4RealArtifactProfile.allCases {
            XCTAssertFalse(profile.matches(configSHA256: String(repeating: "0", count: 64),
                indexSHA256: profile.indexSHA256, tensorCount: profile.tensorCount,
                shardCount: profile.shardCount, mtpTensorCount: profile.mtpTensorCount))
            XCTAssertFalse(profile.matches(configSHA256: profile.configSHA256,
                indexSHA256: String(repeating: "0", count: 64), tensorCount: profile.tensorCount,
                shardCount: profile.shardCount, mtpTensorCount: profile.mtpTensorCount))
            XCTAssertFalse(profile.matches(configSHA256: profile.configSHA256,
                indexSHA256: profile.indexSHA256, tensorCount: profile.tensorCount - 1,
                shardCount: profile.shardCount, mtpTensorCount: profile.mtpTensorCount))
            XCTAssertFalse(profile.matches(configSHA256: profile.configSHA256,
                indexSHA256: profile.indexSHA256, tensorCount: profile.tensorCount,
                shardCount: profile.shardCount + 1, mtpTensorCount: profile.mtpTensorCount))
            XCTAssertFalse(profile.matches(configSHA256: profile.configSHA256,
                indexSHA256: profile.indexSHA256, tensorCount: profile.tensorCount,
                shardCount: profile.shardCount, mtpTensorCount: profile.mtpTensorCount - 1))
        }
    }
}
