import Foundation
import Testing

@testable import BenchCBv2Core

extension UnitTests {

    @Suite("Window lifetime control model configuration")
    struct BenchWindowLifetimeConfigurationTests {
        @Test func nativeGemmaAndGPTFamiliesAreAcceptedBeforeModelLoad() throws {
            for family in ["gemma4", "gemma4_text", "gpt_oss"] {
                let data = try JSONSerialization.data(withJSONObject: ["model_type": family])
                try validateWindowLifetimeModelConfiguration(data)
            }
        }

        @Test func nonObjectConfigurationIsAReportedError() {
            #expect(throws: (any Error).self) {
                try validateWindowLifetimeModelConfiguration(Data("[]".utf8))
            }
        }

        @Test func otherFamiliesAndMissingTypeAreRejected() {
            for json in ["{}", "{\"model_type\":\"mimo_v2\"}", "{\"model_type\":\"unknown\"}"] {
                #expect(throws: (any Error).self) {
                    try validateWindowLifetimeModelConfiguration(Data(json.utf8))
                }
            }
        }
    }
}
