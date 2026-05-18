import Foundation
@testable import MLXLLM
import MLXLMCommon
import XCTest

final class LLMModelFactoryCompatibilityTests: XCTestCase {

    func testJambaLegacyRegistryAliasStillExists() {
        XCTAssertEqual(LLMRegistry.jamba_3b.name, LLMRegistry.jamba_3b_4bit.name)
    }

    func testFactorySelectsParoQuantLoaderForParoQuantConfig() throws {
        let directory = try makeTemporaryModelDirectory(configJSON: """
            {
              "model_type": "qwen3_5",
              "architectures": ["Qwen3_5ForConditionalGeneration"],
              "quantization_config": {
                "quant_method": "paroquant",
                "bits": 4,
                "group_size": 128,
                "krot": 8
              }
            }
            """)

        let configuration = ResolvedModelConfiguration(directory: directory)
        XCTAssertTrue(LLMModelFactory.shouldUseParoQuantLoader(configuration: configuration))
    }

    func testFactoryDoesNotSelectParoQuantLoaderForRegularConfig() throws {
        let directory = try makeTemporaryModelDirectory(configJSON: """
            {
              "model_type": "qwen3_5",
              "architectures": ["Qwen3_5ForConditionalGeneration"]
            }
            """)

        let configuration = ResolvedModelConfiguration(directory: directory)
        XCTAssertFalse(LLMModelFactory.shouldUseParoQuantLoader(configuration: configuration))
    }

    private func makeTemporaryModelDirectory(configJSON: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try configJSON.data(using: .utf8)!.write(to: directory.appendingPathComponent("config.json"))
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return directory
    }
}
