import CoreImage
import Foundation
import MLXLMCommon
import XCTest
@testable import MLXVLM

final class Qwen4ExpProcessorTests: XCTestCase {
    private var data: Data { Data(#"""
        {"size":{"shortest_edge":65536,"longest_edge":16777216},
         "patch_size":16,"temporal_patch_size":2,"merge_size":2,
         "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
         "processor_class":"Qwen3VLProcessor","image_processor_type":"Qwen2VLImageProcessorFast"}
        """#.utf8) }

    func testCheckpointSizesAndFlatOverridePrecedence() throws {
        let config = try JSONDecoder().decode(Qwen4ExpProcessorConfiguration.self, from: data)
        XCTAssertEqual(config.minPixels, 65536)
        XCTAssertEqual(config.maxPixels, 16777216)
        var raw = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        raw["max_pixels"] = 4_194_304
        let overridden = try JSONDecoder().decode(Qwen4ExpProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: raw))
        XCTAssertEqual(overridden.maxPixels, 4_194_304)
        let restored = try JSONDecoder().decode(Qwen4ExpProcessorConfiguration.self,
            from: JSONEncoder().encode(overridden))
        XCTAssertEqual(restored.minPixels, overridden.minPixels)
        XCTAssertEqual(restored.maxPixels, overridden.maxPixels)
        raw.removeValue(forKey: "size")
        raw.removeValue(forKey: "max_pixels")
        let defaults = try JSONDecoder().decode(Qwen4ExpProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: raw))
        XCTAssertEqual(defaults.minPixels, 56 * 56)
        XCTAssertEqual(defaults.maxPixels, 28 * 28 * 1280)
    }

    func testNativeRegistryAndImageGridDoNotInherit1280TokenCap() async throws {
        XCTAssertEqual(VLMModelFactory.processorType(modelType: "qwen4_exp", declaredClass: "Qwen3VLProcessor"),
                       "Qwen4ExpProcessor")
        XCTAssertEqual(VLMModelFactory.processorType(modelType: "qwen3_5", declaredClass: "Qwen3VLProcessor"),
                       "Qwen3VLProcessor")
        XCTAssertEqual(VLMModelFactory.processorType(modelType: "mistral3", declaredClass: "PixtralProcessor"),
                       "Mistral3Processor")
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: data, processorType: "Qwen4ExpProcessor", tokenizer: ImageTokenFixture())
        XCTAssertTrue(processor is Qwen4ExpProcessor)
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 2240, height: 1120))
        let input = UserInput(prompt: "Read the image", images: [.ciImage(image)])
        let native = try await processor.prepare(input: input)
        let grid = try XCTUnwrap(native.image?.frames?.first)
        XCTAssertEqual(grid.h, 70)
        XCTAssertEqual(grid.w, 140)
        XCTAssertEqual(grid.product / 4, 2450)
        XCTAssertEqual(native.text.tokens.size, 2452)

        let legacyConfig = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: data)
        let legacy = try await Qwen3VLProcessor(legacyConfig, tokenizer: ImageTokenFixture()).prepare(input: input)
        XCTAssertLessThanOrEqual(try XCTUnwrap(legacy.image?.frames?.first).product / 4, 1280,
                                "Older model serving policy must not change")
        var explicit = input
        explicit.processing.maxPixels = 1280 * 32 * 32
        let limited = try await processor.prepare(input: explicit)
        XCTAssertLessThanOrEqual(try XCTUnwrap(limited.image?.frames?.first).product / 4, 1280,
                                "Explicit caller processing choices remain supported")
    }

    func testMalformedSettingsFailClosed() throws {
        for (key, value): (String, Any) in [("image_std", [0.5, 0.0, 0.5]),
            ("image_mean", [0.5]), ("patch_size", 0), ("max_pixels", 100),
            ("size", ["longest_edge": 16777216]), ("image_processor_type", "UnrelatedProcessor")] {
            var raw = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            raw[key] = value
            XCTAssertThrowsError(try JSONDecoder().decode(Qwen4ExpProcessorConfiguration.self,
                from: JSONSerialization.data(withJSONObject: raw)))
        }
    }
}

private struct ImageTokenFixture: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [1] + Array(repeating: 7, count: text.components(separatedBy: "<|image_pad|>").count - 1) + [2]
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                           additionalContext: [String: any Sendable]?) throws -> [Int] { [1,7,2] }
}
