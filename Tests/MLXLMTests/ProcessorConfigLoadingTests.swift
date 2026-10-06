import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite
struct ProcessorConfigLoadingTests {
    private func data(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func makeDirectory(_ files: [String: Data]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "processor-loading-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        do {
            for (name, contents) in files {
                try contents.write(to: directory.appendingPathComponent(name))
            }
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
        return directory
    }

    @Test(arguments: ["preprocessor_config.json", "processor_config.json"])
    func singleFileIsReturnedUnchanged(filename: String) async throws {
        let original = Data("{\"processor_class\": \"Gemma4Processor\", \"patch_size\": 16}".utf8)
        let directory = try makeDirectory([filename: original])
        defer { try? FileManager.default.removeItem(at: directory) }
        let (loaded, config) = try await loadProcessorConfig(from: directory)
        #expect(loaded == original)
        #expect(config.processorClass == "Gemma4Processor")
    }

    @Test func primaryNullAndNestedValuesAreNotRecursivelyFilled() throws {
        let primary = try data([
            "processor_class": "Qwen3VLProcessor", "image_seq_len": NSNull(),
            "image_processor": ["patch_size": 16],
        ])
        let secondary = try data([
            "processor_class": "OtherProcessor", "image_seq_len": 81,
            "image_processor": ["patch_size": 32, "merge_size": 4],
            "chat_template": "template",
        ])
        let merged = try fillingMissingProcessorKeys(of: primary, from: secondary)
        let object = try #require(JSONSerialization.jsonObject(with: merged) as? [String: Any])
        #expect(object["processor_class"] as? String == "Qwen3VLProcessor")
        #expect(object["image_seq_len"] is NSNull)
        let nested = try #require(object["image_processor"] as? [String: Any])
        #expect(nested["patch_size"] as? Int == 16)
        #expect(nested["merge_size"] == nil)
        #expect(object["chat_template"] as? String == "template")
    }

    @Test func supplementalNestedQwenFieldsDoNotOverrideFlatConfiguration() async throws {
        let primary = try data([
            "processor_class": "Qwen3VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
            "min_pixels": 3136, "max_pixels": 12_845_056,
        ])
        let secondary = try data([
            "processor_class": "Qwen3VLProcessor",
            "image_processor": ["patch_size": 32, "merge_size": 4],
            "video_processor": ["temporal_patch_size": 4],
        ])
        let directory = try makeDirectory([
            "preprocessor_config.json": primary, "processor_config.json": secondary,
        ])
        defer { try? FileManager.default.removeItem(at: directory) }
        let (merged, base) = try await loadProcessorConfig(from: directory)
        let before = try JSONDecoder.json5().decode(
            Qwen3VLProcessorConfiguration.self, from: primary)
        let after = try JSONDecoder.json5().decode(Qwen3VLProcessorConfiguration.self, from: merged)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        #expect(try encoder.encode(before) == encoder.encode(after))
        #expect(base.processorClass == "Qwen3VLProcessor")
        #expect(after.patchSize == 16)
        #expect(after.mergeSize == 2)
        #expect(after.temporalPatchSize == 2)
    }

    @Test(arguments: ["{", "[]", "null", "42"])
    func malformedSupplementThrowsEvenWhenPrimaryIsComplete(contents: String) async throws {
        let directory = try makeDirectory([
            "preprocessor_config.json": try data(["processor_class": "Gemma4Processor"]),
            "processor_config.json": Data(contents.utf8),
        ])
        defer { try? FileManager.default.removeItem(at: directory) }
        await #expect(throws: (any Error).self) {
            _ = try await loadProcessorConfig(from: directory)
        }
    }

    @Test func missingFilesThrow() async throws {
        let directory = try makeDirectory([:])
        defer { try? FileManager.default.removeItem(at: directory) }
        await #expect(throws: (any Error).self) {
            _ = try await loadProcessorConfig(from: directory)
        }
    }

    @Test(arguments: ["preprocessor_config.json", "processor_config.json"])
    func malformedConfigurationReportsItsOwnFilename(filename: String) async throws {
        let valid = try data(["processor_class": "Gemma4Processor"])
        var files = ["preprocessor_config.json": valid, "processor_config.json": valid]
        files[filename] = Data("{".utf8)
        let directory = try makeDirectory(files)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await loadProcessorConfig(from: directory)
            Issue.record("Malformed configuration should throw")
        } catch let error as ProcessorConfigError {
            #expect(error.filename == filename)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test func concurrentLoadsLeaveConfigurationFilesUnchanged() async throws {
        let primary = try data(["processor_class": "Idefics3Processor", "image_seq_len": 16])
        let secondary = try data(["processor_class": "OtherProcessor", "chat_template": "t"])
        let directory = try makeDirectory([
            "preprocessor_config.json": primary, "processor_config.json": secondary,
        ])
        defer { try? FileManager.default.removeItem(at: directory) }
        let expected = try fillingMissingProcessorKeys(of: primary, from: secondary)
        try await withThrowingTaskGroup(of: Data.self) { group in
            for _ in 0 ..< 32 {
                group.addTask {
                    let (loaded, _) = try await loadProcessorConfig(from: directory)
                    return loaded
                }
            }
            for try await loaded in group {
                #expect(loaded == expected)
            }
        }
        #expect(
            try Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json"))
                == primary)
        #expect(
            try Data(contentsOf: directory.appendingPathComponent("processor_config.json"))
                == secondary)
    }
}
