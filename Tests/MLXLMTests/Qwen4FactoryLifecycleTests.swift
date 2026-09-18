import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// These tests execute real miniature MLX loads/forwards and require an
/// exclusive GPU lane. A loaded full vision tower remains a separate gate.
final class Qwen4FactoryLifecycleTests: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        guard Qwen4ExpPLEResidency.useMmap else { throw XCTSkip("Requires default mmap PLE") }
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0, "A live model owns the lane")
    }

    private func snapshot() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-factory-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        try Qwen4FactoryFixture.write(to: directory)
        return directory
    }

    private func load(_ directory: URL, vision: Bool, failTokenizer: Bool = false)
        async throws -> ModelContainer
    {
        let tokenizer = Qwen4FactoryFixture.Loader(fail: failTokenizer)
        if vision {
            return try await VLMModelFactory.shared.loadContainer(from: directory, using: tokenizer)
        }
        return try await LLMModelFactory.shared.loadContainer(from: directory, using: tokenizer)
    }

    func testBothFactoriesBindBeforeConstructAndReleaseOnTeardown() async throws {
        let directory = try snapshot()
        for vision in [false, true] {
            var container: ModelContainer? = try await load(directory, vision: vision)
            XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 1)
            try await container!.perform { context in
                try XCTUnwrap(context.model as? any Qwen4ExpExternalPLEValidating)
                    .validateExternalPLEResources()
                let ids = MLXArray([Int32(1), 2, 3], [1, 3])
                let logits = context.model(ids, cache: context.model.newCache(parameters: nil))
                eval(logits)
                XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
                XCTAssertEqual(logits.shape, [1, 3, 64])
            }
            weak var weakContainer = container
            container = nil
            XCTAssertNil(weakContainer)
            XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
            XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
            XCTAssertEqual(Qwen4ExpPLEResourceMetrics.snapshot().mappedFiles, 0)
        }
    }

    func testFailedTokenizerAndMissingPLEIndexReleaseBothFactoryOwners() async throws {
        for vision in [false, true] {
            let directory = try snapshot()
            do {
                _ = try await load(directory, vision: vision, failTokenizer: true)
                XCTFail("Tokenizer failure must escape the loader")
            } catch Qwen4FactoryFixture.Loader.Failure.tokenizer { }
            XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
            XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
            try FileManager.default.removeItem(at: directory.appendingPathComponent("model.safetensors.index.json"))
            do {
                _ = try await load(directory, vision: vision)
                XCTFail("Missing PLE resources must fail at load, not first forward")
            } catch { }
            XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
            XCTAssertEqual(Qwen4ExpPLEResourceMetrics.snapshot().mappedFiles, 0)
        }
    }

    func testFailedProcessorAndWeightLoadsLeaveNoBinding() async throws {
        let directory = try snapshot()
        try Data("{}".utf8).write(to: directory.appendingPathComponent("preprocessor_config.json"))
        do {
            _ = try await load(directory, vision: true)
            XCTFail("Invalid processor must fail")
        } catch { }
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
        XCTAssertEqual(Qwen4ExpPLEResourceMetrics.snapshot().mappedFiles, 0)
        try FileManager.default.removeItem(at: directory.appendingPathComponent("model.safetensors"))
        for vision in [false, true] {
            do {
                _ = try await load(directory, vision: vision)
                XCTFail("Missing weights must fail")
            } catch { }
            XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
            XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
        }
    }

    func testFreshAndInterleavedCachesOwnPLEStateAndGenericGenerationRefuses() async throws {
        let container = try await load(snapshot(), vision: false)
        do {
            try await container.perform { context in
                let model = context.model
                let first = model.newCache(parameters: nil)
                let second = model.newCache(parameters: nil)
                let control = model.newCache(parameters: nil)
                let ids = MLXArray([Int32(1), 2, 3], [1, 3])
                let expected = model(ids, cache: first)
                eval(expected)
                let repeated = model(ids, cache: second)
                eval(repeated)
                XCTAssertTrue(all(expected .== repeated).item(Bool.self))
                eval(model(ids, cache: control))
                let saved = first.map { $0.copy() }
                let unrelated = model(MLXArray([Int32(7), 8], [1, 2]), cache: second)
                eval(unrelated)
                let token = MLXArray([Int32(4)], [1, 1])
                let continuation = model(token, cache: first)
                let isolated = model(token, cache: control)
                let cloned = model(token, cache: saved)
                eval(continuation, isolated, cloned)
                XCTAssertTrue(all(continuation .== isolated).item(Bool.self))
                XCTAssertTrue(all(continuation .== cloned).item(Bool.self))
                XCTAssertEqual(first.last?.offset, 4)
                XCTAssertEqual(second.last?.offset, 5)
                XCTAssertThrowsError(try TokenIterator(input: LMInput(tokens: ids.reshaped(-1)),
                    model: model, parameters: GenerateParameters(maxTokens: 1))) { error in
                    XCTAssertTrue(error is GenericGenerationError)
                }
            }
        } catch {
            await container.perform { ($0.model as? any Qwen4ExpExternalPLEReleasing)?.releaseExternalPLEResources() }
            throw error
        }
        await container.perform { ($0.model as? any Qwen4ExpExternalPLEReleasing)?.releaseExternalPLEResources() }
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
    }
}
