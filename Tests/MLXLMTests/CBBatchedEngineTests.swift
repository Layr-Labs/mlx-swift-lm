// Tests for BatchedEngine — §12 of ContinuousBatchingTestPlan.md

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBBatchedEngineTests: XCTestCase {

    private func makeEngine() -> BatchedEngine {
        BatchedEngine(
            scheduler: makeTestScheduler(eosTokenIds: [5]),
            tokenizer: IdentityTokenizer(),
            modelName: "test-model"
        )
    }

    func testBatchedEngineGenerateReturnsString() async throws {
        let engine = makeEngine()
        await engine.start()

        // "3" encodes to [3] via IdentityTokenizer → model generates 4, then 5 (EOS)
        // outputTokenIds = [4], decoded = "4" (non-empty)
        let result = try await engine.generate(
            prompt: "3",
            samplingParams: SamplingParams(maxTokens: 10)
        )

        XCTAssertFalse(result.isEmpty, "generate must return a non-empty string")
        await engine.stop()
    }

    func testBatchedEngineStreamGenerateYieldsChunks() async throws {
        let engine = makeEngine()
        await engine.start()

        var chunks: [String] = []
        for await chunk in engine.streamGenerate(
            prompt: "3",
            samplingParams: SamplingParams(maxTokens: 10)
        ) {
            chunks.append(chunk)
        }

        XCTAssertFalse(chunks.isEmpty, "stream must yield at least one chunk")
        await engine.stop()
    }

    func testBatchedEngineChatAppliesTemplate() async throws {
        let engine = makeEngine()
        await engine.start()

        // applyChatTemplate on IdentityTokenizer wraps messages as "[role]:content"
        // and encodes them. The resulting prompt should produce valid output.
        let messages: [[String: String]] = [
            ["role": "user", "content": "hi"],
        ]
        let result = try await engine.chat(
            messages: messages,
            samplingParams: SamplingParams(maxTokens: 5)
        )

        // Just verify it completes without throwing.
        XCTAssertNotNil(result)
        await engine.stop()
    }
}
