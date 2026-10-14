// Copyright © 2026 Eigen Labs Inc.
import Foundation
import XCTest

@testable import MLXDecisions

/// Opt-in tokenizer checks. No model weights are loaded or downloaded by these tests.
final class LayaPromptCheckpointTests: XCTestCase {
    func testOversizedOptionHeaderRejectsInsteadOfDroppingDecisions() async throws {
        guard let path = ProcessInfo.processInfo.environment["LAYA_TEST_CHECKPOINT"] else {
            throw XCTSkip(
                "Set LAYA_TEST_CHECKPOINT to a local Laya checkpoint to test its tokenizer")
        }
        let directory = URL(fileURLWithPath: path)
        let prompt = try await LayaPrompt(directory: directory)
        let config = try JSONDecoder().decode(
            LayaAgentConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "rl_agent_config.json")))
        let options = (0 ..< 255).map {
            DecisionJSON.Field(
                "option_\($0)",
                .string(
                    "A detailed description with enough tokens to exceed the option header budget"))
        }
        func request(criteria: [DecisionJSON.Field]) throws -> SystemOneRequest {
            let value: DecisionJSON = .object([
                .init("model", .string("laya")), .init("state", .string("Choose an action.")),
                .init(
                    "questions",
                    .object([
                        .init(
                            "q",
                            .object([
                                .init("type", .string("choice")),
                                .init("instructions", .string("Pick the best action.")),
                                .init("criteria", .object(criteria)),
                            ]))
                    ])),
            ])
            return try SystemOneRequest(data: Data(value.rendered().utf8))
        }
        let oversized = try request(criteria: options)
        XCTAssertEqual(oversized.questions[0].labels.count, 255)
        XCTAssertThrowsError(try prompt.prepare(oversized, config: config)) { error in
            guard case LayaError.invalidRequest(let message) = error else {
                return XCTFail("Expected a request validation error, got \(error)")
            }
            XCTAssertTrue(message.contains("token budget"))
        }
        let valid = try prompt.prepare(request(criteria: Array(options.prefix(3))), config: config)
        XCTAssertEqual(valid[0].markers.count, 3)
        XCTAssertLessThanOrEqual(valid[0].ids.count, config.max_len)
        XCTAssertEqual(valid[0].ids.last, Int32(prompt.sepID))
    }
}
