// Copyright © 2026 Eigen Labs Inc.
import Foundation
import XCTest

@testable import MLXDecisions

final class LayaResponseTests: XCTestCase {
    private let configuration = LayaAgentConfiguration(
        encoder: "modernbert", head_layers: 1,
        max_len: 512, head_max_len: 192, act_costs: ["abstain": 1],
        temperature: [1, 1, 1], temperature_by_options: ["choice:2": 2])

    func testCalibrationEntropyAndActionProbability() throws {
        let question = try request(
            #"{"type":"choice","instructions":"Pick","criteria":{"first":null,"second":null}}"#)
        let answer = try LayaResponse.answer(
            question: question, logits: [2, 0],
            actionLogits: [0, 0], configuration: configuration)
        XCTAssertEqual(answer["choice"], .string("first"))
        XCTAssertEqual(answer["probabilities"]?["first"], .number("0.7311"))
        XCTAssertEqual(answer["confidence"], .number("0.1601"))
        XCTAssertEqual(answer["action"]?["act_probability"], .number("0.5"))
    }

    func testSingleChoiceAndTiesSelectFirstLabel() throws {
        for criteria in [#"{"first":null}"#, #"{"first":null,"second":null}"#] {
            let question = try request(
                #"{"type":"choice","instructions":"Pick","criteria":\#(criteria)}"#)
            let answer = try LayaResponse.answer(
                question: question, logits: [0, 0],
                actionLogits: [0], configuration: configuration)
            XCTAssertEqual(answer["choice"], .string("first"))
            XCTAssertEqual(
                answer["confidence"], .number(question.labels.count == 1 ? "1.0" : "0.0"))
        }
    }

    func testScoreExpectationAndNoulConfidence() throws {
        let score = try request(
            #"{"type":"score","instructions":"Rank","criteria":["low","mid","high"]}"#)
        let answer = try LayaResponse.answer(
            question: score, logits: [0, 0, 0],
            actionLogits: [0], configuration: configuration)
        XCTAssertEqual(answer["score"], .number("1.0"))
        XCTAssertEqual(answer["legend"]?["2"], .string("high"))
        let noul = try request(#"{"type":"noul","instructions":"Valid?"}"#)
        let boolean = try LayaResponse.answer(
            question: noul, logits: [0, 0],
            actionLogits: [0], configuration: configuration)
        XCTAssertEqual(boolean["noul"], .number("0.5"))
        XCTAssertEqual(boolean["confidence"], .number("0.5"))
        XCTAssertNil(boolean["probabilities"])
    }

    func testRejectsNonFiniteOutput() throws {
        let question = try request(#"{"type":"noul","instructions":"Valid?"}"#)
        XCTAssertThrowsError(
            try LayaResponse.answer(
                question: question, logits: [.nan, 0],
                actionLogits: [0], configuration: configuration))
    }

    func testCalibrationUsesUpstreamMinimumTemperature() throws {
        let question = try request(
            #"{"type":"choice","instructions":"Pick","criteria":{"first":null,"second":null}}"#)
        var small = configuration
        small.temperature_by_options["choice:2"] = 0.0001
        var floor = configuration
        floor.temperature_by_options["choice:2"] = 0.001
        let first = try LayaResponse.answer(
            question: question, logits: [0.001, 0],
            actionLogits: [0], configuration: small)
        let second = try LayaResponse.answer(
            question: question, logits: [0.001, 0],
            actionLogits: [0], configuration: floor)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first["probabilities"]?["first"], .number("0.7311"))
    }

    private func request(_ question: String) throws -> SystemOneRequest.Question {
        try SystemOneRequest(
            data: Data(#"{"model":"laya","state":"x","questions":{"q":\#(question)}}"#.utf8)
        ).questions[0]
    }
}
