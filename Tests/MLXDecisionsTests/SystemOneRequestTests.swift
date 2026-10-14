// Copyright © 2026 Eigen Labs Inc.
import Foundation
import XCTest

@testable import MLXDecisions

final class SystemOneRequestTests: XCTestCase {
    func testPreservesQuestionOptionAndStructuredStateOrder() throws {
        let request = try SystemOneRequest(
            data: Data(
                #"{"model":"laya","state":{"z":"last","a":"first"},"questions":{"z":{"type":"choice","instructions":"Pick","criteria":{"later":"b","earlier":"a"}},"a":{"type":"noul","instructions":"Valid?"}}}"#
                    .utf8))
        XCTAssertEqual(request.questions.map(\.id), ["z", "a"])
        XCTAssertEqual(request.questions[0].labels, ["later", "earlier"])
        XCTAssertEqual(request.state.rendered(), #"{"z": "last", "a": "first"}"#)
    }

    func testRejectsDuplicateKeysAndStreaming() throws {
        XCTAssertThrowsError(try DecisionJSON.parse(Data(#"{"x":1,"x":2}"#.utf8)))
        XCTAssertThrowsError(
            try SystemOneRequest(
                data: Data(
                    #"{"model":"laya","state":"x","stream":true,"questions":{"q":{"type":"noul","instructions":"Valid?"}}}"#
                        .utf8)))
        XCTAssertThrowsError(
            try SystemOneRequest(
                data: Data(
                    #"{"model":"laya","state":"x","questions":{"q":{"type":"score","instructions":"Rank","criteria":["only"]}}}"#
                        .utf8)))
    }

    func testStructuredUnicodeRenderingMatchesPythonDefaults() throws {
        let value = try DecisionJSON.parse(Data(#"{"text":"café 🎵","lines":["a\nb",null]}"#.utf8))
        XCTAssertEqual(value.rendered(), #"{"text": "café 🎵", "lines": ["a\nb", null]}"#)
        XCTAssertEqual(
            value.rendered(ascii: true),
            #"{"text": "caf\u00e9 \ud83c\udfb5", "lines": ["a\nb", null]}"#)
    }

    func testEnforcesQuestionAndCriteriaBounds() throws {
        func envelope(_ questions: String) -> Data {
            Data(#"{"model":"laya","state":"x","questions":{\#(questions)}}"#.utf8)
        }
        let fields = (0 ..< 65).map { #""q\#($0)":{"type":"noul","instructions":"Valid?"}"# }
        XCTAssertNoThrow(
            try SystemOneRequest(data: envelope(fields.prefix(64).joined(separator: ","))))
        XCTAssertThrowsError(try SystemOneRequest(data: envelope(fields.joined(separator: ","))))
        XCTAssertThrowsError(try SystemOneRequest(data: envelope("")))
        for criteria in ["{}", "[]", "null"] {
            XCTAssertThrowsError(
                try SystemOneRequest(
                    data: envelope(
                        #""q":{"type":"choice","instructions":"Pick","criteria":\#(criteria)}"#)))
        }
        let criteria = (0 ..< 256).map { #""o\#($0)":null"# }.joined(separator: ",")
        XCTAssertThrowsError(
            try SystemOneRequest(
                data: envelope(
                    #""q":{"type":"choice","instructions":"Pick","criteria":{\#(criteria)}}"#)))
    }

    func testRejectsGenerationControlsEvenWhenZero() throws {
        let request =
            #"{"model":"laya","state":"x","max_tokens":0,"questions":{"q":{"type":"noul","instructions":"Valid?"}}}"#
        XCTAssertThrowsError(try SystemOneRequest(data: Data(request.utf8)))
    }

    func testStructuredNumbersRenderParsedValuesAndRetainIntegerPrecision() throws {
        let value = try DecisionJSON.parse(
            Data(#"[1e2,1e-7,-0,1.23000,-0.0,123456789012345678901234567890]"#.utf8))
        XCTAssertEqual(
            value.rendered(), "[100.0, 1e-07, 0, 1.23, -0.0, 123456789012345678901234567890]")
        let extremes = try DecisionJSON.parse(
            Data("[1e-4,1e-5,1e15,1e16,1e20,5e-324,1.7976931348623157e308]".utf8))
        XCTAssertEqual(
            extremes.rendered(),
            "[0.0001, 1e-05, 1000000000000000.0, 1e+16, 1e+20, 5e-324, 1.7976931348623157e+308]")
        XCTAssertThrowsError(try DecisionJSON.parse(Data("[1e999]".utf8)))
    }

    func testUnicodeLineSeparatorsFollowStructuredTextAndInstructionConventions() throws {
        let value = DecisionJSON.array([.string("line\u{2028}paragraph\u{2029}end")])
        XCTAssertEqual(value.rendered(), "[\"line\u{2028}paragraph\u{2029}end\"]")
        XCTAssertEqual(value.rendered(ascii: true), #"["line\u2028paragraph\u2029end"]"#)
    }
}
