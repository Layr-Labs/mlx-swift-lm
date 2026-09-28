import Foundation
import MLXLMCommon
import MLXLMServer
import XCTest

/// Prepared real parser/processor tests. No model, service or generated-output
/// capability is inferred from these wire fixtures; root executes the suite.
final class MiMoV2ToolCodecTests: XCTestCase {
    private func frame(_ value: String, name: String = "record_text", key: String = "text") -> String {
        "<tool_call><function=\(name)><parameter=\(key)>\(value)</parameter></function></tool_call>"
    }
    private func tools(_ type: String = "string", name: String = "record_text", key: String = "text") -> [[String: any Sendable]] {
        let property: [String: any Sendable] = ["type": type]
        let properties: [String: any Sendable] = [key: property]
        let parameters: [String: any Sendable] = ["type": "object", "properties": properties,
            "required": [key], "additionalProperties": false]
        let function: [String: any Sendable] = ["name": name, "parameters": parameters]
        return [["type": "function", "function": function]]
    }
    private func text(_ call: ToolCall?, key: String = "text") throws -> String {
        let call = try XCTUnwrap(call)
        guard case .string(let value) = call.function.arguments[key] else {
            XCTFail("expected raw string"); return ""
        }
        return value
    }
    private func pieces(_ value: String, width: Int) -> [String] {
        let scalars = Array(value.unicodeScalars)
        return stride(from: 0, to: scalars.count, by: width).map {
            String(String.UnicodeScalarView(scalars[$0..<min($0 + width, scalars.count)]))
        }
    }

    func testExactNativeInferenceAndExplicitFormat() throws {
        XCTAssertEqual(ToolCallFormat.infer(from: "mimo_v2"), .mimoV2)
        for legacy in ["mimo", "mimo_v2_flash", "mimo_v2_extra", "qwen3"] {
            XCTAssertNotEqual(ToolCallFormat.infer(from: legacy), .mimoV2, legacy)
        }
        XCTAssertEqual(try ServerToolParser.resolve(requested: nil, modelType: "mimo_v2"), .mimoV2)
        XCTAssertEqual(try ServerToolParser.resolve(requested: "mimo", modelType: nil), .mimoV2)
        XCTAssertEqual(try ServerToolParser.resolve(requested: "mimo-v2", modelType: nil), .mimoV2)
        XCTAssertEqual(ToolCallFormat(rawValue: "mimo_v2"), .mimoV2)
    }

    func testDirectRawStringsPreserveBytes() throws {
        let values = [
            "\nKeep both boundary newlines.\n", " literal \\n and actual\nline; \\\\ \"quote\" ",
            "café 雪 🌊 e\u{301}", "\u{301}leading combining mark", "null", "\"quoted JSON-looking string\"",
            "&amp; &#10; &lt; are literal bytes", "\u{00a0}non-XML whitespace\u{00a0}",
            "literal <think>thought</think> and <tool_call>data</tool_call> </function>",
        ]
        let parser = MiMoV2ToolCallParser()
        for value in values {
            XCTAssertEqual(Array(try text(parser.parse(content: frame(value), tools: tools())).utf8), Array(value.utf8))
        }
    }

    func testStrictTypedJSONValuesAndNoRepairs() throws {
        let parser = MiMoV2ToolCallParser()
        XCTAssertEqual(parser.parse(content: frame("1"), tools: tools("integer"))?.function.arguments["text"], .int(1))
        XCTAssertEqual(parser.parse(content: frame("false"), tools: tools("boolean"))?.function.arguments["text"], .bool(false))
        XCTAssertEqual(parser.parse(content: frame("null"), tools: tools("null"))?.function.arguments["text"], .null)
        XCTAssertEqual(parser.parse(content: frame("[0,1,true]"), tools: tools("array"))?.function.arguments["text"], .array([.int(0), .int(1), .bool(true)]))
        XCTAssertEqual(parser.parse(content: frame(#"{"x":"\n","n":1}"#), tools: tools("object"))?.function.arguments["text"], .object(["x": .string("\n"), "n": .int(1)]))
        for (value, type) in [("yes", "boolean"), ("garbage", "boolean"), ("01", "integer"),
                              ("1.5", "integer"), ("nan", "number"), ("{'x':1}", "object")] {
            XCTAssertNil(parser.parse(content: frame(value), tools: tools(type)), value)
        }
        XCTAssertEqual(try text(parser.parse(content: frame("null"), tools: tools())), "null")
    }

    func testLeadingCombiningScalarParameterNameRemainsByteExact() throws {
        let key = "\u{301}name", value = "literal <tool_call>data</tool_call> &amp;\n"
        let encoded = frame(value, key: key), declared = tools(key: key)
        let parser = MiMoV2ToolCallParser()
        let call = try XCTUnwrap(parser.parse(content: encoded, tools: declared))
        XCTAssertEqual(call.function.arguments.count, 1)
        XCTAssertEqual(Array(try XCTUnwrap(call.function.arguments.keys.first).utf8), Array(key.utf8))
        XCTAssertEqual(Array(try text(call, key: key).utf8), Array(value.utf8))
        XCTAssertNil(call.function.arguments["name"], "leading combining scalar must not disappear")
        for width in [1, 2, 7, 4096] {
            let processor = ToolCallProcessor(format: .mimoV2, tools: declared)
            for chunk in pieces(encoded, width: width) { XCTAssertNil(processor.processChunk(chunk)) }
            XCTAssertNil(processor.processEOS(returnBufferedText: true))
            XCTAssertEqual(processor.toolCalls.count, 1)
            let streamed = try XCTUnwrap(processor.toolCalls.first)
            XCTAssertEqual(Array(try XCTUnwrap(streamed.function.arguments.keys.first).utf8), Array(key.utf8))
            XCTAssertEqual(Array(try text(streamed, key: key).utf8), Array(value.utf8))
            XCTAssertEqual(processor.parseFailureCount, 0)
        }
        // The shared lexer already consumes Unicode scalars and never removes
        // a name prefix by Character count. Only the final OUTER marker closes.
        let body = encoded.unicodeScalars.dropFirst("<tool_call>".unicodeScalars.count)
        var scanner = Qwen35ToolFrameScanner(), endPositions: [Int] = []
        for (index, scalar) in body.enumerated() { if scanner.consume(scalar) { endPositions.append(index) } }
        XCTAssertEqual(endPositions, [body.count - 1])
    }

    func testUnknownFunctionAndEmptyDeclarationsReject() {
        let parser = MiMoV2ToolCallParser()
        XCTAssertNil(parser.parse(content: frame("x", name: "unknown"), tools: tools()))
        XCTAssertNil(parser.parse(content: frame("x"), tools: []))
        XCTAssertNil(parser.parse(content: frame("x"), tools: tools() + tools()))
        XCTAssertNotNil(parser.parse(content: frame("x"), tools: nil))
    }

    func testMissingTypeRetainsRawAndAmbiguousTypesRefuse() throws {
        let parser = MiMoV2ToolCallParser()
        XCTAssertEqual(try text(parser.parse(content: frame("123"), tools: nil)), "123")
        for schema: [String: any Sendable] in [
            ["type": ["string", "integer"]],
            ["anyOf": [["type": "string"], ["type": "integer"]]],
            ["type": "unsupported"],
        ] {
            let properties: [String: any Sendable] = ["text": schema]
            let parameters: [String: any Sendable] = ["properties": properties]
            let function: [String: any Sendable] = ["name": "record_text", "parameters": parameters]
            let declared: [[String: any Sendable]] = [["function": function]]
            XCTAssertNil(parser.parse(content: frame("123"), tools: declared))
        }
    }

    func testNoExtraDialectsOrIncompleteEOS() {
        let parser = MiMoV2ToolCallParser()
        let complete = frame("Paris")
        let invalid = [String(complete.dropLast("</tool_call>".count)), String(complete.dropFirst("<tool_call>".count)),
            #"<tool_call>{"name":"record_text","arguments":{"text":"Paris"}}</tool_call>"#,
            #"{"name":"record_text","arguments":{"text":"Paris"}}"#,
            "<function=record_text><parameter=text>Paris</parameter></function>",
            complete + "trailing prose", "unrelated prefix" + complete,
        ]
        for value in invalid {
            XCTAssertNil(parser.parse(content: value, tools: tools()), value)
            XCTAssertTrue(parser.parseEOS(value, tools: tools()).isEmpty, value)
        }
        XCTAssertNotNil(parser.parse(content: "\r\n" + complete + "\t ", tools: tools()))
        XCTAssertNil(parser.parse(content: "\u{00a0}" + complete, tools: tools()))
    }

    func testDuplicateAndMalformedParametersReject() {
        let parser = MiMoV2ToolCallParser()
        for body in [
            "<parameter=text>a</parameter><parameter=text>b</parameter>",
            "<parameter=text>unfinished", "<parameter=text>a</parameter>leftovers",
            "<parameter=>a</parameter>", "<parameter=text>literal </parameter> still data</parameter>",
        ] {
            XCTAssertNil(parser.parse(content: "<tool_call><function=record_text>" + body + "</function></tool_call>", tools: tools()))
        }
    }

    func testActualProcessorRetainsRawArgumentsAtAllChunkWidths() throws {
        let value = "\n\u{301}café 雪 \\n literal <tool_call>fake</tool_call> <think>data</think> </function>\n"
        for width in [1, 2, 7, 4096] {
            let processor = ToolCallProcessor(format: .mimoV2, tools: tools())
            var visible = ""
            for chunk in pieces(frame(value), width: width) { visible += processor.processChunk(chunk) ?? "" }
            visible += processor.processEOS(returnBufferedText: true) ?? ""
            XCTAssertEqual(processor.toolCalls.count, 1)
            XCTAssertEqual(Array(try text(processor.toolCalls.first).utf8), Array(value.utf8))
            XCTAssertEqual(processor.parseFailureCount, 0); XCTAssertTrue(visible.isEmpty)
        }
    }

    func testActualHandlerAdjacentCallsAndDrainOrder() throws {
        let output = ["Paris", "Tokyo", "Lima", "Oslo"].map { frame($0) }.joined()
        for width in [1, 3, 17, 4096] {
            let handler = BatchedToolStreamHandler(format: .mimoV2, tools: tools())
            for chunk in pieces(output, width: width) { XCTAssertNil(handler.processChunk(chunk)) }
            let calls = handler.finish()
            XCTAssertEqual(calls.count, 4)
            XCTAssertEqual(try calls.map { try text($0) }, ["Paris", "Tokyo", "Lima", "Oslo"])
            XCTAssertEqual(handler.parseFailureCount, 0); XCTAssertNil(handler.takeResidualText())
        }
        let processor = ToolCallProcessor(format: .mimoV2, tools: tools())
        _ = processor.processChunk(output)
        XCTAssertEqual(processor.drainToolCalls().count, 4)
        XCTAssertTrue(processor.drainToolCalls().isEmpty)
    }

    func testProcessorTrueEOSDoesNotRecoverMissingOuterClose() {
        let unfinished = String(frame("Paris").dropLast("</tool_call>".count))
        for width in [1, 4096] {
            let processor = ToolCallProcessor(format: .mimoV2, tools: tools())
            for chunk in pieces(unfinished, width: width) { XCTAssertNil(processor.processChunk(chunk)) }
            XCTAssertEqual(processor.processEOS(returnBufferedText: true), unfinished)
            XCTAssertTrue(processor.toolCalls.isEmpty)
            XCTAssertEqual(processor.parseFailureCount, 1)
        }
    }

    func testDirectEOSRequiresEverySuppliedFrameComplete() {
        let parser = MiMoV2ToolCallParser()
        XCTAssertEqual(parser.parseEOS(frame("one") + frame("two"), tools: tools()).count, 2)
        XCTAssertTrue(parser.parseEOS(frame("one") + String(frame("two").dropLast()), tools: tools()).isEmpty)
    }

    func testNativeParameterDelimiterCollisionIsExplicitlyUnrepresentable() throws {
        // The official raw-value template maps two distinct argument objects to
        // exactly these same bytes. No codec can recover original intent. This
        // test proves the limitation; it is NOT a semantic-copy success claim.
        let oneRawValue = "a</parameter><parameter=other>b"
        let encodedOne = frame(oneRawValue)
        let encodedTwo = "<tool_call><function=record_text><parameter=text>a</parameter><parameter=other>b</parameter></function></tool_call>"
        XCTAssertEqual(Array(encodedOne.utf8), Array(encodedTwo.utf8))
        let structural = try XCTUnwrap(MiMoV2ToolCallParser().parse(content: encodedTwo, tools: nil))
        XCTAssertEqual(structural.function.arguments, ["text": .string("a"), "other": .string("b")])
        XCTAssertNotEqual(Array(try text(structural).utf8), Array(oneRawValue.utf8))
    }

    func testOtherFamiliesRetainTheirPreviousPolicies() throws {
        XCTAssertEqual(ToolCallFormat.infer(from: "qwen3_5"), .qwen35)
        XCTAssertEqual(ToolCallFormat.infer(from: "qwen4_exp"), .qwen35)
        XCTAssertEqual(ToolCallFormat.infer(from: "nemotron_h"), .nemotron)
        let value = "\nraw\n"
        let old = Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        XCTAssertEqual(try text(old.parse(content: frame(value), tools: tools())), "raw")
        XCTAssertEqual(old.parseEOS(String(frame("x").dropLast("</tool_call>".count)), tools: tools()).count, 1)
        let bare = "<function=record_text><parameter=text>x</parameter></function>"
        XCTAssertNotNil(ToolCallFormat.nemotron.createParser().parse(content: bare, tools: tools()))
        XCTAssertNil(MiMoV2ToolCallParser().parse(content: bare, tools: tools()))
    }
}
