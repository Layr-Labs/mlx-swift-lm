// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
import Testing

/// LFM2.5 pythonic tool frames, in the exact spelling the LFM2.5 chat
/// template renders (`format_arg_value`: single-quoted strings escaping
/// `\`, `'`, newline, CR; `tojson` collections; `True`/`False`/`None`).
struct LFM2ToolCallParserTests {
    static let start = "<|tool_call_start|>"
    static let end = "<|tool_call_end|>"

    /// Rendered by the official template for
    /// search_notes(query="it's a \"test\"\nline2 \\ end", limit=5, score=0.75, exact=true,
    /// tags=["a","b"], filter={"k":"v","n":2}, lang="日本語 ✓") + get_weather(location="Paris, France").
    static let templateFrame = start
        + #"[search_notes(exact=True, filter={"k": "v", "n": 2}, lang='日本語 ✓', limit=5, "#
        + #"query='it\'s a "test"\nline2 \\ end', score=0.75, tags=["a", "b"]), "#
        + #"get_weather(location='Paris, France')]"# + end

    @Test func formatResolvesToTheLFM2Parser() {
        #expect(ToolCallFormat.lfm2.createParser() is LFM2ToolCallParser)
        #expect(ToolCallFormat.infer(from: "lfm2_moe") == .lfm2)
    }

    @Test func templateRenderedFrameRoundTripsEveryValue() throws {
        let calls = LFM2ToolCallParser().parseFrame(content: Self.templateFrame, tools: nil)
        #expect(calls.map(\.function.name) == ["search_notes", "get_weather"])
        let arguments = try #require(calls.first?.function.arguments)
        #expect(arguments["query"] == .string("it's a \"test\"\nline2 \\ end"))
        #expect(arguments["limit"] == .int(5))
        #expect(arguments["score"] == .double(0.75))
        #expect(arguments["exact"] == .bool(true))
        #expect(arguments["tags"] == .array([.string("a"), .string("b")]))
        #expect(arguments["filter"] == .object(["k": .string("v"), "n": .int(2)]))
        #expect(arguments["lang"] == .string("日本語 ✓"))
        #expect(calls.last?.function.arguments == ["location": .string("Paris, France")])
    }

    @Test func pythonLiteralsAndNestedValues() throws {
        let frame = Self.start
            + "[f(a=None, b=False, c=-3, d=1e3, e=[1, [2, {'x': (1, 2)}]], g={'k': 'v', 'n': None}, h=\"dq 'x'\")]"
            + Self.end
        let call = try #require(LFM2ToolCallParser().parseFrame(content: frame, tools: nil).first)
        let arguments = call.function.arguments
        #expect(arguments["a"] == .null)
        #expect(arguments["b"] == .bool(false))
        #expect(arguments["c"] == .int(-3))
        // JSONValue classifies an integral Double as .int; JSON-equal to 1000.0.
        #expect(arguments["d"] == .int(1000))
        #expect(arguments["e"] == .array([.int(1), .array([.int(2), .object(["x": .array([.int(1), .int(2)])])])]))
        #expect(arguments["g"] == .object(["k": .string("v"), "n": .null]))
        #expect(arguments["h"] == .string("dq 'x'"))
    }

    @Test func delimitersInsideStringsNeverSplitCalls() throws {
        let frame = Self.start + "[run(code='f(x), g(y)] [z]', note=\"a=b, c\"), noop()]" + Self.end
        let calls = LFM2ToolCallParser().parseFrame(content: frame, tools: nil)
        #expect(calls.map(\.function.name) == ["run", "noop"])
        #expect(calls.first?.function.arguments["code"] == .string("f(x), g(y)] [z]"))
        #expect(calls.first?.function.arguments["note"] == .string("a=b, c"))
        #expect(calls.last?.function.arguments.isEmpty == true)
    }

    @Test func declaredSchemaTypesConvertQuotedScalars() throws {
        let tools: [[String: any Sendable]] = [[
            "function": [
                "name": "set_temperature",
                "parameters": ["properties": [
                    "value": ["type": "integer"], "enabled": ["type": "boolean"],
                ] as [String: any Sendable]] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let frame = Self.start + "[set_temperature(value='25', enabled='true')]" + Self.end
        let call = try #require(LFM2ToolCallParser().parseFrame(content: frame, tools: tools).first)
        #expect(call.function.arguments == ["value": .int(25), "enabled": .bool(true)])
    }

    @Test func wrapperObjectIsUnwrappedOnlyWhenUndeclared() throws {
        let frame = Self.start + #"[get_weather(properties={"location": "Tokyo"})]"# + Self.end
        let call = try #require(LFM2ToolCallParser().parseFrame(content: frame, tools: nil).first)
        #expect(call.function.arguments == ["location": .string("Tokyo")])
    }

    @Test func malformedFramesParseToNothingAndNeverCrash() {
        let parser = LFM2ToolCallParser()
        for body in [
            "[f(a='x'", "[f(a='x')", "[]", "[f('positional')]", "[f(a=1, a=2)]", "[f(a=)]",
            "[f(a=undefined_name)]", "[not a call]", "[f(a='x')] trailing", "[f(a='x'), g(]",
            "[1bad name(a=1)]", String(repeating: "[", count: 500), "[f(a=\(String(repeating: "[", count: 200)))]",
            "[f(a='unterminated)]",
        ] {
            #expect(parser.parseFrame(content: Self.start + body + Self.end, tools: nil).isEmpty, "\(body)")
        }
    }

    @Test func processorEmitsEveryCallOfOneFrameAtAnyChunkWidth() {
        let characters = Array("Sure." + Self.templateFrame + " ")
        for width in [1, 2, 5, 19, characters.count] {
            let processor = ToolCallProcessor(format: .lfm2)
            var visible = ""
            for start in stride(from: 0, to: characters.count, by: width) {
                visible += processor.processChunk(
                    String(characters[start..<min(start + width, characters.count)])) ?? ""
            }
            visible += processor.processEOS(returnBufferedText: true) ?? ""
            #expect(visible == "Sure. ")
            #expect(processor.toolCalls.map(\.function.name) == ["search_notes", "get_weather"])
            #expect(processor.parseFailureCount == 0)
        }
    }

    /// Regression: text before a start tag split across chunks was either
    /// dropped or the whole frame leaked as text (generic tagged path).
    @Test func leadingTextBeforeSplitStartTagIsKept() {
        let processor = ToolCallProcessor(format: .lfm2)
        var visible = ""
        for chunk in ["Calling a < b now.<|tool", "_call_start|>[f(", "a=1)]<|tool_call_end|>", " Done"] {
            visible += processor.processChunk(chunk) ?? ""
        }
        #expect(visible == "Calling a < b now. Done")
        #expect(processor.toolCalls.map(\.function.name) == ["f"])
    }

    @Test func malformedFrameIsReturnedAsVisibleTextNotDropped() {
        let processor = ToolCallProcessor(format: .lfm2)
        let frame = Self.start + "[f(a='x'" + Self.end
        let visible = processor.processChunk(frame)
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.parseFailureCount == 1)
        #expect(visible == frame)
    }

    @Test func frameWithoutEndTagParsesAtEndOfSequence() {
        let processor = ToolCallProcessor(format: .lfm2)
        _ = processor.processChunk(Self.start + "[a(x=1), b(y='z')]")
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.processEOS(returnBufferedText: true) == nil)
        #expect(processor.toolCalls.map(\.function.name) == ["a", "b"])
    }

    @Test func llama3InlinePythonicKeepsItsHistoricalParser() throws {
        // `.lfm2` moved to LFM2ToolCallParser; Llama 3 still uses the legacy
        // PythonicToolCallParser (unquoted scalars stay strings there).
        let call = try #require(ToolCallFormat.llama3.createParser().parse(
            content: "<|python_tag|>calculate(expression=\"2 + 2\", precision=4)", tools: nil))
        #expect(call.function.arguments["precision"] == .string("4"))
    }
}
