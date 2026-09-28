import MLXLMCommon
import MLXLMServer
import Testing

@Suite("Gemma constrained grammar parser")
struct GemmaFunctionConstraintParserTests {
    private let strictTools: [[String: any Sendable]] = [[
        "type": "function", "function": ["name": "record_text", "parameters": [
            "type": "object", "properties": ["text": ["type": "string"]], "required": ["text"],
            "additionalProperties": false,
        ] as [String: any Sendable]] as [String: any Sendable],
    ]]

    @Test func strictNativeParserPreservesLiteralWrappersInsideArguments() throws {
        let value = "keep <|tool_call>literal<tool_call|> and <|channel>thought<channel|> with \\slashes"
        let frame = "<|tool_call>call:record_text{text:<|\"|>" + value + "<|\"|>}<tool_call|>"
        for width in [1, 3, frame.count] {
            let handler = BatchedToolStreamHandler(format: .gemma, tools: strictTools, strictGemma: true)
            let chars = Array(frame)
            for start in stride(from: 0, to: chars.count, by: width) {
                #expect(handler.processChunk(String(chars[start..<min(start + width, chars.count)])) == nil)
            }
            let calls = handler.finish()
            #expect(calls.count == 1 && handler.parseFailureCount == 0)
            #expect(calls.first?.function.arguments["text"] == .string(value))
        }
    }

    @Test func strictNativeParserNeverRepairsNamesArgumentsOrIncompleteFrames() {
        for frame in [
            #"<|tool_call>call:record_tex{text:<|"|>x<|"|>}<tool_call|>"#,
            #"<|tool_call>call:record_text{text:bare text}<tool_call|>"#,
            #"<|tool_call>call:record_text{text:<|"|>x<|"|>,text:<|"|>y<|"|>}<tool_call|>"#,
            #"<|tool_call>call:record_text{text:<|"|>x<|"|>}"#,
        ] {
            let handler = BatchedToolStreamHandler(format: .gemma, tools: strictTools, strictGemma: true)
            _ = handler.processChunk(frame)
            #expect(handler.finish().isEmpty)
            #expect(handler.parseFailureCount == 1)
        }
    }

    @Test("parser accepts recursive values emitted by the pinned template grammar")
    func recursiveGemmaValues() throws {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: [[
            "type": "function",
            "function": [
                "name": "submit",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "payload": ["type": "object"] as [String: any Sendable],
                        "tags": ["type": "array"] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]])
        #expect(handler.processChunk(
            #"<|tool_call>call:submit{payload:{count:2,ok:true},tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>"#
        ) == nil)
        let call = try #require(handler.finish().first)
        #expect(call.function.name == "submit")
        #expect(call.function.arguments["payload"] == .object([
            "count": .int(2), "ok": .bool(true),
        ]))
        #expect(call.function.arguments["tags"] == .array([
            .string("a"), .string("b"),
        ]))
    }

    @Test("malformed tagged auto output is visible instead of silently dropped")
    func malformedTaggedOutputFallsBackToText() {
        let malformed =
            #"<|tool_call>call:bad{payload:[}<tool_call|>"#
        let handler = BatchedToolStreamHandler(format: .gemma, tools: [[
            "type": "function",
            "function": ["name": "submit"] as [String: any Sendable],
        ]])
        #expect(handler.processChunk(malformed) == malformed)
        #expect(handler.finish().isEmpty)
        #expect(handler.parseFailureCount == 1)
    }

    @Test("malformed auto output is preserved before a subsequent valid call")
    func malformedThenValidCall() throws {
        let malformed =
            #"<|tool_call>call:bad{payload:[}<tool_call|>"#
        let valid =
            #"<|tool_call>call:submit{payload:<|"|>ok<|"|>}<tool_call|>"#
        let handler = BatchedToolStreamHandler(format: .gemma, tools: [[
            "type": "function",
            "function": ["name": "submit"] as [String: any Sendable],
        ]])
        #expect(handler.processChunk(malformed + valid) == malformed)
        let call = try #require(handler.finish().first)
        #expect(call.function.name == "submit")
        #expect(handler.parseFailureCount == 1)
    }
}
