import Foundation
import MLXLMCommon
import Testing

@Suite("Qwen dual-dialect opaque tool frame boundaries")
struct Qwen35OpaqueFramingTests {
    private let value = #"A "quoted" value; backslash \; café 雪; <think>data</think> <tool_call>data</tool_call> </function>."#
    private var tools: [[String: any Sendable]] {
        [["type": "function", "function": [
            "name": "record_text", "parameters": [
                "type": "object", "properties": ["text": ["type": "string"]],
                "required": ["text"],
            ] as [String: any Sendable],
        ] as [String: any Sendable]]]
    }

    private func jsonPayload(_ value: String) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: [
            "name": "record_text", "arguments": ["text": value],
        ]), as: UTF8.self)
    }

    private func process(_ text: String, width: Int) -> (String, [ToolCall], Int) {
        let processor = ToolCallProcessor(format: .qwen35, tools: tools)
        let characters = Array(text)
        var visible = ""
        for offset in stride(from: 0, to: characters.count, by: width) {
            visible += processor.processChunk(String(characters[offset..<min(offset + width, characters.count)])) ?? ""
        }
        visible += processor.processEOS(returnBufferedText: true) ?? ""
        return (visible, processor.drainToolCalls(), processor.parseFailureCount)
    }

    @Test func literalWrappersAndEscapesRemainOpaqueAcrossEverySmallChunkWidth() throws {
        let parser = ToolCallFormat.qwen35.createParser()
        for text in [value, #"an escaped-looking \\"quote\\" and \\\\ slash </tool_call> <tool_call>"#,
                     "\u{0301}combining-first <tool_call>data</tool_call> \"\u{FE0F} quoted"] {
            let json = try jsonPayload(text)
            let xml = "<function=record_text><parameter=text>" + text + "</parameter></function>"
            for body in [json, xml] {
                let frame = "<tool_call>" + body + "</tool_call>"
                for direct in [body, body + "</tool_call>", frame] {
                    let call = try #require(parser.parse(content: direct, tools: tools))
                    #expect(call.function.arguments["text"] == .string(text))
                }
                for width in [1, 2, 3, 7, frame.count] {
                    let (visible, calls, failures) = process("before" + frame + "after", width: width)
                    #expect(visible == "beforeafter")
                    #expect(failures == 0)
                    #expect(calls.count == 1)
                    #expect(calls.first?.function.arguments["text"] == .string(text))
                }
            }
        }
    }

    @Test func directEOSPreservesMultipleOuterFramesWithoutSplittingInnerTags() throws {
        let parser = ToolCallFormat.qwen35.createParser()
        let first = "<tool_call>" + (try jsonPayload(value)) + "</tool_call>"
        let second = "<tool_call><function=record_text><parameter=text>second <tool_call>literal</tool_call></parameter></function></tool_call>"
        let calls = parser.parseEOS(first + "\n" + second, tools: tools)
        #expect(calls.count == 2)
        #expect(calls.first?.function.arguments["text"] == .string(value))
        #expect(calls.last?.function.arguments["text"] == .string("second <tool_call>literal</tool_call>"))
        let withTruncatedTail = parser.parseEOS(first + second + #"<tool_call>{"name":"record_text","arguments":{"text":"unfinished <tool_call>"#, tools: tools)
        #expect(withTruncatedTail == calls)
    }

    @Test func combiningMarkAfterOuterEndIsPreservedAsTrailingText() throws {
        let frame = "<tool_call>" + (try jsonPayload(value)) + "</tool_call>"
        let (visible, calls, failures) = process(frame + "\u{0301}tail", width: frame.count + 1)
        #expect(visible == "\u{0301}tail")
        #expect(calls.count == 1 && failures == 0)
        #expect(calls.first?.function.arguments["text"] == .string(value))
    }

    @Test func adjacentCallsRemainFIFOAndOuterTrailingTextIsNotDiscarded() throws {
        let first = "<tool_call>" + (try jsonPayload(value)) + "</tool_call>"
        let second = "<tool_call><function=record_text><parameter=text>second</parameter></function></tool_call>"
        let (visible, calls, failures) = process(first + second, width: 1)
        #expect(visible.isEmpty && failures == 0)
        #expect(calls.map { $0.function.arguments["text"] } == [.string(value), .string("second")])
        #expect(ToolCallFormat.qwen35.createParser().parse(content: first + "unrelated", tools: tools) == nil)
    }

    @Test func malformedAndTruncatedFramesRemainExactVisibleFailuresAtEOS() {
        for frame in [
            #"<tool_call>{"name":"record_text","arguments":{"text":"unfinished </tool_call>"#,
            #"<tool_call><function=record_text><parameter=text>unfinished </tool_call>"#,
            #"<tool_call>{"name":"record_text","arguments":invalid}</tool_call>"#,
            #"<tool_call><function=record_text>junk</function></tool_call>"#,
        ] {
            for width in [1, 3, frame.count] {
                let (visible, calls, failures) = process(frame, width: width)
                #expect(visible == frame)
                #expect(calls.isEmpty)
                #expect(failures == 1)
            }
        }
    }

    @Test func EOSDoesNotSplitLiteralOpeningTagsInsideACompletePayload() throws {
        // Existing EOS behavior accepts a complete payload when the outer
        // closing marker was intercepted upstream; its values must stay whole.
        let frame = "<tool_call>" + (try jsonPayload(value))
        let (visible, calls, failures) = process(frame, width: 2)
        #expect(visible.isEmpty && failures == 0)
        #expect(calls.count == 1)
        #expect(calls.first?.function.arguments["text"] == .string(value))
    }

    @Test func XMLFramingNewlinesAndSchemaConversionRemainUnchanged() throws {
        let parser = ToolCallFormat.qwen35.createParser()
        let frame = "<tool_call><function=record_text><parameter=text>\n" + value + "\n</parameter></function></tool_call>"
        let call = try #require(parser.parse(content: frame, tools: tools))
        #expect(call.function.arguments["text"] == .string(value))
        #expect(parser.parse(content: "<tool_call><function=undeclared></function></tool_call>", tools: tools) == nil)
    }

    @Test func scannerStateIsBoundedAcrossLongParameterAndJSONStringValues() throws {
        let long = String(repeating: "x", count: 65536) + value
        for payload in [try jsonPayload(long), "<function=record_text><parameter=text>" + long + "</parameter></function>"] {
            var scanner = Qwen35ToolFrameScanner()
            for scalar in payload.unicodeScalars {
                let completed = scanner.consume(scalar)
                #expect(!completed)
                #expect(scanner.bufferedCharacterCount <= 13)
            }
            let end = Array("</tool_call>".unicodeScalars)
            for (i, scalar) in end.enumerated() {
                let completed = scanner.consume(scalar)
                #expect(completed == (i == end.count - 1))
            }
        }
    }

    @Test func OtherTaggedFormatsKeepTheirExistingBasicBehavior() {
        let json = ToolCallProcessor(format: .json, tools: tools)
        #expect(json.processChunk(#"<tool_call>{"name":"record_text","arguments":{"text":"plain"}}</tool_call>tail"#) == "tail")
        #expect(json.drainToolCalls().first?.function.arguments["text"] == .string("plain"))
        let xml = ToolCallProcessor(format: .nemotron, tools: tools)
        #expect(xml.processChunk("<function=record_text><parameter=text>plain</parameter></function>tail") == "tail")
        #expect(xml.drainToolCalls().first?.function.arguments["text"] == .string("plain"))
    }
}
