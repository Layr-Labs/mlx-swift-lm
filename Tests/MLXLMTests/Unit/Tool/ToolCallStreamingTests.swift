import Foundation
import MLXLMCommon
import Testing

extension UnitTests {

    /// Calls exposed while they are generated (`ToolCallStreamingOptions`).
    /// Every case feeds the same text at several chunk widths and checks the
    /// OpenAI-style reconstruction (name from the first fragment, arguments
    /// concatenated) against the complete-frame parse.
    @Suite
    struct ToolCallStreamingTests {

        private static let weatherTools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": ["type": "string"] as [String: any Sendable],
                            "unit": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            [
                "type": "function",
                "function": [
                    "name": "write_file",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable],
                            "content": ["type": "string"] as [String: any Sendable],
                            "mode": ["type": "integer"] as [String: any Sendable],
                            "append": ["type": "boolean"] as [String: any Sendable],
                            "meta": ["type": "object"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
        ]

        private struct Reconstructed: Equatable {
            var name: String
            var arguments: String
            var completed: ToolCall?
            var streamed: Bool
        }

        private struct Run {
            var segments: [ToolCallStreamSegment] = []
            /// Index in `segments` of the first segment produced by each chunk.
            var chunkStarts: [Int] = []
            var parseFailures = 0

            var text: String {
                segments.reduce(into: "") { text, segment in
                    if case .text(let piece) = segment { text += piece }
                }
            }

            var failures: [ToolCallStreamFailure] {
                segments.compactMap {
                    if case .callFailed(let failure) = $0 { failure } else { nil }
                }
            }

            var calls: [Reconstructed] {
                var calls: [Reconstructed] = []
                for segment in segments {
                    switch segment {
                    case .callStarted(let name):
                        calls.append(
                            .init(name: name, arguments: "", completed: nil, streamed: true))
                    case .callArguments(let fragment):
                        calls[calls.count - 1].arguments += fragment
                    case .callCompleted(let call, let remainder):
                        calls[calls.count - 1].arguments += remainder
                        calls[calls.count - 1].completed = call
                    case .call(let call):
                        let data = try! JSONEncoder().encode(call.function.arguments)
                        calls.append(
                            .init(
                                name: call.function.name,
                                arguments: String(decoding: data, as: UTF8.self), completed: call,
                                streamed: false))
                    case .text, .callFailed:
                        break
                    }
                }
                return calls
            }
        }

        private static func run(
            _ format: ToolCallFormat, _ text: String, width: Int,
            tools: [[String: any Sendable]]? = weatherTools,
            exposes: @escaping @Sendable (String) -> Bool = { _ in true }
        ) -> Run {
            let processor = ToolCallProcessor(
                format: format, tools: tools, streaming: .init(exposesName: exposes))
            var run = Run()
            let characters = Array(text)
            for start in stride(from: 0, to: characters.count, by: width) {
                run.chunkStarts.append(run.segments.count)
                run.segments += processor.processChunkSegments(
                    String(characters[start ..< min(start + width, characters.count)]))
            }
            run.segments += processor.processEOSSegments()
            run.parseFailures = processor.parseFailureCount
            #expect(processor.toolCalls.isEmpty)
            return run
        }

        private static func decoded(_ arguments: String) -> [String: JSONValue]? {
            try? JSONDecoder().decode([String: JSONValue].self, from: Data(arguments.utf8))
        }

        /// The exposed arguments decode to exactly the complete-frame parse.
        private static func expectFaithful(
            _ call: Reconstructed?, name: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            guard let call, let completed = call.completed else {
                Issue.record("missing completed call", sourceLocation: sourceLocation)
                return
            }
            #expect(call.name == name, sourceLocation: sourceLocation)
            #expect(completed.function.name == name, sourceLocation: sourceLocation)
            #expect(
                decoded(call.arguments) == completed.function.arguments,
                sourceLocation: sourceLocation)
        }

        private static let widths = [1, 2, 3, 7, 1000]

        // MARK: - Gemma

        @Test func gemmaStringArgumentsStreamBeforeTheFrameCloses() throws {
            let text =
                #"<|tool_call>call:get_weather{location:<|"|>Paris, France<|"|>,unit:<|"|>celsius<|"|>}<tool_call|>"#
            for width in Self.widths {
                let run = Self.run(.gemma, text, width: width)
                #expect(run.failures.isEmpty)
                #expect(run.text.isEmpty)
                let call = try #require(run.calls.first)
                #expect(run.calls.count == 1)
                #expect(call.streamed)
                Self.expectFaithful(call, name: "get_weather")
                #expect(
                    Self.decoded(call.arguments)
                        == ["location": .string("Paris, France"), "unit": .string("celsius")])
                guard width == 1 else { continue }
                // Name and string content are exposed while the frame is open.
                let started = try #require(
                    run.segments.firstIndex { segment in
                        if case .callStarted = segment { true } else { false }
                    })
                let lastChunkStart = try #require(run.chunkStarts.last)
                #expect(started < lastChunkStart)
                let fragments = run.segments.filter { segment in
                    if case .callArguments = segment { true } else { false }
                }
                #expect(fragments.count > 10)
            }
        }

        @Test func gemmaCompositeValuesAreExposedWhenComplete() {
            let text =
                #"<|tool_call>call:write_file{path:<|"|>a.py<|"|>,mode:420,append:false,meta:{owner:<|"|>me<|"|>,tags:[1,<|"|>x, y<|"|>]},content:"q\"uote"}<tool_call|>"#
            for width in Self.widths {
                let run = Self.run(.gemma, text, width: width)
                #expect(run.failures.isEmpty)
                Self.expectFaithful(run.calls.first, name: "write_file")
                #expect(Self.decoded(run.calls.first?.arguments ?? "")?["mode"] == .int(420))
            }
        }

        /// The lenient complete parse accepts a missing marker; exposure stops
        /// at the deviation and the parse supplies the rest.
        @Test func gemmaLenientOutputCompletesFromTheParse() {
            let text =
                #"<|tool_call>call:get_weather{unit:<|"|>fahrenheit<|"|>,location:Boston, MA}<tool_call|>"#
            for width in Self.widths {
                let run = Self.run(.gemma, text, width: width)
                #expect(run.failures.isEmpty)
                Self.expectFaithful(run.calls.first, name: "get_weather")
                #expect(
                    Self.decoded(run.calls.first?.arguments ?? "")
                        == ["location": .string("Boston, MA"), "unit": .string("fahrenheit")])
            }
        }

        /// A repeated key keeps its last value in the complete parse, which
        /// disagrees with the already exposed first value.
        @Test func gemmaRepeatedKeyAfterExposureDiverges() {
            let text =
                #"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,location:<|"|>Rome<|"|>}<tool_call|>"#
            for width in Self.widths {
                let run = Self.run(.gemma, text, width: width)
                #expect(run.failures == [.diverged])
            }
        }

        @Test func gemmaKeepsTextBeforeTheCallAndEscapesStrings() {
            let value = "say \"hi\"\\n\ttab\u{1}<|x <tool and é漢"
            let text = "Sure. <|tool_call>call:write_file{content:<|\"|>\(value)<|\"|>}<tool_call|>"
            for width in Self.widths {
                let run = Self.run(.gemma, text, width: width)
                #expect(run.text == "Sure. ")
                Self.expectFaithful(run.calls.first, name: "write_file")
                #expect(
                    Self.decoded(run.calls.first?.arguments ?? "")?["content"] == .string(value))
            }
        }

        @Test func gemmaTruncatedFrameIsUnterminated() {
            let text = #"<|tool_call>call:get_weather{location:<|"|>Par"#
            for width in Self.widths {
                let run = Self.run(.gemma, text, width: width)
                #expect(run.calls.first?.streamed == true)
                #expect(run.failures == [.unterminated])
            }
        }

        /// A rejected name is never exposed; the call still completes whole so
        /// the caller can validate (and refuse) it before anything is shown.
        @Test func rejectedNameCompletesInOnePiece() {
            let text = #"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>"#
            for width in Self.widths {
                let run = Self.run(.gemma, text, width: width, exposes: { _ in false })
                #expect(run.calls.count == 1)
                #expect(run.calls.first?.streamed == false)
                #expect(run.calls.first?.completed?.function.name == "get_weather")
                #expect(
                    !run.segments.contains { segment in
                        if case .callStarted = segment { true } else { false }
                    })
            }
        }

        // MARK: - Qwen 3.5

        @Test func qwenXMLStringParametersStreamWithFramingStripped() {
            let content = "def main():\n    print('</param')\n\n"
            let text =
                "<tool_call>\n<function=write_file>\n<parameter=path>\nfib.py\n</parameter>\n"
                + "<parameter=content>\n\(content)\n</parameter>\n</function>\n</tool_call>"
            for width in Self.widths {
                let run = Self.run(.qwen35, text, width: width)
                #expect(run.failures.isEmpty)
                #expect(run.text.isEmpty)
                Self.expectFaithful(run.calls.first, name: "write_file")
                #expect(
                    Self.decoded(run.calls.first?.arguments ?? "")
                        == ["path": .string("fib.py"), "content": .string(content)])
            }
        }

        @Test func qwenXMLConvertedParametersMatchTheParse() {
            let text =
                "<tool_call>\n<function=write_file>\n<parameter=mode>\n420\n</parameter>\n"
                + "<parameter=append>\ntrue\n</parameter>\n<parameter=meta>\n{\"a\": [1, 2]}\n</parameter>\n"
                + "<parameter=path>\r\nx\r\n</parameter>\n</function>\n</tool_call>"
            for width in Self.widths {
                let run = Self.run(.qwen35, text, width: width)
                #expect(run.failures.isEmpty)
                Self.expectFaithful(run.calls.first, name: "write_file")
                let arguments = Self.decoded(run.calls.first?.arguments ?? "")
                #expect(arguments?["mode"] == .int(420))
                #expect(arguments?["append"] == .bool(true))
                #expect(arguments?["path"] == .string("\r\nx\r\n"))
            }
        }

        @Test func qwenJSONDialectStreamsRawArguments() {
            let arguments = #"{"location": "Paris, \"FR\"", "unit": "celsius"}"#
            let text =
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \(arguments)}\n</tool_call>"
            for width in Self.widths {
                let run = Self.run(.qwen35, text, width: width)
                #expect(run.failures.isEmpty)
                Self.expectFaithful(run.calls.first, name: "get_weather")
                #expect(run.calls.first?.arguments == arguments)
            }
        }

        @Test func jsonFormatWithArgumentsBeforeNameStillCompletes() {
            let text =
                #"<tool_call>{"arguments": {"location": "Oslo"}, "name": "get_weather"}</tool_call>"#
            for width in Self.widths {
                let run = Self.run(.json, text, width: width)
                #expect(run.failures.isEmpty)
                Self.expectFaithful(run.calls.first, name: "get_weather")
            }
        }

        @Test func qwenConsecutiveCallsKeepOrderAndTextBetween() {
            let call = { (city: String) in
                "<tool_call>\n<function=get_weather>\n<parameter=location>\n\(city)\n</parameter>\n</function>\n</tool_call>"
            }
            let text = "Checking.\n" + call("Paris") + "\n" + call("Tokyo")
            for width in Self.widths {
                let run = Self.run(.qwen35, text, width: width)
                #expect(run.failures.isEmpty)
                #expect(run.calls.map(\.name) == ["get_weather", "get_weather"])
                #expect(
                    run.calls.compactMap { Self.decoded($0.arguments)?["location"] }
                        == [.string("Paris"), .string("Tokyo")])
                #expect(run.text == "Checking.\n\n")
            }
        }

        @Test func qwenTrailingTextAfterFunctionCloseIsMalformedOnceExposed() {
            let text =
                "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
                + "oops</function>\n</tool_call>"
            for width in Self.widths {
                let run = Self.run(.qwen35, text, width: width)
                #expect(run.calls.first?.streamed == true)
                #expect(run.failures == [.malformed])
                #expect(run.parseFailures == 1)
            }
        }

        /// The Qwen parser rejects undeclared names, so such a frame keeps the
        /// historical visible-text fallback.
        @Test func qwenUndeclaredFunctionKeepsTextFallback() {
            let text = "<tool_call>\n<function=delete_all>\n</function>\n</tool_call>"
            for width in Self.widths {
                let run = Self.run(.qwen35, text, width: width)
                #expect(run.calls.isEmpty)
                #expect(run.text == text)
            }
        }

        // MARK: - Harmony

        @Test func harmonyCallClosedByStopTokenCompletesAtEOS() {
            let text =
                "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json"
                + #"<|message|> {"location":"Paris","unit":"celsius"} "#
            for width in Self.widths {
                let run = Self.run(.harmony, text, width: width)
                #expect(run.failures.isEmpty)
                Self.expectFaithful(run.calls.first, name: "get_weather")
                #expect(run.calls.first?.arguments == #"{"location":"Paris","unit":"celsius"}"#)
                guard width == 1 else { continue }
                // Exposed before EOS: the name at its terminator, arguments as generated.
                let completed = run.segments.firstIndex { segment in
                    if case .callCompleted = segment { true } else { false }
                }
                let firstFragment = run.segments.firstIndex { segment in
                    if case .callArguments = segment { true } else { false }
                }
                #expect(firstFragment != nil && completed != nil && firstFragment! < completed!)
            }
        }

        @Test func harmonyCallWithEndTag() {
            let text =
                "<|channel|>analysis<|message|>Need weather.<|end|>"
                + "<|start|>assistant to=functions.get_weather<|channel|>commentary json"
                + #"<|message|>{"location":"Tokyo"}<|call|>"#
            for width in Self.widths {
                let run = Self.run(.harmony, text, width: width)
                #expect(run.text == "<|channel|>analysis<|message|>Need weather.<|end|>")
                Self.expectFaithful(run.calls.first, name: "get_weather")
            }
        }

        // MARK: - Legacy contract

        /// Without streaming options the segment API is the historical text
        /// API: identical visible text, calls still accumulated in `toolCalls`.
        @Test func noStreamingOptionsKeepsHistoricalOutput() {
            let inputs: [(ToolCallFormat, String)] = [
                (
                    .gemma,
                    #"Sure<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|> done"#
                ),
                (
                    .qwen35,
                    "a <tool_call>\n<function=get_weather>\n<parameter=location>\nX\n</parameter>\n</function>\n</tool_call>b"
                ),
                (.qwen35, "<tool_call>\n<function=nope>\n</function>\n</tool_call>"),
                (
                    .harmony,
                    #"<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"location":"Lima"}"#
                ),
                // GLM5 keeps its response-wide parse: the quoted end tag is data.
                (
                    .glm5,
                    #"Sure <tool_call>write_file<arg_key>path</arg_key><arg_value>a.txt</arg_value><arg_key>meta</arg_key><arg_value>{"tag":"</tool_call>"}</arg_value></tool_call>"#
                ),
            ]
            for (format, text) in inputs {
                for width in Self.widths {
                    let legacy = ToolCallProcessor(format: format, tools: Self.weatherTools)
                    let segmented = ToolCallProcessor(format: format, tools: Self.weatherTools)
                    let characters = Array(text)
                    for start in stride(from: 0, to: characters.count, by: width) {
                        let chunk = String(
                            characters[start ..< min(start + width, characters.count)])
                        let visible = legacy.processChunk(chunk)
                        let segments = segmented.processChunkSegments(chunk)
                        #expect(segments.allSatisfy { if case .text = $0 { true } else { false } })
                        let joined: String? =
                            segments.isEmpty
                            ? nil
                            : segments.reduce(into: "") { text, segment in
                                if case .text(let piece) = segment { text += piece }
                            }
                        #expect(joined == visible)
                    }
                    #expect(
                        segmented.processEOSSegments().map { segment -> String in
                            if case .text(let piece) = segment { piece } else { "" }
                        }.joined() == (legacy.processEOS(returnBufferedText: true) ?? ""))
                    #expect(segmented.toolCalls == legacy.toolCalls)
                    #expect(segmented.parseFailureCount == legacy.parseFailureCount)
                }
            }
        }

        @Test func unsupportedFormatsIgnoreStreamingOptions() {
            let processor = ToolCallProcessor(
                format: .mistral, tools: nil, streaming: .init(exposesName: { _ in true }))
            #expect(!processor.streamsCalls)
            let strict = ToolCallProcessor(
                format: .gemma, tools: nil, strictGemma: true,
                streaming: .init(exposesName: { _ in true }))
            #expect(!strict.streamsCalls)
        }
    }
}
