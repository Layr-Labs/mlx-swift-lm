import Foundation
import MLXLMCommon
import Testing

extension UnitTests {

    /// MiMo V2 and Nemotron frames exposed while they are generated. Every
    /// case is fed one scalar at a time and at every two-chunk split; the
    /// call a client rebuilds from the fragments (name from the first
    /// fragment, arguments concatenated) must be the complete-frame parse,
    /// and the name must leave before the frame closes.
    @Suite
    struct NativeXMLToolCallStreamingTests {

        private static func property(_ schema: [String: any Sendable]) -> [String: any Sendable] { schema }

        private static let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "write_file",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": property(["type": "string"]),
                            "content": property(["type": "string"]),
                            "mode": property(["type": "integer"]),
                            "append": property(["type": "boolean"]),
                            "meta": property(["type": "object"]),
                            "untyped": property(["description": "no type"]),
                            "either": property(["anyOf": [["type": "string"], ["type": "integer"]]]),
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "parameters": [
                        "type": "object",
                        "properties": ["city": property(["type": "string"])] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
        ]

        private static let allowed: Set<String> = ["write_file", "get_weather"]

        private struct Run {
            var segments: [ToolCallStreamSegment] = []
            var nameBeforeClose = false

            var text: String {
                segments.reduce(into: "") { if case .text(let piece) = $1 { $0 += piece } }
            }

            var failures: [ToolCallStreamFailure] {
                segments.compactMap { if case .callFailed(let failure) = $0 { failure } else { nil } }
            }

            /// (name, arguments JSON, completed) per call, as a client sees it.
            var calls: [(name: String, arguments: String, completed: ToolCall?)] {
                var calls: [(name: String, arguments: String, completed: ToolCall?)] = []
                for segment in segments {
                    switch segment {
                    case .callStarted(let name): calls.append((name, "", nil))
                    case .callArguments(let fragment): calls[calls.count - 1].arguments += fragment
                    case .callCompleted(let call, let remainder):
                        calls[calls.count - 1].arguments += remainder
                        calls[calls.count - 1].completed = call
                    case .call(let call):
                        let data = try! JSONEncoder().encode(call.function.arguments)
                        calls.append((call.function.name, String(decoding: data, as: UTF8.self), call))
                    case .text, .callFailed: break
                    }
                }
                return calls
            }
        }

        private static func run(_ format: ToolCallFormat, _ chunks: [String]) -> Run {
            let processor = ToolCallProcessor(
                format: format, tools: tools,
                streaming: ToolCallStreamingOptions(exposesName: { allowed.contains($0) }))
            var run = Run()
            for chunk in chunks {
                run.segments += processor.processChunkSegments(chunk)
                if run.segments.contains(where: {
                    if case .callStarted = $0 { true } else { false }
                }), !run.segments.contains(where: {
                    switch $0 {
                    case .callCompleted, .callFailed, .call: true
                    default: false
                    }
                }) {
                    run.nameBeforeClose = true
                }
            }
            run.segments += processor.processEOSSegments()
            return run
        }

        private static func complete(_ format: ToolCallFormat, _ text: String) -> (calls: [ToolCall], text: String) {
            let processor = ToolCallProcessor(format: format, tools: tools)
            let visible = (processor.processChunk(text) ?? "") + (processor.processEOS(returnBufferedText: true) ?? "")
            return (processor.toolCalls, visible)
        }

        private static func decode(_ json: String) -> [String: JSONValue]? {
            try? JSONDecoder().decode([String: JSONValue].self, from: Data(json.utf8))
        }

        private static func splits(_ text: String) -> [[String]] {
            let scalars = Array(text.unicodeScalars)
            var all = [scalars.map { String($0) }]
            for cut in 0 ... scalars.count {
                all.append([
                    String(String.UnicodeScalarView(scalars[..<cut])),
                    String(String.UnicodeScalarView(scalars[cut...])),
                ])
            }
            return all
        }

        /// The streamed calls agree with the complete parse at every split.
        private static func checkAgreement(
            _ format: ToolCallFormat, _ text: String, streamsName: Bool = true,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let expected = complete(format, text)
            #expect(!expected.calls.isEmpty, "fixture parses", sourceLocation: sourceLocation)
            for chunks in splits(text) {
                let run = run(format, chunks)
                #expect(run.failures.isEmpty, "\(chunks.count) chunks", sourceLocation: sourceLocation)
                let calls = run.calls
                #expect(calls.count == expected.calls.count, sourceLocation: sourceLocation)
                for (call, whole) in zip(calls, expected.calls) {
                    #expect(call.name == whole.function.name, sourceLocation: sourceLocation)
                    #expect(decode(call.arguments) == whole.function.arguments,
                        "\(call.arguments)", sourceLocation: sourceLocation)
                }
                #expect(run.text == expected.text, sourceLocation: sourceLocation)
                if chunks.count > 2, streamsName {
                    #expect(run.nameBeforeClose, sourceLocation: sourceLocation)
                }
            }
        }

        // MARK: - MiMo V2

        private static func mimo(_ name: String, _ parameters: [(String, String)]) -> String {
            "<tool_call><function=\(name)>"
                + parameters.map { "<parameter=\($0.0)>\($0.1)</parameter>" }.joined()
                + "</function></tool_call>"
        }

        @Test func mimoRawStringsStreamByteForByte() {
            let value = "line 1\n  </tool_call><think>x</think> </function> &amp; \"q\" \\ café e\u{301}\r\n"
            Self.checkAgreement(.mimoV2, Self.mimo("write_file", [("path", "a.py"), ("content", value)]))
            // Framing whitespace between tags, and text around the frame.
            Self.checkAgreement(.mimoV2,
                "Writing.\n<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>\n"
                    + "</function>\n</tool_call>\nDone.")
        }

        @Test func mimoTypedAndUnionValuesLeaveConvertedWhenTheyClose() {
            Self.checkAgreement(.mimoV2, Self.mimo("write_file", [
                ("path", "a.py"), ("mode", "420"), ("append", "true"), ("meta", #"{"k":[1,"two"]}"#),
                ("untyped", "kept as text"), ("either", "7"), ("undeclared", "raw"),
            ]))
        }

        @Test func mimoStringValueLeavesBeforeItsParameterCloses() {
            let processor = ToolCallProcessor(
                format: .mimoV2, tools: Self.tools,
                streaming: ToolCallStreamingOptions(exposesName: { Self.allowed.contains($0) }))
            let segments = processor.processChunkSegments(
                "<tool_call><function=write_file><parameter=content>print('hi')\n")
            #expect(segments.contains(.callStarted(name: "write_file")))
            let arguments = segments.compactMap { if case .callArguments(let a) = $0 { a } else { nil } }.joined()
            #expect(arguments == #"{"content":"print('hi')\n"#)
        }

        @Test func mimoTwoCallsStreamInOrder() {
            Self.checkAgreement(.mimoV2,
                Self.mimo("get_weather", [("city", "Paris")]) + "\n" + Self.mimo("write_file", [("path", "b")]))
        }

        /// A frame the parser refuses after its name left fails; one that
        /// never named a declared function completes (or not) in one piece.
        @Test func mimoFramesThatDoNotParseFail() {
            let duplicate = Self.run(.mimoV2, Self.mimo("write_file", [("path", "a"), ("path", "b")]).map { String($0) })
            #expect(duplicate.failures == [.malformed])
            let open = Self.run(.mimoV2, ["<tool_call><function=write_file><parameter=path>a"])
            #expect(open.failures == [.unterminated])
            let undeclared = Self.run(.mimoV2, Self.mimo("rm_rf", [("path", "/")]).map { String($0) })
            #expect(undeclared.calls.isEmpty && undeclared.failures.isEmpty)
            let json = Self.run(.mimoV2,
                [#"<tool_call>{"name":"write_file","arguments":{"path":"a"}}</tool_call>"#])
            #expect(json.calls.isEmpty && json.failures.isEmpty)
        }

        // MARK: - Nemotron

        private static func nemotron(_ name: String, _ parameters: [(String, String)], wrapped: Bool = true)
            -> String
        {
            let function = "<function=\(name)>\n"
                + parameters.map { "<parameter=\($0.0)>\n\($0.1)\n</parameter>\n" }.joined() + "</function>"
            return wrapped ? "<tool_call>\n" + function + "\n</tool_call>" : function
        }

        @Test func nemotronStringsStreamAndTypedValuesLeaveWhenTheyClose() {
            Self.checkAgreement(.nemotron, Self.nemotron("write_file", [
                ("path", "a.py"), ("content", "def f():\n    return \"x\" </function> café\n"), ("mode", "420"),
                ("append", "false"), ("meta", #"{"k":1}"#),
            ]))
        }

        @Test func nemotronBareFunctionFramesStreamToo() {
            Self.checkAgreement(.nemotron,
                "Sure.\n" + Self.nemotron("write_file", [("path", "a.py"), ("content", "x")], wrapped: false))
        }

        @Test func nemotronTwoCallsStreamInOrder() {
            Self.checkAgreement(.nemotron,
                Self.nemotron("get_weather", [("city", "Paris")]) + "\n" + Self.nemotron("write_file", [("path", "b")]))
        }

        @Test func nemotronFramesThatDoNotParseFailOrStayText() {
            let open = Self.run(.nemotron, ["<tool_call>\n<function=write_file>\n<parameter=path>\na"])
            #expect(open.failures == [.unterminated])
            // The JSON dialect is not Nemotron's: never exposed, text as before.
            let json = #"<tool_call>{"name":"write_file","arguments":{"path":"a"}}</tool_call>"#
            let run = Self.run(.nemotron, json.map { String($0) })
            #expect(run.calls.isEmpty && run.failures.isEmpty)
            #expect(run.text == Self.complete(.nemotron, json).text)
        }

        /// Formats without these streamers keep complete frames.
        @Test func onlyTheNativeXMLFormatsAreAdded() {
            #expect(ToolCallStreamingOptions.supports(.mimoV2))
            #expect(ToolCallStreamingOptions.supports(.nemotron))
            #expect(!ToolCallStreamingOptions.supports(.xmlFunction))
            #expect(!ToolCallStreamingOptions.supports(.glm5))
            #expect(!ToolCallStreamingOptions.supports(.flint))
        }
    }
}
