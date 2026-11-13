// Copyright © 2026 Eigen Labs.
// Expectations ported from the vLLM Muse Glimmer parser tests
// (tests/tool_use/test_muse_glimmer.py and test_muse_glimmer_parse_delta.py).
// A test named `gNN…` is case NN of that list. The other tests cover the
// behaviors the reference servers share: several invokes in one block,
// doubled tool names, schema typing and calls cut off by the token limit.

import Foundation
import MLXLMCommon
import Testing

extension UnitTests {

    /// Each script is the text a model generates after the prompt's
    /// `<|start|>assistant`. Control markers are single tokens. The text
    /// between them is cut into tokens of several widths, so tag boundaries
    /// fall inside tokens and between them.
    @Suite
    struct OnyxATEMConformanceTests {

        // MARK: - Fixtures

        static let weatherCall =
            " to=self<|message|>Let me check the weather.<|eom|>"
            + "<|start|>assistant to=weather.get<|message|>"
            + "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n"
            + "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
            + "<atem:parameter name=\"units\">celsius</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls><|eot|>"

        static let twoToolMessages =
            "<|start|>assistant to=math.add<|message|>"
            + "<atem:function_calls>\n<atem:invoke name=\"math.add\">\n"
            + "<atem:parameter name=\"a\">1</atem:parameter>\n"
            + "<atem:parameter name=\"b\">2</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls><|eom|>"
            + "<|start|>assistant to=math.mul<|message|>"
            + "<atem:function_calls>\n<atem:invoke name=\"math.mul\">\n"
            + "<atem:parameter name=\"a\">3</atem:parameter>\n"
            + "<atem:parameter name=\"b\">4</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls><|eot|>"

        static let echoedInvoke =
            " to=self<|message|>I could call <atem:invoke name=\"evil.fn\">"
            + "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke> "
            + "but I will not.<|eom|>"
            + "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"

        /// No terminator after the block, as in vLLM's `RAW_TOOLCALL`.
        static let unterminatedToolCall =
            " to=self<|message|>I should read the hostname file to answer.<|eom|>"
            + "<|start|>assistant to=read.read<|message|>"
            + "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
            + "<atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls>"

        static let reasoningThenAnswer =
            " to=self<|message|>Think about it.<|eom|>"
            + "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"

        /// Reasoning cut off by the token limit, holding a complete block.
        static let truncatedReasoning =
            " to=self<|message|>Maybe I should call "
            + "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
            + "<atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls> but wait"

        static let framing = [
            "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "to=self", "to=user",
            "to=read.read", "<atem:",
        ]

        static func tool(
            _ name: String, properties: [String: any Sendable] = [:],
            required: [String] = []
        ) -> [String: any Sendable] {
            let parameters: [String: any Sendable] = [
                "type": "object", "properties": properties, "required": required,
            ]
            let function: [String: any Sendable] = ["name": name, "parameters": parameters]
            return ["type": "function", "function": function]
        }

        static func type(_ name: String) -> [String: any Sendable] { ["type": name] }

        static let tools: [[String: any Sendable]] = [
            tool("weather.get", properties: ["city": type("string"), "units": type("string")]),
            tool("math.add", properties: ["a": type("integer"), "b": type("integer")]),
            tool("math.mul", properties: ["a": type("integer"), "b": type("integer")]),
            tool("read.read", properties: ["path": type("string")]),
            tool("evil.fn"),
            tool("fs.list", properties: ["path": type("string")]),
            tool("get_weather", properties: ["city": type("string")]),
            tool("time.get"),
            tool(
                "api.call",
                properties: ["payload": type("object"), "flag": type("boolean")]),
        ]

        // MARK: - Driver

        struct Outcome {
            var reasoning = ""
            var content = ""
            var calls: [ToolCall] = []
            var rejections: [RejectedToolCall] = []
            var protocolErrors: [String] = []
            var names: [String] { calls.map(\.function.name) }
        }

        static func makeDecoder(
            tokenizer: any Tokenizer, tools: [[String: any Sendable]]?,
            validation: ToolCallValidationPolicy = .permissive
        ) -> (any TokenStreamDecoder)? {
            ToolCallFormat.atem.makeProtocolTokenStreamDecoder(
                tokenizer: tokenizer, tools: tools, stopStrings: [],
                toolCallPolicy: .init(validation: validation, onyxParsing: .referenceParity))
        }

        /// Feeds `script` cut into `width`-character tokens, then ends the
        /// generation. A stop token is passed to the decoder and ends the run.
        static func run(
            _ script: String, width: Int = 3,
            tools: [[String: any Sendable]]? = OnyxATEMConformanceTests.tools,
            validation: ToolCallValidationPolicy = .permissive
        ) throws -> Outcome {
            let tokenizer = OnyxScriptTokenizer()
            var decoder = try #require(
                makeDecoder(tokenizer: tokenizer, tools: tools, validation: validation))
            var outcome = Outcome()
            let record: (TokenStreamEvent) -> Bool = { event in
                switch event {
                case .reasoning(let text): outcome.reasoning += text
                case .response(let text): outcome.content += text
                case .toolCall(let call): outcome.calls.append(call)
                case .rejectedToolCall(let rejection): outcome.rejections.append(rejection)
                case .protocolError(let message): outcome.protocolErrors.append(message)
                case .stop: break
                }
                return true
            }
            for token in tokenizer.tokens(script, width: width) {
                _ = decoder.push(token, emit: record)
                if decoder.additionalStopTokenIDs.contains(token) { break }
            }
            _ = decoder.finish(emit: record)
            return outcome
        }

        static func expectNoFraming(_ outcome: Outcome, width: Int) {
            for marker in framing {
                #expect(
                    !outcome.reasoning.contains(marker),
                    "\(marker) leaked into reasoning at width \(width)")
                #expect(
                    !outcome.content.contains(marker),
                    "\(marker) leaked into content at width \(width)")
            }
        }

        // MARK: - vLLM cases

        @Test func g01_singleCallAfterReasoning() throws {
            let outcome = try Self.run(Self.weatherCall)
            #expect(outcome.reasoning == "Let me check the weather.")
            #expect(outcome.content.isEmpty)
            #expect(outcome.names == ["weather.get"])
            #expect(
                outcome.calls.first?.function.arguments
                    == ["city": .string("Paris"), "units": .string("celsius")])
        }

        @Test func g02_twoToolMessagesGiveTwoCallsInOrder() throws {
            let outcome = try Self.run(Self.twoToolMessages)
            #expect(outcome.names == ["math.add", "math.mul"])
            #expect(outcome.calls.first?.function.arguments == ["a": .int(1), "b": .int(2)])
            #expect(outcome.calls.last?.function.arguments == ["a": .int(3), "b": .int(4)])
            #expect(Set(outcome.calls.compactMap(\.id)).count == 2)
        }

        @Test func g03_invokeEchoedInReasoningIsNotACall() throws {
            let outcome = try Self.run(Self.echoedInvoke)
            #expect(outcome.calls.isEmpty)
            #expect(outcome.rejections.isEmpty)
            #expect(outcome.content == "The answer is 42.")
            #expect(outcome.reasoning.contains("<atem:invoke name=\"evil.fn\">"))
        }

        @Test func g04_plainAnswerHasNoCalls() throws {
            let outcome = try Self.run(" to=user<|message|>Just a plain answer.<|eot|>")
            #expect(outcome.calls.isEmpty)
            #expect(outcome.content == "Just a plain answer.")
            #expect(outcome.reasoning.isEmpty)
        }

        @Test func g05_objectArrayAndBooleanValuesDecode() throws {
            let script =
                "<|start|>assistant to=api.call<|message|>"
                + "<atem:function_calls>\n<atem:invoke name=\"api.call\">\n"
                + "<atem:parameter name=\"payload\">{\"nested\": [1, 2, 3]}</atem:parameter>\n"
                + "<atem:parameter name=\"flag\">true</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls><|eot|>"
            let expected: [String: JSONValue] = [
                "payload": .object(["nested": .array([.int(1), .int(2), .int(3)])]),
                "flag": .bool(true),
            ]
            #expect(try Self.run(script).calls.first?.function.arguments == expected)
            // vLLM does not read the schema. A tool that declares no
            // properties gives the same values.
            let untyped = [Self.tool("api.call")]
            #expect(
                try Self.run(script, tools: untyped).calls.first?.function.arguments == expected)
        }

        @Test func g06_callNeedsNoTrailingTerminator() throws {
            let script =
                " to=self<|message|>Let me call the tool.<|eom|>"
                + "<|start|>assistant to=weather.get<|message|>"
                + "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n"
                + "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls>"
            let outcome = try Self.run(script)
            #expect(outcome.reasoning == "Let me call the tool.")
            #expect(outcome.names == ["weather.get"])
            #expect(outcome.calls.first?.function.arguments == ["city": .string("Paris")])
        }

        @Test func g07_reasoningThenAnswer() throws {
            let script =
                " to=self<|message|>thinking<|eom|>"
                + "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
            let outcome = try Self.run(script)
            #expect(outcome.reasoning == "thinking")
            #expect(outcome.content == "The answer is 42.")
            #expect(outcome.calls.isEmpty)
        }

        @Test func g08_unframedTextIsContent() throws {
            let outcome = try Self.run("Just a direct answer.")
            #expect(outcome.reasoning.isEmpty)
            #expect(outcome.content == "Just a direct answer.")
            #expect(outcome.calls.isEmpty)
        }

        @Test func g09_reasoningThenTwoCalls() throws {
            let script =
                " to=self<|message|>need two calls<|eom|>"
                + "<|start|>assistant to=math.add<|message|>"
                + "<atem:function_calls>\n<atem:invoke name=\"math.add\">\n"
                + "<atem:parameter name=\"a\">1</atem:parameter>\n</atem:invoke>\n"
                + "</atem:function_calls><|eom|>"
                + "<|start|>assistant to=math.mul<|message|>"
                + "<atem:function_calls>\n<atem:invoke name=\"math.mul\">\n"
                + "<atem:parameter name=\"a\">3</atem:parameter>\n</atem:invoke>\n"
                + "</atem:function_calls><|eot|>"
            let outcome = try Self.run(script)
            #expect(outcome.reasoning == "need two calls")
            #expect(outcome.names == ["math.add", "math.mul"])
        }

        @Test(arguments: [1, 3, 17])
        func g10_g11_streamedToolCall(width: Int) throws {
            let outcome = try Self.run(Self.unterminatedToolCall, width: width)
            Self.expectNoFraming(outcome, width: width)
            #expect(outcome.calls.count == 1)
            #expect(outcome.calls.first?.function.name == "read.read")
            #expect(
                outcome.calls.first?.function.arguments == ["path": .string("/etc/hostname")])
            #expect(outcome.calls.first?.id?.isEmpty == false)
            #expect(outcome.reasoning == "I should read the hostname file to answer.")
            #expect(outcome.content.isEmpty)
        }

        @Test(arguments: [1, 3, 17])
        func g12_streamedReasoningThenContent(width: Int) throws {
            let outcome = try Self.run(Self.reasoningThenAnswer, width: width)
            Self.expectNoFraming(outcome, width: width)
            #expect(outcome.reasoning == "Think about it.")
            #expect(outcome.content == "The answer is 42.")
            #expect(outcome.calls.isEmpty)
        }

        @Test(arguments: [1, 3, 17])
        func g13_g26_truncatedReasoningYieldsNoCall(width: Int) throws {
            let outcome = try Self.run(Self.truncatedReasoning, width: width)
            #expect(outcome.calls.isEmpty)
            #expect(outcome.rejections.isEmpty)
            #expect(outcome.content.isEmpty)
            // The partial reasoning is recovered whole, markup included.
            #expect(
                outcome.reasoning
                    == String(Self.truncatedReasoning.dropFirst(" to=self<|message|>".count)))
        }

        @Test func g14_doubledBareNameCollapses() throws {
            let outcome = try Self.run(Self.call("get_weather.get_weather"))
            #expect(outcome.names == ["get_weather"])
            #expect(outcome.calls.first?.function.arguments == ["city": .string("Paris")])
        }

        @Test func g15_namespacedNameIsKept() throws {
            let tools = [Self.tool("weather.get", properties: ["city": Self.type("string")])]
            #expect(try Self.run(Self.call("weather.get"), tools: tools).names == ["weather.get"])
        }

        /// vLLM passes these names through unchanged, so the client receives a
        /// call to a tool it never offered. Here an undeclared name never
        /// becomes a call; it is reported as a rejection.
        @Test(arguments: ["foo.get_weather", "x.get"])
        func g16_g17_undeclaredNameIsRejected(name: String) throws {
            let tools = [Self.tool("get_weather"), Self.tool("weather.get"), Self.tool("time.get")]
            let outcome = try Self.run(Self.call(name), tools: tools)
            #expect(outcome.calls.isEmpty)
            #expect(outcome.rejections.map(\.reason) == [.undeclaredTool])
            #expect(outcome.rejections.first?.toolName == name)
        }

        /// vLLM passes every name through when the request has no tools.
        @Test func g18_noToolsMeansNoCalls() throws {
            for tools in [nil, []] as [[[String: any Sendable]]?] {
                let outcome = try Self.run(Self.call("get_weather.get_weather"), tools: tools)
                #expect(outcome.calls.isEmpty)
                #expect(outcome.rejections.map(\.reason) == [.undeclaredTool])
            }
        }

        @Test func g19_exactNameIsKept() throws {
            #expect(try Self.run(Self.call("get_weather")).names == ["get_weather"])
        }

        @Test(arguments: [1, 3, 100])
        func g21_recipientLessMessageIsContent(width: Int) throws {
            let outcome = try Self.run("<|message|>Hello.<|eot|>", width: width)
            #expect(outcome.content == "Hello.")
            #expect(outcome.reasoning.isEmpty)
            #expect(outcome.calls.isEmpty)
        }

        @Test func g23_requestWithoutToolsStillSplitsReasoning() throws {
            let script =
                " to=self<|message|>Let me think step by step about the sum.<|eom|>"
                + "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
            for width in [1, 3, 17] {
                let outcome = try Self.run(script, width: width, tools: nil)
                Self.expectNoFraming(outcome, width: width)
                #expect(outcome.reasoning == "Let me think step by step about the sum.")
                #expect(outcome.content == "The answer is 42.")
                #expect(outcome.calls.isEmpty)
            }
        }

        @Test func g24_contentOnly() throws {
            let outcome = try Self.run(" to=user<|message|>Just a direct answer.<|eot|>")
            #expect(outcome.content == "Just a direct answer.")
            #expect(outcome.reasoning.isEmpty)
            #expect(outcome.calls.isEmpty)
        }

        // MARK: - Shared reference behavior

        static func call(_ name: String) -> String {
            "<|start|>assistant to=\(name)<|message|>"
                + "<atem:function_calls>\n<atem:invoke name=\"\(name)\">\n"
                + "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls><|eot|>"
        }

        @Test(arguments: [1, 3, 17])
        func severalInvokesInOneBlockGiveCallsInOrder(width: Int) throws {
            let script =
                " to=math.add<|message|><atem:function_calls>\n"
                + "<atem:invoke name=\"math.add\">\n"
                + "<atem:parameter name=\"a\">1</atem:parameter>\n"
                + "<atem:parameter name=\"b\">2</atem:parameter>\n</atem:invoke>\n"
                + "<atem:invoke name=\"math.mul\">\n"
                + "<atem:parameter name=\"a\">3</atem:parameter>\n"
                + "<atem:parameter name=\"b\">4</atem:parameter>\n</atem:invoke>\n"
                + "<atem:invoke name=\"time.get\">\n</atem:invoke>\n"
                + "</atem:function_calls><|eot|>"
            let outcome = try Self.run(script, width: width)
            #expect(outcome.names == ["math.add", "math.mul", "time.get"])
            #expect(outcome.calls.first?.function.arguments == ["a": .int(1), "b": .int(2)])
            #expect(outcome.calls.last?.function.arguments.isEmpty == true)
            #expect(Set(outcome.calls.compactMap(\.id)).count == 3)
            #expect(outcome.content.isEmpty)
        }

        /// The references read the name from the invoke and never compare it
        /// with the frame's recipient.
        @Test func recipientIsNotComparedWithTheInvokeName() throws {
            let script =
                " to=weather<|message|><atem:function_calls>\n"
                + "<atem:invoke name=\"weather.get\">\n"
                + "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls><|eot|>"
            #expect(try Self.run(script).names == ["weather.get"])
        }

        @Test func textAroundABlockInAToolFrameIsDropped() throws {
            let script =
                " to=weather.get<|message|>Calling now.\n<atem:function_calls>\n"
                + "<atem:invoke name=\"weather.get\">\n"
                + "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls>\nDone.<|eot|>"
            let outcome = try Self.run(script)
            #expect(outcome.names == ["weather.get"])
            #expect(outcome.content.isEmpty)
            #expect(outcome.reasoning.isEmpty)
        }

        @Test func answerAfterAToolMessageIsContent() throws {
            let script =
                " to=weather.get<|message|><atem:function_calls>\n"
                + "<atem:invoke name=\"weather.get\">\n"
                + "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls><|eom|>"
                + "<|start|>assistant to=user<|message|>Checking.<|eot|>"
            let outcome = try Self.run(script)
            #expect(outcome.names == ["weather.get"])
            #expect(outcome.content == "Checking.")
        }

        @Test(arguments: [1, 3, 17])
        func stringValuesAreByteExact(width: Int) throws {
            let value = "  line one\n\tline <atem:value> two </atem:invoke>\n\n  "
            let script =
                " to=read.read<|message|><atem:function_calls>\n"
                + "<atem:invoke name=\"read.read\">\n"
                + "<atem:parameter name=\"path\">\(value)</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls><|eot|>"
            let outcome = try Self.run(script, width: width)
            #expect(outcome.calls.first?.function.arguments == ["path": .string(value)])
        }

        @Test func valuesFollowTheDeclaredTypes() throws {
            let nullable: [String: any Sendable] = ["type": ["string", "null"]]
            let tools = [
                Self.tool(
                    "typed",
                    properties: [
                        "text": Self.type("string"), "digits": Self.type("string"),
                        "word": Self.type("string"), "json": Self.type("string"),
                        "count": Self.type("integer"), "ratio": Self.type("number"),
                        "flag": Self.type("boolean"), "items": Self.type("array"),
                        "shape": Self.type("object"), "note": nullable,
                        "empty": Self.type("string"),
                    ])
            ]
            let script =
                " to=typed<|message|><atem:function_calls>\n<atem:invoke name=\"typed\">\n"
                + "<atem:parameter name=\"text\"> spaced </atem:parameter>\n"
                + "<atem:parameter name=\"digits\">123</atem:parameter>\n"
                + "<atem:parameter name=\"word\">true</atem:parameter>\n"
                + "<atem:parameter name=\"json\">{\"a\": 1}</atem:parameter>\n"
                + "<atem:parameter name=\"count\">7</atem:parameter>\n"
                + "<atem:parameter name=\"ratio\">0.5</atem:parameter>\n"
                + "<atem:parameter name=\"flag\">false</atem:parameter>\n"
                + "<atem:parameter name=\"items\">[\"x\", 2, null]</atem:parameter>\n"
                + "<atem:parameter name=\"shape\">{\"w\": 2, \"tags\": [\"a\"]}</atem:parameter>\n"
                + "<atem:parameter name=\"note\">null</atem:parameter>\n"
                + "<atem:parameter name=\"empty\"></atem:parameter>\n"
                + "<atem:parameter name=\"extra\">12</atem:parameter>\n"
                + "<atem:parameter name=\"plain\">not json</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls><|eot|>"
            let arguments = try #require(try Self.run(script, tools: tools).calls.first)
                .function.arguments
            #expect(arguments["text"] == .string(" spaced "))
            #expect(arguments["digits"] == .string("123"))
            #expect(arguments["word"] == .string("true"))
            #expect(arguments["json"] == .string("{\"a\": 1}"))
            #expect(arguments["count"] == .int(7))
            #expect(arguments["ratio"] == .double(0.5))
            #expect(arguments["flag"] == .bool(false))
            #expect(arguments["items"] == .array([.string("x"), .int(2), .null]))
            #expect(
                arguments["shape"]
                    == .object(["w": .int(2), "tags": .array([.string("a")])]))
            #expect(arguments["note"] == .null)
            #expect(arguments["empty"] == .string(""))
            // Undeclared parameters decode as JSON when they parse, as in vLLM.
            #expect(arguments["extra"] == .int(12))
            #expect(arguments["plain"] == .string("not json"))
        }

        // MARK: - Tool-frame bodies without the stream

        static func bodyCalls(
            _ body: String, tools: [[String: any Sendable]]? = OnyxATEMConformanceTests.tools
        ) -> [ToolCall] {
            // A tool frame that holds only `body`.
            var decoder = OnyxTextStreamDecoder(tools: tools)
            let events = decoder.process(" to=tool<|message|>" + body) + decoder.finish()
            return events.compactMap { event in
                if case .toolCall(let call) = event { return call }
                return nil
            }
        }

        static func invoke(_ name: String, _ parameters: [(String, String)] = []) -> String {
            "<atem:invoke name=\"\(name)\">\n"
                + parameters.map {
                    "<atem:parameter name=\"\($0.0)\">\($0.1)</atem:parameter>\n"
                }.joined() + "</atem:invoke>"
        }

        static func block(_ invokes: String...) -> String {
            "<atem:function_calls>\n" + invokes.joined(separator: "\n")
                + "\n</atem:function_calls>"
        }

        @Test func bodyWithSeveralInvokes() {
            let calls = Self.bodyCalls(
                Self.block(
                    Self.invoke("math.add", [("a", "1"), ("b", "2")]),
                    Self.invoke("weather.get", [("city", "Paris")])))
            #expect(calls.map(\.function.name) == ["math.add", "weather.get"])
            #expect(calls.first?.function.arguments == ["a": .int(1), "b": .int(2)])
            #expect(calls.last?.function.arguments == ["city": .string("Paris")])
        }

        @Test func bodyWithDoubledName() {
            let calls = Self.bodyCalls(
                Self.block(Self.invoke("get_weather.get_weather", [("city", "Paris")])))
            #expect(calls.map(\.function.name) == ["get_weather"])
        }

        @Test func bodyWithoutTheWrapper() {
            let calls = Self.bodyCalls(Self.invoke("weather.get", [("city", "Paris")]))
            #expect(calls.map(\.function.name) == ["weather.get"])
        }

        @Test func bodyWithTextAroundTheBlock() {
            let calls = Self.bodyCalls(
                "First.\n" + Self.block(Self.invoke("weather.get", [("city", "Paris")]))
                    + "\nThen done.")
            #expect(calls.map(\.function.name) == ["weather.get"])
        }

        @Test func bodyKeepsStringBytes() {
            let value = "\n  two  spaces\t</atem:invoke> <atem:invoke name=\"x\">\n"
            let calls = Self.bodyCalls(Self.block(Self.invoke("read.read", [("path", value)])))
            #expect(calls.first?.function.arguments == ["path": .string(value)])
        }

        @Test func bodyCutOffYieldsNoCall() throws {
            let whole = Self.block(Self.invoke("weather.get", [("city", "Paris")]))
            let close = try #require(whole.range(of: "</atem:invoke>"))
            for end in whole.indices where end < close.upperBound {
                #expect(
                    Self.bodyCalls(String(whole[..<end])).isEmpty,
                    "a call was read from a body cut at \(whole.distance(from: whole.startIndex, to: end))"
                )
            }
        }

        @Test func proseAboutToolsIsNotACall() {
            for text in [
                "I would use weather.get with city Paris.",
                "<atem:function_calls> is the wrapper and <atem:invoke is the element.",
                "<atem:invoke name=\"weather.get\"> opens a call that is never closed",
                "weather.get(city=\"Paris\")",
                "{\"name\": \"weather.get\", \"arguments\": {\"city\": \"Paris\"}}",
            ] {
                #expect(Self.bodyCalls(text).isEmpty, "\(text)")
            }
        }

        // MARK: - Calls cut off by the token limit

        @Test(arguments: [1, 3, 17])
        func callCutOffInsideAParameterYieldsNoCall(width: Int) throws {
            let script =
                " to=weather.get<|message|><atem:function_calls>\n"
                + "<atem:invoke name=\"weather.get\">\n"
                + "<atem:parameter name=\"city\">Par"
            let outcome = try Self.run(script, width: width)
            #expect(outcome.calls.isEmpty)
            #expect(outcome.rejections.map(\.reason) == [.incompleteOutput])
            #expect(outcome.content.isEmpty)
        }

        @Test func callCutOffBeforeTheInvokeClosesYieldsNoCall() throws {
            // Every parameter is complete, but the model could still add one.
            let script =
                " to=weather.get<|message|><atem:function_calls>\n"
                + "<atem:invoke name=\"weather.get\">\n"
                + "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
            let outcome = try Self.run(script)
            #expect(outcome.calls.isEmpty)
            #expect(outcome.rejections.map(\.reason) == [.incompleteOutput])
        }

        @Test func completeInvokeBeforeACutOffOneIsKept() throws {
            let script =
                " to=math.add<|message|><atem:function_calls>\n"
                + "<atem:invoke name=\"math.add\">\n"
                + "<atem:parameter name=\"a\">1</atem:parameter>\n</atem:invoke>\n"
                + "<atem:invoke name=\"math.mul\">\n<atem:parameter name=\"a\">3"
            let outcome = try Self.run(script)
            #expect(outcome.names == ["math.add"])
            #expect(outcome.calls.first?.function.arguments == ["a": .int(1)])
            #expect(outcome.rejections.map(\.reason) == [.incompleteOutput])
        }

        @Test func truncatedAnswerKeepsItsText() throws {
            let outcome = try Self.run(
                " to=self<|message|>plan<|eom|><|start|>assistant to=user<|message|>The ans")
            #expect(outcome.reasoning == "plan")
            #expect(outcome.content == "The ans")
        }

        // MARK: - Reasoning left open before a tool header

        /// vLLM bounds an open reasoning block at the next `to=…<|message|>`
        /// header. The model leaves reasoning this way for calls without
        /// arguments.
        @Test(arguments: [1, 3, 17])
        func bareToolHeaderEndsOpenReasoning(width: Int) throws {
            let script =
                " to=self<|message|>I will list the files. to=fs.list<|message|>"
                + "<atem:function_calls>\n<atem:invoke name=\"fs.list\">\n"
                + "</atem:invoke>\n</atem:function_calls><|eot|>"
            let outcome = try Self.run(script, width: width)
            Self.expectNoFraming(outcome, width: width)
            #expect(!outcome.reasoning.contains("to=fs.list"))
            #expect(outcome.reasoning == "I will list the files.")
            #expect(outcome.names == ["fs.list"])
            #expect(outcome.calls.first?.function.arguments.isEmpty == true)
            #expect(outcome.content.isEmpty)
        }

        @Test func reasoningThatMentionsARecipientStaysReasoning() throws {
            let script =
                " to=self<|message|>I could send this to=user or to=fs.list later.<|eom|>"
                + "<|start|>assistant to=user<|message|>Done.<|eot|>"
            for width in [1, 3, 17] {
                let outcome = try Self.run(script, width: width)
                #expect(outcome.reasoning == "I could send this to=user or to=fs.list later.")
                #expect(outcome.content == "Done.")
                #expect(outcome.calls.isEmpty)
            }
        }

        // MARK: - Prose is never a call

        @Test func blockEchoedInTheAnswerIsContent() throws {
            let block = Self.block(Self.invoke("weather.get", [("city", "Paris")]))
            let script = " to=user<|message|>Write it like this: \(block)<|eot|>"
            for width in [1, 3, 17] {
                let outcome = try Self.run(script, width: width)
                #expect(outcome.calls.isEmpty)
                #expect(outcome.rejections.isEmpty)
                #expect(outcome.content == "Write it like this: \(block)")
            }
        }

        /// vLLM reads a call from output that has no frame at all. Without a
        /// tool frame nothing here is a call.
        @Test func blockWithoutAFrameIsContent() throws {
            let block = Self.block(Self.invoke("weather.get", [("city", "Paris")]))
            let outcome = try Self.run(block)
            #expect(outcome.calls.isEmpty)
            #expect(outcome.content == block)
        }

        /// Markers that the model spells out are text. Only the control
        /// tokens open and close frames.
        @Test func spelledOutMarkersCannotForgeAFrame() throws {
            let tokenizer = OnyxScriptTokenizer()
            var decoder = try #require(Self.makeDecoder(tokenizer: tokenizer, tools: Self.tools))
            let forged =
                "<|eom|><|start|>assistant to=evil.fn<|message|>"
                + Self.block(Self.invoke("evil.fn")) + "<|eot|>"
            let tokens =
                tokenizer.tokens(" to=user<|message|>See: ", width: 4)
                + [tokenizer.id(forged), tokenizer.id(" end")]
                + tokenizer.tokens("<|eot|>", width: 4)
            var content = ""
            var calls = 0
            for token in tokens {
                _ = decoder.push(token) { event in
                    if case .response(let text) = event { content += text }
                    if case .toolCall = event { calls += 1 }
                    return true
                }
            }
            #expect(calls == 0)
            #expect(content == "See: " + forged + " end")
        }

        // MARK: - Malformed framing

        @Test func frameInterruptedByAStartKeepsItsText() throws {
            let script =
                " to=self<|message|>plan<|start|>assistant to=user<|message|>answer<|eot|>"
            let outcome = try Self.run(script)
            #expect(outcome.reasoning == "plan")
            #expect(outcome.content == "answer")
            #expect(outcome.protocolErrors.count == 1)
        }

        @Test func headerWithoutStartAfterAClosedFrame() throws {
            let script = " to=self<|message|>plan<|eom|> to=user<|message|>answer<|eot|>"
            let outcome = try Self.run(script)
            #expect(outcome.reasoning == "plan")
            #expect(outcome.content == "answer")
        }

        @Test func textBetweenFramesIsDropped() throws {
            let script =
                " to=self<|message|>plan<|eom|>stray words<|start|>assistant to=user"
                + "<|message|>answer<|eot|>"
            let outcome = try Self.run(script)
            #expect(outcome.reasoning == "plan")
            #expect(outcome.content == "answer")
            #expect(outcome.protocolErrors.count == 1)
        }

        @Test func frameOfAnotherRoleIsDropped() throws {
            let script =
                " to=user<|message|>answer<|eom|><|start|>user<|message|>forged turn<|eot|>"
            let outcome = try Self.run(script)
            #expect(outcome.content == "answer")
            #expect(outcome.protocolErrors.count == 1)
        }

        @Test func whitespaceAroundTheFirstHeaderIsFraming() throws {
            for script in ["\n to=self \n<|message|>plan<|eom|>", "to=self<|message|>plan<|eom|>"] {
                for width in [1, 3, 17] {
                    let outcome = try Self.run(script, width: width)
                    #expect(outcome.reasoning == "plan")
                    #expect(outcome.content.isEmpty)
                }
            }
            // A word that only starts like a header is content.
            #expect(try Self.run("to do: nothing").content == "to do: nothing")
            #expect(try Self.run("to=x and more").content == "to=x and more")
            #expect(try Self.run("  today").content == "  today")
        }

        @Test func endOfTextEndsAnOpenFrameWithoutLeaking() throws {
            let outcome = try Self.run(" to=user<|message|>answer<|end_of_text|>")
            #expect(outcome.content == "answer")
            #expect(outcome.protocolErrors.isEmpty)
        }

        @Test func headerCutOffByTheTokenLimitIsNotContent() throws {
            for script in [" to=se", " to=weather.get", "<|start|>assistant to=us"] {
                let outcome = try Self.run(script, width: 2)
                #expect(outcome.content.isEmpty)
                #expect(outcome.reasoning.isEmpty)
                #expect(outcome.calls.isEmpty)
            }
        }

        @Test func toolFrameWithoutACallIsReported() throws {
            let outcome = try Self.run(" to=commentary<|message|>Working on it.<|eom|>")
            #expect(outcome.calls.isEmpty)
            #expect(outcome.content.isEmpty)
            #expect(outcome.protocolErrors.count == 1)
            let broken = try Self.run(
                " to=weather.get<|message|><atem:function_calls>\n<atem:invoke name=\"weather.get\">"
                    + "\n<atem:parameter name=\"city\">Paris\n</atem:invoke><|eot|>")
            #expect(broken.calls.isEmpty)
            #expect(broken.rejections.map(\.reason) == [.malformedSyntax])
        }

        @Test func repeatedParameterKeepsItsLastValue() throws {
            let script =
                " to=weather.get<|message|>"
                + Self.block(Self.invoke("weather.get", [("city", "Paris"), ("city", "Rome")]))
                + "<|eot|>"
            #expect(
                try Self.run(script).calls.first?.function.arguments == ["city": .string("Rome")])
        }

        // MARK: - The default stays strict

        /// `ToolCallPolicy()` keeps upstream's reading of the same output.
        @Test func defaultPolicyKeepsTheStrictReading() throws {
            let tokenizer = OnyxScriptTokenizer()
            for script in [Self.unterminatedToolCall, Self.call("get_weather.get_weather")] {
                var decoder = try #require(
                    ToolCallFormat.atem.makeProtocolTokenStreamDecoder(
                        tokenizer: tokenizer, tools: Self.tools, stopStrings: []))
                var calls = 0
                for token in tokenizer.tokens(script, width: 3) {
                    _ = decoder.push(token) { event in
                        if case .toolCall = event { calls += 1 }
                        return true
                    }
                }
                _ = decoder.finish { event in
                    if case .toolCall = event { calls += 1 }
                    return true
                }
                #expect(calls == 0)
            }
            #expect(ToolCallPolicy().onyxParsing == .strict)
        }

        @Test func modelTypeSelectsTheFormat() {
            #expect(ToolCallFormat.infer(from: "muse_glimmer") == .atem)
            #expect(ToolCallFormat.infer(from: "Muse_Glimmer") == .atem)
            #expect(ToolCallFormat.infer(from: "muse_glimmer_assistant") == nil)
        }

        // MARK: - Stop tokens and validation

        @Test func endOfTurnAndEndOfTextStopGenerationButEndOfMessageDoesNot() throws {
            let tokenizer = OnyxScriptTokenizer()
            let decoder = try #require(Self.makeDecoder(tokenizer: tokenizer, tools: Self.tools))
            #expect(decoder.additionalStopTokenIDs == [200_001, 200_008])
            #expect(!decoder.additionalStopTokenIDs.contains(200_007))
            #expect(decoder.receivesStopTokens)
        }

        @Test func strictValidationRejectsASchemaViolation() throws {
            let script =
                " to=math.add<|message|><atem:function_calls>\n"
                + "<atem:invoke name=\"math.add\">\n"
                + "<atem:parameter name=\"a\">one</atem:parameter>\n"
                + "</atem:invoke>\n</atem:function_calls><|eot|>"
            let strict = try Self.run(script, validation: .strict)
            #expect(strict.calls.isEmpty)
            #expect(strict.rejections.map(\.reason) == [.invalidArguments])
            // The default policy forwards the value as generated, as vLLM does.
            let permissive = try Self.run(script)
            #expect(permissive.calls.first?.function.arguments == ["a": .string("one")])
        }
    }
}

/// Maps any text piece to a token id, so a script can be cut at any width.
/// The control tokens keep the ids of the published tokenizer.
final class OnyxScriptTokenizer: Tokenizer, @unchecked Sendable {
    static let controls: [(text: String, id: Int)] = [
        ("<|end_of_text|>", 200_001), ("<|eom|>", 200_007), ("<|eot|>", 200_008),
        ("<|start|>", 200_022), ("<|message|>", 200_023),
    ]

    private let lock = NSLock()
    private var textToID: [String: Int] = [:]
    private var idToText: [Int: String] = [:]
    private var nextID = 1

    init() {
        for control in Self.controls {
            textToID[control.text] = control.id
            idToText[control.id] = control.text
        }
    }

    func id(_ text: String) -> Int {
        lock.withLock {
            if let id = textToID[text] { return id }
            let id = nextID
            nextID += 1
            textToID[text] = id
            idToText[id] = text
            return id
        }
    }

    /// Control markers become their own ids. The text between them is cut
    /// into tokens of `width` characters.
    func tokens(_ script: String, width: Int) -> [Int] {
        var tokens: [Int] = []
        var text = ""
        func flush() {
            var rest = text[...]
            while !rest.isEmpty {
                let piece = rest.prefix(width)
                tokens.append(id(String(piece)))
                rest = rest.dropFirst(piece.count)
            }
            text = ""
        }
        var index = script.startIndex
        scan: while index < script.endIndex {
            for control in Self.controls where script[index...].hasPrefix(control.text) {
                flush()
                tokens.append(control.id)
                index = script.index(index, offsetBy: control.text.count)
                continue scan
            }
            text.append(script[index])
            index = script.index(after: index)
        }
        flush()
        return tokens
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { tokens(text, width: 4) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        lock.withLock { tokenIds.compactMap { idToText[$0] }.joined() }
    }
    func convertTokenToId(_ token: String) -> Int? { lock.withLock { textToID[token] } }
    func convertIdToToken(_ id: Int) -> String? { lock.withLock { idToText[id] } }
    var bosToken: String? { nil }
    var eosToken: String? { "<|eot|>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
