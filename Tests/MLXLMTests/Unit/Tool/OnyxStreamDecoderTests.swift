// Copyright © 2026 Eigen Labs.

import Foundation
import MLXLMCommon
import Testing

extension UnitTests {

    /// The public decoders a host uses: token ids, or text with the control
    /// markers visible. Both give what the conformance suite pins.
    @Suite
    struct OnyxStreamDecoderTests {
        typealias Fixtures = OnyxATEMConformanceTests

        struct Summary: Equatable {
            var reasoning = ""
            var content = ""
            var calls: [ToolCall.Function] = []
            var rejections: [RejectedToolCall.Reason] = []

            mutating func add(_ events: [OnyxStreamEvent]) {
                for event in events {
                    switch event {
                    case .reasoning(let text): reasoning += text
                    case .content(let text): content += text
                    case .toolCall(let call): calls.append(call.function)
                    case .rejectedToolCall(let rejection): rejections.append(rejection.reason)
                    case .protocolError: break
                    }
                }
            }
        }

        static let scripts: [String] = [
            Fixtures.weatherCall, Fixtures.twoToolMessages, Fixtures.echoedInvoke,
            Fixtures.unterminatedToolCall, Fixtures.reasoningThenAnswer,
            Fixtures.truncatedReasoning, Fixtures.call("get_weather.get_weather"),
            Fixtures.call("foo.get_weather"), "<|message|>Hello.<|eot|>",
            "Just a direct answer.",
            " to=self<|message|>I will list the files. to=fs.list<|message|>"
                + Fixtures.block(Fixtures.invoke("fs.list")) + "<|eot|>",
            " to=math.add<|message|>"
                + Fixtures.block(
                    Fixtures.invoke("math.add", [("a", "1"), ("b", "2")]),
                    Fixtures.invoke("math.mul", [("a", "3")])) + "<|eom|>"
                + "<|start|>assistant to=user<|message|>Both asked. <|eot|>",
            " to=weather.get<|message|><atem:function_calls>\n<atem:invoke name=\"weather.get\">"
                + "\n<atem:parameter name=\"city\">Par",
            " to=self<|message|>a < b, and <|e is not a marker<|eom|>"
                + "<|start|>assistant to=user<|message|>1 <| 2 <<|eot|>",
        ]

        static func text(_ script: String, width: Int) -> Summary {
            var decoder = OnyxTextStreamDecoder(tools: Fixtures.tools)
            var summary = Summary()
            var rest = script[...]
            while !rest.isEmpty {
                let chunk = rest.prefix(width)
                summary.add(decoder.process(String(chunk)))
                rest = rest.dropFirst(chunk.count)
            }
            summary.add(decoder.finish())
            return summary
        }

        static func tokens(_ script: String, width: Int) throws -> Summary {
            let tokenizer = OnyxScriptTokenizer()
            var decoder = try #require(
                OnyxTokenStreamDecoder(tokenizer: tokenizer, tools: Fixtures.tools))
            var summary = Summary()
            for token in tokenizer.tokens(script, width: width) {
                summary.add(decoder.push(token))
            }
            summary.add(decoder.finish())
            return summary
        }

        @Test func textGivesTheSameResultAtEveryChunkWidth() {
            for script in Self.scripts {
                let whole = Self.text(script, width: script.count)
                for width in [1, 2, 3, 5, 7, 11, 17] {
                    #expect(Self.text(script, width: width) == whole, "width \(width): \(script)")
                }
            }
        }

        @Test func textGivesTheSameResultForEverySplitPoint() {
            for script in Self.scripts {
                let whole = Self.text(script, width: script.count)
                for split in script.indices {
                    var decoder = OnyxTextStreamDecoder(tools: Fixtures.tools)
                    var summary = Summary()
                    summary.add(decoder.process(String(script[..<split])))
                    summary.add(decoder.process(String(script[split...])))
                    summary.add(decoder.finish())
                    #expect(summary == whole, "\(script)")
                }
            }
        }

        @Test func textAndTokenDecodersAgree() throws {
            for script in Self.scripts {
                #expect(
                    try Self.tokens(script, width: 3) == Self.text(script, width: 3), "\(script)")
            }
        }

        @Test(arguments: [1, 3, 17])
        func textToolCallStream(width: Int) {
            let summary = Self.text(Fixtures.unterminatedToolCall, width: width)
            for marker in Fixtures.framing {
                #expect(!summary.reasoning.contains(marker))
                #expect(!summary.content.contains(marker))
            }
            #expect(summary.reasoning == "I should read the hostname file to answer.")
            #expect(summary.content.isEmpty)
            #expect(summary.calls.map(\.name) == ["read.read"])
            #expect(summary.calls.first?.arguments == ["path": .string("/etc/hostname")])
        }

        @Test func textKeepsMarkerLookalikes() {
            let summary = Self.text(Self.scripts[Self.scripts.count - 1], width: 1)
            #expect(summary.reasoning == "a < b, and <|e is not a marker")
            #expect(summary.content == "1 <| 2 <")
        }

        @Test func tokenDecoderReportsStopTokensAndReasoningState() throws {
            let tokenizer = OnyxScriptTokenizer()
            var decoder = try #require(
                OnyxTokenStreamDecoder(tokenizer: tokenizer, tools: Fixtures.tools))
            #expect(decoder.stopTokenIDs == [200_001, 200_008])
            var reasoningTokens = 0
            var events: [OnyxStreamEvent] = []
            for token in tokenizer.tokens(Fixtures.call("foo.get_weather"), width: 4) {
                events += decoder.push(token)
            }
            #expect(decoder.rejectedToolCallCount == 1)
            decoder = try #require(
                OnyxTokenStreamDecoder(tokenizer: tokenizer, tools: Fixtures.tools))
            for token in tokenizer.tokens(Fixtures.reasoningThenAnswer, width: 5) {
                if decoder.isInsideReasoning { reasoningTokens += 1 }
                events += decoder.push(token)
            }
            // "Think about it." in tokens of five characters, and its `<|eom|>`.
            #expect(reasoningTokens == 4)
            #expect(!decoder.isInsideReasoning)
        }

        @Test func tokenDecoderNeedsTheControlTokens() {
            #expect(OnyxTokenStreamDecoder(tokenizer: PlainTokenizer(), tools: nil) == nil)
        }

        @Test func strictValidationIsAvailableToHosts() {
            var decoder = OnyxTextStreamDecoder(tools: Fixtures.tools, validation: .strict)
            var summary = Summary()
            summary.add(
                decoder.process(
                    " to=math.add<|message|>"
                        + Fixtures.block(Fixtures.invoke("math.add", [("a", "one")])) + "<|eot|>"))
            summary.add(decoder.finish())
            #expect(summary.calls.isEmpty)
            #expect(summary.rejections == [.invalidArguments])
        }
    }
}

private struct PlainTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
