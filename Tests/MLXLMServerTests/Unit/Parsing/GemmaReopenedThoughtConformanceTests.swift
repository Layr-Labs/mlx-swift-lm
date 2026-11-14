import Foundation
import MLXLMServer
import Testing

extension UnitTests {
    @Suite
    struct GemmaReopenedThoughtConformanceTests {
        // A process-level switch keeps these tests independent of parallel suites.
        private let enabled =
            ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA_REOPENED_THOUGHTS"] == "1"

        @Test(
            "a reopened Gemma thought and stray close never become answer text",
            arguments: [1, 2, 7, 1000])
        func reopenedThought(width: Int) {
            let first = "<|channel>thought\nFirst.<channel|>Answer."
            let reopened = "<|channel>thought\nSecond.<channel|>Done.<channel|>"
            let result = parse(first + reopened, width: width)
            #expect(result.content == (enabled ? "Answer.Done." : "Answer." + reopened))
            #expect(result.reasoningContent == (enabled ? "First.Second." : "First."))
        }

        @Test("collected output routes every thought and suppresses a stray close")
        func collectedThoughts() {
            let reopened = "<|channel>thought\nSecond.<channel|>Done.<channel|>"
            let result = ReasoningParser(format: .gemma4).parse(
                "<|channel>thought\nFirst.<channel|>Answer." + reopened)
            #expect(result.content == (enabled ? "Answer.Done." : "Answer." + reopened))
            #expect(result.reasoningContent == (enabled ? "First.Second." : "First."))
        }

        @Test("a stray close before any thought is absorbed", arguments: [1, 4, 1000])
        func strayClose(width: Int) {
            let text = "Hello<channel|> world"
            let result = parse(text, width: width)
            #expect(result.content == (enabled ? "Hello world" : text))
            #expect(result.reasoningContent == nil)
        }

        @Test("collected output keeps a close with no opener as reasoning unless enabled")
        func collectedCloseWithoutOpener() {
            // Enabled, it is a stray marker; a caller that pre-opens the thought
            // in the prompt must leave the switch off.
            let result = ReasoningParser(format: .gemma4).parse("Hello<channel|> world")
            #expect(result.content == (enabled ? "Hello world" : "world"))
            #expect(result.reasoningContent == (enabled ? nil : "Hello"))
        }

        @Test(
            "channel markers in native tool strings remain argument data", arguments: [1, 3, 1000])
        func toolArgumentMarkers(width: Int) {
            guard enabled else { return }
            let frame =
                #"<|tool_call>call:save{text:<|"|><|channel>thought<channel|><turn|><tool_call|><|"|>}<tool_call|>"#
            let result = parse("<|channel>thought\nFirst.<channel|>" + frame, width: width)
            #expect(result.content == frame)
            #expect(result.reasoningContent == "First.")
        }

        @Test("reopened unclosed and empty thoughts stay out of the answer")
        func unfinishedThoughts() {
            guard enabled else { return }
            for width in [1, 2, 1000] {
                let unfinished = parse(
                    "<|channel>thought\nOne<channel|>A<|channel>thought\nTwo", width: width)
                #expect(unfinished.content == "A")
                #expect(unfinished.reasoningContent == "OneTwo")
                let empty = parse(
                    "<|channel>thought\nOne<channel|>A<|channel>thought<channel|>B", width: width)
                #expect(empty.content == "AB")
                #expect(empty.reasoningContent == "One")
            }
        }

        @Test(
            "bare thought prefixes are stripped across chunk boundaries", arguments: [1, 2, 1000])
        func barePrefix(width: Int) {
            guard enabled else { return }
            let text = "thought\nHello"
            #expect(parse(text, width: width).content == "Hello")
            #expect(ReasoningParser(format: .gemma4).parse(text).content == "Hello")
        }

        @Test("tool strings stay opaque before reopening a channel", arguments: [1, 2, 1000])
        func quoteForms(width: Int) {
            guard enabled else { return }
            for value in [
                #""<|channel>thought<channel|><tool_call|>""#,
                #"<escape><|channel>thought<channel|><tool_call|><escape>"#,
            ] {
                let frame = "<|tool_call>call:save{text:" + value + "}<tool_call|>"
                let text = frame + "<|channel>thought\nNext<channel|>Answer"
                let streamed = parse(text, width: width)
                let collected = ReasoningParser(format: .gemma4).parse(text)
                #expect(streamed.content == frame + "Answer")
                #expect(streamed.reasoningContent == "Next")
                #expect(collected.content == streamed.content)
                #expect(collected.reasoningContent == streamed.reasoningContent)
            }
        }

        @Test("truncated channel prefixes remain literal at EOS", arguments: [1, 2, 1000])
        func partialMarkers(width: Int) {
            guard enabled else { return }
            for suffix in ["<|chan", "<chann", "<|tool_ca"] {
                let text = "Answer" + suffix
                #expect(parse(text, width: width).content == text)
                #expect(ReasoningParser(format: .gemma4).parse(text).content == text)
            }
        }

        private func parse(_ text: String, width: Int) -> ParsedReasoning {
            var parser = StreamingReasoningParser(format: .gemma4)
            let characters = Array(text)
            var pieces: [ParsedReasoning] = []
            for start in stride(from: 0, to: characters.count, by: width) {
                pieces += parser.parse(
                    String(characters[start ..< min(start + width, characters.count)]))
            }
            pieces += parser.finish()
            let reasoning = pieces.compactMap(\.reasoningContent).joined()
            return .init(
                content: pieces.map(\.content).joined(),
                reasoningContent: reasoning.isEmpty ? nil : reasoning)
        }
    }
}
