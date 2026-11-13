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
            let characters = Array(first + reopened)
            var parser = StreamingReasoningParser(format: .gemma4)
            var pieces: [ParsedReasoning] = []
            for start in stride(from: 0, to: characters.count, by: width) {
                pieces += parser.parse(
                    String(characters[start ..< min(start + width, characters.count)]))
            }
            pieces += parser.finish()
            #expect(
                pieces.map(\.content).joined() == (enabled ? "Answer.Done." : "Answer." + reopened))
            #expect(
                pieces.compactMap(\.reasoningContent).joined()
                    == (enabled ? "First.Second." : "First."))
        }
    }
}
