// Copyright © 2026 Eigen Labs.

import MLXLMCommon

/// Splits Muse Glimmer text, with its control markers visible, into
/// reasoning and content. Tool frames are left out: their calls come from
/// the engine's tool parser, and their markup is never content.
struct StreamingOnyxReasoningParser: Sendable {
    private var decoder = OnyxTextStreamDecoder(tools: nil)

    mutating func parse(_ chunk: String) -> [ParsedReasoning] {
        pieces(decoder.process(chunk))
    }

    mutating func finish() -> [ParsedReasoning] {
        pieces(decoder.finish())
    }

    private func pieces(_ events: [OnyxStreamEvent]) -> [ParsedReasoning] {
        var output: [ParsedReasoning] = []
        for event in events {
            switch event {
            case .reasoning(let text): appendReasoning(text, to: &output)
            case .content(let text): appendContent(text, to: &output)
            case .toolCall, .rejectedToolCall, .protocolError: break
            }
        }
        return output
    }
}
