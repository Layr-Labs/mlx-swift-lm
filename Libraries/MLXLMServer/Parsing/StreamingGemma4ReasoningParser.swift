// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

struct StreamingGemma4ReasoningParser: Sendable {
    static let reopenedThoughtsEnabled =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA_REOPENED_THOUGHTS"] == "1"

    private enum State: Sendable {
        case undecided
        case thinking
        case content
        case toolCall
    }

    private let openTag = "<|channel>"
    private let closeTag = "<channel|>"
    private let thoughtPrefix = "thought\n"
    private let trailingMarkers = ["<turn|>", "<eos>"]

    private var state: State = .undecided
    private var buffer = ""
    private var strippedThoughtPrefix = false
    private var toolFrame = GemmaToolFrameScanner()

    mutating func parse(_ chunk: String) -> [ParsedReasoning] {
        buffer += chunk
        return drain(final: false)
    }

    mutating func finish() -> [ParsedReasoning] {
        drain(final: true)
    }

    private mutating func drain(final: Bool) -> [ParsedReasoning] {
        var output: [ParsedReasoning] = []
        var shouldContinue = true

        while shouldContinue {
            switch state {
            case .undecided:
                drainUndecided(final: final, output: &output, shouldContinue: &shouldContinue)
            case .thinking:
                drainThinking(final: final, output: &output, shouldContinue: &shouldContinue)
            case .content:
                drainContent(final: final, output: &output, shouldContinue: &shouldContinue)
            case .toolCall:
                drainToolCall(output: &output, shouldContinue: &shouldContinue)
            }
        }

        return output
    }

    private mutating func drainUndecided(
        final: Bool,
        output: inout [ParsedReasoning],
        shouldContinue: inout Bool
    ) {
        if Self.reopenedThoughtsEnabled {
            // Decide the optional bare prefix once, before emitting any content.
            if !final && thoughtPrefix.hasPrefix(buffer) && buffer != thoughtPrefix {
                shouldContinue = false
                return
            }
            if buffer.hasPrefix(thoughtPrefix) {
                buffer = String(buffer.dropFirst(thoughtPrefix.count))
            }
            state = .content
            drainChannelContent(final: final, output: &output, shouldContinue: &shouldContinue)
            return
        }
        if let openRange = buffer.range(of: openTag) {
            appendContent(String(buffer[..<openRange.lowerBound]), to: &output)
            buffer.removeSubrange(buffer.startIndex ..< openRange.upperBound)
            state = .thinking
            strippedThoughtPrefix = false
            return
        }

        if final {
            // No channel marker found -- strip bare thought\n prefix and emit as content
            if buffer.hasPrefix(thoughtPrefix) {
                buffer = String(buffer.dropFirst(thoughtPrefix.count))
            }
            appendContent(buffer, to: &output)
            buffer.removeAll(keepingCapacity: true)
            shouldContinue = false
            return
        }

        // Buffer might contain a partial open tag -- hold it back
        let safe = consumeSafePrefix(
            from: &buffer,
            preservingPotentialPrefixOf: openTag,
            final: false
        )
        appendContent(safe, to: &output)
        shouldContinue = false
    }

    private mutating func drainThinking(
        final: Bool,
        output: inout [ParsedReasoning],
        shouldContinue: inout Bool
    ) {
        // Strip "thought\n" prefix once after entering thinking state
        if !strippedThoughtPrefix {
            let emptyThought = "thought" + closeTag
            if Self.reopenedThoughtsEnabled && buffer.hasPrefix(emptyThought) {
                buffer = String(buffer.dropFirst("thought".count))
                strippedThoughtPrefix = true
            } else if Self.reopenedThoughtsEnabled && !final && emptyThought.hasPrefix(buffer) {
                shouldContinue = false
                return
            } else if buffer.hasPrefix(thoughtPrefix) {
                buffer = String(buffer.dropFirst(thoughtPrefix.count))
                strippedThoughtPrefix = true
            } else if !final && buffer.count < thoughtPrefix.count {
                // Could be a partial "thought\n" -- wait for more data
                shouldContinue = false
                return
            } else {
                strippedThoughtPrefix = true
            }
        }

        if let closeRange = buffer.range(of: closeTag) {
            appendReasoning(String(buffer[..<closeRange.lowerBound]), to: &output)
            buffer.removeSubrange(buffer.startIndex ..< closeRange.upperBound)
            state = .content
            return
        }

        let reasoning = consumeSafePrefix(
            from: &buffer,
            preservingPotentialPrefixOf: closeTag,
            final: final
        )
        appendReasoning(reasoning, to: &output)
        shouldContinue = false
    }

    private mutating func drainContent(
        final: Bool,
        output: inout [ParsedReasoning],
        shouldContinue: inout Bool
    ) {
        if Self.reopenedThoughtsEnabled {
            drainChannelContent(final: final, output: &output, shouldContinue: &shouldContinue)
            return
        }
        if final {
            // Clean trailing markers
            for marker in trailingMarkers {
                buffer = buffer.replacingOccurrences(of: marker, with: "")
            }
            appendContent(buffer, to: &output)
            buffer.removeAll(keepingCapacity: true)
            shouldContinue = false
            return
        }

        // While streaming, hold back potential trailing markers
        let content = consumeSafePrefix(
            from: &buffer,
            preservingPotentialPrefixOfAny: trailingMarkers,
            final: false
        )
        appendContent(content, to: &output)
        shouldContinue = false
    }

    private mutating func drainChannelContent(
        final: Bool,
        output: inout [ParsedReasoning],
        shouldContinue: inout Bool
    ) {
        let toolStart = "<|tool_call>"
        let markers = [openTag, closeTag, toolStart] + trailingMarkers
        if let range = firstRange(
            ofAny: markers, in: buffer, range: buffer.startIndex ..< buffer.endIndex)
        {
            let opensThought = buffer[range] == openTag
            let opensTool = buffer[range] == toolStart
            appendContent(String(buffer[..<range.lowerBound]), to: &output)
            buffer.removeSubrange(buffer.startIndex ..< range.upperBound)
            if opensThought {
                state = .thinking
                strippedThoughtPrefix = false
            } else if opensTool {
                appendContent(toolStart, to: &output)
                toolFrame = GemmaToolFrameScanner()
                state = .toolCall
            }
            return
        }
        appendContent(
            consumeSafePrefix(from: &buffer, preservingPotentialPrefixOfAny: markers, final: final),
            to: &output)
        shouldContinue = false
    }

    /// Tool payloads are opaque to channel routing, including marker-looking
    /// strings and quoted tool-end tags.
    private mutating func drainToolCall(
        output: inout [ParsedReasoning], shouldContinue: inout Bool
    ) {
        for index in buffer.unicodeScalars.indices {
            if toolFrame.consume(buffer.unicodeScalars[index]) {
                let end = buffer.unicodeScalars.index(after: index)
                appendContent(String(buffer[..<end]), to: &output)
                buffer.removeSubrange(buffer.startIndex ..< end)
                state = .content
                return
            }
        }
        appendContent(buffer, to: &output)
        buffer.removeAll(keepingCapacity: true)
        shouldContinue = false
    }
}
