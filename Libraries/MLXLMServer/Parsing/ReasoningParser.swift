// Copyright © 2026 Eigen Labs Inc.

import Foundation

public enum ReasoningParserFormat: String, Codable, Sendable, CaseIterable {
    case none
    case deepseekR1 = "deepseek_r1"
    case qwen3
    case harmony

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")

        switch raw {
        case "none", "off", "disabled":
            self = .none
        case "deepseek_r1", "deepseek", "r1", "think":
            self = .deepseekR1
        case "qwen3", "qwen":
            self = .qwen3
        case "harmony", "gpt_oss", "openai_harmony":
            self = .harmony
        default:
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported reasoning parser '\(raw)'"
            )
        }
    }
}

public struct ParsedReasoning: Sendable, Equatable {
    public var content: String
    public var reasoningContent: String?
}

public struct ReasoningParser: Sendable {
    public var format: ReasoningParserFormat

    public init(format: ReasoningParserFormat) {
        self.format = format
    }

    public func parse(_ text: String) -> ParsedReasoning {
        switch format {
        case .none:
            return .init(content: text, reasoningContent: nil)
        case .deepseekR1, .qwen3:
            return parseThinkTags(text)
        case .harmony:
            return parseHarmony(text)
        }
    }

    private func parseThinkTags(_ text: String) -> ParsedReasoning {
        var remaining = text
        var reasoning: [String] = []

        while let start = remaining.range(of: "<think>"),
            let end = remaining.range(of: "</think>", range: start.upperBound..<remaining.endIndex)
        {
            reasoning.append(String(remaining[start.upperBound..<end.lowerBound]))
            remaining.removeSubrange(start.lowerBound..<end.upperBound)
        }

        if reasoning.isEmpty, let end = remaining.range(of: "</think>") {
            reasoning.append(String(remaining[..<end.lowerBound]))
            remaining.removeSubrange(remaining.startIndex..<end.upperBound)
        }

        let reasoningText = reasoning.joined(separator: "\n").trimmedForReasoning
        return .init(
            content: remaining.trimmedForReasoning,
            reasoningContent: reasoningText.isEmpty ? nil : reasoningText
        )
    }

    private func parseHarmony(_ text: String) -> ParsedReasoning {
        let channelMarker = "<|channel|>"
        let messageMarker = "<|message|>"
        let endMarkers = ["<|end|>", "<|return|>", "<|call|>"]
        var final: [String] = []
        var analysis: [String] = []
        var cursor = text.startIndex

        while let channelStart = text.range(of: channelMarker, range: cursor..<text.endIndex) {
            let channelNameStart = channelStart.upperBound
            guard let messageStart = text.range(
                of: messageMarker,
                range: channelNameStart..<text.endIndex
            ) else { break }

            let channel = String(text[channelNameStart..<messageStart.lowerBound])
            let contentStart = messageStart.upperBound
            let contentEnd = firstRange(ofAny: endMarkers, in: text, range: contentStart..<text.endIndex)
            let messageEnd = contentEnd?.lowerBound ?? text.endIndex
            let content = String(text[contentStart..<messageEnd]).removingHarmonyRoleMarkers

            switch channel {
            case "analysis":
                analysis.append(content)
            case "final":
                final.append(content)
            default:
                break
            }

            cursor = contentEnd?.upperBound ?? text.endIndex
        }

        if final.isEmpty && analysis.isEmpty {
            return .init(content: text, reasoningContent: nil)
        }

        let reasoningText = analysis.joined(separator: "\n").trimmedForReasoning
        return .init(
            content: final.joined(separator: "\n").trimmedForReasoning,
            reasoningContent: reasoningText.isEmpty ? nil : reasoningText
        )
    }

    private func firstRange(
        ofAny markers: [String],
        in text: String,
        range: Range<String.Index>
    ) -> Range<String.Index>? {
        var first: Range<String.Index>?
        for marker in markers {
            guard let candidate = text.range(of: marker, range: range) else { continue }
            if first == nil || candidate.lowerBound < first!.lowerBound {
                first = candidate
            }
        }
        return first
    }
}

public struct StreamingReasoningParser: Sendable {
    public var format: ReasoningParserFormat

    private enum ThinkState: Sendable {
        case undecided
        case reasoning
        case content
    }

    private enum HarmonyState: Sendable {
        case outside
        case channel
        case message
    }

    private var thinkState: ThinkState = .undecided
    private var thinkBuffer = ""
    private var harmonyBuffer = ""
    private var harmonyChannel = ""
    private var harmonyState: HarmonyState = .outside

    public init(format: ReasoningParserFormat) {
        self.format = format
    }

    public mutating func parse(_ chunk: String) -> [ParsedReasoning] {
        guard !chunk.isEmpty else { return [] }
        switch format {
        case .none:
            return [.init(content: chunk, reasoningContent: nil)]
        case .deepseekR1, .qwen3:
            thinkBuffer += chunk
            return drainThinkBuffer(final: false)
        case .harmony:
            harmonyBuffer += chunk
            return drainHarmonyBuffer(final: false)
        }
    }

    public mutating func finish() -> [ParsedReasoning] {
        switch format {
        case .none:
            return []
        case .deepseekR1, .qwen3:
            return drainThinkBuffer(final: true)
        case .harmony:
            return drainHarmonyBuffer(final: true)
        }
    }

    private mutating func drainThinkBuffer(final: Bool) -> [ParsedReasoning] {
        let opening = "<think>"
        let closing = "</think>"
        var output: [ParsedReasoning] = []
        var shouldContinue = true

        while shouldContinue {
            switch thinkState {
            case .undecided:
                if let open = thinkBuffer.range(of: opening) {
                    appendContent(String(thinkBuffer[..<open.lowerBound]), to: &output)
                    thinkBuffer.removeSubrange(thinkBuffer.startIndex..<open.upperBound)
                    thinkState = .reasoning
                } else if let close = thinkBuffer.range(of: closing) {
                    appendReasoning(String(thinkBuffer[..<close.lowerBound]), to: &output)
                    thinkBuffer.removeSubrange(thinkBuffer.startIndex..<close.upperBound)
                    thinkState = .content
                } else if final {
                    appendContent(thinkBuffer, to: &output)
                    thinkBuffer.removeAll(keepingCapacity: true)
                    thinkState = .content
                } else {
                    shouldContinue = false
                }
            case .reasoning:
                if let close = thinkBuffer.range(of: closing) {
                    appendReasoning(String(thinkBuffer[..<close.lowerBound]), to: &output)
                    thinkBuffer.removeSubrange(thinkBuffer.startIndex..<close.upperBound)
                    thinkState = .content
                } else {
                    let reasoning = consumeSafePrefix(from: &thinkBuffer, preservingPotentialPrefixOf: closing, final: final)
                    appendReasoning(reasoning, to: &output)
                    shouldContinue = false
                }
            case .content:
                if let open = thinkBuffer.range(of: opening) {
                    appendContent(String(thinkBuffer[..<open.lowerBound]), to: &output)
                    thinkBuffer.removeSubrange(thinkBuffer.startIndex..<open.upperBound)
                    thinkState = .reasoning
                } else {
                    let content = consumeSafePrefix(from: &thinkBuffer, preservingPotentialPrefixOf: opening, final: final)
                    appendContent(content, to: &output)
                    shouldContinue = false
                }
            }
        }

        return output
    }

    private mutating func drainHarmonyBuffer(final: Bool) -> [ParsedReasoning] {
        let channelMarker = "<|channel|>"
        let messageMarker = "<|message|>"
        let roleMarker = "<|start|>assistant"
        let endMarkers = ["<|end|>", "<|return|>", "<|call|>"]
        let outsideControlMarkers = [channelMarker, roleMarker]
        let messageControlMarkers = endMarkers + [roleMarker]
        var output: [ParsedReasoning] = []
        var shouldContinue = true

        while shouldContinue {
            switch harmonyState {
            case .outside:
                if let marker = firstRange(
                    ofAny: outsideControlMarkers,
                    in: harmonyBuffer,
                    range: harmonyBuffer.startIndex..<harmonyBuffer.endIndex
                ) {
                    appendContent(String(harmonyBuffer[..<marker.lowerBound]), to: &output)
                    let matchedMarker = String(harmonyBuffer[marker])
                    harmonyBuffer.removeSubrange(harmonyBuffer.startIndex..<marker.upperBound)
                    if matchedMarker == channelMarker {
                        harmonyChannel.removeAll(keepingCapacity: true)
                        harmonyState = .channel
                    }
                } else {
                    let content = consumeSafePrefix(
                        from: &harmonyBuffer,
                        preservingPotentialPrefixOfAny: outsideControlMarkers,
                        final: final
                    )
                    appendContent(content, to: &output)
                    shouldContinue = false
                }
            case .channel:
                if let messageStart = harmonyBuffer.range(of: messageMarker) {
                    harmonyChannel += String(harmonyBuffer[..<messageStart.lowerBound])
                    harmonyBuffer.removeSubrange(harmonyBuffer.startIndex..<messageStart.upperBound)
                    harmonyState = .message
                } else {
                    let channel = consumeSafePrefix(
                        from: &harmonyBuffer,
                        preservingPotentialPrefixOf: messageMarker,
                        final: final
                    )
                    harmonyChannel += channel
                    if final {
                        harmonyChannel.removeAll(keepingCapacity: true)
                        harmonyState = .outside
                    }
                    shouldContinue = false
                }
            case .message:
                if let marker = firstRange(
                    ofAny: messageControlMarkers,
                    in: harmonyBuffer,
                    range: harmonyBuffer.startIndex..<harmonyBuffer.endIndex
                ) {
                    appendHarmonyMessage(
                        String(harmonyBuffer[..<marker.lowerBound]),
                        channel: harmonyChannel,
                        to: &output
                    )
                    let matchedMarker = String(harmonyBuffer[marker])
                    harmonyBuffer.removeSubrange(harmonyBuffer.startIndex..<marker.upperBound)
                    if matchedMarker != roleMarker {
                        harmonyChannel.removeAll(keepingCapacity: true)
                        harmonyState = .outside
                    }
                } else {
                    let content = consumeSafePrefix(
                        from: &harmonyBuffer,
                        preservingPotentialPrefixOfAny: messageControlMarkers,
                        final: final
                    )
                    appendHarmonyMessage(content, channel: harmonyChannel, to: &output)
                    if final {
                        harmonyChannel.removeAll(keepingCapacity: true)
                        harmonyState = .outside
                    }
                    shouldContinue = false
                }
            }
        }

        return output
    }

    private func firstRange(
        ofAny markers: [String],
        in text: String,
        range: Range<String.Index>
    ) -> Range<String.Index>? {
        var first: Range<String.Index>?
        for marker in markers {
            guard let candidate = text.range(of: marker, range: range) else { continue }
            if first == nil || candidate.lowerBound < first!.lowerBound {
                first = candidate
            }
        }
        return first
    }

    private func consumeSafePrefix(
        from buffer: inout String,
        preservingPotentialPrefixOf marker: String,
        final: Bool
    ) -> String {
        if final {
            let output = buffer
            buffer.removeAll(keepingCapacity: true)
            return output
        }

        let suffixLength = potentialMarkerPrefixSuffixLength(in: buffer, marker: marker)
        guard suffixLength > 0 else {
            let output = buffer
            buffer.removeAll(keepingCapacity: true)
            return output
        }

        let suffixStart = buffer.index(buffer.endIndex, offsetBy: -suffixLength)
        let output = String(buffer[..<suffixStart])
        buffer = String(buffer[suffixStart...])
        return output
    }

    private func consumeSafePrefix(
        from buffer: inout String,
        preservingPotentialPrefixOfAny markers: [String],
        final: Bool
    ) -> String {
        if final {
            let output = buffer
            buffer.removeAll(keepingCapacity: true)
            return output
        }

        let suffixLength = markers
            .map { potentialMarkerPrefixSuffixLength(in: buffer, marker: $0) }
            .max() ?? 0
        guard suffixLength > 0 else {
            let output = buffer
            buffer.removeAll(keepingCapacity: true)
            return output
        }

        let suffixStart = buffer.index(buffer.endIndex, offsetBy: -suffixLength)
        let output = String(buffer[..<suffixStart])
        buffer = String(buffer[suffixStart...])
        return output
    }

    private func potentialMarkerPrefixSuffixLength(in text: String, marker: String) -> Int {
        let maxLength = min(text.count, max(marker.count - 1, 0))
        guard maxLength > 0 else { return 0 }

        for length in stride(from: maxLength, through: 1, by: -1) {
            let suffix = text.suffix(length)
            if marker.hasPrefix(suffix) {
                return length
            }
        }
        return 0
    }

    private func appendContent(_ content: String, to output: inout [ParsedReasoning]) {
        guard !content.isEmpty else { return }
        output.append(.init(content: content, reasoningContent: nil))
    }

    private func appendReasoning(_ reasoning: String, to output: inout [ParsedReasoning]) {
        guard !reasoning.isEmpty else { return }
        output.append(.init(content: "", reasoningContent: reasoning))
    }

    private func appendHarmonyMessage(_ text: String, channel: String, to output: inout [ParsedReasoning]) {
        switch channel.trimmedForReasoning {
        case "analysis":
            appendReasoning(text.removingHarmonyRoleMarkers, to: &output)
        case "final":
            appendContent(text.removingHarmonyRoleMarkers, to: &output)
        default:
            break
        }
    }
}

extension String {
    fileprivate var trimmedForReasoning: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }

    fileprivate var removingHarmonyRoleMarkers: String {
        replacingOccurrences(of: "<|start|>assistant", with: "")
    }
}
