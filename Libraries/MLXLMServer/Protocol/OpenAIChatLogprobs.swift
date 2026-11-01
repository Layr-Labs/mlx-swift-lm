// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// Captured sampled-token metadata. It travels in generation order, never in a
/// shared side channel. Token text is response content, not telemetry.
public struct OpenAITokenLogprob: Codable, Sendable, Equatable {
    /// One `top_logprobs` alternative (same wire shape minus the nesting).
    public struct Top: Codable, Sendable, Equatable {
        public let token: String
        public let logprob: Float
        /// Raw UTF-8 bytes of `token` — the OpenAI escape hatch for BPE
        /// intermediate tokens whose text is not valid standalone UTF-8.
        public let bytes: [Int]?

        public init(token: String, logprob: Float, bytes: [Int]?) {
            self.token = token
            self.logprob = logprob
            self.bytes = bytes ?? []
        }
    }

    public let token: String
    public let logprob: Float
    public let bytes: [Int]?
    public let topLogprobs: [Top]

    enum CodingKeys: String, CodingKey {
        case token
        case logprob
        case bytes
        case topLogprobs = "top_logprobs"
    }

    public init(token: String, logprob: Float, bytes: [Int]?, topLogprobs: [Top]) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes ?? []
        self.topLogprobs = topLogprobs
    }

}

public struct OpenAIChatLogprobs: Codable, Sendable, Equatable {
    public var content: [OpenAITokenLogprob]

    public init(content: [OpenAITokenLogprob]) {
        self.content = content
    }
}

/// A bounded request-local holdback. Parsers can delay visible content; the
/// freshest 4096 entries are retained, matching the provider's former bound.
/// It never drains metadata belonging to a future, unconsumed stream event.
struct ChatLogprobsBuffer {
    static let maximumEntries = 4096
    private(set) var entries: [OpenAITokenLogprob] = []

    mutating func append(_ incoming: [OpenAITokenLogprob]) {
        if incoming.count >= Self.maximumEntries {
            entries = Array(incoming.suffix(Self.maximumEntries))
        } else {
            let overflow = entries.count + incoming.count - Self.maximumEntries
            if overflow > 0 { entries.removeFirst(overflow) }
            entries.append(contentsOf: incoming)
        }
    }

    mutating func take() -> OpenAIChatLogprobs? {
        guard !entries.isEmpty else { return nil }
        defer { entries.removeAll(keepingCapacity: true) }
        return OpenAIChatLogprobs(content: entries)
    }
}
