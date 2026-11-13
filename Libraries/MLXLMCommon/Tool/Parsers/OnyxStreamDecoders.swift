// Copyright © 2026 Eigen Labs.

/// How Muse Glimmer's Onyx frames and ATEM tool calls are read.
public enum OnyxParsingMode: String, Hashable, Sendable, CaseIterable {
    /// Upstream's reading. A tool frame holds exactly one invoke, its name
    /// equals the frame's recipient, a value that does not fit the schema
    /// rejects the call, and a frame that generation leaves open gives
    /// nothing.
    case strict

    /// The reading of the reference servers (vLLM, SGLang, llama.cpp): one
    /// call for each complete invoke, a doubled tool name `X.X` read as the
    /// declared tool `X`, values typed by the schema without rejecting the
    /// call, and no call from an invoke that was cut off. A tool that was
    /// not declared is still rejected.
    case referenceParity
}

/// One decoded part of a Muse Glimmer response.
public enum OnyxStreamEvent: Sendable, Equatable {
    /// Text of a `to=self` frame.
    case reasoning(String)
    /// Text of a `to=user` frame, or of output without a frame.
    case content(String)
    case toolCall(ToolCall)
    /// Call-shaped output that was not made a call: an undeclared tool, a
    /// call cut off by the token limit, or arguments a strict policy refused.
    case rejectedToolCall(RejectedToolCall)
    /// Malformed framing was skipped. The message holds no model output.
    case protocolError(String)

    init?(_ event: TokenStreamEvent) {
        switch event {
        case .reasoning(let text): self = .reasoning(text)
        case .response(let text): self = .content(text)
        case .toolCall(let call): self = .toolCall(call)
        case .rejectedToolCall(let rejection): self = .rejectedToolCall(rejection)
        case .protocolError(let message): self = .protocolError(message)
        case .stop: return nil
        }
    }
}

/// Decodes the generated token ids of Muse Glimmer into reasoning, content
/// and tool calls, with ``OnyxParsingMode/referenceParity`` rules.
///
/// Use this when the engine reports token ids. Frames are found by the ids
/// of the control tokens, so text that spells a marker cannot forge a frame.
/// Feed every generated id, the stop token included, then call ``finish()``.
/// One value decodes one response and is used from one task.
public struct OnyxTokenStreamDecoder {
    private var adapter: OnyxReferenceStreamAdapter

    /// The ids that end generation: `<|eot|>` and `<|end_of_text|>`.
    /// `<|eom|>` is not one of them; more frames follow it.
    public var stopTokenIDs: Set<Int> { adapter.additionalStopTokenIDs }

    /// Whether the next token belongs to a reasoning frame. Read it before
    /// ``push(_:)`` to count reasoning tokens.
    public var isInsideReasoning: Bool { adapter.isInsideReasoning }

    public var rejectedToolCallCount: Int { adapter.rejectedToolCallCount }

    /// - Returns: `nil` when the tokenizer does not hold `<|start|>`,
    ///   `<|message|>`, `<|eom|>` and `<|eot|>` as four different tokens.
    public init?(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        validation: ToolCallValidationPolicy = .permissive
    ) {
        guard
            let adapter = OnyxReferenceStreamAdapter(
                tokenizer: tokenizer, tools: tools, stopStrings: [],
                toolCallPolicy: .init(validation: validation, onyxParsing: .referenceParity))
        else { return nil }
        self.adapter = adapter
    }

    public mutating func push(_ token: Int) -> [OnyxStreamEvent] {
        var events: [OnyxStreamEvent] = []
        _ = adapter.push(token) { event in
            if let event = OnyxStreamEvent(event) { events.append(event) }
            return true
        }
        return events
    }

    public mutating func finish() -> [OnyxStreamEvent] {
        var events: [OnyxStreamEvent] = []
        _ = adapter.finish { event in
            if let event = OnyxStreamEvent(event) { events.append(event) }
            return true
        }
        return events
    }
}

/// Decodes the generated text of Muse Glimmer, with its control markers
/// visible, into reasoning, content and tool calls. The rules are those of
/// ``OnyxParsingMode/referenceParity``.
///
/// Prefer ``OnyxTokenStreamDecoder`` when token ids are available: in text, a
/// marker the model spells out is the same as the control token. Chunks can
/// be cut anywhere, inside a marker too.
public struct OnyxTextStreamDecoder: Sendable {
    private var decoder: OnyxReferenceDecoder
    private var buffer = ""

    public var isInsideReasoning: Bool { decoder.isInsideReasoning }

    public init(
        tools: [[String: any Sendable]]?,
        validation: ToolCallValidationPolicy = .permissive
    ) {
        self.decoder = OnyxReferenceDecoder(tools: tools, validation: validation)
    }

    public mutating func process(_ chunk: String) -> [OnyxStreamEvent] {
        buffer += chunk
        var events: [TokenStreamEvent] = []
        while let (range, control) = nextMarker() {
            events += decoder.text(String(buffer[..<range.lowerBound]))
            events += decoder.control(control)
            buffer.removeSubrange(..<range.upperBound)
        }
        // The end of the buffer may be the first half of a marker.
        let ready = buffer.index(buffer.endIndex, offsetBy: -partialMarkerLength())
        events += decoder.text(String(buffer[..<ready]))
        buffer.removeSubrange(..<ready)
        return events.compactMap(OnyxStreamEvent.init)
    }

    public mutating func finish() -> [OnyxStreamEvent] {
        let events = decoder.text(buffer) + decoder.finish()
        buffer = ""
        return events.compactMap(OnyxStreamEvent.init)
    }

    private func nextMarker() -> (Range<String.Index>, OnyxControl)? {
        var first: (Range<String.Index>, OnyxControl)?
        for control in OnyxControl.allCases {
            guard let range = buffer.range(of: control.marker) else { continue }
            if first.map({ range.lowerBound < $0.0.lowerBound }) ?? true {
                first = (range, control)
            }
        }
        return first
    }

    private func partialMarkerLength() -> Int {
        let longest = OnyxControl.allCases.map(\.marker.count).max() ?? 1
        var length = min(buffer.count, longest - 1)
        while length > 0 {
            let tail = buffer.suffix(length)
            if OnyxControl.allCases.contains(where: { $0.marker.hasPrefix(tail) }) {
                return length
            }
            length -= 1
        }
        return 0
    }
}
