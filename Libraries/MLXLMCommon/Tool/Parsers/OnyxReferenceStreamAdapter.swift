// Copyright © 2026 Eigen Labs.

/// Token-stream decoder for Muse Glimmer that reads Onyx frames and ATEM
/// calls the way the reference servers do (`OnyxReferenceDecoder`).
/// `ToolCallPolicy.onyxParsing` selects it; `OnyxStreamAdapter` stays the
/// default.
///
/// Control markers are compared as token ids, as in `OnyxStreamAdapter`. Text
/// that only looks like a marker can never open or close a frame.
struct OnyxReferenceStreamAdapter: TokenStreamDecoder {
    private let tokenizer: any Tokenizer
    private let controls: [Int: OnyxControl]
    private var detokenizer: any StreamingDetokenizer
    private var decoder: OnyxReferenceDecoder
    private var stopStringFilter: StopStringFilter
    private(set) var rejectedToolCallCount = 0
    let additionalStopTokenIDs: Set<Int>
    let receivesStopTokens = true

    var isInsideReasoning: Bool { decoder.isInsideReasoning }

    init?(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        var controls: [Int: OnyxControl] = [:]
        for control in OnyxControl.allCases {
            guard let id = tokenizer.convertTokenToId(control.marker) else {
                // Only the end-of-text token is optional.
                if control == .endOfText { continue }
                return nil
            }
            guard controls[id] == nil else { return nil }
            controls[id] = control
        }
        self.tokenizer = tokenizer
        self.controls = controls
        self.detokenizer = tokenizer.makeStreamingDetokenizer()
        self.decoder = OnyxReferenceDecoder(tools: tools, validation: toolCallPolicy.validation)
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
        // `<|eom|>` ends a message, not the turn: more frames follow it.
        self.additionalStopTokenIDs = Set(
            ["<|eot|>", "<|end_of_text|>"].compactMap(tokenizer.convertTokenToId))
    }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        var events: [TokenStreamEvent] = []
        if let control = controls[token] {
            events += flushText()
            events += decoder.control(control)
        } else {
            detokenizer.append(token: token)
            if let piece = detokenizer.next() { events += decoder.text(piece) }
        }
        return deliver(events, emit: emit)
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        guard deliver(flushText() + decoder.finish(), emit: emit) else { return false }
        if let text = stopStringFilter.finish() { return emit(.response(text)) }
        return true
    }

    /// Text is decoded one frame part at a time, so a control token never
    /// sits between two halves of a character.
    private mutating func flushText() -> [TokenStreamEvent] {
        let rest = detokenizer.finish()
        detokenizer = tokenizer.makeStreamingDetokenizer()
        return rest.map { decoder.text($0) } ?? []
    }

    private mutating func deliver(
        _ events: [TokenStreamEvent], emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        for event in events {
            switch event {
            case .response(let text):
                // Stop strings apply to the answer, not to reasoning.
                let result = stopStringFilter.process(text)
                if let response = result.text, !emit(.response(response)) { return false }
                if result.stopped {
                    _ = emit(.stop)
                    return false
                }
            case .toolCall, .rejectedToolCall:
                if case .rejectedToolCall = event { rejectedToolCallCount += 1 }
                if let response = stopStringFilter.finish(), !emit(.response(response)) {
                    return false
                }
                if !emit(event) { return false }
            case .reasoning, .protocolError, .stop:
                if !emit(event) { return false }
            }
        }
        return true
    }
}
