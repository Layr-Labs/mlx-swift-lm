// Copyright © 2026 Eigen Labs.

import Foundation

/// The control markers of an Onyx stream.
enum OnyxControl: Sendable, CaseIterable {
    case start
    case message
    case endOfMessage
    case endOfTurn
    case endOfText

    var marker: String {
        switch self {
        case .start: "<|start|>"
        case .message: "<|message|>"
        case .endOfMessage: "<|eom|>"
        case .endOfTurn: "<|eot|>"
        case .endOfText: "<|end_of_text|>"
        }
    }
}

/// Onyx framing read the way the reference servers (vLLM, SGLang, llama.cpp)
/// read it. `OnyxStreamAdapter` keeps upstream's strict reading.
///
/// The caller splits the stream into control markers and the text between
/// them. A caller that has token ids recognizes a marker by its id, so text
/// that only looks like a marker stays text.
///
/// What differs from the strict reading:
/// - A tool frame gives one call for each complete invoke
///   (`ATEMReferenceScanner`), and the invoke name is not compared with the
///   frame's recipient.
/// - A frame that generation leaves open keeps its text. An open tool frame
///   gives the calls whose invoke closed, and no call from one that did not.
/// - A frame left open before a bare `to=…<|message|>` header ends there.
///   The model does this before calls that have no arguments.
/// - Output that never opens a frame is content.
///
/// Error messages never hold model output.
struct OnyxReferenceDecoder: Sendable {
    private enum Channel: Equatable {
        case reasoning
        case content
        case tool
        /// `to=` without a recipient.
        case ignored
    }

    private enum State {
        /// Text of a header before its `<|message|>`. The first header
        /// continues the prompt's `<|start|>assistant`.
        case header(String, continuesPrompt: Bool)
        case body(Channel)
        /// After a frame closed. Text here can only be a bare header.
        case between(String)
    }

    private enum BodyEnd {
        case terminator
        case nextFrame
        case generationEnd
    }

    private static let maximumHeaderLength = 512

    private let tools: ATEMDeclaredTools
    private let toolSpecs: [[String: any Sendable]]?
    private let validation: ToolCallValidationPolicy
    private var state: State = .header("", continuesPrompt: true)
    /// The end of a reasoning or content body that is not sent yet: trailing
    /// whitespace, and a word that can still become a bare header.
    private var held = ""
    private var bodyHasText = false
    private var toolBody = ""

    init(tools: [[String: any Sendable]]?, validation: ToolCallValidationPolicy) {
        self.tools = ATEMDeclaredTools(tools)
        self.toolSpecs = tools
        self.validation = validation
    }

    var isInsideReasoning: Bool {
        if case .body(.reasoning) = state { return true }
        return false
    }

    // MARK: - Input

    mutating func text(_ piece: String) -> [TokenStreamEvent] {
        guard !piece.isEmpty else { return [] }
        switch state {
        case .header(let text, let continuesPrompt):
            let header = text + piece
            if continuesPrompt, !Self.canBecomeHeader(header) {
                // No frame is coming: the output is plain content.
                _ = open(.content)
                return bodyText(header)
            }
            guard header.count <= Self.maximumHeaderLength else {
                state = .between("")
                return [.protocolError("Onyx frame header is too long")]
            }
            state = .header(header, continuesPrompt: continuesPrompt)
            return []
        case .body(.reasoning), .body(.content):
            return bodyText(piece)
        case .body(.tool):
            toolBody += piece
            return []
        case .body(.ignored):
            return []
        case .between(let text):
            state = .between(String((text + piece).suffix(Self.maximumHeaderLength)))
            return []
        }
    }

    mutating func control(_ control: OnyxControl) -> [TokenStreamEvent] {
        switch control {
        case .start:
            var events = endBody(.nextFrame)
            switch state {
            case .header(let text, _) where Self.hasText(text):
                events.append(.protocolError("Onyx frame header was interrupted by <|start|>"))
            case .between(let text) where Self.hasText(text):
                events.append(.protocolError("Unexpected text between Onyx frames"))
            case .body:
                events.append(.protocolError("Onyx frame was interrupted by <|start|>"))
            default:
                break
            }
            state = .header("", continuesPrompt: false)
            return events

        case .message:
            switch state {
            case .header(let text, let continuesPrompt):
                guard let channel = Self.channel(header: text, continuesPrompt: continuesPrompt)
                else {
                    state = .between("")
                    return [.protocolError("Invalid Onyx frame header")]
                }
                return open(channel)
            case .body(let channel):
                return bareHeader(inside: channel)
            case .between(let text):
                // A header without `<|start|>assistant`. Without a recipient
                // the frame is content.
                let word = text.split(whereSeparator: \.isWhitespace).last ?? ""
                if let recipient = Self.bareRecipient(word) {
                    return open(Self.channel(recipient: recipient))
                }
                let stray: [TokenStreamEvent] =
                    Self.hasText(text)
                    ? [.protocolError("Unexpected text between Onyx frames")] : []
                return stray + open(.content)
            }

        case .endOfMessage, .endOfTurn:
            var events = endBody(.terminator)
            if case .header(let text, _) = state, Self.hasText(text) {
                events.append(.protocolError("Onyx frame header ended before <|message|>"))
            }
            state = .between("")
            return events

        case .endOfText:
            return finish()
        }
    }

    /// Generation ended. An open frame keeps what it holds.
    mutating func finish() -> [TokenStreamEvent] {
        var events = endBody(.generationEnd)
        if case .header(let text, _) = state, Self.hasText(text) {
            // The start of a header is framing, never answer text.
            events.append(.protocolError("Incomplete Onyx frame header at generation end"))
        }
        state = .between("")
        return events
    }

    // MARK: - Frames

    private mutating func open(_ channel: Channel) -> [TokenStreamEvent] {
        state = .body(channel)
        held = ""
        bodyHasText = false
        toolBody = ""
        return channel == .ignored
            ? [.protocolError("Rejected Onyx frame with an empty recipient")] : []
    }

    private mutating func bodyText(_ piece: String) -> [TokenStreamEvent] {
        held += piece
        let keep = Self.heldTailStart(held, atBodyStart: !bodyHasText)
        let ready = String(held[..<keep])
        held = String(held[keep...])
        return send(ready)
    }

    private mutating func send(_ text: String) -> [TokenStreamEvent] {
        guard !text.isEmpty else { return [] }
        bodyHasText = true
        return [isInsideReasoning ? .reasoning(text) : .response(text)]
    }

    private mutating func endBody(_ end: BodyEnd) -> [TokenStreamEvent] {
        guard case .body(let channel) = state else { return [] }
        defer {
            held = ""
            toolBody = ""
        }
        switch channel {
        case .reasoning, .content:
            return send(held)
        case .tool:
            return toolEvents(toolBody, cutOff: end == .generationEnd)
        case .ignored:
            return []
        }
    }

    /// `<|message|>` inside a body. When the body ends with `to=…`, that is
    /// the header of the next frame and this frame ends before it.
    private mutating func bareHeader(inside channel: Channel) -> [TokenStreamEvent] {
        let body = channel == .tool ? toolBody : held
        var wordStart = body.endIndex
        while wordStart > body.startIndex, !body[body.index(before: wordStart)].isWhitespace {
            wordStart = body.index(before: wordStart)
        }
        var separatorStart = wordStart
        while separatorStart > body.startIndex,
            body[body.index(before: separatorStart)].isWhitespace
        {
            separatorStart = body.index(before: separatorStart)
        }
        let startsBody = channel == .tool ? true : !bodyHasText
        guard let recipient = Self.bareRecipient(body[wordStart...]),
            separatorStart < wordStart || (wordStart == body.startIndex && startsBody)
        else {
            return [.protocolError("Unexpected <|message|> inside an Onyx frame")]
        }
        // The whitespace before the header separates the frames.
        let before = String(body[..<separatorStart])
        var events: [TokenStreamEvent]
        switch channel {
        case .reasoning, .content: events = send(before)
        case .tool: events = toolEvents(before, cutOff: false)
        case .ignored: events = []
        }
        events += open(Self.channel(recipient: recipient))
        return events
    }

    // MARK: - Tool frames

    private func toolEvents(_ body: String, cutOff: Bool) -> [TokenStreamEvent] {
        let scan = ATEMReferenceScanner.scan(body, cutOff: cutOff)
        var events = scan.invokes.map(event(for:))
        if scan.endsInsideInvoke {
            events.append(
                .rejectedToolCall(
                    RejectedToolCall(
                        reason: cutOff ? .incompleteOutput : .malformedSyntax, format: .atem,
                        rawText: String(scan.unfinished ?? ""))))
        } else if scan.invokes.isEmpty {
            if cutOff {
                events.append(
                    .rejectedToolCall(
                        RejectedToolCall(
                            reason: .incompleteOutput, format: .atem, rawText: body)))
            } else if body.contains("<atem:") {
                events.append(
                    .rejectedToolCall(
                        RejectedToolCall(
                            reason: .malformedSyntax, format: .atem, rawText: body)))
            } else {
                events.append(.protocolError("Onyx tool frame carried no ATEM call"))
            }
        }
        return events
    }

    private func event(for invoke: ATEMReferenceScanner.Invoke) -> TokenStreamEvent {
        let text = String(invoke.text)
        guard let generated = invoke.name, !generated.isEmpty else {
            return .rejectedToolCall(
                RejectedToolCall(reason: .missingToolName, format: .atem, rawText: text))
        }
        guard let name = tools.resolve(generated) else {
            // A name that was not offered never reaches dispatch.
            return .rejectedToolCall(
                RejectedToolCall(
                    reason: .undeclaredTool, format: .atem, toolName: generated, rawText: text))
        }
        var arguments: [String: JSONValue] = [:]
        for parameter in invoke.parameters {
            arguments[parameter.name] = tools.value(
                parameter.value, parameter: parameter.name, ofTool: name)
        }
        let call = ToolArgumentNormalization.normalize(
            ToolCall(function: .init(name: name, arguments: arguments)), tools: toolSpecs)
        if validation == .strict,
            case .invalid(let violations) = ToolSchemaValidator.validate(
                arguments: call.function.arguments, forToolNamed: name, in: toolSpecs)
        {
            return .rejectedToolCall(
                RejectedToolCall(
                    reason: .invalidArguments, format: .atem, toolName: name, rawText: text,
                    detail: ToolSchemaValidator.describe(violations)))
        }
        return .toolCall(
            ToolCall(function: call.function, id: ToolCallFormat.atem.generateToolCallID()))
    }

    // MARK: - Headers

    private static func hasText(_ text: String) -> Bool {
        text.contains { !$0.isWhitespace }
    }

    private static func channel(header: String, continuesPrompt: Bool) -> Channel? {
        let fields = ((continuesPrompt ? "assistant " : "") + header)
            .split(whereSeparator: \.isWhitespace)
        guard fields.first == "assistant" else { return nil }
        guard let field = fields.dropFirst().first(where: { $0.hasPrefix("to=") }) else {
            return .content
        }
        return channel(recipient: field.dropFirst(3))
    }

    private static func channel(recipient: Substring) -> Channel {
        switch recipient {
        case "self": .reasoning
        case "user": .content
        case "": .ignored
        default: .tool
        }
    }

    /// The recipient of `to=…` when `word` is exactly that.
    private static func bareRecipient(_ word: Substring) -> Substring? {
        guard word.hasPrefix("to=") else { return nil }
        let recipient = word.dropFirst(3)
        guard !recipient.isEmpty, !recipient.contains(where: { $0.isWhitespace || $0 == "<" })
        else { return nil }
        return recipient
    }

    /// Whether `word` is `t`, `to` or `to=…`: a header that is not complete yet.
    private static func canBecomeBareHeader(_ word: Substring) -> Bool {
        if word.isEmpty { return false }
        if "to=".hasPrefix(word) { return true }
        return word.hasPrefix("to=") && !word.contains("<")
    }

    /// Whether the first text of a stream can still be its header.
    private static func canBecomeHeader(_ text: String) -> Bool {
        let header = text.drop(while: \.isWhitespace)
        if header.isEmpty { return true }
        return canBecomeBareHeader(header) && !header.contains(where: \.isWhitespace)
    }

    /// Where the part of a body starts that must wait: its trailing
    /// whitespace, and a last word that can become a bare header when
    /// whitespace or the start of the body comes before it.
    private static func heldTailStart(_ text: String, atBodyStart: Bool) -> String.Index {
        var wordStart = text.endIndex
        while wordStart > text.startIndex, !text[text.index(before: wordStart)].isWhitespace {
            wordStart = text.index(before: wordStart)
        }
        var separatorStart = wordStart
        while separatorStart > text.startIndex,
            text[text.index(before: separatorStart)].isWhitespace
        {
            separatorStart = text.index(before: separatorStart)
        }
        let word = text[wordStart...]
        if word.isEmpty { return separatorStart }
        guard canBecomeBareHeader(word),
            separatorStart < wordStart || (wordStart == text.startIndex && atBodyStart)
        else { return text.endIndex }
        return separatorStart
    }
}
