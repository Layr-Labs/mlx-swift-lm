// Copyright © 2025 Apple Inc.

import Foundation

/// Processes generated text to detect and extract tool calls during streaming generation.
///
/// `ToolCallProcessor` handles the streaming detection of tool calls in model output,
/// buffering partial content and extracting complete tool calls when detected.
///
/// Example:
/// ```swift
/// let processor = ToolCallProcessor(format: .lfm2)
/// for chunk in generatedChunks {
///     if let text = processor.processChunk(chunk) {
///         // Regular text to display
///         print(text)
///     }
/// }
/// // After generation completes:
/// for toolCall in processor.toolCalls {
///     // Handle extracted tool calls
///     print(toolCall.function.name)
/// }
/// ```
public class ToolCallProcessor {

    // MARK: - Properties

    private let format: ToolCallFormat
    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    private let qwenStructuredFrames: Bool
    private var qwenFrameScanner: Qwen35ToolFrameScanner?
    private let strictGemma: Bool
    private let glm5Stream: GLM5ToolStream?
    private var gemmaFrameScanner: GemmaToolFrameScanner?
    /// Unmatched end of the scanned payload, shorter than the longest end
    /// tag: other formats search only it plus each appended chunk.
    private var endTagScanTail: String.UnicodeScalarView?
    private var state = State.normal
    private var toolCallBuffer = ""
    private var activeEndTags: [String]?
    private let streaming: ToolCallStreamingOptions?
    private var callStream: CallStream?

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []
    /// Number of tool-shaped buffers that failed parser validation and were
    /// returned as visible text. Content is never included in telemetry.
    public private(set) var parseFailureCount = 0

    // MARK: - State Enum

    private enum State {
        case normal
        case potentialToolCall
        case collectingToolCall
    }

    /// Exposure state of the frame being collected in streaming mode. A
    /// reference, so mutating the streamer never copies the argument text it
    /// has accumulated (a value copy made each chunk quadratic).
    private final class CallStream {
        var streamer: any ToolCallArgumentStreamer
        let startTagScalars: Int
        /// Payload scalars handed to the streamer; `pending` follows them.
        var fedScalars = 0
        /// Payload tail withheld because it may begin the end tag.
        var pending = String.UnicodeScalarView()
        var primed = false
        var exposed = false
        /// The call completes in one piece (name rejected or not renderable).
        var declined = false

        init(streamer: any ToolCallArgumentStreamer, startTagScalars: Int) {
            self.streamer = streamer
            self.startTagScalars = startTagScalars
        }
    }

    // MARK: - Initialization

    /// Initialize with a specific tool call format.
    /// - Parameters:
    ///   - format: The tool call format to use (defaults to `.json` for standard JSON format)
    ///   - tools: Optional tool schemas for type-aware parsing
    ///   - strictGemma: Enables strict function-call parsing and frame scanning for `.gemma` format.
    ///   - streaming: Exposes calls while they are generated through the
    ///     segment API, for formats with an incremental argument renderer.
    public init(
        format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil,
        strictGemma: Bool = false, streaming: ToolCallStreamingOptions? = nil
    ) {
        self.format = format
        self.strictGemma = strictGemma && format == .gemma
        self.parser = self.strictGemma ? GemmaFunctionParser(strict: true) : format.createParser()
        self.tools = tools
        self.glm5Stream = format == .glm5 ? GLM5ToolStream(tools: tools) : nil
        // MiMo shares only the structural XML frame scanner. Its payload
        // parser/EOS policy remains a separate strict dialect.
        self.qwenStructuredFrames = format == .qwen35 || format == .mimoV2
        // Strict Gemma frames may legally contain their end tag inside
        // quoted values; they keep complete-frame exposure.
        self.streaming =
            ToolCallStreamingOptions.supports(format) && !self.strictGemma ? streaming : nil
    }

    /// True when `processChunkSegments` exposes calls before their frames
    /// close. Complete calls then arrive as segments, never in `toolCalls`.
    public var streamsCalls: Bool { streaming != nil }

    // MARK: - Computed Properties

    /// Whether this processor uses inline format (no start tag).
    private var isInlineFormat: Bool {
        parser.acceptedStartTags.isEmpty
    }

    /// The first characters of accepted start tags for quick detection.
    private var startTagFirstChars: [Character] {
        Array(Set(parser.acceptedStartTags.compactMap(\.first)))
    }

    // MARK: - Public Methods

    /// Process a generated text chunk and extract any tool call content.
    /// - Parameter chunk: The text chunk to process
    /// - Returns: Regular text that should be displayed (non-tool call content), or `nil` if buffering
    public func processChunk(_ chunk: String) -> String? {
        if let glm5Stream {
            let text = glm5Stream.processChunk(chunk)
            parseFailureCount = glm5Stream.failures
            return text
        }
        if isInlineFormat {
            return processInlineChunk(chunk)
        }
        var segments: [ToolCallStreamSegment] = []
        processTaggedChunk(chunk, into: &segments)
        return Self.visibleText(of: segments)
    }

    /// Ordered visible text and call events for one chunk. Without streaming
    /// options the text is exactly `processChunk`'s and complete calls still
    /// accumulate in `toolCalls`.
    public func processChunkSegments(_ chunk: String) -> [ToolCallStreamSegment] {
        // GLM5 never streams calls; it keeps its response-wide validation.
        if glm5Stream != nil {
            return processChunk(chunk).map { [.text($0)] } ?? []
        }
        if isInlineFormat {
            return processInlineChunk(chunk).map { [.text($0)] } ?? []
        }
        var segments: [ToolCallStreamSegment] = []
        processTaggedChunk(chunk, into: &segments)
        return segments
    }

    /// Drain all extracted tool calls in parse (FIFO) order, clearing the queue.
    ///
    /// Prefer this over reading/`popLast()`-ing `toolCalls` directly: it emits
    /// every complete tool call from a multi-call turn exactly once and in
    /// order. Upstream 1335fb5.
    public func drainToolCalls() -> [ToolCall] {
        let drained = toolCalls
        toolCalls = []
        return drained
    }

    /// Process end-of-sequence, parsing any buffered content as tool call(s).
    ///
    /// Call this when generation ends (e.g., on EOS token) to handle formats
    /// whose end tag is never delivered as text (e.g., Mistral where `</s>`
    /// is intercepted at the token ID level).
    ///
    /// For formats with end tags that appear in the text stream, the buffer
    /// will already be empty at generation end, making this a no-op.
    ///
    /// - Parameter returnBufferedText: when `true`, if the residual buffer did
    ///   not parse into any tool call it is returned as ordinary text instead of
    ///   being discarded (it was withheld only while probing for a tool call).
    /// - Returns: residual non-tool text when `returnBufferedText` is `true`,
    ///   otherwise `nil`.
    @discardableResult
    public func processEOS(returnBufferedText: Bool = false) -> String? {
        if let glm5Stream {
            let result = glm5Stream.finish()
            toolCalls.append(contentsOf: result.calls)
            parseFailureCount = glm5Stream.failures
            return returnBufferedText ? result.text : nil
        }
        guard state == .collectingToolCall || state == .potentialToolCall else { return nil }
        guard !toolCallBuffer.isEmpty else {
            state = .normal
            activeEndTags = nil
            qwenFrameScanner = nil
            gemmaFrameScanner = nil
            endTagScanTail = nil
            callStream = nil
            return nil
        }

        let parsed = parser.parseEOS(toolCallBuffer, tools: tools)
        toolCalls.append(contentsOf: parsed)
        if parsed.isEmpty { parseFailureCount += 1 }

        let buffered = toolCallBuffer
        toolCallBuffer = ""
        state = .normal
        activeEndTags = nil
        qwenFrameScanner = nil
        gemmaFrameScanner = nil
        endTagScanTail = nil
        callStream = nil

        if returnBufferedText && parsed.isEmpty {
            return buffered
        }
        return nil
    }

    /// End of generation for the segment API. An exposed call completes from
    /// the EOS parse (formats whose end tag is a stop token) or fails:
    /// `unterminated` when nothing parses, `diverged` when the parse
    /// disagrees with the exposed prefix. An unexposed buffer yields its
    /// calls, or itself as text when nothing parses
    /// (`processEOS(returnBufferedText: true)`).
    public func processEOSSegments() -> [ToolCallStreamSegment] {
        guard streaming != nil else {
            return processEOS(returnBufferedText: true).map { [.text($0)] } ?? []
        }
        guard state == .collectingToolCall || state == .potentialToolCall,
            !toolCallBuffer.isEmpty
        else {
            _ = processEOS()
            return []
        }
        var segments: [ToolCallStreamSegment] = []
        if state == .collectingToolCall, callStream != nil {
            feedRemainingPayload(upTo: toolCallBuffer.endIndex, into: &segments)
        }
        let parsed = parser.parseEOS(toolCallBuffer, tools: tools)
        if parsed.isEmpty { parseFailureCount += 1 }
        if let stream = callStream, stream.exposed {
            if parsed.count == 1, let call = parsed.first,
                call.function.name == stream.streamer.name,
                let remainder = stream.streamer.remainder(completing: call.function.arguments)
            {
                segments.append(.callCompleted(call: call, remainingArguments: remainder))
            } else {
                segments.append(.callFailed(parsed.isEmpty ? .unterminated : .diverged))
            }
        } else {
            segments += parsed.map { .call($0) }
            if parsed.isEmpty { segments.append(.text(toolCallBuffer)) }
        }
        toolCallBuffer = ""
        state = .normal
        activeEndTags = nil
        qwenFrameScanner = nil
        gemmaFrameScanner = nil
        endTagScanTail = nil
        callStream = nil
        return segments
    }

    // MARK: - Private Methods

    /// Process chunk for inline formats (no wrapper tags).
    ///
    /// Uses brace counting to detect when output looks like a JSON tool call.
    /// While braces are unbalanced the content is buffered (returns `nil`)
    /// so partial JSON is never leaked to the UI.
    private func processInlineChunk(_ chunk: String) -> String? {
        switch state {
        case .normal:
            // Check if this chunk starts what looks like a JSON tool call
            if let braceIndex = chunk.firstIndex(of: "{") {
                let leading = String(chunk[..<braceIndex])
                let jsonPart = String(chunk[braceIndex...])
                toolCallBuffer = jsonPart
                state = .collectingToolCall

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    toolCalls.append(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    return leading.isEmpty ? nil : leading
                }

                // Still collecting — check if braces are balanced (would mean parse
                // failed on complete JSON, so it's not a tool call)
                if jsonBracesBalanced(toolCallBuffer) {
                    parseFailureCount += 1
                    state = .normal
                    let buffer = toolCallBuffer
                    toolCallBuffer = ""
                    return leading + buffer
                }

                return leading.isEmpty ? nil : leading
            }

            // No brace seen — pass through as regular text
            return chunk

        case .potentialToolCall, .collectingToolCall:
            toolCallBuffer += chunk

            if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                toolCalls.append(toolCall)
                toolCallBuffer = ""
                state = .normal
                return nil
            }

            // If braces are balanced but parse failed, this isn't a tool call — flush
            if jsonBracesBalanced(toolCallBuffer) {
                parseFailureCount += 1
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return buffer
            }

            // Still collecting
            return nil
        }
    }

    /// Check whether open/close braces are balanced in the string.
    ///
    /// String-aware: braces inside a JSON string value (and escaped quotes) are
    /// ignored, so streaming content like `{"path": "a}b"` is not treated as a
    /// complete object the moment the in-string `}` arrives. A naive counter
    /// would prematurely flush partial JSON and corrupt the tool call. The
    /// result also requires the scanner to not end mid-string. (Upstream 1335fb5
    /// inString/isEscaped scanner, re-derived for our multi-tag processor.)
    private func jsonBracesBalanced(_ text: String) -> Bool {
        var depth = 0
        var inString = false
        var isEscaped = false
        for ch in text {
            if isEscaped {
                isEscaped = false
                continue
            }
            if ch == "\\" {
                if inString { isEscaped = true }
                continue
            }
            if ch == "\"" {
                inString.toggle()
                continue
            }
            if inString { continue }
            if ch == "{" {
                depth += 1
            } else if ch == "}" {
                depth -= 1
            }
        }
        return depth == 0 && !inString
    }

    /// Legacy `processChunk` text: nil exactly when no text segment exists.
    private static func visibleText(of segments: [ToolCallStreamSegment]) -> String? {
        var text: String?
        for case .text(let piece) in segments { text = (text ?? "") + piece }
        return text
    }

    /// Process chunk for tagged formats. Text segments concatenate to the
    /// historical visible text; call segments appear only in streaming mode.
    private func processTaggedChunk(_ chunk: String, into segments: inout [ToolCallStreamSegment]) {
        let startTags = parser.acceptedStartTags
        var endTags = activeEndTags ?? parser.acceptedEndTags

        guard !startTags.isEmpty else {
            segments.append(.text(chunk))
            return
        }

        guard
            (state == .normal && startTagFirstChars.contains(where: { chunk.contains($0) }))
                || state != .normal
        else {
            segments.append(.text(chunk))
            return
        }

        toolCallBuffer += chunk
        var leadingToken: String?
        // Qwen frames (and every streamed format) keep visible text that
        // precedes a still-open frame in the same chunk.
        let keepsLeadingText = qwenStructuredFrames || streaming != nil

        switch state {
        case .normal:
            // Change state to potential tool call
            state = .potentialToolCall

            leadingToken = separateToken(
                from: &toolCallBuffer, separators: startTags, returnLeading: true)
            if keepsLeadingText, leadingToken == nil {
                leadingToken = separatePartialStart(startTags)
            }

            fallthrough
        case .potentialToolCall:
            if let startTag = partialMatch(buffer: toolCallBuffer, tags: startTags) {
                if toolCallBuffer.starts(with: startTag) {
                    activeEndTags = parser.endTags(forStartTag: startTag)
                    endTags = activeEndTags!
                    state = .collectingToolCall
                    beginCallStream(startTag: startTag)
                    fallthrough
                } else {
                    if keepsLeadingText { appendText(leadingToken, to: &segments) }
                    return
                }
            } else {
                // Otherwise, return the collected text and reset the state
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                segments.append(.text((leadingToken ?? "") + buffer))
                return
            }
        case .collectingToolCall:
            guard !endTags.isEmpty else {
                return
            }

            if let endRange = completedEndRange(endTags, appendedChunk: chunk) {
                appendText(leadingToken, to: &segments)
                // Separate the trailing token
                let trailingToken: String? = String(toolCallBuffer[endRange.upperBound...])
                if callStream != nil {
                    feedRemainingPayload(upTo: endRange.lowerBound, into: &segments)
                }
                toolCallBuffer = String(toolCallBuffer[..<endRange.upperBound])

                // Parse the tool call using the parser
                let failedBuffer = toolCallBuffer
                let parsedCall = parser.parse(content: toolCallBuffer, tools: tools)
                completeCall(parsedCall, failedBuffer: failedBuffer, into: &segments)

                state = .normal
                activeEndTags = nil
                qwenFrameScanner = nil
                gemmaFrameScanner = nil
                endTagScanTail = nil
                toolCallBuffer = ""

                // If the token contains the start character, there may be more tool calls to come
                if let trailingToken,
                    startTagFirstChars.contains(where: { trailingToken.contains($0) })
                {
                    var trailing: [ToolCallStreamSegment] = []
                    processTaggedChunk(trailingToken, into: &trailing)
                    segments += trailing.filter { segment in
                        if case .text(let text) = segment { return !text.isEmpty }
                        return true
                    }
                } else {
                    appendText(trailingToken, to: &segments)
                }
            } else {
                if keepsLeadingText { appendText(leadingToken, to: &segments) }
                feedCallStream(chunk: chunk, endTags: endTags, into: &segments)
            }
        }
    }

    private func appendText(_ text: String?, to segments: inout [ToolCallStreamSegment]) {
        if let text, !text.isEmpty { segments.append(.text(text)) }
    }

    /// Records a closed frame: the historical call/text fallback, a whole
    /// call in streaming mode, or the completion of an exposed call checked
    /// against everything already exposed.
    private func completeCall(
        _ parsedCall: ToolCall?, failedBuffer: String, into segments: inout [ToolCallStreamSegment]
    ) {
        defer { callStream = nil }
        if parsedCall == nil { parseFailureCount += 1 }
        guard let stream = callStream, stream.exposed else {
            if let parsedCall {
                if streaming == nil {
                    toolCalls.append(parsedCall)
                } else {
                    segments.append(.call(parsedCall))
                }
            } else {
                // Historical auto output may be malformed. Returning the
                // exact withheld buffer as text is a visible, typed-safe
                // degradation; silently dropping it produced empty 200s.
                segments.append(.text(failedBuffer))
            }
            return
        }
        guard let parsedCall else {
            segments.append(.callFailed(.malformed))
            return
        }
        guard parsedCall.function.name == stream.streamer.name,
            let remainder = stream.streamer.remainder(completing: parsedCall.function.arguments)
        else {
            segments.append(.callFailed(.diverged))
            return
        }
        segments.append(.callCompleted(call: parsedCall, remainingArguments: remainder))
    }

    // MARK: - Call exposure

    private func beginCallStream(startTag: String) {
        callStream = nil
        guard streaming != nil,
            let streamer = makeToolCallArgumentStreamer(format: format, tools: tools)
        else { return }
        callStream = CallStream(streamer: streamer, startTagScalars: startTag.unicodeScalars.count)
    }

    /// Feeds an open frame's new payload, withholding a tail that may begin
    /// one of `endTags` so the streamer never sees closing markup.
    private func feedCallStream(
        chunk: String, endTags: [String], into segments: inout [ToolCallStreamSegment]
    ) {
        guard let stream = callStream, !stream.declined else { return }
        var text = stream.pending
        if stream.primed {
            text.append(contentsOf: chunk.unicodeScalars)
        } else {
            let scalars = toolCallBuffer.unicodeScalars
            text.append(
                contentsOf: scalars[
                    scalars.index(scalars.startIndex, offsetBy: stream.startTagScalars)...])
        }
        let held = Self.endTagPrefixLength(of: text, endTags: endTags)
        let split = text.index(text.endIndex, offsetBy: -held)
        stream.pending = String.UnicodeScalarView(text[split...])
        feed(Substring(text[..<split]), into: &segments)
    }

    /// Feeds the whole unfed payload before `end` (the closing tag, or the
    /// end of the buffer at EOS), including any withheld tail.
    private func feedRemainingPayload(
        upTo end: String.Index, into segments: inout [ToolCallStreamSegment]
    ) {
        guard let stream = callStream, !stream.declined else { return }
        let scalars = toolCallBuffer.unicodeScalars
        let start = scalars.index(
            scalars.startIndex, offsetBy: stream.startTagScalars + stream.fedScalars)
        stream.pending = String.UnicodeScalarView()
        guard start < end else { return }
        feed(Substring(scalars[start ..< end]), into: &segments)
    }

    private func feed(_ text: Substring, into segments: inout [ToolCallStreamSegment]) {
        guard let stream = callStream, let streaming else { return }
        stream.primed = true
        stream.fedScalars += text.unicodeScalars.count
        let fragment = stream.streamer.append(String(text))
        if !stream.exposed {
            if let name = stream.streamer.name {
                if streaming.exposesName(name) {
                    stream.exposed = true
                    segments.append(.callStarted(name: name))
                } else {
                    stream.declined = true
                }
            } else if stream.streamer.abandoned {
                stream.declined = true
            }
        }
        if stream.exposed, !fragment.isEmpty {
            segments.append(.callArguments(fragment))
        }
    }

    /// Length of the longest suffix of `text` that is a proper prefix of an
    /// end tag.
    private static func endTagPrefixLength(of text: String.UnicodeScalarView, endTags: [String])
        -> Int
    {
        let tags = endTags.map { Array($0.unicodeScalars) }
        let tail = Array(text.suffix(max(0, (tags.map(\.count).max() ?? 1) - 1)))
        for length in stride(from: tail.count, to: 0, by: -1) {
            let suffix = tail[(tail.count - length)...]
            if tags.contains(where: { $0.count > length && $0.starts(with: suffix) }) {
                return length
            }
        }
        return 0
    }

    /// Preserve visible prefix text while retaining only a split opening
    /// marker (Qwen frames and every streamed format). Other parsers keep
    /// their existing start-tag behavior.
    private func separatePartialStart(_ tags: [String]) -> String? {
        let maximum = min(toolCallBuffer.count, (tags.map(\.count).max() ?? 1) - 1)
        guard maximum > 0 else { return nil }
        for count in (1 ... maximum).reversed() {
            let suffix = String(toolCallBuffer.suffix(count))
            guard tags.contains(where: { $0.hasPrefix(suffix) }) else { continue }
            let split = toolCallBuffer.index(toolCallBuffer.endIndex, offsetBy: -count)
            let leading = String(toolCallBuffer[..<split])
            toolCallBuffer = suffix
            return leading.isEmpty ? nil : leading
        }
        return nil
    }

    private func completedEndRange(_ endTags: [String], appendedChunk: String) -> Range<
        String.Index
    >? {
        if strictGemma, let start = parser.startTag, let end = parser.endTag {
            let input =
                gemmaFrameScanner == nil
                ? toolCallBuffer.dropFirst(start.count) : appendedChunk[...]
            var scanner = gemmaFrameScanner ?? GemmaToolFrameScanner()
            let scalars = input.unicodeScalars
            for index in scalars.indices {
                if scanner.consume(scalars[index]) {
                    let remaining = scalars.distance(
                        from: scalars.index(after: index), to: scalars.endIndex)
                    let upper = toolCallBuffer.unicodeScalars.index(
                        toolCallBuffer.unicodeScalars.endIndex, offsetBy: -remaining)
                    gemmaFrameScanner = scanner
                    return toolCallBuffer.unicodeScalars.index(
                        upper, offsetBy: -end.unicodeScalars.count) ..< upper
                }
            }
            gemmaFrameScanner = scanner
            return nil
        }
        if qwenStructuredFrames, let start = parser.startTag, let end = parser.endTag {
            // The first pass includes any split opening wrapper. Subsequent
            // passes scan only newly appended characters, never the entire
            // growing payload (which would make one-token chunks quadratic).
            let input =
                qwenFrameScanner == nil ? toolCallBuffer.dropFirst(start.count) : appendedChunk[...]
            var scanner = qwenFrameScanner ?? Qwen35ToolFrameScanner(endTag: end)
            let scalars = input.unicodeScalars
            for index in scalars.indices {
                if scanner.consume(scalars[index]) {
                    let remaining = scalars.distance(
                        from: scalars.index(after: index), to: scalars.endIndex)
                    let upper = toolCallBuffer.unicodeScalars.index(
                        toolCallBuffer.unicodeScalars.endIndex, offsetBy: -remaining)
                    qwenFrameScanner = scanner
                    return toolCallBuffer.unicodeScalars.index(
                        upper, offsetBy: -end.unicodeScalars.count) ..< upper
                }
            }
            qwenFrameScanner = scanner
            return nil
        }
        // The first pass searches the whole buffer, as before. Later passes
        // search only the appended chunk and the unmatched tail before it,
        // the only place a new end tag can complete; rescanning the growing
        // payload every chunk made long calls quadratic.
        let window: String
        if let tail = endTagScanTail {
            var scalars = tail
            scalars.append(contentsOf: appendedChunk.unicodeScalars)
            window = String(scalars)
        } else {
            window = toolCallBuffer
        }
        guard let separator = firstSeparator(in: window, separators: endTags),
            let match = window.range(of: separator)
        else {
            let longest = endTags.map(\.unicodeScalars.count).max() ?? 1
            endTagScanTail = String.UnicodeScalarView(window.unicodeScalars.suffix(longest - 1))
            return nil
        }
        // The window is a suffix of the buffer: map the match by its distance
        // from the end.
        let fromEnd = window.unicodeScalars.distance(
            from: match.lowerBound, to: window.unicodeScalars.endIndex)
        let scalars = toolCallBuffer.unicodeScalars
        let lower = scalars.index(scalars.endIndex, offsetBy: -fromEnd)
        return lower ..< scalars.index(lower, offsetBy: separator.unicodeScalars.count)
    }

    private func separateToken(
        from buffer: inout String, separators: [String], returnLeading: Bool
    ) -> String? {
        guard let separator = firstSeparator(in: buffer, separators: separators) else {
            return nil
        }
        return separateToken(from: &buffer, separator: separator, returnLeading: returnLeading)
    }

    /// Separates a token from a string buffer based on a separator
    /// - Parameters:
    ///   - buffer: The string buffer to modify
    ///   - separator: The separator string to search for
    ///   - returnLeading: If true, returns text before separator; if false, returns text after
    /// - Returns: The separated token, or nil if separator not found
    private func separateToken(from buffer: inout String, separator: String, returnLeading: Bool)
        -> String?
    {
        guard let range = buffer.range(of: separator) else { return nil }

        let token: String
        if returnLeading {
            token = String(buffer[..<range.lowerBound])
            buffer = String(buffer[range.lowerBound...])
        } else {
            token = String(buffer[range.upperBound...])
            buffer = String(buffer[..<range.upperBound])
        }

        return token
    }

    private func partialMatch(buffer: String, tag: String) -> Bool {
        for (tagIndex, bufferIndex) in zip(tag.indices, buffer.indices) {
            if buffer[bufferIndex] != tag[tagIndex] {
                return false
            }
        }

        return true
    }

    private func partialMatch(buffer: String, tags: [String]) -> String? {
        tags.first { partialMatch(buffer: buffer, tag: $0) }
    }

    private func firstSeparator(in buffer: String, separators: [String]) -> String? {
        var match: (range: Range<String.Index>, separator: String)?
        for separator in separators {
            guard let range = buffer.range(of: separator) else { continue }
            if match == nil || range.lowerBound < match!.range.lowerBound {
                match = (range, separator)
            }
        }
        return match?.separator
    }
}
