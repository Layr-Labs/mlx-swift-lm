// Copyright © 2026 Eigen Labs.

import Foundation

/// GLM's JSON arguments may contain its XML delimiters as quoted data. Keep the
/// response's complete tool suffix until EOS instead of guessing frame endings.
/// This also prevents a later malformed call from committing an earlier call.
final class GLM5ToolStream {
    private static let opening = "<tool_call>"
    private static let maximumBytes = 1 << 20

    private let tools: [[String: any Sendable]]?
    private var buffer = ""
    private var collecting = false
    private var rejected = false
    private(set) var failures = 0

    init(tools: [[String: any Sendable]]?) {
        self.tools = tools
    }

    func processChunk(_ chunk: String) -> String? {
        if rejected { return chunk.isEmpty ? nil : chunk }
        // Check before appending: the retained tool suffix is bounded even when
        // one model delivery is much larger than the accepted frame budget.
        if collecting && chunk.utf8.count > Self.maximumBytes - buffer.utf8.count {
            rejected = true
            failures += 1
            let visible = buffer + chunk
            buffer = ""
            return visible
        }
        buffer += chunk
        if collecting { return nil }

        if let start = buffer.range(of: Self.opening) {
            let visible = String(buffer[..<start.lowerBound])
            buffer = String(buffer[start.lowerBound...])
            collecting = true
            if buffer.utf8.count > Self.maximumBytes {
                rejected = true
                failures += 1
                let result = visible + buffer
                buffer = ""
                return result
            }
            return visible.isEmpty ? nil : visible
        }

        // Only an unfinished opening marker needs buffering before a call.
        // Use character boundaries; generated String chunks cannot split UTF-8.
        let keep =
            (1 ..< Self.opening.count).reversed().first {
                buffer.hasSuffix(String(Self.opening.prefix($0)))
            } ?? 0
        let split = buffer.index(buffer.endIndex, offsetBy: -keep)
        let visible = String(buffer[..<split])
        buffer = String(buffer[split...])
        return visible.isEmpty ? nil : visible
    }

    func finish() -> (calls: [ToolCall], text: String?) {
        defer {
            buffer = ""
            collecting = false
            rejected = false
        }
        guard !buffer.isEmpty else { return ([], nil) }
        guard collecting else { return ([], buffer) }
        let calls = GLM5ToolCallParser().parseEOS(buffer, tools: tools)
        guard !calls.isEmpty else {
            failures += 1
            return ([], buffer)
        }
        return (calls, nil)
    }
}
