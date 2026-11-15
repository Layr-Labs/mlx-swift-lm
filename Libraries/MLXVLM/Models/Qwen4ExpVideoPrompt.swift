// Copyright © 2026 Eigen Labs.
// Timestamp layout follows Transformers v5.8.0; see docs/qwen4/preprocessing.md.
import Foundation
import MLXLMCommon

enum Qwen4ExpVideoPrompt {
    private static let startToken = "<|vision_start|>"
    private static let videoToken = "<|video_pad|>"
    private static let endToken = "<|vision_end|>"

    static func expandedTokens(
        _ tokens: [Int], tokenizer: any Tokenizer,
        grids: [THW], timestamps: [[Double]], merge: Int
    ) throws -> [Int] {
        guard grids.count == timestamps.count, merge > 0 else {
            throw VLMError.processing("Qwen video prompt could not preserve its rendered template")
        }
        let source = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        guard tokenizer.encode(text: source, addSpecialTokens: false) == tokens else {
            return try expandedTokenIDs(
                tokens, tokenizer: tokenizer, grids: grids, timestamps: timestamps, merge: merge)
        }
        let slots = source.ranges(of: videoToken)
        guard slots.count == grids.count else {
            throw VLMError.processing("Qwen video placeholder count does not match supplied clips")
        }
        var result = ""
        var cursor = source.startIndex
        for (index, slot) in slots.enumerated() {
            let count = try groupTokenCount(grids[index], timestamps[index], merge: merge)
            let hasStart = source[..<slot.lowerBound].hasSuffix(startToken)
            let hasEnd = source[slot.upperBound...].hasPrefix(endToken)
            guard hasStart == hasEnd else {
                throw VLMError.processing("Qwen video placeholder is malformed")
            }
            let lower =
                hasStart
                ? source.index(slot.lowerBound, offsetBy: -startToken.count) : slot.lowerBound
            let upper =
                hasEnd ? source.index(slot.upperBound, offsetBy: endToken.count) : slot.upperBound
            guard lower >= cursor else {
                throw VLMError.processing("Qwen video placeholders overlap")
            }
            result += source[cursor ..< lower]
            for timestamp in timestamps[index] {
                result += try label(timestamp)
                result += startToken + String(repeating: videoToken, count: count) + endToken
            }
            cursor = upper
        }
        result += source[cursor...]
        return tokenizer.encode(text: result, addSpecialTokens: false)
    }

    /// The same expansion in token space, for a template the tokenizer does not
    /// decode and re-encode unchanged. Each label is encoded on its own, so a
    /// label after a space or punctuation may split where the text path merges.
    private static func expandedTokenIDs(
        _ tokens: [Int], tokenizer: any Tokenizer,
        grids: [THW], timestamps: [[Double]], merge: Int
    ) throws -> [Int] {
        guard let start = tokenizer.convertTokenToId(startToken),
            let pad = tokenizer.convertTokenToId(videoToken),
            let end = tokenizer.convertTokenToId(endToken)
        else {
            throw VLMError.processing("Qwen video prompt could not preserve its rendered template")
        }
        let slots = tokens.indices.filter { tokens[$0] == pad }
        guard slots.count == grids.count else {
            throw VLMError.processing("Qwen video placeholder count does not match supplied clips")
        }
        var result: [Int] = []
        var cursor = 0
        for (index, slot) in slots.enumerated() {
            let count = try groupTokenCount(grids[index], timestamps[index], merge: merge)
            let hasStart = slot > 0 && tokens[slot - 1] == start
            let hasEnd = slot + 1 < tokens.count && tokens[slot + 1] == end
            guard hasStart == hasEnd else {
                throw VLMError.processing("Qwen video placeholder is malformed")
            }
            let lower = hasStart ? slot - 1 : slot
            guard lower >= cursor else {
                throw VLMError.processing("Qwen video placeholders overlap")
            }
            result += tokens[cursor ..< lower]
            for timestamp in timestamps[index] {
                result += tokenizer.encode(text: try label(timestamp), addSpecialTokens: false)
                result += [start] + Array(repeating: pad, count: count) + [end]
            }
            cursor = hasEnd ? slot + 2 : slot + 1
        }
        return result + tokens[cursor...]
    }

    /// Video pad tokens of one temporal group of `grid`.
    private static func groupTokenCount(_ grid: THW, _ timestamps: [Double], merge: Int) throws
        -> Int
    {
        guard grid.t > 0, grid.h > 0, grid.w > 0, grid.h % merge == 0, grid.w % merge == 0,
            timestamps.count == grid.t
        else {
            throw VLMError.processing("Qwen video grid and timestamps disagree")
        }
        return try Qwen4ExpMediaGeometry.product(grid.h / merge, grid.w / merge)
    }

    private static func label(_ timestamp: Double) throws -> String {
        guard timestamp.isFinite, timestamp >= 0 else {
            throw Qwen4ExpMediaGeometry.Failure.invalidFrameMetadata
        }
        // Unlocalized formatting rounds the binary value as Python's `:.1f` does
        // (0.35 → 0.3); the `locale:` form rounds its decimal digits (0.35 → 0.4).
        return String(format: "<%.1f seconds>", timestamp)
    }

    /// The tower receives original T×H×W grids; M-RoPE sees one T=1 grid
    /// for each timestamp-delimited temporal group, exactly as the reference.
    static func positionGrids(_ grids: [THW]?) throws -> [THW]? {
        guard let grids else { return nil }
        var result: [THW] = []
        for grid in grids {
            guard grid.t > 0, grid.h > 0, grid.w > 0 else {
                throw Qwen4ExpMediaGeometry.Failure.invalidGeometry
            }
            _ = try Qwen4ExpMediaGeometry.product(grid.t, grid.h, grid.w)
            for _ in 0 ..< grid.t { result.append(THW(1, grid.h, grid.w)) }
        }
        return result
    }
}
