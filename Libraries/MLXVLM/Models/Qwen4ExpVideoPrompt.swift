// Copyright © 2026 Eigen Labs.
// Timestamp layout follows Transformers v5.8.0 Qwen3VLProcessor; see NOTICE-QWEN4-PREPROCESSING.md.
import Foundation
import MLXLMCommon

enum Qwen4ExpVideoPrompt {
    static func expandedTokens(_ tokens: [Int], tokenizer: any Tokenizer,
                               grids: [THW], timestamps: [[Double]], merge: Int) throws -> [Int] {
        let source = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        guard tokenizer.encode(text: source, addSpecialTokens: false) == tokens,
              grids.count == timestamps.count, merge > 0 else {
            throw VLMError.processing("Qwen4 video prompt could not preserve its rendered template")
        }
        let startToken = "<|vision_start|>", videoToken = "<|video_pad|>", endToken = "<|vision_end|>"
        let slots = source.ranges(of: videoToken)
        guard slots.count == grids.count else {
            throw VLMError.processing("Qwen4 video placeholder count does not match supplied clips")
        }
        var result = "", cursor = source.startIndex
        for (index, slot) in slots.enumerated() {
            let grid = grids[index]
            guard grid.t > 0, grid.h > 0, grid.w > 0, grid.h % merge == 0, grid.w % merge == 0,
                  timestamps[index].count == grid.t else {
                throw VLMError.processing("Qwen4 video grid and timestamps disagree")
            }
            let count = try Qwen4ExpMediaGeometry.product(grid.h / merge, grid.w / merge)
            let hasStart = source[..<slot.lowerBound].hasSuffix(startToken)
            let hasEnd = source[slot.upperBound...].hasPrefix(endToken)
            guard hasStart == hasEnd else { throw VLMError.processing("Qwen4 video placeholder is malformed") }
            let lower = hasStart ? source.index(slot.lowerBound, offsetBy: -startToken.count) : slot.lowerBound
            let upper = hasEnd ? source.index(slot.upperBound, offsetBy: endToken.count) : slot.upperBound
            guard lower >= cursor else { throw VLMError.processing("Qwen4 video placeholders overlap") }
            result += source[cursor..<lower]
            for timestamp in timestamps[index] {
                guard timestamp.isFinite, timestamp >= 0 else { throw Qwen4ExpMediaGeometry.Failure.invalidFrameMetadata }
                result += String(format: "<%.1f seconds>", locale: Locale(identifier: "en_US_POSIX"), timestamp)
                result += startToken + String(repeating: videoToken, count: count) + endToken
            }
            cursor = upper
        }
        result += source[cursor...]
        return tokenizer.encode(text: result, addSpecialTokens: false)
    }

    /// The tower receives original T×H×W grids; M-RoPE sees one T=1 grid
    /// for each timestamp-delimited temporal group, exactly as the reference.
    static func positionGrids(_ grids: [THW]?) throws -> [THW]? {
        guard let grids else { return nil }
        var result: [THW] = []
        for grid in grids {
            guard grid.t > 0, grid.h > 0, grid.w > 0 else { throw Qwen4ExpMediaGeometry.Failure.invalidGeometry }
            _ = try Qwen4ExpMediaGeometry.product(grid.t, grid.h, grid.w)
            for _ in 0..<grid.t { result.append(THW(1, grid.h, grid.w)) }
        }
        return result
    }
}
