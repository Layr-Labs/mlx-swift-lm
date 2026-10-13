// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of dflash-mlx context-cache semantics at the revision in NOTICE.

import MLX

struct DFlash2ContextWindowPlan: Equatable, Sendable {
    let sinkSize: Int
    let windowSize: Int

    func spans(cacheLength: Int, inputLength: Int) -> [Range<Int>] {
        guard inputLength > 0 else { return [] }
        if cacheLength == 0 {
            let maximum = sinkSize + windowSize
            if inputLength <= maximum { return [0 ..< inputLength] }
            var result = [Range<Int>]()
            let sinkEnd = min(sinkSize, inputLength)
            if sinkEnd > 0 { result.append(0 ..< sinkEnd) }
            let tailStart = max(sinkEnd, inputLength - windowSize)
            if tailStart < inputLength { result.append(tailStart ..< inputLength) }
            return result
        }
        if inputLength <= windowSize { return [0 ..< inputLength] }
        return [(inputLength - windowSize) ..< inputLength]
    }
}

public func dflash2PrefillRanges(
    tokenCount: Int, stepSize: Int = 2048
) -> [Range<Int>] {
    stride(from: 0, to: tokenCount, by: stepSize).map {
        $0 ..< min($0 + stepSize, tokenCount)
    }
}

/// Context-only cache used by the fixed DFlash2 draft. It retains the same
/// 64-token sink plus 2,048-token tail as the pinned Python runtime while its
/// logical offset advances by every committed target row, including rows not
/// materialized in the window.
final class DFlash2ContextKVCache {
    private let plan: DFlash2ContextWindowPlan
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var positions: MLXArray?
    private(set) var offset = 0

    init(sinkSize: Int, windowSize: Int) {
        plan = DFlash2ContextWindowPlan(
            sinkSize: sinkSize, windowSize: windowSize)
    }

    var cacheLength: Int { keys?.dim(2) ?? 0 }

    func contextSpans(inputLength: Int) -> [Range<Int>] {
        plan.spans(cacheLength: cacheLength, inputLength: inputLength)
    }

    func appendContext(
        keys newKeys: MLXArray,
        values newValues: MLXArray,
        positions newPositions: MLXArray,
        inputLength: Int
    ) {
        if let keys, let values, let positions {
            self.keys = concatenated([keys, newKeys], axis: 2)
            self.values = concatenated([values, newValues], axis: 2)
            self.positions = concatenated([positions, newPositions], axis: 0)
        } else {
            keys = newKeys
            values = newValues
            positions = newPositions
        }
        offset += inputLength
        applyWindow()
    }

    private func applyWindow() {
        guard let keys, let values, let positions else { return }
        let maximum = plan.sinkSize + plan.windowSize
        guard keys.dim(2) > maximum else { return }
        self.keys = concatenated(
            [
                keys[0..., 0..., 0 ..< plan.sinkSize, 0...],
                keys[0..., 0..., (keys.dim(2) - plan.windowSize)..., 0...],
            ], axis: 2)
        self.values = concatenated(
            [
                values[0..., 0..., 0 ..< plan.sinkSize, 0...],
                values[0..., 0..., (values.dim(2) - plan.windowSize)..., 0...],
            ], axis: 2)
        self.positions = concatenated(
            [
                positions[0 ..< plan.sinkSize],
                positions[(positions.dim(0) - plan.windowSize)...],
            ], axis: 0)
    }

    func innerState() -> [MLXArray] {
        [keys, values, positions].compactMap { $0 }
    }
}
