import Foundation
import MLX
import MLXLMCommon
import Testing

extension KernelTests {
    /// The rewindable MTP history against the bounded reference FP8 cache.
    struct NemotronSuperFP8KVHistoryTests {
        private func rows(_ count: Int, seed: UInt64, heads: Int = 1, dimension: Int = 4)
            -> MLXArray
        {
            MLXRandom.normal([1, heads, count, dimension], key: MLXRandom.key(seed)) * 40
        }

        private func exact(_ lhs: [MLXArray], _ rhs: [MLXArray]) {
            #expect(lhs.count == rhs.count)
            for (a, b) in zip(lhs, rhs) {
                #expect(a.shape == b.shape && a.dtype == b.dtype)
                #expect(arrayEqual(a, b, equalNAN: true).item(Bool.self))
            }
        }

        @Test(arguments: [DType.float32, .bfloat16, .float16])
        func matchesBoundedCacheAcrossAppendsAndBlocks(dtype: DType) throws {
            let reference = try NemotronSuperFP8KVCache(
                heads: 2, headDimension: 4, capacityTokens: 600, dtype: dtype)
            let history = try NemotronSuperFP8KVHistory(
                heads: 2, headDimension: 4, capacityTokens: 600, dtype: dtype)
            var seed: UInt64 = 1
            for count in [1, 3, 250, 1, 7, 64, 1] {
                var keys = rows(count, seed: seed, heads: 2).asType(dtype)
                if seed == 3 { keys[0..., 0..., 0, 0] = MLXArray(Float.nan).asType(dtype) }
                let values = rows(count, seed: seed + 50, heads: 2).asType(dtype)
                seed += 1
                let expected = try reference.checkedUpdate(keys: keys, values: values)
                let actual = try history.checkedUpdate(keys: keys, values: values)
                exact([actual.0, actual.1], [expected.0, expected.1])
                exact(history.committedCodes(), reference.innerState())
                #expect(history.offset == reference.offset)
                // The whole capacity from the first write on; appends never grow it.
                #expect(history.allocatedTokens == 600)
            }
            #expect(history.offset == 327)
        }

        @Test func rewindOverwritesAbandonedRowsExactly() throws {
            let history = try NemotronSuperFP8KVHistory(
                heads: 1, headDimension: 4, capacityTokens: 16, dtype: .float32)
            let fresh = try NemotronSuperFP8KVHistory(
                heads: 1, headDimension: 4, capacityTokens: 16, dtype: .float32)
            let prefix = rows(6, seed: 1)
            _ = try history.checkedUpdate(keys: prefix, values: prefix + 1)
            _ = try history.checkedUpdate(keys: rows(4, seed: 2), values: rows(4, seed: 3))
            try history.rewind(to: 6)
            #expect(throws: NemotronSuperFP8KVHistory.Failure.rewind) {
                try history.rewind(to: 7)
            }
            let replacement = rows(4, seed: 4)
            let replayed = try history.checkedUpdate(keys: replacement, values: replacement * 2)
            _ = try fresh.checkedUpdate(keys: prefix, values: prefix + 1)
            let expected = try fresh.checkedUpdate(keys: replacement, values: replacement * 2)
            exact([replayed.0, replayed.1], [expected.0, expected.1])
            exact(history.committedCodes(), fresh.committedCodes())
            #expect(history.trim(3) == 3 && history.offset == 7)
            #expect(history.trim(100) == 7 && history.committedCodes().isEmpty)
        }

        @Test func restoreCopyAndRejectedAppendsPreserveState() throws {
            let history = try NemotronSuperFP8KVHistory(
                heads: 1, headDimension: 4, capacityTokens: 4, dtype: .float32)
            let first = rows(3, seed: 9)
            _ = try history.checkedUpdate(keys: first, values: first)
            let before = history.committedCodes()
            #expect(throws: NemotronSuperFP8KVHistory.Failure.capacity) {
                try history.checkedUpdate(keys: rows(2, seed: 1), values: rows(2, seed: 2))
            }
            #expect(throws: NemotronSuperFP8KVHistory.Failure.dtype) {
                try history.checkedUpdate(
                    keys: rows(1, seed: 1).asType(.bfloat16),
                    values: rows(1, seed: 1).asType(.bfloat16))
            }
            #expect(throws: NemotronSuperFP8KVHistory.Failure.shape) {
                try history.checkedUpdate(
                    keys: rows(1, seed: 1, dimension: 2), values: rows(1, seed: 1, dimension: 2))
            }
            #expect(history.offset == 3)
            exact(history.committedCodes(), before)
            let copied = try #require(history.copy() as? NemotronSuperFP8KVHistory)
            copied.trim(2)
            _ = try copied.checkedUpdate(keys: rows(1, seed: 5), values: rows(1, seed: 6))
            exact(history.committedCodes(), before)
            let restored = try NemotronSuperFP8KVHistory(
                heads: 1, headDimension: 4, capacityTokens: 4, dtype: .float32)
            try restored.restore(before)
            exact(restored.committedCodes(), before)
            let tail = rows(1, seed: 7)
            exact(
                [try restored.checkedUpdate(keys: tail, values: tail).0],
                [try history.checkedUpdate(keys: tail, values: tail).0])
            #expect(throws: NemotronSuperFP8KVHistory.Failure.state) {
                try restored.restore([before[0]])
            }
            #expect(restored.offset == 4)
        }

        /// Measurement only (`DARKBLOOM_SUPER_HISTORY_PEAK=1`, run alone): MLX
        /// peak bytes above the resident history of a chained draft round at
        /// the Super head geometry (2 KV heads x 128, 32 query heads, BF16
        /// mirror), with the round evaluated once (as the engine does) or per
        /// step, and of decoding the codes per step instead of a mirror.
        @Test(
            .enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_SUPER_HISTORY_PEAK"] == "1")
        )
        func draftRoundPeakMeasurement() throws {
            let rows = 65_536
            let capacity = rows + 8
            let scale = Float(1) / Float(128).squareRoot()
            func codes(_ seed: UInt64) -> MLXArray {
                (MLXRandom.randInt(0 ..< 0x38, [1, 2, rows, 128], key: MLXRandom.key(seed)))
                    .asType(.uint8)
            }
            func prepared() throws -> NemotronSuperFP8KVHistory {
                let history = try NemotronSuperFP8KVHistory(
                    heads: 2, headDimension: 128, capacityTokens: capacity)
                try history.restore([codes(1), codes(2)])
                eval(history.innerState())
                return history
            }
            func measure(_ label: String, _ body: () throws -> Void) rethrows {
                Memory.clearCache()
                let before = Memory.activeMemory
                Memory.peakMemory = 0
                try body()
                let peak = Memory.peakMemory - before
                print(
                    "SUPER_HISTORY_PEAK \(label) rows=\(rows) peak_above_resident=\(peak) "
                        + "per_row=\(Double(peak) / Double(rows))")
            }
            let q0 = MLXRandom.normal([1, 32, 1, 128], key: MLXRandom.key(3)).asType(.bfloat16)
            func step(_ history: NemotronSuperFP8KVHistory, _ q: MLXArray) -> MLXArray {
                let row = q[0..., 0 ..< 2, 0..., 0...]
                let (k, v) = history.update(keys: row, values: row * 0.5)
                return MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: .none)
            }
            for depth in [1, 3] {
                let once = try prepared()
                measure("mirror_round_eval_once depth=\(depth)") {
                    var q = q0
                    for _ in 0 ..< depth { q = step(once, q) }
                    eval([q] + once.innerState())
                }
                let each = try prepared()
                measure("mirror_eval_per_step depth=\(depth)") {
                    var q = q0
                    for _ in 0 ..< depth {
                        q = step(each, q)
                        eval([q] + each.innerState())
                    }
                }
            }
            // No mirror: codes only, decoded per step through a 256-entry table.
            let table = try NemotronSuperFP8KVCodec.decode(
                MLXArray((0 ..< 256).map { UInt8($0) }), dtype: .bfloat16)
            for depth in [1, 3] {
                let keys = MLXArray.zeros([1, 2, capacity, 128], dtype: .uint8)
                let values = MLXArray.zeros([1, 2, capacity, 128], dtype: .uint8)
                keys[.ellipsis, 0 ..< rows, 0...] = codes(1)
                values[.ellipsis, 0 ..< rows, 0...] = codes(2)
                eval(keys, values)
                try measure("codes_lut_round_eval_once depth=\(depth)") {
                    var q = q0
                    var offset = rows
                    for _ in 0 ..< depth {
                        let row = q[0..., 0 ..< 2, 0..., 0...]
                        keys[.ellipsis, offset ..< (offset + 1), 0...] =
                            try NemotronSuperFP8KVCodec.encode(row)
                        values[.ellipsis, offset ..< (offset + 1), 0...] =
                            try NemotronSuperFP8KVCodec.encode(row * 0.5)
                        offset += 1
                        let k = table.take(keys[.ellipsis, ..<offset, 0...], axis: 0)
                        let v = table.take(values[.ellipsis, ..<offset, 0...], axis: 0)
                        q = MLXFast.scaledDotProductAttention(
                            queries: q, keys: k, values: v, scale: scale, mask: .none)
                    }
                    eval([q, keys, values])
                }
            }
        }

        @Test func boundsAndAccounting() throws {
            #expect(throws: NemotronSuperFP8KVHistory.Failure.configuration) {
                try NemotronSuperFP8KVHistory(heads: 0, headDimension: 4, capacityTokens: 4)
            }
            #expect(throws: NemotronSuperFP8KVHistory.Failure.capacity) {
                try NemotronSuperFP8KVHistory(
                    heads: 2, headDimension: 128, capacityTokens: (1 << 20) + 1)
            }
            // Real Super head geometry: 2 KV heads x 128, BF16 mirror.
            #expect(
                try NemotronSuperFP8KVHistory.bytesPerToken(
                    heads: 2, headDimension: 128, dtype: .bfloat16) == 1536)
            let history = try NemotronSuperFP8KVHistory(
                heads: 2, headDimension: 128, capacityTokens: 16_386)
            #expect(history.maximumResidentBytes == 1536 * 16_386)
            #expect(history.allocatedTokens == 0)
            #expect(history.maximumAppendRows == 4096)
            #expect(history.maxSize == nil && history.isTrimmable)
            // Codec temporaries only: an append allocates no history rows.
            #expect(try history.appendAllowanceBytes(rows: 1) == 256 * 64)
            #expect(throws: NemotronSuperFP8KVHistory.Failure.capacity) {
                try history.appendAllowanceBytes(rows: 4097)
            }
            if case .none = try history.checkedMask(n: 1, windowSize: nil, returnArray: false) {
            } else {
                Issue.record("a single-row draft query must not build a mask")
            }
            #expect(throws: NemotronSuperFP8KVHistory.Failure.capacity) {
                try history.checkedMask(n: 2, windowSize: 4, returnArray: false)
            }
        }
    }
}
