// CBv2MTPKVStagingTests.swift
//
// W-B (Gemma-4 MTP integration) tests: staged speculative KV writes and the
// rectangular [B > 1, L > 1] attention relaxation.
//
//  - Staged == plain equivalence on the windowed ring (THE regression test
//    for the sharpest MTP edge): begin/update/rollback/commit must be
//    value-exactly a plain update of only the confirmed tokens, across
//    pre-wrap and post-wrap fills, staged counts, and rollbacks including
//    full (cancelled-row) rollback.
//  - Mid-flight staged views: borrowableViews() returns the staged update's
//    views; rollback invalidates them down to the confirmed content.
//  - Full/Quantized/Paged speculative eligibility flags and value-exact
//    write-n/rollback-m/continue rounds.
//  - Rectangular verify parity: one [B, L=3] updateAndAttend equals running
//    each row alone as [1, 3] (full, windowed with kL > window, sinks,
//    KV-shared borrowing), at the attention layer and through a full
//    TinyTestModel + CBv2LayerCacheBank forward.
//  - Compiled-decode interaction guard: the violation predicate backing the
//    compiledStorage/noteCompiledAdvance preconditions tracks the staged
//    lifecycle exactly.
//
// No model weights required — everything runs on tiny random tensors.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

// MARK: - Helpers

/// K/V whose element [0, h, t, d] encodes the token's absolute position (plus
/// small head/dim offsets) so storage order is exactly checkable.
private func positionCoded(from: Int, count: Int, kvHeads: Int = 2, headDim: Int = 4) -> MLXArray {
    let positions = MLXArray((0 ..< count).map { Float(from + $0) }).reshaped([1, 1, count, 1])
    let heads = MLXArray((0 ..< kvHeads).map { Float($0) * 0.25 }).reshaped([1, kvHeads, 1, 1])
    let dims = MLXArray((0 ..< headDim).map { Float($0) * 0.001 }).reshaped([1, 1, 1, headDim])
    return broadcast(positions + heads + dims, to: [1, kvHeads, count, headDim])
}

private func expectedPositions(_ positions: [Int], kvHeads: Int = 2, headDim: Int = 4) -> MLXArray {
    guard !positions.isEmpty else {
        return MLXArray.zeros([1, kvHeads, 0, headDim])
    }
    let pos = MLXArray(positions.map { Float($0) }).reshaped([1, 1, positions.count, 1])
    let heads = MLXArray((0 ..< kvHeads).map { Float($0) * 0.25 }).reshaped([1, kvHeads, 1, 1])
    let dims = MLXArray((0 ..< headDim).map { Float($0) * 0.001 }).reshaped([1, 1, 1, headDim])
    return broadcast(pos + heads + dims, to: [1, kvHeads, positions.count, headDim])
}

/// Bitwise value equality (tolerance 0) — the staging design promises
/// VALUE-EXACT state, not approximately-equal state.
private func expectExact(
    _ a: MLXArray, _ b: MLXArray, _ label: String = "",
    sourceLocation: Testing.SourceLocation = #_sourceLocation
) {
    #expect(
        a.shape == b.shape, "shape mismatch \(label): \(a.shape) vs \(b.shape)",
        sourceLocation: sourceLocation)
    guard a.shape == b.shape else { return }
    guard a.size > 0 else { return }
    #expect(
        arrayEqual(a, b).item(Bool.self), "values differ \(label)",
        sourceLocation: sourceLocation)
}

private func expectClose(
    _ a: MLXArray, _ b: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-6,
    _ label: String = "", sourceLocation: Testing.SourceLocation = #_sourceLocation
) {
    #expect(
        a.shape == b.shape, "shape mismatch \(label): \(a.shape) vs \(b.shape)",
        sourceLocation: sourceLocation)
    guard a.shape == b.shape else { return }
    let close = allClose(a, b, rtol: rtol, atol: atol).item(Bool.self)
    #expect(close, "values differ \(label)", sourceLocation: sourceLocation)
}

/// Decode-fill a windowed row with position-coded tokens [0, count).
private func fillWindowed(_ row: CBv2WindowedSequenceKV, count: Int) {
    for t in 0 ..< count {
        _ = row.update(
            keys: positionCoded(from: t, count: 1),
            values: positionCoded(from: 5000 + t, count: 1))
    }
}

// MARK: - (a) Staged == plain equivalence

@Suite("CBv2MTPKVStaging: staged == plain equivalence", .serialized)
struct CBv2MTPKVStagingEquivalenceTests {

    /// One geometry: base fill via plain decode updates, then
    /// (B) begin + update(n) + rollback(m) + commit vs
    /// (A) a plain update of only the confirmed n-m tokens.
    private func runCase(
        window: Int, baseFill: Int, n: Int, m: Int,
        sourceLocation: Testing.SourceLocation = #_sourceLocation
    ) {
        let label = "w=\(window) base=\(baseFill) n=\(n) m=\(m)"
        let confirmed = n - m
        let chunkKeys = positionCoded(from: baseFill, count: n)
        let chunkValues = positionCoded(from: 5000 + baseFill, count: n)

        // (B) staged round.
        let stagedRow = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)
        fillWindowed(stagedRow, count: baseFill)
        #expect(stagedRow.supportsSpeculativeWrites, sourceLocation: sourceLocation)
        stagedRow.beginSpeculativeWrite()
        let (stagedK, stagedV) = stagedRow.update(keys: chunkKeys, values: chunkValues)
        #expect(
            stagedRow.absoluteOffset == baseFill + n, "staged offset \(label)",
            sourceLocation: sourceLocation)

        // The staged update's returned views equal a PLAIN update's views
        // (decode return for n == 1, history ++ chunk for n > 1).
        let viewReference = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)
        fillWindowed(viewReference, count: baseFill)
        let (plainK, plainV) = viewReference.update(keys: chunkKeys, values: chunkValues)
        expectExact(stagedK, plainK, "returned keys \(label)", sourceLocation: sourceLocation)
        expectExact(stagedV, plainV, "returned values \(label)", sourceLocation: sourceLocation)

        stagedRow.rollback(m)
        stagedRow.commitSpeculativeWrite()

        // (A) plain updates of the confirmed tokens only.
        let plainRow = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)
        fillWindowed(plainRow, count: baseFill)
        if confirmed > 0 {
            _ = plainRow.update(
                keys: positionCoded(from: baseFill, count: confirmed),
                values: positionCoded(from: 5000 + baseFill, count: confirmed))
        }

        #expect(
            stagedRow.absoluteOffset == plainRow.absoluteOffset,
            "post-commit offset \(label)", sourceLocation: sourceLocation)
        #expect(
            stagedRow.retainedCount == plainRow.retainedCount,
            "post-commit retained \(label)", sourceLocation: sourceLocation)
        let stagedSnap = stagedRow.snapshot()
        let plainSnap = plainRow.snapshot()
        #expect(
            stagedSnap.offset == plainSnap.offset, "snapshot offset \(label)",
            sourceLocation: sourceLocation)
        expectExact(
            stagedSnap.keys, plainSnap.keys, "snapshot keys \(label)",
            sourceLocation: sourceLocation)
        expectExact(
            stagedSnap.values, plainSnap.values, "snapshot values \(label)",
            sourceLocation: sourceLocation)

        // Next decode update behaves identically on both rows.
        let nextPosition = baseFill + confirmed
        let nextKeys = positionCoded(from: nextPosition, count: 1)
        let nextValues = positionCoded(from: 5000 + nextPosition, count: 1)
        let (stagedNextK, stagedNextV) = stagedRow.update(keys: nextKeys, values: nextValues)
        let (plainNextK, plainNextV) = plainRow.update(keys: nextKeys, values: nextValues)
        expectExact(
            stagedNextK, plainNextK, "next-decode keys \(label)", sourceLocation: sourceLocation)
        expectExact(
            stagedNextV, plainNextV, "next-decode values \(label)",
            sourceLocation: sourceLocation)
        expectExact(
            stagedRow.snapshot().keys, plainRow.snapshot().keys,
            "post-next-decode snapshot \(label)", sourceLocation: sourceLocation)
    }

    @Test func stagedEqualsPlainAcrossGeometries() {
        for window in [4, 8, 16] {
            // Pre-wrap AND post-wrap base fills.
            for baseFill in [0, window - 2, window - 1, window, window + 3, 3 * window + 1] {
                for n in [1, 2, 4] {
                    for m in 0 ... n {  // includes m == n: cancelled row
                        runCase(window: window, baseFill: baseFill, n: n, m: m)
                    }
                }
            }
        }
    }

    /// Pin the n == 1 staged-return equivalence explicitly at the exact ring
    /// states where plain decode and history++token could diverge: a full
    /// ring (plain returns `window` entries [oldest+1, offset+1)) and a
    /// below-full ring (plain returns everything).
    @Test func singleTokenStagedReturnEqualsPlainDecodeReturn() {
        for (window, baseFill) in [(4, 4), (4, 9), (8, 3), (8, 8)] {
            runCase(window: window, baseFill: baseFill, n: 1, m: 0)
            runCase(window: window, baseFill: baseFill, n: 1, m: 1)
        }
    }
}

// MARK: - (b) Mid-flight staged views

@Suite("CBv2MTPKVStaging: mid-flight staged views", .serialized)
struct CBv2MTPKVStagingMidFlightTests {

    @Test func borrowableViewsMatchStagedViewsAndRollbackInvalidates() {
        let window = 4
        let row = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)
        fillWindowed(row, count: 6)  // wrapped: ring holds [2, 6)
        let ringOnlyBytes = row.byteCount

        row.beginSpeculativeWrite()
        let (stagedK, stagedV) = row.update(
            keys: positionCoded(from: 6, count: 3),
            values: positionCoded(from: 5006, count: 3))
        // history = min(retained 4, window-1 = 3) = 3 → [3, 6) ++ chunk [6, 9).
        expectExact(stagedK, expectedPositions([3, 4, 5, 6, 7, 8]), "staged keys")

        // borrowableViews() during the staged window IS the staged views.
        let mid = row.borrowableViews()
        expectExact(mid.keys, stagedK, "mid-flight borrowable keys")
        expectExact(mid.values, stagedV, "mid-flight borrowable values")

        // Staged tensors are physically held (byteCount accounting).
        #expect(row.byteCount > ringOnlyBytes)

        // Rollback invalidates the staged views: the fallback exposes the
        // confirmed content only (ring [2, 6) ++ confirmed staged [6, 8)).
        row.rollback(1)
        let after = row.borrowableViews()
        expectExact(after.keys, expectedPositions([2, 3, 4, 5, 6, 7]), "post-rollback keys")
        expectExact(
            after.values,
            expectedPositions([5002, 5003, 5004, 5005, 5006, 5007]), "post-rollback values")

        row.commitSpeculativeWrite()
        #expect(row.byteCount == ringOnlyBytes)
        expectExact(row.snapshot().keys, expectedPositions([4, 5, 6, 7]), "post-commit ring")
    }
}

// MARK: - (c) Full / Quantized speculative rounds

@Suite("CBv2MTPKVStaging: full & quantized rounds", .serialized)
struct CBv2MTPKVStagingFullQuantizedTests {

    @Test func fullSequenceRoundEqualsPlain() {
        let speculated = CBv2FullSequenceKV(
            promptLength: 5, maxLength: 64, kvHeads: 2, headDim: 4)
        #expect(speculated.supportsSpeculativeWrites)
        _ = speculated.update(
            keys: positionCoded(from: 0, count: 5), values: positionCoded(from: 5000, count: 5))
        speculated.beginSpeculativeWrite()
        _ = speculated.update(
            keys: positionCoded(from: 5, count: 3), values: positionCoded(from: 5005, count: 3))
        speculated.rollback(2)
        speculated.commitSpeculativeWrite()
        _ = speculated.update(
            keys: positionCoded(from: 60, count: 2), values: positionCoded(from: 5060, count: 2))

        let plain = CBv2FullSequenceKV(promptLength: 5, maxLength: 64, kvHeads: 2, headDim: 4)
        _ = plain.update(
            keys: positionCoded(from: 0, count: 5), values: positionCoded(from: 5000, count: 5))
        _ = plain.update(
            keys: positionCoded(from: 5, count: 1), values: positionCoded(from: 5005, count: 1))
        _ = plain.update(
            keys: positionCoded(from: 60, count: 2), values: positionCoded(from: 5060, count: 2))

        #expect(speculated.absoluteOffset == plain.absoluteOffset)
        expectExact(speculated.snapshot().keys, plain.snapshot().keys, "full keys")
        expectExact(speculated.snapshot().values, plain.snapshot().values, "full values")
        expectExact(
            speculated.snapshot().keys,
            expectedPositions([0, 1, 2, 3, 4, 5, 60, 61]), "full content")
    }

    /// The QuantizedSequenceKV verdict: quantization groups span the headDim
    /// axis only, so writing n tokens then rolling back m leaves the
    /// confirmed prefix's quantized content BIT-identical to never having
    /// written the rejected suffix.
    @Test func quantizedRoundEqualsPlainBitExactly() {
        MLXRandom.seed(0xDEAD)
        let base = MLXRandom.normal([1, 2, 5, 64]).asType(.float16)
        let chunk = MLXRandom.normal([1, 2, 3, 64]).asType(.float16)
        let next = MLXRandom.normal([1, 2, 2, 64]).asType(.float16)

        func makeRow() -> CBv2QuantizedSequenceKV {
            CBv2QuantizedSequenceKV(
                promptLength: 5, maxLength: 64, kvHeads: 2, headDim: 64,
                groupSize: 64, bits: 4)
        }

        let speculated = makeRow()
        #expect(speculated.supportsSpeculativeWrites)
        _ = speculated.update(keys: base, values: base)
        speculated.beginSpeculativeWrite()
        _ = speculated.update(keys: chunk, values: chunk)
        speculated.rollback(2)
        speculated.commitSpeculativeWrite()
        _ = speculated.update(keys: next, values: next)

        let plain = makeRow()
        _ = plain.update(keys: base, values: base)
        _ = plain.update(
            keys: chunk[.ellipsis, ..<1, 0...], values: chunk[.ellipsis, ..<1, 0...])
        _ = plain.update(keys: next, values: next)

        #expect(speculated.absoluteOffset == plain.absoluteOffset)
        // Dequantized snapshots must match BITWISE — same quantization grid.
        expectExact(speculated.snapshot().keys, plain.snapshot().keys, "quantized keys")
        expectExact(speculated.snapshot().values, plain.snapshot().values, "quantized values")
    }
}

// MARK: - (d) Paged eligibility flags

@Suite("CBv2MTPKVStaging: paged flags")
struct CBv2MTPKVStagingPagedFlagTests {

    @Test func fullRowsEligibleWindowedRowsNot() throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .slidingWindow(32), headDim: 64, kvHeads: 2, queryHeads: 4),
        ]
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 8 << 20, maxPrefillChunk: 64,
                nominalMaxSequenceLength: 1024))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 256)
        #expect(state[0]?.supportsSpeculativeWrites == true)
        #expect(state[1]?.supportsSpeculativeWrites == false)
        backend.release(state)
    }
}

// MARK: - (e) Rectangular attention parity

@Suite("CBv2MTPKVStaging: rectangular attention parity", .serialized)
struct CBv2MTPKVStagingAttentionParityTests {

    private let queryHeads = 4
    private let kvHeads = 2
    private let headDim = 16
    private let scale: Float = 0.25
    private let verifyTokens = 3
    /// Row lengths chosen so a window-8 layer sees BOTH mask modes in one
    /// rectangular call: 5 → kL = 8 ≤ window (causal), 12 → retained 8,
    /// history 7, kL = 10 > window (array mask).
    private let lengths = [5, 12]

    private func makeRow(kind: CBv2LayerKind, length: Int) -> CBv2SequenceKV {
        switch kind.attention {
        case .full:
            return CBv2FullSequenceKV(
                promptLength: length, maxLength: 256, kvHeads: kvHeads, headDim: headDim)
        case .slidingWindow(let window):
            return CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)
        }
    }

    /// Two identical row sets (batch + solo) fed the same K/V content.
    private func makeRowPairs(kind: CBv2LayerKind) -> (batch: [CBv2SequenceKV], solo: [CBv2SequenceKV]) {
        var batch: [CBv2SequenceKV] = []
        var solo: [CBv2SequenceKV] = []
        for (index, length) in lengths.enumerated() {
            MLXRandom.seed(UInt64(9100 + index))
            let keys = MLXRandom.normal([1, kvHeads, length, headDim]).asType(.float16)
            let values = MLXRandom.normal([1, kvHeads, length, headDim]).asType(.float16)
            let batchRow = makeRow(kind: kind, length: length)
            _ = batchRow.update(keys: keys, values: values)
            let soloRow = makeRow(kind: kind, length: length)
            _ = soloRow.update(keys: keys, values: values)
            batch.append(batchRow)
            solo.append(soloRow)
        }
        return (batch, solo)
    }

    private func verifyBatchQKV() -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        MLXRandom.seed(31337)
        let B = lengths.count
        return (
            q: MLXRandom.normal([B, queryHeads, verifyTokens, headDim]).asType(.float16),
            k: MLXRandom.normal([B, kvHeads, verifyTokens, headDim]).asType(.float16),
            v: MLXRandom.normal([B, kvHeads, verifyTokens, headDim]).asType(.float16)
        )
    }

    private func runParity(kind: CBv2LayerKind, sinks: MLXArray?, staged: Bool = false) {
        let (batchRows, soloRows) = makeRowPairs(kind: kind)
        let (q, k, v) = verifyBatchQKV()

        if staged {
            for row in batchRows + soloRows { row.beginSpeculativeWrite() }
        }

        let batchCache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: batchRows)
        let out = batchCache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: sinks)
        #expect(out.shape == [lengths.count, queryHeads, verifyTokens, headDim])

        for (index, soloRow) in soloRows.enumerated() {
            let soloCache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: [soloRow])
            let soloOut = soloCache.updateAndAttend(
                queries: q[index ..< (index + 1)],
                keys: k[index ..< (index + 1)],
                values: v[index ..< (index + 1)],
                scale: scale, sinks: sinks)
            expectExact(
                out[index ..< (index + 1)], soloOut,
                "row \(index) (\(kind.attention), sinks: \(sinks != nil), staged: \(staged))")
        }

        if staged {
            // Finalize like an MTP round: reject the last token, commit,
            // and require identical post-round state per row.
            for row in batchRows + soloRows {
                row.rollback(1)
                row.commitSpeculativeWrite()
            }
            for (index, soloRow) in soloRows.enumerated() {
                expectExact(
                    batchRows[index].snapshot().keys, soloRow.snapshot().keys,
                    "post-round snapshot row \(index)")
            }
        }
    }

    @Test func fullAttentionRectangularParity() {
        runParity(
            kind: CBv2LayerKind(
                attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads),
            sinks: nil)
    }

    @Test func windowedRectangularParityIncludingArrayMask() {
        runParity(
            kind: CBv2LayerKind(
                attention: .slidingWindow(8), headDim: headDim, kvHeads: kvHeads,
                queryHeads: queryHeads),
            sinks: nil)
    }

    @Test func windowedWithSinksRectangularParity() {
        MLXRandom.seed(4321)
        let sinks = MLXRandom.normal([queryHeads]).asType(.float16)
        runParity(
            kind: CBv2LayerKind(
                attention: .slidingWindow(8), hasSinks: true, headDim: headDim,
                kvHeads: kvHeads, queryHeads: queryHeads),
            sinks: sinks)
    }

    /// The real MTP shape: windowed rows under an armed speculative write
    /// take the rectangular verify forward, then rollback + commit.
    @Test func stagedWindowedRectangularVerifyMatchesSolo() {
        runParity(
            kind: CBv2LayerKind(
                attention: .slidingWindow(8), headDim: headDim, kvHeads: kvHeads,
                queryHeads: queryHeads),
            sinks: nil, staged: true)
    }

    @Test func kvSharedBorrowRectangularParity() {
        let sourceKind = CBv2LayerKind(
            attention: .slidingWindow(8), headDim: headDim, kvHeads: kvHeads,
            queryHeads: queryHeads)
        let sharedKind = CBv2LayerKind(
            attention: .slidingWindow(8), sharesKVWithLayer: 0, headDim: headDim,
            kvHeads: kvHeads, queryHeads: queryHeads)
        let (batchRows, soloRows) = makeRowPairs(kind: sourceKind)
        let (q, k, v) = verifyBatchQKV()

        // The source layer's rectangular update installs each row's
        // borrowable chunk views (pre-eviction history ++ chunk).
        let batchSource = CBv2LayerCache(layerIndex: 0, kind: sourceKind, rows: batchRows)
        _ = batchSource.updateAndAttend(queries: q, keys: k, values: v, scale: scale, sinks: nil)

        MLXRandom.seed(7331)
        let borrowQ = MLXRandom.normal(
            [lengths.count, queryHeads, verifyTokens, headDim]
        ).asType(.float16)
        let shared = CBv2LayerCache(layerIndex: 5, kind: sharedKind)
        let out = shared.attendBorrowing(
            source: batchSource, queries: borrowQ, scale: scale, sinks: nil)
        #expect(out.shape == [lengths.count, queryHeads, verifyTokens, headDim])

        for (index, soloRow) in soloRows.enumerated() {
            let soloSource = CBv2LayerCache(layerIndex: 0, kind: sourceKind, rows: [soloRow])
            _ = soloSource.updateAndAttend(
                queries: q[index ..< (index + 1)],
                keys: k[index ..< (index + 1)],
                values: v[index ..< (index + 1)],
                scale: scale, sinks: nil)
            let soloShared = CBv2LayerCache(layerIndex: 5, kind: sharedKind)
            let soloOut = soloShared.attendBorrowing(
                source: soloSource, queries: borrowQ[index ..< (index + 1)],
                scale: scale, sinks: nil)
            expectExact(out[index ..< (index + 1)], soloOut, "borrowed row \(index)")
        }
    }

    /// End-to-end: a TinyTestModel (full + windowed + KV-shared + sinks
    /// layers) over a CBv2LayerCacheBank, rows prefilled at different
    /// lengths, one rectangular [B, 3] forward vs each row alone as [1, 3]
    /// on cloned state. Model-level matmuls batch over B, so this pins
    /// close-parity (the attention dispatch itself is pinned bit-exact by
    /// the suites above).
    @Test func tinyModelRectangularForwardMatchesSoloRows() throws {
        let model = TinyTestModel.make(seed: 0xB00, withSinks: true, withKVSharing: true)
        let kinds = model.layerKinds
        let promptLengths = [9, 23]  // 23 > window 16: wrapped windowed rows

        func buildStates(_ backend: CBv2ContiguousKVBackend, bank: CBv2LayerCacheBank)
            throws -> [[CBv2SequenceKV?]]
        {
            var states: [[CBv2SequenceKV?]] = []
            for (index, length) in promptLengths.enumerated() {
                let state = try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: length, maxLength: 64)
                let prompt = makePromptTokens(length: length, seed: UInt64(41 + index))
                let tokens = MLXArray(prompt.map { Int32($0) }).reshaped(1, length)
                _ = model.forward(tokens: tokens, caches: bank.layerCaches(rowStates: [state]))
                states.append(state)
            }
            return states
        }

        let backendA = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 26, kvDType: .float32))
        let bankA = CBv2LayerCacheBank(layerKinds: kinds)
        let statesA = try buildStates(backendA, bank: bankA)

        let backendB = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 26, kvDType: .float32))
        let bankB = CBv2LayerCacheBank(layerKinds: kinds)
        let statesB = try buildStates(backendB, bank: bankB)

        // Per-row verify tokens, stacked into one rectangular [B, 3] step.
        let rowTokens = promptLengths.indices.map { index in
            makePromptTokens(length: 3, seed: UInt64(1000 + index)).map { Int32($0) }
        }
        let batchTokens = MLXArray(rowTokens.flatMap { $0 }).reshaped(promptLengths.count, 3)
        let batchLogits = model.forward(
            tokens: batchTokens, caches: bankA.layerCaches(rowStates: statesA))
        #expect(batchLogits.dim(0) == promptLengths.count)
        #expect(batchLogits.dim(1) == 3)

        for index in promptLengths.indices {
            let soloTokens = MLXArray(rowTokens[index]).reshaped(1, 3)
            let soloLogits = model.forward(
                tokens: soloTokens, caches: bankB.layerCaches(rowStates: [statesB[index]]))
            expectClose(
                batchLogits[index ..< (index + 1)], soloLogits,
                rtol: 1e-4, atol: 1e-5, "model logits row \(index)")
        }

        backendA.release(statesA[0])
        backendA.release(statesA[1])
        backendB.release(statesB[0])
        backendB.release(statesB[1])
    }
}

// MARK: - (f) Compiled-decode interaction guard

@Suite("CBv2MTPKVStaging: compiled-decode guard")
struct CBv2MTPKVStagingCompiledGuardTests {

    /// `compiledStorage` / `noteCompiledAdvance` preconditionFailure on
    /// exactly this predicate (untestable in-process); pin its lifecycle:
    /// non-nil from arm through commit, nil before and after.
    @Test func speculativePendingViolationTracksStagedLifecycle() {
        let row = CBv2WindowedSequenceKV(window: 8, kvHeads: 2, headDim: 4)
        #expect(row.cbv2SpeculativePendingViolation() == nil)

        row.beginSpeculativeWrite()
        #expect(row.cbv2SpeculativePendingViolation() != nil)  // armed

        _ = row.update(
            keys: positionCoded(from: 0, count: 2), values: positionCoded(from: 5000, count: 2))
        #expect(row.cbv2SpeculativePendingViolation() != nil)  // staged

        row.rollback(1)
        #expect(row.cbv2SpeculativePendingViolation() != nil)  // still staged

        row.commitSpeculativeWrite()
        #expect(row.cbv2SpeculativePendingViolation() == nil)
        expectExact(row.snapshot().keys, expectedPositions([0]), "committed token survives")

        // Once clear, the compiled bridge works again.
        #expect(row.compiledStorage(keysDType: .float32, valuesDType: .float32) != nil)
        row.noteCompiledAdvance()
        #expect(row.absoluteOffset == 2)
    }

    @Test func beginThenCommitWithoutUpdateIsCleanNoOp() {
        let row = CBv2WindowedSequenceKV(window: 8, kvHeads: 2, headDim: 4)
        fillWindowed(row, count: 3)
        row.beginSpeculativeWrite()
        row.commitSpeculativeWrite()  // disarms without an update
        #expect(row.cbv2SpeculativePendingViolation() == nil)
        let (k, _) = row.update(
            keys: positionCoded(from: 3, count: 1), values: positionCoded(from: 5003, count: 1))
        expectExact(k, expectedPositions([0, 1, 2, 3]), "plain update after disarm")
    }
}
