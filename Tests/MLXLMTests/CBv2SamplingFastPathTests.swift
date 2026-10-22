// CBv2SamplingFastPathTests.swift
//
// Exactness of the CBv2 sampling-stage fast paths against the pre-change
// implementations (kept verbatim below as references): the stage-(4)
// filter plans of `LogitsPipelineV2`. Every comparison is bit for bit (NaN
// payloads included).

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

// MARK: - Pre-change references (verbatim)

/// `LogitsPipelineV2.applyTopKTopPMinP` before the filter plans.
private func referenceFilter(
    _ logits: MLXArray, topK: MLXArray, topP: MLXArray, minP: MLXArray
) -> MLXArray {
    let vocab = logits.dim(-1)
    let sortedIndices = argSort(-logits, axis: -1)
    let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
    let probs = softmax(sortedLogits, axis: -1)
    let cumulative = cumsum(probs, axis: -1)
    let ranks = MLXArray(Array(0 ..< Int32(vocab))).reshaped([1, vocab])
    var keep = ranks .< topK
    keep = keep .&& ((cumulative - probs) .<= topP)
    let maxProb = probs[0..., ..<1]
    keep = keep .&& (probs .>= (minP * maxProb))
    let masked = which(keep, sortedLogits, MLXArray(-Float.infinity))
    let inverse = putAlong(
        MLXArray.zeros(like: sortedIndices),
        sortedIndices,
        values: broadcast(ranks.asType(sortedIndices.dtype), to: sortedIndices.shape),
        axis: -1
    )
    return takeAlong(masked, inverse, axis: -1)
}

// MARK: - Helpers

/// Deterministic key stream for device-side test data.
private struct Mix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func key() -> MLXArray { MLXRandom.key(next()) }
}

/// Row flavours that stress ranks, ties and the NaN-softmax rows.
private enum RowKind: CaseIterable {
    case continuous, coarseTies, allEqual, maskedMostly, withNaN, withPosInf, allNegInf
}

/// One [1, vocab] float32 row, built on device.
private func makeRow(_ kind: RowKind, vocab: Int, rng: inout Mix64) -> MLXArray {
    let shape = [1, vocab]
    func uniform(_ low: Float, _ high: Float) -> MLXArray {
        MLXRandom.uniform(low: low, high: high, shape, type: Float.self, key: rng.key())
    }
    func spike(_ value: Float) -> MLXArray {
        let position = Int32(rng.next() % UInt64(vocab))
        let columns = MLXArray(Array(0 ..< Int32(vocab))).reshaped(shape)
        return which(columns .== position, MLXArray(value), uniform(-8, 8))
    }
    switch kind {
    case .continuous:
        return uniform(-8, 8)
    case .coarseTies:
        // 17 distinct values over the whole vocabulary: ties everywhere,
        // including across every top-k boundary.
        return floor(uniform(0, 17)) / 2 - 4
    case .allEqual:
        return MLXArray.full(shape, values: MLXArray(Float(0.25)))
    case .maskedMostly:
        let finite = floor(uniform(0, 5))
        return which(uniform(0, 1) .< 0.97, MLXArray(-Float.infinity), finite)
    case .withNaN:
        return spike(.nan)
    case .withPosInf:
        return spike(.infinity)
    case .allNegInf:
        return MLXArray.full(shape, values: MLXArray(-Float.infinity))
    }
}

private func makeBatch(_ kinds: [RowKind], vocab: Int, rng: inout Mix64) -> MLXArray {
    let rows = kinds.map { makeRow($0, vocab: vocab, rng: &rng) }
    let batch = rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
    eval(batch)
    return batch
}

/// Number of elements whose float32 bit patterns differ (NaN payloads and
/// signed zeros included).
private func bitMismatches(_ a: MLXArray, _ b: MLXArray) -> Int {
    let x = a.asType(.float32).view(dtype: .uint32)
    let y = b.asType(.float32).view(dtype: .uint32)
    return sum((x .!= y).asType(.int32)).item(Int.self)
}

private func column(_ values: [Int32]) -> MLXArray {
    MLXArray(values).reshaped([values.count, 1])
}

private func column(_ values: [Float]) -> MLXArray {
    MLXArray(values).reshaped([values.count, 1])
}

// MARK: - Stage (4) plans

final class CBv2SamplingFastPathTests: XCTestCase {

    /// Each case: a batch of logits rows and the kind of each row. One
    /// homogeneous batch per kind, then mixed batches.
    private func filterCases(vocab: Int, seed: UInt64, rowsPerCase: Int)
        -> [(logits: MLXArray, kinds: [RowKind])]
    {
        var rng = Mix64(state: seed)
        var cases: [(MLXArray, [RowKind])] = []
        for kind in RowKind.allCases {
            let kinds = Array(repeating: kind, count: rowsPerCase)
            cases.append((makeBatch(kinds, vocab: vocab, rng: &rng), kinds))
        }
        for _ in 0 ..< 2 {
            let kinds = (0 ..< rowsPerCase).map { _ in
                RowKind.allCases[Int(rng.next() % UInt64(RowKind.allCases.count))]
            }
            cases.append((makeBatch(kinds, vocab: vocab, rng: &rng), kinds))
        }
        return cases
    }

    private func assertBitIdentical(
        _ got: MLXArray, _ want: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.shape, want.shape, label, file: file, line: line)
        guard got.shape == want.shape else { return }
        let mismatches = bitMismatches(got, want)
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches) differing elements", file: file, line: line)
    }

    /// Stable head selection == the first k columns of the full stable sort,
    /// across ties, NaN, ±inf, padding and multi-level selection.
    func testStableHeadMatchesFullArgSortPrefix() {
        for (vocab, limit) in [(1000, 16), (1000, 64), (517, 8), (4096, 2048), (150_144, 2048)] {
            for (logits, kinds) in filterCases(vocab: vocab, seed: UInt64(vocab), rowsPerCase: 3) {
                let keys = -logits
                let full = argSort(keys, axis: -1).asType(.int32)
                for k in [1, 2, 5, 13, 20, 64] where k <= vocab {
                    let head = LogitsPipelineV2.stableHead(keys: keys, k: k, sortBlockLimit: limit)
                    let want = full[0..., ..<k].asArray(Int32.self)
                    // Rows with NaN may legitimately differ only past their
                    // non-NaN keys; every case here has >= k non-NaN keys.
                    XCTAssertEqual(
                        head.indices.asArray(Int32.self), want,
                        "vocab \(vocab) limit \(limit) k \(k) kinds \(kinds)")
                    let nan = any(isNaN(logits), axes: [1]).asArray(Bool.self)
                    XCTAssertEqual(head.anyNaN.asArray(Bool.self), nan, "NaN detection")
                }
            }
        }
    }

    func testSelectedHeadBitIdenticalToFullSortReference() {
        for (vocab, limit) in [(1000, 16), (517, 8), (150_144, 2048)] {
            for (logits, kinds) in filterCases(
                vocab: vocab, seed: 7 &+ UInt64(vocab), rowsPerCase: 4)
            {
                let b = logits.dim(0)
                for ks in [[1, 1, 1, 1], [20, 20, 20, 20], [3, 20, 7, 1], [64, 13, 2, 40]] {
                    let topK = column(ks.prefix(b).map(Int32.init))
                    let k = ks.prefix(b).max()!
                    let got = LogitsPipelineV2.applyTopKSelectedHead(
                        logits, topK: topK, headRanks: LogitsPipelineV2.ranks(k),
                        sortBlockLimit: limit)
                    let want = referenceFilter(
                        logits, topK: topK, topP: column([Float](repeating: 2, count: b)),
                        minP: column([Float](repeating: 0, count: b)))
                    assertBitIdentical(got, want, "selectedHead vocab \(vocab) ks \(ks) \(kinds)")
                }
            }
        }
    }

    func testSortedHeadBitIdenticalToFullSortReference() {
        for vocab in [1000, 150_144] {
            for (logits, kinds) in filterCases(
                vocab: vocab, seed: 11 &+ UInt64(vocab), rowsPerCase: 4)
            {
                let b = logits.dim(0)
                let settings: [([Int32], [Float], [Float])] = [
                    ([20, 20, 20, 20], [0.95, 0.95, 0.95, 0.95], [0, 0, 0, 0]),
                    ([20, 5, 50, 1], [0.5, 2, 0.99, 0], [0, 0.05, 0, 0.2]),
                    ([1024, 300, 64, 7], [0.999, 0.3, 2, 0.8], [0.01, 0, 0.5, 0]),
                    ([40, 40, 40, 40], [2, 2, 2, 2], [0.1, 0.02, 0.3, 1]),
                ]
                for (ks, ps, ms) in settings {
                    let topK = column(Array(ks.prefix(b)))
                    let topP = column(Array(ps.prefix(b)))
                    let minP = column(Array(ms.prefix(b)))
                    let k = Int(ks.prefix(b).max()!)
                    guard k < vocab else { continue }
                    let got = LogitsPipelineV2.applyTopKTopPMinPSortedHead(
                        logits, topK: topK, topP: topP, minP: minP,
                        headRanks: LogitsPipelineV2.ranks(k))
                    let want = referenceFilter(logits, topK: topK, topP: topP, minP: minP)
                    assertBitIdentical(got, want, "sortedHead vocab \(vocab) k \(ks) \(kinds)")
                }
            }
        }
    }

    func testFullSortScatterBitIdenticalToInversePermutation() {
        for vocab in [1000, 150_144] {
            for (logits, kinds) in filterCases(
                vocab: vocab, seed: 13 &+ UInt64(vocab), rowsPerCase: 4)
            {
                let b = logits.dim(0)
                let topK = column(Array([Int32(vocab), 20, Int32(vocab), 3].prefix(b)))
                let topP = column(Array([Float(2), 0.9, 0.7, 2].prefix(b)))
                let minP = column(Array([Float(0), 0, 0.05, 0.1].prefix(b)))
                let got = LogitsPipelineV2.applyTopKTopPMinP(
                    logits, topK: topK, topP: topP, minP: minP,
                    ranks: LogitsPipelineV2.ranks(vocab))
                let uncached = LogitsPipelineV2.applyTopKTopPMinP(
                    logits, topK: topK, topP: topP, minP: minP)
                let want = referenceFilter(logits, topK: topK, topP: topP, minP: minP)
                assertBitIdentical(got, want, "fullSort vocab \(vocab) \(kinds)")
                assertBitIdentical(uncached, want, "fullSort uncached vocab \(vocab)")
            }
        }
    }

    func testFilterPlanSelection() {
        let vocab = 1000
        func plan(_ params: [CBv2SamplingParams]) -> LogitsPipelineV2.FilterPlan {
            let pipeline = LogitsPipelineV2(vocabSize: vocab)
            pipeline.setRows(
                params.enumerated().map {
                    CBv2SamplerRow(
                        id: CBv2RequestID(UInt64($0.offset + 1)), params: $0.element,
                        promptTokens: [])
                })
            return pipeline.filterPlan
        }
        let production = CBv2SamplingParams(temperature: 1, topP: 0.95, topK: 20)
        XCTAssertEqual(plan([production, production]), .sortedHead(k: 20))
        XCTAssertEqual(
            plan([CBv2SamplingParams(temperature: 0.7, topK: 20), CBv2SamplingParams(topK: 5)]),
            .selectedHead(k: 20))
        XCTAssertEqual(
            plan([CBv2SamplingParams(temperature: 1, topK: 8, minP: 0.1)]), .sortedHead(k: 8))
        XCTAssertEqual(
            plan([CBv2SamplingParams(temperature: 1, topP: 0, topK: 8)]), .sortedHead(k: 8))
        // A greedy row keeps the whole vocabulary.
        XCTAssertEqual(plan([production, CBv2SamplingParams(temperature: 0)]), .fullSort)
        // A sampled row without top-k (or k >= vocab) keeps the whole vocabulary.
        XCTAssertEqual(plan([production, CBv2SamplingParams(topP: 0.9)]), .fullSort)
        XCTAssertEqual(plan([CBv2SamplingParams(topK: vocab)]), .fullSort)
        XCTAssertEqual(
            plan([CBv2SamplingParams(topK: LogitsPipelineV2.maxHeadTopK + 1)]), .fullSort)
    }

    /// The whole `process` output (temperature + stage 4) per plan, against
    /// the reference filter on the same tempered logits.
    func testPipelineProcessBitIdenticalAcrossPlans() {
        let vocab = 150_144
        var rng = Mix64(state: 99)
        let batches: [[CBv2SamplingParams]] = [
            Array(repeating: CBv2SamplingParams(temperature: 1, topP: 0.95, topK: 20), count: 4),
            [
                CBv2SamplingParams(temperature: 0.6, topP: 0.8, topK: 40),
                CBv2SamplingParams(temperature: 1.3, topK: 5, minP: 0.05),
                CBv2SamplingParams(temperature: 1, topK: 64),
            ],
            [
                CBv2SamplingParams(temperature: 0.8, topK: 20),
                CBv2SamplingParams(temperature: 1, topK: 3),
            ],
            [
                CBv2SamplingParams(temperature: 0),
                CBv2SamplingParams(temperature: 1, topP: 0.9, topK: 20),
                CBv2SamplingParams(temperature: 0.5, topP: 0.7),
            ],
        ]
        for params in batches {
            let b = params.count
            let rows = params.enumerated().map {
                CBv2SamplerRow(
                    id: CBv2RequestID(UInt64($0.offset + 1)), params: $0.element, promptTokens: [])
            }
            let pipeline = LogitsPipelineV2(vocabSize: vocab)
            pipeline.setRows(rows)
            for kind in [RowKind.continuous, .coarseTies, .maskedMostly] {
                let logits = makeBatch(Array(repeating: kind, count: b), vocab: vocab, rng: &rng)
                let got = pipeline.process(logits).sampling
                let greedy = params.map { $0.temperature < LogitsPipelineV2.greedyEpsilon }
                let temps = zip(params, greedy).map { $1 ? Float(1) : $0.temperature }
                var x = logits
                if temps.contains(where: { $0 != 1 }) { x = x / column(temps) }
                let topK = column(
                    zip(params, greedy).map { p, g in
                        !g && p.topK > 0 && p.topK < vocab ? Int32(p.topK) : Int32(vocab)
                    })
                let topP = column(
                    zip(params, greedy).map { p, g in
                        g ? Float(2) : (p.topP > 0 && p.topP < 1 ? p.topP : (p.topP <= 0 ? 0 : 2))
                    })
                let minP = column(zip(params, greedy).map { p, g in g ? 0 : min(p.minP, 1) })
                let want = referenceFilter(x, topK: topK, topP: topP, minP: minP)
                assertBitIdentical(got, want, "process \(pipeline.filterPlan) \(kind)")
            }
        }
    }
}
