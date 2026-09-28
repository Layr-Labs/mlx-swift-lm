// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Source-prepared component gates; no measured performance/peak or model claim.

import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest
@testable import MLXLMCommon

final class MiMoV26NAXAttentionKeyRangeTests: XCTestCase {
    private let payloadLimit = 64 << 20

    private func requireGPU() throws {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_TEST_MIMO_NAX_KEY_RANGES"] == "1"
        else { throw XCTSkip("Requires exclusive native lane and explicit key-range component opt-in") }
        guard MiMoV26NAXGatherQMM.gpuStream(.default), MiMoV26NAXGatherQMM.naxAvailable
        else { throw XCTSkip("M5 NAX required; M3 control is not kernel qualification") }
    }

    private func schedule(_ q: MLXArray, _ k: MLXArray, _ edges: [Int]) throws
        -> MiMoV26NAXAttentionKeyRanges.Schedule {
        try XCTUnwrap(MiMoV26NAXAttentionKeyRanges.makeSchedule(
            batch: q.dim(0), heads: q.dim(1), queries: q.dim(2), keys: k.dim(2),
            edges: edges, maximumPayloadBytes: payloadLimit))
    }

    private func fixture(_ length: Int, _ keyCount: Int, _ dtype: DType, batch: Int = 1)
        -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        ((MLXRandom.normal([batch, 4, length, 192], key: MLXRandom.key(40320)) * 0.5).asType(dtype),
         (MLXRandom.normal([batch, 2, keyCount, 192], key: MLXRandom.key(40321)) * 0.5).asType(dtype),
         (MLXRandom.normal([batch, 2, keyCount, 128], key: MLXRandom.key(40322)) * 0.5).asType(dtype))
    }

    private final class FaultRoots {
        var arrays: [MLXArray] = []
    }

    /// Keep every array until actual synchronous evaluation and readback.
    /// Failed completion retains real roots until this isolated test process
    /// exits; it is not a refund/retirement proof and normal cells must stop.
    @discardableResult
    private func compare(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, edges: [Int],
                         mask: MLXFast.ScaledDotProductAttentionMaskMode,
                         sinks: MLXArray? = nil, nativeExact: Bool = false,
                         firstNativeRowExact: Bool = false) throws -> Data {
        let roots = FaultRoots()
        roots.arrays = [q, k, v] + (sinks.map { [$0] } ?? [])
        var completed = false
        defer { if !completed { _ = Unmanaged.passRetained(roots) } }
        var bytes = Data()
        try withError { errors in
            let scale = 1 / Float(192).squareRoot()
            let plan = try XCTUnwrap(MiMoV26NAXAttention.makePlan(
                queries: q, keys: k, values: v, scale: scale,
                mask: mask, sinks: sinks, production: false))
            let old = MiMoV26NAXAttention.launch(
                queries: q, keys: k, values: v, scale: scale, plan: plan)
            roots.arrays.append(old)
            let bounds = try schedule(q, k, edges)
            let split = try XCTUnwrap(MiMoV26NAXAttentionKeyRanges.launch(
                queries: q, keys: k, values: v, scale: scale, mask: mask, sinks: sinks,
                schedule: bounds, maximumPayloadBytes: payloadLimit,
                retain: { roots.arrays.append(contentsOf: $0) }))
            roots.arrays.append(split.output)
            // Every non-final phase/range output is full FP32 state. No
            // ping-pong/reclamation assumption: all actual objects stay held.
            XCTAssertEqual(split.retainedArrays.count, 4 + 3 * bounds.dispatches)
            XCTAssertEqual(split.retainedArrays.filter {
                $0.dtype == .float32 && $0.size == bounds.stateElements
            }.count, bounds.fullStateBuffers)
            let native = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask, sinks: sinks)
            roots.arrays.append(native)
            try errors.check()
            eval(roots.arrays)
            try errors.check()
            XCTAssertEqual(split.output.shape, old.shape)
            XCTAssertEqual(split.output.dtype, old.dtype)
            bytes = split.output.asData().data
            XCTAssertEqual(bytes, old.asData().data,
                "range handoff must preserve stored bytes, including signed zero/NaN payloads")
            if nativeExact { XCTAssertEqual(bytes, native.asData().data) }
            if firstNativeRowExact {
                // Overflow cells deliberately have unrelated all-infinite
                // rows; never require finiteness or equal NaN payloads there.
                let a = split.output[0..., 0..., 0..<1, 0...]
                let b = native[0..., 0..., 0..<1, 0...]
                XCTAssertEqual(a.asData().data, b.asData().data)
            }
            let a = split.output.view(dtype: .uint16).asArray(UInt16.self)
            let b = native.view(dtype: .uint16).asArray(UInt16.self)
            let exponent: UInt16 = q.dtype == .bfloat16 ? 0x7f80 : 0x7c00
            func ordered(_ x: UInt16) -> UInt32 {
                x & 0x8000 == 0 ? UInt32(x) + 0x8000 : UInt32(~x)
            }
            var maxULP: UInt32 = 0, nonfinitePairs = 0
            for (x, y) in zip(a, b) {
                if x & exponent == exponent || y & exponent == exponent {
                    nonfinitePairs += 1
                } else {
                    let xx = ordered(x), yy = ordered(y)
                    maxULP = max(maxULP, xx > yy ? xx - yy : yy - xx)
                }
            }
            print("MiMoKeyRanges shape=\(q.shape)/\(k.shape) dtype=\(q.dtype)"
                + " edges=\(edges) oldBytesExact=\(bytes == old.asData().data)"
                + " nativeMaxFiniteULP=\(maxULP) nativeNonfinitePairs=\(nonfinitePairs)"
                + " heldStateBuffers=\(bounds.fullStateBuffers) arrayPayloadBytes=\(bounds.arrayPayloadBytes)")
            try errors.check()
        }
        completed = true
        return bytes
    }

    func testBalancedRangesAndAllPendingStatePayloadAreBounded() throws {
        typealias R = MiMoV26NAXAttentionKeyRanges
        XCTAssertEqual(R.balancedEdges(batch: 1, heads: 64, queries: 448, keys: 65536), [0, 2048])
        XCTAssertEqual(R.balancedEdges(batch: 1, heads: 64, queries: 512, keys: 65536),
                       Array(stride(from: 0, through: 2048, by: 256)))
        XCTAssertEqual(R.balancedEdges(batch: 1, heads: 64, queries: 512, keys: 12288), [0, 192, 384])
        // Python ties-to-even: 2.5 ranges -> 2, not 3.
        XCTAssertEqual(R.balancedEdges(batch: 1, heads: 64, queries: 512, keys: 20480), [0, 320, 640])
        XCTAssertEqual(R.balancedEdges(batch: 1, heads: 64, queries: 512, keys: 21001), [0, 219, 438, 657])
        XCTAssertNil(R.balancedEdges(batch: 1, heads: 64, queries: 512, keys: 8192,
                                    preferredKeys: 1, minimumThreadgroups: 0))
        let edges = [0, 1, 3, 9] // 257 keys, deliberately uneven and padded.
        let value = try XCTUnwrap(R.makeSchedule(batch: 2, heads: 4, queries: 65, keys: 257,
                                               edges: edges, maximumPayloadBytes: Int.max))
        XCTAssertEqual(value.stateElements, 2 * 4 * 128 * 130)
        XCTAssertEqual(value.fullStateBuffers, 8)
        XCTAssertEqual(value.dispatches, 9)
        XCTAssertGreaterThan(value.arrayPayloadBytes, 2 * value.stateBufferBytes)
        XCTAssertNotNil(R.makeSchedule(batch: 2, heads: 4, queries: 65, keys: 257,
                                      edges: edges, maximumPayloadBytes: value.arrayPayloadBytes))
        XCTAssertNil(R.makeSchedule(batch: 2, heads: 4, queries: 65, keys: 257,
                                   edges: edges, maximumPayloadBytes: value.arrayPayloadBytes - 1))
        for bad in [[0, 0, 9], [0, 5, 4, 9], [1, 9], [0, 8], [0, 10], []] {
            XCTAssertNil(R.makeSchedule(batch: 2, heads: 4, queries: 65, keys: 257,
                                       edges: bad, maximumPayloadBytes: Int.max))
        }
        XCTAssertNil(R.makeSchedule(batch: Int.max, heads: 64, queries: 65, keys: 257,
                                   edges: edges, maximumPayloadBytes: Int.max))
        XCTAssertNil(R.makeSchedule(batch: 64, heads: 64, queries: 1_048_576, keys: 1_048_576,
                                   edges: [0, 32768], maximumPayloadBytes: Int.max))
        XCTAssertNil(R.makeSchedule(batch: 1, heads: 1, queries: 8, keys: 257,
                                   edges: edges, maximumPayloadBytes: Int.max))
    }

    func testCPUStreamRefusesBeforeCustomEncoding() throws {
        let f = fixture(64, 128, .bfloat16)
        let bounds = try schedule(f.q, f.k, [0, 2, 4])
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.launch(
            queries: f.q, keys: f.k, values: f.v, scale: 0.07, mask: .causal,
            sinks: nil, schedule: bounds, maximumPayloadBytes: payloadLimit, stream: .cpu,
            retain: { _ in XCTFail("CPU refusal must precede range graph allocation") }))
    }

    func testOneManyUnevenRangesPreserveStoredBytesAcrossMasksAndSinks() throws {
        try requireGPU()
        for dtype in [DType.bfloat16, .float16] {
            for (length, keyCount, edges) in [(64, 128, [0, 4]), (65, 257, [0, 1, 3, 9]),
                                            (100, 300, [0, 3, 6, 10]), (16, 8193, [0, 128, 257])] {
                let f = fixture(length, keyCount, dtype, batch: length == 65 ? 2 : 1)
                let sink = MLXArray([Float(-0.75), 0, 0.5, 1.25]).asType(dtype)
                for mode in 0..<4 {
                    let mask: MLXFast.ScaledDotProductAttentionMaskMode
                    switch mode {
                    case 0: mask = .none
                    case 1: mask = .causal
                    case 2: mask = .array(createCausalMask(n: length, offset: keyCount - length, windowSize: 47))
                    default: mask = .array(MLXArray.zeros([length, keyCount], dtype: .bool))
                    }
                    try compare(f.q, f.k, f.v, edges: edges, mask: mask,
                                sinks: mode % 2 == 0 ? nil : sink)
                }
            }
        }
    }

    func testLazyStridedKVMaskAndSinksPreserveExactRangeState() throws {
        try requireGPU()
        for dtype in [DType.bfloat16, .float16] {
            let qWide = MLXRandom.normal([2, 65, 4, 384], key: MLXRandom.key(410)).asType(dtype)
            let kWide = MLXRandom.normal([2, 2, 514, 384], key: MLXRandom.key(411)).asType(dtype)
            let vWide = MLXRandom.normal([2, 2, 514, 256], key: MLXRandom.key(412)).asType(dtype)
            let q = qWide.transposed(0, 2, 1, 3)[.ellipsis, .stride(by: 2)]
            let k = kWide[0..., 0..., .stride(by: 2), .stride(by: 2)]
            let v = vWide[0..., 0..., .stride(by: 2), .stride(by: 2)]
            let maskWide = createCausalMask(n: 130, offset: 514 - 130, windowSize: 94)
            let mask = maskWide[.stride(by: 2), .stride(by: 2)]
            let sinks = MLXArray([Float(0), 99, 0.5, 99, -0.25, 99, 0.75, 99])
                .asType(dtype)[.stride(by: 2)]
            // No eval before handing lazy inner-head and sequence views in.
            let a = try compare(q, k, v, edges: [0, 1, 3, 9], mask: .array(mask), sinks: sinks)
            let b = try compare(contiguous(q), contiguous(k), contiguous(v),
                                edges: [0, 1, 3, 9], mask: .array(contiguous(mask)),
                                sinks: contiguous(sinks))
            XCTAssertEqual(a, b)
        }
    }

    func testNativeUniformRoundingAndFullyMaskedRowsRemainExact() throws {
        try requireGPU()
        for dtype in [DType.bfloat16, .float16] {
            let q = MLXArray.zeros([1, 4, 65, 192], dtype: dtype)
            for (count, sink, masked) in [(128, false, false), (127, true, false), (127, false, true)] {
                let k = MLXArray.zeros([1, 2, count, 192], dtype: dtype)
                let v = MLXArray.ones([1, 2, count, 128], dtype: dtype)
                let mask: MLXFast.ScaledDotProductAttentionMaskMode = masked
                    ? .array(MLXArray.zeros([65, count], dtype: .bool)) : .none
                try compare(q, k, v, edges: [0, 1, 4], mask: mask,
                            sinks: sink ? MLXArray.zeros([4], dtype: dtype) : nil, nativeExact: true)
            }
        }
    }

    func testOverflowAndNonfiniteRowsDoNotTruncateRealMaskedKeys() throws {
        try requireGPU()
        let q = MLXArray.full([1, 4, 256, 192], values: MLXArray(Float(2048)), dtype: .float16)
        let k = MLXArray.full([1, 2, 256, 192], values: MLXArray(Float(-2048)), dtype: .float16)
        let v = concatenated([MLXArray.zeros([1, 2, 128, 128], dtype: .float16),
                              MLXArray.ones([1, 2, 128, 128], dtype: .float16)], axis: 2)
        // Native QK overflows to -inf. The first row's real masked keys use
        // finite minimum and contribute the out-of-prefix nonzero V values.
        // A causal-range truncation would incorrectly produce 0 or NaN.
        try compare(q, k, v, edges: [0, 1, 4, 8], mask: .causal, firstNativeRowExact: true)
        for dtype in [DType.bfloat16, .float16] {
            let normal = fixture(65, 127, dtype)
            let nanQ = MLXArray.full(normal.q.shape, values: MLXArray(Float.nan), dtype: dtype)
            try compare(nanQ, normal.k, normal.v, edges: [0, 1, 4], mask: .none)
            let infiniteV = MLXArray.full(normal.v.shape, values: MLXArray(Float.infinity), dtype: dtype)
            try compare(normal.q, normal.k, infiniteV, edges: [0, 1, 4],
                        mask: .array(MLXArray.zeros([65, 127], dtype: .bool)))
        }
    }
}
