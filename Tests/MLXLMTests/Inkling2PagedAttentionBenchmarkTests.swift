// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Microbenchmark of `Inkling2PagedAttention` against the engine's paged decode
/// kernel (no bias, no sinks) and MLX's fused SDPA (the engine's prefill
/// terminal) at the same shapes. Opt-in: `INKLING2_ATTN_BENCH=1`. Results go to
/// stdout and, when set, `INKLING2_ATTN_BENCH_OUT` (JSON lines).
final class Inkling2PagedAttentionBenchmarkTests: XCTestCase {
    typealias Attention = Inkling2PagedAttention

    let qh = 24
    let kvh = 8
    let d = 128
    let s = 16
    let scale = Float(1.0 / Double(128).squareRoot())

    var enabled: Bool { ProcessInfo.processInfo.environment["INKLING2_ATTN_BENCH"] == "1" }

    /// The command-buffer limits Inkling 2 decodes under
    /// (`Inkling2Model.decodeCommandBufferLimits`): no commit per call for the
    /// megabytes of the bound K/V pool.
    let decodeLimits = GPU.CommandBufferLimits(operations: 400, megabytes: 1 << 20)

    /// Per-call cost of a chain of calls, from the slope between chains of 32
    /// and 160 calls (the per-evaluation constant cancels), under the decode
    /// command-buffer limits: graph building, CPU encoding (`asyncEval`
    /// returns once every command buffer is encoded and committed) and the
    /// whole evaluation until the GPU is done.
    func chainCost(_ build: (Int) -> MLXArray) -> (host: Double, encode: Double, total: Double) {
        func best(_ n: Int) -> (host: Double, encode: Double, total: Double) {
            var result = (host: Double.infinity, encode: Double.infinity, total: Double.infinity)
            for attempt in 0 ... 8 {
                Stream.gpu.synchronize()
                let t0 = DispatchTime.now().uptimeNanoseconds
                let out = build(n)
                let t1 = DispatchTime.now().uptimeNanoseconds
                GPU.withCommandBufferLimits(decodeLimits) { asyncEval(out) }
                let t2 = DispatchTime.now().uptimeNanoseconds
                Stream.gpu.synchronize()
                let t3 = DispatchTime.now().uptimeNanoseconds
                if attempt > 0 {
                    result.host = min(result.host, Double(t1 - t0) / 1e3)
                    result.encode = min(result.encode, Double(t2 - t1) / 1e3)
                    result.total = min(result.total, Double(t3 - t1) / 1e3)
                }
            }
            return result
        }
        let short = best(32)
        let long = best(160)
        func slope(_ a: Double, _ b: Double) -> Double { ((b - a) / 128 * 10).rounded() / 10 }
        return (
            slope(short.host, long.host), slope(short.encode, long.encode),
            slope(short.total, long.total)
        )
    }

    /// GPU time per call: the whole chain goes into one command buffer, so
    /// `asyncEval` returns once every call is encoded and the GPU only starts
    /// then; the wait after it is the chain's GPU execution (with the same
    /// barriers between dependent dispatches as in a decode step). Slope
    /// between chains of 16 and 80 calls; best of 6.
    func gpuCost(_ build: (Int) -> MLXArray) -> (encode: Double, gpu: Double) {
        let oneBuffer = GPU.CommandBufferLimits(operations: 1 << 30, megabytes: 1 << 20)
        func best(_ n: Int) -> (encode: Double, gpu: Double) {
            var result = (encode: Double.infinity, gpu: Double.infinity)
            for attempt in 0 ... 6 {
                let out = build(n)
                Stream.gpu.synchronize()
                let t1 = DispatchTime.now().uptimeNanoseconds
                GPU.withCommandBufferLimits(oneBuffer) { asyncEval(out) }
                let t2 = DispatchTime.now().uptimeNanoseconds
                Stream.gpu.synchronize()
                let t3 = DispatchTime.now().uptimeNanoseconds
                if attempt > 0 {
                    result.encode = min(result.encode, Double(t2 - t1) / 1e3)
                    result.gpu = min(result.gpu, Double(t3 - t2) / 1e3)
                }
            }
            return result
        }
        let short = best(16)
        let long = best(80)
        func slope(_ a: Double, _ b: Double) -> Double { ((b - a) / 64 * 10).rounded() / 10 }
        return (slope(short.encode, long.encode), slope(short.gpu, long.gpu))
    }

    func record(_ fields: [String: Any]) {
        let data = try! JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys])
        let line = String(decoding: data, as: UTF8.self)
        print("[Inkling2AttnBench] \(line)")
        if let path = ProcessInfo.processInfo.environment["INKLING2_ATTN_BENCH_OUT"] {
            let handle =
                FileHandle(forWritingAtPath: path)
                ?? {
                    FileManager.default.createFile(atPath: path, contents: nil)
                    return FileHandle(forWritingAtPath: path)!
                }()
            handle.seekToEndOfFile()
            handle.write((line + "\n").data(using: .utf8)!)
            try? handle.close()
        }
    }

    /// Best of `repeats` timings of `count` chained calls, microseconds per call.
    func time(count: Int, repeats: Int = 5, _ chain: (Int) -> MLXArray) -> Double {
        var best = Double.infinity
        for attempt in 0 ... repeats {
            Stream.gpu.synchronize()
            let start = DispatchTime.now().uptimeNanoseconds
            let out = chain(count)
            eval(out)
            Stream.gpu.synchronize()
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / Double(count)
            if attempt > 0 { best = min(best, elapsed) }
        }
        return best
    }

    struct Pool {
        let k: MLXArray
        let v: MLXArray
        let pages: Int
    }

    func makePool(pages: Int) -> Pool {
        MLXRandom.seed(1)
        let k = (MLXRandom.normal([pages, kvh, s, d]) * 0.6).asType(.bfloat16)
        let v = MLXRandom.normal([pages, kvh, s, d]).asType(.bfloat16)
        eval(k, v)
        return Pool(k: k, v: v, pages: pages)
    }

    /// Random page tables; rows alias pages once the pool is exhausted.
    func tables(rows: Int, tokens: Int, window: Int?, pool: Pool) -> (MLXArray, Int) {
        let need = (tokens + s - 1) / s
        let ring = window.map { _ in 65 }
        let length = ring.map { min(need, $0) } ?? need
        let width = max(8, ring ?? length)
        var flat = [Int32](repeating: 0, count: rows * width)
        var next = 1
        for r in 0 ..< rows {
            for p in 0 ..< length {
                flat[r * width + p] = Int32(next)
                next = next + 1 >= pool.pages ? 1 : next + 1
            }
        }
        return (MLXArray(flat, [rows, width]), ring ?? max(1, length))
    }

    func testDecodeAndPrefillMicrobenchmark() throws {
        guard enabled else { throw XCTSkip("set INKLING2_ATTN_BENCH=1") }
        MLX.GPU.set(memoryLimit: 12 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let pool = makePool(pages: 16_384)
        let source = try PagedAttentionResources.loadSourceForCurrentProcess()
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        let chain = Int(ProcessInfo.processInfo.environment["INKLING2_ATTN_BENCH_CHAIN"] ?? "48")!
        let contexts = [1024, 8192, 32768, 131_072]
        // `INKLING2_ATTN_BENCH_PREFILL_ONLY=1` skips the decode sweep.
        let decodeWindows: [Int?] =
            ProcessInfo.processInfo.environment["INKLING2_ATTN_BENCH_PREFILL_ONLY"] == "1"
            ? [] : [1024, nil]

        for window in decodeWindows {
            for ctx in contexts {
                for b in [1, 8, 32] {
                    let (table, tableLength) = tables(
                        rows: b, tokens: ctx + 1, window: window, pool: pool)
                    let q0 = MLXRandom.normal([b, qh, 1, d]).asType(.bfloat16)
                    let kNew = MLXRandom.normal([b, kvh, 1, d]).asType(.bfloat16)
                    let vNew = MLXRandom.normal([b, kvh, 1, d]).asType(.bfloat16)
                    let states = (MLXRandom.normal([b, qh, 1, 16]) * 0.5).asType(.bfloat16)
                    let sinks = MLXRandom.normal([b, qh, 1])
                    let fence = MLXArray.zeros([1], dtype: .int32)
                    eval(table, q0, kNew, vNew, states, sinks)
                    let plan = Attention.Plan(
                        rows: Array(
                            repeating: .init(queryStart: ctx, tableLength: tableLength), count: b),
                        queries: 1, window: window, scale: scale)
                    func mine(_ kernel: Attention.Kernel) -> Double {
                        time(count: chain) { n in
                            var q = q0
                            for _ in 0 ..< n {
                                q = Attention.attend(
                                    queries: q, newKeys: kNew, newValues: vNew,
                                    relativeStates: states, proj: proj, sinks: sinks,
                                    cache: .init(
                                        keys: pool.k, values: pool.v, tables: table,
                                        writeFence: fence),
                                    plan: plan, kernel: kernel
                                ).output.reshaped(b, qh, 1, d)
                            }
                            return q
                        }
                    }
                    let split = mine(.split)
                    let sequential = mine(.sequential)
                    // Engine kernel: one query over the same attended range.
                    let start = window.map { max(0, ctx - $0 + 1) } ?? 0
                    let (seqinfo, maxLength) = PagedAttentionKernel.seqinfo(
                        Array(
                            repeating: PagedAttentionKernel.SeqInfoRow(
                                attendStart: start, attendLength: ctx + 1 - start,
                                tableLength: tableLength), count: b))
                    let params = MLXArray([1, scale, 0, 0, 0, 0, 0, 0] as [Float])
                    let engine = time(count: chain) { n in
                        var q = q0.squeezed(axis: 2)
                        for _ in 0 ..< n {
                            q = try! PagedAttentionKernel.decode(
                                queries: q, kSlab: pool.k, vSlab: pool.v, tables: table,
                                seqinfo: seqinfo, maxAttendLength: maxLength, sinks: nil,
                                params: params, softcap: false, pageSize: s, writeFence: fence,
                                kernelSource: source
                            ).out
                        }
                        return q
                    }
                    record([
                        "phase": "decode", "kind": window == nil ? "full" : "sliding", "B": b,
                        "ctx": ctx, "inkling2_split_us": (split * 10).rounded() / 10,
                        "inkling2_sequential_us": (sequential * 10).rounded() / 10,
                        "engine_us": (engine * 10).rounded() / 10,
                        "split_over_engine": (split / engine * 100).rounded() / 100,
                    ])
                }
            }
        }

        let l = 2048
        for window in [nil, 1024] as [Int?] {
            for ctx in [0, 8192, 32768] {
                let (table, tableLength) = tables(
                    rows: 1, tokens: ctx + l, window: window, pool: pool)
                let q = MLXRandom.normal([1, qh, l, d]).asType(.bfloat16)
                let kNew = MLXRandom.normal([1, kvh, l, d]).asType(.bfloat16)
                let vNew = MLXRandom.normal([1, kvh, l, d]).asType(.bfloat16)
                let states = (MLXRandom.normal([1, qh, l, 16]) * 0.5).asType(.bfloat16)
                let sinks = MLXRandom.normal([1, qh, l])
                let fence = MLXArray.zeros([1], dtype: .int32)
                eval(table, q, kNew, vNew, states, sinks)
                let plan = Attention.Plan(
                    rows: [.init(queryStart: ctx, tableLength: tableLength)], queries: l,
                    window: window, scale: scale)
                func prefill(_ kernel: Attention.Kernel) -> Double {
                    time(count: 3, repeats: 3) { n in
                        var outs: [MLXArray] = []
                        for _ in 0 ..< n {
                            outs.append(
                                Attention.attend(
                                    queries: q, newKeys: kNew, newValues: vNew,
                                    relativeStates: states, proj: proj, sinks: sinks,
                                    cache: .init(
                                        keys: pool.k, values: pool.v, tables: table,
                                        writeFence: fence),
                                    plan: plan, kernel: kernel
                                ).output)
                        }
                        return stacked(outs)
                    }
                }
                let mine = prefill(.sequential)
                let accelerated =
                    Attention.neuralAcceleratorsAvailable ? prefill(.prefill) : Double.nan
                // The engine's prefill terminal: fused SDPA over gathered keys
                // (no bias, no per-query sink), causal or windowed.
                let history = window.map { min(ctx, $0 - 1) } ?? ctx
                let keys = MLXRandom.normal([1, kvh, history + l, d]).asType(.bfloat16)
                let values = MLXRandom.normal([1, kvh, history + l, d]).asType(.bfloat16)
                eval(keys, values)
                let mask: MLXFast.ScaledDotProductAttentionMaskMode
                if let window {
                    let qpos = MLXArray(Int32(history) ..< Int32(history + l)).expandedDimensions(
                        axis: 1)
                    let kpos = MLXArray(Int32(0) ..< Int32(history + l)).expandedDimensions(axis: 0)
                    let m = (kpos .<= qpos) .&& (kpos .> (qpos - Int32(window)))
                    eval(m)
                    mask = .array(m)
                } else {
                    mask = .causal
                }
                let sdpa = time(count: 3, repeats: 3) { n in
                    var outs: [MLXArray] = []
                    for _ in 0 ..< n {
                        outs.append(
                            MLXFast.scaledDotProductAttention(
                                queries: q, keys: keys, values: values, scale: scale, mask: mask))
                    }
                    return stacked(outs)
                }
                record([
                    "phase": "prefill", "kind": window == nil ? "full" : "sliding", "L": l,
                    "ctx": ctx, "inkling2_ms": (mine / 100).rounded() / 10,
                    "inkling2_tok_s": Int(Double(l) / (mine / 1e6)),
                    "inkling2_na_ms": accelerated.isNaN ? -1 : (accelerated / 100).rounded() / 10,
                    "sdpa_ms": (sdpa / 100).rounded() / 10,
                    "sdpa_tok_s": Int(Double(l) / (sdpa / 1e6)),
                ])
            }
        }
    }

    /// Per-call graph building, CPU encoding and chained time of decode calls
    /// (one per layer, each consuming the previous output) against the
    /// engine's paged decode kernel at the same shapes (no bias, no sinks),
    /// under the decode command-buffer limits. Opt-in:
    /// `INKLING2_ATTN_LATENCY=1`; `INKLING2_ATTN_LATENCY_BATCHES` (default
    /// `1,8,32`).
    func testDecodeLatencySplit() throws {
        guard ProcessInfo.processInfo.environment["INKLING2_ATTN_LATENCY"] == "1" else {
            throw XCTSkip("set INKLING2_ATTN_LATENCY=1")
        }
        MLX.GPU.set(memoryLimit: 12 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let pool = makePool(pages: 16_384)
        let source = try PagedAttentionResources.loadSourceForCurrentProcess()
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        let batches =
            (ProcessInfo.processInfo.environment["INKLING2_ATTN_LATENCY_BATCHES"]
            ?? "1,8,32").split(separator: ",").compactMap { Int($0) }
        let warm = MLXRandom.normal([4096, 4096])
        for _ in 0 ..< 20 { eval(matmul(warm, warm)) }
        for (window, ctx) in [(1024, 8192), (nil, 8192), (nil, 32768)] as [(Int?, Int)] {
            for b in batches {
                let (table, tableLength) = tables(
                    rows: b, tokens: ctx + 1, window: window, pool: pool)
                let q0 = MLXRandom.normal([b, 1, qh, d]).asType(.bfloat16)
                let kNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
                let vNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
                let states = (MLXRandom.normal([b, 1, qh, 16]) * 0.5).asType(.bfloat16)
                let sinks = MLXRandom.normal([b, 1, qh])
                let fence = MLXArray.zeros([1], dtype: .int32)
                eval(table, q0, kNew, vNew, states, sinks, proj)
                let rows = Array(
                    repeating: Attention.Row(queryStart: ctx, tableLength: tableLength), count: b)
                let plan = Attention.Plan(rows: rows, queries: 1, window: window, scale: scale)
                let mine = chainCost { n in
                    var q = q0
                    for _ in 0 ..< n {
                        q = Attention.attend(
                            queries: q, newKeys: kNew, newValues: vNew, relativeStates: states,
                            proj: proj, sinks: sinks,
                            cache: .init(
                                keys: pool.k, values: pool.v, tables: table, writeFence: fence),
                            plan: plan, layout: .tokenMajor
                        ).output.reshaped(b, 1, qh, d)
                    }
                    return q
                }
                let start = window.map { max(0, ctx - $0 + 1) } ?? 0
                let (seqinfo, maxLength) = PagedAttentionKernel.seqinfo(
                    Array(
                        repeating: PagedAttentionKernel.SeqInfoRow(
                            attendStart: start, attendLength: ctx + 1 - start,
                            tableLength: tableLength), count: b))
                let params = MLXArray([1, scale, 0, 0, 0, 0, 0, 0] as [Float])
                let engine = chainCost { n in
                    var q = q0.reshaped(b, qh, d)
                    for _ in 0 ..< n {
                        q = try! PagedAttentionKernel.decode(
                            queries: q, kSlab: pool.k, vSlab: pool.v, tables: table,
                            seqinfo: seqinfo, maxAttendLength: maxLength, sinks: nil,
                            params: params, softcap: false, pageSize: s, writeFence: fence,
                            kernelSource: source
                        ).out
                    }
                    return q
                }
                record([
                    "phase": "decode-latency", "kind": window == nil ? "full" : "sliding", "B": b,
                    "ctx": ctx, "inkling2_host_us": mine.host, "inkling2_encode_us": mine.encode,
                    "inkling2_call_us": mine.total, "engine_host_us": engine.host,
                    "engine_encode_us": engine.encode, "engine_call_us": engine.total,
                ])
            }
        }
    }

    /// Where a B=1 decode call's time goes, per call from the slope between
    /// chains of 32 and 160 calls (the per-evaluation constant cancels): a
    /// trivial chained kernel, a no-op kernel with the split pass's bindings and
    /// grid, the split pass alone (chained through the write fence, so the
    /// merge is never evaluated) and the whole call. Opt-in:
    /// `INKLING2_ATTN_LATENCY=1`.
    func testDecodeLatencyAnatomy() throws {
        guard ProcessInfo.processInfo.environment["INKLING2_ATTN_LATENCY"] == "1" else {
            throw XCTSkip("set INKLING2_ATTN_LATENCY=1")
        }
        MLX.GPU.set(memoryLimit: 12 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let pool = makePool(pages: 16_384)
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        let copy = MLXFast.metalKernel(
            name: "inkling2_latency_floor", inputNames: ["x"], outputNames: ["y"],
            source: "uint i = thread_position_in_grid.x; y[i] = x[i];")
        let noop = MLXFast.metalKernel(
            name: "inkling2_latency_noop",
            inputNames: [
                "q", "rel", "proj", "knew", "vnew", "kcache", "vcache", "tables", "rowinfo",
                "wfence",
            ],
            outputNames: ["partials", "fence"],
            source: """
                const int L = q_shape[1] + knew_shape[2] + tables_shape[1];
                if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
                    && thread_position_in_grid.z == 0) {
                    fence[0] = wfence[0] + 1 + (L < 0 ? 1 : 0);
                }
                """,
            header: Inkling2PagedAttentionMetal.header)
        // Warm the GPU clocks before the first timed configuration.
        let warm = MLXRandom.normal([4096, 4096])
        for _ in 0 ..< 20 { eval(matmul(warm, warm)) }
        for (window, ctx) in [(nil, 15), (nil, 1023), (1024, 8192), (nil, 8192)] as [(Int?, Int)] {
            let b = 1
            let (table, tableLength) = tables(rows: b, tokens: ctx + 1, window: window, pool: pool)
            let q0 = MLXRandom.normal([b, 1, qh, d]).asType(.bfloat16)
            let kNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
            let vNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
            let states = (MLXRandom.normal([b, 1, qh, 16]) * 0.5).asType(.bfloat16)
            let sinks = MLXRandom.normal([b, 1, qh])
            let fence = MLXArray.zeros([1], dtype: .int32)
            eval(table, q0, kNew, vNew, states, sinks, proj)
            let rows = [Attention.Row(queryStart: ctx, tableLength: tableLength)]
            let plan = Attention.Plan(rows: rows, queries: 1, window: window, scale: scale)
            eval(plan.rowInfo)
            func call(_ q: MLXArray, _ f: MLXArray) -> Attention.Output {
                Attention.attend(
                    queries: q, newKeys: kNew, newValues: vNew, relativeStates: states,
                    proj: proj, sinks: sinks,
                    cache: .init(keys: pool.k, values: pool.v, tables: table, writeFence: f),
                    plan: plan, layout: .tokenMajor)
            }
            let floor = chainCost { n in
                var x = q0.reshaped(-1)
                for _ in 0 ..< n {
                    x =
                        copy(
                            [x], grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
                            outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
                }
                return x
            }
            let parts = plan.maxPartitions
            let noopCall = chainCost { n in
                var f = fence
                for _ in 0 ..< n {
                    f =
                        noop(
                            [
                                q0, states, proj, kNew, vNew, pool.k, pool.v, table,
                                plan.rowInfo, f,
                            ], grid: (kvh * 32, b, parts), threadGroup: (32, 1, 1),
                            outputShapes: [[b, qh, 1, parts, d + 2], [1]],
                            outputDTypes: [.float32, .int32])[1]
                }
                return f
            }
            let split = chainCost { n in
                var f = fence
                for _ in 0 ..< n { f = call(q0, f).writeFence }
                return f
            }
            let whole = chainCost { n in
                var q = q0
                for _ in 0 ..< n { q = call(q, fence).output.reshaped(b, 1, qh, d) }
                return q
            }
            record([
                "phase": "decode-anatomy", "kind": window == nil ? "full" : "sliding", "B": b,
                "ctx": ctx, "partitions": parts,
                "floor_us": "\(floor)", "noop_split_signature_us": "\(noopCall)",
                "split_only_us": "\(split)", "whole_us": "\(whole)",
            ])
        }
    }

    /// Chained no-op kernels with subsets of the split pass's bindings: which
    /// binding costs encode or launch time. Opt-in: `INKLING2_ATTN_LATENCY=1`.
    func testDispatchBindingCost() throws {
        guard ProcessInfo.processInfo.environment["INKLING2_ATTN_LATENCY"] == "1" else {
            throw XCTSkip("set INKLING2_ATTN_LATENCY=1")
        }
        MLX.GPU.set(memoryLimit: 12 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let pool = makePool(pages: 16_384)
        let small = makePool(pages: 64)
        let fence = MLXArray.zeros([1], dtype: .int32)
        let q = MLXRandom.normal([1, 1, qh, d]).asType(.bfloat16)
        let table = MLXArray.zeros([1, 72], dtype: .int32)
        eval(q, table, fence)
        let warm = MLXRandom.normal([4096, 4096])
        for _ in 0 ..< 20 { eval(matmul(warm, warm)) }
        let cases: [(String, [MLXArray], Int)] = [
            ("fence_only_1out", [], 1),
            ("fence_only_3out", [], 3),
            ("fence_only_3tinyout", [], -3),
            ("small_pool_3out", [small.k, small.v], 3),
            ("big_pool_3out", [pool.k, pool.v], 3),
            ("big_pool_1out", [pool.k, pool.v], 1),
            ("big_k_only_1out", [pool.k], 1),
            ("q_table_1out", [q, table], 1),
        ]
        var fields: [String: Any] = ["phase": "binding-cost"]
        for (name, inputs, outputs) in cases {
            let names = inputs.indices.map { "in\($0)" } + ["wfence"]
            let outNames = ["fence"] + (abs(outputs) > 1 ? ["o1", "o2"] : [])
            let kernel = MLXFast.metalKernel(
                name: "inkling2_binding_\(name)", inputNames: names, outputNames: outNames,
                source: """
                    if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
                        && thread_position_in_grid.z == 0) {
                        fence[0] = wfence[0] + 1;
                    }
                    """)
            let shapes: [[Int]] =
                [[1]]
                + (outputs > 1 ? [[1, qh, 1, 9, d], [432]] : outputs < -1 ? [[1], [1]] : [])
            let dtypes: [DType] = [.int32] + (abs(outputs) > 1 ? [.float32, .float32] : [])
            let cost = chainCost { n in
                var f = fence
                for _ in 0 ..< n {
                    f =
                        kernel(
                            inputs + [f], grid: (kvh * 32, 1, 9), threadGroup: (32, 1, 1),
                            outputShapes: shapes, outputDTypes: dtypes)[0]
                }
                return f
            }
            fields[name + "_us"] = "\(cost)"
        }
        record(fields)
    }

    /// GPU execution time per decode call (one command buffer per chain, see
    /// `gpuCost`) for the whole call, the split pass alone, the engine's
    /// decode kernel at the same shapes, a no-op kernel with the split pass's
    /// bindings and a trivial copy. Opt-in: `INKLING2_ATTN_LATENCY=1`.
    /// GPU time per call with one or two simdgroups per partition (same
    /// bits), interleaved per configuration, with the thermal state recorded
    /// and an idle pause between configurations. Opt-in:
    /// `INKLING2_ATTN_PSUB=1`; `INKLING2_ATTN_PSUB_SHAPES` lists `BxL` shapes,
    /// `INKLING2_ATTN_PSUB_KINDS` the layer kinds, `INKLING2_ATTN_IDLE_SECONDS`
    /// the pause (default 60).
    func testDecodeSubPartitions() throws {
        guard ProcessInfo.processInfo.environment["INKLING2_ATTN_PSUB"] == "1" else {
            throw XCTSkip("set INKLING2_ATTN_PSUB=1")
        }
        MLX.GPU.set(memoryLimit: 12 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let pool = makePool(pages: 16_384)
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        let idle = Double(
            ProcessInfo.processInfo.environment["INKLING2_ATTN_IDLE_SECONDS"] ?? "60")!
        let shapes =
            (ProcessInfo.processInfo.environment["INKLING2_ATTN_PSUB_SHAPES"]
            ?? "1x1,2x1,4x1,8x1,16x1,32x1,1x8").split(separator: ",").map { shape in
                let parts = shape.split(separator: "x").compactMap { Int($0) }
                return (b: parts[0], l: parts[1])
            }
        let warm = MLXRandom.normal([4096, 4096])
        for _ in 0 ..< 20 { eval(matmul(warm, warm)) }
        // `INKLING2_ATTN_PSUB_KINDS`: sliding, full or both (default).
        let kinds =
            (ProcessInfo.processInfo.environment["INKLING2_ATTN_PSUB_KINDS"] ?? "sliding,full")
            .split(separator: ",")
        let layers = ([(1024, 8192), (nil, 8192)] as [(Int?, Int)]).filter {
            kinds.contains($0.0 == nil ? "full" : "sliding")
        }
        var first = true
        for (window, ctx) in layers {
            for (b, l) in shapes {
                if !first { Thread.sleep(forTimeInterval: idle) }
                first = false
                let (table, tableLength) = tables(
                    rows: b, tokens: ctx + l, window: window, pool: pool)
                let q0 = MLXRandom.normal([b, l, qh, d]).asType(.bfloat16)
                let kNew = MLXRandom.normal([b, l, kvh, d]).asType(.bfloat16)
                let vNew = MLXRandom.normal([b, l, kvh, d]).asType(.bfloat16)
                let states = (MLXRandom.normal([b, l, qh, 16]) * 0.5).asType(.bfloat16)
                let sinks = MLXRandom.normal([b, l, qh])
                let fence = MLXArray.zeros([1], dtype: .int32)
                let rows = Array(
                    repeating: Attention.Row(queryStart: ctx, tableLength: tableLength), count: b)
                let plan = Attention.Plan(rows: rows, queries: l, window: window, scale: scale)
                eval(table, q0, kNew, vNew, states, sinks, proj, plan.rowInfo)
                func cost(_ psub: Int) -> (encode: Double, gpu: Double) {
                    gpuCost { n in
                        var q = q0
                        for _ in 0 ..< n {
                            q = Attention.attend(
                                queries: q, newKeys: kNew, newValues: vNew,
                                relativeStates: states, proj: proj, sinks: sinks,
                                cache: .init(
                                    keys: pool.k, values: pool.v, tables: table, writeFence: fence),
                                plan: plan, layout: .tokenMajor, subPartitions: psub
                            ).output.reshaped(b, l, qh, d)
                        }
                        return q
                    }
                }
                // Same bits either way.
                let one = Attention.attend(
                    queries: q0, newKeys: kNew, newValues: vNew, relativeStates: states,
                    proj: proj, sinks: sinks,
                    cache: .init(keys: pool.k, values: pool.v, tables: table, writeFence: fence),
                    plan: plan, layout: .tokenMajor, subPartitions: 1
                ).output
                let two = Attention.attend(
                    queries: q0, newKeys: kNew, newValues: vNew, relativeStates: states,
                    proj: proj, sinks: sinks,
                    cache: .init(keys: pool.k, values: pool.v, tables: table, writeFence: fence),
                    plan: plan, layout: .tokenMajor, subPartitions: 2
                ).output
                XCTAssertTrue(arrayEqual(one, two).item(Bool.self), "B=\(b) L=\(l) bits")
                var best: [Int: (encode: Double, gpu: Double)] = [
                    1: (encode: Double.infinity, gpu: Double.infinity),
                    2: (Double.infinity, Double.infinity),
                ]
                var thermal: [Int] = []
                for psub in [1, 2, 1, 2] {
                    thermal.append(ProcessInfo.processInfo.thermalState.rawValue)
                    let c = cost(psub)
                    best[psub] = (min(best[psub]!.encode, c.encode), min(best[psub]!.gpu, c.gpu))
                }
                thermal.append(ProcessInfo.processInfo.thermalState.rawValue)
                let automatic = Attention.subPartitionCount(
                    requested: nil, b: b, kvHeads: kvh, queries: l, gqa: qh / kvh,
                    maxPartitions: plan.maxPartitions, headDim: d, elementSize: 2)
                record([
                    "phase": "decode-subpartitions", "kind": window == nil ? "full" : "sliding",
                    "ctx": ctx, "B": b, "L": l, "partitions": plan.maxPartitions,
                    "psub1_gpu_us": best[1]!.gpu, "psub2_gpu_us": best[2]!.gpu,
                    "psub1_encode_us": best[1]!.encode, "psub2_encode_us": best[2]!.encode,
                    "automatic_psub": automatic, "thermal_states": thermal,
                    "discard": thermal.contains { $0 >= 2 },
                ])
            }
        }
    }

    func testDecodeGPUTime() throws {
        guard ProcessInfo.processInfo.environment["INKLING2_ATTN_LATENCY"] == "1" else {
            throw XCTSkip("set INKLING2_ATTN_LATENCY=1")
        }
        MLX.GPU.set(memoryLimit: 12 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let pool = makePool(pages: 16_384)
        let source = try PagedAttentionResources.loadSourceForCurrentProcess()
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        let copy = MLXFast.metalKernel(
            name: "inkling2_gpu_floor", inputNames: ["x"], outputNames: ["y"],
            source: "uint i = thread_position_in_grid.x; y[i] = x[i];")
        let noop = MLXFast.metalKernel(
            name: "inkling2_gpu_noop",
            inputNames: [
                "q", "rel", "proj", "knew", "vnew", "kcache", "vcache", "tables", "rowinfo",
                "wfence",
            ],
            outputNames: ["partials", "fence"],
            source: """
                if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
                    && thread_position_in_grid.z == 0) {
                    fence[0] = wfence[0] + 1 + (q_shape[1] + knew_shape[2] + tables_shape[1] < 0);
                }
                """,
            header: Inkling2PagedAttentionMetal.header)
        let batches =
            (ProcessInfo.processInfo.environment["INKLING2_ATTN_LATENCY_BATCHES"]
            ?? "1,8").split(separator: ",").compactMap { Int($0) }
        let warm = MLXRandom.normal([4096, 4096])
        for _ in 0 ..< 20 { eval(matmul(warm, warm)) }
        for (window, ctx) in [(1024, 8192), (nil, 8192)] as [(Int?, Int)] {
            for b in batches {
                let (table, tableLength) = tables(
                    rows: b, tokens: ctx + 1, window: window, pool: pool)
                let q0 = MLXRandom.normal([b, 1, qh, d]).asType(.bfloat16)
                let kNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
                let vNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
                let states = (MLXRandom.normal([b, 1, qh, 16]) * 0.5).asType(.bfloat16)
                let sinks = MLXRandom.normal([b, 1, qh])
                let fence = MLXArray.zeros([1], dtype: .int32)
                let rows = Array(
                    repeating: Attention.Row(queryStart: ctx, tableLength: tableLength), count: b)
                let plan = Attention.Plan(rows: rows, queries: 1, window: window, scale: scale)
                eval(table, q0, kNew, vNew, states, sinks, proj, plan.rowInfo)
                func call(_ q: MLXArray, _ f: MLXArray) -> Attention.Output {
                    Attention.attend(
                        queries: q, newKeys: kNew, newValues: vNew, relativeStates: states,
                        proj: proj, sinks: sinks,
                        cache: .init(keys: pool.k, values: pool.v, tables: table, writeFence: f),
                        plan: plan, layout: .tokenMajor)
                }
                let whole = gpuCost { n in
                    var q = q0
                    for _ in 0 ..< n { q = call(q, fence).output.reshaped(b, 1, qh, d) }
                    return q
                }
                let split = gpuCost { n in
                    var f = fence
                    for _ in 0 ..< n { f = call(q0, f).writeFence }
                    return f
                }
                let start = window.map { max(0, ctx - $0 + 1) } ?? 0
                let (seqinfo, maxLength) = PagedAttentionKernel.seqinfo(
                    Array(
                        repeating: PagedAttentionKernel.SeqInfoRow(
                            attendStart: start, attendLength: ctx + 1 - start,
                            tableLength: tableLength), count: b))
                let params = MLXArray([1, scale, 0, 0, 0, 0, 0, 0] as [Float])
                let engine = gpuCost { n in
                    var q = q0.reshaped(b, qh, d)
                    for _ in 0 ..< n {
                        q = try! PagedAttentionKernel.decode(
                            queries: q, kSlab: pool.k, vSlab: pool.v, tables: table,
                            seqinfo: seqinfo, maxAttendLength: maxLength, sinks: nil,
                            params: params, softcap: false, pageSize: s, writeFence: fence,
                            kernelSource: source
                        ).out
                    }
                    return q
                }
                let parts = plan.maxPartitions
                let noopCost = gpuCost { n in
                    var f = fence
                    for _ in 0 ..< n {
                        f =
                            noop(
                                [
                                    q0, states, proj, kNew, vNew, pool.k, pool.v, table,
                                    plan.rowInfo, f,
                                ], grid: (kvh * 32, b, parts), threadGroup: (32, 1, 1),
                                outputShapes: [[b, qh, 1, parts, d + 2], [1]],
                                outputDTypes: [.float32, .int32])[1]
                    }
                    return f
                }
                let floor = gpuCost { n in
                    var x = q0.reshaped(-1)
                    for _ in 0 ..< n {
                        x =
                            copy(
                                [x], grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
                                outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
                    }
                    return x
                }
                record([
                    "phase": "decode-gpu-time", "kind": window == nil ? "full" : "sliding",
                    "B": b, "ctx": ctx, "partitions": parts,
                    "inkling2_gpu_us": whole.gpu, "inkling2_encode_us": whole.encode,
                    "split_only_gpu_us": split.gpu, "engine_gpu_us": engine.gpu,
                    "engine_encode_us": engine.encode, "noop_split_bindings_gpu_us": noopCost.gpu,
                    "copy_floor_gpu_us": floor.gpu,
                ])
            }
        }
    }

    /// Per-call cost of the integrated path (kernel core through
    /// `PagedLayerCache.attendWithModelKernel`, segmented pool, fused decode
    /// writes) over a step of 10 sliding and 2 full layers, each call feeding
    /// the next: graph building, CPU encoding, chained time under the decode
    /// limits, and GPU time (one command buffer). Opt-in:
    /// `INKLING2_ATTN_LATENCY=1`; `INKLING2_ATTN_LATENCY_BATCHES` (default `1,8`).
    func testIntegratedDecodeCallCost() throws {
        guard ProcessInfo.processInfo.environment["INKLING2_ATTN_LATENCY"] == "1" else {
            throw XCTSkip("set INKLING2_ATTN_LATENCY=1")
        }
        MLX.GPU.set(memoryLimit: 12 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let batches =
            (ProcessInfo.processInfo.environment["INKLING2_ATTN_LATENCY_BATCHES"]
            ?? "1,8").split(separator: ",").compactMap { Int($0) }
        let sliding = CBv2LayerKind(
            attention: .slidingWindow(1024), headDim: d, kvHeads: kvh, queryHeads: qh)
        let full = CBv2LayerKind(attention: .full, headDim: d, kvHeads: kvh, queryHeads: qh)
        let kinds = Array(repeating: sliding, count: 10) + [full, full]
        let history = 8192
        // `INKLING2_ATTN_PROFILE_SECONDS`: run the chain that long first (for a
        // sampling profiler), with room in the rows for it.
        let profileSeconds = Double(
            ProcessInfo.processInfo.environment["INKLING2_ATTN_PROFILE_SECONDS"] ?? "0")!
        let headroom = profileSeconds > 0 ? 32_768 : 2048
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        let warm = MLXRandom.normal([4096, 4096])
        for _ in 0 ..< 20 { eval(matmul(warm, warm)) }
        for b in batches {
            let backend = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: 4 << 30, dtype: .bfloat16, maxPrefillChunk: 512,
                    nominalMaxSequenceLength: history + headroom, segmentSizeBytes: 64 << 20))
            var states: [[CBv2SequenceKV?]] = []
            for _ in 0 ..< b {
                states.append(
                    try backend.makeSequenceState(
                        layerKinds: kinds, promptLength: history, maxLength: history + headroom))
            }
            defer { for state in states { backend.release(state) } }
            let chunk = (MLXRandom.normal([kvh, 512, d]) * 0.6).asType(.bfloat16)
            eval(chunk)
            for state in states {
                for layer in kinds.indices {
                    let row = try XCTUnwrap(state[layer] as? PagedSequenceKV)
                    for _ in 0 ..< history / 512 { row.write(keys: chunk, values: chunk) }
                }
            }
            let caches = backend.makeLayerCaches()
            for layer in kinds.indices { caches[layer].setRows(states.map { $0[layer]! }) }
            let q0 = MLXRandom.normal([b, 1, qh, d]).asType(.bfloat16)
            let kNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
            let vNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
            let rel = (MLXRandom.normal([b, 1, qh, 16]) * 0.5).asType(.bfloat16)
            let sinks = MLXRandom.normal([b, 1, qh])
            eval(q0, kNew, vNew, rel, sinks, proj)
            let core = Inkling2KernelAttentionCore()
            var layer = 0
            func chain(_ n: Int) -> MLXArray {
                var q = q0
                for _ in 0 ..< n {
                    let cache = caches[layer]
                    let row = states[0][layer] as! PagedSequenceKV
                    let start = row.absoluteOffset
                    let inputs = Inkling2AttentionInputs(
                        queries: q.transposed(0, 2, 1, 3), keys: kNew.transposed(0, 2, 1, 3),
                        values: vNew.transposed(0, 2, 1, 3), relativeLogits: q0,
                        relativeStates: rel, relativeProjection: proj,
                        sinks: sinks.transposed(0, 2, 1),
                        positions: MLXArray.zeros([b, 1], dtype: .int32),
                        window: kinds[layer].attention == .full ? nil : 1024,
                        scale: scale)
                    q = core.attend(
                        slot: layer, cache: cache, rowOffsets: Array(repeating: start, count: b),
                        inputs
                    ).reshaped(b, 1, qh, d)
                    layer = (layer + 1) % kinds.count
                }
                return q
            }
            if profileSeconds > 0 {
                print("[Inkling2AttnBench] profiling pid \(getpid()) for \(profileSeconds) s")
                let end = DispatchTime.now().uptimeNanoseconds + UInt64(profileSeconds * 1e9)
                while DispatchTime.now().uptimeNanoseconds < end {
                    let out = chain(120)
                    GPU.withCommandBufferLimits(decodeLimits) { asyncEval(out) }
                    Stream.gpu.synchronize()
                }
            }
            let cpu = chainCost(chain)
            let gpu = gpuCost(chain)
            try backend.pool.writeValidation.check()
            record([
                "phase": "integrated-decode-call", "B": b, "history": history,
                "layers": "10 sliding + 2 full", "host_us": cpu.host, "encode_us": cpu.encode,
                "call_us": cpu.total, "gpu_us": gpu.gpu,
            ])
        }
    }

    /// One decode step's attention over Inkling 2's 48 applications (40 sliding,
    /// 8 full) for B streams of 4K history on the serving segmented pool (256 MB
    /// segments shared by the layers of a group), pages allocated interleaved
    /// across streams and layers as decode does. Reports the per-step attention
    /// time (GPU, CPU) and how many calls fell back to the gathered copy.
    /// Opt-in: `INKLING2_ATTN_STREAMS` (e.g. `8,16,32`).
    func testDecodeStepManyStreams() throws {
        guard let list = ProcessInfo.processInfo.environment["INKLING2_ATTN_STREAMS"] else {
            throw XCTSkip("set INKLING2_ATTN_STREAMS")
        }
        MLX.GPU.set(memoryLimit: 40 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let sliding = CBv2LayerKind(
            attention: .slidingWindow(1024), headDim: d, kvHeads: kvh, queryHeads: qh)
        let full = CBv2LayerKind(attention: .full, headDim: d, kvHeads: kvh, queryHeads: qh)
        // Six blocks of five sliding and one full layer, then the rest: 40 + 8.
        var kinds: [CBv2LayerKind] = []
        for i in 0 ..< 48 { kinds.append(i % 6 == 5 ? full : sliding) }
        let history = Int(
            ProcessInfo.processInfo.environment["INKLING2_ATTN_STREAMS_HISTORY"] ?? "4096")!
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        // 1: segment buckets past the binding limit (default); 0: the gathered
        // copy the seam used before.
        let modes = (ProcessInfo.processInfo.environment["INKLING2_ATTN_STREAMS_BUCKETS"] ?? "1")
            .split(separator: ",").map { $0 == "1" }
        for b in list.split(separator: ",").compactMap({ Int($0) }) {
            let backend = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: b * (360 << 20) + (1 << 30), dtype: .bfloat16,
                    maxPrefillChunk: 512, nominalMaxSequenceLength: history + 256,
                    segmentSizeBytes: 256 << 20))
            var states: [[CBv2SequenceKV?]] = []
            for _ in 0 ..< b {
                states.append(
                    try backend.makeSequenceState(
                        layerKinds: kinds, promptLength: history, maxLength: history + 256))
            }
            defer { for state in states { backend.release(state) } }
            print(
                "[Inkling2AttnBench] setup B=\(b) states mlx_active_mb=\(Memory.activeMemory >> 20)"
            )
            fflush(stdout)
            let chunk = (MLXRandom.normal([kvh, 64, d]) * 0.6).asType(.bfloat16)
            eval(chunk)
            // Interleaved: every stream and layer takes 64 tokens in turn; both
            // groups' write chains are evaluated every round.
            let fences = {
                [0, 5].map { backend.pool.group(backend.pool.groupKey(forLayer: $0)).writeFence }
            }
            for round in 0 ..< history / 64 {
                for state in states {
                    for layer in kinds.indices {
                        (state[layer] as! PagedSequenceKV).write(keys: chunk, values: chunk)
                    }
                }
                eval(fences())
                if round % 16 == 0 {
                    print(
                        "[Inkling2AttnBench] setup B=\(b) round \(round) mlx_active_mb="
                            + "\(Memory.activeMemory >> 20) mlx_peak_mb=\(Memory.peakMemory >> 20)")
                    fflush(stdout)
                }
            }
            let caches = backend.makeLayerCaches()
            for layer in kinds.indices { caches[layer].setRows(states.map { $0[layer]! }) }
            let q0 = MLXRandom.normal([b, 1, qh, d]).asType(.bfloat16)
            let kNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
            let vNew = MLXRandom.normal([b, 1, kvh, d]).asType(.bfloat16)
            let rel = (MLXRandom.normal([b, 1, qh, 16]) * 0.5).asType(.bfloat16)
            let sinks = MLXRandom.normal([b, 1, qh])
            eval(q0, kNew, vNew, rel, sinks, proj)
            let core = Inkling2KernelAttentionCore()
            func steps(_ n: Int) -> MLXArray {
                var q = q0
                for _ in 0 ..< n {
                    for layer in kinds.indices {
                        let row = states[0][layer] as! PagedSequenceKV
                        let inputs = Inkling2AttentionInputs(
                            queries: q.transposed(0, 2, 1, 3), keys: kNew.transposed(0, 2, 1, 3),
                            values: vNew.transposed(0, 2, 1, 3), relativeLogits: q0,
                            relativeStates: rel, relativeProjection: proj,
                            sinks: sinks.transposed(0, 2, 1),
                            positions: MLXArray.zeros([b, 1], dtype: .int32),
                            window: kinds[layer].attention == .full ? nil : 1024, scale: scale)
                        q = core.attend(
                            slot: layer, cache: caches[layer],
                            rowOffsets: Array(repeating: row.absoluteOffset, count: b), inputs
                        ).reshaped(b, 1, qh, d)
                    }
                }
                return q
            }
            for bucketed in modes {
                core.bucketing = bucketed ? Inkling2KernelAttentionCore.defaultBucketing : nil
                eval(steps(1))
                print(
                    "[Inkling2AttnBench] warm B=\(b) mlx_active_mb=\(Memory.activeMemory >> 20) "
                        + "mlx_peak_mb=\(Memory.peakMemory >> 20)"
                )
                fflush(stdout)
                PagedLayerCache.modelAttentionCounters.reset()
                func best(_ n: Int) -> (host: Double, encode: Double, gpu: Double) {
                    var r = (host: Double.infinity, encode: Double.infinity, gpu: Double.infinity)
                    for _ in 0 ..< 3 {
                        Stream.gpu.synchronize()
                        let t0 = DispatchTime.now().uptimeNanoseconds
                        let out = steps(n)
                        let t1 = DispatchTime.now().uptimeNanoseconds
                        GPU.withCommandBufferLimits(.init(operations: 1 << 30, megabytes: 1 << 20))
                        {
                            asyncEval(out)
                        }
                        let t2 = DispatchTime.now().uptimeNanoseconds
                        Stream.gpu.synchronize()
                        let t3 = DispatchTime.now().uptimeNanoseconds
                        r.host = min(r.host, Double(t1 - t0) / 1e6)
                        r.encode = min(r.encode, Double(t2 - t1) / 1e6)
                        r.gpu = min(r.gpu, Double(t3 - t2) / 1e6)
                    }
                    return r
                }
                let one = best(1)
                let three = best(3)
                let counts = PagedLayerCache.modelAttentionCounters.snapshot
                func ms(_ a: Double, _ c: Double) -> Double { ((c - a) / 2 * 100).rounded() / 100 }
                try backend.pool.writeValidation.check()
                let group = backend.pool.group(backend.pool.groupKey(forLayer: 5))
                record([
                    "phase": "decode-step-streams", "B": b, "history": history,
                    "mode": bucketed ? "segment-buckets" : "gathered",
                    "step_host_ms": ms(one.host, three.host),
                    "step_encode_ms": ms(one.encode, three.encode),
                    "step_gpu_ms": ms(one.gpu, three.gpu), "calls": counts.calls,
                    "gathered_calls": counts.gathered, "bucket_dispatches": counts.buckets,
                    "full_group_segments": group.segments.count,
                ])
            }
        }
    }
}
