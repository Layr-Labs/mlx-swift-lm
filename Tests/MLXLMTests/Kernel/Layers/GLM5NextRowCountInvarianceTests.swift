// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Opt-in (`GLM53_ROW_COUNT_PROBE=1`) diagnostic for tensor-parallel
    /// prefill micro-batches: does a row's arithmetic depend on how many rows
    /// share its call? (Synthetic data: an op whose result depends on the
    /// cut only near ties passes here and still differs on the model.) One TP2 rank's KDA sublayer, DSA attention, MoE layer
    /// and router (serving members, synthetic 4-bit weights at rank geometry)
    /// run the same 8,192 rows as one call and as consecutive calls carrying
    /// the cache. Prints `GLM53_ROW_COUNT <op> <rows> bitwise=… differing=…
    /// max_abs=…` per partition; asserts nothing. All bitwise would make larger
    /// or uniform micro-batches a schedule change rather than a numerics one.
    @Suite(
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["GLM53_ROW_COUNT_PROBE"] != nil))
    struct GLM5NextRowCountInvarianceTests {
        static let tokens = 8192
        static let partitions: [[Int]] = [
            [8192], [4096, 4096], [2048, 2048, 2048, 2048], [3766, 4426], [1000, 7192],
        ]

        static func report(_ op: String, _ reference: MLXArray, _ value: MLXArray, _ rows: [Int]) {
            let a = reference.asType(.float32)
            let b = value.asType(.float32)
            let differing = (a .!= b).asType(.int32).sum().item(Int32.self)
            let maxAbs = abs(a - b).max().item(Float.self)
            let raw: DType = reference.dtype.size == 2 ? .uint16 : .uint32
            let bitwise =
                reference.shape == value.shape
                && (reference.view(dtype: raw) .== value.view(dtype: raw)).all().item(Bool.self)
            print(
                "GLM53_ROW_COUNT \(op) rows=\(rows.map(String.init).joined(separator: "+")) bitwise=\(bitwise ? 1 : 0)"
                    + " differing=\(differing)/\(reference.size) max_abs=\(maxAbs)")
        }

        /// `run` over consecutive pieces of `x` (cache carried by the caller's closure), concatenated.
        static func pieces(_ x: MLXArray, _ rows: [Int], _ run: (MLXArray) throws -> MLXArray)
            rethrows -> MLXArray
        {
            var start = 0
            var outputs: [MLXArray] = []
            for count in rows {
                let output = try run(x[0..., start ..< start + count])
                eval(output)
                outputs.append(output)
                start += count
            }
            return concatenated(outputs, axis: 1)
        }

        @Test func kdaSublayer() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextKDALayerFixture(heads: 32, hidden: 4096)
                let layer = try GLM5NextKDASublayer(
                    fixture.text, prefix: GLM5NextKDALayerFixture.prefix,
                    loader: GLM5NextTensorLoader(
                        fixture.inventory, fixture.tensor, releaseSourceTensors: { _ in }),
                    linearAttention: .nativeRecurrenceChunkedPrefill, decodeStep: .fusedEveryStep,
                    projections: .fusedInputs,
                    prefillProjections: GLM5NextExecutionPolicy.serving.prefillProjections)
                let x = fixture.input(tokens: Self.tokens, seed: 7)
                eval(x)
                var reference: (MLXArray, MLXArray)?
                for rows in Self.partitions {
                    let cache = GLM5NextKDACache()
                    let output = try Self.pieces(x, rows) { piece in
                        try layer.forward(
                            piece, cache: cache,
                            validity: MLXArray.ones([1, piece.dim(1)], dtype: .bool))
                    }
                    let state = cache.state[1]
                    eval(output, state)
                    if let reference {
                        Self.report("kda_out", reference.0, output, rows)
                        Self.report("kda_state", reference.1, state, rows)
                    } else {
                        reference = (output, state)
                    }
                }
            }
        }

        @Test func dsaAttention() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextDSARankFixture()
                let dsa = try fixture.attention(policy: .serving)
                let x = fixture.input(tokens: Self.tokens, seed: 11)
                let history = fixture.input(tokens: 4096, seed: 12)
                eval(x, history)
                for start in [0, 4096] {
                    var reference: MLXArray?
                    for rows in Self.partitions {
                        let cache = GLM5NextDSACache()
                        if start > 0 {
                            _ = try dsa.forward(
                                history, cache: cache,
                                validity: MLXArray.ones([1, 4096], dtype: .bool))
                            eval(cache.state)
                        }
                        let output = try Self.pieces(x, rows) { piece in
                            try dsa.forward(
                                piece, cache: cache,
                                validity: MLXArray.ones([1, piece.dim(1)], dtype: .bool))
                        }
                        eval(output)
                        if let reference {
                            Self.report("dsa_out_h\(start)", reference, output, rows)
                        } else {
                            reference = output
                        }
                    }
                }
            }
        }

        /// Median ms of `body` (synchronized), after one warm-up.
        static func time(_ body: () throws -> MLXArray) rethrows -> Double {
            eval(try body())
            Stream.gpu.synchronize()
            var samples: [Double] = []
            for _ in 0 ..< 5 {
                let start = ContinuousClock.now
                eval(try body())
                Stream.gpu.synchronize()
                let elapsed = ContinuousClock.now - start
                samples.append(
                    Double(elapsed.components.seconds) * 1000 + Double(
                        elapsed.components.attoseconds) / 1e15)
            }
            return samples.sorted()[2]
        }

        /// Per-op ms for N rows as one call, 2 and 4 pieces (the micro-batch
        /// sizes the schedules use), with `GLM53_ROW_COUNT_TIMING=N` (8,192 if not a count).
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_ROW_COUNT_TIMING"] != nil))
        func timings() throws {
            try Device.withDefaultDevice(.gpu) {
                typealias Decode = GLM5NextMoEDecodeKernelTests
                let kdaFixture = try GLM5NextKDALayerFixture(heads: 32, hidden: 4096)
                let kda = try GLM5NextKDASublayer(
                    kdaFixture.text, prefix: GLM5NextKDALayerFixture.prefix,
                    loader: GLM5NextTensorLoader(
                        kdaFixture.inventory, kdaFixture.tensor, releaseSourceTensors: { _ in }),
                    linearAttention: .nativeRecurrenceChunkedPrefill, decodeStep: .fusedEveryStep,
                    projections: .fusedInputs,
                    prefillProjections: GLM5NextExecutionPolicy.serving.prefillProjections)
                let dsaFixture = try GLM5NextDSARankFixture()
                let dsa = try dsaFixture.attention(policy: .serving)
                let moe = try Decode.moe(.tensorParallelRank, fusedSharedGateUp: true, nvfp4: true)
                moe.gate.usesNativeKernel = true
                // `GLM53_ROW_COUNT_TIMING=N`: N rows as one call, 2 and 4 pieces.
                let tokens = max(
                    1024,
                    Int(ProcessInfo.processInfo.environment["GLM53_ROW_COUNT_TIMING"] ?? "")
                        ?? Self.tokens)
                let x = kdaFixture.input(tokens: tokens, seed: 7)
                let history = dsaFixture.input(tokens: 4096, seed: 12)
                eval(x, history)
                let warmDSA = GLM5NextDSACache()
                _ = try dsa.forward(
                    history, cache: warmDSA, validity: MLXArray.ones([1, 4096], dtype: .bool))
                eval(warmDSA.state)
                for rows in [
                    [tokens], [tokens / 2, tokens - tokens / 2],
                    [tokens / 4, tokens / 4, tokens / 4, tokens - 3 * (tokens / 4)],
                ] {
                    let label = rows.map(String.init).joined(separator: "+")
                    let kdaMs = try Self.time {
                        let cache = GLM5NextKDACache()
                        return try Self.pieces(x, rows) { piece in
                            try kda.forward(
                                piece, cache: cache,
                                validity: MLXArray.ones([1, piece.dim(1)], dtype: .bool))
                        }
                    }
                    let dsaMs = try Self.time {
                        let cache = warmDSA.copy() as! GLM5NextDSACache
                        return try Self.pieces(x, rows) { piece in
                            try dsa.forward(
                                piece, cache: cache,
                                validity: MLXArray.ones([1, piece.dim(1)], dtype: .bool))
                        }
                    }
                    let moeMs = try Self.time {
                        try Self.pieces(x, rows) {
                            try moe(
                                $0, routedExperts: GLM5NextExecutionPolicy.serving.routedExperts)
                        }
                    }
                    let routerMs = try Self.time {
                        try Self.pieces(x, rows) {
                            try moe.gate.route($0).logits.reshaped(1, $0.dim(1), -1)
                        }
                    }
                    print(
                        String(
                            format:
                                "GLM53_ROW_COUNT_TIMING rows=%@ kda=%.2f dsa_h4096=%.2f moe_nvfp4=%.2f router=%.2f ms (%d rows; pieces evaluated one by one)",
                            label, kdaMs, dsaMs, moeMs, routerMs, tokens))
                }
            }
        }

        /// One prefill mHC site (folded expand + normalized collapse, the
        /// serving form) at hidden 4,096: ms and implied bytes per second, with
        /// `GLM53_ROW_COUNT_TIMING` set.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_ROW_COUNT_TIMING"] != nil))
        func hyperConnectionSite() throws {
            typealias Kernel = GLM5NextHyperConnectionKernel
            try Device.withDefaultDevice(.gpu) {
                let width = 4096
                for rows in [4096, 16384] {
                    let keys = MLXRandom.split(key: MLXRandom.key(41), into: 7)
                    let streams = MLXRandom.normal([1, rows, 4, width], key: keys[0]).asType(
                        .bfloat16)
                    let output = MLXRandom.normal([1, rows, width], key: keys[1]).asType(.bfloat16)
                    let fn = (MLXRandom.normal([24, 4 * width], key: keys[2]) * 0.05).asType(
                        .bfloat16)
                    let base = MLXRandom.normal([24], key: keys[3]) * 0.5
                    let scale = MLXArray([Float(0.75), 1.25, 1])
                    let weight = MLXRandom.uniform(
                        low: Float(0.5), high: 1.5, [width], key: keys[4]
                    ).asType(.bfloat16)
                    let post = MLXRandom.uniform(
                        low: Float(0), high: 2, [1, rows, 4], key: keys[5])
                    let comb = MLXRandom.uniform(
                        low: Float(0), high: 0.5, [1, rows, 4, 4], key: keys[6])
                    eval(streams, output, fn, base, scale, weight, post, comb)
                    func site(_ source: Kernel.StreamsSource) throws -> MLXArray {
                        let result = try Kernel.collapseNormalized(
                            source, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                            epsilon: 1e-6,
                            sinkhornIterations: 20, normWeight: weight, normEpsilon: 1e-5)
                        // One launch writes every output; evaluating one runs it.
                        return result.normalized
                    }
                    let folded = try Self.time {
                        try site(.expand(output: output, residual: streams, post: post, comb: comb))
                    }
                    let plain = try Self.time { try site(.streams(streams)) }
                    let copy = try Self.time { (streams * 1).reshaped(-1) }
                    // Minimum bytes: read output + residual, write streams + normalized (folded).
                    let bytes = Double(rows * width) * 2 * (1 + 4 + 4 + 1)
                    print(
                        String(
                            format:
                                "GLM53_HC_SITE rows=%d folded=%.3f ms (%.0f GB/s min-bytes) plain=%.3f ms streams_copy=%.3f ms (%.0f GB/s)",
                            rows, folded, bytes / folded / 1e6, plain, copy,
                            Double(rows * width * 4) * 2 * 2 / copy / 1e6))
                }
            }
        }

        /// The DSA selection (serving execution, native prefill scores) of a
        /// step's sparse rows against the same history, as one call and as
        /// its two halves, and the router over halves of one micro-batch.
        /// Random scores rarely tie: the selection can still differ on real
        /// data (its score blocks round by how rows are cut; the pair's
        /// 8,192-row micro-batches moved the model's perplexity 0.8 %).
        @Test func selectionAndRouterHalves() throws {
            try Device.withDefaultDevice(.gpu) {
                let execution = GLM5NextExecutionPolicy.serving.sparseSelection
                let keys = MLXRandom.split(key: MLXRandom.key(77), into: 6)
                for (start, rows) in [
                    (2048, 2048), (4096, 4096), (12288, 4096), (20480, 8192), (2048, 6144),
                ] {
                    let history = start + rows
                    let queries = (MLXRandom.normal([1, rows, 32, 128], key: keys[0]) * 0.5).asType(
                        .bfloat16)
                    let indexKeys = MLXRandom.normal([1, history, 128], key: keys[1]).asType(
                        .bfloat16)
                    let gates = MLXRandom.normal([1, history, 128], key: keys[2]).asType(.bfloat16)
                    let bias = (MLXRandom.normal([4, 128], key: keys[3]) * 0.1).asType(.bfloat16)
                    let weights = (MLXRandom.normal([1, rows, 32], key: keys[4]) * 0.2).asType(
                        .bfloat16)
                    let keyValidity = MLXArray.ones([1, history], dtype: .bool)
                    eval(queries, indexKeys, gates, bias, weights)
                    func select(_ range: Range<Int>) throws -> MLXArray {
                        try GLM5NextSparseAttention.selectTokens(
                            indexQueries: queries[0..., range], indexKeys: indexKeys,
                            poolGates: gates,
                            poolPositionBias: bias, headWeights: weights[0..., range],
                            keyValidity: keyValidity,
                            queryValidity: MLXArray.ones([1, range.count], dtype: .bool),
                            queryStart: start + range.lowerBound, tokenBudget: 2048,
                            execution: execution,
                            nativePrefillScores: range.count
                                > GLM5NextSparseDecodeKernel.maximumQueries
                        ).indices
                    }
                    let whole = try select(0 ..< rows)
                    let half = rows / 2
                    let split = concatenated(
                        [try select(0 ..< half), try select(half ..< rows)], axis: 1)
                    eval(whole, split)
                    Self.report(
                        "selection_h\(start)", whole.asType(.float32), split.asType(.float32),
                        [half, rows - half])
                }
                typealias Decode = GLM5NextMoEDecodeKernelTests
                let moe = try Decode.moe(.tensorParallelRank, fusedSharedGateUp: true, nvfp4: true)
                moe.gate.usesNativeKernel = true
                for rows in [2048, 2828, 4096, 8192, 16384] {
                    let x = Decode.input(.tensorParallelRank, tokens: rows, seed: 23)
                    let whole = try moe.gate.route(x)
                    let half = rows / 2
                    let a = try moe.gate.route(x[0..., ..<half])
                    let b = try moe.gate.route(x[0..., half...])
                    func flat(_ r: (logits: MLXArray, weights: MLXArray, indices: MLXArray))
                        -> MLXArray
                    {
                        concatenated([
                            r.weights.reshaped(-1).asType(.float32),
                            r.indices.reshaped(-1).asType(.float32),
                        ])
                    }
                    let joined = concatenated(
                        [a.weights.reshaped(-1, 8), b.weights.reshaped(-1, 8)], axis: 0)
                    let joinedIndices = concatenated(
                        [a.indices.reshaped(-1, 8), b.indices.reshaped(-1, 8)], axis: 0)
                    eval(whole.weights, whole.indices, joined, joinedIndices)
                    Self.report(
                        "router_weights_halves", whole.weights.reshaped(-1, 8), joined,
                        [half, rows - half])
                    Self.report(
                        "router_indices_halves", whole.indices.reshaped(-1, 8).asType(.float32),
                        joinedIndices.asType(.float32), [half, rows - half])
                    _ = flat
                }
            }
        }

        /// NVFP4 codes, E4M3 block scales and a global scale (the dense layers' format).
        static func nvfp4Linear(rows: Int, inputs: Int, seed: UInt64) throws -> GLM5NextNVFP4Linear
        {
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 5)
            let high = MLXRandom.randInt(Int32(0) ..< 65536, [rows, inputs / 8], key: keys[0])
                .asType(.uint32)
            let low = MLXRandom.randInt(Int32(0) ..< 65536, [rows, inputs / 8], key: keys[1])
                .asType(.uint32)
            let weight = bitwiseOr(leftShift(high, MLXArray(UInt32(16))), low)
            let exponent = MLXRandom.randInt(Int32(4) ..< 7, [rows, inputs / 16], key: keys[2])
            let mantissa = MLXRandom.randInt(Int32(0) ..< 8, [rows, inputs / 16], key: keys[3])
            let scales = (exponent * 8 + mantissa).asType(.uint8)
            let target = 1 / (Float(inputs).squareRoot() * 2.93 * 0.25)
            let global = MLXRandom.uniform(
                low: 0.75 * target, high: 1.25 * target, [1], key: keys[4])
            eval(weight, scales, global)
            return try GLM5NextNVFP4Linear(weight: weight, scales: scales, globalScale: global)
        }

        /// The dense MLP (one rank: 6,144 of 12,288 channels, NVFP4) and the
        /// final readout (stream mean, fused RMSNorm) over the same rows.
        @Test func denseMLPAndReadout() throws {
            try Device.withDefaultDevice(.gpu) {
                let mlp = try GLM5NextMLP(
                    hiddenSize: 4096, intermediateSize: 6144,
                    gateProj: Self.nvfp4Linear(rows: 6144, inputs: 4096, seed: 1),
                    upProj: Self.nvfp4Linear(rows: 6144, inputs: 4096, seed: 2),
                    downProj: Self.nvfp4Linear(rows: 4096, inputs: 6144, seed: 3), swigluLimit: 7)
                let norm = GLM5NextRMSNorm(
                    weight: (1 + 0.1 * MLXRandom.normal([4096], key: MLXRandom.key(4))).asType(
                        .bfloat16),
                    epsilon: 1e-5)
                norm.usesFusedKernel = true
                let x = (MLXRandom.normal([1, Self.tokens, 4096], key: MLXRandom.key(5)) * 0.5)
                    .asType(.bfloat16)
                let streams = MLXRandom.normal([1, Self.tokens, 4, 4096], key: MLXRandom.key(6))
                    .asType(.bfloat16)
                eval(x, streams)
                var reference: (MLXArray, MLXArray)?
                for rows in Self.partitions {
                    let dense = Self.pieces(x, rows) { mlp($0) }
                    let readout = Self.pieces(streams, rows) { norm($0.mean(axis: 2)) }
                    eval(dense, readout)
                    if let reference {
                        Self.report("dense_mlp_nvfp4", reference.0, dense, rows)
                        Self.report("readout", reference.1, readout, rows)
                    } else {
                        reference = (dense, readout)
                    }
                }
            }
        }

        /// The router alone over finer piece sizes: where MLX's FP32 GEMM
        /// changes its K order.
        @Test func routerThreshold() throws {
            try Device.withDefaultDevice(.gpu) {
                typealias Decode = GLM5NextMoEDecodeKernelTests
                let moe = try Decode.moe(.tensorParallelRank, fusedSharedGateUp: true, nvfp4: true)
                moe.gate.usesNativeKernel = true
                let x = Decode.input(.tensorParallelRank, tokens: Self.tokens, seed: 21)
                eval(x)
                let whole = try moe.gate.route(x).logits.reshaped(1, Self.tokens, -1)
                eval(whole)
                for piece in [
                    256, 512, 768, 1000, 1024, 1154, 1280, 1414, 1536, 1718, 1792, 2047, 2048, 3000,
                ] {
                    let rows = [piece, Self.tokens - piece]
                    let logits = try Self.pieces(x, rows) {
                        try moe.gate.route($0).logits.reshaped(1, $0.dim(1), -1)
                    }
                    eval(logits)
                    Self.report(
                        "router_logits_first", whole[0..., ..<piece], logits[0..., ..<piece],
                        [piece])
                }
            }
        }

        @Test func moeAndRouter() throws {
            try Device.withDefaultDevice(.gpu) {
                typealias Decode = GLM5NextMoEDecodeKernelTests
                for nvfp4 in [true, false] {
                    let moe = try Decode.moe(
                        .tensorParallelRank, fusedSharedGateUp: true, nvfp4: nvfp4)
                    moe.gate.usesNativeKernel = true
                    let x = Decode.input(.tensorParallelRank, tokens: Self.tokens, seed: 21)
                    eval(x)
                    var reference: (MLXArray, MLXArray, MLXArray)?
                    for rows in Self.partitions {
                        let routing = try Self.pieces(x, rows) { piece in
                            let r = try moe.gate.route(piece)
                            return concatenated(
                                [r.weights.asType(.float32), r.indices.asType(.float32)], axis: -1
                            )
                            .reshaped(1, piece.dim(1), -1)
                        }
                        let logits = try Self.pieces(x, rows) { piece in
                            try moe.gate.route(piece).logits.reshaped(1, piece.dim(1), -1)
                        }
                        let output = try Self.pieces(x, rows) { piece in
                            try moe(
                                piece, routedExperts: GLM5NextExecutionPolicy.serving.routedExperts)
                        }
                        eval(routing, logits, output)
                        let tag = nvfp4 ? "nvfp4" : "affine"
                        if let reference {
                            Self.report("router_logits", reference.0, logits, rows)
                            Self.report("router_choice", reference.1, routing, rows)
                            Self.report("moe_\(tag)", reference.2, output, rows)
                        } else {
                            reference = (logits, routing, output)
                        }
                    }
                }
            }
        }
    }
}
