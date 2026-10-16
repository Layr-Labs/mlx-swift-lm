import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Guards, at the pinned Super geometry in BF16, every multi-row operation the
/// rectangular verifier batches: each must reproduce the single-row decode
/// result bit for bit. Operations not proven here run one row at a time.
@Suite(.tags(.integration))
struct NemotronSuperMTPRowExactnessTests {
    static let widths = [2, 3, 4]

    static func normal(_ shape: [Int], seed: UInt64, scale: Float = 1) -> MLXArray {
        (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(.bfloat16)
    }

    static func nvfp4Weights(input: Int, output: Int, seed: UInt64) throws -> ModelOptNVFP4Weights {
        let packed = MLXRandom.randInt(
            Int32(0) ..< 256, [output, input / 2], key: MLXRandom.key(seed))
        let scales = MLXRandom.randInt(
            Int32(40) ..< 60, [output, input / 16], key: MLXRandom.key(seed + 1))
        return try ModelOptNVFP4Weights(
            packedBytes: packed.asType(.uint8), e4m3ScaleBytes: scales.asType(.uint8),
            tensorScale: 0.01)
    }

    /// Bitwise comparison of a batched op on `[1, W, ...]` with `reference`
    /// applied to each `[1, 1, ...]` row.
    static func exact(
        _ x: MLXArray, _ batched: (MLXArray) throws -> MLXArray,
        reference: (MLXArray) throws -> MLXArray
    ) rethrows -> Bool {
        let width = x.dim(1)
        let a = try batched(x)
        let b = concatenated(
            try (0 ..< width).map { try reference(x[0..., $0 ..< ($0 + 1)]) }, axis: 1)
        eval(a, b)
        return a.shape == b.shape && arrayEqual(a, b).item(Bool.self)
    }

    @Test(.timeLimit(.minutes(2))) func batchedProjectionsAndNormsMatchDecodeRows() throws {
        let sharedUp = try ModelOptCalibratedLinear(
            weights: Self.nvfp4Weights(input: 4096, output: 5376, seed: 3),
            inputScale: 0.01, weightDType: .bfloat16)
        let latent = Linear(4096, 1024, bias: false)
        latent.update(
            parameters: .unflattened([
                ("weight", Self.normal([1024, 4096], seed: 4, scale: 0.02))
            ]))
        let norm = RMSNorm(dimensions: 4096, eps: 1e-5)
        let gated = NemotronHRMSNormGated(dimensions: 8192, eps: 1e-5, groupSize: 1024)
        let configuration = NemotronHConfiguration(
            vocabSize: 32, hiddenSize: 4096, numHiddenLayers: 1, numAttentionHeads: 32,
            numKeyValueHeads: 2, mambaNumHeads: 128, mambaHeadDim: 64, ssmStateSize: 128,
            convKernel: 4, nGroups: 8, intermediateSize: 8192, moeIntermediateSize: 2688,
            moeSharedExpertIntermediateSize: 5376, nRoutedExperts: 512, numExpertsPerTok: 22,
            hybridOverridePattern: "E", headDim: 128, nSharedExperts: 1, routedScalingFactor: 5,
            moeLatentSize: 1024)
        let router = NemotronHMoEGate(configuration)
        router.update(
            parameters: .unflattened([
                ("weight", Self.normal([512, 4096], seed: 5, scale: 0.02)),
                ("e_score_correction_bias", MLXRandom.normal([512], key: MLXRandom.key(6)) * 0.01),
            ]))
        for width in Self.widths + [16, 20] {
            let x = Self.normal([1, width, 4096], seed: UInt64(10 + width))
            #expect(
                Self.exact(x, { superRowExactRows(sharedUp, $0) }) { sharedUp($0) },
                "NVFP4 rows W=\(width)")
            #expect(
                Self.exact(x, { superRowExactRows(latent, $0) }) { latent($0) },
                "dense rows W=\(width)")
            #expect(
                Self.exact(x, { superRowExactDense(latent, $0) }) { latent($0) },
                "dense gather W=\(width)")
            if width <= 16 {
                let (batchedIDs, batchedScores) = router.superRowExactForwardRows(x)
                let (rowIDs, rowScores) = router.mtpForwardRows(x)
                eval(batchedIDs, batchedScores, rowIDs, rowScores)
                #expect(
                    arrayEqual(batchedIDs, rowIDs).item(Bool.self)
                        && arrayEqual(batchedScores, rowScores).item(Bool.self),
                    "router W=\(width)")
            }
            #expect(Self.exact(x, { norm($0) }) { norm($0) }, "RMSNorm W=\(width)")
            let wide = Self.normal([1, width, 8192], seed: UInt64(20 + width))
            let gate = Self.normal([1, width, 8192], seed: UInt64(30 + width))
            #expect(
                Self.exact(
                    concatenated([wide, gate], axis: -1),
                    { gated($0[.ellipsis, ..<8192], gate: $0[.ellipsis, 8192...]) }
                ) { gated($0[.ellipsis, ..<8192], gate: $0[.ellipsis, 8192...]) },
                "gated norm W=\(width)")
        }
    }

    /// Median evaluated wall time of `body` over seven runs after one warm-up.
    static func medianNanos(_ body: () throws -> MLXArray) rethrows -> UInt64 {
        eval(try body())
        var samples: [UInt64] = []
        for _ in 0 ..< 7 {
            let start = DispatchTime.now().uptimeNanoseconds
            eval(try body())
            samples.append(DispatchTime.now().uptimeNanoseconds - start)
        }
        return samples.sorted()[3]
    }

    /// Synthetic weights at Super's FP8 shapes, one shape at a time: a W-row
    /// `rowExactForward` call against W one-row exact calls in one graph.
    @Test(.serialized, arguments: [(4096, 18560), (8192, 4096), (4096, 4096), (4096, 131072)])
    func fp8RowsTiming(inputs: Int, outputs: Int) throws {
        let codes = MLXRandom.randInt(
            UInt8(0) ..< UInt8(127), [min(outputs, 4096), inputs], key: MLXRandom.key(5))
        let bytes = tiled(codes, repetitions: [(outputs + 4095) / 4096, 1])[..<outputs]
        let layer = try ModelOptCalibratedLinear(
            weights: ModelOptFP8Weights(e4m3Bytes: bytes, tensorScale: 0.00123),
            inputScale: 0.0071, weightDType: .bfloat16)
        var line = "SYNTHETIC-WEIGHTS super-mtp FP8 K=\(inputs) N=\(outputs) median ns:"
        for width in [1, 2, 4] {
            let x = Self.normal([1, width, inputs], seed: 40, scale: 0.8)
            let rows = try Self.medianNanos { try layer.rowExactForward(x) }
            let calls = try Self.medianNanos {
                concatenated(
                    try (0 ..< width).map { try layer.graphForward(x[0..., $0 ..< ($0 + 1)]) },
                    axis: 1)
            }
            line += " W=\(width) rows=\(rows) oneRowCalls=\(calls)"
        }
        print(line)
    }

    /// Bit patterns, so NaN outputs of invalid E4M3 codes compare equal.
    static func bitwise(_ a: MLXArray, _ b: MLXArray) -> Bool {
        eval(a, b)
        return a.shape == b.shape && a.dtype == .bfloat16 && b.dtype == .bfloat16
            && arrayEqual(a.view(dtype: .uint16), b.view(dtype: .uint16)).item(Bool.self)
    }

    /// Every Super FP8 shape (Mamba in/out, attention q/o and k/v, the
    /// 131072-row output head) plus a K leftover, N < 4 and NaN-code case:
    /// each row of the batched product the Super verifier and batched decode
    /// use (`superRowExactProjection`) equals the one-row exact graph call.
    @Test(
        .timeLimit(.minutes(4)),
        arguments: [
            (4096, 18560), (8192, 4096), (4096, 4096), (4096, 256), (4096, 131072), (200, 40),
            (136, 3),
        ])
    func fp8RowsMatchExactDecodeRows(inputs: Int, outputs: Int) throws {
        let block = min(outputs, 4096)
        let raw = MLXRandom.randInt(
            UInt8(0) ..< UInt8(255), [block, inputs], key: MLXRandom.key(UInt64(inputs ^ outputs)))
        let small = inputs * outputs < 1 << 16
        // Small shapes keep E4M3 NaN codes to exercise the invalid-output mask.
        let codes = small ? raw : which((raw & UInt8(0x7F)) .== UInt8(0x7F), raw - UInt8(1), raw)
        let bytes = tiled(codes, repetitions: [(outputs + block - 1) / block, 1])[..<outputs]
        let layer = try ModelOptCalibratedLinear(
            weights: ModelOptFP8Weights(e4m3Bytes: bytes, tensorScale: 0.00123),
            inputScale: 0.0071, weightDType: .bfloat16)
        for width in outputs > 65536 ? [1, 4, 16] : [1, 2, 3, 4, 8, 16, 17] {
            let x = Self.normal([1, width, inputs], seed: UInt64(width), scale: 0.8)
            let rows = superRowExactRows(layer, x)
            let reference = concatenated(
                try (0 ..< width).map { try layer.graphForward(x[0..., $0 ..< ($0 + 1)]) }, axis: 1)
            #expect(Self.bitwise(rows, reference), "K=\(inputs) N=\(outputs) W=\(width)")
        }
    }

    @Test(.timeLimit(.minutes(2))) func routedExpertsAndWeightedSumMatchDecodeRows() throws {
        let experts = 32
        let mlp = NemotronHSwitchMLP(inputDims: 1024, hiddenDims: 2688, numExperts: experts)
        let fc1 = try ModelOptNVFP4SwitchLinear(
            experts: (0 ..< experts).map {
                try Self.nvfp4Weights(input: 1024, output: 2688, seed: 100 + 2 * UInt64($0))
            },
            inputScales: Array(repeating: 0.01, count: experts), weightDType: .bfloat16)
        let fc2 = try ModelOptNVFP4SwitchLinear(
            experts: (0 ..< experts).map {
                try Self.nvfp4Weights(input: 2688, output: 1024, seed: 300 + 2 * UInt64($0))
            },
            inputScales: Array(repeating: 0.05, count: experts), weightDType: .bfloat16)
        try mlp.update(
            modules: .unflattened([("fc1", fc1 as Module), ("fc2", fc2 as Module)]),
            verify: .all)
        for width in Self.widths {
            let x = Self.normal([1, width, 1024], seed: UInt64(40 + width))
            let ids = argSort(
                MLXRandom.uniform(0 ..< 1, [1, width, experts], key: MLXRandom.key(50)), axis: -1
            )[.ellipsis, ..<22].asType(.int32)
            let scores = MLXRandom.uniform(0 ..< 1, [1, width, 22], key: MLXRandom.key(60))
            func route(_ rows: MLXArray, _ range: Range<Int>) -> MLXArray {
                let y = mlp(rows, ids[0..., range])
                return (y * scores[0..., range][.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
            }
            let batched = route(x, 0 ..< width)
            let rows = concatenated(
                (0 ..< width).map { route(x[0..., $0 ..< ($0 + 1)], $0 ..< ($0 + 1)) }, axis: 1)
            eval(batched, rows)
            #expect(arrayEqual(batched, rows).item(Bool.self), "routed W=\(width)")
        }
    }

    /// Under the tiled default the routed graph stays per row below MLX's
    /// sorted matrix route (16 rows and four per expert on average); the
    /// verifier's predicate admits batched routing only there. Fails if that
    /// threshold moves down.
    @Test(.timeLimit(.minutes(2))) func tiledRoutingIsRowExactBelowTheMatrixRoute() throws {
        let experts = 32
        let topK = 4
        let mlp = NemotronHSwitchMLP(inputDims: 1024, hiddenDims: 512, numExperts: experts)
        let fc1 = try ModelOptNVFP4SwitchLinear(
            experts: (0 ..< experts).map {
                try Self.nvfp4Weights(input: 1024, output: 512, seed: 500 + 2 * UInt64($0))
            },
            inputScales: Array(repeating: 0.01, count: experts), weightDType: .bfloat16)
        let fc2 = try ModelOptNVFP4SwitchLinear(
            experts: (0 ..< experts).map {
                try Self.nvfp4Weights(input: 512, output: 1024, seed: 700 + 2 * UInt64($0))
            },
            inputScales: Array(repeating: 0.05, count: experts), weightDType: .bfloat16)
        try mlp.update(
            modules: .unflattened([("fc1", fc1 as Module), ("fc2", fc2 as Module)]),
            verify: .all)
        #expect(fc1.accumulation == .tiled && fc2.accumulation == .tiled)
        let threshold = 4 * experts
        #expect(superRowExactRouting(mlp, routedRows: threshold - 1))
        #expect(!superRowExactRouting(mlp, routedRows: threshold))
        // 31 tokens x 4 routes: sorted (above 64 assignments), below the route.
        let tokens = (threshold - 1) / topK
        let x = Self.normal([1, tokens, 1024], seed: 41)
        let ids = argSort(
            MLXRandom.uniform(0 ..< 1, [1, tokens, experts], key: MLXRandom.key(51)), axis: -1
        )[.ellipsis, ..<topK].asType(.int32)
        let batched = mlp(x, ids)
        let rows = concatenated(
            (0 ..< tokens).map { mlp(x[0..., $0 ..< ($0 + 1)], ids[0..., $0 ..< ($0 + 1)]) },
            axis: 1)
        eval(batched, rows)
        #expect(arrayEqual(batched, rows).item(Bool.self))
    }

    @Test(.timeLimit(.minutes(2))) func windowConvolutionAndSSMMatchDecodeSteps() throws {
        let heads = 128
        let headDim = 64
        let state = 128
        let groups = 8
        let kernel = 4
        let convDim = heads * headDim + 2 * groups * state
        let conv = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: kernel,
            groups: convDim, bias: true)
        conv.update(
            parameters: .unflattened([
                ("weight", Self.normal([convDim, kernel, 1], seed: 70, scale: 0.3)),
                ("bias", Self.normal([convDim], seed: 71, scale: 0.1)),
            ]))
        let aLog = MLX.log(MLXArray(0 ..< heads).asType(.float32) + 1)
        let d = MLXArray.ones([heads])
        let dtBias = MLXRandom.normal([heads], key: MLXRandom.key(72)) * 0.1
        let limit: (Float, Float) = (0, .infinity)
        for width in Self.widths {
            let history = Self.normal([1, kernel - 1, convDim], seed: 80)
            let inputs = Self.normal([1, width, convDim], seed: UInt64(81 + width))
            let padded = concatenated([history, inputs], axis: 1)
            let window = silu(conv(padded))
            let steps = concatenated(
                (0 ..< width).map { silu(conv(padded[0..., $0 ..< ($0 + kernel)])) }, axis: 1)
            eval(window, steps)
            #expect(arrayEqual(window, steps).item(Bool.self), "conv W=\(width)")

            let x = Self.normal([1, width, heads, headDim], seed: UInt64(90 + width))
            let b = Self.normal([1, width, groups, state], seed: UInt64(91 + width))
            let c = Self.normal([1, width, groups, state], seed: UInt64(92 + width))
            let dt = Self.normal([1, width, heads], seed: UInt64(93 + width))
            let initial = MLXRandom.normal([1, heads, headDim, state], key: MLXRandom.key(94))
            let captured = nemotronMTPSSMWindow(
                hiddenStates: x, ALog: aLog, B: b, C: c, D: d, dt: dt, dtBias: dtBias,
                state: initial, timeStepLimit: limit)
            var current = initial
            for t in 0 ..< width {
                let (y, next) = ssmUpdate(
                    hiddenStates: x[0..., t ..< (t + 1)], ALog: aLog,
                    B: b[0..., t ..< (t + 1)], C: c[0..., t ..< (t + 1)], D: d,
                    dt: dt[0..., t ..< (t + 1)], dtBias: dtBias, state: current,
                    timeStepLimit: limit, mask: nil)
                current = next.asType(.float32)
                eval(y, current, captured.output, captured.capturedStates)
                #expect(
                    arrayEqual(y.asType(.bfloat16), captured.output[0..., t ..< (t + 1)])
                        .item(Bool.self), "SSM output W=\(width) t=\(t)")
                #expect(
                    arrayEqual(current, captured.capturedStates[t ..< (t + 1)]).item(Bool.self),
                    "SSM state W=\(width) t=\(t)")
            }
        }
    }
}
