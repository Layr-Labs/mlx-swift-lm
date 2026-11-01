import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    /// The two activation overflow policies on every quantizer that applies
    /// them. Overflowing activations are NaN under `.eagerNaN` and equal an
    /// explicit clamped reference under `.saturating`; in-range activations are
    /// bit-identical under both. Synthetic values, float32 host oracles.
    struct ModelOptActivationOverflowTests {
        // MARK: Host oracles (no MLX quantization)

        /// SATFINITE E4M3FN: clamp to ±448, then round to nearest even.
        static func e4m3Saturated(_ x: Float) -> Float {
            if x.isNaN { return .nan }
            let c = min(abs(x), 448)
            guard c > 0 else { return 0 }
            let quantum = Float(sign: .plus, exponent: max(-6, Int(c.exponent)) - 3, significand: 1)
            let r = (c / quantum).rounded(.toNearestOrEven) * quantum
            return x < 0 ? -r : r
        }

        /// SATFINITE E2M1: clamp to ±6, then round to nearest even.
        static func e2m1Saturated(_ x: Float) -> Float {
            if x.isNaN { return .nan }
            let c = min(abs(x), 6)
            let quantum: Float = c < 2 ? 0.5 : (c < 4 ? 1 : 2)
            let r = (c / quantum).rounded(.toNearestOrEven) * quantum
            return x < 0 ? -r : r
        }

        /// The FP8 activation kernel's value with |x * m| clamped to 448.
        static func fp8Reference(_ x: Float, inputScale: Float) -> Float {
            let multiplier = 448 / (inputScale * 448)
            return e4m3Saturated(x * multiplier) * (1 / multiplier)
        }

        /// One NVFP4 row quantized and dequantized with the kernel's float32
        /// steps: block scales saturate at 448 and E2M1 values at 6.
        static func nvfp4Reference(_ row: [Float], inputScale: Float) -> [Float] {
            let scaleEnc = Float(448 * 6) / (inputScale * (448 * 6))
            return stride(from: 0, to: row.count, by: 16).flatMap { start -> [Float] in
                let block = row[start ..< start + 16]
                if block.contains(where: \.isNaN) { return block.map { _ in Float.nan } }
                let amax = block.map(abs).max()!
                let scale = e4m3Saturated((amax / 6) * scaleEnc)
                let scaleEncB = scale == 0 ? 0 : scaleEnc / scale
                return block.map { (scale / scaleEnc) * e2m1Saturated($0 * scaleEncB) }
            }
        }

        static func floats(_ array: MLXArray) -> [Float] {
            array.asType(.float32).asArray(Float.self)
        }

        static func same(_ actual: [Float], _ expected: [Float]) -> Bool {
            actual.count == expected.count
                && zip(actual, expected).allSatisfy { $0 == $1 || ($0.isNaN && $1.isNaN) }
        }

        static let policies: [ModelOptActivationOverflow] = [.eagerNaN, .saturating]

        // MARK: Activation kernels

        @Test func fp8ActivationKernel() throws {
            let inputScale: Float = 0.01  // multiplier 100, eager NaN beyond |x| = 4.64
            let inRange: [Float] = [0, -0.0, 1e-5, -0.3, 2.5, 4.48, -4.6, 4.64, 0.01171875]
            let overflow: [Float] = [4.65, -5, 40, -1e30, .infinity, -.infinity]
            let x = MLXArray(inRange + overflow + [.nan])
            let multiplier = 448 / (inputScale * 448)
            let multipliers = MLXArray([multiplier, 1 / multiplier])
            var outputs: [ModelOptActivationOverflow: [Float]] = [:]
            for policy in Self.policies {
                outputs[policy] = Self.floats(
                    ModelOptActivationKernels.fp8(
                        x, multipliers: multipliers,
                        overflowBound: MLXArray([policy.fp8NaNBound])))
            }
            let eager = try #require(outputs[.eagerNaN])
            let saturating = try #require(outputs[.saturating])
            let reference = (inRange + overflow).map {
                Self.fp8Reference($0, inputScale: inputScale)
            }
            let n = inRange.count
            #expect(Self.same(Array(eager[..<n]), Array(reference[..<n])))
            let eagerBits = eager[..<n].map { $0.bitPattern }
            let saturatingBits = saturating[..<n].map { $0.bitPattern }
            #expect(eagerBits == saturatingBits)
            #expect(eager[n...].allSatisfy { $0.isNaN })
            #expect(Self.same(Array(saturating[..<(n + overflow.count)]), reference))
            let largest = 448 * (1 / multiplier)
            #expect(saturating[n ..< (n + overflow.count)].allSatisfy { abs($0) == largest })
            #expect(saturating.last!.isNaN)
        }

        /// Rows of K = 64 (four blocks): in range, in range with a block scale
        /// that rounds to 448, one overflowing block, an infinity, and a NaN.
        static func nvfp4Rows(inputScale: Float) -> (rows: [[Float]], overflowing: Set<Int>) {
            let bound = 464 * 6 * inputScale
            let base = (0 ..< 64).map { Float(($0 * 37) % 29 - 14) / 14 * bound * 0.9 }
            var spike = base
            spike[20] = bound * 3
            var infinite = base
            infinite[47] = -.infinity
            var nan = base
            nan[5] = .nan
            let edge = base.enumerated().map { $0.offset == 33 ? bound * 0.999 : $0.element }
            return ([base, edge, spike, infinite, nan], [2, 3])
        }

        @Test func nvfp4DenseActivationKernel() throws {
            let inputScale: Float = 0.02
            let (rows, overflowing) = Self.nvfp4Rows(inputScale: inputScale)
            let x = MLXArray(rows.flatMap { $0 }).reshaped(rows.count, 64)
            var outputs: [ModelOptActivationOverflow: [Float]] = [:]
            for policy in Self.policies {
                outputs[policy] = Self.floats(
                    ModelOptActivationKernels.nvfp4Dense(
                        x, globalScale: MLXArray([inputScale * (448 * 6)]),
                        overflowDivisor: MLXArray([
                            policy.nvfp4OverflowDivisor(inputScale: inputScale)
                        ])))
            }
            try checkRows(outputs, rows: rows, overflowing: overflowing, scales: [inputScale])
        }

        @Test func nvfp4RoutedActivationKernel() throws {
            let inputScales: [Float] = [0.02, 0.005]
            let (rows, overflowing) = Self.nvfp4Rows(inputScale: 0.02)
            let x = MLXArray(rows.flatMap { $0 }).reshaped(rows.count, 64)
            // Each source row twice: expert 0 (its own calibration), then
            // expert 1, whose smaller scale also overflows the in-range rows.
            let sources = MLXArray((0 ..< rows.count).flatMap { [Int32($0), Int32($0)] })
            let experts = MLXArray((0 ..< rows.count).flatMap { _ in [Int32(0), 1] })
            var outputs: [ModelOptActivationOverflow: [Float]] = [:]
            for policy in Self.policies {
                let tables = ModelOptNVFP4RouteTables(
                    inputScales: inputScales, tensorScales: [1, 1], activationOverflow: policy)
                outputs[policy] = Self.floats(
                    ModelOptActivationKernels.nvfp4Routed(
                        x, sources: sources, experts: experts, globalScale: tables.globalScale,
                        overflowDivisor: tables.overflowDivisor,
                        inverseWeightScale: tables.inverseWeightScale
                    ).activation)
            }
            let routed = rows.flatMap { [$0, $0] }
            let overflowingRouted = Set(
                (0 ..< routed.count).filter { $0 % 2 == 1 || overflowing.contains($0 / 2) })
            try checkRows(
                outputs, rows: routed, overflowing: overflowingRouted,
                scales: (0 ..< routed.count).map { inputScales[$0 % 2] })
        }

        private func checkRows(
            _ outputs: [ModelOptActivationOverflow: [Float]], rows: [[Float]],
            overflowing: Set<Int>, scales: [Float]
        ) throws {
            let eager = try #require(outputs[.eagerNaN])
            let saturating = try #require(outputs[.saturating])
            for (index, row) in rows.enumerated() {
                let span = (index * 64) ..< ((index + 1) * 64)
                let reference = Self.nvfp4Reference(
                    row, inputScale: scales[scales.count == 1 ? 0 : index])
                let hasNaN = row.contains(where: \.isNaN)
                if hasNaN {
                    #expect(
                        eager[span].allSatisfy { $0.isNaN }
                            && saturating[span].allSatisfy { $0.isNaN })
                } else if overflowing.contains(index) {
                    #expect(eager[span].allSatisfy { $0.isNaN }, "row \(index)")
                    #expect(saturating[span].allSatisfy { $0.isFinite }, "row \(index)")
                    #expect(Self.same(Array(saturating[span]), reference), "row \(index)")
                } else {
                    #expect(Self.same(Array(eager[span]), reference), "row \(index)")
                    let eagerBits = eager[span].map { $0.bitPattern }
                    let saturatingBits = saturating[span].map { $0.bitPattern }
                    #expect(eagerBits == saturatingBits, "row \(index)")
                }
            }
        }

        // MARK: Projections: saturating(x) == eager(clamp(x)) bitwise

        /// Saturation equals the eager rule on the activation clamped to the
        /// largest representable magnitude: `448 * inputScale` for FP8,
        /// `448 * 6 * inputScale` for NVFP4 (block scale 448, E2M1 value 6).
        static func clamped(_ x: MLXArray, limit: Float) -> MLXArray {
            clip(x, min: MLXArray(-limit).asType(x.dtype), max: MLXArray(limit).asType(x.dtype))
        }

        static func fp8Weights() throws -> ModelOptFP8Weights {
            var fixtures = ModelOptGraphFixtures(seed: 23)
            return try ModelOptFP8Weights(
                e4m3Bytes: fixtures.fp8Weights(rows: 24, columns: 96), tensorScale: 1.0 / 512)
        }

        /// Row 0 in range, row 1 with one overflowing element, row 2 NaN.
        static func fp8Input(inputScale: Float, dtype: DType) -> MLXArray {
            var fixtures = ModelOptGraphFixtures(seed: 5)
            var values = fixtures.values(3 * 96, magnitude: 400 * inputScale)
            values[96 + 17] = 9000 * inputScale
            values[96 + 60] = -600 * inputScale
            values[2 * 96 + 3] = .nan
            return MLXArray(values).reshaped(3, 96).asType(dtype)
        }

        @Test(arguments: [DType.float32, .bfloat16])
        func fp8CalibratedMatmul(dtype: DType) throws {
            let weights = try Self.fp8Weights()
            let inputScale: Float = 0.02
            let x = Self.fp8Input(inputScale: inputScale, dtype: dtype)
            func run(_ input: MLXArray, _ policy: ModelOptActivationOverflow) throws -> MLXArray {
                try weights.calibratedMatmul(
                    input, inputScale: inputScale, weightDType: dtype, outputRows: 0 ..< 24,
                    activationOverflow: policy)
            }
            let eager = try run(x, .eagerNaN)
            let saturating = try run(x, .saturating)
            let reference = try run(Self.clamped(x, limit: 448 * inputScale), .eagerNaN)
            #expect(isNaN(eager[1 ... 2]).all().item(Bool.self))
            #expect(isFinite(saturating[0 ... 1]).all().item(Bool.self))
            #expect(isNaN(saturating[2]).all().item(Bool.self))
            #expect(modelOptBitwiseEqual(saturating, reference))
            #expect(modelOptBitwiseEqual(eager[0], saturating[0]))
        }

        @Test(arguments: [DType.float32, .bfloat16])
        func lazyFP8Projection(dtype: DType) throws {
            let inputScale: Float = 0.02
            func layer(_ policy: ModelOptActivationOverflow) throws -> ModelOptCalibratedLinear {
                try ModelOptCalibratedLinear(
                    weights: Self.fp8Weights(), inputScale: inputScale, weightDType: dtype,
                    activationOverflow: policy)
            }
            let budget = try ModelOptLazyFP8Projection.requiredArrayWorkspaceBytes(
                rows: 3, inputDimensions: 96, outputDimensions: 24)
            let x = Self.fp8Input(inputScale: inputScale, dtype: dtype)
            let eagerLayer = try layer(.eagerNaN)
            let saturatingLayer = try layer(.saturating)
            let clampedX = Self.clamped(x, limit: 448 * inputScale)
            // Lazy, graph, row-exact (batched decode) and eager-reference paths.
            let paths: [(ModelOptCalibratedLinear, MLXArray) throws -> MLXArray] = [
                { try $0.lazyFP8Forward($1, admittedArrayWorkspaceBytes: budget).value },
                { try $0.graphForward($1) },
                { try $0.rowExactForward($1) },
                { try $0.checkedForward($1) },
            ]
            for (index, path) in paths.enumerated() {
                let eager = try path(eagerLayer, x)
                let saturating = try path(saturatingLayer, x)
                #expect(isNaN(eager[1 ... 2]).all().item(Bool.self), "path \(index)")
                #expect(isFinite(saturating[0 ... 1]).all().item(Bool.self), "path \(index)")
                #expect(isNaN(saturating[2]).all().item(Bool.self), "path \(index)")
                #expect(
                    modelOptBitwiseEqual(saturating, try path(eagerLayer, clampedX)),
                    "path \(index)")
                #expect(modelOptBitwiseEqual(eager[0], saturating[0]), "path \(index)")
            }
            #expect(
                modelOptBitwiseEqual(
                    try saturatingLayer.graphForward(x), try saturatingLayer.checkedForward(x)))
        }

        static func expertWeights(inputs: Int, outputs: Int, seed: UInt64) throws
            -> [ModelOptNVFP4Weights]
        {
            var fixtures = ModelOptGraphFixtures(seed: seed)
            return try (0 ..< 2).map { _ in
                try ModelOptNVFP4Weights(
                    packedBytes: fixtures.packedNVFP4(rows: outputs, columns: inputs),
                    e4m3ScaleBytes: fixtures.blockScales(rows: outputs, columns: inputs),
                    tensorScale: 0.004)
            }
        }

        @Test(arguments: [DType.float32, .bfloat16])
        func nvfp4SwitchLinear(dtype: DType) throws {
            let inputScales: [Float] = [0.01, 0.02]
            func bank(_ policy: ModelOptActivationOverflow) throws -> ModelOptNVFP4SwitchLinear {
                let layer = try ModelOptNVFP4SwitchLinear(
                    experts: Self.expertWeights(inputs: 128, outputs: 32, seed: 31),
                    inputScales: inputScales, weightDType: dtype, activationOverflow: policy)
                layer.accumulation = .rowwise
                return layer
            }
            var fixtures = ModelOptGraphFixtures(seed: 9)
            var values = fixtures.values(3 * 128, magnitude: 2000 * 0.01)
            values[128 + 70] = 6 * 464 * 0.02 * 4  // row 1 overflows both experts
            values[2 * 128 + 9] = .nan
            let x = MLXArray(values).reshaped(3, 1, 128).asType(dtype)
            let ids = MLXArray([Int32(0), 1, 1])
            // Per-row clamp to the selected expert's limit.
            let limits = MLXArray(ids.asArray(Int32.self).map { 448 * 6 * inputScales[Int($0)] })
                .reshaped(3, 1, 1).asType(dtype)
            let clampedX = clip(x, min: -limits, max: limits)
            let eagerBank = try bank(.eagerNaN)
            let saturatingBank = try bank(.saturating)
            let quote = try eagerBank.lazyDecodeQuote(assignments: 3)
            func paths(_ layer: ModelOptNVFP4SwitchLinear, _ input: MLXArray) throws -> [MLXArray] {
                [
                    try layer.graphForward(input, ids).value,
                    try layer.checkedForward(input, ids),
                    try layer.lazyDecode(
                        input, ids, admittedArrayBytes: quote.arrayBytes,
                        admittedGraphUnits: quote.graphUnits),
                ]
            }
            let eager = try paths(eagerBank, x)
            let saturating = try paths(saturatingBank, x)
            let reference = try paths(eagerBank, clampedX)
            for (index, value) in eager.enumerated() {
                #expect(isNaN(value[1 ... 2]).all().item(Bool.self), "path \(index)")
                #expect(isFinite(saturating[index][0 ... 1]).all().item(Bool.self), "path \(index)")
                #expect(isNaN(saturating[index][2]).all().item(Bool.self), "path \(index)")
                #expect(modelOptBitwiseEqual(saturating[index], reference[index]), "path \(index)")
                #expect(modelOptBitwiseEqual(value[0], saturating[index][0]), "path \(index)")
            }
            #expect(modelOptBitwiseEqual(saturating[0], saturating[1]))
        }

        @Test(arguments: [DType.float32, .bfloat16])
        func nvfp4CalibratedLinear(dtype: DType) throws {
            let inputScale: Float = 0.01
            let weights = try Self.expertWeights(inputs: 128, outputs: 32, seed: 41)[0]
            func layer(_ policy: ModelOptActivationOverflow) throws -> ModelOptCalibratedLinear {
                let layer = try ModelOptCalibratedLinear(
                    weights: weights, inputScale: inputScale, weightDType: dtype,
                    activationOverflow: policy)
                layer.accumulation = .rowwise
                return layer
            }
            var fixtures = ModelOptGraphFixtures(seed: 13)
            var values = fixtures.values(3 * 128, magnitude: 2000 * inputScale)
            values[128 + 100] = -6 * 464 * inputScale * 3
            values[2 * 128] = .nan
            let x = MLXArray(values).reshaped(3, 128).asType(dtype)
            let clampedX = Self.clamped(x, limit: 448 * 6 * inputScale)
            let eagerLayer = try layer(.eagerNaN)
            let saturatingLayer = try layer(.saturating)
            for forward in [
                { (l: ModelOptCalibratedLinear, v: MLXArray) in try l.graphForward(v) },
                { (l: ModelOptCalibratedLinear, v: MLXArray) in try l.checkedForward(v) },
            ] {
                let eager = try forward(eagerLayer, x)
                let saturating = try forward(saturatingLayer, x)
                #expect(isNaN(eager[1 ... 2]).all().item(Bool.self))
                #expect(isFinite(saturating[0 ... 1]).all().item(Bool.self))
                #expect(isNaN(saturating[2]).all().item(Bool.self))
                #expect(modelOptBitwiseEqual(saturating, try forward(eagerLayer, clampedX)))
                #expect(modelOptBitwiseEqual(eager[0], saturating[0]))
            }
        }
    }
}
