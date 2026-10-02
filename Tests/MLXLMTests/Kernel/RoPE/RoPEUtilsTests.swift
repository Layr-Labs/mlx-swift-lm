import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of `initializeRope` and the RoPE layers in `RoPEUtils.swift`
    /// and `SuScaledRoPE.swift`.
    ///
    /// Each layer is compared with a reference rotation that the test
    /// computes from the published formula of its scaling type, with
    /// `cos` and `sin` in Double. The reference does not use
    /// `MLXFast.RoPE`.
    @Suite
    struct RoPEUtilsTests {

        /// A rotation that keeps positions and frequencies in Double.
        ///
        /// - `periods[i]` is the period of pair `i`: the angle at position
        ///   `p` is `p / periods[i]`. An infinite period leaves the pair as
        ///   it is.
        /// - The pairs are `(i, i + dims / 2)`, as in the non-traditional
        ///   layout. Dimensions from `dims` on pass through.
        /// - `amplitude` multiplies the rotated dimensions.
        static func reference(
            _ x: MLXArray, dims: Int, periods: [Double], positions: [Double],
            amplitude: Double = 1
        ) -> MLXArray {
            let half = dims / 2
            precondition(periods.count == half)
            let length = x.dim(-2)
            precondition(positions.count == length)
            var cosines: [Float] = []
            var sines: [Float] = []
            for position in positions {
                for period in periods {
                    let angle = period.isInfinite ? 0 : position / period
                    cosines.append(Float(cos(angle)))
                    sines.append(Float(sin(angle)))
                }
            }
            let c = MLXArray(cosines).reshaped(1, 1, length, half)
            let s = MLXArray(sines).reshaped(1, 1, length, half)
            let x1 = x[.ellipsis, 0 ..< half]
            let x2 = x[.ellipsis, half ..< dims]
            var parts = [
                (x1 * c - x2 * s) * Float(amplitude), (x1 * s + x2 * c) * Float(amplitude),
            ]
            if x.dim(-1) > dims {
                parts.append(x[.ellipsis, dims...])
            }
            return concatenated(parts, axis: -1)
        }

        static func input(dims: Int = 8, length: Int = 5, batch: Int = 1, seed: UInt64 = 1)
            -> MLXArray
        {
            let x = MLXRandom.normal([batch, 2, length, dims], key: MLXRandom.key(seed))
            eval(x)
            return x
        }

        static func defaultPeriods(dims: Int, base: Double) -> [Double] {
            stride(from: 0, to: dims, by: 2).map { pow(base, Double($0) / Double(dims)) }
        }

        static func positions(offset: Int, length: Int, scale: Double = 1) -> [Double] {
            (0 ..< length).map { Double(offset + $0) * scale }
        }

        // Tolerance: MLXFast.RoPE computes the angles and the rotation in
        // float32. For inputs of size 1 and positions up to 10 the error is
        // near 1e-6.
        static let tolerance: Float = 1e-5

        static func rope(_ config: [String: StringOrNumber]?, dims: Int = 8) -> RoPELayer {
            initializeRope(
                dims: dims, base: 10000, traditional: false, scalingConfig: config,
                maxPositionEmbeddings: 256)
        }

        @Test(arguments: [nil, "default", "mrope"])
        func defaultRopeRotatesByPositionOverPeriod(type: String?) {
            let config: [String: StringOrNumber]? =
                type.map { ["type": .string($0), "mrope_section": .ints([1, 1, 2])] }
            let rope = Self.rope(config)
            let x = Self.input()
            let expected = Self.reference(
                x, dims: 8, periods: Self.defaultPeriods(dims: 8, base: 10000),
                positions: Self.positions(offset: 3, length: 5))
            #expect(SyntheticModel.maxAbsDifference(rope(x, offset: 3), expected) <= Self.tolerance)
        }

        @Test func linearScalingDividesThePosition() {
            let rope = Self.rope(["type": .string("linear"), "factor": .float(4)])
            let x = Self.input()
            let expected = Self.reference(
                x, dims: 8, periods: Self.defaultPeriods(dims: 8, base: 10000),
                positions: Self.positions(offset: 3, length: 5, scale: 0.25))
            #expect(SyntheticModel.maxAbsDifference(rope(x, offset: 3), expected) <= Self.tolerance)
        }

        /// The Llama 3 rule: pairs with a short wavelength keep their
        /// frequency, pairs with a long wavelength get `factor` times the
        /// period, and pairs between the two are interpolated.
        @Test func llama3ScalingFollowsTheThreeBands() {
            let factor = 8.0
            let low = 1.0
            let high = 4.0
            let original = 64.0
            let rope = Self.rope([
                "rope_type": .string("llama3"), "factor": .float(Float(factor)),
                "low_freq_factor": .float(Float(low)), "high_freq_factor": .float(Float(high)),
                "original_max_position_embeddings": .int(Int(original)),
            ])
            // Wavelengths 2 pi x (1, 10, 100, 1000): one short, one between
            // 16 and 64, two long.
            let periods = Self.defaultPeriods(dims: 8, base: 10000).map { period -> Double in
                let wavelength = 2 * Double.pi * period
                if wavelength < original / high { return period }
                if wavelength > original / low { return period * factor }
                let smooth = (original / wavelength - low) / (high - low)
                return 1 / ((1 - smooth) / (period * factor) + smooth / period)
            }
            let x = Self.input()
            let expected = Self.reference(
                x, dims: 8, periods: periods, positions: Self.positions(offset: 2, length: 5))
            #expect(SyntheticModel.maxAbsDifference(rope(x, offset: 2), expected) <= Self.tolerance)
            // The interpolation must change the long pairs.
            let plain = Self.reference(
                x, dims: 8, periods: Self.defaultPeriods(dims: 8, base: 10000),
                positions: Self.positions(offset: 2, length: 5))
            #expect(SyntheticModel.maxAbsDifference(expected, plain) > 1e-3)
        }

        /// The YaRN rule (DeepSeek form): pairs blend between the original
        /// frequency and the frequency divided by `factor` with a linear ramp
        /// between the correction dimensions, and the rotated dimensions are
        /// scaled by the attention factor.
        @Test(arguments: [8, 16])
        func yarnScalingBlendsFrequenciesAndScalesTheRotatedPart(ropeDims: Int) {
            let base = 10000.0
            let factor = 4.0
            let original = 64.0
            let betaFast = 32.0
            let betaSlow = 1.0
            let mscale = 1.0
            let mscaleAllDim = 0.0
            let rope = initializeRope(
                dims: ropeDims, base: Float(base), traditional: false,
                scalingConfig: [
                    "type": .string("yarn"), "factor": .float(Float(factor)),
                    "original_max_position_embeddings": .int(Int(original)),
                    "beta_fast": .float(Float(betaFast)), "beta_slow": .float(Float(betaSlow)),
                    "mscale": .float(Float(mscale)), "mscale_all_dim": .float(Float(mscaleAllDim)),
                ],
                maxPositionEmbeddings: 256)

            func correctionDim(_ rotations: Double) -> Double {
                Double(ropeDims) * log(original / (rotations * 2 * Double.pi)) / (2 * log(base))
            }
            let lowDim = max(floor(correctionDim(betaFast)), 0)
            var highDim = min(ceil(correctionDim(betaSlow)), Double(ropeDims - 1))
            if lowDim == highDim { highDim += 0.001 }
            let periods = (0 ..< ropeDims / 2).map { i -> Double in
                let ramp = min(max((Double(i) - lowDim) / (highDim - lowDim), 0), 1)
                let extrapolation = 1 - ramp
                let period = pow(base, Double(2 * i) / Double(ropeDims))
                let inverse = extrapolation / period + (1 - extrapolation) / (period * factor)
                return 1 / inverse
            }
            func getMscale(_ m: Double) -> Double { factor <= 1 ? 1 : 0.1 * m * log(factor) + 1 }
            let amplitude = getMscale(mscale) / getMscale(mscaleAllDim)

            // Head dimension 16: with ropeDims 8 the last 8 dimensions pass
            // through without rotation and without the attention factor.
            let x = Self.input(dims: 16)
            let saved = x[.ellipsis] + 0
            eval(saved)
            let expected = Self.reference(
                x, dims: ropeDims, periods: periods,
                positions: Self.positions(offset: 4, length: 5), amplitude: amplitude)
            let output = rope(x, offset: 4)
            #expect(SyntheticModel.maxAbsDifference(output, expected) <= Self.tolerance)
            // The layer scales a copy of the input, not the input itself.
            #expect(SyntheticModel.maxAbsDifference(x, saved) == 0)
        }

        /// Proportional RoPE rotates the first `partial_rotary_factor` of the
        /// pairs with periods over the full head dimension, and leaves the
        /// other pairs as they are.
        @Test func proportionalRopeRotatesOnlyThePartialPairs() {
            let factor = 2.0
            let rope = Self.rope([
                "type": .string("proportional"), "factor": .float(Float(factor)),
                "partial_rotary_factor": .float(0.5),
            ])
            let periods =
                [0, 1].map { factor * pow(10000, Double(2 * $0) / 8) } + [
                    Double.infinity, Double.infinity,
                ]
            let x = Self.input()
            let expected = Self.reference(
                x, dims: 8, periods: periods, positions: Self.positions(offset: 3, length: 5))
            #expect(SyntheticModel.maxAbsDifference(rope(x, offset: 3), expected) <= Self.tolerance)

            // With no rotated pair the layer returns its input.
            let none = Self.rope([
                "type": .string("proportional"), "partial_rotary_factor": .float(0),
            ])
            #expect(SyntheticModel.maxAbsDifference(none(x, offset: 3), x) == 0)
            #expect(
                SyntheticModel.maxAbsDifference(none(x, offset: MLXArray([Int32(3)])), x) == 0)
        }

        /// LongRoPE (`SuScaledRoPE`) multiplies each period by its long
        /// factor and scales the rotated dimensions by
        /// `sqrt(1 + ln(max / original) / ln(original))`.
        @Test func longropeUsesTheLongFactorsAndTheScale() {
            let longFactor: [Float] = [1, 2, 4, 8]
            let rope = initializeRope(
                dims: 8, base: 10000, traditional: false,
                scalingConfig: [
                    "type": .string("longrope"),
                    "original_max_position_embeddings": .int(64),
                    "short_factor": .floats([1, 1, 1, 1]),
                    "long_factor": .floats(longFactor),
                ],
                maxPositionEmbeddings: 256)
            let periods = zip(Self.defaultPeriods(dims: 8, base: 10000), longFactor).map {
                $0 * Double($1)
            }
            let amplitude = (1 + log(256.0 / 64.0) / log(64.0)).squareRoot()
            let x = Self.input()
            let saved = x[.ellipsis] + 0
            eval(saved)
            let expected = Self.reference(
                x, dims: 8, periods: periods, positions: Self.positions(offset: 1, length: 5),
                amplitude: amplitude)
            #expect(SyntheticModel.maxAbsDifference(rope(x, offset: 1), expected) <= Self.tolerance)
            #expect(SyntheticModel.maxAbsDifference(x, saved) == 0)
        }

        static let allTypes: [[String: StringOrNumber]?] = [
            nil,
            ["type": .string("linear"), "factor": .float(2)],
            [
                "rope_type": .string("llama3"), "factor": .float(8),
                "original_max_position_embeddings": .int(64),
            ],
            [
                "type": .string("yarn"), "factor": .float(4),
                "original_max_position_embeddings": .int(64),
            ],
            ["type": .string("deepseek_yarn"), "factor": .float(4), "mscale": .float(0.7)],
            ["type": .string("proportional"), "partial_rotary_factor": .float(0.5)],
            [
                "type": .string("longrope"), "original_max_position_embeddings": .int(64),
                "short_factor": .floats([1, 1, 1, 1]), "long_factor": .floats([1, 2, 3, 4]),
            ],
        ]

        /// One token at offset `k` gets the same rotation as position `k` of
        /// a longer sequence. This is what a KV cache relies on.
        @Test(arguments: allTypes)
        func oneTokenAtAnOffsetMatchesTheSamePositionOfASequence(
            config: [String: StringOrNumber]?
        ) {
            let rope = Self.rope(config)
            let x = Self.input(length: 6)
            let whole = rope(x, offset: 0)
            for k in [1, 5] {
                let single = rope(x[0..., 0..., k ..< k + 1], offset: k)
                #expect(
                    SyntheticModel.maxAbsDifference(single, whole[0..., 0..., k ..< k + 1])
                        <= Self.tolerance, "position \(k)")
            }
        }

        /// A per-row offset array gives each row the rotation of its own
        /// scalar offset.
        @Test(arguments: allTypes)
        func offsetArrayMatchesScalarOffsets(config: [String: StringOrNumber]?) {
            let rope = Self.rope(config)
            let x = Self.input(length: 3, batch: 2)
            let offsets = [2, 5]
            let batched = rope(x, offset: MLXArray(offsets.map { Int32($0) }))
            for (row, offset) in offsets.enumerated() {
                let alone = rope(x[row ..< row + 1], offset: offset)
                #expect(
                    SyntheticModel.maxAbsDifference(batched[row ..< row + 1], alone)
                        <= Self.tolerance, "row \(row)")
            }
        }

        /// The dot product of a rotated query and a rotated key depends only
        /// on the distance between their positions.
        @Test(arguments: allTypes)
        func attentionScoreDependsOnlyOnRelativePosition(config: [String: StringOrNumber]?) {
            let rope = Self.rope(config)
            let q = Self.input(length: 1, seed: 2)
            let k = Self.input(length: 1, seed: 3)
            func score(_ m: Int, _ n: Int) -> MLXArray {
                (rope(q, offset: m) * rope(k, offset: n)).sum(axis: -1)
            }
            #expect(
                SyntheticModel.maxAbsDifference(score(3, 1), score(9, 7)) <= 1e-4,
                "same distance")
            #expect(
                SyntheticModel.maxAbsDifference(score(3, 1), score(3, 2)) > 1e-4,
                "other distance")
        }
    }
}
