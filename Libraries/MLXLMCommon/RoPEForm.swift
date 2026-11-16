// Copyright © 2026 Eigen Labs / Darkbloom.
//
// How MLX's prebuilt `rope` kernel rounds in this process. The kernel writes
// `x1 * cos - x2 * sin` and `x1 * sin + x2 * cos`, and the compiler that built
// MLX's metallib decides which product of each pair it fuses into a
// multiply-add; its inverse frequency (`exp2(-d * log2(base))` or `1 / freq`)
// may use the precise or the fast function. A kernel copy compiled at run time
// must make the same choices to write the same bits, so this measures them:
// float32 rows against MLX (rows `(1, 0)` give MLX's cosine and sine exactly;
// random rows separate the multiply-add forms), then the chosen form against
// MLX in the product dtype. A form is reported only when exactly one candidate
// matches.

import Foundation
import MLX

public struct RoPEForm: Hashable, Sendable {
    /// The inverse frequency uses `metal::fast::exp2` / `fast::divide`
    /// (otherwise the precise functions).
    public let fastInverse: Bool
    /// `x1 * c - x2 * s`: 1 = `fma(x1, c, -(x2 * s))`, 2 = `fma(-x2, s, x1 * c)`.
    public let first: Int
    /// `x1 * s + x2 * c`: 1 = `fma(x1, s, x2 * c)`, 2 = `fma(x2, c, x1 * s)`.
    public let second: Int

    /// Template values for `rotationSource`.
    public var templates: [(String, any KernelTemplateArg)] {
        [("ROPE_FAST_INV", fastInverse ? 1 : 0), ("ROPE_RX1", first), ("ROPE_RX2", second)]
    }

    /// The inverse frequency of pair `i` (see `rotationSource`).
    static let inverseFrequencySource = """
        const float inv_freq = ROPE_FREQS == 1
          ? (ROPE_FAST_INV == 1 ? metal::fast::divide(1.0f, rope_freqs[i])
                                : metal::precise::divide(1.0f, rope_freqs[i]))
          : (ROPE_FAST_INV == 1
              ? metal::fast::exp2(-(static_cast<float>(i) / static_cast<float>(HALF)) * rope_base[0])
              : metal::precise::exp2(-(static_cast<float>(i) / static_cast<float>(HALF)) * rope_base[0]));
        """

    /// One pair of a non-traditional rotation, as MLX's `rope` kernel writes it.
    /// Reads `i` (pair index), `HALF` (template, `dims / 2`), `ROPE_FREQS`
    /// (template: 1 reads `rope_freqs[i]`, 0 uses `rope_base[0] = log2(base)`),
    /// `position` (float), `x1`, `x2`; defines `rx1` and `rx2`.
    public static let rotationSource = inverseFrequencySource + """
        const float theta = position * inv_freq;
        const float rope_c = metal::fast::cos(theta);
        const float rope_s = metal::fast::sin(theta);
        const float rx1 = ROPE_RX1 == 2 ? metal::fma(-x2, rope_s, x1 * rope_c)
                                        : metal::fma(x1, rope_c, -(x2 * rope_s));
        const float rx2 = ROPE_RX2 == 2 ? metal::fma(x2, rope_c, x1 * rope_s)
                                        : metal::fma(x1, rope_s, x2 * rope_c);
        """

    /// The form that writes MLX's prebuilt bits for a non-traditional
    /// rotation over all `dimensions` with scale 1, either from `base` (where
    /// `dimensions / 2` is a power of two, so `d = i / (dimensions / 2)` is
    /// exact in any form) or from `frequencies` (`dimensions / 2` float32
    /// values); nil where none is shown. Measured once per configuration.
    public static func measured(
        dimensions: Int, base: Float?, frequencies: MLXArray?, dtype: DType
    ) -> RoPEForm? {
        guard Device.defaultDevice().deviceType == .gpu,
            ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family()),
            [.bfloat16, .float16, .float32].contains(dtype),
            dimensions >= 2, dimensions % 2 == 0, dimensions <= 4096
        else { return nil }
        let half = dimensions / 2
        let source: Calibration.Source
        if let frequencies {
            guard base == nil, frequencies.ndim == 1, frequencies.dim(0) == half,
                frequencies.dtype == .float32
            else { return nil }
            source = .frequencies(frequencies.asArray(Float.self).map(\.bitPattern))
        } else {
            guard let base, half & (half - 1) == 0 else { return nil }
            source = .base(base.bitPattern)
        }
        return Calibration.shared.form(
            .init(dimensions: dimensions, source: source, dtype: dtype))
    }

    final class Calibration: @unchecked Sendable {
        static let shared = Calibration()

        enum Source: Hashable {
            case base(UInt32)
            case frequencies([UInt32])
        }

        struct Key: Hashable {
            let dimensions: Int
            let source: Source
            let dtype: DType
        }

        private let lock = NSLock()
        private var forms: [Key: RoPEForm?] = [:]

        func form(_ key: Key) -> RoPEForm? {
            if let known = lock.withLock({ forms[key] }) { return known }
            let measured = Self.measure(key)
            lock.withLock { forms[key] = measured }
            return measured
        }

        static let candidates: [RoPEForm] = [false, true].flatMap { fast in
            [1, 2].flatMap { first in
                [1, 2].map { RoPEForm(fastInverse: fast, first: first, second: $0) }
            }
        }

        /// Each candidate against MLX: float32 unit rows (cosine and sine),
        /// float32 random rows, and the product dtype; `matches` holds the
        /// candidates equal to MLX on all of them.
        struct Measurement {
            let matches: [RoPEForm]
            /// The fast and precise inverse frequencies differ for some pair.
            /// When they are equal for every pair, the two forms are the same
            /// arithmetic at every position.
            let inverseFormsDiffer: Bool
            /// Both multiply-add forms of each output differ on the random rows.
            let productFormsDiffer: Bool
        }

        static func measure(_ key: Key) -> RoPEForm? {
            guard let m = measurement(key), m.productFormsDiffer else { return nil }
            let matches = m.inverseFormsDiffer ? m.matches : m.matches.filter { !$0.fastInverse }
            return matches.count == 1 ? matches[0] : nil
        }

        /// Positions: every one below 1,024, then spread up to 2^20 so a
        /// different inverse frequency moves the angle by many units.
        static func positions() -> [Int32] {
            var state: UInt64 = 0x5851_F42D_4C95_7F2D
            var values = (0 ..< 1024).map { Int32($0) }
            for _ in 0 ..< 1024 {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                values.append(Int32(1024 + (state >> 33) % UInt64((1 << 20) - 1024)))
            }
            return values
        }

        static func measurement(_ key: Key) -> Measurement? {
            let dimensions = key.dimensions
            let half = dimensions / 2
            let positions = positions()
            let rows = positions.count
            let offsets = MLXArray(positions)
            let ropeBase: MLXArray
            let ropeFreqs: MLXArray
            let base: Float?
            let frequencies: MLXArray?
            switch key.source {
            case .base(let bits):
                base = Float(bitPattern: bits)
                frequencies = nil
                ropeBase = MLXArray([log2(Float(bitPattern: bits))])
                ropeFreqs = MLXArray([Float(1)])
            case .frequencies(let bits):
                base = nil
                let values = bits.map { Float(bitPattern: $0) }
                frequencies = MLXArray(values)
                ropeBase = MLXArray([Float(0)])
                ropeFreqs = frequencies!
            }
            func reference(_ x: MLXArray) -> MLXArray {
                MLXFast.RoPE(
                    x.reshaped(rows, 1, 1, dimensions), dimensions: dimensions,
                    traditional: false, base: base, scale: 1, offset: offsets, freqs: frequencies
                ).reshaped(rows, dimensions)
            }
            func copy(_ x: MLXArray, _ form: RoPEForm) -> MLXArray {
                kernel(
                    [x, offsets, ropeBase, ropeFreqs],
                    template: [
                        ("T", x.dtype), ("HALF", half),
                        ("ROPE_FREQS", frequencies == nil ? 0 : 1),
                    ] + form.templates,
                    grid: (half, rows, 1), threadGroup: (min(half, 256), 1, 1),
                    outputShapes: [[rows, dimensions]], outputDTypes: [x.dtype])[0]
            }
            func bytes(_ a: MLXArray) -> Data { a.asData(access: .copy).data }

            // Rows (1, 0): every multiply-add form writes cos and sin exactly.
            let unit = concatenated(
                [MLXArray.ones([rows, half]), MLXArray.zeros([rows, half])], axis: 1)
            // Random rows: values k / 64 with |k| < 512.
            var state: UInt64 = 0x9E37_79B9_7F4A_7C15 ^ UInt64(dimensions)
            let values: [Float] = (0 ..< rows * dimensions).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Float(Int((state >> 33) % 1023) - 511) / 64
            }
            let random = MLXArray(values, [rows, dimensions])
            let product = random.asType(key.dtype)

            let unitReference = bytes(reference(unit))
            let randomReference = bytes(reference(random))
            let productReference = bytes(reference(product))
            var matches: [RoPEForm] = []
            var unitOutputs: [Bool: Data] = [:]
            var randomOutputs: [RoPEForm: Data] = [:]
            for form in candidates {
                let u = bytes(copy(unit, form))
                let r = bytes(copy(random, form))
                unitOutputs[form.fastInverse] = u
                randomOutputs[form] = r
                if u == unitReference, r == randomReference,
                    bytes(copy(product, form)) == productReference
                {
                    matches.append(form)
                }
            }
            // MLX's single-row kernel (one row, one offset) for a few rows.
            if let chosen = matches.first {
                for row in stride(from: 0, to: rows, by: rows / 8) {
                    let x = product[row ..< row + 1]
                    let expected = MLXFast.RoPE(
                        x.reshaped(1, 1, 1, dimensions), dimensions: dimensions,
                        traditional: false, base: base, scale: 1,
                        offset: Int(positions[row]), freqs: frequencies
                    ).reshaped(1, dimensions)
                    let actual = kernel(
                        [x, offsets[row ..< row + 1], ropeBase, ropeFreqs],
                        template: [
                            ("T", x.dtype), ("HALF", half),
                            ("ROPE_FREQS", frequencies == nil ? 0 : 1),
                        ] + chosen.templates,
                        grid: (half, 1, 1), threadGroup: (min(half, 256), 1, 1),
                        outputShapes: [[1, dimensions]], outputDTypes: [x.dtype])[0]
                    if bytes(actual) != bytes(expected) { matches.removeAll() }
                }
            }
            // Every pair's inverse frequency in both forms, compared exactly.
            let inverses = [false, true].map { fast in
                bytes(
                    inverseKernel(
                        [ropeBase, ropeFreqs],
                        template: [
                            ("HALF", half), ("ROPE_FREQS", frequencies == nil ? 0 : 1),
                            ("ROPE_FAST_INV", fast ? 1 : 0),
                        ],
                        grid: (half, 1, 1), threadGroup: (min(half, 256), 1, 1),
                        outputShapes: [[half]], outputDTypes: [.float32])[0])
            }
            let inverseFormsDiffer = inverses[0] != inverses[1]
            // Where they differ, the unit rows must tell the two forms apart.
            if inverseFormsDiffer, unitOutputs[false] == unitOutputs[true] { return nil }
            func differ(_ a: RoPEForm, _ b: RoPEForm) -> Bool { randomOutputs[a] != randomOutputs[b] }
            let p = RoPEForm(fastInverse: false, first: 1, second: 1)
            let productFormsDiffer =
                differ(p, RoPEForm(fastInverse: false, first: 2, second: 1))
                && differ(p, RoPEForm(fastInverse: false, first: 1, second: 2))
            return Measurement(
                matches: matches, inverseFormsDiffer: inverseFormsDiffer,
                productFormsDiffer: productFormsDiffer)
        }

        private static let inverseKernel = MLXFast.metalKernel(
            name: "rope_form_inverse", inputNames: ["rope_base", "rope_freqs"],
            outputNames: ["out"],
            source: """
                const uint i = thread_position_in_grid.x;
                \(inverseFrequencySource)
                out[i] = inv_freq;
                """)

        // One rotation per thread: x [rows, 2 * HALF], row r at offsets[r].
        private static let kernel = MLXFast.metalKernel(
            name: "rope_form_probe", inputNames: ["x", "offsets", "rope_base", "rope_freqs"],
            outputNames: ["out"],
            source: """
                const uint i = thread_position_in_grid.x;
                const uint r = thread_position_in_grid.y;
                const size_t row = size_t(r) * (2 * HALF);
                const float position = static_cast<float>(uint(offsets[r]));
                const float x1 = static_cast<float>(x[row + i]);
                const float x2 = static_cast<float>(x[row + i + HALF]);
                \(rotationSource)
                out[row + i] = static_cast<T>(rx1);
                out[row + i + HALF] = static_cast<T>(rx2);
                """)
    }
}
