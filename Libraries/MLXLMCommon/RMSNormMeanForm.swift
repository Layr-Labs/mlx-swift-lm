// Copyright © 2026 Eigen Labs / Darkbloom.
//
// How MLX's prebuilt `rms_norm` rounds its mean in this process. The compiler
// that built MLX's metallib may emit `acc / axis_size + eps` as a division and
// an add or as one fused multiply-add, so a kernel copy compiled at run time
// must use the same form to write the same bits. Random rows rarely show the
// difference, so the probe builds rows of values k / 16 (exact sums of squares
// in any order) on which the two forms round differently. A form is reported
// only when the probe separates the compiled forms and exactly one matches MLX.

import Foundation
import MLX

public enum RMSNormMeanForm: Int, Sendable {
    /// `acc / axis_size + eps` as the run-time compiler emits it.
    case divide = 0
    /// `fma(acc, 1 / axis_size, eps)`.
    case fusedMultiplyAdd = 1

    /// The inverse mean for a `MEAN_FORM` template value, from `acc`,
    /// `axis_size` and `eps[0]`.
    public static let inverseMeanExpression = """
        (MEAN_FORM == 1 \
            ? metal::precise::rsqrt(metal::fma(acc, 1.0f / axis_size, eps[0])) \
            : metal::precise::rsqrt(acc / axis_size + eps[0]))
        """

    /// The form that writes MLX's prebuilt bits for rows of `width` (MLX's
    /// single-row kernel, at most 4096) in `dtype`; nil where none is shown.
    /// Measured once per width and dtype.
    public static func measured(width: Int, dtype: DType) -> RMSNormMeanForm? {
        guard Device.defaultDevice().deviceType == .gpu,
            ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family()),
            [.bfloat16, .float16, .float32].contains(dtype), (1 ... 4096).contains(width)
        else { return nil }
        return Calibration.shared.form(width: width, dtype: dtype)
    }

    package final class Calibration: @unchecked Sendable {
        static let shared = Calibration()

        private struct Key: Hashable {
            let width: Int
            let dtype: DType
        }

        private let lock = NSLock()
        private var forms: [Key: RMSNormMeanForm?] = [:]

        /// The form is the compiler's lowering, not a function of epsilon.
        package static let epsilon: Float = 1e-6
        static let separatingRows = 8
        static let plainRows = 24

        func form(width: Int, dtype: DType) -> RMSNormMeanForm? {
            let key = Key(width: width, dtype: dtype)
            if let known = lock.withLock({ forms[key] }) { return known }
            let measured = Self.measure(width: width, dtype: dtype)
            lock.withLock { forms[key] = measured }
            return measured
        }

        /// Each form against MLX on the probe rows.
        struct Measurement {
            /// A power-of-two width: `1 / D` and `acc / D` are exact, so the
            /// two forms are the same arithmetic.
            let formsCoincide: Bool
            let divideMatches: Bool
            let fusedMultiplyAddMatches: Bool
            let formsDiffer: Bool
        }

        static func measure(width: Int, dtype: DType) -> RMSNormMeanForm? {
            guard let m = measurement(width: width, dtype: dtype) else { return nil }
            if m.formsCoincide {
                return m.divideMatches && m.fusedMultiplyAddMatches ? .divide : nil
            }
            guard m.formsDiffer else { return nil }
            switch (m.divideMatches, m.fusedMultiplyAddMatches) {
            case (true, false): return .divide
            case (false, true): return .fusedMultiplyAdd
            default: return nil
            }
        }

        static func measurement(width: Int, dtype: DType) -> Measurement? {
            let coincide = width & (width - 1) == 0
            let (separating, plain) = rows(width: width, dtype: dtype, separate: !coincide)
            guard coincide || separating.count == separatingRows else { return nil }
            let all = separating + plain
            let x = MLXArray(all.flatMap { $0 }, [all.count, width]).asType(dtype)
            let weight = MLXArray.ones([width], dtype: dtype)
            let eps = MLXArray([epsilon])
            let reference = MLXFast.rmsNorm(x, weight: weight, eps: epsilon)
            let threads = 32 * (((width + 3) / 4 + 31) / 32)
            func copy(_ form: RMSNormMeanForm) -> MLXArray {
                kernel(
                    [x, weight, eps],
                    template: [("T", dtype), ("D", width), ("MEAN_FORM", form.rawValue)],
                    grid: (all.count * threads, 1, 1), threadGroup: (threads, 1, 1),
                    outputShapes: [[all.count, width]], outputDTypes: [dtype])[0]
            }
            let divide = copy(.divide)
            let fused = copy(.fusedMultiplyAdd)
            eval(reference, divide, fused)
            func bytes(_ a: MLXArray) -> Data { a.asData(access: .copy).data }
            let r = bytes(reference)
            let d = bytes(divide)
            let f = bytes(fused)
            return Measurement(
                formsCoincide: coincide, divideMatches: d == r, fusedMultiplyAddMatches: f == r,
                formsDiffer: d != f)
        }

        /// Rows on which the two forms' outputs differ (with a correctly rounded
        /// inverse square root), and plain rows.
        package static func rows(width: Int, dtype: DType, separate: Bool = true) -> (
            separating: [[Float]], plain: [[Float]]
        ) {
            var state = 0x243F_6A88_85A3_08D3 ^ UInt64(width) &* 0x9E37_79B9_7F4A_7C15
            func next() -> UInt64 {
                state &+= 0x9E37_79B9_7F4A_7C15
                var z = state
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                return z ^ (z >> 31)
            }
            // |k| <= bound keeps the sum of k^2 below 2^24, so it is exact in Float.
            let bound = max(1, min(48, Int((Double(1 << 24) / Double(width)).squareRoot())))
            let width32 = Float(width)
            let reciprocal = 1 / width32
            var separating: [[Float]] = []
            var plain: [[Float]] = []
            let wanted = separate ? separatingRows : 0
            for _ in 0 ..< 20_000 where separating.count < wanted || plain.count < plainRows {
                var prefixSquares = 0
                var row: [Float] = (0 ..< width).map { _ in
                    let magnitude = Int(next() % UInt64(bound)) + 1
                    prefixSquares += magnitude * magnitude
                    return Float(next() & 1 == 0 ? magnitude : -magnitude) / 16
                }
                let last = Int(abs(row[width - 1]) * 16)
                prefixSquares -= last * last
                if plain.count < plainRows { plain.append(row) }
                guard separating.count < wanted else { continue }
                // Sweep the last value: each magnitude gives another pair of means.
                for magnitude in 1 ... bound {
                    row[width - 1] = Float(magnitude) / 16
                    let sum = Float(prefixSquares + magnitude * magnitude) / 256
                    let inverseDivide = Float(1 / Double(sum / width32 + epsilon).squareRoot())
                    let inverseFused = Float(
                        1 / Double(epsilon.addingProduct(sum, reciprocal)).squareRoot())
                    guard inverseDivide != inverseFused else { continue }
                    if row.contains(where: {
                        rounded($0 * inverseDivide, dtype) != rounded($0 * inverseFused, dtype)
                    }) {
                        separating.append(row)
                        break
                    }
                }
            }
            return (separating, plain)
        }

        /// `value` stored as `dtype` (round to nearest even), back as Float.
        static func rounded(_ value: Float, _ dtype: DType) -> Float {
            switch dtype {
            case .bfloat16:
                let bits = value.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return Float(bitPattern: rounded & 0xFFFF_0000)
            case .float16:
                return Float(Float16(value))
            default:
                return value
            }
        }

        // MLX's rms_single_row (N_READS = 4) with the mean in either form.
        private static let kernel = MLXFast.metalKernel(
            name: "rms_norm_mean_form_probe", inputNames: ["x", "w", "eps"],
            outputNames: ["out"],
            source: """
                constexpr int N_READS = 4;
                constexpr int SIMD_SIZE = 32;
                const uint gid = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                const uint axis_size = D;
                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[SIMD_SIZE];
                const size_t row = gid * size_t(axis_size);
                const uint base = lid * N_READS;
                float acc = 0;
                float thread_x[N_READS];
                for (int i = 0; i < N_READS; i++) {
                  thread_x[i] = (base + i < axis_size) ? static_cast<float>(x[row + base + i]) : 0;
                  acc += thread_x[i] * thread_x[i];
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) { local_sums[simd_lane_id] = 0; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane_id == 0) { local_sums[simd_group_id] = acc; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                  acc = simd_sum(local_sums[simd_lane_id]);
                  if (simd_lane_id == 0) { local_inv_mean[0] = \(inverseMeanExpression); }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = 0; i < N_READS; i++) {
                  if (base + i < axis_size) {
                    out[row + base + i] = w[base + i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
                  }
                }
                """)
    }
}
