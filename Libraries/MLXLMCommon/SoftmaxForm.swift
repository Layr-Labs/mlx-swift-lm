// Copyright © 2026 Eigen Labs / Darkbloom.
//
// How MLX's prebuilt block softmax rounds its normalizer in this process.
// The kernel writes `1 / normalizer`; the compiler that built MLX's metallib
// may emit a precise or a fast reciprocal, so a kernel copy compiled at run
// time must use the same one to write the same bits. Measured on float32
// rows (where the two forms differ on many rows), then checked against MLX
// in the product dtype. Covered: rows of at most 128 values, which MLX
// reduces in one SIMD group.

import Foundation
import MLX

public enum SoftmaxForm: Int, Sendable {
    /// `metal::precise::divide(1.0f, normalizer)`.
    case precise = 0
    /// `metal::fast::divide(1.0f, normalizer)`.
    case fast = 1

    /// MLX's `softmax_single_row` (`N_READS = 4`, `AccT = float`) for one row
    /// of `AXIS` (template, at most 128) values in SIMD group 0. Every
    /// thread of the thread group runs it (it has barriers). Reads
    /// `softmax_load(p)` (a float, for `p < AXIS`), writes
    /// `softmax_store(p, value)` (value as float, before the cast to the
    /// output type), and uses thread-group arrays `local_max[32]` and
    /// `local_normalizer[32]` and the template `SOFTMAX_FORM`.
    public static let rowSource = """
        {
          const bool sm_active = simd_group_id == 0;
          const int sm_lid = int(simd_lane_id);
          float ld[4];
          if (sm_active) {
            for (int i = 0; i < 4; i++) {
              ld[i] = (sm_lid * 4 + i < AXIS) ? softmax_load(sm_lid * 4 + i)
                                               : -metal::numeric_limits<float>::infinity();
            }
            local_max[simd_lane_id] = -metal::numeric_limits<float>::infinity();
            local_normalizer[simd_lane_id] = 0;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          float maxval = -metal::numeric_limits<float>::max();
          if (sm_active) {
            for (int i = 0; i < 4; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
            maxval = simd_max(maxval);
            if (simd_lane_id == 0) { local_max[0] = maxval; }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sm_active) {
            maxval = simd_max(local_max[simd_lane_id]);
            if (simd_lane_id == 0) { local_max[0] = maxval; }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          float normalizer = 0;
          if (sm_active) {
            maxval = local_max[0];
            for (int i = 0; i < 4; i++) {
              const float exp_x = metal::fast::exp(ld[i] - maxval);
              ld[i] = exp_x;
              normalizer += exp_x;
            }
            normalizer = simd_sum(normalizer);
            if (simd_lane_id == 0) { local_normalizer[0] = normalizer; }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sm_active) {
            normalizer = simd_sum(local_normalizer[simd_lane_id]);
            if (simd_lane_id == 0) { local_normalizer[0] = normalizer; }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sm_active) {
            normalizer = SOFTMAX_FORM == 1 ? metal::fast::divide(1.0f, local_normalizer[0])
                                           : metal::precise::divide(1.0f, local_normalizer[0]);
            for (int i = 0; i < 4; i++) {
              if (sm_lid * 4 + i < AXIS) { softmax_store(sm_lid * 4 + i, ld[i] * normalizer); }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """

    /// The form that writes MLX's prebuilt bits for `softmax(precise: true)`
    /// over rows of `axis` values in `dtype`; nil where none is shown.
    public static func measured(axis: Int, dtype: DType) -> SoftmaxForm? {
        guard Device.defaultDevice().deviceType == .gpu,
            ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family()),
            [.bfloat16, .float16, .float32].contains(dtype), (1 ... 128).contains(axis)
        else { return nil }
        return Calibration.shared.form(axis: axis, dtype: dtype)
    }

    final class Calibration: @unchecked Sendable {
        static let shared = Calibration()

        private struct Key: Hashable {
            let axis: Int
            let dtype: DType
        }

        private let lock = NSLock()
        private var forms: [Key: SoftmaxForm?] = [:]

        func form(axis: Int, dtype: DType) -> SoftmaxForm? {
            let key = Key(axis: axis, dtype: dtype)
            if let known = lock.withLock({ forms[key] }) { return known }
            let measured = Self.measure(axis: axis, dtype: dtype)
            lock.withLock { forms[key] = measured }
            return measured
        }

        struct Measurement {
            let preciseMatches: Bool
            let fastMatches: Bool
            /// The two forms write different float32 rows.
            let formsDiffer: Bool
        }

        static func measure(axis: Int, dtype: DType) -> SoftmaxForm? {
            guard let m = measurement(axis: axis, dtype: dtype) else { return nil }
            if !m.formsDiffer {
                // The same values on every probed row; the precise form must
                // match (and the fast one then does too).
                return m.preciseMatches ? .precise : nil
            }
            switch (m.preciseMatches, m.fastMatches) {
            case (true, false): return .precise
            case (false, true): return .fast
            default: return nil
            }
        }

        static let rows = 4096

        static func measurement(axis: Int, dtype: DType) -> Measurement? {
            var state: UInt64 = 0xD1B5_4A32_D192_ED03 ^ UInt64(axis)
            let values: [Float] = (0 ..< rows * axis).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Float(Int((state >> 33) % 4095) - 2047) / 512
            }
            let x = MLXArray(values, [rows, axis])
            func bytes(_ a: MLXArray) -> Data { a.asData(access: .copy).data }
            func copy(_ x: MLXArray, _ form: SoftmaxForm) -> MLXArray {
                kernel(
                    [x],
                    template: [("T", x.dtype), ("AXIS", axis), ("SOFTMAX_FORM", form.rawValue)],
                    grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
                    outputShapes: [[rows, axis]], outputDTypes: [x.dtype])[0]
            }
            let reference = bytes(softmax(x, axis: -1, precise: true))
            let precise = bytes(copy(x, .precise))
            let fast = bytes(copy(x, .fast))
            var preciseMatches = precise == reference
            var fastMatches = fast == reference
            // The product dtype: the chosen form(s) must match MLX there too.
            let product = x.asType(dtype)
            let productReference = bytes(softmax(product, axis: -1, precise: true))
            if preciseMatches {
                preciseMatches = bytes(copy(product, .precise)) == productReference
            }
            if fastMatches { fastMatches = bytes(copy(product, .fast)) == productReference }
            return Measurement(
                preciseMatches: preciseMatches, fastMatches: fastMatches,
                formsDiffer: precise != fast)
        }

        private static let kernel = MLXFast.metalKernel(
            name: "softmax_form_probe", inputNames: ["x"], outputNames: ["out"],
            source: """
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                const size_t row = size_t(threadgroup_position_in_grid.x) * AXIS;
                threadgroup float local_max[32];
                threadgroup float local_normalizer[32];
                #define softmax_load(p) static_cast<float>(x[row + (p)])
                #define softmax_store(p, value) out[row + (p)] = static_cast<T>(value)
                \(rowSource)
                #undef softmax_load
                #undef softmax_store
                """)
    }
}
