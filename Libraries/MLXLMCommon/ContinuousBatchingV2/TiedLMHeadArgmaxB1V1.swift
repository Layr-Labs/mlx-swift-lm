// LGH-B1 --- the one-row half of LGH-001: the tied LM head returns the TOKEN,
// never the logits.
//
// WHAT THE TAIL COSTS TODAY AT ONE ROW. `Gemma4TextModel.applyLMHead` tries
// the matrix-unit head first (`Gemma4MMAQuantizedGEMV`, pinned to the eight-row
// cohort) and then the tight-grid QMV host (`CBv2TiedLMHeadQMVV1`, also pinned
// to eight rows); at one row BOTH fail closed and the head is stock
// `QuantizedEmbedding.asLinear` --- ordinary `affine_qmv` (K = 2816 is not a
// multiple of 512, so `affine_qmv_fast` never launches, and every promoted
// cross-row tier in `quantized.h` is gated on `ntg.x == 8`). Three dispatches
// then close the step, all of them on the token-serial tail with nothing to
// overlap them:
//
//   1. `affine_qmv`      -> `[1, 262144]` bf16 logits, a 0.5 MB store;
//   2. `tanh(x / 30) * 30` (the final softcap) -> reads that plane and writes
//      a float32 one, 1.5 MB of traffic;
//   3. `ArgReduce` -> reads the capped plane back, 1 MB, to produce ONE int.
//
// WHAT THIS DOES. Two dispatches, no logits plane:
//
//   1. `cbv2_b1_tied_lmhead_qmv_argmax_v1` --- the stock `qmv_impl` body with
//      the vocabulary STORE replaced by an in-register top-1 over the four
//      results each simdgroup already holds. Each threadgroup emits one
//      `(float, uint)` record instead of eight bf16 logits: 256 KB of partials
//      at the tied head's geometry against the 512 KB the store cost.
//   2. `cbv2_b1_tied_lmhead_argmax_reduce_v1` --- one threadgroup folds a
//      row's records into the token id.
//
// Net: one fewer dispatch at the position arm A28 priced at 3.3% of decode for
// a single ADDED dispatch (`ARGMAX-B1`, 81d3a98), and ~2.5 MB of tail traffic
// removed.
//
// EXACTNESS --- argmax-preserving, and bit-exact in the values compared.
//
//   * The GEMV is `qmv_impl`'s body verbatim (helpers `load_vector`,
//     `load_vector_safe`, `qdot`, `qdot_safe` are copied byte for byte from
//     the pinned `mlx/backend/metal/kernels/quantized.h`). Every output
//     column keeps its own K-chain, its own `simd_sum`, and the same
//     `static_cast<T>` rounding the store would have applied --- the fused
//     comparison therefore compares the SAME bf16 values stock's `argMax`
//     would have seen. (`CBv2TiedLMHeadQMVV1`'s LINEAGE note applies: a
//     hand-inlined expansion of this arithmetic measured 1-4 ulp off the
//     library kernel under this compiler, so the helpers are included
//     verbatim rather than rewritten.)
//   * The reduction orders records lexicographically on `(value, -index)`:
//     the higher value wins and equal values keep the LOWER index, which is
//     stock `argMax`'s first-index-wins rule. That order is associative and
//     commutative, so the per-simdgroup fold, the threadgroup fold and the
//     second-stage fold return the same answer whatever order they visit
//     partials in.
//   * THE SOFTCAP IS ELIDED, and eliding it is a THEOREM, not a tolerance:
//     `f(x) = tanh(x / c) * c` has `f'(x) = sech^2(x / c) > 0` for every
//     finite `x` and every `c > 0`, so `f` is strictly increasing and
//     `argmax_i f(x_i) = argmax_i x_i` --- ties included, because `f` is a
//     function (equal inputs give equal outputs, so the tie set is
//     preserved and first-index-wins picks the same element).
//
//     The one place FLOATING-POINT `f` can differ from mathematical `f` is
//     saturation: `tanh` in float32 reaches exactly 1.0 for arguments beyond
//     roughly 9.4, so two DISTINCT stored bf16 logits above `c * 9.4 = 282`
//     map to one float and stock's argmax would then take the lower index
//     while this kernel takes the strictly larger logit. Gemma 4's
//     pre-softcap logits live two orders of magnitude below that (the cap
//     exists precisely because the model is trained against it), and
//     `Gemma4TextModel`'s `DARKBLOOM_GEMMA4_LOGITSLESS_HEAD_VERIFY` arm
//     reports the observed peak alongside every verified step. It is the
//     same argument the eight-row `Gemma4MMAQuantizedGEMV.applyArgmax` path
//     already ships under.
//
// FAIL-CLOSED. `argmax` returns nil --- and the caller keeps the full-logits
// road byte for byte --- unless every one of these holds:
//
//   * `DARKBLOOM_CBV2_LOGITSLESS_GREEDY_HEAD` is not one of `0`/`false`/
//     `no`/`off`;
//   * the activation is bf16, shaped `[1, K]` or `[1, 1, K]` --- ONE row.
//     More rows are refused on purpose: this body serves one activation row
//     per x-group, so an eight-row plane would stream the 369 MB vocabulary
//     eight times, which is exactly what the promoted cross-row tiers and
//     `Gemma4MMAQuantizedGEMV` exist to avoid. The cohort keeps them.
//   * affine group size 64, bits 4, bf16 scales and biases, uint32 weight;
//   * `K % 64 == 0` (whole affine groups) and `K % 8 == 0` (whole uint32
//     words);
//   * `N % 8 == 0`, so `used_out_row == out_row` for every threadgroup and
//     the "last tile moved back" branch of `qmv_impl` cannot fire. (Even if
//     it did the answer would be unchanged --- a repeated output column is a
//     duplicate candidate with an identical value and index --- but pinning
//     it keeps the emitted partial count exactly `N / 8`.)
//
// Engage mark: `logitsless-greedy-head-b1`.

import Foundation
import MLX
import MLXFast

/// Engine-side admission for the LGH-001 seam, factored out for a device-free
/// test. `EngineLoopV2.launchChainedDecode` takes the fused road exactly when
/// every term holds.
///
/// `anyTokenConstraint` is the term the protocol file's own doc comment does
/// NOT cover: a hard grammar mask sends forbidden ids to `-infinity`, and if
/// the raw argmax is one of the forbidden ids the masked argmax is a
/// different token. Such a step's tokens are therefore not `argMax` of the
/// raw logits and the seam does not apply.
@inline(__always)
public func cbv2LogitslessGreedyStepAdmits(
    enabled: Bool,
    anyTokenConstraint: Bool,
    anyPositionState: Bool,
    isRecurrent: Bool,
    modelAdmitsArgmaxDecode: Bool,
    samplerAdmitsFusedGreedy: Bool
) -> Bool {
    enabled && !anyTokenConstraint && !anyPositionState && !isRecurrent
        && modelAdmitsArgmaxDecode && samplerAdmitsFusedGreedy
}

public enum CBv2TiedLMHeadArgmaxB1V1 {

    /// One switch for the whole candidate (C3). `0`/`false`/`no`/`off`
    /// restores the logits-then-softcap-then-argmax tail inside the same
    /// executable; the engine seam in `EngineLoopV2` reads the same flag, so
    /// off means the fused head is never even asked about.
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_LOGITSLESS_GREEDY_HEAD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    /// Rows one x-group serves. See FAIL-CLOSED.
    private static let rows = 1
    private static let groupSize = 64
    private static let bits = 4
    /// `num_simdgroups * results_per_simdgroup` in `qmv_impl`.
    private static let outputsPerGroup = 8
    private static let simdWidth = 32
    private static let simdGroups = 2
    /// Threads in the second-stage reduce threadgroup. One threadgroup per
    /// activation row; `NT` partials are strided across these threads and
    /// folded by simd butterfly plus one threadgroup pass.
    private static let reduceThreads = 256

    /// `load_vector`, `load_vector_safe`, `qdot` and `qdot_safe` verbatim from
    /// the pinned `mlx/backend/metal/kernels/quantized.h`, plus the pack-factor
    /// helpers they need. Only the bits == 4 arms are ever instantiated here;
    /// the rest are kept so the compiler sees the same code shape the JIT
    /// library compiles.
    private static let helpers = """
        #define METAL_FUNC inline
        constant constexpr const int SIMD_SIZE = 32;

        template <int bits, int wsize = 8>
        inline constexpr short get_pack_factor() {
          return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
        }

        template <int bits, int wsize = 8>
        inline constexpr short get_bytes_per_pack() {
          constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
          return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
        }

        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector(const device T* x, thread U* x_thread) {
          static_assert(
              bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
                  bits == 8,
              "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

          U sum = 0;

          if (bits == 4) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 16.0f;
              x_thread[i + 2] = x[i + 2] / 256.0f;
              x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
          }

          else if (bits == 8) {
            for (int i = 0; i < values_per_thread; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }

          return sum;
        }

        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
          static_assert(
              bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
                  bits == 8,
              "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

          U sum = 0;

          if (bits == 4) {
            for (int i = 0; i < N; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 16.0f;
              x_thread[i + 2] = x[i + 2] / 256.0f;
              x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
          }

          else if (bits == 8) {
            for (int i = 0; i < N; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }

          for (int i = N; i < values_per_thread; i++) {
            x_thread[i] = 0;
          }

          return sum;
        }

        template <typename U, int values_per_thread, int bits>
        inline U qdot(
            const device uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum) {
          static_assert(
              bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
                  bits == 8,
              "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

          U accum = 0;

          if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < (values_per_thread / 4); i++) {
              accum +=
                  (x_thread[4 * i] * (ws[i] & 0x000f) +
                   x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                   x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                   x_thread[4 * i + 3] * (ws[i] & 0xf000));
            }
          }

          else if (bits == 8) {
            for (int i = 0; i < values_per_thread; i++) {
              accum += x_thread[i] * w[i];
            }
          }

          return scale * accum + sum * bias;
        }

        template <typename U, int values_per_thread, int bits>
        inline U qdot_safe(
            const device uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum,
            int N) {
          static_assert(
              bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
                  bits == 8,
              "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

          U accum = 0;

          if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < (N / 4); i++) {
              accum +=
                  (x_thread[4 * i] * (ws[i] & 0x000f) +
                   x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                   x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                   x_thread[4 * i + 3] * (ws[i] & 0xf000));
            }
          }

          else if (bits == 8) {
            for (int i = 0; i < N; i++) {
              accum += x_thread[i] * w[i];
            }
          }

          return scale * accum + sum * bias;
        }

        // `qmv_impl`'s wide-N branch (`out_vec_size >= num_simdgroups *
        // results_per_simdgroup`), verbatim down to the last accumulation,
        // with ONE change: the four completed results are folded into a
        // (value, index) top-1 instead of being stored. `static_cast<T>` is
        // kept exactly where the store applied it, so the values compared are
        // the bf16 the logits plane would have held.
        template <typename T, int group_size, int bits>
        METAL_FUNC void qmv_argmax_impl(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            device float* pv,
            device uint* pi,
            const int in_vec_size,
            const int out_vec_size,
            const int partials_per_row,
            threadgroup float* tgv,
            threadgroup uint* tgi,
            uint3 tid,
            uint simd_gid,
            uint simd_lid) {
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;

          const device uint8_t* ws = (const device uint8_t*)w;

          typedef float U;

          thread U x_thread[values_per_thread];
          thread U result[results_per_simdgroup] = {0};

          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
              simd_gid * results_per_simdgroup;
          const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

          ws += used_out_row * in_vec_size_w +
              simd_lid * packs_per_thread * bytes_per_pack;
          scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += tid.x * in_vec_size + simd_lid * values_per_thread;

          int k = 0;
          for (; k <= in_vec_size - block_size; k += block_size) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;

              U s = sl[0];
              U b = bl[0];
              result[row] +=
                  qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }

            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
          }
          const int tail_values = static_cast<int>(in_vec_size - k);
          if (tail_values > 0) {
            if (tail_values % values_per_thread == 0) {
              const uint active_tail_lanes = uint(tail_values / values_per_thread);
              if (simd_lid < active_tail_lanes) {
                U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

                for (int row = 0; row < results_per_simdgroup; row++) {
                  auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                  const device T* sl = scales + row * in_vec_size_g;
                  const device T* bl = biases + row * in_vec_size_g;

                  U s = sl[0];
                  U b = bl[0];
                  result[row] +=
                      qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
                }
              }
            } else {
              const int remaining = clamp(
                  static_cast<int>(tail_values - simd_lid * values_per_thread),
                  0,
                  values_per_thread);
              if (remaining > 0) {
                U sum = load_vector_safe<T, U, values_per_thread, bits>(
                    x, x_thread, remaining);

                for (int row = 0; row < results_per_simdgroup; row++) {
                  auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                  const device T* sl = scales + row * in_vec_size_g;
                  const device T* bl = biases + row * in_vec_size_g;

                  U s = sl[0];
                  U b = bl[0];
                  result[row] += qdot_safe<U, values_per_thread, bits>(
                      wl, x_thread, s, b, sum, remaining);
                }
              }
            }
          }

          // The store the reference makes here is
          //   `y[row] = static_cast<T>(result[row])`.
          // Same reduction, same rounding; only the consumer differs.
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
          if (simd_lid == 0) {
            float bv = -INFINITY;
            uint bi = 0xFFFFFFFFu;
            // Ascending column order with STRICTLY-greater displacement, so a
            // tie keeps the lower column --- stock argMax's rule.
            for (int row = 0; row < results_per_simdgroup; row++) {
              const float v = float(static_cast<T>(result[row]));
              const uint idx = uint(used_out_row + row);
              if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
            }
            tgv[simd_gid] = bv;
            tgi[simd_gid] = bi;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_gid == 0 && simd_lid == 0) {
            float bv = tgv[0];
            uint bi = tgi[0];
            for (uint s = 1; s < uint(num_simdgroups); ++s) {
              const float ov = tgv[s];
              const uint oi = tgi[s];
              if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            pv[tid.x * uint(partials_per_row) + tid.y] = bv;
            pi[tid.x * uint(partials_per_row) + tid.y] = bi;
          }
        }
        """

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b1_tied_lmhead_qmv_argmax_v1",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["pv", "pi"],
        source: """
            const uint3 tid = threadgroup_position_in_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            threadgroup float tgv[2];
            threadgroup uint tgi[2];

            qmv_argmax_impl<T, 64, 4>(
                w,
                scales,
                biases,
                x,
                pv,
                pi,
                K,
                OUTN,
                OUTN / 8,
                tgv,
                tgi,
                tid,
                simd_gid,
                simd_lid);
            return;
            """,
        header: helpers,
        ensureRowContiguous: true
    )

    /// Stage two. One threadgroup per activation row folds that row's `NT`
    /// records under the same total order and emits the token id.
    private static let reduceKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_b1_tied_lmhead_argmax_reduce_v1",
        inputNames: ["pv", "pi"],
        outputNames: ["tokens"],
        source: """
            const uint m = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            constexpr uint NSG = uint(RT) / 32u;

            threadgroup float sv[NSG];
            threadgroup uint si[NSG];

            const device float* rowV = pv + m * uint(NT);
            const device uint* rowI = pi + m * uint(NT);

            float rv = -INFINITY;
            uint ri = 0xFFFFFFFFu;
            for (uint i = lid; i < uint(NT); i += uint(RT)) {
                const float v = rowV[i];
                const uint idx = rowI[i];
                if (v > rv || (v == rv && idx < ri)) { rv = v; ri = idx; }
            }
            for (ushort xm = 1; xm < 32; xm <<= 1) {
                const float ov = simd_shuffle_xor(rv, xm);
                const uint oi = simd_shuffle_xor(ri, xm);
                if (ov > rv || (ov == rv && oi < ri)) { rv = ov; ri = oi; }
            }
            if (lane == 0) {
                sv[sg] = rv;
                si[sg] = ri;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0) {
                for (uint s = 1; s < NSG; ++s) {
                    const float ov = sv[s];
                    const uint oi = si[s];
                    if (ov > rv || (ov == rv && oi < ri)) { rv = ov; ri = oi; }
                }
                tokens[m] = int32_t(ri);
            }
            """,
        ensureRowContiguous: true
    )

    /// Everything the admission decides, as plain numbers and flags. Kept
    /// separate from the arrays so the predicate is exercised without a
    /// Metal device (this file's tests construct no `MLXArray`).
    public struct Geometry: Equatable {
        public var rows: Int
        public var k: Int
        public var n: Int
        /// `w.dim(1)` — packed uint32 columns.
        public var weightPackedColumns: Int
        public var scaleRows: Int
        public var scaleGroups: Int
        public var biasRows: Int
        public var biasGroups: Int
        public var groupSize: Int
        public var bits: Int
        /// `[rows, K]`, or `[rows, 1, K]` — one activation position.
        public var singlePosition: Bool
        public var bf16Activations: Bool
        public var bf16ScalesAndBiases: Bool
        public var uint32Weight: Bool
        public var hasBiases: Bool

        public init(
            rows: Int, k: Int, n: Int, weightPackedColumns: Int,
            scaleRows: Int, scaleGroups: Int, biasRows: Int, biasGroups: Int,
            groupSize: Int, bits: Int, singlePosition: Bool,
            bf16Activations: Bool, bf16ScalesAndBiases: Bool,
            uint32Weight: Bool, hasBiases: Bool
        ) {
            self.rows = rows
            self.k = k
            self.n = n
            self.weightPackedColumns = weightPackedColumns
            self.scaleRows = scaleRows
            self.scaleGroups = scaleGroups
            self.biasRows = biasRows
            self.biasGroups = biasGroups
            self.groupSize = groupSize
            self.bits = bits
            self.singlePosition = singlePosition
            self.bf16Activations = bf16Activations
            self.bf16ScalesAndBiases = bf16ScalesAndBiases
            self.uint32Weight = uint32Weight
            self.hasBiases = hasBiases
        }
    }

    /// Device-free core of the admission. See FAIL-CLOSED above for why each
    /// term is there.
    public static func admits(enabled: Bool, _ g: Geometry) -> Bool {
        guard enabled, g.hasBiases, g.singlePosition else { return false }
        guard g.groupSize == Self.groupSize, g.bits == Self.bits else { return false }
        guard g.bf16Activations, g.bf16ScalesAndBiases, g.uint32Weight else { return false }
        // ONE activation row: this body serves one row per x-group.
        guard g.rows == Self.rows else { return false }
        guard g.n >= outputsPerGroup, g.n % outputsPerGroup == 0 else { return false }
        guard g.k > 0, g.k % g.groupSize == 0, g.k % 8 == 0 else { return false }
        // `qmv_impl` walks K in whole `values_per_thread * SIMD` blocks and
        // then a tail; the plane must be at least one block deep for the
        // first walk to be well formed.
        guard g.k >= 8 * simdWidth else { return false }
        guard g.weightPackedColumns == g.k * g.bits / 32 else { return false }
        guard g.scaleRows == g.n, g.biasRows == g.n else { return false }
        guard g.scaleGroups == g.k / g.groupSize, g.biasGroups == g.k / g.groupSize
        else { return false }
        return true
    }

    /// True when `argmax` would take the fused path for this geometry. Pure
    /// host predicate over shapes, dtypes and the kill switch, so the engine
    /// can choose the seam BEFORE it builds the forward graph.
    public static func admits(
        x shape: [Int],
        xDType: DType,
        w: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int
    ) -> Bool {
        guard w.ndim == 2, scales.ndim == 2, (biases?.ndim ?? 2) == 2 else { return false }
        guard shape.count == 2 || (shape.count == 3 && shape[1] == 1) else { return false }
        guard let rowCount = shape.first, let k = shape.last else { return false }
        return admits(
            enabled: enabled,
            Geometry(
                rows: rowCount,
                k: k,
                n: w.dim(0),
                weightPackedColumns: w.dim(1),
                scaleRows: scales.dim(0),
                scaleGroups: scales.dim(1),
                biasRows: biases?.dim(0) ?? -1,
                biasGroups: biases?.dim(1) ?? -1,
                groupSize: groupSize,
                bits: bits,
                singlePosition: true,
                bf16Activations: xDType == .bfloat16,
                bf16ScalesAndBiases: scales.dtype == .bfloat16
                    && (biases?.dtype ?? .float32) == .bfloat16,
                uint32Weight: w.dtype == .uint32,
                hasBiases: biases != nil))
    }

    /// The tied head's top-1 token, `[1]` int32, or `nil` when `admits` does
    /// not hold. The `[1, N]` logits plane is never materialised and the
    /// final softcap never runs.
    public static func argmax(
        x: MLXArray,
        w: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        guard
            admits(
                x: x.shape, xDType: x.dtype, w: w, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits),
            let biases
        else { return nil }

        let k = x.dim(-1)
        let n = w.dim(0)
        let flatX = x.reshaped([rows, k])
        let partialsPerRow = n / outputsPerGroup

        let partials = kernel(
            [flatX, w, scales, biases],
            template: [("T", x.dtype), ("K", k), ("OUTN", n)],
            grid: (rows * simdWidth, partialsPerRow * simdGroups, 1),
            threadGroup: (simdWidth, simdGroups, 1),
            outputShapes: [[rows * partialsPerRow], [rows * partialsPerRow]],
            outputDTypes: [.float32, .uint32]
        )

        return reduceKernel(
            [partials[0], partials[1]],
            template: [("NT", partialsPerRow), ("RT", reduceThreads)],
            grid: (rows * reduceThreads, 1, 1),
            threadGroup: (reduceThreads, 1, 1),
            outputShapes: [[rows]],
            outputDTypes: [.int32]
        )[0]
    }
}
