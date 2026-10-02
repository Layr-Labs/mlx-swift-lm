// SPDX-License-Identifier: Apache-2.0 AND MIT
// Matrix fragment layout/QK intent: oMLX #4033, c10e4ad57b180fa1aa88bf40532cba4362b6851a.
// Scalar recurrence/key blocks: Apple MLX core3fa8f25e6451174d7b06be372c3a24272b77d88e.
// Modified: native strided splits; per-key max/fast::exp/FP32 output recurrence;
// explicit float K fragments; no tile softmax, PV MMA, HS2 or partial downcast.
// Retained notices/full licenses are in this packet's ATTRIBUTION.md and
// LICENSE-OMLX / LICENSE-MLX; root must carry them when integrating.
enum MiMoV26SplitKeyAttentionMetalSources {
    static let pass1 = #"""
          constexpr int D = 192;
          constexpr int V = 128;
          constexpr int G = 16;
          constexpr int NKV = 4;
          constexpr int TILE = 8;
          constexpr int BANDS = ROWS * G / 8;
          constexpr int NT = BANDS * 32;
          constexpr int KLD = D + 8;
          constexpr int VLD = V + 8;
          constexpr int DK = D / 8;
          typedef float U;

          threadgroup T ktile[TILE * KLD];
          threadgroup T vtile[TILE * VLD];
          threadgroup U scores[BANDS * 64];
          const int kh = threadgroup_position_in_grid.x;
          const int block = threadgroup_position_in_grid.z;
          const int band = simdgroup_index_in_threadgroup;
          const int lane = thread_index_in_simdgroup;
          const int tid = thread_index_in_threadgroup;
          const int N = keys_shape[2];
          const int row = band / (G / 8);
          const int firstHead = kh * G + (band % (G / 8)) * 8;
          // Metal 8x8 fragment mapping, same as pinned oMLX. No half accumulation.
          const int qid = lane / 4;
          const int fm = (qid & 4) + ((lane / 2) % 4);
          const int fn = (qid & 2) * 2 + (lane % 2) * 2;
          const U sc = scale[0];
          const int64_t qHeadStride = queries_strides[1];
          const int64_t qRowStride = queries_strides[2];
          const int64_t qDimStride = queries_strides[3];
          const int64_t kSeqStride = keys_strides[2];
          const int64_t kDimStride = keys_strides[3];
          const int64_t vSeqStride = values_strides[2];
          const int64_t vDimStride = values_strides[3];
          const device T* kp = keys + int64_t(kh) * keys_strides[1];
          const device T* vp = values + int64_t(kh) * values_strides[1];

          simdgroup_matrix<U, 8, 8> Q[DK];
          if (MATRIX_QK) {
            const device T* qp = queries + int64_t(firstHead + fm) * qHeadStride + row * qRowStride;
            for (int kd = 0; kd < DK; ++kd) {
              // Scale BEFORE dot products in FP32, exactly like native vector Q.
              reinterpret_cast<thread vec<U, 2>&>(Q[kd].thread_elements()) =
                  vec<U, 2>(sc * U(qp[(kd * 8 + fn) * qDimStride]),
                            sc * U(qp[(kd * 8 + fn + 1) * qDimStride]));
            }
          }

          // Four lanes own one head's independent value dimensions. Their scalar
          // online m/l updates are identical; value elements never cross-reduce.
          const int headInBand = lane / 4;
          const int valueLane = lane % 4;
          const int head = firstHead + headInBand;
          U numerator[V / 4] = {0};
          U maximum = Limits<U>::finite_min;
          U denominator = 0;
          if (HAS_SINKS && block == 0) {
            maximum = U(sinks[head]);
            denominator = 1;
          }
          const int visible = N - ROWS + row + 1;
          const int iterations = block < N ? (N - 1 - block) / BLOCKS + 1 : 0;
          for (int base = 0; base < iterations; base += TILE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Each native split is block, block+BLOCKS, ... in the original order.
            // Ragged slots are zeroed before any matrix read and never enter softmax.
            for (int c = tid; c < TILE * D; c += NT) {
              const int r = c / D, d = c % D;
              const int key = block + (base + r) * BLOCKS;
              ktile[r * KLD + d] = key < N ? kp[int64_t(key) * kSeqStride + d * kDimStride] : T(0);
            }
            for (int c = tid; c < TILE * V; c += NT) {
              const int r = c / V, d = c % V;
              const int key = block + (base + r) * BLOCKS;
              vtile[r * VLD + d] = key < N ? vp[int64_t(key) * vSeqStride + d * vDimStride] : T(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (MATRIX_QK) {
              simdgroup_matrix<U, 8, 8> first = simdgroup_matrix<U, 8, 8>(0);
              simdgroup_matrix<U, 8, 8> second = simdgroup_matrix<U, 8, 8>(0);
              // Only QK reduction association differs from the native six-lane-
              // elements + simd_sum tree. K is widened exactly, not requantized.
              for (int kd = 0; kd < DK / 2; ++kd) {
                simdgroup_matrix<U, 8, 8> ka, kb;
                reinterpret_cast<thread vec<U, 2>&>(ka.thread_elements()) =
                    vec<U, 2>(U(ktile[fn * KLD + kd * 8 + fm]),
                              U(ktile[(fn + 1) * KLD + kd * 8 + fm]));
                reinterpret_cast<thread vec<U, 2>&>(kb.thread_elements()) =
                    vec<U, 2>(U(ktile[fn * KLD + (DK / 2 + kd) * 8 + fm]),
                              U(ktile[(fn + 1) * KLD + (DK / 2 + kd) * 8 + fm]));
                simdgroup_multiply_accumulate(first, Q[kd], ka, first);
                simdgroup_multiply_accumulate(second, Q[DK / 2 + kd], kb, second);
              }
              reinterpret_cast<thread vec<U, 2>&>(first.thread_elements()) +=
                  reinterpret_cast<thread vec<U, 2>&>(second.thread_elements());
              simdgroup_store(first, scores + band * 64, 8);
            } else {
              // Qualification control: exact pinned vector-QK reduction, but the
              // same staging, recurrence, partial layout and second pass as candidate.
              for (int h = 0; h < 8; ++h) {
                const device T* qp = queries + int64_t(firstHead + h) * qHeadStride + row * qRowStride;
                U query[D / 32];
                for (int d = 0; d < D / 32; ++d) {
                  query[d] = sc * U(qp[(lane * (D / 32) + d) * qDimStride]);
                }
                for (int r = 0; r < TILE; ++r) {
                  U score = 0;
                  for (int d = 0; d < D / 32; ++d) {
                    score += query[d] * U(ktile[r * KLD + lane * (D / 32) + d]);
                  }
                  score = simd_sum(score);
                  if (lane == 0) { scores[band * 64 + h * TILE + r] = score; }
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int r = 0; r < TILE; ++r) {
              const int key = block + (base + r) * BLOCKS;
              // Never replace prefix exclusion with a finite mask or p=0 product:
              // future/padded V (including NaN) must not enter this row's recurrence.
              if (key < visible) {
                const U score = scores[band * 64 + headInBand * TILE + r];
                const U newMaximum = max(maximum, score);
                const U factor = fast::exp(maximum - newMaximum);
                const U exponential = fast::exp(score - newMaximum);
                maximum = newMaximum;
                denominator = denominator * factor + exponential;
                for (int d = 0; d < V / 4; ++d) {
                  const int column = valueLane * (V / 4) + d;
                  numerator[d] = numerator[d] * factor + exponential * U(vtile[r * VLD + column]);
                }
              }
            }
          }
          const size_t partial = (size_t(head) * ROWS + row) * BLOCKS + block;
          if (valueLane == 0) {
            sums[partial] = denominator;
            maxs[partial] = maximum;
          }
          for (int d = 0; d < V / 4; ++d) {
            partials[partial * V + valueLane * (V / 4) + d] = numerator[d];
          }
        """#
}
