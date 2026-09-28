// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0 AND MIT
// Grouped descriptor wrapper for oMLX #4032 @12e5084f8cf888bb6e846147929419f044802343.
// The shared native-rounded attention_nax_bdv_impl arithmetic is unchanged.
enum MiMoV26NAXKeyRangeGroupedMetalSources {
    static let source = #"""
  const uint block = threadgroup_position_in_grid.z;
  const device int* row = params + block * 20;
  const device omlx_nax::AttnParams* p =
      reinterpret_cast<const device omlx_nax::AttnParams*>(row);
  // Uniform for this threadgroup, before any barrier in the shared body.
  if (threadgroup_position_in_grid.x >= uint(p->NQ)) return;
  const int query_offset = row[14];
  const int key_offset = row[15];
  const int mask_offset = row[16];
  const int state_offset = row[17];
  const int64_t qs[4] = {q_strides[0], q_strides[1], q_strides[2], q_strides[3]};
  const int64_t ks[4] = {k_strides[0], k_strides[1], k_strides[2], k_strides[3]};
  const int64_t vs[4] = {v_strides[0], v_strides[1], v_strides[2], v_strides[3]};
  const int64_t ms[4] = {0, 0, p->kL, 1};
  const uint3 local_tid(threadgroup_position_in_grid.x,
                        threadgroup_position_in_grid.y, 0);
  // Preserve the generated input address space: tiny inputs may be constant
  // pointers, while the full retained states are device pointers.
  auto state_in = state + ((SCORE_PASS == 0 && FIRST_RANGE)
      ? int(block) : state_offset);
  device float* state_destination = state_out +
      ((SCORE_PASS == 2 && LAST_RANGE) ? int(block) : state_offset);
  device T* destination = out + ((SCORE_PASS == 2 && LAST_RANGE)
      ? int64_t(query_offset) * p->H * 128 : int64_t(block));
  // Each descriptor owns its dummy output/state slot too. Passing local z=0
  // with one shared dummy would otherwise create same-address concurrent writes.
  if constexpr (HAS_MASK) {
    omlx_nax::attention_nax_bdv_impl<
        T, 64, 32, 192, 128, 4, 1, ALIGN_Q, ALIGN_K,
        true, DO_CAUSAL, HAS_SINKS, true, FIRST_RANGE, LAST_RANGE,
        SCORE_PASS, bool, float>(
        q + int64_t(query_offset) * q_strides[2],
        k + int64_t(key_offset) * k_strides[2],
        v + int64_t(key_offset) * v_strides[2], destination, p,
        qs, ks, vs, ms, mask + mask_offset, sinks,
        state_in, state_destination, row[18], row[19],
        simdgroup_index_in_threadgroup, local_tid);
  } else {
    omlx_nax::attention_nax_bdv_impl<
        T, 64, 32, 192, 128, 4, 1, ALIGN_Q, ALIGN_K,
        false, DO_CAUSAL, HAS_SINKS, true, FIRST_RANGE, LAST_RANGE,
        SCORE_PASS, bool, float>(
        q + int64_t(query_offset) * q_strides[2],
        k + int64_t(key_offset) * k_strides[2],
        v + int64_t(key_offset) * v_strides[2], destination, p,
        qs, ks, vs, ms, mask, sinks,
        state_in, state_destination, row[18], row[19],
        simdgroup_index_in_threadgroup, local_tid);
  }
"""#
}
