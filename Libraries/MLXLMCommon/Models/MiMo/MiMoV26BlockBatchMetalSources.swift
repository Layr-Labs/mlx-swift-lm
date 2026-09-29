// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0 AND MIT
// Wrapper only. No edit/copy of the shared ordinary attention_nax_bdv body.
enum MiMoV26BlockBatchMetalSources {
    static let source = #"""
          const uint block = threadgroup_position_in_grid.z;
          const device int* row = descriptors + block * 18;
          const device omlx_nax::AttnParams* params =
              reinterpret_cast<const device omlx_nax::AttnParams*>(row);
          // Uniform for the entire threadgroup, before any shared-body barrier.
          if (threadgroup_position_in_grid.x >= uint(params->NQ)) return;
          const int q_offset = row[14];
          const int k_offset = row[15];
          const int mask_offset = row[16];
          const int64_t qs[4] = {q_strides[0],q_strides[1],q_strides[2],q_strides[3]};
          const int64_t ks[4] = {k_strides[0],k_strides[1],k_strides[2],k_strides[3]};
          const int64_t vs[4] = {v_strides[0],v_strides[1],v_strides[2],v_strides[3]};
          // Original masks are head-broadcast rank2 Boolean arrays, packed by block.
          const int64_t ms[4] = {0,0,params->kL,1};
          const uint3 local_tid(threadgroup_position_in_grid.x,threadgroup_position_in_grid.y,0);
          omlx_nax::attention_nax_bdv<
              T,64,32,192,128,4,1,ALIGN_Q,ALIGN_K,HAS_MASK,DO_CAUSAL,HAS_SINKS,bool,float>(
              q + int64_t(q_offset)*qs[2],
              k + int64_t(k_offset)*ks[2],
              v + int64_t(k_offset)*vs[2],
              out + int64_t(q_offset)*params->H*128,
              params,qs,ks,vs,ms,mask + mask_offset,sinks,
              simdgroup_index_in_threadgroup,local_tid);
        """#
}
