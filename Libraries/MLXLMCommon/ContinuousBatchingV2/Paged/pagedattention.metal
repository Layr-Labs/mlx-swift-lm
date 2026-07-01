// pagedattention.metal
//
// CBv2 paged-attention decode kernel (ContinuousBatchingV2, WS-C).
//
// This kernel computes single-token (decode) attention for a batch of
// sequences whose K/V live in a paged slab pool. Design goals, in order:
//
//   1. Batch-composition invariance: each threadgroup touches exactly one
//      (sequence, kv-head) pair. A row's output can never depend on its
//      batchmates: there is no cross-row arithmetic anywhere in the kernel.
//   2. KV bytes are read exactly once per step: all GQA query heads that
//      share a kv head are processed by the same threadgroup, so K/V rows
//      are loaded once and reused for every query head (unlike the
//      vLLM/mistral.rs port which launches one threadgroup per QUERY head
//      and re-reads K/V gqa times).
//   3. Sliding windows are start-offset arithmetic (storage eviction is
//      handled by the host-side ring page tables) — never mask arrays.
//   4. Attention sinks (GPT-OSS) contribute to the softmax DENOMINATOR
//      only, exactly like vLLM's `s_aux` / SGLang's `deno += exp(sink-max)`.
//
// Attribution
// -----------
// The overall structure (block tables, online softmax over paged K/V,
// denominator-only sinks with exp2 arithmetic, -FLT_MAX sentinel instead of
// -INF) is adapted from:
//   - vLLM's paged_attention CUDA kernel (Apache-2.0)
//     https://github.com/vllm-project/vllm
//   - mistral.rs Metal port, mistralrs-paged-attn/src/metal/kernels/
//     pagedattention.metal (MIT, Copyright (c) 2024 Eric Buehler)
//     https://github.com/EricLBuehler/mistral.rs
//   - vllm-metal kernels_v2/pagedattention.metal (Apache-2.0,
//     "Copyright contributors to the vLLM project")
//     https://github.com/vllm-project/vllm-metal
//   - Apple MLX sdpa_vector kernels (MIT-like MLX license, © Apple Inc.)
//     https://github.com/ml-explore/mlx
// The code below is a re-derivation for the MLXFast.metalKernel calling
// convention (auto-generated signature, body + header split); no lines are
// copied verbatim, but keep this header when editing.
//
// Calling convention (see PagedAttentionKernel.swift)
// ---------------------------------------------------
// Template parameters:
//   T          element type of q / kcache / vcache / out (half or float)
//   D          head dimension, one of {64, 128, 256, 512} (D % 32 == 0)
//   S          page size in tokens (16)
//   GQA        query heads per kv head (queryHeads / kvHeads)
//   NSG        simdgroups per threadgroup (chosen for the 32 KB smem budget)
//   HAS_SINKS  fold per-head sink logits into the softmax denominator
//   HAS_SOFTCAP apply logit softcapping: qk = cap * tanh(qk / cap)
//
// Inputs (all row-contiguous):
//   q        [B, KVH*GQA, D]   queries for this step (NOT pre-scaled;
//                              scale arrives via params[1])
//   kcache   [P, KVH, S, D]    key slab for this layer group
//   vcache   [P, KVH, S, D]    value slab for this layer group
//   tables   [B, MAXP] int32   physical page id per logical page slot;
//                              windowed rows are ring-indexed (see below)
//   seqinfo  [B, 8]    int32   {attendStart, attendLen, tableLen, 0, ...}
//   sinks    [>=KVH*GQA] float sink logits (dummy when !HAS_SINKS)
//   params   [8]       float   {softcap, scale, 0...}
// Output:
//   out      [B, KVH*GQA, D]   attention output, dtype T
//
// Page resolution: token at absolute position p lives in table slot
// (p / S) % tableLen, slot p % S. Full-attention rows set tableLen to the
// table capacity so the modulo is the identity; windowed rows set tableLen
// to their ring length. attendStart/attendLen select exactly the window
// the current query may attend to (window clamping happens on the host
// from absolute positions — plain Swift Int math, no device sync).
//
// Grid: threads = (KVH * 32 * NSG, B, 1), threadgroup = (32 * NSG, 1, 1).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_math>

using namespace metal;

namespace cbv2 {

// Vectorized row load: each lane owns EPT contiguous elements of a K/V row.
template <typename T, int EPT>
inline void load_row(const device T* row, uint lane, thread float* dst) {
    if (EPT % 4 == 0) {
        const device vec<T, 4>* v4 = reinterpret_cast<const device vec<T, 4>*>(row + lane * EPT);
#pragma unroll
        for (int e = 0; e < EPT / 4; e++) {
            vec<T, 4> v = v4[e];
            dst[e * 4 + 0] = float(v.x);
            dst[e * 4 + 1] = float(v.y);
            dst[e * 4 + 2] = float(v.z);
            dst[e * 4 + 3] = float(v.w);
        }
    } else {
        const device vec<T, 2>* v2 = reinterpret_cast<const device vec<T, 2>*>(row + lane * EPT);
#pragma unroll
        for (int e = 0; e < EPT / 2; e++) {
            vec<T, 2> v = v2[e];
            dst[e * 2 + 0] = float(v.x);
            dst[e * 2 + 1] = float(v.y);
        }
    }
}

// One threadgroup handles one (sequence b, kv head) pair and all GQA query
// heads that map onto that kv head. Each simdgroup consumes attended tokens
// strided by NSG, maintaining a private online-softmax state per query head;
// the per-simdgroup states are merged through threadgroup memory at the end
// (flash-decoding style split-K merge).
template <typename T, int D, int S, int GQA, int NSG, bool HAS_SINKS, bool HAS_SOFTCAP>
inline void paged_attention_impl(
    const device T* q,
    const device T* kcache,
    const device T* vcache,
    const device int32_t* tables,
    const device int32_t* seqinfo,
    const device float* sinks,
    const device float* params,
    const int kvh,
    const int maxp,
    threadgroup float* q_smem,   // [GQA * D]
    threadgroup float* red_smem, // [NSG * GQA * (D + 2)]
    device T* out,
    uint3 tgpig,
    uint3 tpitg,
    uint sgitg,
    uint lane
) {
    constexpr int EPT = D / 32;      // K/V elements owned per lane
    constexpr int TG = 32 * NSG;     // threads per threadgroup
    constexpr int RSTRIDE = D + 2;   // per-(sg, head) merge record: acc[D], m, l

    const int kv_head = tgpig.x;
    const int b = tgpig.y;
    const int lin = tpitg.x;

    const device int32_t* info = seqinfo + b * 8;
    const int attend_start = info[0];
    const int attend_len = info[1];
    const int table_len = info[2];
    const device int32_t* table = tables + (size_t)b * (size_t)maxp;

    const float softcap = params[0];
    const float scale = params[1];

    // Stage the (scaled) queries for this kv head's GQA query heads.
    const int q_head0 = kv_head * GQA;
    const device T* q_base = q + ((size_t)b * (size_t)(kvh * GQA) + (size_t)q_head0) * D;
    for (int i = lin; i < GQA * D; i += TG) {
        q_smem[i] = float(q_base[i]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-simdgroup online softmax state (identical across lanes for m/l,
    // per-lane slices of the value accumulator).
    float m[GQA];
    float l[GQA];
    float acc[GQA][EPT];
#pragma unroll
    for (int g = 0; g < GQA; g++) {
        m[g] = -FLT_MAX;
        l[g] = 0.0f;
#pragma unroll
        for (int e = 0; e < EPT; e++) {
            acc[g][e] = 0.0f;
        }
    }

    float kf[EPT];
    float vf[EPT];
    for (int i = sgitg; i < attend_len; i += NSG) {
        const int pos = attend_start + i;
        const int lpage = pos / S;
        const int slot = pos % S;
        const int phys = table[lpage % table_len];
        const size_t base = (((size_t)phys * (size_t)kvh + (size_t)kv_head) * S + (size_t)slot) * D;
        load_row<T, EPT>(kcache + base, lane, kf);
        load_row<T, EPT>(vcache + base, lane, vf);

#pragma unroll
        for (int g = 0; g < GQA; g++) {
            float partial = 0.0f;
#pragma unroll
            for (int e = 0; e < EPT; e++) {
                partial += q_smem[g * D + lane * EPT + e] * kf[e];
            }
            float qk = simd_sum(partial);
            if (HAS_SOFTCAP) {
                qk = softcap * precise::tanh(qk / softcap);
            }
            qk *= M_LOG2E_F;

            const float m_new = max(m[g], qk);
            const float corr = exp2(m[g] - m_new);
            const float p = exp2(qk - m_new);
            l[g] = l[g] * corr + p;
#pragma unroll
            for (int e = 0; e < EPT; e++) {
                acc[g][e] = acc[g][e] * corr + p * vf[e];
            }
            m[g] = m_new;
        }
    }

    // Publish per-simdgroup states.
    threadgroup float* mine = red_smem + (size_t)(sgitg * GQA) * RSTRIDE;
#pragma unroll
    for (int g = 0; g < GQA; g++) {
#pragma unroll
        for (int e = 0; e < EPT; e++) {
            mine[g * RSTRIDE + lane * EPT + e] = acc[g][e];
        }
        if (lane == 0) {
            mine[g * RSTRIDE + D] = m[g];
            mine[g * RSTRIDE + D + 1] = l[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge across simdgroups and write the output. Query heads are
    // distributed round-robin over simdgroups; lanes cover the head dim.
    for (int g = sgitg; g < GQA; g += NSG) {
        float gm = -FLT_MAX;
        for (int s = 0; s < NSG; s++) {
            gm = max(gm, red_smem[(size_t)(s * GQA + g) * RSTRIDE + D]);
        }
        float sink_scaled = -FLT_MAX;
        if (HAS_SINKS) {
            sink_scaled = sinks[q_head0 + g] * M_LOG2E_F;
            gm = max(gm, sink_scaled);
        }
        float gl = 0.0f;
        for (int s = 0; s < NSG; s++) {
            const float ms = red_smem[(size_t)(s * GQA + g) * RSTRIDE + D];
            const float ls = red_smem[(size_t)(s * GQA + g) * RSTRIDE + D + 1];
            gl += ls * exp2(ms - gm);
        }
        if (HAS_SINKS) {
            // Denominator-only: the sink contributes exp(sink - max) to the
            // normalizer and nothing to the value accumulation.
            gl += exp2(sink_scaled - gm);
        }
        const float inv = gl > 0.0f ? 1.0f / gl : 0.0f;

        device T* orow = out + ((size_t)b * (size_t)(kvh * GQA) + (size_t)(q_head0 + g)) * D;
        for (int e = (int)lane; e < D; e += 32) {
            float v = 0.0f;
            for (int s = 0; s < NSG; s++) {
                const float ms = red_smem[(size_t)(s * GQA + g) * RSTRIDE + D];
                v += red_smem[(size_t)(s * GQA + g) * RSTRIDE + e] * exp2(ms - gm);
            }
            orow[e] = T(v * inv);
        }
    }
}

} // namespace cbv2
