// pagedattention.metal
//
// CBv2 paged-attention decode kernels (ContinuousBatchingV2, WS-C).
//
// Two-pass flash-decoding over paged K/V slabs. Pass A ("part") computes
// unnormalized attention partials over fixed-size token PARTITIONS of each
// sequence; pass B ("merge") reduces the partials per query head with a
// global online-softmax merge. Design goals, in order:
//
//   1. Batch-composition invariance: each pass-A threadgroup touches exactly
//      one (sequence, kv-head, partition) triple, and each pass-B threadgroup
//      one (sequence, query-head) pair. A row's partition count depends only
//      on ITS OWN attended length (fixed PTOK), so its arithmetic — including
//      summation order — is bit-identical regardless of batchmates. Excess
//      threadgroups launched for longer batchmates return early and their
//      output slots are never read.
//   2. Enough parallelism to saturate memory bandwidth at B=1: splitting the
//      context into PTOK-token partitions launches B*KVH*ceil(len/PTOK)
//      threadgroups (vLLM V1/V2 partitioning, MLX sdpa_vector_2pass shape),
//      instead of B*KVH.
//   3. KV bytes are read exactly once per step: all GQA query heads that
//      share a kv head are processed by the same pass-A threadgroup, so K/V
//      rows are loaded once and reused for every query head (unlike the
//      vLLM/mistral.rs ports which launch one threadgroup per QUERY head).
//   4. Sliding windows are start-offset arithmetic (storage eviction is
//      handled by the host-side ring page tables) — never mask arrays.
//   5. Attention sinks (GPT-OSS) contribute to the softmax DENOMINATOR only,
//      folded in at merge time exactly like vLLM's `s_aux` / SGLang's
//      `deno += exp(sink - max)`.
//
// Attribution
// -----------
// The overall structure (block tables, online softmax over paged K/V,
// split-K partials + merge, denominator-only sinks with exp2 arithmetic,
// -FLT_MAX sentinel instead of -INF) is adapted from:
//   - vLLM's paged_attention_v2 CUDA kernels (Apache-2.0)
//     https://github.com/vllm-project/vllm
//   - mistral.rs Metal port, mistralrs-paged-attn/src/metal/kernels/
//     pagedattention.metal (MIT, Copyright (c) 2024 Eric Buehler)
//     https://github.com/EricLBuehler/mistral.rs
//   - vllm-metal kernels_v2/pagedattention.metal (Apache-2.0,
//     "Copyright contributors to the vLLM project")
//     https://github.com/vllm-project/vllm-metal
//   - Apple MLX sdpa_vector_2pass kernels (MIT-like MLX license, © Apple)
//     https://github.com/ml-explore/mlx
// The code below is a re-derivation for the MLXFast.metalKernel calling
// convention (auto-generated signature, body + header split); no lines are
// copied verbatim, but keep this header when editing.
//
// Calling convention (see PagedAttentionKernel.swift)
// ---------------------------------------------------
// Template parameters (pass A / pass B):
//   T          element type of q / kcache / vcache / out (half or float)
//   D          head dimension, one of {64, 128, 256, 512} (D % 32 == 0)
//   S          page size in tokens (16)
//   GQA        query heads per kv head (queryHeads / kvHeads)
//   NSG        simdgroups per pass-A threadgroup (32 KB smem budget)
//   PTOK       tokens per partition (512; PTOK % S == 0)
//   HAS_WRITE  (pass A) fuse the decode KV write: each row's new K/V tile
//              is written IN PLACE into the slabs before attending (see
//              "In-place slab writes" below)
//   HAS_SOFTCAP (pass A) apply logit softcapping: qk = cap * tanh(qk / cap)
//   HAS_SINKS   (pass B) fold per-head sink logits into the denominator
//
// Pass A inputs (all row-contiguous):
//   q        [B, KVH*GQA, D]   queries for this step (NOT pre-scaled;
//                              scale arrives via params[1])
//   knew     [B, KVH, D]       this step's key tile, slab dtype
//                              (HAS_WRITE only)
//   vnew     [B, KVH, D]       this step's value tile (HAS_WRITE only)
//   kcache   [P, KVH, S, D]    key slab for this layer group
//   vcache   [P, KVH, S, D]    value slab for this layer group
//   tables   [B, MAXP] int32   physical page id per logical page slot;
//                              windowed rows are ring-indexed (see below)
//   seqinfo  [B, 8]    int32   {attendStart, attendLen, tableLen,
//                               writePage, writeSlot, 0, 0, 0}
//                              (writePage/writeSlot read under HAS_WRITE)
//   params   [8]       float   {softcap, scale, 0...}
//
// In-place slab writes (HAS_WRITE + paged_kv_write)
// -------------------------------------------------
// The slabs are STABLE buffers written in place through const-cast device
// pointers — never versioned through MLX slice updates. Rationale: MLX's
// gpu::eval retains every op's INPUT data handles until the command buffer
// completes, so a slice-update on a slab that an attention kernel read in
// the same (or any in-flight) command buffer fails donation and degrades
// into a FULL-SLAB copy — measured at ~370 GiB of copies per decoded token
// on GPT-OSS-20B (docs/engine-v2/kernel-research.md §3).
//
// Safety argument (MLX encoders are concurrent with hazard tracking at
// declared-buffer granularity, so in-place writes are invisible to the
// tracker and ordering must come from graph edges):
//   - Decode (HAS_WRITE): the ONLY threadgroup that reads a row's newest
//     position is the one owning its last partition — the same threadgroup
//     that writes it (device-scope barrier in between). All other
//     threadgroups touch disjoint bytes. Windowed rings cannot alias: the
//     overwritten slot held a position >= ring*S older than the query,
//     which is outside every attendable window.
//   - Bulk writes (paged_kv_write, prefill chunks / prefix adoption): the
//     kernel emits a fence array chained through the group
//     (fence = prev + 1); every gather of the group's slabs consumes the
//     latest fence, giving both scheduling order and the memory barrier.
//   - Across engine steps, step N+1's graph always consumes step N's
//     outputs (chained decode feeds lazy tokens) or the loop host-syncs
//     at finalize — either forces step N's writes to complete first.
// Pass A outputs:
//   partials [B, KVH*GQA, MAXPART, D] float  unnormalized value partials
//   meta     [B, KVH*GQA, MAXPART, 2] float  {m, l} per partition (log2
//                                            domain; m == -FLT_MAX when the
//                                            partition holds no tokens)
// Pass B inputs: partials, meta, seqinfo, sinks [>=KVH*GQA] float
// Pass B output: out [B, KVH*GQA, D] dtype T
//
// Page resolution: token at absolute position p lives in table slot
// (p / S) % tableLen, slot p % S. Full-attention rows set tableLen to the
// table capacity so the modulo is the identity; windowed rows set tableLen
// to their ring length. attendStart/attendLen select exactly the window
// the current query may attend to (window clamping happens on the host
// from absolute positions — plain Swift Int math, no device sync).
//
// Pass A grid: threads = (KVH * 32 * NSG, B, MAXPART), tg = (32 * NSG, 1, 1).
// Pass B grid: threads = (KVH * GQA * 32, B, 1),       tg = (32, 1, 1).
//
// Threadgroup memory — SINGLE SOURCE OF TRUTH for eligibility
// -----------------------------------------------------------
// Pass A allocates exactly two threadgroup buffers (in the generated body,
// PagedAttentionMSL.partBody):
//   threadgroup float q_smem[GQA * D];
//   threadgroup float red_smem[NSG * GQA * (D + 2)];   // RSTRIDE = D + 2
// Pass B allocates NONE. Both buffers are float regardless of T, and the
// HAS_SOFTCAP / HAS_SINKS variants add none. The Swift-side eligibility
// math (PagedAttentionKernel.partThreadgroupBytes / ineligibilityReason /
// mergeRecordMetaFloats / threadgroupMemoryLimit) models these allocations
// byte-for-byte against Metal's 32 KB per-threadgroup cap; a dispatch over
// the cap is an UNCATCHABLE process fatal, so PagedKVBackend refuses such
// shapes at construction. If you change any threadgroup allocation here or
// in partBody, update those Swift constants (CBv2PagedEligibilityTests
// guards the pairing).

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

// Pass A: one threadgroup handles one (sequence b, kv head, partition)
// triple and all GQA query heads that map onto that kv head. Each simdgroup
// consumes the partition's tokens strided by NSG with a private online
// softmax per query head; the simdgroup states are merged through
// threadgroup memory and written out UNNORMALIZED with their (m, l) meta
// (flash-decoding split-K).
template <
    typename T, int D, int S, int GQA, int NSG, int PTOK, bool HAS_WRITE,
    bool HAS_SOFTCAP>
inline void paged_attention_part_impl(
    const device T* q,
    const device T* knew,
    const device T* vnew,
    const device T* kcache,
    const device T* vcache,
    const device int32_t* tables,
    const device int32_t* seqinfo,
    const device float* params,
    const int kvh,
    const int maxp,
    const int maxpart,
    threadgroup float* q_smem,   // [GQA * D]
    threadgroup float* red_smem, // [NSG * GQA * (D + 2)]
    device float* partials,
    device float* meta,
    uint3 tgpig,
    uint3 tpitg,
    uint sgitg,
    uint lane
) {
    constexpr int EPT = D / 32;      // K/V elements owned per lane
    constexpr int TG = 32 * NSG;     // threads per threadgroup
    constexpr int RSTRIDE = D + 2;   // per-(sg, head) merge record: acc[D], m, l
                                     // (mirrored: PagedAttentionKernel.mergeRecordMetaFloats)

    const int kv_head = tgpig.x;
    const int b = tgpig.y;
    const int part = tgpig.z;
    const int lin = tpitg.x;

    const device int32_t* info = seqinfo + b * 8;
    const int attend_start = info[0];
    const int attend_len = info[1];
    const int table_len = info[2];

    // Partitions past this row's attended length: nothing to do, and the
    // output slot is never read by the merge pass.
    const int lo = part * PTOK;
    if (lo >= attend_len) {
        return;
    }
    const int hi = min(attend_len, lo + PTOK);

    const device int32_t* table = tables + (size_t)b * (size_t)maxp;
    const float softcap = params[0];
    const float scale = params[1];

    // Fused decode write: the row's newest position lives in THIS row's
    // last partition, and only this threadgroup ever reads it — write the
    // step's K/V tile in place before attending (see file header). The
    // slabs are stable buffers; the const_cast is the documented in-place
    // write contract, not a hack around MLX donation. The (0, 0, 0)
    // threadgroup's lane 0 also advances the group's write-fence chain so
    // later dispatches that read the slabs order after this one.
    if (HAS_WRITE && part == (attend_len - 1) / PTOK) {
        const int wpage = info[3];
        const int wslot = info[4];
        const size_t dst =
            (((size_t)wpage * (size_t)kvh + (size_t)kv_head) * S + (size_t)wslot) * D;
        device T* kdst = const_cast<device T*>(kcache) + dst;
        device T* vdst = const_cast<device T*>(vcache) + dst;
        const size_t src = ((size_t)b * (size_t)kvh + (size_t)kv_head) * D;
        for (int i = lin; i < D; i += TG) {
            kdst[i] = knew[src + i];
            vdst[i] = vnew[src + i];
        }
    }

    // Stage the (scaled) queries for this kv head's GQA query heads.
    const int q_head0 = kv_head * GQA;
    const device T* q_base = q + ((size_t)b * (size_t)(kvh * GQA) + (size_t)q_head0) * D;
    for (int i = lin; i < GQA * D; i += TG) {
        q_smem[i] = float(q_base[i]) * scale;
    }
    // mem_device: the fused write above must be visible to this
    // threadgroup's own K/V loads below.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

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
    for (int i = lo + (int)sgitg; i < hi; i += NSG) {
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

    // Merge across simdgroups and write the UNNORMALIZED partition partial
    // plus its (m, l) meta. Query heads round-robin over simdgroups; lanes
    // cover the head dim.
    for (int g = sgitg; g < GQA; g += NSG) {
        float pm = -FLT_MAX;
        for (int s = 0; s < NSG; s++) {
            pm = max(pm, red_smem[(size_t)(s * GQA + g) * RSTRIDE + D]);
        }
        float pl = 0.0f;
        for (int s = 0; s < NSG; s++) {
            const float ms = red_smem[(size_t)(s * GQA + g) * RSTRIDE + D];
            const float ls = red_smem[(size_t)(s * GQA + g) * RSTRIDE + D + 1];
            pl += ls * exp2(ms - pm);
        }

        const size_t slot =
            ((size_t)b * (size_t)(kvh * GQA) + (size_t)(q_head0 + g)) * (size_t)maxpart
            + (size_t)part;
        device float* prow = partials + slot * D;
        for (int e = (int)lane; e < D; e += 32) {
            float v = 0.0f;
            for (int s = 0; s < NSG; s++) {
                const float ms = red_smem[(size_t)(s * GQA + g) * RSTRIDE + D];
                v += red_smem[(size_t)(s * GQA + g) * RSTRIDE + e] * exp2(ms - pm);
            }
            prow[e] = v;
        }
        if (lane == 0) {
            meta[slot * 2] = pm;
            meta[slot * 2 + 1] = pl;
        }
    }
}

// Pass B: one threadgroup (a single simdgroup) merges one (sequence, query
// head)'s partition partials with a global softmax merge; sinks fold into
// the denominator only. The partition count is recomputed from the row's
// OWN attended length, so slots written for longer batchmates are ignored.
template <typename T, int D, int PTOK, bool HAS_SINKS>
inline void paged_attention_merge_impl(
    const device float* partials,
    const device float* meta,
    const device int32_t* seqinfo,
    const device float* sinks,
    const int heads,
    const int maxpart,
    device T* out,
    uint3 tgpig,
    uint lane
) {
    constexpr int EPT = D / 32;

    const int head = tgpig.x;
    const int b = tgpig.y;
    const int attend_len = seqinfo[b * 8 + 1];
    const int parts = (attend_len + PTOK - 1) / PTOK;

    const size_t row = (size_t)b * (size_t)heads + (size_t)head;
    const device float* mrow = meta + row * (size_t)maxpart * 2;
    const device float* prow = partials + row * (size_t)maxpart * D;

    float gm = -FLT_MAX;
    for (int p = 0; p < parts; p++) {
        gm = max(gm, mrow[p * 2]);
    }
    float sink_scaled = -FLT_MAX;
    if (HAS_SINKS) {
        sink_scaled = sinks[head] * M_LOG2E_F;
        gm = max(gm, sink_scaled);
    }

    float gl = 0.0f;
    float acc[EPT];
#pragma unroll
    for (int e = 0; e < EPT; e++) {
        acc[e] = 0.0f;
    }
    for (int p = 0; p < parts; p++) {
        const float ms = mrow[p * 2];
        const float ls = mrow[p * 2 + 1];
        const float w = exp2(ms - gm);
        gl += ls * w;
        const device float* pp = prow + p * D;
#pragma unroll
        for (int e = 0; e < EPT; e++) {
            acc[e] += pp[lane * EPT + e] * w;
        }
    }
    if (HAS_SINKS) {
        // Denominator-only: the sink contributes exp(sink - max) to the
        // normalizer and nothing to the value accumulation.
        gl += exp2(sink_scaled - gm);
    }
    const float inv = gl > 0.0f ? 1.0f / gl : 0.0f;

    device T* orow = out + row * D;
#pragma unroll
    for (int e = 0; e < EPT; e++) {
        orow[lane * EPT + e] = T(acc[e] * inv);
    }
}

// Bulk in-place KV write (prefill chunks, prefix adoption): scatter an
// [KVH, N, D] tile into the slabs at host-computed physical slots
// (page * S + slot per token). Emits a chained fence value so gathers of
// the same group order after every prior bulk write (file header,
// "In-place slab writes"). Grid: threads = (D, KVH, N).
template <typename T, int S>
inline void paged_kv_write_impl(
    const device T* ktile,        // [KVH, N, D]
    const device T* vtile,        // [KVH, N, D]
    const device int32_t* slots,  // [>= max(N, 8)] physical page * S + slot
                                  // (padded to >= 8 so the generated
                                  // signature keeps the device space)
    const constant int32_t* prev, // [1] previous fence in the group chain
                                  // (constant space: MLX places < 8-element
                                  // inputs there)
    const device T* kcache,       // [P, KVH, S, D] written in place
    const device T* vcache,       // [P, KVH, S, D] written in place
    const int kvh,
    const int n,
    const int d,
    device int32_t* fence,        // [1] fence output (prev + 1)
    uint3 tpig
) {
    const int e = tpig.x;
    const int h = tpig.y;
    const int t = tpig.z;
    if (e >= d || h >= kvh || t >= n) {
        return;
    }
    const int idx = slots[t];
    const int page = idx / S;
    const int slot = idx % S;
    const size_t dst = (((size_t)page * (size_t)kvh + (size_t)h) * S + (size_t)slot) * d;
    const size_t src = ((size_t)h * (size_t)n + (size_t)t) * d;
    const_cast<device T*>(kcache)[dst + e] = ktile[src + e];
    const_cast<device T*>(vcache)[dst + e] = vtile[src + e];
    if (e == 0 && h == 0 && t == 0) {
        fence[0] = prev[0] + 1;
    }
}

} // namespace cbv2
