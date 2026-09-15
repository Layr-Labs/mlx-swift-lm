// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Capability-gated Qwen4 sparse-GQA main attention. Fusion's production
// primitive is Steel `qwen4_qsa_sparse_gqa` (24q/2kv/D256, WM=2, BK=64,
// DC=64). `MLXFast.metalKernel` cannot `#include` mlx steel headers, so
// `Qwen4ExpQSASteel.swift` copies the 8×8 MMA helpers. Default dispatches
// that s64 kernel; `DARKBLOOM_QWEN4_QSA_STEEL=0` keeps scalar c8s.
// sg8 (`simdgroup_half8x8` QK) JIT'd but 8K 20.5 s vs 16.3 s scalar —
// not dispatched. Contract: chronological selected block IDs, four-token
// expansion + causal tail, FP32 softmax. Missing ABI / kill switch /
// bad shape → nil (portable gathered SDPA).

import Foundation
import MLX
import MLXLMCommon

public enum Qwen4ExpQSAInvocation: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var native = 0
    nonisolated(unsafe) private static var portable = 0
    nonisolated(unsafe) private static var dense = 0
    nonisolated(unsafe) private static var decode = 0
    nonisolated(unsafe) private static var verify = 0

    public struct Snapshot: Sendable, Equatable {
        public var native: Int
        public var portable: Int
        public var dense: Int
        /// Fusion `_gathered_text_decode` singleton steps (L=1 past budget).
        public var decode: Int = 0
        /// mlx-serve `qsaVerifyGatherAttn` (S=2..15 past the KV floor).
        public var verify: Int = 0

        public init(
            native: Int, portable: Int, dense: Int, decode: Int = 0, verify: Int = 0
        ) {
            self.native = native
            self.portable = portable
            self.dense = dense
            self.decode = decode
            self.verify = verify
        }

        public var line: String {
            "qsa native=\(native) portable=\(portable) dense=\(dense) decode=\(decode) verify=\(verify)"
        }
    }

    public static func reset() {
        lock.lock()
        native = 0
        portable = 0
        dense = 0
        decode = 0
        verify = 0
        lock.unlock()
    }

    public static func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            native: native, portable: portable, dense: dense, decode: decode, verify: verify)
    }

    static func recordDecode() {
        lock.lock()
        decode += 1
        lock.unlock()
    }

    static func recordVerify() {
        lock.lock()
        verify += 1
        lock.unlock()
    }

    static func recordNative() {
        lock.lock()
        native += 1
        lock.unlock()
    }

    static func recordPortable() {
        lock.lock()
        portable += 1
        lock.unlock()
    }

    static func recordDense() {
        lock.lock()
        dense += 1
        lock.unlock()
    }
}

/// Early native-QSA SIGKILL workaround: `eval` every gathered layer and
/// `Memory.clearCache` once keys pass 32K. Fusion only evals the first
/// native output (`qsa_fast._NATIVE_QSA_MAIN_PROVEN`). The per-layer drain
/// is the 128K taper (12 QSA layers × chunks after 32K). Default **off**.
/// Lab restore: `DARKBLOOM_QWEN4_QSA_LAYER_SYNC=1`.
public enum Qwen4ExpQSALayerSync: Sendable {
    public static let envFlag = "DARKBLOOM_QWEN4_QSA_LAYER_SYNC"
    public static let clearCacheKeyTokens = 32_768

    public static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    public static func shouldClearCache(
        keyTokens: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        isEnabled(environment: environment) && keyTokens >= clearCacheKeyTokens
    }

    static func afterAttend(
        _ output: MLXArray,
        keyTokens: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) {
        guard isEnabled(environment: environment) else { return }
        CBv2DeferredHostFill.resolveBeforeEvaluation()
        eval(output)
        if shouldClearCache(keyTokens: keyTokens, environment: environment) {
            Memory.clearCache()
        }
    }
}

public enum Qwen4ExpNativeSparseGQA: Sendable {
    public static let envFlag = "DARKBLOOM_QWEN4_NATIVE_QSA"
    /// Fusion uses 256 to bound the Python score tile. Listing chunk is 8192;
    /// 2048 is 4 native indexer/GQA launches per QSA layer instead of 32.
    /// 4096 was measured 128K parity with 2048 — do not default 8192
    /// (score tile ~1 GB × 12 QSA layers). Restore Fusion:
    /// `DARKBLOOM_QWEN4_QSA_QUERY_TILE=256`.
    public static let queryTileEnvFlag = "DARKBLOOM_QWEN4_QSA_QUERY_TILE"
    public static let queryHeads = 24
    public static let kvHeads = 2
    public static let headDim = 256
    public static let blockBudget = 512
    public static let compressRatio = 4
    public static let fusionQueryTile = 256
    public static let queryTile = 2048

    public static func resolvedQueryTile(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        let raw = environment[queryTileEnvFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let raw, let parsed = Int(raw), parsed > 0 {
            return parsed
        }
        return queryTile
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var disabled = false
    nonisolated(unsafe) private static var provenStream = false

    public static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    public static func matchesGeometry(
        queryHeads: Int, kvHeads: Int, headDim: Int, selectedWidth: Int
    ) -> Bool {
        queryHeads == Self.queryHeads
            && kvHeads == Self.kvHeads
            && headDim == Self.headDim
            && selectedWidth == blockBudget
    }

    /// Fusion `qsa_fast._native_sparse_gqa_attention` accepts only fp16/bf16.
    public static func matchesDtype(_ dtype: DType) -> Bool {
        dtype == .float16 || dtype == .bfloat16
    }

    /// Quantized Swift linears often emit float32. Store QSA KV in Fusion's
    /// native dtype so the uint4-half kernel actually runs.
    public static func activationType(_ dtype: DType) -> DType {
        matchesDtype(dtype) ? dtype : .bfloat16
    }

    static var isDisabled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return disabled
    }

    static func disable() {
        lock.lock()
        disabled = true
        lock.unlock()
    }

    static func markProven() {
        lock.lock()
        provenStream = true
        lock.unlock()
    }

    static var isProven: Bool {
        lock.lock()
        defer { lock.unlock() }
        return provenStream
    }

    static func canAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, selectedWidth: Int
    ) -> Bool {
        guard isEnabled(), !isDisabled else { return false }
        guard matchesGeometry(
            queryHeads: queries.dim(1), kvHeads: keys.dim(1),
            headDim: queries.dim(3), selectedWidth: selectedWidth)
        else { return false }
        guard queries.ndim == 4, queries.dim(0) == 1, keys.ndim == 4,
            values.shape == keys.shape, keys.dim(0) == 1, keys.dim(3) == headDim,
            queries.dtype == keys.dtype, queries.dtype == values.dtype,
            matchesDtype(queries.dtype)
        else { return false }
        return true
    }

    /// Fusion `_native_sparse_gqa_attention`: Q `[1,24,qL,256]`, KV `[1,2,kL,256]`,
    /// selected `[1,qL,512]` → output `[1,qL,24,256]`.
    static func attend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        selectedBlocks: MLXArray,
        qOffset: Int,
        outputPartitions: Int = 1,
        compactLogicalKeyTokens: Int? = nil,
        preserveKVStrides: Bool? = nil,
        parallelScores: Bool? = nil,
        parallelValuePartitions: Int? = nil,
        parallelFullKV: Bool? = nil
    ) -> MLXArray? {
        let qL = queries.dim(2)
        let kL = compactLogicalKeyTokens ?? keys.dim(2)
        if compactLogicalKeyTokens != nil {
            guard steelEnabled(),
                  (1...Qwen4ExpGatheredQSA.optimizedVerifyMaxQueryTokens).contains(qL),
                  keys.dim(2) == qL * Qwen4ExpCompactQSA.slotsPerQuery else { return nil }
        }
        // Explicit decode/verify experiments default to one partition.
        // Each partition repeats identical QK/softmax and owns disjoint output
        // dimensions, with the same ordered PV accumulation. No split-K sum.
        guard [1, 2, 4].contains(outputPartitions),
            outputPartitions == 1
                || ((1...Qwen4ExpGatheredQSA.optimizedVerifyMaxQueryTokens).contains(qL)
                    && steelEnabled())
        else { return nil }
        guard canAttend(
            queries: queries, keys: keys, values: values, selectedWidth: blockBudget)
        else { return nil }
        guard qOffset >= 0, qOffset + qL <= kL, qL > 0 else { return nil }
        // Fusion `mx.contiguous(native_blocks[:, None])` after the uint32
        // cast so Steel/c8s see a packed [1,1,qL,512] selector, not a
        // strided sort output.
        let blocks: MLXArray
        if selectedBlocks.ndim == 3, selectedBlocks.shape == [1, qL, blockBudget] {
            blocks = selectedBlocks.asType(.uint32).reshaped([1, 1, qL, blockBudget])
                .contiguous()
        } else if selectedBlocks.ndim == 4,
            selectedBlocks.shape == [1, 1, qL, blockBudget]
        {
            blocks = selectedBlocks.asType(.uint32).contiguous()
        } else {
            return nil
        }

        let offset = MLXArray(compactLogicalKeyTokens == nil
            ? [Int32(clamping: qOffset)] : [Int32(clamping: qOffset), Int32(clamping: kL)])
        let stridedKV = (preserveKVStrides ?? Qwen4ExpStridedQSA.enabled())
            && Qwen4ExpStridedQSA.eligible(
                queries: queries, keys: keys, values: values, compactLogicalKeyTokens: compactLogicalKeyTokens)
        // Only these small arrays are packed explicitly. KV pointer/stride
        // metadata comes from the evaluated views, not speculative host flags.
        let kernelQueries = stridedKV ? queries.contiguous() : queries
        let inputs: [MLXArray] = [kernelQueries, keys, values, blocks, offset]
        let tiles = resolvedSteelTiles()
        let template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype), ("BK", tiles.keyTile), ("DC", tiles.dimensionTile),
            ("OPARTS", outputPartitions),
            ("COMPACT_KV", compactLogicalKeyTokens != nil),
            ("STRIDED_KV", stridedKV),
        ]
        let outputShapes = [[1, queryHeads, qL, headDim]]
        let outputDTypes = [queries.dtype]
        let output: MLXArray
        let parallelLayoutEnabled = compactLogicalKeyTokens != nil
            ? (parallelScores ?? Qwen4ExpParallelQSA.enabled())
            : (parallelFullKV ?? Qwen4ExpParallelQSA.fullKVEnabled())
        if parallelLayoutEnabled, steelEnabled(), !stridedKV,
            (1...Qwen4ExpGatheredQSA.optimizedVerifyMaxQueryTokens).contains(qL),
            tiles.keyTile == 64, tiles.dimensionTile == 64
        {
            output = Qwen4ExpParallelQSA.attend(
                inputs: inputs, queryTokens: qL, outputPartitions: outputPartitions,
                valuePartitions: parallelValuePartitions,
                compactKV: compactLogicalKeyTokens != nil,
                outputShape: outputShapes[0], dtype: queries.dtype)
        } else if steelEnabled() {
            let kernel = stridedKV ? qwen4SparseGQASteelStridedKernel : qwen4SparseGQASteelKernel
            output = kernel(
                inputs, template: template,
                grid: (qL * steelSimdWidth, kvHeads * steelWarps, outputPartitions),
                threadGroup: (steelSimdWidth, steelWarps, 1),
                outputShapes: outputShapes, outputDTypes: outputDTypes)[0]
        } else {
            output = qwen4SparseGQAKernel(
                inputs, template: [("T", queries.dtype)],
                grid: (qL * 32, kvHeads, 1),
                threadGroup: (32, 1, 1),
                outputShapes: outputShapes, outputDTypes: outputDTypes)[0]
        }
        let needsStridedProof = stridedKV && Qwen4ExpStridedQSA.needsProof
        if !isProven || needsStridedProof {
            CBv2DeferredHostFill.resolveBeforeEvaluation()
            eval(output)
            markProven()
            if needsStridedProof { Qwen4ExpStridedQSA.markProven() }
        }
        if stridedKV { Qwen4ExpStridedQSA.recordCall() }
        if (2...Qwen4ExpGatheredQSA.optimizedVerifyMaxQueryTokens).contains(qL),
            outputPartitions > 1
        {
            Qwen4ExpVerifyPartitionInvocation.record(outputPartitions)
        }
        return output.transposed(0, 2, 1, 3)
    }
}

/// Numeric evidence for opt-in rectangular dispatches, not GPU completion or
/// an assertion that they improved throughput. The default path does not lock.
public enum Qwen4ExpVerifyPartitionInvocation {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var two = 0
    nonisolated(unsafe) private static var four = 0

    static func record(_ count: Int) {
        lock.withLock {
            if count == 2 { two += 1 }
            if count == 4 { four += 1 }
        }
    }

    public static func snapshot() -> (two: Int, four: Int) { lock.withLock { (two, four) } }
}

/// One simdgroup owns one (query row, KV head) and updates all 12 GQA
/// query heads from a single K/V load. Each lane owns 8 *contiguous*
/// head-dim elements (`d0 = lane * 8`) so fp16/bf16 can `uint4`-load
/// 16 bytes. The previous stride-32 ownership issued 8 scalar gathers
/// per token. Fusion's Steel MMA kernel still cannot drop into
/// `MLXFast.metalKernel` (mlx steel headers).
private let qwen4SparseGQAHeader = """
    template <typename T>
    inline void darkbloom_qwen4_qsa_load8(
        const device T* ptr,
        ulong stride,
        thread float out[8]
    ) {
        for (int i = 0; i < 8; ++i) {
            out[i] = float(ptr[ulong(i) * stride]);
        }
    }

    template <>
    inline void darkbloom_qwen4_qsa_load8<half>(
        const device half* ptr,
        ulong stride,
        thread float out[8]
    ) {
        if (stride == 1) {
            uint4 packed = *((const device uint4*)(ptr));
            ushort2 a = as_type<ushort2>(packed.x);
            ushort2 b = as_type<ushort2>(packed.y);
            ushort2 c = as_type<ushort2>(packed.z);
            ushort2 d = as_type<ushort2>(packed.w);
            out[0] = float(as_type<half>(a.x));
            out[1] = float(as_type<half>(a.y));
            out[2] = float(as_type<half>(b.x));
            out[3] = float(as_type<half>(b.y));
            out[4] = float(as_type<half>(c.x));
            out[5] = float(as_type<half>(c.y));
            out[6] = float(as_type<half>(d.x));
            out[7] = float(as_type<half>(d.y));
        } else {
            for (int i = 0; i < 8; ++i) {
                out[i] = float(ptr[ulong(i) * stride]);
            }
        }
    }

    template <>
    inline void darkbloom_qwen4_qsa_load8<bfloat16_t>(
        const device bfloat16_t* ptr,
        ulong stride,
        thread float out[8]
    ) {
        if (stride == 1) {
            uint4 packed = *((const device uint4*)(ptr));
            ushort2 a = as_type<ushort2>(packed.x);
            ushort2 b = as_type<ushort2>(packed.y);
            ushort2 c = as_type<ushort2>(packed.z);
            ushort2 d = as_type<ushort2>(packed.w);
            out[0] = float(as_type<bfloat16_t>(a.x));
            out[1] = float(as_type<bfloat16_t>(a.y));
            out[2] = float(as_type<bfloat16_t>(b.x));
            out[3] = float(as_type<bfloat16_t>(b.y));
            out[4] = float(as_type<bfloat16_t>(c.x));
            out[5] = float(as_type<bfloat16_t>(c.y));
            out[6] = float(as_type<bfloat16_t>(d.x));
            out[7] = float(as_type<bfloat16_t>(d.y));
        } else {
            for (int i = 0; i < 8; ++i) {
                out[i] = float(ptr[ulong(i) * stride]);
            }
        }
    }

    template <>
    inline void darkbloom_qwen4_qsa_load8<float>(
        const device float* ptr,
        ulong stride,
        thread float out[8]
    ) {
        if (stride == 1) {
            float4 a = *((const device float4*)(ptr));
            float4 b = *((const device float4*)(ptr + 4));
            out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
            out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
        } else {
            for (int i = 0; i < 8; ++i) {
                out[i] = float(ptr[ulong(i) * stride]);
            }
        }
    }

    template <typename T>
    inline void darkbloom_qwen4_qsa_store8(
        device T* ptr,
        thread const float acc[8],
        float inv
    ) {
        for (int i = 0; i < 8; ++i) {
            ptr[i] = T(acc[i] * inv);
        }
    }

    template <>
    inline void darkbloom_qwen4_qsa_store8<half>(
        device half* ptr,
        thread const float acc[8],
        float inv
    ) {
        ushort2 a = ushort2(
            as_type<ushort>(half(acc[0] * inv)),
            as_type<ushort>(half(acc[1] * inv)));
        ushort2 b = ushort2(
            as_type<ushort>(half(acc[2] * inv)),
            as_type<ushort>(half(acc[3] * inv)));
        ushort2 c = ushort2(
            as_type<ushort>(half(acc[4] * inv)),
            as_type<ushort>(half(acc[5] * inv)));
        ushort2 d = ushort2(
            as_type<ushort>(half(acc[6] * inv)),
            as_type<ushort>(half(acc[7] * inv)));
        uint4 packed;
        packed.x = as_type<uint>(a);
        packed.y = as_type<uint>(b);
        packed.z = as_type<uint>(c);
        packed.w = as_type<uint>(d);
        *((device uint4*)(ptr)) = packed;
    }

    template <>
    inline void darkbloom_qwen4_qsa_store8<bfloat16_t>(
        device bfloat16_t* ptr,
        thread const float acc[8],
        float inv
    ) {
        ushort2 a = ushort2(
            as_type<ushort>(bfloat16_t(acc[0] * inv)),
            as_type<ushort>(bfloat16_t(acc[1] * inv)));
        ushort2 b = ushort2(
            as_type<ushort>(bfloat16_t(acc[2] * inv)),
            as_type<ushort>(bfloat16_t(acc[3] * inv)));
        ushort2 c = ushort2(
            as_type<ushort>(bfloat16_t(acc[4] * inv)),
            as_type<ushort>(bfloat16_t(acc[5] * inv)));
        ushort2 d = ushort2(
            as_type<ushort>(bfloat16_t(acc[6] * inv)),
            as_type<ushort>(bfloat16_t(acc[7] * inv)));
        uint4 packed;
        packed.x = as_type<uint>(a);
        packed.y = as_type<uint>(b);
        packed.z = as_type<uint>(c);
        packed.w = as_type<uint>(d);
        *((device uint4*)(ptr)) = packed;
    }

    template <>
    inline void darkbloom_qwen4_qsa_store8<float>(
        device float* ptr,
        thread const float acc[8],
        float inv
    ) {
        *((device float4*)(ptr)) = float4(
            acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv);
        *((device float4*)(ptr + 4)) = float4(
            acc[4] * inv, acc[5] * inv, acc[6] * inv, acc[7] * inv);
    }

    inline void darkbloom_qwen4_qsa_accumulate(
        thread float acc[12][8],
        thread float m[12],
        thread float l[12],
        thread const float q_reg[12][8],
        thread const float k_reg[8],
        thread const float v_reg[8],
        float scale
    ) {
        for (uint h = 0; h < 12u; ++h) {
            float local = 0.0f;
            for (int i = 0; i < 8; ++i) {
                local += q_reg[h][i] * k_reg[i];
            }
            float score = simd_sum(local) * scale;
            float new_m = m[h] == -INFINITY ? score : max(m[h], score);
            float alpha = m[h] == -INFINITY ? 0.0f : exp(m[h] - new_m);
            float w = exp(score - new_m);
            l[h] = l[h] * alpha + w;
            for (int i = 0; i < 8; ++i) {
                acc[h][i] = acc[h][i] * alpha + w * v_reg[i];
            }
            m[h] = new_m;
        }
    }
"""

private let qwen4SparseGQAKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen4_qsa_sparse_gqa_c8s",
    inputNames: ["queries", "keys", "values", "selected", "q_offset"],
    outputNames: ["output"],
    source: """
        uint qi = threadgroup_position_in_grid.x;
        uint kv_head = threadgroup_position_in_grid.y;
        uint lane = thread_index_in_threadgroup;
        uint qL = uint(queries_shape[2]);
        uint kL = uint(keys_shape[2]);
        if (qi >= qL || kv_head >= 2u || lane >= 32u) {
            return;
        }

        uint head0 = kv_head * 12u;
        uint d0 = lane * 8u;
        int q_abs = int(q_offset[0]) + int(qi);
        int complete_blocks = (q_abs + 1) / 4;
        int valid_blocks = complete_blocks < 512 ? complete_blocks : 512;
        float scale = 1.0f / 16.0f;

        float q_reg[12][8];
        float acc[12][8];
        float m[12];
        float l[12];
        for (uint h = 0; h < 12u; ++h) {
            ulong q_base = ulong(head0 + h) * ulong(queries_strides[1])
                + ulong(qi) * ulong(queries_strides[2])
                + ulong(d0) * ulong(queries_strides[3]);
            m[h] = -INFINITY;
            l[h] = 0.0f;
            darkbloom_qwen4_qsa_load8(
                queries + q_base, ulong(queries_strides[3]), q_reg[h]);
            for (int i = 0; i < 8; ++i) {
                acc[h][i] = 0.0f;
            }
        }

        for (int block_slot = 0; block_slot < 512; ++block_slot) {
            int pos0 = -1;
            if (block_slot < valid_blocks) {
                ulong sel = ulong(qi) * ulong(selected_strides[2])
                    + ulong(block_slot) * ulong(selected_strides[3]);
                pos0 = int(selected[sel]) * 4;
            }
            for (int t = 0; t < 4; ++t) {
                int pos = pos0 + t;
                if (pos0 < 0 || pos >= int(kL) || pos > q_abs) {
                    continue;
                }
                ulong k_base = ulong(kv_head) * ulong(keys_strides[1])
                    + ulong(pos) * ulong(keys_strides[2])
                    + ulong(d0) * ulong(keys_strides[3]);
                ulong v_base = ulong(kv_head) * ulong(values_strides[1])
                    + ulong(pos) * ulong(values_strides[2])
                    + ulong(d0) * ulong(values_strides[3]);
                float k_reg[8];
                float v_reg[8];
                darkbloom_qwen4_qsa_load8(
                    keys + k_base, ulong(keys_strides[3]), k_reg);
                darkbloom_qwen4_qsa_load8(
                    values + v_base, ulong(values_strides[3]), v_reg);
                darkbloom_qwen4_qsa_accumulate(acc, m, l, q_reg, k_reg, v_reg, scale);
            }
        }

        for (int t = 0; t < 3; ++t) {
            int pos = complete_blocks * 4 + t;
            if (pos >= int(kL) || pos > q_abs) {
                continue;
            }
            ulong k_base = ulong(kv_head) * ulong(keys_strides[1])
                + ulong(pos) * ulong(keys_strides[2])
                + ulong(d0) * ulong(keys_strides[3]);
            ulong v_base = ulong(kv_head) * ulong(values_strides[1])
                + ulong(pos) * ulong(values_strides[2])
                + ulong(d0) * ulong(values_strides[3]);
            float k_reg[8];
            float v_reg[8];
            darkbloom_qwen4_qsa_load8(
                keys + k_base, ulong(keys_strides[3]), k_reg);
            darkbloom_qwen4_qsa_load8(
                values + v_base, ulong(values_strides[3]), v_reg);
            darkbloom_qwen4_qsa_accumulate(acc, m, l, q_reg, k_reg, v_reg, scale);
        }

        for (uint h = 0; h < 12u; ++h) {
            ulong o_base = ((ulong(head0 + h) * ulong(qL)) + ulong(qi)) * ulong(256)
                + ulong(d0);
            float inv = l[h] > 0.0f ? 1.0f / l[h] : 0.0f;
            darkbloom_qwen4_qsa_store8(output + o_base, acc[h], inv);
        }
    """,
    header: qwen4SparseGQAHeader,
    ensureRowContiguous: true
)
