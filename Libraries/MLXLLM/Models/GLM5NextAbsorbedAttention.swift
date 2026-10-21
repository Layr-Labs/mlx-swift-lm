// Copyright © 2026 Eigen Labs.
// Absorbed NoPE MLA for the serving GLM5Next DSA layer. Equations ported from
// huggingface/transformers@f0705f991a71f05dad9569336b181119f5546f93
//   src/transformers/models/glm5_next/modeling_glm5_next.py (Glm5NextTextAttention)
// Absorbed MLA / selected-latent execution reference:
// Blaizzy/mlx-vlm@4623cf9d2633c5ee2b1ac125eeae51423447c41f
//   mlx_vlm/models/glm5_next/language.py

import Foundation
import MLX
import MLXNN

extension GLM5NextDSAAttention {
    /// Default bound on one batched block's FP32 selected latents. Bounds the
    /// contiguous prefill transient; decode is always one query.
    static let absorbedLatentBlockBytes = 64 << 20

    /// How the absorbed path reads the non-padding selected latents.
    enum AbsorbedLatentSource {
        /// One read of `[B,Tq,L]` ids into `[B,Tq,L,C]` latents per block of at
        /// most `absorbedBlockBytes` of FP32 latents.
        case batched((MLXArray) throws -> MLXArray)
        /// One read of `[B,L]` ids into `[B,L,C]` latents per query, as the
        /// source's owner prices it. Each read is its own block and is consumed
        /// in place, so no unpriced copy stacks several queries' reads.
        case perQuery((MLXArray) throws -> MLXArray)
    }

    func absorbedAttention(
        q: MLXArray, selection: GLM5NextSparseAttention.Selection,
        projections: GLM5NextAbsorbedProjections, source: AbsorbedLatentSource
    ) throws -> MLXArray {
        let c = configuration
        let (batch, count) = (q.dim(0), q.dim(1))
        let padding = selection.staticPaddingColumns
        var ids = selection.indices
        if !padding.isEmpty {
            ids = concatenated(
                [ids[.ellipsis, ..<padding.lowerBound], ids[.ellipsis, padding.upperBound...]],
                axis: -1)
        }
        let live = ids.dim(-1)
        // A faulted MLX array reads as zero-sized; refuse before dividing.
        guard batch > 0, count > 0, live > 0 else {
            throw GLM5NextLoadError.invalid("absorbed DSA selection geometry")
        }
        let scale = 1 / sqrt(Float(c.qkNopeHeadDim))
        // [B,H,T,C], one batched kernel for every head and query of the chunk.
        let queryLatents = projections.absorbQueries(q)
        let blockQueries: Int
        let read: (MLXArray) throws -> MLXArray
        switch source {
        case .batched(let gather):
            blockQueries = max(1, absorbedBlockBytes / (batch * live * c.kvLoraRank * 4))
            read = gather
        case .perQuery(let gather):
            blockQueries = 1
            read = { try gather($0[0..., 0]).expandedDimensions(axis: 1) }
        }
        var latentBlocks: [MLXArray] = []
        for start in stride(from: 0, to: count, by: blockQueries) {
            let queries = start ..< min(start + blockQueries, count)
            let blockIds = ids[0..., queries]
            let selected = try read(blockIds).asType(.float32)
            // [B,Tq,H,C] x [B,Tq,C,L] -> [B,Tq,H,L]
            let score =
                matmul(
                    queryLatents[0..., 0..., queries].transposed(0, 2, 1, 3).asType(.float32),
                    selected.transposed(0, 1, 3, 2)) * scale
            let mask = (blockIds .>= 0).expandedDimensions(axis: 2)
            let masked = MLX.where(mask, score, -Float.infinity)
            let maximum = MLX.where(
                mask.any(axis: -1, keepDims: true), masked.max(axis: -1, keepDims: true), 0)
            let exps = MLX.where(mask, exp(masked - maximum), 0)
            let total = exps.sum(axis: -1, keepDims: true)
            // Same attention-weight rounding as the expanded reference.
            let probability = (exps / MLX.where(total .> 0, total, 1)).asType(q.dtype)
            // [B,Tq,H,L] x [B,Tq,L,C] -> [B,Tq,H,C], already in the kernel dtype.
            latentBlocks.append(
                matmul(probability.asType(.float32), selected).asType(projections.inputDType))
        }
        let latents =
            latentBlocks.count == 1 ? latentBlocks[0] : concatenated(latentBlocks, axis: 1)
        // [B,H,T,C] -> [B,H,T,Dv]
        let values = projections.projectValues(latents.transposed(0, 2, 1, 3))
        return try executionPolicy.prefillProjections.project(
            output,
            values.transposed(0, 2, 1, 3)
                .reshaped(batch, count, c.attentionHeads * c.vHeadDim)
                .asType(q.dtype))
    }
}

extension GLM5NextDSAAttention {
    /// `SparseDecode.nativeKernels` (at most `GLM5NextSparseDecodeKernel
    /// .maximumQueries` queries, split-K) or `SparsePrefill.nativeKernel`
    /// (longer steps) attention over a contiguous `[B,N,C]` latent history:
    /// the kernel reads each selected latent in place through
    /// `selection.indices`, so no gathered latent copy exists.
    func nativeSparseAttention(
        q: MLXArray, selection: GLM5NextSparseAttention.Selection, history: MLXArray,
        projections: GLM5NextAbsorbedProjections
    ) throws -> MLXArray {
        let c = configuration
        let (batch, count) = (q.dim(0), q.dim(1))
        guard batch > 0, count > 0, history.shape == [batch, selection.historyLength, c.kvLoraRank]
        else { throw GLM5NextLoadError.invalid("sparse DSA history geometry") }
        let queryLatents = projections.absorbQueries(q)
        let scale = 1 / sqrt(Float(c.qkNopeHeadDim))
        let latents: MLXArray
        if count <= GLM5NextSparseDecodeKernel.maximumQueries {
            latents = try GLM5NextSparseDecodeKernel.attend(
                queryLatents: queryLatents, latents: history, indices: selection.indices,
                scale: scale, outputType: projections.inputDType, layout: sparseDecodeLayout)
        } else {
            // FP32 CBv2 storage widened the activation dtype losslessly; the
            // tensor-unit kernel reads 16-bit latents.
            latents = try GLM5NextSparsePrefillKernel.attend(
                queryLatents: queryLatents,
                latents: history.dtype == .float32 && q.dtype != .float32
                    ? history.asType(q.dtype) : history,
                indices: selection.indices, scale: scale, outputType: projections.inputDType)
        }
        let values = projections.projectValues(latents)
        return try executionPolicy.prefillProjections.project(
            output,
            values.transposed(0, 2, 1, 3)
                .reshaped(batch, count, c.attentionHeads * c.vHeadDim)
                .asType(q.dtype))
    }

    /// Absorbed MLA over the whole contiguous latent history `[B,N,C]` while
    /// the selection budget covers every pool (`N <= index_topk`), so each
    /// query's selected set is a function of key validity alone.
    /// `GLM5NextDenseSelectionKernel` derives that set in-kernel and applies the
    /// masked softmax, so no selection indices or gathered latents exist.
    func denseHistoryAttention(
        q: MLXArray, history: MLXArray, keyValidity: MLXArray, queryValidity: MLXArray,
        queryStart: Int, projections: GLM5NextAbsorbedProjections
    ) throws -> MLXArray {
        let c = configuration
        let (batch, count) = (q.dim(0), q.dim(1))
        let length = history.dim(1)
        guard batch > 0, count > 0, length > 0, history.shape == [batch, length, c.kvLoraRank],
            keyValidity.shape == [batch, length], queryValidity.shape == [batch, count],
            length <= c.indexTopK, queryStart + count <= length
        else { throw GLM5NextLoadError.invalid("dense DSA history geometry") }
        let scale = 1 / sqrt(Float(c.qkNopeHeadDim))
        let heads = c.attentionHeads
        // [B,H,T,C] queries. Every head shares the [B,N,C] latents, so heads
        // and queries form one GEMM row axis and each latent is read once.
        let queryLatents = projections.absorbQueries(q)
        let keys = history.asType(.float32)
        let keysTransposed = keys.transposed(0, 2, 1)
        let blockQueries = max(1, absorbedBlockBytes / max(1, batch * heads * length * 4))
        var latentBlocks: [MLXArray] = []
        for start in stride(from: 0, to: count, by: blockQueries) {
            let queries = start ..< min(start + blockQueries, count)
            // Causal: a block's queries can select no key past its last position.
            let visible = min(length, queryStart + queries.upperBound)
            let rows = queryLatents[0..., 0..., queries].asType(.float32)
                .reshaped(batch, heads * queries.count, c.kvLoraRank)
            let score = matmul(rows, keysTransposed[0..., 0..., ..<visible])
                .reshaped(batch, heads, queries.count, visible)
            let probability = try GLM5NextDenseSelectionKernel.maskedSoftmax(
                score: score, keyValidity: keyValidity[0..., ..<visible],
                queryValidity: queryValidity[0..., queries], queryStart: queryStart + start,
                scale: scale, roundingType: q.dtype)
            // [B,H*Tb,N'] x [B,N',C] -> [B,H,Tb,C], already in the kernel dtype.
            latentBlocks.append(
                matmul(
                    probability.reshaped(batch, heads * queries.count, visible),
                    keys[0..., ..<visible]
                )
                .reshaped(batch, heads, queries.count, c.kvLoraRank)
                .asType(projections.inputDType))
        }
        let latents =
            latentBlocks.count == 1 ? latentBlocks[0] : concatenated(latentBlocks, axis: 2)
        let values = projections.projectValues(latents)
        return try executionPolicy.prefillProjections.project(
            output,
            values.transposed(0, 2, 1, 3)
                .reshaped(batch, count, c.attentionHeads * c.vHeadDim)
                .asType(q.dtype))
    }
}

/// The every-pool selection of `GLM5NextSparseAttention.selectTokens` as a
/// membership test, fused with the absorbed attention's masked softmax.
///
/// For query position `pos` in row b (first valid key f, v = valid keys at or
/// before pos): key n is selected iff the query is valid and either n lies in
/// the complete pool starting at f + 4*floor((n-f)/4) whose last key is at or
/// before pos, or n is a valid key at or before pos in the tail
/// [f + v - v%4, f + v). That is exactly the selector's set whenever the
/// budget covers every pool. The softmax keeps the reference's FP32 scale,
/// max, exponential and division, and rounds each probability to the
/// activation dtype; its sums use this kernel's order, so not bitwise.
enum GLM5NextDenseSelectionKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry
        case unsupportedDeviceOrStream
    }

    static let threads = 256

    /// `score` FP32 `[B,H,T,N]` (unscaled); returns FP32 rounded probabilities.
    /// Off the default GPU stream the same membership and softmax run as MLX
    /// operations (`maskedSoftmaxOperations`).
    static func maskedSoftmax(
        score: MLXArray, keyValidity: MLXArray, queryValidity: MLXArray, queryStart: Int,
        scale: Float, roundingType: DType
    ) throws -> MLXArray {
        guard score.ndim == 4, score.dtype == .float32, score.shape.allSatisfy({ $0 > 0 }),
            keyValidity.shape == [score.dim(0), score.dim(3)], keyValidity.dtype == .bool,
            queryValidity.shape == [score.dim(0), score.dim(2)], queryValidity.dtype == .bool,
            queryStart >= 0, queryStart + score.dim(2) <= score.dim(3),
            score.size <= Int(Int32.max), [.float16, .bfloat16, .float32].contains(roundingType)
        else { throw Failure.invalidGeometry }
        guard glm5NextOnDefaultGPUStream() else {
            return maskedSoftmaxOperations(
                score: score, keyValidity: keyValidity, queryValidity: queryValidity,
                queryStart: queryStart, scale: scale, roundingType: roundingType)
        }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let rows = score.dim(0) * score.dim(1) * score.dim(2)
            return kernel(
                [
                    score, keyValidity, queryValidity,
                    GLM5NextKernelConstants.shared.int32(Int32(queryStart)),
                    GLM5NextKernelConstants.shared.float(scale),
                ],
                template: [("T", roundingType), ("Threads", threads)],
                grid: (threads, rows, 1), threadGroup: (threads, 1, 1),
                outputShapes: [score.shape], outputDTypes: [.float32])[0]
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    /// `[B,T,N]` membership of the every-pool selection, as MLX operations.
    static func membership(keyValidity: MLXArray, queryValidity: MLXArray, queryStart: Int)
        -> MLXArray
    {
        let batch = keyValidity.dim(0)
        let length = keyValidity.dim(1)
        let queries = queryValidity.dim(1)
        let valid = keyValidity.asType(.int32)
        let first = MLX.where(
            keyValidity.any(axis: 1), valid.argMax(axis: 1).asType(.int32), Int32(length)
        ).reshaped(batch, 1)
        let keys = MLXArray.arange(length, dtype: .int32).reshaped(1, length)
        let relative = keys - first
        let poolStart = first + maximum(relative, 0).floorDivide(4) * 4
        var complete = (relative .>= 0) .&& ((poolStart + 3) .< Int32(length))
        for offset in 0 ..< 4 {
            complete =
                complete
                .&& takeAlong(
                    keyValidity, clip(poolStart + Int32(offset), min: 0, max: length - 1), axis: 1)
        }
        let positions = (MLXArray.arange(queries, dtype: .int32) + Int32(queryStart))
            .reshaped(1, queries, 1)
        let visible = take(cumsum(valid, axis: 1), positions.reshaped(queries), axis: 1)
            .reshaped(batch, queries, 1)
        let tailStart = first.expandedDimensions(axis: 1) + visible - visible % 4
        let tailEnd = first.expandedDimensions(axis: 1) + visible
        let keyAxis = keys.expandedDimensions(axis: 1)
        let pooled =
            complete.expandedDimensions(axis: 1)
            .&& ((poolStart + 3).expandedDimensions(axis: 1) .<= positions)
        let tail =
            (keyAxis .>= tailStart) .&& (keyAxis .< tailEnd) .&& (keyAxis .<= positions)
            .&& keyValidity.expandedDimensions(axis: 1)
        return queryValidity.expandedDimensions(axis: -1) .&& (pooled .|| tail)
    }

    /// The reference masked softmax over `membership`, as MLX operations.
    static func maskedSoftmaxOperations(
        score: MLXArray, keyValidity: MLXArray, queryValidity: MLXArray, queryStart: Int,
        scale: Float, roundingType: DType
    ) -> MLXArray {
        let mask = membership(
            keyValidity: keyValidity, queryValidity: queryValidity, queryStart: queryStart
        ).expandedDimensions(axis: 1)
        let masked = MLX.where(mask, score * scale, -Float.infinity)
        let maximum = MLX.where(
            mask.any(axis: -1, keepDims: true), masked.max(axis: -1, keepDims: true), 0)
        let exps = MLX.where(mask, exp(masked - maximum), 0)
        let total = exps.sum(axis: -1, keepDims: true)
        return (exps / MLX.where(total .> 0, total, 1)).asType(roundingType).asType(.float32)
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let kernel = MLXFast.metalKernel(
            name: "glm5_next_dense_selection_softmax",
            inputNames: ["score", "key_validity", "query_validity", "query_start", "scale"],
            outputNames: ["probability"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr uint Groups = uint(Threads) / 32;
                threadgroup int shared_int[Groups];
                threadgroup float shared_float[Groups];
                const int heads = score_shape[1];
                const int queries = score_shape[2];
                const int length = score_shape[3];
                const uint row = threadgroup_position_in_grid.y;
                const int b = int(row) / (heads * queries);
                const int t = int(row) % queries;
                const int pos = query_start + t;
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint group = simdgroup_index_in_threadgroup;
                const size_t keys = size_t(b) * length;
                const size_t offset = size_t(row) * length;

                int first = length;
                for (int n = int(tid); n < length; n += Threads) {
                    if (key_validity[keys + n]) { first = n; break; }
                }
                first = simd_min(first);
                if (lane == 0) { shared_int[group] = first; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                first = length;
                for (uint i = 0; i < Groups; ++i) { first = metal::min(first, shared_int[i]); }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                int visible = 0;
                for (int n = int(tid); n <= pos; n += Threads) {
                    visible += key_validity[keys + n] ? 1 : 0;
                }
                visible = simd_sum(visible);
                if (lane == 0) { shared_int[group] = visible; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                visible = 0;
                for (uint i = 0; i < Groups; ++i) { visible += shared_int[i]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const int tail_start = first + visible - visible % 4;
                const int tail_end = first + visible;
                const bool query_valid = query_validity[size_t(b) * queries + t];
                #define GLM5_MEMBER(n) (query_valid && ( \
                    ((n) >= first && (first + 4 * (((n) - first) / 4) + 3) <= pos \
                        && key_validity[keys + first + 4 * (((n) - first) / 4)] \
                        && key_validity[keys + first + 4 * (((n) - first) / 4) + 1] \
                        && key_validity[keys + first + 4 * (((n) - first) / 4) + 2] \
                        && key_validity[keys + first + 4 * (((n) - first) / 4) + 3]) \
                    || ((n) >= tail_start && (n) < tail_end && (n) <= pos \
                        && key_validity[keys + (n)])))

                float maximum = -INFINITY;
                int any = 0;
                for (int n = int(tid); n < length; n += Threads) {
                    if (GLM5_MEMBER(n)) {
                        const float value = score[offset + n] * scale;
                        maximum = metal::max(maximum, value);
                        any = 1;
                    }
                }
                maximum = simd_max(maximum);
                any = simd_max(any);
                if (lane == 0) { shared_float[group] = maximum; shared_int[group] = any; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                maximum = -INFINITY;
                any = 0;
                for (uint i = 0; i < Groups; ++i) {
                    maximum = metal::max(maximum, shared_float[i]);
                    any = metal::max(any, shared_int[i]);
                }
                if (any == 0) { maximum = 0.0f; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float total = 0.0f;
                for (int n = int(tid); n < length; n += Threads) {
                    if (GLM5_MEMBER(n)) {
                        const float value = score[offset + n] * scale;
                        total += metal::precise::exp(value - maximum);
                    }
                }
                total = simd_sum(total);
                if (lane == 0) { shared_float[group] = total; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                total = 0.0f;
                for (uint i = 0; i < Groups; ++i) { total += shared_float[i]; }
                const float denominator = total > 0.0f ? total : 1.0f;
                for (int n = int(tid); n < length; n += Threads) {
                    float result = 0.0f;
                    if (GLM5_MEMBER(n)) {
                        const float value = score[offset + n] * scale;
                        const float e = metal::precise::exp(value - maximum);
                        result = static_cast<float>(static_cast<T>(e / denominator));
                    }
                    probability[offset + n] = result;
                }
                #undef GLM5_MEMBER
                """,
            ensureRowContiguous: true)
    #endif
}

/// Per-head key and value views of the checkpoint's kv_b, never copies.
///
/// kv_b is `[H*(Dq+Dv), C]`, quantized along C. Head h's key rows form a
/// `[Dq, C]` matrix, so MLX `quantized_matmul(transpose: false)` contracts the
/// Dq rows and expands the packed C axis: q_h W_k[h]. Its `[Dv, C]` value rows
/// with `transpose: true` give x W_v[h]^T. Each view's inner matrix is
/// row-contiguous; the head axis is strided, which MLX's batched quantized
/// kernels read in place. Q4 arithmetic runs in the scale dtype, as the
/// reference kv_b call does, so no per-call FP32 scale copy is streamed.
///
/// The views are taken once at construction. Replacing kv_b afterwards (by a
/// parameter update, quantization or an adapter) would leave this path on the
/// old buffers and keep them alive; nothing in this package does so.
struct GLM5NextAbsorbedProjections {
    struct Packed {
        let words: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }
    enum Weights {
        case quantized(key: Packed, value: Packed, groupSize: Int, bits: Int)
        /// Unquantized storage: `[H,Dq,C]` and `[H,Dv,C]` views.
        case dense(key: MLXArray, value: MLXArray)
    }

    let weights: Weights

    init(kvB: Linear, configuration c: GLM5NextTextConfiguration) throws {
        let heads = c.attentionHeads
        let keyWidth = c.qkNopeHeadDim
        guard kvB.shape == (heads * (keyWidth + c.vHeadDim), c.kvLoraRank), kvB.bias == nil else {
            throw GLM5NextLoadError.invalid("absorbed DSA requires bias-free kv_b geometry")
        }
        func split(_ array: MLXArray) -> (key: MLXArray, value: MLXArray) {
            let perHead = array.reshaped(heads, keyWidth + c.vHeadDim, -1)
            return (perHead[0..., ..<keyWidth], perHead[0..., keyWidth...])
        }
        if let packed = kvB as? QuantizedLinear {
            guard packed.mode == .affine, let offsets = packed.biases else {
                throw GLM5NextLoadError.invalid("absorbed DSA requires affine kv_b")
            }
            let (words, scales, biases) = (
                split(packed.weight), split(packed.scales), split(offsets)
            )
            weights = .quantized(
                key: .init(words: words.key, scales: scales.key, biases: biases.key),
                value: .init(words: words.value, scales: scales.value, biases: biases.value),
                groupSize: packed.groupSize, bits: packed.bits)
        } else {
            let views = split(kvB.weight)
            weights = .dense(key: views.key, value: views.value)
        }
        // Evaluating reshape/slice views shares kv_b's buffers; nothing is copied.
        let arrays = views
        try MLX.withError { error in
            eval(arrays)
            try error.check()
        }
    }

    var views: [MLXArray] {
        switch weights {
        case .quantized(let key, let value, _, _):
            [key.words, key.scales, key.biases, value.words, value.scales, value.biases]
        case .dense(let key, let value): [key, value]
        }
    }

    /// Input and output dtype of both projections: the Q4 scale dtype, or the
    /// unquantized weight dtype.
    var inputDType: DType {
        switch weights {
        case .quantized(let key, _, _, _): key.scales.dtype
        case .dense(let key, _): key.dtype
        }
    }

    /// `[B,T,H,Dq]` -> `[B,H,T,C]`.
    func absorbQueries(_ q: MLXArray) -> MLXArray {
        let queries = q.transposed(0, 2, 1, 3).asType(inputDType)
        switch weights {
        case .quantized(let key, _, let groupSize, let bits):
            return quantizedMM(
                queries, key.words, scales: key.scales, biases: key.biases, transpose: false,
                groupSize: groupSize, bits: bits)
        case .dense(let key, _):
            return matmul(queries, key)
        }
    }

    /// `[B,H,T,C]` -> `[B,H,T,Dv]`.
    func projectValues(_ latents: MLXArray) -> MLXArray {
        let latents = latents.asType(inputDType)
        switch weights {
        case .quantized(_, let value, let groupSize, let bits):
            return quantizedMM(
                latents, value.words, scales: value.scales, biases: value.biases,
                transpose: true, groupSize: groupSize, bits: bits)
        case .dense(_, let value):
            return matmul(latents, value.transposed(0, 2, 1))
        }
    }
}
