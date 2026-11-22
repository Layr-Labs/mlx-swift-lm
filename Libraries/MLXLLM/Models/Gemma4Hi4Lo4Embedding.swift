// Gemma4Hi4Lo4Embedding.swift
//
// Zero-Extra-RAM Hi4/Lo4 bit-plane decomposition and Cauchy-Schwarz exact
// sparse LM-head sampling for 8-bit group-64 affine `QuantizedEmbedding`.
//
// Splits an 8-bit quantized vocabulary table `W` (`[V, H/4]` uint32, 4 bytes
// per uint32) into two 4-bit MLX-packed bit-planes (`hiWeight` = `q >> 4`,
// `loWeight` = `q & 0xF`, each `[V, H/8]` uint32) plus a 1 MB per-row FP32
// exact residual L2 norm `rowResidualNorm` (`[V]`).
//
// Pass 1 runs MLX's native 4-bit `quantizedMM` on `hiWeight` alone (half the
// weight bandwidth). Pass 2 computes sound Cauchy-Schwarz `[L, U]` bounds that
// absorb both the exact `Lo4` residual and output-dtype / BF16 scale-bias
// rounding, then compacts surviving candidates `U >= tau` on GPU in ascending
// token-ID order. Pass 3 gathers and interleaves `hiWeight` + `loWeight` only
// for the surviving candidates and refines them with MLX's native 8-bit
// `quantizedMM` + softcap. Any row that overflows `K_cap` or straddles a
// `topP`/`minP` partition-function boundary falls back on GPU without host
// syncs, preserving 0-ULP bit-exact parity across `.greedy`, `.gumbel`, and
// `.filtered`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class Gemma4Hi4Lo4Embedding: Embedding, Quantized {

    /// Execution mode for `Gemma4Hi4Lo4Embedding` sparse LM-head sampling:
    /// - `off` (default): leave stock 8-bit `QuantizedEmbedding` untouched.
    /// - `exact`: certified 0-ULP bit-exact Cauchy-Schwarz pruning with GPU dense fallback.
    /// - `refine`: fast 2-pass Hi4 screening + exact 8-bit refinement of top-K (`K = 64`)
    ///   candidates across all GPU cores (100% top-1 and 160/160 top-16 exact match on
    ///   real Gemma 4 26B decode states with zero dense-fallback overhead).
    public enum SamplingStrategy: String, Sendable, Equatable {
        case off
        case exact
        case refine
    }

    /// Environment flag for the Gemma 4 Hi4/Lo4 sparse LM-head path (default `off`;
    /// accepts `off`/`0`, `exact`/`1`/`true`, or `refine`/`fast`).
    public static let envKey = "DARKBLOOM_GEMMA4_HI4_LM_HEAD"

    /// Default per-row candidate buffer capacity for Pass 2 exact Cauchy-Schwarz compaction.
    public static let defaultCandidateCapacity = 2048

    /// Default top-K candidate count refined at exact 8-bit precision in `.refine` mode.
    public static let defaultRefineTopK = 64

    /// Row chunk size used during load-time bit-plane repacking to keep peak
    /// transient RAM small.
    public static let repackChunkRows = 4_096

    /// Row chunk size used during dense `asLinear(_:)` fallback so full 8-bit
    /// weight reconstruction never allocates the whole `[V, H/4]` table at once.
    public static let denseChunkRows = 32_768

    public let vocabSize: Int
    public let hiddenSize: Int
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let samplingStrategy: SamplingStrategy

    /// Upper 4-bit plane (`q >> 4`), stored in `Embedding.weight` (`[V, H/8]` uint32)
    /// so no duplicate weight tensor is retained.
    public var hiWeight: MLXArray { weight }

    /// Lower 4-bit plane (`q & 0xF`), MLX 4-bit little-endian packed (`[V, H/8]` uint32).
    public let loWeight: MLXArray

    /// Original 8-bit group scales (`[V, H/64]`), reused directly from source.
    public let scales: MLXArray

    /// Original 8-bit group biases (`[V, H/64]`), reused directly from source.
    public let biases: MLXArray

    /// Pre-scaled 4-bit upper-plane scales (`16 * scales`, `[V, H/64]`, `scales.dtype`).
    public let hiScales: MLXArray

    /// Pre-shifted 4-bit upper-plane biases (`biases + 7.5 * scales`, `[V, H/64]`, `biases.dtype`).
    public let hiBiases: MLXArray

    /// Exact per-row L2 residual norm `||W_8bit[v] - W_hi4[v]||_2` (`[V]` float32),
    /// absorbing BF16/FP16 `hiScales`/`hiBiases` and Steel QMM dequantization rounding.
    public let rowResidualNorm: MLXArray

    public override var shape: (Int, Int) {
        (vocabSize, hiddenSize)
    }

    /// Reconstruct the exact 8-bit `QuantizedEmbedding` (used when checkpoint
    /// weights are reloaded over a prepared model).
    public func toQuantizedEmbedding() -> QuantizedEmbedding {
        let w8 = Self.interleaveHiLoToW8(hi: hiWeight, lo: loWeight)
        return QuantizedEmbedding(
            weight: w8, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode)
    }

    /// Diagnostic output returned by ``sparseExactHeadSampleWithDiagnostics``.
    public struct SparseHeadSampleDiagnostics {
        /// Sampled token IDs (`[B]` int32).
        public let tokens: MLXArray
        /// Non-finite row flags (`[B]` int32, `1` if row produced NaN/Inf logits).
        public let nonFinite: MLXArray
        /// Number of candidates compacted per row before capping (`[B]` int32).
        public let candidateCounts: MLXArray
        /// Per-row compaction overflow indicator (`[B]` uint8, `1` if `> K_cap`).
        public let overflow: MLXArray
        /// Per-row GPU dense fallback indicator (`[B]` uint8, `1` if fallback ran).
        public let needsFallback: MLXArray
    }

    public static func resolvedStrategy(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> SamplingStrategy {
        guard
            let raw = environment[envKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased(),
            !raw.isEmpty
        else { return .off }
        switch raw {
        case "0", "false", "no", "off":
            return .off
        case "refine", "fast", "top64":
            return .refine
        case "1", "true", "yes", "on", "exact":
            return .exact
        default:
            return .off
        }
    }

    public static func isEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        resolvedStrategy(environment: environment) != .off
    }

    private init(
        hiWeight: MLXArray,
        loWeight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        hiScales: MLXArray,
        hiBiases: MLXArray,
        rowResidualNorm: MLXArray,
        vocabSize: Int,
        hiddenSize: Int,
        samplingStrategy: SamplingStrategy
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.groupSize = 64
        self.bits = 8
        self.mode = .affine
        self.samplingStrategy = samplingStrategy
        self.loWeight = loWeight
        self.scales = scales
        self.biases = biases
        self.hiScales = hiScales
        self.hiBiases = hiBiases
        self.rowResidualNorm = rowResidualNorm
        super.init(weight: hiWeight)
        self.freeze()
    }

    // MARK: - Zero-Extra-RAM Load-Time Repack

    /// Convert an 8-bit group-64 affine `QuantizedEmbedding` into a bit-plane
    /// `Gemma4Hi4Lo4Embedding` in bounded chunks. Returns `nil` for any other
    /// bit width, group size, quantization mode, or incompatible tensor layout.
    public static func from(
        _ source: QuantizedEmbedding,
        samplingStrategy: SamplingStrategy = .exact,
        chunkRows: Int = repackChunkRows
    ) -> Gemma4Hi4Lo4Embedding? {
        guard source.bits == 8,
            source.groupSize == 64,
            source.mode == .affine,
            source.globalScale == nil,
            let sourceBiases = source.biases,
            source.weight.ndim == 2,
            source.weight.dtype == .uint32
        else { return nil }

        let vocabSize = source.weight.dim(0)
        let packedCols8 = source.weight.dim(1)
        let hiddenSize = packedCols8 * 4
        guard vocabSize > 0, packedCols8 > 0, hiddenSize % 64 == 0 else { return nil }

        let groups = hiddenSize / 64
        guard source.scales.shape == [vocabSize, groups],
            sourceBiases.shape == [vocabSize, groups],
            source.scales.dtype == sourceBiases.dtype,
            source.scales.dtype == .bfloat16 || source.scales.dtype == .float16
                || source.scales.dtype == .float32
        else { return nil }

        let step = Swift.max(64, chunkRows)
        let scaleDType = source.scales.dtype
        let numChunks = (vocabSize + step - 1) / step

        var hiChunks: [MLXArray] = []
        var loChunks: [MLXArray] = []
        var hiScaleChunks: [MLXArray] = []
        var hiBiasChunks: [MLXArray] = []
        var rhoChunks: [MLXArray] = []
        hiChunks.reserveCapacity(numChunks)
        loChunks.reserveCapacity(numChunks)
        hiScaleChunks.reserveCapacity(numChunks)
        hiBiasChunks.reserveCapacity(numChunks)
        rhoChunks.reserveCapacity(numChunks)

        for start in stride(from: 0, to: vocabSize, by: step) {
            let end = Swift.min(start + step, vocabSize)
            let w8Chunk = source.weight[start ..< end]
            let sChunk = source.scales[start ..< end]
            let bChunk = sourceBiases[start ..< end]

            let (hiChunk, loChunk) = splitW8ToHiLo(w8Chunk)
            let sF32 = sChunk.asType(.float32)
            let bF32 = bChunk.asType(.float32)
            let hiSChunk = (sF32 * Float(16)).asType(scaleDType)
            let hiBChunk = (bF32 + Float(7.5) * sF32).asType(scaleDType)

            let w8F32 = dequantized(
                w8Chunk, scales: sChunk, biases: bChunk,
                groupSize: 64, bits: 8, mode: .affine, dtype: .float32)
            let w4F32 = dequantized(
                hiChunk, scales: hiSChunk, biases: hiBChunk,
                groupSize: 64, bits: 4, mode: .affine, dtype: .float32)
            let diffF32 = w8F32 - w4F32
            var rhoChunk = sqrt(sum(diffF32 * diffF32, axis: -1))

            // Cover both QMV (FP32 dequant) and Steel QMM (scale-dtype dequant)
            // as well as both symmetric cross-path pairs (QMM <-> QMV).
            if scaleDType != .float32 {
                let w8T = w8F32.asType(scaleDType).asType(.float32)
                let w4T = w4F32.asType(scaleDType).asType(.float32)
                let diffT = w8T - w4T
                let diffCross1 = w8F32 - w4T
                let diffCross2 = w8T - w4F32
                let rhoT = sqrt(sum(diffT * diffT, axis: -1))
                let rhoCross1 = sqrt(sum(diffCross1 * diffCross1, axis: -1))
                let rhoCross2 = sqrt(sum(diffCross2 * diffCross2, axis: -1))
                rhoChunk = maximum(rhoChunk, maximum(rhoT, maximum(rhoCross1, rhoCross2)))
            }

            eval(hiChunk, loChunk, hiSChunk, hiBChunk, rhoChunk)
            hiChunks.append(hiChunk)
            loChunks.append(loChunk)
            hiScaleChunks.append(hiSChunk)
            hiBiasChunks.append(hiBChunk)
            rhoChunks.append(rhoChunk)
        }

        // Materialize and release chunk arrays sequentially to minimize peak RAM.
        let hiWeight = hiChunks.count == 1 ? hiChunks[0] : concatenated(hiChunks, axis: 0)
        eval(hiWeight)
        hiChunks.removeAll(keepingCapacity: false)

        let loWeight = loChunks.count == 1 ? loChunks[0] : concatenated(loChunks, axis: 0)
        eval(loWeight)
        loChunks.removeAll(keepingCapacity: false)

        let hiScales =
            hiScaleChunks.count == 1 ? hiScaleChunks[0] : concatenated(hiScaleChunks, axis: 0)
        let hiBiases =
            hiBiasChunks.count == 1 ? hiBiasChunks[0] : concatenated(hiBiasChunks, axis: 0)
        let rowResidualNorm =
            rhoChunks.count == 1 ? rhoChunks[0] : concatenated(rhoChunks, axis: 0)
        eval(hiScales, hiBiases, rowResidualNorm, source.scales, sourceBiases)
        hiScaleChunks.removeAll(keepingCapacity: false)
        hiBiasChunks.removeAll(keepingCapacity: false)
        rhoChunks.removeAll(keepingCapacity: false)

        return Gemma4Hi4Lo4Embedding(
            hiWeight: hiWeight,
            loWeight: loWeight,
            scales: source.scales,
            biases: sourceBiases,
            hiScales: hiScales,
            hiBiases: hiBiases,
            rowResidualNorm: rowResidualNorm,
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            samplingStrategy: samplingStrategy == .off ? .exact : samplingStrategy)
    }

    // MARK: - Embedding & Dense Linear Overrides (0-ULP Bit-Exact)

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalShape = x.shape
        let flatIndices = x.flattened()
        guard flatIndices.size > 0 else {
            return MLXArray.zeros(originalShape + [hiddenSize], dtype: scales.dtype)
        }
        let w8 = gatherInterleaveHiLoToW8(indices: flatIndices)
        let out = dequantized(
            w8, scales: scales[flatIndices], biases: biases[flatIndices],
            groupSize: groupSize, bits: bits, mode: mode)
        return out.reshaped(originalShape + [-1])
    }

    public override func asLinear(_ x: MLXArray) -> MLXArray {
        asLinear(x, chunkRows: Self.denseChunkRows)
    }

    /// Row-exact batch-decode vocabulary projection (`B in 2...8, L == 1`):
    /// interleaves each chunk ONCE and shares it across all `B` rows via
    /// `Gemma4BatchDecodeProjection.project`, avoiding `B`-fold weight interleaving.
    func asBatchDecodeLinear(_ x: MLXArray, chunkRows: Int = denseChunkRows) -> MLXArray {
        projectChunked(chunkRows: chunkRows) { w8, s, b in
            Gemma4BatchDecodeProjection.project(
                x, weight: w8, scales: s, biases: b, groupSize: groupSize, bits: bits)
        }
    }

    /// Dense exact 8-bit vocabulary projection, reconstructing `w8` in bounded
    /// row chunks to cap transient memory while remaining 0-ULP bit-exact.
    public func asLinear(_ x: MLXArray, chunkRows: Int) -> MLXArray {
        projectChunked(chunkRows: chunkRows) { w8, s, b in
            quantizedMM(
                x, w8, scales: s, biases: b,
                transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        }
    }

    private func projectChunked(
        chunkRows: Int,
        project: (_ w8: MLXArray, _ scales: MLXArray, _ biases: MLXArray) -> MLXArray
    ) -> MLXArray {
        let alignedStep = Swift.max(64, (Swift.max(64, chunkRows) / 64) * 64)
        if vocabSize <= alignedStep {
            let w8 = Self.interleaveHiLoToW8(hi: hiWeight, lo: loWeight)
            return project(w8, scales, biases)
        }
        var partials: [MLXArray] = []
        partials.reserveCapacity((vocabSize + alignedStep - 1) / alignedStep)
        for start in stride(from: 0, to: vocabSize, by: alignedStep) {
            let end = Swift.min(start + alignedStep, vocabSize)
            let w8Chunk = Self.interleaveHiLoToW8(
                hi: hiWeight[start ..< end], lo: loWeight[start ..< end])
            partials.append(project(w8Chunk, scales[start ..< end], biases[start ..< end]))
        }
        return concatenated(partials, axis: -1)
    }

    // MARK: - Bit-Exact Sparse LM-Head Sampling

    /// Exact sparse-candidate LM-head sampling directly from post-norm hidden
    /// states `[B, H]` or `[B, 1, H]`. Graph-build only (zero host syncs).
    /// Returns `(tokens: [B] int32, nonFinite: [B] int32)` or `nil` if input
    /// shapes/dtypes are unsupported.
    public func sparseExactHeadSample(
        hidden: MLXArray,
        mode: CBv2SparseHeadSamplingMode,
        softcap: Float = 0,
        candidateCapacity: Int = defaultCandidateCapacity
    ) -> (tokens: MLXArray, nonFinite: MLXArray)? {
        guard
            let diag = sparseExactHeadSampleWithDiagnostics(
                hidden: hidden, mode: mode, softcap: softcap,
                candidateCapacity: candidateCapacity)
        else { return nil }
        return (diag.tokens, diag.nonFinite)
    }

    /// Same as ``sparseExactHeadSample(hidden:mode:softcap:candidateCapacity:)``
    /// with per-row candidate counts, overflow flags, and fallback flags exposed
    /// for testing and telemetry.
    public func sparseExactHeadSampleWithDiagnostics(
        hidden: MLXArray,
        mode: CBv2SparseHeadSamplingMode,
        softcap: Float = 0,
        candidateCapacity: Int = defaultCandidateCapacity
    ) -> SparseHeadSampleDiagnostics? {
        let x2D: MLXArray
        if hidden.ndim == 2 {
            x2D = hidden
        } else if hidden.ndim == 3 && hidden.dim(1) == 1 {
            x2D = hidden.squeezed(axis: 1)
        } else {
            return nil
        }

        let batchSize = x2D.dim(0)
        guard batchSize > 0,
            x2D.dim(1) == hiddenSize,
            candidateCapacity >= 1,
            x2D.dtype == .bfloat16 || x2D.dtype == .float16 || x2D.dtype == .float32
        else { return nil }

        switch mode {
        case .greedy:
            break
        case .gumbel(let temperatures, let expNoise):
            guard temperatures.size == batchSize,
                expNoise.shape == [batchSize, vocabSize]
            else { return nil }
        case .filtered(let temperatures, let topK, let topP, let minP, let expNoise):
            guard temperatures.size == batchSize,
                topK.count == batchSize,
                topP.count == batchSize,
                minP.count == batchSize,
                expNoise.shape == [batchSize, vocabSize]
            else { return nil }
        }

        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        if Device.defaultDevice().deviceType == .gpu {
            return sparseExactHeadSampleGPU(
                x2D: x2D, mode: mode, softcap: softcap, candidateCapacity: candidateCapacity)
        }
        #endif
        return sparseExactHeadSampleCPUFallback(
            x2D: x2D, mode: mode, softcap: softcap)
    }

    // MARK: - GPU Sparse Cauchy-Schwarz Pipeline

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private func sparseExactHeadSampleGPU(
        x2D: MLXArray,
        mode: CBv2SparseHeadSamplingMode,
        softcap: Float,
        candidateCapacity: Int
    ) -> SparseHeadSampleDiagnostics {
        let effectiveCap = Swift.max(1, Swift.min(candidateCapacity, vocabSize))
        let hasSoftcap = softcap > 0

        // Pass 1: MLX native 4-bit quantizedMM on the upper bit-plane only.
        let zMidRaw = quantizedMM(
            x2D, hiWeight, scales: hiScales, biases: hiBiases,
            transpose: true, groupSize: 64, bits: 4, mode: .affine)

        if samplingStrategy == .refine {
            return runRefineTopKGPU(
                x2D: x2D, zMidRaw: zMidRaw, mode: mode, softcap: softcap,
                topKBudget: Swift.min(Swift.max(Self.defaultRefineTopK, 1), vocabSize))
        }

        let zMid = zMidRaw.asType(.float32)

        // Non-finite detection matching CBv2NonFiniteLogits.flags after softcap:
        // with softcap > 0, tanh(±Inf/c)*c is finite and only NaN stays non-finite.
        let midNonFinite: MLXArray =
            hasSoftcap
            ? MLX.any(isNaN(zMid), axis: -1)
            : isNaN(sum(zMid - zMid, axis: -1))

        // Pass 2: Fused Cauchy-Schwarz [lower, upper] bounds.
        let xF32 = x2D.asType(.float32)
        let xNorm = sqrt(sum(xF32 * xF32, axis: -1, keepDims: true))
        let relEps: Float
        switch zMidRaw.dtype {
        case .bfloat16:
            relEps = Float(1.0 / 128.0)
        case .float16:
            relEps = Float(1.0 / 1024.0)
        default:
            relEps = Float(1e-5)
        }
        let rhoRow = rowResidualNorm.reshaped([1, vocabSize])
        let bounds: [MLXArray]
        if hasSoftcap {
            bounds = Self.compiledBoundsSoftcap([
                zMid, xNorm, rhoRow, MLXArray(relEps), MLXArray(softcap),
            ])
        } else {
            bounds = Self.compiledBoundsNoSoftcap([
                zMid, xNorm, rhoRow, MLXArray(relEps),
            ])
        }
        let lower = bounds[0]
        let upper = bounds[1]

        switch mode {
        case .greedy:
            return runGreedyGPU(
                x2D: x2D, outputDType: zMidRaw.dtype,
                lower: lower, upper: upper, midNonFinite: midNonFinite,
                softcap: softcap, effectiveCap: effectiveCap)

        case .gumbel(let temperatures, let expNoise):
            return runGumbelGPU(
                x2D: x2D, outputDType: zMidRaw.dtype,
                lower: lower, upper: upper, midNonFinite: midNonFinite,
                temperatures: temperatures, expNoise: expNoise,
                softcap: softcap, effectiveCap: effectiveCap)

        case .filtered(let temperatures, let topK, let topP, let minP, let expNoise):
            return runFilteredGPU(
                x2D: x2D, outputDType: zMidRaw.dtype,
                lower: lower, upper: upper, midNonFinite: midNonFinite,
                temperatures: temperatures, topK: topK, topP: topP, minP: minP,
                expNoise: expNoise, softcap: softcap, effectiveCap: effectiveCap)
        }
    }

    /// Fast single-pass O(V) Hi4 screening + exact 8-bit refinement of `topKBudget`
    /// candidates (`K = 64` for greedy/gumbel, `128..256` for filtered). For large
    /// vocabularies (`V >= 8192` with enough 256-token stripes for `targetK`), uses
    /// a 32-token warp-striped Metal pass (`fastChunkTopK`) that spreads low-ID
    /// SentencePiece clusters across chunks and merges NaN/Inf scanning into one pass.
    private func runRefineTopKGPU(
        x2D: MLXArray,
        zMidRaw: MLXArray,
        mode: CBv2SparseHeadSamplingMode,
        softcap: Float,
        topKBudget: Int
    ) -> SparseHeadSampleDiagnostics {
        let batchSize = x2D.dim(0)
        let hasSoftcap = softcap > 0
        let zeroU8 = MLXArray.zeros([batchSize], dtype: .uint8)

        func detectNonFinite(_ scores: MLXArray, checkInf: Bool) -> MLXArray {
            guard checkInf else { return MLX.any(isNaN(scores), axis: -1) }
            let f32 = scores.dtype == .float32 ? scores : scores.asType(.float32)
            return isNaN(sum(f32 - f32, axis: -1))
        }

        func screenCandidates(
            scores: MLXArray,
            targetK: Int,
            checkInf: Bool
        ) -> (candIDs: MLXArray, effectiveK: Int, midNonFinite: MLXArray) {
            let k = Swift.min(targetK, vocabSize)
            let emitPerChunk = k > 64 ? 4 : 2
            let numChunks = Swift.max(1, (k + emitPerChunk - 1) / emitPerChunk)
            if vocabSize >= 8192 && k < vocabSize && numChunks <= (vocabSize / 256) {
                let (ids, bad) = Self.fastChunkTopK(
                    scores: scores,
                    numChunks: numChunks,
                    emitPerChunk: emitPerChunk,
                    checkInf: checkInf)
                return (ids, numChunks * emitPerChunk, bad)
            }
            let midBad = detectNonFinite(scores, checkInf: checkInf)
            let ids: MLXArray
            if k < vocabSize {
                ids = argPartition(-scores, kth: k - 1, axis: -1)[0..., ..<k].asType(.int32)
            } else {
                ids = broadcast(
                    MLXArray.arange(vocabSize, dtype: .int32).reshaped([1, vocabSize]),
                    to: [batchSize, vocabSize])
            }
            return (ids, k, midBad)
        }

        switch mode {
        case .greedy:
            let (candIDs, k, midNonFinite) = screenCandidates(
                scores: zMidRaw, targetK: topKBudget, checkInf: !hasSoftcap)
            let counts = MLXArray.full([batchSize], values: MLXArray(Int32(k)))
            let (candLogits, _, candNonFinite) = refineCandidates(
                x2D: x2D, candIDs: candIDs, candCounts: counts,
                softcap: softcap, effectiveCap: k)
            let bestSlot = argMax(candLogits, axis: -1).asType(.int32)
            let tokens = takeAlong(
                candIDs, bestSlot.expandedDimensions(axis: 1), axis: -1
            ).squeezed(axis: 1)
            let nonFinite = (midNonFinite .|| candNonFinite).asType(.int32)
            return SparseHeadSampleDiagnostics(
                tokens: which(nonFinite .!= Int32(0), MLXArray(Int32(0)), tokens),
                nonFinite: nonFinite,
                candidateCounts: counts, overflow: zeroU8, needsFallback: zeroU8)

        case .gumbel(let temperatures, let expNoise):
            let temps1D = temperatures.asType(.float32).reshaped([batchSize])
            let greedyMask = temps1D .< LogitsPipelineV2.greedyEpsilon
            let greedyMask2D = greedyMask.expandedDimensions(axis: 1)
            let safeTemps2D = which(
                greedyMask2D, MLXArray(Float(1)), temps1D.expandedDimensions(axis: 1))
            let noiseF32 = expNoise.asType(.float32)
            let zMidF32 = zMidRaw.asType(.float32)
            let midNonFinite = detectNonFinite(zMidF32, checkInf: !hasSoftcap)
            let zMidCapped = hasSoftcap ? (tanh(zMidF32 / softcap) * softcap) : zMidF32
            // Shift-invariant centering prevents finite logits / tiny temperature from overflowing FP32.
            let zMidShifted = zMidCapped - MLX.max(zMidCapped, axis: -1, keepDims: true)
            let g = -log(maximum(noiseF32, MLXArray(Float(1e-30))))
            let scoreMid = which(greedyMask2D, zMidCapped, zMidShifted / safeTemps2D + g)
            let (candIDs, k, _) = screenCandidates(
                scores: scoreMid, targetK: topKBudget, checkInf: false)
            let counts = MLXArray.full([batchSize], values: MLXArray(Int32(k)))
            let (candLogits, _, candNonFinite) = refineCandidates(
                x2D: x2D, candIDs: candIDs, candCounts: counts,
                softcap: softcap, effectiveCap: k)
            let probsCand = softmax(candLogits / safeTemps2D, axis: -1)
            let eCand = takeAlong(noiseF32, candIDs, axis: -1)
            let scores = which(greedyMask2D, candLogits, probsCand / eCand)
            let bestSlot = argMax(scores, axis: -1).asType(.int32)
            let tokens = takeAlong(
                candIDs, bestSlot.expandedDimensions(axis: 1), axis: -1
            ).squeezed(axis: 1)
            let nonFinite = (midNonFinite .|| candNonFinite).asType(.int32)
            return SparseHeadSampleDiagnostics(
                tokens: which(nonFinite .!= Int32(0), MLXArray(Int32(0)), tokens),
                nonFinite: nonFinite,
                candidateCounts: counts, overflow: zeroU8, needsFallback: zeroU8)

        case .filtered(let temperatures, let topK, let topP, let minP, let expNoise):
            let maxRequestedK = topK.filter { $0 > 0 && $0 < vocabSize }.max() ?? 0
            let targetK = Swift.min(
                Swift.max(Swift.max(topKBudget, 128), Swift.min(maxRequestedK * 2, 256)), vocabSize)
            let (candIDs, k, midNonFinite) = screenCandidates(
                scores: zMidRaw, targetK: targetK, checkInf: !hasSoftcap)
            let counts = MLXArray.full([batchSize], values: MLXArray(Int32(k)))
            let (candLogits, _, candNonFinite) = refineCandidates(
                x2D: x2D, candIDs: candIDs, candCounts: counts,
                softcap: softcap, effectiveCap: k)
            let temps1D = temperatures.asType(.float32).reshaped([batchSize])
            let greedyMask = temps1D .< LogitsPipelineV2.greedyEpsilon
            let safeTemps2D = which(
                greedyMask.expandedDimensions(axis: 1),
                MLXArray(Float(1)), temps1D.expandedDimensions(axis: 1))
            let topKDev = MLXArray(topK.map { $0 > 0 ? Int32(Swift.min($0, k)) : Int32(k) })
            let topPDev = MLXArray(topP.map { Float($0) })
            let minPDev = MLXArray(minP.map { Float($0) })

            let order = argSort(-candLogits, axis: -1)
            let sortedLogits = takeAlong(candLogits, order, axis: -1)
            let sortedIDs = takeAlong(candIDs, order, axis: -1)
            let sortedScaled = sortedLogits / safeTemps2D
            let sMax = sortedScaled[0..., 0 ..< 1]
            let weights = exp(sortedScaled - sMax)
            let zCand = sum(weights, axis: -1, keepDims: true)
            let precedingW = cumsum(weights, axis: -1) - weights

            let ranks = MLXArray.arange(k, dtype: .int32).reshaped([1, k])
            let keepK = ranks .< topKDev.expandedDimensions(axis: 1)
            let keepMinP = weights .>= minPDev.expandedDimensions(axis: 1)
            let topP2D = topPDev.expandedDimensions(axis: 1)
            let keepP = which(
                topP2D .<= Float(0),
                ranks .== Int32(0),
                which(topP2D .< Float(1), precedingW .<= (topP2D * zCand), MLXArray(true)))
            let keep = (keepK .&& keepMinP .&& keepP) .|| (ranks .== Int32(0))

            let keptWeights = which(keep, weights, MLXArray(Float(0)))
            let probs =
                keptWeights
                / maximum(sum(keptWeights, axis: -1, keepDims: true), MLXArray(Float(1e-30)))
            let eSorted = takeAlong(expNoise.asType(.float32), sortedIDs, axis: -1)
            let scores = which(
                greedyMask.expandedDimensions(axis: 1),
                which(keep, sortedLogits, MLXArray(-Float.infinity)),
                which(keep, probs / eSorted, MLXArray(-Float.infinity)))
            let bestSlot = argMax(scores, axis: -1).asType(.int32)
            let tokens = takeAlong(
                sortedIDs, bestSlot.expandedDimensions(axis: 1), axis: -1
            ).squeezed(axis: 1)
            let nonFinite = (midNonFinite .|| candNonFinite).asType(.int32)
            return SparseHeadSampleDiagnostics(
                tokens: which(nonFinite .!= Int32(0), MLXArray(Int32(0)), tokens),
                nonFinite: nonFinite,
                candidateCounts: counts, overflow: zeroU8, needsFallback: zeroU8)
        }
    }

    private func runGreedyGPU(
        x2D: MLXArray,
        outputDType: DType,
        lower: MLXArray,
        upper: MLXArray,
        midNonFinite: MLXArray,
        softcap: Float,
        effectiveCap: Int
    ) -> SparseHeadSampleDiagnostics {
        let batchSize = x2D.dim(0)
        let rowMaxLower = MLX.max(lower, axis: -1)
        let tauMargin = maximum(MLXArray(Float(1e-4)), MLX.abs(rowMaxLower) * Float(1e-5))
        let tau = rowMaxLower - tauMargin

        let (candIDs, candCounts, overflow) = Self.compactCandidates(
            upper: upper, threshold: tau, capacity: effectiveCap)

        let (candLogits, validMask, candNonFinite) = refineCandidates(
            x2D: x2D, candIDs: candIDs, candCounts: candCounts,
            softcap: softcap, effectiveCap: effectiveCap)

        let maskedCand = which(validMask, candLogits, MLXArray(-Float.infinity))
        let bestSlot = argMax(maskedCand, axis: -1).asType(.int32)
        let sparseTokens = takeAlong(
            candIDs, bestSlot.expandedDimensions(axis: 1), axis: -1
        ).squeezed(axis: 1)
        let sparseNonFinite = (midNonFinite .|| candNonFinite).asType(.int32)

        let needsFallbackBool =
            ((overflow .!= UInt8(0)) .|| (candCounts .<= Int32(0)))
            .&& logicalNot(midNonFinite)
        let needsFallbackU8 = needsFallbackBool.asType(.uint8)

        let dummyTemps = MLXArray.zeros([batchSize], dtype: .float32)
        let dummyTopK = MLXArray.full([batchSize], values: MLXArray(Int32(vocabSize)))
        let dummyTopP = MLXArray.full([batchSize], values: MLXArray(Float(2)))
        let dummyMinP = MLXArray.zeros([batchSize], dtype: .float32)
        let dummyNoise = MLXArray.ones([1, 1], type: Float.self)

        let (finalTokens, finalNonFinite) = runConditionalFallbackGPU(
            x2D: x2D, outputDType: outputDType,
            needsFallback: needsFallbackU8,
            sparseTokens: sparseTokens, sparseNonFinite: sparseNonFinite,
            temps: dummyTemps, topK: dummyTopK, topP: dummyTopP, minP: dummyMinP,
            expNoise: dummyNoise, softcap: softcap, modeCode: 0)

        return SparseHeadSampleDiagnostics(
            tokens: finalTokens, nonFinite: finalNonFinite,
            candidateCounts: candCounts, overflow: overflow,
            needsFallback: needsFallbackU8)
    }

    private func runGumbelGPU(
        x2D: MLXArray,
        outputDType: DType,
        lower: MLXArray,
        upper: MLXArray,
        midNonFinite: MLXArray,
        temperatures: MLXArray,
        expNoise: MLXArray,
        softcap: Float,
        effectiveCap: Int
    ) -> SparseHeadSampleDiagnostics {
        let batchSize = x2D.dim(0)
        let temps1D = temperatures.asType(.float32).reshaped([batchSize])
        let greedyMask = temps1D .< LogitsPipelineV2.greedyEpsilon
        let greedyMask2D = greedyMask.expandedDimensions(axis: 1)
        let safeTemps2D = which(
            greedyMask2D, MLXArray(Float(1)), temps1D.expandedDimensions(axis: 1))

        let noiseF32 = expNoise.asType(.float32)
        let g = -log(maximum(noiseF32, MLXArray(Float(1e-30))))
        let scoreLo = which(greedyMask2D, lower, lower / safeTemps2D + g)
        let scoreHi = which(greedyMask2D, upper, upper / safeTemps2D + g)

        let rowMaxLo = MLX.max(scoreLo, axis: -1)
        let tauMargin = maximum(MLXArray(Float(1e-3)), MLX.abs(rowMaxLo) * Float(1e-4))
        let tau = rowMaxLo - tauMargin

        let (candIDs, candCounts, overflow) = Self.compactCandidates(
            upper: scoreHi, threshold: tau, capacity: effectiveCap)

        let (candLogits, validMask, candNonFinite) = refineCandidates(
            x2D: x2D, candIDs: candIDs, candCounts: candCounts,
            softcap: softcap, effectiveCap: effectiveCap)

        let maskedCand = which(validMask, candLogits, MLXArray(-Float.infinity))
        let scaledCand = maskedCand / safeTemps2D
        let probsCand = softmax(scaledCand, axis: -1)
        let eCandRaw = takeAlong(noiseF32, candIDs, axis: -1)
        let eCand = which(validMask, eCandRaw, MLXArray(Float(1)))

        let sampledSlot = argMax(probsCand / eCand, axis: -1).asType(.int32)
        let greedySlot = argMax(maskedCand, axis: -1).asType(.int32)
        let bestSlot = which(greedyMask, greedySlot, sampledSlot)
        let sparseTokens = takeAlong(
            candIDs, bestSlot.expandedDimensions(axis: 1), axis: -1
        ).squeezed(axis: 1)
        let sparseNonFinite = (midNonFinite .|| candNonFinite).asType(.int32)

        let needsFallbackBool =
            ((overflow .!= UInt8(0)) .|| (candCounts .<= Int32(0)))
            .&& logicalNot(midNonFinite)
        let needsFallbackU8 = needsFallbackBool.asType(.uint8)

        let dummyTopK = MLXArray.full([batchSize], values: MLXArray(Int32(vocabSize)))
        let dummyTopP = MLXArray.full([batchSize], values: MLXArray(Float(2)))
        let dummyMinP = MLXArray.zeros([batchSize], dtype: .float32)

        let (finalTokens, finalNonFinite) = runConditionalFallbackGPU(
            x2D: x2D, outputDType: outputDType,
            needsFallback: needsFallbackU8,
            sparseTokens: sparseTokens, sparseNonFinite: sparseNonFinite,
            temps: temps1D, topK: dummyTopK, topP: dummyTopP, minP: dummyMinP,
            expNoise: noiseF32, softcap: softcap, modeCode: 1)

        return SparseHeadSampleDiagnostics(
            tokens: finalTokens, nonFinite: finalNonFinite,
            candidateCounts: candCounts, overflow: overflow,
            needsFallback: needsFallbackU8)
    }

    private func runFilteredGPU(
        x2D: MLXArray,
        outputDType: DType,
        lower: MLXArray,
        upper: MLXArray,
        midNonFinite: MLXArray,
        temperatures: MLXArray,
        topK: [Int],
        topP: [Float],
        minP: [Float],
        expNoise: MLXArray,
        softcap: Float,
        effectiveCap: Int
    ) -> SparseHeadSampleDiagnostics {
        let batchSize = x2D.dim(0)
        let temps1D = temperatures.asType(.float32).reshaped([batchSize])
        let greedyMask = temps1D .< LogitsPipelineV2.greedyEpsilon
        let safeTemps1D = which(greedyMask, MLXArray(Float(1)), temps1D)
        let safeTemps2D = safeTemps1D.expandedDimensions(axis: 1)

        // Normalize filter parameters identically to LogitsPipelineV2.setRows.
        var normTopK = [Int32](repeating: Int32(vocabSize), count: batchSize)
        var normTopP = [Float](repeating: 2, count: batchSize)
        var normMinP = [Float](repeating: 0, count: batchSize)
        var anyActiveTopP = false
        var activeTopKs: [Int] = []
        activeTopKs.reserveCapacity(batchSize)

        for i in 0 ..< batchSize {
            if topK[i] > 0 && topK[i] < vocabSize {
                normTopK[i] = Int32(topK[i])
                if topK[i] <= effectiveCap {
                    activeTopKs.append(topK[i])
                }
            }
            if topP[i] > 0 && topP[i] < 1 {
                normTopP[i] = topP[i]
                anyActiveTopP = true
            } else if topP[i] <= 0 {
                normTopP[i] = 0
            }
            if minP[i] > 0 {
                normMinP[i] = Swift.min(minP[i], 1)
            }
        }

        // Greedy rows disable all filtering transforms (argmax-invariant).
        let topKDev = which(greedyMask, MLXArray(Int32(vocabSize)), MLXArray(normTopK))
        let topPDev = which(greedyMask, MLXArray(Float(2)), MLXArray(normTopP))
        let minPDev = which(greedyMask, MLXArray(Float(0)), MLXArray(normMinP))

        let notGreedy = logicalNot(greedyMask)
        let hasUsableTopK =
            notGreedy .&& (topKDev .< Int32(vocabSize)) .&& (topKDev .<= Int32(effectiveCap))
        let hasTooLargeTopK =
            notGreedy .&& (topKDev .< Int32(vocabSize)) .&& (topKDev .> Int32(effectiveCap))
        let hasMinP = notGreedy .&& (minPDev .> Float(0))
        let hasTopPZero = notGreedy .&& (topPDev .<= Float(0))
        let hasActiveTopP = notGreedy .&& (topPDev .> Float(0)) .&& (topPDev .< Float(1))
        let unfilteredStochastic =
            notGreedy .&& (topKDev .>= Int32(vocabSize)) .&& (topPDev .>= Float(1))
            .&& (minPDev .<= Float(0))
        let topKOverflowStatic =
            hasTooLargeTopK .&& logicalNot(hasMinP) .&& logicalNot(hasTopPZero)
            .&& logicalNot(hasActiveTopP)
        let staticFallback = unfilteredStochastic .|| topKOverflowStatic

        // Build sound per-row compaction threshold tau[b] in post-softcap logit space.
        let maxLower = MLX.max(lower, axis: -1)
        var tau = MLXArray.full([batchSize], values: MLXArray(-Float.infinity))

        if let maxActiveK = activeTopKs.max() {
            let tauTopK: MLXArray
            let minActiveK = activeTopKs.min() ?? maxActiveK
            if minActiveK == maxActiveK {
                tauTopK = MLX.min(top(lower, k: maxActiveK, axis: -1), axis: -1)
            } else {
                let topVals = sorted(top(lower, k: maxActiveK, axis: -1), axis: -1)
                let clampedK = minimum(
                    maximum(topKDev, MLXArray(Int32(1))), MLXArray(Int32(maxActiveK)))
                let kthCol = (MLXArray(Int32(maxActiveK)) - clampedK).expandedDimensions(axis: 1)
                tauTopK = takeAlong(topVals, kthCol, axis: -1).squeezed(axis: 1)
            }
            tau = which(hasUsableTopK, maximum(tau, tauTopK), tau)
        }

        let safeMinPForLog = maximum(minPDev, MLXArray(Float(1e-30)))
        let tauMinP = maxLower + safeTemps1D * log(safeMinPForLog)
        tau = which(hasMinP, maximum(tau, tauMinP), tau)

        tau = which(hasTopPZero .|| greedyMask, maxLower, tau)

        // Probe threshold for any active top-P row (including combined topK+topP,
        // e.g. Gemma 4 default top_k=64, top_p=0.95); certified below via exact [Z_lo, Z_hi].
        let tauTopPProbe = maxLower + safeTemps1D * Float(Foundation.log(1e-3 as Double))
        let usedTopPProbe =
            hasActiveTopP .&& logicalNot(hasTopPZero) .&& logicalNot(greedyMask)
            .&& (tauTopPProbe .> tau)
        tau = which(usedTopPProbe, tauTopPProbe, tau)

        // Ensure the true global maximum token always survives into the candidate set.
        tau = minimum(tau, maxLower)
        let tauMargin = maximum(MLXArray(Float(1e-4)), MLX.abs(tau) * Float(1e-5))
        tau = tau - tauMargin
        tau = which(staticFallback, MLXArray(Float.infinity), tau)

        let (candIDs, candCounts, overflow) = Self.compactCandidates(
            upper: upper, threshold: tau, capacity: effectiveCap)

        let (candLogits, validMask, candNonFinite) = refineCandidates(
            x2D: x2D, candIDs: candIDs, candCounts: candCounts,
            softcap: softcap, effectiveCap: effectiveCap)

        let maskedCand = which(validMask, candLogits, MLXArray(-Float.infinity))
        let scaledCand = maskedCand / safeTemps2D

        // Sort candidates descending identically to LogitsPipelineV2.applyTopKTopPMinP.
        let sortedOrder = argSort(-scaledCand, axis: -1)
        let sortedScaled = takeAlong(scaledCand, sortedOrder, axis: -1)
        let sortedValid = takeAlong(validMask, sortedOrder, axis: -1)
        let sortedUnscaled = takeAlong(maskedCand, sortedOrder, axis: -1)

        let sMax = sortedScaled[0..., ..<1]
        let weights = which(
            sortedValid, exp(minimum(sortedScaled - sMax, MLXArray(Float(0)))),
            MLXArray(Float(0)))
        let zC = sum(weights, axis: -1, keepDims: true)
        let cumW = cumsum(weights, axis: -1)
        let precedingW = cumW - weights

        let ranks = MLXArray.arange(effectiveCap, dtype: .int32).reshaped([1, effectiveCap])
        let topK2D = topKDev.expandedDimensions(axis: 1)
        let minP2D = minPDev.expandedDimensions(axis: 1)

        let keepTopK = sortedValid .&& (ranks .< topK2D)
        let keepMinP = weights .>= minP2D
        let keepKM = keepTopK .&& keepMinP

        // Guard rare FP32 rounding boundary on minP when Z_C != Z_full.
        let minPAmbiguous = MLX.any(
            keepTopK .&& (minP2D .> Float(0))
                .&& (MLX.abs(weights - minP2D) .<= (minP2D * Float(1e-5))),
            axis: -1)

        let aboveTau = sortedValid .&& (sortedUnscaled .>= tau.expandedDimensions(axis: 1))

        let keepTopP: MLXArray
        let topPAmbiguous: MLXArray
        if anyActiveTopP {
            let tailMask = upper .< tau.expandedDimensions(axis: 1)
            let tailExpLo = exp(minimum(lower / safeTemps2D - sMax, MLXArray(Float(80))))
            let tailExpHi = exp(minimum(upper / safeTemps2D - sMax, MLXArray(Float(80))))
            let zTailLo = sum(
                which(tailMask, tailExpLo, MLXArray(Float(0))), axis: -1, keepDims: true)
            let zTailHi = sum(
                which(tailMask, tailExpHi, MLXArray(Float(0))), axis: -1, keepDims: true)
            let zLo = (zC + zTailLo) * Float(1.0 - 1e-4)
            let zHi = (zC + zTailHi) * Float(1.0 + 1e-4)

            let topP2D = topPDev.expandedDimensions(axis: 1)
            let activeTopP2D = hasActiveTopP.expandedDimensions(axis: 1)
            let topPZero2D = hasTopPZero.expandedDimensions(axis: 1)

            let keepPLo = which(
                activeTopP2D,
                precedingW .<= (topP2D * zLo),
                which(topPZero2D, ranks .== Int32(0), MLXArray(true)))
            let keepPHi = which(
                activeTopP2D,
                precedingW .<= (topP2D * zHi),
                which(topPZero2D, ranks .== Int32(0), MLXArray(true)))

            let boundaryAmbiguous = MLX.any(keepKM .&& (keepPLo .!= keepPHi), axis: -1)
            let massAboveTau = sum(which(aboveTau, weights, MLXArray(Float(0))), axis: -1)
            let topPProbeInsufficient =
                usedTopPProbe .&& (massAboveTau .<= (topPDev * zHi.squeezed(axis: 1)))

            topPAmbiguous = (boundaryAmbiguous .|| topPProbeInsufficient)
            keepTopP = keepPLo
        } else {
            let topPZero2D = hasTopPZero.expandedDimensions(axis: 1)
            keepTopP = which(topPZero2D, ranks .== Int32(0), MLXArray(true))
            topPAmbiguous = MLXArray.zeros([batchSize], dtype: .bool)
        }

        // Rank 0 is always kept by construction (matches LogitsPipelineV2).
        let keepSorted = (keepKM .&& keepTopP) .|| (sortedValid .&& (ranks .== Int32(0)))
        // Belt-and-suspenders check: every surviving kept token must sit above tau
        // so no pruned non-candidate can precede it in the full vocabulary.
        let orderingUnverified = MLX.any(keepSorted .&& logicalNot(aboveTau), axis: -1)
        let maskedSorted = which(keepSorted, sortedScaled, MLXArray(-Float.infinity))

        // Scatter back to ascending token-ID order so softmax + argMax tie-breaking
        // matches LogitsPipelineV2 + SamplerV2 identically.
        let inverseOrder = putAlong(
            MLXArray.zeros(like: sortedOrder),
            sortedOrder,
            values: broadcast(ranks.asType(sortedOrder.dtype), to: sortedOrder.shape),
            axis: -1)
        let maskedCandFiltered = takeAlong(maskedSorted, inverseOrder, axis: -1)
        let keepCandOrder = takeAlong(keepSorted, inverseOrder, axis: -1)

        let noiseF32 = expNoise.asType(.float32)
        let eCandRaw = takeAlong(noiseF32, candIDs, axis: -1)
        let eCand = which(keepCandOrder, eCandRaw, MLXArray(Float(1)))
        let probsKept = softmax(maskedCandFiltered, axis: -1)

        let sampledSlot = argMax(probsKept / eCand, axis: -1).asType(.int32)
        let greedySlot = argMax(maskedCand, axis: -1).asType(.int32)
        let bestSlot = which(greedyMask, greedySlot, sampledSlot)
        let sparseTokens = takeAlong(
            candIDs, bestSlot.expandedDimensions(axis: 1), axis: -1
        ).squeezed(axis: 1)
        let sparseNonFinite = (midNonFinite .|| candNonFinite).asType(.int32)

        let needsFallbackBool =
            ((overflow .!= UInt8(0))
                .|| (candCounts .<= Int32(0))
                .|| staticFallback
                .|| minPAmbiguous
                .|| topPAmbiguous
                .|| orderingUnverified)
            .&& logicalNot(midNonFinite)
        let needsFallbackU8 = needsFallbackBool.asType(.uint8)

        let (finalTokens, finalNonFinite) = runConditionalFallbackGPU(
            x2D: x2D, outputDType: outputDType,
            needsFallback: needsFallbackU8,
            sparseTokens: sparseTokens, sparseNonFinite: sparseNonFinite,
            temps: temps1D, topK: topKDev, topP: topPDev, minP: minPDev,
            expNoise: noiseF32, softcap: softcap, modeCode: 2)

        return SparseHeadSampleDiagnostics(
            tokens: finalTokens, nonFinite: finalNonFinite,
            candidateCounts: candCounts, overflow: overflow,
            needsFallback: needsFallbackU8)
    }

    /// Pass 3: gather + interleave surviving candidates into exact 8-bit weights
    /// and score them with MLX's native batched `qmv` 8-bit `quantizedMM`.
    private func refineCandidates(
        x2D: MLXArray,
        candIDs: MLXArray,
        candCounts: MLXArray,
        softcap: Float,
        effectiveCap: Int
    ) -> (logits: MLXArray, validMask: MLXArray, nonFinite: MLXArray) {
        let batchSize = x2D.dim(0)
        let packedCols8 = hiddenSize / 4
        let groups = hiddenSize / 64

        let flatIDs = candIDs.reshaped([-1])
        let w8Cand = gatherInterleaveHiLoToW8(indices: flatIDs)
            .reshaped([batchSize, effectiveCap, packedCols8])
        let scalesCand = scales[flatIDs].reshaped([batchSize, effectiveCap, groups])
        let biasesCand = biases[flatIDs].reshaped([batchSize, effectiveCap, groups])

        let x3D = x2D.expandedDimensions(axis: 1)
        let rawCandLogits = quantizedMM(
            x3D, w8Cand, scales: scalesCand, biases: biasesCand,
            transpose: true, groupSize: 64, bits: 8, mode: .affine
        ).squeezed(axis: 1)

        let candLogits =
            softcap > 0
            ? Self.applySoftcap(rawCandLogits, softcap: softcap)
            : rawCandLogits.asType(.float32)

        let slotIndices = MLXArray.arange(effectiveCap, dtype: .int32).reshaped([1, effectiveCap])
        let validMask = slotIndices .< candCounts.expandedDimensions(axis: 1)
        let validResidue = which(validMask, candLogits - candLogits, MLXArray(Float(0)))
        let candNonFinite = isNaN(sum(validResidue, axis: -1))
        return (candLogits, validMask, candNonFinite)
    }

    /// Zero-host-sync GPU fallback executed only for rows with `needsFallback[b] != 0`.
    /// Rows with `needsFallback[b] == 0` exit on the first instruction with zero
    /// weight reads.
    private func runConditionalFallbackGPU(
        x2D: MLXArray,
        outputDType: DType,
        needsFallback: MLXArray,
        sparseTokens: MLXArray,
        sparseNonFinite: MLXArray,
        temps: MLXArray,
        topK: MLXArray,
        topP: MLXArray,
        minP: MLXArray,
        expNoise: MLXArray,
        softcap: Float,
        modeCode: Int
    ) -> (tokens: MLXArray, nonFinite: MLXArray) {
        let batchSize = x2D.dim(0)
        let tilesY = Swift.min((vocabSize + 7) / 8, 2048)
        let softcapArr = MLXArray([softcap])

        let fallbackLogits = Self.conditionalDenseLogitsKernel(
            [x2D, hiWeight, loWeight, scales, biases, needsFallback, softcapArr],
            template: [("TX", x2D.dtype), ("TS", scales.dtype), ("TO", outputDType)],
            grid: (32, tilesY * 8, batchSize),
            threadGroup: (32, 8, 1),
            outputShapes: [[batchSize, vocabSize]],
            outputDTypes: [.float32])[0]

        let outputs = Self.conditionalSampleKernel(
            [
                fallbackLogits, needsFallback, sparseTokens, sparseNonFinite,
                temps, topK, topP, minP, expNoise,
            ],
            template: [("MODE", modeCode)],
            grid: (256, batchSize, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[batchSize], [batchSize]],
            outputDTypes: [.int32, .int32])
        return (outputs[0], outputs[1])
    }
    #endif

    // MARK: - Portable CPU Fallback

    private func sparseExactHeadSampleCPUFallback(
        x2D: MLXArray,
        mode: CBv2SparseHeadSamplingMode,
        softcap: Float
    ) -> SparseHeadSampleDiagnostics {
        let batchSize = x2D.dim(0)
        let rawLogits = asLinear(x2D)
        let logits =
            softcap > 0
            ? Self.applySoftcap(rawLogits, softcap: softcap)
            : rawLogits.asType(.float32)
        let nonFinite = isNaN(sum(logits - logits, axis: -1)).asType(.int32)
        let fullCounts = MLXArray.full([batchSize], values: MLXArray(Int32(vocabSize)))
        let noOverflow = MLXArray.zeros([batchSize], dtype: .uint8)
        let didFallback = MLXArray.ones([batchSize], dtype: .uint8)

        switch mode {
        case .greedy:
            let tokens = argMax(logits, axis: -1).asType(.int32)
            return SparseHeadSampleDiagnostics(
                tokens: tokens, nonFinite: nonFinite,
                candidateCounts: fullCounts, overflow: noOverflow,
                needsFallback: didFallback)

        case .gumbel(let temperatures, let expNoise):
            let temps1D = temperatures.asType(.float32).reshaped([batchSize])
            let greedyMask = temps1D .< LogitsPipelineV2.greedyEpsilon
            let safeTemps2D = which(
                greedyMask.expandedDimensions(axis: 1), MLXArray(Float(1)),
                temps1D.expandedDimensions(axis: 1))
            let greedyTokens = argMax(logits, axis: -1).asType(.int32)
            let probs = softmax(logits / safeTemps2D, axis: -1)
            let sampledTokens = argMax(probs / expNoise.asType(.float32), axis: -1).asType(.int32)
            let tokens = which(greedyMask, greedyTokens, sampledTokens)
            return SparseHeadSampleDiagnostics(
                tokens: tokens, nonFinite: nonFinite,
                candidateCounts: fullCounts, overflow: noOverflow,
                needsFallback: didFallback)

        case .filtered(let temperatures, let topK, let topP, let minP, let expNoise):
            let temps1D = temperatures.asType(.float32).reshaped([batchSize])
            let greedyMask = temps1D .< LogitsPipelineV2.greedyEpsilon
            let safeTemps2D = which(
                greedyMask.expandedDimensions(axis: 1), MLXArray(Float(1)),
                temps1D.expandedDimensions(axis: 1))

            var normTopK = [Int32](repeating: Int32(vocabSize), count: batchSize)
            var normTopP = [Float](repeating: 2, count: batchSize)
            var normMinP = [Float](repeating: 0, count: batchSize)
            for i in 0 ..< batchSize {
                if topK[i] > 0 && topK[i] < vocabSize { normTopK[i] = Int32(topK[i]) }
                if topP[i] > 0 && topP[i] < 1 {
                    normTopP[i] = topP[i]
                } else if topP[i] <= 0 {
                    normTopP[i] = 0
                }
                if minP[i] > 0 { normMinP[i] = Swift.min(minP[i], 1) }
            }
            let topK2D = which(greedyMask, MLXArray(Int32(vocabSize)), MLXArray(normTopK))
                .expandedDimensions(axis: 1)
            let topP2D = which(greedyMask, MLXArray(Float(2)), MLXArray(normTopP))
                .expandedDimensions(axis: 1)
            let minP2D = which(greedyMask, MLXArray(Float(0)), MLXArray(normMinP))
                .expandedDimensions(axis: 1)

            let scaled = logits / safeTemps2D
            let sortedIndices = argSort(-scaled, axis: -1)
            let sortedLogits = takeAlong(scaled, sortedIndices, axis: -1)
            let sortedProbs = softmax(sortedLogits, axis: -1)
            let cumulative = cumsum(sortedProbs, axis: -1)
            let ranks = MLXArray.arange(vocabSize, dtype: .int32).reshaped([1, vocabSize])
            let keepK = ranks .< topK2D
            let keepP = (cumulative - sortedProbs) .<= topP2D
            let keepM = sortedProbs .>= (minP2D * sortedProbs[0..., ..<1])
            let keep = keepK .&& keepP .&& keepM
            let masked = which(keep, sortedLogits, MLXArray(-Float.infinity))
            let inverse = putAlong(
                MLXArray.zeros(like: sortedIndices),
                sortedIndices,
                values: broadcast(ranks.asType(sortedIndices.dtype), to: sortedIndices.shape),
                axis: -1)
            let filtered = takeAlong(masked, inverse, axis: -1)

            let greedyTokens = argMax(logits, axis: -1).asType(.int32)
            let probs = softmax(filtered, axis: -1)
            let sampledTokens = argMax(probs / expNoise.asType(.float32), axis: -1).asType(.int32)
            let tokens = which(greedyMask, greedyTokens, sampledTokens)
            return SparseHeadSampleDiagnostics(
                tokens: tokens, nonFinite: nonFinite,
                candidateCounts: fullCounts, overflow: noOverflow,
                needsFallback: didFallback)
        }
    }

    // MARK: - Public Bit-Plane & Compaction Primitives

    /// Apply Gemma 4 final-logit softcapping (`tanh(x / cap) * cap`) in FP32.
    public static func applySoftcap(_ logits: MLXArray, softcap: Float) -> MLXArray {
        guard softcap > 0 else { return logits.asType(.float32) }
        return compiledSoftcap(logits, MLXArray(softcap))
    }

    /// Split MLX 8-bit packed `uint32` weights `[..., H/4]` into exact MLX 4-bit
    /// little-endian packed `hi` (`q >> 4`) and `lo` (`q & 0xF`) planes `[..., H/8]`.
    public static func splitW8ToHiLo(_ w8: MLXArray) -> (hi: MLXArray, lo: MLXArray) {
        precondition(
            w8.dtype == .uint32 && w8.ndim >= 1 && w8.dim(-1) % 2 == 0,
            "Gemma4Hi4Lo4Embedding.splitW8ToHiLo expects uint32 [..., H/4] with even last dim")
        let leadingShape = Array(w8.shape.dropLast())
        let cols4 = w8.dim(-1) / 2
        let rows = w8.size / w8.dim(-1)
        let outShape = leadingShape + [cols4]
        guard rows > 0 && cols4 > 0 else {
            let empty = MLXArray.zeros(outShape, dtype: .uint32)
            return (empty, empty)
        }

        let flat = w8.reshaped([rows, cols4 * 2])
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        if Device.defaultDevice().deviceType == .gpu {
            let gridX = ((cols4 + 255) / 256) * 256
            let gridY = Swift.min(rows, 32_768)
            let outputs = splitKernel(
                [flat],
                grid: (gridX, gridY, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[rows, cols4], [rows, cols4]],
                outputDTypes: [.uint32, .uint32])
            return (outputs[0].reshaped(outShape), outputs[1].reshaped(outShape))
        }
        #endif
        let (hi, lo) = splitW8ToHiLoPureMLX(flat, rows: rows, cols4: cols4)
        return (hi.reshaped(outShape), lo.reshaped(outShape))
    }

    /// Re-interleave MLX 4-bit `hi` and `lo` bit-planes `[..., H/8]` back into
    /// bit-exact MLX 8-bit packed `uint32` weights `[..., H/4]`.
    public static func interleaveHiLoToW8(hi: MLXArray, lo: MLXArray) -> MLXArray {
        precondition(
            hi.dtype == .uint32 && lo.dtype == .uint32 && hi.shape == lo.shape && hi.ndim >= 1,
            "Gemma4Hi4Lo4Embedding.interleaveHiLoToW8 expects matching uint32 [..., H/8] arrays")
        let leadingShape = Array(hi.shape.dropLast())
        let cols4 = hi.dim(-1)
        let rows = cols4 > 0 ? hi.size / cols4 : 0
        let outShape = leadingShape + [cols4 * 2]
        guard rows > 0 && cols4 > 0 else {
            return MLXArray.zeros(outShape, dtype: .uint32)
        }

        let hiFlat = hi.reshaped([rows, cols4])
        let loFlat = lo.reshaped([rows, cols4])
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        if Device.defaultDevice().deviceType == .gpu {
            let gridX = ((cols4 + 255) / 256) * 256
            let gridY = Swift.min(rows, 32_768)
            let w8 = interleaveKernel(
                [hiFlat, loFlat],
                grid: (gridX, gridY, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[rows, cols4 * 2]],
                outputDTypes: [.uint32])[0]
            return w8.reshaped(outShape)
        }
        #endif
        return interleaveHiLoToW8PureMLX(hiFlat, loFlat, rows: rows, cols4: cols4)
            .reshaped(outShape)
    }

    /// Compact per-row surviving candidate token IDs where `upper[b, v] >= threshold[b]`
    /// in ascending token-ID order up to `capacity`.
    public static func compactCandidates(
        upper: MLXArray,
        threshold: MLXArray,
        capacity: Int = defaultCandidateCapacity
    ) -> (candidateIDs: MLXArray, candidateCounts: MLXArray, overflow: MLXArray) {
        precondition(upper.ndim == 2, "compactCandidates expects 2D [B, V] upper bounds")
        let batchSize = upper.dim(0)
        let vocab = upper.dim(1)
        let kCap = Swift.max(1, capacity)
        guard batchSize > 0 else {
            return (
                MLXArray.zeros([0, kCap], dtype: .int32),
                MLXArray.zeros([0], dtype: .int32),
                MLXArray.zeros([0], dtype: .uint8)
            )
        }
        let upperF32 = upper.asType(.float32)
        let tauF32 = threshold.asType(.float32).reshaped([batchSize])

        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        if Device.defaultDevice().deviceType == .gpu {
            let outputs = compactCandidatesKernel(
                [upperF32, tauF32],
                template: [("K_CAP", kCap)],
                grid: (512, batchSize, 1),
                threadGroup: (512, 1, 1),
                outputShapes: [[batchSize, kCap], [batchSize], [batchSize]],
                outputDTypes: [.int32, .int32, .uint8])
            return (outputs[0], outputs[1], outputs[2])
        }
        #endif

        // Portable CPU fallback for compaction.
        let mask = upperF32 .>= tauF32.expandedDimensions(axis: 1)
        let countsRaw = sum(mask.asType(.int32), axis: -1)
        let overflow = (countsRaw .> Int32(kCap)).asType(.uint8)
        let counts = minimum(countsRaw, MLXArray(Int32(kCap)))
        let vocabIDs = broadcast(
            MLXArray.arange(vocab, dtype: .int32).reshaped([1, vocab]), to: [batchSize, vocab])
        let sentinel = MLXArray(Int32(vocab))
        let sortedIDs = sorted(which(mask, vocabIDs, sentinel), axis: -1)
        let candIDs: MLXArray
        if vocab >= kCap {
            let sliced = sortedIDs[0..., ..<kCap]
            candIDs = which(sliced .< Int32(vocab), sliced, MLXArray(Int32(0)))
        } else {
            let cleaned = which(sortedIDs .< Int32(vocab), sortedIDs, MLXArray(Int32(0)))
            let pad = MLXArray.zeros([batchSize, kCap - vocab], dtype: .int32)
            candIDs = concatenated([cleaned, pad], axis: -1)
        }
        return (candIDs, counts, overflow)
    }

    // MARK: - Internal Gather + Interleave & Pure-MLX Bit Helpers

    private func gatherInterleaveHiLoToW8(indices: MLXArray) -> MLXArray {
        let rows = indices.size
        let cols4 = hiddenSize / 8
        guard rows > 0 else {
            return MLXArray.zeros([0, cols4 * 2], dtype: .uint32)
        }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        if Device.defaultDevice().deviceType == .gpu && indices.dtype == .int32 {
            let flatIdx = indices.reshaped([rows])
            let gridX = ((cols4 + 255) / 256) * 256
            let gridY = Swift.min(rows, 32_768)
            return Self.gatherInterleaveKernel(
                [hiWeight, loWeight, flatIdx],
                grid: (gridX, gridY, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[rows, cols4 * 2]],
                outputDTypes: [.uint32])[0]
        }
        #endif
        return Self.interleaveHiLoToW8(hi: hiWeight[indices], lo: loWeight[indices])
    }

    private static func packLowNibblesPureMLX(_ w0: MLXArray, _ w1: MLXArray) -> MLXArray {
        let m0 = MLXArray(UInt32(0x0000_000F))
        let m1 = MLXArray(UInt32(0x0000_0F00))
        let m2 = MLXArray(UInt32(0x000F_0000))
        let m3 = MLXArray(UInt32(0x0F00_0000))
        let s4 = MLXArray(UInt32(4))
        let s8 = MLXArray(UInt32(8))
        let s12 = MLXArray(UInt32(12))
        let s16 = MLXArray(UInt32(16))

        let lo0 = bitwiseOr(
            bitwiseOr(bitwiseAnd(w0, m0), rightShift(bitwiseAnd(w0, m1), s4)),
            bitwiseOr(rightShift(bitwiseAnd(w0, m2), s8), rightShift(bitwiseAnd(w0, m3), s12)))
        let lo1 = bitwiseOr(
            bitwiseOr(bitwiseAnd(w1, m0), rightShift(bitwiseAnd(w1, m1), s4)),
            bitwiseOr(rightShift(bitwiseAnd(w1, m2), s8), rightShift(bitwiseAnd(w1, m3), s12)))
        return bitwiseOr(lo0, leftShift(lo1, s16))
    }

    private static func splitW8ToHiLoPureMLX(
        _ w8Flat: MLXArray, rows: Int, cols4: Int
    ) -> (hi: MLXArray, lo: MLXArray) {
        let paired = w8Flat.reshaped([rows, cols4, 2])
        let w0 = paired[0..., 0..., 0]
        let w1 = paired[0..., 0..., 1]
        let s4 = MLXArray(UInt32(4))
        let lo = packLowNibblesPureMLX(w0, w1)
        let hi = packLowNibblesPureMLX(rightShift(w0, s4), rightShift(w1, s4))
        return (hi, lo)
    }

    private static func unpackHalfToW8PureMLX(_ h16: MLXArray, _ l16: MLXArray) -> MLXArray {
        let m0 = MLXArray(UInt32(0x000F))
        let m1 = MLXArray(UInt32(0x00F0))
        let m2 = MLXArray(UInt32(0x0F00))
        let m3 = MLXArray(UInt32(0xF000))
        let s4 = MLXArray(UInt32(4))
        let s8 = MLXArray(UInt32(8))
        let s12 = MLXArray(UInt32(12))
        let s16 = MLXArray(UInt32(16))

        let b0 = bitwiseOr(leftShift(bitwiseAnd(h16, m0), s4), bitwiseAnd(l16, m0))
        let b1 = bitwiseOr(leftShift(bitwiseAnd(h16, m1), s8), leftShift(bitwiseAnd(l16, m1), s4))
        let b2 = bitwiseOr(leftShift(bitwiseAnd(h16, m2), s12), leftShift(bitwiseAnd(l16, m2), s8))
        let b3 = bitwiseOr(
            leftShift(bitwiseAnd(h16, m3), s16), leftShift(bitwiseAnd(l16, m3), s12))
        return bitwiseOr(bitwiseOr(b0, b1), bitwiseOr(b2, b3))
    }

    private static func interleaveHiLoToW8PureMLX(
        _ hiFlat: MLXArray, _ loFlat: MLXArray, rows: Int, cols4: Int
    ) -> MLXArray {
        let s16 = MLXArray(UInt32(16))
        let w0 = unpackHalfToW8PureMLX(hiFlat, loFlat)
        let w1 = unpackHalfToW8PureMLX(rightShift(hiFlat, s16), rightShift(loFlat, s16))
        return stacked([w0, w1], axis: -1).reshaped([rows, cols4 * 2])
    }

    // MARK: - Compiled Elementwise Fusions

    private static let compiledSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { x, cap in
            tanh(x / cap) * cap
        }
        return MLXHardwareInfo.isCompiledDecodeSupported
            ? compile(shapeless: true, body) : body
    }()

    private static let compiledBoundsNoSoftcap: @Sendable ([MLXArray]) -> [MLXArray] = {
        let body: @Sendable ([MLXArray]) -> [MLXArray] = { inputs in
            let zMid = inputs[0]
            let xNorm = inputs[1]
            let rho = inputs[2]
            let relEps = inputs[3]
            let csErr = xNorm * (rho + Float(1e-4))
            let err = csErr + relEps * (MLX.abs(zMid) + csErr) + Float(1e-4)
            return [zMid - err, zMid + err]
        }
        return MLXHardwareInfo.isCompiledDecodeSupported
            ? compile(shapeless: true, body) : body
    }()

    private static let compiledBoundsSoftcap: @Sendable ([MLXArray]) -> [MLXArray] = {
        let body: @Sendable ([MLXArray]) -> [MLXArray] = { inputs in
            let zMid = inputs[0]
            let xNorm = inputs[1]
            let rho = inputs[2]
            let relEps = inputs[3]
            let cap = inputs[4]
            let csErr = xNorm * (rho + Float(1e-4))
            let err = csErr + relEps * (MLX.abs(zMid) + csErr) + Float(1e-4)
            let lower = tanh((zMid - err) / cap) * cap - Float(1e-5)
            let upper = tanh((zMid + err) / cap) * cap + Float(1e-5)
            return [lower, upper]
        }
        return MLXHardwareInfo.isCompiledDecodeSupported
            ? compile(shapeless: true, body) : body
    }()

    // MARK: - Metal Kernels

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private static let splitKernel = MLXFast.metalKernel(
        name: "gemma4_hi4lo4_split",
        inputNames: ["w8"],
        outputNames: ["hi", "lo"],
        source: """
            const uint col = thread_position_in_grid.x;
            const uint rows = uint(w8_shape[0]);
            const uint cols4 = uint(w8_shape[1]) >> 1;
            if (col >= cols4) { return; }
            for (uint row = thread_position_in_grid.y; row < rows; row += threads_per_grid.y) {
                const size_t out_idx = size_t(row) * size_t(cols4) + size_t(col);
                const size_t in_idx = out_idx * 2;
                const uint w0 = w8[in_idx];
                const uint w1 = w8[in_idx + 1];
                const uint lo0 = (w0 & 0x0000000Fu)
                               | ((w0 & 0x00000F00u) >> 4)
                               | ((w0 & 0x000F0000u) >> 8)
                               | ((w0 & 0x0F000000u) >> 12);
                const uint lo1 = (w1 & 0x0000000Fu)
                               | ((w1 & 0x00000F00u) >> 4)
                               | ((w1 & 0x000F0000u) >> 8)
                               | ((w1 & 0x0F000000u) >> 12);
                const uint h0 = w0 >> 4;
                const uint h1 = w1 >> 4;
                const uint hi0 = (h0 & 0x0000000Fu)
                               | ((h0 & 0x00000F00u) >> 4)
                               | ((h0 & 0x000F0000u) >> 8)
                               | ((h0 & 0x0F000000u) >> 12);
                const uint hi1 = (h1 & 0x0000000Fu)
                               | ((h1 & 0x00000F00u) >> 4)
                               | ((h1 & 0x000F0000u) >> 8)
                               | ((h1 & 0x0F000000u) >> 12);
                hi[out_idx] = hi0 | (hi1 << 16);
                lo[out_idx] = lo0 | (lo1 << 16);
            }
            """,
        ensureRowContiguous: true
    )

    private static let interleaveKernel = MLXFast.metalKernel(
        name: "gemma4_hi4lo4_interleave",
        inputNames: ["hi", "lo"],
        outputNames: ["w8"],
        source: """
            const uint col = thread_position_in_grid.x;
            const uint rows = uint(hi_shape[0]);
            const uint cols4 = uint(hi_shape[1]);
            if (col >= cols4) { return; }
            for (uint row = thread_position_in_grid.y; row < rows; row += threads_per_grid.y) {
                const size_t in_idx = size_t(row) * size_t(cols4) + size_t(col);
                const size_t out_idx = in_idx * 2;
                const uint h = hi[in_idx];
                const uint l = lo[in_idx];
                const uint w0 = ((h & 0x000Fu) << 4)  | (l & 0x000Fu)
                              | ((h & 0x00F0u) << 8)  | ((l & 0x00F0u) << 4)
                              | ((h & 0x0F00u) << 12) | ((l & 0x0F00u) << 8)
                              | ((h & 0xF000u) << 16) | ((l & 0xF000u) << 12);
                const uint hhi = h >> 16;
                const uint lhi = l >> 16;
                const uint w1 = ((hhi & 0x000Fu) << 4)  | (lhi & 0x000Fu)
                              | ((hhi & 0x00F0u) << 8)  | ((lhi & 0x00F0u) << 4)
                              | ((hhi & 0x0F00u) << 12) | ((lhi & 0x0F00u) << 8)
                              | ((hhi & 0xF000u) << 16) | ((lhi & 0xF000u) << 12);
                w8[out_idx] = w0;
                w8[out_idx + 1] = w1;
            }
            """,
        ensureRowContiguous: true
    )

    private static let gatherInterleaveKernel = MLXFast.metalKernel(
        name: "gemma4_hi4lo4_gather_interleave",
        inputNames: ["hi_weight", "lo_weight", "indices"],
        outputNames: ["w8"],
        source: """
            const uint col = thread_position_in_grid.x;
            const int vocab = int(hi_weight_shape[0]);
            const uint cols4 = uint(hi_weight_shape[1]);
            const uint rows = uint(indices_shape[0]);
            if (col >= cols4) { return; }
            for (uint row = thread_position_in_grid.y; row < rows; row += threads_per_grid.y) {
                int tok = indices[row];
                if (tok < 0) { tok += vocab; }
                const size_t src_idx = size_t(tok) * size_t(cols4) + size_t(col);
                const size_t dst_idx = (size_t(row) * size_t(cols4) + size_t(col)) * 2;
                const uint h = hi_weight[src_idx];
                const uint l = lo_weight[src_idx];
                const uint w0 = ((h & 0x000Fu) << 4)  | (l & 0x000Fu)
                              | ((h & 0x00F0u) << 8)  | ((l & 0x00F0u) << 4)
                              | ((h & 0x0F00u) << 12) | ((l & 0x0F00u) << 8)
                              | ((h & 0xF000u) << 16) | ((l & 0xF000u) << 12);
                const uint hhi = h >> 16;
                const uint lhi = l >> 16;
                const uint w1 = ((hhi & 0x000Fu) << 4)  | (lhi & 0x000Fu)
                              | ((hhi & 0x00F0u) << 8)  | ((lhi & 0x00F0u) << 4)
                              | ((hhi & 0x0F00u) << 12) | ((lhi & 0x0F00u) << 8)
                              | ((hhi & 0xF000u) << 16) | ((lhi & 0xF000u) << 12);
                w8[dst_idx] = w0;
                w8[dst_idx + 1] = w1;
            }
            """,
        ensureRowContiguous: true
    )

    private static let compactCandidatesKernel = MLXFast.metalKernel(
        name: "gemma4_hi4_compact_candidates",
        inputNames: ["upper", "tau"],
        outputNames: ["cand_ids", "cand_counts", "overflow"],
        source: """
            constexpr uint THREADS = 512u;
            constexpr uint WARPS = THREADS / 32u;

            const uint tid = thread_position_in_threadgroup.x;
            const uint b = threadgroup_position_in_grid.y;
            const uint B = uint(upper_shape[0]);
            const uint V = uint(upper_shape[1]);
            if (b >= B) { return; }

            const uint lane = tid & 31u;
            const uint warp = tid >> 5;

            threadgroup uint warp_totals[WARPS];
            threadgroup uint warp_offsets[WARPS];
            threadgroup uint tile_total_tg;

            const float threshold = tau[b];
            const size_t row_base = size_t(b) * size_t(V);
            const size_t cand_base = size_t(b) * size_t(K_CAP);

            uint written = 0u;
            uint ovf = 0u;
            const bool valid_tau = !isnan(threshold) && (threshold != INFINITY);

            if (valid_tau) {
                for (uint base = 0u; base < V; base += THREADS) {
                    const uint idx = base + tid;
                    const bool active = (idx < V) && (upper[row_base + idx] >= threshold);
                    const uint flag = active ? 1u : 0u;
                    const uint lane_off = metal::simd_prefix_exclusive_sum(flag);
                    const uint warp_sum = metal::simd_sum(flag);
                    if (lane == 0u) {
                        warp_totals[warp] = warp_sum;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (warp == 0u) {
                        const uint wt = lane < WARPS ? warp_totals[lane] : 0u;
                        const uint wo = metal::simd_prefix_exclusive_sum(wt);
                        if (lane < WARPS) {
                            warp_offsets[lane] = wo;
                        }
                        if (lane == WARPS - 1u) {
                            tile_total_tg = wo + wt;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    const uint tile_total = tile_total_tg;
                    if (tile_total > 0u) {
                        if (active) {
                            const uint pos = written + warp_offsets[warp] + lane_off;
                            if (pos < uint(K_CAP)) {
                                cand_ids[cand_base + pos] = int(idx);
                            }
                        }
                        if (written + tile_total > uint(K_CAP)) {
                            ovf = 1u;
                            written = uint(K_CAP);
                            break;
                        }
                        written += tile_total;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            for (uint i = written + tid; i < uint(K_CAP); i += THREADS) {
                cand_ids[cand_base + i] = 0;
            }
            if (tid == 0u) {
                cand_counts[b] = int(written);
                overflow[b] = static_cast<uint8_t>(ovf);
            }
            """,
        header: """
            #include <metal_stdlib>
            #include <metal_simdgroup>
            using namespace metal;
            """,
        ensureRowContiguous: true
    )

    private static func fastChunkTopK(
        scores: MLXArray,
        numChunks: Int,
        emitPerChunk: Int,
        checkInf: Bool
    ) -> (candIDs: MLXArray, nonFinite: MLXArray) {
        let b = scores.dim(0)
        let flags = MLXArray([Int32(checkInf ? 1 : 0), Int32(numChunks)])
        let out = fastChunkTopKKernel(
            [scores, flags],
            template: [
                ("T", scores.dtype),
                ("EMIT", emitPerChunk),
            ],
            grid: (256, numChunks, b),
            threadGroup: (256, 1, 1),
            outputShapes: [[b, numChunks * emitPerChunk], [b, numChunks]],
            outputDTypes: [.int32, .uint8])
        let nonFinite = MLX.any(out[1] .!= UInt8(0), axis: -1)
        return (out[0], nonFinite)
    }

    private static let fastChunkTopKKernel = MLXFast.metalKernel(
        name: "gemma4_hi4_fast_chunk_topk",
        inputNames: ["scores", "flags"],
        outputNames: ["cand_ids", "chunk_nonfinite"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint chunk = threadgroup_position_in_grid.y;
            const uint b = threadgroup_position_in_grid.z;
            const uint B = uint(scores_shape[0]);
            const uint V = uint(scores_shape[1]);
            const uint check_inf = uint(flags[0]);
            const uint C = uint(flags[1]);
            if (b >= B || chunk >= C) { return; }

            const uint lane = tid & 31u;
            const uint sg = tid >> 5;
            const size_t row_base = size_t(b) * size_t(V);

            float tv[EMIT];
            int ti[EMIT];
            for (int r = 0; r < EMIT; ++r) {
                tv[r] = -INFINITY;
                ti[r] = -1;
            }
            uint bad = 0u;

            for (uint block = 0u; block < V; block += C * 256u) {
                const uint v = block + (sg * C + chunk) * 32u + lane;
                if (v < V) {
                    const float val = static_cast<float>(scores[row_base + v]);
                    if (isnan(val) || (check_inf != 0u && !isfinite(val))) {
                        bad = 1u;
                    } else if (ti[EMIT - 1] < 0 || val > tv[EMIT - 1]) {
                        int pos = EMIT - 1;
                        while (pos > 0 && (ti[pos - 1] < 0 || val > tv[pos - 1])) {
                            tv[pos] = tv[pos - 1];
                            ti[pos] = ti[pos - 1];
                            pos -= 1;
                        }
                        tv[pos] = val;
                        ti[pos] = int(v);
                    }
                }
            }

            threadgroup float tg_v[8];
            threadgroup int tg_i[8];
            threadgroup uint tg_bad[8];
            threadgroup int winner_idx;

            uint sg_bad = metal::simd_max(bad);
            if (lane == 0u) { tg_bad[sg] = sg_bad; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint any_bad = 0u;
                for (uint s = 0u; s < 8u; ++s) { any_bad |= tg_bad[s]; }
                chunk_nonfinite[size_t(b) * size_t(C) + size_t(chunk)] =
                    static_cast<uint8_t>(any_bad);
            }

            int cursor = 0;
            for (int r = 0; r < EMIT; ++r) {
                float cur_v = (cursor < EMIT) ? tv[cursor] : -INFINITY;
                int cur_i = (cursor < EMIT) ? ti[cursor] : -1;
                float sm = metal::simd_max(cur_v);
                uint match_idx = (cur_v == sm && cur_i >= 0) ? uint(cur_i) : 0xFFFFFFFFu;
                uint best_lane_idx = metal::simd_min(match_idx);
                if (lane == 0u) {
                    tg_v[sg] = sm;
                    tg_i[sg] = (best_lane_idx == 0xFFFFFFFFu) ? -1 : int(best_lane_idx);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float bv = -INFINITY;
                    int bi = -1;
                    for (uint s = 0u; s < 8u; ++s) {
                        if (tg_i[s] >= 0
                            && (tg_v[s] > bv || (tg_v[s] == bv && (bi < 0 || tg_i[s] < bi))))
                        {
                            bv = tg_v[s];
                            bi = tg_i[s];
                        }
                    }
                    winner_idx = bi;
                    cand_ids[size_t(b) * size_t(C * uint(EMIT)) + size_t(chunk * uint(EMIT) + uint(r))] =
                        (bi >= 0) ? bi : 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (cur_i == winner_idx && winner_idx >= 0) {
                    cursor += 1;
                }
            }
            """,
        header: """
            #include <metal_stdlib>
            #include <metal_simdgroup>
            using namespace metal;
            """,
        ensureRowContiguous: true
    )

    private static let conditionalDenseLogitsKernel = MLXFast.metalKernel(
        name: "gemma4_hi4lo4_conditional_dense_logits",
        inputNames: [
            "x", "hi_weight", "lo_weight", "scales", "biases", "needs_fallback", "softcap_arr",
        ],
        outputNames: ["logits"],
        source: """
            const uint b = thread_position_in_grid.z;
            const uint B = uint(x_shape[0]);
            if (b >= B) { return; }
            if (needs_fallback[b] == 0) { return; }

            const uint V = uint(hi_weight_shape[0]);
            const uint H = uint(x_shape[1]);
            const uint G = H / 64u;
            const uint cols4 = H / 8u;
            const uint lane = thread_position_in_grid.x;
            const float cap = softcap_arr[0];

            const size_t x_base = size_t(b) * size_t(H);
            const size_t out_base = size_t(b) * size_t(V);

            for (uint v = thread_position_in_grid.y; v < V; v += threads_per_grid.y) {
                const size_t w_row_base = size_t(v) * size_t(cols4);
                const size_t g_row_base = size_t(v) * size_t(G);

                float acc = 0.0f;
                for (uint g = lane; g < G; g += 32u) {
                    const float s = static_cast<float>(scales[g_row_base + g]);
                    const float bi = static_cast<float>(biases[g_row_base + g]);
                    float g_dot = 0.0f;
                    float g_sum = 0.0f;
                    const uint w_base = g * 8u;
                    const uint k_group = g * 64u;
                    #pragma unroll
                    for (uint w = 0u; w < 8u; ++w) {
                        const uint hw = hi_weight[w_row_base + w_base + w];
                        const uint lw = lo_weight[w_row_base + w_base + w];
                        const uint k_base = k_group + w * 8u;
                        #pragma unroll
                        for (uint n = 0u; n < 8u; ++n) {
                            const uint q = (((hw >> (4u * n)) & 0xFu) << 4)
                                         | ((lw >> (4u * n)) & 0xFu);
                            const float xv = static_cast<float>(x[x_base + k_base + n]);
                            g_dot += xv * static_cast<float>(q);
                            g_sum += xv;
                        }
                    }
                    acc += s * g_dot + bi * g_sum;
                }
                acc = metal::simd_sum(acc);
                if (lane == 0u) {
                    float val = static_cast<float>(static_cast<TO>(acc));
                    if (cap > 0.0f) {
                        val = metal::tanh(val / cap) * cap;
                    }
                    logits[out_base + v] = val;
                }
            }
            """,
        header: """
            #include <metal_stdlib>
            #include <metal_simdgroup>
            using namespace metal;
            """,
        ensureRowContiguous: true
    )

    private static let conditionalSampleHeader = """
        #include <metal_stdlib>
        #include <metal_simdgroup>
        using namespace metal;

        inline uint gemma4_ordered_float_key(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7FFFFFFFu) == 0u) {
                bits = 0u;
            }
            return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        }

        inline void gemma4_tg_atomic_add_float(threadgroup atomic_uint* addr, float delta) {
            uint expected = atomic_load_explicit(addr, memory_order_relaxed);
            while (true) {
                const float next_val = as_type<float>(expected) + delta;
                const uint desired = as_type<uint>(next_val);
                if (desired == expected) {
                    break;
                }
                if (atomic_compare_exchange_weak_explicit(
                        addr, &expected, desired,
                        memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        }

        inline void gemma4_argmax_reduce(
            float val,
            int idx,
            threadgroup float* tg_vals,
            threadgroup int* tg_idxs,
            uint lane,
            uint warp
        ) {
            #pragma unroll
            for (ushort offset = 16u; offset > 0u; offset >>= 1) {
                const float other_val = metal::simd_shuffle_down(val, offset);
                const int other_idx = metal::simd_shuffle_down(idx, offset);
                if (other_val > val || (other_val == val && other_idx < idx)) {
                    val = other_val;
                    idx = other_idx;
                }
            }
            if (lane == 0u) {
                tg_vals[warp] = val;
                tg_idxs[warp] = idx;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (warp == 0u) {
                float w_val = lane < 8u ? tg_vals[lane] : -INFINITY;
                int w_idx = lane < 8u ? tg_idxs[lane] : 0x7FFFFFFF;
                #pragma unroll
                for (ushort offset = 4u; offset > 0u; offset >>= 1) {
                    const float other_val = metal::simd_shuffle_down(w_val, offset);
                    const int other_idx = metal::simd_shuffle_down(w_idx, offset);
                    if (other_val > w_val || (other_val == w_val && other_idx < w_idx)) {
                        w_val = other_val;
                        w_idx = other_idx;
                    }
                }
                if (lane == 0u) {
                    tg_vals[0] = w_val;
                    tg_idxs[0] = w_idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        inline float gemma4_sum_reduce_f32(
            float val,
            threadgroup float* partials,
            uint lane,
            uint warp
        ) {
            const float w_sum = metal::simd_sum(val);
            if (lane == 0u) {
                partials[warp] = w_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (warp == 0u) {
                const float v = lane < 8u ? partials[lane] : 0.0f;
                const float total = metal::simd_sum(v);
                if (lane == 0u) {
                    partials[0] = total;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float result = partials[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return result;
        }

        inline uint gemma4_sum_reduce_u32(
            uint val,
            threadgroup uint* partials,
            uint lane,
            uint warp
        ) {
            const uint w_sum = metal::simd_sum(val);
            if (lane == 0u) {
                partials[warp] = w_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (warp == 0u) {
                const uint v = lane < 8u ? partials[lane] : 0u;
                const uint total = metal::simd_sum(v);
                if (lane == 0u) {
                    partials[0] = total;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint result = partials[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return result;
        }

        inline uint gemma4_exclusive_prefix_sum_u32(
            uint val,
            threadgroup uint* partials,
            uint lane,
            uint warp
        ) {
            const uint lane_pre = metal::simd_prefix_exclusive_sum(val);
            const uint warp_sum = metal::simd_sum(val);
            if (lane == 0u) {
                partials[warp] = warp_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (warp == 0u) {
                const uint wv = lane < 8u ? partials[lane] : 0u;
                const uint wp = metal::simd_prefix_exclusive_sum(wv);
                if (lane < 8u) {
                    partials[lane] = wp;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint result = partials[warp] + lane_pre;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return result;
        }
        """

    private static let conditionalSampleKernel = MLXFast.metalKernel(
        name: "gemma4_hi4lo4_conditional_sample",
        inputNames: [
            "logits", "needs_fallback", "sparse_tokens", "sparse_non_finite",
            "temps", "top_k", "top_p", "min_p", "exp_noise",
        ],
        outputNames: ["out_tokens", "out_non_finite"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint b = threadgroup_position_in_grid.y;
            const uint B = uint(logits_shape[0]);
            const uint V = uint(logits_shape[1]);
            if (b >= B) { return; }

            if (needs_fallback[b] == 0) {
                if (tid == 0u) {
                    out_tokens[b] = sparse_tokens[b];
                    out_non_finite[b] = sparse_non_finite[b];
                }
                return;
            }

            const uint lane = tid & 31u;
            const uint warp = tid >> 5;

            threadgroup atomic_uint tg_hist[256];
            threadgroup float tg_partials_f[8];
            threadgroup uint tg_partials_u[8];
            threadgroup float tg_vals[8];
            threadgroup int tg_idxs[8];
            threadgroup uint tg_state_u[2];
            threadgroup float tg_state_f[1];

            const uint seg = (V + 255u) / 256u;
            const uint start = tid * seg;
            const uint stop = metal::min(start + seg, V);
            const size_t row_base = size_t(b) * size_t(V);

            uint local_nf = 0u;
            float local_max = -INFINITY;
            int local_idx = 0x7FFFFFFF;
            for (uint v = start; v < stop; ++v) {
                const float val = logits[row_base + v];
                if (isnan(val) || isinf(val)) {
                    local_nf = 1u;
                } else if (val > local_max || (val == local_max && int(v) < local_idx)) {
                    local_max = val;
                    local_idx = int(v);
                }
            }

            const uint total_nf = gemma4_sum_reduce_u32(local_nf, tg_partials_u, lane, warp);
            if (total_nf > 0u) {
                if (tid == 0u) {
                    out_tokens[b] = 0;
                    out_non_finite[b] = 1;
                }
                return;
            }

            gemma4_argmax_reduce(local_max, local_idx, tg_vals, tg_idxs, lane, warp);
            const float row_max_raw = tg_vals[0];
            const int greedy_idx = tg_idxs[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float temp = temps[b];
            const int k_limit = top_k[b];
            const float p_limit = top_p[b];
            const float min_p_limit = min_p[b];

            if (MODE == 0 || temp < 1e-5f || (MODE == 2 && (k_limit <= 1 || p_limit <= 0.0f))) {
                if (tid == 0u) {
                    out_tokens[b] = greedy_idx;
                    out_non_finite[b] = 0;
                }
                return;
            }

            const float max_s = row_max_raw / temp;
            float local_z = 0.0f;
            for (uint v = start; v < stop; ++v) {
                local_z += metal::exp(logits[row_base + v] / temp - max_s);
            }
            const float z_full = gemma4_sum_reduce_f32(local_z, tg_partials_f, lane, warp);

            const bool no_filters =
                (MODE == 1) || (k_limit >= int(V) && p_limit >= 1.0f && min_p_limit <= 0.0f);
            if (no_filters) {
                float best_score = -INFINITY;
                int best_tok = 0x7FFFFFFF;
                for (uint v = start; v < stop; ++v) {
                    const float prob = metal::exp(logits[row_base + v] / temp - max_s) / z_full;
                    const float score = prob / exp_noise[row_base + v];
                    if (score > best_score || (score == best_score && int(v) < best_tok)) {
                        best_score = score;
                        best_tok = int(v);
                    }
                }
                gemma4_argmax_reduce(best_score, best_tok, tg_vals, tg_idxs, lane, warp);
                if (tid == 0u) {
                    out_tokens[b] = tg_idxs[0];
                    out_non_finite[b] = 0;
                }
                return;
            }

            // Exact top-K 4-pass radix selection (ascending token-ID tie-break).
            const bool use_top_k = (k_limit < int(V));
            uint k_thresh = 0u;
            uint k_tie_budget = 0u;
            if (use_top_k) {
                uint prefix = 0u;
                uint mask = 0u;
                uint greater_total = 0u;
                for (int shift = 24; shift >= 0; shift -= 8) {
                    atomic_store_explicit(&tg_hist[tid], 0u, memory_order_relaxed);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint v = start; v < stop; ++v) {
                        const uint key = gemma4_ordered_float_key(logits[row_base + v] / temp);
                        if (shift == 24 || (key & mask) == prefix) {
                            atomic_fetch_add_explicit(
                                &tg_hist[(key >> uint(shift)) & 255u], 1u, memory_order_relaxed);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid == 0u) {
                        uint g = greater_total;
                        uint sel = 0u;
                        for (int byte = 255; byte >= 0; --byte) {
                            const uint cnt = atomic_load_explicit(
                                &tg_hist[byte], memory_order_relaxed);
                            if (g + cnt >= uint(k_limit)) {
                                sel = uint(byte);
                                break;
                            }
                            g += cnt;
                        }
                        tg_state_u[0] = sel;
                        tg_state_u[1] = g;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    prefix |= tg_state_u[0] << uint(shift);
                    mask |= 0xFFu << uint(shift);
                    greater_total = tg_state_u[1];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                k_thresh = prefix;
                k_tie_budget = uint(k_limit) - greater_total;
            }

            // Exact top-P 4-pass probability-weighted radix selection.
            const bool use_top_p = (p_limit < 1.0f);
            bool p_all_pass = !use_top_p;
            uint p_thresh = 0u;
            uint p_tie_budget = 0u;
            if (use_top_p) {
                uint prefix = 0u;
                uint mask = 0u;
                float mass_greater = 0.0f;
                bool early_bucket_bottom = false;
                for (int shift = 24; shift >= 0; shift -= 8) {
                    atomic_store_explicit(&tg_hist[tid], 0u, memory_order_relaxed);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint v = start; v < stop; ++v) {
                        const float sv = logits[row_base + v] / temp;
                        const uint key = gemma4_ordered_float_key(sv);
                        if (shift == 24 || (key & mask) == prefix) {
                            const float p = metal::exp(sv - max_s) / z_full;
                            if (p > 0.0f) {
                                gemma4_tg_atomic_add_float(
                                    &tg_hist[(key >> uint(shift)) & 255u], p);
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid == 0u) {
                        float m = mass_greater;
                        uint sel = 0u;
                        uint found = 0u;
                        for (int byte = 255; byte >= 0; --byte) {
                            const float bm = as_type<float>(
                                atomic_load_explicit(&tg_hist[byte], memory_order_relaxed));
                            if (m + bm > p_limit) {
                                sel = uint(byte);
                                found = 1u;
                                break;
                            }
                            m += bm;
                        }
                        tg_state_u[0] = sel;
                        tg_state_u[1] = found;
                        tg_state_f[0] = m;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    const uint found = tg_state_u[1];
                    if (found == 0u) {
                        if (shift == 24) {
                            p_all_pass = true;
                        } else {
                            early_bucket_bottom = true;
                            p_thresh = prefix;
                            p_tie_budget = V;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        break;
                    }
                    prefix |= tg_state_u[0] << uint(shift);
                    mask |= 0xFFu << uint(shift);
                    mass_greater = tg_state_f[0];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (!p_all_pass && !early_bucket_bottom) {
                    p_thresh = prefix;
                    const uint raw_bits =
                        (p_thresh & 0x80000000u) ? (p_thresh & 0x7FFFFFFFu) : ~p_thresh;
                    const float s_eq = as_type<float>(raw_bits);
                    const float p_eq = metal::exp(s_eq - max_s) / z_full;
                    if (p_eq > 0.0f && p_limit >= mass_greater) {
                        p_tie_budget =
                            uint(metal::min(float(V), metal::floor((p_limit - mass_greater) / p_eq)))
                            + 1u;
                    } else {
                        p_tie_budget = 1u;
                    }
                }
            }

            uint local_k_ties = 0u;
            uint local_p_ties = 0u;
            if (use_top_k || !p_all_pass) {
                for (uint v = start; v < stop; ++v) {
                    const uint key = gemma4_ordered_float_key(logits[row_base + v] / temp);
                    if (use_top_k && key == k_thresh) { local_k_ties += 1u; }
                    if (!p_all_pass && key == p_thresh) { local_p_ties += 1u; }
                }
            }
            const uint pre_k_ties = gemma4_exclusive_prefix_sum_u32(
                local_k_ties, tg_partials_u, lane, warp);
            const uint pre_p_ties = gemma4_exclusive_prefix_sum_u32(
                local_p_ties, tg_partials_u, lane, warp);

            const float min_p_thresh = min_p_limit * (1.0f / z_full);
            float local_z_kept = 0.0f;
            uint k_seen = pre_k_ties;
            uint p_seen = pre_p_ties;
            for (uint v = start; v < stop; ++v) {
                const float sv = logits[row_base + v] / temp;
                const uint key = gemma4_ordered_float_key(sv);
                const float p_full = metal::exp(sv - max_s) / z_full;
                bool keep = (p_full >= min_p_thresh);
                if (use_top_k) {
                    if (key < k_thresh) {
                        keep = false;
                    } else if (key == k_thresh) {
                        if (k_seen >= k_tie_budget) { keep = false; }
                        k_seen += 1u;
                    }
                }
                if (!p_all_pass) {
                    if (key < p_thresh) {
                        keep = false;
                    } else if (key == p_thresh) {
                        if (p_seen >= p_tie_budget) { keep = false; }
                        p_seen += 1u;
                    }
                }
                if (int(v) == greedy_idx) { keep = true; }
                if (keep) {
                    local_z_kept += metal::exp(sv - max_s);
                }
            }
            const float z_kept = gemma4_sum_reduce_f32(local_z_kept, tg_partials_f, lane, warp);

            float best_score = -INFINITY;
            int best_tok = 0x7FFFFFFF;
            k_seen = pre_k_ties;
            p_seen = pre_p_ties;
            for (uint v = start; v < stop; ++v) {
                const float sv = logits[row_base + v] / temp;
                const uint key = gemma4_ordered_float_key(sv);
                const float p_full = metal::exp(sv - max_s) / z_full;
                bool keep = (p_full >= min_p_thresh);
                if (use_top_k) {
                    if (key < k_thresh) {
                        keep = false;
                    } else if (key == k_thresh) {
                        if (k_seen >= k_tie_budget) { keep = false; }
                        k_seen += 1u;
                    }
                }
                if (!p_all_pass) {
                    if (key < p_thresh) {
                        keep = false;
                    } else if (key == p_thresh) {
                        if (p_seen >= p_tie_budget) { keep = false; }
                        p_seen += 1u;
                    }
                }
                if (int(v) == greedy_idx) { keep = true; }
                if (keep) {
                    const float prob = metal::exp(sv - max_s) / z_kept;
                    const float score = prob / exp_noise[row_base + v];
                    if (score > best_score || (score == best_score && int(v) < best_tok)) {
                        best_score = score;
                        best_tok = int(v);
                    }
                }
            }
            gemma4_argmax_reduce(best_score, best_tok, tg_vals, tg_idxs, lane, warp);
            if (tid == 0u) {
                out_tokens[b] = tg_idxs[0];
                out_non_finite[b] = 0;
            }
            """,
        header: conditionalSampleHeader,
        ensureRowContiguous: true
    )
    #endif
}
