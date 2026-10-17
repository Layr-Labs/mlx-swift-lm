// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import MLX
import MLXLMCommon

/// Explicit GLM5Next arithmetic choices, fixed once per target: at
/// `GLM5NextArtifactLoader(executionPolicy:)` or `GLM5NextTextModel(executionPolicy:)`.
/// Every layer keeps its target's value, the MTP layer must match it, and no
/// forward entry point accepts a per-call override. Every member defaults to the
/// unchanged reference graph. Nothing here reads an environment flag, no choice
/// adds resident bytes or per-token decode weight reads, and a native choice
/// that cannot run on the current device or stream throws instead of falling
/// back.
public struct GLM5NextExecutionPolicy: Sendable, Hashable {
    public enum SparseAttention: Sendable, Hashable {
        /// Expand every selected latent through kv_b for every query.
        case reference
        /// Absorbed MLA directly on the checkpoint's kv_b: score = (q W_k)·c and
        /// value = W_v Σ p c, over only the non-padding selection columns. Q4
        /// weights stay packed; nothing is dequantized or copied at load.
        /// Mathematically equivalent to `.reference`, not bit-identical: the
        /// reference rounds expanded K/V to the activation dtype, this path
        /// rounds q W_k and Σ p c instead.
        case absorbed
        /// `absorbed`, except that every query at a raw position below
        /// index_topk (whose budget covers every pool it can see) attends over
        /// the first index_topk latents, masked in-kernel to exactly its
        /// selected set (`GLM5NextDenseSelectionKernel`), with no selection
        /// indices or per-query gathered latents; later queries keep the
        /// selector. Same membership and rounding points; the softmax and
        /// products sum over the masked keys, so not bitwise.
        case absorbedDenseHistory
    }

    public enum SparseDecode: Sendable, Hashable {
        /// The `sparseSelection` graphs and the `sparseAttention` member.
        case reference
        /// Steps whose sparse (selector) queries number at most
        /// `GLM5NextSparseDecodeKernel.maximumQueries` (decode and MTP
        /// verification) select through `GLM5NextSparseAttention
        /// .selectDecodeTokens` (in-kernel pooling and scoring, the
        /// `sparseSelection` selector, in-kernel expansion), and contiguous
        /// histories attend with the split-K kernel reading latents through
        /// the indices (`GLM5NextSparseDecodeKernel.attend`); paged reads keep
        /// the absorbed gathers. Same equations and membership rules; FP32
        /// score order differs (a near-tie pool can change) and split-K keeps
        /// FP32 probabilities where absorbed rounds them, so not bitwise.
        /// Requires an absorbed `sparseAttention`; default GPU stream only.
        case nativeKernels
    }

    public enum SparsePrefill: Sendable, Hashable {
        /// The `sparseAttention` member.
        case reference
        /// Steps with more than `GLM5NextSparseDecodeKernel.maximumQueries`
        /// sparse (selector) queries over a contiguous history attend with
        /// `GLM5NextSparsePrefillKernel.attend`, reading each selected latent
        /// through the indices instead of gathering per-query copies; a paged
        /// row does so over its one latent-history read
        /// (`GLM5NextPagedPayload.latentHistory`), and a paged row below that
        /// read's threshold keeps the absorbed gathers. Same equations; the online softmax
        /// keeps FP32-accurate probabilities where absorbed rounds them to the
        /// activation dtype, and sums in its own order, so not bitwise. Such
        /// steps' blocked selection (`sparseSelection.queryBlocks`) also
        /// scores pools with `GLM5NextPrefillIndexScoreKernel` where it runs
        /// (M5 tensor units, 16-bit index projections): same equations and
        /// mask, its own head-sum order, so a near-tie pool can change.
        /// Requires an absorbed `sparseAttention`; default GPU stream only.
        case nativeKernel
    }

    public enum HyperConnection: Sendable, Hashable {
        /// Unfused RMS, FP32 projection and Sinkhorn chain (~150 launches per site).
        case reference
        /// Four fused Metal launches per site on the default GPU stream only.
        /// FP32 reductions use a fixed tree that differs from MLX's; not bitwise.
        case nativeRequired
        /// `nativeRequired`, except each collapse and the RMSNorm consuming it
        /// are two launches, and the attention expand is computed inside the
        /// FFN collapse's projection launch (`GLM5NextHyperConnectionKernel
        /// .collapseNormalized`). The mHC RMS factor scales each projected sum
        /// rather than every value; not bitwise. Default GPU stream only.
        case nativeNormFused
    }

    public enum RoutedExperts: Sendable, Hashable {
        case reference
        /// The reference's unsorted per-assignment projections and clamped GLU,
        /// then one ascending-expert-ID weighted combine launch in place of the
        /// reference's sort, gathers, casts and K additions. Bitwise identical
        /// to `.reference`.
        case nativeCombine
        /// `nativeCombine`, plus an expert-sorted gathered matmul for prefill
        /// (sequence length > 1, at least `sortedPrefillMinimumAssignments`, and
        /// expert widths that are multiples of 64). The tiled matmul changes
        /// accumulation order; not bitwise.
        case nativeCombineSortedPrefill
        /// `nativeCombineSortedPrefill`, except that steps of at most
        /// `GLM5NextMoEDecodeKernel.maximumTokens` tokens (decode and MTP
        /// verification) run the whole MoE in three launches: router logits
        /// with sigmoid and bias; gate/up with in-kernel top-K selection, the
        /// shared expert as an extra slot and the clamped GLU; down with the
        /// ascending-expert-ID weighted combine and the shared down projection.
        /// Those steps ignore `routing`. Same rounding points as `nativeCombine`;
        /// the router's FP32 dot products and the projections' lane order
        /// differ, so not bitwise, and a near-tie expert can change. Layers the
        /// kernel cannot read (grouped routing, banks that are neither affine
        /// 4/8-bit nor NVFP4, widths that are not whole 512-value blocks) keep
        /// `nativeCombineSortedPrefill`. Default GPU stream only.
        case fusedDecode
        /// `fusedDecode`, except that steps the sorted gather would serve run
        /// `GLM5NextMoEPrefillKernel` on Neural-Accelerator GPUs: single-expert
        /// sorted tiles, joint gate/up reading token rows through the sort
        /// order with the clamped GLU in its epilogue, down on the same tiles,
        /// and a scatter-inverted combine. Same rounding points; tensor-op K
        /// order and dequantized-weight rounding differ, so not bitwise. Banks
        /// it cannot read (neither affine 4-bit nor NVFP4, widths not multiples
        /// of 64) and GPUs without Neural Accelerators keep the sorted gather.
        case fusedDecodeNAXPrefill
    }

    public enum LinearAttention: Sendable, Hashable {
        /// Preprocess and recur one token at a time in MLX operations.
        case reference
        /// Preprocess the whole step once, then run `GLM5NextLinearKernel` for
        /// every step length and batch: decode, MTP verification and prefill.
        /// Same equations, FP32 state and rounding boundaries; the kernel's FP32
        /// reduction order differs, so not bitwise. Default GPU stream only.
        case nativeRecurrence
        /// `nativeRecurrence` for steps shorter than `linearAttentionChunkSize`;
        /// longer prefill steps use `GLM5NextLinearChunkedRecurrence`.
        /// Reassociated matrix products; not bitwise, and FP32 GEMMs may run at
        /// TF32 precision on Neural-Accelerator GPUs (MLX's default). Requires
        /// a configured gate lower bound that admits the chunk size.
        case nativeRecurrenceChunkedPrefill
    }

    public enum StepEvaluation: Sendable, Hashable {
        /// The caller evaluates the whole step graph.
        case deferred
        /// Steps of at most `earlyEvaluationMaximumTokens` rows start GPU work
        /// (`asyncEval`) every `earlyEvaluationLayerStride` layers while the
        /// rest of the graph is still being built. Same graph and results.
        case earlyAsync
    }

    /// Row and layer bounds of `StepEvaluation.earlyAsync`.
    public static let earlyEvaluationMaximumTokens = 8
    public static let earlyEvaluationLayerStride = 8

    public enum Projections: Sendable, Hashable {
        case separate
        /// Bias-free projections of one input (KDA q/k/v/f_a/g_a/b, DSA
        /// q_a/kv_a/indexer wk/weights_proj, MLP gate/up) stored as one
        /// row-concatenated buffer whose row views are the checkpoint modules,
        /// run as one matmul (`GLM5NextFusedProjection`). Same bytes and
        /// equations; matmul kernels pick their reduction split by output
        /// width, so not bitwise.
        case fusedInputs
    }

    public enum PrefillProjections: Sendable, Hashable {
        /// MLX's quantized matmul for every step.
        case mlx
        /// Affine 4-bit projections of steps with at least
        /// `GLM5NextPrefillGEMMKernel.minimumRows` rows run
        /// `GLM5NextPrefillGEMMKernel` on Neural-Accelerator GPUs (larger
        /// double-buffered tiles in a swizzled order); other banks, smaller
        /// steps and GPUs without Neural Accelerators keep MLX. Weights are
        /// dequantized exactly as MLX's NAX qmm does and accumulated in FP32
        /// in the same K order, so results matched MLX bitwise at every
        /// measured shape; only the FP32 grouping could differ. Default GPU
        /// stream only.
        case nativeGEMM
    }

    public enum Routing: Sendable, Hashable {
        case reference
        /// `GLM5NextRouterKernel` for steps of at most
        /// `GLM5NextRouterKernel.maximumTokens` rows: two launches reading the
        /// stored router weight in place instead of an FP32 copy per call. Same
        /// equations; FP32 dot-product and weight-sum order differ, so not
        /// bitwise. Requires ungrouped routing; default GPU stream only.
        case nativeKernel
    }

    public enum LinearDecodeStep: Sendable, Hashable {
        /// The load-time `linearAttention` member's composed operations.
        case composed
        /// One `GLM5NextLinearDecodeKernel` launch per KDA layer for every
        /// step of at most `GLM5NextLinearDecodeKernel.maximumTokens` tokens
        /// (decode and MTP verification); longer steps keep `linearAttention`.
        /// Same equations and rounding boundaries, fixed reduction tree; not
        /// bitwise. Requires a native `linearAttention` member and a configured
        /// gate lower bound; default GPU stream only.
        case fusedKernel
        /// `fusedKernel` for every step length, prefill included. Longer steps
        /// run three launches: a per-channel causal-convolution walk, the
        /// recurrence split across value rows with its state in registers, and
        /// the gated output norm. Same equations and rounding boundaries.
        case fusedEveryStep
    }

    public enum Normalization: Sendable, Hashable {
        /// FP32 cast, MLX mean of squares, rsqrt, cast, weight multiply.
        case reference
        /// MLX's fused `rms_norm` kernel for the target's RMSNorms. Same
        /// rounding point; its own sum order, so not bitwise.
        case fusedKernel
    }

    /// Chunk size of `LinearAttention.nativeRecurrenceChunkedPrefill`.
    /// With GLM-5.3's gate lower bound of -5, 32 is the largest admitted size.
    public static let linearAttentionChunkSize = 32

    /// Mirrors SwitchGLU's sorted-gather threshold.
    public static let sortedPrefillMinimumAssignments = 64

    public var sparseAttention: SparseAttention
    /// Pool top-k kernel and position ramps of every DSA selection. Exact: each
    /// value selects the reference token indices.
    public var sparseSelection: GLM5NextSparseAttention.Execution
    /// Paged metadata addressing of backends built for this target. Exact: both
    /// forms copy identical bytes.
    public var pagedMetadataReads: GLM5NextPagedMetadataReads
    public var hyperConnection: HyperConnection
    public var routedExperts: RoutedExperts
    public var linearAttention: LinearAttention
    public var normalization: Normalization
    public var linearDecodeStep: LinearDecodeStep
    public var routing: Routing
    public var projections: Projections
    public var stepEvaluation: StepEvaluation
    public var sparseDecode: SparseDecode
    public var sparsePrefill: SparsePrefill
    public var prefillProjections: PrefillProjections

    public init(
        sparseAttention: SparseAttention = .reference,
        sparseSelection: GLM5NextSparseAttention.Execution = .reference,
        pagedMetadataReads: GLM5NextPagedMetadataReads = .hostRecords,
        hyperConnection: HyperConnection = .reference,
        routedExperts: RoutedExperts = .reference,
        linearAttention: LinearAttention = .reference,
        normalization: Normalization = .reference,
        linearDecodeStep: LinearDecodeStep = .composed,
        routing: Routing = .reference,
        projections: Projections = .separate,
        stepEvaluation: StepEvaluation = .deferred,
        sparseDecode: SparseDecode = .reference,
        sparsePrefill: SparsePrefill = .reference,
        prefillProjections: PrefillProjections = .mlx
    ) {
        self.sparseAttention = sparseAttention
        self.sparseSelection = sparseSelection
        self.pagedMetadataReads = pagedMetadataReads
        self.hyperConnection = hyperConnection
        self.routedExperts = routedExperts
        self.linearAttention = linearAttention
        self.normalization = normalization
        self.linearDecodeStep = linearDecodeStep
        self.routing = routing
        self.projections = projections
        self.stepEvaluation = stepEvaluation
        self.sparseDecode = sparseDecode
        self.sparsePrefill = sparsePrefill
        self.prefillProjections = prefillProjections
    }

    public static let reference = GLM5NextExecutionPolicy()

    /// The serving choice: every member qualified against the official oracle
    /// and the real Q4 checkpoint. The reference KDA and selection graphs grow
    /// per token and are not servable at real prompt lengths.
    ///
    /// Qualified on the real Q4 checkpoint: coherent greedy output at 15.8 ms
    /// per decode step (one-token fused MoE decode is bitwise to the
    /// `nativeCombineSortedPrefill` combine), perplexity within noise of
    /// `.reference` (+0.11%, 95% CI [-0.04%, +0.16%] on a 13.8K-token mixed
    /// panel at 512-token chunks, 98.6% top-1 agreement; long context at
    /// 8-token steps over 6,144-token documents, mean NLL 0.72412 vs 0.72413),
    /// and the official-oracle suites. On the NVFP4 hybrid, expert prefill
    /// runs `fusedDecodeNAXPrefill`: 1,955 versus 1,585 tok/s at 2,048-token
    /// chunks with unchanged decode, bitwise to the sorted gather at 512 tokens.
    /// Prefill chunks past position 2,048 run `SparsePrefill.nativeKernel`:
    /// 8,192 tokens at 2,048-token chunks 2,000 versus 1,340 tok/s, perplexity
    /// +0.018% (95% CI [-0.057%, +0.097%] NLL, 99.2% top-1) on 6,144-token
    /// documents at 2,048-token chunks.
    public static let serving = GLM5NextExecutionPolicy(
        sparseAttention: .absorbedDenseHistory,
        sparseSelection: .init(
            selection: .parallelThreshold, deviceRanges: true, queryBlocks: true),
        pagedMetadataReads: .deviceTable, hyperConnection: .nativeNormFused,
        routedExperts: .fusedDecodeNAXPrefill,
        linearAttention: .nativeRecurrenceChunkedPrefill, normalization: .fusedKernel,
        linearDecodeStep: .fusedEveryStep, routing: .nativeKernel, projections: .fusedInputs,
        sparseDecode: .nativeKernels, sparsePrefill: .nativeKernel)
}

/// Native GLM kernels run only on the default GPU stream. Callers throw their
/// own failure when this is false; there is no CPU or foreign-stream fallback.
package func glm5NextOnDefaultGPUStream() -> Bool {
    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        return Device.defaultDevice().deviceType == .gpu
            && StreamOrDevice.default.stream == Stream.gpu
    #else
        return false
    #endif
}

/// Whether this GPU has Neural Accelerators (MLX's NAX kernels' condition:
/// OS, architecture and build). Evaluated once per process.
let glm5NextNAXAvailable: Bool = {
    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // MLX exposes the device NAX probe only through this diagnostics record.
        return GPU.gemma4ExpertQMMDiagnostics().naxAvailable
    #else
        return false
    #endif
}()
