// Gemma4BlockSDCM.swift
//
// Block-Atomic Causal-Adjoint Multigrid (Block-SDCM) prefill for Gemma 4 26B.
//
// Exploits Gemma 4's 5:1 sliding-to-full attention topology (full attention at
// layers 5, 11, 17, 23, 29; sliding window W=1024 elsewhere):
//  - Runs 100% of tokens through early layers (L0..L11 in lightcone/balanced
//    modes so even top-of-prompt instructions complete their 2-full-layer
//    causal lightcone prefix -> tail(L5) -> target(L11) before any compaction).
//  - At full-attention boundaries, scores document tokens with head-balanced
//    causal boundary attention + block-diverse 2-hop bidirectional adjoint.
//  - Selects atomic contiguous 64-token blocks with neighbor-block halos,
//    mandatory prefix/tail retention, and 4-strata coverage so functions,
//    clauses, and multi-token values are never torn mid-span.
//  - Preserves exact original-position RoPE phases and sliding-window distances
//    for every kept token, and commits compacted KV with `logicalDelta` so
//    subsequent decode steps compute RoPE at the exact original prompt length.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for Gemma 4 Block-Atomic Causal-Adjoint Multigrid prefill.
public struct Gemma4BlockSDCMConfig: Sendable, Equatable {
    public enum Mode: String, Sendable, Equatable, CaseIterable {
        /// Disabled: 100% exact unmodified prefill.
        case off
        /// Causal-lightcone-exact L12-start compaction + interior top-5 MoE on L0..L11.
        /// Highest quality across top-instruction bug hunting, 2/3-hop chains, JSON, and summaries.
        case lightcone
        /// Causal-lightcone-exact L12-start compaction + interior top-4 MoE across trunk.
        /// Higher throughput (~2.0x) while preserving the full L0..L11 causal lightcone.
        case balanced
        /// Aggressive L6-start block-atomic compaction with exact top-8 MoE on kept blocks.
        /// Highest single-turn throughput (~2.3-2.5x) on bottom-query long-context prompts.
        case fast
    }

    public static let envKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM"
    public static let minTokensEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_MIN_TOKENS"
    public static let earlyMoEKEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_EARLY_K"

    public var mode: Mode
    /// Prompts shorter than this bypass Block-SDCM and run exact unmodified prefill.
    public var minPromptTokens: Int
    /// Map of `layerIdx -> target fraction of original prompt length` to compact to before `layerIdx`.
    public var keepSchedule: [Int: Float]
    /// Active MoE experts for interior tokens (`pos < origT - tailKeep`) on layers `< earlyMoEUntilLayer`.
    public var earlyMoEK: Int
    /// Active MoE experts for interior tokens on layers `>= earlyMoEUntilLayer` (excluding final layer).
    public var lateMoEK: Int
    /// Layer boundary separating `earlyMoEK` and `lateMoEK`.
    public var earlyMoEUntilLayer: Int
    /// Contiguous atomic block width in tokens (never split mid-block).
    public var blockSize: Int
    /// Mandatory prefix tokens kept intact at every compaction step (preserves BOS + top instructions).
    public var prefixKeep: Int
    /// Mandatory tail tokens kept intact and routed at exact top-8 MoE (preserves bottom questions).
    public var tailKeep: Int
    /// Halo weight propagated from a hot block `b` to `b + 1` (preserves function bodies / values).
    public var forwardHalo: Float
    /// Halo weight propagated from a hot block `b` to `b - 1` (preserves function headers / labels).
    public var backwardHalo: Float
    /// Number of document macro-strata that each guarantee at least one top block (global summary coverage).
    public var strataCount: Int
    /// Number of distinct block peaks used for bidirectional 2-hop causal adjoint scoring.
    public var twoHopBlockPeaks: Int
    /// Query chunk width for banded O(L * W) original-position sliding-window SDPA.
    public var slidingChunkSize: Int

    public init(
        mode: Mode,
        minPromptTokens: Int = 1536,
        keepSchedule: [Int: Float],
        earlyMoEK: Int = 8,
        lateMoEK: Int = 8,
        earlyMoEUntilLayer: Int = 12,
        blockSize: Int = 64,
        prefixKeep: Int = 128,
        tailKeep: Int = 192,
        forwardHalo: Float = 0.85,
        backwardHalo: Float = 0.85,
        strataCount: Int = 4,
        twoHopBlockPeaks: Int = 24,
        slidingChunkSize: Int = 1024
    ) {
        self.mode = mode
        self.minPromptTokens = max(1, minPromptTokens)
        self.keepSchedule = keepSchedule
        self.earlyMoEK = max(1, min(8, earlyMoEK))
        self.lateMoEK = max(1, min(8, lateMoEK))
        self.earlyMoEUntilLayer = earlyMoEUntilLayer
        self.blockSize = max(1, blockSize)
        self.prefixKeep = max(0, prefixKeep)
        self.tailKeep = max(1, tailKeep)
        self.forwardHalo = forwardHalo
        self.backwardHalo = backwardHalo
        self.strataCount = max(0, strataCount)
        self.twoHopBlockPeaks = max(0, twoHopBlockPeaks)
        self.slidingChunkSize = max(64, slidingChunkSize)
    }

    public static let off = Gemma4BlockSDCMConfig(mode: .off, keepSchedule: [:])

    public static let lightcone = Gemma4BlockSDCMConfig(
        mode: .lightcone,
        minPromptTokens: 1536,
        keepSchedule: [12: 0.40, 18: 0.26, 24: 0.18],
        earlyMoEK: 5,
        lateMoEK: 8,
        earlyMoEUntilLayer: 12,
        blockSize: 64,
        prefixKeep: 128,
        tailKeep: 192,
        forwardHalo: 0.85,
        backwardHalo: 0.85,
        strataCount: 4,
        twoHopBlockPeaks: 24,
        slidingChunkSize: 1024
    )

    public static let balanced = Gemma4BlockSDCMConfig(
        mode: .balanced,
        minPromptTokens: 1536,
        keepSchedule: [12: 0.36, 18: 0.22, 24: 0.16],
        earlyMoEK: 4,
        lateMoEK: 4,
        earlyMoEUntilLayer: 12,
        blockSize: 64,
        prefixKeep: 128,
        tailKeep: 192,
        forwardHalo: 0.85,
        backwardHalo: 0.85,
        strataCount: 4,
        twoHopBlockPeaks: 24,
        slidingChunkSize: 1024
    )

    public static let fast = Gemma4BlockSDCMConfig(
        mode: .fast,
        minPromptTokens: 1536,
        keepSchedule: [6: 0.45, 12: 0.25, 18: 0.18, 24: 0.14],
        earlyMoEK: 8,
        lateMoEK: 8,
        earlyMoEUntilLayer: 12,
        blockSize: 64,
        prefixKeep: 128,
        tailKeep: 192,
        forwardHalo: 0.85,
        backwardHalo: 0.50,
        strataCount: 4,
        twoHopBlockPeaks: 24,
        slidingChunkSize: 1024
    )

    public static func preset(for mode: Mode) -> Gemma4BlockSDCMConfig {
        switch mode {
        case .off: return .off
        case .lightcone: return .lightcone
        case .balanced: return .balanced
        case .fast: return .fast
        }
    }

    public static func parseMode(_ raw: String?) -> Mode {
        guard let raw else { return .off }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "", "0", "false", "no", "off", "disabled":
            return .off
        case "1", "true", "yes", "on", "lightcone", "l12":
            return .lightcone
        case "balanced", "bal":
            return .balanced
        case "fast", "aggressive", "l6":
            return .fast
        default:
            return .off
        }
    }

    /// Resolve configuration from environment variables (injectable dictionary for unit tests).
    public static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Gemma4BlockSDCMConfig {
        let mode = parseMode(environment[envKey])
        guard mode != .off else { return .off }
        var config = preset(for: mode)
        if let rawMin = environment[minTokensEnvKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            let value = Int(rawMin), value > 0
        {
            config.minPromptTokens = value
        }
        if let rawK = environment[earlyMoEKEnvKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            let value = Int(rawK), (1 ... 8).contains(value)
        {
            config.earlyMoEK = value
        }
        return config
    }

    /// Current process-level configuration (resolved once from environment).
    public static let current: Gemma4BlockSDCMConfig = resolve()
}

// MARK: - Lazy Multi-Turn Checkpoint

/// Lightweight post-full-layer hidden-state checkpoint (`[1, T, D]`, ~50 MB at T=8825)
/// enabling demand-driven multi-turn wake-up of previously pruned prompt spans.
public struct Gemma4SDCMCheckpoint {
    public let hiddenStates: MLXArray
    public let positions: [Int32]
    public let checkpointLayer: Int
    public let originalLength: Int

    public init(
        hiddenStates: MLXArray,
        positions: [Int32],
        checkpointLayer: Int,
        originalLength: Int
    ) {
        self.hiddenStates = hiddenStates
        self.positions = positions
        self.checkpointLayer = checkpointLayer
        self.originalLength = originalLength
    }

    /// Identify newly relevant indices into `positions` / `hiddenStates` for a follow-up turn's
    /// queries `queryQ` (`[1, nHeads, Lq, D]`) attending to the full checkpoint keys
    /// `keyK` (`[1, nKvHeads, T, D]`), excluding positions already present in upper-layer KV caches.
    public func selectWakeUpIndices(
        queryQ: MLXArray,
        keyK: MLXArray,
        scale: Float,
        nHeads: Int,
        nKvHeads: Int,
        alreadyKeptPositions: Set<Int32>,
        wakeFraction: Float = 0.18,
        blockSize: Int = 64,
        prefixSkip: Int = 16
    ) -> [Int32] {
        guard !positions.isEmpty, originalLength > 0, queryQ.dim(2) > 0,
            blockSize > 0, wakeFraction > 0
        else { return [] }
        let kRep = Gemma4BlockSDCM.repeatKVHeads(keyK, nHeads: nHeads, nKvHeads: nKvHeads)
        let scores =
            matmul(queryQ.asType(.float32), kRep.asType(.float32).transposed(0, 1, 3, 2)) * scale
        let probs = softmax(scores, axis: -1)[0]  // [nHeads, Lq, T]
        let docMask2D = (MLXArray(positions) .>= Int32(prefixSkip)).expandedDimensions(axis: 0)
        let combined = Gemma4BlockSDCM.headBalancedDocumentSaliency(
            probs: probs, docMask2D: docMask2D)
        eval(combined)
        let scHost = combined.asArray(Float.self)

        let nBlocks = (originalLength + blockSize - 1) / blockSize
        let aggregated = Gemma4BlockSDCM.aggregateBlockScores(
            scores: scHost, positions: positions, blockSize: blockSize, nBlocks: nBlocks)
        guard !aggregated.aliveBlocks.isEmpty else { return [] }

        // Symmetric block halo across alive blocks so woken spans remain intact.
        let halo = Gemma4BlockSDCM.applyBlockHalo(
            scores: aggregated.scores,
            aliveBlocks: aggregated.aliveBlocks,
            blockAlive: aggregated.alive,
            forwardHalo: 0.85,
            backwardHalo: 0.85)
        let targetBlocks = max(1, Int((Float(nBlocks) * wakeFraction).rounded()))
        let ranked = aggregated.aliveBlocks.sorted { lhs, rhs in
            if halo[lhs] != halo[rhs] { return halo[lhs] > halo[rhs] }
            return lhs < rhs
        }
        let chosenBlocks = Set(ranked.prefix(targetBlocks))

        var wakeIndices: [Int32] = []
        wakeIndices.reserveCapacity(min(positions.count, targetBlocks * blockSize))
        for (i, pos) in positions.enumerated() {
            let b = Int(pos) / blockSize
            if chosenBlocks.contains(b) && !alreadyKeptPositions.contains(pos) {
                wakeIndices.append(Int32(i))
            }
        }
        return wakeIndices
    }
}

// MARK: - Block-SDCM Core Math

/// Stateless mathematical kernels for Block-Atomic Causal-Adjoint Multigrid prefill.
public enum Gemma4BlockSDCM {

    /// True when a Gemma 4 text topology supports Block-SDCM (non-KV-shared,
    /// non-PLE, non-fully-bidirectional MoE, e.g. Gemma 4 26B-A4B).
    public static func supportsModelConfig(_ modelConfig: Gemma4TextConfiguration) -> Bool {
        modelConfig.numKvSharedLayers == 0
            && modelConfig.hiddenSizePerLayerInput == 0
            && modelConfig.useBidirectionalAttention != "all"
            && modelConfig.enableMoeBlock
    }

    /// True when a cache instance supports writing compacted prompt K/V with a
    /// logical RoPE position delta without corrupting physical cursors.
    static func supportsCache(_ cache: KVCache) -> Bool {
        if cache is any CBv2AttendingLayerCache || cache is KVCacheSimple {
            return true
        }
        // `CompilableRotatingKVCache` inherits from `RotatingKVCache` but reads
        // `offsetArray` instead of `offset`, so it cannot accept host offset deltas.
        return cache is RotatingKVCache && !(cache is CompilableRotatingKVCache)
    }

    /// Check whether a Gemma 4 prefill call qualifies for Block-SDCM.
    /// Restricts execution to solo cold text prefills on non-KV-shared, non-PLE
    /// Gemma 4 topologies (e.g. 26B-A4B) above `sdcm.minPromptTokens`.
    public static func canRun(
        sdcm: Gemma4BlockSDCMConfig,
        modelConfig: Gemma4TextConfiguration,
        schedulePrefill: Bool,
        batchSize: Int,
        sequenceLength: Int,
        hasInputEmbedding: Bool,
        hasImageMask: Bool,
        hasCaptureHook: Bool,
        initialOffset: Int
    ) -> Bool {
        sdcm.mode != .off
            && schedulePrefill
            && batchSize == 1
            && sequenceLength >= sdcm.minPromptTokens
            && sequenceLength > sdcm.prefixKeep + sdcm.tailKeep
            && !hasInputEmbedding
            && !hasImageMask
            && !hasCaptureHook
            && initialOffset == 0
            && supportsModelConfig(modelConfig)
            && !sdcm.keepSchedule.isEmpty
    }

    /// Expand GQA key heads to match query head count.
    fileprivate static func repeatKVHeads(
        _ keys: MLXArray, nHeads: Int, nKvHeads: Int
    ) -> MLXArray {
        nHeads == nKvHeads ? keys : MLX.repeated(keys, count: nHeads / nKvHeads, axis: 1)
    }

    /// Reduce `[H, Q, L]` attention probabilities into a head-balanced `[L]` document
    /// saliency vector: per-head max over queries, zero non-doc positions, normalize
    /// each head to `[0, 1]`, and combine `max + 0.25 * mean` across heads.
    fileprivate static func headBalancedDocumentSaliency(
        probs: MLXArray, docMask2D: MLXArray
    ) -> MLXArray {
        let hMax = probs.max(axis: 1)  // [H, L]
        let hDoc = MLX.where(docMask2D, hMax, MLXArray(Float(0)))
        let hBal = hDoc / (hDoc.max(axis: -1, keepDims: true) + Float(1e-8))
        return hBal.max(axis: 0) + Float(0.25) * hBal.mean(axis: 0)
    }

    /// Aggregate per-token saliency scores into per-block maxima and track which blocks
    /// still contain at least one active token.
    fileprivate static func aggregateBlockScores(
        scores: [Float], positions: [Int32], blockSize: Int, nBlocks: Int
    ) -> (scores: [Float], alive: [Bool], aliveBlocks: [Int]) {
        var blockScore = [Float](repeating: -1e9, count: nBlocks)
        var blockAlive = [Bool](repeating: false, count: nBlocks)
        var aliveBlocks: [Int] = []
        aliveBlocks.reserveCapacity(nBlocks)
        for i in 0 ..< positions.count {
            let b = Int(positions[i]) / blockSize
            guard b >= 0 && b < nBlocks else { continue }
            if !blockAlive[b] {
                blockAlive[b] = true
                aliveBlocks.append(b)
            }
            if scores[i] > blockScore[b] {
                blockScore[b] = scores[i]
            }
        }
        return (blockScore, blockAlive, aliveBlocks)
    }

    /// Propagate neighbor-block halos across alive adjacent blocks so multi-token
    /// spans straddling a block boundary stay intact.
    fileprivate static func applyBlockHalo(
        scores: [Float],
        aliveBlocks: [Int],
        blockAlive: [Bool],
        forwardHalo: Float,
        backwardHalo: Float
    ) -> [Float] {
        let nBlocks = scores.count
        var halo = scores
        for b in aliveBlocks {
            let s = scores[b]
            if b + 1 < nBlocks && blockAlive[b + 1] {
                halo[b + 1] = max(halo[b + 1], forwardHalo * s)
            }
            if b - 1 >= 0 && blockAlive[b - 1] {
                halo[b - 1] = max(halo[b - 1], backwardHalo * s)
            }
        }
        return halo
    }

    /// Select up to `twoHopBlockPeaks` distinct document block peak indices, breaking
    /// score ties deterministically by ascending token index.
    private static func selectTwoHopPeakIndices(
        scores: [Float],
        positionsHost: [Int32],
        originalLength: Int,
        config: Gemma4BlockSDCMConfig
    ) -> [Int32] {
        let L = positionsHost.count
        var bestPerBlock: [Int: (score: Float, index: Int32)] = [:]
        bestPerBlock.reserveCapacity(L / config.blockSize + 1)
        let minDocPos = Int32(config.prefixKeep)
        let maxDocPos = Int32(originalLength - config.tailKeep)
        for i in 0 ..< L {
            let p = positionsHost[i]
            guard p >= minDocPos && p < maxDocPos else { continue }
            let b = Int(p) / config.blockSize
            let v = scores[i]
            if let prev = bestPerBlock[b] {
                if v > prev.score { bestPerBlock[b] = (v, Int32(i)) }
            } else {
                bestPerBlock[b] = (v, Int32(i))
            }
        }
        let rankedPeaks = bestPerBlock.values
            .sorted { lhs, rhs in
                if lhs.score != rhs.score { return lhs.score > rhs.score }
                return lhs.index < rhs.index
            }
            .prefix(config.twoHopBlockPeaks)
        return rankedPeaks.map(\.index).sorted()
    }

    /// Apply exact RoPE at arbitrary non-contiguous integer positions `positions` (`[L]` int32)
    /// to a `[1, H, L, D]` tensor using MLX's fused batched-offset `MLXFast.RoPE` kernel.
    public static func applyArbitraryPositionsRoPE<R: RoPELayer>(
        _ rope: R,
        to x: MLXArray,
        positions: MLXArray
    ) -> MLXArray {
        precondition(x.ndim == 4 && x.dim(0) == 1, "Block-SDCM RoPE expects [1, H, L, D]")
        precondition(
            positions.ndim == 1 && positions.dim(0) == x.dim(2),
            "Block-SDCM RoPE positions length mismatch")
        // Transpose [1, H, L, D] -> [L, H, 1, D] so axis 0 aligns with batched `offset: [L]`.
        let transposed = x.transposed(2, 1, 0, 3)
        let rotated = gemma4ApplyRotaryPosition(rope, to: transposed, offset: .batch(positions))
        return rotated.transposed(2, 1, 0, 3)
    }

    /// Compute head-balanced causal-boundary + block-diverse 2-hop bidirectional adjoint
    /// saliency scores for all `L` currently active tokens at a full-attention boundary.
    public static func computeBoundaryScores(
        queries: MLXArray,
        keys: MLXArray,
        positions: MLXArray,
        positionsHost: [Int32],
        originalLength: Int,
        scale: Float,
        nHeads: Int,
        nKvHeads: Int,
        config: Gemma4BlockSDCMConfig
    ) -> [Float] {
        let L = queries.dim(2)
        guard L > 0 else { return [] }
        precondition(
            positionsHost.count == L && positions.dim(0) == L,
            "Block-SDCM boundary scoring positions length mismatch")
        let kRep = repeatKVHeads(keys, nHeads: nHeads, nKvHeads: nKvHeads)
        let kRepF32T = kRep.asType(.float32).transposed(0, 1, 3, 2)  // [1, H, D, L]
        let posRow = positions.expandedDimensions(axis: 0)  // [1, L]
        let docMask =
            (positions .>= Int32(config.prefixKeep))
            .&& (positions .< Int32(originalLength - config.tailKeep))
        let docMask2D = docMask.expandedDimensions(axis: 0)  // [1, L]
        let negInf = MLXArray(-Float(1e9))
        let zero = MLXArray(Float(0))
        let eps = Float(1e-8)

        func headBalanced(qSub: MLXArray, pSub: MLXArray) -> MLXArray {
            let sc = matmul(qSub.asType(.float32), kRepF32T) * scale  // [1, H, Q, L]
            let causalInvalid = posRow .> pSub  // [Q, L]
            let masked = MLX.where(
                causalInvalid.expandedDimensions(axes: [0, 1]), negInf, sc)
            let probs = softmax(masked, axis: -1)[0]  // [H, Q, L]
            return headBalancedDocumentSaliency(probs: probs, docMask2D: docMask2D)
        }

        // 1. Causal output boundary (last 16 tokens) + tail query window (last 96 tokens).
        let endCount = min(16, L)
        let sEnd = headBalanced(
            qSub: queries[0..., 0..., (L - endCount)..., 0...],
            pSub: positions[(L - endCount)...].expandedDimensions(axis: 1))
        let tailCount = min(96, L)
        var score: MLXArray
        if tailCount > endCount {
            let sTail = headBalanced(
                qSub: queries[0..., 0..., (L - tailCount)..., 0...],
                pSub: positions[(L - tailCount)...].expandedDimensions(axis: 1))
            score = maximum(sEnd, Float(0.70) * sTail)
        } else {
            score = sEnd
        }

        // 2. Block-diverse 2-hop bidirectional causal adjoint (captures A -> B chains
        //    regardless of whether B precedes or follows A in the prompt).
        if config.twoHopBlockPeaks > 0,
            L > config.prefixKeep + config.tailKeep + config.blockSize
        {
            eval(score)
            let peakIndices = selectTwoHopPeakIndices(
                scores: score.asArray(Float.self),
                positionsHost: positionsHost,
                originalLength: originalLength,
                config: config)
            if !peakIndices.isEmpty {
                let hIdx = MLXArray(peakIndices)
                let qPeaks = queries.take(hIdx, axis: 2)
                let pPeaks = positions.take(hIdx, axis: 0)

                // Backward hop: peak tokens attend to earlier keys.
                let sBwd = headBalanced(
                    qSub: qPeaks, pSub: pPeaks.expandedDimensions(axis: 1))

                // Forward hop: later doc tokens attending back to peak keys (z-scored per head).
                let kPeaksF32T = kRep.take(hIdx, axis: 2)
                    .asType(.float32).transposed(0, 1, 3, 2)  // [1, H, D, P]
                let scFwd = matmul(queries.asType(.float32), kPeaksF32T) * scale  // [1, H, L, P]
                let fwdInvalid =
                    positions.expandedDimensions(axis: 1) .<= pPeaks.expandedDimensions(axis: 0)
                let scFwdMasked = MLX.where(
                    fwdInvalid.expandedDimensions(axes: [0, 1]), negInf, scFwd)
                let hf = scFwdMasked[0].max(axis: -1)  // [H, L]
                let valid =
                    (hf .> Float(-1e8)).asType(.float32) * docMask2D.asType(.float32)
                let vCnt = valid.sum(axis: -1, keepDims: true) + eps
                let validPos = valid .> Float(0)
                let mu = MLX.where(validPos, hf, zero).sum(axis: -1, keepDims: true) / vCnt
                let diff = hf - mu
                let variance =
                    MLX.where(validPos, diff * diff, zero).sum(axis: -1, keepDims: true) / vCnt
                let z = maximum(diff / (sqrt(variance) + Float(1e-6)), zero) * valid
                let zBal = z / (z.max(axis: -1, keepDims: true) + eps)
                let sFwd = zBal.max(axis: 0)

                score = maximum(score, maximum(Float(0.80) * sBwd, Float(0.70) * sFwd))
            }
        }

        eval(score)
        return score.asArray(Float.self)
    }

    /// Pure deterministic block-atomic token selection with neighbor-block halos,
    /// mandatory prefix/tail retention, and stratified macro-segment coverage.
    /// Returns strictly ascending indices into `0 ..< positions.count`.
    public static func selectCompactedIndices(
        scores: [Float],
        positions: [Int32],
        originalLength: Int,
        targetFraction: Float,
        config: Gemma4BlockSDCMConfig
    ) -> [Int32] {
        let count = positions.count
        guard count > 0, originalLength > 0 else { return [] }
        precondition(scores.count == count, "Block-SDCM scores/positions length mismatch")
        let blockSize = config.blockSize
        let nBlocks = (originalLength + blockSize - 1) / blockSize

        let aggregated = aggregateBlockScores(
            scores: scores, positions: positions, blockSize: blockSize, nBlocks: nBlocks)
        let blockAlive = aggregated.alive
        let aliveBlockList = aggregated.aliveBlocks

        let minMustBlocks =
            (config.prefixKeep + config.tailKeep + blockSize - 1) / blockSize + config.strataCount
        let targetBlocks = max(
            minMustBlocks, Int((Float(nBlocks) * targetFraction).rounded()))
        if targetBlocks >= aliveBlockList.count {
            return (0 ..< count).map { Int32($0) }
        }

        let haloScore = applyBlockHalo(
            scores: aggregated.scores,
            aliveBlocks: aliveBlockList,
            blockAlive: blockAlive,
            forwardHalo: config.forwardHalo,
            backwardHalo: config.backwardHalo)

        var keptBlocks = Set<Int>()
        keptBlocks.reserveCapacity(targetBlocks)

        // Mandatory prefix blocks (BOS + top-of-prompt instructions).
        let prefixBlockEnd = (config.prefixKeep + blockSize - 1) / blockSize
        for b in 0 ..< min(prefixBlockEnd, nBlocks) where blockAlive[b] {
            keptBlocks.insert(b)
        }
        // Mandatory tail blocks (bottom-of-prompt questions + final frontier token).
        let tailStartBlock = max(0, (originalLength - config.tailKeep) / blockSize)
        for b in tailStartBlock ..< nBlocks where blockAlive[b] {
            keptBlocks.insert(b)
        }

        // Stratified macro-segment coverage (top-1 block per document stratum).
        if config.strataCount > 0 {
            let step = max(1, nBlocks / config.strataCount)
            for s in 0 ..< config.strataCount {
                let lo = s * step
                let hi = s == config.strataCount - 1 ? nBlocks : min(nBlocks, (s + 1) * step)
                guard lo < hi else { continue }
                var bestB = -1
                var bestVal: Float = -Float.infinity
                for b in lo ..< hi where blockAlive[b] && !keptBlocks.contains(b) {
                    if haloScore[b] > bestVal {
                        bestVal = haloScore[b]
                        bestB = b
                    }
                }
                if bestB >= 0 {
                    keptBlocks.insert(bestB)
                }
            }
        }

        // Fill remaining block budget with highest-scoring alive blocks.
        let remaining = max(0, targetBlocks - keptBlocks.count)
        if remaining > 0 {
            let candidates = aliveBlockList
                .filter { !keptBlocks.contains($0) }
                .sorted { lhs, rhs in
                    if haloScore[lhs] != haloScore[rhs] {
                        return haloScore[lhs] > haloScore[rhs]
                    }
                    return lhs < rhs
                }
            for b in candidates.prefix(remaining) {
                keptBlocks.insert(b)
            }
        }

        var selected: [Int32] = []
        selected.reserveCapacity(min(count, keptBlocks.count * blockSize))
        for i in 0 ..< count {
            let b = Int(positions[i]) / blockSize
            if keptBlocks.contains(b) {
                selected.append(Int32(i))
            }
        }
        return selected
    }

    /// Exact O(L * W) chunked banded sliding-window SDPA using original integer token positions.
    /// Guarantees `0 <= pos[q] - pos[k] < slidingWindow` without materializing an `L x L` mask.
    public static func slidingAttentionWithPositions(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        positions: MLXArray,
        positionsHost: [Int32],
        slidingWindow: Int,
        scale: Float,
        chunkSize: Int = 1024
    ) -> MLXArray {
        let L = queries.dim(2)
        guard L > 0 else {
            return MLXArray.zeros(
                [queries.dim(0), queries.dim(1), 0, values.dim(3)], dtype: queries.dtype)
        }
        precondition(slidingWindow > 0, "Block-SDCM slidingWindow must be positive")
        precondition(
            positionsHost.count == L && positions.dim(0) == L,
            "Block-SDCM sliding attention positions length mismatch")
        let effectiveChunkSize = max(1, chunkSize)
        let windowI32 = Int32(slidingWindow)
        if L <= effectiveChunkSize {
            let diff =
                positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)
            let mask = (diff .>= Int32(0)) .&& (diff .< windowI32)
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .array(mask))
        }

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((L + effectiveChunkSize - 1) / effectiveChunkSize)
        let windowSpan = Int64(slidingWindow - 1)
        var s0 = 0
        while s0 < L {
            let s1 = min(L, s0 + effectiveChunkSize)
            let minPos = Int64(positionsHost[s0]) - windowSpan
            // Binary search for earliest kept index within the sliding window of positionsHost[s0].
            var lo = 0
            var hi = s0
            while lo < hi {
                let mid = lo + ((hi - lo) >> 1)
                if Int64(positionsHost[mid]) < minPos {
                    lo = mid + 1
                } else {
                    hi = mid
                }
            }
            let k0 = lo
            let qPos = positions[s0 ..< s1].expandedDimensions(axis: 1)
            let kPos = positions[k0 ..< s1].expandedDimensions(axis: 0)
            let diff = qPos - kPos
            let mask = (diff .>= Int32(0)) .&& (diff .< windowI32)
            let out = MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., s0 ..< s1, 0...],
                keys: keys[0..., 0..., k0 ..< s1, 0...],
                values: values[0..., 0..., k0 ..< s1, 0...],
                scale: scale,
                mask: .array(mask))
            chunks.append(out)
            s0 = s1
        }
        return concatenated(chunks, axis: 2)
    }
}

// MARK: - Gemma4DecoderLayer & Trunk Execution

extension Gemma4DecoderLayer {
    /// Project, Q-norm, transpose, and RoPE queries at arbitrary original positions.
    private func projectQueriesWithRoPE(_ input: MLXArray, positions: MLXArray) -> MLXArray {
        let attn = selfAttn
        let qLength = input.dim(1)
        var q = attn.qProj(input).reshaped(1, qLength, attn.nHeads, attn.effectiveHeadDim)
        q = attn.qNorm(q).transposed(0, 2, 1, 3)
        return Gemma4BlockSDCM.applyArbitraryPositionsRoPE(attn.rope, to: q, positions: positions)
    }

    /// Reshape SDPA output `[1, H, qL, D]`, restore original activation dtype, and project via `oProj`.
    private func finishAttentionOutput(
        _ attnResult: MLXArray, length: Int, outDType: DType
    ) -> MLXArray {
        var attnOut = attnResult.transposed(0, 2, 1, 3).reshaped(1, length, -1)
        if attnOut.dtype != outDType { attnOut = attnOut.asType(outDType) }
        return selfAttn.oProj(attnOut)
    }

    /// Commit compacted K/V to the layer cache with logical RoPE delta. Sliding layers
    /// store only the trailing `min(L, slidingWindow)` tokens so paged rings never overflow.
    private func commitCompactedKV(
        keys k: MLXArray, values v: MLXArray, originalLength: Int, cache: KVCache
    ) {
        let attn = selfAttn
        let L = k.dim(2)
        let kStore: MLXArray
        let vStore: MLXArray
        if attn.isSliding && L > attn.config.slidingWindow {
            let start = L - attn.config.slidingWindow
            kStore = k[0..., 0..., start..., 0...]
            vStore = v[0..., 0..., start..., 0...]
        } else {
            kStore = k
            vStore = v
        }
        let logicalDelta = originalLength - kStore.dim(2)
        if let v2Cache = cache as? (any CBv2AttendingLayerCache) {
            v2Cache.writeCompactedPromptKV(
                keys: kStore, values: vStore, logicalDelta: logicalDelta)
        } else if let simpleCache = cache as? KVCacheSimple {
            _ = simpleCache.update(keys: kStore, values: vStore)
            simpleCache.positionDelta += logicalDelta
        } else if let rotCache = cache as? RotatingKVCache, !(cache is CompilableRotatingKVCache) {
            _ = rotCache.update(keys: kStore, values: vStore)
            rotCache.offset += logicalDelta
        } else {
            preconditionFailure(
                "Block-SDCM requires a cache supporting logical position deltas, got \(type(of: cache))"
            )
        }
    }

    /// Execute one decoder layer inside the Block-SDCM prefill trunk.
    /// Preserves exact original-position RoPE, exact banded sliding-window
    /// distances, dense KV storage with logical `positionDelta`, and optional
    /// interior adaptive top-k MoE (keeping the prompt tail at exact top-8).
    func forwardBlockSDCM(
        _ x: MLXArray,
        positions: MLXArray,
        positionsHost: [Int32],
        originalLength: Int,
        cache: KVCache?,
        sdcm: Gemma4BlockSDCMConfig,
        isFinalLayer: Bool,
        shouldScoreForNextLayer: Bool,
        isExpertPrefill: Bool
    ) -> (hidden: MLXArray, nextLayerScores: [Float]?) {
        let L = x.dim(1)
        let attn = selfAttn
        guard let kProj = attn.kProj, let kNorm = attn.kNorm, let vNorm = attn.vNorm else {
            preconditionFailure("Block-SDCM requires non-shared attention layers")
        }

        let normed = inputLayernorm(x)

        // K and V projections across all L currently active tokens.
        let kRaw = kProj(normed).reshaped(1, L, attn.nKvHeads, attn.effectiveHeadDim)
        var k = kNorm(kRaw).transposed(0, 2, 1, 3)
        k = Gemma4BlockSDCM.applyArbitraryPositionsRoPE(attn.rope, to: k, positions: positions)

        var v: MLXArray
        if let vProj = attn.vProj {
            v = vProj(normed).reshaped(1, L, attn.nKvHeads, attn.effectiveHeadDim)
        } else {
            v = kRaw
        }
        v = vNorm(v).transposed(0, 2, 1, 3)

        if let cache {
            commitCompactedKV(keys: k, values: v, originalLength: originalLength, cache: cache)
        }

        let outDType = k.dtype
        let attnK = outDType == .float16 ? k.asType(.float32) : k
        let attnV = outDType == .float16 ? v.asType(.float32) : v

        // Final-layer last-query specialization: project Q, attend, O-proj, MLP, and exact
        // top-8 MoE for ONLY the final frontier token.
        if isFinalLayer {
            let qLast = projectQueriesWithRoPE(
                normed[0..., (L - 1)..., 0...], positions: positions[(L - 1)...])
            let attnQ = outDType == .float16 ? qLast.asType(.float32) : qLast
            let finalMask: MLXFast.ScaledDotProductAttentionMaskMode
            if attn.isSliding {
                let diff =
                    positions[(L - 1)...].expandedDimensions(axis: 1)
                    - positions.expandedDimensions(axis: 0)
                finalMask = .array(
                    (diff .>= Int32(0)) .&& (diff .< Int32(attn.config.slidingWindow)))
            } else {
                finalMask = .none
            }
            let attnResult = MLXFast.scaledDotProductAttention(
                queries: attnQ, keys: attnK, values: attnV, scale: attn.scale, mask: finalMask)
            let attnOut = finishAttentionOutput(attnResult, length: 1, outDType: outDType)

            var h = x[0..., (L - 1)..., 0...] + postAttentionLayernorm(attnOut)
            h = feedforward(h, isExpertPrefill: false)
            h = h * layerScalar
            return (h, nil)
        }

        // Standard layer Q projection + arbitrary-position RoPE.
        let q = projectQueriesWithRoPE(normed, positions: positions)
        let attnQ = outDType == .float16 ? q.asType(.float32) : q

        let attnResult: MLXArray
        var nextScores: [Float]? = nil
        if attn.isSliding {
            attnResult = Gemma4BlockSDCM.slidingAttentionWithPositions(
                queries: attnQ,
                keys: attnK,
                values: attnV,
                positions: positions,
                positionsHost: positionsHost,
                slidingWindow: attn.config.slidingWindow,
                scale: attn.scale,
                chunkSize: sdcm.slidingChunkSize)
        } else {
            // Kept positions are strictly increasing, so `pos[k] <= pos[q]` is identical
            // to index-causal order -> uses MLX's fused FlashAttention `.causal` kernel.
            attnResult = MLXFast.scaledDotProductAttention(
                queries: attnQ, keys: attnK, values: attnV, scale: attn.scale, mask: .causal)
            if shouldScoreForNextLayer {
                nextScores = Gemma4BlockSDCM.computeBoundaryScores(
                    queries: q,
                    keys: k,
                    positions: positions,
                    positionsHost: positionsHost,
                    originalLength: originalLength,
                    scale: attn.scale,
                    nHeads: attn.nHeads,
                    nKvHeads: attn.nKvHeads,
                    config: sdcm)
            }
        }

        let attnOut = finishAttentionOutput(attnResult, length: L, outDType: outDType)
        let ffIn = x + postAttentionLayernorm(attnOut)
        let interiorMoEK =
            attn.layerIdx < sdcm.earlyMoEUntilLayer ? sdcm.earlyMoEK : sdcm.lateMoEK
        var h = feedforwardBlockSDCM(
            ffIn,
            interiorMoEK: interiorMoEK,
            tailKeep: sdcm.tailKeep,
            isExpertPrefill: isExpertPrefill)
        h = h * layerScalar
        return (h, nextScores)
    }

    /// Parallel dense + MoE feedforward with optional interior adaptive top-k MoE.
    /// Tail tokens (`L - tailKeep ..< L`) ALWAYS execute exact full top-8 MoE.
    private func feedforwardBlockSDCM(
        _ input: MLXArray,
        interiorMoEK: Int,
        tailKeep: Int,
        isExpertPrefill: Bool
    ) -> MLXArray {
        let L = input.dim(1)
        let topK = router?.topK ?? 8
        guard isMoE,
            interiorMoEK < topK,
            L > tailKeep,
            let router,
            let experts,
            let postFeedforwardLayernorm1,
            let preFeedforwardLayernorm2,
            let postFeedforwardLayernorm2
        else {
            return feedforward(input, isExpertPrefill: isExpertPrefill)
        }

        let residual2 = input
        var h1 = preFeedforwardLayernorm(input)
        h1 = mlp(h1)
        h1 = postFeedforwardLayernorm1(h1)

        let cut = L - tailKeep
        let interiorIn = input[0..., ..<cut, 0...]
        let tailIn = input[0..., cut..., 0...]

        // Interior tokens: sort router weights descending and keep top `interiorMoEK`.
        let (intIndices, intWeights) = router(interiorIn)
        let order = argSort(-intWeights, axis: -1)[.ellipsis, ..<interiorMoEK]
        let prunedIndices = takeAlong(intIndices, order, axis: -1)
        let prunedWeights = takeAlong(intWeights, order, axis: -1)
        let h2IntRaw = experts(
            preFeedforwardLayernorm2(interiorIn),
            topKIndices: prunedIndices,
            topKWeights: prunedWeights,
            isExpertPrefill: isExpertPrefill)
        let h2Int = postFeedforwardLayernorm2(h2IntRaw)

        // Tail tokens: exact top-8 MoE (preserves question & frontier representations).
        let (tailIndices, tailWeights) = router(tailIn)
        let h2TailRaw = experts(
            preFeedforwardLayernorm2(tailIn),
            topKIndices: tailIndices,
            topKWeights: tailWeights,
            isExpertPrefill: isExpertPrefill)
        let h2Tail = postFeedforwardLayernorm2(h2TailRaw)

        let h2 = concatenated([h2Int, h2Tail], axis: 1)
        let out = postFeedforwardLayernorm(h1 + h2)
        return residual2 + out
    }
}

extension Gemma4TextModelInner {
    /// Execute the full Gemma 4 trunk using Block-SDCM progressive block compaction.
    /// Returns `[1, 1, hiddenSize]` (the final frontier position after `norm`).
    func forwardBlockSDCMTrunk(
        _ initialHidden: MLXArray,
        fullCache: [KVCache?],
        sdcm: Gemma4BlockSDCMConfig,
        isExpertPrefill: Bool,
        captureCheckpointAtLayer: Int? = nil
    ) -> (postNorm: MLXArray, checkpoint: Gemma4SDCMCheckpoint?) {
        let originalLength = initialHidden.dim(1)
        var h = initialHidden
        var positionsHost = (0 ..< originalLength).map { Int32($0) }
        var positions = MLXArray(positionsHost)
        var pendingScores: [Float]? = nil
        var checkpoint: Gemma4SDCMCheckpoint? = nil

        for (idx, layer) in layers.enumerated() {
            if let targetFraction = sdcm.keepSchedule[idx], let scores = pendingScores {
                let keepIndices = Gemma4BlockSDCM.selectCompactedIndices(
                    scores: scores,
                    positions: positionsHost,
                    originalLength: originalLength,
                    targetFraction: targetFraction,
                    config: sdcm)
                if keepIndices.count < h.dim(1) && !keepIndices.isEmpty {
                    let idxArray = MLXArray(keepIndices)
                    h = h.take(idxArray, axis: 1)
                    positionsHost = keepIndices.map { positionsHost[Int($0)] }
                    positions = MLXArray(positionsHost)
                    asyncEval(h, positions)
                }
                pendingScores = nil
            }

            let isFinalLayer = idx == layers.count - 1
            let shouldScore = sdcm.keepSchedule[idx + 1] != nil
            let (out, nextScores) = layer.forwardBlockSDCM(
                h,
                positions: positions,
                positionsHost: positionsHost,
                originalLength: originalLength,
                cache: fullCache[idx],
                sdcm: sdcm,
                isFinalLayer: isFinalLayer,
                shouldScoreForNextLayer: shouldScore,
                isExpertPrefill: isExpertPrefill)
            h = out
            if let nextScores {
                pendingScores = nextScores
            }
            if let ckptLayer = captureCheckpointAtLayer, idx == ckptLayer {
                checkpoint = Gemma4SDCMCheckpoint(
                    hiddenStates: h,
                    positions: positionsHost,
                    checkpointLayer: idx,
                    originalLength: originalLength)
            }
            // Keep Metal command buffers bounded on long prompts without forcing host syncs
            // except at scoring boundaries where `computeBoundaryScores` already reads `[L]` floats.
            if !isFinalLayer && nextScores == nil && (idx + 1).isMultiple(of: 6) {
                asyncEval(h)
            }
        }

        return (norm(h), checkpoint)
    }
}

