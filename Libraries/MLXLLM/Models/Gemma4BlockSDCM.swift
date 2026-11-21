// Gemma4BlockSDCM.swift
//
// Block-Atomic Causal-Adjoint Multigrid (Block-SDCM v2) prefill for Gemma 4 26B.
//
// Exploits Gemma 4's 5:1 sliding-to-full attention topology (full attention at
// layers 5, 11, 17, 23, 29; sliding window W=1024 elsewhere):
//  - Runs energy-matched depth-tapered interior MoE top-k on early layers
//    (exact softmax renormalization over selected k; tail tokens and late layers
//    stay exact top-8) and O(T) Prefix+Local+FullTail sparse attention on uncompacted L5.
//  - At full-attention compaction boundaries, scores blocks bilaterally using:
//      1. Dual-scale frontier-focused forward causal boundary (`F8` frontier +
//         `F64` tail, zero-copy GQA) + block-diverse 2-hop bidirectional adjoint.
//      2. Quota-partitioned pure log-odds Reverse Causal Prefix Resonance
//         (`doc -> prefix[4..<128]` vs sink `[0..<4)`, head-max z-score), catching
//         top-of-prompt instructions at L5/L11/L17 without starving bottom queries.
//  - Selects atomic contiguous 32/64-token blocks with symmetric neighbor halos,
//    mandatory prefix/tail retention, and 4-strata coverage.
//  - Preserves exact original-position RoPE phases, O(L * W) banded original-position
//    sliding SDPA, and true-window sliding KV commits (`origPos >= T - W + 1`) so
//    compacted decode steps never attend to out-of-window sliding keys.

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
        /// Causal-lightcone L12-start compaction (`40/26/20%`, 64-tok blocks) with
        /// energy-tapered early MoE and bilateral frontier + reverse prefix scoring.
        /// ~2.0x faster with near-exact quality across top-instruction and multi-hop suites.
        case lightcone
        /// Bilateral L6-start compaction (`46/30/24/20%`, 64-tok blocks) with dedicated
        /// Reverse Causal Prefix Resonance quotas and energy-tapered early MoE.
        /// ~2.6-2.8x faster while matching `lightcone` quality across top- and bottom-query tasks.
        case balanced
        /// Bilateral L6-start fine-block compaction (`36/22/16/12%`, 32-tok blocks) with
        /// Reverse Causal Prefix Resonance quotas and energy-tapered early MoE.
        /// ~3.1-3.4x faster with strong long-context recall, summarization, bug-hunt, and 16K accuracy.
        case fast
        /// Early L1 Tri-Signal Pre-Compaction (`55%` after Layer 0 via `L0_emb` ColBERT MaxSim+2-Hop
        /// + `after_L0` `C@C^T` Outlier Novelty + CPU Document-IDF Token Rarity with targeted top-K
        /// neighbor halo) followed by `fast` L6/L12/L18/L24 bilateral compaction and position-aware
        /// sparse L5 attention. ~3.8-4.1x faster (>10,500 tok/s on M3 Ultra) while preserving ~90%
        /// accuracy across 9K-16K code and natural-language prose suites.
        case turbo
    }

    public static let envKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM"
    public static let minTokensEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_MIN_TOKENS"
    public static let earlyMoEKEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_EARLY_K"

    /// Default energy-matched early-layer interior MoE top-k schedule for Gemma 4 26B
    /// (`L2` has the lowest MoE/total FFN energy ratio at 0.19; tail tokens stay exact top-8).
    public static let defaultEarlyLayerMoEK: [Int: Int] = [
        0: 3, 1: 3, 2: 2, 3: 3, 4: 4, 5: 4,
    ]

    public var mode: Mode
    /// Prompts shorter than this bypass Block-SDCM and run exact unmodified prefill.
    public var minPromptTokens: Int
    /// Map of `layerIdx -> target fraction of original prompt length` to compact to before `layerIdx`.
    public var keepSchedule: [Int: Float]
    /// Active MoE experts for interior tokens (`pos < origT - tailKeep`) on layers `< earlyMoEUntilLayer`
    /// when not overridden by `layerMoEK`.
    public var earlyMoEK: Int
    /// Active MoE experts for interior tokens on layers `>= earlyMoEUntilLayer` (excluding final layer).
    public var lateMoEK: Int
    /// Layer boundary separating `earlyMoEK` and `lateMoEK`.
    public var earlyMoEUntilLayer: Int
    /// Optional per-layer interior MoE top-k overrides (e.g. energy-tapered L0..L5 schedule).
    public var layerMoEK: [Int: Int]
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
    /// Number of document macro-strata for global summary coverage.
    public var strataCount: Int
    /// Number of distinct block peaks used for bidirectional 2-hop causal adjoint scoring.
    public var twoHopBlockPeaks: Int
    /// Query chunk width for banded O(L * W) original-position sliding-window SDPA.
    public var slidingChunkSize: Int
    /// Dedicated fraction of interior block budget reserved for Reverse Causal Prefix Resonance
    /// (`doc -> prefix[4..<reversePrefixEnd)` log-odds vs sink) keyed by target compaction layer.
    public var reversePrefixQuota: [Int: Float]
    /// Exclusive upper original-position bound of prefix tokens used for reverse prefix resonance.
    public var reversePrefixEnd: Int
    /// Optional early uncompacted full-attention layer (e.g. layer 5) using O(T) Prefix+Local+FullTail
    /// sparse attention instead of O(T^2) full SDPA.
    public var earlySparseFullLayer: Int?
    /// Prefix tokens attended to by all interior queries in `earlySparseFullLayer`.
    public var sparsePrefixTokens: Int
    /// Local sliding window attended to by interior queries in `earlySparseFullLayer`.
    public var sparseLocalWindow: Int
    /// Trailing tokens receiving exact full-sequence causal attention in `earlySparseFullLayer`.
    public var sparseTailTokens: Int
    /// Optional early sliding-layer index after which to run Tri-Signal Pre-Compaction (e.g. `0`
    /// compacts right after Layer 0 before Layer 1). `nil` disables early pre-compaction.
    public var earlyPreCompactAfterLayer: Int?
    /// Target fraction of original prompt length retained by early Tri-Signal Pre-Compaction.
    public var earlyPreCompactRatio: Float
    /// Atomic block width in tokens used during early Tri-Signal Pre-Compaction.
    public var earlyPreCompactBlockSize: Int
    /// Number of sharpest MaxSim+2-Hop blocks granted an explicit `±1` neighbor halo at early compaction.
    public var earlyPreCompactTopHaloK: Int

    public init(
        mode: Mode,
        minPromptTokens: Int = 1536,
        keepSchedule: [Int: Float],
        earlyMoEK: Int = 8,
        lateMoEK: Int = 8,
        earlyMoEUntilLayer: Int = 12,
        layerMoEK: [Int: Int] = [:],
        blockSize: Int = 64,
        prefixKeep: Int = 128,
        tailKeep: Int = 192,
        forwardHalo: Float = 0.85,
        backwardHalo: Float = 0.85,
        strataCount: Int = 4,
        twoHopBlockPeaks: Int = 24,
        slidingChunkSize: Int = 1024,
        reversePrefixQuota: [Int: Float] = [:],
        reversePrefixEnd: Int = 128,
        earlySparseFullLayer: Int? = 5,
        sparsePrefixTokens: Int = 256,
        sparseLocalWindow: Int = 1024,
        sparseTailTokens: Int = 256,
        earlyPreCompactAfterLayer: Int? = nil,
        earlyPreCompactRatio: Float = 0.55,
        earlyPreCompactBlockSize: Int = 64,
        earlyPreCompactTopHaloK: Int = 8
    ) {
        self.mode = mode
        self.minPromptTokens = max(1, minPromptTokens)
        self.keepSchedule = keepSchedule
        self.earlyMoEK = max(1, min(8, earlyMoEK))
        self.lateMoEK = max(1, min(8, lateMoEK))
        self.earlyMoEUntilLayer = earlyMoEUntilLayer
        self.layerMoEK = layerMoEK.mapValues { max(1, min(8, $0)) }
        self.blockSize = max(1, blockSize)
        self.prefixKeep = max(0, prefixKeep)
        self.tailKeep = max(1, tailKeep)
        self.forwardHalo = forwardHalo
        self.backwardHalo = backwardHalo
        self.strataCount = max(0, strataCount)
        self.twoHopBlockPeaks = max(0, twoHopBlockPeaks)
        self.slidingChunkSize = max(64, slidingChunkSize)
        self.reversePrefixQuota = reversePrefixQuota.mapValues { max(0, min(0.8, $0)) }
        self.reversePrefixEnd = max(8, reversePrefixEnd)
        self.earlySparseFullLayer = earlySparseFullLayer
        self.sparsePrefixTokens = max(16, sparsePrefixTokens)
        self.sparseLocalWindow = max(64, sparseLocalWindow)
        self.sparseTailTokens = max(64, sparseTailTokens)
        self.earlyPreCompactAfterLayer = earlyPreCompactAfterLayer
        self.earlyPreCompactRatio =
            earlyPreCompactRatio.isFinite
            ? max(0.1, min(1.0, earlyPreCompactRatio)) : 1.0
        self.earlyPreCompactBlockSize = max(1, earlyPreCompactBlockSize)
        self.earlyPreCompactTopHaloK = max(0, earlyPreCompactTopHaloK)
    }

    /// Resolve interior MoE top-k for `layerIdx` (tail tokens always run exact full top-k).
    public func interiorMoEK(forLayer layerIdx: Int) -> Int {
        if let override = layerMoEK[layerIdx] {
            return override
        }
        return layerIdx < earlyMoEUntilLayer ? earlyMoEK : lateMoEK
    }

    public static let off = Gemma4BlockSDCMConfig(
        mode: .off, keepSchedule: [:], earlySparseFullLayer: nil)

    public static let lightcone = Gemma4BlockSDCMConfig(
        mode: .lightcone,
        minPromptTokens: 1536,
        keepSchedule: [12: 0.40, 18: 0.26, 24: 0.20],
        earlyMoEK: 5,
        lateMoEK: 8,
        earlyMoEUntilLayer: 12,
        layerMoEK: defaultEarlyLayerMoEK,
        blockSize: 64,
        prefixKeep: 128,
        tailKeep: 192,
        forwardHalo: 0.85,
        backwardHalo: 0.85,
        strataCount: 4,
        twoHopBlockPeaks: 24,
        slidingChunkSize: 1024,
        reversePrefixQuota: [12: 0.25, 18: 0.25, 24: 0.20],
        reversePrefixEnd: 128,
        earlySparseFullLayer: 5,
        sparsePrefixTokens: 256,
        sparseLocalWindow: 1024,
        sparseTailTokens: 256
    )

    public static let balanced = Gemma4BlockSDCMConfig(
        mode: .balanced,
        minPromptTokens: 1536,
        keepSchedule: [6: 0.46, 12: 0.30, 18: 0.24, 24: 0.20],
        earlyMoEK: 6,
        lateMoEK: 8,
        earlyMoEUntilLayer: 12,
        layerMoEK: defaultEarlyLayerMoEK,
        blockSize: 64,
        prefixKeep: 128,
        tailKeep: 192,
        forwardHalo: 0.85,
        backwardHalo: 0.85,
        strataCount: 4,
        twoHopBlockPeaks: 24,
        slidingChunkSize: 1024,
        reversePrefixQuota: [6: 0.30, 12: 0.25, 18: 0.25, 24: 0.20],
        reversePrefixEnd: 128,
        earlySparseFullLayer: 5,
        sparsePrefixTokens: 256,
        sparseLocalWindow: 1024,
        sparseTailTokens: 256
    )

    public static let fast = Gemma4BlockSDCMConfig(
        mode: .fast,
        minPromptTokens: 1536,
        keepSchedule: [6: 0.36, 12: 0.22, 18: 0.16, 24: 0.12],
        earlyMoEK: 6,
        lateMoEK: 8,
        earlyMoEUntilLayer: 12,
        layerMoEK: defaultEarlyLayerMoEK,
        blockSize: 32,
        prefixKeep: 128,
        tailKeep: 192,
        forwardHalo: 0.85,
        backwardHalo: 0.85,
        strataCount: 4,
        twoHopBlockPeaks: 24,
        slidingChunkSize: 1024,
        reversePrefixQuota: [6: 0.32, 12: 0.28, 18: 0.25, 24: 0.20],
        reversePrefixEnd: 128,
        earlySparseFullLayer: 5,
        sparsePrefixTokens: 256,
        sparseLocalWindow: 1024,
        sparseTailTokens: 256
    )

    public static let turbo = Gemma4BlockSDCMConfig(
        mode: .turbo,
        minPromptTokens: 1536,
        keepSchedule: [6: 0.36, 12: 0.22, 18: 0.16, 24: 0.12],
        earlyMoEK: 6,
        lateMoEK: 8,
        earlyMoEUntilLayer: 12,
        layerMoEK: defaultEarlyLayerMoEK,
        blockSize: 32,
        prefixKeep: 128,
        tailKeep: 192,
        forwardHalo: 0.85,
        backwardHalo: 0.85,
        strataCount: 4,
        twoHopBlockPeaks: 24,
        slidingChunkSize: 1024,
        reversePrefixQuota: [6: 0.32, 12: 0.28, 18: 0.25, 24: 0.20],
        reversePrefixEnd: 128,
        earlySparseFullLayer: 5,
        sparsePrefixTokens: 256,
        sparseLocalWindow: 1024,
        sparseTailTokens: 256,
        earlyPreCompactAfterLayer: 0,
        earlyPreCompactRatio: 0.55,
        earlyPreCompactBlockSize: 64,
        earlyPreCompactTopHaloK: 8
    )

    public static func preset(for mode: Mode) -> Gemma4BlockSDCMConfig {
        switch mode {
        case .off: return .off
        case .lightcone: return .lightcone
        case .balanced: return .balanced
        case .fast: return .fast
        case .turbo: return .turbo
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
        case "turbo", "v3", "l1", "ultra":
            return .turbo
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
            // Explicit early-k override applies uniformly across early layers.
            config.layerMoEK = config.layerMoEK.filter { $0.key >= config.earlyMoEUntilLayer }
        }
        return config
    }

    /// Current process-level configuration (resolved once from environment).
    public static let current: Gemma4BlockSDCMConfig = resolve()
}

// MARK: - Bilateral Boundary Scores

/// Per-block bilateral saliency computed at a full-attention compaction boundary.
public struct Gemma4SDCMBoundaryScores: Sendable {
    /// Dual-scale frontier-focused (`F8 + F64`) forward + 2-hop bidirectional adjoint block scores.
    public let forwardBlockScores: [Float]
    /// Optional pure log-odds Reverse Causal Prefix Resonance (`doc -> prefix` vs sink, head-max z) block scores.
    public let reverseBlockScores: [Float]?
    /// Dedicated fraction of the interior block budget reserved for `reverseBlockScores`.
    public let reverseQuotaFraction: Float

    public init(
        forwardBlockScores: [Float],
        reverseBlockScores: [Float]? = nil,
        reverseQuotaFraction: Float = 0
    ) {
        self.forwardBlockScores = forwardBlockScores
        self.reverseBlockScores = reverseBlockScores
        self.reverseQuotaFraction = reverseQuotaFraction
    }
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
            blockSize > 0, wakeFraction.isFinite, wakeFraction > 0
        else { return [] }
        let Lq = queryQ.dim(2)
        let T = keyK.dim(2)
        let D = queryQ.dim(3)
        precondition(
            nKvHeads > 0 && nHeads > 0 && nHeads.isMultiple(of: nKvHeads),
            "Block-SDCM wake-up requires nHeads to be a positive multiple of nKvHeads")
        precondition(
            queryQ.ndim == 4 && keyK.ndim == 4
                && queryQ.dim(0) == 1 && keyK.dim(0) == 1
                && queryQ.dim(1) == nHeads && keyK.dim(1) == nKvHeads
                && keyK.dim(3) == D && T == positions.count,
            "Block-SDCM wake-up queries/keys shape mismatch")
        let g = nHeads / nKvHeads
        let qG = queryQ.reshaped(nKvHeads, g, Lq, D).asType(.float32)
        let kG = keyK.reshaped(nKvHeads, 1, T, D).asType(.float32)
        let scores = matmul(qG, kG.transposed(0, 1, 3, 2)) * scale
        let probs = softmax(scores, axis: -1).reshaped(nHeads, Lq, T)
        let docMask2D = (MLXArray(positions) .>= Int32(prefixSkip)).expandedDimensions(axis: 0)
        let combined = Gemma4BlockSDCM.headBalancedDocumentSaliency(
            probs: probs, docMask2D: docMask2D)
        eval(combined)
        let scHost = combined.asArray(Float.self)

        let nBlocks = Gemma4BlockSDCM.blockCount(
            originalLength: originalLength, blockSize: blockSize)
        let aggregated = Gemma4BlockSDCM.aggregateBlockScores(
            scores: scHost, positions: positions, blockSize: blockSize, nBlocks: nBlocks)
        guard !aggregated.aliveBlocks.isEmpty else { return [] }

        let halo = Gemma4BlockSDCM.applyBlockHalo(
            scores: aggregated.scores,
            aliveBlocks: aggregated.aliveBlocks,
            blockAlive: aggregated.alive,
            forwardHalo: 0.85,
            backwardHalo: 0.85)
        let clampedWake = min(1, wakeFraction)
        let targetBlocks = max(1, Int((Float(nBlocks) * clampedWake).rounded()))
        let ranked = Gemma4BlockSDCM.rankBlocksDescending(aggregated.aliveBlocks, scores: halo)
        let chosenBlocks = Set(ranked.prefix(targetBlocks))

        var wakeIndices: [Int32] = []
        wakeIndices.reserveCapacity(min(positions.count, targetBlocks * blockSize))
        for (i, pos) in positions.enumerated() {
            if let b = Gemma4BlockSDCM.blockIndex(
                forPosition: pos, blockSize: blockSize, nBlocks: nBlocks),
                chosenBlocks.contains(b) && !alreadyKeptPositions.contains(pos)
            {
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

    /// Number of contiguous blocks covering `originalLength` (always `>= 1`).
    fileprivate static func blockCount(originalLength: Int, blockSize: Int) -> Int {
        let effectiveBlockSize = max(1, blockSize)
        guard originalLength > 0 else { return 1 }
        return (originalLength + effectiveBlockSize - 1) / effectiveBlockSize
    }

    /// Map an original token position to its block index in `0 ..< nBlocks`, or `nil` if out of range.
    fileprivate static func blockIndex(
        forPosition position: Int32, blockSize: Int, nBlocks: Int
    ) -> Int? {
        guard position >= 0 else { return nil }
        let b = Int(position) / blockSize
        return b < nBlocks ? b : nil
    }

    /// Block boundaries for mandatory pinned prefix (`..< prefBlocks`) and tail (`tailStartBlock...`).
    private static func pinnedBlockBounds(
        nBlocks: Int, blockSize: Int, config: Gemma4BlockSDCMConfig
    ) -> (prefBlocks: Int, tailStartBlock: Int) {
        let prefixKeep = max(0, config.prefixKeep)
        let tailKeep = max(1, config.tailKeep)
        let prefBlocks = prefixKeep > 0 ? (prefixKeep + blockSize - 1) / blockSize : 0
        let tailBlocks = max(1, (tailKeep + blockSize - 1) / blockSize)
        let tailStartBlock = max(0, nBlocks - tailBlocks)
        return (prefBlocks, tailStartBlock)
    }

    /// Rank block indices by descending score with deterministic ascending-index tie-breaking
    /// and strict weak ordering even when `scores` contains `NaN`.
    fileprivate static func rankBlocksDescending<S: Sequence>(
        _ blocks: S, scores: [Float], excluding excluded: Set<Int> = []
    ) -> [Int] where S.Element == Int {
        let candidates =
            excluded.isEmpty ? Array(blocks) : blocks.filter { !excluded.contains($0) }
        return candidates.sorted { lhs, rhs in
            let ls = scores[lhs]
            let rs = scores[rhs]
            if ls.isNaN != rs.isNaN { return !ls.isNaN }
            if !ls.isNaN && ls != rs { return ls > rs }
            return lhs < rhs
        }
    }

    /// Normalize a `[H, L]` per-head saliency matrix by each head's mean and average across heads.
    private static func headMeanNormalized(_ perHeadToken: MLXArray, eps: Float) -> MLXArray {
        (perHeadToken / (perHeadToken.mean(axis: -1, keepDims: true) + eps)).mean(axis: 0)
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
            guard
                let b = blockIndex(
                    forPosition: positions[i], blockSize: blockSize, nBlocks: nBlocks)
            else { continue }
            let s = scores[i].isFinite ? scores[i] : -1e9
            if !blockAlive[b] {
                blockAlive[b] = true
                aliveBlocks.append(b)
                blockScore[b] = s
            } else if s > blockScore[b] {
                blockScore[b] = s
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
            guard s.isFinite else { continue }
            if b + 1 < nBlocks && blockAlive[b + 1] {
                let propagated = forwardHalo * s
                if propagated.isFinite {
                    halo[b + 1] = halo[b + 1].isFinite ? max(halo[b + 1], propagated) : propagated
                }
            }
            if b - 1 >= 0 && blockAlive[b - 1] {
                let propagated = backwardHalo * s
                if propagated.isFinite {
                    halo[b - 1] = halo[b - 1].isFinite ? max(halo[b - 1], propagated) : propagated
                }
            }
        }
        return halo
    }

    /// Z-score a per-block vector using mean and standard deviation over `interiorBlocks`.
    /// Returns all zeros when interior variance is zero or non-finite; non-finite entries map to 0.
    private static func zScoreBlocks(_ values: [Float], interiorBlocks: [Int]) -> [Float] {
        guard interiorBlocks.count > 2 else {
            return values.map { $0.isFinite ? $0 : 0 }
        }
        var sum: Double = 0
        var minVal = Float.greatestFiniteMagnitude
        var maxVal = -Float.greatestFiniteMagnitude
        var finiteCount = 0
        for b in interiorBlocks {
            let v = values[b]
            if v.isFinite {
                sum += Double(v)
                if v < minVal { minVal = v }
                if v > maxVal { maxVal = v }
                finiteCount += 1
            }
        }
        guard finiteCount > 2, maxVal > minVal else {
            return [Float](repeating: 0, count: values.count)
        }
        let mean = sum / Double(finiteCount)
        var varSum: Double = 0
        for b in interiorBlocks {
            let v = values[b]
            if v.isFinite {
                let d = Double(v) - mean
                varSum += d * d
            }
        }
        guard varSum > 0 && varSum.isFinite else {
            return [Float](repeating: 0, count: values.count)
        }
        let std = Float(sqrt(varSum / Double(finiteCount))) + 1e-6
        let meanF = Float(mean)
        var out = [Float](repeating: 0, count: values.count)
        for i in 0 ..< values.count {
            let v = values[i]
            if v.isFinite {
                let z = (v - meanF) / std
                out[i] = z.isFinite ? z : 0
            }
        }
        return out
    }

    /// Sum per-head per-token probabilities `[H, L]` (flat row-major) into blocks, normalize
    /// each head by its mean across `interiorBlocks`, and return `(headMean, headMax)` per block.
    private static func aggregateHeadBalancedBlocks(
        flatHeadToken: [Float],
        nHeads: Int,
        tokenCount: Int,
        tokenBlocks: [Int],
        nBlocks: Int,
        interiorBlocks: [Int]
    ) -> (meanOverHeads: [Float], maxOverHeads: [Float]) {
        var meanOut = [Float](repeating: 0, count: nBlocks)
        var maxOut = [Float](repeating: 0, count: nBlocks)
        var bh = [Float](repeating: 0, count: nBlocks)
        let invH = 1.0 / Float(max(1, nHeads))
        for h in 0 ..< nHeads {
            for b in 0 ..< nBlocks { bh[b] = 0 }
            let rowBase = h * tokenCount
            for i in 0 ..< tokenCount {
                let b = tokenBlocks[i]
                if b >= 0 {
                    let v = flatHeadToken[rowBase + i]
                    if v.isFinite { bh[b] += v }
                }
            }
            var interSum: Double = 0
            for b in interiorBlocks { interSum += Double(bh[b]) }
            let denom =
                interiorBlocks.isEmpty
                ? Float(1e-6)
                : Float(interSum / Double(interiorBlocks.count)) + 1e-6
            let invDenom = 1.0 / denom
            for b in 0 ..< nBlocks {
                let nb = bh[b] * invDenom
                meanOut[b] += nb * invH
                if h == 0 || nb > maxOut[b] { maxOut[b] = nb }
            }
        }
        return (meanOut, maxOut)
    }

    /// Find the first index `i` in strictly sorted `positionsHost` with `Int64(positionsHost[i]) >= target`.
    private static func lowerBound(_ positionsHost: [Int32], target: Int64) -> Int {
        var lo = 0
        var hi = positionsHost.count
        while lo < hi {
            let mid = lo + ((hi - lo) >> 1)
            if Int64(positionsHost[mid]) < target {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        return lo
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
        guard x.dim(2) > 0 else { return x }
        // Transpose [1, H, L, D] -> [L, H, 1, D] so axis 0 aligns with batched `offset: [L]`.
        let transposed = x.transposed(2, 1, 0, 3)
        let rotated = gemma4ApplyRotaryPosition(rope, to: transposed, offset: .batch(positions))
        return rotated.transposed(2, 1, 0, 3)
    }

    /// Compute Block-SDCM v2 bilateral per-block scores at a full-attention boundary:
    ///   1. Dual-scale frontier-focused forward causal boundary (`F8` frontier + `F64` tail,
    ///      zero-copy GQA matmul) + block-diverse 2-hop bidirectional causal adjoint.
    ///   2. Pure log-odds Reverse Causal Prefix Resonance (`doc -> prefix[4..<reversePrefixEnd)`
    ///      relative to sink `[0..<4)`, head-max z-score across present interior blocks).
    public static func computeBilateralBlockScores(
        queries: MLXArray,
        keys: MLXArray,
        positions: MLXArray,
        positionsHost: [Int32],
        originalLength: Int,
        scale: Float,
        nHeads: Int,
        nKvHeads: Int,
        reverseQuotaFraction: Float,
        config: Gemma4BlockSDCMConfig
    ) -> Gemma4SDCMBoundaryScores {
        let L = queries.dim(2)
        let blockSize = max(1, config.blockSize)
        let nBlocks = blockCount(originalLength: originalLength, blockSize: blockSize)
        precondition(
            nKvHeads > 0 && nHeads > 0 && nHeads.isMultiple(of: nKvHeads),
            "Block-SDCM requires nHeads to be a positive multiple of nKvHeads")
        precondition(
            queries.ndim == 4 && keys.ndim == 4
                && queries.dim(0) == 1 && keys.dim(0) == 1
                && queries.dim(1) == nHeads && keys.dim(1) == nKvHeads
                && keys.dim(2) == L && queries.dim(3) == keys.dim(3),
            "Block-SDCM queries/keys shape mismatch")
        precondition(
            positionsHost.count == L && positions.dim(0) == L,
            "Block-SDCM boundary scoring positions length mismatch")
        guard L > 0 else {
            return Gemma4SDCMBoundaryScores(
                forwardBlockScores: [Float](repeating: 0, count: nBlocks),
                reverseBlockScores: nil,
                reverseQuotaFraction: 0)
        }

        let g = nHeads / nKvHeads
        let D = queries.dim(3)
        let qG = queries.reshaped(nKvHeads, g, L, D)
        let kG = keys.reshaped(nKvHeads, 1, L, D)
        let kGF32T = kG.asType(.float32).transposed(0, 1, 3, 2)  // [Hk, 1, D, L]
        let negInf = MLXArray(-Float(1e9))
        let eps = Float(1e-6)

        // Precompute token -> block mapping and present interior blocks.
        var tokenBlocks = [Int](repeating: -1, count: L)
        var blockTokenCount = [Int](repeating: 0, count: nBlocks)
        for i in 0 ..< L {
            if let b = blockIndex(
                forPosition: positionsHost[i], blockSize: blockSize, nBlocks: nBlocks)
            {
                tokenBlocks[i] = b
                blockTokenCount[b] += 1
            }
        }
        let (prefBlocks, tailStartBlock) = pinnedBlockBounds(
            nBlocks: nBlocks, blockSize: blockSize, config: config)
        let interiorStart = min(prefBlocks, nBlocks)
        let interiorEnd = max(interiorStart, tailStartBlock)
        var interiorBlocks: [Int] = []
        interiorBlocks.reserveCapacity(interiorEnd - interiorStart)
        for b in interiorStart ..< interiorEnd where blockTokenCount[b] > 0 {
            interiorBlocks.append(b)
        }

        // 1. Dual-scale frontier-focused forward causal boundary (F8 frontier + F64 full tail).
        // `positionsHost` is strictly ascending, so index order matches original-position causal order.
        let qtLen = min(64, L)
        let qTail = qG[0..., 0..., (L - qtLen)..., 0...].asType(.float32)  // [Hk, G, qtLen, D]
        var logitsF = matmul(qTail, kGF32T) * scale  // [Hk, G, qtLen, L]
        let qIdx = MLXArray((L - qtLen ..< L).map { Int32($0) }).expandedDimensions(axis: 1)
        let kIdx = MLXArray((0 ..< L).map { Int32($0) }).expandedDimensions(axis: 0)
        let invalidCausal = (kIdx .> qIdx).expandedDimensions(axes: [0, 1])
        logitsF = MLX.where(invalidCausal, negInf, logitsF)
        let probsF = softmax(logitsF, axis: -1).reshaped(nHeads, qtLen, L)

        let tokF64 = probsF.mean(axis: 1)  // [Hq, L]
        let f8Len = min(8, qtLen)
        let tokF8 = probsF[0..., (qtLen - f8Len)..., 0...].mean(axis: 1)  // [Hq, L]
        let tokCombo = headMeanNormalized(tokF8, eps: eps) + headMeanNormalized(tokF64, eps: eps)
        eval(tokF64, tokF8, tokCombo)

        let f64Host = tokF64.asArray(Float.self)
        let f8Host = tokF8.asArray(Float.self)
        let tokComboHost = tokCombo.asArray(Float.self)

        let (b64Mean, _) = aggregateHeadBalancedBlocks(
            flatHeadToken: f64Host, nHeads: nHeads, tokenCount: L,
            tokenBlocks: tokenBlocks, nBlocks: nBlocks, interiorBlocks: interiorBlocks)
        let (b8Mean, b8Max) = aggregateHeadBalancedBlocks(
            flatHeadToken: f8Host, nHeads: nHeads, tokenCount: L,
            tokenBlocks: tokenBlocks, nBlocks: nBlocks, interiorBlocks: interiorBlocks)

        let zB64 = zScoreBlocks(b64Mean, interiorBlocks: interiorBlocks)
        let zB8M = zScoreBlocks(b8Mean, interiorBlocks: interiorBlocks)
        let zB8X = zScoreBlocks(b8Max, interiorBlocks: interiorBlocks)
        var blkFwd = [Float](repeating: 0, count: nBlocks)
        for b in 0 ..< nBlocks {
            blkFwd[b] = max(zB8M[b], zB64[b]) + 0.35 * zB8X[b]
        }

        // 2. Block-diverse 2-hop bidirectional causal adjoint from top-K interior blocks.
        if config.twoHopBlockPeaks > 0, !interiorBlocks.isEmpty,
            L > max(0, config.prefixKeep) + max(1, config.tailKeep) + blockSize
        {
            let topBlockSet = Set(
                rankBlocksDescending(interiorBlocks, scores: blkFwd).prefix(config.twoHopBlockPeaks)
            )
            var bestInBlock: [Int: (score: Float, index: Int32)] = [:]
            bestInBlock.reserveCapacity(topBlockSet.count)
            for i in 0 ..< L {
                let b = tokenBlocks[i]
                guard topBlockSet.contains(b) else { continue }
                let s = tokComboHost[i]
                guard s.isFinite else { continue }
                if let prev = bestInBlock[b] {
                    if s > prev.score { bestInBlock[b] = (s, Int32(i)) }
                } else {
                    bestInBlock[b] = (s, Int32(i))
                }
            }
            let peakIndices = bestInBlock.values.map(\.index).sorted()
            if !peakIndices.isEmpty {
                let pArr = MLXArray(peakIndices)
                let pCount = peakIndices.count
                let qpG = qG.take(pArr, axis: 2).asType(.float32)  // [Hk, G, P, D]
                let kpG = kG.take(pArr, axis: 2).asType(.float32)  // [Hk, 1, P, D]

                // Backward hop: peak queries attend earlier keys.
                var lgB = matmul(qpG, kGF32T) * scale  // [Hk, G, P, L]
                let invalidB = (kIdx .> pArr.expandedDimensions(axis: 1)).expandedDimensions(
                    axes: [0, 1])
                lgB = MLX.where(invalidB, negInf, lgB)
                let pBHead = softmax(lgB, axis: -1).reshaped(nHeads, pCount, L).mean(axis: 1)
                let pB = headMeanNormalized(pBHead, eps: eps)

                // Forward hop: later queries attend peak keys (zero-copy GQA, [Hk, G, P, L]).
                var lgF =
                    matmul(kpG, qG.asType(.float32).transposed(0, 1, 3, 2)) * scale
                let invalidF = (kIdx .< pArr.expandedDimensions(axis: 1)).expandedDimensions(
                    axes: [0, 1])
                lgF = MLX.where(invalidF, negInf, lgF)
                let pFHead = softmax(lgF, axis: -1).reshaped(nHeads, pCount, L).mean(axis: 1)
                let pF = headMeanNormalized(pFHead, eps: eps)

                let hop2Tok = pB + Float(0.5) * pF
                eval(hop2Tok)
                let hop2Host = hop2Tok.asArray(Float.self)
                var blkHop2 = [Float](repeating: 0, count: nBlocks)
                for i in 0 ..< L {
                    let b = tokenBlocks[i]
                    if b >= 0 {
                        let v = hop2Host[i]
                        if v.isFinite { blkHop2[b] += v }
                    }
                }
                let zHop2 = zScoreBlocks(blkHop2, interiorBlocks: interiorBlocks)
                for b in 0 ..< nBlocks {
                    blkFwd[b] += 0.45 * zHop2[b]
                }
            }
        }

        // 3. Pure Log-Odds Reverse Causal Prefix Resonance (`doc -> prefix[4..<P]` vs sink `[0..<4)`).
        var revScores: [Float]? = nil
        let reversePrefixEnd = max(0, config.reversePrefixEnd)
        if reverseQuotaFraction.isFinite, reverseQuotaFraction > 0,
            L > reversePrefixEnd + 256, interiorBlocks.count > 4
        {
            let sinkEnd = lowerBound(positionsHost, target: 4)
            let prefEnd = lowerBound(positionsHost, target: Int64(reversePrefixEnd))
            let sinkCount = sinkEnd
            let prefCount = max(0, prefEnd - sinkEnd)
            if sinkCount >= 1 && prefCount >= 4 {
                // Zero-copy GQA matmul in model activation dtype (e.g. BF16); promote FP16 inputs
                // to FP32 first to avoid ±65504 dot-product overflow before logSumExp.
                let promoteF16 = queries.dtype == .float16 || keys.dtype == .float16
                let qRev = promoteF16 ? qG.asType(.float32) : qG
                let kRev = promoteF16 ? keys.asType(.float32) : keys
                let qFlat = qRev.reshaped(nKvHeads, g * L, D)
                let kSink = kRev[0..., 0..., ..<sinkEnd, 0...].reshaped(nKvHeads, sinkCount, D)
                let kPref = kRev[0..., 0..., sinkEnd ..< prefEnd, 0...].reshaped(
                    nKvHeads, prefCount, D)
                let dotP =
                    matmul(qFlat, kPref.transposed(0, 2, 1))
                    .reshaped(nHeads, L, prefCount).asType(.float32) * scale
                let dotS =
                    matmul(qFlat, kSink.transposed(0, 2, 1))
                    .reshaped(nHeads, L, sinkCount).asType(.float32) * scale
                let resLO = sigmoid(
                    logSumExp(dotP, axis: -1) - logSumExp(dotS, axis: -1))  // [Hq, L]
                eval(resLO)
                let resHost = resLO.asArray(Float.self)

                var interiorMask = [Bool](repeating: false, count: nBlocks)
                for b in interiorBlocks { interiorMask[b] = true }

                // Scale-free per-head reciprocal rank (`max_h(1/(4+rank_h)) + 0.15 * sum_h(...)`):
                // immune to heavy-tail outlier heads so every head's #1 semantic match is preserved.
                // For fine blocks (<= 32 tokens), smooth across immediate alive interior neighbors
                // to recover a ~64-token function/clause receptive field before ranking.
                var rrMax = [Float](repeating: 0, count: nBlocks)
                var rrSum = [Float](repeating: 0, count: nBlocks)
                var bh = [Float](repeating: 0, count: nBlocks)
                var smoothed = [Float](repeating: 0, count: nBlocks)
                for h in 0 ..< nHeads {
                    for b in 0 ..< nBlocks { bh[b] = 0 }
                    let rowBase = h * L
                    for i in 0 ..< L {
                        let b = tokenBlocks[i]
                        if b >= 0 {
                            let v = resHost[rowBase + i]
                            if v.isFinite { bh[b] += v }
                        }
                    }
                    for b in interiorBlocks {
                        bh[b] /= Float(blockTokenCount[b])
                    }
                    let rankSource: [Float]
                    if blockSize <= 32 {
                        for b in interiorBlocks {
                            var acc = Float(0.50) * bh[b]
                            var w: Float = 0.50
                            if b > 0 && interiorMask[b - 1] {
                                acc += Float(0.25) * bh[b - 1]
                                w += 0.25
                            }
                            if b + 1 < nBlocks && interiorMask[b + 1] {
                                acc += Float(0.25) * bh[b + 1]
                                w += 0.25
                            }
                            smoothed[b] = acc / w
                        }
                        rankSource = smoothed
                    } else {
                        rankSource = bh
                    }
                    var minV = Float.greatestFiniteMagnitude
                    var maxV = -Float.greatestFiniteMagnitude
                    for b in interiorBlocks {
                        let v = rankSource[b]
                        if v.isFinite {
                            if v < minV { minV = v }
                            if v > maxV { maxV = v }
                        }
                    }
                    guard maxV > minV else { continue }
                    let ranked = rankBlocksDescending(interiorBlocks, scores: rankSource)
                    for (rank, b) in ranked.enumerated() {
                        let rr = Float(1.0) / Float(4 + rank)
                        if rr > rrMax[b] { rrMax[b] = rr }
                        rrSum[b] += rr
                    }
                }
                var revCombined = [Float](repeating: 0, count: nBlocks)
                for b in interiorBlocks {
                    revCombined[b] = rrMax[b] + Float(0.15) * rrSum[b]
                }
                revScores = revCombined
            }
        }

        return Gemma4SDCMBoundaryScores(
            forwardBlockScores: blkFwd,
            reverseBlockScores: revScores,
            reverseQuotaFraction: revScores != nil ? reverseQuotaFraction : 0)
    }

    /// Backwards-compatible token-level boundary score helper (maps bilateral block scores back
    /// onto active tokens).
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
        let bilateral = computeBilateralBlockScores(
            queries: queries, keys: keys, positions: positions, positionsHost: positionsHost,
            originalLength: originalLength, scale: scale, nHeads: nHeads, nKvHeads: nKvHeads,
            reverseQuotaFraction: 0, config: config)
        let bs = max(1, config.blockSize)
        let nBlocks = bilateral.forwardBlockScores.count
        return positionsHost.map { p in
            if let b = blockIndex(forPosition: p, blockSize: bs, nBlocks: nBlocks) {
                return bilateral.forwardBlockScores[b]
            }
            return 0
        }
    }

    /// Quota-partitioned deterministic block-atomic token selection with symmetric halos,
    /// mandatory prefix/tail retention, stratified macro-segment floor, and dedicated
    /// Reverse Causal Prefix Resonance quota.
    public static func selectCompactedIndices(
        boundaryScores: Gemma4SDCMBoundaryScores,
        positions: [Int32],
        originalLength: Int,
        targetFraction: Float,
        config: Gemma4BlockSDCMConfig
    ) -> [Int32] {
        let count = positions.count
        guard count > 0, originalLength > 0 else { return [] }
        let blockSize = max(1, config.blockSize)
        let nBlocks = blockCount(originalLength: originalLength, blockSize: blockSize)
        precondition(
            boundaryScores.forwardBlockScores.count == nBlocks,
            "Block-SDCM forward block score count mismatch")
        if let revRaw = boundaryScores.reverseBlockScores {
            precondition(
                revRaw.count == nBlocks,
                "Block-SDCM reverse block score count mismatch")
        }

        let (prefBlocks, tailStartBlock) = pinnedBlockBounds(
            nBlocks: nBlocks, blockSize: blockSize, config: config)

        var blockAlive = [Bool](repeating: false, count: nBlocks)
        var aliveBlockList: [Int] = []
        aliveBlockList.reserveCapacity(nBlocks)
        var pinnedTokens = 0
        for p in positions {
            guard let b = blockIndex(forPosition: p, blockSize: blockSize, nBlocks: nBlocks) else {
                continue
            }
            if !blockAlive[b] {
                blockAlive[b] = true
                aliveBlockList.append(b)
            }
            if b < prefBlocks || b >= tailStartBlock {
                pinnedTokens += 1
            }
        }

        var pinnedBlocks = Set<Int>()
        var interiorBlocks: [Int] = []
        interiorBlocks.reserveCapacity(aliveBlockList.count)
        for b in aliveBlockList {
            if b < prefBlocks || b >= tailStartBlock {
                pinnedBlocks.insert(b)
            } else {
                interiorBlocks.append(b)
            }
        }

        let clampedFraction = targetFraction.isFinite ? max(0, min(1, targetFraction)) : 1
        let minKeptTokens = max(0, config.prefixKeep) + max(1, config.tailKeep)
        let desiredTokens = max(
            minKeptTokens,
            Int((Float(originalLength) * clampedFraction).rounded()))
        if desiredTokens >= count {
            return (0 ..< count).map { Int32($0) }
        }
        let needInteriorBlocks = max(
            4, (max(0, desiredTokens - pinnedTokens) + blockSize - 1) / blockSize)
        if needInteriorBlocks >= interiorBlocks.count {
            return (0 ..< count).map { Int32($0) }
        }

        // Propagate halos strictly among alive interior blocks so pinned prefix sinks and
        // tail self-attention scores never bleed into adjacent boilerplate interior blocks.
        var interiorAlive = [Bool](repeating: false, count: nBlocks)
        for b in interiorBlocks { interiorAlive[b] = true }

        let fwdHalo = applyBlockHalo(
            scores: boundaryScores.forwardBlockScores,
            aliveBlocks: interiorBlocks,
            blockAlive: interiorAlive,
            forwardHalo: config.forwardHalo,
            backwardHalo: config.backwardHalo)

        let clampedRevQuota: Float =
            boundaryScores.reverseQuotaFraction.isFinite
            ? max(0, min(1, boundaryScores.reverseQuotaFraction)) : 0
        let revHalo: [Float]? =
            clampedRevQuota > 0
            ? boundaryScores.reverseBlockScores.map { revRaw in
                applyBlockHalo(
                    scores: revRaw,
                    aliveBlocks: interiorBlocks,
                    blockAlive: interiorAlive,
                    forwardHalo: config.forwardHalo,
                    backwardHalo: config.backwardHalo)
            }
            : nil

        var keptBlocks = pinnedBlocks
        keptBlocks.reserveCapacity(pinnedBlocks.count + needInteriorBlocks)

        // 1. Stratified macro-segment diversity floor (10% of interior budget across strata).
        if config.strataCount > 0 && !interiorBlocks.isEmpty {
            let strataPerSegment = max(
                1, Int(Float(needInteriorBlocks) * 0.10) / config.strataCount)
            for s in 0 ..< config.strataCount {
                let lo = s * interiorBlocks.count / config.strataCount
                let hi = (s + 1) * interiorBlocks.count / config.strataCount
                guard lo < hi else { continue }
                let segment = rankBlocksDescending(
                    interiorBlocks[lo ..< hi], scores: fwdHalo, excluding: keptBlocks)
                for b in segment.prefix(strataPerSegment) {
                    keptBlocks.insert(b)
                }
            }
        }

        // 2. Dedicated Reverse Causal Prefix Resonance quota (cannot starve forward or be starved).
        if let revHalo {
            let revQuota = max(
                2, Int((Float(needInteriorBlocks) * clampedRevQuota).rounded()))
            let revCandidates = rankBlocksDescending(
                interiorBlocks, scores: revHalo, excluding: keptBlocks)
            for b in revCandidates.prefix(revQuota) {
                keptBlocks.insert(b)
            }
        }

        // 3. Fill all remaining interior budget with highest Forward+2-Hop blocks.
        let keptInteriorCount = keptBlocks.count - pinnedBlocks.count
        let remaining = max(0, needInteriorBlocks - keptInteriorCount)
        if remaining > 0 {
            let fwdCandidates = rankBlocksDescending(
                interiorBlocks, scores: fwdHalo, excluding: keptBlocks)
            for b in fwdCandidates.prefix(remaining) {
                keptBlocks.insert(b)
            }
        }

        var selected: [Int32] = []
        selected.reserveCapacity(min(count, keptBlocks.count * blockSize))
        for i in 0 ..< count {
            if let b = blockIndex(
                forPosition: positions[i], blockSize: blockSize, nBlocks: nBlocks),
                keptBlocks.contains(b)
            {
                selected.append(Int32(i))
            }
        }
        return selected
    }

    /// Pure deterministic block-atomic token selection from per-token scores.
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
        let blockSize = max(1, config.blockSize)
        let nBlocks = blockCount(originalLength: originalLength, blockSize: blockSize)
        let aggregated = aggregateBlockScores(
            scores: scores, positions: positions, blockSize: blockSize, nBlocks: nBlocks)
        return selectCompactedIndices(
            boundaryScores: Gemma4SDCMBoundaryScores(forwardBlockScores: aggregated.scores),
            positions: positions,
            originalLength: originalLength,
            targetFraction: targetFraction,
            config: config)
    }

    /// Average of the `k` largest finite values in `slice`, or `0` if empty/non-finite.
    private static func topKMean(_ slice: ArraySlice<Float>, k: Int) -> Float {
        guard k > 0, !slice.isEmpty else { return 0 }
        var finite: [Float] = []
        finite.reserveCapacity(slice.count)
        for v in slice where v.isFinite { finite.append(v) }
        guard !finite.isEmpty else { return 0 }
        let takeCount = min(k, finite.count)
        if takeCount < finite.count {
            finite.sort(by: >)
        }
        var sum: Double = 0
        for i in 0 ..< takeCount { sum += Double(finite[i]) }
        return Float(sum / Double(takeCount))
    }

    /// Zero-bloat Early Tri-Signal Block Pre-Compaction (designed for post-Layer-0 compaction
    /// before Layers 1..5 run) combining three orthogonal signals:
    ///  1. `L0_emb` Centered ColBERT MaxSim (`prefix[4..<160] ∪ tail`) + 2-Hop MaxSim
    ///     (catches lexical, semantic, and multi-hop query targets).
    ///  2. Post-L0 `C @ C^T` Block-Centroid Outlier Novelty (`-top8-mean` cosine similarity
    ///     to non-adjacent interior blocks; catches zero-lexical-overlap bugs/tables/modules).
    ///  3. CPU Document-IDF Token Rarity (`1 / doc_freq(token_id)`, top-8 per block;
    ///     catches unique constants, digits, ports, and identifiers at ~0.05 ms CPU cost).
    ///
    /// Applies an explicit `±1` neighbor halo ONLY to the top `topHaloK` sharpest query hits
    /// (preserving multi-line facts without bloating the entire budget), reserves a 10% 4-strata
    /// diversity floor, and partitions the remaining interior budget 35% Query / 35% Novelty / 30% IDF.
    public static func selectEarlyTriSignalIndices(
        tokenIDs: [Int32]?,
        embeddingHidden: MLXArray,
        layerHidden: MLXArray,
        targetFraction: Float,
        blockSize: Int,
        topHaloK: Int,
        config: Gemma4BlockSDCMConfig
    ) -> [Int32] {
        precondition(
            embeddingHidden.ndim == 3 && embeddingHidden.dim(0) == 1
                && layerHidden.ndim == 3 && layerHidden.dim(0) == 1
                && embeddingHidden.dim(1) == layerHidden.dim(1)
                && embeddingHidden.dim(2) == layerHidden.dim(2),
            "Block-SDCM early pre-compaction expects matching [1, T, D] tensors")
        let T = embeddingHidden.dim(1)
        guard T > 0 else { return [] }
        guard embeddingHidden.dim(2) > 0 else { return (0 ..< T).map { Int32($0) } }
        let bs = max(1, blockSize)
        let nBlocks = blockCount(originalLength: T, blockSize: bs)
        let (prefBlocks, tailStartBlock) = pinnedBlockBounds(
            nBlocks: nBlocks, blockSize: bs, config: config)
        let interiorStart = min(prefBlocks, nBlocks)
        let interiorEnd = max(interiorStart, tailStartBlock)
        let interiorBlocks = Array(interiorStart ..< interiorEnd)
        guard interiorBlocks.count > 4 else { return (0 ..< T).map { Int32($0) } }

        let clampedFraction = targetFraction.isFinite ? max(0, min(1, targetFraction)) : 1
        let minKeptTokens = max(0, config.prefixKeep) + max(1, config.tailKeep)
        let desiredTokens = max(minKeptTokens, Int((Float(T) * clampedFraction).rounded()))
        if desiredTokens >= T {
            return (0 ..< T).map { Int32($0) }
        }
        let s0 = min(T, interiorStart * bs)
        let s1 = min(T, interiorEnd * bs)
        let pinnedTokens = s0 + (T - s1)
        let needInteriorBlocks = max(4, (max(0, desiredTokens - pinnedTokens) + bs - 1) / bs)
        if needInteriorBlocks >= interiorBlocks.count {
            return (0 ..< T).map { Int32($0) }
        }

        let eps = Float(1e-6)
        let epsStd = Float(1e-5)

        // 1. L0_emb Centered ColBERT MaxSim (prefix U tail) + 2-Hop MaxSim.
        let h0 = embeddingHidden[0].asType(.float32)  // [T, D]
        let u0F32 = h0 / (sqrt((h0 * h0).sum(axis: -1, keepDims: true)) + eps)
        let useBF16Matmul = embeddingHidden.dtype == .bfloat16
        let u0Mat = useBF16Matmul ? u0F32.asType(.bfloat16) : u0F32

        let p0 = min(4, T)
        let p1 = max(p0, min(160, T / 4))
        let tailStartTok = max(p1, T - max(1, config.tailKeep))
        var qParts: [MLXArray] = []
        if p1 > p0 { qParts.append(u0Mat[p0 ..< p1]) }
        if tailStartTok < T { qParts.append(u0Mat[tailStartTok ..< T]) }

        var blkMS = [Float](repeating: 0, count: nBlocks)
        var blkHop = [Float](repeating: 0, count: nBlocks)
        if !qParts.isEmpty && s1 > s0 {
            let uQ = qParts.count == 1 ? qParts[0] : concatenated(qParts, axis: 0)  // [Q, D]
            let sim = matmul(u0Mat, uQ.transposed(1, 0)).asType(.float32)  // [T, Q]
            let simInt = sim[s0 ..< s1]
            let muQ = simInt.mean(axis: 0, keepDims: true)
            let diffQ = simInt - muQ
            let sdQ = sqrt((diffQ * diffQ).mean(axis: 0, keepDims: true)) + epsStd
            let tokMS = ((sim - muQ) / sdQ).max(axis: -1)  // [T]
            eval(tokMS)
            let tokMSHost = tokMS.asArray(Float.self)

            for b in interiorBlocks {
                let lo = b * bs
                let hi = min(T, lo + bs)
                if hi > lo {
                    blkMS[b] = topKMean(tokMSHost[lo ..< hi], k: 4)
                }
            }

            // 2-Hop MaxSim from top-24 interior blocks (2 peak tokens each).
            let topHopBlocks = rankBlocksDescending(interiorBlocks, scores: blkMS).prefix(24)
            var hopIdx: [Int32] = []
            hopIdx.reserveCapacity(topHopBlocks.count * 2)
            for b in topHopBlocks {
                let lo = b * bs
                let hi = min(T, lo + bs)
                guard hi > lo else { continue }
                var best1 = -1
                var best2 = -1
                var val1 = -Float.infinity
                var val2 = -Float.infinity
                for i in lo ..< hi {
                    let v = tokMSHost[i]
                    guard v.isFinite else { continue }
                    if v > val1 {
                        best2 = best1
                        val2 = val1
                        best1 = i
                        val1 = v
                    } else if v > val2 {
                        best2 = i
                        val2 = v
                    }
                }
                if best1 >= 0 { hopIdx.append(Int32(best1)) }
                if best2 >= 0 { hopIdx.append(Int32(best2)) }
            }
            if !hopIdx.isEmpty {
                let uHop = u0Mat.take(MLXArray(hopIdx), axis: 0)  // [P, D]
                let simH = matmul(u0Mat, uHop.transposed(1, 0)).asType(.float32)  // [T, P]
                let simHInt = simH[s0 ..< s1]
                let muH = simHInt.mean(axis: 0, keepDims: true)
                let diffH = simHInt - muH
                let sdH = sqrt((diffH * diffH).mean(axis: 0, keepDims: true)) + epsStd
                var zH = (simH - muH) / sdH  // [T, P]
                // Mask out self-block matches so 2-hop discovers cross-block links.
                let tokBlkArr = MLXArray((0 ..< T).map { Int32($0 / bs) }).expandedDimensions(
                    axis: 1)
                let hopBlkArr = MLXArray(hopIdx.map { Int32(Int($0) / bs) }).expandedDimensions(
                    axis: 0)
                zH = MLX.where(tokBlkArr .== hopBlkArr, MLXArray(-Float(10.0)), zH)
                let tokHop = zH.max(axis: -1)
                eval(tokHop)
                let tokHopHost = tokHop.asArray(Float.self)
                for b in interiorBlocks {
                    let lo = b * bs
                    let hi = min(T, lo + bs)
                    if hi > lo {
                        blkHop[b] = topKMean(tokHopHost[lo ..< hi], k: 4)
                    }
                }
            }
        }

        // 2. Post-L0 `C @ C^T` Block-Centroid Outlier Novelty (`[nBlocks, nBlocks]`).
        let hc = layerHidden[0].asType(.float32)  // [T, D]
        let pad = nBlocks * bs - T
        let hcPadded: MLXArray
        if pad > 0 {
            hcPadded = concatenated(
                [hc, MLXArray.zeros([pad, hc.dim(1)], dtype: .float32)], axis: 0)
        } else {
            hcPadded = hc
        }
        let centroidsRaw = hcPadded.reshaped(nBlocks, bs, hc.dim(1)).mean(axis: 1)  // [nBlocks, D]
        let centroids =
            centroidsRaw / (sqrt((centroidsRaw * centroidsRaw).sum(axis: -1, keepDims: true)) + eps)
        let simMat = matmul(centroids, centroids.transposed(1, 0))  // [nBlocks, nBlocks]
        eval(simMat)
        let simHost = simMat.asArray(Float.self)

        var blkNov = [Float](repeating: 0, count: nBlocks)
        var others: [Float] = []
        others.reserveCapacity(interiorBlocks.count)
        for b in interiorBlocks {
            others.removeAll(keepingCapacity: true)
            let rowBase = b * nBlocks
            for b2 in interiorBlocks where abs(b2 - b) > 1 {
                let s = simHost[rowBase + b2]
                if s.isFinite { others.append(s) }
            }
            if !others.isEmpty {
                blkNov[b] = -topKMean(others[...], k: 8)
            }
        }

        // 3. CPU Document-IDF Token Rarity (`1 / doc_freq(tokID)`, top-8 rare tokens per block).
        var blkIDF = [Float](repeating: 0, count: nBlocks)
        let hasTokenIDF: Bool
        if let tokenIDs, tokenIDs.count == T, s1 > s0 {
            var freq: [Int32: Int] = [:]
            freq.reserveCapacity(min(s1 - s0, 4096))
            for i in s0 ..< s1 {
                freq[tokenIDs[i], default: 0] += 1
            }
            var tokIDF = [Float](repeating: 0, count: T)
            for i in s0 ..< s1 {
                let c = freq[tokenIDs[i]] ?? 1
                tokIDF[i] = 1.0 / Float(max(1, c))
            }
            for b in interiorBlocks {
                let lo = b * bs
                let hi = min(T, lo + bs)
                if hi > lo {
                    blkIDF[b] = topKMean(tokIDF[lo ..< hi], k: 8)
                }
            }
            hasTokenIDF = true
        } else {
            hasTokenIDF = false
        }

        let zMS = zScoreBlocks(blkMS, interiorBlocks: interiorBlocks)
        let zHop = zScoreBlocks(blkHop, interiorBlocks: interiorBlocks)
        var qScore = [Float](repeating: 0, count: nBlocks)
        for b in 0 ..< nBlocks { qScore[b] = zMS[b] + Float(0.50) * zHop[b] }
        let novScore = zScoreBlocks(blkNov, interiorBlocks: interiorBlocks)
        let idfScore =
            hasTokenIDF
            ? zScoreBlocks(blkIDF, interiorBlocks: interiorBlocks)
            : [Float](repeating: 0, count: nBlocks)

        var keptBlocks = Set<Int>()
        keptBlocks.reserveCapacity((nBlocks - interiorBlocks.count) + needInteriorBlocks)
        for b in 0 ..< interiorStart { keptBlocks.insert(b) }
        for b in interiorEnd ..< nBlocks { keptBlocks.insert(b) }
        let pinnedBlockCount = keptBlocks.count
        let interiorSet = Set(interiorBlocks)

        // 0. Targeted ±1 neighbor halo ONLY on top-K sharpest query hits (zero halo bloat).
        if topHaloK > 0 {
            for b in rankBlocksDescending(interiorBlocks, scores: qScore).prefix(topHaloK) {
                guard keptBlocks.count - pinnedBlockCount < needInteriorBlocks else { break }
                keptBlocks.insert(b)
                if keptBlocks.count - pinnedBlockCount < needInteriorBlocks,
                    interiorSet.contains(b + 1)
                {
                    keptBlocks.insert(b + 1)
                }
                if keptBlocks.count - pinnedBlockCount < needInteriorBlocks,
                    interiorSet.contains(b - 1)
                {
                    keptBlocks.insert(b - 1)
                }
            }
        }

        // 1. 4-strata diversity floor (10% of interior budget).
        if config.strataCount > 0, keptBlocks.count - pinnedBlockCount < needInteriorBlocks {
            let strataPerSeg = max(1, Int(Float(needInteriorBlocks) * 0.10) / config.strataCount)
            for s in 0 ..< config.strataCount {
                let avail = needInteriorBlocks - (keptBlocks.count - pinnedBlockCount)
                guard avail > 0 else { break }
                let lo = s * interiorBlocks.count / config.strataCount
                let hi = (s + 1) * interiorBlocks.count / config.strataCount
                guard lo < hi else { continue }
                let seg = rankBlocksDescending(
                    interiorBlocks[lo ..< hi], scores: qScore, excluding: keptBlocks)
                for b in seg.prefix(min(strataPerSeg, avail)) { keptBlocks.insert(b) }
            }
        }

        // 2. Partition remaining budget across Query (35%), Novelty (35%), and IDF (30%).
        let remBudget = max(0, needInteriorBlocks - (keptBlocks.count - pinnedBlockCount))
        let partitions: [(Float, [Float])] =
            hasTokenIDF
            ? [(Float(0.35), qScore), (Float(0.35), novScore), (Float(0.30), idfScore)]
            : [(Float(0.50), qScore), (Float(0.50), novScore)]
        for (frac, sc) in partitions where remBudget > 0 {
            let quota = max(1, Int((Float(remBudget) * frac).rounded()))
            let cands = rankBlocksDescending(interiorBlocks, scores: sc, excluding: keptBlocks)
            for b in cands.prefix(quota) { keptBlocks.insert(b) }
        }

        // 3. Fill any rounding residual using union max across all three signals.
        let left = max(0, needInteriorBlocks - (keptBlocks.count - pinnedBlockCount))
        if left > 0 {
            var unionScore = [Float](repeating: 0, count: nBlocks)
            for b in interiorBlocks {
                unionScore[b] =
                    hasTokenIDF
                    ? max(qScore[b], max(novScore[b], idfScore[b]))
                    : max(qScore[b], novScore[b])
            }
            let cands = rankBlocksDescending(
                interiorBlocks, scores: unionScore, excluding: keptBlocks)
            for b in cands.prefix(left) { keptBlocks.insert(b) }
        }

        var selected: [Int32] = []
        selected.reserveCapacity(min(T, keptBlocks.count * bs))
        for i in 0 ..< T {
            if keptBlocks.contains(i / bs) {
                selected.append(Int32(i))
            }
        }
        return selected
    }

    /// O(T * (P + W)) Prefix + Local Band + Exact Full-Tail sparse causal attention for
    /// uncompacted early full-attention layers (e.g. layer 5).
    /// - Interior queries attend exactly to `[0 ..< P]` (sinks + top instruction) union
    ///   `[q - W + 1 .. q]` (exact local sliding receptive field).
    /// - Trailing `tailTokens` queries attend exactly across the full sequence `[0 ..< L]`.
    public static func prefixLocalFullTailAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        prefixTokens: Int,
        localWindow: Int,
        tailTokens: Int,
        scale: Float
    ) -> MLXArray {
        let L = queries.dim(2)
        guard L > 0 else {
            return MLXArray.zeros(
                [queries.dim(0), queries.dim(1), 0, values.dim(3)], dtype: queries.dtype)
        }
        let P = max(0, prefixTokens)
        let W = localWindow
        let T = max(0, tailTokens)
        guard W > 0, P < L, T < L, L - P - T > W else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .causal)
        }
        let positionsHost = (0 ..< L).map { Int32($0) }
        return prefixLocalFullTailAttention(
            queries: queries,
            keys: keys,
            values: values,
            positions: MLXArray(positionsHost),
            positionsHost: positionsHost,
            prefixTokens: P,
            localWindow: W,
            tailTokens: T,
            scale: scale,
            chunkSize: W)
    }

    /// Position-aware O(L * (P + W)) Prefix + Local Band + Exact Full-Tail sparse causal attention
    /// supporting both contiguous and already-compacted (e.g. post-L0 `turbo`) sequences.
    /// - Interior queries attend to keys with `origPos < prefixTokens` OR `0 <= qPos - kPos < localWindow`.
    /// - Trailing `tailTokens` queries attend causally across all `0 ..< L` active keys.
    public static func prefixLocalFullTailAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        positions: MLXArray,
        positionsHost: [Int32],
        prefixTokens: Int,
        localWindow: Int,
        tailTokens: Int,
        scale: Float,
        chunkSize: Int = 1024
    ) -> MLXArray {
        let L = queries.dim(2)
        guard L > 0 else {
            return MLXArray.zeros(
                [queries.dim(0), queries.dim(1), 0, values.dim(3)], dtype: queries.dtype)
        }
        precondition(
            positionsHost.count == L && positions.dim(0) == L
                && keys.dim(2) == L && values.dim(2) == L,
            "Block-SDCM sparse full-attention positions/KV length mismatch")
        let P = max(0, prefixTokens)
        let W = localWindow
        let T = max(0, tailTokens)
        guard W > 0, T < L else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .causal)
        }
        let pEnd = P > 0 ? lowerBound(positionsHost, target: Int64(P)) : 0
        let cut = L - T
        guard pEnd < cut,
            Int64(positionsHost[cut - 1]) - Int64(positionsHost[pEnd]) >= Int64(W)
        else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .causal)
        }
        let effectiveChunk = chunkSize > 0 ? chunkSize : 1024
        let pI32 = Int32(clamping: P)
        let wI32 = Int32(clamping: W)
        let windowSpan = Int64(W - 1)
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((cut + effectiveChunk - 1) / effectiveChunk + 1)

        var s0 = 0
        while s0 < cut {
            let s1 = min(cut, s0 + effectiveChunk)
            let qChunk = queries[0..., 0..., s0 ..< s1, 0...]
            let qPos = positions[s0 ..< s1].expandedDimensions(axis: 1)
            let minPos = Int64(positionsHost[s0]) - windowSpan
            let k0 = min(s0, lowerBound(positionsHost, target: minPos))
            let kChunk: MLXArray
            let vChunk: MLXArray
            let kPos: MLXArray
            if k0 <= pEnd {
                kChunk = keys[0..., 0..., ..<s1, 0...]
                vChunk = values[0..., 0..., ..<s1, 0...]
                kPos = positions[..<s1].expandedDimensions(axis: 0)
            } else if pEnd > 0 {
                kChunk = concatenated(
                    [keys[0..., 0..., ..<pEnd, 0...], keys[0..., 0..., k0 ..< s1, 0...]], axis: 2)
                vChunk = concatenated(
                    [values[0..., 0..., ..<pEnd, 0...], values[0..., 0..., k0 ..< s1, 0...]],
                    axis: 2)
                kPos = concatenated(
                    [positions[..<pEnd], positions[k0 ..< s1]], axis: 0
                ).expandedDimensions(axis: 0)
            } else {
                kChunk = keys[0..., 0..., k0 ..< s1, 0...]
                vChunk = values[0..., 0..., k0 ..< s1, 0...]
                kPos = positions[k0 ..< s1].expandedDimensions(axis: 0)
            }
            let mask = (kPos .<= qPos) .&& ((kPos .< pI32) .|| ((qPos - kPos) .< wI32))
            chunks.append(
                MLXFast.scaledDotProductAttention(
                    queries: qChunk, keys: kChunk, values: vChunk, scale: scale,
                    mask: .array(mask)))
            s0 = s1
        }
        if cut < L {
            let qPos = positions[cut ..< L].expandedDimensions(axis: 1)
            let kPos = positions.expandedDimensions(axis: 0)
            chunks.append(
                MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., cut ..< L, 0...],
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: .array(kPos .<= qPos)))
        }
        return chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 2)
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
            let k0 = min(s0, lowerBound(positionsHost, target: minPos))
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

    /// Compute the true-window start index in `positionsHost` for committing a sliding layer's
    /// KV cache so decode never attends to out-of-window compacted keys (`origPos < T - W + 1`).
    /// Returns `positionsHost.count` when `slidingWindow <= 0` or no token lies within the window.
    public static func trueSlidingWindowStart(
        positionsHost: [Int32],
        originalLength: Int,
        slidingWindow: Int
    ) -> Int {
        let L = positionsHost.count
        guard L > 0, slidingWindow > 0 else { return L }
        if L == originalLength,
            positionsHost.first == 0,
            positionsHost.last == Int32(originalLength - 1)
        {
            return max(0, L - slidingWindow)
        }
        let minPos = Int64(originalLength) - Int64(slidingWindow) + 1
        return lowerBound(positionsHost, target: minPos)
    }
}

// MARK: - Gemma4DecoderLayer & Trunk Execution

extension Gemma4DecoderLayer {
    /// Project, Q-norm, transpose, and RoPE queries (using zero-transpose scalar offset when contiguous).
    private func projectQueriesWithRoPE(
        _ input: MLXArray, positions: MLXArray, isContiguousFromZero: Bool
    ) -> MLXArray {
        let attn = selfAttn
        let qLength = input.dim(1)
        var q = attn.qProj(input).reshaped(1, qLength, attn.nHeads, attn.effectiveHeadDim)
        q = attn.qNorm(q).transposed(0, 2, 1, 3)
        if isContiguousFromZero {
            return gemma4ApplyRotaryPosition(attn.rope, to: q, offset: .scalar(0))
        }
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

    /// Commit compacted K/V to the layer cache with logical RoPE delta.
    /// Sliding layers store ONLY tokens inside the true original-position sliding window
    /// (`origPos >= originalLength - slidingWindow + 1` when compacted) so decode queries
    /// never attend to out-of-window sliding RoPE distances (`deltaPos >= W`).
    private func commitCompactedKV(
        keys k: MLXArray,
        values v: MLXArray,
        positionsHost: [Int32],
        originalLength: Int,
        cache: KVCache
    ) {
        let attn = selfAttn
        let kStore: MLXArray
        let vStore: MLXArray
        if attn.isSliding {
            let start = Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: positionsHost,
                originalLength: originalLength,
                slidingWindow: attn.config.slidingWindow)
            if start > 0 {
                let clampedStart = min(start, k.dim(2))
                kStore = k[0..., 0..., clampedStart..., 0...]
                vStore = v[0..., 0..., clampedStart..., 0...]
            } else {
                kStore = k
                vStore = v
            }
        } else {
            kStore = k
            vStore = v
        }
        let stored = kStore.dim(2)
        let logicalDelta = originalLength - stored
        if let v2Cache = cache as? (any CBv2AttendingLayerCache) {
            v2Cache.writeCompactedPromptKV(
                keys: kStore, values: vStore, logicalDelta: logicalDelta)
        } else if let simpleCache = cache as? KVCacheSimple {
            if stored > 0 {
                _ = simpleCache.update(keys: kStore, values: vStore)
            }
            simpleCache.positionDelta += logicalDelta
        } else if let rotCache = cache as? RotatingKVCache, !(cache is CompilableRotatingKVCache) {
            if stored > 0 {
                _ = rotCache.update(keys: kStore, values: vStore)
            }
            rotCache.offset += logicalDelta
        } else {
            preconditionFailure(
                "Block-SDCM requires a cache supporting logical position deltas, got \(type(of: cache))"
            )
        }
    }

    /// Execute one decoder layer inside the Block-SDCM prefill trunk.
    func forwardBlockSDCM(
        _ x: MLXArray,
        positions: MLXArray,
        positionsHost: [Int32],
        originalLength: Int,
        cache: KVCache?,
        sdcm: Gemma4BlockSDCMConfig,
        isFinalLayer: Bool,
        nextReverseQuota: Float?,
        isExpertPrefill: Bool
    ) -> (hidden: MLXArray, nextLayerScores: Gemma4SDCMBoundaryScores?) {
        let L = x.dim(1)
        let isContiguous = L == originalLength
        let attn = selfAttn
        guard let kProj = attn.kProj, let kNorm = attn.kNorm, let vNorm = attn.vNorm else {
            preconditionFailure("Block-SDCM requires non-shared attention layers")
        }

        let normed = inputLayernorm(x)

        // K and V projections across all L currently active tokens.
        let kRaw = kProj(normed).reshaped(1, L, attn.nKvHeads, attn.effectiveHeadDim)
        var k = kNorm(kRaw).transposed(0, 2, 1, 3)
        if isContiguous {
            k = gemma4ApplyRotaryPosition(attn.rope, to: k, offset: .scalar(0))
        } else {
            k = Gemma4BlockSDCM.applyArbitraryPositionsRoPE(attn.rope, to: k, positions: positions)
        }

        var v: MLXArray
        if let vProj = attn.vProj {
            v = vProj(normed).reshaped(1, L, attn.nKvHeads, attn.effectiveHeadDim)
        } else {
            v = kRaw
        }
        v = vNorm(v).transposed(0, 2, 1, 3)

        if let cache {
            commitCompactedKV(
                keys: k, values: v, positionsHost: positionsHost,
                originalLength: originalLength, cache: cache)
        }

        let outDType = k.dtype
        let attnK = outDType == .float16 ? k.asType(.float32) : k
        let attnV = outDType == .float16 ? v.asType(.float32) : v

        // Final-layer last-query specialization: project Q, attend, O-proj, MLP, and exact
        // top-8 MoE for ONLY the final frontier token.
        if isFinalLayer {
            let qLast = projectQueriesWithRoPE(
                normed[0..., (L - 1)..., 0...],
                positions: positions[(L - 1)...],
                isContiguousFromZero: false)
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

        // Standard layer Q projection + RoPE.
        let q = projectQueriesWithRoPE(
            normed, positions: positions, isContiguousFromZero: isContiguous)
        let attnQ = outDType == .float16 ? q.asType(.float32) : q

        let attnResult: MLXArray
        var nextScores: Gemma4SDCMBoundaryScores? = nil
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
            if sdcm.earlySparseFullLayer == attn.layerIdx {
                attnResult = Gemma4BlockSDCM.prefixLocalFullTailAttention(
                    queries: attnQ,
                    keys: attnK,
                    values: attnV,
                    positions: positions,
                    positionsHost: positionsHost,
                    prefixTokens: sdcm.sparsePrefixTokens,
                    localWindow: sdcm.sparseLocalWindow,
                    tailTokens: sdcm.sparseTailTokens,
                    scale: attn.scale,
                    chunkSize: sdcm.slidingChunkSize)
            } else {
                // Kept positions are strictly increasing, so `pos[k] <= pos[q]` is identical
                // to index-causal order -> uses MLX's fused FlashAttention `.causal` kernel.
                attnResult = MLXFast.scaledDotProductAttention(
                    queries: attnQ, keys: attnK, values: attnV, scale: attn.scale, mask: .causal)
            }
            if let reverseQuota = nextReverseQuota {
                nextScores = Gemma4BlockSDCM.computeBilateralBlockScores(
                    queries: attnQ,
                    keys: attnK,
                    positions: positions,
                    positionsHost: positionsHost,
                    originalLength: originalLength,
                    scale: attn.scale,
                    nHeads: attn.nHeads,
                    nKvHeads: attn.nKvHeads,
                    reverseQuotaFraction: reverseQuota,
                    config: sdcm)
            }
        }

        let attnOut = finishAttentionOutput(attnResult, length: L, outDType: outDType)
        let ffIn = x + postAttentionLayernorm(attnOut)
        let interiorMoEK = sdcm.interiorMoEK(forLayer: attn.layerIdx)
        var h = feedforwardBlockSDCM(
            ffIn,
            interiorMoEK: interiorMoEK,
            tailKeep: sdcm.tailKeep,
            isExpertPrefill: isExpertPrefill)
        h = h * layerScalar
        return (h, nextScores)
    }

    /// Parallel dense + MoE feedforward with optional interior depth-tapered top-k MoE
    /// (exact softmax renormalization over the selected k experts).
    /// Tail tokens (`L - tailKeep ..< L`) ALWAYS execute exact full top-8 MoE.
    private func feedforwardBlockSDCM(
        _ input: MLXArray,
        interiorMoEK: Int,
        tailKeep: Int,
        isExpertPrefill: Bool
    ) -> MLXArray {
        let L = input.dim(1)
        let effectiveTailKeep = max(1, tailKeep)
        let topK = router?.topK ?? 8
        guard isMoE,
            interiorMoEK < topK,
            L > effectiveTailKeep,
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

        let cut = L - effectiveTailKeep
        let interiorIn = input[0..., ..<cut, 0...]
        let tailIn = input[0..., cut..., 0...]

        // Interior tokens: direct top-k argPartition with exact softmax renormalization.
        let (prunedIndices, prunedWeights) = router.routeTopK(interiorIn, topK: interiorMoEK)
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
        inputTokenIDs: [Int32]? = nil,
        fullCache: [KVCache?],
        sdcm: Gemma4BlockSDCMConfig,
        isExpertPrefill: Bool,
        captureCheckpointAtLayer: Int? = nil
    ) -> (postNorm: MLXArray, checkpoint: Gemma4SDCMCheckpoint?) {
        let originalLength = initialHidden.dim(1)
        var h = initialHidden
        let earlyCompactLayer: Int? = sdcm.earlyPreCompactAfterLayer.flatMap {
            (0 ..< max(0, layers.count - 1)).contains($0) ? $0 : nil
        }
        var earlyEmbeddingHidden: MLXArray? = earlyCompactLayer != nil ? initialHidden : nil
        var positionsHost = (0 ..< originalLength).map { Int32($0) }
        var positions = MLXArray(positionsHost)
        var pendingScores: Gemma4SDCMBoundaryScores? = nil
        var checkpoint: Gemma4SDCMCheckpoint? = nil

        for (idx, layer) in layers.enumerated() {
            if let targetFraction = sdcm.keepSchedule[idx], let scores = pendingScores {
                let keepIndices = Gemma4BlockSDCM.selectCompactedIndices(
                    boundaryScores: scores,
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
            let nextReverseQuota: Float? =
                sdcm.keepSchedule[idx + 1] != nil
                ? (sdcm.reversePrefixQuota[idx + 1] ?? 0) : nil
            let (out, nextScores) = layer.forwardBlockSDCM(
                h,
                positions: positions,
                positionsHost: positionsHost,
                originalLength: originalLength,
                cache: fullCache[idx],
                sdcm: sdcm,
                isFinalLayer: isFinalLayer,
                nextReverseQuota: nextReverseQuota,
                isExpertPrefill: isExpertPrefill)
            h = out
            if let nextScores {
                pendingScores = nextScores
            }
            if let earlyLayer = earlyCompactLayer, idx == earlyLayer,
                let embHidden = earlyEmbeddingHidden
            {
                if h.dim(1) == originalLength {
                    let keepEarly = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
                        tokenIDs: inputTokenIDs,
                        embeddingHidden: embHidden,
                        layerHidden: h,
                        targetFraction: sdcm.earlyPreCompactRatio,
                        blockSize: sdcm.earlyPreCompactBlockSize,
                        topHaloK: sdcm.earlyPreCompactTopHaloK,
                        config: sdcm)
                    if keepEarly.count < h.dim(1) && !keepEarly.isEmpty {
                        let idxArray = MLXArray(keepEarly)
                        h = h.take(idxArray, axis: 1)
                        positionsHost = keepEarly.map { positionsHost[Int($0)] }
                        positions = MLXArray(positionsHost)
                        asyncEval(h, positions)
                    }
                }
                earlyEmbeddingHidden = nil
            }
            if let ckptLayer = captureCheckpointAtLayer, idx == ckptLayer {
                checkpoint = Gemma4SDCMCheckpoint(
                    hiddenStates: h,
                    positions: positionsHost,
                    checkpointLayer: idx,
                    originalLength: originalLength)
            }
            // Keep Metal command buffers bounded on long prompts without forcing host syncs
            // except at scoring boundaries where `computeBilateralBlockScores` reads host floats.
            if !isFinalLayer && nextScores == nil && (idx + 1).isMultiple(of: 6) {
                asyncEval(h)
            }
        }

        return (norm(h), checkpoint)
    }
}
