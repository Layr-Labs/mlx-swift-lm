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
        /// High-compression L1 Tri-Signal Pre-Compaction (`34%` after Layer 0) + aggressive
        /// bilateral schedule (`18/11/9/8%`, 32-tok blocks) + low-k early MoE (`k = 2` on L0..L2,
        /// `k = 3` on L3..L5, `k = 4` on L6..L11, `k = 6` on L12..L28, exact `k = 8` on L29 & tail 160).
        /// Targets 5.5-7.5x prefill speedup on long contexts.
        case hyper
    }

    public static let envKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM"
    public static let minTokensEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_MIN_TOKENS"
    public static let earlyMoEKEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_EARLY_K"
    public static let densityGuardEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_DENSITY_GUARD"
    public static let maxBatchEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_MAX_BATCH"
    public static let maxTokensEnvKey = "DARKBLOOM_GEMMA4_PREFILL_SDCM_MAX_TOKENS"

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
    /// Enables the CPU dense-sequence safety gate (`passesDensityGuard`) that vetoes
    /// low-redundancy / high-density prompts back to exact prefill.
    public var densityGuardEnabled: Bool
    /// Maximum equal-length compaction batch size (`1...32`) packed into one Block-SDCM forward.
    public var maxPackedBatch: Int
    /// Upper bound on total original prompt tokens across one packed compaction batch (`>= 3072`).
    public var maxPackedTokens: Int

    public init(
        mode: Mode,
        minPromptTokens: Int = 3072,
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
        earlyPreCompactTopHaloK: Int = 8,
        densityGuardEnabled: Bool = true,
        maxPackedBatch: Int = 8,
        maxPackedTokens: Int = 32_768
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
        self.densityGuardEnabled = densityGuardEnabled
        self.maxPackedBatch = max(1, min(32, maxPackedBatch))
        self.maxPackedTokens = max(3072, maxPackedTokens)
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
        minPromptTokens: 3072,
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
        minPromptTokens: 3072,
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
        minPromptTokens: 3072,
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
        minPromptTokens: 3072,
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

    public static let hyper = Gemma4BlockSDCMConfig(
        mode: .hyper,
        minPromptTokens: 3072,
        keepSchedule: [6: 0.18, 12: 0.11, 18: 0.09, 24: 0.08],
        earlyMoEK: 4,
        lateMoEK: 6,
        earlyMoEUntilLayer: 12,
        layerMoEK: [
            0: 2, 1: 2, 2: 2,
            3: 3, 4: 3, 5: 3,
            6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4,
        ],
        blockSize: 32,
        prefixKeep: 96,
        tailKeep: 160,
        forwardHalo: 0.85,
        backwardHalo: 0.85,
        strataCount: 4,
        twoHopBlockPeaks: 24,
        slidingChunkSize: 1024,
        reversePrefixQuota: [6: 0.30, 12: 0.25, 18: 0.20, 24: 0.15],
        reversePrefixEnd: 96,
        earlySparseFullLayer: 5,
        sparsePrefixTokens: 192,
        sparseLocalWindow: 1024,
        sparseTailTokens: 192,
        earlyPreCompactAfterLayer: 0,
        earlyPreCompactRatio: 0.34,
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
        case .hyper: return .hyper
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
        case "hyper", "extreme", "v4":
            return .hyper
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
        if let rawGuard = environment[densityGuardEnvKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        {
            switch rawGuard {
            case "0", "false", "no", "off", "disabled":
                config.densityGuardEnabled = false
            case "1", "true", "yes", "on", "enabled":
                config.densityGuardEnabled = true
            default:
                break
            }
        }
        if let rawBatch = environment[maxBatchEnvKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            let value = Int(rawBatch)
        {
            config.maxPackedBatch = max(1, min(32, value))
        }
        if let rawTokens = environment[maxTokensEnvKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            let value = Int(rawTokens)
        {
            config.maxPackedTokens = max(3072, value)
        }
        return config
    }

    /// Current process-level configuration (resolved once from environment).
    public static let current: Gemma4BlockSDCMConfig = resolve()

    func resolvedEarlyCompactLayer(layerCount: Int) -> Int? {
        earlyPreCompactAfterLayer.flatMap { (0 ..< max(0, layerCount - 1)).contains($0) ? $0 : nil }
    }
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

    /// Pure O(N) CPU token-level safety gate vetoing compactable-length prompts whose
    /// interior token stream is too dense or low-redundancy for block pruning (e.g.
    /// low-alphabet biological sequences or short ultra-dense prose/proofs).
    public static func passesDensityGuard<C: RandomAccessCollection>(
        tokens: C, config: Gemma4BlockSDCMConfig
    ) -> Bool where C.Element: BinaryInteger {
        guard config.densityGuardEnabled else { return true }
        let n = tokens.count
        let s0 = min(n, max(0, config.prefixKeep))
        let s1 = max(s0, n - max(0, config.tailKeep))
        let interiorCount = s1 - s0
        guard interiorCount >= 512 else { return true }

        var freq: [Int64: Int] = [:]
        freq.reserveCapacity(min(interiorCount, 4096))
        var seen4 = Set<UInt64>()
        seen4.reserveCapacity(max(0, interiorCount - 3))
        var dup4 = 0
        var t0: UInt64 = 0
        var t1: UInt64 = 0
        var t2: UInt64 = 0

        var idx = tokens.index(tokens.startIndex, offsetBy: s0)
        for k in 0 ..< interiorCount {
            let tok = Int64(truncatingIfNeeded: tokens[idx])
            tokens.formIndex(after: &idx)
            freq[tok, default: 0] += 1
            let t3 = UInt64(bitPattern: tok)
            if k >= 3 {
                var h: UInt64 = 14_695_981_039_346_656_037
                h = (h ^ t0) &* 1_099_511_628_211
                h = (h ^ t1) &* 1_099_511_628_211
                h = (h ^ t2) &* 1_099_511_628_211
                h = (h ^ t3) &* 1_099_511_628_211
                h ^= h >> 30
                h &*= 0xBF58_476D_1CE4_E5B9
                h ^= h >> 27
                h &*= 0x94D0_49BB_1331_11EB
                h ^= h >> 31
                if !seen4.insert(h).inserted {
                    dup4 += 1
                }
            }
            t0 = t1
            t1 = t2
            t2 = t3
        }

        let invN = 1.0 / Double(interiorCount)
        let uniqueRatio = Double(freq.count) * invN
        var entropyBits: Double = 0
        for count in freq.values {
            let p = Double(count) * invN
            entropyBits -= p * log2(p)
        }
        let repeated4GramRatio = Double(dup4) / Double(max(1, interiorCount - 3))

        // Veto low-alphabet high-density bio/symbol streams.
        if interiorCount < 16_384 && entropyBits < 5.0 && repeated4GramRatio < 0.20 {
            return false
        }
        // Veto short ultra-dense prose/proofs with minimal 4-gram redundancy.
        if n < 4096 && repeated4GramRatio < 0.05 && uniqueRatio > 0.50 {
            return false
        }
        return true
    }

    /// Compute the cold-start `initialOffset` gate for `canRun` (`0` iff all active caches
    /// start at RoPE offset 0).
    static func initialOffset(for fullCache: [KVCache?]) -> Int {
        for layerCache in fullCache {
            if let v2 = layerCache as? (any CBv2AttendingLayerCache), !v2.rows.isEmpty {
                return v2.rows.allSatisfy { $0.ropePositionOffset == 0 } ? 0 : 1
            }
            if let layerCache {
                if case .scalar(let value) = layerCache.ropeOffset {
                    return value
                }
                return layerCache.offset
            }
        }
        return 0
    }

    /// Check whether a Gemma 4 prefill call qualifies for Block-SDCM.
    /// Restricts execution to cold text prefills (batch `1...sdcm.maxPackedBatch`) on
    /// non-KV-shared, non-PLE Gemma 4 topologies (e.g. 26B-A4B) above `sdcm.minPromptTokens`.
    public static func canRun(
        sdcm: Gemma4BlockSDCMConfig,
        modelConfig: Gemma4TextConfiguration,
        schedulePrefill: Bool,
        batchSize: Int,
        sequenceLength: Int,
        hasInputEmbedding: Bool,
        hasImageMask: Bool,
        hasCaptureHook: Bool,
        initialOffset: Int,
        passesDensityGuard: Bool = true
    ) -> Bool {
        let hasReachableCompaction =
            sdcm.resolvedEarlyCompactLayer(layerCount: modelConfig.numHiddenLayers) != nil
            || sdcm.keepSchedule.keys.contains { $0 > 0 && $0 < modelConfig.numHiddenLayers }
        return sdcm.mode != .off
            && schedulePrefill
            && (1 ... max(1, sdcm.maxPackedBatch)).contains(batchSize)
            && sequenceLength >= sdcm.minPromptTokens
            && sequenceLength > sdcm.prefixKeep + sdcm.tailKeep
            && passesDensityGuard
            && !hasInputEmbedding
            && !hasImageMask
            && !hasCaptureHook
            && initialOffset == 0
            && supportsModelConfig(modelConfig)
            && hasReachableCompaction
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

    /// Apply exact RoPE at arbitrary non-contiguous integer positions to `[B, H, L, D]`
    /// with `[L]` (when `B == 1`) or `[B, L]` (when `B >= 1`) int32 positions using MLX's
    /// fused batched-offset `MLXFast.RoPE` kernel.
    public static func applyArbitraryPositionsRoPE<R: RoPELayer>(
        _ rope: R,
        to x: MLXArray,
        positions: MLXArray
    ) -> MLXArray {
        precondition(x.ndim == 4 && x.dim(0) >= 1, "Block-SDCM RoPE expects [B, H, L, D]")
        let B = x.dim(0)
        let H = x.dim(1)
        let L = x.dim(2)
        let D = x.dim(3)
        if positions.ndim == 1 {
            precondition(
                B == 1 && positions.dim(0) == L,
                "Block-SDCM RoPE 1D positions require B == 1 and length L")
        } else {
            precondition(
                positions.ndim == 2 && positions.dim(0) == B && positions.dim(1) == L,
                "Block-SDCM RoPE 2D positions require shape [B, L]")
        }
        guard L > 0 else { return x }
        let flatPos = positions.reshaped(B * L)
        let flat = x.transposed(0, 2, 1, 3).reshaped(B * L, H, 1, D)
        let rotated = gemma4ApplyRotaryPosition(rope, to: flat, offset: .batch(flatPos))
        return rotated.reshaped(B, L, H, D).transposed(0, 2, 1, 3)
    }

    /// Batched Block-SDCM v2 bilateral per-block scores for `B >= 1` rows at a full-attention
    /// boundary, coalescing all `B` rows' GPU tensors into a single `eval(...)` per stage.
    static func computeBilateralBlockScoresBatched(
        queries: MLXArray,
        keys: MLXArray,
        positions: MLXArray,
        positionsHostRows: [[Int32]],
        originalLength: Int,
        scale: Float,
        nHeads: Int,
        nKvHeads: Int,
        reverseQuotaFraction: Float,
        config: Gemma4BlockSDCMConfig
    ) -> [Gemma4SDCMBoundaryScores] {
        precondition(
            nKvHeads > 0 && nHeads > 0 && nHeads.isMultiple(of: nKvHeads),
            "Block-SDCM requires nHeads to be a positive multiple of nKvHeads")
        precondition(
            queries.ndim == 4 && keys.ndim == 4 && queries.dim(0) >= 1
                && keys.dim(0) == queries.dim(0)
                && queries.dim(1) == nHeads && keys.dim(1) == nKvHeads
                && queries.dim(2) == keys.dim(2) && queries.dim(3) == keys.dim(3),
            "Block-SDCM queries/keys shape mismatch")
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(
            positionsHostRows.count == B && positionsHostRows.allSatisfy { $0.count == L }
                && positions.ndim == 2 && positions.dim(0) == B && positions.dim(1) == L,
            "Block-SDCM boundary scoring positions length mismatch")
        let blockSize = max(1, config.blockSize)
        let nBlocks = blockCount(originalLength: originalLength, blockSize: blockSize)
        guard L > 0 else {
            return Array(
                repeating: Gemma4SDCMBoundaryScores(
                    forwardBlockScores: [Float](repeating: 0, count: nBlocks),
                    reverseBlockScores: nil,
                    reverseQuotaFraction: 0),
                count: B)
        }

        let g = nHeads / nKvHeads
        let D = queries.dim(3)
        let negInf = MLXArray(-Float(1e9))
        let eps = Float(1e-6)
        let (prefBlocks, tailStartBlock) = pinnedBlockBounds(
            nBlocks: nBlocks, blockSize: blockSize, config: config)
        let interiorStart = min(prefBlocks, nBlocks)
        let interiorEnd = max(interiorStart, tailStartBlock)

        let qtLen = min(64, L)
        let f8Len = min(8, qtLen)
        let qIdx = MLXArray((L - qtLen ..< L).map { Int32($0) }).expandedDimensions(axis: 1)
        let kIdx = MLXArray((0 ..< L).map { Int32($0) }).expandedDimensions(axis: 0)
        let invalidCausal = (kIdx .> qIdx).expandedDimensions(axes: [0, 1])
        let reversePrefixEnd = max(0, config.reversePrefixEnd)
        let reverseEligibleGlobal =
            reverseQuotaFraction.isFinite && reverseQuotaFraction > 0
            && L > reversePrefixEnd + 256
        let promoteF16 = queries.dtype == .float16 || keys.dtype == .float16

        struct RowState {
            let qG: MLXArray
            let kG: MLXArray
            let kGF32T: MLXArray
            let tokenBlocks: [Int]
            let blockTokenCount: [Int]
            let interiorBlocks: [Int]
            let tokF64: MLXArray
            let tokF8: MLXArray
            let tokCombo: MLXArray
            let resLO: MLXArray?
        }

        var rowStates: [RowState] = []
        rowStates.reserveCapacity(B)
        var stage1Eval: [MLXArray] = []
        stage1Eval.reserveCapacity(B * 4)

        for b in 0 ..< B {
            let posHost = positionsHostRows[b]
            var tokenBlocks = [Int](repeating: -1, count: L)
            var blockTokenCount = [Int](repeating: 0, count: nBlocks)
            for i in 0 ..< L {
                if let blk = blockIndex(
                    forPosition: posHost[i], blockSize: blockSize, nBlocks: nBlocks)
                {
                    tokenBlocks[i] = blk
                    blockTokenCount[blk] += 1
                }
            }
            var interiorBlocks: [Int] = []
            interiorBlocks.reserveCapacity(interiorEnd - interiorStart)
            for blk in interiorStart ..< interiorEnd where blockTokenCount[blk] > 0 {
                interiorBlocks.append(blk)
            }

            let qRow = B == 1 ? queries : queries[b ..< (b + 1)]
            let kRow = B == 1 ? keys : keys[b ..< (b + 1)]
            let qG = qRow.reshaped(nKvHeads, g, L, D)
            let kG = kRow.reshaped(nKvHeads, 1, L, D)
            let kGF32T = kG.asType(.float32).transposed(0, 1, 3, 2)  // [Hk, 1, D, L]

            // 1. Dual-scale frontier-focused forward causal boundary (F8 + F64).
            let qTail = qG[0..., 0..., (L - qtLen)..., 0...].asType(.float32)  // [Hk, G, qtLen, D]
            var logitsF = matmul(qTail, kGF32T) * scale  // [Hk, G, qtLen, L]
            logitsF = MLX.where(invalidCausal, negInf, logitsF)
            let probsF = softmax(logitsF, axis: -1).reshaped(nHeads, qtLen, L)

            let tokF64 = probsF.mean(axis: 1)  // [Hq, L]
            let tokF8 = probsF[0..., (qtLen - f8Len)..., 0...].mean(axis: 1)  // [Hq, L]
            let tokCombo =
                headMeanNormalized(tokF8, eps: eps) + headMeanNormalized(tokF64, eps: eps)
            stage1Eval.append(tokF64)
            stage1Eval.append(tokF8)
            stage1Eval.append(tokCombo)

            // 3. Pure Log-Odds Reverse Causal Prefix Resonance (independent of stage 1 host work;
            // evaluated in the exact same batched `eval` as stage 1).
            var resLO: MLXArray? = nil
            if reverseEligibleGlobal, interiorBlocks.count > 4 {
                let sinkEnd = lowerBound(posHost, target: 4)
                let prefEnd = lowerBound(posHost, target: Int64(reversePrefixEnd))
                let sinkCount = sinkEnd
                let prefCount = max(0, prefEnd - sinkEnd)
                if sinkCount >= 1 && prefCount >= 4 {
                    let qRev = promoteF16 ? qG.asType(.float32) : qG
                    let kRev = promoteF16 ? kRow.asType(.float32) : kRow
                    let qFlat = qRev.reshaped(nKvHeads, g * L, D)
                    let kSink = kRev[0..., 0..., ..<sinkEnd, 0...].reshaped(
                        nKvHeads, sinkCount, D)
                    let kPref = kRev[0..., 0..., sinkEnd ..< prefEnd, 0...].reshaped(
                        nKvHeads, prefCount, D)
                    let dotP =
                        matmul(qFlat, kPref.transposed(0, 2, 1))
                        .reshaped(nHeads, L, prefCount).asType(.float32) * scale
                    let dotS =
                        matmul(qFlat, kSink.transposed(0, 2, 1))
                        .reshaped(nHeads, L, sinkCount).asType(.float32) * scale
                    let lo = sigmoid(
                        logSumExp(dotP, axis: -1) - logSumExp(dotS, axis: -1))  // [Hq, L]
                    resLO = lo
                    stage1Eval.append(lo)
                }
            }

            rowStates.append(
                RowState(
                    qG: qG, kG: kG, kGF32T: kGF32T,
                    tokenBlocks: tokenBlocks, blockTokenCount: blockTokenCount,
                    interiorBlocks: interiorBlocks,
                    tokF64: tokF64, tokF8: tokF8, tokCombo: tokCombo, resLO: resLO))
        }

        // Single batched GPU->CPU sync across all B rows for forward + reverse signals.
        eval(stage1Eval)

        var blkFwdRows = [[Float]](repeating: [Float](repeating: 0, count: nBlocks), count: B)
        var revScoreRows = [[Float]?](repeating: nil, count: B)
        var hop2Tensors = [MLXArray?](repeating: nil, count: B)
        var stage2Eval: [MLXArray] = []
        stage2Eval.reserveCapacity(B)
        let twoHopEligibleGlobal =
            config.twoHopBlockPeaks > 0
            && L > max(0, config.prefixKeep) + max(1, config.tailKeep) + blockSize

        for b in 0 ..< B {
            let state = rowStates[b]
            let f64Host = state.tokF64.asArray(Float.self)
            let f8Host = state.tokF8.asArray(Float.self)
            let tokComboHost = state.tokCombo.asArray(Float.self)

            let (b64Mean, _) = aggregateHeadBalancedBlocks(
                flatHeadToken: f64Host, nHeads: nHeads, tokenCount: L,
                tokenBlocks: state.tokenBlocks, nBlocks: nBlocks,
                interiorBlocks: state.interiorBlocks)
            let (b8Mean, b8Max) = aggregateHeadBalancedBlocks(
                flatHeadToken: f8Host, nHeads: nHeads, tokenCount: L,
                tokenBlocks: state.tokenBlocks, nBlocks: nBlocks,
                interiorBlocks: state.interiorBlocks)

            let zB64 = zScoreBlocks(b64Mean, interiorBlocks: state.interiorBlocks)
            let zB8M = zScoreBlocks(b8Mean, interiorBlocks: state.interiorBlocks)
            let zB8X = zScoreBlocks(b8Max, interiorBlocks: state.interiorBlocks)
            var blkFwd = [Float](repeating: 0, count: nBlocks)
            for blk in 0 ..< nBlocks {
                blkFwd[blk] = max(zB8M[blk], zB64[blk]) + 0.35 * zB8X[blk]
            }
            blkFwdRows[b] = blkFwd

            // 2. Block-diverse 2-hop bidirectional causal adjoint from top-K interior blocks.
            if twoHopEligibleGlobal, !state.interiorBlocks.isEmpty {
                let topBlockSet = Set(
                    rankBlocksDescending(state.interiorBlocks, scores: blkFwd)
                        .prefix(config.twoHopBlockPeaks)
                )
                var bestInBlock: [Int: (score: Float, index: Int32)] = [:]
                bestInBlock.reserveCapacity(topBlockSet.count)
                for i in 0 ..< L {
                    let blk = state.tokenBlocks[i]
                    guard topBlockSet.contains(blk) else { continue }
                    let s = tokComboHost[i]
                    guard s.isFinite else { continue }
                    if let prev = bestInBlock[blk] {
                        if s > prev.score { bestInBlock[blk] = (s, Int32(i)) }
                    } else {
                        bestInBlock[blk] = (s, Int32(i))
                    }
                }
                let peakIndices = bestInBlock.values.map(\.index).sorted()
                if !peakIndices.isEmpty {
                    let pArr = MLXArray(peakIndices)
                    let pCount = peakIndices.count
                    let qpG = state.qG.take(pArr, axis: 2).asType(.float32)  // [Hk, G, P, D]
                    let kpG = state.kG.take(pArr, axis: 2).asType(.float32)  // [Hk, 1, P, D]

                    // Backward hop: peak queries attend earlier keys.
                    var lgB = matmul(qpG, state.kGF32T) * scale  // [Hk, G, P, L]
                    let invalidB = (kIdx .> pArr.expandedDimensions(axis: 1)).expandedDimensions(
                        axes: [0, 1])
                    lgB = MLX.where(invalidB, negInf, lgB)
                    let pBHead = softmax(lgB, axis: -1).reshaped(nHeads, pCount, L).mean(axis: 1)
                    let pB = headMeanNormalized(pBHead, eps: eps)

                    // Forward hop: later queries attend peak keys (zero-copy GQA, [Hk, G, P, L]).
                    var lgF =
                        matmul(kpG, state.qG.asType(.float32).transposed(0, 1, 3, 2)) * scale
                    let invalidF = (kIdx .< pArr.expandedDimensions(axis: 1)).expandedDimensions(
                        axes: [0, 1])
                    lgF = MLX.where(invalidF, negInf, lgF)
                    let pFHead = softmax(lgF, axis: -1).reshaped(nHeads, pCount, L).mean(axis: 1)
                    let pF = headMeanNormalized(pFHead, eps: eps)

                    let hop2Tok = pB + Float(0.5) * pF
                    hop2Tensors[b] = hop2Tok
                    stage2Eval.append(hop2Tok)
                }
            }

            // CPU reciprocal-rank aggregation for reverse prefix resonance.
            if let resLO = state.resLO {
                let resHost = resLO.asArray(Float.self)
                var interiorMask = [Bool](repeating: false, count: nBlocks)
                for blk in state.interiorBlocks { interiorMask[blk] = true }

                var rrMax = [Float](repeating: 0, count: nBlocks)
                var rrSum = [Float](repeating: 0, count: nBlocks)
                var bh = [Float](repeating: 0, count: nBlocks)
                var smoothed = [Float](repeating: 0, count: nBlocks)
                for h in 0 ..< nHeads {
                    for blk in 0 ..< nBlocks { bh[blk] = 0 }
                    let rowBase = h * L
                    for i in 0 ..< L {
                        let blk = state.tokenBlocks[i]
                        if blk >= 0 {
                            let v = resHost[rowBase + i]
                            if v.isFinite { bh[blk] += v }
                        }
                    }
                    for blk in state.interiorBlocks {
                        bh[blk] /= Float(state.blockTokenCount[blk])
                    }
                    let rankSource: [Float]
                    if blockSize <= 32 {
                        for blk in state.interiorBlocks {
                            var acc = Float(0.50) * bh[blk]
                            var w: Float = 0.50
                            if blk > 0 && interiorMask[blk - 1] {
                                acc += Float(0.25) * bh[blk - 1]
                                w += 0.25
                            }
                            if blk + 1 < nBlocks && interiorMask[blk + 1] {
                                acc += Float(0.25) * bh[blk + 1]
                                w += 0.25
                            }
                            smoothed[blk] = acc / w
                        }
                        rankSource = smoothed
                    } else {
                        rankSource = bh
                    }
                    var minV = Float.greatestFiniteMagnitude
                    var maxV = -Float.greatestFiniteMagnitude
                    for blk in state.interiorBlocks {
                        let v = rankSource[blk]
                        if v.isFinite {
                            if v < minV { minV = v }
                            if v > maxV { maxV = v }
                        }
                    }
                    guard maxV > minV else { continue }
                    let ranked = rankBlocksDescending(state.interiorBlocks, scores: rankSource)
                    for (rank, blk) in ranked.enumerated() {
                        let rr = Float(1.0) / Float(4 + rank)
                        if rr > rrMax[blk] { rrMax[blk] = rr }
                        rrSum[blk] += rr
                    }
                }
                var revCombined = [Float](repeating: 0, count: nBlocks)
                for blk in state.interiorBlocks {
                    revCombined[blk] = rrMax[blk] + Float(0.15) * rrSum[blk]
                }
                revScoreRows[b] = revCombined
            }
        }

        // Single batched GPU->CPU sync across all B rows for 2-hop bidirectional adjoint.
        if !stage2Eval.isEmpty {
            eval(stage2Eval)
            for b in 0 ..< B {
                guard let hop2Tok = hop2Tensors[b] else { continue }
                let state = rowStates[b]
                let hop2Host = hop2Tok.asArray(Float.self)
                var blkHop2 = [Float](repeating: 0, count: nBlocks)
                for i in 0 ..< L {
                    let blk = state.tokenBlocks[i]
                    if blk >= 0 {
                        let v = hop2Host[i]
                        if v.isFinite { blkHop2[blk] += v }
                    }
                }
                let zHop2 = zScoreBlocks(blkHop2, interiorBlocks: state.interiorBlocks)
                for blk in 0 ..< nBlocks {
                    blkFwdRows[b][blk] += 0.45 * zHop2[blk]
                }
            }
        }

        return (0 ..< B).map { b in
            Gemma4SDCMBoundaryScores(
                forwardBlockScores: blkFwdRows[b],
                reverseBlockScores: revScoreRows[b],
                reverseQuotaFraction: revScoreRows[b] != nil ? reverseQuotaFraction : 0)
        }
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
        precondition(
            queries.ndim == 4 && keys.ndim == 4 && queries.dim(0) == 1 && keys.dim(0) == 1,
            "Block-SDCM queries/keys shape mismatch")
        precondition(
            positions.ndim == 1 && positions.dim(0) == queries.dim(2),
            "Block-SDCM boundary scoring positions length mismatch")
        return computeBilateralBlockScoresBatched(
            queries: queries,
            keys: keys,
            positions: positions.expandedDimensions(axis: 0),
            positionsHostRows: [positionsHost],
            originalLength: originalLength,
            scale: scale,
            nHeads: nHeads,
            nKvHeads: nKvHeads,
            reverseQuotaFraction: reverseQuotaFraction,
            config: config)[0]
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

    /// Ensure every row in a batched `B >= 1` compaction selection retains the exact same
    /// surviving token count `targetLen` (`<= min(currentLength, rowMaxLengths.min())`) so
    /// `[B, targetLen, D]` stays rectangular without ever selecting right-pad slots. Pads
    /// shorter rows with highest-scoring unkept tokens and trims longer rows by dropping
    /// lowest-scoring interior tokens while keeping pinned prefix and frontier tail tokens.
    static func equalizeBatchedCompactedIndices(
        keepRows: [[Int32]],
        rowBlockScores: [[Float]],
        positionsHostRows: [[Int32]],
        originalLengths: [Int],
        rowMaxLengths: [Int],
        blockSize: Int,
        currentLength: Int
    ) -> [[Int32]] {
        let B = keepRows.count
        guard B > 1, currentLength > 0 else { return keepRows }
        var minLen = Int.max
        var maxLen = 0
        for row in keepRows {
            if row.count < minLen { minLen = row.count }
            if row.count > maxLen { maxLen = row.count }
        }
        let minValidRowCap = rowMaxLengths.min() ?? currentLength
        let targetLen = max(1, min(min(currentLength, minValidRowCap), maxLen))
        if minLen == maxLen && maxLen <= targetLen {
            return keepRows
        }
        let bs = max(1, blockSize)

        func rankByBlockScore<S: Sequence>(
            _ indices: S,
            posHost: [Int32],
            scores: [Float],
            nBlocks: Int
        ) -> [Int32] where S.Element == Int32 {
            var ranked = indices.map { i32 -> (index: Int32, block: Int, score: Float) in
                let i = Int(i32)
                let blk =
                    i < posHost.count
                    ? (blockIndex(forPosition: posHost[i], blockSize: bs, nBlocks: nBlocks) ?? -1)
                    : -1
                let rawScore = (blk >= 0 && blk < scores.count) ? scores[blk] : -Float.infinity
                let score = rawScore.isFinite ? rawScore : -Float.infinity
                return (index: i32, block: blk, score: score)
            }
            ranked.sort { lhs, rhs in
                if lhs.score != rhs.score { return lhs.score > rhs.score }
                if lhs.block != rhs.block { return lhs.block < rhs.block }
                return lhs.index < rhs.index
            }
            return ranked.map(\.index)
        }

        var out = keepRows
        for b in 0 ..< B {
            let posHost = positionsHostRows[b]
            let rowOrigLen = b < originalLengths.count ? originalLengths[b] : currentLength
            let rowCap = min(
                currentLength, b < rowMaxLengths.count ? rowMaxLengths[b] : currentLength)
            let nBlocks = blockCount(originalLength: rowOrigLen, blockSize: bs)
            let scores = b < rowBlockScores.count ? rowBlockScores[b] : []
            if out[b].count < targetLen {
                let need = targetLen - out[b].count
                let keptSet = Set(out[b])
                let unkept = (0 ..< rowCap).lazy.map(Int32.init).filter { !keptSet.contains($0) }
                let ranked = rankByBlockScore(
                    unkept, posHost: posHost, scores: scores, nBlocks: nBlocks)
                var merged = out[b]
                merged.reserveCapacity(targetLen)
                merged.append(contentsOf: ranked.prefix(need))
                merged.sort()
                out[b] = merged
            } else if out[b].count > targetLen {
                // Trim lowest-scoring interior tokens while keeping pinned prefix and frontier tail.
                let rowSorted = out[b].sorted()
                let tailPin = min(targetLen, max(1, min(rowSorted.count, bs)))
                let prefixPin = min(targetLen - tailPin, min(rowSorted.count - tailPin, bs))
                let headSlice = rowSorted.prefix(prefixPin)
                let tailSlice = rowSorted.suffix(tailPin)
                let interiorBudget = max(0, targetLen - headSlice.count - tailSlice.count)
                var trimmed: [Int32] = []
                trimmed.reserveCapacity(targetLen)
                trimmed.append(contentsOf: headSlice)
                if interiorBudget > 0 && rowSorted.count > prefixPin + tailPin {
                    let rankedInterior = rankByBlockScore(
                        rowSorted[prefixPin ..< (rowSorted.count - tailPin)],
                        posHost: posHost, scores: scores, nBlocks: nBlocks)
                    trimmed.append(contentsOf: rankedInterior.prefix(interiorBudget))
                }
                trimmed.append(contentsOf: tailSlice)
                trimmed.sort()
                out[b] = trimmed
            }
        }
        return out
    }

    /// Batched Early Tri-Signal Block Pre-Compaction for `B >= 1` rows, coalescing all `B`
    /// rows' GPU tensors into one `eval(...)` per stage (MaxSim + Novelty, then 2-Hop MaxSim).
    static func selectEarlyTriSignalIndicesBatched(
        tokenIDRows: [[Int32]?],
        embeddingHidden: MLXArray,
        layerHidden: MLXArray,
        targetFraction: Float,
        blockSize: Int,
        topHaloK: Int,
        config: Gemma4BlockSDCMConfig
    ) -> (keepRows: [[Int32]], unionBlockScores: [[Float]]) {
        precondition(
            embeddingHidden.ndim == 3 && embeddingHidden.dim(0) >= 1
                && layerHidden.ndim == 3 && layerHidden.dim(0) == embeddingHidden.dim(0)
                && embeddingHidden.dim(1) == layerHidden.dim(1)
                && embeddingHidden.dim(2) == layerHidden.dim(2)
                && tokenIDRows.count == embeddingHidden.dim(0),
            "Block-SDCM early pre-compaction expects matching [B, T, D] tensors")
        let B = embeddingHidden.dim(0)
        let T = embeddingHidden.dim(1)
        let bs = max(1, blockSize)
        let nBlocks = blockCount(originalLength: T, blockSize: bs)
        let zeroScores = Array(repeating: [Float](repeating: 0, count: nBlocks), count: B)
        guard T > 0 else { return (Array(repeating: [], count: B), zeroScores) }
        let allIndices = (0 ..< T).map { Int32($0) }
        guard embeddingHidden.dim(2) > 0 else {
            return (Array(repeating: allIndices, count: B), zeroScores)
        }
        let (prefBlocks, tailStartBlock) = pinnedBlockBounds(
            nBlocks: nBlocks, blockSize: bs, config: config)
        let interiorStart = min(prefBlocks, nBlocks)
        let interiorEnd = max(interiorStart, tailStartBlock)
        let interiorBlocks = Array(interiorStart ..< interiorEnd)
        guard interiorBlocks.count > 4 else {
            return (Array(repeating: allIndices, count: B), zeroScores)
        }

        let clampedFraction = targetFraction.isFinite ? max(0, min(1, targetFraction)) : 1
        let minKeptTokens = max(0, config.prefixKeep) + max(1, config.tailKeep)
        let desiredTokens = max(minKeptTokens, Int((Float(T) * clampedFraction).rounded()))
        if desiredTokens >= T {
            return (Array(repeating: allIndices, count: B), zeroScores)
        }
        let s0 = min(T, interiorStart * bs)
        let s1 = min(T, interiorEnd * bs)
        let pinnedTokens = s0 + (T - s1)
        let needInteriorBlocks = max(4, (max(0, desiredTokens - pinnedTokens) + bs - 1) / bs)
        if needInteriorBlocks >= interiorBlocks.count {
            return (Array(repeating: allIndices, count: B), zeroScores)
        }

        let eps = Float(1e-6)
        let epsStd = Float(1e-5)
        let useBF16Matmul = embeddingHidden.dtype == .bfloat16
        let p0 = min(4, T)
        let p1 = max(p0, min(160, T / 4))
        let tailStartTok = max(p1, T - max(1, config.tailKeep))
        let pad = nBlocks * bs - T
        let dim = embeddingHidden.dim(2)
        let padZeros: MLXArray? = pad > 0 ? MLXArray.zeros([pad, dim], dtype: .float32) : nil

        var u0Mats: [MLXArray] = []
        u0Mats.reserveCapacity(B)
        var tokMSTensors: [MLXArray?] = []
        tokMSTensors.reserveCapacity(B)
        var simMatTensors: [MLXArray] = []
        simMatTensors.reserveCapacity(B)
        var stage1Eval: [MLXArray] = []
        stage1Eval.reserveCapacity(B * 2)

        for b in 0 ..< B {
            // 1. L0_emb Centered ColBERT MaxSim (prefix U tail).
            let h0 = embeddingHidden[b].asType(.float32)  // [T, D]
            let u0F32 = h0 / (sqrt((h0 * h0).sum(axis: -1, keepDims: true)) + eps)
            let u0Mat = useBF16Matmul ? u0F32.asType(.bfloat16) : u0F32
            u0Mats.append(u0Mat)

            var qParts: [MLXArray] = []
            if p1 > p0 { qParts.append(u0Mat[p0 ..< p1]) }
            if tailStartTok < T { qParts.append(u0Mat[tailStartTok ..< T]) }
            if !qParts.isEmpty && s1 > s0 {
                let uQ = qParts.count == 1 ? qParts[0] : concatenated(qParts, axis: 0)  // [Q, D]
                let sim = matmul(u0Mat, uQ.transposed(1, 0)).asType(.float32)  // [T, Q]
                let simInt = sim[s0 ..< s1]
                let muQ = simInt.mean(axis: 0, keepDims: true)
                let diffQ = simInt - muQ
                let sdQ = sqrt((diffQ * diffQ).mean(axis: 0, keepDims: true)) + epsStd
                let tokMS = ((sim - muQ) / sdQ).max(axis: -1)  // [T]
                tokMSTensors.append(tokMS)
                stage1Eval.append(tokMS)
            } else {
                tokMSTensors.append(nil)
            }

            // 2. Post-L0 `C @ C^T` Block-Centroid Outlier Novelty (`[nBlocks, nBlocks]`).
            let hc = layerHidden[b].asType(.float32)  // [T, D]
            let hcPadded = padZeros.map { concatenated([hc, $0], axis: 0) } ?? hc
            let centroidsRaw = hcPadded.reshaped(nBlocks, bs, dim).mean(axis: 1)  // [nBlocks, D]
            let centroids =
                centroidsRaw
                / (sqrt((centroidsRaw * centroidsRaw).sum(axis: -1, keepDims: true)) + eps)
            let simMat = matmul(centroids, centroids.transposed(1, 0))  // [nBlocks, nBlocks]
            simMatTensors.append(simMat)
            stage1Eval.append(simMat)
        }

        // Single batched GPU->CPU sync for MaxSim + Novelty across all B rows.
        eval(stage1Eval)

        var blkMSRows = [[Float]](repeating: [Float](repeating: 0, count: nBlocks), count: B)
        var tokHopTensors = [MLXArray?](repeating: nil, count: B)
        var stage2Eval: [MLXArray] = []
        stage2Eval.reserveCapacity(B)
        let tokBlkArr = MLXArray((0 ..< T).map { Int32($0 / bs) }).expandedDimensions(axis: 1)

        for b in 0 ..< B {
            guard let tokMS = tokMSTensors[b] else { continue }
            let tokMSHost = tokMS.asArray(Float.self)
            var blkMS = [Float](repeating: 0, count: nBlocks)
            for blk in interiorBlocks {
                let lo = blk * bs
                let hi = min(T, lo + bs)
                if hi > lo {
                    blkMS[blk] = topKMean(tokMSHost[lo ..< hi], k: 4)
                }
            }
            blkMSRows[b] = blkMS

            // 2-Hop MaxSim from top-24 interior blocks (2 peak tokens each).
            let topHopBlocks = rankBlocksDescending(interiorBlocks, scores: blkMS).prefix(24)
            var hopIdx: [Int32] = []
            hopIdx.reserveCapacity(topHopBlocks.count * 2)
            for blk in topHopBlocks {
                let lo = blk * bs
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
                let u0Mat = u0Mats[b]
                let uHop = u0Mat.take(MLXArray(hopIdx), axis: 0)  // [P, D]
                let simH = matmul(u0Mat, uHop.transposed(1, 0)).asType(.float32)  // [T, P]
                let simHInt = simH[s0 ..< s1]
                let muH = simHInt.mean(axis: 0, keepDims: true)
                let diffH = simHInt - muH
                let sdH = sqrt((diffH * diffH).mean(axis: 0, keepDims: true)) + epsStd
                var zH = (simH - muH) / sdH  // [T, P]
                // Mask out self-block matches so 2-hop discovers cross-block links.
                let hopBlkArr = MLXArray(hopIdx.map { Int32(Int($0) / bs) }).expandedDimensions(
                    axis: 0)
                zH = MLX.where(tokBlkArr .== hopBlkArr, MLXArray(-Float(10.0)), zH)
                let tokHop = zH.max(axis: -1)
                tokHopTensors[b] = tokHop
                stage2Eval.append(tokHop)
            }
        }

        // Single batched GPU->CPU sync for 2-Hop MaxSim across all B rows.
        if !stage2Eval.isEmpty {
            eval(stage2Eval)
        }

        let interiorSet = Set(interiorBlocks)
        var keepRows: [[Int32]] = []
        keepRows.reserveCapacity(B)
        var unionScoresRows: [[Float]] = []
        unionScoresRows.reserveCapacity(B)
        var others: [Float] = []
        others.reserveCapacity(interiorBlocks.count)

        for b in 0 ..< B {
            var blkHop = [Float](repeating: 0, count: nBlocks)
            if let tokHop = tokHopTensors[b] {
                let tokHopHost = tokHop.asArray(Float.self)
                for blk in interiorBlocks {
                    let lo = blk * bs
                    let hi = min(T, lo + bs)
                    if hi > lo {
                        blkHop[blk] = topKMean(tokHopHost[lo ..< hi], k: 4)
                    }
                }
            }

            let simHost = simMatTensors[b].asArray(Float.self)
            var blkNov = [Float](repeating: 0, count: nBlocks)
            for blk in interiorBlocks {
                others.removeAll(keepingCapacity: true)
                let rowBase = blk * nBlocks
                for blk2 in interiorBlocks where abs(blk2 - blk) > 1 {
                    let s = simHost[rowBase + blk2]
                    if s.isFinite { others.append(s) }
                }
                if !others.isEmpty {
                    blkNov[blk] = -topKMean(others[...], k: 8)
                }
            }

            // 3. CPU Document-IDF Token Rarity (`1 / doc_freq(tokID)`, top-8 rare tokens per block).
            var blkIDF = [Float](repeating: 0, count: nBlocks)
            let hasTokenIDF: Bool
            if let tokenIDs = tokenIDRows[b], tokenIDs.count == T, s1 > s0 {
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
                for blk in interiorBlocks {
                    let lo = blk * bs
                    let hi = min(T, lo + bs)
                    if hi > lo {
                        blkIDF[blk] = topKMean(tokIDF[lo ..< hi], k: 8)
                    }
                }
                hasTokenIDF = true
            } else {
                hasTokenIDF = false
            }

            let zMS = zScoreBlocks(blkMSRows[b], interiorBlocks: interiorBlocks)
            let zHop = zScoreBlocks(blkHop, interiorBlocks: interiorBlocks)
            var qScore = [Float](repeating: 0, count: nBlocks)
            for blk in 0 ..< nBlocks { qScore[blk] = zMS[blk] + Float(0.50) * zHop[blk] }
            let novScore = zScoreBlocks(blkNov, interiorBlocks: interiorBlocks)
            let idfScore =
                hasTokenIDF
                ? zScoreBlocks(blkIDF, interiorBlocks: interiorBlocks)
                : [Float](repeating: 0, count: nBlocks)
            var unionScore = [Float](repeating: 0, count: nBlocks)
            for blk in interiorBlocks {
                unionScore[blk] =
                    hasTokenIDF
                    ? max(qScore[blk], max(novScore[blk], idfScore[blk]))
                    : max(qScore[blk], novScore[blk])
            }
            unionScoresRows.append(unionScore)

            var keptBlocks = Set<Int>()
            keptBlocks.reserveCapacity((nBlocks - interiorBlocks.count) + needInteriorBlocks)
            for blk in 0 ..< interiorStart { keptBlocks.insert(blk) }
            for blk in interiorEnd ..< nBlocks { keptBlocks.insert(blk) }
            let pinnedBlockCount = keptBlocks.count

            // 0. Targeted ±1 neighbor halo ONLY on top-K sharpest query hits (zero halo bloat).
            if topHaloK > 0 {
                for blk in rankBlocksDescending(interiorBlocks, scores: qScore).prefix(topHaloK) {
                    guard keptBlocks.count - pinnedBlockCount < needInteriorBlocks else { break }
                    keptBlocks.insert(blk)
                    if keptBlocks.count - pinnedBlockCount < needInteriorBlocks,
                        interiorSet.contains(blk + 1)
                    {
                        keptBlocks.insert(blk + 1)
                    }
                    if keptBlocks.count - pinnedBlockCount < needInteriorBlocks,
                        interiorSet.contains(blk - 1)
                    {
                        keptBlocks.insert(blk - 1)
                    }
                }
            }

            // 1. 4-strata diversity floor (10% of interior budget).
            if config.strataCount > 0, keptBlocks.count - pinnedBlockCount < needInteriorBlocks {
                let strataPerSeg = max(
                    1, Int(Float(needInteriorBlocks) * 0.10) / config.strataCount)
                for s in 0 ..< config.strataCount {
                    let avail = needInteriorBlocks - (keptBlocks.count - pinnedBlockCount)
                    guard avail > 0 else { break }
                    let lo = s * interiorBlocks.count / config.strataCount
                    let hi = (s + 1) * interiorBlocks.count / config.strataCount
                    guard lo < hi else { continue }
                    let seg = rankBlocksDescending(
                        interiorBlocks[lo ..< hi], scores: qScore, excluding: keptBlocks)
                    for blk in seg.prefix(min(strataPerSeg, avail)) { keptBlocks.insert(blk) }
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
                for blk in cands.prefix(quota) { keptBlocks.insert(blk) }
            }

            // 3. Fill any rounding residual using union max across all three signals.
            let left = max(0, needInteriorBlocks - (keptBlocks.count - pinnedBlockCount))
            if left > 0 {
                let cands = rankBlocksDescending(
                    interiorBlocks, scores: unionScore, excluding: keptBlocks)
                for blk in cands.prefix(left) { keptBlocks.insert(blk) }
            }

            var selected: [Int32] = []
            selected.reserveCapacity(min(T, keptBlocks.count * bs))
            for i in 0 ..< T {
                if keptBlocks.contains(i / bs) {
                    selected.append(Int32(i))
                }
            }
            keepRows.append(selected)
        }
        return (keepRows, unionScoresRows)
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
                && layerHidden.ndim == 3 && layerHidden.dim(0) == 1,
            "Block-SDCM early pre-compaction expects matching [1, T, D] tensors")
        return selectEarlyTriSignalIndicesBatched(
            tokenIDRows: [tokenIDs],
            embeddingHidden: embeddingHidden,
            layerHidden: layerHidden,
            targetFraction: targetFraction,
            blockSize: blockSize,
            topHaloK: topHaloK,
            config: config
        ).keepRows[0]
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
    /// Q-norm, transpose, and RoPE pre-projected queries `[B, qLength, nHeads * headDim]`
    /// (using zero-transpose scalar offset when contiguous from 0).
    private func normalizeQueriesWithRoPE(
        _ qFlat: MLXArray, positions: MLXArray, isContiguousFromZero: Bool
    ) -> MLXArray {
        let attn = selfAttn
        let B = qFlat.dim(0)
        let qLength = qFlat.dim(1)
        var q = qFlat.reshaped(B, qLength, attn.nHeads, attn.effectiveHeadDim)
        q = attn.qNorm(q).transposed(0, 2, 1, 3)
        if isContiguousFromZero {
            return gemma4ApplyRotaryPosition(attn.rope, to: q, offset: .scalar(0))
        }
        return Gemma4BlockSDCM.applyArbitraryPositionsRoPE(attn.rope, to: q, positions: positions)
    }

    /// Run per-row attention while keeping `B == 1` zero-slice and zero-concat.
    private static func attendPerRow(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        _ attend: (_ row: Int, _ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray
    ) -> MLXArray {
        let B = queries.dim(0)
        if B == 1 { return attend(0, queries, keys, values) }
        var rowOuts: [MLXArray] = []
        rowOuts.reserveCapacity(B)
        for b in 0 ..< B {
            rowOuts.append(
                attend(b, queries[b ..< (b + 1)], keys[b ..< (b + 1)], values[b ..< (b + 1)]))
        }
        return concatenated(rowOuts, axis: 0)
    }

    /// Reshape SDPA output `[B, H, qL, D]`, restore original activation dtype, and project via `oProj`.
    private func finishAttentionOutput(
        _ attnResult: MLXArray, length: Int, outDType: DType
    ) -> MLXArray {
        let B = attnResult.dim(0)
        var attnOut = attnResult.transposed(0, 2, 1, 3).reshaped(B, length, -1)
        if attnOut.dtype != outDType { attnOut = attnOut.asType(outDType) }
        return selfAttn.oProj(attnOut)
    }

    /// Commit compacted K/V to the layer cache with logical RoPE delta for `B >= 1` rows.
    /// Sliding layers store ONLY tokens inside each row's true original-position sliding window
    /// (`origPos >= originalLength - slidingWindow + 1` when compacted) so decode queries
    /// never attend to out-of-window sliding RoPE distances (`deltaPos >= W`).
    private func commitCompactedKV(
        keys k: MLXArray,
        values v: MLXArray,
        positionsHostRows: [[Int32]],
        originalLengths: [Int],
        hasActiveRightPad: Bool,
        cache: KVCache
    ) {
        let attn = selfAttn
        let B = k.dim(0)
        let L = k.dim(2)
        let hasHeterogeneousLengths =
            B > 1 && originalLengths.contains { $0 != originalLengths[0] }

        if let v2Cache = cache as? (any CBv2AttendingLayerCache),
            hasActiveRightPad || hasHeterogeneousLengths || (B > 1 && attn.isSliding)
        {
            var perRowK: [MLXArray] = []
            var perRowV: [MLXArray] = []
            var deltas: [Int] = []
            perRowK.reserveCapacity(B)
            perRowV.reserveCapacity(B)
            deltas.reserveCapacity(B)
            for b in 0 ..< B {
                let rowOrigLen = originalLengths[b]
                let rowEnd = hasActiveRightPad ? min(L, rowOrigLen) : L
                let start: Int
                if attn.isSliding {
                    let posSlice =
                        rowEnd < positionsHostRows[b].count
                        ? Array(positionsHostRows[b][0 ..< rowEnd]) : positionsHostRows[b]
                    start = min(
                        Gemma4BlockSDCM.trueSlidingWindowStart(
                            positionsHost: posSlice,
                            originalLength: rowOrigLen,
                            slidingWindow: attn.config.slidingWindow),
                        rowEnd)
                } else {
                    start = 0
                }
                let rowK = (start > 0 || rowEnd < L) ? k[b, 0..., start ..< rowEnd, 0...] : k[b]
                let rowV = (start > 0 || rowEnd < L) ? v[b, 0..., start ..< rowEnd, 0...] : v[b]
                let stored = rowEnd - start
                perRowK.append(rowK)
                perRowV.append(rowV)
                deltas.append(rowOrigLen - stored)
            }
            v2Cache.writeCompactedPromptKV(
                perRowKeys: perRowK, perRowValues: perRowV, logicalDeltas: deltas)
            return
        }

        let originalLength = originalLengths[0]
        let v2Cache = cache as? (any CBv2AttendingLayerCache)
        if v2Cache == nil {
            precondition(
                B == 1,
                "Block-SDCM batched compaction (B=\(B)) requires CBv2AttendingLayerCache, got \(type(of: cache))"
            )
        }

        let kStore: MLXArray
        let vStore: MLXArray
        if attn.isSliding {
            let start = Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: positionsHostRows[0],
                originalLength: originalLength,
                slidingWindow: attn.config.slidingWindow)
            if start > 0 {
                let clampedStart = min(start, L)
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
        if let v2Cache {
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

    /// Execute one decoder layer inside the Block-SDCM prefill trunk for `B >= 1` rows
    /// (supporting both equal-length and right-padded heterogeneous cohorts).
    func forwardBlockSDCM(
        _ x: MLXArray,
        positions: MLXArray,
        positionsHostRows: [[Int32]],
        paddedLength: Int,
        originalLengths: [Int],
        hasActiveRightPad: Bool,
        cache: KVCache?,
        sdcm: Gemma4BlockSDCMConfig,
        isFinalLayer: Bool,
        nextReverseQuota: Float?,
        isExpertPrefill: Bool
    ) -> (hidden: MLXArray, nextLayerScores: [Gemma4SDCMBoundaryScores]?) {
        let B = x.dim(0)
        let L = x.dim(1)
        let isContiguous = L == paddedLength
        let attn = selfAttn
        guard let kProj = attn.kProj, let kNorm = attn.kNorm, let vNorm = attn.vNorm else {
            preconditionFailure("Block-SDCM requires non-shared attention layers")
        }

        let normed = inputLayernorm(x)

        // Non-final layers project Q/K/V in one fused QKV call; the final layer
        // projects K/V for all L tokens and Q only for the frontier token `L - 1`.
        let kRaw: MLXArray
        let vRaw: MLXArray
        let qPreProjected: MLXArray?
        if isFinalLayer {
            kRaw = kProj(normed).reshaped(B, L, attn.nKvHeads, attn.effectiveHeadDim)
            if let vProj = attn.vProj {
                vRaw = vProj(normed).reshaped(B, L, attn.nKvHeads, attn.effectiveHeadDim)
            } else {
                vRaw = kRaw
            }
            qPreProjected = nil
        } else {
            let (qFlat, kFlat, vFlat) = attn.projectQKV(normed, useBatchDecodeProjection: false)
            qPreProjected = qFlat
            kRaw = kFlat.reshaped(B, L, attn.nKvHeads, attn.effectiveHeadDim)
            vRaw = vFlat.reshaped(B, L, attn.nKvHeads, attn.effectiveHeadDim)
        }

        var k = kNorm(kRaw).transposed(0, 2, 1, 3)
        if isContiguous {
            k = gemma4ApplyRotaryPosition(attn.rope, to: k, offset: .scalar(0))
        } else {
            k = Gemma4BlockSDCM.applyArbitraryPositionsRoPE(attn.rope, to: k, positions: positions)
        }
        let v = vNorm(vRaw).transposed(0, 2, 1, 3)

        if let cache {
            commitCompactedKV(
                keys: k, values: v, positionsHostRows: positionsHostRows,
                originalLengths: originalLengths, hasActiveRightPad: hasActiveRightPad,
                cache: cache)
        }

        let outDType = k.dtype
        let attnK = outDType == .float16 ? k.asType(.float32) : k
        let attnV = outDType == .float16 ? v.asType(.float32) : v

        // Final-layer last-query specialization: project Q, attend, O-proj, MLP, and exact
        // top-8 MoE for ONLY the final frontier token across all B rows.
        if isFinalLayer {
            precondition(
                !hasActiveRightPad,
                "Block-SDCM right-padded tokens must be compacted before the final layer")
            let qLast = normalizeQueriesWithRoPE(
                attn.qProj(normed[0..., (L - 1)..., 0...]),
                positions: positions[0..., (L - 1)...],
                isContiguousFromZero: false)
            let attnQ = outDType == .float16 ? qLast.asType(.float32) : qLast
            let attnResult: MLXArray
            if attn.isSliding {
                let windowI32 = Int32(attn.config.slidingWindow)
                attnResult = Self.attendPerRow(queries: attnQ, keys: attnK, values: attnV) {
                    b, qRow, kRow, vRow in
                    let posB = positions[b]
                    let diff =
                        posB[(L - 1)...].expandedDimensions(axis: 1)
                        - posB.expandedDimensions(axis: 0)
                    let rowMask: MLXFast.ScaledDotProductAttentionMaskMode = .array(
                        (diff .>= Int32(0)) .&& (diff .< windowI32))
                    return MLXFast.scaledDotProductAttention(
                        queries: qRow, keys: kRow, values: vRow, scale: attn.scale, mask: rowMask)
                }
            } else {
                attnResult = MLXFast.scaledDotProductAttention(
                    queries: attnQ, keys: attnK, values: attnV, scale: attn.scale, mask: .none)
            }
            let attnOut = finishAttentionOutput(attnResult, length: 1, outDType: outDType)

            var h = x[0..., (L - 1)..., 0...] + postAttentionLayernorm(attnOut)
            h = feedforward(h, isExpertPrefill: false, allowDecodeMoETopK: false)
            h = h * layerScalar
            return (h, nil)
        }

        // Standard layer Q norm + RoPE batched across all B rows (reuses fused Q projection).
        let q = normalizeQueriesWithRoPE(
            qPreProjected!, positions: positions, isContiguousFromZero: isContiguous)
        let attnQ = outDType == .float16 ? q.asType(.float32) : q

        let attnResult: MLXArray
        var nextScores: [Gemma4SDCMBoundaryScores]? = nil
        if attn.isSliding {
            attnResult = Self.attendPerRow(queries: attnQ, keys: attnK, values: attnV) {
                b, qRow, kRow, vRow in
                Gemma4BlockSDCM.slidingAttentionWithPositions(
                    queries: qRow,
                    keys: kRow,
                    values: vRow,
                    positions: positions[b],
                    positionsHost: positionsHostRows[b],
                    slidingWindow: attn.config.slidingWindow,
                    scale: attn.scale,
                    chunkSize: sdcm.slidingChunkSize)
            }
        } else {
            if sdcm.earlySparseFullLayer == attn.layerIdx {
                attnResult = Self.attendPerRow(queries: attnQ, keys: attnK, values: attnV) {
                    b, qRow, kRow, vRow in
                    if hasActiveRightPad, originalLengths[b] < L {
                        let realLen = originalLengths[b]
                        let posB = positions[b, 0 ..< realLen]
                        let posHostB = Array(positionsHostRows[b][0 ..< realLen])
                        let realOut = Gemma4BlockSDCM.prefixLocalFullTailAttention(
                            queries: qRow[0..., 0..., 0 ..< realLen, 0...],
                            keys: kRow[0..., 0..., 0 ..< realLen, 0...],
                            values: vRow[0..., 0..., 0 ..< realLen, 0...],
                            positions: posB,
                            positionsHost: posHostB,
                            prefixTokens: sdcm.sparsePrefixTokens,
                            localWindow: sdcm.sparseLocalWindow,
                            tailTokens: sdcm.sparseTailTokens,
                            scale: attn.scale,
                            chunkSize: sdcm.slidingChunkSize)
                        let padOut = MLXArray.zeros(
                            [1, realOut.dim(1), L - realLen, realOut.dim(3)], dtype: realOut.dtype)
                        return concatenated([realOut, padOut], axis: 2)
                    }
                    return Gemma4BlockSDCM.prefixLocalFullTailAttention(
                        queries: qRow,
                        keys: kRow,
                        values: vRow,
                        positions: positions[b],
                        positionsHost: positionsHostRows[b],
                        prefixTokens: sdcm.sparsePrefixTokens,
                        localWindow: sdcm.sparseLocalWindow,
                        tailTokens: sdcm.sparseTailTokens,
                        scale: attn.scale,
                        chunkSize: sdcm.slidingChunkSize)
                }
            } else {
                // Kept positions are strictly increasing per row, so `pos[k] <= pos[q]` is
                // identical to index-causal order -> batched fused FlashAttention `.causal`.
                attnResult = MLXFast.scaledDotProductAttention(
                    queries: attnQ, keys: attnK, values: attnV, scale: attn.scale, mask: .causal)
            }
            if let reverseQuota = nextReverseQuota {
                let hasHeterogeneousLengths =
                    B > 1 && originalLengths.contains { $0 != originalLengths[0] }
                if !hasActiveRightPad && !hasHeterogeneousLengths {
                    nextScores = Gemma4BlockSDCM.computeBilateralBlockScoresBatched(
                        queries: attnQ,
                        keys: attnK,
                        positions: positions,
                        positionsHostRows: positionsHostRows,
                        originalLength: originalLengths[0],
                        scale: attn.scale,
                        nHeads: attn.nHeads,
                        nKvHeads: attn.nKvHeads,
                        reverseQuotaFraction: reverseQuota,
                        config: sdcm)
                } else {
                    nextScores = (0 ..< B).map { b in
                        let rowEnd = hasActiveRightPad ? min(L, originalLengths[b]) : L
                        let qSlice =
                            rowEnd < L
                            ? attnQ[b ..< (b + 1), 0..., 0 ..< rowEnd, 0...]
                            : attnQ[b ..< (b + 1)]
                        let kSlice =
                            rowEnd < L
                            ? attnK[b ..< (b + 1), 0..., 0 ..< rowEnd, 0...]
                            : attnK[b ..< (b + 1)]
                        let posSlice = rowEnd < L ? positions[b, 0 ..< rowEnd] : positions[b]
                        let posHostSlice =
                            rowEnd < positionsHostRows[b].count
                            ? Array(positionsHostRows[b][0 ..< rowEnd]) : positionsHostRows[b]
                        return Gemma4BlockSDCM.computeBilateralBlockScores(
                            queries: qSlice,
                            keys: kSlice,
                            positions: posSlice,
                            positionsHost: posHostSlice,
                            originalLength: originalLengths[b],
                            scale: attn.scale,
                            nHeads: attn.nHeads,
                            nKvHeads: attn.nKvHeads,
                            reverseQuotaFraction: reverseQuota,
                            config: sdcm)
                    }
                }
            }
        }

        let attnOut = finishAttentionOutput(attnResult, length: L, outDType: outDType)
        let ffIn = x + postAttentionLayernorm(attnOut)
        let interiorMoEK = sdcm.interiorMoEK(forLayer: attn.layerIdx)
        let effectiveTailKeep: Int
        if hasActiveRightPad, let minOrig = originalLengths.min(), minOrig < L {
            // Ensure every shorter row's true tail `[minOrig - tailKeep ..< minOrig)`
            // stays in the exact top-8 MoE suffix during uncompacted early layers.
            effectiveTailKeep = min(L, (L - minOrig) + sdcm.tailKeep)
        } else {
            effectiveTailKeep = sdcm.tailKeep
        }
        var h = feedforwardBlockSDCM(
            ffIn,
            interiorMoEK: interiorMoEK,
            tailKeep: effectiveTailKeep,
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

        let h2: MLXArray
        if interiorMoEK < topK, L > effectiveTailKeep {
            let cut = L - effectiveTailKeep
            let interiorIn = input[0..., ..<cut, 0...]
            let tailIn = input[0..., cut..., 0...]

            // Interior tokens: direct top-k argPartition with exact softmax renormalization.
            let (prunedIndices, prunedWeights) = router.routeTopK(interiorIn, topK: interiorMoEK)
            let h2IntRaw = experts(
                preFeedforwardLayernorm2(interiorIn),
                topKIndices: prunedIndices,
                topKWeights: prunedWeights,
                isExpertPrefill: isExpertPrefill,
                allowVariableTopKRowMap: true)
            let h2Int = postFeedforwardLayernorm2(h2IntRaw)

            // Tail tokens: exact top-8 MoE (preserves question & frontier representations).
            let (tailIndices, tailWeights) = router(tailIn)
            let h2TailRaw = experts(
                preFeedforwardLayernorm2(tailIn),
                topKIndices: tailIndices,
                topKWeights: tailWeights,
                isExpertPrefill: isExpertPrefill,
                allowVariableTopKRowMap: true)
            let h2Tail = postFeedforwardLayernorm2(h2TailRaw)

            h2 = concatenated([h2Int, h2Tail], axis: 1)
        } else {
            // Full top-8 Block-SDCM layers (L12..L29) still route through
            // `allowVariableTopKRowMap: true` so large compacted/batched token
            // counts (> 2048) stay on the NAX row-mapped expert kernel.
            let (indices, weights) = router(input)
            let h2Raw = experts(
                preFeedforwardLayernorm2(input),
                topKIndices: indices,
                topKWeights: weights,
                isExpertPrefill: isExpertPrefill,
                allowVariableTopKRowMap: true)
            h2 = postFeedforwardLayernorm2(h2Raw)
        }

        let out = postFeedforwardLayernorm(h1 + h2)
        return residual2 + out
    }
}

extension Gemma4TextModelInner {
    /// Execute the full Gemma 4 trunk using Block-SDCM progressive block compaction for `B >= 1`
    /// equal-length or right-padded heterogeneous cold text prompts. Returns `[B, 1, hiddenSize]`
    /// (the final frontier positions after `norm`).
    func forwardBlockSDCMTrunk(
        _ initialHidden: MLXArray,
        inputTokenIDs: [Int32]? = nil,
        inputTokenIDRows: [[Int32]]? = nil,
        originalLengths: [Int]? = nil,
        fullCache: [KVCache?],
        sdcm: Gemma4BlockSDCMConfig,
        isExpertPrefill: Bool,
        captureCheckpointAtLayer: Int? = nil,
        dflashTap: DFlashContextTap? = nil,
        onDFlashContext: ((MLXArray) -> Void)? = nil
    ) -> (postNorm: MLXArray, checkpoint: Gemma4SDCMCheckpoint?) {
        let B = initialHidden.dim(0)
        let paddedLength = initialHidden.dim(1)
        let origLens =
            if let originalLengths, originalLengths.count == B {
                originalLengths
            } else {
                Array(repeating: paddedLength, count: B)
            }
        var hasActiveRightPad = origLens.contains { $0 < paddedLength }
        var h = initialHidden
        let earlyCompactLayer = sdcm.resolvedEarlyCompactLayer(layerCount: layers.count)
        var earlyEmbeddingHidden: MLXArray? = earlyCompactLayer != nil ? initialHidden : nil
        let resolvedTokenIDRows: [[Int32]?]
        if let inputTokenIDRows, inputTokenIDRows.count == B {
            resolvedTokenIDRows = inputTokenIDRows.map { Optional($0) }
        } else if let inputTokenIDs {
            if B == 1 {
                resolvedTokenIDRows = [inputTokenIDs]
            } else if inputTokenIDs.count == B * paddedLength {
                resolvedTokenIDRows = (0 ..< B).map { b in
                    Array(inputTokenIDs[(b * paddedLength) ..< (b * paddedLength + origLens[b])])
                }
            } else {
                resolvedTokenIDRows = Array(repeating: nil, count: B)
            }
        } else {
            resolvedTokenIDRows = Array(repeating: nil, count: B)
        }

        let basePositions = (0 ..< paddedLength).map { Int32($0) }
        var positionsHostRows = Array(repeating: basePositions, count: B)
        var flatInitPos: [Int32] = []
        flatInitPos.reserveCapacity(B * paddedLength)
        for _ in 0 ..< B { flatInitPos.append(contentsOf: basePositions) }
        var positions = MLXArray(flatInitPos, [B, paddedLength])

        var pendingScores: [Gemma4SDCMBoundaryScores]? = nil
        var checkpoint: Gemma4SDCMCheckpoint? = nil
        var dflashAccumulator = (onDFlashContext != nil ? dflashTap : nil)
            .map(DFlashTapAccumulator.init)

        func applyCompaction(keepRows: [[Int32]]) {
            guard let nextLen = keepRows.first?.count,
                nextLen > 0,
                keepRows.count == B,
                keepRows.allSatisfy({ $0.count == nextLen })
            else { return }
            let mustRewrite = nextLen < h.dim(1) || hasActiveRightPad
            guard mustRewrite else { return }
            if B == 1 {
                let idxArray = MLXArray(keepRows[0])
                h = h.take(idxArray, axis: 1)
                dflashAccumulator?.compact { $0.take(idxArray, axis: 1) }
                let nextPos = keepRows[0].map { positionsHostRows[0][Int($0)] }
                positionsHostRows[0] = nextPos
                positions = MLXArray(nextPos, [1, nextLen])
            } else {
                var flatKeep: [Int32] = []
                flatKeep.reserveCapacity(B * nextLen)
                var flatPos: [Int32] = []
                flatPos.reserveCapacity(B * nextLen)
                for b in 0 ..< B {
                    flatKeep.append(contentsOf: keepRows[b])
                    let nextPos = keepRows[b].map { positionsHostRows[b][Int($0)] }
                    positionsHostRows[b] = nextPos
                    flatPos.append(contentsOf: nextPos)
                }
                let idxArray = MLXArray(flatKeep, [B, nextLen, 1])
                h = MLX.takeAlong(h, idxArray, axis: 1)
                dflashAccumulator?.compact { MLX.takeAlong($0, idxArray, axis: 1) }
                positions = MLXArray(flatPos, [B, nextLen])
            }
            hasActiveRightPad = false
            asyncEval(h, positions)
        }

        for (idx, layer) in layers.enumerated() {
            if let targetFraction = sdcm.keepSchedule[idx], let scoresList = pendingScores {
                let curLen = h.dim(1)
                let rowMaxLens =
                    hasActiveRightPad ? origLens : Array(repeating: curLen, count: B)
                let rawKeepRows = (0 ..< B).map { b in
                    let posSlice =
                        rowMaxLens[b] < positionsHostRows[b].count
                        ? Array(positionsHostRows[b][0 ..< rowMaxLens[b]]) : positionsHostRows[b]
                    return Gemma4BlockSDCM.selectCompactedIndices(
                        boundaryScores: scoresList[b],
                        positions: posSlice,
                        originalLength: origLens[b],
                        targetFraction: targetFraction,
                        config: sdcm)
                }
                let keepRows = Gemma4BlockSDCM.equalizeBatchedCompactedIndices(
                    keepRows: rawKeepRows,
                    rowBlockScores: scoresList.map(\.forwardBlockScores),
                    positionsHostRows: positionsHostRows,
                    originalLengths: origLens,
                    rowMaxLengths: rowMaxLens,
                    blockSize: sdcm.blockSize,
                    currentLength: curLen)
                applyCompaction(keepRows: keepRows)
                pendingScores = nil
            }

            let isFinalLayer = idx == layers.count - 1
            let nextReverseQuota: Float? =
                sdcm.keepSchedule[idx + 1] != nil
                ? (sdcm.reversePrefixQuota[idx + 1] ?? 0) : nil
            let (out, nextScores) = layer.forwardBlockSDCM(
                h,
                positions: positions,
                positionsHostRows: positionsHostRows,
                paddedLength: paddedLength,
                originalLengths: origLens,
                hasActiveRightPad: hasActiveRightPad,
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
                if h.dim(1) == paddedLength {
                    let rawKeepEarly: [[Int32]]
                    let unionScores: [[Float]]
                    if !hasActiveRightPad {
                        (rawKeepEarly, unionScores) =
                            Gemma4BlockSDCM.selectEarlyTriSignalIndicesBatched(
                                tokenIDRows: resolvedTokenIDRows,
                                embeddingHidden: embHidden,
                                layerHidden: h,
                                targetFraction: sdcm.earlyPreCompactRatio,
                                blockSize: sdcm.earlyPreCompactBlockSize,
                                topHaloK: sdcm.earlyPreCompactTopHaloK,
                                config: sdcm)
                    } else {
                        var keeps: [[Int32]] = []
                        var scores: [[Float]] = []
                        keeps.reserveCapacity(B)
                        scores.reserveCapacity(B)
                        for b in 0 ..< B {
                            let rowLen = origLens[b]
                            let (kRow, sRow) =
                                Gemma4BlockSDCM.selectEarlyTriSignalIndicesBatched(
                                    tokenIDRows: [resolvedTokenIDRows[b]],
                                    embeddingHidden: embHidden[
                                        b ..< (b + 1), 0 ..< rowLen, 0...],
                                    layerHidden: h[b ..< (b + 1), 0 ..< rowLen, 0...],
                                    targetFraction: sdcm.earlyPreCompactRatio,
                                    blockSize: sdcm.earlyPreCompactBlockSize,
                                    topHaloK: sdcm.earlyPreCompactTopHaloK,
                                    config: sdcm)
                            keeps.append(kRow[0])
                            scores.append(sRow[0])
                        }
                        rawKeepEarly = keeps
                        unionScores = scores
                    }
                    let keepEarly = Gemma4BlockSDCM.equalizeBatchedCompactedIndices(
                        keepRows: rawKeepEarly,
                        rowBlockScores: unionScores,
                        positionsHostRows: positionsHostRows,
                        originalLengths: origLens,
                        rowMaxLengths: origLens,
                        blockSize: sdcm.earlyPreCompactBlockSize,
                        currentLength: h.dim(1))
                    applyCompaction(keepRows: keepEarly)
                }
                earlyEmbeddingHidden = nil
            }
            dflashAccumulator?.record(layerIdx: idx, hidden: h)
            if let ckptLayer = captureCheckpointAtLayer, idx == ckptLayer, B == 1 {
                checkpoint = Gemma4SDCMCheckpoint(
                    hiddenStates: h,
                    positions: positionsHostRows[0],
                    checkpointLayer: idx,
                    originalLength: origLens[0])
            }
            // Keep Metal command buffers bounded on long prompts without forcing host syncs
            // except at scoring boundaries where `computeBilateralBlockScoresBatched` reads host floats.
            if !isFinalLayer && nextScores == nil && (idx + 1).isMultiple(of: 6) {
                asyncEval(h)
            }
        }

        if let projected = dflashAccumulator?.projectIfComplete() {
            onDFlashContext?(projected)
        }
        return (norm(h), checkpoint)
    }
}
