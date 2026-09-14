// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion #3351 Qwen4 prefill admission contract. Do not copy Python.
// GDN is fixed recurrent state. QSA keeps ordinary K/V plus raw indexer
// keys, retained pooled keys, and up to three int64 mRoPE coordinates per
// token. Gathered text core attention is capped at `indexer_budget`; the
// indexer still scores every compressed block (`kv_len / r`). Dense
// `Q × kv_len` SDPA is the leftover ~90 GB gulp that 429'd a prompt Macs
// can gather.

import Foundation

public struct Qwen4ExpPrefillMemory: Sendable, Equatable {
    public var qsaLayers: Int
    public var numAttentionHeads: Int
    public var numKVHeads: Int
    public var headDim: Int
    public var indexerNHeads: Int
    public var indexerHeadDim: Int
    public var indexerBudget: Int
    public var compressRatio: Int
    public var dtypeSize: Int
    public var scoreDtypeSize: Int

    public init(
        qsaLayers: Int,
        numAttentionHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        indexerNHeads: Int,
        indexerHeadDim: Int,
        indexerBudget: Int,
        compressRatio: Int,
        dtypeSize: Int = 2,
        scoreDtypeSize: Int = 2
    ) {
        self.qsaLayers = max(0, qsaLayers)
        self.numAttentionHeads = max(0, numAttentionHeads)
        self.numKVHeads = max(0, numKVHeads)
        self.headDim = max(0, headDim)
        self.indexerNHeads = max(0, indexerNHeads)
        self.indexerHeadDim = max(0, indexerHeadDim)
        self.indexerBudget = max(0, indexerBudget)
        self.compressRatio = max(1, compressRatio)
        self.dtypeSize = max(1, dtypeSize)
        self.scoreDtypeSize = max(1, scoreDtypeSize)
    }

    /// Flash-Next oQ4e live `text_config` (12 QSA / 24q / 2kv / D256).
    public static let flashNext = Qwen4ExpPrefillMemory(
        qsaLayers: 12,
        numAttentionHeads: 24,
        numKVHeads: 2,
        headDim: 256,
        indexerNHeads: 4,
        indexerHeadDim: 128,
        indexerBudget: 2048,
        compressRatio: 4)

    /// Three int64 mRoPE coordinates. Text uses one; charging all three
    /// keeps image requests conservative (Fusion `estimate_qwen4_exp_kv`).
    public static let mropeCoordinateBytes = 3 * 8

    public static func rawIndexerKeyBytesPerToken(
        indexerHeadDim: Int,
        dtypeSize: Int = 2
    ) -> Int {
        max(0, indexerHeadDim) * max(1, dtypeSize)
    }

    /// QSA retains one key per complete compressed block so decode does not
    /// rebuild the entire selector index. Charge that lossless derived state
    /// as part of the cache instead of treating only raw keys as resident.
    public static func pooledIndexerKeyBytesPerToken(
        indexerHeadDim: Int,
        compressRatio: Int,
        dtypeSize: Int = 2
    ) -> Int {
        let raw = rawIndexerKeyBytesPerToken(
            indexerHeadDim: indexerHeadDim, dtypeSize: dtypeSize)
        let ratio = max(1, compressRatio)
        return (raw + ratio - 1) / ratio
    }

    public static func qsaSidecarBytesPerToken(
        indexerHeadDim: Int,
        compressRatio: Int = 4,
        dtypeSize: Int = 2
    ) -> Int {
        rawIndexerKeyBytesPerToken(
            indexerHeadDim: indexerHeadDim, dtypeSize: dtypeSize)
            + pooledIndexerKeyBytesPerToken(
                indexerHeadDim: indexerHeadDim,
                compressRatio: compressRatio,
                dtypeSize: dtypeSize)
            + mropeCoordinateBytes
    }

    public var qsaSidecarBytesPerToken: Int {
        Self.qsaSidecarBytesPerToken(
            indexerHeadDim: indexerHeadDim,
            compressRatio: compressRatio,
            dtypeSize: dtypeSize)
    }

    public var mainKVBytesPerLayer: Int {
        2 * numKVHeads * headDim * dtypeSize
    }

    public var kvBytesPerLayer: Int {
        mainKVBytesPerLayer + qsaSidecarBytesPerToken
    }

    /// Fusion `estimate_qwen4_exp_kv_bytes_per_token`.
    public var kvBytesPerToken: Int {
        qsaLayers * kvBytesPerLayer
    }

    public func residentKVBytes(tokens: Int) -> Int {
        guard tokens > 0, qsaLayers > 0 else { return 0 }
        return kvBytesPerToken * tokens
    }

    /// Conservative steady-state allocation for the production QSA path.
    /// Main K/V and pooled selector keys have logical width. Raw selector
    /// keys and positions use the stepped capacity bank. Text positions use
    /// one plane in practice; retaining the three-plane charge covers media.
    public func allocatedQSABytes(
        logicalTokens: Int,
        indexCapacityTokens: Int
    ) -> Int {
        guard logicalTokens > 0, qsaLayers > 0 else { return 0 }
        let logical = max(0, logicalTokens)
        let capacity = max(logical, indexCapacityTokens)
        let rawKeyBytes = Self.rawIndexerKeyBytesPerToken(
            indexerHeadDim: indexerHeadDim, dtypeSize: dtypeSize)
        let pooledKeyBytes = Self.pooledIndexerKeyBytesPerToken(
            indexerHeadDim: indexerHeadDim,
            compressRatio: compressRatio,
            dtypeSize: dtypeSize)
        let perLayer =
            logical * (mainKVBytesPerLayer + pooledKeyBytes)
            + capacity * (rawKeyBytes + Self.mropeCoordinateBytes)
        return qsaLayers * perLayer
    }

    /// Fusion `estimate_unfused_sdpa_call_bytes`: `[H, Q, K]` scores plus
    /// an fp32 output. Shared by gathered and leftover-dense prices.
    public static func unfusedSDPACallBytes(
        queryHeads: Int,
        queryTokens: Int,
        kvLen: Int,
        headDim: Int,
        scoreDtypeSize: Int
    ) -> Int {
        guard queryHeads > 0, queryTokens > 0, kvLen > 0, headDim > 0 else {
            return 0
        }
        let scores = queryHeads * queryTokens * kvLen * max(1, scoreDtypeSize)
        let output = queryHeads * queryTokens * headDim * 4
        return scores + output
    }

    /// Per-chunk attention transient. `gatheredCore` is an argument, not a
    /// toggle on shared monitor state: text-only Qwen4 charges the budget
    /// cap; image / leftover dense still prices `Q × kv_len`.
    public func prefillTransientBytes(
        queryTokens: Int,
        kvLen: Int,
        gatheredCore: Bool
    ) -> Int {
        guard queryTokens > 0, kvLen > 0 else { return 0 }
        let pooled = max(kvLen / compressRatio, 1)
        let indexer =
            indexerNHeads * queryTokens * pooled * 4
            + indexerNHeads * queryTokens * indexerHeadDim * 4
        var coreKV = kvLen
        if gatheredCore, kvLen > indexerBudget {
            coreKV = min(kvLen, indexerBudget + compressRatio - 1)
        }
        let core = Self.unfusedSDPACallBytes(
            queryHeads: numAttentionHeads,
            queryTokens: queryTokens,
            kvLen: coreKV,
            headDim: headDim,
            scoreDtypeSize: scoreDtypeSize)
        return indexer + core
    }

    /// True when this request can take gathered QSA pricing. Callers that
    /// do not know text-vs-image must pass false (Fusion fail-closed).
    public static func chargeGatheredCore(textOnly: Bool) -> Bool {
        textOnly
    }
}
