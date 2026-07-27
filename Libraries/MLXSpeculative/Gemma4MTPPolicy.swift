// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM

/// Gemma 4 target family inferred from the loaded text-model config.
public enum Gemma4MTPModelFamily: Sendable, Equatable {
    case e2b
    case e4b
    case moeA4B
    case unknownDense
    case unknownMoE
}

/// Execution strategy for a requested MTP batch.
public enum Gemma4MTPBatchStrategy: Sendable, Equatable {
    /// Run each request as a single stream (`B=1`) using this block size.
    case singleStream(blockSize: Int)
    /// Run the requests through the B>1 round loop using this block size.
    case batched(blockSize: Int)

    public var blockSize: Int {
        switch self {
        case .singleStream(let blockSize), .batched(let blockSize):
            return blockSize
        }
    }

    public var usesBatchedRoundLoop: Bool {
        switch self {
        case .batched: return true
        case .singleStream: return false
        }
    }
}

/// Single-stream runtime action selected by the automatic MTP policy.
public enum Gemma4MTPSingleStreamAction: Sendable, Equatable {
    /// Run an MTP round with this block size.
    case mtp(blockSize: Int)
    /// Emit the next token from the target model only.
    case targetOnly

    public var blockSize: Int? {
        switch self {
        case .mtp(let blockSize): blockSize
        case .targetOnly: nil
        }
    }
}

/// Model-aware defaults for Gemma 4 MTP on Apple Silicon.
///
/// The policy is intentionally conservative: only model families with
/// measured B>1 wins opt into the batched round loop. Unknown dense and
/// MoE variants fall back to the B=1 path until benchmarked.
public struct Gemma4MTPAutomaticPolicy: Sendable, Equatable {
    public let family: Gemma4MTPModelFamily
    public let singleStreamBlockSize: Int
    public let batchedBlockSize: Int?

    public var supportsBatchedMTP: Bool {
        batchedBlockSize != nil
    }

    /// Whether the public automatic generation entry point should use MTP
    /// for this family when the caller does not force a block size.
    public var enablesAutomaticMTP: Bool {
        family != .e4b
    }

    public func strategy(forBatchSize batchSize: Int) -> Gemma4MTPBatchStrategy {
        precondition(batchSize >= 1, "batchSize must be at least 1")
        if batchSize == 1 {
            return .singleStream(blockSize: singleStreamBlockSize)
        }
        if let batchedBlockSize {
            return .batched(blockSize: batchedBlockSize)
        }
        return .singleStream(blockSize: singleStreamBlockSize)
    }

    /// Single-stream (`B=1`) runtime decision for a request that has
    /// generated `generatedTokens` tokens so far out of `maxTokens` (if
    /// bounded). Kept separate from `strategy(forBatchSize:)`, which only
    /// picks a block size and does not know about progress through a
    /// specific request.
    public func singleStreamAction(
        generatedTokens: Int,
        maxTokens: Int?
    ) -> Gemma4MTPSingleStreamAction {
        precondition(generatedTokens >= 0, "generatedTokens must be non-negative")

        if !enablesAutomaticMTP {
            return .targetOnly
        }

        if family == .e2b && singleStreamBlockSize == 3 && batchedBlockSize == nil {
            if let maxTokens, maxTokens <= 16 {
                return .mtp(blockSize: 4)
            }
            if generatedTokens < 256 {
                return .targetOnly
            }
            return .mtp(blockSize: 4)
        }

        return .mtp(blockSize: singleStreamBlockSize)
    }

    public static func automatic(for target: Gemma4TextModel) -> Self {
        automatic(for: target.configuration)
    }

    public static func automatic(for config: Gemma4TextConfiguration) -> Self {
        if config.enableMoeBlock {
            if config.hiddenSize == 2816 && config.numHiddenLayers == 30 {
                return Self(
                    family: .moeA4B,
                    singleStreamBlockSize: 3,
                    batchedBlockSize: nil)
            }
            return Self(
                family: .unknownMoE,
                singleStreamBlockSize: 3,
                batchedBlockSize: nil)
        }

        if config.hiddenSize == 1536 && config.numHiddenLayers == 35 {
            if config.quantizationBits == 4 {
                return Self(
                    family: .e2b,
                    singleStreamBlockSize: 3,
                    batchedBlockSize: nil)
            }
            return Self(
                family: .e2b,
                singleStreamBlockSize: 5,
                batchedBlockSize: 3)
        }

        if config.hiddenSize == 2560 && config.numHiddenLayers == 42 {
            return Self(
                family: .e4b,
                singleStreamBlockSize: 3,
                batchedBlockSize: nil)
        }

        return Self(
            family: .unknownDense,
            singleStreamBlockSize: 4,
            batchedBlockSize: nil)
    }
}
