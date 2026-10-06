// Copyright © 2026 Eigen Labs Inc.
// Model-owned state; adapted to the framework checkpoint codec separately.
import MLX
import MLXLMCommon

public enum Qwen4ExpMTPStateError: Error, Equatable {
    case lifecycle(String)
    case incompatible(String)
}

/// Canonical state of one autoregressive assistant attention layer.
public struct Qwen4ExpMTPSequenceSnapshot {
    public let layerIndex: Int
    public let keys: MLXArray
    public let values: MLXArray
    public let offset: Int
    public let qwen4Indexer: CBv2Qwen4IndexerSnapshot?

    public init(
        layerIndex: Int,
        keys: MLXArray,
        values: MLXArray,
        offset: Int,
        qwen4Indexer: CBv2Qwen4IndexerSnapshot? = nil
    ) {
        self.layerIndex = layerIndex
        self.keys = keys
        self.values = values
        self.offset = offset
        self.qwen4Indexer = qwen4Indexer
    }
}

/// Settled request-owned assistant state. Round-local proposals and rejected
/// columns are never representable here.
public struct Qwen4ExpMTPStateSnapshot {
    public let cacheLayers: [Qwen4ExpMTPSequenceSnapshot]
    public let backlogHidden: [MLXArray]
    public let backlogTokens: [MLXArray]
    public let targetHiddenFrontier: MLXArray?
    public let committedInputCount: Int
    /// Trusted target inputs intentionally absent from the assistant cache
    /// after a committed cold no-replay transition.
    public let logicalInputBase: Int

    public init(
        cacheLayers: [Qwen4ExpMTPSequenceSnapshot],
        backlogHidden: [MLXArray],
        backlogTokens: [MLXArray],
        targetHiddenFrontier: MLXArray?,
        committedInputCount: Int,
        logicalInputBase: Int = 0
    ) {
        self.cacheLayers = cacheLayers
        self.backlogHidden = backlogHidden
        self.backlogTokens = backlogTokens
        self.targetHiddenFrontier = targetHiddenFrontier
        self.committedInputCount = committedInputCount
        self.logicalInputBase = logicalInputBase
    }

    public var arrays: [MLXArray] {
        cacheLayers.flatMap { layer in
            [layer.keys, layer.values] + (layer.qwen4Indexer?.arrays ?? [])
        } + zip(backlogHidden, backlogTokens).flatMap { [$0.0, $0.1] }
            + [targetHiddenFrontier].compactMap { $0 }
    }
}
