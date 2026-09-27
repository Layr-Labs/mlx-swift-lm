import Foundation
import MLX

/// Process-local exact recurrent checkpoints. Constructed for one loaded
/// model; the factory reserves this budget inside that slot's KV allocation.
public struct CBv2HybridPrefixCacheConfig: Sendable, Equatable {
    public var maximumBytes: Int
    public var maximumEntries: Int
    public var maximumCheckpointsPerRequest: Int
    public var modelID: String
    public var promptContractID: String
    public var buildID: String

    public init(
        maximumBytes: Int, maximumEntries: Int = 32,
        maximumCheckpointsPerRequest: Int = 2,
        modelID: String, promptContractID: String, buildID: String
    ) {
        self.maximumBytes = maximumBytes
        self.maximumEntries = maximumEntries
        self.maximumCheckpointsPerRequest = maximumCheckpointsPerRequest
        self.modelID = modelID
        self.promptContractID = promptContractID
        self.buildID = buildID
    }

    var isValid: Bool {
        maximumBytes > 0 && maximumEntries > 0 && maximumCheckpointsPerRequest > 0
            && !modelID.isEmpty && !promptContractID.isEmpty && !buildID.isEmpty
    }
}

public struct CBv2HybridPrefixCacheStats: Sendable, Equatable {
    public var residentBytes: Int = 0
    public var stagedBytes: Int = 0
    public var publishingBytes: Int = 0
    public var entries: Int = 0
    public var checkpoints: Int = 0
    public var lookupMatches: Int = 0
    public var misses: Int = 0
    public var adoptions: Int = 0
    public var tokensSaved: Int = 0
    public var capacityRefusals: Int = 0
    public var evictions: Int = 0
    /// Cumulative compact-copy attempts and reserved destination bytes.
    public var kvCompactions: Int = 0
    public var kvCompactionBytes: Int = 0

    public var retainedBytes: Int { residentBytes + stagedBytes + publishingBytes }
}

struct CBv2RecurrentCheckpoint {
    let storageID = UUID()
    let position: Int
    let chunkSize: Int
    let layers: [Int: CBv2RecurrentLayerState]
    let byteCount: Int
    var assistant: (any CBv2MTPPrefixCheckpoint)? = nil
    var qwen4: [Int: CBv2Qwen4IndexerSnapshot] = [:]
    var mediaIdentity: CBv2HybridPrefixIdentity? = nil
    var mediaTargetOnly = false

    var evaluationRoots: [MLXArray] {
        layers.values.flatMap { [$0.conv, $0.ssm].compactMap { $0 } }
            + (assistant?.evaluationTargets ?? [])
            + qwen4.values.flatMap { $0.arrays }
    }
}

struct CBv2HybridPrefixHit {
    let pin: UInt64
    let checkpoint: CBv2RecurrentCheckpoint
    let kvPrefix: [(keys: MLXArray, values: MLXArray, offset: Int)?]
    /// Actual backing bytes, including slack beyond the adopted prefix.
    let kvBackingBytes: Int
}

/// Recurrent exactness requires every donor chunk below a checkpoint to have
/// identical launch geometry: GDN/SSM state exists only at chunk ends and its
/// value depends on the chunk partition. Packed rows, ragged chunks and
/// preemption disarm capture.
///
/// Historical (attention-only) layouts carry no such state: a checkpoint at
/// `p` is the full-attention rows `[0, p)` plus each sliding row's `[p-W, p)`,
/// all written with absolute positions, so any stride-aligned position is a
/// complete model state regardless of the chunk size that produced it.
struct CBv2RecurrentCheckpointGeometry {
    /// Historical checkpoints are captured at every multiple of this stride
    /// that a computed range covers. It is the engine's alignment; the store's
    /// `acceptsCheckpoint` floor and block alignment remain the provider's.
    static let historicalCheckpointStrideTokens = 1024

    var position: Int = 0
    var chunkSize: Int?
    var isArmed = true

    mutating func record(range: Range<Int>, cap: Int, promptLength: Int, packed: Bool) -> Bool {
        guard isArmed else { return false }
        guard !packed, cap > 1, range.lowerBound == position,
            range.upperBound <= promptLength, range.count == cap,
            range.lowerBound % cap == 0, chunkSize == nil || chunkSize == cap
        else {
            isArmed = false
            return false
        }
        position = range.upperBound
        chunkSize = cap
        return true
    }

    /// Stride-aligned positions inside `(range.lowerBound, range.upperBound]`
    /// for a historical layout. The chunk cap is deliberately not part of the
    /// rule: a request may change cap between ranges (solo stripe gaining or
    /// losing decode company) and a ragged final range still yields its
    /// aligned interior boundaries. Packed rows and a non-contiguous range
    /// still disarm the request for the rest of its prompt.
    mutating func recordHistorical(
        range: Range<Int>, promptLength: Int, packed: Bool,
        stride: Int = Self.historicalCheckpointStrideTokens
    ) -> [Int] {
        guard isArmed else { return [] }
        guard !packed, stride > 1, !range.isEmpty, range.lowerBound == position,
            range.upperBound <= promptLength
        else {
            isArmed = false
            return []
        }
        position = range.upperBound
        let first = (range.lowerBound / stride + 1) * stride
        guard first <= range.upperBound else { return [] }
        return Array(Swift.stride(from: first, through: range.upperBound, by: stride))
    }
}
