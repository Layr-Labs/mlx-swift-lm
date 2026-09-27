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

/// Where a recurrent (GDN/SSM) donor's complete checkpoints may be taken.
///
/// A recurrent checkpoint is the model's exact state at a token position:
/// full-attention K/V rows, conv and SSM state and MTP history. Measured on
/// real weights (Qwen3.5-9B dense; the chunk-partition parity experiment),
/// that state at a boundary is bit-identical whatever chunk partition
/// produced it (uniform 512, uniform 2,048, the 4,096 stripe, 2,048 then
/// 512s, 512s then 2,048, with or without decode company), and a restore
/// continues token-exactly under any partition. On the MoE (Qwen3.6-35B-A3B)
/// the state depends on the partition from layer 1 on, but so does a cold
/// run's output (it already differs by chunk width and by decode batch
/// width), so a uniform-chunk rule guards a property serving never had
/// there: a restore reproduces its donor partition's cold run, one of the
/// outputs cold serving produces for that prompt, and continuing the
/// donor's chunk on the adopter would not make a hit match an uncached
/// request in the adopter's own workload, whose prefix partition and decode
/// company both differ anyway. Capture is therefore chunk-agnostic: a boundary exists at
/// every contiguous computed-range end that is a multiple of
/// `recurrentCheckpointStrideTokens` (the provider's 256-token block hash)
/// and of the attention query block. Packed rows, a non-contiguous range
/// and an overrun past the prompt still disarm the request for the rest of
/// its prompt; preemption and media are refused before this rule runs.
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

    /// Why the recurrent rule stopped capturing for the rest of the prompt.
    /// A cap change between ranges is no reason: capture is chunk-agnostic.
    enum DisarmReason: Equatable, Sendable {
        /// The range ran in a packed cohort. The engine reports this one to
        /// the provider store (`CBv2CompletePrefixCache
        /// .recordRecurrentCaptureDisarmed(packedAt:)`), once per request.
        case packed
        /// A non-contiguous range, or one overrunning the prompt (the decode
        /// range after the last prompt token).
        case geometry
    }

    /// Recurrent boundaries are captured at multiples of this many tokens,
    /// the provider's block-hash size (`PrefixCachePolicy.blockSize`), so
    /// every boundary is also a routing anchor. The store's effective-token
    /// floor and block alignment remain the provider's.
    static let recurrentCheckpointStrideTokens = 256

    var position: Int = 0
    /// The chunk cap of the last recorded range: provenance for the
    /// manifest, not a constraint on the next range.
    var chunkSize: Int?
    var isArmed = true
    /// Set once, by the `record` that disarmed; nil while armed.
    private(set) var disarmReason: DisarmReason?

    init(position: Int = 0, chunkSize: Int? = nil) {
        self.position = position
        self.chunkSize = chunkSize
    }

    /// Advance over one computed range and say whether its end is a
    /// boundary: contiguous with the previous range, strictly inside the
    /// prompt, and aligned to the attention query block and to `stride` or
    /// to the range's own cap (a full chunk end, the old rule's boundaries;
    /// a no-op in production, where every cap is a multiple of the stride).
    /// The prompt end itself is never a boundary: export requires a token
    /// after the checkpoint (`checkpoint.position < tokens.count`), so a
    /// terminal capture could only be staged, never written, and would
    /// stand in for the deepest boundary when publication drops a target
    /// adjacent to it. The historical path filters the same way. The cap is
    /// recorded as provenance, never compared with earlier ranges.
    mutating func record(
        range: Range<Int>, cap: Int, promptLength: Int, packed: Bool,
        stride: Int = Self.recurrentCheckpointStrideTokens
    ) -> Bool {
        guard isArmed else { return false }
        guard !packed else { return disarm(.packed) }
        guard cap > 0, stride > 0, !range.isEmpty, range.lowerBound == position,
            range.upperBound <= promptLength
        else { return disarm(.geometry) }
        position = range.upperBound
        chunkSize = cap
        return position < promptLength
            && Self.isRecurrentBoundary(position, chunkSize: cap, stride: stride)
    }

    /// Alignment shared by capture and manifest validation: a multiple of
    /// `stride` that is also query-block aligned, or the end of a full chunk
    /// of `chunkSize` (the old rule's boundaries, whose caps the scheduler
    /// already keeps query-block aligned). A `chunkSize` of 1 or less has no
    /// chunk clause, since every position ends a "chunk" of 1; only the
    /// stride rule applies then.
    static func isRecurrentBoundary(
        _ position: Int, chunkSize: Int = 0, stride: Int = recurrentCheckpointStrideTokens
    ) -> Bool {
        guard position > 0 else { return false }
        if chunkSize > 1, position % chunkSize == 0 { return true }
        let block = CBv2AttentionV1.queryBlockSize
        return stride > 0 && position % stride == 0 && (block <= 0 || position % block == 0)
    }

    private mutating func disarm(_ reason: DisarmReason) -> Bool {
        isArmed = false
        disarmReason = reason
        return false
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
