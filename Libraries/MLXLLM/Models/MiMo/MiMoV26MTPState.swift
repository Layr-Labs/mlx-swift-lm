// Copyright © 2026 Eigen Labs.
import Cmlx
import Foundation
import MLX
import MLXLMCommon

/// A bounded private owner of a previously measured root set. Context snapshots
/// pin immutable MLX descriptors even if indexed updates replace the original
/// Swift MLXArray's context. They are not allocation identities and never escape
/// this request. The next drained fence retires them before measuring successors.
private final class MiMoV26MTPPhysicalOwner {
    private var contexts: [MLXArray] = []
    private(set) var bytes = 0
    var count: Int { contexts.count }

    func clearAfterFence() {
        contexts.removeAll()
        bytes = 0
    }

    func measureAfterFence(_ arrays: [MLXArray]) throws {
        clearAfterFence()
        for array in arrays {
            guard let info = try array.evaluatedBufferInfo(), info.isUnique,
                info.isRowContiguous, info.dataOffset == 0,
                info.dataElements == array.size, info.allocatedBytes >= array.nbytes,
                info.allocatedBytes
                    <= (try Memory.allocationFootprintUpperBound(byteCount: array.nbytes))
            else {
                throw MiMoV26MTPError.invalidHistory(
                    "request fence did not prove independent compact backing")
            }
            let next = bytes.addingReportingOverflow(info.allocatedBytes)
            guard !next.overflow else {
                throw MiMoV26MTPError.invalidHistory("physical residency overflow")
            }
            var context = mlx_array_new()
            guard mlx_array_set(&context, array.ctx) == 0 else {
                mlx_array_free(context)
                throw MiMoV26MTPError.invalidHistory("unable to retain measured array context")
            }
            // Retain immediately: if a later root shares this descriptor/data,
            // its isUnique proof fails. No guessed identity/deduplication.
            contexts.append(MLXArray(context))
            bytes = next.partialValue
        }
    }
}

/// One request, three trained-head SWA histories. Mutable caches never escape;
/// historical copies use the assistant codec and Common's admitted capture /
/// import owners, not an unpriced public snapshot of this state.
public final class MiMoV26MTPState: CBv2MTPRequestState, CBv2MTPRequestResidencyReporting {
    weak var owner: MiMoV26MTPAssistant?
    let generation: UInt64
    var cache: MiMoV26MTPRequestCache?
    var observedCount = 0
    var tail: MLXArray?
    var pendingTokens: MLXArray?
    var pendingHidden: MLXArray?
    var pendingLastToken: MLXArray?
    var maximumSequenceLength: Int?
    var round: Round?
    // Historical prefix eligibility is request-local and cannot be re-enabled
    // after any observation or draft. Restored states get one new context bind
    // before their first actual suffix observation.
    var prefixCaptureMayInstall = true
    var prefixCapturePromptOnly = true
    var prefixCaptureContext: MiMoV26MTPPrefixContext?
    var prefixObservedTokens: MLXArray?
    var prefixRestoreSource: MiMoV26MTPPrefixCheckpoint?
    var prefixRestoredBoundary: Int?
    private let physicalOwner = MiMoV26MTPPhysicalOwner()
    private var constructionStream: StreamOrDevice?
    public private(set) var hasUnmeasuredResidency = false
    public private(set) var isReleased = false
    public internal(set) var headProposalCounts = [0, 0, 0]

    final class Round {
        let cache: MiMoV26MTPRequestCache
        let baseCount: Int
        var inputs: [MLXArray] = []
        init(cache: MiMoV26MTPRequestCache, baseCount: Int) {
            self.cache = cache.detachedCopy()
            self.baseCount = baseCount
        }
    }

    init(owner: MiMoV26MTPAssistant, generation: UInt64, cache: MiMoV26MTPRequestCache) {
        self.owner = owner
        self.generation = generation
        self.cache = cache
    }
    public var committedInputCount: Int {
        observedCount + (pendingTokens?.dim(1) ?? 0) + (pendingLastToken == nil ? 0 : 1)
    }
    public var stagedInputCount: Int { round?.inputs.count ?? 0 }
    public var hasPendingPrefillForCostAccounting: Bool { false }
    public var retainedFeatureRows: Int { tail?.dim(1) ?? 0 }
    var hasRequiredRestoredSuffix: Bool {
        prefixRestoredBoundary.map { observedCount > $0 } ?? true
    }
    public var headInputCounts: [Int] { cache?.consumedTokenCounts ?? [] }
    public var retainedHistoryShapes: [[[Int]]] { cache?.stateShapesByDepth ?? [] }
    public var measuredRootCount: Int { physicalOwner.count }

    var ownedArrays: [MLXArray] {
        var result = cache?.innerState() ?? []
        result += [tail, pendingTokens, pendingHidden, pendingLastToken, prefixObservedTokens]
            .compactMap { $0 }
        if let round { result += round.cache.innerState() + round.inputs }
        return result
    }
    /// Shape diagnostic only. This never claims allocator backing or residency.
    public var logicalRootBytes: Int {
        ownedArrays.reduce(0) { total, array in
            let sum = total.addingReportingOverflow(array.nbytes)
            return sum.overflow ? Int.max : sum.partialValue
        }
    }
    /// Exact bytes of the explicitly retained measured owner. When
    /// hasUnmeasuredResidency is true, this is a known subset, not total usage.
    /// The getter performs no metadata inspection, evaluation or synchronization.
    public var materializedBytes: Int { physicalOwner.bytes }

    func willMutate() {
        let stream = StreamOrDevice.default
        precondition(
            constructionStream == nil || constructionStream == stream,
            "MiMo request changed native streams before its completion fence")
        constructionStream = stream
        hasUnmeasuredResidency = true
    }

    /// Caller has joined every evaluationTargets root. Drain native completion
    /// references on the construction stream before replacing measured owners.
    /// This never constructs or evaluates a graph and is never called by a getter.
    func didFinishEvaluation() throws {
        guard !isReleased else { throw MiMoV26MTPError.invalidHistory("fenced a released request") }
        hasUnmeasuredResidency = true
        if let constructionStream { try withError { constructionStream.stream.synchronize() } }
        try physicalOwner.measureAfterFence(ownedArrays)
        prefixRestoreSource = nil
        constructionStream = nil
        hasUnmeasuredResidency = false
    }

    // Tracked-engine split of the existing method above. It adds no fence and
    // changes no graph/math. Only Common's exact current-generation commit may
    // call the second half after the first half returns successfully.
    func fenceForNativeCompletion() throws {
        guard !isReleased else { throw MiMoV26MTPError.invalidHistory("fenced a released request") }
        hasUnmeasuredResidency = true
        if let constructionStream { try withError { constructionStream.stream.synchronize() } }
    }

    func commitNativeCompletion() throws {
        guard !isReleased else {
            throw MiMoV26MTPError.invalidHistory("committed a released request")
        }
        hasUnmeasuredResidency = true
        // evaluatedBufferInfo is explicitly non-evaluating/non-waiting; the
        // footprint bound is scalar arithmetic. No stream or host-ledger lock
        // is taken here. Snapshot/context ownership is queue-confined.
        try physicalOwner.measureAfterFence(ownedArrays)
        constructionStream = nil
        prefixRestoreSource = nil
        hasUnmeasuredResidency = false
    }

    /// Engine-owned retirement only, after its existing native drain boundary.
    func releaseAfterFence() {
        cache = nil
        tail = nil
        pendingTokens = nil
        pendingHidden = nil
        pendingLastToken = nil
        round = nil
        observedCount = 0
        maximumSequenceLength = nil
        prefixCaptureMayInstall = false
        prefixCapturePromptOnly = false
        prefixCaptureContext = nil
        prefixObservedTokens = nil
        prefixRestoreSource = nil
        prefixRestoredBoundary = nil
        physicalOwner.clearAfterFence()
        constructionStream = nil
        hasUnmeasuredResidency = false
        isReleased = true
    }
}
