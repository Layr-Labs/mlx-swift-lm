// Copyright © 2026 Eigen Labs.
// Native three-next-N request lifecycle. No Gemma/DFlash/Qwen head substitute.
import Foundation
import MLX
import MLXLMCommon

public final class MiMoV26MTPAssistant: CBv2MTPRequestStatefulDrafter, CBv2MTPBoundedAllocationProviding,
    CBv2NativeMTPCompletionSplitting {
    weak var target: MiMoV26TextModel?
    let predictor: MiMoV26MTP
    let generation: UInt64
    /// Immutable per-loaded-binding choice. Rectangular is an experimental
    /// target scoring strategy, not a declaration of numerical equivalence.
    public let verificationMode: CBv2MTPVerificationMode
    /// Native factory passes its non-Module loaded owner. The owner contains
    /// the verified components/reservation, never this assistant, so no cycle.
    private let loadLifetimeOwner: AnyObject?
    private var loadedSessionIsActive = true

    public init(target: MiMoV26TextModel, predictor: MiMoV26MTP,
                retaining owner: AnyObject? = nil,
                verificationMode: CBv2MTPVerificationMode = .serialTarget) throws {
        guard verificationMode == .serialTarget || verificationMode == .rectangular else {
            throw MiMoV26MTPError.unsupportedConfiguration("MiMo verification requires explicit serial or rectangular mode")
        }
        guard predictor.belongs(to: target) else { throw MiMoV26MTPError.incompatibleOwner }
        guard predictor.isLoaded else { throw MiMoV26MTPError.weightsNotLoaded }
        self.target = target
        self.predictor = predictor
        loadLifetimeOwner = owner
        generation = predictor.loadedGeneration
        self.verificationMode = verificationMode
    }

    func isCompatible(with target: MiMoV26TextModel) -> Bool {
        loadedSessionIsActive && self.target === target && predictor.belongs(to: target) && predictor.isLoaded
            && predictor.loadedGeneration == generation
    }
    /// The loaded bundle is immutable while serving. A host supporting target
    /// replacement/reload must quiesce and retire requests, invalidate this
    /// assistant, then construct a new verified target/predictor/assistant set.
    /// Arbitrary inherited Module/child mutation is unsupported and is not
    /// detected by a per-token hash or by the named predictor-load generation.
    public func invalidateLoadedSession() { loadedSessionIsActive = false }
    public var mtpTargetIdentity: ObjectIdentifier? {
        guard let target, isCompatible(with: target) else { return nil }
        return ObjectIdentifier(target)
    }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { verificationMode }
    public var maximumDraftTokens: Int? { 3 }
    public var maximumSpeculativeBatch: Int? { 1 }
    public var supportsTargetPrefixAcceptance: Bool { false }
    public var requiresCommittedObservationFence: Bool { true }

    // Conservative admission bound: current settled/speculative head KV plus
    // the prior fence's two immutable measured generations retained until the
    // successor fence. FP32 bounds RoPE storage even with BF16 activations. The
    // legacy getter remains a conservative fallback for consumers that have
    // not adopted resolved bounded admission. EngineV2 does not install this
    // context-linear floor when the bounded declaration below is active.
    public var requestStateBytesPerToken: Int {
        let c = predictor.configuration, g = c.slidingAttention
        func product(_ values: [Int]) -> Int {
            values.reduce(1) { value, factor in
                let result = value.multipliedReportingOverflow(by: factor)
                return result.overflow ? Int.max : result.partialValue
            }
        }
        return [product([4, 3, g.keyValueHeads, g.headDim + g.valueHeadDim, 4]),
                product([4, c.hiddenSize, 4]), 64].reduce(0) { total, value in
            let result = total.addingReportingOverflow(value)
            return result.overflow ? Int.max : result.partialValue
        }
    }
    // RotatingKVCache allocates in blocks of 256 before reaching its window.
    public var requestStateTokenGranularity: Int { 256 }
    public var requestStateTokenAllocationPadding: Int { 12 }

    public func boundedRequestAllocation(limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec? {
        guard let target, isCompatible(with: target) else { return nil }
        return Self.boundedRequestAllocation(configuration: predictor.configuration, limits: limits)
    }

    /// Metadata-only shape proof, intentionally independent of native array
    /// allocation. Four banks cover settled/speculative roots PLUS the last
    /// completed fence's two retained context banks during their replacement.
    /// Working buffers cover lazy graph dependencies until the successor fence;
    /// they are not an assertion that kernel-internal workspace is zero.
    static func boundedRequestAllocation(configuration c: MiMoV26Configuration,
                                         limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec? {
        let g = c.slidingAttention, w = c.slidingWindow, h = c.hiddenSize
        guard c.numNextnPredictLayers == 3, w > 0, h > 0,
              g.keyValueHeads > 0, g.queryHeads > 0, g.headDim > 0, g.valueHeadDim > 0,
              c.intermediateSize > 0, c.vocabularySize > 0,
              limits.maximumPrefillTokens > 0, (1...3).contains(limits.maximumDraftTokens) else { return nil }
        func add(_ a: Int, _ b: Int) -> Int? {
            let (value, overflow) = a.addingReportingOverflow(b)
            return a >= 0 && b >= 0 && !overflow ? value : nil
        }
        func bytes(_ factors: [Int]) -> Int? {
            var value = 4 // FP32 bounds both supported activation/storage types.
            for factor in factors {
                let result = value.multipliedReportingOverflow(by: factor)
                guard factor > 0, !result.overflow else { return nil }
                value = result.partialValue
            }
            return value
        }
        var resident: [CBv2MTPFixedBufferSpec] = [], working: [CBv2MTPFixedBufferSpec] = []
        func append(_ destination: inout [CBv2MTPFixedBufferSpec], _ dimensions: [Int], _ count: Int) -> Bool {
            guard let size = bytes(dimensions), count > 0 else { return false }
            destination.append(.init(logicalBytes: size, allocationCount: count))
            return true
        }
        // Single-token rotating allocation is min(step=256, window); retained
        // multi-token updates are compacted to W. K/V remain separate owners.
        guard append(&resident, [g.keyValueHeads, w, g.headDim], 12),
              append(&resident, [g.keyValueHeads, w, g.valueHeadDim], 12),
              append(&resident, [3, h], 6), // current/prior tail and accepted-row frontiers
              append(&resident, [4], 12) // pending tokens, carry, per-depth inputs and prior contexts
        else { return nil }

        // Before an ordinary observation, accepted-prefix replay is <=3 rows
        // and carry repair <=1 row. Price each invocation independently because
        // allocator padding and all three heads' lazy K/V graphs may coexist.
        // primeTrusted does not project the vocabulary or retain head outputs:
        // only embedding/norm/eh/input-norm/K/V dependencies are evaluated.
        for rows in [limits.maximumPrefillTokens, 3, 1] {
            guard let concatRows = add(w, rows), let featureRows = add(rows, 3),
                  append(&working, [rows, h], 24), // 3 heads * (embedding, norms, 2H concat, projection, input norm + spare)
                  append(&working, [featureRows, h], 2), // saved-feature concat and frontier copy source
                  append(&working, [rows, g.keyValueHeads, g.headDim], 12), // projection, rotary input/output and copy
                  append(&working, [rows, g.keyValueHeads, g.valueHeadDim], 6), // projection and native value scaling
                  append(&working, [concatRows, g.keyValueHeads, g.headDim], 6),
                  append(&working, [concatRows, g.keyValueHeads, g.valueHeadDim], 6),
                  append(&working, [w, g.keyValueHeads, g.headDim], 6),
                  append(&working, [w, g.keyValueHeads, g.valueHeadDim], 6)
            else { return nil }
        }
        // Draft depths have at most 1/2/3 rows, never a context-wide readout.
        // Enumerate the full head forward, masks and readout at each depth;
        // target-forward work and opaque fused-kernel scratch keep their host
        // safety reserve and receive no credit from this declaration.
        for rows in 1...limits.maximumDraftTokens {
            guard let attentionRows = add(w, rows),
                  append(&working, [rows, h], 18),
                  append(&working, [rows, c.intermediateSize], 4),
                  append(&working, [rows, g.queryHeads, g.headDim], 4),
                  append(&working, [rows, g.queryHeads, g.valueHeadDim], 2),
                  append(&working, [rows, g.keyValueHeads, g.headDim], 4),
                  append(&working, [rows, g.keyValueHeads, g.valueHeadDim], 2),
                  append(&working, [attentionRows, g.keyValueHeads, g.headDim], 2),
                  append(&working, [attentionRows, g.keyValueHeads, g.valueHeadDim], 2),
                  append(&working, [rows, attentionRows], 6),
                  append(&working, [attentionRows], 4),
                  append(&working, [rows, c.vocabularySize], 1)
            else { return nil }
        }
        // Historical prompt context is optional at runtime, but its complete
        // envelope is included in this same resolved admission even when MTP
        // is used without persistence. No caller Array backing is retained.
        // Native witness: current/prior measured descriptor plus replacement,
        // concat and bit-copy dependencies. Serialized copies are additionally
        // priced by the existing complete-checkpoint transient reservation.
        guard c.maxPositionEmbeddings > 0, c.maxPositionEmbeddings <= Int(Int32.max),
              append(&resident, [c.maxPositionEmbeddings], 3),
              append(&working, [c.maxPositionEmbeddings], 4),
              let tokenHostBytes = bytes([c.maxPositionEmbeddings, 8]),
              let hostBytes = add(64 << 10, tokenHostBytes) else { return nil }

        // Independent where-copy predicates, scalar/index temporaries, and
        // tiny token concatenations. Four-byte extents bound Bool predicates.
        guard append(&working, [4], 128) else { return nil }
        return .init(resident: resident, working: working, hostBytes: hostBytes)
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        do {
            guard let target, isCompatible(with: target) else { throw MiMoV26MTPError.incompatibleOwner }
            return MiMoV26MTPState(owner: self, generation: generation, cache: try predictor.newCache())
        } catch { preconditionFailure("MiMo MTP request construction: \(error)") }
    }

    func checked(_ requestState: any CBv2MTPRequestState) throws -> MiMoV26MTPState {
        guard let state = requestState as? MiMoV26MTPState, state.owner === self,
              let target, isCompatible(with: target), state.generation == generation else {
            throw MiMoV26MTPError.incompatibleOwner
        }
        guard !state.isReleased, state.cache != nil else {
            throw MiMoV26MTPError.invalidHistory("released request state")
        }
        return state
    }

    public func configureRequestState(_ requestState: any CBv2MTPRequestState,
                                      maximumSequenceLength: Int) throws {
        let state = try checked(requestState)
        guard maximumSequenceLength > 0,
              maximumSequenceLength <= predictor.configuration.maxPositionEmbeddings,
              maximumSequenceLength >= state.committedInputCount,
              state.maximumSequenceLength == nil || state.maximumSequenceLength == maximumSequenceLength else {
            throw MiMoV26MTPError.invalidHistory("request context bound changed or is invalid")
        }
        state.maximumSequenceLength = maximumSequenceLength
    }

    private func validateRows(tokens: MLXArray, hidden: MLXArray) throws {
        guard let target, tokens.ndim == 2, tokens.dim(0) == 1, tokens.dim(1) > 0,
              tokens.dtype == .int32 || tokens.dtype == .uint32,
              hidden.shape == [1, tokens.dim(1), predictor.configuration.hiddenSize],
              hidden.dtype == target.activationDType else {
            throw MiMoV26MTPError.invalidInput("MiMo MTP requires one aligned native target row")
        }
    }

    /// Teacher-force only committed inputs. At depth d, token position p is
    /// paired with the original normalized target feature at p-d-1. Chunk
    /// boundaries consume the saved final three target rows, never a head output.
    private func observeAligned(tokens: MLXArray, hidden: MLXArray,
                                state: MiMoV26MTPState) throws {
        try validateRows(tokens: tokens, hidden: hidden)
        let oldCount = state.observedCount, count = tokens.dim(1)
        state.prefixCaptureMayInstall = false
        let maximum = state.maximumSequenceLength ?? predictor.configuration.maxPositionEmbeddings
        guard count <= maximum - oldCount, let target else { throw MiMoV26MTPError.contextExceeded }
        state.recordPrefixObservation(tokens: tokens)
        let tailCount = state.tail?.dim(1) ?? 0
        let featureStart = oldCount - tailCount
        let features = state.tail.map { concatenated([$0, hidden], axis: 1) } ?? hidden
        for depth in 0..<3 {
            let tokenStart = max(oldCount, depth + 1)
            let length = oldCount + count - tokenStart
            guard length > 0 else { continue }
            let firstFeature = tokenStart - depth - 1
            let selected = firstFeature - featureStart
            try predictor.primeTrusted(depth: depth,
                features: .init(trustedTargetHidden: features[0..., selected..<selected + length, 0...],
                                firstPosition: firstFeature, target: target),
                inputIDs: tokens[0..., (tokenStart - oldCount)..<count],
                target: target, cache: state.cache!)
            state.cache!.compactRetainedHistory(depth: depth)
        }
        state.observedCount += count
        let kept = min(3, state.observedCount)
        // A slice alone can pin the complete prefill allocation after eval.
        state.tail = mimoV26MTPCopy(features[0..., (features.dim(1) - kept)..., 0...])
    }

    private func flushAccepted(state: MiMoV26MTPState) throws {
        if let tokens = state.pendingTokens, let hidden = state.pendingHidden {
            try observeAligned(tokens: tokens, hidden: hidden, state: state)
            state.pendingTokens = nil; state.pendingHidden = nil
        }
    }

    public func observeCommittedTarget(_ observation: CBv2MTPCommittedTargetObservation,
                                        requestState: any CBv2MTPRequestState) {
        do {
            let state = try checked(requestState)
            guard state.round == nil else { throw MiMoV26MTPError.invalidHistory("observation during draft round") }
            try validateRows(tokens: observation.tokens, hidden: observation.hidden)
            state.willMutate()
            try flushAccepted(state: state)
            if let token = state.pendingLastToken {
                // Generic historyCarry pairs the NEW carry token with the
                // PREVIOUS target hidden. Complete the pending previous input;
                // the next ordinary observation supplies the new token's hidden.
                guard observation.tokens.dim(1) == 1 else {
                    throw MiMoV26MTPError.invalidHistory("missing carry history transition")
                }
                try observeAligned(tokens: token, hidden: observation.hidden, state: state)
                state.pendingLastToken = nil
            } else {
                try observeAligned(tokens: observation.tokens, hidden: observation.hidden, state: state)
            }
        } catch { preconditionFailure("MiMo MTP committed target contract: \(error)") }
    }

    private final class UnusedCapture: CBv2MTPPreparedCapture {}
    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { UnusedCapture() }
    public func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("MiMo MTP requires request-owned state")
    }

    public func draftStep(tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
                          requestState: any CBv2MTPRequestState) -> (tokens: MLXArray, hidden: MLXArray) {
        do {
            let state = try checked(requestState)
            try validateRows(tokens: tokens, hidden: hidden)
            guard tokens.dim(1) == 1, shortlist == nil, let target else {
                throw MiMoV26MTPError.invalidInput("native next-N requires one token and full readout")
            }
            guard state.hasRequiredRestoredSuffix else {
                throw MiMoV26MTPError.invalidHistory("historical restore requires an actual target suffix before drafting")
            }
            state.prefixCaptureMayInstall = false
            state.prefixCapturePromptOnly = false
            state.willMutate()
            if state.round == nil {
                try flushAccepted(state: state)
                if let token = state.pendingLastToken {
                    try observeAligned(tokens: token, hidden: hidden, state: state)
                    state.pendingLastToken = nil
                }
                guard state.observedCount > 0, state.tail != nil else {
                    throw MiMoV26MTPError.invalidHistory("target history was not primed from position zero")
                }
                state.round = .init(cache: state.cache!, baseCount: state.observedCount)
            }
            let round = state.round!, depth = round.inputs.count
            guard depth < 3, round.baseCount + depth <
                    (state.maximumSequenceLength ?? predictor.configuration.maxPositionEmbeddings) else {
                throw MiMoV26MTPError.contextExceeded
            }
            round.inputs.append(mimoV26MTPCopy(tokens))
            let length = min(round.baseCount, depth + 1)
            let tail = state.tail!
            let firstPosition = round.baseCount - length
            let tokenHistory = concatenated(round.inputs, axis: 1)
            let output = try predictor.forwardTrusted(depth: depth,
                features: .init(trustedTargetHidden: tail[0..., (tail.dim(1) - length)..., 0...],
                                firstPosition: firstPosition, target: target),
                inputIDs: tokenHistory[0..., (depth + 1 - length)...],
                target: target, cache: round.cache)
            round.cache.compactRetainedHistory(depth: depth)
            let proposed = argMax(output.logits[0..., -1, 0...], axis: -1).asType(.int32)
            // Proposal output belongs to the engine. Keeping a second native
            // view here would make the state claim an engine-owned allocation.
            state.headProposalCounts[depth] += 1
            // The engine loops its returned hidden. Return the SAME target
            // feature every time, never output.normalizedHiddenStates.
            return (proposed, mimoV26MTPCopy(tail[0..., (-1)..., 0...]))
        } catch { preconditionFailure("MiMo MTP draft contract: \(error)") }
    }

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        guard let state = requestState as? MiMoV26MTPState, state.owner === self, !state.isReleased else { return [] }
        return state.ownedArrays
    }

    /// Opt-in Common completion hook. Common has already joined every current
    /// assistant root; state drains construction-stream references and records
    /// independent backing receipts without constructing or evaluating graphs.
    public func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        try checked(requestState).didFinishEvaluation()
    }

    package func fenceRequestStateForNativeCompletion(_ requestState: any CBv2MTPRequestState) throws {
        try checked(requestState).fenceForNativeCompletion()
    }

    package func commitRequestStateNativeCompletion(_ requestState: any CBv2MTPRequestState) throws {
        try checked(requestState).commitNativeCompletion()
    }

    public func finalizeRound(requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
                              committedDraftTokens: MLXArray, committedTargetHidden: MLXArray) {
        do {
            let state = try checked(requestState)
            guard let round = state.round else { throw MiMoV26MTPError.invalidHistory("finalize without round") }
            guard committedDraftTokens.ndim == 2 else {
                throw MiMoV26MTPError.invalidInput("accepted tokens require rank two")
            }
            let count = committedDraftTokens.dim(1)
            guard committedDraftTokens.dim(0) == 1,
                  committedTargetHidden.shape == [1, count, predictor.configuration.hiddenSize],
                  count <= round.inputs.count, confirmedInputTokens == count + 1 else {
                throw MiMoV26MTPError.invalidHistory("accepted prefix does not match target verification")
            }
            state.willMutate()
            let canonicalInputs = concatenated([round.inputs[0], committedDraftTokens], axis: 1)
            if count > 0 {
                // The generic API pairs each accepted draft with its preceding
                // target hidden; realign by prepending the round's seed input.
                state.pendingTokens = mimoV26MTPCopy(canonicalInputs[0..., 0..<count])
                state.pendingHidden = mimoV26MTPCopy(committedTargetHidden)
            }
            state.pendingLastToken = mimoV26MTPCopy(canonicalInputs[0..., count..<count + 1])
            // Every proposal cache is discarded, including accepted proposals:
            // canonical head KV is replayed only from verified TARGET features.
            state.round = nil
        } catch { preconditionFailure("MiMo MTP finalize contract: \(error)") }
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? MiMoV26MTPState, state.owner === self, !state.isReleased else { return }
        state.willMutate()
        state.round = nil
    }
    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? MiMoV26MTPState, state.owner === self else { return }
        state.releaseAfterFence()
    }
}
