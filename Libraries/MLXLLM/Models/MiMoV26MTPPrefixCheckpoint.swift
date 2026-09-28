// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon

/// Private codec value, not a request/carry/sampler snapshot. Its copy graphs
/// remain owned by Common's existing complete-checkpoint reservation until
/// required evaluation AND every export/adoption alias have retired.
final class MiMoV26MTPPrefixCheckpoint: CBv2MTPPrefixCheckpoint {
    weak var owner: MiMoV26MTPAssistant?
    let generation: UInt64
    let targetInputCount: Int
    let heads: [MiMoV26MTPPrefixHead]
    let tail: MLXArray
    let tokens: MLXArray
    let metadata: MLXArray
    // Independent, fixed-sized host witness. Never retain the donor's complete
    // prompt/COW allocation after publication of a shorter prefix.
    let expectedTokens: ContiguousArray<Int32>
    /// Conservative allocator bound, NOT measured native residency/refund.
    /// Physical accounting stays with the complete-checkpoint owner.
    let materializedBytes: Int
    let imported: Bool
    var importedStorageOwner: AnyObject?

    init(owner: MiMoV26MTPAssistant, generation: UInt64, count: Int,
         heads: [MiMoV26MTPPrefixHead], tail: MLXArray, tokens: MLXArray,
         metadata: MLXArray, expectedTokens: ContiguousArray<Int32>,
         allocationBound: Int, imported: Bool = false) {
        self.owner = owner; self.generation = generation; targetInputCount = count
        self.heads = heads; self.tail = tail; self.tokens = tokens
        self.metadata = metadata; self.expectedTokens = expectedTokens
        materializedBytes = allocationBound
        self.imported = imported
    }
    var evaluationTargets: [MLXArray] {
        heads.flatMap { [$0.keys, $0.values] } + [tail, tokens, metadata]
    }
}

extension MiMoV26MTPAssistant: CBv2HistoricalMTPPrefixCheckpointCoding {
    package func bindImportedPrefixCheckpointOwner(
        _ checkpoint: any CBv2MTPPrefixCheckpoint, owner: AnyObject
    ) throws {
        guard let value = checkpoint as? MiMoV26MTPPrefixCheckpoint,
              value.imported, belongsToCurrentOwner(value),
              value.importedStorageOwner == nil || value.importedStorageOwner === owner else {
            throw MiMoV26MTPError.incompatibleOwner
        }
        value.importedStorageOwner = owner
    }

    public var prefixCheckpointCodecID: String {
        "mimo-v26-three-head-swa-bf16-target-tail-v1"
    }

    private var prefixConfiguration: MiMoV26Configuration? {
        guard let target, isCompatible(with: target), target.activationDType == .bfloat16,
              predictor.headCount == 3 else { return nil }
        return predictor.configuration
    }

    /// Common calls this exactly once on a fresh admitted state, or on a
    /// pristine off-to-the-side restored state before its first suffix input.
    /// The full caller request is not retained; each token is copied separately.
    package func installPrefixCaptureContext(requestState: any CBv2MTPRequestState,
                                             promptTokens: [Int]) throws {
        let state = try checked(requestState)
        guard let c = prefixConfiguration, state.prefixCaptureMayInstall,
              state.prefixCaptureContext == nil, state.prefixCapturePromptOnly,
              state.round == nil, state.pendingTokens == nil, state.pendingHidden == nil,
              state.pendingLastToken == nil, let maximum = state.maximumSequenceLength,
              !promptTokens.isEmpty, promptTokens.count <= maximum,
              promptTokens.count <= c.maxPositionEmbeddings,
              promptTokens.allSatisfy({ $0 >= 0 && $0 < c.vocabularySize && $0 <= Int(Int32.max) })
        else { throw MiMoV26MTPError.invalidHistory("invalid or late historical capture context") }
        if let boundary = state.prefixRestoredBoundary {
            guard state.observedCount == boundary, boundary < promptTokens.count else {
                throw MiMoV26MTPError.invalidHistory("restored prompt prefix changed")
            }
            let matches: Bool
            if let source = state.prefixRestoreSource {
                matches = source.expectedTokens.enumerated().allSatisfy {
                    Int($0.element) == promptTokens[$0.offset]
                }
            } else if let tokens = state.prefixObservedTokens {
                matches = MiMoV26MTPPrefixValidation.tokensMatch(tokens, count: boundary) {
                    Int32(promptTokens[$0])
                }
            } else { matches = false }
            guard matches else { throw MiMoV26MTPError.invalidHistory("restored prompt prefix changed") }
        } else {
            guard state.observedCount == 0, !state.hasUnmeasuredResidency,
                  state.prefixObservedTokens == nil,
                  state.cache?.consumedTokenCounts == [0, 0, 0] else {
                throw MiMoV26MTPError.invalidHistory("context did not precede target observations")
            }
        }
        state.prefixCaptureContext = MiMoV26MTPPrefixContext(promptTokens)
        state.prefixCaptureMayInstall = false
    }

    public func prefixCheckpointTensorDescriptors(targetInputCount count: Int)
        -> [CBv2CheckpointTensorDescriptor]? {
        guard let c = prefixConfiguration, count >= 4, count < c.maxPositionEmbeddings else { return nil }
        let g = c.slidingAttention
        do {
            var result: [CBv2CheckpointTensorDescriptor] = []
            for depth in 0..<3 {
                let rows = min(count - depth - 1, c.slidingWindow)
                result.append(try .init(role: .assistantKeys, layer: depth,
                    shape: [1, g.keyValueHeads, rows, g.headDim], dtype: .bfloat16))
                result.append(try .init(role: .assistantValues, layer: depth,
                    shape: [1, g.keyValueHeads, rows, g.valueHeadDim], dtype: .bfloat16))
            }
            result.append(try .init(role: .assistantHidden, shape: [1, 3, c.hiddenSize], dtype: .bfloat16))
            result.append(try .init(role: .assistantTokens, shape: [1, count], dtype: .int32))
            result.append(try .init(role: .assistantCacheMetadata, shape: [3, 7], dtype: .int64))
            return result
        } catch { return nil }
    }

    public func capturePrefixCheckpoint(requestState: any CBv2MTPRequestState,
                                        targetInputCount count: Int)
        -> (any CBv2MTPPrefixCheckpoint)? {
        guard let state = try? checked(requestState), let context = state.prefixCaptureContext,
              state.prefixCapturePromptOnly, count >= 4, count < context.count,
              state.observedCount == count, state.committedInputCount == count,
              !state.hasUnmeasuredResidency, state.round == nil,
              state.pendingTokens == nil, state.pendingHidden == nil, state.pendingLastToken == nil,
              let tokens = state.prefixObservedTokens, context.matches(tokens, count: count),
              let tail = state.tail, tail.shape == [1, 3, predictor.configuration.hiddenSize],
              tail.dtype == .bfloat16,
              let heads = state.cache?.prefixHeads(count: count, expectedOwner: predictor),
              let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: count),
              let bound = allocationBound(descriptors),
              let expected = copiedTokens(tokens, count: count) else { return nil }
        // Common must reserve BEFORE calling this. No stream fence/eval or
        // request mutation occurs here. The donor's existing settled fence is
        // required; the new copy graphs remain Common-owned until completion.
        let copied = heads.map { MiMoV26MTPPrefixHead(keys: mimoV26MTPCopy($0.keys),
            values: mimoV26MTPCopy($0.values), metadata: $0.metadata) }
        let metadata = MLXArray(heads.flatMap(\.metadata), [3, 7])
        return MiMoV26MTPPrefixCheckpoint(owner: self, generation: generation, count: count,
            heads: copied, tail: mimoV26MTPCopy(tail), tokens: mimoV26MTPCopy(tokens),
            metadata: metadata, expectedTokens: expected, allocationBound: bound)
    }

    public func encodePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint) -> [MLXArray]? {
        guard let value = checkpoint as? MiMoV26MTPPrefixCheckpoint,
              belongsToCurrentOwner(value),
              let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: value.targetInputCount),
              matches(value.evaluationTargets, descriptors) else { return nil }
        // Encoding may precede copy completion; the enclosing export owns and
        // evaluates these exact roots. It must not release staging on failure.
        return value.evaluationTargets
    }

    public func decodePrefixCheckpoint(tensors: [MLXArray], prefixTokens: [Int])
        -> (any CBv2MTPPrefixCheckpoint)? {
        guard let c = prefixConfiguration,
              let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: prefixTokens.count),
              matches(tensors, descriptors),
              prefixTokens.allSatisfy({ $0 >= 0 && $0 < c.vocabularySize && $0 <= Int(Int32.max) }),
              MiMoV26MTPPrefixValidation.tokensMatch(tensors[7], count: prefixTokens.count,
                  expected: { Int32(prefixTokens[$0]) }),
              let metadata = MiMoV26MTPPrefixValidation.metadata(tensors[8]),
              let bound = allocationBound(descriptors) else { return nil }
        let heads = (0..<3).map { MiMoV26MTPPrefixHead(keys: tensors[2 * $0],
            values: tensors[2 * $0 + 1], metadata: metadata[$0]) }
        guard heads.enumerated().allSatisfy({
            MiMoV26MTPPrefixValidation.headIsValid($0.element, depth: $0.offset,
                count: prefixTokens.count, configuration: c)
        }), let expected = copiedTokens(tensors[7], count: prefixTokens.count) else { return nil }
        // Outer import validates artifact/numerics/tenant/actual prompt identity.
        // Bind decoded state to THIS current assistant/generation; serialized
        // object identifiers or prior load generations are never trusted.
        return MiMoV26MTPPrefixCheckpoint(owner: self, generation: generation, count: prefixTokens.count,
            heads: heads, tail: tensors[6], tokens: tensors[7], metadata: tensors[8],
            expectedTokens: expected, allocationBound: bound, imported: true)
    }

    public func restorePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint)
        -> (any CBv2MTPRequestState)? {
        guard let value = checkpoint as? MiMoV26MTPPrefixCheckpoint,
              belongsToCurrentOwner(value), let c = prefixConfiguration,
              let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: value.targetInputCount),
              matches(value.evaluationTargets, descriptors),
              value.evaluationTargets.allSatisfy({ (try? $0.evaluatedBufferInfo()) != nil }),
              MiMoV26MTPPrefixValidation.tokensMatch(value.tokens, count: value.targetInputCount,
                  expected: { value.expectedTokens[$0] }),
              let metadata = MiMoV26MTPPrefixValidation.metadata(value.metadata),
              metadata == value.heads.map(\.metadata),
              value.heads.enumerated().allSatisfy({
                  MiMoV26MTPPrefixValidation.headIsValid($0.element, depth: $0.offset,
                      count: value.targetInputCount, configuration: c)
              }), let cache = try? predictor.newCache() else { return nil }
        // All input validation precedes copy graph construction or mutation.
        // Build a NEW mutable owner, never alter donor or live target state.
        let state = MiMoV26MTPState(owner: self, generation: generation, cache: cache)
        state.willMutate()
        guard cache.restorePrefixHeads(value.heads, count: value.targetInputCount,
                                       expectedOwner: predictor) else { return nil }
        state.observedCount = value.targetInputCount
        state.prefixRestoredBoundary = value.targetInputCount
        state.tail = mimoV26MTPCopy(value.tail)
        state.prefixObservedTokens = mimoV26MTPCopy(value.tokens)
        state.prefixRestoreSource = value
        // No next-token carry/logits/sampler/proposal state is synthesized.
        // Common configures the admitted context and evaluates all roots before
        // atomic target+assistant adoption. A suffix is still mandatory.
        return state
    }

    private func belongsToCurrentOwner(_ checkpoint: MiMoV26MTPPrefixCheckpoint) -> Bool {
        checkpoint.owner === self && checkpoint.generation == generation && prefixConfiguration != nil
    }
    private func matches(_ arrays: [MLXArray], _ descriptors: [CBv2CheckpointTensorDescriptor]) -> Bool {
        arrays.count == descriptors.count && zip(arrays, descriptors).allSatisfy {
            $0.shape == $1.shape && $0.dtype == $1.dtype.mlxDType
        }
    }
    private func allocationBound(_ descriptors: [CBv2CheckpointTensorDescriptor]) -> Int? {
        var result = 0
        for descriptor in descriptors {
            guard let bytes = try? Memory.allocationFootprintUpperBound(byteCount: descriptor.byteCount) else { return nil }
            let next = result.addingReportingOverflow(bytes)
            guard !next.overflow else { return nil }
            result = next.partialValue
        }
        return result
    }
    private func copiedTokens(_ array: MLXArray, count: Int) -> ContiguousArray<Int32>? {
        // Validate evaluated storage through the same read-only helper first.
        // No asArray/item call here may secretly submit or wait for native work.
        var result = ContiguousArray<Int32>(repeating: 0, count: count)
        guard MiMoV26MTPPrefixValidation.copyTokens(array, into: &result) else { return nil }
        return result
    }
}
