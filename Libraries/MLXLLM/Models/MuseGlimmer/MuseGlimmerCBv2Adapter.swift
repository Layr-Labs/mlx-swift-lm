// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// The engine's view of one Muse Glimmer text target. It holds a reference to
/// the model, not a second parameter tree.
///
/// The generic `CBv2SteppableLanguageModelAdapter` serves plain decode. This
/// adapter adds what a request-stateful drafter needs from an attention-only
/// target: it is not a recurrent MTP model, so the round driver accepts the
/// drafter, and its prompt forward keeps the drafter hidden of every row while
/// projecting logits for the last row only.
public final class MuseGlimmerCBv2Adapter: CBv2SteppableModel, CBv2PrefillSteppableModel,
    CBv2ModelCapabilityProviding, CBv2MTPPrefillSteppableModel, CBv2MTPPolicyTopTwoProviding,
    CBv2HistoricalAttentionCheckpointProviding, CBv2CompleteCheckpointKVTypeProviding,
    CBv2DecodeCommandBufferLimitsProviding
{
    public let target: MuseGlimmerTextModel

    public init(_ target: MuseGlimmerTextModel) { self.target = target }

    private func kvCaches(_ caches: [CBv2AttendingLayerCache]) -> [KVCache] {
        caches.map { cache -> KVCache in
            guard let kv = cache as? KVCache else {
                preconditionFailure("CBv2 layer cache \(type(of: cache)) is not a KVCache")
            }
            return kv
        }
    }

    public func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        target(tokens, cache: kvCaches(caches))
    }

    public func prefill(
        tokens: MLXArray, inputEmbeddings: MLXArray?, caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        target.cbv2Prefill(
            tokens, inputEmbedding: inputEmbeddings, cache: kvCaches(caches),
            requirement: requirement)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { target.cbv2Capabilities }
    public var cbv2SupportsHistoricalAttentionCheckpoint: Bool {
        target.cbv2SupportsHistoricalAttentionCheckpoint
    }
    public var cbv2CompleteCheckpointKVDTypes: [DType]? { target.cbv2CompleteCheckpointKVDTypes }
    public var decodeCommandBufferLimits: GPU.CommandBufferLimits? {
        target.decodeCommandBufferLimits
    }

    // MARK: MTP target

    /// Unused by request-stateful drafters, which read hidden rows, not KV.
    public var mtpCaptureLayers: CBv2MTPCaptureLayers? { nil }
    public var supportsRequestStatefulMTP: Bool { !target.mtpHiddenLayerIDs.isEmpty }
    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }

    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        cbv2TopTwoRows(logits)
    }

    public func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
    {
        let result = target.forwardWithMTPHidden(tokens, cache: kvCaches(caches))
        return (result.logits, result.hidden)
    }

    public func forwardWithHiddenForPrefill(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let result = target.prefillWithMTPHidden(
            tokens, cache: kvCaches(caches), requirement: requirement)
        return (result.logits, result.hidden)
    }
}
