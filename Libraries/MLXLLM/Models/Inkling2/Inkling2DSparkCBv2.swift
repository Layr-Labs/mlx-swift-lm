import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Target side of the DSpark MTP seam. Every hidden row the engine hands the
// drafter is that position's four captured features, concatenated in
// `target_layer_ids` order: [B, L, 4 * hidden]. Logits come from exactly the
// forward the target runs without the drafter, so drafting never changes them.

extension Inkling2Model: Inkling2DSparkTarget {
    public func dsparkNormedEmbedding(_ ids: MLXArray) -> MLXArray { model.embed(ids) }
    public var dsparkUnembedWeight: MLXArray { lmHead.weight }
    /// The drafter shares the head the certified sampler builds from `lm_head`.
    public func dsparkCertifiedHead() -> Inkling2CertifiedHead? { certifiedHeadIfSupported() }
}

extension Inkling2Model {
    /// The drafter's target visits; the last one is the final-norm output.
    public var dsparkVisits: [Int] {
        configuration.mtp?.targetLayerIds ?? [5, 23, 41, configuration.text.visitCount]
    }

    func dsparkFeatures(_ captured: Inkling2CapturedHidden) -> MLXArray {
        concatenated(
            dsparkVisits.map { visit in
                guard let row = captured[visit] else {
                    preconditionFailure("Inkling2 DSpark: visit \(visit) was not captured")
                }
                return row
            }, axis: -1)
    }
}

extension Inkling2Model: CBv2RecurrentMTPForwardable {
    /// Decode, seed and serial verify steps: the row-exact `cbv2Forward` path.
    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let (logits, captured) = cbv2ForwardCapturing(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds,
            captureVisits: Set(dsparkVisits))
        return (logits, dsparkFeatures(captured))
    }
}

extension Inkling2Model: CBv2RecurrentCaptureMTPForwardable {
    /// One verify window staging every position's conv state.
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let (logits, captured) = cbv2ForwardCapturing(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds,
            captureVisits: Set(dsparkVisits), capturesPositions: true)
        return (logits, dsparkFeatures(captured))
    }
}

extension Inkling2Model: CBv2RecurrentPrefillHiddenForwardable {
    /// Prompt chunks: the same trunk as `cbv2RecurrentPrefill`, keeping every
    /// row's features and projecting only the last row.
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        // The same prefill step as cbv2RecurrentPrefill, so drafting never changes the prompt's bits.
        let (hidden, captured) = Inkling2PrefillStep.$active.withValue(true) {
            cbv2Hidden(
                tokens, inputEmbedding: nil, cache: caches, recurrentState: recurrentState,
                positionIds: positionIds, capture: Set(dsparkVisits))
        }
        Inkling2Diagnostics.stepHook?("prefill-built", tokens.dim(1))
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        let features = dsparkFeatures(captured)
        switch requirement {
        case .evaluationOnly: return (last[0..., 0..., 0 ..< 1], features)
        case .lastPositionLogits: return (logits(last), features)
        }
    }
}

/// Media requests run this target's ordinary recurrent token and embedding
/// forwards for their whole lifetime (target-only: no drafts, verify, carry
/// or assistant state) while text requests keep drafting in the same batch.
/// Exact because Inkling 2 decode is row-exact (`Inkling2RowExact`: decode
/// steps and verify windows of at most 16 positions, so a row never depends
/// on the other rows of its forward) and its prompt chunks run the same
/// prefill step either way (`Inkling2PrefillStep`).
extension Inkling2Model: CBv2RecurrentMTPRequestScopedMediaFallback {}

extension Inkling2Model: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        cbv2TopTwoRows(logits)
    }
}

extension Inkling2DSparkDrafter {
    /// The checkpoint's drafter (`mtp.safetensors` beside `config.json`)
    /// bound to its loaded target. Verification is the target's row-exact
    /// window (decode steps and windows of at most 16 positions).
    public static func load(directory: URL, target: Inkling2Model) throws
        -> Inkling2DSparkDrafter
    {
        let model = try Inkling2DSparkModel.load(directory: directory)
        guard model.configuration.targetLayerIds == target.dsparkVisits,
            model.configuration.hiddenSize == target.configuration.text.hiddenSize
        else {
            throw Inkling2DSparkError.target("drafter and target layer visits or width differ")
        }
        return Inkling2DSparkDrafter(
            model: model, target: target, requiredVerificationMode: .rectangularExact,
            commitsAcceptedPrefixPerRow: true)
    }
}

extension Inkling2DSparkDrafter {
    /// Red control: #552's constant.
    public static let servingVerificationMode: CBv2MTPVerificationMode = .rectangularExact
}
