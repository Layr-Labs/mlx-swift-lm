import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Certified draft selection (`Inkling2DSparkCertifiedUnembed`) equals the
/// full product's drafts (`rowProduct` base logits, fold bias, softcap, first
/// maximum) at every Markov step, including ties from duplicated vocabulary
/// rows and a saturated softcap, for one and several rows.
@Suite(.serialized)
struct Inkling2DSparkCertifiedTests {
    static let hidden = 256
    static let padded = 4096
    static let vocabulary = 4000

    static func model() throws -> Inkling2DSparkModel {
        let configuration = Inkling2DSparkConfiguration(
            hiddenSize: hidden, numHiddenLayers: 3, numAttentionHeads: 2, numKeyValueHeads: 1,
            headDim: 128, intermediateSize: 64, localLayerIds: [0, 2], slidingWindowSize: 16,
            blockSize: 8, markovRank: 16, targetLayerIds: [1, 2, 3, 4],
            vocabularySize: vocabulary, paddedVocabularySize: padded, finalLogitSoftcapping: 30,
            targetLayerVisits: 4)
        try configuration.validate()
        let model = Inkling2DSparkModel(configuration)
        SyntheticModel.randomize(model, seed: 9)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(model)
        model.prepareDerivedWeights()
        return model
    }

    /// Random bf16 unembedding with a block of duplicated rows (exact ties).
    static func unembedding(scale: Float) -> MLXArray {
        var w = MLXRandom.normal([padded, hidden], key: MLXRandom.key(3)) * scale
        let copies = w[100 ..< 132]
        w[2000 ..< 2032] = copies
        w[3990 ..< 4022] = copies  // straddles the unpadded vocabulary end
        let result = w.asType(.bfloat16)
        eval(result)
        return result
    }

    /// Drafts of `steps` Markov steps for `rows` [B, S, hidden], with and
    /// without the certified holder, from the same states and inputs.
    static func drafts(
        _ model: Inkling2DSparkModel, full: Inkling2DSparkFoldedHead,
        certified: Inkling2DSparkFoldedHead, rows: MLXArray, features: MLXArray, seed: MLXArray
    ) -> (full: [[Int32]], certified: [[Int32]]) {
        let fullPlan = model.markovPlan(rows: rows, markovFeatures: features, head: full)
        let certifiedPlan = model.markovPlan(rows: rows, markovFeatures: features, head: certified)
        #expect(fullPlan.baseLogits != nil && certifiedPlan.lower != nil)
        var a: [[Int32]] = []
        var b: [[Int32]] = []
        var stateA = model.initialMarkovState(batch: rows.dim(0))
        var stateB = stateA
        var previousA = seed
        var previousB = seed
        for step in 0 ..< rows.dim(1) {
            let nextA = model.markovStep(
                state: stateA, previous: previousA, context: fullPlan.context[0..., step, 0...])
            let nextB = model.markovStep(
                state: stateB, previous: previousB,
                context: certifiedPlan.context[0..., step, 0...])
            stateA = nextA.state
            stateB = nextB.state
            previousA = model.draftTokens(
                plan: fullPlan, step: step, latent: nextA.latent, head: full)
            previousB = model.draftTokens(
                plan: certifiedPlan, step: step, latent: nextB.latent, head: certified)
            a.append(previousA.asArray(Int32.self))
            b.append(previousB.asArray(Int32.self))
        }
        return (a, b)
    }

    @Test(arguments: [(0.05 as Float, 1 as Float), (0.05, 40), (0.6, 8)])
    func certifiedDraftsEqualTheFullProduct(scale: Float, hiddenScale: Float) throws {
        try Inkling2Kernels.$override.withValue(true) {
            let model = try Self.model()
            let unembedding = Self.unembedding(scale: scale)
            let full = model.foldHead(unembedding: unembedding, chunk: 1024)
            var certified = full
            let head = try Inkling2CertifiedHead(weight: unembedding) { $0 }
            certified.certified = .init(
                head: head, cap: 30, vocabularySize: Self.vocabulary, dtype: .bfloat16)
            var steps = 0
            for (trial, batch) in [1, 1, 2, 3, 8].enumerated() {
                let key = MLXRandom.key(UInt64(100 + trial))
                var rows = MLXRandom.normal([batch, 7, Self.hidden], key: key) * hiddenScale
                if trial == 1 {
                    // Rows that point at a duplicated vocabulary block: exact ties.
                    rows = broadcast(
                        unembedding[100 ..< 101].asType(.float32).reshaped(1, 1, Self.hidden)
                            * (20 * hiddenScale),
                        to: [batch, 7, Self.hidden])
                }
                let features = MLXRandom.normal(
                    [batch, Self.hidden], key: MLXRandom.key(UInt64(200 + trial)))
                let seed = MLXArray((0 ..< batch).map { Int32(7 + $0) })
                let (a, b) = Self.drafts(
                    model, full: full, certified: certified, rows: rows.asType(.bfloat16),
                    features: features.asType(.bfloat16), seed: seed)
                #expect(a == b, "trial \(trial) batch \(batch) scale \(scale)x\(hiddenScale)")
                steps += a.count
            }
            #expect(steps == 7 * 5)
        }
    }

    /// The real drafter binds the target's certified head when it has one and
    /// reads the full unembedding when it does not.
    @Test(arguments: [true, false])
    func drafterBindsTheTargetsCertifiedHead(enabled: Bool) throws {
        try Inkling2DSparkDrafter.$certifiedOverride.withValue(enabled) {
            try bindsTheTargetsCertifiedHead(enabled: enabled)
        }
    }

    private func bindsTheTargetsCertifiedHead(enabled: Bool) throws {
        try Inkling2Kernels.$override.withValue(true) {
            let model = try Self.model()
            let unembedding = Self.unembedding(scale: 0.05)
            let embedding = MLXRandom.normal([Self.padded, Self.hidden], key: MLXRandom.key(4))
                .asType(.bfloat16)
            let norm = MLXArray.ones([Self.hidden], dtype: .bfloat16)
            let head = try Inkling2CertifiedHead(weight: unembedding) { $0 }
            let withHead = Inkling2DSparkDrafter(
                model: model,
                target: Inkling2DSparkArrayTarget(
                    embedding: embedding, embeddingNorm: norm, unembedding: unembedding, eps: 1e-6,
                    certifiedHead: head))
            let without = Inkling2DSparkDrafter(
                model: model,
                target: Inkling2DSparkArrayTarget(
                    embedding: embedding, embeddingNorm: norm, unembedding: unembedding, eps: 1e-6))
            #expect((withHead.head.certified != nil) == enabled)
            #expect(without.head.certified == nil)
        }
    }
}
