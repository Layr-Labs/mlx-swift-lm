import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A recurrent CBv2 target whose greedy next token is `token + 1` (mod 32),
/// plus 2 at every position `p % 5 == 3`. Its hidden rows are deterministic
/// position/token features of the DSpark width. A row's state is its own
/// token counter, so rows never interact.
private final class DSparkChainTarget: CBv2RecurrentMTPSteppableModel,
    CBv2MTPPolicyTopTwoProviding
{
    static let vocabulary = 32
    let featureWidth: Int
    let cbv2Capabilities = CBv2ModelCapabilities(
        supportsPrefixReuse: false, supportsRecurrentCheckpointReuse: false,
        supportsPagedKV: false, supportsCompiledDecode: false, supportsPackedPrefill: false,
        supportsMTP: true)
    let recurrentStateSpec: CBv2RecurrentStateSpec? = .init(layers: [
        .init(
            modelLayerIndex: 0, convShape: [1, 1, 1], convDType: .float32,
            ssmShape: [1, 1, 1, 1], ssmDType: .float32)
    ])
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    var supportsRequestStatefulMTP: Bool { true }
    var supportsCapturedVerifyWindow: Bool { true }

    init(featureWidth: Int) { self.featureWidth = featureWidth }

    func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        let first = argMax(logits, axis: -1).asType(.int32)
        let ids = stacked([first, (first + 1) % Int32(Self.vocabulary)], axis: -1)
        return (ids, takeAlong(logits, ids, axis: -1))
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("recurrent fixture")
    }

    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
    {
        preconditionFailure("recurrent fixture")
    }

    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        run(tokens: tokens, caches: caches, recurrentState: recurrentState, captured: false).0
    }

    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        run(tokens: tokens, caches: caches, recurrentState: recurrentState, captured: false)
    }

    func forwardWithHiddenCaptured(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        run(tokens: tokens, caches: caches, recurrentState: recurrentState, captured: true)
    }

    private func run(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], captured: Bool
    ) -> (MLXArray, MLXArray) {
        let batch = tokens.dim(0)
        let length = tokens.dim(1)
        let qkv = tokens.asType(.float32).reshaped([batch, 1, length, 1])
        for cache in caches {
            _ = cache.updateAndAttend(queries: qkv, keys: qkv, values: qkv, scale: 1, sinks: nil)
        }
        var starts: [MLXArray] = []
        for evaluation in recurrentState {
            let previous =
                evaluation.inputState(modelLayerIndex: 0)?.conv ?? MLXArray.zeros([1, 1, 1])
            if captured {
                let stack = concatenated(
                    (0 ..< length).map { previous + Float($0 + 1) }, axis: 0)
                try! evaluation.stageCaptured(
                    modelLayerIndex: 0, conv: stack, ssm: stack.reshaped([length, 1, 1, 1]),
                    positions: length)
            } else {
                let next = previous + Float(length)
                try! evaluation.stage(
                    modelLayerIndex: 0, conv: next, ssm: next.reshaped([1, 1, 1, 1]))
            }
            starts.append(previous.reshaped([1, 1]))
        }
        let start = concatenated(starts, axis: 0).asType(.int32)
        let positions = start + MLXArray(Int32(0) ..< Int32(length)).reshaped([1, length])
        let bump = MLX.where((positions % 5) .== 3, Int32(2), Int32(0))
        let next = (tokens.asType(.int32) + 1 + bump) % Int32(Self.vocabulary)
        let vocab = MLXArray(Int32(0) ..< Int32(Self.vocabulary)).reshaped([1, 1, -1])
        let logits = MLX.where(vocab .== next[0..., 0..., .newAxis], Float(10), Float(-10))
        let lanes = MLXArray(0 ..< featureWidth).asType(.float32).reshaped([1, 1, -1])
        let hidden = sin(
            positions.asType(.float32)[0..., 0..., .newAxis] * 0.37 + lanes * 0.11
                + tokens.asType(.float32)[0..., 0..., .newAxis] * 0.05)
        return (logits, hidden)
    }
}

@Suite("Inkling2 DSpark engine", .serialized)
struct Inkling2DSparkEngineTests {
    /// A drafter whose Markov head proposes `previous + 1`: one-hot
    /// `markov_w1`, an identity path to the `out` gate, and an identity
    /// unembedding folded with `hidden_out = 10 I`.
    fileprivate static func chainDrafter(target: DSparkChainTarget) throws -> Inkling2DSparkDrafter
    {
        let v = DSparkChainTarget.vocabulary
        let hidden = 32
        let configuration = Inkling2DSparkConfiguration(
            hiddenSize: hidden, numHiddenLayers: 3, numAttentionHeads: 2, numKeyValueHeads: 1,
            headDim: 64, intermediateSize: 64, localLayerIds: [0, 2], slidingWindowSize: 8,
            blockSize: 8, markovRank: v, targetLayerIds: [5, 23, 41, 48],
            vocabularySize: v, paddedVocabularySize: v, finalLogitSoftcapping: 30,
            targetLayerVisits: 48)
        try configuration.validate()
        let model = Inkling2DSparkModel(configuration)
        let shift = (0 ..< v).map { Int32(($0 + 1) % v) }
        let w1 = MLX.where(
            MLXArray(Int32(0) ..< Int32(v)).reshaped([1, v]) .== MLXArray(shift).reshaped([v, 1]),
            Float(4), Float(0))
        var rnn = MLXArray.zeros([3 * v, 2 * v + 2 * hidden])
        rnn[(2 * v)..., v ..< (2 * v)] = MLXArray.eye(v)
        model.update(
            parameters: ModuleParameters.unflattened([
                "markov_head.markov_w1.weight": w1,
                "markov_head.rnn_proj.weight": rnn,
                "markov_head.rnn_proj.bias": MLXArray.zeros([3 * v]),
                "markov_head.hidden_out.weight": MLXArray.eye(v) * 10,
                "markov_head.hidden_out.bias": MLXArray.zeros([v]),
            ]))
        eval(model)
        model.prepareDerivedWeights()
        let arrays = Inkling2DSparkArrayTarget(
            embedding: MLXRandom.normal([v, hidden], key: MLXRandom.key(7)),
            embeddingNorm: MLXArray.ones([hidden]), unembedding: MLXArray.eye(v),
            eps: 1e-6)
        return Inkling2DSparkDrafter(
            model: model, target: arrays, targetIdentity: ObjectIdentifier(target))
    }

    fileprivate static func engine(drafting: Bool, maxRows: Int)
        throws -> (EngineV2, Inkling2DSparkDrafter?)
    {
        let target = DSparkChainTarget(featureWidth: 4 * 32)
        let drafter = drafting ? try chainDrafter(target: target) : nil
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let engine = EngineV2(
            model: target, layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 26, kvDType: .float32)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: maxRows, maxBatchedTokensPerStep: max(256, 8 * maxRows),
                prefillChunkSize: 16, maxWaiting: max(16, maxRows)),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: .init(
                enabled: drafting, maxDraftTokens: 7, maxSpeculativeBatch: max(maxRows, 8),
                fixedDraftTokens: 7, verificationMode: .rectangular,
                maxAutomaticRectangularTokens: 8 * max(maxRows, 8),
                speculativeBatchLimit: max(maxRows, 8)))
        return (engine, drafter)
    }

    static func prompts(_ rows: Int) -> [[Int]] {
        (0 ..< rows).map { row in (0 ..< (3 + 5 * row)).map { ($0 * 7 + row * 3) % 32 } }
    }

    static func run(_ engine: EngineV2, prompts: [[Int]], maxTokens: Int) async throws -> [[Int]] {
        let streams = try prompts.enumerated().map { index, prompt in
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(UInt64(index + 1)), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: maxTokens))
        }
        var outputs: [[Int]] = []
        for (index, stream) in streams.enumerated() {
            outputs.append(
                try await cbv2CollectFinished(
                    stream, label: "row \(index)", expectedTokens: maxTokens
                ).tokens)
        }
        return outputs
    }

    @Test("the shipped configuration picks depths from the measured per-depth costs")
    func shippedConfigurationUsesTheCostCurve() {
        #expect(Inkling2DSparkDrafter.defaultMTPConfig().costCurveDepthSelection)
        #expect(!CBv2MTPConfig(enabled: true).costCurveDepthSelection)
    }

    @Test("resetting the depth policy's cost learning while idle keeps every token")
    func costLearningResetKeepsTokens() async throws {
        let prompts = Self.prompts(2)
        let (baseline, _) = try Self.engine(drafting: false, maxRows: 2)
        let expected = try await Self.run(baseline, prompts: prompts, maxTokens: 40)
        #expect(!baseline.resetMTPCostLearning())  // no MTP driver
        await baseline.shutdown()

        let (engine, _) = try Self.engine(drafting: true, maxRows: 2)
        let first = try await Self.run(engine, prompts: prompts, maxTokens: 40)
        #expect(engine.resetMTPCostLearning())
        let second = try await Self.run(engine, prompts: prompts, maxTokens: 40)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(first == expected)
        #expect(second == expected)
        #expect(metrics.rounds > 0)
    }

    @Test(
        "drafter on equals drafter off for 1 to 32 rows (above 8 with a raised limit)",
        arguments: [1, 2, 4, 8, 16, 32])
    func greedyParity(rows: Int) async throws {
        let prompts = Self.prompts(rows)
        let (baseline, _) = try Self.engine(drafting: false, maxRows: rows)
        let expected = try await Self.run(baseline, prompts: prompts, maxTokens: 40)
        await baseline.shutdown()

        let (engine, _) = try Self.engine(drafting: true, maxRows: rows)
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.config.maxDraftTokens == 7)
        #expect(driver.config.maxSpeculativeBatch == max(rows, 8))
        let actual = try await Self.run(engine, prompts: prompts, maxTokens: 40)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(actual == expected)
        #expect(expected.allSatisfy { $0.count == 40 })
        #expect(metrics.rounds > 0)
        #expect(metrics.acceptedTokens > 0)
        #expect(metrics.serialVerificationRounds == 0)
        #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
    }
}
