import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A recurrent CBv2 target whose most likely next token is `token + 1` (mod
/// 32), plus 2 at every position `p % 5 == 3`, with the next two tokens close
/// behind (logits 2, 1.2, 0.5; every other token -8), so sampled rows often
/// leave the greedy path. Hidden rows are deterministic position/token
/// features of the DSpark width; a row's state is its own token counter.
private final class CoupledChainTarget: CBv2RecurrentMTPSteppableModel,
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
        let v = Int32(Self.vocabulary)
        let next = ((tokens.asType(.int32) + 1 + bump) % v)[0..., 0..., .newAxis]
        let vocab = MLXArray(Int32(0) ..< v).reshaped([1, 1, -1])
        var logits = MLX.where(vocab .== next, Float(2), Float(-8))
        logits = MLX.where(vocab .== (next + 1) % v, MLXArray(Float(1.2)), logits)
        logits = MLX.where(vocab .== (next + 2) % v, MLXArray(Float(0.5)), logits)
        let lanes = MLXArray(0 ..< featureWidth).asType(.float32).reshaped([1, 1, -1])
        let hidden = sin(
            positions.asType(.float32)[0..., 0..., .newAxis] * 0.37 + lanes * 0.11
                + tokens.asType(.float32)[0..., 0..., .newAxis] * 0.05)
        return (logits, hidden)
    }
}

/// Coupled drafting for Inkling 2's DSpark drafter
/// (`CBv2MTPConfig.coupledProposals`, `CBv2MTPCoupledBlockDrafter`): sampled
/// rows under exact acceptance propose the keyed race of their verify draw.
/// Output must equal the drafter-off engine's; greedy rows, typical
/// acceptance and the default configuration never couple; a rejected coupled
/// round leaves the request's drafter state as a greedy round would.
@Suite(.serialized, .tags(.integration))
struct Inkling2DSparkCoupledDraftingTests {
    /// A drafter whose Markov head favours `previous + 1` by `markovScale`
    /// logits (one-hot `markov_w1`, an identity path to the `out` gate, an
    /// identity unembedding folded with `hidden_out = markovScale * I`). A
    /// small scale leaves the draft distribution spread, so a coupled race
    /// often proposes something other than the argmax.
    private static func makeDrafter(target: AnyObject, markovScale: Float) throws
        -> Inkling2DSparkDrafter
    {
        let v = CoupledChainTarget.vocabulary
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
                "markov_head.hidden_out.weight": MLXArray.eye(v) * markovScale,
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

    private static func engine(
        drafting: Bool, coupled: Bool = false, rows: Int, markovScale: Float = 2,
        acceptance: CBv2MTPAcceptance = .exact
    ) throws -> (EngineV2, Inkling2DSparkDrafter?) {
        let target = CoupledChainTarget(featureWidth: 4 * 32)
        let drafter = drafting ? try makeDrafter(target: target, markovScale: markovScale) : nil
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let engine = EngineV2(
            model: target, layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 26, kvDType: .float32)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 41),
            schedulerConfig: .init(
                maxConcurrentRequests: rows, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: .init(
                enabled: drafting, maxDraftTokens: 7, maxSpeculativeBatch: 8,
                fixedDraftTokens: 7, verificationMode: .rectangular,
                maxAutomaticRectangularTokens: 64, acceptance: acceptance,
                coupledProposals: coupled))
        return (engine, drafter)
    }

    private static func prompts(_ rows: Int) -> [[Int]] {
        (0 ..< rows).map { row in (0 ..< (3 + 5 * row)).map { ($0 * 7 + row * 3) % 32 } }
    }

    private static func run(
        _ engine: EngineV2, rows: Int, sampling: CBv2SamplingParams, maxTokens: Int = 40
    ) async throws -> [[Int]] {
        let streams = try prompts(rows).enumerated().map { index, prompt in
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(UInt64(index + 1)), promptTokens: prompt,
                    sampling: sampling, maxTokens: maxTokens))
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

    static let seededSampling: [CBv2SamplingParams] = [
        .init(temperature: 1.0, seed: 7),
        .init(temperature: 0.7, topP: 0.9, topK: 3, seed: 11),
    ]

    @Test(
        "seeded sampled rows: coupled drafting on equals coupled off equals drafter off",
        arguments: [1, 4], [0, 1])
    func seededSampledOutputIsUnchanged(rows: Int, samplingIndex: Int) async throws {
        let sampling = Self.seededSampling[samplingIndex]
        let (baseline, _) = try Self.engine(drafting: false, rows: rows)
        let expected = try await Self.run(baseline, rows: rows, sampling: sampling)
        await baseline.shutdown()

        let (greedyBaseline, _) = try Self.engine(drafting: false, rows: rows)
        let greedy = try await Self.run(
            greedyBaseline, rows: rows, sampling: .init(temperature: 0))
        await greedyBaseline.shutdown()
        // The sampled stream leaves the greedy path, so drafts can be rejected.
        #expect(expected != greedy)

        for coupled in [false, true] {
            let (engine, drafter) = try Self.engine(drafting: true, coupled: coupled, rows: rows)
            let actual = try await Self.run(engine, rows: rows, sampling: sampling)
            let metrics = try #require(engine.mtpMetricsSnapshot())
            await engine.shutdown()
            #expect(actual == expected, "coupled \(coupled)")
            #expect(metrics.rounds > 0)
            #expect(metrics.draftedTokens > metrics.acceptedTokens)
            let coupledRounds = try #require(drafter).coupledRoundCountForTesting
            #expect(coupled ? coupledRounds > 0 : coupledRounds == 0)
            #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
    }

    @Test("greedy rows never couple and keep the drafter-off output")
    func greedyRowsNeverCouple() async throws {
        let rows = 4
        let greedy = CBv2SamplingParams(temperature: 0)
        let (baseline, _) = try Self.engine(drafting: false, rows: rows)
        let expected = try await Self.run(baseline, rows: rows, sampling: greedy)
        await baseline.shutdown()

        let (engine, drafter) = try Self.engine(
            drafting: true, coupled: true, rows: rows, markovScale: 40)
        let actual = try await Self.run(engine, rows: rows, sampling: greedy)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(actual == expected)
        #expect(metrics.rounds > 0)
        #expect(metrics.acceptedTokens > 0)
        #expect(try #require(drafter).coupledRoundCountForTesting == 0)
    }

    @Test("typical acceptance never couples")
    func typicalAcceptanceNeverCouples() async throws {
        let (engine, drafter) = try Self.engine(
            drafting: true, coupled: true, rows: 2,
            acceptance: .typical(delta: CBv2MTPAcceptance.defaultTypicalDelta))
        _ = try await Self.run(engine, rows: 2, sampling: Self.seededSampling[0])
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(metrics.rounds > 0)
        #expect(try #require(drafter).coupledRoundCountForTesting == 0)
    }

    // MARK: Drafter state

    private struct Round {
        let token: MLXArray
        let hidden: MLXArray
    }

    private static func features(_ positions: Range<Int>, width: Int) -> MLXArray {
        let p = MLXArray(Int32(positions.lowerBound) ..< Int32(positions.upperBound))
            .asType(.float32).reshaped([1, -1, 1])
        let lanes = MLXArray(0 ..< width).asType(.float32).reshaped([1, 1, -1])
        return sin(p * 0.37 + lanes * 0.11)
    }

    /// A configured state that has observed a 6-token prompt.
    private static func primedState(_ drafter: Inkling2DSparkDrafter) throws
        -> any CBv2MTPRequestState
    {
        let width = drafter.model.configuration.featureWidth
        let state = drafter.makeRequestState()
        try drafter.configureRequestState(state, maximumSequenceLength: 128)
        drafter.observeCommittedTarget(
            .init(
                tokens: MLXArray([Int32(3), 9, 14, 1, 22, 5]).reshaped([1, 6]),
                hidden: features(0 ..< 6, width: width)),
            requestState: state)
        return state
    }

    private static func ints(_ x: MLXArray) -> [Int32] {
        eval(x)
        return x.asArray(Int32.self)
    }

    @Test("a selection drives the chain and a selection that declines keeps the argmax")
    func selectionDrivesTheChain() throws {
        let target = CoupledChainTarget(featureWidth: 4 * 32)
        let drafter = try Self.makeDrafter(target: target, markovScale: 40)
        let width = drafter.model.configuration.featureWidth
        let seed = [MLXArray([Int32(5)]).reshaped([1, 1])]
        let hidden = [Self.features(6 ..< 7, width: width)]

        let greedyState = try Self.primedState(drafter)
        let greedy = Self.ints(
            drafter.draftBlock(
                tokens: seed, hidden: hidden, requestStates: [greedyState], depth: 4))
        #expect(greedy == [6, 7, 8, 9])

        // Step 0 forced to 20; later steps decline, so the chain continues from 20.
        let forcedState = try Self.primedState(drafter)
        let forced = Self.ints(
            drafter.draftBlock(
                tokens: seed, hidden: hidden, requestStates: [forcedState], depth: 4,
                selection: CBv2MTPBlockProposalSelection { logits, step in
                    step == 0 ? MLXArray(Array(repeating: Int32(20), count: logits.dim(0))) : nil
                }))
        #expect(forced == [20, 21, 22, 23])

        // A selection that returns each row's argmax is the greedy chain.
        let argmaxState = try Self.primedState(drafter)
        let argmaxDrafts = Self.ints(
            drafter.draftBlock(
                tokens: seed, hidden: hidden, requestStates: [argmaxState], depth: 4,
                selection: CBv2MTPBlockProposalSelection { logits, _ in
                    argMax(logits, axis: -1).asType(.int32)
                }))
        #expect(argmaxDrafts == greedy)
        for state in [greedyState, forcedState, argmaxState] {
            #expect(state.stagedInputCount == 4)
            drafter.releaseRequestState(state)
        }
    }

    @Test("a rejected coupled round leaves the drafter state a greedy round leaves")
    func rejectedCoupledRoundRollsBack() throws {
        let target = CoupledChainTarget(featureWidth: 4 * 32)
        let drafter = try Self.makeDrafter(target: target, markovScale: 40)
        let width = drafter.model.configuration.featureWidth
        let coupledState = try Self.primedState(drafter)
        let greedyState = try Self.primedState(drafter)
        let seed = [MLXArray([Int32(5)]).reshaped([1, 1])]
        let hidden = [Self.features(6 ..< 7, width: width)]

        let coupled = Self.ints(
            drafter.draftBlock(
                tokens: seed, hidden: hidden, requestStates: [coupledState], depth: 5,
                selection: CBv2MTPBlockProposalSelection { logits, _ in
                    MLXArray(Array(repeating: Int32(30), count: logits.dim(0)))
                }))
        let greedy = Self.ints(
            drafter.draftBlock(
                tokens: seed, hidden: hidden, requestStates: [greedyState], depth: 5))
        #expect(coupled != greedy)

        // The target rejects the first draft of both rounds: only the seed commits.
        for state in [coupledState, greedyState] {
            drafter.finalizeRound(
                requestState: state, confirmedInputTokens: 1,
                committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
                committedTargetHidden: MLXArray.zeros([1, 0, width]))
            #expect(state.stagedInputCount == 0)
        }
        #expect(coupledState.committedInputCount == greedyState.committedInputCount)

        // The next round, from the target's correction, drafts identically.
        let nextSeed = [MLXArray([Int32(12)]).reshaped([1, 1])]
        let nextHidden = [Self.features(7 ..< 8, width: width)]
        let afterCoupled = drafter.draftBlock(
            tokens: nextSeed, hidden: nextHidden, requestStates: [coupledState], depth: 5)
        let afterGreedy = drafter.draftBlock(
            tokens: nextSeed, hidden: nextHidden, requestStates: [greedyState], depth: 5)
        #expect(Self.ints(afterCoupled) == Self.ints(afterGreedy))
        let coupledContext = try #require((coupledState as? Inkling2DSparkDrafter.State)?.paged)
        let greedyContext = try #require((greedyState as? Inkling2DSparkDrafter.State)?.paged)
        #expect(coupledContext.written == greedyContext.written)

        // And a discarded coupled round leaves it as a discarded greedy round.
        for state in [coupledState, greedyState] {
            drafter.discardRound(requestState: state)
            #expect(state.stagedInputCount == 0)
        }
        drafter.releaseRequestState(coupledState)
        drafter.releaseRequestState(greedyState)
        #expect(drafter.coupledRoundCountForTesting == 1)
    }
}
