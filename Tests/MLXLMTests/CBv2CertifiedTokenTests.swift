import MLX
import XCTest

@testable import MLXLMCommon

/// A TinyTestModel that fulfills certified offers with exactly what the default sampler would
/// draw from its full logits (argmax for greedy rows, `argmax(softmax(x / T) / noise)` for the
/// rest), so every engine output must equal the plain model's.
private final class CertifyingTinyModel: CBv2SteppableModel, CBv2CertifiedTokenModel {
    let inner: TinyTestModel
    private(set) var offers = 0
    private(set) var fulfilled = 0
    private(set) var warmed = 0
    var cbv2SupportsCertifiedTokens: Bool { true }
    var cbv2CertifiedVocabularySize: Int { inner.vocabularySize }

    init(_ inner: TinyTestModel) {
        self.inner = inner
    }

    func cbv2WarmCertifiedTokens() {
        warmed += 1
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let logits = inner.forward(tokens: tokens, caches: caches)
        guard let request = CBv2CertifiedTokenRequest.current else { return logits }
        offers += 1
        guard !request.isFulfilled, tokens.dim(1) == 1, tokens.dim(0) == request.rows else {
            return logits
        }
        let last = logits[0..., -1, 0...].asType(.float32)
        var chosen = argMax(last, axis: -1).asType(.int32)
        if let noise = request.noise {
            let x = request.temperature.map { last / $0 } ?? last
            let raced = argMax(softmax(x, axis: -1) / noise, axis: -1).asType(.int32)
            chosen = which(MLXArray(request.greedy), chosen, raced)
        }
        request.fulfill(
            tokens: chosen, nonFinite: isNaN(sum(last - last, axes: [1])).asType(.int32),
            candidateCounts: MLXArray([Int32(0)]))
        fulfilled += 1
        return MLXArray.zeros([tokens.dim(0), 1, 1], dtype: .float32)
    }
}

/// Allows every token (a constraint that changes nothing but must still block offers).
private final class CertifiedOpenConstraint: CBv2TokenConstraint, @unchecked Sendable {
    let mode: CBv2TokenConstraintMode = .required
    let maxTokens: Int
    let fallbackTokenID = 0
    let initialState = 0
    private let vocabSize: Int

    init(maxTokens: Int, vocabSize: Int) {
        self.maxTokens = maxTokens
        self.vocabSize = vocabSize
    }

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
        remainingTokens > 0 ? Array(0 ..< vocabSize) : []
    }

    func nextState(state: Int, tokenID: Int) -> Int? {
        tokenID >= 0 && tokenID < vocabSize ? 0 : nil
    }
}

final class CBv2CertifiedTokenTests: XCTestCase {

    private func engine(_ model: any CBv2SteppableModel, kinds: [CBv2LayerKind]) -> EngineV2 {
        EngineV2(
            model: model, layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 11),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256, prefillChunkSize: 16,
                maxWaiting: 16, enablePrefixCache: false),
            reasoningBudgetsEnabled: true)
    }

    private func run(
        _ model: any CBv2SteppableModel, kinds: [CBv2LayerKind], _ requests: [CBv2Request]
    ) async throws -> (outputs: [CBv2SchedCollected], certifiedSteps: Int) {
        let engine = engine(model, kinds: kinds)
        var streams: [AsyncStream<CBv2Event>] = []
        for request in requests { streams.append(try engine.submit(request)) }
        var outputs: [CBv2SchedCollected] = []
        for stream in streams { outputs.append(await cbv2SchedCollect(stream)) }
        let steps = engine.certifiedTokenSteps
        await engine.shutdown()
        return (outputs, steps)
    }

    private func requests(_ sampling: [CBv2SamplingParams]) -> [CBv2Request] {
        sampling.enumerated().map { i, params in
            CBv2Request(
                id: CBv2RequestID(UInt64(i + 1)),
                promptTokens: makePromptTokens(length: 5 + 9 * i, seed: UInt64(31 + i)),
                sampling: params, maxTokens: 24 + 4 * i)
        }
    }

    /// Runs `requests` on the plain and the certifying model; streams must be identical.
    private func compare(
        _ requests: [CBv2Request], file: StaticString = #filePath, line: UInt = #line
    ) async throws -> (certifying: CertifyingTinyModel, certifiedSteps: Int) {
        let tiny = TinyTestModel.make()
        let plain = try await run(tiny, kinds: tiny.layerKinds, requests)
        let certifying = CertifyingTinyModel(tiny)
        let certified = try await run(certifying, kinds: tiny.layerKinds, requests)
        XCTAssertEqual(plain.certifiedSteps, 0, file: file, line: line)
        XCTAssertEqual(certifying.offers, certifying.fulfilled, file: file, line: line)
        XCTAssertEqual(
            certifying.warmed, 1, "warmed once at engine construction", file: file, line: line)
        for (a, b) in zip(plain.outputs, certified.outputs) {
            XCTAssertEqual(a.tokens, b.tokens, file: file, line: line)
            XCTAssertEqual(a.finishReason, b.finishReason, file: file, line: line)
        }
        return (certifying, certified.certifiedSteps)
    }

    func testGreedyStreamsEqualFullProduct() async throws {
        let greedy = Array(repeating: CBv2SamplingParams(temperature: 0), count: 3)
        let result = try await compare(requests(greedy))
        XCTAssertGreaterThan(result.certifiedSteps, 0)
    }

    /// Temperature-only rows (alone and next to greedy rows) are offered, and the race keeps the
    /// sampler's RNG stream: same tokens as the plain model.
    func testTemperatureRowsAreOfferedAndKeepTheRNGStream() async throws {
        let sampled = [
            CBv2SamplingParams(temperature: 0.7, seed: 9), CBv2SamplingParams(temperature: 1.0),
            CBv2SamplingParams(temperature: 0),
        ]
        let result = try await compare(requests(sampled))
        XCTAssertGreaterThan(result.certifiedSteps, 0)
        let alone = try await compare(requests([CBv2SamplingParams(temperature: 0.6, seed: 4)]))
        XCTAssertGreaterThan(alone.certifiedSteps, 0)
    }

    /// Every transform the sampler supports beyond temperature blocks the offer, greedy or not:
    /// logit bias, repetition / frequency / presence penalties, top-k, top-p (below 1 and 0),
    /// min-p, logprobs, a token constraint and a reasoning budget.
    func testEveryOtherTransformIsNeverOffered() async throws {
        var cases: [(String, CBv2Request)] = []
        for temperature: Float in [0, 0.7] {
            let transforms: [(String, CBv2SamplingParams)] = [
                ("bias", CBv2SamplingParams(temperature: temperature, logitBias: [3: 2])),
                (
                    "repetition",
                    CBv2SamplingParams(temperature: temperature, repetitionPenalty: 1.3)
                ),
                ("frequency", CBv2SamplingParams(temperature: temperature, frequencyPenalty: 0.5)),
                ("presence", CBv2SamplingParams(temperature: temperature, presencePenalty: 0.5)),
                ("topK", CBv2SamplingParams(temperature: temperature, topK: 5)),
                ("topP", CBv2SamplingParams(temperature: temperature, topP: 0.9)),
                ("topP0", CBv2SamplingParams(temperature: temperature, topP: 0)),
                ("minP", CBv2SamplingParams(temperature: temperature, minP: 0.1)),
                ("logprobs", CBv2SamplingParams(temperature: temperature, topLogprobs: 2)),
            ]
            for (name, params) in transforms {
                // Greedy rows ignore top-k / top-p / min-p (argmax-invariant): they stay offered.
                if temperature == 0, ["topK", "topP", "topP0", "minP"].contains(name) { continue }
                cases.append(("\(name) T=\(temperature)", requests([params])[0]))
            }
            var constrained = requests([CBv2SamplingParams(temperature: temperature)])[0]
            constrained.tokenConstraint = CertifiedOpenConstraint(maxTokens: 24, vocabSize: 128)
            cases.append(("constraint T=\(temperature)", constrained))
            var budgeted = requests([CBv2SamplingParams(temperature: temperature)])[0]
            budgeted.reasoningBudget = try CBv2ReasoningBudget(
                maximumTokens: 4, openTokenID: 1, closeTokenID: 2, startsInsideThought: false)
            cases.append(("budget T=\(temperature)", budgeted))
        }
        for (name, request) in cases {
            let result = try await compare([request])
            XCTAssertEqual(result.certifying.offers, 0, name)
            XCTAssertEqual(result.certifiedSteps, 0, name)
        }
    }

    /// A hard-masked row (token constraint or reasoning budget) blocks the offer for the whole
    /// step, even next to a plain row: the masked request is admitted first and outlives the
    /// plain one, so no step of the pair is ever offered.
    func testHardMaskedRowBlocksTheStepNextToPlainRows() async throws {
        for temperature: Float in [0, 0.7] {
            for kind in ["constraint", "budget"] {
                var pair = requests([
                    CBv2SamplingParams(temperature: temperature),
                    CBv2SamplingParams(temperature: 0),
                ])
                pair[0].maxTokens = 40
                if kind == "constraint" {
                    pair[0].tokenConstraint = CertifiedOpenConstraint(
                        maxTokens: 40, vocabSize: 128)
                } else {
                    pair[0].reasoningBudget = try CBv2ReasoningBudget(
                        maximumTokens: 4, openTokenID: 1, closeTokenID: 2,
                        startsInsideThought: false)
                }
                let result = try await compare(pair)
                XCTAssertEqual(result.certifying.offers, 0, "\(kind) + plain T=\(temperature)")
                XCTAssertEqual(result.certifiedSteps, 0, "\(kind) + plain T=\(temperature)")
            }
        }
    }

    /// The pipeline predicate the offer uses: true only when nothing but temperature applies.
    func testPipelinePredicate() {
        let pipeline = LogitsPipelineV2(vocabSize: 16)
        func rows(_ p: CBv2SamplingParams) -> [CBv2SamplerRow] {
            [CBv2SamplerRow(id: CBv2RequestID(1), params: p, promptTokens: [1, 2])]
        }
        for p in [CBv2SamplingParams(temperature: 0), CBv2SamplingParams(temperature: 0.7)] {
            pipeline.setRows(rows(p))
            XCTAssertTrue(pipeline.appliesOnlyTemperature, "\(p)")
        }
        for p in [
            CBv2SamplingParams(temperature: 0.7, logitBias: [1: 1]),
            CBv2SamplingParams(temperature: 0.7, repetitionPenalty: 1.2),
            CBv2SamplingParams(temperature: 0.7, frequencyPenalty: 0.2),
            CBv2SamplingParams(temperature: 0.7, presencePenalty: 0.2),
            CBv2SamplingParams(temperature: 0.7, topK: 3),
            CBv2SamplingParams(temperature: 0.7, topP: 0.9),
            CBv2SamplingParams(temperature: 0.7, minP: 0.1),
            CBv2SamplingParams(temperature: 0, topLogprobs: 1),
        ] {
            pipeline.setRows(rows(p))
            XCTAssertFalse(pipeline.appliesOnlyTemperature, "\(p)")
        }
    }
}
