import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// An image embedding beyond a layer's text-calibrated input scale. Under the
/// ModelOpt eager rule the activation becomes NaN and the request's logits are
/// NaN (greedy then emits token 0); under the Super policy it saturates, the
/// logits stay finite, and the plain and speculating owners emit the same
/// tokens. Text, which stays within the calibration, is bit-identical under
/// both policies. Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperActivationOverflowTests {
    typealias Media = NemotronSuperMTPMediaFallbackTests
    typealias Batched = NemotronSuperBatchedEngineTests
    typealias Speculative = NemotronSuperMTPBatchedEngineTests

    static let depth = 2
    static let rows = NemotronSuperCoScheduledServingTests.rows
    /// Layer 0's Mamba input projection, calibrated tighter than the fixture
    /// default (multiplier 1 / 0.012, NaN beyond |x| = 5.57 under the eager
    /// rule). RMS normalization keeps text coordinates near 1 but maps one
    /// dominant embedding coordinate to about sqrt(32) = 5.66 times its norm
    /// weight, so only the spiked image rows overflow.
    static let tightProjection = "backbone.layers.0.mixer.in_proj"
    static let tightInputScale: Float = 0.012

    static func stack(_ policy: ModelOptActivationOverflow) throws -> NemotronSuperMTPFixture.Stack
    {
        let model = try NemotronSuperMTPFixture.target(
            mixerScale: 0.25, quantized: true, activationOverflow: policy)
        let leaves = Dictionary(uniqueKeysWithValues: model.languageModel.leafModules().flattened())
        let projection = try #require(leaves[tightProjection] as? ModelOptCalibratedLinear)
        try projection.update(
            parameters: .unflattened([("input_scale", MLXArray(tightInputScale))]),
            verify: .noUnusedKeys)
        #expect(projection.graphCalibrationValid)
        let target = try NemotronSuperMTPCaptureTarget(capturing: model, maximumRequests: rows)
        let head = try NemotronSuperMTPFixture.head(
            for: model.languageModel, kind: .predictive, dtype: .bfloat16,
            activationOverflow: policy)
        var settings = NemotronSuperMTPFixture.settings(
            depth: depth, verification: .rectangularExact)
        settings.maximumSpeculativeRows = rows
        let assistant = try target.makeAssistant(head: head, settings: settings)
        return .init(model: model, target: target, head: head, assistant: assistant)
    }

    /// The fixture image's features plus one dominant coordinate, the one with
    /// the largest layer-0 norm weight.
    static func spikedFeatures(_ model: NemotronSuperModel) throws -> MLXArray {
        let features = try Media.features(model)
        let parameters = Dictionary(
            uniqueKeysWithValues: model.languageModel.parameters().flattened())
        let weight = try #require(parameters["backbone.layers.0.norm.weight"])
        let coordinate = argMax(weight.asType(.float32)).item(Int.self)
        var spike = [Float](repeating: 0, count: NemotronSuperMTPFixture.hidden)
        spike[coordinate] = 64
        let spiked = features + MLXArray(spike).asType(features.dtype)
        eval(spiked)
        return spiked
    }

    /// Last-position prefill logits of one prompt, through the model's own
    /// preparation and prefill.
    static func lastLogits(_ model: NemotronSuperModel, prompt: [Int], features: MLXArray?)
        throws -> MLXArray
    {
        let tokens = prompt.map { Int32($0) }
        let prepared =
            try features.map { try model.prepare(prompt: tokens, imageFeatures: [$0]) }
            ?? model.prepare(prompt: tokens)
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 4 << 20, kvDType: .bfloat16))
        let bank = CBv2LayerCacheBank(
            caches: model.languageModel.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let rows = try backend.makeSequenceState(
            layerKinds: model.cbv2LayerKinds, promptLength: prompt.count,
            maxLength: prompt.count + 1)
        defer { backend.release(rows) }
        let caches = try bank.layerCaches(rowStates: [rows]).map { try #require($0 as? KVCache) }
        let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        let binding = try state.bind()
        let logits = try model.prefill(
            prepared, cache: caches, recurrentState: [binding], requirement: .lastPositionLogits)
        eval([logits] + (try binding.evaluate()))
        return logits.asType(.float32)
    }

    static func speculativeImage(_ stack: NemotronSuperMTPFixture.Stack, features: MLXArray)
        async throws -> NemotronSuperMTPEngineTests.Collected
    {
        let (engine, backend) = Media.speculativeEngine(stack, kvDType: .bfloat16)
        let result = try await Batched.collect(try engine.submit(Media.mediaRequest(features))) {
            _ in
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        return result
    }

    @Test(.timeLimit(.minutes(3)))
    func imageBeyondCalibrationStaysFiniteAndExactOnBothOwners() async throws {
        let stack = try Self.stack(.nemotronSuper)
        let features = try Self.spikedFeatures(stack.model)
        let logits = try Self.lastLogits(
            stack.model, prompt: Media.mediaPrompt, features: features)
        #expect(isFinite(logits).all().item(Bool.self))
        let plain = try await Media.plainImage(stack.model, features: features, kvDType: .bfloat16)
        let speculating = try await Self.speculativeImage(stack, features: features)
        #expect(speculating.tokens == plain.tokens)
        #expect(speculating.finish == .length)
        #expect(plain.tokens.contains { $0 != 0 })
        print("SYNTHETIC super-activation-overflow saturating image tokens=\(plain.tokens)")
    }

    /// Proves the fixture reproduces the defect: the same image under the
    /// eager rule gives NaN logits and token 0 on both owners.
    @Test(.timeLimit(.minutes(3)))
    func eagerRuleTurnsTheSameImageIntoNaN() async throws {
        let stack = try Self.stack(.eagerNaN)
        let features = try Self.spikedFeatures(stack.model)
        let logits = try Self.lastLogits(
            stack.model, prompt: Media.mediaPrompt, features: features)
        #expect(isNaN(logits).all().item(Bool.self))
        let plain = try await Media.plainImage(stack.model, features: features, kvDType: .bfloat16)
        let speculating = try await Self.speculativeImage(stack, features: features)
        #expect(plain.tokens.allSatisfy { $0 == 0 })
        #expect(speculating.tokens == plain.tokens)
    }

    /// Text within the calibration (and the unspiked image) is bit-identical
    /// under both policies, and speculation matches the plain owner.
    @Test(.timeLimit(.minutes(3)))
    func textAndInRangeImageAreBitIdenticalUnderBothPolicies() async throws {
        let eager = try Self.stack(.eagerNaN)
        let saturating = try Self.stack(.nemotronSuper)
        for request in Speculative.requests {
            let a = try Self.lastLogits(eager.model, prompt: request.prompt, features: nil)
            let b = try Self.lastLogits(saturating.model, prompt: request.prompt, features: nil)
            #expect(isFinite(a).all().item(Bool.self))
            #expect(modelOptBitwiseEqual(a, b), "prompt \(request.prompt)")
        }
        let features = try Media.features(eager.model)
        let a = try Self.lastLogits(eager.model, prompt: Media.mediaPrompt, features: features)
        let b = try Self.lastLogits(
            saturating.model, prompt: Media.mediaPrompt, features: features)
        #expect(isFinite(a).all().item(Bool.self))
        #expect(modelOptBitwiseEqual(a, b))
        let soloEager = try await Speculative.solo(eager.model, kvDType: .bfloat16)
        let soloSaturating = try await Speculative.solo(saturating.model, kvDType: .bfloat16)
        let soloTokens = soloSaturating.map { $0.tokens }
        #expect(soloEager.map { $0.tokens } == soloTokens)
        let (engine, backend) = Speculative.speculativeEngine(
            saturating,
            testCase: .init(head: .predictive, depth: Self.depth, quantized: true, rows: Self.rows))
        var speculated: [[Int]] = []
        for index in Speculative.requests.indices {
            let stream = try engine.submit(Speculative.request(index, base: 700))
            speculated.append(try await Batched.collect(stream) { _ in }.tokens)
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(speculated == soloTokens)
    }

    /// The public MTP head loader selects the Super policy for every head
    /// projection and expert bank (`NemotronSuperInstallationTests` checks the
    /// language model installation).
    @Test func productionHeadLoaderSelectsTheSuperPolicy() throws {
        let model = try NemotronSuperMTPFixture.target(quantized: true)
        let head = try NemotronSuperMTPFixture.withSource(
            NemotronSuperMTPFixture.headTensors(.random, seed: 3, dtype: .bfloat16)
        ) { source in
            let quote = try NemotronSuperMTPModule.loadQuote(
                configuration: model.languageModel.configuration, source: source)
            return try NemotronSuperMTPModule.load(
                for: model.languageModel, source: source, quote: quote,
                construction: NativeConstructionScope())
        }
        var policies: [ModelOptActivationOverflow] = []
        for (_, module) in head.leafModules().flattened() {
            if let projection = module as? ModelOptCalibratedLinear {
                policies.append(projection.activationOverflow)
            } else if let experts = module as? ModelOptNVFP4SwitchLinear {
                policies.append(experts.activationOverflow)
            }
        }
        #expect(policies.count == 4)
        #expect(policies.allSatisfy { $0 == .saturating })
        #expect(ModelOptActivationOverflow.nemotronSuper == .saturating)
    }
}
