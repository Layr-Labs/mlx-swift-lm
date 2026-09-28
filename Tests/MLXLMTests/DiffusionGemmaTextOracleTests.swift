import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

private final class DiffusionFixtureEncoder: Module {
    @ModuleInfo(key: "language_model") var languageModel: DiffusionGemmaEncoderTextParameters
    init(_ config: DiffusionGemmaTextConfiguration) {
        _languageModel.wrappedValue = DiffusionGemmaEncoderTextParameters(
            layerCount: config.layerCount)
    }
}

private final class DiffusionFixtureBackbone: Module {
    @ModuleInfo var encoder: DiffusionFixtureEncoder
    @ModuleInfo var decoder: DiffusionGemmaTextDecoder
    init(_ config: DiffusionGemmaTextConfiguration) {
        _encoder.wrappedValue = DiffusionFixtureEncoder(config)
        _decoder.wrappedValue = DiffusionGemmaTextDecoder(config)
    }
}

private final class DiffusionFixtureModel: Module {
    @ModuleInfo var model: DiffusionFixtureBackbone
    init(_ config: DiffusionGemmaTextConfiguration) {
        _model.wrappedValue = DiffusionFixtureBackbone(config)
    }
}

@Suite("DiffusionGemma shared text oracle", .serialized)
struct DiffusionGemmaTextOracleTests {
    struct Fixture: Decodable {
        let reference: String
        let model_config: DiffusionGemmaTextConfiguration
        let parameter_count: Int
        let cases: [Case]
        struct Case: Decodable {
            let name: String
            let prompt_length: Int
        }
    }

    private func loadFixture() throws -> (Fixture, [String: MLXArray], DiffusionFixtureModel) {
        let metadata = try #require(
            Bundle.module.url(forResource: "diffusiongemma-text-oracle", withExtension: "json"))
        let tensors = try #require(
            Bundle.module.url(
                forResource: "diffusiongemma-text-oracle", withExtension: "safetensors"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: metadata))
        let arrays = try loadArrays(url: tensors)
        let model = DiffusionFixtureModel(fixture.model_config)
        let weights = arrays.filter { $0.key.hasPrefix("model.") }
        #expect(weights.count == fixture.parameter_count)
        #expect(model.parameters().flattened().count == fixture.parameter_count)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return (fixture, arrays, model)
    }

    private func exact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        eval(actual, expected)
        #expect(actual.shape == expected.shape, "\(label): shape")
        #expect(actual.dtype == expected.dtype, "\(label): dtype")
        #expect(
            actual.asArray(Float.self).map(\.bitPattern)
                == expected.asArray(Float.self).map(\.bitPattern),
            "\(label): raw FP32 mismatch; do not relax oracle")
    }

    @Test func sharedWeightsConditioningAndCommittedCacheMatchReference() throws {
        let (fixture, arrays, model) = try loadFixture()
        #expect(fixture.reference == "e79b0e041677ec4ca5333ba750376bb4e8c434cb")
        #expect(fixture.cases.count == 3)
        let decoder = model.model.decoder
        let scalars = model.model.encoder.languageModel
        for item in fixture.cases {
            let cache = try DiffusionGemmaRequestCache(
                configuration: fixture.model_config,
                expectedPromptLength: item.prompt_length)
            let tokens = try #require(arrays[item.name + ".tokens"])
            let canvas = try #require(arrays[item.name + ".canvas"])
            let hidden = try decoder.encode(
                tokenIds: tokens, cache: cache, encoderParameters: scalars)
            exact(hidden, arrays[item.name + ".encoder_hidden"]!, item.name + ": encoder")
            #expect(cache.position == item.prompt_length)
            for (index, snapshot) in cache.snapshots().enumerated() {
                exact(snapshot.keys, arrays[item.name + ".cache\(index).keys"]!, "encoder keys")
                exact(
                    snapshot.values, arrays[item.name + ".cache\(index).values"]!, "encoder values")
            }
            let before = cache.stateArrays().map { $0.asArray(Float.self).map(\.bitPattern) }
            let first = try decoder.denoise(canvasIds: canvas, cache: cache)
            exact(first, arrays[item.name + ".logits0"]!, item.name + ": denoise zero")
            let second = try decoder.denoise(
                canvasIds: canvas, cache: cache,
                selfConditioningLogits: arrays[item.name + ".conditioning"]!)
            exact(second, arrays[item.name + ".logits1"]!, item.name + ": conditioning")
            #expect(cache.position == item.prompt_length)
            #expect(
                cache.stateArrays().map { $0.asArray(Float.self).map(\.bitPattern) } == before,
                "Provisional canvas must not mutate encoder KV")
            let appended = arrays[item.name + ".appended"]!
            let appendedHidden = try decoder.encode(
                tokenIds: appended, cache: cache, encoderParameters: scalars)
            exact(appendedHidden, arrays[item.name + ".appended_hidden"]!, "committed append")
            #expect(cache.position == item.prompt_length + 4)
            for (index, snapshot) in cache.snapshots().enumerated() {
                exact(
                    snapshot.keys, arrays[item.name + "_appended.cache\(index).keys"]!,
                    "appended keys")
                exact(
                    snapshot.values, arrays[item.name + "_appended.cache\(index).values"]!,
                    "appended values")
            }
            let last = try decoder.denoise(canvasIds: canvas, cache: cache)
            exact(last, arrays[item.name + ".logits_after_append"]!, "denoise after append")
        }
    }

    @Test func rejectsDifferentModelCacheWithoutMutatingIt() throws {
        let (fixture, arrays, model) = try loadFixture()
        let cache = try DiffusionGemmaRequestCache(
            configuration: fixture.model_config, expectedPromptLength: 2)
        let tokens = arrays["prompt2.tokens"]!
        _ = try model.model.decoder.encode(
            tokenIds: tokens, cache: cache,
            encoderParameters: model.model.encoder.languageModel)
        eval(cache.stateArrays())
        let other = DiffusionFixtureModel(fixture.model_config)
        #expect(throws: DiffusionGemmaModelError.self) {
            try other.model.decoder.denoise(canvasIds: arrays["prompt2.canvas"]!, cache: cache)
        }
        #expect(cache.position == 2)
        #expect(throws: DiffusionGemmaModelError.self) {
            try model.model.decoder.encode(
                tokenIds: MLXArray([Int32(-1)]).reshaped(1, 1), cache: cache,
                encoderParameters: model.model.encoder.languageModel)
        }
        #expect(cache.position == 2)
    }

    private func identity(_ changedField: String? = nil) throws -> DiffusionGemmaPrefixIdentity {
        func field(_ name: String) -> String { name == changedField ? "different-" + name : name }
        return try DiffusionGemmaPrefixIdentity(
            tenantScope: field("tenant"), artifact: field("artifact"), template: field("template"),
            media: field("media"), numericalProfile: field("numerics"), epoch: field("epoch"))
    }

    @Test func committedPrefixRestoreIsExactAfterDonorAppendAndRetirement() throws {
        let (fixture, arrays, model) = try loadFixture()
        let decoder = model.model.decoder
        let scalars = model.model.encoder.languageModel
        let context = try identity()
        for item in fixture.cases {
            let tokens = arrays[item.name + ".tokens"]!
            let canvas = arrays[item.name + ".canvas"]!
            let append = arrays[item.name + ".appended"]!
            var donor: DiffusionGemmaRequestCache? = try DiffusionGemmaRequestCache(
                configuration: fixture.model_config, expectedPromptLength: item.prompt_length)
            weak var weakDonor = donor
            _ = try decoder.encode(tokenIds: tokens, cache: donor!, encoderParameters: scalars)
            let checkpoint = try decoder.checkpoint(cache: donor!, identity: context)
            #expect(checkpoint.tokenCount == item.prompt_length)
            #expect(checkpoint.retainedBytes > 0)
            // Real read-only canvas refinement must not appear in the checkpoint.
            eval(try decoder.denoise(canvasIds: canvas, cache: donor!))
            _ = try decoder.encode(tokenIds: append, cache: donor!, encoderParameters: scalars)
            eval(donor!.stateArrays())
            donor = nil
            #expect(weakDonor == nil, "Checkpoint must not keep a request owner alive")

            let resumed = try decoder.restorePrefix(
                checkpoint, identity: context,
                promptTokenIds: concatenated([tokens, append], axis: 1))
            #expect(resumed.position == item.prompt_length)
            for (index, snapshot) in resumed.snapshots().enumerated() {
                exact(snapshot.keys, arrays[item.name + ".cache\(index).keys"]!, "restored keys")
                exact(
                    snapshot.values, arrays[item.name + ".cache\(index).values"]!, "restored values"
                )
            }
            exact(
                try decoder.denoise(canvasIds: canvas, cache: resumed),
                arrays[item.name + ".logits0"]!, "restored logits")
            exact(
                try decoder.encode(tokenIds: append, cache: resumed, encoderParameters: scalars),
                arrays[item.name + ".appended_hidden"]!, "restored append")
            exact(
                try decoder.denoise(canvasIds: canvas, cache: resumed),
                arrays[item.name + ".logits_after_append"]!, "restored append logits")

            // Restoring and updating one borrower must not contaminate a second.
            let another = try decoder.restorePrefix(
                checkpoint, identity: context, promptTokenIds: tokens)
            exact(
                try decoder.denoise(canvasIds: canvas, cache: another),
                arrays[item.name + ".logits0"]!, "independent borrower logits")
            #expect(another.position == item.prompt_length)
        }
    }

    @Test func prefixIdentityTokensOwnerAndEmptyStateFailClosed() throws {
        let (fixture, arrays, model) = try loadFixture()
        let decoder = model.model.decoder
        let context = try identity()
        let tokens = arrays["prompt2.tokens"]!
        let cache = try DiffusionGemmaRequestCache(
            configuration: fixture.model_config, expectedPromptLength: 2)
        #expect(throws: DiffusionGemmaModelError.self) {
            try decoder.checkpoint(cache: cache, identity: context)
        }
        _ = try decoder.encode(
            tokenIds: tokens, cache: cache, encoderParameters: model.model.encoder.languageModel)
        let checkpoint = try decoder.checkpoint(cache: cache, identity: context)
        for field in ["tenant", "artifact", "template", "media", "numerics", "epoch"] {
            #expect(throws: DiffusionGemmaModelError.self) {
                try decoder.restorePrefix(
                    checkpoint, identity: identity(field), promptTokenIds: tokens)
            }
        }
        for invalid in [
            tokens + 1, tokens[0..., ..<1], tokens.asType(.float32),
            MLXArray([Int32(-1), -1]).reshaped(1, 2),
        ] {
            #expect(throws: DiffusionGemmaModelError.self) {
                try decoder.restorePrefix(checkpoint, identity: context, promptTokenIds: invalid)
            }
        }
        let other = DiffusionFixtureModel(fixture.model_config)
        #expect(throws: DiffusionGemmaModelError.self) {
            try other.model.decoder.restorePrefix(
                checkpoint, identity: context, promptTokenIds: tokens)
        }
        #expect(throws: DiffusionGemmaModelError.self) {
            try DiffusionGemmaPrefixIdentity(
                tenantScope: " ", artifact: "artifact", template: "template", media: "none",
                numericalProfile: "fp32", epoch: "epoch")
        }
        #expect(cache.position == 2)
        exact(
            try decoder.denoise(canvasIds: arrays["prompt2.canvas"]!, cache: cache),
            arrays["prompt2.logits0"]!, "negative probes leave donor unchanged")
    }
}
