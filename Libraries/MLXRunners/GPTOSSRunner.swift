// Copyright © 2026 Eigen Labs.
//
// MLXRunners — GPT-OSS runner.
//
// Every layer carries learned per-head attention sinks, so both KV backends
// must fold sinks into the softmax denominator; the cache construction funnel
// (`newCacheV2`) also primes the model's sinks-activation probe, which is why
// this runner never builds caches any other way.
//
// The family's layer-kind derivation moved out of the engine directory and
// into `GPTOSS.swift`, next to `GPTOSSModelInner.init` which it mirrors.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public final class GPTOSSRunner: Runner, @unchecked Sendable {

    public static let manifest = RunnerManifest(
        runnerID: "layr/gptoss",
        modelTypes: ["gpt_oss"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: true,
            supportsPagedKV: true,
            supportsCompiledDecode: true,
            supportsPackedPrefill: true,
            // No drafter exists for this family in the fork, so speculation
            // is declared off rather than declared on and never resolvable.
            supportsMTP: false,
            supportsCompactRecurrentMTPReplay: false),
        kvBackends: [.contiguous, .paged],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil)
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(
                batch: .upTo(CBv2MTPConfig.testedMaxSpeculativeBatch), timing: .freeRun,
                perStreamTiming: false),
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false),
        ],
        multimodal: false,
        recurrentLayers: false,
        requiresKeepMask: false)

    public let servingModel: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let eosTokenIDs: Set<Int>
    public let layerKinds: [CBv2LayerKind]
    public let loadedDecoders: [DecoderID] = [.serial]
    public let headProvenance: HeadProvenance? = nil
    public let loadedModelType: String

    private let model: GPTOSSModel
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        model: GPTOSSModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.model = model
        self.servingModel = model
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.layerKinds = model.cbv2LayerKinds
        self.loadedModelType = loadedModelType
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
    }

    public static func load(
        _ directory: URL, options: RunnerLoadOptions
    ) async throws -> GPTOSSRunner {
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())
        guard let model = context.model as? GPTOSSModel else {
            throw RunnerError.unexpectedModel(String(describing: type(of: context.model)))
        }
        guard options.drafterDirectory == nil else {
            throw RunnerError.drafterUnavailable(
                "gpt_oss declares no speculative decoder")
        }
        return GPTOSSRunner(
            model: model,
            tokenizer: context.tokenizer,
            eosTokenIDs: RunnerCheckpoint.eosTokenIDs(
                at: directory, tokenizer: context.tokenizer),
            loadedModelType: modelType,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    public func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        let model = self.model
        return try RunnerEngineAssembly.makeEngine(
            manifest: Self.manifest,
            loadedDecoders: loadedDecoders,
            model: model,
            tokenizer: tokenizer,
            layerKinds: layerKinds,
            newCaches: { make in try model.newCacheV2(makeLayerCache: make) },
            mtpDrafter: nil,
            build: build)
    }

    public func makeStepper() throws -> any TeacherForcedStepper {
        let model = self.model
        return CBv2SingleRowStepper(
            model: model,
            layerKinds: layerKinds,
            newCaches: { make in try model.newCacheV2(makeLayerCache: make) },
            kvBytesCapacity: kvBytesCapacity,
            maxLength: maxSequenceLength)
    }
}
