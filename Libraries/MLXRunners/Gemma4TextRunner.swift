// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Gemma 4 text runner.
//
// Serves the `Gemma4TextModel` tower: the LLM checkpoint directly, or the
// `Gemma4Model` wrapper's own text model (the wrapper OWNS that instance, so
// nothing here constructs a second language module or duplicates weights).
//
// The family's layer-kind derivation moved out of the engine directory and
// into `Gemma4Text.swift` next to the constructors it mirrors; this runner
// reads it off the model, never re-derives it.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public final class Gemma4TextRunner: Runner, @unchecked Sendable {

    public static let manifest = RunnerManifest(
        runnerID: "layr/gemma4-text",
        modelTypes: ["gemma4", "gemma4_text"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: true,
            supportsPagedKV: true,
            supportsCompiledDecode: true,
            supportsPackedPrefill: true,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: false),
        kvBackends: [.contiguous, .paged],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil),
            // The Gemma 4 drafter is a separate assistant checkpoint bound to
            // the loaded target (`Gemma4CBv2MTPDrafter`), stateless across
            // rounds. Depths are the engine's production-tested envelope.
            DecoderDeclaration(
                mode: DecoderID.mtp.rawValue, drafter: .assistantCheckpoint,
                state: .stateless,
                depth: 1 ... CBv2MTPConfig.testedMaxDraftTokens),
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
    public let loadedDecoders: [DecoderID]
    public let headProvenance: HeadProvenance?
    public let loadedModelType: String

    private let textModel: Gemma4TextModel
    private let drafter: (any CBv2MTPDrafter)?
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        textModel: Gemma4TextModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        drafter: (any CBv2MTPDrafter)?,
        headProvenance: HeadProvenance?,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.textModel = textModel
        self.servingModel = textModel
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.layerKinds = textModel.cbv2LayerKinds
        self.loadedModelType = loadedModelType
        self.drafter = drafter
        self.headProvenance = headProvenance
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
        self.loadedDecoders = drafter == nil ? [.serial] : [.serial, .mtp]
    }

    public static func load(
        _ directory: URL, options: RunnerLoadOptions
    ) async throws -> Gemma4TextRunner {
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())

        let textModel: Gemma4TextModel
        switch context.model {
        case let text as Gemma4TextModel: textModel = text
        case let wrapper as Gemma4Model: textModel = wrapper.textModel
        default:
            throw RunnerError.unexpectedModel(String(describing: type(of: context.model)))
        }

        var drafter: (any CBv2MTPDrafter)?
        var provenance: HeadProvenance?
        if let drafterDirectory = options.drafterDirectory {
            do {
                let assistant = try await Gemma4AssistantDraftModel.load(
                    from: drafterDirectory)
                drafter = try Gemma4CBv2MTPDrafter(drafter: assistant, target: textModel)
                provenance = try RunnerCheckpoint.provenance(ofHeadAt: drafterDirectory)
            } catch {
                // A drafter directory that was ASKED for and cannot load is a
                // refusal, not a silent demotion to serial: an mtp leg that
                // quietly measured serial would look like a very fast serial
                // engine.
                throw RunnerError.drafterUnavailable("\(error)")
            }
        }

        return Gemma4TextRunner(
            textModel: textModel,
            tokenizer: context.tokenizer,
            eosTokenIDs: RunnerCheckpoint.eosTokenIDs(
                at: directory, tokenizer: context.tokenizer),
            loadedModelType: modelType,
            drafter: drafter,
            headProvenance: provenance,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    public func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        let model = textModel
        return try RunnerEngineAssembly.makeEngine(
            manifest: Self.manifest,
            loadedDecoders: loadedDecoders,
            model: model,
            tokenizer: tokenizer,
            layerKinds: layerKinds,
            newCaches: { make in try model.newCacheV2(makeLayerCache: make) },
            mtpDrafter: drafter,
            build: build)
    }

    public func makeStepper() throws -> any TeacherForcedStepper {
        let model = textModel
        return CBv2SingleRowStepper(
            model: model,
            layerKinds: layerKinds,
            newCaches: { make in try model.newCacheV2(makeLayerCache: make) },
            kvBytesCapacity: kvBytesCapacity,
            maxLength: maxSequenceLength)
    }
}
