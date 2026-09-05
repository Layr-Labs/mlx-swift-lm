// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Qwen 3.5 runner, dense and MoE.
//
// A HYBRID trunk: only every `full_attention_interval`-th layer owns KV, and
// the rest are gated-delta-net recurrent layers carried as request-owned
// recurrent state. `cbv2LayerKinds` is therefore the COMPACT attention
// storage layout with `modelLayerIndex` mapping each stored row back to its
// transformer layer — which is why paged is not declared: the pool's dense
// storage subscript does not survive a hybrid trunk (`supportsPagedKV` is
// false on the model too).
//
// Speculation is the checkpoint's own `mtp.*` head
// (`Qwen35InlineMTPAssistant`), request-stateful across rounds. It lives in
// the model directory unless the caller points at a separate export.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public final class Qwen35Runner: Runner, @unchecked Sendable {

    public static let manifest = RunnerManifest(
        runnerID: "layr/qwen35",
        modelTypes: ["qwen3_5", "qwen3_5_moe", "qwen3_5_text"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: false,
            supportsCompiledDecode: false,
            supportsPackedPrefill: true,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: true),
        kvBackends: [.contiguous],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil),
            DecoderDeclaration(
                mode: DecoderID.mtp.rawValue, drafter: .embeddedHead,
                state: .requestStateful,
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
        recurrentLayers: true,
        requiresKeepMask: false)

    public let servingModel: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let eosTokenIDs: Set<Int>
    public let layerKinds: [CBv2LayerKind]
    public let loadedDecoders: [DecoderID]
    public let headProvenance: HeadProvenance?
    public let loadedModelType: String

    /// Captured at load so `makeEngine`/`makeStepper` never re-switch on the
    /// concrete model type: the dense/MoE `Qwen35Model` and the text-only
    /// `Qwen35TextModel` expose the same hooks under different types.
    private let newCaches:
        (
            (_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache
        ) throws -> [any CBv2AttendingLayerCache]
    private let drafter: (any CBv2MTPDrafter)?
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        servingModel: any LanguageModel,
        layerKinds: [CBv2LayerKind],
        newCaches: @escaping (
            (_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache
        ) throws -> [any CBv2AttendingLayerCache],
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        drafter: (any CBv2MTPDrafter)?,
        headProvenance: HeadProvenance?,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.servingModel = servingModel
        self.layerKinds = layerKinds
        self.newCaches = newCaches
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.loadedModelType = loadedModelType
        self.drafter = drafter
        self.headProvenance = headProvenance
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
        self.loadedDecoders = drafter == nil ? [.serial] : [.serial, .mtp]
    }

    public static func load(
        _ directory: URL, options: RunnerLoadOptions
    ) async throws -> Qwen35Runner {
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())

        let servingModel: any LanguageModel
        let layerKinds: [CBv2LayerKind]
        let newCaches:
            (
                (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
                    any CBv2AttendingLayerCache
            ) throws -> [any CBv2AttendingLayerCache]
        switch context.model {
        case let model as Qwen35Model:
            // Covers `Qwen35MoEModel`, which subclasses it.
            servingModel = model
            layerKinds = model.cbv2LayerKinds
            newCaches = { make in try model.newCacheV2(makeLayerCache: make) }
        case let model as Qwen35TextModel:
            servingModel = model
            layerKinds = model.cbv2LayerKinds
            newCaches = { make in try model.newCacheV2(makeLayerCache: make) }
        default:
            throw RunnerError.unexpectedModel(String(describing: type(of: context.model)))
        }

        // The embedded head ships inside the checkpoint, so the drafter
        // directory DEFAULTS to the model directory. A caller that wants
        // serial passes a build with `decoder == .serial`; it does not have
        // to hide the head.
        let drafterDirectory = options.drafterDirectory ?? directory
        var drafter: (any CBv2MTPDrafter)?
        var provenance: HeadProvenance?
        do {
            let assistant = try Qwen35InlineMTPAssistant.load(
                from: drafterDirectory, target: servingModel)
            drafter = assistant
            // §12c: the one embedded-head rule, in the one shared helper.
            provenance = try RunnerCheckpoint.provenance(
                ofEmbeddedHeadAt: drafterDirectory)
        } catch {
            // An EXPLICIT drafter directory that fails is a refusal; the
            // implicit in-checkpoint head simply may not be there, and a
            // checkpoint without an `mtp.*` block is a serial-only model,
            // not a broken one.
            if options.drafterDirectory != nil {
                throw RunnerError.drafterUnavailable("\(error)")
            }
        }

        return Qwen35Runner(
            servingModel: servingModel,
            layerKinds: layerKinds,
            newCaches: newCaches,
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
        try RunnerEngineAssembly.makeEngine(
            manifest: Self.manifest,
            loadedDecoders: loadedDecoders,
            model: servingModel,
            tokenizer: tokenizer,
            layerKinds: layerKinds,
            newCaches: newCaches,
            mtpDrafter: drafter,
            build: build)
    }

    public func makeStepper() throws -> any TeacherForcedStepper {
        CBv2SingleRowStepper(
            model: servingModel,
            layerKinds: layerKinds,
            newCaches: newCaches,
            kvBytesCapacity: kvBytesCapacity,
            maxLength: maxSequenceLength)
    }
}
