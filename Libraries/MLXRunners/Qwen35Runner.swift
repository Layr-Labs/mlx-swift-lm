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
import MLXVLM
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

    /// The dense/MoE `Qwen35Model`, the text-only `Qwen35TextModel` and the
    /// MULTIMODAL `MLXVLM.Qwen35` wrapper all resolve here. Resolved ONCE so
    /// `makeEngine` and `makeStepper` never re-switch.
    ///
    /// Unlike Gemma 4, the VLM wrapper does not own an MLXLLM target: it
    /// carries its own inline text model, so CBv2 needs a separate
    /// `Qwen35Model` built over the SAME immutable weight arrays. That is
    /// `QwenVLMTextExtraction`, ported here from the provider — it shares
    /// arrays, copies nothing, and reads no tensor from disk.
    static func hooks(
        of model: any LanguageModel,
        directory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> (
        serving: any LanguageModel,
        layerKinds: [CBv2LayerKind],
        newCaches: (
            (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
                any CBv2AttendingLayerCache
        ) throws -> [any CBv2AttendingLayerCache]
    ) {
        switch model {
        case let model as Qwen35Model:
            // Covers `Qwen35MoEModel`, which subclasses it.
            return (
                model, model.cbv2LayerKinds,
                { make in try model.newCacheV2(makeLayerCache: make) }
            )
        case let model as Qwen35TextModel:
            return (
                model, model.cbv2LayerKinds,
                { make in try model.newCacheV2(makeLayerCache: make) }
            )
        case is MLXVLM.Qwen35:
            // Extraction is memoized per WRAPPER INSTANCE, so the target the
            // engine steps and the target a drafter binds to are the same
            // object — MTP gates on that identity.
            let target = try QwenVLMTextExtraction.target(
                for: model, directory: directory, environment: environment)
            return (
                target, target.cbv2LayerKinds,
                { make in try target.newCacheV2(makeLayerCache: make) }
            )
        default:
            throw RunnerError.unexpectedModel(String(describing: type(of: model)))
        }
    }

    public static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> Qwen35Runner {
        // Checkpoint facts FIRST, module second: these two reads are the
        // whole of this method's filesystem access.
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let eosTokenIDs = RunnerCheckpoint.eosTokenIDs(
            at: directory, tokenizer: tokenizer)
        let hooks = try Self.hooks(
            of: model, directory: directory, environment: options.environment)

        // The embedded head ships INSIDE the checkpoint, so the provenance
        // of a head loaded from there is the checkpoint's own directory.
        // §12c: the one embedded-head rule, in the one shared helper.
        //
        // FAIL CLOSED. The provenance is sealed into the hello, so a head
        // that cannot be hashed must stop the adoption. A checkpoint that
        // declares `mtp.*` tensors but whose index cannot be read is broken,
        // and swallowing that error would serve an UNATTRIBUTED head. Only a
        // checkpoint with no head at all gives nil, which the helper returns
        // without throwing.
        var provenance: HeadProvenance?
        if options.preloadedDrafter != nil {
            provenance = try RunnerCheckpoint.provenance(
                ofEmbeddedHeadAt: options.drafterDirectory ?? directory)
        }

        return Qwen35Runner(
            servingModel: hooks.serving,
            layerKinds: hooks.layerKinds,
            newCaches: hooks.newCaches,
            tokenizer: tokenizer,
            eosTokenIDs: eosTokenIDs,
            loadedModelType: modelType,
            drafter: options.preloadedDrafter,
            headProvenance: provenance,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    /// The head ships inside the checkpoint, so the drafter directory
    /// DEFAULTS to the model directory. Reading it is why this is not part
    /// of `adopt`.
    public static func loadDrafter(
        options: RunnerLoadOptions,
        directory: URL,
        target: any LanguageModel
    ) async throws -> (any CBv2MTPDrafter)? {
        if let preloaded = options.preloadedDrafter { return preloaded }
        let drafterDirectory = options.drafterDirectory ?? directory
        do {
            return try Qwen35InlineMTPAssistant.load(
                from: drafterDirectory,
                target: try Self.hooks(
                    of: target, directory: directory,
                    environment: options.environment
                ).serving)
        } catch {
            // An EXPLICIT drafter directory that fails is a refusal; the
            // implicit in-checkpoint head simply may not be there, and a
            // checkpoint without an `mtp.*` block is a serial-only model,
            // not a broken one.
            if options.drafterDirectory != nil {
                throw RunnerError.drafterUnavailable("\(error)")
            }
            return nil
        }
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
