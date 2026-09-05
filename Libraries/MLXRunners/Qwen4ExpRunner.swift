// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Qwen 3.8 Flash-Next 125B-A6B (`qwen4_exp`, `qwen4_exp_text`).
//
// A HYBRID trunk like Qwen 3.5: of the 48 layers only the 12 full-attention
// ones own a key-value tape, and `cbv2LayerKinds` is that compact storage
// layout with `modelLayerIndex` mapping each row back to its decoder layer.
// The other 36 layers are gated-deltanet recurrence carried as request-owned
// recurrent state.
//
// Two things separate this family from Qwen 3.5:
//
//   * QSA. Every full-attention layer runs an indexer with a 2048-token
//     budget and emits a keep mask. The mask is not an optimization: without
//     it the model answers differently. The manifest therefore declares
//     `requiresKeepMask`, the family owns its own layer cache (a second
//     per-row indexer tape beside the key-value tape), and the engine refuses
//     to start over a cache provider that cannot apply the mask.
//   * The n-gram PLE table. It is 29.8 GiB and it is never held as model
//     parameters. The caller passes it in through
//     `RunnerLoadOptions.resources` under ``ngramRowSourceResource``, either
//     as the PATH of the n-gram shard directory or as an already built
//     `Qwen4ExpNGramRowSource`; without it a checkpoint that has PLE layers
//     is refused, because the model's forward pass cannot run.
//
// Speculation is the checkpoint's own `mtp.*` head
// (`Qwen4ExpInlineMTPAssistant`), request-stateful across rounds, depth 1...3.
// Paged storage, prefix reuse, compiled decode and packed prefill stay off,
// and only single-stream regimes are declared: the indexer scores one tape
// per call.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public final class Qwen4ExpRunner: Runner, @unchecked Sendable {

    /// Name under which the caller passes the n-gram row source in
    /// `RunnerLoadOptions.resources`.
    ///
    /// Two value shapes are accepted, and only two:
    ///
    ///   * a path, as a `URL` or a `String`, of the n-gram shard DIRECTORY
    ///     that the offline transform writes. bench-worker passes its
    ///     `--resource` value in this shape. A path to a single file is
    ///     refused by name.
    ///   * an already built `Qwen4ExpNGramRowSource`, for an in-process
    ///     caller that holds one.
    public static let ngramRowSourceResource = "qwen4exp.ngramRowSource"

    public static let manifest = RunnerManifest(
        runnerID: "layr/qwen4exp-125b-a6b",
        modelTypes: ["qwen4_exp", "qwen4_exp_text"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: false,
            supportsCompiledDecode: false,
            supportsPackedPrefill: false,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: false),
        kvBackends: [.contiguous],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil),
            DecoderDeclaration(
                mode: DecoderID.mtp.rawValue, drafter: .embeddedHead,
                state: .requestStateful,
                depth: 1 ... Qwen4ExpInlineMTPAssistant.maximumDepth),
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false),
        ],
        multimodal: false,
        recurrentLayers: true,
        requiresKeepMask: true)

    public let servingModel: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let eosTokenIDs: Set<Int>
    public let layerKinds: [CBv2LayerKind]
    public let loadedDecoders: [DecoderID]
    public let headProvenance: HeadProvenance?
    public let loadedModelType: String

    private let model: Qwen4ExpModel
    private let drafter: (any CBv2MTPDrafter)?
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        model: Qwen4ExpModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        drafter: (any CBv2MTPDrafter)?,
        headProvenance: HeadProvenance?,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.model = model
        self.servingModel = model
        self.layerKinds = model.cbv2LayerKinds
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
    ) async throws -> Qwen4ExpRunner {
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())
        guard let model = context.model as? Qwen4ExpModel else {
            throw RunnerError.unexpectedModel(String(describing: type(of: context.model)))
        }

        // The PLE layers read their rows through the injected source. A model
        // that has PLE layers and no source cannot run a forward pass at all,
        // so the load refuses here rather than at the first token.
        if !model.pleEmbeddings.isEmpty {
            model.install(
                ngramRowSource: try Self.ngramRowSource(
                    options.resources[ngramRowSourceResource], for: model))
        }

        // The head is a block of the checkpoint, not a separate artifact, so
        // a caller pointing at another directory is asking for something this
        // family does not serve.
        if let drafterDirectory = options.drafterDirectory,
            drafterDirectory.standardizedFileURL != directory.standardizedFileURL
        {
            throw RunnerError.drafterUnavailable(
                "the mtp head is embedded in the checkpoint; "
                    + "\(drafterDirectory.path) is not served")
        }

        let drafter = Qwen4ExpInlineMTPAssistant(target: model)
        // §12c: the one embedded-head rule, in the one shared helper. It
        // hashes the shards carrying the `mtp.*` tensors, not the whole
        // checkpoint.
        let provenance =
            drafter == nil
            ? nil : try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: directory)

        return Qwen4ExpRunner(
            model: model,
            tokenizer: context.tokenizer,
            eosTokenIDs: RunnerCheckpoint.eosTokenIDs(
                at: directory, tokenizer: context.tokenizer),
            loadedModelType: modelType,
            drafter: drafter,
            headProvenance: provenance,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    /// Resolve the n-gram row source from the resource the caller gave.
    ///
    /// The runner names the `Qwen4ExpNGramRowSource` seam and one construction
    /// entry point, `Qwen4ExpNGramRowSourceLoader`, and no conformer. A later
    /// conformer is chosen inside the loader, so this stays as it is.
    private static func ngramRowSource(
        _ resource: AnyObject?, for model: Qwen4ExpModel
    ) throws -> any Qwen4ExpNGramRowSource {
        if let source = resource as? Qwen4ExpNGramRowSource {
            return source
        }
        if let url = resource as? URL {
            return try Qwen4ExpNGramRowSourceLoader.rowSource(at: url, for: model)
        }
        if let path = resource as? String {
            return try Qwen4ExpNGramRowSourceLoader.rowSource(
                at: URL(fileURLWithPath: path), for: model)
        }
        throw RunnerError.resourceMissing(
            "\(ngramRowSourceResource): this checkpoint has "
                + "\(model.pleEmbeddings.count) PLE layers and the n-gram table "
                + "is never model parameters; pass the n-gram shard directory")
    }

    /// The family's own caches. `newCacheV2` still runs the vending closure
    /// for every layer and then discards what it returns: the QSA indexer
    /// needs `Qwen4ExpCBv2LayerCache`, whose second tape trims with the
    /// key-value tape so an MTP rollback keeps the two in step.
    private func newCaches(
        _ make: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) throws -> [any CBv2AttendingLayerCache] {
        try model.newCacheV2(makeLayerCache: make)
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
