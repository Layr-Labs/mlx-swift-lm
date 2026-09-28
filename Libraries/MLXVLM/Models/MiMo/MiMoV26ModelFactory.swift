// Copyright © 2026 Eigen Labs.
import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

public enum MiMoV26FactoryError: Error, Equatable, LocalizedError, Sendable {
    case invalidMetadata(String)
    case incompatiblePreparation
    case changedTokenizerAssets
    case unsupportedInput(String)
    case nativeCBv2Required

    public var errorDescription: String? {
        switch self {
        case .invalidMetadata(let reason): "Native MiMo factory metadata: \(reason)."
        case .incompatiblePreparation: "Tokenizer preparation does not belong to this exact native MiMo load session."
        case .changedTokenizerAssets: "Native MiMo tokenizer/template assets changed after preparation. Prepare a fresh load session."
        case .unsupportedInput(let reason): "Native MiMo text input: \(reason)."
        case .nativeCBv2Required: "MiMo V2.6 generation requires MiMoV26LoadedModel.makeCBv2Binding() and the native CBv2 engine; this entrypoint does not enable the generic TokenIterator path."
        }
    }
}

/// Explicit two-stage factory. It is deliberately absent from public registries
/// and has no downloader/directory-only overload that could bypass admission.
public enum MiMoV26ModelFactory {
    /// Only factory-created metadata/tokenizer state crosses the async boundary.
    /// No model, native reservation or non-Sendable owner is sent between tasks.
    public struct Prepared: Sendable {
        public let request: MiMoV26SerialLoadRequest
        public let configuration: ResolvedModelConfiguration
        public let nativeConfiguration: MiMoV26Configuration
        public let templateSHA256: String
        public let stopTokenIDs: Set<Int>
        public let processor: MiMoV26TextProcessor
        let tokenizer: any Tokenizer
        let nativeEOS: Set<Int>
        let metadata: Metadata
    }

    struct Asset: Equatable, Sendable {
        let name: String
        let object: MiMoV26FilesystemObjectState
    }
    struct Metadata: Equatable, Sendable {
        let root: URL
        let assets: [Asset]
        let configSHA256: String
        let tokenizerConfigSHA256: String
        let template: String
        let templateSource: String
        let generationSHA256: String?
        let nativeEOS: Set<Int>
    }
    private static let assetNames = ["config.json", "generation_config.json", "chat_template.jinja",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "tokenizer.model", "merges.txt", "vocab.json", "vocab.txt"]
    private static let maximumSmallMetadataBytes = 8 << 20

    public static func prepare(request: MiMoV26SerialLoadRequest,
                               configuration: ResolvedModelConfiguration,
                               tokenizerLoader: any TokenizerLoader) async throws -> Prepared {
        try Task.checkCancellation()
        let metadata = try readMetadata(request: request, configuration: configuration)
        let native = try JSONDecoder().decode(MiMoV26Configuration.self,
            from: readSmall(metadata.root.appendingPathComponent("config.json")))
        guard configuration.eosTokenIds.isEmpty || configuration.eosTokenIds == metadata.nativeEOS else {
            throw MiMoV26FactoryError.invalidMetadata("resolved EOS IDs conflict with checkpoint EOS; use extraEOSTokens for explicit additions")
        }
        // The protocol is explicitly local. No Downloader, alternate root,
        // fallback architecture or tokenizer substitution is selected here.
        let tokenizer = try await tokenizerLoader.load(from: metadata.root)
        try Task.checkCancellation()
        guard try readMetadata(request: request, configuration: configuration) == metadata else {
            throw MiMoV26FactoryError.changedTokenizerAssets
        }
        let processor = try MiMoV26TextProcessor(tokenizer: tokenizer, chatTemplate: metadata.template,
            vocabularySize: native.vocabularySize, maximumSequenceLength: native.maxPositionEmbeddings)
        var stopIDs = metadata.nativeEOS
        if let spelling = tokenizer.eosToken {
            guard let eos = tokenizer.convertTokenToId(spelling) else {
                throw MiMoV26FactoryError.invalidMetadata("tokenizer EOS spelling has no token ID")
            }
            stopIDs.insert(eos)
        }
        for spelling in configuration.extraEOSTokens {
            guard let id = tokenizer.convertTokenToId(spelling) else {
                throw MiMoV26FactoryError.invalidMetadata("unresolved explicit EOS token")
            }
            stopIDs.insert(id)
        }
        guard stopIDs.allSatisfy({ $0 >= 0 && $0 < native.vocabularySize }) else {
            throw MiMoV26FactoryError.invalidMetadata("EOS outside the native vocabulary")
        }
        // Force template/tokenizer failures before consuming the native session.
        // This verifies delegation/control response, not real tokenizer parity.
        let unset = try processor.renderTokens(input: UserInput(prompt: ""))
        let enabled = try processor.renderTokens(input: UserInput(prompt: "", additionalContext: ["enable_thinking": true]))
        let disabled = try processor.renderTokens(input: UserInput(prompt: "", additionalContext: ["enable_thinking": false]))
        guard unset == enabled, disabled != enabled else {
            throw MiMoV26FactoryError.invalidMetadata("tokenizer/template does not preserve MiMo's default-on Boolean thinking contract")
        }
        try Task.checkCancellation()
        guard try readMetadata(request: request, configuration: configuration) == metadata else {
            throw MiMoV26FactoryError.changedTokenizerAssets
        }
        return .init(request: request, configuration: configuration, nativeConfiguration: native,
            templateSHA256: hash(Data(metadata.template.utf8)), stopTokenIDs: stopIDs,
            processor: processor, tokenizer: tokenizer, nativeEOS: metadata.nativeEOS, metadata: metadata)
    }

    /// Synchronous actor-local construction. Retain work before calling and
    /// keep it until process restart if any captured stream fails completion.
    public static func load(session: MiMoV26SerialLoadSession,
                            reservation: any MiMoV26SerialLoadReservation,
                            prepared: Prepared, retaining work: NativeConstructionScope,
                            isCancelled: () -> Bool = { false },
                            progress: (MiMoV26SerialLoadProgress) throws -> Void = { _ in }) throws -> ModelContext {
        try work.withPhase(.modelFactory) {
            try work.retainOwner(reservation)
            guard prepared.request == session.request, reservation.request == session.request else {
                throw MiMoV26FactoryError.incompatiblePreparation
            }
            guard reservation.reservedLoadBytes >= session.request.requiredLoadBytes else {
                throw MiMoV26SerialLoadError.insufficientReservation
            }
            guard !Task.isCancelled, !isCancelled() else { throw MiMoV26SerialLoadError.cancelled }
            guard try readMetadata(request: session.request, configuration: prepared.configuration) == prepared.metadata else {
                throw MiMoV26FactoryError.changedTokenizerAssets
            }
            try work.capture(StreamOrDevice.default.stream)
            let loaded = try session.load(reservation: reservation, retaining: work,
                isCancelled: { Task.isCancelled || isCancelled() }, progress: progress)
            // The outer scope still owns the exact four-component bundle here.
            try work.checkpoint("factory.beforePublication")
            guard !Task.isCancelled else { throw MiMoV26SerialLoadError.cancelled }
            guard try readMetadata(request: session.request, configuration: prepared.configuration) == prepared.metadata else {
                throw MiMoV26FactoryError.changedTokenizerAssets
            }
            let customCancelled = isCancelled()
            guard !Task.isCancelled, !customCancelled else { throw MiMoV26SerialLoadError.cancelled }
            let model = MiMoV26LoadedModel(loaded: loaded, prepared: prepared)
            try work.retainOwner(model)
            let config = ModelConfiguration(directory: prepared.metadata.root,
                defaultPrompt: prepared.configuration.defaultPrompt,
                extraEOSTokens: prepared.configuration.extraEOSTokens,
                eosTokenIds: prepared.nativeEOS,
                toolCallFormat: prepared.configuration.toolCallFormat ?? ToolCallFormat.infer(from: "mimo_v2"))
            return ModelContext(configuration: config, model: model,
                processor: prepared.processor, tokenizer: prepared.tokenizer)
        }
    }

    /// The host preinstalls work in its strong transaction before this await.
    /// The separately shared reservation must be genuinely synchronized.
    public static func loadContainer(session: consuming MiMoV26SerialLoadSession,
                                     reservation: any MiMoV26SerialLoadReservation & Sendable,
                                     prepared: Prepared, retaining work: NativeConstructionWork,
                                     isCancelled: @escaping @Sendable () -> Bool = { false },
                                     progress: @escaping @Sendable (MiMoV26SerialLoadProgress) throws -> Void = { _ in }) async throws -> ModelContainer {
        let session = SendableBox(session)
        return try await work.constructContainer { scope in
            ModelContainer(context: try load(session: session.consume(), reservation: reservation,
                prepared: prepared, retaining: scope, isCancelled: isCancelled, progress: progress))
        }
    }

    /// Construction only, before an active engine exists. The fixed work ->
    /// container lock order serializes the local model/scope across the await.
    /// Return existing Sendable handles/metadata, never a raw model or binding.
    public static func withNativeConstruction<Result: Sendable>(
        container: ModelContainer, retaining work: NativeConstructionWork,
        _ body: @escaping @Sendable (MiMoV26LoadedModel, NativeConstructionScope) throws -> Result
    ) async throws -> Result {
        try await work.withContainer(container) { context, scope in
            guard let model = context.model as? MiMoV26LoadedModel else {
                throw MiMoV26FactoryError.incompatiblePreparation
            }
            try scope.authorizeImmutableLoadedOwner(model.resources)
            return try body(model, scope)
        }
    }

    private static func readMetadata(request: MiMoV26SerialLoadRequest,
                                     configuration: ResolvedModelConfiguration) throws -> Metadata {
        let root = configuration.modelDirectory.resolvingSymlinksInPath().standardizedFileURL
        let tokenizerRoot = configuration.tokenizerDirectory.resolvingSymlinksInPath().standardizedFileURL
        guard root.isFileURL, tokenizerRoot == root, root.path == request.binding.canonicalRoot else {
            throw MiMoV26FactoryError.invalidMetadata("model and tokenizer must use the exact bound local checkpoint root")
        }
        let assets = try captureAssets(root)
        guard assets.contains(where: { $0.name == "tokenizer.json" }),
              assets.contains(where: { $0.name == "tokenizer_config.json" }) else {
            throw MiMoV26FactoryError.invalidMetadata("native checkpoint tokenizer files are missing")
        }
        let data = try readSmall(root.appendingPathComponent("config.json"))
        guard hash(data) == request.binding.configSHA256 else { throw MiMoV26FactoryError.incompatiblePreparation }
        let config = try JSONDecoder().decode(MiMoV26Configuration.self, from: data)
        let tokenizerData = try readSmall(root.appendingPathComponent("tokenizer_config.json"))
        guard let tokenizerConfig = try JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any] else {
            throw MiMoV26FactoryError.invalidMetadata("tokenizer_config.json requires an object")
        }
        let template: String, source: String
        if assets.contains(where: { $0.name == "chat_template.jinja" }) {
            guard let text = String(data: try readSmall(root.appendingPathComponent("chat_template.jinja")), encoding: .utf8) else {
                throw MiMoV26FactoryError.invalidMetadata("template is not UTF-8")
            }
            template = text; source = "chat_template.jinja"
        } else if let text = tokenizerConfig["chat_template"] as? String {
            template = text; source = "tokenizer_config.json:chat_template"
        } else { throw MiMoV26FactoryError.invalidMetadata("checkpoint has no unambiguous native chat template") }
        guard !template.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MiMoV26FactoryError.invalidMetadata("checkpoint chat template is empty")
        }
        var eos = config.eosTokenIDs
        var generationHash: String?
        if assets.contains(where: { $0.name == "generation_config.json" }) {
            let generation = try readSmall(root.appendingPathComponent("generation_config.json"))
            generationHash = hash(generation)
            if let explicit = try JSONDecoder().decode(GenerationConfigFile.self, from: generation).eosTokenIds {
                eos = Set(explicit.values)
            }
        }
        guard eos.allSatisfy({ $0 >= 0 && $0 < config.vocabularySize }) else {
            throw MiMoV26FactoryError.invalidMetadata("checkpoint EOS outside native vocabulary")
        }
        guard try captureAssets(root) == assets else { throw MiMoV26FactoryError.changedTokenizerAssets }
        return .init(root: root, assets: assets, configSHA256: hash(data),
            tokenizerConfigSHA256: hash(tokenizerData), template: template,
            templateSource: source, generationSHA256: generationHash, nativeEOS: eos)
    }

    private static func captureAssets(_ root: URL) throws -> [Asset] {
        try assetNames.compactMap { name in
            let url = root.appendingPathComponent(name)
            guard let object = try mimoFactoryCurrentObject(url, allowMissing: true) else { return nil }
            return .init(name: name, object: object)
        }
    }
    private static func readSmall(_ url: URL) throws -> Data {
        try MiMoV26FactoryMetadataFile(url).read(maximumBytes: maximumSmallMetadataBytes)
    }
    private static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

// The existing filesystem gate's stat helpers are private. Reuse its public
// object-state representation with the same regular-file/ns-time semantics.
private func mimoFactoryObject(_ value: stat, name: String) throws -> MiMoV26FilesystemObjectState {
    guard mode_t(value.st_mode) & mode_t(S_IFMT) == mode_t(S_IFREG),
          let bytes = Int(exactly: value.st_size), bytes >= 0 else {
        throw MiMoV26FactoryError.invalidMetadata("non-regular tokenizer asset: \(name)")
    }
    #if canImport(Darwin)
    let modified = value.st_mtimespec, changed = value.st_ctimespec
    #else
    let modified = value.st_mtim, changed = value.st_ctim
    #endif
    return .init(device: String(value.st_dev), inode: String(value.st_ino), bytes: bytes,
        modifiedSeconds: Int64(modified.tv_sec), modifiedNanoseconds: Int64(modified.tv_nsec),
        changedSeconds: Int64(changed.tv_sec), changedNanoseconds: Int64(changed.tv_nsec))
}

private func mimoFactoryCurrentObject(_ url: URL, allowMissing: Bool = false) throws -> MiMoV26FilesystemObjectState? {
    guard url.isFileURL else { throw MiMoV26FactoryError.invalidMetadata("tokenizer asset is not local") }
    var value = stat()
    if lstat(url.path, &value) != 0 {
        if allowMissing && errno == ENOENT { return nil }
        throw MiMoV26FactoryError.invalidMetadata("unreadable tokenizer asset: \(url.lastPathComponent)")
    }
    guard url.resolvingSymlinksInPath().standardizedFileURL == url else {
        throw MiMoV26FactoryError.invalidMetadata("unbound tokenizer asset: \(url.lastPathComponent)")
    }
    return try mimoFactoryObject(value, name: url.lastPathComponent)
}

/// Internal bounded-reader seam. The only injection is a per-call test action
/// after validation and before the first bounded read; no global mutable hook.
final class MiMoV26FactoryMetadataFile {
    private let url: URL
    private let handle: FileHandle
    let object: MiMoV26FilesystemObjectState
    private(set) var bytesRead = 0
    private(set) var largestReadRequest = 0
    private var consumed = false

    init(_ url: URL) throws {
        self.url = url
        guard let before = try mimoFactoryCurrentObject(url) else {
            throw MiMoV26FactoryError.changedTokenizerAssets
        }
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        guard descriptor >= 0 else { throw MiMoV26FactoryError.invalidMetadata("cannot open tokenizer metadata") }
        do {
            var value = stat()
            guard fstat(descriptor, &value) == 0,
                  try mimoFactoryObject(value, name: url.lastPathComponent) == before,
                  try mimoFactoryCurrentObject(url) == before else {
                throw MiMoV26FactoryError.changedTokenizerAssets
            }
            object = before
            handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        } catch { close(descriptor); throw error }
    }

    private func validate() throws {
        var value = stat()
        guard fstat(handle.fileDescriptor, &value) == 0,
              try mimoFactoryObject(value, name: url.lastPathComponent) == object,
              try mimoFactoryCurrentObject(url) == object else {
            throw MiMoV26FactoryError.changedTokenizerAssets
        }
    }

    func read(maximumBytes: Int, beforeFirstRead: () throws -> Void = {}) throws -> Data {
        guard !consumed else { throw MiMoV26FactoryError.invalidMetadata("metadata reader was reused") }
        consumed = true
        guard maximumBytes >= 0, maximumBytes < Int.max, object.bytes <= maximumBytes else {
            throw MiMoV26FactoryError.invalidMetadata("metadata exceeds bounded size or reader was reused")
        }
        var data = Data()
        data.reserveCapacity(object.bytes)
        var first = true
        while data.count <= maximumBytes {
            try validate()
            if first { first = false; try beforeFirstRead() }
            let requested = min(65_536, maximumBytes + 1 - data.count)
            largestReadRequest = max(largestReadRequest, requested)
            let chunk = try handle.read(upToCount: requested) ?? Data()
            bytesRead += chunk.count
            try validate()
            guard chunk.count <= maximumBytes - data.count else {
                throw MiMoV26FactoryError.invalidMetadata("metadata exceeds bounded size")
            }
            if chunk.isEmpty { break }
            data.append(chunk)
        }
        try validate()
        guard data.count == object.bytes else {
            throw MiMoV26FactoryError.changedTokenizerAssets
        }
        return data
    }
}
