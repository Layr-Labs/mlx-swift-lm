import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// In-process only. The native owner/epoch binds evaluated weights; these
/// strings must come from trusted serving configuration, not request metadata.
public struct DiffusionGemmaResidentPrefixConfiguration: Sendable {
    public let maximumBytes: Int
    public let maximumEntries: Int
    public let artifactIdentity: String
    public let templateIdentity: String
    public let numericalProfile: String

    public init(
        maximumBytes: Int, maximumEntries: Int = 8,
        artifactIdentity: String, templateIdentity: String, numericalProfile: String
    ) throws {
        guard maximumBytes > 0, maximumEntries > 0, maximumEntries <= 1024,
            [artifactIdentity, templateIdentity, numericalProfile].allSatisfy({
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.utf8.count <= 4096
            })
        else {
            throw CBv2NativeBlockError.invalidConfiguration
        }
        self.maximumBytes = maximumBytes
        self.maximumEntries = maximumEntries
        self.artifactIdentity = artifactIdentity
        self.templateIdentity = templateIdentity
        self.numericalProfile = numericalProfile
    }
}

/// Engine-queue confined: compact copies are evaluated synchronously before
/// publication. Staged and published snapshots share one bounded LRU budget.
final class DiffusionGemmaResidentPrefixCache: @unchecked Sendable {
    private struct Entry {
        let request: UUID
        let checkpoint: DiffusionGemmaPrefixCheckpoint
        let identity: DiffusionGemmaPrefixIdentity
        let chargedBytes: Int
        var published: Bool
    }
    let configuration: DiffusionGemmaResidentPrefixConfiguration
    let retainsPublishedEntries: Bool
    let persistence: DiffusionGemmaPrefixPersistence?
    private let epoch = UUID().uuidString
    private let chunkSize: Int
    private var entries = [Entry]()  // Oldest use first.
    private var byteLimit: Int
    var enabled: Bool { byteLimit > 0 }
    var retainedBytes: Int { entries.reduce(0) { $0 + $1.chargedBytes } }

    init(configuration: DiffusionGemmaResidentPrefixConfiguration, chunkSize: Int,
         retainsPublishedEntries: Bool = true, persistence: DiffusionGemmaPrefixPersistence? = nil) {
        self.configuration = configuration
        self.chunkSize = chunkSize
        self.byteLimit = configuration.maximumBytes
        self.retainsPublishedEntries = retainsPublishedEntries
        self.persistence = persistence
    }

    func identity(for request: CBv2Request) throws -> DiffusionGemmaPrefixIdentity? {
        guard request.prefixCacheEnabled, request.positionState == nil,
            let scope = request.cacheSalt,
            !scope.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            scope.utf8.count <= 4096
        else { return nil }
        let mediaIdentity: CBv2HybridPrefixIdentity?
        if let media = request.multimodal {
            guard (retainsPublishedEntries || persistence != nil), media.attention == .bidirectionalSpans,
                media.positionState == nil, media.deepstackEmbeddings == nil,
                let bound = request.hybridPrefixIdentity else { return nil }
            mediaIdentity = bound
        } else {
            guard request.hybridPrefixIdentity == nil else { return nil }
            mediaIdentity = nil
        }
        return try .init(
            tenantScope: request.checkpointCacheSalt ?? scope, artifact: configuration.artifactIdentity,
            template: configuration.templateIdentity,
            media: mediaIdentity.map { "media:" + $0.digest.map { String(format: "%02x", $0) }.joined() } ?? "text-only",
            numericalProfile: configuration.numericalProfile + ":prefill-\(chunkSize)", epoch: epoch,
            mediaIdentity: mediaIdentity
        )
    }

    func lookup(tokens: [Int32], identity: DiffusionGemmaPrefixIdentity,
                geometry: DiffusionGemmaPrefillGeometry? = nil)
        -> DiffusionGemmaPrefixCheckpoint?
    {
        guard identity.mediaIdentity == nil || geometry != nil else { return nil }
        let candidates = entries.indices.filter {
            let entry = entries[$0]
            // An appended request may reuse only the exact cold chunk boundary.
            // Full short/unaligned snapshots remain usable for identical prompts.
            return entry.published && entry.checkpoint.matches(tokens: tokens, identity: identity)
                && (entry.checkpoint.tokenCount == tokens.count
                    || (geometry?.permitsRestore(position: entry.checkpoint.tokenCount)
                        ?? (entry.checkpoint.tokenCount % chunkSize == 0)))
        }
        guard
            let index = candidates.max(by: {
                entries[$0].checkpoint.tokenCount < entries[$1].checkpoint.tokenCount
            })
        else { return nil }
        let hit = entries.remove(at: index)
        entries.append(hit)
        return hit.checkpoint
    }

    static func metadataCharge(tokens: Int, layers: Int, identity: DiffusionGemmaPrefixIdentity)
        -> Int
    {
        // Token-array capacity + bounded bookkeeping, separate from device KV.
        tokens * 8 + layers * 512 + 1024
            + [
                identity.tenantScope, identity.artifact, identity.template, identity.media,
                identity.numericalProfile, identity.epoch,
            ].reduce(0) { $0 + $1.utf8.count * 2 }
    }

    func capture(
        model: DiffusionGemma, cache: DiffusionGemmaRequestCache,
        promptTokens: [Int32], request: UUID, identity: DiffusionGemmaPrefixIdentity,
        geometry: DiffusionGemmaPrefillGeometry? = nil
    ) throws {
        let promptCount = promptTokens.count
        let position = cache.position
        guard identity.mediaIdentity == nil || geometry != nil else { return }
        let captures = geometry?.capturePositions
            ?? Set([promptCount, promptCount / chunkSize * chunkSize, min(promptCount, chunkSize)])
        let firstBoundary = geometry?.boundaries.first ?? min(promptCount, chunkSize)
        let lastBoundary = geometry?.lastStableBoundary ?? (promptCount / chunkSize * chunkSize)
        guard position > 0, captures.contains(position)
        else { return }
        let bytes = try
            cache.stateArrays().reduce(0) { try $0 + Memory.allocationFootprintUpperBound(byteCount: $1.nbytes) }
            + Self.metadataCharge(
                tokens: position, layers: cache.configuration.layerCount, identity: identity)
        guard bytes > 0, bytes <= byteLimit else { return }
        // Preserve the donor's aligned branch point when an unaligned exact
        // endpoint cannot coexist; losing it would turn every appended turn cold.
        func protected(_ entry: Entry) -> Bool {
            entry.checkpoint.matches(tokens: promptTokens, identity: identity)
                && (entry.checkpoint.tokenCount == firstBoundary
                    || (entry.checkpoint.tokenCount == lastBoundary && position != lastBoundary))
        }
        while retainedBytes > byteLimit - bytes || entries.count >= configuration.maximumEntries {
            guard let victim = entries.firstIndex(where: { !protected($0) }) else { return }
            entries.remove(at: victim)
        }
        let checkpoint = try model.model.decoder.checkpoint(
            cache: cache, identity: identity, compact: true)
        guard checkpoint.physicalRetainedBytes <= bytes else {
            throw CBv2NativeBlockError.invalidConfiguration
        }
        entries.append(
            .init(
                request: request, checkpoint: checkpoint, identity: identity,
                chargedBytes: bytes, published: false))
    }

    func finish(request: UUID, successful: Bool, input: CBv2Request? = nil,
                reused: DiffusionGemmaPrefixCheckpoint? = nil) {
        if successful {
            let staged = entries.filter { $0.request == request && !$0.published }
            entries.removeAll { $0.request == request && !$0.published }
            if let persistence, let input {
                let candidates = staged.map(\.checkpoint) + (reused.map { [$0] } ?? [])
                if let donor = candidates.filter({
                    persistence.accepts($0, request: input)
                }).max(by: { $0.tokenCount < $1.tokenCount }) {
                    persistence.donate(donor, request: input)
                }
            }
            guard retainsPublishedEntries else { return }
            for var entry in staged {
                entries.removeAll { $0.published && $0.checkpoint.samePrefix(as: entry.checkpoint) }
                entry.published = true
                entries.append(entry)
            }
        } else {
            entries.removeAll { $0.request == request && !$0.published }
        }
    }

    func trim(to bytes: Int) {
        byteLimit = min(max(0, bytes), configuration.maximumBytes)
        while retainedBytes > byteLimit { entries.removeFirst() }
    }
    func clear() {
        entries.removeAll()
        byteLimit = 0
        persistence?.store.close()
    }
}
