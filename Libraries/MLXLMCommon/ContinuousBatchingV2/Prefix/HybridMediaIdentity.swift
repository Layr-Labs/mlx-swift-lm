import CryptoKit
import Foundation

/// Provider-computed identity of evaluated media embeddings, ordered spans,
/// attention and complete request positions. No pixels or tensor bytes are
/// retained here. A digest is necessary, but not alone sufficient, for reuse.
public struct CBv2HybridPrefixIdentity: Codable, Sendable, Equatable, Hashable {
    public let digest: Data

    public init(digest: Data) throws {
        guard digest.count == 32 else { throw CBv2CompleteCheckpointError.invalidManifest }
        self.digest = digest
    }

    private enum CodingKeys: String, CodingKey { case digest }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(digest: container.decode(Data.self, forKey: .digest))
    }

    /// A media namespace cannot collide with the nil/empty original tenant
    /// scopes. Call once on the original authenticated scope, never on an
    /// already-bound scope. Text requests keep their old scope byte-for-byte.
    func binding(cacheSalt: String?) -> String {
        var hash = SHA256()
        hash.update(data: Data("darkbloom.hybrid-media-scope.v1".utf8))
        hash.update(data: Data([cacheSalt == nil ? 0 : 1]))
        let bytes = Data((cacheSalt ?? "").utf8)
        var count = UInt64(bytes.count).bigEndian
        withUnsafeBytes(of: &count) { hash.update(data: Data($0)) }
        hash.update(data: bytes)
        hash.update(data: digest)
        return "media-v1:" + hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

extension CBv2Request {
    /// The bridge must use this same resolved scope for staging and receipts.
    /// cacheSalt remains the original authenticated tenant scope.
    public var checkpointCacheSalt: String? {
        hybridPrefixIdentity?.binding(cacheSalt: cacheSalt) ?? cacheSalt
    }

    var hasOutOfBandCheckpointInput: Bool { multimodal != nil || positionState != nil }
    var hasBoundCheckpointInput: Bool { !hasOutOfBandCheckpointInput || hybridPrefixIdentity != nil }
    /// Persistent-history MTP deliberately does not speculate on media spans.
    /// Such a request must donate/restore target state without fabricated heads.
    var usesTargetOnlyMediaCheckpoint: Bool { multimodal?.spans.isEmpty == false }

    func permitsHybridCheckpoint(layerKinds: [CBv2LayerKind]) -> Bool {
        guard prefixCacheEnabled, hasBoundCheckpointInput else { return false }
        guard hasOutOfBandCheckpointInput else { return true }
        // This restoration is qualified for native Qwen4's three-plane QSA
        // state, not blanket cache admission for other media architectures.
        return layerKinds.contains { $0.qwen4IndexerCompressRatio != nil }
            && (positionState ?? multimodal?.positionState)?.axisCount == 3
    }
}
