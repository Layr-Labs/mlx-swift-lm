import Foundation

extension CBv2CompleteCheckpointManifest {
    /// Prompts longer than this store their prefix tokens bit-packed. As a
    /// JSON array, about 150K-190K tokens of a 200K-token vocabulary already
    /// fill the 1 MiB manifest bound; packed, 262,144 18-bit tokens take
    /// about 786 KB. Shorter prefixes keep the array, so every checkpoint
    /// that fits today is encoded exactly as before.
    public static let packedPrefixTokenThreshold = 65_536

    /// The one encoding of a manifest's prefix tokens: packed for the native
    /// block layout and for any prefix longer than the threshold, a JSON
    /// array otherwise. Decoding refuses the other form.
    static func packsPrefixTokens(backendLayout: String, position: Int) -> Bool {
        backendLayout == diffusionBlockLayout || position > packedPrefixTokenThreshold
    }

    /// The manifest exactly as the provider's encrypted envelope encodes it
    /// (sorted keys, slashes unescaped), whose size the store bounds.
    public func storageEncoding() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, identity, backendLayout, position, chunkSize
        case prefixTokens, cacheSalt, assistantCodecID, tensors, attentionLayers
        case mediaIdentity, mediaTargetOnly, nativeBlockState, packedPrefixTokens
        case checkpointQuantization, checkpointNativeDTypes
        case glm5NextState
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let backendLayout = try values.decode(String.self, forKey: .backendLayout)
        let position = try values.decode(Int.self, forKey: .position)
        let tokens: [Int]
        if Self.packsPrefixTokens(backendLayout: backendLayout, position: position) {
            guard !values.contains(.prefixTokens) else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            tokens = try values.decode(
                CBv2NativeBlockPackedTokens.self, forKey: .packedPrefixTokens
            )
            .unpack(count: position)
        } else {
            guard !values.contains(.packedPrefixTokens) else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            tokens = try values.decode([Int].self, forKey: .prefixTokens)
        }
        self.init(
            schemaVersion: try values.decode(Int.self, forKey: .schemaVersion),
            identity: try values.decode(CBv2CompleteCheckpointIdentity.self, forKey: .identity),
            backendLayout: backendLayout,
            position: position,
            chunkSize: try values.decode(Int.self, forKey: .chunkSize),
            cacheSalt: try values.decodeIfPresent(String.self, forKey: .cacheSalt),
            assistantCodecID: try values.decodeIfPresent(String.self, forKey: .assistantCodecID),
            mediaIdentity: try values.decodeIfPresent(
                CBv2HybridPrefixIdentity.self, forKey: .mediaIdentity),
            mediaTargetOnly: try values.decodeIfPresent(Bool.self, forKey: .mediaTargetOnly)
                ?? false,
            nativeBlockState: try values.decodeIfPresent(
                CBv2NativeBlockCheckpointState.self, forKey: .nativeBlockState),
            glm5NextState: try values.decodeIfPresent(
                CBv2GLM5NextCheckpointState.self, forKey: .glm5NextState),
            checkpointQuantization: try values.decodeIfPresent(
                PagedKVQuantizationConfig.self, forKey: .checkpointQuantization),
            checkpointNativeDTypes: try values.decodeIfPresent(
                [CBv2CheckpointDType].self, forKey: .checkpointNativeDTypes),
            metadata: .init(
                tokens: tokens,
                tensors: try values.decode([CBv2CheckpointTensorDescriptor].self, forKey: .tensors),
                attentionLayers: try values.decodeIfPresent(
                    [CBv2CheckpointAttentionLayer].self, forKey: .attentionLayers)))
    }

    public func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(schemaVersion, forKey: .schemaVersion)
        try values.encode(identity, forKey: .identity)
        try values.encode(backendLayout, forKey: .backendLayout)
        try values.encode(position, forKey: .position)
        try values.encode(chunkSize, forKey: .chunkSize)
        if Self.packsPrefixTokens(backendLayout: backendLayout, position: position) {
            try values.encode(
                CBv2NativeBlockPackedTokens(tokens: prefixTokens), forKey: .packedPrefixTokens)
        } else {
            try values.encode(prefixTokens, forKey: .prefixTokens)
        }
        try values.encodeIfPresent(cacheSalt, forKey: .cacheSalt)
        try values.encodeIfPresent(assistantCodecID, forKey: .assistantCodecID)
        try values.encodeIfPresent(mediaIdentity, forKey: .mediaIdentity)
        if mediaTargetOnly { try values.encode(true, forKey: .mediaTargetOnly) }
        try values.encodeIfPresent(nativeBlockState, forKey: .nativeBlockState)
        try values.encodeIfPresent(glm5NextState, forKey: .glm5NextState)
        try values.encodeIfPresent(checkpointQuantization, forKey: .checkpointQuantization)
        try values.encodeIfPresent(checkpointNativeDTypes, forKey: .checkpointNativeDTypes)
        try values.encode(tensors, forKey: .tensors)
        try values.encodeIfPresent(attentionLayers, forKey: .attentionLayers)
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.schemaVersion == rhs.schemaVersion && lhs.identity == rhs.identity
            && lhs.backendLayout == rhs.backendLayout && lhs.position == rhs.position
            && lhs.chunkSize == rhs.chunkSize && lhs.prefixTokens == rhs.prefixTokens
            && lhs.cacheSalt == rhs.cacheSalt && lhs.assistantCodecID == rhs.assistantCodecID
            && lhs.mediaIdentity == rhs.mediaIdentity && lhs.mediaTargetOnly == rhs.mediaTargetOnly
            && lhs.nativeBlockState == rhs.nativeBlockState
            && lhs.glm5NextState == rhs.glm5NextState
            && lhs.checkpointQuantization == rhs.checkpointQuantization
            && lhs.checkpointNativeDTypes == rhs.checkpointNativeDTypes
            && lhs.tensors == rhs.tensors && lhs.attentionLayers == rhs.attentionLayers
    }
}
