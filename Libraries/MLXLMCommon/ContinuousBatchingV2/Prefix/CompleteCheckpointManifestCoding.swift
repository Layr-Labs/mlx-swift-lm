import Foundation

extension CBv2CompleteCheckpointManifest {
    private enum CodingKeys: String, CodingKey {
        case schemaVersion, identity, backendLayout, position, chunkSize
        case prefixTokens, cacheSalt, assistantCodecID, tensors, attentionLayers
        case mediaIdentity, mediaTargetOnly, nativeBlockState, packedPrefixTokens,
            tokenByteTopologies
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let backendLayout = try values.decode(String.self, forKey: .backendLayout)
        let position = try values.decode(Int.self, forKey: .position)
        let tokens: [Int]
        if backendLayout == Self.diffusionBlockLayout {
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
            metadata: .init(
                tokens: tokens,
                tensors: try values.decode([CBv2CheckpointTensorDescriptor].self, forKey: .tensors),
                attentionLayers: try values.decodeIfPresent(
                    [CBv2CheckpointAttentionLayer].self, forKey: .attentionLayers),
                tokenByteTopologies: try Self.decodeTokenByteTopologies(from: values)))
    }

    /// Provider I/O reserves the positive host envelope before decoding. Keep
    /// the new fixed-record array inside that envelope even for invalid input.
    private static func decodeTokenByteTopologies(
        from values: KeyedDecodingContainer<CodingKeys>
    ) throws -> [CBv2CheckpointTokenByteTopology]? {
        guard values.contains(.tokenByteTopologies),
            try !values.decodeNil(forKey: .tokenByteTopologies)
        else { return nil }
        var records = try values.nestedUnkeyedContainer(forKey: .tokenByteTopologies)
        if let count = records.count {
            guard count >= 0, count <= maximumTokenByteTopologyRecordCount else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
        }
        var result: [CBv2CheckpointTokenByteTopology] = []
        result.reserveCapacity(records.count ?? 0)
        while !records.isAtEnd {
            guard result.count < maximumTokenByteTopologyRecordCount else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            result.append(try records.decode(CBv2CheckpointTokenByteTopology.self))
        }
        return result
    }

    public func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(schemaVersion, forKey: .schemaVersion)
        try values.encode(identity, forKey: .identity)
        try values.encode(backendLayout, forKey: .backendLayout)
        try values.encode(position, forKey: .position)
        try values.encode(chunkSize, forKey: .chunkSize)
        if backendLayout == Self.diffusionBlockLayout {
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
        try values.encode(tensors, forKey: .tensors)
        try values.encodeIfPresent(attentionLayers, forKey: .attentionLayers)
        try values.encodeIfPresent(tokenByteTopologies, forKey: .tokenByteTopologies)
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.schemaVersion == rhs.schemaVersion && lhs.identity == rhs.identity
            && lhs.backendLayout == rhs.backendLayout && lhs.position == rhs.position
            && lhs.chunkSize == rhs.chunkSize && lhs.prefixTokens == rhs.prefixTokens
            && lhs.cacheSalt == rhs.cacheSalt && lhs.assistantCodecID == rhs.assistantCodecID
            && lhs.mediaIdentity == rhs.mediaIdentity && lhs.mediaTargetOnly == rhs.mediaTargetOnly
            && lhs.nativeBlockState == rhs.nativeBlockState
            && lhs.tensors == rhs.tensors && lhs.attentionLayers == rhs.attentionLayers
            && lhs.tokenByteTopologies == rhs.tokenByteTopologies
    }
}
