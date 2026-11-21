import Foundation

public enum Inkling2DSparkError: Error, Equatable, CustomStringConvertible {
    case missingConfiguration(String)
    case unsupported(field: String, reason: String)
    case weights(String)
    case target(String)

    public var description: String {
        switch self {
        case .missingConfiguration(let path): "Inkling2 DSpark: no mtp_config in \(path)"
        case .unsupported(let field, let reason): "Inkling2 DSpark: \(field): \(reason)"
        case .weights(let reason): "Inkling2 DSpark weights: \(reason)"
        case .target(let reason): "Inkling2 DSpark target: \(reason)"
        }
    }
}

/// The checkpoint's `mtp_config` plus the target fields the drafter shares
/// (vocabulary, logit softcap, layer-visit count).
public struct Inkling2DSparkConfiguration: Codable, Sendable, Equatable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var localLayerIds: [Int]
    public var slidingWindowSize: Int
    public var blockSize: Int
    public var markovRank: Int
    public var markovHeadType: String
    public var targetLayerIds: [Int]
    public var qBias: Bool
    public var oBias: Bool

    /// Unpadded vocabulary: drafts are argmaxed over `0..<vocabularySize`.
    public var vocabularySize: Int
    /// Rows of the shared embedding, unembedding and `markov_w1`.
    public var paddedVocabularySize: Int
    public var finalLogitSoftcapping: Float
    /// The target's layer visits (48 for the three-pass Mini).
    public var targetLayerVisits: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case localLayerIds = "local_layer_ids"
        case slidingWindowSize = "sliding_window_size"
        case blockSize = "block_size"
        case markovRank = "markov_rank"
        case markovHeadType = "markov_head_type"
        case targetLayerIds = "target_layer_ids"
        case qBias = "q_bias"
        case oBias = "o_bias"
        case vocabularySize = "vocab_size"
        case paddedVocabularySize = "padded_vocab_size"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case targetLayerVisits = "target_layer_visits"
    }

    public init(
        hiddenSize: Int, numHiddenLayers: Int, numAttentionHeads: Int, numKeyValueHeads: Int,
        headDim: Int, intermediateSize: Int, rmsNormEps: Float = 1e-6, ropeTheta: Float = 1e6,
        localLayerIds: [Int], slidingWindowSize: Int, blockSize: Int, markovRank: Int,
        markovHeadType: String = "rnn_hidden", targetLayerIds: [Int], qBias: Bool = false,
        oBias: Bool = false, vocabularySize: Int, paddedVocabularySize: Int,
        finalLogitSoftcapping: Float, targetLayerVisits: Int
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.localLayerIds = localLayerIds
        self.slidingWindowSize = slidingWindowSize
        self.blockSize = blockSize
        self.markovRank = markovRank
        self.markovHeadType = markovHeadType
        self.targetLayerIds = targetLayerIds
        self.qBias = qBias
        self.oBias = oBias
        self.vocabularySize = vocabularySize
        self.paddedVocabularySize = paddedVocabularySize
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.targetLayerVisits = targetLayerVisits
    }

    /// Number of captured target features per committed token.
    public var featureCount: Int { targetLayerIds.count }
    /// Width of one committed token's concatenated target features.
    public var featureWidth: Int { featureCount * hiddenSize }
    /// Draft tokens one block proposes (the anchor row predicts nothing).
    public var draftTokens: Int { blockSize - 1 }

    public func isLocal(layer: Int) -> Bool { localLayerIds.contains(layer) }

    /// Rejects anything the drafter port does not implement.
    public func validate() throws {
        func require(_ ok: Bool, _ field: String, _ reason: String) throws {
            if !ok { throw Inkling2DSparkError.unsupported(field: field, reason: reason) }
        }
        try require(markovHeadType == "rnn_hidden", "markov_head_type", "only rnn_hidden")
        try require(!qBias && !oBias, "q_bias/o_bias", "attention biases are not supported")
        try require(blockSize >= 2, "block_size", "needs an anchor and at least one mask")
        try require(
            targetLayerIds.count >= 2 && targetLayerIds == Array(Set(targetLayerIds)).sorted()
                && targetLayerIds.last == targetLayerVisits,
            "target_layer_ids", "sorted unique visits ending at the final-norm visit")
        try require(
            numAttentionHeads % numKeyValueHeads == 0 && headDim % 2 == 0, "heads",
            "invalid attention geometry")
        try require(
            localLayerIds.allSatisfy { (0 ..< numHiddenLayers).contains($0) }, "local_layer_ids",
            "out of range")
        try require(
            paddedVocabularySize >= vocabularySize && vocabularySize > 0, "vocab_size",
            "invalid vocabulary")
        try require(slidingWindowSize > 0 && markovRank > 0, "geometry", "invalid sizes")
    }

    /// Read `mtp_config` and the shared target fields from an Inkling 2
    /// `config.json`.
    public static func read(configURL: URL) throws -> Inkling2DSparkConfiguration {
        let data = try Data(contentsOf: configURL)
        return try parse(rootConfig: data, path: configURL.path)
    }

    public static func parse(rootConfig data: Data, path: String = "config.json") throws
        -> Inkling2DSparkConfiguration
    {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let mtp = root["mtp_config"] as? [String: Any],
            let text = root["text_config"] as? [String: Any]
        else { throw Inkling2DSparkError.missingConfiguration(path) }

        func int(_ dict: [String: Any], _ key: String) throws -> Int {
            guard let value = dict[key] as? NSNumber else {
                throw Inkling2DSparkError.unsupported(field: key, reason: "missing integer")
            }
            return value.intValue
        }
        func float(_ dict: [String: Any], _ key: String, _ fallback: Float) -> Float {
            (dict[key] as? NSNumber)?.floatValue ?? fallback
        }
        func ints(_ dict: [String: Any], _ key: String) throws -> [Int] {
            guard let values = dict[key] as? [NSNumber] else {
                throw Inkling2DSparkError.unsupported(field: key, reason: "missing list")
            }
            return values.map(\.intValue)
        }

        let textLayers = try int(text, "num_hidden_layers")
        var visits = textLayers
        if let loop = text["loop_config"] as? [String: Any] {
            let start = try int(loop, "core_start")
            let end = try int(loop, "core_end")
            let passes = try int(loop, "num_passes")
            visits += (end - start) * (passes - 1)
        }
        let padded = try int(text, "vocab_size")
        let configuration = Inkling2DSparkConfiguration(
            hiddenSize: try int(mtp, "hidden_size"),
            numHiddenLayers: try int(mtp, "num_hidden_layers"),
            numAttentionHeads: try int(mtp, "num_attention_heads"),
            numKeyValueHeads: try int(mtp, "num_key_value_heads"),
            headDim: try int(mtp, "head_dim"),
            intermediateSize: try int(mtp, "intermediate_size"),
            rmsNormEps: float(mtp, "rms_norm_eps", 1e-6),
            ropeTheta: float(mtp, "rope_theta", 1e6),
            localLayerIds: try ints(mtp, "local_layer_ids"),
            slidingWindowSize: try int(mtp, "sliding_window_size"),
            blockSize: try int(mtp, "block_size"),
            markovRank: try int(mtp, "markov_rank"),
            markovHeadType: mtp["markov_head_type"] as? String ?? "",
            targetLayerIds: try ints(mtp, "target_layer_ids"),
            qBias: mtp["q_bias"] as? Bool ?? false,
            oBias: mtp["o_bias"] as? Bool ?? false,
            vocabularySize: (text["unpadded_vocab_size"] as? NSNumber)?.intValue ?? padded,
            paddedVocabularySize: padded,
            finalLogitSoftcapping: float(text, "final_logit_softcapping", 0),
            targetLayerVisits: visits)
        guard configuration.hiddenSize == (try int(text, "hidden_size")) else {
            throw Inkling2DSparkError.unsupported(
                field: "hidden_size", reason: "drafter and target widths differ")
        }
        try configuration.validate()
        return configuration
    }
}
