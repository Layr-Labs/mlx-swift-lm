public enum Qwen38Quantization: Equatable, Hashable, Sendable {
    case affine(bits: Int, groupSize: Int)
}

public struct Qwen38TargetContract: Equatable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let linearKeyHeads: Int
    public let linearValueHeads: Int
    public let linearKeyHeadDimension: Int
    public let linearValueHeadDimension: Int
    public let fullAttentionInterval: Int
    public let vocabularySize: Int
    public let activationDType: String
    public let defaultQuantization: Qwen38Quantization
    public let requiredQuantizationIslands: Set<Qwen38Quantization>

    public static let production = Qwen38TargetContract(
        modelType: "qwen3_5_text",
        hiddenSize: 5_120,
        hiddenLayers: 64,
        intermediateSize: 17_408,
        attentionHeads: 24,
        keyValueHeads: 4,
        headDimension: 256,
        linearKeyHeads: 16,
        linearValueHeads: 48,
        linearKeyHeadDimension: 128,
        linearValueHeadDimension: 128,
        fullAttentionInterval: 4,
        vocabularySize: 248_320,
        activationDType: "bfloat16",
        defaultQuantization: .affine(bits: 4, groupSize: 32),
        requiredQuantizationIslands: [.affine(bits: 8, groupSize: 64)])
}

public struct Qwen38DraftContract: Equatable, Sendable {
    public let blockSize: Int
    public let targetLayerIDs: [Int]
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let vocabularySize: Int
    public let maskTokenID: Int
    public let quantization: Qwen38Quantization

    public static let production = Qwen38DraftContract(
        blockSize: 8,
        targetLayerIDs: [5, 19, 33, 47, 61],
        hiddenLayers: 5,
        hiddenSize: 5_120,
        intermediateSize: 17_408,
        attentionHeads: 32,
        keyValueHeads: 8,
        headDimension: 128,
        vocabularySize: 248_320,
        maskTokenID: 248_070,
        quantization: .affine(bits: 4, groupSize: 64))
}
