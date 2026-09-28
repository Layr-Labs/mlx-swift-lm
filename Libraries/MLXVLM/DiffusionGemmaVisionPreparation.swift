import MLX
import MLXLLM
import MLXLMCommon

extension DiffusionGemma {
    /// Mirrors the native tower's padding to MLX's supported fused head sizes.
    /// Serving admission uses the real fallback head multiplicity otherwise.
    public var visionAttentionGeometry: (patchSize: Int, attentionHeadFactor: Int)? {
        guard let vision = configuration.visionConfig else { return nil }
        let padded = [64, 80, 128].first { vision.headDim <= $0 } ?? vision.headDim
        return (vision.patchSize, [64, 80, 128].contains(padded) ? 1 : vision.numAttentionHeads)
    }
    /// Evaluate the actual native tower/projector one frame at a time. The
    /// callback lets serving code check its scoped MLX error box after each
    /// evaluation, before beginning another frame.
    public func prepareVision(_ input: DiffusionGemmaMediaInput,
        afterEvaluation: () throws -> Void = {}) throws -> CBv2MultimodalInput?
    {
        guard !input.frames.isEmpty else { return nil }
        guard let vision = configuration.visionConfig, let tower = model.encoder.visionTower,
            let projection = model.encoder.embedVision
        else { throw DiffusionGemmaModelError.invalidInput("missing native vision modules") }
        let spans = input.frames.map(\.span)
        try DiffusionGemmaVisualEmbeddings.validate(spans: spans, promptCount: input.tokens.count)
        let dtype = try model.decoder.scaledEmbeddings(MLXArray([Int32(configuration.textConfig.padTokenId ?? 0)]).reshaped(1, 1)).dtype
        var features = [MLXArray]()
        for frame in input.frames {
            try Task.checkCancellation()
            let pixels = frame.pixels
            let side = vision.patchSize * vision.poolingKernelSize
            guard pixels.ndim == 4, pixels.dim(0) == 1, pixels.dim(1) == 3,
                pixels.dtype.isFloatingPoint, pixels.dim(2) > 0, pixels.dim(3) > 0,
                pixels.dim(2).isMultiple(of: side), pixels.dim(3).isMultiple(of: side),
                (pixels.dim(2) / side) * (pixels.dim(3) / side) == frame.span.length,
                input.tokens[frame.span.tokenOffset..<(frame.span.tokenOffset + frame.span.length)]
                    .allSatisfy({ $0 == Int32(configuration.imageTokenId) })
            else { throw DiffusionGemmaModelError.invalidInput("native frame/prompt geometry mismatch") }
            let value = projection(tower(pixels, outputLength: frame.span.length)).asType(dtype)
            eval(value)
            try afterEvaluation()
            guard value.shape == [1, frame.span.length, configuration.textConfig.hiddenSize] else {
                throw DiffusionGemmaModelError.invalidInput("native visual feature shape")
            }
            features.append(value)
        }
        return CBv2MultimodalInput(spans: spans) { features }
    }
}
