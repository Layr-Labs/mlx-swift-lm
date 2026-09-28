import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Native processor, separate from Gemma4's fixed image-token expansion. It
/// preserves typed message/tool history and explicit image/text ordering.
public struct DiffusionGemmaProcessor: Sendable {
    let configuration: DiffusionGemmaProcessorConfiguration
    let model: DiffusionGemmaConfiguration
    let tokenizer: any Tokenizer
    let template: String

    public init(configuration: DiffusionGemmaProcessorConfiguration,
        model: DiffusionGemmaConfiguration, tokenizer: any Tokenizer, template: String) throws
    {
        try configuration.validate(model: model)
        self.configuration = configuration
        self.model = model
        self.tokenizer = tokenizer
        self.template = template
    }

    public func prepare(input: UserInput,
        afterEvaluation: () throws -> Void = {}) async throws -> DiffusionGemmaMediaInput
    {
        try Task.checkCancellation()
        guard input.processing.resize == nil, input.processing.minPixels == nil, input.processing.maxPixels == nil else {
            throw DiffusionGemmaModelError.invalidInput("native media uses its declared pixel budget")
        }
        let messages = Gemma4MessageGenerator().generate(from: input)
        let rendered = try tokenizer.applyChatTemplate(messages: messages, chatTemplate: template,
            tools: input.tools, additionalContext: input.additionalContext)
        let imageID = try checkedToken("<|image|>", expected: model.imageTokenId)
        let videoID = try checkedToken("<|video|>", expected: model.videoTokenId)
        guard rendered.filter({ $0 == imageID }).count == input.images.count,
            rendered.filter({ $0 == videoID }).count == input.videos.count
        else { throw DiffusionGemmaModelError.invalidInput("unbound media placeholders") }
        var tokens = [Int32]()
        var frames = [DiffusionGemmaMediaInput.Frame]()
        var imageIndex = 0, videoIndex = 0
        func append(_ values: [Int]) throws {
            guard values.count <= model.textConfig.maxPositionEmbeddings - tokens.count,
                values.allSatisfy({ $0 >= 0 && $0 < model.textConfig.vocabularySize })
            else { throw DiffusionGemmaModelError.invalidInput("expanded media prompt capacity") }
            tokens.append(contentsOf: values.map(Int32.init))
        }
        func prepareImage(_ image: CIImage, budget: Int) throws -> (pixels: MLXArray, geometry: DiffusionGemmaMediaGeometry) {
            let value = try pixels(image, budget: budget)
            eval(value.pixels)
            try afterEvaluation()
            try Task.checkCancellation()
            return value
        }
        func appendImage(_ prepared: (pixels: MLXArray, geometry: DiffusionGemmaMediaGeometry),
            kind: DiffusionGemmaMediaInput.Frame.Kind, timestamp: Double?) throws
        {
            try append([model.beginImageTokenId])
            let span = CBv2ImageSpan(tokenOffset: tokens.count, length: prepared.geometry.softTokens)
            try append(Array(repeating: imageID, count: span.length))
            try append([model.endImageTokenId])
            frames.append(.init(kind: kind, pixels: prepared.pixels, span: span, timestampSeconds: timestamp))
        }
        for token in rendered {
            try Task.checkCancellation()
            if token == imageID {
                try appendImage(prepareImage(input.images[imageIndex].asCIImage(), budget: configuration.imageSoftTokenBudget),
                    kind: .image, timestamp: nil)
                imageIndex += 1
            } else if token == videoID {
                // The released native model consumes videos as timestamped image
                // frames, not a separate Gemma4 video tower/patch payload.
                let selected = try await DiffusionGemmaVideoFrames.sample(input.videos[videoIndex]) { frame in
                    // No separate video budget is declared by this artifact.
                    // Preserve its image budget for the documented frame route.
                    (image: try prepareImage(frame.frame, budget: configuration.imageSoftTokenBudget), seconds: frame.timeStamp.seconds)
                }
                for (index, frame) in selected.enumerated() {
                    let second = Int(frame.seconds.rounded(.towardZero))
                    let timestamp = String(format: "%02d:%02d ", second / 60, second % 60)
                    try append(tokenizer.encode(text: (index == 0 ? "" : " ") + timestamp, addSpecialTokens: false))
                    try appendImage(frame.image, kind: .videoFrame, timestamp: frame.seconds)
                }
                videoIndex += 1
            } else { try append([token]) }
        }
        guard !tokens.isEmpty else { throw DiffusionGemmaModelError.invalidInput("empty native prompt") }
        return DiffusionGemmaMediaInput(tokens: tokens, frames: frames)
    }

    private func checkedToken(_ name: String, expected: Int?) throws -> Int {
        guard let value = tokenizer.convertTokenToId(name), value >= 0,
            value < model.textConfig.vocabularySize, expected == nil || expected == value
        else { throw DiffusionGemmaModelError.invalidInput("native media vocabulary mismatch") }
        return value
    }

    private func pixels(_ original: CIImage, budget: Int) throws
        -> (pixels: MLXArray, geometry: DiffusionGemmaMediaGeometry)
    {
        let extent = original.extent
        guard extent.origin.x.isFinite, extent.origin.y.isFinite,
            let width = Int(exactly: extent.width), let height = Int(exactly: extent.height)
        else { throw DiffusionGemmaModelError.invalidInput("nonfinite/nonintegral media extent") }
        let geometry = try DiffusionGemmaMediaGeometry.resized(width: width, height: height,
            patchSize: configuration.patchSize, poolingSize: configuration.poolingSize, maxSoftTokens: budget)
        let image = original.transformed(by: .init(translationX: -extent.minX, y: -extent.minY))
        return (try DiffusionGemmaImagePixels.prepare(image, width: width, height: height,
            geometry: geometry), geometry)
    }
}
