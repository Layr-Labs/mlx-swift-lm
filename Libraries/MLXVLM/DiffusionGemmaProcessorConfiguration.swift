import Foundation
import MLXLLM

/// Decode-only native image contract. Unsupported transforms reject instead of
/// being silently ignored. Video-as-images uses the same trained pixel path.
public struct DiffusionGemmaProcessorConfiguration: Decodable, Sendable {
    public let patchSize: Int
    public let poolingSize: Int
    public let imageSoftTokenBudget: Int

    private enum Keys: String, CodingKey {
        case processorClass = "processor_class", image = "image_processor"
    }
    private enum ImageKeys: String, CodingKey {
        case type = "image_processor_type", patch = "patch_size", pool = "pooling_kernel_size"
        case budget = "max_soft_tokens", resize = "do_resize", rescale = "do_rescale"
        case normalize = "do_normalize", rgb = "do_convert_rgb", factor = "rescale_factor", resample
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: Keys.self)
        guard try root.decode(String.self, forKey: .processorClass) == "DiffusionGemma4Processor" else {
            throw DiffusionGemmaModelError.invalidInput("native processor class")
        }
        let image = try root.nestedContainer(keyedBy: ImageKeys.self, forKey: .image)
        patchSize = try image.decode(Int.self, forKey: .patch)
        poolingSize = try image.decode(Int.self, forKey: .pool)
        imageSoftTokenBudget = try image.decode(Int.self, forKey: .budget)
        guard try image.decode(String.self, forKey: .type) == "Gemma4ImageProcessor",
            try image.decodeIfPresent(Bool.self, forKey: .resize) ?? true,
            try image.decodeIfPresent(Bool.self, forKey: .rescale) ?? true,
            try !(image.decodeIfPresent(Bool.self, forKey: .normalize) ?? false),
            try image.decodeIfPresent(Bool.self, forKey: .rgb) ?? true,
            try image.decodeIfPresent(Double.self, forKey: .factor) ?? (1 / 255) == 1 / 255,
            try image.decodeIfPresent(Int.self, forKey: .resample) ?? 3 == 3
        else { throw DiffusionGemmaModelError.invalidInput("unsupported native image transform") }
        _ = try DiffusionGemmaMediaGeometry.resized(width: 1, height: 1,
            patchSize: patchSize, poolingSize: poolingSize, maxSoftTokens: imageSoftTokenBudget)
    }

    public func validate(model: DiffusionGemmaConfiguration) throws {
        guard let vision = model.visionConfig,
            patchSize == vision.patchSize, poolingSize == vision.poolingKernelSize,
            imageSoftTokenBudget == model.visionSoftTokensPerImage
        else { throw DiffusionGemmaModelError.invalidInput("processor/model vision geometry mismatch") }
    }
}
