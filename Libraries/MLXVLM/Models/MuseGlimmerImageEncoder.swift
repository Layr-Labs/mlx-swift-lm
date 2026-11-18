// Copyright © 2026 Eigen Labs.

import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Refusals of the Muse Glimmer image path. Each is a property of the input
/// or the artifact, never of capacity.
public enum MuseGlimmerImageError: Error, Equatable {
    /// Not the pinned artifact, or its `config.json` disagrees with the pinned stack.
    case identity(String)
    /// A tensor the vision stack needs is absent.
    case missing(String)
    /// A tensor is present that the vision stack does not have.
    case unexpected(String)
    /// A tensor has the wrong shape or byte count.
    case layout(String)
    /// A tensor is not BF16.
    case dtype(String)
    /// A size, token budget, pixel array or grid that is not an image of this model.
    case geometry(String)
    /// The prompt's image placeholders and the images disagree.
    case placeholders(String)
    /// Media this build does not take yet.
    case unsupportedMedia(String)
}

/// Patch grids and token counts, from upstream's processor arithmetic. CPU only.
public enum MuseGlimmerImageGeometry {
    public static let patchSize = 14
    public static let mergeSize = 2
    public static let temporalPatchSize = 2
    /// `max_image_tokens` of `processor_config.json`.
    public static let maximumImageTokens = 4096
    /// `<|image_start|>` and `<|image_end|>` around each image's patch run.
    public static let framingTokens = 2
    /// The most prompt tokens one image adds.
    public static let maximumPromptTokensPerImage = maximumImageTokens + framingTokens
    /// Values of one patch row: temporal x channels x patch x patch.
    public static let patchValues = temporalPatchSize * 3 * patchSize * patchSize

    /// One image's grid, in patches of 14 pixels.
    public struct Grid: Sendable, Equatable {
        public let patchRows: Int
        public let patchColumns: Int

        /// A grid of `mergedRows x mergedColumns` merged tokens (2 x 2 patches each).
        public init(mergedRows: Int, mergedColumns: Int) throws {
            guard mergedRows >= 1, mergedColumns >= 1,
                mergedRows <= maximumImageTokens, mergedColumns <= maximumImageTokens,
                mergedRows * mergedColumns <= maximumImageTokens
            else {
                throw MuseGlimmerImageError.geometry(
                    "grid \(mergedRows)x\(mergedColumns) merged tokens")
            }
            patchRows = mergedRows * mergeSize
            patchColumns = mergedColumns * mergeSize
        }

        public var pixelHeight: Int { patchRows * patchSize }
        public var pixelWidth: Int { patchColumns * patchSize }
        public var thw: THW { THW(1, patchRows, patchColumns) }
        /// Rows the encoder returns, and `<|patch|>` tokens in the image's span.
        public var featureRows: Int { patchRows * patchColumns / (mergeSize * mergeSize) }
        /// Prompt tokens the image adds: its patches and the two framing tokens.
        public var promptTokens: Int { featureRows + framingTokens }
    }

    /// The grid upstream's `MuseGlimmerProcessor.preprocess` picks for a
    /// source of this size: its `smart_resize` (float32, sorted ties) with a
    /// unit of 28 pixels and at most `maximumImageTokens` merged tokens.
    public static func grid(
        sourceHeight: Int, sourceWidth: Int, maximumImageTokens: Int = maximumImageTokens
    ) throws -> Grid {
        guard sourceHeight > 0, sourceWidth > 0 else {
            throw MuseGlimmerImageError.geometry("source \(sourceWidth)x\(sourceHeight)")
        }
        guard (1 ... Self.maximumImageTokens).contains(maximumImageTokens) else {
            throw MuseGlimmerImageError.geometry("token budget \(maximumImageTokens)")
        }
        let unit = patchSize * mergeSize
        let (height, width) = museSmartResize(
            height: sourceHeight, width: sourceWidth, factor: unit, maxTokens: maximumImageTokens)
        return try Grid(mergedRows: height / unit, mergedColumns: width / unit)
    }
}

/// Expansion of the template's image placeholders. CPU only.
public enum MuseGlimmerImagePrompt {
    public static let imageTokenID = 200092
    public static let imageStartTokenID = 200080
    public static let imageEndTokenID = 200081
    public static let videoTokenID = 200091

    public struct Expanded: Sendable, Equatable {
        public let tokens: [Int]
        /// One span per image, in prompt order: its `<|patch|>` run, framing excluded.
        public let spans: [CBv2ImageSpan]
    }

    /// Replaces the i-th `<|patch|>` with `<|image_start|>`, `grids[i].featureRows`
    /// patches and `<|image_end|>`, as upstream's processor does.
    public static func expand(
        promptTokens: [Int], grids: [MuseGlimmerImageGeometry.Grid]
    ) throws -> Expanded {
        guard !promptTokens.contains(videoTokenID) else {
            throw MuseGlimmerImageError.unsupportedMedia("video placeholder")
        }
        let placeholders = promptTokens.lazy.filter { $0 == imageTokenID }.count
        guard placeholders == grids.count, !grids.isEmpty else {
            throw MuseGlimmerImageError.placeholders(
                "\(placeholders) image placeholders for \(grids.count) images")
        }
        var tokens = [Int]()
        tokens.reserveCapacity(
            promptTokens.count + grids.reduce(0) { $0 + $1.promptTokens - 1 })
        var spans = [CBv2ImageSpan]()
        var image = 0
        for token in promptTokens {
            guard token == imageTokenID else {
                tokens.append(token)
                continue
            }
            let rows = grids[image].featureRows
            tokens.append(imageStartTokenID)
            spans.append(CBv2ImageSpan(tokenOffset: tokens.count, length: rows))
            tokens.append(contentsOf: repeatElement(imageTokenID, count: rows))
            tokens.append(imageEndTokenID)
            image += 1
        }
        return Expanded(tokens: tokens, spans: spans)
    }
}

/// The Muse Glimmer vision stack, loaded from the installer's verified
/// tensors: tower, adapter, projection and `perception_emb_norm`, with the
/// arithmetic of upstream's classes (`MuseGlimmerVisionFeatures`).
///
/// Thread rules: `features` evaluates on the calling thread under
/// `MLX.withError` and serializes calls on one encoder. Call it on the
/// submit side, never on the engine step thread, and hand the evaluated
/// arrays to the request's `CBv2MultimodalInput`. Nothing else changes after
/// `load`, so one encoder may be shared between tasks.
public final class MuseGlimmerImageEncoder: @unchecked Sendable {
    /// The published configuration; the decoders' defaults are its values.
    public static let pinnedConfiguration: MuseGlimmerConfiguration = {
        try! JSONDecoder().decode(
            MuseGlimmerConfiguration.self,
            from: Data(#"{"text_config": {}, "vision_config": {}}"#.utf8))
    }()

    static let checkpointPrefix = "model."

    let stack: MuseGlimmerVisionFeatures
    public let configuration: MuseGlimmerConfiguration
    /// Text-width feature columns.
    public var featureWidth: Int { configuration.textConfiguration.hiddenSize }
    /// Bytes of the loaded parameters.
    public let parameterBytes: Int
    private let lock = NSLock()

    private init(stack: MuseGlimmerVisionFeatures, configuration: MuseGlimmerConfiguration) {
        self.stack = stack
        self.configuration = configuration
        self.parameterBytes = stack.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    }

    // MARK: Loading

    /// Production entry: the vision tensors of the pinned NVIDIA artifact,
    /// exactly the stack's parameters, each BF16 and of the module's shape.
    /// `checkpointConfiguration` (the artifact's `config.json` bytes), when
    /// given, must agree with the pinned stack; it never drives loading.
    public static func load(
        installed: MuseGlimmerInstallation.Installed, checkpointConfiguration: Data? = nil
    ) throws -> MuseGlimmerImageEncoder {
        guard installed.receipt.indexSHA256 == MuseGlimmerCheckpointPin.index,
            installed.receipt.shardSHA256 == MuseGlimmerCheckpointPin.shards
        else { throw MuseGlimmerImageError.identity("checkpoint pin") }
        if let checkpointConfiguration {
            try crossCheck(checkpointConfiguration: checkpointConfiguration)
        }
        return try load(configuration: pinnedConfiguration, sources: installed.visionSources)
    }

    /// Unpinned so tests can load tiny stacks; production goes through the
    /// pinned entry above.
    static func load(
        configuration: MuseGlimmerConfiguration, sources: [String: ModelOptTensorSource]
    ) throws -> MuseGlimmerImageEncoder {
        let stack = MuseGlimmerVisionFeatures(configuration)
        let expected = Dictionary(uniqueKeysWithValues: stack.parameters().flattened())
        var byModelKey = [String: ModelOptTensorSource]()
        for (key, source) in sources {
            guard key.hasPrefix(checkpointPrefix), source.name == key else {
                throw MuseGlimmerImageError.unexpected(key)
            }
            let modelKey = String(key.dropFirst(checkpointPrefix.count))
            guard let parameter = expected[modelKey] else {
                throw MuseGlimmerImageError.unexpected(key)
            }
            guard source.layout.dtype == "BF16" else { throw MuseGlimmerImageError.dtype(key) }
            guard source.layout.shape == parameter.shape,
                source.layout.byteCount == parameter.size * DType.bfloat16.size
            else { throw MuseGlimmerImageError.layout(key) }
            byModelKey[modelKey] = source
        }
        if let key = Set(expected.keys).subtracting(byModelKey.keys).sorted().first {
            throw MuseGlimmerImageError.missing(checkpointPrefix + key)
        }
        var mapped = [String: MLXArray]()
        for (key, source) in byModelKey.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            mapped[key] = try source.array(maximumBytes: source.layout.byteCount)
        }
        try stack.update(parameters: .unflattened(mapped), verify: [.all])
        try MLX.withError { errors in
            eval(stack)
            try errors.check()
        }
        return MuseGlimmerImageEncoder(stack: stack, configuration: configuration)
    }

    /// The fields of `config.json` the stack depends on must be the pinned ones.
    static func crossCheck(checkpointConfiguration data: Data) throws {
        let decoded: MuseGlimmerConfiguration
        do {
            decoded = try JSONDecoder().decode(MuseGlimmerConfiguration.self, from: data)
        } catch {
            throw MuseGlimmerImageError.identity("config.json")
        }
        let p = pinnedConfiguration
        let (v, w) = (decoded.visionConfiguration, p.visionConfiguration)
        let agrees =
            decoded.modelType == p.modelType && decoded.imageTokenId == p.imageTokenId
            && decoded.videoTokenId == p.videoTokenId && decoded.outHiddenSize == p.outHiddenSize
            && decoded.projectorHiddenSize == p.projectorHiddenSize
            && decoded.textConfiguration.hiddenSize == p.textConfiguration.hiddenSize
            && decoded.textConfiguration.rmsNormEps == p.textConfiguration.rmsNormEps
            && v.hiddenSize == w.hiddenSize && v.intermediateSize == w.intermediateSize
            && v.attentionHeads == w.attentionHeads && v.hiddenLayers == w.hiddenLayers
            && v.patchSize == w.patchSize && v.patchTemporal == w.patchTemporal
            && v.mergeSize == w.mergeSize
            && v.positionEmbeddingHeight == w.positionEmbeddingHeight
            && v.positionEmbeddingWidth == w.positionEmbeddingWidth
            && v.layerNormEps == w.layerNormEps && v.ropeTheta == w.ropeTheta
            && v.layerTypes == w.layerTypes
        guard agrees else { throw MuseGlimmerImageError.identity("config.json") }
        // The stack's GELUs are exact; another activation is another model.
        let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let vision = root?["vision_config"] as? [String: Any]
        for activation in [root?["projector_hidden_act"], vision?["hidden_act"]] {
            guard activation == nil || activation as? String == "gelu" else {
                throw MuseGlimmerImageError.identity("config.json activation")
            }
        }
    }

    // MARK: Encoding

    /// Upstream's preprocessing of one image (`MuseGlimmerProcessor.preprocess`
    /// without per-call processing): sRGB tone curve, LANCZOS to the
    /// `smart_resize` grid, mean and std 0.5, patchify. Returns evaluated
    /// float32 pixels `[patchRows * patchColumns, 1176]` and the grid.
    public static func preprocess(
        image: CIImage,
        maximumImageTokens: Int = MuseGlimmerImageGeometry.maximumImageTokens
    ) throws -> (pixels: MLXArray, grid: MuseGlimmerImageGeometry.Grid) {
        let grid = try MuseGlimmerImageGeometry.grid(
            sourceHeight: Int(image.extent.height.rounded()),
            sourceWidth: Int(image.extent.width.rounded()),
            maximumImageTokens: maximumImageTokens)
        let targetSize = CGSize(width: grid.pixelWidth, height: grid.pixelHeight)
        var processing = UserInput.Processing()
        processing.resize = targetSize
        let processed = MediaProcessing.apply(image, processing: processing)
        let srgb = MediaProcessing.inSRGBToneCurveSpace(processed)
        let resized = MediaProcessing.resampleLanczos(srgb, to: targetSize)
        let normalized = MediaProcessing.normalize(
            resized, mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))
        let pixels = museGlimmerPatchify(
            MediaProcessing.asMLXArray(normalized), gridH: grid.patchRows,
            gridW: grid.patchColumns, patchSize: MuseGlimmerImageGeometry.patchSize,
            temporalPatchSize: MuseGlimmerImageGeometry.temporalPatchSize)
        try MLX.withError { errors in
            eval(pixels)
            try errors.check()
        }
        return (pixels, grid)
    }

    /// The evaluated `[grid.featureRows, featureWidth]` rows to splice at the
    /// image's span, after `perception_emb_norm`, in the stack's dtype.
    public func features(pixels: MLXArray, grid: MuseGlimmerImageGeometry.Grid) throws -> MLXArray {
        try Task.checkCancellation()
        guard pixels.ndim == 2, pixels.dim(0) == grid.patchRows * grid.patchColumns,
            pixels.dim(1) == MuseGlimmerImageGeometry.patchValues,
            [.float32, .bfloat16, .float16].contains(pixels.dtype)
        else {
            throw MuseGlimmerImageError.geometry("pixels \(pixels.shape) \(pixels.dtype)")
        }
        lock.lock()
        defer { lock.unlock() }
        var result: MLXArray?
        try MLX.withError { errors in
            let features = stack(pixels, grid: [grid.thw])
            eval(features)
            try errors.check()
            result = features
        }
        guard let result, result.shape == [grid.featureRows, featureWidth] else {
            throw MuseGlimmerImageError.geometry("features \(result?.shape ?? [])")
        }
        return result
    }
}
