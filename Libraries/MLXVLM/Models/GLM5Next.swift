// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Explicit native integration, intentionally absent from model factories and
/// serving capability registries until independent qualification is complete.
public final class GLM5NextModel: Module, LanguageModel {
    public let configuration: GLM5NextConfiguration
    public let textTower: GLM5NextTextModel
    public let visionTower: GLM5NextVisionTower
    public private(set) var mtp: GLM5NextMTP?
    /// Retains identity and the complete inventory, including an unloaded MTP
    /// partition. It does not retain a duplicate dictionary of source weights.
    public let artifact: GLM5NextArtifactLoader
    public var vocabularySize: Int { configuration.textConfig.vocabularySize }

    public init(artifact: GLM5NextArtifactLoader, loadMTP: Bool = false) throws {
        self.artifact = artifact
        configuration = artifact.configuration
        textTower = try artifact.loadTextModel()
        visionTower = try GLM5NextVisionTower(configuration: configuration.visionConfig)
        mtp = nil
        super.init()
        artifact.constructionRoots?.retainOwner(visionTower)
        try MLX.withError { error in
            var weights: [String: MLXArray] = [:]
            for entry in artifact.inventory.entries(in: .vision) {
                weights[entry.tensor.name] = try artifact.tensor(entry.tensor.name)
            }
            try visionTower.loadNativeWeights(
                weights, layout: .pytorch,
                dtype: artifact.inventory.storage == .float32 ? .float32 : .bfloat16)
            // Layout transposes must complete before their source arrays retire.
            eval(visionTower.parameters().flattened().map(\.1))
            try error.check()
        }
        try artifact.requireConsumed(.vision)
        if loadMTP { try loadMTPPartition() }
        try artifact.verifyUnchanged()
    }

    /// Load-time operation, serialized by the model owner, not a request cache.
    /// A deferred MTP source is revalidated instead of silently discarded.
    public func loadMTPPartition() throws {
        guard mtp == nil else { throw GLM5NextArtifactError.invalid("MTP already loaded") }
        guard configuration.textConfig.numNextNPredictLayers == 1 else {
            throw GLM5NextArtifactError.invalid("artifact has no trained MTP partition")
        }
        mtp = try artifact.loadMTP(target: textTower)
    }

    public func newTextCache() -> GLM5NextTextCache { textTower.newTextCache() }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        precondition(
            parameters?.maxKVSize == nil,
            "GLM5Next recurrent/sparse state cannot use a sliding-window KV cache")
        return textTower.newCache(parameters: parameters)
    }

    /// Preferred multimodal seam: metadata travels with its input, not through
    /// model-global mutable state. The text candidate commits only after every
    /// output and cache root has been evaluated and checked successfully.
    public func forward(
        _ prepared: GLM5NextPreparedInput, cache: GLM5NextTextCache? = nil,
        limits: GLM5NextVisionLimits = .init(), checkpoint: (([MLXArray]) throws -> Void)? = nil
    ) throws -> GLM5NextTextOutput {
        try forward(
            prepared.input, placements: prepared.placements, cache: cache,
            limits: limits, checkpoint: checkpoint)
    }

    public func forward(
        _ input: LMInput, placements: [GLM5NextMediaPlacement]? = nil,
        cache: GLM5NextTextCache? = nil, limits: GLM5NextVisionLimits = .init(),
        checkpoint: (([MLXArray]) throws -> Void)? = nil
    ) throws -> GLM5NextTextOutput {
        try MLX.withError { error in
            func fence(_ arrays: [MLXArray]) throws {
                try error.check()
                try checkpoint?(arrays)
                try error.check()
                eval(arrays)
                try error.check()
            }
            let text = input.text
            guard text.tokens.ndim == 2, text.tokens.dim(0) > 0, text.tokens.dim(1) > 0,
                [.int32, .uint32, .int64].contains(text.tokens.dtype),
                text.tokens.dim(1) <= configuration.textConfig.maxPositionEmbeddings
            else {
                throw GLM5NextVisionError.invalidInput(
                    "integer [B,T] tokens within context limit required")
            }
            if let mask = text.mask {
                guard mask.shape == text.tokens.shape, mask.dtype == .bool else {
                    throw GLM5NextVisionError.invalidInput("boolean [B,T] validity required")
                }
            }
            let tokenArray = text.tokens.asType(.int64)
            try fence([tokenArray] + [text.mask].compactMap { $0 })
            let tokens = tokenArray.asArray(Int64.self)
            try error.check()
            guard tokens.allSatisfy({ $0 >= 0 && $0 < Int64(vocabularySize) }) else {
                throw GLM5NextVisionError.invalidInput("token outside vocabulary")
            }
            let hasMedia = input.image != nil || input.video != nil
            let destination = cache ?? newTextCache()
            if !hasMedia {
                guard placements?.isEmpty != false,
                    !tokens.contains(Int64(configuration.imageTokenId)),
                    !tokens.contains(Int64(configuration.videoTokenId))
                else {
                    throw GLM5NextVisionError.invalidInput("media tokens without media")
                }
                let candidate = try textTower.prepareForward(
                    tokens: text.tokens,
                    cache: destination, validity: text.mask)
                return try candidate.commit(afterEvaluating: fence)
            }
            // Batch metadata is row-local in the processor. This wrapper does
            // not guess ownership from concatenated multirow media tensors.
            guard text.tokens.dim(0) == 1, destination.offset == 0 else {
                throw GLM5NextVisionError.unsupported(
                    "media requires one row and a fresh request cache")
            }
            let images = try media(
                input.image?.pixels, frames: input.image?.frames, image: true, limits: limits)
            let videos = try media(
                input.video?.pixels, frames: input.video?.frames, image: false, limits: limits)
            let totalPatches = (images?.pixels.dim(0) ?? 0) + (videos?.pixels.dim(0) ?? 0)
            guard totalPatches <= limits.maximumPatches else {
                throw GLM5NextVisionError.budgetExceeded("aggregate image/video patches")
            }
            let planned: [GLM5NextMediaPlacement]
            if let placements {
                planned = placements
            } else {
                planned = try derivePlacements(tokens: tokens, images: images, videos: videos)
            }
            let validity = text.mask?.asArray(Bool.self)
            try error.check()
            try validatePlacements(
                planned, tokens: tokens, validity: validity, images: images, videos: videos)

            var embeddings = textTower.embed(text.tokens)
            // A modality is processed one image/temporal pair at a time. The
            // tower itself bounds per-frame attention queries and checks eval.
            for (modality, source) in [
                (GLM5NextMediaPlacement.Modality.image, images), (.video, videos),
            ] {
                guard let source else { continue }
                var patchOffset = 0
                var featureOffset = 0
                for (item, grid) in source.grids.enumerated() {
                    let patchesPerFrame = try GLM5NextVisionGrid(
                        temporal: 1, height: grid.height, width: grid.width
                    ).patchCount()
                    let featuresPerFrame = patchesPerFrame / 4
                    for _ in 0 ..< grid.temporal {
                        let frameGrid = try GLM5NextVisionGrid(
                            temporal: 1, height: grid.height, width: grid.width)
                        let features = try visionTower.forward(
                            patches: source.pixels[patchOffset ..< (patchOffset + patchesPerFrame)],
                            grids: [frameGrid], limits: limits, checkpoint: fence)
                        let end = featureOffset + featuresPerFrame
                        let matching = planned.filter {
                            $0.modality == modality && $0.itemIndex == item
                                && $0.featureRange.lowerBound == featureOffset
                                && $0.featureRange.upperBound == end
                        }
                        // validatePlacements establishes exactly one matching
                        // frame (or still-image) placement before tower execution.
                        guard let placement = matching.first, matching.count == 1 else {
                            throw GLM5NextVisionError.invalidInput("frame placement")
                        }
                        embeddings[0, placement.tokenRange, 0...] = features.asType(
                            embeddings.dtype)
                        try fence([embeddings])
                        patchOffset += patchesPerFrame
                        featureOffset = end
                    }
                }
            }
            let candidate = try textTower.prepareForward(
                embeddings: embeddings,
                cache: destination, validity: text.mask)
            return try candidate.commit(afterEvaluating: fence)
        }
    }

    /// LMInput-only callers can reconstruct a single modality deterministically.
    /// Mixed image/video prompts must retain prepareDetailed's pure metadata;
    /// identical image-token runs otherwise cannot prove their modality.
    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard windowSize == nil || windowSize! > 0 else {
            throw GLM5NextVisionError.invalidInput("nonpositive prefill window")
        }
        // windowSize is a chunking hint, not permission to truncate GLM state.
        return .logits(
            LMOutput(logits: try forward(input, cache: GLM5NextTextCache(layers: cache)).logits))
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        precondition(state == nil, "GLM5Next does not use cross-attention state")
        do {
            return LMOutput(
                logits: try forward(
                    LMInput(text: input),
                    cache: cache.map { GLM5NextTextCache(layers: $0) }
                ).logits)
        } catch { preconditionFailure("GLM5Next forward failed: \(error)") }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        callAsFunction(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    private struct Media {
        let pixels: MLXArray
        let grids: [GLM5NextVisionGrid]
    }

    private func media(
        _ pixels: MLXArray?, frames: [THW]?, image: Bool,
        limits: GLM5NextVisionLimits
    ) throws -> Media? {
        guard let pixels else { return nil }
        guard let frames, !frames.isEmpty, pixels.ndim == 2,
            pixels.dtype.isFloatingPoint,
            limits.maximumPatches > 0, limits.maximumFramePatches > 0
        else {
            throw GLM5NextVisionError.invalidInput(
                "flattened floating patches and nonempty THW required")
        }
        let grids = try frames.map {
            try GLM5NextVisionGrid(temporal: $0.t, height: $0.h, width: $0.w)
        }
        var count = 0
        for grid in grids {
            let patches = try grid.patchCount()
            _ = try grid.featureCount()
            guard !image || grid.temporal == 1 else {
                throw GLM5NextVisionError.unsupported("still image with multiple temporal pairs")
            }
            guard patches <= limits.maximumPatches - count,
                patches / grid.temporal <= limits.maximumFramePatches
            else {
                throw GLM5NextVisionError.budgetExceeded("aggregate modality patches")
            }
            count += patches
        }
        let c = configuration.visionConfig
        guard
            pixels.shape == [count, c.inChannels * c.temporalPatchSize * c.patchSize * c.patchSize]
        else {
            throw GLM5NextVisionError.invalidInput("patch tensor does not match THW")
        }
        return Media(pixels: pixels, grids: grids)
    }

    private func derivePlacements(tokens: [Int64], images: Media?, videos: Media?) throws
        -> [GLM5NextMediaPlacement]
    {
        guard images == nil || videos == nil else {
            throw GLM5NextVisionError.unsupported(
                "mixed image/video LMInput requires prepareDetailed placements")
        }
        let modality: GLM5NextMediaPlacement.Modality = images != nil ? .image : .video
        guard let source = images ?? videos else { return [] }
        var cursor = 0
        var feature = 0
        var result: [GLM5NextMediaPlacement] = []
        for (item, grid) in source.grids.enumerated() {
            let count = try grid.featureCount() / grid.temporal
            for _ in 0 ..< grid.temporal {
                while cursor < tokens.count && tokens[cursor] != Int64(configuration.imageTokenId) {
                    cursor += 1
                }
                guard count <= tokens.count - cursor else {
                    throw GLM5NextVisionError.invalidInput("missing expanded media tokens")
                }
                result.append(
                    .init(
                        modality: modality, itemIndex: item,
                        tokenRange: cursor ..< (cursor + count),
                        featureRange: feature ..< (feature + count)))
                cursor += count
                feature += count
            }
        }
        return result
    }

    private func validatePlacements(
        _ placements: [GLM5NextMediaPlacement], tokens: [Int64], validity: [Bool]?,
        images: Media?, videos: Media?
    ) throws {
        var occupied = Set<Int>()
        for placement in placements {
            guard placement.tokenRange.lowerBound >= 0,
                placement.tokenRange.upperBound <= tokens.count,
                !placement.tokenRange.isEmpty,
                placement.tokenRange.count == placement.featureRange.count,
                placement.tokenRange.allSatisfy({
                    tokens[$0] == Int64(configuration.imageTokenId)
                        && (validity?[$0] ?? true) && occupied.insert($0).inserted
                })
            else {
                throw GLM5NextVisionError.invalidInput("media token coverage or masked placement")
            }
        }
        guard !tokens.contains(Int64(configuration.videoTokenId)),
            tokens.enumerated().allSatisfy({
                $0.element != Int64(configuration.imageTokenId) || occupied.contains($0.offset)
            })
        else {
            throw GLM5NextVisionError.invalidInput("unexpanded or uncovered media tokens")
        }
        for (modality, source) in [
            (GLM5NextMediaPlacement.Modality.image, images), (.video, videos),
        ] {
            let selected = placements.filter { $0.modality == modality }
            var expected: [(Int, Range<Int>)] = []
            var feature = 0
            for (item, grid) in (source?.grids ?? []).enumerated() {
                let count = try grid.featureCount() / grid.temporal
                for _ in 0 ..< grid.temporal {
                    expected.append((item, feature ..< (feature + count)))
                    feature += count
                }
            }
            let ordered = selected.sorted {
                $0.featureRange.lowerBound < $1.featureRange.lowerBound
            }
            guard ordered.count == expected.count,
                zip(ordered, expected).allSatisfy({
                    $0.0.itemIndex == $0.1.0 && $0.0.featureRange == $0.1.1
                }),
                zip(ordered, ordered.dropFirst()).allSatisfy({
                    $0.tokenRange.upperBound <= $1.tokenRange.lowerBound
                })
            else {
                throw GLM5NextVisionError.invalidInput("modality/item/frame feature coverage")
            }
        }
    }
}
