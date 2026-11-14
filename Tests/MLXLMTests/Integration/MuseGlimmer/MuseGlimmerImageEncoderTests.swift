// Copyright © 2026 Eigen Labs.

import CoreImage
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Returns a fixed rendered prompt, so the upstream processor's expansion can
/// be compared with ``MuseGlimmerImagePrompt``.
private struct MuseGlimmerFixedPromptTokenizer: Tokenizer {
    let prompt: [Int]
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { prompt }
}

/// The image entry point against upstream's MLXVLM classes, on tiny stacks
/// with host-generated weights, and the pinned stack's shape. No checkpoint
/// or downloaded data.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerImageEncoderTests {
    typealias Geometry = MuseGlimmerImageGeometry
    typealias Prompt = MuseGlimmerImagePrompt
    typealias Failure = MuseGlimmerImageError

    static let visionPrefixes = ["vision_tower.", "vision_adapter.", "vision_projection."]

    /// The upstream fixture's vision weights, by module path.
    static func visionWeights(_ upstream: MuseGlimmer) -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: upstream.parameters().flattened().filter { key, _ in
                visionPrefixes.contains { key.hasPrefix($0) }
            })
    }

    static func stack(_ upstream: MuseGlimmer) throws -> MuseGlimmerVisionFeatures {
        let stack = MuseGlimmerVisionFeatures(upstream.config)
        try stack.update(parameters: .unflattened(visionWeights(upstream)), verify: [.all])
        eval(stack)
        return stack
    }

    /// Normalized pixels for `grid`, in [-1, 1).
    static func pixels(_ grid: Geometry.Grid, seed: UInt64) -> MLXArray {
        MuseGlimmerFixture.hostTensor(
            [grid.patchRows * grid.patchColumns, Geometry.patchValues], seed: seed)
    }

    /// A colored checkerboard over a gradient: every resampled pixel differs.
    static func pattern(width: Int, height: Int) -> CIImage {
        let board = CIFilter(
            name: "CICheckerboardGenerator",
            parameters: [
                "inputColor0": CIColor(red: 0.9, green: 0.2, blue: 0.1),
                "inputColor1": CIColor(red: 0.1, green: 0.3, blue: 0.8), "inputWidth": 13,
            ])!.outputImage!
        let ramp = CIFilter(
            name: "CILinearGradient",
            parameters: [
                "inputPoint0": CIVector(x: 0, y: 0),
                "inputPoint1": CIVector(x: CGFloat(width), y: CGFloat(height)),
                "inputColor0": CIColor(red: 0, green: 0, blue: 0, alpha: 0.6),
                "inputColor1": CIColor(red: 1, green: 1, blue: 1, alpha: 0.6),
            ])!.outputImage!
        return ramp.composited(over: board).cropped(
            to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    // MARK: Geometry and prompts

    @Test func gridsAndTokenCountsAreTheProcessors() throws {
        // (height, width) -> merged rows x columns.
        let published: [(Int, Int, Int, Int)] = [
            (448, 448, 16, 16), (1080, 1920, 39, 69), (1792, 1792, 64, 64),
            (2160, 3840, 48, 85),
        ]
        for (height, width, rows, columns) in published {
            let grid = try Geometry.grid(sourceHeight: height, sourceWidth: width)
            #expect(grid.patchRows == 2 * rows && grid.patchColumns == 2 * columns)
            #expect(grid.featureRows == rows * columns)
            #expect(grid.promptTokens == rows * columns + 2)
        }
        #expect(try Geometry.grid(sourceHeight: 448, sourceWidth: 448).promptTokens == 258)
        #expect(Geometry.maximumPromptTokensPerImage == 4098)
        for (height, width) in [(10, 10), (28, 10_000), (10_000, 28), (517, 333), (3000, 37)] {
            for budget in [1, 256, 4096] {
                let grid = try Geometry.grid(
                    sourceHeight: height, sourceWidth: width, maximumImageTokens: budget)
                let (h, w) = museSmartResize(
                    height: height, width: width, factor: 28, maxTokens: budget)
                #expect(grid.pixelHeight == h && grid.pixelWidth == w)
                #expect(grid.featureRows == (h / 28) * (w / 28) && grid.featureRows >= 1)
                // Upstream's fallback exceeds the budget only when the aspect
                // ratio does, so reservations count `featureRows`, not the budget.
                #expect(grid.featureRows <= budget || max(h, w) / min(h, w) > budget)
                #expect(grid.thw.t == 1 && grid.thw.h == h / 14 && grid.thw.w == w / 14)
            }
        }
        let thin = try Geometry.grid(sourceHeight: 28, sourceWidth: 10_000, maximumImageTokens: 256)
        #expect(thin.patchRows == 2 && thin.patchColumns == 604 && thin.featureRows == 302)
        // At the full budget that fallback would pass 4,096 merged tokens; it is refused.
        #expect(throws: Failure.self) {
            try Geometry.grid(sourceHeight: 28, sourceWidth: 28 * 5000)
        }
        for (height, width, budget) in [(0, 10, 4096), (10, -1, 4096), (10, 10, 0), (10, 10, 4097)]
        {
            #expect(throws: Failure.self) {
                try Geometry.grid(
                    sourceHeight: height, sourceWidth: width, maximumImageTokens: budget)
            }
        }
        #expect(throws: Failure.self) { try Geometry.Grid(mergedRows: 65, mergedColumns: 64) }
        #expect(throws: Failure.self) { try Geometry.Grid(mergedRows: 0, mergedColumns: 4) }
    }

    @Test func expansionFramesEachImageAndRefusesMismatches() throws {
        let six = try Geometry.Grid(mergedRows: 2, mergedColumns: 3)
        let one = try Geometry.Grid(mergedRows: 1, mergedColumns: 1)
        let (patch, start, end) = (
            Prompt.imageTokenID, Prompt.imageStartTokenID, Prompt.imageEndTokenID
        )
        let expanded = try Prompt.expand(
            promptTokens: [5, patch, 6, 7, patch, 8], grids: [six, one])
        #expect(
            expanded.tokens
                == [5, start] + Array(repeating: patch, count: 6) + [
                    end, 6, 7, start, patch, end, 8,
                ])
        #expect(
            expanded.spans == [
                CBv2ImageSpan(tokenOffset: 2, length: 6), CBv2ImageSpan(tokenOffset: 12, length: 1),
            ])
        #expect(throws: Failure.placeholders("2 image placeholders for 1 images")) {
            try Prompt.expand(promptTokens: [patch, patch], grids: [one])
        }
        #expect(throws: Failure.placeholders("0 image placeholders for 0 images")) {
            try Prompt.expand(promptTokens: [1, 2], grids: [])
        }
        #expect(throws: Failure.unsupportedMedia("video placeholder")) {
            try Prompt.expand(promptTokens: [patch, Prompt.videoTokenID], grids: [one])
        }
    }

    /// The upstream processor's prompt and pixels for two images, from the same
    /// rendered tokens: the fork's expansion and preprocessing are identical.
    @Test func expansionAndPixelsEqualTheUpstreamProcessor() async throws {
        let rendered = [9, Prompt.imageTokenID, 10, 11, Prompt.imageTokenID, 12]
        let images = [Self.pattern(width: 300, height: 200), Self.pattern(width: 90, height: 700)]
        let processor = MuseGlimmerProcessor(
            MuseGlimmerProcessorConfiguration(),
            tokenizer: MuseGlimmerFixedPromptTokenizer(prompt: rendered))
        let input = try await processor.prepare(
            input: UserInput(
                prompt: "two images", images: images.map { .init(source: .ciImage($0)) }))
        let prepared = try images.map { try MuseGlimmerImageEncoder.preprocess(image: $0) }
        let expanded = try Prompt.expand(promptTokens: rendered, grids: prepared.map(\.grid))
        #expect(input.text.tokens.reshaped(-1).asArray(Int.self) == expanded.tokens)
        let upstreamPixels = try #require(input.image?.pixels)
        #expect(
            MuseGlimmerFixture.sameBits(
                upstreamPixels, concatenated(prepared.map(\.pixels), axis: 0)))
        let frames = try #require(input.image?.frames)
        #expect(frames.map(\.h) == prepared.map(\.grid.patchRows))
        #expect(frames.map(\.w) == prepared.map(\.grid.patchColumns))
    }

    @Test func preprocessingIsTheProcessorsBitForBit() throws {
        let processor = MuseGlimmerImageBudgetTests.processor()
        for (width, height) in [(448, 448), (1920, 1080), (333, 517), (37, 3000)] {
            let image = Self.pattern(width: width, height: height)
            let (expected, frame) = try processor.preprocess(image: image, processing: nil)
            let (pixels, grid) = try MuseGlimmerImageEncoder.preprocess(image: image)
            #expect(grid.patchRows == frame.h && grid.patchColumns == frame.w && frame.t == 1)
            #expect(pixels.shape == [grid.patchRows * grid.patchColumns, 1176])
            #expect(pixels.dtype == .float32)
            #expect(MuseGlimmerFixture.sameBits(pixels, expected))
        }
        let large = Self.pattern(width: 2400, height: 2400)
        let (_, capped) = try MuseGlimmerImageEncoder.preprocess(
            image: large, maximumImageTokens: 256)
        #expect(capped.featureRows <= 256 && capped.featureRows > 128)
    }

    // MARK: Encoding

    /// Features spliced into the native tower's prompt embeddings give the
    /// upstream model's image prefill and the decode after it, byte for byte.
    @Test(arguments: [DType.bfloat16, DType.float32])
    func splicedFeaturesReproduceTheUpstreamImagePrefill(dtype: DType) throws {
        let (upstream, native) = try MuseGlimmerFixture.upstreamPair(dtype: dtype)
        let stack = try Self.stack(upstream)
        // 36 x 8 patches: two 32-patch windows on the windowed layer.
        let grid = try Geometry.Grid(mergedRows: 18, mergedColumns: 4)
        let pixels = Self.pixels(grid, seed: 41)
        let features = stack(pixels, grid: [grid.thw])
        #expect(features.shape == [grid.featureRows, 32] && features.dtype == dtype)

        // Text ids avoid the fixture's image (7) and video (8) ids.
        let text = MuseGlimmerFixture.tokens(10, vocabulary: 64, salt: 3).map {
            $0 == 7 || $0 == 8 ? $0 + 2 : $0
        }
        let prompt: [Int32] = [1, 2, 5] + Array(repeating: 7, count: grid.featureRows) + [6] + text
        let span = CBv2ImageSpan(tokenOffset: 3, length: grid.featureRows)
        let ids = MLXArray(prompt).reshaped(1, -1)

        let upstreamCache = try upstream.newCache(parameters: nil)
        let result = try upstream.prepare(
            LMInput(text: .init(tokens: ids), image: .init(pixels: pixels, frames: [grid.thw])),
            cache: upstreamCache, state: nil, prefill: PrefillParameters(chunking: .unchunked))
        guard case .logits(let output) = result else {
            Issue.record("expected logits")
            return
        }
        let adapter = MuseGlimmerCBv2Adapter(native)
        let spliced = CBv2MultimodalPlan.spliceEmbeddings(
            textEmbeddings: adapter.embedPromptTokens(ids), chunkStart: 0,
            spans: [(span, features.expandedDimensions(axis: 0))])
        let nativeCache = try native.newCache(parameters: nil)
        var logits = native(inputEmbeddings: spliced, cache: nativeCache)
        #expect(logits.shape == [1, prompt.count, 64])
        #expect(MuseGlimmerFixture.sameBits(logits, output.logits))

        var next = argMax(logits[0, -1], axis: -1).item(Int.self)
        var generated = [Int]()
        for _ in 0 ..< 16 {
            generated.append(next)
            let token = MLXArray([Int32(next)]).reshaped(1, 1)
            let expected = upstream(token, cache: upstreamCache)
            logits = native(token, cache: nativeCache)
            #expect(MuseGlimmerFixture.sameBits(logits, expected))
            next = argMax(logits[0, -1], axis: -1).item(Int.self)
        }
        #expect(Set(generated).count > 2)
        // The image is read: other pixels give other logits.
        let other = stack(Self.pixels(grid, seed: 42), grid: [grid.thw])
        let otherLogits = native(
            inputEmbeddings: CBv2MultimodalPlan.spliceEmbeddings(
                textEmbeddings: adapter.embedPromptTokens(ids), chunkStart: 0,
                spans: [(span, other.expandedDimensions(axis: 0))]),
            cache: nil)
        #expect(!MuseGlimmerFixture.sameBits(otherLogits, output.logits))
    }

    // MARK: Loading

    /// The fixture's vision weights as checkpoint tensors (`model.` keys).
    static func checkpointTensors(_ upstream: MuseGlimmer)
        -> [String: MuseGlimmerTinyCheckpoint.Tensor]
    {
        var tensors = [String: MuseGlimmerTinyCheckpoint.Tensor]()
        for (key, value) in visionWeights(upstream) {
            tensors["model." + key] = tensor(value)
        }
        return tensors
    }

    static func tensor(_ value: MLXArray, dtype: DType = .bfloat16)
        -> MuseGlimmerTinyCheckpoint.Tensor
    {
        let stored = value.asType(dtype)
        return .init(
            dtype: dtype == .float32 ? "F32" : "BF16", shape: value.shape,
            bytes: Array(stored.asData(access: .copy).data))
    }

    /// Verified sources over a vision-only checkpoint with these tensors.
    static func sources(_ tensors: [String: MuseGlimmerTinyCheckpoint.Tensor]) throws
        -> (sources: [String: ModelOptTensorSource], directory: URL)
    {
        var checkpoint = try MuseGlimmerTinyCheckpoint()
        checkpoint.tensors = tensors
        let written = try checkpoint.write()
        return (try written.source().tensors, written.directory)
    }

    @Test func strictLoadingTakesExactlyTheStacksBF16Tensors() throws {
        let (upstream, _) = try MuseGlimmerFixture.upstreamPair(dtype: .bfloat16)
        let stack = try Self.stack(upstream)
        let tensors = Self.checkpointTensors(upstream)
        let (sources, directory) = try Self.sources(tensors)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(sources.count == stack.parameters().flattened().count)

        let encoder = try MuseGlimmerImageEncoder.load(
            configuration: upstream.config, sources: sources)
        #expect(encoder.featureWidth == 32)
        #expect(
            encoder.parameterBytes
                == stack.parameters().flattened().reduce(0) { $0 + $1.1.size * 2 })
        let grid = try Geometry.Grid(mergedRows: 3, mergedColumns: 5)
        let pixels = Self.pixels(grid, seed: 5)
        let features = try encoder.features(pixels: pixels, grid: grid)
        #expect(features.shape == [15, 32] && features.dtype == .bfloat16)
        #expect(MuseGlimmerFixture.sameBits(features, stack(pixels, grid: [grid.thw])))
        // Pixels that are not this grid's are refused before any graph.
        for bad in [pixels[0 ..< 4], pixels.reshaped(1, -1, 1176), pixels.asType(.int32)] {
            #expect(throws: Failure.self) { try encoder.features(pixels: bad, grid: grid) }
        }

        // One tensor short, one extra, one of another dtype or shape, one
        // bound to another key: each is refused by name.
        let missing = "model.vision_adapter.fc2.weight"
        #expect(throws: Failure.missing(missing)) {
            try MuseGlimmerImageEncoder.load(
                configuration: upstream.config, sources: sources.filter { $0.key != missing })
        }
        var renamed = sources
        renamed["vision_projection.weight"] = renamed.removeValue(
            forKey: "model.vision_projection.weight")
        #expect(throws: Failure.unexpected("vision_projection.weight")) {
            try MuseGlimmerImageEncoder.load(configuration: upstream.config, sources: renamed)
        }
        var swapped = sources
        swapped["model.vision_tower.ln_pre.bias"] = sources["model.vision_tower.ln_pre.weight"]
        #expect(throws: Failure.unexpected("model.vision_tower.ln_pre.bias")) {
            try MuseGlimmerImageEncoder.load(configuration: upstream.config, sources: swapped)
        }
        let extra = "model.vision_tower.extra.weight"
        let fc1 = "model.vision_adapter.fc1.weight"
        let norm = "model.vision_tower.ln_post.weight"
        let weights = Self.visionWeights(upstream)
        let edits: [(String, [String: MuseGlimmerTinyCheckpoint.Tensor], Failure)] = [
            (
                "extra", tensors.merging([extra: Self.tensor(MLXArray.zeros([4]))]) { a, _ in a },
                .unexpected(extra)
            ),
            (
                "dtype",
                tensors.merging([
                    norm: Self.tensor(weights["vision_tower.ln_post.weight"]!, dtype: .float32)
                ]) { _, b in b },
                .dtype(norm)
            ),
            (
                "shape",
                tensors.merging([
                    fc1: Self.tensor(weights["vision_adapter.fc1.weight"]!.transposed())
                ]) { _, b in b },
                .layout(fc1)
            ),
        ]
        for (label, edited, failure) in edits {
            let (bad, badDirectory) = try Self.sources(edited)
            defer { try? FileManager.default.removeItem(at: badDirectory) }
            #expect(throws: failure, "\(label)") {
                try MuseGlimmerImageEncoder.load(configuration: upstream.config, sources: bad)
            }
        }
    }

    /// The production entry takes only the pinned artifact's receipt.
    @Test func theProductionEntryRefusesAnUnpinnedInstallation() throws {
        let (upstream, _) = try MuseGlimmerFixture.upstreamPair(dtype: .bfloat16)
        var checkpoint = try MuseGlimmerTinyCheckpoint()
        checkpoint.tensors = checkpoint.tensors.filter { !$0.key.hasPrefix("model.vision_") }
        checkpoint.tensors.merge(Self.checkpointTensors(upstream)) { a, _ in a }
        let written = try checkpoint.write()
        defer { try? FileManager.default.removeItem(at: written.directory) }
        let installed = try MuseGlimmerInstallation.stage(
            configuration: checkpoint.configuration, source: try written.source(),
            dtype: .bfloat16, activationOverflow: .museGlimmer)
        #expect(installed.visionSources.count == Self.visionWeights(upstream).count)
        #expect(throws: Failure.identity("checkpoint pin")) {
            try MuseGlimmerImageEncoder.load(installed: installed)
        }
        // The same verified sources load through the unpinned test entry.
        _ = try MuseGlimmerImageEncoder.load(
            configuration: upstream.config, sources: installed.visionSources)
    }

    @Test func checkpointConfigurationIsCrossCheckedNeverUsed() throws {
        func check(_ json: String) throws {
            try MuseGlimmerImageEncoder.crossCheck(checkpointConfiguration: Data(json.utf8))
        }
        try check(#"{"text_config": {}, "vision_config": {}}"#)
        try check(
            #"{"text_config": {}, "vision_config": {"hidden_act": "gelu"}, "projector_hidden_act": "gelu"}"#
        )
        let refused = [
            #"{"text_config": {}, "vision_config": {"hidden_size": 1280}}"#,
            #"{"text_config": {}, "vision_config": {"num_hidden_layers": 32}}"#,
            #"{"text_config": {"rms_norm_eps": 1e-6}, "vision_config": {}}"#,
            #"{"text_config": {"hidden_size": 5120}, "vision_config": {}}"#,
            #"{"text_config": {}, "vision_config": {}, "image_token_id": 7}"#,
            #"{"text_config": {}, "vision_config": {}, "out_hidden_size": 4096}"#,
            #"{"text_config": {}, "vision_config": {"rope_parameters": {"rope_theta": 5000}}}"#,
            "{}", "not json",
        ]
        for json in refused {
            #expect(throws: Failure.identity("config.json"), "\(json)") { try check(json) }
        }
        #expect(throws: Failure.identity("config.json activation")) {
            try check(#"{"text_config": {}, "vision_config": {}, "projector_hidden_act": "relu"}"#)
        }
        #expect(throws: Failure.identity("config.json activation")) {
            try check(#"{"text_config": {}, "vision_config": {"hidden_act": "gelu_pytorch_tanh"}}"#)
        }
    }

    /// The published stack: 809 BF16 tensors of 3,843,691,520 bytes.
    @Test func thePinnedStackHasThePublishedShape() throws {
        let c = MuseGlimmerImageEncoder.pinnedConfiguration
        #expect(c.imageTokenId == Prompt.imageTokenID && c.videoTokenId == Prompt.videoTokenID)
        #expect(c.outHiddenSize == 6144 && c.projectorHiddenSize == 4096)
        #expect(c.textConfiguration.hiddenSize == 6656 && c.textConfiguration.rmsNormEps == 1e-5)
        let v = c.visionConfiguration
        #expect(v.hiddenSize == 1536 && v.intermediateSize == 8960 && v.attentionHeads == 16)
        #expect(v.hiddenLayers == 50 && v.layerTypes.filter { $0 == "full_attention" }.count == 13)
        #expect(v.patchSize == 14 && v.patchTemporal == 2 && v.mergeSize == 2)
        #expect(v.positionEmbeddingHeight == 32 && v.ropeTheta == 10_000)
        let parameters = MuseGlimmerVisionFeatures(c).parameters().flattened()
        #expect(parameters.count == 809)
        #expect(parameters.reduce(0) { $0 + $1.1.size * 2 } == 3_843_691_520)
        let shapes = Dictionary(uniqueKeysWithValues: parameters.map { ($0.0, $0.1.shape) })
        #expect(shapes["vision_tower.patch_embedder.patch_embedding.weight"] == [1536, 1176])
        #expect(
            shapes["vision_tower.patch_embedder.position_embedding_table.weight"] == [1024, 1536])
        #expect(shapes["vision_adapter.fc1.weight"] == [4096, 6144])
        #expect(shapes["vision_adapter.fc2.weight"] == [4096, 4096])
        #expect(shapes["vision_projection.weight"] == [6656, 4096])
        #expect(!shapes.keys.contains { $0.hasPrefix("perception_emb_norm") })
    }
}
