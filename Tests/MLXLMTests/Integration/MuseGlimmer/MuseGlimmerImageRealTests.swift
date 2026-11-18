// Copyright © 2026 Eigen Labs.

import CoreGraphics
import CoreImage
import CoreText
import Foundation
import MLX
import MLXHuggingFace
import MLXNN
import Testing
import Tokenizers

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The image path on the pinned NVIDIA checkpoint. `MUSE_GLIMMER_NVFP4_DIR`
/// names it; `MUSE_GLIMMER_VISION_ORACLE_DIR` holds the float32 CPU
/// reference's fixtures and `MUSE_GLIMMER_VISION_BF16_ORACLE_DIR` the same
/// reference with bf16 storage emulated; the drafter row needs
/// `MUSE_GLIMMER_DFLASH_QUANTIZED_DIR` or `MUSE_GLIMMER_ASSISTANT_DIR`. Run
/// under the GPU lane.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerImageRealTests {
    typealias Geometry = MuseGlimmerImageGeometry
    typealias Prompt = MuseGlimmerImagePrompt
    static let environment = ProcessInfo.processInfo.environment
    static let oracleDirectory = environment["MUSE_GLIMMER_VISION_ORACLE_DIR"]
    static let emulatedOracleDirectory = environment["MUSE_GLIMMER_VISION_BF16_ORACLE_DIR"]
    static let quantizedDrafter = environment["MUSE_GLIMMER_DFLASH_QUANTIZED_DIR"]
    static let bf16Drafter = environment["MUSE_GLIMMER_ASSISTANT_DIR"]
    static var hasWeights: Bool { MuseGlimmerRealArtifact.directory != nil }
    static var hasOracle: Bool {
        hasWeights && oracleDirectory != nil && emulatedOracleDirectory != nil
    }
    static var hasDrafter: Bool { hasWeights && (quantizedDrafter != nil || bf16Drafter != nil) }

    struct Staged {
        let installed: MuseGlimmerInstallation.Installed
        let encoder: MuseGlimmerImageEncoder
        let encoderSeconds: Double
        let encoderActiveBytes: Int
    }

    /// The tower with exact windows (the provider's setting) and its encoder.
    nonisolated(unsafe) static let staged: Result<Staged, Error> = Result {
        guard let directory = MuseGlimmerRealArtifact.directory else {
            throw MuseGlimmerNPZ.Failure.malformed("no directory")
        }
        let installed = try MuseGlimmerInstallation.stage(directory: directory)
        try installed.model.setRowExactWindows(true)
        let configuration = try ModelOptIndexedCheckpoint.readRegularFile(
            at: directory.appendingPathComponent("config.json"), maximumBytes: 1 << 20)
        let before = Memory.activeMemory
        let start = Date()
        let encoder = try MuseGlimmerImageEncoder.load(
            installed: installed, checkpointConfiguration: configuration)
        return Staged(
            installed: installed, encoder: encoder,
            encoderSeconds: Date().timeIntervalSince(start),
            encoderActiveBytes: Memory.activeMemory - before)
    }

    static func gib(_ bytes: Int) -> String {
        String(format: "%.3f", Double(bytes) / 1_073_741_824)
    }

    @Test(.enabled(if: hasWeights))
    func theEncoderTakesExactlyTheCheckpointsVisionTensors() throws {
        let staged = try Self.staged.get()
        let keys = Set(staged.installed.visionSources.keys)
        let module = Set(staged.encoder.stack.parameters().flattened().map { "model." + $0.0 })
        #expect(keys.count == 809 && keys == module)
        #expect(staged.installed.visionSources.values.allSatisfy { $0.layout.dtype == "BF16" })
        #expect(staged.encoder.parameterBytes == 3_843_691_520)
        #expect(staged.encoder.featureWidth == 6656)
        MuseGlimmerRealArtifact.record(
            "vision_load seconds=\(String(format: "%.1f", staged.encoderSeconds)) "
                + "mlx_active_delta_gib=\(Self.gib(staged.encoderActiveBytes)) "
                + "parameter_gib=\(Self.gib(staged.encoder.parameterBytes))")
    }

    struct Agreement {
        var maxAbs: Float = 0
        var relativeL2: Float = 0
        var minimumRowCosine: Float = 1
        var meanRowCosine: Float = 0

        init(_ ours: [Float], _ reference: [Float], width: Int) {
            var difference: Double = 0
            var norm: Double = 0
            var cosines = [Double]()
            for row in stride(from: 0, to: reference.count, by: width) {
                var (dot, a2, b2) = (0.0, 0.0, 0.0)
                for i in row ..< row + width {
                    let (a, b) = (Double(ours[i]), Double(reference[i]))
                    maxAbs = max(maxAbs, Float(abs(a - b)))
                    difference += (a - b) * (a - b)
                    norm += b * b
                    dot += a * b
                    a2 += a * a
                    b2 += b * b
                }
                cosines.append(dot / max((a2 * b2).squareRoot(), 1e-30))
            }
            relativeL2 = Float((difference / max(norm, 1e-30)).squareRoot())
            minimumRowCosine = Float(cosines.min() ?? 1)
            meanRowCosine = Float(cosines.reduce(0, +) / Double(max(cosines.count, 1)))
        }

        var line: String {
            String(
                format: "max_abs=%.5f rel_l2=%.6f min_row_cos=%.6f mean_row_cos=%.6f", maxAbs,
                relativeL2, minimumRowCosine, meanRowCosine)
        }
    }

    /// Proof 4. The same weights in float32 on the GPU match the float32 CPU
    /// reference within 1e-4 relative L2: the arithmetic is the reference's.
    /// The BF16 stack's distance from that reference stays within twice the
    /// distance of the reference itself run with bf16 storage, on relative L2
    /// and on the minimum and mean row cosine gaps. The tolerance first set
    /// (relative L2 at most 2e-2, every row's cosine at least 0.99) failed at
    /// 256 and 720 tokens; the emulated reference showed that drift is what
    /// bf16 storage gives across 50 layers, so the bound is measured from it.
    @Test(.enabled(if: hasOracle), arguments: ["g2x2", "g32x32", "g40x72"])
    func featuresMatchTheFloat32CPUReference(name: String) throws {
        let staged = try Self.staged.get()
        let file = try MuseGlimmerNPZ(
            url: URL(fileURLWithPath: try #require(Self.oracleDirectory)).appendingPathComponent(
                "\(name).npz"))
        let emulatedFile = try MuseGlimmerNPZ(
            url: URL(fileURLWithPath: try #require(Self.emulatedOracleDirectory))
                .appendingPathComponent("\(name).npz"))
        let thw = try file.ints("grid").map(Int.init)
        try #require(thw.count == 3 && thw[0] == 1)
        let grid = try Geometry.Grid(mergedRows: thw[1] / 2, mergedColumns: thw[2] / 2)
        try #require(grid.patchRows == thw[1] && grid.patchColumns == thw[2])
        let pixels = MLXArray(try file.floats("pixels")).reshaped(file.shape("pixels"))
        let reference = try file.floats("features")
        try #require(file.shape("features") == [grid.featureRows, 6656])
        try #require(try emulatedFile.floats("pixels") == file.floats("pixels"))
        try #require(try emulatedFile.ints("grid") == file.ints("grid"))
        let emulated = try Agreement(emulatedFile.floats("features"), reference, width: 6656)

        let features = try staged.encoder.features(pixels: pixels, grid: grid)
        #expect(features.dtype == .bfloat16)
        let bf16 = Agreement(
            features.asType(.float32).asArray(Float.self), reference, width: 6656)

        let f32 = MuseGlimmerVisionFeatures(MuseGlimmerImageEncoder.pinnedConfiguration)
        let upcast = staged.encoder.stack.parameters().flattened().map {
            ($0.0, $0.1.asType(.float32))
        }
        try f32.update(parameters: .unflattened(upcast), verify: [.all])
        var exact = MLXArray.zeros([1])
        try MLX.withError { errors in
            exact = f32(pixels, grid: [grid.thw])
            eval(exact)
            try errors.check()
        }
        #expect(exact.dtype == .float32)
        let float32 = Agreement(exact.asArray(Float.self), reference, width: 6656)
        MuseGlimmerRealArtifact.record(
            "vision_parity \(name) patches=\(grid.patchRows)x\(grid.patchColumns) "
                + "tokens=\(grid.featureRows) bf16 \(bf16.line) | f32 \(float32.line) "
                + "| emulated_bf16 \(emulated.line)")
        #expect(float32.relativeL2 <= 1e-4, "\(name) \(float32.line)")
        #expect(
            bf16.relativeL2 <= 2 * emulated.relativeL2
                && 1 - bf16.minimumRowCosine <= 2 * (1 - emulated.minimumRowCosine)
                && 1 - bf16.meanRowCosine <= 2 * (1 - emulated.meanRowCosine),
            "\(name) \(bf16.line) against \(emulated.line)")
    }

    /// The tower's time and MLX peak above the resident weights for the
    /// production grids: 448² (256 tokens), 1920×1080 (2,691) and the
    /// 4,096-token cap (1792²). The second call of each grid is reported.
    @Test(.enabled(if: hasWeights))
    func towerWorkingSetPerGrid() throws {
        let staged = try Self.staged.get()
        for (height, width) in [(448, 448), (1080, 1920), (1792, 1792)] {
            let grid = try Geometry.grid(sourceHeight: height, sourceWidth: width)
            let pixels = MuseGlimmerFixture.hostTensor(
                [grid.patchRows * grid.patchColumns, Geometry.patchValues], seed: 9)
            eval(pixels)
            _ = try staged.encoder.features(pixels: pixels, grid: grid)
            Memory.clearCache()
            let resident = Memory.activeMemory
            Memory.peakMemory = 0
            let start = Date()
            let features = try staged.encoder.features(pixels: pixels, grid: grid)
            let seconds = Date().timeIntervalSince(start)
            let peak = Memory.peakMemory
            #expect(features.shape == [grid.featureRows, 6656])
            MuseGlimmerRealArtifact.record(
                "vision_tower source=\(width)x\(height) tokens=\(grid.featureRows) "
                    + "patches=\(grid.patchRows * grid.patchColumns) "
                    + "seconds=\(String(format: "%.3f", seconds)) "
                    + "peak_above_resident_gib=\(Self.gib(peak - resident)) "
                    + "load=\(Self.loadAverage())")
        }
    }

    static func loadAverage() -> String {
        var loads = [Double](repeating: 0, count: 3)
        return getloadavg(&loads, 3) == 3 ? String(format: "%.1f", loads[0]) : "?"
    }

    // MARK: S3: an image row beside text rows

    static func drafterModel() throws -> MuseGlimmerDFlashDraftModel {
        if let quantizedDrafter {
            return try MuseGlimmerDFlashDraftModel.load(
                directory: URL(fileURLWithPath: quantizedDrafter), pin: .museGlimmer30BQuantized,
                maximumBytes: 2 << 30)
        }
        return try MuseGlimmerDFlashDraftModel.load(
            directory: URL(fileURLWithPath: try #require(bf16Drafter)), pin: .museGlimmer30BBF16,
            maximumBytes: 5_111_970_304)
    }

    private func engine(_ model: MuseGlimmerTextModel, drafter: (any CBv2MTPDrafter)?) throws
        -> (EngineV2, PagedKVBackend)
    {
        let backend = try PagedKVBackend(
            layerKinds: model.cbv2LayerKinds,
            config: .init(
                capacityBytes: 2 << 30, dtype: .bfloat16, maxPrefillChunk: 1024,
                nominalMaxSequenceLength: 8192, segmentSizeBytes: 8 << 20,
                layerDTypes: model.kvStorageDTypes))
        let engine = EngineV2(
            model: MuseGlimmerCBv2Adapter(model), layerKinds: model.cbv2LayerKinds,
            backend: backend, cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            schedulerConfig: .init(
                maxConcurrentRequests: 3, maxBatchedTokensPerStep: 1024, prefillChunkSize: 1024,
                soloPrefillStripeTokens: 1024, maxWaiting: 4, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0), mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: 7, maxSpeculativeBatch: 2,
                fixedDraftTokens: 7, verificationMode: .rectangularExact,
                maxAutomaticRectangularTokens: 16))
        return (engine, backend)
    }

    /// The chat prompt for `text` after `images`, through the published
    /// template and the fork's expansion, and the evaluated features.
    static func imagePrompt(
        _ text: String, images: [CIImage], tokenizer: any MLXLMCommon.Tokenizer,
        encoder: MuseGlimmerImageEncoder
    ) throws -> (tokens: [Int], input: CBv2MultimodalInput) {
        var content: [[String: any Sendable]] = images.map { _ in ["type": "image"] }
        content.append(["type": "text", "text": text])
        let rendered = try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": "You are a helpful AI assistant."],
                ["role": "user", "content": content],
            ],
            tools: nil, additionalContext: ["reasoning_strength": "low"])
        let prepared = try images.map { try MuseGlimmerImageEncoder.preprocess(image: $0) }
        let expanded = try Prompt.expand(promptTokens: rendered, grids: prepared.map(\.grid))
        let features = try prepared.map { try encoder.features(pixels: $0.pixels, grid: $0.grid) }
        return (
            expanded.tokens,
            CBv2MultimodalInput(spans: expanded.spans, attention: .causal) { features }
        )
    }

    /// An image request and two oracle text requests share the drafting
    /// engine: every output equals its drafter-off solo run and only the text
    /// requests get drafter state. A step whose cohort holds the image row
    /// takes no drafts, so the image request is the shorter one: the text rows
    /// must speculate once it has left.
    @Test(.enabled(if: hasDrafter && MuseGlimmerRealArtifact.hasOracle), .timeLimit(.minutes(40)))
    func anImageRowBesideTextRowsRunsOnTheTargetAlone() async throws {
        let staged = try Self.staged.get()
        let model = staged.installed.model
        let directory = try #require(MuseGlimmerRealArtifact.directory)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        let (imageTokens, media) = try Self.imagePrompt(
            "Read the store name and the total on this receipt.",
            images: [MuseGlimmerProofImages.receipt()], tokenizer: tokenizer,
            encoder: staged.encoder)
        let (imageMaxTokens, maxTokens) = (24, 96)
        let greedy = CBv2SamplingParams(temperature: 0, seed: 0)
        var requests = [
            CBv2Request(
                id: .init(1), promptTokens: imageTokens, sampling: greedy,
                maxTokens: imageMaxTokens, multimodal: media)
        ]
        for (index, name) in ["P1", "P2"].enumerated() {
            let prompt = try MuseGlimmerExactWindowRealTests.oracle(name).prompt
            requests.append(
                CBv2Request(
                    id: .init(UInt64(index + 2)), promptTokens: prompt, sampling: greedy,
                    maxTokens: maxTokens))
        }
        var solo = [[Int]]()
        for request in requests {
            let (plain, backend) = try engine(model, drafter: nil)
            let result = await cbv2SchedCollect(try plain.submit(request), timeoutSeconds: 1200)
            await plain.shutdown()
            #expect(result.finishReason != nil && backend.bytesReserved == 0)
            solo.append(result.tokens)
        }
        let draft = try MuseGlimmerDFlashDrafter(
            target: model, model: try Self.drafterModel(),
            settings: .verifying(model, maximumSpeculativeRows: 2))
        // Native proposals; the wrapper only counts drafter states.
        let scripted = MuseGlimmerDFlashScriptedDrafter(
            draft, mode: .native, vocabulary: model.vocabularySize)
        let (speculative, backend) = try engine(model, drafter: scripted)
        #expect(speculative.mtpInactiveReason == nil)
        let streams = try requests.map { try speculative.submit($0) }
        var together = [[Int]]()
        for stream in streams {
            together.append(await cbv2SchedCollect(stream, timeoutSeconds: 1800).tokens)
        }
        let metrics = try #require(speculative.mtpMetricsSnapshot())
        speculative.loopForTesting.onEngineQueueSync {
            #expect(speculative.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
        await speculative.shutdown()
        let text = tokenizer.decode(tokenIds: solo[0], skipSpecialTokens: false)
        MuseGlimmerRealArtifact.record(
            "vision_beside_text image_prompt=\(imageTokens.count) spans=\(media.spans.count) "
                + "equal=\(zip(together, solo).map { $0 == $1 }) "
                + "rounds=\(metrics.rounds) drafted=\(metrics.draftedTokens) "
                + "accepted=\(metrics.acceptedTokens) states=\(scripted.states.count) "
                + "image_text=\(text.debugDescription)")
        #expect(together == solo)
        #expect(scripted.states.count == 2)
        #expect(metrics.rounds > 0 && metrics.serialVerificationRounds == 0)
        #expect(backend.bytesReserved == 0)
    }
}

/// Proof inputs rendered here: a receipt with known text, a colour-swatch
/// card and a plain shape. No third-party images.
enum MuseGlimmerProofImages {
    static let receiptLines = [
        "FERN & FOUNDRY MARKET",
        "2026-11-14",
        "Oat milk 1L 3.49",
        "Sourdough loaf 5.25",
        "Free-range eggs x12 6.80",
        "Green apples 1kg 4.10",
        "Dark chocolate 2.95",
        "Coffee beans 250g 11.40",
        "Subtotal 33.99",
        "Tax 8% 2.72",
        "Total 36.71",
    ]
    /// Swatches 1 to 4, left to right.
    static let swatches: [(name: String, red: CGFloat, green: CGFloat, blue: CGFloat)] = [
        ("red", 0.86, 0.1, 0.1), ("green", 0.1, 0.62, 0.2), ("blue", 0.1, 0.25, 0.86),
        ("yellow", 0.96, 0.84, 0.1),
    ]

    static func canvas(width: Int, height: Int, _ draw: (CGContext) -> Void) -> CIImage {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        draw(context)
        return CIImage(cgImage: context.makeImage()!)
    }

    static func text(_ string: String, x: CGFloat, y: CGFloat, size: CGFloat, in context: CGContext)
    {
        let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(
                red: 0, green: 0, blue: 0, alpha: 1),
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: string, attributes: attributes))
        context.textPosition = CGPoint(x: x, y: y)
        CTLineDraw(line, context)
    }

    /// 640 × 900: item name left, price right, one line each.
    static func receipt() -> CIImage {
        canvas(width: 640, height: 900) { context in
            var y: CGFloat = 830
            for (index, line) in receiptLines.enumerated() {
                let size: CGFloat = index == 0 ? 34 : 28
                if let split = line.lastIndex(of: " "), index >= 2 {
                    text(String(line[..<split]), x: 40, y: y, size: size, in: context)
                    text(
                        String(line[line.index(after: split)...]), x: 500, y: y, size: size,
                        in: context)
                } else {
                    text(line, x: 40, y: y, size: size, in: context)
                }
                y -= index == 1 || index == 7 ? 90 : 62
            }
        }
    }

    /// 800 × 300: four squares, each numbered 1 to 4 below it.
    static func swatchCard() -> CIImage {
        canvas(width: 800, height: 300) { context in
            for (index, swatch) in swatches.enumerated() {
                let x = CGFloat(40 + index * 190)
                context.setFillColor(
                    CGColor(red: swatch.red, green: swatch.green, blue: swatch.blue, alpha: 1))
                context.fill(CGRect(x: x, y: 90, width: 150, height: 170))
                text("\(index + 1)", x: x + 64, y: 30, size: 40, in: context)
            }
        }
    }

    /// `image` centre-cropped to the aspect of `width x height`, then scaled to it.
    static func fill(_ image: CIImage, _ width: Int, _ height: Int) -> CIImage {
        let extent = image.extent
        let aspect = CGFloat(width) / CGFloat(height)
        var crop = extent
        if extent.width / extent.height > aspect {
            crop.size.width = extent.height * aspect
            crop.origin.x += (extent.width - crop.width) / 2
        } else {
            crop.size.height = extent.width / aspect
            crop.origin.y += (extent.height - crop.height) / 2
        }
        let cropped = image.cropped(to: crop).transformed(
            by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
        let scaled = MediaProcessing.resampleLanczos(
            cropped, to: CGSize(width: width, height: height))
        return scaled.cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    /// 600 × 600: one black circle, the stand-in third image when no photo is given.
    static func circle() -> CIImage {
        canvas(width: 600, height: 600) { context in
            context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            context.fillEllipse(in: CGRect(x: 100, y: 100, width: 400, height: 400))
        }
    }
}
