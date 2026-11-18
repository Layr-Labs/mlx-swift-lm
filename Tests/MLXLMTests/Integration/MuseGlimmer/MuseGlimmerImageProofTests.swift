// Copyright © 2026 Eigen Labs.

import CoreGraphics
import CoreImage
import CoreText
import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXNN
import Testing
import Tokenizers

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The image proofs (zebra, receipt, multi-image) for one arm, written as
/// JSON lines to `MUSE_GLIMMER_PROOF_OUT`; a script compares the arms.
/// `MUSE_GLIMMER_PROOF_ARM` is `nvfp4` (the pinned NVIDIA checkpoint through
/// EngineV2, `MUSE_GLIMMER_NVFP4_DIR`) or `bf16` (upstream `MuseGlimmer` on
/// the pinned original checkpoint, `MUSE_GLIMMER_BF16_DIR`). A photo for the
/// zebra cases comes from `MUSE_GLIMMER_PROOF_ZEBRA`; without one the third
/// image is a drawn circle. Greedy decoding. Run under the GPU lane.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerImageProofTests {
    static let environment = ProcessInfo.processInfo.environment
    static let arm = environment["MUSE_GLIMMER_PROOF_ARM"]
    static let output = environment["MUSE_GLIMMER_PROOF_OUT"]
    static let zebra = environment["MUSE_GLIMMER_PROOF_ZEBRA"]
    static var enabled: Bool {
        guard output != nil else { return false }
        switch arm {
        case "nvfp4": return MuseGlimmerRealArtifact.directory != nil
        case "bf16": return environment["MUSE_GLIMMER_BF16_DIR"] != nil
        default: return false
        }
    }
    static let maxTokens = 1024
    static let stopTokens: Set<Int> = [200001, 200008]

    struct Case {
        let name: String
        let images: [(label: String, image: CIImage)]
        let question: String
    }

    static func cases() throws -> [Case] {
        let receipt = ("receipt", MuseGlimmerProofImages.receipt())
        let swatches = ("swatches", MuseGlimmerProofImages.swatchCard())
        var cases = [Case]()
        var third = ("circle", MuseGlimmerProofImages.circle())
        if let zebra {
            let photo = try #require(CIImage(contentsOf: URL(fileURLWithPath: zebra)))
            let question = "What animal is this? One word."
            cases.append(
                Case(
                    name: "zebra-448",
                    images: [("zebra-448", MuseGlimmerProofImages.fill(photo, 448, 448))],
                    question: question))
            cases.append(
                Case(
                    name: "zebra-1920x1080",
                    images: [("zebra-1920x1080", MuseGlimmerProofImages.fill(photo, 1920, 1080))],
                    question: question))
            third = ("zebra-448", MuseGlimmerProofImages.fill(photo, 448, 448))
        }
        cases.append(
            Case(
                name: "receipt", images: [receipt],
                question:
                    "Transcribe this receipt exactly, one printed line per line, top to bottom. "
                    + "Put each item and its price on one line separated by a single space. "
                    + "Output only the transcription."))
        let multi =
            "Answer in three lines. 1. Which image (first, second or third) shows a receipt? "
            + "2. What is the receipt's total? 3. On the card of four numbered colour swatches, "
            + "what colour is swatch 3?"
        cases.append(Case(name: "multi-a", images: [third, receipt, swatches], question: multi))
        cases.append(Case(name: "multi-b", images: [receipt, swatches, third], question: multi))
        return cases
    }

    static func renderedPrompt(
        _ question: String, images: Int, tokenizer: any MLXLMCommon.Tokenizer
    ) throws
        -> [Int]
    {
        var content: [[String: any Sendable]] = (0 ..< images).map { _ in ["type": "image"] }
        content.append(["type": "text", "text": question])
        return try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": "You are a helpful AI assistant."],
                ["role": "user", "content": content],
            ],
            tools: nil, additionalContext: ["reasoning_strength": "low"])
    }

    /// SHA-256 of the RGBA8 sRGB bytes the image renders to.
    static func pixelDigest(_ image: CIImage) -> String {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        CIContext().render(
            image, toBitmap: &bytes, rowBytes: width * 4, bounds: image.extent, format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
    }

    static func write(_ record: [String: Any]) throws {
        let path = try #require(output)
        var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        data.append(0x0A)
        if let handle = FileHandle(forWritingAtPath: path) {
            handle.seekToEndOfFile()
            handle.write(data)
            try handle.close()
        } else {
            try data.write(to: URL(fileURLWithPath: path))
        }
    }

    typealias Geometry = MuseGlimmerImageGeometry
    /// Greedy continuation of the expanded prompt with the images' features.
    typealias Generate = (
        _ prompt: MuseGlimmerImagePrompt.Expanded,
        _ prepared: [(pixels: MLXArray, grid: Geometry.Grid)]
    ) async throws -> [Int]

    @Test(.enabled(if: enabled), .timeLimit(.minutes(90)))
    func proofsForOneArm() async throws {
        let start = Date()
        let (directory, generate) = try (Self.arm == "bf16" ? bf16Arm() : nvfp4Arm())
        let loadSeconds = Date().timeIntervalSince(start)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        if let zebra = Self.zebra {
            let data = try Data(contentsOf: URL(fileURLWithPath: zebra))
            try Self.write([
                "kind": "input", "name": "zebra-source",
                "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            ])
        }
        try Self.write([
            "kind": "arm", "arm": Self.arm ?? "", "load_seconds": loadSeconds,
            "directory": directory.lastPathComponent,
        ])
        for item in try Self.cases() {
            let rendered = try Self.renderedPrompt(
                item.question, images: item.images.count, tokenizer: tokenizer)
            let prepared = try item.images.map {
                try MuseGlimmerImageEncoder.preprocess(image: $0.image)
            }
            let expanded = try MuseGlimmerImagePrompt.expand(
                promptTokens: rendered, grids: prepared.map(\.grid))
            let caseStart = Date()
            let tokens = try await generate(expanded, prepared)
            let seconds = Date().timeIntervalSince(caseStart)
            try Self.write([
                "kind": "case", "arm": Self.arm ?? "", "name": item.name,
                "images": item.images.map(\.label),
                "image_pixels_sha256": item.images.map { Self.pixelDigest($0.image) },
                "grids": prepared.map { [$0.grid.patchRows, $0.grid.patchColumns] },
                "prompt_tokens": expanded.tokens.count,
                "prompt_sha256": SHA256.hash(
                    data: Data(expanded.tokens.map(String.init).joined(separator: ",").utf8)
                ).map { String(format: "%02x", $0) }.joined(),
                "tokens": tokens, "seconds": seconds,
                "text": tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false),
            ])
        }
    }

    // MARK: Arms

    /// The production path: pinned installation, exact windows, the
    /// encoder, causal spans through EngineV2.
    private func nvfp4Arm() throws -> (URL, Generate) {
        let directory = try #require(MuseGlimmerRealArtifact.directory)
        let installed = try MuseGlimmerInstallation.stage(directory: directory)
        let model = installed.model
        try model.setRowExactWindows(true)
        let encoder = try MuseGlimmerImageEncoder.load(
            installed: installed,
            checkpointConfiguration: try ModelOptIndexedCheckpoint.readRegularFile(
                at: directory.appendingPathComponent("config.json"), maximumBytes: 1 << 20))
        var next: UInt64 = 1
        let generate: Generate = { prompt, prepared in
            let features = try prepared.map {
                try encoder.features(pixels: $0.pixels, grid: $0.grid)
            }
            let backend = try PagedKVBackend(
                layerKinds: model.cbv2LayerKinds,
                config: .init(
                    capacityBytes: 4 << 30, dtype: .bfloat16, maxPrefillChunk: 1024,
                    nominalMaxSequenceLength: 16384, segmentSizeBytes: 8 << 20,
                    layerDTypes: model.kvStorageDTypes))
            let engine = EngineV2(
                model: MuseGlimmerCBv2Adapter(model), layerKinds: model.cbv2LayerKinds,
                backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 1024,
                    prefillChunkSize: 1024, soloPrefillStripeTokens: 1024, maxWaiting: 2,
                    enablePrefixCache: false),
                admissionConfig: .init(watermarkFraction: 0))
            var request = CBv2Request(
                id: .init(next), promptTokens: prompt.tokens,
                sampling: .init(temperature: 0, seed: 0), maxTokens: Self.maxTokens,
                multimodal: CBv2MultimodalInput(spans: prompt.spans, attention: .causal) {
                    features
                })
            request.stopTokens = Self.stopTokens
            next += 1
            let result = await cbv2SchedCollect(try engine.submit(request), timeoutSeconds: 3000)
            await engine.shutdown()
            #expect(result.finishReason != nil && backend.bytesReserved == 0)
            return result.tokens
        }
        return (directory, generate)
    }

    /// The reference: upstream `MuseGlimmer` in BF16 on the original
    /// checkpoint, every file pinned, upstream's own splice and prefill.
    private func bf16Arm() throws -> (URL, Generate) {
        let directory = URL(
            fileURLWithPath: try #require(Self.environment["MUSE_GLIMMER_BF16_DIR"]),
            isDirectory: true)
        let model = try MuseGlimmerBF16Reference.load(directory: directory)
        let generate: Generate = { prompt, prepared in
            let cache = try model.newCache(parameters: nil)
            let tokens = MLXArray(prompt.tokens.map(Int32.init)).expandedDimensions(axis: 0)
            let input = LMInput(
                text: .init(tokens: tokens),
                image: .init(
                    pixels: concatenated(prepared.map(\.pixels)), frames: prepared.map(\.grid.thw)))
            var generated = [Int]()
            var logits: MLXArray
            switch try model.prepare(
                input, cache: cache, state: nil, prefill: PrefillParameters(stepSize: 1024))
            {
            case .logits(let output): logits = output.logits
            case .tokens: throw MuseGlimmerImageError.geometry("upstream returned tokens")
            }
            while generated.count < Self.maxTokens {
                var token = 0
                try MLX.withError { errors in
                    token = argMax(logits[0..., -1, 0...], axis: -1).item(Int.self)
                    try errors.check()
                }
                generated.append(token)
                if Self.stopTokens.contains(token) { break }
                logits = model(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
            }
            return generated
        }
        return (directory, generate)
    }
}

/// Upstream `MuseGlimmer` on `meta-models/Muse-Glimmer-30B` at revision
/// `cec94fcfd6512f2cfe708138c32533b346ea4f69`, loaded the strict way: the
/// config, index and both shards pinned by SHA-256, every tensor of the
/// module present with its shape (`verify: .all`). A test reference only.
enum MuseGlimmerBF16Reference {
    static let configSHA256 = "5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14"
    static let indexSHA256 = "7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba"
    static let shards = [
        "model-00001-of-00002.safetensors":
            "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694",
        "model-00002-of-00002.safetensors":
            "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381",
    ]

    enum Failure: Error { case identity(String) }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func load(directory: URL) throws -> MuseGlimmer {
        let configData = try ModelOptIndexedCheckpoint.readRegularFile(
            at: directory.appendingPathComponent("config.json"), maximumBytes: 1 << 20)
        guard sha256(configData) == configSHA256 else { throw Failure.identity("config.json") }
        let configuration = try JSONDecoder().decode(
            MuseGlimmerConfiguration.self, from: configData)
        let indexData = try ModelOptIndexedCheckpoint.readRegularFile(
            at: directory.appendingPathComponent("model.safetensors.index.json"),
            maximumBytes: ModelOptIndexedCheckpoint.maximumIndexBytes)
        let names = try ModelOptIndexedCheckpoint.shardNames(
            indexData: indexData, indexSHA256: indexSHA256)
        guard Set(names) == Set(shards.keys) else { throw Failure.identity("shard set") }
        var files = [String: ModelOptTensorFile]()
        for name in names.sorted() {
            files[name] = try ModelOptTensorFile(
                url: directory.appendingPathComponent(name), sha256: shards[name]!)
        }
        let checkpoint = try ModelOptIndexedCheckpoint(
            indexData: indexData, indexSHA256: indexSHA256, files: files)
        let model = MuseGlimmer(configuration)
        var weights = [String: MLXArray]()
        for (key, source) in checkpoint.tensors.sorted(by: { $0.key < $1.key }) {
            guard source.layout.dtype == "BF16" else { throw Failure.identity("dtype \(key)") }
            weights[key] = try source.array(maximumBytes: source.layout.byteCount)
        }
        try model.update(parameters: .unflattened(model.sanitize(weights: weights)), verify: [.all])
        try MLX.withError { errors in
            eval(model)
            try errors.check()
        }
        return model
    }
}
