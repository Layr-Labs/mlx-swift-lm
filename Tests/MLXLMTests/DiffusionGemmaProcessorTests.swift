import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("DiffusionGemma native media preparation", .serialized)
struct DiffusionGemmaProcessorTests {
    private struct FixedTokenizer: Tokenizer {
        let rendered: [Int]
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? {
            ["<|image|>": 258880, "<|video|>": 258884][token]
        }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] { rendered }
        func applyChatTemplate(messages: [Message], chatTemplate: String, tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] { rendered }
    }
    private func configuration() throws -> (DiffusionGemmaConfiguration, DiffusionGemmaProcessorConfiguration) {
        let url = try #require(Bundle.module.url(forResource: "diffusiongemma-root-config", withExtension: "json"))
        let model = try JSONDecoder().decode(DiffusionGemmaConfiguration.self, from: Data(contentsOf: url))
        let settings = try JSONDecoder().decode(DiffusionGemmaProcessorConfiguration.self, from: Data(#"{"processor_class":"DiffusionGemma4Processor","image_processor":{"image_processor_type":"Gemma4ImageProcessor","patch_size":16,"pooling_kernel_size":3,"max_soft_tokens":280}}"#.utf8))
        return (model, settings)
    }
    private func color(_ value: CIColor, width: Int = 64, height: Int = 64) -> CIImage {
        CIImage(color: value).cropped(to: .init(x: 0, y: 0, width: width, height: height))
    }

    @Test func imagesAndVideoFramesKeepInterleavedOrderAndActualGrids() async throws {
        let (model, settings) = try configuration()
        let processor = try DiffusionGemmaProcessor(configuration: settings, model: model,
            tokenizer: FixedTokenizer(rendered: [2, 258880, 17, 258884, 18, 258880, 3]), template: "fixture")
        let video: UserInput.Video = .frames([
            .init(frame: color(.blue), timeStamp: .zero),
            .init(frame: color(.green), timeStamp: CMTime(seconds: 1, preferredTimescale: 100)),
        ])
        let input = UserInput(messages: [["role": "user", "content": "fixture"]],
            images: [.ciImage(color(.red)), .ciImage(color(.white, width: 128, height: 64))], videos: [video])
        let result = try await processor.prepare(input: input)
        #expect(result.frames.map(\.kind) == [.image, .videoFrame, .videoFrame, .image])
        #expect(result.frames.map(\.span.length) == [256, 256, 256, 253])
        #expect(result.frames.compactMap(\.timestampSeconds) == [0, 1])
        #expect(!result.tokens.contains(258884), "Video uses native image-frame tokens, not an unsupported tower")
        #expect(result.tokens.filter { $0 == 258880 }.count == 1021)
        #expect(result.tokens.filter { $0 == 255999 }.count == 4)
        #expect(result.tokens.filter { $0 == 258882 }.count == 4)
        for frame in result.frames {
            #expect(frame.pixels.shape[0...1] == [1, 3])
            #expect(result.tokens[frame.span.tokenOffset..<(frame.span.tokenOffset + frame.span.length)].allSatisfy { $0 == 258880 })
        }
        let channelMeans = result.frames.map { $0.pixels.mean(axes: [0, 2, 3]).asArray(Float.self) }
        #expect(channelMeans[0][0] > 0.99 && channelMeans[0][1] < 0.01)
        #expect(channelMeans[1][2] > 0.99 && channelMeans[1][0] < 0.01)
    }

    @Test func unboundAndUnsafeInputsFailRecoverably() async throws {
        let (model, settings) = try configuration()
        let processor = try DiffusionGemmaProcessor(configuration: settings, model: model,
            tokenizer: FixedTokenizer(rendered: [2, 258880, 3]), template: "fixture")
        await #expect(throws: (any Error).self) { try await processor.prepare(input: UserInput(prompt: "missing media")) }
        await #expect(throws: (any Error).self) {
            try await processor.prepare(input: UserInput(prompt: "invalid extent", images: [.ciImage(CIImage(color: .red))]))
        }
        await #expect(throws: (any Error).self) {
            try await DiffusionGemmaVideoFrames.sample(.frames([]), transform: { $0.timeStamp.seconds })
        }
        for timestamps in [[0.0, 61.0], [1.0, 0.0]] {
            let frames = timestamps.map { UserInput.VideoFrame(frame: color(.red),
                timeStamp: CMTime(seconds: $0, preferredTimescale: 100)) }
            await #expect(throws: (any Error).self) {
                try await DiffusionGemmaVideoFrames.sample(.frames(frames), transform: { $0.timeStamp.seconds })
            }
        }
        // CMTime(seconds: .nan, ...) can coerce to a numeric zero. Construct
        // the actual invalid/nonfinite CoreMedia representations instead.
        for timestamp: CMTime in [.invalid, .positiveInfinity, .negativeInfinity] {
            await #expect(throws: (any Error).self) {
                try await DiffusionGemmaVideoFrames.sample(.frames([
                    .init(frame: color(.red), timeStamp: timestamp),
                ]), transform: { $0.timeStamp.seconds })
            }
        }
    }

    /// Independent pinned processor pixels, not a color-name generation proxy.
    /// Output files contain only generated synthetic fixtures and never overwrite.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_PIXEL_REFERENCE_LIVE"] == "1"))
    func imagePixelsMatchIndependentReference() async throws {
        struct Item: Decodable {
            let name: String
            let width: Int, height: Int, targetWidth: Int, targetHeight: Int, softTokens: Int
        }
        struct Manifest: Decodable { let cases: [Item] }
        let root = URL(fileURLWithPath: try #require(
            ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_PIXEL_REFERENCE_DIR"]), isDirectory: true)
        let items = try JSONDecoder().decode(Manifest.self,
            from: Data(contentsOf: root.appendingPathComponent("manifest.json"))).cases
        try #require(!items.isEmpty)
        let (model, settings) = try configuration()
        let processor = try DiffusionGemmaProcessor(configuration: settings, model: model,
            tokenizer: FixedTokenizer(rendered: [2, 258880, 3]), template: "fixture")
        for item in items {
            try #require(!item.name.isEmpty && item.name.allSatisfy { $0.isASCII && ($0.isLetter || $0 == "-") })
            try #require(item.width > 0 && item.height > 0 && item.width <= 4096 && item.height <= 4096)
            let input = try Data(contentsOf: root.appendingPathComponent(item.name + ".rgba8"))
            try #require(input.count == item.width * item.height * 4)
            let image = CIImage(bitmapData: input, bytesPerRow: item.width * 4,
                size: CGSize(width: item.width, height: item.height), format: .RGBA8,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
            let prepared = try await processor.prepare(input: UserInput(prompt: "fixture", images: [.ciImage(image)]))
            let frame = try #require(prepared.frames.first)
            #expect(prepared.frames.count == 1 && frame.span.length == item.softTokens)
            try #require(frame.pixels.shape == [1, 3, item.targetHeight, item.targetWidth])
            let actual = frame.pixels.asArray(Float.self)
            let reference = try Data(contentsOf: root.appendingPathComponent(item.name + ".reference.f32"))
            try #require(reference.count == actual.count * 4)
            let expected = reference.withUnsafeBytes { bytes in
                (0..<actual.count).map { bytes.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
            }
            let finite = actual.allSatisfy(\.isFinite)
            try #require(finite)
            let differences = zip(actual, expected).map { abs($0 - $1) }
            let mismatches = zip(actual, expected).filter { $0.bitPattern != $1.bitPattern }.count
            let output = root.appendingPathComponent(item.name + ".native-gpu.f32")
            try actual.withUnsafeBytes { try Data($0).write(to: output, options: .withoutOverwriting) }
            print("DIFFUSION_PIXEL_REFERENCE \(item.name) count=\(actual.count) mismatches=\(mismatches) "
                + "mae=\(differences.reduce(0, +) / Float(actual.count)) max=\(differences.max() ?? 0) "
                + "belowZero=\(actual.filter { $0 < 0 }.count) aboveOne=\(actual.filter { $0 > 1 }.count)")
            #expect(mismatches == 0, "Native processor differs from the independent pixel oracle; preserve both outputs")
        }
    }
}
