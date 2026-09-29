import CryptoKit
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Routing tokenizer only; NOT a Jinja/BPE numerical oracle. Native feature
/// and engine tests below use real MiMo modules, caches and actual EngineV2.
final class MiMoMediaTestTokenizer: Tokenizer, @unchecked Sendable {
    let ids = [
        "<|vision_start|>": 4, "<|vision_end|>": 5, "<|image_pad|>": 2, "<|video_pad|>": 3,
        "<|mimo_video_start|>": 9, "<|mimo_video_end|>": 10, "<|mimo_audio_start|>": 7,
        "<|mimo_audio_end|>": 8, "<|audio_pad|>": 6,
    ]
    var override: [Int]?
    var messages: [Message] = []
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map { 20 + Int($0) % 32 }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "routing-fixture" }
    func convertTokenToId(_ token: String) -> Int? { ids[token] }
    func convertIdToToken(_ id: Int) -> String? { ids.first { $0.value == id }?.key }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages, chatTemplate: "fixture", tools: tools,
            additionalContext: additionalContext)
    }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(
            messages: messages, chatTemplate: chatTemplate, tools: nil, additionalContext: nil)
    }
    func applyChatTemplate(
        messages: [Message], chatTemplate: String, tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        self.messages = messages
        if let override { return override }
        var result = [11]
        for message in messages {
            for part in message["content"] as! [[String: any Sendable]] {
                switch part["type"] as! String {
                case "image": result += [4, 2, 5]
                case "video": result += [4, 3, 5]
                case "audio": result += [7, 6, 8]
                default: result += encode(text: part["text"] as! String, addSpecialTokens: true)
                }
            }
        }
        return result + [12]
    }
}

enum MiMoMediaFixture {
    final class Owner {}
    final class Reservation: MiMoV26MediaWorkReservation {
        let digest: String
        var calls = 0, revoked = false
        var quarantine: ((MiMoV26FailedMediaWork) -> Void)?
        init(_ plan: MiMoV26MultimodalPlan) { digest = plan.preparationSHA256 }
        func validate(plan: MiMoV26MultimodalPlan) throws {
            calls += 1
            guard !revoked, plan.preparationSHA256 == digest else {
                throw MiMoV26MultimodalError.reservationRejected
            }
        }
        func retainAfterFailedDrain(_ work: MiMoV26FailedMediaWork) {
            guard let quarantine else {
                preconditionFailure("test must install a real failed-work holder")
            }
            quarantine(work)
        }
    }
    static func native() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_MULTIMODAL_NATIVE_TESTS"] == "1" else {
            throw XCTSkip("Requires root's exclusive native lane; no component pass inferred")
        }
    }
    static let limits = MiMoV26MultimodalLimits(
        maximumMedia: 16, maximumVideoFrames: 16, maximumPromptTokens: 120,
        maximumMetadataBytes: 65536, maximumMetadataNodes: 10000, maximumMetadataDepth: 32,
        pixels: .init(
            maximumInputElements: 1_000_000, maximumOutputElements: 1_000_000,
            maximumWorkingBytes: 16 << 20),
        vision: .init(maximumPatches: 1024, maximumAttentionScoreElements: 1_000_000),
        audio: .init(
            maximumClips: 8, maximumChannels: 2, maximumSampleRate: 48000,
            maximumInputSamples: 100000,
            maximumResampledSamples: 100000, maximumResampleCoefficients: 100000,
            maximumMelFrames: 10000,
            maximumSegments: 16, maximumPaddedMelFrames: 100000, maximumWorkingElements: 1_000_000,
            frontendFrameBlockSize: 8, rvqTileFrames: 8),
        audioPatch: .init(
            maximumClips: 8, maximumFrames: 10000, maximumPatches: 4096,
            maximumWorkingElements: 1_000_000))
    static func configuration(_ dtype: String = "float32") throws -> MiMoV26Configuration {
        guard let path = ProcessInfo.processInfo.environment["MIMO_V26_MULTIMODAL_TINY_CONFIG"]
        else {
            throw XCTSkip(
                "Requires existing identity/filesystem-load-fixtures/tiny-bundle/config.json (metadata only)"
            )
        }
        var fields = try JSONDecoder().decode(
            [String: MiMoV26JSONValue].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        fields["attention_projection_layout"] = .string("split")
        fields["moe_router_dtype"] = .string("bfloat16")
        fields["dtype"] = .string(dtype)
        fields["quantization"] = nil
        fields["quantization_config"] = nil
        fields["v_head_dim"] = .number(16)
        fields["swa_v_head_dim"] = .number(16)
        fields["omlx_mimo_mtp"] = nil
        guard case .object(var processor) = fields["processor_config"] else {
            throw MiMoV26MultimodalError.incompatiblePlan
        }
        processor["video_start_token_id"] = .number(9)
        processor["video_end_token_id"] = .number(10)
        fields["processor_config"] = .object(processor)
        return try .init(rawFields: fields)
    }
    static func values(_ name: String, _ shape: [Int], _ dtype: DType) -> MLXArray {
        let count = shape.reduce(1, *)
        let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 997 }
        return MLXArray(
            (0 ..< count).map { i -> Float in
                name.contains("norm") && name.hasSuffix("weight")
                    ? 1 : sin(Float((i * 7 + salt) % 73)) * 0.035
            }, shape
        ).asType(dtype)
    }
    struct Models {
        let target: MiMoV26TextModel, vision: MiMoV26VisionTower, patch: MiMoV26AudioPatchEncoder
        let adapter: MiMoV26CBv2Adapter, processor: MiMoV26MultimodalProcessor
        let tokenizer: MiMoMediaTestTokenizer, generation: MiMoV26MediaGeneration, owner: Owner
        var binding: MiMoV26CBv2Binding {
            .init(adapter: adapter, assistant: nil, stopTokenIDs: [], mediaGeneration: generation)
        }
    }
    static func models(_ dtype: String = "float32") throws -> Models {
        try native()
        let c = try configuration(dtype)
        let d: DType = dtype == "float32" ? .float32 : .bfloat16
        let target = try MiMoV26TextModel(c)
        try target.update(
            parameters: .unflattened(
                target.parameters().flattened().map { name, value in
                    (
                        name,
                        values(
                            name, value.shape,
                            name.hasSuffix("e_score_correction_bias")
                                ? .float32 : name.hasSuffix("mlp.gate.weight") ? .bfloat16 : d)
                    )
                }), verify: .all)
        let vision = try MiMoV26VisionTower(configuration: XCTUnwrap(c.vision))
        try vision.loadNativeWeights(
            try MiMoV26VisionTower.expectedTensorShapes(configuration: XCTUnwrap(c.vision))
                .mapValues { values("vision", $0, d) }, expectedDType: d)
        let patch = try MiMoV26AudioPatchEncoder(configuration: XCTUnwrap(c.audio))
        try patch.loadNativeWeights(
            try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: XCTUnwrap(c.audio))
                .mapValues { values("audio", $0, d) }, expectedDType: d)
        let adapter = try MiMoV26CBv2Adapter(target: target)
        let tokenizer = MiMoMediaTestTokenizer()
        let generation = MiMoV26MediaGeneration()
        let owner = Owner()
        let template = "routing fixture, not Jinja"
        let processor = try MiMoV26MultimodalProcessor(
            configuration: c, tokenizer: tokenizer, chatTemplate: template,
            templateSHA256: MiMoV26MultimodalProcessor.hash(Data(template.utf8)), limits: limits,
            vision: vision,
            audioPatch: patch, adapter: adapter, stopTokens: [], generation: generation,
            retaining: owner, audioCodec: nil)
        return .init(
            target: target, vision: vision, patch: patch, adapter: adapter, processor: processor,
            tokenizer: tokenizer, generation: generation, owner: owner)
    }
    static func image(_ value: Float = 80) -> MiMoV26Pixels.DecodedRGB {
        .init(
            height: 64, width: 64, planarRGB: (0 ..< (3 * 64 * 64)).map { value + Float($0 % 31) })
    }
    static func request(_ content: [MiMoV26MultimodalContent]) -> MiMoV26MultimodalInput {
        .init(messages: [.init(role: .user, content: content)], maximumOutputTokens: 3)
    }
}

final class MiMoV26MultimodalInputTests: XCTestCase {
    func testNativeTimestampFormattingIsFloorAndUnboundedMinutes() throws {
        let cases: [(Float, String)] = [
            (0, "00:00"), (59.999, "00:59"), (60, "01:00"), (3600, "60:00"),
        ]
        for (value, expected) in cases {
            XCTAssertEqual(try MiMoV26MultimodalProfile.timestamp(value), expected)
        }
        let invalid: [Float] = [-1, .nan, .infinity]
        for value in invalid { XCTAssertThrowsError(try MiMoV26MultimodalProfile.timestamp(value)) }
    }
    func testOrderedImageVideoAndAllHistoryRolesSurvive() throws {
        let f = try MiMoMediaFixture.models()
        let image = MiMoMediaFixture.image()
        let video = MiMoV26SilentVideo(frames: [image, image, image], timestamps: [0, 1, 2])
        let input = MiMoV26MultimodalInput(
            messages: [
                .init(role: .system, content: [.text("s")]),
                .init(
                    role: .assistant, content: [.image(image)],
                    templateFields: ["reasoning_content": "kept"]),
                .init(
                    role: .tool, content: [.silentVideo(video)],
                    templateFields: ["tool_call_id": "tool-1"]),
                .init(role: .user, content: [.text("u"), .image(image)]),
            ], maximumOutputTokens: 3)
        let plan = try f.processor.plan(input)
        XCTAssertEqual(plan.spans.map(\.kind), [.image, .video, .video, .image])
        XCTAssertEqual(plan.spans.map(\.length), [9, 9, 9, 9])
        XCTAssertEqual(plan.spans.map(\.mediaIndex), [0, 1, 1, 2])
        XCTAssertEqual(plan.spans.map(\.featureOffset), [0, 0, 9, 0])
        XCTAssertEqual(
            f.tokenizer.messages.map { $0["role"] as! String },
            ["system", "assistant", "tool", "user"])
        XCTAssertEqual(f.tokenizer.messages[1]["reasoning_content"] as? String, "kept")
        XCTAssertEqual(f.tokenizer.messages[2]["tool_call_id"] as? String, "tool-1")
        XCTAssertEqual(plan.promptTokens.filter { $0 == 3 }.count, 18)
    }
    func testAmbiguousMissingExtraAndWrongKindMarkersRefuse() throws {
        let f = try MiMoMediaFixture.models()
        let input = MiMoMediaFixture.request([.image(MiMoMediaFixture.image())])
        for ids in [[11, 12], [11, 4, 3, 5, 12], [11, 4, 2, 5, 4, 2, 5, 12], [11, 2, 12]] {
            f.tokenizer.override = ids
            XCTAssertThrowsError(try f.processor.plan(input))
        }
        f.tokenizer.override = nil
        XCTAssertThrowsError(
            try f.processor.plan(
                MiMoMediaFixture.request([
                    .text("literal <|image_pad|>"), .image(MiMoMediaFixture.image()),
                ])))
    }
    func testMissingCodecAndMalformedVideoRefuseWithoutAuthorization() throws {
        let f = try MiMoMediaFixture.models()
        let image = MiMoMediaFixture.image()
        let pcm = try MiMoV26DecodedPCM(
            samples: Array(repeating: 0, count: 960),
            descriptor: .init(
                sourceIdentity: "test", channels: 1, frameCount: 960, sampleRate: 24000))
        XCTAssertThrowsError(try f.processor.plan(MiMoMediaFixture.request([.audio(pcm)]))) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .missingAudioCodec)
        }
        let invalidStamps: [[Float]] = [[], [0], [1, 0], [0, .nan], [0, 0]]
        for stamps in invalidStamps {
            XCTAssertThrowsError(
                try f.processor.plan(
                    MiMoMediaFixture.request([
                        .silentVideo(.init(frames: [image, image], timestamps: stamps))
                    ])))
        }
    }
    func testMetadataBoundsControlsAndActualPixelsBindPlan() throws {
        let f = try MiMoMediaFixture.models()
        let a = try f.processor.plan(MiMoMediaFixture.request([.image(MiMoMediaFixture.image(50))]))
        let b = try f.processor.plan(MiMoMediaFixture.request([.image(MiMoMediaFixture.image(51))]))
        XCTAssertEqual(a.promptTokens, b.promptTokens)
        XCTAssertNotEqual(a.preparationSHA256, b.preparationSHA256)
        XCTAssertThrowsError(
            try f.processor.plan(
                .init(
                    messages: [.init(role: .user, content: [.image(MiMoMediaFixture.image())])],
                    additionalContext: ["input_audio": "forbidden"], maximumOutputTokens: 3)))
        XCTAssertThrowsError(
            try f.processor.plan(
                .init(
                    messages: [.init(role: .user, content: [.image(MiMoMediaFixture.image())])],
                    additionalContext: ["enable_thinking": "false"], maximumOutputTokens: 3)))
        XCTAssertThrowsError(
            try f.processor.plan(
                .init(
                    messages: [
                        .init(
                            role: .user, content: [.image(MiMoMediaFixture.image())],
                            templateFields: ["audio_url": "unbound"])
                    ], maximumOutputTokens: 3)))
        XCTAssertThrowsError(
            try f.processor.plan(
                .init(
                    messages: [.init(role: .user, content: [.image(MiMoMediaFixture.image())])],
                    maximumOutputTokens: Int.max)))
    }
}
