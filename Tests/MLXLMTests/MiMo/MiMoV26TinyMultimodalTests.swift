import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Decoded-media planning and preparation with tiny float32 MiMo media models
/// (MiMoV26TinyCheckpoint.mediaModels). The routing tokenizer is not a Jinja or
/// BPE oracle. Image and video frames are 64 x 64; with patch 2, merge 2 and a
/// 150-pixel budget each frame resizes to 12 x 12, which gives 9 tokens.
final class MiMoV26TinyMultimodalTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint

    private func limits(
        maximumMedia: Int = 16, maximumVideoFrames: Int = 16, maximumPromptTokens: Int = 120,
        maximumMetadataBytes: Int = 65536, maximumMetadataNodes: Int = 10000,
        maximumMetadataDepth: Int = 32
    ) -> MiMoV26MultimodalLimits {
        let base = MiMoMediaFixture.limits
        return .init(
            maximumMedia: maximumMedia, maximumVideoFrames: maximumVideoFrames,
            maximumPromptTokens: maximumPromptTokens, maximumMetadataBytes: maximumMetadataBytes,
            maximumMetadataNodes: maximumMetadataNodes, maximumMetadataDepth: maximumMetadataDepth,
            pixels: base.pixels, vision: base.vision, audio: base.audio,
            audioPatch: base.audioPatch)
    }
    private func processor(_ f: MiMoMediaFixture.Models, limits: MiMoV26MultimodalLimits) throws
        -> MiMoV26MultimodalProcessor
    {
        let template = "routing fixture, not Jinja"
        return try MiMoV26MultimodalProcessor(
            configuration: f.target.configuration, tokenizer: f.tokenizer, chatTemplate: template,
            templateSHA256: MiMoV26MultimodalProcessor.hash(Data(template.utf8)), limits: limits,
            vision: f.vision, audioPatch: f.patch, adapter: f.adapter, stopTokens: [],
            generation: f.generation, retaining: f.owner, audioCodec: nil)
    }

    // MARK: - Profile

    func testProfileReadsProcessorGeometryAndTheTokenNamespace() throws {
        let tokenizer = MiMoMediaTestTokenizer()
        let profile = try MiMoV26MultimodalProfile(
            configuration: Fixture.configuration(Fixture.baseFields()), tokenizer: tokenizer)
        XCTAssertEqual(profile.settings.patchSize, 2)
        XCTAssertEqual(profile.settings.mergeSize, 2)
        XCTAssertEqual(profile.settings.temporalPatchSize, 2)
        XCTAssertEqual(profile.settings.imageMaxPixels, 150)
        XCTAssertEqual(profile.settings.videoTotalMaxPixels, 1000)
        XCTAssertEqual(profile.tokens.count, 9)
        XCTAssertEqual(profile.tokens["video_start_token_id"], 9)
        XCTAssertEqual(profile.tokens["audio_token_id"], 6)
        func refusal(_ change: (inout [String: Any]) -> Void) throws -> MiMoV26MultimodalError? {
            var fields = Fixture.baseFields()
            var processor = try XCTUnwrap(fields["processor_config"] as? [String: Any])
            change(&processor)
            fields["processor_config"] = processor
            do {
                _ = try MiMoV26MultimodalProfile(
                    configuration: Fixture.configuration(fields), tokenizer: tokenizer)
                return nil
            } catch let error as MiMoV26MultimodalError {
                return error
            }
        }
        XCTAssertEqual(
            try refusal { $0["rope_type"] = "mrope" },
            .unsupportedProfile("native 1D causal profile"))
        XCTAssertEqual(
            try refusal { $0["use_per_grid_t_timestamps"] = true },
            .unsupportedProfile("native 1D causal profile"))
        XCTAssertEqual(
            try refusal { $0["patch_size"] = 4 }, .unsupportedProfile("processor/tower geometry"))
        XCTAssertEqual(
            try refusal { $0["video_start_token_id"] = 11 },
            .unsupportedProfile("tokenizer/processor token namespace"))
        XCTAssertEqual(
            try refusal { _ = $0.removeValue(forKey: "video_end_token_id") },
            .unsupportedProfile("video_end_token_id"))
        XCTAssertEqual(
            try refusal { $0["image_min_pixels"] = 1.5 }, .unsupportedProfile("image_min_pixels"))
    }

    func testRequestClockAcceptsOnlyRealGregorianDays() throws {
        XCTAssertEqual(
            try MiMoV26MediaRequestClock(utcGregorianDay: "2024-02-29").value, "2024-02-29")
        XCTAssertNoThrow(try MiMoV26MediaRequestClock(utcGregorianDay: "2000-02-29"))
        for value in [
            "2023-02-29", "1900-02-29", "2024-13-01", "2024-04-31", "0000-01-01",
            "2024/01/01", "24-01-01", "2024-01-0x",
        ] {
            XCTAssertThrowsError(try MiMoV26MediaRequestClock(utcGregorianDay: value), value) {
                XCTAssertEqual(
                    $0 as? MiMoV26MultimodalError, .invalidInput("invalid request clock snapshot"))
            }
        }
        let f = try Fixture.mediaModels()
        let clock = try MiMoV26MediaRequestClock(utcGregorianDay: "2025-01-31")
        let image = MiMoMediaFixture.image()
        let plan = try f.processor.plan(
            .init(
                messages: [.init(role: .user, content: [.image(image)])],
                additionalContext: ["_darkbloom_request_clock": clock, "enable_thinking": false],
                maximumOutputTokens: 3))
        XCTAssertEqual(plan.spans.map(\.length), [9])
        XCTAssertThrowsError(
            try f.processor.plan(
                .init(
                    messages: [.init(role: .user, content: [.image(image)])],
                    additionalContext: ["_darkbloom_request_clock": "2025-01-31"],
                    maximumOutputTokens: 3))
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError,
                .invalidInput("request clock requires a data-only snapshot"))
        }
    }

    // MARK: - Planning

    func testOrderedImageVideoAndAllHistoryRolesSurvive() throws {
        let f = try Fixture.mediaModels()
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
            f.tokenizer.messages.map { $0["role"] as? String },
            ["system", "assistant", "tool", "user"])
        XCTAssertEqual(f.tokenizer.messages[1]["reasoning_content"] as? String, "kept")
        XCTAssertEqual(f.tokenizer.messages[2]["tool_call_id"] as? String, "tool-1")
        XCTAssertEqual(plan.promptTokens.filter { $0 == 3 }.count, 18)
        // Three sampled frames align to four: two timestamp groups of 9 tokens.
        XCTAssertEqual(plan.visionGeometryByMediaIndex[1]?.gridT, 2)
        XCTAssertEqual(plan.visionGeometryByMediaIndex[1]?.alignedFrames, 4)
        XCTAssertEqual(plan.profile, MiMoV26MultimodalProfile.name)
        XCTAssertNil(plan.audioPlan)
        XCTAssertEqual(plan.loadedOwnerIdentity, f.generation.identity)
        XCTAssertEqual(plan.spans.first?.engineSpan.length, 9)
        // Video wrapper: start marker, then per group a "00:00" stamp (5
        // tokens), vision start, 9 pads, vision end; then the end marker.
        let start = try XCTUnwrap(plan.promptTokens.firstIndex(of: 9))
        XCTAssertEqual(
            Array(plan.promptTokens[(start + 1) ..< (start + 6)]),
            f.tokenizer.encode(text: "00:00", addSpecialTokens: true))
    }

    func testAmbiguousMissingExtraAndWrongKindMarkersRefuse() throws {
        let f = try Fixture.mediaModels()
        let input = MiMoMediaFixture.request([.image(MiMoMediaFixture.image())])
        for ids in [[11, 12], [11, 4, 3, 5, 12], [11, 4, 2, 5, 4, 2, 5, 12], [11, 2, 12]] {
            f.tokenizer.override = ids
            XCTAssertThrowsError(try f.processor.plan(input), "\(ids)")
        }
        f.tokenizer.override = nil
        XCTAssertThrowsError(
            try f.processor.plan(
                MiMoMediaFixture.request([
                    .text("literal <|image_pad|>"), .image(MiMoMediaFixture.image()),
                ]))
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError,
                .invalidInput("metadata limit or ambiguous literal media marker"))
        }
        let textOnly = MiMoMediaFixture.request([.text("only text")])
        XCTAssertThrowsError(try f.processor.plan(textOnly)) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError,
                .invalidInput("use the native text processor for text-only input"))
        }
    }

    func testMissingCodecAndMalformedMediaRefuseBeforeTokenization() throws {
        let f = try Fixture.mediaModels()
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
        let small = MiMoV26Pixels.DecodedRGB(
            height: 32, width: 32, planarRGB: Array(repeating: 1, count: 3 * 32 * 32))
        XCTAssertThrowsError(
            try f.processor.plan(
                MiMoMediaFixture.request([
                    .silentVideo(.init(frames: [image, small], timestamps: [0, 1]))
                ]))
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError,
                .invalidInput("video geometry or strictly increasing FP32 timestamps"))
        }
        for frame in [
            MiMoV26Pixels.DecodedRGB(height: 2, width: 2, planarRGB: [1, 2, 3]),
            .init(height: 2, width: 2, planarRGB: Array(repeating: 256, count: 12)),
            .init(height: 2, width: 2, planarRGB: Array(repeating: .nan, count: 12)),
            .init(height: 0, width: 2, planarRGB: []),
        ] {
            XCTAssertThrowsError(try f.processor.plan(MiMoMediaFixture.request([.image(frame)]))) {
                XCTAssertEqual($0 as? MiMoV26MultimodalError, .invalidInput("decoded planar RGB"))
            }
        }
    }

    func testMetadataControlsAndRequestLimitsBindThePlan() throws {
        let f = try Fixture.mediaModels()
        let a = try f.processor.plan(MiMoMediaFixture.request([.image(MiMoMediaFixture.image(50))]))
        let b = try f.processor.plan(MiMoMediaFixture.request([.image(MiMoMediaFixture.image(51))]))
        XCTAssertEqual(a.promptTokens, b.promptTokens)
        XCTAssertNotEqual(a.preparationSHA256, b.preparationSHA256)
        let image = MiMoMediaFixture.image()
        func plan(
            _ processor: MiMoV26MultimodalProcessor? = nil,
            content: [MiMoV26MultimodalContent]? = nil,
            tools: [ToolSpec]? = nil, context: [String: any Sendable]? = nil,
            fields: [String: any Sendable] = [:], output: Int = 3
        ) throws -> MiMoV26MultimodalPlan {
            try (processor ?? f.processor).plan(
                .init(
                    messages: [
                        .init(
                            role: .user, content: content ?? [.image(image)], templateFields: fields
                        )
                    ], tools: tools, additionalContext: context, maximumOutputTokens: output))
        }
        XCTAssertThrowsError(try plan(context: ["input_audio": "forbidden"]))
        XCTAssertThrowsError(try plan(context: ["enable_thinking": "false"])) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError, .invalidInput("native Boolean render controls only"))
        }
        XCTAssertNoThrow(
            try plan(context: ["enable_thinking": true, "add_generation_prompt": true]))
        XCTAssertThrowsError(try plan(fields: ["audio_url": "unbound"])) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError,
                .invalidInput("structured role/content override or unbound media field"))
        }
        XCTAssertNoThrow(
            try plan(fields: ["name": "n", "count": 2, "flag": true, "ratio": 0.5, "list": ["x"]]))
        XCTAssertThrowsError(try plan(fields: ["ratio": Double.infinity])) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError, .invalidInput("non-JSON native template field"))
        }
        XCTAssertThrowsError(try plan(fields: ["date": Date()])) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError, .invalidInput("non-JSON native template field"))
        }
        let tool: ToolSpec = [
            "type": "function",
            "function": ["name": "lookup", "description": "uses <|video_pad|>"]
                as [String: any Sendable],
        ]
        XCTAssertThrowsError(try plan(tools: [tool])) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError,
                .invalidInput("metadata limit or ambiguous literal media marker"))
        }
        XCTAssertThrowsError(try plan(output: Int.max))
        XCTAssertThrowsError(try plan(output: 0)) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError, .invalidInput("empty request or invalid limits"))
        }
        XCTAssertThrowsError(try plan(output: 256)) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .limit("output context"))
        }
        let oneMedia = try processor(f, limits: limits(maximumMedia: 1))
        XCTAssertThrowsError(try plan(oneMedia, content: [.image(image), .image(image)])) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .limit("media count"))
        }
        let shortPrompt = try processor(f, limits: limits(maximumPromptTokens: 8))
        XCTAssertThrowsError(try plan(shortPrompt)) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .limit("media span"))
        }
        let fewFrames = try processor(f, limits: limits(maximumVideoFrames: 2))
        XCTAssertThrowsError(
            try plan(
                fewFrames,
                content: [
                    .silentVideo(.init(frames: [image, image, image], timestamps: [0, 1, 2]))
                ])
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError, .invalidInput("complete silent sampled video"))
        }
        let smallMetadata = try processor(f, limits: limits(maximumMetadataBytes: 4))
        XCTAssertThrowsError(try plan(smallMetadata, fields: ["note": "too long"]))
        let fewNodes = try processor(f, limits: limits(maximumMetadataNodes: 2))
        XCTAssertThrowsError(try plan(fewNodes, fields: ["a": 1, "b": 2])) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .limit("metadata traversal"))
        }
        let unsafeDepth = try processor(f, limits: limits(maximumMetadataDepth: 129))
        XCTAssertThrowsError(try plan(unsafeDepth)) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .limit("native metadata safety bounds"))
        }
        let invalid = try processor(f, limits: limits(maximumMedia: 0))
        XCTAssertThrowsError(try plan(invalid)) {
            XCTAssertEqual(
                $0 as? MiMoV26MultimodalError, .invalidInput("empty request or invalid limits"))
        }
    }

    // MARK: - Preparation

    func testPreparedVisionFeaturesArePairedInPromptOrderIncludingOddVideo() throws {
        let f = try Fixture.mediaModels()
        let a = MiMoMediaFixture.image(40)
        let b = MiMoMediaFixture.image(90)
        let video = MiMoV26SilentVideo(frames: [a, b, a], timestamps: [0, 1, 2])
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(b), .text("between"), .silentVideo(video), .image(a)]))
        let value = try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        let request = try value.makeRequest(binding: f.binding, id: .init(1))
        XCTAssertFalse(request.prefixCacheEnabled)
        XCTAssertEqual(request.maxTokens, 3)
        XCTAssertEqual(request.multimodal?.attention, .causal)
        let arrays = try XCTUnwrap(request.multimodal).embeddings()
        XCTAssertEqual(arrays.count, 4)
        var expected: [MLXArray] = []
        for content in [MiMoV26MultimodalContent.image(b), .silentVideo(video), .image(a)] {
            let pixels: MiMoV26Pixels.Prepared
            switch content {
            case .image(let frame):
                pixels = try MiMoV26Pixels.image(
                    frame, settings: f.processor.profile.settings,
                    limits: MiMoMediaFixture.limits.pixels)
            case .silentVideo(let clip):
                pixels = try MiMoV26Pixels.video(
                    frames: clip.frames, sampledFrameCount: clip.frames.count,
                    settings: f.processor.profile.settings, limits: MiMoMediaFixture.limits.pixels)
            default: throw MiMoV26MultimodalError.incompatiblePlan
            }
            let g = pixels.geometry
            let feature = try f.vision.forward(
                patches: MLXArray(pixels.patchValues, [g.patchCount, g.patchVectorSize]),
                grids: [.init(temporal: g.gridT, height: g.gridH, width: g.gridW)],
                limits: MiMoMediaFixture.limits.vision)
            let per = g.gridH * g.gridW / 4
            for t in 0 ..< g.gridT { expected.append(feature[(t * per) ..< ((t + 1) * per), 0...]) }
        }
        XCTAssertEqual(expected.count, 4)
        for (actual, reference) in zip(arrays, expected) {
            eval(actual, reference)
            XCTAssertEqual(actual.shape, [9, 64])
            XCTAssertEqual(actual.shape, reference.shape)
            // Float32. The bounded tower evaluates block by block, so the
            // order of sums can differ from one lazy graph: 1e-4.
            XCTAssertLessThanOrEqual(abs(actual - reference).max().item(Float.self), 1e-4)
        }
        XCTAssertThrowsError(try XCTUnwrap(request.multimodal).embeddings())
    }

    func testForeignPlanBindingAndInvalidatedGenerationRefuse() throws {
        let a = try Fixture.mediaModels()
        let b = try Fixture.mediaModels()
        let plan = try a.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        var authorizations = 0
        XCTAssertThrowsError(
            try b.processor.prepare(
                plan,
                authorize: {
                    authorizations += 1
                    return MiMoMediaFixture.Reservation($0)
                })
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatiblePlan) }
        XCTAssertEqual(authorizations, 0)
        let value = try a.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        XCTAssertThrowsError(try value.makeRequest(binding: b.binding, id: .init(2))) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatibleOwner)
        }
        let request = try value.makeRequest(binding: a.binding, id: .init(3))
        XCTAssertNotNil(request.multimodal)
        XCTAssertThrowsError(try value.makeRequest(binding: a.binding, id: .init(4))) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatiblePlan)
        }
        let first = try a.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        let second = try a.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        let queued = try second.makeRequest(binding: a.binding, id: .init(5))
        a.generation.invalidate()
        var calls = 0
        XCTAssertThrowsError(
            try a.processor.prepare(
                plan,
                authorize: {
                    calls += 1
                    return MiMoMediaFixture.Reservation($0)
                })
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .invalidatedOwner) }
        XCTAssertEqual(calls, 0)
        XCTAssertThrowsError(try first.makeRequest(binding: a.binding, id: .init(6)))
        XCTAssertThrowsError(try XCTUnwrap(queued.multimodal).embeddings())
    }

    func testReservationRejectionAndCancellationDrainThenReleaseTheOwner() throws {
        let f = try Fixture.mediaModels()
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        var calls = 0
        XCTAssertThrowsError(
            try f.processor.prepare(
                plan,
                authorize: {
                    calls += 1
                    return MiMoMediaFixture.Reservation($0)
                }, isCancelled: { true })
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .cancelled) }
        XCTAssertEqual(calls, 0)
        weak var observed: MiMoMediaFixture.Reservation?
        var cancelled = false
        XCTAssertThrowsError(
            try f.processor.prepare(
                plan,
                authorize: { value in
                    let reservation = MiMoMediaFixture.Reservation(value)
                    observed = reservation
                    cancelled = true
                    return reservation
                }, isCancelled: { cancelled })
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .cancelled) }
        XCTAssertNil(observed, "a failed preparation drains and releases the only work owner")
        XCTAssertThrowsError(
            try f.processor.prepare(
                plan,
                authorize: { value in
                    let reservation = MiMoMediaFixture.Reservation(value)
                    reservation.revoked = true
                    return reservation
                })
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .reservationRejected) }
        weak var kept: MiMoMediaFixture.Reservation?
        var prepared: MiMoV26PreparedMultimodal? = try f.processor.prepare(
            plan,
            authorize: { value in
                let reservation = MiMoMediaFixture.Reservation(value)
                kept = reservation
                return reservation
            })
        var request: CBv2Request? = try prepared!.makeRequest(binding: f.binding, id: .init(7))
        prepared = nil
        XCTAssertNotNil(kept)
        _ = try request!.multimodal!.embeddings()
        XCTAssertNotNil(kept)
        request = nil
        XCTAssertNil(kept)
    }

    func testUnloadedOrWrongPrecisionTowerIsNotSilentlyAccepted() throws {
        let f = try Fixture.mediaModels(dtype: "bfloat16")
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        XCTAssertEqual(plan.logicalFeatureBytes, 9 * 64 * 2)
        let good = try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        let features = try XCTUnwrap(good.makeRequest(binding: f.binding, id: .init(8)).multimodal)
            .embeddings()
        XCTAssertEqual(features.first?.dtype, .bfloat16)
        try f.vision.update(parameters: f.vision.parameters(), verify: .all)
        XCTAssertThrowsError(
            try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) }))
        let source = try MiMoV26VisionTower.expectedTensorShapes(
            configuration: f.vision.configuration
        ).mapValues { MiMoMediaFixture.values("wrong-type", $0, .float32) }
        try f.vision.loadNativeWeights(source, expectedDType: .float32)
        XCTAssertThrowsError(
            try f.processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .invalidFeatures) }
    }

    func testFailedDrainKeepsRootsSourceAndLeaseUntilRecovery() throws {
        let f = try Fixture.mediaModels()
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        enum Injected: Error { case failure }
        var quarantined: MiMoV26FailedMediaWork?
        var lease: MiMoMediaFixture.Reservation? = .init(plan)
        weak var weakLease = lease
        lease!.quarantine = { quarantined = $0 }
        var source: NSObject? = NSObject()
        weak var weakSource = source
        var array: MLXArray? = MLXArray([Float(1), 2])
        weak var weakArray = array
        var work: MiMoV26FailedMediaWork? = .init(
            plan: plan, owner: source!, codec: nil, reservation: lease!)
        XCTAssertEqual(work?.preparationSHA256, plan.preparationSHA256)
        XCTAssertThrowsError(try work!.recoverAndReleaseAfterDrain()) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatiblePlan)
        }
        work!.track([array!])
        array = nil
        source = nil
        lease = nil
        XCTAssertThrowsError(
            try work!.drain(generation: f.generation, synchronize: { throw Injected.failure })
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .drainFailed) }
        work = nil
        XCTAssertNotNil(weakArray)
        XCTAssertNotNil(weakSource)
        XCTAssertNotNil(weakLease)
        XCTAssertEqual(quarantined?.retainedRootCount, 1)
        XCTAssertEqual(quarantined?.failedDrain, true)
        XCTAssertThrowsError(try f.generation.validate())
        XCTAssertThrowsError(
            try quarantined!.verifyRecoveryDrain(synchronize: { throw Injected.failure }))
        XCTAssertNotNil(weakArray)
        try quarantined!.recoverAndReleaseAfterDrain()
        XCTAssertNil(weakArray)
        XCTAssertNil(weakSource)
        XCTAssertNil(weakLease)
        XCTAssertThrowsError(try quarantined!.recoverAndReleaseAfterDrain())
        quarantined = nil
    }

    func testManagedCommitmentPricesPixelsPatchesFeaturesAndOneFrameGraph() throws {
        let f = try Fixture.mediaModels()
        let image = MiMoMediaFixture.image()
        let video = MiMoV26SilentVideo(frames: [image, image, image], timestamps: [0, 1, 2])
        let plan = try f.processor.plan(
            MiMoMediaFixture.request([.image(image), .silentVideo(video)]))
        // Decoded RGB: four 64 x 64 frames. Patches: 36 for the image, 72 for
        // the video (two temporal groups), 24 values each. Features: 27 tokens.
        XCTAssertEqual(plan.decodedElements, 4 * 3 * 64 * 64)
        XCTAssertEqual(plan.patchElements, (36 + 72) * 24)
        XCTAssertEqual(plan.featureElements, 27 * 64)
        XCTAssertEqual(plan.logicalFeatureBytes, 27 * 64 * 4)
        let vision = try XCTUnwrap(f.target.configuration.vision)
        let imageGeometry = try XCTUnwrap(plan.visionGeometryByMediaIndex[0])
        let videoGeometry = try XCTUnwrap(plan.visionGeometryByMediaIndex[1])
        let pixels = max(
            try MiMoV26Pixels.workingByteCount(
                inputElements: 3 * 64 * 64, frameCount: 1, plan: imageGeometry),
            try MiMoV26Pixels.workingByteCount(
                inputElements: 3 * 3 * 64 * 64, frameCount: 3, plan: videoGeometry))
        XCTAssertEqual(try MiMoV26MultimodalProcessor.managedPixelWorkingBytes(plan), pixels)
        let graph = max(
            try MiMoV26VisionWorkingSet.frameBytes(imageGeometry, configuration: vision),
            try MiMoV26VisionWorkingSet.frameBytes(videoGeometry, configuration: vision))
        XCTAssertEqual(
            try f.processor.managedVisualCommitmentBytes(plan),
            pixels + 4 * 3 * 64 * 64 * 4 + (36 + 72) * 24 * 8 + 27 * 64 * 16 + graph)
        XCTAssertThrowsError(try f.processor.managedAudioCommitmentBytes(plan)) {
            XCTAssertEqual($0 as? MiMoV26MultimodalError, .missingAudioCodec)
        }
    }
}
