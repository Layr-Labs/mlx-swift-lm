import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Controllable tokenizer seam; not an implementation of the official Jinja template.
struct GLM5NextTestTokenizer: Tokenizer {
    let specials: [String: Int] = [
        "<|image|>": 154854, "<|video|>": 154855,
        "<|begin_of_image|>": 154830, "<|end_of_image|>": 154831,
        "<|begin_of_video|>": 154832, "<|end_of_video|>": 154833,
    ]
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func convertTokenToId(_ token: String) -> Int? { specials[token] }
    func convertIdToToken(_ id: Int) -> String? { specials.first { $0.value == id }?.key }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens = [1]
        for message in messages {
            guard let content = message["content"] as? [[String: any Sendable]] else { continue }
            for part in content {
                switch part["type"] as? String {
                case "image": tokens += [154830, 154854, 154831]
                case "video": tokens += [154832, 154855, 154833]
                default:
                    tokens += encode(text: part["text"] as? String ?? "", addSpecialTokens: false)
                }
            }
        }
        return tokens
    }
}

extension UnitTests {
    @Suite struct GLM5NextProcessorTests {
        @Test func pinnedSmartResizeAndPadding() throws {
            // Computed from smart_resize in the pinned f0705 Transformers file.
            let small = try GLM5NextMediaGeometry.resizePlan(width: 31, height: 23)
            #expect(small.width == 140 && small.height == 112)
            #expect(small.contentWidth == 140 && small.contentHeight == 103)
            #expect(try small.grid.featureCount() == 20)
            let normal = try GLM5NextMediaGeometry.resizePlan(width: 448, height: 448)
            #expect(normal.width == 448 && normal.height == 448)
            #expect(try normal.grid.featureCount() == 256)
            let large = try GLM5NextMediaGeometry.resizePlan(width: 16000, height: 8000)
            #expect(large.width == 3528 && large.height == 1764)
            #expect(try large.grid.featureCount() == 7938)
            let video = try GLM5NextMediaGeometry.resizePlan(
                width: 1280, height: 720,
                frames: 6, minimumTokens: 16, maximumTokens: 20)
            #expect(video.width == 84 && video.height == 56)
            #expect(try video.grid.featureCount() == 18)
        }

        @Test func exactFrameSamplingAndTemporalPadding() throws {
            #expect(
                try GLM5NextMediaGeometry.sampleFrameIndices(
                    totalFrames: 3, sourceFPS: 2, duration: 1.5) == [0, 1])
            #expect(
                try GLM5NextMediaGeometry.sampleFrameIndices(
                    totalFrames: 10, sourceFPS: 5, duration: 2) == [0, 3, 5, 8])
            #expect(
                try GLM5NextMediaGeometry.sampleFrameIndices(
                    totalFrames: 3, sourceFPS: 1, duration: 3) == [0, 1, 2, 2])
            #expect(
                try GLM5NextMediaGeometry.sampleFrameIndices(
                    totalFrames: 1, sourceFPS: 2, duration: 0.5) == [0, 0])
            #expect(
                try GLM5NextMediaGeometry.sampleFrameIndices(
                    totalFrames: 31, sourceFPS: 30, duration: 31.0 / 30) == [0, 15])
        }

        @Test func samplingPolicyPreservesResizeInputCount() throws {
            let frame = try GLM5NextRGBImage(
                width: 47, height: 29,
                pixels: Array(repeating: 0, count: 47 * 29 * 3))
            let video = try GLM5NextDecodedVideo(
                frames: [frame, frame, frame],
                framesPerSecond: 2, duration: 1.5)
            let enabledConfig = try GLM5NextProcessorConfiguration(videoMaximumTokens: 128)
            #expect(enabledConfig.videoSamplingPolicy == .official)
            let enabled = try GLM5NextProcessor(
                configuration: enabledConfig,
                tokenizer: GLM5NextTestTokenizer()
            ).videoPreparationPlan(video)
            #expect(enabled.frameIndices == [0, 1])
            #expect(enabled.resize.grid == (try .init(temporal: 1, height: 8, width: 12)))
            let disabled = try GLM5NextProcessor(
                configuration: .init(
                    videoMaximumTokens: 128,
                    videoSamplingPolicy: .disabled), tokenizer: GLM5NextTestTokenizer()
            ).videoPreparationPlan(video)
            #expect(disabled.frameIndices == [0, 1, 2])
            #expect(disabled.resize.width == 140 && disabled.resize.height == 84)
            #expect(disabled.resize.contentWidth == 136 && disabled.resize.contentHeight == 84)
            #expect(disabled.resize.grid == (try .init(temporal: 2, height: 6, width: 10)))
            #expect(disabled.preparedBytes == 120 * 1176 * MemoryLayout<Float>.size)
            let four = try GLM5NextMediaGeometry.resizePlan(
                width: 47, height: 29,
                frames: 4, minimumTokens: 16, maximumTokens: 128)
            #expect(four.grid == (try .init(temporal: 2, height: 6, width: 8)))
            #expect(four.contentWidth == 112 && four.contentHeight == 69)
        }

        @Test func retainedOfficialSampleMatchesTheCompleteSource() throws {
            let frames = try (0 ..< 10).map { value in
                try GLM5NextRGBImage(
                    width: 47, height: 29,
                    pixels: Array(repeating: UInt8(value * 20), count: 47 * 29 * 3))
            }
            let sampled = try GLM5NextMediaGeometry.sampleFrameIndices(
                totalFrames: 10, sourceFPS: 5, duration: 2)
            #expect(sampled == [0, 3, 5, 8])
            func video(_ indices: [Int]) throws -> GLM5NextDecodedVideo {
                try GLM5NextDecodedVideo(
                    frames: indices.map { frames[$0] }, framesPerSecond: 5, duration: 2,
                    sourceFrameCount: 10, sourceIndices: indices)
            }
            let processor = try GLM5NextProcessor(
                configuration: .init(videoMaximumTokens: 128), tokenizer: GLM5NextTestTokenizer())
            let complete = try processor.videoPreparationPlan(
                GLM5NextDecodedVideo(frames: frames, framesPerSecond: 5, duration: 2))
            let retained = try processor.videoPreparationPlan(video(sampled))
            #expect(complete.frameIndices == sampled && complete.framePositions == sampled)
            #expect(retained.frameIndices == sampled && retained.framePositions == [0, 1, 2, 3])
            #expect(retained.resize == complete.resize)
            #expect(retained.preparedBytes == complete.preparedBytes)
            // Extra retained frames are ignored; a missing sampled frame is refused.
            let extra = try processor.videoPreparationPlan(video([0, 1, 3, 5, 8]))
            #expect(extra.framePositions == [0, 2, 3, 4])
            let missing = GLM5NextVisionError.invalidInput("sampled video frame was not retained")
            #expect(throws: missing) {
                try processor.videoPreparationPlan(video([0, 3, 5]))
            }
            let disabled = try GLM5NextProcessor(
                configuration: .init(videoMaximumTokens: 128, videoSamplingPolicy: .disabled),
                tokenizer: GLM5NextTestTokenizer())
            #expect(throws: GLM5NextVisionError.self) {
                try disabled.videoPreparationPlan(video(sampled))
            }
            // Source indices must ascend within the source.
            for indices in [[3, 0], [9, 10]] {
                #expect(throws: GLM5NextVisionError.self) {
                    try GLM5NextDecodedVideo(
                        frames: [frames[0], frames[1]], framesPerSecond: 5, duration: 2,
                        sourceFrameCount: 10, sourceIndices: indices)
                }
            }
        }

        /// Past the decoded-frame cap, the official sample keeps at most the
        /// frames the cap holds at the clip's frame size (the last one as its
        /// RGBA render), spread over the whole clip as the sampler spreads
        /// `maximumFrames`, an odd count padded as usual; without
        /// `fitsVideoSampleToDecodedBytes` the clip is refused. A sample whose
        /// distinct frames fit is the official one either way.
        @Test func decodedByteCapSamplesDownAcrossTheClip() throws {
            let frameBytes = 47 * 29 * 3
            let frames = try (0 ..< 30).map { value in
                try GLM5NextRGBImage(
                    width: 47, height: 29, pixels: Array(repeating: UInt8(value), count: frameBytes)
                )
            }
            func video(_ indices: [Int]) throws -> GLM5NextDecodedVideo {
                try GLM5NextDecodedVideo(
                    frames: indices.map { frames[$0] }, framesPerSecond: 5, duration: 6,
                    sourceFrameCount: 30, sourceIndices: indices)
            }
            // `cap` frames: all but the last kept as RGB8, the last as its RGBA render.
            func processor(frames cap: Int, fits: Bool) throws -> GLM5NextProcessor {
                try GLM5NextProcessor(
                    configuration: .init(
                        videoMaximumTokens: 128,
                        maximumDecodedBytes: (cap - 1) * frameBytes + 47 * 29 * 4,
                        fitsVideoSampleToDecodedBytes: fits),
                    tokenizer: GLM5NextTestTokenizer())
            }
            let official = try GLM5NextMediaGeometry.sampleFrameIndices(
                totalFrames: 30, sourceFPS: 5, duration: 6)
            #expect(official == [0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28])
            #expect(throws: GLM5NextVisionError.budgetExceeded("decoded source video")) {
                try processor(frames: 10, fits: false).videoPreparationPlan(video(official))
            }
            for (cap, spread) in [
                (10, [0, 3, 6, 9, 12, 16, 19, 22, 25, 29]),
                (11, [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29]),
            ] {
                let fitted = try processor(frames: cap, fits: true)
                #expect(
                    fitted.configuration.maximumSampledVideoFrames(frameBytes: frameBytes) == cap)
                let padded = spread.count.isMultiple(of: 2) ? spread : spread + [29]
                #expect(
                    try fitted.configuration.sampledVideoIndices(
                        totalFrames: 30, sourceFPS: 5, duration: 6, frameBytes: frameBytes)
                        == padded)
                let plan = try fitted.videoPreparationPlan(video(spread))
                #expect(plan.frameIndices == padded)
                #expect(plan.framePositions == padded.map { spread.firstIndex(of: $0)! })
                #expect(plan.resize.grid.temporal == (cap + 1) / 2)
                #expect(
                    throws: GLM5NextVisionError.invalidInput("sampled video frame was not retained")
                ) {
                    try fitted.videoPreparationPlan(video(official))
                }
            }
            for fits in [false, true] {
                let plan = try processor(frames: 12, fits: fits).videoPreparationPlan(
                    video(official))
                #expect(plan.frameIndices == official)
            }
            // An odd official sample (11 distinct frames, the last repeated)
            // that the cap holds exactly is kept as it is.
            let odd = try GLM5NextMediaGeometry.sampleFrameIndices(
                totalFrames: 28, sourceFPS: 5, duration: 5.5)
            #expect(Set(odd).count == 11 && odd.count == 12)
            for fits in [false, true] {
                #expect(
                    try processor(frames: 11, fits: fits).configuration.sampledVideoIndices(
                        totalFrames: 28, sourceFPS: 5, duration: 5.5, frameBytes: frameBytes) == odd
                )
            }
            let unbounded = try GLM5NextProcessorConfiguration()
            #expect(unbounded.maximumSampledVideoFrames(frameBytes: frameBytes) == 2048)
            let tiny = try GLM5NextProcessorConfiguration(
                maximumDecodedBytes: frameBytes, fitsVideoSampleToDecodedBytes: true)
            #expect(tiny.maximumSampledVideoFrames(frameBytes: frameBytes) == 1)
            // 512 MiB at 720p and 1080p: the largest samples the decoder kept before.
            let product = try GLM5NextProcessorConfiguration(fitsVideoSampleToDecodedBytes: true)
            #expect(product.maximumSampledVideoFrames(frameBytes: 1280 * 720 * 3) == 193)
            #expect(product.maximumSampledVideoFrames(frameBytes: 1920 * 1080 * 3) == 85)
        }

        @Test func disabledSamplingRetainsLimitsAndQuotesAllPackedFrames() throws {
            let frame = try GLM5NextRGBImage(
                width: 4, height: 4,
                pixels: Array(repeating: 0, count: 48))
            let video = try GLM5NextDecodedVideo(
                frames: Array(repeating: frame, count: 5),
                framesPerSecond: 2, duration: 2.5)
            let exactBytes = 3 * 4 * 24 * MemoryLayout<Float>.size
            func processor(bytes: Int = 4096, decodedBytes: Int = 4096) throws -> GLM5NextProcessor
            {
                try GLM5NextProcessor(
                    configuration: .init(
                        patchSize: 2,
                        imageMinimumTokens: 1, videoMinimumTokens: 1, videoMaximumTokens: 128,
                        maximumDecodedBytes: decodedBytes, maximumPreparedBytes: bytes,
                        videoSamplingPolicy: .disabled), tokenizer: GLM5NextTestTokenizer())
            }
            let plan = try processor(bytes: exactBytes).videoPreparationPlan(video)
            #expect(plan.frameIndices == [0, 1, 2, 3, 4])
            #expect(plan.resize.grid == (try .init(temporal: 3, height: 2, width: 2)))
            #expect(plan.preparedBytes == exactBytes)
            #expect(throws: GLM5NextVisionError.self) {
                try processor(bytes: exactBytes - 1).videoPreparationPlan(video)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try processor(decodedBytes: 239).videoPreparationPlan(video)
            }
            let tooFewTokens = try GLM5NextProcessor(
                configuration: .init(
                    patchSize: 2,
                    videoMinimumTokens: 1, videoMaximumTokens: 1, videoSamplingPolicy: .disabled),
                tokenizer: GLM5NextTestTokenizer())
            #expect(throws: GLM5NextVisionError.self) {
                try tooFewTokens.videoPreparationPlan(video)
            }
            // smart_resize's aligned count is four, but patchify packs six.
            // Two tokens would fit the former; the actual three must be refused.
            let insufficientPackedTokens = try GLM5NextProcessor(
                configuration: .init(
                    patchSize: 2,
                    videoMinimumTokens: 1, videoMaximumTokens: 2, videoSamplingPolicy: .disabled),
                tokenizer: GLM5NextTestTokenizer())
            #expect(throws: GLM5NextVisionError.budgetExceeded("packed video tokens")) {
                try insufficientPackedTokens.videoPreparationPlan(video)
            }
            let many = try GLM5NextDecodedVideo(
                frames: Array(repeating: frame, count: 2049),
                framesPerSecond: 2, duration: 1024.5)
            #expect(throws: GLM5NextVisionError.self) { try processor().videoPreparationPlan(many) }
            let tooShort = try GLM5NextDecodedVideo(
                frames: [frame], framesPerSecond: 30, duration: 0.01)
            #expect(throws: GLM5NextVisionError.self) {
                try processor().videoPreparationPlan(tooShort)
            }
        }

        @Test func invalidGeometryAndBudgetsFailClosed() throws {
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextMediaGeometry.resizePlan(width: 0, height: 10)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextMediaGeometry.resizePlan(width: Int.max, height: 2)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextMediaGeometry.resizePlan(
                    width: 10, height: 10, minimumTokens: 32, maximumTokens: 16)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextMediaGeometry.resizePlan(
                    width: 10, height: 10, frames: 100, minimumTokens: 1, maximumTokens: 1)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextMediaGeometry.sampleFrameIndices(totalFrames: 0, sourceFPS: 1)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextMediaGeometry.sampleFrameIndices(totalFrames: 3, sourceFPS: .nan)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextProcessorConfiguration(imageMaximumTokens: 8001)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextVisionGrid(temporal: 1, height: 3, width: 2).featureCount()
            }
        }

        @Test func videoExpansionAndHeterogeneousLeftPadding() throws {
            let processor = try GLM5NextProcessor(
                configuration: .init(), tokenizer: GLM5NextTestTokenizer())
            let image = try GLM5NextVisionGrid(temporal: 1, height: 2, width: 2)
            let video = try GLM5NextVisionGrid(temporal: 2, height: 2, width: 2)
            let expanded = try processor.expandPrompt(
                [154830, 154854, 154831, 42, 154832, 154855, 154833],
                images: [image], videos: [(video, [0, 1])])
            #expect(expanded.placements.count == 3)
            #expect(expanded.placements[0].tokenRange == 1 ..< 2)
            #expect(expanded.placements[1].tokenRange == 6 ..< 7)
            #expect(expanded.placements[1].featureRange == 0 ..< 1)
            #expect(expanded.placements[2].featureRange == 1 ..< 2)
            #expect(expanded.tokens.filter { $0 == 154854 }.count == 3)
            let short = try processor.expandPrompt([154854], images: [image], videos: [])
            let batch = try GLM5NextProcessor.leftPad(rows: [expanded, short])
            #expect(batch.tokens[0].count == batch.tokens[1].count)
            #expect(batch.mask[1].filter { $0 }.count == 1)
            #expect(batch.placements[1][0].tokenRange.lowerBound == expanded.tokens.count - 1)
            #expect(batch.placements[1][0].featureRange == 0 ..< 1)
        }

        @Test func malformedPromptMediaCountsRejected() throws {
            let processor = try GLM5NextProcessor(
                configuration: .init(), tokenizer: GLM5NextTestTokenizer())
            let image = try GLM5NextVisionGrid(temporal: 1, height: 2, width: 2)
            #expect(throws: GLM5NextVisionError.self) {
                try processor.expandPrompt([], images: [image], videos: [])
            }
            #expect(throws: GLM5NextVisionError.self) {
                try processor.expandPrompt([154854], images: [], videos: [])
            }
            #expect(throws: GLM5NextVisionError.self) {
                try processor.expandPrompt([154855], images: [], videos: [(image, [])])
            }
            #expect(throws: GLM5NextVisionError.self) { try GLM5NextProcessor.leftPad(rows: []) }
        }
    }
}
