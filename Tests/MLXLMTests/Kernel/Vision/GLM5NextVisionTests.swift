@preconcurrency import AVFoundation
import CoreImage
import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXVLM

extension KernelTests {
    @Suite struct GLM5NextVisionTests {
        private static func configuration(depth: Int = 1, intermediate: Int = 5) throws
            -> GLM5NextVisionConfiguration
        {
            let values: [String: Any] = [
                "model_type": "glm5_next_vision", "depth": depth, "hidden_size": 4,
                "intermediate_size": 6, "num_heads": 1, "image_size": 4, "patch_size": 2,
                "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
                "out_hidden_size": 4, "projection_intermediate_size": intermediate,
                "hidden_act": "silu", "swiglu_limit": 0.15, "rms_norm_eps": 1e-5,
                "attention_bias": true, "attention_dropout": 0.0, "initializer_range": 0.02,
            ]
            return try JSONDecoder().decode(
                GLM5NextVisionConfiguration.self,
                from: JSONSerialization.data(withJSONObject: values))
        }

        private static func weights(_ configuration: GLM5NextVisionConfiguration) -> [String:
            MLXArray]
        {
            var result: [String: MLXArray] = [:]
            for (name, shape) in GLM5NextVisionTower.expectedTensorShapes(
                configuration: configuration)
            {
                let count = shape.reduce(1, *)
                let seed = Int(name.utf8.reduce(0) { $0 + Int($1) } % 7) - 3
                let values: [Float] = (0 ..< count).map { i in
                    if name.contains("norm") && name.hasSuffix(".weight") {
                        return 1 + Float(i % 3 - 1) * 0.02
                    }
                    return Float(i % 11 - 5) * 0.02 + Float(seed) * 0.01
                }
                result["model.visual." + name] = MLXArray(values).reshaped(shape)
            }
            return result
        }

        @Test func completeTinyTowerSourceEquationVector() throws {
            let config = try Self.configuration()
            let tower = try GLM5NextVisionTower(configuration: config)
            try tower.loadNativeWeights(Self.weights(config), dtype: .float32)
            let patchValues = (0 ..< (8 * 24)).map { (index: Int) -> Float in
                Float(index % 17 - 8) * 0.1
            }
            let patches = MLXArray(patchValues).reshaped(8, 24)
            let grid = try GLM5NextVisionGrid(temporal: 2, height: 2, width: 2)
            let output = try tower.forward(
                patches: patches, grids: [grid],
                limits: .init(maximumQueryTokens: 1, maximumScoreElements: 4))
            // Independent scalar Float64 evaluation of the pinned HF equations:
            // Conv3D, RMSNorm, axial RoPE, attention, clipped GLU, residuals,
            // post norm, actual Conv2D, LayerNorm/GELU and the width-5 merger.
            // This vector is not a pretrained/full-model or BF16 parity result.
            let expected: [Float] = [
                -0.000423703781, -0.0000277325314, 0.000268788919, -0.000106926781,
                -0.000453256625, -0.0000301732973, 0.000286443423, -0.000114789963,
            ]
            #expect(output.shape == [2, 4])
            for (actual, expected) in zip(output.asArray(Float.self), expected) {
                #expect(abs(actual - expected) < 5e-7)
            }
            let separate = try (0 ..< 2).map { i in
                try tower.forward(
                    patches: patches[(i * 4) ..< (i * 4 + 4)],
                    grids: [.init(temporal: 1, height: 2, width: 2)])
            }
            let joined = concatenated(separate, axis: 0)
            eval(joined)
            #expect((output - joined).abs().max().item(Float.self) < 5e-7)
        }

        @Test func axialRotationUsesFP32AndBothAxes() {
            let angles = GLM5NextVisionTower.angles(height: 2, width: 2, dimension: 4)
            eval(angles)
            #expect(angles.shape == [4, 1, 4])
            #expect(angles.asArray(Float.self) == [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1])
            let input = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 1, 4)
            let rotated = GLM5NextVisionTower.rotary(input, angles: angles[3 ..< 4])
            eval(rotated)
            let expected = [
                cos(Float(1)) - 3 * sin(Float(1)),
                2 * cos(Float(1)) - 4 * sin(Float(1)),
                3 * cos(Float(1)) + sin(Float(1)), 4 * cos(Float(1)) + 2 * sin(Float(1)),
            ]
            for (actual, expected) in zip(rotated.asArray(Float.self), expected) {
                #expect(abs(actual - expected) < 1e-6)
            }
            let eight = GLM5NextVisionTower.angles(height: 2, width: 12, dimension: 8)
                .asArray(Float.self)
            var expectedAngles: [Float] = []
            for column in stride(from: 0, to: 12, by: 2) {
                for row in 0 ..< 2 {
                    for dx in 0 ..< 2 {
                        let half: [Float] = [
                            Float(row), Float(row) * 0.01,
                            Float(column + dx), Float(column + dx) * 0.01,
                        ]
                        expectedAngles += half + half
                    }
                }
            }
            #expect(eight == expectedAngles)
        }

        @Test func strictWeightsFrameIsolationAndFailureFence() throws {
            let config = try Self.configuration()
            let tower = try GLM5NextVisionTower(configuration: config)
            var weights = Self.weights(config)
            weights["model.visual.unexpected"] = MLXArray(Float(0))
            #expect(throws: GLM5NextVisionError.self) {
                try tower.loadNativeWeights(weights, dtype: .float32)
            }
            weights.removeValue(forKey: "model.visual.unexpected")
            try tower.loadNativeWeights(weights, dtype: .float32)
            let input = MLXArray.ones([4, 24])
            let grid = try GLM5NextVisionGrid(temporal: 1, height: 2, width: 2)
            #expect(throws: GLM5NextVisionError.self) {
                try tower.forward(
                    patches: input, grids: [grid], limits: .init(maximumFramePatches: 3))
            }
            #expect(throws: GLM5NextVisionError.self) {
                try tower.forward(
                    patches: input, grids: [grid], limits: .init(maximumScoreElements: 3))
            }
            enum Stop: Error { case beforeEvaluation }
            var checkpoints = 0
            #expect(throws: Stop.self) {
                try tower.forward(patches: input, grids: [grid]) { _ in
                    checkpoints += 1
                    throw Stop.beforeEvaluation
                }
            }
            #expect(checkpoints == 1)
            #expect(try tower.forward(patches: input, grids: [grid]).shape == [1, 4])
        }

        private static func processor(maximumPreparedBytes: Int = 1024 * 1024) throws
            -> GLM5NextProcessor
        {
            try GLM5NextProcessor(
                configuration: .init(
                    patchSize: 2, imageMinimumTokens: 1,
                    imageMaximumTokens: 64, videoMinimumTokens: 1, videoMaximumTokens: 128,
                    maximumPreparedBytes: maximumPreparedBytes), tokenizer: GLM5NextTestTokenizer())
        }

        @Test func rgb8BicubicMatchesPinnedTorchUint8Bytes() throws {
            // Torch 2.14.1 (5c4886908584029761b579af026dcfb627c84070), CPU,
            // torchvision 0.29.1 resize(BICUBIC, antialias=True), CHW uint8.
            // Includes both real oracle content sizes, downsampling, clipping,
            // and singleton axes. Pillow's fixed-22-bit path differs on these.
            let cases: [(Int, Int, Int, Int, String)] = [
                (
                    29, 47, 103, 168,
                    "962ebd55e312ec34b6b15719589ed21f10fade5300d5b4a6b08ded3100b28dd1"
                ),
                (
                    29, 47, 84, 136,
                    "19d98d035ac081f9d1f1b8cc5db7820de4cf93f21f76158a335e3f592ea87cb6"
                ),
                (9, 13, 3, 5, "e3acc86ea31a69cb181cab253910c80de75cb21afd033525e4ef66aa857106ca"),
                (3, 5, 8, 9, "5fa82d2b2b68fcc4494d699ece03902e1aa538a5f24b8ef030eb01d5e132eafe"),
                (8, 8, 1, 1, "ccd5192eaed473e83c731994435cc7ef20a1b0f876122082b03e6cd43452a09a"),
                (1, 7, 9, 2, "a8c9bf0f30f3c2c4293f9a24b7e2ccf85091e361e3f06873a9e1eb28cab4c573"),
                (7, 1, 2, 9, "06a42d9f9a84ef9f51e422394c64a67cff51b1e9fce8464a83f1e946d824bf39"),
            ]
            for (height, width, targetHeight, targetWidth, digest) in cases {
                let pixels: [UInt8] = (0 ..< (height * width)).flatMap { pixel in
                    (0 ..< 3).map { channel in
                        UInt8(((channel * height * width + pixel) * 19) % 256)
                    }
                }
                let output = try GLM5NextRGB8Bicubic.resize(
                    pixels, width: width, height: height,
                    targetWidth: targetWidth, targetHeight: targetHeight)
                #expect(output.count == targetWidth * targetHeight * 3)
                let actual = SHA256.hash(data: Data(output)).map { String(format: "%02x", $0) }
                    .joined()
                #expect(actual == digest)
            }
        }

        @Test func imageRGBNormalizationAndMergeMajorPatchOrdering() throws {
            let rgb = (0 ..< 16).flatMap { [UInt8($0 * 10), UInt8(100), UInt8(200)] }
            let image = try GLM5NextRGBImage(width: 4, height: 4, pixels: rgb)
            let media = try Self.processor().prepareImage(image)
            #expect(media.patches.shape == [4, 24])
            #expect(media.grid == (try .init(temporal: 1, height: 2, width: 2)))
            let first = media.patches[0].asArray(Float.self)
            let red = [0, 10, 40, 50].map { (Float($0) / 255 - 0.48145466) / 0.26862954 }
            for (actual, expected) in zip(Array(first.prefix(8)), red + red) {
                #expect(abs(actual - expected) < 2e-6)
            }
            for value in first[8 ..< 16] {
                #expect(abs(value - (100.0 / 255 - 0.4578275) / 0.26130258) < 2e-6)
            }
            #expect(media.timestamps.isEmpty)
        }

        @Test func videoTemporalPairsRemainDistinct() throws {
            let dark = try GLM5NextRGBImage(
                width: 4, height: 4, pixels: Array(repeating: 0, count: 48))
            let bright = try GLM5NextRGBImage(
                width: 4, height: 4, pixels: Array(repeating: 255, count: 48))
            let video = try GLM5NextDecodedVideo(
                frames: [dark, bright, bright, dark], framesPerSecond: 2, duration: 2)
            let prepared = try Self.processor().prepareVideo(video)
            #expect(prepared.grid == (try .init(temporal: 2, height: 2, width: 2)))
            #expect(prepared.patches.shape == [8, 24])
            #expect(prepared.timestamps == [0, 1])
            let first = prepared.patches[0].asArray(Float.self)
            #expect(abs(first[0] - (-0.48145466 / 0.26862954)) < 2e-6)
            #expect(abs(first[4] - ((1 - 0.48145466) / 0.26862954)) < 2e-6)
            let second = prepared.patches[4].asArray(Float.self)
            #expect(abs(second[0] - first[4]) < 2e-6 && abs(second[4] - first[0]) < 2e-6)
        }

        @Test func disabledSamplingResizesThreeFramesBeforeLastFrameDuplication() throws {
            let frames = try [UInt8(0), 96, 255].map { value in
                try GLM5NextRGBImage(
                    width: 47, height: 29,
                    pixels: Array(repeating: value, count: 47 * 29 * 3))
            }
            let video = try GLM5NextDecodedVideo(frames: frames, framesPerSecond: 2, duration: 1.5)
            let disabled = try GLM5NextProcessor(
                configuration: .init(
                    videoMaximumTokens: 128,
                    videoSamplingPolicy: .disabled), tokenizer: GLM5NextTestTokenizer())
            let plan = try disabled.videoPreparationPlan(video)
            let prepared = try disabled.prepareVideo(video)
            #expect(prepared.grid == (try .init(temporal: 2, height: 6, width: 10)))
            #expect(prepared.patches.shape == [120, 1176])
            #expect(prepared.patches.size * MemoryLayout<Float>.size == plan.preparedBytes)
            #expect(prepared.timestamps == [0, 1])
            // The second temporal pair repeats resized source frame 2 exactly,
            // including right-hand black padding, channel by channel.
            let lastPair = prepared.patches[60 ..< 120].asArray(Float.self)
            for patch in 0 ..< 60 {
                for channel in 0 ..< 3 {
                    let offset = patch * 1176 + channel * 392
                    #expect(
                        Array(lastPair[offset ..< (offset + 196)])
                            == Array(lastPair[(offset + 196) ..< (offset + 392)]))
                }
            }
            let enabled = try GLM5NextProcessor(
                configuration: .init(videoMaximumTokens: 128),
                tokenizer: GLM5NextTestTokenizer()
            ).prepareVideo(video)
            #expect(enabled.grid == (try .init(temporal: 1, height: 8, width: 12)))
            #expect(enabled.timestamps == [0])
        }

        @Test func disabledUserInputQuotesTheSameOddFramePlan() async throws {
            let frames = (0 ..< 3).map { i in
                UserInput.VideoFrame(
                    frame: CIImage(color: i % 2 == 0 ? .black : .white)
                        .cropped(to: CGRect(x: 0, y: 0, width: 47, height: 29)),
                    timeStamp: CMTime(value: Int64(i), timescale: 2))
            }
            let input = UserInput(prompt: "inspect", videos: [.frames(frames)])
            let exactBytes = 120 * 1176 * MemoryLayout<Float>.size
            func processor(bytes: Int) throws -> GLM5NextProcessor {
                try GLM5NextProcessor(
                    configuration: .init(
                        videoMaximumTokens: 128,
                        maximumPreparedBytes: bytes, videoSamplingPolicy: .disabled),
                    tokenizer: GLM5NextTestTokenizer())
            }
            let prepared = try await processor(bytes: exactBytes).prepareDetailed(input: input)
            #expect(prepared.input.video?.pixels.shape == [120, 1176])
            #expect(prepared.placements.map(\.tokenRange.count) == [15, 15])
            await #expect(throws: GLM5NextVisionError.self) {
                _ = try await processor(bytes: exactBytes - 1).prepare(input: input)
            }
        }

        @Test func paddingIsBlackBeforeNormalization() throws {
            let image = try GLM5NextRGBImage(
                width: 100, height: 200,
                pixels: Array(repeating: 255, count: 100 * 200 * 3))
            let processor = try GLM5NextProcessor(
                configuration: .init(), tokenizer: GLM5NextTestTokenizer())
            let prepared = try processor.prepareImage(image)
            #expect(prepared.grid == (try .init(temporal: 1, height: 16, width: 8)))
            let lastPatch = prepared.patches[127].asArray(Float.self)
            for channel in 0 ..< 3 {
                let expected =
                    -GLM5NextProcessor.mean[channel] / GLM5NextProcessor.standardDeviation[channel]
                for value in lastPatch[(channel * 392) ..< ((channel + 1) * 392)] {
                    #expect(abs(value - expected) < 2e-6)
                }
            }
        }

        @Test func userInputImageVideoMasksAndBudgetRejection() async throws {
            func image(_ color: CIColor) -> CIImage {
                CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
            }
            let frames = (0 ..< 4).map { i in
                UserInput.VideoFrame(
                    frame: image(i % 2 == 0 ? .black : .white),
                    timeStamp: CMTime(value: Int64(i), timescale: 2))
            }
            let input = UserInput(
                prompt: "inspect", images: [.ciImage(image(.white))], videos: [.frames(frames)])
            let prepared = try await Self.processor().prepareDetailed(input: input)
            #expect(prepared.input.image?.pixels.shape == [4, 24])
            #expect(prepared.input.video?.pixels.shape == [8, 24])
            #expect(prepared.placements.map(\.modality) == [.image, .video, .video])
            #expect(prepared.input.text.mask?.asArray(Bool.self).allSatisfy { $0 } == true)
            #expect(prepared.input.text.tokens.asArray(Int.self).filter { $0 == 154854 }.count == 3)
            let tiny = try Self.processor(maximumPreparedBytes: 383)
            await #expect(throws: GLM5NextVisionError.self) {
                _ = try await tiny.prepare(input: input)
            }
            let audio = UserInput(messages: [
                ["role": "user", "content": [["type": "input_audio"]]]
            ])
            await #expect(throws: GLM5NextVisionError.self) {
                _ = try await Self.processor().prepare(input: audio)
            }
        }

        @Test func encodedVideoUsesRealFramesNotOnlyMarkers() async throws {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString + ".mov")
            defer { try? FileManager.default.removeItem(at: url) }
            let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
            let input = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 16,
                    AVVideoHeightKey: 16,
                    AVVideoCompressionPropertiesKey: [AVVideoExpectedSourceFrameRateKey: 2],
                ])
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: 16, kCVPixelBufferHeightKey as String: 16,
                ])
            writer.add(input)
            #expect(writer.startWriting())
            writer.startSession(atSourceTime: .zero)
            for i in 0 ..< 4 {
                var buffer: CVPixelBuffer?
                #expect(
                    CVPixelBufferCreate(
                        kCFAllocatorDefault, 16, 16, kCVPixelFormatType_32BGRA, nil, &buffer)
                        == kCVReturnSuccess)
                let pixelBuffer = try #require(buffer)
                CVPixelBufferLockBaseAddress(pixelBuffer, [])
                let address = try #require(CVPixelBufferGetBaseAddress(pixelBuffer))
                memset(address, i % 2 == 0 ? 0 : 255, CVPixelBufferGetBytesPerRow(pixelBuffer) * 16)
                CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
                // Appending while the input is not ready raises an Objective-C
                // exception, so wait on wall-clock time and require readiness.
                let deadline = ContinuousClock.now.advanced(by: .seconds(10))
                while !input.isReadyForMoreMediaData, writer.status == .writing,
                    ContinuousClock.now < deadline
                {
                    try await Task.sleep(for: .milliseconds(5))
                }
                try #require(
                    input.isReadyForMoreMediaData,
                    "writer status \(writer.status.rawValue): \(String(describing: writer.error))")
                #expect(
                    adaptor.append(
                        pixelBuffer, withPresentationTime: CMTime(value: Int64(i), timescale: 2)))
            }
            writer.endSession(atSourceTime: CMTime(value: 2, timescale: 1))
            input.markAsFinished()
            await writer.finishWriting()
            #expect(writer.status == .completed)
            let prepared = try await Self.processor().prepareDetailed(
                input: UserInput(prompt: "video", videos: [.url(url)]))
            #expect(prepared.input.video?.frames?.first?.t == 2)
            #expect(prepared.input.video?.pixels.dim(0) == 128)
            #expect(prepared.placements.count == 2)
            let pixels = try #require(prepared.input.video?.pixels)
            #expect((pixels.max() - pixels.min()).item(Float.self) > 1)
        }
    }
}
