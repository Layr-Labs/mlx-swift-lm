@preconcurrency import AVFoundation
import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Writes small Photo-JPEG QuickTime movies for the video tests of this file.
private enum MediaKernelTestMovie {

    enum Failure: Error {
        case cannotAddInput
        case pixelBuffer
        case writeFailed
    }

    /// The side of each square frame, in pixels.
    static let side = 32

    /// Writes a movie to `url`. Frame `i` shows `colors[i]` (8-bit RGB) and
    /// starts at second `i`. The movie ends at second `endSeconds`.
    static func write(colors: [(UInt8, UInt8, UInt8)], endSeconds: Int64, to url: URL)
        async throws
    {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.jpeg,
            AVVideoWidthKey: side,
            AVVideoHeightKey: side,
            AVVideoCompressionPropertiesKey: [AVVideoQualityKey: 1.0],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard writer.canAdd(input) else { throw Failure.cannotAddInput }
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? Failure.writeFailed }
        writer.startSession(atSourceTime: .zero)

        for (index, color) in colors.enumerated() {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 1_000_000)
            }
            let buffer = try pixelBuffer(color: color)
            let time = CMTime(value: CMTimeValue(index), timescale: 1)
            guard adaptor.append(buffer, withPresentationTime: time) else {
                throw writer.error ?? Failure.writeFailed
            }
        }
        input.markAsFinished()
        writer.endSession(atSourceTime: CMTime(value: endSeconds, timescale: 1))
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else { throw writer.error ?? Failure.writeFailed }
    }

    private static func pixelBuffer(color: (UInt8, UInt8, UInt8)) throws -> CVPixelBuffer {
        var created: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, side, side, kCVPixelFormatType_32BGRA, nil, &created)
        guard status == kCVReturnSuccess, let buffer = created else { throw Failure.pixelBuffer }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw Failure.pixelBuffer }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for y in 0 ..< side {
            for x in 0 ..< side {
                let pixel = bytes + y * rowBytes + x * 4
                pixel[0] = color.2
                pixel[1] = color.1
                pixel[2] = color.0
                pixel[3] = 255
            }
        }
        return buffer
    }

    /// Makes a new empty folder in the temporary folder.
    static func makeFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("media-kernel-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    /// Mean of each RGB channel of `image`, as rendered by `asMLXArray`.
    static func meanRGB(_ image: CIImage) -> [Float] {
        image.asMLXArray().mean(axes: [0, 2, 3]).asArray(Float.self)
    }
}

extension KernelTests {

    /// Image helpers of `MediaProcessing` with small uniform-color images.
    ///
    /// CoreImage works in half float by default, and the rendered values are
    /// the values of its linear working space. The test colors are sRGB
    /// colors. Pure 0 and 1 channels stay 0 and 1 in sRGB, in the linear
    /// working space and in both tone curves, so most tests use pure colors
    /// and compare with exact values.
    @Suite
    struct MediaProcessingImageTests {

        /// A color in the sRGB color space. The working space of CoreImage
        /// has the sRGB primaries, so pure channels stay pure.
        static func sRGB(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat) -> CIColor {
            CIColor(
                red: red, green: green, blue: blue,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)!
        }

        static var red: CIColor { sRGB(1, 0, 0) }
        static var white: CIColor { sRGB(1, 1, 1) }
        static var black: CIColor { sRGB(0, 0, 0) }

        static func solid(_ color: CIColor, width: Int, height: Int) -> CIImage {
            CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
        }

        /// Largest absolute difference between `array` and `expected`.
        static func maxDifference(_ array: MLXArray, _ expected: [Float]) -> Float {
            abs(array.reshaped(-1) - MLXArray(expected)).max().item(Float.self)
        }

        @Test
        func asMLXArrayIsPlanarWithThreeChannels() {
            let image = Self.solid(Self.red, width: 3, height: 2)

            let array = MediaProcessing.asMLXArray(image)
            #expect(array.shape == [1, 3, 2, 3], "planar shape [1, C, H, W]")
            #expect(array.dtype == .float32, "float32 pixels")
            // Tolerance 1e-3: CoreImage keeps 0 and 1 exact in half float.
            #expect(
                Self.maxDifference(array, [1, 1, 1, 1, 1, 1] + [Float](repeating: 0, count: 12))
                    < 1e-3, "red planes")

            let tagged = image.asMLXArray(colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
            #expect(tagged.shape == [1, 3, 2, 3], "shape with an sRGB color space")
            // Tolerance 1e-3: pure red is (1, 0, 0) in sRGB too.
            #expect(
                Self.maxDifference(tagged, [1, 1, 1, 1, 1, 1] + [Float](repeating: 0, count: 12))
                    < 1e-3, "red planes in sRGB")
        }

        @Test
        func normalizeUsesMeanAndStandardDeviation() {
            let image = Self.solid(Self.red, width: 2, height: 2)

            // (1 - 0.5) / 0.25 = 2, (0 - 0.25) / 0.5 = -0.5, (0 - 0.5) / 0.5 = -1.
            let normalized = MediaProcessing.normalize(
                image, mean: (0.5, 0.25, 0.5), std: (0.25, 0.5, 0.5))
            let expected: [Float] =
                [Float](repeating: 2, count: 4) + [Float](repeating: -0.5, count: 4)
                + [Float](repeating: -1, count: 4)
            // Tolerance 1e-3: 2, -0.5 and -1 are exact in half float.
            #expect(
                Self.maxDifference(normalized.asMLXArray(), expected) < 1e-3,
                "normalized red")

            // The extension method gives the same values.
            let viaExtension = image.normalized(mean: (0.5, 0.25, 0.5), std: (0.25, 0.5, 0.5))
            #expect(
                Self.maxDifference(viaExtension.asMLXArray(), expected) < 1e-3,
                "normalized red through the extension")
        }

        @Test
        func toneCurvesAreInverseFunctions() {
            let gray = Self.solid(Self.sRGB(0.5, 0.5, 0.5), width: 2, height: 2)
            let original = MediaProcessing.asMLXArray(gray)
            let linear = MediaProcessing.inLinearToneCurveSpace(gray).asMLXArray()
            let roundTrip = MediaProcessing.inSRGBToneCurveSpace(
                MediaProcessing.inLinearToneCurveSpace(gray)
            ).asMLXArray()

            let originalValue = original[0, 0, 0, 0].item(Float.self)
            let linearValue = linear[0, 0, 0, 0].item(Float.self)
            #expect(originalValue > 0.01 && originalValue < 0.99, "gray is not an end value")
            // The sRGB to linear curve maps each value in (0, 1) to a smaller value.
            #expect(linearValue < originalValue - 0.01, "linear curve makes gray darker")
            // Tolerance 1e-2: two curves in half float lose about 3 digits.
            #expect(
                abs(roundTrip - original).max().item(Float.self) < 1e-2,
                "sRGB curve undoes the linear curve")

            // The extension methods call the same filters.
            let viaExtension = gray.toLinear().toSRGB().asMLXArray()
            // Tolerance 1e-2: as above.
            #expect(
                abs(viaExtension - original).max().item(Float.self) < 1e-2,
                "extension round trip")

            // 0 and 1 do not change.
            let white = Self.solid(Self.white, width: 1, height: 1)
            let black = Self.solid(Self.black, width: 1, height: 1)
            // Tolerance 1e-3: 0 and 1 are fixed points of both curves.
            #expect(
                Self.maxDifference(white.toLinear().asMLXArray(), [1, 1, 1]) < 1e-3,
                "white under the linear curve")
            #expect(
                Self.maxDifference(black.toSRGB().asMLXArray(), [0, 0, 0]) < 1e-3,
                "black under the sRGB curve")
        }

        @Test
        func padToSquareCentersAWideImage() {
            let image = Self.solid(Self.red, width: 4, height: 2)
            let square = MediaProcessing.padToSquare(image)
            #expect(square.extent == CGRect(x: 0, y: 0, width: 4, height: 4), "square extent")

            let array = square.asMLXArray()
            #expect(array.shape == [1, 3, 4, 4], "square shape")
            // One background row above and one below the image. The pattern is
            // the same in both row orders.
            let redPlane: [Float] = [0, 0, 0, 0] + [1, 1, 1, 1] + [1, 1, 1, 1] + [0, 0, 0, 0]
            // Tolerance 1e-3: copies of 0 and 1.
            #expect(
                Self.maxDifference(array[0, 0], redPlane) < 1e-3, "red plane of the padded image")
            #expect(
                Self.maxDifference(array[0, 1 ..< 3], [Float](repeating: 0, count: 32)) < 1e-3,
                "green and blue planes of the padded image")
        }

        @Test
        func paddingToSquareCentersATallImageOnAColor() {
            let image = Self.solid(Self.red, width: 2, height: 4)
            let square = image.paddingToSquare(backgroundColor: Self.white)
            #expect(square.extent == CGRect(x: 0, y: 0, width: 4, height: 4), "square extent")

            let array = square.asMLXArray()
            // One white column at the left and one at the right.
            let row: [Float] = [1, 0, 0, 1]
            let greenPlane = row + row + row + row
            // Tolerance 1e-3: copies of 0 and 1.
            #expect(
                Self.maxDifference(array[0, 0], [Float](repeating: 1, count: 16)) < 1e-3,
                "red plane of the white padded image")
            #expect(
                Self.maxDifference(array[0, 1], greenPlane) < 1e-3,
                "green plane of the white padded image")
            #expect(
                Self.maxDifference(array[0, 2], greenPlane) < 1e-3,
                "blue plane of the white padded image")
        }

        @Test
        func lanczosResampleHasTheTargetSize() {
            let image = Self.solid(Self.red, width: 64, height: 32)
            let target = CGSize(width: 16, height: 16)

            #expect(
                MediaProcessing.aspectRatioForResample(image, size: target) == 0.5,
                "aspect ratio for resample")

            let resampled = MediaProcessing.resampleLanczos(image, to: target)
            #expect(resampled.extent == CGRect(origin: .zero, size: target), "lanczos extent")

            let viaExtension = image.resampled(to: CGSize(width: 24, height: 8), method: .lanczos)
            #expect(viaExtension.extent.size == CGSize(width: 24, height: 8), "extension extent")

            let bicubic = image.resampled(to: CGSize(width: 24, height: 8))
            #expect(bicubic.extent.size == CGSize(width: 24, height: 8), "bicubic extent")

            // Far from the edges a uniform image keeps its color.
            let array = resampled.asMLXArray()
            let center = (0 ..< 3).map { array[0, $0, 8, 8].item(Float.self) }
            // Tolerance 2e-2: the Lanczos weights in half float.
            #expect(
                abs(center[0] - 1) < 2e-2 && abs(center[1]) < 2e-2 && abs(center[2]) < 2e-2,
                "center pixel stays red")
        }

        @Test
        func centerCropOfAnImage() {
            let image = Self.solid(Self.red, width: 8, height: 6)

            let cropped = MediaProcessing.centerCrop(image, size: CGSize(width: 4, height: 4))
            #expect(
                cropped.extent == CGRect(x: 0, y: 0, width: 4, height: 4),
                "cropped image moves to the origin")
            #expect(cropped.asMLXArray().shape == [1, 3, 4, 4], "cropped pixels shape")

            let same = MediaProcessing.centerCrop(image, size: CGSize(width: 8, height: 10))
            #expect(same.extent == image.extent, "no crop when the image is small enough")
        }

        @Test
        func applyScalesOnlyWhenAResizeIsSet() {
            let image = Self.solid(Self.red, width: 64, height: 32)

            #expect(
                MediaProcessing.apply(image, processing: nil).extent == image.extent,
                "no processing")
            #expect(
                MediaProcessing.apply(image, processing: .init()).extent == image.extent,
                "processing without resize")

            // Scale min(16 / 64, 16 / 32) = 0.25.
            let resized = MediaProcessing.apply(
                image, processing: .init(resize: CGSize(width: 16, height: 16)))
            #expect(
                resized.extent == CGRect(x: 0, y: 0, width: 16, height: 8), "resized extent")
        }
    }

    /// Video sampling of `MediaProcessing` with frames in memory and with
    /// small movies that the test writes.
    @Suite
    struct MediaProcessingVideoTests {

        static func solid(_ color: CIColor) -> CIImage {
            CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        }

        @Test
        func zeroTargetRateStillGivesOneFrame() async throws {
            let frames = (0 ..< 3).map {
                VideoFrame(
                    frame: Self.solid(.red), timeStamp: CMTime(value: Int64($0), timescale: 1))
            }
            let processed = try await MediaProcessing.asProcessedSequence(
                .frames(frames), targetFPS: { _ in 0 })
            #expect(processed.frames.count == 1, "at least one frame")
            #expect(processed.timestamps == [CMTime.zero], "first frame")
            #expect(processed.totalDuration == CMTime(value: 2, timescale: 1), "frame range")
            #expect(processed.frames[0].shape == [1, 3, 4, 4], "frame shape")
        }

        @Test
        func framesThatStartAfterZeroAreSampled() async throws {
            // Frames at 5, 6 and 7 s. The duration is 2 s, so 1 fps asks for
            // round(1 * 2) = 2 frames.
            let frames = (5 ..< 8).map {
                VideoFrame(
                    frame: Self.solid(.red), timeStamp: CMTime(value: Int64($0), timescale: 1))
            }
            let processed = try await MediaProcessing.asProcessedSequence(
                .frames(frames), samplesPerSecond: 1)
            #expect(processed.totalDuration == CMTime(value: 2, timescale: 1), "frame range")
            let frameCount = processed.frames.count
            // A synchronous function, so that the synchronous
            // `withKnownIssue` is used.
            func check() {
                withKnownIssue(
                    """
                    MediaProcessing._asProcessedSequence([VideoFrame]) builds the sample \
                    times from 0 to the duration, but compares them with the absolute frame \
                    time stamps. When the first frame is after 0 no frame is at or before a \
                    sample time, so the result has no frames.
                    """
                ) {
                    #expect(frameCount == 2, "frames after time zero")
                } matching: {
                    $0.isFailedExpectation(["frames after time zero"])
                }
            }
            check()
        }

        @Test
        func emptyAssetHasNoVideoTrack() async throws {
            await #expect(throws: VLMError.noVideoTrackFound) {
                _ = try await MediaProcessing.asProcessedSequence(
                    .avAsset(AVMutableComposition()), samplesPerSecond: 1)
            }
        }

        @Test
        func movieAssetIsSampled() async throws {
            let folder = try MediaKernelTestMovie.makeFolder()
            defer { try? FileManager.default.removeItem(at: folder) }
            let url = folder.appendingPathComponent("red-blue.mov")
            // Red from 0 s, blue from 1 s, end at 2 s.
            try await MediaKernelTestMovie.write(
                colors: [(255, 0, 0), (0, 0, 255)], endSeconds: 2, to: url)
            let asset = AVURLAsset(url: url)
            let side = MediaKernelTestMovie.side

            // 1 sample per second over 2 s: samples at 0 s and 2 s.
            let images = try await MediaProcessing.asCIImageSequence(asset, samplesPerSecond: 1)
            #expect(images.count == 2, "image sequence count")
            #expect(
                images.map(\.extent.size)
                    == Array(repeating: CGSize(width: side, height: side), count: images.count),
                "image sequence size")
            if let first = images.first, let last = images.last {
                let firstMean = MediaKernelTestMovie.meanRGB(first)
                let lastMean = MediaKernelTestMovie.meanRGB(last)
                // Loose bounds: JPEG changes the colors a little.
                #expect(firstMean[0] > 0.8 && firstMean[2] < 0.2, "first image is red")
                #expect(lastMean[2] > 0.8 && lastMean[0] < 0.2, "last image is blue")
            }

            // maxFrames 1 keeps only the sample at 0 s.
            let capped = try await MediaProcessing.asProcessedSequence(
                .avAsset(asset), targetFPS: { _ in 1 }, maxFrames: 1)
            #expect(capped.frames.count == 1, "capped frame count")
            #expect(capped.timestamps.count == 1, "capped time stamp count")
            // Tolerance 1e-3 s: the asset time scale can differ from 1.
            #expect(abs(capped.totalDuration.seconds - 2) < 1e-3, "asset duration")
            #expect(capped.frames.first?.shape == [1, 3, side, side], "capped frame shape")

            // The deprecated asset overload with a sample rate.
            let deprecated = try await MediaProcessing.asProcessedSequence(
                asset as AVAsset, samplesPerSecond: 1)
            #expect(deprecated.frames.count == 2, "deprecated overload frame count")
        }
    }

    /// `DiffusionGemmaVideoFrames.sample` with movie assets. The tests of
    /// `DiffusionGemmaProcessorTests` cover the in-memory frames.
    @Suite
    struct DiffusionGemmaVideoFramesKernelTests {

        typealias Sample = (seconds: Double, mean: [Float])

        static func sample(_ video: UserInput.Video) async throws -> [Sample] {
            try await DiffusionGemmaVideoFrames.sample(video) { frame -> Sample in
                (seconds: frame.timeStamp.seconds, mean: MediaKernelTestMovie.meanRGB(frame.frame))
            }
        }

        static func check(_ samples: [Sample], _ label: String) {
            #expect(samples.count == 2, "one frame per second (\(label))")
            guard samples.count == 2 else { return }
            // Tolerance 1e-3 s: the decoder reports the frame time.
            #expect(abs(samples[0].seconds - 0) < 1e-3, "first frame at 0 s (\(label))")
            #expect(abs(samples[1].seconds - 1) < 1e-3, "second frame at 1 s (\(label))")
            // Loose bounds: JPEG changes the colors a little.
            #expect(
                samples[0].mean[0] > 0.8 && samples[0].mean[2] < 0.2,
                "first frame is red (\(label))")
            #expect(
                samples[1].mean[2] > 0.8 && samples[1].mean[0] < 0.2,
                "second frame is blue (\(label))")
        }

        @Test
        func movieIsSampledOncePerSecondFromEachSource() async throws {
            let folder = try MediaKernelTestMovie.makeFolder()
            defer { try? FileManager.default.removeItem(at: folder) }
            let url = folder.appendingPathComponent("red-blue.mov")
            try await MediaKernelTestMovie.write(
                colors: [(255, 0, 0), (0, 0, 255)], endSeconds: 2, to: url)

            Self.check(try await Self.sample(.url(url)), "url")
            Self.check(try await Self.sample(.avAsset(AVURLAsset(url: url))), "asset")
            let owner = try MemoryBackedVideoAsset(videoData: Data(contentsOf: url))
            Self.check(try await Self.sample(.memoryBacked(owner)), "memory")
        }

        @Test
        func movieLongerThanSixtySecondsIsRefused() async throws {
            let folder = try MediaKernelTestMovie.makeFolder()
            defer { try? FileManager.default.removeItem(at: folder) }
            let url = folder.appendingPathComponent("long.mov")
            try await MediaKernelTestMovie.write(colors: [(255, 0, 0)], endSeconds: 61, to: url)

            await #expect(
                throws: DiffusionGemmaModelError.invalidInput(
                    "native video duration must not exceed60s")
            ) {
                _ = try await Self.sample(.url(url))
            }
        }

        @Test
        func assetWithoutVideoIsRefused() async throws {
            await #expect(throws: DiffusionGemmaModelError.invalidInput("undecodable video")) {
                _ = try await Self.sample(.avAsset(AVMutableComposition()))
            }
        }
    }
}
