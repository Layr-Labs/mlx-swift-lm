// Copyright © 2026 Eigen Labs.
// Native sampler derived from SGLang Apache-2.0, commit
// 67bb6a58d0dad4a39af80fa1b2bf86f0de0cb99b (qwen_vl.smart_nframes and
// mimo_v2._decode_frames_and_timestamps). No downloaded code is executed.
@preconcurrency import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import MLXLLM
import MLXLMCommon

/// Encoded transport is separate from model pixel normalization.
/// No MLX arrays, remote URL, plaintext file, resize or mean/std operation.
public enum MiMoV26EncodedVisualDecoder {
    public enum Failure: Error, Equatable, Sendable {
        case invalidImage, invalidVideo, unsupportedRepresentation, audioTrackRequiresAudiovisualProfile
        case limit, arithmeticOverflow, inconsistentFrames
    }
    public struct Limits: Sendable {
        public let maximumPixels, maximumWorkingBytes, maximumSourceFrames, maximumSampledFrames: Int
        public init(maximumPixels: Int, maximumWorkingBytes: Int,
                    maximumSourceFrames: Int, maximumSampledFrames: Int) {
            self.maximumPixels = maximumPixels; self.maximumWorkingBytes = maximumWorkingBytes
            self.maximumSourceFrames = maximumSourceFrames; self.maximumSampledFrames = maximumSampledFrames
        }
    }
    public struct Sampling: Sendable {
        public let fps: Double
        public let minimumFrames, maximumFrames: Int
        public init(configuration: MiMoV26Configuration) throws {
            guard let values = configuration.processorFields else { throw Failure.invalidVideo }
            func number(_ key: String, fallback: Double) throws -> Double {
                guard let value = values[key], value != .null else { return fallback }
                guard case .number(let number) = value else { throw Failure.invalidVideo }
                let parsed = NSDecimalNumber(decimal:number).doubleValue
                guard parsed.isFinite, parsed >= 0 else { throw Failure.invalidVideo }
                return parsed == 0 ? fallback : parsed // native "or" fallback
            }
            let samplesPerSecond = try number("fps", fallback: 2)
            let low = try number("min_frames", fallback: 8)
            let high = try number("max_frames", fallback: 256)
            guard low.rounded(.towardZero) == low, high.rounded(.towardZero) == high,
                  low < Double(Int.max), high < Double(Int.max), samplesPerSecond > 0 else { throw Failure.invalidVideo }
            fps = samplesPerSecond
            minimumFrames = Int(low); maximumFrames = Int(high)
            // MiMo passes "num_frames", but pinned smart_nframes consumes
            // "nframes". No client sampling override is added here; follow
            // the actually consumed fps/min/max fields, including that detail.
        }
        public init(fps: Double, minimumFrames: Int, maximumFrames: Int) throws {
            guard fps.isFinite, fps > 0, minimumFrames > 0, maximumFrames > 0 else { throw Failure.invalidVideo }
            self.fps = fps; self.minimumFrames = minimumFrames; self.maximumFrames = maximumFrames
        }
        public func indices(totalFrames: Int, averageFPS: Double, admissionLimit: Int) throws -> [Int] {
            guard totalFrames >= 2, totalFrames <= 9_007_199_254_740_991,
                  averageFPS.isFinite, averageFPS > 0, admissionLimit > 0,
                  minimumFrames < Int.max else { throw Failure.invalidVideo }
            let minimum = (minimumFrames / 2 + minimumFrames % 2) * 2
            let maximum = maximumFrames / 2 * 2
            let estimate = Double(totalFrames) / averageFPS * fps
            guard estimate.isFinite else { throw Failure.invalidVideo }
            let chosen = min(min(max(estimate,Double(minimum)),Double(maximum)),Double(totalFrames))
            let count = Int(floor(chosen / 2)) * 2
            guard count >= 2, count <= totalFrames else { throw Failure.invalidVideo }
            // Resource refusal, NEVER a replacement sampler/max-frame clamp.
            guard count <= admissionLimit else { throw Failure.limit }
            let step = Double(totalFrames - 1) / Double(count - 1)
            var result: [Int] = []; result.reserveCapacity(count)
            for index in 0..<count {
                let value = index == count - 1 ? totalFrames - 1 : Int(floor(Double(index) * step))
                if result.last != value { result.append(value) } // np.unique
            }
            return result
        }
    }
    public struct VideoPlan: Sendable {
        public let sourceFrameCount: Int
        public let averageFPS: Double
        public let sampledIndices: [Int]
        public let timestamps: [Float]
        public let codedPixels: Int
        public let hasAudioTrack: Bool
        fileprivate let owner: MemoryBackedVideoAsset
        /// AV composition reuses identical owned bytes, never a second URL fetch
        /// or a bare AVAsset escape.
        package var sourceOwner: MemoryBackedVideoAsset { owner }
        public func decodeWorkingByteBound() throws -> Int {
            let sourceBuffers = try MiMoV26EncodedVisualDecoder.product(sourceFrameCount,codedPixels,32)
            let retainedRGB = try MiMoV26EncodedVisualDecoder.product(sampledIndices.count,codedPixels,12)
            let metadata = try MiMoV26EncodedVisualDecoder.product(sourceFrameCount,64)
            var total = owner.byteCount
            for value in [sourceBuffers,retainedRGB,metadata,1 << 20] {
                let (next,overflow) = total.addingReportingOverflow(value)
                guard !overflow else { throw Failure.arithmeticOverflow }
                total = next
            }
            return total
        }
        fileprivate init(owner: MemoryBackedVideoAsset, count: Int, fps: Double,
                         indices: [Int], pixels: Int, audio: Bool) {
            self.owner = owner; sourceFrameCount = count; averageFPS = fps
            sampledIndices = indices; codedPixels = pixels; hasAudioTrack = audio
            // Native Float32 tensor / weak scalar conversion, not actual PTS.
            timestamps = indices.map { Float($0) / Float(fps) }
        }
    }

    private static func product(_ values: Int...) throws -> Int {
        var result = 1
        for value in values {
            let (next,overflow) = result.multipliedReportingOverflow(by:value)
            guard value >= 0, !overflow else { throw Failure.arithmeticOverflow }
            result = next
        }
        return result
    }
    private static func checked(_ limits: Limits) throws {
        guard limits.maximumPixels > 0, limits.maximumWorkingBytes > 0,
              limits.maximumSourceFrames > 0, limits.maximumSampledFrames > 0 else { throw Failure.limit }
    }

    /// Application EXIF orientation is applied once. This preserves DarkBloom
    /// ingest policy, not identity with SGLang's non-smart RGB/EXIF bypass.
    /// Straight channels are preserved; alpha is discarded, never composited.
    public static func image(_ data: Data, limits: Limits) throws -> MiMoV26Pixels.DecodedRGB {
        try checked(limits); try Task.checkCancellation()
        guard let source = CGImageSourceCreateWithData(data as CFData,
                [kCGImageSourceShouldCache:false] as CFDictionary),
              let properties = CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, try product(width,height) <= limits.maximumPixels else {
            throw Failure.invalidImage
        }
        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        guard ((properties[kCGImagePropertyDepth] as? NSNumber)?.intValue ?? 8) <= 8 else {
            throw Failure.unsupportedRepresentation
        }
        guard (1...8).contains(orientation),
              let image = CGImageSourceCreateImageAtIndex(source,0,
                [kCGImageSourceShouldCache:false, kCGImageSourceShouldAllowFloat:false] as CFDictionary) else {
            throw Failure.invalidImage
        }
        return try straightRGB(image,orientation:orientation,limits:limits)
    }

    /// Copy actual straight raster channels without CGContext color matching,
    /// premultiplication or a CIContext that may submit untracked Metal work.
    static func straightRGB(_ image: CGImage, orientation: Int,
                            limits: Limits) throws -> MiMoV26Pixels.DecodedRGB {
        try checked(limits)
        let width = image.width, height = image.height
        let pixels = try product(width,height)
        let rawBytes = try product(image.bytesPerRow,height)
        let floatBytes = try product(pixels,3,MemoryLayout<Float>.stride)
        let (working,overflow) = rawBytes.addingReportingOverflow(floatBytes)
        guard !overflow, pixels <= limits.maximumPixels, working <= limits.maximumWorkingBytes,
              image.bitsPerComponent == 8, let space = image.colorSpace,
              [.rgb,.monochrome].contains(space.model),
              let data = image.dataProvider?.data, CFDataGetLength(data) >= rawBytes,
              let pointer = CFDataGetBytePtr(data) else { throw Failure.unsupportedRepresentation }
        let components = image.bitsPerPixel / 8
        let channels = space.model == .rgb ? 3 : 1
        let minimumRowBytes = try product(width,components)
        guard image.bitsPerPixel % 8 == 0, components == channels || components == channels + 1,
              image.bytesPerRow >= minimumRowBytes else { throw Failure.unsupportedRepresentation }
        let alpha = image.alphaInfo
        guard components != channels || alpha == .none else { throw Failure.unsupportedRepresentation }
        let first = alpha == .first || alpha == .premultipliedFirst || alpha == .noneSkipFirst
        let premultiplied = alpha == .premultipliedFirst || alpha == .premultipliedLast
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let reverse = (components == 4 && order == .byteOrder32Little)
            || (components == 2 && order == .byteOrder16Little)
        let oriented = orientation >= 5
        let outWidth = oriented ? height : width, outHeight = oriented ? width : height
        var output = [Float](repeating:0,count:try product(pixels,3))
        for y in 0..<outHeight {
            try Task.checkCancellation()
            for x in 0..<outWidth {
                let sx: Int, sy: Int
                switch orientation {
                case 1: (sx,sy) = (x,y)
                case 2: (sx,sy) = (width-1-x,y)
                case 3: (sx,sy) = (width-1-x,height-1-y)
                case 4: (sx,sy) = (x,height-1-y)
                case 5: (sx,sy) = (y,x)
                case 6: (sx,sy) = (y,height-1-x)
                case 7: (sx,sy) = (width-1-y,height-1-x)
                case 8: (sx,sy) = (width-1-y,x)
                default: throw Failure.invalidImage
                }
                let offset = sy * image.bytesPerRow + sx * components
                func component(_ i: Int) -> UInt8 { pointer[offset + (reverse ? components-1-i : i)] }
                // No heuristic division/white-compositing for a decoder that
                // already lost the original straight channels.
                if premultiplied, component(first ? 0 : components-1) != 255 {
                    throw Failure.unsupportedRepresentation
                }
                let start = first ? 1 : 0
                let out = y * outWidth + x
                for channel in 0..<3 {
                    output[channel * pixels + out] = Float(component(start + (channels == 1 ? 0 : channel)))
                }
            }
        }
        return .init(height:outHeight,width:outWidth,planarRGB:output)
    }

    /// Complete compressed-sample enumeration provides actual frame count and
    /// sample-duration average; no duration*nominal-fps estimate or AV seek.
    /// Compressed output is decode-ordered; decoded output is presentation-
    /// ordered. Source indices refer to the latter order, including VFR.
    public static func inspectVideo(_ owner: MemoryBackedVideoAsset, sampling: Sampling,
                                    limits: Limits) async throws -> VideoPlan {
        try checked(limits)
        return try await owner.withAsset { asset in
            try Task.checkCancellation()
            let tracks = try await asset.loadTracks(withMediaType:.video)
            let audio = try await asset.loadTracks(withMediaType:.audio)
            guard tracks.count == 1, let track = tracks.first else { throw Failure.unsupportedRepresentation }
            let descriptions = try await track.load(.formatDescriptions)
            var pixels = 0
            for description in descriptions {
                let metadata = CMFormatDescriptionGetExtensions(description) as? [String:Any]
                let transfer = metadata?[kCMFormatDescriptionExtension_TransferFunction as String] as? String
                guard transfer != (kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String),
                      transfer != (kCVImageBufferTransferFunction_ITU_R_2100_HLG as String) else {
                    throw Failure.unsupportedRepresentation // no implicit HDR tone mapping
                }
                let dimensions = CMVideoFormatDescriptionGetDimensions(description)
                guard dimensions.width > 0, dimensions.height > 0 else { throw Failure.invalidVideo }
                pixels = max(pixels,try product(Int(dimensions.width),Int(dimensions.height)))
            }
            guard pixels > 0, pixels <= limits.maximumPixels else { throw Failure.limit }
            let reader = try AVAssetReader(asset:asset)
            let output = AVAssetReaderTrackOutput(track:track,outputSettings:nil)
            guard reader.canAdd(output) else { throw Failure.invalidVideo }
            reader.add(output)
            guard reader.startReading() else { throw Failure.invalidVideo }
            // Apple's cancelReading contract stops background reads. This
            // codec cleanup is not a fabricated MLX/native-shutdown receipt.
            defer { reader.cancelReading() }
            var count = 0
            var duration = CMTime.zero
            while let sample = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                let amount = CMSampleBufferGetNumSamples(sample)
                let (next,overflow) = count.addingReportingOverflow(amount)
                guard amount > 0, !overflow, next <= limits.maximumSourceFrames else { throw Failure.limit }
                let span = CMSampleBufferGetDuration(sample)
                guard span.isNumeric, span.seconds.isFinite, span > .zero else { throw Failure.invalidVideo }
                duration = CMTimeAdd(duration,span); count = next
                guard duration.isNumeric, duration.seconds.isFinite else { throw Failure.invalidVideo }
            }
            guard reader.status == .completed, count >= 2, duration > .zero else { throw Failure.invalidVideo }
            let fps = Double(count) / duration.seconds
            let indices = try sampling.indices(totalFrames:count,averageFPS:fps,
                admissionLimit:limits.maximumSampledFrames)
            return VideoPlan(owner:owner,count:count,fps:fps,indices:indices,pixels:pixels,audio:!audio.isEmpty)
        }
    }

    /// A silent input remains distinct from VideoAudioInput. Sound is never
    /// dropped; only the paired audiovisual decoder can consume both tracks.
    public static func silentVideo(_ plan: VideoPlan, limits: Limits) async throws -> MiMoV26SilentVideo {
        try checked(limits)
        guard !plan.hasAudioTrack else { throw Failure.audioTrackRequiresAudiovisualProfile }
        return try await decodedFrames(plan, limits: limits)
    }

    package static func audiovisualFrames(_ plan: VideoPlan, limits: Limits) async throws -> MiMoV26SilentVideo {
        guard plan.hasAudioTrack else { throw Failure.invalidVideo }
        return try await decodedFrames(plan, limits: limits)
    }

    private static func decodedFrames(_ plan: VideoPlan, limits: Limits) async throws -> MiMoV26SilentVideo {
        try checked(limits)
        guard plan.codedPixels <= limits.maximumPixels,
              plan.sampledIndices.count <= limits.maximumSampledFrames,
              plan.sourceFrameCount <= limits.maximumSourceFrames,
              try plan.decodeWorkingByteBound() <= limits.maximumWorkingBytes else { throw Failure.limit }
        return try await plan.owner.withAsset { asset in
            let tracks = try await asset.loadTracks(withMediaType:.video)
            guard tracks.count == 1, let track = tracks.first else { throw Failure.invalidVideo }
            let transform = try await track.load(.preferredTransform)
            let reader = try AVAssetReader(asset:asset)
            let output = AVAssetReaderTrackOutput(track:track,outputSettings:[
                kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
            guard reader.canAdd(output) else { throw Failure.invalidVideo }
            reader.add(output)
            guard reader.startReading() else { throw Failure.invalidVideo }
            defer { reader.cancelReading() }
            var index = 0, selected = 0
            var frames: [MiMoV26Pixels.DecodedRGB] = []
            var lastPTS: CMTime?
            while let sample = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                guard index < plan.sourceFrameCount, CMSampleBufferGetNumSamples(sample) == 1 else {
                    throw Failure.inconsistentFrames
                }
                let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                guard pts.isNumeric, lastPTS.map({ pts > $0 }) ?? true else { throw Failure.inconsistentFrames }
                lastPTS = pts
                if selected < plan.sampledIndices.count, index == plan.sampledIndices[selected] {
                    guard let pixel = CMSampleBufferGetImageBuffer(sample) else { throw Failure.invalidVideo }
                    frames.append(try frame(pixel,transform:transform,limits:limits))
                    selected += 1
                }
                index += 1
            }
            guard reader.status == .completed, index == plan.sourceFrameCount,
                  selected == plan.sampledIndices.count, frames.count == plan.timestamps.count else {
                throw Failure.inconsistentFrames
            }
            return .init(frames:frames,timestamps:plan.timestamps)
        }
    }

    private static func frame(_ buffer: CVPixelBuffer, transform: CGAffineTransform,
                              limits: Limits) throws -> MiMoV26Pixels.DecodedRGB {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else {
            throw Failure.unsupportedRepresentation
        }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let pixels = try product(width,height), stride = CVPixelBufferGetBytesPerRow(buffer)
        let minimumStride = try product(width,4)
        guard pixels <= limits.maximumPixels,
              try product(pixels,12) <= limits.maximumWorkingBytes,
              stride >= minimumStride else { throw Failure.limit }
        // Exact orientation transforms only: no arbitrary affine resampling
        // or guessed pixel-aspect correction. Translation normalizes to origin.
        let shape = [transform.a,transform.b,transform.c,transform.d]
        let orientations: [[CGFloat]] = [
            [1,0,0,1],[-1,0,0,1],[-1,0,0,-1],[1,0,0,-1],
            [0,1,1,0],[0,1,-1,0],[0,-1,-1,0],[0,-1,1,0]]
        guard let position = orientations.firstIndex(of:shape),
              CVPixelBufferLockBaseAddress(buffer,.readOnly) == kCVReturnSuccess else {
            throw Failure.unsupportedRepresentation
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer,.readOnly) }
        guard let address = CVPixelBufferGetBaseAddress(buffer) else { throw Failure.invalidVideo }
        let bytes = try product(stride,height)
        guard bytes <= limits.maximumWorkingBytes else { throw Failure.limit }
        let data = Data(bytes:address,count:bytes)
        guard let provider = CGDataProvider(data:data as CFData),
              let image = CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:32,
                bytesPerRow:stride,space:CGColorSpaceCreateDeviceRGB(),
                bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.noneSkipFirst.rawValue)
                    .union(.byteOrder32Little),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else {
            throw Failure.invalidVideo
        }
        return try straightRGB(image,orientation:position+1,limits:limits)
    }
}
