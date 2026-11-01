// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Modified from Transformers f0705f991a71f05dad9569336b181119f5546f93:
// image_processing_{pil_,}glm5_next.py, video_processing_glm5_next.py,
// processing_glm5_next.py. Native RGB8/Torch-compatible resize, bounded video
// decoding, explicit patch/token ownership, and checked Swift geometry are local changes.

@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon

private func glm5Product(_ values: [Int]) throws -> Int {
    var result = 1
    for value in values {
        let next = result.multipliedReportingOverflow(by: value)
        guard value > 0, !next.overflow else {
            throw GLM5NextVisionError.invalidInput("geometry overflow")
        }
        result = next.partialValue
    }
    return result
}

/// Shared encoded-sRGB, unpremultiplied RGB8 contract, with the top row first.
/// Cluster decode selects software rendering so its host budget owns the work.
enum GLM5NextRGBDecoder {
    static func render(
        _ image: CIImage, maximumBytes: Int, useSoftwareRenderer: Bool = false
    ) throws -> GLM5NextRGBImage {
        try Task.checkCancellation()
        let extent = image.extent
        guard extent.width.isFinite, extent.height.isFinite, extent.width > 0, extent.height > 0,
            extent.width <= 65536, extent.height <= 65536,
            extent.width.rounded() == extent.width, extent.height.rounded() == extent.height
        else {
            throw GLM5NextVisionError.invalidInput("image extent")
        }
        let width = Int(extent.width)
        let height = Int(extent.height)
        let count = try glm5Product([width, height, 4])
        guard count <= maximumBytes else {
            throw GLM5NextVisionError.budgetExceeded("decoded image")
        }
        let context = CIContext(options: [
            .cacheIntermediates: false, .useSoftwareRenderer: useSoftwareRenderer,
        ])
        defer { context.clearCaches() }
        var bytes = [UInt8](repeating: 0, count: count)
        try bytes.withUnsafeMutableBytes { buffer in
            let destination = CIRenderDestination(
                bitmapData: buffer.baseAddress!, width: width,
                height: height, bytesPerRow: width * 4, format: .RGBA8)
            destination.colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
            destination.alphaMode = .unpremultiplied
            destination.isFlipped = true
            destination.isDithered = false
            destination.isClamped = true
            let task = try context.startTask(
                toRender: image, from: extent, to: destination, at: .zero)
            _ = try task.waitUntilCompleted()
        }
        var result = [UInt8]()
        result.reserveCapacity(width * height * 3)
        for pixel in 0 ..< (width * height) {
            if pixel.isMultiple(of: 16384) { try Task.checkCancellation() }
            result.append(contentsOf: bytes[pixel * 4 ..< (pixel * 4 + 3)])
        }
        return try .init(width: width, height: height, pixels: result)
    }
}

// The following RGB8 interpolation adapts HelperInterpBase's antialias weight
// construction/int16 quantization and separable uint8 convolution in PyTorch
// 5c4886908584029761b579af026dcfb627c84070, aten/src/ATen/native/cpu/UpSampleKernel.cpp.
// This is not Pillow's 22-bit resampler: Torch chooses precision per axis from
// the greatest normalized positive coefficient. Both passes round and clamp.
// Swift translation, checked allocations and cancellation are local changes.
//
// PyTorch copyright and BSD notice for this interpolation adaptation:
// From PyTorch:
// Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
// Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
// Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
// Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
// Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
// Copyright (c) 2011-2013 NYU                      (Clement Farabet)
// Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
// Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
// Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
// From Caffe2:
// Copyright (c) 2016-present, Facebook Inc. All rights reserved.
// All contributions by Facebook: Copyright (c) 2016 Facebook Inc.
// All contributions by Google: Copyright (c) 2015 Google Inc. All rights reserved.
// All contributions by Yangqing Jia: Copyright (c) 2015 Yangqing Jia. All rights reserved.
// All contributions by Kakao Brain: Copyright 2019-2020 Kakao Brain
// All contributions by Cruise LLC: Copyright (c) 2022 Cruise LLC. All rights reserved.
// All contributions by Tri Dao: Copyright (c) 2024 Tri Dao. All rights reserved.
// All contributions by Arm: Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
// All contributions from Caffe: Copyright(c) 2013, 2014, 2015, the respective contributors. All rights reserved.
// All other contributions: Copyright(c) 2015, 2016 the respective contributors. All rights reserved.
// Caffe2 uses a copyright model similar to Caffe: each contributor holds
// copyright over their contributions to Caffe2. The project versioning records
// all such contribution and copyright details. If a contributor wants to further
// mark their specific copyright on a particular contribution, they should
// indicate their copyright solely in the commit message of the change when it is
// committed.
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
//    and IDIAP Research Institute nor the names of its contributors may be
//    used to endorse or promote products derived from this software without
//    specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
enum GLM5NextRGB8Bicubic {
    private struct Span {
        let start: Int
        let weights: [Int16]
    }
    private struct Axis {
        let spans: [Span]
        let precision: Int
    }

    private static func cubic(_ distance: Double) -> Double {
        let x = abs(distance)
        if x < 1 { return (1.5 * x - 2.5) * x * x + 1 }
        if x < 2 { return ((-0.5 * x + 2.5) * x - 4) * x + 2 }
        return 0
    }

    private static func axis(input: Int, output: Int) throws -> Axis {
        let scale = Double(input) / Double(output)
        let support = 2 * max(scale, 1)
        let maximumTaps = Int(ceil(support)) * 2 + 1
        let inverseScale = scale >= 1 ? 1 / scale : 1
        var rows: [(start: Int, weights: [Double])] = []
        rows.reserveCapacity(output)
        var maximumWeight = 0.0
        for index in 0 ..< output {
            if index.isMultiple(of: 64) { try Task.checkCancellation() }
            let center = scale * (Double(index) + 0.5)
            let start = max(Int(center - support + 0.5), 0)
            let count = min(max(min(Int(center + support + 0.5), input) - start, 0), maximumTaps)
            guard count > 0 else { throw GLM5NextVisionError.invalidInput("empty bicubic support") }
            var weights: [Double] = []
            var sum = 0.0
            for offset in 0 ..< count {
                let weight = cubic((Double(offset + start) - center + 0.5) * inverseScale)
                weights.append(weight)
                sum += weight
            }
            if sum != 0 {
                for i in weights.indices {
                    weights[i] /= sum
                    maximumWeight = max(maximumWeight, weights[i])
                }
            }
            rows.append((start, weights))
        }
        var precision = 0
        while precision < 22 {
            let next = Int(0.5 + maximumWeight * Double(1 << (precision + 1)))
            if next >= (1 << 15) { break }
            precision += 1
        }
        guard precision > 0 else {
            throw GLM5NextVisionError.invalidInput("bicubic coefficient precision")
        }
        let multiplier = Double(1 << precision)
        let spans = try rows.map { row in
            let weights = try row.weights.map { weight -> Int16 in
                let scaled = weight * multiplier
                let rounded = Int((scaled < 0 ? -0.5 : 0.5) + scaled)
                guard let result = Int16(exactly: rounded) else {
                    throw GLM5NextVisionError.invalidInput("bicubic coefficient range")
                }
                return result
            }
            return Span(start: row.start, weights: weights)
        }
        return Axis(spans: spans, precision: precision)
    }

    static func resize(
        _ pixels: [UInt8], width: Int, height: Int,
        targetWidth: Int, targetHeight: Int
    ) throws -> [UInt8] {
        let sourceCount = try glm5Product([width, height, 3])
        let outputCount = try glm5Product([targetWidth, targetHeight, 3])
        let intermediateCount = try glm5Product([targetWidth, height, 3])
        guard pixels.count == sourceCount,
            max(sourceCount, max(outputCount, intermediateCount)) <= 64 * 1024 * 1024 * 3,
            max(max(width, height), max(targetWidth, targetHeight)) <= 65536
        else {
            throw GLM5NextVisionError.budgetExceeded("bicubic RGB8 geometry")
        }
        try Task.checkCancellation()
        if width == targetWidth && height == targetHeight { return pixels }
        var horizontal: [UInt8]
        if width == targetWidth {
            horizontal = pixels
        } else {
            let coefficients = try axis(input: width, output: targetWidth)
            horizontal = [UInt8](repeating: 0, count: intermediateCount)
            for y in 0 ..< height {
                if y.isMultiple(of: 16) { try Task.checkCancellation() }
                for x in 0 ..< targetWidth {
                    let span = coefficients.spans[x]
                    for channel in 0 ..< 3 {
                        var sum = Int64(1 << (coefficients.precision - 1))
                        for (offset, weight) in span.weights.enumerated() {
                            sum +=
                                Int64(pixels[(y * width + span.start + offset) * 3 + channel])
                                * Int64(weight)
                        }
                        horizontal[(y * targetWidth + x) * 3 + channel] = UInt8(
                            clamping: sum >> coefficients.precision)
                    }
                }
            }
        }
        if height == targetHeight { return horizontal }
        let coefficients = try axis(input: height, output: targetHeight)
        var result = [UInt8](repeating: 0, count: outputCount)
        for y in 0 ..< targetHeight {
            if y.isMultiple(of: 16) { try Task.checkCancellation() }
            let span = coefficients.spans[y]
            for x in 0 ..< targetWidth {
                for channel in 0 ..< 3 {
                    var sum = Int64(1 << (coefficients.precision - 1))
                    for (offset, weight) in span.weights.enumerated() {
                        sum +=
                            Int64(
                                horizontal[((span.start + offset) * targetWidth + x) * 3 + channel])
                            * Int64(weight)
                    }
                    result[(y * targetWidth + x) * 3 + channel] = UInt8(
                        clamping: sum >> coefficients.precision)
                }
            }
        }
        return result
    }
}

public struct GLM5NextVisionGrid: Sendable, Equatable {
    public let temporal: Int
    public let height: Int
    public let width: Int

    public init(temporal: Int, height: Int, width: Int) throws {
        _ = try glm5Product([temporal, height, width])
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    public func patchCount() throws -> Int { try glm5Product([temporal, height, width]) }
    public func featureCount() throws -> Int {
        guard height.isMultiple(of: 2), width.isMultiple(of: 2) else {
            throw GLM5NextVisionError.invalidInput("grid must divide into 2x2 merge groups")
        }
        return try patchCount() / 4
    }
    public var thw: THW { THW(temporal, height, width) }
}

public struct GLM5NextRGBImage: Sendable {
    public let width: Int
    public let height: Int
    /// Row-major RGB8, top row first, unpremultiplied in encoded sRGB.
    public let pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) throws {
        let count = try glm5Product([width, height, 3])
        guard count <= 64 * 1024 * 1024 * 3, count == pixels.count else {
            throw GLM5NextVisionError.invalidInput("RGB8 pixel count")
        }
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

public struct GLM5NextDecodedVideo: Sendable {
    public let frames: [GLM5NextRGBImage]
    public let framesPerSecond: Double
    public let duration: Double?
    /// Frame count of the source; frame i is timed at i / `framesPerSecond`.
    public let sourceFrameCount: Int
    /// Ascending source index of each retained frame; nil when `frames` is the
    /// complete source.
    public let sourceIndices: [Int]?

    /// Frames are the complete source in presentation order, before GLM sampling.
    public init(frames: [GLM5NextRGBImage], framesPerSecond: Double, duration: Double? = nil) throws
    {
        try self.init(
            frames: frames, framesPerSecond: framesPerSecond, duration: duration,
            sourceFrameCount: frames.count, sourceIndices: nil)
    }

    /// Frames are the retained `sourceIndices` of a `sourceFrameCount`-frame
    /// source, so unsampled frames are never kept.
    init(
        frames: [GLM5NextRGBImage], framesPerSecond: Double, duration: Double?,
        sourceFrameCount: Int, sourceIndices: [Int]?
    ) throws {
        let indicesValid =
            sourceIndices.map { indices in
                indices.count == frames.count
                    && indices.indices.allSatisfy {
                        indices[$0] >= 0 && indices[$0] < sourceFrameCount
                            && ($0 == 0 || indices[$0 - 1] < indices[$0])
                    }
            } ?? (sourceFrameCount == frames.count)
        guard !frames.isEmpty, framesPerSecond.isFinite, framesPerSecond > 0,
            duration.map({ $0.isFinite && $0 > 0 }) ?? true, indicesValid,
            frames.allSatisfy({ $0.width == frames[0].width && $0.height == frames[0].height })
        else {
            throw GLM5NextVisionError.invalidInput("video frame metadata")
        }
        self.frames = frames
        self.framesPerSecond = framesPerSecond
        self.duration = duration
        self.sourceFrameCount = sourceFrameCount
        self.sourceIndices = sourceIndices
    }
}

/// `.official` applies the pinned 2 FPS sampler. `.disabled` corresponds to
/// Transformers `do_sample_frames=false`; supplied frames still undergo the
/// same geometry, duration, frame-count and byte-limit validation.
public enum GLM5NextVideoSamplingPolicy: Sendable, Equatable {
    case official
    case disabled
}

/// Pure geometry/sampling operations: these functions do not create MLX arrays.
public enum GLM5NextMediaGeometry {
    public struct ResizePlan: Sendable, Equatable {
        public let width: Int
        public let height: Int
        public let contentWidth: Int
        public let contentHeight: Int
        public let grid: GLM5NextVisionGrid
    }

    public static func resizePlan(
        width: Int, height: Int, frames: Int = 2,
        patchSize: Int = 14, minimumTokens: Int = 16, maximumTokens: Int = 8000
    ) throws -> ResizePlan {
        let original = try glm5Product([frames, height, width])
        let factor = try glm5Product([patchSize, 2])
        guard width <= 65536, height <= 65536, frames <= 100_000,
            minimumTokens > 0, maximumTokens >= minimumTokens, maximumTokens <= 240_000
        else {
            throw GLM5NextVisionError.invalidInput("resize dimensions/token budget")
        }
        let pixelsPerToken = try glm5Product([2, factor, factor])
        let minimum = try glm5Product([minimumTokens, pixelsPerToken])
        let maximum = try glm5Product([maximumTokens, pixelsPerToken])
        func align(_ value: Int) -> Int { ((value + factor - 1) / factor) * factor }
        let alignedFrames = max(2, Int((Double(frames) / 2).rounded(.toNearestOrEven)) * 2)
        var h = align(height)
        var w = align(width)
        if try glm5Product([alignedFrames, h, w]) < minimum {
            let scale = sqrt(Double(minimum) / Double(original))
            h = align(max(1, Int(ceil(Double(height) * scale))))
            w = align(max(1, Int(ceil(Double(width) * scale))))
        }
        if try glm5Product([alignedFrames, h, w]) > maximum {
            guard try glm5Product([alignedFrames, factor, factor]) <= maximum else {
                throw GLM5NextVisionError.budgetExceeded(
                    "not enough pixels for one aligned video patch")
            }
            var low = 1
            var high = height
            h = factor
            w = factor
            while low <= high {
                let middle = low + (high - low) / 2
                let candidateH = align(middle)
                let candidateW = align(
                    max(1, Int(floor(Double(width) * Double(middle) / Double(height)))))
                if try glm5Product([alignedFrames, candidateH, candidateW]) <= maximum {
                    h = candidateH
                    w = candidateW
                    low = middle + 1
                } else {
                    high = middle - 1
                }
            }
        }
        var scale = min(Double(h) / Double(height), Double(w) / Double(width))
        if original >= minimum { scale = min(1, scale) }
        let contentH = max(1, min(h, Int(floor(Double(height) * scale))))
        let contentW = max(1, min(w, Int(floor(Double(width) * scale))))
        return ResizePlan(
            width: w, height: h, contentWidth: contentW, contentHeight: contentH,
            grid: try .init(temporal: (frames + 1) / 2, height: h / patchSize, width: w / patchSize)
        )
    }

    public static func sampleFrameIndices(
        totalFrames: Int, sourceFPS: Double, duration: Double? = nil,
        targetFPS: Double = 2, maximumFrames: Int = 2048
    ) throws -> [Int] {
        guard totalFrames > 0, totalFrames <= 100_000, sourceFPS.isFinite, sourceFPS > 0,
            targetFPS.isFinite, targetFPS > 0, maximumFrames > 0, maximumFrames <= 2048,
            duration.map({ $0.isFinite && $0 > 0 }) ?? true
        else {
            throw GLM5NextVisionError.invalidInput("frame sampling metadata")
        }
        let seconds =
            duration ?? (Double(totalFrames - 1) / sourceFPS).rounded(.toNearestOrEven) + 1
        guard seconds < Double(Int.max), seconds * targetFPS >= 1,
            (seconds * targetFPS).isFinite
        else {
            throw GLM5NextVisionError.invalidInput("video too short or unbounded duration")
        }
        let count = Int(min(Double(maximumFrames), floor(seconds * targetFPS)))
        func linspace(_ first: Int, _ last: Int) -> [Int] {
            if count == 1 { return [first] }
            return (0 ..< count).map {
                Int(floor(Double(first) + Double(last - first) * Double($0) / Double(count - 1)))
            }
        }
        var indices: [Int] = []
        if totalFrames < count {
            indices = linspace(0, totalFrames - 1)
        } else {
            var next = 0.0
            for i in 0 ..< totalFrames where Double(i) / sourceFPS >= next {
                next += 1 / targetFPS
                indices.append(i)
                if next >= Double(Int(seconds)) { break }
            }
        }
        if indices.count < count {
            indices = linspace(indices.first ?? 0, indices.last ?? (totalFrames - 1))
        } else if indices.count > count {
            indices = linspace(0, totalFrames - 1)
        }
        var seen = Set<Int>()
        indices = indices.filter { seen.insert($0).inserted }
        if indices.count % 2 == 1, let last = indices.last { indices.append(last) }
        guard !indices.isEmpty, indices.count <= maximumFrames + 1 else {
            throw GLM5NextVisionError.invalidInput("empty sampled video")
        }
        return indices
    }
}

public struct GLM5NextProcessorConfiguration: Sendable {
    public let patchSize: Int
    public let imageMinimumTokens: Int
    public let imageMaximumTokens: Int
    public let videoMinimumTokens: Int
    public let videoMaximumTokens: Int
    public let maximumDecodedBytes: Int
    public let maximumPreparedBytes: Int
    /// Default `.official` resamples the source at 2 FPS. `.disabled` retains
    /// every supplied frame, then pads only the final temporal pair after resize.
    public let videoSamplingPolicy: GLM5NextVideoSamplingPolicy

    public init(
        patchSize: Int = 14, imageMinimumTokens: Int = 16, imageMaximumTokens: Int = 8000,
        videoMinimumTokens: Int = 16, videoMaximumTokens: Int = 240000,
        maximumDecodedBytes: Int = 512 * 1024 * 1024, maximumPreparedBytes: Int = 512 * 1024 * 1024,
        videoSamplingPolicy: GLM5NextVideoSamplingPolicy = .official
    ) throws {
        guard patchSize > 0, patchSize <= 128, imageMinimumTokens > 0,
            imageMaximumTokens >= imageMinimumTokens, imageMaximumTokens <= 8000,
            videoMinimumTokens > 0, videoMaximumTokens >= videoMinimumTokens,
            videoMaximumTokens <= 240000,
            maximumDecodedBytes > 0, maximumPreparedBytes > 0
        else {
            throw GLM5NextVisionError.invalidInput("processor limits")
        }
        self.patchSize = patchSize
        self.imageMinimumTokens = imageMinimumTokens
        self.imageMaximumTokens = imageMaximumTokens
        self.videoMinimumTokens = videoMinimumTokens
        self.videoMaximumTokens = videoMaximumTokens
        self.maximumDecodedBytes = maximumDecodedBytes
        self.maximumPreparedBytes = maximumPreparedBytes
        self.videoSamplingPolicy = videoSamplingPolicy
    }
}

/// One immutable preparation plan is shared by aggregate quotation and execution.
/// Frame indices describe the input to resize, before patchify's final-frame copy.
struct GLM5NextVideoPreparationPlan: Sendable {
    /// Source frame indices; they also give each temporal pair's timestamp.
    let frameIndices: [Int]
    /// Position of each `frameIndices` entry in `GLM5NextDecodedVideo.frames`.
    let framePositions: [Int]
    let resize: GLM5NextMediaGeometry.ResizePlan
    let preparedBytes: Int
}

public struct GLM5NextPreparedMedia {
    public let patches: MLXArray
    public let grid: GLM5NextVisionGrid
    /// One timestamp per temporal pair; empty for still images.
    public let timestamps: [Double]
}

/// One item's prepared patches as host values (`GLM5NextProcessor.prepareMediaValues`).
public struct GLM5NextPreparedMediaValues: Sendable {
    /// Row-major float32 patches, `grid.patchCount()` rows of `6 * patchSize²`.
    public let values: [Float]
    public let grid: GLM5NextVisionGrid
    /// One timestamp per temporal pair; empty for still images.
    public let timestamps: [Double]
}

public struct GLM5NextMediaPlacement: Sendable, Equatable {
    public enum Modality: Sendable, Equatable { case image, video }
    public let modality: Modality
    public let itemIndex: Int
    public let tokenRange: Range<Int>
    /// Relative to the concatenated features of this modality, not to the other modality.
    public let featureRange: Range<Int>
}

public struct GLM5NextPreparedInput {
    public let input: LMInput
    public let placements: [GLM5NextMediaPlacement]
}

public struct GLM5NextTokenBatch: Sendable {
    public let tokens: [[Int]]
    public let mask: [[Bool]]
    /// Feature ranges remain row-local. Token positions include each row's left padding.
    public let placements: [[GLM5NextMediaPlacement]]
}

/// Standalone official-prompt processor. A parent model must inject the returned
/// visual features at `placements`; no text model or factory is assumed here.
public struct GLM5NextProcessor: UserInputProcessor {
    public let configuration: GLM5NextProcessorConfiguration
    private let tokenizer: (any Tokenizer)?
    public static let imageToken = 154854
    public static let videoToken = 154855
    public static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    public static let standardDeviation: [Float] = [0.26862954, 0.26130258, 0.27577711]

    public init(configuration: GLM5NextProcessorConfiguration, tokenizer: any Tokenizer) throws {
        for (text, id) in [
            ("<|image|>", Self.imageToken), ("<|video|>", Self.videoToken),
            ("<|begin_of_image|>", 154830), ("<|end_of_image|>", 154831),
            ("<|begin_of_video|>", 154832), ("<|end_of_video|>", 154833),
        ] {
            guard tokenizer.convertTokenToId(text) == id else {
                throw GLM5NextVisionError.invalidInput("tokenizer media-token identity")
            }
        }
        self.configuration = configuration
        self.tokenizer = tokenizer
    }

    /// Image-only worker preparation uses the same native resize/patchify path.
    /// Prompt expansion remains with the authenticated host tokenizer.
    public init(imageConfiguration: GLM5NextProcessorConfiguration) {
        configuration = imageConfiguration
        tokenizer = nil
    }

    private func plan(_ image: GLM5NextRGBImage, frames: Int, video: Bool) throws
        -> GLM5NextMediaGeometry.ResizePlan
    {
        let c = configuration
        let plan = try GLM5NextMediaGeometry.resizePlan(
            width: image.width, height: image.height,
            frames: frames, patchSize: c.patchSize,
            minimumTokens: video ? c.videoMinimumTokens : c.imageMinimumTokens,
            maximumTokens: video ? c.videoMaximumTokens : c.imageMaximumTokens)
        let bytes = try glm5Product([plan.grid.patchCount(), 3, 2, c.patchSize, c.patchSize, 4])
        guard bytes <= c.maximumPreparedBytes else {
            throw GLM5NextVisionError.budgetExceeded("prepared pixels")
        }
        return plan
    }

    private func resize(_ image: GLM5NextRGBImage, plan: GLM5NextMediaGeometry.ResizePlan) throws
        -> [UInt8]
    {
        let count = try glm5Product([plan.width, plan.height, 3])
        let contentCount = try glm5Product([plan.contentWidth, plan.contentHeight, 3])
        let intermediateCount = try glm5Product([plan.contentWidth, image.height, 3])
        guard count <= configuration.maximumDecodedBytes,
            contentCount <= configuration.maximumDecodedBytes - count,
            intermediateCount <= configuration.maximumDecodedBytes - count - contentCount
        else {
            throw GLM5NextVisionError.budgetExceeded("resized RGB canvases")
        }
        // Match the reference's uint8 Torchvision backend, not Pillow's different
        // coefficient precision or a linear-light CoreImage resampling path.
        let content = try GLM5NextRGB8Bicubic.resize(
            image.pixels, width: image.width, height: image.height,
            targetWidth: plan.contentWidth, targetHeight: plan.contentHeight)
        var result = [UInt8](repeating: 0, count: count)
        for y in 0 ..< plan.contentHeight {
            try Task.checkCancellation()
            result.replaceSubrange(
                (y * plan.width * 3) ..< ((y * plan.width + plan.contentWidth) * 3),
                with: content[(y * plan.contentWidth * 3) ..< ((y + 1) * plan.contentWidth * 3)])
        }
        return result
    }

    private func appendPair(
        _ first: [UInt8], _ second: [UInt8], plan: GLM5NextMediaGeometry.ResizePlan,
        to result: inout [Float]
    ) throws {
        let p = configuration.patchSize
        // TorchvisionBackend fuses rescale+normalize in byte space. Preserve
        // its Float32 multiply/subtract/divide order, including black padding.
        let mean = Self.mean.map { $0 * 255 }
        let standardDeviation = Self.standardDeviation.map { $0 * 255 }
        // Group-row, group-column, sub-row, sub-column, C,T,patch-H,patch-W.
        // This is the reference's permute before flattening, not raster patch order.
        for gy in stride(from: 0, to: plan.grid.height, by: 2) {
            try Task.checkCancellation()
            for gx in stride(from: 0, to: plan.grid.width, by: 2) {
                for dy in 0 ..< 2 {
                    for dx in 0 ..< 2 {
                        for channel in 0 ..< 3 {
                            for pixels in [first, second] {
                                for py in 0 ..< p {
                                    for px in 0 ..< p {
                                        let y = (gy + dy) * p + py
                                        let x = (gx + dx) * p + px
                                        let value = Float(
                                            pixels[(y * plan.width + x) * 3 + channel])
                                        result.append(
                                            (value - mean[channel]) / standardDeviation[channel])
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public func prepareImage(
        _ image: GLM5NextRGBImage,
        checkpoint: ([MLXArray]) throws -> Void = { _ in }
    ) throws -> GLM5NextPreparedMedia {
        let plan = try plan(image, frames: 2, video: false)
        let pixels = try resize(image, plan: plan)
        var values: [Float] = []
        values.reserveCapacity(
            try glm5Product([
                plan.grid.patchCount(), 6,
                configuration.patchSize, configuration.patchSize,
            ]))
        try appendPair(pixels, pixels, plan: plan, to: &values)
        return try makeMedia(values, grid: plan.grid, timestamps: [], checkpoint: checkpoint)
    }

    func videoPreparationPlan(_ video: GLM5NextDecodedVideo) throws -> GLM5NextVideoPreparationPlan
    {
        // Validate the source duration/frame metadata in both modes. Disabling
        // sampling must not disable the existing source or retained-frame caps.
        let sampled = try GLM5NextMediaGeometry.sampleFrameIndices(
            totalFrames: video.sourceFrameCount,
            sourceFPS: video.framesPerSecond, duration: video.duration)
        let indices: [Int]
        switch configuration.videoSamplingPolicy {
        case .official:
            indices = sampled
        case .disabled:
            guard video.sourceIndices == nil else {
                throw GLM5NextVisionError.invalidInput(
                    "disabled sampling requires every source frame")
            }
            guard video.frames.count <= 2048 else {
                throw GLM5NextVisionError.budgetExceeded("retained video frames")
            }
            indices = Array(video.frames.indices)
        }
        let positions: [Int]
        if let retained = video.sourceIndices {
            let position = Dictionary(
                uniqueKeysWithValues: retained.enumerated().map { ($0.element, $0.offset) })
            positions = try indices.map {
                guard let value = position[$0] else {
                    throw GLM5NextVisionError.invalidInput("sampled video frame was not retained")
                }
                return value
            }
        } else {
            positions = indices
        }
        var decodedBytes = 0
        for frame in video.frames {
            guard frame.pixels.count <= configuration.maximumDecodedBytes - decodedBytes else {
                throw GLM5NextVisionError.budgetExceeded("decoded source video")
            }
            decodedBytes += frame.pixels.count
        }
        let resize = try plan(video.frames[0], frames: indices.count, video: true)
        // For odd input counts, nearest-even resize alignment can be below the
        // actual padded frame count. Preserve the reference geometry, but refuse
        // an output that exceeds the configured retained-token budget.
        guard try resize.grid.featureCount() <= configuration.videoMaximumTokens else {
            throw GLM5NextVisionError.budgetExceeded("packed video tokens")
        }
        // The grid uses ceil(frameCount/2), independently of smart_resize's
        // nearest-even aligned count. Charge every physically packed frame.
        let bytes = try glm5Product([
            resize.grid.patchCount(), 6,
            configuration.patchSize, configuration.patchSize, 4,
        ])
        return .init(
            frameIndices: indices, framePositions: positions, resize: resize, preparedBytes: bytes)
    }

    public func prepareVideo(_ video: GLM5NextDecodedVideo) throws -> GLM5NextPreparedMedia {
        try prepareVideo(video, plan: videoPreparationPlan(video))
    }

    private func prepareVideo(_ video: GLM5NextDecodedVideo, plan: GLM5NextVideoPreparationPlan)
        throws -> GLM5NextPreparedMedia
    {
        let indices = plan.frameIndices
        let positions = plan.framePositions
        var values: [Float] = []
        values.reserveCapacity(plan.preparedBytes / MemoryLayout<Float>.size)
        var times: [Double] = []
        for i in stride(from: 0, to: indices.count, by: 2) {
            try Task.checkCancellation()
            let first = try resize(video.frames[positions[i]], plan: plan.resize)
            // Match patchify: an odd final frame is copied only after resize.
            let second: [UInt8]
            if i + 1 < indices.count {
                second = try resize(video.frames[positions[i + 1]], plan: plan.resize)
            } else {
                second = first
            }
            try appendPair(first, second, plan: plan.resize, to: &values)
            times.append(Double(indices[i]) / video.framesPerSecond)
        }
        return try makeMedia(values, grid: plan.resize.grid, timestamps: times)
    }

    private func makeMedia(
        _ values: [Float], grid: GLM5NextVisionGrid, timestamps: [Double],
        checkpoint: ([MLXArray]) throws -> Void = { _ in }
    ) throws
        -> GLM5NextPreparedMedia
    {
        let count = try grid.patchCount()
        return try MLX.withError { error in
            let patches = MLXArray(values).reshaped(
                count, 6 * configuration.patchSize * configuration.patchSize)
            try checkpoint([patches])
            eval(patches)
            try error.check()
            return .init(patches: patches, grid: grid, timestamps: timestamps)
        }
    }

    /// Host-only token planning; useful to preserve each row's positions before
    /// a parent batcher applies left padding. It never decodes/re-encodes the prompt.
    public func expandPrompt(
        _ tokens: [Int], images: [GLM5NextVisionGrid],
        videos: [(grid: GLM5NextVisionGrid, timestamps: [Double])]
    ) throws -> (tokens: [Int], placements: [GLM5NextMediaPlacement]) {
        guard tokens.count <= 1_048_576 else {
            throw GLM5NextVisionError.budgetExceeded("prompt tokens")
        }
        let featureBytes = try glm5Product([
            4, 6, configuration.patchSize, configuration.patchSize, 4,
        ])
        let maximumFeatures = configuration.maximumPreparedBytes / featureBytes
        var output: [Int] = []
        var placements: [GLM5NextMediaPlacement] = []
        var imageIndex = 0
        var videoIndex = 0
        var imageFeatures = 0
        var videoFeatures = 0
        for token in tokens {
            switch token {
            case Self.imageToken:
                guard imageIndex < images.count else {
                    throw GLM5NextVisionError.invalidInput("extra image marker")
                }
                let count = try images[imageIndex].featureCount()
                let start = output.count
                guard count <= maximumFeatures - imageFeatures - videoFeatures,
                    count <= 1_048_576 - output.count
                else {
                    throw GLM5NextVisionError.budgetExceeded("image prompt expansion")
                }
                output.append(contentsOf: repeatElement(Self.imageToken, count: count))
                placements.append(
                    .init(
                        modality: .image, itemIndex: imageIndex,
                        tokenRange: start ..< output.count,
                        featureRange: imageFeatures ..< (imageFeatures + count)))
                imageFeatures += count
                imageIndex += 1
            case Self.videoToken:
                guard videoIndex < videos.count else {
                    throw GLM5NextVisionError.invalidInput("extra video marker")
                }
                let video = videos[videoIndex]
                guard video.timestamps.count == video.grid.temporal,
                    video.timestamps.allSatisfy({ $0.isFinite && $0 >= 0 })
                else {
                    throw GLM5NextVisionError.invalidInput("video timestamp count")
                }
                let count = try video.grid.featureCount() / video.grid.temporal
                guard
                    try video.grid.featureCount() <= maximumFeatures - imageFeatures - videoFeatures
                else {
                    throw GLM5NextVisionError.budgetExceeded("video prompt expansion")
                }
                for time in video.timestamps {
                    guard let tokenizer else {
                        throw GLM5NextVisionError.invalidInput("video requires tokenizer")
                    }
                    let timestampTokens = tokenizer.encode(
                        text:
                            String(
                                format: "%.1f seconds", locale: Locale(identifier: "en_US_POSIX"),
                                time),
                        addSpecialTokens: false)
                    guard count + 2 + timestampTokens.count <= 1_048_576 - output.count else {
                        throw GLM5NextVisionError.budgetExceeded("video prompt expansion")
                    }
                    output.append(154830)
                    let start = output.count
                    output.append(contentsOf: repeatElement(Self.imageToken, count: count))
                    placements.append(
                        .init(
                            modality: .video, itemIndex: videoIndex,
                            tokenRange: start ..< output.count,
                            featureRange: videoFeatures ..< (videoFeatures + count)))
                    videoFeatures += count
                    output.append(154831)
                    output.append(contentsOf: timestampTokens)
                }
                videoIndex += 1
            default: output.append(token)
            }
        }
        guard imageIndex == images.count, videoIndex == videos.count else {
            throw GLM5NextVisionError.invalidInput("media count does not match rendered prompt")
        }
        guard output.count <= 1_048_576 else {
            throw GLM5NextVisionError.budgetExceeded("expanded prompt tokens")
        }
        return (output, placements)
    }

    public static func leftPad(
        rows: [(tokens: [Int], placements: [GLM5NextMediaPlacement])],
        padToken: Int = 154820
    ) throws -> GLM5NextTokenBatch {
        guard !rows.isEmpty, rows.allSatisfy({ !$0.tokens.isEmpty }), padToken >= 0 else {
            throw GLM5NextVisionError.invalidInput("empty token batch or invalid padding")
        }
        let width = rows.map { $0.tokens.count }.max()!
        var tokens: [[Int]] = []
        var masks: [[Bool]] = []
        var placements: [[GLM5NextMediaPlacement]] = []
        for row in rows {
            let padding = width - row.tokens.count
            var shifted: [GLM5NextMediaPlacement] = []
            for placement in row.placements {
                guard placement.tokenRange.lowerBound >= 0,
                    placement.tokenRange.upperBound <= row.tokens.count,
                    placement.featureRange.lowerBound >= 0,
                    placement.featureRange.count == placement.tokenRange.count,
                    row.tokens[placement.tokenRange].allSatisfy({ $0 == imageToken })
                else {
                    throw GLM5NextVisionError.invalidInput("media token placement")
                }
                shifted.append(
                    .init(
                        modality: placement.modality, itemIndex: placement.itemIndex,
                        tokenRange: (placement.tokenRange.lowerBound + padding)
                            ..< (placement.tokenRange
                                .upperBound + padding),
                        featureRange: placement.featureRange))
            }
            tokens.append(Array(repeating: padToken, count: padding) + row.tokens)
            masks.append(
                Array(repeating: false, count: padding)
                    + Array(repeating: true, count: row.tokens.count))
            placements.append(shifted)
        }
        return .init(tokens: tokens, mask: masks, placements: placements)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        (try await prepareDetailed(input: input)).input
    }

    public func prepareDetailed(input: UserInput) async throws -> GLM5NextPreparedInput {
        guard input.processing.resize == nil, input.processing.minPixels == nil,
            input.processing.maxPixels == nil
        else {
            throw GLM5NextVisionError.unsupported(
                "pixel overrides: configure explicit GLM token budgets instead")
        }
        guard let tokenizer else {
            throw GLM5NextVisionError.invalidInput("prompt preparation requires tokenizer")
        }
        let messages = try Self.messages(input)
        let originalTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)
        guard originalTokens.filter({ $0 == Self.imageToken }).count == input.images.count,
            originalTokens.filter({ $0 == Self.videoToken }).count == input.videos.count
        else {
            throw GLM5NextVisionError.invalidInput("media count does not match rendered prompt")
        }
        var images: [GLM5NextPreparedMedia] = []
        var videos: [GLM5NextPreparedMedia] = []
        var retainedBytes = 0
        func charge(_ bytes: Int) throws {
            guard bytes <= configuration.maximumPreparedBytes - retainedBytes else {
                throw GLM5NextVisionError.budgetExceeded("aggregate prepared media")
            }
            retainedBytes += bytes
        }
        for image in input.images {
            try Task.checkCancellation()
            let decoded = try Self.decode(image, maximumBytes: configuration.maximumDecodedBytes)
            let grid = try plan(decoded, frames: 2, video: false).grid
            try charge(
                glm5Product([
                    grid.patchCount(), 6, configuration.patchSize, configuration.patchSize, 4,
                ]))
            images.append(try prepareImage(decoded))
        }
        for video in input.videos {
            try Task.checkCancellation()
            let decoded = try await decode(video)
            let plan = try videoPreparationPlan(decoded)
            try charge(plan.preparedBytes)
            videos.append(try prepareVideo(decoded, plan: plan))
        }
        let expansion = try expandPrompt(
            originalTokens, images: images.map(\.grid),
            videos: videos.map { ($0.grid, $0.timestamps) })
        return try MLX.withError { error in
            let tokens = MLXArray(expansion.tokens).expandedDimensions(axis: 0)
            let mask = MLXArray.ones(tokens.shape, dtype: .bool)
            let imagePixels = images.isEmpty ? nil : concatenated(images.map(\.patches), axis: 0)
            let videoPixels = videos.isEmpty ? nil : concatenated(videos.map(\.patches), axis: 0)
            eval([tokens, mask] + [imagePixels, videoPixels].compactMap { $0 })
            try error.check()
            return .init(
                input: LMInput(
                    text: .init(tokens: tokens, mask: mask),
                    image: imagePixels.map {
                        .init(pixels: $0, frames: images.map { $0.grid.thw })
                    },
                    video: videoPixels.map { .init(pixels: $0, frames: videos.map { $0.grid.thw }) }
                ),
                placements: expansion.placements)
        }
    }

    /// The media half of `prepareDetailed(input:)` for a host that renders no
    /// prompt (a cluster rank whose leader expanded it), as host values: the
    /// same decode, plan, aggregate prepared-byte charge, resize and patchify,
    /// every image and then every video in input order. Each item's values are
    /// the row-major float32 patches `prepareDetailed` puts in its arrays
    /// (`MLXArray(values).reshaped(patchCount, 6·p·p)`), with the same grid and
    /// timestamps. No MLX array is made, so a caller may run it beside MLX
    /// work on another thread. Keep in step with `prepareImage` and
    /// `prepareVideo`; the cluster compares the patches' digests with its
    /// leader's `prepareDetailed` on every request.
    public func prepareMediaValues(images: [UserInput.Image], videos: [UserInput.Video])
        async throws
        -> (images: [GLM5NextPreparedMediaValues], videos: [GLM5NextPreparedMediaValues])
    {
        var preparedImages: [GLM5NextPreparedMediaValues] = []
        var preparedVideos: [GLM5NextPreparedMediaValues] = []
        var retainedBytes = 0
        func charge(_ bytes: Int) throws {
            guard bytes <= configuration.maximumPreparedBytes - retainedBytes else {
                throw GLM5NextVisionError.budgetExceeded("aggregate prepared media")
            }
            retainedBytes += bytes
        }
        for image in images {
            try Task.checkCancellation()
            let decoded = try Self.decode(image, maximumBytes: configuration.maximumDecodedBytes)
            let plan = try plan(decoded, frames: 2, video: false)
            try charge(
                glm5Product([
                    plan.grid.patchCount(), 6, configuration.patchSize, configuration.patchSize, 4,
                ]))
            // `prepareImage`'s values.
            let pixels = try resize(decoded, plan: plan)
            var values: [Float] = []
            values.reserveCapacity(
                try glm5Product([
                    plan.grid.patchCount(), 6, configuration.patchSize, configuration.patchSize,
                ]))
            try appendPair(pixels, pixels, plan: plan, to: &values)
            preparedImages.append(.init(values: values, grid: plan.grid, timestamps: []))
        }
        for video in videos {
            try Task.checkCancellation()
            let decoded = try await decode(video)
            let plan = try videoPreparationPlan(decoded)
            try charge(plan.preparedBytes)
            // `prepareVideo(_:plan:)`'s values.
            let indices = plan.frameIndices
            let positions = plan.framePositions
            var values: [Float] = []
            values.reserveCapacity(plan.preparedBytes / MemoryLayout<Float>.size)
            var times: [Double] = []
            for i in stride(from: 0, to: indices.count, by: 2) {
                try Task.checkCancellation()
                let first = try resize(decoded.frames[positions[i]], plan: plan.resize)
                let second: [UInt8]
                if i + 1 < indices.count {
                    second = try resize(decoded.frames[positions[i + 1]], plan: plan.resize)
                } else {
                    second = first
                }
                try appendPair(first, second, plan: plan.resize, to: &values)
                times.append(Double(indices[i]) / decoded.framesPerSecond)
            }
            preparedVideos.append(.init(values: values, grid: plan.resize.grid, timestamps: times))
        }
        return (preparedImages, preparedVideos)
    }

    private static func rejectAudio(_ value: Any) throws {
        if let object = value as? [String: Any] {
            if let type = object["type"] as? String,
                ["audio", "audio_url", "input_audio"].contains(type)
            {
                throw GLM5NextVisionError.unsupported("audio input")
            }
            for item in object.values { try rejectAudio(item) }
        } else if let array = value as? [Any] {
            for item in array { try rejectAudio(item) }
        }
    }

    private static func messages(_ input: UserInput) throws -> [Message] {
        let result: [Message]
        switch input.prompt {
        case .messages(let messages): result = messages
        case .text(let text):
            result = [
                makeMessage(
                    role: "user", text: text, imageCount: input.images.count,
                    videoCount: input.videos.count)
            ]
        case .chat(let messages):
            result = messages.map { message in
                var result = makeMessage(
                    role: message.role.rawValue, text: message.content,
                    imageCount: message.images.count, videoCount: message.videos.count)
                for key in ["name", "tool_call_id", "tool_calls", "reasoning_content"] {
                    if let value = message.templateFields[key] { result[key] = value }
                }
                return result
            }
        }
        for message in result { try rejectAudio(message) }
        return result
    }

    private static func makeMessage(role: String, text: String, imageCount: Int, videoCount: Int)
        -> Message
    {
        var content: [[String: any Sendable]] = [["type": "text", "text": text]]
        content += (0 ..< imageCount).map { _ in ["type": "image"] }
        content += (0 ..< videoCount).map { _ in ["type": "video"] }
        return ["role": role, "content": content]
    }

    private static func decode(_ image: UserInput.Image, maximumBytes: Int) throws
        -> GLM5NextRGBImage
    {
        switch image.source {
        case .ciImage(let image):
            return try GLM5NextRGBDecoder.render(image, maximumBytes: maximumBytes)
        case .url(let url):
            guard url.isFileURL, let image = CIImage(contentsOf: url) else {
                throw GLM5NextVisionError.unsupported("nonlocal or invalid image URL")
            }
            return try GLM5NextRGBDecoder.render(image, maximumBytes: maximumBytes)
        case .array(let array):
            guard array.ndim == 3, array.dtype == .uint8 else {
                throw GLM5NextVisionError.unsupported("image arrays must be RGB8 HWC or CHW")
            }
            let isCHW = array.dim(0) == 3
            let isHWC = array.dim(2) == 3
            guard isCHW != isHWC else {
                throw GLM5NextVisionError.invalidInput("ambiguous RGB array layout")
            }
            let pixels = isCHW ? array.transposed(1, 2, 0) : array
            guard try glm5Product(pixels.shape) <= maximumBytes else {
                throw GLM5NextVisionError.budgetExceeded("RGB array")
            }
            return try MLX.withError { error in
                eval(pixels)
                try error.check()
                let bytes = pixels.asArray(UInt8.self)
                try error.check()
                return try .init(width: pixels.dim(1), height: pixels.dim(0), pixels: bytes)
            }
        }
    }

    private func decode(_ video: UserInput.Video) async throws -> GLM5NextDecodedVideo {
        switch video.source {
        case .frames(let frames):
            guard frames.count >= 2, frames.count <= 100_000 else {
                throw GLM5NextVisionError.invalidInput(
                    "frames require CFR timestamps and 2...100000 source frames")
            }
            let first = frames[0].timeStamp.seconds
            let step = frames[1].timeStamp.seconds - first
            guard first.isFinite, abs(first) < 1e-6, step.isFinite, step > 0,
                frames.enumerated().allSatisfy({
                    abs($0.element.timeStamp.seconds - Double($0.offset) * step) < 1e-4
                })
            else {
                throw GLM5NextVisionError.unsupported("nonuniform video frame timestamps")
            }
            var images: [GLM5NextRGBImage] = []
            var bytes = 0
            for frame in frames {
                let image = try GLM5NextRGBDecoder.render(
                    frame.frame, maximumBytes: configuration.maximumDecodedBytes - bytes)
                guard image.pixels.count <= configuration.maximumDecodedBytes - bytes else {
                    throw GLM5NextVisionError.budgetExceeded("decoded video frames")
                }
                bytes += image.pixels.count
                images.append(image)
            }
            return try .init(frames: images, framesPerSecond: 1 / step)
        case .avAsset(let asset): return try await decodeAsset(asset)
        case .memoryBacked(let owner):
            return try await owner.withAsset { try await decodeAsset($0) }
        case .url(let url):
            guard url.isFileURL else { throw GLM5NextVisionError.unsupported("nonlocal video URL") }
            return try await decodeAsset(AVURLAsset(url: url))
        }
    }

    private func decodeAsset(_ asset: AVAsset) async throws -> GLM5NextDecodedVideo {
        try Task.checkCancellation()
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard tracks.count == 1, let track = tracks.first else {
            throw GLM5NextVisionError.unsupported(
                "requires exactly one video track; audio-only input is unsupported")
        }
        let fps = Double(try await track.load(.nominalFrameRate))
        let duration = try await track.load(.timeRange).duration.seconds
        let transform = try await track.load(.preferredTransform)
        guard fps.isFinite, fps > 0, duration.isFinite, duration > 0,
            duration * fps <= 100_000
        else { throw GLM5NextVisionError.invalidInput("video track metadata") }
        guard configuration.videoSamplingPolicy == .official else {
            let all = try decodeFrames(asset, track: track, transform: transform, retaining: nil)
            return try .init(frames: all.frames, framesPerSecond: fps, duration: duration)
        }
        // The official sampler needs only the source frame count, so decoding
        // retains just its frames: first for the count the track metadata
        // predicts, and again only if the decoded count needs other frames.
        func sampled(_ count: Int) throws -> Set<Int> {
            Set(
                try GLM5NextMediaGeometry.sampleFrameIndices(
                    totalFrames: count, sourceFPS: fps, duration: duration))
        }
        var pass = try decodeFrames(
            asset, track: track, transform: transform,
            retaining: sampled(max(1, Int((duration * fps).rounded()))))
        let needed = try sampled(pass.decoded)
        if !needed.isSubset(of: pass.indices) {
            let count = pass.decoded
            pass = try decodeFrames(asset, track: track, transform: transform, retaining: needed)
            guard pass.decoded == count else {
                throw GLM5NextVisionError.invalidInput("video decoding is not repeatable")
            }
        }
        return try .init(
            frames: pass.frames, framesPerSecond: fps, duration: duration,
            sourceFrameCount: pass.decoded, sourceIndices: pass.indices)
    }

    /// Decodes every frame in presentation order, but converts and retains
    /// only `retaining` (every frame when nil). Timestamps need only increase:
    /// like the reference loader, frame i is timed at i / `nominalFrameRate`,
    /// the track's average rate, so a variable-frame-rate track is indexed
    /// rather than refused.
    private func decodeFrames(
        _ asset: AVAsset, track: AVAssetTrack, transform: CGAffineTransform,
        retaining retained: Set<Int>?
    ) throws -> (frames: [GLM5NextRGBImage], indices: [Int], decoded: Int) {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw GLM5NextVisionError.unsupported("video decoder format")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw GLM5NextVisionError.invalidInput("video decoder start")
        }
        defer { reader.cancelReading() }
        var frames: [GLM5NextRGBImage] = []
        var indices: [Int] = []
        var decoded = 0
        var bytes = 0
        var previous: Double?
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixels = CMSampleBufferGetImageBuffer(sample), decoded < 100_000 else {
                throw GLM5NextVisionError.invalidInput("video decoder sample")
            }
            let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            guard time.isFinite, previous.map({ time > $0 }) ?? true else {
                throw GLM5NextVisionError.invalidInput("video presentation order")
            }
            previous = time
            let index = decoded
            decoded += 1
            guard retained?.contains(index) ?? true else { continue }
            let image = try autoreleasepool {
                try GLM5NextRGBDecoder.render(
                    CIImage(cvPixelBuffer: pixels).transformed(by: transform),
                    maximumBytes: configuration.maximumDecodedBytes - bytes)
            }
            guard image.pixels.count <= configuration.maximumDecodedBytes - bytes else {
                throw GLM5NextVisionError.budgetExceeded("decoded video frames")
            }
            bytes += image.pixels.count
            frames.append(image)
            indices.append(index)
        }
        guard reader.status == .completed else {
            throw GLM5NextVisionError.invalidInput("video decoding failed")
        }
        return (frames, indices, decoded)
    }
}
