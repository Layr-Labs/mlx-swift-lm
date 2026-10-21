// Image grid planning translated from NVIDIA's image_processing_nemotron_h_omni.py,
// revision 7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c.
// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import Foundation

/// CPU-only numeric preprocessing. Input is RGB8 HWC; output is normalized Float32 CHW.
/// Does not decode files, schedule media, or claim video support. The resize preserves
/// floating overshoot: clipping/rounding to RGB8 (as in PIL) changes model inputs.
public enum NemotronSuperImagePreprocessing {
    public enum Failure: Error, Equatable {
        case invalidGeometry, invalidConfiguration, invalidPixels, workspaceLimit
    }
    public struct Grid: Equatable, Sendable {
        public let widthPatches: Int
        public let heightPatches: Int
        public let width: Int
        public let height: Int
        public let tokens: Int
    }
    private static let maximumPixels = 64 * 1024 * 1024
    private static func pixelCount(_ width: Int, _ height: Int) throws -> Int {
        guard width > 0, height > 0 else { throw Failure.invalidGeometry }
        let (count, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow, count <= maximumPixels else { throw Failure.invalidGeometry }
        return count
    }

    // Local CPU safety policy, not an architectural/model resolution limit.
    // Allows every default-budget target grid (including 106496 x 32 pixels).
    private static let maximumAxis = 131072
    private static let maximumWorkspaceBytes = 256 * 1024 * 1024

    /// Checks all buffers and coefficient tables before allocating any of them.
    /// Conservative accounting includes input RGB8, both Float32 image buffers,
    /// both coefficient tables, and 64 bytes per span for array/allocator overhead.
    /// `normalizeRGB` applies it; a host calls it with the declared source size and
    /// the grid's target size to refuse an image before decoding its pixels.
    public static func validateWorkspace(
        width: Int, height: Int,
        targetWidth: Int, targetHeight: Int
    ) throws {
        guard width <= maximumAxis, height <= maximumAxis,
            targetWidth <= maximumAxis, targetHeight <= maximumAxis
        else { throw Failure.workspaceLimit }
        let input = try pixelCount(width, height)
        let output = try pixelCount(targetWidth, targetHeight)
        let intermediate = try pixelCount(targetWidth, height)
        func coefficientBytes(_ input: Int, _ output: Int) -> Int {
            if input == output { return 0 }
            // At most ceil(4 * max(scale, 1)) + 2 taps, clipped to input length.
            let taps = min(input, Int(ceil(4 * max(Double(input) / Double(output), 1))) + 2)
            return output * (64 + taps * MemoryLayout<Float>.stride)
        }
        // Bounds above keep all arithmetic below Int64.max before this sum.
        let bytes =
            input * 3 + (intermediate + output) * 3 * MemoryLayout<Float>.stride
            + coefficientBytes(width, targetWidth) + coefficientBytes(height, targetHeight)
        guard bytes <= maximumWorkspaceBytes else { throw Failure.workspaceLimit }
    }

    /// Mirrors the pinned image rule, including Python's ties-to-even round and
    /// its per-image budget (which is not divided by the image count).
    /// Rejects degenerate grids where the upstream rule divides by zero.
    public static func grid(
        width: Int, height: Int, imageCount: Int = 1,
        patchSize: Int = 16, downsampleFactor: Int = 2,
        minimumPatches: Int = 1024, maximumPatches: Int = 13312,
        maximumModelLength: Int = 16384
    ) throws -> Grid {
        _ = try pixelCount(width, height)
        guard imageCount > 0, imageCount <= maximumPixels,
            patchSize > 0, patchSize <= 1024,
            downsampleFactor > 0, downsampleFactor <= 1024,
            minimumPatches > 0, minimumPatches <= maximumPixels,
            maximumPatches >= 0, maximumPatches <= maximumPixels,
            maximumModelLength > 4, maximumModelLength <= maximumPixels
        else { throw Failure.invalidConfiguration }
        let divisor = downsampleFactor
        let budget = max(
            (maximumModelLength - 4) * divisor * divisor,
            minimumPatches * imageCount)
        let available = max(
            min(budget, maximumPatches > 0 ? maximumPatches : budget),
            minimumPatches)
        let closestH = Int((Double(height) / Double(patchSize) + 0.5).rounded(.toNearestOrEven))
        let closestW = Int((Double(width) / Double(patchSize) + 0.5).rounded(.toNearestOrEven))
        guard closestH > 0, closestW > 0 else { throw Failure.invalidGeometry }
        let factor = min(sqrt(Double(available) / Double(closestH * closestW)), 1)
        var h = Int(floor(factor * Double(closestH)))
        var w = Int(floor(factor * Double(closestW)))
        guard h > 0, w > 0 else { throw Failure.invalidGeometry }
        if available > minimumPatches && h * w < minimumPatches {
            let up = sqrt(Double(minimumPatches) / Double(h * w))
            h = Int(ceil(up * Double(h)))
            w = Int(ceil(up * Double(w)))
        }
        let remH = h % divisor
        if remH != 0 {
            h =
                (h + divisor - remH) * w <= available
                ? h + divisor - remH : max(divisor, h - remH)
        }
        let remW = w % divisor
        if remW != 0 {
            w =
                h * (w + divisor - remW) <= available
                ? w + divisor - remW : max(divisor, w - remW)
        }
        // Bound output allocations and reject extreme aspect ratios that cannot fit.
        guard h * w <= available else { throw Failure.invalidGeometry }
        _ = try pixelCount(w * patchSize, h * patchSize)
        return Grid(
            widthPatches: w, heightPatches: h, width: w * patchSize,
            height: h * patchSize, tokens: w * h / (divisor * divisor))
    }

    private struct Span {
        let start: Int
        let weights: [Float]
    }

    // CPU PyTorch 2.12 float32 `_upsample_bicubic2d_aa` rounding, which makes the
    // result bitwise equal rather than within a tolerance: a float32 scale and
    // centre (a Double scale moved large-index centres by one float ulp), a tap
    // argument rounded once from double, and the fused multiply-adds that its
    // arm64 build uses in Keys a=-0.5 (not non-AA bicubic's a=-0.75) and in
    // accumulation. Sums start from the first product, as PyTorch's do.
    private static func cubic(_ distance: Float) -> Float {
        let x = abs(distance)
        if x < 1 { return Float(1).addingProduct(Float(-2.5).addingProduct(1.5, x) * x, x) }
        if x < 2 {
            return Float(2).addingProduct(
                Float(-4).addingProduct(Float(2.5).addingProduct(-0.5, x), x), x)
        }
        return 0
    }
    private static func spans(input: Int, output: Int) throws -> [Span] {
        let scale = Float(input) / Float(output)
        let support = scale >= 1 ? 2 * scale : 2
        let inverseScale = scale >= 1 ? Float(1 / Double(scale)) : 1
        // PyTorch clamps each span to ceil(support) * 2 + 1 taps for rounding edge cases.
        let maximumTaps = Int(support.rounded(.up)) * 2 + 1
        return try (0 ..< output).map { index in
            if index.isMultiple(of: 64) { try Task.checkCancellation() }
            let center = Float(Double(scale) * (Double(index) + 0.5))
            let start = max(Int(Double(center - support) + 0.5), 0)
            let end = min(Int(Double(center + support) + 0.5), input)
            let count = min(max(end - start, 0), maximumTaps)
            var weights = (0 ..< count).map { tap in
                cubic(Float((Double(Float(start + tap) - center) + 0.5) * Double(inverseScale)))
            }
            let total = weights.reduce(Float(0), +)
            if total != 0 { for i in weights.indices { weights[i] /= total } }
            return Span(start: start, weights: weights)
        }
    }

    /// Separable float32 antialiased bicubic, align_corners=false, then CLIP normalization.
    /// Output is normalized CHW. Vectorization changes only the memory layout
    /// (planar intermediate), never the per-element operation order.
    public static func normalizeRGB(
        _ pixels: [UInt8], width: Int, height: Int,
        targetWidth: Int, targetHeight: Int
    ) throws -> [Float] {
        try validateWorkspace(
            width: width, height: height,
            targetWidth: targetWidth, targetHeight: targetHeight)
        let inputCount = try pixelCount(width, height)
        let outputCount = try pixelCount(targetWidth, targetHeight)
        let intermediateCount = try pixelCount(targetWidth, height)
        guard pixels.count == inputCount * 3 else { throw Failure.invalidPixels }
        try Task.checkCancellation()
        let horizontal = try horizontalPass(
            pixels, width: width, height: height, targetWidth: targetWidth,
            planeSize: intermediateCount)
        let vertical = height == targetHeight ? nil : try spans(input: height, output: targetHeight)
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        let std: [Float] = [0.26862954, 0.26130258, 0.27577711]
        var result = [Float](repeating: 0, count: outputCount * 3)
        try horizontal.withUnsafeBufferPointer { planes in
            try result.withUnsafeMutableBufferPointer { output in
                for c in 0 ..< 3 {
                    let plane = planes.baseAddress! + c * intermediateCount
                    let m = mean[c]
                    let s = std[c]
                    for y in 0 ..< targetHeight {
                        if y.isMultiple(of: 16) { try Task.checkCancellation() }
                        let line = output.baseAddress! + c * outputCount + y * targetWidth
                        if let vertical {
                            accumulate(vertical[y], rows: plane, rowLength: targetWidth, into: line)
                        } else {
                            line.update(from: plane + y * targetWidth, count: targetWidth)
                        }
                        for x in 0 ..< targetWidth { line[x] = (line[x] / 255 - m) / s }
                    }
                }
            }
        }
        return result
    }

    /// Planar `[channel][row][column]` float32 samples after the width resize.
    private static func horizontalPass(
        _ pixels: [UInt8], width: Int, height: Int, targetWidth: Int, planeSize: Int
    ) throws -> [Float] {
        var planes = [Float](repeating: 0, count: planeSize * 3)
        // No coefficients when the width is unchanged: samples are copied.
        let coefficients = width == targetWidth ? nil : try spans(input: width, output: targetWidth)
        try pixels.withUnsafeBufferPointer { source in
            try planes.withUnsafeMutableBufferPointer { destination in
                let out = destination.baseAddress!
                for y in 0 ..< height {
                    if y.isMultiple(of: 16) { try Task.checkCancellation() }
                    let row = source.baseAddress! + y * width * 3
                    let base = y * targetWidth
                    for x in 0 ..< targetWidth {
                        var sum: SIMD4<Float>
                        if let span = coefficients?[x] {
                            sum = span.weights.withUnsafeBufferPointer { weights in
                                guard !weights.isEmpty else { return .zero }
                                var p = row + span.start * 3
                                var total =
                                    SIMD4(Float(p[0]), Float(p[1]), Float(p[2]), 0) * weights[0]
                                for tap in 1 ..< weights.count {
                                    p += 3
                                    total.addProduct(
                                        SIMD4(Float(p[0]), Float(p[1]), Float(p[2]), 0),
                                        weights[tap])
                                }
                                return total
                            }
                        } else {
                            let p = row + x * 3
                            sum = SIMD4(Float(p[0]), Float(p[1]), Float(p[2]), 0)
                        }
                        out[base + x] = sum[0]
                        out[planeSize + base + x] = sum[1]
                        out[2 * planeSize + base + x] = sum[2]
                    }
                }
            }
        }
        return planes
    }

    /// `line[x] = sum over taps of weight * rows[(start + tap) * rowLength + x]`,
    /// accumulated in tap order with fused multiply-adds.
    private static func accumulate(
        _ span: Span, rows: UnsafePointer<Float>, rowLength: Int,
        into line: UnsafeMutablePointer<Float>
    ) {
        let lanes = 8
        let vectorEnd = rowLength - rowLength % lanes
        span.weights.withUnsafeBufferPointer { weights in
            guard !weights.isEmpty else { return line.update(repeating: 0, count: rowLength) }
            let first = rows + span.start * rowLength
            var x = 0
            while x < vectorEnd {
                var sum =
                    UnsafeRawPointer(first + x).loadUnaligned(as: SIMD8<Float>.self) * weights[0]
                for tap in 1 ..< weights.count {
                    let next = UnsafeRawPointer(first + tap * rowLength + x).loadUnaligned(
                        as: SIMD8<Float>.self)
                    sum.addProduct(next, weights[tap])
                }
                UnsafeMutableRawPointer(line + x).storeBytes(of: sum, as: SIMD8<Float>.self)
                x += lanes
            }
            while x < rowLength {
                var sum = first[x] * weights[0]
                for tap in 1 ..< weights.count {
                    sum.addProduct(first[tap * rowLength + x], weights[tap])
                }
                line[x] = sum
                x += 1
            }
        }
    }
}
