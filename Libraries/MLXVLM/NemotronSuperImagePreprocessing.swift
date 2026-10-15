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
    static func validateWorkspace(
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
    // PyTorch antialiased bicubic uses Keys a=-0.5, not non-AA bicubic's a=-0.75.
    private static func cubic(_ distance: Float) -> Float {
        let x = abs(distance)
        if x < 1 { return ((1.5 * x - 2.5) * x) * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * -0.5 }
        return 0
    }
    private static func spans(input: Int, output: Int) throws -> [Span] {
        let scale = Double(input) / Double(output)
        let support = Float(2 * max(scale, 1))
        let inverseScale = Float(1 / max(scale, 1))
        return try (0 ..< output).map { index in
            if index.isMultiple(of: 64) { try Task.checkCancellation() }
            let center = Float(scale * (Double(index) + 0.5))
            let start = max(Int(center - support + 0.5), 0)
            let end = min(Int(center + support + 0.5), input)
            var weights = (start ..< end).map { cubic((Float($0) - center + 0.5) * inverseScale) }
            let total = weights.reduce(Float(0), +)
            if total != 0 { for i in weights.indices { weights[i] /= total } }
            return Span(start: start, weights: weights)
        }
    }

    /// Separable float32 antialiased bicubic, align_corners=false, then CLIP normalization.
    /// Tolerance-based parity is expected; CPU vectorization can change last-bit rounding.
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
        var horizontal: [Float]
        if width == targetWidth {
            horizontal = pixels.map(Float.init)
        } else {
            let coefficients = try spans(input: width, output: targetWidth)
            horizontal = [Float](repeating: 0, count: intermediateCount * 3)
            for y in 0 ..< height {
                if y.isMultiple(of: 16) { try Task.checkCancellation() }
                for x in 0 ..< targetWidth {
                    let span = coefficients[x]
                    for c in 0 ..< 3 {
                        var value: Float = 0
                        for (offset, weight) in span.weights.enumerated() {
                            value +=
                                Float(pixels[(y * width + span.start + offset) * 3 + c]) * weight
                        }
                        horizontal[(y * targetWidth + x) * 3 + c] = value
                    }
                }
            }
        }
        let vertical = height == targetHeight ? [] : try spans(input: height, output: targetHeight)
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        let std: [Float] = [0.26862954, 0.26130258, 0.27577711]
        var result = [Float](repeating: 0, count: outputCount * 3)
        for y in 0 ..< targetHeight {
            if y.isMultiple(of: 16) { try Task.checkCancellation() }
            for x in 0 ..< targetWidth {
                for c in 0 ..< 3 {
                    var value: Float = 0
                    if height == targetHeight {
                        value = horizontal[(y * targetWidth + x) * 3 + c]
                    } else {
                        let span = vertical[y]
                        for (offset, weight) in span.weights.enumerated() {
                            value +=
                                horizontal[((span.start + offset) * targetWidth + x) * 3 + c]
                                * weight
                        }
                    }
                    result[c * outputCount + y * targetWidth + x] = (value / 255 - mean[c]) / std[c]
                }
            }
        }
        return result
    }
}
