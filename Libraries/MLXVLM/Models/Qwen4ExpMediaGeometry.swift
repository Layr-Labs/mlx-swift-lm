// Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
// Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
// Adapted from Transformers v5.8.0 (Apache-2.0); see docs/qwen4/preprocessing.md.

import Foundation

/// Reference geometry only. This does not grant memory admission, decode
/// pixels, or silently reduce a requested budget to make a request fit.
enum Qwen4ExpMediaGeometry {
    enum Failure: Error, Equatable {
        case invalidGeometry, invalidBudget, overflow, invalidFrameMetadata
    }

    struct Size: Equatable, Sendable {
        let height: Int
        let width: Int
    }

    static func product(_ values: Int...) throws -> Int {
        var result = 1
        for value in values {
            guard value > 0 else { throw Failure.invalidGeometry }
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else { throw Failure.overflow }
            result = next
        }
        // Above this integer boundary Double no longer represents dimensions
        // exactly. Refuse malicious geometry before conversion/allocation.
        guard result <= 9_007_199_254_740_991 else { throw Failure.overflow }
        return result
    }

    private static func multiple(_ value: Double, factor: Int,
                                 rounding: FloatingPointRoundingRule) throws -> Int {
        let rounded = value.rounded(rounding)
        guard rounded.isFinite, rounded >= 0, rounded < Double(Int.max / factor) else {
            throw Failure.overflow
        }
        return Int(rounded) * factor
    }

    static func image(height: Int, width: Int, factor: Int,
                      minPixels: Int, maxPixels: Int) throws -> Size {
        try resize(frames: 1, height: height, width: width, temporalFactor: 1,
                   factor: factor, minPixels: minPixels, maxPixels: maxPixels, video: false)
    }

    static func video(frames: Int, height: Int, width: Int, temporalFactor: Int,
                      factor: Int, minPixels: Int, maxPixels: Int) throws -> Size {
        try resize(frames: frames, height: height, width: width, temporalFactor: temporalFactor,
                   factor: factor, minPixels: minPixels, maxPixels: maxPixels, video: true)
    }

    private static func resize(frames: Int, height: Int, width: Int, temporalFactor: Int,
                               factor: Int, minPixels: Int, maxPixels: Int, video: Bool) throws -> Size {
        guard factor > 0, temporalFactor > 0, frames > 0, height > 0, width > 0,
              Double(max(height, width)) / Double(min(height, width)) <= 200,
              !video || (height >= factor && width >= factor) else { throw Failure.invalidGeometry }
        guard minPixels > 0, maxPixels >= minPixels else { throw Failure.invalidBudget }
        let pixels = try product(frames, height, width)
        var h = try multiple(Double(height) / Double(factor), factor: factor, rounding: .toNearestOrEven)
        var w = try multiple(Double(width) / Double(factor), factor: factor, rounding: .toNearestOrEven)
        // v5.8.0 uses CEIL for the temporal group count, not the newer main
        // branch's round. Pin the model's published processor version.
        let temporal = try multiple(Double(frames) / Double(temporalFactor),
                                    factor: temporalFactor, rounding: .up)
        let roundedPixels = h == 0 || w == 0 ? 0 : try product(temporal, h, w)
        if roundedPixels > maxPixels {
            let beta = sqrt(Double(pixels) / Double(maxPixels))
            h = max(factor, try multiple(Double(height) / beta / Double(factor), factor: factor, rounding: .down))
            w = max(factor, try multiple(Double(width) / beta / Double(factor), factor: factor, rounding: .down))
        } else if roundedPixels < minPixels {
            let beta = sqrt(Double(minPixels) / Double(pixels))
            h = try multiple(Double(height) * beta / Double(factor), factor: factor, rounding: .up)
            w = try multiple(Double(width) * beta / Double(factor), factor: factor, rounding: .up)
        }
        guard h > 0, w > 0 else { throw Failure.invalidGeometry }
        _ = try product(temporal, h, w)
        return .init(height: h, width: w)
    }

    static func sampleIndices(totalFrames: Int, sourceFPS: Double, targetFPS: Double = 2,
                              minFrames: Int = 4, maxFrames: Int = 768) throws -> [Int] {
        guard totalFrames > 0, sourceFPS.isFinite, sourceFPS > 0,
              targetFPS.isFinite, targetFPS > 0, minFrames > 0, maxFrames >= minFrames else {
            throw Failure.invalidFrameMetadata
        }
        _ = try product(totalFrames)
        let requested = Double(totalFrames) / sourceFPS * targetFPS
        guard requested.isFinite, requested >= 0, requested < Double(Int.max) else { throw Failure.overflow }
        let count = min(max(Int(requested), minFrames), maxFrames, totalFrames)
        if count == 1 { return [0] }
        let step = Double(totalFrames - 1) / Double(count - 1)
        return (0..<count).map { index in
            index == count - 1 ? totalFrames - 1 : Int((Double(index) * step).rounded(.toNearestOrEven))
        }
    }

    static func timestamps(indices: [Int], sourceFPS: Double, temporalFactor: Int) throws -> [Double] {
        guard !indices.isEmpty, indices.allSatisfy({ $0 >= 0 }),
              sourceFPS.isFinite, sourceFPS > 0, temporalFactor > 0 else { throw Failure.invalidFrameMetadata }
        var result: [Double] = []
        for start in stride(from: 0, to: indices.count, by: temporalFactor) {
            let end = min(start + min(temporalFactor - 1, indices.count - start - 1), indices.count - 1)
            // The reference pads the last group by repeating the last index.
            let timestamp = (Double(indices[start]) / sourceFPS + Double(indices[end]) / sourceFPS) / 2
            guard timestamp.isFinite else { throw Failure.invalidFrameMetadata }
            result.append(timestamp)
        }
        return result
    }
}
