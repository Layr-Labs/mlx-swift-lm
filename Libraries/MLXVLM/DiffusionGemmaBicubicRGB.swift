// Native DiffusionGemma RGB8 resize; other model processors are unchanged.
// Adapted from Pillow 12.1.0 Resample.c, commit 46f45f674d47b5d8bc54230dda8fe9e214598b87.
// Copyright © 1997-2011 Secret Labs AB; © 1995-2011 Fredrik Lundh and contributors;
// © 2010 Jeffrey A. Clark and contributors. MIT-CMU: docs/diffusiongemma/LICENSE-PILLOW.
import Foundation

enum DiffusionGemmaBicubicRGB {
    enum Failure: Error { case invalidGeometry }
    private struct Span { let start: Int; let weights: [Int32] }
    private static let precision = 22
    private static let maximumPixels = 64 * 1024 * 1024

    static func checkedPixelCount(_ width: Int, _ height: Int) throws -> Int {
        guard width > 0, height > 0, width <= maximumPixels, height <= maximumPixels else {
            throw Failure.invalidGeometry
        }
        let (count, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow, count <= maximumPixels else { throw Failure.invalidGeometry }
        return count
    }
    private static func cubic(_ distance: Double) -> Double {
        let x = abs(distance)
        if x < 1 { return (1.5 * x - 2.5) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * -0.5 }
        return 0
    }
    private static func coefficients(input: Int, output: Int) throws -> [Span] {
        // Pillow's full-image box endpoints are Float32 before widening.
        let scale = Double(Float(input)) / Double(output)
        let filterScale = max(scale, 1), support = 2 * max(scale, 1)
        return try (0..<output).map { index in
            if index.isMultiple(of: 64) { try Task.checkCancellation() }
            let center = (Double(index) + 0.5) * scale
            let start = max(0, Int(center - support + 0.5))
            let end = min(input, Int(center + support + 0.5))
            var values = (start..<end).map { cubic((Double($0) - center + 0.5) * (1 / filterScale)) }
            let sum = values.reduce(0, +)
            if sum != 0 { for i in values.indices { values[i] /= sum } }
            return Span(start: start, weights: values.map {
                Int32(($0 < 0 ? -0.5 : 0.5) + $0 * Double(1 << precision))
            })
        }
    }
    private static func clipped(_ accumulator: Int64) -> UInt8 {
        UInt8(clamping: accumulator >> precision)
    }
    static func resize(_ input: [UInt8], width: Int, height: Int,
        targetWidth: Int, targetHeight: Int) throws -> [UInt8]
    {
        let inputPixels = try checkedPixelCount(width, height)
        let outputPixels = try checkedPixelCount(targetWidth, targetHeight)
        let intermediatePixels = try checkedPixelCount(targetWidth, height)
        guard input.count == inputPixels * 3 else { throw Failure.invalidGeometry }
        try Task.checkCancellation()
        if width == targetWidth && height == targetHeight { return input }
        let rounding = Int64(1 << (precision - 1))
        var horizontal: [UInt8]
        if width == targetWidth { horizontal = input }
        else {
            let spans = try coefficients(input: width, output: targetWidth)
            horizontal = [UInt8](repeating: 0, count: intermediatePixels * 3)
            for y in 0..<height {
                if y.isMultiple(of: 16) { try Task.checkCancellation() }
                for x in 0..<targetWidth {
                    let span = spans[x]
                    for channel in 0..<3 {
                        var value = rounding
                        for (offset, weight) in span.weights.enumerated() {
                            value += Int64(input[(y * width + span.start + offset) * 3 + channel]) * Int64(weight)
                        }
                        horizontal[(y * targetWidth + x) * 3 + channel] = clipped(value)
                    }
                }
            }
        }
        if height == targetHeight { return horizontal }
        let spans = try coefficients(input: height, output: targetHeight)
        var output = [UInt8](repeating: 0, count: outputPixels * 3)
        for y in 0..<targetHeight {
            if y.isMultiple(of: 16) { try Task.checkCancellation() }
            let span = spans[y]
            for x in 0..<targetWidth {
                for channel in 0..<3 {
                    var value = rounding
                    for (offset, weight) in span.weights.enumerated() {
                        value += Int64(horizontal[((span.start + offset) * targetWidth + x) * 3 + channel]) * Int64(weight)
                    }
                    output[(y * targetWidth + x) * 3 + channel] = clipped(value)
                }
            }
        }
        return output
    }
}
