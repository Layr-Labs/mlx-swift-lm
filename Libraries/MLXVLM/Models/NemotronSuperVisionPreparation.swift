import Foundation
import MLX
import MLXLMCommon

/// Unregistered native preparation seam. Host owns decoded-media validation,
/// memory admission, serialization against model unload, and request retirement.
/// This function does not attach a provider vision adapter or permit cache reuse.
public enum NemotronSuperVisionPreparation {
    public enum Failure: Error { case geometry, placeholders, capacity }
    public struct Prepared {
        public let prompt: [Int32]
        public let features: [MLXArray]
        /// Absolute causal placeholder ranges, one per image, in input order.
        public let ranges: [Range<Int>]
    }

    /// CPU-only validation performed before constructing any tower graph.
    /// Adjacent images remain separate ranges; visual attention is causal.
    public static func ranges(prompt: [Int32], featureCounts: [Int],
        imageTokenID: Int32 = 18) throws -> [Range<Int>] {
        guard !prompt.isEmpty, !featureCounts.isEmpty,
            featureCounts.allSatisfy({ $0 > 0 }) else { throw Failure.placeholders }
        var cursor = 0
        var result: [Range<Int>] = []
        for count in featureCounts {
            while cursor < prompt.count && prompt[cursor] != imageTokenID { cursor += 1 }
            guard count <= prompt.count - cursor else { throw Failure.placeholders }
            let end = cursor + count
            guard prompt[cursor..<end].allSatisfy({ $0 == imageTokenID }) else {
                throw Failure.placeholders
            }
            result.append(cursor..<end)
            cursor = end
        }
        guard !prompt[cursor...].contains(imageTokenID) else { throw Failure.placeholders }
        return result
    }

    /// Builds normalized CPU samples into native arrays in a separate completed
    /// epoch. Retain each raw array BEFORE reshape, then retain/evaluate each root
    /// before the scope fences. Failed completion retains every submitted root.
    public static func makeNormalizedImages(installed: NemotronSuperInstallation.Installed,
        pixels: [[Float]], widths: [Int], heights: [Int], construction: NativeConstructionScope
    ) throws -> [MLXArray] {
        guard !pixels.isEmpty, pixels.count == widths.count, pixels.count == heights.count else {
            throw Failure.geometry
        }
        for index in pixels.indices {
            let width = widths[index], height = heights[index]
            guard width > 0, height > 0, width <= 131072, height <= 131072 else { throw Failure.geometry }
            let (count, overflow) = width.multipliedReportingOverflow(by: height)
            guard !overflow, count <= 13312 * 16 * 16, pixels[index].count == count * 3 else {
                throw Failure.geometry
            }
        }
        return try installed.withNativeSetup(construction: construction) { _ in
            var images: [MLXArray] = []
            for index in pixels.indices {
                try Task.checkCancellation()
                let raw = MLXArray(pixels[index])
                try construction.retain(raw)
                let image = raw.reshaped(1, 3, heights[index], widths[index])
                try construction.retain(image)
                try MLX.withError { errors in
                    eval(image)
                    try errors.check()
                }
                images.append(image)
            }
            return images
        }
    }

    /// All input/output arrays are retained BEFORE native evaluation. A failed
    /// fence keeps them in the supplied scope; the host must quarantine that
    /// scope and its request charge. `maximumFeatureBytes` is a caller grant,
    /// not a measurement of RADIO's intermediate workspace.
    public static func prepare(installed: NemotronSuperInstallation.Installed,
        prompt: [Int32], normalizedImages: [MLXArray], maximumFeatureBytes: Int,
        construction: NativeConstructionScope) throws -> Prepared {
        guard maximumFeatureBytes > 0 else { throw Failure.capacity }
        return try installed.withNativeSetup(construction: construction) { model in
            guard !prompt.isEmpty, prompt.allSatisfy({
                $0 >= 0 && Int($0) < model.languageModel.vocabularySize
            }) else { throw Failure.placeholders }
            let dimensions = model.imageComposition.tower.dimensions
            var counts: [Int] = []
            var total = 0
            for pixels in normalizedImages {
                guard pixels.ndim == 4, pixels.dim(0) == 1,
                    pixels.dim(1) == dimensions.channels,
                    pixels.dim(2) > 0, pixels.dim(3) > 0,
                    pixels.dim(2) % (2 * dimensions.patch) == 0,
                    pixels.dim(3) % (2 * dimensions.patch) == 0 else { throw Failure.geometry }
                let (patches, overflow) = (pixels.dim(2) / dimensions.patch)
                    .multipliedReportingOverflow(by: pixels.dim(3) / dimensions.patch)
                guard !overflow, patches <= 13312 else { throw Failure.geometry }
                let count = patches / 4
                let (bytes, byteOverflow) = count.multipliedReportingOverflow(
                    by: model.imageComposition.width * 4)
                guard !byteOverflow, bytes <= maximumFeatureBytes - total else { throw Failure.capacity }
                total += bytes; counts.append(count)
            }
            let spans = try ranges(prompt: prompt, featureCounts: counts,
                imageTokenID: model.imageComposition.imageTokenID)
            var features: [MLXArray] = []
            for pixels in normalizedImages {
                try Task.checkCancellation()
                try construction.retain(pixels)
                let feature = try model.imageComposition.imageFeatures(normalizedPixels: pixels)
                try construction.retain(feature)
                try MLX.withError { errors in
                    eval(feature)
                    try errors.check()
                }
                features.append(feature)
            }
            return Prepared(prompt: prompt, features: features, ranges: spans)
        }
    }
}
