import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// Builds tiny models with seeded random weights for the forward-pass tests.
///
/// The weights are synthetic. No test downloads or reads real model weights.
enum SyntheticModel {

    /// Decodes a model configuration from a JSON dictionary.
    ///
    /// Each model test keeps a small base dictionary and passes `overrides`
    /// for the variant it needs. A key in `overrides` replaces the same key
    /// in `base`.
    static func configuration<C: Decodable>(
        _ type: C.Type, _ base: [String: Any], overrides: [String: Any] = [:]
    ) throws -> C {
        let merged = base.merging(overrides) { _, new in new }
        let data = try JSONSerialization.data(withJSONObject: merged)
        return try JSONDecoder().decode(C.self, from: data)
    }

    /// Replaces every floating-point parameter of `model` with seeded random
    /// values and evaluates them.
    ///
    /// - A 1-D parameter with the name `weight` is a norm scale. It gets
    ///   values near 1.
    /// - Another 1-D parameter (a bias, a router bias, `A_log`, `dt_bias`)
    ///   gets small values near 0.
    /// - A parameter with 2 or more dimensions gets a standard deviation of
    ///   `1 / sqrt(fan-in)`, so that activations stay near 1. The fan-in is
    ///   the last dimension, or all dimensions but the first for a
    ///   convolution.
    ///
    /// The parameters of a quantized layer (the packed weight, `scales` and
    /// `biases`) keep their values.
    ///
    /// The values depend only on `seed` and on the parameter names. The
    /// function evaluates the new parameters before it returns, so that no
    /// test runs on unevaluated random weights.
    static func randomize(_ model: Module, seed: UInt64) {
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        var updated: [(String, MLXArray)] = []
        for (index, (name, value)) in parameters.enumerated() {
            guard value.dtype.isFloatingPoint,
                !name.hasSuffix(".scales"), !name.hasSuffix(".biases")
            else { continue }
            let key = MLXRandom.key(seed &* 1_000_003 &+ UInt64(index))
            let noise = MLXRandom.normal(value.shape, key: key)
            let random: MLXArray
            if value.ndim == 1 {
                random = name.hasSuffix("weight") ? 1 + 0.1 * noise : 0.1 * noise
            } else {
                let fanIn =
                    name.contains("conv")
                    ? value.shape.dropFirst().reduce(1, *) : value.dim(-1)
                random = noise * (1 / Float(fanIn).squareRoot())
            }
            updated.append((name, random.asType(value.dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    /// The parameters of `model` as a flat dictionary, keyed like a
    /// checkpoint.
    static func flatParameters(_ model: Module) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    }

    /// Token IDs from a fixed linear congruential generator. The result does
    /// not use MLX random state.
    static func tokens(count: Int, vocabularySize: Int, seed: Int) -> [Int] {
        var state = UInt64(truncatingIfNeeded: seed) &+ 0x9E37_79B9
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabularySize))
        }
    }

    /// A `[rows, length]` int32 array of token IDs.
    static func batch(_ rows: [[Int]]) -> MLXArray {
        let length = rows[0].count
        precondition(rows.allSatisfy { $0.count == length })
        return MLXArray(rows.flatMap { $0.map { Int32($0) } }).reshaped(rows.count, length)
    }

    /// The largest absolute difference between two arrays, as a float.
    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// The largest absolute value of an array, as a float.
    static func maxAbs(_ a: MLXArray) -> Float {
        abs(a.asType(.float32)).max().item(Float.self)
    }

    /// Writes `weights` to a `.safetensors` file in a new temporary folder,
    /// loads them into `model` through ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``,
    /// and deletes the folder.
    ///
    /// The loader calls the model's `sanitize(weights:)` and then updates
    /// the model with `verify: [.all]`. So the load fails when a key is
    /// missing, a key is not used, or a shape does not match.
    static func load(
        _ weights: [String: MLXArray], into model: any LanguageModel,
        quantization: BaseConfiguration.Quantization? = nil
    ) throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("synthetic-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try MLX.save(
            arrays: weights, url: folder.appendingPathComponent("model.safetensors"))
        try loadWeights(modelDirectory: folder, model: model, quantization: quantization)
    }
}

/// Forward-pass checks that each model test runs on its tiny model.
///
/// Each check records its result with `#expect` at the caller's source
/// location. The tolerances are arguments, so that each model test states
/// its own tolerance and the reason for it.
enum ForwardPassChecks {

    /// Runs the model on `rows` and returns the logits.
    static func logits(
        _ model: any LanguageModel, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let output = model(SyntheticModel.batch(rows), cache: cache)
        eval(output)
        return output
    }

    /// Checks the logits shape and dtype for batch sizes 1 and 2, and that
    /// every logit is finite.
    static func checkShapeDTypeAndFinite(
        _ model: any LanguageModel, vocabularySize: Int, length: Int,
        dtype: DType = .float32,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        for batchSize in [1, 2] {
            let rows = (0 ..< batchSize).map {
                SyntheticModel.tokens(count: length, vocabularySize: vocabularySize, seed: $0)
            }
            let output = logits(model, rows)
            #expect(
                output.shape == [batchSize, length, vocabularySize],
                "batch size \(batchSize)", sourceLocation: sourceLocation)
            #expect(
                output.dtype == dtype, "batch size \(batchSize)", sourceLocation: sourceLocation)
            #expect(
                isFinite(output).all().item(Bool.self),
                "batch size \(batchSize): a logit is not finite", sourceLocation: sourceLocation)
        }
    }

    /// Checks that the model gives the same logits when it runs the same
    /// input twice, and that two models built with the same seed give the
    /// same logits. A model built with another seed must give other logits,
    /// so that the check cannot pass on a constant output.
    static func checkDeterminism(
        make: () throws -> any LanguageModel, seed: UInt64, vocabularySize: Int,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let rows = [SyntheticModel.tokens(count: 8, vocabularySize: vocabularySize, seed: 7)]

        let first = try make()
        SyntheticModel.randomize(first, seed: seed)
        let a = logits(first, rows)
        let b = logits(first, rows)

        let second = try make()
        SyntheticModel.randomize(second, seed: seed)
        let c = logits(second, rows)

        let other = try make()
        SyntheticModel.randomize(other, seed: seed + 1)
        let d = logits(other, rows)

        #expect(
            SyntheticModel.maxAbsDifference(a, b) == 0, "same model, same input",
            sourceLocation: sourceLocation)
        #expect(
            SyntheticModel.maxAbsDifference(a, c) == 0, "same seed, new model",
            sourceLocation: sourceLocation)
        #expect(
            SyntheticModel.maxAbsDifference(a, d) > 1e-3, "another seed must change the logits",
            sourceLocation: sourceLocation)
    }

    /// Checks that the cache gives the same logits as a full forward pass.
    ///
    /// The check runs the whole sequence once without a cache. Then it runs
    /// the same sequence in `chunks` with a new cache from
    /// `model.newCache(parameters:)`. A chunk of 1 is a decode step. The
    /// logits of every position must match the full pass within
    /// `tolerance`.
    ///
    /// - Returns: the largest difference, for the caller's records.
    @discardableResult
    static func checkCacheConsistency(
        _ model: any LanguageModel, rows: [[Int]], chunks: [Int], tolerance: Float,
        parameters: GenerateParameters? = nil,
        sourceLocation: SourceLocation = #_sourceLocation
    ) -> Float {
        let length = rows[0].count
        precondition(chunks.reduce(0, +) == length, "the chunks must cover the sequence")

        let full = logits(model, rows)

        let cache = model.newCache(parameters: parameters)
        var start = 0
        var worst: Float = 0
        for chunk in chunks {
            let part = rows.map { Array($0[start ..< start + chunk]) }
            let stepped = logits(model, part, cache: cache)
            let reference = full[0..., start ..< start + chunk, 0...]
            let difference = SyntheticModel.maxAbsDifference(stepped, reference)
            worst = max(worst, difference)
            #expect(
                difference <= tolerance,
                "positions \(start) ..< \(start + chunk): cached logits differ by \(difference)",
                sourceLocation: sourceLocation)
            start += chunk
        }
        return worst
    }

    /// Checks that each row of a batch of 2 gives the same logits as the
    /// row alone.
    static func checkBatchInvariance(
        _ model: any LanguageModel, rowA: [Int], rowB: [Int], tolerance: Float,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let batched = logits(model, [rowA, rowB])
        let aloneA = logits(model, [rowA])
        let aloneB = logits(model, [rowB])
        let differenceA = SyntheticModel.maxAbsDifference(batched[0 ..< 1], aloneA)
        let differenceB = SyntheticModel.maxAbsDifference(batched[1 ..< 2], aloneB)
        #expect(
            differenceA <= tolerance, "row 0 differs by \(differenceA)",
            sourceLocation: sourceLocation)
        #expect(
            differenceB <= tolerance, "row 1 differs by \(differenceB)",
            sourceLocation: sourceLocation)
        #expect(
            SyntheticModel.maxAbsDifference(aloneA, aloneB) > 1e-3,
            "the two rows must give different logits", sourceLocation: sourceLocation)
    }

    /// Checks that a change to the token at `position` does not change the
    /// logits before `position`, and does change the logits at `position`.
    static func checkCausality(
        _ model: any LanguageModel, row: [Int], position: Int, vocabularySize: Int,
        tolerance: Float,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        var changed = row
        changed[position] = (row[position] + 1) % vocabularySize
        let original = logits(model, [row])
        let modified = logits(model, [changed])

        let before = SyntheticModel.maxAbsDifference(
            original[0..., ..<position], modified[0..., ..<position])
        let after = SyntheticModel.maxAbsDifference(
            original[0..., position...], modified[0..., position...])
        #expect(
            before <= tolerance, "positions before \(position) changed by \(before)",
            sourceLocation: sourceLocation)
        #expect(
            after > 1e-3, "the change at \(position) must change its own logits",
            sourceLocation: sourceLocation)
    }
}
