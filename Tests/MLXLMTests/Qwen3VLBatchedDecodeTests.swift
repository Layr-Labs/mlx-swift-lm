import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Cached Qwen3-VL positions must add M-RoPE deltas per row. Wrong-axis
/// addition can either fail broadcasting or silently corrupt a square chunk.
/// A tiny random model checks text cached/full logits; exact integer oracles
/// check unequal media deltas computed by separate supported B=1 prompts.
/// This does not assert support for multimodal prefill at B>1.
///
/// Tolerance 1e-4: the cached pass and the full pass differ only in the
/// order of the attention sums. The differences are near 1e-6.
@Suite
struct Qwen3VLBatchedDecodeTests {

    static let vocabularySize = 64
    static let tolerance: Float = 1e-4

    static func makeModel(seed: UInt64) throws -> Qwen3VL {
        let values: [String: Any] = [
            "model_type": "qwen3_vl",
            "image_token_id": 60,
            "video_token_id": 61,
            "vision_start_token_id": 57,
            "vision_end_token_id": 58,
            "vision_token_id": 59,
            "text_config": [
                "model_type": "qwen3_vl_text", "hidden_size": 32, "intermediate_size": 48,
                "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 256,
                "vocab_size": vocabularySize, "rope_theta": 10000, "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "rope_scaling": [
                    "type": "mrope", "mrope_interleaved": true, "mrope_section": [1, 1, 2],
                ],
            ] as [String: Any],
            "vision_config": [
                "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                "intermediate_size": 48, "out_hidden_size": 32, "num_heads": 4,
                "patch_size": 4, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "num_position_embeddings": 16, "deepstack_visual_indexes": [0],
            ] as [String: Any],
        ]
        let configuration = try JSONDecoder().decode(
            Qwen3VLConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
        let model = Qwen3VL(configuration)
        Qwen3VLDecodeTinyModel.randomize(model, seed: seed)
        return model
    }

    /// The prompt in chunks of 5, 3, 1, 1 and 1 tokens with a new cache
    /// must give the logits of one full pass. The chunks after the first
    /// use the stored M-RoPE deltas, and the chunks of 1 are decode steps.
    @Test(arguments: [1, 2, 3])
    func cachedDecodeMatchesTheFullForwardPass(batchSize: Int) throws {
        let model = try Self.makeModel(seed: 1)
        let rows = (1 ... batchSize).map {
            Qwen3VLDecodeTinyModel.tokens(count: 11, vocabularySize: Self.vocabularySize, seed: $0)
        }
        Qwen3VLDecodeTinyModel.checkCacheConsistency(
            model, rows: rows, chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
    }

    @Test(arguments: [1, 2, 3], [false, true])
    func cachedPositionsAddUnequalMediaDeltas(batchSize: Int, video: Bool) throws {
        let model = try Self.makeModel(seed: 1)
        let grids =
            video
            ? [THW(1, 4, 4), THW(1, 4, 6), THW(3, 4, 4)]
            : [THW(1, 4, 4), THW(1, 4, 6), THW(1, 6, 6)]
        let expectedDeltas: [Int32] = video ? [-2, -3, -9] : [-2, -3, -6]
        let deltas = try grids.prefix(batchSize).map { grid in
            let ids: [Int32] =
                [5, 57] + Array(repeating: video ? 61 : 60, count: grid.product / 4)
                + [58, 9, 11]
            let result = try model.positionResult(
                tokens: MLXArray(ids, [1, ids.count]),
                imageGrids: video ? nil : [grid], videoGrids: video ? [grid] : nil)
            return try #require(result.decodeState.deltas.first)
        }
        #expect(deltas == Array(expectedDeltas.prefix(batchSize)))
        // Square chunks catch silent per-column addition; the other widths
        // catch the broadcast failure. Compare every axis, row and position.
        for width in [batchSize, 1, 4] {
            let positions = Qwen3VLLanguage.cachedPositionIds(
                batch: batchSize, seqLength: width, offset: 24, ropeDeltas: MLXArray(deltas))
            #expect(positions.shape == [3, batchSize, width])
            let rows = expectedDeltas.prefix(batchSize).flatMap { delta in
                (0 ..< width).map { Int32(24 + $0) + delta }
            }
            #expect(
                positions.asArray(Int32.self) == Array(repeating: rows, count: 3).flatMap { $0 })
        }
    }

    @Test
    func cachedPositionsRepeatASingleStoredDelta() {
        let positions = Qwen3VLLanguage.cachedPositionIds(
            batch: 3, seqLength: 2, offset: 24, ropeDeltas: MLXArray([Int32(-3)]))
        #expect(positions.shape == [3, 3, 2])
        #expect(
            positions.asArray(Int32.self)
                == Array(repeating: [Int32(21), 22], count: 9).flatMap { $0 })
    }
}

/// Tiny-model helpers for this file. They are copied from the kernel test
/// support of PR #183 (`Tests/MLXLMTests/Kernel/Support/SyntheticModel.swift`
/// and `ForwardPassChecks.swift`) and are private, so that this file does
/// not depend on that PR or collide with it.
private enum Qwen3VLDecodeTinyModel {

    /// Replaces every floating-point parameter with seeded random values:
    /// norm scales near 1, other 1-D values near 0, and matrices with a
    /// standard deviation of `1 / sqrt(fan-in)`.
    static func randomize(_ model: Module, seed: UInt64) {
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        var updated: [(String, MLXArray)] = []
        for (index, (name, value)) in parameters.enumerated() where value.dtype.isFloatingPoint {
            let noise = MLXRandom.normal(
                value.shape, key: MLXRandom.key(seed &* 1_000_003 &+ UInt64(index)))
            let random: MLXArray
            if value.ndim <= 1 {
                random = name.hasSuffix("weight") ? 1 + 0.1 * noise : 0.1 * noise
            } else {
                let fanIn =
                    name.contains("conv") ? value.shape.dropFirst().reduce(1, *) : value.dim(-1)
                random = noise * (1 / Float(fanIn).squareRoot())
            }
            // The GPU has no float64. A float64 initial value becomes float32.
            let dtype: DType = value.dtype == .float64 ? .float32 : value.dtype
            updated.append((name, random.asType(dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    /// Token IDs from a fixed linear congruential generator.
    static func tokens(count: Int, vocabularySize: Int, seed: Int) -> [Int] {
        var state = UInt64(truncatingIfNeeded: seed) &+ 0x9E37_79B9
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabularySize))
        }
    }

    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Runs the model on `rows` and returns the logits.
    static func logits(
        _ model: any LanguageModel, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let batch = MLXArray(rows.flatMap { $0.map { Int32($0) } })
            .reshaped(rows.count, rows[0].count)
        let output = model(batch, cache: cache)
        eval(output)
        return output
    }

    /// The logits of the rows run in `chunks` with a new cache must match
    /// the logits of one full pass without a cache.
    static func checkCacheConsistency(
        _ model: any LanguageModel, rows: [[Int]], chunks: [Int], tolerance: Float,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let full = logits(model, rows)
        let cache = model.newCache(parameters: nil)
        var start = 0
        for chunk in chunks {
            let part = rows.map { Array($0[start ..< start + chunk]) }
            let difference = maxAbsDifference(
                logits(model, part, cache: cache), full[0..., start ..< start + chunk, 0...])
            #expect(
                difference <= tolerance,
                "positions \(start) ..< \(start + chunk): cached logits differ by \(difference)",
                sourceLocation: sourceLocation)
            start += chunk
        }
    }
}
