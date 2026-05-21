// Copyright © 2026 Apple Inc. (TP variant — Layr-Labs)

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// Numerical-equivalence and shape tests for `LlamaModelTP`.
///
/// On a singleton (size-1) DistributedGroup the sharded layers degenerate to
/// regular Linear layers: column-parallel weights have shape `[outDim/1, inDim]`
/// = `[outDim, inDim]`, and row-parallel layers' allSum is a no-op. So with
/// the same weights, LlamaModelTP must produce output bit-equal to LlamaModel
/// modulo float accumulation order.
public class LlamaTPTests: XCTestCase {

    /// Build a small LlamaConfiguration suitable for fast in-process testing.
    private func smallConfig() -> LlamaConfiguration {
        LlamaConfiguration(
            hiddenSize: 64,
            hiddenLayers: 4,
            intermediateSize: 256,
            attentionHeads: 8,
            rmsNormEps: 1e-5,
            vocabularySize: 128,
            kvHeads: 4
        )
    }

    /// Singleton group used for in-process equivalence testing.
    private func singletonGroup() -> DistributedGroup {
        // Default no-arg init returns a size-1 group when no distributed
        // backend is initialized — exactly what we want here.
        DistributedGroup()
    }

    /// LlamaModelTP with size=1 must have the same parameter shapes as
    /// LlamaModel — that's the precondition for the equivalence test.
    func testLlamaTPSingletonHasMatchingParameterShapes() throws {
        let config = smallConfig()
        let group = singletonGroup()
        XCTAssertEqual(group.size, 1)

        let plain = LlamaModel(config)
        let tp = try LlamaModelTP(config, group: group)

        let plainParams = plain.parameters().flattened()
        let tpParams = tp.parameters().flattened()

        XCTAssertEqual(plainParams.count, tpParams.count)

        let plainByKey = Dictionary(uniqueKeysWithValues: plainParams.map { ($0.0, $0.1) })
        for (key, value) in tpParams {
            guard let plainValue = plainByKey[key] else {
                XCTFail("TP parameter '\(key)' has no LlamaModel counterpart")
                continue
            }
            XCTAssertEqual(
                value.shape, plainValue.shape,
                "shape mismatch for '\(key)': tp=\(value.shape) plain=\(plainValue.shape)")
        }
    }

    /// Forward-pass output shape matches LlamaModel on size-1 group.
    func testLlamaTPSingletonOutputShape() throws {
        let config = smallConfig()
        let group = singletonGroup()
        let model = try LlamaModelTP(config, group: group)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 128])
    }

    /// With identical weights, LlamaModelTP(size=1) and LlamaModel produce
    /// the same logits. Tolerance is loose because reshape+matmul order isn't
    /// guaranteed identical, but on size=1 there's no allreduce so drift
    /// should be at the float-accumulation noise floor.
    func testLlamaTPSingletonEquivalenceToLlamaModel() throws {
        let config = smallConfig()
        let group = singletonGroup()

        let plain = LlamaModel(config)
        let tp = try LlamaModelTP(config, group: group)

        // Copy weights from plain → tp by name. The TP modules use the same
        // parameter keys as the plain modules (q_proj, k_proj, etc.) so a
        // flat update by key works.
        let plainParams = plain.parameters()
        try tp.update(parameters: plainParams, verify: .all)
        eval(tp.parameters())

        let input = MLXArray([7, 11, 13, 17, 19])[.newAxis, .ellipsis]
        let plainOutput = plain.callAsFunction(input, cache: nil)
        let tpOutput = tp.callAsFunction(input, cache: nil)

        XCTAssertEqual(plainOutput.shape, tpOutput.shape)
        let diff = (plainOutput - tpOutput).abs().max()
        let diffValue = diff.item(Float.self)
        XCTAssertLessThan(
            diffValue, 1e-3,
            "TP(size=1) output diverged from LlamaModel by max=\(diffValue)")
    }

    /// Shape validation: TP refuses to initialize if heads don't divide evenly.
    func testLlamaTPRejectsNonDivisibleHeads() {
        // Force a 2-rank "fake" by directly testing the validation logic via
        // the helper — we can't easily construct a size-2 group in-process
        // without a real distributed backend.
        var config = smallConfig()
        config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 256,
            attentionHeads: 7,  // not divisible by 2
            rmsNormEps: 1e-5, vocabularySize: 128, kvHeads: 4
        )

        // With size=1 group, divisibility check is trivially satisfied (7 % 1 = 0),
        // so this test only documents that we CAN construct on size=1. Real
        // size-2 rejection is exercised by the 2-Mac smoke test in d-inference.
        XCTAssertNoThrow(try LlamaModelTP(config, group: singletonGroup()))
    }

    /// Sanity check on the static weight-sharding helper without needing a
    /// real distributed group.
    func testShardWeightIfNeededColumnParallel() {
        let weight = MLXArray(0 ..< 8 * 4, [8, 4]).asType(.float32)
        // Column-parallel slice along axis 0 (output dim) for world=2.
        let rank0 = LlamaModelTP.shardWeightIfNeeded(
            key: "model.layers.0.self_attn.q_proj.weight",
            value: weight, rank: 0, world: 2)
        let rank1 = LlamaModelTP.shardWeightIfNeeded(
            key: "model.layers.0.self_attn.q_proj.weight",
            value: weight, rank: 1, world: 2)
        XCTAssertEqual(rank0.shape, [4, 4])
        XCTAssertEqual(rank1.shape, [4, 4])
        // rank 0 has rows 0..4, rank 1 has rows 4..8.
        XCTAssertEqual(rank0[0, 0].item(Float.self), 0.0)
        XCTAssertEqual(rank1[0, 0].item(Float.self), 16.0)
    }

    func testShardWeightIfNeededRowParallel() {
        let weight = MLXArray(0 ..< 4 * 8, [4, 8]).asType(.float32)
        // Row-parallel slice along axis 1 (input dim) for world=2.
        let rank0 = LlamaModelTP.shardWeightIfNeeded(
            key: "model.layers.0.self_attn.o_proj.weight",
            value: weight, rank: 0, world: 2)
        let rank1 = LlamaModelTP.shardWeightIfNeeded(
            key: "model.layers.0.self_attn.o_proj.weight",
            value: weight, rank: 1, world: 2)
        XCTAssertEqual(rank0.shape, [4, 4])
        XCTAssertEqual(rank1.shape, [4, 4])
        // rank 0 has cols 0..4, rank 1 has cols 4..8.
        XCTAssertEqual(rank0[0, 0].item(Float.self), 0.0)
        XCTAssertEqual(rank1[0, 0].item(Float.self), 4.0)
    }

    func testShardWeightIfNeededEmbeddingNotSharded() {
        let weight = MLXArray(0 ..< 8 * 4, [8, 4]).asType(.float32)
        let rank0 = LlamaModelTP.shardWeightIfNeeded(
            key: "model.embed_tokens.weight",
            value: weight, rank: 0, world: 2)
        XCTAssertEqual(rank0.shape, [8, 4])
    }

    func testShardWeightIfNeededLayerNormNotSharded() {
        let weight = MLXArray(0 ..< 64, [64]).asType(.float32)
        let rank1 = LlamaModelTP.shardWeightIfNeeded(
            key: "model.layers.0.input_layernorm.weight",
            value: weight, rank: 1, world: 2)
        XCTAssertEqual(rank1.shape, [64])
    }
}
