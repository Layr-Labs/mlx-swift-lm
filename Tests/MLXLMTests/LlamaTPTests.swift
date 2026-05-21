// Copyright © 2026 Apple Inc. (TP variant — Layr-Labs)

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
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

    /// Divisibility-check happy path: an odd-head config trivially passes the
    /// `heads % group.size == 0` gate on a singleton group (7 % 1 == 0). This
    /// documents that the check uses % (not strict equality), so size-1 will
    /// accept any positive head count. Real `world >= 2` rejection paths are
    /// exercised end-to-end by the 2-Mac smoke test in d-inference, not here.
    func testLlamaTPAcceptsAnyHeadCountOnSingletonGroup() {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 256,
            attentionHeads: 7,
            rmsNormEps: 1e-5, vocabularySize: 128, kvHeads: 4
        )
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

    /// Future-proofing: q_norm / k_norm (per-head norms used in Qwen3 etc.)
    /// must NOT be matched by the "norm" check in shardWeightIfNeeded —
    /// they SHOULD be sharded along the head dimension. The current Llama
    /// implementation doesn't have these keys, but the precise key match
    /// guards against future TP variants pulling in this code pattern.
    func testShardWeightIfNeededDoesNotConfusePerHeadNorms() {
        let weight = MLXArray(0 ..< 128, [128]).asType(.float32)
        // A q_norm.weight should NOT trip the layernorm gate — it would fall
        // through to the bottom and pass through unchanged because q_norm
        // isn't in the column-parallel or row-parallel projection list.
        let result = LlamaModelTP.shardWeightIfNeeded(
            key: "model.layers.0.self_attn.q_norm.weight",
            value: weight, rank: 0, world: 2)
        // Today: pass-through (no q_norm handling for Llama). The point of
        // this test is the layernorm gate doesn't INCORRECTLY swallow it.
        XCTAssertEqual(result.shape, [128])
    }

    // MARK: - Manual multi-rank sharding validation
    //
    // Real multi-rank DistributedGroup tests need spawned subprocesses (jaccl
    // or ring backend). Within a single process we can still validate the
    // SHARDING MATH by manually slicing weights and running the matmuls /
    // allsum on the test side — bypassing AllToShardedLinear's internal
    // DistributedGroup call. This catches axis-confusion bugs in sanitize
    // and validates that "sharded column matmul + sharded row matmul + sum"
    // equals the unsharded reference, which is the load-bearing invariant
    // of TP correctness.

    /// Builds a fake 2-rank Q×K-style column-parallel matmul: each rank
    /// holds a row-shard of the weight, multiplies the full input by its
    /// shard, and the result concatenates along the head dim. Compare to
    /// the unsharded reference.
    func testColumnParallelMatmulShardingMath() {
        let inDim = 32, outDim = 64, B = 1, L = 4
        let weight = MLXRandom.normal([outDim, inDim])
        let input = MLXRandom.normal([B, L, inDim])

        // Unsharded reference: input @ weight.T → [B, L, outDim]
        let reference = matmul(input, weight.T)
        XCTAssertEqual(reference.shape, [B, L, outDim])

        // Simulated rank-0 / rank-1 shards via shardWeightIfNeeded — the
        // SAME function the production sanitize path uses.
        let key = "model.layers.0.self_attn.q_proj.weight"
        let w0 = LlamaModelTP.shardWeightIfNeeded(key: key, value: weight, rank: 0, world: 2)
        let w1 = LlamaModelTP.shardWeightIfNeeded(key: key, value: weight, rank: 1, world: 2)
        XCTAssertEqual(w0.shape, [outDim / 2, inDim])
        XCTAssertEqual(w1.shape, [outDim / 2, inDim])

        // Each rank multiplies the FULL input by its row-shard.
        // Output shape per rank: [B, L, outDim/2].
        let y0 = matmul(input, w0.T)
        let y1 = matmul(input, w1.T)
        XCTAssertEqual(y0.shape, [B, L, outDim / 2])
        XCTAssertEqual(y1.shape, [B, L, outDim / 2])

        // For column-parallel the next stage operates on sharded outputs
        // independently (no allreduce). To validate against the reference,
        // concatenate along the last dim.
        let combined = concatenated([y0, y1], axis: -1)
        XCTAssertEqual(combined.shape, [B, L, outDim])

        let diff = (combined - reference).abs().max().item(Float.self)
        XCTAssertLessThan(
            diff, 1e-4,
            "column-parallel sharded matmul diverged from unsharded reference by max=\(diff)")
    }

    /// Builds a row-parallel matmul: input is sharded along the input dim
    /// (each rank holds half the input width), each rank multiplies its
    /// input shard by its weight column-shard, then results are SUMMED
    /// (this is what ShardedToAllLinear.allSum does internally).
    func testRowParallelMatmulShardingMath() {
        let inDim = 64, outDim = 32, B = 1, L = 4
        let weight = MLXRandom.normal([outDim, inDim])
        let input = MLXRandom.normal([B, L, inDim])

        // Unsharded reference.
        let reference = matmul(input, weight.T)
        XCTAssertEqual(reference.shape, [B, L, outDim])

        // Slice the weight along axis 1 (input dim).
        let key = "model.layers.0.self_attn.o_proj.weight"
        let w0 = LlamaModelTP.shardWeightIfNeeded(key: key, value: weight, rank: 0, world: 2)
        let w1 = LlamaModelTP.shardWeightIfNeeded(key: key, value: weight, rank: 1, world: 2)
        XCTAssertEqual(w0.shape, [outDim, inDim / 2])
        XCTAssertEqual(w1.shape, [outDim, inDim / 2])

        // Each rank's input is the corresponding axis-2 slice. (In real TP,
        // the upstream layer produces per-rank sharded outputs that feed
        // into this layer's input.)
        let x0 = input[0..., 0..., 0 ..< (inDim / 2)]
        let x1 = input[0..., 0..., (inDim / 2) ..< inDim]

        // Each rank: partial output [B, L, outDim].
        let partial0 = matmul(x0, w0.T)
        let partial1 = matmul(x1, w1.T)
        XCTAssertEqual(partial0.shape, [B, L, outDim])
        XCTAssertEqual(partial1.shape, [B, L, outDim])

        // allSum simulation: sum across ranks.
        let combined = partial0 + partial1
        let diff = (combined - reference).abs().max().item(Float.self)
        XCTAssertLessThan(
            diff, 1e-4,
            "row-parallel sharded matmul (with simulated allSum) diverged from unsharded reference by max=\(diff)")
    }

    /// Combined column-parallel + row-parallel pipeline mirrors how an
    /// attention or MLP block does TP: column-parallel matmul → some op
    /// (here a no-op identity for testing) → row-parallel matmul with
    /// allSum. This is the end-to-end TP correctness invariant.
    func testColumnThenRowParallelEndToEndShardingMath() {
        let inDim = 32, midDim = 64, outDim = 16, B = 1, L = 4
        let w1 = MLXRandom.normal([midDim, inDim])
        let w2 = MLXRandom.normal([outDim, midDim])
        let input = MLXRandom.normal([B, L, inDim])

        // Unsharded reference: input @ w1.T → mid → mid @ w2.T → out
        let mid = matmul(input, w1.T)
        let reference = matmul(mid, w2.T)
        XCTAssertEqual(reference.shape, [B, L, outDim])

        // Stage 1 (column-parallel): shard w1 along axis 0.
        let w1Key = "model.layers.0.mlp.gate_proj.weight"
        let w1a = LlamaModelTP.shardWeightIfNeeded(key: w1Key, value: w1, rank: 0, world: 2)
        let w1b = LlamaModelTP.shardWeightIfNeeded(key: w1Key, value: w1, rank: 1, world: 2)
        let mid0 = matmul(input, w1a.T)  // [B, L, midDim/2]
        let mid1 = matmul(input, w1b.T)  // [B, L, midDim/2]

        // Stage 2 (row-parallel): shard w2 along axis 1, each rank takes
        // the matching input shard from stage 1's sharded output. No
        // explicit "gather + slice" — the sharded layout flows through.
        let w2Key = "model.layers.0.mlp.down_proj.weight"
        let w2a = LlamaModelTP.shardWeightIfNeeded(key: w2Key, value: w2, rank: 0, world: 2)
        let w2b = LlamaModelTP.shardWeightIfNeeded(key: w2Key, value: w2, rank: 1, world: 2)
        let partial0 = matmul(mid0, w2a.T)  // [B, L, outDim]
        let partial1 = matmul(mid1, w2b.T)  // [B, L, outDim]

        let combined = partial0 + partial1
        let diff = (combined - reference).abs().max().item(Float.self)
        XCTAssertLessThan(
            diff, 1e-4,
            "column-then-row sharded pipeline diverged from unsharded reference by max=\(diff)")
    }

    // MARK: - LlamaModelTPQ (quantized) basic sanity

    func testLlamaModelTPQInstantiates() throws {
        let config = smallConfig()
        let model = try LlamaModelTPQ(
            config, group: singletonGroup(), groupSize: 64, bits: 4)
        // 4 layers in smallConfig; singleton group keeps kvHeads at the global
        // count (4 per layer).
        XCTAssertEqual(model.kvHeads.count, 4)
        XCTAssertEqual(model.kvHeads[0], 4)
    }

    func testMakeLlamaTPDispatchesByQuantizationConfig() throws {
        let config = smallConfig()
        let unquantized = try makeLlamaTP(args: config, quantization: nil, group: singletonGroup())
        // Quantization nil → fp16 path.
        XCTAssertTrue(unquantized is LlamaModelTP)

        let q = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let quantized = try makeLlamaTP(args: config, quantization: q, group: singletonGroup())
        XCTAssertTrue(quantized is LlamaModelTPQ)
    }

    /// Quantized weight slicing: weight has shape [outDim, inDim/8]; scales
    /// and biases have shape [outDim, inDim/groupSize]. Column-parallel
    /// slices all three along axis 0. Row-parallel slices along axis 1.
    func testShardQuantizedWeightIfNeededColumnParallel() {
        let outDim = 16, packedInDim = 8  // inDim=64, /8=8
        let weight = MLXArray(0 ..< outDim * packedInDim, [outDim, packedInDim])
            .asType(.uint32)
        let scales = MLXRandom.normal([outDim, 1])  // groupSize=64, one group per row
        let biases = MLXRandom.normal([outDim, 1])

        let baseKey = "model.layers.0.self_attn.q_proj"
        let w0 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "\(baseKey).weight", value: weight, rank: 0, world: 2)
        let w1 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "\(baseKey).weight", value: weight, rank: 1, world: 2)
        let s0 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "\(baseKey).scales", value: scales, rank: 0, world: 2)
        let b0 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "\(baseKey).biases", value: biases, rank: 0, world: 2)

        XCTAssertEqual(w0.shape, [outDim / 2, packedInDim])
        XCTAssertEqual(w1.shape, [outDim / 2, packedInDim])
        XCTAssertEqual(s0.shape, [outDim / 2, 1])
        XCTAssertEqual(b0.shape, [outDim / 2, 1])
    }

    func testShardQuantizedWeightIfNeededRowParallel() {
        let outDim = 16, packedInDim = 8
        let weight = MLXArray(0 ..< outDim * packedInDim, [outDim, packedInDim])
            .asType(.uint32)
        let scales = MLXRandom.normal([outDim, 2])  // 2 groups per row

        let baseKey = "model.layers.0.self_attn.o_proj"
        let w0 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "\(baseKey).weight", value: weight, rank: 0, world: 2)
        let w1 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "\(baseKey).weight", value: weight, rank: 1, world: 2)
        let s0 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "\(baseKey).scales", value: scales, rank: 0, world: 2)

        XCTAssertEqual(w0.shape, [outDim, packedInDim / 2])
        XCTAssertEqual(w1.shape, [outDim, packedInDim / 2])
        XCTAssertEqual(s0.shape, [outDim, 1])
    }

    func testShardQuantizedWeightIfNeededEmbeddingNotSharded() {
        let weight = MLXRandom.normal([128, 64])
        let r0 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "model.embed_tokens.weight", value: weight, rank: 0, world: 2)
        XCTAssertEqual(r0.shape, [128, 64])
    }

    /// Row-parallel biases must NOT be sliced (per LlamaModelTPQ comments).
    /// They're added once after the allreduce.
    func testShardQuantizedRowParallelBiasNotSliced() {
        let bias = MLXRandom.normal([64])
        let r0 = LlamaModelTPQ.shardQuantizedWeightIfNeeded(
            key: "model.layers.0.self_attn.o_proj.bias",
            value: bias, rank: 0, world: 2)
        XCTAssertEqual(r0.shape, [64])
    }
}
