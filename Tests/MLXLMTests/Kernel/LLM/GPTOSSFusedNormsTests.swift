// Copyright © 2026 Eigen Labs / Darkbloom.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

extension KernelTests {

    /// `GPTOSSFusedNorms` writes the bits of the unfused ops it replaces: the
    /// output projection's bias, the residual add and the post-attention norm;
    /// and the experts' biases, routing weights, sum, residual add and the
    /// next norm.
    @Suite(.serialized)
    struct GPTOSSFusedNormsTests {
        private func bytes(_ x: MLXArray) -> Data {
            eval(x)
            return x.asData(access: .copy).data
        }

        /// Byte parity is promised where the norm's form is measured: a GPU
        /// whose family is verified. Elsewhere (CPU, a virtual GPU) the
        /// product keeps the unfused ops.
        private var verified: Bool {
            Device.defaultDevice().deviceType == .gpu
                && ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family())
        }

        private func normal(_ shape: [Int], _ seed: UInt64, _ dtype: DType, scale: Float = 1)
            -> MLXArray
        {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(dtype)
        }

        /// Signed zeros and exact cancellations in a few places, as real rows
        /// have them.
        private func withZeros(_ x: MLXArray, _ seed: UInt64) -> MLXArray {
            let pick = MLXRandom.uniform(0 ..< 1, x.shape, key: MLXRandom.key(seed)) .< 0.02
            let sign = MLXRandom.uniform(0 ..< 1, x.shape, key: MLXRandom.key(seed + 1)) .< 0.5
            let zero = MLX.where(sign, MLXArray(-0.0), MLXArray(0.0)).asType(x.dtype)
            return MLX.where(pick, zero, x)
        }

        @Test func attentionTailKeepsTheUnfusedBits() throws {
            for dtype in [DType.bfloat16, .float16] {
                for width in [2880, 1024, 4096, 1000] {
                    guard GPTOSSFusedNorms.supports(width: width, dtype: dtype) else {
                        #expect(!verified, "\(dtype) \(width)")
                        continue
                    }
                    for rows in [1, 3, 8] {
                        let seed = UInt64(width * 7 + rows)
                        let y = withZeros(normal([rows, 1, width], seed, dtype, scale: 4), seed)
                        let bias = normal([width], seed + 1, dtype)
                        let residual = withZeros(
                            normal([rows, 1, width], seed + 2, dtype, scale: 8), seed + 3)
                        let weight = (normal([width], seed + 4, .float32, scale: 0.2) + 1).asType(
                            dtype)
                        let fused = GPTOSSFusedNorms.attentionTail(
                            y, bias: bias, residual: residual, weight: weight, eps: 1e-5)
                        // The unfused ops: `QuantizedLinear`'s bias add, the
                        // block's residual add, `RMSNorm`.
                        let h = residual + (y + bias)
                        let normed = MLXFast.rmsNorm(h, weight: weight, eps: 1e-5)
                        #expect(bytes(fused.sum) == bytes(h), "sum \(dtype) \(width) \(rows)")
                        #expect(
                            bytes(fused.normed) == bytes(normed), "norm \(dtype) \(width) \(rows)")
                    }
                }
            }
        }

        @Test func expertTailKeepsTheUnfusedBits() throws {
            let experts = 32
            for dtype in [DType.bfloat16, .float16] {
                for width in [2880, 1024, 1000] {
                    guard GPTOSSFusedNorms.supports(width: width, dtype: dtype) else {
                        #expect(!verified, "\(dtype) \(width)")
                        continue
                    }
                    for (rows, topK) in [(1, 4), (3, 4), (8, 4), (2, 2), (1, 8)] {
                        let seed = UInt64(width * 11 + rows * 3 + topK)
                        // Rows as the down projection writes them `[N, 1, K, D]`.
                        let expertRows = withZeros(
                            normal([rows, 1, topK, width], seed, dtype, scale: 2), seed + 1)
                        // Distinct experts per row, in router order.
                        let order = argSort(
                            MLXRandom.uniform(
                                0 ..< 1, [rows, 1, experts], key: MLXRandom.key(seed + 2)),
                            axis: -1)
                        let indices = order[.ellipsis, 0 ..< topK].asType(.uint32)
                        let biasTable = normal([experts, width], seed + 3, dtype, scale: 0.5)
                        let weights = softmax(
                            normal([rows, 1, topK], seed + 4, dtype, scale: 2), axis: -1,
                            precise: true)
                        let residual = withZeros(
                            normal([rows, 1, width], seed + 5, dtype, scale: 8), seed + 6)
                        let next = (normal([width], seed + 7, .float32, scale: 0.2) + 1).asType(
                            dtype)
                        let fused = GPTOSSFusedNorms.expertTail(
                            rows: expertRows, indices: indices, biases: biasTable, weights: weights,
                            residual: residual, next: next, eps: 1e-5)
                        // The unfused ops: `QuantizedSwitchLinear`'s bias,
                        // `MLPBlock`'s weighting and sum, the residual add,
                        // the next `RMSNorm`.
                        let biased = expertRows + biasTable[indices]
                        let summed = (biased * expandedDimensions(weights, axis: -1)).sum(axis: -2)
                        let out = residual + summed
                        let normed = MLXFast.rmsNorm(out, weight: next, eps: 1e-5)
                        let label = "\(dtype) \(width) rows \(rows) k \(topK)"
                        #expect(bytes(fused.out) == bytes(out), "out \(label)")
                        #expect(bytes(fused.normed) == bytes(normed), "norm \(label)")
                    }
                }
            }
        }

        /// The routing kernel picks the experts `mlxTopK` picks, in its order,
        /// and writes `softmax(precise: true)`'s bits, with ties, infinities
        /// and NaN among the logits.
        @Test func routeKeepsTheUnfusedBits() throws {
            for dtype in [DType.bfloat16, .float16] {
                for (experts, topK) in [(32, 4), (128, 8), (64, 6), (16, 16)] {
                    guard
                        let form = GPTOSSFusedNorms.routingForm(
                            experts: experts, topK: topK, dtype: dtype)
                    else {
                        #expect(!verified, "\(dtype) \(experts)")
                        continue
                    }
                    for rows in [1, 3, 64] {
                        let seed = UInt64(experts * 5 + topK * 3 + rows)
                        var logits = normal([rows, 1, experts], seed, dtype, scale: 3)
                        // Ties (rounded to a coarse grid) in every other row.
                        let coarse = (logits * 2).round() / 2
                        let odd = (MLXArray(0 ..< rows) % 2 .== 1).reshaped(rows, 1, 1)
                        logits = MLX.where(odd, coarse, logits)
                        let bias = normal([experts], seed + 1, dtype)
                        if rows == 64 {
                            let special = MLXArray([Float.infinity, -Float.infinity, Float.nan, 0])
                                .asType(dtype)
                            let flat = logits.reshaped(-1)
                            flat[0 ..< 4] = special
                            flat[experts ..< experts + 4] = special
                            logits = flat.reshaped(rows, 1, experts)
                        }
                        let fused = GPTOSSFusedNorms.route(
                            logits: logits, bias: bias, topK: topK, form: form)
                        // The unfused ops: `QuantizedLinear`'s bias add, `mlxTopK`, `softmax`.
                        let g = logits + bias
                        let partitioned = argPartition(g, kth: -topK, axis: -1)
                        let indices = partitioned[.ellipsis, (-topK)...]
                        let values = takeAlong(g, indices, axis: -1)
                        let weights = softmax(values, axis: -1, precise: true)
                        let label = "\(dtype) \(experts)/\(topK) rows \(rows)"
                        #expect(
                            bytes(fused.indices) == bytes(indices.asType(.uint32)),
                            "indices \(label)")
                        #expect(bytes(fused.weights) == bytes(weights), "weights \(label)")
                    }
                }
            }
        }

        /// The gate/up bias folded into the compiled SwiGLU writes what the
        /// bias add and the SwiGLU write, for every 16-bit gate and linear value.
        @Test func biasedSwigluKeepsTheUnfusedBits() {
            for dtype in [DType.bfloat16, .float16] {
                let values: [Float] = (0 ..< 65_536).map { pattern in
                    dtype == .bfloat16
                        ? Float(bitPattern: UInt32(pattern) << 16)
                        : Float(Float16(bitPattern: UInt16(pattern)))
                }
                let every = MLXArray(values).asType(dtype).reshaped(256, 256)
                let other = normal([256, 256], 71, dtype, scale: 4)
                let biasA = normal([256, 256], 72, dtype)
                let biasB = normal([256, 256], 73, dtype)
                for (linear, glu) in [(every, other), (other, every)] {
                    let fused = compiledBiasedSwiglu([linear, glu, biasA, biasB])[0]
                    let unfused = compiledSwiglu(linear + biasA, glu + biasB)
                    #expect(bytes(fused) == bytes(unfused), "\(dtype)")
                }
            }
        }

        /// The prologue writes what the bias adds, transpose and `YarnRoPE`
        /// (its scale and `MLXFast.RoPE` with its frequencies) write.
        @Test func attentionPrologueKeepsTheUnfusedBits() throws {
            // gpt-oss-20b's rotation: head 64, theta 150,000, YaRN factor 32.
            let yarn = YarnRoPE(
                dimensions: 64, base: 150_000, scalingFactor: 32,
                originalMaxPositionEmbeddings: 4096, betaFast: 32, betaSlow: 1)
            let exact = try #require(yarn.exactRotation)
            #expect(exact.mscale != 1)
            for dtype in [DType.bfloat16, .float16] {
                guard
                    let rotation = Gemma4FusedNorms.RotationCache().rotation(
                        headDim: 64, base: nil, frequencies: exact.frequencies, dtype: dtype)
                else {
                    #expect(!verified, "\(dtype)")
                    continue
                }
                for (batch, tokens, offsets) in [
                    (1, 1, [0]), (4, 1, [0, 5, 4_095, 131_071]), (2, 3, [17, 70_000]),
                ] {
                    let (heads, kvHeads, D) = (64, 8, 64)
                    let seed = UInt64(batch * 31 + tokens * 7 + offsets.count)
                    let qLinear = normal([batch, tokens, heads * D], seed, dtype, scale: 3)
                    let kLinear = normal([batch, tokens, kvHeads * D], seed + 1, dtype, scale: 3)
                    let vLinear = normal([batch, tokens, kvHeads * D], seed + 2, dtype, scale: 3)
                    let qBias = normal([heads * D], seed + 3, dtype)
                    let kBias = normal([kvHeads * D], seed + 4, dtype)
                    let vBias = normal([kvHeads * D], seed + 5, dtype)
                    let positions = MLXArray(offsets.map { Int32($0) })
                    let fused = GPTOSSFusedNorms.attentionPrologue(
                        queries: qLinear, keys: kLinear, values: vLinear,
                        queryBias: qBias, keyBias: kBias, valueBias: vBias,
                        queryHeads: heads, keyValueHeads: kvHeads, headDim: D,
                        mscale: exact.mscale, offsets: positions, rotation: rotation)
                    // The unfused ops: each `QuantizedLinear`'s bias add,
                    // reshape and transpose, then `YarnRoPE`.
                    let q = (qLinear + qBias).reshaped(batch, tokens, heads, D).swappedAxes(1, 2)
                    let k = (kLinear + kBias).reshaped(batch, tokens, kvHeads, D).swappedAxes(1, 2)
                    let v = (vLinear + vBias).reshaped(batch, tokens, kvHeads, D).swappedAxes(1, 2)
                    let qRef = yarn(q, offset: positions)
                    let kRef = yarn(k, offset: positions)
                    let label = "\(dtype) batch \(batch) tokens \(tokens)"
                    #expect(bytes(fused.queries) == bytes(contiguous(qRef)), "q \(label)")
                    #expect(bytes(fused.keys) == bytes(contiguous(kRef)), "k \(label)")
                    #expect(bytes(fused.values) == bytes(contiguous(v)), "v \(label)")
                }
            }
        }

        @Test func switchOffKeepsTheUnfusedOps() {
            GPTOSSFusedNorms.$enabledOverride.withValue(false) {
                #expect(!GPTOSSFusedNorms.supports(width: 2880, dtype: .bfloat16))
                #expect(GPTOSSFusedNorms.routingForm(experts: 32, topK: 4, dtype: .bfloat16) == nil)
            }
            GPTOSSFusedNorms.$routingOverride.withValue(false) {
                #expect(GPTOSSFusedNorms.routingForm(experts: 32, topK: 4, dtype: .bfloat16) == nil)
            }
        }
    }
}
