// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast
import Testing
@testable import MLXLMCommon

/// Native synthetic gates for the scheduled-prefill paths.
/// These use MLX/Metal but no checkpoint. An authorized GPU lane is required.
@Suite("Gemma4 scheduled-prefill glue", .serialized)
struct Gemma4PrefillGlueTests {
    private func context(vectorized: Bool = false, chained: Bool = true,
                         prefix: Bool = false, scatter: Bool = false,
                         expertTail: Bool = false) -> Gemma4PrefillGluePolicy.Context {
        Gemma4PrefillGluePolicy(environment: [
            "DARKBLOOM_GEMMA4_PREFILL_GLUE": "1",
            "DARKBLOOM_GEMMA4_PREFILL_GLUE_VEC4": vectorized ? "1" : "0",
            "DARKBLOOM_GEMMA4_PREFILL_GLUE_CHAIN": chained ? "1" : "0",
            "DARKBLOOM_GEMMA4_PREFILL_BRANCH_PREFIX": prefix ? "1" : "0",
            "DARKBLOOM_GEMMA4_PREFILL_PRENORM_GATHER": scatter ? "1" : "0",
            "DARKBLOOM_GEMMA4_PREFILL_EXPERT_TAIL_FUSION": expertTail ? "1" : "0",
        ]).context(scheduledPrefill: true, eligibleModel: true)!
    }

    private func values(_ shape: [Int], salt: Int) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0..<count).map { Float(($0 * 17 + salt) % 113 - 56) / 64 }, shape)
            .asType(.bfloat16)
    }

    private func identical(_ actual: MLXArray, _ expected: MLXArray) {
        #expect(actual.shape == expected.shape)
        #expect(actual.dtype == expected.dtype)
        #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
    }

    @Test func meanRoundingMatchesNativeAcrossExponents() throws {
        // Same deterministic adversarial distribution that exposed decode's
        // FP32 mean-rounding error. Prefill uses a separate shared helper.
        var state: UInt32 = 123456789
        func random() -> UInt32 {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5
            return state
        }
        for _ in 0..<8 {
            let data: [UInt16] = (0..<(256 * 2816)).map { _ in
                let r = random()
                return UInt16((r & 0x8000) | ((117 + r % 21) << 7) | ((r >> 16) & 127))
            }
            let weights: [UInt16] = (0..<2816).map { _ in
                let r = random()
                return UInt16(((126 + r % 3) << 7) | ((r >> 16) & 127))
            }
            let x = MLXArray(data).view(dtype: .bfloat16).reshaped(1, 256, 2816)
            let w = MLXArray(weights).view(dtype: .bfloat16)
            let zeros = MLXArray.zeros(x.shape, dtype: .bfloat16)
            let expected = MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
            let tail = MLXFast.rmsNorm(expected + expected, weight: w, eps: 1e-6)
            let scalar = MLXArray(Float(0.875)).asType(.bfloat16)
            let expectedChain = (zeros + tail) * scalar
            eval(x, w)
            for vectorized in [false, true] {
                let ctx = context(vectorized: vectorized)
                identical(try #require(Gemma4PrefillGlueV1.preNorm(
                    x: x, weight: w, eps: 1e-6, context: ctx)), expected)
                identical(try #require(Gemma4PrefillGlueV1.normResidual(
                    x: x, weight: w, residual: zeros, eps: 1e-6, context: ctx)), zeros + expected)
                let dual = try #require(Gemma4PrefillGlueV1.dualPreNorm(
                    x: x, w1: w, w2: w, eps: 1e-6, context: ctx))
                identical(dual.0, expected)
                identical(dual.1, expected)
                let chained = try #require(Gemma4PrefillGlueV1.branchTailChained(
                    h1: x, h2: x, w1: w, w2: w, w3: w, residual2: zeros,
                    layerScalar: scalar, nextInputNormWeight: w, eps: 1e-6, context: ctx))
                identical(chained.out, expectedChain)
                identical(chained.normedNext, MLXFast.rmsNorm(expectedChain, weight: w, eps: 1e-6))
            }
        }
    }

    @Test func meanRoundingRetainsExceptionalWords() throws {
        let x = MLXArray((0..<(24 * 2816)).map { UInt16(truncatingIfNeeded: $0) })
            .view(dtype: .bfloat16).reshaped(1, 24, 2816)
        let w = MLXArray.ones([2816], dtype: .bfloat16)
        let expected = MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
        eval(x, w)
        for vectorized in [false, true] {
            identical(try #require(Gemma4PrefillGlueV1.preNorm(
                x: x, weight: w, eps: 1e-6, context: context(vectorized: vectorized))), expected)
        }
    }

    @Test func allFourFusionsAreByteExact() throws {
        for vectorized in [false, true] {
            for shape in [[1, 2, 2816], [2, 3, 2816], [4, 7, 2816], [8, 2, 2816],
                          [1, 1024, 2816], [8, 1024, 2816]] {
                let x = values(shape, salt: 1), h2 = values(shape, salt: 3)
                let residual = values(shape, salt: 7)
                let w1 = values([2816], salt: 5), w2 = values([2816], salt: 11)
                let w3 = values([2816], salt: 13), next = values([2816], salt: 19)
                let scalar = MLXArray(Float(0.875)).asType(.bfloat16)
                let ctx = context(vectorized: vectorized)
                if vectorized {
                    eval(x, h2, residual)
                    #expect(Gemma4PrefillGlueV1.usesVectorLoads(ctx, activations: [x, h2, residual]))
                }
                let n1 = MLXFast.rmsNorm(x, weight: w1, eps: 1e-6)
                let n2 = MLXFast.rmsNorm(x, weight: w2, eps: 1e-6)
                let direct = try #require(Gemma4PrefillGlueV1.preNorm(
                    x: x, weight: w1, eps: 1e-6, context: ctx))
                identical(direct, n1)
                let fusedResidual = try #require(Gemma4PrefillGlueV1.normResidual(
                    x: x, weight: w1, residual: residual, eps: 1e-6, context: ctx))
                identical(fusedResidual, residual + n1)
                let dual = try #require(Gemma4PrefillGlueV1.dualPreNorm(
                    x: x, w1: w1, w2: w2, eps: 1e-6, context: ctx))
                identical(dual.0, n1)
                identical(dual.1, n2)
                let branch = n1 + MLXFast.rmsNorm(h2, weight: w2, eps: 1e-6)
                let tail = residual + MLXFast.rmsNorm(branch, weight: w3, eps: 1e-6)
                let fusedTail = try #require(Gemma4PrefillGlueV1.branchTail(
                    h1: x, h2: h2, w1: w1, w2: w2, w3: w3,
                    residual2: residual, eps: 1e-6, context: ctx))
                identical(fusedTail, tail)
                let chained = try #require(Gemma4PrefillGlueV1.branchTailChained(
                    h1: x, h2: h2, w1: w1, w2: w2, w3: w3, residual2: residual,
                    layerScalar: scalar, nextInputNormWeight: next, eps: 1e-6, context: ctx))
                identical(chained.out, tail * scalar)
                identical(chained.normedNext, MLXFast.rmsNorm(tail * scalar, weight: next, eps: 1e-6))
            }
        }
    }

    @Test func unalignedViewsAndLazyInputsDoNotUseVectorLoads() throws {
        let ctx = context(vectorized: true)
        let backing = values([2 * 2816 + 1], salt: 23)
        let x = backing[1...].reshaped([1, 2, 2816])
        #expect(!Gemma4PrefillGlueV1.usesVectorLoads(ctx, activations: [x]))
        eval(x)
        // The view is row contiguous but offset by one BF16 element (2 bytes).
        #expect(!Gemma4PrefillGlueV1.usesVectorLoads(ctx, activations: [x]))
        let w = values([2816], salt: 7)
        let residual = values([1, 2, 2816], salt: 11)
        let actual = try #require(Gemma4PrefillGlueV1.normResidual(
            x: x, weight: w, residual: residual, eps: 1e-6, context: ctx))
        identical(actual, residual + MLXFast.rmsNorm(x, weight: w, eps: 1e-6))
    }

    @Test func cpuStreamAndMalformedMetadataFallBack() {
        let ctx = context()
        let x = values([1, 2, 2816], salt: 1), w = values([2816], salt: 3)
        #expect(Gemma4PrefillGlueV1.normResidual(x: x, weight: w, residual: x,
            eps: 1e-6, context: ctx, stream: .cpu) == nil)
        #expect(Gemma4PrefillGlueV1.normResidual(x: x, weight: w, residual: x,
            eps: 1e-5, context: ctx) == nil)
        #expect(Gemma4PrefillGlueV1.normResidual(x: x.asType(.float32), weight: w,
            residual: x, eps: 1e-6, context: ctx) == nil)
        #expect(Gemma4PrefillGlueV1.dualPreNorm(x: x, w1: w, w2: w[0..<2815],
            eps: 1e-6, context: ctx) == nil)
        #expect(Gemma4PrefillGlueV1.branchTailChained(h1: x, h2: x, w1: w, w2: w, w3: w,
            residual2: x, layerScalar: MLXArray(Float(1)), nextInputNormWeight: w,
            eps: 1e-6, context: context(chained: false)) == nil)
    }

    @Test func changingWeightsDoesNotReusePriorParameters() throws {
        let x = values([1, 2, 2816], salt: 29), residual = values([1, 2, 2816], salt: 13)
        let weight = values([2816], salt: 31)
        let ctx = context()
        let first = try #require(Gemma4PrefillGlueV1.normResidual(
            x: x, weight: weight, residual: residual, eps: 1e-6, context: ctx))
        let original = residual + MLXFast.rmsNorm(x, weight: weight, eps: 1e-6)
        weight._updateInternal(values([2816], salt: 47))
        let second = try #require(Gemma4PrefillGlueV1.normResidual(
            x: x, weight: weight, residual: residual, eps: 1e-6, context: ctx))
        identical(first, original)
        identical(second, residual + MLXFast.rmsNorm(x, weight: weight, eps: 1e-6))
    }

    @Test func branchPrefixPreservesAllThreeOutputs() throws {
        for vectorized in [false, true] {
            for shape in [[1, 8, 2816], [8, 1024, 2816]] {
                let x = values(shape, salt: 5), residual = values(shape, salt: 11)
                let w = values([2816], salt: 7), dense = values([2816], salt: 13)
                let router = values([2816], salt: 17) * (1 / Float(2816).squareRoot())
                if vectorized { eval(x, residual) }
                let actual = try #require(Gemma4PrefillGlueV1.attentionBranchPrefix(attn: x,
                    residual: residual, wPostAttn: w, wDense: dense, wRouter: router, eps: 1e-6,
                    context: context(vectorized: vectorized, prefix: true)))
                let out = residual + MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
                identical(actual.out, out)
                identical(actual.denseNorm, MLXFast.rmsNorm(out, weight: dense, eps: 1e-6))
                identical(actual.routerNorm, MLXFast.rmsNorm(out, weight: router, eps: 1e-6))
                #expect(Gemma4PrefillGlueV1.attentionBranchPrefix(attn: x, residual: residual,
                    wPostAttn: w, wDense: dense, wRouter: router, eps: 1e-6, context: context()) == nil)
            }
        }
    }

    @Test func ownedScatterMatchesNormThenOriginalGatherOrder() throws {
        for vectorized in [false, true] {
            for shape in [[1, 8, 2816], [2, 7, 2816], [8, 1024, 2816]] {
                let rows = shape[0] * shape[1]
                for tied in [false, true] {
                    let ids = (0..<(rows * 8)).map { UInt32(tied ? 7 : ($0 * 17) % 128) }
                    let indices = MLXArray(ids, [rows, 8])
                    let x = values(shape, salt: 19), w = values([2816], salt: 23)
                    if vectorized { eval(x) }
                    let ctx = context(vectorized: vectorized, scatter: true)
                    let owned = try #require(Gemma4PrefillExpertOrder.make(indices: indices, rows: rows, context: ctx))
                    let actual = try #require(Gemma4PrefillGlueV1.preNormScatter(
                        x: x, weight: w, order: owned, eps: 1e-6, context: ctx))
                    // Frozen gatherSort contract: same argSort/inverse operations,
                    // and gather the normalized token row for each assignment.
                    let flat = indices.flattened()
                    let order = argSort(flat)
                    let inverse = argSort(order)
                    let normal = MLXFast.rmsNorm(x, weight: w, eps: 1e-6).reshaped(rows, 1, 2816)
                    identical(actual, normal[order.floorDivide(8)])
                    identical(owned.sortedIndices, flat[order])
                    identical(owned.inverseOrder, inverse)
                }
            }
        }
    }

    @Test func orderAdmissionAndPerCallLifetime() throws {
        let ctx = context(scatter: true)
        let indices = MLXArray.zeros([8, 8], dtype: .uint32)
        #expect(Gemma4PrefillExpertOrder.make(indices: indices, rows: 8, context: context()) == nil)
        #expect(Gemma4PrefillExpertOrder.make(indices: indices.asType(.int32), rows: 8, context: ctx) == nil)
        #expect(Gemma4PrefillExpertOrder.make(indices: indices, rows: 7, context: ctx) == nil)
        let first = try #require(Gemma4PrefillExpertOrder.make(indices: indices, rows: 8, context: ctx))
        indices._updateInternal(MLXArray(Array(repeating: UInt32(127), count: 64), [8, 8]))
        let second = try #require(Gemma4PrefillExpertOrder.make(indices: indices, rows: 8, context: ctx))
        #expect(first.sortedIndices.asArray(UInt32.self) == Array(repeating: UInt32(0), count: 64))
        #expect(second.sortedIndices.asArray(UInt32.self) == Array(repeating: UInt32(127), count: 64))
    }

    @Test func fusedExpertTailMatchesOriginalReductionAndChain() throws {
        for vectorized in [false, true] {
            for shape in [[1, 8, 2816], [2, 7, 2816], [8, 1024, 2816]] {
                let rows = shape[0] * shape[1]
                let ctx = context(vectorized: vectorized, expertTail: true)
                let indices = MLXArray((0..<(rows * 8)).map { UInt32(($0 * 17) % 128) }, [rows, 8])
                let order = try #require(Gemma4PrefillExpertOrder.make(indices: indices, rows: rows, context: ctx))
                let sorted = values([rows * 8, 2816], salt: 23), weights = values([rows, 8], salt: 7)
                let h1 = values(shape, salt: 31), residual = values(shape, salt: 13)
                let w1 = values([2816], salt: 3), w2 = values([2816], salt: 5)
                let w3 = values([2816], salt: 11), next = values([2816], salt: 19)
                let scalar = MLXArray(Float(0.875)).asType(.bfloat16)
                if vectorized { eval(h1, residual) }
                let referenceExpert = weightedExpertUnsort(sortedOutputs: sorted,
                    inverseOrder: order.inverseOrder, weights: weights).reshaped(shape)
                let branch = MLXFast.rmsNorm(h1, weight: w1, eps: 1e-6)
                    + MLXFast.rmsNorm(referenceExpert, weight: w2, eps: 1e-6)
                let expected = (residual + MLXFast.rmsNorm(branch, weight: w3, eps: 1e-6)) * scalar
                let pending = try #require(Gemma4PrefillExpertProjection(sorted: sorted, order: order, weights: weights))
                resetWeightedExpertUnsortStats()
                let actual = try #require(Gemma4PrefillGlueV1.branchTailChainedUnsort(h1: h1, expert: pending,
                    w1: w1, w2: w2, w3: w3, residual2: residual, layerScalar: scalar,
                    nextInputNormWeight: next, eps: 1e-6, context: ctx))
                #expect(weightedExpertUnsortStats().effectiveCalls == 1)
                #expect(pending.pendingForFusion == nil)
                identical(actual.out, expected)
                identical(actual.normedNext, MLXFast.rmsNorm(expected, weight: next, eps: 1e-6))
            }
        }
    }

    @Test func pendingSnapshotsAndFallbackResolveOnlyOnce() throws {
        let ctx = context(expertTail: true)
        let order = try #require(Gemma4PrefillExpertOrder.make(
            indices: MLXArray.zeros([8, 8], dtype: .uint32), rows: 8, context: ctx))
        let sorted = values([64, 2816], salt: 11), weights = values([8, 8], salt: 17)
        let original = weightedExpertUnsort(sortedOutputs: sorted, inverseOrder: order.inverseOrder, weights: weights)
        let pending = try #require(Gemma4PrefillExpertProjection(sorted: sorted, order: order, weights: weights))
        sorted._updateInternal(MLXArray.zeros([64, 2816], dtype: .bfloat16))
        weights._updateInternal(MLXArray.ones([8, 8], dtype: .bfloat16))
        order.inverseOrder._updateInternal(MLXArray(Array((0..<64).reversed()).map(UInt32.init)))
        let x = values([1, 8, 2816], salt: 7), w = values([2816], salt: 3)
        let scalar = MLXArray(Float(1)).asType(.bfloat16)
        resetWeightedExpertUnsortStats()
        #expect(Gemma4PrefillGlueV1.branchTailChainedUnsort(h1: x, expert: pending,
            w1: w, w2: w, w3: w, residual2: x, layerScalar: scalar,
            nextInputNormWeight: w, eps: 1e-5, context: ctx) == nil)
        #expect(weightedExpertUnsortStats().effectiveCalls == 0)
        #expect(pending.pendingForFusion != nil)
        let originalStream = pending.stream
        Stream.withNewDefaultStream(device: .gpu) {
            #expect(pending.stream == originalStream)
            #expect(Gemma4PrefillGlueV1.branchTailChainedUnsort(h1: x, expert: pending,
                w1: w, w2: w, w3: w, residual2: x, layerScalar: scalar,
                nextInputNormWeight: w, eps: 1e-6, context: ctx) == nil)
        }
        resetWeightedExpertUnsortStats()
        let once = pending.resolve()
        #expect(pending.resolve() === once)
        #expect(weightedExpertUnsortStats().effectiveCalls == 1)
        identical(once, original)
    }
}
