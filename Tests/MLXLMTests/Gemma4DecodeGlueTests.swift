// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast
import Testing
@testable import MLXLMCommon

/// Native tests authored in the source-only phase. No checkpoint dependency,
/// but actual MLX/Metal execution requires an authorized lane and resources.
@Suite("Gemma4 per-row decode glue", .serialized)
struct Gemma4DecodeGlueTests {
    private func context(paired: Bool, assistant: Bool = false, localBroadcast: Bool = false) -> Gemma4DecodeGluePolicy.Context {
        Gemma4DecodeGluePolicy(environment: ["DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE": "1",
            "DARKBLOOM_GEMMA4_DRAFTER_NORM_RESIDUAL_FUSE": "1",
            "DARKBLOOM_GEMMA4_DECODE_PAIRED_RMS": paired ? "1" : "0",
            "DARKBLOOM_GEMMA4_NORM_TG_BARRIER_HALVE": localBroadcast ? "1" : "0"])
            .context(target: !assistant, validatedAssistant: assistant)!
    }
    private func values(_ shape: [Int], salt: Int) -> MLXArray {
        MLXArray((0..<shape.reduce(1, *)).map { Float(($0 * 13 + salt) % 127 - 63) / 64 }, shape).asType(.bfloat16)
    }
    private func identical(_ a: MLXArray, _ b: MLXArray) {
        #expect(a.shape == b.shape && a.dtype == b.dtype)
        #expect(a.asData(access: .copy).data == b.asData(access: .copy).data)
    }

    @Test func targetRowsAndPairedVariantsAreByteExact() throws {
        for batch in [1, 2, 4, 8, 9] {
            for paired in [false, true] {
              for localBroadcast in [false, true] {
                let x = values([batch, 1, 2816], salt: 1), y = values([batch, 1, 2816], salt: 7)
                let residual = values([batch, 1, 2816], salt: 11)
                let w1 = values([2816], salt: 13), w2 = values([2816], salt: 17)
                let w3 = values([2816], salt: 19), next = values([2816], salt: 23)
                let scalar = MLXArray(Float(0.875)).asType(.bfloat16), ctx = context(paired: paired, localBroadcast: localBroadcast)
                let n1 = MLXFast.rmsNorm(x, weight: w1, eps: 1e-6)
                let residualOut = try #require(Gemma4DecodeGlueV1.normResidual(
                    x: x, residual: residual, weight: w1, eps: 1e-6, context: ctx))
                identical(residualOut, residual + n1)
                let dual = try #require(Gemma4DecodeGlueV1.dualPreNorm(x: x, w1: w1, w2: w2, eps: 1e-6, context: ctx))
                identical(dual.0, n1)
                identical(dual.1, MLXFast.rmsNorm(x, weight: w2, eps: 1e-6))
                let sum = n1 + MLXFast.rmsNorm(y, weight: w2, eps: 1e-6)
                let expected = (residual + MLXFast.rmsNorm(sum, weight: w3, eps: 1e-6)) * scalar
                let tail = try #require(Gemma4DecodeGlueV1.branchTail(h1: x, h2: y, residual: residual,
                    w1: w1, w2: w2, w3: w3, layerScalar: scalar, eps: 1e-6, context: ctx))
                identical(tail, expected)
                let chain = try #require(Gemma4DecodeGlueV1.branchTailChained(h1: x, h2: y, residual: residual,
                    w1: w1, w2: w2, w3: w3, layerScalar: scalar, nextWeight: next, eps: 1e-6, context: ctx))
                identical(chain.out, expected)
                identical(chain.normalized, MLXFast.rmsNorm(expected, weight: next, eps: 1e-6))
              }
            }
        }
    }

    @Test func assistantNormKeepsItsOwnGeometry() throws {
        for batch in [1, 2, 8] {
            let x = values([batch, 1, 1024], salt: 1), residual = values([batch, 1, 1024], salt: 7)
            let w = values([1024], salt: 11), ctx = context(paired: false, assistant: true, localBroadcast: true)
            #expect(!ctx.localRMSBroadcast)
            let out = try #require(Gemma4DecodeGlueV1.normResidual(x: x, residual: residual, weight: w, eps: 1e-6, context: ctx))
            identical(out, residual + MLXFast.rmsNorm(x, weight: w, eps: 1e-6))
            #expect(Gemma4DecodeGlueV1.dualPreNorm(x: x, w1: w, w2: w, eps: 1e-6, context: ctx) == nil)
        }
    }

    @Test func carriesRejectDescriptorChangesAndConsumeOnce() throws {
        let source = values([1, 1, 2816], salt: 3), weight = values([2816], salt: 7)
        let normalized = MLXFast.rmsNorm(source, weight: weight, eps: 1e-6)
        let valid = try #require(Gemma4DecodeNormalizationCarry.capture(source: source, normalized: normalized, weight: weight, eps: 1e-6))
        eval(source, normalized)
        #expect(valid.take(source: source, weight: weight, eps: 1e-6) === normalized)
        #expect(valid.take(source: source, weight: weight, eps: 1e-6) == nil)
        let staleWeight = try #require(Gemma4DecodeNormalizationCarry.capture(source: source, normalized: normalized, weight: weight, eps: 1e-6))
        weight._updateInternal(values([2816], salt: 13))
        #expect(staleWeight.take(source: source, weight: weight, eps: 1e-6) == nil)
        let staleSource = try #require(Gemma4DecodeNormalizationCarry.capture(source: source, normalized: normalized, weight: weight, eps: 1e-6))
        source._updateInternal(values([1, 1, 2816], salt: 17))
        #expect(staleSource.take(source: source, weight: weight, eps: 1e-6) == nil)
        let otherStream = try #require(Gemma4DecodeNormalizationCarry.capture(source: source, normalized: normalized, weight: weight, eps: 1e-6))
        Stream.withNewDefaultStream(device: .gpu) {
            #expect(otherStream.take(source: source, weight: weight, eps: 1e-6) == nil)
        }
    }

    @Test func localBroadcastSourceKeepsTreesAndSeparateScratch() throws {
        let sources = [Gemma4DecodeGlueSources.normResidual, Gemma4DecodeGlueSources.dualPreNorm,
            Gemma4DecodeGlueSources.tail, Gemma4DecodeGlueSources.tailChained,
            Gemma4DecodeGlueSources.pairedRmsTailSource(Gemma4DecodeGlueSources.tail),
            Gemma4DecodeGlueSources.pairedRmsTailSource(Gemma4DecodeGlueSources.tailChained)]
        for original in sources {
            let transformed = try #require(Gemma4RMSBroadcastSources.transform(original))
            let oldBarriers = original.components(separatedBy: "threadgroup_barrier(").count - 1
            let newBarriers = transformed.components(separatedBy: "threadgroup_barrier(").count - 1
            #expect(oldBarriers == newBarriers * 2)
            let reductions = original.components(separatedBy: "metal::precise::rsqrt").count - 1
            #expect(transformed.components(separatedBy: "threadgroup float tb_sums_").count - 1 == reductions)
            #expect(transformed.components(separatedBy: "metal::precise::rsqrt").count - 1 == reductions)
            #expect(!transformed.contains("threadgroup float local_inv"))
            #expect(Gemma4RMSBroadcastSources.transform(transformed) == nil)
        }
        #expect(Gemma4RMSBroadcastSources.transform(Gemma4DecodeGlueSources.normResidual1024) == nil)
    }

    @Test func localBroadcastRetainsExceptionalWords() throws {
        let x = MLXArray((0..<(24 * 2816)).map { UInt16(truncatingIfNeeded: $0) })
            .view(dtype: .bfloat16).reshaped(24, 1, 2816)
        let w = MLXArray.ones([2816], dtype: .bfloat16)
        let residual = MLXArray.zeros(x.shape, dtype: .bfloat16)
        let stock = residual + MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
        for local in [false, true] {
            let actual = try #require(Gemma4DecodeGlueV1.normResidual(x: x, residual: residual,
                weight: w, eps: 1e-6, context: context(paired: false, localBroadcast: local)))
            identical(actual, stock)
        }
    }

    @Test func meanRoundingMatchesNativeAcrossExponents() throws {
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
            let x = MLXArray(data).view(dtype: .bfloat16).reshaped(256, 1, 2816)
            let w = MLXArray(weights).view(dtype: .bfloat16)
            let residual = MLXArray.zeros(x.shape, dtype: .bfloat16)
            let expected = residual + MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
            for local in [false, true] {
                let actual = try #require(Gemma4DecodeGlueV1.normResidual(x: x, residual: residual,
                    weight: w, eps: 1e-6, context: context(paired: false, localBroadcast: local)))
                identical(actual, expected)
            }
        }
    }
}
