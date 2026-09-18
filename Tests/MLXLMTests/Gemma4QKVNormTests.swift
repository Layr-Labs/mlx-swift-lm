// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXFast
import Testing
@testable import MLXLMCommon

/// Native tests authored only: odd prefill tails, asymmetric last-query shapes,
/// K=V ownership, both rope forms and scalar/vector decode require GPU execution.
@Suite("Gemma4 QKV norm and RoPE", .serialized)
struct Gemma4QKVNormTests {
    private func values(_ shape: [Int], salt: Int) -> MLXArray {
        let count = shape.reduce(1, *)
        let data: [Float] = (0..<count).map { Float(($0 * 17 + salt) % 127 - 63) / 64 }
        return MLXArray(data, shape).asType(.bfloat16)
    }
    @Test func fusedOutputsMatchCurrentNormAndRope() throws {
      // Independent platform expectation: a broken production admission flag
      // must not turn these strict RoPE comparisons into norm-only passes.
      let platformHasRelaxedMath: Bool
      if #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) { platformHasRelaxedMath = true }
      else { platformHasRelaxedMath = false }
      for ropeEnabled in [false, true] {
        let policy = Gemma4QKVNormPolicy(environment: ["DARKBLOOM_GEMMA4_QKV_NORM": "1",
            "DARKBLOOM_GEMMA4_QKV_NORM_PREFILL": "1", "DARKBLOOM_GEMMA4_QKV_NORM_ROPE": ropeEnabled ? "1" : "0"])
        for (dimension, heads, shared) in [(256, 8, false), (512, 2, true)] {
            for (batch, lq, lk, equalOffsets) in [(1, 1, 1, true), (8, 1, 1, true),
                                                (1, 1025, 1025, true), (1, 1, 1025, false), (2, 1024, 1024, true)] {
                let q = values([batch, lq, 16, dimension], salt: 1)
                let k = values([batch, lk, heads, dimension], salt: 7)
                let v = shared ? k : values([batch, lk, heads, dimension], salt: 11)
                let qw = values([dimension], salt: 13), kw = values([dimension], salt: 17)
                let offsets = MLXArray((0..<batch).map { Int32(17 + $0 * 1024) })
                let frequencies = MLXArray((0..<(dimension / 2)).map { index -> Float in
                    index < dimension / 8 ? pow(Float(1_000_000), Float(index * 2) / Float(dimension)) : .infinity
                })
                let rope: Gemma4QKVNormV1.RopeParameters = shared ? .init(frequencies: frequencies) : .init(log2Base: log2f(10_000))
                for ready in [false, true] {
                    if ready { eval(q, k, v, qw, kw) }
                    let actual = try #require(Gemma4QKVNormV1.apply(q: q, k: k, v: v, qWeight: qw, kWeight: kw,
                        eps: 1e-6, keyValueShared: shared, positionOffsets: offsets, rope: rope,
                        equalQueryKeyOffsets: equalOffsets, targetEligible: true, scheduledPrefill: lk > 1, policy: policy))
                    var qr = MLXFast.rmsNorm(q, weight: qw, eps: 1e-6).transposed(0, 2, 1, 3)
                    var kr = MLXFast.rmsNorm(k, weight: kw, eps: 1e-6).transposed(0, 2, 1, 3)
                    let vr = MLXFast.rmsNorm(v, weight: MLXArray.mlxNone, eps: 1e-6).transposed(0, 2, 1, 3)
                    #expect(actual.appliedRope == (ropeEnabled && equalOffsets && platformHasRelaxedMath))
                    if actual.appliedRope {
                        qr = MLXFast.RoPE(qr, dimensions: dimension, traditional: false,
                            base: shared ? nil : 10_000, scale: 1, offset: offsets, freqs: shared ? frequencies : nil)
                        kr = MLXFast.RoPE(kr, dimensions: dimension, traditional: false,
                            base: shared ? nil : 10_000, scale: 1, offset: offsets, freqs: shared ? frequencies : nil)
                    }
                    let label = "D\(dimension) B\(batch) LQ\(lq) LK\(lk) shared\(shared) ready\(ready) rope\(ropeEnabled)"
                    gemma4ExpectExactBytes(actual.q, qr, label: label + " Q")
                    gemma4ExpectExactBytes(actual.k, kr, label: label + " K")
                    gemma4ExpectExactBytes(actual.v, vr, label: label + " V")
                }
            }
        }
      }
    }

    @Test func sharedInputAndPrefillPhaseMustBeProven() {
        let policy = Gemma4QKVNormPolicy(environment: ["DARKBLOOM_GEMMA4_QKV_NORM": "1",
            "DARKBLOOM_GEMMA4_QKV_NORM_PREFILL": "1"])
        let q = values([1, 1025, 16, 512], salt: 1), k = values([1, 1025, 2, 512], salt: 3)
        let fakeShared = k + 0, w = values([512], salt: 5), offsets = MLXArray([Int32(0)])
        #expect(Gemma4QKVNormV1.apply(q: q, k: k, v: fakeShared, qWeight: w, kWeight: w, eps: 1e-6,
            keyValueShared: true, positionOffsets: offsets, rope: nil, equalQueryKeyOffsets: true,
            targetEligible: true, scheduledPrefill: true, policy: policy) == nil)
        #expect(Gemma4QKVNormV1.apply(q: q, k: k, v: k, qWeight: w, kWeight: w, eps: 1e-6,
            keyValueShared: true, positionOffsets: offsets, rope: nil, equalQueryKeyOffsets: true,
            targetEligible: true, scheduledPrefill: false, policy: policy) == nil)
    }
}
