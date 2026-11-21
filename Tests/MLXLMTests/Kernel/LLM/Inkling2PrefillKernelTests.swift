import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

/// Prefill kernels of Inkling 2 on this device: the full-FP32 tiled product
/// (accurate, row-independent, used only outside row-exact steps) and the
/// expert combine from sorted rows (byte-identical to the MLX expression).
extension KernelTests {
    @Suite(.serialized)
    struct Inkling2PrefillKernelTests {
        static let shapes: [(String, Int, Int)] = [
            ("router", 2304, 128), ("gate", 64, 2304), ("messages", 2304, 48), ("mix", 2304, 3),
        ]

        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && arrayEqual(a, b).item(Bool.self)
        }

        /// Close to a float64 reference: |err| <= 1e-5 * sum_k |x_k w_k| per output.
        @Test func exactFP32MatchesFloat64() throws {
            for (index, (name, k, n)) in Self.shapes.enumerated() {
                let x = MLXRandom.normal([37, k], key: MLXRandom.key(UInt64(40 + index)))
                let w = MLXRandom.normal([n, k], key: MLXRandom.key(UInt64(50 + index))) * 0.05
                let y = Inkling2ExactFP32.matmul(x, transposedWeight: w).asArray(Float.self)
                let xs = x.asArray(Float.self)
                let ws = w.asArray(Float.self)
                var worst = 0.0
                for r in 0 ..< 37 {
                    for c in 0 ..< n {
                        var exact = 0.0
                        var scale = 0.0
                        for i in 0 ..< k {
                            let term = Double(xs[r * k + i]) * Double(ws[c * k + i])
                            exact += term
                            scale += abs(term)
                        }
                        worst = max(worst, abs(Double(y[r * n + c]) - exact) / max(scale, 1e-30))
                    }
                }
                #expect(worst <= 1e-5, "\(name): relative error \(worst)")
            }
        }

        /// A row's result does not depend on the rows computed with it.
        @Test(arguments: [1, 5, 63, 64, 65, 130])
        func exactFP32RowsAreIndependent(rows: Int) throws {
            for (index, (name, k, n)) in Self.shapes.enumerated() {
                let x = MLXRandom.normal([200, k], key: MLXRandom.key(UInt64(60 + index)))
                let w = MLXRandom.normal([n, k], key: MLXRandom.key(UInt64(70 + index))) * 0.05
                let all = Inkling2ExactFP32.matmul(x, transposedWeight: w)
                let part = Inkling2ExactFP32.matmul(x[7 ..< 7 + rows], transposedWeight: w)
                #expect(Self.bitsEqual(all[7 ..< 7 + rows], part), "\(name) at \(rows) rows")
            }
        }

        /// Prefill takes the kernel, row-exact steps keep the per-row gemv.
        @Test func fp32RowsRoutesByStepKind() throws {
            let x = MLXRandom.normal([2, 9, 2304], key: MLXRandom.key(80))
            let w = MLXRandom.normal([128, 2304], key: MLXRandom.key(81)) * 0.05
            let prefill = Inkling2ExactFP32.$override.withValue(true) {
                Inkling2RowExact.fp32Rows(x, w)
            }
            #expect(
                Self.bitsEqual(
                    prefill,
                    Inkling2ExactFP32.matmul(x.reshaped(18, 2304), transposedWeight: w)
                        .reshaped(2, 9, 128)))
            let rowExact = Inkling2RowExact.$active.withValue(true) {
                Inkling2RowExact.fp32Rows(x, w)
            }
            let gemv = MLX.matmul(w.expandedDimensions(axis: 0), x.reshaped(18, 2304, 1))
                .reshaped(2, 9, 128)
            #expect(Self.bitsEqual(rowExact, gemv))
        }

        /// The prefill mix kernel agrees with the plain graph to FP32 rounding
        /// (|err| <= 2e-5 * max |out|) for every state count and query count.
        @Test(arguments: [1, 2, 4, 6])
        func residualMixMatchesThePlainGraph(states: Int) throws {
            for queries in [1, 3, 48] {
                let q = MLXRandom.normal([queries, 2304], key: MLXRandom.key(100)) * 0.05
                var inputs = (0 ..< states - 1).map { i in
                    (MLXRandom.normal([1, 23, 2304], key: MLXRandom.key(UInt64(110 + i))) * 3)
                        .asType(.bfloat16)
                }
                inputs.append(MLXRandom.normal([1, 23, 2304], key: MLXRandom.key(120)) * 2)
                let kernel = Inkling2ResidualMix.mix(queries: q, states: inputs)
                let plain = plainAttentionResidual(
                    queries: q, blocks: Array(inputs.dropLast()), partial: inputs.last!)
                #expect(kernel.shape == plain.shape)
                let error = abs(kernel - plain).max().item(Float.self)
                let scale = abs(plain).max().item(Float.self)
                #expect(error <= 2e-5 * scale, "S=\(states) nq=\(queries): \(error) vs \(scale)")
            }
        }

        /// A token's mix does not depend on the tokens mixed with it.
        @Test func residualMixRowsAreIndependent() throws {
            let q = MLXRandom.normal([3, 2304], key: MLXRandom.key(130)) * 0.05
            let inputs = (0 ..< 4).map { i in
                MLXRandom.normal([1, 70, 2304], key: MLXRandom.key(UInt64(140 + i)))
            }
            let all = Inkling2ResidualMix.mix(queries: q, states: inputs)
            for range in [0 ..< 1, 5 ..< 17, 40 ..< 70] {
                let part = Inkling2ResidualMix.mix(
                    queries: q, states: inputs.map { $0[0..., range] })
                #expect(Self.bitsEqual(all[0..., range], part), "\(range)")
            }
        }

        /// The prefill gate-input kernel equals the plain normalize + concat +
        /// cast to within one BF16 rounding step, and rows are independent.
        @Test func gateInputMatchesThePlainGraph() throws {
            let h = MLXRandom.normal([1, 70, 2304], key: MLXRandom.key(150)) * 3
            let m = MLXRandom.normal([1, 70, 2304], key: MLXRandom.key(151)) * 0.5
            let kernel = Inkling2GateInput.normalized(h, m, dtype: .bfloat16)
            let plain = concatenated(
                [
                    Inkling2Math.plainUnweightedRMSNorm(h, eps: 1e-6),
                    Inkling2Math.plainUnweightedRMSNorm(m, eps: 1e-6),
                ], axis: -1
            ).asType(.bfloat16)
            #expect(kernel.shape == plain.shape)
            let a = kernel.asType(.float32)
            let b = plain.asType(.float32)
            #expect((abs(a - b) .<= abs(b) * Float(1.0 / 128) + 1e-30).all().item(Bool.self))
            let part = Inkling2GateInput.normalized(
                h[0..., 9 ..< 30], m[0..., 9 ..< 30], dtype: .bfloat16)
            #expect(Self.bitsEqual(kernel[0..., 9 ..< 30], part))
        }

        /// The counting sort gives exactly gatherSort's permutation (stable by
        /// expert id), sorted ids and inverse.
        @Test(arguments: [8, 300, 2048, 3072])
        func countingSortMatchesGatherSort(tokens: Int) throws {
            let indices = MLXRandom.randInt(0 ..< 128, [tokens, 8], key: MLXRandom.key(160))
                .asType(.uint32)
            let x = MLXRandom.normal([tokens, 1, 1, 16], key: MLXRandom.key(161))
            let (order, ids, inverse) = Inkling2ExpertSort.sort(indices, experts: 128)
            let (rows, sortedIds, inverseOrder) = gatherSort(x: x, indices: indices)
            #expect(Self.bitsEqual(ids, sortedIds))
            #expect(Self.bitsEqual(inverse, inverseOrder.asType(.uint32)))
            #expect(Self.bitsEqual(x.flattened(start: 0, end: -3)[order.floorDivide(8)], rows))
        }

        /// The fused combine equals the unsort gather + FP32 weighted sum,
        /// plain and compiled, byte for byte (including signed zeros).
        @Test(arguments: [37, 2048])
        func expertCombineMatchesTheMLXExpression(tokens: Int) throws {
            let k = 8
            let hidden = 2304
            var raw = MLXRandom.normal([tokens * k, 1, hidden], key: MLXRandom.key(90))
            // Signed zeros and exact zeros in the rows and in the weights.
            raw = which(abs(raw) .< 0.01, MLXArray(Float(-0.0)), raw)
            let sorted = raw.asType(.bfloat16)
            var weights = MLXRandom.uniform(
                Float(-1) ..< Float(1), [tokens, k], key: MLXRandom.key(91))
            weights = which(abs(weights) .< 0.02, MLXArray(Float(0)), weights)
            let experts = MLXRandom.randInt(0 ..< 128, [tokens, k], key: MLXRandom.key(92))
            let order = argSort(experts.flattened())
            let inverse = argSort(order).asType(.uint32)
            eval(sorted, weights, inverse)
            #expect(
                Inkling2ExpertCombine.supported(
                    sortedOutputs: sorted, inverseOrder: inverse, weights: weights))
            let fused = Inkling2ExpertCombine.combine(
                sortedOutputs: sorted, inverseOrder: inverse, weights: weights)
            let rows = scatterUnsort(x: sorted, invOrder: inverse, shape: [tokens, k])
                .squeezed(axis: -2)
            let plain = (rows.asType(.float32) * weights.expandedDimensions(axis: -1))
                .sum(axis: -2).asType(.bfloat16)
            let compiled = Inkling2Fused.$override.withValue(true) {
                Inkling2Fused.expertCombine(rows, weights: weights, dtype: .bfloat16)
            }
            func bytes(_ a: MLXArray) -> Data { a.asData(access: .copy).data }
            #expect(fused.shape == plain.shape && bytes(fused) == bytes(plain))
            #expect(bytes(fused) == bytes(compiled))
        }
    }
}
