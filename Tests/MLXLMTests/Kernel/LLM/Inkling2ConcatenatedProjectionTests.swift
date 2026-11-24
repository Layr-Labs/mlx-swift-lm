import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

/// Concatenated row-exact projections (`Inkling2Attention.prepareDecodeProjections`):
/// on this device, every output column of the concatenated product equals the
/// separate product bit for bit at Inkling 2 Mini's shapes, and `prepare()`
/// leaves parameters, prefill and plain decode unchanged.
extension KernelTests {
    @Suite(.serialized)
    struct Inkling2ConcatenatedProjectionTests {
        /// Output rows of the products that share an input (K = 2304):
        /// sliding `[wq; wr; sink; gate; wk; wv]`, full `[wq; wr; sink; gate]`,
        /// `[wk; k_sconv.weight_proj]` and `[wv; v_sconv.weight_proj]`.
        static let groups: [(String, [Int])] = [
            ("sliding shared", [3072, 384, 24, 3072, 1024, 1024]),
            ("query", [3072, 384, 24, 3072]),
            ("key with dynamic taps", [1024, 32]),
        ]

        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && arrayEqual(a.asType(.float32), b.asType(.float32)).item(Bool.self)
        }

        @Test(arguments: [1, 2, 3, 4, 5, 6, 8, 11, 16])
        func everyColumnEqualsItsSeparateProduct(rows: Int) throws {
            let k = 2304
            for (name, sizes) in Self.groups {
                let weights = sizes.enumerated().map { i, n in
                    (MLXRandom.normal([n, k], key: MLXRandom.key(UInt64(10 + i))) * 0.02)
                        .asType(.bfloat16)
                }
                let fused = concatenated(weights, axis: 0)
                let x = MLXRandom.normal([1, rows, k], key: MLXRandom.key(UInt64(rows)))
                    .asType(.bfloat16)
                eval(weights + [fused, x])
                let (together, separate) = Inkling2RowExact.$active.withValue(true) {
                    (
                        Inkling2RowExact.matmul(x, transposedWeight: fused),
                        weights.map { Inkling2RowExact.matmul(x, transposedWeight: $0) }
                    )
                }
                var start = 0
                for (i, part) in separate.enumerated() {
                    let column = together[.ellipsis, start ..< start + sizes[i]]
                    #expect(
                        Self.bitsEqual(column, part),
                        "\(name) part \(i) (\(sizes[i]) rows) at \(rows) input rows")
                    start += sizes[i]
                }
            }
        }

        /// Prefill-size GEMMs (NAX on M5): is every column of the concatenated
        /// product equal to its separate product? Reported, not asserted.
        @Test func prefillGemmColumnsProbe() throws {
            guard ProcessInfo.processInfo.environment["INKLING2_GEMM_PROBE"] == "1" else { return }
            var lines: [String] = []
            for (name, sizes) in Self.groups {
                let weights = sizes.enumerated().map { i, n in
                    (MLXRandom.normal([n, 2304], key: MLXRandom.key(UInt64(10 + i))) * 0.02)
                        .asType(.bfloat16)
                }
                let fused = concatenated(weights, axis: 0)
                for rows in [17, 64, 512, 2048] {
                    let x = MLXRandom.normal([1, rows, 2304], key: MLXRandom.key(UInt64(rows)))
                        .asType(.bfloat16)
                    let together = matmul(x, fused.T)
                    var start = 0
                    var equal: [Bool] = []
                    for (i, w) in weights.enumerated() {
                        equal.append(
                            Self.bitsEqual(
                                together[.ellipsis, start ..< start + sizes[i]], matmul(x, w.T)))
                        start += sizes[i]
                    }
                    lines.append("\(name) rows \(rows): \(equal)")
                }
            }
            print("INKLING2_GEMM_PROBE\n" + lines.joined(separator: "\n"))
        }

        /// Invariance class of prefill GEMMs, separate vs concatenated: is a
        /// row's result the same at M = 512 and M = 2,048, and with different
        /// rows beside it at M = 512? Reported, not asserted.
        @Test func prefillGemmInvarianceProbe() throws {
            guard ProcessInfo.processInfo.environment["INKLING2_GEMM_PROBE"] == "1" else { return }
            var lines: [String] = []
            let x = MLXRandom.normal([2048, 2304], key: MLXRandom.key(3)).asType(.bfloat16)
            let other = MLXRandom.normal([256, 2304], key: MLXRandom.key(4)).asType(.bfloat16)
            let mixed = concatenated([x[0 ..< 256], other], axis: 0)
            for (name, sizes) in Self.groups {
                let weights = sizes.enumerated().map { i, n in
                    (MLXRandom.normal([n, 2304], key: MLXRandom.key(UInt64(10 + i))) * 0.02)
                        .asType(.bfloat16)
                }
                let candidates =
                    weights.enumerated().map { ("part \($0.offset)", $0.element) }
                    + [("concatenated", concatenated(weights, axis: 0))]
                for (label, w) in candidates {
                    let big = matmul(x, w.T)
                    let small = matmul(x[0 ..< 512], w.T)
                    let company = matmul(mixed, w.T)
                    let mInvariant = Self.bitsEqual(big[0 ..< 512], small)
                    let companyInvariant = Self.bitsEqual(company[0 ..< 256], small[0 ..< 256])
                    lines.append(
                        "\(name) \(label): M 512 vs 2048 \(mInvariant), company \(companyInvariant)"
                    )
                }
            }
            print("INKLING2_GEMM_INVARIANCE\n" + lines.joined(separator: "\n"))
        }

        /// The message gates' `[forget; gain]` product (K = 64, FP32) equals the
        /// two separate products column for column, on the per-row gemv of
        /// row-exact steps and on the prefill kernel.
        @Test(arguments: [1, 3, 5, 16, 17, 64, 2048])
        func gateColumnsEqualTheirSeparateProducts(rows: Int) throws {
            let forget = MLXRandom.normal([2304, 64], key: MLXRandom.key(41)) * 0.1
            let gain = MLXRandom.normal([2304, 64], key: MLXRandom.key(42)) * 0.1
            let fused = concatenated([forget, gain], axis: 0)
            let g = MLXRandom.normal([1, rows, 64], key: MLXRandom.key(43))
            eval(forget, gain, fused, g)
            for rowExact in [true, false] {
                let (both, f, h) = Inkling2RowExact.$active.withValue(rowExact) {
                    (
                        Inkling2RowExact.fp32Rows(g, fused),
                        Inkling2RowExact.fp32Rows(g, forget),
                        Inkling2RowExact.fp32Rows(g, gain)
                    )
                }
                #expect(
                    Self.bitsEqual(both[.ellipsis, ..<2304], f), "forget, row-exact \(rowExact)")
                #expect(Self.bitsEqual(both[.ellipsis, 2304...], h), "gain, row-exact \(rowExact)")
            }
        }

        /// One loop layer's messages equal that layer's slice of all messages,
        /// in row-exact steps (plain graph) and in prefill (mix kernel).
        @Test func perLayerMessagesEqualTheSliceOfAllMessages() throws {
            let model = try Inkling2ModelTests.makeModel(seed: 7)
            let text = model.model
            let loopLayers = text.configuration.loop.coreEnd - text.configuration.loop.coreStart
            let hidden = text.configuration.hiddenSize
            let blocks = [
                MLXRandom.normal([2, 5, hidden], key: MLXRandom.key(61)).asType(.bfloat16)
            ]
            let partial = MLXRandom.normal([2, 5, hidden], key: MLXRandom.key(62)).asType(.bfloat16)
            for rowExact in [true, false] {
                for fused in [true, false] {
                    Inkling2RowExact.$active.withValue(rowExact) {
                        Inkling2Fused.$override.withValue(fused) {
                            let all = text.loopMessages(
                                previousBlocks: blocks, previousHidden: partial)
                            for i in 0 ..< loopLayers {
                                let one = text.messageInput(
                                    layer: i, blocks: blocks, partial: partial)
                                #expect(
                                    Self.bitsEqual(all[.ellipsis, i, 0..., 0...], one),
                                    "layer \(i) row-exact \(rowExact) fused \(fused)")
                            }
                        }
                    }
                }
            }
        }

        /// Row views of one buffer (the prepared layout) give the same products
        /// as standalone weights, in row-exact steps and in prefill-size GEMMs.
        @Test(arguments: [1, 3, 64, 512])
        func rowViewsMatchStandaloneWeights(rows: Int) throws {
            let k = 2304
            let sizes = [3072, 384, 24, 3072, 1024, 1024]
            let weights = sizes.enumerated().map { i, n in
                (MLXRandom.normal([n, k], key: MLXRandom.key(UInt64(30 + i))) * 0.02)
                    .asType(.bfloat16)
            }
            let fused = concatenated(weights, axis: 0)
            let x = MLXRandom.normal([1, rows, k], key: MLXRandom.key(7)).asType(.bfloat16)
            eval(weights + [fused, x])
            var start = 0
            for (i, w) in weights.enumerated() {
                let view = fused[start ..< start + sizes[i]]
                eval(view)
                start += sizes[i]
                #expect(Self.bitsEqual(matmul(x, view.T), matmul(x, w.T)), "part \(i) at \(rows)")
                let (a, b) = Inkling2RowExact.$active.withValue(true) {
                    (
                        Inkling2RowExact.matmul(x, transposedWeight: view),
                        Inkling2RowExact.matmul(x, transposedWeight: w)
                    )
                }
                #expect(Self.bitsEqual(a, b), "row-exact part \(i) at \(rows)")
            }
        }

        /// `prepare()` keeps parameter keys and values, the no-cache forward,
        /// and a later parameter update drops the buffers. (Half-precision
        /// weights, as served: FP32 projections stay separate.)
        @Test func prepareKeepsParametersAndOutputs() throws {
            let model = try Inkling2ModelTests.makeModel(seed: 5)
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map {
                        (
                            $0.0,
                            $0.1.dtype == .float32 && !Inkling2Checkpoint.keepsFP32($0.0)
                                ? $0.1.asType(.bfloat16) : $0.1
                        )
                    }))
            eval(model)
            let before = model.parameters().flattened()
            let tokens = Inkling2ModelTests.tokens(3, count: 13)
            // Concatenation off: the separate projections, now row views, keep their bits.
            let logitsBefore = Inkling2Fused.$concatenationOverride.withValue(false) {
                model.forwardNoCache(tokens).logits
            }
            eval(logitsBefore)
            try model.prepare()
            eval(model)
            let after = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            #expect(Set(before.map(\.0)) == Set(after.keys))
            for (key, value) in before {
                #expect(Self.bitsEqual(value, after[key]!), "\(key)")
            }
            let logitsAfter = Inkling2Fused.$concatenationOverride.withValue(false) {
                model.forwardNoCache(tokens).logits
            }
            #expect(Self.bitsEqual(logitsAfter, logitsBefore))
            // Concatenation on: prefill takes the concatenated products (same values to rounding).
            let concatenatedLogits = model.forwardNoCache(tokens).logits
            #expect(abs(concatenatedLogits - logitsBefore).max().item(Float.self) < 0.05)
            let attention = model.model.layers[0].attention
            #expect(attention.hasDecodeProjections)
            let gates = model.model.layers.compactMap { $0.controllers }.flatMap(\.gates)
            #expect(!gates.isEmpty && gates.allSatisfy(\.hasFusedGates))
            SyntheticModel.randomize(model, seed: 6)
            #expect(!attention.hasDecodeProjections)
            #expect(gates.allSatisfy { !$0.hasFusedGates })
        }
    }
}
