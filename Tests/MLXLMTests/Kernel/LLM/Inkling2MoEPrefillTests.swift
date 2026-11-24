import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// The prefill expert kernels (M5 neural accelerators): accuracy against an
/// FP32 reference next to the sorted `gather_qmm` route, outputs independent of
/// chunking and company, and the module taking them only outside row-exact
/// steps. Skipped without the neural accelerators.
extension KernelTests {
    @Suite(.enabled(if: Inkling2PagedAttention.neuralAcceleratorsAvailable))
    struct Inkling2MoEPrefillTests {
        typealias Components = Inkling2ComponentTests

        static let gateCap: Float = 4
        static let upCap: Float = 25

        /// A bank of `format` with `rows` outputs and `columns` inputs, and its
        /// host-decoded FP32 weights `[experts, rows, columns]`.
        static func bank(
            _ format: Inkling2ExpertFormat, experts: Int, rows: Int, columns: Int, seed: UInt64
        ) throws -> (Inkling2ExpertBank, MLXArray) {
            if format == .dense {
                let w =
                    (MLXRandom.normal(
                        [experts, rows, columns], key: MLXRandom.key(seed)) * 0.05).asType(
                        .bfloat16)
                let bank = Inkling2DenseExpertBank(
                    experts: experts, inputDimensions: columns, outputDimensions: rows)
                try bank.update(
                    parameters: ModuleParameters.unflattened([("weight", w)]), verify: [.all])
                return (bank, w.asType(.float32))
            }
            let made =
                format == .nvfp4
                ? Components.nvfp4Bank(experts: experts, rows: rows, columns: columns, seed: seed)
                : Components.mxfp8Bank(experts: experts, rows: rows, columns: columns, seed: seed)
            let converted = try Inkling2Checkpoint.convertExperts(
                Components.checkpoint("w2", made.tensors),
                configuration: Inkling2ModelTests.config(), layer: "test")
            let bank = Inkling2QuantizedExpertBank(
                format: format, experts: experts, inputDimensions: columns, outputDimensions: rows)
            let parameters = converted.compactMap { key, value -> (String, MLXArray)? in
                key.hasPrefix("w2.") ? (String(key.dropFirst(3)), value) : nil
            }
            try bank.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
            return (bank, MLXArray(made.dense, [experts, rows, columns]))
        }

        /// A bank of `format` filled from random bytes on the GPU (no host copy).
        static func randomBank(
            _ format: Inkling2ExpertFormat, experts: Int, rows: Int, columns: Int, seed: UInt64
        ) throws -> Inkling2ExpertBank {
            let key = MLXRandom.key(seed)
            let bank = Inkling2QuantizedExpertBank(
                format: format, experts: experts, inputDimensions: columns, outputDimensions: rows)
            var parameters: [(String, MLXArray)]
            if format == .nvfp4 {
                let words = MLXRandom.randInt(
                    low: 0, high: 256, [experts, rows, columns / 2], key: key
                )
                .asType(.uint8).view(dtype: .uint32)
                // Normal positive E4M3 block scales between 2^-3 and 2^2.
                let exponent = MLXRandom.randInt(
                    low: 4, high: 10, [experts, rows, columns / 16], key: MLXRandom.key(seed + 1))
                let mantissa = MLXRandom.randInt(
                    low: 0, high: 8, [experts, rows, columns / 16], key: MLXRandom.key(seed + 2))
                parameters = [
                    ("weight", words), ("scales", (exponent * 8 + mantissa).asType(.uint8)),
                    (
                        "global_scale",
                        MLXRandom.uniform(
                            low: 0.5, high: 3.5, [experts], key: MLXRandom.key(seed + 3))
                    ),
                ]
            } else {
                // E4M3 bytes with exponents 0-14 (no NaN codes), E8M0 scales 2^-16 to 2^-12.
                let shape = [experts, rows, columns]
                let sign = MLXRandom.randInt(low: 0, high: 2, shape, key: key)
                let exponent = MLXRandom.randInt(
                    low: 0, high: 15, shape, key: MLXRandom.key(seed + 1))
                let mantissa = MLXRandom.randInt(
                    low: 0, high: 8, shape, key: MLXRandom.key(seed + 2))
                let bytes = (sign * 128 + exponent * 8 + mantissa).asType(.uint8)
                parameters = [
                    ("weight", bytes.view(dtype: .uint32)),
                    (
                        "scales",
                        MLXRandom.randInt(
                            low: 111, high: 116, [experts, rows, columns / 32],
                            key: MLXRandom.key(seed + 3)
                        ).asType(.uint8)
                    ),
                ]
            }
            try bank.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
            eval(bank)
            return bank
        }

        /// `tokens` rows of `topK` distinct experts each, drawn with uneven
        /// expert popularity (some experts get many rows, some none), and FP32
        /// weights that sum to 8 per row.
        static func routing(tokens: Int, experts: Int, topK: Int, seed: UInt64)
            -> (MLXArray, MLXArray)
        {
            var rng = Components.Lcg(state: seed)
            let popularity = (0 ..< experts).map { e -> Double in
                let u = Double(rng.next() % 10_000) / 10_000
                return 0.02 + pow(u, 4) * Double(e % 7 + 1)
            }
            var ids: [UInt32] = []
            var weights: [Float] = []
            for _ in 0 ..< tokens {
                var picked: [Int] = []
                while picked.count < topK {
                    var target =
                        Double(rng.next() % 1_000_000) / 1_000_000
                        * popularity.reduce(0, +)
                    var e = 0
                    while e < experts - 1, target >= popularity[e] {
                        target -= popularity[e]
                        e += 1
                    }
                    if !picked.contains(e) { picked.append(e) }
                }
                let raw = picked.map { _ in 0.1 + rng.unit() }
                let total = raw.reduce(0, +)
                ids += picked.map(UInt32.init)
                weights += raw.map { $0 / total * 8 }
            }
            return (MLXArray(ids, [tokens, topK]), MLXArray(weights, [tokens, topK]))
        }

        /// The sorted route: rows sorted by expert, `gather_qmm` on the
        /// sorted indices, the activation, down, unsort and the FP32 combine.
        static func sortedRoute(
            _ x: MLXArray, indices: MLXArray, weights: MLXArray, gateUp: Inkling2ExpertBank,
            down: Inkling2ExpertBank
        ) -> MLXArray {
            let rows = expandedDimensions(x, axes: [-2, -3])
            let (sortedRows, sortedIndices, inverse) = gatherSort(x: rows, indices: indices)
            let gu = gateUp(sortedRows, sortedIndices, sorted: true).asType(.float32)
            let gate = gu[.ellipsis, .stride(from: 0, by: 2)]
            let up = gu[.ellipsis, .stride(from: 1, by: 2)]
            let activated =
                (Inkling2Math.softcap(gate, gateCap) * sigmoid(gate)
                * Inkling2Math.softcap(up, upCap)).asType(x.dtype)
            let projected = scatterUnsort(
                x: down(activated, sortedIndices, sorted: true), invOrder: inverse,
                shape: indices.shape)
            let perExpert = projected.squeezed(axis: -2).asType(.float32)
            return (perExpert * weights.expandedDimensions(axis: -1)).sum(axis: -2).asType(
                x.dtype)
        }

        /// FP32 from the decoded weights without intermediate rounding (products
        /// on the CPU, so no TF32).
        static func reference(
            _ x: MLXArray, indices: MLXArray, weights: MLXArray, gateUp: MLXArray, down: MLXArray
        ) -> MLXArray {
            let xs = x.asType(.float32)
            let ids = indices.asArray(UInt32.self).map(Int.init)
            let topK = indices.dim(1)
            var rows: [MLXArray] = []
            for t in 0 ..< x.dim(0) {
                var acc = MLXArray.zeros([x.dim(1)], dtype: .float32)
                for slot in 0 ..< topK {
                    let e = ids[t * topK + slot]
                    let gu = matmul(gateUp[e], xs[t], stream: .cpu)
                    let gate = gu[.stride(from: 0, by: 2)]
                    let up = gu[.stride(from: 1, by: 2)]
                    let act =
                        tanh(gate / gateCap) * gateCap * sigmoid(gate) * tanh(up / upCap)
                        * upCap
                    acc = acc + matmul(down[e], act, stream: .cpu) * weights[t, slot]
                }
                rows.append(acc)
            }
            return stacked(rows)
        }

        static func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
            let x = a.asType(.float32).asArray(Float.self).map(Double.init)
            let y = b.asType(.float32).asArray(Float.self).map(Double.init)
            let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
            let nx = x.reduce(0) { $0 + $1 * $1 }.squareRoot()
            let ny = y.reduce(0) { $0 + $1 * $1 }.squareRoot()
            return dot / (nx * ny)
        }

        static func run(
            _ x: MLXArray, _ indices: MLXArray, _ weights: MLXArray, _ gateUp: Inkling2ExpertBank,
            _ down: Inkling2ExpertBank
        ) -> MLXArray {
            let (g, d) = (Inkling2MoEPrefill.Bank(gateUp)!, Inkling2MoEPrefill.Bank(down)!)
            let tile = Inkling2MoEPrefill.defaultTile
            precondition(
                Inkling2MoEPrefill.covers(
                    rows: x.dim(0), hidden: x.dim(1), dtype: x.dtype, topK: indices.dim(1),
                    gateUp: g, down: d, tile: tile))
            let y = Inkling2MoEPrefill.experts(
                x, indices: indices, weights: weights, gateUp: g, down: d, gateCap: gateCap,
                upCap: upCap, tile: tile)
            eval(y)
            return y
        }

        // MARK: Accuracy

        /// Cosine and max |error| / max |reference| against the FP32 reference,
        /// for the kernels and for the sorted `gather_qmm` route. The kernels
        /// may be more accurate (NVFP4 global scale in FP32), never notably less.
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8, .dense])
        func matchesTheFP32ReferenceLikeTheSortedRoute(format: Inkling2ExpertFormat) throws {
            let (experts, hidden, intermediate, tokens, topK) = (16, 256, 128, 96, 8)
            let (gateUp, gateUpDense) = try Self.bank(
                format, experts: experts, rows: 2 * intermediate, columns: hidden, seed: 11)
            let (down, downDense) = try Self.bank(
                format, experts: experts, rows: hidden, columns: intermediate, seed: 12)
            let (indices, weights) = Self.routing(
                tokens: tokens, experts: experts, topK: topK, seed: 3)
            let x = (MLXRandom.normal([tokens, hidden], key: MLXRandom.key(4))).asType(.bfloat16)

            let kernel = Self.run(x, indices, weights, gateUp, down)
            let route = Self.sortedRoute(
                x, indices: indices, weights: weights, gateUp: gateUp, down: down)
            let reference = Self.reference(
                x, indices: indices, weights: weights, gateUp: gateUpDense, down: downDense)
            let scale = abs(reference).max().item(Float.self)
            let kernelError =
                abs(kernel.asType(.float32) - reference).max().item(Float.self) / scale
            let routeError = abs(route.asType(.float32) - reference).max().item(Float.self) / scale
            let (kernelCosine, routeCosine) = (
                Self.cosine(kernel, reference), Self.cosine(route, reference)
            )
            print(
                "inkling2-moe-prefill accuracy \(format): kernel cosine \(kernelCosine) max rel \(kernelError); sorted route cosine \(routeCosine) max rel \(routeError)"
            )
            #expect(kernelCosine >= 0.9999)
            #expect(kernelCosine >= routeCosine - 2e-6)
            #expect(kernelError <= routeError * 1.25 + 1e-4)
        }

        // MARK: Invariance

        /// Every token's output is bit-identical however the tokens are chunked,
        /// ordered or accompanied, and from run to run.
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8])
        func outputsDoNotDependOnChunkingOrCompany(format: Inkling2ExpertFormat) throws {
            let (experts, hidden, intermediate, tokens, topK) = (32, 512, 256, 1024, 8)
            let gateUp = try Self.randomBank(
                format, experts: experts, rows: 2 * intermediate, columns: hidden, seed: 21)
            let down = try Self.randomBank(
                format, experts: experts, rows: hidden, columns: intermediate, seed: 22)
            try Self.checkInvariance(
                gateUp, down, hidden: hidden, tokens: tokens, topK: topK, experts: experts,
                chunks: [512, 256, 100, 7, 1])
        }

        /// The real expert shapes (128 experts, hidden 2,304, intermediate 1,152)
        /// at 2,048 and 512 tokens.
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8])
        func realShapesAreInvariantAt2048And512(format: Inkling2ExpertFormat) throws {
            let (experts, hidden, intermediate, topK) = (128, 2304, 1152, 8)
            let gateUp = try Self.randomBank(
                format, experts: experts, rows: 2 * intermediate, columns: hidden, seed: 31)
            let down = try Self.randomBank(
                format, experts: experts, rows: hidden, columns: intermediate, seed: 32)
            try Self.checkInvariance(
                gateUp, down, hidden: hidden, tokens: 2048, topK: topK, experts: experts,
                chunks: [512])
        }

        static func checkInvariance(
            _ gateUp: Inkling2ExpertBank, _ down: Inkling2ExpertBank, hidden: Int, tokens: Int,
            topK: Int, experts: Int, chunks: [Int]
        ) throws {
            let (indices, weights) = routing(
                tokens: tokens, experts: experts, topK: topK, seed: 9)
            let x = MLXRandom.normal([tokens, hidden], key: MLXRandom.key(5)).asType(.bfloat16)
            let full = run(x, indices, weights, gateUp, down)
            #expect(arrayEqual(run(x, indices, weights, gateUp, down), full).item(Bool.self))

            for chunk in chunks {
                var parts: [MLXArray] = []
                var start = 0
                while start < tokens {
                    let end = min(tokens, start + chunk)
                    parts.append(
                        run(
                            x[start ..< end], indices[start ..< end], weights[start ..< end],
                            gateUp, down))
                    start = end
                }
                let joined = concatenated(parts, axis: 0)
                #expect(arrayEqual(joined, full).item(Bool.self), "chunk \(chunk)")
            }

            var rng = Components.Lcg(state: 77)
            var permutation = Array(0 ..< tokens)
            for i in stride(from: tokens - 1, to: 0, by: -1) {
                permutation.swapAt(i, Int(rng.next() % UInt64(i + 1)))
            }
            let p = MLXArray(permutation.map(Int32.init))
            let shuffled = run(x[p], indices[p], weights[p], gateUp, down)
            #expect(arrayEqual(shuffled, full[p]).item(Bool.self), "shuffled")

            // The first 100 tokens among 300 others, in a shuffled order.
            let company = Array(permutation.filter { $0 >= 100 }.prefix(300)) + Array(0 ..< 100)
            var mixed = company
            for i in stride(from: mixed.count - 1, to: 0, by: -1) {
                mixed.swapAt(i, Int(rng.next() % UInt64(i + 1)))
            }
            let m = MLXArray(mixed.map(Int32.init))
            let accompanied = run(x[m], indices[m], weights[m], gateUp, down)
            #expect(arrayEqual(accompanied, full[m]).item(Bool.self), "company")
        }

        // MARK: Module

        static func module() throws -> Inkling2SparseMoE {
            var base = Inkling2ModelTests.base
            var text = base["text_config"] as! [String: Any]
            text["hidden_size"] = 128
            text["intermediate_size"] = 64
            text["moe_intermediate_size"] = 64
            text["n_routed_experts"] = 16
            text["num_experts_per_tok"] = 8
            base["text_config"] = text
            let c = try SyntheticModel.configuration(Inkling2Configuration.self, base).text
            let moe = Inkling2SparseMoE(c, format: .dense)
            SyntheticModel.randomize(moe, seed: 8)
            let banks = moe.experts.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
            try moe.experts.update(parameters: ModuleParameters.unflattened(banks), verify: [.all])
            eval(moe)
            return moe
        }

        /// Prefill calls take the kernels (same result as calling them on the
        /// router's output); row-exact steps and the off switch keep the old path.
        @Test func moduleTakesTheKernelsOnlyOutsideRowExactSteps() throws {
            let moe = try Self.module()
            let x = MLXRandom.normal([1, 40, 128], key: MLXRandom.key(6)).asType(.bfloat16)
            let flat = x.reshaped(40, 128)
            let (indices, weights) = moe.router(flat)
            let direct = Self.run(flat, indices, weights, moe.experts.gateUp, moe.experts.down)

            let on = Inkling2MoEPrefill.$override.withValue(true) { moe(x) }
            let off = Inkling2MoEPrefill.$override.withValue(false) { moe(x) }
            let rowExactOn = Inkling2RowExact.$active.withValue(true) {
                Inkling2MoEPrefill.$override.withValue(true) { moe(x) }
            }
            let rowExactOff = Inkling2RowExact.$active.withValue(true) {
                Inkling2MoEPrefill.$override.withValue(false) { moe(x) }
            }
            #expect(arrayEqual(on.reshaped(40, 128), direct).item(Bool.self))
            #expect(arrayEqual(rowExactOn, rowExactOff).item(Bool.self))
            // The sorted route differs only by rounding.
            let scale = abs(off.asType(.float32)).max().item(Float.self)
            #expect(
                abs(on.asType(.float32) - off.asType(.float32)).max().item(Float.self)
                    <= 0.02 * scale)
        }
        // MARK: Bench

        /// `INKLING2_MOE_BENCH=1`: ms per call at the real shapes, 2,048 tokens,
        /// kernels (total and sectioned) vs the sorted route. Routing from
        /// `INKLING2_MOE_ROUTER` (an oracle `router.safetensors`), else synthetic.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["INKLING2_MOE_BENCH"] == "1"))
        func benchRealShapes() throws {
            let env = ProcessInfo.processInfo.environment
            let tokens = Int(env["INKLING2_MOE_TOKENS"] ?? "2048")!
            let (experts, hidden, intermediate, topK) = (128, 2304, 1152, 8)
            let router =
                env["INKLING2_MOE_ROUTER"].map {
                    try? loadArrays(url: URL(fileURLWithPath: $0))
                } ?? nil
            for (format, app) in [(Inkling2ExpertFormat.nvfp4, 10), (.mxfp8, 21)] {
                let gateUp = try Self.randomBank(
                    format, experts: experts, rows: 2 * intermediate, columns: hidden, seed: 41)
                let down = try Self.randomBank(
                    format, experts: experts, rows: hidden, columns: intermediate, seed: 42)
                var (indices, weights) = Self.routing(
                    tokens: tokens, experts: experts, topK: topK, seed: 9)
                if let r = router, let i = r["router_topk_indices"],
                    let w = r["router_topk_weights"]
                {
                    indices = i[app, 0 ..< tokens].asType(.uint32)
                    weights = w[app, 0 ..< tokens].asType(.float32)
                }
                let x = MLXRandom.normal([tokens, hidden], key: MLXRandom.key(5)).asType(.bfloat16)
                eval(indices, weights, x)
                func time(_ body: () -> MLXArray) -> Double {
                    for _ in 0 ..< 3 { eval(body()) }
                    let start = Date()
                    let n = 20
                    for _ in 0 ..< n { eval(body()) }
                    return Date().timeIntervalSince(start) / Double(n) * 1000
                }
                let kernel = time { Self.run(x, indices, weights, gateUp, down) }
                let route = time {
                    Self.sortedRoute(
                        x, indices: indices, weights: weights, gateUp: gateUp, down: down)
                }
                let clock = Clock()
                for _ in 0 ..< 3 { _ = Self.run(x, indices, weights, gateUp, down) }
                clock.reset()
                let hook: @Sendable (String, [MLXArray]) -> Void = { clock.mark($0, $1) }
                for _ in 0 ..< 10 {
                    clock.last = Date()
                    let y = Inkling2Diagnostics.$section.withValue(hook) {
                        Self.run(x, indices, weights, gateUp, down)
                    }
                    clock.mark("combine", [y])
                }
                let parts = clock.totals.mapValues { ($0 / 10 * 1000 * 1000).rounded() / 1000 }
                let flops = 2.0 * Double(tokens * topK) * Double(3 * hidden * intermediate)
                print(
                    "inkling2-moe-prefill bench \(format) tokens \(tokens) routing \(router == nil ? "synthetic" : "app \(app)"): kernels \(kernel) ms (\(flops / kernel / 1e9) TFLOPS), sorted route \(route) ms; sections \(parts)"
                )
            }
        }

        final class Clock: @unchecked Sendable {
            var last = Date()
            var totals: [String: Double] = [:]
            func reset() { totals = [:] }
            func mark(_ name: String, _ arrays: [MLXArray]) {
                eval(arrays)
                let t = Date()
                totals[name, default: 0] += t.timeIntervalSince(last)
                last = t
            }
        }
    }
}
