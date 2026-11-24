import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Inkling 2 row-exact kernels: a row's bits never depend on the other rows of
/// the step (none, or 1 to 63 others, with different companions and expert
/// overlaps), verify-window positions equal decode rows, and the results stay
/// close to FP32 references.
extension KernelTests {
    @Suite(.serialized)
    struct Inkling2KernelsTests {
        static let companionCounts = [0, 1, 3, 7, 15, 31, 63]

        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape
                && arrayEqual(a.asType(.float32), b.asType(.float32)).item(Bool.self)
        }

        static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
            let x = a.asType(.float32).reshaped(-1)
            let y = b.asType(.float32).reshaped(-1)
            return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
        }

        static func relativeError(_ a: MLXArray, _ reference: MLXArray) -> Float {
            let diff = abs(a.asType(.float32) - reference).max().item(Float.self)
            return diff / max(abs(reference).max().item(Float.self), 1e-12)
        }

        /// Batches holding pool row `target` among `count` other pool rows, at a
        /// varying position, with different companions for each count.
        static func compositions(pool: Int, target: Int, seed: UInt64) -> [[Int]] {
            var rng = Inkling2ComponentTests.Lcg(state: seed)
            return companionCounts.map { count in
                var others = (0 ..< pool).filter { $0 != target }
                for i in stride(from: others.count - 1, to: 0, by: -1) {
                    others.swapAt(i, Int(rng.next() % UInt64(i + 1)))
                }
                var rows = Array(others.prefix(count))
                rows.insert(target, at: Int(rng.next() % UInt64(count + 1)))
                return rows
            }
        }

        static func take(_ x: MLXArray, _ rows: [Int]) -> MLXArray {
            x[MLXArray(rows.map { Int32($0) })]
        }

        // MARK: Projections

        static let projectionShapes: [(n: Int, k: Int)] = [
            (3072, 2304), (2304, 3072), (1024, 2304), (384, 2304), (24, 2304), (64, 4608),
            (1024, 16), (130, 72), (200, 512),
        ]

        @Test(arguments: [DType.bfloat16, .float32])
        func projectionRowsDoNotDependOnTheirCompany(dtype: DType) throws {
            try Inkling2Kernels.$override.withValue(true) {
                for (index, shape) in Self.projectionShapes.enumerated() {
                    let w =
                        (MLXRandom.normal(
                            [shape.n, shape.k], key: MLXRandom.key(UInt64(100 + index))) * 0.05)
                        .asType(dtype)
                    let pool = MLXRandom.normal(
                        [64, shape.k], key: MLXRandom.key(UInt64(200 + index))
                    ).asType(dtype)
                    for target in [0, 37, 63] {
                        let alone = try #require(
                            Inkling2Kernels.rowProduct(
                                pool[target ..< target + 1], transposedWeight: w))
                        #expect(alone.dtype == dtype)
                        for rows in Self.compositions(
                            pool: 64, target: target, seed: UInt64(index * 64 + target))
                        {
                            let out = try #require(
                                Inkling2Kernels.rowProduct(
                                    Self.take(pool, rows), transposedWeight: w))
                            let at = rows.firstIndex(of: target)!
                            #expect(
                                Self.bitsEqual(out[at ..< at + 1], alone),
                                "n \(shape.n) k \(shape.k) rows \(rows.count)")
                        }
                    }
                }
            }
        }

        /// A `[B, L, K]` verify window gives each position the bits of a decode
        /// row `[1, 1, K]` with the same input.
        @Test func projectionWindowPositionsEqualDecodeRows() throws {
            try Inkling2Kernels.$override.withValue(true) {
                let w = (MLXRandom.normal([3072, 2304], key: MLXRandom.key(5)) * 0.05)
                    .asType(.bfloat16)
                let window = MLXRandom.normal([4, 16, 2304], key: MLXRandom.key(6))
                    .asType(.bfloat16)
                let all = try #require(Inkling2Kernels.rowProduct(window, transposedWeight: w))
                #expect(all.shape == [4, 16, 3072])
                for (b, j) in [(0, 0), (1, 7), (3, 15), (2, 9)] {
                    let step = try #require(
                        Inkling2Kernels.rowProduct(
                            window[b ..< b + 1, j ..< j + 1], transposedWeight: w))
                    #expect(Self.bitsEqual(all[b ..< b + 1, j ..< j + 1], step))
                }
            }
        }

        /// FP32 reference: per-row FP32 products (TF32-free).
        @Test func projectionsAgreeWithFP32() throws {
            try Inkling2Kernels.$override.withValue(true) {
                for (index, shape) in Self.projectionShapes.enumerated() {
                    let w =
                        (MLXRandom.normal(
                            [shape.n, shape.k], key: MLXRandom.key(UInt64(300 + index))) * 0.02)
                        .asType(.bfloat16)
                    let x = MLXRandom.normal(
                        [16, shape.k], key: MLXRandom.key(UInt64(400 + index))
                    )
                    .asType(.bfloat16)
                    let ours = try #require(Inkling2Kernels.rowProduct(x, transposedWeight: w))
                    let reference = Inkling2RowExact.fp32Rows(x, w)
                    let cos = Self.cosine(ours, reference)
                    let maxAbs = abs(ours.asType(.float32) - reference).max().item(Float.self)
                    print(
                        "INKLING2_KERNELS projection n \(shape.n) k \(shape.k) cosine \(cos) max abs \(maxAbs) rel \(Self.relativeError(ours, reference))"
                    )
                    #expect(cos >= 0.99999, "n \(shape.n) k \(shape.k) cosine \(cos)")
                }
            }
        }

        /// FP32 inputs and weights: the MMA products are full FP32 (not TF32):
        /// against a float64 host reference the error stays at FP32 rounding
        /// level (TF32 would be ~1e-3).
        @Test func fp32ProjectionsAreFullPrecision() throws {
            try Inkling2Kernels.$override.withValue(true) {
                let (n, k, rows) = (64, 2304, 4)
                let w = MLXRandom.normal([n, k], key: MLXRandom.key(61)) * 0.05
                let x = MLXRandom.normal([rows, k], key: MLXRandom.key(62))
                let ours = try #require(Inkling2Kernels.rowProduct(x, transposedWeight: w))
                let (ws, xs) = (w.asArray(Float.self), x.asArray(Float.self))
                var reference = [Float](repeating: 0, count: rows * n)
                for r in 0 ..< rows {
                    for j in 0 ..< n {
                        var acc = 0.0
                        for i in 0 ..< k { acc += Double(xs[r * k + i]) * Double(ws[j * k + i]) }
                        reference[r * n + j] = Float(acc)
                    }
                }
                let rel = Self.relativeError(ours, MLXArray(reference, [rows, n]))
                print("INKLING2_KERNELS fp32 projection max relative error vs float64 \(rel)")
                #expect(rel < 2e-6)
            }
        }

        // MARK: Experts

        static func uniformInts(_ shape: [Int], _ low: Float, _ high: Float, seed: UInt64)
            -> MLXArray
        {
            floor(MLXRandom.uniform(low: low, high: high, shape, key: MLXRandom.key(seed)))
                .asType(.uint32)
        }

        /// Random bank bytes of `format` (no NaN codes), or dense BF16 weights.
        static func randomBank(
            _ format: Inkling2ExpertFormat, experts: Int, outputs: Int, inputs: Int, seed: UInt64
        ) -> Inkling2ExpertWeights {
            switch format {
            case .dense:
                return Inkling2ExpertWeights(
                    format: .dense,
                    weight: (MLXRandom.normal(
                        [experts, outputs, inputs], key: MLXRandom.key(seed)) * 0.05)
                        .asType(.bfloat16))
            case .nvfp4:
                let nibbles = uniformInts([experts, outputs, inputs / 8, 8], 0, 16, seed: seed)
                let place = MLXArray((0 ..< 8).map { UInt32(1) << UInt32(4 * $0) })
                return Inkling2ExpertWeights(
                    format: .nvfp4, weight: (nibbles * place).sum(axis: -1).asType(.uint32),
                    scales: uniformInts(
                        [experts, outputs, inputs / 16], 40, 52, seed: seed + 1
                    ).asType(.uint8),
                    globalScale: MLXRandom.uniform(
                        low: 0.5, high: 3.0, [experts], key: MLXRandom.key(seed + 2)))
            case .mxfp8:
                let bytes =
                    uniformInts([experts, outputs, inputs / 4, 4], 0, 120, seed: seed)
                    + uniformInts([experts, outputs, inputs / 4, 4], 0, 2, seed: seed + 3)
                    * MLXArray(UInt32(128))
                let place = MLXArray((0 ..< 4).map { UInt32(1) << UInt32(8 * $0) })
                return Inkling2ExpertWeights(
                    format: .mxfp8, weight: (bytes * place).sum(axis: -1).asType(.uint32),
                    scales: uniformInts(
                        [experts, outputs, inputs / 32], 118, 130, seed: seed + 1
                    ).asType(.uint8))
            }
        }

        /// Distinct top-k experts per row from a shared preference plus row noise
        /// (real-ish routing: popular experts shared, rows differ), and FP32
        /// renormalized weights times 8.
        static func routing(rows: Int, experts: Int, topK: Int, seed: UInt64)
            -> (indices: MLXArray, weights: MLXArray)
        {
            let preference = MLXRandom.normal([1, experts], key: MLXRandom.key(seed))
            let noise = MLXRandom.normal([rows, experts], key: MLXRandom.key(seed + 1))
            let indices = argPartition(-(preference + noise), kth: topK - 1, axis: -1)[
                0..., ..<topK
            ].asType(.uint32)
            let raw = MLXRandom.uniform(
                low: 0.1, high: 1, [rows, topK], key: MLXRandom.key(seed + 2))
            return (indices, raw / raw.sum(axis: -1, keepDims: true) * 8)
        }

        static func experts(
            _ x: MLXArray, _ indices: MLXArray, _ weights: MLXArray, _ up: Inkling2ExpertWeights,
            _ down: Inkling2ExpertWeights
        ) throws -> MLXArray {
            try #require(
                Inkling2Kernels.$override.withValue(true) {
                    Inkling2Kernels.expertProduct(
                        x, indices: indices, weights: weights, gateUp: up, down: down,
                        gateCap: 4, upCap: 25)
                })
        }

        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8, .dense])
        func expertRowsDoNotDependOnTheirCompany(format: Inkling2ExpertFormat) throws {
            let (experts, hidden, intermediate, topK) = (128, 512, 256, 8)
            let up = Self.randomBank(
                format, experts: experts, outputs: 2 * intermediate, inputs: hidden, seed: 11)
            let down = Self.randomBank(
                format, experts: experts, outputs: hidden, inputs: intermediate, seed: 21)
            let pool = (MLXRandom.normal([64, hidden], key: MLXRandom.key(31)) * 0.5)
                .asType(.bfloat16)
            let (indices, weights) = Self.routing(rows: 64, experts: experts, topK: topK, seed: 41)
            for target in [0, 21, 63] {
                let alone = try Self.experts(
                    pool[target ..< target + 1], indices[target ..< target + 1],
                    weights[target ..< target + 1], up, down)
                #expect(alone.shape == [1, hidden] && alone.dtype == .bfloat16)
                for rows in Self.compositions(pool: 64, target: target, seed: UInt64(target)) {
                    let out = try Self.experts(
                        Self.take(pool, rows), Self.take(indices, rows), Self.take(weights, rows),
                        up, down)
                    let at = rows.firstIndex(of: target)!
                    #expect(
                        Self.bitsEqual(out[at ..< at + 1], alone),
                        "\(format) rows \(rows.count)")
                }
                // Companions on exactly the target's experts (other inputs, other
                // weights), and companions on none of them.
                let targetExperts = Set(indices[target].asArray(UInt32.self))
                let disjoint = (0 ..< UInt32(experts)).filter { !targetExperts.contains($0) }
                for overlap in [true, false] {
                    for count in [1, 15, 63] {
                        let ids = (0 ..< count).map { i -> [UInt32] in
                            overlap
                                ? indices[target].asArray(UInt32.self).shuffledDeterministic(i)
                                : (0 ..< topK).map { disjoint[(i * 7 + $0 * 13) % disjoint.count] }
                        }
                        let companions = MLXRandom.normal(
                            [count, hidden], key: MLXRandom.key(UInt64(500 + count))
                        ).asType(.bfloat16)
                        let x = concatenated([companions, pool[target ..< target + 1]], axis: 0)
                        let idx = concatenated(
                            [
                                MLXArray(ids.flatMap { $0 }, [count, topK]),
                                indices[target ..< target + 1],
                            ], axis: 0)
                        let w = concatenated(
                            [
                                Self.routing(
                                    rows: count, experts: experts, topK: topK,
                                    seed: UInt64(600 + count)
                                ).weights, weights[target ..< target + 1],
                            ], axis: 0)
                        let out = try Self.experts(x, idx, w, up, down)
                        #expect(
                            Self.bitsEqual(out[count ..< count + 1], alone),
                            "\(format) overlap \(overlap) companions \(count)")
                    }
                }
            }
        }

        /// The NVFP4 byte table gives the same weights as the bit decode, so
        /// the outputs are bit-identical.
        @Test func nvfp4TableMatchesTheBitDecode() throws {
            let (experts, hidden, intermediate, topK) = (64, 512, 256, 8)
            let up = Self.randomBank(
                .nvfp4, experts: experts, outputs: 2 * intermediate, inputs: hidden, seed: 13)
            let down = Self.randomBank(
                .nvfp4, experts: experts, outputs: hidden, inputs: intermediate, seed: 23)
            let x = (MLXRandom.normal([40, hidden], key: MLXRandom.key(33)) * 0.5)
                .asType(.bfloat16)
            let (indices, weights) = Self.routing(rows: 40, experts: experts, topK: topK, seed: 43)
            defer { Inkling2Kernels.nvfp4Table = true }
            var outputs: [MLXArray] = []
            for table in [true, false] {
                Inkling2Kernels.nvfp4Table = table
                for rows in [1, 7, 40] {
                    let out = try Self.experts(
                        x[0 ..< rows], indices[0 ..< rows], weights[0 ..< rows], up, down)
                    eval(out)
                    outputs.append(out)
                }
            }
            for i in 0 ..< 3 {
                #expect(Self.bitsEqual(outputs[i], outputs[i + 3]))
            }
        }

        /// BF16 MMA operands (every value exact in BF16) give the bits of FP32
        /// operands: same products, same FP32 accumulation.
        @Test func narrowOperandsMatchFloatOperands() throws {
            defer { Inkling2Kernels.narrowOperands = true }
            let w = (MLXRandom.normal([1056, 2304], key: MLXRandom.key(71)) * 0.05)
                .asType(.bfloat16)
            let x = MLXRandom.normal([37, 2304], key: MLXRandom.key(72)).asType(.bfloat16)
            let (experts, hidden, intermediate, topK) = (32, 512, 256, 8)
            let banks = [Inkling2ExpertFormat.nvfp4, .mxfp8].map { format in
                (
                    Self.randomBank(
                        format, experts: experts, outputs: 2 * intermediate, inputs: hidden,
                        seed: 81),
                    Self.randomBank(
                        format, experts: experts, outputs: hidden, inputs: intermediate, seed: 91)
                )
            }
            let xe = (MLXRandom.normal([29, hidden], key: MLXRandom.key(73)) * 0.5)
                .asType(.bfloat16)
            let (indices, weights) = Self.routing(rows: 29, experts: experts, topK: topK, seed: 74)
            var results: [[MLXArray]] = []
            for narrow in [true, false] {
                Inkling2Kernels.narrowOperands = narrow
                var outs = [
                    try #require(
                        Inkling2Kernels.$override.withValue(true) {
                            Inkling2Kernels.rowProduct(x, transposedWeight: w)
                        })
                ]
                for (up, down) in banks {
                    outs.append(try Self.experts(xe, indices, weights, up, down))
                }
                eval(outs)
                results.append(outs)
            }
            for (a, b) in zip(results[0], results[1]) {
                #expect(Self.bitsEqual(a, b))
            }
        }

        /// Flattened `[B * L]` verify-window rows equal decode rows: positions of
        /// a 16-wide window for 4 rows against each position run alone.
        @Test func expertWindowPositionsEqualDecodeRows() throws {
            let (experts, hidden, intermediate, topK) = (128, 512, 256, 8)
            let up = Self.randomBank(
                .nvfp4, experts: experts, outputs: 2 * intermediate, inputs: hidden, seed: 12)
            let down = Self.randomBank(
                .nvfp4, experts: experts, outputs: hidden, inputs: intermediate, seed: 22)
            let x = (MLXRandom.normal([64, hidden], key: MLXRandom.key(32)) * 0.5)
                .asType(.bfloat16)
            let (indices, weights) = Self.routing(rows: 64, experts: experts, topK: topK, seed: 42)
            let window = try Self.experts(x, indices, weights, up, down)
            for p in [0, 15, 16, 47, 63] {
                let step = try Self.experts(
                    x[p ..< p + 1], indices[p ..< p + 1], weights[p ..< p + 1], up, down)
                #expect(Self.bitsEqual(window[p ..< p + 1], step), "position \(p)")
            }
        }

        /// Host-decoded checkpoint banks (`Inkling2ComponentTests`) give an FP32
        /// reference with no intermediate rounding; the kernels and MLX's own
        /// `gather_qmm` path (same rounding points) must be equally close to it.
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8])
        func expertsAgreeWithTheDequantizedReference(format: Inkling2ExpertFormat) throws {
            let (experts, hidden, intermediate, topK, rows) = (16, 256, 128, 4, 12)
            func made(_ outputs: Int, _ inputs: Int, _ seed: UInt64)
                -> (tensors: [String: MLXArray], dense: [Float])
            {
                format == .nvfp4
                    ? Inkling2ComponentTests.nvfp4Bank(
                        experts: experts, rows: outputs, columns: inputs, seed: seed)
                    : Inkling2ComponentTests.mxfp8Bank(
                        experts: experts, rows: outputs, columns: inputs, seed: seed)
            }
            let upMade = made(2 * intermediate, hidden, 3)
            let downMade = made(hidden, intermediate, 4)
            let converted = try Inkling2Checkpoint.convertExperts(
                Inkling2ComponentTests.checkpoint("w13", upMade.tensors).merging(
                    Inkling2ComponentTests.checkpoint("w2", downMade.tensors)
                ) { a, _ in a },
                configuration: Inkling2ModelTests.config(), layer: "test")
            func bank(_ name: String) -> Inkling2ExpertWeights {
                Inkling2ExpertWeights(
                    format: format, weight: converted["\(name).weight"]!,
                    scales: converted["\(name).scales"],
                    globalScale: converted["\(name).global_scale"])
            }
            let (up, down) = (bank("w13"), bank("w2"))
            let x = (MLXRandom.normal([rows, hidden], key: MLXRandom.key(7)) * 0.5)
                .asType(.bfloat16)
            let (indices, weights) = Self.routing(rows: rows, experts: experts, topK: topK, seed: 8)
            let ours = try Self.experts(x, indices, weights, up, down)

            // MLX's path with the module's rounding points.
            let mode: QuantizationMode = format == .nvfp4 ? .nvfp4 : .mxfp8
            let (bits, group) = format == .nvfp4 ? (4, 16) : (8, 32)
            let gathered = gatherQuantizedMM(
                expandedDimensions(x, axes: [-2, -3]), up.weight, scales: up.scales!, biases: nil,
                rhsIndices: indices, transpose: true, groupSize: group, bits: bits, mode: mode,
                globalScale: up.globalScale, sortedIndices: false
            ).asType(.float32)
            let gate = gathered[.ellipsis, .stride(from: 0, by: 2)]
            let upHalf = gathered[.ellipsis, .stride(from: 1, by: 2)]
            let act = (tanh(gate / 4) * 4 * sigmoid(gate) * (tanh(upHalf / 25) * 25))
                .asType(.bfloat16)
            let projected = gatherQuantizedMM(
                act, down.weight, scales: down.scales!, biases: nil, rhsIndices: indices,
                transpose: true, groupSize: group, bits: bits, mode: mode,
                globalScale: down.globalScale, sortedIndices: false)
            let plain =
                (projected.squeezed(axis: -2).asType(.float32)
                * weights.expandedDimensions(axis: -1)).sum(axis: -2).asType(.bfloat16)

            // FP32 reference from the host-decoded weights, no intermediate rounding.
            let w13 = MLXArray(upMade.dense, [experts, 2 * intermediate, hidden])
            let w2 = MLXArray(downMade.dense, [experts, hidden, intermediate])
            let ids = indices.asArray(UInt32.self)
            let xs = x.asType(.float32)
            var referenceRows: [MLXArray] = []
            for r in 0 ..< rows {
                var acc = MLXArray.zeros([1, hidden])
                for s in 0 ..< topK {
                    let e = Int(ids[r * topK + s])
                    let h = Inkling2RowExact.fp32Rows(xs[r ..< r + 1], w13[e])
                    let g = h[0..., .stride(from: 0, by: 2)]
                    let u = h[0..., .stride(from: 1, by: 2)]
                    let a = tanh(g / 4) * 4 * sigmoid(g) * (tanh(u / 25) * 25)
                    acc = acc + Inkling2RowExact.fp32Rows(a, w2[e]) * weights[r, s]
                }
                referenceRows.append(acc)
            }
            let reference = concatenated(referenceRows, axis: 0)
            let (oursRel, plainRel) = (
                Self.relativeError(ours, reference), Self.relativeError(plain, reference)
            )
            let (oursCos, plainCos) = (Self.cosine(ours, reference), Self.cosine(plain, reference))
            let oursAbs = abs(ours.asType(.float32) - reference).max().item(Float.self)
            let plainAbs = abs(plain.asType(.float32) - reference).max().item(Float.self)
            print(
                "INKLING2_KERNELS experts \(format) vs FP32: kernel rel \(oursRel) cos \(oursCos) max abs \(oursAbs); gather_qmm rel \(plainRel) cos \(plainCos) max abs \(plainAbs)"
            )
            #expect(oursCos >= 0.9999)
            #expect(oursRel <= 1.5 * plainRel + 1e-4)
        }

        // MARK: Router

        /// The fused select picks the module router's top-k set with weights
        /// within FP32 rounding, and a row's outputs do not depend on its company.
        @Test func routerSelectMatchesTheModuleRouter() throws {
            let router = Inkling2Router(hidden: 256, experts: 128, topK: 8, routeScale: 8)
            SyntheticModel.randomize(router, seed: 17)
            let x = MLXRandom.normal([61, 256], key: MLXRandom.key(18)).asType(.bfloat16)
            let (plainIndices, plainWeights) = router(x)
            let (indices, weights, selection) = Inkling2Kernels.route(router, x)
            let plainSets = plainIndices.asArray(UInt32.self)
            let ours = indices.asArray(UInt32.self)
            let plainW = plainWeights.asArray(Float.self)
            let oursW = weights.asArray(Float.self)
            var worst: Float = 0
            for r in 0 ..< 61 {
                let a = Array(plainSets[(r * 8) ..< (r * 8 + 8)])
                let b = Array(ours[(r * 8) ..< (r * 8 + 8)])
                #expect(Set(a) == Set(b), "row \(r)")
                for (k, e) in b.enumerated() {
                    let j = a.firstIndex(of: e)!
                    worst = max(
                        worst, abs(oursW[r * 8 + k] - plainW[r * 8 + j]) / plainW[r * 8 + j])
                }
                // Rank order: selection scores descending.
                let sel = selection[r].asArray(Float.self)
                for k in 1 ..< 8 { #expect(sel[Int(b[k - 1])] >= sel[Int(b[k])]) }
            }
            print("INKLING2_KERNELS router weights max relative difference \(worst)")
            #expect(worst <= 1e-6)
            let (alone, aloneWeights, _) = Inkling2Kernels.route(router, x[40 ..< 41])
            #expect(Self.bitsEqual(alone, indices[40 ..< 41]))
            #expect(Self.bitsEqual(aloneWeights, weights[40 ..< 41]))
        }

        // MARK: Fused layer application

        /// Decode steps through the fused layer application against the module
        /// graph (kernels on for both), on an FP32 and a BF16 model: every visit's
        /// output and the logits agree to FP32-reference level, and the fused
        /// path actually runs.
        @Test(arguments: [DType.float32, .bfloat16])
        func fusedLayerMatchesTheModuleGraph(dtype: DType) throws {
            let model = try Inkling2ModelTests.makeModel(seed: 31)
            if dtype != .float32 {
                model.update(
                    parameters: ModuleParameters.unflattened(
                        model.parameters().flattened().map {
                            (
                                $0.0,
                                $0.1.dtype == .float32 && !Inkling2Checkpoint.keepsFP32($0.0)
                                    ? $0.1.asType(dtype) : $0.1
                            )
                        }))
            }
            let prompt = Inkling2ModelTests.tokens(12, count: 11)
            final class Visits: @unchecked Sendable { var outputs: [Int: MLXArray] = [:] }
            var results: [[MLXArray]] = []
            for fused in [false, true] {
                let state = Inkling2ModelTests.IncrementalState()
                state.begin(11)
                _ = model.forward(prompt, state: state)
                state.end(11)
                var logits: [MLXArray] = []
                var token = MLXArray([Int32(5)]).reshaped(1, 1)
                for _ in 0 ..< 4 {
                    state.begin(1)
                    let visits = Visits()
                    let out = Inkling2Kernels.$override.withValue(true) {
                        Inkling2Kernels.$fusedLayerOverride.withValue(fused) {
                            Inkling2RowExact.$active.withValue(true) {
                                model.forward(
                                    token, state: state,
                                    visitOutput: { visits.outputs[$0] = $1 }
                                ).logits
                            }
                        }
                    }
                    eval(out)
                    state.end(1)
                    logits.append(out)
                    logits += visits.outputs.keys.sorted().map { visits.outputs[$0]! }
                    token = argMax(out[0..., -1, 0...], axis: -1).reshaped(1, 1).asType(.int32)
                }
                results.append(logits)
            }
            var worst: Float = 1
            for (a, b) in zip(results[0], results[1]) {
                worst = min(worst, Self.cosine(a, b))
            }
            print("INKLING2_KERNELS fused layer \(dtype) worst cosine vs module graph \(worst)")
            #expect(worst >= 0.99999)
            let layer = model.model.layers[3]
            let x = MLXRandom.normal([1, 1, 64], key: MLXRandom.key(3)).asType(dtype)
            #expect(
                Inkling2Kernels.$override.withValue(true) {
                    Inkling2Kernels.fusedLayerActive
                } && layer.attention.heads > 0 && x.dim(2) == 64)
        }

        /// Loop messages for one row count in several B x L layouts (the launch
        /// is shared by row count): each result keeps its own batch and
        /// position axes, and every row's messages are the same bits.
        @Test func loopMessagesKeepTheirBatchAndPositionAxes() throws {
            let model = try Inkling2ModelTests.makeModel(seed: 31)
            let text = model.model
            let hidden = text.configuration.hiddenSize
            let loopLayers = text.configuration.loop.coreEnd - text.configuration.loop.coreStart
            let blocks = [
                MLXRandom.normal([8, hidden], key: MLXRandom.key(71)).asType(.bfloat16)
            ]
            let partial = MLXRandom.normal([8, hidden], key: MLXRandom.key(72)).asType(.bfloat16)
            eval(blocks + [partial])
            var reference: MLXArray?
            for (b, l) in [(8, 1), (1, 8), (2, 4), (8, 1)] {
                let messages = Inkling2Kernels.$override.withValue(true) {
                    Inkling2Kernels.$fusedLayerOverride.withValue(true) {
                        Inkling2RowExact.$active.withValue(true) {
                            Inkling2Kernels.loopMessages(
                                queries: text.messageInput.query,
                                blocks: blocks.map { $0.reshaped(b, l, hidden) },
                                partial: partial.reshaped(b, l, hidden))
                        }
                    }
                }
                guard let messages else {
                    Issue.record("loop messages not covered at \(b) x \(l)")
                    continue
                }
                eval(messages)
                #expect(Array(messages.shape.prefix(3)) == [loopLayers, b, l], "\(b) x \(l)")
                let rows = messages.reshaped(loopLayers, 8, -1)
                if let reference {
                    #expect(Self.bitsEqual(rows, reference), "\(b) x \(l) rows")
                } else {
                    reference = rows
                }
            }
        }

        /// One row: the fused mix + message-gate down launch gives the same h
        /// planes, down projections, residual and conv states as the mix kernel
        /// followed by the down product, for the attention-input mix (three
        /// gates) and the post-attention mix (convolved, one gate).
        @Test(arguments: [false, true])
        func mixDownEqualsMixThenDown(convolved: Bool) throws {
            let model = try Inkling2ModelTests.makeModel(seed: 37)
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map {
                        (
                            $0.0,
                            $0.1.dtype == .float32 && !Inkling2Checkpoint.keepsFP32($0.0)
                                ? $0.1.asType(.bfloat16) : $0.1
                        )
                    }))
            let c = model.model.configuration
            let layer = model.model.layers[c.loop.coreStart]
            let h = c.hiddenSize
            guard let controllers = layer.controllers else {
                Issue.record("loop layer without message gates")
                return
            }
            let x = MLXRandom.normal([1, h], key: MLXRandom.key(81)).asType(.bfloat16)
            let blocks = (0 ..< 2).map {
                MLXRandom.normal([1, h], key: MLXRandom.key(UInt64(82 + $0))).asType(.bfloat16)
            }
            let messages = MLXRandom.normal([1, 4, h], key: MLXRandom.key(85)).asType(.bfloat16)
            let gates =
                convolved
                ? [controllers.mlp] : [controllers.query, controllers.key, controllers.value]
            let norms = Array(
                repeating: (layer.inputNorm ?? layer.queryNorm!).weight, count: gates.count)
            let queries =
                convolved
                ? layer.mlpInput.query.asType(.float32)
                : layer.attentionInput!.query.asType(.float32)
            let state = MLXRandom.normal(
                [1, layer.postAttentionConvolution.kernel - 1, h], key: MLXRandom.key(86)
            ).asType(.bfloat16)
            let conv = (
                a: x, state: state, weight: layer.postAttentionConvolution.weight,
                hidden: blocks[0], boundary: false, tokens: 1
            )
            let msg0 = convolved ? 3 : 0
            let (fused, mixed, down) = Inkling2Kernels.$override.withValue(true) {
                Inkling2RowExact.$active.withValue(true) {
                    () -> ([MLXArray]?, [MLXArray], MLXArray?) in
                    let fused = Inkling2Kernels.rowMixDown(
                        partial: convolved ? nil : x, convolved: convolved ? conv : nil,
                        blocks: blocks, queries: queries, norms: norms, gates: gates,
                        messages: messages, msg0: msg0, eps: 1e-6, dtype: .bfloat16, rows: 1,
                        hidden: h)
                    let mixed = Inkling2Kernels.rowMix(
                        partial: convolved ? nil : x, convolved: convolved ? conv : nil,
                        blocks: blocks, queries: queries, norms: norms, gates: true,
                        messages: messages, msg0: msg0, eps: 1e-6, dtype: .bfloat16, rows: 1,
                        hidden: h)
                    let down = Inkling2Kernels.segmentedProduct(
                        (0 ..< gates.count).map { Inkling2Kernels.plane(mixed[0], $0) },
                        gates.enumerated().map {
                            Inkling2Kernels.Segment(input: $0.offset, weight: $0.element.down)
                        },
                        outType: Inkling2Kernels.referenceRounding ? .bfloat16 : .float32)
                    return (fused, mixed, down)
                }
            }
            guard let fused, let down else {
                Issue.record("fused mix + down not covered (convolved \(convolved))")
                return
            }
            eval(fused + mixed + [down])
            #expect(Self.bitsEqual(fused[0], mixed[1]), "h planes, convolved \(convolved)")
            #expect(Self.bitsEqual(fused[1], down), "down projections, convolved \(convolved)")
            if convolved {
                #expect(Self.bitsEqual(fused[2], mixed[2]), "residual")
                #expect(Self.bitsEqual(fused[3], mixed[3]), "conv state")
            }
        }

        /// Launches are cached process-wide and shared by every caller: kernels
        /// run from several threads at once (different row counts and B x L
        /// layouts, fresh launch keys) give the same shapes and bits as the
        /// same calls one at a time.
        @Test func kernelsGiveTheSameBitsFromSeveralThreads() throws {
            let model = try Inkling2ModelTests.makeModel(seed: 43)
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map {
                        (
                            $0.0,
                            $0.1.dtype == .float32 && !Inkling2Checkpoint.keepsFP32($0.0)
                                ? $0.1.asType(.bfloat16) : $0.1
                        )
                    }))
            let text = model.model
            let c = text.configuration
            let h = c.hiddenSize
            let layer = text.layers[c.loop.coreStart]
            let weight = layer.attention.queryProjection.weight
            let queries = layer.attentionInput!.query.asType(.float32)
            let norm = (layer.inputNorm ?? layer.queryNorm!).weight
            let layouts = [(3, 1), (1, 3), (5, 1), (1, 5), (2, 3), (3, 2), (9, 1), (1, 9)]
            let inputs = layouts.enumerated().map { i, bl in
                (0 ..< 2).map {
                    MLXRandom.normal(
                        [bl.0, bl.1, h], key: MLXRandom.key(UInt64(300 + 10 * i + $0))
                    ).asType(.bfloat16)
                }
            }
            eval(inputs.flatMap { $0 } + [weight, queries, norm])
            func work(_ i: Int) -> [MLXArray] {
                let (b, l) = layouts[i]
                let rows = b * l
                let partial = inputs[i][0]
                let block = inputs[i][1]
                return Inkling2Kernels.$override.withValue(true) {
                    Inkling2Kernels.$fusedLayerOverride.withValue(true) {
                        Inkling2RowExact.$active.withValue(true) {
                            let messages = Inkling2Kernels.loopMessages(
                                queries: text.messageInput.query, blocks: [block], partial: partial)
                            let product = Inkling2Kernels.segmentedProduct(
                                [partial.reshaped(rows, h)],
                                [Inkling2Kernels.Segment(input: 0, weight: weight)],
                                outType: .bfloat16)
                            let mixed = Inkling2Kernels.rowMix(
                                partial: partial.reshaped(rows, h), convolved: nil,
                                blocks: [block.reshaped(rows, h)], queries: queries, norms: [norm],
                                gates: false, messages: nil, msg0: 0, eps: 1e-6, dtype: .bfloat16,
                                rows: rows, hidden: h)
                            let out = [messages!, product!] + mixed
                            eval(out)
                            return out
                        }
                    }
                }
            }
            final class Results: @unchecked Sendable {
                let lock = NSLock()
                var values: [[MLXArray]?]
                init(_ n: Int) { values = Array(repeating: nil, count: n) }
            }
            let concurrent = Results(layouts.count)
            DispatchQueue.concurrentPerform(iterations: layouts.count) { i in
                let out = work(i)
                concurrent.lock.withLock { concurrent.values[i] = out }
            }
            let serial = layouts.indices.map { work($0) }
            for (i, (b, l)) in layouts.enumerated() {
                guard let got = concurrent.values[i] else {
                    Issue.record("layout \(b) x \(l) did not finish")
                    continue
                }
                #expect(Array(got[0].shape.prefix(3)) == [c.loop.coreEnd - c.loop.coreStart, b, l])
                for (a, e) in zip(got, serial[i]) {
                    #expect(a.shape == e.shape, "layout \(b) x \(l)")
                    #expect(Self.bitsEqual(a, e), "layout \(b) x \(l)")
                }
            }
        }

        /// The engine's per-slot state gather: 2 ... 8 rows of conv and SSM
        /// states (fresh rows, rows with their own history, row views of an
        /// earlier packed step) pack to the bits of concatenating them; one row,
        /// more than 8 rows or rows that are not whole 16-byte multiples fall
        /// back (nil).
        @Test(arguments: [1, 2, 3, 5, 8, 9])
        func rowStateGatherEqualsConcatenation(rows: Int) throws {
            let packedConv = MLXRandom.normal([9, 3, 128], key: MLXRandom.key(91)).asType(.bfloat16)
            let packedSSM = MLXRandom.normal([9, 3, 256], key: MLXRandom.key(92)).asType(.bfloat16)
            func state(_ r: Int, _ packed: MLXArray, seed: UInt64) -> MLXArray {
                switch r % 3 {
                case 0: MLXArray.zeros([1, 3, packed.dim(2)], dtype: .bfloat16)
                case 1:
                    MLXRandom.normal([1, 3, packed.dim(2)], key: MLXRandom.key(seed + UInt64(r)))
                        .asType(.bfloat16)
                default: packed[r ..< r + 1]
                }
            }
            let conv = (0 ..< rows).map { state($0, packedConv, seed: 100) }
            let ssm = (0 ..< rows).map { state($0, packedSSM, seed: 200) }
            eval(conv + ssm)
            let gathered = Inkling2Kernels.$override.withValue(true) {
                Inkling2Kernels.gatherRowStates(conv: conv, ssm: ssm)
            }
            if rows < 2 || rows > 8 {
                #expect(gathered == nil)
                return
            }
            guard let gathered else {
                Issue.record("gather not covered at \(rows) rows")
                return
            }
            eval(gathered)
            #expect(Self.bitsEqual(gathered[0], concatenated(conv, axis: 0)), "conv, \(rows) rows")
            #expect(Self.bitsEqual(gathered[1], concatenated(ssm, axis: 0)), "ssm, \(rows) rows")
            // Rows of an odd byte size are not covered.
            let odd = (0 ..< rows).map { _ in MLXArray.zeros([1, 3, 5], dtype: .bfloat16) }
            #expect(
                Inkling2Kernels.$override.withValue(true) {
                    Inkling2Kernels.gatherRowStates(conv: odd, ssm: ssm)
                } == nil)
        }

        // MARK: Module hooks

        /// The MoE module under `Inkling2RowExact.active`: the kernel path stays
        /// close to the plain path, and a lone row equals the row in a batch.
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8, .dense])
        func sparseMoEHookIsRowExact(format: Inkling2ExpertFormat) throws {
            let c = Inkling2ModelTests.config()
            let moe = Inkling2SparseMoE(c, format: format)
            SyntheticModel.randomize(moe, seed: 13)
            if format != .dense {
                let make =
                    format == .nvfp4
                    ? Inkling2ComponentTests.nvfp4Bank : Inkling2ComponentTests.mxfp8Bank
                let converted = try Inkling2Checkpoint.convertExperts(
                    Inkling2ComponentTests.checkpoint(
                        "w13",
                        make(c.experts, 2 * c.moeIntermediateSize, c.hiddenSize, 5).tensors
                    ).merging(
                        Inkling2ComponentTests.checkpoint(
                            "w2", make(c.experts, c.hiddenSize, c.moeIntermediateSize, 6).tensors)
                    ) { a, _ in a }, configuration: c, layer: "test")
                try moe.experts.update(
                    parameters: ModuleParameters.unflattened(Array(converted)), verify: [.all])
            }
            let x = (MLXRandom.normal([1, 20, c.hiddenSize], key: MLXRandom.key(14)) * 0.5)
                .asType(.bfloat16)
            if format == .dense {
                moe.experts.update(
                    parameters: ModuleParameters.unflattened(
                        moe.experts.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
                    ))
            }
            let (plain, ours, alone) = Inkling2RowExact.$active.withValue(true) {
                (
                    Inkling2Kernels.$override.withValue(false) { moe(x) },
                    Inkling2Kernels.$override.withValue(true) { moe(x) },
                    Inkling2Kernels.$override.withValue(true) { moe(x[0..., 7 ..< 8]) }
                )
            }
            #expect(
                Inkling2Kernels.$override.withValue(true) {
                    Inkling2Kernels.sparseMoE(moe, x) != nil
                })
            #expect(Self.bitsEqual(ours[0..., 7 ..< 8], alone))
            let cos = Self.cosine(ours, plain)
            print("INKLING2_KERNELS module \(format) kernel vs plain cosine \(cos)")
            #expect(cos >= 0.9999)
        }
    }
}

extension Array where Element == UInt32 {
    /// A fixed permutation per `seed` (rotation), so overlap rows list the same
    /// experts in another slot order.
    fileprivate func shuffledDeterministic(_ seed: Int) -> [UInt32] {
        guard !isEmpty else { return self }
        let shift = seed % count
        return Array(self[shift...] + self[..<shift])
    }
}
