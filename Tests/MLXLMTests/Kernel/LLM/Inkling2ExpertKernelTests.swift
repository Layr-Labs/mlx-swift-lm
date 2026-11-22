import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Inkling 2 one-row expert GEMVs and the fused router: the same bits as the
/// tiles and the two-launch router, a row alone equal to the same row in any
/// company, verify-window positions equal to one-token decode steps, and
/// expert outputs close to a host reference.
extension KernelTests {
    @Suite(.serialized)
    struct Inkling2ExpertKernelTests {
        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape
                && arrayEqual(a.asType(.float32), b.asType(.float32)).item(Bool.self)
        }

        static func uniformInts(_ shape: [Int], _ low: Float, _ high: Float, seed: UInt64)
            -> MLXArray
        {
            floor(MLXRandom.uniform(low: low, high: high, shape, key: MLXRandom.key(seed)))
                .asType(.uint32)
        }

        /// Random 32-bit words: every bit pattern (two 16-bit draws).
        static func randomWords(_ shape: [Int], seed: UInt64) -> MLXArray {
            let lo = MLXRandom.randInt(Int32(0) ..< 65536, shape, key: MLXRandom.key(seed))
            let hi = MLXRandom.randInt(Int32(0) ..< 65536, shape, key: MLXRandom.key(seed &+ 977))
            return lo.asType(.uint32) + hi.asType(.uint32) * MLXArray(UInt32(65536))
        }

        /// Random bank bytes of `format` (every E2M1 code; E4M3 values without
        /// NaN codes), or dense BF16 weights.
        static func randomBank(
            _ format: Inkling2ExpertFormat, experts: Int, outputs: Int, inputs: Int, seed: UInt64
        ) -> Inkling2ExpertWeights {
            let bank: Inkling2ExpertWeights
            switch format {
            case .dense:
                bank = Inkling2ExpertWeights(
                    format: .dense,
                    weight: (MLXRandom.normal(
                        [experts, outputs, inputs], key: MLXRandom.key(seed)) * 0.05)
                        .asType(.bfloat16))
            case .nvfp4:
                bank = Inkling2ExpertWeights(
                    format: .nvfp4,
                    weight: randomWords([experts, outputs, inputs / 8], seed: seed),
                    scales: uniformInts(
                        [experts, outputs, inputs / 16], 40, 52, seed: seed + 1
                    ).asType(.uint8),
                    globalScale: MLXRandom.uniform(
                        low: 0.5, high: 3.0, [experts], key: MLXRandom.key(seed + 2)))
            case .mxfp8:
                // Exponent field <= 7 (bit 6 clear): no NaN codes.
                bank = Inkling2ExpertWeights(
                    format: .mxfp8,
                    weight: randomWords([experts, outputs, inputs / 4], seed: seed)
                        & MLXArray(UInt32(0xBFBF_BFBF)),
                    scales: uniformInts(
                        [experts, outputs, inputs / 32], 118, 130, seed: seed + 1
                    ).asType(.uint8))
            }
            eval([bank.weight] + [bank.scales, bank.globalScale].compactMap { $0 })
            return bank
        }

        /// Distinct top-k experts per row around a shared preference.
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

        /// The per-pair expert outputs `[R * topK, H]` (gate/up, activation, down).
        static func pairOutputs(
            _ x: MLXArray, _ indices: MLXArray, _ up: Inkling2ExpertWeights,
            _ down: Inkling2ExpertWeights
        ) throws -> MLXArray {
            let out = try #require(
                Inkling2Kernels.$override.withValue(true) {
                    Inkling2Kernels.expertsUpDown(
                        x, indices: indices, gateUp: up, down: down, gateCap: 4, upCap: 25)
                })
            eval(out)
            return out
        }

        static func combined(
            _ x: MLXArray, _ indices: MLXArray, _ weights: MLXArray, _ up: Inkling2ExpertWeights,
            _ down: Inkling2ExpertWeights
        ) throws -> MLXArray {
            let out = try #require(
                Inkling2Kernels.$override.withValue(true) {
                    Inkling2Kernels.expertProduct(
                        x, indices: indices, weights: weights, gateUp: up, down: down,
                        gateCap: 4, upCap: 25)
                })
            eval(out)
            return out
        }

        /// Runs `body` with the one-row GEMV version and rows per lane set.
        static func withGemv<R>(
            _ version: Int, downRows: Int = 1, upRows: Int = 2, _ body: () throws -> R
        ) rethrows -> R {
            let (v, d, u) = (
                Inkling2Kernels.expertGemvVersion, Inkling2Kernels.expertDownRowsPerLane,
                Inkling2Kernels.expertUpRowsPerLane
            )
            defer {
                Inkling2Kernels.expertGemvVersion = v
                Inkling2Kernels.expertDownRowsPerLane = d
                Inkling2Kernels.expertUpRowsPerLane = u
            }
            Inkling2Kernels.expertGemvVersion = version
            Inkling2Kernels.expertDownRowsPerLane = downRows
            Inkling2Kernels.expertUpRowsPerLane = upRows
            return try body()
        }

        static let shapes: [(hidden: Int, intermediate: Int)] = [(512, 256), (2304, 1152)]

        // MARK: Experts

        /// The version-2 GEMV (1 or 2 gate/up and down rows per lane) gives the version-1
        /// GEMV's bits, and both give the bits of the same row run by the tiles
        /// next to companions (rows 2, 8 and 33: unplanned and planned tiles).
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8, .dense])
        func gemvVersionsMatchTheTiles(format: Inkling2ExpertFormat) throws {
            let topK = 8
            for (si, shape) in Self.shapes.enumerated() {
                let (hidden, intermediate) = shape
                let experts = hidden > 1024 ? 32 : 128
                let up = Self.randomBank(
                    format, experts: experts, outputs: 2 * intermediate, inputs: hidden,
                    seed: UInt64(11 + si))
                let down = Self.randomBank(
                    format, experts: experts, outputs: hidden, inputs: intermediate,
                    seed: UInt64(21 + si))
                let x = (MLXRandom.normal([33, hidden], key: MLXRandom.key(31)) * 0.5)
                    .asType(.bfloat16)
                let (indices, _) = Self.routing(rows: 33, experts: experts, topK: topK, seed: 41)
                for target in [0, 32] {
                    let row = x[target ..< target + 1]
                    let ids = indices[target ..< target + 1]
                    let v2 = try Self.withGemv(2) { try Self.pairOutputs(row, ids, up, down) }
                    let v2d2 = try Self.withGemv(2, downRows: 2) {
                        try Self.pairOutputs(row, ids, up, down)
                    }
                    let upOneRow = try Self.withGemv(2, upRows: 1) {
                        try Self.pairOutputs(row, ids, up, down)
                    }
                    let v1 = try Self.withGemv(1) { try Self.pairOutputs(row, ids, up, down) }
                    #expect(Self.bitsEqual(v2, v1), "\(format) \(shape) v2 vs v1")
                    #expect(Self.bitsEqual(v2d2, v1), "\(format) \(shape) v2 down 2 vs v1")
                    #expect(
                        Self.bitsEqual(upOneRow, v1),
                        "\(format) \(shape) v2 gate/up one row per lane vs v1")
                    for rows in [2, 8, 33] {
                        let first = max(0, min(target - rows + 1, 33 - rows))
                        let at = target - first
                        let tiles = try Self.pairOutputs(
                            x[first ..< first + rows], indices[first ..< first + rows], up, down)
                        #expect(
                            Self.bitsEqual(tiles[(at * topK) ..< (at * topK + topK)], v2),
                            "\(format) \(shape) rows \(rows)")
                    }
                }
            }
        }

        /// The grouped GEMVs of 2 to 8 rows (one or two gate/up rows per lane)
        /// give the tiles' bits for every pair, and each row equals the same row
        /// alone (one-row GEMV), with experts shared by several rows.
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8, .dense])
        func groupedGemvMatchesTheTiles(format: Inkling2ExpertFormat) throws {
            let topK = 8
            for (si, shape) in Self.shapes.enumerated() {
                let (hidden, intermediate) = shape
                let experts = hidden > 1024 ? 24 : 48
                let up = Self.randomBank(
                    format, experts: experts, outputs: 2 * intermediate, inputs: hidden,
                    seed: UInt64(131 + si))
                let down = Self.randomBank(
                    format, experts: experts, outputs: hidden, inputs: intermediate,
                    seed: UInt64(141 + si))
                let x = (MLXRandom.normal([8, hidden], key: MLXRandom.key(151)) * 0.5)
                    .asType(.bfloat16)
                let (indices, _) = Self.routing(rows: 8, experts: experts, topK: topK, seed: 161)
                let alone = try (0 ..< 8).map {
                    try Self.pairOutputs(x[$0 ..< $0 + 1], indices[$0 ..< $0 + 1], up, down)
                }
                let saved = Inkling2Kernels.expertGroupGemv
                defer { Inkling2Kernels.expertGroupGemv = saved }
                for rows in [2, 3, 4, 5, 8] {
                    Inkling2Kernels.expertGroupGemv = false
                    let tiles = try Self.pairOutputs(x[0 ..< rows], indices[0 ..< rows], up, down)
                    Inkling2Kernels.expertGroupGemv = true
                    for upRows in [2, 1] {
                        let grouped = try Self.withGemv(2, upRows: upRows) {
                            try Self.pairOutputs(x[0 ..< rows], indices[0 ..< rows], up, down)
                        }
                        #expect(
                            Self.bitsEqual(grouped, tiles),
                            "\(format) \(shape) rows \(rows) gate/up rows/lane \(upRows)")
                        for r in 0 ..< rows {
                            #expect(
                                Self.bitsEqual(grouped[(r * topK) ..< (r * topK + topK)], alone[r]),
                                "\(format) \(shape) rows \(rows) row \(r)")
                        }
                    }
                }
            }
        }

        /// A row alone (one-row GEMVs) equals the same row in steps of 2, 8 and
        /// 33 rows (tiles, planned above 64 pairs) whose companions share all,
        /// some or none of its experts, at varying positions.
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8])
        func rowAloneEqualsRowInCompany(format: Inkling2ExpertFormat) throws {
            let (experts, hidden, intermediate, topK) = (48, 2304, 1152, 8)
            let up = Self.randomBank(
                format, experts: experts, outputs: 2 * intermediate, inputs: hidden, seed: 51)
            let down = Self.randomBank(
                format, experts: experts, outputs: hidden, inputs: intermediate, seed: 61)
            let target = (MLXRandom.normal([1, hidden], key: MLXRandom.key(71)) * 0.5)
                .asType(.bfloat16)
            let (targetIds, targetWeights) = Self.routing(
                rows: 1, experts: experts, topK: topK, seed: 81)
            let alone = try Self.combined(target, targetIds, targetWeights, up, down)
            let mine = targetIds.asArray(UInt32.self)
            let others = (0 ..< UInt32(experts)).filter { !mine.contains($0) }
            for rows in [2, 8, 33] {
                for (mi, mix) in ["same", "some", "none", "random"].enumerated() {
                    let count = rows - 1
                    let companions =
                        (MLXRandom.normal(
                            [count, hidden], key: MLXRandom.key(UInt64(100 + rows * 4 + mi))) * 0.5)
                        .asType(.bfloat16)
                    let (randomIds, weights) = Self.routing(
                        rows: count, experts: experts, topK: topK, seed: UInt64(200 + rows))
                    let ids: MLXArray
                    switch mix {
                    case "same":
                        ids = MLXArray(
                            (0 ..< count).flatMap { i in (0 ..< topK).map { mine[(i + $0) % topK] }
                            },
                            [count, topK])
                    case "some":
                        ids = MLXArray(
                            (0 ..< count).flatMap { i in
                                (0 ..< topK).map {
                                    $0 < 3
                                        ? mine[(i + $0) % topK]
                                        : others[(i * 5 + $0 * 11) % others.count]
                                }
                            }, [count, topK])
                    case "none":
                        ids = MLXArray(
                            (0 ..< count).flatMap { i in
                                (0 ..< topK).map { others[(i * 7 + $0 * 13) % others.count] }
                            }, [count, topK])
                    default:
                        ids = randomIds
                    }
                    let position = (rows * 3 + mi) % rows
                    let before = position
                    var xs: [MLXArray] = []
                    var idxs: [MLXArray] = []
                    var ws: [MLXArray] = []
                    if before > 0 {
                        xs.append(companions[0 ..< before])
                        idxs.append(ids[0 ..< before])
                        ws.append(weights[0 ..< before])
                    }
                    xs.append(target)
                    idxs.append(targetIds)
                    ws.append(targetWeights)
                    if before < count {
                        xs.append(companions[before...])
                        idxs.append(ids[before...])
                        ws.append(weights[before...])
                    }
                    let out = try Self.combined(
                        concatenated(xs, axis: 0), concatenated(idxs, axis: 0),
                        concatenated(ws, axis: 0), up, down)
                    #expect(
                        Self.bitsEqual(out[position ..< position + 1], alone),
                        "\(format) rows \(rows) \(mix) at \(position)")
                }
            }
        }

        /// Verify windows of L = 2..8 tokens: position i equals the one-token
        /// decode step of the same input.
        @Test func verifyWindowPositionsEqualDecodeSteps() throws {
            let (experts, hidden, intermediate, topK) = (32, 2304, 1152, 8)
            for format in [Inkling2ExpertFormat.nvfp4, .mxfp8] {
                let up = Self.randomBank(
                    format, experts: experts, outputs: 2 * intermediate, inputs: hidden,
                    seed: 91)
                let down = Self.randomBank(
                    format, experts: experts, outputs: hidden, inputs: intermediate, seed: 92)
                let x = (MLXRandom.normal([8, hidden], key: MLXRandom.key(93)) * 0.5)
                    .asType(.bfloat16)
                let (indices, weights) = Self.routing(
                    rows: 8, experts: experts, topK: topK, seed: 94)
                let steps = try (0 ..< 8).map {
                    try Self.combined(
                        x[$0 ..< $0 + 1], indices[$0 ..< $0 + 1], weights[$0 ..< $0 + 1], up,
                        down)
                }
                for l in 2 ... 8 {
                    let window = try Self.combined(
                        x[0 ..< l], indices[0 ..< l], weights[0 ..< l], up, down)
                    for i in 0 ..< l {
                        #expect(
                            Self.bitsEqual(window[i ..< i + 1], steps[i]), "\(format) L \(l) i \(i)"
                        )
                    }
                }
            }
        }

        // MARK: Host reference

        static let e2m1: [Double] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

        static func e4m3(_ b: UInt8) -> Double {
            let sign: Double = b & 0x80 != 0 ? -1 : 1
            let e = Int((b >> 3) & 0xF)
            let m = Double(b & 7)
            return sign * (e == 0 ? m / 8 * pow(2, -6) : (1 + m / 8) * pow(2, Double(e - 7)))
        }

        /// A bank as Doubles `[E][N][K]` (global scale included).
        static func dequantized(_ bank: Inkling2ExpertWeights, outputs: Int, inputs: Int)
            -> [[[Double]]]
        {
            let experts = bank.experts
            switch bank.format {
            case .dense:
                let v = bank.weight.asType(.float32).asArray(Float.self)
                return (0 ..< experts).map { e in
                    (0 ..< outputs).map { r in
                        (0 ..< inputs).map { Double(v[(e * outputs + r) * inputs + $0]) }
                    }
                }
            case .nvfp4:
                let words = bank.weight.asArray(UInt32.self)
                let scales = bank.scales!.asArray(UInt8.self)
                let gs = bank.globalScale!.asArray(Float.self)
                return (0 ..< experts).map { e in
                    (0 ..< outputs).map { r in
                        let row = e * outputs + r
                        return (0 ..< inputs).map { k in
                            let word = words[row * inputs / 8 + k / 8]
                            let nibble = Int((word >> UInt32(4 * (k % 8))) & 0xF)
                            let v = (nibble & 8 != 0 ? -1 : 1) * e2m1[nibble & 7]
                            return v * e4m3(scales[row * inputs / 16 + k / 16]) * Double(gs[e])
                                / 2688
                        }
                    }
                }
            case .mxfp8:
                let words = bank.weight.asArray(UInt32.self)
                let scales = bank.scales!.asArray(UInt8.self)
                return (0 ..< experts).map { e in
                    (0 ..< outputs).map { r in
                        let row = e * outputs + r
                        return (0 ..< inputs).map { k in
                            let byte = UInt8(
                                (words[row * inputs / 4 + k / 4] >> UInt32(8 * (k % 4))) & 0xFF)
                            return e4m3(byte)
                                * pow(2, Double(Int(scales[row * inputs / 32 + k / 32]) - 127))
                        }
                    }
                }
            }
        }

        /// Round to the nearest BF16 (ties to even), through FP32.
        static func bf16(_ v: Double) -> Double {
            let bits = Float(v).bitPattern
            let rounded = (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000
            return Double(Float(bitPattern: rounded))
        }

        /// One-row expert outputs against a Double reference with the module's
        /// rounding points (projection outputs and the activation in BF16):
        /// every pair output within 1 BF16 ulp plus 0.1 % of the largest output,
        /// cosine >= 0.99999.
        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8, .dense])
        func oneRowOutputsMatchAHostReference(format: Inkling2ExpertFormat) throws {
            let (experts, hidden, intermediate, topK) = (16, 256, 128, 4)
            let up = Self.randomBank(
                format, experts: experts, outputs: 2 * intermediate, inputs: hidden, seed: 7)
            let down = Self.randomBank(
                format, experts: experts, outputs: hidden, inputs: intermediate, seed: 8)
            let x = (MLXRandom.normal([1, hidden], key: MLXRandom.key(9)) * 0.5).asType(.bfloat16)
            let (indices, _) = Self.routing(rows: 1, experts: experts, topK: topK, seed: 10)
            let ours = try Self.pairOutputs(x, indices, up, down).asType(.float32)
            let w13 = Self.dequantized(up, outputs: 2 * intermediate, inputs: hidden)
            let w2 = Self.dequantized(down, outputs: hidden, inputs: intermediate)
            let xs = x.asType(.float32).asArray(Float.self).map(Double.init)
            func dot(_ a: [Double], _ b: [Double]) -> Double {
                zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
            }
            var reference: [Float] = []
            for e in indices.asArray(UInt32.self).map(Int.init) {
                let act = (0 ..< intermediate).map { i -> Double in
                    let g = Self.bf16(dot(w13[e][2 * i], xs))
                    let u = Self.bf16(dot(w13[e][2 * i + 1], xs))
                    return Self.bf16(tanh(g / 4) * 4 / (1 + exp(-g)) * (tanh(u / 25) * 25))
                }
                for n in 0 ..< hidden {
                    reference.append(Float(dot(w2[e][n], act)))
                }
            }
            let ref = MLXArray(reference, [topK, hidden])
            let tolerance = abs(ref) * Float(1.0 / 128) + abs(ref).max() * Float(1e-3)
            let within = (abs(ours - ref) .<= tolerance).all().item(Bool.self)
            let a = ours.reshaped(-1)
            let b = ref.reshaped(-1)
            let cos = ((a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))).item(Float.self)
            print("INKLING2_EXPERTS one-row \(format) vs host reference: cosine \(cos)")
            #expect(within)
            #expect(cos >= 0.99999)
        }

        // MARK: Router

        static func router(hidden: Int, seed: UInt64) -> Inkling2Router {
            let router = Inkling2Router(hidden: hidden, experts: 128, topK: 8, routeScale: 8)
            SyntheticModel.randomize(router, seed: seed)
            // The checkpoint's router weights are BF16 values stored as FP32.
            let w = router.weight.asType(.bfloat16).asType(.float32)
            let b = router.bias
            eval(w, b)
            router.update(parameters: ModuleParameters.unflattened(["weight": w, "bias": b]))
            return router
        }

        /// The two-launch router of `x` rows: FP32 logits by the segmented
        /// product, then the select kernel.
        static func twoLaunch(_ router: Inkling2Router, _ x: MLXArray) throws -> (
            MLXArray, MLXArray, MLXArray
        ) {
            let logits = try #require(
                Inkling2Kernels.segmentedProduct(
                    [x],
                    [
                        Inkling2Kernels.Segment(
                            input: 0, weight: Inkling2Kernels.halfCopiesForRouter(router.weight))
                    ], outType: .float32))
            let out = Inkling2Kernels.select(router, logits: logits)
            eval(out.0, out.1, out.2)
            return out
        }

        static func fused(_ router: Inkling2Router, _ x: MLXArray) throws -> (
            MLXArray, MLXArray, MLXArray
        ) {
            let out = try #require(Inkling2Kernels.routerSelectOneRow(router, x))
            eval(out.0, out.1, out.2)
            return out
        }

        /// The fused one-launch router gives the two-launch router's indices,
        /// weights and selection scores bit for bit, for a row alone and for the
        /// same row inside a verify window or a batch (L = 2..8 and 33 rows).
        @Test(arguments: [256, 2304])
        func fusedRouterMatchesTheTwoLaunchRouter(hidden: Int) throws {
            #expect(Inkling2Kernels.fusedRouter)
            let router = Self.router(hidden: hidden, seed: UInt64(hidden))
            let x = MLXRandom.normal([33, hidden], key: MLXRandom.key(5)).asType(.bfloat16)
            let saved = (
                Inkling2Kernels.fusedRouterRank, Inkling2Kernels.fusedRouterOutputsPerGroup
            )
            defer {
                Inkling2Kernels.fusedRouterRank = saved.0
                Inkling2Kernels.fusedRouterOutputsPerGroup = saved.1
            }
            for (rank, opg) in [(0, 4), (1, 4), (1, 2)] {
                Inkling2Kernels.fusedRouterRank = rank
                Inkling2Kernels.fusedRouterOutputsPerGroup = opg
                let alone = try (0 ..< 33).map { try Self.fused(router, x[$0 ..< $0 + 1]) }
                for l in [2, 3, 4, 5, 6, 7, 8, 33] {
                    let (indices, weights, selection) = try Self.twoLaunch(router, x[0 ..< l])
                    for i in 0 ..< l {
                        let tag = "rank select \(rank) outputs/group \(opg) L \(l) i \(i)"
                        #expect(Self.bitsEqual(alone[i].0, indices[i ..< i + 1]), "\(tag)")
                        #expect(Self.bitsEqual(alone[i].1, weights[i ..< i + 1]), "\(tag)")
                        #expect(Self.bitsEqual(alone[i].2, selection[i ..< i + 1]), "\(tag)")
                    }
                }
            }
        }

        /// Many fused routers in one graph (independent launches of one router,
        /// and a chain of dependent ones) all match their two-launch results:
        /// the router's workspace on this stream resets between launches.
        @Test func fusedRouterLaunchesAreIndependent() throws {
            let router = Self.router(hidden: 512, seed: 3)
            let x = MLXRandom.normal([24, 512], key: MLXRandom.key(4)).asType(.bfloat16)
            let independent = (0 ..< 24).map {
                Inkling2Kernels.routerSelectOneRow(router, x[$0 ..< $0 + 1])!
            }
            eval(independent.flatMap { [$0.0, $0.1, $0.2] })
            let (indices, weights, _) = try Self.twoLaunch(router, x)
            for i in 0 ..< 24 {
                #expect(Self.bitsEqual(independent[i].0, indices[i ..< i + 1]), "row \(i)")
                #expect(Self.bitsEqual(independent[i].1, weights[i ..< i + 1]), "row \(i)")
            }
            // A dependent chain: each step's input depends on the previous picks.
            var h = x[0 ..< 1]
            var fusedChain: [MLXArray] = []
            for _ in 0 ..< 12 {
                let (idx, w, _) = Inkling2Kernels.routerSelectOneRow(router, h)!
                fusedChain.append(idx)
                h = (h.asType(.float32) + w.sum() * 0.01).asType(.bfloat16)
            }
            eval(fusedChain)
            h = x[0 ..< 1]
            for step in 0 ..< 12 {
                let (idx, w, _) = try Self.twoLaunch(router, h)
                #expect(Self.bitsEqual(fusedChain[step], idx), "step \(step)")
                h = (h.asType(.float32) + w.sum() * 0.01).asType(.bfloat16)
            }
        }

        /// One router module routing on two MLX streams at once (two engines on
        /// one loaded model): two chains of dependent one-row launches are in
        /// flight together and each matches the two-launch router step by step.
        /// Each stream counts on its own workspace.
        @Test func fusedRouterOnTwoStreamsAtOnce() throws {
            let router = Self.router(hidden: 2304, seed: 23)
            let x = MLXRandom.normal([2, 2304], key: MLXRandom.key(24)).asType(.bfloat16)
            eval(x)
            func next(_ h: MLXArray, _ w: MLXArray) -> MLXArray {
                (h.asType(.float32) + w.sum() * 0.01).asType(.bfloat16)
            }
            for round in 0 ..< 4 {
                var chains: [[MLXArray]] = [[], []]
                for c in 0 ..< 2 {
                    Stream.withNewDefaultStream(device: .gpu) {
                        var h = x[c ..< c + 1]
                        for _ in 0 ..< 48 {
                            let (idx, w, _) = Inkling2Kernels.routerSelectOneRow(router, h)!
                            chains[c] += [idx, w]
                            h = next(h, w)
                        }
                        asyncEval(chains[c])
                    }
                }
                eval(chains[0] + chains[1])
                for c in 0 ..< 2 {
                    var h = x[c ..< c + 1]
                    for step in 0 ..< 48 {
                        let (idx, w, _) = try Self.twoLaunch(router, h)
                        #expect(
                            Self.bitsEqual(chains[c][2 * step], idx)
                                && Self.bitsEqual(chains[c][2 * step + 1], w),
                            "round \(round) stream \(c) step \(step)")
                        h = next(h, w)
                    }
                }
            }
        }

        /// With the fused router off and the `router` GEMV site off, one-row
        /// routing falls back to the two launches (nil here).
        @Test func fusedRouterSwitchFallsBack() throws {
            let router = Self.router(hidden: 256, seed: 9)
            let x = MLXRandom.normal([1, 256], key: MLXRandom.key(10)).asType(.bfloat16)
            let saved = Inkling2Kernels.fusedRouter
            defer { Inkling2Kernels.fusedRouter = saved }
            Inkling2Kernels.fusedRouter = false
            if !Inkling2Kernels.gemvSite("router") {
                #expect(Inkling2Kernels.routerSelectOneRow(router, x) == nil)
            }
        }
    }
}
