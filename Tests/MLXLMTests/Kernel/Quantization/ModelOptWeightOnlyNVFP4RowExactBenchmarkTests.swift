// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLMCommon

/// Synthetic projections at the shapes of a dense 30B decoder with mixed
/// weight-only NVFP4, calibrated FP8 and BF16 projections. Opt-in: set
/// `MODELOPT_ROWEXACT_BENCH` to `probe` (row-identity facts of MLX's own
/// kernels), `time` (wall-clock tables) or `all`. Reports, not assertions.
extension KernelTests {
    @Suite(
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MODELOPT_ROWEXACT_BENCH"] != nil))
    struct ModelOptWeightOnlyNVFP4RowExactBenchmarkTests {
        enum Format: String { case nvfp4, fp8, bf16 }

        /// One projection shape with the number of modules of each format
        /// that a decode step runs.
        struct Shape {
            let name: String
            let outputs: Int
            let inputs: Int
            let counts: [Format: Int]
        }

        static let shapes: [Shape] = [
            Shape(
                name: "mlp.gate/up", outputs: 19968, inputs: 6656,
                counts: [.nvfp4: 90, .fp8: 14]),
            Shape(
                name: "mlp.down", outputs: 6656, inputs: 19968, counts: [.nvfp4: 40, .fp8: 12]),
            Shape(
                name: "attn.q/gate", outputs: 4096, inputs: 6656,
                counts: [.nvfp4: 36, .fp8: 56, .bf16: 12]),
            Shape(
                name: "attn.k/v", outputs: 256, inputs: 6656,
                counts: [.nvfp4: 36, .fp8: 56, .bf16: 12]),
            Shape(name: "attn.o", outputs: 6656, inputs: 4096, counts: [.nvfp4: 10, .fp8: 42]),
            Shape(name: "lm_head", outputs: 202_048, inputs: 6656, counts: [.nvfp4: 1]),
        ]

        static var mode: String {
            ProcessInfo.processInfo.environment["MODELOPT_ROWEXACT_BENCH"] ?? ""
        }

        static func wants(_ what: String) -> Bool { mode == what || mode == "all" }

        struct NVFP4 {
            let packed: MLXArray
            let scales: MLXArray
            var words: MLXArray { packed.view(dtype: .uint32) }
        }

        static func nvfp4(_ shape: Shape, seed: UInt64) -> NVFP4 {
            let packed = MLXRandom.randInt(
                Int32(0) ..< 256, [shape.outputs, shape.inputs / 2], key: MLXRandom.key(seed)
            ).asType(.uint8)
            // E4M3 block scales between 0.25 (0x28) and 4 (0x48).
            let scales = MLXRandom.randInt(
                Int32(0x28) ..< 0x49, [shape.outputs, shape.inputs / 16],
                key: MLXRandom.key(seed + 1)
            ).asType(.uint8)
            eval(packed, scales)
            return NVFP4(packed: packed, scales: scales)
        }

        static func fp8(_ shape: Shape, seed: UInt64) throws -> ModelOptCalibratedLinear {
            let raw = MLXRandom.randInt(
                Int32(0) ..< 256, [shape.outputs, shape.inputs], key: MLXRandom.key(seed)
            ).asType(.uint8)
            let bytes = which((raw & UInt8(0x7F)) .== UInt8(0x7F), raw - UInt8(1), raw)
            eval(bytes)
            return try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(e4m3Bytes: bytes, tensorScale: 0.0011),
                inputScale: 0.009, weightDType: .bfloat16, activationOverflow: .saturating)
        }

        static func dense(_ shape: Shape, seed: UInt64) -> MLXArray {
            let weight = (MLXRandom.normal([shape.outputs, shape.inputs], key: MLXRandom.key(seed))
                * 0.02).asType(.bfloat16)
            eval(weight)
            return weight
        }

        static func input(rows: Int, width: Int, seed: UInt64) -> MLXArray {
            let x = (MLXRandom.normal([rows, width], key: MLXRandom.key(seed)) * 0.5)
                .asType(.bfloat16)
            eval(x)
            return x
        }

        static func mlxNVFP4(_ x: MLXArray, _ weight: NVFP4) -> MLXArray {
            quantizedMM(
                x, weight.words, scales: weight.scales, biases: nil, transpose: true,
                groupSize: 16, bits: 4, mode: .nvfp4)
        }

        /// The row-exact product; `scalarRows` picks which kernel takes how
        /// many rows (nil: the device's default).
        static func rowExact(
            _ x: MLXArray, _ weight: NVFP4, scalarRows: Int? = nil, tensorScale: MLXArray? = nil
        ) -> MLXArray {
            let saved = ModelOptWeightOnlyNVFP4RowExact.scalarRows
            defer { ModelOptWeightOnlyNVFP4RowExact.scalarRows = saved }
            if let scalarRows { ModelOptWeightOnlyNVFP4RowExact.scalarRows = scalarRows }
            return ModelOptWeightOnlyNVFP4RowExact.product(
                x, weight: weight.words, blockScales: weight.scales, tensorScale: tensorScale)
        }

        /// `rowExactForward` with the FP8 activation cast as its own pass
        /// instead of inside the product (the same values either way).
        static func fp8SeparateCast(
            _ layer: ModelOptCalibratedLinear, _ x: MLXArray, geometry: (Int, Int)? = nil
        ) -> MLXArray {
            ModelOptRowExact.fusedActivation = false
            ModelOptRowExact.geometryOverride = geometry
            defer {
                ModelOptRowExact.fusedActivation = true
                ModelOptRowExact.geometryOverride = nil
            }
            return try! layer.rowExactForward(x)
        }

        /// Raw 16-bit patterns of each row.
        static func rowBits(_ y: MLXArray) -> [[UInt16]] {
            let rows = y.dim(0)
            let width = y.size / rows * y.dtype.size / 2
            let flat = y.reshaped(rows, -1).view(dtype: .uint16).asArray(UInt16.self)
            return (0 ..< rows).map { Array(flat[($0 * width) ..< (($0 + 1) * width)]) }
        }

        /// Row counts grouped so that every vector has the same bytes at each
        /// count of a group (compared on the vectors two counts share).
        static func classes(_ rows: [Int], _ product: (Int) -> [[UInt16]]) -> String {
            var groups: [(bits: [[UInt16]], rows: [Int])] = []
            for count in rows {
                let bits = product(count)
                if let index = groups.firstIndex(where: { group in
                    zip(group.bits, bits).allSatisfy { $0 == $1 }
                }) {
                    groups[index].rows.append(count)
                    if bits.count > groups[index].bits.count { groups[index].bits = bits }
                } else {
                    groups.append((bits, [count]))
                }
            }
            return groups.map { "{" + $0.rows.map(String.init).joined(separator: ",") + "}" }
                .joined(separator: " ")
        }

        static let probeRows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16]

        /// For each row count: how many rows of the R-row product differ in any
        /// byte from the one-row product of the same vector, and whether a
        /// vector's bytes depend on its position in the call.
        static func report(
            _ label: String, width: Int, dtype: DType = .bfloat16,
            single: (MLXArray) -> MLXArray, batched: (MLXArray) -> MLXArray
        ) {
            let x = input(rows: 16, width: width, seed: 11).asType(dtype)
            eval(x)
            let alone = (0 ..< 16).map { rowBits(single(x[$0 ..< $0 + 1]))[0] }
            var lines: [String] = []
            for count in probeRows {
                let together = rowBits(batched(x[0 ..< count]))
                let differing = (0 ..< count).filter { together[$0] != alone[$0] }.count
                let elements = (0 ..< count).reduce(0) { total, row in
                    total + zip(together[row], alone[row]).filter { $0 != $1 }.count
                }
                // The same vectors in reverse order.
                let order = MLXArray((0 ..< count).reversed().map { Int32($0) })
                let reversed = rowBits(batched(x[0 ..< count][order]))
                let moved = (0 ..< count).filter { reversed[count - 1 - $0] != together[$0] }.count
                lines.append(
                    "R=\(count): rows!=one-row \(differing)/\(count) "
                        + "(elements \(elements)/\(count * alone[0].count)), "
                        + "rows changed by position \(moved)/\(count)")
            }
            let groups = classes(probeRows) { rowBits(batched(x[0 ..< $0])) }
            print("PROBE \(label): classes by R: \(groups)")
            for line in lines { print("PROBE \(label): \(line)") }
        }

        @Test(.enabled(if: wants("probe")))
        func rowIdentityOfExistingPaths() throws {
            print("PROBE device \(GPU.deviceInfo().architecture)")
            for shape in Self.shapes {
                let tag = "\(shape.name) [\(shape.outputs), \(shape.inputs)]"
                if shape.counts[.nvfp4] != nil {
                    let weight = Self.nvfp4(shape, seed: 21)
                    Self.report(
                        "mlx.nvfp4 \(tag)", width: shape.inputs,
                        single: { Self.mlxNVFP4($0, weight) },
                        batched: { Self.mlxNVFP4($0, weight) })
                    // Float32 activations hold the same values and show the
                    // sums before the bfloat16 rounding hides most differences.
                    Self.report(
                        "mlx.nvfp4.float32 \(tag)", width: shape.inputs, dtype: .float32,
                        single: { Self.mlxNVFP4($0, weight) },
                        batched: { Self.mlxNVFP4($0, weight) })
                    for dtype in [DType.bfloat16, .float32] {
                        Self.report(
                            "rowexact.nvfp4.\(dtype) \(tag)", width: shape.inputs, dtype: dtype,
                            single: { Self.rowExact($0, weight) },
                            batched: { Self.rowExact($0, weight) })
                    }
                }
                if shape.counts[.fp8] != nil {
                    let layer = try Self.fp8(shape, seed: 31)
                    Self.report(
                        "fp8.rowExactForward \(tag)", width: shape.inputs,
                        single: { try! layer.graphForward($0) },
                        batched: { try! layer.rowExactForward($0) })
                    Self.report(
                        "fp8.rowExactForward.separateCast \(tag)", width: shape.inputs,
                        single: { try! layer.graphForward($0) },
                        batched: { Self.fp8SeparateCast(layer, $0) })
                    Self.report(
                        "fp8.graphForward \(tag)", width: shape.inputs,
                        single: { try! layer.graphForward($0) },
                        batched: { try! layer.graphForward($0) })
                }
                if shape.counts[.bf16] != nil {
                    let weight = Self.dense(shape, seed: 41)
                    Self.report(
                        "bf16.rowExact \(tag)", width: shape.inputs,
                        single: { matmul($0, weight.transposed()) },
                        batched: { ModelOptRowExact.matmul($0, weight) })
                    Self.report(
                        "bf16.mlx.matmul \(tag)", width: shape.inputs,
                        single: { matmul($0, weight.transposed()) },
                        batched: { matmul($0, weight.transposed()) })
                }
            }
        }

        /// At the real shapes: every row of an R-row product has the bytes of
        /// its one-row product, and the two kernels agree where both run.
        @Test(.enabled(if: wants("probe")))
        func realShapeRowsHaveTheBytesOfOneRowProducts() {
            for shape in Self.shapes where shape.counts[.nvfp4] != nil {
                let weight = Self.nvfp4(shape, seed: 21)
                for dtype in [DType.bfloat16, .float32] {
                    let x = Self.input(rows: 16, width: shape.inputs, seed: 11).asType(dtype)
                    eval(x)
                    let alone = (0 ..< 16).map {
                        Self.rowBits(Self.rowExact(x[$0 ..< $0 + 1], weight, scalarRows: 0))[0]
                    }
                    var splits = [0]
                    if ModelOptWeightOnlyNVFP4RowExact.scalarKernelMeasured { splits += [1, 2] }
                    var compared = 0
                    for scalarRows in splits {
                        for rows in [1, 2, 3, 5, 8, 9, 16] {
                            let together = Self.rowBits(
                                Self.rowExact(x[0 ..< rows], weight, scalarRows: scalarRows))
                            #expect(
                                together == Array(alone[0 ..< rows]),
                                "\(shape.name) \(dtype) rows \(rows) scalar \(scalarRows)")
                            compared += rows * shape.outputs
                        }
                    }
                    print(
                        "PROBE rowexact.nvfp4 \(shape.name) [\(shape.outputs), \(shape.inputs)] "
                            + "\(dtype): \(compared) elements compared with one-row products, "
                            + "kernel splits \(splits)")
                }
            }
        }

        /// The existing FP8 and BF16 row-exact paths at the real shapes, and the
        /// FP8 path with its activation cast as a separate pass.
        @Test(.enabled(if: wants("probe")))
        func realShapeExistingRowExactPaths() throws {
            for shape in Self.shapes {
                let x = Self.input(rows: 16, width: shape.inputs, seed: 11)
                if shape.counts[.fp8] != nil {
                    let layer = try Self.fp8(shape, seed: 31)
                    let alone = (0 ..< 16).map {
                        Self.rowBits(try! layer.graphForward(x[$0 ..< $0 + 1]))[0]
                    }
                    for rows in [2, 8, 16] {
                        let expected = Array(alone[0 ..< rows])
                        #expect(Self.rowBits(try layer.rowExactForward(x[0 ..< rows])) == expected)
                        #expect(Self.rowBits(Self.fp8SeparateCast(layer, x[0 ..< rows])) == expected)
                    }
                }
                if shape.counts[.bf16] != nil {
                    let weight = Self.dense(shape, seed: 41)
                    let alone = (0 ..< 16).map {
                        Self.rowBits(matmul(x[$0 ..< $0 + 1], weight.transposed()))[0]
                    }
                    for rows in [2, 8, 16] {
                        #expect(
                            Self.rowBits(ModelOptRowExact.matmul(x[0 ..< rows], weight))
                                == Array(alone[0 ..< rows]))
                    }
                }
            }
        }

        struct Stats {
            let median: Double
            let p10: Double
            let p90: Double
            let runs: Int
            var minimum = 0.0
        }

        /// Milliseconds per call, warm: `depth` calls built into one graph
        /// and evaluated together, at least 15 runs and 0.4 s.
        static func measure(depth: Int, _ body: () -> MLXArray) -> Stats {
            func once() -> Double {
                let start = DispatchTime.now().uptimeNanoseconds
                eval((0 ..< depth).map { _ in body() })
                return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(depth)
            }
            for _ in 0 ..< 3 { _ = once() }
            var times: [Double] = []
            let begin = DispatchTime.now().uptimeNanoseconds
            while times.count < 15
                || (times.count < 400 && DispatchTime.now().uptimeNanoseconds - begin < 400_000_000)
            {
                times.append(once())
            }
            times.sort()
            return Stats(
                median: times[times.count / 2], p10: times[times.count / 10],
                p90: times[times.count * 9 / 10], runs: times.count)
        }

        /// `measure` for several paths in turn, so that load from other
        /// processes falls on all of them alike: 40 rounds of one evaluation each.
        static func interleaved(depth: Int, rounds: Int = 40, _ paths: [(String, () -> MLXArray)])
            -> [(String, Stats)]
        {
            var times = Array(repeating: [Double](), count: paths.count)
            for round in 0 ..< rounds + 3 {
                for (index, path) in paths.enumerated() {
                    let start = DispatchTime.now().uptimeNanoseconds
                    eval((0 ..< depth).map { _ in path.1() })
                    let elapsed =
                        Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(depth)
                    if round >= 3 { times[index].append(elapsed) }
                }
            }
            return paths.indices.map { index in
                let sorted = times[index].sorted()
                return (
                    paths[index].0,
                    Stats(
                        median: sorted[sorted.count / 2], p10: sorted[sorted.count / 10],
                        p90: sorted[sorted.count * 9 / 10], runs: sorted.count,
                        minimum: sorted[0])
                )
            }
        }

        static let timedRows = [1, 2, 4, 8, 16]
        nonisolated(unsafe) static let tensorScale = MLXArray([Float(0.0037)])

        /// The weight-only linear as the model runs it: the packed product,
        /// then the tensor scale in float32.
        static func scaled(_ product: MLXArray) -> MLXArray {
            (product.asType(.float32) * tensorScale).asType(product.dtype)
        }

        static func perRow(_ x: MLXArray, _ forward: (MLXArray) -> MLXArray) -> MLXArray {
            let rows = x.dim(0)
            return rows == 1
                ? forward(x) : concatenated((0 ..< rows).map { forward(x[$0 ..< $0 + 1]) }, axis: 0)
        }

        /// Paths of one projection by name; each maps `[R, K]` to `[R, N]`.
        typealias Paths = [(name: String, exact: Bool, forward: (MLXArray) -> MLXArray)]

        static func paths(_ format: Format, _ shape: Shape) throws -> (Paths, bytes: Double) {
            switch format {
            case .nvfp4:
                let weight = nvfp4(shape, seed: 21)
                return (
                    [
                        ("mlx", false, { scaled(mlxNVFP4($0, weight)) }),
                        ("mlx-per-row", true, { x in perRow(x) { scaled(mlxNVFP4($0, weight)) } }),
                        (
                            "row-exact", true,
                            { rowExact($0, weight, tensorScale: tensorScale) }
                        ),
                        (
                            "row-exact-matrix", true,
                            { rowExact($0, weight, scalarRows: 0, tensorScale: tensorScale) }
                        ),
                    ], Double(weight.packed.nbytes + weight.scales.nbytes)
                )
            case .fp8:
                let layer = try fp8(shape, seed: 31)
                return (
                    [
                        ("graph", false, { try! layer.graphForward($0) }),
                        ("row-exact", true, { try! layer.rowExactForward($0) }),
                        ("row-exact-cast", true, { fp8SeparateCast(layer, $0) }),
                        ("row-exact-cast-tm2", true, { fp8SeparateCast(layer, $0, geometry: (2, 1)) }),
                        ("row-exact-cast-tm8", true, { fp8SeparateCast(layer, $0, geometry: (8, 1)) }),
                    ], Double(shape.outputs * shape.inputs)
                )
            case .bf16:
                let weight = dense(shape, seed: 41)
                return (
                    [
                        ("mlx", false, { matmul($0, weight.transposed()) }),
                        (
                            "row-exact", true,
                            { $0.dim(0) == 1
                                ? matmul($0, weight.transposed())
                                : ModelOptRowExact.matmul($0, weight) }
                        ),
                    ], Double(weight.nbytes)
                )
            }
        }

        /// Per-call time of every format, shape, path and row count (the paths
        /// of a cell in turn), then the projection time of a whole decode step
        /// summed over the module counts.
        @Test(.enabled(if: wants("time")))
        func projectionTimes() throws {
            print("BENCH device \(GPU.deviceInfo().architecture)")
            // (format, path, rows) -> milliseconds of all modules of a step.
            var medians: [String: Double] = [:]
            var minima: [String: Double] = [:]
            for shape in Self.selectedShapes() {
                let depth = shape.outputs > 100_000 ? 2 : 8
                for format in [Format.nvfp4, .fp8, .bf16] {
                    guard let count = shape.counts[format] else { continue }
                    let (paths, bytes) = try Self.paths(format, shape)
                    for rows in Self.timedRows {
                        let x = Self.input(rows: rows, width: shape.inputs, seed: 11)
                        let results = Self.interleaved(
                            depth: depth, paths.map { path in (path.name, { path.forward(x) }) })
                        for (path, (name, stats)) in zip(paths, results) {
                            print(
                                String(
                                    format:
                                        "BENCH %@ %@ [%d, %d] x%d R=%d %@%@: median %.3f ms "
                                        + "(min %.3f, p10 %.3f, p90 %.3f, %d runs of %d), %.0f GB/s",
                                    format.rawValue, shape.name, shape.outputs, shape.inputs,
                                    count, rows, name, path.exact ? " (exact)" : "",
                                    stats.median, stats.minimum, stats.p10, stats.p90, stats.runs,
                                    depth, bytes / stats.median / 1e6))
                            let key = "\(format.rawValue) \(name) R=\(rows)"
                            medians[key, default: 0] += stats.median * Double(count)
                            minima[key, default: 0] += stats.minimum * Double(count)
                        }
                    }
                }
            }
            for key in medians.keys.sorted() {
                print(
                    String(
                        format: "BENCH step total %@: median %.2f ms, min %.2f ms", key,
                        medians[key]!, minima[key]!))
            }
        }

        /// Every projection of one decode step in one graph and one
        /// evaluation: the module counts of each shape and format, each
        /// reading its own activations. `nvfp4`, `fp8` and `bf16` name the path
        /// of each format.
        static func step(
            rows: Int, nvfp4: String, fp8: String, bf16: String,
            _ modules: [(Format, Shape, Paths)]
        ) -> Stats {
            var calls: [() -> MLXArray] = []
            for (format, shape, paths) in modules {
                let name = [Format.nvfp4: nvfp4, .fp8: fp8, .bf16: bf16][format]!
                let path = paths.first { $0.name == name }!
                let x = input(rows: rows, width: shape.inputs, seed: 11)
                for _ in 0 ..< shape.counts[format]! { calls.append({ path.forward(x) }) }
            }
            func once() -> Double {
                let start = DispatchTime.now().uptimeNanoseconds
                eval(calls.map { $0() })
                return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
            }
            for _ in 0 ..< 3 { _ = once() }
            let times = (0 ..< 30).map { _ in once() }.sorted()
            return Stats(
                median: times[15], p10: times[3], p90: times[27], runs: 30, minimum: times[0])
        }

        @Test(.enabled(if: wants("step")))
        func decodeStepProjections() throws {
            var modules: [(Format, Shape, Paths)] = []
            for shape in Self.shapes {
                for format in [Format.nvfp4, .fp8, .bf16] where shape.counts[format] != nil {
                    modules.append((format, shape, try Self.paths(format, shape).0))
                }
            }
            let cases: [(String, Int, String, String, String)] = [
                ("stock one row (MLX nvfp4, graph fp8, MLX bf16)", 1, "mlx", "graph", "mlx"),
                ("exact one row", 1, "row-exact", "graph", "mlx"),
                ("exact 2 rows", 2, "row-exact", "row-exact-cast", "row-exact"),
                ("exact 4 rows", 4, "row-exact", "row-exact-cast", "row-exact"),
                ("exact 8 rows", 8, "row-exact", "row-exact-cast", "row-exact"),
                ("exact 8 rows, fp8 cast inside", 8, "row-exact", "row-exact", "row-exact"),
                ("exact 8 rows, per-row nvfp4", 8, "mlx-per-row", "row-exact", "row-exact"),
                ("not exact 8 rows (MLX nvfp4, graph fp8, MLX bf16)", 8, "mlx", "graph", "mlx"),
                ("exact 16 rows", 16, "row-exact", "row-exact-cast", "row-exact"),
            ]
            // Two passes, so that load from other processes shows as a difference.
            for pass in 1 ... 2 {
                for (label, rows, nvfp4, fp8, bf16) in cases {
                    let stats = Self.step(
                        rows: rows, nvfp4: nvfp4, fp8: fp8, bf16: bf16, modules)
                    print(
                        String(
                            format: "STEP pass %d %@: median %.2f ms (min %.2f, p10 %.2f, p90 %.2f)",
                            pass, label, stats.median, stats.minimum, stats.p10, stats.p90))
                }
            }
        }

        /// A kernel read from files, for tuning without a rebuild: `<name>.metal`
        /// (body), optional `<name>.header.metal`, and `<name>.params` with
        /// `KEY=integer` lines. TGX, TGY (threadgroup), TILE (output rows per
        /// threadgroup) and MAXV (rows per dispatch) lay out the launch; every
        /// other key is a template constant. Inputs `w`, `scales`, `x`,
        /// `tensor_scale`; output `y`; templates T, V, K, N, SCALED.
        struct Candidate {
            let name: String
            let kernel: MLXFast.MLXFastKernel
            var layout: [String: Int] = ["TGX": 32, "TGY": 2, "TILE": 8, "MAXV": 8]
            var constants: [(String, Int)] = []

            init?(directory: URL, name: String) {
                let base = directory.appendingPathComponent(name)
                guard
                    let source = try? String(
                        contentsOf: base.appendingPathExtension("metal"), encoding: .utf8)
                else { return nil }
                let header =
                    (try? String(
                        contentsOf: base.appendingPathExtension("header.metal"), encoding: .utf8))
                    ?? ""
                let params =
                    (try? String(
                        contentsOf: base.appendingPathExtension("params"), encoding: .utf8)) ?? ""
                for line in params.split(separator: "\n") {
                    let parts = line.split(separator: "=").map {
                        $0.trimmingCharacters(in: .whitespaces)
                    }
                    guard parts.count == 2, let value = Int(parts[1]) else { continue }
                    if layout[parts[0]] != nil {
                        layout[parts[0]] = value
                    } else {
                        constants.append((parts[0], value))
                    }
                }
                self.name = name
                kernel = MLXFast.metalKernel(
                    name: "candidate_" + name.filter { $0.isLetter || $0.isNumber },
                    inputNames: ["w", "scales", "x", "tensor_scale"], outputNames: ["y"],
                    source: source, header: header)
            }

            func callAsFunction(_ x: MLXArray, _ weight: NVFP4, scaled: Bool) -> MLXArray {
                let rows = x.dim(0)
                let limit = layout["MAXV"]!
                if rows > limit {
                    let pieces = (rows + limit - 1) / limit
                    let step = (rows + pieces - 1) / pieces
                    return concatenated(
                        stride(from: 0, to: rows, by: step).map {
                            self(x[$0 ..< min(rows, $0 + step)], weight, scaled: scaled)
                        }, axis: 0)
                }
                let outputs = weight.scales.dim(0)
                let tiles = (outputs + layout["TILE"]! - 1) / layout["TILE"]!
                var template: [(String, any KernelTemplateArg)] = [
                    ("T", x.dtype), ("V", rows), ("K", x.dim(1)), ("N", outputs),
                    ("SCALED", scaled),
                ]
                for (key, value) in constants { template.append((key, value)) }
                return kernel(
                    [weight.words, weight.scales, x, tensorScale], template: template,
                    grid: (layout["TGX"]!, layout["TGY"]! * tiles, 1),
                    threadGroup: (layout["TGX"]!, layout["TGY"]!, 1),
                    outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
            }
        }

        static func candidates() -> [Candidate] {
            let environment = ProcessInfo.processInfo.environment
            guard let path = environment["MODELOPT_ROWEXACT_CANDIDATES"] else { return [] }
            let directory = URL(fileURLWithPath: path)
            let names =
                environment["MODELOPT_ROWEXACT_CANDIDATE_NAMES"].map {
                    $0.split(separator: ",").map(String.init)
                }
                ?? ((try? FileManager.default.contentsOfDirectory(atPath: path)) ?? [])
                .filter { $0.hasSuffix(".params") }.map { String($0.dropLast(7)) }.sorted()
            return names.compactMap { Candidate(directory: directory, name: $0) }
        }

        static func selectedShapes() -> [Shape] {
            guard let filter = ProcessInfo.processInfo.environment["MODELOPT_ROWEXACT_SHAPES"]
            else { return shapes }
            let wanted = filter.split(separator: ",").map(String.init)
            return shapes.filter { wanted.contains($0.name) }
        }

        /// Candidate kernels: row identity (against their own one-row call and
        /// MLX's), the error against a dense float32 product, and timings.
        @Test(.enabled(if: wants("candidates")))
        func candidateKernels() throws {
            let candidates = Self.candidates()
            for shape in Self.selectedShapes() where shape.counts[.nvfp4] != nil {
                let weight = Self.nvfp4(shape, seed: 21)
                let count = shape.counts[.nvfp4]!
                let tag = "\(shape.name) [\(shape.outputs), \(shape.inputs)]"
                for dtype in [DType.bfloat16, .float32] {
                    let x = Self.input(rows: 16, width: shape.inputs, seed: 11).asType(dtype)
                    eval(x)
                    let mlx = (0 ..< 16).map { Self.rowBits(Self.mlxNVFP4(x[$0 ..< $0 + 1], weight))[0] }
                    let counts = [1, 2, 3, 5, 7, 8, 11, 16]
                    let mlxSame = counts.map { Self.rowBits(Self.mlxNVFP4(x[0 ..< $0], weight)) }
                    var first: [[[UInt16]]]?
                    for candidate in candidates {
                        let alone = (0 ..< 16).map {
                            Self.rowBits(candidate(x[$0 ..< $0 + 1], weight, scaled: false))[0]
                        }
                        let together = counts.map {
                            Self.rowBits(candidate(x[0 ..< $0], weight, scaled: false))
                        }
                        func differing(_ other: (Int, Int) -> [UInt16]) -> Int {
                            counts.indices.reduce(0) { total, index in
                                total
                                    + (0 ..< counts[index]).filter {
                                        together[index][$0] != other(index, $0)
                                    }.count
                            }
                        }
                        let own = differing { _, row in alone[row] }
                        let versusMLX = differing { _, row in mlx[row] }
                        let versusSame = differing { index, row in mlxSame[index][row] }
                        let versusFirst = first.map { reference in
                            differing { index, row in reference[index][row] }
                        }
                        first = first ?? together
                        print(
                            "CAND \(candidate.name) \(tag) \(dtype): rows != own one-row "
                                + "\(own)/53, != MLX one-row \(versusMLX)/53, != MLX same call "
                                + "\(versusSame)/53, != first candidate "
                                + "\(versusFirst.map(String.init) ?? "-")/53")
                    }
                }
                let depth = shape.outputs > 100_000 ? 2 : 8
                for rows in Self.timedRows {
                    let x = Self.input(rows: rows, width: shape.inputs, seed: 11)
                    var paths: [(String, () -> MLXArray)] = [
                        ("mlx", { Self.scaled(Self.mlxNVFP4(x, weight)) })
                    ]
                    for candidate in candidates {
                        paths.append((candidate.name, { candidate(x, weight, scaled: true) }))
                    }
                    for (name, stats) in Self.interleaved(depth: depth, paths) {
                        print(
                            String(
                                format: "CAND %@ %@ x%d R=%d: median %.3f ms, min %.3f, p10 %.3f",
                                name, tag, count, rows, stats.median, stats.minimum, stats.p10))
                    }
                }
            }
        }

        /// What one 8 x 8 simdgroup multiply-accumulate computes per element,
        /// against scalar orders over k.
        @Test(.enabled(if: wants("mma")))
        func simdgroupMatrixOrder() {
            let kernel = MLXFast.metalKernel(
                name: "modelopt_mma_order_probe", inputNames: ["a", "b", "c"],
                outputNames: ["mma", "up", "down", "plain"],
                source: """
                    const uint lid = thread_index_in_simdgroup;
                    const uint g = threadgroup_position_in_grid.x;
                    const short qid = lid / 4;
                    const short fm = (qid & 4) + ((lid / 2) % 4);
                    const short fn = (qid & 2) * 2 + (lid % 2) * 2;
                    const device float* A = a + g * 64;
                    const device float* B = b + g * 64;
                    const device float* C = c + g * 64;
                    metal::simdgroup_matrix<float, 8, 8> A_mat, B_mat, C_mat, D_mat;
                    reinterpret_cast<thread float2&>(A_mat.thread_elements()) =
                        float2(A[fm * 8 + fn], A[fm * 8 + fn + 1]);
                    reinterpret_cast<thread float2&>(B_mat.thread_elements()) =
                        float2(B[fm * 8 + fn], B[fm * 8 + fn + 1]);
                    reinterpret_cast<thread float2&>(C_mat.thread_elements()) =
                        float2(C[fm * 8 + fn], C[fm * 8 + fn + 1]);
                    simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
                    const float2 d = reinterpret_cast<thread float2&>(D_mat.thread_elements());
                    for (int j = 0; j < 2; j++) {
                        const int n = fn + j;
                        const int e = g * 64 + fm * 8 + n;
                        mma[e] = d[j];
                        float u = C[fm * 8 + n];
                        float w = C[fm * 8 + n];
                        float p = C[fm * 8 + n];
                        for (int k = 0; k < 8; k++) {
                            u = fma(A[fm * 8 + k], B[k * 8 + n], u);
                            w = fma(A[fm * 8 + 7 - k], B[(7 - k) * 8 + n], w);
                            const float product = A[fm * 8 + k] * B[k * 8 + n];
                            p = p + product;
                        }
                        up[e] = u;
                        down[e] = w;
                        plain[e] = p;
                    }
                    """,
                header: "#include <metal_simdgroup_matrix>\n")
            let groups = 4096
            let a = MLXRandom.normal([groups, 64], key: MLXRandom.key(1))
            let b = MLXRandom.normal([groups, 64], key: MLXRandom.key(2))
            let c = MLXRandom.normal([groups, 64], key: MLXRandom.key(3))
            let outputs = kernel(
                [a, b, c], grid: (32 * groups, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: Array(repeating: [groups, 64], count: 4),
                outputDTypes: Array(repeating: .float32, count: 4))
            let bits = outputs.map { $0.view(dtype: .uint32).asArray(UInt32.self) }
            for (index, name) in ["fma ascending k", "fma descending k", "mul then add"].enumerated() {
                let differing = zip(bits[0], bits[index + 1]).filter { $0 != $1 }.count
                print("PROBE mma vs \(name): \(differing)/\(bits[0].count) elements differ")
            }
        }
    }
}
