// Copyright © 2026 Eigen Labs / Darkbloom.
//
// A batch-invariant affine quantized matrix-vector product for decode,
// verify windows and batched rows alike.
//
// MLX multiplies M affine-quantized vectors with `qmv` (M = 1), `qmv_wide`
// (2 <= M < the device's vector limit) or split-K QMM (beyond it): three
// different arithmetics, so the bits a vector gets depend on how many other
// vectors share its call. A verify window's position therefore does not
// equal the plain decode step that would compute the same token, and a
// request's rows depend on its batchmates' count. This kernel fixes ONE
// arithmetic for every vector, whatever M is, wherever the vector sits in
// the call and whatever kernel layout (rows per SIMD group, SIMD groups per
// threadgroup, vectors per tile, schedule) serves it:
//
//   lane l of a row's SIMD group owns the 32-bit weight words
//   l*W ... l*W + W-1 of every step of 32*W words (W words per lane, one
//   quantization group's part); for each of its words in order, with m_j the
//   masked value q_j * 2^(bits * (j mod 16/bits)) and x_j the input widened to
//   float, either
//     scaled inputs (MLX qmv's form): xs_j = x_j * 2^-(bits * (j mod 16/bits)),
//       acc = xs_0 * m_0, then acc = fma(xs_j, m_j, acc) for j = 1...;
//       xsum = x_0 + x_1 + ... (left to right);
//       part = fma(scale, acc, part); part = fma(bias, xsum, part);
//     or dequantized weights: w_j = fma(scale * 2^-(bits * (j mod 16/bits)),
//       m_j, bias) (= scale * q_j + bias, rounded once),
//       part = fma(x_j, w_j, part) for j = 0...;
//   then out = simd_sum(part) over the 32 lanes, rounded once to the
//   output dtype.
//
// The arithmetic (the form and W) is one per process: this GPU family's
// (`defaultArithmetic`, `defaultWordsPerLane`), W capped per weight to its
// group. The source pins the order: contraction is off (`#pragma clang fp
// contract(off)`), every fused step is an explicit `fma`, and the custom
// kernel compiles in the safe math mode (no reassociation). A different
// template instantiation can therefore not round one vector differently,
// unlike an MLX kernel whose `a * b + c` the compiler may or may not fuse
// depending on the surrounding loop (seen with a token-folded qmv variant:
// `Qwen4ExpAffineQMV`). Each lane's partial depends only on its own words
// and the one vector; the SIMD reduction is the same instruction for every
// row and vector.
//
// It is NOT bit-equal to MLX's qmv: a caller that wants a window (or a
// batch) to equal plain decode must route the plain step through it too.
// Byte-parity tests compare every vector of every call against the same
// vector alone (M = 1) and across layouts, for both forms and several W.

import Foundation
import MLX
import MLXFast
import MLXNN

public enum AffineQuantizedVectors {
    /// How one call is divided on the GPU. Never changes a result.
    public struct Layout: Hashable, Sendable, CustomStringConvertible {
        /// How a SIMD group walks its rows and vectors. Every schedule
        /// performs each (row, vector) sum's operations in the same order.
        public enum Schedule: Int, Sendable, CaseIterable {
            /// Each lane widens its inputs for every vector of the tile into
            /// registers, then sweeps its rows.
            case vectorsInRegisters = 0
            /// Each lane decodes its rows' weights into registers, then
            /// widens one vector at a time.
            case rowsInRegisters = 1
            /// The threadgroup widens the tile's inputs once into threadgroup
            /// memory, shared by all its SIMD groups; lanes decode their rows
            /// and sweep the vectors.
            case sharedInputs = 2
        }

        /// Output rows each SIMD group (lane) computes.
        public let rowsPerSimdgroup: Int
        /// SIMD groups per threadgroup.
        public let simdgroups: Int
        /// Vectors a threadgroup carries per weight read (1...32); a call of
        /// more vectors takes several tiles, each reading the weights again,
        /// and a tile past the call's last vector skips the missing ones.
        public let vectorsPerTile: Int
        public let schedule: Schedule
        /// Weight values built from their bits instead of converted (the same values).
        public let magicDecode: Bool

        public init(
            rowsPerSimdgroup: Int, simdgroups: Int, vectorsPerTile: Int,
            schedule: Schedule = .vectorsInRegisters, magicDecode: Bool = false
        ) {
            self.rowsPerSimdgroup = rowsPerSimdgroup
            self.simdgroups = simdgroups
            self.vectorsPerTile = vectorsPerTile
            self.schedule = schedule
            self.magicDecode = magicDecode
        }

        public var description: String {
            "\(["a", "b", "t"][schedule.rawValue])r\(rowsPerSimdgroup)s\(simdgroups)v\(vectorsPerTile)"
                + (magicDecode ? "m" : "")
        }

        var valid: Bool {
            [1, 2, 4, 8].contains(rowsPerSimdgroup) && [1, 2, 4, 8].contains(simdgroups)
                && (1 ... Self.maximumVectorsPerTile).contains(vectorsPerTile)
        }

        /// The most vectors one weight read serves: a cohort's whole verify
        /// rectangle (eight rows of four columns) in one tile.
        public static let maximumVectorsPerTile = 32
    }

    /// Words (32 bits each) a lane takes per step: part of the arithmetic
    /// (which lane sums which words, in which order). One value serves every
    /// call of a process (`defaultWordsPerLane`, this GPU family's, fixed at
    /// process start), capped per weight so that a step's words lie in one
    /// quantization group: a function of the family and the weight's packing
    /// only, never of the vector count, the layout or the call.
    public static func wordsPerLane(bits: Int, groupSize: Int) -> Int {
        max(1, min(wordsPerLaneOverride ?? defaultWordsPerLane, groupSize * bits / 32))
    }

    /// This GPU family's words per lane (`wordsPerLane` before its cap).
    public static let defaultWordsPerLane: Int = familyWordsPerLane(family())

    /// Measured per GPU family (kernel throughput at tensor-parallel shard
    /// shapes, one to five vectors); unmeasured families take one word.
    public static func familyWordsPerLane(_ family: Int) -> Int {
        switch family {
        default: return 1
        }
    }

    /// A test or bench's own words per lane (nil: the family's). Changes
    /// results; never set by a product path.
    @TaskLocal public static var wordsPerLaneOverride: Int?

    /// How each (row, vector) sum is formed; part of the arithmetic, one per
    /// process (`defaultArithmetic`, this GPU family's, fixed at process
    /// start). Both are exact functions of the weight, the vector and the
    /// words per lane, whatever the vector count or layout.
    public enum Arithmetic: String, Sendable, CaseIterable {
        /// MLX's qmv form: inputs pre-scaled by powers of two, a word's
        /// masked values summed against them, then the group's scale times
        /// that sum plus its bias times the inputs' sum.
        case scaledInputs = "scaled"
        /// Each weight dequantized (`s * q + b`, rounded once), then one fused
        /// multiply-add per weight and vector: fewer operations per weight
        /// once several vectors share it.
        case dequantizedWeights = "dequant"
    }

    /// This GPU family's arithmetic.
    public static let defaultArithmetic: Arithmetic = familyArithmetic(family())

    /// Measured per GPU family (`throughputAgainstMLXCalls` at tensor-parallel
    /// shard shapes, one, four and five vectors, both arithmetics at one, two,
    /// four and eight words per lane): the M3 Ultra (15) and the M5 Max (17)
    /// both run the dequantized weights fastest, at one word per lane; other
    /// families take the scaled inputs (MLX qmv's form).
    public static func familyArithmetic(_ family: Int) -> Arithmetic {
        switch family {
        case 15, 17: return .dequantizedWeights
        default: return .scaledInputs
        }
    }

    /// A test or bench's own arithmetic (nil: the family's). Changes results;
    /// never set by a product path.
    @TaskLocal public static var arithmeticOverride: Arithmetic?

    /// The arithmetic of this call's process (or a test's override).
    public static var arithmetic: Arithmetic { arithmeticOverride ?? defaultArithmetic }

    /// Whether a dense MLP's gate, up and SwiGLU take the fused kernel
    /// (`projectSwiGLU`) for `vectors` vectors on this GPU family. It has the
    /// bits of the two projections and the SwiGLU, so this is a choice of
    /// speed only, measured per family.
    public static func fusesSwiGLU(vectors: Int) -> Bool {
        swigluOverride ?? familyFusesSwiGLU(family(), vectors: vectors)
    }

    /// Measured on the 27B tensor-parallel shards (M3 Ultra at 60%, S=4
    /// verify window): the fused kernel at four vectors cost the window 15 ms
    /// (39.3 ms against 24.3 ms with the two projections), while one vector
    /// gains a little. Every family keeps it for single vectors only.
    static func familyFusesSwiGLU(_ family: Int, vectors: Int) -> Bool {
        vectors == 1
    }

    /// A test or bench's own choice (nil: the family's); never set by a
    /// product path.
    @TaskLocal public static var swigluOverride: Bool?

    /// Layouts by vector count: a count without an entry takes the entry of
    /// the smallest listed count above it, else of the largest.
    public struct LayoutTable: Sendable, CustomStringConvertible {
        public let entries: [Entry]
        public struct Entry: Sendable {
            public let vectors: Int
            public let layout: Layout
        }

        /// `1=a:8:4:1/4=b:8:4:4` (or `,`-separated); nil if no entry parses.
        public init?(parsing text: String) {
            let entries = text.split(whereSeparator: { $0 == "," || $0 == "/" }).compactMap {
                item -> Entry? in
                let parts = item.split(separator: "=", maxSplits: 1).map(String.init)
                guard parts.count == 2, let vectors = Int(parts[0]), vectors >= 1,
                    let layout = AffineQuantizedVectors.parseLayout(parts[1])
                else { return nil }
                return Entry(vectors: vectors, layout: layout)
            }
            guard !entries.isEmpty else { return nil }
            self.entries = entries.sorted { $0.vectors < $1.vectors }
        }

        func layout(vectors: Int) -> Layout {
            (entries.first { $0.vectors == vectors } ?? entries.first { $0.vectors > vectors }
                ?? entries.last!).layout
        }

        public var description: String {
            entries.map { "\($0.vectors)=\($0.layout)" }.joined(separator: ",")
        }
    }

    /// A test or bench's own layouts (nil: the family's). Layouts never
    /// change a result; never set by a product path.
    @TaskLocal public static var layoutsOverride: LayoutTable?

    /// The measured layouts per GPU family (the fastest at one, four and five
    /// vectors; past five, tiles of four vectors in registers). Families
    /// without an entry take the generic rule in `layout`.
    ///
    /// Multi-vector tiles for six or more vectors (`b:4:4:8`, `b:2:4:16`) are
    /// supported (`maximumVectorsPerTile`, tests at tiles 1-32) but not
    /// selected: on two M5 Ultra 256 GB (TB5) serving the 27B tensor-parallel
    /// pair they lowered the four-request aggregate by 19 % against these tiles
    /// (73.1 against 90.1 tok/s, identical outputs). A family takes them again
    /// when a pair measures a gain.
    static let familyLayouts: [Int: LayoutTable] = [
        15: LayoutTable(parsing: "1=a:4:4:1/4=a:2:4:4/5=a:2:4:5/6=a:2:4:4")!,
        17: LayoutTable(parsing: "1=a:4:4:1/4=a:2:4:4/5=a:2:4:5/6=a:2:4:4")!,
    ]

    /// A layout written `schedule:rows:simdgroups:vectors`, the schedule
    /// `a` (vectors in registers), `b` (rows in registers) or `t` (shared
    /// inputs), with a trailing `m` for the bit-built decode: `b:8:4:4`.
    public static func parseLayout(_ text: String) -> Layout? {
        let fields = text.split(separator: ":").map(String.init)
        guard fields.count == 4, ["a", "b", "t", "am", "bm", "tm"].contains(fields[0]) else {
            return nil
        }
        let schedules: [String: Layout.Schedule] = [
            "a": .vectorsInRegisters, "b": .rowsInRegisters, "t": .sharedInputs,
        ]
        guard let schedule = schedules[String(fields[0].prefix(1))] else { return nil }
        let numbers = fields.dropFirst().compactMap { Int($0) }
        guard numbers.count == 3 else { return nil }
        let layout = Layout(
            rowsPerSimdgroup: numbers[0], simdgroups: numbers[1], vectorsPerTile: numbers[2],
            schedule: schedule, magicDecode: fields[0].count == 2)
        return layout.valid ? layout : nil
    }

    /// The resolved choice for weights of `bits`/`groupSize` (`[inputs ->
    /// many]` products, before small-output spreading), for a load's log:
    /// `family=17 words_per_lane=1 layouts=1=ar8s4v1,2=...`.
    public static func resolvedDescription(bits: Int, groupSize: Int, inputs: Int) -> String {
        let layouts = describedCounts.map {
            "\($0)=\(layout(vectors: $0, inputs: inputs, outputs: 1 << 20, bits: bits))"
        }
        let swiglu = describedCounts.map { fusesSwiGLU(vectors: $0) ? "1" : "0" }.joined()
        return "family=\(family()) arch=\(architecture) arithmetic=\(arithmetic.rawValue)"
            + " words_per_lane=\(wordsPerLane(bits: bits, groupSize: groupSize))"
            + " layouts=" + layouts.joined(separator: ",") + " swiglu_fused=\(swiglu)"
    }

    /// The vector counts a load's line describes: a plain step and a window
    /// (one to eight rows, four or five columns) up to an eight-row cohort's
    /// four-column window.
    static let describedCounts = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]

    /// The device's GPU architecture (for example `applegpu_g15d`), read once.
    /// Unlike MLX's own kernel choice, no environment variable overrides it:
    /// the arithmetic (`defaultWordsPerLane`) follows the actual device.
    public static let architecture: String = GPU.deviceInfo().architecture

    /// The GPU family of an architecture string (`applegpu_g17s` is 17), 0 if unknown.
    public static func family(_ architecture: String = Self.architecture) -> Int {
        let code = Array(architecture.utf8)
        guard code.count >= 3 else { return 0 }
        let tens = Int(code[code.count - 3]) - 48
        let ones = Int(code[code.count - 2]) - 48
        guard (0 ... 9).contains(tens), (0 ... 9).contains(ones) else { return 0 }
        return tens * 10 + ones
    }

    /// The layout for `vectors` vectors of an `[inputs -> outputs]` product
    /// on `architecture`. A layout only decides which threads compute what;
    /// every layout gives the same bits.
    public static func layout(
        vectors: Int, inputs: Int, outputs: Int, bits: Int,
        architecture: String = Self.architecture
    ) -> Layout {
        if let table = layoutsOverride ?? familyLayouts[family(architecture)] {
            return spread(table.layout(vectors: vectors), outputs: outputs)
        }
        // Up to five vectors share a weight read in registers; more take
        // tiles of four. One SIMD group's rows share each input conversion;
        // more rows a lane amortize it further at the cost of registers.
        let tile = vectors <= 5 ? max(1, vectors) : 4
        let rows = tile <= 2 ? 8 : (tile <= 4 ? 4 : 2)
        _ = family(architecture)
        return spread(
            Layout(rowsPerSimdgroup: rows, simdgroups: 4, vectorsPerTile: tile), outputs: outputs)
    }

    /// Fewer rows a SIMD group for a projection with few outputs (key and
    /// value projections, small shards), so its rows still spread over at
    /// least 256 SIMD groups instead of a few long-running ones.
    static func spread(_ layout: Layout, outputs: Int) -> Layout {
        var rows = layout.rowsPerSimdgroup
        while rows > 1, (outputs + rows - 1) / rows < 256 { rows /= 2 }
        guard rows != layout.rowsPerSimdgroup else { return layout }
        return Layout(
            rowsPerSimdgroup: rows, simdgroups: layout.simdgroups,
            vectorsPerTile: layout.vectorsPerTile,
            schedule: layout.schedule, magicDecode: layout.magicDecode)
    }

    /// Whether this kernel computes `[..., K] x [N, K]^T` for these tensors
    /// on the current default device: affine mode, 2/4/8 bits, group 32, 64
    /// or 128, K a multiple of the group, a float input (bf16, f16 or f32)
    /// with bf16/f16/f32 scales and biases, on the GPU.
    public static func supports(
        input: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) -> Bool {
        guard mode == .affine, [2, 4, 8].contains(bits), [32, 64, 128].contains(groupSize),
            let biases, input.ndim >= 1, weight.ndim == 2, weight.dtype == .uint32,
            Device.defaultDevice().deviceType == .gpu
        else { return false }
        let inputs = input.dim(-1)
        guard inputs > 0, inputs % groupSize == 0, weight.dim(1) * 32 / bits == inputs,
            scales.ndim == 2, scales.dim(0) == weight.dim(0), scales.dim(1) == inputs / groupSize,
            biases.shape == scales.shape, biases.dtype == scales.dtype
        else { return false }
        guard [.bfloat16, .float16, .float32].contains(input.dtype),
            [.bfloat16, .float16, .float32].contains(scales.dtype)
        else { return false }
        // A half-precision input takes scales of its own type, as MLX requires.
        return input.dtype == .float32 || scales.dtype == input.dtype
    }

    /// `x [..., K]` through the affine `[N, K]` weight: `[..., N]` in
    /// `outputDType` (default the input's dtype), or nil where the kernel
    /// does not apply. Every vector's bits are independent of the other
    /// vectors in `x` and of `layout` (which replaces the device's choice).
    public static func project(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode = .affine,
        outputDType: DType? = nil, layout: Layout? = nil
    ) -> MLXArray? {
        guard
            supports(
                input: x, weight: weight, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits, mode: mode), let biases
        else { return nil }
        let inputs = x.dim(-1)
        let outputs = weight.dim(0)
        let vectors = x.size / inputs
        guard vectors >= 1 else { return nil }
        let chosen =
            layout
            ?? Self.layout(vectors: vectors, inputs: inputs, outputs: outputs, bits: bits)
        guard chosen.valid else { return nil }
        let output = outputDType ?? x.dtype
        guard [.bfloat16, .float16, .float32].contains(output) else { return nil }
        let flat = x.reshaped([vectors, inputs])
        let perLane = wordsPerLane(bits: bits, groupSize: groupSize)
        let dequantized = arithmetic == .dequantizedWeights
        if chosen.schedule == .sharedInputs {
            // Threadgroup memory: the tile's widened inputs of one step (and,
            // scaled, their sums).
            let floats = chosen.vectorsPerTile * 32 * perLane * (32 / bits + (dequantized ? 0 : 1))
            guard floats * 4 <= 28 * 1024 else { return nil }
        }
        let tiles = (vectors + chosen.vectorsPerTile - 1) / chosen.vectorsPerTile
        let rowsPerGroup = chosen.rowsPerSimdgroup * chosen.simdgroups
        let rowBlocks = (outputs + rowsPerGroup - 1) / rowsPerGroup
        let y = kernel(
            [weight, scales, biases, flat, Constants.shared.count(vectors)],
            template: [
                ("XT", flat.dtype), ("ST", scales.dtype), ("YT", output),
                ("BITS", bits), ("GS", groupSize), ("K_SIZE", inputs), ("N_SIZE", outputs),
                ("R", chosen.rowsPerSimdgroup), ("NSG", chosen.simdgroups),
                ("VT", chosen.vectorsPerTile), ("ALG", chosen.schedule.rawValue), ("WPL", perLane),
                ("MAGIC", chosen.magicDecode), ("DQ", dequantized),
            ],
            grid: (tiles * 32, rowBlocks * chosen.simdgroups, 1),
            threadGroup: (32, chosen.simdgroups, 1),
            outputShapes: [[vectors, outputs]], outputDTypes: [output])[0]
        return y.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    /// `x` through `linear`'s weights (no `bias` term is added: callers add
    /// a linear bias themselves); nil where the kernel does not apply.
    public static func project(
        _ x: MLXArray, _ linear: QuantizedLinear, outputDType: DType? = nil,
        layout: Layout? = nil
    ) -> MLXArray? {
        project(
            x, weight: linear.weight, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode,
            outputDType: outputDType, layout: layout)
    }

    /// `silu(x W_gate^T) * (x W_up^T)` for two affine weights of the same
    /// shape and quantization, in one launch: each product is this kernel's
    /// (the bits `project` gives it, rounded to the activation dtype), then
    /// MLX's compiled SwiGLU on them (`x * sigmoid(x)`, then the product with
    /// up, each op at the activation dtype). The same values as two `project`
    /// calls and the SwiGLU, without writing or rereading the two products.
    /// Schedules: registers (`vectorsInRegisters`, `rowsInRegisters`) only.
    public static func projectSwiGLU(
        _ x: MLXArray, gate: QuantizedLinear, up: QuantizedLinear, layout: Layout? = nil
    ) -> MLXArray? {
        guard gate.bias == nil, up.bias == nil, gate.bits == up.bits,
            gate.groupSize == up.groupSize,
            gate.mode == up.mode, gate.weight.shape == up.weight.shape,
            gate.scales.dtype == up.scales.dtype,
            supports(
                input: x, weight: gate.weight, scales: gate.scales, biases: gate.biases,
                groupSize: gate.groupSize, bits: gate.bits, mode: gate.mode),
            supports(
                input: x, weight: up.weight, scales: up.scales, biases: up.biases,
                groupSize: up.groupSize, bits: up.bits, mode: up.mode),
            [.bfloat16, .float16, .float32].contains(x.dtype)
        else { return nil }
        let inputs = x.dim(-1)
        let outputs = gate.weight.dim(0)
        let bits = gate.bits
        let vectors = x.size / inputs
        guard vectors >= 1 else { return nil }
        let base =
            layout ?? Self.layout(vectors: vectors, inputs: inputs, outputs: outputs, bits: bits)
        // Two weights a row: half the rows a SIMD group would otherwise take.
        let chosen =
            base.schedule == .sharedInputs
            ? Layout(
                rowsPerSimdgroup: max(1, base.rowsPerSimdgroup / 2), simdgroups: base.simdgroups,
                vectorsPerTile: base.vectorsPerTile, schedule: .rowsInRegisters,
                magicDecode: base.magicDecode)
            : (layout == nil
                ? Layout(
                    rowsPerSimdgroup: max(1, base.rowsPerSimdgroup / 2),
                    simdgroups: base.simdgroups,
                    vectorsPerTile: base.vectorsPerTile, schedule: base.schedule,
                    magicDecode: base.magicDecode) : base)
        guard chosen.valid else { return nil }
        let flat = x.reshaped([vectors, inputs])
        let tiles = (vectors + chosen.vectorsPerTile - 1) / chosen.vectorsPerTile
        let rowsPerGroup = chosen.rowsPerSimdgroup * chosen.simdgroups
        let rowBlocks = (outputs + rowsPerGroup - 1) / rowsPerGroup
        let y = swigluKernel(
            [
                gate.weight, gate.scales, gate.biases!, up.weight, up.scales, up.biases!, flat,
                Constants.shared.count(vectors),
            ],
            template: [
                ("XT", flat.dtype), ("ST", gate.scales.dtype), ("YT", x.dtype),
                ("BITS", bits), ("GS", gate.groupSize), ("K_SIZE", inputs), ("N_SIZE", outputs),
                ("R", chosen.rowsPerSimdgroup), ("NSG", chosen.simdgroups),
                ("VT", chosen.vectorsPerTile), ("ALG", chosen.schedule.rawValue),
                ("WPL", wordsPerLane(bits: bits, groupSize: gate.groupSize)),
                ("MAGIC", chosen.magicDecode),
                ("DQ", arithmetic == .dequantizedWeights),
            ],
            grid: (tiles * 32, rowBlocks * chosen.simdgroups, 1),
            threadGroup: (32, chosen.simdgroups, 1),
            outputShapes: [[vectors, outputs]], outputDTypes: [x.dtype])[0]
        return y.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    /// Cached `[1]` int32 vector counts (a kernel input, not a template:
    /// one compiled kernel serves every count).
    final class Constants: @unchecked Sendable {
        static let shared = Constants()
        private let lock = NSLock()
        private var counts: [Int: MLXArray] = [:]

        func count(_ value: Int) -> MLXArray {
            lock.lock()
            defer { lock.unlock() }
            if let cached = counts[value] { return cached }
            let made = MLXArray([Int32(value)])
            eval(made)
            if counts.count < 512 { counts[value] = made }
            return made
        }
    }

    private static let swigluKernel = MLXFast.metalKernel(
        name: "affine_quantized_vectors_swiglu",
        inputNames: ["w", "scales", "biases", "w2", "scales2", "biases2", "x", "count"],
        outputNames: ["y"], source: swigluSource, header: header + swigluHeader)

    // MLX's `Sigmoid` functor (mlx/backend/metal/kernels/unary_ops.h).
    private static let swigluHeader = """

        struct AqvSigmoid {
          template <typename U>
          U operator()(U x) thread {
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
          }
        };
        """

    private static let kernel = MLXFast.metalKernel(
        name: "affine_quantized_vectors", inputNames: ["w", "scales", "biases", "x", "count"],
        outputNames: ["y"], source: source, header: header)

    private static let header = """
        // Input scale of value j of a 16-bit half: the mask leaves value j at
        // bit BITS * j, so x * 2^-(BITS * j) times the masked half is x * q.
        template <int BITS>
        constexpr float aqv_input_scale(int j) {
          return 1.0f / float(1u << (BITS * (j % (16 / BITS))));
        }

        // A masked half-word (below 2^16) as float: a conversion, or the same
        // value built from its bits (2^23 + m, less 2^23: exact).
        template <bool MAGIC>
        inline __attribute__((always_inline)) float aqv_decode(uint masked) {
          if (MAGIC) {
            return as_type<float>(0x4B000000u | masked) - 8388608.0f;
          }
          return static_cast<float>(masked);
        }

        // The 32 / BITS values of one weight word, value j at q[j]: its masked
        // (unshifted) value m_j = q_j * 2^(BITS * (j mod 16/BITS)), or with DQ
        // the dequantized weight fma(s * 2^-(BITS * (j mod 16/BITS)), m_j, b),
        // which is s * q_j + b rounded once (the power-of-two scalings are
        // exact).
        template <int BITS, bool MAGIC, bool DQ>
        inline __attribute__((always_inline)) void aqv_word(uint packed, float s, float b, thread float* q) {
        #pragma clang fp contract(off)
          constexpr int HV = 16 / BITS;
          constexpr uint MASK = (1u << BITS) - 1u;
          float sk[HV];
          #pragma unroll
          for (int j = 0; j < HV; ++j) { sk[j] = DQ ? s * aqv_input_scale<BITS>(j) : 0.0f; }
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint half_word = (h == 0) ? (packed & 0xffffu) : (packed >> 16);
            #pragma unroll
            for (int j = 0; j < HV; ++j) {
              const float m = aqv_decode<MAGIC>(half_word & (MASK << (BITS * j)));
              q[h * HV + j] = DQ ? fma(sk[j], m, b) : m;
            }
          }
        }

        // One word's share of one (row, vector) sum, in the arithmetic's
        // fixed order. Scaled inputs: xs_j = x_j * 2^-(BITS * (j mod 16/BITS)),
        // acc = xs_0 * m_0 then fma(xs_j, m_j, acc); part += s * acc, then
        // b * xsum (each an fma). DQ: part = fma(x_j, w_j, part) for j in order.
        template <int WV, bool DQ>
        inline __attribute__((always_inline)) void aqv_accumulate(
            thread const float* xs, thread const float* q, float xsum, float s, float b, thread float& part) {
        #pragma clang fp contract(off)
          if (DQ) {
            #pragma unroll
            for (int j = 0; j < WV; ++j) { part = fma(xs[j], q[j], part); }
          } else {
            float acc = xs[0] * q[0];
            #pragma unroll
            for (int j = 1; j < WV; ++j) { acc = fma(xs[j], q[j], acc); }
            part = fma(s, acc, part);
            part = fma(b, xsum, part);
          }
        }
        """

    // Threadgroup arrays are declared at the kernel's outermost scope (Metal
    // allows them nowhere else); the arithmetic's block follows.
    private static let source = """
        // Used by the shared-input schedule only.
        threadgroup float xs_tg[ALG == 2 ? VT * (32 / BITS) * 32 * WPL : 1];
        threadgroup float xsum_tg[ALG == 2 && !DQ ? VT * 32 * WPL : 1];
        {
        #pragma clang fp contract(off)
        constexpr int WV = 32 / BITS;          // values per 32-bit word
        constexpr int KW = K_SIZE / WV;        // words per weight row
        constexpr int KG = K_SIZE / GS;        // groups per weight row
        const int M = int(count[0]);
        const int lane = int(thread_index_in_simdgroup);
        const int sgi = int(simdgroup_index_in_threadgroup);
        const int row0 = (int(threadgroup_position_in_grid.y) * NSG + sgi) * R;
        const int vec0 = int(threadgroup_position_in_grid.x) * VT;
        // The tile's vectors in this call, the same for every thread of the
        // threadgroup (so each `v < live` branch is uniform): a last tile
        // past the call's end skips its missing vectors. Vector v of the tile
        // starts v * K_SIZE (a constant once unrolled) after the first.
        const int live = min(VT, M - vec0);
        auto xt = x + vec0 * K_SIZE;

        float part[R][VT];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          #pragma unroll
          for (int v = 0; v < VT; ++v) { part[r][v] = 0.0f; }
        }
        int wbase[R];
        int gbase[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          const int row = min(row0 + r, N_SIZE - 1);
          wbase[r] = row * KW;
          gbase[r] = row * KG;
        }

        // Lane `lane` owns the words lane * WPL + i (i < WPL) of every step of
        // 32 * WPL words: one quantization group (WPL * WV divides the group).
        constexpr int STEP_WORDS = 32 * WPL;
        if (ALG == 0) {
          // Inputs of every vector in registers; rows inner.
          for (int base = lane * WPL; base < KW; base += STEP_WORDS) {
            const int g = (base * WV) / GS;
            float s[R];
            float b[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
              s[r] = static_cast<float>(scales[gbase[r] + g]);
              b[r] = static_cast<float>(biases[gbase[r] + g]);
            }
            #pragma unroll
            for (int wi = 0; wi < WPL; ++wi) {
              const int word = base + wi;
              const int k0 = word * WV;
              float xs[VT][WV];
              float xsum[VT];
              #pragma unroll
              for (int v = 0; v < VT; ++v) {
                if (v < live) {
                  auto xp = xt + v * K_SIZE + k0;
                  float total = 0.0f;
                  #pragma unroll
                  for (int j = 0; j < WV; ++j) {
                    const float value = static_cast<float>(xp[j]);
                    total = (j == 0) ? value : total + value;
                    xs[v][j] = DQ ? value : value * aqv_input_scale<BITS>(j);
                  }
                  xsum[v] = total;
                }
              }
              #pragma unroll
              for (int r = 0; r < R; ++r) {
                float q[WV];
                aqv_word<BITS, MAGIC, DQ>(w[wbase[r] + word], s[r], b[r], q);
                #pragma unroll
                for (int v = 0; v < VT; ++v) {
                  if (v < live) { aqv_accumulate<WV, DQ>(xs[v], q, xsum[v], s[r], b[r], part[r][v]); }
                }
              }
            }
          }
        } else if (ALG == 1) {
          // Decoded weights of every row in registers; vectors outer.
          for (int base = lane * WPL; base < KW; base += STEP_WORDS) {
            const int g = (base * WV) / GS;
            float s[R];
            float b[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
              s[r] = static_cast<float>(scales[gbase[r] + g]);
              b[r] = static_cast<float>(biases[gbase[r] + g]);
            }
            #pragma unroll
            for (int wi = 0; wi < WPL; ++wi) {
              const int word = base + wi;
              const int k0 = word * WV;
              float q[R][WV];
              #pragma unroll
              for (int r = 0; r < R; ++r) {
                aqv_word<BITS, MAGIC, DQ>(w[wbase[r] + word], s[r], b[r], q[r]);
              }
              #pragma unroll
              for (int v = 0; v < VT; ++v) {
                if (v < live) {
                  auto xp = xt + v * K_SIZE + k0;
                  float xs[WV];
                  float total = 0.0f;
                  #pragma unroll
                  for (int j = 0; j < WV; ++j) {
                    const float value = static_cast<float>(xp[j]);
                    total = (j == 0) ? value : total + value;
                    xs[j] = DQ ? value : value * aqv_input_scale<BITS>(j);
                  }
                  #pragma unroll
                  for (int r = 0; r < R; ++r) {
                    aqv_accumulate<WV, DQ>(xs, q[r], total, s[r], b[r], part[r][v]);
                  }
                }
              }
            }
          }
        } else {
          // The threadgroup widens each step's inputs once (shared by its SIMD
          // groups) into threadgroup memory; lanes decode rows, vectors outer.
          const int flat = sgi * 32 + lane;
          for (int step = 0; step * STEP_WORDS < KW; ++step) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int item = flat; item < live * STEP_WORDS; item += NSG * 32) {
              const int v = item / STEP_WORDS;
              const int l = item % STEP_WORDS;
              const int word = step * STEP_WORDS + l;
              if (word < KW) {
                auto xp = xt + v * K_SIZE + word * WV;
                float total = 0.0f;
                #pragma unroll
                for (int j = 0; j < WV; ++j) {
                  const float value = static_cast<float>(xp[j]);
                  total = (j == 0) ? value : total + value;
                  xs_tg[(v * WV + j) * STEP_WORDS + l] = DQ ? value : value * aqv_input_scale<BITS>(j);
                }
                if (!DQ) { xsum_tg[v * STEP_WORDS + l] = total; }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int base = step * STEP_WORDS + lane * WPL;
            if (base < KW) {
              const int g = (base * WV) / GS;
              float s[R];
              float b[R];
              #pragma unroll
              for (int r = 0; r < R; ++r) {
                s[r] = static_cast<float>(scales[gbase[r] + g]);
                b[r] = static_cast<float>(biases[gbase[r] + g]);
              }
              #pragma unroll
              for (int wi = 0; wi < WPL; ++wi) {
                const int word = base + wi;
                const int l = lane * WPL + wi;
                float q[R][WV];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                  aqv_word<BITS, MAGIC, DQ>(w[wbase[r] + word], s[r], b[r], q[r]);
                }
                #pragma unroll
                for (int v = 0; v < VT; ++v) {
                  if (v < live) {
                    float xs[WV];
                    #pragma unroll
                    for (int j = 0; j < WV; ++j) { xs[j] = xs_tg[(v * WV + j) * STEP_WORDS + l]; }
                    const float total = DQ ? 0.0f : xsum_tg[v * STEP_WORDS + l];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                      aqv_accumulate<WV, DQ>(xs, q[r], total, s[r], b[r], part[r][v]);
                    }
                  }
                }
              }
            }
          }
        }

        #pragma unroll
        for (int r = 0; r < R; ++r) {
          #pragma unroll
          for (int v = 0; v < VT; ++v) {
            if (v < live) {
              const float total = simd_sum(part[r][v]);
              if (lane == 0 && row0 + r < N_SIZE) {
                y[(vec0 + v) * N_SIZE + row0 + r] = static_cast<YT>(total);
              }
            }
          }
        }
        }
        """

    // The main kernel's arithmetic for two weights of one shape (rows of gate
    // and up of the same index in one SIMD group), then the SwiGLU on the
    // rounded products.
    private static let swigluSource = """
        {
        #pragma clang fp contract(off)
        constexpr int WV = 32 / BITS;
        constexpr int KW = K_SIZE / WV;
        constexpr int KG = K_SIZE / GS;
        constexpr int STEP_WORDS = 32 * WPL;
        const int M = int(count[0]);
        const int lane = int(thread_index_in_simdgroup);
        const int sgi = int(simdgroup_index_in_threadgroup);
        const int row0 = (int(threadgroup_position_in_grid.y) * NSG + sgi) * R;
        const int vec0 = int(threadgroup_position_in_grid.x) * VT;

        float pg[R][VT];
        float pu[R][VT];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          #pragma unroll
          for (int v = 0; v < VT; ++v) { pg[r][v] = 0.0f; pu[r][v] = 0.0f; }
        }
        int xbase[VT];
        #pragma unroll
        for (int v = 0; v < VT; ++v) { xbase[v] = min(vec0 + v, M - 1) * K_SIZE; }
        int wbase[R];
        int gbase[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          const int row = min(row0 + r, N_SIZE - 1);
          wbase[r] = row * KW;
          gbase[r] = row * KG;
        }

        for (int base = lane * WPL; base < KW; base += STEP_WORDS) {
          const int g = (base * WV) / GS;
          float sg[R];
          float bg[R];
          float su[R];
          float bu[R];
          #pragma unroll
          for (int r = 0; r < R; ++r) {
            sg[r] = static_cast<float>(scales[gbase[r] + g]);
            bg[r] = static_cast<float>(biases[gbase[r] + g]);
            su[r] = static_cast<float>(scales2[gbase[r] + g]);
            bu[r] = static_cast<float>(biases2[gbase[r] + g]);
          }
          #pragma unroll
          for (int wi = 0; wi < WPL; ++wi) {
            const int word = base + wi;
            const int k0 = word * WV;
            if (ALG == 0) {
              float xs[VT][WV];
              float xsum[VT];
              #pragma unroll
              for (int v = 0; v < VT; ++v) {
                auto xp = x + xbase[v] + k0;
                float total = 0.0f;
                #pragma unroll
                for (int j = 0; j < WV; ++j) {
                  const float value = static_cast<float>(xp[j]);
                  total = (j == 0) ? value : total + value;
                  xs[v][j] = DQ ? value : value * aqv_input_scale<BITS>(j);
                }
                xsum[v] = total;
              }
              #pragma unroll
              for (int r = 0; r < R; ++r) {
                float qg[WV];
                float qu[WV];
                aqv_word<BITS, MAGIC, DQ>(w[wbase[r] + word], sg[r], bg[r], qg);
                aqv_word<BITS, MAGIC, DQ>(w2[wbase[r] + word], su[r], bu[r], qu);
                #pragma unroll
                for (int v = 0; v < VT; ++v) {
                  aqv_accumulate<WV, DQ>(xs[v], qg, xsum[v], sg[r], bg[r], pg[r][v]);
                  aqv_accumulate<WV, DQ>(xs[v], qu, xsum[v], su[r], bu[r], pu[r][v]);
                }
              }
            } else {
              float qg[R][WV];
              float qu[R][WV];
              #pragma unroll
              for (int r = 0; r < R; ++r) {
                aqv_word<BITS, MAGIC, DQ>(w[wbase[r] + word], sg[r], bg[r], qg[r]);
                aqv_word<BITS, MAGIC, DQ>(w2[wbase[r] + word], su[r], bu[r], qu[r]);
              }
              #pragma unroll
              for (int v = 0; v < VT; ++v) {
                auto xp = x + xbase[v] + k0;
                float xs[WV];
                float total = 0.0f;
                #pragma unroll
                for (int j = 0; j < WV; ++j) {
                  const float value = static_cast<float>(xp[j]);
                  total = (j == 0) ? value : total + value;
                  xs[j] = DQ ? value : value * aqv_input_scale<BITS>(j);
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                  aqv_accumulate<WV, DQ>(xs, qg[r], total, sg[r], bg[r], pg[r][v]);
                  aqv_accumulate<WV, DQ>(xs, qu[r], total, su[r], bu[r], pu[r][v]);
                }
              }
            }
          }
        }

        #pragma unroll
        for (int r = 0; r < R; ++r) {
          #pragma unroll
          for (int v = 0; v < VT; ++v) {
            const float tg = simd_sum(pg[r][v]);
            const float tu = simd_sum(pu[r][v]);
            if (lane == 0 && row0 + r < N_SIZE && vec0 + v < M) {
              const YT gq = static_cast<YT>(tg);
              const YT uq = static_cast<YT>(tu);
              const YT sig = AqvSigmoid()(gq);
              const YT gs = gq * sig;
              y[(vec0 + v) * N_SIZE + row0 + r] = gs * uq;
            }
          }
        }
        }
        """
}
