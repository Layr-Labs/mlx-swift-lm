import Foundation
import MLX
import MLXNN

/// A cluster decode frame may request singleton projection arithmetic for
/// independent Qwen 3.5 rows. The default, prefill and speculative paths do not
/// opt in. This scope changes no MLX dispatcher and carries no request state.
public enum Qwen35BatchDecodeScope {
    @TaskLocal public static var enabled = false
    /// Which rows carry explicit M-RoPE positions. A mixed rectangle must
    /// preserve ordinary text's fused RoPE; equal numeric axes are not a
    /// byte-equivalent replacement for that operator on Metal. The owner
    /// supplies this transient mask beside the request-owned position slices.
    @TaskLocal public static var explicitPositionRows: [Bool]? = nil
    /// A batched target-verify rectangle `[B, S]` (S of 2...5) under
    /// `enabled`: each row's S columns keep the arithmetic of that row alone
    /// (`[1, S, H]`), its single-request verify window. Quantized matmul picks
    /// its kernel from the total row count, so one `[B·S, H]` product leaves
    /// qmv_wide for split-K QMM once B·S reaches the device's vector limit,
    /// and a near-tie can then choose another token. Below that limit the
    /// vector kernels compute every vector the same way at any count, so an
    /// affine projection takes as many rows per call as stay under it
    /// (`Qwen35QuantizedVectorLimit`): fewer launches, and qmv_wide reads
    /// the weights once per tile of up to five vectors.
    @TaskLocal public static var verifyRows = false
}

/// MLX's affine quantized matmul multiplies fewer than `limit` input vectors
/// with its vector kernels (qmv, or qmv_wide on Apple GPU family 15 and
/// later) and switches to QMM at `limit` (`get_qmv_batch_limit` in
/// mlx/backend/metal/quantized.cpp, mirrored here). Below the limit each
/// vector's reduction is the same at every vector count: qmv gives a vector
/// its own threadgroup, and qmv_wide's tile only decides which vectors share
/// a weight read. Independent verify windows can therefore share one call
/// while their total vector count stays below the limit.
public enum Qwen35QuantizedVectorLimit {
    /// The architecture MLX reads: `MLX_METAL_GPU_ARCH`, else the device's
    /// (for example `applegpu_g15d`).
    public static let architecture: String = {
        let forced = ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH"] ?? ""
        return forced.isEmpty ? GPU.deviceInfo().architecture : forced
    }()

    /// The first vector count MLX multiplies with QMM for an affine
    /// `[inputs -> outputs]` projection on `architecture`.
    public static func limit(inputs: Int, outputs: Int, architecture: String = architecture) -> Int
    {
        let code = Array(architecture.utf8)
        func digit(_ fromEnd: Int) -> Int {
            guard code.count >= 3 else { return 0 }
            let value = Int(code[code.count - fromEnd]) - 48
            return (0 ... 9).contains(value) ? value : 0
        }
        let generation = digit(3) * 10 + digit(2)
        let ultra = code.last == UInt8(ascii: "d")
        let small = inputs <= 2048 && outputs <= 2048
        let medium = inputs <= 4096 && outputs <= 4096
        switch (generation, ultra) {
        case (17..., false): return small ? 33 : medium ? 25 : 13
        case (15..., false): return small ? 13 : medium ? 15 : 13
        case (13..., true): return small ? 32 : medium ? 18 : 12
        case (13..., false): return small ? 14 : medium ? 10 : 6
        case (_, true): return small ? 32 : medium ? 18 : 12
        default: return small ? 18 : medium ? 12 : 10
        }
    }

    /// How many `[1, S, inputs]` windows of an affine `[inputs -> outputs]`
    /// projection share one call on the current default device and keep
    /// their own arithmetic: below the vector limit on the GPU; one (each
    /// window alone) on the CPU, for another quantization mode, or for MLX's
    /// separate 64/128-wide quad kernel.
    public static func sharedWindows(
        _ input: MLXArray, outputs: Int, mode: QuantizationMode
    ) -> Int {
        guard input.ndim == 3, input.dim(1) >= 1, mode == .affine,
            ![64, 128].contains(input.dim(-1)),
            Device.defaultDevice().deviceType == .gpu
        else { return 1 }
        let limit = limit(inputs: input.dim(-1), outputs: outputs)
        return max(1, (limit - 1) / input.dim(1))
    }

    /// `forward` over consecutive groups of `count` rows of `[B, S, H]`.
    public static func rows(
        _ input: MLXArray, by count: Int, _ forward: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let total = input.dim(0)
        return concatenated(
            stride(from: 0, to: total, by: max(1, count)).map {
                forward(input[$0 ..< min($0 + max(1, count), total)])
            }, axis: 0)
    }
}

/// MLX's `qmv_wide` for an affine 4-bit projection, vector for vector: each
/// vector's reduction is `qmv_wide_impl`'s (eight lanes per output row over
/// strided groups, eight-value sub-chunks dequantized as `scale * q + bias`,
/// then the same shuffle ladder), so a window's rows equal its own call's.
/// MLX tiles at most five vectors per weight read; this tiles up to eight (two
/// four-column verify windows), half the weight reads for a cohort. A thread
/// may also carry two output rows, converting each input value once for both;
/// either way every row and vector keeps its lanes, sub-chunk sums and ladder.
/// Only where MLX itself multiplies one request's window with `qmv_wide`
/// (affine, not the 64/128-wide quad kernel), on the GPU families measured
/// byte-equal and faster than MLX's own calls (`layout`).
public enum Qwen35WideQuantizedKernel {
    /// How the kernel divides a projection: output rows per thread and the
    /// vectors that share one weight read.
    public struct Layout: Equatable, Sendable {
        public let rowsPerThread: Int
        public let maximumVectors: Int

        public init(rowsPerThread: Int, maximumVectors: Int) {
            self.rowsPerThread = rowsPerThread
            self.maximumVectors = maximumVectors
        }
    }

    /// The layout for `input` on `architecture`, or nil where MLX's own calls
    /// stay. Measured against MLX's per-window calls on real Qwen27 shard
    /// shapes (two to eight four-column windows):
    /// - Apple GPU family 17 (M5 Max): one row a thread, eight vectors: up
    ///   to 20% faster with bfloat16 inputs and level with float32.
    /// - Family 15 (M3 Ultra): float32 inputs (tensor-parallel row partials)
    ///   two rows a thread, eight vectors: 1.5x-2.1x faster; one row a thread
    ///   was up to 1.6x slower there. bfloat16 inputs keep MLX's calls, which
    ///   were as fast or faster.
    /// Other families have no byte-equality measurement and keep MLX's calls.
    public static func layout(
        input: DType, architecture: String = Qwen35QuantizedVectorLimit.architecture
    ) -> Layout? {
        let code = Array(architecture.utf8)
        guard code.count >= 3 else { return nil }
        let tens = Int(code[code.count - 3]) - 48
        let ones = Int(code[code.count - 2]) - 48
        guard (0 ... 9).contains(tens), (0 ... 9).contains(ones) else { return nil }
        switch tens * 10 + ones {
        case 17: return Layout(rowsPerThread: 1, maximumVectors: 8)
        case 15: return input == .float32 ? Layout(rowsPerThread: 2, maximumVectors: 8) : nil
        default: return nil
        }
    }

    private static let header = """
        inline void qwen35_wide_dequantize4(
            const device uint8_t* w, float scale, float bias, thread float* w_local) {
          const float s = float(scale);
          const float b = float(bias);
          float sc[2] = {s, s / 16.0f};
          for (int i = 0; i < 4; i++) {
            w_local[2 * i] = static_cast<float>(sc[0] * (w[i] & 0x0f) + b);
            w_local[2 * i + 1] = static_cast<float>(sc[1] * (w[i] & 0xf0) + b);
          }
        }
        """

    /// A simdgroup holds four row groups of eight lanes; with `R` rows a
    /// thread, its rows are `base + 4j`.
    private static let source = """
        constexpr int k_lanes = 8;
        constexpr int num_simdgroups = 2;
        constexpr int results_per_simdgroup = 32 / k_lanes;
        constexpr int sub = 8;
        constexpr int bits = 4;
        const int in_vec_size = dims[0];
        const int out_vec_size = dims[1];
        const int M = dims[2];
        const uint3 tid = threadgroup_position_in_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const short k_lane = simd_lid % k_lanes;
        const short sg_row = simd_lid / k_lanes;
        const int base_row = tid.y * (results_per_simdgroup * num_simdgroups * R) +
            results_per_simdgroup * R * simd_gid + sg_row;
        const int vec0 = tid.x * V;
        const int in_vec_size_w = in_vec_size * bits / 8;
        const int in_vec_size_g = in_vec_size / GROUP;
        const device uint8_t* wrow[R];
        const device TS* srow[R];
        const device TS* brow[R];
        for (int j = 0; j < R; j++) {
          const int row = min(base_row + j * results_per_simdgroup, out_vec_size - 1);
          wrow[j] = (const device uint8_t*)w + row * in_vec_size_w;
          srow[j] = scales + row * in_vec_size_g;
          brow[j] = biases + row * in_vec_size_g;
        }
        const device TX* xv[V];
        for (int v = 0; v < V; v++) {
          xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
        }
        float result[R][V];
        for (int j = 0; j < R; j++) {
          for (int v = 0; v < V; v++) { result[j][v] = 0; }
        }
        for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
          float scale[R];
          float bias[R];
          for (int j = 0; j < R; j++) { scale[j] = srow[j][g]; bias[j] = brow[j][g]; }
        #pragma unroll
          for (int sc = 0; sc < GROUP / sub; sc++) {
            const int k0 = g * GROUP + sc * sub;
            float w_dq[R][sub];
        #pragma unroll
            for (int j = 0; j < R; j++) {
              qwen35_wide_dequantize4(wrow[j] + k0 * bits / 8, scale[j], bias[j], w_dq[j]);
            }
        #pragma unroll
            for (int v = 0; v < V; v++) {
              const device TX* xc = xv[v] + k0;
              float xf[sub];
        #pragma unroll
              for (int i = 0; i < sub; i++) { xf[i] = static_cast<float>(xc[i]); }
        #pragma unroll
              for (int j = 0; j < R; j++) {
                float acc = 0;
        #pragma unroll
                for (int i = 0; i < sub; i++) {
                  acc += xf[i] * w_dq[j][i];
                }
                result[j][v] += acc;
              }
            }
          }
        }
        for (int j = 0; j < R; j++) {
          for (int v = 0; v < V; v++) {
            result[j][v] += simd_shuffle_down(result[j][v], 4);
            result[j][v] += simd_shuffle_down(result[j][v], 2);
            result[j][v] += simd_shuffle_down(result[j][v], 1);
          }
        }
        if (k_lane == 0) {
          for (int j = 0; j < R; j++) {
            const int out_row = base_row + j * results_per_simdgroup;
            if (out_row < out_vec_size) {
              for (int v = 0; v < V; v++) {
                if (vec0 + v < M) {
                  y[(vec0 + v) * out_vec_size + out_row] = static_cast<TX>(result[j][v]);
                }
              }
            }
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_wide_qmv_affine4", inputNames: ["w", "scales", "biases", "x", "dims"],
        outputNames: ["y"], source: source, header: header)

    /// MLX multiplies one request's window with `qmv_wide` here (affine 4-bit,
    /// not the quad kernel), and this kernel reproduces that dtype pairing.
    static func supports(
        input: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) -> Bool {
        guard mode == .affine, bits == 4, [64, 128].contains(groupSize), biases != nil,
            input.ndim == 3, (2 ... 5).contains(input.dim(1)),
            Device.defaultDevice().deviceType == .gpu
        else { return false }
        let inputs = input.dim(-1)
        guard inputs % groupSize == 0, ![64, 128].contains(inputs), weight.ndim == 2,
            weight.dim(1) * 32 / bits == inputs, scales.dim(0) == weight.dim(0)
        else { return false }
        switch (input.dtype, scales.dtype) {
        case (.bfloat16, .bfloat16), (.float16, .float16), (.float32, .bfloat16),
            (.float32, .float16), (.float32, .float32):
            return true
        default:
            return false
        }
    }

    /// `[B, S, inputs]` windows through an affine 4-bit `[outputs, inputs]`
    /// weight: `[B, S, outputs]` in the input's dtype, or nil where the
    /// kernel does not apply (callers keep MLX's own calls). `layout` replaces
    /// this device's choice (checks compare every layout on one device).
    public static func project(
        _ input: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode, layout: Layout? = nil
    ) -> MLXArray? {
        guard
            supports(
                input: input, weight: weight, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits, mode: mode), let biases,
            let layout = layout ?? Self.layout(input: input.dtype),
            [1, 2].contains(layout.rowsPerThread), (1 ... 8).contains(layout.maximumVectors)
        else { return nil }
        let rows = input.dim(0)
        let columns = input.dim(1)
        let inputs = input.dim(2)
        let outputs = weight.dim(0)
        let vectors = rows * columns
        let tiles = (vectors + layout.maximumVectors - 1) / layout.maximumVectors
        let perTile = (vectors + tiles - 1) / tiles
        let block = 8 * layout.rowsPerThread
        let dims = GLM5NextKernelConstants.shared.int32s([
            Int32(inputs), Int32(outputs), Int32(vectors),
        ])
        let flat = input.reshaped([vectors, inputs])
        let result = kernel(
            [weight, scales, biases, flat, dims],
            template: [
                ("TX", flat.dtype), ("TS", scales.dtype), ("GROUP", groupSize), ("V", perTile),
                ("R", layout.rowsPerThread),
            ],
            grid: (tiles * 32, ((outputs + block - 1) / block) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[vectors, outputs]], outputDTypes: [flat.dtype])[0]
        return result.reshaped([rows, columns, outputs])
    }
}

/// A model-local projection may preserve singleton arithmetic while grouping
/// its own collective across the independent decode rows. Only Qwen's scoped
/// [B,1,H] decode path calls this seam; arbitrary Linear subclasses still use
/// their original method one row at a time.
public protocol Qwen35BatchDecodeProjecting: AnyObject {
    func qwen35BatchDecode(_ input: MLXArray) -> MLXArray
}

enum Qwen35BatchDecodeProjection {
    /// Set only by Qwen35TextModel's two ordinary CBv2 decode entry points and
    /// its captured verify forward (rectangles only under `verifyRows`).
    /// Other models sharing Qwen layers cannot inherit the cluster request.
    @TaskLocal static var active = false

    static func applies(_ input: MLXArray) -> Bool {
        active && input.ndim == 3 && (2 ... 8).contains(input.dim(0)) && columns(input.dim(1))
    }

    /// One decode column, or a scoped verify rectangle's columns.
    private static func columns(_ count: Int) -> Bool {
        count == 1 || (Qwen35BatchDecodeScope.verifyRows && (2 ... 5).contains(count))
    }

    static func withModel<T>(_ tokens: MLXArray, _ body: () throws -> T) rethrows -> T {
        let decode =
            Qwen35BatchDecodeScope.enabled && tokens.ndim == 2
            && (2 ... 8).contains(tokens.dim(0)) && columns(tokens.dim(1))
        return try $active.withValue(decode, operation: body)
    }

    static func linear(_ linear: Linear, _ input: MLXArray) -> MLXArray {
        guard applies(input) else { return linear(input) }
        if let projection = linear as? any Qwen35BatchDecodeProjecting {
            return projection.qwen35BatchDecode(input)
        }
        // A verify rectangle: every row keeps its single-request window's
        // arithmetic. An exactly typed affine QuantizedLinear takes as many
        // rows per call as stay below the vector limit; anything else runs
        // each row through the module's own forward.
        if input.dim(1) > 1 { return windows(input, linear) }
        // Subclasses can rotate inputs, add adapters, or have another numeric
        // contract. Preserve their actual forward method instead of reading
        // base-class arrays and bypassing the override.
        guard type(of: linear) == QuantizedLinear.self,
            let packed = linear as? QuantizedLinear,
            supports(
                packed.mode, bits: packed.bits,
                groupSize: packed.groupSize, input: input, scales: packed.scales)
        else { return perRow(input, linear.callAsFunction) }
        var result = project(
            input, weight: packed.weight, scales: packed.scales,
            biases: packed.biases, groupSize: packed.groupSize, bits: packed.bits)
        if let bias = packed.bias { result = result + bias }
        return result
    }

    static func embedding(_ embedding: Embedding, _ input: MLXArray) -> MLXArray {
        guard applies(input) else { return embedding.asLinear(input) }
        if input.dim(1) > 1 { return perRow(input, embedding.asLinear) }
        guard type(of: embedding) == QuantizedEmbedding.self,
            let packed = embedding as? QuantizedEmbedding,
            supports(
                packed.mode, bits: packed.bits, groupSize: packed.groupSize,
                input: input, scales: packed.scales)
        else { return perRow(input, embedding.asLinear) }
        return project(
            input, weight: packed.weight, scales: packed.scales,
            biases: packed.biases, groupSize: packed.groupSize, bits: packed.bits)
    }

    /// Keep unqualified quantizers and dtype combinations on the original
    /// module. The supported path has native-byte synthetic evidence on M3/M5.
    static func supports(
        _ mode: QuantizationMode, bits: Int, groupSize: Int,
        input: MLXArray, scales: MLXArray
    ) -> Bool {
        guard mode == .affine, [2, 3, 4, 8].contains(bits), [64, 128].contains(groupSize) else {
            return false
        }
        switch (input.dtype, scales.dtype) {
        case (.bfloat16, .bfloat16), (.float16, .float16),
            (.float32, .float16), (.float32, .bfloat16), (.float32, .float32):
            return true
        default:
            return false
        }
    }

    private static func project(
        _ input: MLXArray, weight: MLXArray, scales: MLXArray,
        biases: MLXArray?, groupSize: Int, bits: Int
    ) -> MLXArray {
        // A leading broadcast dimension keeps native matmul M=1, B=rows.
        // 2D weights would flatten B into M and select qmv_wide/QMM. These are
        // views, not duplicated weights. Native affine promotion is unchanged.
        quantizedMM(
            input, weight.expandedDimensions(axis: 0),
            scales: scales.expandedDimensions(axis: 0),
            biases: biases.map { $0.expandedDimensions(axis: 0) },
            transpose: true, groupSize: groupSize, bits: bits)
    }

    static func perRow(_ input: MLXArray, _ forward: (MLXArray) -> MLXArray) -> MLXArray {
        concatenated((0 ..< input.dim(0)).map { forward(input[$0 ..< $0 + 1]) }, axis: 0)
    }

    /// Verify windows through `linear`: shared calls only for the exact base
    /// QuantizedLinear, whose forward is MLX's quantized matmul.
    static func windows(_ input: MLXArray, _ linear: Linear) -> MLXArray {
        guard type(of: linear) == QuantizedLinear.self, let packed = linear as? QuantizedLinear
        else {
            return perRow(input, linear.callAsFunction)
        }
        if var wide = Qwen35WideQuantizedKernel.project(
            input, weight: packed.weight, scales: packed.scales, biases: packed.biases,
            groupSize: packed.groupSize, bits: packed.bits, mode: packed.mode)
        {
            if let bias = packed.bias { wide = wide + bias }
            return wide
        }
        let shared = Qwen35QuantizedVectorLimit.sharedWindows(
            input, outputs: packed.weight.dim(0), mode: packed.mode)
        guard shared > 1 else { return perRow(input, linear.callAsFunction) }
        return Qwen35QuantizedVectorLimit.rows(input, by: shared, linear.callAsFunction)
    }
}
