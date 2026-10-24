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
        let shared = Qwen35QuantizedVectorLimit.sharedWindows(
            input, outputs: packed.weight.dim(0), mode: packed.mode)
        guard shared > 1 else { return perRow(input, linear.callAsFunction) }
        return Qwen35QuantizedVectorLimit.rows(input, by: shared, linear.callAsFunction)
    }
}
