// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// A 4-bit affine group-64 projection that multiplies 4...16 rows on the
/// Neural Accelerators in one pass over its weights. MLX's vector kernel
/// rereads the weights for every five rows and its matrix kernel pads to 32,
/// so decode batches and MTP verify windows paid two to three weight reads
/// per step. Weights are dequantized to the activation type, as in MLX's
/// matrix kernels, and products accumulate in FP32.
final class Qwen35SmallBatchLinear: QuantizedLinear {
    static let rows = 4 ... 16

    /// Neural Accelerators and an OS whose Metal compiler builds tensor-op
    /// kernels at first use (see `ModelOptNVFP4NAXExperts`).
    /// `DARKBLOOM_QWEN_SMALL_BATCH=0` keeps the stock layers and one speculating row.
    static let available =
        ModelOptNVFP4NAXExperts.naxAvailable && ModelOptNVFP4NAXExperts.runtimeCompilerSupported
        && ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_SMALL_BATCH"] != "0"

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let k = x.dim(-1)
        let n = weight.dim(0)
        let m = x.size / k
        guard Self.rows.contains(m), k % 1024 == 0, n % 32 == 0, let biases,
            x.dtype == scales.dtype, Device.defaultDevice().deviceType == .gpu
        else { return super.callAsFunction(x) }
        let y = Self.kernel(
            [x.reshaped(m, k), weight, scales, biases], template: [("T", x.dtype)],
            grid: (128 * n / 32, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [x.dtype])[0]
        return y.reshaped(Array(x.shape.dropLast()) + [n])
    }

    /// The small-batch form of a stock 4-bit group-64 layer on a GPU with
    /// Neural Accelerators, sharing its arrays; nil otherwise.
    static func make(from layer: QuantizedLinear) -> Qwen35SmallBatchLinear? {
        guard available,
            ObjectIdentifier(type(of: layer)) == ObjectIdentifier(QuantizedLinear.self),
            layer.mode == .affine, layer.bits == 4, layer.groupSize == 64,
            layer.bias == nil, let biases = layer.biases
        else { return nil }
        let swapped = Qwen35SmallBatchLinear(
            weight: layer.weight, scales: layer.scales, biases: biases, groupSize: 64, bits: 4)
        swapped.freeze()
        return swapped
    }

    /// Swaps the model's eligible layers in place. The GatedDeltaNet input
    /// projections stay stock: the layer fuses them into one projection.
    static func install(in model: Module) {
        var swaps: [(String, Module)] = []
        for (path, module) in model.leafModules().flattened() {
            guard !path.contains("linear_attn.in_proj_"), let layer = module as? QuantizedLinear,
                let swapped = make(from: layer)
            else { continue }
            swaps.append((path, swapped))
        }
        if !swaps.isEmpty { model.update(modules: ModuleChildren.unflattened(swaps)) }
    }

    // One simdgroup per quarter of K computes a 16 x 32 output tile with the
    // NAX tensor ops; the four partial tiles are summed in threadgroup memory.
    // Each lane loads its weights straight into its fragments. A and B take
    // the same permutation of every 256 columns, so the four lanes of a row
    // read one 128-byte line and each lane covers exactly one group.
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_small_batch_w4g64",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            using namespace mlx::steel;
            constexpr short SN = 32, NSG = 4, TN = SN / 16;
            threadgroup float partial[NSG * 16 * SN];
            const int M = x_shape[0], K = x_shape[1], N = w_shape[0];
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int n0 = threadgroup_position_in_grid.x * SN;
            const short2 coord = BaseNAXFrag::get_coord();
            const short c4 = coord.x / 4, r0 = coord.y;
            const device uint8_t* wb = (const device uint8_t*)w;
            NAXTile<float, 1, TN> acc;
            acc.clear();
            const int slice = K / NSG, end = (sg + 1) * slice;
            for (int k = sg * slice; k < end; k += 256) {
                uint4 packed[TN][2][2];
                float scale[TN][2], bias[TN][2];
                for (short nt = 0; nt < TN; nt++) {
                    for (short i = 0; i < 2; i++) {
                        const int n = n0 + nt * 16 + r0 + i * 8;
                        const device uint4* p =
                            (const device uint4*)(wb + n * (K / 2) + k / 2 + c4 * 32);
                        packed[nt][i][0] = p[0];
                        packed[nt][i][1] = p[1];
                        scale[nt][i] = float(scales[n * (K / 64) + k / 64 + c4]);
                        bias[nt][i] = float(biases[n * (K / 64) + k / 64 + c4]);
                    }
                }
                for (short s = 0; s < 4; s++) {
                    NAXTile<T, TN, 4> b;
                    for (short nt = 0; nt < TN; nt++) {
                        for (short i = 0; i < 2; i++) {
                            const uint4 v = packed[nt][i][s / 2];
                            const uint words[4] = {v.x, v.y, v.z, v.w};
                            for (short j = 0; j < 4; j++) {
                                const uint word = words[(s % 2) * 2 + j / 2];
                                for (short t = 0; t < 4; t++) {
                                    const uint q = (word >> (4 * ((j % 2) * 4 + t))) & 0xf;
                                    b.frag_at(nt, j)[i * 4 + t] =
                                        T(scale[nt][i] * float(q) + bias[nt][i]);
                                }
                            }
                        }
                    }
                    NAXTile<T, 1, 4> a;
                    for (short i = 0; i < 2; i++) {
                        const int m = r0 + i * 8;
                        for (short j = 0; j < 4; j++) {
                            for (short t = 0; t < 4; t++) {
                                a.frag_at(0, j)[i * 4 + t] =
                                    m < M ? x[m * K + k + c4 * 64 + s * 16 + j * 4 + t] : T(0);
                            }
                        }
                    }
                    tile_matmad_nax(
                        acc, a, metal::bool_constant<false>{}, b, metal::bool_constant<true>{});
                }
            }
            acc.template store<float, SN, 1>(partial + sg * 16 * SN);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = sg * 32 + lane; i < M * SN; i += NSG * 32) {
                float sum = 0;
                for (int g = 0; g < NSG; g++) sum += partial[g * 16 * SN + i];
                y[(i / SN) * N + n0 + (i % SN)] = T(sum);
            }
            """,
        header: MiMoV26NAXMetalSources.mlxHeader)
}
