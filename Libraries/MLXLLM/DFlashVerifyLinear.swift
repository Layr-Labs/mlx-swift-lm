// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Quantized linear wrapper with an M=16 verify-time matmul fast path.
///
/// DFlash verifies one speculative block with the target model. For K=16
/// verification the target's linear layers see exactly 16 flattened rows, where
/// the generic quantized matmul path leaves measurable throughput on the table.
/// This wrapper keeps normal `QuantizedLinear` behavior for every other shape.
public final class DFlashVerifyQuantizedLinear: QuantizedLinear {
    private let enableQMM: Bool
    private let qmmWeight: MLXArray
    private let qmmScales: MLXArray
    private let qmmBiases: MLXArray?

    private lazy var pipeBF16Kernel = Self.makePipeKernel(groupSize: groupSize, dtypeTag: "bf16")
    private lazy var pipeFP16Kernel = Self.makePipeKernel(groupSize: groupSize, dtypeTag: "fp16")

    public init(_ other: QuantizedLinear, enableQMM: Bool = true) {
        self.enableQMM = enableQMM
        self.qmmWeight = other.weight.contiguous()
        self.qmmScales = other.scales.contiguous()
        self.qmmBiases = other.biases?.contiguous()
        super.init(
            weight: other.weight,
            bias: other.bias,
            scales: other.scales,
            biases: other.biases,
            groupSize: other.groupSize,
            bits: other.bits,
            mode: other.mode
        )
        if let qmmBiases {
            eval(qmmWeight, qmmScales, qmmBiases)
        } else {
            eval(qmmWeight, qmmScales)
        }
        self.freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y =
            enableQMM
            ? verifyQMM(x) ?? fallback(x)
            : fallback(x)
        if let bias {
            y = y + bias
        }
        return y
    }

    public static func isEligible(_ linear: QuantizedLinear, maxOutputDimensions: Int = 100_000)
        -> Bool
    {
        guard !(linear is DFlashVerifyQuantizedLinear) else { return false }
        guard linear.bits == 4 else { return false }
        guard [32, 64, 128].contains(linear.groupSize) else { return false }
        guard linear.mode == .affine else { return false }
        guard linear.biases != nil else { return false }

        let outputDimensions = linear.weight.shape[0]
        let inputDimensions = linear.weight.shape[1] * (32 / linear.bits)
        guard outputDimensions < maxOutputDimensions else { return false }
        guard outputDimensions % 32 == 0 else { return false }
        // This first Swift port implements the Python "mma2big_pipe" path with
        // K_PARTS=8, which requires K to split into 8 32-wide chunks.
        return inputDimensions % (32 * Self.pipeKParts) == 0
    }

    private func fallback(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    private func verifyQMM(_ x: MLXArray) -> MLXArray? {
        guard bits == 4, mode == .affine else { return nil }
        guard qmmBiases != nil else { return nil }
        guard x.dtype == .bfloat16 || x.dtype == .float16 else { return nil }
        guard let inputDimensions = x.shape.last else { return nil }

        let rowCount = x.shape.dropLast().reduce(1, *)
        guard rowCount == Self.blockRows else { return nil }

        let outputDimensions = qmmWeight.shape[0]
        guard outputDimensions % 32 == 0 else { return nil }
        guard inputDimensions % (32 * Self.pipeKParts) == 0 else { return nil }
        guard inputDimensions == qmmWeight.shape[1] * (32 / bits) else { return nil }

        let x2 = x.reshaped([Self.blockRows, inputDimensions]).contiguous()
        let kernel = x.dtype == .bfloat16 ? pipeBF16Kernel : pipeFP16Kernel
        guard let qmmBiases else { return nil }
        let partials = kernel(
            [x2, qmmWeight, qmmScales, qmmBiases, Self.blockRows, inputDimensions, outputDimensions, Self.pipeKParts],
            template: [("T", x.dtype)],
            grid: (64, outputDimensions / 32, Self.pipeKParts),
            threadGroup: (64, 1, 1),
            outputShapes: [[Self.pipeKParts, Self.blockRows, outputDimensions]],
            outputDTypes: [.float32]
        )[0]
        let y = partials.sum(axis: 0).asType(x.dtype)
        return y.reshaped(Array(x.shape.dropLast()) + [outputDimensions])
    }

    private static let blockRows = 16
    private static let pipeKParts = 8

    private static func makePipeKernel(groupSize: Int, dtypeTag: String) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "dflash_verify_mma2big_pipe_gs\(groupSize)_\(dtypeTag)",
            inputNames: ["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size", "K_parts"],
            outputNames: ["partials"],
            source: pipeKernelSource(groupSize: groupSize)
        )
    }

    private static func pipeKernelSource(groupSize: Int) -> String {
        """
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = \(groupSize);

        uint tid       = thread_position_in_threadgroup.x;
        uint sg_id     = tid / 32;
        uint tg_n      = threadgroup_position_in_grid.y;
        uint tg_k_part = threadgroup_position_in_grid.z;

        int K = int(K_size);
        int N = int(N_size);
        int KP = int(K_parts);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int k_slice = K / KP;
        int k_begin = k_slice * int(tg_k_part);
        int k_end   = k_begin + k_slice;

        threadgroup T B_tile[2][BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        #define STAGE_B(slot, k0_stage) {{                                              \\
            {{                                                                          \\
                int n_global = n0 + dq_n_a;                                             \\
                int k_base = (k0_stage) + dq_k_a * 8;                                   \\
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];               \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);            \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);            \\
                _Pragma("unroll")                                                       \\
                for (int ki = 0; ki < 8; ++ki) {{                                       \\
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;                         \\
                    B_tile[slot][(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b); \\
                }}                                                                      \\
            }}                                                                          \\
            {{                                                                          \\
                int n_global = n0 + dq_n_b;                                             \\
                int k_base = (k0_stage) + dq_k_b * 8;                                   \\
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];               \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);            \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);            \\
                _Pragma("unroll")                                                       \\
                for (int ki = 0; ki < 8; ++ki) {{                                       \\
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;                         \\
                    B_tile[slot][(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b); \\
                }}                                                                      \\
            }}                                                                          \\
        }}

        STAGE_B(0, k_begin);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int read_slot = 0;
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            int write_slot = 1 - read_slot;
            int k0_next = k0 + BK;

            if (k0_next < k_end) {{
                STAGE_B(write_slot, k0_next);
            }}

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            read_slot = write_slot;
        }}

        int part_off = int(tg_k_part) * BM * N;
        simdgroup_store(c_tL, partials + part_off + n0 + sg_n_off,                     N);
        simdgroup_store(c_tR, partials + part_off + n0 + sg_n_off + 8,                 N);
        simdgroup_store(c_bL, partials + part_off + 8 * N + n0 + sg_n_off,             N);
        simdgroup_store(c_bR, partials + part_off + 8 * N + n0 + sg_n_off + 8,         N);

        #undef STAGE_B
        """
    }
}

public enum DFlashVerifyLinear {
    /// Replace compatible 4-bit `QuantizedLinear` leaves with
    /// `DFlashVerifyQuantizedLinear`.
    ///
    /// The replacement is shape guarded and falls back to `quantizedMM` except
    /// for the DFlash verify shape `M == 16`.
    @discardableResult
    public static func install(
        on model: Module,
        enableQMM: Bool = true,
        maxOutputDimensions: Int = 100_000
    ) -> Int {
        let updates =
            model
            .leafModules()
            .flattened()
            .compactMap { path, module -> (String, Module)? in
            guard let linear = module as? QuantizedLinear else { return nil }
            guard DFlashVerifyQuantizedLinear.isEligible(
                linear, maxOutputDimensions: maxOutputDimensions)
            else {
                return nil
            }
            return (path, DFlashVerifyQuantizedLinear(linear, enableQMM: enableQMM))
        }

        guard !updates.isEmpty else { return 0 }
        model.update(modules: ModuleChildren.unflattened(updates))
        return updates.count
    }
}
