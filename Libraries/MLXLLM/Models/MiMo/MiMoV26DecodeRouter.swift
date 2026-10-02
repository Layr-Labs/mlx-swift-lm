// SPDX-License-Identifier: Apache-2.0
// Adapted from jundot/omlx #3990 decode_fast.py, exact head
// e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb.
// GEMV reduction derived from MLX gemv.h, Copyright © 2023-2024 Apple Inc.
// Swift adaptation Copyright © 2026 Eigen Labs.
// Attribution: docs/mimo-v26/DECODE-ROUTER-ATTRIBUTION.md.

import Foundation
import MLX
import MLXLMCommon

/// Streams each router weight once for 1...7 rows. Every row uses the pinned
/// M=1 FP32 GEMV reduction. Multi-row GEMM may have another FP32 sum order.
/// This helper does not select experts, normalize scores or retain weights.
enum MiMoV26DecodeRouter {
    static let enabledByEnvironment = MiMoV26DecodeDefaults.isEnabled(
        MiMoV26DecodeDefaults.routerKey)

    static func supports(
        shape: [Int], weightShape: [Int], inputDType: DType,
        weightDType: DType, operandDType: DType,
        device: DeviceType?
    ) -> Bool {
        guard device == .gpu, shape.count == 3, weightShape.count == 2,
            shape[0] == 1, (1 ... 7).contains(shape[1]),
            shape[2] == weightShape[1], (1024 ... 4096).contains(shape[2]),
            shape[2] % 1024 == 0, (4 ... 256).contains(weightShape[0]),
            weightShape[0] % 4 == 0, shape[2] >= 16 * weightShape[0],
            operandDType == .bfloat16 || operandDType == .float32
        else { return false }
        let floats: [DType] = [.float16, .bfloat16, .float32]
        return floats.contains(inputDType) && floats.contains(weightDType)
    }

    static func logits(
        _ x: MLXArray, weight: MLXArray, operandDType: DType,
        enabled: Bool
    ) -> MLXArray? {
        let stream = StreamOrDevice.default
        guard enabled,
            supports(
                shape: x.shape, weightShape: weight.shape,
                inputDType: x.dtype, weightDType: weight.dtype,
                operandDType: operandDType,
                device: MiMoV26DecodeStream.deviceType(of: stream))
        else { return nil }

        // Preserve MiMoV26Router's declared operand rounding BEFORE widening.
        // For loaded BF16 weights these are no-op views; no persistent FP32
        // copy is added. The kernel widens each already-rounded element exactly.
        let operands = x.asType(operandDType, stream: stream)
        let matrix = weight.asType(operandDType, stream: stream)
        let outputs = kernel(
            [operands, matrix],
            template: [
                ("T", operands.dtype), ("W", matrix.dtype),
                ("ROWS", x.dim(1)), ("KDIM", x.dim(2)), ("NOUT", weight.dim(0)),
            ],
            grid: (32 * (weight.dim(0) / 4), 8, 1), threadGroup: (32, 8, 1),
            outputShapes: [[1, x.dim(1), weight.dim(0)]], outputDTypes: [.float32], stream: stream)
        return outputs[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mimo_v26_decode_router_gemv",
        inputNames: ["vec", "mat"], outputNames: ["out"],
        source: """
            constexpr int TM = 4;
            constexpr int TN = 4;
            constexpr int SN = 32;
            constexpr int BN = 8;
            constexpr int blockM = 4;
            constexpr int blockN = BN * SN * TN;
            threadgroup float tgp_memory[ROWS * BN * (blockM + TM)];
            uint tid = threadgroup_position_in_grid.x;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;
            int thrN = simd_lid;
            int sgN = simd_gid;
            int bn = (SN * sgN + thrN) * TN;
            int out_row = tid * blockM;
            const device W* matp = mat + size_t(out_row) * KDIM;

            float result[ROWS][TM];
            for (int r = 0; r < ROWS; r++) {
                for (int tm = 0; tm < TM; tm++) { result[r][tm] = 0; }
            }
            for (int i = 0; i < KDIM / blockN; ++i) {
                float v_coeff[ROWS][TN];
                for (int r = 0; r < ROWS; r++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[r][tn] = static_cast<float>(vec[size_t(r) * KDIM + bn + tn]);
                    }
                }
                int mat_offset = 0;
                for (int tm = 0; tm < TM; tm++) {
                    float inter[TN];
                    for (int tn = 0; tn < TN; tn++) {
                        inter[tn] = static_cast<float>(matp[mat_offset + bn + tn]);
                    }
                    for (int r = 0; r < ROWS; r++) {
                        for (int tn = 0; tn < TN; tn++) {
                            result[r][tm] += inter[tn] * v_coeff[r][tn];
                        }
                    }
                    mat_offset += KDIM;
                }
                bn += blockN;
            }
            for (int r = 0; r < ROWS; r++) {
                for (int tm = 0; tm < TM; tm++) {
                    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                        result[r][tm] += simd_shuffle_down(result[r][tm], sn);
                    }
                }
            }
            threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM);
            if (thrN == 0) {
                for (int r = 0; r < ROWS; r++) {
                    for (int tm = 0; tm < TM; tm++) {
                        tgp_results[r * BN * (blockM + TM) + tm] = result[r][tm];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgN == 0 && thrN == 0) {
                for (int r = 0; r < ROWS; r++) {
                    for (int sgn = 1; sgn < BN; sgn++) {
                        for (int tm = 0; tm < TM; tm++) {
                            result[r][tm] += tgp_memory[
                                r * BN * (blockM + TM) + sgn * (blockM + TM) + tm];
                        }
                    }
                    for (int tm = 0; tm < TM; tm++) {
                        out[size_t(r) * NOUT + out_row + tm] = result[r][tm];
                    }
                }
            }
            """, ensureRowContiguous: true)
}
