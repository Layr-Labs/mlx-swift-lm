// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of the retained DFlash2 M5/M6 affine-Q4 projection geometry.

import MLX
import MLXFast
import MLXLLM
import MLXNN

enum Qwen38ProjectionRoute: Equatable {
    case stock
    case m4KConstSplit
    case m5Exact
    case m6(kParts: Int, barrierFree: Bool)
    case m8NAX(simdgroups: Int)
    case m16NAX
}

func qwen38ProjectionRoute(width: Int, k: Int, n: Int) -> Qwen38ProjectionRoute {
    switch width {
    case 4:
        return .m4KConstSplit
    case 5:
        return .m5Exact
    case 6:
        if k == 5_120 && (n == 10_240 || n == 17_408) {
            return .m6(kParts: 1, barrierFree: true)
        }
        return .m6(kParts: 2, barrierFree: false)
    case 7:
        if (k == 6_144 && n == 5_120) || (k == 5_120 && n == 6_144) {
            return .m8NAX(simdgroups: 8)
        }
        return .stock
    case 8:
        if k == 5_120 && (n == 1_024 || n == 10_240 || n == 17_408) {
            return .m8NAX(simdgroups: 8)
        }
        return .stock
    default:
        return .stock
    }
}

/// PR #335's promoted receipts leave every production draft width on stock
/// MLX quantized matmul (`custom_draft_qmv.active_modules == 0`). M16 remains
/// available only as an isolated kernel fixture; the width-8 runtime never
/// dispatches it.
func qwen38DraftProjectionRoute(width: Int, k: Int, n: Int) -> Qwen38ProjectionRoute {
    switch width {
    case 16 where k % 256 == 0 && n % 32 == 0:
        return .m16NAX
    default:
        return .stock
    }
}

final class Qwen38M4ProjectionKernel {
    private let k: Int
    private let kParts: Int
    private let kernel: MLXFast.MLXFastKernel

    init(k: Int, groupSize: Int, kParts: Int) {
        precondition(k % 64 == 0)
        precondition([32, 64, 128].contains(groupSize))
        precondition(kParts == 2 || kParts == 4)
        self.k = k
        self.kParts = kParts
        kernel = MLXFast.metalKernel(
            name: "qwen38_vk_ks_m4_q4_kp\(kParts)_k\(k)_g\(groupSize)",
            inputNames: ["x", "w_q", "scales", "biases", "K_size", "N_size"],
            outputNames: ["y"],
            source: """
                    using namespace metal;
                    constexpr int M = 4;
                    constexpr int BN = 4;
                    constexpr int K_PARTS = \(kParts);
                    constexpr int GS = \(groupSize);

                    uint part = simdgroup_index_in_threadgroup;
                    uint lane = thread_index_in_simdgroup;
                    uint tg_n = threadgroup_position_in_grid.y;

                    constexpr int K = KCONST;
                    int N = int(N_size);
                    constexpr int K_by_8 = K / 8;
                    constexpr int K_by_gs = K / GS;
                    constexpr int packs_per_part = K_by_8 / K_PARTS;
                    int n0 = int(tg_n) * BN;
                    int pack_start = int(part) * packs_per_part;
                    int pack_end = (int(part) == K_PARTS - 1)
                        ? K_by_8 : pack_start + packs_per_part;

                    float acc[BN * M];
                    for (int i = 0; i < BN * M; ++i) acc[i] = 0.0f;

                    using Vec8 = vec<T, 8>;
                    const device Vec8 *xv = (const device Vec8*)x;
                    for (int pack = pack_start + int(lane);
                         pack < pack_end; pack += 32) {
                        int k_base = pack * 8;
                        Vec8 v0 = xv[(0 * K + k_base) / 8];
                        Vec8 v1 = xv[(1 * K + k_base) / 8];
                        Vec8 v2 = xv[(2 * K + k_base) / 8];
                        Vec8 v3 = xv[(3 * K + k_base) / 8];
                        uint32_t p0 = w_q[(n0 + 0) * K_by_8 + pack];
                        uint32_t p1 = w_q[(n0 + 1) * K_by_8 + pack];
                        uint32_t p2 = w_q[(n0 + 2) * K_by_8 + pack];
                        uint32_t p3 = w_q[(n0 + 3) * K_by_8 + pack];
                        float s0 = float(scales[(n0 + 0) * K_by_gs + (k_base / GS)]);
                        float s1 = float(scales[(n0 + 1) * K_by_gs + (k_base / GS)]);
                        float s2 = float(scales[(n0 + 2) * K_by_gs + (k_base / GS)]);
                        float s3 = float(scales[(n0 + 3) * K_by_gs + (k_base / GS)]);
                        float b0 = float(biases[(n0 + 0) * K_by_gs + (k_base / GS)]);
                        float b1 = float(biases[(n0 + 1) * K_by_gs + (k_base / GS)]);
                        float b2 = float(biases[(n0 + 2) * K_by_gs + (k_base / GS)]);
                        float b3 = float(biases[(n0 + 3) * K_by_gs + (k_base / GS)]);

                        {
                            uint32_t packed = p0;
                            float s = s0;
                            float b = b0;
                            _Pragma("unroll")
                            for (int ki = 0; ki < 8; ++ki) {
                                float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                                acc[0 * M + 0] += float(v0[ki]) * wv;
                                acc[0 * M + 1] += float(v1[ki]) * wv;
                                acc[0 * M + 2] += float(v2[ki]) * wv;
                                acc[0 * M + 3] += float(v3[ki]) * wv;
                            }
                        }
                        {
                            uint32_t packed = p1;
                            float s = s1;
                            float b = b1;
                            _Pragma("unroll")
                            for (int ki = 0; ki < 8; ++ki) {
                                float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                                acc[1 * M + 0] += float(v0[ki]) * wv;
                                acc[1 * M + 1] += float(v1[ki]) * wv;
                                acc[1 * M + 2] += float(v2[ki]) * wv;
                                acc[1 * M + 3] += float(v3[ki]) * wv;
                            }
                        }
                        {
                            uint32_t packed = p2;
                            float s = s2;
                            float b = b2;
                            _Pragma("unroll")
                            for (int ki = 0; ki < 8; ++ki) {
                                float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                                acc[2 * M + 0] += float(v0[ki]) * wv;
                                acc[2 * M + 1] += float(v1[ki]) * wv;
                                acc[2 * M + 2] += float(v2[ki]) * wv;
                                acc[2 * M + 3] += float(v3[ki]) * wv;
                            }
                        }
                        {
                            uint32_t packed = p3;
                            float s = s3;
                            float b = b3;
                            _Pragma("unroll")
                            for (int ki = 0; ki < 8; ++ki) {
                                float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                                acc[3 * M + 0] += float(v0[ki]) * wv;
                                acc[3 * M + 1] += float(v1[ki]) * wv;
                                acc[3 * M + 2] += float(v2[ki]) * wv;
                                acc[3 * M + 3] += float(v3[ki]) * wv;
                            }
                        }
                    }

                    for (int i = 0; i < BN * M; ++i) acc[i] = simd_sum(acc[i]);
                    threadgroup float partial[K_PARTS * BN * M];
                    if (lane == 0) {
                        for (int i = 0; i < BN * M; ++i) {
                            partial[int(part) * BN * M + i] = acc[i];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (part == 0 && lane < BN * M) {
                        float total = 0.0f;
                        for (int p = 0; p < K_PARTS; ++p) {
                            total += partial[p * BN * M + int(lane)];
                        }
                        int j = int(lane) / M;
                        int row = int(lane) - j * M;
                        int n_global = n0 + j;
                        if (n_global < N) y[row * N + n_global] = T(total);
                    }
                """,
            ensureRowContiguous: false)
    }

    func callAsFunction(
        input: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray
    ) -> MLXArray {
        precondition(input.dim(1) == k)
        let n = weight.dim(0)
        return kernel(
            [contiguous(input), weight, scales, biases, MLXArray(k), MLXArray(n)],
            template: [("T", input.dtype), ("KCONST", k)],
            grid: (32 * kParts, n / 4, 1),
            threadGroup: (32 * kParts, 1, 1),
            outputShapes: [[4, n]],
            outputDTypes: [input.dtype]
        )[0]
    }
}

final class Qwen38M78NAXProjectionKernel {
    private let tileRows: Int
    private let k: Int
    private let simdgroups: Int
    private let kernel: MLXFast.MLXFastKernel

    init(tileRows: Int = 8, k: Int, groupSize: Int, simdgroups: Int) {
        precondition(tileRows == 8 || tileRows == 16)
        precondition(k % (16 * simdgroups) == 0)
        precondition([32, 64, 128].contains(groupSize))
        precondition([4, 8, 16].contains(simdgroups))
        self.tileRows = tileRows
        self.k = k
        self.simdgroups = simdgroups

        var names = (0 ..< simdgroups / 2).map { "acc\($0 * 2)_\($0 * 2 + 1)" }
        var lines = (0 ..< simdgroups / 2).map {
            "float \(names[$0]) = partial[\($0 * 2)][off] + partial[\($0 * 2 + 1)][off];"
        }
        while names.count > 1 {
            var next = [String]()
            for index in stride(from: 0, to: names.count, by: 2) {
                let name = "sum_\(next.count)_\(lines.count)"
                lines.append("float \(name) = \(names[index]) + \(names[index + 1]);")
                next.append(name)
            }
            names = next
        }
        lines.append("float acc = \(names[0]);")
        let reduction = lines.map { "            \($0)" }.joined(separator: "\n")

        kernel = MLXFast.metalKernel(
            name: "qwen38_m\(tileRows)_nax_k\(k)_nsg\(simdgroups)_g\(groupSize)",
            inputNames: ["x", "w_q", "scales", "biases", "N_size"],
            outputNames: ["y"],
            source: """
                    using namespace metal;
                    using namespace mpp::tensor_ops;
                    constexpr int BM = \(tileRows);
                    constexpr int BN = 32;
                    constexpr int BK = 16;
                    constexpr int NSG = \(simdgroups);
                    constexpr int GS = \(groupSize);
                    constexpr int K = KCONST;
                    constexpr int K_by_8 = K / 8;
                    constexpr int K_by_gs = K / GS;
                    constexpr int K_chunk = K / NSG;

                    uint tid = thread_position_in_threadgroup.x;
                    uint sg_id = simdgroup_index_in_threadgroup;
                    uint lane = thread_index_in_simdgroup;
                    uint tg_n = threadgroup_position_in_grid.y;
                    int N = int(N_size);
                    int n0 = int(tg_n) * BN;
                    int k_begin = int(sg_id) * K_chunk;
                    int k_end = k_begin + K_chunk;

                    threadgroup T B_tile[NSG][BK * BN];
                    threadgroup float partial[NSG][BM * BN];
                    constexpr auto desc = matmul2d_descriptor(
                        \(tileRows), 32, 16, false, false, false,
                        matmul2d_descriptor::mode::multiply_accumulate);
                    matmul2d<desc, metal::execution_simdgroup> op;

                    tensor<device T, dextents<int, 2>, tensor_inline> A(
                        (device T*)x, dextents<int, 2>{K, BM}, array<int, 2>{1, K});
                    tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
                        B_tile[sg_id], dextents<int, 2>{BN, BK}, array<int, 2>{1, BN});
                    tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
                        partial[sg_id], dextents<int, 2>{BN, BM}, array<int, 2>{1, BN});

                    auto ct_c = op.template get_destination_cooperative_tensor<
                        tensor<device T, extents<int, \(tileRows), 16>, tensor_inline>,
                        tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>, float>();
                    _Pragma("unroll")
                    for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = 0.0f;

                    int n_global = n0 + int(lane);
                    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
                        uint32_t p0 = w_q[n_global * K_by_8 + ((k0 + 0) >> 3)];
                        uint32_t p1 = w_q[n_global * K_by_8 + ((k0 + 8) >> 3)];
                        float s0 = float(scales[n_global * K_by_gs + ((k0 + 0) / GS)]);
                        float s1 = float(scales[n_global * K_by_gs + ((k0 + 8) / GS)]);
                        float b0 = float(biases[n_global * K_by_gs + ((k0 + 0) / GS)]);
                        float b1 = float(biases[n_global * K_by_gs + ((k0 + 8) / GS)]);
                        _Pragma("unroll")
                        for (int ki = 0; ki < 8; ++ki) {
                            uint32_t nib = (p0 >> (ki * 4)) & 0xFu;
                            B_tile[sg_id][ki * BN + int(lane)] = T(float(nib) * s0 + b0);
                        }
                        _Pragma("unroll")
                        for (int ki = 0; ki < 8; ++ki) {
                            uint32_t nib = (p1 >> (ki * 4)) & 0xFu;
                            B_tile[sg_id][(8 + ki) * BN + int(lane)] = T(float(nib) * s1 + b1);
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        auto tA = A.template slice<16, \(tileRows)>(k0, 0);
                        auto tB = B.template slice<32, 16>(0, 0);
                        op.run(tA, tB, ct_c);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                    }

                    auto tC = C.template slice<32, \(tileRows)>(0, 0);
                    ct_c.store(tC);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (int off = int(tid); off < BM * BN; off += NSG * 32) {
                    \(reduction)
                        int row = off / BN;
                        int col = off - row * BN;
                        y[row * N + n0 + col] = T(acc);
                    }
                """,
            header: "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
            ensureRowContiguous: false)
    }

    func callAsFunction(
        input: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray
    ) -> MLXArray {
        precondition(input.dim(1) == k)
        let rows = input.dim(0)
        precondition(rows > 0 && rows <= tileRows)
        let tiledInput =
            rows == tileRows
            ? contiguous(input)
            : contiguous(
                concatenated([
                    input, MLXArray.zeros([tileRows - rows, k], dtype: input.dtype),
                ]))
        let n = weight.dim(0)
        let output = kernel(
            [tiledInput, weight, scales, biases, MLXArray(n)],
            template: [("T", input.dtype), ("KCONST", k)],
            grid: (32 * simdgroups, n / 32, 1),
            threadGroup: (32 * simdgroups, 1, 1),
            outputShapes: [[tileRows, n]],
            outputDTypes: [input.dtype]
        )[0]
        return rows == tileRows ? output : output[0 ..< rows]
    }
}

final class Qwen38M56ProjectionKernel {
    private let rows: Int
    private let kParts: Int
    private let kernel: MLXFast.MLXFastKernel

    init(rows: Int, groupSize: Int, kParts: Int, barrierFree: Bool) {
        precondition(rows == 5 || rows == 6)
        precondition([32, 64, 128].contains(groupSize))
        precondition(kParts > 0 && (!barrierFree || kParts == 1))
        self.rows = rows
        self.kParts = kParts

        let rowLoads = (0 ..< rows).map {
            "            Vec8 v\($0) = xv[(\($0) * K + k_base) / 8];"
        }.joined(separator: "\n")
        let rowFMAs = (0 ..< rows).map {
            "                    acc[j * M + \($0)] += float(v\($0)[ki]) * wv;"
        }.joined(separator: "\n")
        let reduction: String
        if barrierFree {
            reduction = """
                    if (lane < BN * M) {
                        int j = int(lane) / M;
                        int row = int(lane) - j * M;
                        int n_global = n0 + j;
                        if (n_global < N) {
                            y[row * N + n_global] = T(acc[int(lane)]);
                        }
                    }
                """
        } else {
            reduction = """
                    threadgroup float partial[K_PARTS * BN * M];
                    if (lane == 0) {
                        _Pragma("unroll")
                        for (int i = 0; i < BN * M; ++i) {
                            partial[int(part) * BN * M + i] = acc[i];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (part == 0 && lane < BN * M) {
                        float total = 0.0f;
                        _Pragma("unroll")
                        for (int p = 0; p < K_PARTS; ++p) {
                            total += partial[p * BN * M + int(lane)];
                        }
                        int j = int(lane) / M;
                        int row = int(lane) - j * M;
                        int n_global = n0 + j;
                        if (n_global < N) {
                            y[row * N + n_global] = T(total);
                        }
                    }
                """
        }

        kernel = MLXFast.metalKernel(
            name: "qwen38_m\(rows)_kp\(kParts)_\(barrierFree ? "direct" : "reduce")_g\(groupSize)",
            inputNames: ["x", "w_q", "scales", "biases", "K_size", "N_size"],
            outputNames: ["y"],
            source: """
                    using namespace metal;
                    constexpr int M = \(rows);
                    constexpr int BN = 4;
                    constexpr int K_PARTS = \(kParts);
                    constexpr int GS = \(groupSize);

                    uint part = simdgroup_index_in_threadgroup;
                    uint lane = thread_index_in_simdgroup;
                    uint tg_n = threadgroup_position_in_grid.y;

                    int K = int(K_size);
                    int N = int(N_size);
                    int K_by_8 = K / 8;
                    int K_by_gs = K / GS;
                    int n0 = int(tg_n) * BN;
                    int packs_per_part = K_by_8 / K_PARTS;
                    int pack_start = int(part) * packs_per_part;
                    int pack_end = (int(part) == K_PARTS - 1)
                        ? K_by_8 : pack_start + packs_per_part;

                    float acc[BN * M];
                    _Pragma("unroll")
                    for (int i = 0; i < BN * M; ++i) {
                        acc[i] = 0.0f;
                    }

                    using Vec8 = vec<T, 8>;
                    const device Vec8 *xv = (const device Vec8*)x;

                    for (int pack = pack_start + int(lane);
                         pack < pack_end; pack += 32) {
                        int k_base = pack * 8;
                    \(rowLoads)
                        _Pragma("unroll")
                        for (int j = 0; j < BN; ++j) {
                            uint32_t packed = w_q[(n0 + j) * K_by_8 + pack];
                            float s = float(scales[(n0 + j) * K_by_gs + (k_base / GS)]);
                            float b = float(biases[(n0 + j) * K_by_gs + (k_base / GS)]);
                            _Pragma("unroll")
                            for (int ki = 0; ki < 8; ++ki) {
                                float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    \(rowFMAs)
                            }
                        }
                    }

                    _Pragma("unroll")
                    for (int i = 0; i < BN * M; ++i) {
                        acc[i] = simd_sum(acc[i]);
                    }

                    \(reduction)
                """,
            ensureRowContiguous: false)
    }

    func callAsFunction(
        input: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray
    ) -> MLXArray {
        let k = input.dim(1)
        let n = weight.dim(0)
        return kernel(
            [contiguous(input), weight, scales, biases, MLXArray(k), MLXArray(n)],
            template: [("T", input.dtype)],
            grid: (32 * kParts, n / 4, 1),
            threadGroup: (32 * kParts, 1, 1),
            outputShapes: [[rows, n]],
            outputDTypes: [input.dtype]
        )[0]
    }
}

func qwen38M56QuantizedMM(
    input: MLXArray,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    groupSize: Int,
    kParts: Int,
    barrierFree: Bool
) -> MLXArray {
    Qwen38M56ProjectionKernel(
        rows: input.dim(0),
        groupSize: groupSize,
        kParts: kParts,
        barrierFree: barrierFree
    )(input: input, weight: weight, scales: scales, biases: biases)
}

func qwen38M4QuantizedMM(
    input: MLXArray,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    groupSize: Int
) -> MLXArray {
    Qwen38M4ProjectionKernel(
        k: input.dim(1), groupSize: groupSize,
        kParts: weight.dim(0) >= 4_096 ? 2 : 4)(
            input: input, weight: weight, scales: scales, biases: biases)
}

func qwen38M78NAXQuantizedMM(
    input: MLXArray,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    groupSize: Int,
    simdgroups: Int
) -> MLXArray {
    Qwen38M78NAXProjectionKernel(
        k: input.dim(1), groupSize: groupSize, simdgroups: simdgroups)(
            input: input, weight: weight, scales: scales, biases: biases)
}

func qwen38M16NAXQuantizedMM(
    input: MLXArray,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    groupSize: Int
) -> MLXArray {
    Qwen38M78NAXProjectionKernel(
        tileRows: 16, k: input.dim(1), groupSize: groupSize, simdgroups: 8)(
            input: input, weight: weight, scales: scales, biases: biases)
}

private final class Qwen38OptimizedQuantizedLinear: QuantizedLinear {
    private let k: Int
    private let n: Int
    private let m4: Qwen38M4ProjectionKernel
    private let m5: Qwen38M56ProjectionKernel
    private let m6: Qwen38M56ProjectionKernel
    private let m78: Qwen38M78NAXProjectionKernel?
    private let m16: Qwen38M78NAXProjectionKernel?
    private let m7Route: Qwen38ProjectionRoute
    private let m8Route: Qwen38ProjectionRoute

    init(source: QuantizedLinear) {
        let shape = source.shape
        n = shape.0
        k = shape.1
        m4 = Qwen38M4ProjectionKernel(
            k: shape.1, groupSize: source.groupSize,
            kParts: shape.0 >= 4_096 ? 2 : 4)
        m5 = Qwen38M56ProjectionKernel(
            rows: 5, groupSize: source.groupSize, kParts: 2, barrierFree: false)
        let m6Route = qwen38ProjectionRoute(width: 6, k: shape.1, n: shape.0)
        switch m6Route {
        case .m6(let kParts, let barrierFree):
            m6 = Qwen38M56ProjectionKernel(
                rows: 6, groupSize: source.groupSize,
                kParts: kParts, barrierFree: barrierFree)
        default:
            preconditionFailure("Qwen 3.8 M6 route table is incomplete")
        }
        m7Route = qwen38ProjectionRoute(width: 7, k: shape.1, n: shape.0)
        m8Route = qwen38ProjectionRoute(width: 8, k: shape.1, n: shape.0)
        if case .m8NAX(let simdgroups) = m7Route {
            m78 = Qwen38M78NAXProjectionKernel(
                tileRows: 8, k: shape.1, groupSize: source.groupSize,
                simdgroups: simdgroups)
        } else if case .m8NAX(let simdgroups) = m8Route {
            m78 = Qwen38M78NAXProjectionKernel(
                tileRows: 8, k: shape.1, groupSize: source.groupSize,
                simdgroups: simdgroups)
        } else {
            m78 = nil
        }
        if shape.1 % 256 == 0 && shape.0 % 32 == 0 {
            m16 = Qwen38M78NAXProjectionKernel(
                tileRows: 16, k: shape.1, groupSize: source.groupSize,
                simdgroups: 8)
        } else {
            m16 = nil
        }
        super.init(
            weight: source.weight, bias: source.bias,
            scales: source.scales, biases: source.biases,
            groupSize: source.groupSize, bits: source.bits, mode: source.mode)
        freeze()
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let width = input.shape.dropLast().reduce(1, *)
        let input2 = input.reshaped([width, k])
        let output2: MLXArray
        switch width {
        case 4:
            output2 = m4(
                input: input2, weight: weight, scales: scales, biases: biases!)
        case 5:
            output2 = m5(
                input: input2, weight: weight, scales: scales, biases: biases!)
        case 6:
            output2 = m6(
                input: input2, weight: weight, scales: scales, biases: biases!)
        case 7 where m7Route != .stock:
            output2 = m78!(
                input: input2, weight: weight, scales: scales, biases: biases!)
        case 8 where m8Route != .stock:
            output2 = m78!(
                input: input2, weight: weight, scales: scales, biases: biases!)
        case 7 where m16 != nil, 8 where m16 != nil:
            output2 = m16!(
                input: input2, weight: weight, scales: scales, biases: biases!)
        default:
            return super.callAsFunction(input)
        }
        return output2.reshaped(Array(input.shape.dropLast()) + [n])
    }
}

/// Construction-only M16 fixture. The promoted PR #335 generation route does
/// not install this wrapper around the draft model.
private final class Qwen38DraftQuantizedLinear: QuantizedLinear {
    private let k: Int
    private let n: Int
    private let m4: Qwen38M4ProjectionKernel
    private let m16: Qwen38M78NAXProjectionKernel?

    init(source: QuantizedLinear) {
        let shape = source.shape
        n = shape.0
        k = shape.1
        m4 = Qwen38M4ProjectionKernel(
            k: shape.1, groupSize: source.groupSize,
            kParts: shape.0 >= 4_096 ? 2 : 4)
        if qwen38DraftProjectionRoute(width: 16, k: shape.1, n: shape.0) == .m16NAX {
            m16 = Qwen38M78NAXProjectionKernel(
                tileRows: 16, k: shape.1, groupSize: source.groupSize,
                simdgroups: 8)
        } else {
            m16 = nil
        }
        super.init(
            weight: source.weight, bias: source.bias,
            scales: source.scales, biases: source.biases,
            groupSize: source.groupSize, bits: source.bits, mode: source.mode)
        freeze()
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let width = input.shape.dropLast().reduce(1, *)
        let input2 = input.reshaped([width, k])
        let output2: MLXArray
        switch width {
        case 4:
            output2 = m4(
                input: input2, weight: weight, scales: scales, biases: biases!)
        case 16 where m16 != nil:
            output2 = m16!(
                input: input2, weight: weight, scales: scales, biases: biases!)
        default:
            return super.callAsFunction(input)
        }
        return output2.reshaped(Array(input.shape.dropLast()) + [n])
    }
}

private enum Qwen38ProjectionInstallProfile {
    case targetFullStack
    case dflashDraft
}

public struct Qwen38ProjectionInstallReport: Sendable, Equatable {
    public let installed: Int
    public let preservedFusedGDNInputs: Int
    public let stockQuantized: Int

    public init(installed: Int, preservedFusedGDNInputs: Int, stockQuantized: Int) {
        self.installed = installed
        self.preservedFusedGDNInputs = preservedFusedGDNInputs
        self.stockQuantized = stockQuantized
    }
}

public enum Qwen38ProjectionInstallError: Error, CustomStringConvertible {
    case invalidModule(path: String, reason: String)
    case emptyOptimizedLane
    case incompleteInstall(
        expected: Qwen38ProjectionInstallReport, actual: Qwen38ProjectionInstallReport)

    public var description: String {
        switch self {
        case .invalidModule(let path, let reason):
            "Qwen 3.8 projection invariant failed at \(path): \(reason)"
        case .emptyOptimizedLane:
            "Qwen 3.8 projection lane found no affine Q4 modules"
        case .incompleteInstall(let expected, let actual):
            "Qwen 3.8 projection lane is incomplete: expected "
                + "\(expected.installed)/\(expected.preservedFusedGDNInputs)/"
                + "\(expected.stockQuantized) installed/preserved/stock, got "
                + "\(actual.installed)/\(actual.preservedFusedGDNInputs)/"
                + "\(actual.stockQuantized)"
        }
    }
}

private let qwen38ProductionProjectionInstall = Qwen38ProjectionInstallReport(
    installed: 232,
    preservedFusedGDNInputs: 192,
    stockQuantized: 73)

private let qwen38DraftProjectionInstall = Qwen38ProjectionInstallReport(
    installed: 47,
    preservedFusedGDNInputs: 0,
    stockQuantized: 0)

struct Qwen38ProductionProjectionInventory {
    let optimized: Set<String>
    let preserved: Set<String>
    let stock: Set<String>
}

func qwen38ProductionProjectionInventory() -> Qwen38ProductionProjectionInventory {
    var optimized = Set<String>()
    var preserved = Set<String>()
    var stock: Set<String> = ["lm_head"]
    for layer in 0 ..< 64 {
        let prefix = "model.layers.\(layer)"
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let path = "\(prefix).mlp.\(projection)"
            if layer >= 56 {
                stock.insert(path)
            } else {
                optimized.insert(path)
            }
        }
        if layer % 4 == 3 {
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                optimized.insert("\(prefix).self_attn.\(projection)")
            }
        } else {
            for projection in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"] {
                preserved.insert("\(prefix).linear_attn.\(projection)")
            }
            stock.insert("\(prefix).linear_attn.out_proj")
        }
    }
    return Qwen38ProductionProjectionInventory(
        optimized: optimized, preserved: preserved, stock: stock)
}

func qwen38DraftProjectionInventory() -> Set<String> {
    var paths: Set<String> = ["fc", "candidate_selector.hidden_projection"]
    for layer in 0 ..< 5 {
        let prefix = "layers.\(layer)"
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            paths.insert("\(prefix).self_attn.\(projection)")
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            paths.insert("\(prefix).mlp.\(projection)")
        }
        paths.insert("\(prefix).attention_conv.kernel_projection")
        paths.insert("\(prefix).mlp_conv.kernel_projection")
    }
    return paths
}

private func qwen38TargetRelativeProjectionPath(_ path: String) -> String {
    let prefix = "language_model."
    return path.hasPrefix(prefix) ? String(path.dropFirst(prefix.count)) : path
}

private func qwen38DescribeInventoryMismatch(
    expected: Set<String>, actual: Set<String>
) -> String {
    let missing = expected.subtracting(actual).sorted()
    let unexpected = actual.subtracting(expected).sorted()
    return "missing=\(missing), unexpected=\(unexpected)"
}

func validateQwen38AffineQuantizedLinear(
    _ linear: QuantizedLinear,
    path: String,
    bits: Int,
    groupSize: Int
) throws {
    guard linear.bits == bits,
        linear.groupSize == groupSize,
        linear.mode.rawValue == QuantizationMode.affine.rawValue
    else {
        throw Qwen38ProjectionInstallError.invalidModule(
            path: path,
            reason:
                "expected affine Q\(bits)/G\(groupSize), got Q\(linear.bits)/G\(linear.groupSize)")
    }
    guard linear.bias == nil, linear.biases != nil else {
        throw Qwen38ProjectionInstallError.invalidModule(
            path: path, reason: "expected bias-free affine quantization")
    }
    guard linear.weight.dtype == .uint32,
        linear.scales.dtype == .bfloat16,
        linear.biases!.dtype == .bfloat16
    else {
        throw Qwen38ProjectionInstallError.invalidModule(
            path: path, reason: "expected uint32 weights with BF16 scale and affine bias")
    }
}

@discardableResult
func validateQwen38DraftQuantization(in model: Module) throws -> Int {
    var actual = Set<String>()
    for (path, module) in model.leafModules().flattened() {
        guard let linear = module as? QuantizedLinear else { continue }
        guard ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self) else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path,
                reason: "expected an unwrapped QuantizedLinear after draft quantization")
        }
        try validateQwen38AffineQuantizedLinear(
            linear, path: path, bits: 4, groupSize: 64)
        actual.insert(path)
    }
    let expected = qwen38DraftProjectionInventory()
    guard actual == expected else {
        throw Qwen38ProjectionInstallError.invalidModule(
            path: "<draft quantization inventory>",
            reason: qwen38DescribeInventoryMismatch(expected: expected, actual: actual))
    }
    return actual.count
}

public func validateQwen38StockTargetQuantization(in model: Module) throws {
    let expected = qwen38ProductionProjectionInventory()
    var optimized = Set<String>()
    var preserved = Set<String>()
    var stock = Set<String>()
    for (path, module) in model.leafModules().flattened() {
        guard let linear = module as? QuantizedLinear else { continue }
        guard ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self) else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path,
                reason: "expected an unwrapped QuantizedLinear in the stock target")
        }
        let relativePath = qwen38TargetRelativeProjectionPath(path)
        if expected.stock.contains(relativePath) {
            try validateQwen38AffineQuantizedLinear(
                linear, path: path, bits: 8, groupSize: 64)
            stock.insert(relativePath)
        } else if expected.optimized.contains(relativePath) {
            try validateQwen38AffineQuantizedLinear(
                linear, path: path, bits: 4, groupSize: 32)
            optimized.insert(relativePath)
        } else if expected.preserved.contains(relativePath) {
            try validateQwen38AffineQuantizedLinear(
                linear, path: path, bits: 4, groupSize: 32)
            preserved.insert(relativePath)
        } else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path, reason: "unexpected stock target projection")
        }
    }
    for (label, expectedPaths, actualPaths) in [
        ("optimized", expected.optimized, optimized),
        ("preserved", expected.preserved, preserved),
        ("stock", expected.stock, stock),
    ] where expectedPaths != actualPaths {
        throw Qwen38ProjectionInstallError.invalidModule(
            path: "<stock target \(label) inventory>",
            reason: qwen38DescribeInventoryMismatch(
                expected: expectedPaths, actual: actualPaths))
    }
}

func validateQwen38ProductionProjectionInstall(
    _ report: Qwen38ProjectionInstallReport
) throws {
    guard report == qwen38ProductionProjectionInstall else {
        throw Qwen38ProjectionInstallError.incompleteInstall(
            expected: qwen38ProductionProjectionInstall,
            actual: report)
    }
}

private func validateQwen38DraftProjectionInstall(
    _ report: Qwen38ProjectionInstallReport
) throws {
    guard report == qwen38DraftProjectionInstall else {
        throw Qwen38ProjectionInstallError.incompleteInstall(
            expected: qwen38DraftProjectionInstall,
            actual: report)
    }
}

/// Installs the retained verify-width projection routes once, after weights are
/// loaded and before measured generation. The GDN source views stay stock so
/// Qwen35 can construct its one-allocation fused input projection; row-24 owns
/// that fused boundary in the complete DFlash stack.
@discardableResult
public func installQwen38ProjectionStack(in model: Module) throws
    -> Qwen38ProjectionInstallReport
{
    try installQwen38ProjectionStack(in: model, profile: .targetFullStack)
}

/// Installs only the projection routes used by MTPLX's draft wrapper. The
/// distinction is fixed during model construction and adds no measured-path
/// eligibility branch.
@discardableResult
public func installQwen38DraftProjectionStack(in model: Module) throws
    -> Qwen38ProjectionInstallReport
{
    try installQwen38ProjectionStack(in: model, profile: .dflashDraft)
}

private func installQwen38ProjectionStack(
    in model: Module,
    profile: Qwen38ProjectionInstallProfile
) throws -> Qwen38ProjectionInstallReport {
    let gdnSourceSuffixes = [".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a"]
    let leaves = model.leafModules().flattened()
    var replacements = [String: Module]()
    var preservedFusedGDNInputs = 0
    var stockQuantized = 0
    var optimizedPaths = Set<String>()
    var preservedPaths = Set<String>()
    var stockPaths = Set<String>()

    for (path, module) in leaves {
        guard let linear = module as? QuantizedLinear else { continue }
        guard ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self) else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path,
                reason: "expected an unwrapped QuantizedLinear at installation")
        }
        let relativePath: String
        switch profile {
        case .targetFullStack:
            relativePath = qwen38TargetRelativeProjectionPath(path)
        case .dflashDraft:
            relativePath = path
        }
        guard linear.bits == 4 && linear.mode.rawValue == QuantizationMode.affine.rawValue else {
            try validateQwen38AffineQuantizedLinear(
                linear, path: path, bits: 8, groupSize: 64)
            stockQuantized += 1
            stockPaths.insert(relativePath)
            continue
        }
        if gdnSourceSuffixes.contains(where: path.hasSuffix) {
            try validateQwen38AffineQuantizedLinear(
                linear, path: path, bits: 4, groupSize: 32)
            preservedFusedGDNInputs += 1
            preservedPaths.insert(relativePath)
            continue
        }
        let (n, k) = linear.shape
        guard [32, 64, 128].contains(linear.groupSize) else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path, reason: "unsupported group size \(linear.groupSize)")
        }
        guard k % 32 == 0 && n % 4 == 0 else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path, reason: "KxN \(k)x\(n) violates split-K geometry")
        }
        guard linear.bias == nil, linear.biases != nil else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path, reason: "expected bias-free affine quantization")
        }
        guard linear.weight.dtype == .uint32,
            linear.scales.dtype == .bfloat16,
            linear.biases!.dtype == .bfloat16
        else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path, reason: "expected uint32 weights with BF16 scale and affine bias")
        }
        switch profile {
        case .targetFullStack:
            try validateQwen38AffineQuantizedLinear(
                linear, path: path, bits: 4, groupSize: 32)
            replacements[path] = Qwen38OptimizedQuantizedLinear(source: linear)
            optimizedPaths.insert(relativePath)
        case .dflashDraft:
            try validateQwen38AffineQuantizedLinear(
                linear, path: path, bits: 4, groupSize: 64)
            replacements[path] = Qwen38DraftQuantizedLinear(source: linear)
            optimizedPaths.insert(relativePath)
        }
    }
    guard !replacements.isEmpty else { throw Qwen38ProjectionInstallError.emptyOptimizedLane }
    let report = Qwen38ProjectionInstallReport(
        installed: replacements.count,
        preservedFusedGDNInputs: preservedFusedGDNInputs,
        stockQuantized: stockQuantized)
    switch profile {
    case .targetFullStack:
        try validateQwen38ProductionProjectionInstall(report)
        let expected = qwen38ProductionProjectionInventory()
        guard optimizedPaths == expected.optimized else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: "<optimized inventory>",
                reason: qwen38DescribeInventoryMismatch(
                    expected: expected.optimized, actual: optimizedPaths))
        }
        guard preservedPaths == expected.preserved else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: "<preserved inventory>",
                reason: qwen38DescribeInventoryMismatch(
                    expected: expected.preserved, actual: preservedPaths))
        }
        guard stockPaths == expected.stock else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: "<stock inventory>",
                reason: qwen38DescribeInventoryMismatch(
                    expected: expected.stock, actual: stockPaths))
        }
    case .dflashDraft:
        try validateQwen38DraftProjectionInstall(report)
        let expected = qwen38DraftProjectionInventory()
        guard optimizedPaths == expected else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: "<draft inventory>",
                reason: qwen38DescribeInventoryMismatch(
                    expected: expected, actual: optimizedPaths))
        }
    }
    // Apply each replacement at its direct owning module. This avoids both
    // sparse-array reconstruction and unrelated non-ModuleInfo leaves (such as
    // RoPE helpers) while keeping every mutation at the construction boundary.
    var owners = [String: Module]()
    model.visit { path, module in owners[path] = module }
    for (path, replacement) in replacements.sorted(by: { $0.key < $1.key }) {
        let components = path.split(separator: ".").map(String.init)
        let childKey = components.last!
        let ownerPath = components.dropLast().joined(separator: ".")
        guard let owner = owners[ownerPath] else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: path, reason: "direct owner \(ownerPath) is missing")
        }
        try owner.update(
            modules: ModuleChildren(values: [childKey: .value(replacement)]),
            verify: .none)
    }
    switch profile {
    case .targetFullStack:
        let installedDFlashInputs: Bool
        switch model {
        case let target as Qwen35TextModel:
            installedDFlashInputs = target.installDFlashInputProjections()
        case let target as Qwen35Model:
            installedDFlashInputs = target.installDFlashInputProjections()
        default:
            throw Qwen38ProjectionInstallError.invalidModule(
                path: "<root>", reason: "target is not a Qwen 3.8 model")
        }
        guard installedDFlashInputs else {
            throw Qwen38ProjectionInstallError.invalidModule(
                path: "<root>",
                reason: "cannot install every fused DFlash recurrent input")
        }
    case .dflashDraft:
        break
    }
    return report
}
