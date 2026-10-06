// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Flash-Next routed MoE gather_qmm expert tiles (E=512, W4G64). Same
// `affine_gather_qmm_gemma4_expert_tiles` microkernel Fusion/mlx-swift
// compile for E=128/256, plus a JIT descriptor builder instantiated at
// NE=512. Assignment counts are T×10 (80000 / 81920 at listing chunks),
// which the Gemma classifier rejects. Kill: DARKBLOOM_QWEN4_GATHER_QMM=0.
// 27B is E=256 and never matches.

import Foundation
import MLX
import MLXFast

public enum Qwen4ExpGatherQMMInvocation: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var native = 0
    nonisolated(unsafe) private static var fallback = 0
    nonisolated(unsafe) private static var missGeometry = 0
    nonisolated(unsafe) private static var missDtype = 0
    nonisolated(unsafe) private static var missSorted = 0

    public struct Snapshot: Sendable, Equatable {
        public var native: Int
        public var fallback: Int
        public var missGeometry: Int
        public var missDtype: Int
        public var missSorted: Int

        public init(native: Int, fallback: Int, missGeometry: Int, missDtype: Int, missSorted: Int) {
            self.native = native
            self.fallback = fallback
            self.missGeometry = missGeometry
            self.missDtype = missDtype
            self.missSorted = missSorted
        }

        public var line: String {
            "gatherQmm native=\(native) fallback=\(fallback) geo=\(missGeometry) dtype=\(missDtype) sorted=\(missSorted)"
        }
    }

    public static func reset() {
        lock.lock()
        native = 0
        fallback = 0
        missGeometry = 0
        missDtype = 0
        missSorted = 0
        lock.unlock()
    }

    public static func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            native: native, fallback: fallback, missGeometry: missGeometry,
            missDtype: missDtype, missSorted: missSorted)
    }

    static func recordNative() {
        lock.lock()
        native += 1
        lock.unlock()
    }

    static func recordFallback() {
        lock.lock()
        fallback += 1
        lock.unlock()
    }

    static func recordMissGeometry() {
        lock.lock()
        missGeometry += 1
        lock.unlock()
    }

    static func recordMissDtype() {
        lock.lock()
        missDtype += 1
        lock.unlock()
    }

    static func recordMissSorted() {
        lock.lock()
        missSorted += 1
        lock.unlock()
    }
}

enum Qwen4ExpGatherQMM: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_GATHER_QMM"
    static let expertCount = 512
    static let minAssignments = 2048
    static let bits = 4
    static let groupSize = 64
    static let blockM = 32
    static let blockK = 32
    static let blockN = 32
    static let simdWidth = 32
    static let warpsM = 2
    static let warpsN = 2

    private static let lock = NSLock()
    nonisolated(unsafe) private static var disabled = false

    static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    static func matchesGeometry(assignments: Int, inputDim: Int, outputDim: Int, experts: Int)
        -> Bool
    {
        guard experts == expertCount else { return false }
        guard assignments >= minAssignments else { return false }
        guard inputDim > 0, outputDim > 0 else { return false }
        guard inputDim % groupSize == 0, inputDim % blockK == 0 else { return false }
        guard outputDim % blockN == 0 else { return false }
        // Flash-Next fused gate_up [512, 1280, 2560], split gate/up [512, 640, 2560],
        // down [512, 2560, 640].
        let fused = inputDim == 2560 && outputDim == 1280
        let split = inputDim == 2560 && outputDim == 640
        let down = inputDim == 640 && outputDim == 2560
        return fused || split || down
    }

    static func tryMatmul(
        x: MLXArray,
        indices: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        affineBiases: MLXArray?,
        sorted: Bool,
        bits: Int,
        groupSize: Int,
        mode: QuantizationMode
    ) -> MLXArray? {
        lock.lock()
        let dead = disabled || !isEnabled()
        lock.unlock()
        if dead { return nil }
        guard weight.ndim == 3, weight.dim(0) == expertCount else { return nil }

        guard let affineBiases,
            bits == Self.bits,
            groupSize == Self.groupSize,
            mode == .affine,
            weight.dtype == .uint32,
            weight.ndim == 3,
            scales.dtype == .bfloat16,
            affineBiases.dtype == .bfloat16
        else {
            Qwen4ExpGatherQMMInvocation.recordMissDtype()
            logMiss("quant/dtype bits=\(bits) gs=\(groupSize) mode=\(mode) w=\(weight.dtype)")
            return nil
        }
        let compute = scales.dtype
        let xCompute = x.dtype == compute ? x : x.asType(compute)
        let inputDim = xCompute.dim(-1)
        let assignments = xCompute.size / max(inputDim, 1)
        if assignments <= Qwen4ExpAffineQMV.maxAssignments,
            let y = Qwen4ExpAffineQMV.tryGather(
                x: xCompute,
                indices: indices,
                weight: weight,
                scales: scales,
                biases: affineBiases,
                bits: bits,
                groupSize: groupSize,
                environment: Qwen4ExpEnvironment.snapshot)
        {
            Qwen4ExpGatherQMMInvocation.recordNative()
            return y
        }
        if !sorted {
            // Decode top-10 that missed QMV geometry stay on stock gather.
            Qwen4ExpGatherQMMInvocation.recordMissSorted()
            return nil
        }
        guard builderKernel != nil, tileKernel != nil else {
            logMiss("kernel header missing")
            return nil
        }

        let experts = weight.dim(0)
        let outputDim = weight.dim(1)
        guard matchesGeometry(
            assignments: assignments, inputDim: inputDim, outputDim: outputDim, experts: experts)
        else {
            Qwen4ExpGatherQMMInvocation.recordMissGeometry()
            if assignments >= minAssignments {
                logMiss(
                    "geometry M=\(assignments) K=\(inputDim) N=\(outputDim) E=\(experts)")
            }
            return nil
        }
        let packedK = inputDim * bits / 32
        let groups = inputDim / groupSize
        guard weight.dim(2) == packedK,
            scales.ndim == 3, affineBiases.ndim == 3,
            scales.dim(0) == experts, scales.dim(1) == outputDim, scales.dim(2) == groups,
            affineBiases.shape == scales.shape,
            indices.size == assignments
        else {
            Qwen4ExpGatherQMMInvocation.recordMissGeometry()
            logMiss(
                "packed w=\(weight.shape) scales=\(scales.shape) idx=\(indices.size) M=\(assignments)"
            )
            return nil
        }

        let x2d = xCompute.ndim == 2 ? xCompute : xCompute.reshaped([assignments, inputDim])
        let idx = indices.dtype == .uint32 ? indices.flattened() : indices.asType(.uint32).flattened()
        let maxTiles = (assignments + blockM - 1) / blockM + expertCount - 1
        guard let builder = builderKernel, let tiles = tileKernel else { return nil }

        let built = builder(
            [idx, MLXArray(Int32(assignments))],
            grid: (expertCount, 1, 1),
            threadGroup: (expertCount, 1, 1),
            outputShapes: [[maxTiles, 4], [2]],
            outputDTypes: [.uint32, .uint32])
        guard built.count == 2 else { return nil }
        let descriptors = built[0]
        let count = built[1]
        let nTiles = outputDim / blockN
        let outputs = tiles(
            [
                x2d, weight, scales, affineBiases, descriptors, count,
                MLXArray(Int32(inputDim)),
                MLXArray(Int32(outputDim)),
            ],
            grid: (nTiles * simdWidth, maxTiles * warpsM, warpsN),
            threadGroup: (simdWidth, warpsN, warpsM),
            outputShapes: [[assignments, outputDim]],
            outputDTypes: [compute])
        guard var y = outputs.first else { return nil }
        if x.ndim != 2 {
            y = y.reshaped(Array(x.shape.dropLast()) + [outputDim])
        }
        y = Qwen4ExpActivation.nativeOutput(y, matching: x, compute: compute)
        Qwen4ExpGatherQMMInvocation.recordNative()
        return y
    }

    static func disable() {
        lock.lock()
        disabled = true
        lock.unlock()
    }

    private static let missLock = NSLock()
    nonisolated(unsafe) private static var didLogMiss = false

    private static func logMiss(_ message: String) {
        missLock.lock()
        defer { missLock.unlock() }
        if didLogMiss { return }
        didLogMiss = true
        #if DEBUG
            print("gatherQmm first-miss: \(message)")
        #endif
    }
}

private let builderKernel: MLXFast.MLXFastKernel? = {
    guard qwen4GatherQMMHeader() != nil else { return nil }
    return MLXFast.metalKernel(
        name: "qwen4_build_sorted_expert_tiles_e512",
        inputNames: ["indices", "M"],
        outputNames: ["descriptors", "count"],
        source: qwen4GatherBuilderSource,
        header: "",
        ensureRowContiguous: true)
}()

private let tileKernel: MLXFast.MLXFastKernel? = {
    guard let header = qwen4GatherQMMHeader() else { return nil }
    return MLXFast.metalKernel(
        name: "qwen4_gather_qmm_expert_tiles",
        inputNames: ["x", "w", "scales", "biases", "descriptors", "count", "K", "N"],
        outputNames: ["y"],
        source: qwen4GatherTileSource,
        header: header,
        ensureRowContiguous: true)
}()

private let qwen4GatherBuilderSource = """
    constexpr uint expert_count = 512;
    constexpr uint BM = 32;
    constexpr uint simdgroup_count = expert_count / 32;
    const int M_i = (int)M;
    uint lid = thread_index_in_threadgroup;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    threadgroup uint segment_starts[expert_count + 1];
    threadgroup uint inclusive_tile_offsets[expert_count];
    threadgroup uint violation_votes[simdgroup_count];

    int lower = 0;
    int upper = M_i;
    while (lower < upper) {
      const int midpoint = lower + (upper - lower) / 2;
      if (indices[midpoint] < lid) {
        lower = midpoint + 1;
      } else {
        upper = midpoint;
      }
    }
    segment_starts[lid] = uint(lower);
    if (lid == expert_count - 1) {
      segment_starts[expert_count] = uint(M_i);
    }

    bool boundary_ok = true;
    if (lower > 0) {
      boundary_ok = boundary_ok && indices[lower - 1] < lid;
    }
    if (lower < M_i) {
      boundary_ok = boundary_ok && indices[lower] >= lid;
    }
    bool adjacent_ok = true;
    for (int i = int(lid) + 1; i < M_i; i += int(expert_count)) {
      adjacent_ok = adjacent_ok && indices[i - 1] <= indices[i];
    }
    const uint violation_vote = simd_or((boundary_ok && adjacent_ok) ? 0u : 1u);
    if (simd_lid == 0) {
      violation_votes[simd_gid] = violation_vote;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool sorted_violation = false;
    for (uint group = 0; group < simdgroup_count; ++group) {
      sorted_violation = sorted_violation || violation_votes[group] != 0u;
    }
    if (lid == 0) {
      count[1] = sorted_violation ? 1u : 0u;
    }

    const uint segment_rows = segment_starts[lid + 1] - segment_starts[lid];
    inclusive_tile_offsets[lid] = (segment_rows + BM - 1) / BM;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < expert_count; stride <<= 1) {
      const uint addend = lid >= stride ? inclusive_tile_offsets[lid - stride] : 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lid >= stride) {
        inclusive_tile_offsets[lid] += addend;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint descriptor_count = inclusive_tile_offsets[expert_count - 1];
    if (lid == expert_count - 1) {
      count[0] = sorted_violation ? 0u : descriptor_count;
    }

    for (uint slot = lid; slot < descriptor_count; slot += expert_count) {
      uint expert_lower = 0;
      uint expert_upper = expert_count;
      while (expert_lower < expert_upper) {
        const uint midpoint = expert_lower + (expert_upper - expert_lower) / 2;
        if (inclusive_tile_offsets[midpoint] <= slot) {
          expert_lower = midpoint + 1;
        } else {
          expert_upper = midpoint;
        }
      }
      const uint expert = expert_lower;
      const uint expert_tile_begin =
          expert == 0 ? 0 : inclusive_tile_offsets[expert - 1];
      const uint row =
          segment_starts[expert] + (slot - expert_tile_begin) * BM;
      const uint row_count = min(BM, segment_starts[expert + 1] - row);
      descriptors[slot * 4 + 0] = row;
      descriptors[slot * 4 + 1] = row_count;
      descriptors[slot * 4 + 2] = expert;
      descriptors[slot * 4 + 3] = 0;
    }
    """

private let qwen4GatherTileSource = """
    constexpr int BM = 32;
    constexpr int BK = 32;
    constexpr int BN = 32;
    constexpr int group_size = 64;
    constexpr int bits = 4;
    using T = bfloat16_t;
    constexpr int pack_factor = get_pack_factor<bits, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
    constexpr int BK_padded = BK + 16 / sizeof(T);
    threadgroup T Xs[BM * BK_padded];
    threadgroup T Ws[BN * BK_padded];

    const uint descriptor_count = count[0];
    const uint slot = threadgroup_position_in_grid.y;
    if (slot >= descriptor_count) {
      return;
    }

    const int K_i = (int)K;
    const int N_i = (int)N;
    const uint row = descriptors[slot * 4 + 0];
    const int row_count = int(descriptors[slot * 4 + 1]);
    const size_t expert = size_t(descriptors[slot * 4 + 2]);
    const int K_w = K_i * bytes_per_pack / pack_factor;
    const int K_g = K_i / group_size;
    const size_t expert_w_stride = size_t(N_i) * size_t(K_w);
    const size_t expert_sb_stride = size_t(N_i) * size_t(K_g);

    const device T* x_e = x + size_t(row) * size_t(K_i);
    device T* y_e = y + size_t(row) * size_t(N_i);
    const device uint8_t* expert_w =
        reinterpret_cast<const device uint8_t*>(w) + expert * expert_w_stride;
    const device T* scales_e = scales + expert * expert_sb_stride;
    const device T* biases_e = biases + expert * expert_sb_stride;
    const uint3 local_tid = uint3(threadgroup_position_in_grid.x, 0, 0);

    if (row_count <= 16) {
      qwen4_qmm_t_i32<T, group_size, bits, 16, BK, BN>(
          reinterpret_cast<const device uint32_t*>(expert_w),
          scales_e, biases_e, x_e, y_e, Xs, Ws, K_i, N_i, row_count,
          local_tid,
          simdgroup_index_in_threadgroup,
          thread_index_in_simdgroup);
    } else {
      qwen4_qmm_t_i32<T, group_size, bits, 32, BK, BN>(
          reinterpret_cast<const device uint32_t*>(expert_w),
          scales_e, biases_e, x_e, y_e, Xs, Ws, K_i, N_i, row_count,
          local_tid,
          simdgroup_index_in_threadgroup,
          thread_index_in_simdgroup);
    }
    """

private let qwen4GatherQmmValueWrapper = """

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM,
    const int BK,
    const int BN>
METAL_FUNC void qwen4_qmm_t_i32(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

"""

private func qwen4GatherQMMHeader() -> String? {
    let gemm = Qwen4ExpMetalHeaders.gemm
    let quantizedUtils = Qwen4ExpMetalHeaders.quantizedUtils
    var quantized = Qwen4ExpMetalHeaders.quantized
    if let kernel = quantized.range(
        of: "\ntemplate <typename T, int group_size, int bits, int D, bool batched>\n[[kernel]]")
        ?? quantized.range(of: "\n[[kernel]]")
    {
        quantized = String(quantized[..<kernel.lowerBound])
    }
    return gemm + "\n" + quantizedUtils + "\n" + quantized + qwen4GatherQmmValueWrapper
}
