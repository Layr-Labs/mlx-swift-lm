// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion `qwen3_5_target_verify_qmv` (Apache-2.0) for Qwen4-only affine
// decode: T=1 serial and Lightning MTP verify widths S=2..15. Variant-8
// BM=64 is a prefill tile; M=1 on that tile is wrong/slow, which is why
// `Qwen4ExpAffineQMM.minTokens` stays 2048. This kernel is the lossless
// quantized GEMV those engines hold at 55–70 tok/s. Same nibble qdot as
// `Qwen4ExpHCHybrid` / MLX `quantized.h` (Apple MIT). Unsorted MoE gather
// (decode top-10) uses the expert-indexed variant instead of the sorted
// BM=32 prefill tile. Kill: DARKBLOOM_QWEN4_AFFINE_QMV=0.

import Foundation
import MLX
import MLXFast
import MLXNN

public enum Qwen4ExpAffineQMVInvocation: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var native = 0
    nonisolated(unsafe) private static var gatherNative = 0

    public static func reset() {
        lock.lock()
        native = 0
        gatherNative = 0
        lock.unlock()
    }

    static func recordNative() {
        lock.lock()
        native += 1
        lock.unlock()
    }

    static func recordGatherNative() {
        lock.lock()
        gatherNative += 1
        lock.unlock()
    }

    public static func snapshot() -> (native: Int, gatherNative: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (native, gatherNative)
    }
}

public enum Qwen4ExpAffineQMV: Sendable {
    public static let envFlag = "DARKBLOOM_QWEN4_AFFINE_QMV"
    /// Serial decode plus Lightning MTP verify (S=2..15). Prefill stays on
    /// the BM=64 / BM=32 tiles.
    public static let maxTokens = 16
    static let maxAssignments = 256
    static let simdWidth = 32
    static let resultsPerSimdgroup = 4
    static let simdgroups = 2
    static let blockN = resultsPerSimdgroup * simdgroups
    static let fastPacksPerThread = 2

    private static let lock = NSLock()
    nonisolated(unsafe) private static var disabled = false

    public static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    static func disable() {
        lock.lock()
        disabled = true
        lock.unlock()
    }

    /// `DARKBLOOM_QWEN4_QMV_LOG_SHAPES=1`: print each distinct dispatch
    /// geometry once (diagnostic; off by default).
    #if DEBUG
        public static let logShapes = Qwen4ExpEnvironment.snapshot["DARKBLOOM_QWEN4_QMV_LOG_SHAPES"] == "1"
    #else
        public static let logShapes = false
    #endif
    #if DEBUG
        nonisolated(unsafe) private static var loggedShapes: Set<String> = []
    #endif

    public static func logShapeOnce(_ line: String) {
        #if DEBUG
            lock.lock()
            let inserted = loggedShapes.insert(line).inserted
            lock.unlock()
            if inserted {
                FileHandle.standardError.write(("[qwen4-qmv-shape] " + line + "\n").data(using: .utf8)!)
            }
        #endif
    }

    static func packFactor(_ bits: Int) -> Int {
        bits == 5 ? 8 : (bits == 6 ? 4 : 32 / bits)
    }

    static func bytesPerPack(_ bits: Int) -> Int {
        let powerOfTwo = (bits & (bits - 1)) == 0
        return powerOfTwo ? 4 : (bits == 5 ? 5 : 3)
    }

    /// MLX dispatches `qmv_fast` (two packs per lane, 512-value blocks for
    /// 4-bit) only when `N % 8 == 0 && K % fastAlignment == 0`; otherwise the
    /// non-fast `qmv` walks K with ONE pack per lane (256-value blocks). The
    /// two orders round differently (layer-7 block_inject, N=4: 1 ULP on a
    /// real row, enough to flip a greedy token downstream). Mirror the
    /// dispatch so every shape reproduces stock bit for bit.
    public static func packsPerThread(inputDim: Int, outputDim: Int, bits: Int) -> Int {
        // DIAGNOSTIC ONLY (non-exact): DARKBLOOM_QWEN4_QMV_FORCE_FAST=1 keeps
        // the two-pack layout on every shape to measure the non-fast cost.
        #if DEBUG
            let force = Qwen4ExpEnvironment.snapshot["DARKBLOOM_QWEN4_QMV_FORCE_FAST"]
        #else
            let force: String? = nil
        #endif
        if force == "1" { return 2 }
        let fastAlignment = packFactor(bits) * 2 * 32
        // DIAGNOSTIC ONLY (non-exact vs stock): "k" keeps the two-pack layout
        // on K-misaligned shapes whose rows are 8-aligned (HC up K=320,
        // down projections K=640) while N tails stay non-fast.
        if force == "k", outputDim % blockN == 0 { return 2 }
        return (outputDim % blockN == 0 && inputDim % fastAlignment == 0) ? 2 : 1
    }

    public static func matchesDecodeGeometry(
        tokens: Int, inputDim: Int, outputDim: Int, bits: Int, groupSize: Int
    ) -> Bool {
        guard tokens >= 1, tokens <= maxTokens else { return false }
        guard bits == 4 || bits == 5 || bits == 6 || bits == 8 else { return false }
        guard groupSize == 64 || groupSize == 128 else { return false }
        guard inputDim > 0, outputDim > 0 else { return false }
        // Any N: the kernels tile N in BN-row tiles and guard the tail rows
        // (Flash-Next has N=1 shared_expert_gate and N=4 block_inject banks
        // that must take this exact path at verify width too).
        #if DEBUG
            if Qwen4ExpEnvironment.snapshot["DARKBLOOM_QWEN4_QMV_ANY_N"] == "0",
                outputDim % blockN != 0
            {
                return false
            }
        #endif
        guard inputDim % groupSize == 0 else { return false }
        let vpt = packFactor(bits) * packsPerThread(inputDim: inputDim, outputDim: outputDim, bits: bits)
        return inputDim % vpt == 0 && groupSize % vpt == 0
    }

    public static func tryMatmul(
        x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        bits: Int,
        groupSize: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray? {
        lock.lock()
        let dead = disabled || !isEnabled(environment: environment)
        lock.unlock()
        if dead { return nil }
        guard let kernel = qmvKernel else { return nil }
        guard weight.ndim == 2, scales.ndim == 2, biases.ndim == 2 else { return nil }
        let compute = scales.dtype
        guard compute == .bfloat16 || compute == .float16,
            biases.dtype == compute,
            x.dtype == compute || x.dtype == .bfloat16 || x.dtype == .float16
        else { return nil }
        let xCompute = x.dtype == compute ? x : x.asType(compute)
        guard xCompute.ndim >= 2 else { return nil }
        let inputDim = xCompute.dim(-1)
        let tokens = xCompute.dim(-2)
        let outputDim = weight.dim(0)
        guard matchesDecodeGeometry(
            tokens: tokens, inputDim: inputDim, outputDim: outputDim, bits: bits,
            groupSize: groupSize)
        else { return nil }
        let packedK = inputDim * bits / 32
        let groups = inputDim / groupSize
        guard weight.dtype == .uint32,
            weight.dim(1) == packedK,
            scales.dim(0) == outputDim, scales.dim(1) == groups,
            biases.shape == scales.shape
        else { return nil }

        let leading = Array(xCompute.shape.dropLast())
        let rows = xCompute.size / inputDim
        guard rows >= 1, rows <= maxTokens else { return nil }
        if Self.logShapes {
            Self.logShapeOnce(
                "qmv T=\(rows) K=\(inputDim) N=\(outputDim) bits=\(bits) gs=\(groupSize)")
        }
        let x2d = xCompute.ndim == 2 ? xCompute : xCompute.reshaped([rows, inputDim])
        let nTiles = (outputDim + blockN - 1) / blockN
        let outputs = kernel(
            [x2d, weight, scales, biases],
            template: [
                ("T", compute),
                ("BITS", bits),
                ("GS", groupSize),
                ("K_SIZE", inputDim),
                ("N_SIZE", outputDim),
                ("PACKS_PER_THREAD", Self.packsPerThread(inputDim: inputDim, outputDim: outputDim, bits: bits)),
            ],
            grid: (simdWidth, simdgroups * nTiles, rows),
            threadGroup: (simdWidth, simdgroups, 1),
            outputShapes: [[rows, outputDim]],
            outputDTypes: [compute])
        guard var y = outputs.first else { return nil }
        if leading.count != 1 {
            y = y.reshaped(leading + [outputDim])
        }
        y = Qwen4ExpActivation.nativeOutput(y, matching: x, compute: compute)
        Qwen4ExpAffineQMVInvocation.recordNative()
        return y
    }

    public static func tryGather(
        x: MLXArray,
        indices: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        bits: Int,
        groupSize: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray? {
        lock.lock()
        let dead = disabled || !isEnabled(environment: environment)
        lock.unlock()
        if dead { return nil }
        guard let kernel = gatherQmvKernel else { return nil }
        guard weight.ndim == 3, scales.ndim == 3, biases.ndim == 3 else { return nil }
        let compute = scales.dtype
        guard compute == .bfloat16 || compute == .float16,
            biases.dtype == compute
        else { return nil }
        let xCompute = x.dtype == compute ? x : x.asType(compute)
        guard xCompute.ndim >= 2 else { return nil }
        let inputDim = xCompute.dim(-1)
        let outputDim = weight.dim(1)
        let experts = weight.dim(0)
        let assignments = xCompute.size / inputDim
        guard assignments >= 1, assignments <= maxAssignments else { return nil }
        guard matchesDecodeGeometry(
            tokens: 1, inputDim: inputDim, outputDim: outputDim, bits: bits,
            groupSize: groupSize)
        else { return nil }
        let packedK = inputDim * bits / 32
        let groups = inputDim / groupSize
        guard weight.dtype == .uint32,
            weight.dim(2) == packedK,
            scales.dim(0) == experts, scales.dim(1) == outputDim, scales.dim(2) == groups,
            biases.shape == scales.shape,
            indices.size == assignments
        else { return nil }

        let leading = Array(xCompute.shape.dropLast())
        let x2d = xCompute.ndim == 2 ? xCompute : xCompute.reshaped([assignments, inputDim])
        let idx =
            indices.dtype == .uint32 ? indices.flattened() : indices.asType(.uint32).flattened()
        let nTiles = (outputDim + blockN - 1) / blockN
        let outputs = kernel(
            [x2d, idx, weight, scales, biases],
            template: [
                ("T", compute),
                ("BITS", bits),
                ("GS", groupSize),
                ("K_SIZE", inputDim),
                ("N_SIZE", outputDim),
                ("PACKS_PER_THREAD", Self.packsPerThread(inputDim: inputDim, outputDim: outputDim, bits: bits)),
            ],
            grid: (simdWidth, simdgroups * nTiles, assignments),
            threadGroup: (simdWidth, simdgroups, 1),
            outputShapes: [[assignments, outputDim]],
            outputDTypes: [compute])
        guard var y = outputs.first else { return nil }
        if leading.count != 1 {
            y = y.reshaped(leading + [outputDim])
        }
        y = Qwen4ExpActivation.nativeOutput(y, matching: x, compute: compute)
        Qwen4ExpAffineQMVInvocation.recordGatherNative()
        return y
    }
}

private let qmvKernel: MLXFast.MLXFastKernel? = MLXFast.metalKernel(
    name: "qwen4_affine_qmv",
    inputNames: ["x", "w", "scales", "biases"],
    outputNames: ["y"],
    source: qwen4AffineQMVSource,
    header: qwen4AffineQMVHeader,
    ensureRowContiguous: true)

private let gatherQmvKernel: MLXFast.MLXFastKernel? = MLXFast.metalKernel(
    name: "qwen4_gather_qmv",
    inputNames: ["x", "indices", "w", "scales", "biases"],
    outputNames: ["y"],
    source: qwen4GatherQMVSource,
    header: qwen4AffineQMVHeader,
    ensureRowContiguous: true)

// MLX is Copyright © 2023 Apple Inc. and licensed under the MIT License.
// Transcribed from mlx/backend/metal/kernels/quantized.h (qmv / qdot)
// and Fusion qwen3_5_target_verify_qmv. Arithmetic is the stock affine
// nibble reconstruction — bit-identical to QuantizedLinear / gather QMM.
private let qwen4AffineQMVHeader = """
    using namespace metal;

    template <int bits>
    inline constexpr short qmv_pack_factor() {
        return bits == 5 ? 8 : (bits == 6 ? 4 : 32 / bits);
    }

    template <int bits>
    inline constexpr short qmv_bytes_per_pack() {
        constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
        return power_of_2_bits ? 4 : (bits == 5 ? 5 : 3);
    }

    template <typename T, int N, int bits>
    inline float qmv_load_vector(const device T* x, thread float* xt) {
        float sum = 0.0f;
        if (bits == 4) {
            for (int i = 0; i < N; i += 4) {
                sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
                xt[i] = x[i];
                xt[i + 1] = x[i + 1] / 16.0f;
                xt[i + 2] = x[i + 2] / 256.0f;
                xt[i + 3] = x[i + 3] / 4096.0f;
            }
        } else if (bits == 5) {
            for (int i = 0; i < N; i += 8) {
                sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
                    + x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
                xt[i] = x[i];
                xt[i + 1] = x[i + 1] / 32.0f;
                xt[i + 2] = x[i + 2] / 4.0f;
                xt[i + 3] = x[i + 3] / 128.0f;
                xt[i + 4] = x[i + 4] / 16.0f;
                xt[i + 5] = x[i + 5] / 2.0f;
                xt[i + 6] = x[i + 6] / 64.0f;
                xt[i + 7] = x[i + 7] / 8.0f;
            }
        } else if (bits == 6) {
            for (int i = 0; i < N; i += 4) {
                sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
                xt[i] = x[i];
                xt[i + 1] = x[i + 1] / 64.0f;
                xt[i + 2] = x[i + 2] / 16.0f;
                xt[i + 3] = x[i + 3] / 4.0f;
            }
        } else if (bits == 8) {
            for (int i = 0; i < N; ++i) {
                sum += x[i];
                xt[i] = x[i];
            }
        }
        return sum;
    }

    template <typename T, int N, int bits>
    inline float qmv_load_vector_bounded(
        const device T* x_row, int col, int K, thread float* xt) {
        T tmp[N];
        for (int i = 0; i < N; ++i) {
            tmp[i] = (col + i < K) ? x_row[col + i] : T(0);
        }
        float sum = 0.0f;
        if (bits == 4) {
            for (int i = 0; i < N; i += 4) {
                sum += tmp[i] + tmp[i + 1] + tmp[i + 2] + tmp[i + 3];
                xt[i] = tmp[i];
                xt[i + 1] = tmp[i + 1] / 16.0f;
                xt[i + 2] = tmp[i + 2] / 256.0f;
                xt[i + 3] = tmp[i + 3] / 4096.0f;
            }
        } else if (bits == 5) {
            for (int i = 0; i < N; i += 8) {
                sum += tmp[i] + tmp[i + 1] + tmp[i + 2] + tmp[i + 3]
                    + tmp[i + 4] + tmp[i + 5] + tmp[i + 6] + tmp[i + 7];
                xt[i] = tmp[i];
                xt[i + 1] = tmp[i + 1] / 32.0f;
                xt[i + 2] = tmp[i + 2] / 4.0f;
                xt[i + 3] = tmp[i + 3] / 128.0f;
                xt[i + 4] = tmp[i + 4] / 16.0f;
                xt[i + 5] = tmp[i + 5] / 2.0f;
                xt[i + 6] = tmp[i + 6] / 64.0f;
                xt[i + 7] = tmp[i + 7] / 8.0f;
            }
        } else if (bits == 6) {
            for (int i = 0; i < N; i += 4) {
                sum += tmp[i] + tmp[i + 1] + tmp[i + 2] + tmp[i + 3];
                xt[i] = tmp[i];
                xt[i + 1] = tmp[i + 1] / 64.0f;
                xt[i + 2] = tmp[i + 2] / 16.0f;
                xt[i + 3] = tmp[i + 3] / 4.0f;
            }
        } else if (bits == 8) {
            for (int i = 0; i < N; ++i) {
                sum += tmp[i];
                xt[i] = tmp[i];
            }
        }
        return sum;
    }

    template <int N, int bits>
    inline float qmv_qdot(
        const device uint8_t* w,
        const thread float* xt,
        float scale,
        float bias,
        float sum) {
        float accum = 0.0f;
        if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < N / 4; ++i) {
                accum +=
                    (xt[4 * i] * (ws[i] & 0x000f)
                     + xt[4 * i + 1] * (ws[i] & 0x00f0)
                     + xt[4 * i + 2] * (ws[i] & 0x0f00)
                     + xt[4 * i + 3] * (ws[i] & 0xf000));
            }
        } else if (bits == 5) {
            for (int i = 0; i < N / 8; ++i) {
                const thread float* x8 = xt + 8 * i;
                const device uint8_t* wb = w + 5 * i;
                accum += (wb[0] & 0x1f) * x8[0];
                accum += (wb[0] & 0xe0) * x8[1];
                accum += (wb[1] & 0x3) * (x8[1] * 256.0f);
                accum += (wb[1] & 0x7c) * x8[2];
                accum += (wb[1] & 0x80) * x8[3];
                accum += (wb[2] & 0xf) * (x8[3] * 256.0f);
                accum += (wb[2] & 0xf0) * x8[4];
                accum += (wb[3] & 0x1) * (x8[4] * 256.0f);
                accum += (wb[3] & 0x3e) * x8[5];
                accum += (wb[3] & 0xc0) * x8[6];
                accum += (wb[4] & 0x7) * (x8[6] * 256.0f);
                accum += (wb[4] & 0xf8) * x8[7];
            }
        } else if (bits == 6) {
            for (int i = 0; i < N / 4; ++i) {
                const thread float* x4 = xt + 4 * i;
                const device uint8_t* wb = w + 3 * i;
                accum += (wb[0] & 0x3f) * x4[0];
                accum += (wb[0] & 0xc0) * x4[1];
                accum += (wb[1] & 0x0f) * (x4[1] * 256.0f);
                accum += (wb[1] & 0xf0) * x4[2];
                accum += (wb[2] & 0x03) * (x4[2] * 256.0f);
                accum += (wb[2] & 0xfc) * x4[3];
            }
        } else if (bits == 8) {
            for (int i = 0; i < N; ++i) accum += xt[i] * w[i];
        }
        return scale * accum + sum * bias;
    }
    """

// Tokens ride the grid z-axis: every column runs the SAME kernel binary
// MLX's qmv would, so a verify window is bit-identical to serial decode by
// construction. A token-folded variant (one weight pass, tokens looped
// in-thread) was tried on 2026-09-05: its per-width template instantiations
// let the Metal compiler contract FMAs differently and one element in ~3M
// diverged at width 2 (K=10240, N=320) — enough to flip a greedy token.
// The live model showed no speed gain from the fold either. N tails
// (Flash-Next N=1 shared_expert_gate, N=4 block_inject) are guarded so those
// banks take this exact path at every width.
private let qwen4AffineQMVSource = """
    constexpr int PF = qmv_pack_factor<BITS>();
    constexpr int BP = qmv_bytes_per_pack<BITS>();
    constexpr int PPT = PACKS_PER_THREAD;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SCALE_STEP = GS / VPT;
    constexpr int BN = 8;
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(sg) * 4;
    if (out_row >= N_SIZE) {
      return;
    }
    int in_vec_size_w = K_SIZE * BP / PF;
    int in_vec_size_g = K_SIZE / GS;
    const device uint8_t* ws = (const device uint8_t*)w
        + out_row * in_vec_size_w + int(lane) * PPT * BP;
    auto sc = scales + out_row * in_vec_size_g + int(lane) / SCALE_STEP;
    auto bs = biases + out_row * in_vec_size_g + int(lane) / SCALE_STEP;
    const device T* x_row = x + int(b_idx) * K_SIZE;
    const device T* xp = x_row + int(lane) * VPT;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float xv[VPT];
    int k = 0;
    for (; k + BLOCK <= K_SIZE; k += BLOCK) {
      float sum = qmv_load_vector<T, VPT, BITS>(xp, xv);
      for (int row = 0; row < 4; ++row) {
        if (out_row + row >= N_SIZE) break;
        result[row] += qmv_qdot<VPT, BITS>(
            ws + row * in_vec_size_w, xv,
            float(sc[row * in_vec_size_g]), float(bs[row * in_vec_size_g]), sum);
      }
      ws += BLOCK * BP / PF;
      sc += BLOCK / GS;
      bs += BLOCK / GS;
      xp += BLOCK;
    }
    if (k < K_SIZE) {
      const int col = k + int(lane) * VPT;
      if (col < K_SIZE) {
        float sum = qmv_load_vector_bounded<T, VPT, BITS>(x_row, col, K_SIZE, xv);
        for (int row = 0; row < 4; ++row) {
          if (out_row + row >= N_SIZE) break;
          result[row] += qmv_qdot<VPT, BITS>(
              ws + row * in_vec_size_w, xv,
              float(sc[row * in_vec_size_g]), float(bs[row * in_vec_size_g]), sum);
        }
      }
    }
    for (int row = 0; row < 4; ++row) {
      int n = out_row + row;
      float r = simd_sum(result[row]);
      if (lane == 0 && n < N_SIZE) {
        y[int(b_idx) * N_SIZE + n] = T(r);
      }
    }
    """

private let qwen4GatherQMVSource = """
    constexpr int PF = qmv_pack_factor<BITS>();
    constexpr int BP = qmv_bytes_per_pack<BITS>();
    constexpr int PPT = PACKS_PER_THREAD;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SCALE_STEP = GS / VPT;
    constexpr int BN = 8;
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(sg) * 4;
    if (out_row >= N_SIZE) {
      return;
    }
    const uint expert = indices[b_idx];
    int in_vec_size_w = K_SIZE * BP / PF;
    int in_vec_size_g = K_SIZE / GS;
    const size_t expert_w_stride = size_t(N_SIZE) * size_t(in_vec_size_w);
    const size_t expert_g_stride = size_t(N_SIZE) * size_t(in_vec_size_g);
    const device uint8_t* ws = (const device uint8_t*)w
        + size_t(expert) * expert_w_stride
        + out_row * in_vec_size_w + int(lane) * PPT * BP;
    auto sc = scales + size_t(expert) * expert_g_stride
        + out_row * in_vec_size_g + int(lane) / SCALE_STEP;
    auto bs = biases + size_t(expert) * expert_g_stride
        + out_row * in_vec_size_g + int(lane) / SCALE_STEP;
    const device T* x_row = x + int(b_idx) * K_SIZE;
    const device T* xp = x_row + int(lane) * VPT;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float xv[VPT];
    int k = 0;
    for (; k + BLOCK <= K_SIZE; k += BLOCK) {
      float sum = qmv_load_vector<T, VPT, BITS>(xp, xv);
      for (int row = 0; row < 4; ++row) {
        if (out_row + row >= N_SIZE) break;
        result[row] += qmv_qdot<VPT, BITS>(
            ws + row * in_vec_size_w, xv,
            float(sc[row * in_vec_size_g]), float(bs[row * in_vec_size_g]), sum);
      }
      ws += BLOCK * BP / PF;
      sc += BLOCK / GS;
      bs += BLOCK / GS;
      xp += BLOCK;
    }
    if (k < K_SIZE) {
      const int col = k + int(lane) * VPT;
      if (col < K_SIZE) {
        float sum = qmv_load_vector_bounded<T, VPT, BITS>(x_row, col, K_SIZE, xv);
        for (int row = 0; row < 4; ++row) {
          if (out_row + row >= N_SIZE) break;
          result[row] += qmv_qdot<VPT, BITS>(
              ws + row * in_vec_size_w, xv,
              float(sc[row * in_vec_size_g]), float(bs[row * in_vec_size_g]), sum);
        }
      }
    }
    for (int row = 0; row < 4; ++row) {
      int n = out_row + row;
      float r = simd_sum(result[row]);
      if (lane == 0 && n < N_SIZE) {
        y[int(b_idx) * N_SIZE + n] = T(r);
      }
    }
    """
