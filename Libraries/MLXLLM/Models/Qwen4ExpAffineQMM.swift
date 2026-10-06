// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion `qwen35_q*_affine_qmm_t` variant 8 (BM=64, BK=32, BN=64, WM=2, WN=2)
// for Qwen4-only 2D QuantizedLinear prefill. Same `qmm_t_impl` steel tile
// Fusion compiles into its metallib; `MLXFast.metalKernel` cannot `#include`
// mlx headers, so this concatenates the already-inlined mlx-generated
// gemm + quantized_utils + quantized preambles packaged with this SDK
// and instantiates the Fusion tile (plus BN=32 for fused GDN
// N=16480, which is 32-aligned but not 64-aligned).
//
// Mixed bits {4,5,6,8}, group {64,128}, bf16, affine, no bias, M>=2048
// (W8 stays at Fusion's 16384 floor + variant-8 BM=64). Listing leftover
// at chunk 8192 is shared-expert W8 `T=8192 K=2560 N=640 gs=128`.
// `DARKBLOOM_QWEN4_AFFINE_Q8_BM=128` is Fusion variant 9 (BM=128 BK=32
// BN=64) and lowers the W8 floor to 8192. BM=64 at that floor was
// +308 ms at 50K — do not default that. HC `block_inject_weight` is N=4
// (bits=5) and cannot tile BN=32/64; pad N=4 to 32 and slice. Unfused
// GDN N=48 pad-to-64 was +176 ms at 128K vs stock. Chunk 16384
// OOM-killed at 128K (exit 137) even with bf16; leave 8192. Kill:
// DARKBLOOM_QWEN4_AFFINE_QMM=0. Lab restore unpadded inject:
// DARKBLOOM_QWEN4_AFFINE_INJECT_PAD=0. 27B `qwen3_5` never calls this.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum Qwen4ExpAffineQMMInvocation: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var native = 0
    nonisolated(unsafe) private static var fallback = 0
    nonisolated(unsafe) private static var missGeometry = 0
    nonisolated(unsafe) private static var missDtype = 0
    nonisolated(unsafe) private static var missType = 0
    nonisolated(unsafe) private static var missPacked = 0

    public struct Snapshot: Sendable, Equatable {
        public var native: Int
        public var fallback: Int
        public var missGeometry: Int
        public var missDtype: Int
        public var missType: Int
        public var missPacked: Int

        public init(
            native: Int, fallback: Int, missGeometry: Int, missDtype: Int, missType: Int,
            missPacked: Int
        ) {
            self.native = native
            self.fallback = fallback
            self.missGeometry = missGeometry
            self.missDtype = missDtype
            self.missType = missType
            self.missPacked = missPacked
        }

        public var line: String {
            "affineQmm native=\(native) fallback=\(fallback) geo=\(missGeometry) dtype=\(missDtype) type=\(missType) packed=\(missPacked)"
        }
    }

    public static func reset() {
        lock.lock()
        native = 0
        fallback = 0
        missGeometry = 0
        missDtype = 0
        missType = 0
        missPacked = 0
        lock.unlock()
    }

    public static func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            native: native, fallback: fallback, missGeometry: missGeometry,
            missDtype: missDtype, missType: missType, missPacked: missPacked)
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

    static func recordMissType() {
        lock.lock()
        missType += 1
        lock.unlock()
    }

    static func recordMissPacked() {
        lock.lock()
        missPacked += 1
        lock.unlock()
    }
}

enum Qwen4ExpAffineQMM: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_AFFINE_QMM"
    static let q8MinTokensEnv = "DARKBLOOM_QWEN4_AFFINE_Q8_MIN_TOKENS"
    static let q8BlockMEnv = "DARKBLOOM_QWEN4_AFFINE_Q8_BM"
    static let injectPadEnvFlag = "DARKBLOOM_QWEN4_AFFINE_INJECT_PAD"
    static let minTokens = 2048
    /// Decode is T=1. Prefill stays behind this floor (BM=64 variant-8).
    /// M=1 / MTP S=2..15 use `Qwen4ExpAffineQMV` instead of lowering it.
    static let q8MinTokens = 16384
    static let q8WideMinTokens = 8192
    static let blockM = 64
    static let wideBlockM = 128
    static let blockK = 32
    static let blockN = 64
    static let narrowN = 32
    static let injectOutputDim = 4
    static let simdWidth = 32
    static let warpsM = 2
    static let warpsN = 2

    private static let lock = NSLock()
    nonisolated(unsafe) private static var disabled = false
    private static let paddedInject = Qwen4ExpPaddedProjectionCache()

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

    /// HC inject is N=4 (pad to 32). Unfused GDN N=48 pad-to-64 was
    /// +176 ms at 128K vs stock; do not default. Default on. Restore: `=0`.
    static func injectPadEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[injectPadEnvFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    static func q8Floor(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        let raw = environment[q8MinTokensEnv]?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let raw, let parsed = Int(raw), parsed > 0 {
            return parsed
        }
        return q8TileM(environment: environment) == wideBlockM ? q8WideMinTokens : q8MinTokens
    }

    /// Fusion variant 9 for W8 only. Default stays variant 8 (BM=64).
    static func q8TileM(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        let raw = environment[q8BlockMEnv]?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let raw, let parsed = Int(raw), parsed == wideBlockM {
            return wideBlockM
        }
        return blockM
    }

    static func tileM(
        bits: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        bits == 8 ? q8TileM(environment: environment) : blockM
    }

    static func padTarget(
        for outputDim: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int? {
        guard injectPadEnabled(environment: environment) else { return nil }
        if outputDim == injectOutputDim { return narrowN }
        return nil
    }

    static func shouldPad(
        outputDim: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        padTarget(for: outputDim, environment: environment) != nil
    }

    static func tokenFloor(
        bits: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        bits == 8 ? q8Floor(environment: environment) : minTokens
    }

    static func blockN(for outputDim: Int) -> Int? {
        if outputDim % blockN == 0 { return blockN }
        if outputDim % narrowN == 0 { return narrowN }
        return nil
    }

    static func matchesGeometry(
        tokens: Int, inputDim: Int, outputDim: Int, bits: Int, groupSize: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        guard tokens >= tokenFloor(bits: bits, environment: environment) else { return false }
        guard bits == 4 || bits == 5 || bits == 6 || bits == 8 else { return false }
        guard groupSize == 64 || groupSize == 128 else { return false }
        guard inputDim > 0, outputDim > 0 else { return false }
        guard inputDim % 64 == 0, inputDim % groupSize == 0, inputDim % blockK == 0 else {
            return false
        }
        if blockN(for: outputDim) != nil { return true }
        return shouldPad(outputDim: outputDim, environment: environment)
    }

    /// Qwen4-only 2D affine qmm. Ineligible shapes and the kill switch fall
    /// through to stock `QuantizedLinear` / `Linear`.
    static func apply(
        _ linear: Linear, _ x: MLXArray,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray {
        if let y = tryMatmul(x, linear, environment: environment) {
            Qwen4ExpAffineQMMInvocation.recordNative()
            return y
        }
        if x.ndim >= 2, x.dim(-2) >= minTokens {
            Qwen4ExpAffineQMMInvocation.recordFallback()
        }
        return Qwen4ExpActivation.keep(linear(x))
    }

    static func applyMLP(gate: Linear, up: Linear, down: Linear, x: MLXArray) -> MLXArray {
        let gated = silu(apply(gate, x)) * apply(up, x)
        return apply(down, gated)
    }

    static func tryMatmul(
        _ x: MLXArray, _ linear: Linear,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray? {
        lock.lock()
        let dead = disabled || !isEnabled(environment: environment)
        lock.unlock()
        if dead { return nil }
        guard let quantized = linear as? QuantizedLinear,
            ObjectIdentifier(type(of: quantized)) == ObjectIdentifier(QuantizedLinear.self),
            quantized.bias == nil,
            quantized.mode == .affine,
            let affineBiases = quantized.biases,
            x.ndim >= 2
        else {
            Qwen4ExpAffineQMMInvocation.recordMissType()
            logMiss(
                "not affine QuantizedLinear dtype=\(x.dtype) ndim=\(x.ndim) type=\(type(of: linear))")
            return nil
        }

        let packedWeight = quantized.weight
        let packedScales = quantized.scales
        let compute = packedScales.dtype
        guard compute == .bfloat16 || compute == .float16 else {
            Qwen4ExpAffineQMMInvocation.recordMissDtype()
            logMiss("scale dtype \(compute)")
            return nil
        }
        let xCompute = x.dtype == compute ? x : x.asType(compute)
        let inputDim = xCompute.dim(-1)
        let tokens = xCompute.dim(-2)
        let outputDim = quantized.shape.0
        if tokens < tokenFloor(bits: quantized.bits, environment: environment),
            let y = Qwen4ExpAffineQMV.tryMatmul(
                x: xCompute,
                weight: packedWeight,
                scales: packedScales,
                biases: affineBiases,
                bits: quantized.bits,
                groupSize: quantized.groupSize,
                environment: environment)
        {
            return y
        }
        guard let kernel = affineQMMKernel else {
            logMiss("kernel header missing")
            return nil
        }
        guard matchesGeometry(
            tokens: tokens, inputDim: inputDim, outputDim: outputDim,
            bits: quantized.bits, groupSize: quantized.groupSize,
            environment: environment)
        else {
            Qwen4ExpAffineQMMInvocation.recordMissGeometry()
            if tokens >= minTokens {
                logMiss(
                    "geometry T=\(tokens) K=\(inputDim) N=\(outputDim) bits=\(quantized.bits) gs=\(quantized.groupSize) x=\(x.dtype)"
                )
            }
            return nil
        }
        let kernelN: Int
        let weight: MLXArray
        let scales: MLXArray
        let kernelBiases: MLXArray
        if blockN(for: outputDim) != nil {
            kernelN = outputDim
            weight = packedWeight
            scales = packedScales
            kernelBiases = affineBiases
        } else {
            guard let target = padTarget(for: outputDim, environment: environment),
                let bank = paddedInjectBank(quantized, compute: compute, targetRows: target)
            else {
                Qwen4ExpAffineQMMInvocation.recordMissPacked()
                logMiss("inject pad N=\(outputDim)")
                return nil
            }
            kernelN = target
            weight = bank.weight
            scales = bank.scales
            kernelBiases = bank.biases
        }
        guard let bn = blockN(for: kernelN) else { return nil }
        guard weight.dtype == .uint32, weight.ndim == 2,
            kernelBiases.dtype == compute,
            scales.ndim == 2, kernelBiases.ndim == 2,
            weight.dim(0) == kernelN,
            weight.dim(1) * 32 == inputDim * quantized.bits,
            scales.dim(0) == kernelN,
            scales.dim(1) == inputDim / quantized.groupSize,
            kernelBiases.shape == scales.shape
        else {
            Qwen4ExpAffineQMMInvocation.recordMissPacked()
            logMiss(
                "packed/scales weight=\(weight.dtype)/\(weight.shape) scales=\(scales.dtype)/\(scales.shape) biases=\(kernelBiases.dtype)/\(kernelBiases.shape) bits=\(quantized.bits) T=\(tokens) K=\(inputDim) N=\(outputDim)"
            )
            return nil
        }

        let leading = Array(xCompute.shape.dropLast())
        let m = xCompute.size / inputDim
        let x2d = xCompute.ndim == 2 ? xCompute : xCompute.reshaped([m, inputDim])
        let bm = tileM(bits: quantized.bits, environment: environment)
        let nTiles = kernelN / bn
        let mTiles = (m + bm - 1) / bm
        let outputs = kernel(
            [
                weight, scales, kernelBiases, x2d,
                MLXArray(Int32(inputDim)),
                MLXArray(Int32(kernelN)),
                MLXArray(Int32(m)),
            ],
            template: [
                ("T", compute),
                ("BITS", quantized.bits),
                ("GROUP_SIZE", quantized.groupSize),
                ("BM", bm),
                ("BK", blockK),
                ("BN", bn),
            ],
            grid: (nTiles * simdWidth, mTiles * warpsM, warpsN),
            threadGroup: (simdWidth, warpsM, warpsN),
            outputShapes: [[m, kernelN]],
            outputDTypes: [compute])
        guard var y = outputs.first else { return nil }
        if kernelN != outputDim {
            y = y[0..., 0 ..< outputDim]
        }
        if leading.count != 1 {
            y = y.reshaped(leading + [outputDim])
        }
        return Qwen4ExpActivation.nativeOutput(y, matching: x, compute: compute)
    }

    static func disable() {
        lock.lock()
        disabled = true
        lock.unlock()
    }

    static func paddedInjectBank(
        _ quantized: QuantizedLinear, compute: DType, targetRows: Int
    ) -> (
        weight: MLXArray, scales: MLXArray, biases: MLXArray
    )? {
        guard let biases = quantized.biases else { return nil }
        let rows = quantized.shape.0
        guard rows > 0, rows < targetRows, blockN(for: targetRows) != nil else { return nil }
        let bank = paddedInject.bank(
            owner: quantized, weight: quantized.weight, scales: quantized.scales,
            biases: biases, compute: compute, targetRows: targetRows)
        return (bank.weight, bank.scales, bank.biases)
    }

    private static let missLock = NSLock()
    nonisolated(unsafe) private static var didLogMiss = false

    private static func logMiss(_ message: String) {
        missLock.lock()
        defer { missLock.unlock() }
        if didLogMiss { return }
        didLogMiss = true
        #if DEBUG
            print("affineQmm first-miss: \(message)")
        #endif
    }
}

extension Qwen3NextMLP {
    func qwen4AffinePrefill(_ x: MLXArray) -> MLXArray {
        Qwen4ExpAffineQMM.applyMLP(gate: gateProj, up: upProj, down: downProj, x: x)
    }
}

private let affineQMMKernel: MLXFast.MLXFastKernel? = {
    guard let header = qwen4AffineQMMHeader() else { return nil }
    return MLXFast.metalKernel(
        name: "qwen4_affine_qmm_t",
        inputNames: ["w", "scales", "biases", "x", "K", "N", "M"],
        outputNames: ["y"],
        source: qwen4AffineQMMSource,
        header: header,
        ensureRowContiguous: true)
}()

private let qwen4AffineQMMSource = """
    constexpr int BK_padded = (BK + 16 / sizeof(T));
    threadgroup T Xs[BM * BK_padded];
    threadgroup T Ws[BN * BK_padded];
    const int K_i = (int)K;
    const int N_i = (int)N;
    const int M_i = (int)M;
    qwen4_qmm_t_i32<T, GROUP_SIZE, BITS, BM, BK, BN>(
        w, scales, biases, x, y, Xs, Ws, K_i, N_i, M_i,
        threadgroup_position_in_grid,
        simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
    """

private let qwen4QmmValueWrapper = """

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

private func qwen4AffineQMMHeader() -> String? {
    let gemm = Qwen4ExpMetalHeaders.gemm
    let quantizedUtils = Qwen4ExpMetalHeaders.quantizedUtils
    var quantized = Qwen4ExpMetalHeaders.quantized
    if let kernel = quantized.range(
        of: "\ntemplate <typename T, int group_size, int bits, int D, bool batched>\n[[kernel]]")
        ?? quantized.range(of: "\n[[kernel]]")
    {
        quantized = String(quantized[..<kernel.lowerBound])
    }
    return gemm + "\n" + quantizedUtils + "\n" + quantized + qwen4QmmValueWrapper
}
