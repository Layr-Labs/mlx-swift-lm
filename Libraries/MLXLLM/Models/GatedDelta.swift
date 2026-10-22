//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN
import os

// MARK: - Compute G

func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    // Keep g in fp32 (do not downcast to a.dtype). g/beta inheriting bf16 from
    // the input tensors forces a Metal kernel recompile between turns vs the
    // fp32 SSM state, which triggers a use-after-free when reusing
    // QuantizedKVCache. Upstream 93cf322.
    return decay
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                {
                  // Preserve Kahan summation under Metal's default fast math.
                  #pragma clang fp reassociate(off)
                  #pragma clang fp contract(off)
                  float kv_compensation = 0.0f;
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    auto product = state[i] * k_[s_idx];
                    auto corrected = product - kv_compensation;
                    auto next_sum = kv_mem + corrected;
                    kv_compensation = (next_sum - kv_mem) - corrected;
                    kv_mem = next_sum;
                  }
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

/// `gated_delta_step` with one extra output: the fp32 state after EVERY
/// position, `state_stack [B, T, Hv, Dv, Dk]`. The recurrence body is the
/// same code as the unstacked kernel and the state stays in the same fp32
/// registers across `t`; a per-position caller round-trips that state
/// through an fp32 buffer, which is exact, so `y`, `state_out` and each
/// stack entry are bit-identical to `T` chained single-position launches.
/// Lightning MTP capture-verify needs exactly those `T` states (finalize
/// commits the accepted position's state); without this kernel it paid one
/// launch chain per position.
/// The state-load, decay/Kahan-dot, and delta/output source blocks match the
/// current unstacked kernel, including both fast-math pragmas. Compiled
/// per-position byte equality is a separate gate in
/// `Qwen4GDNStackedKernelParityTests`, not implied by source text alone.
private func makeGatedDeltaStackedKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;
            // state_stack: [B, T, Hv, Dv, Dk]; position t of row b_idx
            // starts at ((b_idx * T + t) * Hv + hv_idx) * Dv * Dk.
            auto s_stack = state_stack
                + ((b_idx * T) * Hv + hv_idx) * Dv * Dk + dv_idx * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              {
                // Preserve Kahan summation under Metal's default fast math.
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float kv_compensation = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  auto product = state[i] * k_[s_idx];
                  auto corrected = product - kv_compensation;
                  auto next_sum = kv_mem + corrected;
                  kv_compensation = (next_sum - kv_mem) - corrected;
                  kv_mem = next_sum;
                }
              }
              kv_mem = simd_sum(kv_mem);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
                s_stack[s_idx] = static_cast<StT>(state[i]);
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
              s_stack += Hv * Dv * Dk;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """

    return MLXFast.metalKernel(
        name: "gated_delta_step_stacked",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out", "state_stack"],
        source: source
    )
}

/// Unmasked prefill layout of `gated_delta_step`. The stock kernel gives each
/// value row a whole SIMD group (Dk / 32 state elements per lane) and pays two
/// 32-lane reductions per row and position. Here eight lanes own one row, so a
/// SIMD group carries four rows, and each lane runs four stock lanes' Kahan
/// chains with the stock per-element source. The chains are then combined in
/// simd_sum's ascending butterfly order (its xor 1 and 2 levels inside the
/// lane, its xor 4, 8 and 16 levels as xor 1, 2 and 4 shuffles within the row
/// group), so `y` and `state_out` equal the stock kernel's bit for bit
/// wherever simd_sum lowers to that butterfly (as on Apple M5;
/// `GatedDeltaPrefill` checks it on each GPU before use).
private func makeGatedDeltaPackedKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            constexpr int n_per_t = Dk / 32;
            constexpr int lanes_per_row = 8;
            constexpr int chains_per_lane = 32 / lanes_per_row;
            constexpr int rows_per_simdgroup = 32 / lanes_per_row;
            constexpr int values_per_lane = n_per_t * chains_per_lane;
            static_assert(chains_per_lane == 4, "the chain tree below is written for 4 chains");

            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);

            // Lane l of a row group holds stock lanes 4l ..< 4l + 4 of value
            // row dv_idx, i.e. Dk values [l, l + 1) * values_per_lane.
            auto lane_in_row = thread_index_in_simdgroup % lanes_per_row;
            auto dv_idx = thread_position_in_grid.y * rows_per_simdgroup
                + thread_index_in_simdgroup / lanes_per_row;
            auto dk_base = lane_in_row * values_per_lane;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk + dk_base;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk + dk_base;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk + dk_base;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk + dk_base;

            float state[values_per_lane];
            for (int i = 0; i < values_per_lane; ++i) {
              state[i] = static_cast<float>(i_state[i]);
            }

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              {
                // Preserve Kahan summation and the chain tree order under
                // Metal's default fast math.
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float kv_chain[chains_per_lane];
                for (int c = 0; c < chains_per_lane; ++c) {
                  float chain_sum = 0.0f;
                  float kv_compensation = 0.0f;
                  for (int i = c * n_per_t; i < (c + 1) * n_per_t; ++i) {
                    state[i] = state[i] * g_[hv_idx];
                    auto product = state[i] * k_[i];
                    auto corrected = product - kv_compensation;
                    auto next_sum = chain_sum + corrected;
                    kv_compensation = (next_sum - chain_sum) - corrected;
                    chain_sum = next_sum;
                  }
                  kv_chain[c] = chain_sum;
                }
                kv_mem = (kv_chain[0] + kv_chain[1]) + (kv_chain[2] + kv_chain[3]);
              }
              kv_mem += simd_shuffle_xor(kv_mem, 1);
              kv_mem += simd_shuffle_xor(kv_mem, 2);
              kv_mem += simd_shuffle_xor(kv_mem, 4);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out_chain[chains_per_lane];
              for (int c = 0; c < chains_per_lane; ++c) {
                float chain_out = 0.0f;
                for (int i = c * n_per_t; i < (c + 1) * n_per_t; ++i) {
                  state[i] = state[i] + k_[i] * delta;
                  chain_out += state[i] * q_[i];
                }
                out_chain[c] = chain_out;
              }
              float out = 0.0f;
              {
                #pragma clang fp reassociate(off)
                out = (out_chain[0] + out_chain[1]) + (out_chain[2] + out_chain[3]);
              }
              out += simd_shuffle_xor(out, 1);
              out += simd_shuffle_xor(out, 2);
              out += simd_shuffle_xor(out, 4);
              if (lane_in_row == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < values_per_lane; ++i) {
              o_state[i] = static_cast<StT>(state[i]);
            }
        """

    return MLXFast.metalKernel(
        name: "gated_delta_step_packed",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: source
    )
}

/// `gated_delta_step` for four value rows per SIMD group. Each row keeps the
/// stock lane layout (Dk / 32 state elements per lane) and the stock per-row
/// source, both `simd_sum` calls included, so a row's arithmetic does not
/// depend on how `simd_sum` lowers. The rows share each position's q/k loads
/// and their reductions are independent, which hides part of their latency.
/// The prefill fallback where the packed kernel differs from the stock one.
private func makeGatedDeltaRowsKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            constexpr int n_per_t = Dk / 32;
            constexpr int rows = 4;

            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_index_in_simdgroup;
            auto dv_base = thread_position_in_grid.y * rows;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]; rows dv_base ..< dv_base + rows
            auto i_state = state_in + (n * Dv + dv_base) * Dk;
            auto o_state = state_out + (n * Dv + dv_base) * Dk;

            float state[rows][n_per_t];
            for (int r = 0; r < rows; ++r) {
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[r][i] = static_cast<float>(i_state[r * Dk + s_idx]);
              }
            }

            for (int t = 0; t < T; ++t) {
              float kv_mem[rows];
              for (int r = 0; r < rows; ++r) {
                kv_mem[r] = 0.0f;
                {
                  // Preserve Kahan summation under Metal's default fast math.
                  #pragma clang fp reassociate(off)
                  #pragma clang fp contract(off)
                  float kv_compensation = 0.0f;
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[r][i] = state[r][i] * g_[hv_idx];
                    auto product = state[r][i] * k_[s_idx];
                    auto corrected = product - kv_compensation;
                    auto next_sum = kv_mem[r] + corrected;
                    kv_compensation = (next_sum - kv_mem[r]) - corrected;
                    kv_mem[r] = next_sum;
                  }
                }
              }
              for (int r = 0; r < rows; ++r) {
                kv_mem[r] = simd_sum(kv_mem[r]);
              }

              float out[rows];
              for (int r = 0; r < rows; ++r) {
                auto delta = (v_[dv_base + r] - kv_mem[r]) * beta_[hv_idx];
                out[r] = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[r][i] = state[r][i] + k_[s_idx] * delta;
                  out[r] += state[r][i] * q_[s_idx];
                }
              }
              for (int r = 0; r < rows; ++r) {
                out[r] = simd_sum(out[r]);
              }
              // Lane r writes row r.
              if (thread_index_in_simdgroup < rows) {
                float row_out = out[0];
                for (int r = 1; r < rows; ++r) {
                  row_out = (thread_index_in_simdgroup == uint(r)) ? out[r] : row_out;
                }
                y[dv_base + thread_index_in_simdgroup] = static_cast<InT>(row_out);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int r = 0; r < rows; ++r) {
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                o_state[r * Dk + s_idx] = static_cast<StT>(state[r][i]);
              }
            }
        """

    return MLXFast.metalKernel(
        name: "gated_delta_step_rows",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let kernelStacked: MLXFast.MLXFastKernel?
    let kernelPacked: MLXFast.MLXFastKernel?
    let kernelRows: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true)
        kernelStacked = makeGatedDeltaStackedKernel()
        kernelPacked = makeGatedDeltaPackedKernel()
        kernelRows = makeGatedDeltaRowsKernel()
    }
}

/// A prompt-window kernel for `gatedDeltaUpdate`. Both take four value rows
/// per SIMD group and keep the `gated_delta_step` contract.
enum GatedDeltaPrefillKernel: String, CaseIterable, Sendable {
    /// `gated_delta_step_packed`: eight lanes per value row.
    case packed
    /// `gated_delta_step_rows`: the stock lane layout, four rows at a time.
    case rows
}

/// Which kernel `gatedDeltaUpdate` runs on a prompt window. Decode (T == 1),
/// MTP verify windows and masked updates keep `gated_delta_step`.
///
/// The prefix cache relies on a dense GDN state being bit-identical whatever
/// chunk partition produced it (`CBv2RecurrentCheckpointGeometry`), and every
/// window shorter than `minT` still runs the stock kernel, so a prompt kernel
/// must equal `gated_delta_step` bit for bit. The packed kernel does only
/// where `simd_sum` lowers to the ascending butterfly; the rows kernel keeps
/// the stock per-row source and `simd_sum`. So on the first prompt window of
/// each kernel instantiation (input dtype and head geometry) the update tries
/// them in order (packed, then rows) on a fixed probe window and keeps the
/// first that reproduces the stock kernel's bits on this GPU, or none.
///
/// `DARKBLOOM_TTFT_GDN_BLOCKED`: unset or any other value probes; `0`,
/// `false`, `no` or `off` keep the stock kernel; `packed` or `rows` skip the
/// probe and force that kernel (A/B measurement only).
enum GatedDeltaPrefill {
    static let envFlag = "DARKBLOOM_TTFT_GDN_BLOCKED"
    /// Shorter windows are launch-bound. This also keeps decode and MTP verify
    /// windows (at most 1 + `CBv2MTPConfig.testedMaxDraftTokens` = 8
    /// positions) on the stock kernel.
    static let minT = 16
    static let keyHeadDim = 128
    static let rowsPerSimdgroup = 4
    static let simdgroupsPerThreadgroup = 4

    enum Mode: Equatable, Sendable {
        case stock
        case probe
        case forced(GatedDeltaPrefillKernel)

        init(environmentValue: String?) {
            let raw =
                environmentValue?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                ?? ""
            if ["0", "false", "no", "off"].contains(raw) {
                self = .stock
            } else if let kernel = GatedDeltaPrefillKernel(rawValue: raw) {
                self = .forced(kernel)
            } else {
                self = .probe
            }
        }
    }

    static let mode = Mode(environmentValue: ProcessInfo.processInfo.environment[envFlag])

    static func matchesGeometry(T: Int, keyHeadDim: Int, valueHeadDim: Int) -> Bool {
        T >= minT && keyHeadDim == Self.keyHeadDim && valueHeadDim > 0
            && valueHeadDim % rowsPerSimdgroup == 0
    }

    /// A kernel instantiation: everything in its template but Dk (always
    /// `keyHeadDim` here). The probe runs at exactly this geometry.
    struct Geometry: Hashable, Sendable {
        let dtype: DType
        let keyHeads: Int
        let valueHeads: Int
        let valueHeadDim: Int
    }

    /// The kernel for this window, or nil for `gated_delta_step`. Probing
    /// evaluates a small window once per geometry; the result is cached.
    static func kernel(
        T: Int, keyHeadDim: Int, hasMask: Bool, geometry: Geometry, mode: Mode = Self.mode
    ) -> GatedDeltaPrefillKernel? {
        guard !hasMask,
            matchesGeometry(T: T, keyHeadDim: keyHeadDim, valueHeadDim: geometry.valueHeadDim)
        else { return nil }
        switch mode {
        case .stock: return nil
        case .forced(let kernel): return kernel
        case .probe: return probedKernel(geometry)
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var probed: [Geometry: GatedDeltaPrefillKernel?] = [:]
    private static let logger = Logger(subsystem: "darkbloom", category: "GatedDeltaPrefill")

    static func probedKernel(_ geometry: Geometry) -> GatedDeltaPrefillKernel? {
        lock.lock()
        let cached = probed[geometry]
        lock.unlock()
        if let cached { return cached }
        // Probe outside the lock: a concurrent first window probes too and
        // stores the same answer.
        let selected = probe(geometry)
        lock.lock()
        probed[geometry] = selected
        lock.unlock()
        logger.notice(
            "gdn_prefill_kernel dtype=\(String(describing: geometry.dtype), privacy: .public) hk=\(geometry.keyHeads, privacy: .public) hv=\(geometry.valueHeads, privacy: .public) dv=\(geometry.valueHeadDim, privacy: .public) kernel=\(selected?.rawValue ?? "stock", privacy: .public)"
        )
        return selected
    }

    /// The first candidate whose `y` and `state_out` on a fixed 32-position
    /// window equal `gated_delta_step`'s bit for bit. The window has a
    /// non-zero state; any reduction-order difference shows in its per-row
    /// dot products.
    static func probe(
        _ geometry: Geometry,
        candidates: [GatedDeltaPrefillKernel] = GatedDeltaPrefillKernel.allCases
    ) -> GatedDeltaPrefillKernel? {
        guard GatedDeltaKernelManager.shared.kernel != nil else { return nil }
        let (B, T, Dk) = (1, 32, keyHeadDim)
        let (Hk, Hv, Dv, dtype) = (
            geometry.keyHeads, geometry.valueHeads, geometry.valueHeadDim, geometry.dtype
        )
        let keys = MLXRandom.split(key: MLXRandom.key(0x4744_4e50), into: 6)
        let q = (MLXRandom.normal([B, T, Hk, Dk], key: keys[0]) * 0.1).asType(dtype)
        let k = (MLXRandom.normal([B, T, Hk, Dk], key: keys[1]) * 0.1).asType(dtype)
        let v = MLXRandom.normal([B, T, Hv, Dv], key: keys[2]).asType(dtype)
        let g = MLXRandom.uniform(low: 0.5, high: 1.0, [B, T, Hv], key: keys[3])
        let beta = MLXRandom.uniform(low: 0.0, high: 1.0, [B, T, Hv], key: keys[4])
        let state = MLXRandom.normal([B, Hv, Dv, Dk], key: keys[5]) * 0.1
        let (stockY, stockState) = gatedDeltaKernel(
            q: q, k: k, v: v, g: g, beta: beta, state: state)
        for candidate in candidates {
            guard
                let (y, candidateState) = gatedDeltaKernelPrefill(
                    candidate, q: q, k: k, v: v, g: g, beta: beta, state: state)
            else { continue }
            eval(stockY, stockState, y, candidateState)
            if bitsEqual(y, stockY), bitsEqual(candidateState, stockState) {
                return candidate
            }
        }
        return nil
    }

    /// Same shape, dtype and bytes.
    static func bitsEqual(_ x: MLXArray, _ y: MLXArray) -> Bool {
        guard x.shape == y.shape, x.dtype == y.dtype else { return false }
        let bits: DType = x.dtype.size == 2 ? .uint16 : .uint32
        return all(x.view(dtype: bits) .== y.view(dtype: bits)).item(Bool.self)
    }
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.kernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.kernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

/// Stacked dispatch: `(y [B, T, Hv, Dv], state_out [B, Hv, Dv, Dk],
/// state_stack [B, T, Hv, Dv, Dk])`. nil when the Metal kernel is
/// unavailable so callers keep the per-position chain.
func gatedDeltaKernelStacked(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray
) -> (y: MLXArray, state: MLXArray, stack: MLXArray)? {
    guard let kernel = GatedDeltaKernelManager.shared.kernelStacked else { return nil }
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let outputs = kernel(
        [q, k, v, g, beta, state, MLXArray(T)],
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape, [B, T, Hv, Dv, Dk]],
        outputDTypes: [inputType, stateType, stateType]
    )
    return (outputs[0], outputs[1], outputs[2])
}

/// Prompt-window dispatch: the unmasked `gatedDeltaKernel` contract (same
/// inputs, template, output shapes and dtypes). Callers check
/// `GatedDeltaPrefill.matchesGeometry` first; nil when the Metal kernel is
/// unavailable.
func gatedDeltaKernelPrefill(
    _ selected: GatedDeltaPrefillKernel,
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray
) -> (MLXArray, MLXArray)? {
    let manager = GatedDeltaKernelManager.shared
    guard let kernel = selected == .packed ? manager.kernelPacked : manager.kernelRows
    else { return nil }
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let outputs = kernel(
        [q, k, v, g, beta, state, MLXArray(T)],
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv / GatedDeltaPrefill.rowsPerSimdgroup, B * Hv),
        threadGroup: (32, GatedDeltaPrefill.simdgroupsPerThreadgroup, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )
    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y.asType(q.dtype), state)
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = gatedDeltaStepOps(
            q: qT,
            k: kT,
            v: vT,
            g: gT,
            beta: betaT,
            state: state,
            mask: maskT
        )
        ys.append(y)
        state = newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Public API

/// Shared gated-delta (GDN) recurrence for the Qwen3.5 / Qwen3-Next family.
///
/// This is the single source of truth for the recurrence: the LLM models
/// (`Qwen35`, `Qwen3Next`) and the VLM `Qwen35` model all call it. It computes
/// `g`/`beta` in fp32 and keeps the recurrent `state` in fp32 across the
/// T-step recurrence (upstream c566c95 + 93cf322); bf16 state loses precision
/// and, on the kernel path, forces a recompile vs. the fp32 state. `public`
/// so `MLXVLM` can reuse it instead of carrying a private (drift-prone) copy.
public func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    useBlockedSeq: Bool = false,
    fuseQwen35Gates: Bool = false
) -> (MLXArray, MLXArray) {
    let g: MLXArray
    let beta: MLXArray
    if useBlockedSeq, Qwen4ExpFusions.isEnabled {
        // qwen4: one compiled kernel for both gates (same ops and dtypes).
        (g, beta) = Qwen4ExpFusions.gdnGates(a: a, b: b, aLog: aLog, dtBias: dtBias)
    } else if fuseQwen35Gates {
        // qwen3_5: the same expressions as below, one compiled kernel.
        (g, beta) = Qwen35DecodeFusions.gdnGates(a: a, b: b, aLog: aLog, dtBias: dtBias)
    } else {
        beta = sigmoid(b).asType(.float32)
        g = computeGatedDeltaG(aLog, a, dtBias)
    }

    let B = q.dim(0)
    let T = q.dim(1)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    // State kept in fp32 to match Python mlx-lm. Using q.dtype (bf16) loses
    // precision across T-step recurrence, compounding rounding error.
    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if state.dtype != .float32 {
        state = state.asType(.float32)
    }

    // Fusion blocked-seq: Qwen4 prefill only (`useBlockedSeq`). Decode and
    // masked verify stay on the stock kernel.
    if useBlockedSeq,
        Qwen4ExpGDNBlockedSeq.shouldDispatch(
            T: T, keyHeadDim: Dk, valueHeadDim: Dv, hasMask: mask != nil)
    {
        return gatedDeltaBlockedSeq(q: q, k: k, v: v, g: g, beta: beta, state: state)
    }

    // Prompt windows of the other GDN families. The selected kernel equals
    // the stock one bit for bit, so a recurrent state is the same whichever
    // kernel ran each chunk (checkpoint capture/restore, replay and decode).
    if !useBlockedSeq,
        let selected = GatedDeltaPrefill.kernel(
            T: T, keyHeadDim: Dk, hasMask: mask != nil,
            geometry: .init(
                dtype: q.dtype, keyHeads: q.dim(2), valueHeads: Hv, valueHeadDim: Dv)),
        let result = gatedDeltaKernelPrefill(
            selected, q: q, k: k, v: v, g: g, beta: beta, state: state)
    {
        return result
    }

    if GatedDeltaKernelManager.shared.kernel != nil {
        return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}

/// Whole-window recurrence that also returns the fp32 state after every
/// position: `(y [B, T, Hv, Dv], stateStack [B, T, Hv, Dv, Dk])`, with
/// `stateStack[:, T-1]` the final state. Bit-identical to calling
/// `gatedDeltaUpdate` once per position and stacking the results (same
/// kernel body, same fp32 register state; `g`/`beta` are elementwise so
/// computing them over the window equals computing them per slice). Used by
/// capture-verify, which needs the per-position stack for commit/rollback.
/// nil when the Metal kernel is unavailable.
public func gatedDeltaUpdateStacked(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?
) -> (y: MLXArray, stateStack: MLXArray)? {
    let g: MLXArray
    let beta: MLXArray
    // Stacked verify is qwen4-only: same fused gates as the T=1 decode path.
    if Qwen4ExpFusions.isEnabled {
        (g, beta) = Qwen4ExpFusions.gdnGates(a: a, b: b, aLog: aLog, dtBias: dtBias)
    } else {
        beta = sigmoid(b).asType(.float32)
        g = computeGatedDeltaG(aLog, a, dtBias)
    }

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if state.dtype != .float32 {
        state = state.asType(.float32)
    }
    guard
        let result = gatedDeltaKernelStacked(
            q: q, k: k, v: v, g: g, beta: beta, state: state)
    else { return nil }
    return (result.y, result.stack)
}
