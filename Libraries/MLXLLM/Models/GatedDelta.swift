//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

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

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let kernelStacked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true)
        kernelStacked = makeGatedDeltaStackedKernel()
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
    useBlockedSeq: Bool = false
) -> (MLXArray, MLXArray) {
    let g: MLXArray
    let beta: MLXArray
    if useBlockedSeq, Qwen4ExpFusions.isEnabled {
        // qwen4: one compiled kernel for both gates (same ops and dtypes).
        (g, beta) = Qwen4ExpFusions.gdnGates(a: a, b: b, aLog: aLog, dtBias: dtBias)
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

    // Fusion blocked-seq: Qwen4 prefill only (`useBlockedSeq`). Decode,
    // masked verify, and 27B qwen3_5 stay on the stock kernel.
    if useBlockedSeq,
        Qwen4ExpGDNBlockedSeq.shouldDispatch(
            T: T, keyHeadDim: Dk, valueHeadDim: Dv, hasMask: mask != nil)
    {
        return gatedDeltaBlockedSeq(q: q, k: k, v: v, g: g, beta: beta, state: state)
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
