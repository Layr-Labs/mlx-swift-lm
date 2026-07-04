import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

/// Compiled SiLU-gated product (`silu(gate) * up`) for the common MoE GLU path.
/// Fusing activation + product into one compiled, shapeless kernel cuts kernel
/// dispatches and intermediates on the hot decode path. Upstream ef85ed0.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on) like the sibling `compiledSwiGLU` / `safeGeluApproximate` fusions.
/// The default SiLU `SwitchGLU` path wires this in as `activationProduct` (the
/// highest-precedence branch in `callAsFunction`) and `LFM2MoE` calls it directly,
/// so without the gate both would keep hitting compiled kernels on the very M1/M2 +
/// macOS Tahoe machines the opt-out (MLX #3329) is meant to protect. Falls back to
/// the plain uncompiled closure when off; the default (env unset) stays compiled.
public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled weighted expert-output combine (`(outputs * weights[..., None]).sum(-2)`).
/// Shared by MoE routers (e.g. Gemma 4) to fuse the scale + reduce. Upstream ef85ed0.
public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

// MARK: - Compiled activation fusions (vMLX / osaurus-main port)

/// Approximate (tanh) GELU written with `x * x * x` instead of the Power
/// primitive (`x ** 3`). The Power primitive returns zero results under the
/// macOS Tahoe Metal JIT (MLX #3329), so the explicit multiplies keep it safe
/// under `compile(shapeless: true)`. Numerically identical to
/// `MLXNN.geluApproximate`.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on); falls back to the plain closure when compiled fusions are off.
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Drop-in replacement for `MLXNN.GELU(approximation: .tanh)` that avoids the
/// Power primitive crash. Use anywhere a tanh-approx GELU unary layer is needed.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

/// Compiled SiLU-gated GLU product (`silu(gate) * up`). Same math as
/// `compiledSiluProduct` above, but gated by `MLXHardwareInfo` so M1/M2 + macOS
/// Tahoe can opt out. Used by `SwitchGLU` when a SiLU activation is supplied via
/// the custom-activation initializer (where `activationProduct` is nil).
private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled GELU-gated GLU product (`geluApprox(gate) * up`), fusing the tanh
/// GELU and the element-wise multiply into one shapeless kernel. Uses the
/// Power-free `x * x * x` GELU so it is safe under `compile(shapeless: true)`.
private let compiledGeGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear?
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear?
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    /// Optional fused (activation * up) kernel. Set for the default SiLU path so
    /// the GLU product runs as one compiled op; nil when a custom activation is
    /// supplied (we then fall back to `activation(gate) * up`). Upstream ef85ed0.
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    /// Activation-type flags detected once at init from a tiny test input (vMLX
    /// approach — no per-token check). Only consulted when `activationProduct` is
    /// nil (the custom-activation path): they let SiLU/GELU custom activations use
    /// the compiled `compiledSwiGLU` / `compiledGeGLU` fusions instead of the
    /// uncompiled `activation(gate) * up`. On any mismatch we fall back to that
    /// exact uncompiled path, so detection only ever enables a numerically
    /// equivalent fast path — it can never change results.
    let isSiluActivation: Bool
    let isGeluActivation: Bool

    // Lazy fused gate+up gatherQuantizedMM cache.
    //
    // When both gate_proj and up_proj are QuantizedSwitchLinear with
    // matching (groupSize, bits, mode), we concatenate their weight,
    // scales and biases along the output axis once on first forward
    // and run a single `gatherQuantizedMM` for gate+up instead of two.
    // The compiled SwiGLU/GeGLU then splits the result and multiplies.
    //
    // Why: the standard 4-bit MoE path dispatches 3 separate
    // gatherQuantizedMM Metal kernels per layer (gate, up, down).
    // Halving the gate+up dispatches to one wider matmul saves one
    // Metal dispatch per layer per step, and the wider matmul has better
    // GPU occupancy because more output tiles share the same input read.
    //
    // Disabled via `BENCH_NO_FUSED_GATE_UP=1` env var for A/B.
    private var fusedGateUpWeight: MLXArray? = nil
    private var fusedGateUpScales: MLXArray? = nil
    private var fusedGateUpBiases: MLXArray? = nil
    private var fusedGroupSize: Int = 64
    private var fusedBits: Int = 4
    private var fusedMode: QuantizationMode = .affine
    private var fusionAttempted: Bool = false

    /// Default SiLU GLU path -- uses the compiled fused (silu * up) kernel.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct
        // Default path is SiLU and `activationProduct` is non-nil, so these are
        // not consulted on the hot path; set them accurately for completeness
        // (and to avoid a needless probe eval at load for every MoE layer).
        self.isSiluActivation = true
        self.isGeluActivation = false

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Explicit gate×up product path — for activations that are NOT a pure
    /// unary function of the gate (e.g. DeepSeek-V4's limited SwiGLU, which
    /// clamps BOTH gate and up). Bypasses the single-point SiLU/GELU probe
    /// below entirely: that probe evaluates the activation at x=1.0, where a
    /// clamped SwiGLU is indistinguishable from plain SiLU, and would silently
    /// route the hot path onto the UNclamped compiled kernel.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activationProduct: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        // Unused on the hot path (activationProduct wins in callAsFunction);
        // kept as a sane fallback.
        self.activation = MLXNN.silu
        self.activationProduct = activationProduct
        self.isSiluActivation = false
        self.isGeluActivation = false

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Custom-activation GLU path -- runs `activation(gate) * up` uncompiled.
    ///
    /// WARNING: the SiLU/GELU detection below is a SINGLE-POINT probe (x=1.0).
    /// It is only sound for activations that are exactly silu/gelu everywhere.
    /// Activations that merely AGREE with silu at 1.0 (clamped variants) must
    /// use the `activationProduct:` initializer above instead.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil
        // Detect SiLU/GELU once via a tiny test input (vMLX approach) so the hot
        // path can select the compiled fusion without a per-token check. Exact
        // equality is intentional: a match means the supplied closure computes
        // that exact function; any non-match falls back to `activation(gate) * up`
        // in callAsFunction, so this can only ever enable an equivalent fast path.
        let probe = MLXArray([Float(1.0)])
        let probeOut = activation(probe)
        let detectedSilu = (probeOut .== MLXNN.silu(probe)).all().item(Bool.self)
        self.isSiluActivation = detectedSilu
        self.isGeluActivation =
            !detectedSilu && (probeOut .== safeGeluApproximate(probe)).all().item(Bool.self)

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Populate the fused gate+up weight cache on first forward. Safe to
    /// call multiple times — guarded by `fusionAttempted` so the work runs
    /// exactly once per SwitchGLU instance.
    private func ensureFusedGateUp() {
        if fusionAttempted { return }
        fusionAttempted = true

        // Feature flag — opt out for A/B comparison.
        if ProcessInfo.processInfo.environment["BENCH_NO_FUSED_GATE_UP"] == "1" {
            return
        }

        // Only fuse when we have separate gate_proj / up_proj (not gateUpProj,
        // which is already a single fused weight from the checkpoint).
        guard let g = gateProj as? QuantizedSwitchLinear,
              let u = upProj as? QuantizedSwitchLinear,
              g.groupSize == u.groupSize,
              g.bits == u.bits,
              g.mode == u.mode
        else {
            // Non-quantized, mismatched quantization params, or gateUpProj
            // path — can't fuse.
            // Non-quantized, mismatched quantization params, or gateUpProj
            // path — can't fuse.
            return
        }

        let fusedBytes =
            g.weight.nbytes + u.weight.nbytes
            + g.scales.nbytes + u.scales.nbytes
            + (g.biases?.nbytes ?? 0) + (u.biases?.nbytes ?? 0)
        let cacheLimit = fusedGateUpCacheByteLimit()
        if cacheLimit >= 0 && fusedBytes > cacheLimit {
            return
        }

        // Concatenate along output axis. Quantized SwitchLinear weights are
        // shaped `[E, out, in_packed]`, so axis -2 stacks gate and up along
        // the output dimension, giving `[E, 2*hidden, in_packed]`. Scales
        // and biases track the same output axis at group granularity.
        let fusedW = concatenated([g.weight, u.weight], axis: -2)
        let fusedS = concatenated([g.scales, u.scales], axis: -2)
        var fusedB: MLXArray? = nil
        if let gb = g.biases, let ub = u.biases {
            fusedB = concatenated([gb, ub], axis: -2)
        }

        // Force materialization now so the first forward pass doesn't pay
        // the concat cost mid-generation.
        var toMaterialize: [MLXArray] = [fusedW, fusedS]
        if let fb = fusedB { toMaterialize.append(fb) }
        MLX.eval(toMaterialize)

        self.fusedGateUpWeight = fusedW
        self.fusedGateUpScales = fusedS
        self.fusedGateUpBiases = fusedB
        self.fusedGroupSize = g.groupSize
        self.fusedBits = g.bits
        self.fusedMode = g.mode
    }

    /// Build this SwitchGLU's fused gate+up cache now (if eligible) and hand
    /// the SAME fused arrays to `other`, marking `other`'s fusion as done so
    /// it never builds its own copy.
    ///
    /// For weight-sharing module trees — e.g. a VLM wrapper's language model
    /// and an MLXLLM text model extracted over the same quantized arrays.
    /// Each SwitchGLU instance otherwise lazily concatenates its own fused
    /// gate+up copy on first forward (~540 MB per layer on Gemma4-26B-8bit,
    /// ~15 GiB model-wide), so two trees over the same weights retain TWO
    /// identical multi-GiB copies in unified memory. Sharing keeps the total
    /// at one copy, built eagerly here (deterministic at load time, where
    /// admission checks can see it) instead of mid-request.
    ///
    /// Must be called before either tree serves traffic: the fused-cache
    /// fields are unsynchronized module state, and this is the one write
    /// point that both trees' later (read-only) forwards rely on.
    ///
    /// If this instance is ineligible for fusion (non-quantized projections,
    /// mismatched quantization, cache-limit exceeded, or the
    /// `BENCH_NO_FUSED_GATE_UP` opt-out), `other` inherits the same verdict:
    /// neither tree will build a fused cache, and both use the unfused path.
    ///
    /// Returns whether a fused cache actually exists and was shared (false
    /// when the eligibility verdict — equally propagated — was "no fusion"),
    /// so callers can log truthful share counts.
    @discardableResult
    public func shareFusedGateUpCache(with other: SwitchGLU) -> Bool {
        ensureFusedGateUp()
        other.fusionAttempted = true
        other.fusedGateUpWeight = fusedGateUpWeight
        other.fusedGateUpScales = fusedGateUpScales
        other.fusedGateUpBiases = fusedGateUpBiases
        other.fusedGroupSize = fusedGroupSize
        other.fusedBits = fusedBits
        other.fusedMode = fusedMode
        return fusedGateUpWeight != nil
    }

    /// Whether the fused gate+up cache is populated, its total byte size, and
    /// the identity of its weight buffer. Diagnostic/test hooks for the
    /// cache-sharing contract (`shareFusedGateUpCache(with:)`): tests assert
    /// the shared tree holds the SAME array instance (not a second
    /// concatenation) and bound memory growth by the cache's real size.
    public var hasFusedGateUpCache: Bool { fusedGateUpWeight != nil }
    public var fusedGateUpWeightForVerification: MLXArray? { fusedGateUpWeight }
    public var fusedGateUpCacheBytes: Int {
        (fusedGateUpWeight?.nbytes ?? 0) + (fusedGateUpScales?.nbytes ?? 0)
            + (fusedGateUpBiases?.nbytes ?? 0)
    }

    private func fusedGateUpCacheByteLimit() -> Int {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["FUSED_GATE_UP_CACHE_LIMIT_BYTES"],
            let bytes = Int(raw)
        {
            return bytes
        }
        if let raw = env["FUSED_GATE_UP_CACHE_LIMIT_MB"],
            let mb = Int(raw)
        {
            return mb < 0 ? -1 : mb * 1024 * 1024
        }
        // Default: 1024 MiB. Gemma4-26B-A4B-8bit experts are ~540 MB per
        // fused gate+up pair; the prior 512 MB default excluded them.
        return 1024 * 1024 * 1024
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        ensureFusedGateUp()

        // Fused gate+up is a net win for DECODE (single-token forward pass,
        // compute-bound per-expert matmul) but a net LOSS for PREFILL
        // (multi-token batches are memory-bandwidth bound, and the single
        // wider matmul has worse cache locality than two narrower ones).
        //
        // Decide per-call which path to take. indices.size is the number
        // of (token, expert) dispatches: at decode with B=1 and top_k=8
        // it's 8; at prefill with 512 tokens and top_k=8 it's 4096. The
        // threshold (32 by default) admits single-token + a few prompt
        // tokens as "decode-shaped" and bounces large prefill chunks to
        // the two-call path. Override via BENCH_FUSED_GATE_UP_THRESHOLD.
        // Restrict fused path to true solo decode. With top_k=8, B=1 gives
        // indices.size=8. B=2 gives 16 which is slower fused (wider matmul has
        // worse cache locality). Default threshold 8 = solo only.
        let decodeThreshold: Int =
            Int(ProcessInfo.processInfo.environment["BENCH_FUSED_GATE_UP_THRESHOLD"] ?? "8") ?? 8
        let useFused =
            (fusedGateUpWeight != nil)
            && (indices.size <= decodeThreshold)

        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xGate: MLXArray
        let xUp: MLXArray
        if useFused, let fusedW = fusedGateUpWeight, let fusedS = fusedGateUpScales {
            // FUSED PATH — single gatherQuantizedMM for gate+up, then
            // split along output axis and apply compiled SwiGLU/GeGLU.
            // Decode-only per the threshold check above.
            let combined = MLX.gatherQuantizedMM(
                x, fusedW,
                scales: fusedS, biases: fusedGateUpBiases,
                rhsIndices: idx, transpose: true,
                groupSize: fusedGroupSize, bits: fusedBits, mode: fusedMode,
                sortedIndices: doSort)
            var splits = MLX.split(combined, parts: 2, axis: -1)
            // If the original SwitchLinear layers have a learned bias (separate
            // from quantization biases), add it to each half. Gemma4 uses
            // bias=false, but this ensures correctness for biased MoE layers.
            if let g = gateProj as? QuantizedSwitchLinear, let gb = g.bias,
               let u = upProj as? QuantizedSwitchLinear, let ub = u.bias {
                let fusedBias = concatenated([gb, ub], axis: -1)
                // Match the unfused QuantizedSwitchLinear path, which adds
                // `MLX.expandedDimensions(bias[indices], axis: -2)`. The fused
                // matmul result (`splits[*]`) is shaped [..., topk, 1, H]
                // because `x` was expanded on axes [-2, -3]. The gathered bias
                // is [..., topk, H]; without the singleton -2 axis it
                // right-aligns the bias's topk axis against the result's
                // matmul axis and mis-broadcasts to [..., topk, topk, H],
                // corrupting activations (or erroring). Expand on -2 so the
                // bias broadcasts over the matmul axis like the unfused path.
                let gatheredBias = MLX.expandedDimensions(fusedBias[idx], axis: -2)
                let biasSplits = MLX.split(gatheredBias, parts: 2, axis: -1)
                splits[0] = splits[0] + biasSplits[0]
                splits[1] = splits[1] + biasSplits[1]
            }
            xGate = splits[0]
            xUp = splits[1]
        } else if let gateUpProj {
            // Pre-fused gate_up_proj weight from checkpoint — one gathered
            // matmul via the polymorphic SwitchLinear call, then split.
            let xGateUp = gateUpProj(x, idx, sortedIndices: doSort)
            xGate = xGateUp[.ellipsis, ..<hiddenDims]
            xUp = xGateUp[.ellipsis, hiddenDims...]
        } else {
            // FALLBACK — original two-call path for non-quantized models,
            // prefill batches (indices.size > threshold), or when the
            // feature flag is off.
            guard let gateProj, let upProj else {
                fatalError("SwitchGLU requires either gate_up_proj or gate_proj/up_proj")
            }
            xUp = upProj(x, idx, sortedIndices: doSort)
            xGate = gateProj(x, idx, sortedIndices: doSort)
        }

        let activated: MLXArray
        if let activationProduct {
            activated = activationProduct(xGate, xUp)
        } else if isSiluActivation {
            activated = compiledSwiGLU(xGate, xUp)
        } else if isGeluActivation {
            activated = compiledGeGLU(xGate, xUp)
        } else {
            activated = activation(xGate) * xUp
        }

        x = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
