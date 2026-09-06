// Copyright © 2026 Eigen Labs.
//
// DIAGNOSTIC ONLY — never on a scored path. Ruled by David 2026-09-06: run
// the numerics-changing levers on a scratch branch with exactness waived and
// report how far decode gets; that number decides the tape re-record.
//
// This lever: the whole forward of a gated-deltanet decoder layer (mixer,
// GDN, inject, mixer, MoE, inject) as ONE `compile`d function per layer, the
// conv and SSM states as explicit inputs and outputs, swapped into the LEGACY
// forward at single-token steps. If a per-layer compile amortises its ~112
// nodes per call, `bench-worker diag-parity`'s legacy column drops; if not,
// MLX's fused graph does not pay at real dimensions.
//
// MLXLM_SCRATCH_LAYER_COMPILE=1 enables it. Layers with a PLE block keep the
// eager path (their n-gram gather is host work).

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

let qwen4ExpScratchLayerCompileEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXLM_SCRATCH_LAYER_COMPILE"] == "1"

extension Qwen4ExpGatedDeltaNet {
    /// The legacy forward with the states as arguments instead of a cache.
    func scratchForward(
        _ x: MLXArray, convState: MLXArray, ssmState: MLXArray
    ) -> (out: MLXArray, convState: MLXArray, ssmState: MLXArray) {
        let B = x.dim(0)
        let S = x.dim(1)
        let mixedQKV = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, valueHeads, valueHeadDim)
        let b = inProjB(x)
        let a = inProjA(x)
        let convInput = concatenated([convState, mixedQKV], axis: 1)
        let newConvState = contiguous(convInput[0..., (1 - convKernelSize)..., 0...])
        let convOut = silu(conv1d(convInput))
        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = parts[0].reshaped(B, S, keyHeads, keyHeadDim)
        var k = parts[1].reshaped(B, S, keyHeads, keyHeadDim)
        let v = parts[2].reshaped(B, S, valueHeads, valueHeadDim)
        let invScale = Foundation.pow(Float(keyHeadDim), -0.5)
        q = MLXArray(invScale * invScale).asType(x.dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k = MLXArray(invScale).asType(x.dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)
        let (out, state) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: ssmState, mask: nil)
        return (outProj(norm(out, gate: z).reshaped(B, S, -1)), newConvState, state)
    }

    var scratchConvShape: (kernel: Int, dim: Int) { (convKernelSize, convDim) }
    var scratchSSMShape: [Int] { [1, valueHeads, valueHeadDim, keyHeadDim] }
}

/// One compiled function per layer, built on first use.
final class Qwen4ExpScratchCompiledLayer {
    private var compiled: (([MLXArray]) -> [MLXArray])?

    func function(for layer: Qwen4ExpDecoderLayer) -> ([MLXArray]) -> [MLXArray] {
        if let compiled { return compiled }
        let f: ([MLXArray]) -> [MLXArray] = compile(inputs: [layer], outputs: [layer]) {
            (inputs: [MLXArray]) -> [MLXArray] in
            let hyper = inputs[0]
            let convState = inputs[1]
            let ssmState = inputs[2]
            var (input, residual, inject) = layer.attnHyperConnection.mixWithInject(hyper)
            let gdn = layer.linearAttn!.scratchForward(
                input, convState: convState, ssmState: ssmState)
            var stream = qwen4ExpInject(residual: residual, output: gdn.out, inject: inject)
            (input, residual, inject) = layer.mlpHyperConnection.mixWithInject(stream)
            stream = qwen4ExpInject(residual: residual, output: layer.mlp(input), inject: inject)
            return [stream, gdn.convState, gdn.ssmState]
        }
        compiled = f
        return f
    }
}

extension Qwen4ExpDecoderLayer {
    /// The compiled path, or nil when this call must take the eager path.
    func scratchCompiledStep(
        _ hyper: MLXArray, convMask: MLXArray?, cache: KVCache?
    ) -> MLXArray? {
        guard qwen4ExpScratchLayerCompileEnabled, isLinear, ple == nil, convMask == nil,
            hyper.dim(1) == 1, let cache = cache as? Qwen4ExpLayerCache, let gdn = linearAttn
        else { return nil }
        let B = hyper.dim(0)
        let conv = gdn.scratchConvShape
        let convState =
            cache[Qwen4ExpLayerCache.deltaConvSlot]
            ?? MLXArray.zeros([B, conv.kernel - 1, conv.dim], dtype: hyper.dtype)
        let ssmState =
            cache[Qwen4ExpLayerCache.deltaStateSlot]
            ?? MLXArray.zeros(gdn.scratchSSMShape, dtype: .float32)
        let outputs = scratchCompiled.function(for: self)([hyper, convState, ssmState])
        cache[Qwen4ExpLayerCache.deltaConvSlot] = outputs[1]
        cache[Qwen4ExpLayerCache.deltaStateSlot] = outputs[2]
        cache.advance(1)
        return outputs[0]
    }
}
