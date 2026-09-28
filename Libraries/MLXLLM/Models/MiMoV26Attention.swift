// Copyright © 2026 Eigen Labs.
// Native MiMo V2.6 text attention. Architecture reference: XiaomiMiMo
// MiMo-V2.6-Flash-RL, revision 5711b268169967567844e1e560e8a3966da959b1.

import MLX
import MLXLMCommon
import MLXNN

/// Unequal K/V widths are intentional. Do not pad values to the key width:
/// that changes the cache/storage contract and adds unnecessary residency.
final class MiMoV26Attention: Module {
    let geometry: MiMoV26AttentionGeometry
    let scale: Float
    let valueScale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "attention_sink_bias") var attentionSinkBias: MLXArray?

    init(_ config: MiMoV26Configuration, geometry: MiMoV26AttentionGeometry) {
        self.geometry = geometry
        scale = 1 / Float(geometry.headDim).squareRoot()
        valueScale = Float(config.attentionValueScale)
        rope = RoPE(dimensions: geometry.rotaryDimensions, traditional: false,
                    base: Float(geometry.ropeTheta))
        _qProj.wrappedValue = Linear(config.hiddenSize, geometry.queryHeads * geometry.headDim,
                                    bias: config.attentionBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, geometry.keyValueHeads * geometry.headDim,
                                    bias: config.attentionBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, geometry.keyValueHeads * geometry.valueHeadDim,
                                    bias: config.attentionBias)
        _oProj.wrappedValue = Linear(geometry.queryHeads * geometry.valueHeadDim,
                                    config.hiddenSize, bias: false)
        _attentionSinkBias.wrappedValue = geometry.hasSinks ? MLXArray.zeros([geometry.queryHeads]) : nil
        // No updateMissing override: a missing learned sink must fail strict loading.
    }

    /// The text component validates cache kinds before any layer mutates state.
    /// CBv2 integration needs its own unequal-width cache contract and is not
    /// enabled by this ordinary-cache component.
    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(batch, length, geometry.queryHeads, geometry.headDim)
            .transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, length, geometry.keyValueHeads, geometry.headDim)
            .transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, length, geometry.keyValueHeads, geometry.valueHeadDim)
            .transposed(0, 2, 1, 3)
        // Scaling is part of the native model and happens before cache insertion.
        v = v * valueScale
        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }
        let attended = MiMoV26NAXAttention.tryAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask,
            sinks: attentionSinkBias)
            ?? MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask,
                sinks: attentionSinkBias)
        return oProj(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}
