import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

// Serialized: `fusedGateUpQuantizedLearnedBiasMatchesUnfused` toggles the
// process-global BENCH_FUSED_GATE_UP_THRESHOLD env var, which other tests in
// this suite read in SwitchGLU.callAsFunction. Serializing avoids the race.
@Suite(.serialized)
struct SwitchGLUTests {
    @Test func fusedGateUpMatchesSeparateProjections() {
        let inputDims = 8
        let hiddenDims = 4
        let numExperts = 3

        let separate = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false)
        let fused = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false, fuseGateUp: true)

        let gate = values(numExperts * hiddenDims * inputDims)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let up = values(numExperts * hiddenDims * inputDims, offset: 1000)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let down = values(numExperts * inputDims * hiddenDims, offset: 2000)
            .reshaped(numExperts, inputDims, hiddenDims)
            .asType(.float32)

        separate.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": gate,
            "up_proj.weight": up,
            "down_proj.weight": down,
        ]))
        fused.update(parameters: ModuleParameters.unflattened([
            "gate_up_proj.weight": concatenated([gate, up], axis: -2),
            "down_proj.weight": down,
        ]))

        let x = values(2 * inputDims, offset: 3000)
            .reshaped(2, inputDims)
            .asType(.float32)
        let indices = MLXArray([Int32(0), 2, 1, 0]).reshaped(2, 2)

        let separateOut = separate(x, indices)
        let fusedOut = fused(x, indices)
        eval(separateOut, fusedOut)

        let maxDiff = max(abs(separateOut - fusedOut)).item(Float.self)
        #expect(maxDiff == 0)
    }

    /// N1 fast-follow: same parity check for the *quantized* fused path. Affine
    /// quantization is independent per output row, so quantizing the fused
    /// `[gate; up]` weight yields gate/up rows bit-identical to quantizing the
    /// two projections separately -- the single gathered quantized matmul + split
    /// must therefore match two separate quantized matmuls exactly.
    @Test func fusedGateUpQuantizedMatchesSeparateProjections() {
        let inputDims = 64
        let hiddenDims = 64
        let numExperts = 3
        let groupSize = 64
        let bits = 4

        let separate = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false)
        let fused = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false, fuseGateUp: true)

        let gate = values(numExperts * hiddenDims * inputDims)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let up = values(numExperts * hiddenDims * inputDims, offset: 20000)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let down = values(numExperts * inputDims * hiddenDims, offset: 40000)
            .reshaped(numExperts, inputDims, hiddenDims)
            .asType(.float32)

        separate.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": gate,
            "up_proj.weight": up,
            "down_proj.weight": down,
        ]))
        fused.update(parameters: ModuleParameters.unflattened([
            "gate_up_proj.weight": concatenated([gate, up], axis: -2),
            "down_proj.weight": down,
        ]))

        quantize(model: separate, groupSize: groupSize, bits: bits)
        quantize(model: fused, groupSize: groupSize, bits: bits)

        let x = values(2 * inputDims, offset: 60000)
            .reshaped(2, inputDims)
            .asType(.float32)
        let indices = MLXArray([Int32(0), 2, 1, 0]).reshaped(2, 2)

        let separateOut = separate(x, indices)
        let fusedOut = fused(x, indices)
        eval(separateOut, fusedOut)

        let maxDiff = max(abs(separateOut - fusedOut)).item(Float.self)
        #expect(maxDiff == 0)
    }

    /// The runtime-fused gate+up DECODE path (built by `ensureFusedGateUp`)
    /// must add the learned per-expert bias with the same broadcast as the
    /// unfused `QuantizedSwitchLinear` path, which uses
    /// `MLX.expandedDimensions(bias[indices], axis: -2)`. Regression test for
    /// the missing `-2` expansion: without it the gathered bias mis-broadcasts
    /// across the top-k axis (→ wrong activations / shape blow-up). Gemma4 ships
    /// `bias=false`, so this guards biased MoE models on the fused path.
    @Test func fusedGateUpQuantizedLearnedBiasMatchesUnfused() {
        let inputDims = 64
        let hiddenDims = 64
        let numExperts = 4
        let groupSize = 64
        let bits = 4

        let glu = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: true)

        let gate = values(numExperts * hiddenDims * inputDims)
            .reshaped(numExperts, hiddenDims, inputDims).asType(.float32)
        let up = values(numExperts * hiddenDims * inputDims, offset: 20000)
            .reshaped(numExperts, hiddenDims, inputDims).asType(.float32)
        let down = values(numExperts * inputDims * hiddenDims, offset: 40000)
            .reshaped(numExperts, inputDims, hiddenDims).asType(.float32)
        let gateBias = values(numExperts * hiddenDims, offset: 5000)
            .reshaped(numExperts, hiddenDims).asType(.float32)
        let upBias = values(numExperts * hiddenDims, offset: 7000)
            .reshaped(numExperts, hiddenDims).asType(.float32)
        let downBias = values(numExperts * inputDims, offset: 9000)
            .reshaped(numExperts, inputDims).asType(.float32)

        glu.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": gate, "gate_proj.bias": gateBias,
            "up_proj.weight": up, "up_proj.bias": upBias,
            "down_proj.weight": down, "down_proj.bias": downBias,
        ]))
        quantize(model: glu, groupSize: groupSize, bits: bits)

        // Solo-decode shape: indices.size = top_k (<= fused threshold 8).
        let x = values(inputDims, offset: 60000).reshaped(1, inputDims).asType(.float32)
        let indices = MLXArray([Int32(0), 1, 2]).reshaped(1, 3)

        // Reference: force the UNFUSED path (threshold 0 => useFused == false).
        // The first call also builds the fused weight via ensureFusedGateUp.
        setenv("BENCH_FUSED_GATE_UP_THRESHOLD", "0", 1)
        let unfused = glu(x, indices)
        eval(unfused)
        unsetenv("BENCH_FUSED_GATE_UP_THRESHOLD")

        // Fused path (default threshold 8 => useFused for indices.size == 3).
        let fused = glu(x, indices)
        eval(fused)

        let maxDiff = max(abs(unfused - fused)).item(Float.self)
        #expect(maxDiff < 1e-4)
    }

    private func values(_ count: Int, offset: Int = 0) -> MLXArray {
        MLXArray((offset ..< offset + count).map { Float($0) / 1000 })
    }
}
