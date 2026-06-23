import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite
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

    private func values(_ count: Int, offset: Int = 0) -> MLXArray {
        MLXArray((offset ..< offset + count).map { Float($0) / 1000 })
    }
}
