import MLX
import MLXNN
import Testing
@testable import MLXLLM

@Suite(.tags(.integration))
struct NemotronLatentMoETests {
    private func config(latent: Int?) -> NemotronHConfiguration {
        NemotronHConfiguration(vocabSize: 32, hiddenSize: 4, numHiddenLayers: 1,
            numAttentionHeads: 1, numKeyValueHeads: 1, mambaNumHeads: 1, mambaHeadDim: 4,
            ssmStateSize: 4, convKernel: 4, nGroups: 1, intermediateSize: 3,
            moeIntermediateSize: 3, moeSharedExpertIntermediateSize: 3,
            nRoutedExperts: 2, numExpertsPerTok: 2, hybridOverridePattern: "E",
            nSharedExperts: 1, routedScalingFactor: 5, moeLatentSize: latent)
    }
    @Test func latentRouterRoutedAndSharedMatchScalarOracle() throws {
        let moe = NemotronHMoE(config(latent: 2))
        let parameters: [String: MLXArray] = [
            "gate.weight": MLXArray([-0.1875, 0.25, -0.125, 0.3125, -0.0625, 0.375, 0.0, -0.375] as [Float]).reshaped(2, 4),
            "gate.e_score_correction_bias": MLXArray([0.05, -0.1] as [Float]).reshaped(2),
            "fc1_latent_proj.weight": MLXArray([-0.1875, 0.25, -0.125, 0.3125, -0.0625, 0.375, 0.0, -0.375] as [Float]).reshaped(2, 4),
            "fc2_latent_proj.weight": MLXArray([-0.1875, 0.25, -0.125, 0.3125, -0.0625, 0.375, 0.0, -0.375] as [Float]).reshaped(4, 2),
            "switch_mlp.fc1.weight": MLXArray([-0.1875, 0.25, -0.125, 0.3125, -0.0625, 0.375, 0.0, -0.375, 0.0625, -0.3125, 0.125, -0.25] as [Float]).reshaped(2, 3, 2),
            "switch_mlp.fc2.weight": MLXArray([-0.1875, 0.25, -0.125, 0.3125, -0.0625, 0.375, 0.0, -0.375, 0.0625, -0.3125, 0.125, -0.25] as [Float]).reshaped(2, 2, 3),
            "shared_experts.up_proj.weight": MLXArray([-0.1875, 0.25, -0.125, 0.3125, -0.0625, 0.375, 0.0, -0.375, 0.0625, -0.3125, 0.125, -0.25] as [Float]).reshaped(3, 4),
            "shared_experts.down_proj.weight": MLXArray([-0.1875, 0.25, -0.125, 0.3125, -0.0625, 0.375, 0.0, -0.375, 0.0625, -0.3125, 0.125, -0.25] as [Float]).reshaped(4, 3),
        ]
        try moe.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        let x = MLXArray([0.25, -0.5, 1, 0.75, -0.75, 0.5, 0.125, -0.25] as [Float]).reshaped(1, 2, 4)
        let actual = moe(x)
        let expected = MLXArray([-0.0044055452301415, -0.0009224292794231337, -0.007159161961517266, 0.005927996190059771, 0.024633265012978956, 0.006466651885275312, -0.035702036437740836, -0.00038579688475294614] as [Float]).reshaped(1, 2, 4)
        #expect(actual.shape == [1, 2, 4])
        #expect(abs(actual - expected).max().item(Float.self) < 0.000002)
        // The MTP row helper must retain both projections instead of its legacy fast path.
        #expect(abs(moe.mtpForwardRowsUncompiled(x) - expected).max().item(Float.self) < 0.000002)
    }
    @Test func legacyHasNoLatentModules() {
        let moe = NemotronHMoE(config(latent: nil))
        #expect(moe.fc1LatentProj == nil && moe.fc2LatentProj == nil)
        #expect(moe.switchMLP.inputDims == 4)
    }
}
