import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.tags(.integration))
struct NemotronCalibratedProjectionTests {
    private func config() -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 32, hiddenSize: 4, numHiddenLayers: 1,
            numAttentionHeads: 1, numKeyValueHeads: 1, mambaNumHeads: 1, mambaHeadDim: 4,
            ssmStateSize: 4, convKernel: 4, nGroups: 1, intermediateSize: 4,
            moeIntermediateSize: 4, moeSharedExpertIntermediateSize: 4,
            nRoutedExperts: 2, numExpertsPerTok: 2, hybridOverridePattern: "E",
            nSharedExperts: 1)
    }

    private func projection(input: Int, output: Int) throws -> ModelOptCalibratedLinear {
        let bytes = (0 ..< (input * output)).map { UInt8($0 / input == $0 % input ? 56 : 0) }
        let weights = try ModelOptFP8Weights(
            e4m3Bytes: MLXArray(bytes).reshaped(output, input), tensorScale: 1)
        return try ModelOptCalibratedLinear(weights: weights, inputScale: 1, weightDType: .float32)
    }

    private func install(_ module: Module, paths: [String]) throws {
        let children = Dictionary(uniqueKeysWithValues: module.leafModules().flattened())
        for path in paths {
            let original = try #require(children[path] as? Linear)
            let (output, input) = original.shape
            let replacement = try projection(input: input, output: output)
            #expect(replacement.inputDimensions == input && replacement.outputDimensions == output)
            // Module update verifies type/key compatibility; the loader must separately
            // validate dimensions (as done above), dtype and checkpoint provenance.
            try module.update(modules: .unflattened([path: replacement as Module]), verify: .all)
        }
    }

    @Test func allRequestedSlotsAcceptCalibratedModules() throws {
        let mamba = NemotronHMamba2Mixer(config())
        try install(mamba, paths: ["in_proj", "out_proj"])
        #expect(
            mamba.inProj is ModelOptCalibratedLinear && mamba.outProj is ModelOptCalibratedLinear)
        let attention = NemotronHAttention(config())
        try install(attention, paths: ["q_proj", "k_proj", "v_proj", "o_proj"])
        #expect(
            attention.wq is ModelOptCalibratedLinear && attention.wo is ModelOptCalibratedLinear)
        let x = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 1, 4)
        #expect(arrayEqual(attention.wq(x), x).item(Bool.self))
        #expect(arrayEqual(nemotronMTPLinearRows(x, attention.wq), x).item(Bool.self))
    }

    @Test func sharedMLPMatchesEquationAndBypassesCompiledCache() throws {
        let moe = NemotronHMoE(config())
        let shared = try #require(moe.sharedExperts)
        try install(shared, paths: ["up_proj", "down_proj"])
        let x = MLXArray([Float(-1), 2, 3, 4, 1, -2, 2, 0]).reshaped(1, 2, 4)
        let expected = MLXArray([Float(0), 4, 9, 16, 1, 0, 4, 0]).reshaped(1, 2, 4)
        #expect(arrayEqual(shared(x), expected).item(Bool.self))
        #expect(arrayEqual(shared.mtpForwardRows(x), expected).item(Bool.self))
        #expect(moe.hasHostCalibratedProjections)
        let cache = NemotronH35MTPGraphCache()
        let actual = cache(x, owner: moe)
        let reference = moe.mtpForwardRowsUncompiled(x)
        #expect(abs(actual - reference).max().item(Float.self) < 0.00001)
        #expect(cache.generationCount == 0 && cache.callCount == 0)
        _ = moe.mtpForwardRows(x)
        #expect(moe.mtpCompiledRowsCache.generationCount == 0)
    }

    @Test func installationPreflightRejectsWholeBatchWithoutMutation() throws {
        let model = NemotronHModel(config())
        let before = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let path = "backbone.layers.0.mixer.shared_experts.up_proj"
        let original = try #require(before[path] as? Linear)
        let valid = try projection(input: original.shape.1, output: original.shape.0)
        #expect(throws: (any Error).self) {
            try model.installModelOptProjections([path: valid, "lm_head": valid])
        }
        let after = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        for (name, value) in before { #expect(after[name] === value) }
        #expect(throws: (any Error).self) {
            try model.installModelOptProjections([path: projection(input: 1, output: 1)])
        }
        try model.installModelOptProjections([path: valid])
        #expect(Dictionary(uniqueKeysWithValues: model.leafModules().flattened())[path] === valid)
        // A second installation cannot reinterpret an already calibrated module.
        #expect(throws: (any Error).self) { try model.installModelOptProjections([path: valid]) }
    }

    @Test func packedExpertInstallationChecksDimensionsAndWholeBatch() throws {
        var configuration = config()
        configuration.hiddenSize = 16
        configuration.moeIntermediateSize = 16
        let model = NemotronHModel(configuration)
        let matrix = try ModelOptNVFP4Weights(
            packedBytes: MLXArray([UInt8](repeating: 0x22, count: 128)).reshaped(16, 8),
            e4m3ScaleBytes: MLXArray([UInt8](repeating: 56, count: 16)).reshaped(16, 1),
            tensorScale: 1)
        let packed = try ModelOptNVFP4SwitchLinear(
            experts: [matrix, matrix], inputScales: [1, 1], weightDType: .float32)
        let path = "backbone.layers.0.mixer.switch_mlp.fc1"
        let before = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        #expect(throws: (any Error).self) {
            try model.installModelOptExperts([path: packed, "lm_head": packed])
        }
        #expect(
            Dictionary(uniqueKeysWithValues: model.leafModules().flattened())[path] === before[path]
        )
        let wrongCount = try ModelOptNVFP4SwitchLinear(
            experts: [matrix], inputScales: [1], weightDType: .float32)
        #expect(throws: (any Error).self) { try model.installModelOptExperts([path: wrongCount]) }
        try model.installModelOptExperts([path: packed])
        #expect(Dictionary(uniqueKeysWithValues: model.leafModules().flattened())[path] === packed)
        #expect(throws: (any Error).self) { try model.installModelOptExperts([path: packed]) }
    }

}
