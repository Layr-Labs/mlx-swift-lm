import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The fused Super MoE kernels against the op chains they replace: expert
/// selection against `groupExpertSelect` of the same gate (indices and scores,
/// including tied scores, saturated sigmoids and NaN rows) and the weighted
/// expert sum, bit for bit. Synthetic values only.
@Suite(.tags(.integration))
struct NemotronSuperMoEKernelTests {
    static func gate(experts: Int, topK: Int, normalize: Bool, hidden: Int = 64)
        -> NemotronHMoEGate
    {
        NemotronHMoEGate(
            NemotronHConfiguration(
                vocabSize: 16, hiddenSize: hidden, numHiddenLayers: 1, numAttentionHeads: 1,
                numKeyValueHeads: 1, mambaNumHeads: 1, mambaHeadDim: 8, ssmStateSize: 8,
                convKernel: 4, nGroups: 1, intermediateSize: 8, moeIntermediateSize: 8,
                moeSharedExpertIntermediateSize: 8, nRoutedExperts: experts,
                numExpertsPerTok: topK, hybridOverridePattern: "E", normTopkProb: normalize,
                routedScalingFactor: 5))
    }

    /// Both selections of `gate` for `x`, fused second.
    static func both(_ gate: NemotronHMoEGate, _ x: MLXArray) -> [MLXArray] {
        gate.fusedSuperSelection = false
        let (referenceIndices, referenceScores) = gate(x)
        gate.fusedSuperSelection = true
        let (indices, scores) = gate(x)
        eval(referenceIndices, referenceScores, indices, scores)
        return [referenceIndices, referenceScores, indices, scores]
    }

    static func expectEqual(_ outputs: [MLXArray]) {
        #expect(outputs[2].dtype == outputs[0].dtype)
        #expect(outputs[3].dtype == outputs[1].dtype)
        #expect(outputs[2].shape == outputs[0].shape)
        #expect(modelOptBitwiseEqual(outputs[0], outputs[2]))
        #expect(modelOptBitwiseEqual(outputs[1], outputs[3]))
    }

    @Test(arguments: [(512, 22, true), (512, 22, false), (64, 8, true), (40, 1, true)])
    func fusedSelectionMatchesReference(experts: Int, topK: Int, normalize: Bool) throws {
        let gate = Self.gate(experts: experts, topK: topK, normalize: normalize)
        try gate.update(
            parameters: .unflattened([
                "weight": (MLXRandom.normal([experts, 64], key: MLXRandom.key(3)) * 0.2)
                    .asType(.bfloat16),
                "e_score_correction_bias": MLXRandom.normal([experts], key: MLXRandom.key(4))
                    * 0.05,
            ]), verify: .all)
        for (rows, seed) in [(1, 1), (5, 2), (37, 3), (300, 4)] {
            let x = (MLXRandom.normal([1, rows, 64], key: MLXRandom.key(UInt64(seed))) * 2)
                .asType(.bfloat16)
            Self.expectEqual(Self.both(gate, x))
        }
    }

    /// Duplicated router rows and a zero bias give exactly tied scores; huge
    /// logits saturate the sigmoid to 0 and 1; NaN inputs give NaN rows.
    @Test func tiesSaturationAndNaNMatchReference() throws {
        let experts = 512
        let gate = Self.gate(experts: experts, topK: 22, normalize: true)
        let base = (MLXRandom.normal([16, 64], key: MLXRandom.key(9)) * 0.3).asType(.bfloat16)
        let tiled = concatenated(Array(repeating: base, count: experts / 16), axis: 0)
        try gate.update(
            parameters: .unflattened([
                "weight": tiled, "e_score_correction_bias": MLXArray.zeros([experts]),
            ]), verify: .all)
        let x = (MLXRandom.normal([1, 7, 64], key: MLXRandom.key(10))).asType(.bfloat16)
        Self.expectEqual(Self.both(gate, x))
        let saturated = (x * 1e4).asType(.bfloat16)
        Self.expectEqual(Self.both(gate, saturated))
        let poisoned = which(
            MLXArray([true, false, true]).reshaped(1, 3, 1),
            MLXArray(Float.nan).asType(.bfloat16), x[0..., 0 ..< 3, 0...])
        Self.expectEqual(Self.both(gate, poisoned))
    }

    @Test func fusedSelectionNeedsOneGroup() {
        #expect(!Self.gate(experts: 512, topK: 22, normalize: true).fusedSuperSelection)
        #expect(NemotronSuperMoEKernels.supports(experts: 512, topK: 22, groups: 1))
        #expect(!NemotronSuperMoEKernels.supports(experts: 512, topK: 22, groups: 2))
    }

    @Test(arguments: [
        (1, 22, 1024), (37, 22, 64), (15, 3, 96), (4, 1, 40), (2, 31, 72), (3, 9, 33),
    ])
    func fusedCombineMatchesReference(rows: Int, experts: Int, width: Int) {
        for dtype in [DType.bfloat16, .float32] {
            let y = (MLXRandom.normal([1, rows, experts, width], key: MLXRandom.key(5)) * 3)
                .asType(dtype)
            var scores = MLXRandom.uniform(0 ..< 1, [1, rows, experts], key: MLXRandom.key(6))
            if rows > 2 {
                // NaN and infinite routing weights propagate identically.
                scores = which(
                    MLXArray(0 ..< rows * experts).reshaped(1, rows, experts) .== 3,
                    MLXArray(Float.nan), scores)
                scores = which(
                    MLXArray(0 ..< rows * experts).reshaped(1, rows, experts) .== 7,
                    MLXArray(Float.infinity), scores)
            }
            let reference = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
            let fused = NemotronSuperMoEKernels.combine(y, scores: scores)
            eval(reference, fused)
            #expect(fused.shape == reference.shape && fused.dtype == reference.dtype)
            #expect(modelOptBitwiseEqual(reference, fused))
        }
        #expect(NemotronSuperMoEKernels.supportsCombine(experts: 31))
        #expect(!NemotronSuperMoEKernels.supportsCombine(experts: 32))
    }
}
