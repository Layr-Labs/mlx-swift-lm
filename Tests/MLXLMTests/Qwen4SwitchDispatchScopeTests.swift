import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Run this filter alone when using the native-dispatch counter assertions.
/// The fixtures share projection objects across owners to detect profile leaks.
@Suite("Qwen4 expert dispatch ownership", .serialized)
struct Qwen4SwitchDispatchScopeTests {
    private func stockProjection(
        _ layer: QuantizedSwitchLinear, _ x: MLXArray, _ indices: MLXArray
    ) -> MLXArray {
        MLX.gatherQuantizedMM(
            x, layer.weight, scales: layer.scales, biases: layer.biases,
            rhsIndices: indices, transpose: true, groupSize: layer.groupSize,
            bits: layer.bits, mode: layer.mode, sortedIndices: false)
    }

    @Test func directUnrelatedE512ProjectionRetainsStockDtypeAndNeverAttemptsQwen() {
        let weights = MLXRandom.normal([512, 32, 64], key: MLXRandom.key(8340))
            .asType(.bfloat16)
        let x = MLXArray.ones([1, 1, 1, 64], dtype: .float32) * 0.1234567
        let indices = MLXArray([UInt32(0), 511], [1, 2])
        // Affine can reach a Qwen candidate; MXFP4 must take its ordinary
        // fallback. Previously both paths could be narrowed just because E=512.
        for mode in [QuantizationMode.affine, .mxfp4] {
            let layer = QuantizedSwitchLinear(
                SwitchLinear(inputDims: 64, outputDims: 32, numExperts: 512, weight: weights),
                groupSize: mode == .affine ? 64 : 32, bits: 4, mode: mode)
            let before = Qwen4ExpGatherQMMInvocation.snapshot()
            let actual = layer(x, indices)
            let after = Qwen4ExpGatherQMMInvocation.snapshot()
            let expected = stockProjection(layer, x, indices)
            eval(actual, expected)
            #expect(actual.dtype == .float32)
            #expect(actual.dtype == expected.dtype)
            #expect(actual.asData().data == expected.asData().data)
            #expect(after == before)
        }
    }

    private func prepare(_ glu: SwitchGLU) {
        for (_, module) in glu.leafModules().flattened() {
            if let projection = module as? SwitchLinear, !(projection is Quantized) {
                projection.update(parameters: ModuleParameters.unflattened([
                    ("weight", projection.weight.asType(.bfloat16))
                ]))
            }
        }
        quantize(model: glu, groupSize: 64, bits: 4, mode: .affine)
        eval(glu)
    }

    private func owner(_ profile: SwitchGLUWeightedReductionProfile) -> SwitchGLU {
        let glu = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 512,
            fuseGateUp: true, weightedReductionProfile: profile)
        prepare(glu)
        return glu
    }

    /// Reconstruct the already-selected primitive math independently of the
    /// owning SwitchGLU dispatch. Top-K one avoids a sorting/reduction change.
    private func reference(
        _ glu: SwitchGLU, _ x: MLXArray, _ indices: MLXArray, nativeQwen4: Bool
    ) throws -> MLXArray {
        func project(_ projection: SwitchLinear, _ input: MLXArray) throws -> MLXArray {
            let layer = try #require(projection as? QuantizedSwitchLinear)
            if nativeQwen4 {
                return try #require(Qwen4ExpGatherQMM.tryMatmul(
                    x: input, indices: indices, weight: layer.weight,
                    scales: layer.scales, affineBiases: layer.biases, sorted: false,
                    bits: layer.bits, groupSize: layer.groupSize, mode: layer.mode))
            }
            return stockProjection(layer, input, indices)
        }
        let input = expandedDimensions(x, axes: [-2, -3])
        let gate: MLXArray
        let up: MLXArray
        if let combined = glu.gateUpProj {
            let values = try project(combined, input)
            gate = values[.ellipsis, ..<glu.hiddenDims]
            up = values[.ellipsis, glu.hiddenDims...]
        } else {
            gate = try project(#require(glu.gateProj), input)
            up = try project(#require(glu.upProj), input)
        }
        return try project(glu.downProj, compiledSiluProduct(gate, up)).squeezed(axis: -2)
    }

    private func check(_ glu: SwitchGLU, nativeQwen4: Bool) throws {
        let x = MLXRandom.normal([1, 64], key: MLXRandom.key(8341))
        let indices = MLXArray([UInt32(511)], [1, 1])
        let before = Qwen4ExpGatherQMMInvocation.snapshot()
        let actual = glu(x, indices)
        let after = Qwen4ExpGatherQMMInvocation.snapshot()
        let expected = try reference(glu, x, indices, nativeQwen4: nativeQwen4)
        eval(actual, expected)
        #expect(actual.dtype == (nativeQwen4 ? .bfloat16 : .float32))
        #expect(actual.dtype == expected.dtype)
        #expect(actual.asData().data == expected.asData().data)
        if nativeQwen4 {
            #expect(after.native - before.native == (glu.hasFusedGateUp ? 2 : 3))
        } else {
            #expect(after == before)
        }
    }

    @Test func ownerProfileSurvivesQuantizationReplacementAndCannotLeakThroughSharedChildren() throws {
        let generic = owner(.generic)
        let qwen = owner(.qwen4ProductionSwiGLU)
        // Exactly the same quantized children are called with different owner
        // profiles. No child mutation is allowed to turn on the generic owner.
        qwen.update(modules: ModuleChildren.unflattened(generic.leafModules().flattened()))
        try check(generic, nativeQwen4: false)
        try check(qwen, nativeQwen4: true)
        try check(generic, nativeQwen4: false)

        for (original, native) in [(generic, false), (qwen, true)] {
            let split = original.splittingGateUp()
            prepare(split)
            #expect(!split.hasFusedGateUp)
            try check(split, nativeQwen4: native)
            let fused = split.fusingGateUp()
            prepare(fused)
            #expect(fused.hasFusedGateUp)
            try check(fused, nativeQwen4: native)
            // Direct use of even a Qwen-owned child remains the generic API.
            let child = try #require(fused.downProj as? QuantizedSwitchLinear)
            let input = MLXArray.ones([1, 1, 1, 64], dtype: .float32)
            let indices = MLXArray([UInt32(0)], [1, 1])
            let before = Qwen4ExpGatherQMMInvocation.snapshot()
            let actual = child(input, indices)
            let after = Qwen4ExpGatherQMMInvocation.snapshot()
            let expected = stockProjection(child, input, indices)
            eval(actual, expected)
            #expect(actual.dtype == .float32)
            #expect(actual.asData().data == expected.asData().data)
            #expect(after == before)
        }
    }
}
