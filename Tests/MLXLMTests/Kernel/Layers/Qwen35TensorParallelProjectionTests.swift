import MLX
import MLXLMCommon
import MLXNN
import Testing

extension KernelTests {
    @Suite("Qwen tensor expert projection")
    struct Qwen35TensorParallelProjectionTests {
        private func source(bias: Bool = false) -> QuantizedSwitchLinear {
            let values: [Float] = (0 ..< (4 * 64 * 128)).map { index in
                let integer: Int = (index * 31 + 17) % 257 - 128
                return Float(integer) / Float(137)
            }
            let weight = MLXArray(values, [4, 64, 128]).asType(.bfloat16)
            return QuantizedSwitchLinear(
                SwitchLinear(
                    inputDims: 128, outputDims: 64,
                    numExperts: 4, weight: weight,
                    bias: bias ? MLXArray.ones([4, 64], dtype: .bfloat16) : nil),
                groupSize: 64, bits: 4)
        }

        @Test func sumSeesUnroundedExpertRowsAndRoundsOnlyAfterPeerContribution() throws {
            try Device.withDefaultDevice(.cpu) {
                let packed = source()
                for rows in [1, 8, 64] {
                    let x = MLXArray(
                        (0 ..< (rows * 128)).map {
                            Float(($0 * 19 + 3) % 127 - 63) / 91
                        }, [rows, 1, 128]
                    ).asType(.bfloat16)
                    let ids = MLXArray((0 ..< rows).map { UInt32($0 * 4 / rows) })
                    let expectedPartial = packed(x.asType(.float32), ids, sortedIndices: true)
                    let peer = MLXArray(
                        (0 ..< (rows * 64)).map {
                            Float(($0 * 7 + 5) % 31 - 15) / 193
                        }, [rows, 1, 64])
                    var calls = 0
                    let projection = try Qwen35TensorParallelSwitchLinear(packed) { partial in
                        calls += 1
                        #expect(partial.dtype == .float32)
                        #expect(partial.shape == [rows, 1, 64])
                        #expect(arrayEqual(partial, expectedPartial).item(Bool.self))
                        return partial + peer
                    }
                    let result = projection(x, ids, sortedIndices: true)
                    let expected = (expectedPartial + peer).asType(.bfloat16)
                    let prematurelyRounded =
                        (expectedPartial.asType(.bfloat16).asType(.float32)
                        + peer).asType(.bfloat16)
                    eval(result, expected, prematurelyRounded)
                    #expect(calls == 1)
                    #expect(result.dtype == .bfloat16)
                    #expect(arrayEqual(result, expected).item(Bool.self))
                    #expect(!arrayEqual(result, prematurelyRounded).item(Bool.self))
                }
            }
        }

        @Test func switchGLUCallsTheCollectiveBeforeRouterWeighting() throws {
            try Device.withDefaultDevice(.cpu) {
                let glu = SwitchGLU(
                    inputDims: 64, hiddenDims: 128, numExperts: 4,
                    fuseGateUp: true)
                let values: [Float] = (0 ..< (4 * 256 * 64)).map { index in
                    let integer: Int = (index * 11 + 3) % 127 - 63
                    return Float(integer) / Float(197)
                }
                let gateUpWeight = MLXArray(values, [4, 256, 64]).asType(.bfloat16)
                let gateUp = QuantizedSwitchLinear(
                    SwitchLinear(
                        inputDims: 64, outputDims: 256,
                        numExperts: 4, weight: gateUpWeight), groupSize: 64, bits: 4)
                let down = source()
                var calls = 0
                let reduced = try Qwen35TensorParallelSwitchLinear(down) { partial in
                    calls += 1
                    #expect(partial.dtype == .float32 && partial.dim(-1) == 64)
                    // A recognizable completed expert value makes misplaced
                    // weighting, route reduction or a bypass observable.
                    return MLXArray.ones(partial.shape, dtype: .float32) * Float(2.25)
                }
                try glu.update(
                    modules: ModuleChildren.unflattened([
                        ("gate_up_proj", gateUp), ("down_proj", reduced),
                    ]), verify: .none)
                for rows in [1, 32] {
                    let x = MLXArray.ones([rows, 64], dtype: .bfloat16)
                    let ids = MLXArray((0 ..< (rows * 2)).map { UInt32($0 % 4) }, [rows, 2])
                    let weights = broadcast(
                        MLXArray([Float(0.125), 0.5]).asType(.bfloat16), to: [rows, 2])
                    let result = glu.callAndWeightedReduce(
                        x, ids, weights: weights,
                        fuseSortedReduction: true)
                    let expected = weightedExpertSum(
                        MLXArray.ones([rows, 2, 64], dtype: .bfloat16)
                            * Float(2.25), weights)
                    eval(result, expected)
                    #expect(arrayEqual(result, expected).item(Bool.self))
                }
                #expect(calls == 2)
            }
        }

        @Test func replicatedBiasCannotBeReducedTwice() throws {
            Device.withDefaultDevice(.cpu) {
                _ = #expect(throws: Qwen35TensorParallelSwitchLinear.InstallationError.self) {
                    try Qwen35TensorParallelSwitchLinear(source(bias: true), allSum: { $0 })
                }
            }
        }
    }
}
