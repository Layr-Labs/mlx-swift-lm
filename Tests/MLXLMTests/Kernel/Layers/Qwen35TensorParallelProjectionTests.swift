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
                    let inputValues: [Float] = (0 ..< (rows * 128)).map { index in
                        let integer: Int = (index * 19 + 3) % 127 - 63
                        return Float(integer) / Float(91)
                    }
                    let x = MLXArray(inputValues, [rows, 1, 128]).asType(.bfloat16)
                    let ids = MLXArray((0 ..< rows).map { UInt32($0 * 4 / rows) })
                    let expectedPartial = packed(x.asType(.float32), ids, sortedIndices: true)
                    let peerValues: [Float] = (0 ..< (rows * 64)).map { index in
                        let integer: Int = (index * 7 + 5) % 31 - 15
                        return Float(integer) / Float(193)
                    }
                    let peer = MLXArray(peerValues, [rows, 1, 64])
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

        /// Packed rows ride behind the expert rows of the one call inside
        /// `carrying`; an element-wise peer sum (what the collective is)
        /// gives every value exactly what two separate collectives give.
        @Test func carriedRowsShareTheExpertCollectiveWithoutChangingAnyValue() throws {
            try Device.withDefaultDevice(.cpu) {
                func peer(_ value: MLXArray) -> MLXArray { value * Float(0.75) - Float(0.03125) }
                let packed = source()
                let rows = 8
                let inputValues: [Float] = (0 ..< (rows * 128)).map { index in
                    Float((index * 19 + 3) % 127 - 63) / Float(91)
                }
                let x = MLXArray(inputValues, [rows, 1, 128]).asType(.bfloat16)
                let ids = MLXArray((0 ..< rows).map { UInt32($0 % 4) })
                let carriedValues: [Float] = (0 ..< (3 * 64)).map { index in
                    Float((index * 7 + 5) % 31 - 15) / Float(193)
                }
                let carried = MLXArray(carriedValues, [1, 3, 64])
                var payloads: [[Int]] = []
                let projection = try Qwen35TensorParallelSwitchLinear(packed) { value in
                    payloads.append(value.shape)
                    #expect(value.dtype == .float32)
                    return value + peer(value)
                }
                let other = try Qwen35TensorParallelSwitchLinear(packed) { $0 + peer($0) }
                // Another projection inside the scope keeps its own collective.
                let (output, rowsSum) = projection.carrying(carried) {
                    _ = other(x, ids)
                    return projection(x, ids)
                }
                let partial = packed(x.asType(.float32), ids)
                let expected = (partial + peer(partial)).asType(.bfloat16)
                let expectedRows = carried + peer(carried)
                eval(output, rowsSum, expected, expectedRows)
                #expect(payloads == [[rows + 3, 64]])
                #expect(output.shape == expected.shape && output.dtype == .bfloat16)
                #expect(arrayEqual(output, expected).item(Bool.self))
                #expect(rowsSum.shape == carried.shape && rowsSum.dtype == .float32)
                #expect(arrayEqual(rowsSum, expectedRows).item(Bool.self))
                // Outside the scope the projection sums its expert rows alone.
                let alone = projection(x, ids)
                eval(alone)
                #expect(payloads.last == [rows, 1, 64])
                #expect(arrayEqual(alone, expected).item(Bool.self))
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
