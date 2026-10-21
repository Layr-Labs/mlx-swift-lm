import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    struct ModelOptNVFP4SwitchLinearTests {
        private func experts() throws -> [ModelOptNVFP4Weights] {
            try (0 ..< 2).map { expert in
                try ModelOptNVFP4Weights(
                    packedBytes: MLXArray([UInt8](repeating: expert == 0 ? 0x21 : 0x43, count: 16))
                        .reshaped(2, 8),
                    e4m3ScaleBytes: MLXArray([UInt8(56), 64]).reshaped(2, 1),
                    tensorScale: expert == 0 ? 0.01 : 0.02)
            }
        }

        @Test func broadcastDecodePreservesRepeatedUnsortedAssignments() throws {
            let experts = try experts()
            let scales: [Float] = [0.01, 0.02]
            let layer = try ModelOptNVFP4SwitchLinear(experts: experts, inputScales: scales)
            let input = MLXArray((0 ..< 16).map { Float($0 - 8) / 32 }).reshaped(1, 1, 16)
            let ids: [Int32] = [1, 0, 1]
            let actual = try layer.checkedForward(input, MLXArray(ids), sortedIndices: true)
            let expected = try concatenated(
                ids.map {
                    try experts[Int($0)].calibratedMatmul(
                        input.reshaped(1, 16), inputScale: scales[Int($0)])
                }, axis: 0
            ).reshaped(3, 1, 2)
            #expect(actual.shape == [3, 1, 2])
            #expect(abs(actual - expected).max().item(Float.self) < 0.0001)
        }

        @Test func over64AlignedAssignmentsRestoreOrderAndMatchSingleRowOracle() throws {
            let experts = try experts()
            let scales: [Float] = [0.01, 0.02]
            let layer = try ModelOptNVFP4SwitchLinear(experts: experts, inputScales: scales)
            let ids = (0 ..< 70).map { Int32(($0 * 7 + 1) % 2) }
            let input = MLXArray((0 ..< (70 * 16)).map { Float($0 % 23 - 11) / 64 }).reshaped(
                70, 1, 16)
            let actual = try layer.checkedForward(input, MLXArray(ids), sortedIndices: false)
            let references = try ids.enumerated().map { row, expert in
                try experts[Int(expert)].calibratedMatmul(
                    input[row].reshaped(1, 16), inputScale: scales[Int(expert)])
            }
            let expected = concatenated(references, axis: 0).reshaped(70, 1, 2)
            #expect(abs(actual - expected).max().item(Float.self) < 0.0001)
            let withHint = try layer.checkedForward(input, MLXArray(ids), sortedIndices: true)
            #expect(arrayEqual(actual, withHint).item(Bool.self))
        }

        @Test func matrixRowsAndBroadcastIndicesFollowGatherShape() throws {
            let experts = try experts()
            let scales: [Float] = [0.01, 0.02]
            let layer = try ModelOptNVFP4SwitchLinear(experts: experts, inputScales: scales)
            let input = MLXArray((0 ..< 32).map { Float($0 % 9) / 32 }).reshaped(1, 2, 16)
            let actual = try layer.checkedForward(input, MLXArray([Int32(0), 1]))
            let expected = try stacked(
                experts.enumerated().map {
                    try $0.element.calibratedMatmul(
                        input.reshaped(2, 16), inputScale: scales[$0.offset])
                })
            #expect(actual.shape == [2, 2, 2])
            #expect(abs(actual - expected).max().item(Float.self) < 0.0001)
        }

        @Test func rejectsMetadataIndicesAndWorkspace() throws {
            let experts = try experts()
            #expect(throws: (any Error).self) {
                try ModelOptNVFP4SwitchLinear(experts: experts, inputScales: [0.01, 0])
            }
            let layer = try ModelOptNVFP4SwitchLinear(experts: experts, inputScales: [0.01, 0.02])
            #expect(throws: (any Error).self) {
                try layer.checkedForward(MLXArray.zeros([1, 1, 16]), MLXArray([Int32(2)]))
            }
            #expect(throws: (any Error).self) {
                try layer.checkedForward(
                    MLXArray.zeros([1, 1, 16]), MLXArray.zeros([8193], dtype: .int32))
            }
            #expect(throws: (any Error).self) {
                try layer.checkedForward(
                    MLXArray.zeros([2, 1, 16]), MLXArray.zeros([3], dtype: .int32))
            }
        }
    }
}
