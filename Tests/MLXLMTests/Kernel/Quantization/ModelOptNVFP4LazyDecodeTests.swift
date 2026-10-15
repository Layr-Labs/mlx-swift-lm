import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    struct ModelOptNVFP4LazyDecodeTests {
        private func layer() throws -> ModelOptNVFP4SwitchLinear {
            let experts = try (0 ..< 2).map { expert in
                try ModelOptNVFP4Weights(
                    packedBytes: MLXArray([UInt8](repeating: expert == 0 ? 0x21 : 0x43, count: 16))
                        .reshaped(2, 8),
                    e4m3ScaleBytes: MLXArray([UInt8(56), 64]).reshaped(2, 1),
                    tensorScale: expert == 0 ? 0.01 : 0.03)
            }
            return try ModelOptNVFP4SwitchLinear(experts: experts, inputScales: [0.01, 0.02])
        }

        @Test func unequalCalibrationRepeatedRoutingAndRequestIsolation() throws {
            let layer = try layer()
            let ids = MLXArray([Int32(1), 0, 1])
            let input = MLXArray((0 ..< 48).map { Float($0 % 17 - 8) / 32 }).reshaped(3, 1, 16)
            let quote = try layer.lazyDecodeQuote(assignments: 3)
            let first = try layer.lazyDecode(
                input, ids, admittedArrayBytes: quote.arrayBytes,
                admittedGraphUnits: quote.graphUnits)
            let second = try layer.lazyDecode(
                input * 2, ids, admittedArrayBytes: quote.arrayBytes,
                admittedGraphUnits: quote.graphUnits)
            let reference = try layer.checkedForward(input, ids)
            let reference2 = try layer.checkedForward(input * 2, ids)
            #expect(abs(first - reference).max().item(Float.self) < 0.0001)
            #expect(abs(second - reference2).max().item(Float.self) < 0.0001)
            let broadcast = try layer.lazyDecode(
                input[0 ..< 1], ids, admittedArrayBytes: quote.arrayBytes,
                admittedGraphUnits: quote.graphUnits)
            let broadcastReference = try layer.checkedForward(input[0 ..< 1], ids)
            #expect(abs(broadcast - broadcastReference).max().item(Float.self) < 0.0001)
        }

        @Test func invalidIDsAreNaNAndValidRowsUnaffected() throws {
            let layer = try layer()
            let input = MLXArray.ones([1, 1, 16])
            let quote = try layer.lazyDecodeQuote(assignments: 3)
            for ids in [MLXArray([Int32(-1), 1, 2]), MLXArray([UInt32.max, 1, 2])] {
                let result = try layer.lazyDecode(
                    input, ids, admittedArrayBytes: quote.arrayBytes,
                    admittedGraphUnits: quote.graphUnits)
                #expect(isNaN(result[0]).all().item(Bool.self))
                #expect(isNaN(result[2]).all().item(Bool.self))
                let expected = try layer.checkedForward(input, MLXArray([Int32(1)]))
                #expect(abs(result[1] - expected[0]).max().item(Float.self) < 0.0001)
            }
        }

        @Test func admissionRejectsBeforeGraphConstruction() throws {
            let layer = try layer()
            let quote = try layer.lazyDecodeQuote(assignments: 1)
            #expect(throws: (any Error).self) { try layer.lazyDecodeQuote(assignments: 65) }
            #expect(throws: (any Error).self) {
                try layer.lazyDecode(
                    MLXArray.ones([1, 1, 16]), MLXArray([Int32(0)]),
                    admittedArrayBytes: quote.arrayBytes - 1, admittedGraphUnits: quote.graphUnits)
            }
            #expect(throws: (any Error).self) {
                try layer.lazyDecode(
                    MLXArray.ones([1, 1, 16]), MLXArray([Int32(0)]),
                    admittedArrayBytes: quote.arrayBytes, admittedGraphUnits: quote.graphUnits - 1)
            }
        }
    }
}
