import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

extension KernelTests {
    /// The multi-row FP8 graph path against the eager reference at Super's
    /// shapes, near the reference's 64 MiB bound and away from it.
    @Suite(.serialized)
    struct ModelOptFP8ProductPlanParityTests {
        private func layer(inputs: Int, outputs: Int) throws -> ModelOptCalibratedLinear {
            let key = MLXRandom.key(UInt64(inputs ^ outputs))
            let raw = MLXRandom.randInt(Int32(0) ..< 256, [outputs, inputs], key: key)
                .asType(.uint8)
            let bytes = which((raw & UInt8(0x7F)) .== UInt8(0x7F), raw - UInt8(1), raw)
            return try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(e4m3Bytes: bytes, tensorScale: 0.00123),
                inputScale: 0.0071, weightDType: .bfloat16, activationOverflow: .saturating)
        }

        /// Bit-identical to `checkedForward` wherever the plan keeps MLX's
        /// product class; the near-bound rows regroup thousands of eager tiles.
        @Test(arguments: [
            (4096, 18560, [16, 48, 65, 77, 80, 82, 84, 86, 88, 89, 90, 91, 92]),
            (8192, 4096, [91, 128, 149, 150, 160, 165, 169, 170]),
            (4096, 4096, [143, 224, 240, 241, 250, 255]),
            (4096, 256, [452, 470, 480]),
        ])
        func graphMatchesTheEagerReferenceNearTheBound(inputs: Int, outputs: Int, rows: [Int])
            throws
        {
            let layer = try layer(inputs: inputs, outputs: outputs)
            let routing = ModelOptMLXProductRouting.current
            for r in rows {
                let x = (MLXRandom.normal([1, r, inputs], key: MLXRandom.key(UInt64(r))) * 0.8)
                    .asType(.bfloat16)
                let eager = layer.fp8EagerTiles(rows: r)
                let pieces = layer.fp8ProductPlan(rows: r, eager: eager)
                let sliced =
                    pieces.map(\.columns)
                    == ModelOptCalibratedLinear.fp8Slices(inputs: inputs, outputs: outputs)
                if routing != nil {
                    #expect(
                        pieces.dispatches
                            <= min(eager.count, ModelOptFP8ProductPiece.maximumDispatches))
                }
                let graph = try layer.graphForward(x)
                let reference = try layer.checkedForward(x)
                #expect(graph.shape == [1, r, outputs])
                if sliced {
                    let difference = abs(graph.asType(.float32) - reference.asType(.float32))
                    let scale = abs(reference.asType(.float32)).max().item(Float.self)
                    #expect(difference.max().item(Float.self) <= scale * 0.01, "rows \(r)")
                } else {
                    #expect(
                        modelOptBitwiseEqual(graph, reference), "rows \(r) \(inputs)->\(outputs)")
                }
            }
        }

        /// Where the eager reference refuses (no output row fits its bound),
        /// the graph path takes 32 Mi-element slices instead of one-row tiles.
        @Test(arguments: [(4096, 4096, 256), (4096, 256, 481)])
        func graphTakesSlicesWhereTheEagerReferenceRefuses(inputs: Int, outputs: Int, rows: Int)
            throws
        {
            let layer = try layer(inputs: inputs, outputs: outputs)
            let x = (MLXRandom.normal([1, rows, inputs], key: MLXRandom.key(9)) * 0.8)
                .asType(.bfloat16)
            #expect(throws: (any Error).self) { try layer.checkedForward(x) }
            #expect(
                layer.fp8EagerTiles(rows: rows)
                    == ModelOptCalibratedLinear.fp8Slices(inputs: inputs, outputs: outputs))
            let graph = try layer.graphForward(x)
            #expect(graph.shape == [1, rows, outputs])
            #expect(!isNaN(graph).any().item(Bool.self))
        }
    }
}
