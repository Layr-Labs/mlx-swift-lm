// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Native synthetic tests. Source-only qualification never executes these.
@Suite("Gemma4 B8 expert execution", .serialized)
struct Gemma4B8ExpertTests {
    private func fixture() -> [MLXArray] {
        let shapes = [[128, 704, 352], [128, 704, 44], [128, 704, 44],
                      [128, 704, 352], [128, 704, 44], [128, 704, 44],
                      [128, 2816, 88], [128, 2816, 11], [128, 2816, 11]]
        return shapes.enumerated().map { index, shape in
            let count = shape.reduce(1, *)
            if index % 3 == 0 {
                return (MLXArray(0..<count).asType(.uint32) * UInt32(2654435761))
                    .reshaped(shape)
            }
            let divisor: Float = index % 3 == 1 ? 512 : 128
            return ((MLXArray(0..<count).asType(.float32) % 7 - 3) / divisor)
                .asType(.bfloat16).reshaped(shape)
        }
    }

    private func routing(shift: Int, prefix: Bool = false) throws -> Gemma4B8ExpertRouting {
        let scores = (0..<1024).map { i in -Float((i % 128 - (i / 128) * shift + 128) % 128) }
        return try #require(Gemma4B8ExpertRouting.make(
            scores: MLXArray(scores, [8, 1, 128]).asType(.bfloat16),
            perExpertScale: MLXArray.ones([128], dtype: .bfloat16),
            policy: .init(environment: ["DARKBLOOM_GEMMA4_B8_EXPERT_EXECUTION": "1",
                "DARKBLOOM_GEMMA4_B8_ROUTE_RANK": "1",
                "DARKBLOOM_GEMMA4_B8_ROUTE_PREFIX": prefix ? "1" : "0"])))
    }

    @Test func pairedStorageAndDescriptorInvalidation() throws {
        let original = fixture()
        let storage = try #require(Gemma4B8ExpertStorage(original))
        for i in 0..<3 {
            let tail = i == 0 ? 352 : 44
            let paired = storage.gateUp[i].reshaped(128, 44, 2, 16, tail)
            for side in 0..<2 {
                let restored = paired[0..., 0..., side, 0..., 0...].reshaped(128, 704, tail)
                #expect(restored.asData(access: .copy).data == original[i + side * 3].asData(access: .copy).data)
            }
        }
        #expect(storage.matches(original))
        for changed in original.indices {
            var replacement = original
            replacement[changed] = MLXArray.zeros(original[changed].shape, dtype: original[changed].dtype)
            #expect(!storage.matches(replacement))
        }
        original[0]._updateInternal(MLXArray.zeros(original[0].shape, dtype: .uint32))
        #expect(!storage.matches(original))
    }

    @Test func projectionClosesMustMatch() throws {
        let original = fixture()
        let storage = try #require(Gemma4B8ExpertStorage(original))
        let x = MLX.sin(MLXArray(0..<(8 * 2816)).asType(.float32) * 0.017)
            .asType(.bfloat16).reshaped(8, 2816)
        let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { (gate: MLXArray, up: MLXArray) in
            (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
        }
        let product = compile(shapeless: true, body)
        for shift in [0, 1, 16] {
          for prefix in [false, true] {
            let route = try routing(shift: shift, prefix: prefix)
            let gathered = x.reshaped(8, 1, 2816)[route.rowOrder]
            func projection(_ plane: MLXArray, offset: Int) -> MLXArray {
                gatherQuantizedMM(plane, original[offset], scales: original[offset + 1],
                    biases: original[offset + 2], rhsIndices: route.sortedKeys,
                    transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true)
            }
            let expectedGU = product(projection(gathered, offset: 0), projection(gathered, offset: 3))
            let actualGU = Gemma4B8ExpertExecution.gateUp(storage.gateUp + [x, route.rowOrder, route.executionKeys],
                                                        tagged: route.usesPrefixBounds)
            let label = "shift\(shift) prefix\(prefix)"
            gemma4ExpectExactBytes(actualGU, expectedGU, label: label + " GU")
            let identity = MLXArray(0..<64).asType(.uint32)
            let expectedDown = projection(expectedGU, offset: 6)
            let actualDown = Gemma4B8ExpertExecution.down(storage.down + [actualGU, identity, route.executionKeys],
                                                        tagged: route.usesPrefixBounds)
            gemma4ExpectExactBytes(actualDown, expectedDown, label: label + " down-composed")
            // Also isolate down arithmetic from any upstream GU mismatch.
            let isolatedDown = Gemma4B8ExpertExecution.down(storage.down + [expectedGU, identity, route.executionKeys],
                                                          tagged: route.usesPrefixBounds)
            gemma4ExpectExactBytes(isolatedDown, expectedDown, label: label + " down-isolated")
            let compiled = Gemma4B8ExpertExecution.compiledProject(storage: storage, x: x, routing: route, identity: identity)
            gemma4ExpectExactBytes(compiled, expectedDown, label: label + " compiled-composed")
          }
        }
    }
}
