// Copyright © 2026 Eigen Labs.

import MLX
import MLXNN
import Testing

@testable import MLXLLM

/// Native MLX tests, NOT part of the no-device host harness. They have not
/// been executed in the source-only catch-up phase. Synthetic tensors avoid
/// a checkpoint dependency but still require an authorized native/GPU lane.
@Suite("Gemma4 dense gate/up derived storage", .serialized)
struct Gemma4DenseGateUpStorageTests {
    private func fixture() -> [MLXArray] {
        let packed = 2112 * 704
        let groups = 2112 * 44
        let gate = (0..<packed).map { UInt32(truncatingIfNeeded: $0 &* 2654435761) }
        let up = (0..<packed).map { UInt32(truncatingIfNeeded: ($0 &* 2246822519) ^ 0x13579BDF) }
        let scales = (0..<groups).map { Float($0 % 7 + 1) / 512 }
        let offsets = (0..<groups).map { Float($0 % 13 - 6) / 128 }
        return [
            MLXArray(gate, [2112, 704]),
            MLXArray(scales, [2112, 44]).asType(.bfloat16),
            MLXArray(offsets, [2112, 44]).asType(.bfloat16),
            MLXArray(up, [2112, 704]),
            MLXArray(scales.reversed(), [2112, 44]).asType(.bfloat16),
            MLXArray(offsets.reversed(), [2112, 44]).asType(.bfloat16),
        ]
    }

    private func storage(_ p: [MLXArray]) -> Gemma4DenseGateUpStorage? {
        Gemma4DenseGateUpStorage(p[0], p[1], p[2], p[3], p[4], p[5])
    }

    private func projections(_ p: [MLXArray]) -> (QuantizedLinear, QuantizedLinear) {
        (QuantizedLinear(weight: p[0], scales: p[1], biases: p[2], groupSize: 64, bits: 8),
         QuantizedLinear(weight: p[3], scales: p[4], biases: p[5], groupSize: 64, bits: 8))
    }

    @Test func splitPreservesEveryValueAndDType() throws {
        let original = fixture()
        let pair = try #require(storage(original))
        for (source, split) in zip(original, pair.splitParameters) {
            #expect(source.shape == split.shape)
            #expect(source.dtype == split.dtype)
            #expect(source.asData(access: .copy).data == split.asData(access: .copy).data)
            if source.dtype == .uint32 {
                #expect(source.asArray(UInt32.self) == split.asArray(UInt32.self))
            } else {
                #expect(source.asArray(Float.self) == split.asArray(Float.self))
            }
        }
        let (gate, up) = projections(pair.splitParameters)
        #expect(pair.matches(gate: gate, up: up))
    }

    @Test func pairedProjectionMustMatchOrdinaryProjectionExactly() throws {
        let original = fixture()
        let pair = try #require(storage(original))
        let (gate, up) = projections(original)
        let x = MLXArray((0..<2816).map { Float($0 % 17 - 8) / 32 }, [1, 1, 2816])
            .asType(.bfloat16)
        let actual = try #require(pair.project(x))
        let expected = concatenated([gate(x), up(x)], axis: -1)
        #expect(actual.shape == [1, 1, 4224])
        #expect(actual.dtype == expected.dtype)
        #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
        #expect(actual.asArray(Float.self) == expected.asArray(Float.self))
        // No tolerance/near-tie exemption: failure holds this candidate OFF.
    }

    @Test func everyParameterUpdateInvalidatesSameWrapper() throws {
        for changed in 0..<6 {
            let original = fixture()
            let pair = try #require(storage(original))
            let (gate, up) = projections(pair.splitParameters)
            #expect(pair.matches(gate: gate, up: up))
            let projection = changed < 3 ? gate : up
            let key = ["weight", "scales", "biases"][changed % 3]
            let wrapper = pair.splitParameters[changed]
            let replacement = MLXArray.zeros(wrapper.shape, dtype: wrapper.dtype)
            projection.update(parameters: ModuleParameters.unflattened([key: replacement]))
            let live = [gate.weight, gate.scales, try #require(gate.biases),
                        up.weight, up.scales, try #require(up.biases)]
            #expect(live[changed] === wrapper)  // Swift identity alone is unsafe.
            #expect(!pair.matches(gate: gate, up: up))
        }
    }

    @Test func indexedMutationAndForeignPairInvalidate() throws {
        let pair = try #require(storage(fixture()))
        let (gate, up) = projections(pair.splitParameters)
        let foreign = try #require(storage(fixture()))
        let (otherGate, otherUp) = projections(foreign.splitParameters)
        #expect(!pair.matches(gate: otherGate, up: otherUp))
        #expect(pair.matches(gate: gate, up: up))
        gate.weight[0] = MLXArray.zeros([704], dtype: .uint32)
        #expect(!pair.matches(gate: gate, up: up))
    }

    @Test func malformedShapeOrDTypeNeverConstructsStorage() {
        let original = fixture()
        for changed in 0..<6 {
            var wrongDType = original
            wrongDType[changed] = original[changed].asType(.float32)
            #expect(storage(wrongDType) == nil)
            var wrongShape = original
            wrongShape[changed] = original[changed][0..<2111]
            #expect(storage(wrongShape) == nil)
        }
    }
}
