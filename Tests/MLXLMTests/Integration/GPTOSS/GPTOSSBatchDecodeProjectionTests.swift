import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.tags(.integration), .serialized)
struct GPTOSSBatchDecodeProjectionTests {
    struct ProjectionCase: Codable {
        let name: String
        let inputs: Int
        let outputs: Int
        let rows: Int
        let candidateDifferentBytes: Int
        let ordinaryWideDifferentBytes: Int
    }

    static func bytes(_ value: MLXArray) -> Data {
        eval(value)
        return value.asData(access: .copy).data
    }

    static func differences(_ lhs: MLXArray, _ rhs: MLXArray) -> Int {
        guard lhs.shape == rhs.shape, lhs.dtype == rhs.dtype else { return Int.max }
        return zip(bytes(lhs), bytes(rhs)).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
    }

    static func packed(
        inputs: Int, outputs: Int, bias: Bool, firstOutput: Int = 0, evaluate: Bool = true
    ) -> QuantizedLinear {
        // Direct stored affine8 arrays avoid a temporary full FP32 head. The
        // packed matrix has the registered dimensions, including 201088 logits.
        let count = outputs * inputs / 4
        let weight =
            ((arange(count) + firstOutput * inputs / 4).asType(.uint32) * UInt32(1_664_525)
            + UInt32(1_013_904_223)).reshaped(outputs, inputs / 4)
        let index = (arange(outputs * inputs / 64) + firstOutput * inputs / 64).asType(.float32)
        let scale = ((sin(index * 0.017) + 1.25) / 1024).reshaped(outputs, inputs / 64).asType(
            .bfloat16)
        let offsets = (-127 * scale).asType(.bfloat16)
        let additive =
            bias
            ? (sin((arange(outputs) + firstOutput).asType(.float32) * 0.13) / 16).asType(.bfloat16)
            : nil
        let result = QuantizedLinear(
            weight: weight, bias: additive, scales: scale,
            biases: offsets, groupSize: 64, bits: 8)
        if evaluate { eval(result) }
        return result
    }

    static func input(rows: Int, inputs: Int, tokens: Int = 1) -> MLXArray {
        let index = arange(rows * tokens * inputs).asType(.float32)
        return (sin(index * 0.031) * 0.7 + cos(index * 0.073) / 3)
            .reshaped(rows, tokens, inputs).asType(.bfloat16)
    }

    static func projectionCases(fullVocabulary: Bool = true) -> [ProjectionCase] {
        var cases: [ProjectionCase] = []
        // Registered 20B dimensions and additive-bias contracts. K and V have
        // the same shape but distinct input histories and affine parameters
        // are covered by eight different rows, not duplicated request inputs.
        for (name, inputs, outputs, bias) in [
            ("q", 2880, 4096, true), ("kv", 2880, 512, true),
            ("o", 4096, 2880, true), ("router", 2880, 32, true),
            ("head", 2880, 201088, false),
        ] {
            // CPU quantizedMM may materialize a broadcast weight for every
            // lane. Exercise actual packed head rows at the registered input
            // width without multiplying the whole 201088-row bank by eight.
            // The separate opt-in hardware probe keeps the full vocabulary.
            let ranges =
                name == "head" && !fullVocabulary
                ? [0 ..< 256, outputs / 2 ..< outputs / 2 + 256, outputs - 256 ..< outputs]
                : [0 ..< outputs]
            for range in ranges {
                let layer = packed(
                    inputs: inputs, outputs: range.count, bias: bias, firstOutput: range.lowerBound)
                for rows in [1, 2, 4, 8] {
                    let x = input(rows: rows, inputs: inputs)
                    let expected = concatenated(
                        (0 ..< rows).map { layer(x[$0 ..< $0 + 1]) }, axis: 0)
                    let candidate = GPTOSSBatchDecodeProjection.linear(layer, x)
                    let wide = layer(x)
                    cases.append(
                        ProjectionCase(
                            name: range.count == outputs
                                ? name : "\(name)[\(range.lowerBound)..<\(range.upperBound)]",
                            inputs: inputs, outputs: range.count,
                            rows: rows, candidateDifferentBytes: differences(candidate, expected),
                            ordinaryWideDifferentBytes: differences(wide, expected)))
                }
            }
        }
        return cases
    }

    @Test func registeredAffine8GeometryMatchesSingletonBytes() {
        Device.withDefaultDevice(.cpu) {
            for item in Self.projectionCases(fullVocabulary: false) {
                #expect(item.candidateDifferentBytes == 0, "\(item.name) rows=\(item.rows)")
            }
        }
    }

    @Test func registeredFullVocabularyGraphKeepsShapeWithoutMaterializingBroadcastWeights() throws
    {
        try Device.withDefaultDevice(.cpu) { () throws -> Void in
            let layer = Self.packed(inputs: 2880, outputs: 201088, bias: false, evaluate: false)
            #expect(layer.weight.shape == [201088, 720])
            for rows in [1, 2, 4, 8] {
                let x = Self.input(rows: rows, inputs: 2880)
                let candidate = GPTOSSBatchDecodeProjection.linear(layer, x)
                #expect(candidate.shape == [rows, 1, 201088])
                #expect(candidate.dtype == .bfloat16)
                // Shape propagation must not allocate the full CPU broadcast.
                #expect(try candidate.evaluatedBufferInfo() == nil)
                #expect(try layer.weight.evaluatedBufferInfo() == nil)
            }
        }
    }

    @Test func customModulesRemainOriginalSingletonsAndPrefillStaysOriginal() {
        Device.withDefaultDevice(.cpu) {
            let custom = GPTOSSObservedLinear()
            for (rows, tokens) in [(1, 1), (2, 1), (4, 1), (8, 1), (4, 3), (9, 1)] {
                let x = Self.input(rows: rows, inputs: 64, tokens: tokens)
                custom.shapes = []
                let actual = GPTOSSBatchDecodeProjection.linear(custom, x)
                let expectedShapes =
                    (2 ... 8).contains(rows) && tokens == 1
                    ? Array(repeating: [1, 1, 64], count: rows) : [x.shape]
                #expect(custom.shapes == expectedShapes)
                #expect(Self.differences(actual, x + 7) == 0)
            }
        }
    }

    @Test func unsupportedPackingRetainsOriginalRowForward() {
        Device.withDefaultDevice(.cpu) {
            for bits in [4, 8] {
                let layer = QuantizedLinear(
                    weight: Self.input(rows: 4, inputs: 64).reshaped(4, 64),
                    bias: nil, groupSize: 32, bits: bits)
                let x = Self.input(rows: 8, inputs: 64)
                let expected = concatenated((0 ..< 8).map { layer(x[$0 ..< $0 + 1]) }, axis: 0)
                #expect(
                    Self.differences(GPTOSSBatchDecodeProjection.linear(layer, x), expected) == 0)
            }
        }
    }
}

private final class GPTOSSObservedLinear: Linear {
    var shapes: [[Int]] = []
    init() { super.init(weight: MLXArray.zeros([64, 64]), bias: nil) }
    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        shapes.append(input.shape)
        return input + 7
    }
}
