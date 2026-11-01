import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.tags(.integration), .serialized)
struct Gemma4PrefillProjectionTests {
    private func bytes(_ x: MLXArray) -> Data {
        eval(x)
        return x.asData(access: .copy).data
    }
    private func input(_ count: Int, dtype: DType = .bfloat16) -> MLXArray {
        MLXArray((0 ..< count * 128).map { sin(Float($0) * 0.17) }, [1, count, 128]).asType(dtype)
    }
    private func layer(_ bits: Int, dtype: DType = .bfloat16) -> QuantizedLinear {
        let weight = MLXArray((0 ..< 96 * 128).map { cos(Float($0) * 0.07) }, [96, 128]).asType(
            dtype)
        return QuantizedLinear(
            weight: weight, bias: MLXArray.ones([96], dtype: dtype), groupSize: 64, bits: bits)
    }

    @Test func promptTokensPreserveOriginalSingletonReductions() {
        Device.withDefaultDevice(.cpu) {
            for bits in [4, 8] {
                let linear = layer(bits)
                for count in [2, 8, 127, 128, 511, 512] {
                    let x = input(count)
                    let expected = concatenated(
                        (0 ..< count).map { linear(x[0..., $0 ..< $0 + 1, 0...]) }, axis: 1)
                    let actual = Gemma4PrefillProjectionScope.withPrefillFrame(
                        tokens: MLXArray.zeros([1, count], dtype: .int32), enabled: true
                    ) {
                        Gemma4BatchDecodeProjection.linear(linear, x)
                    }
                    #expect(actual.shape == [1, count, 96] && actual.dtype == .bfloat16)
                    #expect(bytes(actual) == bytes(expected), "bits=\(bits), count=\(count)")
                }
            }
        }
    }

    @Test func explicitScopeRestoresAndDoesNotActivateDecodeOrSingleton() {
        Device.withDefaultDevice(.cpu) {
            for shape in [[1, 1], [2, 1], [8, 1], [2, 8], [1, 8, 1]] {
                Gemma4PrefillProjectionScope.withPrefillFrame(
                    tokens: MLXArray.zeros(shape, dtype: .int32), enabled: true
                ) {
                    #expect(!Gemma4PrefillProjection.active)
                }
            }
            let tokens = MLXArray.zeros([1, 128], dtype: .int32)
            Gemma4PrefillProjectionScope.withPrefillFrame(tokens: tokens, enabled: true) {
                #expect(Gemma4PrefillProjection.active && !Gemma4BatchDecodeProjection.active)
                Gemma4PrefillProjectionScope.withPrefillFrame(tokens: tokens, enabled: false) {
                    #expect(!Gemma4PrefillProjection.active)
                }
                #expect(Gemma4PrefillProjection.active)
            }
            #expect(!Gemma4PrefillProjection.active)
        }
    }

    @Test func defaultCustomAndUnsupportedDtypesKeepOriginalRectangle() {
        Device.withDefaultDevice(.cpu) {
            let base = layer(4)
            let custom = PrefillObservedLinear(
                weight: base.weight, scales: base.scales,
                biases: base.biases, groupSize: 64, bits: 4)
            let x = input(8)
            let expected = custom(x)
            custom.shapes = []
            let actual = Gemma4PrefillProjectionScope.withPrefillFrame(
                tokens: MLXArray.zeros([1, 8], dtype: .int32), enabled: true
            ) {
                Gemma4BatchDecodeProjection.linear(custom, x)
            }
            #expect(custom.shapes == [[1, 8, 128]])
            #expect(bytes(actual) == bytes(expected))
            #expect(bytes(Gemma4BatchDecodeProjection.linear(base, x)) == bytes(base(x)))
            let floating = layer(4, dtype: .float32)
            let y = input(8, dtype: .float32)
            let preserved = Gemma4PrefillProjectionScope.withPrefillFrame(
                tokens: MLXArray.zeros([1, 8], dtype: .int32), enabled: true
            ) {
                Gemma4BatchDecodeProjection.linear(floating, y)
            }
            #expect(bytes(preserved) == bytes(floating(y)))
        }
    }

    @Test func promptScopeDoesNotChangeTiedHeadOrSingleton() {
        Device.withDefaultDevice(.cpu) {
            let weight = MLXArray((0 ..< 96 * 128).map { sin(Float($0) * 0.13) }, [96, 128]).asType(
                .bfloat16)
            for bits in [4, 8] {
                let embedding = QuantizedEmbedding(weight: weight, groupSize: 64, bits: bits)
                let x = input(8)
                let expected = embedding.asLinear(x)
                let actual = Gemma4PrefillProjectionScope.withPrefillFrame(
                    tokens: MLXArray.zeros([1, 8], dtype: .int32), enabled: true
                ) {
                    Gemma4BatchDecodeScope.projectHead(embedding, x)
                }
                #expect(bytes(actual) == bytes(expected))
                let one = x[0..., 0 ..< 1, 0...]
                let singleton = Gemma4PrefillProjection.$active.withValue(true) {
                    Gemma4BatchDecodeScope.projectHead(embedding, one)
                }
                #expect(bytes(singleton) == bytes(embedding.asLinear(one)))
            }
        }
    }
}

private final class PrefillObservedLinear: QuantizedLinear {
    var shapes: [[Int]] = []
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        shapes.append(x.shape)
        return super.callAsFunction(x * 2) + 3
    }
}
