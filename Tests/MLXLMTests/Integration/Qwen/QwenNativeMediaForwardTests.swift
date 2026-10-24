import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.tags(.integration), .serialized)
struct QwenNativeMediaForwardTests {
    private func model() throws -> Qwen35TextModel {
        let json =
            #"{"model_type":"qwen3_5_text","hidden_size":16,"num_hidden_layers":1,"intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":64,"full_attention_interval":1,"partial_rotary_factor":1.0,"rope_parameters":{"mrope_section":[2,1,1],"rope_theta":10000.0},"linear_num_key_heads":1,"linear_num_value_heads":1,"linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":2}"#
        return Qwen35TextModel(
            try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8)))
    }
    private func bytes(_ x: MLXArray) -> Data {
        eval(x)
        return x.asData(access: .copy).data
    }
    private func positions() -> MLXArray {
        MLXArray([Int32(0), 1, 2, 2, 3, 0, 1, 2, 3, 4, 0, 1, 3, 2, 4], [3, 1, 5])
    }
    @Test func explicitEmbeddingsAndPositionsRemainRequestOwned() throws {
        try Device.withDefaultDevice(.cpu) {
            try MLX.withError { errors in
                let model = try model()
                let tokens = MLXArray([Int32(1), 2, 3, 3, 4], [1, 5])
                let cache = model.newCache(parameters: nil)
                let ordinaryEmbeddings = model.scaledInputEmbeddings(tokens)
                let left = try model.forwardWithMedia(
                    tokens, inputEmbeddings: nil,
                    positionIds: positions(), cache: cache)
                let right = try model.forwardWithMedia(
                    tokens, inputEmbeddings: ordinaryEmbeddings,
                    positionIds: positions(), cache: model.newCache(parameters: nil))
                #expect(bytes(left) == bytes(right))
                // Identical placeholder IDs with distinct features are distinct requests.
                var imageEmbeddings = ordinaryEmbeddings
                imageEmbeddings[0..., 2 ..< 4, 0...] = MLXArray.ones([1, 2, 16]) * 0.125
                let image = try model.forwardWithMedia(
                    tokens, inputEmbeddings: imageEmbeddings,
                    positionIds: positions(), cache: model.newCache(parameters: nil))
                #expect(bytes(left) != bytes(image))
                let another = try model.forwardWithMedia(
                    tokens, inputEmbeddings: nil,
                    positionIds: MLXArray(
                        [Int32(0), 1, 8, 9, 10, 0, 1, 9, 8, 10, 0, 1, 8, 8, 10], [3, 1, 5]),
                    cache: model.newCache(parameters: nil))
                #expect(bytes(left) != bytes(another))
                let repeatFirst = try model.forwardWithMedia(
                    tokens, inputEmbeddings: nil,
                    positionIds: positions(), cache: model.newCache(parameters: nil))
                #expect(bytes(left) == bytes(repeatFirst))
                // The original ordinary text entry point still has its own scalar positions.
                let text1 = model(tokens, cache: model.newCache(parameters: nil))
                let text2 = model(tokens, cache: model.newCache(parameters: nil))
                #expect(bytes(text1) == bytes(text2))
                try errors.check()
            }
        }
    }
    @Test func malformedMediaInputsCannotAdvanceCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try model()
            let tokens = MLXArray([Int32(1), 2, 3, 3, 4], [1, 5])
            let cache = model.newCache(parameters: nil)
            for badPositions in [
                MLXArray.zeros([1, 1, 5], dtype: .int32), MLXArray.zeros([3, 1, 5]),
            ] {
                #expect(throws: (any Error).self) {
                    try model.forwardWithMedia(
                        tokens, inputEmbeddings: nil,
                        positionIds: badPositions, cache: cache)
                }
            }
            #expect(throws: (any Error).self) {
                try model.forwardWithMedia(
                    tokens, inputEmbeddings: MLXArray.zeros([1, 4, 16]),
                    positionIds: positions(), cache: cache)
            }
            #expect(throws: (any Error).self) {
                try model.forwardWithMedia(
                    tokens, inputEmbeddings: nil, positionIds: positions(), cache: [])
            }
            let wrongKind = MambaCache()
            #expect(throws: (any Error).self) {
                try model.forwardWithMedia(
                    tokens, inputEmbeddings: nil,
                    positionIds: positions(), cache: [wrongKind])
            }
            #expect(cache.allSatisfy { $0.offset == 0 })
            #expect(wrongKind.state.isEmpty)
        }
    }
}
