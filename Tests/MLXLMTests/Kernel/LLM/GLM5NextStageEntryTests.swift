// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// A decode step's entry and exit: the packed embedding's broadcast streams
    /// in one launch, the shared all-true validity and the skipped identity
    /// `where` must give the same bits as the separate operations.
    @Suite(.serialized)
    struct GLM5NextStageEntryTests {
        private static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        private static func fence(_ arrays: [MLXArray]) throws {
            try MLX.withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        /// Every packing the stream kernel reads, every activation dtype, and
        /// tokens at both ends of the vocabulary (and -1, which `take` wraps).
        @Test func packedEmbeddingStreamsAreBitwiseOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let vocabulary = 300
                for (bits, groupSize, dtype, hidden) in [
                    (4, 64, DType.bfloat16, 4096), (4, 64, .bfloat16, 512), (8, 64, .bfloat16, 512),
                    (4, 32, .float16, 256), (4, 128, .float32, 1024), (8, 128, .bfloat16, 256),
                ] {
                    let weight = MLXRandom.normal(
                        [vocabulary, hidden], key: MLXRandom.key(UInt64(hidden + bits)))
                    let (words, scales, biases) = MLX.quantized(
                        weight, groupSize: groupSize, bits: bits)
                    let embedding = GLM5NextPackedEmbedding(
                        weight: words, scales: scales.asType(dtype),
                        biases: biases!.asType(dtype),
                        packing: .init(bits: bits, groupSize: groupSize, mode: .affine))
                    for tokens in [
                        MLXArray([Int32(7)], [1, 1]), MLXArray([Int32(0), 299, 150], [1, 3]),
                        MLXArray([Int32(-1), 1, 2, 3, 4, 5, 6, 298], [2, 4]),
                        MLXArray([UInt32(299)], [1, 1]),
                    ] {
                        let streams = try #require(embedding.streams(tokens, copies: 4))
                        let embedded = embedding(tokens)
                        let expected = broadcast(
                            embedded.expandedDimensions(axis: 2),
                            to: [tokens.dim(0), tokens.dim(1), 4, hidden])
                        try Self.fence([streams, expected])
                        let label = "Q\(bits)/\(groupSize) \(dtype) D\(hidden) \(tokens.shape)"
                        #expect(streams.shape == expected.shape, "\(label) shape")
                        #expect(streams.dtype == expected.dtype, "\(label) dtype")
                        let differing = zip(Self.bits(streams), Self.bits(expected)).filter {
                            $0 != $1
                        }.count
                        #expect(differing == 0, "\(label): \(differing) values differ")
                        print("GLM53_EMBEDDING_STREAMS \(label) differing=\(differing)")
                    }
                }
            }
        }

        private func configuration() throws -> GLM5NextConfiguration {
            let base = try GLM5NextInventoryFixture.configuration(layers: 8, q4: true)
            var json = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
            let policy = GLM5NextNVFP4InventoryFixture.policy(base.textConfig)
            json["quantization"] = policy
            json["quantization_config"] = policy
            return try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
        }
    }
}

extension KernelTests {
    @Suite struct GLM5NextAllTrueValidityTests {
        @Test func sharedAllTrueValidityEqualsOnes() throws {
            let constants = GLM5NextKernelConstants()
            for shape in [[1, 1], [1, 8], [2, 4], [8, 8]] {
                let cached = try #require(constants.allTrue(shape))
                let ones = MLXArray.ones(shape, dtype: .bool)
                #expect(cached.shape == ones.shape && cached.dtype == .bool)
                #expect(cached.asArray(Bool.self) == ones.asArray(Bool.self))
                #expect(constants.allTrue(shape) === cached)
            }
            #expect(constants.allTrue([1, 65]) == nil)
            #expect(constants.allTrue([0, 1]) == nil)
        }
    }
}
