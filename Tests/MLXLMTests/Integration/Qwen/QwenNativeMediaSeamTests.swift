// Copyright © 2026 Eigen Labs.
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite(.tags(.integration), .serialized)
struct QwenNativeMediaSeamTests {
    private func bytes(_ changes: [String: Any] = [:]) throws -> Data {
        var value: [String: Any] = [
            "model_type": "qwen3_5", "image_token_id": 60, "video_token_id": 61,
            "vision_start_token_id": 57, "vision_end_token_id": 58,
            "text_config": [
                "model_type": "qwen3_5_text", "hidden_size": 8,
                "num_hidden_layers": 1, "intermediate_size": 16, "num_attention_heads": 1,
                "num_key_value_heads": 1, "head_dim": 8, "linear_num_key_heads": 1,
                "linear_num_value_heads": 1, "linear_key_head_dim": 8,
                "linear_value_head_dim": 8, "linear_conv_kernel_dim": 2,
                "vocab_size": 64, "full_attention_interval": 1,
            ],
            "vision_config": [
                "model_type": "qwen3_5_vision", "depth": 1,
                "hidden_size": 8, "intermediate_size": 16, "out_hidden_size": 8,
                "num_heads": 2, "patch_size": 1, "spatial_merge_size": 2,
                "temporal_patch_size": 1, "num_position_embeddings": 16,
                "in_channels": 3, "hidden_act": "gelu_pytorch_tanh",
                "deepstack_visual_indexes": [],
            ],
        ]
        value.merge(changes) { _, new in new }
        return try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    }
    private func config() throws -> Qwen35Configuration {
        try JSONDecoder().decode(Qwen35Configuration.self, from: bytes())
    }
    private var processorBytes: Data {
        Data(
            #"{"image_processor_type":"Qwen3VLImageProcessor","patch_size":4,"merge_size":2,"temporal_patch_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"min_pixels":64,"max_pixels":4096}"#
                .utf8)
    }
    private func equal(_ a: MLXArray, _ b: MLXArray) -> Bool {
        eval(a, b)
        return a.shape == b.shape && a.dtype == b.dtype
            && a.asArray(Float.self).map(\.bitPattern) == b.asArray(Float.self).map(\.bitPattern)
    }

    @Test func mergedProcessorIdentityPreservesStandalonePrecedence() throws {
        let pre = Data(#"{"z":null,"nested":{"winner":1},"a":2}"#.utf8)
        let proc = Data(#"{"z":3,"nested":{"loser":4},"b":1}"#.utf8)
        let resolved = try Qwen3VLMediaProcessor.resolvedConfiguration(
            preprocessor: pre, processor: proc)
        #expect(resolved == Data(#"{"a":2,"b":1,"nested":{"winner":1},"z":null}"#.utf8))
        #expect(
            try Qwen3VLMediaProcessor.resolvedConfiguration(
                preprocessor: pre,
                processor: Data(#"{"a":0}"#.utf8)) == pre)
        #expect(
            try Qwen3VLMediaProcessor.resolvedConfiguration(preprocessor: nil, processor: proc)
                == proc)
        #expect(throws: (any Error).self) {
            try Qwen3VLMediaProcessor.resolvedConfiguration(preprocessor: nil, processor: nil)
        }
    }

    @Test func processorGeometryAndInvalidMetadata() throws {
        let processor = try Qwen3VLMediaProcessor(
            modelConfiguration: bytes(), processorConfiguration: processorBytes)
        #expect(processor.configuration.patchSize == 4)
        #expect(processor.configuration.temporalPatchSize == 2)
        #expect(processor.configuration.mergeSize == 2)
        #expect(Qwen3VLMediaProcessor.policyID == "qwen35-media-preparation-v1")
        #expect(throws: (any Error).self) {
            try Qwen3VLMediaProcessor(
                modelConfiguration: bytes(["model_type": "gemma4"]),
                processorConfiguration: processorBytes)
        }
        let invalid = Data(
            String(decoding: processorBytes, as: UTF8.self)
                .replacingOccurrences(of: "[0.5,0.5,0.5]", with: "[0,0,0]").utf8)
        #expect(throws: (any Error).self) {
            try Qwen3VLMediaProcessor(modelConfiguration: bytes(), processorConfiguration: invalid)
        }
    }

    @Test func admissionGridUsesTheActualResizeWithoutRasterAllocation() throws {
        let processor = try Qwen3VLMediaProcessor(
            modelConfiguration: bytes(), processorConfiguration: processorBytes)
        for (height, width) in [(8, 8), (64, 64), (191, 317), (2160, 3840), (10000, 1000)] {
            let expected = try QwenVL.targetSize(
                height: height, width: width, factor: 8,
                minPixels: 64, maxPixels: 4096)
            let image = try processor.imageGrid(height: height, width: width)
            #expect(image.t == 1 && image.h == expected.0 / 4 && image.w == expected.1 / 4)
            let video = try processor.videoFrameGrid(height: height, width: width)
            #expect(video.t == image.t && video.h == image.h && video.w == image.w)
        }
        for (height, width) in [(1, 1), (0, 8), (-1, 8), (Int.max, 8), (Int.max / 2, 3)] {
            #expect(throws: (any Error).self) {
                try processor.imageGrid(height: height, width: width)
            }
            #expect(throws: (any Error).self) {
                try processor.videoFrameGrid(height: height, width: width)
            }
        }
        let maximum = try processor.maximumVideoPatchCount()
        for short in [8, 9, 15, 32, 127, 511, 8192] {
            for ratio in [1, 2, 19, 199, 200] {
                for (height, width) in [(short, short * ratio), (short * ratio, short)] {
                    if let grid = try? processor.videoFrameGrid(height: height, width: width) {
                        #expect(grid.h * grid.w * 4 <= maximum)
                    }
                }
            }
        }
    }

    @Test func metadataPositionsPreserveRequestOwnedDelta() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try config()
            let tokens = MLXArray([Int32(4), 57, 60, 58, 5, 57, 61, 61, 58, 6])
            let positions = try QwenVisionSeamSupport.positionResult(
                tokens: tokens,
                imageGrids: [THW(1, 2, 2)], videoGrids: [THW(2, 2, 2)],
                attentionMask: nil, seam: c.visionSeamConfiguration)
            let ordinary = Qwen35(c)
            let expected = try ordinary.positionResult(
                tokens: tokens,
                imageGrids: [THW(1, 2, 2)], videoGrids: [THW(2, 2, 2)])
            #expect(
                positions.promptPositionIds.asArray(Int32.self)
                    == expected.promptPositionIds.asArray(Int32.self))
            #expect(positions.decodeState == expected.decodeState)
            let first = positions.decodeState.decodePositionIds(cacheOffset: 10, length: 2)
            _ = try QwenVisionSeamSupport.positionResult(
                tokens: MLXArray([Int32(7), 8]),
                imageGrids: nil, videoGrids: nil, attentionMask: nil,
                seam: c.visionSeamConfiguration)
            #expect(
                first.asArray(Int32.self)
                    == positions.decodeState.decodePositionIds(cacheOffset: 10, length: 2).asArray(
                        Int32.self))
            #expect(positions.promptPositionIds.shape == [3, 1, 10])
        }
    }

    @Test func visionOnlyFeaturesMatchOrdinaryTowerForImageAndWholeVideo() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try config()
            let ordinary = Qwen35(c)
            let tower = try Qwen35VisionTower(configuration: c, textDType: .float32)
            let weights = ordinary.parameters().flattened().filter {
                $0.0.hasPrefix("vision_tower.")
            }
            try tower.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            #expect(!tower.parameters().flattened().contains { $0.0.contains("language_model") })
            for temporal in [1, 2] {
                let grid = THW(temporal, 2, 2)
                let patches = MLXArray((0 ..< (temporal * 12)).map { Float($0 % 7) * 0.125 })
                    .reshaped(temporal * 4, 3)
                let actual = try tower.visionFeatures(
                    imagePixels: temporal == 1 ? patches : nil,
                    imageGrids: temporal == 1 ? [grid] : nil,
                    videoPixels: temporal == 2 ? patches : nil,
                    videoGrids: temporal == 2 ? [grid] : nil)
                let expected = try ordinary.visionFeatures(
                    imagePixels: temporal == 1 ? patches : nil,
                    imageGrids: temporal == 1 ? [grid] : nil,
                    videoPixels: temporal == 2 ? patches : nil,
                    videoGrids: temporal == 2 ? [grid] : nil)
                #expect(equal(actual.flattenedFeatures, expected.flattenedFeatures))
                #expect(actual.ordered.map(\.kind) == expected.ordered.map(\.kind))
            }
        }
    }

    @Test func towerRefusesCombinedSubjectsAndMalformedPatches() throws {
        try Device.withDefaultDevice(.cpu) {
            let tower = try Qwen35VisionTower(configuration: config(), textDType: .float32)
            #expect(throws: (any Error).self) {
                try tower.visionFeatures(
                    imagePixels: MLXArray.zeros([8, 3]),
                    imageGrids: [THW(1, 2, 2), THW(1, 2, 2)], videoPixels: nil, videoGrids: nil)
            }
            #expect(throws: (any Error).self) {
                try tower.visionFeatures(
                    imagePixels: MLXArray.zeros([4, 4]),
                    imageGrids: [THW(1, 2, 2)], videoPixels: nil, videoGrids: nil)
            }
            #expect(throws: (any Error).self) {
                try tower.sanitizeVisionWeights([
                    "language_model.model.norm.weight": MLXArray.ones([8])
                ])
            }
            #expect(
                Qwen35VisionTower.canonicalWeightName("model.visual.blocks.0.norm1.weight")
                    == "vision_tower.blocks.0.norm1.weight")
            #expect(
                Qwen35VisionTower.canonicalWeightName("model.language_model.norm.weight") == nil)
        }
    }
    @Test func serialInventoryMatchesRealModuleAndKeepsExactStorage() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try config()
            let tower = try Qwen35VisionTower(configuration: c, textDType: .float16)
            let expected = try Qwen35VisionTower.expectedTensorShapes(configuration: c)
            #expect(
                expected
                    == Dictionary(
                        uniqueKeysWithValues: tower.parameters().flattened().map {
                            ($0.0, $0.1.shape)
                        }))
            #expect(expected.count == 21)
            var readNames: [String] = []
            var checkpoints = 0
            try tower.loadNativeWeightsSerial(
                expectedDType: .float16,
                read: { name, shape in
                    readNames.append(name)
                    return MLXArray.ones(shape, dtype: .float16)
                },
                checkpoint: { arrays in
                    eval(arrays)
                    checkpoints += 1
                })
            #expect(readNames == expected.keys.sorted())
            #expect(checkpoints == expected.count)
            #expect(tower.parameters().flattened().allSatisfy { $0.1.dtype == .float16 })
            #expect(tower.trainableParameters().flattened().isEmpty)
            let second = try Qwen35VisionTower(configuration: c, textDType: .float32)
            var reads = 0
            #expect(throws: (any Error).self) {
                try second.loadNativeWeightsSerial(
                    expectedDType: .bfloat16,
                    read: { _, shape in
                        reads += 1
                        return MLXArray.zeros(shape, dtype: .float32)
                    },
                    checkpoint: { _ in
                        Issue.record("invalid source must not pass an evaluation checkpoint")
                    })
            }
            #expect(reads == 1)
        }
    }

}
