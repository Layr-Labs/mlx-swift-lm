// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLLM
import Testing

@testable import MLXVLM

/// The fused vision attention kernel (the GPU default) against the eager
/// reference-ordered graph on the same BF16 synthetic tower, and the `.automatic` selection.
extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextVisionFusedAttentionTests {
        /// Head dimension 64 (hidden 128, 2 heads): one of the fused head sizes.
        private static func tower(depth: Int = 2) throws -> GLM5NextVisionTower {
            let configuration = try JSONDecoder().decode(
                GLM5NextVisionConfiguration.self,
                from: Data(
                    """
                    {"model_type":"glm5_next_vision","depth":\(depth),"hidden_size":128,
                     "intermediate_size":256,"num_heads":2,"out_hidden_size":96,"patch_size":14,
                     "projection_intermediate_size":192,"spatial_merge_size":2,"temporal_patch_size":2,
                     "in_channels":3,"attention_bias":true,"attention_dropout":0.0,"rms_norm_eps":1e-5,
                     "swiglu_limit":10.0,"hidden_act":"silu","image_size":448,"initializer_range":0.02}
                    """.utf8))
            let tower = try GLM5NextVisionTower(configuration: configuration)
            var weights: [String: MLXArray] = [:]
            MLXRandom.seed(0x474C_4D35)
            for (name, shape) in GLM5NextVisionTower.expectedTensorShapes(
                configuration: configuration)
            {
                let scale: Float =
                    name.hasSuffix("norm1.weight") || name.hasSuffix("norm2.weight")
                        || name.hasSuffix("_norm.weight") || name.hasSuffix("post_layernorm.weight")
                        || name.hasSuffix("post_projection_norm.weight") ? 0 : 0.05
                let value =
                    scale == 0
                    ? MLXArray.ones(shape) : MLXRandom.normal(shape) * scale
                weights["model.visual." + name] = value.asType(.bfloat16)
            }
            try tower.loadNativeWeights(weights)
            return tower
        }

        private static func metrics(_ value: MLXArray, _ target: MLXArray) -> (
            cosine: Float, relative: Float
        ) {
            let a = value.asType(.float32)
            let b = target.asType(.float32)
            let cosine =
                ((a * b).sum(axis: -1) / (sqrt((a * a).sum(axis: -1)) * sqrt((b * b).sum(axis: -1))))
                .min().item(Float.self)
            let relative = (sqrt(((a - b) * (a - b)).sum()) / sqrt((b * b).sum())).item(Float.self)
            return (cosine, relative)
        }

        @Test func fusedMatchesTiledReferenceGraphOnGPU() throws {
            let tower = try Self.tower()
            for (height, width) in [(8, 8), (24, 32), (40, 18)] {
                let grid = try GLM5NextVisionGrid(temporal: 1, height: height, width: width)
                MLXRandom.seed(UInt64(height * 1000 + width))
                let patches = MLXRandom.normal([height * width, 3 * 2 * 14 * 14])
                let tiled = try tower.forward(
                    patches: patches, grids: [grid], limits: .init(attention: .tiled))
                let fused = try tower.forward(
                    patches: patches, grids: [grid], limits: .init(attention: .fused))
                let automatic = try tower.forward(
                    patches: patches, grids: [grid], limits: .init(attention: .automatic))
                let defaulted = try tower.forward(patches: patches, grids: [grid])
                #expect(fused.shape == tiled.shape && fused.shape == [height * width / 4, 96])
                let (cosine, relative) = Self.metrics(fused, tiled)
                // Both are BF16 graphs; they differ only in score rounding.
                #expect(
                    cosine > 0.999 && relative < 0.02,
                    "\(height)x\(width): cos=\(cosine) rel=\(relative)")
                #expect(Self.metrics(automatic, fused).relative == 0, "GPU automatic selects fused")
                #expect(Self.metrics(defaulted, fused).relative == 0, "the GPU default is fused")
                // Multi-frame calls keep one fused launch per frame.
                let twoFrames = try tower.forward(
                    patches: concatenated([patches, patches], axis: 0),
                    grids: [grid, grid], limits: .init(attention: .fused))
                #expect(Self.metrics(twoFrames[..<(height * width / 4)], fused).relative == 0)
            }
        }

        /// Off the default GPU stream MLX would not fuse: `.automatic` takes the
        /// tiled graph bitwise, and an explicit `.fused` refuses instead of
        /// falling back to a materialized frame-square score tensor.
        @Test func offDefaultGPUStreamAutomaticIsTiledAndFusedRefuses() throws {
            let tower = try Self.tower(depth: 1)
            let grid = try GLM5NextVisionGrid(temporal: 1, height: 8, width: 12)
            let patches = MLXRandom.normal([96, 3 * 2 * 14 * 14])
            func check() throws {
                #expect(!glm5NextVisionOnDefaultGPUStream())
                let automatic = try tower.forward(
                    patches: patches, grids: [grid], limits: .init(attention: .automatic))
                let defaulted = try tower.forward(patches: patches, grids: [grid])
                let tiled = try tower.forward(
                    patches: patches, grids: [grid], limits: .init(attention: .tiled))
                #expect(arrayEqual(automatic, tiled).item(Bool.self))
                #expect(arrayEqual(defaulted, tiled).item(Bool.self))
                #expect(throws: GLM5NextVisionError.self) {
                    _ = try tower.forward(
                        patches: patches, grids: [grid], limits: .init(attention: .fused))
                }
            }
            try Device.withDefaultDevice(.cpu) { try check() }
            try Stream.withNewDefaultStream(device: .cpu) { try check() }
            try Stream.withNewDefaultStream(device: .gpu) { try check() }
            // Head dimensions MLX does not fuse refuse even on the default GPU stream.
            #expect(glm5NextVisionOnDefaultGPUStream())
            #expect(try !GLM5NextVisionAttentionKernel.automatic.usesFused(headDimension: 4))
            #expect(throws: GLM5NextVisionError.self) {
                _ = try GLM5NextVisionAttentionKernel.fused.usesFused(headDimension: 4)
            }
            #expect(try GLM5NextVisionAttentionKernel.fused.usesFused(headDimension: 64))
        }
    }
}
