import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXVLM

extension KernelTests {
    @Suite struct GLM5NextMultimodalTests {
        private func close(_ actual: MLXArray, _ expected: MLXArray) {
            #expect(actual.shape == expected.shape)
            #expect(
                abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
                    < 0.0001)
        }

        private func patches(frames: Int = 1, offset: Float = 0) -> MLXArray {
            MLXArray(
                (0 ..< (frames * 4 * 24)).map { Float($0 % 19 - 9) * 0.07 + offset },
                [frames * 4, 24])
        }

        @Test func localRootLoadsAllPartitionsAndTextMatchesTargetWithoutGlobalRequestState() throws
        {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let loader = try fixture.loader()
                let model = try GLM5NextModel(artifact: loader)
                #expect(model.mtp == nil)
                #expect(loader.unresolved(in: .text).isEmpty)
                #expect(loader.unresolved(in: .vision).isEmpty)
                #expect(loader.unresolved(in: .mtp).count == 31)
                let tokens = MLXArray([Int32(2), 3, 4], [1, 3])
                let input = LMInput(tokens: tokens)
                let first = model.newTextCache()
                let second = model.newTextCache()
                let expected = try model.textTower.forward(tokens: tokens, cache: nil)
                close(try model.forward(input, cache: first).logits, expected.logits)
                #expect(first.offset == 3 && second.offset == 0)
                close(try model.forward(input, cache: second).hidden, expected.hidden)
                let next = LMInput(tokens: MLXArray([Int32(5)], [1, 1]))
                close(
                    try model.forward(next, cache: first).logits,
                    try model.forward(next, cache: second).logits)
                #expect(first.offset == 4 && second.offset == 4)
                let standardCache = model.newCache(parameters: nil)
                let prepared = try model.prepare(input, cache: standardCache, state: nil, prefill: PrefillParameters(stepSize: 1, chunking: .remainder))
                if case .logits(let output) = prepared {
                    close(output.logits, expected.logits)
                } else {
                    Issue.record("root should return evaluated prompt logits")
                }
            }
        }

        @Test func imagePlacementMatchesExplicitTowerEmbeddingReplacementAndLegacyLifecycle() throws
        {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let model = try GLM5NextModel(artifact: fixture.loader())
                let tokens = MLXArray([Int32(2), 10, 3], [1, 3])
                let pixels = patches()
                let input = LMInput(
                    text: .init(tokens: tokens),
                    image: .init(pixels: pixels, frames: [THW(1, 2, 2)]))
                let placement = GLM5NextMediaPlacement(
                    modality: .image, itemIndex: 0,
                    tokenRange: 1 ..< 2, featureRange: 0 ..< 1)
                let detailed = GLM5NextPreparedInput(input: input, placements: [placement])
                var embedded = model.textTower.embed(tokens)
                let features = try model.visionTower.forward(
                    patches: pixels,
                    grids: [GLM5NextVisionGrid(temporal: 1, height: 2, width: 2)])
                embedded[0, 1 ..< 2, 0...] = features
                let expected = try model.textTower.forward(embeddings: embedded, cache: nil)
                close(try model.forward(detailed).logits, expected.logits)
                close(try model.forward(input).hidden, expected.hidden)
                let plain = try model.textTower.forward(tokens: tokens, cache: nil)
                #expect(abs(expected.hidden - plain.hidden).max().item(Float.self) > 0.0001)
                let cache = model.newCache(parameters: nil)
                if case .logits(let result) = try model.prepare(
                    input, cache: cache, state: nil, prefill: PrefillParameters())
                {
                    close(result.logits, expected.logits)
                } else {
                    Issue.record("media prompt was silently left unprocessed")
                }
            }
        }

        @Test func twoVideoFramesUseExactIndependentFramePlacements() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let model = try GLM5NextModel(artifact: fixture.loader())
                // Expanded video uses image IDs plus delimiters/timestamp text.
                let tokens = MLXArray([Int32(2), 12, 10, 13, 4, 12, 10, 13, 5], [1, 9])
                let pixels = patches(frames: 2)
                let input = LMInput(
                    text: .init(tokens: tokens),
                    video: .init(pixels: pixels, frames: [THW(2, 2, 2)]))
                let placements: [GLM5NextMediaPlacement] = [
                    .init(
                        modality: .video, itemIndex: 0, tokenRange: 2 ..< 3, featureRange: 0 ..< 1),
                    .init(
                        modality: .video, itemIndex: 0, tokenRange: 6 ..< 7, featureRange: 1 ..< 2),
                ]
                var embedded = model.textTower.embed(tokens)
                let grid = try GLM5NextVisionGrid(temporal: 1, height: 2, width: 2)
                embedded[0, 2 ..< 3, 0...] = try model.visionTower.forward(
                    patches: pixels[0 ..< 4], grids: [grid])
                embedded[0, 6 ..< 7, 0...] = try model.visionTower.forward(
                    patches: pixels[4 ..< 8], grids: [grid])
                let expected = try model.textTower.forward(embeddings: embedded, cache: nil)
                close(try model.forward(input, placements: placements).hidden, expected.hidden)
                close(try model.forward(input).logits, expected.logits)
                // Timestamp/text rows are not overwritten with media features.
                close(embedded[0, 4], model.textTower.embed(tokens)[0, 4])
            }
        }

        @Test func mixedMediaRequiresPureDetailedMetadataAndRetainsSeparateFeatureOrigins() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let model = try GLM5NextModel(artifact: fixture.loader())
                let tokens = MLXArray([Int32(10), 2, 12, 10, 13, 3], [1, 6])
                let image = patches(offset: 0.1)
                let video = patches(offset: -0.1)
                let input = LMInput(
                    text: .init(tokens: tokens),
                    image: .init(pixels: image, frames: [THW(1, 2, 2)]),
                    video: .init(pixels: video, frames: [THW(1, 2, 2)]))
                let placements: [GLM5NextMediaPlacement] = [
                    .init(
                        modality: .image, itemIndex: 0, tokenRange: 0 ..< 1, featureRange: 0 ..< 1),
                    .init(
                        modality: .video, itemIndex: 0, tokenRange: 3 ..< 4, featureRange: 0 ..< 1),
                ]
                let output = try model.forward(.init(input: input, placements: placements))
                #expect(output.logits.shape == [1, 6, 32])
                #expect(output.logits.asArray(Float.self).allSatisfy { $0.isFinite })
                #expect(throws: GLM5NextVisionError.self) { try model.forward(input) }
            }
        }

        @Test func mediaMismatchMaskLimitsAndUnexpandedTokensFailWithoutAdvancingCache() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let model = try GLM5NextModel(artifact: fixture.loader())
                let cache = model.newTextCache()
                let tokens = MLXArray([Int32(2), 10, 3], [1, 3])
                let input = LMInput(
                    text: .init(tokens: tokens),
                    image: .init(pixels: patches(), frames: [THW(1, 2, 2)]))
                for placement in [
                    GLM5NextMediaPlacement(
                        modality: .image, itemIndex: 0, tokenRange: 0 ..< 1, featureRange: 0 ..< 1),
                    .init(
                        modality: .image, itemIndex: 1, tokenRange: 1 ..< 2, featureRange: 0 ..< 1),
                    .init(
                        modality: .image, itemIndex: 0, tokenRange: 1 ..< 2, featureRange: 1 ..< 2),
                ] {
                    #expect(throws: GLM5NextVisionError.self) {
                        try model.forward(input, placements: [placement], cache: cache)
                    }
                    #expect(cache.offset == 0)
                }
                let masked = LMInput(
                    text: .init(tokens: tokens, mask: MLXArray([true, false, true], [1, 3])),
                    image: input.image)
                #expect(throws: GLM5NextVisionError.self) {
                    try model.forward(masked, cache: cache)
                }
                #expect(throws: GLM5NextVisionError.self) {
                    try model.forward(input, cache: cache, limits: .init(maximumPatches: 3))
                }
                #expect(throws: GLM5NextVisionError.self) {
                    try model.forward(LMInput(tokens: tokens), cache: cache)
                }
                let missingGrid = LMInput(text: input.text, image: .init(pixels: patches()))
                #expect(throws: GLM5NextVisionError.self) {
                    try model.forward(missingGrid, cache: cache)
                }
                let rawVideo = LMInput(
                    text: .init(tokens: MLXArray([Int32(11)], [1, 1])),
                    video: .init(pixels: patches(), frames: [THW(1, 2, 2)]))
                #expect(throws: GLM5NextVisionError.self) {
                    try model.forward(rawVideo, cache: cache)
                }
                #expect(cache.offset == 0)
            }
        }

        @Test func failingEvaluationFenceDoesNotPublishTextOrMediaRequestState() throws {
            enum Stop: Error { case requested }
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let model = try GLM5NextModel(artifact: fixture.loader())
                let cache = model.newTextCache()
                let plain = LMInput(tokens: MLXArray([Int32(2), 3], [1, 2]))
                // Input validation evaluates first; stop at the output fence.
                var calls = 0
                #expect(throws: Stop.self) {
                    try model.forward(
                        plain, cache: cache,
                        checkpoint: { _ in
                            calls += 1
                            if calls == 2 { throw Stop.requested }
                        })
                }
                #expect(cache.offset == 0)
                let media = LMInput(
                    text: .init(tokens: MLXArray([Int32(10)], [1, 1])),
                    image: .init(pixels: patches(), frames: [THW(1, 2, 2)]))
                #expect(throws: Stop.self) {
                    try model.forward(
                        media, cache: cache, checkpoint: { _ in throw Stop.requested })
                }
                #expect(cache.offset == 0)
                #expect(
                    try model.forward(plain, cache: cache).logits.asArray(Float.self).allSatisfy(
                        \.isFinite))
                #expect(cache.offset == 2)
            }
        }

        @Test func deferredRealMTPUsesItsRetainedPartitionAndIndependentRequestState() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let loader = try fixture.loader()
                let model = try GLM5NextModel(artifact: loader)
                try model.loadMTPPartition()
                let mtp = try #require(model.mtp)
                #expect(loader.unresolved(in: .mtp).isEmpty)
                #expect(mtp.targetIdentity == ObjectIdentifier(model.textTower))
                let first = try mtp.newRequestState()
                let second = try mtp.newRequestState()
                let tokens = MLXArray([Int32(2), 3], [1, 2])
                let hidden = try model.forward(LMInput(tokens: tokens)).hidden
                let result = try mtp.primePrompt(
                    tokens: tokens, targetHidden: hidden,
                    carryToken: MLXArray([Int32(4)], [1, 1]), state: first)
                #expect(result.logits.shape == [1, 1, 32])
                #expect(first.committedInputCount == 2 && second.committedInputCount == 0)
                #expect(throws: GLM5NextArtifactError.self) { try model.loadMTPPartition() }
            }
        }

        @Test func changedDeferredMTPAndAbsentMTPFailExplicitly() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let model = try GLM5NextModel(artifact: fixture.loader())
                var config = try Data(contentsOf: fixture.url("config.json"))
                config.append(32)
                try config.write(to: fixture.url("config.json"))
                #expect(throws: GLM5NextArtifactError.self) { try model.loadMTPPartition() }
                #expect(model.mtp == nil)
                let noMTP = try GLM5NextLocalArtifactFixture(mtp: false)
                let noMTPModel = try GLM5NextModel(artifact: noMTP.loader())
                #expect(noMTPModel.artifact.unresolved(in: .mtp).isEmpty)
                #expect(throws: GLM5NextArtifactError.self) { try noMTPModel.loadMTPPartition() }
            }
        }
    }
}
