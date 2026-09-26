import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXVLM

@Suite("DiffusionGemma pinned media oracle", .serialized)
struct DiffusionGemmaVisionOracleTests {
    struct Fixture: Decodable {
        let reference: String
        let model_config: DiffusionGemmaConfiguration
        let parameter_count: Int
        let cases: [Case]
        struct Case: Decodable {
            let name: String
            let images: Int
            let soft_tokens_per_image: Int
        }
    }

    func fixture(_ precision: String = "fp32") throws -> (
        Fixture, [String: MLXArray], DiffusionGemma
    ) {
        let resource =
            precision == "video" ? "diffusiongemma-video-types-oracle"
            : precision == "bf16" ? "diffusiongemma-vision-bf-oracle" : "diffusiongemma-vision-oracle"
        let metadata = try #require(
            Bundle.module.url(forResource: resource, withExtension: "json"))
        let payload = try #require(
            Bundle.module.url(
                forResource: resource, withExtension: "safetensors"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: metadata))
        let arrays = try loadArrays(url: payload)
        let model = DiffusionGemma(fixture.model_config)
        let weights = model.sanitize(weights: arrays.filter { $0.key.hasPrefix("model.") })
        #expect(weights.count == fixture.parameter_count)
        #expect(model.parameters().flattened().count == fixture.parameter_count)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return (fixture, arrays, model)
    }

    private func exact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        eval(actual, expected)
        #expect(actual.shape == expected.shape, "\(label): shape")
        #expect(actual.dtype == expected.dtype, "\(label): dtype")
        let matches =
            actual.asArray(Float.self).map(\.bitPattern)
            == expected.asArray(Float.self).map(\.bitPattern)
        #expect(matches, "\(label): raw FP32 mismatch; preserve oracle")
    }

    @Test(arguments: ["fp32", "bf16"]) func towerAndProjectionMatchReference(_ precision: String)
        throws
    {
        let (fixture, arrays, model) = try fixture(precision)
        #expect(fixture.reference == "e79b0e041677ec4ca5333ba750376bb4e8c434cb")
        let tower = try #require(model.model.encoder.visionTower)
        let projection = try #require(model.model.encoder.embedVision)
        let tracePath =
            precision == "fp32"
            ? ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_VISION_TRACE"] : nil
        var traced = [String: MLXArray]()
        for item in fixture.cases {
            let observe: ((String, MLXArray) -> Void)? = tracePath.map { _ in
                { name, array in traced[item.name + "." + name] = array }
            }
            let features = tower(
                arrays[item.name + ".pixels"]!, outputLength: item.soft_tokens_per_image,
                trace: observe
            )
            .reshaped(1, item.images * item.soft_tokens_per_image, -1)
            exact(features, arrays[item.name + ".features"]!, item.name + ": vision features")
            exact(
                projection(features), arrays[item.name + ".projected"]!, item.name + ": projected")
        }
        if let tracePath {
            try #require(!FileManager.default.fileExists(atPath: tracePath))
            try save(arrays: traced, url: URL(fileURLWithPath: tracePath))
        }
    }

    @Test(arguments: ["fp32", "bf16"]) func mediaEncoderStateAndCanvasLogitsMatchReference(
        _ precision: String
    ) throws {
        let (fixture, arrays, model) = try fixture(precision)
        for item in fixture.cases {
            let tokens = arrays[item.name + ".tokens"]!
            let cache = try model.makeCache(expectedPromptLength: tokens.dim(1))
            let hidden = try model.encode(
                tokenIds: tokens, cache: cache, pixelValues: arrays[item.name + ".pixels"]!,
                visualOutputLengths: Array(
                    repeating: item.soft_tokens_per_image, count: item.images))
            exact(hidden, arrays[item.name + ".hidden"]!, item.name + ": media encoder")
            for (index, snapshot) in cache.snapshots().enumerated() {
                exact(snapshot.keys, arrays[item.name + ".cache\(index).keys"]!, "media keys")
                exact(snapshot.values, arrays[item.name + ".cache\(index).values"]!, "media values")
            }
            exact(
                try model.denoise(canvasIds: arrays[item.name + ".canvas"]!, cache: cache),
                arrays[item.name + ".logits"]!, item.name + ": media canvas logits")
        }
    }

    @Test func existingGemmaVisionContractRemainsBitExact() throws {
        let (fixture, arrays, _) = try fixture()
        let url = try #require(
            Bundle.module.url(
                forResource: "diffusiongemma-legacy-vision", withExtension: "safetensors"))
        let expected = try loadArrays(url: url)
        let tower = Gemma4VisionTower(try #require(fixture.model_config.visionConfig))
        let prefix = "model.encoder.vision_tower."
        let weights = Dictionary(
            uniqueKeysWithValues: arrays.filter {
                $0.key.hasPrefix(prefix)
            }.map { key, value in
                (
                    String(key.dropFirst(prefix.count)).replacingOccurrences(
                        of: ".linear.", with: "."), value
                )
            })
        try tower.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        for item in fixture.cases {
            let actual = tower(
                arrays[item.name + ".pixels"]!, outputLength: item.soft_tokens_per_image
            )
            .reshaped(1, item.images * item.soft_tokens_per_image, -1)
            exact(actual, expected[item.name]!, item.name + ": existing Gemma4 contract")
        }
    }

    @Test func videoMediaTypesBindFeaturesWithoutConfiguredVideoToken() throws {
        let (_, arrays, model) = try fixture("video")
        #expect(model.configuration.videoTokenId == nil)
        let tokens = arrays["video.tokens"]!
        let types = arrays["video.token_types"]!.asArray(Int32.self)
        var next: Int32 = -1
        var previous = false
        let blocks = types.map { type -> Int32 in
            let visual = type == 1 || type == 2
            if visual && !previous { next += 1 }
            previous = visual
            return visual ? next : -1
        }
        let cache = try model.makeCache(expectedPromptLength: tokens.dim(1))
        let hidden = try model.encode(tokenIds: tokens, cache: cache,
            pixelValues: arrays["video.pixels"]!, visualOutputLengths: [4],
            visualBlockIds: MLXArray(blocks).reshaped(tokens.shape))
        exact(hidden, arrays["video.hidden"]!, "video-type encoder")
        for (index, snapshot) in cache.snapshots().enumerated() {
            exact(snapshot.keys, arrays["video.cache\(index).keys"]!, "video keys")
            exact(snapshot.values, arrays["video.cache\(index).values"]!, "video values")
        }
        exact(try model.denoise(canvasIds: arrays["video.canvas"]!, cache: cache),
            arrays["video.logits"]!, "video canvas logits")
    }

    @Test func preparedFeaturesAndChunkedMediaPreserveReferenceState() throws {
        let (fixture, arrays, model) = try fixture("bf16")
        for item in fixture.cases {
            let tokens = arrays[item.name + ".tokens"]!
            let tokenIDs = tokens.asArray(Int32.self)
            var frames = [DiffusionGemmaMediaInput.Frame]()
            var index = 0, image = 0
            while index < tokenIDs.count {
                if tokenIDs[index] != Int32(model.configuration.imageTokenId) { index += 1; continue }
                let start = index
                while index < tokenIDs.count && tokenIDs[index] == Int32(model.configuration.imageTokenId) { index += 1 }
                let pixels = arrays[item.name + ".pixels"]![image..<(image + 1), 0..., 0..., 0...]
                frames.append(.init(kind: .image, pixels: pixels,
                    span: .init(tokenOffset: start, length: index - start), timestampSeconds: nil))
                image += 1
            }
            let input = DiffusionGemmaMediaInput(tokens: tokenIDs, frames: frames)
            let prepared = try #require(try model.prepareVision(input))
            for width in [1, 3, 512] {
                let plan = try DiffusionGemmaVisualEmbeddings(input: prepared, promptCount: tokenIDs.count,
                    hiddenSize: model.configuration.textConfig.hiddenSize)
                let cache = try model.makeCache(expectedPromptLength: tokenIDs.count)
                while cache.position < tokenIDs.count {
                    let start = cache.position
                    let count = try plan.chunkLength(start: start, requested: width, promptCount: tokenIDs.count)
                    try plan.encode(model: model, tokens: tokens[0..., start..<(start + count)], cache: cache, start: start)
                    eval(cache.stateArrays())
                }
                for (layer, snapshot) in cache.snapshots().enumerated() {
                    exact(snapshot.keys, arrays[item.name + ".cache\(layer).keys"]!, "prepared media keys width\(width)")
                    exact(snapshot.values, arrays[item.name + ".cache\(layer).values"]!, "prepared media values width\(width)")
                }
                exact(try model.denoise(canvasIds: arrays[item.name + ".canvas"]!, cache: cache),
                    arrays[item.name + ".logits"]!, "prepared media logits width\(width)")
            }
        }
    }
}
