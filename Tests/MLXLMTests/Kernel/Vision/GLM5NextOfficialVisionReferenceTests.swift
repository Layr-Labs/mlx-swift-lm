import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXVLM

extension KernelTests {
    @Suite(
        .serialized,
        .enabled(
            if: GLM5NextReferenceFixture.enabled,
            "Set GLM53_REFERENCE_FIXTURE_DIR to the prepared canonical official CPU oracle"))
    struct GLM5NextOfficialVisionReferenceTests {
        @Test(arguments: ["image", "video"])
        func fixedPatchFeaturesMatchOfficial(caseName: String) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let test = try fixture.caseObject(caseName)
                let tower = try GLM5NextVisionTower(
                    configuration: fixture.configuration.visionConfig)
                let weights = fixture.weights.filter { $0.key.hasPrefix("model.visual.") }
                try tower.loadNativeWeights(weights, layout: .pytorch, dtype: .float32)
                let patches = try fixture.tensor(fixture.key(test, "pixel_values"))
                let output = try tower.forward(patches: patches, grids: grids(fixture, test: test))
                try fixture.compare(output, key: fixture.key(test, "vision_features"))
            }
        }

        @Test func rawImageResizeNormalizationAndPatchOrderMatchOfficial() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let test = try fixture.caseObject("image")
                let raw = try fixture.tensor(fixture.key(test, "raw_pixels"))
                let image = try rgbImage(raw)
                let prepared = try processor(fixture).prepareImage(image)
                let actualGrid = MLXArray(
                    [
                        Int64(prepared.grid.temporal), Int64(prepared.grid.height),
                        Int64(prepared.grid.width),
                    ], [1, 3])
                try fixture.compareIntegers(actualGrid, key: fixture.key(test, "grid_thw"))
                // Pixel tolerance is the oracle's explicit 1e-6, not the model's
                // looser matmul tolerance. This covers padding and patch order.
                try fixture.compare(
                    prepared.patches, key: fixture.key(test, "pixel_values"), absolute: 1e-6,
                    relative: 1e-6)
                #expect(prepared.timestamps.isEmpty)
            }
        }

        @Test func rawVideoFramesSamplingAndTemporalPackingMatchOfficial() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let test = try fixture.caseObject("video")
                let raw = try fixture.tensor(fixture.key(test, "raw_pixels"))
                guard raw.ndim == 4 && raw.dim(0) == 3 && raw.dim(1) == 3 else {
                    throw GLM5NextReferenceFixture.Failure(
                        "video fixture must contain three actual RGB frames")
                }
                let frames = try (0 ..< raw.dim(0)).map { try rgbImage(raw[$0]) }
                // The recorded oracle supplied all three frames, sampling off;
                // its separate sampling vector uses this exact FPS/duration.
                let selected = try GLM5NextMediaGeometry.sampleFrameIndices(
                    totalFrames: frames.count, sourceFPS: 2, duration: 1.5)
                try fixture.compareIntegers(
                    MLXArray(selected.map(Int64.init)), key: fixture.key(test, "sample_indices"))
                let decoded = try GLM5NextDecodedVideo(
                    frames: frames, framesPerSecond: 2, duration: 1.5)
                // This frozen case index contains tensor keys, not a sampling
                // flag. The authenticated run manifest records the input mode.
                // Read that mode rather than infer it from the expected grid.
                guard
                    let directory = ProcessInfo.processInfo.environment[
                        GLM5NextReferenceFixture.environmentKey]
                else { throw GLM5NextReferenceFixture.Failure("missing fixture directory") }
                let manifestURL = URL(fileURLWithPath: directory, isDirectory: true)
                    .appendingPathComponent("manifest.json")
                let attributes = try manifestURL.resourceValues(forKeys: [
                    .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
                ])
                guard attributes.isRegularFile == true, attributes.isSymbolicLink != true,
                    let size = attributes.fileSize, size > 0, size <= 2 * 1024 * 1024
                else { throw GLM5NextReferenceFixture.Failure("invalid video-mode manifest") }
                let manifestData = try Data(contentsOf: manifestURL)
                let manifestHash = SHA256.hash(data: manifestData)
                    .map { String(format: "%02x", $0) }.joined()
                guard manifestData.count == size,
                    manifestHash
                        == "028a6a2f2aa4beb30a2a846913ed139e278b65f5006aa77099bd4baaa8671f0d",
                    let manifest = try JSONSerialization.jsonObject(with: manifestData)
                        as? [String: Any],
                    let media = manifest["media"] as? [String: Any],
                    let mode = media["video_sampling"] as? String,
                    mode.hasSuffix("; sampling disabled")
                else {
                    throw GLM5NextReferenceFixture.Failure(
                        "unrecognized recorded video sampling mode")
                }
                let prepared = try processor(fixture, videoSamplingPolicy: .disabled).prepareVideo(
                    decoded)
                let actualGrid = MLXArray(
                    [
                        Int64(prepared.grid.temporal), Int64(prepared.grid.height),
                        Int64(prepared.grid.width),
                    ], [1, 3])
                // Keep the official three-source-frame geometry. Do not replace
                // it with an expected grid recomputed after duplicating frame 2.
                try fixture.compareIntegers(actualGrid, key: fixture.key(test, "grid_thw"))
                try fixture.compare(
                    prepared.patches, key: fixture.key(test, "pixel_values"), absolute: 1e-6,
                    relative: 1e-6)
                #expect(prepared.timestamps == [0, 1])
            }
        }

        @Test(arguments: ["image", "video"])
        func rootMediaInjectionAndLogitsMatchOfficial(caseName: String) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let test = try fixture.caseObject(caseName)
                let tokens = try fixture.tensor(fixture.key(test, "input_ids"))
                let patches = try fixture.tensor(fixture.key(test, "pixel_values"))
                let frames = try grids(fixture, test: test).map(\.thw)
                let text = LMInput.Text(
                    tokens: tokens, mask: MLXArray.ones(tokens.shape, dtype: .bool))
                let input: LMInput
                if caseName == "image" {
                    input = LMInput(
                        text: text, image: LMInput.ProcessedImage(pixels: patches, frames: frames))
                } else {
                    input = LMInput(
                        text: text, video: LMInput.ProcessedVideo(pixels: patches, frames: frames))
                }
                try fixture.withRootModel { model in
                    let result = try model.forward(input)
                    try diagnoseMedia(
                        fixture, test: test, name: caseName, model: model,
                        tokens: tokens, patches: patches, actual: result)
                    try fixture.compare(result.hidden, key: fixture.key(test, "final_hidden"))
                    try fixture.compare(result.logits, key: fixture.key(test, "logits"))
                }
            }
        }

        /// Opt-in synthetic evidence only; never changes the frozen assertions.
        private func diagnoseMedia(
            _ fixture: GLM5NextReferenceFixture, test: [String: Any], name: String,
            model: GLM5NextModel, tokens: MLXArray, patches: MLXArray, actual: GLM5NextTextOutput
        ) throws {
            guard let path = ProcessInfo.processInfo.environment["GLM53_VISION_DIAGNOSTIC_DIR"]
            else { return }
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            let attributes = try FileManager.default.attributesOfItem(atPath: directory.path)
            guard attributes[.type] as? FileAttributeType == .typeDirectory,
                (attributes[.posixPermissions] as? NSNumber)?.intValue == 0o700
            else {
                throw GLM5NextReferenceFixture.Failure(
                    "diagnostic directory must be a private directory")
            }
            let destination = directory.appendingPathComponent(name + "-native.safetensors")
            guard !FileManager.default.fileExists(atPath: destination.path) else {
                throw GLM5NextReferenceFixture.Failure("refusing to overwrite diagnostic evidence")
            }
            try MLX.withError { error in
                var arrays: [String: MLXArray] = [
                    "root.hidden": actual.hidden, "root.logits": actual.logits,
                ]
                var stage = 0
                let features = try model.visionTower.forward(
                    patches: patches, grids: grids(fixture, test: test)
                ) { roots in
                    arrays["vision.stage.\(stage)"] = roots.last!
                    stage += 1
                }
                let referenceFeatures = try fixture.tensor(fixture.key(test, "vision_features"))
                arrays["vision.features"] = features
                arrays["vision.reference_features"] = referenceFeatures
                try diagnosticStats(
                    features, reference: referenceFeatures, label: name + ".vision.features")
                let raw = model.textTower.embed(tokens)
                let positions = tokens.asArray(Int.self).enumerated()
                    .filter { $0.element == model.configuration.imageTokenId }.map(\.offset)
                guard positions.count == features.dim(0) else {
                    throw GLM5NextReferenceFixture.Failure("diagnostic feature count")
                }
                arrays["text.raw_embeddings"] = raw
                let mask = MLXArray.ones(tokens.shape, dtype: .bool)
                for (label, featureSource) in [
                    ("official", referenceFeatures), ("native", features),
                ] {
                    var embeddings = raw[.ellipsis]
                    for (row, position) in positions.enumerated() {
                        embeddings[0, position, 0...] = featureSource[row]
                    }
                    eval(embeddings)
                    try error.check()
                    arrays["text.\(label).embeddings"] = embeddings
                    let result = try model.textTower.forward(
                        embeddings: embeddings, cache: nil, validity: mask)
                    eval(result.hidden, result.logits)
                    try error.check()
                    arrays["text.\(label).hidden"] = result.hidden
                    arrays["text.\(label).logits"] = result.logits
                    try diagnosticStats(
                        result.hidden,
                        reference: fixture.tensor(fixture.key(test, "final_hidden")),
                        label: name + ".text.\(label).hidden")
                    let cache = model.textTower.newTextCache()
                    var streams = broadcast(
                        embeddings.expandedDimensions(axis: 2),
                        to: [
                            1, tokens.dim(1), model.configuration.textConfig.hcMult,
                            model.configuration.textConfig.hiddenSize,
                        ])
                    for (index, layer) in model.textTower.model.layers.enumerated() {
                        streams = try layer.forward(
                            streams, cache: cache.layers[index], validity: mask)
                        eval(streams)
                        try error.check()
                        arrays["text.\(label).layer.\(index)"] = streams
                    }
                }
                try diagnosticStats(
                    arrays["text.native.hidden"]!, reference: actual.hidden,
                    label: name + ".manual_injection_vs_root")
                eval(Array(arrays.values))
                try error.check()
                try save(arrays: arrays, url: destination)
                try error.check()
                print("GLM53_DIAGNOSTIC saved \(destination.path); stages=\(stage)")
            }
        }

        private func diagnosticStats(_ actual: MLXArray, reference: MLXArray, label: String) throws
        {
            guard actual.shape == reference.shape else {
                throw GLM5NextReferenceFixture.Failure("diagnostic shape \(label)")
            }
            let (a, b) = try MLX.withError { error in
                eval(actual, reference)
                try error.check()
                let values = (actual.asArray(Float.self), reference.asArray(Float.self))
                try error.check()
                return values
            }
            var maximum: Float = 0
            var ratio: Float = 0
            var index = 0
            for i in a.indices {
                let difference = abs(a[i] - b[i])
                if difference > maximum {
                    maximum = difference
                    index = i
                }
                ratio = max(ratio, difference / (3e-5 + 3e-5 * abs(b[i])))
            }
            print(
                "GLM53_DIAGNOSTIC \(label) max_abs=\(maximum) max_bound_ratio=\(ratio) index=\(index)"
            )
        }

        private func processor(
            _ fixture: GLM5NextReferenceFixture,
            videoSamplingPolicy: GLM5NextVideoSamplingPolicy = .official
        ) throws -> GLM5NextProcessor {
            // Tokenization is not invoked in these numerical preprocessing
            // tests. The shared seam only supplies fixed official marker IDs.
            try GLM5NextProcessor(
                configuration: .init(
                    patchSize: fixture.configuration.visionConfig.patchSize,
                    imageMinimumTokens: 16, imageMaximumTokens: 64,
                    videoMinimumTokens: 16, videoMaximumTokens: 128,
                    videoSamplingPolicy: videoSamplingPolicy),
                tokenizer: GLM5NextTestTokenizer())
        }

        private func grids(_ fixture: GLM5NextReferenceFixture, test: [String: Any]) throws
            -> [GLM5NextVisionGrid]
        {
            let values = try fixture.tensor(fixture.key(test, "grid_thw"))
            guard values.ndim == 2 && values.dim(1) == 3 else {
                throw GLM5NextReferenceFixture.Failure("grid fixture must be [items,3]")
            }
            let numbers = values.asType(.int64).asArray(Int64.self)
            return try (0 ..< values.dim(0)).map {
                try GLM5NextVisionGrid(
                    temporal: Int(numbers[3 * $0]), height: Int(numbers[3 * $0 + 1]),
                    width: Int(numbers[3 * $0 + 2]))
            }
        }

        private func rgbImage(_ raw: MLXArray) throws -> GLM5NextRGBImage {
            guard raw.ndim == 3 && raw.dim(0) == 3 && raw.dtype == .uint8 else {
                throw GLM5NextReferenceFixture.Failure("raw image must be uint8 CHW")
            }
            let height = raw.dim(1)
            let width = raw.dim(2)
            let values = raw.asArray(UInt8.self)
            var rgb: [UInt8] = []
            rgb.reserveCapacity(values.count)
            for pixel in 0 ..< (height * width) {
                for channel in 0 ..< 3 { rgb.append(values[channel * height * width + pixel]) }
            }
            return try GLM5NextRGBImage(width: width, height: height, pixels: rgb)
        }
    }
}
