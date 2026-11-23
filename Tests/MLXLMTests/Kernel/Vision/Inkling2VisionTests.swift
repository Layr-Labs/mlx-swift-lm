// Copyright © 2026 Eigen Labs.

import CoreGraphics
import CoreImage
import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

// The oracle resources come from the pinned transformers implementation
// (PR 51050 at e3a65e2c) on CPU in float32: a random-weight two-layer tower
// and the reference image processor on deterministic synthetic images. The
// generator script is `make_tiny_vision_fixture.py` in the onboarding
// evidence folder.

private struct Inkling2TinyOracle: Decodable {
    struct Case: Decodable {
        let name: String
        let height: Int?
        let width: Int?
        let channels: Int?
        let png: Bool?
        let grid: [Int]?
        let grids: [[Int]]?
        let pixel_sha256: String?
    }
    struct Large: Decodable {
        let height: Int
        let width: Int
        let grid: [Int]
        let pixel_sha256: String
    }

    let reference: String
    let vision_config: Inkling2VisionConfiguration
    let weight_count: Int
    let cases: [Case]
    let large: [Large]

    static func load() throws -> (Inkling2TinyOracle, [String: MLXArray]) {
        let json = try #require(
            Bundle.module.url(forResource: "inkling2-vision-tiny-oracle", withExtension: "json"))
        let payload = try #require(
            Bundle.module.url(
                forResource: "inkling2-vision-tiny-oracle", withExtension: "safetensors"))
        return (
            try JSONDecoder().decode(Self.self, from: Data(contentsOf: json)),
            try loadArrays(url: payload)
        )
    }
}

/// The synthetic pattern the oracle script draws.
private func inkling2Pattern(height: Int, width: Int, channels: Int) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: height * width * channels)
    for y in 0 ..< height {
        for x in 0 ..< width {
            for c in 0 ..< 3 {
                bytes[(y * width + x) * channels + c] = UInt8(
                    (x * 7 + y * 13 + c * 101 + (x * y) % 97) % 256)
            }
            if channels == 4 {
                bytes[(y * width + x) * 4 + 3] = UInt8((x * 5 + y * 3) % 256)
            }
        }
    }
    return bytes
}

/// A CIImage over straight (non-premultiplied) RGB or RGBA bytes, as a decoder
/// would hand it to the processor.
private func inkling2CIImage(_ bytes: [UInt8], height: Int, width: Int, channels: Int) -> CIImage {
    var rgba = [UInt8](repeating: 255, count: height * width * 4)
    for index in 0 ..< height * width {
        for c in 0 ..< channels { rgba[index * 4 + c] = bytes[index * channels + c] }
    }
    let alpha: CGImageAlphaInfo = channels == 4 ? .last : .noneSkipLast
    let image = CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue),
        provider: CGDataProvider(data: Data(rgba) as CFData)!, decode: nil,
        shouldInterpolate: false, intent: .defaultIntent)!
    return CIImage(cgImage: image)
}

private func sha256(_ array: MLXArray) -> String {
    let data = array.asType(.float32).asData(access: .copy).data
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

/// A tiny Inkling 2 VLM: the text shape of `Inkling2PagedEngineTests` plus a
/// two-layer tower projecting to its hidden size.
enum Inkling2TinyVLM {
    static let imageTokenId = 50

    static var configuration: [String: Any] {
        [
            "model_type": "inkling2_loop",
            "image_token_id": imageTokenId,
            "text_config": [
                "hidden_size": 128, "num_hidden_layers": 6, "vocab_size": 64,
                "unpadded_vocab_size": 60, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 64, "d_rel": 4, "rel_extent": 24, "log_scaling_n_floor": 8,
                "log_scaling_alpha": 0.5, "qk_l2_clip_threshold": 8.0,
                "rel_l2_clip_threshold": 2.0, "local_layer_ids": [0, 2, 4],
                "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
                "sliding_window_size": 24, "n_layers_per_attn_res_block": 2,
                "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32,
                "loop_config": ["core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8],
            ] as [String: Any],
            "vision_config": [
                "model_type": "inkling2_vision", "hidden_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 4, "mlp_ratio": 4.0, "native_patch_size": 20,
                "adapter_merge_size": 2, "temporal_patch_size": 2, "posemb_grid_size": 8,
                "projection_hidden_dim": 128, "projector_bias": false, "text_hidden_size": 128,
                "layer_norm_eps": 1e-5,
            ] as [String: Any],
        ]
    }

    static func model(seed: UInt64 = 7) throws -> Inkling2VLM {
        let model = Inkling2VLM(
            try SyntheticModel.configuration(Inkling2VLMConfiguration.self, configuration))
        SyntheticModel.randomize(model, seed: seed)
        eval(model)
        return model
    }

    /// Patches of a deterministic `height x width` RGB image.
    static func image(height: Int, width: Int) throws -> LMInput.ProcessedImage {
        var rgba = [UInt8](repeating: 255, count: height * width * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                for c in 0 ..< 3 {
                    rgba[(y * width + x) * 4 + c] = UInt8((x * 7 + y * 13 + c * 101) % 256)
                }
            }
        }
        let cgImage = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: CGDataProvider(data: Data(rgba) as CFData)!, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent)!
        let processor = Inkling2Processor(
            Inkling2ProcessorConfiguration(), tokenizer: TestTokenizer())
        let (pixels, grid) = try processor.preprocess(
            image: CIImage(cgImage: cgImage), processing: nil)
        return LMInput.ProcessedImage(pixels: pixels, frames: [grid])
    }
}

extension KernelTests {

    /// The Inkling 2 processor and tower against the transformers oracle.
    @Suite(.serialized)
    struct Inkling2VisionTests {
        @Test func towerParametersMatchTheCheckpointNames() {
            // 4 embedding tensors + 2 ln_pre + 23 x 18 per layer + 2 projector weights.
            let model = Inkling2VisionModel(Inkling2VisionConfiguration())
            let names = Set(model.parameters().flattened().map(\.0))
            #expect(names.count == 422)
            #expect(names.contains("backbone.embeddings.position_embedding"))
            #expect(names.contains("backbone.layers.22.ls_2.gamma"))
            #expect(names.contains("backbone.layers.0.self_attn.out_proj.bias"))
            #expect(names.contains("proj_in.weight") && !names.contains("proj_in.bias"))
        }

        @Test func oversizedImagesScaleWithoutEdgeBleed() throws {
            let (width, height) = (5000, 300)
            let image = inkling2CIImage(
                [UInt8](repeating: 200, count: width * height * 3), height: height, width: width,
                channels: 3)
            let processor = Inkling2Processor(
                Inkling2ProcessorConfiguration(), tokenizer: TestTokenizer())
            let (pixels, grid) = try processor.preprocess(image: image, processing: nil)
            #expect([grid.t, grid.h, grid.w] as [Int] == [1, 7, 103])
            // Rows 246..<280 are padding; every image pixel keeps the flat value.
            let table = Inkling2ImagePatches.normalizationTable()
            let values = Set(pixels.asArray(Float.self))
            #expect(values == [table[0][200], table[0][256]])
        }

        @Test func featuresReplacePlaceholderRowsInOrder() throws {
            let tokens = [5, 9, 9, 6, 9, 7]
            let text = MLXArray((0 ..< 12).map { Float($0) }, [1, 6, 2])
            let first = MLXArray([100, 101, 102, 103] as [Float], [2, 2])
            let second = MLXArray([200, 201] as [Float], [1, 2])
            let merged = try inkling2SpliceImageFeatures(
                textEmbeddings: text, tokens: tokens, imageTokenId: 9, features: [first, second])
            #expect(
                merged.asArray(Float.self) == [0, 1, 100, 101, 102, 103, 6, 7, 200, 201, 10, 11])
            #expect(throws: VLMError.self) {
                try inkling2SpliceImageFeatures(
                    textEmbeddings: text, tokens: tokens, imageTokenId: 9, features: [first])
            }
        }

        @Test func fixtureIsPinned() throws {
            let (oracle, _) = try Inkling2TinyOracle.load()
            #expect(oracle.reference == "transformers e3a65e2c21abb9c25a03704f2f25852f9ad03a18")
        }

        /// The processor output for a fixture case, from its raw bytes or PNG.
        private func processed(
            _ item: Inkling2TinyOracle.Case, _ arrays: [String: MLXArray]
        ) throws -> (MLXArray, THW) {
            let processor = Inkling2Processor(
                Inkling2ProcessorConfiguration(), tokenizer: TestTokenizer())
            let image: CIImage
            if item.png == true {
                let png = try #require(arrays["case.\(item.name).png"])
                image = try #require(CIImage(data: Data(png.asArray(UInt8.self))))
            } else {
                let raw = try #require(arrays["case.\(item.name).raw"])
                image = inkling2CIImage(
                    raw.asArray(UInt8.self), height: try #require(item.height),
                    width: try #require(item.width), channels: try #require(item.channels))
            }
            return try processor.preprocess(image: image, processing: nil)
        }

        @Test func processorPixelsAreBitIdentical() throws {
            let (oracle, arrays) = try Inkling2TinyOracle.load()
            for item in oracle.cases where item.height != nil {
                let (pixels, grid) = try processed(item, arrays)
                #expect([grid.t, grid.h, grid.w] == item.grid, "\(item.name): grid")
                #expect(sha256(pixels) == item.pixel_sha256, "\(item.name): pixel hash")
            }
        }

        @Test func largeImagesMatchTheReferenceHash() throws {
            let (oracle, _) = try Inkling2TinyOracle.load()
            let processor = Inkling2Processor(
                Inkling2ProcessorConfiguration(), tokenizer: TestTokenizer())
            for item in oracle.large {
                let image = inkling2CIImage(
                    inkling2Pattern(height: item.height, width: item.width, channels: 3),
                    height: item.height, width: item.width, channels: 3)
                let (pixels, grid) = try processor.preprocess(image: image, processing: nil)
                #expect([grid.t, grid.h, grid.w] == item.grid, "\(item.width)x\(item.height)")
                #expect(sha256(pixels) == item.pixel_sha256, "\(item.width)x\(item.height)")
            }
        }

        private func tower() throws -> (Inkling2TinyOracle, [String: MLXArray], Inkling2VisionModel)
        {
            let (oracle, arrays) = try Inkling2TinyOracle.load()
            let model = Inkling2VisionModel(oracle.vision_config)
            var weights: [String: MLXArray] = [:]
            for (key, value) in arrays where key.hasPrefix("model.visual.") {
                weights[String(key.dropFirst("model.visual.".count))] = value
            }
            #expect(weights.count == oracle.weight_count)
            #expect(model.parameters().flattened().count == oracle.weight_count)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            return (oracle, arrays, model)
        }

        private func close(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
            #expect(actual.shape == expected.shape, "\(label): shape")
            let error = abs(actual.asType(.float32) - expected).max().item(Float.self)
            let scale = abs(expected).max().item(Float.self)
            #expect(error <= 1e-4 * max(scale, 1), "\(label): max error \(error) of \(scale)")
        }

        @Test func towerMatchesTheFloat32Reference() throws {
            let (oracle, arrays, model) = try tower()
            for item in oracle.cases where item.grid != nil {
                let (pixels, grid) = try processed(item, arrays)
                let features = try model.imageFeatures(pixels: pixels, grids: [grid])
                #expect(features.count == 1)
                close(features[0], try #require(arrays["case.\(item.name).features"]), item.name)
            }
        }

        @Test func imagesAttendOnlyWithinThemselves() throws {
            let (oracle, arrays, model) = try tower()
            let pair = try #require(oracle.cases.first { $0.name == "pair" }?.grids)
            let inputs = try ["odd_75x130", "one_token_40"].map { name in
                try processed(try #require(oracle.cases.first { $0.name == name }), arrays).0
            }
            let pixels = concatenated(inputs, axis: 0)
            let features = try model.imageFeatures(
                pixels: pixels, grids: pair.map { THW($0[0], $0[1], $0[2]) })
            #expect(features.map { $0.dim(0) } == [8, 1])
            close(
                concatenated(features, axis: 0), try #require(arrays["pair.features"]), "pair")
        }

        @Test func rejectsVideoGridsAndMismatchedPatches() throws {
            let (oracle, arrays, model) = try tower()
            let (pixels, _) = try processed(
                try #require(oracle.cases.first { $0.name == "odd_75x130" }), arrays)
            #expect(throws: VLMError.self) {
                try model.imageFeatures(pixels: pixels, grids: [THW(2, 2, 4)])
            }
            #expect(throws: VLMError.self) {
                try model.imageFeatures(pixels: pixels, grids: [THW(1, 2, 3)])
            }
        }

        /// The fused bf16 / fp16 LayerNorm rounds once: against float64 math on
        /// the same inputs, every output in the dtype's normal range is within
        /// one unit in the last place of the correctly rounded value, and
        /// nearly all equal it. (GPUs may flush fp16 subnormals.)
        @Test func layerNormRoundsOnce() throws {
            MLXRandom.seed(7)
            let rows = 37
            for (dtype, mantissa, minimumNormal) in [
                (DType.bfloat16, 7, 0x1p-126), (DType.float16, 10, 0x1p-14),
            ] {
                for width in [1024, 72] {
                    let x = (MLXRandom.normal([rows, width]) * 3 + 0.5).asType(dtype)
                    let w = (MLXRandom.normal([width]) * 0.2 + 1).asType(dtype)
                    let b = (MLXRandom.normal([width]) * 0.1).asType(dtype)
                    let norm = Inkling2VisionLayerNorm(dimensions: width, eps: 1e-5)
                    try norm.update(
                        parameters: ModuleParameters.unflattened(["weight": w, "bias": b]),
                        verify: .all)
                    let out = norm(x)
                    #expect(out.dtype == dtype && out.shape == [rows, width])

                    let xs = x.asType(.float32).asArray(Float.self).map(Double.init)
                    let ws = w.asType(.float32).asArray(Float.self).map(Double.init)
                    let bs = b.asType(.float32).asArray(Float.self).map(Double.init)
                    var exact: [Double] = []
                    for r in 0 ..< rows {
                        let row = xs[(r * width) ..< ((r + 1) * width)]
                        let mean = row.reduce(0, +) / Double(width)
                        let variance =
                            row.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(width)
                        let rstd = 1 / (variance + 1e-5).squareRoot()
                        exact += row.enumerated().map { ($1 - mean) * rstd * ws[$0] + bs[$0] }
                    }
                    let nearest = MLXArray(exact.map(Float.init)).asType(dtype)
                        .asType(.float32).asArray(Float.self)
                    let got = out.asType(.float32).asArray(Float.self)
                    var equal = 0
                    var worst = 0.0
                    for (g, n) in zip(got, nearest) where Double(abs(n)) >= minimumNormal {
                        let ulp =
                            pow(2, log2(Double(abs(n))).rounded(.down)) / pow(2, Double(mantissa))
                        worst = max(worst, abs(Double(g - n)) / ulp)
                        equal += g == n ? 1 : 0
                    }
                    let label = "\(dtype) width \(width)"
                    #expect(worst <= 1, "\(label): \(worst) ulp")
                    #expect(Double(equal) >= 0.995 * Double(got.count), "\(label): \(equal)")
                }
            }
        }
    }

    /// The `inkling2_loop` VLM wrapper over the native text model.
    @Suite
    struct Inkling2VLMWrapperTests {
        @Test func registersTheModelAndProcessor() async throws {
            let data = try JSONSerialization.data(withJSONObject: Inkling2TinyVLM.configuration)
            let model = try await VLMTypeRegistry.shared.createModel(
                configuration: data, modelType: "inkling2_loop")
            #expect(model is Inkling2VLM)
            let processorConfig = Data(
                """
                {"image_processor": {"do_convert_rgb": true, "do_normalize": true,
                 "do_rescale": true, "image_mean": [0.5, 0.5, 0.5],
                 "image_processor_type": "Inkling2ImageProcessor", "image_std": [0.5, 0.5, 0.5],
                 "rescale_factor": 0.00392156862745098, "size": {"height": 40, "width": 40}},
                 "image_token": "<|unused_200054|>", "processor_class": "Inkling2Processor"}
                """.utf8)
            let processor = try await VLMProcessorTypeRegistry.shared.createModel(
                configuration: processorConfig, processorType: "Inkling2Processor",
                tokenizer: TestTokenizer())
            #expect(processor is Inkling2Processor)
        }

        @Test func checkpointNamesSplitBetweenTowerAndText() throws {
            let model = try Inkling2TinyVLM.model()
            let names = model.parameters().flattened().map(\.0)
            #expect(names.contains("vision_tower.backbone.embeddings.position_embedding"))
            #expect(names.contains("language_model.model.embed.weight"))
            #expect(model.excludedCheckpointNamespaces.isEmpty)
            let sanitized = try model.sanitize(weights: [
                "model.visual.proj_in.weight": MLXArray.zeros([1])
            ])
            #expect(Array(sanitized.keys) == ["vision_tower.proj_in.weight"])
        }

        /// Every CBv2 protocol the text model serves through, the wrapper serves
        /// through too, so loading the checkpoint as a VLM never drops a path.
        @Test func wrapperKeepsEveryTextModelCBv2Conformance() throws {
            let vlm = try Inkling2TinyVLM.model()
            let text: Any = vlm.languageModel
            let wrapper: Any = vlm
            let checks: [(String, (Any) -> Bool)] = [
                ("recurrent", { $0 is any CBv2RecurrentLanguageModelForwardable }),
                (
                    "positioned recurrent",
                    { $0 is any CBv2PositionedRecurrentLanguageModelForwardable }
                ),
                ("embedding", { $0 is any CBv2EmbeddingForwardable }),
                (
                    "positioned recurrent embedding",
                    { $0 is any CBv2PositionedRecurrentEmbeddingForwardable }
                ),
                ("recurrent prefill", { $0 is any CBv2RecurrentLanguageModelPrefillForwardable }),
                ("capabilities", { $0 is any CBv2ModelCapabilityProviding }),
                ("decode command buffers", { $0 is any CBv2DecodeCommandBufferLimitsProviding }),
                ("MTP", { $0 is any CBv2MTPForwardable }),
                ("recurrent MTP", { $0 is any CBv2RecurrentMTPForwardable }),
                ("captured-window MTP", { $0 is any CBv2RecurrentCaptureMTPForwardable }),
                ("prefill hidden", { $0 is any CBv2RecurrentPrefillHiddenForwardable }),
                ("MTP policy top two", { $0 is any CBv2MTPPolicyTopTwoProviding }),
                (
                    "media fallback",
                    { $0 is any CBv2RecurrentMTPRequestScopedMediaFallback }
                ),
            ]
            for (name, conforms) in checks where conforms(text) {
                #expect(conforms(wrapper), "wrapper lacks \(name)")
            }
            // The engine's runtime view of both, as the steppable adapter answers it.
            let textView = CBv2SteppableLanguageModelAdapter(vlm.languageModel)
            let wrapperView = CBv2SteppableLanguageModelAdapter(vlm)
            #expect(wrapperView.supportsRequestStatefulMTP == textView.supportsRequestStatefulMTP)
            #expect(wrapperView.mtpCaptureLayers == textView.mtpCaptureLayers)
            #expect(wrapperView.mtpTargetIdentity == textView.mtpTargetIdentity)
            #expect(wrapperView.cbv2Capabilities == textView.cbv2Capabilities)
            #expect(
                wrapperView.supportsRecurrentRequestScopedMediaFallback
                    == textView.supportsRecurrentRequestScopedMediaFallback)
            #expect(vlm.cbv2LayerKinds == vlm.languageModel.cbv2LayerKinds)
            #expect(vlm.cbv2RecurrentStateSpec == vlm.languageModel.cbv2RecurrentStateSpec)
            #expect(vlm.cbv2CausalPositionRequirement == .scalarCacheOffset)
            #expect(vlm.supportsCausalVisionPrefill && !vlm.supportsVisionSpanPrefill)
        }

        @Test func imageRowsReplacePlaceholdersWithoutTheEmbeddingNorm() throws {
            let model = try Inkling2TinyVLM.model()
            let image = try Inkling2TinyVLM.image(height: 75, width: 130)
            let id = Inkling2TinyVLM.imageTokenId
            let tokens = [5, 6] + Array(repeating: id, count: 8) + [9, 10]
            let embeddings = try model.inputEmbeddings(
                tokens: MLXArray(tokens.map(Int32.init)), image: image)
            let features = try model.perImageVisionFeatures(
                pixels: image.pixels, frames: image.frames)
            let text = model.languageModel.scaledInputEmbeddings(
                MLXArray(tokens.map(Int32.init)).reshaped(1, -1))
            #expect(embeddings.shape == [1, 12, 128])
            #expect(features.count == 1 && features[0].shape == [1, 8, 128])
            #expect(
                allClose(embeddings[0..., 2 ..< 10], features[0], rtol: 0, atol: 0).item(Bool.self))
            #expect(
                allClose(embeddings[0..., 0 ..< 2], text[0..., 0 ..< 2], rtol: 0, atol: 0).item(
                    Bool.self))
            #expect(
                allClose(embeddings[0..., 10...], text[0..., 10...], rtol: 0, atol: 0).item(
                    Bool.self))
        }

        @Test func legacyPrepareRejectsMediaAndTextOnlyCheckpointsRejectImages() throws {
            let model = try Inkling2TinyVLM.model()
            let image = try Inkling2TinyVLM.image(height: 40, width: 40)
            let input = LMInput(
                text: .init(tokens: MLXArray([Int32(5), Int32(Inkling2TinyVLM.imageTokenId)])),
                image: image)
            #expect(throws: VLMError.self) {
                try model.prepare(input, cache: [], state: nil, prefill: .init())
            }
            var textOnly = Inkling2TinyVLM.configuration
            textOnly["vision_config"] = nil
            let bare = Inkling2VLM(
                try SyntheticModel.configuration(Inkling2VLMConfiguration.self, textOnly))
            #expect(!bare.servesVision && !bare.supportsCausalVisionPrefill)
            #expect(bare.excludedCheckpointNamespaces == ["model.visual"])
            #expect(throws: VLMError.self) {
                try bare.perImageVisionFeatures(pixels: image.pixels, frames: image.frames)
            }
        }
    }
}
