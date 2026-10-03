import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// One tiny vision-language model for `VisionModelForwardPassTests`.
struct VisionCase: Sendable, CustomTestStringConvertible {
    /// The checks of `VisionModelForwardPassTests`.
    enum Check: Sendable {
        case shape, image, decode, loading

        /// Text in the comment of each expectation that a defect in the
        /// model can fail. `run` records only these as the known issue.
        /// The control expectation of the image check, that another image
        /// changes the logits, fails the test also when the check has a
        /// known issue.
        var failingExpectations: [String] {
            switch self {
            case .shape: ["logits shape", "not finite"]
            case .image: ["same image"]
            case .decode: ["differs by"]
            case .loading: ["loaded logits"]
            }
        }
    }

    let name: String
    let vocabularySize: Int
    /// Builds the model with seeded random weights.
    let make: @Sendable (_ seed: UInt64) throws -> any LanguageModel
    /// The image token and how many of them one test image needs.
    let imageToken: Int
    let imageTokens: Int
    /// A synthetic image, and its `frames` value.
    let pixels: @Sendable (_ seed: UInt64) -> MLXArray
    let frames: [THW]?
    /// Converts the model's parameters to the checkpoint layout (for
    /// example PyTorch convolution layout).
    let checkpoint: @Sendable ([String: MLXArray]) -> [String: MLXArray]
    /// Checks that fail because of a known production defect, with the
    /// defect. Only the expectations in `Check.failingExpectations` are the
    /// known issue; any other failure in the check fails the test.
    let knownIssues: [Check: String]

    init(
        _ name: String, vocabularySize: Int = 64, imageToken: Int = 60, imageTokens: Int = 4,
        frames: [THW]? = nil, knownIssues: [Check: String] = [:],
        checkpoint: @escaping @Sendable ([String: MLXArray]) -> [String: MLXArray] = { $0 },
        pixels: @escaping @Sendable (_ seed: UInt64) -> MLXArray,
        make: @escaping @Sendable (_ seed: UInt64) throws -> any LanguageModel
    ) {
        self.name = name
        self.vocabularySize = vocabularySize
        self.make = make
        self.imageToken = imageToken
        self.imageTokens = imageTokens
        self.pixels = pixels
        self.frames = frames
        self.checkpoint = checkpoint
        self.knownIssues = knownIssues
    }

    var testDescription: String { name }

    /// Text, the image tokens, then more text.
    var prompt: [Int] {
        [5, 7] + Array(repeating: imageToken, count: imageTokens) + [9, 11, 13]
    }

    /// Runs `body`, inside `withKnownIssue` when `check` has a known
    /// defect. Only a failure of the expectations in
    /// `Check.failingExpectations` is the known issue. For `.loading`, a
    /// thrown error of the load is also the known issue, because a wrong
    /// `sanitize(weights:)` makes the strict update throw.
    func run(_ check: Check, _ body: () throws -> Void) rethrows {
        if let issue = knownIssues[check] {
            try withKnownIssue(Comment(rawValue: "\(name): \(issue)")) {
                try body()
            } matching: { recorded in
                (check == .loading && recorded.error != nil)
                    || recorded.isFailedExpectation(check.failingExpectations)
            }
        } else {
            try body()
        }
    }

    /// Runs `prepare` on `prompt` with the image and returns the logits of
    /// the prompt.
    func prefill(
        _ model: any LanguageModel, prompt: [Int], pixels: MLXArray, cache: [KVCache]
    ) throws -> MLXArray {
        let tokens = SyntheticModel.batch([prompt])
        let input = LMInput(
            text: .init(tokens: tokens, mask: MLXArray.ones(tokens.shape, dtype: .int32)),
            image: .init(pixels: pixels, frames: frames))
        guard case .logits(let output) = try model.prepare(input, cache: cache, windowSize: nil)
        else {
            Issue.record("\(name): prepare must return logits")
            return MLXArray.zeros([1])
        }
        eval(output.logits)
        return output.logits
    }

    /// Random normal pixels of `shape`.
    static func random(_ shape: [Int]) -> @Sendable (UInt64) -> MLXArray {
        { seed in
            let x = MLXRandom.normal(shape, key: MLXRandom.key(seed))
            eval(x)
            return x
        }
    }

    /// Moves the channel axis of convolution weights from last (MLX) to
    /// second (PyTorch) for the keys that end with one of `suffixes`.
    static func pytorchConvolutions(_ suffixes: [String])
        -> @Sendable ([String: MLXArray]) -> [String: MLXArray]
    {
        { weights in
            var result = weights
            for (key, value) in weights where suffixes.contains(where: { key.hasSuffix($0) }) {
                result[key] = value.movedAxis(source: -1, destination: 1)
            }
            return result
        }
    }
}

extension KernelTests {

    /// Forward-pass tests of vision-language models with tiny random models
    /// and small synthetic images (random pixels, no media files).
    ///
    /// Each model has a text model with hidden size 32 and a vision tower
    /// with 1 or 2 layers. The test image gives 4 image tokens.
    ///
    /// The Qwen-style models take flattened patches `[t * h * w, C * T * p *
    /// p]` with a 4 x 4 patch grid. They use patch size 4, so that the
    /// checkpoint convolution `[O, 3, 2, 4, 4]` is not mistaken for the MLX
    /// layout by the shape test of `sanitize(weights:)`.
    @Suite
    struct VisionModelForwardPassTests {

        static let cases: [VisionCase] = [
            VisionCase(
                "Qwen2VL", frames: [THW(1, 4, 4)],
                checkpoint: VisionCase.pytorchConvolutions(["patch_embed.proj.weight"]),
                pixels: VisionCase.random([16, 96])
            ) { seed in
                try vision(Qwen2VLConfiguration.self, seed: seed, Qwen2VL.init) {
                    [
                        "model_type": "qwen2_vl", "hidden_size": 32, "num_hidden_layers": 2,
                        "intermediate_size": 48, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "vocab_size": 64, "image_token_id": 60,
                        "video_token_id": 61,
                        "rope_scaling": ["type": "mrope", "mrope_section": [1, 1, 2]],
                        "vision_config": [
                            "depth": 1, "embed_dim": 32, "hidden_size": 32, "num_heads": 4,
                            "patch_size": 4, "mlp_ratio": 2.0, "spatial_patch_size": 2,
                            "spatial_merge_size": 2, "temporal_patch_size": 2,
                        ],
                    ]
                }
            },
            VisionCase(
                "Qwen25VL", frames: [THW(1, 4, 4)],
                checkpoint: VisionCase.pytorchConvolutions(["patch_embed.proj.weight"]),
                pixels: VisionCase.random([16, 96])
            ) { seed in
                try vision(Qwen25VLConfiguration.self, seed: seed, Qwen25VL.init) {
                    [
                        "model_type": "qwen2_5_vl", "hidden_size": 32, "num_hidden_layers": 2,
                        "intermediate_size": 48, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "vocab_size": 64, "image_token_id": 60,
                        "video_token_id": 61, "vision_start_token_id": 57,
                        "vision_end_token_id": 58, "vision_token_id": 59,
                        "sliding_window": 16, "use_sliding_window": false,
                        "max_window_layers": 2,
                        "rope_scaling": ["type": "mrope", "mrope_section": [1, 1, 2]],
                        "vision_config": [
                            "depth": 2, "hidden_size": 32, "intermediate_size": 48,
                            "out_hidden_size": 32, "num_heads": 4, "patch_size": 4,
                            "spatial_patch_size": 2, "spatial_merge_size": 2,
                            "temporal_patch_size": 2, "window_size": 8,
                            "fullatt_block_indexes": [1], "tokens_per_second": 2,
                        ],
                    ]
                }
            },
            VisionCase(
                "GlmOcr", frames: [THW(1, 4, 4)],
                checkpoint: VisionCase.pytorchConvolutions(["patch_embed.proj.weight"]),
                pixels: VisionCase.random([16, 96])
            ) { seed in
                try vision(GlmOcrConfiguration.self, seed: seed, GlmOcr.init) {
                    [
                        "model_type": "glm_ocr", "vocab_size": 64, "image_token_id": 60,
                        "video_token_id": 61, "image_start_token_id": 58, "hidden_size": 32,
                        "text_config": [
                            "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                            "vocab_size": 64,
                            "rope_parameters": [
                                "mrope_section": [1, 1, 2], "partial_rotary_factor": 1.0,
                                "rope_theta": 10000,
                            ],
                        ],
                        "vision_config": [
                            "depth": 1, "hidden_size": 32, "intermediate_size": 48,
                            "num_heads": 4, "patch_size": 4, "out_hidden_size": 32,
                            "spatial_merge_size": 2, "temporal_patch_size": 2,
                        ],
                    ]
                }
            },
            VisionCase(
                "LFM2VL", frames: [THW(1, 4, 4)], pixels: VisionCase.random([1, 16, 12])
            ) { seed in
                try vision(LFM2VLConfiguration.self, seed: seed, LFM2VL.init) {
                    [
                        "model_type": "lfm2_vl", "downsample_factor": 2, "image_token_id": 60,
                        "projector_hidden_size": 32,
                        "text_config": [
                            "model_type": "lfm2", "hidden_size": 32, "num_hidden_layers": 2,
                            "num_attention_heads": 4, "num_key_value_heads": 2,
                            "vocab_size": 64, "block_ff_dim": 48,
                            "block_auto_adjust_ff_dim": false,
                            "layer_types": ["conv", "full_attention"],
                        ],
                        "vision_config": [
                            "model_type": "siglip2_vision_model", "hidden_size": 32,
                            "intermediate_size": 48, "num_hidden_layers": 1,
                            "num_attention_heads": 4, "patch_size": 2, "num_patches": 16,
                        ],
                    ]
                }
            },
            VisionCase("Idefics3", pixels: VisionCase.random([1, 8, 8, 3])) { seed in
                try vision(Idefics3Configuration.self, seed: seed, Idefics3.init) {
                    [
                        "model_type": "idefics3", "vocab_size": 64, "scale_factor": 2,
                        "image_token_id": 60,
                        "text_config": [
                            "model_type": "llama", "hidden_size": 32, "intermediate_size": 48,
                            "num_attention_heads": 4, "rms_norm_eps": 1e-6, "vocab_size": 64,
                            "num_key_value_heads": 2, "rope_theta": 10000,
                            "num_hidden_layers": 2,
                        ],
                        "vision_config": [
                            "model_type": "idefics3_vision", "hidden_size": 32,
                            "num_attention_heads": 4, "patch_size": 2, "image_size": 8,
                            "num_hidden_layers": 1, "intermediate_size": 48,
                        ],
                    ]
                }
            },
            VisionCase(
                "Mistral3", frames: [THW(1, 8, 8)], pixels: VisionCase.random([1, 3, 8, 8])
            ) { seed in
                try vision(Mistral3VLMConfiguration.self, seed: seed, Mistral3VLM.init) {
                    [
                        "model_type": "mistral3", "image_token_index": 60, "vocab_size": 64,
                        "spatial_merge_size": 2,
                        "text_config": [
                            "model_type": "ministral3", "hidden_size": 32,
                            "num_hidden_layers": 2, "intermediate_size": 48,
                            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                            "rms_norm_eps": 1e-6, "vocab_size": 64,
                            "rope_parameters": ["rope_type": "default", "rope_theta": 10000],
                        ],
                        "vision_config": pixtralVision,
                    ]
                }
            },
            VisionCase("Pixtral", pixels: VisionCase.random([1, 3, 4, 4])) { seed in
                try vision(PixtralConfiguration.self, seed: seed, PixtralVLM.init) {
                    [
                        "model_type": "pixtral", "image_token_index": 60, "vocab_size": 64,
                        "text_config": [
                            "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                            "intermediate_size": 48, "num_attention_heads": 4,
                            "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6,
                            "vocab_size": 64, "rope_theta": 10000,
                        ],
                        "vision_config": pixtralVision,
                    ]
                }
            },
            VisionCase(
                "Gemma3", vocabularySize: 262_208, imageToken: 262_144,
                checkpoint: VisionCase.pytorchConvolutions(["patch_embedding.weight"]),
                pixels: VisionCase.random([1, 3, 8, 8])
            ) { seed in
                try vision(Gemma3Configuration.self, seed: seed, Gemma3.init) {
                    [
                        "model_type": "gemma3", "mm_tokens_per_image": 4,
                        "text_config": [
                            "model_type": "gemma3_text", "hidden_size": 32,
                            "num_hidden_layers": 6, "intermediate_size": 48,
                            "sliding_window": 4, "num_attention_heads": 2,
                            "num_key_value_heads": 1, "head_dim": 16,
                            "query_pre_attn_scalar": 16,
                        ],
                        "vision_config": [
                            "model_type": "siglip_vision_model", "num_hidden_layers": 1,
                            "hidden_size": 32, "intermediate_size": 48,
                            "num_attention_heads": 4, "patch_size": 2, "image_size": 8,
                        ],
                    ]
                }
            },
            VisionCase(
                "FastVLM", imageToken: -200, imageTokens: 1,
                checkpoint: { weights in
                    // The original layout: `mm_projector.N`, `layer_scale_N` and
                    // `network.N.M` without `layers`.
                    var result: [String: MLXArray] = [:]
                    for (key, value) in weights {
                        var name = key.replacingOccurrences(
                            of: "mm_projector.layers.", with: "mm_projector.")
                        name = name.replacingOccurrences(of: "layerScale", with: "layer_scale_")
                        name = name.replacingOccurrences(
                            of: #"network\.(\d+)\.layers\.(\d+)\."#, with: "network.$1.$2.",
                            options: .regularExpression)
                        result[name] = value
                    }
                    return result
                },
                pixels: VisionCase.random([1, 3, 32, 32])
            ) { seed in
                let model = try vision(FastVLMConfiguration.self, seed: seed, FastVLM.init) {
                    [
                        "model_type": "fastvlm", "hidden_size": 32, "num_hidden_layers": 2,
                        "intermediate_size": 48, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "vocab_size": 64, "eos_token_id": 1,
                        "mm_projector_type": "mlp2x_gelu", "mm_hidden_size": 64,
                        "tokenizer_model_max_length": 256, "tokenizer_padding_side": "right",
                        "image_token_index": -200,
                        "vision_config": [
                            "cls_ratio": 2.0, "down_patch_size": 3, "down_stride": 2,
                            "downsamples": [true, true], "embed_dims": [16, 32],
                            "hidden_size": 32, "image_size": 32, "intermediate_size": 64,
                            "layers": [1, 1], "layer_scale_init_value": 1e-5,
                            "mlp_ratios": [2, 2], "num_classes": 0, "patch_size": 16,
                            "pos_embs_shapes": [NSNull(), NSNull()], "projection_dim": 32,
                            "repmixer_kernel_size": 3, "token_mixers": ["repmixer", "repmixer"],
                        ] as [String: Any],
                    ]
                }
                // BatchNorm must use its running statistics, and a running
                // variance must be positive.
                model.train(false)
                var variances: [(String, MLXArray)] = []
                for (key, value) in SyntheticModel.flatParameters(model)
                where key.hasSuffix("running_var") {
                    variances.append((key, abs(value) + 1))
                }
                model.update(parameters: ModuleParameters.unflattened(variances))
                eval(model)
                return model
            },
        ]

        static var pixtralVision: [String: Any] {
            [
                "model_type": "pixtral", "hidden_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 4, "intermediate_size": 48, "patch_size": 2,
                "image_size": 8, "head_dim": 8,
            ]
        }

        static func vision<C: Decodable>(
            _ type: C.Type, seed: UInt64, _ initializer: (C) -> any LanguageModel,
            _ configuration: () -> [String: Any]
        ) throws -> any LanguageModel {
            try ModelCase.build(type, configuration(), seed: seed, initializer)
        }

        // Tolerance of the float32 comparisons: the paths differ in the order
        // of the attention sums; differences are near 1e-6 for logits of size
        // 1 to 5.
        static let tolerance: Float = 1e-4

        @Test(arguments: cases) func prefillLogitsHaveTheExpectedShapeAndAreFinite(_ c: VisionCase)
            throws
        {
            try c.run(.shape) {
                let model = try c.make(1)
                let logits = try c.prefill(
                    model, prompt: c.prompt, pixels: c.pixels(1),
                    cache: model.newCache(parameters: nil))
                #expect(logits.dim(0) == 1, "\(c.name): logits shape, batch size")
                // Some models return the logits of every prompt position,
                // others only of the last one.
                #expect(
                    [1, c.prompt.count].contains(logits.dim(1)),
                    "\(c.name): logits shape, prompt positions")
                #expect(
                    logits.dim(2) >= c.vocabularySize, "\(c.name): logits shape, vocabulary")
                #expect(
                    isFinite(logits).all().item(Bool.self), "\(c.name): a logit is not finite")
            }
        }

        /// The same image gives the same logits; another image gives other
        /// logits at the last position, which comes after the image tokens.
        @Test(arguments: cases) func theImageChangesTheLogitsAfterIt(_ c: VisionCase) throws {
            try c.run(.image) {
                let model = try c.make(1)
                func run(_ seed: UInt64) throws -> MLXArray {
                    try c.prefill(
                        model, prompt: c.prompt, pixels: c.pixels(seed),
                        cache: model.newCache(parameters: nil))
                }
                let a = try run(1)
                let again = try run(1)
                let b = try run(2)
                #expect(
                    SyntheticModel.maxAbsDifference(a, again) == 0,
                    "\(c.name): same image, same logits")
                // The last position comes after the image.
                #expect(
                    SyntheticModel.maxAbsDifference(a[0..., -1], b[0..., -1]) > 1e-3,
                    "\(c.name): another image must change the logits")
            }
        }

        /// After the prompt, a decode step with the cache gives the logits
        /// of the last position of the same prompt with that token appended.
        @Test(arguments: cases) func decodeAfterThePromptMatchesALongerPrompt(_ c: VisionCase)
            throws
        {
            try c.run(.decode) {
                let model = try c.make(1)
                let pixels = c.pixels(1)
                let cache = model.newCache(parameters: nil)
                _ = try c.prefill(model, prompt: c.prompt, pixels: pixels, cache: cache)
                let step = model(SyntheticModel.batch([[17]]), cache: cache)
                eval(step)
                let longer = try c.prefill(
                    model, prompt: c.prompt + [17], pixels: pixels,
                    cache: model.newCache(parameters: nil))
                let difference = SyntheticModel.maxAbsDifference(
                    step[0..., -1], longer[0..., -1])
                #expect(difference <= Self.tolerance, "\(c.name): differs by \(difference)")
            }
        }

        /// Loads the model's parameters in the checkpoint layout through
        /// `loadWeights` and compares the logits. Only the load and the
        /// loaded logits can be a known issue.
        @Test(arguments: cases) func loaderAcceptsACheckpoint(_ c: VisionCase) throws {
            let reference = try c.make(5)
            let checkpoint = c.checkpoint(SyntheticModel.flatParameters(reference))
            let loaded = try c.make(6)
            let pixels = c.pixels(1)
            try c.run(.loading) {
                try SyntheticModel.load(checkpoint, into: loaded)
                let a = try c.prefill(
                    reference, prompt: c.prompt, pixels: pixels,
                    cache: reference.newCache(parameters: nil))
                let b = try c.prefill(
                    loaded, prompt: c.prompt, pixels: pixels,
                    cache: loaded.newCache(parameters: nil))
                #expect(
                    SyntheticModel.maxAbsDifference(a, b) == 0, "\(c.name): loaded logits")
            }
        }
    }
}
