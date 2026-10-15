import CoreImage
import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

extension KernelTests {
    @Suite(.serialized)
    struct PrismHadamardProcessorTests {
        private var processorData: Data {
            Data(
                #"""
                {"size":{"shortest_edge":65536,"longest_edge":16777216},
                 "patch_size":16,"temporal_patch_size":2,"merge_size":2,
                 "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
                 "processor_class":"Qwen3VLProcessor","image_processor_type":"Qwen2VLImageProcessorFast"}
                """#.utf8)
        }

        /// Metadata-only fixture: no model construction or fabricated weights.
        private var modelObject: [String: Any] {
            [
                "schema_version": 2, "model_type": "prism_hadamard_qwen35",
                "base_model_type": "qwen3_5", "gdn_activation_layout": "grouped",
                "tensor_namespace": "mlx-vlm-qwen3_5", "hadamard_config": "hadamard.json",
                "components": ["text": true, "vision": true, "mtp": false],
                "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
                "modules": [
                    [
                        "path": "model.embed_tokens", "block": 4096, "embedding": true,
                        "dtype": "float16",
                    ],
                    ["path": "lm_head", "block": 4096, "embedding": false, "dtype": "float16"],
                ],
                "text_config": [
                    "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 248320,
                    "intermediate_size": 12288, "num_attention_heads": 16,
                    "num_key_value_heads": 4, "head_dim": 256, "full_attention_interval": 4,
                    "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                    "mtp_num_hidden_layers": 0,
                ],
                "vision_config": [
                    "model_type": "qwen3_5", "depth": 27, "hidden_size": 1152, "num_heads": 16,
                    "intermediate_size": 4304, "out_hidden_size": 4096,
                    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "in_channels": 3, "num_position_embeddings": 2304,
                    "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [],
                ],
            ]
        }

        private func profile(_ object: [String: Any]) throws -> PrismHadamardImageProfile {
            try PrismHadamardImageProfile.resolve(
                configurationData: JSONSerialization.data(withJSONObject: object))
        }

        private func route(_ object: [String: Any]) throws -> String {
            VLMModelFactory.processorType(
                modelType: "prism_hadamard_qwen35",
                declaredClass: "Qwen3VLProcessor", prismImageProfile: try profile(object))
        }

        private func processor(_ object: [String: Any]? = nil) async throws
            -> any UserInputProcessor
        {
            try await VLMProcessorTypeRegistry.shared.createModel(
                configuration: processorData,
                processorType: route(object ?? modelObject), tokenizer: PrismImageTokenizer())
        }

        private func image(width: Int, height: Int) -> UserInput.Image {
            .ciImage(
                CIImage(color: .red).cropped(
                    to: CGRect(
                        x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))))
        }

        @Test
        func testOnlyValidated9BProfileSelectsCheckpointProcessor() throws {
            #expect((try profile(modelObject)) == (.checkpoint9B))
            #expect((try route(modelObject)) == ("PrismHadamardImageProcessor"))
            for type in ["qwen3_5", "qwen3_vl", "qwen3_5_moe"] {
                #expect(
                    (VLMModelFactory.processorType(
                        modelType: type,
                        declaredClass: "Qwen3VLProcessor", prismImageProfile: .checkpoint9B))
                        == ("Qwen3VLProcessor"))
            }
            #expect(
                (VLMModelFactory.processorType(
                    modelType: "qwen4_exp",
                    declaredClass: "Qwen3VLProcessor")) == ("Qwen4ExpProcessor"))
            // A block size is not permission to admit a different model geometry.
            for (field, replacement) in [
                ("hidden_size", 2560), ("num_hidden_layers", 64),
                ("vocab_size", 151936), ("num_key_value_heads", 8),
            ] {
                var changed = modelObject
                var text = changed["text_config"] as! [String: Any]
                text[field] = replacement
                changed["text_config"] = text
                #expect((try profile(changed)) == (.legacy), Comment(rawValue: field))
                #expect((try route(changed)) == ("Qwen3VLProcessor"), Comment(rawValue: field))
            }
            var legacyPack = modelObject
            var modules = legacyPack["modules"] as! [[String: Any]]
            for i in modules.indices { modules[i]["block"] = 1024 }
            legacyPack["modules"] = modules
            #expect(
                (try route(legacyPack)) == ("Qwen3VLProcessor"),
                Comment(rawValue: "27B packed contract stays legacy"))
            var invalid = modelObject
            invalid["components"] = ["text": true, "vision": true, "mtp": true]
            #expect(
                throws: (any Error).self,
                Comment(rawValue: "Processor selection must retain strict packing validation")
            ) { try profile(invalid) }
        }

        @Test
        func testSmallImagesUseCheckpointMinimumAndKeepImageOrder() async throws {
            let processor = try await processor()
            let input = UserInput(
                prompt: "Describe the images",
                images: [
                    image(width: 64, height: 64), image(width: 128, height: 64),
                ])
            let prepared = try await processor.prepare(input: input)
            let grids = try #require(prepared.image?.frames)
            #expect((grids.count) == (2))
            #expect((grids[0].h) == (16))  // 256 / patch_size 16
            #expect((grids[0].w) == (16))
            #expect((grids[0].product / 4) == (64))
            // Reference ceil-to-32 resize of 128x64 at the 65,536-pixel minimum.
            #expect((grids[1].h) == (12))
            #expect((grids[1].w) == (24))
            #expect((grids[1].product / 4) == (72))
            #expect((prepared.text.tokens.size) == (64 + 72 + 4))
        }

        @Test
        func testLegacyImageSizingAndResolutionCapRemainUnchanged() async throws {
            let legacy = try await VLMProcessorTypeRegistry.shared.createModel(
                configuration: processorData,
                processorType: "Qwen3VLProcessor", tokenizer: PrismImageTokenizer())
            let small = try await legacy.prepare(
                input: UserInput(
                    prompt: "Describe",
                    images: [image(width: 64, height: 64)]))
            let smallGrid = try #require(small.image?.frames?.first)
            #expect((smallGrid.h) == (4))
            #expect((smallGrid.w) == (4))
            let input = UserInput(prompt: "Describe", images: [image(width: 2240, height: 1120)])
            let legacyLarge = try await legacy.prepare(input: input)
            #expect((try #require(legacyLarge.image?.frames?.first).product / 4) <= (1280))
            let checkpoint = try await processor()
            let checkpointLarge = try await checkpoint.prepare(input: input)
            let grid = try #require(checkpointLarge.image?.frames?.first)
            #expect(
                (grid.product / 4) <= (1280),
                Comment(
                    rawValue: "Checkpoint minimum does not widen the default serving pixel budget"))

            var explicit = input
            explicit.processing.maxPixels = 16_777_216
            let uncapped = try await checkpoint.prepare(input: explicit)
            let explicitGrid = try #require(uncapped.image?.frames?.first)
            #expect((explicitGrid.h) == (70))
            #expect((explicitGrid.w) == (140))
            #expect(
                (explicitGrid.product / 4) == (2450),
                Comment(
                    rawValue: "Explicit image budgets remain subject to independent tower admission"
                ))
        }

        @Test
        func testExplicitProcessingBoundsRemainEffective() async throws {
            let processor = try await processor()
            var small = UserInput(prompt: "Describe", images: [image(width: 64, height: 64)])
            small.processing.minPixels = 4096
            small.processing.maxPixels = 4096
            let preparedSmall = try await processor.prepare(input: small)
            let smallGrid = try #require(preparedSmall.image?.frames?.first)
            #expect((smallGrid.h) == (4))
            #expect((smallGrid.w) == (4))

            var large = UserInput(prompt: "Describe", images: [image(width: 1024, height: 1024)])
            large.processing.maxPixels = 262144
            let preparedLarge = try await processor.prepare(input: large)
            let largeGrid = try #require(preparedLarge.image?.frames?.first)
            #expect((largeGrid.h) == (32))
            #expect((largeGrid.w) == (32))
            large.processing.maxPixels = 0
            do {
                _ = try await processor.prepare(input: large)
                Issue.record(
                    Comment(
                        rawValue: "Invalid explicit pixel budget must fail before preprocessing"))
            } catch {}
        }

        @Test
        func testMalformedCheckpointBoundsAndNormalizationFailClosed() throws {
            for (field, replacement): (String, Any) in [
                ("size", ["longest_edge": 16_777_216]),
                ("size", ["shortest_edge": 65536]),
                ("size", ["shortest_edge": 0, "longest_edge": 16_777_216]),
                ("size", ["shortest_edge": 65536, "longest_edge": 4096]),
                ("max_pixels", 32768), ("patch_size", 0), ("merge_size", Int.max),
                ("image_std", [0.5, 0.0, 0.5]), ("image_mean", [0.5]),
                ("image_processor_type", "UnrelatedProcessor"),
            ] {
                var object = try JSONSerialization.jsonObject(with: processorData) as! [String: Any]
                object[field] = replacement
                #expect(throws: (any Error).self, Comment(rawValue: field)) {
                    try JSONDecoder().decode(
                        QwenCheckpointImageConfiguration.self,
                        from: JSONSerialization.data(withJSONObject: object))
                }
            }
        }

        @Test
        func testMaxOnlyBelowCheckpointMinimumFailsClosed() async throws {
            for type in ["PrismHadamardImageProcessor", "Qwen4ExpProcessor"] {
                let processor = try await VLMProcessorTypeRegistry.shared.createModel(
                    configuration: processorData, processorType: type,
                    tokenizer: PrismImageTokenizer())
                var input = UserInput(prompt: "Describe", images: [image(width: 64, height: 64)])
                input.processing.maxPixels = 4096
                do {
                    _ = try await processor.prepare(input: input)
                    Issue.record(
                        Comment(
                            rawValue:
                                "Max-only override below checkpoint minimum must fail: \(type)"))
                } catch {
                    #expect(
                        (error as? Qwen4ExpMediaGeometry.Failure) == (.invalidBudget),
                        Comment(rawValue: type))
                }
            }
        }

        @Test
        func testCheckpointMinimumCannotExceedDefaultServingBudget() async throws {
            var object = try JSONSerialization.jsonObject(with: processorData) as! [String: Any]
            object["size"] = ["shortest_edge": 2_097_152, "longest_edge": 16_777_216]
            do {
                _ = try await VLMProcessorTypeRegistry.shared.createModel(
                    configuration: JSONSerialization.data(withJSONObject: object),
                    processorType: "PrismHadamardImageProcessor", tokenizer: PrismImageTokenizer())
                Issue.record(
                    Comment(
                        rawValue:
                            "An incompatible checkpoint minimum must not silently widen serving policy"
                    ))
            } catch {}
        }
    }
}

private struct PrismImageTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [1] + Array(repeating: 7, count: text.components(separatedBy: "<|image_pad|>").count - 1)
            + [2]
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let imageCount = messages.flatMap { $0["content"] as? [[String: String]] ?? [] }
            .filter { $0["type"] == "image" }.count
        return (0 ..< imageCount).flatMap { _ in [1, 7, 2] }
    }
}
