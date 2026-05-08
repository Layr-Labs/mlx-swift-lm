// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXSpeculative
import Testing

@Suite("generateDFlash")
struct DFlashGenerateTests {
    private func tinyQwen3Config() throws -> Qwen3Configuration {
        let json = """
        {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128
        }
        """
        return try JSONDecoder.json5().decode(Qwen3Configuration.self, from: Data(json.utf8))
    }

    private func dflashConfig() throws -> DFlashConfiguration {
        let json = """
        {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "max_position_embeddings": 128,
            "block_size": 4,
            "num_target_layers": 3,
            "layer_types": ["full_attention", "full_attention"],
            "tie_word_embeddings": true,
            "dflash_config": {
                "target_layer_ids": [0, 1],
                "mask_token_id": 4
            }
        }
        """
        return try JSONDecoder.json5().decode(DFlashConfiguration.self, from: Data(json.utf8))
    }

    private func qwenContext() throws -> (ModelContext, DFlashDraftModel) {
        let target = Qwen3Model(try tinyQwen3Config())
        let drafter = DFlashDraftModel(config: try dflashConfig())
        try drafter.bind(target: target)
        eval(target, drafter)

        let tokenizer = TestTokenizer(vocabularySize: 128)
        let modelConfig = ModelConfiguration(
            id: "qwen3-dflash-test",
            defaultPrompt: "",
            extraEOSTokens: [],
            toolCallFormat: nil
        )
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: modelConfig,
            messageGenerator: DefaultMessageGenerator()
        )
        return (
            ModelContext(
                configuration: modelConfig,
                model: target,
                processor: processor,
                tokenizer: tokenizer
            ),
            drafter
        )
    }

    @Test func generateDFlashEmitsChunksAndInfo() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let (context, drafter) = try qwenContext()
            let input = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))
            let stream = try generateDFlash(
                input: input,
                parameters: GenerateParameters(maxTokens: 4, temperature: 0),
                target: context,
                drafter: drafter,
                blockSize: 3
            )

            var chunkCount = 0
            var info: GenerateCompletionInfo?
            for await generation in stream {
                switch generation {
                case .chunk:
                    chunkCount += 1
                case .info(let completion):
                    info = completion
                case .toolCall:
                    break
                }
            }

            #expect(chunkCount == 4)
            #expect(info?.generationTokenCount == 4)
        }
    }

    @Test func generateDFlashTokensEmitsRawTokensAndInfo() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let (context, drafter) = try qwenContext()
            let input = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))
            let stream = try generateDFlashTokens(
                input: input,
                parameters: GenerateParameters(maxTokens: 3, temperature: 0),
                target: context,
                drafter: drafter,
                blockSize: 3
            )

            var tokens: [Int] = []
            var info: GenerateCompletionInfo?
            for await generation in stream {
                switch generation {
                case .token(let token):
                    tokens.append(token)
                case .info(let completion):
                    info = completion
                }
            }

            #expect(tokens.count == 3)
            #expect(info?.generationTokenCount == 3)
        }
    }

    @Test func generateDFlashRejectsUnsupportedTarget() async throws {
        let tokenizer = TestTokenizer(vocabularySize: 128)
        let modelConfig = ModelConfiguration(id: "unsupported-dflash-target")
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: modelConfig,
            messageGenerator: DefaultMessageGenerator()
        )
        let context = ModelContext(
            configuration: modelConfig,
            model: UnsupportedLanguageModel(),
            processor: processor,
            tokenizer: tokenizer
        )
        let drafter = DFlashDraftModel(config: try dflashConfig())
        let input = LMInput(text: .init(tokens: MLXArray([Int32(1), 2, 3])))

        #expect(throws: DFlashError.self) {
            _ = try generateDFlash(
                input: input,
                parameters: GenerateParameters(maxTokens: 3, temperature: 0),
                target: context,
                drafter: drafter
            )
        }
    }
}

@Suite("DFlash real checkpoint smoke", .serialized)
struct DFlashRealCheckpointSmokeTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment[
        "MLX_SWIFT_LM_DFLASH_TARGET_DIR"] != nil
        && ProcessInfo.processInfo.environment[
            "MLX_SWIFT_LM_DFLASH_DRAFTER_DIR"] != nil))
    func loadsBindsAndGeneratesFromLocalDirectories() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard
            let targetPath = environment["MLX_SWIFT_LM_DFLASH_TARGET_DIR"],
            let drafterPath = environment["MLX_SWIFT_LM_DFLASH_DRAFTER_DIR"]
        else { return }

        let targetDirectory = URL(fileURLWithPath: targetPath)
        let drafterDirectory = URL(fileURLWithPath: drafterPath)
        let target = try await LLMModelFactory.shared.load(
            from: targetDirectory,
            using: DFlashSmokeTokenizerLoader()
        )
        let targetModel = try #require(target.model as? any DFlashTargetModel)
        let drafter = try await DFlashDraftModel.load(
            from: drafterDirectory,
            bindTo: targetModel
        )

        let promptTokens = parseSmokePromptTokens(
            environment["MLX_SWIFT_LM_DFLASH_PROMPT_TOKENS"])
        let input = LMInput(text: .init(tokens: MLXArray(promptTokens.map(Int32.init))))
        let stream = try generateDFlashTokens(
            input: input,
            parameters: GenerateParameters(maxTokens: 2, temperature: 0),
            target: target,
            drafter: drafter
        )

        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await generation in stream {
            switch generation {
            case .token(let token):
                tokens.append(token)
            case .info(let completion):
                info = completion
            }
        }

        #expect(tokens.count <= 2)
        #expect(info?.generationTokenCount == tokens.count)
    }

    private func parseSmokePromptTokens(_ value: String?) -> [Int] {
        let parsed = value?
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        if let parsed, !parsed.isEmpty {
            return parsed
        }
        return [1, 2, 3]
    }
}

private struct DFlashSmokeTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        TestTokenizer(vocabularySize: 300_000)
    }
}

private final class UnsupportedLanguageModel: Module, LanguageModel {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, inputs.size, 32])
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }
}
