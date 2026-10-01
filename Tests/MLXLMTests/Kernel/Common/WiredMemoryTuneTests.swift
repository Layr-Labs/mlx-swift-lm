import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

/// Encodes each Unicode scalar as its value modulo 64, so every token is in
/// the vocabulary of the tiny Llama model. `</s>` is token 3.
private struct ModuloTokenizer: MLXLMCommon.Tokenizer {
    /// When true, `encode` gives no tokens.
    var empty = false

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        empty ? [] : text.unicodeScalars.map { Int($0.value) % 64 }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { token == "</s>" ? 3 : nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { "</s>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(
            text: messages.compactMap { $0["content"] as? String }.joined(),
            addSpecialTokens: false)
    }
}

/// Makes one token for each character of the chat content.
private struct ModuloProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        let tokens = try ModuloTokenizer().applyChatTemplate(
            messages: DefaultMessageGenerator().generate(from: input), tools: nil,
            additionalContext: nil)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

/// A model with no parameters whose `prepare` gives logits, as a model
/// that embeds media in `prepare` does. It makes no cache.
private final class LogitsOnlyModel: Module, LanguageModel {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .logits(LMOutput(logits: MLXArray.zeros([1, 1, 8])))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, inputs.dim(-1), 8])
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

extension KernelTests {

    /// Tests of `WiredMemoryUtils.tune` with a tiny Llama model: hidden
    /// size 32, 2 layers, 4 query heads, 2 KV heads of size 8, MLP size 64,
    /// vocabulary 64, tied embeddings, float32.
    ///
    /// The weight and KV byte counts are exact and computed by hand:
    /// - weights: embedding 64 x 32 = 2048; each layer has q 32 x 32,
    ///   k and v 32 x 16, o 32 x 32, gate and up 32 x 64, down 64 x 32 and
    ///   two norms of 32, which is 9280; the final norm is 32. The total is
    ///   2048 + 2 x 9280 + 32 = 20640 values, 82560 bytes.
    /// - KV: 2 layers x (keys and values) x 2 heads x 8 values x 4 bytes is
    ///   256 bytes for each token.
    /// The peak memory is a process-wide value, so the tests check only
    /// bounds for it.
    @Suite(.serialized)
    struct WiredMemoryTuneTests {

        static let weightBytes = 82_560
        static let kvBytesPerToken = 256

        private func makeContext(tokenizer: ModuloTokenizer = ModuloTokenizer()) throws
            -> ModelContext
        {
            let configuration = try SyntheticModel.configuration(
                LlamaConfiguration.self,
                [
                    "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                    "rms_norm_eps": 1e-5, "vocab_size": 64, "tie_word_embeddings": true,
                ])
            let model = LlamaModel(configuration)
            SyntheticModel.randomize(model, seed: 3)
            return ModelContext(
                configuration: ModelConfiguration(id: "test/tiny-llama"), model: model,
                processor: ModuloProcessor(), tokenizer: tokenizer)
        }

        /// Checks the derived values. The weights and the KV arrays are
        /// alive during the prefill, so the peak is at least their sum. The
        /// workspace is the rest of the peak.
        private func checkBounds(
            _ measurement: WiredMemoryMeasurement,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let weights = measurement.weightBytes
            let kv = measurement.kvBytes
            let workspace = measurement.workspaceBytes
            #expect(measurement.peakActiveBytes >= weights + kv, sourceLocation: sourceLocation)
            #expect(
                workspace == measurement.peakActiveBytes - weights - kv,
                sourceLocation: sourceLocation)
            #expect(
                measurement.totalBytes == weights + kv + workspace, sourceLocation: sourceLocation)
        }

        /// The seed text " hello" gives 6 tokens. For 10 tokens the helper
        /// encodes " hello hello" (12 tokens) and keeps the first 10.
        @Test func tuneWithATokenCount() async throws {
            let context = try makeContext()
            let parameters = GenerateParameters(prefillStepSize: 4)
            let measurement = try await WiredMemoryUtils.tune(
                context: context, tokenCount: 10, parameters: parameters)

            #expect(measurement.weightBytes == Self.weightBytes)
            #expect(measurement.kvBytes == 10 * Self.kvBytesPerToken)
            #expect(measurement.tokenCount == 10)
            #expect(measurement.prefillStepSize == 4)
            checkBounds(measurement)
        }

        /// A tokenizer that gives no tokens makes a prompt of EOS tokens.
        @Test func tuneFillsWithTheEOSTokenWhenEncodingIsEmpty() async throws {
            let context = try makeContext(tokenizer: ModuloTokenizer(empty: true))
            let measurement = try await WiredMemoryUtils.tune(
                context: context, tokenCount: 5, parameters: GenerateParameters(),
                resetPeakMemory: false)

            #expect(measurement.kvBytes == 5 * Self.kvBytesPerToken)
            #expect(measurement.tokenCount == 5)
            #expect(measurement.prefillStepSize == 512)
            checkBounds(measurement)
        }

        @Test func tuneWithAPreparedInput() async throws {
            let context = try makeContext()
            let input = LMInput(tokens: MLXArray((0 ..< 7).map { Int32($0 * 5) }))
            let measurement = try await WiredMemoryUtils.tune(
                input: input, context: context, parameters: GenerateParameters())

            #expect(measurement.weightBytes == Self.weightBytes)
            #expect(measurement.kvBytes == 7 * Self.kvBytesPerToken)
            #expect(measurement.tokenCount == 7)
            checkBounds(measurement)
        }

        @Test func tuneWithAUserInputRunsTheProcessor() async throws {
            let context = try makeContext()
            let measurement = try await WiredMemoryUtils.tune(
                userInput: UserInput(prompt: "abc"), context: context,
                parameters: GenerateParameters())

            #expect(measurement.tokenCount == 3)
            #expect(measurement.kvBytes == 3 * Self.kvBytesPerToken)
            checkBounds(measurement)
        }

        /// A model whose `prepare` gives logits and that has no cache and no
        /// parameters has no weight and no KV bytes.
        @Test func tuneWithAModelThatPreparesLogits() async throws {
            let context = ModelContext(
                configuration: ModelConfiguration(id: "test/logits"), model: LogitsOnlyModel(),
                processor: ModuloProcessor(), tokenizer: ModuloTokenizer())
            let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
            let measurement = try await WiredMemoryUtils.tune(
                input: input, context: context, parameters: GenerateParameters())

            #expect(measurement.weightBytes == 0)
            #expect(measurement.kvBytes == 0)
            #expect(measurement.tokenCount == 3)
            checkBounds(measurement)
        }
    }
}
