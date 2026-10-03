import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of `WiredMemoryUtils.tune` and `makePromptCache` with a tiny
    /// Llama model and the test tokenizer.
    ///
    /// The weight and cache sizes are exact byte counts, computed by hand
    /// from the model shapes. The peak memory is global and depends on the
    /// other tests, so the tests check only its bounds. No test resets the
    /// global peak memory, so that the tests do not change the measurements
    /// of other tests.
    @Suite
    struct WiredMemoryMeasurementTests {

        /// Hidden 32, 2 layers, 4 query heads over 2 key heads (head size 8),
        /// vocabulary 128 (the test tokenizer gives IDs below 100).
        static func context(
            hidden: Int = 32, heads: Int = 4, kvHeads: Int = 2
        ) -> ModelContext {
            let configuration = LlamaConfiguration(
                hiddenSize: hidden, hiddenLayers: 2, intermediateSize: 48,
                attentionHeads: heads, rmsNormEps: 1e-5, vocabularySize: 128, kvHeads: kvHeads)
            let model = LlamaModel(configuration)
            SyntheticModel.randomize(model, seed: 1)
            let processor = TestInputProcessor()
            return ModelContext(
                configuration: processor.configuration, model: model, processor: processor,
                tokenizer: processor.tokenizer)
        }

        /// Float32 bytes of the tiny model with hidden 32: the tied
        /// embedding (128 x 32), the final norm (32), and per layer the
        /// projections q (32 x 32), k and v (16 x 32 each), o (32 x 32),
        /// gate and up (48 x 32 each), down (32 x 48) and two norms (32 each).
        static let weightBytes = 4 * (128 * 32 + 32 + 2 * (2 * 1024 + 2 * 512 + 3 * 1536 + 64))

        static func checkBounds(_ measurement: WiredMemoryMeasurement) {
            #expect(measurement.workspaceBytes >= 0)
            #expect(
                measurement.workspaceBytes
                    == max(
                        0,
                        measurement.peakActiveBytes - measurement.weightBytes
                            - measurement.kvBytes))
            #expect(
                measurement.totalBytes
                    == measurement.weightBytes + measurement.kvBytes + measurement.workspaceBytes)
        }

        /// 20 tokens with prefill steps of 8: the model prepares 16 tokens in
        /// two chunks, and `tune` runs the last 4. Each layer then keeps
        /// float32 keys and values of shape [1, 2, 20, 8].
        @Test func tuneFromATokenCountMeasuresTheWeightsAndTheCache() async throws {
            let context = Self.context()
            let measurement = try await WiredMemoryUtils.tune(
                context: context, tokenCount: 20,
                parameters: GenerateParameters(prefillStepSize: 8), resetPeakMemory: false)
            #expect(measurement.tokenCount == 20)
            #expect(measurement.prefillStepSize == 8)
            #expect(measurement.weightBytes == Self.weightBytes)
            #expect(measurement.kvBytes == 2 * 2 * (2 * 20 * 8) * 4)
            Self.checkBounds(measurement)
        }

        /// The processor of the test context gives 8 tokens.
        @Test func tuneFromAUserInputUsesTheProcessor() async throws {
            let context = Self.context()
            let measurement = try await WiredMemoryUtils.tune(
                userInput: UserInput(prompt: "hello"), context: context,
                parameters: GenerateParameters(), resetPeakMemory: false)
            #expect(measurement.tokenCount == 8)
            #expect(measurement.prefillStepSize == 512)
            #expect(measurement.kvBytes == 2 * 2 * (2 * 8 * 8) * 4)
            Self.checkBounds(measurement)
        }

        /// With `kvBits`, the cache is quantized after the prefill, so the
        /// measured cache is the quantized one. Head size 32 (hidden 64, 2
        /// heads over 1 key head) is a multiple of the group size 32.
        @Test func tuneFromAPreparedInputMeasuresTheQuantizedCache() async throws {
            let context = Self.context(hidden: 64, heads: 2, kvHeads: 1)
            let tokens = MLXArray((0 ..< 6).map { Int32($0) })
            let measurement = try await WiredMemoryUtils.tune(
                input: LMInput(tokens: tokens), context: context,
                parameters: GenerateParameters(kvBits: 8, kvGroupSize: 32),
                resetPeakMemory: false)
            #expect(measurement.tokenCount == 6)
            // Per layer, keys and values each: 6 rows of 8 packed uint32
            // words (32 values of 8 bits) plus one float32 scale and one
            // float32 bias.
            let quantizedBytes = 2 * 2 * 6 * (8 * 4 + 4 + 4)
            #expect(measurement.kvBytes == quantizedBytes)
            #expect(measurement.kvBytes < 2 * 2 * (1 * 6 * 32) * 4, "smaller than float32")
            Self.checkBounds(measurement)
        }

        @Test func makePromptCacheFollowsTheMaximumCacheSize() {
            let model = Self.context().model
            let plain = makePromptCache(model: model, parameters: nil)
            #expect(plain.count == 2)
            #expect(plain.allSatisfy { type(of: $0) == KVCacheSimple.self })

            let rotating = makePromptCache(model: model, maxKVSize: 16)
            #expect(rotating.count == 2)
            #expect(rotating.allSatisfy { $0 is RotatingKVCache })
            #expect(rotating.allSatisfy { $0.maxSize == 16 })
            #expect(rotating[0].metaState[0] == "4", "the cache keeps the first 4 tokens")

            let fromParameters = makePromptCache(
                model: model, parameters: GenerateParameters(maxKVSize: 8))
            #expect(fromParameters.allSatisfy { $0.maxSize == 8 })
        }
    }
}
