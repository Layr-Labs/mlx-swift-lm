import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXEmbedders
@testable import MLXLMServer

/// An embedding model whose hidden state for token `t` is row `t` of a
/// fixed table: `[sin(t), cos(t), sin(2t), cos(3t)]`. It ignores the
/// attention mask, so a padded position gets the hidden state of the pad
/// token.
private final class TableEmbeddingModel: Module, EmbeddingModel {
    let vocabularySize = 128
    let table: MLXArray

    override init() {
        let rows = (0 ..< 128).flatMap { t -> [Float] in
            let x = Float(t)
            return [sin(x), cos(x), sin(2 * x), cos(3 * x)]
        }
        table = MLXArray(rows).reshaped(128, 4)
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray?, tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput {
        EmbeddingModelOutput(hiddenStates: table[inputs], pooledOutput: nil)
    }
}

extension KernelTests {

    /// Tests of `MLXEmbedderContainerEngine.createEmbedding(request:)` with
    /// a table model of 4 dimensions and the scalar tokenizer of the unit
    /// tests (BOS 1 first, pad token `</s>` = 2, one token for each
    /// character).
    @Suite
    struct EmbedderContainerEngineTests {

        // The embeddings are float32 values of size near 1 after a layer
        // norm and an L2 norm of 4 values. The tolerance allows for the
        // order of the sums.
        static let tolerance: Float = 1e-5

        private func makeEngine(strategy: Pooling.Strategy = .mean) -> MLXEmbedderContainerEngine {
            MLXEmbedderContainerEngine(
                modelID: "table",
                model: EmbedderModelContainer(
                    context: EmbedderModelContext(
                        configuration: ModelConfiguration(id: "test/table"),
                        model: TableEmbeddingModel(), tokenizer: UnitTests.ScalarTokenizer(),
                        pooling: Pooling(strategy: strategy))))
        }

        private func embed(
            _ engine: MLXEmbedderContainerEngine, _ input: OpenAIEmbeddingInput,
            normalize: Bool? = nil
        ) async throws -> OpenAIEmbeddingResponse {
            try await engine.createEmbedding(
                request: .init(model: "requested", input: input, normalize: normalize))
        }

        private func length(_ vector: [Float]) -> Float {
            vector.map { $0 * $0 }.reduce(0, +).squareRoot()
        }

        private func maxDifference(_ a: [Float], _ b: [Float]) -> Float {
            zip(a, b).map { abs($0 - $1) }.max() ?? 0
        }

        @Test func responseHasOneItemPerInputAndTheUsage() async throws {
            let response = try await embed(makeEngine(), .texts(["ab", "c"]))

            #expect(response.object == "list")
            #expect(response.model == "requested")
            #expect(response.data.map(\.index) == [0, 1])
            #expect(response.data.allSatisfy { $0.object == "embedding" })
            #expect(response.data.allSatisfy { $0.embedding.count == 4 })
            // "ab" is BOS + 2 tokens, "c" is BOS + 1 token.
            #expect(response.usage.promptTokens == 5)
            #expect(response.usage.completionTokens == 0)
            #expect(response.usage.totalTokens == 5)
        }

        /// The engine normalizes by default. Without the L2 norm the layer
        /// norm output of 4 values has a length of 2: mean 0 and variance 1
        /// give a sum of squares of 4.
        @Test func normalizeControlsTheLength() async throws {
            let engine = makeEngine()
            let normalized = try await embed(engine, .text("hello"))
            #expect(abs(length(normalized.data[0].embedding) - 1) <= Self.tolerance)

            let raw = try await embed(engine, .text("hello"), normalize: false)
            // The layer norm epsilon 1e-5 makes the length a little less
            // than 2.
            #expect(abs(length(raw.data[0].embedding) - 2) <= 1e-3)
        }

        /// Texts of the same length need no padding, so the batch gives the
        /// same embeddings as each text alone.
        @Test func textsOfTheSameLengthAreBatchInvariant() async throws {
            let engine = makeEngine()
            let batch = try await embed(engine, .texts(["ab", "cd"]))
            let first = try await embed(engine, .text("ab"))
            let second = try await embed(engine, .text("cd"))
            #expect(
                maxDifference(batch.data[0].embedding, first.data[0].embedding) <= Self.tolerance)
            #expect(
                maxDifference(batch.data[1].embedding, second.data[0].embedding) <= Self.tolerance)
            #expect(maxDifference(first.data[0].embedding, second.data[0].embedding) > 1e-3)
        }

        /// A short text in a batch with a long text gets pad tokens. The
        /// engine gives the attention mask to the model but not to the
        /// pooling, so mean pooling averages the pad positions too.
        @Test func paddedTextKeepsItsEmbedding() async throws {
            let engine = makeEngine(strategy: .mean)
            let batch = try await embed(engine, .texts(["a", "abcdef"]))
            let alone = try await embed(engine, .text("a"))
            let longAlone = try await embed(engine, .text("abcdef"))

            // The longest text has no padding.
            #expect(
                maxDifference(batch.data[1].embedding, longAlone.data[0].embedding)
                    <= Self.tolerance)

            let difference = maxDifference(batch.data[0].embedding, alone.data[0].embedding)
            withKnownIssue(
                "MLXEmbeddingServerEngine.swift:45 pools without the padding mask"
            ) {
                #expect(
                    difference <= Self.tolerance, "a padded text must keep its embedding")
            } matching: { issue in
                guard case .expectationFailed = issue.kind else { return false }
                return issue.comments.contains {
                    $0.rawValue.contains("a padded text must keep its embedding")
                }
            }
        }
    }
}
