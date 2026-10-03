import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXEmbedders
@testable import MLXLMServer

/// Regression tests for the pooling mask of `MLXEmbedderContainerEngine`:
/// the engine gave the padding mask to the model but not to the pooling, so
/// mean pooling also averaged the pad positions of a short text in a batch.
///
/// `paddedTextKeepsItsEmbedding` is copied from `EmbedderContainerEngineTests`
/// of PR #237, without the known issue. The table model and the tokenizer
/// are private copies of the test doubles of that PR.
@Suite
struct EmbeddingServerPoolingMaskTests {

    // The embeddings are float32 values of size near 1 after a layer norm
    // and an L2 norm of 4 values. The tolerance allows for the order of the
    // sums.
    static let tolerance: Float = 1e-5

    private func makeEngine(strategy: Pooling.Strategy, appendsEOS: Bool = false)
        -> MLXEmbedderContainerEngine
    {
        MLXEmbedderContainerEngine(
            modelID: "table",
            model: EmbedderModelContainer(
                context: EmbedderModelContext(
                    configuration: ModelConfiguration(id: "test/table"),
                    model: PoolingMaskTableModel(),
                    tokenizer: PoolingMaskScalarTokenizer(appendsEOS: appendsEOS),
                    pooling: Pooling(strategy: strategy))))
    }

    /// The mean-pooled embedding of all of `tokens`, with no mask, computed
    /// directly with the table model and `Pooling`.
    private func referenceMeanEmbedding(_ tokens: [Int]) -> [Float] {
        let output = PoolingMaskTableModel()(
            MLXArray(tokens).expandedDimensions(axis: 0), positionIds: nil, tokenTypeIds: nil,
            attentionMask: nil)
        let pooled = Pooling(strategy: .mean)(output, normalize: true, applyLayerNorm: true)
        return pooled[0].asArray(Float.self)
    }

    private func embed(_ engine: MLXEmbedderContainerEngine, _ input: OpenAIEmbeddingInput)
        async throws -> OpenAIEmbeddingResponse
    {
        try await engine.createEmbedding(request: .init(model: "requested", input: input))
    }

    private func maxDifference(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    /// A short text in a batch with a long text gets pad tokens. With the
    /// mask, mean pooling averages only the tokens of the text, so the text
    /// keeps the embedding it has alone.
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
        #expect(difference <= Self.tolerance, "a padded text must keep its embedding")
    }

    /// Last-token pooling must take the last token of the text, not a pad
    /// token, so a padded text also keeps its embedding.
    @Test func paddedTextKeepsItsLastTokenEmbedding() async throws {
        let engine = makeEngine(strategy: .last)
        let batch = try await embed(engine, .texts(["ab", "abcdef"]))
        let alone = try await embed(engine, .text("ab"))
        let difference = maxDifference(batch.data[0].embedding, alone.data[0].embedding)
        #expect(difference <= Self.tolerance, "a padded text must keep its last-token embedding")
    }

    /// A tokenizer that appends EOS (the XLM-R family, for example
    /// intfloat/multilingual-e5-small with mean pooling) puts the EOS token
    /// last. The EOS token is also the pad token of the engine, but the real
    /// EOS token is a token of the text and must stay in the mean, alone and
    /// in a batch.
    @Test func appendedEOSStaysInTheMean() async throws {
        let engine = makeEngine(strategy: .mean, appendsEOS: true)

        // "ab" becomes <s> a b </s> = 1 97 98 2.
        let alone = try await embed(engine, .text("ab"))
        let expected = referenceMeanEmbedding([1, 97, 98, 2])
        #expect(
            maxDifference(alone.data[0].embedding, expected) <= Self.tolerance,
            "the appended EOS token must stay in the mean")

        // "a" (1 97 2) is padded with 2 in a batch with "abcdef".
        let batch = try await embed(engine, .texts(["a", "abcdef"]))
        #expect(
            maxDifference(batch.data[0].embedding, referenceMeanEmbedding([1, 97, 2]))
                <= Self.tolerance,
            "a padded text must keep its appended EOS token and lose only the pad tokens")
        #expect(
            maxDifference(
                batch.data[1].embedding,
                referenceMeanEmbedding([1, 97, 98, 99, 100, 101, 102, 2])) <= Self.tolerance)
    }
}

/// An embedding model whose hidden state for token `t` is row `t` of a fixed
/// table: `[sin(t), cos(t), sin(2t), cos(3t)]`. It ignores the attention
/// mask, so a padded position gets the hidden state of the pad token.
/// Copied from `TableEmbeddingModel` of PR #237.
private final class PoolingMaskTableModel: Module, EmbeddingModel {
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

/// A tokenizer that maps each Unicode scalar to its value. With
/// `addSpecialTokens` it puts the BOS token 1 first, and with `appendsEOS`
/// also the EOS token 2 last. `<s>` is 1 and `</s>` (the pad token of the
/// engine) is 2. Copied from `UnitTests.ScalarTokenizer` of PR #237, without
/// the recorder.
private struct PoolingMaskScalarTokenizer: MLXLMCommon.Tokenizer {
    var appendsEOS = false

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let tokens = text.unicodeScalars.map { Int($0.value) }
        guard addSpecialTokens else { return tokens }
        return [1] + tokens + (appendsEOS ? [2] : [])
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in
            if id < 32 {
                return skipSpecialTokens ? "" : "<\(id)>"
            }
            return UnicodeScalar(UInt32(id)).map { String(Character($0)) } ?? "?"
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token == "<s>" { return 1 }
        if token == "</s>" { return 2 }
        let scalars = Array(token.unicodeScalars)
        return scalars.count == 1 ? Int(scalars[0].value) : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        decode(tokenIds: [id], skipSpecialTokens: false)
    }

    var bosToken: String? { "<s>" }
    var eosToken: String? { "</s>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
