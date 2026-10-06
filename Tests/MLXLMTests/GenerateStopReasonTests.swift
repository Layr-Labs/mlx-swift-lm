import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Regression tests for two defects in `Evaluate.swift`:
///
/// - The async generate loop iterated over a copy of the iterator and then
///   read `tokenCount` of the original, which stayed 0. A stream that
///   stopped at `maxTokens` reported `.cancelled`, not `.length`.
/// - `SpeculativeTokenIterator` sampled the first token of a `.logits`
///   prepare result into `y` but did not return it, so the output started
///   one token late.
///
/// The tests are copied from `GenerateLoopTests.swift` of PR #240
/// (`GenerateStreamTests.tokensTaskReportsLength`,
/// `GenerateSpeculativeTests.streamWithRejectedDraftsMatchesTheMainModel`
/// and `GenerateSpeculativeTests.logitsPrepareResultKeepsTheFirstToken`),
/// without the known issues. The model is a tiny Markov model: the logits
/// of a position depend only on the token at that position, so a greedy run
/// follows a chain that the test computes on the CPU.
@Suite
struct GenerateStopReasonTests {

    @Test func tokensTaskReportsLength() async throws {
        let (stream, task) = try generateTokensTask(
            input: StopReasonFixture.input([3, 9, 4]),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0),
            context: StopReasonFixture.context(StopReasonMarkovModel()))
        let result = await StopReasonFixture.collect(stream)
        await task.value

        #expect(result.tokens == [5, 6, 7, 8])
        #expect(result.info?.generationTokenCount == 4)
        #expect(result.info?.promptTokenCount == 3)
        #expect(result.info?.stopReason == .length, "async loop stop reason")
        #expect((result.info?.promptTime ?? -1) >= 0)
    }

    @Test func speculativeStreamWithRejectedDraftsReportsLength() async throws {
        let tokenizer = StopReasonTokenizer()
        // The draft agrees with the main model on even tokens and proposes
        // a wrong token after each odd token.
        let draft = StopReasonMarkovModel(
            next: (0 ..< 16).map { $0 % 2 == 0 ? ($0 + 1) % 16 : ($0 + 2) % 16 })
        let stream = try generate(
            input: StopReasonFixture.input([3, 9, 4]),
            parameters: GenerateParameters(maxTokens: 7, temperature: 0),
            context: StopReasonFixture.context(StopReasonMarkovModel(), tokenizer: tokenizer),
            draftModel: draft, numDraftTokens: 3)
        let result = await StopReasonFixture.collect(stream)

        #expect(result.text == tokenizer.decode(tokenIds: [5, 6, 7, 8, 9, 10, 11]))
        #expect(result.info?.generationTokenCount == 7)
        #expect(result.info?.stopReason == .length, "async loop stop reason")
    }

    @Test func logitsPrepareResultKeepsTheFirstToken() throws {
        let prompt = [3, 9, 4]
        let expected = StopReasonMarkovModel().greedyChain(after: prompt, count: 6)
        var iterator = try SpeculativeTokenIterator(
            input: StopReasonFixture.input(prompt),
            mainModel: StopReasonMarkovModel(returnsLogits: true),
            draftModel: StopReasonMarkovModel(returnsLogits: true),
            parameters: GenerateParameters(maxTokens: 6, temperature: 0), numDraftTokens: 2)
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens.count == 6)
        #expect(tokens == expected, "speculative logits-prepare tokens")
    }
}

// MARK: - Fixtures

// The fixtures below are minimal copies of `GenLoopMarkovModel`,
// `GenLoopTokenizer` and `GenLoopFixture` from `GenerateLoopTests.swift` of
// PR #240. They are private and have other names, so they do not collide
// with that PR.

/// A tiny Markov language model. In row `t` of a 16 x 16 table the logit of
/// `next[t]` is 6.3. The other logits are `0.1 * ((j * 5 + t * 3) % 16)`,
/// all different in a row, so the argmax is never a tie. The model writes
/// one zero key and value per token to its cache.
private final class StopReasonMarkovModel: Module, LanguageModel {
    static let vocabularySize = 16
    static let successor: [Int] = (0 ..< 16).map { ($0 + 1) % 16 }

    let table: MLXArray
    let rows: [[Float]]
    let returnsLogits: Bool

    /// - Parameters:
    ///   - next: the most likely next token for each token
    ///   - returnsLogits: when true, `prepare` runs the whole prompt and
    ///     returns `.logits`; otherwise it returns `.tokens`
    init(next: [Int] = StopReasonMarkovModel.successor, returnsLogits: Bool = false) {
        let size = Self.vocabularySize
        let rows: [[Float]] = (0 ..< size).map { t in
            (0 ..< size).map { j in
                j == next[t] ? 6.3 : 0.1 * Float((j * 5 + t * 3) % size)
            }
        }
        self.rows = rows
        self.table = MLXArray(rows.flatMap { $0 }).reshaped(size, size)
        self.returnsLogits = returnsLogits
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        if returnsLogits {
            return .logits(
                self(input.text[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: nil))
        }
        let step = windowSize ?? 512
        var y = input.text
        while y.tokens.size > step {
            _ = self(y[.newAxis, ..<step], cache: cache.isEmpty ? nil : cache, state: nil)
            y = y[step...]
        }
        return .tokens(y)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let tokens = input.tokens.reshaped(-1, input.tokens.dim(-1))
        if let first = cache?.first {
            let kv = MLXArray.zeros([tokens.dim(0), 1, tokens.dim(1), 1], type: Float32.self)
            _ = first.update(keys: kv, values: kv)
        }
        return LMOutput(logits: take(table, tokens.asType(.int32), axis: 0))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        self(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    /// The greedy tokens after `prompt`, computed on the CPU from `rows`.
    func greedyChain(after prompt: [Int], count: Int) -> [Int] {
        var last = prompt[prompt.count - 1]
        var result: [Int] = []
        for _ in 0 ..< count {
            let row = rows[last]
            var best = 0
            for j in row.indices where row[j] > row[best] {
                best = j
            }
            last = best
            result.append(last)
        }
        return result
    }
}

/// A tokenizer that decodes token `id` to `t<id> ` and has no stop token.
private struct StopReasonTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "t\($0) " }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { nil }

    func convertIdToToken(_ id: Int) -> String? { "t\(id) " }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private enum StopReasonFixture {

    static func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map { Int32($0) }))
    }

    static func context(
        _ model: any LanguageModel, tokenizer: StopReasonTokenizer = StopReasonTokenizer()
    ) -> ModelContext {
        ModelContext(
            configuration: ModelConfiguration(id: "test/stop-reason"), model: model,
            processor: StandInUserInputProcessor(), tokenizer: tokenizer)
    }

    static func collect(_ stream: AsyncStream<Generation>) async -> (
        text: String, info: GenerateCompletionInfo?
    ) {
        var text = ""
        var info: GenerateCompletionInfo?
        for await generation in stream {
            switch generation {
            case .chunk(let chunk): text += chunk
            case .toolCall: break
            case .info(let value): info = value
            }
        }
        return (text, info)
    }

    static func collect(_ stream: AsyncStream<TokenGeneration>) async -> (
        tokens: [Int], info: GenerateCompletionInfo?
    ) {
        var tokens: [Int] = []
        var info: GenerateCompletionInfo?
        for await generation in stream {
            if let token = generation.token { tokens.append(token) }
            if let value = generation.info { info = value }
        }
        return (tokens, info)
    }
}
