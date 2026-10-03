import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

// MARK: - Fixtures

/// A tiny Markov language model for the generation loop tests.
///
/// The logits of a position depend only on the token at that position:
/// they are row `token` of a fixed 16 x 16 table. In row `t` the logit of
/// `next[t]` is 6.3. The other logits are `0.1 * ((j * 5 + t * 3) % 16)`.
/// These values are all different in a row (5 is coprime to 16), so the
/// argmax is never a tie. A greedy run thus follows the chain
/// `t -> next[t] -> next[next[t]] ...`, and a test can compute the
/// expected tokens on the CPU from `rows`.
///
/// The model writes one zero key and value per token to the first cache,
/// so the cache offset counts the tokens that the model has processed.
/// It records the length of each call and the state that each call gets.
final class GenLoopMarkovModel: Module, LanguageModel, GenericGenerationValidating {
    static let vocabularySize = 16
    static let successor: [Int] = (0 ..< 16).map { ($0 + 1) % 16 }

    let table: MLXArray
    let rows: [[Float]]
    let returnsLogits: Bool
    let rejectsGenericGeneration: Bool
    let emitsState: Bool

    private(set) var callLengths: [Int] = []
    private(set) var receivedStates: [Int?] = []

    /// - Parameters:
    ///   - next: the most likely next token for each token
    ///   - returnsLogits: when true, `prepare` runs the whole prompt and
    ///     returns `.logits`; otherwise it returns `.tokens`
    ///   - rejectsGenericGeneration: when true, `validateGenericGeneration`
    ///     throws
    ///   - emitsState: when true, each call returns a state that holds the
    ///     number of calls so far
    init(
        next: [Int] = GenLoopMarkovModel.successor, returnsLogits: Bool = false,
        rejectsGenericGeneration: Bool = false, emitsState: Bool = false
    ) {
        let size = Self.vocabularySize
        let rows: [[Float]] = (0 ..< size).map { t in
            (0 ..< size).map { j in
                j == next[t] ? 6.3 : 0.1 * Float((j * 5 + t * 3) % size)
            }
        }
        self.rows = rows
        self.table = MLXArray(rows.flatMap { $0 }).reshaped(size, size)
        self.returnsLogits = returnsLogits
        self.rejectsGenericGeneration = rejectsGenericGeneration
        self.emitsState = emitsState
        super.init()
    }

    func validateGenericGeneration() throws {
        if rejectsGenericGeneration {
            throw GenericGenerationError.nativeCBv2Required(modelType: "genloop")
        }
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        if returnsLogits {
            return .logits(
                self(input.text[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: nil))
        }
        // Same chunk loop as the default `LLMModel.prepare`.
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
        callLengths.append(tokens.dim(1))
        receivedStates.append(state?.crossAttentionStates?.item(Int.self))
        if let first = cache?.first {
            let kv = MLXArray.zeros([tokens.dim(0), 1, tokens.dim(1), 1], type: Float32.self)
            _ = first.update(keys: kv, values: kv)
        }
        let logits = take(table, tokens.asType(.int32), axis: 0)
        let newState =
            emitsState
            ? LMOutput.State(crossAttentionStates: MLXArray(Int32(callLengths.count))) : nil
        return LMOutput(logits: logits, state: newState)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        self(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    /// The index of the largest value. The rows have no ties.
    static func argmax(_ row: [Float]) -> Int {
        var best = 0
        for j in row.indices where row[j] > row[best] {
            best = j
        }
        return best
    }

    /// The greedy tokens after `prompt`, computed on the CPU from `rows`.
    /// `banned` is a token that can never be chosen.
    func greedyChain(after prompt: [Int], count: Int, banned: Int? = nil) -> [Int] {
        var last = prompt[prompt.count - 1]
        var result: [Int] = []
        for _ in 0 ..< count {
            var row = rows[last]
            if let banned {
                row[banned] = -.infinity
            }
            last = Self.argmax(row)
            result.append(last)
        }
        return result
    }
}

/// A tokenizer whose decode joins one fixed text fragment per token.
///
/// A token without a fragment decodes to `t<id> `. The text has no `<`,
/// so the tool call processor passes it through unchanged. The stop
/// tokens come from `eosToken`, `unknownToken` and `ids`, through the
/// `eosTokenId` and `unknownTokenId` extension properties of `Tokenizer`.
struct GenLoopTokenizer: MLXLMCommon.Tokenizer {
    var fragments: [Int: String] = [:]
    var ids: [String: Int] = [:]
    var eosToken: String? = nil
    var unknownToken: String? = nil
    var bosToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { fragments[$0] ?? "t\($0) " }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { ids[token] }

    func convertIdToToken(_ id: Int) -> String? { fragments[id] ?? "t\(id) " }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// A logit processor that bans one token and records what it sees.
struct GenLoopBanProcessor: LogitProcessor {
    final class Log {
        var prompts: [[Int]] = []
        var sampled: [Int] = []
    }

    let banned: Int
    let log: Log

    mutating func prompt(_ prompt: MLXArray) {
        log.prompts.append(prompt.asArray(Int.self))
    }

    func process(logits: MLXArray) -> MLXArray {
        var row = [Float](repeating: 0, count: logits.dim(-1))
        row[banned] = -1e9
        return logits + MLXArray(row)
    }

    mutating func didSample(token: MLXArray) {
        log.sampled.append(token.item(Int.self))
    }
}

enum GenLoopFixture {

    static func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map { Int32($0) }))
    }

    static func context(
        _ model: any LanguageModel, tokenizer: GenLoopTokenizer = GenLoopTokenizer(),
        configuration: ModelConfiguration = ModelConfiguration(id: "test/genloop")
    ) -> ModelContext {
        ModelContext(
            configuration: configuration, model: model, processor: StandInUserInputProcessor(),
            tokenizer: tokenizer)
    }

    /// A tokenizer with `<eos>` as the EOS token.
    static func tokenizer(eos: Int, fragments: [Int: String] = [:]) -> GenLoopTokenizer {
        GenLoopTokenizer(fragments: fragments, ids: ["<eos>": eos], eosToken: "<eos>")
    }

    static func drain(_ iterator: inout TokenIterator) -> [Int] {
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return tokens
    }

    /// The greedy tokens of `GenLoopMarkovModel` with penalties, computed
    /// on the CPU in the order of `PenaltyProcessor`: repetition, then
    /// presence, then frequency. Each window holds the last tokens of the
    /// prompt and of the sampled tokens.
    ///
    /// - Returns: the tokens and the smallest gap between the best and the
    ///   second best logit, so that a test can show that no step is a
    ///   near tie.
    static func penalizedChain(
        rows: [[Float]], prompt: [Int], count: Int,
        repetition: Float, repetitionWindow: Int,
        presence: Float, presenceWindow: Int,
        frequency: Float, frequencyWindow: Int
    ) -> (tokens: [Int], margin: Float) {
        var history = prompt
        var tokens: [Int] = []
        var margin = Float.infinity
        for _ in 0 ..< count {
            var row = rows[history[history.count - 1]]
            for token in Set(history.suffix(repetitionWindow)) {
                row[token] = row[token] < 0 ? row[token] * repetition : row[token] / repetition
            }
            for token in Set(history.suffix(presenceWindow)) {
                row[token] -= presence
            }
            for token in history.suffix(frequencyWindow) {
                row[token] -= frequency
            }
            let best = GenLoopMarkovModel.argmax(row)
            var second = -Float.infinity
            for j in row.indices where j != best {
                second = max(second, row[j])
            }
            margin = min(margin, row[best] - second)
            tokens.append(best)
            history.append(best)
        }
        return (tokens, margin)
    }

    /// A tiny Llama model with seeded random weights.
    static func llama(seed: UInt64 = 3) -> LlamaModel {
        let model = LlamaModel(
            LlamaConfiguration(
                hiddenSize: 32, hiddenLayers: 1, intermediateSize: 64, attentionHeads: 4,
                rmsNormEps: 1e-5, vocabularySize: 64, kvHeads: 2))
        SyntheticModel.randomize(model, seed: seed)
        return model
    }

    /// Checks that each generated token has the largest logit, within
    /// `tolerance`, in one forward pass without a cache over the prompt and
    /// the generated tokens. This check does not need the argmax of the two
    /// passes to agree on a near tie.
    static func checkGreedy(
        _ model: any LanguageModel, prompt: [Int], generated: [Int], tolerance: Float,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let all = prompt + generated.dropLast()
        let logits = ForwardPassChecks.logits(model, [all])
        for (index, token) in generated.enumerated() {
            let row = logits[0, prompt.count - 1 + index]
            let best = row.max().item(Float.self)
            let chosen = row[token].item(Float.self)
            #expect(
                chosen >= best - tolerance,
                "token \(index) (\(token)) has logit \(chosen), the largest is \(best)",
                sourceLocation: sourceLocation)
        }
    }
}

// MARK: - TokenIterator

extension KernelTests {

    /// Tests of `TokenIterator` in `Evaluate.swift`: prefill, the prepare
    /// results, the state, the processor and sampler hooks, and the cache.
    @Suite
    struct GenerateTokenIteratorTests {

        @Test func greedyTokensFollowTheArgmaxChain() throws {
            let model = GenLoopMarkovModel()
            let prompt = [3, 9, 4]
            var iterator = try TokenIterator(
                input: GenLoopFixture.input(prompt), model: model,
                parameters: GenerateParameters(maxTokens: 6, temperature: 0))
            let tokens = GenLoopFixture.drain(&iterator)

            #expect(tokens == model.greedyChain(after: prompt, count: 6))
            #expect(tokens == [5, 6, 7, 8, 9, 10])
            #expect(iterator.maxTokens == 6)
            #expect(iterator.tokenCount == 6)
            let after = iterator.next()
            #expect(after == nil, "no token after maxTokens")
            // The prompt runs in one call, then one call per returned token:
            // the iterator computes one token ahead.
            #expect(model.callLengths == [3, 1, 1, 1, 1, 1, 1])
            #expect(iterator.cache[0].offset == prompt.count + 6)
            #expect(iterator.promptPrefillTime >= 0)
        }

        @Test func prefillStepSizeSplitsThePrompt() throws {
            let model = GenLoopMarkovModel()
            let prompt = [1, 2, 3, 4, 5, 6, 7]
            var iterator = try TokenIterator(
                input: GenLoopFixture.input(prompt), model: model,
                parameters: GenerateParameters(maxTokens: 3, temperature: 0, prefillStepSize: 2))
            let tokens = GenLoopFixture.drain(&iterator)

            #expect(tokens == [8, 9, 10])
            // Chunks of 2 while more than 2 tokens remain, then the last
            // prompt token in the first step.
            #expect(model.callLengths == [2, 2, 2, 1, 1, 1, 1])
            #expect(iterator.cache[0].offset == prompt.count + 3)
        }

        @Test func logitsPrepareResultGivesTheSameTokens() throws {
            let prompt = [3, 9, 4]
            let tokensModel = GenLoopMarkovModel()
            var tokensIterator = try TokenIterator(
                input: GenLoopFixture.input(prompt), model: tokensModel,
                parameters: GenerateParameters(maxTokens: 5, temperature: 0))
            let logitsModel = GenLoopMarkovModel(returnsLogits: true)
            var logitsIterator = try TokenIterator(
                input: GenLoopFixture.input(prompt), model: logitsModel,
                parameters: GenerateParameters(maxTokens: 5, temperature: 0))

            let expected = GenLoopFixture.drain(&tokensIterator)
            let tokens = GenLoopFixture.drain(&logitsIterator)
            #expect(tokens == expected)
            #expect(expected == [5, 6, 7, 8, 9])
            // `prepare` runs the prompt and the iterator samples the first
            // token from its logits without another call.
            #expect(logitsModel.callLengths == [3, 1, 1, 1, 1, 1])
        }

        @Test func stateOfEachStepGoesToTheNextStep() throws {
            let model = GenLoopMarkovModel(emitsState: true)
            var iterator = try TokenIterator(
                input: GenLoopFixture.input([3, 9, 4]), model: model,
                parameters: GenerateParameters(maxTokens: 3, temperature: 0))
            _ = GenLoopFixture.drain(&iterator)

            // Call n returns state n. The first call gets no state.
            #expect(model.receivedStates == [nil, 1, 2, 3])
        }

        @Test func modelThatRejectsGenericGenerationThrows() {
            let model = GenLoopMarkovModel(rejectsGenericGeneration: true)
            #expect(throws: GenericGenerationError.nativeCBv2Required(modelType: "genloop")) {
                _ = try TokenIterator(
                    input: GenLoopFixture.input([1, 2]), model: model,
                    parameters: GenerateParameters(temperature: 0))
            }
            #expect(model.callLengths.isEmpty, "the model must not run")
        }

        @Test func deprecatedPromptInitializerGivesTheSameTokens() throws {
            let model = GenLoopMarkovModel()
            var iterator = try TokenIterator(
                prompt: MLXArray([Int32(3), 9, 4]), model: model,
                parameters: GenerateParameters(maxTokens: 4, temperature: 0))
            let tokens = GenLoopFixture.drain(&iterator)
            #expect(tokens == [5, 6, 7, 8])
        }

        @Test func customProcessorAndSamplerAreUsed() throws {
            let model = GenLoopMarkovModel()
            let log = GenLoopBanProcessor.Log()
            let prompt = [3, 9, 4]
            var iterator = try TokenIterator(
                input: GenLoopFixture.input(prompt), model: model,
                processor: GenLoopBanProcessor(banned: 6, log: log), sampler: ArgMaxSampler(),
                maxTokens: 8)
            let tokens = GenLoopFixture.drain(&iterator)

            // After 5 the chain goes to 6, which is banned, so the second
            // best token of row 5 follows.
            let expected = model.greedyChain(after: prompt, count: 9, banned: 6)
            #expect(tokens == Array(expected.prefix(8)))
            #expect(tokens == [5, 0, 1, 2, 3, 4, 5, 0])
            #expect(log.prompts == [prompt])
            // The processor also sees the token that the iterator computes
            // ahead.
            #expect(log.sampled == expected)
        }

        @Test func customInitializerWithoutMaxTokensKeepsGoing() throws {
            let model = GenLoopMarkovModel()
            var iterator = try TokenIterator(
                input: GenLoopFixture.input([3]), model: model, processor: nil,
                sampler: ArgMaxSampler(), prefillStepSize: 1)
            #expect(iterator.maxTokens == nil)
            var tokens: [Int] = []
            for _ in 0 ..< 20 {
                if let token = iterator.next() {
                    tokens.append(token)
                }
            }
            // 20 steps of the chain from 3 wrap around the 16 tokens.
            #expect(tokens == (4 ..< 24).map { $0 % 16 })
        }

        @Test func penaltyParametersChangeTheGreedyChain() throws {
            let model = GenLoopMarkovModel()
            let prompt = [5, 6, 4]
            let parameters = GenerateParameters(
                maxTokens: 10, temperature: 0, repetitionPenalty: 8, presencePenalty: 0.3,
                presenceContextSize: 4, frequencyPenalty: 0.2)
            var iterator = try TokenIterator(
                input: GenLoopFixture.input(prompt), model: model, parameters: parameters)
            let tokens = GenLoopFixture.drain(&iterator)

            let expected = GenLoopFixture.penalizedChain(
                rows: model.rows, prompt: prompt, count: 10,
                repetition: 8, repetitionWindow: 20, presence: 0.3, presenceWindow: 4,
                frequency: 0.2, frequencyWindow: 20)
            // Control: the gaps between the best and second logits are at
            // least 0.1, far above float32 rounding, so the GPU and the CPU
            // pick the same token.
            #expect(expected.margin > 1e-3)
            #expect(tokens == expected.tokens)
            // Without penalties the chain after 4 is 5, 6, ...; the prompt
            // holds 5, so the repetition penalty moves the first token.
            #expect(tokens.first != 5)
            #expect(model.greedyChain(after: prompt, count: 1) == [5])
        }

        @Test func seededSamplerGivesReproducibleTopKTokens() throws {
            func run(seed: UInt64) throws -> [Int] {
                let sampler = TopPSampler(temperature: 1, topK: 2)
                sampler.randomState.seed(seed)
                var iterator = try TokenIterator(
                    input: GenLoopFixture.input([3]), model: GenLoopMarkovModel(),
                    processor: nil, sampler: sampler, maxTokens: 16)
                return GenLoopFixture.drain(&iterator)
            }
            let first = try run(seed: 21)
            let second = try run(seed: 21)
            #expect(first.count == 16)
            #expect(first == second, "the same seed gives the same tokens")

            // Each token is one of the 2 largest logits of the row of the
            // token before it.
            let rows = GenLoopMarkovModel().rows
            var previous = 3
            for token in first {
                let topTwo = rows[previous].indices.sorted {
                    rows[previous][$0] > rows[previous][$1]
                }
                .prefix(2)
                #expect(topTwo.contains(token), "\(token) after \(previous)")
                previous = token
            }
        }

        // MARK: Real attention model

        @Test func llamaGreedyMatchesArgmaxOfTheModelLogits() throws {
            let model = GenLoopFixture.llama()
            let prompt = SyntheticModel.tokens(count: 7, vocabularySize: 64, seed: 1)
            let count = 8

            // Reference: the same calls that the iterator makes, by hand.
            // The ops and the data are the same, so the tokens must be equal.
            let cache = model.newCache(parameters: nil)
            var logits = model(MLXArray(prompt.map { Int32($0) })[.newAxis], cache: cache)
            var token = argMax(logits[0..., -1, 0...], axis: -1)
            var expected: [Int] = []
            for _ in 0 ..< count {
                expected.append(token.item(Int.self))
                logits = model(token[.newAxis], cache: cache)
                token = argMax(logits[0..., -1, 0...], axis: -1)
            }

            var iterator = try TokenIterator(
                input: GenLoopFixture.input(prompt), model: model,
                parameters: GenerateParameters(maxTokens: count, temperature: 0))
            let tokens = GenLoopFixture.drain(&iterator)
            #expect(tokens == expected)
            #expect(iterator.cache[0].offset == prompt.count + count)

            // Tolerance 1e-3: the cached and the full pass differ only in the
            // order of float32 sums (near 1e-6 for these sizes).
            GenLoopFixture.checkGreedy(model, prompt: prompt, generated: tokens, tolerance: 1e-3)
        }

        @Test func llamaWithChunkedPrefillAndPromptCache() throws {
            let model = GenLoopFixture.llama()
            let prefix = SyntheticModel.tokens(count: 4, vocabularySize: 64, seed: 2)
            let suffix = SyntheticModel.tokens(count: 5, vocabularySize: 64, seed: 3)

            // The cache already holds the prefix. The iterator gets only the
            // suffix, in chunks of 2.
            let cache = model.newCache(parameters: nil)
            _ = ForwardPassChecks.logits(model, [prefix], cache: cache)
            var iterator = try TokenIterator(
                input: GenLoopFixture.input(suffix), model: model, cache: cache,
                parameters: GenerateParameters(maxTokens: 6, temperature: 0, prefillStepSize: 2))
            let tokens = GenLoopFixture.drain(&iterator)

            #expect(tokens.count == 6)
            #expect(cache[0].offset == prefix.count + suffix.count + 6)
            // Tolerance 1e-3: chunked and cached passes change only the order
            // of float32 sums.
            GenLoopFixture.checkGreedy(
                model, prompt: prefix + suffix, generated: tokens, tolerance: 1e-3)
        }

        @Test func llamaWithMaxKVSizeUsesRotatingCache() throws {
            let model = GenLoopFixture.llama()
            let prompt = SyntheticModel.tokens(count: 6, vocabularySize: 64, seed: 4)
            var iterator = try TokenIterator(
                input: GenLoopFixture.input(prompt), model: model,
                parameters: GenerateParameters(maxTokens: 5, maxKVSize: 32, temperature: 0))
            let tokens = GenLoopFixture.drain(&iterator)

            #expect(iterator.cache.allSatisfy { $0 is RotatingKVCache })
            #expect(tokens.count == 5)
            // 11 tokens stay below the 32-token window, so the result matches
            // a full pass. Tolerance 1e-3 as above.
            GenLoopFixture.checkGreedy(model, prompt: prompt, generated: tokens, tolerance: 1e-3)
        }
    }

    // MARK: - Synchronous generate

    /// Tests of the deprecated synchronous `generate` functions and of
    /// `GenerateResult` in `Evaluate.swift`.
    @Suite
    struct GenerateSynchronousLoopTests {

        @Test func promptTokensGenerateStopsOnExtraEOSToken() throws {
            let model = GenLoopMarkovModel()
            let tokenizer = GenLoopTokenizer(ids: ["<end>": 8])
            let result = try generate(
                promptTokens: [3, 9, 4], parameters: GenerateParameters(temperature: 0),
                model: model, tokenizer: tokenizer, extraEOSTokens: ["<end>"]
            ) { _ in .more }

            #expect(result.tokenIds == [5, 6, 7])
            #expect(result.output == "t5 t6 t7 ")
            #expect(result.promptTokenIds == [3, 9, 4])
            #expect(result.promptTokenCount == 3)
            #expect(result.generationTokenCount == 3)
            #expect(result.tokens == result.tokenIds)
            #expect(result.promptTokens == [3, 9, 4])
            #expect(result.promptTime >= 0)
            #expect(result.generateTime >= 0)
            let summary = result.summary()
            #expect(summary.contains("Prompt:     3 tokens"))
            #expect(summary.contains("Generation: 3 tokens"))
        }

        @Test func generateResultRatesAndDeprecatedInitializer() {
            let result = GenerateResult(
                inputText: LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4])), tokens: [7, 8],
                output: "x", promptTime: 2, generateTime: 4)
            #expect(result.tokenIds == [7, 8])
            #expect(result.promptTokensPerSecond == 2)
            #expect(result.tokensPerSecond == 0.5)
            #expect(result.output == "x")
        }

        @Test func tokenArrayCallbackCanStop() throws {
            let model = GenLoopMarkovModel()
            // maxTokens 12 only keeps the test finite if the callback fails.
            let result = try generate(
                input: GenLoopFixture.input([3, 9, 4]),
                parameters: GenerateParameters(maxTokens: 12, temperature: 0),
                context: GenLoopFixture.context(model)
            ) { (tokens: [Int]) -> GenerateDisposition in
                tokens.count >= 2 ? .stop : .more
            }
            #expect(result.tokenIds == [5, 6])
            #expect(result.output == "t5 t6 ")
        }

        @Test func completionInfoStopsOnTokenizerEOS() throws {
            var seen: [Int] = []
            let info = try generate(
                input: GenLoopFixture.input([3, 9, 4]),
                parameters: GenerateParameters(maxTokens: 12, temperature: 0),
                context: GenLoopFixture.context(
                    GenLoopMarkovModel(), tokenizer: GenLoopFixture.tokenizer(eos: 8))
            ) { (token: Int) -> GenerateDisposition in
                seen.append(token)
                return .more
            }
            #expect(seen == [5, 6, 7])
            #expect(info.stopReason == .stop)
            #expect(info.promptTokenCount == 3)
            #expect(info.generationTokenCount == 3)
        }

        @Test func completionInfoStopsOnUnknownTokenAndConfigurationEOS() throws {
            let unknown = GenLoopTokenizer(ids: ["<unk>": 6], unknownToken: "<unk>")
            let first = try generate(
                input: GenLoopFixture.input([4]),
                parameters: GenerateParameters(maxTokens: 12, temperature: 0),
                context: GenLoopFixture.context(GenLoopMarkovModel(), tokenizer: unknown)
            ) { (_: Int) -> GenerateDisposition in .more }
            #expect(first.stopReason == .stop)
            #expect(first.generationTokenCount == 1, "5, then the unknown token 6")

            var configuration = ModelConfiguration(id: "test/genloop-eos")
            configuration.eosTokenIds = [7]
            let second = try generate(
                input: GenLoopFixture.input([4]),
                parameters: GenerateParameters(maxTokens: 12, temperature: 0),
                context: GenLoopFixture.context(
                    GenLoopMarkovModel(), configuration: configuration)
            ) { (_: Int) -> GenerateDisposition in .more }
            #expect(second.stopReason == .stop)
            #expect(second.generationTokenCount == 2, "5, 6, then the EOS id 7")
        }

        @Test func completionInfoReportsLengthAndCancel() throws {
            let length = try generate(
                input: GenLoopFixture.input([3]),
                parameters: GenerateParameters(maxTokens: 4, temperature: 0),
                context: GenLoopFixture.context(GenLoopMarkovModel())
            ) { (_: Int) -> GenerateDisposition in .more }
            #expect(length.stopReason == .length)
            #expect(length.generationTokenCount == 4)

            let cancelled = try generate(
                input: GenLoopFixture.input([3]),
                parameters: GenerateParameters(maxTokens: 4, temperature: 0),
                context: GenLoopFixture.context(GenLoopMarkovModel())
            ) { (_: Int) -> GenerateDisposition in .stop }
            #expect(cancelled.stopReason == .cancelled)
            #expect(cancelled.generationTokenCount == 1)
        }
    }

    // MARK: - Async streams

    /// Tests of the `AsyncStream` generate functions in `Evaluate.swift`:
    /// text chunks, tool calls, raw tokens, stop tokens, completion info,
    /// cancellation and the wired memory ticket.
    @Suite
    struct GenerateStreamTests {

        static let toolCallJSON = #"{"name": "get_time", "arguments": {"zone": "UTC"}}"#

        /// Collects the text, the tool calls and the completion info.
        static func collect(_ stream: AsyncStream<Generation>) async -> (
            text: String, toolCalls: [ToolCall], info: GenerateCompletionInfo?, lastIsInfo: Bool
        ) {
            var text = ""
            var toolCalls: [ToolCall] = []
            var info: GenerateCompletionInfo?
            var lastIsInfo = false
            for await generation in stream {
                lastIsInfo = false
                switch generation {
                case .chunk(let chunk): text += chunk
                case .toolCall(let call): toolCalls.append(call)
                case .info(let value):
                    info = value
                    lastIsInfo = true
                }
            }
            return (text, toolCalls, info, lastIsInfo)
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

        @Test func textChunksAndToolCall() async throws {
            let tokenizer = GenLoopFixture.tokenizer(
                eos: 13,
                fragments: [
                    8: "Hi ", 9: "there ", 10: "<tool_call>", 11: Self.toolCallJSON,
                    12: "</tool_call>",
                ])
            let stream = try generate(
                input: GenLoopFixture.input([2, 7]),
                parameters: GenerateParameters(maxTokens: 12, temperature: 0),
                context: GenLoopFixture.context(GenLoopMarkovModel(), tokenizer: tokenizer))
            let result = await Self.collect(stream)

            #expect(result.text == "Hi there ")
            #expect(result.toolCalls.count == 1)
            #expect(result.toolCalls.first?.function.name == "get_time")
            #expect(result.toolCalls.first?.function.arguments.keys.sorted() == ["zone"])
            #expect(result.lastIsInfo, "the info comes last")
            #expect(result.info?.stopReason == .stop)
            #expect(result.info?.promptTokenCount == 2)
            #expect(result.info?.generationTokenCount == 5)
        }

        @Test func toolCallOpenAtEOSIsParsedAtTheEnd() async throws {
            // The EOS token comes before `</tool_call>`. The handler parses
            // the buffer when generation ends.
            let tokenizer = GenLoopFixture.tokenizer(
                eos: 12, fragments: [10: "<tool_call>", 11: Self.toolCallJSON])
            let stream = try generate(
                input: GenLoopFixture.input([9]),
                parameters: GenerateParameters(maxTokens: 12, temperature: 0),
                context: GenLoopFixture.context(GenLoopMarkovModel(), tokenizer: tokenizer))
            let result = await Self.collect(stream)

            #expect(result.toolCalls.map(\.function.name) == ["get_time"])
            #expect(result.text == "")
            #expect(result.info?.stopReason == .stop)
        }

        @Test func unparsedBufferAtEOSBecomesText() async throws {
            let tokenizer = GenLoopFixture.tokenizer(
                eos: 12, fragments: [10: "<tool_call>", 11: "oops"])
            let stream = try generate(
                input: GenLoopFixture.input([9]),
                parameters: GenerateParameters(maxTokens: 12, temperature: 0),
                context: GenLoopFixture.context(GenLoopMarkovModel(), tokenizer: tokenizer))
            let result = await Self.collect(stream)

            #expect(result.toolCalls.isEmpty)
            #expect(result.text == "<tool_call>oops", "the withheld text is not lost")
        }

        @Test func rawTokensWithAndWithoutTheStopToken() async throws {
            let context = GenLoopFixture.context(
                GenLoopMarkovModel(), tokenizer: GenLoopFixture.tokenizer(eos: 8))
            let parameters = GenerateParameters(maxTokens: 12, temperature: 0)

            let withoutStream = try generateTokens(
                input: GenLoopFixture.input([3, 9, 4]), parameters: parameters, context: context)
            let without = await Self.collect(withoutStream)
            #expect(without.tokens == [5, 6, 7])
            #expect(without.info?.generationTokenCount == 3)
            #expect(without.info?.stopReason == .stop)

            let includedStream = try generateTokens(
                input: GenLoopFixture.input([3, 9, 4]), parameters: parameters, context: context,
                includeStopToken: true)
            let included = await Self.collect(includedStream)
            #expect(included.tokens == [5, 6, 7, 8])
            #expect(included.info?.generationTokenCount == 4)
            #expect(included.info?.stopReason == .stop)
        }

        @Test func tokensTaskReportsLength() async throws {
            let (stream, task) = try generateTokensTask(
                input: GenLoopFixture.input([3, 9, 4]),
                parameters: GenerateParameters(maxTokens: 4, temperature: 0),
                context: GenLoopFixture.context(GenLoopMarkovModel()))
            let result = await Self.collect(stream)
            await task.value

            #expect(result.tokens == [5, 6, 7, 8])
            #expect(result.info?.generationTokenCount == 4)
            #expect(result.info?.promptTokenCount == 3)
            #expect(result.info?.stopReason == .length, "async loop stop reason")
            #expect((result.info?.promptTime ?? -1) >= 0)
        }

        @Test func cancelledTokenTaskReportsCancelled() async throws {
            // No EOS. maxTokens 2000 only keeps the test finite if the
            // cancel fails.
            let iterator = try TokenIterator(
                input: GenLoopFixture.input([3]), model: GenLoopMarkovModel(),
                parameters: GenerateParameters(maxTokens: 2000, temperature: 0))
            let (stream, task) = generateTokenTask(
                promptTokenCount: 1, modelConfiguration: ModelConfiguration(id: "test/genloop"),
                tokenizer: GenLoopTokenizer(), iterator: iterator)
            task.cancel()
            let result = await Self.collect(stream)
            await task.value

            #expect(result.info?.stopReason == .cancelled)
            #expect(result.tokens.count < 2000)
        }

        @Test func breakingTheStreamStopsTheTask() async throws {
            let model = GenLoopMarkovModel()
            // No EOS. maxTokens 2000 only keeps the test finite if the
            // stream end does not cancel the task.
            let iterator = try TokenIterator(
                input: GenLoopFixture.input([3]), model: model,
                parameters: GenerateParameters(maxTokens: 2000, temperature: 0))
            // The stream lives only inside this function. When the function
            // returns, the stream is released, the stream ends with
            // `.cancelled`, and its `onTermination` handler cancels the task.
            func readFirstChunk() async -> (String?, Task<Void, Never>) {
                let (stream, task) = generateTask(
                    promptTokenCount: 1,
                    modelConfiguration: ModelConfiguration(id: "test/genloop"),
                    tokenizer: GenLoopTokenizer(), iterator: iterator)
                for await generation in stream {
                    if let chunk = generation.chunk {
                        return (chunk, task)
                    }
                }
                return (nil, task)
            }
            let (first, task) = await readFirstChunk()
            await task.value

            #expect(first == "t4 ")
            #expect(model.callLengths.count < 2000, "the task must stop early")
        }

        @Test func deprecatedIteratorStreamWithWiredMemoryTicket() async throws {
            let model = GenLoopMarkovModel()
            let context = GenLoopFixture.context(model, tokenizer: GenLoopFixture.tokenizer(eos: 8))
            let input = GenLoopFixture.input([3, 9, 4])
            let iterator = try TokenIterator(
                input: input, model: model,
                parameters: GenerateParameters(maxTokens: 12, temperature: 0))
            // Size 0 with the max policy keeps the wired limit at its
            // baseline, so the test does not change the limit of other tests.
            let ticket = WiredMemoryTicket(size: 0, policy: MLXLMCommon.WiredMaxPolicy())
            let stream = generate(
                input: input, context: context, iterator: iterator, wiredMemoryTicket: ticket)
            let result = await Self.collect(stream)

            #expect(result.text == "t5 t6 t7 ")
            #expect(result.info?.stopReason == .stop)
            #expect(result.info?.generationTokenCount == 3)
        }
    }

    // MARK: - Speculative decoding

    /// Tests of `SpeculativeTokenIterator` and the speculative generate
    /// functions in `Evaluate.swift` with tiny Markov models.
    @Suite
    struct GenerateSpeculativeTests {

        /// A draft model that agrees with the main model on even tokens and
        /// proposes a wrong token after each odd token.
        static func disagreeingDraft() -> GenLoopMarkovModel {
            GenLoopMarkovModel(next: (0 ..< 16).map { $0 % 2 == 0 ? ($0 + 1) % 16 : ($0 + 2) % 16 })
        }

        @Test func streamWithRejectedDraftsMatchesTheMainModel() async throws {
            let tokenizer = GenLoopTokenizer()
            let stream = try generate(
                input: GenLoopFixture.input([3, 9, 4]),
                parameters: GenerateParameters(maxTokens: 7, temperature: 0),
                context: GenLoopFixture.context(GenLoopMarkovModel(), tokenizer: tokenizer),
                draftModel: Self.disagreeingDraft(), numDraftTokens: 3)
            let result = await GenerateStreamTests.collect(stream)

            #expect(result.text == tokenizer.decode(tokenIds: [5, 6, 7, 8, 9, 10, 11]))
            #expect(result.info?.generationTokenCount == 7)
            #expect(result.info?.stopReason == .length, "async loop stop reason")
        }

        @Test func acceptedDraftsMatchTheMainModel() async throws {
            let main = GenLoopMarkovModel()
            let draft = GenLoopMarkovModel()
            let stream = try generateTokens(
                input: GenLoopFixture.input([3, 9, 4]),
                parameters: GenerateParameters(maxTokens: 9, temperature: 0),
                context: GenLoopFixture.context(main), draftModel: draft, numDraftTokens: 2)
            let result = await GenerateStreamTests.collect(stream)

            #expect(result.tokens == main.greedyChain(after: [3, 9, 4], count: 9))
            // The draft agrees each time, so each verify call checks the
            // last token and 2 drafts: 3 rounds of 3 tokens after the prompt.
            #expect(main.callLengths == [5, 3, 3])
        }

        @Test func logitsPrepareResultKeepsTheFirstToken() throws {
            let prompt = [3, 9, 4]
            let expected = GenLoopMarkovModel().greedyChain(after: prompt, count: 6)
            var iterator = try SpeculativeTokenIterator(
                input: GenLoopFixture.input(prompt),
                mainModel: GenLoopMarkovModel(returnsLogits: true),
                draftModel: GenLoopMarkovModel(returnsLogits: true),
                parameters: GenerateParameters(maxTokens: 6, temperature: 0), numDraftTokens: 2)
            var tokens: [Int] = []
            while let token = iterator.next() {
                tokens.append(token)
            }

            #expect(tokens.count == 6)
            #expect(tokens == expected, "speculative logits-prepare tokens")
        }

        @Test func untrimmableCacheIsRejected() {
            #expect(throws: KVCacheError.self) {
                _ = try SpeculativeTokenIterator(
                    input: GenLoopFixture.input([1, 2]), mainModel: GenLoopMarkovModel(),
                    draftModel: GenLoopMarkovModel(), mainCache: [MambaCache()],
                    parameters: GenerateParameters(temperature: 0), numDraftTokens: 2)
            }
        }

        @Test func draftThatRejectsGenericGenerationThrows() {
            #expect(throws: GenericGenerationError.nativeCBv2Required(modelType: "genloop")) {
                _ = try SpeculativeTokenIterator(
                    input: GenLoopFixture.input([1, 2]), mainModel: GenLoopMarkovModel(),
                    draftModel: GenLoopMarkovModel(rejectsGenericGeneration: true),
                    parameters: GenerateParameters(temperature: 0), numDraftTokens: 2)
            }
        }
    }
}
