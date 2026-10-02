import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// One combination of sampler filters and the tokens that it may return.
struct GenSamplerFilterCase: Sendable, CustomTestStringConvertible {
    let topP: Float
    let minP: Float
    let topK: Int
    let allowed: [Int]

    var testDescription: String { "topP \(topP) minP \(minP) topK \(topK)" }
}

extension KernelTests {

    /// Tests of the samplers in `Evaluate.swift`: the choice of sampler,
    /// seeded random state, temperature, the bfloat16 path and combinations
    /// of the top-p, min-p and top-k filters.
    @Suite
    struct GenerateSamplerTests {

        /// 16 logits `0.35 * ((j * 5 + 3) % 16)`. The probabilities in
        /// token order are (3 decimals):
        /// 0.004 0.026 0.147 0.003 0.018 0.104 0.002 0.013
        /// 0.073 0.002 0.009 0.052 0.296 0.006 0.036 0.209
        static func logits() -> MLXArray {
            MLXArray((0 ..< 16).map { 0.35 * Float(($0 * 5 + 3) % 16) })[.newAxis, .ellipsis]
        }

        static func draws(_ sampler: LogitSampler, _ logits: MLXArray, count: Int) -> [Int] {
            (0 ..< count).map { _ in sampler.sample(logits: logits).item(Int.self) }
        }

        /// True when `value` is not nil. The tests use it for optionals of
        /// types that are not `Equatable`, and for `MLXArray`, which has its
        /// own `==` and `!=` operators.
        static func isSet<T>(_ value: T?) -> Bool {
            if case .some = value { return true }
            return false
        }

        @Test func parametersChooseTheSampler() {
            #expect(GenerateParameters(temperature: 0.7).sampler() is CategoricalSampler)
            #expect(GenerateParameters(temperature: 0.7, topP: 1).sampler() is CategoricalSampler)
            #expect(GenerateParameters(temperature: 0.7, topP: 0).sampler() is CategoricalSampler)
            #expect(GenerateParameters(temperature: 0.7, topP: 0.9).sampler() is TopPSampler)
            #expect(GenerateParameters(temperature: 0, topK: 5).sampler() is ArgMaxSampler)
        }

        @Test func topPSamplerIgnoresDisabledFilters() {
            let sampler = TopPSampler(temperature: 1, topP: 1.5, topK: 0, minP: 0)
            #expect(!Self.isSet(sampler.topP))
            #expect(sampler.topK == nil)
            #expect(!Self.isSet(sampler.minP))

            let enabled = TopPSampler(temperature: 1, topP: 0.5, topK: 3, minP: 0.25)
            #expect(enabled.topP?.item(Float.self) == 0.5)
            #expect(enabled.topK == 3)
            #expect(enabled.minP?.item(Float.self) == 0.25)
        }

        @Test func categoricalSamplerIsReproducibleWithASeed() {
            let logits = Self.logits()
            let a = CategoricalSampler(temperature: 1)
            let b = CategoricalSampler(temperature: 1)
            let c = CategoricalSampler(temperature: 1)
            a.randomState.seed(7)
            b.randomState.seed(7)
            c.randomState.seed(8)
            let first = Self.draws(a, logits, count: 24)
            #expect(first == Self.draws(b, logits, count: 24))
            // Control: another seed gives other draws. 24 draws from this
            // distribution are equal for two seeds with a negligible chance.
            #expect(first != Self.draws(c, logits, count: 24))
            #expect(first.allSatisfy { (0 ..< 16).contains($0) })
        }

        @Test func lowTemperatureGivesTheArgmax() {
            // At temperature 0.01 the gap of 0.35 between the best logit
            // (token 12) and the next (token 15) becomes 35, so another
            // token has a probability below 1e-14.
            let logits = Self.logits()
            let categorical = CategoricalSampler(temperature: 0.01)
            categorical.randomState.seed(1)
            #expect(Self.draws(categorical, logits, count: 20).allSatisfy { $0 == 12 })

            let filtered = TopPSampler(temperature: 0.01, topK: 6)
            filtered.randomState.seed(1)
            #expect(Self.draws(filtered, logits, count: 20).allSatisfy { $0 == 12 })
        }

        @Test func topKAtOrAboveVocabularySizeKeepsEveryToken() {
            // With topK >= 16 the filter returns the log probabilities
            // unchanged, so the draws equal those of a sampler without
            // filters that has the same seed (same ops on the same data).
            let logits = Self.logits()
            let reference = TopPSampler(temperature: 1)
            reference.randomState.seed(3)
            let expected = Self.draws(reference, logits, count: 40)
            for topK in [16, 100] {
                let sampler = TopPSampler(temperature: 1, topK: topK)
                sampler.randomState.seed(3)
                #expect(Self.draws(sampler, logits, count: 40) == expected, "topK \(topK)")
            }
            // Control: the draws are not all the same few tokens.
            #expect(Set(expected).count > 3)
        }

        @Test func bfloat16LogitsAreSampled() {
            let logits = Self.logits().asType(.bfloat16)
            let sampler = TopPSampler(temperature: 1, topK: 1)
            sampler.randomState.seed(5)
            #expect(Self.draws(sampler, logits, count: 5) == [12, 12, 12, 12, 12])

            let nucleus = TopPSampler(temperature: 1, topP: 0.9)
            nucleus.randomState.seed(5)
            let allowed: Set<Int> = [2, 5, 8, 11, 12, 14, 15]
            #expect(Self.draws(nucleus, logits, count: 40).allSatisfy { allowed.contains($0) })
        }

        /// The allowed sets come from the probabilities in `logits()`:
        /// - top-p 0.9 removes the ascending tail with cumulative mass up to
        ///   0.1: it keeps {2, 5, 8, 11, 12, 14, 15} (the tail sums to 0.083
        ///   before token 14 and to 0.119 with it).
        /// - min-p 0.2 keeps probabilities >= 0.2 * 0.296 = 0.059:
        ///   {2, 5, 8, 12, 15}.
        /// - min-p 0.05 keeps probabilities >= 0.015: {1, 2, 4, 5, 8, 11,
        ///   12, 14, 15}.
        /// - top-k 6 keeps {2, 5, 8, 11, 12, 15}; top-k 3 keeps {2, 12, 15}.
        ///
        /// Each filter keeps the tokens above a threshold, so the chain keeps
        /// the intersection.
        @Test(arguments: [
            GenSamplerFilterCase(topP: 0.9, minP: 0, topK: 6, allowed: [2, 5, 8, 11, 12, 15]),
            GenSamplerFilterCase(topP: 0.9, minP: 0.2, topK: 6, allowed: [2, 5, 8, 12, 15]),
            GenSamplerFilterCase(topP: 1, minP: 0.05, topK: 3, allowed: [2, 12, 15]),
            GenSamplerFilterCase(
                topP: 0.9, minP: 0.05, topK: 0, allowed: [2, 5, 8, 11, 12, 14, 15]),
        ])
        func combinedFiltersKeepOnlyTheIntersection(_ filter: GenSamplerFilterCase) {
            let sampler = TopPSampler(
                temperature: 1, topP: filter.topP, topK: filter.topK, minP: filter.minP)
            sampler.randomState.seed(11)
            let tokens = Self.draws(sampler, Self.logits(), count: 200)
            let outside = Set(tokens).subtracting(filter.allowed)
            #expect(outside.isEmpty, "tokens outside \(filter.allowed): \(outside.sorted())")
            // Control: the sampler does not collapse to one token.
            #expect(Set(tokens).count >= 3)
        }
    }

    /// Tests of the penalty processors and of `TokenRing` in
    /// `Evaluate.swift`. The values are exact: each penalty multiplies,
    /// divides or subtracts once with numbers that float32 holds exactly.
    @Suite
    struct GeneratePenaltyTests {

        static let logits: [Float] = [3, -1, 2, 5]

        static func values(_ processor: LogitProcessor) -> [Float] {
            processor.process(logits: MLXArray(logits)[.newAxis, .ellipsis])[0].asArray(Float.self)
        }

        static func token(_ value: Int) -> MLXArray {
            MLXArray([Int32(value)])
        }

        @Test func tokenRingLoadsAppendsAndWraps() {
            var ring = TokenRing(capacity: 4)
            #expect(!GenerateSamplerTests.isSet(ring.validTokens))
            #expect(ring.count == 0)
            #expect(ring.capacity == 4)

            ring.loadPrompt(MLXArray([Int32(7), 8]))
            #expect(ring.validTokens?.asArray(Int32.self) == [7, 8])
            #expect(ring.buffer.asArray(Int32.self) == [7, 8, 0, 0])

            ring.append(Self.token(9))
            ring.append(Self.token(10))
            #expect(ring.count == 4)
            #expect(ring.validTokens?.asArray(Int32.self) == [7, 8, 9, 10])

            // Full: the next token replaces the oldest one.
            ring.append(Self.token(11))
            #expect(ring.count == 4)
            #expect(ring.buffer.asArray(Int32.self) == [11, 8, 9, 10])
        }

        @Test func tokenRingPromptOfExactCapacity() {
            var ring = TokenRing(capacity: 3)
            ring.loadPrompt(MLXArray([Int32(1), 2, 3]))
            #expect(ring.count == 3)
            #expect(ring.buffer.asArray(Int32.self) == [1, 2, 3])
            ring.append(Self.token(4))
            #expect(ring.buffer.asArray(Int32.self) == [4, 2, 3])
        }

        @Test func contextsWithoutTokensDoNotChangeTheLogits() {
            #expect(
                Self.values(RepetitionContext(repetitionPenalty: 2, repetitionContextSize: 4))
                    == Self.logits)
            #expect(
                Self.values(PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 4))
                    == Self.logits)
            #expect(
                Self.values(
                    FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 4))
                    == Self.logits)
        }

        @Test func sampledTokensArePenalizedWithoutAPrompt() {
            var repetition = RepetitionContext(repetitionPenalty: 2, repetitionContextSize: 4)
            repetition.didSample(token: Self.token(1))
            repetition.didSample(token: Self.token(2))
            // A negative logit is multiplied by the penalty, a positive one
            // is divided by it.
            #expect(Self.values(repetition) == [3, -2, 1, 5])

            var presence = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 4)
            presence.didSample(token: Self.token(1))
            presence.didSample(token: Self.token(1))
            // Presence counts a token once.
            #expect(Self.values(presence) == [3, -1.5, 2, 5])

            var frequency = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 4)
            frequency.didSample(token: Self.token(1))
            frequency.didSample(token: Self.token(1))
            // Frequency counts each occurrence.
            #expect(Self.values(frequency) == [3, -2, 2, 5])
        }

        @Test func oldTokensLeaveTheWindow() {
            var repetition = RepetitionContext(repetitionPenalty: 2, repetitionContextSize: 2)
            repetition.prompt(MLXArray([Int32(0), 1]))
            repetition.didSample(token: Self.token(2))
            // The window holds 1 and 2. Token 0 left it.
            let processed = repetition.process(
                logits: MLXArray([Float](repeating: 4, count: 4))[
                    .newAxis, .ellipsis])
            #expect(processed[0].asArray(Float.self) == [4, 2, 2, 4])
        }

        @Test func penaltyProcessorForwardsSampledTokens() {
            var processor = GenerateParameters(
                repetitionPenalty: 2, repetitionContextSize: 4, presencePenalty: 0.5,
                presenceContextSize: 4, frequencyPenalty: 0.25, frequencyContextSize: 4
            ).processor()
            processor?.prompt(MLXArray([Int32(3)]))
            processor?.didSample(token: Self.token(0))
            guard let processor else {
                Issue.record("the parameters must give a processor")
                return
            }
            // Token 0: 3 / 2 - 0.5 - 0.25 = 0.75. Token 3: 5 / 2 - 0.5 - 0.25
            // = 1.75. Tokens 1 and 2 are not in the window.
            #expect(Self.values(processor) == [0.75, -1, 2, 1.75])
        }

        @Test func parametersWithoutAnEffectiveContextGiveNoProcessor() {
            #expect(
                !GenerateSamplerTests.isSet(
                    GenerateParameters(repetitionPenalty: 1.5, repetitionContextSize: 0).processor()
                ))
            #expect(!GenerateSamplerTests.isSet(GenerateParameters(presencePenalty: 0).processor()))
            #expect(
                !GenerateSamplerTests.isSet(
                    GenerateParameters(presencePenalty: 0.5, presenceContextSize: 0).processor()))
            #expect(
                !GenerateSamplerTests.isSet(GenerateParameters(frequencyPenalty: 0).processor()))
            #expect(
                !GenerateSamplerTests.isSet(
                    GenerateParameters(frequencyPenalty: 0.5, frequencyContextSize: 0).processor()))
            #expect(!GenerateSamplerTests.isSet(GenerateParameters().processor()))
        }

        @Test func parametersBuildOnlyTheRequestedContexts() {
            let presenceOnly =
                GenerateParameters(presencePenalty: 0.5).processor() as? PenaltyProcessor
            #expect(GenerateSamplerTests.isSet(presenceOnly))
            #expect(!GenerateSamplerTests.isSet(presenceOnly?.repetitionContext))
            #expect(GenerateSamplerTests.isSet(presenceOnly?.presenceContext))
            #expect(!GenerateSamplerTests.isSet(presenceOnly?.frequencyContext))

            let frequencyOnly =
                GenerateParameters(frequencyPenalty: 0.5).processor() as? PenaltyProcessor
            #expect(!GenerateSamplerTests.isSet(frequencyOnly?.repetitionContext))
            #expect(!GenerateSamplerTests.isSet(frequencyOnly?.presenceContext))
            #expect(GenerateSamplerTests.isSet(frequencyOnly?.frequencyContext))
        }
    }

    /// Tests of the value types in `Evaluate.swift`: `GenerateCompletionInfo`,
    /// `Generation` and `TokenGeneration`.
    @Suite
    struct GenerationValueTests {

        static let info = GenerateCompletionInfo(
            promptTokenCount: 10, generationTokenCount: 20, promptTime: 2, generationTime: 4)

        @Test func completionInfoRatesAndSummary() {
            #expect(Self.info.stopReason == .stop, "the default stop reason")
            #expect(Self.info.promptTokensPerSecond == 5)
            #expect(Self.info.tokensPerSecond == 5)
            #expect(Self.info.generateTime == 4)
            let summary = Self.info.summary()
            #expect(summary.contains("Prompt:     10 tokens"))
            #expect(summary.contains("Generation: 20 tokens"))

            let length = GenerateCompletionInfo(
                promptTokenCount: 1, generationTokenCount: 1, promptTime: 1, generationTime: 1,
                stopReason: .length)
            #expect(length.stopReason == .length)
        }

        @Test func generationAccessors() {
            let call = ToolCall(
                function: ToolCall.Function(name: "f", arguments: [:] as [String: JSONValue]))

            let chunk = Generation.chunk("a")
            #expect(chunk.chunk == "a")
            #expect(!GenerateSamplerTests.isSet(chunk.info))
            #expect(chunk.toolCall == nil)

            let info = Generation.info(Self.info)
            #expect(info.chunk == nil)
            #expect(info.info?.generationTokenCount == 20)
            #expect(info.toolCall == nil)

            let tool = Generation.toolCall(call)
            #expect(tool.chunk == nil)
            #expect(!GenerateSamplerTests.isSet(tool.info))
            #expect(tool.toolCall == call)

            let batch = Generation.collect(Generation.collect(nil, chunk), tool)
            #expect(batch.count == 2)
            #expect(batch[0].chunk == "a")
            #expect(batch[1].toolCall == call)
        }

        @Test func tokenGenerationAccessors() {
            let token = TokenGeneration.token(4)
            #expect(token.token == 4)
            #expect(!GenerateSamplerTests.isSet(token.info))

            let info = TokenGeneration.info(Self.info)
            #expect(info.token == nil)
            #expect(info.info?.promptTokenCount == 10)

            let batch = TokenGeneration.collect(TokenGeneration.collect(nil, token), info)
            #expect(batch.count == 2)
            #expect(batch[0].token == 4)
            #expect(GenerateSamplerTests.isSet(batch[1].info))
        }
    }
}
