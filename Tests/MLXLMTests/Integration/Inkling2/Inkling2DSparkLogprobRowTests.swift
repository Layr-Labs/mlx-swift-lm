import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Rows that request logprobs, drafting on the tiny Inkling 2 target of
/// `Inkling2DSparkTargetEngineTests` under the logprob opt-in. Its verify
/// windows are `.rectangularExact`, so every committed token's logprob and
/// top logprobs must equal the drafter-off engine's bit for bit: same token,
/// same logprob, same top-k set, order and tie-breaks.
@Suite(.serialized, .tags(.integration))
struct Inkling2DSparkLogprobRowTests {
    struct Collected: Equatable {
        var tokens: [Int] = []
        var logprobs: [Entry] = []
        var finishReason: CBv2FinishReason?
    }

    /// One emitted logprob, compared by bit pattern.
    struct Entry: Equatable {
        let token: Int
        let logprobBits: UInt32
        let top: [Int]
        let topBits: [UInt32]

        init(_ value: CBv2TokenLogprob) {
            token = value.token
            logprobBits = value.logprob.bitPattern
            top = value.topLogprobs.map(\.token)
            topBits = value.topLogprobs.map(\.logprob.bitPattern)
        }
    }

    struct Row {
        let prompt: [Int]
        let sampling: CBv2SamplingParams
    }

    static func engine(
        _ model: Inkling2Model, drafter: Inkling2DSparkDrafter?, rows: Int,
        logprobSpeculation: Bool, adaptive: Bool = false
    ) throws -> EngineV2 {
        let dtype = DType.bfloat16
        let kinds = model.cbv2LayerKinds
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 128 << 20, dtype: dtype, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20,
                layerDTypes: Array(repeating: dtype, count: kinds.count)))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        var config = Inkling2DSparkDrafter.defaultMTPConfig(adaptive: adaptive)
        config.logprobSpeculation = logprobSpeculation
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches),
            sampler: CBv2DefaultSampler(fallbackSeed: 3),
            schedulerConfig: .init(
                maxConcurrentRequests: rows, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: drafter == nil ? .init() : config)
    }

    static func collect(_ stream: AsyncStream<CBv2Event>) async -> Collected {
        var out = Collected()
        for await event in stream {
            switch event {
            case .delta(_, let tokens, let logprobs):
                out.tokens.append(contentsOf: tokens)
                out.logprobs.append(contentsOf: (logprobs ?? []).map(Entry.init))
            case .finished(let reason, _):
                out.finishReason = reason
                return out
            }
        }
        return out
    }

    static func run(_ engine: EngineV2, rows: [Row], tokens: Int) async throws -> [Collected] {
        let streams = try rows.enumerated().map { index, row in
            try engine.submit(
                .init(
                    id: .init(UInt64(index + 1)), promptTokens: row.prompt,
                    sampling: row.sampling, maxTokens: tokens, prefixCacheEnabled: false))
        }
        var outputs: [Collected] = []
        for stream in streams { outputs.append(await collect(stream)) }
        return outputs
    }

    static func rows(_ count: Int, temperature: Float, topLogprobs: [Int]) -> [Row] {
        (0 ..< count).map { index in
            Row(
                prompt: SyntheticModel.tokens(
                    count: 9 + 13 * index, vocabularySize: 60, seed: 40 + index),
                sampling: .init(
                    temperature: temperature, seed: UInt64(70 + index),
                    topLogprobs: topLogprobs[index % topLogprobs.count]))
        }
    }

    /// Drafter off, then drafter on with the opt-in; returns both outputs and
    /// the drafting engine's metrics.
    static func compare(
        _ rows: [Row], tokens: Int = 24, logprobSpeculation: Bool = true, adaptive: Bool = false
    ) async throws -> (off: [Collected], on: [Collected], metrics: CBv2MTPMetrics) {
        let model = try Inkling2DSparkTargetEngineTests.target(dtype: .bfloat16)
        let drafter = try Inkling2DSparkTargetEngineTests.drafter(
            target: model, dtype: .bfloat16)
        let baseline = try engine(
            model, drafter: nil, rows: rows.count, logprobSpeculation: false)
        let off = try await run(baseline, rows: rows, tokens: tokens)
        await baseline.shutdown()

        let drafting = try engine(
            model, drafter: drafter, rows: rows.count, logprobSpeculation: logprobSpeculation,
            adaptive: adaptive)
        #expect(drafting.mtpEffectiveConfig?.verificationMode == .rectangularExact)
        #expect(
            drafting.mtpEffectiveConfig?.effectiveLogprobSpeculation
                == (logprobSpeculation && CBv2MTPConfig.envLogprobSpeculation))
        let on = try await run(drafting, rows: rows, tokens: tokens)
        let metrics = try #require(drafting.mtpMetricsSnapshot())
        await drafting.shutdown()
        return (off, on, metrics)
    }

    /// Counts of compared logprob values and the largest difference among
    /// unequal ones, printed so a failure shows how far apart the bits are.
    static func report(_ label: String, off: [Collected], on: [Collected]) {
        var compared = 0
        var unequal = 0
        var largest: Float = 0
        for (a, b) in zip(off, on) {
            for (x, y) in zip(a.logprobs, b.logprobs) {
                let pairs =
                    [(x.logprobBits, y.logprobBits)] + Array(zip(x.topBits, y.topBits))
                for (p, q) in pairs {
                    compared += 1
                    guard p != q else { continue }
                    unequal += 1
                    largest = max(
                        largest, abs(Float(bitPattern: p) - Float(bitPattern: q)))
                }
            }
        }
        print(
            "LOGPROB_ROWS \(label) compared=\(compared) unequal=\(unequal) "
                + "largestDifference=\(largest) tokensEqual=\(off.map(\.tokens) == on.map(\.tokens))"
        )
    }

    static func expectLogprobsPerToken(_ outputs: [Collected], rows: [Row]) {
        for (output, row) in zip(outputs, rows) {
            #expect(output.finishReason == .length)
            if row.sampling.topLogprobs > 0 {
                #expect(output.logprobs.count == output.tokens.count)
                #expect(output.logprobs.map(\.token) == output.tokens)
                #expect(output.logprobs.allSatisfy { $0.top.count == row.sampling.topLogprobs })
            } else {
                #expect(output.logprobs.isEmpty)
            }
        }
    }

    @Test(arguments: [1, 2, 4])
    func greedyLogprobRowsDraftWithDrafterOffBits(rows count: Int) async throws {
        let rows = Self.rows(count, temperature: 0, topLogprobs: [5])
        let (off, on, metrics) = try await Self.compare(rows)
        Self.report("greedy rows=\(count) accepted=\(metrics.acceptedTokens)", off: off, on: on)
        Self.expectLogprobsPerToken(off, rows: rows)
        #expect(on == off)
        #expect(metrics.rounds > 0)
        #expect(metrics.draftedTokens > 0)
        #expect(metrics.rectangularVerificationRounds > 0)
    }

    @Test(arguments: [1, 2, 4])
    func sampledLogprobRowsDraftWithDrafterOffBits(rows count: Int) async throws {
        // The shipped profile: adaptive depth.
        let rows = Self.rows(count, temperature: 0.7, topLogprobs: [3, 8])
        let (off, on, metrics) = try await Self.compare(rows, adaptive: true)
        Self.report("sampled rows=\(count) accepted=\(metrics.acceptedTokens)", off: off, on: on)
        Self.expectLogprobsPerToken(off, rows: rows)
        #expect(on == off)
        #expect(metrics.rounds > 0)
        #expect(metrics.draftedTokens > 0)
    }

    /// Logprob rows beside rows that ask for none, at mixed temperatures.
    @Test func mixedCohortKeepsEveryRowsBits() async throws {
        var rows = Self.rows(4, temperature: 0.7, topLogprobs: [0, 4, 0, 20])
        rows[2] = Row(prompt: rows[2].prompt, sampling: .init(temperature: 0, topLogprobs: 2))
        let (off, on, metrics) = try await Self.compare(rows)
        Self.report("mixed accepted=\(metrics.acceptedTokens)", off: off, on: on)
        Self.expectLogprobsPerToken(off, rows: rows)
        #expect(on == off)
        #expect(metrics.rounds > 0)
    }

    /// The opt-in is off by default and takes effect only on a verify path
    /// whose logits equal one-token decode's.
    @Test func logprobOptInNeedsExactVerification() {
        #expect(!CBv2MTPConfig().logprobSpeculation)
        #expect(!CBv2MTPConfig(verificationMode: .rectangularExact).effectiveLogprobSpeculation)
        #expect(
            CBv2MTPConfig(verificationMode: .rectangularExact, logprobSpeculation: true)
                .effectiveLogprobSpeculation == CBv2MTPConfig.envLogprobSpeculation)
        for mode in [CBv2MTPVerificationMode.rectangular, .automatic, .serialTarget] {
            #expect(
                !CBv2MTPConfig(verificationMode: mode, logprobSpeculation: true)
                    .effectiveLogprobSpeculation)
        }
        #expect(
            !CBv2MTPConfig(
                verificationMode: .rectangularExact, acceptance: .typical(delta: 0.2),
                logprobSpeculation: true
            ).effectiveLogprobSpeculation)
    }

    /// The shipped profile declares `.rectangular` and adaptive depth; the
    /// drafter's required mode makes it `.rectangularExact`, so the opt-in
    /// takes effect where the provider sets it.
    @Test func shippedProfileTakesTheOptIn() {
        var config = Inkling2DSparkDrafter.defaultMTPConfig()
        config.logprobSpeculation = true
        #expect(config.verificationMode == .rectangular)
        #expect(!config.effectiveLogprobSpeculation)
        let running = config.requiring(Inkling2DSparkDrafter.servingVerificationMode)
        #expect(running.verificationMode == .rectangularExact)
        #expect(running.effectiveLogprobSpeculation == CBv2MTPConfig.envLogprobSpeculation)
    }

    /// Without the opt-in a cohort of logprob rows never drafts (the
    /// historical gate) and its output is unchanged.
    @Test func withoutTheOptInLogprobRowsStayTargetOnly() async throws {
        let rows = Self.rows(2, temperature: 0.7, topLogprobs: [5])
        let (off, on, metrics) = try await Self.compare(rows, logprobSpeculation: false)
        #expect(on == off)
        #expect(metrics.draftedTokens == 0)
    }
}
