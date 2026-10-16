import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Exact prefix reuse on the speculating Super owner: for text the resident
/// bank keeps, beside the chunk-aligned recurrent state and attention KV, the
/// assistant's queued prompt history, from which the first draft step builds
/// its FP8 draft history in canonical chunks. A restored request therefore
/// drafts, verifies and emits exactly what a cold one does. Synthetic weights,
/// depth-3 exact rectangular verification, the owner's fixed-chunk policy.
@Suite(.tags(.integration))
struct NemotronSuperMTPPrefixReuseTests {
    typealias Prefix = NemotronSuperPrefixReuseFixture

    static let chunk: Int = 8
    static let rows: Int = 4
    static let depth: Int = 3
    /// The image token of `NemotronSuperMTPFixture.target`.
    static let placeholder: Int = 18
    /// Text that avoids the image placeholder.
    static let conversation: [Int] =
        (0 ..< 3 * chunk + 6).map { (i: Int) -> Int in (i * 7 + 3) % 61 + 1 }
        .filter { (token: Int) -> Bool in token != placeholder }
    static let donor: [Int] = Array(conversation.prefix(2 * chunk + 5))

    /// The synthetic weights every engine here serves.
    static func model() throws -> NemotronSuperModel {
        try NemotronSuperMTPFixture.target(mixerScale: 0.25, quantized: true)
    }

    /// The serial plain owner on the same weights: the cold reference.
    static func plainEngine() throws -> EngineV2 {
        try NemotronSuperMTPEngineTests.plainEngine(Self.model(), kvDType: .bfloat16).0
    }

    static func stack() throws -> NemotronSuperMTPFixture.Stack {
        let model = try Self.model()
        let target = try NemotronSuperMTPCaptureTarget(
            capturing: model, maximumRequests: rows, exactPrefixCheckpoints: true)
        let head = try NemotronSuperMTPFixture.head(
            for: model.languageModel, kind: .predictive, dtype: .bfloat16)
        var settings = NemotronSuperMTPFixture.settings(
            depth: depth, verification: .rectangularExact)
        settings.maximumSpeculativeRows = rows
        let assistant = try target.makeAssistant(head: head, settings: settings)
        return .init(model: model, target: target, head: head, assistant: assistant)
    }

    /// The owner's policy: one whole prompt chunk per step beside every
    /// request's verify row.
    static func engine(
        _ stack: NemotronSuperMTPFixture.Stack, bank: CBv2HybridPrefixCacheConfig
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: rows, maxBatchedTokensPerStep: chunk + rows * (1 + depth),
            prefillChunkSize: chunk, maxConcurrentPartialPrefills: 1,
            mixedStepPrefillTokenCap: chunk, maxWaiting: 8, enablePrefixCache: true)
        config.maxPrefillRowsPerStep = 1
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: .bfloat16))
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(stack.target),
            layerKinds: stack.target.cbv2LayerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: stack.target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }),
            sampler: CBv2GreedySampler(), schedulerConfig: config,
            admissionConfig: .init(watermarkFraction: 0), hybridPrefixCache: bank,
            mtpDrafter: stack.assistant,
            mtpConfig: .init(
                enabled: true, maxDraftTokens: depth, maxSpeculativeBatch: rows,
                fixedDraftTokens: depth, verificationMode: .rectangularExact))
        return (engine, backend)
    }

    struct Served {
        var tokens: [Int] = []
        var finish: CBv2FinishReason?
        var saved = 0
    }

    static func request(_ id: UInt64, _ prompt: [Int], maxTokens: Int = 20) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt, sampling: .init(temperature: 0),
            maxTokens: maxTokens)
    }

    static func serve(
        _ engine: EngineV2, _ request: CBv2Request, onTokens: (Int) -> Void = { _ in }
    ) async throws -> Served {
        var served = Served()
        for await event in try engine.submit(request) {
            switch event {
            case .delta(_, let tokens, _):
                served.tokens += tokens
                onTokens(served.tokens.count)
            case .finished(let reason, let usage):
                served.finish = reason
                served.saved = usage.prefixCachePrefillTokensSaved
            }
        }
        return served
    }

    /// Rounds, drafted and accepted tokens of one request served alone.
    static func drafting(_ engine: EngineV2, _ body: () async throws -> Void) async rethrows
        -> [Int]
    {
        let before = engine.mtpMetricsSnapshot()
        try await body()
        let after = engine.mtpMetricsSnapshot()
        let rounds: Int = (after?.rounds ?? 0) - (before?.rounds ?? 0)
        let drafted: Int = (after?.draftedTokens ?? 0) - (before?.draftedTokens ?? 0)
        let accepted: Int = (after?.acceptedTokens ?? 0) - (before?.acceptedTokens ?? 0)
        return [rounds, drafted, accepted]
    }

    /// The donor publishes C and 2C. The next turn restores 2C and emits the
    /// cold run's tokens with the same rounds, drafts and acceptances; its
    /// restored recurrent state, KV and assistant history equal the cold
    /// request's own at 2C, and its next checkpoint (3C) equals the cold one.
    @Test(.timeLimit(.minutes(5)))
    func restoredSpeculativeRequestMatchesColdExactly() async throws {
        let restoredAt: Int = 2 * Self.chunk
        let next: Int = 3 * Self.chunk
        let bank = Prefix.bank(checkpointsPerRequest: 4)

        let plain = try Self.plainEngine()
        let reference = try await Self.serve(plain, Self.request(1, Self.conversation))
        await plain.shutdown()

        let coldStack = try Self.stack()
        let (cold, coldBackend) = Self.engine(coldStack, bank: bank)
        var coldServed = Served()
        let coldDrafting = try await Self.drafting(cold) {
            coldServed = try await Self.serve(cold, Self.request(2, Self.conversation))
        }
        #expect(coldServed.tokens == reference.tokens)
        #expect(coldServed.saved == 0)
        let coldRounds: Int = coldDrafting[0]
        #expect(coldRounds > 0)

        let warmStack = try Self.stack()
        let (warm, warmBackend) = Self.engine(warmStack, bank: bank)
        let warmBank = try #require(warm.hybridPrefixCache)
        _ = try await Self.serve(warm, Self.request(3, Self.donor))
        let published = await Prefix.eventually { warmBank.stats.entries == 1 }
        #expect(published)
        var restored = Served()
        let warmDrafting = try await Self.drafting(warm) {
            restored = try await Self.serve(warm, Self.request(4, Self.conversation))
        }
        #expect(restored.saved == restoredAt)
        #expect(restored.tokens == reference.tokens)
        #expect(warmDrafting == coldDrafting)

        let coldBank = try #require(cold.hybridPrefixCache)
        let bothPublished = await Prefix.eventually { warmBank.stats.entries == 2 }
        #expect(bothPublished)
        for position in [restoredAt, next] {
            let tokens = Array(Self.conversation.prefix(position + 1))
            let matches = Prefix.checkpointsMatch(
                warmBank, coldBank, tokens: tokens, position: position,
                maximumChunkSize: Self.chunk)
            #expect(matches, "position \(position)")
        }
        await cold.shutdown()
        await warm.shutdown()
        #expect(coldBackend.bytesReserved == 0 && warmBackend.bytesReserved == 0)
        #expect(warmBank.stats.retainedBytes == 0 && coldBank.stats.retainedBytes == 0)
        cold.loopForTesting.onEngineQueueSync {
            #expect(cold.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
        warm.loopForTesting.onEngineQueueSync {
            #expect(warm.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
    }

    /// Restored, partially restored and cold requests speculating together
    /// emit the tokens each emits served alone and cold.
    @Test(.timeLimit(.minutes(5)))
    func restoredRequestsSpeculatingTogetherMatchSoloColdServing() async throws {
        let shared: Int = Self.chunk + 3
        let branch: [Int] =
            Array(Self.donor.prefix(shared)) + [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        let prompts: [[Int]] = [Self.conversation, branch, [11, 13, 17, 19, 23, 29, 31, 37, 2]]
        let saved: [Int] = [2 * Self.chunk, Self.chunk, 0]

        let plain = try Self.plainEngine()
        var solo: [Served] = []
        for (index, prompt) in prompts.enumerated() {
            solo.append(try await Self.serve(plain, Self.request(UInt64(10 + index), prompt)))
        }
        await plain.shutdown()

        let (engine, backend) = Self.engine(try Self.stack(), bank: Prefix.bank())
        let bank = try #require(engine.hybridPrefixCache)
        _ = try await Self.serve(engine, Self.request(20, Self.donor))
        let published = await Prefix.eventually { bank.stats.entries == 1 }
        #expect(published)
        let served = try await withThrowingTaskGroup(of: (Int, Served).self) { group in
            for (index, prompt) in prompts.enumerated() {
                group.addTask {
                    (index, try await Self.serve(engine, Self.request(UInt64(30 + index), prompt)))
                }
            }
            var results: [Int: Served] = [:]
            for try await (index, result) in group { results[index] = result }
            return results
        }
        for index in prompts.indices {
            let result = try #require(served[index])
            #expect(result.tokens == solo[index].tokens, "request \(index)")
            #expect(result.saved == saved[index], "request \(index)")
        }
        let metrics = try #require(engine.mtpMetricsSnapshot())
        #expect(metrics.rectangularVerificationRounds > 0)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }

    /// A restored request cancelled while it speculates releases its
    /// assistant state, its pin and its inherited checkpoints: the donor's
    /// entry stays intact and evictable, and every charge returns to zero.
    /// Each checkpoint charges the assistant history beside the recurrent
    /// state.
    @Test(.timeLimit(.minutes(5)))
    func cancelledRestoredRequestRefundsEveryCharge() async throws {
        let stack = try Self.stack()
        let (engine, backend) = Self.engine(stack, bank: Prefix.bank())
        let bank = try #require(engine.hybridPrefixCache)
        _ = try await Self.serve(engine, Self.request(40, Self.donor))
        let published = await Prefix.eventually { bank.stats.entries == 1 }
        #expect(published)
        let donorBytes = bank.stats.retainedBytes
        let recurrent = try stack.model.cbv2RecurrentStateSpec.fixedBytesPerRequest()
        let twoCheckpoints: Int = 2 * recurrent
        #expect(donorBytes > twoCheckpoints, "assistant history and KV are charged too")

        let restored = try await Self.serve(
            engine, Self.request(41, Self.conversation, maxTokens: 60)
        ) { count in
            if count >= 2 { engine.cancel(CBv2RequestID(41)) }
        }
        #expect(restored.finish == .cancelled)
        let restoredAt: Int = 2 * Self.chunk
        #expect(restored.saved == restoredAt)
        let settled = await Prefix.eventually { bank.stats.retainedBytes == donorBytes }
        #expect(settled)
        #expect(bank.stats.entries == 1)
        let reservation = bank.resizeReservation(to: 0)
        #expect(reservation == 0)
        #expect(bank.stats.retainedBytes == 0)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        engine.loopForTesting.onEngineQueueSync {
            #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
    }

    /// Image requests on the speculating owner run target-only and checkpoint
    /// without assistant history, in the scope of their image identity: the
    /// next turn with the same image restores and emits the cold tokens.
    @Test(.timeLimit(.minutes(5)))
    func targetOnlyImageTurnsRestoreExactly() async throws {
        let imageTokens: Int = 2
        let turn: [Int] =
            [5, 9, 1] + [Int](repeating: Self.placeholder, count: imageTokens)
            + Array(Self.conversation.prefix(2 * Self.chunk + 5 - 3 - imageTokens))
        let nextTurn: [Int] = turn + [11, 13, 17, 19, 23, 29, 31, 37, 41]
        let width: Int = NemotronSuperMTPFixture.hidden
        let count: Int = imageTokens * width
        let values: [Float] = (0 ..< count).map { (index: Int) -> Float in Float(index % 13) / 50 }
        let embedding = MLXArray(values).reshaped([1, imageTokens, width]).asType(.bfloat16)
        let identity = try Prefix.identity(0x4d)
        func request(_ id: UInt64, _ prompt: [Int], bound: Bool) -> CBv2Request {
            var request = Self.request(id, prompt)
            request.multimodal = CBv2MultimodalInput(
                spans: [CBv2ImageSpan(tokenOffset: 3, length: imageTokens)], attention: .causal
            ) { [embedding] }
            request.hybridPrefixIdentity = bound ? identity : nil
            return request
        }

        let plain = try Self.plainEngine()
        let reference = try await Self.serve(plain, request(50, nextTurn, bound: false))
        await plain.shutdown()

        let (engine, backend) = Self.engine(try Self.stack(), bank: Prefix.bank())
        let bank = try #require(engine.hybridPrefixCache)
        _ = try await Self.serve(engine, request(51, turn, bound: true))
        let published = await Prefix.eventually { bank.stats.entries == 1 }
        #expect(published)
        let restored = try await Self.serve(engine, request(52, nextTurn, bound: true))
        let restoredAt: Int = 2 * Self.chunk
        #expect(restored.saved == restoredAt)
        #expect(restored.tokens == reference.tokens)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(bank.stats.retainedBytes == 0)
    }
}
