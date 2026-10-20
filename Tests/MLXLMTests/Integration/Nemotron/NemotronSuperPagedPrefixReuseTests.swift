import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The resident recurrent checkpoint bank over segmented paged KV. A donor's
/// prompt KV is gathered into the bank's own buffers before its pages return
/// to the pool; an adopter writes that prefix into fresh pages and restores
/// the conv/SSM state. Within paged storage the restored request equals a cold
/// paged run bit for bit, even after other requests reuse the donor's pages.
/// Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperPagedPrefixReuseTests {
    typealias Fixture = NemotronSuperPrefixReuseFixture

    static let chunk: Int = Fixture.chunk
    static let conversation: [Int] = Fixture.prompt(3 * chunk + 50, seed: 7)
    static let donor: [Int] = Array(conversation.prefix(2 * chunk + 40))
    /// Same length as the donor, disjoint tokens: takes over freed pages.
    static let unrelated: [Int] = Fixture.prompt(2 * chunk + 40, seed: 59)

    static func pagedBank(permitsPaged: Bool = true) -> CBv2HybridPrefixCacheConfig {
        var bank = Fixture.bank(checkpointsPerRequest: 4)
        bank.permitsPagedStorage = permitsPaged
        return bank
    }

    static func engine(_ model: NemotronSuperModel, bank: CBv2HybridPrefixCacheConfig?) throws
        -> (EngineV2, PagedKVBackend)
    {
        let adapter = try NemotronSuperRecurrentAdapter(
            model, maximumDecodeRows: 1, decodeProducts: .rowExact,
            exactPrefixCheckpoints: bank != nil)
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk + 1,
            prefillChunkSize: chunk, maxConcurrentPartialPrefills: 1,
            mixedStepPrefillTokenCap: chunk, maxWaiting: 8, enablePrefixCache: bank != nil)
        config.maxPrefillRowsPerStep = 1
        let storage = try NemotronSuperPagedStorage(
            model.languageModel, maxPrefillChunk: chunk, maximumLength: 1024)
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(adapter),
            layerKinds: adapter.cbv2LayerKinds, backend: storage.backend,
            cacheProvider: CBv2LayerCacheBank(caches: storage.caches),
            sampler: CBv2DefaultSampler(), schedulerConfig: config,
            admissionConfig: .init(watermarkFraction: 0), hybridPrefixCache: bank)
        return (engine, storage.backend)
    }

    static func retire(_ engine: EngineV2, _ backend: PagedKVBackend) async {
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(backend.bytesWired == 0)
    }

    /// The donor publishes C and 2C; an unrelated request then takes over the
    /// donor's freed pages; the next turn restores 2C and its tokens and
    /// full-vocabulary log-softmax equal a cold paged run's bit for bit. The
    /// restored request's checkpoint at 3C equals the cold paged bank's.
    @Test(.timeLimit(.minutes(5)))
    func restoredPrefixOnPagesContinuesTheColdPagedRunBitForBit() async throws {
        let model = try Fixture.model()
        let adopter = Self.conversation
        let restoredAt: Int = 2 * Self.chunk
        let nextBoundary: Int = 3 * Self.chunk

        let (plain, plainBackend) = try Self.engine(model, bank: nil)
        let reference = try await Fixture.serve(plain, Fixture.request(1, adopter))
        await Self.retire(plain, plainBackend)
        #expect(reference.finish == .length)

        let (cold, coldBackend) = try Self.engine(model, bank: Self.pagedBank())
        let coldBank = try #require(cold.hybridPrefixCache)
        let coldServed = try await Fixture.serve(cold, Fixture.request(2, adopter))
        #expect(coldServed.tokens == reference.tokens)
        #expect(coldServed.logprobBits == reference.logprobBits)

        let (warm, warmBackend) = try Self.engine(model, bank: Self.pagedBank())
        let warmBank = try #require(warm.hybridPrefixCache)
        _ = try await Fixture.serve(warm, Fixture.request(3, Self.donor))
        let published = await Fixture.eventually {
            warmBank.stats.entries == 1 && warmBackend.bytesReserved == 0
        }
        #expect(published, "the bank holds the donor and its pages are back in the pool")
        _ = try await Fixture.serve(warm, Fixture.request(4, Self.unrelated))
        let restored = try await Fixture.serve(warm, Fixture.request(5, adopter))
        #expect(restored.usage?.prefixCacheOutcome == .hit)
        #expect(restored.usage?.prefixCacheTier == .resident)
        #expect(restored.usage?.prefixCachePrefillTokensSaved == restoredAt)
        #expect(restored.tokens == reference.tokens)
        #expect(restored.logprobBits == reference.logprobBits)
        #expect(warmBank.stats.adoptions == 1)
        #expect(warmBank.stats.tokensSaved == restoredAt)

        let settled = await Fixture.eventually {
            warmBank.stats.entries >= 2 && coldBank.stats.entries >= 1
        }
        #expect(settled)
        let atNext = Array(adopter.prefix(nextBoundary + 1))
        #expect(
            Fixture.checkpointsMatch(warmBank, coldBank, tokens: atNext, position: nextBoundary))

        await Self.retire(cold, coldBackend)
        await Self.retire(warm, warmBackend)
        #expect(warmBank.stats.retainedBytes == 0)
        #expect(coldBank.stats.retainedBytes == 0)
    }

    /// A restored request cancelled mid-decode returns its pages and pin; the
    /// donor's entry stays intact and every charge reaches zero.
    @Test(.timeLimit(.minutes(5)))
    func cancelledRestoredPagedRequestRefundsEveryCharge() async throws {
        let model = try Fixture.model()
        let (engine, backend) = try Self.engine(model, bank: Self.pagedBank())
        let bank = try #require(engine.hybridPrefixCache)
        _ = try await Fixture.serve(engine, Fixture.request(10, Self.donor))
        let published = await Fixture.eventually { bank.stats.entries == 1 }
        #expect(published)
        let donorBytes = bank.stats.retainedBytes
        #expect(donorBytes > 0)

        let restored = try await Fixture.serve(
            engine, Fixture.request(11, Self.conversation, maxTokens: 40)
        ) { count in
            if count >= 2 { engine.cancel(CBv2RequestID(11)) }
        }
        #expect(restored.finish == .cancelled)
        #expect(restored.usage?.prefixCachePrefillTokensSaved == 2 * Self.chunk)
        let refunded = await Fixture.eventually {
            bank.stats.retainedBytes == donorBytes && backend.bytesReserved == 0
        }
        #expect(refunded)
        #expect(bank.stats.entries == 1)
        #expect(bank.resizeReservation(to: 0) == 0)
        #expect(bank.stats.retainedBytes == 0)
        await Self.retire(engine, backend)
    }

    /// Without the opt-in a paged engine keeps no bank, as before.
    @Test func pagedStorageKeepsNoBankWithoutTheOptIn() async throws {
        let model = try Fixture.model()
        let (engine, backend) = try Self.engine(model, bank: Self.pagedBank(permitsPaged: false))
        #expect(engine.hybridPrefixCache == nil)
        await Self.retire(engine, backend)
        let (opted, optedBackend) = try Self.engine(model, bank: Self.pagedBank())
        #expect(opted.hybridPrefixCache != nil)
        await Self.retire(opted, optedBackend)
    }
}
