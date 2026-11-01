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

    /// Prompt gathers the bank asked for, as (donor, bytes), recorded on the
    /// engine queue by `hybridPublicationGatherForTesting`.
    final class Gathers: @unchecked Sendable {
        private let lock = NSLock()
        private var records: [(id: CBv2RequestID, bytes: Int)] = []
        func append(_ id: CBv2RequestID, _ bytes: Int) {
            lock.withLock { records.append((id, bytes)) }
        }
        var snapshot: [(id: CBv2RequestID, bytes: Int)] { lock.withLock { records } }
    }

    struct InjectedGatherFailure: Error {}

    /// Paged KV bytes of one prompt token over every attention row.
    static func kvBytesPerToken(_ model: NemotronSuperModel) throws -> Int {
        let language = model.languageModel
        let types = try #require(language.observedAttentionKVDTypes())
        return zip(language.cbv2LayerKinds, types).reduce(0) { total, layer in
            total + layer.0.kvHeads * (layer.0.headDim + layer.0.valueHeadDim) * layer.1.size
        }
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

    /// The bank decides before the engine copies. The donor's kept
    /// publication gathers its prompt KV once, at the bytes the bank charges;
    /// a repeat turn that adds no checkpoint refreshes that entry and
    /// gathers nothing.
    @Test(.timeLimit(.minutes(5)))
    func onlyAKeptPublicationGathersThePromptKV() async throws {
        let model = try Fixture.model()
        let (engine, backend) = try Self.engine(model, bank: Self.pagedBank())
        let bank = try #require(engine.hybridPrefixCache)
        let gathers = Gathers()
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.hybridPublicationGatherForTesting = { gathers.append($0, $1) }
        }
        let donor = try await Fixture.serve(engine, Fixture.request(20, Self.donor))
        let published = await Fixture.eventually {
            bank.stats.entries == 1 && backend.bytesReserved == 0
        }
        #expect(published)
        let promptBytes = try Self.kvBytesPerToken(model) * Self.donor.count
        #expect(gathers.snapshot.map { $0.id } == [CBv2RequestID(20)])
        #expect(gathers.snapshot.map { $0.bytes } == [promptBytes])
        let hit = try #require(
            bank.lookup(tokens: Self.donor, cacheSalt: nil, maximumChunkSize: Self.chunk))
        #expect(hit.kvBackingBytes == promptBytes)
        bank.endAdoption(pin: hit.pin)
        let kept = bank.stats.retainedBytes

        let repeated = try await Fixture.serve(engine, Fixture.request(21, Self.donor))
        #expect(repeated.usage?.prefixCachePrefillTokensSaved == 2 * Self.chunk)
        #expect(repeated.tokens == donor.tokens)
        let settled = await Fixture.eventually {
            backend.bytesReserved == 0 && bank.stats.retainedBytes == kept
        }
        #expect(settled)
        #expect(gathers.snapshot.count == 1, "a refreshed entry copies no prompt KV")
        #expect(bank.stats.entries == 1)
        #expect(bank.stats.capacityRefusals == 0)
        #expect(bank.stats.evictions == 0)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.hybridPublicationGatherForTesting = nil
        }
        await Self.retire(engine, backend)
    }

    /// A gather that fails refuses its publication: the bank keeps nothing,
    /// the donor's pages and every reservation return, and the next request
    /// with the same prompt runs cold, equals the plain paged run bit for bit
    /// and publishes.
    @Test(.timeLimit(.minutes(5)))
    func aFailedGatherRefusesThePublicationAndServingContinues() async throws {
        let model = try Fixture.model()
        let (plain, plainBackend) = try Self.engine(model, bank: nil)
        let reference = try await Fixture.serve(plain, Fixture.request(30, Self.donor))
        await Self.retire(plain, plainBackend)

        let (engine, backend) = try Self.engine(model, bank: Self.pagedBank())
        let bank = try #require(engine.hybridPrefixCache)
        let failing = CBv2RequestID(31)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.hybridPublicationGatherForTesting = { id, _ in
                if id == failing { throw InjectedGatherFailure() }
            }
        }
        let refused = try await Fixture.serve(engine, Fixture.request(failing.raw, Self.donor))
        #expect(refused.finish == .length)
        #expect(refused.tokens == reference.tokens)
        let released = await Fixture.eventually {
            backend.bytesReserved == 0 && bank.stats.retainedBytes == 0
        }
        #expect(released)
        #expect(bank.stats.entries == 0)
        #expect(bank.stats.capacityRefusals == 1)

        let cold = try await Fixture.serve(engine, Fixture.request(32, Self.donor))
        #expect(cold.usage?.prefixCachePrefillTokensSaved == 0)
        #expect(cold.tokens == reference.tokens)
        #expect(cold.logprobBits == reference.logprobBits)
        let published = await Fixture.eventually { bank.stats.entries == 1 }
        #expect(published)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.hybridPublicationGatherForTesting = nil
        }
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
