import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Rows the speculating Super owner cannot draft for (a stop string, a
/// penalty, a logit bias, logprobs, a token constraint) publish the resident
/// checkpoints an unconstrained row publishes, assistant prompt history
/// included, and emit the plain owner's tokens with the cache on or off.
/// Synthetic weights, the owner's fixed-chunk policy.
@Suite(.tags(.integration))
struct NemotronSuperMTPControlledRowCheckpointTests {
    typealias Reuse = NemotronSuperMTPPrefixReuseTests
    typealias Prefix = NemotronSuperPrefixReuseFixture
    typealias Control = CBv2NonSpeculatingControl
    typealias Served = CBv2ControlledServed

    static let chunk: Int = Reuse.chunk

    static func request(
        _ id: UInt64, _ prompt: [Int], _ control: Control?, cached: Bool = true
    ) -> CBv2Request {
        var request = Reuse.request(id, prompt)
        request.prefixCacheEnabled = cached
        control?.apply(to: &request)
        return request
    }

    /// The speculating owner with a resident bank and the production sampler.
    static func engine(
        _ bank: CBv2HybridPrefixCacheConfig = Prefix.bank(),
        sampler: any CBv2StepSampler = CBv2DefaultSampler()
    ) throws -> (EngineV2, CBv2ContiguousKVBackend, CBv2HybridPrefixCache) {
        let (engine, backend) = Reuse.engine(try Reuse.stack(), bank: bank, sampler: sampler)
        return (engine, backend, try #require(engine.hybridPrefixCache))
    }

    /// The serial plain owner: no assistant, no cache.
    static func uncached(_ request: CBv2Request) async throws -> Served {
        let plain = try NemotronSuperMTPEngineTests.plainEngine(
            Reuse.model(), kvDType: .bfloat16, sampler: CBv2DefaultSampler()
        ).0
        let served = try await cbv2ServeControlled(plain, request)
        await plain.shutdown()
        return served
    }

    static func matches(
        _ first: CBv2HybridPrefixCache, _ second: CBv2HybridPrefixCache, prompt: [Int],
        position: Int
    ) -> Bool {
        Prefix.checkpointsMatch(
            first, second, tokens: Array(prompt.prefix(position + 1)), position: position,
            maximumChunkSize: chunk)
    }

    static func idle(_ engine: EngineV2) -> Bool {
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.mtp?.requestStateCountForTesting == 0
        }
    }

    /// A controlled donor emits the plain owner's tokens with capture on and
    /// off, never drafts or stores a carry, and publishes C and 2C equal to an
    /// unconstrained donor's. An identical second request restores 2C.
    @Test(.timeLimit(.minutes(5)), arguments: Control.allCases)
    func controlledDonorPublishesTheUnconstrainedCheckpoints(_ control: Control) async throws {
        let reference = try await Self.uncached(Self.request(1, Reuse.donor, control))
        #expect(reference.finish == .length)

        let (other, _, otherBank) = try Self.engine()
        _ = try await cbv2ServeControlled(other, Self.request(2, Reuse.donor, nil))
        let otherPublished = await Prefix.eventually { otherBank.stats.entries == 1 }
        #expect(otherPublished)

        let (engine, backend, bank) = try Self.engine()
        let loop = engine.loopForTesting
        let carries = NemotronSuperMTPMediaFallbackTests.CarryLog()
        loop.onEngineQueueSync {
            loop.mtp?.carryStoredObserverForTesting = { carries.record($0.id, mediaTokens: nil) }
        }
        var off = Served()
        var on = Served()
        let drafting = try await Reuse.drafting(engine) {
            off = try await cbv2ServeControlled(
                engine, Self.request(3, Reuse.donor, control, cached: false))
            #expect(bank.stats.entries == 0)
            on = try await cbv2ServeControlled(engine, Self.request(4, Reuse.donor, control))
        }
        #expect(off.tokens == reference.tokens)
        #expect(on.tokens == reference.tokens)
        #expect(on.logprobBits == off.logprobBits)
        #expect(on.finish == reference.finish && off.finish == reference.finish)
        #expect(on.saved == 0 && off.saved == 0)
        #expect(drafting == [0, 0, 0])
        #expect(carries.stored.isEmpty)
        #expect(Self.idle(engine))

        let published = await Prefix.eventually { bank.stats.entries == 1 }
        #expect(published)
        for position in [Self.chunk, 2 * Self.chunk] {
            let same = Self.matches(bank, otherBank, prompt: Reuse.donor, position: position)
            #expect(same, "position \(position)")
        }
        let again = try await cbv2ServeControlled(engine, Self.request(5, Reuse.donor, control))
        #expect(again.saved == 2 * Self.chunk)
        #expect(again.tokens == reference.tokens)
        #expect(again.logprobBits == on.logprobBits)
        #expect(again.finish == reference.finish)

        await other.shutdown()
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(bank.stats.retainedBytes == 0)
        #expect(Self.idle(engine))
    }

    /// The next unconstrained turn restores a controlled donor's 2C and
    /// drafts, verifies and emits exactly what it does cold; its restored
    /// state and its next checkpoint (3C) equal the cold request's own.
    @Test(.timeLimit(.minutes(5)), arguments: Control.allCases)
    func unconstrainedTurnRestoresControlledDonorExactly(_ control: Control) async throws {
        let restoredAt: Int = 2 * Self.chunk
        let bank = Prefix.bank(checkpointsPerRequest: 4)
        let reference = try await Self.uncached(Self.request(1, Reuse.conversation, nil))

        let (cold, _, coldBank) = try Self.engine(bank)
        var coldServed = Served()
        let coldDrafting = try await Reuse.drafting(cold) {
            coldServed = try await cbv2ServeControlled(
                cold, Self.request(2, Reuse.conversation, nil))
        }
        #expect(coldServed.tokens == reference.tokens)
        #expect(coldDrafting[0] > 0)

        let (warm, warmBackend, warmBank) = try Self.engine(bank)
        _ = try await cbv2ServeControlled(warm, Self.request(3, Reuse.donor, control))
        let published = await Prefix.eventually { warmBank.stats.entries == 1 }
        #expect(published)
        var restored = Served()
        let warmDrafting = try await Reuse.drafting(warm) {
            restored = try await cbv2ServeControlled(
                warm, Self.request(4, Reuse.conversation, nil))
        }
        #expect(restored.saved == restoredAt)
        #expect(restored.tokens == reference.tokens)
        #expect(warmDrafting == coldDrafting)
        let bothPublished = await Prefix.eventually { warmBank.stats.entries == 2 }
        #expect(bothPublished)
        for position in [restoredAt, 3 * Self.chunk] {
            let same = Self.matches(
                warmBank, coldBank, prompt: Reuse.conversation, position: position)
            #expect(same, "position \(position)")
        }
        await cold.shutdown()
        await warm.shutdown()
        #expect(warmBackend.bytesReserved == 0)
        #expect(warmBank.stats.retainedBytes == 0)
        #expect(Self.idle(warm))
    }

    /// A controlled turn that restores an unconstrained donor's 2C emits its
    /// uncached tokens and publishes its own next checkpoint (3C), equal to
    /// the one an unconstrained request publishes cold.
    @Test(.timeLimit(.minutes(5)), arguments: Control.allCases)
    func controlledTurnRestoresAndPublishesTheNextCheckpoint(_ control: Control) async throws {
        let next: Int = 3 * Self.chunk
        let bank = Prefix.bank(checkpointsPerRequest: 4)
        let reference = try await Self.uncached(Self.request(1, Reuse.conversation, control))

        let (cold, _, coldBank) = try Self.engine(bank)
        _ = try await cbv2ServeControlled(cold, Self.request(2, Reuse.conversation, nil))
        let coldPublished = await Prefix.eventually { coldBank.stats.entries == 1 }
        #expect(coldPublished)

        let (warm, warmBackend, warmBank) = try Self.engine(bank)
        _ = try await cbv2ServeControlled(warm, Self.request(3, Reuse.donor, nil))
        let published = await Prefix.eventually { warmBank.stats.entries == 1 }
        #expect(published)
        var restored = Served()
        let drafting = try await Reuse.drafting(warm) {
            restored = try await cbv2ServeControlled(
                warm, Self.request(4, Reuse.conversation, control))
        }
        #expect(restored.saved == 2 * Self.chunk)
        #expect(restored.tokens == reference.tokens)
        #expect(restored.finish == reference.finish)
        #expect(drafting == [0, 0, 0])
        #expect(Self.idle(warm))
        let bothPublished = await Prefix.eventually { warmBank.stats.entries == 2 }
        #expect(bothPublished)
        let same = Self.matches(warmBank, coldBank, prompt: Reuse.conversation, position: next)
        #expect(same, "position \(next)")
        await cold.shutdown()
        await warm.shutdown()
        #expect(warmBackend.bytesReserved == 0)
        #expect(warmBank.stats.retainedBytes == 0)
    }

    /// A sampled request cannot speculate on an engine whose sampler has no
    /// target-prefix acceptance (the greedy sampler); it publishes C and 2C
    /// equal to a temperature-0 donor's, and its tokens do not change.
    @Test(.timeLimit(.minutes(5)))
    func sampledRowWithoutTargetPrefixAcceptancePublishesToo() async throws {
        func sampled(_ id: UInt64, cached: Bool = true) -> CBv2Request {
            var request = Self.request(id, Reuse.donor, nil, cached: cached)
            request.sampling.temperature = 0.7
            request.sampling.seed = 11
            return request
        }
        let plain = try Reuse.plainEngine()
        let reference = try await cbv2ServeControlled(plain, sampled(1, cached: false))
        await plain.shutdown()

        let (other, _, otherBank) = try Self.engine(sampler: CBv2GreedySampler())
        _ = try await cbv2ServeControlled(other, Self.request(2, Reuse.donor, nil))
        let otherPublished = await Prefix.eventually { otherBank.stats.entries == 1 }
        #expect(otherPublished)

        let (engine, backend, bank) = try Self.engine(sampler: CBv2GreedySampler())
        var off = Served()
        var on = Served()
        let drafting = try await Reuse.drafting(engine) {
            off = try await cbv2ServeControlled(engine, sampled(3, cached: false))
            #expect(bank.stats.entries == 0)
            on = try await cbv2ServeControlled(engine, sampled(4))
        }
        #expect(off.tokens == reference.tokens)
        #expect(on.tokens == reference.tokens)
        #expect(on.finish == reference.finish && off.finish == reference.finish)
        #expect(drafting == [0, 0, 0])
        #expect(Self.idle(engine))

        let published = await Prefix.eventually { bank.stats.entries == 1 }
        #expect(published)
        for position in [Self.chunk, 2 * Self.chunk] {
            let same = Self.matches(bank, otherBank, prompt: Reuse.donor, position: position)
            #expect(same, "position \(position)")
        }
        let again = try await cbv2ServeControlled(engine, sampled(5))
        #expect(again.saved == 2 * Self.chunk)
        #expect(again.tokens == reference.tokens)

        await other.shutdown()
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(bank.stats.retainedBytes == 0)
        #expect(Self.idle(engine))
    }

    /// Controlled donors fill the bank through its ordinary budget: entries
    /// stay within the limit, older ones are evicted, every byte is refunded.
    @Test(.timeLimit(.minutes(5)))
    func controlledDonorsRespectTheBankBudget() async throws {
        let limit: Int = 2
        let config = CBv2HybridPrefixCacheConfig(
            maximumBytes: 64 << 20, maximumEntries: limit,
            modelID: "nemotron-super-fixture", promptContractID: "fixture-template",
            buildID: "fixture-build")
        let (engine, backend, bank) = try Self.engine(config)
        for (index, control) in [Control.stopString, .presencePenalty, .logitBias, .logprobs]
            .enumerated()
        {
            // Distinct prompts that avoid the image placeholder.
            let prompt = Reuse.donor.map { (token: Int) -> Int in
                let shifted: Int = (token + 3 * index) % 61 + 1
                return shifted == Reuse.placeholder ? 62 : shifted
            }
            _ = try await cbv2ServeControlled(
                engine, Self.request(UInt64(10 + index), prompt, control))
            let published = await Prefix.eventually {
                bank.stats.entries == min(index + 1, limit)
                    && bank.stats.evictions == max(0, index + 1 - limit)
            }
            #expect(published, "donor \(index)")
            #expect(bank.stats.retainedBytes <= config.maximumBytes)
        }
        #expect(bank.stats.entries == limit)
        #expect(bank.stats.evictions == 2)
        let reservation = bank.resizeReservation(to: 0)
        #expect(reservation == 0)
        #expect(bank.stats.retainedBytes == 0)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(Self.idle(engine))
    }
}
