import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Bounded ownership of the Super resident checkpoint bank: eviction under
/// its byte cap, no publication and full refund on cancellation, and no
/// reuse of any kind for image requests. Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperPrefixBankLifecycleTests {
    typealias Fixture = NemotronSuperPrefixReuseFixture

    static let chunk: Int = Fixture.chunk

    /// One recurrent checkpoint plus the KV of one chunk, from the loaded
    /// fixture's own shapes and the BF16 KV element width.
    static func entryBytes(_ model: NemotronSuperModel) throws -> (checkpoint: Int, kv: Int) {
        let checkpoint = try model.cbv2RecurrentStateSpec.fixedBytesPerRequest()
        let perToken = model.cbv2LayerKinds.reduce(0) { (total: Int, kind: CBv2LayerKind) -> Int in
            total + 2 * kind.kvHeads * kind.headDim * 2
        }
        return (checkpoint, perToken * chunk)
    }

    /// The cap holds one donor (one checkpoint and its compacted KV) but not
    /// two: the second publication evicts the least recently used donor,
    /// whose prompt then serves cold with the reference output. Retained
    /// bytes never exceed the cap.
    @Test(.timeLimit(.minutes(5)))
    func secondDonorEvictsTheFirstUnderTheByteCap() async throws {
        let model = try Fixture.model()
        let (checkpoint, kv) = try Self.entryBytes(model)
        let cap: Int = checkpoint + kv + checkpoint / 2
        let first = Fixture.prompt(Self.chunk + 40, seed: 5)
        let second = Fixture.prompt(Self.chunk + 40, seed: 9)

        let (plain, _) = try Fixture.engine(model, bank: nil)
        let reference = try await Fixture.serve(plain, Fixture.request(1, first))
        await plain.shutdown()

        let (engine, backend) = try Fixture.engine(model, bank: Fixture.bank(bytes: cap))
        let bank = try #require(engine.hybridPrefixCache)
        _ = try await Fixture.serve(engine, Fixture.request(2, first))
        let firstPublished = await Fixture.eventually { bank.stats.entries == 1 }
        #expect(firstPublished)
        #expect(bank.stats.retainedBytes <= cap)
        _ = try await Fixture.serve(engine, Fixture.request(3, second))
        let evicted = await Fixture.eventually { bank.stats.evictions == 1 }
        #expect(evicted)
        #expect(bank.stats.entries == 1)
        #expect(bank.stats.retainedBytes <= cap)

        let cold = try await Fixture.serve(engine, Fixture.request(4, first))
        #expect(cold.usage?.prefixCachePrefillTokensSaved == 0)
        #expect(cold.tokens == reference.tokens)
        #expect(cold.logprobBits == reference.logprobBits)
        let warm = try await Fixture.serve(engine, Fixture.request(5, first))
        #expect(warm.usage?.prefixCachePrefillTokensSaved == Self.chunk)
        #expect(warm.tokens == reference.tokens)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(bank.stats.retainedBytes == 0)
    }

    /// A donor cancelled after its first checkpoint was staged publishes
    /// nothing and its staged copy is released; an adopter cancelled after
    /// restoring releases its pin and its inherited checkpoints, leaving the
    /// donor's entry intact and evictable. KV and bank bytes return to zero.
    @Test(.timeLimit(.minutes(5)))
    func cancellationNeverPublishesAndRefundsEveryCharge() async throws {
        let model = try Fixture.model()
        let prompt = Fixture.prompt(3 * Self.chunk + 10, seed: 13)
        let (engine, backend) = try Fixture.engine(model, bank: Fixture.bank())
        let bank = try #require(engine.hybridPrefixCache)
        let donorID = CBv2RequestID(30)
        let firstBoundary: Int = Self.chunk
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.recurrentGeometryObserverForTesting = {
                id, range, _, _, phase, outcome in
                guard id == donorID, phase == "record", outcome == "capture",
                    range.upperBound == firstBoundary
                else { return }
                engine.cancel(id)
            }
        }
        let cancelled = try await Fixture.serve(
            engine, Fixture.request(donorID.raw, prompt, maxTokens: 64))
        #expect(cancelled.finish == .cancelled)
        let dropped = await Fixture.eventually { bank.stats.retainedBytes == 0 }
        #expect(dropped)
        #expect(bank.stats.entries == 0)
        #expect(backend.bytesReserved == 0)

        let donor = try await Fixture.serve(engine, Fixture.request(31, prompt))
        #expect(donor.usage?.prefixCacheOutcome == .miss)
        let published = await Fixture.eventually { bank.stats.entries == 1 }
        #expect(published)
        let donorBytes = bank.stats.retainedBytes

        // Two checkpoints per request: the first (C) and the newest (3C).
        let adopterID = CBv2RequestID(32)
        let adopter = prompt + Fixture.prompt(2 * Self.chunk, seed: 17)
        let restoredAt: Int = 3 * Self.chunk
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.recurrentGeometryObserverForTesting = {
                id, range, _, _, phase, _ in
                guard id == adopterID, phase == "record", range.lowerBound >= restoredAt
                else { return }
                engine.cancel(id)
            }
        }
        let restored = try await Fixture.serve(
            engine, Fixture.request(adopterID.raw, adopter, maxTokens: 64))
        #expect(restored.finish == .cancelled)
        #expect(restored.usage?.prefixCachePrefillTokensSaved == restoredAt)
        let settled = await Fixture.eventually { bank.stats.retainedBytes == donorBytes }
        #expect(settled)
        #expect(bank.stats.entries == 1)
        #expect(backend.bytesReserved == 0)
        // No pin survives: shrinking to zero evicts the donor's entry.
        #expect(bank.resizeReservation(to: 0) == 0)
        #expect(bank.stats.retainedBytes == 0)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.recurrentGeometryObserverForTesting = nil
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }

    /// Image requests carry no checkpoint identity: one whose text before
    /// the image equals a published donor's prefix restores nothing, and
    /// neither it nor a later turn repeating its image history publishes or
    /// restores anything. Text reuse is unaffected.
    @Test(.timeLimit(.minutes(5)))
    func imageRequestsAndImageHistoryNeverReuseOrDonate() async throws {
        let model = try Fixture.model()
        let text = Fixture.prompt(2 * Self.chunk + 40, seed: 19)
        let imageAt: Int = 2 * Self.chunk
        let imageTokens: Int = 4
        let placeholder = Int(NemotronSuperBatchFixture.imageToken)
        let firstTurn =
            Array(text.prefix(imageAt)) + [Int](repeating: placeholder, count: imageTokens)
            + Fixture.prompt(30, seed: 23)
        let secondTurn = firstTurn + Fixture.prompt(Self.chunk, seed: 29)
        func image() -> CBv2MultimodalInput {
            CBv2MultimodalInput(
                spans: [CBv2ImageSpan(tokenOffset: imageAt, length: imageTokens)],
                attention: .causal
            ) {
                [MLXArray.ones([1, imageTokens, Fixture.hidden], dtype: .bfloat16) * 0.05]
            }
        }
        let (engine, backend) = try Fixture.engine(model, bank: Fixture.bank())
        let bank = try #require(engine.hybridPrefixCache)
        _ = try await Fixture.serve(engine, Fixture.request(40, text))
        let published = await Fixture.eventually { bank.stats.entries == 1 }
        #expect(published)
        let before = bank.stats

        for (id, prompt) in [(UInt64(41), firstTurn), (UInt64(42), secondTurn)] {
            let served = try await Fixture.serve(
                engine, Fixture.request(id, prompt, multimodal: image()))
            #expect(served.finish == .length)
            #expect(served.usage?.prefixCachePrefillTokensSaved == 0)
            #expect(served.usage?.prefixCacheOutcome != .hit)
        }
        try await Task.sleep(for: .milliseconds(100))
        let after = bank.stats
        #expect(after.entries == before.entries)
        #expect(after.checkpoints == before.checkpoints)
        #expect(after.adoptions == before.adoptions)
        #expect(after.lookupMatches == before.lookupMatches)
        #expect(after.retainedBytes == before.retainedBytes)

        let textRepeat = try await Fixture.serve(engine, Fixture.request(43, text))
        #expect(textRepeat.usage?.prefixCachePrefillTokensSaved == imageAt)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(bank.stats.retainedBytes == 0)
    }
}
