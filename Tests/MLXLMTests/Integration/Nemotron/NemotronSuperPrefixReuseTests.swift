import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Exact text prefix reuse through EngineV2's resident recurrent checkpoint
/// bank. Tiled prefill output depends on the chunk partition, so the Super
/// adapter checkpoints only at multiples of the fixed prefill chunk C reached
/// by whole chunks. A cold prompt runs `[0, C), [C, 2C), ...`; an adopter
/// restored at `kC` runs `[kC, (k+1)C), ...`: both partitions are the same
/// chunk grid, so the restored request computes exactly what the cold one
/// does. Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperPrefixReuseTests {
    typealias Fixture = NemotronSuperPrefixReuseFixture

    static let chunk: Int = Fixture.chunk
    /// Shared conversation: the donor is its first turn, the adopter extends it.
    static let conversation: [Int] = Fixture.prompt(3 * chunk + 50, seed: 7)
    static let donorLength: Int = 2 * chunk + 40

    /// The donor publishes checkpoints at C and 2C (its prompt end is off the
    /// grid). The next turn restores 2C and its tokens and full-vocabulary
    /// log-softmax at every position equal a cold run's bit for bit. The
    /// restored conv/SSM state and KV equal the cold run's own state at 2C,
    /// and the restored request's next checkpoint (3C) equals the cold one.
    @Test(.timeLimit(.minutes(5)))
    func restoredPrefixContinuesTheColdPrefillBitForBit() async throws {
        let model = try Fixture.model()
        let adopter = Self.conversation
        let donor = Array(adopter.prefix(Self.donorLength))
        let restoredAt: Int = 2 * Self.chunk
        let nextBoundary: Int = 3 * Self.chunk

        let (plain, plainBackend) = try Fixture.engine(model, bank: nil)
        let reference = try await Fixture.serve(plain, Fixture.request(1, adopter))
        await plain.shutdown()
        #expect(plainBackend.bytesReserved == 0)
        #expect(reference.finish == .length)
        #expect(reference.logprobBits.count == reference.tokens.count)

        // Every boundary kept, so the cold bank holds its own state at 2C.
        let bank = Fixture.bank(checkpointsPerRequest: 4)
        let (cold, coldBackend) = try Fixture.engine(model, bank: bank)
        let coldServed = try await Fixture.serve(cold, Fixture.request(2, adopter))
        #expect(coldServed.tokens == reference.tokens)
        #expect(coldServed.logprobBits == reference.logprobBits)
        #expect(coldServed.usage?.prefixCachePrefillTokensSaved == 0)

        let (warm, warmBackend) = try Fixture.engine(model, bank: bank)
        let donorServed = try await Fixture.serve(warm, Fixture.request(3, donor))
        #expect(donorServed.finish == .length)
        let warmBank = try #require(warm.hybridPrefixCache)
        let published = await Fixture.eventually { warmBank.stats.entries == 1 }
        #expect(published)
        let restored = try await Fixture.serve(warm, Fixture.request(4, adopter))
        #expect(restored.usage?.prefixCacheOutcome == .hit)
        #expect(restored.usage?.prefixCacheTier == .resident)
        #expect(restored.usage?.prefixCachePrefillTokensSaved == restoredAt)
        #expect(restored.tokens == reference.tokens)
        #expect(restored.logprobBits == reference.logprobBits)
        #expect(warmBank.stats.adoptions == 1)
        #expect(warmBank.stats.tokensSaved == restoredAt)

        let coldBank = try #require(cold.hybridPrefixCache)
        let bothPublished = await Fixture.eventually { warmBank.stats.entries == 2 }
        #expect(bothPublished)
        let atRestore = Array(adopter.prefix(restoredAt + 1))
        let atNext = Array(adopter.prefix(nextBoundary + 1))
        #expect(
            Fixture.checkpointsMatch(warmBank, coldBank, tokens: atRestore, position: restoredAt))
        #expect(
            Fixture.checkpointsMatch(warmBank, coldBank, tokens: atNext, position: nextBoundary))

        await cold.shutdown()
        await warm.shutdown()
        #expect(coldBackend.bytesReserved == 0)
        #expect(warmBackend.bytesReserved == 0)
        #expect(warmBank.stats.retainedBytes == 0)
        #expect(coldBank.stats.retainedBytes == 0)
    }

    /// A request sharing only `[0, C + 30)` with the donor restores the
    /// first-branch checkpoint C and matches its cold run; a byte-identical
    /// repeat restores the deepest checkpoint below its last token; another
    /// cache scope never sees the entry.
    @Test(.timeLimit(.minutes(5)))
    func partialPrefixRepeatAndScopeHitsAreExact() async throws {
        let model = try Fixture.model()
        let donor = Array(Self.conversation.prefix(Self.donorLength))
        let shared: Int = Self.chunk + 30
        // Fixture streams are one cycle at different offsets; reversing gives
        // a continuation that leaves the donor at `shared`.
        let branch = Array(donor.prefix(shared)) + Fixture.prompt(Self.chunk, seed: 41).reversed()
        #expect(branch[shared] != donor[shared])
        let firstBoundary: Int = Self.chunk
        let lastBoundary: Int = 2 * Self.chunk

        let (plain, _) = try Fixture.engine(model, bank: nil)
        let branchReference = try await Fixture.serve(plain, Fixture.request(10, branch))
        let donorReference = try await Fixture.serve(plain, Fixture.request(11, donor))
        await plain.shutdown()

        let (warm, backend) = try Fixture.engine(model, bank: Fixture.bank())
        let bank = try #require(warm.hybridPrefixCache)
        _ = try await Fixture.serve(warm, Fixture.request(12, donor, salt: "tenant-a"))
        let published = await Fixture.eventually { bank.stats.entries == 1 }
        #expect(published)

        let other = try await Fixture.serve(warm, Fixture.request(13, branch, salt: "tenant-b"))
        #expect(other.usage?.prefixCachePrefillTokensSaved == 0)
        #expect(other.tokens == branchReference.tokens)

        let partial = try await Fixture.serve(warm, Fixture.request(14, branch, salt: "tenant-a"))
        #expect(partial.usage?.prefixCacheOutcome == .hit)
        #expect(partial.usage?.prefixCachePrefillTokensSaved == firstBoundary)
        #expect(partial.tokens == branchReference.tokens)
        #expect(partial.logprobBits == branchReference.logprobBits)

        let repeated = try await Fixture.serve(warm, Fixture.request(15, donor, salt: "tenant-a"))
        #expect(repeated.usage?.prefixCachePrefillTokensSaved == lastBoundary)
        #expect(repeated.tokens == donorReference.tokens)
        #expect(repeated.logprobBits == donorReference.logprobBits)

        await warm.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(bank.stats.retainedBytes == 0)
    }

    /// Off-grid range ends are never captured. With a 384-token chunk a
    /// 512-token prompt ends on a 256-token block (a boundary of the
    /// chunk-agnostic stride rule) that no cold run under this chunking has:
    /// only the grid boundary 384 is published.
    @Test(.timeLimit(.minutes(3)))
    func offGridPromptEndsAreNeverCaptured() async throws {
        let model = try Fixture.model()
        let chunk: Int = 384
        let (engine, backend) = try Fixture.engine(model, bank: Fixture.bank(), chunk: chunk)
        let bank = try #require(engine.hybridPrefixCache)
        final class Positions: @unchecked Sendable {
            private let lock = NSLock()
            private var values: [[Int]] = []
            func append(_ positions: [Int]) { lock.withLock { values.append(positions) } }
            var snapshot: [[Int]] { lock.withLock { values } }
        }
        let positions = Positions()
        engine.setResidentPrefixPublicationHandler { positions.append($1) }
        var request = Fixture.request(20, Fixture.prompt(512, seed: 3))
        request.prefixCacheReceiptID = CBv2RequestID(2020)
        let served = try await Fixture.serve(engine, request)
        #expect(served.finish == .length)
        let published = await Fixture.eventually { !positions.snapshot.isEmpty }
        #expect(published)
        let grid: [Int] = [chunk]
        #expect(positions.snapshot.last == grid)
        #expect(bank.stats.checkpoints == 1)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }
}
