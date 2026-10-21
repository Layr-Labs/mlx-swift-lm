import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// The resident recurrent bank decides a publication before any KV copy
/// exists: a refreshed or ineligible donor never asks for its rows, a kept or
/// compacting donor asks once and keeps exactly those rows, and a copy that
/// fails refuses the publication and returns its reservation.
@Suite(.tags(.integration))
struct CBv2HybridPublicationDecisionTests {
    typealias Row = (keys: MLXArray, values: MLXArray, offset: Int)

    static let spec = CBv2RecurrentStateSpec(layers: [
        .init(
            modelLayerIndex: 0, convShape: [1, 2, 2], convDType: .float32,
            ssmShape: [1, 1, 2, 2], ssmDType: .float32)
    ])

    static func bank(bytes: Int = 4096) -> CBv2HybridPrefixCache {
        .init(
            config: .init(
                maximumBytes: bytes, maximumEntries: 4, maximumCheckpointsPerRequest: 2,
                modelID: "model", promptContractID: "template", buildID: "test-build"))
    }

    /// One 32-byte checkpoint (float32 conv and SSM state).
    static func stage(_ bank: CBv2HybridPrefixCache, id: UInt64, position: Int) {
        let roots = bank.capture(
            requestID: .init(id), position: position, chunkSize: 4, spec: spec,
            layers: [
                0: .init(conv: MLXArray.zeros([1, 2, 2]), ssm: MLXArray.zeros([1, 1, 2, 2]))
            ])
        if !roots.isEmpty { asyncEval(roots) }
    }

    /// One `[1, 1, count, 2]` float32 row of 0, 1, 2, ..., detached as a
    /// retired request's KV is.
    static func rows(_ count: Int) -> [Row?] {
        let row = MLXArray(0 ..< Int32(2 * count)).asType(.float32).reshaped([1, 1, count, 2])
        asyncEval(row)
        return [(keys: row, values: row, offset: count)]
    }

    /// Publishes through the deciding entry point, waits for completion and
    /// returns how many times the bank asked for the rows.
    static func publish(
        _ bank: CBv2HybridPrefixCache, id: UInt64, tokens: [Int], backingBytes: Int,
        rows: [Row?]?, receipt: UInt64? = nil
    ) async -> Int {
        var asked = 0
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            bank.publish(
                requestID: .init(id), tokens: tokens, cacheSalt: nil,
                receiptID: receipt.map(CBv2RequestID.init), backingBytes: backingBytes,
                kv: {
                    asked += 1
                    return rows
                }
            ) { done.resume() }
        }
        return asked
    }

    @Test func aKeptPublicationAsksOnceAndKeepsTheRows() async throws {
        let bank = Self.bank()
        Self.stage(bank, id: 1, position: 4)
        Self.stage(bank, id: 1, position: 8)
        let asked = await Self.publish(
            bank, id: 1, tokens: Array(0 ..< 12), backingBytes: 192, rows: Self.rows(12))
        #expect(asked == 1)
        #expect(bank.stats.entries == 1)
        #expect(bank.stats.kvCompactions == 0)
        let hit = try #require(
            bank.lookup(tokens: Array(0 ..< 12), cacheSalt: nil, maximumChunkSize: 4))
        #expect(hit.checkpoint.position == 8)
        #expect(hit.kvBackingBytes == 192)
        let prefix = try #require(hit.kvPrefix[0])
        #expect(prefix.keys.asArray(Float.self) == (0 ..< 16).map(Float.init))
        bank.endAdoption(pin: hit.pin)
        bank.close()
        #expect(bank.stats.retainedBytes == 0)
    }

    /// A repeat that adds no checkpoint refreshes its donor's entry.
    @Test func aRefreshedDonorNeverAsksForRows() async throws {
        let bank = Self.bank()
        Self.stage(bank, id: 1, position: 4)
        let first = await Self.publish(
            bank, id: 1, tokens: Array(0 ..< 8), backingBytes: 128, rows: Self.rows(8))
        #expect(first == 1)
        let kept = bank.stats.retainedBytes
        let hit = try #require(
            bank.lookup(tokens: Array(0 ..< 8), cacheSalt: nil, maximumChunkSize: 4))
        bank.inheritCheckpoint(pin: hit.pin, checkpoint: hit.checkpoint, requestID: .init(2))
        bank.endAdoption(pin: hit.pin, tokensSaved: 4)
        let repeated = await Self.publish(
            bank, id: 2, tokens: Array(0 ..< 8), backingBytes: 128, rows: Self.rows(8))
        #expect(repeated == 0)
        #expect(bank.stats.entries == 1)
        #expect(bank.stats.retainedBytes == kept)
        bank.close()
    }

    /// No staged checkpoint (released here by `close`), a checkpoint beyond
    /// the prompt or no backing bytes: nothing to keep, so nothing to copy.
    @Test func anIneligibleDonorNeverAsksForRows() async {
        let closed = Self.bank()
        Self.stage(closed, id: 1, position: 4)
        closed.close()
        let afterClose = await Self.publish(
            closed, id: 1, tokens: Array(0 ..< 8), backingBytes: 128, rows: Self.rows(8))
        #expect(afterClose == 0)

        let bank = Self.bank()
        Self.stage(bank, id: 2, position: 8)
        let beyondPrompt = await Self.publish(
            bank, id: 2, tokens: Array(0 ..< 6), backingBytes: 96, rows: Self.rows(6))
        #expect(beyondPrompt == 0)
        Self.stage(bank, id: 3, position: 4)
        let unbacked = await Self.publish(
            bank, id: 3, tokens: Array(0 ..< 8), backingBytes: 0, rows: Self.rows(8))
        #expect(unbacked == 0)
        #expect(bank.stats.entries == 0)
        #expect(bank.stats.retainedBytes == 0)
        bank.close()
    }

    /// A donor too large for the free budget asks once; the bank keeps its
    /// compact prefix: 128 KV bytes plus two 32-byte checkpoints.
    @Test func aCompactingDonorAsksOnce() async {
        let bank = Self.bank(bytes: 192)
        Self.stage(bank, id: 1, position: 4)
        Self.stage(bank, id: 1, position: 8)
        let asked = await Self.publish(
            bank, id: 1, tokens: Array(0 ..< 12), backingBytes: 192, rows: Self.rows(12))
        #expect(asked == 1)
        #expect(bank.stats.kvCompactions == 1)
        #expect(bank.stats.retainedBytes == 192)
        bank.close()
    }

    /// Publication receipts, recorded on the bank's queue.
    final class Receipts: @unchecked Sendable {
        private let lock = NSLock()
        private var received = 0
        func record() { lock.withLock { received += 1 } }
        var count: Int { lock.withLock { received } }
    }

    /// Kept (4096-byte bank) or compacting (160): a copy that fails publishes
    /// nothing, sends no receipt and returns every reservation.
    @Test(arguments: [4096, 160])
    func aFailedCopyRefusesThePublication(bankBytes: Int) async {
        let bank = Self.bank(bytes: bankBytes)
        let receipts = Receipts()
        bank.setPublicationHandler { _, _ in receipts.record() }
        Self.stage(bank, id: 1, position: 4)
        Self.stage(bank, id: 1, position: 8)
        let asked = await Self.publish(
            bank, id: 1, tokens: Array(0 ..< 12), backingBytes: 192, rows: nil, receipt: 101)
        #expect(asked == 1)
        let stats = bank.stats
        #expect(stats.entries == 0)
        #expect(stats.retainedBytes == 0)
        #expect(stats.capacityRefusals == 1)
        #expect(stats.kvCompactions == 0)
        #expect(receipts.count == 0)
        bank.close()
    }
}
