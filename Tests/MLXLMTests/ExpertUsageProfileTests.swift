// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Tests for the expert-usage frequency profile and background cache
/// warming (ExpertUsageProfile.swift / ExpertCacheWarmer.swift):
///  (a) checkpoint-identity keying (path + total shard bytes)
///  (b) profile persistence round-trip (save -> load-merge, with decay)
///  (c) warm-order computation from a synthetic profile (pure function)
///  (d) ExpertCacheWarmer: actually warms the cache, respects the byte
///      budget, inserts at the cold end (evicted before real entries),
///      dedupes against already-resident (foreground-fetched) experts,
///      and is disable-able.
final class ExpertUsageProfileTests: XCTestCase {

    private func makeTempDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv4-expert-profile-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - (a) Checkpoint identity

    func testCheckpointIdentitySumsOnlySafetensorsShardBytes() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try Data(repeating: 0xAB, count: 100).write(to: dir.appendingPathComponent("model-00001.safetensors"))
        try Data(repeating: 0xCD, count: 50).write(to: dir.appendingPathComponent("model-00002.safetensors"))
        // Non-shard files must not affect identity (config.json can be
        // legitimately touched up without the weights changing).
        try Data(repeating: 0x00, count: 999).write(to: dir.appendingPathComponent("config.json"))

        let identity = ExpertUsageProfile.checkpointIdentity(modelDirectory: dir)
        XCTAssertEqual(identity.totalBytes, 150)
        XCTAssertEqual(identity.path, dir.standardizedFileURL.path)
    }

    func testCheckpointIdentityDiffersWhenShardBytesChange() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try Data(repeating: 0xAB, count: 100).write(to: dir.appendingPathComponent("model.safetensors"))
        let before = ExpertUsageProfile.checkpointIdentity(modelDirectory: dir)

        // Simulate a re-downloaded/different checkpoint at the SAME path.
        try Data(repeating: 0xAB, count: 200).write(to: dir.appendingPathComponent("model.safetensors"))
        let after = ExpertUsageProfile.checkpointIdentity(modelDirectory: dir)

        XCTAssertNotEqual(before, after, "identity must change when the checkpoint's bytes change")
    }

    func testDefaultProfileURLIsStableAndDistinctPerIdentity() {
        let a = ExpertUsageProfile.CheckpointIdentity(path: "/models/a", totalBytes: 100)
        let b = ExpertUsageProfile.CheckpointIdentity(path: "/models/b", totalBytes: 100)
        let aAgain = ExpertUsageProfile.CheckpointIdentity(path: "/models/a", totalBytes: 100)

        let urlA = ExpertUsageProfile.defaultProfileURL(for: a)
        let urlB = ExpertUsageProfile.defaultProfileURL(for: b)
        let urlAAgain = ExpertUsageProfile.defaultProfileURL(for: aAgain)

        XCTAssertEqual(urlA, urlAAgain, "same identity must resolve to the same file, deterministically")
        XCTAssertNotEqual(urlA, urlB, "different identities must not collide")
        XCTAssertTrue(urlA.path.contains("darkbloom/expert-profile"))
        XCTAssertEqual(urlA.pathExtension, "json")
    }

    // MARK: - (b) Persistence round-trip + decay-on-merge

    func testFlushWritesReadableProfileAndLoadMergedRecoversIt() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let profileURL = dir.appendingPathComponent("profile.json")
        let identity = ExpertUsageProfile.CheckpointIdentity(path: "/fake/model", totalBytes: 12345)

        let profile = ExpertUsageProfile(
            identity: identity, profileURL: profileURL, numExperts: 8, totalLayers: 4)
        profile.record(layer: 0, groups: [(expert: 1, range: 0 ..< 5), (expert: 2, range: 5 ..< 7)])
        profile.record(layer: 1, groups: [(expert: 1, range: 0 ..< 3)])
        profile.flush()

        XCTAssertTrue(FileManager.default.fileExists(atPath: profileURL.path))

        // Load with decay=1.0 (no decay) so the round-trip is exact.
        let reloaded = ExpertUsageProfile.loadMerged(
            identity: identity, profileURL: profileURL, numExperts: 8, totalLayers: 4,
            historyDecay: 1.0)
        let counts = reloaded.snapshotCounts()
        XCTAssertEqual(counts[ExpertCache.Key(layer: 0, expert: 1)], 5)
        XCTAssertEqual(counts[ExpertCache.Key(layer: 0, expert: 2)], 2)
        XCTAssertEqual(counts[ExpertCache.Key(layer: 1, expert: 1)], 3)
    }

    func testLoadMergedAppliesHistoryDecay() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let profileURL = dir.appendingPathComponent("profile.json")
        let identity = ExpertUsageProfile.CheckpointIdentity(path: "/fake/model", totalBytes: 999)

        let profile = ExpertUsageProfile(
            identity: identity, profileURL: profileURL, numExperts: 8, totalLayers: 4)
        profile.record(layer: 0, groups: [(expert: 0, range: 0 ..< 100)])
        profile.flush()

        let reloaded = ExpertUsageProfile.loadMerged(
            identity: identity, profileURL: profileURL, numExperts: 8, totalLayers: 4,
            historyDecay: 0.5)
        XCTAssertEqual(
            reloaded.snapshotCounts()[ExpertCache.Key(layer: 0, expert: 0)], 50,
            "loaded counts must be scaled by historyDecay before becoming this session's baseline")
    }

    func testLoadMergedDiscardsProfileOnCheckpointIdentityMismatch() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let profileURL = dir.appendingPathComponent("profile.json")
        let identityA = ExpertUsageProfile.CheckpointIdentity(path: "/fake/model", totalBytes: 100)
        let identityB = ExpertUsageProfile.CheckpointIdentity(path: "/fake/model", totalBytes: 200)

        let profile = ExpertUsageProfile(
            identity: identityA, profileURL: profileURL, numExperts: 8, totalLayers: 4)
        profile.record(layer: 0, groups: [(expert: 0, range: 0 ..< 10)])
        profile.flush()

        // Different total byte count at the same path -- must NOT merge in
        // the stale counts (a swapped checkpoint's history is meaningless
        // for the new one).
        let reloaded = ExpertUsageProfile.loadMerged(
            identity: identityB, profileURL: profileURL, numExperts: 8, totalLayers: 4)
        XCTAssertTrue(
            reloaded.snapshotCounts().isEmpty,
            "a checkpoint-identity mismatch must discard the on-disk profile entirely")
    }

    func testLoadMergedWithNoExistingFileStartsEmpty() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let profileURL = dir.appendingPathComponent("does-not-exist.json")
        let identity = ExpertUsageProfile.CheckpointIdentity(path: "/fake/model", totalBytes: 1)

        let profile = ExpertUsageProfile.loadMerged(
            identity: identity, profileURL: profileURL, numExperts: 8, totalLayers: 4)
        XCTAssertTrue(profile.snapshotCounts().isEmpty)
    }

    /// Regression: `record` must debounce saves by call count, not save on
    /// every single call (would be an fsync storm at decode speed).
    func testRecordDoesNotSaveBeforePersistThreshold() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let profileURL = dir.appendingPathComponent("profile.json")
        let identity = ExpertUsageProfile.CheckpointIdentity(path: "/fake/model", totalBytes: 1)

        let profile = ExpertUsageProfile(
            identity: identity, profileURL: profileURL, numExperts: 8, totalLayers: 4,
            persistEveryNCalls: 1000)
        for _ in 0 ..< 10 {
            profile.record(layer: 0, groups: [(expert: 0, range: 0 ..< 1)])
        }
        // Give any (unexpected) background save a moment to land, then
        // assert the file still doesn't exist -- 10 << 1000 calls.
        Thread.sleep(forTimeInterval: 0.05)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: profileURL.path),
            "must not persist before persistEveryNCalls is reached")
        XCTAssertEqual(profile.totalForwardCalls, 10)
    }

    func testRecordSavesAfterPersistThreshold() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let profileURL = dir.appendingPathComponent("profile.json")
        let identity = ExpertUsageProfile.CheckpointIdentity(path: "/fake/model", totalBytes: 1)

        let profile = ExpertUsageProfile(
            identity: identity, profileURL: profileURL, numExperts: 8, totalLayers: 4,
            persistEveryNCalls: 3)
        for _ in 0 ..< 3 {
            profile.record(layer: 0, groups: [(expert: 0, range: 0 ..< 1)])
        }

        let deadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: profileURL.path), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: profileURL.path),
            "debounced background save must fire once the threshold is crossed")
    }

    // MARK: - (c) warmOrder: pure computation

    func testWarmOrderRanksByGlobalFrequencyDescending() {
        let counts: [ExpertCache.Key: Int] = [
            ExpertCache.Key(layer: 0, expert: 0): 10,
            ExpertCache.Key(layer: 1, expert: 5): 100,
            ExpertCache.Key(layer: 2, expert: 9): 50,
        ]
        let order = ExpertUsageProfile.warmOrder(counts: counts, byteBudget: 1_000_000, bytesPerExpert: 1)
        XCTAssertEqual(
            order,
            [
                ExpertCache.Key(layer: 1, expert: 5),
                ExpertCache.Key(layer: 2, expert: 9),
                ExpertCache.Key(layer: 0, expert: 0),
            ], "must rank by count descending across ALL layers combined, not per-layer")
    }

    func testWarmOrderStopsAtByteBudget() {
        let counts: [ExpertCache.Key: Int] = [
            ExpertCache.Key(layer: 0, expert: 0): 10,
            ExpertCache.Key(layer: 0, expert: 1): 9,
            ExpertCache.Key(layer: 0, expert: 2): 8,
        ]
        // Budget for exactly 2 experts at 100 bytes each.
        let order = ExpertUsageProfile.warmOrder(counts: counts, byteBudget: 200, bytesPerExpert: 100)
        XCTAssertEqual(order.count, 2, "must stop once the byte budget is exhausted")
        XCTAssertEqual(order, [ExpertCache.Key(layer: 0, expert: 0), ExpertCache.Key(layer: 0, expert: 1)])
    }

    func testWarmOrderTiesBrokenDeterministically() {
        let counts: [ExpertCache.Key: Int] = [
            ExpertCache.Key(layer: 2, expert: 0): 5,
            ExpertCache.Key(layer: 1, expert: 9): 5,
            ExpertCache.Key(layer: 1, expert: 3): 5,
        ]
        let order = ExpertUsageProfile.warmOrder(counts: counts, byteBudget: 1_000_000, bytesPerExpert: 1)
        // Tie-break: lower layer first, then lower expert -- reproducible
        // regardless of Dictionary iteration order.
        XCTAssertEqual(
            order,
            [
                ExpertCache.Key(layer: 1, expert: 3),
                ExpertCache.Key(layer: 1, expert: 9),
                ExpertCache.Key(layer: 2, expert: 0),
            ])
    }

    func testWarmOrderEmptyWhenBudgetTooSmallForOneExpert() {
        let counts: [ExpertCache.Key: Int] = [ExpertCache.Key(layer: 0, expert: 0): 10]
        XCTAssertTrue(
            ExpertUsageProfile.warmOrder(counts: counts, byteBudget: 50, bytesPerExpert: 100).isEmpty)
    }

    func testWarmOrderEmptyWhenCountsEmpty() {
        XCTAssertTrue(
            ExpertUsageProfile.warmOrder(counts: [:], byteBudget: 1_000_000, bytesPerExpert: 100).isEmpty)
    }

    // MARK: - (d) ExpertCacheWarmer

    /// Build a tiny on-disk checkpoint with `numLayers` layers x
    /// `numExperts` experts, each expert's stacked tensors sized so
    /// `ExpertShardStore.fetch` reports a known, uniform byte count —
    /// makes budget-respecting assertions exact.
    private func makeTinyCheckpoint(numExperts: Int, numLayers: Int) throws -> (
        dir: URL, store: ExpertShardStore
    ) {
        let dir = try makeTempDirectory()
        MLXRandom.seed(31)
        func mkStack() -> MLXArray {
            MLXRandom.randInt(low: 0, high: 1000, [numExperts, 8, 8], type: UInt32.self)
        }
        var arrays: [String: MLXArray] = [:]
        for layer in 0 ..< numLayers {
            let prefix = "model.layers.\(layer).ffn.switch_mlp"
            arrays["\(prefix).gate_proj.weight"] = mkStack()
            arrays["\(prefix).gate_proj.scales"] = mkStack()
            arrays["\(prefix).up_proj.weight"] = mkStack()
            arrays["\(prefix).up_proj.scales"] = mkStack()
            arrays["\(prefix).down_proj.weight"] = mkStack()
            arrays["\(prefix).down_proj.scales"] = mkStack()
        }
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))
        let layout = try SafetensorsLayout.load(modelDirectory: dir)
        return (dir, ExpertShardStore(layout: layout, numExperts: numExperts))
    }

    private func makeProfile(
        counts: [ExpertCache.Key: Int], numExperts: Int, totalLayers: Int
    ) -> ExpertUsageProfile {
        let identity = ExpertUsageProfile.CheckpointIdentity(path: "/fake", totalBytes: 1)
        return ExpertUsageProfile(
            identity: identity, profileURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("unused-\(UUID().uuidString).json"),
            numExperts: numExperts, totalLayers: totalLayers, initialCounts: counts)
    }

    func testWarmerFetchesHighestFrequencyExpertsFirstAndRespectsBudget() throws {
        let (dir, store) = try makeTinyCheckpoint(numExperts: 8, numLayers: 2)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Determine one expert's real byte size first so the budget math
        // below is exact rather than guessed.
        let probe = try store.fetch(layerIndex: 0, experts: [0])
        let perExpertBytes = probe[0]!.byteCount

        let counts: [ExpertCache.Key: Int] = [
            ExpertCache.Key(layer: 0, expert: 3): 100,  // hottest
            ExpertCache.Key(layer: 1, expert: 5): 80,
            ExpertCache.Key(layer: 0, expert: 1): 60,
            ExpertCache.Key(layer: 1, expert: 2): 40,  // coldest -- should NOT get warmed
        ]
        let profile = makeProfile(counts: counts, numExperts: 8, totalLayers: 2)

        // Budget fits exactly 3 experts (with 80% target fraction and 4 candidates).
        let cache = ExpertCache(byteBudget: perExpertBytes * 3)
        let warmer = ExpertCacheWarmer(
            cache: cache, store: store, profile: profile, targetFraction: 1.0, maxInFlight: 2,
            stopAfterForegroundCalls: 1_000_000, enabled: true)

        warmer.start()
        warmer.waitUntilFinishedForTesting()

        XCTAssertTrue(cache.contains(layer: 0, expert: 3), "hottest expert must be warmed")
        XCTAssertTrue(cache.contains(layer: 1, expert: 5), "second-hottest expert must be warmed")
        XCTAssertTrue(cache.contains(layer: 0, expert: 1), "third-hottest expert must be warmed")
        XCTAssertFalse(
            cache.contains(layer: 1, expert: 2),
            "coldest expert must NOT be warmed once the (byte-limited) candidate list is exhausted")
        XCTAssertLessThanOrEqual(cache.stats.residentBytes, cache.currentByteBudget)

        let stats = warmer.stats
        XCTAssertTrue(stats.finished)
        XCTAssertEqual(stats.warmed, 3)
    }

    func testWarmerSkipsExpertsAlreadyResidentFromForegroundTraffic() throws {
        let (dir, store) = try makeTinyCheckpoint(numExperts: 8, numLayers: 2)
        defer { try? FileManager.default.removeItem(at: dir) }

        let counts: [ExpertCache.Key: Int] = [
            ExpertCache.Key(layer: 0, expert: 3): 100,
            ExpertCache.Key(layer: 1, expert: 5): 80,
        ]
        let profile = makeProfile(counts: counts, numExperts: 8, totalLayers: 2)
        let cache = ExpertCache(byteBudget: 1_000_000_000)

        // Simulate a real (foreground) request already having fetched the
        // hottest expert before warming starts.
        _ = try cache.fetch(layer: 0, experts: [3], from: store)
        XCTAssertEqual(cache.stats.misses, 1)

        let warmer = ExpertCacheWarmer(
            cache: cache, store: store, profile: profile, targetFraction: 1.0, maxInFlight: 2,
            stopAfterForegroundCalls: 1_000_000, enabled: true)
        warmer.start()
        warmer.waitUntilFinishedForTesting()

        XCTAssertTrue(cache.contains(layer: 0, expert: 3))
        XCTAssertTrue(cache.contains(layer: 1, expert: 5))
        // The foreground-fetched expert must be skipped, not re-fetched.
        XCTAssertEqual(
            warmer.stats.skippedAlreadyResident, 1,
            "already-resident (foreground) expert must be dedupe-skipped by the warmer")
        XCTAssertEqual(warmer.stats.warmed, 1, "only the genuinely cold expert should be warmed")
    }

    func testWarmInsertsAreEvictedBeforeRealEntriesUnderPressure() throws {
        let (dir, store) = try makeTinyCheckpoint(numExperts: 8, numLayers: 1)
        defer { try? FileManager.default.removeItem(at: dir) }
        let probe = try store.fetch(layerIndex: 0, experts: [0])
        let perExpertBytes = probe[0]!.byteCount

        let counts: [ExpertCache.Key: Int] = [
            ExpertCache.Key(layer: 0, expert: 1): 100,
            ExpertCache.Key(layer: 0, expert: 2): 90,
        ]
        let profile = makeProfile(counts: counts, numExperts: 8, totalLayers: 1)
        // Budget for exactly 2 experts.
        let cache = ExpertCache(byteBudget: perExpertBytes * 2)

        let warmer = ExpertCacheWarmer(
            cache: cache, store: store, profile: profile, targetFraction: 1.0, maxInFlight: 1,
            stopAfterForegroundCalls: 1_000_000, enabled: true)
        warmer.start()
        warmer.waitUntilFinishedForTesting()
        XCTAssertEqual(cache.stats.residentCount, 2, "both warm candidates fit the budget")

        // Now real (foreground) traffic fetches a THIRD, different expert
        // -- this must evict a WARM entry, never force out anything that
        // was already real (there isn't any real entry yet, but the point
        // is the warm entries must be the ones sacrificed).
        _ = try cache.fetch(layer: 0, experts: [7], from: store)
        XCTAssertTrue(cache.contains(layer: 0, expert: 7), "real fetch must succeed")
        XCTAssertEqual(cache.stats.residentCount, 2, "budget-for-2 cache must have evicted one warm entry")
        // The lower-priority (later-warmed) candidate must be the one
        // evicted -- see ExpertCache.insertAtColdEnd's ordering rationale.
        XCTAssertTrue(
            cache.contains(layer: 0, expert: 1),
            "the HIGHEST-frequency warmed expert must survive longest among the warm cohort")
        XCTAssertFalse(cache.contains(layer: 0, expert: 2), "the lower-priority warmed expert is sacrificed")
    }

    /// Regression: once a warm-inserted entry is actually used by real
    /// traffic, it must be promoted like any organic hit (protected from
    /// eviction going forward), not remain pinned to the cold end forever.
    func testWarmEntryPromotedToMRUOnRealUse() throws {
        let (dir, store) = try makeTinyCheckpoint(numExperts: 8, numLayers: 1)
        defer { try? FileManager.default.removeItem(at: dir) }
        let probe = try store.fetch(layerIndex: 0, experts: [0])
        let perExpertBytes = probe[0]!.byteCount

        let cache = ExpertCache(byteBudget: perExpertBytes * 2)
        let fetched = try store.fetch(layerIndex: 0, experts: [1, 2])
        cache.insertAtColdEnd(layer: 0, expert: 1, weights: fetched[1]!)
        cache.insertAtColdEnd(layer: 0, expert: 2, weights: fetched[2]!)

        // Real traffic actually uses expert 1 (a cache hit) -- must
        // promote it to MRU.
        XCTAssertNotNil(cache.get(layer: 0, expert: 1))

        // A new real fetch now must evict expert 2 (still at the cold
        // end), not the just-used expert 1.
        let fetched2 = try store.fetch(layerIndex: 0, experts: [3])
        cache.insert(layer: 0, expert: 3, weights: fetched2[3]!)

        XCTAssertTrue(cache.contains(layer: 0, expert: 1), "recently-used warm entry must survive")
        XCTAssertFalse(cache.contains(layer: 0, expert: 2), "untouched warm entry is evicted first")
        XCTAssertTrue(cache.contains(layer: 0, expert: 3))
    }

    func testWarmerDisabledIsANoOp() throws {
        let (dir, store) = try makeTinyCheckpoint(numExperts: 8, numLayers: 1)
        defer { try? FileManager.default.removeItem(at: dir) }

        let counts: [ExpertCache.Key: Int] = [ExpertCache.Key(layer: 0, expert: 1): 100]
        let profile = makeProfile(counts: counts, numExperts: 8, totalLayers: 1)
        let cache = ExpertCache(byteBudget: 1_000_000_000)

        let warmer = ExpertCacheWarmer(
            cache: cache, store: store, profile: profile, enabled: false)
        warmer.start()
        warmer.waitUntilFinishedForTesting()

        XCTAssertFalse(cache.contains(layer: 0, expert: 1), "a disabled warmer must never fetch anything")
        XCTAssertEqual(warmer.stats.stopReason, "disabled")
    }

    func testWarmerStopsEarlyOnceForegroundTrafficThresholdIsReached() throws {
        let (dir, store) = try makeTinyCheckpoint(numExperts: 8, numLayers: 1)
        defer { try? FileManager.default.removeItem(at: dir) }

        var counts: [ExpertCache.Key: Int] = [:]
        for e in 0 ..< 8 { counts[ExpertCache.Key(layer: 0, expert: e)] = 8 - e }
        let profile = makeProfile(counts: counts, numExperts: 8, totalLayers: 1)
        let cache = ExpertCache(byteBudget: 1_000_000_000)

        // Simulate real traffic having ALREADY produced enough signal
        // (totalForwardCalls starts past the (very low) threshold) before
        // the warmer even starts.
        profile.record(layer: 0, groups: [(expert: 0, range: 0 ..< 1)])

        let warmer = ExpertCacheWarmer(
            cache: cache, store: store, profile: profile, targetFraction: 1.0, maxInFlight: 1,
            stopAfterForegroundCalls: 1, enabled: true)
        warmer.start()
        warmer.waitUntilFinishedForTesting()

        XCTAssertEqual(warmer.stats.stopReason, "foreground traffic active")
    }

    func testWarmerCancelStopsRemainingWork() throws {
        let (dir, store) = try makeTinyCheckpoint(numExperts: 32, numLayers: 1)
        defer { try? FileManager.default.removeItem(at: dir) }

        var counts: [ExpertCache.Key: Int] = [:]
        for e in 0 ..< 32 { counts[ExpertCache.Key(layer: 0, expert: e)] = 32 - e }
        let profile = makeProfile(counts: counts, numExperts: 32, totalLayers: 1)
        let cache = ExpertCache(byteBudget: 1_000_000_000)

        let warmer = ExpertCacheWarmer(
            cache: cache, store: store, profile: profile, targetFraction: 1.0, maxInFlight: 1,
            stopAfterForegroundCalls: 1_000_000, enabled: true)
        warmer.start()
        warmer.cancel()
        warmer.waitUntilFinishedForTesting()

        XCTAssertEqual(warmer.stats.stopReason, "cancelled")
        XCTAssertLessThan(
            warmer.stats.warmed, 32, "cancellation must stop the warm task before it processes everything")
    }
}
