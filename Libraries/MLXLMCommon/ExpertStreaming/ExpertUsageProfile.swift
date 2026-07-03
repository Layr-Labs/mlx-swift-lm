// Persisted per-(layer, expert) usage-frequency profile for MoE SSD
// streaming, used to warm `ExpertCache` before real traffic arrives.
//
// THE COLD-START PROBLEM: `ExpertCache` is a pure LRU that only learns what
// is "hot" from experience -- a fresh process starts at 0% hit rate and
// only converges to the steady-state ~90%+ hit rate after enough forward
// passes have organically populated the cache. MoE routing frequency for a
// GIVEN checkpoint is heavily skewed (a small subset of routed experts
// absorb most tokens) and stable across prompts (it is a property of the
// trained gate weights, not the specific prompt text), so the set of
// experts a warm cache converges to is largely predictable ahead of time
// from a PAST run against the same checkpoint. This type records that
// distribution to disk so the NEXT process can skip straight to a
// warm-ish cache instead of re-discovering it from scratch.
//
// WHAT IS COUNTED: `record(layer:groups:)` is called once per streamed
// layer per forward pass with the SAME run-length-encoded `(expert,
// rowRange)` groups `StreamingQuantizedSwitchGLU` already computes on the
// host to plan its fetch chunks -- no extra GPU work, no extra host
// computation beyond an integer add per group. Each group's `range.count`
// (how many rows/tokens in this forward pass routed to that expert) is
// added to that (layer, expert)'s running count, so the profile reflects
// TOKEN-weighted frequency, not just "was this expert touched at all" --
// an expert that wins a 64-token prefill batch should outweigh one that
// wins a single decode step.
//
// PERSISTENCE: debounced by forward-pass count (`persistEveryNCalls`, not
// wall-clock, so cadence naturally matches generation speed rather than
// firing at a fixed interval regardless of whether anything happened) and
// written on a background utility-QoS queue so a save in progress never
// blocks the forward pass that triggered it. At most one save is in
// flight at a time (a flag guards re-entry) -- a slow disk write bunching
// up several debounce intervals' worth of counts is fine; a second
// concurrent write racing the first is not worth the complexity to
// support since the counts are monotonically-updated in memory regardless
// of whether the on-disk copy is current.
import Foundation

public final class ExpertUsageProfile: @unchecked Sendable {

    // MARK: - Checkpoint identity

    /// Identifies WHICH checkpoint a profile belongs to. Path alone is not
    /// enough (a re-downloaded or swapped checkpoint can reuse the same
    /// directory path with different weights); a full content hash is too
    /// expensive to compute just to key a cache-warming hint (that's what
    /// `WeightHasher.computeHash(for:)` is for, on-demand, for
    /// attestation). Path + total checkpoint byte size is a cheap,
    /// good-enough fingerprint for this purpose: the profile is a
    /// best-effort warming HINT, not a correctness-critical value -- if a
    /// swapped checkpoint happens to land on the exact same total byte
    /// count at the exact same path, the worst case is a few
    /// warmed-but-wrong experts that just get evicted (see
    /// `ExpertCache.insertAtColdEnd`), not incorrect output.
    public struct CheckpointIdentity: Codable, Equatable, Sendable {
        public let path: String
        public let totalBytes: Int64

        public init(path: String, totalBytes: Int64) {
            self.path = path
            self.totalBytes = totalBytes
        }
    }

    /// Sums the size of every `*.safetensors` shard in `modelDirectory`.
    /// Only the weight shards are hashed into identity (not config.json /
    /// tokenizer files, which can be legitimately touched up without the
    /// weights changing) so the identity check is exactly "same weight
    /// bytes on disk", matching the invariant the warmed profile actually
    /// depends on.
    public static func checkpointIdentity(modelDirectory: URL) -> CheckpointIdentity {
        var total: Int64 = 0
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: modelDirectory, includingPropertiesForKeys: [.fileSizeKey])
        {
            for url in entries where url.pathExtension == "safetensors" {
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                total += Int64(size)
            }
        }
        return CheckpointIdentity(path: modelDirectory.standardizedFileURL.path, totalBytes: total)
    }

    /// Default on-disk location for a checkpoint's profile:
    /// `~/.cache/darkbloom/expert-profile/<sha256(identity)>.json`.
    ///
    /// Keyed by a hash of the FULL identity (path + size), not the bare
    /// directory name, so two different checkpoints that happen to share a
    /// leaf directory name (e.g. two different `DeepSeek-V4-Flash-4bit`
    /// checkouts under different parents) never collide, and the profile
    /// naturally becomes stale/unreadable-as-a-hit (falls through to "no
    /// profile") the moment the checkpoint's bytes change without needing
    /// an explicit migration step.
    public static func defaultProfileURL(for identity: CheckpointIdentity) -> URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache")
        let dir = cacheDir.appendingPathComponent("darkbloom/expert-profile", isDirectory: true)
        let digest = stableHash("\(identity.path)|\(identity.totalBytes)")
        return dir.appendingPathComponent("\(digest).json")
    }

    /// Small dependency-free stable hash (FNV-1a) -- this is a cache-file
    /// naming scheme, not a security boundary, so cryptographic strength
    /// is unnecessary and pulling in CryptoKit/CommonCrypto for it would
    /// be pure overhead.
    private static func stableHash(_ s: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in s.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01B3
        }
        return String(format: "%016llx", hash)
    }

    /// Reads `DSV4_STREAM_PROFILE` (default enabled -- set to `"0"` to
    /// disable both collection and persistence entirely, e.g. for a
    /// short-lived debug process that shouldn't perturb a shared
    /// checkpoint's profile file).
    public static func enabledFromEnv() -> Bool {
        ProcessInfo.processInfo.environment["DSV4_STREAM_PROFILE"] != "0"
    }

    /// Reads `DSV4_STREAM_PROFILE_PERSIST_EVERY` (int, default 200 forward
    /// calls). 200 streamed-layer forward calls is a handful of decode
    /// tokens (numHiddenLayers of them per token) or a fraction of one
    /// large prefill -- frequent enough that a short-lived process (a
    /// smoke test, a single request) still gets a save in, infrequent
    /// enough that a long decode session isn't debounce-saving every
    /// token.
    public static func persistEveryNCallsFromEnv() -> Int {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_PROFILE_PERSIST_EVERY"],
            let n = Int(raw), n > 0
        {
            return n
        }
        return 200
    }

    /// Reads `DSV4_STREAM_PROFILE_DECAY` (double in [0, 1], default 0.5).
    /// See `loadMerged` for how this is applied.
    public static func historyDecayFromEnv() -> Double {
        if let raw = ProcessInfo.processInfo.environment["DSV4_STREAM_PROFILE_DECAY"],
            let d = Double(raw), d >= 0, d <= 1
        {
            return d
        }
        return 0.5
    }

    // MARK: - On-disk format

    struct FileFormat: Codable {
        var version: Int
        var checkpoint: CheckpointIdentity
        var numExperts: Int
        var totalLayers: Int
        /// Flat "layer:expert" -> weighted count. A flat string-keyed dict
        /// (rather than nested `[[Int]]`/2D array) keeps the JSON
        /// human-inspectable and naturally sparse -- most (layer, expert)
        /// pairs across 41 layers x 256 experts are never populated in any
        /// single profiling run, so a dense array would be mostly zeros.
        var counts: [String: Int]
    }

    // MARK: - State

    private let lock = NSLock()
    private var counts: [ExpertCache.Key: Int] = [:]
    private var callsSinceSave: Int = 0
    private var saveInFlight = false
    private var totalForwardCallsLocked: Int = 0

    public let identity: CheckpointIdentity
    public let profileURL: URL
    public let numExperts: Int
    public let totalLayers: Int
    private let persistEveryNCalls: Int

    public init(
        identity: CheckpointIdentity, profileURL: URL, numExperts: Int, totalLayers: Int,
        persistEveryNCalls: Int = ExpertUsageProfile.persistEveryNCallsFromEnv(),
        initialCounts: [ExpertCache.Key: Int] = [:]
    ) {
        self.identity = identity
        self.profileURL = profileURL
        self.numExperts = numExperts
        self.totalLayers = totalLayers
        self.persistEveryNCalls = max(1, persistEveryNCalls)
        self.counts = initialCounts
    }

    /// Number of streamed-layer forward calls `record` has seen THIS
    /// process (not carried over from a loaded profile) -- used by
    /// `ExpertCacheWarmer` as a proxy for "how much real traffic has this
    /// process already served", so it knows when to stop background
    /// warming in favor of organic cache growth (see ExpertCacheWarmer.swift).
    public var totalForwardCalls: Int {
        lock.lock()
        defer { lock.unlock() }
        return totalForwardCallsLocked
    }

    /// Record one streamed layer's forward-pass expert selection. `groups`
    /// is the same run-length-encoded `(expert, rowRange)` list
    /// `StreamingQuantizedSwitchGLU` computes for its own chunk-fetch
    /// planning -- passed by reference here rather than recomputed.
    public func record(layer: Int, groups: [(expert: Int, range: Range<Int>)]) {
        var shouldSave = false
        lock.lock()
        for g in groups {
            let key = ExpertCache.Key(layer: layer, expert: g.expert)
            counts[key, default: 0] += g.range.count
        }
        totalForwardCallsLocked += 1
        callsSinceSave += 1
        if callsSinceSave >= persistEveryNCalls && !saveInFlight {
            callsSinceSave = 0
            saveInFlight = true
            shouldSave = true
        }
        lock.unlock()

        if shouldSave {
            DispatchQueue.global(qos: .utility).async { [weak self] in
                self?.saveAndClearInFlightFlag()
            }
        }
    }

    /// Snapshot of current in-memory counts (locked copy). Exposed for
    /// `ExpertCacheWarmer.warmOrder` and tests.
    public func snapshotCounts() -> [ExpertCache.Key: Int] {
        lock.lock()
        defer { lock.unlock() }
        return counts
    }

    /// Synchronous, unconditional save -- for callers that need a
    /// guarantee the on-disk file reflects current state before
    /// proceeding (model unload/reload, process shutdown, tests). Safe to
    /// call even if a debounced background save is concurrently in
    /// flight: both write the same directory atomically (see `save`),
    /// last writer wins, and neither corrupts the file.
    public func flush() {
        save()
    }

    private func saveAndClearInFlightFlag() {
        save()
        lock.lock()
        saveInFlight = false
        lock.unlock()
    }

    private func save() {
        let snapshot = snapshotCounts()
        var flat: [String: Int] = [:]
        flat.reserveCapacity(snapshot.count)
        for (key, count) in snapshot {
            flat["\(key.layer):\(key.expert)"] = count
        }
        let format = FileFormat(
            version: 1, checkpoint: identity, numExperts: numExperts, totalLayers: totalLayers,
            counts: flat)
        do {
            let dir = profileURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(format)
            // Write to a sibling temp file then rename, so a reader (or a
            // concurrently-racing save from another process) never
            // observes a partially-written file -- `rename(2)` on the same
            // volume is atomic.
            let tmpURL = profileURL.appendingPathExtension("tmp-\(UUID().uuidString)")
            try data.write(to: tmpURL, options: .atomic)
            _ = try FileManager.default.replaceItemAt(profileURL, withItemAt: tmpURL)
        } catch {
            // Best-effort: a failed profile save (e.g. read-only cache dir,
            // disk full) must never affect inference correctness or crash
            // the process -- it just means the next process starts cold
            // again, exactly like today without this feature.
        }
    }

    // MARK: - Load + merge

    /// Load an existing on-disk profile for `identity` (if present, and
    /// its recorded identity matches -- a mismatch means the checkpoint at
    /// this path changed since the profile was written, so the stale
    /// counts are discarded entirely rather than silently mixed with a
    /// different checkpoint's distribution) and construct a fresh
    /// in-process `ExpertUsageProfile` seeded from it.
    ///
    /// DECAY, NOT A PLAIN SUM: loaded counts are multiplied by
    /// `historyDecay` (default 0.5) before becoming this session's
    /// starting point, rather than kept at full weight or discarded
    /// entirely. Justification:
    ///   - A plain sum across many process lifetimes lets counts grow
    ///     unboundedly and makes very old sessions numerically dominate a
    ///     checkpoint's TRUE current distribution just because they ran
    ///     first and for longer -- undesirable if usage patterns ever
    ///     shift (different downstream traffic mix, a template change that
    ///     favors different tool-call routing, etc.).
    ///   - Discarding history entirely on every load would throw away
    ///     exactly the cross-session signal this feature exists to
    ///     capture, forcing every process to rebuild its own profile from
    ///     zero before the NEXT process can warm from it.
    ///   - A 0.5 decay each load is a simple exponential-moving-average
    ///     across process lifetimes: after N restarts a given historical
    ///     session's contribution has fallen off by 0.5^N, so the profile
    ///     converges to reflect recent behavior within a handful of
    ///     restarts while still surviving a single short-lived process
    ///     (one 512-token generation) with enough weight to usefully warm
    ///     the very next process.
    public static func loadMerged(
        identity: CheckpointIdentity, profileURL: URL, numExperts: Int, totalLayers: Int,
        historyDecay: Double = ExpertUsageProfile.historyDecayFromEnv()
    ) -> ExpertUsageProfile {
        var initialCounts: [ExpertCache.Key: Int] = [:]
        if let data = try? Data(contentsOf: profileURL),
            let format = try? JSONDecoder().decode(FileFormat.self, from: data),
            format.checkpoint == identity
        {
            for (flatKey, count) in format.counts {
                guard let key = parseFlatKey(flatKey) else { continue }
                let decayed = Int((Double(count) * historyDecay).rounded())
                if decayed > 0 { initialCounts[key] = decayed }
            }
        }
        return ExpertUsageProfile(
            identity: identity, profileURL: profileURL, numExperts: numExperts,
            totalLayers: totalLayers, initialCounts: initialCounts)
    }

    static func parseFlatKey(_ s: String) -> ExpertCache.Key? {
        let parts = s.split(separator: ":")
        guard parts.count == 2, let layer = Int(parts[0]), let expert = Int(parts[1]) else {
            return nil
        }
        return ExpertCache.Key(layer: layer, expert: expert)
    }

    // MARK: - Warm-order computation (pure, no I/O -- independently testable)

    /// Compute which (layer, expert) pairs to warm, and in what order,
    /// given a snapshot of counts and a target byte budget.
    ///
    /// GLOBAL FREQUENCY ORDER, NOT PER-LAYER-PROPORTIONAL: candidates are
    /// ranked by count DESCENDING ACROSS ALL LAYERS COMBINED, not by a
    /// per-layer quota. This mirrors `ExpertCache`'s own design (see its
    /// file-level comment): the cache is a SINGLE shared byte budget
    /// across every layer, specifically so "hot" experts from any layer
    /// compete for space rather than each layer getting a fixed
    /// pre-partitioned share. A per-layer-proportional warm order would
    /// fight that design -- it would spend warm budget on a layer's
    /// merely-average expert (to satisfy that layer's quota) ahead of a
    /// truly hot expert in a DIFFERENT layer, which is exactly backwards
    /// from what the organic LRU cache converges to under real traffic.
    /// Global ranking reproduces the same priority order the cache would
    /// reach on its own given enough forward passes, just without waiting
    /// for them.
    ///
    /// `byteBudget` bounds the total bytes selected (checked against
    /// `bytesPerExpert`, a caller-supplied estimate since this function has
    /// no I/O access to real per-expert sizes) -- stops as soon as adding
    /// the next candidate would exceed it, OR the candidate list is
    /// exhausted (profile has fewer distinct experts than would fill the
    /// budget), whichever comes first.
    public static func warmOrder(
        counts: [ExpertCache.Key: Int], byteBudget: Int, bytesPerExpert: Int
    ) -> [ExpertCache.Key] {
        guard byteBudget > 0, bytesPerExpert > 0, !counts.isEmpty else { return [] }
        // Sort by count descending; break ties deterministically (layer
        // then expert) so warm order -- and therefore test output -- is
        // reproducible rather than dependent on Dictionary's iteration
        // order.
        let ranked = counts.sorted { a, b in
            if a.value != b.value { return a.value > b.value }
            if a.key.layer != b.key.layer { return a.key.layer < b.key.layer }
            return a.key.expert < b.key.expert
        }
        let maxCount = byteBudget / bytesPerExpert
        guard maxCount > 0 else { return [] }
        return ranked.prefix(maxCount).map { $0.key }
    }
}
