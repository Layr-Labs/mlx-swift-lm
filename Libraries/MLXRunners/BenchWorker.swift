// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Engine Protocol v1 server over `Runner`
// (Darkbloom runner contract §6.1, §8, §9).
//
// ONE server for every runner: benchd never links a runner, and this file
// carries no family name. Everything it advertises derives from the loaded
// runner's manifest plus what actually loaded; everything it reports is a
// counter the engine or the stepper already keeps.
//
// It NEVER emits a timing. benchd times the round trip; a worker that
// reported its own numbers would be scoring itself.
//
// WHY THIS IS A LIBRARY FILE while `Executables/bench-worker/main.swift` is
// a thin shim: a test target that depends on an executable target makes
// SwiftPM run that binary as the swift-testing host, which aborts the whole
// package's `@Test` pass — the reason `BenchCBv2Core`/`BenchCBv2` are split
// the same way (see Package.swift). The conformance test drives
// `BenchWorkerServer` in process over a mock runner.

import Foundation
import MLX
import MLXLMCommon

// MARK: - Transport

/// One NDJSON line in, one NDJSON line out. Abstracted so the conformance
/// test drives the same loop the binary runs.
public protocol BenchWorkerTransport: AnyObject {
    /// Next request line, or nil at EOF.
    func readLine() -> String?
    func write(line: String)
}

/// stdin/stdout.
public final class StdioTransport: BenchWorkerTransport {
    public init() {}
    public func readLine() -> String? { Swift.readLine(strippingNewline: true) }
    public func write(line: String) {
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
    }
}

// MARK: - Memory reporting

/// Allocator reporting for `phase_diagnostics`.
///
/// Every member is OPTIONAL because absence is a real answer: a worker with
/// no MLX allocator has nothing to report, and the schema says an absent
/// `cache_memory` is "not asserted", which is a different claim from zero.
public protocol WorkerMemoryReporter: Sendable {
    /// Allocator watermarks read PRE-drain.
    func preDrainSnapshot() -> (active: Int, cache: Int, peak: Int)?
    /// Drain the allocator's free-buffer cache.
    func drain()
    /// Free-buffer cache size AFTER the drain. The parent fails the run
    /// closed unless this is exactly 0.
    func cacheMemoryAfterDrain() -> Int?
    /// Engine-reported peak RSS in GB. Distrusted for scoring.
    func peakRAMGB() -> Double?
}

/// The real MLX allocator.
public struct MLXMemoryReporter: WorkerMemoryReporter {
    public init() {}
    public func preDrainSnapshot() -> (active: Int, cache: Int, peak: Int)? {
        (Memory.activeMemory, Memory.cacheMemory, Memory.peakMemory)
    }
    public func drain() { Memory.clearCache() }
    public func cacheMemoryAfterDrain() -> Int? { Memory.cacheMemory }
    public func peakRAMGB() -> Double? { Double(Memory.peakMemory) / 1_000_000_000 }
}

// MARK: - Engine audit seams

/// Cumulative speculative counters an engine may expose. `EngineV2` does.
public protocol CBv2MTPCountersReporting: AnyObject {
    func mtpMetricsSnapshot() -> CBv2MTPMetrics?
}

extension EngineV2: CBv2MTPCountersReporting {}

/// Per-round free-run audit, ASSEMBLED from the engine's own journal
/// (`CBv2MTPMetrics.roundAudits`). Every field here is a reading of records
/// the engine wrote at its finalize boundary; nothing is computed from the
/// tokens the worker happened to drain.
public struct FreeRunRoundAudit: Sendable, Equatable {
    /// Per round, the committed width. Length is the round count R.
    public var acceptanceLengths: [Int]
    /// Per row, per round, the PRE-min natural accept-walk length. This is
    /// the record's raw `accepted`, not the `min(accepted, confirmed)` the
    /// cumulative counter accumulates — the difference is exactly the
    /// straggler throttling the cohort's one common width imposes, which is
    /// the thing this field exists to make visible.
    public var naturalAcceptedByStream: [[Int]]
    /// Rows still generating at each round. Non-increasing under the fixed-N
    /// cohort policy.
    public var activeStreamsByRound: [Int]
    /// Draft tokens proposed: the sum of every record's `k`.
    public var draftedTotal: Int
    /// Drafts that passed verification AND were committed: the sum of
    /// `min(accepted, confirmed)`, which is the `observedAccepted` the
    /// engine's own cumulative counter accumulates.
    public var acceptedTotal: Int
    /// Tokens the journal accounts for: the sum of every record's
    /// `confirmed`.
    public var committedTotal: Int

    public var rounds: Int { acceptanceLengths.count }

    /// Refusals from journal assembly. Each is a refusal to publish an
    /// audit, never a partial one.
    public enum AuditError: Error, CustomStringConvertible, Equatable {
        /// The journal reached its retention cap, so its head was evicted. A
        /// TRUNCATED head cannot prove coverage of the window, and an audit
        /// assembled from what survives would look complete while missing
        /// rounds. Same rule the 125B worker applies.
        case journalTruncated(records: Int, cap: Int)
        /// A record names a row this window never submitted.
        case unknownRow(requestID: UInt64)
        /// Two rows report different committed widths for the same round.
        /// The cohort commits ONE common width per round, so this means the
        /// window boundary or the row grouping is wrong.
        case widthDisagreement(round: Int, widths: [Int])
        /// The journal accounts for more committed tokens than the worker
        /// drained. Under-count is legitimate — a target-only decode step
        /// commits a token and finalizes no verify round — but over-count
        /// means the window covers records from outside it.
        case journalOvercount(journal: Int, drained: Int)

        public var description: String {
            switch self {
            case .journalTruncated(let records, let cap):
                return "mtp round journal truncated at its \(cap)-record cap "
                    + "(\(records) retained): coverage cannot be proven"
            case .unknownRow(let requestID):
                return "mtp round journal names row \(requestID), which this "
                    + "window did not submit"
            case .widthDisagreement(let round, let widths):
                return "mtp round \(round) reports committed widths "
                    + "\(widths.map(String.init).joined(separator: ", ")); a cohort "
                    + "commits one common width"
            case .journalOvercount(let journal, let drained):
                return "mtp round journal accounts for \(journal) committed tokens "
                    + "but \(drained) were drained"
            }
        }
    }

    /// Assemble the window's audit from the journal.
    ///
    /// Records arrive in finalize order, one per (row, round). Grouping by
    /// row and reading position-wise recovers the rounds: a row's r-th
    /// record IS its r-th round, and a row that finished early simply has no
    /// record past its last one — which is what makes
    /// `activeStreamsByRound` non-increasing without computing it from
    /// anything else.
    public static func assemble(
        records: [CBv2MTPRoundAuditRecord],
        slotForRequestID: [UInt64: Int],
        streamCount: Int
    ) throws -> FreeRunRoundAudit? {
        guard !records.isEmpty else { return nil }

        var byRow = [[CBv2MTPRoundAuditRecord]](repeating: [], count: streamCount)
        for record in records {
            guard let slot = slotForRequestID[record.requestID] else {
                throw AuditError.unknownRow(requestID: record.requestID)
            }
            byRow[slot].append(record)
        }

        let rounds = byRow.map(\.count).max() ?? 0
        var acceptanceLengths: [Int] = []
        var activeStreams: [Int] = []
        for round in 0 ..< rounds {
            let widths = byRow.compactMap { row in
                round < row.count ? row[round].confirmed : nil
            }
            guard let width = widths.first else { continue }
            guard widths.allSatisfy({ $0 == width }) else {
                throw AuditError.widthDisagreement(round: round, widths: widths)
            }
            acceptanceLengths.append(width)
            activeStreams.append(widths.count)
        }

        let natural = byRow.map { row in
            (0 ..< rounds).map { round in round < row.count ? row[round].accepted : 0 }
        }
        return FreeRunRoundAudit(
            acceptanceLengths: acceptanceLengths,
            naturalAcceptedByStream: natural,
            activeStreamsByRound: activeStreams,
            draftedTotal: records.reduce(0) { $0 + $1.k },
            acceptedTotal: records.reduce(0) { $0 + min($1.accepted, $1.confirmed) },
            committedTotal: records.reduce(0) { $0 + $1.confirmed })
    }

    public init(
        acceptanceLengths: [Int],
        naturalAcceptedByStream: [[Int]],
        activeStreamsByRound: [Int],
        draftedTotal: Int,
        acceptedTotal: Int,
        committedTotal: Int
    ) {
        self.acceptanceLengths = acceptanceLengths
        self.naturalAcceptedByStream = naturalAcceptedByStream
        self.activeStreamsByRound = activeStreamsByRound
        self.draftedTotal = draftedTotal
        self.acceptedTotal = acceptedTotal
        self.committedTotal = committedTotal
    }
}

// MARK: - Speculative protocol gate

/// The `--speculative-protocol` argument.
///
/// benchd's official spawn always appends `--speculative-protocol v1.1`
/// (`free_run_spawn_args`), and the flag is what turns the SPECULATIVE
/// surface on. Absent, the worker speaks plain v1: `capabilities`,
/// `spec_modes` and `max_batch_size` are all v1.1-or-later additions to a
/// frozen v1 hello, so a v1-only engine omits them, and a `spec` on a
/// request is refused rather than honoured. That is not a degraded mode with
/// speculation quietly disabled — it is a different protocol version, and
/// answering a `spec` under it would be a worker resolving something the
/// parent never established it could ask for.
public enum SpeculativeProtocol: Sendable, Equatable {
    /// The only accepted value.
    public static let version = "v1.1"

    /// Refusals from the argument. Before the load, always.
    public enum ArgumentError: Error, CustomStringConvertible, Equatable {
        case unsupportedVersion(String)

        public var description: String {
            switch self {
            case .unsupportedVersion(let raw):
                return "--speculative-protocol \(raw) is not \(SpeculativeProtocol.version)"
            }
        }
    }

    /// Whether the speculative surface is enabled. `nil` (flag absent) is
    /// plain v1; any value other than `v1.1` is a refusal, never a fallback
    /// to plain v1 — a parent that asked for a version this worker does not
    /// speak must hear so, not be silently served an older one.
    public static func isEnabled(_ raw: String?) throws -> Bool {
        guard let raw else { return false }
        guard raw == version else { throw ArgumentError.unsupportedVersion(raw) }
        return true
    }
}

// MARK: - Capabilities

/// Capability strings the hello advertises.
public enum WorkerCapability {
    public static let freeRunDecode = "free_run_decode"
    public static let batchedFreeRunDecode = "batched_free_run_decode"
    public static let perStreamTiming = "per_stream_timing"
    public static let cohortReferenceReplay = "cohort_reference_replay"
}

// MARK: - Server

/// Engine Protocol v1 server over one loaded runner.
///
/// The model is loaded ONCE by the caller, before the hello. This object
/// never loads, never downloads, and never tokenizes: token ids cross the
/// boundary in both directions.
public final class BenchWorkerServer: @unchecked Sendable {

    /// The protocol version this worker implements.
    public static let protocolVersion = 1

    private let runner: any Runner
    private let manifest: RunnerManifest
    private let transport: any BenchWorkerTransport
    private let trusted: Bool
    /// `--speculative-protocol v1.1` was given. See ``SpeculativeProtocol``.
    private let speculative: Bool
    private let build: String
    private let device: String
    private let kvBytesCapacity: Int
    private let maxDecodeTokens: Int
    private let memory: any WorkerMemoryReporter
    /// Non-nil only when this session is served by a RESIDENT. See
    /// `BenchWorkerResident`.
    private let residentIdentity: ResidentIdentity?
    private let residentWeights: String?
    private let nonce: String
    private let decoder: JSONDecoder

    /// Timed step-requests completed since the last `phase_diagnostics`.
    ///
    /// A WORKER counter, not `stepper.forwards`: a phase issues verbs across
    /// several stepper sessions (`decode_begin` starts a fresh one), and the
    /// parent's phase-close barrier compares this against the number of timed
    /// steps IT issued. `prefill` and `correctness` are not timed steps and
    /// do not move it.
    private var completedWork = 0

    private var stepper: (any TeacherForcedStepper)?
    private var freeRun: FreeRunSession?

    public init(
        runner: any Runner,
        transport: any BenchWorkerTransport,
        trusted: Bool,
        speculative: Bool = true,
        build: String,
        device: String,
        kvBytesCapacity: Int,
        maxDecodeTokens: Int = 4096,
        memory: any WorkerMemoryReporter = MLXMemoryReporter(),
        residentIdentity: ResidentIdentity? = nil,
        residentWeights: String? = nil,
        nonce: String = UUID().uuidString
    ) {
        self.runner = runner
        self.manifest = type(of: runner).manifest
        self.transport = transport
        self.trusted = trusted
        self.speculative = speculative
        self.build = build
        self.device = device
        self.kvBytesCapacity = kvBytesCapacity
        self.maxDecodeTokens = maxDecodeTokens
        self.memory = memory
        self.residentIdentity = residentIdentity
        self.residentWeights = residentWeights
        self.nonce = nonce
        self.decoder = JSONDecoder()
    }

    // MARK: Hello (§6.1)

    /// Every field derived from the manifest and the loaded state. No runner
    /// writes a hello field by hand, and none is withheld by a constant here.
    public func hello() -> WorkerResponse {
        var response = WorkerResponse(id: 0, ok: true, nonce: nonce)
        response.expertStats = ExpertStreamingStats()
        response.protocolVersion = Self.protocolVersion
        response.backend = manifest.backend
        response.device = device

        // `capabilities`, `spec_modes` and `max_batch_size` are all
        // v1.1-or-later additions to a frozen v1 hello, so a worker speaking
        // plain v1 omits the three of them together rather than advertising
        // an empty capability list — an empty array is a claim ("I support
        // none of these"), absence is the v1 wire.
        if speculative {
            var capabilities: [String] = []
            if manifest.regimes.contains(where: { $0.timing == .freeRun }) {
                capabilities.append(WorkerCapability.freeRunDecode)
            }
            if manifest.regimes.contains(where: { $0.batch.maxWidth > 1 }) {
                capabilities.append(WorkerCapability.batchedFreeRunDecode)
            }
            if manifest.regimes.contains(where: { $0.perStreamTiming }) {
                capabilities.append(WorkerCapability.perStreamTiming)
            }
            if trusted {
                capabilities.append(WorkerCapability.cohortReferenceReplay)
            }
            response.capabilities = capabilities
            response.specModes = runner.loadedDecoders.map(\.rawValue)
            response.maxBatchSize =
                manifest.regimes.map(\.batch.maxWidth).filter { $0 > 1 }.max()
        }
        response.headProvenance = runner.headProvenance.map(WireHeadProvenance.init)
        // Resident identity rides only on the SOCKET, where the attaching
        // worker reads it, checks the weights path and decides what reaches
        // benchd. An in-process worker has no resident and emits neither.
        response.resident = residentIdentity
        response.residentWeights = residentWeights
        response.runner = RunnerIdentity(
            id: manifest.runnerID,
            modelType: runner.loadedModelType,
            manifestSHA256: manifest.sha256Digest(),
            build: build)
        return response
    }

    // MARK: Loop

    /// Emit the unsolicited hello, then serve until EOF.
    public func run() async {
        send(hello())
        while let line = transport.readLine() {
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            let request: WorkerRequest
            do {
                request = try decoder.decode(WorkerRequest.self, from: Data(line.utf8))
            } catch {
                var response = WorkerResponse(id: -1, ok: false, nonce: nonce)
                response.error = "unparseable request: \(error)"
                send(response)
                continue
            }
            send(await handle(request))
        }
    }

    private func send(_ response: WorkerResponse) {
        transport.write(line: response.jsonLine())
    }

    /// One verb. Any throw becomes `ok:false` + `error`; the parent discards
    /// the phase.
    public func handle(_ request: WorkerRequest) async -> WorkerResponse {
        do {
            return try await dispatch(request)
        } catch {
            var response = WorkerResponse(id: request.id, ok: false, nonce: nonce)
            response.error = "\(error)"
            return response
        }
    }

    private func dispatch(_ request: WorkerRequest) async throws -> WorkerResponse {
        var response = WorkerResponse(id: request.id, ok: true, nonce: nonce)
        switch request.kind {

        case .prefill:
            let prompt = try require(request.promptTokens, "prompt_tokens")
            response.token = try beginStepper().forward(prompt).argmax

        case .decodeBegin:
            let seed = try require(request.seedTokens, "seed_tokens")
            response.seedToken = try beginStepper().forward(seed).argmax
            if let spec = request.spec {
                response.effectiveSpec = try effectiveSpec(spec)
            }
            completedWork += 1

        case .decodeStep:
            let token = try require(request.token, "token")
            response.token = try currentStepper().forward([token]).argmax
            completedWork += 1

        case .correctness:
            let prompt = try require(request.promptTokens, "prompt_tokens")
            let steps = try require(request.steps, "steps")
            response.tokens = try await freeRunGreedy(prompt: prompt, steps: steps)
            response.peakRAMGB = memory.peakRAMGB()

        case .correctnessBegin:
            let prompt = try require(request.promptTokens, "prompt_tokens")
            fill(&response, with: try beginStepper().forward(prompt))
            completedWork += 1

        case .correctnessStep:
            let token = try require(request.token, "token")
            fill(&response, with: try currentStepper().forward([token]))
            completedWork += 1

        case .freeDecodeBegin:
            response = try await freeDecodeBegin(request)

        case .freeDecodeRun:
            response = try await freeDecodeRun(request)

        case .cohortReferenceReplay:
            response = try cohortReferenceReplay(request)

        case .phaseDiagnostics:
            // PRE-drain watermarks first: the drain is what makes
            // `cache_memory` assertable, and reading after it would describe
            // the drain rather than the phase.
            let snapshot = memory.preDrainSnapshot()
            releaseSessions()
            memory.drain()
            response.expertStats = ExpertStreamingStats()
            response.peakRAMGB = memory.peakRAMGB()
            response.completedWork = completedWork
            response.cacheMemory = memory.cacheMemoryAfterDrain()
            response.mlxActiveMemoryBytes = snapshot?.active
            response.mlxCacheMemoryBytes = snapshot?.cache
            response.mlxPeakMemoryBytes = snapshot?.peak
            completedWork = 0
        }
        return response
    }

    // MARK: Stepper verbs (§8)

    private func beginStepper() throws -> any TeacherForcedStepper {
        let stepper = try runner.makeStepper()
        try stepper.begin()
        self.stepper = stepper
        return stepper
    }

    private func currentStepper() throws -> any TeacherForcedStepper {
        guard let stepper else { throw StepperError.notBegun }
        return stepper
    }

    /// `correctness_begin` / `correctness_step` payload. `decode_step` and
    /// `correctness_step` reach the SAME `forward`; only the reporting
    /// differs, which is the invariant benchd's architecture doc calls
    /// non-negotiable.
    private func fill(_ response: inout WorkerResponse, with output: StepOutput) {
        response.token = output.argmax
        response.topLogits = output.topLogits.map {
            CorrectnessTraceLogit(token: $0.token, logit: $0.logit)
        }
        response.expertStats = ExpertStreamingStats()
        response.peakRAMGB = memory.peakRAMGB()
    }

    // MARK: Engine verbs (§8, §9)

    /// The FIXED build (§9): contiguous, `maxConcurrentRequests == batch`, no
    /// prefix cache, no waiting-queue use, no stop tokens on cohort requests.
    /// Nothing here is policy the worker invents — it is the one build the
    /// contract pins for benchd.
    private func engineBuild(batch: Int, spec: SpecConfig) throws -> EngineBuild {
        let decoderID = DecoderID(spec.mode)
        var mtpConfig = CBv2MTPConfig(enabled: false)
        if decoderID != .serial {
            let depth = max(1, spec.depth)
            mtpConfig = CBv2MTPConfig(
                enabled: true,
                maxDraftTokens: depth,
                maxSpeculativeBatch: batch,
                fixedDraftTokens: depth,
                verificationMode: .automatic,
                // The envelope is the INTEGRATOR's claim that rectangular
                // target evaluation is argmax-exact at every shape inside it.
                // For a benchmark leg the shape set is exactly this one — B
                // rows by 1+depth verify columns — so the claim is the leg.
                maxAutomaticRectangularTokens: batch * (1 + depth))
        }
        return EngineBuild(
            kvBackend: .contiguous,
            kvBytesCapacity: kvBytesCapacity,
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: batch,
                maxWaiting: batch,
                enablePrefixCache: false),
            loopConfig: CBv2EngineLoopConfig(),
            prefixCache: nil,
            decoder: decoderID,
            mtpConfig: mtpConfig,
            environment: [:])
    }

    /// `correctness`: free-run greedy through the engine, tokens only.
    private func freeRunGreedy(prompt: [Int], steps: Int) async throws -> [Int] {
        let engine = try runner.makeEngine(
            try engineBuild(batch: 1, spec: SpecConfig(mode: DecoderID.serial.rawValue)))
        var request = CBv2Request(
            id: CBv2RequestID(1), promptTokens: prompt, maxTokens: steps)
        request.sampling = Self.greedySampling
        request.stopTokens = []
        var tokens: [Int] = []
        for await event in try engine.submit(request) {
            if case .delta(_, let emitted, _) = event {
                tokens.append(contentsOf: emitted)
                if tokens.count >= steps { break }
            }
        }
        await engine.shutdown()
        return Array(tokens.prefix(steps))
    }

    /// One free-run window: the engine, its streams, and the counter baseline
    /// taken at `free_decode_begin` so the run response reports THIS window's
    /// audit, not the process's.
    private final class FreeRunSession {
        let engine: any CBv2Engine
        let batch: Int
        let spec: SpecConfig
        let cohort: Bool
        /// Request id -> cohort slot, in SLOT ORDER. The journal names rows
        /// by request id; the wire names them by slot.
        let slotForRequestID: [UInt64: Int]
        /// Records already reported. The seed forward is not a verify round,
        /// but a row CAN finalize one before `free_decode_run` is called, so
        /// the window starts where the journal stood after the seed drain.
        var journalCursor = 0
        /// Controller clamp counts as they stood at that same boundary.
        var clampBaseline: [String: Int] = [:]
        var streams: [AsyncStream<CBv2Event>.Iterator]

        init(
            engine: any CBv2Engine, batch: Int, spec: SpecConfig, cohort: Bool,
            slotForRequestID: [UInt64: Int],
            streams: [AsyncStream<CBv2Event>.Iterator]
        ) {
            self.engine = engine
            self.batch = batch
            self.spec = spec
            self.cohort = cohort
            self.slotForRequestID = slotForRequestID
            self.streams = streams
        }
    }

    /// Controller clamp/skip reasons as one histogram. Both maps are
    /// cumulative engine counters; the wire wants the window's delta.
    private static func clampReasons(_ metrics: CBv2MTPMetrics) -> [String: Int] {
        var reasons = metrics.skippedRows
        for (reason, count) in metrics.controllerFallbacks {
            reasons[reason, default: 0] += count
        }
        return reasons
    }

    private func freeDecodeBegin(_ request: WorkerRequest) async throws -> WorkerResponse {
        // The `spec` refusal comes FIRST because it is the more specific
        // answer: under plain v1 the parent asked for a speculative surface
        // by name, and "free_run_decode is not served" would send it looking
        // at the manifest instead of at its own spawn.
        let spec = try effectiveSpec(request.spec)
        guard speculative, manifest.regimes.contains(where: { $0.timing == .freeRun })
        else {
            throw WorkerError.capabilityNotAdvertised(WorkerCapability.freeRunDecode)
        }

        let seeds: [[Int]]
        let cohort: Bool
        if let byStream = request.seedTokensByStream {
            seeds = byStream
            cohort = true
        } else {
            seeds = [try require(request.seedTokens, "seed_tokens")]
            cohort = request.batchSize != nil
        }
        let batch = request.batchSize ?? seeds.count
        guard batch == seeds.count else {
            throw WorkerError.batchMismatch(requested: batch, streams: seeds.count)
        }
        if batch > 1 {
            guard manifest.regimes.contains(where: { $0.batch.maxWidth >= batch }) else {
                throw WorkerError.capabilityNotAdvertised(
                    WorkerCapability.batchedFreeRunDecode)
            }
        }

        let engine = try runner.makeEngine(try engineBuild(batch: batch, spec: spec))
        // Submit ALL streams before consuming ANY: a cohort drained stream by
        // stream is a sequence of solo runs wearing a cohort's name.
        var iterators: [AsyncStream<CBv2Event>.Iterator] = []
        for (slot, seed) in seeds.enumerated() {
            var cbv2 = CBv2Request(
                id: CBv2RequestID(UInt64(slot + 1)), promptTokens: seed,
                maxTokens: maxDecodeTokens)
            cbv2.sampling = Self.greedySampling
            cbv2.stopTokens = []
            iterators.append(try engine.submit(cbv2).makeAsyncIterator())
        }

        var slotForRequestID: [UInt64: Int] = [:]
        for slot in 0 ..< batch { slotForRequestID[UInt64(slot + 1)] = slot }
        let session = FreeRunSession(
            engine: engine, batch: batch, spec: spec, cohort: cohort,
            slotForRequestID: slotForRequestID, streams: iterators)
        var seedTokens: [Int] = []
        for slot in 0 ..< batch {
            seedTokens.append(
                contentsOf: try await drain(session: session, slot: slot, count: 1))
        }
        // The audit window opens HERE, after the seed forward: whatever the
        // journal already holds belongs to the seed, not to the timed run.
        if let metrics = (engine as? any CBv2MTPCountersReporting)?.mtpMetricsSnapshot() {
            session.journalCursor = metrics.roundAudits.count
            session.clampBaseline = Self.clampReasons(metrics)
        }
        self.freeRun = session

        var response = WorkerResponse(id: request.id, ok: true, nonce: nonce)
        if cohort {
            response.seedTokenByStream = seedTokens
            response.effectiveBatchSize = batch
        } else {
            response.seedToken = seedTokens.first
        }
        response.effectiveSpec = spec
        completedWork += 1
        return response
    }

    private func freeDecodeRun(_ request: WorkerRequest) async throws -> WorkerResponse {
        guard let session = freeRun else { throw WorkerError.noFreeRunSession }
        let count = try require(request.count, "count")
        if let batch = request.batchSize, batch != session.batch {
            throw WorkerError.batchMismatch(requested: batch, streams: session.batch)
        }
        var byStream: [[Int]] = []
        for slot in 0 ..< session.batch {
            byStream.append(try await drain(session: session, slot: slot, count: count))
        }

        var response = WorkerResponse(id: request.id, ok: true, nonce: nonce)
        if session.cohort {
            response.tokensByStream = byStream
            response.effectiveBatchSize = session.batch
        } else {
            response.tokens = byStream.first
        }

        let drained = byStream.reduce(0) { $0 + $1.count }
        response.committedTotal = drained

        // AUDIT, assembled from the engine's OWN per-round journal. Nothing
        // here is recomputed from the tokens the worker drained: the journal
        // states what each round drafted, accepted and committed, and the
        // worker only groups the records it was given.
        var audit: FreeRunRoundAudit?
        if let metrics = (session.engine as? any CBv2MTPCountersReporting)?
            .mtpMetricsSnapshot()
        {
            guard metrics.roundAudits.count < CBv2MTPRoundAuditRecord.retainedRecordCap
            else {
                throw FreeRunRoundAudit.AuditError.journalTruncated(
                    records: metrics.roundAudits.count,
                    cap: CBv2MTPRoundAuditRecord.retainedRecordCap)
            }
            let window = Array(metrics.roundAudits.dropFirst(session.journalCursor))
            audit = try FreeRunRoundAudit.assemble(
                records: window,
                slotForRequestID: session.slotForRequestID,
                streamCount: session.batch)
            if let audit, audit.committedTotal > drained {
                throw FreeRunRoundAudit.AuditError.journalOvercount(
                    journal: audit.committedTotal, drained: drained)
            }
            session.journalCursor = metrics.roundAudits.count

            var clamps = Self.clampReasons(metrics)
            for (reason, base) in session.clampBaseline {
                clamps[reason, default: 0] -= base
            }
            clamps = clamps.filter { $0.value > 0 }
            session.clampBaseline = Self.clampReasons(metrics)
            if !clamps.isEmpty { response.depthClampReasons = clamps }
        }

        if let audit {
            response.acceptanceLengths = audit.acceptanceLengths
            response.draftedTotal = audit.draftedTotal
            response.acceptedTotal = audit.acceptedTotal
            // `rounds`, `natural_accepted_by_stream` and
            // `active_streams_by_round` are the v1.2 COHORT trio; the
            // single-stream form carries `acceptance_lengths` alone, whose
            // length already IS R.
            if session.cohort {
                response.rounds = audit.rounds
                response.naturalAcceptedByStream = audit.naturalAcceptedByStream
                response.activeStreamsByRound = audit.activeStreamsByRound
            }
        }
        // `spec_decoder` and `verify_replay_disagreements` have no journal
        // field at this pin, so both stay ABSENT — which is not the same
        // claim as naming the request's own mode back, nor as reporting 0.

        // Timed work for a free-run window is engine STEPS, not tokens: one
        // per verify round when the engine journals rounds, otherwise one per
        // committed token (a serial window's steps and tokens coincide).
        completedWork += audit?.rounds ?? count
        return response
    }

    /// Drain exactly `count` committed tokens from one stream.
    private func drain(
        session: FreeRunSession, slot: Int, count: Int
    ) async throws -> [Int] {
        var out: [Int] = []
        // The iterator is moved out and back rather than mutated in place:
        // a mutating call across an `await` on `session.streams[slot]` would
        // hold an exclusive access over a suspension point.
        var iterator = session.streams[slot]
        defer { session.streams[slot] = iterator }
        while out.count < count {
            guard let event = await iterator.next() else {
                throw WorkerError.streamEndedEarly(slot: slot, got: out.count, want: count)
            }
            switch event {
            case .delta(_, let tokens, _):
                out.append(contentsOf: tokens)
            case .finished:
                guard out.count >= count else {
                    throw WorkerError.streamEndedEarly(
                        slot: slot, got: out.count, want: count)
                }
            }
        }
        return Array(out.prefix(count))
    }

    /// Trusted-build oracle: teacher-forced replay of the candidate's own
    /// committed journal, through the engine's own path.
    private func cohortReferenceReplay(_ request: WorkerRequest) throws -> WorkerResponse {
        guard trusted else {
            throw WorkerError.capabilityNotAdvertised(
                WorkerCapability.cohortReferenceReplay)
        }
        let width = request.replayWidth ?? "cohort"
        guard width == "cohort" || width == "canonical" else {
            throw WorkerError.badReplayWidth(width)
        }
        let seeds = try require(request.replaySeedsByStream, "replay_seeds_by_stream")
        let committed = try require(request.committedByStream, "committed_by_stream")
        guard seeds.count == committed.count else {
            throw WorkerError.batchMismatch(
                requested: seeds.count, streams: committed.count)
        }
        let engine = try runner.makeEngine(
            try engineBuild(batch: 1, spec: SpecConfig(mode: DecoderID.serial.rawValue)))

        var streams: [CohortReferenceReplayReport.Stream] = []
        for (slot, seed) in seeds.enumerated() {
            let journal = committed[slot]
            let reference = try engine.teacherForcedTop1(
                promptTokens: seed, continuation: journal)
            streams.append(
                CohortReferenceReplayReport.Stream(
                    slot: slot,
                    positions: zip(journal, reference).map {
                        CohortReferenceReplayReport.Position(
                            committedToken: $0, sequentialArgmax: $1)
                    }))
        }

        var response = WorkerResponse(id: request.id, ok: true, nonce: nonce)
        response.effectiveBatchSize = seeds.count
        response.cohortReferenceReplay = CohortReferenceReplayReport(
            replayWidth: width, streams: streams)
        return response
    }

    // MARK: Helpers

    private func effectiveSpec(_ requested: SpecConfig?) throws -> SpecConfig {
        guard let requested else { return SpecConfig(mode: DecoderID.serial.rawValue) }
        // A `spec` under plain v1 is refused, never honoured and never
        // ignored: the parent is asking for a surface this worker did not
        // advertise, and silently running serial while it believes it asked
        // for mtp is how a leg measures one thing and is scored as another.
        guard speculative else { throw WorkerError.speculativeProtocolRequired }
        guard runner.loadedDecoders.contains(DecoderID(requested.mode)) else {
            throw WorkerError.modeNotAdvertised(requested.mode)
        }
        if requested.mode == DecoderID.mtp.rawValue {
            return SpecConfig(mode: requested.mode, mtp: MtpSpec(depth: requested.depth))
        }
        return SpecConfig(mode: requested.mode)
    }

    private func require<T>(_ value: T?, _ field: String) throws -> T {
        guard let value else { throw WorkerError.missingField(field) }
        return value
    }

    private func releaseSessions() {
        stepper = nil
        freeRun = nil
    }

    private static let greedySampling = CBv2SamplingParams(temperature: 0, topP: 1, topK: 0)
}

/// Worker-level refusals. Every one is an `ok:false` line.
public enum WorkerError: Error, CustomStringConvertible, Equatable {
    case missingField(String)
    case modeNotAdvertised(String)
    case capabilityNotAdvertised(String)
    case batchMismatch(requested: Int, streams: Int)
    case noFreeRunSession
    case streamEndedEarly(slot: Int, got: Int, want: Int)
    case badReplayWidth(String)
    case speculativeProtocolRequired

    public var description: String {
        switch self {
        case .missingField(let field):
            return "request is missing \(field)"
        case .modeNotAdvertised(let mode):
            return "spec mode \(mode) was not advertised"
        case .capabilityNotAdvertised(let capability):
            return "\(capability) is not served by this adapter"
        case .batchMismatch(let requested, let streams):
            return "batch_size \(requested) disagrees with \(streams) streams"
        case .noFreeRunSession:
            return "free_decode_run without a free_decode_begin"
        case .streamEndedEarly(let slot, let got, let want):
            return "stream \(slot) produced \(got) of \(want) tokens"
        case .badReplayWidth(let width):
            return "replay_width \(width) is not cohort or canonical"
        case .speculativeProtocolRequired:
            return "spec requires --speculative-protocol "
                + "\(SpeculativeProtocol.version)"
        }
    }
}
