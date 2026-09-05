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

/// Per-ROUND free-run audit an engine may expose.
///
/// `EngineV2` deliberately does NOT conform at this pin: it keeps cumulative
/// MTP counters and no per-round journal, so `acceptance_lengths`,
/// `natural_accepted_by_stream`, `active_streams_by_round`, `rounds`,
/// `depth_clamp_reasons` and `verify_replay_disagreements` are ABSENT on its
/// responses rather than fabricated — and absent means NOT REPORTED, which
/// is not the same claim as zero.
public protocol CBv2FreeRunRoundAuditing: AnyObject {
    /// Audit for the window since the last `free_decode_begin`.
    func freeRunRoundAudit() -> FreeRunRoundAudit?
}

/// Per-round free-run audit. Every field optional: an engine reports what it
/// keeps and nothing else.
public struct FreeRunRoundAudit: Sendable, Equatable {
    public var specDecoder: String?
    public var acceptanceLengths: [Int]?
    public var naturalAcceptedByStream: [[Int]]?
    public var rounds: Int?
    public var activeStreamsByRound: [Int]?
    public var depthClampReasons: [String: Int]?
    public var verifyReplayDisagreements: Int?

    public init(
        specDecoder: String? = nil,
        acceptanceLengths: [Int]? = nil,
        naturalAcceptedByStream: [[Int]]? = nil,
        rounds: Int? = nil,
        activeStreamsByRound: [Int]? = nil,
        depthClampReasons: [String: Int]? = nil,
        verifyReplayDisagreements: Int? = nil
    ) {
        self.specDecoder = specDecoder
        self.acceptanceLengths = acceptanceLengths
        self.naturalAcceptedByStream = naturalAcceptedByStream
        self.rounds = rounds
        self.activeStreamsByRound = activeStreamsByRound
        self.depthClampReasons = depthClampReasons
        self.verifyReplayDisagreements = verifyReplayDisagreements
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
    private let build: String
    private let device: String
    private let kvBytesCapacity: Int
    private let maxDecodeTokens: Int
    private let memory: any WorkerMemoryReporter
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
        build: String,
        device: String,
        kvBytesCapacity: Int,
        maxDecodeTokens: Int = 4096,
        memory: any WorkerMemoryReporter = MLXMemoryReporter(),
        nonce: String = UUID().uuidString
    ) {
        self.runner = runner
        self.manifest = type(of: runner).manifest
        self.transport = transport
        self.trusted = trusted
        self.build = build
        self.device = device
        self.kvBytesCapacity = kvBytesCapacity
        self.maxDecodeTokens = maxDecodeTokens
        self.memory = memory
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
        response.headProvenance = runner.headProvenance.map(WireHeadProvenance.init)
        response.maxBatchSize = manifest.regimes.map(\.batch.maxWidth).filter { $0 > 1 }.max()
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
            if request.spec != nil {
                response.effectiveSpec = try effectiveSpec(request.spec)
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
        let baseline: CBv2MTPMetrics?
        var streams: [AsyncStream<CBv2Event>.Iterator]

        init(
            engine: any CBv2Engine, batch: Int, spec: SpecConfig, cohort: Bool,
            baseline: CBv2MTPMetrics?, streams: [AsyncStream<CBv2Event>.Iterator]
        ) {
            self.engine = engine
            self.batch = batch
            self.spec = spec
            self.cohort = cohort
            self.baseline = baseline
            self.streams = streams
        }
    }

    private func freeDecodeBegin(_ request: WorkerRequest) async throws -> WorkerResponse {
        guard manifest.regimes.contains(where: { $0.timing == .freeRun }) else {
            throw WorkerError.capabilityNotAdvertised(WorkerCapability.freeRunDecode)
        }
        let spec = try effectiveSpec(request.spec)

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

        let session = FreeRunSession(
            engine: engine, batch: batch, spec: spec, cohort: cohort,
            baseline: (engine as? any CBv2MTPCountersReporting)?.mtpMetricsSnapshot(),
            streams: iterators)
        var seedTokens: [Int] = []
        for slot in 0 ..< batch {
            seedTokens.append(
                contentsOf: try await drain(session: session, slot: slot, count: 1))
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

        // AUDIT, all read off engine surfaces — never computed here.
        let audit = (session.engine as? any CBv2FreeRunRoundAuditing)?.freeRunRoundAudit()
        response.acceptanceLengths = audit?.acceptanceLengths
        if let metrics = (session.engine as? any CBv2MTPCountersReporting)?
            .mtpMetricsSnapshot()
        {
            let base = session.baseline
            response.draftedTotal = metrics.draftedTokens - (base?.draftedTokens ?? 0)
            response.acceptedTotal = metrics.acceptedTokens - (base?.acceptedTokens ?? 0)
        }
        response.committedTotal = byStream.reduce(0) { $0 + $1.count }
        response.specDecoder = audit?.specDecoder
        response.naturalAcceptedByStream = audit?.naturalAcceptedByStream
        response.rounds = audit?.rounds
        response.activeStreamsByRound = audit?.activeStreamsByRound
        response.depthClampReasons = audit?.depthClampReasons
        response.verifyReplayDisagreements = audit?.verifyReplayDisagreements

        // Timed work for a free-run window is engine STEPS, not tokens: one
        // per verify round when the engine journals rounds, otherwise one per
        // committed token (a serial window's steps and tokens coincide).
        completedWork += audit?.acceptanceLengths?.count ?? audit?.rounds ?? count
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
        }
    }
}
