// MockRunner.swift
//
// A scripted `Runner` for the protocol tests: no weights, no Metal, no MLX
// allocator. Its token outputs are a pure function of the input, chosen to
// reproduce the shared conformance fixture
// (`Resources/engine-wire-v1-adapter.ndjson`, pinned on both sides).

import Foundation
import MLXLMCommon
import MLXRunners

// MARK: - Token rules

enum MockTokens {
    /// Which answer block a prompt lands in.
    ///
    /// In the fixture, a prompt whose first token is in the 20s or 50s is a
    /// DECODE seed (`decode_begin`, `free_decode_begin`) and answers in the
    /// 200000 block; every other prompt is a prefill/correctness anchor and
    /// answers in the 100000 block.
    static func block(_ prompt: [Int]) -> Int {
        guard let first = prompt.first else { return 100_000 }
        return [2, 5].contains(first / 10) ? 200_000 : 100_000
    }

    /// A whole-prompt forward answers the block base plus the prompt length.
    static func promptAnswer(_ prompt: [Int]) -> Int {
        block(prompt) + prompt.count
    }

    /// A single forced token answers itself plus one.
    static func stepAnswer(_ token: Int) -> Int { token + 1 }

    /// First token an engine free-run emits.
    ///
    /// A decode window has already answered its seed forward
    /// (`promptAnswer`), so its stream continues from there; a `correctness`
    /// window has no seed response and starts at the block base.
    static func freeRunStart(_ prompt: [Int]) -> Int {
        block(prompt) == 200_000 ? promptAnswer(prompt) : block(prompt)
    }
}

// MARK: - Stepper

final class MockStepper: TeacherForcedStepper {
    private(set) var forwards = 0
    private var begun = false

    func begin() throws {
        begun = true
        forwards = 0
    }

    func forward(_ tokens: [Int]) throws -> StepOutput {
        guard begun else { throw StepperError.notBegun }
        guard !tokens.isEmpty else { throw StepperError.emptyForward }
        forwards += 1
        let argmax =
            tokens.count == 1
            ? MockTokens.stepAnswer(tokens[0])
            : MockTokens.promptAnswer(tokens)
        // A descending ladder from 10.0, so the top-logit margin is 1.0 and
        // the ordering is unambiguous.
        let top = (0 ..< 8).map { index in
            (token: argmax + index, logit: 10.0 - Double(index))
        }
        return StepOutput(argmax: argmax, topLogits: top, margin: 1.0)
    }
}

// MARK: - Engine

/// The audit a scripted engine replays for one free-run window.
struct MockRoundScript: Sendable {
    var draftedTotal: Int
    var acceptedTotal: Int
    var acceptanceLengths: [Int]
}

final class MockEngine: CBv2Engine, CBv2MTPCountersReporting, CBv2FreeRunRoundAuditing,
    @unchecked Sendable
{
    private let script: MockRoundScript?
    private let lock = NSLock()
    private var reads = 0

    init(script: MockRoundScript?) {
        self.script = script
    }

    func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        let start = MockTokens.freeRunStart(request.promptTokens)
        let maxTokens = request.maxTokens
        return AsyncStream { continuation in
            for index in 0 ..< maxTokens {
                continuation.yield(
                    .delta(text: "", tokens: [start + index], logprobs: nil))
            }
            continuation.yield(
                .finished(
                    reason: .length,
                    usage: CBv2Usage(
                        promptTokens: request.promptTokens.count,
                        completionTokens: maxTokens)))
            continuation.finish()
        }
    }

    func cancel(_ id: CBv2RequestID) {}

    func capacity() -> CBv2CapacitySnapshot {
        CBv2CapacitySnapshot(
            activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0,
            kvBytesCapacity: 0, kvBytesBackendCapacity: 0, kvBytesReserved: 0,
            activeTokens: 0)
    }

    func shutdown() async {}

    /// Cumulative and monotonic, like the real engine's: the FIRST read is
    /// the baseline `free_decode_begin` takes before the window drafts
    /// anything, and every read after it is the drained window's total.
    func mtpMetricsSnapshot() -> CBv2MTPMetrics? {
        guard let script else { return nil }
        lock.lock()
        defer { lock.unlock() }
        defer { reads += 1 }
        var metrics = CBv2MTPMetrics()
        metrics.draftedTokens = reads == 0 ? 0 : script.draftedTotal
        metrics.acceptedTokens = reads == 0 ? 0 : script.acceptedTotal
        return metrics
    }

    func freeRunRoundAudit() -> FreeRunRoundAudit? {
        guard let script else { return nil }
        return FreeRunRoundAudit(acceptanceLengths: script.acceptanceLengths)
    }
}

// MARK: - Runner

/// Manifests the CONTRACT itself pins, built here from the document.
///
/// The section 11 manifest for Qwen 3.8 Flash-Next is the cross-repo digest
/// vector, and the shared conformance fixture's hello carries its runner
/// identity. No `Qwen4ExpRunner` exists on this branch — it lands from
/// `feat/qwen4-exp-cbv2` — so this is DECLARATION DATA only: nothing here
/// loads, registers, or serves that family.
enum ContractManifests {
    static let sectionEleven = RunnerManifest(
        runnerID: "layr/qwen4exp-125b-a6b",
        modelTypes: ["qwen4_exp", "qwen4_exp_text"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: false,
            supportsCompiledDecode: false,
            supportsPackedPrefill: false,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: false),
        kvBackends: [.contiguous],
        decoders: [
            DecoderDeclaration(
                mode: "serial", drafter: .none, state: .stateless, depth: nil),
            DecoderDeclaration(
                mode: "mtp", drafter: .embeddedHead, state: .requestStateful,
                depth: 1 ... 3),
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(
                batch: .single, timing: .teacherForced, perStreamTiming: false),
        ],
        multimodal: false,
        recurrentLayers: true,
        requiresKeepMask: true)
}

/// The mock adapter's manifest, DECODED from the file benchd checked in
/// beside the fixture (`Resources/engine-wire-v1-adapter.manifest.json`).
///
/// Decoded rather than declared in Swift on purpose: both sides load the
/// SAME bytes, so the hello's `manifest_sha256` is a digest this repo
/// actually computed over the other repo's file, not two declarations that
/// happen to agree today.
enum FixtureManifest {
    static let mockAdapter: RunnerManifest = {
        guard
            let url = Bundle.module.url(
                forResource: "engine-wire-v1-adapter.manifest", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let manifest = try? JSONDecoder().decode(RunnerManifest.self, from: data)
        else {
            fatalError("the shared mock-adapter manifest is missing or undecodable")
        }
        return manifest
    }()
}

/// The scripted runner the conformance fixture is driven over.
///
/// Its manifest is the shared mock-adapter file, so `hello.backend` and
/// `hello.runner.manifest_sha256` are two readings of ONE declaration —
/// which is what §6.1 requires and what a hand-written hello field would
/// break.
final class MockRunner: Runner, @unchecked Sendable {

    static var manifest: RunnerManifest { FixtureManifest.mockAdapter }

    /// The window audit the scripted engine replays.
    let script: MockRoundScript?

    var servingModel: any LanguageModel {
        preconditionFailure("mock runner holds no model")
    }
    var tokenizer: any MLXLMCommon.Tokenizer {
        preconditionFailure("mock runner holds no tokenizer")
    }
    let eosTokenIDs: Set<Int> = []
    let layerKinds: [CBv2LayerKind] = []
    let loadedDecoders: [DecoderID] = [.serial, .mtp]
    let headProvenance: HeadProvenance? = nil
    let loadedModelType = "qwen4_exp_text"

    init(
        script: MockRoundScript? = MockRoundScript(
            draftedTotal: 6, acceptedTotal: 4, acceptanceLengths: [3, 1, 2])
    ) {
        self.script = script
    }

    static func load(_ directory: URL, options: RunnerLoadOptions) async throws -> MockRunner {
        MockRunner()
    }

    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        guard loadedDecoders.contains(build.decoder) else {
            throw RunnerError.decoderNotLoaded(
                requested: build.decoder.rawValue,
                loaded: loadedDecoders.map(\.rawValue))
        }
        return MockEngine(script: build.decoder == .serial ? nil : script)
    }

    func makeStepper() throws -> any TeacherForcedStepper { MockStepper() }
}

/// The same mock with every FREE-RUN regime removed. A worker over this one
/// must REFUSE `free_decode_begin` rather than serve a regime the manifest
/// does not declare (contract §6.2 rule 3).
final class TeacherForcedOnlyMockRunner: Runner, @unchecked Sendable {

    static let manifest = RunnerManifest(
        runnerID: "layr/mock-teacher-forced",
        modelTypes: ["mock-teacher-forced"],
        engine: MockRunner.manifest.engine,
        kvBackends: [.contiguous],
        decoders: MockRunner.manifest.decoders,
        regimes: [
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false)
        ],
        multimodal: false,
        recurrentLayers: false,
        requiresKeepMask: false)

    private let inner = MockRunner()

    var servingModel: any LanguageModel { inner.servingModel }
    var tokenizer: any MLXLMCommon.Tokenizer { inner.tokenizer }
    var eosTokenIDs: Set<Int> { inner.eosTokenIDs }
    var layerKinds: [CBv2LayerKind] { inner.layerKinds }
    var loadedDecoders: [DecoderID] { inner.loadedDecoders }
    var headProvenance: HeadProvenance? { inner.headProvenance }
    var loadedModelType: String { inner.loadedModelType }

    static func load(
        _ directory: URL, options: RunnerLoadOptions
    ) async throws -> TeacherForcedOnlyMockRunner {
        TeacherForcedOnlyMockRunner()
    }

    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        try inner.makeEngine(build)
    }
    func makeStepper() throws -> any TeacherForcedStepper { try inner.makeStepper() }
}

// MARK: - Transport

/// Scripted transport: the request lines go in, the response lines come out.
final class ScriptedTransport: BenchWorkerTransport {
    private var pending: [String]
    private(set) var written: [String] = []

    init(lines: [String]) {
        self.pending = lines
    }

    func readLine() -> String? {
        pending.isEmpty ? nil : pending.removeFirst()
    }

    func write(line: String) {
        written.append(line)
    }
}

/// A worker with no MLX allocator: it reports the peak RSS the fixture
/// pins and nothing else. Absence is a real answer — the schema reads an
/// absent `cache_memory` as "not asserted", which is a different claim
/// from zero.
struct FixtureMemoryReporter: WorkerMemoryReporter {
    func preDrainSnapshot() -> (active: Int, cache: Int, peak: Int)? { nil }
    func drain() {}
    func cacheMemoryAfterDrain() -> Int? { nil }
    func peakRAMGB() -> Double? { 18.5 }
}
