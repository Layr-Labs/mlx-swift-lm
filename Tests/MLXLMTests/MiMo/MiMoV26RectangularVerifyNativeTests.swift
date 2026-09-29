import CryptoKit
import Cmlx
import Foundation
import MLX
import MLXHuggingFace
@testable import MLXLLM
import Tokenizers
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Reuses the existing strict tiny-target/owned-media fixture mechanism.
/// Rectangular candidate gates only; all methods are source-prepared, UNRUN.
/// Requires an explicitly supplied small audio-compatible target fixture; this
/// class never weakens the selected sidecar hash/geometry or invents a loaded
/// owner. Reservation counters are SDK lifetime witnesses, NOT host ledger or
/// measured-peak qualification. Each fault selector needs its own process.
final class MiMoV26RectangularVerifyNativeTests: XCTestCase {
    private enum Failure: Error { case inputRequired, nativeLaneRequired, noProof, injected }
    private enum Profile: Sendable, Equatable { case audio, visual, text }
    private final class RootPermit: MiMoV26SerialLoadReservation, Sendable {
        let request: MiMoV26SerialLoadRequest
        let reservedLoadBytes: UInt64
        init(_ request: MiMoV26SerialLoadRequest) {
            self.request = request; reservedLoadBytes = request.requiredLoadBytes
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private final class AudioPermit: MiMoV26AudioSidecarLoadReservation, @unchecked Sendable {
        let request: MiMoV26AudioSidecarLoadRequest
        let reservedLoadBytes: UInt64
        private let lock = NSLock()
        private var invalid = false
        init(_ request: MiMoV26AudioSidecarLoadRequest) {
            self.request = request; reservedLoadBytes = request.requiredLoadBytes
        }
        func revoke() { lock.withLock { invalid = true } }
        func validateActive() throws {
            if lock.withLock({ invalid }) { throw MiMoV26AudioSidecarError.invalidatedOwner }
        }
    }
    private final class WorkPermit: MiMoV26MediaWorkReservation, @unchecked Sendable {
        private let lock = NSLock()
        private var digest: String?
        private var charge = 0
        private var releaseCount = 0
        private var failures: [MiMoV26FailedMediaWork] = []
        private var validations = 0
        private var kinds: [String] = []
        private var frames: [Int] = []
        var onValidation: (@Sendable (Int) -> Void)?
        var bytes: Int { lock.withLock { charge } }
        var retired: Int { lock.withLock { releaseCount } }
        var retainedFailures: Int { lock.withLock { failures.count } }
        var failedAudioRootCounts: [Int] {
            lock.withLock { failures.flatMap(\.retainedAudioFailureRootCountsForTesting) }
        }
        var orderedKinds: [String] { lock.withLock { kinds } }
        var pcmFrames: [Int] { lock.withLock { frames } }
        func admit(_ plan: MiMoV26MultimodalPlan, bytes: Int) throws {
            try lock.withLock {
                guard digest == nil, bytes > 0 else { throw MiMoV26MultimodalError.reservationRejected }
                digest = plan.preparationSHA256; charge = bytes
                kinds = plan.spans.map { $0.kind.rawValue }
                frames = plan.audioPlan?.pcmDescriptors?.map(\.frameCount) ?? []
            }
        }
        func validate(plan: MiMoV26MultimodalPlan) throws {
            let count: Int = try lock.withLock {
                guard digest == plan.preparationSHA256, charge > 0, releaseCount == 0 else {
                    throw MiMoV26MultimodalError.reservationRejected
                }
                validations += 1
                return validations
            }
            onValidation?(count)
        }
        func retainAfterFailedDrain(_ work: MiMoV26FailedMediaWork) {
            lock.withLock { if !failures.contains(where: { $0 === work }) { failures.append(work) } }
        }
        func retire() throws {
            try lock.withLock {
                guard failures.isEmpty, releaseCount == 0 else { throw Failure.noProof }
                charge = 0; releaseCount = 1
            }
        }
    }
    private final class Cancel: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        private var beforeRoots: Int?
        var rootsBeforeAudio: Int? { lock.withLock { beforeRoots } }
        func recordRootsBeforeAudio(_ value: Int) { lock.withLock { beforeRoots = value } }
        var cancelled: Bool { lock.withLock { value } }
        func set() { lock.withLock { value = true } }
    }
    private struct Loader: TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let raw = try await AutoTokenizer.from(modelFolder: directory)
            return #adaptHuggingFaceTokenizer(raw)
        }
    }
    private struct Fixture: Sendable {
        let container: ModelContainer
        let engine: EngineV2
        let construction: NativeConstructionWork
        let audioPermit: AudioPermit?
        let contract: CBv2NativeExecutionContract
    }
    private struct Built: Sendable {
        let engine: EngineV2
        let contract: CBv2NativeExecutionContract
    }
    private static let limits = MiMoV26MultimodalLimits(
        maximumMedia: 4, maximumVideoFrames: 4, maximumPromptTokens: 2048,
        maximumMetadataBytes: 1 << 20, maximumMetadataNodes: 10000, maximumMetadataDepth: 32,
        pixels: .init(maximumInputElements: 100000, maximumOutputElements: 100000,
            maximumWorkingBytes: 1 << 20),
        vision: .init(maximumPatches: 256, maximumAttentionScoreElements: 131072),
        audio: .init(maximumClips: 2, maximumChannels: 1, maximumSampleRate: 24000,
            maximumInputSamples: 48000, maximumResampledSamples: 48000,
            maximumResampleCoefficients: 100000, maximumMelFrames: 256, maximumSegments: 4,
            maximumPaddedMelFrames: 256, maximumWorkingElements: 64_000_000,
            frontendFrameBlockSize: 8, rvqTileFrames: 8),
        audioPatch: .init(maximumClips: 2, maximumFrames: 256, maximumPatches: 64,
            maximumWorkingElements: 16_000_000))

    private func fixture(profile: Profile = .text, installAudio: Bool = false,
                         enableMTP: Bool = true, mode: CBv2MTPVerificationMode = .rectangular,
                         depth: Int = 3, shutdownTimeout: TimeInterval = 10) async throws -> Fixture {
        let env = ProcessInfo.processInfo.environment
        guard env["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1",
              env["MIMO_V26_RECTANGULAR_NATIVE_TESTS"] == "1" else { throw Failure.nativeLaneRequired }
        guard let path = env["MIMO_V26_MANAGED_AUDIO_FIXTURE_ROOT"] else { throw Failure.inputRequired }
        if installAudio && env["MIMO_V26_RECTANGULAR_AUDIO_NATIVE_TESTS"] != "1" {
            throw XCTSkip("Authentic selected codec requires separate audio native opt-in")
        }
        let root = URL(fileURLWithPath: path)
        // Deliberately excludes a full target: the real selected SIDE-CAR is
        // separate and authenticated by its existing loader below.
        let provenanceFile = try FileHandle(forReadingFrom: root.appendingPathComponent("provenance.json"))
        defer { try? provenanceFile.close() }
        let provenanceData = try XCTUnwrap(provenanceFile.read(upToCount: 65537))
        guard provenanceData.count <= 65536 else { throw Failure.inputRequired }
        let p = try XCTUnwrap(JSONSerialization.jsonObject(with: provenanceData) as? [String:String])
        let provenance = try MiMoV26ConvertedProvenance(artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]), sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 64 << 20, maximumTotalFileBytes: 128 << 20))
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let config = plan.bundlePlan.configuration
        guard config.vocabularySize > 151674, config.hiddenSize <= 64,
              config.numHiddenLayers <= 4 else { throw Failure.inputRequired }
        let prepared = try await MiMoV26ModelFactory.prepare(request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let construction = NativeConstructionWork()
        defer { if construction.snapshot.isRetainedFault { _ = Unmanaged.passRetained(construction) } }
        let container = try await MiMoV26ModelFactory.loadContainer(session: session,
            reservation: RootPermit(session.request), prepared: prepared, retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        let sidecar: SendableBox<MiMoV26AudioSidecarLoadSession>?
        let permit: AudioPermit?
        if installAudio {
            let value = try MiMoV26AudioSidecarLoadSession(root: root, mainConfiguration: config,
                mainConfigurationSHA256: plan.bundlePlan.configSHA256)
            sidecar = SendableBox(value); permit = AudioPermit(value.request)
        } else { sidecar = nil; permit = nil }
        let built = try await MiMoV26ModelFactory.withNativeConstruction(
            container: container, retaining: construction) { model, scope in
            if let sidecar, let permit {
                _ = try model.installAudioSidecar(session: sidecar.consume(), reservation: permit,
                    retaining: scope, isCancelled: { false })
            }
            let binding = try model.makeCBv2Binding(enableMTP: enableMTP,
                verificationMode: enableMTP ? mode : .serialTarget)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let resources: MiMoV26ManagedMediaExecutionResources
            switch profile {
            case .audio:
                resources = try model.makeManagedAudioExecutionResources(binding: binding,
                    bytesCapacity: 32 << 20, limits: Self.limits, retaining: scope)
            case .visual:
                resources = try model.makeManagedMediaExecutionResources(binding: binding,
                    bytesCapacity: 32 << 20, limits: Self.limits, retaining: scope)
            case .text:
                let text = try binding.adapter.makeNativeExecutionResources(bytesCapacity: 32 << 20, retaining: scope)
                resources = .init(backend: text.backend, cacheProvider: text.cacheProvider, contract: text.contract)
            }
            // A wrong requested mode must neither consume nor relabel the
            // authentic same-owner ticket. The actual engine consumes it next.
            let issuedMode: CBv2MTPVerificationMode = enableMTP ? mode : .serialTarget
            let wrongMode: CBv2MTPVerificationMode = issuedMode == .serialTarget ? .rectangular : .serialTarget
            XCTAssertFalse(resources.contract.consume(model: binding.adapter, backend: resources.backend,
                cacheProvider: resources.cacheProvider, assistant: binding.assistant,
                mtpVerificationMode: wrongMode))
            let engine = EngineV2(model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: 16,
                    prefillChunkSize: 3, maxConcurrentPartialPrefills: 1, enablePrefixCache: false),
                loopConfig: .init(stepTimeout: 60, watchdogInterval: 0.01, shutdownTimeout: shutdownTimeout),
                mtpDrafter: binding.assistant,
                mtpConfig: .init(enabled: enableMTP, maxDraftTokens: depth,
                    maxSpeculativeBatch: 1, fixedDraftTokens: depth,
                    verificationMode: enableMTP ? mode : .serialTarget),
                nativeCompletionTracking: true, nativeExecutionContract: resources.contract)
            try scope.retainOwner(engine)
            XCTAssertFalse(resources.contract.consume(model: binding.adapter, backend: resources.backend,
                cacheProvider: resources.cacheProvider, assistant: binding.assistant,
                mtpVerificationMode: issuedMode), "engine must consume the authentic ticket exactly once")
            return Built(engine: engine, contract: resources.contract)
        }
        guard case .completed(let completion) = construction.snapshot.disposition else { throw Failure.noProof }
        try await construction.sealForPublication(completion)
        return .init(container: container, engine: built.engine, construction: construction,
            audioPermit: permit, contract: built.contract)
    }
    private func input(mixed: Bool = false) throws -> MiMoV26MultimodalInput {
        let pcm = try MiMoV26DecodedPCM(samples: (0..<2400).map { sin(Float($0) * 0.02) * 0.05 },
            descriptor: .init(sourceIdentity: "synthetic-sine-owned-pcm", channels: 1,
                frameCount: 2400, sampleRate: 24000))
        var parts: [MiMoV26MultimodalContent] = []
        if mixed { parts.append(.image(.init(height: 4, width: 4,
            planarRGB: (0..<48).map { Float(($0 * 17) % 256) }))) }
        parts += [.text("listen"), .audio(pcm), .text("describe")]
        return .init(messages: [.init(role: .user, content: parts)], maximumOutputTokens: 2)
    }
    private func prepared(_ fixture: Fixture, input: MiMoV26MultimodalInput,
        reservation: WorkPermit = WorkPermit(), cancel: Cancel = Cancel()) async throws -> (CBv2Request, WorkPermit) {
        let request = try await fixture.container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            if fixture.contract.supportsDecodedAudioMedia {
                return try model.prepareManagedDecodedAudioMedia(input, engine: fixture.engine,
                    authorize: { plan, bytes in try reservation.admit(plan, bytes: bytes); return reservation },
                    retire: { _ in try reservation.retire() }, isCancelled: { cancel.cancelled })
            }
            return try model.prepareManagedDecodedMedia(input, engine: fixture.engine,
                authorize: { plan, bytes in try reservation.admit(plan, bytes: bytes); return reservation },
                retire: { _ in try reservation.retire() }, isCancelled: { cancel.cancelled })
        }
        return (request, reservation)
    }
    private func shutdown(_ fixture: Fixture) async throws {
        guard case .quiescent(let receipt) = await fixture.engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(fixture.engine)
            _ = Unmanaged.passRetained(fixture.container)
            _ = Unmanaged.passRetained(fixture.construction)
            throw Failure.noProof
        }
        XCTAssertEqual(receipt.executionContractID, fixture.contract.id)
        if fixture.contract.supportsDecodedAudioMedia {
            try await fixture.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                try model.releaseManagedAudioAfterNativeRetirement(receipt)
                XCTAssertNil(model.resources.audioSidecar)
            }
        }
    }

    private final class Gate: @unchecked Sendable {
        let entered: XCTestExpectation
        private let lock = NSLock()
        private var first = true
        private let resumed = DispatchSemaphore(value: 0)
        init(_ entered: XCTestExpectation) { self.entered = entered }
        func hold() {
            guard lock.withLock({ let value = first; first = false; return value }) else { return }
            entered.fulfill(); _ = resumed.wait(timeout: .now() + 15)
        }
        func release() { resumed.signal() }
    }
    private func text(_ id: UInt64, count: Int = 11, budget: Int = 16) -> CBv2Request {
        .init(id: .init(id), promptTokens: (0..<count).map { 20 + $0 % 11 },
            sampling: .init(temperature: 0), maxTokens: budget, prefixCacheEnabled: false)
    }
    private func normalProcess() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_RECTANGULAR_FAULT_CASE"] == nil else {
            throw XCTSkip("Normal cells must not share a retained-fault process")
        }
    }
    func testTrackedOFFSerialRectangularTokensAndCompletedWindowEngagement() async throws {
        try normalProcess()
        for depth in 1...3 {
            var outputs: [[Int]] = []
            for selection in 0..<3 {
                let mode: CBv2MTPVerificationMode = selection == 2 ? .rectangular : .serialTarget
                let f = try await fixture(enableMTP: selection > 0, mode: mode, depth: depth)
                XCTAssertNil(f.engine.nativeCompletionFault)
                XCTAssertEqual(f.contract.mtpVerificationMode, mode)
                var evaluatedWidths: [Int] = []
                f.engine.loopForTesting.onEngineQueueSync { [weak engine = f.engine] in
                    engine?.loopForTesting.nativeRetirementBoundaryForTesting = { phase, step in
                        guard phase == "beforeMTPFinalization", let verify = step?.mtpRound?.verify else { return }
                        XCTAssertEqual(verify.rows.count, 1)
                        XCTAssertEqual(verify.lastHidden.shape[0], 1)
                        XCTAssertEqual(verify.lastHidden.shape[1], 1 + verify.k)
                        evaluatedWidths.append(verify.lastHidden.shape[1])
                    }
                }
                let submission = try f.engine.submitWithNativeRetirement(text(1))
                let result = await cbv2SchedCollect(submission.events)
                await submission.retirement.wait()
                XCTAssertEqual(result.finishReason, .length)
                XCTAssertEqual(result.tokens.count, 16)
                outputs.append(result.tokens)
                if selection > 0 {
                    let metrics = try XCTUnwrap(f.engine.mtpMetricsSnapshot())
                    XCTAssertGreaterThan(metrics.rounds, 0)
                    XCTAssertGreaterThan(metrics.draftedTokens, 0)
                    XCTAssertFalse(evaluatedWidths.isEmpty)
                    if selection == 2 {
                        XCTAssertEqual(metrics.serialVerificationRounds, 0)
                        XCTAssertGreaterThan(metrics.rectangularVerificationRounds, 0)
                    } else {
                        XCTAssertEqual(metrics.rectangularVerificationRounds, 0)
                        XCTAssertGreaterThan(metrics.serialVerificationRounds, 0)
                    }
                    print("MIMO_VERIFY_ENGAGEMENT mode=\(mode) depth=\(depth) widths=\(evaluatedWidths) rounds=\(metrics.rounds)")
                } else { XCTAssertTrue(evaluatedWidths.isEmpty) }
                try await shutdown(f)
                f.engine.loopForTesting.onEngineQueueSync {
                    f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
                    XCTAssertEqual(f.engine.loopForTesting.backend.bytesReserved, 0)
                    XCTAssertEqual(f.engine.loopForTesting.nativeShutdownState?.debugRetainedRootCount, 0)
                }
            }
            XCTAssertEqual(outputs[1], outputs[0])
            XCTAssertEqual(outputs[2], outputs[0])
        }
    }

    /// Same native visual profile, actual owned image features, and genuine
    /// head proposals. Media is never an assistant row; B1 applies only to
    /// the eligible text cohort, not the total running rows.
    func testVisualProfileRetainsMediaAndMixedTextRectangularVerification() async throws {
        try normalProcess()
        for mediaFirst in [true, false] {
            var results: [[[Int]]] = []
            for mode: CBv2MTPVerificationMode in [.serialTarget, .rectangular] {
                let f = try await fixture(profile: .visual, mode: mode, depth: 2)
                XCTAssertTrue(f.contract.supportsDecodedVisualMedia)
                XCTAssertEqual(f.contract.mtpVerificationMode, mode)
                let image = MiMoV26MultimodalInput(messages: [.init(role: .user, content: [
                    .text("look"), .image(.init(height: 4, width: 4,
                        planarRGB: (0..<48).map { Float(($0 * 17) % 256) })), .text("describe")])],
                    maximumOutputTokens: 48)
                var (media, permit) = try await prepared(f, input: image)
                media.id = .init(2); media.sampling = .init(temperature: 0); media.stopTokens = []
                let ordinary = text(1, count: media.promptTokens.count, budget: 16)
                var witnessed = 0
                f.engine.loopForTesting.onEngineQueueSync { [weak engine = f.engine] in
                    guard let loop = engine?.loopForTesting else { return }
                    loop.suspendStepExecutionAtCountForTesting = 0
                    loop.nativeRetirementBoundaryForTesting = { [weak engine] phase, step in
                        guard phase == "beforeMTPFinalization", let step, let verify = step.mtpRound?.verify,
                              let loop = engine?.loopForTesting, let driver = loop.mtp else { return }
                        XCTAssertEqual(verify.rows.map(\.id), [ordinary.id])
                        XCTAssertEqual(verify.lastHidden.shape[0], 1)
                        XCTAssertLessThanOrEqual(step.assignments.reduce(0) { $0 + $1.numTokens }, 16)
                        guard step.sampledRows.contains(media.id),
                              let rec = loop.scheduler.record(for: media.id), !rec.isPaused, !rec.cancelRequested else { return }
                        XCTAssertTrue(driver.isTargetOnlyMediaForTesting(media.id))
                        XCTAssertNil(driver.assistantStateCountsForTesting(media.id))
                        XCTAssertFalse(driver.hasValidCarry(for: rec))
                        XCTAssertNil(driver.roundMark(for: media.id))
                        witnessed += 1
                    }
                }
                let ordered = mediaFirst ? [media, ordinary] : [ordinary, media]
                let submissions = try ordered.map { try f.engine.submitWithNativeRetirement($0) }
                f.engine.loopForTesting.onEngineQueueSync {
                    XCTAssertEqual(f.engine.loopForTesting.stepCount, 0)
                    f.engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
                }
                async let a = cbv2SchedCollect(submissions[0].events)
                async let b = cbv2SchedCollect(submissions[1].events)
                let pair = await (a, b)
                let collected = [pair.0, pair.1]
                for submission in submissions { await submission.retirement.wait() }
                XCTAssertTrue(collected.allSatisfy { $0.finishReason == .length })
                XCTAssertGreaterThan(witnessed, 0)
                XCTAssertEqual(permit.retired, 1); XCTAssertEqual(permit.bytes, 0)
                let metrics = try XCTUnwrap(f.engine.mtpMetricsSnapshot())
                XCTAssertGreaterThan(metrics.rounds, 0)
                if mode == .rectangular {
                    XCTAssertGreaterThan(metrics.rectangularVerificationRounds, 0)
                    XCTAssertEqual(metrics.serialVerificationRounds, 0)
                }
                results.append(collected.map(\.tokens))
                try await shutdown(f)
                f.engine.loopForTesting.onEngineQueueSync {
                    f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
                }
            }
            XCTAssertEqual(results[1], results[0])
        }
    }

    func testActualAudioProfileKeepsPCMTargetOnlyThenTextCanVerify() async throws {
        try normalProcess()
        let f = try await fixture(profile: .audio, installAudio: true, depth: 2)
        XCTAssertTrue(f.contract.supportsDecodedAudioMedia)
        XCTAssertEqual(f.contract.mtpVerificationMode, .rectangular)
        let (audio, permit) = try await prepared(f, input: input())
        let before = try XCTUnwrap(f.engine.mtpMetricsSnapshot())
        let media = try f.engine.submitWithNativeRetirement(audio)
        let result = await cbv2SchedCollect(media.events); await media.retirement.wait()
        XCTAssertFalse(result.tokens.isEmpty)
        XCTAssertEqual(f.engine.mtpMetricsSnapshot()?.draftedTokens, before.draftedTokens)
        XCTAssertEqual(permit.retired, 1); XCTAssertEqual(permit.bytes, 0)
        let submission = try f.engine.submitWithNativeRetirement(text(10))
        _ = await cbv2SchedCollect(submission.events); await submission.retirement.wait()
        XCTAssertGreaterThan(try XCTUnwrap(f.engine.mtpMetricsSnapshot()).rectangularVerificationRounds, 0)
        try await shutdown(f) // same actual codec owner retires with this generation
    }

    func testCancelAfterEvaluatedVerifyRetainsRootsUntilDrainAndAllowsIDReuse() async throws {
        try normalProcess()
        let f = try await fixture()
        let gate = Gate(expectation(description: "real rectangular verify evaluated before finalization"))
        defer { gate.release() }
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRetirementBoundaryForTesting = { phase, step in
                if phase == "beforeMTPFinalization", step?.mtpRound?.verify != nil { gate.hold() }
            }
        }
        let submission = try f.engine.submitWithNativeRetirement(text(9, budget: 40))
        await fulfillment(of: [gate.entered], timeout: 10)
        let tracking = try XCTUnwrap(f.engine.loopForTesting.nativeShutdownState)
        XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        f.engine.cancel(.init(9))
        XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        gate.release()
        let result = await cbv2SchedCollect(submission.events); await submission.retirement.wait()
        XCTAssertEqual(result.finishReason, .cancelled)
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
            XCTAssertEqual(f.engine.loopForTesting.backend.bytesReserved, 0)
            XCTAssertNil(f.engine.loopForTesting.mtp?.assistantStateCountsForTesting(.init(9)))
        }
        let again = try f.engine.submitWithNativeRetirement(text(9, budget: 8))
        let retried = await cbv2SchedCollect(again.events)
        XCTAssertEqual(retried.finishReason, .length)
        await again.retirement.wait()
        try await shutdown(f)
    }

    func testRequiredVerifyFenceFailureRemainsRetainedAfterLateSuccessfulFence() async throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_RECTANGULAR_FAULT_CASE"]
            == "testRequiredVerifyFenceFailureRemainsRetainedAfterLateSuccessfulFence" else {
            throw XCTSkip("Run this retained-fault selector alone in an authorized native process")
        }
        let f = try await fixture()
        defer {
            // Deliberately restart-only after the actual required fence fails.
            _ = Unmanaged.passRetained(f.engine); _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
        }
        let entered = expectation(description: "required fence refusal after actual rectangular work")
        var armed = false
        let tracking = try XCTUnwrap(f.engine.loopForTesting.nativeShutdownState)
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRequiredAssistantFenceForTesting = { state in
                guard state.stagedInputCount > 0 else { return }
                if !armed { armed = true; entered.fulfill() }
                throw Failure.injected
            }
        }
        let submission = try f.engine.submitWithNativeRetirement(text(4))
        _ = await cbv2SchedCollect(submission.events)
        await fulfillment(of: [entered], timeout: 10)
        let first = await f.engine.shutdownReportingNativeCompletion()
        guard case .incomplete = first else { return XCTFail("required fence failure minted a receipt") }
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRequiredAssistantFenceForTesting = nil
            XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
            XCTAssertGreaterThan(f.engine.loopForTesting.backend.bytesReserved, 0)
        }
        try withError { StreamOrDevice.default.stream.synchronize() }
        let late = await f.engine.shutdownReportingNativeCompletion()
        XCTAssertEqual(first, late)
        XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        // An incomplete native result must not be relabelled retirement by
        // awaiting an ordinary stream's end; no false lease refund is asserted.
    }

    func testCandidateSamplerFallbacksAndColdCancellationDoNotDraft() async throws {
        try normalProcess()
        let f = try await fixture()
        for (index, sampling) in [
            CBv2SamplingParams(temperature: 0.7, seed: 7),
            CBv2SamplingParams(temperature: 0, repetitionPenalty: 1.1),
            CBv2SamplingParams(temperature: 0, logitBias: [3: 1])
        ].enumerated() {
            var request = text(UInt64(index + 1), budget: 8)
            request.sampling = sampling
            let submission = try f.engine.submitWithNativeRetirement(request)
            let result = await cbv2SchedCollect(submission.events); await submission.retirement.wait()
            XCTAssertEqual(result.finishReason, .length)
            XCTAssertEqual(f.engine.mtpMetricsSnapshot()?.draftedTokens, 0)
            XCTAssertEqual(f.engine.mtpMetricsSnapshot()?.rounds, 0)
        }
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.suspendStepExecutionAtCountForTesting = f.engine.loopForTesting.stepCount
        }
        let submission = try f.engine.submitWithNativeRetirement(text(8))
        f.engine.cancel(.init(8))
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        let cancelled = await cbv2SchedCollect(submission.events); await submission.retirement.wait()
        XCTAssertEqual(cancelled.finishReason, .cancelled)
        XCTAssertEqual(f.engine.mtpMetricsSnapshot()?.draftedTokens, 0)
        try await shutdown(f)
    }

    private func scalarDenseCandidateProcess() throws {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_MIMO_RECTANGULAR_SCALAR_DENSE"] == "1",
              ProcessInfo.processInfo.environment["DARKBLOOM_MIMO_FUSED_DECODE_NORMS"] != "1" else {
            throw XCTSkip("Requires a fresh explicit scalar-dense candidate process, fused norms OFF")
        }
    }

    func testScalarDenseAdmittedNativeVerificationMatchesSerialTokensAndRetires() async throws {
        try normalProcess(); try scalarDenseCandidateProcess()
        for depth in 1...3 {
            var outputs: [[Int]] = []
            var hiddenByArm: [[Data]] = []
            for mode: CBv2MTPVerificationMode in [.serialTarget, .rectangular] {
                let f = try await fixture(mode: mode, depth: depth)
                XCTAssertNil(f.engine.nativeCompletionFault)
                if mode == .rectangular {
                    XCTAssertGreaterThan(f.engine.rectangularDenseScratchBytes, 0,
                        "No pass from a declined/unpriced candidate")
                    XCTAssertGreaterThanOrEqual(f.engine.resolvedFixedBytesPerRequest,
                                               f.engine.rectangularDenseScratchBytes)
                } else { XCTAssertEqual(f.engine.rectangularDenseScratchBytes, 0) }
                var completedHidden: [Data] = []
                f.engine.loopForTesting.onEngineQueueSync {
                    f.engine.loopForTesting.nativeRetirementBoundaryForTesting = { phase, step in
                        guard phase == "beforeMTPFinalization", let verify = step?.mtpRound?.verify else { return }
                        // lastHidden is an explicit asyncEval target of THIS
                        // tracked step. The hook follows its actual readback;
                        // no new model call, cache snapshot or injected output.
                        XCTAssertEqual(verify.lastHidden.shape[0], 1)
                        XCTAssertEqual(verify.lastHidden.shape[1], verify.k + 1)
                        completedHidden.append(verify.lastHidden.asData(access: .copy).data)
                    }
                }
                let submission = try f.engine.submitWithNativeRetirement(text(71, budget: 16))
                let result = await cbv2SchedCollect(submission.events)
                await submission.retirement.wait()
                XCTAssertEqual(result.finishReason, .length)
                XCTAssertEqual(result.tokens.count, 16)
                outputs.append(result.tokens)
                XCTAssertFalse(completedHidden.isEmpty)
                hiddenByArm.append(completedHidden)
                f.engine.loopForTesting.onEngineQueueSync {
                    f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
                }
                let metrics = try XCTUnwrap(f.engine.mtpMetricsSnapshot())
                XCTAssertGreaterThan(metrics.rounds, 0)
                if mode == .rectangular {
                    XCTAssertGreaterThan(metrics.rectangularVerificationRounds, 0)
                    XCTAssertGreaterThan(f.engine.rectangularDenseSubmittedCalls, 0)
                } else { XCTAssertEqual(f.engine.rectangularDenseSubmittedCalls, 0) }
                try await shutdown(f)
                XCTAssertEqual(f.engine.admissionForTesting.bytesReserved, 0)
                XCTAssertEqual(f.engine.loopForTesting.nativeShutdownState?.debugRetainedRootCount, 0)
            }
            XCTAssertEqual(outputs[1], outputs[0], "Exact greedy gate, never a tolerance")
            XCTAssertEqual(hiddenByArm[1], hiddenByArm[0], "Exact completed post-final-norm target rows")
        }
    }

    func testScalarDenseCancellationKeepsActualFixedChargeUntilNativeRetirement() async throws {
        try normalProcess(); try scalarDenseCandidateProcess()
        let f = try await fixture(depth: 3)
        XCTAssertGreaterThan(f.engine.rectangularDenseScratchBytes, 0)
        let gate = Gate(expectation(description: "actual dense-candidate verification completed"))
        let retirementGate = Gate(expectation(description: "actual queued retirement acknowledgement"))
        defer { gate.release(); retirementGate.release() }
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRetirementBoundaryForTesting = { phase, step in
                if phase == "beforeMTPFinalization", step?.mtpRound?.verify != nil { gate.hold() }
                if phase == "beforeAcknowledgement" { retirementGate.hold() }
            }
        }
        let submission = try f.engine.submitWithNativeRetirement(text(72, budget: 40))
        await fulfillment(of: [gate.entered], timeout: 10)
        XCTAssertGreaterThan(f.engine.rectangularDenseSubmittedCalls, 0)
        let tracking = try XCTUnwrap(f.engine.loopForTesting.nativeShutdownState)
        let charged = f.engine.admissionForTesting.bytesReserved
        XCTAssertGreaterThanOrEqual(charged, f.engine.rectangularDenseScratchBytes)
        XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        f.engine.cancel(.init(72))
        XCTAssertGreaterThanOrEqual(f.engine.admissionForTesting.bytesReserved,
                                   f.engine.rectangularDenseScratchBytes)
        gate.release()
        await fulfillment(of: [retirementGate.entered], timeout: 10)
        // Regression for the old prefix-OFF early releaseAll: row/step cleanup
        // is done but genuine retirement has not yet acknowledged this request.
        XCTAssertGreaterThanOrEqual(f.engine.admissionForTesting.bytesReserved,
                                   f.engine.rectangularDenseScratchBytes)
        XCTAssertGreaterThanOrEqual(f.engine.admissionForTesting.nonBackendBytesReserved,
                                   f.engine.rectangularDenseScratchBytes)
        retirementGate.release()
        let result = await cbv2SchedCollect(submission.events)
        await submission.retirement.wait()
        XCTAssertEqual(result.finishReason, .cancelled)
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
        }
        try await shutdown(f)
        XCTAssertEqual(f.engine.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(tracking.debugRetainedRootCount, 0)
    }

    func testScalarDenseRequiredFenceFailureKeepsActualRootsAndFixedCharge() async throws {
        try scalarDenseCandidateProcess()
        guard ProcessInfo.processInfo.environment["MIMO_V26_RECTANGULAR_FAULT_CASE"]
            == "testScalarDenseRequiredFenceFailureKeepsActualRootsAndFixedCharge" else {
            throw XCTSkip("This retained-fault selector requires its own fresh process")
        }
        let f = try await fixture(depth: 3)
        XCTAssertGreaterThan(f.engine.rectangularDenseScratchBytes, 0)
        defer {
            _ = Unmanaged.passRetained(f.engine); _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
        }
        let entered = expectation(description: "required fence failed after dense candidate encoded")
        var first = true
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRequiredAssistantFenceForTesting = { state in
                guard state.stagedInputCount > 0 else { return }
                if first { first = false; entered.fulfill() }
                throw Failure.injected
            }
        }
        let submission = try f.engine.submitWithNativeRetirement(text(73))
        _ = await cbv2SchedCollect(submission.events)
        await fulfillment(of: [entered], timeout: 10)
        XCTAssertGreaterThan(f.engine.rectangularDenseSubmittedCalls, 0)
        let outcome = await f.engine.shutdownReportingNativeCompletion()
        guard case .incomplete = outcome else { return XCTFail("A failed required fence minted retirement") }
        let tracking = try XCTUnwrap(f.engine.loopForTesting.nativeShutdownState)
        XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        XCTAssertGreaterThanOrEqual(f.engine.admissionForTesting.bytesReserved,
                                   f.engine.rectangularDenseScratchBytes)
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeRequiredAssistantFenceForTesting = nil
        }
        try withError { StreamOrDevice.default.stream.synchronize() }
        let late = await f.engine.shutdownReportingNativeCompletion()
        XCTAssertEqual(late, outcome)
        XCTAssertGreaterThanOrEqual(f.engine.admissionForTesting.bytesReserved,
                                   f.engine.rectangularDenseScratchBytes)
        // Never wait for a false success receipt or refund restart-only roots.
    }

    // MARK: Admitted complete-state witnesses (no native work in observers)

    private enum StateWitnessError: Error {
        case geometry, bound, unavailable, layout, owner, packet, observation
    }
    private struct TensorWitness {
        let shape: [Int]
        let strides: [Int]
        let dtype: DType
        let allocatedBytes, dataOffset, dataElements: Int
        /// Tensor elements in the root's declared physical-axis order. This
        /// excludes allocator padding/stride holes, never other allocations.
        let bytes: Data
    }
    private struct NamedBytes {
        let name: String
        let shape: [Int]
        let dtype: DType
        let bytes: Data
    }
    private struct CarryWitness: Equatable {
        let token, tokensCount, kvOffset: Int
        let hiddenShape: [Int]
        init(_ value: CBv2MTPCarryObservationForTesting) {
            token = value.token; tokensCount = value.tokensCount
            kvOffset = value.kvOffset; hiddenShape = value.hiddenShape
        }
    }
    private struct RoundWitness {
        let depth, base, computedEnd, accepted: Int
        let drafts, targets, scalars: [Int]
        let targetMetadata: [[Int]]
        let headMetadata, draftHeadMetadata: [[String]]
        let raw: [TensorWitness]
        let logical: [NamedBytes]
        let inputCarry: CarryWitness
        var outputCarry: CarryWitness?
    }
    private final class WeakRequestState {
        weak var value: MiMoV26MTPState?
    }

    /// Engine-queue confined until hooks are removed and the actual request
    /// retirement has completed. Stores host copies/scalars only, plus one WEAK
    /// request witness. No model/cache/MLXArray is retained across an await.
    private final class StateWitnessRecorder {
        let byteLimit, roundLimit: Int
        private(set) var usedBytes = 0
        private(set) var rounds: [RoundWitness] = []
        private(set) var failure: StateWitnessError?
        private var lastCarry: CarryWitness?
        private var firstIdentity: ObjectIdentifier?
        let firstState = WeakRequestState()

        init(byteLimit: Int, roundLimit: Int) {
            self.byteLimit = byteLimit; self.roundLimit = roundLimit
        }
        static func add(_ a: Int, _ b: Int) throws -> Int {
            let x = a.addingReportingOverflow(b)
            guard a >= 0, b >= 0, !x.overflow else { throw StateWitnessError.bound }
            return x.partialValue
        }
        static func multiply(_ a: Int, _ b: Int) throws -> Int {
            let x = a.multipliedReportingOverflow(by: b)
            guard a >= 0, b >= 0, !x.overflow else { throw StateWitnessError.bound }
            return x.partialValue
        }
        private func charge(_ bytes: Int) throws {
            let next = try Self.add(usedBytes, bytes)
            guard next <= byteLimit else { throw StateWitnessError.bound }
            usedBytes = next
        }

        /// No eval/asData/view/contiguous/astype. Availability is proved before
        /// taking the C pointer. Nonnegative strides and every addressed byte are
        /// checked against the actual evaluated allocation, not guessed nbytes.
        private func read(_ array: MLXArray) throws -> TensorWitness {
            let shape = array.shape, item = array.dtype.size
            guard !shape.isEmpty, shape.count <= 4, shape.allSatisfy({ $0 > 0 }),
                  item > 0, item <= 8,
                  let info = try array.evaluatedBufferInfo(),
                  let stridePointer = mlx_array_strides(array.ctx) else {
                throw StateWitnessError.unavailable
            }
            let strides = (0..<shape.count).compactMap { Int(exactly: stridePointer[$0]) }
            guard strides.count == shape.count, strides.allSatisfy({ $0 >= 0 }) else {
                throw StateWitnessError.layout
            }
            var elements = 1, lastElement = 0
            for (length, stride) in zip(shape, strides) {
                elements = try Self.multiply(elements, length)
                lastElement = try Self.add(lastElement, Self.multiply(length - 1, stride))
            }
            let bytes = try Self.multiply(elements, item)
            let extent = try Self.multiply(Self.add(lastElement, 1), item)
            guard info.dataOffset >= 0, info.dataElements > 0,
                  try Self.add(info.dataOffset, extent) <= info.allocatedBytes,
                  bytes == array.nbytes else { throw StateWitnessError.layout }
            try charge(bytes)
            guard let source = mlx_array_data_uint8(array.ctx) else { throw StateWitnessError.unavailable }
            let copied = withExtendedLifetime(array) {
                var result = Data(count: bytes)
                result.withUnsafeMutableBytes { (destination: UnsafeMutableRawBufferPointer) in
                    for linear in 0..<elements {
                        var rest = linear, offset = 0
                        for axis in shape.indices.reversed() {
                            let coordinate = rest % shape[axis]
                            rest /= shape[axis]
                            offset += coordinate * strides[axis] // bounded by lastElement above
                        }
                        destination.baseAddress!.advanced(by: linear * item).copyMemory(
                            from: source.advanced(by: offset * item), byteCount: item)
                    }
                }
                return result
            }
            return .init(shape: shape, strides: strides, dtype: array.dtype,
                allocatedBytes: info.allocatedBytes, dataOffset: info.dataOffset,
                dataElements: info.dataElements, bytes: copied)
        }

        /// Host-only rank-four chronological selection from the copied raw
        /// root. No new MLX slice/concat or reading allocator padding.
        private func select(_ raw: TensorWitness, slots: [Int], name: String) throws -> NamedBytes {
            guard raw.shape.count == 4, raw.shape[0] == 1,
                  slots.allSatisfy({ $0 >= 0 && $0 < raw.shape[2] }) else {
                throw StateWitnessError.layout
            }
            let heads = raw.shape[1], width = raw.shape[3]
            let rowBytes = try Self.multiply(width, raw.dtype.size)
            let count = try Self.multiply(Self.multiply(heads, slots.count), rowBytes)
            try charge(count)
            var result = Data(count: count)
            raw.bytes.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
                result.withUnsafeMutableBytes { (destination: UnsafeMutableRawBufferPointer) in
                    for head in 0..<heads {
                        for (index, slot) in slots.enumerated() {
                            let from = (head * raw.shape[2] + slot) * rowBytes
                            let to = (head * slots.count + index) * rowBytes
                            destination.baseAddress!.advanced(by: to).copyMemory(
                                from: source.baseAddress!.advanced(by: from), byteCount: rowBytes)
                        }
                    }
                }
            }
            return .init(name: name, shape: [1, heads, slots.count, width], dtype: raw.dtype, bytes: result)
        }
        private func integers(_ raw: TensorWitness) throws -> [Int] {
            guard raw.dtype == .int32, raw.bytes.count.isMultiple(of: 4) else {
                throw StateWitnessError.packet
            }
            return raw.bytes.withUnsafeBytes { (data: UnsafeRawBufferPointer) in
                (0..<(data.count / 4)).map { Int(data.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) }
            }
        }
        private func heads(_ cache: MiMoV26MTPRequestCache, label: String,
                           raw: inout [TensorWitness], logical: inout [NamedBytes]) throws -> [[String]] {
            let metadata = cache.retainedHistoryMetadataForTesting
            let arrays = cache.innerState()
            guard metadata.count == 3, arrays.count == 6,
                  cache.nextTokenPositions.count == 3, cache.consumedTokenCounts.count == 3 else {
                throw StateWitnessError.geometry
            }
            for depth in 0..<3 {
                let numbers = metadata[depth].compactMap(Int.init)
                guard numbers.count == 6, numbers[0] == depth + 1, numbers[1] == 0,
                      numbers[2] > 0, numbers[3] == 256, numbers[4] > 0,
                      numbers[4] == cache.consumedTokenCounts[depth],
                      try Self.add(numbers[0], numbers[4]) == cache.nextTokenPositions[depth] else {
                    throw StateWitnessError.layout
                }
                let window = numbers[2], offset = numbers[4], index = numbers[5]
                let live = min(offset, window)
                guard index > 0, index <= window,
                      offset >= window || index == offset else { throw StateWitnessError.layout }
                let start = offset < window || index == window ? 0 : index
                let slots = (0..<live).map { (start + $0) % window }
                for component in 0..<2 {
                    let bytes = try read(arrays[2 * depth + component])
                    guard bytes.shape.count == 4, bytes.shape[2] >= live,
                          bytes.shape[2] <= window else { throw StateWitnessError.layout }
                    raw.append(bytes)
                    logical.append(try select(bytes, slots: slots,
                        name: "\(label).\(depth).\(component == 0 ? "K" : "V")"))
                }
            }
            return metadata
        }

        /// Called under the finalizer's metadata commit: scalars ONLY.
        func carried(_ value: CBv2MTPCarryObservationForTesting) {
            let carry = CarryWitness(value)
            if let last = rounds.indices.last, rounds[last].outputCarry == nil,
               carry.kvOffset > rounds[last].base, carry.kvOffset <= rounds[last].computedEnd {
                rounds[last].outputCarry = carry
            }
            lastCarry = carry
        }

        func observe(_ step: CBv2InFlightStep, assistant: MiMoV26MTPAssistant) {
            guard failure == nil else { return }
            do { try record(step, assistant: assistant) }
            catch let error as StateWitnessError { failure = error }
            catch { failure = .observation }
        }
        private func record(_ step: CBv2InFlightStep, assistant: MiMoV26MTPAssistant) throws {
            guard let verify = step.mtpRound?.verify, verify.rows.count == 1,
                  (1...3).contains(verify.k), rounds.count < roundLimit,
                  verify.shortlistIDs == nil, let inputCarry = lastCarry,
                  let state = verify.rows[0].assistantState as? MiMoV26MTPState,
                  let committed = state.cache, let draft = state.round,
                  let range = step.computedRanges[verify.rows[0].id],
                  range.count == verify.k + 1, inputCarry.kvOffset == range.lowerBound,
                  state.owner === assistant, state.generation == assistant.generation,
                  !state.isReleased, !state.hasUnmeasuredResidency,
                  state.stagedInputCount == verify.k, draft.inputs.count == verify.k,
                  state.observedCount == range.lowerBound, draft.baseCount == state.observedCount,
                  state.headInputCounts == (0..<3).map({ state.observedCount - $0 - 1 }),
                  draft.cache.consumedTokenCounts == (0..<3).map({
                      $0 < verify.k ? state.observedCount : state.observedCount - $0 - 1
                  }),
                  state.pendingTokens == nil, state.pendingHidden == nil, state.pendingLastToken == nil,
                  state.prefixRestoredBoundary == nil else { throw StateWitnessError.owner }
            if let firstIdentity {
                guard firstIdentity == ObjectIdentifier(state), firstState.value === state else {
                    throw StateWitnessError.owner
                }
            } else {
                firstIdentity = ObjectIdentifier(state); firstState.value = state
            }
            try charge(16 << 10) // bounded metadata, labels, slot tables and temporary host packet
            let packet = try integers(read(verify.acceptancePacket))
            guard packet.count == 2 * verify.k + 1 else { throw StateWitnessError.packet }
            let drafts = Array(packet.prefix(verify.k)), targets = Array(packet.suffix(verify.k + 1))
            var accepted = 0
            while accepted < verify.k && drafts[accepted] == targets[accepted] { accepted += 1 }
            var raw: [TensorWitness] = [], logical: [NamedBytes] = []
            var targetMetadata: [[Int]] = []
            for (index, row) in verify.rows[0].storageRows.enumerated() {
                guard row.absoluteOffset == range.upperBound,
                      let provider = row as? any CBv2InnerStateProviding else { throw StateWitnessError.layout }
                let roots = provider.cbv2InnerState()
                guard roots.count == 2 else { throw StateWitnessError.layout }
                let base = range.lowerBound
                let slots: [Int]
                if let ring = row as? CBv2WindowedSequenceKV {
                    let oldest = row.absoluteOffset - row.retainedCount
                    guard oldest == max(0, base - ring.window) else { throw StateWitnessError.layout }
                    slots = (oldest..<base).map { $0 % ring.window }
                } else if row is CBv2FullSequenceKV {
                    slots = Array(0..<base)
                } else { throw StateWitnessError.layout }
                targetMetadata.append([row.absoluteOffset, row.retainedCount,
                    row.absoluteOffset - row.retainedCount,
                    (row as? CBv2WindowedSequenceKV)?.window ?? 0])
                for component in 0..<2 {
                    let value = try read(roots[component]); raw.append(value)
                    let name = "target.\(index).\(component == 0 ? "K" : "V")"
                    logical.append(try select(value, slots: slots, name: name + ".committed"))
                    // Full-row storage also exposes the genuinely evaluated
                    // current verify suffix. SWA's new suffix is private staged
                    // storage, so its committed bytes are checked NEXT round,
                    // before the next speculative commit can overwrite them.
                    if row is CBv2FullSequenceKV {
                        logical.append(try select(value, slots: Array(0..<range.upperBound), name: name + ".verify"))
                    }
                }
            }
            let headMetadata = try heads(committed, label: "heads", raw: &raw, logical: &logical)
            let draftMetadata = try heads(draft.cache, label: "draftHeads", raw: &raw, logical: &logical)
            let features = [state.tail, Optional(verify.lastHidden)] + draft.inputs.map(Optional.some)
            for (index, array) in features.enumerated() {
                guard let array else { throw StateWitnessError.layout }
                let value = try read(array); raw.append(value)
                logical.append(.init(name: "features.\(index)", shape: value.shape,
                    dtype: value.dtype, bytes: value.bytes))
            }
            rounds.append(.init(depth: verify.k, base: range.lowerBound, computedEnd: range.upperBound,
                accepted: accepted, drafts: drafts, targets: targets,
                scalars: [state.observedCount, state.committedInputCount, state.stagedInputCount,
                    state.retainedFeatureRows] + state.headInputCounts + committed.nextTokenPositions
                    + state.headProposalCounts,
                targetMetadata: targetMetadata,
                headMetadata: headMetadata, draftHeadMetadata: draftMetadata, raw: raw,
                logical: logical, inputCarry: inputCarry, outputCarry: nil))
        }
        func discardHostCopies() { rounds.removeAll(); lastCarry = nil }
    }

    private final class ObservedRequest {
        let recorder: StateWitnessRecorder
        let reservation: CBv2CheckpointReservation
        let tokens: [Int]
        let finish: CBv2FinishReason?
        init(recorder: StateWitnessRecorder, reservation: CBv2CheckpointReservation,
             tokens: [Int], finish: CBv2FinishReason?) {
            self.recorder = recorder; self.reservation = reservation
            self.tokens = tokens; self.finish = finish
        }
        // Constructed only AFTER actual request retirement. Even an assertion
        // unwind drops copied bytes before returning their host commitment.
        deinit { recorder.discardHostCopies(); reservation.release() }
    }

    /// Additive actual Admission charge, with no model/MTP reserve discount.
    /// This test only accepts a bounded small fixture; exceeding the envelope
    /// is an explicit input failure, never truncation of a witness.
    private func witnessBound(configuration c: MiMoV26Configuration, request: CBv2Request) throws -> Int {
        let maximum = try StateWitnessRecorder.add(request.promptTokens.count, request.maxTokens)
        guard request.maxTokens > 0, request.maxTokens <= 32, maximum <= c.maxPositionEmbeddings,
              maximum <= 512, (1...4).contains(c.numHiddenLayers),
              (1...64).contains(c.hiddenSize), (1...128).contains(c.slidingWindow) else {
            throw StateWitnessError.geometry
        }
        var perRound = 16 << 10
        for layer in 0..<c.numHiddenLayers {
            let g = try c.attentionGeometry(at: layer)
            let tokens = g.slidingWindow ?? maximum
            let raw = try StateWitnessRecorder.multiply(
                StateWitnessRecorder.multiply(tokens, g.keyValueHeads),
                StateWitnessRecorder.multiply(StateWitnessRecorder.add(g.headDim, g.valueHeadDim), 4))
            perRound = try StateWitnessRecorder.add(perRound, StateWitnessRecorder.multiply(raw, 4))
        }
        let g = c.slidingAttention
        let heads = try StateWitnessRecorder.multiply(
            StateWitnessRecorder.multiply(c.slidingWindow, g.keyValueHeads),
            StateWitnessRecorder.multiply(StateWitnessRecorder.add(g.headDim, g.valueHeadDim), 4))
        // Three heads, committed + speculative, raw + chronological; price
        // native two-byte arrays at four bytes plus bounded temporary copies.
        perRound = try StateWitnessRecorder.add(perRound, StateWitnessRecorder.multiply(heads, 24))
        perRound = try StateWitnessRecorder.add(perRound, StateWitnessRecorder.multiply(c.hiddenSize, 128))
        let total = try StateWitnessRecorder.add(64 << 10,
            StateWitnessRecorder.multiply(perRound, request.maxTokens + 1))
        guard total <= 8 << 20 else { throw StateWitnessError.bound }
        return total
    }

    private func observeRequest(_ f: Fixture, request: CBv2Request, cancelAtFirstVerify: Bool = false)
        async throws -> ObservedRequest {
        let configuration = try await f.container.perform { context in
            try XCTUnwrap(context.model as? MiMoV26LoadedModel).nativeConfiguration
        }
        let bytes = try witnessBound(configuration: configuration, request: request)
        let reservation = try f.engine.admissionForTesting.reserveTransient(bytes: bytes)
        let recorder = StateWitnessRecorder(byteLimit: bytes, roundLimit: request.maxTokens + 1)
        let gate = cancelAtFirstVerify ? Gate(expectation(description: "owned completed-state cancellation boundary")) : nil
        defer { gate?.release() }
        f.engine.loopForTesting.onEngineQueueSync { [weak engine = f.engine] in
            guard let loop = engine?.loopForTesting else { return }
            loop.mtp?.carryStoredObserverForTesting = { recorder.carried($0) }
            loop.nativeRetirementBoundaryForTesting = { [weak engine] phase, step in
                guard phase == "beforeMTPFinalization", let step, step.mtpRound?.verify != nil,
                      let assistant = engine?.loopForTesting.mtp?.drafter as? MiMoV26MTPAssistant else { return }
                recorder.observe(step, assistant: assistant)
                gate?.hold()
            }
        }
        do {
            let beforeCalls = f.engine.rectangularDenseSubmittedCalls
            let submission = try f.engine.submitWithNativeRetirement(request)
            if let gate {
                await fulfillment(of: [gate.entered], timeout: 10)
                f.engine.cancel(request.id)
                XCTAssertGreaterThan(f.engine.admissionForTesting.transientBytesReserved, 0)
                gate.release()
            }
            let result = await cbv2SchedCollect(submission.events)
            guard f.engine.nativeCompletionFault == nil else { throw Failure.noProof }
            await submission.retirement.wait()
            f.engine.loopForTesting.onEngineQueueSync {
                f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
                f.engine.loopForTesting.mtp?.carryStoredObserverForTesting = nil
            }
            XCTAssertNil(recorder.failure, "Incomplete/invalid native state observation is not a pass")
            XCTAssertGreaterThanOrEqual(recorder.rounds.count, cancelAtFirstVerify ? 1 : 2)
            if f.contract.mtpVerificationMode == .rectangular {
                XCTAssertGreaterThan(f.engine.rectangularDenseScratchBytes, 0)
                XCTAssertGreaterThan(f.engine.rectangularDenseSubmittedCalls, beforeCalls)
            } else {
                XCTAssertEqual(f.engine.rectangularDenseScratchBytes, 0)
                XCTAssertEqual(f.engine.rectangularDenseSubmittedCalls, 0)
            }
            XCTAssertNil(recorder.firstState.value, "real native retirement must release the request state")
            XCTAssertEqual(result.finishReason, cancelAtFirstVerify ? .cancelled : .length)
            return .init(recorder: recorder, reservation: reservation,
                tokens: result.tokens, finish: result.finishReason)
        } catch {
            gate?.release()
            f.engine.loopForTesting.onEngineQueueSync {
                f.engine.loopForTesting.nativeRetirementBoundaryForTesting = nil
                f.engine.loopForTesting.mtp?.carryStoredObserverForTesting = nil
            }
            // A cold refusal can still obtain an authentic whole-engine drain.
            // Only that success allows cleanup; keep the original test error.
            do {
                try await shutdown(f)
                recorder.discardHostCopies(); reservation.release()
            } catch {
                // Unknown/failed completion is restart-only. Never let ARC
                // return its reservation while the actual owner is retained.
                _ = Unmanaged.passRetained(reservation); _ = Unmanaged.passRetained(recorder)
                _ = Unmanaged.passRetained(f.engine); _ = Unmanaged.passRetained(f.container)
                _ = Unmanaged.passRetained(f.construction)
            }
            throw error
        }
    }

    private func maxNativeULP(_ a: NamedBytes, _ b: NamedBytes) -> UInt64? {
        let size: Int
        switch a.dtype {
        case .bfloat16, .float16: size = 2
        case .float32: size = 4
        default: return nil
        }
        guard a.dtype == b.dtype, a.bytes.count == b.bytes.count,
              a.bytes.count.isMultiple(of: size) else { return nil }
        let mask: UInt64 = size == 2 ? 0xffff : 0xffff_ffff
        let sign: UInt64 = size == 2 ? 0x8000 : 0x8000_0000
        func ordered(_ value: UInt64) -> UInt64 {
            value & sign == 0 ? value | sign : (~value) & mask
        }
        return a.bytes.withUnsafeBytes { (left: UnsafeRawBufferPointer) in
            b.bytes.withUnsafeBytes { (right: UnsafeRawBufferPointer) in
                var maximum: UInt64 = 0
                for offset in stride(from: 0, to: left.count, by: size) {
                    let x = ordered(size == 2
                        ? UInt64(left.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
                        : UInt64(left.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
                    let y = ordered(size == 2
                        ? UInt64(right.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
                        : UInt64(right.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
                    maximum = max(maximum, x >= y ? x - y : y - x)
                }
                return maximum
            }
        }
    }
    private func compareObserved(_ actual: ObservedRequest, _ reference: ObservedRequest) {
        XCTAssertEqual(actual.tokens, reference.tokens)
        XCTAssertEqual(actual.finish, reference.finish)
        let a = actual.recorder.rounds, b = reference.recorder.rounds
        XCTAssertEqual(a.count, b.count, "same genuine committed histories must produce corresponding rounds")
        for (index, pair) in zip(a, b).enumerated() {
            let x = pair.0, y = pair.1
            XCTAssertEqual([x.depth, x.base, x.computedEnd, x.accepted], [y.depth, y.base, y.computedEnd, y.accepted])
            XCTAssertEqual(x.drafts, y.drafts); XCTAssertEqual(x.targets, y.targets)
            XCTAssertEqual(x.scalars, y.scalars)
            XCTAssertEqual(x.targetMetadata, y.targetMetadata)
            XCTAssertEqual(x.headMetadata, y.headMetadata)
            XCTAssertEqual(x.draftHeadMetadata, y.draftHeadMetadata)
            XCTAssertEqual(x.inputCarry, y.inputCarry); XCTAssertEqual(x.outputCarry, y.outputCarry)
            XCTAssertEqual(x.logical.count, y.logical.count)
            for (left, right) in zip(x.logical, y.logical) {
                XCTAssertEqual(left.name, right.name); XCTAssertEqual(left.shape, right.shape)
                XCTAssertEqual(left.dtype, right.dtype)
                // Avoid dumping tensor bytes to test logs. Exact native bytes,
                // never tolerance/argmax alone, are the acceptance condition.
                let ulp = maxNativeULP(left, right).map(String.init) ?? "not-floating"
                XCTAssertTrue(left.bytes == right.bytes,
                    "round \(index) \(left.name): native bytes differ; maxNativeULP=\(ulp)")
            }
            let rawDifferences = zip(x.raw, y.raw).filter {
                $0.shape != $1.shape || $0.strides != $1.strides || $0.bytes != $1.bytes
            }.count
            XCTAssertEqual(x.raw.count, y.raw.count)
            for (root, pair) in zip(x.raw, y.raw).enumerated()
            where pair.0.shape != pair.1.shape || pair.0.strides != pair.1.strides || pair.0.bytes != pair.1.bytes {
                let left = pair.0, right = pair.1
                let leftHash = SHA256.hash(data: left.bytes).map { String(format: "%02x", $0) }.joined()
                let rightHash = SHA256.hash(data: right.bytes).map { String(format: "%02x", $0) }.joined()
                print("MIMO_SCALAR_RAW round=\(index) root=\(root) actualShape=\(left.shape) referenceShape=\(right.shape) actualStrides=\(left.strides) referenceStrides=\(right.strides) actualExtent=\(left.dataElements) referenceExtent=\(right.dataElements) actualOffset=\(left.dataOffset) referenceOffset=\(right.dataOffset) actualAllocation=\(left.allocatedBytes) referenceAllocation=\(right.allocatedBytes) actualSHA256=\(leftHash) referenceSHA256=\(rightHash)")
            }
            print("MIMO_SCALAR_STATE round=\(index) depth=\(x.depth) base=\(x.base) accepted=\(x.accepted) rawLayoutOrExtentDifferences=\(rawDifferences)")
        }
    }
    private func releaseObserved(_ value: ObservedRequest, engine: EngineV2) {
        // Hooks are detached and the genuine request retirement was awaited.
        // All bounded host copies die BEFORE their actual Admission refund.
        value.recorder.discardHostCopies()
        value.reservation.release()
        XCTAssertEqual(engine.admissionForTesting.transientBytesReserved, 0)
    }
    private func continuedAcceptedPrefixes(_ value: ObservedRequest, requestedDepth: Int) -> Set<Int> {
        let records = value.recorder.rounds
        return Set(zip(records, records.dropFirst()).compactMap { previous, next in
            guard previous.depth == requestedDepth,
                  let carry = previous.outputCarry, carry == next.inputCarry,
                  carry.kvOffset == next.base,
                  carry.kvOffset - previous.base == previous.accepted + 1 else { return nil }
            return previous.accepted
        })
    }

    func testScalarDenseAdmittedCompleteStateAndNaturalRollbackMatchSerial() async throws {
        try normalProcess(); try scalarDenseCandidateProcess()
        for depth in 1...3 {
            var ownedFixtures: [Fixture] = []
            do {
                let serial = try await fixture(mode: .serialTarget, depth: depth)
                ownedFixtures.append(serial)
                let candidate = try await fixture(mode: .rectangular, depth: depth)
                ownedFixtures.append(candidate)
                let config = try await serial.container.perform { context in
                    try XCTUnwrap(context.model as? MiMoV26LoadedModel).nativeConfiguration
                }
                let window = config.slidingWindow
                guard (1...128).contains(window) else { throw StateWitnessError.geometry }
                let lengths = Set([4, max(4, window - 1), max(4, window),
                    max(4, window + 1), max(4, window + 3), max(4, 2 * window + 1)]).sorted()
                var coverage = Set<Int>()
                for (index, length) in lengths.enumerated() {
                    var request = text(UInt64(9100 + index), count: length, budget: 24)
                    request.promptTokens = (0..<length).map { 20 + ($0 * (2 * index + 1) + index) % 11 }
                    let reference = try await observeRequest(serial, request: request)
                    let actual = try await observeRequest(candidate, request: request)
                    XCTAssertEqual(reference.recorder.rounds.first?.depth, depth)
                    XCTAssertEqual(actual.recorder.rounds.first?.depth, depth)
                    compareObserved(actual, reference)
                    coverage.formUnion(continuedAcceptedPrefixes(actual, requestedDepth: depth))
                    releaseObserved(actual, engine: candidate.engine)
                    releaseObserved(reference, engine: serial.engine)
                }
                print("MIMO_SCALAR_STATE coverage depth=\(depth) actualWindow=\(window) acceptedWithRealContinuation=\(coverage.sorted()) selectedWindow128=\(window == 128)")
                XCTAssertEqual(coverage, Set(0...depth),
                    "INCOMPLETE genuine acceptance/rejection coverage; do not script proposals or relabel this as a state pass")
                try await shutdown(candidate); try await shutdown(serial)
                XCTAssertEqual(candidate.engine.admissionForTesting.bytesReserved, 0)
                XCTAssertEqual(serial.engine.admissionForTesting.bytesReserved, 0)
            } catch {
                // Even a later fixture/input/admission refusal joins each
                // already-created engine; preserve the original failure.
                for f in ownedFixtures { try? await shutdown(f) }
                throw error
            }
        }
    }

    func testScalarDenseAdmittedCancelAndReusedIDMatchFreshSerialState() async throws {
        try normalProcess(); try scalarDenseCandidateProcess()
        var ownedFixtures: [Fixture] = []
        do {
            let serial = try await fixture(mode: .serialTarget, depth: 3)
            ownedFixtures.append(serial)
            let candidate = try await fixture(mode: .rectangular, depth: 3)
            ownedFixtures.append(candidate)
            let fresh = try await fixture(mode: .serialTarget, depth: 3)
            ownedFixtures.append(fresh)
            let cancelled = text(9201, count: 11, budget: 24)
            let serialCancelled = try await observeRequest(serial, request: cancelled, cancelAtFirstVerify: true)
            let candidateCancelled = try await observeRequest(candidate, request: cancelled, cancelAtFirstVerify: true)
            XCTAssertEqual(serialCancelled.recorder.rounds.first?.depth, 3)
            XCTAssertEqual(candidateCancelled.recorder.rounds.first?.depth, 3)
            compareObserved(candidateCancelled, serialCancelled)
            XCTAssertNil(serialCancelled.recorder.firstState.value)
            XCTAssertNil(candidateCancelled.recorder.firstState.value)
            releaseObserved(candidateCancelled, engine: candidate.engine)
            releaseObserved(serialCancelled, engine: serial.engine)
            for f in [serial, candidate] {
                f.engine.loopForTesting.onEngineQueueSync {
                    XCTAssertNil(f.engine.loopForTesting.mtp?.assistantStateCountsForTesting(cancelled.id))
                    XCTAssertEqual(f.engine.loopForTesting.backend.bytesReserved, 0)
                    XCTAssertEqual(f.engine.loopForTesting.nativeShutdownState?.debugRetainedRootCount, 0)
                }
            }
            var replacement = text(9201, count: 13, budget: 24)
            replacement.promptTokens = (0..<13).map { 30 - ($0 * 3) % 11 }
            let reference = try await observeRequest(fresh, request: replacement)
            let serialReused = try await observeRequest(serial, request: replacement)
            let candidateReused = try await observeRequest(candidate, request: replacement)
            compareObserved(serialReused, reference)
            compareObserved(candidateReused, reference)
            releaseObserved(candidateReused, engine: candidate.engine)
            releaseObserved(serialReused, engine: serial.engine)
            releaseObserved(reference, engine: fresh.engine)
            for f in [candidate, serial, fresh] {
                try await shutdown(f)
                XCTAssertEqual(f.engine.admissionForTesting.bytesReserved, 0)
                XCTAssertEqual(f.engine.loopForTesting.nativeShutdownState?.debugRetainedRootCount, 0)
            }
        } catch {
            for f in ownedFixtures { try? await shutdown(f) }
            throw error
        }
    }

}
