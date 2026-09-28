import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
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
}
