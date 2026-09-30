import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import Tokenizers
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Real strict target + authenticated selected codec + actual PCM/engine work.
/// Requires an explicitly supplied small audio-compatible target fixture; this
/// class never weakens the selected sidecar hash/geometry or invents a loaded
/// owner. Reservation counters are SDK lifetime witnesses, NOT host ledger or
/// measured-peak qualification. Each fault selector needs its own process.
final class MiMoV26ManagedDecodedAudioTests: XCTestCase {
    private enum Failure: Error { case inputRequired, noProof, injected }
    private enum Profile: Sendable, Equatable { case audio, visual, text }
    private final class RootPermit: MiMoV26SerialLoadReservation, Sendable {
        let request: MiMoV26SerialLoadRequest
        let reservedLoadBytes: UInt64
        init(_ request: MiMoV26SerialLoadRequest) {
            self.request = request
            reservedLoadBytes = request.requiredLoadBytes
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private final class AudioPermit: MiMoV26AudioSidecarLoadReservation, @unchecked Sendable {
        let request: MiMoV26AudioSidecarLoadRequest
        let reservedLoadBytes: UInt64
        private let lock = NSLock()
        private var invalid = false
        init(_ request: MiMoV26AudioSidecarLoadRequest) {
            self.request = request
            reservedLoadBytes = request.requiredLoadBytes
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
                guard digest == nil, bytes > 0 else {
                    throw MiMoV26MultimodalError.reservationRejected
                }
                digest = plan.preparationSHA256
                charge = bytes
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
            lock.withLock {
                if !failures.contains(where: { $0 === work }) { failures.append(work) }
            }
        }
        func retire() throws {
            try lock.withLock {
                guard failures.isEmpty, releaseCount == 0 else { throw Failure.noProof }
                charge = 0
                releaseCount = 1
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
        pixels: .init(
            maximumInputElements: 100000, maximumOutputElements: 100000,
            maximumWorkingBytes: 1 << 20),
        vision: .init(maximumPatches: 256, maximumAttentionScoreElements: 131072),
        audio: .init(
            maximumClips: 2, maximumChannels: 1, maximumSampleRate: 24000,
            maximumInputSamples: 48000, maximumResampledSamples: 48000,
            maximumResampleCoefficients: 100000, maximumMelFrames: 256, maximumSegments: 4,
            maximumPaddedMelFrames: 256, maximumWorkingElements: 64_000_000,
            frontendFrameBlockSize: 8, rvqTileFrames: 8),
        audioPatch: .init(
            maximumClips: 2, maximumFrames: 256, maximumPatches: 64,
            maximumWorkingElements: 16_000_000))

    private func fixture(
        profile: Profile = .audio, installAudio: Bool = true,
        enableMTP: Bool = false
    ) async throws -> Fixture {
        let env = ProcessInfo.processInfo.environment
        // Skip, not fail, on a machine without the lane, such as the hosted CI runner.
        try XCTSkipUnless(
            env["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1"
                && env["MIMO_V26_MANAGED_AUDIO_NATIVE_TESTS"] == "1",
            "Requires the exclusive native GPU lane. Set MIMO_V26_SERIAL_NATIVE_TESTS=1 and MIMO_V26_MANAGED_AUDIO_NATIVE_TESTS=1 to run it."
        )
        guard let path = env["MIMO_V26_MANAGED_AUDIO_FIXTURE_ROOT"] else {
            throw Failure.inputRequired
        }
        let root = URL(fileURLWithPath: path)
        // Deliberately excludes a full target: the real selected SIDE-CAR is
        // separate and authenticated by its existing loader below.
        let provenanceFile = try FileHandle(
            forReadingFrom: root.appendingPathComponent("provenance.json"))
        defer { try? provenanceFile.close() }
        let provenanceData = try XCTUnwrap(provenanceFile.read(upToCount: 65537))
        guard provenanceData.count <= 65536 else { throw Failure.inputRequired }
        let p = try XCTUnwrap(
            JSONSerialization.jsonObject(with: provenanceData) as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(
            artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]),
            sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 64 << 20, maximumTotalFileBytes: 128 << 20))
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let config = plan.bundlePlan.configuration
        guard config.vocabularySize > 151674, config.hiddenSize <= 64,
            config.numHiddenLayers <= 4
        else { throw Failure.inputRequired }
        let prepared = try await MiMoV26ModelFactory.prepare(
            request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let construction = NativeConstructionWork()
        defer {
            if construction.snapshot.isRetainedFault { _ = Unmanaged.passRetained(construction) }
        }
        let container = try await MiMoV26ModelFactory.loadContainer(
            session: session,
            reservation: RootPermit(session.request), prepared: prepared, retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        let sidecar: SendableBox<MiMoV26AudioSidecarLoadSession>?
        let permit: AudioPermit?
        if installAudio {
            let value = try MiMoV26AudioSidecarLoadSession(
                root: root, mainConfiguration: config,
                mainConfigurationSHA256: plan.bundlePlan.configSHA256)
            sidecar = SendableBox(value)
            permit = AudioPermit(value.request)
        } else {
            sidecar = nil
            permit = nil
        }
        let built = try await MiMoV26ModelFactory.withNativeConstruction(
            container: container, retaining: construction
        ) { model, scope in
            if let sidecar, let permit {
                _ = try model.installAudioSidecar(
                    session: sidecar.consume(), reservation: permit,
                    retaining: scope, isCancelled: { false })
            }
            let binding = try model.makeCBv2Binding(enableMTP: enableMTP)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let resources: MiMoV26ManagedMediaExecutionResources
            switch profile {
            case .audio:
                resources = try model.makeManagedAudioExecutionResources(
                    binding: binding,
                    bytesCapacity: 32 << 20, limits: Self.limits, retaining: scope)
            case .visual:
                resources = try model.makeManagedMediaExecutionResources(
                    binding: binding,
                    bytesCapacity: 32 << 20, limits: Self.limits, retaining: scope)
            case .text:
                let text = try binding.adapter.makeNativeExecutionResources(
                    bytesCapacity: 32 << 20, retaining: scope)
                resources = .init(
                    backend: text.backend, cacheProvider: text.cacheProvider,
                    contract: text.contract)
            }
            let engine = EngineV2(
                model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 4,
                    prefillChunkSize: 3, maxConcurrentPartialPrefills: 1, enablePrefixCache: false),
                loopConfig: .init(stepTimeout: 60, watchdogInterval: 0.01, shutdownTimeout: 10),
                mtpDrafter: binding.assistant,
                mtpConfig: .init(
                    enabled: enableMTP, maxDraftTokens: 2,
                    maxSpeculativeBatch: 1, fixedDraftTokens: 2, verificationMode: .serialTarget),
                nativeCompletionTracking: true, nativeExecutionContract: resources.contract)
            try scope.retainOwner(engine)
            return Built(engine: engine, contract: resources.contract)
        }
        guard case .completed(let completion) = construction.snapshot.disposition else {
            throw Failure.noProof
        }
        try await construction.sealForPublication(completion)
        return .init(
            container: container, engine: built.engine, construction: construction,
            audioPermit: permit, contract: built.contract)
    }
    private func input(mixed: Bool = false) throws -> MiMoV26MultimodalInput {
        let pcm = try MiMoV26DecodedPCM(
            samples: (0 ..< 2400).map { sin(Float($0) * 0.02) * 0.05 },
            descriptor: .init(
                sourceIdentity: "synthetic-sine-owned-pcm", channels: 1,
                frameCount: 2400, sampleRate: 24000))
        var parts: [MiMoV26MultimodalContent] = []
        if mixed {
            parts.append(
                .image(
                    .init(
                        height: 4, width: 4,
                        planarRGB: (0 ..< 48).map { Float(($0 * 17) % 256) })))
        }
        parts += [.text("listen"), .audio(pcm), .text("describe")]
        return .init(messages: [.init(role: .user, content: parts)], maximumOutputTokens: 2)
    }
    private func prepared(
        _ fixture: Fixture, input: MiMoV26MultimodalInput,
        reservation: WorkPermit = WorkPermit(), cancel: Cancel = Cancel()
    ) async throws -> (CBv2Request, WorkPermit) {
        let request = try await fixture.container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            return try model.prepareManagedDecodedAudioMedia(
                input, engine: fixture.engine,
                authorize: { plan, bytes in
                    try reservation.admit(plan, bytes: bytes)
                    return reservation
                },
                retire: { _ in try reservation.retire() }, isCancelled: { cancel.cancelled })
        }
        return (request, reservation)
    }
    private func shutdown(_ fixture: Fixture) async throws {
        guard
            case .quiescent(let receipt) = await fixture.engine.shutdownReportingNativeCompletion()
        else {
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

    func testRealPCMAndOrderedMixedPreparationSubmitRetireThenTextOnSameEngine() async throws {
        let f = try await fixture()
        XCTAssertTrue(f.contract.supportsDecodedAudioMedia)
        XCTAssertFalse(f.contract.supportsDecodedVisualMedia)
        XCTAssertEqual(f.contract.audioSidecarSessionID, f.audioPermit?.request.sessionID)
        let epoch = f.construction.snapshot.epoch
        for mixed in [false, true] {
            var (request, permit) = try await prepared(f, input: input(mixed: mixed))
            request.id = .init(mixed ? 2 : 1)
            request.sampling = .init(temperature: 0)
            XCTAssertNotNil(request.multimodal?.nativeMediaToken)
            XCTAssertGreaterThan(permit.bytes, 0)
            XCTAssertEqual(permit.orderedKinds, mixed ? ["image", "audio"] : ["audio"])
            XCTAssertEqual(permit.pcmFrames, [2400])
            let submission = try f.engine.submitWithNativeRetirement(request)
            let result = await cbv2SchedCollect(submission.events)
            await submission.retirement.wait()
            XCTAssertFalse(result.tokens.isEmpty)
            XCTAssertEqual(permit.retired, 1)
            XCTAssertEqual(permit.bytes, 0)
            XCTAssertEqual(f.audioPermit?.reservedLoadBytes, 3_879_023_504)
            XCTAssertThrowsError(try f.engine.submitWithNativeRetirement(request))
        }
        let text = CBv2Request(
            id: .init(3), promptTokens: [20, 21, 22, 23],
            sampling: .init(temperature: 0), maxTokens: 2, prefixCacheEnabled: false)
        let submission = try f.engine.submitWithNativeRetirement(text)
        let textResult = await cbv2SchedCollect(submission.events)
        XCTAssertFalse(textResult.tokens.isEmpty)
        await submission.retirement.wait()
        XCTAssertEqual(f.construction.snapshot.epoch, epoch)
        try await shutdown(f)
    }

    func testTextAndVisualProfilesDoNotGainAudioFromInstalledSidecar() async throws {
        for profile in [Profile.text, .visual] {
            let f = try await fixture(profile: profile)
            XCTAssertFalse(f.contract.supportsDecodedAudioMedia)
            do {
                _ = try await prepared(f, input: input())
                XCTFail("audio profile bypass")
            } catch { XCTAssertEqual(error as? MiMoV26MultimodalError, .incompatibleOwner) }
            XCTAssertNil(f.engine.nativeCompletionFault)
            if profile == .visual {
                let pcm = try input()
                do {
                    _ = try await f.container.perform { context in
                        let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                        return try model.prepareManagedDecodedMedia(
                            pcm, engine: f.engine,
                            authorize: { _, _ in
                                XCTFail("visual profile admitted PCM")
                                return WorkPermit()
                            },
                            retire: { _ in XCTFail("visual profile issued audio retirement") },
                            isCancelled: { false })
                    }
                    XCTFail("visual PCM accepted")
                } catch { XCTAssertEqual(error as? MiMoV26MultimodalError, .missingAudioCodec) }
            }
            try await shutdown(f)
        }
    }

    func testActualInstalledTextMTPRemainsActiveAfterTargetOnlyAudio() async throws {
        let f = try await fixture(enableMTP: true)
        XCTAssertTrue(CBv2MTPConfig.envEnabled, "run ON selector with the real kill switch enabled")
        let (request, permit) = try await prepared(f, input: input())
        let media = try f.engine.submitWithNativeRetirement(request)
        _ = await cbv2SchedCollect(media.events)
        await media.retirement.wait()
        XCTAssertEqual(permit.retired, 1)
        XCTAssertEqual(
            try XCTUnwrap(f.engine.mtpMetricsSnapshot()).draftedTokens, 0,
            "audio row must use the real target-only media fallback")
        let text = CBv2Request(
            id: .init(19), promptTokens: Array(repeating: 20, count: 13),
            sampling: .init(temperature: 0), maxTokens: 8, stopTokens: [], prefixCacheEnabled: false
        )
        let submission = try f.engine.submitWithNativeRetirement(text)
        let output = await cbv2SchedCollect(submission.events)
        await submission.retirement.wait()
        XCTAssertEqual(output.tokens.count, 8)
        XCTAssertGreaterThan(try XCTUnwrap(f.engine.mtpMetricsSnapshot()).draftedTokens, 0)
        try await shutdown(f)
    }

    func testActualSidecarInvalidationAtBindingRefusesWhenBareCodecStillValid() async throws {
        let f = try await fixture()
        let (request, permit) = try await prepared(f, input: input())
        try await f.container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            let sidecar = try XCTUnwrap(model.resources.audioSidecar)
            try sidecar.codec.validate()
            sidecar.invalidate()
            XCTAssertNoThrow(
                try sidecar.codec.validate(), "discriminator: bare codec is insufficient")
        }
        XCTAssertThrowsError(try f.engine.submitWithNativeRetirement(request))
        XCTAssertEqual(permit.retired, 0)
        f.engine.discardUnsubmittedNativeMedia(try XCTUnwrap(request.multimodal))
        XCTAssertEqual(permit.retired, 1)
        try await shutdown(f)
    }

    func testForeignActualEngineCannotBorrowTheLoadedAudioProfile() async throws {
        let audio = try await fixture()
        let foreign = try await fixture(profile: .text, installAudio: false)
        let pcm = try input()
        do {
            _ = try await audio.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                return try model.prepareManagedDecodedAudioMedia(
                    pcm, engine: foreign.engine,
                    authorize: { _, _ in
                        XCTFail("foreign engine reached audio admission")
                        return WorkPermit()
                    },
                    retire: { _ in XCTFail("foreign engine issued retirement") },
                    isCancelled: { false })
            }
            XCTFail("foreign engine accepted")
        } catch { XCTAssertEqual(error as? MiMoV26MultimodalError, .incompatibleOwner) }
        XCTAssertNil(audio.engine.nativeCompletionFault)
        XCTAssertNil(foreign.engine.nativeCompletionFault)
        try await shutdown(foreign)
        try await shutdown(audio)
    }

    func testSupportedLoadedGenerationInvalidationRejectsAlreadyPreparedAudio() async throws {
        let f = try await fixture()
        let (request, permit) = try await prepared(f, input: input())
        try await f.container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            model.invalidateMultimodalPreparation()
        }
        XCTAssertThrowsError(try f.engine.submitWithNativeRetirement(request))
        XCTAssertEqual(permit.retired, 0)
        f.engine.discardUnsubmittedNativeMedia(try XCTUnwrap(request.multimodal))
        XCTAssertEqual(permit.retired, 1)
        try await shutdown(f)
    }

    func testSeparateSidecarReservationRevocationRefusesBeforeWarmWorkAndAtActualBind() async throws
    {
        let f = try await fixture()
        let (request, permit) = try await prepared(f, input: input())
        try XCTUnwrap(f.audioPermit).revoke()
        do {
            _ = try await prepared(f, input: input())
            XCTFail("revoked load owner prepared audio")
        } catch { XCTAssertEqual(error as? MiMoV26AudioSidecarError, .invalidatedOwner) }
        XCTAssertThrowsError(try f.engine.submitWithNativeRetirement(request))
        XCTAssertNil(f.engine.nativeCompletionFault)
        f.engine.discardUnsubmittedNativeMedia(try XCTUnwrap(request.multimodal))
        XCTAssertEqual(permit.retired, 1)
        try await shutdown(f)
    }

    func testAudioLoanPrecedesCodecAndSuccessfullyDrainedCancellationKeepsEngineHealthy()
        async throws
    {
        let f = try await fixture()
        let permit = WorkPermit()
        let cancel = Cancel()
        // Actual audio-only source sequence: outer authorize/check(1/2), part
        // check(3), audio phase check(4), inner codec pre-front-end check(5),
        // post-front-end/pre-eval check(6). No elapsed-time approximation.
        permit.onValidation = { count in
            if count == 4 {
                cancel.recordRootsBeforeAudio(
                    f.engine.loopForTesting.nativeShutdownState?.debugRetainedRootCount ?? 0)
            }
            if count == 5 {
                guard let before = cancel.rootsBeforeAudio else {
                    return XCTFail("missing actual pre-audio phase")
                }
                XCTAssertEqual(
                    f.engine.loopForTesting.nativeShutdownState?.debugRetainedRootCount, before + 1,
                    "exact new registration must precede codec work; the pre-existing owner root is not proof"
                )
            }
            if count == 6 { cancel.set() }  // real mel eval/readback then typed cancellation/drain
        }
        do {
            _ = try await prepared(f, input: input(), reservation: permit, cancel: cancel)
            XCTFail("cancelled PCM prepared")
        } catch { XCTAssertEqual(error as? MiMoV26MultimodalError, .cancelled) }
        XCTAssertTrue(cancel.cancelled, "must enter actual codec, not a pre-work cancellation")
        XCTAssertEqual(permit.retired, 1)
        XCTAssertEqual(permit.bytes, 0)
        XCTAssertNil(f.engine.nativeCompletionFault)
        let (request, healthy) = try await prepared(f, input: input())
        let submission = try f.engine.submitWithNativeRetirement(request)
        _ = await cbv2SchedCollect(submission.events)
        await submission.retirement.wait()
        XCTAssertEqual(healthy.retired, 1)
        try await shutdown(f)
    }

    func testInnerRequiredCodesReadbackFailureRetainsOriginalInnerRootsBeforeCleanup() async throws
    {
        #if DEBUG
            // Skip, not fail, on a machine without the lane, such as the hosted CI runner.
            try XCTSkipUnless(
                ProcessInfo.processInfo.environment["MIMO_V26_MANAGED_AUDIO_FAULT_TEST"] == "1",
                "Requires the exclusive native GPU lane. Set MIMO_V26_MANAGED_AUDIO_FAULT_TEST=1 to run it."
            )
            let f = try await fixture()
            let permit = WorkPermit()
            let reached = Cancel()
            try await f.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                let input = try XCTUnwrap(model.resources.audioSidecar).codec.input
                input.beforeManagedRequiredCompletionForTesting = { phase in
                    if case .codesReadback = phase {
                        reached.set()
                        throw Failure.injected
                    }
                }
            }
            do {
                _ = try await prepared(f, input: input(), reservation: permit)
                XCTFail("required inner readback succeeded")
            } catch { XCTAssertEqual(error as? MiMoV26MultimodalError, .drainFailed) }
            XCTAssertTrue(
                reached.cancelled,
                "real frontend/encoder/RVQ evaluation must reach the required readback")
            XCTAssertFalse(
                permit.failedAudioRootCounts.isEmpty,
                "the ORIGINAL inner owner must transfer before cleanup")
            XCTAssertTrue(permit.failedAudioRootCounts.allSatisfy { $0 > 0 })
            XCTAssertEqual(permit.retired, 0)
            XCTAssertGreaterThan(permit.bytes, 0)
            XCTAssertEqual(f.audioPermit?.reservedLoadBytes, 3_879_023_504)
            let roots = permit.failedAudioRootCounts
            try await f.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                try XCTUnwrap(model.resources.audioSidecar).codec.input
                    .beforeManagedRequiredCompletionForTesting = nil
            }
            guard case .incomplete = await f.engine.shutdownReportingNativeCompletion() else {
                XCTFail("required inner failure was rehabilitated")
                _ = Unmanaged.passRetained(f.engine)
                return
            }
            XCTAssertEqual(permit.failedAudioRootCounts, roots)
            XCTAssertEqual(permit.retired, 0)
            _ = Unmanaged.passRetained(f.engine)
            _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(permit)
            _ = Unmanaged.passRetained(try XCTUnwrap(f.audioPermit))
        // Refusal injected at a required readback after REAL native evaluation,
        // not evidence of a physical backend readback failure.
        #else
            throw XCTSkip("Requires a debug build. The fault hook exists only in debug builds.")
        #endif
    }

    func testRequiredPreparationFenceFailureRetainsAudioOwnerAndBothObligations() async throws {
        // Skip, not fail, on a machine without the lane, such as the hosted CI runner.
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MIMO_V26_MANAGED_AUDIO_FAULT_TEST"] == "1",
            "Requires the exclusive native GPU lane. Set MIMO_V26_MANAGED_AUDIO_FAULT_TEST=1 to run it."
        )
        let f = try await fixture()
        let permit = WorkPermit()
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeShutdownState?.beforeFenceForTesting = { _ in
                throw Failure.injected
            }
        }
        do {
            _ = try await prepared(f, input: input(), reservation: permit)
            XCTFail("failed fence prepared audio")
        } catch { XCTAssertEqual(error as? MiMoV26MultimodalError, .drainFailed) }
        XCTAssertEqual(permit.retired, 0)
        XCTAssertGreaterThan(permit.bytes, 0)
        XCTAssertGreaterThan(permit.retainedFailures, 0)
        XCTAssertEqual(f.audioPermit?.reservedLoadBytes, 3_879_023_504)
        try await f.container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            XCTAssertNotNil(model.resources.audioSidecar)
        }
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.nativeShutdownState?.beforeFenceForTesting = nil
        }
        guard case .incomplete = await f.engine.shutdownReportingNativeCompletion() else {
            XCTFail("late successful fence rehabilitated the original required failure")
            _ = Unmanaged.passRetained(f.engine)
            return
        }
        XCTAssertEqual(permit.retired, 0)
        _ = Unmanaged.passRetained(f.engine)
        _ = Unmanaged.passRetained(f.container)
        _ = Unmanaged.passRetained(permit)
        _ = Unmanaged.passRetained(try XCTUnwrap(f.audioPermit))
        // Injected BEFORE-fence fault, not evidence of a physical backend failure.
    }
}
