import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import Tokenizers
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Real decoded AV through the same strict target, selected codec and managed engine.
/// New feature oracle compares actual whole-clip/audio and video reference features;
/// it never substitutes a fake encoder, tokenizer, target or retirement receipt.
/// Requires an explicitly supplied small audio-compatible target fixture; this
/// class never weakens the selected sidecar hash/geometry or invents a loaded
/// owner. Reservation counters are SDK lifetime witnesses, NOT host ledger or
/// measured-peak qualification. Each fault selector needs its own process.
final class MiMoV26ManagedAudiovisualTests: XCTestCase {
    private enum Failure: Error { case inputRequired, nativeLaneRequired, noProof, injected }
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
        private var admittedPlan: MiMoV26MultimodalPlan?
        var plan: MiMoV26MultimodalPlan? { lock.withLock { admittedPlan } }
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
                admittedPlan = plan
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
        guard env["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1",
            env["MIMO_V26_MANAGED_AUDIO_NATIVE_TESTS"] == "1",
            env["MIMO_V26_DECODED_AV_NATIVE_TESTS"] == "1"
        else { throw Failure.nativeLaneRequired }
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

    private final class Count: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func increment() { lock.withLock { count += 1 } }
        var value: Int { lock.withLock { count } }
    }
    private func retainUnverified(_ f: Fixture) {
        // Test failure is not permission to drop an unknown native owner.
        _ = Unmanaged.passRetained(f.engine)
        _ = Unmanaged.passRetained(f.container)
        _ = Unmanaged.passRetained(f.construction)
    }
    private func clip(end: MiMoV26DecodedAudiovisual.SegmentEnd = .float32(0.48)) throws
        -> MiMoV26DecodedAudiovisual
    {
        let pcm = try MiMoV26DecodedPCM(
            samples: (0 ..< 11520).map { sin(Float($0) * 0.02) * 0.05 },
            descriptor: .init(
                sourceIdentity: "decoded-av-whole-sine-048", channels: 1,
                frameCount: 11520, sampleRate: 24000))
        let frames = [70, 90, 110].map { value in
            MiMoV26Pixels.DecodedRGB(
                height: 4, width: 4,
                planarRGB: (0 ..< 48).map { Float((value + $0 * 7) % 256) })
        }
        return .init(frames: frames, timestamps: [0, 0.16, 0.32], wholeAudio: pcm, segmentEnd: end)
    }
    private func message(_ content: MiMoV26MultimodalContent) -> MiMoV26MultimodalInput {
        .init(
            messages: [.init(role: .user, content: [.text("Describe."), content])],
            maximumOutputTokens: 2)
    }

    func testWholeEncodeOnceExactFeatureSlicesInterleaveFullBackingPriceAndRealRetirement()
        async throws
    {
        #if DEBUG
            let f = try await fixture()
            var closed = false
            defer { if !closed { retainUnverified(f) } }
            let av = try clip()
            let (whole, wholePermit) = try await prepared(f, input: message(.audio(av.wholeAudio)))
            let (video, videoPermit) = try await prepared(
                f,
                input: message(
                    .silentVideo(
                        .init(frames: av.frames, timestamps: av.timestamps))))
            let count = Count()
            try await f.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                try XCTUnwrap(model.resources.audioSidecar).codec.input
                    .beforeManagedRequiredCompletionForTesting = {
                        if case .codesReadback = $0 { count.increment() }
                    }
            }
            var (request, permit) = try await prepared(f, input: message(.audiovisual(av)))
            try await f.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                try XCTUnwrap(model.resources.audioSidecar).codec.input
                    .beforeManagedRequiredCompletionForTesting = nil
            }
            XCTAssertEqual(
                count.value, 1, "one actual whole clip encode, never one encode per AV unit")
            let plan = try XCTUnwrap(permit.plan)
            let a = try XCTUnwrap(wholePermit.plan)
            let v = try XCTUnwrap(videoPermit.plan)
            XCTAssertEqual(plan.spans.map(\.kind.rawValue), ["video", "audio", "video", "audio"])
            XCTAssertEqual(plan.spans.filter { $0.kind == .audio }.map(\.featureOffset), [0, 2])
            XCTAssertEqual(plan.spans.filter { $0.kind == .audio }.map(\.length), [2, 1])
            XCTAssertEqual(plan.audioPlan?.pcmDescriptors?.map(\.frameCount), [11520])
            XCTAssertEqual(plan.audioPlan?.codeFrameCounts, [13])
            XCTAssertEqual(plan.audioPlan?.patchCounts, [4])
            XCTAssertEqual(
                plan.featureElements, a.featureElements + v.featureElements,
                "last whole-audio feature row remains charged even though prompt spans use only three"
            )
            XCTAssertEqual(plan.logicalFeatureBytes, a.logicalFeatureBytes + v.logicalFeatureBytes)
            let geometry = try XCTUnwrap(plan.visionGeometryByMediaIndex[0])
            XCTAssertEqual(geometry.alignedFrames, 4)
            XCTAssertEqual(geometry.timestampCount, 2)
            XCTAssertEqual(
                permit.bytes,
                wholePermit.bytes + videoPermit.bytes
                    - Self.limits.pixels.maximumWorkingBytes + 32768 * geometry.timestampCount)
            let wholeAudio = try XCTUnwrap(a.spans.first)
            for span in plan.spans where span.kind == .audio {
                XCTAssertEqual(
                    plan.promptTokens[span.tokenOffset - 1],
                    a.promptTokens[wholeAudio.tokenOffset - 1])
                XCTAssertEqual(
                    plan.promptTokens[span.tokenOffset + span.length],
                    a.promptTokens[wholeAudio.tokenOffset + wholeAudio.length])
                XCTAssertEqual(
                    Array(plan.promptTokens[span.tokenOffset ..< (span.tokenOffset + span.length)]),
                    Array(repeating: a.promptTokens[wholeAudio.tokenOffset], count: span.length))
            }
            // Independent prompt oracle: unchanged silent-video timestamps/outer
            // framing, with each native whole-audio marker inserted after that
            // grid's vision_end. This catches grouped/late timestamp alternatives.
            var expectedPrompt: [Int] = []
            var cursor = 0
            for (index, span) in v.spans.enumerated() {
                let end = span.tokenOffset + span.length + 1
                expectedPrompt += v.promptTokens[cursor ..< end]
                expectedPrompt.append(a.promptTokens[wholeAudio.tokenOffset - 1])
                expectedPrompt += Array(
                    repeating: a.promptTokens[wholeAudio.tokenOffset], count: [2, 1][index])
                expectedPrompt.append(a.promptTokens[wholeAudio.tokenOffset + wholeAudio.length])
                cursor = end
            }
            expectedPrompt += v.promptTokens[cursor...]
            XCTAssertEqual(plan.promptTokens, expectedPrompt)
            // Features stay inside the exact owning queue. Diagnostic comparison
            // expressions are retained on the existing AV loan BEFORE evaluation.
            try f.engine.loopForTesting.onEngineQueueSync {
                let target = try XCTUnwrap(request.multimodal?.nativeMediaToken)
                let wholeToken = try XCTUnwrap(whole.multimodal?.nativeMediaToken)
                let videoToken = try XCTUnwrap(video.multimodal?.nativeMediaToken)
                let actual = try XCTUnwrap(target.work.resolved).embeddings
                let audio = try XCTUnwrap(wholeToken.work.resolved).embeddings
                let visual = try XCTUnwrap(videoToken.work.resolved).embeddings
                XCTAssertEqual(actual.count, 4)
                XCTAssertEqual(audio.count, 1)
                XCTAssertEqual(visual.count, 2)
                let expected = [
                    visual[0], audio[0][0..., 0 ..< 2, 0...], visual[1],
                    audio[0][0..., 2 ..< 3, 0...],
                ]
                let equalities = zip(actual, expected).map { pair in all(pair.0 .== pair.1) }
                try target.work.beforeNativeWork(actual + expected + equalities)
                do {
                    try withError { eval(equalities) }
                    for equality in equalities {
                        XCTAssertTrue(try withError { equality.item(Bool.self) })
                    }
                } catch {
                    target.work.requiredCompletionFailed()
                    throw error
                }
            }
            // Comparison graphs retained on the AV loan still reference these
            // independently prepared arrays. Keep BOTH real reference reservations
            // live until AV retirement removes those diagnostic aliases.
            XCTAssertEqual(wholePermit.retired, 0)
            XCTAssertEqual(videoPermit.retired, 0)
            XCTAssertGreaterThan(wholePermit.bytes, 0)
            XCTAssertGreaterThan(videoPermit.bytes, 0)
            XCTAssertEqual(permit.retired, 0)
            XCTAssertGreaterThan(permit.bytes, 0)
            XCTAssertThrowsError(
                try request.multimodal!.embeddings(), "no public native feature alias")
            request.id = .init(701)
            request.sampling = .init(temperature: 0)
            let submission = try f.engine.submitWithNativeRetirement(request)
            let result = await cbv2SchedCollect(submission.events)
            await submission.retirement.wait()
            XCTAssertFalse(result.tokens.isEmpty)
            XCTAssertEqual(permit.retired, 1)
            XCTAssertEqual(permit.bytes, 0)
            f.engine.discardUnsubmittedNativeMedia(try XCTUnwrap(whole.multimodal))
            f.engine.discardUnsubmittedNativeMedia(try XCTUnwrap(video.multimodal))
            XCTAssertEqual(wholePermit.retired, 1)
            XCTAssertEqual(videoPermit.retired, 1)
            XCTAssertThrowsError(
                try f.engine.submitWithNativeRetirement(request), "same native seal is one-shot")
            XCTAssertNil(f.engine.nativeCompletionFault)
            try await shutdown(f)
            closed = true
        #else
            throw Failure.inputRequired  // exact whole-encode observer is DEBUG-only; never silently skip
        #endif
    }

    func testVisualProfileAndEmptyNativeUnitRefuseBeforeAdmissionThenHealthyAVWorks() async throws {
        let visual = try await fixture(profile: .visual, installAudio: false)
        var visualClosed = false
        defer { if !visualClosed { retainUnverified(visual) } }
        let valid = message(.audiovisual(try clip()))
        do {
            _ = try await visual.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                return try model.prepareManagedDecodedMedia(
                    valid, engine: visual.engine,
                    authorize: { _, _ in
                        XCTFail("visual profile admitted linked audio")
                        return WorkPermit()
                    },
                    retire: { _ in XCTFail("visual profile issued audio retirement") },
                    isCancelled: { false })
            }
            XCTFail("visual profile accepted AV")
        } catch { XCTAssertEqual(error as? MiMoV26MultimodalError, .missingAudioCodec) }
        XCTAssertNil(visual.engine.nativeCompletionFault)
        try await shutdown(visual)
        visualClosed = true

        let f = try await fixture()
        var closed = false
        defer { if !closed { retainUnverified(f) } }
        let rejected = WorkPermit()
        do {
            _ = try await prepared(
                f,
                input: message(
                    .audiovisual(
                        try clip(end: .float64(Double(Float(0.48)))))), reservation: rejected)
            XCTFail("native empty last audio unit was padded or silently dropped")
        } catch {
            guard let failure = error as? MiMoV26MultimodalError,
                case .invalidInput = failure
            else { throw error }
        }
        XCTAssertNil(rejected.plan)
        XCTAssertEqual(rejected.bytes, 0)
        XCTAssertEqual(rejected.retired, 0)
        XCTAssertNil(f.engine.nativeCompletionFault)
        let (request, permit) = try await prepared(f, input: valid)
        XCTAssertEqual(permit.orderedKinds, ["video", "audio", "video", "audio"])
        f.engine.discardUnsubmittedNativeMedia(try XCTUnwrap(request.multimodal))
        XCTAssertEqual(permit.retired, 1)
        try await shutdown(f)
        closed = true
    }

    func testDecodedAVTargetOnlyThenActualTextMTPOnSameIssuedEngine() async throws {
        let f = try await fixture(enableMTP: true)
        var closed = false
        defer { if !closed { retainUnverified(f) } }
        XCTAssertTrue(CBv2MTPConfig.envEnabled, "ON selector requires actual kill switch enabled")
        let (request, permit) = try await prepared(f, input: message(.audiovisual(try clip())))
        let media = try f.engine.submitWithNativeRetirement(request)
        let mediaOutput = await cbv2SchedCollect(media.events)
        await media.retirement.wait()
        XCTAssertFalse(mediaOutput.tokens.isEmpty)
        XCTAssertEqual(permit.retired, 1)
        XCTAssertEqual(try XCTUnwrap(f.engine.mtpMetricsSnapshot()).draftedTokens, 0)
        let text = CBv2Request(
            id: .init(719), promptTokens: Array(repeating: 20, count: 13),
            sampling: .init(temperature: 0), maxTokens: 8, stopTokens: [], prefixCacheEnabled: false
        )
        let result = try f.engine.submitWithNativeRetirement(text)
        let output = await cbv2SchedCollect(result.events)
        await result.retirement.wait()
        XCTAssertEqual(output.tokens.count, 8)
        XCTAssertGreaterThan(try XCTUnwrap(f.engine.mtpMetricsSnapshot()).draftedTokens, 0)
        try await shutdown(f)
        closed = true
    }
}
