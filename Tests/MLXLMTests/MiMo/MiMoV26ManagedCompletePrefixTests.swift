import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import Tokenizers
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Prepared native component tests, UNRUN. Strict small target and (audio case)
/// the actual authenticated selected sidecar. Test-only bounded ledger/encoded
/// transport are not provider global-memory or encrypted-store qualification.
final class MiMoV26ManagedCompletePrefixTests: XCTestCase {
    private enum Failure: Error { case inputRequired, noProof, injected }
    private enum Profile: Sendable, Equatable { case audio, visual }
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
        private var charge = 0, releaseCount = 0
        private var failures: [MiMoV26FailedMediaWork] = []
        var bytes: Int { lock.withLock { charge } }
        var retired: Int { lock.withLock { releaseCount } }
        func admit(_ plan: MiMoV26MultimodalPlan, bytes: Int) throws {
            try lock.withLock {
                guard digest == nil, bytes > 0 else {
                    throw MiMoV26MultimodalError.reservationRejected
                }
                digest = plan.preparationSHA256
                charge = bytes
            }
        }
        func validate(plan: MiMoV26MultimodalPlan) throws {
            try lock.withLock {
                guard digest == plan.preparationSHA256, charge > 0, releaseCount == 0 else {
                    throw MiMoV26MultimodalError.reservationRejected
                }
            }
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
    private struct Loader: TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let raw = try await AutoTokenizer.from(modelFolder: directory)
            return #adaptHuggingFaceTokenizer(raw)
        }
    }

    /// Existing encoded checkpoint fixture transport with real async export
    /// jobs and close joins. Provider tests below use the actual encrypted store.
    private final class Store: CBv2NativeCompletePrefixCache, @unchecked Sendable {
        let base = CompleteCheckpointFixtureStore(segmentBytes: 1 << 20)
        private let jobs = DispatchGroup()
        var identity: CBv2CompleteCheckpointIdentity { base.identity }
        func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
            base.acceptsCheckpoint(position: position, packedBytes: packedBytes)
        }
        func takeStaged(
            requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
            maximumSequenceLength: Int
        ) -> CBv2StagedCompleteCheckpoint? {
            base.takeStaged(
                requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                maximumSequenceLength: maximumSequenceLength)
        }
        func donate(
            _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?, tokens: [Int],
            cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
        ) {
            jobs.enter()
            base.donate(source, requestID: requestID, tokens: tokens, cacheSalt: cacheSalt) {
                [self] positions in
                completion(positions)
                jobs.leave()
            }
        }
        func close() { base.close() }
        func closeAndWait() async {
            close()
            await withCheckedContinuation { continuation in
                jobs.notify(queue: .global()) { continuation.resume() }
            }
        }
    }
    private final class ProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
        private let lock = NSLock()
        private var c: UInt64 = 0, m: UInt64 = 0
        private var closed = false
        var bytes: UInt64 { lock.withLock { c } }
        func replaceCharge(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= m, bytes <= 256 << 20 else { throw Failure.noProof }
                c = bytes
            }
        }
        func recordMaterialization(_ bytes: UInt64) throws {
            try lock.withLock {
                guard !closed, bytes >= m, bytes <= c else { throw Failure.noProof }
                m = bytes
            }
        }
        func withdrawCoverage(_ bytes: UInt64) throws {
            try lock.withLock {
                guard bytes <= m else { throw Failure.noProof }
                m -= bytes
            }
        }
        func retire() {
            lock.withLock {
                XCTAssertEqual(c, 0)
                closed = true
            }
        }
    }
    private struct Fixture: Sendable {
        let container: ModelContainer
        let engine: EngineV2
        let construction: NativeConstructionWork
        let audioPermit: AudioPermit?
        let contract: CBv2NativeExecutionContract
        let store: Store
        let process: ProcessOwner
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
            config.numHiddenLayers <= 4, config.maxPositionEmbeddings >= 1024
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
        let store = Store()
        let process = ProcessOwner()
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
            let metadata = try model.nativeCompletePrefixMetadata(
                binding: binding, retaining: scope)
            let resources: MiMoV26ManagedMediaExecutionResources
            switch profile {
            case .audio:
                resources = try model.makeManagedAudioCompletePrefixExecutionResources(
                    binding: binding,
                    bytesCapacity: 32 << 20, expectedMetadata: metadata, completePrefixCache: store,
                    processMemoryOwner: process, limits: Self.limits, retaining: scope)
            case .visual:
                resources = try model.makeManagedMediaCompletePrefixExecutionResources(
                    binding: binding,
                    bytesCapacity: 32 << 20, expectedMetadata: metadata, completePrefixCache: store,
                    processMemoryOwner: process, limits: Self.limits, retaining: scope)
            }
            XCTAssertFalse(
                resources.contract.consume(
                    model: binding.adapter, backend: resources.backend,
                    cacheProvider: resources.cacheProvider, assistant: binding.assistant,
                    completePrefixCache: Store(), processMemoryOwner: process))
            XCTAssertFalse(
                resources.contract.consume(
                    model: binding.adapter, backend: resources.backend,
                    cacheProvider: resources.cacheProvider, assistant: binding.assistant,
                    completePrefixCache: store, processMemoryOwner: ProcessOwner()))
            let engine = EngineV2(
                model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 128,
                    prefillChunkSize: 128, maxConcurrentPartialPrefills: 1, enablePrefixCache: true),
                loopConfig: .init(stepTimeout: 60, watchdogInterval: 0.01, shutdownTimeout: 10),
                completePrefixCache: store,
                mtpDrafter: binding.assistant,
                mtpConfig: .init(
                    enabled: enableMTP, maxDraftTokens: 3,
                    maxSpeculativeBatch: 1, fixedDraftTokens: 3, verificationMode: .serialTarget),
                processMemoryOwner: process,
                nativeCompletionTracking: true, nativeExecutionContract: resources.contract)
            try scope.retainOwner(engine)
            XCTAssertNil(engine.nativeCompletionFault)
            guard engine.nativeCompletionFault == nil else { throw Failure.noProof }
            return Built(engine: engine, contract: resources.contract)
        }
        guard case .completed(let completion) = construction.snapshot.disposition else {
            throw Failure.noProof
        }
        try await construction.sealForPublication(completion)
        return .init(
            container: container, engine: built.engine, construction: construction,
            audioPermit: permit, contract: built.contract, store: store, process: process)
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

    private func collect(
        _ request: CBv2Request, fixture f: Fixture,
        expectDrafting: Bool = false, requireFirstDepthThree: Bool = false
    ) async throws -> CBv2SchedCollected {
        let before = f.engine.mtpMetricsSnapshot()
        if requireFirstDepthThree {
            XCTAssertTrue(
                try XCTUnwrap(before).perPositionAccepted.isEmpty,
                "the fixed-depth witness must start before this engine's first round")
        }
        let submission = try f.engine.submitWithNativeRetirement(request)
        let result = await cbv2SchedCollect(submission.events)
        guard result.finishReason == .length || result.finishReason == .stop else {
            throw Failure.noProof
        }
        await submission.retirement.wait()
        let after = f.engine.mtpMetricsSnapshot()
        let timing = try XCTUnwrap(result.usage).timing
        if expectDrafting {
            let a = try XCTUnwrap(before)
            let b = try XCTUnwrap(after)
            XCTAssertTrue(a.active)
            XCTAssertTrue(b.active)
            XCTAssertEqual(b.verificationMode, .serialTarget)
            XCTAssertGreaterThan(timing.mtpRounds, 0, "this request must actually verify drafts")
            XCTAssertGreaterThan(timing.mtpProposed, 0, "active MTP is not execution")
            XCTAssertEqual(b.rounds - a.rounds, Int(timing.mtpRounds))
            XCTAssertEqual(b.draftedTokens - a.draftedTokens, Int(timing.mtpProposed))
            XCTAssertEqual(
                b.serialVerificationRounds - a.serialVerificationRounds, Int(timing.mtpRounds))
            XCTAssertEqual(b.rectangularVerificationRounds - a.rectangularVerificationRounds, 0)
            if requireFirstDepthThree {
                // recordRound grows this vector by DRAFTED depth even if every
                // proposal is rejected. No acceptance requirement for tiny weights.
                XCTAssertEqual(b.perPositionAccepted.count, 3)
            }
        } else {
            XCTAssertEqual(timing.mtpRounds, 0)
            XCTAssertEqual(timing.mtpProposed, 0)
            if let a = before, let b = after {
                XCTAssertEqual(b.rounds, a.rounds)
                XCTAssertEqual(b.draftedTokens, a.draftedTokens)
                XCTAssertEqual(b.serialVerificationRounds, a.serialVerificationRounds)
                XCTAssertEqual(b.rectangularVerificationRounds, a.rectangularVerificationRounds)
            } else {
                XCTAssertNil(before)
                XCTAssertNil(after)
            }
        }
        return result
    }
    private func shutdown(_ f: Fixture) async throws {
        guard case .quiescent(let receipt) = await f.engine.shutdownReportingNativeCompletion()
        else {
            _ = Unmanaged.passRetained(f.engine)
            _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
            throw Failure.noProof
        }
        XCTAssertEqual(receipt.engineID, f.engine.nativeShutdownEngineID)
        XCTAssertEqual(receipt.executionContractID, f.contract.id)
        try await f.container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            try model.releaseNativeCompletePrefixAfterNativeRetirement(receipt)
            if f.contract.supportsDecodedAudioMedia {
                try model.releaseManagedAudioAfterNativeRetirement(receipt)
            }
        }
        XCTAssertEqual(f.process.bytes, 0)
    }
    private func exercise(audio: Bool, mtp: Bool) async throws {
        let f = try await fixture(
            profile: audio ? .audio : .visual, installAudio: audio, enableMTP: mtp)
        do {
            XCTAssertTrue(f.contract.supportsNativeCompletePrefix)
            XCTAssertTrue(f.contract.supportsManagedDecodedMedia)
            XCTAssertEqual(f.contract.supportsDecodedAudioMedia, audio)
            XCTAssertEqual(f.engine.mtpMetricsSnapshot()?.active == true, mtp)
            let engineID = f.engine.nativeShutdownEngineID
            let tokens = (0 ..< 513).map { 20 + ($0 % 9) }
            let first = CBv2Request(
                id: .init(41), promptTokens: tokens,
                sampling: .init(temperature: 0), maxTokens: 16, cacheSalt: "fixture-tenant",
                prefixCacheEnabled: true, prefixCacheReceiptID: .init(1041))
            let cold = try await collect(
                first, fixture: f, expectDrafting: mtp, requireFirstDepthThree: mtp)
            XCTAssertFalse(f.store.base.saved.isEmpty)
            var hit = first
            hit.id = .init(42)
            hit.prefixCacheReceiptID = .init(1042)
            XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: hit))
            let warm = try await collect(hit, fixture: f, expectDrafting: mtp)
            XCTAssertEqual(warm.tokens, cold.tokens)
            XCTAssertGreaterThan(warm.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
            let beforeMedia = f.store.base.saved.count
            let permit = WorkPermit()
            let mediaInput: MiMoV26MultimodalInput =
                audio
                ? try input(mixed: true)
                : .init(
                    messages: [
                        .init(
                            role: .user,
                            content: [
                                .image(
                                    .init(
                                        height: 4, width: 4, planarRGB: (0 ..< 48).map { Float($0) }
                                    )),
                                .text("describe"),
                            ])
                    ], additionalContext: ["enable_thinking": false], maximumOutputTokens: 2)
            var request = try await f.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                let authorize:
                    (MiMoV26MultimodalPlan, Int) throws -> any MiMoV26MediaWorkReservation = {
                        plan, bytes in
                        try permit.admit(plan, bytes: bytes)
                        return permit
                    }
                if audio {
                    return try model.prepareManagedDecodedAudioMedia(
                        mediaInput, engine: f.engine,
                        authorize: authorize, retire: { _ in try permit.retire() },
                        isCancelled: { false })
                }
                return try model.prepareManagedDecodedMedia(
                    mediaInput, engine: f.engine,
                    authorize: authorize, retire: { _ in try permit.retire() },
                    isCancelled: { false })
            }
            request.id = .init(43)
            request.sampling = .init(temperature: 0)
            XCTAssertFalse(request.prefixCacheEnabled)
            XCTAssertNotNil(request.multimodal?.nativeMediaToken)
            let mediaResult = try await collect(request, fixture: f)
            XCTAssertEqual(mediaResult.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
            XCTAssertEqual(f.store.base.saved.count, beforeMedia)
            XCTAssertEqual(permit.retired, 1)
            XCTAssertEqual(permit.bytes, 0)
            XCTAssertThrowsError(try f.engine.submitWithNativeRetirement(request))
            hit.id = .init(44)
            hit.prefixCacheReceiptID = .init(1044)
            XCTAssertTrue(try f.store.base.stage(engine: f.engine, request: hit))
            let afterMedia = try await collect(hit, fixture: f, expectDrafting: mtp)
            XCTAssertEqual(afterMedia.tokens, cold.tokens)
            XCTAssertGreaterThan(afterMedia.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
            XCTAssertEqual(f.engine.nativeShutdownEngineID, engineID)
            try await shutdown(f)
        } catch {
            _ = Unmanaged.passRetained(f.engine)
            _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
            throw error
        }
    }
    func testJointVisualKeepsActualTextReuseAndNoncacheableImageOnSameEngine() async throws {
        try await exercise(audio: false, mtp: false)
    }
    func testJointAudioAndThreeHeadMTPKeepTextReuseAfterRealMixedPCM() async throws {
        try await exercise(audio: true, mtp: true)
    }
    func testAlreadyIssuedJointProfileRejectsMetadataAndSeparateProfileUpgrade() async throws {
        let f = try await fixture(profile: .visual, installAudio: false)
        do {
            try await f.container.perform { context in
                let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                let scope = NativeConstructionScope()
                try scope.withPhase(.nativeSetup) {
                    try scope.authorizeImmutableLoadedOwner(model.resources)
                    let binding = try model.makeCBv2Binding()
                    XCTAssertThrowsError(
                        try model.nativeCompletePrefixMetadata(binding: binding, retaining: scope))
                    XCTAssertThrowsError(
                        try model.makeManagedMediaExecutionResources(
                            binding: binding,
                            bytesCapacity: 32 << 20, limits: Self.limits, retaining: scope))
                }
            }
            let request = CBv2Request(
                id: .init(51), promptTokens: [20, 21, 22, 23],
                sampling: .init(temperature: 0), maxTokens: 2, prefixCacheEnabled: false)
            let continued = try await collect(request, fixture: f)
            XCTAssertFalse(continued.tokens.isEmpty)
            try await shutdown(f)
        } catch {
            _ = Unmanaged.passRetained(f.engine)
            _ = Unmanaged.passRetained(f.container)
            _ = Unmanaged.passRetained(f.construction)
            throw error
        }
    }
}
