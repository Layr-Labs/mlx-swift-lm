import Foundation
import MLX
import MLXLLM
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Prepared only. Requires an exclusive native test lane and the existing tiny
/// strict BF16 files. Real vision features/target/KV/seals are used throughout.
/// Host reservation callbacks below witness SDK ownership, NOT provider ledger
/// qualification or measured deadline throughput. No fabricated feature arrays.
final class MiMoV26NativeMediaDeadlineTests: XCTestCase {
    private enum FixtureError: Error {
        case fixtureRequired, nativeLaneRequired, missingReceipt, wrongOutcome, rawClosure
    }
    private final class Permit: MiMoV26SerialLoadReservation, Sendable {
        let request: MiMoV26SerialLoadRequest
        let reservedLoadBytes: UInt64
        init(_ request: MiMoV26SerialLoadRequest) {
            self.request = request
            reservedLoadBytes = request.requiredLoadBytes
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private final class Reservation: MiMoV26MediaWorkReservation, @unchecked Sendable {
        private let lock = NSLock()
        private var digest: String?
        private var releaseCount = 0
        private var failed: [MiMoV26FailedMediaWork] = []
        var retired: Int { lock.withLock { releaseCount } }
        func install(_ plan: MiMoV26MultimodalPlan, bytes: Int) throws {
            guard bytes > 0 else { throw MiMoV26MultimodalError.reservationRejected }
            try lock.withLock {
                guard digest == nil else { throw MiMoV26MultimodalError.incompatiblePlan }
                digest = plan.preparationSHA256
            }
        }
        func validate(plan: MiMoV26MultimodalPlan) throws {
            try lock.withLock {
                guard releaseCount == 0, digest == plan.preparationSHA256 else {
                    throw MiMoV26MultimodalError.reservationRejected
                }
            }
        }
        func retainAfterFailedDrain(_ work: MiMoV26FailedMediaWork) {
            lock.withLock { failed.append(work) }
        }
        func retire() throws {
            try lock.withLock {
                guard failed.isEmpty, releaseCount == 0 else { throw FixtureError.missingReceipt }
                releaseCount += 1
            }
        }
    }
    private final class Gate: @unchecked Sendable {
        let entered: XCTestExpectation
        private let released = DispatchSemaphore(value: 0)
        init(_ entered: XCTestExpectation) { self.entered = entered }
        func hold() {
            entered.fulfill()
            released.wait()
        }
        func release() { released.signal() }
    }
    /// Routing-only tokenizer: not a tokenizer/Jinja parity oracle.
    private final class RoutingTokenizer: Tokenizer, Sendable {
        let ids = [
            "<|im_end|>": 1, "<|image_pad|>": 2, "<|video_pad|>": 3,
            "<|vision_start|>": 4, "<|vision_end|>": 5, "<|audio_pad|>": 6,
            "<|mimo_audio_start|>": 7, "<|mimo_audio_end|>": 8,
            "<|mimo_video_start|>": 9, "<|mimo_video_end|>": 10,
        ]
        var bosToken: String? { nil }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.utf8.map { 20 + Int($0) % 32 }
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }
        func convertTokenToId(_ token: String) -> Int? { ids[token] }
        func convertIdToToken(_ id: Int) -> String? { ids.first { $0.value == id }?.key }
        func applyChatTemplate(
            messages: [Message], tools: [ToolSpec]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try applyChatTemplate(
                messages: messages, chatTemplate: "routing-only", tools: tools,
                additionalContext: additionalContext)
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            try applyChatTemplate(
                messages: messages, chatTemplate: chatTemplate, tools: nil, additionalContext: nil)
        }
        func applyChatTemplate(
            messages: [Message], chatTemplate: String, tools: [ToolSpec]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var result = [11]
            for message in messages {
                if let text = message["content"] as? String {
                    result += encode(text: text, addSpecialTokens: false)
                } else if let parts = message["content"] as? [[String: any Sendable]] {
                    for part in parts {
                        switch part["type"] as? String {
                        case "image": result += [4, 2, 5]
                        case "video": result += [4, 3, 5]
                        default:
                            result += encode(
                                text: part["text"] as? String ?? "", addSpecialTokens: false)
                        }
                    }
                }
            }
            return result
                + ((additionalContext?["enable_thinking"] as? Bool) == false ? [12, 13] : [12])
        }
    }
    private struct Loader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { RoutingTokenizer() }
    }
    private struct Fixture: Sendable {
        let container: ModelContainer
        let engine: EngineV2
    }
    private func fixture(tracked: Bool = true, maxWaiting: Int = 4) async throws -> Fixture {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1",
            environment["MIMO_V26_NATIVE_MEDIA_DEADLINE_TESTS"] == "1"
        else {
            throw FixtureError.nativeLaneRequired
        }
        guard let path = environment["MIMO_V26_SERIAL_LOAD_FIXTURES"] else {
            throw FixtureError.fixtureRequired
        }
        let fixtures = URL(fileURLWithPath: path)
        let root = fixtures.appendingPathComponent("media-deadline-" + UUID().uuidString)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("tiny-bf16"), to: root)
        let configURL = root.appendingPathComponent("config.json")
        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any])
        var processor = try XCTUnwrap(config["processor_config"] as? [String: Any])
        processor["patch_size"] = 2
        processor["image_min_pixels"] = 16
        processor["image_max_pixels"] = 256
        processor["video_min_pixels"] = 16
        processor["video_max_pixels"] = 256
        processor["video_total_max_pixels"] = 512
        processor["video_start_token_id"] = 9
        processor["video_end_token_id"] = 10
        config["processor_config"] = processor
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(
            to: configURL)
        try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["eos_token": "<|im_end|>"])
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data("{{ messages }}{% if enable_thinking is false %}<think></think>{% endif %}".utf8)
            .write(to: root.appendingPathComponent("chat_template.jinja"))
        let provenanceJSON = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with:
                    Data(contentsOf: fixtures.appendingPathComponent("provenance.json")))
                as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(
            artifactID: XCTUnwrap(provenanceJSON["artifactID"]),
            sourceRepository: XCTUnwrap(provenanceJSON["sourceRepository"]),
            sourceRevision: XCTUnwrap(provenanceJSON["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(provenanceJSON["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304))
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let prepared = try await MiMoV26ModelFactory.prepare(
            request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let work = NativeConstructionWork()
        defer { if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) } }
        let container = try await MiMoV26ModelFactory.loadContainer(
            session: session,
            reservation: Permit(session.request), prepared: prepared, retaining: work)
        try await work.acknowledgeContainerAdoption(container)
        let engine = try await MiMoV26ModelFactory.withNativeConstruction(
            container: container, retaining: work
        ) { model, scope in
            let binding = try model.makeCBv2Binding()
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let resources = try model.makeManagedMediaExecutionResources(
                binding: binding,
                bytesCapacity: 32 << 20, limits: MiMoMediaFixture.limits, retaining: scope)
            let engine = EngineV2(
                model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 4,
                    prefillChunkSize: 3, maxConcurrentPartialPrefills: 1,
                    maxWaiting: maxWaiting, enablePrefixCache: false),
                loopConfig: .init(stepTimeout: 60, watchdogInterval: 0.01, shutdownTimeout: 10),
                nativeCompletionTracking: tracked, nativeExecutionContract: resources.contract)
            try scope.retainOwner(engine)
            return engine
        }
        guard case .completed(let completion) = work.snapshot.disposition else {
            throw FixtureError.missingReceipt
        }
        try await work.sealForPublication(completion)
        return .init(container: container, engine: engine)
    }
    private func prepared(_ f: Fixture, id: UInt64 = 1) async throws -> (CBv2Request, Reservation) {
        let reservation = Reservation()
        let input = MiMoV26MultimodalInput(
            messages: [
                .init(
                    role: .user,
                    content: [
                        .image(
                            .init(
                                height: 4, width: 4,
                                planarRGB: (0 ..< 48).map { Float(($0 * 17) % 256) })),
                        .text("tail"),
                    ])
            ], maximumOutputTokens: 3)
        var request = try await f.container.perform { context in
            let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
            return try model.prepareManagedDecodedMedia(
                input, engine: f.engine,
                authorize: { plan, bytes in
                    try reservation.install(plan, bytes: bytes)
                    return reservation
                },
                retire: { _ in try reservation.retire() }, isCancelled: { false })
        }
        request.id = .init(id)
        request.sampling = .init(temperature: 0)
        return (request, reservation)
    }
    private func policy(
        _ deadline: ContinuousClock.Instant = .now.advanced(by: .seconds(600)),
        rate: Double? = 1
    ) -> CBv2FirstTokenDeadlineAdmission {
        .init(
            deadline: deadline, conservativePrefillTokensPerSecond: rate,
            conservativeDecodeTokensPerSecond: rate)
    }
    private func checkUnbound(_ f: Fixture, _ request: CBv2Request, _ reservation: Reservation)
        throws
    {
        let token = try XCTUnwrap(request.multimodal?.nativeMediaToken)
        f.engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(token.stream)
            XCTAssertFalse(token.disposed)
            XCTAssertNotNil(token.work.resolved)
            XCTAssertTrue(f.engine.loopForTesting.nativeShutdownState?.hasLoans == true)
            XCTAssertNil(f.engine.loopForTesting.stream(for: request.id))
            XCTAssertNil(f.engine.loopForTesting.multimodalByID[request.id])
            XCTAssertEqual(f.engine.loopForTesting.backend.bytesReserved, 0)
        }
        XCTAssertEqual(reservation.retired, 0)
    }
    private func discard(_ f: Fixture, _ request: CBv2Request, _ reservation: Reservation) throws {
        f.engine.discardUnsubmittedNativeMedia(try XCTUnwrap(request.multimodal))
        let token = try XCTUnwrap(request.multimodal?.nativeMediaToken)
        f.engine.loopForTesting.onEngineQueueSync {
            XCTAssertTrue(token.disposed)
            XCTAssertNil(token.stream)
            XCTAssertNil(token.work.resolved)
            XCTAssertFalse(f.engine.loopForTesting.nativeShutdownState?.hasLoans == true)
        }
        XCTAssertEqual(reservation.retired, 1)
    }
    private func shutdown(_ f: Fixture) async throws {
        guard case .quiescent = await f.engine.shutdownReportingNativeCompletion() else {
            _ = Unmanaged.passRetained(f.engine)
            throw FixtureError.missingReceipt
        }
    }

    func testMediaReservationRefusalLeavesTextEngineUsable() async throws {
        let f = try await fixture()
        let (probe, probeReservation) = try await prepared(f)
        let token = try XCTUnwrap(probe.multimodal?.nativeMediaToken)
        XCTAssertLessThanOrEqual(
            token.work.rootIDs.count, 5,
            "completed vision checkpoints must not accumulate old layer roots")
        try discard(f, probe, probeReservation)
        func text(_ id: UInt64) async throws -> CBv2SchedCollected {
            var request = CBv2Request(id: .init(id), promptTokens: [20, 21], maxTokens: 3)
            request.sampling = .init(temperature: 0)
            let submitted = try f.engine.submitWithNativeRetirement(request)
            let result = await cbv2SchedCollect(submitted.events)
            await submitted.retirement.wait()
            return result
        }
        let baseline = try await text(100)
        XCTAssertFalse(baseline.tokens.isEmpty)
        XCTAssertTrue(baseline.finishReason == .length || baseline.finishReason == .stop)
        for _ in 0 ..< 3 {
            do {
                _ = try await f.container.perform { context in
                    let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
                    return try model.prepareManagedDecodedMedia(
                        MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]),
                        engine: f.engine,
                        authorize: { _, _ in throw MiMoV26MultimodalError.reservationRejected },
                        retire: { _ in XCTFail("an unissued reservation cannot retire") },
                        isCancelled: { false })
                }
                XCTFail("media should have been refused")
            } catch {
                XCTAssertEqual(error as? MiMoV26MultimodalError, .reservationRejected)
            }
            XCTAssertNil(f.engine.nativeCompletionFault)
            f.engine.loopForTesting.onEngineQueueSync {
                XCTAssertFalse(f.engine.loopForTesting.nativeShutdownState?.hasLoans == true)
            }
        }
        let after = try await text(101)
        XCTAssertEqual(after.tokens, baseline.tokens)
        XCTAssertEqual(after.finishReason, baseline.finishReason)
        try await shutdown(f)
    }

    func testPreparedImageUsesBoundedRealProjectionAndMatchesOrdinaryGreedyTokens() async throws {
        let f = try await fixture()
        let (request, owner) = try await prepared(f)
        guard
            case .admitted(let stream, let work, _, let retirement) =
                try await f.engine.submit(request, firstTokenDeadline: policy())
        else { throw FixtureError.wrongOutcome }
        guard case .bounded(let scheduled, let service) = work else {
            throw FixtureError.wrongOutcome
        }
        XCTAssertEqual(scheduled.prefillTokens, request.promptTokens.count)
        XCTAssertGreaterThan(scheduled.scheduledSteps, 0)
        XCTAssertGreaterThan(service, .zero)
        let actual = await cbv2SchedCollect(stream)
        await retirement.wait()
        XCTAssertEqual(owner.retired, 1)
        let (ordinary, ordinaryOwner) = try await prepared(f, id: 2)
        let other = try f.engine.submitWithNativeRetirement(ordinary)
        let expected = await cbv2SchedCollect(other.events)
        await other.retirement.wait()
        XCTAssertEqual(actual.tokens, expected.tokens)
        XCTAssertEqual(actual.finishReason, expected.finishReason)
        XCTAssertEqual(ordinaryOwner.retired, 1)
        do {
            _ = try await f.engine.submit(request, firstTokenDeadline: policy())
            XCTFail("consumed seal replay was accepted")
        } catch { XCTAssertTrue(error is CBv2NativeShutdownError) }
        try await shutdown(f)
    }

    func testExpiredAndMissingRateKeepSealUnboundForRetryOrDiscard() async throws {
        let f = try await fixture()
        let (request, owner) = try await prepared(f)
        for admission in [policy(.now.advanced(by: .seconds(-1))), policy(rate: nil)] {
            guard
                case .deadlineUnreachable = try await f.engine.submit(
                    request, firstTokenDeadline: admission)
            else {
                throw FixtureError.wrongOutcome
            }
            try checkUnbound(f, request, owner)
        }
        guard
            case .admitted(let stream, _, _, let retirement) =
                try await f.engine.submit(request, firstTokenDeadline: policy())
        else { throw FixtureError.wrongOutcome }
        _ = await cbv2SchedCollect(stream)
        await retirement.wait()
        XCTAssertEqual(owner.retired, 1)
        let (discarded, otherOwner) = try await prepared(f, id: 2)
        guard
            case .deadlineUnreachable = try await f.engine.submit(
                discarded,
                firstTokenDeadline: policy(.now.advanced(by: .seconds(-1))))
        else { throw FixtureError.wrongOutcome }
        try discard(f, discarded, otherOwner)
        try await shutdown(f)
    }

    func testCancellationBeforeAtomicCommitDoesNotBindFeaturesOrStrandLoan() async throws {
        let f = try await fixture()
        let (request, owner) = try await prepared(f)
        let gate = Gate(expectation(description: "deadline closure before seal validation"))
        defer { gate.release() }
        f.engine.loopForTesting.setDeadlineAdmissionInitialGuardHookForTesting { _ in gate.hold() }
        let admission = policy()
        let submission = Task { try await f.engine.submit(request, firstTokenDeadline: admission) }
        await fulfillment(of: [gate.entered], timeout: 5)
        submission.cancel()
        gate.release()
        do {
            _ = try await submission.value
            XCTFail("cancelled operation admitted")
        } catch { XCTAssertTrue(error is CancellationError) }
        try checkUnbound(f, request, owner)
        try discard(f, request, owner)
        try await shutdown(f)
    }

    func testGenerationChangeWhileQueuedIsRevalidatedWithoutConsumption() async throws {
        let f = try await fixture()
        let (request, owner) = try await prepared(f)
        let gate = Gate(expectation(description: "registered, not yet queued"))
        defer { gate.release() }
        f.engine.loopForTesting.setDeadlineAdmissionBeforeRegistrationHookForTesting { _ in
            gate.hold()
        }
        let admission = policy()
        let submission = Task { try await f.engine.submit(request, firstTokenDeadline: admission) }
        await fulfillment(of: [gate.entered], timeout: 5)
        try await f.container.perform { context in
            try XCTUnwrap(context.model as? MiMoV26LoadedModel).invalidateMultimodalPreparation()
        }
        gate.release()
        do {
            _ = try await submission.value
            XCTFail("stale generation admitted")
        } catch { XCTAssertEqual(error as? MiMoV26MultimodalError, .invalidatedOwner) }
        try checkUnbound(f, request, owner)
        try discard(f, request, owner)
        try await shutdown(f)
    }

    func testCancellationAfterCommitTransfersRealMediaRetirement() async throws {
        let f = try await fixture()
        let (request, owner) = try await prepared(f)
        let gate = Gate(expectation(description: "media and stream admission committed"))
        defer { gate.release() }
        f.engine.loopForTesting.setDeadlineAdmissionCommittedHookForTesting { _ in gate.hold() }
        let admission = policy()
        let submission = Task { try await f.engine.submit(request, firstTokenDeadline: admission) }
        await fulfillment(of: [gate.entered], timeout: 5)
        submission.cancel()
        gate.release()
        do {
            _ = try await submission.value
            XCTFail("post-commit cancellation lost ownership transfer")
        } catch let transfer as CBv2FirstTokenAdmissionCancellation {
            _ = await cbv2SchedCollect(transfer.stream)
            await transfer.retirement.wait()
        }
        let token = try XCTUnwrap(request.multimodal?.nativeMediaToken)
        f.engine.loopForTesting.onEngineQueueSync {
            XCTAssertTrue(token.disposed)
            XCTAssertNil(token.stream)
            XCTAssertNil(token.work.resolved)
            XCTAssertFalse(f.engine.loopForTesting.nativeShutdownState?.hasLoans == true)
            XCTAssertEqual(f.engine.loopForTesting.backend.bytesReserved, 0)
        }
        XCTAssertEqual(owner.retired, 1)
        try await shutdown(f)
    }

    func testLateShutdownAndExplicitDiscardLeaveNoBoundSealForColdRefusal() async throws {
        let f = try await fixture()
        let (request, owner) = try await prepared(f)
        let gate = Gate(expectation(description: "registration held before native close"))
        defer { gate.release() }
        f.engine.loopForTesting.setDeadlineAdmissionBeforeRegistrationHookForTesting { _ in
            gate.hold()
        }
        let admission = policy()
        let submission = Task { try await f.engine.submit(request, firstTokenDeadline: admission) }
        await fulfillment(of: [gate.entered], timeout: 5)
        // Existing unsubmitted-media disposal is legitimate while shutdown
        // drains; no admitted stream or native target row owns this seal.
        let closing = Task { await f.engine.shutdownReportingNativeCompletion() }
        f.engine.discardUnsubmittedNativeMedia(try XCTUnwrap(request.multimodal))
        // A pending registered submission itself prevents quiescence. Let its
        // real cold refusal retire the gauge before awaiting shutdown.
        gate.release()
        do {
            _ = try await submission.value
            XCTFail("closed generation admitted")
        } catch { XCTAssertTrue(error is CBv2KVError || error is CBv2NativeShutdownError) }
        guard case .quiescent = await closing.value else { throw FixtureError.missingReceipt }
        XCTAssertEqual(owner.retired, 1)
        let token = try XCTUnwrap(request.multimodal?.nativeMediaToken)
        XCTAssertNil(token.stream)
        XCTAssertTrue(token.disposed)
        XCTAssertEqual(
            f.engine.loopForTesting.deadlineAdmissionRegistrationSnapshotForTesting.operationCount,
            0)
    }

    func testRawAndForeignSealsRefuseAndPlainTextDeadlineRemainsBounded() async throws {
        let f = try await fixture()
        let other = try await fixture()
        let (request, owner) = try await prepared(f)
        do {
            _ = try await other.engine.submit(request, firstTokenDeadline: policy())
            XCTFail("foreign native seal accepted")
        } catch { XCTAssertTrue(error is CBv2NativeShutdownError) }
        var raw = request
        raw.multimodal = CBv2MultimodalInput(
            spans: try XCTUnwrap(request.multimodal).spans, attention: .causal
        ) {
            throw FixtureError.rawClosure
        }
        do {
            _ = try await f.engine.submit(raw, firstTokenDeadline: policy())
            XCTFail("raw native media accepted")
        } catch { XCTAssertTrue(error is CBv2NativeShutdownError) }
        try checkUnbound(f, request, owner)
        try discard(f, request, owner)
        guard
            case .admitted(let stream, let work, _, let retirement) =
                try await f.engine.submit(
                    .init(
                        id: .init(2), promptTokens: [20, 21, 22],
                        sampling: .init(temperature: 0), maxTokens: 1), firstTokenDeadline: policy()
                )
        else {
            throw FixtureError.wrongOutcome
        }
        guard case .bounded(let scheduled, _) = work else { throw FixtureError.wrongOutcome }
        XCTAssertEqual(scheduled.prefillTokens, 3)
        _ = await cbv2SchedCollect(stream)
        await retirement.wait()
        try await shutdown(f)
        try await shutdown(other)
    }

    func testElapsedQueueWaitDoesNotResetAbsoluteDeadline() async throws {
        let f = try await fixture()
        let (request, owner) = try await prepared(f)
        let gate = Gate(expectation(description: "hold admitted-queue position past deadline"))
        defer { gate.release() }
        f.engine.loopForTesting.setDeadlineAdmissionInitialGuardHookForTesting { _ in gate.hold() }
        let deadline = ContinuousClock.now.advanced(by: .seconds(1))
        let admission = policy(deadline)
        let submission = Task { try await f.engine.submit(request, firstTokenDeadline: admission) }
        await fulfillment(of: [gate.entered], timeout: 5)
        try await ContinuousClock().sleep(until: deadline.advanced(by: .milliseconds(10)))
        gate.release()
        guard case .deadlineUnreachable = try await submission.value else {
            throw FixtureError.wrongOutcome
        }
        try checkUnbound(f, request, owner)
        try discard(f, request, owner)
        try await shutdown(f)
    }

    func testMaxWaitingCapacityRefusalLeavesPreparedSealDiscardable() async throws {
        let f = try await fixture(maxWaiting: 1)
        let (request, owner) = try await prepared(f)
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.suspendStepExecutionAtCountForTesting =
                f.engine.loopForTesting.stepCount
        }
        let blocking = try f.engine.submitWithNativeRetirement(
            .init(
                id: .init(90),
                promptTokens: [20, 21, 22], sampling: .init(temperature: 0), maxTokens: 1))
        f.engine.loopForTesting.onEngineQueueSync {}  // actual enqueue, no target step
        do {
            _ = try await f.engine.submit(request, firstTokenDeadline: policy())
            XCTFail("full waiting queue accepted media")
        } catch { XCTAssertTrue(error is CBv2KVError) }
        try checkUnbound(f, request, owner)
        try discard(f, request, owner)
        f.engine.cancel(.init(90))
        f.engine.loopForTesting.onEngineQueueSync {
            f.engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        _ = await cbv2SchedCollect(blocking.events)
        await blocking.retirement.wait()
        try await shutdown(f)
    }

    func testUntrackedRawCausalMediaStillReturnsUnboundedWithoutCallingProvider() async throws {
        let f = try await fixture(tracked: false)
        let raw = CBv2Request(
            id: .init(1), promptTokens: [20, 2, 22], maxTokens: 1,
            multimodal: .init(spans: [.init(tokenOffset: 1, length: 1)], attention: .causal) {
                throw FixtureError.rawClosure
            })
        guard
            case .deadlineUnreachable(.unbounded) =
                try await f.engine.submit(raw, firstTokenDeadline: policy())
        else { throw FixtureError.wrongOutcome }
        XCTAssertEqual(f.engine.stepCount, 0)
        XCTAssertEqual(f.engine.capacity().waitingRequests, 0)
        await f.engine.shutdown()
    }
}
