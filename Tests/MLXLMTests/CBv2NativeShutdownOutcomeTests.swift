import Foundation
import MLX
import MLXLLM
import XCTest

@_spi(Diagnostics) @testable import MLXLMCommon
@testable import MLXVLM

/// Source-prepared. These tests load the real bounded BF16 component fixture,
/// run its actual native target/assistant and use the protected resource issuer.
/// The tokenizer controls factory setup only; it is not template parity proof.
final class CBv2NativeShutdownOutcomeTests: XCTestCase {
    private enum Failure: Error { case fence, missingReceipt }
    private final class Permit: MiMoV26SerialLoadReservation, Sendable {
        let request: MiMoV26SerialLoadRequest
        let reservedLoadBytes: UInt64
        init(_ request: MiMoV26SerialLoadRequest) {
            self.request = request
            reservedLoadBytes = request.requiredLoadBytes
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private final class ControlTokenizer: Tokenizer, Sendable {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [6] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined()
        }
        func convertTokenToId(_ token: String) -> Int? { token == "<|im_end|>" ? 3 : nil }
        func convertIdToToken(_ id: Int) -> String? { id == 3 ? "<|im_end|>" : nil }
        var bosToken: String? { nil }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [Message], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            try applyChatTemplate(
                messages: messages, chatTemplate: chatTemplate, tools: nil, additionalContext: nil)
        }
        func applyChatTemplate(
            messages: [Message], chatTemplate: String, tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            [1] + ((additionalContext?["enable_thinking"] as? Bool) == false ? [4, 5] : []) + [6]
        }
    }
    private struct Loader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { ControlTokenizer() }
    }
    private final class OneShotGate: @unchecked Sendable {
        let entered: XCTestExpectation
        private let lock = NSLock()
        private var enteredOnce = false
        private let released = DispatchSemaphore(value: 0)
        init(_ entered: XCTestExpectation) { self.entered = entered }
        func holdOnce(_ beforeHold: () -> Void = {}) {
            let first = lock.withLock { () -> Bool in
                guard !enteredOnce else { return false }
                enteredOnce = true
                return true
            }
            guard first else { return }
            beforeHold()
            entered.fulfill()
            released.wait()
        }
        func release() { released.signal() }
        // No destructor waits; a one-shot signal is never consumed twice.
    }

    private func lane(fault: String? = nil) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1" else {
            throw XCTSkip("Requires an authorized exclusive native test process")
        }
        guard env["MIMO_V26_SHUTDOWN_FAULT_CASE"] == fault else {
            throw XCTSkip(
                "Run each retained-fault selector alone with its exact MIMO_V26_SHUTDOWN_FAULT_CASE"
            )
        }
    }
    private func engine(
        tracked: Bool = true, mtp: Bool = false, text: Bool = false,
        shutdownTimeout: TimeInterval = 10,
        stepTimeout: TimeInterval = 60
    ) async throws -> EngineV2 {
        let fixtureRoot = URL(
            fileURLWithPath: try XCTUnwrap(
                ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_LOAD_FIXTURES"]))
        let root = fixtureRoot.appendingPathComponent("shutdown-work-" + UUID().uuidString)
        try FileManager.default.copyItem(
            at: fixtureRoot.appendingPathComponent("tiny-bf16"), to: root)
        try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["eos_token": "<|im_end|>"])
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data("{{ messages }}{% if enable_thinking is false %}<think></think>{% endif %}".utf8)
            .write(to: root.appendingPathComponent("chat_template.jinja"))
        let p = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with:
                    Data(contentsOf: fixtureRoot.appendingPathComponent("provenance.json")))
                as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(
            artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]),
            sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304))
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let prepared = try await MiMoV26ModelFactory.prepare(
            request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let work = NativeConstructionWork()
        let container = try await MiMoV26ModelFactory.loadContainer(
            session: session,
            reservation: Permit(session.request), prepared: prepared, retaining: work)
        try await work.acknowledgeContainerAdoption(container)
        let actual = try await MiMoV26ModelFactory.withNativeConstruction(
            container: container, retaining: work
        ) { model, scope in
            let binding = try model.makeCBv2Binding(enableMTP: mtp)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let resources = try binding.adapter.makeNativeExecutionResources(
                bytesCapacity: 32 << 20, retaining: scope)
            let detokenizer: any CBv2DetokenizerFactory =
                text
                ? CBv2TextDetokenizerFactory(tokenizer: ControlTokenizer())
                : CBv2NullDetokenizerFactory()
            let engine = EngineV2(
                model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                detokenizerFactory: detokenizer,
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16,
                    prefillChunkSize: 3, maxWaiting: 4, enablePrefixCache: false),
                loopConfig: .init(
                    stepTimeout: stepTimeout, watchdogInterval: 0.01,
                    shutdownTimeout: shutdownTimeout),
                mtpDrafter: binding.assistant,
                mtpConfig: .init(
                    enabled: mtp, maxDraftTokens: 3, maxSpeculativeBatch: 1,
                    fixedDraftTokens: 3, verificationMode: .serialTarget),
                nativeCompletionTracking: tracked, nativeExecutionContract: resources.contract)
            try scope.retainOwner(engine)
            return engine
        }
        guard case .completed(let setup) = work.snapshot.disposition else {
            throw Failure.missingReceipt
        }
        try await work.sealForPublication(setup)
        return actual
    }
    private func request(_ id: UInt64 = 1, budget: Int = 12, logprobs: Int = 0) -> CBv2Request {
        .init(
            id: .init(id), promptTokens: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            sampling: .init(temperature: 0, topLogprobs: logprobs), maxTokens: budget)
    }
    private func collect(_ engine: EngineV2, _ request: CBv2Request) async throws -> [Int] {
        let result = await cbv2SchedCollect(try engine.submit(request))
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens.count, request.maxTokens)
        XCTAssertEqual(result.usage?.completionTokens, request.maxTokens)
        return result.tokens
    }
    private func receipt(_ outcome: CBv2NativeShutdownOutcome, _ engine: EngineV2) throws
        -> CBv2NativeShutdownReceipt
    {
        guard case .quiescent(let receipt) = outcome else {
            XCTFail("not quiescent: \(outcome)")
            throw Failure.missingReceipt
        }
        XCTAssertEqual(receipt.engineID, engine.nativeShutdownEngineID)
        XCTAssertEqual(receipt.generation, 1)
        XCTAssertEqual(
            receipt.executionContractID, engine.loopForTesting.nativeShutdownState?.contractID)
        XCTAssertGreaterThanOrEqual(receipt.capturedStreamCount, 2)
        return receipt
    }

    func testActualOFFAndONNaturalDrainPreserveTokensAndRetireRoots() async throws {
        try lane()
        var reference: [Int]?
        var engineIDs = Set<UUID>()
        var contractIDs = Set<UUID>()
        for mtp in [false, true] {
            let actual = try await engine(mtp: mtp, text: mtp)
            if mtp { XCTAssertNil(actual.mtpInactiveReason) }
            let tokens = try await collect(actual, request())
            if let reference { XCTAssertEqual(tokens, reference) } else { reference = tokens }
            let first = await actual.shutdownReportingNativeCompletion()
            let proof = try receipt(first, actual)
            XCTAssertTrue(engineIDs.insert(proof.engineID).inserted)
            XCTAssertTrue(contractIDs.insert(proof.executionContractID).inserted)
            let again = await actual.shutdownReportingNativeCompletion()
            XCTAssertEqual(first, again)
            await actual.shutdown()  // delegates the same result; no second cleanup
            XCTAssertEqual(actual.loopForTesting.nativeShutdownState?.debugRetainedRootCount, 0)
            actual.loopForTesting.onEngineQueueSync {
                XCTAssertEqual(actual.loopForTesting.backend.bytesInUse, 0)
                XCTAssertEqual(actual.loopForTesting.backend.bytesReserved, 0)
                XCTAssertFalse(actual.loopForTesting.hasInFlightStepForTesting)
                XCTAssertFalse(actual.loopForTesting.scheduler.hasWork)
            }
            XCTAssertThrowsError(try actual.submit(request(2)))
        }
    }

    func testUntrackedActualEngineKeepsLegacyVoidDrainAndCannotMintReceipt() async throws {
        try lane()
        let actual = try await engine(tracked: false)
        XCTAssertNil(actual.loopForTesting.nativeShutdownState)
        _ = try await collect(actual, request())
        let observation = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = observation else { return XCTFail("untracked success") }
        XCTAssertEqual(fault.reason, .notTracked)
        _ = try await collect(actual, request(2))  // observation did not shut down legacy engine
        await actual.shutdown()
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertEqual(actual.loopForTesting.backend.bytesInUse, 0)
            XCTAssertEqual(actual.loopForTesting.backend.bytesReserved, 0)
        }
    }

    func testNaturalOutcomeDoesNotLeaveEngineOwnedByTimeoutTimer() async throws {
        try lane()
        var actual: EngineV2? = try await engine(shutdownTimeout: 60)
        weak var engineWitness = actual
        weak var loopWitness = actual?.loopForTesting
        _ = try await collect(actual!, request())
        _ = try receipt(await actual!.shutdownReportingNativeCompletion(), actual!)
        actual!.loopForTesting.onEngineQueueSync {}  // join the reporting closure tail
        actual = nil
        XCTAssertNil(engineWitness)
        XCTAssertNil(loopWitness, "pending watchdog must not retain a healthy model for 60 seconds")
    }

    func testPendingCancelledShutdownWaiterStillRequiresActualFence() async throws {
        try lane()
        let actual = try await engine(shutdownTimeout: 60)
        _ = try await collect(actual, request())
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(expectation(description: "actual shutdown boundary reached"))
        defer { gate.release() }
        actual.loopForTesting.onEngineQueueSync {
            tracking.beforeFenceForTesting = { _ in gate.holdOnce() }
        }
        let waiter = Task { await actual.shutdownReportingNativeCompletion() }
        let concurrent = Task { await actual.shutdownReportingNativeCompletion() }
        await fulfillment(of: [gate.entered], timeout: 10)
        XCTAssertNil(tracking.outcome)
        waiter.cancel()
        XCTAssertNil(tracking.outcome, "caller cancellation is not a native completion fault")
        gate.release()
        let first = await waiter.value
        _ = try receipt(first, actual)
        let second = await concurrent.value
        XCTAssertEqual(first, second)
    }

    func testDiagnosticLoansAndUnsupportedConsumersRemainExplicit() async throws {
        try lane()
        let actual = try await engine()
        let top = try actual.teacherForcedTop1(promptTokens: [1, 2, 4], continuation: [5, 6, 7])
        XCTAssertEqual(top.count, 3)
        let digest = try actual.prefillLogitDigest([1, 2, 4])
        XCTAssertGreaterThan(digest.count, 0)
        XCTAssertThrowsError(try actual.submit(request(logprobs: 1)))
        XCTAssertThrowsError(try actual.configureAttentionMetadata(nil))
        XCTAssertThrowsError(try actual.configureAttentionPacket(nil))
        XCTAssertThrowsError(try actual.configureLogitDiagnostic(nil))
        _ = try receipt(await actual.shutdownReportingNativeCompletion(), actual)
        XCTAssertEqual(actual.loopForTesting.nativeShutdownState?.debugRetainedRootCount, 0)
    }

    func testTimeoutAfterActualSubmissionRetainsRowsAndRefusesLateRetirement() async throws {
        try lane(fault: "testTimeoutAfterActualSubmissionRetainsRowsAndRefusesLateRetirement")
        let actual = try await engine(shutdownTimeout: 0)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(expectation(description: "real asyncEval submitted"))
        defer { gate.release() }
        weak var row: AnyObject?
        actual.loopForTesting.onEngineQueueSync {
            tracking.afterSubmissionForTesting = {
                gate.holdOnce {
                    row =
                        actual.loopForTesting.kvStates.values.flatMap { $0 }.compactMap { $0 }.first
                }
            }
        }
        let stream = try actual.submit(request())
        await fulfillment(of: [gate.entered], timeout: 10)
        XCTAssertNotNil(row)
        let before = tracking.debugRetainedRootCount
        XCTAssertGreaterThan(before, 0)
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else {
            return XCTFail("held queue minted success")
        }
        XCTAssertEqual(fault.reason, .shutdownTimedOut)
        let terminal = await cbv2SchedCollect(stream)
        guard case .error? = terminal.finishReason else { return XCTFail("missing client error") }
        gate.release()
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertNotNil(row)
            XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
            XCTAssertGreaterThanOrEqual(tracking.debugRetainedRootCount, before)
        }
        actual.updateKVBytesCapacity(0)
        XCTAssertGreaterThan(actual.capacity().kvBytesCapacity, 0, "late shrink must be vetoed")
        XCTAssertThrowsError(try actual.submit(request(2)))
        await actual.shutdown()
        let repeated = await actual.shutdownReportingNativeCompletion()
        XCTAssertEqual(outcome, repeated)
        // Do not reset/release fault roots: this selector runs alone and exits.
    }

    func testCapturedFenceFailureAttemptsAllStreamsAndNeverIssuesLateSuccess() async throws {
        try lane(fault: "testCapturedFenceFailureAttemptsAllStreamsAndNeverIssuesLateSuccess")
        let actual = try await engine()
        _ = try await collect(actual, request())
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        var attempted: [MLX.Stream] = []
        actual.loopForTesting.onEngineQueueSync {
            tracking.beforeFenceForTesting = { stream in
                attempted.append(stream)
                if stream == StreamOrDevice.cpu.stream { throw Failure.fence }
            }
        }
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else {
            return XCTFail("failed fence minted receipt")
        }
        XCTAssertEqual(fault.reason, .capturedFenceFailed)
        actual.loopForTesting.onEngineQueueSync { XCTAssertGreaterThanOrEqual(attempted.count, 2) }
        let again = await actual.shutdownReportingNativeCompletion()
        XCTAssertEqual(outcome, again)
        XCTAssertThrowsError(try actual.submit(request(2)))
    }

    func testStepWatchdogFirstWinnerRemainsIncompleteWithoutShutdownTimeout() async throws {
        try lane(fault: "testStepWatchdogFirstWinnerRemainsIncompleteWithoutShutdownTimeout")
        let actual = try await engine(shutdownTimeout: 60, stepTimeout: 1)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(
            expectation(description: "submitted work held for real step watchdog"))
        let wedged = expectation(description: "watchdog reported actual held step")
        defer { gate.release() }
        actual.loopForTesting.onEngineQueueSync {
            tracking.afterSubmissionForTesting = { gate.holdOnce() }
            actual.loopForTesting.onStepWedge = { _ in wedged.fulfill() }
        }
        let stream = try actual.submit(request())
        await fulfillment(of: [gate.entered, wedged], timeout: 20)
        guard case .incomplete(let fault)? = tracking.outcome else {
            return XCTFail("watchdog did not seal fault")
        }
        XCTAssertEqual(fault.reason, .stepWatchdog)
        let first = await actual.shutdownReportingNativeCompletion()
        gate.release()
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
            XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
        }
        _ = await cbv2SchedCollect(stream)
        let again = await actual.shutdownReportingNativeCompletion()
        XCTAssertEqual(first, again)
    }

    func testMTPFenceLateReturnCannotReplaceMeasuredOwnersAfterTimeout() async throws {
        try lane(fault: "testMTPFenceLateReturnCannotReplaceMeasuredOwnersAfterTimeout")
        let actual = try await engine(mtp: true, shutdownTimeout: 0)
        XCTAssertNil(actual.mtpInactiveReason)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(
            expectation(description: "actual assistant fence completed before commit"))
        defer { gate.release() }
        weak var state: MiMoV26MTPState?
        var measuredBytes = -1
        var measuredRoots = -1
        actual.loopForTesting.onEngineQueueSync {
            tracking.afterAssistantFenceForTesting = { value in
                gate.holdOnce {
                    state = value as? MiMoV26MTPState
                    measuredBytes = state?.materializedBytes ?? -1
                    measuredRoots = state?.measuredRootCount ?? -1
                }
            }
        }
        let stream = try actual.submit(request())
        await fulfillment(of: [gate.entered], timeout: 10)
        XCTAssertNotNil(state)
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else {
            return XCTFail("held assistant minted receipt")
        }
        XCTAssertEqual(fault.reason, .shutdownTimedOut)
        gate.release()
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertNotNil(state)
            XCTAssertFalse(state?.isReleased ?? true)
            XCTAssertEqual(state?.materializedBytes, measuredBytes)
            XCTAssertEqual(state?.measuredRootCount, measuredRoots)
            XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
            XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
        }
        _ = await cbv2SchedCollect(stream)
        let again = await actual.shutdownReportingNativeCompletion()
        XCTAssertEqual(outcome, again)
    }

    func testExecutionTicketRefusesWrongExpiredAndAlreadyConsumedIdentity() throws {
        let model = NSObject()
        let backend = NSObject()
        let bank = NSObject()
        let assistant = NSObject()
        let scope = NativeConstructionScope()
        XCTAssertThrowsError(
            try CBv2NativeExecutionContract(
                model: model, backend: backend,
                cacheProvider: bank, assistant: assistant, construction: scope)
        ) {
            XCTAssertEqual($0 as? NativeConstructionError, .inactiveScope)
        }
        let contract = try scope.withPhase(.nativeSetup) {
            try CBv2NativeExecutionContract(
                model: model, backend: backend,
                cacheProvider: bank, assistant: assistant, construction: scope)
        }
        XCTAssertEqual(contract.constructionOwnerID, scope.snapshot.ownerID)
        XCTAssertEqual(contract.constructionEpoch, scope.snapshot.epoch)
        XCTAssertFalse(
            contract.consume(
                model: model, backend: NSObject(), cacheProvider: bank, assistant: assistant))
        XCTAssertFalse(
            contract.consume(model: model, backend: backend, cacheProvider: bank, assistant: nil))
        XCTAssertTrue(
            contract.consume(
                model: model, backend: backend, cacheProvider: bank, assistant: assistant))
        XCTAssertFalse(
            contract.consume(
                model: model, backend: backend, cacheProvider: bank, assistant: assistant))
        var temporary: NSObject? = NSObject()
        let otherScope = NativeConstructionScope()
        let expired = try otherScope.withPhase(.nativeSetup) {
            try CBv2NativeExecutionContract(
                model: temporary!, backend: backend,
                cacheProvider: bank, assistant: nil, construction: otherScope)
        }
        XCTAssertNotEqual(expired.constructionOwnerID, contract.constructionOwnerID)
        temporary = nil
        XCTAssertFalse(
            expired.consume(
                model: NSObject(), backend: backend, cacheProvider: bank, assistant: nil))
        // This metadata-only test does not mint any successful engine receipt.
    }
}
