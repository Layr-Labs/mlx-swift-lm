import Foundation
import MLX
import MLXLLM
import XCTest
@_spi(Diagnostics) @testable import MLXLMCommon
@testable import MLXVLM

/// Source-prepared. These tests load the real bounded BF16 component fixture,
/// run its actual native target/assistant and use the protected resource issuer.
/// The tokenizer controls factory setup only; it is not template parity proof.
final class CBv2NativeRequestRetirementTests: XCTestCase {
    private enum Failure: Error { case fence, missingReceipt, submittedWork }
    private final class Permit: MiMoV26SerialLoadReservation, Sendable {
        let request: MiMoV26SerialLoadRequest
        let reservedLoadBytes: UInt64
        init(_ request: MiMoV26SerialLoadRequest) {
            self.request = request; reservedLoadBytes = request.requiredLoadBytes
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private final class ControlTokenizer: Tokenizer, Sendable {
        let decoderGate: DecodeGate?
        init(_ decoderGate: DecodeGate? = nil) { self.decoderGate = decoderGate }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [6] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            decoderGate?.visit()
            return tokenIds.map(String.init).joined()
        }
        func convertTokenToId(_ token: String) -> Int? { token == "<|im_end|>" ? 3 : nil }
        func convertIdToToken(_ id: Int) -> String? { id == 3 ? "<|im_end|>" : nil }
        var bosToken: String? { nil }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { nil }
        func applyChatTemplate(messages: [Message], tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, tools: nil, additionalContext: nil)
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String, tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] {
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
                guard !enteredOnce else { return false }; enteredOnce = true; return true
            }
            guard first else { return }
            beforeHold(); entered.fulfill(); released.wait()
        }
        func release() { released.signal() }
        // No destructor waits; a one-shot signal is never consumed twice.
    }
    private final class DecodeGate: @unchecked Sendable {
        let gate: OneShotGate
        let targetCall: Int
        private let lock = NSLock()
        private var calls = 0
        init(_ gate: OneShotGate, targetCall: Int) { self.gate = gate; self.targetCall = targetCall }
        func visit() {
            let current = lock.withLock { calls += 1; return calls }
            if current == targetCall { gate.holdOnce() }
        }
    }
    /// Test-control metadata only. Prevent a regression's unresolved Task from
    /// making the test itself wait forever after its bounded expectation fails.
    private final class SubmissionCompletion: @unchecked Sendable {
        private let lock = NSLock()
        private var finished = false
        func markFinished() { lock.withLock { finished = true } }
        var isFinished: Bool { lock.withLock { finished } }
    }

    private func lane(fault: String? = nil) throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1" else {
            throw XCTSkip("Requires an authorized exclusive native test process")
        }
        guard env["MIMO_V26_REQUEST_RETIREMENT_FAULT_CASE"] == fault else {
            throw XCTSkip("Run each retained-fault selector alone with its exact MIMO_V26_REQUEST_RETIREMENT_FAULT_CASE")
        }
    }
    private func engine(tracked: Bool = true, mtp: Bool = false, text: Bool = false, shutdownTimeout: TimeInterval = 10,
                        stepTimeout: TimeInterval = 60, decoderGate: DecodeGate? = nil,
                        serializedPrefill: Bool = false) async throws -> EngineV2 {
        let fixtureRoot = URL(fileURLWithPath: try XCTUnwrap(
            ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_LOAD_FIXTURES"]))
        let root = fixtureRoot.appendingPathComponent("request-retirement-work-" + UUID().uuidString)
        try FileManager.default.copyItem(at: fixtureRoot.appendingPathComponent("tiny-bf16"), to: root)
        try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["eos_token": "<|im_end|>"])
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data("{{ messages }}{% if enable_thinking is false %}<think></think>{% endif %}".utf8)
            .write(to: root.appendingPathComponent("chat_template.jinja"))
        let p = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: fixtureRoot.appendingPathComponent("provenance.json"))) as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]), sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304))
        let session = try MiMoV26SerialLoadSession(plan: plan)
        let prepared = try await MiMoV26ModelFactory.prepare(request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
        let work = NativeConstructionWork()
        let container = try await MiMoV26ModelFactory.loadContainer(session: session,
            reservation: Permit(session.request), prepared: prepared, retaining: work)
        try await work.acknowledgeContainerAdoption(container)
        let actual = try await MiMoV26ModelFactory.withNativeConstruction(container: container, retaining: work) { model, scope in
            let binding = try model.makeCBv2Binding(enableMTP: mtp)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let resources = try binding.adapter.makeNativeExecutionResources(bytesCapacity: 32 << 20, retaining: scope)
            let detokenizer: any CBv2DetokenizerFactory = text
                ? CBv2TextDetokenizerFactory(tokenizer: ControlTokenizer(decoderGate)) : CBv2NullDetokenizerFactory()
            let engine = EngineV2(model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                detokenizerFactory: detokenizer,
                schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: 16,
                    prefillChunkSize: 3, maxConcurrentPartialPrefills: serializedPrefill ? 1 : nil,
                    maxWaiting: 4, enablePrefixCache: false),
                loopConfig: .init(stepTimeout: stepTimeout, watchdogInterval: 0.01,
                    shutdownTimeout: shutdownTimeout),
                mtpDrafter: binding.assistant,
                mtpConfig: .init(enabled: mtp, maxDraftTokens: 3, maxSpeculativeBatch: 1,
                    fixedDraftTokens: 3, verificationMode: .serialTarget),
                nativeCompletionTracking: tracked, nativeExecutionContract: resources.contract)
            try scope.retainOwner(engine)
            return engine
        }
        guard case .completed(let setup) = work.snapshot.disposition else { throw Failure.missingReceipt }
        try await work.sealForPublication(setup)
        return actual
    }
    private func request(_ id: UInt64 = 1, budget: Int = 12, logprobs: Int = 0) -> CBv2Request {
        .init(id: .init(id), promptTokens: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              sampling: .init(temperature: 0, topLogprobs: logprobs), maxTokens: budget)
    }
    private func collect(_ engine: EngineV2, _ request: CBv2Request) async throws -> [Int] {
        let result = await cbv2SchedCollect(try engine.submit(request))
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens.count, request.maxTokens)
        XCTAssertEqual(result.usage?.completionTokens, request.maxTokens)
        return result.tokens
    }
    private func receipt(_ outcome: CBv2NativeShutdownOutcome, _ engine: EngineV2) throws -> CBv2NativeShutdownReceipt {
        guard case .quiescent(let receipt) = outcome else { XCTFail("not quiescent: \(outcome)"); throw Failure.missingReceipt }
        XCTAssertEqual(receipt.engineID, engine.nativeShutdownEngineID)
        XCTAssertEqual(receipt.generation, 1)
        XCTAssertEqual(receipt.executionContractID, engine.loopForTesting.nativeShutdownState?.contractID)
        XCTAssertGreaterThanOrEqual(receipt.capturedStreamCount, 2)
        return receipt
    }


    private func pausePlanning(_ engine: EngineV2) {
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = engine.loopForTesting.stepCount
        }
    }
    private func resumePlanning(_ engine: EngineV2) {
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
    }
    private func output(_ engine: EngineV2, _ id: UInt64) throws -> CBv2OutputStream {
        try engine.loopForTesting.onEngineQueueSync {
            try XCTUnwrap(engine.loopForTesting.stream(for: .init(id)))
        }
    }
    private func assertRetired(_ engine: EngineV2, _ output: CBv2OutputStream) {
        engine.loopForTesting.onEngineQueueSync {
            XCTAssertTrue(output.isEngineOwnershipReleased)
            XCTAssertEqual(engine.loopForTesting.nativePendingRetirementCountForTesting, 0)
            XCTAssertEqual(engine.loopForTesting.backend.bytesReserved, 0)
            XCTAssertEqual(engine.loopForTesting.backend.bytesInUse, 0)
        }
    }

    func testNormalTerminalAndActualRetirementShareOneGenerationAndTokenPath() async throws {
        try lane()
        let actual = try await engine()
        pausePlanning(actual)
        let submission = try actual.submitWithNativeRetirement(request())
        let registered = try output(actual, 1)
        XCTAssertFalse(registered.isEngineOwnershipReleased)
        resumePlanning(actual)
        let result = await cbv2SchedCollect(submission.events)
        await submission.retirement.wait()
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens.count, 12)
        XCTAssertEqual(result.usage?.completionTokens, 12)
        assertRetired(actual, registered)
        let ordinary = try await collect(actual, request(1)) // same now-reusable ID, ordinary API
        XCTAssertEqual(result.tokens, ordinary)
        _ = try receipt(await actual.shutdownReportingNativeCompletion(), actual)
    }

    func testUntrackedRefusesNewAPIWithoutChangingOrdinarySubmission() async throws {
        try lane()
        let actual = try await engine(tracked: false)
        let before = actual.stepCount
        XCTAssertThrowsError(try actual.submitWithNativeRetirement(request())) {
            XCTAssertTrue($0 is CBv2NativeShutdownError)
        }
        XCTAssertThrowsError(try actual.submitWithNativeRetirement(request(budget: 0)))
        XCTAssertEqual(actual.stepCount, before)
        XCTAssertEqual(actual.capacity().waitingRequests, 0)
        _ = try await collect(actual, request())
        await actual.shutdown()
    }

    func testZeroWorkIsAcknowledgedAndColdRefusalNeverRegisters() async throws {
        try lane()
        let actual = try await engine()
        let before = actual.stepCount
        let zero = try actual.submitWithNativeRetirement(request(budget: 0, logprobs: 1))
        let zeroResult = await cbv2SchedCollect(zero.events)
        await zero.retirement.wait()
        XCTAssertEqual(zeroResult.finishReason, .length)
        XCTAssertEqual(zeroResult.usage?.completionTokens, 0)
        var emptyRequest = request(2); emptyRequest.promptTokens = []
        let empty = try actual.submitWithNativeRetirement(emptyRequest)
        let emptyResult = await cbv2SchedCollect(empty.events)
        await empty.retirement.wait()
        XCTAssertEqual(emptyResult.finishReason, .error("empty prompt"))
        XCTAssertEqual(actual.stepCount, before)
        XCTAssertThrowsError(try actual.submitWithNativeRetirement(request(3, logprobs: 1)))
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertNil(actual.loopForTesting.stream(for: .init(1)))
            XCTAssertNil(actual.loopForTesting.stream(for: .init(2)))
            XCTAssertNil(actual.loopForTesting.stream(for: .init(3)))
            XCTAssertEqual(actual.loopForTesting.backend.bytesReserved, 0)
        }
        XCTAssertEqual(actual.capacity().waitingRequests, 0)
        _ = try receipt(await actual.shutdownReportingNativeCompletion(), actual)
        XCTAssertThrowsError(try actual.submitWithNativeRetirement(request(budget: 0)))
    }

    func testDuplicateIDAndOldGenerationCannotRetireOrCancelItsSuccessor() async throws {
        try lane()
        let actual = try await engine()
        pausePlanning(actual)
        let first = try actual.submitWithNativeRetirement(request())
        let original = try output(actual, 1)
        let oldGeneration = try XCTUnwrap(actual.loopForTesting.registeredStreamGeneration(for: .init(1)))
        XCTAssertThrowsError(try actual.submitWithNativeRetirement(request()))
        XCTAssertTrue(try output(actual, 1) === original)
        XCTAssertFalse(original.isEngineOwnershipReleased)
        resumePlanning(actual)
        let firstResult = await cbv2SchedCollect(first.events)
        await first.retirement.wait()
        XCTAssertEqual(firstResult.finishReason, .length)
        pausePlanning(actual)
        let second = try actual.submitWithNativeRetirement(request())
        let successor = try output(actual, 1)
        XCTAssertFalse(successor === original)
        let newGeneration = try XCTUnwrap(actual.loopForTesting.registeredStreamGeneration(for: .init(1)))
        XCTAssertNotEqual(oldGeneration, newGeneration)
        await first.retirement.wait()
        XCTAssertFalse(successor.isEngineOwnershipReleased)
        actual.loopForTesting.requestCancel(.init(1), streamGeneration: oldGeneration)
        resumePlanning(actual)
        let secondResult = await cbv2SchedCollect(second.events)
        await second.retirement.wait()
        XCTAssertEqual(secondResult.finishReason, .length)
        XCTAssertEqual(secondResult.tokens, firstResult.tokens)
        assertRetired(actual, successor)
        _ = try receipt(await actual.shutdownReportingNativeCompletion(), actual)
    }

    func testExplicitCancellationWaitsForActualInFlightRowRetirement() async throws {
        try lane()
        let actual = try await engine()
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(expectation(description: "actual submitted group held"))
        defer { gate.release() }
        actual.loopForTesting.onEngineQueueSync { tracking.afterSubmissionForTesting = { gate.holdOnce() } }
        let submitted = try actual.submitWithNativeRetirement(request())
        let registered = try output(actual, 1)
        resumePlanning(actual)
        await fulfillment(of: [gate.entered], timeout: 10)
        actual.cancel(.init(1))
        XCTAssertFalse(registered.isEngineOwnershipReleased)
        XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        gate.release()
        let result = await cbv2SchedCollect(submitted.events)
        await submitted.retirement.wait()
        XCTAssertEqual(result.finishReason, .cancelled)
        assertRetired(actual, registered)
        _ = try receipt(await actual.shutdownReportingNativeCompletion(), actual)
    }

    func testConsumerAbandonmentDoesNotAcknowledgeHeldNativeWork() async throws {
        try lane()
        let actual = try await engine()
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(expectation(description: "actual work held before consumer cancellation"))
        let consuming = expectation(description: "real AsyncStream consumer started")
        defer { gate.release() }
        actual.loopForTesting.onEngineQueueSync { tracking.afterSubmissionForTesting = { gate.holdOnce() } }
        let submitted = try actual.submitWithNativeRetirement(request())
        let registered = try output(actual, 1)
        let pump = Task {
            consuming.fulfill()
            for await _ in submitted.events {}
        }
        resumePlanning(actual)
        await fulfillment(of: [gate.entered, consuming], timeout: 10)
        pump.cancel()
        await pump.value
        XCTAssertFalse(registered.isEngineOwnershipReleased)
        gate.release()
        await submitted.retirement.wait()
        assertRetired(actual, registered)
        _ = try receipt(await actual.shutdownReportingNativeCompletion(), actual)
    }

    func testCompletedSubmissionErrorRetiresOnlyAfterItsRealGroupAndRows() async throws {
        try lane()
        let witness = try await exerciseCompletedSubmissionError()
        XCTAssertNil(witness(), "healthy cleanup must not leave an engine/hook retention cycle")
    }

    // Returning only a weak closure makes the lifetime assertion independent
    // of extended locals in this async helper's execution frame.
    private func exerciseCompletedSubmissionError() async throws -> (@Sendable () -> EngineV2?) {
        let actual = try await engine()
        defer {
            actual.loopForTesting.onEngineQueueSync {
                actual.loopForTesting.nativeSubmittedWorkFailureForTesting = nil
                actual.loopForTesting.nativeRetirementBoundaryForTesting = nil
            }
        }
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        var registered: CBv2OutputStream?
        var sawBoundary = false
        actual.loopForTesting.onEngineQueueSync {
            actual.loopForTesting.nativeSubmittedWorkFailureForTesting = { [weak actual] in
                guard let actual else { throw Failure.missingReceipt }
                actual.loopForTesting.nativeSubmittedWorkFailureForTesting = nil
                throw Failure.submittedWork // AFTER real asyncEval, never fake successful work
            }
            actual.loopForTesting.nativeRetirementBoundaryForTesting = { [weak actual] name, _ in
                guard let actual else { return XCTFail("actual engine disappeared during owned work") }
                if name == "beforeFailedRootRetirement" {
                    sawBoundary = true
                    XCTAssertFalse(registered?.isEngineOwnershipReleased ?? true)
                    XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
                    XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
                }
            }
        }
        let submitted = try actual.submitWithNativeRetirement(request())
        registered = try output(actual, 1)
        resumePlanning(actual)
        let result = await cbv2SchedCollect(submitted.events)
        await submitted.retirement.wait()
        guard case .error? = result.finishReason else {
            XCTFail("missing actual submission error")
            throw Failure.missingReceipt
        }
        XCTAssertTrue(sawBoundary)
        assertRetired(actual, try XCTUnwrap(registered))
        XCTAssertNil(tracking.outcome, "proved error cleanup is not an unknown-completion fault")
        _ = try await collect(actual, request(2)) // healthy drain, not forced success
        _ = try receipt(await actual.shutdownReportingNativeCompletion(), actual)
        return { [weak actual] in actual }
    }

    func testIncompleteShutdownLeavesActualRetirementUnresolvedAfterLateReturn() async throws {
        try lane(fault: "testIncompleteShutdownLeavesActualRetirementUnresolvedAfterLateReturn")
        let actual = try await engine(shutdownTimeout: 0)
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(expectation(description: "real work awaiting completion"))
        defer { gate.release() }
        actual.loopForTesting.onEngineQueueSync { tracking.afterSubmissionForTesting = { gate.holdOnce() } }
        let submitted = try actual.submitWithNativeRetirement(request())
        let registered = try output(actual, 1)
        resumePlanning(actual)
        await fulfillment(of: [gate.entered], timeout: 10)
        let fault = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let failure) = fault else { return XCTFail("unknown work acknowledged") }
        XCTAssertEqual(failure.reason, .shutdownTimedOut)
        _ = await cbv2SchedCollect(submitted.events)
        XCTAssertFalse(registered.isEngineOwnershipReleased)
        gate.release()
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertFalse(registered.isEngineOwnershipReleased)
            XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
            XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        }
        let again = await actual.shutdownReportingNativeCompletion()
        XCTAssertEqual(again, fault)
        // submitted.retirement.wait() intentionally cannot complete; its actual
        // OutputStream flag is the causal witness. Dedicated process exits.
    }

    func testMixedPlainMTPTimeoutCannotAcknowledgeOrReleaseThePlainRowEarly() async throws {
        try lane(fault: "testMixedPlainMTPTimeoutCannotAcknowledgeOrReleaseThePlainRowEarly")
        let actual = try await engine(mtp: true, shutdownTimeout: 0)
        XCTAssertNil(actual.mtpInactiveReason)
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(expectation(description: "plain completion before genuine MTP verify finalization"))
        defer { gate.release() }
        var plainOutput: CBv2OutputStream?
        weak var plainRow: AnyObject?
        actual.loopForTesting.onEngineQueueSync {
            tracking.afterSubmissionForTesting = {
                if plainRow == nil { plainRow = actual.loopForTesting.kvStates[.init(1)]?.compactMap { $0 }.first }
            }
            actual.loopForTesting.nativeRetirementBoundaryForTesting = { name, step in
                guard name == "beforeMTPFinalization", step?.mtpRound?.verify != nil,
                      actual.loopForTesting.nativePendingRetirementCountForTesting > 0 else { return }
                gate.holdOnce {
                    XCTAssertTrue(step?.sampledRows.contains(.init(1)) ?? false)
                    XCTAssertFalse(plainOutput?.isEngineOwnershipReleased ?? true)
                    XCTAssertNotNil(plainRow)
                    XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
                    XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
                }
            }
        }
        let plainRequest = CBv2Request(id: .init(1), promptTokens: [1, 2, 4],
            sampling: .init(temperature: 0), maxTokens: 3, stopStrings: ["never-produced-by-null-detokenizer"])
        let mtpRequest = CBv2Request(id: .init(2), promptTokens: [1, 2, 4],
            sampling: .init(temperature: 0), maxTokens: 12)
        let plain = try actual.submitWithNativeRetirement(plainRequest)
        let speculative = try actual.submitWithNativeRetirement(mtpRequest)
        plainOutput = try output(actual, 1)
        let mtpOutput = try output(actual, 2)
        resumePlanning(actual)
        await fulfillment(of: [gate.entered], timeout: 20)
        let result = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = result else { return XCTFail("mixed unknown work acknowledged") }
        XCTAssertEqual(fault.reason, .shutdownTimedOut)
        gate.release()
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertFalse(plainOutput?.isEngineOwnershipReleased ?? true)
            XCTAssertFalse(mtpOutput.isEngineOwnershipReleased)
            XCTAssertNotNil(plainRow)
            XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
            XCTAssertGreaterThan(actual.loopForTesting.nativePendingRetirementCountForTesting, 0)
            XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        }
        _ = await cbv2SchedCollect(plain.events)
        _ = await cbv2SchedCollect(speculative.events)
    }

    func testFailedGroupTimeoutRetainsRealRowsAndItsRetirementAcknowledgement() async throws {
        try lane(fault: "testFailedGroupTimeoutRetainsRealRowsAndItsRetirementAcknowledgement")
        let actual = try await engine(shutdownTimeout: 0)
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        let gate = OneShotGate(expectation(description: "failed submitted group before root retirement"))
        defer { gate.release() }
        weak var row: AnyObject?
        var registered: CBv2OutputStream?
        actual.loopForTesting.onEngineQueueSync {
            actual.loopForTesting.nativeSubmittedWorkFailureForTesting = {
                row = actual.loopForTesting.kvStates[.init(1)]?.compactMap { $0 }.first
                actual.loopForTesting.nativeSubmittedWorkFailureForTesting = nil
                throw Failure.submittedWork
            }
            actual.loopForTesting.nativeRetirementBoundaryForTesting = { name, _ in
                guard name == "beforeFailedRootRetirement" else { return }
                gate.holdOnce {
                    XCTAssertNotNil(row)
                    XCTAssertFalse(registered?.isEngineOwnershipReleased ?? true)
                    XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
                    XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
                }
            }
        }
        let submitted = try actual.submitWithNativeRetirement(request())
        registered = try output(actual, 1)
        resumePlanning(actual)
        await fulfillment(of: [gate.entered], timeout: 10)
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else { return XCTFail("failed group acknowledged early") }
        XCTAssertEqual(fault.reason, .shutdownTimedOut)
        gate.release()
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertNotNil(row)
            XCTAssertFalse(registered?.isEngineOwnershipReleased ?? true)
            XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
            XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        }
        _ = await cbv2SchedCollect(submitted.events)
    }

    func testOneShotRequiredAssistantFenceRefusalCannotBeRehabilitatedByCleanupFence() async throws {
        try lane(fault: "testOneShotRequiredAssistantFenceRefusalCannotBeRehabilitatedByCleanupFence")
        let actual = try await engine(mtp: true)
        XCTAssertNil(actual.mtpInactiveReason)
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        var requiredAttempts = 0, cleanupAttempts = 0
        weak var state: MiMoV26MTPState?
        actual.loopForTesting.onEngineQueueSync {
            tracking.beforeFenceForTesting = { _ in cleanupAttempts += 1 } // allows real fence, if incorrectly retried
            actual.loopForTesting.nativeRequiredAssistantFenceForTesting = { value in
                requiredAttempts += 1
                state = value as? MiMoV26MTPState
                if requiredAttempts == 1 { throw Failure.fence }
            }
        }
        let submitted = try actual.submitWithNativeRetirement(request())
        let registered = try output(actual, 1)
        resumePlanning(actual)
        let terminal = await cbv2SchedCollect(submitted.events)
        guard case .error? = terminal.finishReason else { return XCTFail("missing failed completion terminal") }
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else { return XCTFail("required fence failure was rehabilitated") }
        XCTAssertEqual(fault.reason, .capturedFenceFailed)
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertEqual(requiredAttempts, 1)
            XCTAssertEqual(cleanupAttempts, 0, "later successful fences cannot replace original required proof")
            XCTAssertNotNil(state)
            XCTAssertFalse(state?.isReleased ?? true)
            XCTAssertFalse(registered.isEngineOwnershipReleased)
            XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
            XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        }
        let repeated = await actual.shutdownReportingNativeCompletion()
        XCTAssertEqual(outcome, repeated)
    }

    func testSubmissionCleanupFenceFailureTerminatesConsumerWithoutAcknowledging() async throws {
        try lane(fault: "testSubmissionCleanupFenceFailureTerminatesConsumerWithoutAcknowledging")
        let actual = try await engine()
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        actual.loopForTesting.onEngineQueueSync {
            actual.loopForTesting.nativeSubmittedWorkFailureForTesting = {
                actual.loopForTesting.nativeSubmittedWorkFailureForTesting = nil
                throw Failure.submittedWork
            }
            tracking.beforeFenceForTesting = { _ in throw Failure.fence }
        }
        let submitted = try actual.submitWithNativeRetirement(request())
        let registered = try output(actual, 1)
        resumePlanning(actual)
        // No shutdown request yet: sealing a completion fault itself must
        // finish the consumer, while keeping its ownership acknowledgement.
        let result = await cbv2SchedCollect(submitted.events)
        guard case .error? = result.finishReason else { return XCTFail("sealed cleanup fault stranded the consumer") }
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else { return XCTFail("failed cleanup acknowledged") }
        XCTAssertEqual(fault.reason, .capturedFenceFailed)
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertFalse(registered.isEngineOwnershipReleased)
            XCTAssertGreaterThan(actual.loopForTesting.backend.bytesReserved, 0)
            XCTAssertGreaterThan(tracking.debugRetainedRootCount, 0)
        }
    }

    func testRealTextDetokenizerStopStringPreservesTokenAndSuppressionSemantics() async throws {
        try lane()
        let actual = try await engine(text: true)
        let first = try actual.submitWithNativeRetirement(request(budget: 1))
        let reference = await cbv2SchedCollect(first.events)
        await first.retirement.wait()
        let token = try XCTUnwrap(reference.tokens.first)
        XCTAssertEqual(reference.finishReason, .length)
        XCTAssertEqual(reference.text, String(token))
        var stoppedRequest = request(2, budget: 4)
        stoppedRequest.stopStrings = [String(token)]
        let stopped = try actual.submitWithNativeRetirement(stoppedRequest)
        let result = await cbv2SchedCollect(stopped.events)
        await stopped.retirement.wait()
        XCTAssertEqual(result.finishReason, .stop)
        XCTAssertEqual(result.text, "")
        XCTAssertEqual(result.tokens, [token])
        XCTAssertEqual(result.usage?.completionTokens, 1)
        _ = try receipt(await actual.shutdownReportingNativeCompletion(), actual)
    }

    func testHeldRealTextStopStringPushDoesNotBlockFaultArbitration() async throws {
        try lane(fault: "testHeldRealTextStopStringPushDoesNotBlockFaultArbitration")
        try await heldTextDecoder(targetCall: 1)
    }

    func testHeldRealTextStopStringFlushRetainsLoanAndReachableStream() async throws {
        try lane(fault: "testHeldRealTextStopStringFlushRetainsLoanAndReachableStream")
        try await heldTextDecoder(targetCall: 2)
    }

    private func heldTextDecoder(targetCall: Int) async throws {
        let gate = OneShotGate(expectation(description: targetCall == 1 ? "real SDK text push held" : "real SDK text flush held"))
        let decoder = DecodeGate(gate, targetCall: targetCall)
        defer { gate.release() }
        let actual = try await engine(text: true, shutdownTimeout: 0, decoderGate: decoder)
        pausePlanning(actual)
        let tracking = try XCTUnwrap(actual.loopForTesting.nativeShutdownState)
        var input = request(budget: 1)
        input.stopStrings = ["not-a-decimal-token"]
        let submitted = try actual.submitWithNativeRetirement(input)
        let registered = try output(actual, 1)
        resumePlanning(actual)
        await fulfillment(of: [gate.entered], timeout: 10)
        XCTAssertTrue(tracking.hasLoans, "the actual decoder must remain a counted owner")
        XCTAssertFalse(registered.isEngineOwnershipReleased)
        // Decoding is host-only, but must not hold the first-winner lock.
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else { return XCTFail("held decoder minted completion") }
        XCTAssertEqual(fault.reason, .shutdownTimedOut)
        let terminal = await cbv2SchedCollect(submitted.events)
        guard case .error? = terminal.finishReason else { return XCTFail("watchdog lost the pending terminal stream") }
        gate.release()
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertFalse(registered.isEngineOwnershipReleased)
            XCTAssertTrue(tracking.hasLoans, "late decoder return cannot drop its retained-fault owner")
        }
        let repeated = await actual.shutdownReportingNativeCompletion()
        XCTAssertEqual(outcome, repeated)
    }

    func testFaultBeforeDeadlineOperationRegistrationResolvesColdSubmitter() async throws {
        try lane(fault: "testFaultBeforeDeadlineOperationRegistrationResolvesColdSubmitter")
        let actual = try await engine(shutdownTimeout: 0, serializedPrefill: true)
        // Keep legacy _healthy true: no engine step may overwrite it after the
        // fault. The new native arbitration, not the old health guard, must win.
        pausePlanning(actual)
        let gate = OneShotGate(expectation(description: "real stream before deadline operation registration"))
        defer {
            gate.release()
            actual.loopForTesting.setDeadlineAdmissionBeforeRegistrationHookForTesting(nil)
        }
        actual.loopForTesting.setDeadlineAdmissionBeforeRegistrationHookForTesting { _ in gate.holdOnce() }
        let input = request(budget: 1)
        let admission = CBv2FirstTokenDeadlineAdmission(deadline: .now.advanced(by: .seconds(60)),
            conservativePrefillTokensPerSecond: 1000, conservativeDecodeTokensPerSecond: 1000)
        let completion = SubmissionCompletion()
        let completed = expectation(description: "cold refused submission actually returned")
        let submission = Task {
            defer { completion.markFinished(); completed.fulfill() }
            return try await actual.submit(input, firstTokenDeadline: admission)
        }
        await fulfillment(of: [gate.entered], timeout: 10)
        let registered = try XCTUnwrap(actual.loopForTesting.stream(for: input.id))
        XCTAssertFalse(registered.isEngineOwnershipReleased)
        XCTAssertEqual(actual.loopForTesting.deadlineAdmissionRegistrationSnapshotForTesting.operationCount, 0)
        XCTAssertTrue(actual.loopForTesting.gauges.hasPendingSubmissions)
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else { return XCTFail("pending cold registration minted receipt") }
        XCTAssertEqual(fault.reason, .shutdownTimedOut)
        XCTAssertTrue(actual.loopForTesting.deadlineAdmissionRegistrationSnapshotForTesting.legacyHealthy,
                      "this must discriminate the native outcome from the unchanged legacy health flag")
        gate.release()
        await fulfillment(of: [completed], timeout: 10)
        guard completion.isFinished else {
            submission.cancel()
            return XCTFail("fault-before-registration stranded the real admission waiter")
        }
        do { _ = try await submission.value; XCTFail("cold faulted registration accepted") }
        catch { XCTAssertTrue(error is CBv2KVError) }
        XCTAssertTrue(registered.isEngineOwnershipReleased, "only this never-enqueued generation can retire cold")
        XCTAssertNil(actual.loopForTesting.stream(for: input.id))
        XCTAssertNil(actual.loopForTesting.registeredStreamGeneration(for: input.id))
        XCTAssertEqual(actual.loopForTesting.deadlineAdmissionRegistrationSnapshotForTesting.operationCount, 0)
        XCTAssertFalse(actual.loopForTesting.gauges.hasPendingSubmissions)
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertFalse(actual.loopForTesting.scheduler.hasWork)
            XCTAssertEqual(actual.loopForTesting.backend.bytesReserved, 0)
            XCTAssertEqual(actual.loopForTesting.nativeShutdownState?.debugRetainedRootCount, 0)
        }
        let repeated = await actual.shutdownReportingNativeCompletion()
        XCTAssertEqual(repeated, outcome, "cold unwind must not rehabilitate the engine fault")
    }

    func testHeldInitialDeadlineHookAllowsFaultAndColdRefusalWithoutNativeWork() async throws {
        try lane(fault: "testHeldInitialDeadlineHookAllowsFaultAndColdRefusalWithoutNativeWork")
        let actual = try await engine(shutdownTimeout: 0, serializedPrefill: true)
        let gate = OneShotGate(expectation(description: "real deadline initial guard"))
        defer { gate.release() }
        actual.loopForTesting.setDeadlineAdmissionInitialGuardHookForTesting { _ in gate.holdOnce() }
        let input = request(budget: 1)
        let admission = CBv2FirstTokenDeadlineAdmission(deadline: .now.advanced(by: .seconds(60)),
            conservativePrefillTokensPerSecond: 1000, conservativeDecodeTokensPerSecond: 1000)
        let submission = Task { try await actual.submit(input, firstTokenDeadline: admission) }
        await fulfillment(of: [gate.entered], timeout: 10)
        // Converse ordering: registration won before fault, so its real waiter
        // must be visible to the existing external-failure snapshot.
        XCTAssertEqual(actual.loopForTesting.deadlineAdmissionRegistrationSnapshotForTesting.operationCount, 1)
        XCTAssertTrue(actual.loopForTesting.gauges.hasPendingSubmissions)
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else { return XCTFail("held admission minted receipt") }
        XCTAssertEqual(fault.reason, .shutdownTimedOut)
        gate.release()
        do { _ = try await submission.value; XCTFail("uncommitted failed admission accepted") }
        catch { XCTAssertTrue(error is CBv2KVError) }
        XCTAssertFalse(actual.loopForTesting.gauges.hasPendingSubmissions)
        actual.loopForTesting.onEngineQueueSync {
            XCTAssertFalse(actual.loopForTesting.scheduler.hasWork)
            XCTAssertEqual(actual.loopForTesting.backend.bytesReserved, 0)
            XCTAssertEqual(actual.loopForTesting.nativeShutdownState?.debugRetainedRootCount, 0)
        }
    }

    func testHeldCommittedDeadlineHookTransfersRealRetirementAfterFault() async throws {
        try lane(fault: "testHeldCommittedDeadlineHookTransfersRealRetirementAfterFault")
        let actual = try await engine(shutdownTimeout: 0, serializedPrefill: true)
        let gate = OneShotGate(expectation(description: "real deadline commit held"))
        defer { gate.release() }
        actual.loopForTesting.setDeadlineAdmissionCommittedHookForTesting { _ in gate.holdOnce() }
        let input = request(budget: 1)
        let admission = CBv2FirstTokenDeadlineAdmission(deadline: .now.advanced(by: .seconds(60)),
            conservativePrefillTokensPerSecond: 1000, conservativeDecodeTokensPerSecond: 1000)
        let submission = Task { try await actual.submit(input, firstTokenDeadline: admission) }
        await fulfillment(of: [gate.entered], timeout: 10)
        let registered = try XCTUnwrap(actual.loopForTesting.stream(for: input.id)) // locked metadata; queue is held
        let outcome = await actual.shutdownReportingNativeCompletion()
        guard case .incomplete(let fault) = outcome else { return XCTFail("held committed admission minted receipt") }
        XCTAssertEqual(fault.reason, .shutdownTimedOut)
        gate.release()
        guard case .admitted(let events, _, _, let retirement) = try await submission.value else {
            return XCTFail("post-commit fault was converted to a cold refusal")
        }
        _ = await cbv2SchedCollect(events)
        actual.loopForTesting.onEngineQueueSync { XCTAssertFalse(registered.isEngineOwnershipReleased) }
        withExtendedLifetime(retirement) {} // never fabricate acknowledgement of this retained generation
    }
}
