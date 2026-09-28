import Foundation
import MLX
import MLXLLM
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

/// Ownership tests execute real tiny filesystem/model/probe work. The stateless
/// tokenizer below controls preparation only; it is NOT tokenizer/Jinja parity.
final class MiMoV26FailedConstructionTests: XCTestCase {
    private enum Failure: Error { case boundary, fence, unexpectedlyExecuted }
    private final class Permit: MiMoV26SerialLoadReservation, Sendable {
        let request: MiMoV26SerialLoadRequest
        let reservedLoadBytes: UInt64
        init(_ request: MiMoV26SerialLoadRequest) {
            self.request = request; reservedLoadBytes = request.requiredLoadBytes
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private final class ControlTokenizer: Tokenizer, Sendable {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [6] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.map(String.init).joined() }
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
    private func nativeLane(faultCase: String? = nil) throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1" else {
            throw XCTSkip("Requires a root-authorized exclusive native failure-ownership test process")
        }
        // Every retained-fault selector gets its OWN process and exact selector.
        // Never run another native cell after an unknown-completion outcome.
        guard environment["MIMO_V26_CONSTRUCTION_FAULT_CASE"] == faultCase else {
            throw XCTSkip("Run this retained-fault selector alone with its exact MIMO_V26_CONSTRUCTION_FAULT_CASE name")
        }
    }
    private func preserveFault(_ work: NativeConstructionScope) {
        // Restart-only faults deliberately survive this dedicated test process.
        // This is NOT an in-process recovery or a production global registry.
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    private func preserveFault(_ work: NativeConstructionWork) {
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    private func fixture() throws -> (URL, MiMoV26FilesystemLoadPlan, MiMoV26SerialLoadSession) {
        let path = try XCTUnwrap(ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_LOAD_FIXTURES"])
        let fixtures = URL(fileURLWithPath: path)
        let root = fixtures.appendingPathComponent("construction-work-" + UUID().uuidString)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("tiny-bf16"), to: root)
        try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["eos_token": "<|im_end|>"])
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data("{{ messages }}{% if enable_thinking is false %}<think></think>{% endif %}".utf8)
            .write(to: root.appendingPathComponent("chat_template.jinja"))
        let p = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: fixtures.appendingPathComponent("provenance.json"))) as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]), sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304))
        return (root, plan, try MiMoV26SerialLoadSession(plan: plan))
    }
    private func prepare(_ root: URL, _ session: MiMoV26SerialLoadSession) async throws -> MiMoV26ModelFactory.Prepared {
        try await MiMoV26ModelFactory.prepare(request: session.request,
            configuration: .init(directory: root), tokenizerLoader: Loader())
    }
    private func receipt(_ snapshot: NativeConstructionSnapshot) throws -> NativeConstructionReceipt {
        guard case .completed(let receipt) = snapshot.disposition else {
            XCTFail("missing completed epoch: \(snapshot)"); throw Failure.boundary
        }
        return receipt
    }

    func testMetadataFailureHasNoSubmissionReceiptAndConsumesSession() throws {
        try nativeLane()
        let (_, plan, session) = try fixture()
        let other = try MiMoV26SerialLoadSession(plan: plan)
        let work = NativeConstructionScope()
        defer { preserveFault(work) }
        XCTAssertThrowsError(try session.load(reservation: Permit(other.request), retaining: work)) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .reservationMismatch)
        }
        let first = try receipt(work.snapshot)
        XCTAssertEqual(first.completion, .noNativeSubmission)
        XCTAssertEqual(work.snapshot.retainedArrayCount, 0)
        XCTAssertEqual(work.snapshot.retainedOwnerCount, 0)
        XCTAssertEqual(work.snapshot.capturedStreamCount, 0)
        XCTAssertThrowsError(try session.load(reservation: Permit(session.request), retaining: work)) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .alreadyConsumed)
        }
        XCTAssertThrowsError(try work.validate(first))
        XCTAssertGreaterThan(try receipt(work.snapshot).epoch, first.epoch)
    }

    func testCPUFailureRetainsActualSourceAndPermit() throws {
        try nativeLane(faultCase: "testCPUFailureRetainsActualSourceAndPermit")
        try checkSourceFailure(refuseCPU: true)
    }

    func testDefaultFailureRetainsActualSourceAndPermit() throws {
        try nativeLane(faultCase: "testDefaultFailureRetainsActualSourceAndPermit")
        try checkSourceFailure(refuseCPU: false)
    }

    private func checkSourceFailure(refuseCPU: Bool) throws {
        let (_, _, session) = try fixture()
        var permit: Permit? = Permit(session.request)
        weak var permitWitness = permit
        weak var sourceWitness: MLXArray?
        let work = NativeConstructionScope()
        defer { preserveFault(work) }
        var attempted: [MLX.Stream] = []
        try MLX.Stream.withNewDefaultStream(device: .gpu) {
            let cpu = StreamOrDevice.cpu.stream, gpu = StreamOrDevice.default.stream
            XCTAssertNotEqual(cpu, gpu)
            work.testingBoundary = { name, scope in
                let boundary = refuseCPU ? "serial.sourceMaterialized" : "serial.parameterMaterialized"
                if name == boundary {
                    sourceWitness = scope.retainedArraysForTesting.first {
                        (try? $0.evaluatedBufferInfo()) != nil
                    }
                    throw Failure.boundary
                }
            }
            work.testingBeforeFence = { stream in
                attempted.append(stream)
                if stream == (refuseCPU ? cpu : gpu) { throw Failure.fence }
            }
            XCTAssertThrowsError(try session.load(reservation: XCTUnwrap(permit), retaining: work))
            XCTAssertTrue(attempted.contains(cpu)); XCTAssertTrue(attempted.contains(gpu))
            XCTAssertEqual(attempted.count, 2, "one failed stream must not suppress the other drain attempt")
        }
        permit = nil
        XCTAssertTrue(work.snapshot.isRetainedFault)
        XCTAssertNotNil(permitWitness); XCTAssertNotNil(sourceWitness)
        let source = try XCTUnwrap(sourceWitness)
        XCTAssertTrue(work.retainedArraysForTesting.contains { $0 === source })
        XCTAssertGreaterThan(work.snapshot.retainedOwnerCount, 3)
        XCTAssertEqual(work.snapshot.capturedStreamCount, 2)
        let retry = try MiMoV26SerialLoadSession(plan: fixture().1)
        XCTAssertThrowsError(try retry.load(reservation: Permit(retry.request), retaining: work)) {
            guard let error = $0 as? NativeConstructionError, case .retainedFault = error else {
                return XCTFail("fault reset: \($0)")
            }
        }
    }

    func testFilesystemSourceIdentityAndSuccessfulFailureRetirement() throws {
        try nativeLane()
        let (_, plan, _) = try fixture(), work = NativeConstructionScope()
        defer { preserveFault(work) }
        weak var source: MLXArray?
        work.testingBoundary = { name, scope in
            if name == "filesystem.returnedSourceHandles", source == nil {
                source = scope.retainedArraysForTesting.first
            }
        }
        var loaded = try MiMoV26FilesystemWeights.load(plan: plan, retaining: work)
        let handles = loaded.takeMaterializationHandles()
        let original = try XCTUnwrap(source)
        XCTAssertTrue(handles.contains { $0.array === original }, "no copied source tree at handoff")
        XCTAssertEqual(try receipt(work.snapshot).completion, .noNativeSubmission)
        XCTAssertEqual(work.snapshot.retainedArrayCount, 0)
        XCTAssertEqual(work.snapshot.retainedOwnerCount, 0)
        let session = try MiMoV26SerialLoadSession(plan: plan)
        var permit: Permit? = Permit(session.request)
        weak var permitWitness = permit
        work.testingBoundary = { name, _ in
            if name == "serial.sourceMaterialized" { throw Failure.boundary }
        }
        XCTAssertThrowsError(try session.load(reservation: XCTUnwrap(permit), retaining: work)) {
            XCTAssertTrue($0 is Failure)
        }
        permit = nil
        XCTAssertNil(permitWitness, "proved completion may drop the SDK's extra permit alias")
        XCTAssertEqual(try receipt(work.snapshot).completion, .capturedStreamsCompleted)
        XCTAssertEqual(work.snapshot.retainedArrayCount, 0)
        XCTAssertEqual(work.snapshot.retainedOwnerCount, 0)
    }

    func testFactoryLateCancellationFailedFenceRetainsCompleteBundle() async throws {
        try nativeLane(faultCase: "testFactoryLateCancellationFailedFenceRetainsCompleteBundle")
        let (root, _, session) = try fixture(), prepared = try await prepare(root, session)
        var permit: Permit? = Permit(session.request)
        weak var permitWitness = permit
        weak var vision: MiMoV26VisionTower?
        let work = NativeConstructionScope()
        defer { preserveFault(work) }
        var cancelled = false
        work.testingBoundary = { name, scope in
            if name == "factory.beforePublication" {
                vision = scope.retainedOwnersForTesting.compactMap { $0 as? MiMoV26VisionTower }.first
                cancelled = true
            }
        }
        work.testingBeforeFence = { _ in throw Failure.fence }
        XCTAssertThrowsError(try MiMoV26ModelFactory.load(session: session, reservation: XCTUnwrap(permit),
            prepared: prepared, retaining: work, isCancelled: { cancelled }))
        permit = nil
        XCTAssertTrue(cancelled); XCTAssertTrue(work.snapshot.isRetainedFault)
        XCTAssertNotNil(vision); XCTAssertNotNil(permitWitness)
        XCTAssertGreaterThan(work.snapshot.retainedArrayCount, 0)
        if case .retainedFault(let fault) = work.snapshot.disposition {
            XCTAssertTrue(fault.cause.contains("cancelled"))
            XCTAssertFalse(fault.completionFailures.isEmpty)
        } else { XCTFail("late cancelled factory lost its complete native owner") }
    }

    func testFactoryLateCancellationWithSuccessfulDrainDropsTemporaryOwners() async throws {
        try nativeLane()
        let (root, _, session) = try fixture(), prepared = try await prepare(root, session)
        var permit: Permit? = Permit(session.request)
        weak var permitWitness = permit
        weak var vision: MiMoV26VisionTower?
        let work = NativeConstructionScope()
        defer { preserveFault(work) }
        var cancelled = false
        work.testingBoundary = { name, scope in
            if name == "factory.beforePublication" {
                vision = scope.retainedOwnersForTesting.compactMap { $0 as? MiMoV26VisionTower }.first
                cancelled = true
            }
        }
        XCTAssertThrowsError(try MiMoV26ModelFactory.load(session: session, reservation: XCTUnwrap(permit),
            prepared: prepared, retaining: work, isCancelled: { cancelled })) {
                XCTAssertEqual($0 as? MiMoV26SerialLoadError, .cancelled)
            }
        permit = nil
        XCTAssertTrue(cancelled); XCTAssertNil(vision); XCTAssertNil(permitWitness)
        XCTAssertEqual(try receipt(work.snapshot).completion, .capturedStreamsCompleted)
        XCTAssertEqual(work.snapshot.retainedArrayCount, 0)
        XCTAssertEqual(work.snapshot.retainedOwnerCount, 0)
    }

    func testManagedPreReturnFailureKeepsPreinstalledActualOwner() async throws {
        try nativeLane(faultCase: "testManagedPreReturnFailureKeepsPreinstalledActualOwner")
        let (root, _, session) = try fixture(), prepared = try await prepare(root, session)
        let work = NativeConstructionWork() // installed before invoking the factory
        defer { preserveFault(work) }
        var permit: Permit? = Permit(session.request)
        weak var witness = permit
        try await work.configureFailureForTesting { scope in
            scope.testingBoundary = { name, _ in
                if name == "serial.sourceMaterialized" { throw Failure.boundary }
            }
            scope.testingBeforeFence = { _ in throw Failure.fence }
        }
        do {
            _ = try await MiMoV26ModelFactory.loadContainer(session: session,
                reservation: XCTUnwrap(permit), prepared: prepared, retaining: work)
            XCTFail("failed construction returned a model")
        } catch {
            guard let error = error as? NativeConstructionError, case .retainedFault = error else {
                return XCTFail("managed failure lost its completion error: \(error)")
            }
        }
        permit = nil
        XCTAssertNotNil(witness, "the exact permit survives even though no ModelContainer was returned")
        XCTAssertTrue(work.snapshot.isRetainedFault)
        XCTAssertGreaterThan(work.snapshot.retainedArrayCount, 0)
        XCTAssertGreaterThan(work.snapshot.retainedOwnerCount, 3)
        XCTAssertGreaterThanOrEqual(work.snapshot.capturedStreamCount, 2)
    }

    func testManagedAdoptionGapRetainsIdenticalContainerAndRejectsForeignOwner() async throws {
        try nativeLane()
        let (root, plan, session) = try fixture(), prepared = try await prepare(root, session)
        let work = NativeConstructionWork()
        defer { preserveFault(work) }
        var permit: Permit? = Permit(session.request)
        weak var permitWitness = permit
        var container: ModelContainer? = try await MiMoV26ModelFactory.loadContainer(
            session: session, reservation: XCTUnwrap(permit), prepared: prepared, retaining: work)
        weak var witness = container
        container = nil; permit = nil
        XCTAssertNotNil(witness); XCTAssertNotNil(permitWitness, "ready-unclaimed owner must precede host await/veto")
        let otherSession = try MiMoV26SerialLoadSession(plan: plan)
        let otherPrepared = try await prepare(root, otherSession), otherWork = NativeConstructionWork()
        defer { preserveFault(otherWork) }
        let other = try await MiMoV26ModelFactory.loadContainer(session: otherSession,
            reservation: Permit(otherSession.request), prepared: otherPrepared, retaining: otherWork)
        do { try await work.acknowledgeContainerAdoption(other); XCTFail("foreign container adopted") }
        catch { XCTAssertEqual(error as? NativeConstructionError, .invalidContainerAdoption) }
        XCTAssertNotNil(witness)
        var adopted: ModelContainer? = try XCTUnwrap(witness)
        try await work.acknowledgeContainerAdoption(XCTUnwrap(adopted))
        do { try await work.acknowledgeContainerAdoption(XCTUnwrap(adopted)); XCTFail("duplicate adoption") }
        catch { XCTAssertEqual(error as? NativeConstructionError, .invalidContainerAdoption) }
        adopted = nil
        XCTAssertNil(witness); XCTAssertNil(permitWitness, "successful handoff must not create a facade/owner cycle")
        let releasedContainerReceipt = try receipt(work.snapshot)
        do {
            try await work.sealForPublication(releasedContainerReceipt)
            XCTFail("a released adopted container cannot authorize publication")
        } catch {
            XCTAssertEqual(error as? NativeConstructionError, .invalidContainerAdoption)
        }
        try await otherWork.acknowledgeContainerAdoption(other)
    }

    func testManagedSetupAdvancesEpochAndSealRejectsStaleReceipts() async throws {
        try nativeLane()
        let (root, _, session) = try fixture(), prepared = try await prepare(root, session)
        let work = NativeConstructionWork()
        defer { preserveFault(work) }
        let container = try await MiMoV26ModelFactory.loadContainer(session: session,
            reservation: Permit(session.request), prepared: prepared, retaining: work)
        let loadReceipt = try receipt(work.snapshot)
        do {
            _ = try await MiMoV26ModelFactory.withNativeConstruction(container: container, retaining: work) { _, _ -> Int in
                throw Failure.unexpectedlyExecuted
            }
            XCTFail("unadopted setup accepted")
        } catch { XCTAssertEqual(error as? NativeConstructionError, .invalidContainerAdoption) }
        try await work.acknowledgeContainerAdoption(container)
        let sessionID = try await MiMoV26ModelFactory.withNativeConstruction(container: container, retaining: work) { model, scope in
            let binding = try model.makeCBv2Binding()
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let unowned = try MiMoV26CBv2Adapter(target: binding.adapter.target)
            XCTAssertThrowsError(try unowned.makeNativeExecutionResources(bytesCapacity: 1 << 20, retaining: scope)) {
                XCTAssertEqual($0 as? NativeConstructionError, .unqualifiedNativeOwner)
            }
            let resources = try binding.adapter.makeNativeExecutionResources(bytesCapacity: 1 << 20, retaining: scope)
            XCTAssertEqual(resources.contract.constructionOwnerID, scope.snapshot.ownerID)
            XCTAssertEqual(resources.contract.constructionEpoch, scope.snapshot.epoch)
            XCTAssertEqual(resources.contract.constructionOwnerID, work.snapshot.ownerID)
            XCTAssertTrue(scope.retainedOwnersForTesting.contains { $0 === resources.backend })
            XCTAssertTrue(scope.retainedOwnersForTesting.contains { $0 === resources.cacheProvider })
            return model.loadReceipt.sessionID
        }
        XCTAssertEqual(sessionID, session.request.sessionID)
        let setupReceipt = try receipt(work.snapshot)
        XCTAssertGreaterThan(setupReceipt.epoch, loadReceipt.epoch)
        XCTAssertThrowsError(try work.validate(loadReceipt))
        XCTAssertNoThrow(try work.validate(setupReceipt))
        try await work.sealForPublication(setupReceipt)
        do {
            _ = try await MiMoV26ModelFactory.withNativeConstruction(container: container, retaining: work) { _, _ -> Int in
                throw Failure.unexpectedlyExecuted
            }
            XCTFail("sealed owner reopened")
        } catch { XCTAssertEqual(error as? NativeConstructionError, .invalidContainerAdoption) }
    }

    func testUnstartedManagedFailureAtomicallyRefusesLaterLoad() async throws {
        try nativeLane()
        let (root, _, session) = try fixture(), prepared = try await prepare(root, session)
        let work = NativeConstructionWork()
        let noWork = try await work.finishUnstartedConstruction()
        XCTAssertEqual(noWork.completion, .noNativeSubmission)
        XCTAssertNoThrow(try work.validate(noWork))
        XCTAssertEqual(work.snapshot.capturedStreamCount, 0)
        do {
            _ = try await MiMoV26ModelFactory.loadContainer(session: session,
                reservation: Permit(session.request), prepared: prepared, retaining: work)
            XCTFail("closed metadata failure started native work")
        } catch { XCTAssertEqual(error as? NativeConstructionError, .alreadyConsumed) }
        do { _ = try await work.finishUnstartedConstruction(); XCTFail("duplicate completion") }
        catch { XCTAssertEqual(error as? NativeConstructionError, .alreadyConsumed) }
        XCTAssertEqual(try receipt(work.snapshot), noWork)
    }

    func testProbeFailedFenceRetainsRowsOutputAndPermanentlyDisablesAdapter() throws {
        try nativeLane(faultCase: "testProbeFailedFenceRetainsRowsOutputAndPermanentlyDisablesAdapter")
        let (target, _) = try MiMoV26MTPChecks.fixture()
        let adapter = try MiMoV26CBv2Adapter(target: target), work = NativeConstructionScope()
        defer { preserveFault(work) }
        weak var row: AnyObject?
        weak var output: MLXArray?
        work.testingBoundary = { name, scope in
            if name == "probe.evaluated.decode" {
                row = scope.retainedOwnersForTesting.first { $0 is any CBv2SequenceKV }
                output = scope.retainedArraysForTesting.first {
                    $0.ndim == 3 && $0.shape.last == target.configuration.vocabularySize
                }
                throw Failure.boundary
            }
        }
        work.testingBeforeFence = { _ in throw Failure.fence }
        XCTAssertThrowsError(try adapter.probeNativeKVTypes(retaining: work))
        XCTAssertTrue(work.snapshot.isRetainedFault)
        XCTAssertNotNil(row); XCTAssertNotNil(output)
        XCTAssertEqual((row as? any CBv2SequenceKV)?.absoluteOffset, 3)
        XCTAssertGreaterThan(work.snapshot.retainedArrayCount, 0)
        XCTAssertThrowsError(try adapter.probeNativeKVTypes(retaining: NativeConstructionScope()))
        XCTAssertThrowsError(try adapter.makeBackend(bytesCapacity: 1 << 20))
    }

    func testSuccessfulProbeKeepsRowsUntilFenceThenUnbinds() throws {
        try nativeLane()
        let (target, _) = try MiMoV26MTPChecks.fixture()
        let adapter = try MiMoV26CBv2Adapter(target: target), work = NativeConstructionScope()
        defer { preserveFault(work) }
        weak var row: AnyObject?
        var fenced = 0
        work.testingBoundary = { name, scope in
            if name == "probe.evaluated.decode" {
                row = scope.retainedOwnersForTesting.first { $0 is any CBv2SequenceKV }
            }
        }
        work.testingBeforeFence = { _ in
            XCTAssertNotNil(row)
            XCTAssertEqual((row as? any CBv2SequenceKV)?.absoluteOffset, 3)
            fenced += 1
        }
        let result = try adapter.probeNativeKVTypes(retaining: work)
        XCTAssertFalse(result.observations.isEmpty); XCTAssertGreaterThan(fenced, 0)
        XCTAssertTrue(adapter.target === target)
        XCTAssertNil(row)
        XCTAssertEqual(work.snapshot.retainedArrayCount, 0)
        XCTAssertEqual(work.snapshot.retainedOwnerCount, 0)
        XCTAssertEqual(try receipt(work.snapshot).completion, .capturedStreamsCompleted)
    }

    func testMoreThanEightActualStreamsAreAttemptedAndRetainedOnPartialFailure() throws {
        try nativeLane(faultCase: "testMoreThanEightActualStreamsAreAttemptedAndRetainedOnPartialFailure")
        let work = NativeConstructionScope()
        defer { preserveFault(work) }
        var captured: [MLX.Stream] = [], attempted: [MLX.Stream] = []
        work.testingBeforeFence = { stream in
            attempted.append(stream)
            if stream == captured[4] { throw Failure.fence }
        }
        XCTAssertThrowsError(try work.withPhase(.nativeSetup) {
            for index in 0..<9 {
                try MLX.Stream.withNewDefaultStream(device: .gpu) {
                    let stream = StreamOrDevice.default.stream
                    captured.append(stream)
                    try work.capture(stream)
                    let output = MLXArray([Int32(index)]) + Int32(1)
                    try work.retain(output); try work.willSubmit()
                    eval(output)
                }
            }
        })
        XCTAssertEqual(captured.count, 9); XCTAssertEqual(attempted.count, 9)
        XCTAssertTrue(captured.allSatisfy { attempted.contains($0) })
        XCTAssertEqual(work.snapshot.capturedStreamCount, 9)
        XCTAssertEqual(work.snapshot.retainedArrayCount, 9)
        XCTAssertTrue(work.snapshot.isRetainedFault)
    }

    func testExecutionResourceOwnersSurviveLateVetoAndRequireActiveScope() async throws {
        try nativeLane(faultCase: "testExecutionResourceOwnersSurviveLateVetoAndRequireActiveScope")
        let (root, _, session) = try fixture(), prepared = try await prepare(root, session)
        let loading = NativeConstructionScope()
        defer { preserveFault(loading) }
        let context = try MiMoV26ModelFactory.load(session: session, reservation: Permit(session.request),
            prepared: prepared, retaining: loading)
        let model = try XCTUnwrap(context.model as? MiMoV26LoadedModel)
        let binding = try model.makeCBv2Binding(), adapter = binding.adapter
        let work = NativeConstructionScope()
        defer { preserveFault(work) }
        XCTAssertThrowsError(try adapter.makeNativeExecutionResources(bytesCapacity: 1 << 20, retaining: work)) {
            XCTAssertEqual($0 as? NativeConstructionError, .inactiveScope)
        }
        weak var backend: MiMoV26CBv2Backend?
        weak var bank: CBv2LayerCacheBank?
        var created = false
        work.testingBoundary = { name, scope in
            if name == "adapter.nativeExecutionResources" {
                backend = scope.retainedOwnersForTesting.compactMap { $0 as? MiMoV26CBv2Backend }.first
                bank = scope.retainedOwnersForTesting.compactMap { $0 as? CBv2LayerCacheBank }.first
                created = true
                throw Failure.boundary
            }
        }
        work.testingBeforeFence = { _ in if created { throw Failure.fence } }
        XCTAssertThrowsError(try work.withPhase(.nativeSetup) {
            // The same package-only authorization used by the managed factory;
            // the owner here came from the actual strict filesystem/serial load.
            try work.authorizeImmutableLoadedOwner(model.resources)
            _ = try adapter.probeNativeKVTypes(retaining: work)
            _ = try adapter.makeNativeExecutionResources(bytesCapacity: 1 << 20, retaining: work)
        })
        XCTAssertTrue(created); XCTAssertTrue(work.snapshot.isRetainedFault)
        XCTAssertNotNil(backend); XCTAssertNotNil(bank)
        XCTAssertTrue(work.retainedOwnersForTesting.contains { $0 === adapter })
        XCTAssertThrowsError(try adapter.probeNativeKVTypes(retaining: NativeConstructionScope()),
                             "an outer setup fence must invalidate the already-completed nested probe")
    }
}
