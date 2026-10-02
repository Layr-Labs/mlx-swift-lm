import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

/// Metadata cases require no native model. Native cases explicitly opt in and
/// use only the bounded193-tensor synthetic BF16 fixture, never real weights.

private func withMiMoConstructionScope<Value>(
    _ body: (NativeConstructionScope) throws -> Value
) rethrows -> Value {
    let work = NativeConstructionScope()
    defer {
        // Unexpected failed completion is restart-only, including in this
        // dedicated native test process. Never deallocate its sole SDK owner.
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    return try body(work)
}

final class MiMoV26SerialLoadTests: XCTestCase {
    private enum ProbeError: Error { case revoked, callback }
    private final class Reservation: MiMoV26SerialLoadReservation {
        var request: MiMoV26SerialLoadRequest
        var reservedLoadBytes: UInt64
        var revoked = false
        var validations = 0
        var onValidate: ((MiMoV26SerialLoadProgress) throws -> Void)?
        init(_ request: MiMoV26SerialLoadRequest, bytes: UInt64? = nil) {
            self.request = request
            reservedLoadBytes = bytes ?? request.requiredLoadBytes
        }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {
            validations += 1
            if revoked { throw ProbeError.revoked }
            try onValidate?(progress)
        }
    }

    func testMetadataBindsOneExactRootAndUniqueLoadGeneration() throws {
        let plan = try preflight(fixture(copy: false))
        let first = try MiMoV26SerialLoadSession(plan: plan)
        let second = try MiMoV26SerialLoadSession(plan: plan)
        XCTAssertEqual(first.request.binding, second.request.binding)
        XCTAssertNotEqual(first.request.sessionID, second.request.sessionID)
        XCTAssertEqual(
            try first.request.binding.fingerprint(), try second.request.binding.fingerprint())
        XCTAssertEqual(first.request.binding.tensorBytes, 216_700)
        XCTAssertEqual(first.request.binding.shards.count, 4)
        XCTAssertEqual(first.request.binding.scope, "root-bundle-only")
        XCTAssertEqual(first.request.requiredLoadBytes, first.request.binding.estimate.totalBytes)
        // Exact metadata-only binding for the subsequent same-binary tiny
        // diagnostic grant. This is not load permission or a native result.
        print("MIMO_SERIAL_SYNTHETIC_BINDING " + (try first.request.binding.fingerprint()))
    }

    func testForeignSessionOrInsufficientPermitRejectsBeforeCallbacks() throws {
        let plan = try preflight(fixture(copy: false))
        let first = try MiMoV26SerialLoadSession(plan: plan)
        let second = try MiMoV26SerialLoadSession(plan: plan)
        let foreign = Reservation(second.request)
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try first.load(reservation: foreign, retaining: constructionWork)
            }
        ) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .reservationMismatch)
        }
        XCTAssertEqual(foreign.validations, 0)
        let short = Reservation(second.request, bytes: second.request.requiredLoadBytes - 1)
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try second.load(reservation: short, retaining: constructionWork)
            }
        ) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .insufficientReservation)
        }
        XCTAssertEqual(short.validations, 0)
        // Even a failed attempt is consumed. Never silently reload on retry.
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try second.load(
                    reservation: Reservation(second.request), retaining: constructionWork)
            }
        ) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .alreadyConsumed)
        }
    }

    func testCancellationAndRevocationRejectBeforeNativeConstruction() throws {
        let plan = try preflight(fixture(copy: false))
        let cancelled = try MiMoV26SerialLoadSession(plan: plan)
        let permit = Reservation(cancelled.request)
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try cancelled.load(
                    reservation: permit, retaining: constructionWork, isCancelled: { true })
            }
        ) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .cancelled)
        }
        XCTAssertEqual(permit.validations, 0)
        let revoked = try MiMoV26SerialLoadSession(plan: plan)
        let stale = Reservation(revoked.request)
        stale.revoked = true
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try revoked.load(reservation: stale, retaining: constructionWork)
            }
        ) {
            XCTAssertTrue($0 is ProbeError)
        }
        XCTAssertEqual(stale.validations, 1)
    }

    func testBothReservationAndProgressHooksCannotChangeTheBoundFiles() throws {
        for mutateInReservation in [false, true] {
            let root = try fixture()
            let plan = try preflight(root)
            let session = try MiMoV26SerialLoadSession(plan: plan)
            let permit = Reservation(session.request)
            var mutated = false
            func mutation() throws {
                guard !mutated else { return }
                mutated = true
                try appendByte(root.appendingPathComponent("target.safetensors"))
            }
            if mutateInReservation { permit.onValidate = { _ in try mutation() } }
            XCTAssertThrowsError(
                try withMiMoConstructionScope { constructionWork in
                    try session.load(
                        reservation: permit, retaining: constructionWork,
                        progress: { value in
                            XCTAssertEqual(value.phase, .admitted)
                            if !mutateInReservation { try mutation() }
                        })
                }
            ) { XCTAssertTrue($0 is MiMoV26FilesystemError) }
            XCTAssertTrue(mutated)
        }
    }

    func testProgressCannotSwapOrRevokePermitBeforeCreatingHandles() throws {
        for mode in 0 ..< 3 {
            let plan = try preflight(fixture(copy: false))
            let session = try MiMoV26SerialLoadSession(plan: plan)
            let other = try MiMoV26SerialLoadSession(plan: plan)
            let permit = Reservation(session.request)
            var callbacks = 0
            XCTAssertThrowsError(
                try withMiMoConstructionScope { constructionWork in
                    try session.load(
                        reservation: permit, retaining: constructionWork,
                        progress: { value in
                            callbacks += 1
                            XCTAssertEqual(value.phase, .admitted)
                            switch mode {
                            case 0: permit.request = other.request
                            case 1: permit.reservedLoadBytes = 0
                            default: permit.revoked = true
                            }
                        })
                })
            XCTAssertEqual(callbacks, 1)
        }
    }

    func testProgressThrowCannotPublishAndOneShotIsConsumed() throws {
        let session = try MiMoV26SerialLoadSession(plan: preflight(fixture(copy: false)))
        let permit = Reservation(session.request)
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try session.load(
                    reservation: permit, retaining: constructionWork,
                    progress: { _ in throw ProbeError.callback })
            }
        ) {
            XCTAssertTrue($0 is ProbeError)
        }
        XCTAssertThrowsError(
            try withMiMoConstructionScope { constructionWork in
                try session.load(reservation: permit, retaining: constructionWork)
            }
        ) {
            XCTAssertEqual($0 as? MiMoV26SerialLoadError, .alreadyConsumed)
        }
    }

    func testNativeSerialMaterializationAndExplicitWrapperRetention() throws {
        try nativeLane()
        let session = try MiMoV26SerialLoadSession(plan: preflight(fixture(copy: false)))
        var permit: Reservation? = Reservation(session.request)
        weak var witness = permit
        var events: [MiMoV26SerialLoadProgress] = []
        var loaded: MiMoV26SerialLoadResult? = try withMiMoConstructionScope { constructionWork in
            try session.load(
                reservation: XCTUnwrap(permit), retaining: constructionWork,
                progress: { events.append($0) })
        }
        XCTAssertEqual(loaded?.receipt.sourceTensorCount, 193)
        XCTAssertEqual(loaded?.receipt.parameterCount, 193)
        XCTAssertEqual(loaded?.receipt.materializedSourcePayloadBytes, 216_700)
        XCTAssertEqual(loaded?.receipt.payloadHashesRecomputed, false)
        XCTAssertEqual(loaded?.receipt.externalComponentsLoaded, false)
        XCTAssertEqual(events.first?.phase, .admitted)
        XCTAssertEqual(events.last?.phase, .complete)
        XCTAssertTrue(
            zip(events, events.dropFirst()).allSatisfy { pair in
                pair.0.materializedSourcePayloadBytes <= pair.1.materializedSourcePayloadBytes
            })
        let target = try XCTUnwrap(loaded).bundle.target
        let bundle = try XCTUnwrap(loaded).bundle
        let modules: [Module] = [target, bundle.vision, bundle.audioPatch, bundle.mtp]
        for module in modules {
            for (_, array) in module.parameters().flattened() {
                XCTAssertNotNil(
                    try array.evaluatedBufferInfo(), "no lazy parameter may escape the load permit")
            }
        }
        let visual = Dictionary(uniqueKeysWithValues: bundle.vision.parameters().flattened())
        let patch = try XCTUnwrap(visual["patch_embed.proj.weight"])
        XCTAssertEqual(patch.shape, [8, 24])  // [out,C,T,H,W] → [out,patchWidth], unchanged byte order.
        XCTAssertEqual(patch.dtype, .bfloat16)
        XCTAssertEqual(patch.asArray(Float.self).first, Float(0.010009765625))
        XCTAssertNoThrow(try JSONEncoder().encode(try XCTUnwrap(loaded).receipt))
        permit = nil
        XCTAssertNotNil(
            witness, "result must carry the host permit; SDK must not silently refund it")
        withExtendedLifetime(target) {
            loaded = nil
            // Wrapper-only guarantee. A real host must reconcile residency or
            // retain the permit separately before extracting model aliases.
            XCTAssertNil(
                witness, "no session-owned permit alias may escape the explicit result lifetime")
        }
    }

    func testNativeSafeBoundaryCancellationAndTerminalPermitRevalidation() throws {
        try nativeLane()
        for atCompletion in [false, true] {
            let session = try MiMoV26SerialLoadSession(plan: preflight(fixture(copy: false)))
            let permit = Reservation(session.request)
            var cancel = false
            var reached = false
            XCTAssertThrowsError(
                try withMiMoConstructionScope { constructionWork in
                    try session.load(
                        reservation: permit, retaining: constructionWork, isCancelled: { cancel },
                        progress: { value in
                            if atCompletion && value.phase == .complete {
                                reached = true
                                permit.revoked = true
                            } else if !atCompletion && value.sourceTensorsCompleted > 0 {
                                reached = true
                                cancel = true
                            }
                        })
                })
            XCTAssertTrue(reached)
        }
    }

    private func directory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_LOAD_FIXTURES"] else {
            throw XCTSkip("Set MIMO_V26_SERIAL_LOAD_FIXTURES to generated synthetic BF16 fixtures")
        }
        return URL(fileURLWithPath: path)
    }
    private func fixture(copy: Bool = true) throws -> URL {
        let directory = try directory()
        let original = directory.appendingPathComponent("tiny-bf16")
        if !copy { return original }
        let root = directory.appendingPathComponent("test-work-" + UUID().uuidString)
        try FileManager.default.copyItem(at: original, to: root)
        return root  // Retain each bounded copy for failure inspection.
    }
    private func preflight(_ root: URL) throws -> MiMoV26FilesystemLoadPlan {
        let p = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with:
                    Data(contentsOf: directory().appendingPathComponent("provenance.json")))
                as? [String: String])
        let provenance = try MiMoV26ConvertedProvenance(
            artifactID: XCTUnwrap(p["artifactID"]),
            sourceRepository: XCTUnwrap(p["sourceRepository"]),
            sourceRevision: XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256: XCTUnwrap(p["conversionManifestSHA256"]))
        return try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: provenance,
            limits: .init(maximumShardBytes: 1_048_576, maximumTotalFileBytes: 4_194_304))
    }
    private func appendByte(_ url: URL) throws {
        let file = try FileHandle(forWritingTo: url)
        defer { try? file.close() }
        try file.seekToEnd()
        try file.write(contentsOf: Data([0]))
    }
    private func nativeLane() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_SERIAL_NATIVE_TESTS"] == "1" else {
            throw XCTSkip(
                "Requires coordinator-owned native lane and explicit MIMO_V26_SERIAL_NATIVE_TESTS=1"
            )
        }
    }
}
