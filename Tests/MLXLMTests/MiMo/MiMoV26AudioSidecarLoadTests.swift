import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Component tests only. The native runner must separately own the real
/// physical lane/budget. TestReservation exercises SDK binding, not host C/M.
private final class AudioLoadTestReservation: MiMoV26AudioSidecarLoadReservation {
    let request: MiMoV26AudioSidecarLoadRequest
    let reservedLoadBytes: UInt64
    var revoked = false
    init(_ request: MiMoV26AudioSidecarLoadRequest, bytes: UInt64? = nil) {
        self.request = request
        reservedLoadBytes = bytes ?? request.requiredLoadBytes
    }
    func validateActive() throws {
        if revoked { throw MiMoV26AudioSidecarError.invalidatedOwner }
    }
}

final class MiMoV26AudioSidecarLoadTests: XCTestCase {
    private enum Fault: Error { case injected }
    private func fixture() throws -> MiMoV26AudioSidecarLoadSession {
        guard ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_SIDECAR_NATIVE_TESTS"] == "1",
            let path = ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_SIDECAR_FIXTURE_ROOT"]
        else {
            throw XCTSkip("Requires owned native lane and an immutable selected sidecar fixture")
        }
        let root = URL(fileURLWithPath: path)
        let handle = try FileHandle(forReadingFrom: root.appendingPathComponent("config.json"))
        defer { try? handle.close() }
        let bytes = try XCTUnwrap(handle.read(upToCount: (1 << 20) + 1))
        guard bytes.count <= 1 << 20 else { throw MiMoV26AudioSidecarError.invalidConfiguration }
        return try .init(
            root: root,
            mainConfiguration: JSONDecoder().decode(MiMoV26Configuration.self, from: bytes),
            mainConfigurationSHA256: mimoAudioDigest(bytes))
    }

    func testSelectedSubsetMaterializesWithAuthenticatedWholeFileAndExactGeneration() throws {
        let session = try fixture()
        let scope = NativeConstructionScope()
        defer { if scope.snapshot.isRetainedFault { _ = Unmanaged.passRetained(scope) } }
        let permit = AudioLoadTestReservation(session.request)
        var authenticatedBeforeFirstTensor = false
        let loaded = try session.load(
            reservation: permit, retaining: scope,
            progress: { progress in
                if progress.phase == .authenticated {
                    XCTAssertEqual(progress.authenticatedFileBytes, 1_872_618_384)
                    authenticatedBeforeFirstTensor = true
                }
                if progress.phase == .inputMaterialization {
                    XCTAssertTrue(authenticatedBeforeFirstTensor)
                }
            })
        try loaded.validate()
        XCTAssertEqual(loaded.receipt.materializedInputTensors, 389)
        XCTAssertEqual(loaded.receipt.materializedInputBytes, 634_204_160)
        XCTAssertEqual(loaded.receipt.request.unusedTensorCount, 439)
        XCTAssertEqual(loaded.receipt.request.unusedStoredBytes, 1_238_321_176)
        XCTAssertEqual(loaded.receipt.runtimeRoots, 389)
        XCTAssertEqual(loaded.receipt.codecGeneration, loaded.codec.generation)
        guard case .completed(let receipt) = scope.snapshot.disposition else {
            return XCTFail("No actual construction completion")
        }
        try scope.validate(receipt)
        XCTAssertEqual(receipt.completion, .capturedStreamsCompleted)
        XCTAssertThrowsError(try session.load(reservation: permit, retaining: scope))
    }

    func testInsufficientPermitFailsBeforeFullPayloadAuthenticationOrNativeSubmission() throws {
        let session = try fixture()
        let scope = NativeConstructionScope()
        let permit = AudioLoadTestReservation(session.request, bytes: 0)
        XCTAssertThrowsError(
            try session.load(
                reservation: permit, retaining: scope,
                progress: { _ in XCTFail("Invalid permit reached load progress") })
        ) {
            XCTAssertEqual($0 as? MiMoV26AudioSidecarError, .insufficientReservation)
        }
        guard case .completed(let receipt) = scope.snapshot.disposition else {
            return XCTFail("No actual no-submission completion")
        }
        try scope.validate(receipt)
        XCTAssertEqual(receipt.completion, .noNativeSubmission)
    }

    func testPartialFailureRetainsSourceAndNativeRootsWhenFenceFails() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_SIDECAR_FAULT_TEST"] == "1" else {
            throw XCTSkip("Retained-fault selector must run alone in its owned process")
        }
        let session = try fixture()
        let scope = NativeConstructionScope()
        defer { if scope.snapshot.isRetainedFault { _ = Unmanaged.passRetained(scope) } }
        let permit = AudioLoadTestReservation(session.request)
        scope.testingBoundary = { name, _ in
            if name == "audioSidecar.sourceMaterialized" { throw Fault.injected }
        }
        scope.testingBeforeFence = { _ in throw Fault.injected }
        XCTAssertThrowsError(try session.load(reservation: permit, retaining: scope))
        XCTAssertTrue(scope.snapshot.isRetainedFault)
        XCTAssertGreaterThan(scope.snapshot.retainedArrayCount, 0)
        XCTAssertGreaterThan(scope.snapshot.retainedOwnerCount, 0)
        XCTAssertThrowsError(try session.load(reservation: permit, retaining: scope))
    }

    func testRawBF16DataInitializerPreservesStorageRatherThanConvertingUInt16Numerically() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_SIDECAR_NATIVE_TESTS"] == "1"
        else {
            throw XCTSkip("Requires owned native lane")
        }
        let bits: [UInt16] = [0, 0x8000, 0x3f80, 0xbf80, 0x3f81, 0x0080, 0x0040]
        let data = bits.withUnsafeBytes { Data($0) }
        let actual = MLXArray(data, [bits.count], dtype: .bfloat16)
        eval(actual)
        XCTAssertEqual(actual.view(dtype: .uint16).asArray(UInt16.self), bits)
        XCTAssertNotEqual(actual.asType(.float32).asArray(Float.self), bits.map(Float.init))
    }
}
