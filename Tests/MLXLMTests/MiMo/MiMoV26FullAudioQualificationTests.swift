import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MLXVLM

private final class AudioMemoryLoadPermit: MiMoV26AudioSidecarLoadReservation {
    let request: MiMoV26AudioSidecarLoadRequest
    var reservedLoadBytes: UInt64 { request.requiredLoadBytes }
    init(_ request: MiMoV26AudioSidecarLoadRequest) { self.request = request }
    func validateActive() throws {}
}
private final class AudioMemoryWorkPermit: MiMoV26AudioWorkReservation {
    let plan: MiMoV26AudioInputPlan
    let weights: MiMoV26AudioInputWeights
    init(_ plan: MiMoV26AudioInputPlan, _ weights: MiMoV26AudioInputWeights) {
        self.plan = plan
        self.weights = weights
    }
    func validate(plan: MiMoV26AudioInputPlan, sourceIdentity: String, generation: UUID) throws {
        guard plan == self.plan, sourceIdentity == weights.sourceIdentity,
            generation == weights.generation
        else {
            throw MiMoV26AudioInputError.weightsNotLoaded
        }
    }
    func retainAfterFailedDrain(_ work: MiMoV26FailedAudioWork) {
        _ = Unmanaged.passRetained(work)
    }
}

/// Explicit component qualification; no language model or network requests.
final class MiMoV26FullAudioQualificationTests: XCTestCase {
    func testCapturedAACAndDurationMatrixWithActualCodec() async throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_MEMORY_MATRIX"] == "1",
            let path = ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_SIDECAR_FIXTURE_ROOT"]
        else {
            throw XCTSkip(
                "Requires the verified selected audio codec and explicit memory-matrix opt in")
        }
        let root = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: root.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(MiMoV26Configuration.self, from: data)
        let session = try MiMoV26AudioSidecarLoadSession(
            root: root, mainConfiguration: config,
            mainConfigurationSHA256: SHA256.hash(data: data).map { String(format: "%02x", $0) }
                .joined())
        let scope = NativeConstructionScope()
        defer { if scope.snapshot.isRetainedFault { _ = Unmanaged.passRetained(scope) } }
        let loaded = try session.load(
            reservation: AudioMemoryLoadPermit(session.request), retaining: scope)
        let input = loaded.codec.input
        let limits = mimoAudioInputTestLimits(block: 6000, tile: 1500)
        let patchLimits = MiMoV26AudioPatchLimits(
            maximumClips: 64,
            maximumFrames: 100_000, maximumPatches: 25_000, maximumWorkingElements: 1 << 40)

        func run(_ clip: MiMoV26DecodedPCM, name: String) throws {
            Memory.clearCache()
            let baseline = Memory.snapshot().activeMemory
            Memory.peakMemory = 0
            var reserved = 0
            let codes = try input.encode(
                clips: [clip], limits: limits, retaining: loaded,
                authorize: { plan in
                    reserved = try MiMoV26ManagedAudioCommitment.additionalBytes(
                        input: plan,
                        patchConfiguration: XCTUnwrap(config.audio), limits: patchLimits)
                    return AudioMemoryWorkPermit(plan, input.weights)
                })
            XCTAssertEqual(codes.count, 1)
            XCTAssertFalse(codes[0].codes.isEmpty)
            let snapshot = Memory.snapshot()
            let incremental = max(0, snapshot.peakMemory - baseline)
            XCTAssertLessThanOrEqual(incremental, reserved, name)
            let result: [String: Any] = [
                "fixture": name, "baseline_bytes": baseline,
                "incremental_peak_bytes": incremental, "additional_quote_bytes": reserved,
                "code_frames": codes[0].frameCount,
            ]
            print(
                String(
                    decoding: try JSONSerialization.data(
                        withJSONObject: result, options: [.sortedKeys]), as: UTF8.self))
        }

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "input-video-url.mp4",
                withExtension: nil, subdirectory: "MiMoOpenRouter"))
        let owner = try MemoryBackedVideoAsset(videoData: Data(contentsOf: url))
        let visualLimits = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: 8_388_608,
            maximumWorkingBytes: 1 << 30, maximumSourceFrames: 360_000,
            maximumSampledFrames: 3600, maximumEncodedBytes: 32 << 20)
        let video = try await MiMoV26EncodedVisualDecoder.inspectVideo(
            owner,
            sampling: .init(configuration: config), limits: visualLimits)
        let audioLimits = MiMoV26EncodedAudiovisualDecoder.Limits(
            maximumFrames: 1_000_000, maximumWorkingBytes: 1 << 30)
        let av = try await MiMoV26EncodedAudiovisualDecoder.inspect(video, limits: audioLimits)
        let decoded = try await MiMoV26EncodedAudiovisualDecoder.decode(
            av,
            videoLimits: visualLimits, audioLimits: audioLimits)
        try run(decoded.wholeAudio, name: "captured-5s-stereo-AAC")
        for seconds in [1, 30, 61, 300] {
            let clip = try MiMoV26DecodedPCM(
                samples: (0 ..< (seconds * 24000)).map { Float(sin(Double($0) * 0.02)) * 0.1 },
                descriptor: .init(
                    sourceIdentity: "matrix-\(seconds)", channels: 1,
                    frameCount: seconds * 24000, sampleRate: 24000))
            try run(clip, name: "matrix-\(seconds)s-mono-24k")
        }
    }
}
