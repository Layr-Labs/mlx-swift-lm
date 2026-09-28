import Foundation
import MLXLLM
import MLXVLM
import XCTest

/// Metadata-only plan/accounting tests; no waveform buffers or native arrays.
final class MiMoV26ManagedAudioCommitmentTests: XCTestCase {
    private func metadata() throws -> (MiMoV26AudioInputConfiguration, MiMoV26AudioConfiguration) {
        guard let root = ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_SIDECAR_FIXTURE_ROOT"] else {
            throw XCTSkip("Requires selected main/sidecar configuration metadata only")
        }
        func read(_ relative: String) throws -> Data {
            let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: root).appendingPathComponent(relative))
            defer { try? file.close() }
            let data = try XCTUnwrap(file.read(upToCount: (1 << 20) + 1))
            guard data.count <= 1 << 20 else { throw MiMoV26AudioSidecarError.invalidConfiguration }
            return data
        }
        let main = try JSONDecoder().decode(MiMoV26Configuration.self, from: read("config.json"))
        return (try .init(sidecarJSON: read("audio_tokenizer/config.json"), mainConfiguration: main),
                try XCTUnwrap(main.audio))
    }

    private func limits(tile: Int) -> MiMoV26AudioInputLimits {
        .init(maximumClips: 4, maximumChannels: 2, maximumSampleRate: 48_000,
            maximumInputSamples: 1_000_000, maximumResampledSamples: 1_000_000,
            maximumResampleCoefficients: 1_000_000, maximumMelFrames: 20_000,
            maximumSegments: 8, maximumPaddedMelFrames: 40_000,
            maximumWorkingElements: 1_000_000_000_000, frontendFrameBlockSize: 64, rvqTileFrames: tile)
    }
    private var patchLimits: MiMoV26AudioPatchLimits {
        .init(maximumClips: 4, maximumFrames: 100_000, maximumPatches: 25_000,
              maximumWorkingElements: 1_000_000_000_000)
    }

    func testAdditionalCommitmentPricesMixedOriginalGroupsAndEveryRVQTile() throws {
        let (codec, patch) = try metadata()
        let first = MiMoV26AudioPCMDescriptor(sourceIdentity: "first", channels: 1, frameCount: 24_000, sampleRate: 24_000)
        let second = MiMoV26AudioPCMDescriptor(sourceIdentity: "second", channels: 1, frameCount: 12_001, sampleRate: 24_000)
        let one = try MiMoV26AudioInputPlan.make(clips: [first], configuration: codec, limits: limits(tile: 64))
        let mixed = try MiMoV26AudioInputPlan.make(clips: [first, second], configuration: codec, limits: limits(tile: 64))
        let tiled = try MiMoV26AudioInputPlan.make(clips: [first], configuration: codec, limits: limits(tile: 1))
        let originalGroups = mixed.groups
        let singleBytes = try MiMoV26ManagedAudioCommitment.additionalBytes(input: one, patchConfiguration: patch, limits: patchLimits)
        let mixedBytes = try MiMoV26ManagedAudioCommitment.additionalBytes(input: mixed, patchConfiguration: patch, limits: patchLimits)
        let tiledBytes = try MiMoV26ManagedAudioCommitment.additionalBytes(input: tiled, patchConfiguration: patch, limits: patchLimits)
        XCTAssertGreaterThan(singleBytes, one.workingElementUpperBound * 4)
        XCTAssertGreaterThan(mixedBytes, singleBytes)
        XCTAssertGreaterThan(tiledBytes, singleBytes)
        XCTAssertEqual(mixed.groups, originalGroups, "admission must not regroup audio to fit")
    }

    func testMelOnlyNamespaceAndInsufficientPatchLimitsRefuse() throws {
        let (codec, patch) = try metadata()
        let mel = try MiMoV26AudioInputPlan.makeMelInputs(frameCounts: [101], sourceIdentities: ["mel"],
            configuration: codec, limits: limits(tile: 64))
        XCTAssertThrowsError(try MiMoV26ManagedAudioCommitment.additionalBytes(
            input: mel, patchConfiguration: patch, limits: patchLimits))
        let pcm = try MiMoV26AudioInputPlan.make(clips: [
            .init(sourceIdentity: "pcm", channels: 1, frameCount: 24_000, sampleRate: 24_000)
        ], configuration: codec, limits: limits(tile: 64))
        XCTAssertThrowsError(try MiMoV26ManagedAudioCommitment.additionalBytes(input: pcm,
            patchConfiguration: patch, limits: .init(maximumClips: 1, maximumFrames: 1,
                maximumPatches: 1, maximumWorkingElements: 1)))
    }
}
