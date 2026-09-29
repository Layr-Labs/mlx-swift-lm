import Foundation
import XCTest

@testable import MLXLLM

func mimoAudioInputTestLimits(working: Int = 1 << 40, block: Int = 16, tile: Int = 8)
    -> MiMoV26AudioInputLimits
{
    .init(
        maximumClips: 64, maximumChannels: 8, maximumSampleRate: 192000,
        maximumInputSamples: 8_000_000, maximumResampledSamples: 8_000_000,
        maximumResampleCoefficients: 1_000_000,
        maximumMelFrames: 1_000_000, maximumSegments: 512, maximumPaddedMelFrames: 4_000_000,
        maximumWorkingElements: working, frontendFrameBlockSize: block, rvqTileFrames: tile)
}
func mimoAudioInputFixtureRoot() throws -> URL {
    guard let path = ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_INPUT_FIXTURES"] else {
        throw XCTSkip("Set MIMO_V26_AUDIO_INPUT_FIXTURES to the prepared source/oracle fixtures")
    }
    return URL(fileURLWithPath: path, isDirectory: true)
}
func mimoAudioInputRequireNative() throws {
    guard ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_INPUT_NATIVE_TESTS"] == "1" else {
        throw XCTSkip("Native audio-input component tests require the explicit resource lane")
    }
}

final class MiMoV26AudioInputPlanTests: XCTestCase {
    func testSelectedSidecarAndUnknownRoundTripWithoutArchitectureAlias() throws {
        let root = try mimoAudioInputFixtureRoot()
        let main = try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: Data(contentsOf: root.appendingPathComponent("main-config.json")))
        var sidecar = try JSONDecoder().decode(
            [String: MiMoV26JSONValue].self,
            from: Data(contentsOf: root.appendingPathComponent("sidecar-config.json")))
        sidecar["future_metadata"] = .object([
            "large": .number(Decimal(string: "9007199254740993")!), "null": .null,
        ])
        let encoded = try JSONEncoder().encode(sidecar)
        let config = try MiMoV26AudioInputConfiguration(
            sidecarJSON: encoded, mainConfiguration: main)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.layers, 24)
        XCTAssertEqual(config.skipLayerIndex, 2)
        XCTAssertEqual(config.codebookSizes, [1024, 1024, 256] + Array(repeating: 128, count: 17))
        XCTAssertEqual(
            try JSONDecoder().decode(
                [String: MiMoV26JSONValue].self, from: config.encodedSidecar()), sidecar)
        sidecar["encoder_causal"] = .bool(false)
        XCTAssertThrowsError(
            try MiMoV26AudioInputConfiguration(
                sidecarJSON: JSONEncoder().encode(sidecar), mainConfiguration: main))
    }

    func testExactLocalEdgeAndHybridSchedule() throws {
        let c = try MiMoV26AudioInputConfiguration.fixture(layers: 24, lookBack: 128)
        XCTAssertEqual(
            (0 ..< 24).filter(c.usesLocalAttention(layer:)), Array(stride(from: 0, to: 24, by: 2)))
        XCTAssertTrue(c.allowsAttention(query: 128, key: 0, layer: 0))
        XCTAssertFalse(c.allowsAttention(query: 129, key: 0, layer: 0))
        XCTAssertTrue(c.allowsAttention(query: 129, key: 1, layer: 0))
        XCTAssertTrue(c.allowsAttention(query: 129, key: 0, layer: 1))
        XCTAssertFalse(c.allowsAttention(query: 0, key: 1, layer: 1))
    }

    func testOneSecondCountsAndResampleFloat32Trim() throws {
        let c = try MiMoV26AudioInputConfiguration.fixture(melBands: 128)
        let plan = try MiMoV26AudioInputPlan.make(
            clips: [
                .init(
                    sourceIdentity: "one-second", channels: 1, frameCount: 24000, sampleRate: 24000)
            ],
            configuration: c, limits: mimoAudioInputTestLimits())
        XCTAssertEqual(plan.melFrameCounts, [101])
        XCTAssertEqual(plan.codeFrameCounts, [26])
        XCTAssertEqual(plan.patchCounts, [7])
        XCTAssertEqual(plan.totalPatches + 2, 9)
        let resample = try MiMoV26AudioResamplePlan.make(
            originalRate: 44100, frames: 294068, maximumCoefficients: 1_000_000)
        XCTAssertEqual(resample.originalRatio, 147)
        XCTAssertEqual(resample.targetRatio, 80)
        XCTAssertEqual(resample.width, 12)
        XCTAssertEqual(resample.taps, 171)
        XCTAssertEqual(resample.outputFrames, 160037)
        XCTAssertEqual(try MiMoV26AudioChecked.ceilDivide(80 * 294068, 147), 160038)
    }

    func testOriginalGroupGeometryAndOddTailDistinguishAloneFromBatch() throws {
        let c = try MiMoV26AudioInputConfiguration.fixture()
        let limits = mimoAudioInputTestLimits()
        func plan(_ lengths: [Int]) throws -> MiMoV26AudioInputPlan {
            try .makeMelInputs(
                frameCounts: lengths, sourceIdentities: lengths.indices.map { "clip-\($0)" },
                configuration: c, limits: limits)
        }
        let alone = try plan([5])
        let peer = try plan([5, 6])
        let remainder = try plan([6005])
        XCTAssertEqual(alone.groups[0].maximumMelFrames, 5)
        XCTAssertEqual(peer.groups[0].maximumMelFrames, 6)
        XCTAssertEqual(alone.groups[0].maximumConvFrames, 3)
        XCTAssertEqual(peer.groups[0].maximumConvFrames, 3)
        XCTAssertFalse(alone.groups[0].repeatsLastBeforePooling(segment: alone.segments[0]))
        XCTAssertFalse(peer.groups[0].repeatsLastBeforePooling(segment: peer.segments[0]))
        XCTAssertTrue(remainder.groups[0].repeatsLastBeforePooling(segment: remainder.segments[1]))
        XCTAssertEqual(remainder.segments.map(\.melFrames), [6000, 5])
        XCTAssertEqual(remainder.codeFrameCounts, [1502])
        XCTAssertNotEqual(try alone.preparationIdentityData(), try peer.preparationIdentityData())
    }

    func testConsecutiveGroupsAndOriginalItemRejoin() throws {
        let c = try MiMoV26AudioInputConfiguration.fixture()
        let p = try MiMoV26AudioInputPlan.makeMelInputs(
            frameCounts: Array(repeating: 6000, count: 43),
            sourceIdentities: (0 ..< 43).map(String.init), configuration: c,
            limits: mimoAudioInputTestLimits())
        XCTAssertEqual(p.groups.count, 2)
        XCTAssertEqual(p.groups[0].validMelFrames, 252000)
        XCTAssertEqual(p.groups[0].segmentIndices, Array(0 ..< 42))
        XCTAssertEqual(p.groups[1].segmentIndices, [42])
        XCTAssertEqual(p.codeFrameCounts, Array(repeating: 1500, count: 43))
        let q = try MiMoV26AudioInputPlan.makeMelInputs(
            frameCounts: [6001, 12001], sourceIdentities: ["a", "b"],
            configuration: c, limits: mimoAudioInputTestLimits())
        XCTAssertEqual(q.codeFrameCounts, [1501, 3001])
        XCTAssertEqual(q.patchCounts, [376, 751])
    }

    func testInvalidShortInputsAndNoSilentRegroupOrClamp() throws {
        let c = try MiMoV26AudioInputConfiguration.fixture()
        let limits = mimoAudioInputTestLimits()
        XCTAssertThrowsError(
            try MiMoV26AudioInputPlan.make(
                clips: [
                    .init(sourceIdentity: "short", channels: 1, frameCount: 480, sampleRate: 24000)
                ], configuration: c, limits: limits))
        let p = try MiMoV26AudioInputPlan.make(
            clips: [
                .init(sourceIdentity: "minimum", channels: 1, frameCount: 481, sampleRate: 24000)
            ], configuration: c, limits: limits)
        XCTAssertEqual(p.melFrameCounts, [3])
        XCTAssertThrowsError(
            try MiMoV26AudioResamplePlan.make(
                originalRate: 48001, frames: 48001, maximumCoefficients: 1_000_000))
        XCTAssertThrowsError(try MiMoV26AudioChecked.product([Int.max, 2], "overflow"))
        XCTAssertThrowsError(
            try MiMoV26AudioInputPlan.makeMelInputs(
                frameCounts: [0], sourceIdentities: ["empty"], configuration: c, limits: limits))
        XCTAssertThrowsError(
            try MiMoV26AudioInputPlan.makeMelInputs(
                frameCounts: [6000, 5], sourceIdentities: ["a", "b"], configuration: c,
                limits: mimoAudioInputTestLimits(working: 1)))
        let empty = try MiMoV26AudioInputPlan.make(clips: [], configuration: c, limits: limits)
        XCTAssertTrue(empty.groups.isEmpty)
        XCTAssertEqual(empty.workingElementUpperBound, 0)
    }
}
