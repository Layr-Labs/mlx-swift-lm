// Copyright © 2026 Eigen Labs.

import CoreImage
import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Video sampling, geometry, timestamps, expansion, preprocessing and
/// encoding against the reference's rules. Tiny arrays and fixtures only.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerVideoTests {
    typealias Video = MuseGlimmerVideoGeometry
    typealias Media = MuseGlimmerMediaPrompt
    typealias Image = MuseGlimmerImageGeometry
    typealias Failure = MuseGlimmerImageError

    @Test func frameSamplingFollowsTheReference() throws {
        #expect(try Video.sampledFrameIndices(totalFrames: 1, sourceFPS: 24) == [0])
        #expect(try Video.sampledFrameIndices(totalFrames: 10, sourceFPS: 24) == [0, 9])
        // Ten seconds at 30 fps: 20 frames, every 299/19 frames.
        let ten = try Video.sampledFrameIndices(totalFrames: 300, sourceFPS: 30)
        #expect(ten.count == 20)
        #expect(Array(ten.prefix(6)) == [0, 15, 31, 47, 62, 78])
        #expect(Array(ten.suffix(3)) == [267, 283, 299])
        // Four minutes at 30 fps: capped at 96 frames.
        let long = try Video.sampledFrameIndices(totalFrames: 7200, sourceFPS: 30)
        #expect(long.count == 96)
        #expect(Array(long.prefix(6)) == [0, 75, 151, 227, 303, 378])
        #expect(Array(long.suffix(3)) == [7047, 7123, 7199])
        // The second half counts down from the end, as torch does.
        let tight = try Video.sampledFrameIndices(totalFrames: 97, sourceFPS: 2)
        #expect(tight.count == 96 && Array(tight.suffix(3)) == [93, 94, 96])
        #expect(
            try Video.sampledFrameIndices(totalFrames: 149, sourceFPS: 29.97)
                == [0, 21, 42, 63, 84, 105, 126, 148])
        // No reported rate: the reference's 24 fps.
        #expect(try Video.sampledFrameIndices(totalFrames: 48, sourceFPS: nil) == [0, 15, 31, 47])
        for (frames, fps) in [(0, 24.0), (10, 0), (10, -1), (10, .nan), (10, .infinity)] {
            #expect(throws: Failure.self) {
                try Video.sampledFrameIndices(totalFrames: frames, sourceFPS: fps)
            }
        }
    }

    @Test func gridsCapEachGroupAndTheVideo() throws {
        let full = try Video.grid(frames: 96, sourceHeight: 1080, sourceWidth: 1920)
        #expect(full.groups == 48)
        #expect(full.frame.patchRows == 18 && full.frame.patchColumns == 32)
        #expect(full.tokensPerGroup == 144 && full.featureRows == Video.maximumVideoTokens)
        #expect(full.thw.t == 48 && full.thw.h == 18 && full.thw.w == 32)
        let odd = try Video.grid(frames: 3, sourceHeight: 360, sourceWidth: 640)
        #expect(odd.groups == 2 && odd.tokensPerGroup <= 144)
        let frame = try Image.grid(sourceHeight: 360, sourceWidth: 640, maximumImageTokens: 144)
        #expect(odd.frame == frame)
        for frames in [0, 97] {
            #expect(throws: Failure.self) {
                try Video.grid(frames: frames, sourceHeight: 360, sourceWidth: 640)
            }
        }
        #expect(throws: Failure.self) {
            try Video.Grid(groups: 1, frame: try Image.Grid(mergedRows: 12, mergedColumns: 13))
        }
        #expect(throws: Failure.self) { try Video.Grid(groups: 49, frame: frame) }
    }

    /// vLLM's timestamps (served here) against HF's, which differ whenever
    /// the sampled rate is not exactly 2 fps.
    @Test func timestampsAreVLLMsAndTheHFDifferenceIsPinned() throws {
        #expect(Video.timestamps(groups: 4) == [0, 1, 2, 3])
        #expect(Video.timestampText(9) == "Time: 9.0s")
        // Rounded from the binary value, as Python's format does.
        #expect(Video.timestampText(0.35) == "Time: 0.3s")
        #expect(Video.timestampText(237.3666) == "Time: 237.4s")
        // A 2 fps source of 10 s: the two agree.
        let exact = try Video.sampledFrameIndices(totalFrames: 20, sourceFPS: 2)
        #expect(
            Video.huggingFaceTimestamps(sampledIndices: exact, sourceFPS: 2, groups: 10)
                == Video.timestamps(groups: 10))
        // 30 fps for 10 s: HF drifts ahead, 9.4 s against 9.0 s by group 9.
        let ten = try Video.sampledFrameIndices(totalFrames: 300, sourceFPS: 30)
        let hf = Video.huggingFaceTimestamps(sampledIndices: ten, sourceFPS: 30, groups: 10)
        #expect(hf.map(Video.timestampText).last == "Time: 9.4s")
        #expect(Video.timestamps(groups: 10).map(Video.timestampText).last == "Time: 9.0s")
        // Four minutes: 96 frames span 240 s, so HF's last group is at 237.4 s
        // and vLLM's at 47.0 s.
        let long = try Video.sampledFrameIndices(totalFrames: 7200, sourceFPS: 30)
        let hfLong = Video.huggingFaceTimestamps(sampledIndices: long, sourceFPS: 30, groups: 48)
        #expect(Video.timestampText(hfLong[47]) == "Time: 237.4s")
        #expect(Video.timestampText(Video.timestamps(groups: 48)[47]) == "Time: 47.0s")
        // An odd count pads with the last time.
        #expect(
            Video.huggingFaceTimestamps(sampledIndices: [0], sourceFPS: 24, groups: 1) == [0])
    }

    @Test func expansionInterleavesTimestampsAndRefusesMismatches() throws {
        let (patch, video) = (
            MuseGlimmerImagePrompt.imageTokenID, MuseGlimmerImagePrompt.videoTokenID
        )
        let clip = try Video.Grid(groups: 2, frame: try Image.Grid(mergedRows: 1, mergedColumns: 2))
        let still = try Image.Grid(mergedRows: 1, mergedColumns: 1)
        func time(_ text: String) -> [Int] { text.utf8.map { 1000 + Int($0) } }
        let expanded = try Media.expand(
            promptTokens: [5, video, 6, patch, 7], media: [.video(clip), .image(still)],
            timestampTokens: time)
        let expected =
            [5, Media.videoStartTokenID] + time("Time: 0.0s") + [video, video]
            + [Media.frameSeparatorTokenID] + time("Time: 1.0s") + [video, video]
            + [Media.videoEndTokenID, 6, MuseGlimmerImagePrompt.imageStartTokenID, patch]
            + [MuseGlimmerImagePrompt.imageEndTokenID, 7]
        #expect(expanded.tokens == expected)
        let first = 2 + time("Time: 0.0s").count
        let second = first + 3 + time("Time: 1.0s").count
        #expect(
            expanded.spans == [
                CBv2ImageSpan(tokenOffset: first, length: 2),
                CBv2ImageSpan(tokenOffset: second, length: 2),
                CBv2ImageSpan(tokenOffset: expected.count - 3, length: 1),
            ])
        #expect(Media.Item.video(clip).spanCount == 2 && Media.Item.image(still).spanCount == 1)
        // An image prompt expands as the image-only path does.
        #expect(
            try Media.expand(
                promptTokens: [5, patch, 6], media: [.image(still)], timestampTokens: time)
                == MuseGlimmerImagePrompt.expand(promptTokens: [5, patch, 6], grids: [still]))
        let refused: [([Int], [Media.Item])] = [
            ([5, video, patch], [.image(still), .video(clip)]),
            ([5, video], [.video(clip), .video(clip)]),
            ([5, 6], []),
            ([video], []),
        ]
        for (prompt, media) in refused {
            #expect(throws: Failure.self) {
                try Media.expand(promptTokens: prompt, media: media, timestampTokens: time)
            }
        }
        #expect(throws: Failure.self) {
            try Media.expand(promptTokens: [video], media: [.video(clip)]) { _ in [video] }
        }
    }

    @Test func pairPatchifyIsTheImageLayoutWithASecondFrame() throws {
        let grid = try Image.Grid(mergedRows: 1, mergedColumns: 2)
        let a = MuseGlimmerFixture.hostTensor([1, 3, grid.pixelHeight, grid.pixelWidth], seed: 1)
        let b = MuseGlimmerFixture.hostTensor([1, 3, grid.pixelHeight, grid.pixelWidth], seed: 2)
        func image(_ x: MLXArray) -> MLXArray {
            museGlimmerPatchify(
                x, gridH: grid.patchRows, gridW: grid.patchColumns, patchSize: Image.patchSize,
                temporalPatchSize: Image.temporalPatchSize)
        }
        #expect(MuseGlimmerFixture.sameBits(museGlimmerPatchifyPair(a, a, grid: grid), image(a)))
        let pair = museGlimmerPatchifyPair(a, b, grid: grid)
        let half = Image.patchValues / 2
        #expect(pair.shape == [grid.patchRows * grid.patchColumns, Image.patchValues])
        #expect(MuseGlimmerFixture.sameBits(pair[0..., ..<half], image(a)[0..., ..<half]))
        #expect(MuseGlimmerFixture.sameBits(pair[0..., half...], image(b)[0..., ..<half]))
    }

    @Test func preprocessingPairsFramesAndRepeatsTheLast() throws {
        // The pattern is wider than the frame, so each shifted crop is 640 wide.
        let frames = (0 ..< 3).map {
            MuseGlimmerImageEncoderTests.pattern(width: 660, height: 360).transformed(
                by: CGAffineTransform(translationX: CGFloat(-7 * $0), y: 0)
            ).cropped(to: CGRect(x: 0, y: 0, width: 640, height: 360))
        }
        let (groups, grid) = try MuseGlimmerImageEncoder.preprocessVideo(frames: frames)
        #expect(grid.groups == 2 && groups.count == 2)
        let single = try frames.map {
            try MuseGlimmerImageEncoder.preprocess(image: $0, maximumImageTokens: 144)
        }
        #expect(single.allSatisfy { $0.grid == grid.frame })
        let half = Image.patchValues / 2
        for (group, (a, b)) in [(0, (0, 1)), (1, (2, 2))] {
            #expect(groups[group].dtype == .float32)
            #expect(
                MuseGlimmerFixture.sameBits(
                    groups[group][0..., ..<half], single[a].pixels[0..., ..<half]))
            #expect(
                MuseGlimmerFixture.sameBits(
                    groups[group][0..., half...], single[b].pixels[0..., ..<half]))
        }
        #expect(throws: Failure.self) { try MuseGlimmerImageEncoder.preprocessVideo(frames: []) }
        #expect(throws: Failure.self) {
            try MuseGlimmerImageEncoder.preprocessVideo(
                frames: [
                    frames[0], frames[1].cropped(to: CGRect(x: 0, y: 0, width: 600, height: 360)),
                ])
        }
    }

    // MARK: Encoding

    /// Each group encoded on its own (vLLM) gives upstream's tower on the
    /// whole video (`THW(t, h, w)`, HF), and spliced at the group spans,
    /// with timestamp text between them, upstream's video prefill. Float32,
    /// to a tolerance: the tower runs the two as matrices of different
    /// heights. Whether the bits are equal is printed, not required.
    @Test func groupsEncodedAloneAreUpstreamsVideoTowerAndPrefill() throws {
        let (upstream, native) = try MuseGlimmerFixture.upstreamPair(dtype: .float32)
        let stack = try MuseGlimmerImageEncoderTests.stack(upstream)
        let frame = try Image.Grid(mergedRows: 2, mergedColumns: 3)
        let clip = try Video.Grid(groups: 3, frame: frame)
        let groups = (0 ..< 3).map {
            MuseGlimmerImageEncoderTests.pixels(frame, seed: 60 + UInt64($0))
        }
        let alone = groups.map { stack($0, grid: [frame.thw]) }
        let whole = stack(concatenated(groups, axis: 0), grid: [clip.thw])
        let joined = concatenated(alone, axis: 0)
        #expect(whole.shape == [clip.featureRows, 32] && joined.shape == whole.shape)
        let towerDrift = Self.maximumRelativeDifference(joined, whole)
        print(
            "MUSE_GLIMMER_VIDEO tower bits_equal=\(MuseGlimmerFixture.sameBits(joined, whole))"
                + " max_relative=\(towerDrift)")
        #expect(towerDrift <= 1e-5)

        // Fixture ids: 7 image, 8 video; 11, 12 and 13 stand in for the
        // start, separator and end tokens, 20 to 25 for the timestamps.
        var prompt: [Int32] = [1, 2, 11]
        var spans = [CBv2ImageSpan]()
        for group in 0 ..< 3 {
            prompt += [Int32(20 + 2 * group), Int32(21 + 2 * group)]
            spans.append(CBv2ImageSpan(tokenOffset: prompt.count, length: clip.tokensPerGroup))
            prompt += Array(repeating: 8, count: clip.tokensPerGroup)
            prompt.append(group < 2 ? 12 : 13)
        }
        prompt += MuseGlimmerFixture.tokens(10, vocabulary: 64, salt: 5).map {
            $0 == 7 || $0 == 8 ? $0 + 2 : $0
        }
        let ids = MLXArray(prompt).reshaped(1, -1)
        let result = try upstream.prepare(
            LMInput(
                text: .init(tokens: ids),
                image: .init(pixels: concatenated(groups, axis: 0), frames: [clip.thw])),
            cache: try upstream.newCache(parameters: nil), state: nil,
            prefill: PrefillParameters(chunking: .unchunked))
        guard case .logits(let output) = result else {
            Issue.record("expected logits")
            return
        }
        let adapter = MuseGlimmerCBv2Adapter(native)
        func logits(_ features: [MLXArray]) -> MLXArray {
            native(
                inputEmbeddings: CBv2MultimodalPlan.spliceEmbeddings(
                    textEmbeddings: adapter.embedPromptTokens(ids), chunkStart: 0,
                    spans: zip(spans, features).map { ($0, $1.expandedDimensions(axis: 0)) }),
                cache: nil)
        }
        let spliced = logits(alone)
        #expect(spliced.shape == output.logits.shape)
        let prefillDrift = Self.maximumRelativeDifference(spliced, output.logits)
        print(
            "MUSE_GLIMMER_VIDEO prefill bits_equal="
                + "\(MuseGlimmerFixture.sameBits(spliced, output.logits))"
                + " max_relative=\(prefillDrift)")
        #expect(prefillDrift <= 1e-4)
        // The groups are read in order: swapping two changes the logits.
        let swapped = logits([alone[1], alone[0], alone[2]])
        #expect(Self.maximumRelativeDifference(swapped, output.logits) > 1e-3)
    }

    /// The encoder's video features are its per-group image features, bit for
    /// bit, and a group count or pixel shape that disagrees is refused.
    @Test func videoFeaturesAreTheEncodersGroupFeatures() throws {
        let (upstream, _) = try MuseGlimmerFixture.upstreamPair(dtype: .bfloat16)
        let (sources, directory) = try MuseGlimmerImageEncoderTests.sources(
            MuseGlimmerImageEncoderTests.checkpointTensors(upstream))
        defer { try? FileManager.default.removeItem(at: directory) }
        let encoder = try MuseGlimmerImageEncoder.load(
            configuration: upstream.config, sources: sources)
        let frame = try Image.Grid(mergedRows: 2, mergedColumns: 2)
        let clip = try Video.Grid(groups: 2, frame: frame)
        let groups = (0 ..< 2).map {
            MuseGlimmerImageEncoderTests.pixels(frame, seed: 70 + UInt64($0))
        }
        let features = try encoder.videoFeatures(groups: groups, grid: clip)
        #expect(features.count == 2)
        for (group, pixels) in groups.enumerated() {
            let image = try encoder.features(pixels: pixels, grid: frame)
            #expect(features[group].shape == [clip.tokensPerGroup, 32])
            #expect(MuseGlimmerFixture.sameBits(features[group], image))
        }
        #expect(!MuseGlimmerFixture.sameBits(features[0], features[1]))
        #expect(throws: Failure.self) {
            try encoder.videoFeatures(groups: [groups[0]], grid: clip)
        }
        let wide = try Image.Grid(mergedRows: 2, mergedColumns: 3)
        #expect(throws: Failure.self) {
            try encoder.videoFeatures(
                groups: [groups[0], MuseGlimmerImageEncoderTests.pixels(wide, seed: 72)],
                grid: clip)
        }
    }

    static func maximumRelativeDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a = a.asType(.float32)
        let b = b.asType(.float32)
        let scale = max(abs(b).max().item(Float.self), .leastNormalMagnitude)
        return abs(a - b).max().item(Float.self) / scale
    }
}
