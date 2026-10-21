// Expected values from NVIDIA's processing/image_processing_nemotron_h_omni.py,
// vLLM's uniform video loader (NumPy 2.3 linspace) and vLLM evs.py under PyTorch 2.12.
import Testing

@testable import MLXVLM

extension UnitTests {
    struct NemotronSuperVideoPreprocessingTests {
        private typealias Video = NemotronSuperVideoPreprocessing

        @Test func samplesTwoFramesPerSecondWithTruncatingLinspace() throws {
            // 6 s at 30 fps: 179 / 11 * 3 = 48.8 truncates to 48 (rounding gives 49).
            #expect(
                try Video.sampledFrameIndices(totalFrames: 180, sourceFPS: 30) == [
                    0, 16, 32, 48, 65, 81, 97, 113, 130, 146, 162, 179,
                ])
            // 10.01 s: floor(20.02) frames.
            #expect(
                try Video.sampledFrameIndices(totalFrames: 300, sourceFPS: 29.97) == [
                    0, 15, 31, 47, 62, 78, 94, 110, 125, 141, 157, 173, 188, 204, 220, 236, 251,
                    267, 283, 299,
                ])
            #expect(
                try Video.sampledFrameIndices(totalFrames: 240, sourceFPS: 23.976) == [
                    0, 12, 25, 37, 50, 62, 75, 88, 100, 113, 125, 138, 150, 163, 176, 188, 201,
                    213, 226, 239,
                ])
        }

        @Test func longVideosStopAtTheFrameCap() throws {
            // 60.03 s and 61 s stay below the cap: floor(duration * 2) frames.
            #expect(try Video.sampledFrameIndices(totalFrames: 1801, sourceFPS: 30).count == 120)
            let minute = try Video.sampledFrameIndices(totalFrames: 1830, sourceFPS: 30)
            #expect(minute.count == 122)
            #expect(Array(minute.prefix(10)) == [0, 15, 30, 45, 60, 75, 90, 105, 120, 136])
            #expect(Array(minute.suffix(3)) == [1798, 1813, 1829])
            // 129 s would sample 258 frames; 256 spread over the whole video.
            let capped = try Video.sampledFrameIndices(totalFrames: 3870, sourceFPS: 30)
            #expect(capped.count == Video.maximumFrames)
            #expect(Array(capped.prefix(8)) == [0, 15, 30, 45, 60, 75, 91, 106])
            #expect(Array(capped[127 ..< 130]) == [1926, 1942, 1957])
            #expect(Array(capped.suffix(3)) == [3838, 3853, 3869])
        }

        @Test func shortAndSlowVideos() throws {
            // Under half a second still samples the first frame.
            #expect(try Video.sampledFrameIndices(totalFrames: 10, sourceFPS: 30) == [0])
            #expect(try Video.sampledFrameIndices(totalFrames: 7, sourceFPS: 24) == [0])
            #expect(try Video.sampledFrameIndices(totalFrames: 1, sourceFPS: 30) == [0])
            // At most two frames per second means every frame of a slow source.
            #expect(try Video.sampledFrameIndices(totalFrames: 10, sourceFPS: 2) == Array(0 ..< 10))
            #expect(try Video.sampledFrameIndices(totalFrames: 10, sourceFPS: 1) == Array(0 ..< 10))
        }

        @Test func rejectsEmptyVideosAndInvalidFrameRates() {
            for frames in [0, -1] {
                #expect(throws: Video.Failure.emptyVideo) {
                    try Video.sampledFrameIndices(totalFrames: frames, sourceFPS: 30)
                }
            }
            for fps in [0, -30, Double.nan, .infinity, -.infinity] {
                #expect(throws: Video.Failure.invalidFrameRate) {
                    try Video.sampledFrameIndices(totalFrames: 30, sourceFPS: fps)
                }
            }
        }

        @Test func timestampsTruncateTheFrameDurationToMilliseconds() {
            // 1799 * 33 ms = 59.367 s, not 1799 / 30 = 59.967 s.
            #expect(Video.timestamp(frameIndex: 1799, sourceFPS: 30) == "59.37")
            #expect(Video.timestamp(frameIndex: 0, sourceFPS: 30) == "0.00")
            #expect(Video.timestamp(frameIndex: 15, sourceFPS: 30) == "0.49")
            #expect(Video.timestamp(frameIndex: 29, sourceFPS: 29.97) == "0.96")
            #expect(Video.timestamp(frameIndex: 1, sourceFPS: 0.5) == "2.00")
            // Exact binary ties round to even, as Python's "%.2f" does.
            #expect(Video.timestamp(frameIndex: 1, sourceFPS: 8) == "0.12")
            #expect(Video.timestamp(frameIndex: 3, sourceFPS: 8) == "0.38")
            #expect(Video.timestamp(frameIndex: 5, sourceFPS: 0) == "0.00")
            // Over 1000 fps a frame truncates to 0 ms; Python's integer zero has no sign.
            #expect(Video.timestamp(frameIndex: -3, sourceFPS: 2000) == "0.00")
        }

        @Test func timestampsOfExtremeRatesMatchPythonOrPrintZero() {
            // Accepted rates far below one frame per second: a 1e15 ms frame, and
            // a product beyond Int64 that Python's float still represents.
            #expect(Video.timestamp(frameIndex: 7, sourceFPS: 1e-12) == "7000000000000.00")
            #expect(
                Video.timestamp(frameIndex: 10_000_000, sourceFPS: 1e-9)
                    == "9999999999990000.00")
            // Rates that sampling rejects, and durations or times past the float
            // range, where Python raises.
            for fps in [0, -30, Double.nan, .infinity, -.infinity] {
                for frame in [0, 5] {
                    #expect(Video.timestamp(frameIndex: frame, sourceFPS: fps) == "0.00")
                }
            }
            #expect(Video.timestamp(frameIndex: 0, sourceFPS: .leastNonzeroMagnitude) == "0.00")
            #expect(Video.timestamp(frameIndex: 2, sourceFPS: 1e-305) == "0.00")
        }

        @Test func tubeletLabelsPairFramesAndKeepAnOddLastFrame() throws {
            #expect(
                Video.tubeletLabels(frameIndices: [0, 16, 32, 48], sourceFPS: 30) == [
                    "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.53 seconds: ",
                    "Frame 3 sampled at 1.06 seconds and frame 4 sampled at 1.58 seconds: ",
                ])
            #expect(
                Video.tubeletLabels(frameIndices: [0, 15, 31], sourceFPS: 30) == [
                    "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.49 seconds: ",
                    "Frame 3 sampled at 1.02 seconds: ",
                ])
            #expect(
                Video.tubeletLabels(frameIndices: [0], sourceFPS: 30) == [
                    "Frame 1 sampled at 0.00 seconds: "
                ])
            #expect(Video.tubeletLabels(frameIndices: [], sourceFPS: 30).isEmpty)
            let six = try Video.sampledFrameIndices(totalFrames: 180, sourceFPS: 30)
            let labels = Video.tubeletLabels(frameIndices: six, sourceFPS: 30)
            #expect(labels.count == 6)
            #expect(
                labels.last
                    == "Frame 11 sampled at 5.35 seconds and frame 12 sampled at 5.91 seconds: ")
        }

        @Test func frameGridTargetsAboutOneThousandPatches() throws {
            let cases: [(width: Int, height: Int, columns: Int, rows: Int)] = [
                (1280, 720, 42, 24), (1920, 1080, 42, 24), (3840, 2160, 42, 24),
                (720, 1280, 24, 42), (640, 480, 36, 28), (480, 640, 28, 36),
                (512, 512, 32, 32), (16, 16, 32, 32), (1, 1, 32, 32), (2, 1, 44, 22),
                // Both sides round up to even values: 1040 patches, as upstream.
                (1000, 650, 40, 26),
                (10000, 1, 3200, 2),
            ]
            for (width, height, columns, rows) in cases {
                let grid = try Video.frameGrid(width: width, height: height)
                #expect(
                    grid.widthPatches == columns && grid.heightPatches == rows,
                    "\(width)x\(height)")
                #expect(grid.width == columns * 16 && grid.height == rows * 16)
                #expect(grid.tokens == columns * rows / 4)
            }
            let wide = try Video.frameGrid(width: 1280, height: 720)
            #expect(wide.width == 672 && wide.height == 384 && wide.tokens == 252)
            #expect(try Video.frameGrid(width: 512, height: 512).tokens == 256)
            #expect(try Video.frameGrid(width: 720, height: 1280).tokens == 252)
            #expect(try Video.frameGrid(width: 640, height: 480).tokens == 252)
        }

        @Test func frameGridRejectsDegenerateFrames() {
            // 100000 x 1 needs 10118 x 2 patches, beyond the tower's limit.
            for (width, height) in [(0, 720), (1280, 0), (-1, 5), (100_000, 1), (1, 100_000)] {
                #expect(throws: Video.Failure.invalidGeometry) {
                    try Video.frameGrid(width: width, height: height)
                }
            }
        }

        // Scores of tubelets 1 and 2 against their predecessors, as evs.py computes
        // them for the fixture in NemotronSuperEVSTests.
        private let scores: [Float] = [0, 1, 0, 0.04, 2, 0, 1, 0.4]

        @Test func evsKeepsTheFirstTubeletAndTheMostChangedTokens() {
            let cases: [(rate: Double, kept: [[Int]])] = [
                (0.5, [[0, 1, 2, 3], [1], [0]]),
                // Equal scores keep the earlier tubelet and position first.
                (0.4, [[0, 1, 2, 3], [1], [0, 2]]),
                (0.25, [[0, 1, 2, 3], [1, 3], [0, 2, 3]]),
                (0.1, [[0, 1, 2, 3], [0, 1, 3], [0, 2, 3]]),
                (0.0001, [[0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3]]),
                (1, [[0, 1, 2, 3], [], []]),
                (2, [[0, 1, 2, 3], [], []]),
            ]
            for (rate, expected) in cases {
                let kept = NemotronSuperEVS.keptPositions(
                    dissimilarity: scores, tubelets: 3, tokens: 4, pruningRate: rate)
                #expect(kept == expected, "rate \(rate)")
            }
        }

        @Test func evsDisabledRatesKeepEveryToken() {
            let all = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
            for rate in [0, -0.5, Double.nan, .infinity] {
                #expect(
                    NemotronSuperEVS.keptPositions(
                        dissimilarity: scores, tubelets: 3, tokens: 4, pruningRate: rate) == all)
            }
            #expect(
                NemotronSuperEVS.keptPositions(
                    dissimilarity: [], tubelets: 1, tokens: 4, pruningRate: 0.5) == [[0, 1, 2, 3]])
            #expect(
                NemotronSuperEVS.keptPositions(
                    dissimilarity: [1], tubelets: 3, tokens: 4, pruningRate: 0.5
                ).isEmpty)
        }

        @Test func evsRejectsUnrepresentableShapesWithoutTrapping() {
            // T * N and (T - 1) * N beyond Int are invalid input, not a trap.
            for (tubelets, tokens) in [(Int.max, 2), (3, Int.max / 2 + 1), (2, Int.max)] {
                #expect(
                    NemotronSuperEVS.retainedCount(
                        tubelets: tubelets, tokens: tokens, pruningRate: 0.5) == nil)
                #expect(
                    NemotronSuperEVS.keptPositions(
                        dissimilarity: [], tubelets: tubelets, tokens: tokens, pruningRate: 0.5
                    ).isEmpty)
            }
            for (tubelets, tokens) in [(0, 4), (4, 0), (-1, 4), (4, -1)] {
                #expect(
                    NemotronSuperEVS.retainedCount(
                        tubelets: tubelets, tokens: tokens, pruningRate: 0.5) == nil)
                #expect(
                    NemotronSuperEVS.keptPositions(
                        dissimilarity: [], tubelets: tubelets, tokens: tokens, pruningRate: 0.5
                    ).isEmpty)
            }
            // Python's int(T * N * (1 - q)) is 2^63 here, beyond Int.
            #expect(
                NemotronSuperEVS.retainedCount(tubelets: Int.max, tokens: 1, pruningRate: 1e-20)
                    == nil)
            #expect(
                NemotronSuperEVS.retainedCount(tubelets: Int.max, tokens: 1, pruningRate: 0.5)
                    == 1 << 62)
        }

        @Test func evsRetainedCountFormula() {
            // max(N, int(T * N * (1 - q))), with T * N * 0.6 = 7.199999999999999 here.
            #expect(NemotronSuperEVS.retainedCount(tubelets: 3, tokens: 4, pruningRate: 0.4) == 7)
            #expect(
                NemotronSuperEVS.retainedCount(tubelets: 128, tokens: 252, pruningRate: 0.5)
                    == 16128)
            #expect(
                NemotronSuperEVS.retainedCount(tubelets: 1, tokens: 252, pruningRate: 0.5) == 252)
            #expect(
                NemotronSuperEVS.retainedCount(tubelets: 3, tokens: 256, pruningRate: 0.9) == 256)
            #expect(NemotronSuperEVS.retainedCount(tubelets: 3, tokens: 256, pruningRate: 0) == 768)
            let tubelets = 5
            let tokens = 7
            let ramp = (0 ..< (tubelets - 1) * tokens).map { Float(($0 * 37) % 11) / 8 }
            for rate in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
                let kept = NemotronSuperEVS.keptPositions(
                    dissimilarity: ramp, tubelets: tubelets, tokens: tokens, pruningRate: rate)
                #expect(kept.count == tubelets && kept[0] == Array(0 ..< tokens))
                #expect(
                    kept.reduce(0) { $0 + $1.count }
                        == max(tokens, Int(Double(tubelets * tokens) * (1 - rate))))
                #expect(kept.allSatisfy { $0 == $0.sorted() && Set($0).count == $0.count })
            }
        }

        @Test func evsRanksNaNScoresFirstButNeverDropsTheFirstTubelet() {
            let kept = NemotronSuperEVS.keptPositions(
                dissimilarity: [0.5, .nan, 0.25, 1], tubelets: 3, tokens: 2, pruningRate: 0.5)
            #expect(kept == [[0, 1], [1], []])
        }
    }
}
