import CoreImage
import CoreMedia
import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

extension KernelTests {

    /// `Qwen3VLProcessor` video against Transformers 5.8.0's own Qwen3.5
    /// processor (`Resources/qwen35-video-layout-golden.json`): the
    /// Qwen3.8 checkpoint's processor configs and tokenizer, solid clips of
    /// fixed frame count, rate and size, and per-request frame and pixel caps.
    /// The golden holds the expanded token ids, the video grid, the sampled
    /// frame indices, and the SHA-256 and delta of Qwen3.5's `get_rope_index`.
    @Suite
    struct Qwen3VLVideoReferenceLayoutTests {

        struct Golden: Decodable {
            struct Case: Decodable, CustomTestStringConvertible, Sendable {
                let name: String
                let frames: Int
                let fps: Double
                let height: Int
                let width: Int
                let maxFrames: Int
                let maxPixels: Int
                let sampledIndices: [Int]
                let grid: [Int]
                let tokenCount: Int
                let idsRle: [[Int]]
                let positionsSha256: String
                let delta: Int
                var testDescription: String { name }

                enum CodingKeys: String, CodingKey {
                    case name, frames, fps, height, width, grid, delta
                    case maxFrames = "max_frames"
                    case maxPixels = "max_pixels"
                    case sampledIndices = "sampled_indices"
                    case tokenCount = "token_count"
                    case idsRle = "ids_rle"
                    case positionsSha256 = "positions_sha256"
                }

                var ids: [Int] { idsRle.flatMap { Array(repeating: $0[0], count: $0[1]) } }
            }

            let renderedTemplateIds: [Int]
            let specialTokens: [String: Int]
            let segments: [String: [Int]]
            let cases: [Case]

            enum CodingKeys: String, CodingKey {
                case cases, segments
                case renderedTemplateIds = "rendered_template_ids"
                case specialTokens = "special_tokens"
            }
        }

        static let golden: Golden = {
            let url = Bundle.module.url(
                forResource: "qwen35-video-layout-golden", withExtension: "json")!
            return try! JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
        }()

        /// Transformers splits text at special tokens and tokenizes each
        /// segment on its own, so the golden's segments reproduce its ids for
        /// exactly these prompts. Text the golden lacks encodes to -1.
        struct GoldenTokenizer: MLXLMCommon.Tokenizer {
            let golden: Golden

            private var specials: [(String, Int)] {
                golden.specialTokens.sorted { $0.key.count > $1.key.count }.map {
                    ($0.key, $0.value)
                }
            }

            func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                var ids: [Int] = []
                var rest = Substring(text)
                while !rest.isEmpty {
                    if let special = specials.first(where: { rest.hasPrefix($0.0) }) {
                        ids.append(special.1)
                        rest = rest.dropFirst(special.0.count)
                        continue
                    }
                    var end = rest.startIndex
                    while end < rest.endIndex,
                        !specials.contains(where: { rest[end...].hasPrefix($0.0) })
                    {
                        end = rest.index(after: end)
                    }
                    ids += golden.segments[String(rest[..<end])] ?? [-1]
                    rest = rest[end...]
                }
                return ids
            }

            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                let names = Dictionary(uniqueKeysWithValues: golden.specialTokens.map { ($1, $0) })
                let pieces = golden.segments.sorted { $0.value.count > $1.value.count }
                var text = ""
                var index = 0
                while index < tokenIds.count {
                    if let name = names[tokenIds[index]] {
                        text += name
                        index += 1
                    } else if let piece = pieces.first(where: {
                        tokenIds[index...].starts(with: $0.value)
                    }) {
                        text += piece.key
                        index += piece.value.count
                    } else {
                        return text + "\u{FFFD}"
                    }
                }
                return text
            }

            func convertTokenToId(_ token: String) -> Int? { golden.specialTokens[token] }
            func convertIdToToken(_ id: Int) -> String? { nil }
            var bosToken: String? { nil }
            var eosToken: String? { nil }
            var unknownToken: String? { nil }

            func applyChatTemplate(
                messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                golden.renderedTemplateIds
            }
        }

        /// The Qwen3.8 checkpoint's `preprocessor_config.json`.
        static func configuration() throws -> Qwen3VLProcessorConfiguration {
            let object: [String: Any] = [
                "size": ["longest_edge": 16_777_216, "shortest_edge": 65_536],
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "image_processor_type": "Qwen2VLImageProcessorFast",
            ]
            return try JSONDecoder().decode(
                Qwen3VLProcessorConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }

        static func processor() throws -> Qwen3VLProcessor {
            Qwen3VLProcessor(try configuration(), tokenizer: GoldenTokenizer(golden: golden))
        }

        /// Solid frames at exact multiples of 1 / fps.
        static func clip(_ c: Golden.Case) -> UserInput.Video {
            let image = CIImage(color: CIColor(red: 0.2, green: 0.4, blue: 0.6))
                .cropped(to: CGRect(x: 0, y: 0, width: c.width, height: c.height))
            return .frames(
                (0 ..< c.frames).map {
                    UserInput.VideoFrame(
                        frame: image,
                        timeStamp: CMTime(value: Int64($0), timescale: CMTimeScale(c.fps)))
                })
        }

        static func sha256(_ values: [Int32]) -> String {
            let bytes = values.withUnsafeBytes { Data($0) }
            return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        }

        @Test(arguments: golden.cases)
        func promptGridAndPositionsMatchTransformers(_ c: Golden.Case) async throws {
            var input = UserInput(chat: [
                .user("What happens in the video?", videos: [Self.clip(c)])
            ])
            input.processing.video = .init(maxFrames: c.maxFrames, maxPixels: c.maxPixels)
            let result = try await Self.processor().prepare(input: input)

            let ids = result.text.tokens.asArray(Int32.self).map(Int.init)
            #expect(ids.count == c.tokenCount, "\(c.name): token count")
            #expect(ids == c.ids, "\(c.name): token ids")
            let grids = try #require(result.video?.frames, "\(c.name): video grid")
            #expect(grids.map { [$0.t, $0.h, $0.w] } == [c.grid], "\(c.name): grid")

            let (positions, delta) = Qwen3VLLanguage.getRopeIndex(
                inputIds: result.text.tokens, imageGridTHW: nil, videoGridTHW: grids,
                spatialMergeSize: 2, imageTokenId: 248_056, videoTokenId: 248_057,
                visionStartTokenId: 248_053)
            #expect(
                Self.sha256(positions.asType(.int32).asArray(Int32.self)) == c.positionsSha256,
                "\(c.name): positions")
            #expect(delta.asArray(Int32.self) == [Int32(c.delta)], "\(c.name): delta")
        }

        @Test(arguments: golden.cases)
        func sampledFramesMatchTransformers(_ c: Golden.Case) throws {
            let indices = try Qwen4ExpMediaGeometry.sampleIndices(
                totalFrames: c.frames, sourceFPS: c.fps, targetFPS: 2, minFrames: 4,
                maxFrames: c.maxFrames)
            #expect(indices == c.sampledIndices, "\(c.name): sampled frames")
        }

        /// The seam validates per-group runs against the clip's grid and gives
        /// the same positions as `getRopeIndex`.
        @Test func seamPositionsAcceptTimestampedBlocks() async throws {
            let c = try #require(Self.golden.cases.first)
            var input = UserInput(chat: [
                .user("What happens in the video?", videos: [Self.clip(c)])
            ])
            input.processing.video = .init(maxFrames: c.maxFrames, maxPixels: c.maxPixels)
            let result = try await Self.processor().prepare(input: input)
            let seam = Qwen35VisionSeamConfiguration(
                imagePlaceholderTokenId: 248_056, videoPlaceholderTokenId: 248_057,
                imagePositionTokenId: 248_056, videoPositionTokenId: 248_057,
                visionStartTokenId: 248_053, visionEndTokenId: 248_054,
                spatialMergeSize: 2, temporalPatchSize: 2, attention: .causal)
            let position = try QwenVisionSeamSupport.positionResult(
                tokens: result.text.tokens, imageGrids: nil, videoGrids: result.video?.frames,
                attentionMask: result.text.mask, seam: seam)
            #expect(
                Self.sha256(position.promptPositionIds.asType(.int32).asArray(Int32.self))
                    == c.positionsSha256, "seam positions")
            #expect(position.decodeState.deltas == [Int32(c.delta)], "seam delta")
        }

        /// The pair media contract keeps one vision block per clip.
        @Test func mediaProcessorKeepsTheSingleBlockLayout() async throws {
            let c = try #require(Self.golden.cases.first)
            let base = try Self.processor()
            #expect(base.videoLayout == .timestamped, "the factory layout")
            let pair = Qwen3VLMediaProcessor(try Self.configuration())
                .makeUserInputProcessor(tokenizer: GoldenTokenizer(golden: Self.golden))
            #expect(pair.videoLayout == .singleBlock, "the pair layout")
            let input = UserInput(chat: [
                .user("What happens in the video?", videos: [Self.clip(c)])
            ])
            let result = try await pair.prepare(input: input)
            let ids = result.text.tokens.asArray(Int32.self).map(Int.init)
            #expect(ids.filter { $0 == 248_053 }.count == 1, "one vision block")
            #expect(result.video?.frames?.map(\.t) == [Qwen3VLProcessor.maxVideoFrames / 2])
        }

        // MARK: - Position grids

        static func positionGrids(_ grids: [THW], _ tokens: [Int]) -> [[Int]] {
            QwenVL.positionVideoGrids(grids, tokens: tokens, videoToken: 9, merge: 2)
                .map { [$0.t, $0.h, $0.w] }
        }

        @Test func positionGridsSplitSeparateGroupRuns() {
            #expect(
                Self.positionGrids([THW(2, 2, 4)], [1, 9, 9, 2, 3, 9, 9, 2, 4])
                    == [[1, 2, 4], [1, 2, 4]], "two runs of 2 tokens: two t = 1 grids")
        }

        @Test func positionGridsKeepASingleRun() {
            #expect(
                Self.positionGrids([THW(2, 2, 4)], [1, 9, 9, 9, 9, 2]) == [[2, 2, 4]],
                "one run of t × 2 tokens keeps its grid")
            #expect(
                Self.positionGrids([THW(1, 2, 4)], [9, 9]) == [[1, 2, 4]], "t = 1 is unchanged")
        }

        @Test func positionGridsFollowEachClip() {
            // Clip A in one run (t 2 × 2 tokens), clip B in two group runs.
            #expect(
                Self.positionGrids([THW(2, 2, 4), THW(2, 2, 4)], [9, 9, 9, 9, 1, 9, 9, 2, 9, 9])
                    == [[2, 2, 4], [1, 2, 4], [1, 2, 4]])
        }
    }
}
