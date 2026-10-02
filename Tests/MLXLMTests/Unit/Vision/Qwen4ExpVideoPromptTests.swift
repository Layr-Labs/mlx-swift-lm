import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

extension UnitTests {

    /// `Qwen4ExpVideoPrompt.expandedTokens` and `positionGrids`.
    ///
    /// The tokenizer gives one token per Unicode scalar, so a prompt
    /// round-trips and the expected tokens are the scalars of the expected
    /// prompt text.
    @Suite
    struct Qwen4ExpVideoPromptUnitTests {

        /// One token per Unicode scalar. An ID that is not a valid scalar
        /// decodes to nothing, so it does not round-trip.
        private struct ScalarTokenizer: Tokenizer {
            let bosToken: String? = nil
            let eosToken: String? = nil
            let unknownToken: String? = nil

            func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                text.unicodeScalars.map { Int($0.value) }
            }

            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                var text = ""
                text.unicodeScalars.append(
                    contentsOf: tokenIds.compactMap { Unicode.Scalar(UInt32($0)) })
                return text
            }

            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }

            func applyChatTemplate(
                messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] { [] }
        }

        private let tokenizer = ScalarTokenizer()
        private let start = "<|vision_start|>"
        private let pad = "<|video_pad|>"
        private let end = "<|vision_end|>"

        private func tokens(_ text: String) -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: false)
        }

        private func expand(
            _ text: String, grids: [THW], timestamps: [[Double]], merge: Int = 2
        ) throws -> [Int] {
            try Qwen4ExpVideoPrompt.expandedTokens(
                tokens(text), tokenizer: tokenizer, grids: grids, timestamps: timestamps,
                merge: merge)
        }

        @Test
        func wrappedPlaceholderExpandsToOneBlockPerTemporalGroup() throws {
            // Grid 2x4x4 with merge 2 gives 2 * 2 = 4 tokens for each
            // temporal group. The start and end tokens of the template are
            // replaced, not kept twice.
            let result = try expand(
                "a\(start)\(pad)\(end)b", grids: [THW(2, 4, 4)], timestamps: [[0.5, 1.5]])
            let block = start + String(repeating: pad, count: 4) + end
            let expected = "a<0.5 seconds>\(block)<1.5 seconds>\(block)b"
            #expect(result == tokens(expected), "two timestamped blocks replace the slot")
        }

        @Test
        func barePlaceholderGetsStartAndEndTokens() throws {
            // Grid 1x2x6 with merge 2 gives 1 * 3 = 3 tokens.
            let result = try expand("x\(pad)y", grids: [THW(1, 2, 6)], timestamps: [[12.0]])
            let expected = "x<12.0 seconds>\(start)\(String(repeating: pad, count: 3))\(end)y"
            #expect(result == tokens(expected), "a bare slot gets a full block")
        }

        @Test
        func twoClipsExpandInPromptOrder() throws {
            let result = try expand(
                "\(pad)and\(pad)", grids: [THW(1, 2, 2), THW(1, 4, 2)],
                timestamps: [[0.0], [3.0]])
            let expected =
                "<0.0 seconds>\(start)\(pad)\(end)and"
                + "<3.0 seconds>\(start)\(pad)\(pad)\(end)"
            #expect(result == tokens(expected), "each clip fills its own slot")
        }

        @Test
        func templateThatDoesNotRoundTripIsRejected() {
            // 0xD800 is not a valid scalar, so decode drops it.
            let input = [0xD800] + tokens(pad)
            #expect(
                throws: VLMError.processing(
                    "Qwen4 video prompt could not preserve its rendered template")
            ) {
                try Qwen4ExpVideoPrompt.expandedTokens(
                    input, tokenizer: tokenizer, grids: [THW(1, 2, 2)], timestamps: [[0]],
                    merge: 2)
            }
        }

        @Test
        func mismatchedGridAndTimestampListsAreRejected() {
            let message = "Qwen4 video prompt could not preserve its rendered template"
            #expect(throws: VLMError.processing(message), "two grids, one timestamp list") {
                try expand(pad, grids: [THW(1, 2, 2), THW(1, 2, 2)], timestamps: [[0]])
            }
            #expect(throws: VLMError.processing(message), "merge size zero") {
                try expand(pad, grids: [THW(1, 2, 2)], timestamps: [[0]], merge: 0)
            }
        }

        @Test
        func placeholderCountMustMatchClips() {
            #expect(
                throws: VLMError.processing(
                    "Qwen4 video placeholder count does not match supplied clips")
            ) {
                try expand(
                    "one \(pad)", grids: [THW(1, 2, 2), THW(1, 2, 2)], timestamps: [[0], [1]])
            }
        }

        @Test
        func gridMustAgreeWithMergeAndTimestamps() {
            let message = "Qwen4 video grid and timestamps disagree"
            #expect(throws: VLMError.processing(message), "height 3 is not a multiple of 2") {
                try expand(pad, grids: [THW(1, 3, 4)], timestamps: [[0]])
            }
            #expect(throws: VLMError.processing(message), "t = 2 needs two timestamps") {
                try expand(pad, grids: [THW(2, 2, 2)], timestamps: [[0]])
            }
            #expect(throws: VLMError.processing(message), "t = 0 is not a valid grid") {
                try expand(pad, grids: [THW(0, 2, 2)], timestamps: [[]])
            }
        }

        @Test
        func startTokenWithoutEndTokenIsMalformed() {
            #expect(throws: VLMError.processing("Qwen4 video placeholder is malformed")) {
                try expand("\(start)\(pad)z", grids: [THW(1, 2, 2)], timestamps: [[0]])
            }
        }

        @Test
        func invalidTimestampsAreRejected() {
            #expect(
                throws: Qwen4ExpMediaGeometry.Failure.invalidFrameMetadata, "negative timestamp"
            ) {
                try expand(pad, grids: [THW(1, 2, 2)], timestamps: [[-1]])
            }
            #expect(
                throws: Qwen4ExpMediaGeometry.Failure.invalidFrameMetadata, "NaN timestamp"
            ) {
                try expand(pad, grids: [THW(1, 2, 2)], timestamps: [[.nan]])
            }
        }

        @Test
        func positionGridsSplitEachTemporalGroup() throws {
            #expect(try Qwen4ExpVideoPrompt.positionGrids(nil) == nil, "nil stays nil")
            let grids = try #require(
                try Qwen4ExpVideoPrompt.positionGrids([THW(3, 2, 4), THW(1, 6, 8)]))
            #expect(
                grids.map { [$0.t, $0.h, $0.w] }
                    == [[1, 2, 4], [1, 2, 4], [1, 2, 4], [1, 6, 8]],
                "one t = 1 grid for each temporal group, in order")
            #expect(
                try Qwen4ExpVideoPrompt.positionGrids([])?.isEmpty == true,
                "an empty list stays empty")
        }

        @Test
        func positionGridsRejectInvalidGrids() {
            #expect(throws: Qwen4ExpMediaGeometry.Failure.invalidGeometry, "t = 0") {
                try Qwen4ExpVideoPrompt.positionGrids([THW(0, 2, 2)])
            }
            #expect(throws: Qwen4ExpMediaGeometry.Failure.invalidGeometry, "w = 0") {
                try Qwen4ExpVideoPrompt.positionGrids([THW(1, 2, 0)])
            }
            #expect(throws: Qwen4ExpMediaGeometry.Failure.overflow, "t * h * w overflows") {
                try Qwen4ExpVideoPrompt.positionGrids([THW(Int.max, 2, 2)])
            }
        }
    }

    /// Decoding and encoding of `Qwen4ExpVideoConfiguration`.
    @Suite
    struct Qwen4ExpVideoConfigurationUnitTests {

        private func decode(_ object: [String: Any]) throws -> Qwen4ExpVideoConfiguration {
            try JSONDecoder().decode(
                Qwen4ExpVideoConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }

        private var valid: [String: Any] {
            [
                "size": ["shortest_edge": 64, "longest_edge": 4096],
                "patch_size": 2, "merge_size": 2, "temporal_patch_size": 2,
                "image_mean": [0.25, 0.5, 0.75], "image_std": [0.5, 0.25, 0.125],
                "fps": 3, "min_frames": 2, "max_frames": 9,
                "video_processor_type": "Qwen3VLVideoProcessor",
            ]
        }

        @Test
        func missingFieldsUseReferenceDefaults() throws {
            let config = try decode([:])
            #expect(config.size.shortestEdge == 128 * 32 * 32, "default shortest edge")
            #expect(config.size.longestEdge == 32 * 32 * 768, "default longest edge")
            #expect(config.patchSize == 16, "default patch size")
            #expect(config.mergeSize == 2, "default merge size")
            #expect(config.temporalPatchSize == 2, "default temporal patch size")
            #expect(config.imageMean == [0.5, 0.5, 0.5], "default mean")
            #expect(config.imageStd == [0.5, 0.5, 0.5], "default std")
            #expect(config.fps == 2, "default fps")
            #expect(config.minFrames == 4, "default min frames")
            #expect(config.maxFrames == 768, "default max frames")
        }

        @Test
        func explicitFieldsDecodeAndRoundTrip() throws {
            let config = try decode(valid)
            #expect(config.size.shortestEdge == 64, "shortest edge")
            #expect(config.size.longestEdge == 4096, "longest edge")
            #expect(config.patchSize == 2, "patch size")
            #expect(config.imageMean == [0.25, 0.5, 0.75], "mean")
            #expect(config.imageStd == [0.5, 0.25, 0.125], "std")
            #expect(config.fps == 3, "fps")
            #expect(config.minFrames == 2, "min frames")
            #expect(config.maxFrames == 9, "max frames")

            let data = try JSONEncoder().encode(config)
            let object = try #require(
                try JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(
                object["video_processor_type"] as? String == "Qwen3VLVideoProcessor",
                "encode writes the processor type")
            let again = try JSONDecoder().decode(Qwen4ExpVideoConfiguration.self, from: data)
            #expect(again.size.longestEdge == 4096, "round trip longest edge")
            #expect(again.imageStd == config.imageStd, "round trip std")
            #expect(again.maxFrames == 9, "round trip max frames")
        }

        @Test(arguments: [
            "video_processor_type", "size", "size_zero", "patch_size", "merge_size",
            "temporal_patch_size", "image_mean", "image_std", "fps", "min_frames",
            "max_frames", "patch_overflow",
        ])
        func invalidFieldIsRejected(field: String) throws {
            var object = valid
            switch field {
            case "video_processor_type": object[field] = "OtherProcessor"
            case "size": object[field] = ["shortest_edge": 4096, "longest_edge": 64]
            case "size_zero": object["size"] = ["shortest_edge": 0, "longest_edge": 64]
            case "patch_size", "merge_size", "temporal_patch_size", "min_frames":
                object[field] = 0
            case "image_mean": object[field] = [0.5, 0.5]
            case "image_std": object[field] = [0.5, 0, 0.5]
            case "fps": object[field] = 0
            case "max_frames": object[field] = 1
            default: object["patch_size"] = Int.max
            }
            #expect(throws: (any Error).self, "field \(field)") { try decode(object) }
        }

        @Test
        func processorRejectsVideoGeometryThatDiffersFromImageGeometry() throws {
            var object: [String: Any] = [
                "size": ["shortest_edge": 64, "longest_edge": 4096],
                "patch_size": 2, "merge_size": 2, "temporal_patch_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "image_processor_type": "Qwen2VLImageProcessorFast",
                "video_processor": valid,
            ]
            let config = try JSONDecoder().decode(
                Qwen4ExpProcessorConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
            #expect(config.video?.fps == 3, "nested video configuration decodes")

            var video = valid
            video["patch_size"] = 4
            object["video_processor"] = video
            #expect(throws: (any Error).self, "video patch 4 differs from image patch 2") {
                try JSONDecoder().decode(
                    Qwen4ExpProcessorConfiguration.self,
                    from: JSONSerialization.data(withJSONObject: object))
            }
        }
    }

    /// `Qwen4ExpProcessorFiles.combined`: the order in which the processor
    /// files are read.
    @Suite
    struct Qwen4ExpProcessorFilesUnitTests {

        private func withFolder(_ body: (URL) throws -> Void) throws {
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen4-processor-files-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: folder, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: folder) }
            try body(folder)
        }

        private func write(_ object: Any, _ name: String, in folder: URL) throws {
            try JSONSerialization.data(withJSONObject: object)
                .write(to: folder.appendingPathComponent(name))
        }

        private func combined(_ folder: URL, fallback: Any = ["a": 1]) throws -> NSDictionary {
            let data = try Qwen4ExpProcessorFiles.combined(
                directory: folder, fallbackImage: JSONSerialization.data(withJSONObject: fallback))
            return try #require(try JSONSerialization.jsonObject(with: data) as? NSDictionary)
        }

        @Test
        func emptyFolderGivesFallbackImageAndNullVideo() throws {
            try withFolder { folder in
                let result = try combined(folder)
                #expect(
                    result == ["a": 1, "video_processor": NSNull()] as NSDictionary,
                    "no files: fallback image and a null video entry")
            }
        }

        @Test
        func nestedProcessorEntriesComeFirst() throws {
            try withFolder { folder in
                try write(
                    ["image_processor": ["b": 2], "video_processor": ["v": 3]],
                    "processor_config.json", in: folder)
                try write(["v": 4], "video_preprocessor_config.json", in: folder)
                let result = try combined(folder)
                #expect(
                    result == ["b": 2, "video_processor": ["v": 3]] as NSDictionary,
                    "nested image and video entries win over the other files")
            }
        }

        @Test
        func separateVideoFileComesBeforeImageFile() throws {
            try withFolder { folder in
                try write(["v": 4], "video_preprocessor_config.json", in: folder)
                try write(["p": 5], "preprocessor_config.json", in: folder)
                let result = try combined(folder)
                #expect(
                    result == ["a": 1, "video_processor": ["v": 4]] as NSDictionary,
                    "the video file is the video entry")
            }
        }

        @Test
        func imageFileIsTheLastVideoFallback() throws {
            try withFolder { folder in
                try write(["p": 5], "preprocessor_config.json", in: folder)
                try write(["unrelated": true], "processor_config.json", in: folder)
                let result = try combined(folder)
                #expect(
                    result == ["a": 1, "video_processor": ["p": 5]] as NSDictionary,
                    "the image preprocessor file is the video entry")
            }
        }

        @Test
        func entriesThatAreNotObjectsAreRejected() throws {
            try withFolder { folder in
                #expect(
                    throws: Qwen4ExpMediaGeometry.Failure.invalidGeometry,
                    "fallback image is an array"
                ) {
                    try combined(folder, fallback: [1])
                }
                try write(["image_processor": 1], "processor_config.json", in: folder)
                #expect(
                    throws: Qwen4ExpMediaGeometry.Failure.invalidGeometry,
                    "nested image entry is a number"
                ) {
                    try combined(folder)
                }
                try write(["video_processor": "text"], "processor_config.json", in: folder)
                #expect(
                    throws: Qwen4ExpMediaGeometry.Failure.invalidGeometry,
                    "nested video entry is a string"
                ) {
                    try combined(folder)
                }
            }
        }
    }
}
