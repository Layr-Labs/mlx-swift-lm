import AVFoundation
import CoreImage
import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// Tests of `UserInput` that use no MLX arrays: the prompt kinds, the
    /// media lists that each initializer keeps, the `prompt` observer, the
    /// errors and the stand-in processor. The image array conversion needs
    /// MLX arrays and is in `Kernel/Common/UserInputImageArrayTests.swift`.
    @Suite
    struct UserInputPromptTests {

        private let imageA = UserInput.Image.url(URL(fileURLWithPath: "/nonexistent/a.png"))
        private let imageB = UserInput.Image.url(URL(fileURLWithPath: "/nonexistent/b.png"))
        private let video = UserInput.Video.url(URL(fileURLWithPath: "/nonexistent/v.mp4"))

        private func imagePaths(_ images: [UserInput.Image]) -> [String] {
            images.map {
                if case .url(let url) = $0 { return url.lastPathComponent }
                return "other"
            }
        }

        // MARK: - Prompt

        @Test func promptDescriptionForEachKind() {
            #expect(UserInput.Prompt.text("hello").description == "hello")
            #expect(
                UserInput.Prompt.chat([.system("be brief"), .user("hi")]).description
                    == "be brief\nhi")
            let messages: [Message] = [["role": "user"], ["content": "x"]]
            #expect(
                UserInput.Prompt.messages(messages).description
                    == "[\"role\": \"user\"]\n[\"content\": \"x\"]")
        }

        // MARK: - Initializers

        @Test func promptStringInitBuildsOneUserMessage() throws {
            let input = UserInput(
                prompt: "describe", images: [imageA], videos: [video],
                tools: [["type": "function"]], additionalContext: ["enable_thinking": false])
            guard case .chat(let messages) = input.prompt else {
                Issue.record("expected a chat prompt")
                return
            }
            #expect(messages.count == 1)
            #expect(messages[0].role == .user)
            #expect(messages[0].content == "describe")
            #expect(imagePaths(messages[0].images) == ["a.png"])
            #expect(messages[0].videos.count == 1)
            #expect(imagePaths(input.images) == ["a.png"])
            #expect(input.videos.count == 1)
            #expect(input.tools?.count == 1)
            #expect(input.additionalContext?["enable_thinking"] as? Bool == false)
            #expect(input.processing.resize == nil)
        }

        @Test func messagesInitKeepsTheGivenMedia() {
            let input = UserInput(
                messages: [["role": "user", "content": "hi"]], images: [imageA, imageB],
                videos: [video])
            guard case .messages(let messages) = input.prompt else {
                Issue.record("expected a messages prompt")
                return
            }
            #expect(messages.count == 1)
            #expect(imagePaths(input.images) == ["a.png", "b.png"])
            #expect(input.videos.count == 1)
            #expect(input.tools == nil)
            #expect(input.additionalContext == nil)
        }

        @Test func chatInitCollectsTheMediaOfAllMessages() {
            let processing = UserInput.Processing(
                resize: CGSize(width: 32, height: 16), minPixels: 10, maxPixels: 20)
            let input = UserInput(
                chat: [
                    .system("system"),
                    .user("first", images: [imageA]),
                    .assistant("answer"),
                    .user("second", images: [imageB], videos: [video]),
                ],
                processing: processing,
                tools: [["type": "function"], ["type": "function"]])
            #expect(imagePaths(input.images) == ["a.png", "b.png"])
            #expect(input.videos.count == 1)
            #expect(input.processing.resize == CGSize(width: 32, height: 16))
            #expect(input.processing.minPixels == 10)
            #expect(input.processing.maxPixels == 20)
            #expect(input.tools?.count == 2)
        }

        @Test func promptEnumInitKeepsGivenMediaForTextAndMessages() {
            let text = UserInput(
                prompt: .text("plain"), images: [imageA], videos: [video],
                processing: .init(maxPixels: 64))
            #expect(imagePaths(text.images) == ["a.png"])
            #expect(text.videos.count == 1)
            #expect(text.processing.maxPixels == 64)
            #expect(text.processing.minPixels == nil)

            let messages = UserInput(
                prompt: .messages([["role": "user"]]), images: [imageB], videos: [],
                additionalContext: ["k": 1])
            #expect(imagePaths(messages.images) == ["b.png"])
            #expect(messages.videos.isEmpty)
            #expect(messages.additionalContext?["k"] as? Int == 1)
        }

        /// For a chat prompt the media come from the messages. The `images`
        /// and `videos` arguments are not used.
        @Test func promptEnumInitTakesChatMediaFromTheMessages() {
            let input = UserInput(
                prompt: .chat([.user("x", images: [imageB])]), images: [imageA],
                videos: [video])
            #expect(imagePaths(input.images) == ["b.png"])
            #expect(input.videos.isEmpty)
        }

        // MARK: - prompt observer

        @Test func settingAChatPromptRebuildsTheMedia() {
            var input = UserInput(prompt: .text("start"), images: [imageA])
            input.prompt = .chat([
                .user("one", images: [imageB]),
                .user("two", images: [imageA], videos: [video]),
            ])
            #expect(imagePaths(input.images) == ["b.png", "a.png"])
            #expect(input.videos.count == 1)
        }

        @Test func settingATextOrMessagesPromptKeepsTheMedia() {
            var input = UserInput(chat: [.user("one", images: [imageA])])
            input.prompt = .text("now text")
            #expect(imagePaths(input.images) == ["a.png"])
            input.prompt = .messages([["role": "user"]])
            #expect(imagePaths(input.images) == ["a.png"])
        }

        // MARK: - Processing and media types

        @Test func processingDefaultsToNoOverrides() {
            let processing = UserInput.Processing()
            #expect(processing.resize == nil)
            #expect(processing.minPixels == nil)
            #expect(processing.maxPixels == nil)
        }

        @Test func videoFrameKeepsItsValues() {
            let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))
            let time = CMTime(value: 3, timescale: 2)
            let frame = UserInput.VideoFrame(frame: image, timeStamp: time)
            #expect(frame.frame.extent == CGRect(x: 0, y: 0, width: 2, height: 2))
            #expect(frame.timeStamp == time)
        }

        @Test func ciImageInputIsReturnedUnchanged() throws {
            let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 3, height: 1))
            let result = try UserInput.Image.ciImage(image).asCIImage()
            #expect(result === image)
        }

        @Test func unreadableImageURLThrowsUnableToLoad() {
            let url = URL(fileURLWithPath: "/nonexistent/folder/missing.png")
            do {
                _ = try UserInput.Image.url(url).asCIImage()
                Issue.record("expected an error")
            } catch UserInputError.unableToLoad(let failed) {
                #expect(failed == url)
            } catch {
                Issue.record("unexpected error \(error)")
            }
        }

        /// `asAVAsset()` is deprecated. The test keeps its two working cases
        /// covered until the function is removed.
        @Test func asAVAssetReturnsTheAssetForURLAndAsset() {
            let url = URL(fileURLWithPath: "/nonexistent/clip.mp4")
            let fromURL = UserInput.Video.url(url).asAVAsset()
            #expect((fromURL as? AVURLAsset)?.url == url)

            let asset = AVURLAsset(url: url)
            #expect(UserInput.Video.avAsset(asset).asAVAsset() === asset)
        }

        // MARK: - Errors and the stand-in processor

        @Test func errorDescriptions() {
            #expect(
                UserInputError.notImplemented.errorDescription
                    == "This functionality is not implemented.")
            #expect(
                UserInputError.unableToLoad(URL(fileURLWithPath: "/tmp/x.png")).errorDescription
                    == "Unable to load image from URL: /tmp/x.png.")
            #expect(
                UserInputError.arrayError("bad shape").errorDescription
                    == "Error processing image array: bad shape.")
        }

        @Test func standInProcessorThrowsNotImplemented() {
            let processor = StandInUserInputProcessor()
            do {
                _ = try processor.prepare(input: UserInput(prompt: "x"))
                Issue.record("expected an error")
            } catch UserInputError.notImplemented {
                // Expected.
            } catch {
                Issue.record("unexpected error \(error)")
            }
        }
    }
}
