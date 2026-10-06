import Foundation
import Testing

@testable import MLXLMServer

@Suite("Responses image input preservation")
struct OpenAIResponsesMediaTests {
    @Test func standardImagesKeepTheirBytesAndOrderThroughChatTranslation() throws {
        let urls = ["data:image/png;base64,AA+/==", "https://example.invalid/image.png?version=1&token=fixture"]
        let parts: [[String: Any]] = [
            ["type": "input_text", "text": "First image:"],
            ["type": "input_image", "image_url": urls[0]],
            ["type": "input_text", "text": "Second image:"],
            ["type": "input_image", "image_url": urls[1]],
        ]
        let body = try JSONSerialization.data(withJSONObject: [
            "model": "fixture", "input": [["role": "user", "content": parts]],
            "reasoning": ["effort": "none"],
        ])
        let response = try JSONDecoder().decode(OpenAIResponseRequest.self, from: body)
        let chat = response.chatCompletionRequest
        #expect(chat.reasoning?.effort == "none")
        #expect(chat.messages.count == 1)
        #expect(chat.messages[0].content == .parts([
            .text("First image:"), .imageURL(urls[0]), .text("Second image:"), .imageURL(urls[1]),
        ]))
        #expect(chat.messages[0].content.hasMedia,
                "The built-in text-only engine must reject, not silently discard, Responses images")
        // Provider adapters encode canonical Chat wire content; verify the
        // image survives that additional boundary without loading its URL.
        let data = try JSONEncoder().encode(chat)
        let roundTrip = try JSONDecoder().decode(OpenAIChatCompletionRequest.self, from: data)
        #expect(roundTrip == chat)
        let encoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let messages = try #require(encoded["messages"] as? [[String: Any]])
        let content = try #require(messages[0]["content"] as? [[String: Any]])
        #expect(content[1]["type"] as? String == "image_url")
        #expect((content[1]["image_url"] as? [String: String])?["url"] == urls[0])
        #expect((content[3]["image_url"] as? [String: String])?["url"] == urls[1])
    }

    @Test func chatImagesAndVideoKeepTheirExistingCanonicalShape() throws {
        let input = Data(#"[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"video_url","video_url":{"url":"https://example.invalid/movie.mp4"}}]"#.utf8)
        let parts = try JSONDecoder().decode([OpenAIContentPart].self, from: input)
        #expect(parts == [.imageURL("data:image/png;base64,AAAA"), .videoURL("https://example.invalid/movie.mp4")])
        #expect(try JSONDecoder().decode([OpenAIContentPart].self, from: JSONEncoder().encode(parts)) == parts)
    }

    @Test func unsupportedUploadedFileIDsAndMalformedImagesFailAtDecode() throws {
        for part in [
            #"{"type":"input_image","file_id":"file-fixture"}"#,
            #"{"type":"input_image","file_id":"file-fixture","image_url":"https://example.invalid/a.png"}"#,
            #"{"type":"input_image"}"#,
            #"{"type":"input_image","image_url":null}"#,
            #"{"type":"input_image","image_url":{"url":"https://example.invalid/a.png"}}"#,
        ] {
            #expect(throws: DecodingError.self) {
                try JSONDecoder().decode(OpenAIContentPart.self, from: Data(part.utf8))
            }
        }
    }
}
