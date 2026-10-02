import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {
    @Suite
    struct OpenAIMessageCodingTests {

        private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
            try JSONDecoder().decode(type, from: Data(json.utf8))
        }

        private func encodedObject<T: Encodable>(_ value: T) throws -> [String: Any] {
            let data = try JSONEncoder.openAIServer.encode(value)
            return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        }

        @Test
        func contentPartsDecodeEveryKnownType() throws {
            #expect(
                try decode(OpenAIContentPart.self, #"{"type":"text","text":"a"}"#) == .text("a"))
            #expect(
                try decode(OpenAIContentPart.self, #"{"type":"input_text","text":"b"}"#)
                    == .text("b"))
            #expect(
                try decode(OpenAIContentPart.self, #"{"type":"output_text","text":"c"}"#)
                    == .text("c"))
            #expect(
                try decode(
                    OpenAIContentPart.self, #"{"type":"image_url","image_url":{"url":"u1"}}"#)
                    == .imageURL("u1"))
            #expect(
                try decode(OpenAIContentPart.self, #"{"type":"input_image","image_url":"u2"}"#)
                    == .imageURL("u2"))
            #expect(
                try decode(OpenAIContentPart.self, #"{"type":"video_url","video_url":{"url":"v"}}"#)
                    == .videoURL("v"))
            #expect(
                try decode(
                    OpenAIContentPart.self,
                    #"{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}"#)
                    == .inputAudio(OpenAIInputAudio(data: "AA==", format: .wav)))
            #expect(
                try decode(OpenAIContentPart.self, #"{"type":"input_file"}"#)
                    == .unsupported(type: "input_file"))
        }

        @Test
        func anInputImageWithAFileIDIsRejected() {
            #expect(throws: DecodingError.self) {
                try decode(OpenAIContentPart.self, #"{"type":"input_image","file_id":"file-1"}"#)
            }
        }

        @Test
        func contentPartsEncodeToTheChatShape() throws {
            let parts: [OpenAIContentPart] = [
                .text("t"), .imageURL("i"), .videoURL("v"), .unsupported(type: "x"),
            ]
            let data = try JSONEncoder().encode(parts)
            let decoded = try JSONDecoder().decode([OpenAIContentPart].self, from: data)
            #expect(decoded == parts)
            let image = try encodedObject(OpenAIContentPart.imageURL("i"))
            #expect(image["type"] as? String == "image_url")
            #expect((image["image_url"] as? [String: Any])?["url"] as? String == "i")
        }

        @Test
        func messageContentGivesTextAndMedia() throws {
            let parts = try decode(
                OpenAIMessageContent.self,
                #"[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"u"}},{"type":"text","text":"b"}]"#
            )
            #expect(parts.text == "ab")
            #expect(parts.hasMedia)
            #expect(try decode(OpenAIMessageContent.self, #""plain""#) == .text("plain"))
            #expect(try decode(OpenAIMessageContent.self, "null") == .null)
            #expect(OpenAIMessageContent.null.text.isEmpty)
            #expect(!OpenAIMessageContent.text("x").hasMedia)
            #expect(!OpenAIMessageContent.parts([.text("x"), .unsupported(type: "y")]).hasMedia)
            let encodedNull = try JSONEncoder().encode(OpenAIMessageContent.null)
            #expect(String(decoding: encodedNull, as: UTF8.self) == "null")
        }

        @Test
        func onlyAnAssistantWithToolCallsCanOmitContent() throws {
            let assistant = try decode(
                OpenAIChatMessage.self,
                #"{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}"#
            )
            #expect(assistant.content == .null)
            #expect(assistant.toolCalls?.first?.function.name == "f")

            #expect(throws: DecodingError.self) {
                try decode(OpenAIChatMessage.self, #"{"role":"user"}"#)
            }
            #expect(throws: DecodingError.self) {
                try decode(OpenAIChatMessage.self, #"{"role":"assistant","tool_calls":[]}"#)
            }
        }

        @Test
        func messageFieldsDecodeFromSnakeCase() throws {
            let message = try decode(
                OpenAIChatMessage.self,
                #"{"role":"tool","content":"42","name":"calc","tool_call_id":"c9","reasoning_content":"r"}"#
            )
            #expect(message.role == .tool)
            #expect(message.textContent == "42")
            #expect(message.name == "calc")
            #expect(message.toolCallID == "c9")
            #expect(message.reasoningContent == "r")
        }

        @Test
        func chatMessageKeepsTheRole() {
            #expect(
                OpenAIChatMessage(role: .system, content: .text("s")).chatMessage().role == .system)
            #expect(OpenAIChatMessage(role: .user, content: .text("u")).chatMessage().role == .user)
            #expect(
                OpenAIChatMessage(role: .assistant, content: .text("a")).chatMessage().role
                    == .assistant)
            let tool = OpenAIChatMessage(role: .tool, content: .text("t")).chatMessage()
            #expect(tool.role == .tool)
            #expect(tool.content == "t")
        }
    }
}
