import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {
    @Suite
    struct OpenAIRequestCodingTests {

        private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
            try JSONDecoder().decode(type, from: Data(json.utf8))
        }

        private func encodedString<T: Encodable>(_ value: T) throws -> String {
            String(decoding: try JSONEncoder.openAIServer.encode(value), as: UTF8.self)
        }

        @Test
        func toolDecodesTheNestedAndTheFlatForms() throws {
            let nested = try decode(
                OpenAITool.self,
                #"{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}"#
            )
            #expect(
                nested
                    == OpenAITool(
                        function: .init(
                            name: "f", description: "d",
                            parameters: .object(["type": .string("object")]))
                    ))

            let flat = try decode(OpenAITool.self, #"{"name":"g","parameters":{"type":"object"}}"#)
            #expect(flat.type == "function")
            #expect(flat.function.name == "g")
            #expect(flat.function.parameters == .object(["type": .string("object")]))

            let anthropicStyle = try decode(
                OpenAITool.self, #"{"type":"custom","name":"h","input_schema":{"type":"string"}}"#)
            #expect(anthropicStyle.type == "custom")
            #expect(anthropicStyle.function.parameters == .object(["type": .string("string")]))

            #expect(throws: DecodingError.self) {
                try decode(OpenAITool.self, #"{"type":"function"}"#)
            }
        }

        @Test
        func toolEncodesTheNestedForm() throws {
            let tool = try decode(OpenAITool.self, #"{"name":"g"}"#)
            #expect(try encodedString(tool) == #"{"function":{"name":"g"},"type":"function"}"#)
        }

        @Test
        func toolSpecConvertsTheParameters() throws {
            let tool = OpenAITool(
                function: .init(
                    name: "f", description: "desc",
                    parameters: .object([
                        "type": .string("object"),
                        "required": .array([.string("a")]),
                        "default": .null,
                        "flag": .bool(true),
                        "n": .int(2),
                        "x": .double(0.5),
                    ])))
            let spec = tool.toolSpec()
            #expect(spec["type"] as? String == "function")
            let function = try #require(spec["function"] as? [String: any Sendable])
            #expect(function["name"] as? String == "f")
            #expect(function["description"] as? String == "desc")
            let parameters = try #require(function["parameters"] as? [String: any Sendable])
            #expect(parameters["required"] as? [any Sendable] != nil)
            #expect((parameters["required"] as? [any Sendable])?.first as? String == "a")
            #expect(parameters["flag"] as? Bool == true)
            #expect(parameters["n"] as? Int == 2)
            #expect(parameters["x"] as? Double == 0.5)
            #expect(parameters["default"] != nil)
            #expect(!(parameters["default"] is String))

            let bare = OpenAITool(function: .init(name: "b")).toolSpec()
            let bareFunction = try #require(bare["function"] as? [String: any Sendable])
            #expect(bareFunction.keys.sorted() == ["name"])
        }

        @Test
        func toolChoiceDecodesModesAndNames() throws {
            #expect(try decode(OpenAIToolChoice.self, #""auto""#) == .mode(.auto))
            #expect(try decode(OpenAIToolChoice.self, #""none""#) == .mode(.none))
            #expect(try decode(OpenAIToolChoice.self, #"{"type":"required"}"#) == .mode(.required))
            #expect(
                try decode(OpenAIToolChoice.self, #"{"type":"function","function":{"name":"f"}}"#)
                    == .function(name: "f"))
            #expect(
                try decode(OpenAIToolChoice.self, #"{"type":"function","name":"g"}"#)
                    == .function(name: "g"))
            #expect(
                try decode(
                    OpenAIToolChoice.self,
                    #"{"type":"function","name":"h","function":{"name":"h"}}"#)
                    == .function(name: "h"))
        }

        @Test
        func toolChoiceRejectsInvalidObjects() {
            #expect(throws: DecodingError.self) {
                try decode(OpenAIToolChoice.self, #"{"type":"tool","name":"f"}"#)
            }
            #expect(throws: DecodingError.self) {
                try decode(
                    OpenAIToolChoice.self,
                    #"{"type":"function","name":"a","function":{"name":"b"}}"#)
            }
            #expect(throws: DecodingError.self) {
                try decode(OpenAIToolChoice.self, #"{"type":"function"}"#)
            }
            #expect(throws: DecodingError.self) {
                try decode(OpenAIToolChoice.self, #""sometimes""#)
            }
        }

        @Test
        func toolChoiceEncodesBothForms() throws {
            #expect(try encodedString(OpenAIToolChoice.mode(.required)) == #""required""#)
            #expect(
                try encodedString(OpenAIToolChoice.function(name: "f"))
                    == #"{"function":{"name":"f"},"type":"function"}"#)
        }

        @Test
        func generationParametersUseTheDefaults() {
            let defaults = OpenAIChatCompletionRequest(model: "m", messages: [])
                .generationParameters
            #expect(defaults.maxTokens == nil)
            #expect(defaults.temperature == 0.6)
            #expect(defaults.topP == 1.0)
            #expect(defaults.topK == 0)
            #expect(defaults.minP == 0)

            let set = OpenAIChatCompletionRequest(
                model: "m", messages: [], temperature: 0.1, topP: 0.9, topK: 40, minP: 0.05,
                maxTokens: 12, presencePenalty: 0.2, frequencyPenalty: 0.3, repetitionPenalty: 1.1
            ).generationParameters
            #expect(set.maxTokens == 12)
            #expect(set.temperature == 0.1)
            #expect(set.topP == 0.9)
            #expect(set.topK == 40)
            #expect(set.minP == 0.05)
            #expect(set.presencePenalty == 0.2)
            #expect(set.frequencyPenalty == 0.3)
            #expect(set.repetitionPenalty == 1.1)
        }

        @Test
        func completionPromptDecodesTextOrTexts() throws {
            let single = try decode(
                OpenAICompletionRequest.self,
                #"{"model":"m","prompt":"hi","max_tokens":5,"top_p":0.5}"#)
            #expect(single.prompt == .text("hi"))
            #expect(single.maxTokens == 5)
            #expect(single.topP == 0.5)
            let many = try decode(
                OpenAICompletionRequest.self, #"{"model":"m","prompt":["a","b"]}"#)
            #expect(many.prompt.firstText == "a")
            #expect(OpenAICompletionPrompt.texts([]).firstText.isEmpty)
            #expect(try encodedString(OpenAICompletionPrompt.texts(["a"])) == #"["a"]"#)
            #expect(try encodedString(OpenAICompletionPrompt.text("a")) == #""a""#)
            #expect(throws: DecodingError.self) {
                try decode(OpenAICompletionPrompt.self, "3")
            }
        }

        @Test
        func completionRequestBecomesAUserChatRequest() throws {
            let request = try decode(
                OpenAICompletionRequest.self,
                #"{"model":"m","prompt":["first","second"],"stream":true,"temperature":0.2,"max_tokens":7}"#
            )
            let chat = request.chatCompletionRequest
            #expect(chat.model == "m")
            #expect(chat.messages == [OpenAIChatMessage(role: .user, content: .text("first"))])
            #expect(chat.stream == true)
            #expect(chat.temperature == 0.2)
            #expect(chat.maxTokens == 7)
        }

        @Test
        func embeddingInputDecodesTextOrTexts() throws {
            let request = try decode(
                OpenAIEmbeddingRequest.self,
                #"{"model":"e","input":"one","encoding_format":"float","normalize":true}"#)
            #expect(request.input.texts == ["one"])
            #expect(request.encodingFormat == "float")
            #expect(request.normalize == true)
            let many = try decode(OpenAIEmbeddingRequest.self, #"{"model":"e","input":["a","b"]}"#)
            #expect(many.input.texts == ["a", "b"])
            #expect(try encodedString(OpenAIEmbeddingInput.text("x")) == #""x""#)
            #expect(try encodedString(OpenAIEmbeddingInput.texts(["x"])) == #"["x"]"#)
        }

        @Test
        func reasoningParserFormatAcceptsAliases() throws {
            let cases: [(String, ReasoningParserFormat)] = [
                ("off", .none), ("disabled", .none), ("deepseek", .deepseekR1), ("R1", .deepseekR1),
                ("think", .deepseekR1), ("qwen", .qwen3), ("openai-harmony", .harmony),
                ("gpt_oss", .harmony), ("gemma-4", .gemma4), ("gemma", .gemma4),
            ]
            for (raw, expected) in cases {
                #expect(try decode(ReasoningParserFormat.self, "\"\(raw)\"") == expected, "\(raw)")
            }
        }
    }
}
