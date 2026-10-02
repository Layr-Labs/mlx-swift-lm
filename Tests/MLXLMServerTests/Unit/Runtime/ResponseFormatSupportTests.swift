import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {
    @Suite
    struct ResponseFormatSupportTests {

        private typealias Support = OpenAIResponseFormatSupport

        private func schemaFormat(_ schema: JSONValue, strict: Bool? = nil) -> OpenAIResponseFormat
        {
            .jsonSchema(.init(name: "s", description: "d", strict: strict, schema: schema))
        }

        private func expectInvalid(
            _ content: String, _ format: OpenAIResponseFormat, message: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            #expect(
                throws: MLXOpenAIServiceError.invalidResponseFormatOutput(message),
                sourceLocation: sourceLocation
            ) {
                try Support.normalizedContent(content, for: format)
            }
        }

        @Test
        func textFormatsLeaveTheRequestAndContentUnchanged() throws {
            let request = OpenAIChatCompletionRequest(
                model: "m", messages: [.init(role: .user, content: .text("q"))],
                responseFormat: .text())
            #expect(try Support.preparedRequest(request) == request)
            #expect(try Support.normalizedContent(" not json ", for: .text()) == " not json ")
            #expect(try Support.normalizedContent("x", for: nil) == "x")
            #expect(!OpenAIResponseFormat.text().requiresJSONOutput)
        }

        @Test
        func preparedRequestInsertsTheInstructionAfterSystemMessages() throws {
            let request = OpenAIChatCompletionRequest(
                model: "m",
                messages: [
                    .init(role: .system, content: .text("s1")),
                    .init(role: .system, content: .text("s2")),
                    .init(role: .user, content: .text("q")),
                ],
                responseFormat: .jsonObject())
            let prepared = try Support.preparedRequest(request)
            #expect(prepared.messages.map(\.role) == [.system, .system, .system, .user])
            #expect(
                prepared.messages[2].textContent.hasPrefix(
                    "You must respond with a single valid JSON object."))
        }

        @Test
        func schemaInstructionNamesTheSchema() throws {
            let request = OpenAIChatCompletionRequest(
                model: "m", messages: [.init(role: .system, content: .text("only system"))],
                responseFormat: schemaFormat(.object(["type": .string("object")]), strict: true))
            let prepared = try Support.preparedRequest(request)
            let instruction = try #require(prepared.messages.last).textContent
            #expect(instruction.contains("Schema name: s."))
            #expect(instruction.contains("Description: d"))
            #expect(instruction.contains("The output must strictly conform to the schema."))
            #expect(instruction.hasSuffix(#"{"type":"object"}"#))
        }

        @Test
        func aSchemaFormatWithoutASchemaIsRejected() {
            let format = OpenAIResponseFormat(type: .jsonSchema)
            let request = OpenAIChatCompletionRequest(
                model: "m", messages: [], responseFormat: format)
            let expected = MLXOpenAIServiceError.invalidResponseFormatOutput(
                "response_format json_schema requires a json_schema payload")
            #expect(throws: expected) { try Support.preparedRequest(request) }
            #expect(throws: expected) { try Support.normalizedContent("{}", for: format) }
        }

        @Test
        func jsonIsFoundInFencesAndSurroundingText() throws {
            #expect(
                try Support.normalizedContent("  {\"a\":1}  ", for: .jsonObject()) == "{\"a\":1}")
            #expect(
                try Support.normalizedContent("```json\n{\"a\": [1]}\n```", for: .jsonObject())
                    == "{\"a\": [1]}")
            #expect(
                try Support.normalizedContent(
                    "Here: {\"s\":\"} not end\",\"e\":\"\\\"\"} done", for: .jsonObject())
                    == "{\"s\":\"} not end\",\"e\":\"\\\"\"}")
            #expect(
                try Support.normalizedContent(
                    "list [1, {\"b\": 2}] end",
                    for: schemaFormat(.object(["type": .string("array")])))
                    == "[1, {\"b\": 2}]")
        }

        @Test
        func invalidJSONIsRejected() {
            expectInvalid("no json here", .jsonObject(), message: "model output was not valid JSON")
            expectInvalid("{\"a\": ]", .jsonObject(), message: "model output was not valid JSON")
        }

        @Test
        func jsonObjectRequiresAnObjectRoot() {
            expectInvalid(
                "[1, 2]", .jsonObject(),
                message: "response_format json_object requires a JSON object")
        }

        @Test
        func schemaChecksTypesAndLimits() throws {
            let schema: JSONValue = .object([
                "type": .string("object"),
                "required": .array([.string("name"), .string("age")]),
                "additionalProperties": .bool(false),
                "properties": .object([
                    "name": .object([
                        "type": .string("string"), "minLength": .int(2), "maxLength": .int(4),
                    ]),
                    "age": .object([
                        "type": .string("integer"), "minimum": .int(0), "maximum": .double(130),
                    ]),
                    "tags": .object([
                        "type": .string("array"),
                        "items": .object(["enum": .array([.string("a"), .string("b")])]),
                    ]),
                    "kind": .object(["const": .string("person")]),
                    "note": .object(["type": .array([.string("string"), .string("null")])]),
                ]),
            ])
            let format = schemaFormat(schema)

            #expect(
                try Support.normalizedContent(
                    #"{"name":"Ann","age":30.0,"tags":["a"],"kind":"person","note":null}"#,
                    for: format)
                    == #"{"name":"Ann","age":30.0,"tags":["a"],"kind":"person","note":null}"#)

            expectInvalid(#"{"name":"Ann"}"#, format, message: "required property 'age' is missing")
            expectInvalid(
                #"{"name":"Ann","age":1,"zzz":1,"extra":2}"#, format,
                message: "additional property 'extra' is not allowed")
            expectInvalid(
                #"{"name":"A","age":1}"#, format, message: "string is shorter than schema minLength"
            )
            expectInvalid(
                #"{"name":"Annie","age":1}"#, format,
                message: "string is longer than schema maxLength")
            expectInvalid(
                #"{"name":"Ann","age":-1}"#, format, message: "number is below schema minimum")
            expectInvalid(
                #"{"name":"Ann","age":131}"#, format, message: "number is above schema maximum")
            expectInvalid(
                #"{"name":"Ann","age":1.5}"#, format,
                message: "value does not match schema type 'integer'")
            expectInvalid(
                #"{"name":"Ann","age":1,"tags":["c"]}"#, format,
                message: "value is not one of the schema enum values")
            expectInvalid(
                #"{"name":"Ann","age":1,"kind":"robot"}"#, format,
                message: "value does not match schema const")
            expectInvalid(
                #"{"name":"Ann","age":1,"note":3}"#, format,
                message: "value does not match any schema type")
        }

        @Test
        func additionalPropertiesCanHaveASchema() throws {
            let format = schemaFormat(
                .object([
                    "type": .string("object"),
                    "properties": .object(["id": .object(["type": .string("string")])]),
                    "additionalProperties": .object(["type": .string("number")]),
                ]))
            #expect(
                try Support.normalizedContent(#"{"id":"x","score":0.5}"#, for: format)
                    == #"{"id":"x","score":0.5}"#)
            expectInvalid(
                #"{"id":"x","score":"high"}"#, format,
                message: "value does not match schema type 'number'")
        }

        @Test
        func aSchemaThatIsNotAnObjectAcceptsAnyJSON() throws {
            #expect(
                try Support.normalizedContent("[true]", for: schemaFormat(.bool(true))) == "[true]")
        }
    }
}
