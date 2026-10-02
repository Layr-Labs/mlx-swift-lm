import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct ToolSchemaTests {

        private struct Empty: Codable {}

        private func dictionary(_ value: (any Sendable)?) throws -> [String: any Sendable] {
            try #require(value as? [String: any Sendable])
        }

        @Test
        func initBuildsAFunctionSchema() throws {
            let tool = Tool<Empty, Empty>(
                name: "lookup",
                description: "Looks up a word.",
                parameters: [
                    .required("word", type: .string, description: "The word."),
                    .optional(
                        "lang", type: .string, description: "The language.",
                        extraProperties: ["enum": ["en", "de"]]),
                ]
            ) { _ in Empty() }

            #expect(tool.name == "lookup")
            #expect(tool.schema["type"] as? String == "function")
            let function = try dictionary(tool.schema["function"])
            #expect(function["description"] as? String == "Looks up a word.")
            let parameters = try dictionary(function["parameters"])
            #expect(parameters["type"] as? String == "object")
            #expect(parameters["required"] as? [String] == ["word"])
            let properties = try dictionary(parameters["properties"])
            let lang = try dictionary(properties["lang"])
            #expect(lang["type"] as? String == "string")
            #expect(lang["description"] as? String == "The language.")
            #expect(lang["enum"] as? [String] == ["en", "de"])
        }

        @Test
        func nameIsEmptyWhenTheSchemaHasNoFunctionName() {
            let tool = Tool<Empty, Empty>(schema: ["type": "function"]) { _ in Empty() }
            #expect(tool.name.isEmpty)
        }

        @Test
        func parameterTypesMapToJSONSchemaTypes() throws {
            func schema(_ type: ToolParameterType) -> [String: any Sendable] {
                ToolParameter.required("p", type: type, description: "d").schema
            }
            #expect(schema(.bool)["type"] as? String == "boolean")
            #expect(schema(.int)["type"] as? String == "integer")
            #expect(schema(.double)["type"] as? String == "number")
            #expect(schema(.data)["type"] as? String == "string")
            #expect(schema(.data)["contentEncoding"] as? String == "base64")

            let array = schema(.array(elementType: .int))
            #expect(array["type"] as? String == "array")
            #expect(try dictionary(array["items"])["type"] as? String == "integer")

            let object = schema(
                .object(properties: [
                    .required("x", type: .double, description: "X."),
                    .optional("y", type: .double, description: "Y."),
                ]))
            #expect(object["type"] as? String == "object")
            #expect(object["required"] as? [String] == ["x"])
            let properties = try dictionary(object["properties"])
            #expect(Set(properties.keys) == ["x", "y"])
            #expect(try dictionary(properties["y"])["description"] as? String == "Y.")
        }

        @Test
        func requiredAndOptionalSetIsRequired() {
            #expect(ToolParameter.required("a", type: .int, description: "").isRequired)
            #expect(!ToolParameter.optional("a", type: .int, description: "").isRequired)
        }
    }
}
