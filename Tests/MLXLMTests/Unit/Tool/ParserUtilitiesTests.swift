import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {
    @Suite
    struct ParserUtilitiesTests {

        private let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "search",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "query": ["type": "string"] as [String: any Sendable],
                            "limit": ["type": "integer"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ],
            ["type": "function", "function": ["name": "broken"] as [String: any Sendable]],
        ]

        private func convert(_ value: String, type: String) -> any Sendable {
            let tools: [[String: any Sendable]] = [
                [
                    "function": [
                        "name": "f",
                        "parameters": [
                            "properties": [
                                "p": ["type": type] as [String: any Sendable]
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ]
            ]
            return convertParameterValue(value, paramName: "p", funcName: "f", tools: tools)
        }

        @Test
        func asSendableKeepsBooleansAndNumbersApart() throws {
            let object = try JSONSerialization.jsonObject(
                with: Data(#"{"b": true, "n": 1, "s": "x", "a": [null], "d": {"k": 2}}"#.utf8))
            let value = try #require(asSendable(object) as? [String: any Sendable])
            #expect(value["b"] as? Bool == true)
            #expect(value["n"] is NSNumber)
            #expect((value["n"] as? NSNumber)?.intValue == 1)
            #expect(value["s"] as? String == "x")
            #expect((value["a"] as? [any Sendable])?.first is NSNull)
            #expect(((value["d"] as? [String: any Sendable])?["k"] as? NSNumber)?.intValue == 2)
            #expect(asSendable(Date(timeIntervalSince1970: 0)) as? String != nil)
        }

        @Test
        func decodeToolCallArgumentsDecodesOnlyObjects() throws {
            let object = try #require(
                decodeToolCallArguments(#"{"command": "ls -la"}"#) as? [String: any Sendable])
            #expect(object["command"] as? String == "ls -la")
            #expect(decodeToolCallArguments("[1, 2]") as? String == "[1, 2]")
            #expect(decodeToolCallArguments("not json") as? String == "not json")
        }

        @Test
        func deserializeFallsBackToTheString() {
            #expect((deserialize("[42]") as? [any Sendable])?.count == 1)
            // JSONSerialization reads no top-level scalar, so a bare number
            // stays a string.
            #expect(deserialize("42") as? String == "42")
            #expect(deserialize("plain words") as? String == "plain words")
            #expect(tryParseJSON("{") == nil)
        }

        @Test
        func schemaLookupFindsTheParameterType() {
            #expect(
                getParameterType(funcName: "search", paramName: "limit", tools: tools) == "integer")
            #expect(getParameterType(funcName: "search", paramName: "missing", tools: tools) == nil)
            #expect(getParameterType(funcName: "broken", paramName: "query", tools: tools) == nil)
            #expect(getParameterType(funcName: "search", paramName: "query", tools: nil) == nil)
            #expect(isStringType(funcName: "search", argName: "query", tools: tools))
            #expect(!isStringType(funcName: "search", argName: "limit", tools: tools))
            #expect(!isStringType(funcName: "other", argName: "query", tools: tools))
        }

        @Test
        func parameterConfigGivesThePropertiesOfTheFunction() {
            let config = getParameterConfig(funcName: "search", tools: tools)
            #expect(Set(config.keys) == ["query", "limit"])
            #expect(getParameterConfig(funcName: "broken", tools: tools).isEmpty)
            #expect(getParameterConfig(funcName: "search", tools: nil).isEmpty)
        }

        @Test
        func extractTypesReadsTypeEnumAndChoices() {
            #expect(extractTypesFromSchema(nil) == ["string"])
            #expect(extractTypesFromSchema([:]) == ["string"])
            #expect(extractTypesFromSchema(["type": "integer"]) == ["integer"])
            #expect(
                Set(extractTypesFromSchema(["type": ["integer", "null"]]))
                    == ["integer", "null"])
            let enumSchema: [String: any Sendable] = [
                "enum": [
                    NSNull(), true, 1, 1.5, "s", [1] as [any Sendable],
                    ["k": 1] as [String: any Sendable],
                ]
                    as [any Sendable]
            ]
            #expect(
                Set(extractTypesFromSchema(enumSchema))
                    == ["null", "boolean", "integer", "number", "string", "array", "object"])
            let choices: [String: any Sendable] = [
                "anyOf": [["type": "string"], ["type": "number"]] as [[String: any Sendable]],
                "oneOf": [["type": "boolean"]] as [[String: any Sendable]],
                "allOf": [["enum": [2] as [any Sendable]]] as [[String: any Sendable]],
            ]
            #expect(
                Set(extractTypesFromSchema(choices)) == ["string", "number", "boolean", "integer"])
        }

        @Test
        func isDeclaredToolAcceptsAnyNameWithoutSchemas() {
            #expect(isDeclaredTool("anything", tools: nil))
            #expect(isDeclaredTool("anything", tools: []))
            #expect(isDeclaredTool("search", tools: tools))
            #expect(!isDeclaredTool("delete", tools: tools))
        }

        @Test
        func convertValueWithTypesFollowsThePriority() {
            #expect(convertValueWithTypes("None", types: ["string"]) is NSNull)
            #expect(convertValueWithTypes("7", types: ["string", "integer"]) as? Int == 7)
            #expect(convertValueWithTypes("2.5", types: ["number"]) as? Double == 2.5)
            #expect(convertValueWithTypes("3.0", types: ["float"]) as? Int == 3)
            #expect(convertValueWithTypes(" Yes", types: ["boolean"]) as? Bool == true)
            #expect(convertValueWithTypes("off", types: ["bool"]) as? Bool == false)
            #expect(convertValueWithTypes("maybe", types: ["boolean"]) as? String == "maybe")
            #expect(
                (convertValueWithTypes(#"{"a": 1}"#, types: ["object"]) as? [String: any Sendable])?
                    .keys.first == "a")
            #expect(convertValueWithTypes("x", types: ["integer", "text"]) as? String == "x")
            #expect(
                (convertValueWithTypes("[1]", types: ["unknown"]) as? [any Sendable])?.count == 1)
            #expect(convertValueWithTypes("free", types: ["unknown"]) as? String == "free")
        }

        @Test
        func convertParameterValueFollowsTheSchemaType() {
            #expect(convert("5", type: "string") as? String == "5")
            #expect(convert("5", type: "int64") as? Int == 5)
            #expect(convert("five", type: "uint") as? String == "five")
            #expect(convert("1.25", type: "number") as? Double == 1.25)
            #expect(convert("4.0", type: "float") as? Int == 4)
            #expect(convert("n/a", type: "float") as? String == "n/a")
            #expect(convert("ON", type: "boolean") as? Bool == true)
            #expect(convert("nope", type: "bool") as? Bool == false)
            #expect((convert("[1, 2]", type: "list") as? [any Sendable])?.count == 2)
            #expect(convert("{bad", type: "dict") as? String == "{bad")
            #expect(convert("v", type: "custom") as? String == "v")
            #expect(
                convertParameterValue("v", paramName: "p", funcName: "missing", tools: nil)
                    as? String
                    == "v")
        }

        @Test
        func extractNameRemovesMatchingQuotes() {
            #expect(extractName("  \"get_weather\" ") == "get_weather")
            #expect(extractName("'search'") == "search")
            #expect(extractName("plain") == "plain")
            #expect(extractName("\"unbalanced") == "\"unbalanced")
        }
    }
}
