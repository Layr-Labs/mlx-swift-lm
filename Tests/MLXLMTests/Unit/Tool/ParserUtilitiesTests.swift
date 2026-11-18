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
            // Upstream a56aee36 (#548): a union that admits a string, or a type
            // that is not known, keeps the text; null needs a declared "null".
            #expect(convertValueWithTypes("None", types: ["string"]) as? String == "None")
            #expect(convertValueWithTypes("None", types: ["integer", "null"]) is NSNull)
            #expect(convertValueWithTypes("7", types: ["string", "integer"]) as? String == "7")
            #expect(convertValueWithTypes("7", types: ["integer"]) as? Int == 7)
            #expect(convertValueWithTypes("2.5", types: ["number"]) as? Double == 2.5)
            #expect(convertValueWithTypes("3.0", types: ["float"]) as? Int == 3)
            #expect(convertValueWithTypes(" Yes", types: ["boolean"]) as? Bool == true)
            #expect(convertValueWithTypes("off", types: ["bool"]) as? Bool == false)
            #expect(convertValueWithTypes("maybe", types: ["boolean"]) as? String == "maybe")
            #expect(
                (convertValueWithTypes(#"{"a": 1}"#, types: ["object"]) as? [String: any Sendable])?
                    .keys.first == "a")
            #expect(convertValueWithTypes("x", types: ["integer", "text"]) as? String == "x")
            #expect(convertValueWithTypes("[1]", types: ["unknown"]) as? String == "[1]")
            #expect((convertValueWithTypes("[1]", types: ["array"]) as? [any Sendable])?.count == 1)
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
            // Upstream a56aee36 (#548): a value that does not convert keeps its text.
            #expect(convert("nope", type: "bool") as? String == "nope")
            #expect(convert("no", type: "bool") as? Bool == false)
            #expect((convert("[1, 2]", type: "list") as? [any Sendable])?.count == 2)
            #expect(convert("{bad", type: "dict") as? String == "{bad")
            #expect(convert("v", type: "custom") as? String == "v")
            #expect(
                convertParameterValue("v", paramName: "p", funcName: "missing", tools: nil)
                    as? String
                    == "v")
        }

        /// A Qwen XML value whose parameter names a local `$ref` reads as the
        /// referenced schema says, on the complete-call path too; a reference
        /// that names nothing keeps today's text unless the coordinator's type
        /// stands beside it. Other dialects keep reading references as text.
        @Test
        func qwenXMLParametersReadLocalReferences() throws {
            let tools: [[String: any Sendable]] = [
                [
                    "type": "function",
                    "function": [
                        "name": "plot",
                        "parameters": [
                            "type": "object",
                            "definitions": [
                                "Point": [
                                    "type": "object",
                                    "properties": [
                                        "x": ["type": "integer"] as [String: any Sendable]
                                    ] as [String: any Sendable],
                                ] as [String: any Sendable],
                                "Loop": ["$ref": "#/definitions/Loop"] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "properties": [
                                "at": ["$ref": "#/definitions/Point"] as [String: any Sendable],
                                "loop": ["$ref": "#/definitions/Loop"] as [String: any Sendable],
                                "dangling": ["$ref": "#/definitions/Nowhere"]
                                    as [String: any Sendable],
                                "hinted": [
                                    "$ref": "#/definitions/Nowhere", "type": "integer",
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            ]
            let text =
                "<tool_call>\n<function=plot>\n"
                + "<parameter=at>\n{\"x\": \"7\"}\n</parameter>\n"
                + "<parameter=loop>\n{\"a\": 1}\n</parameter>\n"
                + "<parameter=dangling>\n{\"a\": 1}\n</parameter>\n"
                + "<parameter=hinted>\n3\n</parameter>\n"
                + "</function>\n</tool_call>"
            let call = try #require(
                Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
                    .parse(content: text, tools: tools))
            #expect(call.function.arguments["at"] == .object(["x": .int(7)]))
            #expect(call.function.arguments["loop"] == .string(#"{"a": 1}"#))
            #expect(call.function.arguments["dangling"] == .string(#"{"a": 1}"#))
            #expect(call.function.arguments["hinted"] == .int(3))
            #expect(
                convertParameterValue("{\"x\": 1}", paramName: "at", funcName: "plot", tools: tools)
                    as? String == "{\"x\": 1}")
        }

        private static func referenceTools(
            defs: [String: any Sendable], properties: [String: any Sendable]
        ) -> [[String: any Sendable]] {
            [
                [
                    "type": "function",
                    "function": [
                        "name": "f",
                        "parameters": [
                            "type": "object", "$defs": defs, "properties": properties,
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ]
            ]
        }

        /// Union branches that name the same definitions again (here a diamond
        /// 20 levels deep: 2^20 paths) are read within a fixed budget of
        /// references; past it the value keeps its text, as for a reference
        /// that names nothing, instead of costing time exponential in the
        /// depth. A union of a few references still reads by its targets.
        @Test
        func qwenXMLReferenceReadingIsBoundedOnBranchingReferences() {
            var defs: [String: any Sendable] = [
                "D20": ["type": "integer"] as [String: any Sendable],
                "A": ["type": "integer"] as [String: any Sendable],
                "B": ["type": "integer", "minimum": 0] as [String: any Sendable],
            ]
            for level in 0 ..< 20 {
                let next = ["$ref": "#/$defs/D\(level + 1)"] as [String: any Sendable]
                defs["D\(level)"] = ["anyOf": [next, next]] as [String: any Sendable]
            }
            let tools = Self.referenceTools(
                defs: defs,
                properties: [
                    "deep": ["$ref": "#/$defs/D0"] as [String: any Sendable],
                    "few": [
                        "anyOf": [
                            ["$ref": "#/$defs/A"] as [String: any Sendable],
                            ["$ref": "#/$defs/B"] as [String: any Sendable],
                            ["type": "null"] as [String: any Sendable],
                        ]
                    ] as [String: any Sendable],
                ])
            let elapsed = ContinuousClock().measure {
                #expect(parameterKeepsText(paramName: "deep", funcName: "f", tools: tools))
                #expect(
                    convertParameterValue(
                        "5", paramName: "deep", funcName: "f", tools: tools,
                        resolvingReferences: true) as? String == "5")
            }
            #expect(elapsed < .milliseconds(250), "\(elapsed)")
            #expect(!parameterKeepsText(paramName: "few", funcName: "f", tools: tools))
            #expect(
                JSONValue.from(
                    convertParameterValue(
                        "5", paramName: "few", funcName: "f", tools: tools,
                        resolvingReferences: true)) == .int(5))
        }

        /// A union whose branches name their own definition (a cycle the
        /// coordinator keeps) and a diamond 30 levels deep read within the
        /// budget and keep their text. Without the budget the cycle alone costs
        /// 2^32 steps, so this test has no red run.
        @Test
        func qwenXMLReferenceCyclesKeepTheirText() {
            var defs: [String: any Sendable] = [
                "A": [
                    "anyOf": [
                        ["$ref": "#/$defs/A"] as [String: any Sendable],
                        ["$ref": "#/$defs/A"] as [String: any Sendable],
                    ]
                ] as [String: any Sendable],
                "D30": ["type": "integer"] as [String: any Sendable],
            ]
            for level in 0 ..< 30 {
                let next = ["$ref": "#/$defs/D\(level + 1)"] as [String: any Sendable]
                defs["D\(level)"] = ["oneOf": [next, next]] as [String: any Sendable]
            }
            let tools = Self.referenceTools(
                defs: defs,
                properties: [
                    "cycle": ["$ref": "#/$defs/A"] as [String: any Sendable],
                    "deep": ["$ref": "#/$defs/D0"] as [String: any Sendable],
                ])
            let elapsed = ContinuousClock().measure {
                for key in ["cycle", "deep"] {
                    #expect(parameterKeepsText(paramName: key, funcName: "f", tools: tools))
                    #expect(
                        convertParameterValue(
                            "5", paramName: key, funcName: "f", tools: tools,
                            resolvingReferences: true) as? String == "5")
                }
            }
            #expect(elapsed < .milliseconds(250), "\(elapsed)")
        }

        /// Pointers read as the coordinator's `$ref` resolver reads them: array
        /// members are walked (`#/$defs/U/anyOf/0`, as zod's root strategy
        /// writes), each token is percent-decoded before `~1`/`~0`, and a
        /// malformed escape names nothing.
        @Test
        func qwenXMLReferencePointersReadAsTheCoordinatorReadsThem() {
            let tools = Self.referenceTools(
                defs: [
                    "U": [
                        "anyOf": [
                            ["type": "integer"] as [String: any Sendable],
                            ["type": "string"] as [String: any Sendable],
                        ]
                    ] as [String: any Sendable],
                    "a/b": ["type": "integer"] as [String: any Sendable],
                    "50%": ["type": "integer"] as [String: any Sendable],
                ],
                properties: [
                    "member": ["$ref": "#/$defs/U/anyOf/0"] as [String: any Sendable],
                    "escaped": ["$ref": "#/$defs/a~1b"] as [String: any Sendable],
                    "encoded": ["$ref": "#/$defs/a%7E1b"] as [String: any Sendable],
                    "malformed": ["$ref": "#/$defs/50%"] as [String: any Sendable],
                ])
            func read(_ key: String) -> JSONValue {
                JSONValue.from(
                    convertParameterValue(
                        "7", paramName: key, funcName: "f", tools: tools, resolvingReferences: true)
                )
            }
            #expect(read("member") == .int(7))
            #expect(read("escaped") == .int(7))
            #expect(read("encoded") == .int(7))
            #expect(read("malformed") == .string("7"))
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
