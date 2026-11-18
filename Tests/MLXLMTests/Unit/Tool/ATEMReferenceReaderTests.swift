// Copyright © 2026 Eigen Labs.

import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// Each case the strict `ATEMToolCallParser` rejects comes back as a call,
    /// as the reference servers return it. An integral float is read alike by both.
    @Suite
    struct ATEMReferenceReaderTests {
        private static let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "parameters": [
                        "type": "object", "required": ["city"],
                        "additionalProperties": false,
                        "properties": [
                            "city": ["type": "string"], "days": ["type": "integer"],
                            "tags": ["type": "array", "items": ["type": "string"]]
                                as [String: any Sendable],
                            "trip": ["$ref": "#/$defs/Trip"],
                            "note": ["anyOf": [["$ref": "#/$defs/Trip"], ["type": "null"]]],
                        ] as [String: any Sendable],
                        "$defs": [
                            "Trip": [
                                "type": "object",
                                "properties": ["from": ["type": "string"]]
                                    as [String: any Sendable],
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        private static func invoke(_ name: String?, _ parameters: [(String, String)]) -> String {
            "<atem:invoke" + (name.map { " name=\"\($0)\"" } ?? "") + ">\n"
                + parameters.map { "<atem:parameter name=\"\($0.0)\">\($0.1)</atem:parameter>\n" }
                .joined() + "</atem:invoke>"
        }

        private static func read(_ name: String?, _ parameters: [(String, String)])
            -> ATEMReferenceReader.Reading
        {
            ATEMReferenceReader(tools: tools).read(invoke(name, parameters))
        }

        private static func strict(_ name: String, _ parameters: [(String, String)]) -> ToolCall? {
            ATEMToolCallParser().parse(
                content: "<atem:function_calls>\n" + invoke(name, parameters)
                    + "\n</atem:function_calls>",
                tools: tools)
        }

        @Test func anUndeclaredNameIsReturnedAsGenerated() throws {
            #expect(Self.strict("delete_everything", [("path", "/")]) == nil)
            let reading = Self.read("delete_everything", [("path", "/")])
            let call = try #require(reading.calls.first)
            #expect(call.toolCall.function.name == "delete_everything")
            #expect(call.toolCall.function.arguments == ["path": .string("/")])
            #expect(!call.isDeclared)
        }

        @Test func theDoubledNameIsTheDeclaredTool() throws {
            let call = try #require(
                Self.read("get_weather.get_weather", [("city", "Paris")]).calls.first)
            #expect(call.toolCall.function.name == "get_weather")
            #expect(call.isDeclared)
        }

        @Test func aMissingRequiredParameterOrAnExtraOneIsStillACall() throws {
            #expect(Self.strict("get_weather", [("days", "3")]) == nil)
            #expect(Self.strict("get_weather", [("city", "Paris"), ("mood", "sunny")]) == nil)
            let missing = try #require(Self.read("get_weather", [("days", "3")]).calls.first)
            #expect(missing.toolCall.function.arguments == ["days": .int(3)])
            let extra = try #require(
                Self.read("get_weather", [("city", "Paris"), ("mood", "sunny")]).calls.first)
            #expect(
                extra.toolCall.function.arguments == [
                    "city": .string("Paris"), "mood": .string("sunny"),
                ])
        }

        @Test func anIntegralFloatIsAnIntegerAndAWrongTypeKeepsItsText() throws {
            // The strict parser also reads an integral float as the integer.
            #expect(
                Self.strict("get_weather", [("city", "Paris"), ("days", "6.0")])?.function
                    .arguments["days"] == .int(6))
            let integral = try #require(
                Self.read("get_weather", [("city", "Paris"), ("days", "6.0")]).calls.first)
            #expect(integral.toolCall.function.arguments["days"] == .int(6))
            let words = try #require(
                Self.read("get_weather", [("city", "Paris"), ("days", "soon")]).calls.first)
            #expect(words.toolCall.function.arguments["days"] == .string("soon"))
        }

        @Test func invalidJSONForAnArrayKeepsItsText() throws {
            #expect(Self.strict("get_weather", [("city", "Paris"), ("tags", "[a, b")]) == nil)
            let call = try #require(
                Self.read("get_weather", [("city", "Paris"), ("tags", "[a, b")]).calls.first)
            #expect(call.toolCall.function.arguments["tags"] == .string("[a, b"))
        }

        @Test func aStringParameterKeepsItsTextByteForByte() throws {
            for text in ["12345", "true", "null", " padded ", "{\"a\": 1}"] {
                let call = try #require(Self.read("get_weather", [("city", text)]).calls.first)
                #expect(call.toolCall.function.arguments["city"] == .string(text), "\(text)")
            }
        }

        @Test func aReferencedObjectAndAnOptionalOneAreDecoded() throws {
            let call = try #require(
                Self.read(
                    "get_weather",
                    [("city", "Paris"), ("trip", #"{"from": "Lyon"}"#), ("note", "null")]
                ).calls.first)
            #expect(call.toolCall.function.arguments["trip"] == .object(["from": .string("Lyon")]))
            #expect(call.toolCall.function.arguments["note"] == .null)
            let optional = try #require(
                Self.read("get_weather", [("city", "Paris"), ("note", #"{"from": "Nice"}"#)]).calls
                    .first)
            #expect(
                optional.toolCall.function.arguments["note"] == .object(["from": .string("Nice")]))
        }

        @Test func anInvokeWithoutANameOrWithoutItsCloseIsNoCall() {
            let unnamed = Self.read(nil, [("city", "Paris")])
            #expect(unnamed.calls.isEmpty)
            #expect(unnamed.unnamedInvokes == 1)
            let open = ATEMReferenceReader(tools: Self.tools).read(
                "<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Par")
            #expect(open.calls.isEmpty)
            #expect(open.endsInsideInvoke)
        }

        @Test func everyInvokeOfAFrameIsReadInOrderWithAnID() throws {
            let frame =
                "<atem:function_calls>\n" + Self.invoke("get_weather", [("city", "Paris")]) + "\n"
                + Self.invoke("other", []) + "\n</atem:function_calls>"
            let reading = ATEMReferenceReader(tools: Self.tools).read(frame)
            #expect(reading.calls.map(\.toolCall.function.name) == ["get_weather", "other"])
            #expect(reading.calls.map(\.isDeclared) == [true, false])
            #expect(reading.calls.allSatisfy { $0.toolCall.id?.isEmpty == false })
            // Without declarations every name is undeclared and every value is decoded.
            let bare = ATEMReferenceReader(tools: nil).read(
                Self.invoke("get_weather", [("days", "3")]))
            #expect(bare.calls.first?.isDeclared == false)
            #expect(bare.calls.first?.toolCall.function.arguments == ["days": .int(3)])
        }
    }
}
