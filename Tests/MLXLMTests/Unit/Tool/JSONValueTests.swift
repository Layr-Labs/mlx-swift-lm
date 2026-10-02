import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct JSONValueTests {

        private func decode(_ json: String) throws -> JSONValue {
            try JSONDecoder().decode(JSONValue.self, from: Data(json.utf8))
        }

        @Test
        func decodesEachJSONKind() throws {
            #expect(try decode("null") == .null)
            #expect(try decode("true") == .bool(true))
            #expect(try decode("42") == .int(42))
            #expect(try decode("2.5") == .double(2.5))
            #expect(try decode("\"text\"") == .string("text"))
            #expect(try decode("[1, \"a\"]") == .array([.int(1), .string("a")]))
            #expect(
                try decode("{\"k\": {\"n\": null}}") == .object(["k": .object(["n": .null])]))
        }

        @Test
        func encodingThenDecodingGivesTheSameValue() throws {
            let value = JSONValue.object([
                "flag": .bool(false),
                "count": .int(-3),
                "ratio": .double(0.25),
                "name": .string("x"),
                "items": .array([.null, .int(1)]),
            ])
            let data = try JSONEncoder().encode(value)
            #expect(try JSONDecoder().decode(JSONValue.self, from: data) == value)
        }

        @Test
        func fromKeepsZeroAndOneAsNumbersInDecodedPayloads() throws {
            let payload = try JSONSerialization.jsonObject(
                with: Data(#"{"limit": 1, "offset": 0, "exact": true, "scale": 1.5}"#.utf8))
            let value = JSONValue.from(payload)
            #expect(
                value
                    == .object([
                        "limit": .int(1),
                        "offset": .int(0),
                        "exact": .bool(true),
                        "scale": .double(1.5),
                    ]))
        }

        @Test
        func fromMapsSwiftValues() {
            #expect(JSONValue.from(NSNull()) == .null)
            #expect(JSONValue.from("s") == .string("s"))
            #expect(JSONValue.from([1, 2] as [Any]) == .array([.int(1), .int(2)]))
            let nested: [String: any Sendable] = ["a": ["b": "c"] as [String: any Sendable]]
            #expect(JSONValue.from(nested) == .object(["a": .object(["b": .string("c")])]))
        }

        @Test
        func fromDescribesAnUnknownValueAsAString() {
            struct Opaque: CustomStringConvertible {
                var description: String { "opaque-value" }
            }
            #expect(JSONValue.from(Opaque()) == .string("opaque-value"))
        }

        @Test
        func anyValueGivesFoundationValues() throws {
            let value = JSONValue.object(["list": .array([.int(7), .null]), "on": .bool(true)])
            let any = try #require(value.anyValue as? [String: Any])
            let list = try #require(any["list"] as? [Any])
            #expect(list.first as? Int == 7)
            #expect(list.last is NSNull)
            #expect(any["on"] as? Bool == true)
            #expect(JSONValue.double(1.5).anyValue as? Double == 1.5)
            #expect(JSONValue.string("t").anyValue as? String == "t")
        }

        @Test
        func asSchemaGivesTheJSONSchemaType() throws {
            #expect(JSONValue.null.asSchema["type"] as? String == "null")
            #expect(JSONValue.bool(true).asSchema["type"] as? String == "boolean")
            #expect(JSONValue.int(1).asSchema["type"] as? String == "integer")
            #expect(JSONValue.double(1).asSchema["type"] as? String == "number")
            #expect(JSONValue.string("").asSchema["type"] as? String == "string")

            let emptyArray = JSONValue.array([]).asSchema
            #expect(emptyArray["type"] as? String == "array")
            #expect(emptyArray["items"] == nil)

            let array = JSONValue.array([.int(1)]).asSchema
            let items = try #require(array["items"] as? [String: any Sendable])
            #expect(items["type"] as? String == "integer")

            let object = JSONValue.object(["name": .string("n")]).asSchema
            #expect(object["type"] as? String == "object")
            let properties = try #require(object["properties"] as? [String: any Sendable])
            let name = try #require(properties["name"] as? [String: any Sendable])
            #expect(name["type"] as? String == "string")
        }
    }
}
