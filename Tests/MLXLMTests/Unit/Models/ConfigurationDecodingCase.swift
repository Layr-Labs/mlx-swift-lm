import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    /// A model configuration type with a minimal `config.json` and the stored
    /// fields that decoding it must give.
    ///
    /// `requiredKeys` are the JSON key paths (joined with ".") that decoding
    /// needs. `fields` are the lines of `UnitTests.storedFields(of:)` for the
    /// decoded value: the values from the JSON and the defaults of the other
    /// fields. The configurations are decoded with `JSONDecoder.json5()`, as
    /// the model factories do.
    struct ConfigurationDecodingCase: Sendable, CustomTestStringConvertible {
        let name: String
        let json: String
        let requiredKeys: [String]
        let fields: [String]
        private let decodeFields: @Sendable (Data) throws -> [String]

        init<T: Decodable>(
            _ name: String, _ type: T.Type, json: String, requiredKeys: [String],
            fields: [String]
        ) {
            self.name = name
            self.json = json
            self.requiredKeys = requiredKeys
            self.fields = fields
            self.decodeFields = { data in
                UnitTests.storedFields(of: try JSONDecoder.json5().decode(T.self, from: data))
            }
        }

        var testDescription: String { name }

        /// Decodes the minimal JSON and compares the stored fields.
        func expectMinimalJSONGivesTheFields() throws {
            let decoded = try decodeFields(Data(json.utf8))
            #expect(decoded == fields, "\(name)")
        }

        /// Removes each required key in turn and expects `keyNotFound` for
        /// that key.
        func expectEachRequiredKeyIsRequired() throws {
            for key in requiredKeys {
                var object: Any = try JSONSerialization.jsonObject(with: Data(json.utf8))
                Self.remove(key.split(separator: ".").map(String.init), from: &object)
                let data = try JSONSerialization.data(withJSONObject: object)
                do {
                    _ = try decodeFields(data)
                    Issue.record("\(name) decoded without the required key '\(key)'")
                } catch DecodingError.keyNotFound(let missing, let context) {
                    let path = (context.codingPath + [missing]).map(\.stringValue)
                    #expect(path.joined(separator: ".") == key, "\(name)")
                }
            }
        }

        /// Expects a type mismatch when the JSON root is an array.
        func expectANonObjectRootIsRejected() {
            #expect(throws: DecodingError.self, "\(name)") {
                try decodeFields(Data("[]".utf8))
            }
        }

        private static func remove(_ path: [String], from object: inout Any) {
            guard var dictionary = object as? [String: Any], let first = path.first else { return }
            if path.count == 1 {
                dictionary[first] = nil
            } else if var child = dictionary[first] {
                remove(Array(path.dropFirst()), from: &child)
                dictionary[first] = child
            }
            object = dictionary
        }
    }
}
