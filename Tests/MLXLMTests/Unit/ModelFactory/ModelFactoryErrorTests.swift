import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct ModelFactoryErrorTests {

        private struct Sample: Decodable {
            let outer: Inner
            struct Inner: Decodable {
                let value: Int
            }
        }

        private func decodingError(_ json: String) throws -> DecodingError {
            do {
                _ = try JSONDecoder().decode(Sample.self, from: Data(json.utf8))
            } catch let error as DecodingError {
                return error
            }
            throw TestFailure()
        }

        private struct TestFailure: Error {}

        private func describe(_ json: String) throws -> String? {
            ModelFactoryError.configurationDecodingError(
                "config.json", "org/model", try decodingError(json)
            ).errorDescription
        }

        @Test
        func simpleCasesDescribeTheirValue() {
            #expect(
                ModelFactoryError.unsupportedModelType("abc").errorDescription
                    == "Unsupported model type: abc")
            #expect(
                ModelFactoryError.unsupportedProcessorType("proc").errorDescription
                    == "Unsupported processor type: proc")
            #expect(
                ModelFactoryError.noModelFactoryAvailable.errorDescription
                    == "No model factory available via ModelFactoryRegistry")
        }

        @Test
        func fileErrorNamesTheFileAndTheModel() {
            let underlying = CocoaError(.fileReadNoSuchFile)
            let description = ModelFactoryError.configurationFileError(
                "config.json", "org/model", underlying
            ).errorDescription
            #expect(
                description
                    == "Error reading 'config.json' for model 'org/model': \(underlying.localizedDescription)"
            )
        }

        @Test
        func decodingErrorsNameTheFieldPath() throws {
            #expect(
                try describe(#"{"outer": {}}"#)
                    == "Failed to parse config.json for model 'org/model': Missing field 'outer.value'"
            )
            #expect(
                try describe(#"{"outer": {"value": "x"}}"#)
                    == "Failed to parse config.json for model 'org/model': Type mismatch at 'outer.value'"
            )
            #expect(
                try describe(#"{"outer": {"value": null}}"#)
                    == "Failed to parse config.json for model 'org/model': Missing value at 'outer.value'"
            )
            #expect(
                try describe("not json")
                    == "Failed to parse config.json for model 'org/model': Invalid JSON")
        }

        @Test
        func dataCorruptedWithAPathNamesThePath() {
            let error = DecodingError.dataCorrupted(
                .init(codingPath: [AnyKey("rope_scaling")], debugDescription: "bad"))
            #expect(
                ModelFactoryError.configurationDecodingError("config.json", "m", error)
                    .errorDescription
                    == "Failed to parse config.json for model 'm': Invalid data at 'rope_scaling'")
        }

        private struct AnyKey: CodingKey {
            var stringValue: String
            var intValue: Int? { nil }
            init(_ value: String) { stringValue = value }
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { nil }
        }

        @Test
        func directoryErrorsDescribeTheUnresolvedSource() {
            #expect(
                ModelConfiguration.DirectoryError.unresolvedModelDirectory("org/m").errorDescription
                    == "Model configuration 'org/m' has not been resolved to a local directory.")
            #expect(
                ModelConfiguration.DirectoryError.unresolvedTokenizerDirectory("org/t")
                    .errorDescription
                    == "Tokenizer source 'org/t' has not been resolved to a local directory.")
        }
    }
}
