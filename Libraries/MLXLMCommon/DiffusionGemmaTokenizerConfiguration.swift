import CoreFoundation
import Foundation

/// The released Gemma tokenizer preserves decoded whitespace by default.
/// swift-transformers otherwise assumes cleanup=true when the key is absent,
/// which changes literal punctuation and quoted tool values after generation.
/// Resolve this family default in memory, not by rewriting checkpoint files.
public enum DiffusionGemmaTokenizerConfiguration {
    public struct Inputs: Sendable {
        private let configuration: Data
        private let data: Data

        fileprivate init(configuration: Data, data: Data) {
            self.configuration = configuration
            self.data = data
        }

        public func configurationDictionary() throws -> [NSString: Any] {
            try DiffusionGemmaTokenizerConfiguration.dictionary(configuration)
        }

        public func dataDictionary() throws -> [NSString: Any] {
            try DiffusionGemmaTokenizerConfiguration.dictionary(data)
        }
    }

    /// nil leaves every other processor on the existing loader path. The
    /// processor identity also works for a separate tokenizer-only directory.
    public static func load(from directory: URL) throws -> Inputs? {
        let configurationURL = directory.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: configurationURL.path) else { return nil }
        var configuration = try dictionary(Data(contentsOf: configurationURL))
        guard configuration["processor_class"] as? String == "DiffusionGemma4Processor" else {
            return nil
        }
        guard let tokenizerClass = configuration["tokenizer_class"] as? String,
            ["GemmaTokenizer", "GemmaTokenizerFast"].contains(tokenizerClass)
        else { throw CocoaError(.coderReadCorrupt) }
        if let explicit = configuration["clean_up_tokenization_spaces"] {
            guard CFGetTypeID(explicit as CFTypeRef) == CFBooleanGetTypeID() else {
                throw CocoaError(.coderReadCorrupt)
            }
        } else {
            configuration["clean_up_tokenization_spaces"] = false
        }
        // Preserve the existing local loader's external-template precedence.
        let jinja = directory.appendingPathComponent("chat_template.jinja")
        let json = directory.appendingPathComponent("chat_template.json")
        if FileManager.default.fileExists(atPath: jinja.path) {
            configuration["chat_template"] = try String(contentsOf: jinja, encoding: .utf8)
        } else if FileManager.default.fileExists(atPath: json.path),
            let template = try dictionary(Data(contentsOf: json))["chat_template"] as? String
        {
            configuration["chat_template"] = template
        }
        return Inputs(configuration: try JSONSerialization.data(withJSONObject: configuration),
            data: try Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
    }

    private static func dictionary(_ data: Data) throws -> [NSString: Any] {
        // NSString keys preserve binary-distinct Unicode vocabulary entries,
        // matching the upstream Hub configuration reader.
        guard let value = try JSONSerialization.jsonObject(with: data) as? [NSString: Any] else {
            throw CocoaError(.coderReadCorrupt)
        }
        return value
    }
}
