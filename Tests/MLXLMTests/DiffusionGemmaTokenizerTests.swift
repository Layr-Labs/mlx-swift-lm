import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@Suite("DiffusionGemma native tokenizer decoding")
struct DiffusionGemmaTokenizerTests {
    private func directory(processor: String = "DiffusionGemma4Processor", cleanup: Bool? = nil) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("diffusion-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        do {
            var config: [String: Any] = ["tokenizer_class": "GemmaTokenizer",
                "processor_class": processor, "unk_token": "<unk>"]
            if let cleanup { config["clean_up_tokenization_spaces"] = cleanup }
            var vocabulary: [String: Int] = ["<unk>": 0]
            for value in 32 ... 126 { vocabulary[String(UnicodeScalar(value)!)] = value - 31 }
            let data: [String: Any] = ["model": ["type": "BPE", "vocab": vocabulary,
                "merges": [String](), "unk_token": "<unk>"], "decoder": ["type": "Fuse"]]
            try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("tokenizer_config.json"))
            try JSONSerialization.data(withJSONObject: data).write(to: directory.appendingPathComponent("tokenizer.json"))
            try Data("{{ messages[0]['content'] }}".utf8).write(to: directory.appendingPathComponent("chat_template.jinja"))
            return directory
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    @Test func nativeDefaultPreservesWhitespaceAndPunctuationWithoutChangingIDs() async throws {
        let directory = try directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = try Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        let rawLoader = try await AutoTokenizer.from(modelFolder: directory)
        for text in ["a . b , c ? d !", "x ' value ' y", "we 're", "I 'm",
            "it 's", "they 've", "do n't", "function 'record_payload'", "  exact  spaces  "] {
            let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
            #expect(tokens == rawLoader.encode(text: text, addSpecialTokens: false))
            #expect(tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false) == text)
            #expect(tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true) == text)
            #expect(try tokenizer.applyChatTemplate(messages: [["role": "user", "content": text]]) == tokens)
        }
        #expect(try Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")) == original,
            "Loading defaults must not modify immutable artifact metadata")
    }

    @Test func explicitNativeValuesAreRespected() async throws {
        for cleanup in [false, true] {
            let directory = try directory(cleanup: cleanup)
            defer { try? FileManager.default.removeItem(at: directory) }
            let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
            let tokens = tokenizer.encode(text: "a .", addSpecialTokens: false)
            #expect(tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false) == (cleanup ? "a." : "a ."))
        }
    }

    @Test func unrelatedProcessorKeepsExistingLoaderBehavior() async throws {
        let directory = try directory(processor: "Gemma4Processor")
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        let tokens = tokenizer.encode(text: "a .", addSpecialTokens: false)
        #expect(tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false) == "a.")
    }

    @Test func nativeMalformedExplicitCleanupIsRejected() throws {
        let directory = try directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("tokenizer_config.json")
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        for invalid: Any in ["false", 0, 1, NSNull()] {
            config["clean_up_tokenization_spaces"] = invalid
            try JSONSerialization.data(withJSONObject: config).write(to: file)
            #expect(throws: CocoaError.self) { try DiffusionGemmaTokenizerConfiguration.load(from: directory) }
        }
    }
}
