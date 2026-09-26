import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@Suite("DiffusionGemma released tokenizer oracle", .serialized)
struct DiffusionGemmaTokenizerLiveTests {
    private struct Oracle: Decodable {
        struct Case: Decodable {
            let id: String
            let text: String
            let tokenIds: [Int]
            let decoded: String
            let decodedSkippingSpecial: String
        }
        let cleanup: Bool
        let cases: [Case]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_TOKENIZER_PARITY"] == "1"))
    func actualTokenizerMatchesIndependentIDsAndDecodedBytes() async throws {
        let environment = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: try #require(environment["DARKBLOOM_DIFFUSION_TOKENIZER_DIR"]))
        let oracleURL = URL(fileURLWithPath: try #require(environment["DARKBLOOM_DIFFUSION_TOKENIZER_ORACLE"]))
        let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: oracleURL))
        try #require(!oracle.cleanup && oracle.cases.count == 16)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        for item in oracle.cases {
            #expect(tokenizer.encode(text: item.text, addSpecialTokens: false) == item.tokenIds,
                "Independent encoding: \(item.id)")
            #expect(Array(tokenizer.decode(tokenIds: item.tokenIds, skipSpecialTokens: false).utf8)
                == Array(item.decoded.utf8), "Exact decoded bytes: \(item.id)")
            #expect(Array(tokenizer.decode(tokenIds: item.tokenIds, skipSpecialTokens: true).utf8)
                == Array(item.decodedSkippingSpecial.utf8), "Special-token filtering: \(item.id)")
        }
    }
}
