import Foundation
import Testing

@testable import BenchCBv2Core

@Suite("Benchmark complete token receipts")
struct BenchCBv2TokenReceiptTests {
    @Test func summarizedCellsRetainRepeatedTokensAndRequestOrder() throws {
        let first = RunResult(
            tokens: [9, 9, 31], finish: "length", promptTokens: 37,
            completionTokens: 3, submittedAt: 1, firstTokenAt: 1.1,
            lastTokenAt: 1.3, finishedAt: 1.4)
        let second = RunResult(
            tokens: [31, 9], finish: "stop", promptTokens: 41,
            completionTokens: 2, submittedAt: 1, firstTokenAt: 1.2,
            lastTokenAt: 1.4, finishedAt: 1.5)
        let cell = summarize(
            engine: "v2", batch: 2, promptMix: [37, 41],
            results: [first, second])
        let encoded = try benchmarkJSONString(cell)
        let restored = try JSONDecoder().decode(CellResult.self, from: Data(encoded.utf8))
        let receipts = try #require(restored.tokenReceipts)
        #expect(receipts.map(\.tokenIDs) == [[9, 9, 31], [31, 9]])
        #expect(receipts.map(\.finishReason) == ["length", "stop"])
        #expect(receipts.map(\.promptTokens) == [37, 41])
        #expect(receipts.map(\.completionTokens) == [3, 2])
    }

    @Test func historicalCellJSONWithoutTokenReceiptsStillDecodes() throws {
        let cell = summarize(engine: "v2", batch: 1, promptMix: [37], results: [])
        var historical = try #require(
            JSONSerialization.jsonObject(
                with: Data(try benchmarkJSONString(cell).utf8)) as? [String: Any])
        historical.removeValue(forKey: "tokenReceipts")
        let decoded = try JSONDecoder().decode(
            CellResult.self,
            from: JSONSerialization.data(withJSONObject: historical))
        #expect(decoded.tokenReceipts == nil)
        #expect(decoded.engine == "v2")
    }
}
