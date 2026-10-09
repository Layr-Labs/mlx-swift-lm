import Foundation

/// Complete emitted-token identity for one benchmark request, in request order.
/// Optional inclusion in older cell receipts preserves their decoding contract.
struct RequestTokenReceipt: Codable, Equatable, Sendable {
    let tokenIDs: [Int]
    let finishReason: String
    let promptTokens: Int
    let completionTokens: Int

    init(_ result: RunResult) {
        tokenIDs = result.tokens
        finishReason = result.finish
        promptTokens = result.promptTokens
        completionTokens = result.completionTokens
    }
}
