// Copyright © 2026 Eigen Labs.
import Foundation

public struct Gemma4ScaledEmbeddingPolicy: Sendable {
    public let enabled: Bool
    public let decode: Bool
    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        enabled = environment["DARKBLOOM_GEMMA4_SCALED_EMBEDDING"] == "1"
        decode = environment["DARKBLOOM_GEMMA4_SCALED_EMBEDDING_DECODE"] == "1"
    }
    public struct Plan: Sendable {
        public let rows: Int
        public let outputShape: [Int]
    }
    public func plan(targetEligible: Bool, tokenShape: [Int], tokensInt32: Bool,
                     vocab: Int, hidden: Int, bits: Int, groupSize: Int) -> Plan? {
        guard enabled, targetEligible, tokenShape.count == 2, tokenShape[0] > 0,
            tokenShape[1] > 0, tokenShape[1] > 1 || decode, tokensInt32,
            vocab > 0, vocab <= Int(Int32.max), hidden == 2816, bits == 4, groupSize == 64 else { return nil }
        let rows = tokenShape[0].multipliedReportingOverflow(by: tokenShape[1])
        guard !rows.overflow, rows.partialValue <= Int(Int32.max) else { return nil }
        return Plan(rows: rows.partialValue, outputShape: tokenShape + [hidden])
    }
}
