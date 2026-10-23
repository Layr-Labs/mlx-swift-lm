// TEMPORARY diagnostic (branch diag/pr339-flakes only; never merged).
// Runs the float32 rowwise serial and batched traces twice each and reports
// which traces agree, so a mismatch can be pinned to one side.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.tags(.integration))
struct DiagNemotronStateTests {
    typealias T = NemotronSuperBatchedDecodeTests

    static func compare(_ name: String, _ x: T.Trace, _ y: T.Trace) {
        var logitDiffs = 0
        for (lhs, rhs) in zip(x.logits, y.logits) {
            for (a, b) in zip(lhs, rhs) where !modelOptBitwiseEqual(a, b) { logitDiffs += 1 }
        }
        var lines: [String] = []
        for (request, (lhs, rhs)) in zip(x.snapshots, y.snapshots).enumerated() {
            for (index, (a, b)) in zip(lhs, rhs).enumerated() where !modelOptBitwiseEqual(a, b) {
                let ua = a.view(dtype: .uint32).asType(.int64)
                let ub = b.view(dtype: .uint32).asType(.int64)
                let ulps = abs(ua - ub)
                let count = (ulps .> 0).sum().item(Int.self)
                let maxUlp = ulps.max().item(Int.self)
                lines.append("r\(request)a\(index):n=\(count),maxulp=\(maxUlp)")
            }
        }
        print(
            "DIAGNEMO \(name): logitDiffs=\(logitDiffs) stateDiffs=\(lines.count) "
                + lines.joined(separator: " "))
    }

    @Test func floatRowwiseAgreementMatrix() throws {
        let model = try T.Fixture.model(seed: 31, dtype: .float32, products: .rowwise)
        let s1 = try T.serial(model, T.requests)
        let b1 = try T.batched(model, T.requests, products: .rowExact)
        let s2 = try T.serial(model, T.requests)
        let b2 = try T.batched(model, T.requests, products: .rowExact)
        Self.compare("S1-S2", s1, s2)
        Self.compare("B1-B2", b1, b2)
        Self.compare("S1-B1", s1, b1)
        Self.compare("S2-B2", s2, b2)
        Self.compare("S1-B2", s1, b2)
    }
}
