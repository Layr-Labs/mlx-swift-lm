// Copyright © 2026 Eigen Labs.
import Foundation

public struct Gemma4QKVNormPolicy: Sendable {
    public let enabled: Bool
    public let prefill: Bool
    public let rope: Bool
    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        enabled = environment["DARKBLOOM_GEMMA4_QKV_NORM"] == "1"
        prefill = environment["DARKBLOOM_GEMMA4_QKV_NORM_PREFILL"] == "1"
        rope = environment["DARKBLOOM_GEMMA4_QKV_NORM_ROPE"] == "1"
    }
    public enum Kind: Sendable { case decode, fullPrefill, slidingPrefill }
    public struct Plan: Sendable {
        public let kind: Kind
        public let batch: Int
        public let queryLength: Int
        public let keyLength: Int
        public let keyHeads: Int
        public let dimension: Int
        public let queryRows: Int
        public let keyRows: Int
        public let totalRows: Int
        public let rowsPerGroup: Int
        public let threads: Int
        public let grid: Int
    }
    public func plan(targetEligible: Bool, q: [Int], k: [Int], v: [Int], bf16: Bool,
                     weightsMatch: Bool, eps: Float, keyValueShared: Bool,
                     scheduledPrefill: Bool) -> Plan? {
        guard enabled, targetEligible, bf16, weightsMatch, eps == Float(1e-6),
            q.count == 4, k.count == 4, v == k, q[0] > 0, q[1] > 0, k[1] > 0,
            q[0] == k[0], q[2] == 16, q[3] == k[3],
            (q[3] == 256 && k[2] == 8) || (q[3] == 512 && k[2] == 2) else { return nil }
        func product(_ values: [Int]) -> Int? {
            var value = 1
            for next in values {
                let p = value.multipliedReportingOverflow(by: next)
                guard !p.overflow, p.partialValue <= Int(UInt32.max) else { return nil }
                value = p.partialValue
            }
            return value
        }
        guard let qRows = product([q[0], q[1], 16]), let kRows = product([k[0], k[1], k[2]]),
            product([qRows, q[3]]) != nil, product([kRows, q[3]]) != nil else { return nil }
        let kind: Kind
        if q[1] == 1 && k[1] == 1 {
            kind = .decode
        } else {
            guard prefill, scheduledPrefill,
                let tokens = product([q[0], max(q[1], k[1])]), tokens >= 1024 else { return nil }
            if keyValueShared { kind = .fullPrefill }
            else if q[3] == 256 { kind = .slidingPrefill }
            else { return nil }
        }
        let total = qRows + kRows * (keyValueShared ? 1 : 2)
        guard total <= Int(UInt32.max) else { return nil }
        let rowThreads = q[3] / 4
        let rpt = kind == .decode ? 1 : 512 / rowThreads
        let groups = (total + rpt - 1) / rpt
        guard let grid = product([groups, rpt, rowThreads]), grid <= Int(Int32.max) else { return nil }
        return Plan(kind: kind, batch: q[0], queryLength: q[1], keyLength: k[1], keyHeads: k[2],
            dimension: q[3], queryRows: qRows, keyRows: kRows, totalRows: total,
            rowsPerGroup: rpt, threads: rpt * rowThreads, grid: grid)
    }
}
