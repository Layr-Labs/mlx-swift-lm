// Copyright © 2026 Eigen Labs.
import Foundation

public struct Gemma4RouterFinalistsPolicy: Sendable {
    public let enabled: Bool
    public let prefill: Bool
    public let weights: Bool
    public let nativeOrderKeys: Bool

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        enabled = environment["DARKBLOOM_GEMMA4_ROUTER_FINALISTS32"] == "1"
        prefill = environment["DARKBLOOM_GEMMA4_ROUTER_FINALISTS32_PREFILL"] == "1"
        weights = environment["DARKBLOOM_GEMMA4_ROUTER_WEIGHTS32_PREFILL"] == "1"
        nativeOrderKeys = environment["DARKBLOOM_GEMMA4_PREFILL_ROUTE_ORDER_KEYS"] != "0"
    }

    public struct Plan: Sendable {
        public let scoreShape: [Int]
        public let rows: Int
        public let threads: Int
        public let fusedWeights: Bool
        public let nativeOrderKeys: Bool
    }

    public func plan(targetEligible: Bool, shape: [Int], scoresBF16: Bool, scaleBF16: Bool,
                     scaleShape: [Int], topK: Int, scheduledPrefill: Bool) -> Plan? {
        guard enabled, targetEligible, topK == 8, scoresBF16,
            shape.count == 3, shape[0] > 0, shape[1] > 0, shape[2] == 128 else { return nil }
        if shape[1] > 1 && !(prefill && scheduledPrefill) { return nil }
        let count = shape[0].multipliedReportingOverflow(by: shape[1])
        guard !count.overflow, count.partialValue <= Int(Int32.max) / 128 else { return nil }
        let fuse = shape[1] > 1 && weights && scaleBF16 && scaleShape == [128]
        let useKeys = fuse && nativeOrderKeys
        return Plan(scoreShape: shape, rows: count.partialValue, threads: useKeys ? 32 : 128,
                    fusedWeights: fuse, nativeOrderKeys: useKeys)
    }
}
