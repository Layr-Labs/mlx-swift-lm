// Copyright © 2026 Eigen Labs.
import Foundation

public struct Gemma4DecodeGluePolicy: Sendable {
    public let enabled: Bool
    public let assistant: Bool
    public let paired: Bool
    public let chained: Bool
    public let localRMSBroadcast: Bool

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        enabled = environment["DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE"] == "1"
        assistant = environment["DARKBLOOM_GEMMA4_DRAFTER_NORM_RESIDUAL_FUSE"] == "1"
        paired = environment["DARKBLOOM_GEMMA4_DECODE_PAIRED_RMS"] != "0"
        chained = environment["DARKBLOOM_GEMMA4_DECODE_GLUE_CHAIN"] != "0"
        localRMSBroadcast = environment["DARKBLOOM_GEMMA4_NORM_TG_BARRIER_HALVE"] == "1"
    }

    public func context(target: Bool, validatedAssistant: Bool) -> Context? {
        guard enabled else { return nil }
        if target { return Context(axis: 2816, paired: paired, chained: chained, localRMSBroadcast: localRMSBroadcast) }
        if assistant && validatedAssistant { return Context(axis: 1024, paired: false, chained: false, localRMSBroadcast: false) }
        return nil
    }

    public struct Context: Sendable {
        public let axis: Int
        public let paired: Bool
        public let chained: Bool
        public let localRMSBroadcast: Bool
        fileprivate init(axis: Int, paired: Bool, chained: Bool, localRMSBroadcast: Bool) {
            self.axis = axis; self.paired = paired; self.chained = chained
            self.localRMSBroadcast = localRMSBroadcast
        }
        public func rows(shape: [Int], inputBF16: Bool, weightShape: [Int], weightBF16: Bool, eps: Float) -> Int? {
            guard shape.count == 3, shape[0] > 0, shape[1] == 1, shape[2] == axis,
                inputBF16, weightBF16, weightShape == [axis], eps == Float(1e-6),
                shape[0] <= Int(Int32.max) / (axis / 4),
                shape[0] <= Int(UInt32.max) / axis else { return nil }
            return shape[0]
        }
    }
}
