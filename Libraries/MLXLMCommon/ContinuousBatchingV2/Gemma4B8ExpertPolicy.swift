// Copyright © 2026 Eigen Labs.
import Foundation

/// Exact B8 single-token execution only; no controller or cache policy changes.
public struct Gemma4B8ExpertPolicy: Sendable {
    public let enabled: Bool
    public let tightDown: Bool
    public let compiled: Bool
    public let runCap: Int
    public let tileSpan: Int
    public let packedWordLoads: Bool

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        enabled = environment["DARKBLOOM_GEMMA4_B8_EXPERT_EXECUTION"] == "1"
            && environment["DARKBLOOM_GEMMA4_DECODE_FUSED_GEGLU"] != "0"
        tightDown = environment["DARKBLOOM_GEMMA4_DOWN_TIGHT_GRID"] != "0"
        compiled = environment["DARKBLOOM_GEMMA4_COMPILED_GU_DOWN"] == "1"
        let cap = Int(environment["DARKBLOOM_GEMMA4_GU_RUN_CAP"] ?? "4") ?? 4
        // The retained leader election uses a power-of-two mask.
        runCap = [1, 2, 4].contains(cap) ? cap : 4
        tileSpan = environment["DARKBLOOM_GEMMA4_DOWN_TILE_SPAN2"] == "0" ? 4 : 1
        packedWordLoads = environment["DARKBLOOM_GEMMA4_DOWN_PACKED_WORD_LOAD"] != "0"
    }

    public func admits(targetEligible: Bool, inputShape: [Int], inputBF16: Bool,
                       scheduledPrefill: Bool, compiledActivation: Bool) -> Bool {
        enabled && targetEligible && inputShape == [8, 1, 2816] && inputBF16
            && !scheduledPrefill && compiledActivation
    }

    public static let pairedStorageBytesPerLayer = 285_474_816
}
