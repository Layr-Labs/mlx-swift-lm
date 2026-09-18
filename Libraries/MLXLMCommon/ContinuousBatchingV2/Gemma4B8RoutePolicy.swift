// Copyright © 2026 Eigen Labs.
import Foundation

public struct Gemma4B8RoutePolicy: Sendable {
    public static let process = Gemma4B8RoutePolicy()
    public let rank: Bool
    public let prefixBounds: Bool
    public let directInput: Bool
    public let fold: Bool
    public let nativeOrderKeys: Bool

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        rank = environment["DARKBLOOM_GEMMA4_B8_EXPERT_EXECUTION"] == "1"
            && environment["DARKBLOOM_GEMMA4_B8_ROUTE_RANK"] == "1"
        prefixBounds = rank && environment["DARKBLOOM_GEMMA4_B8_ROUTE_PREFIX"] == "1"
        directInput = rank && environment["DARKBLOOM_GEMMA4_B8_ROUTE_DIRECT"] == "1"
        fold = rank && environment["DARKBLOOM_GEMMA4_B8_ROUTE_FOLD"] == "1"
        nativeOrderKeys = environment["DARKBLOOM_GEMMA4_B8_ROUTE_ORDER_KEYS"] != "0"
    }
}
