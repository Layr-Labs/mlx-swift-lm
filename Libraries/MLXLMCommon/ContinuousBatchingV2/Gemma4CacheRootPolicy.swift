// Copyright © 2026 Eigen Labs.
import Foundation

public enum Gemma4CacheEvaluationScope: Sendable, Equatable { case decode, mtpVerify }

public struct Gemma4CacheRootPolicy: Sendable {
    public static let process = Gemma4CacheRootPolicy()
    public let decode: Bool
    public let verify: Bool
    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        decode = environment["DARKBLOOM_GEMMA4_COMPACT_DECODE_ROOTS"] == "1"
        verify = environment["DARKBLOOM_GEMMA4_COMPACT_MTP_ROOTS"] == "1"
    }
    public func enabled(_ scope: Gemma4CacheEvaluationScope) -> Bool {
        scope == .decode ? decode : verify
    }

    static func validates(binding: UInt64, updates: UInt64, previousBinding: UInt64,
                          previousUpdates: UInt64, expectedUpdates: Int, expectedWidth: Int,
                          completedWidth: Int?, idle: Bool) -> Bool {
        expectedUpdates > 0 && expectedWidth > 0 && idle && binding == previousBinding
            && updates &- previousUpdates == UInt64(expectedUpdates) && completedWidth == expectedWidth
    }
}
