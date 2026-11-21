// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Prompt chunks built in MLX's prefill-row scope (`GPU.withPrefillRowInvariance`).
///
/// In the scope a quantized product of two or more rows splits K as a product of
/// `referenceRows` rows would, and head-dim 256 attention on NAX takes the fused kernel
/// from 16 queries. Paged prompt attention already blocks queries in 128s from the
/// chunk start, so a prompt then has the same bits however it is cut on multiples of
/// 128, and a prefix restored at such a position continues a cold run bit for bit.
/// Decode rows, MTP verification and drafts stay outside the scope.
public enum CBv2PrefillRowInvariance {

    /// A model family with its own gate.
    public enum Family: String, Sendable, CaseIterable {
        /// Dense Qwen 3.5 text models (`Qwen35Model` or `Qwen35TextModel` without
        /// experts): Qwen 3.5 9B, Qwen 3.8 27B.
        case qwen35Dense = "qwen3_5_dense"
    }

    /// Families whose chunk census and reuse proof passed. Only these can be on.
    public static let qualified: Set<Family> = [.qwen35Dense]

    /// Families on without the environment switch. Add a qualified family here to
    /// turn it on by default.
    public static let defaultOn: Set<Family> = []

    /// `1` (`on`, `true`, `yes`) turns the scope on for every qualified family, `0`
    /// (`off`, `false`, `no`) off for all; anything else keeps `defaultOn`. Read once.
    public static let environmentKey = "DARKBLOOM_PREFILL_ROW_INVARIANCE"

    static let environmentValue = ProcessInfo.processInfo.environment[environmentKey]

    /// The smallest reference row count: today's solo stripe.
    public static let minimumReferenceRows = 4096

    public static func enabled(_ family: Family?, environment raw: String?) -> Bool {
        guard let family, qualified.contains(family) else { return false }
        let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch value {
        case "1", "on", "true", "yes": return true
        case "0", "off", "false", "no": return false
        default: return defaultOn.contains(family)
        }
    }

    /// The longest chunk the scheduler can cut keeps today's bits.
    public static func referenceRows(prefillChunkSize: Int, soloPrefillStripeTokens: Int?)
        -> Int
    {
        max(minimumReferenceRows, prefillChunkSize, soloPrefillStripeTokens ?? 0)
    }

    /// The engine's reference rows for `model`, or 0 when the scope is off.
    static func engineRows(
        model: CBv2SteppableModel, config: CBv2SchedulerConfig,
        environment: String? = environmentValue
    ) -> Int {
        let family = (model as? CBv2PrefillRowInvarianceProviding)?.prefillRowInvarianceFamily
        guard enabled(family, environment: environment) else { return 0 }
        return referenceRows(
            prefillChunkSize: config.prefillChunkSize,
            soloPrefillStripeTokens: config.soloPrefillStripeTokens)
    }
}

/// A model that names its prefill-row invariance family (nil: none).
public protocol CBv2PrefillRowInvarianceProviding: AnyObject {
    var prefillRowInvarianceFamily: CBv2PrefillRowInvariance.Family? { get }
}

extension EngineLoopV2 {
    /// Builds a `.prefill` forward in the prefill-row scope when the model's family is on.
    func withPrefillRowInvariance<R>(phase: CBv2ForwardPhase, _ body: () -> R) -> R {
        guard phase == .prefill, prefillRowInvarianceRows > 0 else { return body() }
        return GPU.withPrefillRowInvariance(rows: prefillRowInvarianceRows, body)
    }
}
