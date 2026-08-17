// AttentionExecutionPolicy.swift
//
// Fail-closed selection of CBv2's ordinary and explicitly forced-fused SDPA
// routes. Architecture identity is model metadata; runtime tensor properties
// are checked separately so a matching shape can never qualify a model.

import Foundation
import MLX

/// Model-supplied architecture identity. This is deliberately insufficient
/// for automatic optimization: hardware qualification comes from the caller.
public struct CBv2AttentionExecutionQualification: Sendable, Equatable {
    public enum Architecture: Sendable, Equatable {
        case qwenLike
    }

    public var architecture: Architecture

    public init(architecture: Architecture) {
        self.architecture = architecture
    }
}

/// External evidence that the forced-fused route is qualified on this
/// hardware/runtime combination. Model identity and tensor shape cannot mint
/// this value; the engine builder must supply it after hardware qualification.
public enum CBv2AttentionHardwareQualification: Sendable, Equatable {
    case qwenLikeD256Hq16Hkv2GQA8BF16FullAttentionPrefill
}

/// Operator control for the CBv2 attention execution route.
public enum CBv2AttentionExecutionControl: String, Sendable, Equatable {
    /// Kill switch and default: retain the existing SDPA and query-block paths.
    case fallback
    /// Explicit A/B arm: force the fused kernel for exactly eligible calls.
    case fused
    /// Production selection: also requires external hardware qualification.
    case auto

    static func parse(_ value: String?) -> Self {
        guard let value else { return .fallback }
        return Self(rawValue: value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
            ?? .fallback
    }
}

enum CBv2AttentionExecutionRoute: Sendable, Equatable {
    case fallback
    case forcedFused
}

enum CBv2AttentionExecutionPath: Sendable, Equatable {
    case singleCall
    case queryBlocked
    case span
    case serializedQueries
    case lastQuery
}

struct CBv2AttentionExecutionObservation: Sendable, Equatable {
    var route: CBv2AttentionExecutionRoute
    var path: CBv2AttentionExecutionPath
    var queryLength: Int
}

/// Small, injectable policy for the contiguous CBv2 attention cache.
public struct CBv2AttentionExecutionPolicy: Sendable, Equatable {
    public static let environmentVariable = "DARKBLOOM_CBV2_ATTN_EXECUTION"
    public static let queryBlockEnvironmentVariable = "DARKBLOOM_CBV2_ATTN_QUERY_BLOCK"

    /// Process-default fallback block width. Read once and parsed with the
    /// historical semantics: missing, non-integer, and negative values use
    /// 128; zero disables fallback blocking.
    public static let productionFallbackQueryBlockSize =
        parseFallbackQueryBlockSize(
            ProcessInfo.processInfo.environment[queryBlockEnvironmentVariable])

    public var control: CBv2AttentionExecutionControl
    public var hardwareQualification: CBv2AttentionHardwareQualification?
    public let fallbackQueryBlockSize: Int

    public init(
        control: CBv2AttentionExecutionControl,
        hardwareQualification: CBv2AttentionHardwareQualification? = nil,
        fallbackQueryBlockSize: Int? = nil
    ) {
        let fallbackQueryBlockSize =
            fallbackQueryBlockSize ?? Self.productionFallbackQueryBlockSize
        precondition(fallbackQueryBlockSize >= 0, "fallback query block size must be non-negative")
        self.control = control
        self.hardwareQualification = hardwareQualification
        self.fallbackQueryBlockSize = fallbackQueryBlockSize
    }

    /// Process-level production control. Missing and invalid values fail
    /// closed, and environment control alone never supplies hardware evidence.
    public static var production: Self {
        production(hardwareQualification: nil)
    }

    /// Process-level control paired with externally established hardware
    /// evidence. This is the production entry point for an enabled `auto`.
    public static func production(
        hardwareQualification: CBv2AttentionHardwareQualification?,
        fallbackQueryBlockSize: Int? = nil
    ) -> Self {
        Self(
            control: .parse(
                ProcessInfo.processInfo.environment[environmentVariable]),
            hardwareQualification: hardwareQualification,
            fallbackQueryBlockSize: fallbackQueryBlockSize)
    }

    /// The fallback policy used by lower-level direct callers that do not opt in.
    public static let fallback = Self(control: .fallback)

    func contiguousRoute(
        kind: CBv2LayerKind,
        queryLength: Int,
        queryDType: DType,
        attentionSoftcap: Float?,
        hasSpanMask: Bool,
        serializesQueries: Bool
    ) -> CBv2AttentionExecutionRoute {
        guard control != .fallback,
            kind.attentionExecutionQualification?.architecture == .qwenLike,
            kind.attention == .full,
            kind.sharesKVWithLayer == nil,
            !kind.isBidirectional,
            !kind.hasSinks,
            kind.headDim == 256,
            kind.queryHeads == 16,
            kind.kvHeads == 2,
            kind.queryHeads == kind.kvHeads * 8,
            Self.forcedFusedQueryLengthEligible(queryLength),
            queryDType == .bfloat16,
            attentionSoftcap == nil,
            !hasSpanMask,
            !serializesQueries
        else {
            return .fallback
        }

        switch control {
        case .fallback:
            return .fallback
        case .fused:
            return .forcedFused
        case .auto:
            guard hardwareQualification
                == .qwenLikeD256Hq16Hkv2GQA8BF16FullAttentionPrefill
            else {
                return .fallback
            }
            return .forcedFused
        }
    }

    /// The dependency exposes separate vector (qL 2...4) and full (qL > 8)
    /// fused implementations. qL 5...8 has no compatible forced terminal.
    @inline(__always)
    static func forcedFusedQueryLengthEligible(_ queryLength: Int) -> Bool {
        (2 ... 4).contains(queryLength) || queryLength > 8
    }

    /// Query blocking remains authoritative unless this exact call selected
    /// forced-fused execution. This is the only blocking bypass.
    @inline(__always)
    func shouldBlockQueries(
        queryLength: Int, route: CBv2AttentionExecutionRoute
    ) -> Bool {
        route == .fallback
            && fallbackQueryBlockSize > 0
            && queryLength > fallbackQueryBlockSize
    }

    static func parseFallbackQueryBlockSize(_ raw: String?) -> Int {
        guard let raw, let value = Int(raw), value >= 0 else { return 128 }
        return value
    }
}
