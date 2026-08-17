// AttentionExecutionPolicy.swift
//
// Fail-closed selection of CBv2's ordinary and explicitly forced-fused SDPA
// routes. Architecture identity is model metadata; runtime tensor properties
// are checked separately so a matching shape can never qualify a model.

import Foundation
import MLX

/// Model-supplied qualification for attention execution optimizations.
public struct CBv2AttentionExecutionQualification: Sendable, Equatable {
    public enum Architecture: Sendable, Equatable {
        case qwenLike
    }

    public enum AutomaticOptimization: Sendable, Equatable {
        case forcedFusedD256BF16FullAttentionPrefill
    }

    public var architecture: Architecture
    public var automaticOptimization: AutomaticOptimization?

    public init(
        architecture: Architecture,
        automaticOptimization: AutomaticOptimization? = nil
    ) {
        self.architecture = architecture
        self.automaticOptimization = automaticOptimization
    }
}

/// Operator control for the CBv2 attention execution route.
public enum CBv2AttentionExecutionControl: String, Sendable, Equatable {
    /// Kill switch and default: retain the existing SDPA and query-block paths.
    case fallback
    /// Explicit A/B arm: force the fused kernel for exactly eligible calls.
    case fused
    /// Production selection: also requires model-supplied optimization approval.
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

/// Small, injectable policy shared by the contiguous and paged CBv2 caches.
public struct CBv2AttentionExecutionPolicy: Sendable, Equatable {
    public static let environmentVariable = "DARKBLOOM_CBV2_ATTN_EXECUTION"

    public var control: CBv2AttentionExecutionControl

    public init(control: CBv2AttentionExecutionControl) {
        self.control = control
    }

    /// Process-level production control. Missing and invalid values fail closed.
    public static var production: Self {
        Self(
            control: .parse(
                ProcessInfo.processInfo.environment[environmentVariable]))
    }

    /// The fallback policy used by lower-level direct callers that do not opt in.
    public static let fallback = Self(control: .fallback)

    func route(
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
            queryLength > 1,
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
            guard kind.attentionExecutionQualification?.automaticOptimization
                == .forcedFusedD256BF16FullAttentionPrefill
            else {
                return .fallback
            }
            return .forcedFused
        }
    }

    /// Query blocking remains authoritative unless this exact call selected
    /// forced-fused execution. This is the only blocking bypass.
    @inline(__always)
    func shouldBlockQueries(
        queryLength: Int, blockSize: Int, route: CBv2AttentionExecutionRoute
    ) -> Bool {
        route == .fallback && blockSize > 0 && queryLength > blockSize
    }
}
