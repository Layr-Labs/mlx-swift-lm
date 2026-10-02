// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

/// Policy only. Actual native geometry, device and owned scratch admission
/// remain mandatory at the existing attention/grouped dispatch boundaries.
public enum MiMoV26PrefillPolicy {
    public static let attentionEnvironmentKey = "DARKBLOOM_MIMO_V26_NAX_ATTENTION"
    public static let groupedEnvironmentKey = "DARKBLOOM_MIMO_BLOCK_BATCH_PREFILL"

    public enum ProcessControlMismatch: Error, Sendable, Equatable {
        case incompatible(environmentKey: String, requested: Bool, latched: Bool)
    }

    /// These are the exact cached constants read by kernel dispatch. They
    /// intentionally do not resample ProcessInfo after a control is latched.
    public static var latchedAttentionEnabled: Bool { MiMoV26NAXAttention.requested }
    public static var latchedGroupingEnabled: Bool { MiMoV26BlockBatchAttention.requested }

    /// Injected per-factory dictionaries cannot reconfigure process-wide
    /// kernels. Unset keys inherit the actual latches; contradictory explicit
    /// controls fail recoverably instead of reporting a width-only rollback.
    public static func validateProcessControls(environment: [String: String]) throws {
        try validateInjectedControls(
            environment: environment,
            latchedAttention: latchedAttentionEnabled, latchedGrouping: latchedGroupingEnabled)
    }

    // Scalar seam for the complete truth table. Production always supplies
    // the real dispatch constants through validateProcessControls above.
    static func validateInjectedControls(
        environment: [String: String],
        latchedAttention: Bool, latchedGrouping: Bool
    ) throws {
        for (key, latched) in [
            (attentionEnvironmentKey, latchedAttention),
            (groupedEnvironmentKey, latchedGrouping),
        ] {
            guard let value = environment[key] else { continue }
            let requested = isEnabled(value)
            guard requested == latched else {
                throw ProcessControlMismatch.incompatible(
                    environmentKey: key, requested: requested, latched: latched)
            }
        }
    }

    public static func isEnabled(_ value: String?) -> Bool {
        guard let value else { return true }
        return ["1", "true", "yes", "on", "auto"].contains(
            value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    /// Caller must opt into a default profile. An explicit SDK scheduler
    /// configuration is never widened merely by constructing an EngineV2.
    static func widerWidths(
        requested: Bool, naxAvailable: Bool,
        physicalMemoryBytes: UInt64, maximumTokens: Int?, originalWidth: Int
    ) -> [Int] {
        guard requested, naxAvailable, physicalMemoryBytes >= 128 * 1024 * 1024 * 1024,
            originalWidth > 0, let maximumTokens,
            maximumTokens == 4096 || maximumTokens == 8192
        else { return [] }
        return [8192, 4096].filter { $0 <= maximumTokens && $0 > originalWidth }
    }

    /// Replaces only the previous authentic MTP fixed term. All caller,
    /// target, projection and reserve charges remain in `current` untouched.
    /// A wider declaration may not lower the old MTP charge.
    static func replacingMTPCharge(current: Int, previous: Int, replacement: Int) -> Int? {
        guard previous >= 0, current >= previous, replacement >= previous else { return nil }
        let sum = (current - previous).addingReportingOverflow(replacement)
        return sum.overflow ? nil : sum.partialValue
    }

    /// Returns a new value only; refused candidates cannot change the
    /// caller's prior Admission configuration or its MTP charge.
    static func addingGroupedCharge(
        to original: AdmissionV2.Config,
        scratchBytes: Int, capacityBytes: Int
    ) -> AdmissionV2.Config? {
        guard original.fixedBytesPerRequest >= 0, scratchBytes > 0,
            let total = CBv2MTPBoundedAdmission.add(original.fixedBytesPerRequest, scratchBytes),
            total < capacityBytes
        else { return nil }
        var candidate = original
        candidate.fixedBytesPerRequest = total
        return candidate
    }
}

/// Immutable loaded-model metadata, not permission to allocate or to assert
/// scratch admission. The existing budget installer proves the cache owners.
package protocol MiMoV26PrefillDefaultProviding: AnyObject {
    var cbv2MiMoAutomaticPrefillMaximumTokens: Int? { get }
}

extension SwitchGLU {
    /// The actual existing custom gather can cover all three expert
    /// projections above 32768 sorted rows. No test eval or tensor copy.
    package var mimoV26SupportsOversizedPrefillGather: Bool {
        guard mimoV26NAXGather, MiMoV26NAXGatherQMM.requested,
            inputDims == 4096, hiddenDims == 2048, numExperts == 256
        else { return false }
        func matches(_ projection: SwitchLinear?, input: Int, output: Int) -> Bool {
            guard let p = projection as? QuantizedSwitchLinear,
                ObjectIdentifier(type(of: p)) == ObjectIdentifier(QuantizedSwitchLinear.self),
                p.bias == nil, p.biases == nil, p.mode == .mxfp4,
                p.bits == 4, p.groupSize == 32, p.weight.dtype == .uint32,
                p.scales.dtype == .uint8,
                p.weight.shape == [256, output, input / 8],
                p.scales.shape == [256, output, input / 32]
            else { return false }
            return true
        }
        let gateUpMatches: Bool
        if let gateUpProj {
            gateUpMatches = matches(gateUpProj, input: 4096, output: 4096)
        } else {
            gateUpMatches =
                matches(gateProj, input: 4096, output: 2048)
                && matches(upProj, input: 4096, output: 2048)
        }
        return gateUpMatches && matches(downProj, input: 2048, output: 4096)
    }
}
