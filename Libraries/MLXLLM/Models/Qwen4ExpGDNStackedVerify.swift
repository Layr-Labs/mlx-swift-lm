// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Lightning MTP capture-verify runs the Qwen4 GDN recurrence over the
// `1+k` verify window and must keep the fp32 state after every position so
// finalize can commit the accepted one. The shipped path chained one
// `gated_delta_step` launch per position (slices, gates and a kernel each:
// ~17 graph nodes × width, plus the stack concats) across 36 GDN layers —
// about 2,900 extra nodes per round, and the round is host-encode bound.
// `gated_delta_step_stacked` is the same kernel body with the per-position
// state written out from the same fp32 registers, so one launch replaces
// the chain bit for bit. Qwen4-only (`qwen4L2`).
// Kill: DARKBLOOM_QWEN4_GDN_STACKED_VERIFY=0.

import Foundation
import MLX
import MLXLMCommon

enum Qwen4ExpGDNStackedVerify: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_GDN_STACKED_VERIFY"

    static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    // MARK: - Invocation counters (tests / diagnostics)

    private static let lock = NSLock()
    nonisolated(unsafe) private static var stackedCalls = 0
    nonisolated(unsafe) private static var chainedCalls = 0

    struct Snapshot: Equatable {
        let stacked: Int
        let chained: Int
    }

    static func recordStacked() {
        lock.withLock { stackedCalls += 1 }
    }

    static func recordChained() {
        lock.withLock { chainedCalls += 1 }
    }

    static func snapshot() -> Snapshot {
        lock.withLock { Snapshot(stacked: stackedCalls, chained: chainedCalls) }
    }

    static func resetForTesting() {
        lock.withLock {
            stackedCalls = 0
            chainedCalls = 0
        }
    }
}
