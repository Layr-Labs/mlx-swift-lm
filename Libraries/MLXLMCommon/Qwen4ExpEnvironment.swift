// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Process-wide snapshot of the environment for Qwen4 runtime flags.
//
// `ProcessInfo.processInfo.environment` rebuilds a ~60-entry dictionary on
// every access (~43 µs on the M3 Ultra). The Qwen4 hot path used it as a
// default argument on `Qwen4ExpActivation.keep`, `Qwen4ExpAffineQMM.apply`
// and the QSA / GDN / PLE kill-switch checks, which are called several
// hundred times per decode token — measured at ~40 ms of a 70 ms lazy graph
// build (`DARKBLOOM_QWEN4_DECODE_PROFILE=build`). All Qwen4 flags are fixed
// at launch, so one snapshot is exact. Tests keep passing `environment:`
// explicitly; `refresh()` exists for a caller that mutates the environment
// after first use.

import Foundation

public enum Qwen4ExpEnvironment: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cached: [String: String]?

    /// The environment as of first access (or the last `refresh()`).
    public static var snapshot: [String: String] {
        lock.lock()
        defer { lock.unlock() }
        if let cached { return cached }
        let fresh = ProcessInfo.processInfo.environment
        cached = fresh
        return fresh
    }

    /// Re-read the process environment on the next `snapshot` access.
    public static func refresh() {
        lock.lock()
        cached = nil
        lock.unlock()
    }
}
