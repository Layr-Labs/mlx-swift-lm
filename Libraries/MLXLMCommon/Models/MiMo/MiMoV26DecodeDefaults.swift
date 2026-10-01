// Copyright © 2026 Eigen Labs.

import Foundation

/// Process switches for the MiMo short-forward kernels: fused decode norms,
/// distinct-expert MXFP4 decode and the FP32 router GEMV. Each one matches the
/// stock operation it replaces for its supported one-request, 1...7-row shapes
/// and returns to that operation for every other shape, dtype or device, so all
/// three are on by default. Exact `0` / `false` / `no` / `off` (case and
/// surrounding whitespace ignored) restores the stock path for one process.
public enum MiMoV26DecodeDefaults {
    public static let fusedNormsKey = "DARKBLOOM_MIMO_FUSED_DECODE_NORMS"
    public static let expertsKey = "DARKBLOOM_MIMO_DECODE_EXPERTS"
    public static let routerKey = "DARKBLOOM_MIMO_DECODE_ROUTER_GEMV"
    public static let environmentKeys = [fusedNormsKey, expertsKey, routerKey]
    public static let rollbackValues: Set<String> = ["0", "false", "no", "off"]

    public static func isEnabled(
        _ key: String, environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = environment[key] else { return true }
        return !rollbackValues.contains(
            raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }
}
