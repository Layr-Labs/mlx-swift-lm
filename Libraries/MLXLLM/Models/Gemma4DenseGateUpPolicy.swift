// Copyright © 2026 Eigen Labs.
import Foundation

/// The local catch-up candidate is opt-in until native equality and serving
/// measurements pass. This is David Tai's B1 dense gate/up mechanism, not
/// the challenge's separately specialized B8 dense-MLP kernel.
enum Gemma4DenseGateUpPolicy {
    static let environmentKey = "DARKBLOOM_GEMMA4_DENSE_GATEUP_CONCAT"
    private static let parameterNames = [
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight", "up_proj.scales", "up_proj.biases",
    ]

    static func requested(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        environment[environmentKey] == "1"
    }

    static func admits(
        enabled: Bool, ndim: Int, batch: Int, positions: Int, hidden: Int,
        bits: Int, groupSize: Int
    ) -> Bool {
        enabled && ndim == 3 && batch == 1 && positions == 1 && hidden == 2816
            && bits == 8 && groupSize == 64
    }

    /// Accept one complete language namespace only. A partial or competing
    /// namespace must not silently bind storage from a different module.
    static func parameterKeys(layer: Int, contains: (String) -> Bool) -> [String]? {
        guard layer >= 0 else { return nil }
        let prefixes = ["model.layers.\(layer).mlp", "language_model.model.layers.\(layer).mlp"]
        let groups = prefixes.map { prefix in parameterNames.map { "\(prefix).\($0)" } }
        let counts = groups.map { $0.filter(contains).count }
        if counts == [6, 0] { return groups[0] }
        if counts == [0, 6] { return groups[1] }
        return nil
    }
}
