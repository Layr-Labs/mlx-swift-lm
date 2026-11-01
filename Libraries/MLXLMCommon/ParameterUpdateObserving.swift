// Copyright © 2026 Eigen Labs.

import MLX
import MLXNN

/// A module that keeps inference state derived from its parameters, for example a
/// fused projection, and drops that state when an update changes the parameters.
///
/// mlx-swift 9bd73fc7 (#418) applies a recursive `update(parameters:)` through an
/// internal method. A child's `update(parameters:verify:path:modulePath:)` override
/// runs only when the update starts at that child. An ancestor that starts the update
/// calls ``MLXNN/Module/notifyParameterUpdateObservers(of:)`` so the child still sees it.
public protocol ParameterUpdateObserving: Module {
    /// Called after an update that started at an ancestor reached this module.
    ///
    /// - Parameter keys: the flattened updated keys, relative to this module.
    func ancestorDidUpdateParameters(keys: [String])
}

extension Module {
    /// Tells every ``ParameterUpdateObserving`` module below this one which of its
    /// parameters an update changed. Call it from an `update(parameters:)` override.
    public func notifyParameterUpdateObservers(of parameters: ModuleParameters) {
        let keys = parameters.flattened().map(\.0)
        guard !keys.isEmpty else { return }
        for (path, module) in namedModules() where !path.isEmpty {
            guard let observer = module as? any ParameterUpdateObserving else { continue }
            let prefix = path + "."
            let relative = keys.compactMap { key in
                key.hasPrefix(prefix) ? String(key.dropFirst(prefix.count)) : nil
            }
            if !relative.isEmpty {
                observer.ancestorDidUpdateParameters(keys: relative)
            }
        }
    }
}
