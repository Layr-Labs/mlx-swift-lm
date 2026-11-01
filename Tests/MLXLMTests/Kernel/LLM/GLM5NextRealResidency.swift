// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Cmlx
import Foundation

/// Standing residency for real-artifact runs, as the provider's
/// `GLM5NextWiredResidency` holds for a published engine. Without it every
/// command buffer re-makes the ~165 GiB payload resident and prefill runs at
/// a quarter speed. `GLM53_WIRED_LIMIT_GB` (GiB, > 0) opts in after load.
enum GLM5NextRealResidency {
    static func applyIfRequested(_ environment: [String: String]) {
        guard let gb = environment["GLM53_WIRED_LIMIT_GB"].flatMap(Int.init), gb > 0 else { return }
        var previous = 0
        let status = mlx_set_wired_limit(&previous, gb << 30)
        print("GLM53_RESIDENCY wired_limit_gb=\(gb) status=\(status) previous=\(previous)")
    }
}
