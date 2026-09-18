// Copyright © 2026 Eigen Labs.
import Foundation

/// Fuse selection/weights and rank while retaining the current softmax tape.
/// Historical monolithic softmax arithmetic is deliberately not transplanted.
enum Gemma4B8RouteFoldSources {
    static func make(native: Bool, prefix: Bool) -> String {
        var body = native ? Gemma4RouterFinalistsSources.nativeKeysWeights
            : Gemma4RouterFinalistsSources.bitonicWeights
        func replace(_ old: String, _ new: String) {
            precondition(body.components(separatedBy: old).count == 2)
            body = body.replacingOccurrences(of: old, with: new)
        }
        replace("const uint row = threadgroup_position_in_grid.x;",
                "const uint row = fold_tid / \(native ? 32 : 128)u;")
        if native {
            replace("if (lane >= 24u) indices[row * 8u + lane - 24u] = item & 127u;",
                    "if (lane >= 24u) { indices[row * 8u + lane - 24u] = item & 127u; route_selected[row * 8u + lane - 24u] = item & 127u; }")
        } else {
            replace("const uint group = simdgroup_index_in_threadgroup;",
                    "const uint group = simdgroup_index_in_threadgroup % 4u;")
            for (type, name, count) in [("uint", "finalists", "32"), ("float", "topv", "K"),
                                      ("uint", "topi", "K"), ("float", "local_max", "SIMD_SIZE"),
                                      ("float", "local_normalizer", "SIMD_SIZE")] {
                replace("threadgroup \(type) \(name)[\(count)];",
                        "threadgroup \(type) \(name)_rows[8 * \(count)]; threadgroup \(type)* \(name) = \(name)_rows + row * \(count);")
            }
            replace("indices[row * 8u + lane - 24u] = item & 127u;",
                    "indices[row * 8u + lane - 24u] = item & 127u; route_selected[row * 8u + lane - 24u] = item & 127u;")
        }
        let rank = (prefix ? Gemma4B8RouteSources.prefix : Gemma4B8RouteSources.raw)
            .replacingOccurrences(of: "thread_position_in_grid.x", with: "fold_tid")
            .replacingOccurrences(of: "indices[", with: "route_selected[")
        return "threadgroup uint route_selected[64];\nconst uint fold_tid = thread_position_in_threadgroup.x;\n"
            + body + "\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nif (fold_tid < 64u) {\n" + rank + "\n}\n"
    }
}
