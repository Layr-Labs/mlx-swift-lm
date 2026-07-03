// Load-time exclusion hook for models that stream some weights from disk.

import Foundation

/// Models that stream a subset of their weights from disk at forward time
/// (see `StreamingQuantizedSwitchGLU`) conform to this so `loadWeights` can
/// exclude those keys BEFORE shard materialization.
///
/// Why sanitize-time dropping is not enough: this fork's parallel shard
/// loader `eval`s every shard's arrays as it reads them (to overlap disk
/// I/O across shards) — by the time `sanitize(weights:)` runs, every tensor
/// has already been materialized in memory. For DeepSeek-V4-Flash that's
/// ~125 GB of routed-expert weights transiently resident on a 128 GB box,
/// which defeats streaming entirely. Filtering the key dict between the
/// safetensors parse and the `eval` keeps those arrays as unevaluated lazy
/// loads that are simply freed, so their bytes are never read.
public protocol StreamedWeightsModel {
    /// Return true for checkpoint keys (pre-`sanitize` naming) whose tensors
    /// are served by a streaming path and must not be loaded resident.
    func shouldStreamWeight(key: String) -> Bool
}
