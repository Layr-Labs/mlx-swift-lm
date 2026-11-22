// Copyright © 2026 Eigen Labs.

import MLXLMCommon

/// Exact artifact identity of the served weights, not a catalog entry or an
/// advertised capability: nvidia/Muse-Glimmer-30B-NVFP4 at the revision below.
/// The shard values are the Hugging Face LFS object identifiers; the index
/// value is the SHA-256 of `model.safetensors.index.json` at that revision.
public enum MuseGlimmerCheckpointPin {
    public static let repository = "nvidia/Muse-Glimmer-30B-NVFP4"
    public static let revision = "47818374517751c48c55cde2621594926b1888b6"
    public static let index = "71823cd92701dfd1632fe37bb7dfcf432a0d068cee2cc26dcc1b94189c3c3b04"
    public static let shards: [String: String] = [
        "model-00001-of-00003.safetensors":
            "0e260a8f2c5e16bcc2ad8e90a8a59f1e889431b0de82f4c490c31d9ea9114ce4",
        "model-00002-of-00003.safetensors":
            "6a04e8f3e9e68b9d7e0ed8a5d762694eb73e76d2a13ad4951d909238059a6cc5",
        "model-00003-of-00003.safetensors":
            "017293c10a76b131f36116c597f6a4eb9a4dd52d8773318b8df4b6afcc625132",
    ]
}

extension ModelOptActivationOverflow {
    /// Muse Glimmer's FP8 (W8A8) projections saturate out-of-range activations,
    /// as the deployment runtimes the checkpoint was evaluated on do. Part of
    /// the numerics fingerprint; a reversible decision.
    public static let museGlimmer = Self.saturating
}
