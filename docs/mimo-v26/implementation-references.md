# MiMo implementation references and adaptation boundaries

Reference commits identify source consulted or adapted, not qualification of
this Swift implementation. Preserve existing file notices and the complete
[Apache-2.0](LICENSE-APACHE-2.0), [oMLX](LICENSE-OMLX) and
[Apple/SDK MIT](../../LICENSE) permission texts. The checkpoint weight license
does not replace code licenses.

## Model and processor provenance

The architecture/processor pins and selected checkpoint are maintained in
[provenance.md](provenance.md). Native `mimo_v2` is not a Qwen/GLM architecture
port. Trained MiMo heads borrow target post-final-norm features independently;
head output is not fed as the next head's target feature.

## oMLX / MLX adaptations

| Reference | Exact head | Swift scope / deliberate boundary |
|---|---|---|
| [oMLX3970](https://github.com/jundot/omlx/pull/3970) | `fba858555b7a8c367d9a472f5df8d268097eef15` | Companion provider `MiMoV26WiredResidency`: shared-manager ticket tied to actual retirement; not the reference's direct global setter policy |
| [oMLX3978](https://github.com/jundot/omlx/pull/3978) | `891fdd5aedb21ee749591bca9727093030d0aa49` | Parallel-head and park/resume comparison; Swift already separates committed observations, draft rounds and discarded suffixes. No Python clone flag is treated as native ownership proof |
| [oMLX3990](https://github.com/jundot/omlx/pull/3990) | `e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb` | Guarded decode residual/norm, router and distinct-expert adaptations; original dtype boundaries and ordinary fallback retained |
| [oMLX3993](https://github.com/jundot/omlx/pull/3993) | `a68cab68ae40eeab7062ba7c1eff4ddffa06e373` | Joint projection motivation; Swift reads split gate/up tensors with independent accumulators, without a full resident concatenation or loader rewrite |
| [oMLX3994](https://github.com/jundot/omlx/pull/3994) | `1c487861d1c1d4c82a2b2920dd69c064eec85e18` | 192/128 attention adaptation; rejects literal one-pass softmax arithmetic and preserves the native three-pass rounding contract |
| [oMLX3995](https://github.com/jundot/omlx/pull/3995) | `47876fbc310fbb311cd382c4eba8572ec4368308` | Segmented native MXFP4 gather; MiMo-only sorted/shape gates and Int32 row offsets; affine/general ragged routes excluded |
| [oMLX4022](https://github.com/jundot/omlx/pull/4022) / [4029](https://github.com/jundot/omlx/pull/4029) | `d6dbc9970c6df99b6a8589746e6ff8ddc38c33ff` / `498685bae7b536cbd0d7d5e145b0327544e2fb06` | Separately gated SwiGLU epilogue and sorted row-map loads in `MiMoV26NAXGateUp`; preserve rounded gate/up, sigmoid and products |
| [oMLX4032](https://github.com/jundot/omlx/pull/4032) | `12e5084f8cf888bb6e846147929419f044802343` | Ordered ranges in `MiMoV26NAXAttentionKeyRanges`; preserve EACH native score pass and its state order; not a one-pass FP32 softmax substitution |
| [oMLX4033](https://github.com/jundot/omlx/pull/4033) | `c10e4ad57b180fa1aa88bf40532cba4362b6851a` | `MiMoV26DecodeRows` preserves pinned scalar-row block policies; `MiMoV26SplitKeyAttention` is a separate unintegrated matrix-QK candidate with its own state/admission and numerical gates |

Primitive comparison uses Apple MLX
`3fa8f25e6451174d7b06be372c3a24272b77d88e`, including NAX fragment operations,
MXFP4/E8M0 decoding, reduction and typed compiled activation emission. Embedded
primitive text retains Apple copyright and MIT notices in
`Libraries/MLXLMCommon/Models/MiMo/MiMoV26NAXMetalSources.swift`.

Blocked-attention intent also references
[oMLX3972](https://github.com/jundot/omlx/pull/3972), head
`65d0da092e24200f8a2f09ae72d45e582c6fa18e`. Existing native full/sliding
attention already blocks prefill queries; the grouped-block candidate reuses
each original block's exact key-prefix bounds and the unchanged native-rounded
score body under a separately admitted owner.

The larger scheduler defaults from [oMLX3973](https://github.com/jundot/omlx/pull/3973)
at `9f23edc088c07c91ba70583ca46dcf8a70131b45` and
[oMLX3992](https://github.com/jundot/omlx/pull/3992) at
`3f2b3ca8cbe3707595df54b99fcb0f893761b7a2` are not default promotions here.
A block or key-range port is not permission to hoist full-context keys, change
chunk sizes, reduce reserves or import an unrelated model family's math.

## Interpretation

All listed performance experiments remain default off. Their guards, effective
dispatch and full qualification are separate from reference speed claims.
Key-range and split-key helpers must not be called from serving until their
state allocations and actual native lifetimes are bound. A source file or
environment flag does not create that ownership.

The reference 3978 description itself reports non-identical MTP/OFF greedy output,
including depth 1. That limitation does not explain a particular Swift mismatch
or relax this implementation's exact greedy/state gate. No oMLX parity or
full-model speed claim follows from these reference pins.
