# Flint provenance and licenses

The native Swift Flint implementation is adapted from the following work:

| Source | Author / license | Adaptation |
|---|---|---|
| [mlx-vlm Flint](https://github.com/Blaizzy/mlx-vlm/tree/1a6ebd06459be3b18cb1c7922305c9e362612c62/mlx_vlm/models/flint), [PR #2565](https://github.com/Blaizzy/mlx-vlm/pull/2565) | Prince Canuma, copyright 2025; MIT | Decoder structure, QK normalization, learned attention sinks, sigmoid top-k routing, FP32 expert reduction/output head, expert checkpoint mapping and hybrid cache semantics. Ported to Swift and Darkbloom's CBv2 caches. |
| [Recursive rec](https://github.com/recursive-oss/rec/tree/e53663b262af4cb1ca498e596ca8b1a2d79e0800/engine/src/rec_mlx) | Recursive contributors; Apache-2.0 | Official affine INT4 byte packing, INT8 embedding exception, optional router-bias mapping and checkpoint validation. No rec scheduler or device-specific custom kernel is included. |
| [Recursive Flint release](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288) and [INT4 package](https://huggingface.co/RecursiveSI/flint-int4/tree/3b93f9053b530488405ae9fca51d4900d6ce21f0) | Recursive; Apache-2.0 | Architecture configuration and tokenizer/tool protocol. Weights are downloaded separately and are not included in this repository. |

Full license texts are retained in [mlx-vlm-MIT.txt](mlx-vlm-MIT.txt) and
[recursive-rec-Apache-2.0.txt](recursive-rec-Apache-2.0.txt). Adapted source files
identify their origin. Existing MLX Swift / MLX Swift LM notices remain applicable.

The mlx-vlm implementation was validated on a preview checkpoint. Its numerical,
quality and performance results are not acceptance evidence for this release or
the Swift port. rec's single-session engine is likewise not evidence for native
multi-request serving. The Swift port retains the FP32 expert reduction and
output-head behavior of the BF16 reference; rec's reduced-precision alternatives
are not silently substituted.
