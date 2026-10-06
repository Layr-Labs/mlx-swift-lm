# OpenRouter media working-set check

2026-09-30. Component execution on an Apple M4 Max with 128 GiB RAM.

## Root cause and implementation

The former managed media quote charged a conservative full-frame attention
graph for every vision layer and every temporal grid simultaneously. The
1280×851 JPEG therefore needed about 344 GiB of additional reservation, even
though the actual vision computation could run in far less memory. The host
correctly refused that reservation; lowering the global memory cap or ignoring
the refusal would not fix its accounting.

`MiMoV26VisionTower.forwardBounded` now completes each frame and transformer
block before constructing its successor. `MiMoV26MultimodalProcessor` tracks
the current roots and uses its existing required-native-completion boundary
for every checkpoint. It retains completed features and decoded inputs.
`MiMoV26VisionWorkingSet.frameBytes` charges the largest frame/block graph;
the outer managed quote separately retains pixel/patch/feature allowances.
The published vision shape has 64-wide Q/K/V heads (`qk_channels` defaults
independently of hidden width / head count). On the default Metal stream it
uses the pinned fused SDPA path. The quote counts the live projection, FP32
rotary, MLP, mask and allocator buffers rather than a full per-head score
matrix. CPU/custom streams, nonqualified head geometry and wider local
windows retain the conservative full-score path. This is request scratch,
not a change to the global activation floor.

The native preparation registry also retires completed scratch-array entries
after each successful checked checkpoint. Otherwise its strong references
would retain old outputs even though the next layer had already completed.
The preparation owner and loan survive until real request retirement; a failed
checkpoint retains its roots. Mandatory tests cover bounded registration counts
through real managed preparation and retained roots after injected failures.

## Exact inputs and computed reservations

The encoded files and their hashes are in
`Tests/MLXLMTests/Resources/MiMoOpenRouter/manifest.json`. Both image tests were
sent as inline JPEG data URLs. The remote-video test used the URL recorded in
that manifest. These are test fixtures, with no HTTP authorization headers.

The following compares only the vision graph term. These are calculated
reservations, not measured memory, and exclude separately retained media,
language-model weights, target KV and host reserves.

| OpenRouter test | Encoded dimensions | Temporal grids | Former graph GiB | Bounded graph GiB |
|---|---:|---:|---:|---:|
| input-image-url | 1280×851 | 1 | 343.619 | 0.899 |
| input-image-base64 | 640×461 | 1 | 37.906 | 0.237 |
| input-video-url | 1280×720, 5.056 s | 4 | 958.610 | 0.734 |
| input-video-base64 | 1490×534, 10.833 s | 5 | 1013.802 | 0.666 |

The MP4 also contains stereo 32 kHz AAC. Encoded ingress preserves those PCM
properties for the native resampler; its target is 24 kHz. The mandatory
fixture test exercises the actual AAC decode and includes its native audio
reservation in a combined vision/audio bound. The silent QuickTime test has
no audio track.

## Actual vision-weight execution

The 364 BF16 vision tensors came from
`EigenLabs/MiMo-V2.6-Flash-MOPD-MLX-4bit-mtp` at Hugging Face revision
`bb37ffc73180a94d2f903ae2c794665135e38d83`, shard
`model-00036-of-00037.safetensors`. The extraction preserves tensor bytes and
renames `vision_tower.` to `visual.`. Its packed vision-only file has SHA-256
`26e8da61a429535c564d91c2a0a5c67e172cc5faae5c577721a40512a83342a2`.

`MiMoV26FullVisionQualificationTests.testCapturedMediaWithActualVisionWeights`
ran all four encoded inputs and five broader cases: 1080p, 4K, maximum-area
portrait/panoramic images, and a 32-frame video. Every result was finite with
the expected shape. Each case resets the MLX peak after materializing weights;
its incremental peak must fit the working quote plus retained patches/features.

| Case | Working quote GiB | Additional measured MLX peak GiB |
|---|---:|---:|
| Captured large JPEG | 0.899 | 0.404 |
| Captured small JPEG | 0.237 | 0.111 |
| Captured MP4 vision | 0.734 | 0.420 |
| Captured MOV | 0.666 | 0.413 |
| 1920×1080 | 1.694 | 0.763 |
| 3840×2160 | 6.760 | 2.584 |
| 2048×4096 / 8192×1024 | 6.787 | 2.594 |
| 32 frames at 640×480 | 0.254 | 0.256 |

Measured increments include device patches and accumulated output features;
those are separately charged on top of the working quote. The 32-frame
increment therefore exceeds its frame-only quote while remaining below the
combined reservation. These are allocator measurements, not total process RSS.
The language model, target KV and audio codec were absent from this vision run.

Run that explicit check with a locally prepared, verified vision-only file:

```sh
MIMO_V26_VISION_MEMORY_MATRIX=1 \
MIMO_V26_FULL_VISION_WEIGHTS=/path/to/vision.safetensors \
  scripts/run-nested-suite.sh MiMoV26FullVisionQualificationTests --no-parallel
```

## Audio codec and reservation check

The same MP4 previously requested 9,355,467,872 bytes (8.713 GiB) of native
audio work. The owned path now completes every encoder block and RVQ
codebook step, preserving the original padded groups, causal masks, BF16
rounding and code order. It charges actual tile rows and retains mels, encoded
features and the separate final audio-patch allowance. The captured clip now
requests 346,436,592 bytes (0.323 GiB).

`MiMoV26FullAudioQualificationTests` used the authenticated selected codec
(`077033345d80eef3a315e8d394e0589667e80e4cdaba9bc5a7488410c6657265`),
including its 389 input tensors / 634,204,160 materialized bytes. It passed the
captured stereo AAC and 1, 30, 61 and 300 second PCM clips. Additional measured
MLX peaks were 22.35, 9.72, 94.51, 270.14 and 1102.14 MiB respectively.
Each stayed below its reservation. This run measures PCM-to-codec output; it
does **not** load the language model or final audio-patch weights. Its reservation
includes a conservative final audio-patch allowance not exercised by that
component measurement. Tiny native managed/provider tests separately cover
composition, retirement and fault retention.

```sh
MIMO_V26_AUDIO_MEMORY_MATRIX=1 \
MIMO_V26_AUDIO_SIDECAR_FIXTURE_ROOT=/path/to/selected-main-and-audio-metadata \
  scripts/run-nested-suite.sh MiMoV26FullAudioQualificationTests --no-parallel
```

## Regression boundaries

- `MiMoV26OpenRouterMediaTests`: exact JPEG/MP4/MOV decode, actual production
  geometry, bounded reservations, and tiny native lazy/bounded equivalence.
- `MiMoV26VisionWorkingSetTests`: actual 64-wide Metal attention allocation,
  CPU/custom-stream/geometry fallback and linear reservation scaling.
- `MiMoV26AudioWorkingSetTests`: actual-tile accounting and exact lazy/bounded
  encoder and quantizer equivalence for mixed clips.
- `MiMoV26NativeMediaDeadlineTests.testMediaReservationRefusalLeavesTextEngineUsable`:
  repeated reservation refusals followed by unchanged real text generation on
  the same tiny engine, with no stranded native loan.
- The companion provider/coordinator change carries a distinct media-memory
  reason, retains bounded retry, and avoids invalidating text/KV capacity.
- Required native-completion failures still retain their owners and quarantine
  the affected engine. This change does not permit use after an unsafe drain.

This evidence does not qualify a full-size signed provider, its language-model
answers, or production fleet rollout. Those remain separate release checks.
