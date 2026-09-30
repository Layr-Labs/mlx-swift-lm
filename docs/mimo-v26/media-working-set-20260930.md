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
Full-attention score, FP32 and allocator allowances remain conservative.

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
| input-image-url | 1280×851 | 1 | 343.619 | 11.458 |
| input-image-base64 | 640×461 | 1 | 37.906 | 1.266 |
| input-video-url | 1280×720, 5.056 s | 4 | 958.610 | 7.995 |
| input-video-base64 | 1490×534, 10.833 s | 5 | 1013.802 | 6.765 |

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
ran all four encoded inputs through the full-size native vision tower. All
returned finite features with the expected shapes. Reported cumulative MLX
peak allocation, including the vision weights, was 1.673 GiB after the images
and 1.777 GiB after the videos. This is an allocator measurement; it is not
total process RSS, and the language model, target KV and audio codec weights
were not loaded in this component run.

Run that explicit check with a locally prepared, verified vision-only file:

```sh
MIMO_V26_FULL_VISION_WEIGHTS=/path/to/vision.safetensors \
  scripts/run-nested-suite.sh MiMoV26FullVisionQualificationTests --no-parallel
```

## Regression boundaries

- `MiMoV26OpenRouterMediaTests`: exact JPEG/MP4/MOV decode, actual production
  geometry, bounded reservations, and tiny native lazy/bounded equivalence.
- `MiMoV26NativeMediaDeadlineTests.testMediaReservationRefusalLeavesTextEngineUsable`:
  repeated reservation refusals followed by unchanged real text generation on
  the same tiny engine, with no stranded native loan.
- The companion provider/coordinator change carries a distinct media-memory
  reason, retains bounded retry, and avoids invalidating text/KV capacity.
- Required native-completion failures still retain their owners and quarantine
  the affected engine. This change does not permit use after an unsafe drain.

This evidence does not qualify a full-size signed provider, its language-model
answers, or production fleet rollout. Those remain separate release checks.
