### Qwen3-VL video in Transformers' timestamped layout

- `Qwen3VLProcessor` prepares a clip as Transformers' `Qwen3VLVideoProcessor` and
  `Qwen3VLProcessor` do for the Qwen3.5/3.6/3.8 checkpoints: frames sampled at
  2 fps (at least 4, exact frame indices), one resize per clip so frames ×
  height × width stays within the clip's pixel budget, and per frame pair a
  `<t.t seconds>` label and its own vision block. It gave 8 frames of at most
  512² as one block without timestamps. M-RoPE positions each block as a `t = 1`
  grid, as `get_rope_index` does; a clip in one block keeps its grid.
- `UserInput.VideoProcessing.maxFrames` and `maxPixels` cap a request's clips.
  Qwen3-VL's defaults stay 8 frames and about 1,024 visual tokens. The image
  `maxPixels` no longer narrows video.
- `Qwen3VLMediaProcessor.makeUserInputProcessor` keeps one block per clip, the
  layout its pair media contract carries.
- Video timestamps round like Python's `:.1f` (0.35 s is `<0.3 seconds>`); this
  also changes Qwen4's labels at such ties.
