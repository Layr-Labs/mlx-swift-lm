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
- The timestamped layout reads the selected frames at their presentation times
  (the frame count from the track's shown span, its first to last non-empty edit,
  and nominal rate; the rate then from that count), so a clip trimmed without
  re-encoding, a late start or a gap (an empty edit) and a long clip need no full
  decode. A time inside an empty edit moves to the nearer shown frame; a frame the
  generator drops is read once more. A variable-rate clip is sampled evenly in
  time, the reference evenly in frame number. Qwen4 keeps its frame-ordinal reader.
- A template the tokenizer does not decode and re-encode unchanged expands in
  token space instead of failing. Video geometry and track failures of
  `Qwen3VLProcessor` are `VLMError.imageProcessingFailure` / `videoNotDecodable`,
  as in the single-block layout.
