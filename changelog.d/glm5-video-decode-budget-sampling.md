### GLM5 Next: a clip past the decoded-frame cap can be sampled down across it

- `GLM5NextProcessorConfiguration.fitsVideoSampleToDecodedBytes` (off by default):
  a clip whose official 2 FPS sample has more distinct frames than
  `maximumDecodedBytes` holds at the clip's frame size (`maximumSampledVideoFrames`:
  the RGB8 frames that fit, the last one as its RGBA render; 193 of 720p and 85
  of 1080p under the default 512 MiB) keeps that many, spread over the whole clip
  as the sampler already spreads `maximumFrames` (`sampledVideoIndices`), instead
  of being refused (720p past about 97 s, 1080p past about 43 s). Any other clip
  keeps the official sample, so nothing the processor accepted changes. A decoded
  asset learns the frame size from its first frame, decoded alone.
