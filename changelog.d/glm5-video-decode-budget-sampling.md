### GLM5 Next: a clip past the decoded-frame cap can be sampled down across it

- `GLM5NextProcessorConfiguration.fitsVideoSampleToDecodedBytes` (off by default)
  keeps at most the whole frame pairs `maximumDecodedBytes` holds at the clip's
  frame size (`maximumSampledVideoFrames`, at most 2048), spread over the whole
  clip as the sampler already spreads `maximumFrames`, instead of refusing a
  clip whose official 2 FPS sample passes the cap (with the default 512 MiB,
  720p past about 97 s, 1080p past about 43 s). A decoded asset learns the frame
  size from its first frame, decoded alone. A clip within the cap samples as
  before.
