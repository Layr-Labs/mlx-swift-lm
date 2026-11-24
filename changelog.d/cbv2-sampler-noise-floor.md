### CBv2 sampler noise keeps a floor of 2⁻³³

- The Exp(1) noise of the CBv2 exponential race
  (`SamplerV2.exponentialNoise(fromUniform:)`) is now at least 2⁻³³, the value
  a zero draw already maps to. With an accurate `log1p` the noise never goes
  below that, so seeded sampled output does not change. The floor keeps the
  noise positive if a fast-math `log` rounds a small draw's `log1p` to zero.
  Tests: `CBv2ExponentialNoiseTests`.
