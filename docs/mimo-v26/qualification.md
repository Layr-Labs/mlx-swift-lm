# MiMo V2.6 qualification contract

Source support, a component pass and a releasable serving profile are different
claims. Qualification binds exact source, binary/resources, model payload,
processor/codec, request policy and device; it does not transfer automatically
between artifacts or newly composed code.

## Entry-point and capability boundaries

`MiMoV26ModelFactory` performs strict source-bound loading under real
`NativeConstructionWork`. `MiMoV26LoadedModel.makeCBv2Binding` selects the native
target/cache/assistant owners. Generic TokenIterator entry is refused instead
of substituting older MiMo or Qwen behavior.

The unissued MiMo adapter keeps paging, compiled decode, packed prefill and generic
prefix-reuse capabilities disabled. A sealed native paged binding can advertise
paged KV support; it does not grant the other fast-path capabilities.
A separately issued text-only COMPLETE-prefix
contract is not an override of those flags. It must bind the exact observed
K/V types, codec/store, process owner and immutable loaded validator, and include
import/capture/publication/close work in real native retirement.

A separate native paged contract now binds the actual asymmetric page geometry,
pool, bank, process owner and step-work lifetime through
`MiMoV26LoadedModel.makeNativePagedExecutionResources`. This opt-in profile
supports target-only or explicit serial-target MTP, with authenticated text
complete-prefix composition. Rectangular verification and managed paged media
remain refused. The composed target-page/assistant restoration and late-cancel
store-retirement regressions are prepared, not full-artifact qualification.
Its paging capability requires the sealed binding, not a caller-supplied flag;
it does not establish provider/full-model paging qualification. See the
[composed component record](qualified-composition-20260928.md).

Decoded media requires the actual loaded processor and, for audio, the selected
authenticated codec owner. Preparation and bind both validate source/generation
and reservation identity. Media requests use target-only execution even when
the same engine has a text MTP assistant. Typed decoded audiovisual support is
distinct from decoding a video container's audio; speech output and combined
media-prefix reuse are not granted by these APIs. Encoded WAV and image/silent-video
ingress remain bounded, profile-gated paths.

`MiMoV26EncodedAudioDecoder` accepts classic RIFF/WAVE PCM8, PCM16, PCM24,
PCM32 and IEEE Float32, with one or two channels at 8–192 kHz. It checks chunk,
frame, encoded-byte and decoded-storage bounds before allocating samples,
deinterleaves into planar Float storage, and preserves the original sample rate.
`MiMoV26AudioFrontend` then resamples each channel to the selected codec's rate
and mixes to mono. Caller-supplied channel/rate ceilings can narrow this support.
Nonfinite Float samples, compressed WAV, WAVE_FORMAT_EXTENSIBLE and MP3 remain
unsupported. Container audio additionally accepts one AAC track with one or two channels
and bounded source rates. `MiMoV26EncodedAACAudio` uses AVAssetReader to decompress
at the original rate/channels, honors a single container trim (including encoder
priming), and rejects gaps, retiming, multiple tracks, nonfinite samples and
out-of-bound output. Native audio still performs resampling and mixing. Decoded
PCM provenance is hashed after decompression; AAC output is platform-decoder
output, not a bitwise cross-platform codec guarantee. Existing mono24k LPCM
transport remains exact and retains its stricter timeline contract.

Decoder regressions cover the OpenRouter-shaped 22.05 kHz unsigned PCM8 input,
stereo deinterleaving, integer extrema and malformed/bounded inputs. These are
transport/geometry checks; actual codec execution and authenticated provider
HTTP inference require the downstream provider qualification gate.

The audio tokenizer registers its indexed `encoder.down_sample_layer` as a
one-element module array, matching the checkpoint's `.0.weight` path. Normal
strict `ModuleParameters.unflattened` loading remains mandatory: matching file
hashes, tensor shapes and dtypes alone does not prove module installation. The
load regressions also require exact installed transpose values and rejection of
missing, extra, wrong-shape and wrong-dtype tensors; no weights are ignored.

`MiMoV26EncodedVisualDecoder.Limits.maximumEncodedBytes` rejects compressed
image/video payloads before ImageIO parsing or AVFoundation asset reads. Its
source-compatible default is the supplied working-byte limit; callers can
select a smaller encoded-byte ceiling. Reusing a video plan rechecks the
current ceiling. Sampling rejects an effective even minimum larger than the
effective even maximum, through both direct and configuration initializers;
this validation does not change valid native sampling or clamp resource limits.
Ready empty reader control markers have a separate bounded allowance and do
not consume the real-frame ceiling. Their priced metadata remains in the
immutable plan, and reused decoding cannot exceed that allowance. Re-run the
real three-frame fixture, exact frame-cap negative, malformed-marker refusals
and the complete encoded visual/audio suite against the corrected libraries.

Managed native preparation uses `MiMoV26Pixels.workingByteCount` for the
actual request geometry. The pixel ceiling remains a refusal limit, not an
allocation charged in full. Existing conservative decoded/patch/feature
allowances remain; temporal video scores are counted independently per frame,
matching the attention loop. The full lazy-graph depth and allocator rounding
allowances remain unchanged.

Visual decode working bytes distinguish retained output from sequential
scratch (`MiMoV26VisualDecodeMemory`). Images retain 12 bytes/pixel and allow
20 bytes/pixel + 1 MiB transient; videos retain sampled RGB, encoded ownership
and bounded metadata, plus one 32 bytes/pixel + 1 MiB transient. Unsampled
source frames add metadata only. Image and reader-iteration autorelease pools,
read-only no-copy reader output and direct BGRA-to-Float conversion keep the
application lifetime consistent with the quote. Actual row-stride storage is
validated. This does not bound AVFoundation's private codec pools: the host's
normal system/activation headroom and process ledger remain required. Re-run
`MiMoV26VisualDecodeMemoryTests`, encoded image/video tests, and authenticated
provider media inference, and measure full-artifact peak memory independently.

Joint managed-media/complete-prefix issuance permits cacheable text and bounded
noncacheable media in the same contiguous engine, with one genuine ownership
contract. It does not cache media or make media speculative. Its full-artifact
combined-profile tests remain separate from the selected component cohort.

The companion provider implements exact native MiMo ordinary dispatch and bounded
media/audio policies. Those source paths are not a qualification of public
Chat/Responses, coordinator audio routing or the complete supported media matrix.

The [fast-prefill default policy](FAST-PREFILL-POLICY.md) enables eligible native
attention and admitted grouping in source. Validate the ordinary provider's
actual loaded profile as well as the benchmark factory, resolve effective chunk
widths, and test startup rollback in a separate process. Source defaults do not
establish engagement, losslessness, memory safety or a speed improvement.

## Numerical acceptance

1. Preserve the selected checkpoint's packed codes/scales, activation and state
   dtypes, trained value scaling, sinks, RoPE, sigmoid routing and normalization.
   No precision reduction, approximate nonlinearity or changed reference is
   accepted as a summation-order change.
2. Compare each candidate with the actual selected native fallback, recording
   native storage bits and maxULP. A tolerance-only component test is not a
   lossless declaration.
3. Require exact full-model greedy output, natural stop behavior and complete
   retained target/MTP/cache state at accepted/rejected boundaries. Test widths,
   dtype/stride/ragged cases and MTP depth1–3 explicitly.
4. Serial target verification remains the authority strategy; rectangular
   verification changes whole-trunk shapes even when attention queries are
   serialized. An unresolved rectangular greedy difference blocks qualification.
5. Report failures, skips and not-reached cases separately. M3 hardware skips do
   not qualify M5-only NAX kernels; Metal source extraction does not qualify the
   Swift wrapper, managed dispatch or ownership.

The original literal oMLX one-pass attention arithmetic is not the native
rounding baseline. See [the attention port](NAX-ATTENTION-PORT.md).

## Ownership and cache acceptance

- Require actual selected-stream completion and joins, not a task cancellation,
  timeout, returned flag or empty counter. First required-fence failure remains
  sticky with actual arrays/owners/reservations retained.
- Price extra capture/import/kernel state before construction. Keep target
  rows, trained-head history, transient copies and host/store work owned through
  real retirement; logical accounting is not physical-page release.
- Require exact same-request/engine/store/codec/generation binding. Reject foreign,
  stale, duplicate and replayed seals/receipts before mutation or early success.
- Exercise true interior prompt boundaries, MTP OFF/ON continuation, atomic
  target+assistant adoption, memory/disk reuse, encrypted reopen, wrong artifact/
  runtime/tenant identity and tampered payloads. Target-only KV is not an MTP
  head/carry checkpoint.
- Test actual concurrent requests, cancellation, unload/reload and stop/late
  publication. Full-model API/media, persistent restart and lifecycle qualification
  remain separate from unit-level codec or array-copy tests.

## Tests and reproduction

Existing suites include `MiMoV26TextTests`, `MiMoV26CBv2Tests`,
`MiMoV26MTPEngineTests`, `MiMoV26MTPPrefixCheckpointTests`,
`MiMoV26RectangularVerifyNativeTests`, `CBv2NativeCompletePrefixOwnerTests`,
and the MiMo media/codec/performance tests under `Tests/MLXLMTests/MiMo`.
Use the repository's [contributor instructions](../../CONTRIBUTING.md) and exact
resource/toolchain closure. Native opt-ins and fault selectors require the
declared real fixture and an exclusively owned device lane; retained-fault cases
run in separate fresh processes.

Dated benchmark and pass/fail records belong in the companion release-review
evidence, not in architecture or attribution pages. Neither source review nor
a faster non-equivalent run checks the final end-to-end release gate.
