# Gemma 4 dispatch overhead: initial endpoint screen

The combined dense GeGLU reuse and scoped command-buffer limits improved the
8K-context, single-request HTTP decode proxy by 3.45% and 3.90% in the two
orderings of an A/B/B/A screen. This is an initial result on one M5 Max, not a
200–300 token/s result or evidence of a prefill or multi-device gain.

## Change

`Gemma4MLP` reuses the existing compiled GeGLU product for matching GPU
FP16/BF16/FP32 inputs. Mixed dtypes, CPU execution and disabled compiled decode
retain the prior expression. The shared implementation retains the intermediate
dtype before the final multiplication.

The production Gemma 4 MoE geometry with global affine4/group64 quantization
supplies the existing CBv2 command-buffer scope with 100 operations and input size 512
thresholds. The second field's Swift API name is `megabytes`, but the current
backend counts millions of distinct input elements. These are commit heuristics,
not memory caps. The engine's existing width, admission and pressure checks remain
in effect. Kernel arithmetic, data dependencies and synchronization correctness are
preserved; command-buffer submission boundaries change.

The measured artifact contains both changes. This screen does not attribute the
whole gain to either one individually. Larger compiled FFN regions, staged NAX
gathers and compact kernel metadata are separate experiments and are absent here.

## Endpoint measurements

| Ordering | Baseline decode proxy | Candidate decode proxy | Median paired ratio |
|---|---:|---:|---:|
| A1 → B2 | 99.589 tok/s | 103.029 tok/s | 1.0345× |
| A4 → B3 | 99.209 tok/s | 103.077 tok/s | 1.0390× |

Each leg contains two identical-across-arms synthetic requests, 8,204 reported
prompt tokens, 8,192 cached tokens and 256 completion tokens. Temperature 0 and
seed 1729 are fixed. The normal adaptive MTP controller uses exact acceptance in
both arms. All canonical content/reasoning/tool output and completion usage
matched. Baseline repeat drift was −0.38%; candidate repeat drift was +0.048%.
Warm TTFT moved in both directions, so no prefill or TTFT gain is established.

Every artifact prewarms its own exact corpus under the unchanged cache identity
policy. Each measured cohort follows 30 seconds of observed nominal thermal
state and idle service. AC power remains Automatic. No competing build, test or
known inference process was allowed during the experiment. These checks cannot
prove continuously identical thermals or detect every arbitrary GPU application.

The decode proxy is `(completion_tokens - 1) / (last_generated_frame_time -
first_generated_frame_time)`. SSE frames are not tokens; this is a client HTTP
measurement, not engine-reported decode time. Request-to-first-generated-frame
latency and complete-request throughput are also retained in
[receipt.json](receipt.json).

Measured repeat RSS was approximately 16.269 GiB for the candidate and 16.266 GiB
for the baseline. The first candidate leg's lifetime physical-footprint peak
was 20.06 GiB including a cold prewarm, versus 17.54 GiB in its warmed B3 leg;
this is not a matched warm memory-regression estimate. MLX allocator peak was not exposed by this endpoint.

## Correctness and remaining qualification

The combined focused suite passed 14 tests with zero failures or skips. It covers
exhaustive finite 16-bit GeGLU inputs, special values, actual dense widths,
strides, CPU/mixed-type and compile-off fallbacks, gradients and nested compiled
calls, configuration eligibility, scope restoration, thread isolation, and
byte-identical asynchronous graphs under different command-buffer partitions.
The GeGLU suite separately passed with compiled decode disabled.
The same signed candidate also passed all 12 endpoint capability smokes with
exact baseline output and completion-usage parity, covering reasoning, tools,
structured JSON, images, videos and text requests interleaved with media.

The thread-isolation test emitted a QoS wait warning. It did not fail; this
fixture warning is preserved in the local test receipt. Tests ran with
Swift 6.4/Xcode 27, while repository CI selects Xcode 26. Changed files are unchanged
by the locally available formatter; CI toolchain validation remains required.
The SDK DocC check has not yet run; Xcode 26 CI validation remains pending.

The dedicated post-test refactor review found no further change warranted.
The source changes preserve model parameter names, quantization, loaders,
adapters, attention, prefix caching and SSD policy. The shared text tower makes
the same implementation available to VLM callers and downstream clustering
users when they adopt the SDK revision; clustering and non-M5 hardware have not
been measured here. Broader concurrency, sustained load and longer histories
remain required before generalizing this small screen.
