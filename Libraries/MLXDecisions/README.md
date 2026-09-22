# Native Laya typed decisions

`MLXDecisions` runs Laya's ModernBERT encoder and decision heads directly in
MLX Swift. It accepts System One `choice`, `score`, and `noul` questions and
returns calibrated probabilities without autoregressive generation or KV caches.

The initial qualified checkpoint is the English 421M FP16
[`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) at revision
`20aed815fc6acde75733882e7ec0e3f28aeb9717`. Its 842,609,225-byte weight file has
SHA-256 `b9c07bf14be2fa5c78a9193a3e6d840ac80e89e62fc40f425834c3d8a6eaa3de`.
The original upstream checkpoint is `convaiinnovations/laya` at
`c5d78730f3493e4fe16d61507ef4b78eef7318cf`. The port and weights carry Apache-2.0
notices; see [NOTICE](NOTICE) and [LICENSE-APACHE](LICENSE-APACHE).

## Download and run

```sh
hf download aac6fef/laya-mlx \
  --revision 20aed815fc6acde75733882e7ec0e3f28aeb9717 \
  --local-dir /path/to/laya-mlx
hf cache verify aac6fef/laya-mlx \
  --revision 20aed815fc6acde75733882e7ec0e3f28aeb9717 \
  --local-dir /path/to/laya-mlx --fail-on-missing-files
```

Keep the checkpoint's original layout: `model.safetensors`,
`rl_agent_config.json`, `encoder/config.json`, and `tokenizer/`. Do not flatten
the tokenizer folder or load this checkpoint through an autoregressive factory.

```swift
import Foundation
import MLXDecisions

let runtime = try await LayaRuntime.load(directory: URL(fileURLWithPath: "/path/to/laya-mlx"))
let response = try await runtime.predict(data: Data(#"""
{"model":"laya","state":"I was billed twice. Please refund the duplicate.",
 "questions":{"refund":{"type":"noul","instructions":"Does the customer request a refund?"}}}
"""#.utf8))
```

The `mlx-server` executable has a dedicated local mode:

```sh
swift run mlx-server --model-type laya --model /path/to/laya-mlx --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

As with other MLX Swift executables, install the source-matched
`mlx.metallib` beside the binary before execution. Within Darkbloom, use the
provider package's build and `scripts/fetch-metallib.sh` workflow.

## Contract and limits

- Requests contain `model`, `state` (string, object, or array), and 1–64 named
  `questions`. Streaming is unsupported. Question IDs are response keys and
  are not sent to the encoder.
- Choice criteria are ordered JSON objects with 1–255 labels, subject to the
  checkpoint's actual header token budget. Oversized option headers fail
  instead of dropping options. Scores have 2–10 ordered levels. Noul returns
  the calibrated probability of the `true` option.
- The English checkpoint admits at most 512 tokens **per question**, including
  instructions, options, separators, and state. State tokens beyond the
  remaining space are truncated from the right, matching the reference.
- Every question has its own bidirectional encoding. Batches contain at most
  16 questions; there is no reusable state-only embedding across questions.
- `usage.input_tokens` sums the encoded lengths of all question rows, excluding
  batch padding. `usage.output_tokens` is always zero. Response serialization
  is not counted as generated tokens.
- The runtime preserves object and option order, strips literal `[MASK]`
  occurrences from user text, uses checkpoint temperatures, and retains Laya's
  `action.act_probability` extension. Confidence and probabilities are rounded
  to four decimals; rounded distributions can differ slightly from a sum of 1.
- A runtime actor serializes evaluation. The local HTTP server admits one
  request and rejects excess concurrent requests with 429 and `Retry-After`.
  Request bodies are limited to 1 MiB. The default bind address is loopback.
- Cancellation is checked between question batches. A submitted Metal batch
  must finish before the runtime releases its model. `shutdown()` drops model
  ownership explicitly.

This is an independent port of Laya, not Jev weights. The API follows the
[TypeSafe System One request shape](https://docs.typesafe.ai/api); Laya has its
own context limits, probabilities, quality, and usage accounting. Other Laya
checkpoints are not included in the initial qualification.

## Qualification

`laya-probe MODEL_DIRECTORY request.json --diagnostics` emits local CPU copies
of token IDs, option markers, logits, and action logits. Diagnostics are absent
from HTTP responses. `--repeat 30` reports warm end-to-end runtime timings to
stderr after five warmups.

Run `scripts/qualify-laya.py` in a Python environment containing the independent
[`laya-mlx` reference](https://github.com/mizorewww/laya-mlx), pinned at
`fc1df62828a3fedf4d8229fdac1cbd85f1cdf337`:

```sh
python scripts/qualify-laya.py --probe /path/to/laya-probe \
  --checkpoint /path/to/laya-mlx --output parity.json
```

The comparison covers mixed question types, padding and one-option choices,
structured Unicode and numeric input, the context boundary, option-count calibration, and
an 18-question request spanning batches. It requires exact tokens, markers,
labels, model identity, and usage. Numeric thresholds are recorded in its JSON
report. These checks establish implementation parity; they do not establish
task accuracy, fleet latency, or deployment readiness.
