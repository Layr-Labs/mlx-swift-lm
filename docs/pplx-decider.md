# pplx-decider-v1-27b

The `pplx_decider_qwen35` factories load an MLX pack of
[perplexity-ai/pplx-decider-v1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b).
The model is a fine-tune of Qwen/Qwen3.8-27B. It does not generate text. One
forward pass of the prompt gives one probability for each option of a question.

The pack uses the dense Qwen3.8-27B layout (`qwen3_5`). Only the output head is
different: it has one row for each option, not one row for each vocabulary token.

## Artifact contract

- Tensor names, shapes, quantization and shard metadata are the same as
  `EigenLabs/Qwen3.8-27B-4bit`: mlx-vlm namespace (`language_model.model.*`,
  `vision_tower.*`), affine 4-bit with group size 64 for the 2-D language
  tensors, BF16 vision tower, `format: mlx` shard metadata.
- `language_model.lm_head.weight` is the trained decider readout, BF16
  `[255, 5120]`. It is not quantized. There is no vocabulary head and there are
  no `mtp.*` tensors.
- `config.json` has `model_type: "pplx_decider_qwen35"` and a top-level
  `decision` block. All other keys are the source config. `text_config.vocab_size`
  stays 248320 (the embedding rows).
- The `decision` block is `format_version` 1, `options` (1...255), `temperature`,
  `codes` (one unique `[A-Z]{1,2}` code for each option), `token_ids` (the
  tokenizer id of each code) and `max_input_tokens`.
- `decision_config.json`, the tokenizer files and `chat_template.jinja` are
  copied from the source repository without change.

## Equivalence

The reference model computes `logits = readout(norm(hidden))` at the last prompt
position, sets rows `>= count` to `-1e9`, and returns
`softmax(logits / temperature)[:count]`.

With the readout installed as `lm_head`, the ordinary forward gives the same
logits. The CBv2 engine returns `log_softmax` of the raw head logits at T=1. For
option `i < count`, `p_i = exp(l_i / T) / sum_{j < count} exp(l_j / T)`, where `l`
are those logprobs. The logsumexp shift cancels and the masked rows add zero, so
the result is the reference result to fp32 rounding.

A decision is one CBv2 request with `maxTokens` 1, greedy sampling and
`topLogprobs` 255. Use the top-255 `(token, logprob)` list of the first token.
The token ids of that list are head rows, so token id `i` is option `i`. Do not
use the sampled token text.

## What the fork changes

- `Libraries/MLXLMCommon/PplxDeciderConfiguration.swift`:
  `PplxDeciderConfiguration` decodes and validates the `decision` block.
  `PplxDeciderConfiguration.outputHeadSize(modelType:decision:tieWordEmbeddings:)`
  applies the model-type rule: the decider model type needs the block and an
  untied head; a block under another model type is an error.
  `PplxDecision.optionLogprobs(topLogprobs:count:)` puts the top-k list in option
  order and fails if an option is missing. `PplxDecision.probabilities(optionLogprobs:temperature:)`
  computes the probabilities in Double.
- `Qwen35TextConfiguration.outputHeadSize` (MLXLLM) and
  `Qwen35Configuration.TextConfiguration.outputHeadSize` (MLXVLM) set the head
  width. Only the top-level `Qwen35Configuration` decoders set them, from the
  `decision` block. `Qwen35TextModel` and `Qwen35Language.LanguageModel` build
  `lm_head` with `outputHeadSize ?? vocabularySize` rows. `vocabularySize` stays
  the embedding vocabulary.
- `Qwen35TextConfiguration.cbv2Capabilities` reports no MTP support for a
  decider head, and `Qwen35TextModel` does not attach an MTP head for it.
- `LLMTypeRegistry` and `VLMTypeRegistry` map `pplx_decider_qwen35` to the
  ordinary `Qwen35Model` and `Qwen35` classes. `VLMModelFactory.processorType`
  maps it to `Qwen3VLProcessor`.

## Engine notes

The CBv2 sampler, logits pipeline and logprob gather take the width from
`logits.dim(-1)`, so a 255-row head needs no engine change. `CBv2TeacherForcedScores`
takes a vocabulary size from the caller: pass the head width, not
`vocabularySize`. Logit diagnostic candidate ids must be head rows. Do not enable
the repetition penalty for decision requests: `LogitsPipelineV2.commit` scatters
into the window counts at prompt token ids, which can be larger than the head
width.

Tests: `Tests/MLXLMTests/PplxDeciderConfigurationTests.swift`, with
`Tests/MLXLMTests/Resources/pplx-decider-reference-vectors.json` copied from the
reference vectors.
