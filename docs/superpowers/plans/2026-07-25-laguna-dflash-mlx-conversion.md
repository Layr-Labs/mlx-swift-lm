# Laguna XS 2.1 DFlash → MLX Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `poolside/Laguna-XS-2.1-DFlash-NVFP4` to an MLX draft checkpoint and extend this fork so Laguna target + DFlash draft run end-to-end speculative generation.

**Architecture:** A stdlib-Python script normalizes the poolside checkpoint (splits fused qkv, flattens config) into the schema `DFlashDraftModel.load(from:)` already reads. `MLXSpeculative` gains three config-gated `laguna_xs` behaviors (per-head `g_proj` softplus gating, `aux_hidden_norms` before `fc`, per-layer `input_layernorm` on context K/V). A `LagunaModel` target is ported from the mlxfast-challenge vendored copy and conforms to `DFlashTargetModel`. The existing `mlx-bench` dflash command needs zero CLI changes (it is generic over `any DFlashTargetModel`).

**Tech Stack:** Swift (swift-testing for tests), MLX / MLXNN / MLXLMCommon, Python 3 stdlib (conversion only).

**Spec:** `docs/superpowers/specs/2026-07-25-laguna-dflash-mlx-conversion-design.md` (approved). Reference semantics source: vLLM PR #46853 (`DFlashLagunaForCausalLM`).

## Global Constraints

- Repo: `/Users/asv/projects/MLX-swift-lm-dflash`, branch `dflash-mlx-swift`. All work happens here. Never modify `/Users/asv/projects/mlxfast-challenge-dev` (read-only source for the vendored `Laguna.swift`).
- Tests use **swift-testing** (`import Testing`, `@Suite`, `@Test`, `#expect`) — never XCTest. Unit tests wrap model work in `try Device.withDefaultDevice(.cpu) { ... }` and pin randomness with `MLXRandom.seed(N)` when numerics matter.
- Run a single test suite with:
  `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/<SuiteName> 2>&1 | tail -30`
  (Expect this to take a few minutes; shader compilation is required, plain `swift test` is NOT the repo convention.)
- The qwen3 DFlash path must stay behavior-identical: no edits to existing logic except where a `decoderLayerType == .lagunaXS` branch is added. Existing suites `DFlashConfigurationTests`, `DFlashDraftModelTests`, `DFlashTokenIteratorTests`, `QwenDFlashForwardTests` must stay green.
- Conversion preserves BF16 exactly (raw byte copies; no tensor library, no torch/numpy).
- The draft checkpoint has NO `embed_tokens`/`lm_head` — it borrows both from the target via `bind(target:)`. Never add them.
- Commit after each task with the message given in the task. End commit messages with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
- Machine limits: this Mac has 36 GiB RAM. Never load the 21.6 GB Laguna target here — tiny synthetic configs only. Task 9 (e2e) runs on the M5-B box and is gated on operator confirmation.

## Key numbers (Laguna XS 2.1 DFlash)

| Field | Value |
|---|---|
| draft layers | 5, all `sliding_attention`, window 512 |
| hidden / heads / kv heads / head_dim | 2048 / 64 / 8 / 128 |
| intermediate (silu) | 8192 |
| vocab (= target vocab) | 100352 |
| rms_norm_eps / rope_theta | 1e-6 / 500000 (plain full-dim RoPE) |
| block_size / mask_token_id | 16 / 12 |
| num_target_layers / target_layer_ids | 40 / [1, 13, 25, 33, 39] |
| fused qkv rows | q 8192, k 1024, v 1024 (order [q;k;v]) |
| source tensors | 58 (all BF16), single `model.safetensors`, 924 MB |
| converted tensors | 68 (each of 5 `qkv_proj` → 3 tensors) |

---

### Task 1: Conversion script `scripts/convert_laguna_dflash.py`

**Files:**
- Create: `scripts/convert_laguna_dflash.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: CLI `python3 scripts/convert_laguna_dflash.py --source <dir> --output <dir> [--source-revision <sha>]` and `python3 scripts/convert_laguna_dflash.py --selftest`. Output dir contains `model.safetensors` + `config.json` in the flat schema Task 2's Swift decoder reads (exact JSON below — Task 2's fixture must match it field-for-field).

**Safetensors format facts (implement exactly):** file = 8-byte little-endian u64 header length `N`, then `N` bytes of UTF-8 JSON, then the raw data buffer. Header maps tensor name → `{"dtype": "BF16", "shape": [r, c], "data_offsets": [start, end]}` (offsets relative to the start of the data buffer) plus optional `"__metadata__"` (string→string map). Row-major layout: splitting `[R, C]` along axis 0 at row `r` is the byte range `[r*C*2, ...)` for BF16 (2 bytes/element).

- [ ] **Step 1: Write the script**

Implement in `scripts/convert_laguna_dflash.py` (stdlib only: `argparse, json, hashlib, struct, sys, os, tempfile, pathlib, datetime`):

1. `read_header(path) -> tuple[dict, int]` — returns (header dict without `__metadata__` kept separate is fine, absolute file offset where the data buffer starts).
2. `expected_source_manifest(cfg) -> dict[str, tuple[str, list[int]]]` — builds the exact expected name → (dtype, shape) map from the source config values (`num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, and `len(target_layer_ids)` aux norms). For layer `i`: `layers.{i}.self_attn.qkv_proj.weight [(nH+2*nKV)*hd, H]`, `layers.{i}.self_attn.g_proj.weight [nH, H]`, `layers.{i}.self_attn.q_norm.weight [hd]`, `layers.{i}.self_attn.k_norm.weight [hd]`, `layers.{i}.self_attn.o_proj.weight [H, nH*hd]`, `layers.{i}.mlp.gate_proj.weight [I, H]`, `layers.{i}.mlp.up_proj.weight [I, H]`, `layers.{i}.mlp.down_proj.weight [H, I]`, `layers.{i}.input_layernorm.weight [H]`, `layers.{i}.post_attention_layernorm.weight [H]`. Root: `fc.weight [H, len(ids)*H]`, `hidden_norm.weight [H]`, `norm.weight [H]`, `aux_hidden_norms.{j}.weight [H]` for each j. All dtype `BF16`.
3. `convert(source_dir, output_dir, source_revision)`:
   - Load `source_dir/config.json`. Assert `config["architectures"] == ["DFlashLagunaForCausalLM"]` and `config["model_type"] == "laguna"`; assert `config["draft_vocab_size"] == config["vocab_size"]`; assert every entry of `config["layer_types"]` equals `"sliding_attention"` (uniform drafter — hard error otherwise).
   - Read the source safetensors header. Compare against `expected_source_manifest` **exactly** (same key set, same dtype, same shape per key) — on any mismatch print every discrepancy and exit 1.
   - Build the output tensor list: every non-qkv tensor passes through unchanged; each `layers.{i}.self_attn.qkv_proj.weight` becomes `q_proj.weight` (rows `[0, nH*hd)`), `k_proj.weight` (rows `[nH*hd, nH*hd + nKV*hd)`), `v_proj.weight` (rows `[nH*hd + nKV*hd, nH*hd + 2*nKV*hd)`), same second dim.
   - Write `output_dir/model.safetensors`: keys sorted, contiguous `data_offsets`, `__metadata__ = {"format": "mlx", "source_repo": "poolside/Laguna-XS-2.1-DFlash-NVFP4", "source_revision": <arg or "unknown">, "source_sha256": <streaming sha256 of source file>, "converter": "convert_laguna_dflash.py v1"}`. Stream bytes with seek+read in ≤64 MiB chunks — never load the whole file.
   - Write `output_dir/config.json` **exactly** this structure (values taken from the source config, not hardcoded, except the noted literals):

```json
{
  "architectures": ["DFlashDraftModel"],
  "model_type": "laguna",
  "decoder_layer_type": "laguna_xs",
  "gating": "per-head",
  "hidden_size": 2048,
  "num_hidden_layers": 5,
  "intermediate_size": 8192,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 100352,
  "rms_norm_eps": 1e-06,
  "rope_theta": 500000.0,
  "max_position_embeddings": 262144,
  "sliding_window": 512,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
  "tie_word_embeddings": false,
  "block_size": 16,
  "num_target_layers": 40,
  "dflash_config": {"target_layer_ids": [1, 13, 25, 33, 39], "mask_token_id": 12},
  "_mlx_conversion": {"source_repo": "...", "source_revision": "...", "source_sha256": "...", "converter": "convert_laguna_dflash.py v1", "converted_at": "<UTC ISO8601>"}
}
```

   Literals: `architectures` is `["DFlashDraftModel"]`; `decoder_layer_type` is `"laguna_xs"`; `gating` copies the source `gating` field (assert it equals `"per-head"`); `block_size` and `num_target_layers` are hoisted OUT of the source's nested `dflash_config`; `dflash_config` keeps only `target_layer_ids` + `mask_token_id`. Assert `mask_token_id < vocab_size` and every target layer id `< num_target_layers`.
4. `selftest()`: in a temp dir, synthesize a tiny source checkpoint with the same schema (config: 2 layers, hidden 8, heads 2, kv 1, head_dim 4, intermediate 16, vocab 32, `dflash_config = {"block_size": 4, "mask_token_id": 3, "num_target_layers": 4, "target_layer_ids": [0, 2], "causal": true}`, `layer_types` 2× sliding, `sliding_window` 8, plus `architectures`, `model_type`, `draft_vocab_size` 32, `gating` "per-head"). Fill each tensor with distinct deterministic bytes (e.g. tensor index repeated). Run `convert`. Then assert: output has 29 tensors (per layer 10 source tensors → 12 after the qkv split, ×2 layers = 24; root = 2 aux norms + fc + hidden_norm + norm = 5); q/k/v byte contents equal the corresponding row-slices of the fused source bytes; all pass-through tensors byte-identical; output config parses and contains the hoisted `block_size`/`num_target_layers` and `decoder_layer_type == "laguna_xs"`. Print `SELFTEST OK` and exit 0. (Compute the expected count in the selftest from the synthetic config rather than hardcoding 29.)

- [ ] **Step 2: Run the selftest**

Run: `python3 scripts/convert_laguna_dflash.py --selftest`
Expected: `SELFTEST OK`, exit 0.

- [ ] **Step 3: Download the real checkpoint and convert it**

```bash
mkdir -p ~/models/laguna-dflash-src
curl -sL https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-NVFP4/raw/main/config.json -o ~/models/laguna-dflash-src/config.json
curl -L --progress-bar https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-NVFP4/resolve/main/model.safetensors -o ~/models/laguna-dflash-src/model.safetensors
REV=$(curl -sI https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-NVFP4/resolve/main/config.json | tr -d '\r' | awk -F': ' 'tolower($1)=="x-repo-commit"{print $2}')
python3 scripts/convert_laguna_dflash.py --source ~/models/laguna-dflash-src --output ~/models/laguna-xs-2.1-dflash-mlx --source-revision "$REV"
```

Expected: conversion prints a summary (58 source tensors → 68 output tensors) and exits 0. Sanity: `python3 -c` snippet reading the output header — 68 tensors, `layers.0.self_attn.q_proj.weight` shape `[8192, 2048]`, `k/v` `[1024, 2048]`, all BF16.

- [ ] **Step 4: Commit**

```bash
git add scripts/convert_laguna_dflash.py
git commit -m "Add Laguna DFlash checkpoint converter"
```

---

### Task 2: `DFlashConfiguration` laguna_xs schema support

**Files:**
- Modify: `Libraries/MLXSpeculative/DFlashConfiguration.swift`
- Create: `Tests/MLXLMTests/Resources/dflash-laguna-xs-2.1-config.json`
- Modify: `Package.swift` (add the resource to the `MLXLMTests` target's `resources:` list, alongside `.process("Resources/dflash-gemma4-gated-schema-config.json")` at ~line 151)
- Test: `Tests/MLXLMTests/DFlashConfigurationTests.swift`

**Interfaces:**
- Consumes: existing `DFlashConfiguration` (`Libraries/MLXSpeculative/DFlashConfiguration.swift`).
- Produces: `public enum DFlashDecoderLayerType: String, Codable, Sendable, Equatable { case qwen3; case lagunaXS = "laguna_xs" }`; `DFlashConfiguration.decoderLayerType: DFlashDecoderLayerType` (default `.qwen3`); `DFlashConfiguration.gating: String?`. Tasks 3–4 branch on `config.decoderLayerType == .lagunaXS`.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/MLXLMTests/DFlashConfigurationTests.swift` (fixture loader `loadFixture(named:)` already exists at the top of the file):

```swift
@Test func decodesLagunaXSFixture() throws {
    let config = try JSONDecoder.json5().decode(
        DFlashConfiguration.self,
        from: try loadFixture(named: "dflash-laguna-xs-2.1-config"))
    #expect(config.decoderLayerType == .lagunaXS)
    #expect(config.gating == "per-head")
    #expect(config.blockSize == 16)
    #expect(config.numTargetLayers == 40)
    #expect(config.targetLayerIds == [1, 13, 25, 33, 39])
    #expect(config.maskTokenId == 12)
    #expect(config.slidingWindow == 512)
    #expect(config.layerTypes == Array(repeating: .slidingAttention, count: 5))
    #expect(config.targetHiddenSize == 5 * 2048)
    #expect(config.ropeTheta == 500000.0)
}

@Test func decoderLayerTypeDefaultsToQwen3() throws {
    let config = try JSONDecoder.json5().decode(
        DFlashConfiguration.self, from: Data(validJSON().utf8))
    #expect(config.decoderLayerType == .qwen3)
    #expect(config.gating == nil)
}

@Test func rejectsLagunaXSWithNonPerHeadGating() throws {
    var json = try String(
        data: loadFixture(named: "dflash-laguna-xs-2.1-config"), encoding: .utf8)!
    json = json.replacingOccurrences(of: "\"per-head\"", with: "\"per-element\"")
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder.json5().decode(
            DFlashConfiguration.self, from: Data(json.utf8))
    }
}
```

Create `Tests/MLXLMTests/Resources/dflash-laguna-xs-2.1-config.json` with byte-for-byte the JSON emitted by Task 1's converter for the real checkpoint (the block shown in Task 1 Step 1.3, including a representative `_mlx_conversion` block — unknown keys must land in `ignoredConfigKeys` without failing). Register it in `Package.swift`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/DFlashConfigurationTests 2>&1 | tail -30`
Expected: FAIL — `decoderLayerType` does not exist (compile error is the failure mode here; that counts).

- [ ] **Step 3: Implement**

In `Libraries/MLXSpeculative/DFlashConfiguration.swift`:

```swift
public enum DFlashDecoderLayerType: String, Codable, Sendable, Equatable {
    case qwen3
    case lagunaXS = "laguna_xs"
}
```

Add stored properties `public var decoderLayerType: DFlashDecoderLayerType` and `public var gating: String?`; add `CodingKeys` cases `decoderLayerType = "decoder_layer_type"` and `gating` (the enum is `CaseIterable` — new cases automatically leave `ignoredConfigKeys` accounting correct). In `init(from:)` decode with `?? .qwen3` / `decodeIfPresent`, then validate:

```swift
if decoderLayerType == .lagunaXS, let gating, gating != "per-head" {
    throw DecodingError.dataCorruptedError(
        forKey: .gating, in: container,
        debugDescription:
            "DFlash laguna_xs drafters require per-head gating; got \(gating).")
}
```

- [ ] **Step 4: Run tests to verify they pass**

Same command. Expected: PASS, including all pre-existing tests in the suite.

- [ ] **Step 5: Commit**

```bash
git add Libraries/MLXSpeculative/DFlashConfiguration.swift Tests/MLXLMTests/Resources/dflash-laguna-xs-2.1-config.json Tests/MLXLMTests/DFlashConfigurationTests.swift Package.swift
git commit -m "Add laguna_xs decoder layer type to DFlashConfiguration"
```

---

### Task 3: laguna_xs draft-model behaviors in `DFlashDraftModel`

**Files:**
- Modify: `Libraries/MLXSpeculative/DFlashDraftModel.swift`
- Test: `Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift` (create)

**Interfaces:**
- Consumes: `DFlashConfiguration.decoderLayerType` from Task 2.
- Produces: `DFlashDraftModel` that, for `laguna_xs` configs, (1) builds `@ModuleInfo(key: "aux_hidden_norms") auxHiddenNorms: [RMSNorm]?` (count = `targetLayerIds.count`), (2) builds `@ModuleInfo(key: "g_proj") gProj: Linear?` inside `DFlashAttention` (`Linear(hiddenSize, attentionHeads, bias: false)`), (3) applies the layer's `input_layernorm` to the shared context before context K/V projection. Weight paths must be exactly `aux_hidden_norms.{j}.weight` and `layers.{i}.self_attn.g_proj.weight` (what Task 1 emits). Task 4 tests these numerics; Task 9 loads the real artifact.

Reference semantics (vLLM `DFlashLagunaForCausalLM` — normative):
1. Context combine: view `targetHidden [..., k*H] → [..., k, H]`, apply `auxHiddenNorms[j]` to slice j, re-concat, then existing `fc → hidden_norm`.
2. Each layer's context K/V comes from `input_layernorm_l(context)` (per-layer norm of the SAME shared context vector). The qwen3 path (no per-layer norm) must be untouched.
3. Attention output gating: `softplus(g_proj(x))` in float32 (x = the already-input-layernormed attention input), cast back, broadcast over head_dim, applied to the pre-`o_proj` attention output rows.

- [ ] **Step 1: Write the failing tests**

Create `Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift`:

```swift
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXSpeculative

@Suite("Laguna DFlash draft model")
struct LagunaDFlashDraftModelTests {

    private func lagunaConfigJSON(decoderLayerType: String = "laguna_xs") -> String {
        """
        {
          "architectures": ["DFlashDraftModel"],
          "model_type": "laguna",
          "decoder_layer_type": "\(decoderLayerType)",
          "gating": "per-head",
          "hidden_size": 8,
          "num_hidden_layers": 2,
          "intermediate_size": 16,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "vocab_size": 32,
          "rms_norm_eps": 1e-6,
          "rope_theta": 500000.0,
          "max_position_embeddings": 4096,
          "block_size": 4,
          "num_target_layers": 4,
          "sliding_window": 8,
          "layer_types": ["sliding_attention", "sliding_attention"],
          "dflash_config": {"target_layer_ids": [0, 2], "mask_token_id": 3}
        }
        """
    }

    private func makeConfig(decoderLayerType: String = "laguna_xs") throws -> DFlashConfiguration {
        try JSONDecoder.json5().decode(
            DFlashConfiguration.self,
            from: Data(lagunaConfigJSON(decoderLayerType: decoderLayerType).utf8))
    }

    @Test func lagunaXSBuildsGatingAndAuxNormParameters() throws {
        try Device.withDefaultDevice(.cpu) {
            let draft = DFlashDraftModel(config: try makeConfig())
            let params = Dictionary(
                uniqueKeysWithValues: draft.parameters().flattened())
            #expect(params["aux_hidden_norms.0.weight"]?.shape == [8])
            #expect(params["aux_hidden_norms.1.weight"]?.shape == [8])
            #expect(params["layers.0.self_attn.g_proj.weight"]?.shape == [2, 8])
            #expect(params["layers.1.self_attn.g_proj.weight"]?.shape == [2, 8])
        }
    }

    @Test func qwen3ConfigBuildsNoLagunaParameters() throws {
        try Device.withDefaultDevice(.cpu) {
            let draft = DFlashDraftModel(config: try makeConfig(decoderLayerType: "qwen3"))
            let keys = Set(draft.parameters().flattened().map(\.0))
            #expect(!keys.contains { $0.hasPrefix("aux_hidden_norms") })
            #expect(!keys.contains { $0.contains("g_proj") })
        }
    }

    @Test func lagunaXSForwardDiffersFromQwen3WithSharedWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(7)
            let laguna = DFlashDraftModel(config: try makeConfig())
            MLXRandom.seed(7)
            let qwen = DFlashDraftModel(config: try makeConfig(decoderLayerType: "qwen3"))
            // Copy the shared subset of weights laguna->qwen so the only
            // differences are the laguna_xs behaviors themselves.
            let lagunaParams = Dictionary(
                uniqueKeysWithValues: laguna.parameters().flattened())
            let qwenKeys = Set(qwen.parameters().flattened().map(\.0))
            let shared = lagunaParams.filter { qwenKeys.contains($0.key) }
            try qwen.update(
                parameters: ModuleParameters.unflattened(shared), verify: [.noUnusedKeys])

            let target = LagunaDraftStubTarget(hiddenSize: 8, vocabularySize: 32, layerCount: 4)
            try laguna.bind(target: target)
            try qwen.bind(target: target)
            eval(laguna, qwen)

            let hidden = MLXRandom.normal([1, 3, 16]).asType(.bfloat16)
            let block = MLXArray([Int32(5), 3, 3, 3])[.newAxis, .ellipsis]
            let a = try laguna(
                block, targetHidden: hidden, cache: try laguna.makeCache(), logitsStart: 1)
            let b = try qwen(
                block, targetHidden: hidden, cache: try qwen.makeCache(), logitsStart: 1)
            eval(a, b)
            #expect(!allClose(a, b, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        }
    }
}

/// Minimal deterministic target: embedding table + linear head, seeded.
final class LagunaDraftStubTarget: Module, DFlashTargetModel {
    let vocabularySize: Int
    let kvHeads: [Int] = []
    let dFlashVocabularySize: Int
    let dFlashHiddenSize: Int
    let dFlashLayerCount: Int
    let embedWeight: MLXArray
    let headWeight: MLXArray
    var loraLayers: [Module] { [] }

    init(hiddenSize: Int, vocabularySize: Int, layerCount: Int) {
        self.vocabularySize = vocabularySize
        self.dFlashVocabularySize = vocabularySize
        self.dFlashHiddenSize = hiddenSize
        self.dFlashLayerCount = layerCount
        MLXRandom.seed(99)
        self.embedWeight = MLXRandom.normal([vocabularySize, hiddenSize]).asType(.bfloat16)
        self.headWeight = MLXRandom.normal([vocabularySize, hiddenSize]).asType(.bfloat16)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logitsForDFlashHidden(embedTokensForDFlash(inputs))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

    func forwardForDFlash(
        _ inputs: MLXArray, cache: [KVCache]?, targetLayerIds: [Int]
    ) throws -> DFlashTargetForward {
        let h = embedTokensForDFlash(inputs)
        return DFlashTargetForward(
            logits: logitsForDFlashHidden(h),
            hiddenStates: targetLayerIds.map { _ in h })
    }

    func embedTokensForDFlash(_ tokens: MLXArray) -> MLXArray {
        embedWeight[tokens]
    }

    func logitsForDFlashHidden(_ hidden: MLXArray) -> MLXArray {
        matmul(hidden, headWeight.T)
    }
}
```

(If the stub's protocol conformance needs more members than listed, mirror the stub pattern in `Tests/MLXLMTests/DFlashTokenIteratorTests.swift:360` — `ReplayRequiredTarget` — which is the canonical example.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/LagunaDFlashDraftModelTests 2>&1 | tail -30`
Expected: FAIL (no `aux_hidden_norms` / `g_proj` parameters exist yet).

- [ ] **Step 3: Implement in `DFlashDraftModel.swift`**

Three edits, all gated on `config.decoderLayerType == .lagunaXS`:

1. `DFlashAttention`: add `@ModuleInfo(key: "g_proj") var gProj: Linear?`; in `init`, `if config.decoderLayerType == .lagunaXS { _gProj.wrappedValue = Linear(config.hiddenSize, config.attentionHeads, bias: false) }`. In `callAsFunction`, replace the final `return oProj(...)` with:

```swift
var output = MLXFast.scaledDotProductAttention(
    queries: queries, keys: keys, values: values, scale: scale, mask: mask
).transposed(0, 2, 1, 3).reshaped(B, L, -1)

if let gProj {
    let gate = softplus(gProj(x).asType(.float32)).asType(output.dtype)
    output = (output.reshaped(B, L, config.attentionHeads, config.headDim)
        * gate[.ellipsis, .newAxis]).reshaped(B, L, -1)
}
return oProj(output)
```

2. `DFlashDecoderLayer`: store `let normalizesContext: Bool` (`config.decoderLayerType == .lagunaXS` in init). In `callAsFunction`:

```swift
let layerContext = normalizesContext ? inputLayerNorm(context) : context
let h = try x + selfAttn(
    inputLayerNorm(x), context: layerContext, rope: rope, cache: cache)
return h + mlp(postAttentionLayerNorm(h))
```

3. `DFlashDraftModel`: add `@ModuleInfo(key: "aux_hidden_norms") public var auxHiddenNorms: [RMSNorm]?`; in `init`, when laguna_xs: `_auxHiddenNorms.wrappedValue = (0 ..< config.targetLayerIds.count).map { _ in RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps) }`. In `callAsFunction`, before `contextProjection`:

```swift
var combinedTargetHidden = targetHidden
if let auxHiddenNorms {
    let sliceCount = auxHiddenNorms.count
    let normed = (0 ..< sliceCount).map { j in
        auxHiddenNorms[j](
            targetHidden[.ellipsis, (j * config.hiddenSize) ..< ((j + 1) * config.hiddenSize)])
    }
    combinedTargetHidden = concatenated(normed, axis: -1)
}
let context = hiddenNorm(contextProjection(combinedTargetHidden))
```

If `@ModuleInfo` rejects an optional module array, fall back to a non-optional `[RMSNorm]` that is empty for qwen3 and guard on `!auxHiddenNorms.isEmpty` — empty module arrays contribute no parameters, so `verify: [.all]` stays satisfied for qwen3 checkpoints.

- [ ] **Step 4: Run new suite + qwen3 regression suites**

Run: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/LagunaDFlashDraftModelTests -only-testing:MLXLMTests/DFlashDraftModelTests -only-testing:MLXLMTests/DFlashConfigurationTests -only-testing:MLXLMTests/DFlashTokenIteratorTests 2>&1 | tail -30`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add Libraries/MLXSpeculative/DFlashDraftModel.swift Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift
git commit -m "Add laguna_xs draft behaviors to DFlashDraftModel"
```

---

### Task 4: laguna_xs numeric parity test vs in-test reference implementation

**Files:**
- Test: `Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift` (extend)

**Interfaces:**
- Consumes: Task 3's implementation, `LagunaDraftStubTarget` from Task 3.
- Produces: a parity test that re-derives the full laguna_xs draft forward from raw MLX ops; it is the unit-level correctness authority for the laguna_xs semantics.

- [ ] **Step 1: Write the parity test (it should PASS if Task 3 is correct — it is a verification gate, not TDD red)**

Append to the suite:

```swift
@Test func lagunaXSForwardMatchesReferenceImplementation() throws {
    try Device.withDefaultDevice(.cpu) {
        MLXRandom.seed(11)
        let config = try makeConfig()
        let draft = DFlashDraftModel(config: config)
        let target = LagunaDraftStubTarget(hiddenSize: 8, vocabularySize: 32, layerCount: 4)
        try draft.bind(target: target)
        eval(draft)

        let p = Dictionary(uniqueKeysWithValues: draft.parameters().flattened())
        func w(_ key: String) -> MLXArray { p[key]!.asType(.float32) }
        let eps: Float = 1e-6
        func rms(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(x, weight: weight, eps: eps)
        }

        let contextLength = 3
        MLXRandom.seed(21)
        let targetHidden = MLXRandom.normal([1, contextLength, 16]).asType(.float32)
        let block: [Int32] = [5, 3, 3, 3]
        let blockArray = MLXArray(block)[.newAxis, .ellipsis]

        // --- Reference forward (float32, mirrors vLLM DFlashLagunaForCausalLM) ---
        let slices = (0 ..< 2).map { j in
            rms(targetHidden[.ellipsis, (j * 8) ..< ((j + 1) * 8)],
                w("aux_hidden_norms.\(j).weight"))
        }
        var ctx = concatenated(slices, axis: -1)
        ctx = matmul(ctx, w("fc.weight").T)
        ctx = rms(ctx, w("hidden_norm.weight"))

        var h = target.embedTokensForDFlash(blockArray).asType(.float32)
        let rope = initializeRope(
            dims: 4, base: 500000.0, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: 4096)

        for layer in 0 ..< 2 {
            let prefix = "layers.\(layer)"
            let xin = rms(h, w("\(prefix).input_layernorm.weight"))
            let ctxin = rms(ctx, w("\(prefix).input_layernorm.weight"))

            func heads(_ x: MLXArray, _ n: Int) -> MLXArray {
                x.reshaped(1, x.dim(1), n, 4).transposed(0, 2, 1, 3)
            }
            var q = heads(matmul(xin, w("\(prefix).self_attn.q_proj.weight").T), 2)
            q = rms(q, w("\(prefix).self_attn.q_norm.weight"))
            var pk = heads(matmul(xin, w("\(prefix).self_attn.k_proj.weight").T), 1)
            pk = rms(pk, w("\(prefix).self_attn.k_norm.weight"))
            let pv = heads(matmul(xin, w("\(prefix).self_attn.v_proj.weight").T), 1)
            var ck = heads(matmul(ctxin, w("\(prefix).self_attn.k_proj.weight").T), 1)
            ck = rms(ck, w("\(prefix).self_attn.k_norm.weight"))
            let cv = heads(matmul(ctxin, w("\(prefix).self_attn.v_proj.weight").T), 1)

            q = rope(q, offset: contextLength)
            pk = rope(pk, offset: contextLength)
            ck = rope(ck, offset: 0)

            let keys = concatenated([ck, pk], axis: 2)
            let values = concatenated([cv, pv], axis: 2)
            let mask = createCausalMask(
                n: 4, offset: contextLength, windowSize: 8)
            let attn = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values,
                scale: pow(4.0, -0.5), mask: .array(mask))
            var out = attn.transposed(0, 2, 1, 3).reshaped(1, 4, 8)

            let gate = softplus(
                matmul(xin, w("\(prefix).self_attn.g_proj.weight").T))
            out = (out.reshaped(1, 4, 2, 4) * gate[.ellipsis, .newAxis])
                .reshaped(1, 4, 8)
            let attnOut = matmul(out, w("\(prefix).self_attn.o_proj.weight").T)
            h = h + attnOut

            let hin = rms(h, w("\(prefix).post_attention_layernorm.weight"))
            let mlpOut = matmul(
                silu(matmul(hin, w("\(prefix).mlp.gate_proj.weight").T))
                    * matmul(hin, w("\(prefix).mlp.up_proj.weight").T),
                w("\(prefix).mlp.down_proj.weight").T)
            h = h + mlpOut
        }
        let expected = target.logitsForDFlashHidden(
            rms(h, w("norm.weight")).asType(.bfloat16))[0..., 1..., 0...]

        // --- Model forward ---
        let actual = try draft(
            blockArray,
            targetHidden: targetHidden.asType(.bfloat16),
            cache: try draft.makeCache(),
            logitsStart: 1)
        eval(expected, actual)
        #expect(
            allClose(actual.asType(.float32), expected.asType(.float32),
                rtol: 2e-2, atol: 2e-2).item(Bool.self),
            "laguna_xs draft forward diverged from reference")
    }
}
```

Notes for the implementer: the model runs mixed bf16/f32 while the reference runs f32 with a bf16 boundary at embed/head, hence the loose 2e-2 tolerance — what this test catches is structural miswiring (wrong norm, wrong order, missing gate), which produces O(1) divergence, not O(1e-2). `createCausalMask` comes from `MLXLMCommon` (`createCausalMask(n:offset:windowSize:lengths:leftPadding:)` — pass only the first three). RoPE offsets mirror `DFlashAttention.callAsFunction` (proposal at `baseOffset + contextLength` with fresh cache ⇒ `contextLength`; context at 0). If the assertion fails, debug by comparing intermediates (`ctx` first, then per-layer `h`) between reference and a `@testable` peek — do NOT loosen the tolerance beyond 5e-2; a genuine mismatch means Task 3 has a bug and must be fixed there.

- [ ] **Step 2: Run the suite**

Run: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/LagunaDFlashDraftModelTests 2>&1 | tail -30`
Expected: PASS. If the parity test fails, fix `DFlashDraftModel.swift` (Task 3's deliverable), not the test.

- [ ] **Step 3: Commit**

```bash
git add Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift
git commit -m "Add laguna_xs draft parity test vs reference forward"
```

---

### Task 5: Port `LagunaModel` target into the fork

**Files:**
- Create: `Libraries/MLXLLM/Models/Laguna.swift` (port of `/Users/asv/projects/mlxfast-challenge-dev/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Laguna.swift` — read that file in full first; it is 589 lines)
- Modify: `Libraries/MLXLLM/LLMModelFactory.swift` (~line 35, inside `LLMTypeRegistry`'s creators dictionary)
- Test: `Tests/MLXLMTests/LagunaModelTests.swift` (create)

**Interfaces:**
- Consumes: fork MLXLMCommon APIs — all verified present: `initializeRope(dims:base:traditional:scalingConfig:maxPositionEmbeddings:)` (`RoPEUtils.swift:338`), `applyRotaryPosition(_:to:cache:)` (`RoPEApplication.swift:35`), `attentionWithCacheUpdate(...)` (`AttentionUtils.swift:37`), `SwitchGLU(inputDims:hiddenDims:numExperts:)` (`SwitchLayers.swift:182`), `KVCacheSimple`, `RotatingKVCache(maxSize:)`, `createAttentionMask(h:cache:windowSize:)`, `StringOrNumber`, `QuantizationMode`.
- Produces: `public class LagunaModel: Module, LLMModel, KVCacheDimensionProvider` with `public init(_ config: LagunaConfiguration)`, `public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray`, `public func newCache(parameters: GenerateParameters?) -> [KVCache]`, `public struct LagunaConfiguration: Codable, Sendable`; factory registration `"laguna"`. Task 6 adds `DFlashTargetModel` conformance and needs: `config` accessible internally (`let config: LagunaConfiguration`), inner `model: LagunaModelInner` with `embedTokens` and `layers` reachable from an extension in the same file (keep them `fileprivate`), and `lmHead: Linear?`.

**Port rules (apply while copying):**
1. Copy the vendored file verbatim, then make ONLY these changes:
2. **Delete the constructor-time quantize pass** (the `if let groupSize = config.quantGroupSize ...` block at the end of `LagunaModel.init`, ~lines 328-338) AND the now-unused config fields it reads (`quantGroupSize`, `quantBits`, `quantMode`, plus `LagunaQuantizationBlock` / `LagunaQuantizationCodingKeys` and the trailing decode block in `LagunaConfiguration.init(from:)`). Rationale: this fork's loader (`Libraries/MLXLMCommon/Load.swift:40-52`) quantizes every module that has a `<path>.scales` tensor using `config.json`'s `quantization` block via `BaseConfiguration.PerLayerQuantization` — model-side quantization would double-apply.
3. **Replace `weightedExpertSum`** (absent in this fork). In `LagunaSparseMoeBlock.callAsFunction`, replace
   `y = weightedExpertSum(y, weights.asType(y.dtype))` with
   `y = (expandDims(weights.asType(y.dtype), axis: -1) * y).sum(axis: -2)`
   (documented in the challenge tree as numerically identical).
4. Keep everything else: `LagunaGating`, per-head softplus gating, YaRN-via-`ropeParameters(forLayer:)`, `sanitize` (inv_freq filter + tied-embeddings drop), `newCache` (full → `KVCacheSimple()`, sliding → `RotatingKVCache(maxSize: config.slidingWindow)`), `LoRAModel` extension returning `model.layers`.
5. Register in `LLMModelFactory.swift` creators dictionary (alphabetical placement near `"gemma4_text"`):
   `"laguna": create(LagunaConfiguration.self, LagunaModel.init),`
6. If the compiler flags a missing/renamed helper not listed above, find the fork-native equivalent by reading how `Qwen3.swift` or `Gemma4Text.swift` does the same thing — do not invent new helpers.

- [ ] **Step 1: Write the failing tests**

Create `Tests/MLXLMTests/LagunaModelTests.swift`:

```swift
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("LagunaModel")
struct LagunaModelTests {

    static let tinyConfigJSON = """
        {
          "model_type": "laguna",
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "max_position_embeddings": 4096,
          "rms_norm_eps": 1e-6,
          "attention_bias": false,
          "qkv_bias": false,
          "tie_word_embeddings": false,
          "rope_theta": 500000.0,
          "sliding_window": 8,
          "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
          "gating": "per-head",
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 8,
          "shared_expert_intermediate_size": 8,
          "moe_routed_scaling_factor": 1.0,
          "norm_topk_prob": true,
          "decoder_sparse_step": 1,
          "moe_router_logit_softcapping": 0.0
        }
        """

    private func tinyConfig() throws -> LagunaConfiguration {
        try JSONDecoder.json5().decode(
            LagunaConfiguration.self, from: Data(Self.tinyConfigJSON.utf8))
    }

    @Test func decodesTinyConfig() throws {
        let config = try tinyConfig()
        #expect(config.numHiddenLayers == 4)
        #expect(config.gatingEnabled)
        #expect(config.gatePerHead)
        // mlp_only_layers defaults to [0]: layer 0 dense, the rest sparse.
        #expect(!config.isSparse(layer: 0))
        #expect(config.isSparse(layer: 1))
    }

    @Test func forwardProducesLogitsAndFillsCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LagunaModel(try tinyConfig())
            eval(model)
            let cache = model.newCache(parameters: nil)
            #expect(cache.count == 4)
            #expect(cache[0] is RotatingKVCache)
            #expect(cache[1] is KVCacheSimple)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let logits = model(tokens, cache: cache)
            eval(logits)
            #expect(logits.shape == [1, 3, 32])
            #expect(cache[0].offset == 3)
        }
    }

    @Test func factoryRegistersLaguna() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(Self.tinyConfigJSON.utf8), modelType: "laguna")
        #expect(model is LagunaModel)
    }

    @Test func sanitizeDropsRotaryTables() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = LagunaModel(try tinyConfig())
            let cleaned = model.sanitize(weights: [
                "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2])
            ])
            #expect(cleaned.isEmpty)
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/LagunaModelTests 2>&1 | tail -30`
Expected: FAIL (LagunaModel does not exist — compile error).

- [ ] **Step 3: Port the file per the Port rules above, register the factory entry**

- [ ] **Step 4: Run tests to verify they pass**

Same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Libraries/MLXLLM/Models/Laguna.swift Libraries/MLXLLM/LLMModelFactory.swift Tests/MLXLMTests/LagunaModelTests.swift
git commit -m "Port Laguna target model from mlxfast-challenge vendored fork"
```

---

### Task 6: `DFlashTargetModel` conformance for Laguna

**Files:**
- Modify: `Libraries/MLXLLM/Models/Laguna.swift`
- Test: `Tests/MLXLMTests/LagunaDFlashForwardTests.swift` (create; mirror `Tests/MLXLMTests/QwenMoEDFlashForwardTests.swift`, the smallest template at 103 lines — read it first)

**Interfaces:**
- Consumes: Task 5's `LagunaModel` (same-file `fileprivate` access to `model.layers`, `model.embedTokens`, `model.norm`, `lmHead`); `DFlashTargetForward` / `DFlashTargetValidation` from `Libraries/MLXLLM/DFlashTarget.swift`.
- Produces: `extension LagunaModel: DFlashTargetModel` — the surface `mlx-bench` casts to (`context.model as? any DFlashTargetModel`) and `DFlashDraftModel.bind(target:)` validates against.

- [ ] **Step 1: Write the failing tests**

Create `Tests/MLXLMTests/LagunaDFlashForwardTests.swift`:

```swift
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("LagunaModel forwardForDFlash")
struct LagunaDFlashForwardTests {

    private func tinyModel() throws -> LagunaModel {
        let config = try JSONDecoder.json5().decode(
            LagunaConfiguration.self, from: Data(LagunaModelTests.tinyConfigJSON.utf8))
        return LagunaModel(config)
    }

    @Test func exposesDFlashSurface() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            #expect(model.dFlashVocabularySize == 32)
            #expect(model.dFlashHiddenSize == 8)
            #expect(model.dFlashLayerCount == 4)
        }
    }

    @Test func logitsMatchPlainForward() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let plain = model(tokens, cache: model.newCache(parameters: nil))
            let dflash = try model.forwardForDFlash(
                tokens, cache: model.newCache(parameters: nil), targetLayerIds: [1])
            eval(plain, dflash.logits)
            #expect(allClose(plain, dflash.logits, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
    }

    @Test func capturesRequestedHiddenStatesInRequestedOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            eval(model)
            let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
            let forward = try model.forwardForDFlash(
                tokens, cache: model.newCache(parameters: nil), targetLayerIds: [3, 1])
            #expect(forward.hiddenStates.count == 2)
            #expect(forward.hiddenStates.allSatisfy { $0.shape == [1, 3, 8] })
            #expect(forward.targetHidden.shape == [1, 3, 16])
            // Captured states are the post-layer residual stream: running the
            // same tokens with ids [1] must reproduce the second slot of [3, 1].
            let single = try model.forwardForDFlash(
                tokens, cache: model.newCache(parameters: nil), targetLayerIds: [1])
            eval(forward.hiddenStates[1], single.hiddenStates[0])
            #expect(
                allClose(forward.hiddenStates[1], single.hiddenStates[0],
                    rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
    }

    @Test func rejectsInvalidLayerIds() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyModel()
            let tokens = MLXArray([Int32(1)])[.newAxis, .ellipsis]
            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens, cache: nil, targetLayerIds: [7])
            }
            #expect(throws: DFlashTargetError.self) {
                _ = try model.forwardForDFlash(
                    tokens, cache: nil, targetLayerIds: [])
            }
        }
    }
}
```

(Make `LagunaModelTests.tinyConfigJSON` `static let` as shown in Task 5 so this file can reuse it.)

- [ ] **Step 2: Run to verify failure**

Run: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/LagunaDFlashForwardTests 2>&1 | tail -30`
Expected: FAIL (no DFlash members on LagunaModel).

- [ ] **Step 3: Implement**

In `Laguna.swift`, add to `LagunaModelInner` a capturing forward (mirror the plain `callAsFunction` exactly — same mask construction — plus capture):

```swift
func callCapturingDFlashHiddenStates(
    _ inputs: MLXArray, cache: [KVCache]?, targetLayerIds: [Int]
) throws -> (postNorm: MLXArray, hiddenStates: [MLXArray]) {
    try DFlashTargetValidation.validateTargetLayerIds(
        targetLayerIds, layerCount: layers.count)
    let targetLayerSet = Set(targetLayerIds)
    var captured = [Int: MLXArray]()

    var h = embedTokens(inputs)
    let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIdx])
    let slidingMask = createAttentionMask(
        h: h, cache: cache?[slidingAttentionIdx], windowSize: slidingWindow)
    for (i, layer) in layers.enumerated() {
        let mask = layerTypes[i] == "full_attention" ? fullMask : slidingMask
        h = layer(h, mask: mask, cache: cache?[i])
        if targetLayerSet.contains(i) {
            captured[i] = h
        }
    }
    return (norm(h), targetLayerIds.map { captured[$0]! })
}
```

Then, in the same file:

```swift
extension LagunaModel: DFlashTargetModel {
    public var dFlashVocabularySize: Int { vocabularySize }
    public var dFlashHiddenSize: Int { config.hiddenSize }
    public var dFlashLayerCount: Int { config.numHiddenLayers }

    public func forwardForDFlash(
        _ inputs: MLXArray, cache: [KVCache]?, targetLayerIds: [Int]
    ) throws -> DFlashTargetForward {
        let (postNorm, hiddenStates) = try model.callCapturingDFlashHiddenStates(
            inputs, cache: cache, targetLayerIds: targetLayerIds)
        return DFlashTargetForward(
            logits: logitsForDFlashHidden(postNorm), hiddenStates: hiddenStates)
    }

    public func embedTokensForDFlash(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func logitsForDFlashHidden(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}
```

- [ ] **Step 4: Run to verify pass**

Same command plus `-only-testing:MLXLMTests/LagunaModelTests`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Libraries/MLXLLM/Models/Laguna.swift Tests/MLXLMTests/LagunaDFlashForwardTests.swift Tests/MLXLMTests/LagunaModelTests.swift
git commit -m "Add DFlashTargetModel conformance to LagunaModel"
```

---

### Task 7: Draft ↔ tiny-Laguna-target integration test

**Files:**
- Test: `Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift` (extend)

**Interfaces:**
- Consumes: everything from Tasks 3–6.
- Produces: proof that `bind` + `draftBlock` work against a real (tiny) `LagunaModel`, exercising the exact call sequence `mlx-bench` uses.

- [ ] **Step 1: Write the test**

Append (needs `@testable import MLXLLM` added to the file's imports):

```swift
@Test func draftBlockRunsAgainstTinyLagunaTarget() throws {
    try Device.withDefaultDevice(.cpu) {
        // Draft with num_target_layers = 4 matching the tiny target's 4 layers.
        let draft = DFlashDraftModel(config: try makeConfig())
        let target = LagunaModel(
            try JSONDecoder.json5().decode(
                LagunaConfiguration.self,
                from: Data(LagunaModelTests.tinyConfigJSON.utf8)))
        try draft.bind(target: target)
        eval(draft, target)

        let prompt = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
        let targetOut = try target.forwardForDFlash(
            prompt,
            cache: target.newCache(parameters: nil),
            targetLayerIds: draft.config.targetLayerIds)
        let drafted = try draft.draftBlock(
            bonus: 4,
            targetHidden: targetOut.targetHidden,
            cache: try draft.makeCache(),
            blockSize: draft.config.recommendedBlockSize)
        eval(drafted)
        #expect(drafted.dim(-1) == draft.config.recommendedBlockSize - 1)
        let tokens = drafted.asArray(Int32.self)
        #expect(tokens.allSatisfy { $0 >= 0 && $0 < 32 })
    }
}
```

- [ ] **Step 2: Run the suite**

Run: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/LagunaDFlashDraftModelTests 2>&1 | tail -30`
Expected: PASS. (Note `recommendedBlockSize` for this config is `min(4, max(2, 8)) = 4`.)

- [ ] **Step 3: Commit**

```bash
git add Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift
git commit -m "Add draft-to-Laguna-target integration test"
```

---

### Task 8: Converted-artifact structural load check + full regression + docs

**Files:**
- Create: `docs/laguna-dflash.md`
- Test: `Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift` (extend — env-gated real-artifact test)

**Interfaces:**
- Consumes: converted artifact at `~/models/laguna-xs-2.1-dflash-mlx` (Task 1), everything from Tasks 2–7.
- Produces: an env-gated structural load test (`MLX_SWIFT_LM_LAGUNA_DFLASH_DIR`), the operator doc, and a green regression run across every DFlash-related suite.

- [ ] **Step 1: Add the env-gated real-artifact test**

```swift
@Suite("Laguna DFlash converted artifact", .serialized)
struct LagunaDFlashConvertedArtifactTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment[
        "MLX_SWIFT_LM_LAGUNA_DFLASH_DIR"] != nil))
    func convertedArtifactLoadsWithFullVerification() async throws {
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment[
            "MLX_SWIFT_LM_LAGUNA_DFLASH_DIR"]!)
        let draft = try await DFlashDraftModel.load(from: dir)
        #expect(draft.config.decoderLayerType == .lagunaXS)
        #expect(draft.config.blockSize == 16)
        #expect(draft.config.targetLayerIds == [1, 13, 25, 33, 39])
        let params = Dictionary(
            uniqueKeysWithValues: draft.parameters().flattened())
        #expect(params.count == 68)
        #expect(params["layers.4.self_attn.g_proj.weight"]?.shape == [64, 2048])
        #expect(params["aux_hidden_norms.4.weight"]?.shape == [2048])
        #expect(params["fc.weight"]?.shape == [2048, 10240])
        #expect(params["layers.0.self_attn.q_proj.weight"]?.dtype == .bfloat16)
    }
}
```

(`DFlashDraftModel.load(from:)` runs `update(parameters:verify: [.all])` — a pass means every one of the 68 tensors matched a module parameter exactly. ~1 GB RAM; fine locally.)

- [ ] **Step 2: Run it against the real artifact**

Run: `MLX_SWIFT_LM_LAGUNA_DFLASH_DIR=~/models/laguna-xs-2.1-dflash-mlx xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/LagunaDFlashConvertedArtifactTests 2>&1 | tail -30`
Expected: PASS. (xcodebuild may not inherit `~` expansion — use the absolute `/Users/asv/models/...` path.)

- [ ] **Step 3: Full DFlash regression**

Run: `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/DFlashConfigurationTests -only-testing:MLXLMTests/DFlashDraftModelTests -only-testing:MLXLMTests/DFlashTokenIteratorTests -only-testing:MLXLMTests/DFlashGenerateTests -only-testing:MLXLMTests/QwenDFlashForwardTests -only-testing:MLXLMTests/QwenMoEDFlashForwardTests -only-testing:MLXLMTests/QwenHybridDFlashForwardTests -only-testing:MLXLMTests/GPTOSSDFlashForwardTests -only-testing:MLXLMTests/Gemma4DFlashForwardTests -only-testing:MLXLMTests/LagunaDFlashDraftModelTests -only-testing:MLXLMTests/LagunaModelTests -only-testing:MLXLMTests/LagunaDFlashForwardTests 2>&1 | tail -40`
Expected: ALL PASS.

- [ ] **Step 4: Write `docs/laguna-dflash.md`**

Contents (concise, operator-facing): what the artifact is; conversion rerun commands (Task 1 Step 3 verbatim); the converted config schema and why (`decoder_layer_type: laguna_xs` switch); how to run `mlx-bench` dflash for Laguna:

```bash
swift build -c release
scripts/build-mlx-metallib.sh   # required: plain swift build does not produce mlx.metallib
.build/release/mlx-bench dflash \
  --target /path/to/laguna-xs-2.1-nvfp4-mlx \
  --drafter ~/models/laguna-xs-2.1-dflash-mlx
```

plus the note that `--target` must be the **MLX-format** Laguna checkpoint (mlx-style `quantization` config + `.scales` tensors — e.g. the mlxfast-challenge reference checkpoint), NOT the raw poolside HF NVFP4 repo; the env-gated tests (`MLX_SWIFT_LM_LAGUNA_DFLASH_DIR`, `MLX_SWIFT_LM_DFLASH_TARGET_DIR`/`MLX_SWIFT_LM_DFLASH_DRAFTER_DIR` for `DFlashRealCheckpointSmokeTests`); and the deferred HF-upload step (org/name TBD by operator).

- [ ] **Step 5: Commit**

```bash
git add docs/laguna-dflash.md Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift
git commit -m "Add converted-artifact load test and Laguna DFlash docs"
```

---

### Task 9: E2E validation on M5-B (OPERATOR-GATED)

**HOLD POINT: do not start until the operator confirms a quiet window — M5-B serves the live ranked pipeline; a 22 GB model load plus GPU work can thermally delay ranked jobs.**

**Files:**
- Modify: `docs/laguna-dflash.md` (record measured numbers)

**Interfaces:**
- Consumes: all prior tasks; M5-B access (`ssh -i ~/.ssh/mtp_bench gaj@m5-max-128gb-2.tail618116.ts.net`); an MLX-format Laguna XS 2.1 NVFP4 reference checkpoint on the box (operator-provided path — the mlxfast-challenge reference; ask the operator, do NOT touch `/opt/bench*` or `/Users/Shared/bench-jobs/*`).
- Produces: the three e2e acceptance results from the spec, recorded in `docs/laguna-dflash.md`.

- [ ] **Step 1: Stage the fork on M5-B**

```bash
rsync -a --exclude .build --exclude .git --exclude .serena \
  /Users/asv/projects/MLX-swift-lm-dflash/ \
  gaj@m5-max-128gb-2.tail618116.ts.net:~/projects/laguna-dflash-e2e/
ssh -i ~/.ssh/mtp_bench gaj@m5-max-128gb-2.tail618116.ts.net
cd ~/projects/laguna-dflash-e2e
swift build -c release 2>&1 | tail -5
scripts/build-mlx-metallib.sh
```

(Box gotcha from memory: if git/SPM network access misbehaves, prefix with `GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null` — the box's global git config rewrites https→ssh.)

- [ ] **Step 2: Convert the draft on the box**

Run Task 1 Step 3's download+convert commands on the box (output `~/models/laguna-xs-2.1-dflash-mlx`).

- [ ] **Step 3: Locate the target checkpoint**

Ask the operator for the MLX-format reference checkpoint path on M5-B (or copy one in). Verify it looks right: `config.json` has `"model_type": "laguna"` and a `"quantization"` block; shards total ~21.6 GB.

- [ ] **Step 4: Run the e2e criteria**

```bash
# (a) Greedy equivalence — the bench's own parity check:
.build/release/mlx-bench dflash --target <TARGET_DIR> --drafter ~/models/laguna-xs-2.1-dflash-mlx --parity 2>&1 | tail -20
# (b)+(c) Acceptance + speedup:
.build/release/mlx-bench dflash --target <TARGET_DIR> --drafter ~/models/laguna-xs-2.1-dflash-mlx 2>&1 | tail -40
```

Before running, check `.build/release/mlx-bench dflash --help` (or `Sources/mlx-bench/BenchCommands.swift`) for the exact parity/prompt/step flags — use the same flags the Gemma4 runs use. Pass criteria (spec): (a) parity check reports zero mismatched tokens; (b) mean accepted tokens per 16-block ≥ 4.0 on the bench's coding-style prompts; (c) wall-clock speedup vs plain decode > 1, recorded.

- [ ] **Step 5: Record results, clean up the box, commit**

Append a "Measured results (M5-B, date)" section to `docs/laguna-dflash.md` with the three numbers and exact invocations. Remove `~/projects/laguna-dflash-e2e` and the downloaded models from the box unless the operator wants them kept. Commit:

```bash
git add docs/laguna-dflash.md
git commit -m "Record Laguna DFlash e2e validation results from M5-B"
```

---

## Self-review notes (already applied)

- Spec coverage: conversion (T1), config (T2), draft behaviors (T3+T4), target port (T5), conformance (T6), integration (T7), artifact check + docs + regression (T8), e2e (T9). Deferred spec items (HF upload, challenge-repo work, optimized rollback/sequential verify) are documented in T8's doc, not implemented — matching the spec's Deferred section.
- The `max_anchors` open item from the spec: resolved — the fork's context handling (trim to `sliding_window − 1`) is the operative constraint; `max_anchors` is a speculators training-side field not read by any inference reference; no code change needed. T9's acceptance measurement is the empirical backstop.
- Type consistency: `DFlashDecoderLayerType.lagunaXS` rawValue `"laguna_xs"`; `LagunaModelTests.tinyConfigJSON` is `static let` and reused by T6/T7; stub target name `LagunaDraftStubTarget` used in T3/T4.
