# Qwen3-VL 30B-A3B Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strict-load and run `lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit` through the public `MLXVLM` APIs.

**Architecture:** Extend the existing Qwen3-VL language stack with a construction-time dense-or-MoE module choice. Preserve the existing vision topology, DeepStack, multimodal RoPE, cache, and processor paths; install a split generic `SwitchGLU` only for config-declared MoE layers and select the checkpoint's vision GELU profile once during construction.

**Tech Stack:** Swift 6.1, MLX Swift, MLXNN, MLXLMCommon, XCTest, Hugging Face safetensors, Python `mlx-vlm` reference.

**Assumptions:** Assumes Hugging Face revision `61c11f42d7bc01e00f5ea7f2e667c0a216f48397` remains the requested artifact — this plan will not silently substitute another Qwen3-VL quantization. Assumes the four real shard headers are authoritative — it will not implement the stale 13-shard paths listed by `model.safetensors.index.json`.

---

## File structure

- `Libraries/MLXVLM/Models/Qwen3VL.swift`: MoE config, construction-time validation, sparse block, and immutable layer selection.
- `Libraries/MLXVLM/VLMModelFactory.swift`: model-type registration and exact checkpoint preset.
- `Libraries/MLXVLM/README.md`: supported model and architecture listing.
- `Tests/MLXLMTests/Qwen3VLMoETests.swift`: config, structure, arithmetic,
  activation, and registry regression coverage.
- `benchmarks/qwen3-vl-30b-a3b/`: checked-in commands and compact generation receipts; downloaded weights remain outside Git.

### Task 1: Lock the model and artifact contracts with failing tests

**Files:**
- Create: `Tests/MLXLMTests/Qwen3VLMoETests.swift`
- Test: `Libraries/MLXVLM/Models/Qwen3VL.swift`
- Test: `Libraries/MLXVLM/VLMModelFactory.swift`

**Security flag:** none

**Does NOT cover:** Performance-specialized routing or expert kernels; the tests require the generic reference arithmetic and the published split checkpoint layout.

- [ ] **Step 1: Add a real-shape config fixture and failing decode assertions**

```swift
let config = try JSONDecoder().decode(
    Qwen3VLConfiguration.self, from: Data(qwen3VLMoEConfigJSON.utf8))
XCTAssertEqual(config.modelType, "qwen3_vl_moe")
XCTAssertEqual(config.textConfiguration.numExperts, 128)
XCTAssertEqual(config.textConfiguration.numExpertsPerToken, 8)
XCTAssertEqual(config.textConfiguration.moeIntermediateSize, 768)
XCTAssertTrue(config.textConfiguration.normalizesTopKProbabilities)
```

- [ ] **Step 2: Add failing construction and registry tests**

```swift
let model = Qwen3VLLanguage.Model(config.textConfiguration)
XCTAssertTrue(model.layers.allSatisfy { $0.mlp is Qwen3VLLanguage.SparseMoeBlock })
XCTAssertNoThrow(try VLMTypeRegistry.shared.createModel(configuration: configData))
```

Also assert a one-layer dense fixture still installs `Qwen3VLLanguage.MLP`,
and assert the MoE parameter tree contains
`layers.0.mlp.switch_mlp.gate_proj.weight`, `up_proj.weight`, and
`down_proj.weight` while excluding `shared_expert` and `gate_up_proj`.
Assert that MoE vision blocks install tanh-approximate GELU while patch and
DeepStack mergers retain exact GELU, and that dense Qwen3-VL retains its
existing Swift activation profile.

- [ ] **Step 3: Run the focused tests and observe the expected failures**

Run:

```bash
swift test --scratch-path /Users/davidtai/projects/OpenSourceWTF/mlx-swift-lm/.build --filter Qwen3VLMoETests
```

Expected: compilation or assertions fail because the config fields,
`qwen3_vl_moe` registry entry, and sparse module do not exist.

### Task 2: Implement exact construction and routed arithmetic

**Files:**
- Modify: `Libraries/MLXVLM/Models/Qwen3VL.swift`
- Modify: `Tests/MLXLMTests/Qwen3VLMoETests.swift`

**Security flag:** none

**Does NOT cover:** Qwen3.6 shared experts, fused gate/up tensors, its
256-expert router finalizer, MTP, or an environment-selected execution lane.

- [ ] **Step 1: Decode the optional MoE schema with dense defaults**

Add coding keys for `num_experts`, `num_experts_per_tok`,
`decoder_sparse_step`, `mlp_only_layers`, `moe_intermediate_size`, and
`norm_topk_prob`. Provide dense-safe computed defaults and a
`usesSparseMoE(layerIndex:)` method.

- [ ] **Step 2: Install the target-shaped sparse block**

```swift
final class SparseMoeBlock: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let probabilities = softmax(gate(x), axis: -1, precise: true)
        let kth = probabilities.dim(-1) - topK
        let indices = argPartition(probabilities, kth: kth, axis: -1)[.ellipsis, kth...]
        var scores = takeAlong(probabilities, indices, axis: -1)
        if normalizesTopKProbabilities {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        let expertOutput = switchMLP(x, indices)
        return (expertOutput * expandedDimensions(scores, axis: -1)).sum(axis: -2)
    }
}
```

Construct `SwitchGLU(inputDims:hiddenDims:numExperts:)` with its generic,
split defaults. Change `DecoderLayer.mlp` to a construction-time `Module` and
invoke it as `UnaryLayer`, matching the established Qwen3.5 VLM pattern.

- [ ] **Step 3: Add an independent small-tensor arithmetic test**

Set deterministic router and expert weights, evaluate the sparse block, then
independently compute full softmax, top-k selection, selected expert SwiGLU,
and weighted sum without calling `SparseMoeBlock.callAsFunction`. Require
matching selected IDs and a maximum absolute output error below `2e-5`.

- [ ] **Step 4: Install the checkpoint-specific vision activation profile**

Resolve the vision-block GELU at construction from `vision_config.model_type`.
Use `.tanh` for `qwen3_vl_moe`, retain the existing `.fast` dense profile, and
leave patch/deepstack mergers on exact GELU. Tests must prove both routes.

- [ ] **Step 5: Run focused tests to green**

Run:

```bash
swift test --scratch-path /Users/davidtai/projects/OpenSourceWTF/mlx-swift-lm/.build --filter Qwen3VLMoETests
```

Expected: every `Qwen3VLMoETests` case passes.

### Task 3: Publish the supported surface and prove strict real loading

**Files:**
- Modify: `Libraries/MLXVLM/VLMModelFactory.swift`
- Modify: `Libraries/MLXVLM/README.md`
- Create: `benchmarks/qwen3-vl-30b-a3b/README.md`
- Create: `benchmarks/qwen3-vl-30b-a3b/text-generation.json`
- Create: `benchmarks/qwen3-vl-30b-a3b/image-generation.json`

**Security flag:** none

**Does NOT cover:** Hosting or repairing the external Hugging Face model
repository; the receipts record the pinned revision and actual four shards.

- [ ] **Step 1: Add registry and documentation entries**

Register:

```swift
"qwen3_vl_moe": create(Qwen3VLConfiguration.self, Qwen3VL.init)
```

Add `qwen3VL30BA3BInstruct4Bit` with the exact LM Studio identifier to
`VLMRegistry.all()`, and list `qwen3_vl_moe` plus the checkpoint in the VLM
README.

- [ ] **Step 2: Download the pinned artifact outside Git**

Run the Hugging Face snapshot downloader for revision
`61c11f42d7bc01e00f5ea7f2e667c0a216f48397`, including JSON/Jinja/tokenizer
files and the four `model-*-of-00004.safetensors` shards. Record shard names,
byte sizes, and SHA-256 values in the benchmark README.

- [ ] **Step 3: Strict-load the real checkpoint**

Build and run a temporary validation executable against the local snapshot.
It must load through `VLMModelFactory`, request strict `.all` weight-update
verification, and then generate through the public VLM interfaces. The
temporary target is removed before the PR is committed.

Run the equivalent of:

```bash
swift run --scratch-path /Users/davidtai/projects/OpenSourceWTF/mlx-swift-lm/.build Qwen3VLPortCheck <snapshot-path> both 8 <image-path>
```

Expected: exit 0, model type `qwen3_vl_moe` resolves, all weights update with
strict `.all` verification, and both generation cases complete.

- [ ] **Step 4: Compare greedy text and image generation with Python**

Use the same local snapshot, zero temperature, fixed prompts, fixed generated
token count, and a committed small image fixture or byte-identical generated
PNG. Run Python `mlx-vlm` and the Swift VLM interface. For image comparison,
correct `mlx-vlm 0.6.16`'s reversed Qwen3-VL-MoE vision GELU installation to
the checkpoint/original-Hugging-Face contract before generation. Store model
revision, prompt/media SHA-256, output token IDs, decoded output, prefill time,
decode time, peak memory, and command lines in the JSON receipts.

Expected: Swift and Python produce the same greedy token IDs for both text and
image cases. Any divergence requires paired-logit localization before the port
can be called verified. Video continues to use the existing Qwen3-VL processor
path but is outside this parity gate.

### Task 4: Regression verification and pull-request delivery

**Files:**
- Inspect: every changed file
- Inspect: `docs/specs/2026-08-25-qwen3-vl-30b-a3b-design.md`
- Inspect: `docs/plans/2026-08-25-qwen3-vl-30b-a3b-port.md`

**Security flag:** none

- [ ] **Step 1: Run focused and full verification**

```bash
swift test --scratch-path /Users/davidtai/projects/OpenSourceWTF/mlx-swift-lm/.build --filter Qwen3VLMoETests
swift build --scratch-path /Users/davidtai/projects/OpenSourceWTF/mlx-swift-lm/.build --build-tests
```

Expected: exit 0 with no test failures or compiler errors.

- [ ] **Step 2: Run implementation hygiene and diff audits**

```bash
rg -n '[T]ODO|[F]IXME|[p]laceholder|[N]otImplementedError' Libraries/MLXVLM Tests/MLXLMTests/Qwen3VLMoETests.swift
git diff --check
git status --short
```

Expected: no new stub markers, no whitespace errors, and only scoped port,
test, design, documentation, and receipt files changed.

- [ ] **Step 3: Commit, push, and open the requested PR**

Commit the verified implementation, push `feat/qwen3-vl-30b-a3b` to the
authenticated fork, and open a PR against `Layr-Labs/mlx-swift-lm:main`. The
PR body must include the exact checkpoint revision, architecture boundary,
focused/full test commands, strict-load digest, text/image parity receipts,
and the explicit decision not to transplant Qwen3.6-specific kernels.

- [ ] **Step 4: Verify remote delivery**

Use `gh pr view` to confirm the PR URL, base/head repositories and branches,
commit SHA, title, body, and current checks. Do not report delivery from local
commits alone.
