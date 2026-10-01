# Contributing to MLX Swift Examples

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo. 
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests (if any) and at least one review. 
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   If needed you may need to `brew install swift-format`.
 
   You can also run the formatters manually as follows:
 
     ```
     swift-format format --in-place --recursive Libraries Tools Applications IntegrationTesting
     ```
 
   or run `pre-commit run --all-files` to check all files in the repo.
 
## Running Tests

Unit tests run without any special hardware and do not download models.
Note: `swift test` [does not work yet](https://github.com/ml-explore/mlx-swift?tab=readme-ov-file#xcodebuild) — use `xcodebuild` instead:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS'
```

Some tests compare the raw bits of floating-point results with frozen
references. Other hardware can give results that differ in the last bits. The
comparisons that differ on the GitHub-hosted runner are separate tests, marked
with the trait `.referenceHardware` in
`Tests/MLXLMTests/ReferenceHardware.swift`. They are skipped unless
`MLX_REFERENCE_HARDWARE=1` is set. Set it only on a Mac of the kind that
recorded the references. The other checks of those tests run everywhere. Some
bit-exact tests, such as the bf16 and video tests, have no trait because they
pass on the hosted runner.

### Test types

Each new test has one type. The type is a Swift Testing tag from the
`TestTypeTags.swift` file of its test target, for example
`Tests/MLXLMTests/TestTypeTags.swift`.

| Type | What it may use | Where it lives | How CI runs it |
|---|---|---|---|
| `unit` | Swift logic only. No MLX arrays (also not on the CPU device), no GPU, no Metal library, no model weights, no network, no wall-clock timing | `Tests/<Target>/Unit/<Area>/`, each suite in `extension UnitTests { ... }` | The step "Unit tests (no GPU)" runs them before the Metal library is built. The whole-package pass runs them again for coverage. |
| `kernel` | The Metal device and the Metal library | `Tests/<Target>/Kernel/<Area>/`, tag `.kernel` | The whole-package pass, after the Metal library is staged. |
| `integration` | An engine or a model with more than one component, with tiny test models | `Tests/<Target>/Integration/<Area>/`, tag `.integration` | The whole-package pass. |
| `reference` | Bit-exact comparison with frozen references | Next to the other tests of the area, tag `.reference` and trait `.referenceHardware` | Skipped in CI. Runs only with `MLX_REFERENCE_HARDWARE=1`. |

Test files outside these folders have no type tag.

The `UnitTests` suite gives the tag `unit` to every suite in it. `swift test`
cannot select tests by tag, so CI selects the unit tests by the name of this
suite. Another test target that gets unit tests declares the same tags and the
same `UnitTests` suite in its own `TestTypeTags.swift`, and keeps its unit
tests in its own `Unit/<Area>/` folder. This command runs the unit tests of
all test targets:

```bash
swift test --filter '\.UnitTests/'
```

A `unit` test does not use MLX arrays. On macOS, MLX allocates each array,
also an array on the CPU device, through the Metal allocator. The allocator
loads the Metal library, and without the library the test process stops with
"Failed to load the default metallib".

`scripts/check-unit-test-files.sh` fails when a Swift file under a
`Tests/<Target>/Unit/` folder has no `extension UnitTests`, and when the tag
declarations of a `TestTypeTags.swift` file are not the same as in
`Tests/MLXLMTests/TestTypeTags.swift`. The unit test step runs it.

SwiftPM compiles every Swift file under the folder of a test target, also the
files in subfolders, so a new folder needs no change to `Package.swift`.

#### Model configuration cases

`Tests/MLXLMTests/Unit/Models/` decodes each model configuration type from a
JSON. To add a case for a new configuration type, write it by hand:

1. Read `CodingKeys` and `init(from:)` of the type. The required keys are the
   keys that `init(from:)` reads with `decode`, not `decodeIfPresent`.
2. Add a case to `cases` in `LLMConfigurationDecodingTests.swift` or
   `VLMConfigurationDecodingTests.swift`. Its JSON holds only the required
   keys. Give each key a different value: different numbers, true and false
   in turn, different strings, and one element in an array. `requiredKeys`
   lists these keys as paths joined with ".".
3. Write `fields`: one `path=value` line for each stored property, as
   `UnitTests.storedFields(of:)` prints it. Take the values from the JSON and
   the defaults from `init(from:)`.
4. For an important type, also add a case to `LLMFullKeyCases.swift` or
   `VLMFullKeyCases.swift` with a JSON that sets every key in `CodingKeys` to a
   value that is not the default.
5. Run the tests. When a line is different, read the decoder to find out
   whether the test or the decoder is wrong. Do not copy the decoder output
   into the test without that check.

#### Kernel tests

The `kernel` tests of `MLXLMTests` are in the `KernelTests` suite, which
`Tests/MLXLMTests/Kernel/Support/KernelTests.swift` declares with the tag
`kernel`. Declare each suite under `Kernel/<Area>/` in an extension of
`KernelTests`. This command runs them:

```bash
swift test --filter '\.KernelTests/'
```

| Folder | What it tests |
|---|---|
| `Kernel/Support/` | The `KernelTests` suite, the shared helper `SyntheticModel.swift`, and `ModelCase.swift`, which runs the same checks on a table of models |
| `Kernel/LLM/` | The forward pass of tiny language models with random weights |
| `Kernel/Cache/` | The KV caches, the attention masks and the quantized attention |
| `Kernel/RoPE/` | The RoPE layers and their scaling types |
| `Kernel/Vision/` | Vision-language models with small synthetic images, and the interpolation kernels |
| `Kernel/Embedders/` | The embedding models and the pooling strategies |
| `Kernel/Adapters/` | The LoRA and DoRA layers, `LoRAContainer` and the adapter factory |

`SyntheticModel` builds a tiny model from a configuration dictionary, gives
it seeded random weights and evaluates them before use. It also loads a
synthetic checkpoint through `loadWeights`. `ForwardPassChecks` checks a
model: the logits shape, dtype and finite values, determinism, cache
consistency, batch invariance and causality. Cache consistency means that a
prompt in chunks and decode steps with the cache give the same logits as one
pass without a cache. `ModelCase` runs these checks and a checkpoint load on
each model of a table. A check that fails because of a known production
defect runs inside `withKnownIssue`, with the defect named. Each test states
its tolerance and the reason for it.
The tests use no real weights and compare no frozen reference values.

The float32 tolerances assume full float32 matrix products. On a Mac with an
M5 GPU, MLX uses TF32 for float32 matrix products by default, and the cache
tests fail. Set `MLX_ENABLE_TF32=0` to run them on such a Mac.

The `integration` type is not the same as the Xcode integration tests in the
next section.

### Xcode integration tests

Xcode integration tests verify end-to-end model loading and generation. They require
macOS with Metal and download models from Hugging Face Hub on first run. These
tests do not run in CI.

Open `IntegrationTesting/IntegrationTesting.xcodeproj` in Xcode and run the
test target (`Cmd+U` or via the Test Navigator), or use `xcodebuild`:

```bash
# Run all integration tests
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS'

# Run a single test
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -only-testing:IntegrationTestingTests/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

See [Libraries/IntegrationTestHelpers/README.md](Libraries/IntegrationTestHelpers/README.md) for more details.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift Examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
