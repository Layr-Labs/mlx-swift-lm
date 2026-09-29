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

Each new test in `Tests/MLXLMTests` has one type. The type is a Swift Testing
tag from `Tests/MLXLMTests/TestTypeTags.swift`. The older test files are not in
these folders and have no type tag yet.

| Type | What it may use | Where it lives | How CI runs it |
|---|---|---|---|
| `unit` | Swift logic only. No MLX arrays (also not on the CPU device), no GPU, no Metal library, no model weights, no network, no wall-clock timing | `Tests/MLXLMTests/Unit/<Area>/`, each suite in `extension UnitTests { ... }` | The step "Unit tests (no GPU)" runs them before the Metal library is built. The whole-package pass runs them again for coverage. |
| `kernel` | The Metal device and the Metal library | `Tests/MLXLMTests/Kernel/<Area>/`, tag `.kernel` | The whole-package pass, after the Metal library is staged. |
| `integration` | An engine or a model with more than one component, with tiny test models | `Tests/MLXLMTests/Integration/<Area>/`, tag `.integration` | The whole-package pass. |
| `reference` | Bit-exact comparison with frozen references | Next to the other tests of the area, tag `.reference` and trait `.referenceHardware` | Skipped in CI. Runs only with `MLX_REFERENCE_HARDWARE=1`. |

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

SwiftPM compiles every Swift file under the folder of a test target, also the
files in subfolders, so a new folder needs no change to `Package.swift`.

The Integration tests below are a different thing: an Xcode project that
downloads models.

Integration tests verify end-to-end model loading and generation. They require
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
