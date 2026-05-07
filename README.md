# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

> [!IMPORTANT]
> The `main` branch is a _new_ major version number: 3.x.  In order
> to decouple from tokenizer and downloader packages some breaking
> changes were introduced. See [upgrading documentation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/upgrade) for detailed instructions on upgrading.
>
> If that page shows a 404 you can view the source:
> [upgrading](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/upgrade.md) 
> and [using](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md)

Some key features include:

- Model loading with integrations for a variety of tokenizer and model downloading packages.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM, check out [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [Techniques for developing in mlx-swift-lm](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/developing)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations

## Usage

This package integrates with a variety of tokenizer and downloader packages through protocol conformance. Users can pick from three ways to integrate with these packages, which offer different tradeoffs between freedom and convenience.

See documentation on [how to integrate mlx-swift-lm and downloaders/tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).

## Installation

Add the core package to your `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
```

Then chose an [integration package for downloaders and tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).


## Quick Start

After installing the package you can use LLMs to generate content with only a few lines
of code.  (Note: the exact line to load the model depends on the [integration package](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages)).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).


```swift
import MLXLLM
import MLXLMCommon

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit

// customize this line per the integration package
let model = try await loadModelContainer(
    configuration: modelConfiguration
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

## Gemma 4 MTP Benchmarks

Gemma 4 Multi-Token Prediction speculative decoding performance depends on
hardware, output length, batch size, model family, and draft block size. The
following run uses longer generations than short smoke benchmarks:

- Hardware: Apple M5 Max, 128 GB unified memory
- Batch size: 1
- Decoding: greedy, `temperature = 0`
- Prompts: 3 local prompt fixtures
- Output budget: `max_tokens = 512`
- Warmup: `64` generated tokens
- Throughput: generated-token throughput, excluding prompt prefill

| Model | Draft block size | Baseline tok/s | MTP tok/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B BF16 | 3 | 89.5 | 114.3 | 1.28x |
| Gemma 4 E2B BF16 | 4 | 89.5 | 132.3 | 1.48x |
| Gemma 4 E2B BF16 | 5 | 89.9 | 144.2 | 1.60x |
| Gemma 4 E4B BF16 | 3 | 46.9 | 43.5 | 0.93x |
| Gemma 4 E4B BF16 | 4 | 46.9 | 44.3 | 0.94x |
| Gemma 4 26B-A4B 4-bit | 3 | 109.6 | 122.4 | 1.12x |
| Gemma 4 26B-A4B 4-bit | 4 | 109.6 | 118.6 | 1.08x |

These numbers are single-machine measurements and can move with system load.
In this run, longer outputs amortize the MTP overhead for E2B and 26B-A4B. E4B
BF16 remains slightly below baseline for B=1 on this hardware.
