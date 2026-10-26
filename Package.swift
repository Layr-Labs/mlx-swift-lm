// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
    name: "mlx-swift-lm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MLXVLM",
            targets: ["MLXVLM"]),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
        .library(
            name: "MLXEmbedders",
            targets: ["MLXEmbedders"]),
        .library(
            name: "MLXRerankers",
            targets: ["MLXRerankers"]),
        .library(
            name: "MLXHuggingFace",
            targets: ["MLXHuggingFace"]),
        .library(
            name: "MLXLMServer",
            targets: ["MLXLMServer"]),
        .executable(
            name: "mlx-server",
            targets: ["mlx-server"]),
        .library(
            name: "MLXFoundationModels",
            targets: ["MLXFoundationModels"]),
        .library(
            name: "MLXGuidedGeneration",
            targets: ["MLXGuidedGeneration"]),
        .library(
            name: "BenchmarkHelpers",
            targets: ["BenchmarkHelpers"]),
        .library(
            name: "IntegrationTestHelpers",
            targets: ["IntegrationTestHelpers"]),
        .library(
            name: "MLXScriptedLM",
            targets: ["MLXScriptedLM"]),
    ],
    traits: [
        // Gates the MLXLanguageModel adapter for Apple's FoundationModels
        // framework. Default-on. Disabling the trait compiles MLXFoundationModels
        // to an empty library: the entire `MLXLanguageModel` / `MLXLanguageModel.Executor`
        // surface requires FoundationModels types that are not available on platforms
        // older than iOS/macOS/visionOS 27.0, and the MLXDownloadProgress observable
        // (whose only producer is that adapter) is gated alongside it. Consumers
        // targeting older OS versions can still use this package for MLXLLM /
        // MLXLMCommon / MLXEmbedders etc. by turning the trait off.
        .trait(
            name: "FoundationModelsIntegration",
            description:
                "Enables the MLXLanguageModel adapter for Apple's FoundationModels framework. Disabling removes the MLXLanguageModel / MLXLanguageModel.Executor types."
        ),
        .default(enabledTraits: ["FoundationModelsIntegration"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/Layr-Labs/mlx-swift.git",
            revision: "3832c0de953eefb1084412caf8f50e180f59d8e8"),
        // 602.0.0 floor: swift.org publishes signed prebuilt swift-syntax artifacts only for
        // >= 602 tags on current toolchains; a 600.x/601.x resolution falls back to the full
        // source compile of swift-syntax.
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "602.0.0" ..< "604.0.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.23.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.2"),
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.3.6"),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                .product(name: "Jinja", package: "swift-jinja"),
                "MLXLMCommon",
                "MLXLLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // Cmlx: low-level mlx_slice_dynamic / mlx_slice_update_dynamic
                // entry points used by DynamicSlice.swift (compiled-decode infra).
                .product(name: "Cmlx", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
            ],
            resources: [
                // CBv2 paged-attention MSL source, JIT-compiled at runtime
                // via MLXFast.metalKernel (NOT compiled by SwiftPM).
                .copy("ContinuousBatchingV2/Paged/pagedattention.metal"),
                // Exact MLX preambles used by native Qwen4 JIT kernels.
                // Copy as text; SwiftPM must not compile these header fragments.
                .copy("Resources/Qwen4Metal"),
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/MLXEmbedders",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMServer",
            dependencies: [
                "MLXLLM",
                "MLXVLM",
                "MLXLMCommon",
                "MLXEmbedders",
                "MLXHuggingFace",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Libraries/MLXLMServer"
        ),
        .executableTarget(
            name: "mlx-server",
            dependencies: ["MLXLMServer"],
            path: "Executables/mlx-server"
        ),
        .target(
            name: "MLXRerankers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXEmbedders",
            ],
            path: "Libraries/MLXRerankers"
        ),
        .target(
            name: "BenchmarkHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/BenchmarkHelpers"
        ),
        .target(
            name: "IntegrationTestHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXRerankers",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/IntegrationTestHelpers",
            exclude: ["README.md"]
        ),

        // Scripted tokenizers and models
        .target(
            name: "MLXScriptedLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/MLXScriptedLM",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                "MLXHuggingFace",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXRerankers",
                "MLXScriptedLM",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md",
                // Stale VLM MTP spike: references Gemma4 (MLXVLM) MTP API
                // removed by the vMLX decode port (#55). Broken at the
                // engine-v2 branch base and blocks the whole test target;
                // excluded until the VLM MTP spike is updated or deleted.
                "Gemma4VLMMTPSpikeTests.swift",
            ],
            resources: [
                .process("Resources/1080p_30.mov"),
                .copy("Resources/MiMoOpenRouter"),
                .process("Resources/audio_only.mov"),
                .process("Resources/flint-oracle.json"),
                .process("Resources/Gemma4MTPPrompts.json"),
                .process("Resources/diffusiongemma-text-config.json"),
                .process("Resources/diffusiongemma-root-config.json"),
                .process("Resources/diffusiongemma-text-oracle.json"),
                .process("Resources/diffusiongemma-text-oracle.safetensors"),
                .process("Resources/diffusiongemma-chunked-text-oracle.safetensors"),
                .process("Resources/diffusiongemma-vision-oracle.json"),
                .process("Resources/diffusiongemma-vision-oracle.safetensors"),
                .process("Resources/diffusiongemma-vision-bf-oracle.json"),
                .process("Resources/diffusiongemma-vision-bf-oracle.safetensors"),
                .process("Resources/diffusiongemma-video-types-oracle.json"),
                .process("Resources/diffusiongemma-video-types-oracle.safetensors"),
                .process("Resources/diffusiongemma-media-geometry.json"),
                .process("Resources/diffusiongemma-legacy-vision.safetensors"),
                .process("Resources/diffusiongemma-block-oracle.json"),
                .process("Resources/diffusiongemma-block-oracle.safetensors"),
                .process("Resources/gemma4-26B-A4B-assistant-config.json"),
                .process("Resources/gemma4-E4B-assistant-config.json"),
                .process("Resources/mtp-oracle/gemma4-e2b-block3-max64.json"),
                .process("Resources/block_hash_vectors.json"),
            ]
        ),
        .testTarget(
            name: "OnboardingQualificationTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                "MLXLMCommon",
                "MLXLLM",
            ],
            path: "Tests/OnboardingQualificationTests"
        ),
        .testTarget(
            name: "MLXLMServerTests",
            dependencies: [
                "MLXLMServer",
                "MLXLMCommon",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
            ],
            path: "Tests/MLXLMServerTests"
        ),
        .macro(
            name: "MLXHuggingFaceMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "Libraries/MLXHuggingFaceMacros"
        ),
        .target(
            name: "MLXHuggingFace",
            dependencies: [
                "MLXHuggingFaceMacros",
                "MLXLMCommon",
                .target(
                    name: "MLXFoundationModels",
                    condition: .when(traits: ["FoundationModelsIntegration"])
                ),
            ],
            path: "Libraries/MLXHuggingFace"
        ),
        .executableTarget(
            name: "BenchSegmentedDecode",
            dependencies: ["MLXLMCommon"],
            path: "Sources/BenchSegmentedDecode"
        ),
        .executableTarget(
            name: "BenchLoad",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "BenchmarkHelpers",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/BenchLoad"
        ),
        // WS-G (engine-v2) benchmark harness: tiny-model v2-style step loop
        // (CI-runnable) + `--model <path>` legacy-engine baseline for real
        // weights. Reports land as markdown next to the other benchmarks/
        // reports. `sources:` keeps the target scoped to the one Swift file.
        .executableTarget(
            name: "CBv2Benchmark",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "benchmarks",
            exclude: [
                "mtp-2026-05-09-130614.md",
                "mtp-2026-05-09-142113.md",
                "mtp-2026-05-09-183133.md",
                "mtp-2026-05-09-190948.md",
                "mtp-2026-05-09-203007.md",
            ],
            sources: ["CBv2Benchmark.swift"]
        ),
        // Build-time git-revision stamp for BenchCBv2. The bench report must
        // name the revision the binary was compiled from, not whatever HEAD
        // is when the report is written. Attached to BenchCBv2Core, which is
        // where `buildRevision()` reads the generated constant.
        .plugin(
            name: "BenchRevisionStamp",
            capability: .buildTool(),
            path: "Plugins/BenchRevisionStamp"
        ),
        // Real-weights validation driver for the CBv2 engine: correctness
        // smoke (batch-composition + chunked-prefill invariance) and the
        // v2-vs-v2-paged perf matrix on local model directories.
        //
        // A LIBRARY, with `BenchCBv2` below reduced to a `@main` shim over
        // `BenchCBv2Driver.run()`. The split is load-bearing: when a test
        // target depends on an executable target, SwiftPM runs that binary as
        // the swift-testing host and hands it `--test-bundle-path`, which this
        // driver's strict option parser rejects — aborting the swift-testing
        // pass for the whole package. Every `@Test` in mlx-swift-lm, including
        // the paged-KV CI gates, executed nothing while BenchCBv2Tests
        // depended on the executable. Nothing may depend on `BenchCBv2`.
        .target(
            name: "BenchCBv2Core",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXHuggingFace",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/BenchCBv2Core",
            plugins: ["BenchRevisionStamp"]
        ),
        .executableTarget(
            name: "BenchCBv2",
            dependencies: ["BenchCBv2Core"],
            path: "Sources/BenchCBv2"
        ),
        // Harness-integrity tests: option parsing, engine resolution, and
        // report/optimization provenance. Model-free, so they run in CI.
        .testTarget(
            name: "BenchCBv2Tests",
            dependencies: ["BenchCBv2Core"],
            path: "Tests/BenchCBv2Tests"
        ),
        .testTarget(
            name: "MLXHuggingFaceMacrosTests",
            dependencies: [
                "MLXHuggingFaceMacros",
                .product(name: "SwiftSyntaxMacrosTestSupport", package: "swift-syntax"),
            ],
            path: "Tests/MLXHuggingFaceMacrosTests"
        ),
        // C++ bridge for xgrammar: vendored upstream C++17 source under
        // Libraries/MLXCXGrammar/xgrammar/ compiled directly by SPM, plus our
        // own shim.cc exposing the extern "C" API from xgrammar_c.h.
        //
        // Refresh the vendored tree with scripts/sync-xgrammar-source.sh.
        // The pinned upstream sha lives in Libraries/MLXCXGrammar/xgrammar/VERSION
        // and is mirrored in shim.cc's kXGrammarVersion.
        .target(
            name: "MLXCXGrammar",
            path: "Libraries/MLXCXGrammar",
            exclude: [
                // Compiled via Libraries/MLXCXGrammar/grammar_functor_wrapper.cc to
                // provide out-of-class definitions for static const members that
                // clang ODR-uses through variadic templates.
                "xgrammar/cpp/grammar_functor.cc"
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("xgrammar/include"),
                .headerSearchPath("xgrammar/cpp"),
                .headerSearchPath("xgrammar/3rdparty/picojson"),
                .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
                .define("XGRAMMAR_ENABLE_CPPTRACE", to: "0"),
                .define("XGRAMMAR_ENABLE_INTERNAL_CHECK", to: "0"),
                // Rename the vendored C++ namespaces at compile time so this
                // target's symbols cannot collide with another xgrammar in the
                // same binary (e.g. CoreAI's prebuilt copy). Token-level
                // substitution: it rewrites bare `xgrammar` / `picojson`
                // identifiers (namespace decls and `::` uses) but not header
                // names, string literals, `XGRAMMAR_*` macros, or `xg_*` tokens.
                .define("xgrammar", to: "mlx_xgrammar"),
                .define("picojson", to: "mlx_picojson"),
                // Vendored upstream xgrammar/picojson is compiled as-is and is
                // not warning-clean under Xcode's default warning set (e.g.
                // -Wshorten-64-to-32). Suppress all warnings for this target so
                // the unmodified upstream C++ does not spam consumers' builds.
                // `-w` wins over any preceding -W flags. Scoped to this target
                // only; our own shim (shim.cc) is small and stable.
                .unsafeFlags(["-w"], .when(platforms: [.macOS, .iOS, .visionOS, .tvOS])),
            ],
            linkerSettings: [
                // Apple platforms only: on Linux the Swift toolchain links libstdc++,
                // and there is no libc++ to link against.
                .linkedLibrary("c++", .when(platforms: [.macOS, .iOS, .visionOS, .tvOS]))
            ]
        ),
        // Grammar-constrained ("guided") generation engine built on the
        // vendored xgrammar C++ via CXGrammar. Standalone: depends only on
        // CXGrammar + MLXLMCommon + MLX, with no FoundationModels coupling and
        // no @available floor beyond the package's macOS 14 / iOS 17 minimum.
        .target(
            name: "MLXGuidedGeneration",
            dependencies: [
                "MLXLMCommon",
                "MLXCXGrammar",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/MLXGuidedGeneration"
        ),
        // Bridges Apple's FoundationModels framework to MLX-powered on-device
        // inference. Public surface is gated by @available(macOS 27 / iOS 27 /
        // visionOS 27, *) and #if canImport(FoundationModels), so the target
        // builds on every Xcode that compiles the rest of mlx-swift-lm. The
        // MLXGuidedGeneration dependency is trait-conditional: it is linked only
        // when FoundationModelsIntegration is enabled, since the adapter
        // references the engine exclusively inside that gate.
        .target(
            name: "MLXFoundationModels",
            dependencies: [
                "MLXLMCommon",
                .target(
                    name: "MLXGuidedGeneration",
                    condition: .when(traits: ["FoundationModelsIntegration"])
                ),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Libraries/MLXFoundationModels"
        ),
        .testTarget(
            name: "MLXFoundationModelsTests",
            dependencies: [
                "MLXFoundationModels",
                "MLXLMCommon",
                .target(
                    name: "MLXGuidedGeneration",
                    condition: .when(traits: ["FoundationModelsIntegration"])
                ),
                // MLXLLM is linked here (not by MLXFoundationModels itself) so its
                // module-init registers a factory with MLXLMCommon's
                // ModelFactoryRegistry. Without it, loadModelContainer throws
                // .noModelFactoryAvailable before ever reaching the downloader,
                // which deadlocks AvailabilityTests' in-flight gate. Model-free:
                // the tests inject a stub downloader — no network, no real weights.
                "MLXLLM",
                // Registers the VLM trampoline factory so gemma4 resolves at
                // load time. The MLXFoundationModels product target
                // deliberately does NOT depend on MLXVLM; runtime trampoline
                // discovery is by design and unchanged.
                "MLXVLM",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/MLXFoundationModelsTests"
        ),
        // FM-independent guided-generation tests. Depends only on the engine
        // and MLXLMCommon. No FoundationModels, no direct CXGrammar.
        .testTarget(
            name: "MLXGuidedGenerationTests",
            dependencies: [
                "MLXGuidedGeneration",
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/MLXGuidedGenerationTests"
        ),
        // Direct C-API tests for the CXGrammar shim. No FoundationModels
        // dependency; exercises the vendored xgrammar C++ library through
        // the shim's public C entry points.
        .testTarget(
            name: "CXGrammarTests",
            dependencies: ["MLXCXGrammar"],
            path: "Tests/CXGrammarTests",
            // tokenizer_gemma3.json is read at runtime via a #filePath-relative
            // path (see goldensDirectory in the test sources), not bundled, so
            // the Fixtures tree is excluded from the build graph.
            exclude: ["Fixtures"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
