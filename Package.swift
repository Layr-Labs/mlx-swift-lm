// swift-tools-version: 6.1
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
            name: "MLXHuggingFace",
            targets: ["MLXHuggingFace"]),
        .library(
            name: "MLXLMServer",
            targets: ["MLXLMServer"]),
        .executable(
            name: "mlx-server",
            targets: ["mlx-server"]),
        .library(
            name: "BenchmarkHelpers",
            targets: ["BenchmarkHelpers"]),
        .library(
            name: "IntegrationTestHelpers",
            targets: ["IntegrationTestHelpers"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/Layr-Labs/mlx-swift.git",
            revision: "6923a80f624f5c91fbf456efe4e00e9698a72961"),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "600.0.0" ..< "604.0.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.23.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.2"),
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.3.6"),
        .package(url: "https://github.com/apple/swift-crypto.git", "3.9.0" ..< "5.0.0"),
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
        // Linux: exact CommandLine.arguments in every executable and test that
        // links MLXLMCommon (librocblas calls setenv while it loads). See argv.c.
        .target(
            name: "MLXLinuxArgv",
            path: "Libraries/MLXLinuxArgv"
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .target(name: "MLXLinuxArgv", condition: .when(platforms: [.linux])),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // Cmlx: low-level mlx_slice_dynamic / mlx_slice_update_dynamic
                // entry points used by DynamicSlice.swift (compiled-decode infra).
                .product(name: "Cmlx", package: "mlx-swift"),
                // CryptoKit replacement where CryptoKit does not exist.
                .product(
                    name: "Crypto", package: "swift-crypto",
                    condition: .when(platforms: [.linux])),
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
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/IntegrationTestHelpers",
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
                .product(
                    name: "Crypto", package: "swift-crypto",
                    condition: .when(platforms: [.linux])),
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
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}

// MLXShim builds custom kernels for the CUDA and ROCm backends of MLX. It is
// opt-in: set MLXSHIM_PATH to a local mlx-shim checkout. Without it, MLXLLM
// calls MLXFast.metalKernel directly and the package has no MLXShim
// dependency. On Linux without MLXShim, the gated-delta kernels are nil and
// gatedDeltaUpdate uses gatedDeltaOps.
//
// Limits:
// - MLXSHIM_PATH is an absolute path or a path relative to this package's
//   root. It works only when mlx-swift-lm is the root package or a local
//   (path) dependency. A package that depends on mlx-swift-lm by URL cannot
//   use it, because SwiftPM does not allow a local dependency below a URL
//   dependency.
// - Also set MLXSHIM_MLX_SWIFT=url. By default mlx-shim takes mlx-swift from
//   `../mlx-swift`. That path and the URL pin above have the same package
//   identity, `mlx-swift`. With MLXSHIM_MLX_SWIFT=url, mlx-shim uses the same
//   URL at the same revision.
if let mlxShimPath = Context.environment["MLXSHIM_PATH"] {
    precondition(
        !mlxShimPath.isEmpty,
        "MLXSHIM_PATH is set but empty. Unset it, or set it to a local mlx-shim checkout."
    )
    // SwiftPM names a path dependency after the last path component.
    let mlxShimPackage = String(mlxShimPath.split(separator: "/").last ?? "mlx-shim").lowercased()
    package.dependencies.append(.package(path: mlxShimPath))
    for target in package.targets where target.name == "MLXLLM" {
        target.dependencies.append(.product(name: "MLXShim", package: mlxShimPackage))
        target.swiftSettings = (target.swiftSettings ?? []) + [.define("MLX_SHIM")]
    }
    // Gated-delta kernel vs op-path parity on the current backend.
    package.targets.append(
        .executableTarget(
            name: "GDNParity",
            dependencies: [
                "MLXLLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXShim", package: mlxShimPackage),
            ],
            path: "Sources/GDNParity"
        ))
}
