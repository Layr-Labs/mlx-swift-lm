import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

private let nemotronSuperBatchBenchmarkMode =
    ProcessInfo.processInfo.environment["NEMOTRON_SUPER_BATCH_BENCH"]

/// Opt-in decode step timing at real Super dimensions with synthetic weights:
/// FP8 Mamba/attention projections and FP8 output head over the full
/// vocabulary, latent MoE with 512 NVFP4 experts (top-22) and an NVFP4 shared
/// expert, BF16 latent projections and router. Three stacks ("ME*", "MEME*",
/// "ME**") separate the marginal cost of a Mamba+MoE pair and of an
/// attention layer from the fixed embedding/head cost, so a full 88-layer
/// step can be projected. Synthetic random routing spreads rows over the
/// most experts. Run with NEMOTRON_SUPER_BATCH_BENCH=1 through the GPU lane;
/// `quick` runs only the first stack at B = 1, 2 as a smoke check.
@Suite(
    .tags(.integration),
    .enabled(if: ["1", "quick"].contains(nemotronSuperBatchBenchmarkMode)))
struct NemotronSuperBatchedDecodeBenchmark {
    typealias Fixture = NemotronSuperBatchFixture
    static let mode = nemotronSuperBatchBenchmarkMode
    static let batches = mode == "quick" ? [1, 2] : [1, 2, 4, 8, 16]
    static let patterns = mode == "quick" ? ["ME*"] : ["ME*", "MEME*", "ME**"]
    static let promptTokens = 32
    static let warmupSteps = 3
    static let measuredSteps = 12

    static func configuration(_ pattern: String) -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 131_072, hiddenSize: 4_096, numHiddenLayers: pattern.count,
            numAttentionHeads: 32, numKeyValueHeads: 2, mambaNumHeads: 128, mambaHeadDim: 64,
            ssmStateSize: 128, convKernel: 4, nGroups: 8, intermediateSize: 2_688,
            moeIntermediateSize: 2_688, moeSharedExpertIntermediateSize: 5_376,
            nRoutedExperts: 512, numExpertsPerTok: 22, hybridOverridePattern: pattern,
            headDim: 128, nSharedExperts: 1, routedScalingFactor: 5, moeLatentSize: 1_024)
    }

    final class Keys {
        private var next: UInt64
        init(_ seed: UInt64) { next = seed }
        func callAsFunction() -> MLXArray {
            next &+= 1
            return MLXRandom.key(next)
        }
    }

    static func bytes(_ shape: [Int], _ range: Range<Int32>, _ key: MLXArray) -> MLXArray {
        let value = MLXRandom.randInt(range, shape, key: key).asType(.uint8)
        eval(value)
        return value
    }

    /// E4M3 codes without the two NaN encodings.
    static func fp8(_ outputs: Int, _ inputs: Int, _ key: Keys) throws -> ModelOptCalibratedLinear {
        let raw = MLXRandom.randInt(Int32(0) ..< 254, [outputs, inputs], key: key())
        let codes = which(raw .>= 127, raw + 1, raw).asType(.uint8)
        eval(codes)
        return try ModelOptCalibratedLinear(
            weights: ModelOptFP8Weights(
                e4m3Bytes: codes, tensorScale: 1 / (64 * Float(inputs).squareRoot())),
            inputScale: 8 / 448, weightDType: .bfloat16)
    }

    static func nvfp4(_ outputs: Int, _ inputs: Int, _ key: Keys) throws -> ModelOptNVFP4Weights {
        try ModelOptNVFP4Weights(
            packedBytes: bytes([outputs, inputs / 2], 0 ..< 256, key()),
            e4m3ScaleBytes: bytes([outputs, inputs / 16], 0x28 ..< 0x49, key()),
            tensorScale: 0.05 / Float(inputs).squareRoot())
    }

    static func bank(_ experts: Int, _ outputs: Int, _ inputs: Int, _ key: Keys) throws
        -> ModelOptNVFP4SwitchLinear
    {
        let packed = bytes([experts, outputs, inputs / 2], 0 ..< 256, key())
        let scales = bytes([experts, outputs, inputs / 16], 0x28 ..< 0x49, key())
        let weights = try (0 ..< experts).map {
            try ModelOptNVFP4Weights(
                packedBytes: packed[$0], e4m3ScaleBytes: scales[$0],
                tensorScale: 0.05 / Float(inputs).squareRoot())
        }
        return try ModelOptNVFP4SwitchLinear(
            experts: weights, inputScales: Array(repeating: 16 / 2688, count: experts),
            weightDType: .bfloat16)
    }

    static func model(_ pattern: String, seed: UInt64) throws -> NemotronSuperModel {
        let configuration = configuration(pattern)
        let language = NemotronHModel(configuration)
        let key = Keys(seed)
        var dense: [String: MLXArray] = [:]
        for (name, value) in language.parameters().flattened() {
            // The checkpoint keeps A_log and the router bias outside the BF16 cast.
            if name.hasSuffix("A_log") || name.hasSuffix("e_score_correction_bias") { continue }
            if name.hasSuffix(".D") || name.hasSuffix("dt_bias") {
                dense[name] = value.asType(.bfloat16)
                continue
            }
            let replaced =
                name.contains("switch_mlp")
                || name.contains("_proj.weight")
                    && !name.contains("latent")
                || name == "lm_head.weight"
            if replaced {
                // Replaced by a ModelOpt module below before any evaluation.
                dense[name] = MLXArray.zeros(value.shape, dtype: .bfloat16)
            } else if name.contains("norm") {
                dense[name] = (1 + 0.1 * MLXRandom.normal(value.shape, key: key()))
                    .asType(.bfloat16)
            } else {
                let width = Float(value.dim(-1))
                dense[name] = (MLXRandom.normal(value.shape, key: key()) / width.squareRoot())
                    .asType(.bfloat16)
            }
        }
        try language.update(
            parameters: .unflattened(dense), verify: [.noUnusedKeys, .shapeMismatch])
        var projections: [String: ModelOptCalibratedLinear] = [:]
        var experts: [String: ModelOptNVFP4SwitchLinear] = [:]
        for (path, module) in language.leafModules().flattened() {
            if let linear = module as? Linear, type(of: linear) == Linear.self,
                !path.contains("latent")
            {
                let (outputs, inputs) = linear.shape
                projections[path] =
                    path.contains(".shared_experts.")
                    ? try ModelOptCalibratedLinear(
                        weights: nvfp4(outputs, inputs, key), inputScale: 16 / 2688,
                        weightDType: .bfloat16)
                    : try fp8(outputs, inputs, key)
            } else if module is SwitchLinear,
                let shape = Dictionary(uniqueKeysWithValues: module.parameters().flattened())[
                    "weight"]?.shape
            {
                experts[path] = try bank(shape[0], shape[1], shape[2], key)
            }
        }
        try language.installModelOptProjections(projections)
        try language.installModelOptExperts(experts)
        language.setModelOptExecution(.graph)
        eval(language.parameters())
        let images = NemotronSuperImageComposition(
            tower: NemotronRadio(
                NemotronRadioDimensions(
                    width: 2, layers: 1, heads: 1,
                    channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)),
            projector: NemotronSuperVisionProjector(
                visionWidth: 2, hiddenWidth: 3, outputWidth: 4_096),
            width: 4_096, imageTokenID: 18)
        return try NemotronSuperModel(languageModel: language, imageComposition: images)
    }

    struct Timing {
        let medianMilliseconds: Double
        let p10: Double
        let p90: Double
        let buildMilliseconds: Double
        let peakTransientBytes: Int
    }

    static func percentile(_ values: [Double], _ fraction: Double) -> Double {
        let sorted = values.sorted()
        return sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * fraction + 0.5))]
    }

    /// Distinct token streams per row, so rows route independently.
    static func stream(_ count: Int, row: Int, salt: Int) -> [Int32] {
        (0 ..< count).map { index -> Int32 in
            let value = (index + 1) * 7_919 + row * 104_729 + salt
            return Int32(value % 131_072)
        }
    }

    static func measure(
        _ model: NemotronSuperModel, rows: Int,
        products: NemotronSuperRecurrentAdapter.DecodeProducts
    ) throws -> Timing {
        let runner = try NemotronSuperBatchRunner(
            model: model, maximumRows: 16, kvDType: .bfloat16, products: products)
        let steps = warmupSteps + measuredSteps
        let lanes = try (0 ..< rows).map { row in
            try runner.lane(
                prompt: stream(promptTokens, row: row, salt: 11),
                inputs: stream(steps, row: row, salt: 65_537))
        }
        for lane in lanes { try runner.prefill(lane) }
        var totals: [Double] = []
        var builds: [Double] = []
        var peak = 0
        for step in 0 ..< steps {
            if step == warmupSteps {
                Memory.clearCache()
                Memory.peakMemory = 0
                peak = Memory.activeMemory
            }
            let started = DispatchTime.now().uptimeNanoseconds
            let cache = runner.caches(lanes)
            let bindings = try lanes.map { try $0.state.bind() }
            let tokens = MLXArray(lanes.map { $0.inputs[$0.decoded] }).reshaped(rows, 1)
            let logits = runner.adapter.cbv2Forward(tokens, caches: cache, recurrentState: bindings)
            var roots = [logits] + cache.flatMap { $0.innerState() }
            for binding in bindings { roots += try binding.evaluate() }
            let built = DispatchTime.now().uptimeNanoseconds
            eval(roots)
            let finished = DispatchTime.now().uptimeNanoseconds
            for (binding, lane) in zip(bindings, lanes) {
                try binding.commit()
                lane.decoded += 1
            }
            if step >= warmupSteps {
                totals.append(Double(finished - started) / 1e6)
                builds.append(Double(built - started) / 1e6)
            }
        }
        let transient = Memory.peakMemory - peak
        for lane in lanes { try runner.release(lane) }
        return Timing(
            medianMilliseconds: percentile(totals, 0.5), p10: percentile(totals, 0.1),
            p90: percentile(totals, 0.9), buildMilliseconds: percentile(builds, 0.5),
            peakTransientBytes: transient)
    }

    /// One measured configuration: batched products and ModelOpt settings.
    struct Configuration {
        let name: String
        let products: NemotronSuperRecurrentAdapter.DecodeProducts
        let settings: Fixture.Products
    }

    static let configurations = [
        Configuration(name: "row-exact, tiled default", products: .rowExact, settings: .tiled),
        Configuration(name: "shared, rowwise", products: .shared, settings: .rowwise),
        Configuration(name: "shared, tiled", products: .shared, settings: .tiled),
    ]

    static func apply(_ settings: Fixture.Products, to model: NemotronSuperModel) {
        switch settings {
        case .rowwise: model.languageModel.setModelOptAccumulation(.rowwise)
        case .tiled: model.languageModel.setModelOptAccumulation(.tiled)
        }
    }

    @Test(.timeLimit(.minutes(30)))
    func decodeStepTimeByBatchAtRealDimensions() throws {
        var steps: [String: [String: [Int: Timing]]] = [:]
        for pattern in Self.patterns {
            let model = try Self.model(pattern, seed: UInt64(pattern.count))
            if pattern == "ME*" {
                let charge = try model.languageModel.superDecodeCharge(kvElementBytes: 2)
                print("BENCH real-dimension decode charge: \(charge)")
            }
            for configuration in Self.configurations {
                Self.apply(configuration.settings, to: model)
                let name = configuration.name
                if pattern == "ME*" {
                    // Eight rows against each row decoded alone, real shapes.
                    let requests = (0 ..< 8).map { row in
                        NemotronSuperBatchedDecodeTests.Request(
                            prompt: Self.stream(16, row: row, salt: 3),
                            inputs: Self.stream(4, row: row, salt: 32_771))
                    }
                    let measured = NemotronSuperBatchedDecodeTests.deviation(
                        try NemotronSuperBatchedDecodeTests.batched(
                            model, requests, maximumRows: 8, products: configuration.products),
                        try NemotronSuperBatchedDecodeTests.serial(model, requests))
                    print("BENCH real-dimension B=8 deviation \(name): \(measured)")
                }
                for rows in Self.batches {
                    let timing = try Self.measure(
                        model, rows: rows, products: configuration.products)
                    steps[name, default: [:]][pattern, default: [:]][rows] = timing
                    print(
                        String(
                            format:
                                "BENCH %@ %@ B=%d: step median %.3f ms (p10 %.3f, p90 %.3f), "
                                + "host build %.3f ms, peak transient %.1f MiB",
                            name, pattern, rows, timing.medianMilliseconds, timing.p10,
                            timing.p90, timing.buildMilliseconds,
                            Double(timing.peakTransientBytes) / 1_048_576))
                }
            }
        }
        // Full model: 40 Mamba, 40 MoE and 8 attention layers plus one head.
        for configuration in Self.configurations {
            let name = configuration.name
            for rows in Self.batches {
                guard let base = steps[name]?["ME*"]?[rows]?.medianMilliseconds,
                    let pair = steps[name]?["MEME*"]?[rows]?.medianMilliseconds,
                    let attention = steps[name]?["ME**"]?[rows]?.medianMilliseconds
                else { continue }
                let projected = base + 39 * (pair - base) + 7 * (attention - base)
                print(
                    String(
                        format:
                            "BENCH projected 88-layer %@ B=%d: Mamba+MoE pair %.3f ms, "
                            + "attention %.3f ms, step %.1f ms, %.2f tok/s per request, "
                            + "%.2f tok/s aggregate",
                        name, rows, pair - base, attention - base, projected,
                        1_000 / projected, Double(rows) * 1_000 / projected))
            }
        }
    }
}
