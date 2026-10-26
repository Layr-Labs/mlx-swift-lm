import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Local checkpoint tensors are read only. No weights, downloaded fixtures,
/// cache state or frozen numerical references belong in this test's sources.
@Suite(
    "Gemma 4 checkpoint feed-forward parity", .tags(.integration), .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["GEMMA4_FFN_CHECKPOINT_DIR"] != nil))
struct Gemma4CheckpointFeedforwardTests {
    private static let environment = ProcessInfo.processInfo.environment
    private static let projections = [
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj",
        "experts.switch_glu.gate_proj", "experts.switch_glu.up_proj",
        "experts.switch_glu.down_proj",
    ]
    private static let vectors = [
        "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight", "pre_feedforward_layernorm_2.weight",
        "post_feedforward_layernorm_2.weight", "router.scale", "router.per_expert_scale",
    ]

    private func loadLayer(
        directory: URL, configuration: Gemma4TextConfiguration, index: Int,
        weightMap: [String: String], quantization: [String: Any]
    ) throws -> Gemma4DecoderLayer {
        let prefix = "language_model.model.layers.\(index)."
        let relativeNames =
            Self.vectors
            + Self.projections.flatMap { path in
                ["weight", "scales", "biases"].map { path + "." + $0 }
            }
        let wanted = Set(relativeNames.map { prefix + $0 })
        try #require(wanted.count == 28 && wanted.allSatisfy { weightMap[$0] != nil })
        let shardNames = Set(wanted.compactMap { weightMap[$0] }).sorted()
        try #require(shardNames.allSatisfy { URL(fileURLWithPath: $0).lastPathComponent == $0 })
        let checkpoint = try loadModelCheckpoint(
            urls: shardNames.map { directory.appendingPathComponent($0) },
            isRead: { wanted.contains($0) })
        try #require(Set(checkpoint.weights.keys) == wanted)
        let weights = Dictionary(
            uniqueKeysWithValues: checkpoint.weights.map {
                (String($0.key.dropFirst(prefix.count)), $0.value)
            })
        let layer = Gemma4DecoderLayer(configuration, layerIdx: index)
        var children: [(String, Module)] = []
        for path in Self.projections {
            let expert = path.hasPrefix("experts.")
            let bits = expert ? 4 : 8
            let policy = quantization[prefix + path] as? [String: Any] ?? quantization
            try #require(policy["bits"] as? Int == bits)
            try #require(policy["group_size"] as? Int == 64)
            try #require((policy["mode"] as? String ?? "affine") == "affine")
            let weight = try #require(weights[path + ".weight"])
            let scales = try #require(weights[path + ".scales"])
            let biases = try #require(weights[path + ".biases"])
            try #require(weight.dtype == .uint32)
            try #require(scales.dtype == .bfloat16 && biases.dtype == .bfloat16)
            let input = path.hasSuffix("down_proj") ? (expert ? 704 : 2112) : 2816
            let output =
                path.hasSuffix("down_proj")
                ? 2816 : (path == "router.proj" ? 128 : (expert ? 704 : 2112))
            let expected = expert ? [128, output, input * bits / 32] : [output, input * bits / 32]
            try #require(weight.shape == expected)
            try #require(
                scales.shape == (expert ? [128, output, input / 64] : [output, input / 64]))
            try #require(biases.shape == scales.shape)
            if expert {
                children.append(
                    (
                        path,
                        try QuantizedSwitchLinear(
                            inputDims: input, outputDims: output, numExperts: 128,
                            weight: weight, scales: scales, biases: biases, groupSize: 64, bits: 4)
                    ))
            } else {
                children.append(
                    (
                        path,
                        QuantizedLinear(
                            weight: weight, scales: scales, biases: biases, groupSize: 64, bits: 8)
                    ))
            }
        }
        layer.update(modules: .unflattened(children))
        let vectors = Dictionary(
            uniqueKeysWithValues: try Self.vectors.map { name in
                let value = try #require(weights[name])
                try #require(value.dtype == .bfloat16)
                try #require(value.shape == [name == "router.per_expert_scale" ? 128 : 2816])
                return (name, value)
            })
        layer.update(parameters: .unflattened(vectors))
        layer.train(false)
        let state = try #require(layer.feedforwardCompilationState())
        try #require(state.parameters.count == 28 && state.topK == 8)
        eval(state.parameters)
        // No call to quantize: packed checkpoint tensors are installed directly.
        return layer
    }

    private func compare(
        _ layer: Gemma4DecoderLayer, input: MLXArray, shouldCompile: Bool, label: String
    ) {
        eval(input)
        layer.invalidateCompiledTraces()
        let reference = GPU.withCommandBufferLimits(.init(operations: 0, megabytes: 0)) {
            let value = layer.feedforward(input, isExpertPrefill: false)
            eval(value)
            return value
        }
        let actual = GPU.withCommandBufferLimits(Gemma4DecodeCommandBuffers.limits) {
            let value = layer.feedforwardWithCompilation(input, isExpertPrefill: false)
            eval(value)
            return value
        }
        #expect(layer.feedforwardTraceIsCompiled == shouldCompile, "\(label): route")
        #expect(actual.shape == reference.shape && actual.dtype == reference.dtype, "\(label)")
        #expect(
            actual.asData(access: .copy).data == reference.asData(access: .copy).data,
            "\(label): native output bytes")
    }

    @Test func actualCheckpointShapesPreserveNativeBytes() throws {
        try #require(Device.defaultDevice().deviceType == .gpu)
        try #require(MLXHardwareInfo.isCompiledDecodeSupported)
        let flag = Self.environment["DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD"] ?? "1"
        try #require(["1", "true", "yes", "on"].contains(flag.lowercased()))
        let directory = URL(
            fileURLWithPath: try #require(Self.environment["GEMMA4_FFN_CHECKPOINT_DIR"]))
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let root = try #require(
            try JSONSerialization.jsonObject(with: configData) as? [String: Any])
        let text = try #require(root["text_config"] as? [String: Any])
        let configuration = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: JSONSerialization.data(withJSONObject: text))
        try #require(configuration.hiddenSize == 2816 && configuration.intermediateSize == 2112)
        try #require(configuration.moeIntermediateSize == 704 && configuration.numExperts == 128)
        try #require(configuration.topKExperts == 8 && configuration.numHiddenLayers == 30)
        try #require(configuration.hiddenSizePerLayerInput == 0 && !configuration.useDoubleWideMlp)
        let quantization = try #require(root["quantization"] as? [String: Any])
        let indexData = try Data(
            contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        let index = try #require(
            try JSONSerialization.jsonObject(with: indexData) as? [String: Any])
        let weightMap = try #require(index["weight_map"] as? [String: String])
        let selection = Self.environment["GEMMA4_FFN_CHECKPOINT_LAYERS"] ?? "0,15,29"
        let layers: [Int]
        if selection == "all" {
            layers = Array(0 ..< 30)
        } else {
            let items = selection.split(separator: ",", omittingEmptySubsequences: false)
            layers = items.compactMap { Int($0) }
            try #require(layers.count == items.count)
        }
        try #require(
            !layers.isEmpty && Set(layers).count == layers.count
                && layers.allSatisfy { (0 ..< 30).contains($0) })
        func hash(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        print(
            "GEMMA4_FFN_CHECKPOINT config=\(hash(configData)) index=\(hash(indexData)) layers=\(layers) compiled=\(flag)"
        )
        for index in layers {
            let layer = try loadLayer(
                directory: directory, configuration: configuration, index: index,
                weightMap: weightMap, quantization: quantization)
            // Include actual concurrent ordinary-decode widths below the routing sort boundary.
            for (batch, width, compiled) in [
                (1, 1, true), (2, 1, true), (4, 1, true), (7, 1, true),
                (1, 2, true), (3, 2, true), (4, 2, false),
            ] {
                for amplitude: Float in [0, 0.25, 4] {
                    let input =
                        (MLXRandom.normal(
                            [batch, width, 2816], key: MLXRandom.key(UInt64(12300 + index)))
                        * amplitude).asType(
                            .bfloat16)
                    compare(
                        layer, input: input, shouldCompile: compiled,
                        label: "layer\(index) B\(batch) S\(width) amplitude\(amplitude)")
                }
            }
            // Deliberate local test perturbation makes all router scores tie
            // while real nonzero inputs still exercise distinct expert banks.
            layer.update(
                parameters: .unflattened(["router.scale": MLXArray.zeros([2816], dtype: .bfloat16)])
            )
            let projection = try #require(
                layer.namedModules().first { $0.0 == "router.proj" }?.1 as? QuantizedLinear)
            for batch in [1, 3] {
                let input = MLXRandom.normal([batch, 2, 2816], key: MLXRandom.key(12400)).asType(
                    .bfloat16)
                #expect(
                    abs(projection(MLXArray.zeros([batch, 2, 2816], dtype: .bfloat16))).max().item(
                        Float.self)
                        == 0)
                compare(
                    layer, input: input, shouldCompile: true,
                    label: "layer\(index) tied router B\(batch) S2")
            }
            // Discard this local perturbed layer; checkpoint files are unchanged.
            print(
                "GEMMA4_FFN_CHECKPOINT layer=\(index) comparisons=23 native-byte checks completed")
        }
    }
}
