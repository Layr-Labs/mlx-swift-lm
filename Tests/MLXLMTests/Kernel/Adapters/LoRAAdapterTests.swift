import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the LoRA and DoRA layers and of `LoRAContainer`,
    /// `ModelAdapterFactory` and `ModelAdapterTypeRegistry`.
    ///
    /// The layer tests compare each layer with the formula of its forward
    /// pass, written with plain matrix products in the test.
    @Suite
    struct LoRAAdapterTests {

        static func random(_ shape: [Int], seed: UInt64, scale: Float = 1) -> MLXArray {
            let x = MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale
            eval(x)
            return x
        }

        /// A `Linear(32 -> 16)` with bias and random weights.
        static func linear() -> Linear {
            Linear(weight: random([16, 32], seed: 1, scale: 0.2), bias: random([16], seed: 2))
        }

        static func quantizedLinear() -> QuantizedLinear {
            QuantizedLinear(
                weight: random([64, 64], seed: 1, scale: 0.2), bias: nil, groupSize: 32, bits: 8)
        }

        // Tolerance of the float32 comparisons: the layer and the formula
        // compute the same products in another order; the error is near
        // 1e-6 for outputs of size 1 to 5.
        static let tolerance: Float = 1e-4

        /// Gives a LoRA or DoRA layer random `lora_a` and `lora_b` (and a
        /// random magnitude `m` for DoRA).
        static func randomizeAdapter(_ layer: Module, magnitude: Bool = false) {
            var updates: [(String, MLXArray)] = []
            for (index, (key, value)) in layer.parameters().flattened().enumerated()
            where key == "lora_a" || key == "lora_b" || (magnitude && key == "m") {
                let noise = random(value.shape, seed: 100 + UInt64(index), scale: 0.3)
                // A magnitude is a positive row length.
                updates.append((key, key == "m" ? abs(noise) + 1 : noise))
            }
            layer.update(parameters: ModuleParameters.unflattened(updates))
            eval(layer)
        }

        @Test func freshLoRALayerMatchesTheBaseLayer() throws {
            let base = Self.linear()
            let x = Self.random([2, 3, 32], seed: 3)
            let lora = try #require(LoRALinear.from(linear: base, rank: 4, scale: 2) as? LoRALinear)
            #expect(SyntheticModel.maxAbsDifference(lora(x), base(x)) <= Self.tolerance)
            // Only the adapter parameters are trainable.
            let trainable = Set(lora.trainableParameters().flattened().map(\.0))
            #expect(trainable == ["lora_a", "lora_b"])
        }

        @Test func loRALayerAddsTheScaledLowRankProduct() throws {
            let base = Self.linear()
            let x = Self.random([2, 3, 32], seed: 3)
            let lora = try #require(LoRALinear.from(linear: base, rank: 4, scale: 2) as? LoRALinear)
            Self.randomizeAdapter(lora)
            let a = SyntheticModel.flatParameters(lora)["lora_a"]!
            let b = SyntheticModel.flatParameters(lora)["lora_b"]!
            let expected = base(x) + 2 * matmul(matmul(x, a), b)
            #expect(SyntheticModel.maxAbsDifference(lora(x), expected) <= Self.tolerance)

            // The fused layer computes the same output with one weight.
            let fused = lora.fused()
            #expect(!(fused is LoRALinear))
            let fusedLinear = try #require(fused as? Linear)
            #expect(SyntheticModel.maxAbsDifference(fusedLinear(x), expected) <= Self.tolerance)

            // The reverted layer is the base layer again.
            let reverted = try #require(lora.reverted() as? Linear)
            #expect(SyntheticModel.maxAbsDifference(reverted(x), base(x)) == 0)
        }

        @Test func quantizedLoRALayerAddsTheLowRankProductToTheQuantizedLayer() throws {
            let base = Self.quantizedLinear()
            let x = Self.random([2, 3, 64], seed: 3)
            let lora = try #require(
                LoRALinear.from(linear: base, rank: 4, scale: 2) as? QLoRALinear)
            Self.randomizeAdapter(lora)
            let a = SyntheticModel.flatParameters(lora)["lora_a"]!
            let b = SyntheticModel.flatParameters(lora)["lora_b"]!
            let expected = base(x) + 2 * matmul(matmul(x, a), b)
            #expect(SyntheticModel.maxAbsDifference(lora(x), expected) <= Self.tolerance)

            // Fusing quantizes W + delta again with 8 bits in groups of 32,
            // which moves each weight by up to 1/255 of its group range.
            let fused = try #require(lora.fused() as? QuantizedLinear)
            #expect(SyntheticModel.maxAbsDifference(fused(x), expected) <= 0.1)
            let reverted = try #require(lora.reverted() as? QuantizedLinear)
            #expect(SyntheticModel.maxAbsDifference(reverted(x), base(x)) == 0)
        }

        /// DoRA: `(W x + s B A x) * m / |W + s B A|_row + bias`.
        static func doraReference(
            x: MLXArray, weight: MLXArray, bias: MLXArray?, a: MLXArray, b: MLXArray,
            m: MLXArray, scale: Float
        ) -> MLXArray {
            let adapted = weight + scale * matmul(b.T, a.T)
            let rowNorm = sqrt((adapted * adapted).sum(axis: 1))
            var out = matmul(x, weight.T) + scale * matmul(matmul(x, a), b)
            out = out * (m / rowNorm)
            if let bias { out = out + bias }
            return out
        }

        @Test func freshDoRALayerMatchesTheBaseLayer() throws {
            let base = Self.linear()
            let x = Self.random([2, 3, 32], seed: 3)
            let dora = try #require(DoRALinear.from(linear: base, rank: 4, scale: 2) as? DoRALinear)
            #expect(SyntheticModel.maxAbsDifference(dora(x), base(x)) <= Self.tolerance)
            let trainable = Set(dora.trainableParameters().flattened().map(\.0))
            #expect(trainable == ["lora_a", "lora_b", "m"])
        }

        @Test func doRALayerRescalesTheAdaptedRows() throws {
            let base = Self.linear()
            let x = Self.random([2, 3, 32], seed: 3)
            let dora = try #require(DoRALinear.from(linear: base, rank: 4, scale: 2) as? DoRALinear)
            Self.randomizeAdapter(dora, magnitude: true)
            let p = SyntheticModel.flatParameters(dora)
            let expected = Self.doraReference(
                x: x, weight: base.weight, bias: base.bias, a: p["lora_a"]!,
                b: p["lora_b"]!, m: p["m"]!, scale: 2)
            #expect(SyntheticModel.maxAbsDifference(dora(x), expected) <= Self.tolerance)

            let fused = try #require(dora.fused() as? Linear)
            #expect(SyntheticModel.maxAbsDifference(fused(x), expected) <= Self.tolerance)
            let reverted = try #require(dora.reverted() as? Linear)
            #expect(SyntheticModel.maxAbsDifference(reverted(x), base(x)) == 0)
        }

        @Test func quantizedDoRALayerRescalesTheDequantizedRows() throws {
            let base = Self.quantizedLinear()
            let x = Self.random([2, 3, 64], seed: 3)
            let dora = try #require(
                DoRALinear.from(linear: base, rank: 4, scale: 2) as? QDoRALinear)
            // A fresh layer matches the quantized base layer.
            #expect(SyntheticModel.maxAbsDifference(dora(x), base(x)) <= Self.tolerance)
            Self.randomizeAdapter(dora, magnitude: true)
            let p = SyntheticModel.flatParameters(dora)
            let expected = Self.doraReference(
                x: x, weight: base.dequantizedWeight, bias: base.bias,
                a: p["lora_a"]!, b: p["lora_b"]!, m: p["m"]!,
                scale: 2)
            #expect(SyntheticModel.maxAbsDifference(dora(x), expected) <= Self.tolerance)
            // Fusing quantizes again with 8 bits.
            let fused = try #require(dora.fused() as? QuantizedLinear)
            #expect(SyntheticModel.maxAbsDifference(fused(x), expected) <= 0.1)
        }

        // MARK: - LoRAContainer

        static func model(seed: UInt64 = 1) throws -> Qwen3Model {
            let configuration = try SyntheticModel.configuration(
                Qwen3Configuration.self,
                [
                    "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                    "vocab_size": 64, "rope_theta": 10000, "head_dim": 8,
                ])
            let model = Qwen3Model(configuration)
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static let tokens = [[3, 1, 4, 1, 5, 9, 2, 6]]

        static func loraConfiguration(_ type: LoRAConfiguration.FineTuneType)
            -> LoRAConfiguration
        {
            LoRAConfiguration(
                numLayers: 1, fineTuneType: type,
                loraParameters: .init(
                    rank: 2, scale: 1.5, keys: ["self_attn.q_proj", "mlp.up_proj"]))
        }

        /// Builds an adapter on `model` with random adapter weights and
        /// writes it to a new folder in the layout of `mlx_lm.lora`.
        static func savedAdapter(
            _ type: LoRAConfiguration.FineTuneType, on model: Qwen3Model
        ) throws -> URL {
            let configuration = loraConfiguration(type)
            _ = try LoRAContainer.from(model: model, configuration: configuration)
            for layer in model.loraLayers {
                for (_, module) in layer.namedModules() where module is LoRALayer {
                    randomizeAdapter(module, magnitude: type == .dora)
                }
            }
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("lora-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try JSONEncoder().encode(configuration)
                .write(to: folder.appendingPathComponent("adapter_config.json"))
            try LoRATrain.saveLoRAWeights(
                model: model, url: folder.appendingPathComponent("adapters.safetensors"))
            return folder
        }

        @Test func containerWrapsOnlyTheLastLayersAndTheGivenKeys() throws {
            let model = try Self.model()
            let container = try LoRAContainer.from(
                model: model, configuration: Self.loraConfiguration(.lora))
            let keys = Set(container.parameters.flattened().map(\.0))
            #expect(
                keys == [
                    "model.layers.1.self_attn.q_proj.lora_a",
                    "model.layers.1.self_attn.q_proj.lora_b",
                    "model.layers.1.mlp.up_proj.lora_a", "model.layers.1.mlp.up_proj.lora_b",
                ])
            let first = Dictionary(uniqueKeysWithValues: model.loraLayers[0].namedModules())
            let last = Dictionary(uniqueKeysWithValues: model.loraLayers[1].namedModules())
            #expect(!(first["self_attn.q_proj"] is LoRALinear))
            #expect(last["self_attn.q_proj"] is LoRALinear)
            #expect(last["mlp.up_proj"] is LoRALinear)
            #expect(!(last["self_attn.k_proj"] is LoRALinear))
        }

        /// An adapter saved from one model and loaded into a second model
        /// with the same base weights gives the same logits. Fusing keeps
        /// them, and unloading gives the base logits back.
        @Test(arguments: [LoRAConfiguration.FineTuneType.lora, .dora])
        func savedAdapterLoadsFusesAndUnloads(type: LoRAConfiguration.FineTuneType) throws {
            let adapted = try Self.model()
            let folder = try Self.savedAdapter(type, on: adapted)
            defer { try? FileManager.default.removeItem(at: folder) }
            let expected = ForwardPassChecks.logits(adapted, Self.tokens)

            let model = try Self.model()
            let baseLogits = ForwardPassChecks.logits(model, Self.tokens)
            #expect(SyntheticModel.maxAbsDifference(expected, baseLogits) > 1e-3)

            let adapter = try LoRAContainer.from(directory: folder)
            #expect(adapter.configuration.fineTuneType == type)
            try model.load(adapter: adapter)
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(model, Self.tokens), expected)
                    <= Self.tolerance)

            model.unload(adapter: adapter)
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(model, Self.tokens), baseLogits) == 0)

            let result = try model.perform(with: adapter) {
                ForwardPassChecks.logits(model, Self.tokens)
            }
            #expect(SyntheticModel.maxAbsDifference(result, expected) <= Self.tolerance)
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(model, Self.tokens), baseLogits) == 0,
                "perform(with:) unloads the adapter")

            try model.fuse(with: adapter)
            let layer = Dictionary(uniqueKeysWithValues: model.loraLayers[1].namedModules())
            #expect(!(layer["self_attn.q_proj"] is LoRALayer))
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(model, Self.tokens), expected)
                    <= Self.tolerance)
        }

        /// Loading the same adapter twice gives the same logits as loading
        /// it once, and unloading then gives the base logits.
        @Test func loadingTheSameAdapterTwiceKeepsTheLogits() throws {
            let adapted = try Self.model()
            let folder = try Self.savedAdapter(.lora, on: adapted)
            defer { try? FileManager.default.removeItem(at: folder) }
            let expected = ForwardPassChecks.logits(adapted, Self.tokens)

            let model = try Self.model()
            let baseLogits = ForwardPassChecks.logits(model, Self.tokens)
            let adapter = try LoRAContainer.from(directory: folder)
            try model.load(adapter: adapter)
            try model.load(adapter: adapter)
            let twice = ForwardPassChecks.logits(model, Self.tokens)
            model.unload(adapter: adapter)
            let unloaded = ForwardPassChecks.logits(model, Self.tokens)
            #expect(SyntheticModel.maxAbsDifference(twice, expected) <= Self.tolerance)
            #expect(SyntheticModel.maxAbsDifference(unloaded, baseLogits) == 0)
        }

        @Test func containerRejectsAnUnknownParameter() throws {
            let model = try Self.model()
            let folder = try Self.savedAdapter(.lora, on: try Self.model())
            defer { try? FileManager.default.removeItem(at: folder) }
            var weights = try MLX.loadArrays(
                url: folder.appendingPathComponent("adapters.safetensors"))
            weights["model.layers.1.self_attn.q_proj.lora_c"] = MLXArray.zeros([2])
            let adapter = LoRAContainer(
                configuration: Self.loraConfiguration(.lora),
                parameters: ModuleParameters.unflattened(weights))
            #expect(throws: (any Error).self) { try model.load(adapter: adapter) }
        }

        @Test func factoryReadsTheAdapterTypeFromTheFolder() async throws {
            let folder = try Self.savedAdapter(.dora, on: try Self.model())
            defer { try? FileManager.default.removeItem(at: folder) }
            let adapter = try await ModelAdapterFactory.shared.load(
                from: UnusedDownloader(), configuration: ModelConfiguration(directory: folder))
            let container = try #require(adapter as? LoRAContainer)
            #expect(container.configuration.fineTuneType == .dora)
            #expect(container.configuration.numLayers == 1)

            // An unknown type is an error that names the type.
            let json = #"{"fine_tune_type": "prefix", "num_layers": 1}"#
            try Data(json.utf8).write(to: folder.appendingPathComponent("adapter_config.json"))
            await #expect(throws: ModelAdapterError.self) {
                _ = try await ModelAdapterFactory.shared.load(
                    from: UnusedDownloader(), configuration: ModelConfiguration(directory: folder))
            }
        }

        @Test func registryCreatesRegisteredTypesOnly() throws {
            let registry = ModelAdapterTypeRegistry()
            let folder = FileManager.default.temporaryDirectory
            #expect(throws: ModelAdapterError.self) {
                _ = try registry.createAdapter(directory: folder, adapterType: "lora")
            }
            registry.registerAdapterType("lora") { _ in
                LoRAContainer(
                    configuration: LoRAConfiguration(), parameters: ModuleParameters())
            }
            let adapter = try registry.createAdapter(directory: folder, adapterType: "lora")
            #expect((adapter as? LoRAContainer)?.configuration.numLayers == 16)
        }

        /// `load(into:)` and `unload(from:)` need a `LoRAModel`.
        @Test func containerRejectsAModelWithoutLoRALayers() throws {
            let adapter = LoRAContainer(
                configuration: LoRAConfiguration(), parameters: ModuleParameters())
            let model = NoLoRAModel()
            #expect(throws: ModelAdapterError.self) { try adapter.load(into: model) }
            #expect(throws: ModelAdapterError.self) {
                _ = try LoRAContainer.from(model: model)
            }
            adapter.unload(from: model)
        }
    }
}

/// A downloader that the directory configuration never calls.
private struct UnusedDownloader: Downloader {
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        Issue.record("a directory configuration must not download")
        return URL(fileURLWithPath: "/")
    }
}

/// A language model that is not a `LoRAModel`.
private final class NoLoRAModel: Module, LanguageModel {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}
