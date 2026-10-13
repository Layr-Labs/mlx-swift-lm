// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// The production load path (`LLMTypeRegistry` + `loadWeights` with the
/// config's per-layer quantization, exactly as `LLMModelFactory` loads) on a
/// tiny checkpoint laid out like LiquidAI/LFM2.5-24B-A2B-MLX-4bit: 4-bit g64
/// projections and experts, 6-bit g64 tied embedding, float32 router gate and
/// expert bias, MLX-layout depthwise conv, no lm_head.
final class LFM2MoELoadingTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("lfm2moe-load-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func configJSON(embedBits: Int = 6) -> Data {
        let types = LFM2MoEFixture.layerTypes.map { "\"\($0)\"" }.joined(separator: ",")
        return Data("""
            {"model_type":"lfm2_moe","vocab_size":96,"hidden_size":256,
             "intermediate_size":128,"moe_intermediate_size":64,"num_hidden_layers":6,
             "num_experts":8,"num_experts_per_tok":2,"norm_topk_prob":true,
             "num_attention_heads":4,"num_key_value_heads":1,"max_position_embeddings":4096,
             "use_expert_bias":true,"num_dense_layers":2,"norm_eps":1e-5,"conv_bias":false,
             "conv_L_cache":3,"rope_parameters":{"rope_theta":10000000.0,"rope_type":"default"},
             "routed_scaling_factor":1.0,"tie_word_embeddings":true,"layer_types":[\(types)],
             "bos_token_id":1,"eos_token_id":[2],
             "quantization":{"group_size":64,"bits":4,"mode":"affine",
               "model.embed_tokens":{"group_size":64,"bits":\(embedBits),"mode":"affine"}}}
            """.utf8)
    }

    /// Writes the source model's tensors (optionally edited) as one or two shards.
    @discardableResult
    private func writeCheckpoint(
        source: LFM2MoEModel, embedBits: Int = 6, shards: Int = 1,
        edit: (inout [String: MLXArray]) -> Void = { _ in }
    ) throws -> [String: MLXArray] {
        var tensors = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        edit(&tensors)
        try configJSON(embedBits: embedBits).write(to: directory.appendingPathComponent("config.json"))
        let keys = tensors.keys.sorted()
        for shard in 0 ..< shards {
            let part = keys.enumerated().filter { $0.offset % shards == shard }.map(\.element)
            try MLX.save(
                arrays: Dictionary(uniqueKeysWithValues: part.map { ($0, tensors[$0]!) }),
                metadata: ["format": "mlx"],
                url: directory.appendingPathComponent(
                    "model-\(String(format: "%05d", shard + 1))-of-\(String(format: "%05d", shards)).safetensors"))
        }
        return tensors
    }

    /// Mirrors `LLMModelFactory._load` minus the tokenizer.
    private func productionLoad() async throws -> LFM2MoEModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)
        try loadWeights(
            modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
        return try XCTUnwrap(model as? LFM2MoEModel)
    }

    private func expectLoadFailure(_ fragment: String, file: StaticString = #filePath, line: UInt = #line) async {
        do {
            _ = try await productionLoad()
            XCTFail("load accepted an invalid checkpoint", file: file, line: line)
        } catch {
            let message = "\(error) \(error.localizedDescription)"
            XCTAssertTrue(
                message.contains(fragment), "diagnostic should name \(fragment): \(message)",
                file: file, line: line)
        }
    }

    func testMixedQuantizationFloat32RouterAndTiedQuantizedHeadLoadExactly() async throws {
        let source = try LFM2MoEFixture.model(.quantizedMixed)
        let saved = try writeCheckpoint(source: source)
        XCTAssertNil(saved["lm_head.weight"], "tied head: the checkpoint has no lm_head")
        XCTAssertNil(saved["model.layers.2.feed_forward.gate.scales"], "router gate is unquantized")

        let model = try await productionLoad()
        let embedding = try XCTUnwrap(model.model.embedTokens as? QuantizedEmbedding)
        XCTAssertEqual(embedding.bits, 6)
        XCTAssertEqual(embedding.groupSize, 64)
        for (index, layer) in model.model.layers.enumerated() {
            if let conv = layer.conv {
                let inProj = try XCTUnwrap(conv.inProj as? QuantizedLinear)
                XCTAssertEqual(inProj.bits, 4)
                XCTAssertEqual(inProj.groupSize, 64)
                XCTAssertEqual(conv.conv.weight.shape, [256, 3, 1])
                XCTAssertEqual(conv.conv.weight.dtype, .bfloat16)
            }
            if let moe = layer.feedForward as? Lfm2MoeSparseMoeBlock {
                XCTAssertFalse(moe.gate is QuantizedLinear, "layer \(index) router gate must stay dense")
                XCTAssertEqual(moe.gate.weight.dtype, .float32, "layer \(index) gate narrowed")
                XCTAssertEqual(moe.expertBias?.dtype, .float32, "layer \(index) expert_bias narrowed")
                XCTAssertEqual(
                    moe.expertBias?.asArray(Float.self),
                    saved["model.layers.\(index).feed_forward.expert_bias"]?.asArray(Float.self))
                XCTAssertTrue(moe.switchMLP.gateProj is QuantizedSwitchLinear)
            } else {
                XCTAssertTrue(index < 2, "only the first two layers are dense")
            }
        }

        // Loaded parameters are the saved ones, bit for bit.
        let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(Set(loaded.keys), Set(saved.keys))
        for (key, value) in saved {
            XCTAssertEqual(loaded[key]?.dtype, value.dtype, key)
            XCTAssertEqual(loaded[key]?.asData(access: .copy).data, value.asData(access: .copy).data, key)
        }

        // The tied head is the 6-bit embedding: logits == hidden @ dequant(E)^T.
        let tokens = MLXArray(LFM2MoEFixture.prompt(12).map(Int32.init)).reshaped(1, 12)
        let hidden = model.model(tokens, cache: nil)
        let logits = model(tokens, cache: nil)
        let table = dequantized(
            embedding.weight, scales: embedding.scales, biases: embedding.biases,
            groupSize: 64, bits: 6)
        let explicit = matmul(hidden.asType(.float32), table.asType(.float32).transposed())
        XCTAssertLessThan(
            abs(logits.asType(.float32) - explicit).max().item(Float.self),
            0.05, "head must be the dequantized 6-bit embedding")
        XCTAssertEqual(
            logits.asData(access: .copy).data, source(tokens, cache: nil).asData(access: .copy).data,
            "loaded model reproduces the source model exactly")
    }

    func testMissingTensorIsRejectedWithItsName() async throws {
        let source = try LFM2MoEFixture.model(.quantizedMixed)
        try writeCheckpoint(source: source) { $0.removeValue(forKey: "model.layers.3.feed_forward.expert_bias") }
        await expectLoadFailure("expert_bias")
    }

    func testMissingShardIsRejected() async throws {
        let source = try LFM2MoEFixture.model(.quantizedMixed)
        try writeCheckpoint(source: source, shards: 2)
        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("model-00002-of-00002.safetensors"))
        do {
            _ = try await productionLoad()
            XCTFail("load accepted a checkpoint with a missing shard")
        } catch {
            // Whichever tensor the update reaches first (a missing scale makes
            // the embedding look unquantized), the load must refuse.
            XCTAssertTrue(error is UpdateError, "\(error)")
        }
    }

    func testExtraTensorIsRejectedWithItsName() async throws {
        let source = try LFM2MoEFixture.model(.quantizedMixed)
        try writeCheckpoint(source: source) {
            // An untied output head is not part of this architecture.
            $0["lm_head.weight"] = MLXArray.zeros([96, 256], dtype: .bfloat16)
        }
        await expectLoadFailure("lm_head")
    }

    func testShapeMismatchIsRejected() async throws {
        let source = try LFM2MoEFixture.model(.quantizedMixed)
        try writeCheckpoint(source: source) {
            $0["model.layers.0.conv.conv.weight"] = MLXArray.zeros([256, 4, 1], dtype: .bfloat16)
        }
        await expectLoadFailure("conv")
    }

    func testQuantizationPolicyMismatchIsRejected() async throws {
        // Tensors stored at 6-bit, config claims 4-bit: packed widths disagree.
        let source = try LFM2MoEFixture.model(.quantizedMixed)
        try writeCheckpoint(source: source, embedBits: 4)
        await expectLoadFailure("embed_tokens")
    }
}
