// Copyright © 2026 Eigen Labs.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLLM

extension UnitTests {

    /// Configuration decoding, the artifact inventory and the block mask of
    /// the Muse Glimmer DFlash drafter. Host logic only.
    struct MuseGlimmerDFlashConfigurationTests {
        typealias Configuration = MuseGlimmerDFlashConfiguration
        typealias State = MuseGlimmerDFlashContextState

        /// The published `config.json` of revision `c676cce9`.
        static let published = """
            {"architectures": ["MuseGlimmerAssistantModel"], "attention_dropout": 0,
             "block_size": 16, "bos_token_id": 200000, "dtype": "bfloat16",
             "eos_token_id": 200001, "head_dim": 128, "hidden_act": "silu",
             "hidden_size": 6656, "intermediate_size": 19968,
             "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                             "sliding_attention", "sliding_attention"],
             "mask_token_id": 201818, "max_position_embeddings": 131072,
             "model_type": "muse_glimmer_assistant", "num_attention_heads": 32,
             "num_hidden_layers": 5, "num_key_value_heads": 8, "pad_token_id": 200018,
             "rms_norm_eps": 1e-05,
             "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
             "sliding_window": 2048, "target_layer_ids": [1, 13, 25, 37, 49],
             "transformers_version": "5.15.0.dev0"}
            """

        /// Only the required keys, each with a different value.
        static var required: [String: Any] {
            [
                "model_type": "muse_glimmer_assistant", "hidden_size": 48,
                "intermediate_size": 96, "num_hidden_layers": 3, "num_attention_heads": 6,
                "num_key_value_heads": 2, "head_dim": 8, "block_size": 5, "mask_token_id": 17,
                "target_layer_ids": [4, 9], "sliding_window": 31, "rms_norm_eps": 0.25,
                "rope_parameters": ["rope_theta": 1234.0],
            ]
        }

        private func decode(_ object: [String: Any]) throws -> Configuration {
            try JSONDecoder().decode(
                Configuration.self, from: JSONSerialization.data(withJSONObject: object))
        }

        @Test func publishedConfigurationDecodesToThePinnedValues() throws {
            let decoded = try JSONDecoder().decode(
                Configuration.self, from: Data(Self.published.utf8))
            #expect(decoded == .museGlimmer30B)
            #expect(decoded.contextWidth == 33_280 && decoded.draftTokensPerBlock == 15)
            let again = try JSONDecoder().decode(
                Configuration.self, from: JSONEncoder().encode(decoded))
            #expect(again == decoded)
        }

        @Test func requiredKeysDecodeAndDefaultsApply() throws {
            let c = try decode(Self.required)
            #expect(c.hiddenSize == 48 && c.intermediateSize == 96 && c.hiddenLayers == 3)
            #expect(c.attentionHeads == 6 && c.kvHeads == 2 && c.headDimensions == 8)
            #expect(c.blockSize == 5 && c.maskTokenID == 17 && c.targetLayerIDs == [4, 9])
            #expect(c.slidingWindow == 31 && c.rmsNormEps == 0.25 && c.ropeTheta == 1234)
            #expect(c.maxPositionEmbeddings == 131_072 && c.contextWidth == 96)
        }

        @Test func everyRequiredKeyIsRequired() {
            for key in Self.required.keys {
                var object = Self.required
                object.removeValue(forKey: key)
                #expect(throws: (any Error).self, "\(key)") { try decode(object) }
            }
        }

        @Test func unsupportedValuesAreRefused() {
            let cases: [(String, Any)] = [
                ("model_type", "muse_glimmer"), ("hidden_act", "gelu"),
                ("layer_types", ["sliding_attention", "full_attention", "sliding_attention"]),
                ("layer_types", ["sliding_attention"]),
                ("rope_parameters", ["rope_theta": 1234.0, "rope_type": "yarn"]),
                ("num_key_value_heads", 4), ("head_dim", 7), ("block_size", 1),
                ("sliding_window", 4), ("target_layer_ids", [9, 4]),
                ("target_layer_ids", [Int]()), ("rms_norm_eps", 0), ("num_hidden_layers", 0),
                ("mask_token_id", -1),
            ]
            for (key, value) in cases {
                var object = Self.required
                object[key] = value
                #expect(throws: MuseGlimmerDFlashError.self, "\(key) = \(value)") {
                    try decode(object)
                }
            }
        }

        // MARK: Inventory

        private func tensor(_ dtype: String, _ shape: [Int]) -> ModelOptProjectionLayout.Tensor {
            let width = dtype == "U8" ? 1 : dtype == "F32" ? 4 : 2
            return .init(dtype: dtype, shape: shape, byteCount: shape.reduce(width, *))
        }

        /// A dense header, or one with `quantized` projections at `bits`.
        private func header(
            _ c: Configuration, quantized: [String: Int] = [:], groupSize: Int = 32
        ) -> [String: ModelOptProjectionLayout.Tensor] {
            let inventory = MuseGlimmerDFlashInventory(c)
            var header = inventory.norms.mapValues { tensor("BF16", [$0]) }
            for p in inventory.projections {
                if let bits = quantized[p.path] {
                    header[p.path + ".weight"] = tensor(
                        "U8", [p.outputDimensions, p.inputDimensions * bits / 8])
                    for name in ["scales", "biases"] {
                        header[p.path + "." + name] = tensor(
                            "BF16", [p.outputDimensions, p.inputDimensions / groupSize])
                    }
                } else {
                    header[p.path + ".weight"] = tensor(
                        "BF16", [p.outputDimensions, p.inputDimensions])
                }
            }
            return header
        }

        @Test func publishedInventoryHasFiftyEightTensors() throws {
            let c = Configuration.museGlimmer30B
            let inventory = MuseGlimmerDFlashInventory(c)
            #expect(inventory.projections.count == 36 && inventory.norms.count == 22)
            let dense = header(c)
            #expect(dense.count == 58)
            #expect(dense.values.reduce(0) { $0 + $1.byteCount } == 5_111_970_304)
            let formats = try inventory.formats(header: dense)
            #expect(formats.denseDType == "BF16")
            #expect(
                formats.formats.count == 36 && formats.formats.values.allSatisfy { $0 == .dense })
        }

        @Test func formatsComeFromTheHeader() throws {
            let c = Configuration.museGlimmer30B
            let inventory = MuseGlimmerDFlashInventory(c)
            var bits: [String: Int] = [:]
            for p in inventory.projections { bits[p.path] = p.path == "encoder.fc" ? 8 : 4 }
            let formats = try inventory.formats(header: header(c, quantized: bits, groupSize: 64))
            #expect(formats.formats["encoder.fc"] == .affine(bits: 8, groupSize: 64))
            #expect(
                formats.formats["layers.4.mlp.down_proj"] == .affine(bits: 4, groupSize: 64))
            #expect(
                MuseGlimmerDFlashProjectionFormat.affine(bits: 4, groupSize: 64).bitsPerWeight
                    == 4.5)
        }

        @Test func inventoryRefusesMissingExtraAndMisshapenTensors() throws {
            let c = try decode(Self.required)
            let inventory = MuseGlimmerDFlashInventory(c)
            let good = header(c)
            _ = try inventory.formats(header: good)

            var missing = good
            missing.removeValue(forKey: "layers.1.self_attn.k_norm.weight")
            var extra = good
            extra["layers.3.input_layernorm.weight"] = tensor("BF16", [48])
            var embedding = good
            embedding["embed_tokens.weight"] = tensor("BF16", [8, 48])
            var misshapen = good
            misshapen["layers.0.mlp.gate_proj.weight"] = tensor("BF16", [48, 96])
            var mixed = good
            mixed["norm.weight"] = tensor("F32", [48])
            var lonelyScales = good
            lonelyScales["encoder.fc.scales"] = tensor("BF16", [48, 3])
            var oddBits = header(c, quantized: ["encoder.fc": 8])
            oddBits["encoder.fc.weight"] = tensor("U8", [48, 96 * 7 / 8])
            var noBiases = header(c, quantized: ["encoder.fc": 4])
            noBiases.removeValue(forKey: "encoder.fc.biases")
            var badGroup = header(c, quantized: ["encoder.fc": 4])
            for name in ["scales", "biases"] {
                badGroup["encoder.fc." + name] = tensor("BF16", [48, 96 / 16])
            }
            for (name, header) in [
                ("missing", missing), ("extra", extra), ("embedding", embedding),
                ("misshapen", misshapen), ("mixed", mixed), ("lonelyScales", lonelyScales),
                ("oddBits", oddBits), ("noBiases", noBiases), ("badGroup", badGroup),
            ] {
                #expect(throws: MuseGlimmerDFlashError.self, "\(name)") {
                    try inventory.formats(header: header)
                }
            }
        }

        // MARK: Block mask

        /// No mask until the last block query is more than `window` past the
        /// oldest row; then query `i` loses the rows more than `window`
        /// behind it and keeps the whole block.
        @Test func visibilityFollowsTheReferenceMask() {
            let (window, block) = (2048, 16)
            for count in [1, 100, 2033] {
                #expect(
                    State.visibility(
                        anchor: count, blockSize: block, window: window,
                        contextPositions: Array(0 ..< count)) == nil)
            }
            for (anchor, first) in [(2034, 0), (2048, 0), (5000, 5000 - 2048), (2047, 1)] {
                let positions = Array(first ..< anchor)
                let needsMask = anchor + block - 1 - first > window
                let mask = State.visibility(
                    anchor: anchor, blockSize: block, window: window, contextPositions: positions)
                #expect((mask != nil) == needsMask)
                guard let mask else { continue }
                let width = positions.count + block
                #expect(mask.count == block * width)
                for query in 0 ..< block {
                    for (column, position) in positions.enumerated() {
                        #expect(
                            mask[query * width + column]
                                == (abs(anchor + query - position) <= window))
                    }
                    #expect(
                        mask[(query * width + positions.count) ..< ((query + 1) * width)]
                            .allSatisfy { $0 })
                }
            }
            // A full ring: query 0 sees all 2,048 rows, query i loses i.
            let full = State.visibility(
                anchor: 9000, blockSize: block, window: window,
                contextPositions: Array((9000 - window) ..< 9000))!
            for query in 0 ..< block {
                let row = full[(query * (window + block)) ..< (query * (window + block) + window)]
                #expect(row.filter { $0 }.count == window - query)
            }
        }

        /// Storage order does not matter: the mask follows each row's position.
        @Test func visibilityUsesPositionsNotStorageOrder() {
            let positions = [8, 9, 10, 11, 4, 5, 6, 7]
            let mask = State.visibility(
                anchor: 12, blockSize: 4, window: 8, contextPositions: positions)!
            let width = positions.count + 4
            for query in 0 ..< 4 {
                for (column, position) in positions.enumerated() {
                    #expect(mask[query * width + column] == (12 + query - position <= 8))
                }
            }
        }
    }
}
