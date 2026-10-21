import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// CPU metadata fixtures from the publisher's 9B MLX sidecar at
/// b94a52f3ce9047919dce00392a0fc6b8a7968abe. No model tensors are allocated.
extension UnitTests {
    @Suite(.serialized)
    struct PrismMTPBindingTests {
        private func text(layers: Int = 0) -> [String: Any] {
            [
                "model_type": "qwen3_5_text", "hidden_size": 4096,
                "num_hidden_layers": 32, "intermediate_size": 12288,
                "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
                "vocab_size": 248320, "full_attention_interval": 4,
                "linear_num_value_heads": 32, "linear_num_key_heads": 16,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6, "tie_word_embeddings": false, "attention_bias": false,
                "mtp_num_hidden_layers": layers, "mtp_use_dedicated_embeddings": false,
                "attn_output_gate": true, "hidden_act": "silu", "mamba_ssm_dtype": "float32",
                "rope_parameters": [
                    "mrope_interleaved": true, "mrope_section": [11, 11, 10],
                    "rope_type": "default", "rope_theta": 10_000_000,
                    "partial_rotary_factor": 0.25,
                ],
            ]
        }

        private func config(_ value: [String: Any]) throws -> Qwen35TextConfiguration {
            try JSONDecoder().decode(
                Qwen35TextConfiguration.self,
                from: JSONSerialization.data(withJSONObject: value))
        }

        private func metadata(
            _ value: [String: Any], blockSize: Int = 2,
            prefix: String = "",
            quantization: BaseConfiguration.PerLayerQuantization? = nil
        ) throws
            -> Qwen35InlineMTPMetadata
        {
            .init(
                textConfiguration: try config(value), prefix: prefix,
                blockSize: blockSize, quantization: quantization)
        }

        private func validateText(_ artifact: [String: Any], _ target: [String: Any]) throws {
            try Qwen35InlineMTPAssistant.validatePrism9BTextConfiguration(
                artifactData: JSONSerialization.data(withJSONObject: [
                    "model_type": "qwen3_5_mtp", "block_size": 2, "text_config": artifact,
                ]),
                targetData: JSONSerialization.data(withJSONObject: [
                    "model_type": "prism_hadamard_qwen35", "text_config": target,
                ]))
        }

        @Test
        func testDerivedOneLayerSidecarMatchesUnmodifiedBase() throws {
            let target = text()
            let artifact = text(layers: 1)
            try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                metadata(artifact), against: config(target))
            try validateText(artifact, target)
            #expect((target["mtp_num_hidden_layers"] as? Int) == (0))
        }

        @Test
        func testOriginalZeroLayerConfigIsRejectedBeforeWeights() throws {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("prism-mtp-metadata-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            try JSONSerialization.data(withJSONObject: [
                "model_type": "qwen3_5_mtp", "block_size": 2, "text_config": text(),
            ]).write(to: directory.appendingPathComponent("config.json"))
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
            }
            #expect(throws: (any Error).self) { try validateText(text(), text()) }
        }

        @Test
        func testRejects27BAndOtherPrismGeometryEvenWhenSidecarMatches() throws {
            for (key, value) in [
                ("hidden_size", 5120), ("num_hidden_layers", 64), ("intermediate_size", 17408),
                ("num_attention_heads", 24), ("num_key_value_heads", 8), ("head_dim", 128),
                ("vocab_size", 151936), ("linear_num_value_heads", 48),
                ("linear_key_head_dim", 192), ("full_attention_interval", 8), ("num_experts", 8),
            ] {
                var target = text()
                target[key] = value
                var artifact = target
                artifact["mtp_num_hidden_layers"] = 1
                #expect(throws: (any Error).self, Comment(rawValue: key)) {
                    try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                        metadata(artifact), against: config(target))
                }
            }
        }

        @Test
        func testRejectsInlineIncompleteQuantizationOrDeeperSidecar() throws {
            let target = try config(text())
            for count in [0, 2, 4] {
                #expect(throws: (any Error).self) {
                    try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                        metadata(text(layers: count)), against: target)
                }
            }
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                    metadata(text(layers: 1), blockSize: 3), against: target)
            }
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                    metadata(text(layers: 1), prefix: "mtp."), against: target)
            }
            let quantization = BaseConfiguration.PerLayerQuantization(
                quantization: nil, perLayerQuantization: [:])
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                    metadata(text(layers: 1), quantization: quantization), against: target)
            }
        }

        @Test
        func testPublishedQ4MetadataAndTensorInventory() throws {
            // Header from f4e28a59531731c41b6dfac88cf271bd58f0a287.
            let quantization = BaseConfiguration.PerLayerQuantization(
                quantization: .init(groupSize: 64, bits: 4), perLayerQuantization: [:])
            try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                metadata(text(layers: 1), quantization: quantization), against: config(text()))
            // Explicit packed widths and scale widths from the published header.
            let matrices: [(String, Int, Int, Int)] = [
                ("fc", 4096, 1024, 128),
                ("layers.0.self_attn.q_proj", 8192, 512, 64),
                ("layers.0.self_attn.k_proj", 1024, 512, 64),
                ("layers.0.self_attn.v_proj", 1024, 512, 64),
                ("layers.0.self_attn.o_proj", 4096, 512, 64),
                ("layers.0.mlp.gate_proj", 12288, 512, 64),
                ("layers.0.mlp.up_proj", 12288, 512, 64),
                ("layers.0.mlp.down_proj", 4096, 1536, 192),
            ]
            var weights: [String: (shape: [Int], dtype: DType)] = [:]
            for (path, rows, packed, groups) in matrices {
                weights[path + ".weight"] = ([rows, packed], .uint32)
                weights[path + ".scales"] = ([rows, groups], .float16)
                weights[path + ".biases"] = ([rows, groups], .float16)
            }
            for name in [
                "norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden",
                "layers.0.input_layernorm", "layers.0.post_attention_layernorm",
            ] {
                weights[name + ".weight"] = ([4096], .float16)
            }
            for name in ["q_norm", "k_norm"] {
                weights["layers.0.self_attn." + name + ".weight"] = ([256], .float16)
            }
            try Qwen35InlineMTPAssistant.validatePrism9BWeights(weights, quantization: quantization)
            #expect((weights.count) == (31))
            #expect(
                (weights.values.reduce(0) { $0 + $1.shape.reduce(1, *) * $1.dtype.size })
                    == (136_881_152))
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BWeights(weights)
            }
            for name in weights.keys {
                var missing = weights
                missing.removeValue(forKey: name)
                #expect(throws: (any Error).self, Comment(rawValue: name)) {
                    try Qwen35InlineMTPAssistant.validatePrism9BWeights(
                        missing, quantization: quantization)
                }
            }
            for (name, invalid) in [
                ("fc.weight", ([4096, 8192], DType.float16)),
                ("fc.scales", ([4096, 64], DType.float16)),
                ("fc.biases", ([4096, 128], DType.float32)),
                ("norm.weight", ([4096], DType.float32)),
                ("embed_tokens.weight", ([248320, 512], DType.uint32)),
            ] {
                var changed = weights
                changed[name] = invalid
                #expect(throws: (any Error).self, Comment(rawValue: name)) {
                    try Qwen35InlineMTPAssistant.validatePrism9BWeights(
                        changed, quantization: quantization)
                }
            }
        }

        @Test
        func testRejectsUnpublishedQuantizationAndConflictingDeclarations() throws {
            for global in [
                BaseConfiguration.Quantization(groupSize: 128, bits: 4),
                .init(groupSize: 64, bits: 8), .init(groupSize: 64, bits: 4, mode: .mxfp4),
            ] {
                #expect(throws: (any Error).self) {
                    try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                        metadata(
                            text(layers: 1),
                            quantization: .init(
                                quantization: global, perLayerQuantization: [:])),
                        against: config(text()))
                }
            }
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BMetadata(
                    metadata(
                        text(layers: 1),
                        quantization: .init(
                            quantization: .init(groupSize: 64, bits: 4),
                            perLayerQuantization: ["fc": .skip])),
                    against: config(text()))
            }
            let target = try JSONSerialization.data(withJSONObject: [
                "model_type": "prism_hadamard_qwen35", "text_config": text(),
            ])
            var artifact: [String: Any] = [
                "model_type": "qwen3_5_mtp", "text_config": text(layers: 1),
                "quantization": ["group_size": 64, "bits": 4],
                "quantization_config": ["group_size": 64, "bits": 4],
            ]
            try Qwen35InlineMTPAssistant.validatePrism9BTextConfiguration(
                artifactData: JSONSerialization.data(withJSONObject: artifact), targetData: target)
            artifact["quantization_config"] = ["group_size": 64, "bits": 8]
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BTextConfiguration(
                    artifactData: JSONSerialization.data(withJSONObject: artifact),
                    targetData: target)
            }
        }

        @Test
        func testRejectsChangedRopeNormAndUntypedSemanticFields() throws {
            let target = text()
            for (key, value) in [
                ("rms_norm_eps", 1e-5), ("tie_word_embeddings", true),
                ("mtp_use_dedicated_embeddings", true), ("attn_output_gate", false),
                ("hidden_act", "gelu"), ("mamba_ssm_dtype", "float16"),
                ("rope_parameters", ["rope_type": "default", "rope_theta": 1_000_000]),
                ("unexpected_training_semantics", true),
            ] as [(String, Any)] {
                var artifact = text(layers: 1)
                artifact[key] = value
                #expect(throws: (any Error).self, Comment(rawValue: key)) {
                    try validateText(artifact, target)
                }
            }
        }

        @Test
        func testPublishedTensorSchemaHasExactF16Coverage() throws {
            // Header-derived independent inventory: eight matrices plus seven norms.
            let matrices: [String: [Int]] = [
                "fc.weight": [4096, 8192],
                "layers.0.self_attn.q_proj.weight": [8192, 4096],
                "layers.0.self_attn.k_proj.weight": [1024, 4096],
                "layers.0.self_attn.v_proj.weight": [1024, 4096],
                "layers.0.self_attn.o_proj.weight": [4096, 4096],
                "layers.0.mlp.gate_proj.weight": [12288, 4096],
                "layers.0.mlp.up_proj.weight": [12288, 4096],
                "layers.0.mlp.down_proj.weight": [4096, 12288],
            ]
            var weights = matrices.mapValues { (shape: $0, dtype: DType.float16) }
            for name in [
                "norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden",
                "layers.0.input_layernorm", "layers.0.post_attention_layernorm",
            ] {
                weights[name + ".weight"] = ([4096], .float16)
            }
            for name in ["q_norm", "k_norm"] {
                weights["layers.0.self_attn." + name + ".weight"] = ([256], .float16)
            }
            try Qwen35InlineMTPAssistant.validatePrism9BWeights(weights)
            #expect((weights.count) == (15))
            // Typed in steps: CI's compiler cannot type-check the inline form in time.
            var byteCount = 0
            for entry in weights.values { byteCount += entry.shape.reduce(1, *) * 2 }
            #expect(byteCount == 486_581_248)
            var changed = weights
            changed.removeValue(forKey: "norm.weight")
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BWeights(changed)
            }
            changed = weights
            changed["embed_tokens.weight"] = ([248320, 4096], .float16)
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BWeights(changed)
            }
            changed = weights
            changed["layers.0.self_attn.q_proj.weight"] = ([4096, 8192], .float16)
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BWeights(changed)
            }
            changed = weights
            changed["norm.weight"] = ([4096], .float32)
            #expect(throws: (any Error).self) {
                try Qwen35InlineMTPAssistant.validatePrism9BWeights(changed)
            }
        }

        @Test
        func testPrismShortlistCannotBeEnabledByEnvironmentRequest() {
            for requested in [nil, 256, 16384] as [Int?] {
                #expect(
                    (Qwen35InlineMTPAssistant.resolvedDraftShortlistSize(
                        requested: requested, allowsShortlist: false, forceDoubleForward: false))
                        == nil)
            }
            #expect(
                (Qwen35InlineMTPAssistant.resolvedDraftShortlistSize(
                    requested: 256, allowsShortlist: true, forceDoubleForward: false)) == (256))
            #expect(
                (Qwen35InlineMTPAssistant.resolvedDraftShortlistSize(
                    requested: 256, allowsShortlist: true, forceDoubleForward: true)) == nil)
        }

        @Test
        func testFP32StateAdmissionIncludesHeadKVHiddenAndToken() throws {
            #expect(
                (Qwen35InlineMTPAssistant.stateBytesPerToken(
                    configuration: try config(text()), layerCount: 1,
                    cacheElementBytes: DType.float32.size, hiddenElementBytes: DType.float32.size))
                    == (24580))
        }
    }
}
