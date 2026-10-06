import Foundation

#if !MIMO_CONFIG_CPU_PROBE
    import MLXLLM
    import XCTest
#endif

// These same assertions run under XCTest or a Foundation-only swiftc probe.
// The probe compiles only this file and MiMoV26Configuration.swift; no MLX load.
private enum MiMoV26ConfigurationChecks {
    struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    static func expect(_ condition: Bool, _ message: String) throws {
        if !condition { throw Failure(description: message) }
    }

    static func tiny() throws -> MiMoV26Configuration {
        try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: Data(
                """
                {
                  "model_type":"mimo_v2", "architectures":["MiMoV2ForCausalLM"],
                  "hidden_size":16, "intermediate_size":32, "moe_intermediate_size":8,
                  "vocab_size":128, "num_hidden_layers":2, "max_position_embeddings":8192,
                  "sliding_window_size":7, "sliding_window":7, "num_nextn_predict_layers":3,
                  "hybrid_layer_pattern":[0,1], "moe_layer_freq":[0,1],
                  "partial_rotary_factor":0.5, "attention_value_scale":0.707,
                  "layernorm_epsilon":0.000001, "attention_projection_layout":"split",
                  "moe_router_dtype":"bfloat16", "hidden_act":"silu", "dtype":"bfloat16",
                  "attention_bias":false, "tie_word_embeddings":false, "attention_dropout":0,
                  "scoring_func":"sigmoid", "topk_method":"noaux_tc", "n_routed_experts":4,
                  "num_experts_per_tok":2, "n_group":1, "topk_group":1,
                  "norm_topk_prob":true, "n_shared_experts":null, "routed_scaling_factor":null,
                  "num_attention_heads":4, "num_key_value_heads":1, "head_dim":8, "v_head_dim":4,
                  "swa_num_attention_heads":4, "swa_num_key_value_heads":2,
                  "swa_head_dim":8, "swa_v_head_dim":4, "rope_theta":10000000,
                  "swa_rope_theta":10000, "add_full_attention_sink_bias":false,
                  "add_swa_attention_sink_bias":true, "eos_token_id":3, "pad_token_id":0,
                  "bos_token_id":null
                }
                """.utf8))
    }

    static func decode(_ raw: [String: MiMoV26JSONValue]) throws -> MiMoV26Configuration {
        try JSONDecoder().decode(MiMoV26Configuration.self, from: JSONEncoder().encode(raw))
    }

    static func roundtrip(_ config: MiMoV26Configuration) throws {
        let data = try JSONEncoder().encode(config)
        let tree = try JSONDecoder().decode([String: MiMoV26JSONValue].self, from: data)
        try expect(tree == config.rawFields, "semantic JSON was changed by encoding")
        try expect(
            try JSONDecoder().decode(MiMoV26Configuration.self, from: data) == config,
            "typed configuration changed during roundtrip")
    }

    static func rejects(_ raw: [String: MiMoV26JSONValue], field: String? = nil) throws {
        do {
            _ = try decode(raw)
            throw Failure(description: "invalid configuration was accepted: \(field ?? "unknown")")
        } catch let error as MiMoV26ConfigurationError {
            if let field {
                guard case .invalid(let actual, _) = error else { return }
                try expect(actual.contains(field), "wrong error field \(actual), expected \(field)")
            }
        }
    }

    static func pureCases() -> [(String, () throws -> Void)] {
        [
            (
                "tiny geometry and bounds",
                {
                    let config = try tiny()
                    let full = try config.attentionGeometry(at: 0)
                    let sliding = try config.attentionGeometry(at: 1)
                    try expect(
                        full.headDim == 8 && full.valueHeadDim == 4 && full.rotaryDimensions == 4,
                        "unequal K/V geometry was lost")
                    try expect(
                        full.keyValueHeads == 1 && full.slidingWindow == nil && !full.hasSinks,
                        "full-attention geometry is wrong")
                    try expect(
                        sliding.keyValueHeads == 2 && sliding.slidingWindow == 7
                            && sliding.hasSinks,
                        "sliding-attention geometry is wrong")
                    for index in [-1, 2] {
                        do {
                            _ = try config.attentionGeometry(at: index)
                            throw Failure(description: "out-of-range layer accepted")
                        } catch is MiMoV26ConfigurationError {}
                    }
                    try roundtrip(config)
                }
            ),
            (
                "unknown nested values preserve integer precision and null",
                {
                    var fields = try tiny().rawFields
                    let extra = try JSONDecoder().decode(
                        MiMoV26JSONValue.self,
                        from: Data(
                            """
                            {"null":null,"bool":true,"text":"keep","large":9007199254740993,
                             "fraction":0.1234567890123456789012345678,"array":[false,0,"x",{}],
                             "uint64":18446744073709551615}
                            """.utf8))
                    fields["future_semantics"] = extra
                    fields["rope_parameters"] = .object([
                        "rope_type": .string("default"), "type": .string("default"),
                        "rope_theta": .number(10_000_000), "partial_rotary_factor": .number(0.5),
                        "future": extra,
                    ])
                    let config = try decode(fields)
                    try roundtrip(config)
                    guard case .object(let extraFields) = config.rawFields["future_semantics"]
                    else {
                        throw Failure(description: "missing unknown object")
                    }
                    try expect(
                        extraFields["large"] == .number(Decimal(string: "9007199254740993")!),
                        "large integer lost precision")
                    try expect(
                        extraFields["uint64"] == .number(Decimal(string: "18446744073709551615")!),
                        "unsigned integer lost precision")
                }
            ),
            (
                "nondefault operational values roundtrip",
                {
                    var fields = try tiny().rawFields
                    fields["attention_bias"] = .bool(true)
                    fields["tie_word_embeddings"] = .bool(true)
                    fields["attention_dropout"] = .number(0.125)
                    fields["attention_value_scale"] = .number(0.625)
                    fields["moe_router_dtype"] = .string("float32")
                    fields["dtype"] = .string("float32")
                    fields["routed_scaling_factor"] = .number(1.5)
                    fields["n_shared_experts"] = .number(2)
                    fields["max_position_embeddings"] = .number(12345)
                    let config = try decode(fields)
                    try expect(
                        config.attentionBias && config.tieWordEmbeddings
                            && config.attentionDropout == 0.125
                            && config.attentionValueScale == 0.625
                            && config.routedScalingFactor == 1.5
                            && config.sharedExpertCount == 2
                            && config.maxPositionEmbeddings == 12345,
                        "nondefault typed semantics were replaced")
                    try roundtrip(config)
                }
            ),
            (
                "wrong model and architecture fail recoverably",
                {
                    for type in ["mimo", "mimo_v2_flash", "qwen3", "MIMO_V2"] {
                        var fields = try tiny().rawFields
                        fields["model_type"] = .string(type)
                        try rejects(fields, field: "model_type")
                    }
                    var fields = try tiny().rawFields
                    fields["architectures"] = .array([.string("DFlashDraftModel")])
                    try rejects(fields, field: "architectures")
                }
            ),
            (
                "required semantics are not silently defaulted",
                {
                    for key in [
                        "model_type", "attention_value_scale", "moe_router_dtype", "v_head_dim",
                        "add_swa_attention_sink_bias", "num_nextn_predict_layers",
                        "hybrid_layer_pattern",
                        "eos_token_id",
                    ] {
                        var fields = try tiny().rawFields
                        fields.removeValue(forKey: key)
                        try rejects(fields, field: key)
                    }
                }
            ),
            (
                "JSON type confusion is rejected",
                {
                    for value in [MiMoV26JSONValue.bool(true), .string("16"), .number(16.5), .null]
                    {
                        var fields = try tiny().rawFields
                        fields["hidden_size"] = value
                        try rejects(fields, field: "hidden_size")
                    }
                    var fields = try tiny().rawFields
                    fields["attention_bias"] = .number(0)
                    try rejects(fields, field: "attention_bias")
                }
            ),
            (
                "malformed layer and expert geometry is rejected",
                {
                    let changes: [(String, MiMoV26JSONValue)] = [
                        ("num_hidden_layers", .number(0)),
                        ("hybrid_layer_pattern", .array([.number(0)])),
                        ("moe_layer_freq", .array([.number(0), .number(2)])),
                        ("num_key_value_heads", .number(3)),
                        ("swa_num_key_value_heads", .number(3)),
                        ("n_group", .number(3)), ("topk_group", .number(2)),
                        ("num_experts_per_tok", .number(5)), ("n_routed_experts", .number(1)),
                        ("sliding_window", .number(8)), ("v_head_dim", .number(0)),
                        ("max_position_embeddings", .number(0)), ("layernorm_epsilon", .number(0)),
                        ("attention_dropout", .number(1)), ("attention_value_scale", .number(-1)),
                    ]
                    for (key, value) in changes {
                        var fields = try tiny().rawFields
                        fields[key] = value
                        try rejects(fields)
                    }
                }
            ),
            (
                "RoPE truncation and conflicting nested semantics",
                {
                    for factor: Decimal in [0, 0.125, 0.375, 1.1] {
                        var fields = try tiny().rawFields
                        fields["partial_rotary_factor"] = .number(factor)
                        try rejects(fields, field: "partial_rotary_factor")
                    }
                    var fields = try tiny().rawFields
                    fields["rope_parameters"] = .object([
                        "rope_type": .string("default"), "rope_theta": .number(2),
                        "partial_rotary_factor": .number(0.5),
                    ])
                    try rejects(fields, field: "rope_parameters")
                }
            ),
            (
                "integer overflow is rejected before allocation",
                {
                    var fields = try tiny().rawFields
                    fields["hidden_size"] = .number(Decimal(Int.max))
                    try rejects(fields, field: "dense MLP")
                    fields = try tiny().rawFields
                    fields["head_dim"] = .number(Decimal(Int.max))
                    try rejects(fields)
                    fields = try tiny().rawFields
                    fields["hidden_size"] = .number(Decimal(string: "18446744073709551615")!)
                    try rejects(fields, field: "hidden_size")
                }
            ),
            (
                "native mixed quantization and skip policy",
                {
                    var fields = try tiny().rawFields
                    fields["quantization"] = .object([
                        "mode": .string("mxfp4"), "bits": .number(4), "group_size": .number(32),
                        "mtp.layers.0.eh_proj": .object([
                            "mode": .string("affine"), "bits": .number(4),
                            "group_size": .number(64),
                            "future": .array([.null, .bool(true)]),
                        ]),
                        "model.layers.0.self_attn.o_proj": .bool(false),
                        "lm_head": .bool(false),
                        "future_metadata": .object(["flag": .bool(true)]),
                    ])
                    let config = try decode(fields)
                    try expect(
                        config.quantization.policy(for: "model.layers.1.mlp.switch_mlp.up_proj")?
                            .mode == "mxfp4",
                        "native default missing")
                    try expect(
                        config.quantization.policy(for: "mtp.layers.0.eh_proj")?.groupSize == 64,
                        "MTP policy missing")
                    try expect(
                        config.quantization.policy(for: "model.layers.0.self_attn.o_proj") == nil,
                        "explicit high-precision layer was quantized")
                    try expect(
                        config.quantization.policy(for: "lm_head") == nil,
                        "single-component module override was ignored")
                    try roundtrip(config)
                }
            ),
            (
                "MLX affine8 aliases preserve the native policy",
                {
                    var fields = try tiny().rawFields
                    let policy: MiMoV26JSONValue = .object([
                        "mode": .string("affine"), "bits": .number(4), "group_size": .number(64),
                        "language_model.model.embed_tokens": .object([
                            "mode": .string("affine"), "bits": .number(8),
                            "group_size": .number(64),
                        ]),
                    ])
                    fields["quantization"] = policy
                    fields["quantization_config"] = policy
                    let config = try decode(fields)
                    try expect(
                        config.quantization.sourceMetadata == nil,
                        "MLX policy was labelled source FP8")
                    try expect(
                        config.quantization.policy(for: "language_model.model.embed_tokens")?.bits
                            == 8,
                        "affine8 override was lost")
                    try roundtrip(config)
                    fields.removeValue(forKey: "quantization")
                    try expect(
                        try decode(fields).quantization == config.quantization,
                        "single MLX alias was lost")
                    fields["quantization"] = .object([
                        "mode": .string("mxfp4"), "bits": .number(4), "group_size": .number(32),
                    ])
                    try rejects(fields, field: "quantization_config")
                }
            ),
            (
                "unsupported quantization fails rather than falling back",
                {
                    for (mode, bits, group) in [
                        ("awq", 4, 32), ("mxfp4", 8, 32), ("mxfp4", 4, 64), ("affine", 4, 32),
                    ] {
                        var fields = try tiny().rawFields
                        fields["quantization"] = .object([
                            "mode": .string(mode), "bits": .number(Decimal(bits)),
                            "group_size": .number(Decimal(group)),
                        ])
                        try rejects(fields, field: "quantization")
                    }
                    var fields = try tiny().rawFields
                    fields["quantization_config"] = .object(["quant_method": .string("gptq")])
                    try rejects(fields, field: "quantization_config")
                }
            ),
            (
                "embedded MTP count and path validation",
                {
                    var fields = try tiny().rawFields
                    let valid: [String: MiMoV26JSONValue] = [
                        "architecture": .string("mimo_v2_nextn"), "num_layers": .number(3),
                        "storage": .string("embedded"), "file": .string("model-mtp.safetensors"),
                    ]
                    fields["omlx_mimo_mtp"] = .object(valid)
                    try expect(
                        try decode(fields).embeddedMTP?.numLayers == 3,
                        "three predictors not represented")
                    for (key, value) in [
                        ("num_layers", MiMoV26JSONValue.number(5)),
                        ("architecture", .string("DFlashDraftModel")),
                        ("file", .string("../outside.safetensors")),
                    ] {
                        var bad = valid
                        bad[key] = value
                        fields["omlx_mimo_mtp"] = .object(bad)
                        try rejects(fields, field: "omlx_mimo_mtp")
                    }
                }
            ),
            (
                "token namespace is preserved and range checked",
                {
                    var fields = try tiny().rawFields
                    fields["eos_token_id"] = .number(127)
                    let config = try decode(fields)
                    try expect(
                        config.tokenIDs["eos_token_id"] == 127
                            && config.rawFields["bos_token_id"] == .null,
                        "token IDs/null were normalized")
                    fields["eos_token_id"] = .number(128)
                    try rejects(fields, field: "eos_token_id")
                }
            ),
            (
                "all native EOS IDs survive scalar and array contracts",
                {
                    var fields = try tiny().rawFields
                    fields["eos_token_id"] = .array([.number(3), .number(7), .number(0)])
                    let config = try decode(fields)
                    try expect(config.eosTokenIDs == Set([3, 7, 0]), "EOS set was truncated")
                    try expect(
                        config.tokenIDs["eos_token_id"] == 3, "primary EOS ordering was lost")
                    try roundtrip(config)
                    for invalid: MiMoV26JSONValue in [
                        .array([]), .array([.number(3), .number(3)]),
                        .array([.number(128)]), .array([.number(-1)]), .array([.number(0.5)]),
                        .null,
                    ] {
                        fields["eos_token_id"] = invalid
                        try rejects(fields, field: "eos_token_id")
                    }
                }
            ),
        ]
    }

    static func metadataCases(official: URL, native: URL, dflash: URL) -> [(
        String, () throws -> Void
    )] {
        func read(_ url: URL) throws -> MiMoV26Configuration {
            try JSONDecoder().decode(MiMoV26Configuration.self, from: Data(contentsOf: url))
        }
        return [
            (
                "actual official config geometry and roundtrip",
                {
                    let config = try read(official)
                    try expect(
                        config.numHiddenLayers == 48 && config.numNextnPredictLayers == 3,
                        "wrong target or nextn count")
                    try expect(
                        config.hybridLayerPattern.filter { $0 == 0 }.count == 9
                            && config.moeLayerFrequency.filter { $0 == 1 }.count == 47,
                        "wrong trained layer pattern")
                    try expect(
                        config.maxPositionEmbeddings == 1_048_576 && config.headDim == 192
                            && config.valueHeadDim == 128
                            && config.fullAttention.rotaryDimensions == 64
                            && config.slidingAttention.keyValueHeads == 8,
                        "official context/attention was narrowed")
                    try expect(
                        config.vision?.depth == 28 && config.audio?.channels == 20
                            && config.audio?.zeroEmbeddingIndex == 1024
                            && config.tokenIDs["audio_token_id"] == 151669,
                        "native modality namespace changed")
                    try expect(
                        config.quantization.nativeDefault == nil
                            && config.quantization.sourceMetadata != nil,
                        "raw source policy mistaken for converted native policy")
                    try roundtrip(config)
                }
            ),
            (
                "actual native mixed policy and original metadata coexist",
                {
                    let config = try read(native)
                    try expect(
                        config.quantization.nativeDefault?.mode == "mxfp4"
                            && config.quantization.nativeDefault?.groupSize == 32
                            && config.quantization.sourceMetadata?["quant_method"]
                                == .string("fp8"),
                        "native and provenance policies were conflated")
                    for depth in 0 ..< 3 {
                        for prefix in ["mtp", "language_model.mtp"] {
                            let policy = config.quantization.policy(
                                for: "\(prefix).layers.\(depth).eh_proj")
                            try expect(
                                policy?.mode == "affine" && policy?.bits == 4
                                    && policy?.groupSize == 64,
                                "embedded MTP alias policy lost")
                        }
                    }
                    try expect(
                        config.attentionProjectionLayout == "fused_qkv"
                            && config.embeddedMTP?.numLayers == 3,
                        "original layout label or native component metadata mutated")
                    try roundtrip(config)
                }
            ),
            (
                "actual modality conflicts fail recoverably",
                {
                    let config = try read(official)
                    guard case .object(let original) = config.rawFields["processor_config"] else {
                        throw Failure(description: "official processor missing")
                    }
                    for key in ["audio_token_id", "audio_channels", "image_token_id"] {
                        var fields = config.rawFields
                        var processor = original
                        processor[key] = .number(1)
                        fields["processor_config"] = .object(processor)
                        try rejects(fields, field: "processor_config")
                    }
                    var fields = config.rawFields
                    fields.removeValue(forKey: "processor_config")
                    try rejects(fields, field: "processor_config")
                }
            ),
            (
                "actual DFlash five layers remain distinct from nextn three",
                {
                    let target = try read(official)
                    let data = try Data(contentsOf: dflash)
                    let draft = try JSONDecoder().decode(MiMoV26DFlashMetadata.self, from: data)
                    try draft.validateTarget(target)
                    try expect(
                        draft.numHiddenLayers == 5 && draft.targetLayerIDs == [0, 11, 23, 35, 47]
                            && draft.blockSize == 8 && draft.maskTokenID == 151675
                            && target.numNextnPredictLayers == 3,
                        "draft and nextn metadata were conflated")
                    let roundtrip = try JSONDecoder().decode(
                        MiMoV26DFlashMetadata.self, from: JSONEncoder().encode(draft))
                    try expect(roundtrip == draft, "DFlash metadata roundtrip changed")
                    try rejects(draft.rawFields, field: "model_type")
                    do {
                        try draft.validateTarget(tiny())
                        throw Failure(description: "wrong target draft binding accepted")
                    } catch is MiMoV26ConfigurationError {}
                }
            ),
        ]
    }
}

#if MIMO_CONFIG_CPU_PROBE
    @main
    private struct MiMoV26ConfigurationCPUProbe {
        static func main() throws {
            let args = CommandLine.arguments
            guard args.count == 4 else {
                throw MiMoV26ConfigurationChecks.Failure(
                    description: "expected official, native, DFlash config paths")
            }
            let cases =
                MiMoV26ConfigurationChecks.pureCases()
                + MiMoV26ConfigurationChecks.metadataCases(
                    official: URL(fileURLWithPath: args[1]), native: URL(fileURLWithPath: args[2]),
                    dflash: URL(fileURLWithPath: args[3]))
            var failures = 0
            for (name, body) in cases {
                do {
                    try body()
                    print("PASS \(name)")
                } catch {
                    failures += 1
                    print("FAIL \(name): \(error)")
                }
            }
            print(
                "RESULT discovered=\(cases.count) passed=\(cases.count - failures) failed=\(failures) skipped=0"
            )
            if failures > 0 {
                throw MiMoV26ConfigurationChecks.Failure(description: "configuration probe failed")
            }
        }
    }
#else
    final class MiMoV26ConfigurationTests: XCTestCase {
        func testPureConfigurationContracts() throws {
            for (name, body) in MiMoV26ConfigurationChecks.pureCases() {
                do { try body() } catch { XCTFail("\(name): \(error)") }
            }
        }

        func testActualOfficialAndNativeMetadata() throws {
            let environment = ProcessInfo.processInfo.environment
            guard let official = environment["MIMO_V26_OFFICIAL_CONFIG"],
                let native = environment["MIMO_V26_NATIVE_CONFIG"],
                let dflash = environment["MIMO_V26_DFLASH_CONFIG"]
            else {
                throw XCTSkip(
                    "Set the three MIMO_V26_*_CONFIG paths for artifact-bound metadata tests")
            }
            for (name, body) in MiMoV26ConfigurationChecks.metadataCases(
                official: URL(fileURLWithPath: official), native: URL(fileURLWithPath: native),
                dflash: URL(fileURLWithPath: dflash))
            {
                do { try body() } catch { XCTFail("\(name): \(error)") }
            }
        }
    }
#endif
