import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// A tiny synthetic MiMo V2.6 checkpoint that the tests build in code.
/// It uses no real weights and no external fixture folder. The values are
/// deterministic. They are not trained values and give no numerical oracle.
///
/// Geometry: text hidden 64, two layers (layer 0 dense full attention, layer 1
/// MoE sliding attention with sinks), four experts, vocabulary 128, unequal
/// K/V head widths (32/16). Vision tower hidden 8, depth 4. Audio patch
/// encoder hidden 64, one layer, 20 channels. Three MTP heads.
enum MiMoV26TinyCheckpoint {
    struct Spec {
        let name: String
        let shape: [Int]
        let dtype: MiMoV26ConvertedScalarType
        let file: String
        var byteCount: Int { shape.reduce(dtype.bytes, *) }
    }

    struct Inputs {
        var config: Data
        var index: Data
        var descriptors: [String: MiMoV26ConvertedTensorDescriptor]
        var provenance: MiMoV26ConvertedProvenance
        func plan() throws -> MiMoV26ConvertedLoadPlan {
            try .make(
                configurationData: config, indexData: index, descriptors: descriptors,
                provenance: provenance)
        }
    }

    /// The four root shards of the native layout, in sorted order.
    static let nativeFiles = [
        "audio.safetensors", "mtp.safetensors", "target.safetensors", "vision.safetensors",
    ]
    static let mtpProjections = [
        "eh_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    static let template =
        "{{ messages }}{% if enable_thinking is false %}<think></think>{% endif %}\n"

    static let provenance = MiMoV26ConvertedProvenance(
        artifactID: "tiny-synthetic", sourceRepository: "XiaomiMiMo/MiMo-V2.6-Flash-RL",
        sourceRevision: String(repeating: "a", count: 40),
        conversionManifestSHA256: String(repeating: "b", count: 64))
    static let mlxVLMProvenance = MiMoV26ConvertedProvenance(
        artifactID: "tiny-synthetic-mlx-vlm", sourceRepository: "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
        sourceRevision: String(repeating: "c", count: 40),
        conversionManifestSHA256: String(repeating: "d", count: 64), layout: .mlxVLM)

    // MARK: - Configuration

    /// Text, vision, audio and processor fields. No quantization or MTP file.
    static func baseFields(dtype: String = "bfloat16") -> [String: Any] {
        [
            "model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"],
            "hidden_size": 64, "intermediate_size": 64, "moe_intermediate_size": 32,
            "vocab_size": 128, "num_hidden_layers": 2, "max_position_embeddings": 256,
            "sliding_window_size": 8, "sliding_window": 8, "num_nextn_predict_layers": 3,
            "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
            "partial_rotary_factor": 0.5, "attention_value_scale": 0.75,
            "layernorm_epsilon": 0.00001, "attention_projection_layout": "split",
            "moe_router_dtype": "bfloat16", "hidden_act": "silu", "dtype": dtype,
            "attention_bias": false, "tie_word_embeddings": false, "attention_dropout": 0,
            "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_routed_experts": 4,
            "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
            "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 32,
            "v_head_dim": 16, "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
            "swa_head_dim": 32, "swa_v_head_dim": 16, "rope_theta": 10000,
            "swa_rope_theta": 10000, "add_full_attention_sink_bias": false,
            "add_swa_attention_sink_bias": true, "eos_token_id": [1], "pad_token_id": 0,
            "image_token_id": 2, "video_token_id": 3, "vision_start_token_id": 4,
            "vision_end_token_id": 5, "audio_token_id": 6, "audio_start_token_id": 7,
            "audio_end_token_id": 8,
            "vision_config": [
                "depth": 4, "fullatt_block_indexes": [0], "hidden_act": "silu",
                "hidden_size": 8, "in_chans": 3, "intermediate_size": 16, "num_heads": 2,
                "num_key_value_heads": 1, "num_query_groups": 2, "out_hidden_size": 64,
                "patch_size": 2, "spatial_merge_size": 2, "spatial_patch_size": 2,
                "temporal_patch_size": 2, "tokens_per_second": 2, "use_sink": true,
                "visual_token_window_size": 64, "vit_window_attn_types": [-1, 0, 1, 0],
                "window_size": 128, "qk_channels": 8, "kv_channels": 8,
            ] as [String: Any],
            "audio_config": [
                "add_post_norm": true, "audio_channels": 20, "audio_segment_size": 6000,
                "group_size": 4, "input_full_attention": true, "input_local_attn_heads": 2,
                "input_local_dim": 64, "input_local_head_dim": 32,
                "input_local_hidden_dropout": 0, "input_local_intermediate_size": 128,
                "input_local_layers": 1, "out_hidden_size": 64, "partial_rotary_factor": 1,
                "projection_layers": 2, "rope_theta": 640000, "speech_vocab_size": "7",
                "speech_zeroemb_idx": "6",
            ] as [String: Any],
            "processor_config": [
                "rope_type": "rope", "use_video_timestamps": true,
                "use_per_grid_t_timestamps": false, "temporal_compression_ratio": 1,
                "patch_size": 2, "merge_size": 2, "temporal_patch_size": 2,
                "image_min_pixels": 16, "image_max_pixels": 150, "video_min_pixels": 16,
                "video_max_pixels": 150, "video_total_max_pixels": 1000,
                "image_token_id": 2, "video_token_id": 3, "vision_start_token_id": 4,
                "vision_end_token_id": 5, "video_start_token_id": 9, "video_end_token_id": 10,
                "audio_token_id": 6, "audio_start_token_id": 7, "audio_end_token_id": 8,
                "audio_channels": 20, "audio_group_size": 4, "audio_segment_size": 6000,
                "audio_zeroemb_idx": Array(repeating: 6, count: 20),
                "audio_sampling_rate": 24000, "audio_n_mels": 128,
                "video_audio_interleave_length": 0, "audio_input_id_per_second": 25,
            ] as [String: Any],
        ]
    }

    /// The native converted layout: MXFP4 experts, explicit affine MTP heads
    /// and the embedded MTP file declaration.
    static func nativeFields() -> [String: Any] {
        var fields = baseFields()
        var quantization: [String: Any] = ["mode": "mxfp4", "bits": 4, "group_size": 32]
        for depth in 0 ..< 3 {
            for name in mtpProjections {
                quantization["mtp.layers.\(depth).\(name)"] = [
                    "mode": "affine", "bits": 4, "group_size": 64,
                ]
            }
        }
        fields["quantization"] = quantization
        fields["omlx_mimo_mtp"] = [
            "architecture": "mimo_v2_nextn", "num_layers": 3, "storage": "embedded",
            "file": "mtp.safetensors",
        ]
        return fields
    }

    static func data(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }
    static func object(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
    static func configuration(_ fields: [String: Any]) throws -> MiMoV26Configuration {
        try JSONDecoder().decode(MiMoV26Configuration.self, from: data(fields))
    }

    // MARK: - Tensor inventories

    /// Native converted inventory. The order is the order in the shard files.
    static func nativeSpecs() throws -> [Spec] {
        let c = try configuration(nativeFields())
        var specs: [Spec] = []
        func add(
            _ name: String, _ shape: [Int], _ dtype: MiMoV26ConvertedScalarType, _ file: String
        ) {
            specs.append(.init(name: name, shape: shape, dtype: dtype, file: file))
        }
        let target = "target.safetensors"
        let h = c.hiddenSize
        add("model.embed_tokens.weight", [c.vocabularySize, h], .bfloat16, target)
        add("model.norm.weight", [h], .bfloat16, target)
        add("lm_head.weight", [c.vocabularySize, h], .bfloat16, target)
        for layer in 0 ..< c.numHiddenLayers {
            let p = "model.layers.\(layer)."
            let g = try c.attentionGeometry(at: layer)
            add(p + "input_layernorm.weight", [h], .bfloat16, target)
            add(p + "post_attention_layernorm.weight", [h], .bfloat16, target)
            add(p + "self_attn.q_proj.weight", [g.queryHeads * g.headDim, h], .bfloat16, target)
            add(p + "self_attn.k_proj.weight", [g.keyValueHeads * g.headDim, h], .bfloat16, target)
            add(
                p + "self_attn.v_proj.weight", [g.keyValueHeads * g.valueHeadDim, h], .bfloat16,
                target)
            add(
                p + "self_attn.o_proj.weight", [h, g.queryHeads * g.valueHeadDim], .bfloat16,
                target)
            if g.hasSinks {
                add(p + "self_attn.attention_sink_bias", [g.queryHeads], .bfloat16, target)
            }
            if c.moeLayerFrequency[layer] == 1 {
                let e = c.routedExpertCount
                add(p + "mlp.gate.weight", [e, h], .bfloat16, target)
                add(p + "mlp.gate.e_score_correction_bias", [e], .float32, target)
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    let rows = name == "down_proj" ? h : c.moeIntermediateSize
                    let cols = name == "down_proj" ? c.moeIntermediateSize : h
                    let module = p + "mlp.switch_mlp." + name
                    add(module + ".weight", [e, rows, cols / 8], .uint32, target)
                    add(module + ".scales", [e, rows, cols / 32], .uint8, target)
                }
            } else {
                add(p + "mlp.gate_proj.weight", [c.intermediateSize, h], .bfloat16, target)
                add(p + "mlp.up_proj.weight", [c.intermediateSize, h], .bfloat16, target)
                add(p + "mlp.down_proj.weight", [h, c.intermediateSize], .bfloat16, target)
            }
        }
        let vision = try MiMoV26VisionTower.expectedTensorShapes(configuration: XCTUnwrap(c.vision))
        for key in vision.keys.sorted() {
            add(key, vision[key]!, .bfloat16, "vision.safetensors")
        }
        let audio = try MiMoV26AudioPatchEncoder.expectedTensorShapes(
            configuration: XCTUnwrap(c.audio))
        for key in audio.keys.sorted() {
            add(key, audio[key]!, .bfloat16, "audio.safetensors")
        }
        let g = c.slidingAttention
        let rows: [String: Int] = [
            "eh_proj": h, "self_attn.q_proj": g.queryHeads * g.headDim,
            "self_attn.k_proj": g.keyValueHeads * g.headDim,
            "self_attn.v_proj": g.keyValueHeads * g.valueHeadDim, "self_attn.o_proj": h,
            "mlp.gate_proj": c.intermediateSize, "mlp.up_proj": c.intermediateSize,
            "mlp.down_proj": h,
        ]
        let cols: [String: Int] = [
            "eh_proj": 2 * h, "self_attn.q_proj": h, "self_attn.k_proj": h,
            "self_attn.v_proj": h, "self_attn.o_proj": g.queryHeads * g.valueHeadDim,
            "mlp.gate_proj": h, "mlp.up_proj": h, "mlp.down_proj": c.intermediateSize,
        ]
        for depth in 0 ..< 3 {
            let p = "mtp.layers.\(depth)."
            for norm in [
                "enorm", "hnorm", "input_layernorm", "pre_mlp_layernorm", "final_layernorm",
            ] {
                add(p + norm + ".weight", [h], .bfloat16, "mtp.safetensors")
            }
            if g.hasSinks {
                add(
                    p + "self_attn.attention_sink_bias", [g.queryHeads], .bfloat16,
                    "mtp.safetensors")
            }
            for name in mtpProjections {
                let r = rows[name]!
                let k = cols[name]!
                add(p + name + ".weight", [r, k / 8], .uint32, "mtp.safetensors")
                add(p + name + ".scales", [r, k / 64], .bfloat16, "mtp.safetensors")
                add(p + name + ".biases", [r, k / 64], .bfloat16, "mtp.safetensors")
            }
        }
        return specs
    }

    static func inputs(
        fields: [String: Any], specs: [Spec], provenance: MiMoV26ConvertedProvenance
    ) throws -> Inputs {
        var descriptors: [String: MiMoV26ConvertedTensorDescriptor] = [:]
        var map: [String: String] = [:]
        for spec in specs {
            descriptors[spec.name] = .init(shape: spec.shape, dtype: spec.dtype, file: spec.file)
            map[spec.name] = spec.file
        }
        let total = specs.reduce(0) { $0 + $1.byteCount }
        return .init(
            config: try data(fields),
            index: try data(["metadata": ["total_size": total], "weight_map": map]),
            descriptors: descriptors, provenance: provenance)
    }

    static func nativeInputs() throws -> Inputs {
        try inputs(fields: nativeFields(), specs: nativeSpecs(), provenance: provenance)
    }

    /// The published MLX-VLM layout: source names, affine 4/64 default,
    /// explicit 8-bit packed target and audio matrices, MXFP4 experts, and
    /// floating vision and MTP tensors.
    static func mlxVLMInputs() throws -> (inputs: Inputs, fields: [String: Any], specs: [Spec]) {
        var fields = baseFields()
        let c = try configuration(fields)
        var quantization: [String: Any] = ["mode": "affine", "bits": 4, "group_size": 64]
        var specs: [Spec] = []
        func add(
            _ name: String, _ shape: [Int], _ dtype: MiMoV26ConvertedScalarType, _ file: String
        ) {
            specs.append(.init(name: name, shape: shape, dtype: dtype, file: file))
        }
        func packed(_ module: String, _ shape: [Int], _ file: String, expert: Bool = false) {
            let group = expert ? 32 : 64
            let bits = expert ? 4 : 8
            quantization[module] = [
                "mode": expert ? "mxfp4" : "affine", "bits": bits, "group_size": group,
            ]
            let prefix = Array(shape.dropLast())
            let k = shape.last!
            add(module + ".weight", prefix + [k / (32 / bits)], .uint32, file)
            add(module + ".scales", prefix + [k / group], expert ? .uint8 : .bfloat16, file)
            if !expert { add(module + ".biases", prefix + [k / group], .bfloat16, file) }
        }
        let t = "language_model.safetensors"
        let h = c.hiddenSize
        packed("language_model.model.embed_tokens", [c.vocabularySize, h], t)
        add("language_model.model.norm.weight", [h], .bfloat16, t)
        packed("language_model.lm_head", [c.vocabularySize, h], t)
        for layer in 0 ..< c.numHiddenLayers {
            let p = "language_model.model.layers.\(layer)."
            let g = try c.attentionGeometry(at: layer)
            add(p + "input_layernorm.weight", [h], .bfloat16, t)
            add(p + "post_attention_layernorm.weight", [h], .bfloat16, t)
            packed(p + "self_attn.q_proj", [g.queryHeads * g.headDim, h], t)
            packed(p + "self_attn.k_proj", [g.keyValueHeads * g.headDim, h], t)
            packed(p + "self_attn.v_proj", [g.keyValueHeads * g.valueHeadDim, h], t)
            packed(p + "self_attn.o_proj", [h, g.queryHeads * g.valueHeadDim], t)
            if g.hasSinks {
                add(p + "self_attn.attention_sink_bias", [g.queryHeads], .bfloat16, t)
            }
            if c.moeLayerFrequency[layer] == 1 {
                let e = c.routedExpertCount
                add(p + "mlp.gate.weight", [e, h], .bfloat16, t)
                add(p + "mlp.gate.e_score_correction_bias", [e], .float32, t)
                packed(
                    p + "mlp.switch_mlp.gate_proj", [e, c.moeIntermediateSize, h], t, expert: true)
                packed(
                    p + "mlp.switch_mlp.up_proj", [e, c.moeIntermediateSize, h], t, expert: true)
                packed(
                    p + "mlp.switch_mlp.down_proj", [e, h, c.moeIntermediateSize], t, expert: true)
            } else {
                packed(p + "mlp.gate_proj", [c.intermediateSize, h], t)
                packed(p + "mlp.up_proj", [c.intermediateSize, h], t)
                packed(p + "mlp.down_proj", [h, c.intermediateSize], t)
            }
        }
        let visionConfig = try XCTUnwrap(c.vision)
        let vision = try MiMoV26VisionTower.expectedTensorShapes(configuration: visionConfig)
        for key in vision.keys.sorted() {
            let shape = vision[key]!
            let stored =
                key == "visual.patch_embed.proj.weight"
                ? [shape[0], shape.dropFirst().reduce(1, *)] : shape
            add(
                "vision_tower." + key.dropFirst("visual.".count), stored, .bfloat16,
                "vision.safetensors")
        }
        let audio = try MiMoV26AudioPatchEncoder.expectedTensorShapes(
            configuration: XCTUnwrap(c.audio))
        for key in audio.keys.sorted() {
            let shape = audio[key]!
            if key.hasSuffix(".weight"), shape.count == 2 {
                packed(String(key.dropLast(".weight".count)), shape, "audio.safetensors")
            } else {
                add(key, shape, .bfloat16, "audio.safetensors")
            }
        }
        let g = c.slidingAttention
        for depth in 0 ..< 3 {
            let p = "language_model.model.mtp.layers.\(depth)."
            for norm in [
                "enorm", "hnorm", "input_layernorm", "pre_mlp_layernorm", "final_layernorm",
            ] {
                add(p + norm + ".weight", [h], .bfloat16, "mtp.safetensors")
            }
            if g.hasSinks {
                add(
                    p + "self_attn.attention_sink_bias", [g.queryHeads], .bfloat16,
                    "mtp.safetensors")
            }
            for (name, rows, cols) in [
                ("eh_proj", h, 2 * h),
                ("self_attn.q_proj", g.queryHeads * g.headDim, h),
                ("self_attn.k_proj", g.keyValueHeads * g.headDim, h),
                ("self_attn.v_proj", g.keyValueHeads * g.valueHeadDim, h),
                ("self_attn.o_proj", h, g.queryHeads * g.valueHeadDim),
                ("mlp.gate_proj", c.intermediateSize, h),
                ("mlp.up_proj", c.intermediateSize, h),
                ("mlp.down_proj", h, c.intermediateSize),
            ] {
                add(p + name + ".weight", [rows, cols], .bfloat16, "mtp.safetensors")
            }
        }
        fields["quantization"] = quantization
        fields["quantization_config"] = quantization
        let built = try inputs(fields: fields, specs: specs, provenance: mlxVLMProvenance)
        return (built, fields, specs)
    }

    // MARK: - Values

    /// Deterministic values. Norm scales are one; other floats are small.
    static func floats(_ name: String, count: Int) -> [Float] {
        let salt = name.utf8.reduce(0) { ($0 &* 31 &+ Int($1)) % 1009 }
        let isScale =
            name.hasSuffix(".weight")
            && (name.contains("norm") || name.contains("ln_q"))
        return (0 ..< count).map { i in
            isScale ? 1 : Float(sin(Double((i * 7 + salt) % 97))) * 0.05
        }
    }
    static func words(_ name: String, count: Int) -> [UInt32] {
        let salt = UInt32(truncatingIfNeeded: name.utf8.reduce(0) { ($0 &* 31 &+ Int($1)) })
        return (0 ..< count).map { i in
            (UInt32(truncatingIfNeeded: i) &* 0x9E37_79B9) ^ salt ^ 0x1234_5678
        }
    }
    /// The float value that a truncated BF16 payload element stores.
    static func bfloat16(_ value: Float) -> Float {
        Float(bitPattern: value.bitPattern & 0xFFFF_0000)
    }
    /// MXFP4 E8M0 scales: 120 means 2^-7, which keeps the experts small.
    static func scaleBytes(count: Int) -> [UInt8] { Array(repeating: 120, count: count) }

    static func payload(_ spec: Spec) -> Data {
        let count = spec.shape.reduce(1, *)
        var data = Data()
        data.reserveCapacity(spec.byteCount)
        switch spec.dtype {
        case .uint32:
            for word in words(spec.name, count: count) {
                withUnsafeBytes(of: word.littleEndian) { data.append(contentsOf: $0) }
            }
        case .uint8:
            data.append(contentsOf: scaleBytes(count: count))
        case .float32:
            for value in floats(spec.name, count: count) {
                withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) }
            }
        case .bfloat16:
            // Truncation to the upper 16 bits. `bfloat16(_:)` gives the same value.
            for value in floats(spec.name, count: count) {
                let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
                withUnsafeBytes(of: bits.littleEndian) { data.append(contentsOf: $0) }
            }
        case .float16:
            for value in floats(spec.name, count: count) {
                withUnsafeBytes(of: Float16(value).bitPattern.littleEndian) {
                    data.append(contentsOf: $0)
                }
            }
        }
        return data
    }

    /// Native MLX arrays with the same values as `payload`.
    static func array(_ spec: Spec) -> MLXArray {
        let count = spec.shape.reduce(1, *)
        switch spec.dtype {
        case .uint32: return MLXArray(words(spec.name, count: count)).reshaped(spec.shape)
        case .uint8: return MLXArray(scaleBytes(count: count)).reshaped(spec.shape)
        case .float32: return MLXArray(floats(spec.name, count: count), spec.shape)
        case .bfloat16:
            return MLXArray(floats(spec.name, count: count), spec.shape).asType(.bfloat16)
        case .float16:
            return MLXArray(floats(spec.name, count: count), spec.shape).asType(.float16)
        }
    }
    static func arrays(_ specs: [Spec]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: specs.map { ($0.name, array($0)) })
    }

    // MARK: - Files

    /// Writes one safetensors file. Tensors are stored in the given order,
    /// without gaps. The header is padded with spaces to 8 bytes.
    static func writeSafetensors(_ specs: [Spec], to url: URL) throws {
        var header: [String: Any] = [:]
        var body = Data()
        for spec in specs {
            let bytes = payload(spec)
            header[spec.name] = [
                "dtype": spec.dtype.rawValue, "shape": spec.shape,
                "data_offsets": [body.count, body.count + bytes.count],
            ]
            body.append(bytes)
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        json.append(Data(repeating: 0x20, count: (8 - json.count % 8) % 8))
        let length = UInt64(json.count)
        var file = Data((0 ..< 8).map { UInt8((length >> (UInt64($0) * 8)) & 0xff) })
        file.append(json)
        file.append(body)
        try file.write(to: url)
    }

    /// A new empty folder under the temporary directory, with symlinks resolved.
    static func temporaryRoot(_ label: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mimo-v26-\(label)-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url.resolvingSymlinksInPath().standardizedFileURL
    }

    /// Writes the complete native bundle: config, index, four shards and the
    /// tokenizer sidecars that the factory reads. Returns the root folder.
    @discardableResult
    static func writeNativeBundle(to root: URL, tokenizerSidecars: Bool = true) throws -> URL {
        let specs = try nativeSpecs()
        let inputs = try nativeInputs()
        try inputs.config.write(to: root.appendingPathComponent("config.json"))
        try inputs.index.write(to: root.appendingPathComponent("model.safetensors.index.json"))
        for file in nativeFiles {
            try writeSafetensors(
                specs.filter { $0.file == file }, to: root.appendingPathComponent(file))
        }
        if tokenizerSidecars {
            // Sidecars for the mock tokenizer. They are not real tokenizer assets.
            try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
            try data(["chat_template": "embedded alternate", "eos_token": "<|im_end|>"])
                .write(to: root.appendingPathComponent("tokenizer_config.json"))
            try Data(template.utf8).write(to: root.appendingPathComponent("chat_template.jinja"))
        }
        return root
    }

    static let limits = MiMoV26FilesystemLimits(
        maximumShardBytes: 2 << 20, maximumTotalFileBytes: 8 << 20)

    static func preflight(
        _ root: URL, expectations: MiMoV26FilesystemExpectations = .init()
    ) throws -> MiMoV26FilesystemLoadPlan {
        try MiMoV26FilesystemWeights.preflight(
            root: root, provenance: provenance, limits: limits, expectations: expectations)
    }

    /// Runs the body in a fresh construction scope. A retained fault keeps the
    /// scope alive until process exit, as the production contract requires.
    static func withScope<Value>(_ body: (NativeConstructionScope) throws -> Value) rethrows
        -> Value
    {
        let work = NativeConstructionScope()
        defer { if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) } }
        return try body(work)
    }

    // MARK: - Tokenizer

    /// Routing tokenizer for the factory and the media processor. It is not a
    /// Jinja or BPE oracle. Media parts render as start, pad, end markers.
    final class Tokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        static let ids = [
            "<|im_end|>": 1, "<|image_pad|>": 2, "<|video_pad|>": 3, "<|vision_start|>": 4,
            "<|vision_end|>": 5, "<|audio_pad|>": 6, "<|mimo_audio_start|>": 7,
            "<|mimo_audio_end|>": 8, "<|mimo_video_start|>": 9, "<|mimo_video_end|>": 10,
            "<|extra_eos|>": 15,
        ]
        private let lock = NSLock()
        private var recorded: [[String: any Sendable]?] = []
        private let ignoresThinking: Bool
        private let eosSpelling: String?
        var contexts: [[String: any Sendable]?] { lock.withLock { recorded } }
        /// `ignoresThinking` renders the same tokens for every thinking value.
        init(ignoresThinking: Bool = false, eosToken: String? = "<|im_end|>") {
            self.ignoresThinking = ignoresThinking
            eosSpelling = eosToken
        }
        var bosToken: String? { nil }
        var eosToken: String? { eosSpelling }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.utf8.map { 20 + Int($0) % 32 }
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }
        func convertTokenToId(_ token: String) -> Int? { Self.ids[token] }
        func convertIdToToken(_ id: Int) -> String? { Self.ids.first { $0.value == id }?.key }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(messages: [[String: any Sendable]], chatTemplate: String) throws
            -> [Int]
        {
            try applyChatTemplate(
                messages: messages, chatTemplate: chatTemplate, tools: nil, additionalContext: nil)
        }
        func applyChatTemplate(
            messages: [[String: any Sendable]], chatTemplate: String,
            tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            lock.withLock { recorded.append(additionalContext) }
            var result = [11]
            for message in messages {
                if let parts = message["content"] as? [[String: any Sendable]] {
                    for part in parts {
                        switch part["type"] as? String {
                        case "image": result += [4, 2, 5]
                        case "video": result += [4, 3, 5]
                        case "audio": result += [7, 6, 8]
                        default:
                            result += encode(
                                text: part["text"] as? String ?? "", addSpecialTokens: true)
                        }
                    }
                } else if let text = message["content"] as? String {
                    result += encode(text: text, addSpecialTokens: true)
                }
            }
            if !ignoresThinking, (additionalContext?["enable_thinking"] as? Bool) == false {
                result += [13, 14]
            }
            return result + [12]
        }
    }

    struct Loader: TokenizerLoader {
        let tokenizer: Tokenizer
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { tokenizer }
    }

    /// Host permit stand-in that always grants the exact required bytes.
    final class Permit: MiMoV26SerialLoadReservation, @unchecked Sendable {
        let request: MiMoV26SerialLoadRequest
        var reservedLoadBytes: UInt64 { request.requiredLoadBytes }
        init(_ request: MiMoV26SerialLoadRequest) { self.request = request }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }

    // MARK: - Loaded model

    struct Loaded {
        let model: MiMoV26LoadedModel
        let root: URL
        let tokenizer: Tokenizer
    }

    /// Writes a bundle, runs the factory's prepare and strict serial load, and
    /// returns the loaded wrapper. The caller deletes `root`.
    static func loaded() async throws -> Loaded {
        let root = try writeNativeBundle(to: temporaryRoot("loaded"))
        let session = try MiMoV26SerialLoadSession(plan: preflight(root))
        let tokenizer = Tokenizer()
        let prepared = try await MiMoV26ModelFactory.prepare(
            request: session.request, configuration: .init(directory: root),
            tokenizerLoader: Loader(tokenizer: tokenizer))
        let context = try withScope { work in
            try MiMoV26ModelFactory.load(
                session: session, reservation: Permit(session.request), prepared: prepared,
                retaining: work)
        }
        return .init(
            model: try XCTUnwrap(context.model as? MiMoV26LoadedModel), root: root,
            tokenizer: tokenizer)
    }

    // MARK: - Media models

    /// Float32 media models for the processor tests: target, vision tower,
    /// audio patch encoder, adapter and processor. No MTP and no quantization.
    static func mediaModels(dtype: String = "float32") throws -> MiMoMediaFixture.Models {
        var fields = baseFields(dtype: dtype)
        fields["eos_token_id"] = [1]
        let c = try configuration(fields)
        let d: DType = dtype == "float32" ? .float32 : .bfloat16
        let target = try MiMoV26TextModel(c)
        try target.update(
            parameters: .unflattened(
                target.parameters().flattened().map { name, value in
                    (
                        name,
                        MiMoMediaFixture.values(
                            name, value.shape,
                            name.hasSuffix("e_score_correction_bias")
                                ? .float32 : name.hasSuffix("mlp.gate.weight") ? .bfloat16 : d)
                    )
                }), verify: .all)
        let vision = try MiMoV26VisionTower(configuration: XCTUnwrap(c.vision))
        try vision.loadNativeWeights(
            try MiMoV26VisionTower.expectedTensorShapes(configuration: XCTUnwrap(c.vision))
                .mapValues { MiMoMediaFixture.values("vision", $0, d) }, expectedDType: d)
        let patch = try MiMoV26AudioPatchEncoder(configuration: XCTUnwrap(c.audio))
        try patch.loadNativeWeights(
            try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: XCTUnwrap(c.audio))
                .mapValues { MiMoMediaFixture.values("audio", $0, d) }, expectedDType: d)
        let adapter = try MiMoV26CBv2Adapter(target: target)
        let tokenizer = MiMoMediaTestTokenizer()
        let generation = MiMoV26MediaGeneration()
        let owner = MiMoMediaFixture.Owner()
        let template = "routing fixture, not Jinja"
        let processor = try MiMoV26MultimodalProcessor(
            configuration: c, tokenizer: tokenizer, chatTemplate: template,
            templateSHA256: MiMoV26MultimodalProcessor.hash(Data(template.utf8)),
            limits: MiMoMediaFixture.limits, vision: vision, audioPatch: patch, adapter: adapter,
            stopTokens: [], generation: generation, retaining: owner, audioCodec: nil)
        return .init(
            target: target, vision: vision, patch: patch, adapter: adapter, processor: processor,
            tokenizer: tokenizer, generation: generation, owner: owner)
    }
}
