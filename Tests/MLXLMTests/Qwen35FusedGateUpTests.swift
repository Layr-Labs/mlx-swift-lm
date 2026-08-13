import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

// Serialized for the same reason as SwitchGLUTests: SwitchGLU forwards run
// compiled GLU kernels, and parallel tracing can deadlock the per-
// CompiledFunction locks.
@Suite(.serialized)
struct Qwen35FusedGateUpTests {

    /// Fused (`gate_up_proj`) and split (`gate_proj`/`up_proj`) SwitchGLU
    /// must be numerically identical when the fused weights are the
    /// row-concatenation [gate; up] — the exact layout
    /// `qwen35FuseSwitchMLPGateUp` produces.
    @Test func fusedForwardMatchesSplitForwardFloat() {
        let inputDims = 16
        let hiddenDims = 8
        let numExperts = 4
        let split = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        let fused = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            fuseGateUp: true)

        let splitParams = Dictionary(
            uniqueKeysWithValues: split.parameters().flattened())
        let fusedWeights = concatenated(
            [splitParams["gate_proj.weight"]!, splitParams["up_proj.weight"]!],
            axis: -2)
        try! fused.update(
            parameters: ModuleParameters.unflattened([
                "gate_up_proj.weight": fusedWeights,
                "down_proj.weight": splitParams["down_proj.weight"]!,
            ]),
            verify: [.all])

        // Decode-shaped (tiny) and prefill-shaped (sorted path, >= 64
        // assignments) calls.
        for tokens in [1, 16] {
            let x = MLXRandom.normal([tokens, inputDims])
            let indices = MLXRandom.randInt(0 ..< numExperts, [tokens, 8])
                .asType(.uint32)
            let expected = split(x, indices)
            let actual = fused(x, indices)
            eval(expected, actual)
            #expect(expected.shape == actual.shape)
            let maxErr = (actual - expected).abs().max().item(Float.self)
            #expect(
                actual.allClose(expected, rtol: 1e-5, atol: 1e-6).item(Bool.self),
                Comment(rawValue: "fused/split divergence at tokens=\(tokens): max abs err \(maxErr)"))
        }
    }

    /// The sanitize helper must map an MLX split quantized checkpoint onto
    /// the fused layout such that the fused quantized forward reproduces the
    /// split quantized forward exactly (same codes, same scales — row
    /// concatenation cannot change any dequantized value).
    @Test func quantizedSplitCheckpointFusesExactly() {
        let inputDims = 128  // group size 64 needs K % 64 == 0
        let hiddenDims = 64
        let numExperts = 4
        let groupSize = 64
        let bits = 4
        let prefix = "language_model.model.layers.0.mlp.switch_mlp"

        func randomQuantized(_ rows: Int, _ cols: Int) -> (MLXArray, MLXArray, MLXArray) {
            let w = MLXRandom.normal([numExperts, rows, cols]).asType(.bfloat16)
            let q = quantized(w, groupSize: groupSize, bits: bits, mode: .affine)
            return (q.wq, q.scales, q.biases!)
        }
        let gate = randomQuantized(hiddenDims, inputDims)
        let up = randomQuantized(hiddenDims, inputDims)
        let down = randomQuantized(inputDims, hiddenDims)

        var weights: [String: MLXArray] = [
            "\(prefix).gate_proj.weight": gate.0,
            "\(prefix).gate_proj.scales": gate.1,
            "\(prefix).gate_proj.biases": gate.2,
            "\(prefix).up_proj.weight": up.0,
            "\(prefix).up_proj.scales": up.1,
            "\(prefix).up_proj.biases": up.2,
            "\(prefix).down_proj.weight": down.0,
            "\(prefix).down_proj.scales": down.1,
            "\(prefix).down_proj.biases": down.2,
        ]
        weights = qwen35FuseSwitchMLPGateUp(weights: weights)

        // Split keys gone, fused keys present with concatenated shapes.
        #expect(weights["\(prefix).gate_proj.weight"] == nil)
        #expect(weights["\(prefix).up_proj.scales"] == nil)
        let fusedWeight = weights["\(prefix).gate_up_proj.weight"]!
        let fusedScales = weights["\(prefix).gate_up_proj.scales"]!
        let fusedBiases = weights["\(prefix).gate_up_proj.biases"]!
        #expect(fusedWeight.shape == [numExperts, 2 * hiddenDims, inputDims * bits / 32])
        #expect(fusedScales.shape == [numExperts, 2 * hiddenDims, inputDims / groupSize])
        // Idempotence: a second pass is a no-op.
        let again = qwen35FuseSwitchMLPGateUp(weights: weights)
        #expect(again.keys.sorted() == weights.keys.sorted())

        // Forward equivalence, split vs fused, on the quantized modules.
        let tokens = 16
        let x = MLXRandom.normal([tokens, inputDims]).asType(.bfloat16)
        let indices = MLXRandom.randInt(0 ..< numExperts, [tokens, 8]).asType(.uint32)

        let splitGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        let fusedGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            fuseGateUp: true)
        quantize(model: splitGLU) { _, _ in (groupSize, bits, .affine) }
        quantize(model: fusedGLU) { _, _ in (groupSize, bits, .affine) }
        try! splitGLU.update(
            parameters: ModuleParameters.unflattened([
                "gate_proj.weight": gate.0, "gate_proj.scales": gate.1,
                "gate_proj.biases": gate.2,
                "up_proj.weight": up.0, "up_proj.scales": up.1,
                "up_proj.biases": up.2,
                "down_proj.weight": down.0, "down_proj.scales": down.1,
                "down_proj.biases": down.2,
            ]), verify: [.all])
        try! fusedGLU.update(
            parameters: ModuleParameters.unflattened([
                "gate_up_proj.weight": fusedWeight,
                "gate_up_proj.scales": fusedScales,
                "gate_up_proj.biases": fusedBiases,
                "down_proj.weight": down.0, "down_proj.scales": down.1,
                "down_proj.biases": down.2,
            ]), verify: [.all])

        let expected = splitGLU(x, indices)
        let actual = fusedGLU(x, indices)
        eval(expected, actual)
        #expect(expected.shape == actual.shape)
        let maxErr = (actual - expected).abs().max().item(Float.self)
        #expect(
            actual.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            Comment(rawValue: "quantized fused/split divergence: max abs err \(maxErr)"))
    }

    /// Raw HF stacked exports map directly onto the fused module keys with
    /// gate rows first, and `mtp.*` trees are never rewritten.
    @Test func rawStackedExportMapsDirectlyAndMTPKeysAreUntouched() {
        let e = 2
        let ffn = 4
        let hidden = 8
        let stacked = MLXRandom.normal([e, 2 * ffn, hidden])
        let down = MLXRandom.normal([e, hidden, ffn])
        let mtpGate = MLXRandom.normal([e, ffn, hidden])
        var weights: [String: MLXArray] = [
            "language_model.model.layers.0.mlp.experts.gate_up_proj": stacked,
            "language_model.model.layers.0.mlp.experts.down_proj": down,
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight": mtpGate,
            "mtp.layers.0.mlp.switch_mlp.up_proj.weight": mtpGate,
        ]
        weights = qwen35FuseSwitchMLPGateUp(weights: weights)

        let fused = weights["language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight"]
        #expect(fused != nil)
        #expect(fused!.shape == [e, 2 * ffn, hidden])
        // Gate rows first: identical array, not re-ordered.
        eval(fused!, stacked)
        #expect(fused!.allClose(stacked).item(Bool.self))
        #expect(
            weights["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"]
                != nil)
        #expect(weights["language_model.model.layers.0.mlp.experts.gate_up_proj"] == nil)
        // MTP stays split.
        #expect(weights["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(weights["mtp.layers.0.mlp.switch_mlp.up_proj.weight"] != nil)
        #expect(weights["mtp.layers.0.mlp.switch_mlp.gate_up_proj.weight"] == nil)
    }

    /// PR #107 P1 comment scenario: a mixed-precision checkpoint whose
    /// per-layer quantization table names the split `gate_proj`/`up_proj`
    /// module paths — with NO default quantization — must still quantize the
    /// fused `gate_up_proj` module. `resolveQuantization` consults the
    /// split-path aliases exactly the way `loadWeights` does, the strict
    /// update then accepts the renamed `.scales`/`.biases` tensors, and the
    /// fused forward reproduces the split forward.
    @Test func heterogeneousPerLayerQuantizationLoadsStrictlyThroughAliases() throws {
        struct GateUpAliases: QuantizationPathAliasing {
            func quantizationPathAliases(for path: String) -> [String] {
                qwen35GateUpQuantizationAliases(for: path)
            }
        }

        let inputDims = 128
        let hiddenDims = 64
        let numExperts = 4
        let prefix = "model.layers.0.mlp.switch_mlp"

        // Heterogeneous: W4/g64 experts, W8/g32 down projection, no default.
        let expertQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let downQuant = BaseConfiguration.Quantization(groupSize: 32, bits: 8)
        let table = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(prefix).gate_proj": .quantize(expertQuant),
                "\(prefix).up_proj": .quantize(expertQuant),
                "\(prefix).down_proj": .quantize(downQuant),
            ])

        func randomQuantized(
            _ rows: Int, _ cols: Int, _ quant: BaseConfiguration.Quantization
        ) -> (MLXArray, MLXArray, MLXArray) {
            let w = MLXRandom.normal([numExperts, rows, cols]).asType(.bfloat16)
            let q = quantized(w, groupSize: quant.groupSize, bits: quant.bits, mode: .affine)
            return (q.wq, q.scales, q.biases!)
        }
        let gate = randomQuantized(hiddenDims, inputDims, expertQuant)
        let up = randomQuantized(hiddenDims, inputDims, expertQuant)
        let down = randomQuantized(inputDims, hiddenDims, downQuant)

        var weights: [String: MLXArray] = [
            "\(prefix).gate_proj.weight": gate.0,
            "\(prefix).gate_proj.scales": gate.1,
            "\(prefix).gate_proj.biases": gate.2,
            "\(prefix).up_proj.weight": up.0,
            "\(prefix).up_proj.scales": up.1,
            "\(prefix).up_proj.biases": up.2,
            "\(prefix).down_proj.weight": down.0,
            "\(prefix).down_proj.scales": down.1,
            "\(prefix).down_proj.biases": down.2,
        ]
        weights = qwen35FuseSwitchMLPGateUp(weights: weights)

        // Without aliasing the fused path resolves to nothing — the exact
        // pre-fix failure (module left unquantized, strict update rejects the
        // renamed scales/biases).
        #expect(
            resolveQuantization(
                path: "\(prefix).gate_up_proj", perLayerQuantization: table,
                aliasing: nil) == nil)
        // With aliasing the split experts' policy applies to the fused path;
        // paths with their own entries resolve directly.
        #expect(
            resolveQuantization(
                path: "\(prefix).gate_up_proj", perLayerQuantization: table,
                aliasing: GateUpAliases()) == expertQuant)
        #expect(
            resolveQuantization(
                path: "\(prefix).down_proj", perLayerQuantization: table,
                aliasing: GateUpAliases()) == downQuant)

        // Replicate the loadWeights flow on the fused module: quantize with
        // alias resolution, then a STRICT update with the sanitized weights.
        let fusedGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            fuseGateUp: true)
        quantize(model: fusedGLU) { path, _ in
            let full = "\(prefix).\(path)"
            guard weights["\(full).scales"] != nil else { return nil }
            return resolveQuantization(
                path: full, perLayerQuantization: table,
                aliasing: GateUpAliases())?.asTuple
        }
        let bare = Dictionary(
            uniqueKeysWithValues: weights.map {
                (String($0.key.dropFirst(prefix.count + 1)), $0.value)
            })
        try fusedGLU.update(parameters: ModuleParameters.unflattened(bare), verify: [.all])

        // Split reference carrying the identical quantized codes.
        let splitGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        quantize(model: splitGLU) { path, _ in
            path == "down_proj" ? downQuant.asTuple : expertQuant.asTuple
        }
        try splitGLU.update(
            parameters: ModuleParameters.unflattened([
                "gate_proj.weight": gate.0, "gate_proj.scales": gate.1,
                "gate_proj.biases": gate.2,
                "up_proj.weight": up.0, "up_proj.scales": up.1,
                "up_proj.biases": up.2,
                "down_proj.weight": down.0, "down_proj.scales": down.1,
                "down_proj.biases": down.2,
            ]), verify: [.all])

        let tokens = 16
        let x = MLXRandom.normal([tokens, inputDims]).asType(.bfloat16)
        let indices = MLXRandom.randInt(0 ..< numExperts, [tokens, 8]).asType(.uint32)
        let expected = splitGLU(x, indices)
        let actual = fusedGLU(x, indices)
        eval(expected, actual)
        #expect(expected.shape == actual.shape)
        let maxErr = (actual - expected).abs().max().item(Float.self)
        #expect(
            actual.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            Comment(rawValue: "heterogeneous fused/split divergence: max abs err \(maxErr)"))
    }

    /// PR #107 P2 comment scenario: the `qwen3_5_text` registry entry
    /// constructs `Qwen35TextModel` directly, so its own sanitizer must fuse
    /// routed experts — both converted split checkpoints and raw stacked HF
    /// exports — for the strict update to match the fused module tree.
    @Test func textModelSanitizeFusesExpertsOnDirectLoadPath() throws {
        let json = """
            {
              "model_type": "qwen3_5_text",
              "hidden_size": 32,
              "num_hidden_layers": 2,
              "intermediate_size": 64,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "linear_num_value_heads": 2,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 32,
              "linear_value_head_dim": 32,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 64,
              "full_attention_interval": 2,
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 16,
              "shared_expert_intermediate_size": 16
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = Qwen35TextModel(config)
        let reference = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(reference.keys.contains { $0.hasSuffix(".switch_mlp.gate_up_proj.weight") })

        // Converted split checkpoint: gate/up halves instead of the fused tensor.
        var splitCheckpoint: [String: MLXArray] = [:]
        for (key, value) in reference {
            if key.hasSuffix(".switch_mlp.gate_up_proj.weight") {
                let base = String(key.dropLast("gate_up_proj.weight".count))
                let halves = split(value, parts: 2, axis: -2)
                splitCheckpoint["\(base)gate_proj.weight"] = halves[0]
                splitCheckpoint["\(base)up_proj.weight"] = halves[1]
            } else {
                splitCheckpoint[key] = value
            }
        }
        let sanitizedSplit = model.sanitize(weights: splitCheckpoint)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedSplit), verify: [.all])

        // Raw stacked HF export: `<mlp>.experts.gate_up_proj` + `.experts.down_proj`.
        var rawCheckpoint: [String: MLXArray] = [:]
        for (key, value) in reference {
            if key.hasSuffix(".switch_mlp.gate_up_proj.weight") {
                let base = String(key.dropLast(".switch_mlp.gate_up_proj.weight".count))
                rawCheckpoint["\(base).experts.gate_up_proj"] = value
            } else if key.hasSuffix(".switch_mlp.down_proj.weight") {
                let base = String(key.dropLast(".switch_mlp.down_proj.weight".count))
                rawCheckpoint["\(base).experts.down_proj"] = value
            } else {
                rawCheckpoint[key] = value
            }
        }
        let sanitizedRaw = model.sanitize(weights: rawCheckpoint)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedRaw), verify: [.all])

        // The direct-path model advertises the split-path quantization
        // aliases that `loadWeights` consults for fused modules.
        let aliasing = try #require(model as Any as? QuantizationPathAliasing)
        #expect(
            aliasing.quantizationPathAliases(
                for: "model.layers.0.mlp.switch_mlp.gate_up_proj")
                == [
                    "model.layers.0.mlp.switch_mlp.gate_proj",
                    "model.layers.0.mlp.switch_mlp.up_proj",
                ])
        // MTP trees stay split and get no aliases; unrelated paths get none.
        #expect(
            aliasing.quantizationPathAliases(
                for: "mtp.layers.0.mlp.switch_mlp.gate_up_proj").isEmpty)
        #expect(aliasing.quantizationPathAliases(for: "model.layers.0.mlp.gate").isEmpty)
    }
}
