import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import MLXLLM
#if !MIMO_MTP_COMPONENT_PROBE
import XCTest
#endif

// Prepared for the coordinator's exclusive native lane. This worker did not
// build or run these numerical cases; syntax and metadata checks are separate.
// Shared tiny fixture/oracle for the request-lifecycle tests. Existing 17
// component cases below remain unchanged.
enum MiMoV26MTPChecks {
    struct Failure: Error { let message: String }
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw Failure(message: message) }
    }
    static func config() throws -> MiMoV26Configuration {
        try JSONDecoder().decode(MiMoV26Configuration.self, from: Data("""
        {"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],
         "hidden_size":4,"intermediate_size":8,"moe_intermediate_size":4,
         "vocab_size":32,"num_hidden_layers":2,"max_position_embeddings":64,
         "sliding_window_size":3,"sliding_window":3,"num_nextn_predict_layers":3,
         "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
         "partial_rotary_factor":1,"attention_value_scale":0.707,
         "layernorm_epsilon":0.000001,"attention_projection_layout":"split",
         "moe_router_dtype":"float32","hidden_act":"silu","dtype":"float32",
         "attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0,
         "scoring_func":"sigmoid","topk_method":"noaux_tc","n_routed_experts":2,
         "num_experts_per_tok":1,"n_group":1,"topk_group":1,"norm_topk_prob":true,
         "n_shared_experts":null,"routed_scaling_factor":null,
         "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"v_head_dim":2,
         "swa_num_attention_heads":2,"swa_num_key_value_heads":1,"swa_head_dim":4,
         "swa_v_head_dim":2,"rope_theta":10000000,"swa_rope_theta":10000,
         "add_full_attention_sink_bias":false,"add_swa_attention_sink_bias":true,
         "eos_token_id":3,"pad_token_id":0}
        """.utf8))
    }
    static func fixtureWeights(_ model: Module, prefix: String = "") -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key, value in
            let salt = key.utf8.reduce(0) { ($0 * 31 + Int($1)) % 997 }
            let values = (0..<value.size).map { i -> Float in
                if key.contains("norm.weight") { return 0.8 + Float((salt + i) % 9) * 0.03 }
                return sin(Float(i * 7 + salt)) * 0.2
            }
            return (prefix + key, MLXArray(values, value.shape))
        })
    }
    static func fixture() throws -> (MiMoV26TextModel, MiMoV26MTP) {
        let target = try MiMoV26TextModel(config())
        try target.update(parameters: .unflattened(fixtureWeights(target)), verify: .all)
        let mtp = try MiMoV26MTP(target: target)
        try mtp.loadConvertedWeights(fixtureWeights(mtp, prefix: "mtp."))
        return (target, mtp)
    }

    // Small actual-storage-profile fixture: BF16 activations/norms and all24
    // trained predictor projections affine4/group64, not the trunk MXFP4.
    // Synthetic packed nibbles are constructed directly; expected dense
    // weights never use MLX quantize/dequantize as their oracle.
    static func affineConfiguration(omit: String? = nil, conflict: Bool = false) throws -> MiMoV26Configuration {
        var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(config())) as! [String: Any]
        object["hidden_size"] = 64; object["intermediate_size"] = 128
        object["moe_intermediate_size"] = 64; object["dtype"] = "bfloat16"
        object["moe_router_dtype"] = "bfloat16"
        for key in ["head_dim", "swa_head_dim"] { object[key] = 64 }
        for key in ["v_head_dim", "swa_v_head_dim"] { object[key] = 32 }
        var policies: [String: Any] = ["mode": "mxfp4", "bits": 4, "group_size": 32]
        for depth in 0..<3 {
            for suffix in ["eh_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                let key = "mtp.layers.\(depth).\(suffix)"
                if key != omit { policies[key] = ["mode": "affine", "bits": 4, "group_size": 64] }
            }
        }
        if conflict {
            policies["language_model.mtp.layers.0.eh_proj"] = ["mode": "mxfp4", "bits": 4, "group_size": 32]
        }
        object["quantization"] = policies
        return try JSONDecoder().decode(MiMoV26Configuration.self,
            from: JSONSerialization.data(withJSONObject: object))
    }

    static func packedFixture(_ mtp: MiMoV26MTP, expectedProjections: Int = 24) throws
        -> (stored: [String: MLXArray], expanded: [String: MLXArray]) {
        var stored = fixtureWeights(mtp, prefix: "mtp.").mapValues { $0.asType(.bfloat16) }
        var expanded: [String: MLXArray] = [:]
        let projections = mtp.leafModules().flattened().compactMap { name, module -> (String, QuantizedLinear)? in
            guard let linear = module as? QuantizedLinear else { return nil }
            return (name, linear)
        }
        try require(projections.count == expectedProjections, "declared affine projection count mismatch")
        for (name, linear) in projections {
            try require(linear.mode == .affine && linear.bits == 4 && linear.groupSize == 64,
                        "trunk quantization leaked into predictor")
            let (rows, columns) = linear.shape, groups = columns / 64
            var packed = [UInt32](repeating: 0, count: rows * columns / 8)
            var dense = [Float](repeating: 0, count: rows * columns)
            var scales = [Float](repeating: 0, count: rows * groups)
            let offsets = [Float](repeating: -0.25, count: rows * groups)
            for row in 0..<rows {
                for column in 0..<columns {
                    let code = UInt32((row * 3 + column * 5) % 16)
                    let group = row * groups + column / 64
                    scales[group] = Float(group % 3 + 1) / 64
                    packed[row * (columns / 8) + column / 8] |= code << UInt32((column % 8) * 4)
                    dense[row * columns + column] = Float(code) * scales[group] + offsets[group]
                }
            }
            let prefix = "mtp." + name
            stored[prefix + ".weight"] = MLXArray(packed, [rows, columns / 8])
            stored[prefix + ".scales"] = MLXArray(scales, [rows, groups]).asType(.bfloat16)
            stored[prefix + ".biases"] = MLXArray(offsets, [rows, groups]).asType(.bfloat16)
            expanded[name] = MLXArray(dense, [rows, columns]).asType(.bfloat16)
        }
        return (stored, expanded)
    }
    static func targetOutput(_ target: MiMoV26TextModel, length: Int = 9) throws -> MiMoV26TextOutput {
        try target.forward(inputIDs: MLXArray((1...length).map(Int32.init), [1, length]))
    }
    static func targetFeatures(_ target: MiMoV26TextModel, length: Int = 9) throws -> MiMoV26MTPFeatures {
        try MiMoV26MTPFeatures(targetOutput: targetOutput(target, length: length), target: target)
    }
    static func near(_ actual: MLXArray, _ expected: MLXArray, _ name: String,
                     tolerance: Float = 8e-5) throws {
        eval(actual, expected)
        try require(actual.shape == expected.shape, "\(name): shape mismatch")
        let error = abs(actual - expected).max().item(Float.self)
        try require(error.isFinite && error <= tolerance, "\(name): error=\(error)")
    }
    static func rejected(_ operation: () throws -> Void) throws {
        do {
            try operation()
            throw Failure(message: "invalid MTP operation accepted")
        } catch is MiMoV26MTPError {}
    }

    // Independent scalar CPU math: no MLX norm, attention, RoPE, softmax,
    // activation, cache or concatenation operation forms expected outputs.
    static func linear(_ x: [Float], _ layer: Linear) -> [Float] {
        let weight = layer.weight.asArray(Float.self)
        var y = [Float](repeating: 0, count: layer.weight.dim(0))
        for row in y.indices {
            for column in x.indices { y[row] += weight[row * x.count + column] * x[column] }
        }
        return y
    }
    static func norm(_ x: [Float], _ layer: RMSNorm, eps: Float) -> [Float] {
        let weights = layer.weight.asArray(Float.self)
        let scale = 1 / sqrt(x.reduce(0) { $0 + $1 * $1 } / Float(x.count) + eps)
        return x.indices.map { x[$0] * scale * weights[$0] }
    }
    static func reference(_ head: MiMoV26MTPHead, hidden: [[Float]], tokens: [Int],
                          target: MiMoV26TextModel, firstTokenPosition: Int,
                          reverseConcatenation: Bool = false) -> (hidden: [[Float]], logits: [Float]) {
        let config = target.configuration, width = config.hiddenSize
        let eps = Float(config.layernormEpsilon), geometry = config.slidingAttention
        let table = target.model.embedTokens.weight.asArray(Float.self)
        var projected: [[Float]] = [], queries: [[Float]] = [], keys: [[Float]] = [], values: [[Float]] = []
        func rotate(_ vector: inout [Float], heads: Int, position: Int) {
            let half = geometry.rotaryDimensions / 2
            for head in 0..<heads {
                for dimension in 0..<half {
                    let phase = Float(position) / pow(Float(geometry.ropeTheta), Float(dimension * 2) / Float(geometry.rotaryDimensions))
                    let a = head * geometry.headDim + dimension, b = a + half
                    let first = vector[a], second = vector[b]
                    vector[a] = first * cos(phase) - second * sin(phase)
                    vector[b] = first * sin(phase) + second * cos(phase)
                }
            }
        }
        for row in hidden.indices {
            let embedding = Array(table[tokens[row] * width..<(tokens[row] + 1) * width])
            let e = norm(embedding, head.enorm, eps: eps), h = norm(hidden[row], head.hnorm, eps: eps)
            let projection = linear(reverseConcatenation ? h + e : e + h, head.ehProjection)
            projected.append(projection)
            let normalized = norm(projection, head.inputNorm, eps: eps)
            var q = linear(normalized, head.attention.qProj), k = linear(normalized, head.attention.kProj)
            rotate(&q, heads: geometry.queryHeads, position: firstTokenPosition + row)
            rotate(&k, heads: geometry.keyValueHeads, position: firstTokenPosition + row)
            queries.append(q); keys.append(k)
            values.append(linear(normalized, head.attention.vProj).map { $0 * Float(config.attentionValueScale) })
        }
        let sinks = head.attention.attentionSinkBias!.asArray(Float.self)
        var result: [[Float]] = [], logits: [Float] = []
        for row in hidden.indices {
            var attended = [Float](repeating: 0, count: geometry.queryHeads * geometry.valueHeadDim)
            let start = max(0, row - config.slidingWindow + 1)
            for queryHead in 0..<geometry.queryHeads {
                let kv = queryHead / (geometry.queryHeads / geometry.keyValueHeads)
                var scores: [Float] = []
                for keyRow in start...row {
                    var dot: Float = 0
                    for d in 0..<geometry.headDim {
                        dot += queries[row][queryHead * geometry.headDim + d] * keys[keyRow][kv * geometry.headDim + d]
                    }
                    scores.append(dot / sqrt(Float(geometry.headDim)))
                }
                let maximum = max(scores.max()!, sinks[queryHead])
                let probabilities = scores.map { exp($0 - maximum) }
                let denominator = probabilities.reduce(0, +) + exp(sinks[queryHead] - maximum)
                for (index, keyRow) in (start...row).enumerated() {
                    for d in 0..<geometry.valueHeadDim {
                        attended[queryHead * geometry.valueHeadDim + d] += probabilities[index] / denominator
                            * values[keyRow][kv * geometry.valueHeadDim + d]
                    }
                }
            }
            let attentionOutput = linear(attended, head.attention.oProj)
            let residual = zip(projected[row], attentionOutput).map(+)
            let normalized = norm(residual, head.preMLPNorm, eps: eps)
            let gate = linear(normalized, head.mlp.gateProj), up = linear(normalized, head.mlp.upProj)
            let activated = gate.indices.map { gate[$0] / (1 + exp(-gate[$0])) * up[$0] }
            let down = linear(activated, head.mlp.downProj)
            let output = norm(zip(residual, down).map(+), head.finalNorm, eps: eps)
            result.append(output)
            if let readout = target.lmHead { logits += linear(output, readout) }
            else {
                for token in 0..<config.vocabularySize {
                    logits.append(output.indices.reduce(0) { $0 + output[$1] * table[token * width + $1] })
                }
            }
        }
        return (result, logits)
    }

    static func cases() -> [(String, () throws -> Void)] {
        [
            ("actual affine4 group64 BF16 predictor format loads all90 tensors and executes all3 heads", {
                let target = try MiMoV26TextModel(affineConfiguration())
                let parameters = fixtureWeights(target).mapValues { $0.asType(.bfloat16) }
                try target.update(parameters: .unflattened(parameters), verify: .all)
                let mtp = try MiMoV26MTP(target: target), fixture = try packedFixture(mtp)
                try require(fixture.stored.count == 90, "native predictor tensor count changed")
                var bad = fixture.stored
                bad["mtp.layers.0.eh_proj.scales"] = bad["mtp.layers.0.eh_proj.scales"]!.asType(.float32)
                try rejected { try mtp.loadConvertedWeights(bad) }
                try require(!mtp.isLoaded, "wrong-precision scales loaded")
                try mtp.loadConvertedWeights(fixture.stored)
                for (name, module) in mtp.leafModules().flattened() {
                    guard let linear = module as? QuantizedLinear else { continue }
                    let expected = fixture.expanded[name]!
                    let columns = linear.shape.1
                    let input = MLXArray((0..<(2 * columns)).map { Float(($0 * 7) % 17 - 8) / 32 }, [2, columns]).asType(.bfloat16)
                    try require(linear.weight.asArray(UInt32.self) == fixture.stored["mtp." + name + ".weight"]!.asArray(UInt32.self),
                                "packed predictor codes changed on load")
                    try near(linear(input).asType(.float32), matmul(input, expected.T).asType(.float32),
                             "native affine projection \(name)", tolerance: 0.00390625)
                }
                let features = try targetFeatures(target, length: 3)
                let firstCache = try mtp.newCache(), repeatCache = try mtp.newCache()
                for depth in 0..<3 {
                    let ids = MLXArray([Int32(2 + depth), Int32(3 + depth), Int32(4 + depth)], [1, 3])
                    let result = try mtp.forward(depth: depth, features: features, inputIDs: ids, target: target, cache: firstCache)
                    let repeated = try mtp.forward(depth: depth, features: features, inputIDs: ids, target: target, cache: repeatCache)
                    try require(result.logits.dtype == .bfloat16 && result.logits.shape == [1, 3, 32], "native output contract")
                    try near(result.logits.asType(.float32), repeated.logits.asType(.float32), "affine deterministic repeat", tolerance: 0)
                }
                try require(firstCache.nextTokenPositions == [4, 5, 6], "affine head history mismatch")
            }),
            ("MTP never inherits MXFP4 for a missing or conflicting affine projection policy", {
                for candidate in [try affineConfiguration(omit: "mtp.layers.2.mlp.down_proj"),
                                  try affineConfiguration(conflict: true)] {
                    let target = try MiMoV26TextModel(candidate)
                    try rejected { _ = try MiMoV26MTP(target: target) }
                }
            }),
            ("an explicitly unquantized first head does not break later affine head replacement", {
                var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(affineConfiguration())) as! [String: Any]
                var policies = object["quantization"] as! [String: Any]
                for key in Array(policies.keys) where key.hasPrefix("mtp.layers.0.") { policies[key] = false }
                object["quantization"] = policies
                let configuration = try JSONDecoder().decode(MiMoV26Configuration.self,
                    from: JSONSerialization.data(withJSONObject: object))
                let target = try MiMoV26TextModel(configuration)
                try target.update(parameters: .unflattened(fixtureWeights(target).mapValues { $0.asType(.bfloat16) }), verify: .all)
                let mtp = try MiMoV26MTP(target: target)
                let stored = try packedFixture(mtp, expectedProjections: 16).stored
                try require(stored.count == 74, "preserved first-head tensor closure")
                try require(mtp.layers[0].leafModules().flattened().allSatisfy { !($0.1 is Quantized) },
                            "first head was silently quantized")
                try mtp.loadConvertedWeights(stored)
                let features = try targetFeatures(target, length: 1), cache = try mtp.newCache()
                for depth in 0..<3 {
                    let output = try mtp.forward(depth: depth, features: features,
                        inputIDs: MLXArray([Int32(depth + 2)], [1, 1]), target: target, cache: cache)
                    eval(output.logits)
                    try require(output.logits.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite),
                                "mixed head profile failed")
                }
            }),
            ("fresh positioned-cache copies preserve empty history without invoking state setter", {
                let original = MiMoV26MTPPositionedCache(window: 3, firstPosition: 2)
                let copy = original.copy()
                try require(copy.state.isEmpty && copy.innerState().isEmpty,
                            "fresh copy allocated or fabricated cache state")
                try require(original.offset == 2 && copy.offset == 2 && copy.maxSize == 3,
                            "fresh copy changed absolute position or window")
                try require(copy.metaState == original.metaState, "fresh ring metadata changed")
            }),
            ("all three distinct heads match independent complete scalar oracle", {
                let (target, mtp) = try fixture(), length = 9
                let features = try targetFeatures(target, length: length)
                let flat = features.normalizedHiddenStates.asArray(Float.self)
                let expectedHidden = (0..<length).map { Array(flat[$0 * 4..<($0 + 1) * 4]) }
                let cache = try mtp.newCache()
                var firstValues: [Float] = []
                for depth in 0..<3 {
                    let tokens = (0..<length).map { $0 + depth + 2 }
                    let expected = reference(mtp.layers[depth], hidden: expectedHidden, tokens: tokens,
                                             target: target, firstTokenPosition: depth + 1)
                    let actual = try mtp.forward(depth: depth, features: features,
                                                 inputIDs: MLXArray(tokens.map(Int32.init), [1, length]), target: target, cache: cache)
                    try near(actual.normalizedHiddenStates, MLXArray(expected.hidden.flatMap { $0 }, [1, length, 4]), "head\(depth) hidden")
                    try near(actual.logits, MLXArray(expected.logits, [1, length, 32]), "head\(depth) logits")
                    try require(actual.features.provenance == .predictorPostNorm(depth: depth), "missing output provenance")
                    firstValues.append(actual.logits[0, 0, 0].item(Float.self))
                }
                try require(Set(firstValues).count == 3, "fixture failed to distinguish predictor heads")
                try require(cache.nextTokenPositions == [10, 11, 12], "depth-shifted offsets are wrong")
            }),
            ("embedding-first concatenation has a discriminating negative oracle", {
                let (target, mtp) = try fixture(), features = try targetFeatures(target, length: 1)
                let hidden = [features.normalizedHiddenStates.asArray(Float.self)]
                let correct = reference(mtp.layers[0], hidden: hidden, tokens: [2], target: target, firstTokenPosition: 1)
                let reversed = reference(mtp.layers[0], hidden: hidden, tokens: [2], target: target,
                                         firstTokenPosition: 1, reverseConcatenation: true)
                let difference = zip(correct.logits, reversed.logits).map { abs($0 - $1) }.max()!
                try require(difference > 1e-3, "oracle cannot distinguish concatenation order")
            }),
            ("chunked append agrees across sliding-window wrap", {
                let (target, mtp) = try fixture(), output = try targetOutput(target)
                let features = try MiMoV26MTPFeatures(targetOutput: output, target: target)
                let allIDs = MLXArray((2...10).map(Int32.init), [1, 9])
                let expected = try mtp.forward(depth: 0, features: features, inputIDs: allIDs,
                                               target: target, cache: mtp.newCache())
                let cache = try mtp.newCache()
                var pieces: [MLXArray] = []
                for range in [0..<2, 2..<5, 5..<8, 8..<9] {
                    let segment = try MiMoV26MTPFeatures(targetOutput: output, target: target, range: range)
                    pieces.append(try mtp.forward(depth: 0, features: segment, inputIDs: allIDs[0..., range],
                                                  target: target, cache: cache).logits)
                }
                try near(concatenated(pieces, axis: 1), expected.logits, "window wrap")
                eval(cache)
                try require(cache.consumedTokenCounts == [9, 0, 0], "head caches are not independent")
                try require(cache.stateShapesByDepth[0].map { $0.last! } == [4, 2], "unequal cache widths were changed")
            }),
            ("strict loading rejects absent head norm sink and extra keys before mutation", {
                let target = try MiMoV26TextModel(config()), mtp = try MiMoV26MTP(target: target)
                let complete = fixtureWeights(mtp, prefix: "mtp.")
                for key in ["mtp.layers.2.eh_proj.weight", "mtp.layers.1.pre_mlp_layernorm.weight",
                            "mtp.layers.0.self_attn.attention_sink_bias"] {
                    var bad = complete; bad.removeValue(forKey: key)
                    try rejected { try mtp.loadConvertedWeights(bad) }
                    try require(!mtp.isLoaded, "incomplete predictor set became executable")
                }
                var extra = complete; extra["mtp.lm_head.weight"] = MLXArray.zeros([1])
                try rejected { try mtp.loadConvertedWeights(extra) }
                var wrongShape = complete; wrongShape["mtp.layers.2.eh_proj.weight"] = MLXArray.zeros([1])
                try rejected { try mtp.loadConvertedWeights(wrongShape) }
                try mtp.loadConvertedWeights(complete)
                try require(mtp.isLoaded, "complete trained fixture did not load")
            }),
            ("borrowed target parameters never appear in predictor tree", {
                let (target, mtp) = try fixture()
                let names = mtp.parameters().flattened().map(\.0)
                try require(names.allSatisfy { $0.hasPrefix("layers.") }, "target registered as predictor child")
                try require(names.filter { $0.hasSuffix("pre_mlp_layernorm.weight") }.count == 3,
                            "trained norm spelling changed")
                try require(!names.contains { $0.contains("embed_tokens") || $0.contains("lm_head") }, "readout was duplicated")
                try require(target.model.embedTokens.weight.shape == [32, 4], "target mutated")
            }),
            ("feature provenance depth ownership and history fail before cache writes", {
                let (target, mtp) = try fixture(), features = try targetFeatures(target, length: 1)
                let ids = MLXArray([Int32(2)], [1, 1]), cache = try mtp.newCache()
                for depth in [-1, 3] {
                    try rejected { _ = try mtp.forward(depth: depth, features: features, inputIDs: ids, target: target, cache: cache) }
                }
                let (otherTarget, other) = try fixture()
                try withExtendedLifetime(otherTarget) {
                    let otherCache = try other.newCache()
                    try rejected { _ = try mtp.forward(depth: 0, features: features, inputIDs: ids, target: target, cache: otherCache) }
                }
                let targetCache = target.newCache()
                _ = try target.forward(inputIDs: MLXArray([Int32(1)], [1, 1]), cache: targetCache)
                let displacedOutput = try target.forward(inputIDs: ids, cache: targetCache)
                let displaced = try MiMoV26MTPFeatures(targetOutput: displacedOutput, target: target)
                try rejected { _ = try mtp.forward(depth: 0, features: displaced, inputIDs: ids, target: target, cache: cache) }
                try require(cache.consumedTokenCounts == [0, 0, 0], "failed preflight mutated a cache")
            }),
            ("invalid IDs dimensions dtype and positions fail before cache writes", {
                let (target, mtp) = try fixture(), features = try targetFeatures(target, length: 1)
                let cache = try mtp.newCache()
                for ids in [MLXArray([Int32(-1)], [1, 1]), MLXArray([Int32(32)], [1, 1]),
                            MLXArray([Float(2)], [1, 1]), MLXArray([Int32(2), Int32(3)], [1, 2])] {
                    try rejected { _ = try mtp.forward(depth: 0, features: features, inputIDs: ids, target: target, cache: cache) }
                }
                for hidden in [MLXArray.zeros([1, 1, 3]), MLXArray.zeros([1, 1, 4], dtype: .float16)] {
                    // Internal synthetic metadata intentionally violates the
                    // target-output invariant; public callers cannot mint it.
                    let invalidOutput = MiMoV26TextOutput(logits: MLXArray.zeros([1]),
                        normalizedHiddenStates: hidden, layerFeatures: [:], firstPosition: 0,
                        ownerIdentity: target.identity)
                    try rejected {
                        let bad = try MiMoV26MTPFeatures(targetOutput: invalidOutput, target: target)
                        _ = try mtp.forward(depth: 0, features: bad, inputIDs: MLXArray([Int32(2)], [1, 1]), target: target, cache: cache)
                    }
                }
                for position in [-1, 63, Int.max] {
                    // Explicit internal negative fixture, never a public
                    // promise that callers may choose actual output positions.
                    let invalidOutput = MiMoV26TextOutput(logits: MLXArray.zeros([1]),
                        normalizedHiddenStates: features.normalizedHiddenStates, layerFeatures: [:],
                        firstPosition: position, ownerIdentity: target.identity)
                    try rejected {
                        let bad = try MiMoV26MTPFeatures(targetOutput: invalidOutput, target: target)
                        _ = try mtp.forward(depth: 0, features: bad, inputIDs: MLXArray([Int32(2)], [1, 1]), target: target, cache: cache)
                    }
                }
                try require(cache.consumedTokenCounts == [0, 0, 0], "invalid input appended history")
            }),
            ("an actual target output cannot be relabeled to a different target", {
                let (targetA, mtp) = try fixture(), outputA = try targetOutput(targetA, length: 2)
                let targetB = try MiMoV26TextModel(config()), cache = try mtp.newCache()
                try rejected { _ = try MiMoV26MTPFeatures(targetOutput: outputA, target: targetB) }
                try require(cache.consumedTokenCounts == [0, 0, 0], "owner mismatch mutated cache")
            }),
            ("target feature ranges reject empty negative oversized and overflow bounds", {
                let (target, mtp) = try fixture(), output = try targetOutput(target, length: 4)
                let cache = try mtp.newCache()
                for range in [0..<0, -1..<1, 0..<5, 4..<5, (Int.max - 1)..<Int.max] {
                    try rejected { _ = try MiMoV26MTPFeatures(targetOutput: output, target: target, range: range) }
                }
                try require(cache.consumedTokenCounts == [0, 0, 0], "invalid feature range mutated cache")
            }),
            ("feature positions derive from real cached target outputs and selected rows", {
                let (target, mtp) = try fixture(), targetCache = target.newCache()
                _ = try target.forward(inputIDs: MLXArray([Int32(1), 2, 3], [1, 3]), cache: targetCache)
                let output = try target.forward(inputIDs: MLXArray([Int32(4), 5, 6, 7], [1, 4]), cache: targetCache)
                let full = try MiMoV26MTPFeatures(targetOutput: output, target: target)
                let selected = try MiMoV26MTPFeatures(targetOutput: output, target: target, range: 1..<3)
                try require(output.firstPosition == 3 && full.firstPosition == 3 && selected.firstPosition == 4,
                            "cached target output position was not retained")
                try near(selected.normalizedHiddenStates, output.normalizedHiddenStates[0..., 1..<3, 0...],
                         "selected target rows")
                let emptyHistory = try mtp.newCache()
                try rejected { _ = try mtp.forward(depth: 0, features: selected,
                    inputIDs: MLXArray([Int32(6), 7], [1, 2]), target: target, cache: emptyHistory) }
                try require(emptyHistory.consumedTokenCounts == [0, 0, 0], "invented predictor history was accepted")
            }),
            ("reload invalidates old predictor cache and uninitialized heads refuse forward", {
                let target = try MiMoV26TextModel(config())
                try target.update(parameters: .unflattened(fixtureWeights(target)), verify: .all)
                let mtp = try MiMoV26MTP(target: target), features = try targetFeatures(target, length: 1)
                let ids = MLXArray([Int32(2)], [1, 1])
                try rejected { _ = try mtp.forward(depth: 0, features: features, inputIDs: ids, target: target, cache: mtp.newCache()) }
                let weights = fixtureWeights(mtp, prefix: "mtp.")
                try mtp.loadConvertedWeights(weights)
                let stale = try mtp.newCache()
                try mtp.loadConvertedWeights(weights)
                try rejected { _ = try mtp.forward(depth: 0, features: features, inputIDs: ids, target: target, cache: stale) }
                try require(stale.consumedTokenCounts == [0, 0, 0], "stale cache mutated")
            }),
            ("target lifetime and different live target are enforced", {
                var target: MiMoV26TextModel? = try MiMoV26TextModel(config())
                weak var weakTarget = target
                let mtp = try MiMoV26MTP(target: target!)
                target = nil
                try require(weakTarget == nil, "predictor retains target or duplicated its ownership")
                try rejected { _ = try mtp.newCache() }
                let (live, loaded) = try fixture(), features = try targetFeatures(live, length: 1)
                let other = try MiMoV26TextModel(config())
                try rejected { _ = try loaded.forward(depth: 0, features: features,
                    inputIDs: MLXArray([Int32(2)], [1, 1]), target: other, cache: loaded.newCache()) }
            }),
            ("MiMo reuses target features instead of chaining predictor readouts", {
                let (target, mtp) = try fixture()
                let features = try targetFeatures(target, length: 1)
                let cache = try mtp.newCache()
                let first = try mtp.forward(depth: 0, features: features,
                                            inputIDs: MLXArray([Int32(2)], [1, 1]), target: target, cache: cache)
                try rejected {
                    _ = try mtp.forward(depth: 1, features: first.features,
                                        inputIDs: MLXArray([Int32(3)], [1, 1]), target: target, cache: cache)
                }
                try require(cache.consumedTokenCounts == [1, 0, 0], "rejected predictor chain mutated head1")
                let targetHidden = [features.normalizedHiddenStates.asArray(Float.self)]
                let wrongHidden = [first.normalizedHiddenStates.asArray(Float.self)]
                let expected = reference(mtp.layers[1], hidden: targetHidden, tokens: [3],
                                         target: target, firstTokenPosition: 2)
                let chained = reference(mtp.layers[1], hidden: wrongHidden, tokens: [3],
                                        target: target, firstTokenPosition: 2)
                try require(zip(expected.logits, chained.logits).map { abs($0 - $1) }.max()! > 1e-3,
                            "fixture cannot distinguish target-reuse from incorrect chaining")
                let actual = try mtp.forward(depth: 1, features: features,
                                             inputIDs: MLXArray([Int32(3)], [1, 1]), target: target, cache: cache)
                try near(actual.logits, MLXArray(expected.logits, [1, 1, 32]), "target-feature reuse")
                try require(cache.nextTokenPositions == [2, 3, 3], "independent head positions changed")
            }),
        ]
    }
}

#if MIMO_MTP_COMPONENT_PROBE
@main
private struct MiMoV26MTPComponentProbe {
    static func main() throws {
        let cases = MiMoV26MTPChecks.cases()
        var failures = 0
        for (name, body) in cases {
            do { try body(); print("PASS \(name)") }
            catch { failures += 1; print("FAIL \(name): \(error)") }
        }
        print("RESULT discovered=\(cases.count) passed=\(cases.count - failures) failed=\(failures) skipped=0")
        if failures != 0 { throw MiMoV26MTPChecks.Failure(message: "native MTP component probe failed") }
    }
}
#else
final class MiMoV26MTPTests: XCTestCase {
    func testNativeNextNHeadContracts() throws {
        for (name, body) in MiMoV26MTPChecks.cases() {
            do { try body() } catch { XCTFail("\(name): \(error)") }
        }
    }
}
#endif
