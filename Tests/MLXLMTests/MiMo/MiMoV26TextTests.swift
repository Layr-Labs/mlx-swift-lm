import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXLLM
#if !MIMO_NATIVE_COMPONENT_PROBE
import XCTest
#endif

// These assertions are shared by XCTest and the explicitly owned tiny native
// probe. They do not load a checkpoint or qualify a serving/model profile.
private enum MiMoV26TextChecks {
    struct Failure: Error { let message: String }
    static func require(_ result: Bool, _ message: String) throws {
        if !result { throw Failure(message: message) }
    }

    static func configuration() throws -> MiMoV26Configuration {
        try JSONDecoder().decode(MiMoV26Configuration.self, from: Data("""
        {"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],
         "hidden_size":16,"intermediate_size":32,"moe_intermediate_size":8,
         "vocab_size":32,"num_hidden_layers":2,"max_position_embeddings":64,
         "sliding_window_size":7,"sliding_window":7,"num_nextn_predict_layers":3,
         "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
         "partial_rotary_factor":0.5,"attention_value_scale":0.707,
         "layernorm_epsilon":0.000001,"attention_projection_layout":"split",
         "moe_router_dtype":"float32","hidden_act":"silu","dtype":"float32",
         "attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0,
         "scoring_func":"sigmoid","topk_method":"noaux_tc","n_routed_experts":4,
         "num_experts_per_tok":2,"n_group":1,"topk_group":1,"norm_topk_prob":true,
         "n_shared_experts":null,"routed_scaling_factor":null,
         "num_attention_heads":4,"num_key_value_heads":1,"head_dim":8,"v_head_dim":4,
         "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
         "swa_v_head_dim":4,"rope_theta":10000000,"swa_rope_theta":10000,
         "add_full_attention_sink_bias":false,"add_swa_attention_sink_bias":true,
         "eos_token_id":3,"pad_token_id":0}
        """.utf8))
    }

    static func fill(_ module: Module) throws {
        let values = module.parameters().flattened().map { name, array in
            let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 997 }
            let data = (0..<array.size).map { i -> Float in
                if name.contains("norm.weight") { return 1 }
                return sin(Float((i * 13 + salt) % 109)) * 0.15
            }
            return (name, MLXArray(data, array.shape))
        }
        try module.update(parameters: .unflattened(values), verify: .all)
    }

    static func near(_ actual: MLXArray, _ expected: MLXArray, _ label: String,
                     tolerance: Float = 2e-5) throws {
        eval(actual, expected)
        try require(actual.shape == expected.shape, label + " shape")
        let error = abs(actual - expected).max().item(Float.self)
        try require(error.isFinite && error <= tolerance, "\(label) error=\(error)")
    }

    static func linear(_ x: [Float], _ weight: MLXArray) -> [Float] {
        let w = weight.asType(.float32).asArray(Float.self), outputs = weight.dim(0)
        var result = [Float](repeating: 0, count: outputs)
        for o in 0..<outputs {
            for i in x.indices { result[o] += x[i] * w[o * x.count + i] }
        }
        return result
    }

    // Deliberately independent scalar CPU attention oracle: no MLX attention,
    // softmax, cache or RoPE operation participates in expected outputs.
    static func attentionReference(_ layer: MiMoV26Attention, input: [Float],
                                   length: Int, hidden: Int) -> [Float] {
        let g = layer.geometry, qd = g.headDim, vd = g.valueHeadDim
        let groups = g.queryHeads / g.keyValueHeads
        var qs: [[Float]] = [], ks: [[Float]] = [], vs: [[Float]] = []
        func rotate(_ vector: inout [Float], heads: Int, position: Int) {
            let half = g.rotaryDimensions / 2
            for h in 0..<heads {
                for i in 0..<half {
                    let angle = Float(position) / pow(Float(g.ropeTheta), Float(2 * i) / Float(g.rotaryDimensions))
                    let a = h * qd + i, b = a + half
                    let x = vector[a], y = vector[b]
                    vector[a] = x * cos(angle) - y * sin(angle)
                    vector[b] = x * sin(angle) + y * cos(angle)
                }
            }
        }
        for t in 0..<length {
            let x = Array(input[t * hidden..<(t + 1) * hidden])
            var q = linear(x, layer.qProj.weight), k = linear(x, layer.kProj.weight)
            let v = linear(x, layer.vProj.weight).map { $0 * layer.valueScale }
            rotate(&q, heads: g.queryHeads, position: t)
            rotate(&k, heads: g.keyValueHeads, position: t)
            qs.append(q); ks.append(k); vs.append(v)
        }
        let sinks = layer.attentionSinkBias?.asArray(Float.self)
        var output: [Float] = []
        for t in 0..<length {
            var attended = [Float](repeating: 0, count: g.queryHeads * vd)
            let first = g.slidingWindow.map { max(0, t - $0 + 1) } ?? 0
            for h in 0..<g.queryHeads {
                let kvh = h / groups
                var logits: [Float] = []
                for k in first...t {
                    var dot: Float = 0
                    for d in 0..<qd { dot += qs[t][h * qd + d] * ks[k][kvh * qd + d] }
                    logits.append(dot * layer.scale)
                }
                let maximum = max(logits.max()!, sinks?[h] ?? -Float.infinity)
                let probabilities = logits.map { exp($0 - maximum) }
                let denominator = probabilities.reduce(0, +) + (sinks.map { exp($0[h] - maximum) } ?? 0)
                for (index, k) in (first...t).enumerated() {
                    for d in 0..<vd {
                        attended[h * vd + d] += probabilities[index] / denominator * vs[k][kvh * vd + d]
                    }
                }
            }
            output += linear(attended, layer.oProj.weight)
        }
        return output
    }

    static func cases() -> [(String, () throws -> Void)] {
        [
            ("global and sliding attention match independent scalar oracle", {
                let config = try configuration(), length = 9
                let input = (0..<length * config.hiddenSize).map { cos(Float($0 * 3)) * 0.2 }
                let x = MLXArray(input, [1, length, config.hiddenSize])
                for index in 0..<2 {
                    let g = try config.attentionGeometry(at: index)
                    let layer = MiMoV26Attention(config, geometry: g)
                    try fill(layer)
                    let mask = createAttentionMask(h: x, cache: nil, windowSize: g.slidingWindow)
                    let actual = layer(x, mask: mask, cache: nil)
                    let reference = attentionReference(layer, input: input, length: length, hidden: config.hiddenSize)
                    try near(actual, MLXArray(reference, [1, length, config.hiddenSize]), "attention \(index)")
                }
            }),
            ("learned sinks are required by strict loading", {
                let config = try configuration()
                let layer = MiMoV26Attention(config, geometry: config.slidingAttention)
                let parameters = layer.parameters().flattened().filter { $0.0 != "attention_sink_bias" }
                do {
                    try layer.update(parameters: .unflattened(parameters), verify: .all)
                    throw Failure(message: "missing trained sink accepted")
                } catch is Failure { throw Failure(message: "missing trained sink accepted") }
                  catch { }
            }),
            ("prefill chunk decode and cache widths agree", {
                let config = try configuration(), model = try MiMoV26TextModel(config)
                try fill(model)
                let ids = MLXArray((1...9).map(Int32.init), [1, 9])
                let expected = try model.forward(inputIDs: ids, captureLayers: [0, 1])
                let cache = model.newCache()
                var pieces: [MLXArray] = []
                for range in [0..<3, 3..<7, 7..<8, 8..<9] {
                    pieces.append(try model.forward(inputIDs: ids[0..., range], cache: cache).logits)
                }
                try near(concatenated(pieces, axis: 1), expected.logits, "chunk/decode", tolerance: 5e-5)
                try require(expected.layerFeatures.count == 2, "missing feature capture")
                for (index, entry) in cache.enumerated() {
                    eval(entry)
                    try require(entry.offset == 9 && entry.state[0].dim(3) == 8
                                && entry.state[1].dim(3) == 4, "cache geometry \(index)")
                }
                let frontier = try model.forward(inputIDs: ids, logitsStart: 8)
                try near(frontier.logits, expected.logits[0..., 8..., 0...], "frontier")
            }),
            ("rejections precede all cache mutation", {
                let config = try configuration(), model = try MiMoV26TextModel(config)
                let ids = MLXArray([Int32(1)], [1, 1])
                var cache = model.newCache()
                cache[1] = ArraysCache(size: 2)
                do {
                    _ = try model.forward(inputIDs: ids, cache: cache)
                    throw Failure(message: "unsupported cache accepted")
                } catch is MiMoV26ExecutionError { }
                try require(cache[0].offset == 0 && cache[0].state.isEmpty, "partial mutation")
                for invalid in [-1, 1] {
                    do {
                        _ = try model.forward(inputIDs: ids, logitsStart: invalid)
                        throw Failure(message: "invalid frontier accepted")
                    } catch is MiMoV26ExecutionError { }
                }
                do {
                    _ = try model.forward(inputIDs: MLXArray.zeros([1, 65], dtype: .int32))
                    throw Failure(message: "context overflow accepted")
                } catch MiMoV26ExecutionError.contextExceeded { }
            }),
            ("unsupported preserved operational fields reject construction", {
                for (key, value) in [("n_shared_experts", MiMoV26JSONValue.number(2)),
                                     ("rope_scaling", .object(["type": .string("linear")])),
                                     ("attention_dropout", .number(0.125))] {
                    var fields = try configuration().rawFields
                    fields[key] = value
                    let config = try MiMoV26Configuration(rawFields: fields)
                    do {
                        _ = try MiMoV26TextModel(config)
                        throw Failure(message: "unsupported \(key) accepted")
                    } catch is MiMoV26ExecutionError { }
                }
            }),
            ("invalid ring policy and metadata do not mutate prior layers", {
                let model = try MiMoV26TextModel(configuration())
                let ids = MLXArray([Int32(1)], [1, 1])
                for metadata in [["1", "7", "256", "0", "0"],
                                 ["0", "7", "0", "0", "0"],
                                 ["0", "7", "256", "0", "8"],
                                 ["0", "7", "-1", "0", "0"]] {
                    let cache = model.newCache()
                    (cache[1] as! RotatingKVCache).metaState = metadata
                    do {
                        _ = try model.forward(inputIDs: ids, cache: cache)
                        throw Failure(message: "bad ring accepted")
                    } catch is MiMoV26ExecutionError { }
                    try require(cache[0].offset == 0 && cache[0].state.isEmpty, "ring caused partial append")
                }
                let missing = model.newCache()
                (missing[0] as! BaseKVCache).offset = 8
                (missing[1] as! BaseKVCache).offset = 8
                do {
                    _ = try model.forward(inputIDs: ids, cache: missing)
                    throw Failure(message: "missing prefix backing accepted")
                } catch is MiMoV26ExecutionError { }
                try require(missing[0].offset == 8 && missing[0].state.isEmpty, "empty-prefix mutation")
            }),
            ("aliased layer caches fail before first append", {
                var fields = try configuration().rawFields
                fields["hybrid_layer_pattern"] = .array([.number(1), .number(1)])
                let model = try MiMoV26TextModel(MiMoV26Configuration(rawFields: fields))
                let shared = RotatingKVCache(maxSize: 7)
                do {
                    _ = try model.forward(inputIDs: MLXArray([Int32(1)], [1, 1]), cache: [shared, shared])
                    throw Failure(message: "aliased cache accepted")
                } catch is MiMoV26ExecutionError { }
                try require(shared.offset == 0 && shared.state.isEmpty, "alias mutated before rejection")
            }),
            ("integer and scalar narrowing reject before allocation", {
                for (key, value) in [("hidden_size", MiMoV26JSONValue.number(Decimal(Int64(Int32.max) + 1))),
                                     ("vocab_size", .number(Decimal(Int64(Int32.max) + 1))),
                                     ("attention_value_scale", .number(Decimal(string: "1e100")!)),
                                     ("layernorm_epsilon", .number(Decimal(string: "1e-100")!)),
                                     ("rope_theta", .number(Decimal(string: "1e100")!))] {
                    var fields = try configuration().rawFields
                    fields[key] = value
                    let config = try MiMoV26Configuration(rawFields: fields)
                    do {
                        _ = try MiMoV26TextModel(config)
                        throw Failure(message: "unsafe narrowing accepted for \(key)")
                    } catch is MiMoV26ExecutionError { }
                }
            }),
            ("invalid token values reject without cache mutation", {
                let model = try MiMoV26TextModel(configuration())
                for ids in [MLXArray([Int32(-1)], [1, 1]), MLXArray([Int32(32)], [1, 1]),
                            MLXArray([UInt32.max], [1, 1])] {
                    let cache = model.newCache()
                    do {
                        _ = try model.forward(inputIDs: ids, cache: cache)
                        throw Failure(message: "invalid token accepted")
                    } catch is MiMoV26ExecutionError { }
                    try require(cache.allSatisfy { $0.offset == 0 && $0.state.isEmpty }, "token caused mutation")
                }
            }),
            ("actual target head geometry matches scalar attention", {
                var fields = try configuration().rawFields
                for (key, value) in [("num_attention_heads", 64), ("num_key_value_heads", 4),
                                     ("swa_num_attention_heads", 64), ("swa_num_key_value_heads", 8),
                                     ("head_dim", 192), ("swa_head_dim", 192),
                                     ("v_head_dim", 128), ("swa_v_head_dim", 128),
                                     ("sliding_window", 128), ("sliding_window_size", 128),
                                     ("max_position_embeddings", 256)] {
                    fields[key] = .number(Decimal(value))
                }
                fields["partial_rotary_factor"] = .number(Decimal(string: "0.334")!)
                let config = try MiMoV26Configuration(rawFields: fields), length = 3
                let input = (0..<length * config.hiddenSize).map { sin(Float($0)) * 0.1 }
                let x = MLXArray(input, [1, length, config.hiddenSize])
                for index in 0..<2 {
                    let geometry = try config.attentionGeometry(at: index)
                    try require(geometry.rotaryDimensions == 64, "native rotary width changed")
                    let layer = MiMoV26Attention(config, geometry: geometry)
                    try fill(layer)
                    let expected = attentionReference(layer, input: input, length: length, hidden: config.hiddenSize)
                    let actual = layer(x, mask: createAttentionMask(h: x, cache: nil,
                                       windowSize: geometry.slidingWindow), cache: nil)
                    let reference = MLXArray(expected, [1, length, config.hiddenSize])
                    try near(actual, reference, "native geometry")
                    let cache: KVCache = geometry.slidingWindow.map { RotatingKVCache(maxSize: $0) }
                        ?? KVCacheSimple()
                    var decoded: [MLXArray] = []
                    for token in 0..<length {
                        let row = x[0..., token..<token + 1, 0...]
                        decoded.append(layer(row, mask: createAttentionMask(h: row, cache: cache,
                                             windowSize: geometry.slidingWindow), cache: cache))
                    }
                    try near(concatenated(decoded, axis: 1), reference, "native GQA decode")
                }
            }),
            ("router dtype grouped selection and weights match scalar reference", {
                for dtype in ["float32", "bfloat16"] {
                    var fields = try configuration().rawFields
                    fields["moe_router_dtype"] = .string(dtype)
                    fields["n_group"] = .number(2)
                    fields["topk_group"] = .number(1)
                    let config = try MiMoV26Configuration(rawFields: fields)
                    let router = MiMoV26Router(config)
                    try fill(router)
                    let bias: [Float] = [-2, -1, 1, 2]
                    router.update(parameters: .unflattened([("e_score_correction_bias", MLXArray(bias))]))
                    let input = MLXArray((0..<16).map { cos(Float($0)) * 0.125 }, [1, 1, 16])
                    let operands = input.asType(router.operandDType).asType(.float32).asArray(Float.self)
                    let dots = linear(operands, router.weight.asType(router.operandDType))
                    let scores = dots.map { 1 / (1 + exp(-$0)) }
                    let expectedIDs = [2, 3] // Distinct corrected group scores force the second group.
                    let denominator = scores[2] + scores[3] + Float(1e-20)
                    let result = router(input)
                    let ids = result.indices.asType(.int32).asArray(Int32.self).map(Int.init)
                    let weights = result.weights.asArray(Float.self)
                    try require(ids.sorted() == expectedIDs, "wrong selected group/experts")
                    for index in ids.indices {
                        try require(abs(weights[index] - scores[ids[index]] / denominator) < 1e-6,
                                    "router weight changed")
                    }
                }
            }),
            ("BF16 chunks retain native cache dtype across the window", {
                var fields = try configuration().rawFields
                fields["dtype"] = .string("bfloat16")
                fields["moe_router_dtype"] = .string("bfloat16")
                let model = try MiMoV26TextModel(MiMoV26Configuration(rawFields: fields))
                try fill(model)
                try model.update(parameters: .unflattened(model.parameters().flattened().map { name, value in
                    (name, value.asType(name.contains("e_score_correction_bias") ? .float32 : .bfloat16))
                }), verify: .all)
                let ids = MLXArray((0..<33).map { Int32($0 % 29 + 1) }, [1, 33])
                let expected = try model.forward(inputIDs: ids).logits
                let cache = model.newCache()
                let a = try model.forward(inputIDs: ids[0..., 0..<31], cache: cache).logits
                let b = try model.forward(inputIDs: ids[0..., 31..<32], cache: cache).logits
                let c = try model.forward(inputIDs: ids[0..., 32..<33], cache: cache).logits
                try near(concatenated([a, b, c], axis: 1).asType(.float32), expected.asType(.float32),
                         "BF16 chunk agreement", tolerance: 0.02)
                try require(cache.allSatisfy { $0.offset == 33 && $0.state.allSatisfy { $0.dtype == .bfloat16 } },
                            "BF16 cache dtype or offset changed")
            }),
            ("cache dtype and rank corruption reject before any mutation", {
                let model = try MiMoV26TextModel(configuration())
                let ids = MLXArray([Int32(1)], [1, 1])
                for invalid in [
                    [MLXArray.zeros([1, 2, 1, 8], dtype: .int32), MLXArray.zeros([1, 2, 1, 4], dtype: .int32)],
                    [MLXArray.zeros([1, 2, 1, 8]), MLXArray.zeros([1, 2, 1, 4], dtype: .bfloat16)],
                    [MLXArray.zeros([1, 1]), MLXArray.zeros([1, 1])]
                ] {
                    let cache = model.newCache()
                    (cache[0] as! KVCacheSimple).state = [MLXArray.zeros([1, 1, 1, 8]),
                                                         MLXArray.zeros([1, 1, 1, 4])]
                    let ring = cache[1] as! RotatingKVCache
                    ring.state = invalid
                    ring.metaState = ["0", "7", "256", "1", "1"]
                    let initial = (cache[0] as! BaseKVCache).innerState().map(\.shape)
                    do {
                        _ = try model.forward(inputIDs: ids, cache: cache)
                        throw Failure(message: "corrupt cache dtype/rank accepted")
                    } catch is MiMoV26ExecutionError { }
                    try require(cache[0].offset == 1 && (cache[0] as! BaseKVCache).innerState().map(\.shape) == initial,
                                "corrupt later cache changed earlier owner")
                }
            }),
        ]
    }

    static func run() throws {
        for (name, test) in cases() {
            try test()
            print("PASS \(name)")
        }
        print("RESULT groups=\(cases().count) failed=0 skipped=0")
    }
}

#if MIMO_NATIVE_COMPONENT_PROBE
@main struct MiMoV26TextProbe {
    static func main() throws { try MiMoV26TextChecks.run() }
}
#else
final class MiMoV26TextTests: XCTestCase {
    func testNativeTextComponents() throws { try MiMoV26TextChecks.run() }
}
#endif
