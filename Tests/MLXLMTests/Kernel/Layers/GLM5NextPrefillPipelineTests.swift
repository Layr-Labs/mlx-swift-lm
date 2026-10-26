// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

/// One TP2 rank's DSA layer geometry (32 heads, hidden 4096, q_lora 1536,
/// kv_lora 512, 256-wide nope/value heads, 32 x 128 indexer, top 2048) with
/// deterministic synthetic Q4 weights, built as `GLM5NextKDALayerFixture`.
struct GLM5NextDSARankFixture {
    let configuration: GLM5NextConfiguration
    let inventory: GLM5NextWeightInventory
    var text: GLM5NextTextConfiguration { configuration.textConfig }
    static let prefix = "model.language_model.layers.3.self_attn"

    init(heads: Int = 32, hidden: Int = 4096) throws {
        let base = try GLM5NextInventoryFixture.configuration(layers: 4, q4: true)
        var root = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
        var text = try #require(root["text_config"] as? [String: Any])
        for (key, value) in [
            ("hidden_size", hidden), ("num_attention_heads", heads), ("num_key_value_heads", heads),
            ("qk_head_dim", 256), ("qk_nope_head_dim", 256), ("v_head_dim", 256),
            ("q_lora_rank", 1536), ("kv_lora_rank", 512), ("index_head_dim", 128),
            ("index_n_heads", 32), ("index_topk", 2048), ("index_kpool", 4),
        ] { text[key] = value }
        root["text_config"] = text
        var vision = try #require(root["vision_config"] as? [String: Any])
        vision["out_hidden_size"] = hidden
        root["vision_config"] = vision
        configuration = try JSONDecoder().decode(
            GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        inventory = try GLM5NextWeightInventory.validate(
            configuration: configuration,
            tensors: GLM5NextWeightInventory.expectedTensors(
                configuration: configuration, storage: .vontraQ4, shard: "dsa.safetensors"),
            storage: .vontraQ4)
    }

    private static func scale(_ index: Int) -> Float {
        (1 + Float(2 * (index % 64) + 1) / 128) / 32
    }

    func tensor(_ name: String) throws -> MLXArray {
        let header = try #require(inventory.entries[name]).tensor
        let count = header.shape.reduce(1, *)
        let salt = name.utf8.reduce(0) { ($0 * 31 + Int($1)) % 1009 }
        if header.dtype == "U32" {
            return MLXArray(
                (0 ..< count).map {
                    UInt32(
                        truncatingIfNeeded: (UInt64($0 + salt * 7919) &* 0x9E37_79B9_7F4A_7C15)
                            >> 29)
                }, header.shape)
        }
        let values: [Float] = (0 ..< count).map { index in
            let wave = Float(sin(Double(index) * 0.731 + Double(salt)))
            if name.hasSuffix(".scales") { return Self.scale(index) }
            if name.hasSuffix(".biases") { return -8 * Self.scale(index) }
            if name.contains("norm.weight") { return 1 + 0.25 * wave }
            if name.contains("norm.bias") { return 0.05 * wave }
            return 0.25 * wave
        }
        let array = MLXArray(values, header.shape)
        return header.dtype == "BF16" ? array.asType(.bfloat16) : array
    }

    func attention(policy: GLM5NextExecutionPolicy = .serving) throws -> GLM5NextDSAAttention {
        try GLM5NextDSAAttention(
            text, prefix: Self.prefix,
            loader: GLM5NextTensorLoader(inventory, tensor, releaseSourceTensors: { _ in }),
            executionPolicy: policy)
    }

    func input(tokens: Int, seed: UInt64) -> MLXArray {
        (MLXRandom.normal([1, tokens, text.hiddenSize], key: MLXRandom.key(seed)) * 0.5)
            .asType(.bfloat16)
    }
}

extension KernelTests {
    fileprivate enum Pipeline {
        /// Same bits: equal shape and dtype, and equal raw bit patterns
        /// (so -0 vs +0 and NaN payloads count as differences).
        static func bitwise(_ a: MLXArray, _ b: MLXArray) -> Bool {
            guard a.shape == b.shape, a.dtype == b.dtype else { return false }
            let raw: DType =
                switch a.dtype.size {
                case 1: .uint8
                case 2: .uint16
                case 4: .uint32
                default: .uint64
                }
            return (a.view(dtype: raw) .== b.view(dtype: raw)).all().item(Bool.self)
        }

        /// Median wall milliseconds of `n` synchronized evaluations of `body`.
        static func median(_ n: Int, _ body: () throws -> [MLXArray]) rethrows -> Double {
            var samples: [Double] = []
            for _ in 0 ..< n {
                let start = ContinuousClock.now
                let outputs = try body()
                eval(outputs)
                Stream.gpu.synchronize()
                let elapsed = ContinuousClock.now - start
                samples.append(
                    Double(elapsed.components.seconds) * 1000
                        + Double(elapsed.components.attoseconds) / 1e15)
            }
            samples.sort()
            return samples[n / 2]
        }

        static func report(_ label: String, _ ms: Double) {
            print(String(format: "GLM53_PREFILLK %@ ms=%.3f", label, ms))
        }
    }

    /// Issuing the DSA sparse rows' absorbed query product beside the index
    /// scores (`GLM5NextDSAAttention.overlapsAbsorbedQueries`) keeps every
    /// output bit, at one TP2 rank's DSA geometry, with and without history,
    /// with contiguous and with paged histories (the cluster's).
    @Suite(.serialized)
    struct GLM5NextDSAOverlapTests {
        @Test(arguments: [false, true])
        func overlapKeepsEveryBit(paged: Bool) throws {
            #expect(glm5NextNAXAvailable)
            guard glm5NextNAXAvailable else { return }
            try Device.withDefaultDevice(.gpu) {
                let tokens = 4096
                let fixture = try GLM5NextDSARankFixture()
                let dsa = try fixture.attention()
                defer { dsa.overlapsAbsorbedQueries = true }
                let cache: GLM5NextDSACache
                if paged {
                    let types = try dsa.historyTypes(input: fixture.input(tokens: 1, seed: 1).dtype)
                    let c = fixture.text
                    let pool = try GLM5NextDSAPagePool(
                        pageSize: 64, latentWidth: c.kvLoraRank, keyWidth: c.indexHeadDim,
                        latentType: types.latents, keyType: types.keys, gateType: types.gates)
                    cache = GLM5NextDSACache(pagePool: pool, account: try pool.reserve(pages: 2048))
                } else {
                    cache = GLM5NextDSACache()
                }
                let all = MLXArray.ones([1, tokens], dtype: .bool)
                for step in 0 ..< 4 {
                    let x = fixture.input(tokens: tokens, seed: UInt64(200 + step))
                    eval(x)
                    var outputs: [MLXArray] = []
                    for overlap in [false, true] {
                        dsa.overlapsAbsorbedQueries = overlap
                        let copy = cache.copy() as! GLM5NextDSACache
                        let y = try dsa.forward(x, cache: copy, validity: all)
                        eval([y] + copy.state)
                        outputs.append(y)
                    }
                    #expect(outputs[0].abs().max().item(Float.self).isFinite)
                    #expect(
                        Pipeline.bitwise(outputs[1], outputs[0]),
                        "history \(step * tokens) paged \(paged)")
                    dsa.overlapsAbsorbedQueries = true
                    _ = try dsa.forward(x, cache: cache, validity: all)
                    eval(cache.state)
                }
            }
        }
    }

    /// Per-kernel prefill timings at one TP2 rank's KDA and DSA geometry and
    /// the KDA recurrence co-residency probes. Enabled by `GLM53_PREFILLK_BENCH`.
    @Suite(
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["GLM53_PREFILLK_BENCH"] != nil))
    struct GLM5NextPrefillPipelineBenchTests {
        static let repeats = 9

        @Test func kdaStages() throws {
            try Device.withDefaultDevice(.gpu) {
                let tokens = 4096
                let fixture = try GLM5NextKDALayerFixture(heads: 32, hidden: 4096)
                let layer = try fixture.sublayer()
                let x = fixture.input(tokens: tokens, seed: 21)
                let validity = MLXArray.ones([1, tokens], dtype: .bool)
                eval(x, validity)
                let c = fixture.text.linearAttention
                let shape = [1, tokens, c.numHeads, c.headDim]
                let fused = try #require(layer.fusedInputs)
                Pipeline.report(
                    "kda layer",
                    try Pipeline.median(Self.repeats) {
                        let r = try layer.forward(x, state: nil, validity: validity)
                        return [r.values, r.state.recurrent, r.state.convolution]
                    })
                let rows = try fused.project(x, prefill: .mlx)
                eval(rows.output)
                let m = rows.members
                Pipeline.report(
                    "kda fused_in_qmm n=\(rows.output.dim(-1))",
                    try Pipeline.median(Self.repeats) {
                        [try fused.project(x, prefill: .mlx).output]
                    })
                Pipeline.report(
                    "kda fused_in_qmm_half",
                    try Pipeline.median(Self.repeats) {
                        [try fused.project(x[0..., ..<(tokens / 2)], prefill: .mlx).output]
                    })
                let forget = layer.fB(m[3]).reshaped(shape)
                let gate = layer.gB(m[4]).reshaped(shape)
                eval(forget, gate)
                Pipeline.report(
                    "kda f_b(copy+qmm)", Pipeline.median(Self.repeats) { [layer.fB(m[3])] })
                Pipeline.report(
                    "kda g_b(copy+qmm)", Pipeline.median(Self.repeats) { [layer.gB(m[4])] })
                let inputs = GLM5NextLinearDecodeKernel.Inputs(
                    query: m[0].reshaped(shape), key: m[1].reshaped(shape),
                    value: m[2].reshaped(shape),
                    gates: .projected(forget: forget, output: gate), betaLogits: m[5],
                    queryConvolution: layer.qConv.weight, keyConvolution: layer.kConv.weight,
                    valueConvolution: layer.vConv.weight, aLog: layer.aLog, dtBias: layer.dtBias,
                    outputNorm: layer.outputNorm.weight, state: nil, validity: validity,
                    projectionRows: .init(buffer: rows.output, betaColumn: fused.columns[5]))
                let zeros = MLXArray.zeros([1, 3 * c.numHeads * c.headDim, 4], dtype: .bfloat16)
                let state = MLXArray.zeros([1, c.numHeads, c.headDim, c.headDim], dtype: .float32)
                let walked = GLM5NextLinearDecodeKernel.longConvolution(inputs, convolution: zeros)
                eval(walked.activated, walked.convolution, zeros, state)
                Pipeline.report(
                    "kda conv(blocked,in-place)",
                    Pipeline.median(Self.repeats) {
                        let w = GLM5NextLinearDecodeKernel.longConvolution(
                            inputs, convolution: zeros)
                        return [w.activated, w.convolution]
                    })
                let rec = GLM5NextLinearDecodeKernel.longRecurrence(
                    inputs, activated: walked.activated, forget: forget, recurrent: state,
                    gateLowerBound: -5, queryScale: 1 / Foundation.sqrt(128))
                eval(rec.core, rec.state)
                Pipeline.report(
                    "kda recurrence(per-core)",
                    Pipeline.median(Self.repeats) {
                        let r = GLM5NextLinearDecodeKernel.longRecurrence(
                            inputs, activated: walked.activated, forget: forget, recurrent: state,
                            gateLowerBound: -5, queryScale: 1 / Foundation.sqrt(128))
                        return [r.core, r.state]
                    })
                Pipeline.report(
                    "kda output_norm",
                    Pipeline.median(Self.repeats) {
                        [
                            GLM5NextLinearDecodeKernel.longOutput(
                                inputs, core: rec.core, outputGate: gate, normEpsilon: 1e-5)
                        ]
                    })
                let values = GLM5NextLinearDecodeKernel.longOutput(
                    inputs, core: rec.core, outputGate: gate, normEpsilon: 1e-5
                ).reshaped(1, tokens, c.numHeads * c.headDim)
                eval(values)
                Pipeline.report(
                    "kda o_proj", Pipeline.median(Self.repeats) { [layer.output(values)] })
                Pipeline.report(
                    "sync_empty", Pipeline.median(Self.repeats) { [x[0..., 0 ..< 1] + 0] })
                // Overlap probe: the recurrence and an independent half-row
                // projection in one evaluation, in both dispatch orders.
                let xh = x[0..., ..<(tokens / 2)]
                func recur() -> MLXArray {
                    GLM5NextLinearDecodeKernel.longRecurrence(
                        inputs, activated: walked.activated, forget: forget, recurrent: state,
                        gateLowerBound: -5, queryScale: 1 / Foundation.sqrt(128)
                    ).core
                }
                Pipeline.report(
                    "probe rec_then_qmm",
                    try Pipeline.median(Self.repeats) {
                        let r = recur()
                        let p = try fused.project(
                            depends(input: xh, dependencies: [r]), prefill: .mlx
                        ).output
                        return [r, p]
                    })
                Pipeline.report(
                    "probe qmm_then_rec",
                    try Pipeline.median(Self.repeats) {
                        let p = try fused.project(xh, prefill: .mlx).output
                        let r = GLM5NextLinearDecodeKernel.longRecurrence(
                            inputs, activated: depends(input: walked.activated, dependencies: [p]),
                            forget: forget, recurrent: state,
                            gateLowerBound: -5, queryScale: 1 / Foundation.sqrt(128)
                        ).core
                        return [r, p]
                    })
                Pipeline.report(
                    "probe rec_then_oproj",
                    Pipeline.median(Self.repeats) {
                        let r = recur()
                        return [r, layer.output(depends(input: values, dependencies: [r]))]
                    })
                Pipeline.report(
                    "probe two_recurrences",
                    Pipeline.median(Self.repeats) {
                        let r = recur()
                        let s = GLM5NextLinearDecodeKernel.longRecurrence(
                            inputs, activated: depends(input: walked.activated, dependencies: [r]),
                            forget: forget, recurrent: state + 0,
                            gateLowerBound: -5, queryScale: 1 / Foundation.sqrt(128)
                        ).core
                        return [r, s]
                    })
                // Fewer, wider recurrence threadgroups leave cores free for
                // the tensor-unit work beside them (same per-row arithmetic).
                let beta = rows.output
                let betaLayout: [Int32] = [Int32(rows.output.dim(-1)), Int32(fused.columns[5])]
                let reference = GLM5NextKDAPrefillKernel.recurrence(
                    activated: walked.activated, forget: forget, betaLogits: beta,
                    betaLayout: betaLayout, aLog: layer.aLog, dtBias: layer.dtBias, state: state,
                    validity: validity, gateLowerBound: -5, queryScale: 1 / Foundation.sqrt(128),
                    launch: GLM5NextKDAPrefillKernel.launch(
                        tokens: tokens, heads: 32, dimension: 128)!)
                eval(reference.core, reference.state)
                let big = MLXRandom.normal([8192, 8192]).asType(.bfloat16)
                eval(big)
                Pipeline.report("probe copy_alone", Pipeline.median(Self.repeats) { [big * 2] })
                Pipeline.report(
                    "probe rec80_then_copy",
                    Pipeline.median(Self.repeats) {
                        let r = recur()
                        return [r, depends(input: big, dependencies: [r]) * 2]
                    })
                Pipeline.report(
                    "probe rec80_then_conv",
                    Pipeline.median(Self.repeats) {
                        let r = recur()
                        var shifted = inputs
                        shifted.projectionRows = .init(
                            buffer: depends(input: rows.output, dependencies: [r]),
                            betaColumn: fused.columns[5])
                        return [
                            r,
                            GLM5NextLinearDecodeKernel.longConvolution(shifted, convolution: zeros)
                                .activated,
                        ]
                    })
                for repeatRun in 0 ..< 2 {
                    for (name, launch) in [
                        (
                            "percore80",
                            GLM5NextKDAPrefillKernel.launch(
                                tokens: tokens, heads: 32, dimension: 128)!
                        ),
                        ("blocked64", GLM5NextKDAPrefillKernel.Launch.blocked(heads: 32)!),
                    ] {
                        func run2() -> MLXArray {
                            GLM5NextKDAPrefillKernel.recurrence(
                                activated: walked.activated, forget: forget, betaLogits: beta,
                                betaLayout: betaLayout, aLog: layer.aLog, dtBias: layer.dtBias,
                                state: state, validity: validity, gateLowerBound: -5,
                                queryScale: 1 / Foundation.sqrt(128), launch: launch
                            ).core
                        }
                        Pipeline.report(
                            "probe \(name) run\(repeatRun) alone",
                            Pipeline.median(Self.repeats) { [run2()] })
                        Pipeline.report(
                            "probe \(name) run\(repeatRun) then qmm_half",
                            try Pipeline.median(Self.repeats) {
                                let r = run2()
                                return [
                                    r,
                                    try fused.project(
                                        depends(input: xh, dependencies: [r]), prefill: .mlx
                                    )
                                    .output,
                                ]
                            })
                    }
                }
                for groups in [80, 64, 40] {
                    let widest = (32 * 128 + groups - 1) / groups
                    let launch = GLM5NextKDAPrefillKernel.Launch(
                        threadgroups: groups, capacity: widest <= 64 ? 64 : 128, slots: 2, block: 8)
                    func run(_ activated: MLXArray) -> (core: MLXArray, state: MLXArray) {
                        GLM5NextKDAPrefillKernel.recurrence(
                            activated: activated, forget: forget, betaLogits: beta,
                            betaLayout: betaLayout, aLog: layer.aLog, dtBias: layer.dtBias,
                            state: state, validity: validity, gateLowerBound: -5,
                            queryScale: 1 / Foundation.sqrt(128), launch: launch)
                    }
                    let r = run(walked.activated)
                    eval(r.core, r.state)
                    let same =
                        Pipeline.bitwise(r.core, reference.core)
                        && Pipeline.bitwise(r.state, reference.state)
                    Pipeline.report(
                        "probe rec groups=\(groups) bitwise=\(same ? 1 : 0)",
                        Pipeline.median(Self.repeats) {
                            let r = run(walked.activated)
                            return [r.core, r.state]
                        })
                    Pipeline.report(
                        "probe rec groups=\(groups) then qmm_half",
                        try Pipeline.median(Self.repeats) {
                            let r = run(walked.activated)
                            let p = try fused.project(
                                depends(input: xh, dependencies: [r.core]), prefill: .mlx
                            ).output
                            return [r.core, p]
                        })
                    Pipeline.report(
                        "probe rec groups=\(groups) then o_proj",
                        Pipeline.median(Self.repeats) {
                            let r = run(walked.activated)
                            return [
                                r.core,
                                layer.output(depends(input: values, dependencies: [r.core])),
                            ]
                        })
                }
                Pipeline.report(
                    "probe qmm_half_twice",
                    try Pipeline.median(Self.repeats) {
                        let p = try fused.project(xh, prefill: .mlx).output
                        let q = try fused.project(
                            depends(input: x[0..., (tokens / 2)...], dependencies: [p]),
                            prefill: .mlx
                        ).output
                        return [p, q]
                    })
            }
        }

        @Test func dsaStages() throws {
            try Device.withDefaultDevice(.gpu) {
                let tokens = 4096
                let fixture = try GLM5NextDSARankFixture()
                let dsa = try fixture.attention()
                let c = fixture.text
                for history in [0, 12288] {
                    let cache = GLM5NextDSACache()
                    for offset in stride(from: 0, to: history, by: tokens) {
                        let prefix = fixture.input(tokens: tokens, seed: UInt64(100 + offset))
                        _ = try dsa.forward(
                            prefix, cache: cache,
                            validity: MLXArray.ones([1, tokens], dtype: .bool))
                        eval(cache.state)
                    }
                    let x = fixture.input(tokens: tokens, seed: 31)
                    let validity = MLXArray.ones([1, tokens], dtype: .bool)
                    eval(x, validity)
                    let tag = "dsa h=\(history)"
                    for _ in 0 ..< 2 {
                        Pipeline.report(
                            "\(tag) layer",
                            try Pipeline.median(Self.repeats) {
                                let copy = cache.copy() as! GLM5NextDSACache
                                return [try dsa.forward(x, cache: copy, validity: validity)]
                            })
                    }
                    // The layer's stages (projectInputs, then the dense and
                    // sparse rows) rebuilt from its internal members.
                    let fused = try #require(dsa.fusedInputs)
                    Pipeline.report(
                        "\(tag) fused_in_qmm n=\(fused.columns.last ?? 0)",
                        try Pipeline.median(Self.repeats) {
                            [try fused.project(x, prefill: .mlx).output]
                        })
                    let parts = try fused.project(x, prefill: .mlx).members
                    let residual = dsa.qNorm(parts[0])
                    eval(parts + [residual])
                    Pipeline.report(
                        "\(tag) q_a_norm", Pipeline.median(Self.repeats) { [dsa.qNorm(parts[0])] })
                    Pipeline.report(
                        "\(tag) q_b_qmm", Pipeline.median(Self.repeats) { [dsa.qB(residual)] })
                    Pipeline.report(
                        "\(tag) wq_b_qmm",
                        Pipeline.median(Self.repeats) { [dsa.indexer.query(residual)] })
                    let q = dsa.qB(residual).reshaped(1, tokens, c.attentionHeads, c.qkNopeHeadDim)
                    let latentsNew = dsa.kvNorm(parts[1])
                    let index = dsa.indexer.project(
                        x, residual: residual, rawKeys: parts[2], weights: parts[3])
                    eval(q, latentsNew, index.queries, index.keys, index.gates, index.weights)
                    Pipeline.report(
                        "\(tag) index_key_norm+gate",
                        Pipeline.median(Self.repeats) {
                            let i = dsa.indexer.project(
                                x, residual: residual, rawKeys: parts[2], weights: parts[3])
                            return [i.keys, i.gates]
                        })
                    let absorbed = try #require(dsa.absorbed)
                    let qLatents = absorbed.absorbQueries(q)
                    eval(qLatents)
                    Pipeline.report(
                        "\(tag) absorb_q(qmm_n incl copy)",
                        Pipeline.median(Self.repeats) { [absorbed.absorbQueries(q)] })
                    let qT = q.transposed(0, 2, 1, 3)
                    let qContiguous = qT + 0
                    eval(qContiguous)
                    Pipeline.report(
                        "\(tag) absorb_q_input_copy", Pipeline.median(Self.repeats) { [qT + 0] })
                    // Probe: the same product on the tensor units from
                    // dequantized key rows. Bitwise only if every element matches.
                    if history == 0,
                        case .quantized(let key, _, let groupSize, let bits) = absorbed.weights
                    {
                        let rows16 = dequantized(
                            key.words, scales: key.scales, biases: key.biases, groupSize: groupSize,
                            bits: bits)
                        let rows32 = dequantized(
                            key.words, scales: key.scales.asType(.float32),
                            biases: key.biases.asType(.float32), groupSize: groupSize, bits: bits
                        ).asType(.bfloat16)
                        eval(rows16, rows32)
                        for (name, rows) in [("bf16deq", rows16), ("f32deq", rows32)] {
                            let nax = matmul(qT, rows)
                            eval(nax)
                            let a = nax.asType(.float32)
                            let b = qLatents.asType(.float32)
                            let differing = (a .!= b).asType(.float32).mean().item(Float.self)
                            let worst = (abs(a - b).max() / abs(b).max()).item(Float.self)
                            print(
                                String(
                                    format:
                                        "GLM53_PREFILLK %@ absorb_nax_%@ bitwise=%d differing=%.5f maxrel=%.2e",
                                    tag, name, Pipeline.bitwise(nax, qLatents) ? 1 : 0, differing,
                                    worst))
                            Pipeline.report(
                                "\(tag) absorb_nax_\(name) matmul",
                                Pipeline.median(Self.repeats) { [matmul(qT, rows)] })
                        }
                        Pipeline.report(
                            "\(tag) absorb_dequant_rows",
                            Pipeline.median(Self.repeats) {
                                [
                                    dequantized(
                                        key.words, scales: key.scales, biases: key.biases,
                                        groupSize: groupSize, bits: bits)
                                ]
                            })
                    }
                    // History after appending this step, as the layer does.
                    let working = cache.copy() as! GLM5NextDSACache
                    try working.append(
                        latents: latentsNew, keys: index.keys, gates: index.gates,
                        validity: validity)
                    let latents = try #require(working.latents)
                    let keys = try #require(working.indexKeys)
                    let gates = try #require(working.poolGates)
                    let keyValidity = try #require(working.validity)
                    eval(latents, keys, gates, keyValidity)
                    let dense = max(0, min(tokens, c.indexTopK - history))
                    if dense > 0 {
                        let keyCount = min(working.offset, c.indexTopK)
                        Pipeline.report(
                            "\(tag) dense_rows=\(dense)",
                            try Pipeline.median(Self.repeats) {
                                [
                                    try dsa.denseHistoryAttention(
                                        q: q[0..., ..<dense], history: latents[0..., ..<keyCount],
                                        keyValidity: keyValidity[0..., ..<keyCount],
                                        queryValidity: validity[0..., ..<dense],
                                        queryStart: history,
                                        projections: absorbed)
                                ]
                            })
                    }
                    let rest = dense ..< tokens
                    func select() throws -> GLM5NextSparseAttention.Selection {
                        try GLM5NextSparseAttention.selectTokens(
                            indexQueries: index.queries[0..., rest], indexKeys: keys,
                            poolGates: gates,
                            poolPositionBias: dsa.indexer.positionBias,
                            headWeights: index.weights[0..., rest], keyValidity: keyValidity,
                            queryValidity: validity[0..., rest], queryStart: history + dense,
                            tokenBudget: c.indexTopK,
                            execution: dsa.executionPolicy.sparseSelection,
                            nativePrefillScores: true)
                    }
                    let selection = try select()
                    eval(selection.indices)
                    Pipeline.report(
                        "\(tag) select(scores+pool+topk+expand) rows=\(rest.count)",
                        try Pipeline.median(Self.repeats) { [try select().indices] })
                    let restLatents = absorbed.absorbQueries(q[0..., rest])
                    eval(restLatents)
                    let scale = 1 / Foundation.sqrt(Float(c.qkNopeHeadDim))
                    let sparse = try GLM5NextSparsePrefillKernel.attend(
                        queryLatents: restLatents, latents: latents, indices: selection.indices,
                        scale: scale, outputType: .bfloat16)
                    eval(sparse)
                    Pipeline.report(
                        "\(tag) sparse_kernel rows=\(rest.count)",
                        try Pipeline.median(Self.repeats) {
                            [
                                try GLM5NextSparsePrefillKernel.attend(
                                    queryLatents: restLatents, latents: latents,
                                    indices: selection.indices, scale: scale, outputType: .bfloat16)
                            ]
                        })
                    let valuesSparse = absorbed.projectValues(sparse)
                    eval(valuesSparse)
                    Pipeline.report(
                        "\(tag) value_up rows=\(rest.count)",
                        Pipeline.median(Self.repeats) { [absorbed.projectValues(sparse)] })
                    let o = valuesSparse.transposed(0, 2, 1, 3).reshaped(
                        1, rest.count, c.attentionHeads * c.vHeadDim)
                    eval(o)
                    Pipeline.report(
                        "\(tag) o_proj rows=\(rest.count)",
                        Pipeline.median(Self.repeats) { [dsa.output(o)] })
                }
            }
        }
    }
}
