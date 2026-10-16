import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Wall-clock micro-benchmark of one Super layer of each kind at the dimensions
/// of a real `config.json`, with synthetic weights. Opt-in: set
/// `MODELOPT_GRAPH_BENCH_CONFIG` to the config path. Reports, not assertions.
extension KernelTests {
    @Suite(
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["MODELOPT_GRAPH_BENCH_CONFIG"] != nil))
    struct ModelOptGraphBenchmarkTests {
        private static let prefillTokens = 512
        private static let eagerChunk = 32

        private func configuration() throws -> NemotronHConfiguration {
            let path = ProcessInfo.processInfo.environment["MODELOPT_GRAPH_BENCH_CONFIG"]!
            let root = try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
                    as? [String: Any])
            let c = try #require(root["llm_config"] as? [String: Any])
            func int(_ key: String) throws -> Int { try #require(c[key] as? Int) }
            return NemotronHConfiguration(
                vocabSize: try int("vocab_size"), hiddenSize: try int("hidden_size"),
                numHiddenLayers: 1, numAttentionHeads: try int("num_attention_heads"),
                numKeyValueHeads: try int("num_key_value_heads"),
                mambaNumHeads: try int("mamba_num_heads"), mambaHeadDim: try int("mamba_head_dim"),
                ssmStateSize: try int("ssm_state_size"), convKernel: try int("conv_kernel"),
                nGroups: try int("n_groups"), intermediateSize: try int("intermediate_size"),
                moeIntermediateSize: try int("moe_intermediate_size"),
                moeSharedExpertIntermediateSize: try int("moe_shared_expert_intermediate_size"),
                nRoutedExperts: try int("n_routed_experts"),
                numExpertsPerTok: try int("num_experts_per_tok"), hybridOverridePattern: "E",
                headDim: try int("head_dim"), nSharedExperts: try int("n_shared_experts"),
                routedScalingFactor: 5, moeLatentSize: try int("moe_latent_size"))
        }

        private struct Stats { let median: Double, p10: Double, p90: Double, hostEvals: Int }

        private func measure(runs: Int = 25, warmup: Int = 4, _ body: () -> MLXArray) -> Stats {
            for _ in 0 ..< warmup { eval(body()) }
            var times: [Double] = []
            var hostEvals = 0
            for _ in 0 ..< runs {
                let start = DispatchTime.now().uptimeNanoseconds
                let before = EvalProbe.evalsCompleted
                let output = body()
                hostEvals = EvalProbe.evalsCompleted - before
                eval(output)
                times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            times.sort()
            return Stats(
                median: times[times.count / 2], p10: times[times.count / 10],
                p90: times[times.count * 9 / 10], hostEvals: hostEvals)
        }

        /// Decode as inside a full forward: `depth` layers in one graph, one
        /// evaluation, reported per layer; plus host graph-construction time.
        private func chainedDecode(
            _ label: String, depth: Int = 8, bytes: Double, _ x: MLXArray,
            _ layer: (MLXArray) -> MLXArray
        ) {
            let stats = measure {
                var h = x
                for _ in 0 ..< depth { h = layer(h) }
                return h
            }
            let perLayer = Stats(
                median: stats.median / Double(depth), p10: stats.p10 / Double(depth),
                p90: stats.p90 / Double(depth), hostEvals: stats.hostEvals)
            var builds: [Double] = []
            var evaluations: [Double] = []
            for _ in 0 ..< 20 {
                let start = DispatchTime.now().uptimeNanoseconds
                var h = x
                for _ in 0 ..< depth { h = layer(h) }
                let built = DispatchTime.now().uptimeNanoseconds
                eval(h)
                let done = DispatchTime.now().uptimeNanoseconds
                builds.append(Double(built - start) / 1e6 / Double(depth))
                evaluations.append(Double(done - built) / 1e6 / Double(depth))
            }
            builds.sort()
            evaluations.sort()
            report("\(label) chained x\(depth), per layer", perLayer, bytes: bytes, tokens: 1)
            print(
                String(
                    format: "BENCH %@ per layer: graph build (host) %.3f ms, evaluation %.3f ms",
                    label, builds[10], evaluations[10]))
        }

        private func report(_ label: String, _ stats: Stats, bytes: Double, tokens: Int) {
            let seconds = stats.median / 1000
            print(
                String(
                    format:
                        "BENCH %@: median %.3f ms (p10 %.3f, p90 %.3f), %.1f GB/s, %.1f tok/s, "
                        + "host evals while building %d",
                    label, stats.median, stats.p10, stats.p90, bytes / seconds / 1e9,
                    Double(tokens) / seconds, stats.hostEvals))
        }

        private func randomDense(_ module: Module) throws {
            var values: [String: MLXArray] = [:]
            for (name, value) in module.parameters().flattened() {
                if name.hasSuffix("A_log") || name.hasSuffix("e_score_correction_bias") {
                    values[name] = value
                } else if name.contains("norm") {
                    values[name] = MLXArray.ones(value.shape, dtype: .bfloat16)
                } else {
                    values[name] = (MLXRandom.normal(value.shape) * 0.02).asType(.bfloat16)
                }
            }
            try module.update(parameters: .unflattened(values), verify: .all)
            eval(module.parameters())
        }

        private func fp8Bytes(_ shape: [Int]) -> MLXArray {
            let raw = MLXRandom.randInt(Int32(0) ..< 256, shape).asType(.uint8)
            return which((raw & UInt8(0x7F)) .== UInt8(0x7F), raw - UInt8(1), raw)
        }

        private func fp8(_ outputs: Int, _ inputs: Int) throws -> ModelOptCalibratedLinear {
            try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(
                    e4m3Bytes: fp8Bytes([outputs, inputs]), tensorScale: 1e-3),
                inputScale: 64.0 / 448, weightDType: .bfloat16)
        }

        private func nvfp4Weights(_ outputs: Int, _ inputs: Int) throws -> ModelOptNVFP4Weights {
            try ModelOptNVFP4Weights(
                packedBytes: MLXRandom.randInt(Int32(0) ..< 256, [outputs, inputs / 2]).asType(
                    .uint8),
                e4m3ScaleBytes: MLXRandom.randInt(Int32(0x28) ..< 0x49, [outputs, inputs / 16])
                    .asType(.uint8),
                tensorScale: 1e-3)
        }

        private func nvfp4(_ outputs: Int, _ inputs: Int) throws -> ModelOptCalibratedLinear {
            try ModelOptCalibratedLinear(
                weights: nvfp4Weights(outputs, inputs), inputScale: 64.0 / 2688,
                weightDType: .bfloat16)
        }

        private func bank(experts: Int, outputs: Int, inputs: Int) throws
            -> ModelOptNVFP4SwitchLinear
        {
            let packed = MLXRandom.randInt(Int32(0) ..< 256, [experts, outputs, inputs / 2])
                .asType(.uint8)
            let scales = MLXRandom.randInt(Int32(0x28) ..< 0x49, [experts, outputs, inputs / 16])
                .asType(.uint8)
            eval(packed, scales)
            let matrices = try (0 ..< experts).map {
                try ModelOptNVFP4Weights(
                    packedBytes: packed[$0], e4m3ScaleBytes: scales[$0], tensorScale: 1e-3)
            }
            let layer = try ModelOptNVFP4SwitchLinear(
                experts: matrices, inputScales: Array(repeating: 64.0 / 2688, count: experts),
                weightDType: .bfloat16)
            eval(layer.parameters())
            return layer
        }

        private func install(_ module: Module, _ replacements: [String: Module]) throws {
            try module.update(modules: .unflattened(replacements.map { ($0.key, $0.value) }))
        }

        private func setExecution(_ modules: [Module], _ execution: ModelOptExecution) {
            for module in modules {
                (module as? ModelOptCalibratedLinear)?.execution = execution
                (module as? ModelOptNVFP4SwitchLinear)?.execution = execution
            }
        }

        private func hidden(_ tokens: Int, width: Int) -> MLXArray {
            let x = MLXRandom.normal([1, tokens, width]).asType(.bfloat16)
            eval(x)
            return x
        }

        /// Eager prefill at its admitted 32-token chunk, as the model limit forces.
        private func chunked(_ x: MLXArray, _ body: (MLXArray) -> MLXArray) -> MLXArray {
            let parts = stride(from: 0, to: x.dim(1), by: Self.eagerChunk).map { start in
                let out = body(x[0..., start ..< min(x.dim(1), start + Self.eagerChunk), 0...])
                eval(out)
                return out
            }
            return concatenated(parts, axis: 1)
        }

        private func nvfp4Bytes(_ outputs: Int, _ inputs: Int) -> Double {
            Double(outputs * inputs / 2 + outputs * inputs / 16)
        }

        @Test func moeLayer() throws {
            let c = try configuration()
            let moe = NemotronHMoE(c)
            try randomDense(moe)
            let latent = c.moeLatentSize!
            let fc1 = try bank(
                experts: c.nRoutedExperts, outputs: c.moeIntermediateSize, inputs: latent)
            let fc2 = try bank(
                experts: c.nRoutedExperts, outputs: latent, inputs: c.moeIntermediateSize)
            let up = try nvfp4(c.moeSharedExpertIntermediateSize, c.hiddenSize)
            let down = try nvfp4(c.hiddenSize, c.moeSharedExpertIntermediateSize)
            try install(
                moe,
                [
                    "switch_mlp.fc1": fc1, "switch_mlp.fc2": fc2,
                    "shared_experts.up_proj": up, "shared_experts.down_proj": down,
                ])
            let modules: [Module] = [fc1, fc2, up, down]
            let expertBytes = nvfp4Bytes(c.moeIntermediateSize, latent) * 2
            let fixed =
                nvfp4Bytes(c.moeSharedExpertIntermediateSize, c.hiddenSize) * 2
                + Double(2 * c.hiddenSize * latent * 2 + c.nRoutedExperts * c.hiddenSize * 2)
            print(
                "BENCH config: hidden \(c.hiddenSize), experts \(c.nRoutedExperts), top-k "
                    + "\(c.numExpertsPerTok), intermediate \(c.moeIntermediateSize), latent \(latent), "
                    + "shared \(c.moeSharedExpertIntermediateSize)")
            for tokens in [1, Self.prefillTokens] {
                let x = hidden(tokens, width: c.hiddenSize)
                let (indices, _) = moe.gate(x)
                let touched = Set(indices.asArray(Int32.self)).count
                let bytes = fixed + Double(touched) * expertBytes
                let phase = tokens == 1 ? "decode" : "prefill \(tokens)"
                print("BENCH MoE \(phase): \(touched) distinct experts, \(bytes / 1e9) GB weights")
                setExecution(modules, .eagerReference)
                let eager =
                    tokens == 1
                    ? measure { moe(x) }
                    : measure(runs: 20, warmup: 2) { chunked(x) { moe($0) } }
                report("MoE \(phase) eager", eager, bytes: bytes, tokens: tokens)
                setExecution(modules, .graph)
                for accumulation in [ModelOptAccumulation.rowwise, .tiled] {
                    for module in modules {
                        (module as? ModelOptCalibratedLinear)?.accumulation = accumulation
                        (module as? ModelOptNVFP4SwitchLinear)?.accumulation = accumulation
                    }
                    report(
                        "MoE \(phase) graph \(accumulation)", measure { moe(x) }, bytes: bytes,
                        tokens: tokens)
                    if tokens == 1 {
                        chainedDecode("MoE decode graph", bytes: bytes, x) { moe($0) }
                        break
                    }
                }
            }
        }

        /// Decode time of one component: 8 independent copies in one graph.
        private func component(_ label: String, _ body: () -> [MLXArray]) {
            let copies = 8
            let stats = measure {
                concatenated(
                    (0 ..< copies).flatMap { _ in body() }.map { $0.flattened().asType(.float32) })
            }
            print(
                String(
                    format: "BENCH decode component %@: %.3f ms per call", label,
                    stats.median / Double(copies)))
        }

        @Test func decodeBreakdown() throws {
            let c = try configuration()
            let moe = NemotronHMoE(c)
            try randomDense(moe)
            let latent = c.moeLatentSize!
            let fc1 = try bank(
                experts: c.nRoutedExperts, outputs: c.moeIntermediateSize, inputs: latent)
            let fc2 = try bank(
                experts: c.nRoutedExperts, outputs: latent, inputs: c.moeIntermediateSize)
            let up = try nvfp4(c.moeSharedExpertIntermediateSize, c.hiddenSize)
            let down = try nvfp4(c.hiddenSize, c.moeSharedExpertIntermediateSize)
            try install(
                moe,
                [
                    "switch_mlp.fc1": fc1, "switch_mlp.fc2": fc2,
                    "shared_experts.up_proj": up, "shared_experts.down_proj": down,
                ])
            let x = hidden(1, width: c.hiddenSize)
            let (indices, scores) = moe.gate(x)
            let routed = moe.fc1LatentProj!(x)
            eval(indices, scores, routed)
            component("MoE router (gate matmul, sigmoid, top-k, normalize)") {
                let (i, s) = moe.gate(x)
                return [i.asType(.float32), s]
            }
            component("MoE fc1 latent BF16 4096->1024") { [moe.fc1LatentProj!(x)] }
            component("MoE routed experts fc1+relu2+fc2 (22 of 512)") {
                [moe.switchMLP(routed, indices)]
            }
            let expertOut = moe.switchMLP(routed, indices)
            eval(expertOut)
            component("MoE weighted sum + fc2 latent BF16 1024->4096") {
                [moe.fc2LatentProj!((expertOut * scores[.ellipsis, .newAxis]).sum(axis: -2))]
            }
            component("MoE shared expert NVFP4 4096->5376->4096") { [moe.sharedExperts!(x)] }
            component("MoE whole layer") { [moe(x)] }
            component("NVFP4 expert fc1 alone (22 x 2688x1024)") {
                [fc1(expandedDimensions(routed, axes: [-2, -3]), indices)]
            }

            let mamba = NemotronHMamba2Mixer(c)
            try randomDense(mamba)
            let inProj = try fp8(
                mamba.intermediateSize + mamba.convDim + mamba.numHeads, c.hiddenSize)
            let outProj = try fp8(c.hiddenSize, mamba.intermediateSize)
            try install(mamba, ["in_proj": inProj, "out_proj": outProj])
            let mixed = hidden(1, width: mamba.intermediateSize)
            component("Mamba in_proj FP8 4096->18560") { [inProj(x)] }
            component("Mamba out_proj FP8 8192->4096") { [outProj(mixed)] }
            component("Mamba whole layer (no cache, 1 token)") {
                [mamba(x, attentionMask: .none, ssmMask: nil, cache: nil)]
            }
            let cache = MambaCache()
            _ = mamba(
                hidden(8, width: c.hiddenSize), attentionMask: .none, ssmMask: nil, cache: cache)
            eval(cache.state)
            component("Mamba whole layer (cached step)") {
                let state = cache.state
                let output = mamba(x, attentionMask: .none, ssmMask: nil, cache: cache)
                cache.state = state
                return [output]
            }
        }

        /// One prefill chunk of a MoE layer's expert and shared products.
        @Test func prefillBreakdown() throws {
            let c = try configuration()
            let latent = c.moeLatentSize!
            let tokens = Self.prefillTokens
            let k = c.numExpertsPerTok
            let fc1 = try bank(
                experts: c.nRoutedExperts, outputs: c.moeIntermediateSize, inputs: latent)
            let up = try nvfp4(c.moeSharedExpertIntermediateSize, c.hiddenSize)
            let sortedIDs = sorted(
                MLXRandom.randInt(Int32(0) ..< Int32(c.nRoutedExperts), [tokens * k]), axis: 0)
            let rows = hidden(tokens * k, width: latent).reshaped(tokens * k, 1, latent)
            let sources = MLXArray((0 ..< tokens * k).map { Int32($0) })
            func quantize() -> (activation: MLXArray, safeExperts: MLXArray, rowScale: MLXArray) {
                ModelOptActivationKernels.nvfp4Routed(
                    rows.reshaped(tokens * k, latent), sources: sources, experts: sortedIDs,
                    globalScale: fc1.routeTables.globalScale,
                    overflowDivisor: fc1.routeTables.overflowDivisor,
                    inverseWeightScale: fc1.routeTables.inverseWeightScale)
            }
            let quantized = quantize()
            eval(quantized.activation, quantized.safeExperts, quantized.rowScale)
            func product(_ sorted: Bool) -> MLXArray {
                gatherQuantizedMM(
                    quantized.activation.reshaped(tokens * k, 1, latent), fc1.weight,
                    scales: fc1.blockScales, biases: nil, rhsIndices: quantized.safeExperts,
                    transpose: true, groupSize: 16, bits: 4, mode: .nvfp4, sortedIndices: sorted
                ).reshaped(tokens * k, -1)
            }
            func time(_ label: String, _ body: () -> MLXArray) {
                let stats = measure(runs: 20) { body() }
                print(String(format: "BENCH prefill component %@: %.3f ms", label, stats.median))
            }
            time("fc1 activation kernel (\(tokens * k) rows)") { quantize().activation }
            time("fc1 rowwise sums (MLX gather_qmv)") { product(false) }
            time("fc1 tiled sums (MLX gather_qmm, sorted)") { product(true) }
            time("fc1 graphForward sorted rowwise") { fc1(rows, sortedIDs, sortedIndices: true) }
            let rowwise = product(false)
            let tiled = product(true)
            let difference = abs(tiled - rowwise)
            let magnitude = abs(rowwise).max(axis: -1, keepDims: true)
            print(
                String(
                    format:
                        "BENCH prefill tiled vs rowwise fc1 sums: max abs %.3g, max abs / row max "
                        + "%.3g, elements not bitwise equal %.4f",
                    difference.max().item(Float.self),
                    (difference / maximum(magnitude, 1e-30)).max().item(Float.self),
                    (tiled .!= rowwise).asType(.float32).mean().item(Float.self)))
            let shared = hidden(tokens, width: c.hiddenSize)
            time("shared up rowwise (\(tokens) rows)") { up(shared) }
            up.accumulation = .tiled
            time("shared up tiled") { up(shared) }
        }

        /// Batched decode at B rows: B one-row calls (each weight read B times)
        /// against one row-exact call, per component and for a whole MoE layer.
        @Test func batchedRowExactDecode() throws {
            let c = try configuration()
            let moe = NemotronHMoE(c)
            try randomDense(moe)
            let latent = c.moeLatentSize!
            try install(
                moe,
                [
                    "switch_mlp.fc1": try bank(
                        experts: c.nRoutedExperts, outputs: c.moeIntermediateSize, inputs: latent),
                    "switch_mlp.fc2": try bank(
                        experts: c.nRoutedExperts, outputs: latent, inputs: c.moeIntermediateSize),
                    "shared_experts.up_proj": try nvfp4(
                        c.moeSharedExpertIntermediateSize, c.hiddenSize),
                    "shared_experts.down_proj": try nvfp4(
                        c.hiddenSize, c.moeSharedExpertIntermediateSize),
                ])
            let mamba = NemotronHMamba2Mixer(c)
            let inProj = try fp8(
                mamba.intermediateSize + mamba.convDim + mamba.numHeads, c.hiddenSize)
            let head = try fp8(c.vocabSize, c.hiddenSize)
            let shared = moe.sharedExperts!
            func perRow(_ x: MLXArray, _ f: (MLXArray) -> MLXArray) -> MLXArray {
                concatenated((0 ..< x.dim(0)).map { f(x[$0 ..< $0 + 1]) }, axis: 0)
            }
            for rows in [1, 4, 8, 16] {
                let x = hidden(rows, width: c.hiddenSize).reshaped(rows, 1, c.hiddenSize)
                let cases: [(String, () -> MLXArray, () -> MLXArray)] = [
                    (
                        "router gate", { perRow(x) { moe.gate($0).1 } },
                        {
                            moe.gate.rowExactBatches = true
                            defer { moe.gate.rowExactBatches = false }
                            return moe.gate(x).1
                        }
                    ),
                    (
                        "fc1 latent BF16", { perRow(x) { moe.fc1LatentProj!($0) } },
                        {
                            nemotronRowExactLinear(x, moe.fc1LatentProj!.weight)
                                ?? moe.fc1LatentProj!(x)
                        }
                    ),
                    (
                        "shared expert NVFP4", { perRow(x) { shared($0) } },
                        {
                            let up = shared.upProj as! ModelOptCalibratedLinear
                            let down = shared.downProj as! ModelOptCalibratedLinear
                            return try! down.rowExactForward(
                                up.rowExactForward(x, squaredReLU: true))
                        }
                    ),
                    (
                        "FP8 in_proj", { perRow(x) { inProj($0) } },
                        { try! inProj.rowExactForward(x) }
                    ),
                    ("FP8 lm_head", { perRow(x) { head($0) } }, { try! head.rowExactForward(x) }),
                    (
                        "MoE layer", { perRow(x) { moe($0) } },
                        {
                            moe.rowExactBatches = true
                            moe.gate.rowExactBatches = true
                            defer {
                                moe.rowExactBatches = false
                                moe.gate.rowExactBatches = false
                            }
                            return moe(x)
                        }
                    ),
                ]
                for (name, before, after) in cases {
                    let one = measure(runs: 20) { before() }
                    let batched = measure(runs: 20) { after() }
                    let same = rows == 1 || arrayEqual(before(), after()).item(Bool.self)
                    print(
                        String(
                            format:
                                "BENCH batched B=%d %@: %d one-row calls %.3f ms, row-exact %.3f ms, "
                                + "bitwise %@",
                            rows, name, rows, one.median, batched.median, same ? "yes" : "NO"))
                }
            }
        }

        @Test func fp8ProjectionsAndHead() throws {
            let c = try configuration()
            let mamba = NemotronHMamba2Mixer(c)
            let inputs = c.hiddenSize
            let outputs = mamba.intermediateSize + mamba.convDim + mamba.numHeads
            let projection = try fp8(outputs, inputs)
            let head = try fp8(c.vocabSize, c.hiddenSize)
            for (name, layer) in [("in_proj", projection), ("lm_head", head)] {
                let bytes = Double(layer.outputDimensions * layer.inputDimensions)
                for tokens in name == "lm_head" ? [1] : [1, Self.prefillTokens] {
                    let x = hidden(tokens, width: inputs)
                    let phase = tokens == 1 ? "decode" : "prefill \(tokens)"
                    let label = "FP8 \(name) [\(layer.outputDimensions)x\(inputs)] \(phase)"
                    layer.execution = .eagerReference
                    let eager =
                        tokens == 1
                        ? measure { layer(x) }
                        : measure(runs: 20, warmup: 2) { chunked(x) { layer($0) } }
                    report("\(label) eager", eager, bytes: bytes, tokens: tokens)
                    layer.execution = .graph
                    report("\(label) graph", measure { layer(x) }, bytes: bytes, tokens: tokens)
                }
            }
        }

        @Test func mambaAndAttentionLayers() throws {
            let c = try configuration()
            let mamba = NemotronHMamba2Mixer(c)
            try randomDense(mamba)
            let inProj = try fp8(
                mamba.intermediateSize + mamba.convDim + mamba.numHeads, c.hiddenSize)
            let outProj = try fp8(c.hiddenSize, mamba.intermediateSize)
            try install(mamba, ["in_proj": inProj, "out_proj": outProj])
            let attention = NemotronHAttention(c)
            try randomDense(attention)
            let headDim = c.headDim!
            let q = try fp8(c.numAttentionHeads * headDim, c.hiddenSize)
            let k = try fp8(c.numKeyValueHeads * headDim, c.hiddenSize)
            let v = try fp8(c.numKeyValueHeads * headDim, c.hiddenSize)
            let o = try fp8(c.hiddenSize, c.numAttentionHeads * headDim)
            try install(attention, ["q_proj": q, "k_proj": k, "v_proj": v, "o_proj": o])
            let cases: [(String, [Module], (MLXArray) -> MLXArray)] = [
                (
                    "Mamba", [inProj, outProj],
                    { mamba($0, attentionMask: .none, ssmMask: nil, cache: nil) }
                ),
                (
                    "attention", [q, k, v, o],
                    {
                        attention(
                            $0, attentionMask: $0.dim(1) > 1 ? .causal : .none, ssmMask: nil,
                            cache: nil)
                    }
                ),
            ]
            for (name, modules, forward) in cases {
                let bytes = modules.reduce(0.0) {
                    let p = $1 as! ModelOptCalibratedLinear
                    return $0 + Double(p.inputDimensions * p.outputDimensions)
                }
                for tokens in [1, Self.prefillTokens] {
                    let x = hidden(tokens, width: c.hiddenSize)
                    let phase = tokens == 1 ? "decode" : "prefill \(tokens)"
                    setExecution(modules, .eagerReference)
                    let eager =
                        tokens == 1
                        ? measure { forward(x) }
                        : measure(runs: 20, warmup: 2) { chunked(x) { forward($0) } }
                    report("\(name) layer \(phase) eager", eager, bytes: bytes, tokens: tokens)
                    setExecution(modules, .graph)
                    report(
                        "\(name) layer \(phase) graph", measure { forward(x) }, bytes: bytes,
                        tokens: tokens)
                    if tokens == 1 {
                        chainedDecode("\(name) layer decode graph", bytes: bytes, x, forward)
                    }
                }
            }
        }
    }
}
