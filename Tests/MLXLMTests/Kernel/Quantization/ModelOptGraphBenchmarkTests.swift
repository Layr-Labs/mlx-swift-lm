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

        /// The config's dimensions; one MoE layer unless `layers` is given, then
        /// the config's first `layers` block types (and `vocabulary` if given).
        private func configuration(layers: Int? = nil, vocabulary: Int? = nil) throws
            -> NemotronHConfiguration
        {
            let path = ProcessInfo.processInfo.environment["MODELOPT_GRAPH_BENCH_CONFIG"]!
            let root = try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
                    as? [String: Any])
            let c = try #require(root["llm_config"] as? [String: Any])
            func int(_ key: String) throws -> Int { try #require(c[key] as? Int) }
            var pattern = "E"
            if let layers {
                let types = try #require(c["layers_block_type"] as? [String])
                let symbols = ["mamba": "M", "moe": "E", "attention": "*"]
                pattern = try types.prefix(layers).map { try #require(symbols[$0]) }.joined()
                // An explicit block pattern (e.g. "MMMMMMMM") overrides the prefix.
                if let explicit = ProcessInfo.processInfo.environment[
                    "MODELOPT_GRAPH_BENCH_PATTERN"]
                {
                    pattern = explicit
                }
            }
            return NemotronHConfiguration(
                vocabSize: try vocabulary ?? int("vocab_size"), hiddenSize: try int("hidden_size"),
                numHiddenLayers: pattern.count, numAttentionHeads: try int("num_attention_heads"),
                numKeyValueHeads: try int("num_key_value_heads"),
                mambaNumHeads: try int("mamba_num_heads"), mambaHeadDim: try int("mamba_head_dim"),
                ssmStateSize: try int("ssm_state_size"), convKernel: try int("conv_kernel"),
                nGroups: try int("n_groups"), intermediateSize: try int("intermediate_size"),
                moeIntermediateSize: try int("moe_intermediate_size"),
                moeSharedExpertIntermediateSize: try int("moe_shared_expert_intermediate_size"),
                nRoutedExperts: try int("n_routed_experts"),
                numExpertsPerTok: try int("num_experts_per_tok"), hybridOverridePattern: pattern,
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
            let tokens = 2048
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
            let activationT = ModelOptActivationKernels.nvfp4Routed(
                rows.reshaped(tokens * k, latent), sources: sources, experts: sortedIDs,
                globalScale: fc1.routeTables.globalScale,
                overflowDivisor: fc1.routeTables.overflowDivisor,
                inverseWeightScale: fc1.routeTables.inverseWeightScale,
                activationDType: .bfloat16)
            eval(activationT.activation)
            func grouped() -> MLXArray {
                ModelOptNVFP4GroupedGEMM.product(
                    activationT.activation, weight: fc1.weight, scales: fc1.blockScales,
                    experts: activationT.safeExperts,
                    expertScale: fc1.routeTables.inverseWeightScale, squaredReLU: false)
            }
            time("fc1 grouped NVFP4 GEMM (scaled, bf16 out)") { grouped() }
            // fc2 shape (K 2688, N 1024) and the shared expert as one dense
            // group (E = 1), against MLX's products for the same shapes.
            let fc2 = try bank(
                experts: c.nRoutedExperts, outputs: latent, inputs: c.moeIntermediateSize)
            let hidden2 = (MLXRandom.normal([tokens * k, c.moeIntermediateSize]) * 0.5)
                .asType(.bfloat16)
            eval(hidden2)
            time("fc2 MLX gather_qmm sorted (tiled) sums") {
                gatherQuantizedMM(
                    hidden2.reshaped(tokens * k, 1, -1), fc2.weight, scales: fc2.blockScales,
                    biases: nil, rhsIndices: activationT.safeExperts, transpose: true,
                    groupSize: 16, bits: 4, mode: .nvfp4, sortedIndices: true)
            }
            time("fc2 grouped NVFP4 GEMM") {
                ModelOptNVFP4GroupedGEMM.product(
                    hidden2, weight: fc2.weight, scales: fc2.blockScales,
                    experts: activationT.safeExperts,
                    expertScale: fc2.routeTables.inverseWeightScale, squaredReLU: false)
            }
            let sharedX = (MLXRandom.normal([tokens, c.hiddenSize]) * 0.5).asType(.bfloat16)
            eval(sharedX)
            let zeros = MLXArray.zeros([tokens], type: UInt32.self)
            time("shared up MLX quantized_matmul (float32 x)") {
                quantizedMM(
                    sharedX.asType(.float32), up.weight, scales: up.weightScale, biases: nil,
                    transpose: true, groupSize: 16, bits: 4, mode: .nvfp4)
            }
            time("shared up grouped NVFP4 GEMM (E = 1)") {
                ModelOptNVFP4GroupedGEMM.product(
                    sharedX, weight: up.weight.reshaped(1, up.outputDimensions, -1),
                    scales: up.weightScale.reshaped(1, up.outputDimensions, -1), experts: zeros,
                    expertScale: MLXArray([Float(1)]), squaredReLU: true)
            }
            let reference = ModelOptActivationKernels.scaledCast(
                product(false), scale: quantized.rowScale, dtype: .bfloat16
            ).asType(.float32)
            let ours = grouped().asType(.float32)
            let refMax = abs(reference).max(axis: -1, keepDims: true)
            print(
                String(
                    format: "BENCH prefill grouped vs rowwise fc1: max abs / row max %.3g",
                    (abs(ours - reference) / maximum(refMax, 1e-30)).max().item(Float.self)))
            time("fc1 graphForward sorted (default)") { fc1(rows, sortedIDs, sortedIndices: true) }
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

        /// Sorted expert IDs of one 2048-token chunk: uniform random, or the
        /// per-expert row counts of a real routing dump (`MODELOPT_GRAPH_BENCH_ROUTING`,
        /// one array of per-expert rows per MoE layer, summed over chunks)
        /// rescaled to the chunk's row count.
        private func routedIDs(rows: Int, experts: Int, counts: [Int]?) -> MLXArray {
            guard let counts else {
                return sorted(MLXRandom.randInt(Int32(0) ..< Int32(experts), [rows]), axis: 0)
            }
            let total = Double(counts.reduce(0, +))
            var scaled = counts.map { Int((Double($0) * Double(rows) / total).rounded()) }
            let heaviest = scaled.indices.max { scaled[$0] < scaled[$1] }!
            scaled[heaviest] += rows - scaled.reduce(0, +)
            var ids: [Int32] = []
            ids.reserveCapacity(rows)
            for (expert, count) in scaled.enumerated() {
                ids.append(contentsOf: repeatElement(Int32(expert), count: count))
            }
            return MLXArray(ids)
        }

        private func routingDump() throws -> [[Int]]? {
            guard let path = ProcessInfo.processInfo.environment["MODELOPT_GRAPH_BENCH_ROUTING"]
            else { return nil }
            return try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
                    as? [[Int]])
        }

        /// Routed expert stages (activation quantization + product + epilogue, as
        /// the profile's "expert fc1+relu2" and "expert fc2") at 2048 tokens under
        /// uniform routing and under real per-layer skew.
        @Test func skewedExpertRouting() throws {
            let c = try configuration()
            let latent = c.moeLatentSize!
            let rows = 2048 * c.numExpertsPerTok
            let fc1 = try bank(
                experts: c.nRoutedExperts, outputs: c.moeIntermediateSize, inputs: latent)
            let fc2 = try bank(
                experts: c.nRoutedExperts, outputs: latent, inputs: c.moeIntermediateSize)
            let x1 = hidden(rows, width: latent).reshaped(rows, 1, latent)
            let x2 = (hidden(rows, width: c.moeIntermediateSize) * 0.5).reshaped(
                rows, 1, c.moeIntermediateSize)
            eval(x2)
            let dump = try routingDump()
            var cases: [(String, [Int]?)] = [("uniform", nil)]
            for layer in [0, 11, 18, 37] where dump.map({ layer < $0.count }) ?? false {
                cases.append(("real layer \(layer)", dump![layer]))
            }
            let flops = 2 * Double(rows) * Double(c.moeIntermediateSize * latent)
            for (label, counts) in cases {
                let ids = routedIDs(rows: rows, experts: c.nRoutedExperts, counts: counts)
                eval(ids)
                var perExpert = [Int](repeating: 0, count: c.nRoutedExperts)
                for id in ids.asArray(Int32.self) { perExpert[Int(id)] += 1 }
                let maxRows = perExpert.max()!
                let empty = perExpert.filter { $0 == 0 }.count
                for (stage, run) in [
                    ("fc1+relu2", { fc1.squaredReLU(x1, ids, sortedIndices: true) }),
                    ("fc2", { fc2(x2, ids, sortedIndices: true) }),
                ] as [(String, () -> MLXArray)] {
                    let stats = measure(runs: 20) { run() }
                    print(
                        String(
                            format:
                                "BENCH routed %@ %@ (max %d rows/expert, %d empty): %.2f ms, %.1f TFLOPS",
                            stage, label, maxRows, empty, stats.median,
                            flops / (stats.median / 1000) / 1e12))
                }
            }
        }

        /// Routed expert stages cycling through distinct expert banks (as a real
        /// forward does, one bank per layer) against reusing one bank.
        @Test func expertBanksAcrossLayers() throws {
            let c = try configuration()
            let latent = c.moeLatentSize!
            let rows = 2048 * c.numExpertsPerTok
            let layers =
                Int(ProcessInfo.processInfo.environment["MODELOPT_GRAPH_BENCH_BANKS"] ?? "") ?? 16
            var fc1: [ModelOptNVFP4SwitchLinear] = []
            var fc2: [ModelOptNVFP4SwitchLinear] = []
            for _ in 0 ..< layers {
                fc1.append(
                    try bank(
                        experts: c.nRoutedExperts, outputs: c.moeIntermediateSize, inputs: latent))
                fc2.append(
                    try bank(
                        experts: c.nRoutedExperts, outputs: latent, inputs: c.moeIntermediateSize))
            }
            let x1 = hidden(rows, width: latent).reshaped(rows, 1, latent)
            let x2 = (hidden(rows, width: c.moeIntermediateSize) * 0.5).reshaped(
                rows, 1, c.moeIntermediateSize)
            eval(x2)
            let dump = try routingDump()
            let ids = routedIDs(rows: rows, experts: c.nRoutedExperts, counts: dump?[11])
            eval(ids)
            for (label, count) in [("one bank", 1), ("\(layers) banks in turn", layers)] {
                var next = 0
                let first = measure(runs: 2 * layers) {
                    defer { next = (next + 1) % count }
                    return fc1[next].squaredReLU(x1, ids, sortedIndices: true)
                }
                next = 0
                let second = measure(runs: 2 * layers) {
                    defer { next = (next + 1) % count }
                    return fc2[next](x2, ids, sortedIndices: true)
                }
                print(
                    String(
                        format:
                            "BENCH routed experts, %@ (%@ routing): fc1+relu2 %.2f ms, fc2 %.2f ms",
                        label, dump == nil ? "uniform" : "real layer 11", first.median,
                        second.median))
            }
        }

        /// Random bf16 values for every parameter outside the `installed` module
        /// paths (which keep their packed producer bytes).
        private func randomDense(_ module: Module, excluding installed: Set<String>) throws {
            var values: [String: MLXArray] = [:]
            for (name, value) in module.parameters().flattened() {
                let path = name.split(separator: ".").dropLast().joined(separator: ".")
                guard !installed.contains(path) else { continue }
                if name.hasSuffix("A_log") || name.hasSuffix("e_score_correction_bias") {
                    values[name] = value
                } else if name.contains("norm") {
                    values[name] = MLXArray.ones(value.shape, dtype: .bfloat16)
                } else {
                    values[name] = (MLXRandom.normal(value.shape) * 0.02).asType(.bfloat16)
                }
                eval(values[name]!)
            }
            try module.update(parameters: .unflattened(values), verify: .noUnusedKeys)
        }

        /// The config's first `MODELOPT_GRAPH_BENCH_CHAIN` (default `layers`)
        /// layers at real dimensions with distinct synthetic weights per layer.
        private func chainModel(defaultLayers: Int = 16) throws -> NemotronHModel {
            let layers =
                Int(ProcessInfo.processInfo.environment["MODELOPT_GRAPH_BENCH_CHAIN"] ?? "")
                ?? defaultLayers
            let c = try configuration(layers: layers, vocabulary: 256)
            let model = NemotronHModel(c)
            // bf16 like the checkpoint; lazy, so replaced weights never materialize.
            try model.update(
                parameters: .unflattened(
                    model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }),
                verify: .all)
            var projections: [String: ModelOptCalibratedLinear] = [:]
            var experts: [String: ModelOptNVFP4SwitchLinear] = [:]
            for (path, module) in model.leafModules().flattened() {
                let name = path.split(separator: ".").last.map(String.init) ?? path
                let mixer = path.split(separator: ".").dropLast().last.map(String.init)
                if let linear = module as? Linear, type(of: linear) == Linear.self {
                    let (outputs, inputs) = linear.shape
                    if ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"].contains(
                        name)
                    {
                        projections[path] = try fp8(outputs, inputs)
                    } else if mixer == "shared_experts" {
                        projections[path] = try nvfp4(outputs, inputs)
                    }
                } else if module is SwitchLinear {
                    experts[path] =
                        name == "fc1"
                        ? try bank(
                            experts: c.nRoutedExperts, outputs: c.moeIntermediateSize,
                            inputs: c.moeLatentSize!)
                        : try bank(
                            experts: c.nRoutedExperts, outputs: c.moeLatentSize!,
                            inputs: c.moeIntermediateSize)
                }
            }
            try model.installModelOptProjections(projections)
            try model.installModelOptExperts(experts)
            try randomDense(model, excluding: Set(projections.keys).union(experts.keys))
            return model
        }

        /// The chain model's 2048-token chunks, each evaluating the whole layer
        /// chain at once: sustained-load time per chunk against the sum of the
        /// same layers measured one at a time, then the same chunks profiled.
        @Test func sustainedLayerChain() throws {
            let model = try chainModel()
            let c = model.configuration
            let modules = model.namedModules().map(\.1)
            let moe = modules.compactMap { $0 as? NemotronHMoE }
            let mamba = modules.compactMap { $0 as? NemotronHMamba2Mixer }
            let attention = modules.compactMap { $0 as? NemotronHAttention }
            let tokens = 2048
            let chunks = 4
            let prompt = MLXRandom.randInt(Int32(0) ..< 256, [1, tokens * chunks])
            eval(prompt)
            // Isolated: one layer of each kind, as prefillLayersByChunk.
            let x = hidden(tokens, width: c.hiddenSize)
            var isolated = 0.0
            for (count, forward) in [
                (moe.count, moe.first.map { layer in { layer(x) } }),
                (
                    mamba.count,
                    mamba.first.map { layer in
                        { layer(x, attentionMask: .none, ssmMask: nil, cache: nil) }
                    }
                ),
                (
                    attention.count,
                    attention.first.map { layer in
                        { layer(x, attentionMask: .causal, ssmMask: nil, cache: nil) }
                    }
                ),
            ] as [(Int, (() -> MLXArray)?)] {
                guard let forward else { continue }
                isolated += Double(count) * measure(runs: 10, warmup: 2) { forward() }.median
            }
            func prefill() throws -> [Double] {
                let cache = try model.newCache(parameters: nil)
                return (0 ..< chunks).map { chunk in
                    let start = DispatchTime.now().uptimeNanoseconds
                    let logits = model(
                        prompt[0..., (chunk * tokens) ..< ((chunk + 1) * tokens)], cache: cache)
                    eval([logits] + cache.flatMap(\.state))
                    return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                }
            }
            for _ in 0 ..< 2 { _ = try prefill() }
            for pass in 0 ..< 3 {
                let times = try prefill()
                print(
                    String(
                        format:
                            "BENCH sustained chain %d layers (%d MoE, %d Mamba, %d attention), pass %d: "
                            + "ms per chunk %@; isolated layer sum %.1f ms per chunk",
                        c.numHiddenLayers, moe.count, mamba.count, attention.count, pass,
                        times.map { String(format: "%.1f", $0) }.joined(separator: " "), isolated))
            }
            let recorder = NemotronSuperProfileRecorder()
            try NemotronSuperPrefillProfile.$recorder.withValue(recorder) { _ = try prefill() }
            print("BENCH sustained chain profile:\n" + recorder.report())
        }

        /// Greedy single-request decode through the chain model (KV and Mamba
        /// caches, one evaluation per token) after a 16-token prefill.
        @Test func decodeLayerChain() throws {
            let model = try chainModel()
            let cache = try model.newCache(parameters: nil)
            let prompt = MLXRandom.randInt(Int32(0) ..< 256, [1, 16])
            eval([model(prompt, cache: cache)] + cache.flatMap(\.state))
            let steps =
                Int(ProcessInfo.processInfo.environment["MODELOPT_GRAPH_BENCH_STEPS"] ?? "") ?? 40
            defer { ModelOptRowExact.fusedActivation = true }
            for fused in [false, true, false, true] {
                model.setSuperFusedKernels(fused)
                ModelOptRowExact.fusedActivation = fused
                var token = MLXArray([Int32(3)]).reshaped(1, 1)
                var times: [Double] = []
                var builds: [Double] = []
                for _ in 0 ..< steps {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let logits = model(token, cache: cache)
                    token = argMax(logits[0..., -1, 0...], axis: -1).reshaped(1, 1)
                    let roots = [token] + cache.flatMap(\.state)
                    builds.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                    eval(roots)
                    times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                }
                let sorted = times.dropFirst(4).sorted()
                let sortedBuilds = builds.dropFirst(4).sorted()
                print(
                    String(
                        format:
                            "BENCH decode chain %d layers (%@), fused kernels %@: median %.3f ms per token, p10 %.3f, graph build %.3f ms",
                        model.configuration.numHiddenLayers,
                        model.configuration.hybridOverridePattern, fused ? "on" : "off",
                        sorted[sorted.count / 2], sorted[sorted.count / 10],
                        sortedBuilds[sortedBuilds.count / 2]))
            }
        }

        /// One-token decode components as dependent chains over 16 distinct weight
        /// sets (each step consumes the previous output, weights stream from
        /// DRAM): per-step time of each component next to its weight bytes.
        @Test func decodeComponentChains() throws {
            let c = try configuration()
            let latent = c.moeLatentSize!
            let layers = 16
            let mamba = c.mambaNumHeads * c.mambaHeadDim
            let projection = mamba + mamba + 2 * c.nGroups * c.ssmStateSize + c.mambaNumHeads
            func chained(
                _ label: String, bytes: Double, _ step: (Int, MLXArray) -> MLXArray, _ x0: MLXArray
            ) {
                let stats = measure(runs: 20, warmup: 3) {
                    var x = x0
                    for i in 0 ..< layers { x = step(i, x) }
                    return x
                }
                let perStep = stats.median / Double(layers)
                print(
                    String(
                        format: "BENCH decode chain component %@: %.4f ms per step, %.0f GB/s",
                        label, perStep, bytes / (perStep / 1000) / 1e9))
            }
            // Mamba in_proj then out_proj (FP8 exact).
            let inProj = try (0 ..< layers).map { _ in try fp8(projection, c.hiddenSize) }
            let outProj = try (0 ..< layers).map { _ in try fp8(c.hiddenSize, mamba) }
            let x = hidden(1, width: c.hiddenSize)
            chained(
                "Mamba in_proj + out_proj (FP8)",
                bytes: Double(projection * c.hiddenSize + c.hiddenSize * mamba),
                { i, x in
                    outProj[i](inProj[i](x)[.ellipsis, ..<mamba])
                }, x)
            chained(
                "Mamba in_proj alone (FP8)", bytes: Double(projection * c.hiddenSize),
                { i, x in
                    inProj[i](x)[.ellipsis, ..<c.hiddenSize]
                }, x)
            // Shared expert (NVFP4 dense) up + relu2 + down.
            let up = try (0 ..< layers).map { _ in
                try nvfp4(c.moeSharedExpertIntermediateSize, c.hiddenSize)
            }
            let down = try (0 ..< layers).map { _ in
                try nvfp4(c.hiddenSize, c.moeSharedExpertIntermediateSize)
            }
            chained(
                "shared expert (NVFP4)",
                bytes: nvfp4Bytes(c.moeSharedExpertIntermediateSize, c.hiddenSize) * 2,
                { i, x in
                    down[i](up[i].squaredReLU(x))
                }, x)
            // Latent projections (bf16 gemv) and the router (bf16 gemv + select).
            let fc1Latent = (0 ..< layers).map { _ in
                (MLXRandom.normal([latent, c.hiddenSize]) * 0.02).asType(.bfloat16)
            }
            let fc2Latent = (0 ..< layers).map { _ in
                (MLXRandom.normal([c.hiddenSize, latent]) * 0.02).asType(.bfloat16)
            }
            let router = (0 ..< layers).map { _ in
                (MLXRandom.normal([c.nRoutedExperts, c.hiddenSize]) * 0.02).asType(.bfloat16)
            }
            eval(fc1Latent + fc2Latent + router)
            chained(
                "latent fc1 + fc2 (bf16 gemv)", bytes: Double(4 * latent * c.hiddenSize),
                { i, x in
                    matmul(matmul(x, fc1Latent[i].transposed()), fc2Latent[i].transposed())
                }, x)
            let bias = MLXArray.zeros([c.nRoutedExperts])
            chained(
                "router gemv + select", bytes: Double(2 * c.nRoutedExperts * c.hiddenSize),
                { i, x in
                    let gates = matmul(x, router[i].transposed())
                    let (_, scores) = NemotronSuperMoEKernels.select(
                        gates: gates, bias: bias, topK: c.numExpertsPerTok, scalingFactor: 5,
                        normalize: true)
                    return x * scores.sum(axis: -1, keepDims: true).asType(x.dtype)
                }, x)
            // Routed experts: fc1 (+relu2), fc2 and the combine, 22 of 512.
            let fc1 = try (0 ..< layers).map { _ in
                try bank(experts: c.nRoutedExperts, outputs: c.moeIntermediateSize, inputs: latent)
            }
            let fc2 = try (0 ..< layers).map { _ in
                try bank(experts: c.nRoutedExperts, outputs: latent, inputs: c.moeIntermediateSize)
            }
            let ids = (0 ..< layers).map { i in
                MLXArray(
                    (0 ..< c.numExpertsPerTok).map { UInt32(($0 * 23 + i * 7) % c.nRoutedExperts) }
                )
                .reshaped(1, 1, c.numExpertsPerTok)
            }
            let scores = MLXArray.ones([1, 1, c.numExpertsPerTok]) / Float(c.numExpertsPerTok)
            let x1 = hidden(1, width: latent)
            let expertBytes =
                Double(c.numExpertsPerTok) * 2 * nvfp4Bytes(c.moeIntermediateSize, latent)
            chained(
                "routed experts fc1 + fc2 + combine (NVFP4, 22 of 512)", bytes: expertBytes,
                { i, x in
                    let input = MLX.expandedDimensions(x, axes: [-2, -3])
                    let h = try! fc1[i].graphForward(
                        input, ids[i], squaredReLU: true, indicesInRange: true
                    ).value
                    let y = try! fc2[i].graphForward(h, ids[i], indicesInRange: true).value
                    return NemotronSuperMoEKernels.combine(
                        MLX.squeezed(y, axis: -2), scores: scores)
                }, x1)
        }

        /// One-row decode products at Super's shapes, 48 independent calls per
        /// evaluation (no host sync between them) cycling through distinct
        /// weight copies (about 1 GB) so weights stream from DRAM as in a
        /// forward: MLX's dense gemv against the row-exact replica (dense and
        /// E4M3 weights), per call and GB/s.
        @Test func rowExactDecodeProducts() throws {
            let c = try configuration()
            let mamba = c.mambaNumHeads * c.mambaHeadDim
            let projection = mamba + mamba + 2 * c.nGroups * c.ssmStateSize + c.mambaNumHeads
            let headDim = c.headDim!
            let shapes: [(String, Int, Int)] = [
                ("Mamba in_proj", projection, c.hiddenSize),
                ("Mamba out_proj", c.hiddenSize, mamba),
                ("q_proj", c.numAttentionHeads * headDim, c.hiddenSize),
                ("k_proj", c.numKeyValueHeads * headDim, c.hiddenSize),
                ("o_proj", c.hiddenSize, c.numAttentionHeads * headDim),
                ("router", c.nRoutedExperts, c.hiddenSize),
                ("fc1 latent", c.moeLatentSize!, c.hiddenSize),
                ("fc2 latent", c.hiddenSize, c.moeLatentSize!),
                ("lm_head", c.vocabSize, c.hiddenSize),
            ]
            let calls = 48
            for (label, outputs, inputs) in shapes {
                let copies = max(1, min(calls, (1 << 30) / (2 * outputs * inputs)))
                let dense = (0 ..< copies).map { _ in
                    (MLXRandom.normal([outputs, inputs]) * 0.02).asType(.bfloat16)
                }
                let bytes = (0 ..< copies).map { _ in fp8Bytes([outputs, inputs]) }
                let scale = MLXArray([Float(1e-3)])
                let xs = (0 ..< calls).map { _ in
                    (MLXRandom.normal([1, inputs]) * 0.5).asType(.bfloat16)
                }
                eval(dense + bytes + xs)
                let splits = inputs >= 16 * outputs
                func time(
                    _ name: String, bytesPerWeight: Double, _ body: (MLXArray, Int) -> MLXArray
                ) {
                    let stats = measure(runs: 10, warmup: 2) {
                        stacked(xs.enumerated().map { body($0.element, $0.offset % copies) })
                    }
                    let perCall = stats.median / Double(calls)
                    print(
                        String(
                            format:
                                "BENCH decode product %@ %dx%d %@: %.4f ms per call, %.0f GB/s (%d copies)",
                            label, outputs, inputs, name, perCall,
                            bytesPerWeight * Double(outputs * inputs) / (perCall / 1000) / 1e9,
                            copies))
                }
                time("MLX dense gemv", bytesPerWeight: 2) { matmul($0, dense[$1].transposed()) }
                time("row-exact dense", bytesPerWeight: 2) {
                    ModelOptRowExact.product(dense[$1], scale: nil, vectors: $0, splitsK: splits)
                }
                time("row-exact E4M3", bytesPerWeight: 1) {
                    ModelOptRowExact.product(bytes[$1], scale: scale, vectors: $0, splitsK: splits)
                }
                guard ProcessInfo.processInfo.environment["MODELOPT_GRAPH_BENCH_GEOMETRY"] == "1"
                else { continue }
                let reference = ModelOptRowExact.product(
                    bytes[0], scale: scale, vectors: xs[0], splitsK: splits)
                for geometry in [(1, 8), (2, 8), (4, 4), (4, 16), (8, 4), (8, 8), (2, 16)] {
                    ModelOptRowExact.geometryOverride = geometry
                    defer { ModelOptRowExact.geometryOverride = nil }
                    let same = modelOptBitwiseEqual(
                        reference,
                        ModelOptRowExact.product(
                            bytes[0], scale: scale, vectors: xs[0], splitsK: splits))
                    time(
                        "row-exact E4M3 TM \(geometry.0) BM \(geometry.1) bitwise \(same)",
                        bytesPerWeight: 1
                    ) {
                        ModelOptRowExact.product(
                            bytes[$1], scale: scale, vectors: $0, splitsK: splits)
                    }
                }
            }
        }

        /// Super's fused SSD prefill scan at real dimensions (2048 tokens, with a
        /// previous state) against `ssmAttn`.
        @Test func ssdScanVariants() throws {
            let c = try configuration()
            let (h, p, g, n, l) = (c.mambaNumHeads, c.mambaHeadDim, c.nGroups, c.ssmStateSize, 2048)
            let x = (MLXRandom.normal([1, l, h, p]) * 0.5).asType(.bfloat16)
            let B = (MLXRandom.normal([1, l, g, n]) * 0.5).asType(.bfloat16)
            let C = (MLXRandom.normal([1, l, g, n]) * 0.5).asType(.bfloat16)
            let dt = (MLXRandom.normal([1, l, h]) - 2).asType(.bfloat16)
            let aLog = MLXRandom.uniform(0 ..< 1, [h])
            let D = MLXRandom.normal([h]).asType(.bfloat16)
            let dtBias = (MLXRandom.normal([h]) * 0.1).asType(.bfloat16)
            let state = MLXRandom.normal([1, h, p, n]) * 0.1
            eval(x, B, C, dt, aLog, D, dtBias, state)
            func run() -> MLXArray {
                let (y, next) = NemotronSuperSSD.scan(
                    x: x, ALog: aLog, B: B, C: C, D: D, dt: dt, dtBias: dtBias, state: state,
                    timeStepLimit: (0, .infinity))
                return y.asType(.float32).sum() + next.sum()
            }
            defer { NemotronSuperSSD.batchedWindows = true }
            for batched in [false, true] {
                NemotronSuperSSD.batchedWindows = batched
                let fused = measure(runs: 10, warmup: 2) { run() }
                print(
                    String(
                        format: "BENCH SSD scan 2048 tokens, fused windows, batched %@: %.2f ms",
                        batched ? "on" : "off", fused.median))
            }
            let reference = measure(runs: 5, warmup: 1) {
                let (y, next) = ssmAttn(
                    x: x, ALog: aLog, B: B, C: C, D: D, dt: dt, dtBias: dtBias, state: state,
                    timeStepLimit: (0, .infinity))
                return y.asType(.float32).sum() + next.sum()
            }
            print(String(format: "BENCH SSD scan 2048 tokens, ssmAttn: %.2f ms", reference.median))
        }

        /// The grouped NVFP4 GEMM (fc1 shape, 2048 tokens) per tile shape under
        /// uniform and real routing (`MODELOPT_GRAPH_BENCH_ROUTING`), with the
        /// largest deviation from the default tile shape.
        @Test func groupedGEMMTileShapes() throws {
            let c = try configuration()
            let latent = c.moeLatentSize!
            let rows = 2048 * c.numExpertsPerTok
            let fc1 = try bank(
                experts: c.nRoutedExperts, outputs: c.moeIntermediateSize, inputs: latent)
            let x = (MLXRandom.normal([rows, latent]) * 0.5).asType(.bfloat16)
            eval(x)
            let dump = try routingDump()
            var cases: [(String, [Int]?)] = [("uniform", nil)]
            for layer in [11, 37] where dump.map({ layer < $0.count }) ?? false {
                cases.append(("real layer \(layer)", dump![layer]))
            }
            // Every row on one expert: the kernel's rate without small groups.
            cases.append(("one expert", [1] + Array(repeating: 0, count: c.nRoutedExperts - 1)))
            let flops = 2 * Double(rows) * Double(c.moeIntermediateSize * latent)
            let dense = (MLXRandom.normal([c.moeIntermediateSize, latent]) * 0.02).asType(.bfloat16)
            eval(dense)
            let denseStats = measure(runs: 10, warmup: 2) { matmul(x, dense.transposed()) }
            print(
                String(
                    format: "BENCH dense bf16 GEMM %dx%dx%d: %.2f ms, %.1f TFLOPS", rows,
                    c.moeIntermediateSize, latent, denseStats.median,
                    flops / (denseStats.median / 1000) / 1e12))
            for (label, counts) in cases {
                let ids = routedIDs(rows: rows, experts: c.nRoutedExperts, counts: counts)
                    .asType(.uint32)
                eval(ids)
                func product(_ bm: Int, _ bn: Int, _ bk: Int) -> MLXArray {
                    ModelOptNVFP4GroupedGEMM.product(
                        x, weight: fc1.weight, scales: fc1.blockScales, experts: ids,
                        expertScale: fc1.routeTables.inverseWeightScale, squaredReLU: true,
                        rowsPerTile: bm, outputsPerTile: bn, depthPerTile: bk)
                }
                let reference = product(32, 128, 32).asType(.float32)
                eval(reference)
                for (bm, bn, bk) in [
                    (32, 128, 32), (64, 128, 32), (64, 64, 32), (32, 256, 32), (16, 128, 32),
                    (32, 128, 64), (64, 64, 64),
                ] {
                    let stats = measure(runs: 10, warmup: 2) { product(bm, bn, bk) }
                    let deviation = abs(product(bm, bn, bk).asType(.float32) - reference).max()
                        .item(Float.self)
                    print(
                        String(
                            format:
                                "BENCH grouped GEMM fc1 %@ BM %d BN %d BK %d: %.2f ms, %.1f TFLOPS, max |diff| %.3g",
                            label, bm, bn, bk, stats.median,
                            flops / (stats.median / 1000) / 1e12, deviation))
                }
            }
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

        /// One real-dimension Super layer of each kind with synthetic weights.
        private func superLayers(_ c: NemotronHConfiguration) throws
            -> (moe: NemotronHMoE, mamba: NemotronHMamba2Mixer, attention: NemotronHAttention)
        {
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
            try randomDense(mamba)
            let projection = mamba.intermediateSize + mamba.convDim + mamba.numHeads
            try install(
                mamba,
                [
                    "in_proj": try fp8(projection, c.hiddenSize),
                    "out_proj": try fp8(c.hiddenSize, mamba.intermediateSize),
                ])
            let attention = NemotronHAttention(c)
            try randomDense(attention)
            let headDim = c.headDim!
            try install(
                attention,
                [
                    "q_proj": try fp8(c.numAttentionHeads * headDim, c.hiddenSize),
                    "k_proj": try fp8(c.numKeyValueHeads * headDim, c.hiddenSize),
                    "v_proj": try fp8(c.numKeyValueHeads * headDim, c.hiddenSize),
                    "o_proj": try fp8(c.hiddenSize, c.numAttentionHeads * headDim),
                ])
            return (moe, mamba, attention)
        }

        /// The opt-in per-stage profile (NemotronSuperPrefillProfile) of one
        /// layer of each kind at 2048 tokens, synthetic weights and near-uniform
        /// routing; per forward = per layer x the config's layer counts.
        @Test func profiledPrefillStages() throws {
            let c = try configuration()
            let (moe, mamba, attention) = try superLayers(c)
            let x = hidden(2048, width: c.hiddenSize)
            let forwards: [() -> MLXArray] = [
                { moe(x) }, { mamba(x, attentionMask: .none, ssmMask: nil, cache: nil) },
                { attention(x, attentionMask: .causal, ssmMask: nil, cache: nil) },
            ]
            for forward in forwards { for _ in 0 ..< 2 { eval(forward()) } }
            let recorder = NemotronSuperProfileRecorder()
            NemotronSuperPrefillProfile.$recorder.withValue(recorder) {
                for _ in 0 ..< 5 { for forward in forwards { eval(forward()) } }
            }
            print("BENCH profile 2048 tokens, 5 calls per layer kind:\n" + recorder.report())
        }

        /// Prefill chunks of one layer of each kind on the default (tiled) graph
        /// path, with effective TFLOPS from the layer's matrix products.
        @Test func prefillLayersByChunk() throws {
            let c = try configuration()
            let (moe, mamba, attention) = try superLayers(c)
            let latent = c.moeLatentSize!
            let projection = mamba.intermediateSize + mamba.convDim + mamba.numHeads
            let headDim = c.headDim!
            let h = Double(c.hiddenSize)
            let moeFlops =
                2
                * (Double(c.numExpertsPerTok) * 2 * Double(c.moeIntermediateSize * latent)
                    + 2 * Double(c.moeSharedExpertIntermediateSize) * h + 2 * h * Double(latent)
                    + Double(c.nRoutedExperts) * h)
            let mambaFlops = 2 * (Double(projection) * h + h * Double(mamba.intermediateSize))
            let attentionFlops =
                2
                * (2 * Double(c.numAttentionHeads * headDim) * h
                    + 2 * Double(c.numKeyValueHeads * headDim) * h)
            for tokens in [512, 2048, 4096] {
                let x = hidden(tokens, width: c.hiddenSize)
                for grouped in [false, true] {
                    ModelOptNVFP4GroupedGEMM.isEnabled = grouped
                    let stats = measure(runs: 10, warmup: 2) { moe(x) }
                    print(
                        String(
                            format:
                                "BENCH prefill %d tokens MoE layer, grouped GEMM %@: %.2f ms, %.1f TFLOPS",
                            tokens, grouped ? "on" : "off", stats.median,
                            moeFlops * Double(tokens) / (stats.median / 1000) / 1e12))
                }
                for (name, flops, forward) in [
                    ("MoE", moeFlops, { moe(x) }),
                    (
                        "Mamba", mambaFlops,
                        { mamba(x, attentionMask: .none, ssmMask: nil, cache: nil) }
                    ),
                    (
                        "attention", attentionFlops,
                        {
                            attention(x, attentionMask: .causal, ssmMask: nil, cache: nil)
                        }
                    ),
                ] as [(String, Double, () -> MLXArray)] {
                    let stats = measure(runs: 10, warmup: 2) { forward() }
                    let seconds = stats.median / 1000
                    print(
                        String(
                            format:
                                "BENCH prefill %d tokens %@ layer: %.2f ms, %.0f tok/s, %.1f TFLOPS "
                                + "(matrix products)",
                            tokens, name, stats.median, Double(tokens) / seconds,
                            flops * Double(tokens) / seconds / 1e12))
                }
            }
        }

        /// Peak transient memory of a prefill chunk through real-dimension layers
        /// (one weight set per kind, reused): grows with depth only if MLX keeps
        /// earlier layers' intermediates live.
        @Test func prefillPeakMemory() throws {
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
            try randomDense(mamba)
            try install(
                mamba,
                [
                    "in_proj": try fp8(
                        mamba.intermediateSize + mamba.convDim + mamba.numHeads, c.hiddenSize),
                    "out_proj": try fp8(c.hiddenSize, mamba.intermediateSize),
                ])
            let attention = NemotronHAttention(c)
            try randomDense(attention)
            let headDim = c.headDim!
            try install(
                attention,
                [
                    "q_proj": try fp8(c.numAttentionHeads * headDim, c.hiddenSize),
                    "k_proj": try fp8(c.numKeyValueHeads * headDim, c.hiddenSize),
                    "v_proj": try fp8(c.numKeyValueHeads * headDim, c.hiddenSize),
                    "o_proj": try fp8(c.hiddenSize, c.numAttentionHeads * headDim),
                ])
            for tokens in [512, 2048, 4096, 8192] {
                for groups in [1, 4] {
                    let x = hidden(tokens, width: c.hiddenSize)
                    eval(x)
                    Memory.clearCache()
                    let base = Memory.activeMemory
                    Memory.peakMemory = 0
                    var h = x
                    for group in 0 ..< groups {
                        h = h + mamba(h, attentionMask: .none, ssmMask: nil, cache: nil)
                        h = h + moe(h)
                        if group % 2 == 1 {
                            h = h + attention(h, attentionMask: .causal, ssmMask: nil, cache: nil)
                        }
                    }
                    eval(h)
                    let peak = Memory.peakMemory - base
                    #expect(peak <= NemotronHModel.prefillActivationBoundBytes(tokens: tokens))
                    print(
                        String(
                            format:
                                "BENCH prefill peak %d tokens, %d layer groups: %.2f GB transient "
                                + "(%.0f KiB per token)",
                            tokens, groups, Double(peak) / 1e9, Double(peak) / 1024 / Double(tokens)
                        ))
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
