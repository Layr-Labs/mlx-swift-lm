// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXVLM

/// Explicit offline real-checkpoint throughput profiler. Runs only when
/// `GLM53_REAL_PROFILE_DIR` names the prepared smoke directory (with
/// `identity.json` and `prompts.json`); downloads nothing and asserts no
/// performance threshold. One load serves every measurement.
extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextRealProfileTests {
        private struct Identity: Decodable {
            struct File: Decodable {
                let bytes: Int
                let sha256: String
            }
            let repository: String
            let revision: String
            let files: [String: File]
        }

        private struct Prompt: Decodable {
            let name: String
            let ids: [Int32]
        }

        private static var environment: [String: String] { ProcessInfo.processInfo.environment }
        private static var profileDirectory: URL? {
            environment["GLM53_REAL_PROFILE_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        }

        private static func ms(_ start: ContinuousClock.Instant) -> Double {
            let d = start.duration(to: .now)
            return (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) * 1000
        }

        private static func median(_ values: [Double]) -> Double {
            values.isEmpty ? .nan : values.sorted()[values.count / 2]
        }

        private static func list(_ name: String, _ fallback: [Int]) -> [Int] {
            environment[name].map { $0.split(separator: ",").compactMap { Int($0) } } ?? fallback
        }

        private static func policy(_ name: String) -> GLM5NextExecutionPolicy {
            func fast(_ linear: GLM5NextExecutionPolicy.LinearAttention) -> GLM5NextExecutionPolicy {
                GLM5NextExecutionPolicy(
                    sparseAttention: .absorbed,
                    sparseSelection: .init(selection: .parallelThreshold, deviceRanges: true),
                    pagedMetadataReads: .deviceTable, hyperConnection: .nativeRequired,
                    routedExperts: .nativeCombineSortedPrefill, linearAttention: linear)
            }
            switch name {
            case "fast": return fast(.nativeRecurrence)
            case "fastchunked": return fast(.nativeRecurrenceChunkedPrefill)
            case "fastest":
                var policy = fast(.nativeRecurrenceChunkedPrefill)
                policy.sparseAttention = .absorbedDenseHistory
                policy.sparseSelection.queryBlocks = true
                policy.normalization = .fusedKernel
                policy.linearDecodeStep = .fusedEveryStep
                policy.routing = .nativeKernel
                policy.hyperConnection = .nativeNormFused
                policy.projections = .fusedInputs
                policy.stepEvaluation = .earlyAsync
                policy.routedExperts = .fusedDecode
                policy.sparseDecode = .nativeKernels
                policy.sparsePrefill = .nativeKernel
                return policy
            // Fast everything except KDA, so per-call recurrence overrides work.
            case "fastrefkda": return fast(.reference)
            default: return .reference
            }
        }

        /// Pre-evaluated inputs; min over repetitions of build+eval wall time.
        /// Returns (build ms, total ms) of the fastest repetition.
        private static func time(
            _ repetitions: Int = 5, _ body: () throws -> [MLXArray]
        ) rethrows -> (build: Double, total: Double) {
            var best = (build: Double.infinity, total: Double.infinity)
            for _ in 0 ..< repetitions {
                let start = ContinuousClock.now
                let outputs = try body()
                let build = ms(start)
                eval(outputs)
                let total = ms(start)
                if total < best.total { best = (build, total) }
            }
            return best
        }

        private final class Ledger {
            var totals: [String: Double] = [:]
            var builds: [String: Double] = [:]
            var counts: [String: Int] = [:]
            var order: [String] = []
            func add(_ name: String, _ value: (build: Double, total: Double)) {
                if totals[name] == nil { order.append(name) }
                totals[name, default: 0] += value.total
                builds[name, default: 0] += value.build
                counts[name, default: 0] += 1
            }
            func report(_ label: String, syncOverhead: Double) {
                let ranked = order.sorted { totals[$0]! > totals[$1]! }
                for name in ranked {
                    let n = Double(counts[name]!)
                    let corrected = max(0, totals[name]! - n * syncOverhead)
                    print(
                        String(
                            format: "GLM53_PROFILE %@ component=%@ calls=%d total_ms=%.3f "
                                + "minus_sync_ms=%.3f build_ms=%.3f per_call_ms=%.4f",
                            label, name, counts[name]!, totals[name]!, corrected,
                            builds[name]!, totals[name]! / n))
                }
            }
        }

        private static func repeatedTokens(_ prompts: [Prompt], count: Int) -> MLXArray {
            let source = prompts.flatMap(\.ids)
            return MLXArray((0 ..< count).map { source[$0 % source.count] }).reshaped(1, count)
        }

        @Test(.enabled(if: profileDirectory != nil))
        func realArtifactThroughputProfile() throws {
            let root = try #require(Self.profileDirectory)
            let modelURL = URL(
                fileURLWithPath: try #require(Self.environment["GLM53_REAL_MODEL_DIR"]),
                isDirectory: true)
            let identityFile = try JSONDecoder().decode(
                Identity.self, from: Data(contentsOf: root.appendingPathComponent("identity.json")))
            let prompts = try JSONDecoder().decode(
                [Prompt].self, from: Data(contentsOf: root.appendingPathComponent("prompts.json")))
            let identity = GLM5NextArtifactIdentity(
                repository: identityFile.repository, revision: identityFile.revision,
                files: identityFile.files.mapValues { .init(bytes: $0.bytes, sha256: $0.sha256) })
            let policyName = Self.environment["GLM53_REAL_POLICY"] ?? "reference"
            let policy = Self.policy(policyName)
            let contexts = Self.list("GLM53_PROFILE_CONTEXTS", [32, 2048])
            let prefillLengths = Self.list("GLM53_PROFILE_PREFILL", [512, 2048, 4096])
            let decodeSteps = Self.list("GLM53_PROFILE_DECODE_STEPS", [24])[0]
            print("GLM53_PROFILE policy=\(policyName) contexts=\(contexts) prefill=\(prefillLengths)")

            let loadStart = ContinuousClock.now
            let roots = GLM5NextArtifactLoadRoots()
            let artifact = try GLM5NextArtifactLoader(
                directory: modelURL, expectedIdentity: identity,
                storage: GLM5NextWeightInventory.packedStorage(
                    declaredBy: JSONDecoder().decode(
                        GLM5NextConfiguration.self,
                        from: Data(contentsOf: modelURL.appendingPathComponent("config.json")))),
                constructionRoots: roots, executionPolicy: policy)
            let glm = try GLM5NextModel(artifact: artifact, loadMTP: false)
            try roots.evaluate(glm.parameters().flattened().map(\.1))
            print(
                "GLM53_PROFILE load_ms=\(Self.ms(loadStart)) "
                    + "active_gb=\(Double(Memory.activeMemory) / 1e9)")
            let text = glm.textTower
            let c = text.configuration

            try MLX.withError { error in
                // Sync overhead: one tiny eval round trip.
                let scalar = MLXArray(Float(1))
                eval(scalar)
                var syncSamples: [Double] = []
                for _ in 0 ..< 50 {
                    let start = ContinuousClock.now
                    eval(scalar + 1)
                    syncSamples.append(Self.ms(start))
                }
                let sync = syncSamples.min()!
                print("GLM53_PROFILE sync_overhead_ms min=\(sync) median=\(Self.median(syncSamples))")

                // lm_head alone: achieved bandwidth of one large Q4 GEMV.
                let probe = MLXArray.ones([1, 1, c.hiddenSize], dtype: .bfloat16)
                eval(probe)
                let head = Self.time(10) { [text.lmHead(probe)] }
                let headBytes = text.lmHead.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
                print(
                    "GLM53_PROFILE lm_head_ms=\(head.total) bytes=\(headBytes) "
                        + "gbps=\(Double(headBytes) / (head.total / 1000) / 1e9)")
                try error.check()

                for context in contexts {
                    try profileDecode(
                        glm: glm, text: text, prompts: prompts, context: context,
                        steps: decodeSteps, sync: sync, policyName: policyName,
                        contexts: contexts)
                    try error.check()
                }
                if Self.environment["GLM53_PROFILE_COMPONENTS"] == "1" {
                    for tokens in prefillLengths {
                        try componentPrefill(text: text, tokens: tokens)
                        try error.check()
                    }
                }
                for chunk in [512, 2048] {
                    for length in prefillLengths where length >= chunk {
                        try profilePrefill(glm: glm, prompts: prompts, length: length, chunk: chunk)
                        try error.check()
                    }
                }
                // Single-shot 4096 trapped (SIGTRAP) under `fast`; run it last.
                for length in prefillLengths where length > 2048 {
                    try profilePrefill(glm: glm, prompts: prompts, length: length, chunk: length)
                    try error.check()
                }
            }
        }

        /// One instance of each layer component on a fresh `tokens`-row
        /// prefill input, timed alone (build + eval, best of three), and the
        /// per-chunk estimate from the model's layer counts.
        private func componentPrefill(text: GLM5NextTextModel, tokens: Int) throws {
            let c = text.configuration
            let layers = text.model.layers
            let x = (MLXRandom.normal([1, tokens, c.hiddenSize], key: MLXRandom.key(3)) * 0.5)
                .asType(.bfloat16)
            let streams = broadcast(
                x.expandedDimensions(axis: 2), to: [1, tokens, c.hcMult, c.hiddenSize])
            let validity = MLXArray.ones([1, tokens], dtype: .bool)
            eval(x, streams, validity)
            var rows: [(String, Double, Int)] = []
            if let index = layers.firstIndex(where: { $0.attention is GLM5NextKDASublayer }),
                let kda = layers[index].attention as? GLM5NextKDASublayer
            {
                let t = try Self.time(3) {
                    let result = try kda.forward(x, state: nil, validity: validity)
                    return [result.values, result.state.recurrent, result.state.convolution]
                }
                rows.append(("kda_sublayer", t.total, layers.filter { $0.attention is GLM5NextKDASublayer }.count))
            }
            if let dsa = layers.compactMap(\.sparseAttention).first {
                let t = try Self.time(3) {
                    let cache = GLM5NextDSACache()
                    return [try dsa.forward(x, cache: cache, validity: validity)]
                }
                rows.append(("dsa_sublayer", t.total, layers.compactMap(\.sparseAttention).count))
            }
            let moeLayers = layers.filter { $0.mlp is GLM5NextMoE }
            if let moe = moeLayers.first?.mlp as? GLM5NextMoE {
                let policy = text.executionPolicy.routedExperts
                let t = try Self.time(3) { [try moe(x, routedExperts: policy)] }
                rows.append(("moe", t.total, moeLayers.count))
            }
            let denseLayers = layers.filter { $0.mlp is GLM5NextMLP }
            if let dense = denseLayers.first?.mlp as? GLM5NextMLP {
                let t = Self.time(3) { [dense(x)] }
                rows.append(("dense_mlp", t.total, denseLayers.count))
            }
            if let layer = layers.first, text.executionPolicy.hyperConnection == .nativeNormFused {
                let t = try Self.time(3) {
                    let collapsed = try GLM5NextHyperConnectionKernel.collapseNormalized(
                        .streams(streams), fn: layer.attentionFunction, base: layer.attentionBase,
                        scale: layer.attentionScale, rmsNormEpsilon: Float(c.rmsNormEps),
                        epsilon: Float(c.hcEps), sinkhornIterations: c.hcSinkhornIters,
                        normWeight: layer.inputNorm.weight, normEpsilon: layer.inputNorm.epsilon)
                    let expanded = try GLM5NextHyperConnectionKernel.expand(
                        collapsed.normalized, residual: streams, post: collapsed.post,
                        comb: collapsed.comb)
                    return [expanded]
                }
                rows.append(("hc_collapse_norm_expand", t.total, 2 * layers.count))
            }
            let head = Self.time(3) { [text.lmHead(text.model.norm(x[0..., (tokens - 1)...]))] }
            rows.append(("lm_head_last_row", head.total, 1))
            var estimate = 0.0
            for (name, ms, count) in rows {
                estimate += ms * Double(count)
                print(
                    String(
                        format: "GLM53_PROFILE component tokens=%d %@ per_call_ms=%.3f calls=%d chunk_ms=%.1f",
                        tokens, name, ms, count, ms * Double(count)))
            }
            print(
                String(
                    format: "GLM53_PROFILE component tokens=%d estimate_chunk_ms=%.1f tok_s=%.1f",
                    tokens, estimate, Double(tokens) / (estimate / 1000)))
        }

        private func profilePrefill(
            glm: GLM5NextModel, prompts: [Prompt], length: Int, chunk: Int
        ) throws {
            let tokens = Self.repeatedTokens(prompts, count: length)
            eval(tokens)
            for run in 0 ..< 2 {
                let cache = glm.newTextCache()
                Memory.clearCache()
                var build = 0.0
                let start = ContinuousClock.now
                for offset in stride(from: 0, to: length, by: chunk) {
                    let piece = tokens[0..., offset ..< min(length, offset + chunk)]
                    let buildStart = ContinuousClock.now
                    let output = try glm.textTower.forward(tokens: piece, cache: cache)
                    // Serving computes logits for the last position only; the
                    // unused full-vocabulary logits stay unevaluated.
                    let next = argMax(
                        glm.textTower.lmHead(output.hidden[0..., (piece.dim(1) - 1)...]), axis: -1)
                    build += Self.ms(buildStart)
                    eval([next] + cache.layers.flatMap { $0.state })
                }
                let total = Self.ms(start)
                print(
                    String(
                        format: "GLM53_PROFILE prefill run=%d tokens=%d chunk=%d build_ms=%.1f "
                            + "total_ms=%.1f tok_s=%.1f peak_gb=%.2f",
                        run, length, chunk, build, total, Double(length) / (total / 1000),
                        Double(Memory.peakMemory) / 1e9))
            }
        }

        /// One forward over `tokens` that never mutates `cache`, replicating
        /// GLM5NextTextLayer.forward with optional component removal. Removed
        /// parts are replaced by identities of the same shape.
        private func customStep(
            text: GLM5NextTextModel, cache: GLM5NextTextCache, tokens: MLXArray,
            skip: Set<String>
        ) throws -> [MLXArray] {
            let c = text.configuration
            let policy = text.executionPolicy
            let rms = Float(c.rmsNormEps)
            let hcEps = Float(c.hcEps)
            let length = tokens.dim(1)
            let validity = MLXArray.ones([1, length], dtype: .bool)
            func collapse(_ s: MLXArray, _ fn: MLXArray, _ base: MLXArray, _ scale: MLXArray)
                throws -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray)
            {
                if skip.contains("hc") { return (s, s, s[0..., 0..., 0]) }
                return policy.hyperConnection == .nativeRequired
                    ? try GLM5NextHyperConnectionKernel.collapse(
                        s, fn: fn, base: base, scale: scale, rmsNormEpsilon: rms,
                        epsilon: hcEps, sinkhornIterations: c.hcSinkhornIters)
                    : GLM5NextHyperConnection.collapse(
                        s, fn: fn, base: base, scale: scale, rmsNormEpsilon: rms,
                        epsilon: hcEps, sinkhornIterations: c.hcSinkhornIters)
            }
            func expand(_ o: MLXArray, _ r: MLXArray, _ post: MLXArray, _ comb: MLXArray)
                throws -> MLXArray
            {
                if skip.contains("hc") { return r + o.expandedDimensions(axis: 2) }
                return policy.hyperConnection == .nativeRequired
                    ? try GLM5NextHyperConnectionKernel.expand(
                        o, residual: r, post: post, comb: comb)
                    : GLM5NextHyperConnection.expand(o, residual: r, post: post, comb: comb)
            }
            let embeddings = text.embed(tokens)
            var streams = broadcast(
                embeddings.expandedDimensions(axis: 2),
                to: [1, length, c.hcMult, c.hiddenSize])
            var outputs: [MLXArray] = []
            for (layer, state) in zip(text.model.layers, cache.layers) {
                let copy = state.copy()
                let attn = try collapse(
                    streams, layer.attentionFunction, layer.attentionBase, layer.attentionScale)
                let normalized = skip.contains("norm") ? attn.collapsed : layer.inputNorm(attn.collapsed)
                var attentionOutput = normalized
                if let kda = layer.attention as? GLM5NextKDASublayer {
                    if skip.contains("kda") {
                    } else if skip.contains("kda_core") {
                        // Projections and o_proj only; no conv/recurrence/gating.
                        let m = normalized
                        let parts = [kda.q(m), kda.k(m), kda.fB(kda.fA(m)), kda.gB(kda.gA(m))]
                        let b = kda.beta(m)
                        var v = kda.v(m)
                        for p in parts { v = v + p }
                        attentionOutput = kda.output(v) + b.sum(axis: -1, keepDims: true)
                    } else {
                        attentionOutput = try kda.forward(
                            normalized, cache: copy as! GLM5NextKDACache, validity: validity)
                    }
                } else if let dsa = layer.attention as? GLM5NextDSAAttention, !skip.contains("dsa") {
                    attentionOutput = try dsa.forward(
                        normalized, cache: copy as! GLM5NextDSACache, validity: validity)
                }
                let residual = try expand(attentionOutput, streams, attn.post, attn.comb)
                let ffn = try collapse(residual, layer.ffnFunction, layer.ffnBase, layer.ffnScale)
                let input = skip.contains("norm") ? ffn.collapsed : layer.postNorm(ffn.collapsed)
                var mlpOutput = input
                if let dense = layer.mlp as? GLM5NextMLP {
                    if !skip.contains("dense") { mlpOutput = dense(input) }
                } else if let moe = layer.mlp as? GLM5NextMoE {
                    let flat = input.reshaped(-1, c.hiddenSize)
                    let indices: MLXArray
                    let weights: MLXArray
                    if skip.contains("router") {
                        indices = broadcast(
                            MLXArray((0 ..< Int32(c.numExpertsPerTok)).map { $0 * 31 })
                                .reshaped(1, -1), to: [flat.dim(0), c.numExpertsPerTok])
                        weights = MLXArray.ones([flat.dim(0), c.numExpertsPerTok]) * 0.25
                    } else {
                        let routing = moe.gate(input)
                        indices = routing.indices
                        weights = routing.weights
                    }
                    var routed = flat
                    if !skip.contains("routed") {
                        if policy.routedExperts == .reference {
                            routed = moe.experts(flat, indices: indices, weights: weights)
                        } else {
                            routed = try moe.experts(
                                flat, indices: indices, weights: weights,
                                sortedGather: policy.routedExperts == .nativeCombineSortedPrefill
                                    && length > 1)
                        }
                    }
                    mlpOutput = routed.reshaped(input.shape)
                    if !skip.contains("shared") { mlpOutput = mlpOutput + moe.sharedExperts(input) }
                }
                streams = try expand(mlpOutput, residual, ffn.post, ffn.comb)
                outputs += copy.state
            }
            let hidden = text.model.norm(streams.mean(axis: 2))
            let last = hidden[0..., (length - 1)...]
            return [skip.contains("lm_head") ? last : argMax(text.lmHead(last), axis: -1)] + outputs
        }

        private func ablations(
            text: GLM5NextTextModel, cache: GLM5NextTextCache, tokens: MLXArray, label: String,
            repetitions: Int = 7
        ) throws {
            eval(tokens)
            let variants: [Set<String>] = [
                [], ["hc"], ["norm"], ["kda"], ["kda_core"], ["dsa"], ["router"], ["routed"],
                ["shared"], ["dense"], ["lm_head"],
                ["kda", "dsa", "router", "routed", "shared", "dense", "lm_head"],
                ["hc", "norm", "kda", "dsa", "router", "routed", "shared", "dense", "lm_head"],
            ]
            var baseline = 0.0
            for skip in variants {
                var totals: [Double] = []
                var builds: [Double] = []
                for _ in 0 ..< repetitions {
                    let start = ContinuousClock.now
                    let outputs = try customStep(text: text, cache: cache, tokens: tokens, skip: skip)
                    builds.append(Self.ms(start))
                    eval(outputs)
                    totals.append(Self.ms(start))
                }
                let total = Self.median(totals)
                if skip.isEmpty { baseline = total }
                print(
                    String(
                        format: "GLM53_PROFILE ablation %@ skip=%@ step_ms=%.3f build_ms=%.3f "
                            + "saved_ms=%.3f",
                        label, skip.sorted().joined(separator: "+").isEmpty
                            ? "none" : skip.sorted().joined(separator: "+"),
                        total, Self.median(builds), baseline - total))
            }
        }

        /// Amortized micro-benchmarks: 34 independent copies per eval.
        private func kdaMicro(text: GLM5NextTextModel) throws {
            let c = text.configuration
            let heads = c.linearAttention.numHeads
            let dim = c.linearAttention.headDim
            let channels = 3 * heads * dim
            let kernel = c.linearAttention.shortConvKernelSize
            let copies = 34
            func run(_ name: String, _ body: (Int) throws -> [MLXArray]) rethrows {
                var samples: [Double] = []
                for _ in 0 ..< 5 {
                    let start = ContinuousClock.now
                    let outs = try (0 ..< copies).flatMap { try body($0) }
                    eval(outs)
                    samples.append(Self.ms(start))
                }
                print(
                    String(
                        format: "GLM53_PROFILE kda_micro %@ copies=%d total_ms=%.3f per_layer_ms=%.4f",
                        name, copies, samples.min()!, samples.min()! / Double(copies)))
            }
            let history = (0 ..< copies).map { _ in
                MLXRandom.normal([1, kernel, channels]).asType(.float32)
            }
            let weight = MLXRandom.normal([channels, kernel, 1]).asType(.float32)
            let q = (0 ..< copies).map { _ in MLXRandom.normal([1, 1, heads, dim]) * 0.1 }
            let decay = MLXArray.zeros([1, 1, heads, dim]) - 0.01
            let beta = MLXArray.ones([1, 1, heads]) * 0.5
            let state = (0 ..< copies).map { _ in MLXRandom.normal([1, heads, dim, dim]) * 0.01 }
            eval(history + q + state + [weight, decay, beta])
            run("depthwise_conv1d_fp32") {
                [silu(MLX.conv1d(history[$0], weight, groups: channels))]
            }
            run("depthwise_conv_as_mul_sum") {
                [silu((history[$0] * weight.reshaped(channels, kernel).T).sum(axis: 1))]
            }
            try run("native_recurrence_kernel") {
                let r = try GLM5NextLinearKernel.apply(
                    query: q[$0], key: q[$0], value: q[$0], logDecay: decay, beta: beta,
                    state: state[$0])
                return [r.values, r.state]
            }
            run("state_copy_4MB") { [state[$0] + 0] }
        }

        private func profileDecode(
            glm: GLM5NextModel, text: GLM5NextTextModel, prompts: [Prompt], context: Int,
            steps: Int, sync: Double, policyName: String, contexts: [Int]
        ) throws {
            let c = text.configuration
            let cache = glm.newTextCache()
            let tokens = Self.repeatedTokens(prompts, count: context)
            let chunk = 512
            var logits: MLXArray? = nil
            let prefillStart = ContinuousClock.now
            for start in stride(from: 0, to: context, by: chunk) {
                let piece = tokens[0..., start ..< min(context, start + chunk)]
                logits = try text.forward(tokens: piece, cache: cache).logits
                eval([logits!] + cache.layers.flatMap { $0.state })
            }
            print("GLM53_PROFILE ctx=\(context) prefill_ms=\(Self.ms(prefillStart))")
            var next = argMax(logits![0, -1], axis: -1).asType(.int32).reshaped(1, 1)
            eval(next)

            // (1) Synchronous decode: host build vs GPU wait.
            var builds: [Double] = []
            var waits: [Double] = []
            for _ in 0 ..< steps {
                let start = ContinuousClock.now
                let out = try text.forward(tokens: next, cache: cache).logits
                next = argMax(out[0, -1], axis: -1).asType(.int32).reshaped(1, 1)
                let build = Self.ms(start)
                eval([next] + cache.layers.flatMap { $0.state })
                builds.append(build)
                waits.append(Self.ms(start) - build)
            }
            print(
                String(
                    format: "GLM53_PROFILE ctx=%d sync_decode step_ms=%.3f build_ms=%.3f "
                        + "gpu_wait_ms=%.3f tok_s=%.2f",
                    context, Self.median(zip(builds, waits).map { $0 + $1 }), Self.median(builds),
                    Self.median(waits), 1000 / Self.median(zip(builds, waits).map { $0 + $1 })))

            // (2) Pipelined decode: build step n+1 while step n runs.
            do {
                let start = ContinuousClock.now
                var pending = next
                for _ in 0 ..< steps {
                    let out = try text.forward(tokens: pending, cache: cache).logits
                    let following = argMax(out[0, -1], axis: -1).asType(.int32).reshaped(1, 1)
                    asyncEval([following] + cache.layers.flatMap { $0.state })
                    _ = pending.item(Int32.self)
                    pending = following
                }
                _ = pending.item(Int32.self)
                next = pending
                let total = Self.ms(start)
                print(
                    String(
                        format: "GLM53_PROFILE ctx=%d pipelined_decode step_ms=%.3f tok_s=%.2f",
                        context, total / Double(steps), Double(steps) * 1000 / total))
            }

            // (3) In-graph ablations: marginal cost of each component within
            // one unfenced step, the most faithful attribution of step time.
            try ablations(text: text, cache: cache, tokens: next, label: "decode ctx=\(context)")
            if context == contexts.first {
                try ablations(
                    text: text, cache: cache, tokens: Self.repeatedTokens(prompts, count: 512),
                    label: "prefill512 ctx=\(context)", repetitions: 2)
                try kdaMicro(text: text)
            }
            guard Self.environment["GLM53_PROFILE_FENCED"] == "1" else { return }

            // (4) Component profile of one real decode step (eval after each).
            let ledger = Ledger()
            let validity = MLXArray.ones([1, 1], dtype: .bool)
            eval(validity)
            let embed = Self.time { [text.embed(next)] }
            ledger.add("embedding", embed)
            let embeddings = text.embed(next)
            var streams = broadcast(
                embeddings.expandedDimensions(axis: 2), to: [1, 1, c.hcMult, c.hiddenSize])
            eval(streams)
            let rms = Float(c.rmsNormEps)
            let hcEps = Float(c.hcEps)
            let iters = c.hcSinkhornIters
            let compiledCollapse = compile { (a: [MLXArray]) -> [MLXArray] in
                let r = GLM5NextHyperConnection.collapse(
                    a[0], fn: a[1], base: a[2], scale: a[3], rmsNormEpsilon: rms,
                    epsilon: hcEps, sinkhornIterations: iters)
                return [r.post, r.comb, r.collapsed]
            }
            let nativeHC = text.executionPolicy.hyperConnection == .nativeRequired

            for (index, (layer, state)) in zip(text.model.layers, cache.layers).enumerated() {
                eval(streams)
                func collapse(_ s: MLXArray, _ fn: MLXArray, _ base: MLXArray, _ scale: MLXArray)
                    throws -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray)
                {
                    nativeHC
                        ? try GLM5NextHyperConnectionKernel.collapse(
                            s, fn: fn, base: base, scale: scale, rmsNormEpsilon: rms,
                            epsilon: hcEps, sinkhornIterations: iters)
                        : GLM5NextHyperConnection.collapse(
                            s, fn: fn, base: base, scale: scale, rmsNormEpsilon: rms,
                            epsilon: hcEps, sinkhornIterations: iters)
                }
                func expand(_ o: MLXArray, _ r: MLXArray, _ post: MLXArray, _ comb: MLXArray)
                    throws -> MLXArray
                {
                    nativeHC
                        ? try GLM5NextHyperConnectionKernel.expand(
                            o, residual: r, post: post, comb: comb)
                        : GLM5NextHyperConnection.expand(o, residual: r, post: post, comb: comb)
                }
                ledger.add(
                    "hc_collapse(policy)",
                    try Self.time {
                        let r = try collapse(
                            streams, layer.attentionFunction, layer.attentionBase,
                            layer.attentionScale)
                        return [r.post, r.comb, r.collapsed]
                    })
                if index == 0 {
                    // Alternatives, once: reference, compiled reference, native.
                    let ref = Self.time {
                        let r = GLM5NextHyperConnection.collapse(
                            streams, fn: layer.attentionFunction, base: layer.attentionBase,
                            scale: layer.attentionScale, rmsNormEpsilon: rms, epsilon: hcEps,
                            sinkhornIterations: iters)
                        return [r.post, r.comb, r.collapsed]
                    }
                    let compiled = Self.time {
                        compiledCollapse([
                            streams, layer.attentionFunction, layer.attentionBase,
                            layer.attentionScale,
                        ])
                    }
                    let native = try Self.time {
                        let r = try GLM5NextHyperConnectionKernel.collapse(
                            streams, fn: layer.attentionFunction, base: layer.attentionBase,
                            scale: layer.attentionScale, rmsNormEpsilon: rms, epsilon: hcEps,
                            sinkhornIterations: iters)
                        return [r.post, r.comb, r.collapsed]
                    }
                    print(
                        "GLM53_PROFILE ctx=\(context) hc_collapse_alternatives "
                            + "reference=\(ref) compiled_reference=\(compiled) native=\(native) "
                            + "fn_dtype=\(layer.attentionFunction.dtype)")
                }
                let attn = try collapse(
                    streams, layer.attentionFunction, layer.attentionBase, layer.attentionScale)
                eval(attn.post, attn.comb, attn.collapsed)
                ledger.add("input_norm", Self.time { [layer.inputNorm(attn.collapsed)] })
                let normalized = layer.inputNorm(attn.collapsed)
                eval(normalized)

                if let kda = layer.attention as? GLM5NextKDASublayer,
                    let kdaCache = state as? GLM5NextKDACache
                {
                    ledger.add(
                        "kda_input_projections(q,k,v,fa,fb,ga,gb,b)",
                        Self.time {
                            let m = MLX.where(validity.expandedDimensions(axis: -1), normalized, 0)
                            return [
                                kda.q(m), kda.k(m), kda.v(m), kda.fB(kda.fA(m)),
                                kda.gB(kda.gA(m)), kda.beta(m),
                            ]
                        })
                    let headsOut = MLXArray.ones(
                        [1, 1, c.linearAttention.numHeads * c.linearAttention.headDim],
                        dtype: normalized.dtype)
                    eval(headsOut)
                    ledger.add("kda_o_proj", Self.time { [kda.output(headsOut)] })
                    ledger.add(
                        "kda_sublayer_total",
                        try Self.time {
                            let r = try kda.forward(
                                normalized, state: kdaCache.current, validity: validity)
                            return [r.values, r.state.convolution, r.state.recurrent]
                        })
                    if policyName == "fastrefkda", index == 0 {
                        for execution in [
                            GLM5NextLinearAttention.ExecutionPolicy.reference, .nativeRequired,
                        ] {
                            let t = try Self.time {
                                let r = try kda.forward(
                                    normalized, state: kdaCache.current, validity: validity,
                                    recurrenceExecution: execution)
                                return [r.values, r.state.convolution, r.state.recurrent]
                            }
                            print("GLM53_PROFILE ctx=\(context) kda_layer0 \(execution)=\(t)")
                        }
                    }
                } else if let dsa = layer.attention as? GLM5NextDSAAttention,
                    let dsaCache = state as? GLM5NextDSACache
                {
                    ledger.add(
                        "dsa_projections(qa,qb,indexer,kva)",
                        Self.time {
                            let residual = dsa.qNorm(dsa.qA(normalized))
                            let q = dsa.qB(residual)
                            let p = dsa.indexer.project(normalized, residual: residual)
                            return [
                                q, p.queries, p.keys, p.gates, p.weights,
                                dsa.kvNorm(dsa.kvA(normalized)),
                            ]
                        })
                    let residual = dsa.qNorm(dsa.qA(normalized))
                    let projected = dsa.indexer.project(normalized, residual: residual)
                    let working = dsaCache.copy() as! GLM5NextDSACache
                    try working.append(
                        latents: dsa.kvNorm(dsa.kvA(normalized)), keys: projected.keys,
                        gates: projected.gates, validity: validity)
                    eval([projected.queries, projected.weights] + working.state)
                    ledger.add(
                        "dsa_cache_append(concat)",
                        try Self.time {
                            let w = dsaCache.copy() as! GLM5NextDSACache
                            try w.append(
                                latents: working.latents![0..., (working.offset - 1)...],
                                keys: projected.keys, gates: projected.gates, validity: validity)
                            return w.state
                        })
                    ledger.add(
                        "dsa_indexer_selection",
                        try Self.time {
                            let s = try GLM5NextSparseAttention.selectTokens(
                                indexQueries: projected.queries, indexKeys: working.indexKeys!,
                                poolGates: working.poolGates!,
                                poolPositionBias: dsa.indexer.positionBias,
                                headWeights: projected.weights, keyValidity: working.validity!,
                                queryValidity: validity, queryStart: dsaCache.offset,
                                tokenBudget: c.indexTopK,
                                execution: dsa.executionPolicy.sparseSelection)
                            return [s.indices]
                        })
                    let headsOut = MLXArray.ones(
                        [1, 1, c.attentionHeads * c.vHeadDim], dtype: normalized.dtype)
                    eval(headsOut)
                    ledger.add("dsa_o_proj", Self.time { [dsa.output(headsOut)] })
                    ledger.add(
                        "dsa_sublayer_total",
                        try Self.time {
                            let w = dsaCache.copy() as! GLM5NextDSACache
                            return [try dsa.forward(normalized, cache: w, validity: validity)]
                                + w.state
                        })
                }

                // Advance through the real layer to obtain the true residual inputs.
                let attentionOutput: MLXArray
                if let kda = layer.attention as? GLM5NextKDASublayer,
                    let kdaCache = state as? GLM5NextKDACache
                {
                    attentionOutput =
                        try kda.forward(normalized, state: kdaCache.current, validity: validity)
                        .values
                } else {
                    let w = (state as! GLM5NextDSACache).copy() as! GLM5NextDSACache
                    attentionOutput = try (layer.attention as! GLM5NextDSAAttention).forward(
                        normalized, cache: w, validity: validity)
                }
                eval(attentionOutput)
                ledger.add(
                    "hc_expand(policy)",
                    try Self.time { [try expand(attentionOutput, streams, attn.post, attn.comb)] })
                let residual = try expand(attentionOutput, streams, attn.post, attn.comb)
                eval(residual)
                ledger.add(
                    "hc_collapse(policy)",
                    try Self.time {
                        let r = try collapse(
                            residual, layer.ffnFunction, layer.ffnBase, layer.ffnScale)
                        return [r.post, r.comb, r.collapsed]
                    })
                let ffn = try collapse(residual, layer.ffnFunction, layer.ffnBase, layer.ffnScale)
                eval(ffn.post, ffn.comb, ffn.collapsed)
                ledger.add("post_norm", Self.time { [layer.postNorm(ffn.collapsed)] })
                let input = layer.postNorm(ffn.collapsed)
                eval(input)
                let mlpOutput: MLXArray
                if let dense = layer.mlp as? GLM5NextMLP {
                    ledger.add("dense_mlp", Self.time { [dense(input)] })
                    mlpOutput = dense(input)
                } else {
                    let moe = layer.mlp as! GLM5NextMoE
                    ledger.add(
                        "moe_router",
                        Self.time {
                            let r = moe.gate(input)
                            return [r.indices, r.weights]
                        })
                    let routing = moe.gate(input)
                    let flat = input.reshaped(-1, c.hiddenSize)
                    eval(routing.indices, routing.weights, flat)
                    ledger.add(
                        "moe_routed_experts(reference)",
                        Self.time {
                            [moe.experts(flat, indices: routing.indices, weights: routing.weights)]
                        })
                    ledger.add(
                        "moe_routed_experts(nativeCombine)",
                        try Self.time {
                            [
                                try moe.experts(
                                    flat, indices: routing.indices, weights: routing.weights,
                                    sortedGather: false)
                            ]
                        })
                    ledger.add("moe_shared_expert", Self.time { [moe.sharedExperts(input)] })
                    ledger.add(
                        "moe_total(policy)",
                        try Self.time {
                            [try moe(input, routedExperts: layer.executionPolicy.routedExperts)]
                        })
                    if index == 3 {
                        let compiledRouter = compile { (x: [MLXArray]) -> [MLXArray] in
                            let r = moe.gate(x[0])
                            return [r.indices, r.weights]
                        }
                        let routerCompiled = Self.time { compiledRouter([input]) }
                        let routerWeight = moe.gate.weight
                        print(
                            "GLM53_PROFILE ctx=\(context) router_compiled=\(routerCompiled) "
                                + "router_weight_dtype=\(routerWeight.dtype)")
                    }
                    mlpOutput = try moe(input, routedExperts: layer.executionPolicy.routedExperts)
                }
                eval(mlpOutput)
                ledger.add(
                    "hc_expand(policy)",
                    try Self.time { [try expand(mlpOutput, residual, ffn.post, ffn.comb)] })
                // Whole layer, as executed by the model (no intermediate evals).
                ledger.add(
                    "_layer_total_unfenced",
                    try Self.time {
                        let copy = state.copy()
                        return [try layer.forward(streams, cache: copy, validity: validity)]
                            + copy.state
                    })
                streams = try expand(mlpOutput, residual, ffn.post, ffn.comb)
            }
            eval(streams)
            ledger.add(
                "final_norm+lm_head",
                Self.time {
                    let h = text.model.norm(streams.mean(axis: 2))
                    return [argMax(text.lmHead(h)[0, -1], axis: -1)]
                })
            ledger.report("ctx=\(context)", syncOverhead: sync)
        }
    }
}
