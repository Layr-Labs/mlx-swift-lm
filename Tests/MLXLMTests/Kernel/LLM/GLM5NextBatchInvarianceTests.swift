// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Opt-in (`GLM53_ROW_INVARIANCE=1`, GPU, about 1.5 GB of synthetic
    /// weights; `GLM53_ROW_INVARIANCE_FULL=1` adds the artifact's 288 experts
    /// and 154,880-token vocabulary): a tensor-parallel rank-0 stage of the
    /// serving policy at the artifact's local widths (hidden 4096, 32 KDA and
    /// 32 DSA heads, NVFP4 dense MLPs and routed experts, affine Q4/g64
    /// everywhere else), five layers: three KDA with dense MLPs, one DSA and
    /// one KDA with MoE. Asserts that every lane of a multi-request stage step
    /// (`prepareVerificationRows`, `prepareRows`) is bitwise its own request's
    /// step, and the row-count classes of the kernels the grouping relies on.
    @Suite(.serialized)
    struct GLM5NextBatchInvarianceTests {
        static let enabled = ProcessInfo.processInfo.environment["GLM53_ROW_INVARIANCE"] == "1"
        static let full = ProcessInfo.processInfo.environment["GLM53_ROW_INVARIANCE_FULL"] == "1"

        // MARK: - Fixture

        /// The artifact's text geometry with `layers` layers.
        static func configuration(layers: Int = 5, experts: Int, vocabulary: Int) throws
            -> GLM5NextConfiguration
        {
            let base = try GLM5NextInventoryFixture.configuration(layers: layers, q4: true)
            var root =
                try JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as! [String: Any]
            var text = root["text_config"] as! [String: Any]
            let overrides: [String: Any] = [
                "vocab_size": vocabulary, "hidden_size": 4096, "intermediate_size": 12288,
                "moe_intermediate_size": 2048, "num_attention_heads": 64,
                "num_key_value_heads": 64, "qk_head_dim": 256, "qk_nope_head_dim": 256,
                "v_head_dim": 256, "q_lora_rank": 1536, "kv_lora_rank": 512,
                "index_head_dim": 128, "index_n_heads": 32, "index_topk": 2048,
                "n_routed_experts": experts, "num_experts_per_tok": 8,
                "max_position_embeddings": 1_048_576, "dtype": "bfloat16", "pad_token_id": 0,
            ]
            text.merge(overrides) { _, new in new }
            var linear = text["linear_attn_config"] as! [String: Any]
            linear["num_heads"] = 64
            linear["head_dim"] = 128
            text["linear_attn_config"] = linear
            root["text_config"] = text
            var vision = root["vision_config"] as! [String: Any]
            vision["out_hidden_size"] = 4096
            root["vision_config"] = vision
            let declared = try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
            let policy = GLM5NextNVFP4InventoryFixture.policy(declared.textConfig)
            root["quantization"] = policy
            root["quantization_config"] = policy
            return try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        }

        /// Deterministic synthetic payload of an inventory tensor, in a sane
        /// range: NVFP4 E4M3 scales 2^-3...2^-1, affine Q4 weights about zero.
        static func tensor(_ entry: GLM5NextWeightInventory.Entry) -> MLXArray {
            let name = entry.tensor.name
            let shape = entry.tensor.shape
            var hash: UInt64 = 0xcbf2_9ce4_8422_2325
            for byte in name.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100_0000_01b3 }
            let key = MLXRandom.key(hash)
            func uniform(_ range: Range<Float>) -> MLXArray {
                let value = MLXRandom.uniform(range, shape, key: key)
                return entry.tensor.dtype == "F32" ? value : value.asType(.bfloat16)
            }
            switch entry.tensor.dtype {
            case "U32": return MLXRandom.randInt(UInt32(0) ..< UInt32.max, shape, key: key)
            case "U8": return MLXRandom.randInt(UInt8(32) ..< UInt8(56), shape, key: key)
            default: break
            }
            if name.hasSuffix(".weight_scale_2") { return MLXArray(Float(0.02)) }
            if entry.packing != nil && name.hasSuffix(".scales") { return uniform(0.003 ..< 0.006) }
            if entry.packing != nil && name.hasSuffix(".biases") { return uniform(-0.04 ..< -0.028) }
            if name.hasSuffix("norm.weight") || name.hasSuffix("_scale") { return uniform(0.8 ..< 1.2) }
            if name.hasSuffix("A_log") { return uniform(-1 ..< 0) }
            if name.hasSuffix("dt_bias") { return uniform(-0.5 ..< 0) }
            if name.hasSuffix("_base") || name.hasSuffix("correction_bias") || name.hasSuffix(".bias") {
                return uniform(-0.01 ..< 0.01)
            }
            if name.contains("conv1d") { return uniform(-0.3 ..< 0.3) }
            return uniform(-0.02 ..< 0.02)
        }

        /// One stage per layer count for the whole (serialized) suite.
        private final class Stages: @unchecked Sendable {
            let lock = NSLock()
            var byLayers: [Int: GLM5NextTextStage] = [:]
        }
        private static let stages = Stages()

        static func stage(layers: Int = 5) throws -> GLM5NextTextStage {
            stages.lock.lock()
            defer { stages.lock.unlock() }
            if let stage = stages.byLayers[layers] { return stage }
            let stage = try makeStage(layers: layers)
            stages.byLayers[layers] = stage
            return stage
        }

        /// Rank 0 of a tensor-parallel pair; `reduce` stands in for the
        /// cross-rank sum (elementwise, like the sealed sum).
        static func makeStage(layers: Int) throws -> GLM5NextTextStage {
            let declared = try configuration(
                layers: layers, experts: full ? 288 : 16, vocabulary: full ? 154_880 : 8192)
            let inventory = try GLM5NextNVFP4InventoryFixture.inventory(declared)
            let c = declared.textConfig
            let stage = try GLM5NextTextStage(
                plan: .init(
                    configuration: c, layers: 0 ..< c.hiddenLayers,
                    shard: try GLM5NextTensorShard(rank: 0, size: 2)),
                inventory: inventory,
                tensor: { name in tensor(try #require(inventory.entries[name])) },
                materializeExpertStack: { eval($0) }, releaseSourceTensors: { _ in },
                executionPolicy: .serving, materializeFusedProjections: { eval($0) })
            try stage.installTensorParallelReduction { partial in
                (partial.asType(.float32) + partial.asType(.float32)).asType(partial.dtype)
            }
            eval(stage.parameters().flattened().map(\.1))
            print(
                "GLM53_ROW_FIXTURE layers=\(c.hiddenLayers) experts=\(c.nRoutedExperts)"
                    + " vocabulary=\(c.vocabularySize) weights_bytes=\(Memory.activeMemory)")
            return stage
        }

        private static func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        private static func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

        private static func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && a.dtype == b.dtype && bytes(a) == bytes(b)
        }

        private static func tokens(_ values: [Int32]) -> MLXArray { MLXArray(values, [1, values.count]) }

        private static func final(_ output: GLM5NextTextStageOutput) -> GLM5NextTextOutput? {
            if case .final(let value) = output { return value }
            return nil
        }

        /// Every layer's logical state, comparable byte for byte.
        private static func states(_ cache: GLM5NextTextStageCache) -> [MLXArray] {
            cache.layers.flatMap(\.state)
        }

        /// Lanes at different positions: lane i prefilled with its own prompt
        /// in 1,024-token chunks. Lane 3 is past `index_topk` (sparse DSA
        /// queries); lane 2's third window straddles it (dense and sparse
        /// queries in one window); the others are dense.
        private static func prefilled(_ stage: GLM5NextTextStage, lanes: Int) throws
            -> [GLM5NextTextStageCache]
        {
            let lengths = [5, 40, 2_045, 2_061, 17, 129, 9, 777]
            let vocabulary = stage.plan.configuration.vocabularySize
            return try (0 ..< lanes).map { lane in
                let cache = stage.newCache()
                let length = lengths[lane % lengths.count]
                let prompt = (0 ..< length).map { Int32((($0 + 1) * 7919 + lane * 104_729) % vocabulary) }
                for start in stride(from: 0, to: length, by: 1024) {
                    try stage.prepareForward(
                        tokens: tokens(Array(prompt[start ..< min(length, start + 1024)])),
                        cache: cache, readout: .none
                    ).commit(afterEvaluating: fence)
                }
                return cache
            }
        }

        // MARK: - Stage rows

        /// Lanes `0 ..< B` at different positions verify windows of widths 1...3
        /// for three rounds, each lane keeping its own prefix; then one plain
        /// one-token step. Each lane's logits, hidden rows and committed cache
        /// equal its own `prepareVerification` / `prepareForward` bit for bit.
        @Test(
            .enabled(if: enabled),
            arguments: [(1, true), (2, true), (3, true), (4, true), (8, true), (3, false), (8, false)])
        func laneRowsAreEachLanesOwnStepBitwise(lanes: Int, sharedSites: Bool) throws {
            try Device.withDefaultDevice(.gpu) {
                try #require(glm5NextOnDefaultGPUStream())
                let stage = try Self.stage()
                stage.rowsShareSites = sharedSites
                let vocabulary = stage.plan.configuration.vocabularySize
                let base = try Self.prefilled(stage, lanes: lanes)
                let rows = base.map { $0.copy() }
                let solo = base.map { $0.copy() }
                let widths = [3, 3, 2, 1, 3, 2, 3, 3]
                for round in 0 ..< 3 {
                    let windows = (0 ..< lanes).map { lane -> [Int32] in
                        let width = widths[(lane + round) % widths.count]
                        return (0 ..< width).map {
                            Int32((lane * 31 + round * 17 + $0 * 5 + 11) * 9973 % vocabulary)
                        }
                    }
                    let kept = windows.enumerated().map { lane, window in
                        (lane * 7 + round * 3) % window.count + 1
                    }
                    let candidate = try stage.prepareVerificationRows(
                        zip(windows, rows).map { .init(tokens: Self.tokens($0), cache: $1) })
                    let outputs = try candidate.commit(afterEvaluating: Self.fence) { _ in kept }
                    for lane in 0 ..< lanes {
                        let own = try stage.prepareVerification(
                            tokens: Self.tokens(windows[lane]), cache: solo[lane])
                        try own.commit(afterEvaluating: Self.fence) { _ in kept[lane] }
                        let a = try #require(Self.final(outputs[lane]))
                        let b = try #require(Self.final(own.output))
                        let label = "B=\(lanes) shared=\(sharedSites) round \(round) lane \(lane) width \(windows[lane].count)"
                        #expect(Self.identical(a.logits, b.logits), "\(label) logits")
                        #expect(Self.identical(a.hidden, b.hidden), "\(label) hidden")
                        #expect(rows[lane].offset == solo[lane].offset, "\(label) offset")
                        let left = Self.states(rows[lane])
                        let right = Self.states(solo[lane])
                        #expect(left.count == right.count, "\(label) state count")
                        for (index, (x, y)) in zip(left, right).enumerated() {
                            #expect(Self.identical(x, y), "\(label) state \(index)")
                        }
                    }
                }
                // One ordinary one-token step per lane.
                let next = (0 ..< lanes).map { Int32(($0 * 13 + 5) % vocabulary) }
                let step = try stage.prepareRows(
                    zip(next, rows).map { .init(tokens: Self.tokens([$0]), cache: $1) })
                let outputs = try step.commit(afterEvaluating: Self.fence)
                for lane in 0 ..< lanes {
                    let own = try stage.prepareForward(
                        tokens: Self.tokens([next[lane]]), cache: solo[lane], readout: .lastToken)
                    try own.commit(afterEvaluating: Self.fence)
                    let a = try #require(Self.final(outputs[lane]))
                    let b = try #require(Self.final(own.output))
                    #expect(Self.identical(a.logits, b.logits), "B=\(lanes) step lane \(lane) logits")
                    #expect(Self.identical(a.hidden, b.hidden), "B=\(lanes) step lane \(lane) hidden")
                    for (x, y) in zip(Self.states(rows[lane]), Self.states(solo[lane])) {
                        #expect(Self.identical(x, y), "B=\(lanes) step lane \(lane) state")
                    }
                }
            }
        }

        /// Negative control: groups of 4 three-token lanes (12 rows) put the
        /// widest projections past MLX's `qmv_wide` bound, and the harness
        /// sees lanes differ from their own windows.
        @Test(.enabled(if: enabled))
        func oversizedGroupsAreDetected() throws {
            try Device.withDefaultDevice(.gpu) {
                let stage = try Self.stage()
                stage.rowsShareSites = true
                stage.rowGroupRowsOverride = 12
                defer { stage.rowGroupRowsOverride = nil }
                let vocabulary = stage.plan.configuration.vocabularySize
                let base = try Self.prefilled(stage, lanes: 4)
                let rows = base.map { $0.copy() }
                let solo = base.map { $0.copy() }
                let windows = (0 ..< 4).map { lane in
                    (0 ..< 3).map { Int32((lane * 31 + $0 * 5 + 11) * 9973 % vocabulary) }
                }
                let candidate = try stage.prepareVerificationRows(
                    zip(windows, rows).map { .init(tokens: Self.tokens($0), cache: $1) })
                let outputs = try candidate.commit(afterEvaluating: Self.fence)
                var differing = 0
                for lane in 0 ..< 4 {
                    let own = try stage.prepareVerification(tokens: Self.tokens(windows[lane]), cache: solo[lane])
                    try own.commit(afterEvaluating: Self.fence)
                    let a = try #require(Self.final(outputs[lane])), b = try #require(Self.final(own.output))
                    if !Self.identical(a.logits, b.logits) { differing += 1 }
                }
                print("GLM53_ROW_NEGATIVE_CONTROL 12-row groups: \(differing) of 4 lanes differ")
                #expect(differing > 0)
            }
        }

        // MARK: - Speed

        /// Opt-in (`GLM53_ROW_BENCH=1`; `GLM53_ROW_BENCH_LAYERS`, default 5):
        /// wall time of one verification round of B lanes (depth-2 windows)
        /// as ONE rows forward against B solo windows one after another (the
        /// time-sliced seam), each round evaluated before the next. Prints the
        /// median host build, encode and wait per round.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_ROW_BENCH"] == "1"))
        func verificationRowsRoundTime() throws {
            try Device.withDefaultDevice(.gpu) {
                let layers = Int(ProcessInfo.processInfo.environment["GLM53_ROW_BENCH_LAYERS"] ?? "") ?? 5
                let stage = try Self.stage(layers: layers)
                let vocabulary = stage.plan.configuration.vocabularySize
                let base = try Self.prefilled(stage, lanes: 8)
                func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
                func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
                for lanes in [1, 2, 3, 4, 6, 8] {
                    for mode in ["rows", "serial"] {
                        let caches = base.prefix(lanes).map { $0.copy() }
                        var build: [Double] = [], encode: [Double] = [], wait: [Double] = []
                        Memory.clearCache()
                        let active = Memory.activeMemory
                        Memory.peakMemory = 0
                        for round in 0 ..< 14 {
                            let windows = (0 ..< lanes).map { lane in
                                Self.tokens((0 ..< 3).map { Int32((lane * 31 + round * 17 + $0 * 5 + 11) * 9973 % vocabulary) })
                            }
                            var b = 0.0, e = 0.0, w = 0.0
                            if mode == "rows" {
                                let t0 = now()
                                let candidate = try stage.prepareVerificationRows(
                                    zip(windows, caches).map { .init(tokens: $0, cache: $1) })
                                try candidate.commit(afterEvaluating: { roots in
                                    let t1 = now()
                                    asyncEval(roots)
                                    let t2 = now()
                                    eval(roots)
                                    (b, e, w) = (t1 - t0, t2 - t1, now() - t2)
                                }) { _ in (0 ..< lanes).map { ($0 + round) % 3 + 1 } }
                            } else {
                                for lane in 0 ..< lanes {
                                    let t0 = now()
                                    let candidate = try stage.prepareVerification(tokens: windows[lane], cache: caches[lane])
                                    try candidate.commit(afterEvaluating: { roots in
                                        let t1 = now()
                                        asyncEval(roots)
                                        let t2 = now()
                                        eval(roots)
                                        b += t1 - t0; e += t2 - t1; w += now() - t2
                                    }) { _ in (lane + round) % 3 + 1 }
                                }
                            }
                            if round >= 4 { build.append(b); encode.append(e); wait.append(w) }
                        }
                        let total = median(zip(zip(build, encode), wait).map { $0.0 + $0.1 + $1 })
                        let peak = Double(Memory.peakMemory - active) / 1e6
                        let state = Double(caches.map { cache in
                            cache.layers.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
                        }.reduce(0, +)) / Double(lanes) / 1e6
                        print(String(format: "GLM53_ROW_BENCH layers=%d B=%d %@ round=%.2f ms build=%.2f encode=%.2f wait=%.2f per_lane=%.2f peak_over_active=%.0f MB lane_state=%.1f MB",
                            layers, lanes, mode, total, median(build), median(encode), median(wait), total / Double(lanes), peak, state))
                    }
                }
            }
        }

        // MARK: - Kernel classes

        /// Which rows of an M-row call carry the arithmetic of the reference
        /// call: `=` every row bitwise equal, `x` some row differs.
        private static func classes(
            _ label: String, counts: ClosedRange<Int>, reference: Int,
            rows: (Int) throws -> [MLXArray]
        ) throws -> [Int: Bool] {
            let expected = try rows(reference)
            var result: [Int: Bool] = [:]
            var line = "GLM53_ROW_CLASS \(label) ref=\(reference):"
            for count in counts {
                let actual = try rows(count)
                let same = zip(actual, expected).allSatisfy { identical($0, $1) }
                result[count] = same
                line += " \(count)\(same ? "=" : "x")"
            }
            print(line)
            return result
        }

        /// Row r of an M-row call, for each quantized projection of a decode
        /// step, compared with row r of a 2- and a 1-row call; the pool-gate
        /// matmul, the fused MoE, the mHC sites, KDA rows, the embedding
        /// streams and the readout norm likewise. Asserts every class the
        /// row grouping relies on (`GLM5NextTextStage.maximumGroupRows`).
        @Test(.enabled(if: enabled))
        func rowWiseKernelsKeepEachRowsArithmetic() throws {
            try Device.withDefaultDevice(.gpu) {
                try #require(glm5NextOnDefaultGPUStream())
                let stage = try Self.stage()
                let group = GLM5NextTextStage.maximumGroupRows
                print(
                    "GLM53_ROW_CLASS arch=\(GPU.deviceInfo().architecture)"
                        + " vector_limit=\(GLM5NextTextStage.quantizedVectorLimit) group_rows=\(group)")
                let kda = try #require(stage.layers[4].attention as? GLM5NextKDASublayer)
                let dense = try #require(stage.layers[0].mlp as? GLM5NextMLP)
                let dsa = try #require(stage.layers[3].sparseAttention)
                let moe = try #require(stage.layers[4].mlp as? GLM5NextMoE)
                let head = try #require(stage.head)
                var projections: [(String, Linear)] = [
                    ("kda.o", kda.output), ("dsa.q_b", dsa.qB), ("dsa.wq_b", dsa.indexer.query),
                    ("dsa.o", dsa.output), ("dense.gate", dense.gateProj),
                    ("dense.down", dense.downProj), ("shared.down", moe.sharedExperts.downProj),
                    ("head", head),
                ]
                if let fused = kda.fusedInputs?.linear { projections.append(("kda.in", fused)) }
                if let fused = dsa.fusedInputs?.linear { projections.append(("dsa.in", fused)) }
                if let fused = moe.sharedExperts.fusedGateUp?.linear {
                    projections.append(("shared.gateup", fused))
                }
                for (name, linear) in projections {
                    let width = linear.shape.1
                    let x = (MLXRandom.normal([16, width], key: MLXRandom.key(UInt64(width)))
                        * 0.5).asType(.bfloat16)
                    eval(x)
                    let rows: (Int) -> [MLXArray] = { count in
                        let y = linear(x[0 ..< count])
                        eval(y)
                        return (0 ..< min(count, 2)).map { y[$0 ..< $0 + 1] }
                    }
                    // Row r at every count against row r of the 2-row call over
                    // rows r, r+1 (every row, not only the first two).
                    let pairs = (0 ..< 15).map { r -> MLXArray in
                        let y = linear(x[r ..< r + 2])
                        return y[0 ..< 1]
                    }
                    eval(pairs)
                    for count in 2 ... GLM5NextTextStage.sharedSiteGroupRows {
                        let y = linear(x[0 ..< count])
                        eval(y)
                        for r in 0 ..< min(count, 15) {
                            #expect(Self.identical(y[r ..< r + 1], pairs[r]), "\(name) row \(r) of \(count)")
                        }
                    }
                    let wide = try Self.classes(
                        name + " " + "\(width)x\(linear.shape.0)", counts: 1 ... 16, reference: 2,
                        rows: rows)
                    for count in 2 ... GLM5NextTextStage.sharedSiteGroupRows {
                        #expect(wide[count] == true, "\(name) rows \(count)")
                    }
                }

                // The indexer's unquantized pool-gate product (`gemv_wide`).
                let gate = dsa.indexer.poolGate
                let hidden = (MLXRandom.normal([16, 4096], key: MLXRandom.key(5)) * 0.5)
                    .asType(.bfloat16)
                eval(hidden)
                _ = try Self.classes("pool_gate 4096x128", counts: 1 ... 16, reference: 3) { count in
                    let y = matmul(hidden[0 ..< count], gate.T)
                    eval(y)
                    return (0 ..< min(count, 3)).map { y[$0 ..< $0 + 1] }
                }

                // Fused MoE decode: tokens 1...8 (and the unfused path beyond).
                let moeClasses = try Self.classes("moe_fused_decode", counts: 1 ... 12, reference: 3) {
                    count in
                    let y = try moe(
                        hidden[0 ..< count].expandedDimensions(axis: 0),
                        routedExperts: stage.executionPolicy.routedExperts)
                    eval(y)
                    return (0 ..< min(count, 3)).map { y[0..., $0 ..< $0 + 1] }
                }
                for count in 1 ... group { #expect(moeClasses[count] == true, "moe tokens \(count)") }

                // mHC collapse + RMSNorm at an attention site: rows 1...8 one launch.
                let layer = stage.layers[4]
                let c = stage.plan.configuration
                let streams = (MLXRandom.normal([1, 16, 4, 4096], key: MLXRandom.key(6)))
                    .asType(.bfloat16)
                eval(streams)
                let hcClasses = try Self.classes("mhc_collapse_norm", counts: 1 ... 12, reference: 3) {
                    count in
                    let site = try GLM5NextHyperConnectionKernel.collapseNormalized(
                        .streams(streams[0..., 0 ..< count]), fn: layer.attentionFunction,
                        base: layer.attentionBase, scale: layer.attentionScale,
                        rmsNormEpsilon: Float(c.rmsNormEps), epsilon: Float(c.hcEps),
                        sinkhornIterations: c.hcSinkhornIters, normWeight: layer.inputNorm.weight,
                        normEpsilon: layer.inputNorm.epsilon)
                    eval(site.normalized, site.post, site.comb)
                    return (0 ..< min(count, 3)).flatMap { row in
                        [site.normalized, site.post, site.comb].map { $0[0..., row ..< row + 1] }
                    }
                }
                for count in 1 ... group { #expect(hcClasses[count] == true, "mhc rows \(count)") }
                // The same sites on a multi-request workspace: one launch for up to 32 rows.
                let rowsWorkspace = GLM5NextRowSiteWorkspaces.shared.workspace(rows: 32)
                let wideStreams = (MLXRandom.normal([1, 32, 4, 4096], key: MLXRandom.key(16)))
                    .asType(.bfloat16)
                eval(wideStreams)
                let sharedClasses = try Self.classes(
                    "mhc_collapse_norm_rows_workspace", counts: 1 ... 32, reference: 3
                ) { count in
                    let site = try GLM5NextHyperConnectionKernel.collapseNormalized(
                        .streams(wideStreams[0..., 0 ..< count]), fn: layer.ffnFunction,
                        base: layer.ffnBase, scale: layer.ffnScale,
                        rmsNormEpsilon: Float(c.rmsNormEps), epsilon: Float(c.hcEps),
                        sinkhornIterations: c.hcSinkhornIters, normWeight: layer.postNorm.weight,
                        normEpsilon: layer.postNorm.epsilon,
                        workspace: count > 8 ? rowsWorkspace : .shared)
                    eval(site.normalized, site.post, site.comb)
                    return (0 ..< min(count, 3)).flatMap { row in
                        [site.normalized, site.post, site.comb].map { $0[0..., row ..< row + 1] }
                    }
                }
                for count in 1 ... 32 { #expect(sharedClasses[count] == true, "mhc shared rows \(count)") }

                // KDA rows: B stacked states of three-token windows.
                let local = kda.configuration.linearAttention
                let width = local.numHeads * local.headDim
                let x = (MLXRandom.normal([8, 3, 4096], key: MLXRandom.key(7)) * 0.5).asType(
                    .bfloat16)
                let conv = (MLXRandom.normal(
                    [8, 3 * width, local.shortConvKernelSize], key: MLXRandom.key(8)) * 0.5)
                    .asType(.bfloat16)
                let ssm = MLXRandom.normal(
                    [8, local.numHeads, local.headDim, local.headDim], key: MLXRandom.key(9)) * 0.1
                eval(x, conv, ssm)
                // The sublayer: its projections take 3B rows, so a group holds
                // B <= sharedSiteGroupRows / 3 three-token lanes; beyond that the
                // fused input projection leaves `qmv_wide` (printed, not asserted).
                let kdaClasses = try Self.classes("kda_sublayer_window3", counts: 1 ... 8, reference: 1) {
                    batch in
                    let result = try kda.forwardCapturing(
                        x[0 ..< batch], state: .init(convolution: conv[0 ..< batch], recurrent: ssm[0 ..< batch]),
                        validity: MLXArray.ones([batch, 3], dtype: .bool), capture: true)
                    let captured = try #require(result.captured)
                    let arrays = [
                        result.values, result.state.convolution, result.state.recurrent,
                        captured.convolution, captured.recurrent,
                    ]
                    eval(arrays)
                    return arrays.map { $0[0 ..< 1] }
                }
                for batch in 1 ... GLM5NextTextStage.sharedSiteGroupRows / 3 {
                    #expect(kdaClasses[batch] == true, "kda rows \(batch)")
                }
                // The fused recurrence kernel alone: one threadgroup per (row,
                // head), so any number of stacked rows.
                let heads = local.numHeads, dimension = local.headDim
                func random(_ shape: [Int], _ seed: UInt64, _ scale: Float = 0.5) -> MLXArray {
                    (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(.bfloat16)
                }
                let q = random([8, 3, heads, dimension], 10), k = random([8, 3, heads, dimension], 11)
                let v = random([8, 3, heads, dimension], 12), f = random([8, 3, heads, dimension], 13)
                let g = random([8, 3, heads, dimension], 14), beta = random([8, 3, heads], 15)
                eval(q, k, v, f, g, beta)
                let kernelClasses = try Self.classes("kda_decode_kernel_window3", counts: 1 ... 8, reference: 1) {
                    batch in
                    let r = 0 ..< batch
                    let result = try GLM5NextLinearDecodeKernel.apply(
                        .init(
                            query: q[r], key: k[r], value: v[r], gates: .projected(forget: f[r], output: g[r]),
                            betaLogits: beta[r], queryConvolution: kda.qConv.weight,
                            keyConvolution: kda.kConv.weight, valueConvolution: kda.vConv.weight,
                            aLog: kda.aLog, dtBias: kda.dtBias, outputNorm: kda.outputNorm.weight,
                            state: .init(convolution: conv[r], recurrent: ssm[r]),
                            validity: MLXArray.ones([batch, 3], dtype: .bool)),
                        normEpsilon: Float(c.rmsNormEps),
                        gateLowerBound: Float(local.gateLowerBound), capture: true)
                    let arrays = [
                        result.output.values, result.output.state.convolution,
                        result.output.state.recurrent, try #require(result.capturedConvolution),
                        try #require(result.capturedRecurrent),
                    ]
                    eval(arrays)
                    return arrays.map { $0[0 ..< 1] }
                }
                for batch in 1 ... 8 { #expect(kernelClasses[batch] == true, "kda kernel rows \(batch)") }

                // Packed embedding streams and the readout's mean + norm.
                let ids = MLXArray((0 ..< 16).map { Int32($0 * 389 % c.vocabularySize) }, [1, 16])
                let embedClasses = try Self.classes("embedding_streams", counts: 1 ... 8, reference: 3) {
                    count in
                    let s = try #require(stage.embeddedStreams(ids[0..., 0 ..< count]))
                    eval(s)
                    return (0 ..< min(count, 3)).map { s[0..., $0 ..< $0 + 1] }
                }
                for count in 1 ... group { #expect(embedClasses[count] == true, "embedding \(count)") }
                let norm = try #require(stage.norm)
                let readout = try Self.classes("readout_mean_norm", counts: 1 ... 16, reference: 3) {
                    count in
                    let h = norm(streams[0..., 0 ..< count].mean(axis: 2))
                    eval(h)
                    return (0 ..< min(count, 3)).map { h[0..., $0 ..< $0 + 1] }
                }
                for count in 1 ... GLM5NextTextStage.sharedSiteGroupRows {
                    #expect(readout[count] == true, "readout rows \(count)")
                }
                // Batch-major rows of the same kernels (a group is [B,T]).
                let batchReadout = try Self.classes("readout_mean_norm_BxT3", counts: 1 ... 4, reference: 1) {
                    batch in
                    let s = streams[0, 0 ..< 3 * batch].reshaped(batch, 3, 4, 4096)
                    let h = norm(s.mean(axis: 2))
                    eval(h)
                    return [h[0 ..< 1]]
                }
                for batch in 1 ... 2 { #expect(batchReadout[batch] == true, "readout batch \(batch)") }
            }
        }
    }
}
