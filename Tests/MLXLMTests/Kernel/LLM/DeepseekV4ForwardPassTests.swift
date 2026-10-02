import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of `DeepseekV4Model` with a tiny random model.
    ///
    /// The model has 3 layers, one of each attention type: local attention
    /// (compress ratio 0), sparse compressed attention with an indexer
    /// (ratio 4) and compressed attention (ratio 128). Layer 0 routes its
    /// experts by token ID (hash routing). Every layer uses the hyper
    /// connection with 4 streams and its fused Metal kernel. The index top-k
    /// is 2, so the sparse layer takes the gathered top-k path once it has
    /// more than 2 pooled entries. The main sequence has 136 tokens, so the
    /// ratio-128 layer pools one window.
    @Suite
    struct DeepseekV4ForwardPassTests {

        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "vocab_size": vocabularySize,
                "hidden_size": 32,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 3,
                "num_attention_heads": 2,
                "head_dim": 16,
                "q_lora_rank": 16,
                "qk_rope_head_dim": 8,
                "rms_norm_eps": 1e-6,
                "o_groups": 2,
                "o_lora_rank": 8,
                "sliding_window": 8,
                "compress_ratios": [0, 4, 128],
                "compress_rope_theta": 160000,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "scoring_func": "sqrtsoftplus",
                "routed_scaling_factor": 1.5,
                "swiglu_limit": 10.0,
                "num_hash_layers": 1,
                "num_nextn_predict_layers": 0,
                "norm_topk_prob": true,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 3,
                "hc_eps": 1e-6,
                "rope_theta": 10000,
                "max_position_embeddings": 4096,
                "index_n_heads": 2,
                "index_head_dim": 16,
                "index_topk": 2,
            ]
        }

        static func makeModel(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
            -> DeepseekV4Model
        {
            let configuration = try SyntheticModel.configuration(
                DeepseekV4Configuration.self, base, overrides: overrides)
            let model = DeepseekV4Model(configuration)
            SyntheticModel.randomize(model, seed: seed)
            // The hash layer maps each token to 2 different experts. The
            // random pass does not touch this integer table.
            let table = (0 ..< vocabularySize).flatMap { [Int32($0 % 4), Int32(($0 + 1) % 4)] }
            model.update(
                parameters: ModuleParameters.unflattened([
                    "model.layers.0.ffn.gate.tid2eid": MLXArray(table).reshaped(vocabularySize, 2)
                ]))
            eval(model)
            return model
        }

        static func row(_ seed: Int, count: Int) -> [Int] {
            SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
        }

        static func compressedCache(_ model: DeepseekV4Model) -> [KVCache] {
            model.makeCache(parameters: GenerateParameters())
        }

        /// Runs `rows` in `chunks` through `cache` and returns the logits of
        /// every position.
        static func chunkedLogits(
            _ model: DeepseekV4Model, rows: [[Int]], chunks: [Int], cache: [KVCache]
        ) -> MLXArray {
            var start = 0
            var parts: [MLXArray] = []
            for chunk in chunks {
                let part = rows.map { Array($0[start ..< start + chunk]) }
                parts.append(ForwardPassChecks.logits(model, part, cache: cache))
                start += chunk
            }
            return concatenated(parts, axis: 1)
        }

        // Tolerance of the float32 comparisons: the paths run the same
        // float32 math, with sums in another order (attention with one query,
        // the pooled attention in log space, the Sinkhorn kernel). A cache
        // fault gives differences above 1e-2.
        static let tolerance: Float = 1e-4

        @Test func logitsHaveTheExpectedShapeAndAreFinite() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkShapeDTypeAndFinite(
                model, vocabularySize: Self.vocabularySize, length: 7)
        }

        @Test func sameSeedGivesTheSameLogits() throws {
            try ForwardPassChecks.checkDeterminism(
                make: { try Self.makeModel() }, seed: 3, vocabularySize: Self.vocabularySize)
        }

        @Test func eachRowOfABatchMatchesTheRowAlone() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkBatchInvariance(
                model, rowA: Self.row(1, count: 7), rowB: Self.row(2, count: 7),
                tolerance: Self.tolerance)
        }

        /// The fused Sinkhorn kernel of the hyper connection must give the
        /// same result as the ops path that `hcPre` uses on the CPU.
        @Test func hyperConnectionKernelMatchesTheOpsPath() throws {
            let model = try Self.makeModel()
            let params = model.model.layers[0].attn_hc
            let x = MLXRandom.normal([2, 3, 4, 32], key: MLXRandom.key(9))
            eval(x)
            let (gpuY, gpuPost, gpuComb) = hcPre(
                x: x, hcFn: params.fn, hcScale: params.scale, hcBase: params.base, hcMult: 4,
                sinkhornIters: 3, eps: 1e-6)
            eval(gpuY, gpuPost, gpuComb)
            let (cpuY, cpuPost, cpuComb) = Device.withDefaultDevice(.cpu) {
                let result = hcPre(
                    x: x, hcFn: params.fn, hcScale: params.scale, hcBase: params.base,
                    hcMult: 4, sinkhornIters: 3, eps: 1e-6)
                eval(result.0, result.1, result.2)
                return result
            }
            #expect(SyntheticModel.maxAbsDifference(gpuPost, cpuPost) <= 1e-5, "post")
            #expect(SyntheticModel.maxAbsDifference(gpuComb, cpuComb) <= 1e-5, "comb")
            withKnownIssue(
                """
                The two paths of hcPre disagree on the collapsed output. The ops path \
                divides `pre` by its row sum (DeepseekV4.swift:597). The reference \
                (mlx-lm deepseek_v41.py:241) computes `pre = sigmoid(pre) + hc_eps` with \
                no normalization, and the Metal kernel does the same. The ops path is \
                the side that differs from the reference; the kernel matches it. The \
                GPU path is the one that runs in production.
                """
            ) {
                #expect(SyntheticModel.maxAbsDifference(gpuY, cpuY) <= 1e-4, "collapsed")
            } matching: {
                $0.isFailedExpectation(["collapsed"])
            }
        }

        /// The compressed caches of `makeCache(parameters:)`: a prompt in
        /// chunks and decode steps give the same logits as the same prompt in
        /// one chunk with a new cache.
        @Test func chunkedPrefillAndDecodeMatchOnePrefill() throws {
            let model = try Self.makeModel()
            let rows = [Self.row(1, count: 136)]
            let whole = Self.chunkedLogits(
                model, rows: rows, chunks: [136], cache: Self.compressedCache(model))
            let chunked = Self.chunkedLogits(
                model, rows: rows, chunks: [64, 66, 1, 1, 1, 1, 1, 1],
                cache: Self.compressedCache(model))
            let difference = SyntheticModel.maxAbsDifference(chunked, whole)
            #expect(difference <= Self.tolerance, "differs by \(difference)")
        }

        /// Without pooled windows (local attention, and compressed attention
        /// before its first window of 128 tokens), the compressed caches give
        /// the same logits as the full pass, for a sequence longer than the
        /// sliding window of 8.
        @Test(arguments: [[0, 0, 0], [0, 128, 0]])
        func unpooledLayersMatchTheFullForwardPass(ratios: [Int]) throws {
            let model = try Self.makeModel(["compress_ratios": ratios])
            let rows = [Self.row(1, count: 12), Self.row(2, count: 12)]
            let full = ForwardPassChecks.logits(model, rows)
            for chunks in [[12], [6, 1, 1, 1, 1, 1, 1], [4, 4, 1, 1, 1, 1]] {
                let cached = Self.chunkedLogits(
                    model, rows: rows, chunks: chunks, cache: Self.compressedCache(model))
                let difference = SyntheticModel.maxAbsDifference(cached, full)
                #expect(difference <= Self.tolerance, "chunks \(chunks): \(difference)")
            }
        }

        /// The full pass without a cache matches the pass with the
        /// compressed caches of `makeCache(parameters:)`.
        @Test func compressedCacheMatchesTheFullForwardPass() throws {
            let model = try Self.makeModel()
            let rows = [Self.row(1, count: 136)]
            let full = ForwardPassChecks.logits(model, rows)
            let cached = Self.chunkedLogits(
                model, rows: rows, chunks: [136], cache: Self.compressedCache(model))
            let difference = SyntheticModel.maxAbsDifference(cached, full)
            #expect(difference <= Self.tolerance, "differs by \(difference)")
        }

        @Test func aLaterTokenDoesNotChangeEarlierLogits() throws {
            let model = try Self.makeModel()
            // 3 tokens give no pooled window. 12 tokens give 3 windows of 4.
            ForwardPassChecks.checkCausality(
                model, row: Self.row(1, count: 3), position: 2,
                vocabularySize: Self.vocabularySize, tolerance: Self.tolerance)
            ForwardPassChecks.checkCausality(
                model, row: Self.row(1, count: 12), position: 9,
                vocabularySize: Self.vocabularySize, tolerance: Self.tolerance)
        }

        /// `newCache(parameters:)` is what the generation code calls. It must
        /// give the caches that the attention layers need.
        @Test func newCacheGivesTheCompressedCaches() throws {
            let model = try Self.makeModel()
            let cache = model.newCache(parameters: nil)
            withKnownIssue(
                """
                DeepseekV4Model does not implement newCache(parameters:). The default \
                of KVCacheDimensionProvider gives a KVCacheSimple for each layer, and \
                makeCache(parameters:) (DeepseekV4.swift:1626) is never called. \
                Generation then runs without the sliding window and without the pooled \
                caches.
                """
            ) {
                #expect(cache[0] is RotatingKVCache, "layer 0 cache")
                #expect(cache[1] is DeepseekV4LayerCache, "layer 1 cache")
                #expect(cache[2] is DeepseekV4LayerCache, "layer 2 cache")
            } matching: {
                $0.isFailedExpectation(["layer 0 cache", "layer 1 cache", "layer 2 cache"])
            }
        }

        @Test func parameterTreeHasTheCheckpointKeysAndShapes() throws {
            let parameters = SyntheticModel.flatParameters(try Self.makeModel())
            // hc_mult 4 x hidden 32 = 128 inputs to the hyper connection,
            // (2 + 4) x 4 = 24 mixes.
            let expected: [String: [Int]] = [
                "model.embed_tokens.weight": [64, 32],
                "model.hc_head.fn": [4, 128],
                "model.layers.0.attn_hc.fn": [24, 128],
                "model.layers.0.attn_hc.base": [24],
                "model.layers.0.attn_hc.scale": [3],
                "model.layers.0.attn.wq_a.weight": [16, 32],
                "model.layers.0.attn.wq_b.weight": [32, 16],
                "model.layers.0.attn.wkv.weight": [16, 32],
                "model.layers.0.attn.wo_a.weight": [2, 8, 16],
                "model.layers.0.attn.wo_b.weight": [32, 16],
                "model.layers.0.attn.attn_sink": [2],
                "model.layers.0.ffn.gate.tid2eid": [64, 2],
                "model.layers.1.ffn.gate.e_score_correction_bias": [4],
                "model.layers.1.attn.compressor.wkv.weight": [32, 32],
                "model.layers.1.attn.compressor.ape": [4, 32],
                "model.layers.1.attn.indexer.wq_b.weight": [32, 16],
                "model.layers.1.attn.indexer.weights_proj.weight": [2, 32],
                "model.layers.2.attn.compressor.wkv.weight": [16, 32],
                "model.layers.2.attn.compressor.ape": [128, 16],
                "model.layers.2.ffn.switch_mlp.gate_proj.weight": [4, 16, 32],
                "model.layers.2.ffn.shared_experts.down_proj.weight": [32, 16],
                "lm_head.weight": [64, 32],
            ]
            for (key, shape) in expected {
                #expect(parameters[key]?.shape == shape, "\(key)")
            }
            #expect(parameters["model.layers.0.ffn.gate.e_score_correction_bias"] == nil)
            #expect(parameters["model.layers.1.ffn.gate.tid2eid"] == nil)
        }

        /// The original DeepSeek checkpoint layout: top-level `embed`,
        /// `head` and `hc_head_*` names, `layers.N` without `model.`, flat
        /// `hc_attn_fn` names, the router bias as `gate.bias`, `w1`/`w2`/`w3`
        /// expert names, one tensor per expert and a 2-D `wo_a`.
        static func originalCheckpoint(_ model: DeepseekV4Model) -> [String: MLXArray] {
            var checkpoint: [String: MLXArray] = [:]
            let top = [
                "model.embed_tokens.weight": "embed.weight", "model.norm.weight": "norm.weight",
                "lm_head.weight": "head.weight", "model.hc_head.fn": "hc_head_fn",
                "model.hc_head.base": "hc_head_base", "model.hc_head.scale": "hc_head_scale",
            ]
            let expertNames = ["gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"]
            for (key, value) in SyntheticModel.flatParameters(model) {
                if let name = top[key] {
                    checkpoint[name] = value
                    continue
                }
                var name = String(key.dropFirst("model.".count))
                for sub in ["attn", "ffn"] {
                    for param in ["fn", "base", "scale"] {
                        name = name.replacingOccurrences(
                            of: ".\(sub)_hc.\(param)", with: ".hc_\(sub)_\(param)")
                    }
                }
                name = name.replacingOccurrences(
                    of: ".ffn.gate.e_score_correction_bias", with: ".ffn.gate.bias")
                for (new, old) in expertNames {
                    name = name.replacingOccurrences(
                        of: ".shared_experts.\(new).", with: ".shared_experts.\(old).")
                }
                if name.contains(".switch_mlp.") {
                    for (new, old) in expertNames where name.contains(".\(new).") {
                        for expert in 0 ..< 4 {
                            let expertName = name.replacingOccurrences(
                                of: ".switch_mlp.\(new).", with: ".experts.\(expert).\(old).")
                            checkpoint[expertName] = value[expert]
                        }
                    }
                    continue
                }
                if name.hasSuffix("attn.wo_a.weight") {
                    checkpoint[name] = value.reshaped(-1, value.dim(-1))
                    continue
                }
                checkpoint[name] = value
            }
            return checkpoint
        }

        @Test func loaderConvertsTheOriginalCheckpointLayout() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = Self.originalCheckpoint(reference)
            #expect(checkpoint["layers.1.hc_attn_fn"]?.shape == [24, 128])
            #expect(checkpoint["layers.2.ffn.experts.3.w2.weight"]?.shape == [32, 16])
            #expect(checkpoint["layers.0.attn.wo_a.weight"]?.shape == [16, 16])
            checkpoint["layers.0.attn.rotary_emb.inv_freq"] = MLXArray.ones([4])

            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            let rows = [Self.row(3, count: 9)]
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows)) == 0)
        }

        /// An FP8 checkpoint stores `weight` with a `weight_scale_inv` for
        /// each 128 x 128 block. `sanitize(weights:)` multiplies them out.
        @Test func loaderDequantizesBlockScaledWeights() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = Self.originalCheckpoint(reference)
            let key = "layers.0.attn.wq_a.weight"
            checkpoint[key] = checkpoint[key]! / 4
            checkpoint["layers.0.attn.wq_a.weight_scale_inv"] = MLXArray([4] as [Float])
                .reshaped(1, 1)

            let loaded = try Self.makeModel(seed: 6)
            let sanitized = loaded.sanitize(weights: checkpoint)
            #expect(
                SyntheticModel.maxAbsDifference(
                    sanitized["model.layers.0.attn.wq_a.weight"]!,
                    SyntheticModel.flatParameters(reference)["model.layers.0.attn.wq_a.weight"]!)
                    <= 1e-6)
            // The strict update throws for the unused keys, so a thrown error
            // of the load is also the known issue.
            try withKnownIssue(
                """
                sanitize(weights:) keeps the `weight_scale_inv` tensors after it \
                dequantizes the weights (DeepseekV4.swift:1665, the FP8 loop starts from \
                a copy of every key), so the strict update rejects the unused keys.
                """
            ) {
                #expect(
                    !sanitized.keys.contains { $0.contains("weight_scale_inv") },
                    "weight_scale_inv kept")
                try SyntheticModel.load(checkpoint, into: loaded)
            } matching: {
                $0.error != nil || $0.isFailedExpectation(["weight_scale_inv kept"])
            }
        }

        @Test func loaderRejectsAWrongShape() throws {
            var checkpoint = Self.originalCheckpoint(try Self.makeModel())
            checkpoint["layers.1.hc_ffn_fn"] = MLXArray.zeros([24, 64])
            #expect(throws: (any Error).self) {
                try SyntheticModel.load(checkpoint, into: try Self.makeModel())
            }
        }
    }
}
