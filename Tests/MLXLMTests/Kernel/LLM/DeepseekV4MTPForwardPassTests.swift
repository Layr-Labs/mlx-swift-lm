import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of the DeepSeek-V4 multi-token prediction (MTP)
    /// head with a tiny random model.
    ///
    /// The configuration is the one of `DeepseekV4ForwardPassTests`, with 1
    /// local-attention layer (compress ratio 0), no hash layer and 1 MTP
    /// layer. The MTP block gets the layer index 1, which has no compress
    /// ratio, so it uses local attention. All sequences have 7 tokens, less
    /// than the sliding window of 8.
    ///
    /// `DeepseekV4Model.init` attaches the MTP head only while the global
    /// `_deepseekV4MTPEnabled` is true. The suite is serialized, and each
    /// model build sets the flag and resets it to false.
    ///
    /// Tolerance 1e-4 for the float32 comparisons: the cached path and the
    /// full path differ in the order of the attention sums.
    @Suite(.serialized)
    struct DeepseekV4MTPForwardPassTests {

        static let vocabularySize = 64
        static let tolerance: Float = 1e-4

        static var base: [String: Any] {
            [
                "vocab_size": vocabularySize,
                "hidden_size": 32,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "head_dim": 16,
                "q_lora_rank": 16,
                "qk_rope_head_dim": 8,
                "rms_norm_eps": 1e-6,
                "o_groups": 2,
                "o_lora_rank": 8,
                "sliding_window": 8,
                "compress_ratios": [0],
                "compress_rope_theta": 160000,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "scoring_func": "sqrtsoftplus",
                "routed_scaling_factor": 1.5,
                "swiglu_limit": 10.0,
                "num_hash_layers": 0,
                "num_nextn_predict_layers": 1,
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

        /// Builds the model. With `seed` nil, the model keeps its initial
        /// weights and nothing reads its module tree before the load, as in
        /// the model factory.
        static func makeModel(mtp: Bool = true, seed: UInt64? = 1) throws -> DeepseekV4Model {
            let configuration = try SyntheticModel.configuration(
                DeepseekV4Configuration.self, base)
            _deepseekV4MTPEnabled = mtp
            defer { _deepseekV4MTPEnabled = false }
            let model = DeepseekV4Model(configuration)
            if let seed {
                SyntheticModel.randomize(model, seed: seed)
            }
            return model
        }

        static func tokens(_ seed: Int) -> MLXArray {
            SyntheticModel.batch([
                SyntheticModel.tokens(count: 7, vocabularySize: vocabularySize, seed: seed)
            ])
        }

        /// The raw backbone hidden state `[1, 7, hc, D]` for the MTP head.
        static func rawHidden(_ model: DeepseekV4Model) -> MLXArray {
            let (_, hidden) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens(1)), cache: [], nConfirmed: 0)
            eval(hidden)
            return hidden
        }

        static func draft(
            _ model: DeepseekV4Model, hidden: MLXArray, next: MLXArray, cache: [any KVCache]
        ) -> MLXArray {
            let logits = model.mtpForward(hidden: hidden, nextTokenIds: next, cache: cache)
            eval(logits)
            return logits
        }

        @Test func mtpHeadIsAttachedOnlyWhenEnabled() throws {
            let model = try Self.makeModel()
            #expect(model.hasMTPHead)
            let cache = model.makeMTPCache()
            #expect(cache.count == 1)
            #expect(cache.allSatisfy { $0 is KVCacheSimple })

            let parameters = SyntheticModel.flatParameters(model)
            let expected: [String: [Int]] = [
                "mtp.0.e_proj.weight": [32, 32],
                "mtp.0.h_proj.weight": [32, 32],
                "mtp.0.enorm.weight": [32],
                "mtp.0.hnorm.weight": [32],
                "mtp.0.norm.weight": [32],
                // hc_mult 4 x hidden 32 = 128 inputs.
                "mtp.0.hc_head.fn": [4, 128],
                "mtp.0.hc_head.base": [4],
                "mtp.0.hc_head.scale": [1],
                "mtp.0.block.attn.wq_a.weight": [16, 32],
                "mtp.0.block.ffn.switch_mlp.gate_proj.weight": [4, 16, 32],
                "mtp.0.block.ffn.gate.e_score_correction_bias": [4],
            ]
            for (key, shape) in expected {
                #expect(parameters[key]?.shape == shape, "\(key)")
            }

            let plain = try Self.makeModel(mtp: false)
            #expect(!plain.hasMTPHead)
            #expect(plain.makeMTPCache().isEmpty)
            #expect(!SyntheticModel.flatParameters(plain).keys.contains { $0.hasPrefix("mtp.") })
        }

        /// `callWithHidden` gives the logits of the normal forward pass and
        /// the raw hidden state before the hyper-connection head. Without a
        /// cache, both calls run the same operations, so the logits must be
        /// equal. With the caches of `makeCache(parameters:)` the tolerance
        /// is 1e-4.
        @Test func callWithHiddenMatchesTheForwardPass() throws {
            let model = try Self.makeModel()
            let tokens = Self.tokens(1)
            let full = ForwardPassChecks.logits(
                model, [SyntheticModel.tokens(count: 7, vocabularySize: 64, seed: 1)])

            let (logits, hidden) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens), cache: [], nConfirmed: 0)
            eval(logits, hidden)
            #expect(hidden.shape == [1, 7, 4, 32])
            #expect(SyntheticModel.maxAbsDifference(logits, full) == 0)

            let (cachedLogits, _) = model.callWithHidden(
                input: LMInput.Text(tokens: tokens),
                cache: model.makeCache(parameters: GenerateParameters()), nConfirmed: 0)
            eval(cachedLogits)
            #expect(SyntheticModel.maxAbsDifference(cachedLogits, full) <= Self.tolerance)
        }

        @Test func draftLogitsHaveTheExpectedShapeAndAreFinite() throws {
            let model = try Self.makeModel()
            let logits = Self.draft(
                model, hidden: Self.rawHidden(model), next: Self.tokens(2),
                cache: model.makeMTPCache())
            #expect(logits.shape == [1, 7, Self.vocabularySize])
            #expect(logits.dtype == .float32)
            #expect(isFinite(logits).all().item(Bool.self))
            #expect(SyntheticModel.maxAbs(logits) > 1e-3)
        }

        /// The draft logits of the next tokens in chunks with the MTP cache
        /// must match one pass without a cache.
        @Test func cachedDraftMatchesTheFullDraft() throws {
            let model = try Self.makeModel()
            let hidden = Self.rawHidden(model)
            let next = Self.tokens(2)
            let full = Self.draft(model, hidden: hidden, next: next, cache: [])

            let cache = model.makeMTPCache()
            var start = 0
            for chunk in [4, 1, 1, 1] {
                let stepped = Self.draft(
                    model, hidden: hidden[0..., start ..< start + chunk],
                    next: next[0..., start ..< start + chunk], cache: cache)
                let difference = SyntheticModel.maxAbsDifference(
                    stepped, full[0..., start ..< start + chunk, 0...])
                #expect(
                    difference <= Self.tolerance,
                    "positions \(start) ..< \(start + chunk): \(difference)")
                start += chunk
            }
            #expect(cache[0].offset == 7)
        }

        /// A change of the next token at position 4 must not change the
        /// draft logits of earlier positions.
        @Test func aLaterNextTokenDoesNotChangeEarlierDraftLogits() throws {
            let model = try Self.makeModel()
            let hidden = Self.rawHidden(model)
            var row = SyntheticModel.tokens(count: 7, vocabularySize: 64, seed: 2)
            let original = Self.draft(
                model, hidden: hidden, next: SyntheticModel.batch([row]), cache: [])
            row[4] = (row[4] + 1) % 64
            let changed = Self.draft(
                model, hidden: hidden, next: SyntheticModel.batch([row]), cache: [])
            let before = SyntheticModel.maxAbsDifference(original[0..., ..<4], changed[0..., ..<4])
            let at = SyntheticModel.maxAbsDifference(original[0..., 4...], changed[0..., 4...])
            #expect(before <= Self.tolerance, "positions before 4 changed by \(before)")
            #expect(at > 1e-3, "the change at 4 must change its own logits")
        }

        /// A checkpoint with one tensor per expert for the backbone and for
        /// the MTP layer loads, and the loaded model gives the same backbone
        /// and draft logits as the reference.
        @Test func loaderStacksTheMTPExperts() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = CheckpointLayout.splitExperts(
                SyntheticModel.flatParameters(reference), stacked: "switch_mlp",
                perExpert: "experts",
                names: ["gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"])
            #expect(checkpoint["mtp.0.block.ffn.experts.3.w2.weight"]?.shape == [32, 16])
            #expect(checkpoint["model.layers.0.ffn.experts.0.w1.weight"]?.shape == [16, 32])
            checkpoint["mtp.0.block.attn.rotary_emb.inv_freq"] = MLXArray.ones([4])

            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            #expect(loaded.hasMTPHead)

            let rows = [SyntheticModel.tokens(count: 7, vocabularySize: 64, seed: 3)]
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows)) == 0)
            let hidden = Self.rawHidden(reference)
            #expect(
                SyntheticModel.maxAbsDifference(
                    Self.draft(reference, hidden: hidden, next: Self.tokens(2), cache: []),
                    Self.draft(loaded, hidden: hidden, next: Self.tokens(2), cache: [])) == 0)
        }

        /// A checkpoint without `mtp.` keys detaches the MTP head, so the
        /// strict load succeeds and the model reports no head.
        ///
        /// `sanitize(weights:)` detaches the head with `self.mtp = nil`. The
        /// module tree of MLXNN keeps a cache of the child modules once it is
        /// read, and this assignment does not clear it. So the loaded model
        /// is a new model that nothing has read, as in the model factory.
        @Test func loaderDetachesTheHeadWhenTheCheckpointHasNoMTPWeights() throws {
            let reference = try Self.makeModel(seed: 5)
            let checkpoint = SyntheticModel.flatParameters(reference).filter {
                !$0.key.hasPrefix("mtp.")
            }
            let loaded = try Self.makeModel(seed: nil)
            #expect(loaded.hasMTPHead)
            try SyntheticModel.load(checkpoint, into: loaded)
            #expect(!loaded.hasMTPHead)
            #expect(loaded.makeMTPCache().isEmpty)

            let rows = [SyntheticModel.tokens(count: 7, vocabularySize: 64, seed: 3)]
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows)) == 0)
        }

        @Test func loaderRejectsAWrongMTPShape() throws {
            var checkpoint = SyntheticModel.flatParameters(try Self.makeModel())
            checkpoint["mtp.0.e_proj.weight"] = MLXArray.zeros([33, 32])
            #expect(throws: (any Error).self) {
                try SyntheticModel.load(checkpoint, into: try Self.makeModel())
            }
        }
    }
}
