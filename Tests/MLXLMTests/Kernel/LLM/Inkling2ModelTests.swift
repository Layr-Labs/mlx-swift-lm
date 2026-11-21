import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Forward-pass tests of a tiny random Inkling 2: 6 layers in blocks of 2, the
/// loop over layers 2..<4 run 3 times (10 visits), full layers 1, 3, 5, a
/// 6-token window, 8-token bias band and log scaling from position 4.
extension KernelTests {
    @Suite
    struct Inkling2ModelTests {
        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "model_type": "inkling2_loop",
                "eos_token_id": 3,
                "text_config": [
                    "model_type": "inkling2_loop_text",
                    "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": vocabularySize,
                    "unpadded_vocab_size": 60, "num_attention_heads": 4,
                    "num_key_value_heads": 2, "head_dim": 16, "d_rel": 4, "rel_extent": 8,
                    "log_scaling_n_floor": 4, "log_scaling_alpha": 0.5, "rms_norm_eps": 1e-6,
                    "qk_l2_clip_threshold": 4.0, "rel_l2_clip_threshold": 2.0,
                    "local_layer_ids": [0, 2, 4], "sconv_kernel_size": 4,
                    "final_logit_softcapping": 30.0, "sliding_window_size": 6,
                    "n_layers_per_attn_res_block": 2, "n_routed_experts": 8,
                    "num_experts_per_tok": 2, "n_shared_experts": 0, "intermediate_size": 32,
                    "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
                    "loop_config": [
                        "core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8,
                        "is_varloop": false,
                    ],
                ] as [String: Any],
            ]
        }

        static func config() -> Inkling2TextConfiguration {
            try! SyntheticModel.configuration(Inkling2Configuration.self, base).text
        }

        static func makeModel(seed: UInt64 = 1) throws -> Inkling2Model {
            let model = Inkling2Model(
                try SyntheticModel.configuration(Inkling2Configuration.self, base))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func tokens(_ seed: Int, count: Int) -> MLXArray {
            MLXArray(
                SyntheticModel.tokens(count: count, vocabularySize: 60, seed: seed).map(Int32.init)
            )
            .reshaped(1, count)
        }

        static func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
            abs(a - b).max().item(Float.self)
        }

        /// Test-only incremental state: K/V history and conv states held in
        /// memory per slot, as a reference for the cache semantics.
        final class IncrementalState: Inkling2StepState {
            var positions = MLXArray.zeros([1, 0], dtype: .int32)
            var offset = 0
            var conv: [Int: MLXArray] = [:]
            var keys: [Int: MLXArray] = [:]
            var values: [Int: MLXArray] = [:]

            func begin(_ length: Int) {
                positions = MLXArray((offset ..< offset + length).map { Int32($0) }).reshaped(
                    1, length)
            }
            func end(_ length: Int) { offset += length }

            func convState(slot: Int, _ kind: Inkling2ConvKind) -> MLXArray? {
                conv[slot * 4 + kind.rawValue]
            }
            func setConvState(
                slot: Int, _ kind: Inkling2ConvKind, _ state: MLXArray, input: MLXArray
            ) {
                conv[slot * 4 + kind.rawValue] = state
            }
            func attend(slot: Int, _ inputs: Inkling2AttentionInputs) -> MLXArray {
                let k = keys[slot].map { concatenated([$0, inputs.keys], axis: 2) } ?? inputs.keys
                let v =
                    values[slot].map { concatenated([$0, inputs.values], axis: 2) } ?? inputs.values
                keys[slot] = k
                values[slot] = v
                let keyPositions = MLXArray((0 ..< k.dim(2)).map { Int32($0) })
                return Inkling2ReferenceAttention.attend(
                    inputs, keys: k, values: v, keyPositions: keyPositions)
            }
        }

        @Test func layoutHasFortyEightStyleSlotsAndVisits() throws {
            let c = Self.config()
            #expect(c.visitCount == 10)
            #expect(c.slotLayers == [0, 1, 2, 3, 4, 5, 2, 3, 2, 3])
            #expect(c.slot(layer: 3, pass: 2) == 9)
            #expect(c.visits.map { $0.layer } == [0, 1, 2, 3, 2, 3, 2, 3, 4, 5])
        }

        @Test func logitsHaveTheExpectedShapeAndAreFinite() throws {
            let model = try Self.makeModel()
            let logits = model.forwardNoCache(Self.tokens(1, count: 9)).logits
            #expect(logits.shape == [1, 9, 60])
            #expect(logits.dtype == .float32)
            #expect(all(isFinite(logits)).item(Bool.self))
            #expect(abs(logits).max().item(Float.self) <= 30)
        }

        @Test func laterTokensDoNotChangeEarlierLogits() throws {
            let model = try Self.makeModel()
            let long = Self.tokens(2, count: 14)
            let full = model.forwardNoCache(long).logits
            let prefix = model.forwardNoCache(long[0..., 0 ..< 8]).logits
            #expect(Self.maxDifference(full[0..., 0 ..< 8], prefix) < 1e-4)
        }

        /// Prefill then decode through carried K/V and conv states equals one
        /// full recompute (window 6, bias band 8, log scaling all exercised).
        @Test(arguments: [[14], [5, 1, 1, 7], [1, 1, 1, 1, 10], [7, 7]])
        func incrementalStepsMatchTheFullRecompute(chunks: [Int]) throws {
            let model = try Self.makeModel()
            let tokens = Self.tokens(3, count: chunks.reduce(0, +))
            let full = model.forwardNoCache(tokens).logits
            let state = IncrementalState()
            var pieces: [MLXArray] = []
            for length in chunks {
                state.begin(length)
                pieces.append(
                    model.forward(
                        tokens[0..., state.offset ..< state.offset + length], state: state
                    ).logits)
                state.end(length)
            }
            // FP32 throughout; only attention and conv summation orders differ.
            #expect(Self.maxDifference(concatenated(pieces, axis: 1), full) < 1e-4)
        }

        @Test func eachRowOfABatchMatchesTheRowAlone() throws {
            let model = try Self.makeModel()
            let a = Self.tokens(4, count: 10)
            let b = Self.tokens(5, count: 10)
            let both = model.forwardNoCache(concatenated([a, b], axis: 0)).logits
            #expect(Self.maxDifference(both[0 ..< 1], model.forwardNoCache(a).logits) < 1e-4)
            #expect(Self.maxDifference(both[1 ..< 2], model.forwardNoCache(b).logits) < 1e-4)
        }

        /// Compiled elementwise chains change no bits of the full forward.
        @Test func fusedGraphsMatchThePlainGraphsBitForBit() throws {
            let model = try Self.makeModel(seed: 21)
            let tokens = Self.tokens(8, count: 13)
            let plain = Inkling2Fused.$override.withValue(false) {
                model.forwardNoCache(tokens).logits
            }
            let fused = Inkling2Fused.$override.withValue(true) {
                model.forwardNoCache(tokens).logits
            }
            #expect(arrayEqual(plain, fused).item(Bool.self))
        }

        @Test func captureReturnsVisitInputsAndTheFinalNorm() throws {
            let model = try Self.makeModel()
            let tokens = Self.tokens(6, count: 5)
            let (_, captured) = model.forwardNoCache(tokens, capture: [0, 3, 10])
            #expect(captured.keys.sorted() == [0, 3, 10])
            let embedded = model.scaledInputEmbeddings(tokens)
            #expect(arrayEqual(captured[0]!, embedded).item(Bool.self))
            #expect(captured[10]!.shape == [1, 5, 64])
        }

        /// A checkpoint in the original layout (`model.llm.*`, split message
        /// queries, `attn_res.*`, padded unembedding) loads into the same model.
        @Test func originalCheckpointLayoutLoadsTheSameWeights() throws {
            let source = try Self.makeModel(seed: 11)
            let c = Self.config()
            var original = [String: MLXArray]()
            for (key, value) in SyntheticModel.flatParameters(source) {
                if key == "lm_head.weight" {
                    let padding = MLXArray.zeros([c.vocabularySize - value.dim(0), value.dim(1)])
                    original["model.llm.unembed.weight"] = concatenated([value, padding], axis: 0)
                } else if key == "model.message_input.query" {
                    let rows = value.dim(0)
                    original["model.llm.middle_cycle_cross_pass_queries"] = stacked(
                        [value[0..., 0], value[0..., 3]], axis: 1
                    ).reshaped(2 * rows, -1)
                    original["model.llm.middle_cycle_cross_pass_kv_queries"] = value[0..., 1 ..< 3]
                        .reshaped(2 * rows, -1)
                } else if key == "model.final_input.query" {
                    original["model.llm.attn_res.output_query"] = value
                } else if key.hasSuffix(".attention_input.query") {
                    let layer = Int(key.split(separator: ".")[2])!
                    original["model.llm.attn_res.attn_queries.\(layer - 1)"] = value
                } else if key.hasSuffix(".mlp_input.query") {
                    let layer = Int(key.split(separator: ".")[2])!
                    original["model.llm.attn_res.mlp_queries.\(layer)"] = value
                } else {
                    original["model.llm." + key.dropFirst("model.".count)] = value
                }
            }
            original["model.visual.patch.weight"] = MLXArray.zeros([2, 2])

            let target = try Self.makeModel(seed: 99)
            let checkpoint = try target.prepareCheckpoint(ModelCheckpoint(weights: original))
            try target.update(
                parameters: ModuleParameters.unflattened(checkpoint.weights), verify: [.all])
            let tokens = Self.tokens(7, count: 8)
            #expect(
                arrayEqual(
                    target.forwardNoCache(tokens).logits, source.forwardNoCache(tokens).logits
                )
                .item(Bool.self))
        }
    }
}
