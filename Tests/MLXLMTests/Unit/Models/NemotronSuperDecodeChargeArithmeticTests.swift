import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

extension UnitTests {
    /// Batched decode charges of the pinned Super language configuration.
    /// Configuration arithmetic only; no arrays are created.
    struct NemotronSuperDecodeChargeArithmeticTests {
        static let pinnedLanguage = Data(
            #"{"attention_bias":false,"chunk_size":128,"conv_kernel":4,"dtype":"bfloat16","head_dim":128,"hidden_size":4096,"intermediate_size":2688,"layer_norm_epsilon":1e-05,"layers_block_type":["mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe"],"mamba_head_dim":64,"mamba_hidden_act":"silu","mamba_num_heads":128,"mamba_proj_bias":false,"mamba_ssm_cache_dtype":"float32","mlp_bias":false,"mlp_hidden_act":"relu2","model_type":"nemotron_h","moe_intermediate_size":2688,"moe_latent_size":1024,"moe_shared_expert_intermediate_size":5376,"mtp_layers_block_type":["attention","moe"],"n_group":1,"n_groups":8,"n_routed_experts":512,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":32,"num_experts_per_tok":22,"num_key_value_heads":2,"num_nextn_predict_layers":1,"routed_scaling_factor":5.0,"ssm_state_size":128,"tie_word_embeddings":false,"time_step_limit":[0.0,{"__float__":"Infinity"}],"topk_group":1,"use_bias":false,"use_conv_bias":true,"vocab_size":131072}"#
                .utf8)

        @Test func pinnedConfigurationRecurrentKVAndTrunkCharges() throws {
            let c = try NemotronSuperModel.validatedLanguageConfiguration(Self.pinnedLanguage)
            typealias Charge = NemotronSuperDecodeCharge
            // 40 Mamba layers: BF16 convolution history (kernel 4 - 1 rows of
            // 8192 + 2 * 8 * 128 channels) and float32 SSM (128 x 64 x 128).
            let perLayer = 3 * 10_240 * 2 + 128 * 64 * 128 * 4
            #expect(perLayer == 4_255_744)
            #expect(
                try Charge.recurrentBytesPerGeneration(
                    configuration: c, convolutionDType: .bfloat16)
                    == 40 * perLayer)
            // 8 attention layers, 2 KV heads of 128 for K and V.
            #expect(try Charge.attentionKVBytesPerToken(configuration: c, elementBytes: 2) == 8_192)
            #expect(try Charge.attentionKVBytesPerToken(configuration: c, elementBytes: 1) == 4_096)
            // The Mamba layer is the widest: in_proj 18 560, padded convolution
            // 4 x 10 240, two convolution outputs and four 8 192 SSM/norm rows,
            // plus three hidden rows of residual stream.
            #expect(
                try Charge.trunkElements(configuration: c)
                    == 3 * 4_096 + 18_560 + 4 * 10_240 + 2 * 10_240 + 4 * 8_192)
            #expect(throws: Charge.Failure.invalidBound) {
                try Charge.attentionKVBytesPerToken(configuration: c, elementBytes: 0)
            }
        }

        @Test func requestAndAliasChargesAreCheckedArithmetic() throws {
            let workspace = NemotronSuperDecodeCharge.RowWorkspace(
                modelOptLayerBytes: 1, outputHeadBytes: 2, logitsBytes: 3, trunkBytes: 4,
                backendScratchBytes: 5, totalBytes: 15)
            let charge = NemotronSuperDecodeCharge(
                recurrentBytesPerGeneration: 170_229_760, recurrentGenerations: 3,
                attentionKVBytesPerToken: 8_192, rowWorkspace: workspace)
            #expect(
                try charge.requestBytes(maximumTokens: 16_384) == 3 * 170_229_760 + 8_192 * 16_384
                    + 15)
            #expect(try charge.batchAliasBytes(maximumRows: 1) == 0)
            #expect(try charge.batchAliasBytes(maximumRows: 8) == 7 * 3 * 170_229_760)
            #expect(throws: NemotronSuperDecodeCharge.Failure.invalidBound) {
                try charge.requestBytes(maximumTokens: 0)
            }
            #expect(throws: NemotronSuperDecodeCharge.Failure.overflow) {
                try charge.requestBytes(maximumTokens: Int.max)
            }
        }

        @Test func adapterLayoutAcceptsOnlyRectangularDecodeRows() throws {
            typealias Adapter = NemotronSuperRecurrentAdapter
            func check(
                _ rows: Int, _ length: Int, owners: Int? = nil, positions: Bool = false,
                caches: Int = 8, maximum: Int = 4
            ) throws {
                try Adapter.validateLayout(
                    batchSize: rows, sequenceLength: length, hasExplicitPositions: positions,
                    recurrentOwnerCount: owners ?? rows, attentionCacheCount: caches,
                    expectedAttentionCacheCount: 8, maximumDecodeRows: maximum)
            }
            try check(1, 7)
            try check(4, 1)
            try check(1, 1, maximum: 1)
            #expect(throws: Adapter.Failure.batch) { try check(2, 2) }
            #expect(throws: Adapter.Failure.batch) { try check(5, 1) }
            #expect(throws: Adapter.Failure.batch) { try check(2, 1, maximum: 1) }
            #expect(throws: Adapter.Failure.batch) { try check(0, 1) }
            #expect(throws: Adapter.Failure.position) { try check(2, 1, positions: true) }
            #expect(throws: Adapter.Failure.recurrentOwnership) { try check(3, 1, owners: 2) }
            #expect(throws: Adapter.Failure.recurrentOwnership) { try check(1, 4, owners: 2) }
            #expect(throws: Adapter.Failure.caches) { try check(2, 1, caches: 7) }
            // The serial default is unchanged.
            #expect(throws: Adapter.Failure.batch) {
                try Adapter.validateLayout(
                    batchSize: 2, sequenceLength: 1, hasExplicitPositions: false,
                    recurrentOwnerCount: 2, attentionCacheCount: 8,
                    expectedAttentionCacheCount: 8)
            }
        }
    }
}
