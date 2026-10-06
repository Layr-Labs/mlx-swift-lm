import Foundation
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpConfigurationTests: XCTestCase {
    func testFlashNextTextConfigDecodeAndCBv2Layout() throws {
        let url = URL(
            fileURLWithPath: NSString(
                string:
                    "~/.lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp/config.json"
            ).expandingTildeInPath)
        let data: Data
        if FileManager.default.fileExists(atPath: url.path) {
            data = try Data(contentsOf: url)
        } else {
            data = Data(Self.embeddedFlashNextConfig.utf8)
        }
        let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data)
        XCTAssertEqual(config.modelType, "qwen4_exp")
        let text = config.textConfig
        XCTAssertEqual(text.modelType, "qwen4_exp_text")
        XCTAssertEqual(text.hiddenLayers, 48)
        XCTAssertEqual(text.hiddenSize, 2560)
        XCTAssertEqual(text.attentionHeads, 24)
        XCTAssertEqual(text.kvHeads, 2)
        XCTAssertEqual(text.headDim, 256)
        XCTAssertEqual(text.numExperts, 512)
        XCTAssertEqual(text.numExpertsPerTok, 10)
        XCTAssertEqual(text.pleLayerIds, [2])
        XCTAssertEqual(text.ngramSize, 3)
        XCTAssertEqual(text.headsPerNgram, 8)
        XCTAssertEqual(text.splitNgramParts, 128)
        XCTAssertEqual(text.outputGateType, "sigmoid")
        XCTAssertEqual(text.seed, Qwen4ExpNGramGeometry.defaultSeed)
        XCTAssertEqual(text.qsaLayerCount, 12)
        XCTAssertEqual(text.cbv2LayerKinds.count, 12)
        XCTAssertEqual(
            Set(text.cbv2LayerKinds.map(\.extraStorageBytesPerToken)),
            [Qwen4ExpPrefillMemory.qsaSidecarBytesPerToken(indexerHeadDim: 128)])
        XCTAssertEqual(
            text.cbv2LayerKinds.reduce(0) {
                $0 + 2 * $1.kvHeads * $1.headDim * 2 + $1.extraStorageBytesPerToken
            },
            Qwen4ExpPrefillMemory.flashNext.kvBytesPerToken)
        XCTAssertEqual(
            text.cbv2LayerKinds.compactMap(\.modelLayerIndex),
            Array(stride(from: 3, to: 48, by: 4)))
        XCTAssertTrue(text.cbv2Capabilities.supportsMTP)
        XCTAssertFalse(text.cbv2Capabilities.supportsCompactRecurrentMTPReplay)
        XCTAssertFalse(text.cbv2Capabilities.supportsPrefixReuse)
        let recurrent = text.cbv2RecurrentStateSpec()
        let gdn = recurrent.layers.filter { $0.modelLayerIndex < Qwen4ExpNGramGeometry.pleRecurrentLayerBase }
        let ple = recurrent.layers.filter { $0.modelLayerIndex >= Qwen4ExpNGramGeometry.pleRecurrentLayerBase }
        XCTAssertEqual(gdn.count, 36)
        XCTAssertEqual(ple.count, 1)
        XCTAssertEqual(ple[0].modelLayerIndex, Qwen4ExpNGramGeometry.recurrentLayerIndex(1))
        XCTAssertEqual(ple[0].convShape, [1, 9, 10_240])
        XCTAssertEqual(ple[0].ssmShape, [1, 1, 1, 2])
    }

    func testSanitizeDropsVisionMTPAndMmapPLEShards() {
        let keys = [
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight",
            "language_model.model.layers.1.ple.ple_embedding.layer_multipliers",
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets",
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
            "language_model.model.layers.1.ple.key_proj.weight",
            "vision_tower.patch_embed.proj.weight",
            "mtp.fc_embedding.weight",
            "language_model.lm_head.weight",
        ]
        let kept = keys.filter { !Qwen4ExpWeightSanitizer.shouldDrop($0, mmapPLE: true) }
        XCTAssertEqual(
            Set(kept),
            [
                "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
                "language_model.model.layers.1.ple.key_proj.weight",
                "language_model.lm_head.weight",
            ])
        XCTAssertFalse(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight",
                mmapPLE: false))
    }

    func testNGramGeometryMatchesFusionDefaults() {
        XCTAssertEqual(Qwen4ExpNGramGeometry.nthPrimeAfter(start: 19_999_999, count: 1), 20_000_003)
        let multipliers = Qwen4ExpNGramGeometry.layerMultipliers(
            unigramVocabSize: 248_320, ngramSize: 3, pleLayerIndex: 0, seed: 1234)
        XCTAssertEqual(multipliers.count, 3)
        XCTAssertTrue(multipliers.allSatisfy { $0 % 2 != 0 })
        let sizes = Qwen4ExpNGramGeometry.headVocabSizes(
            ngramHeads: 16, pleLayerIndex: 0, base: 20_000_000)
        XCTAssertEqual(sizes.count, 16)
        XCTAssertEqual(sizes[0], 20_000_003)
        XCTAssertTrue(sizes.allSatisfy(Qwen4ExpNGramGeometry.isPrime))
        XCTAssertEqual(Qwen4ExpNGramGeometry.floorMod(-5, 3), 1)
        XCTAssertEqual(Qwen4ExpNGramGeometry.paddedVocabSize(total: 100, divisor: 128), 128)
    }

    func testHostNGramShiftRightResetsAtEOS() {
        let eos = 248_044
        let tokens = [1, 2, eos, 4, 5]
        XCTAssertEqual(Qwen4ExpNGramIDs.shiftRightIgnoreEOS(tokens: tokens, shift: 0, eos: eos), tokens)
        XCTAssertEqual(
            Qwen4ExpNGramIDs.shiftRightIgnoreEOS(tokens: tokens, shift: 1, eos: eos),
            [eos, 1, 2, eos, 4])
        XCTAssertEqual(
            Qwen4ExpNGramIDs.shiftRightIgnoreEOS(tokens: tokens, shift: 2, eos: eos),
            [eos, eos, 1, eos, eos])
    }

    func testHostNGramIdsStayInPaddedRange() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: Data(Self.embeddedFlashNextConfig.utf8))
        let tables = Qwen4ExpNGramTables(config.textConfig, pleIndex: 0)
        let eos = tables.eosTokenId
        let history = [eos, eos, 10, 20, 30]
        let ids = Qwen4ExpNGramIDs.ids(history: history, inputWidth: 3, tables: tables)
        XCTAssertEqual(ids.count, 3)
        XCTAssertTrue(ids.allSatisfy { $0.count == tables.ngramHeads })
        XCTAssertTrue(ids.joined().allSatisfy { $0 >= 0 && $0 < tables.paddedVocabSize })
        let heads = Set((0 ..< tables.ngramHeads).map { tables.shardIndex(for: ids[0][$0]) })
        XCTAssertFalse(heads.isEmpty)
    }

    func testPLEGatherEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpPLEGather.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpPLEGather.isEnabled(environment: [Qwen4ExpPLEGather.envFlag: "1"]))
        XCTAssertFalse(
            Qwen4ExpPLEGather.isEnabled(environment: [Qwen4ExpPLEGather.envFlag: "0"]))
        XCTAssertFalse(
            Qwen4ExpPLEGather.isEnabled(environment: [Qwen4ExpPLEGather.envFlag: "off"]))
    }

    func testPLEGatherUniqueInversePreservesDuplicatesAndOrder() {
        let empty = Qwen4ExpPLEGather.uniqueInverse([])
        XCTAssertEqual(empty.unique, [])
        XCTAssertEqual(empty.inverse, [])
        let (unique, inverse) = Qwen4ExpPLEGather.uniqueInverse([5, 1, 5, 2, 1, 5])
        XCTAssertEqual(unique, [5, 1, 2])
        XCTAssertEqual(inverse, [0, 1, 0, 2, 1, 0])
        XCTAssertEqual(unique[Int(inverse[0])], 5)
        XCTAssertEqual(unique[Int(inverse[3])], 2)
    }

    func testBlockedSeqGDNEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpGDNBlockedSeq.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpGDNBlockedSeq.isEnabled(environment: [
                Qwen4ExpGDNBlockedSeq.envFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpGDNBlockedSeq.isEnabled(environment: [
                Qwen4ExpGDNBlockedSeq.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpGDNBlockedSeq.isEnabled(environment: [
                Qwen4ExpGDNBlockedSeq.envFlag: "off"
            ]))
    }

    func testWeightedUnsortEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4WeightedExpertUnsort.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4WeightedExpertUnsort.isEnabled(environment: [
                Qwen4WeightedExpertUnsort.envFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4WeightedExpertUnsort.isEnabled(environment: [
                Qwen4WeightedExpertUnsort.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4WeightedExpertUnsort.isEnabled(environment: [
                Qwen4WeightedExpertUnsort.envFlag: "off"
            ]))
    }

    func testBlockedSeqGDNGeometryIsFusionContract() {
        XCTAssertTrue(
            Qwen4ExpGDNBlockedSeq.matchesGeometry(
                T: 64, keyHeadDim: 128, valueHeadDim: 128))
        XCTAssertTrue(
            Qwen4ExpGDNBlockedSeq.shouldDispatch(
                T: 512, keyHeadDim: 128, valueHeadDim: 128, hasMask: false,
                environment: [:]))
        XCTAssertFalse(
            Qwen4ExpGDNBlockedSeq.matchesGeometry(
                T: 63, keyHeadDim: 128, valueHeadDim: 128))
        XCTAssertFalse(
            Qwen4ExpGDNBlockedSeq.matchesGeometry(
                T: 512, keyHeadDim: 64, valueHeadDim: 128))
        XCTAssertFalse(
            Qwen4ExpGDNBlockedSeq.matchesGeometry(
                T: 512, keyHeadDim: 128, valueHeadDim: 16))
        XCTAssertFalse(
            Qwen4ExpGDNBlockedSeq.shouldDispatch(
                T: 512, keyHeadDim: 128, valueHeadDim: 128, hasMask: true,
                environment: [:]))
        XCTAssertFalse(
            Qwen4ExpGDNBlockedSeq.shouldDispatch(
                T: 512, keyHeadDim: 128, valueHeadDim: 128, hasMask: false,
                environment: [Qwen4ExpGDNBlockedSeq.envFlag: "0"]))
    }

    func testAffineQMMEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpAffineQMM.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpAffineQMM.isEnabled(environment: [
                Qwen4ExpAffineQMM.envFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpAffineQMM.isEnabled(environment: [
                Qwen4ExpAffineQMM.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpAffineQMM.isEnabled(environment: [
                Qwen4ExpAffineQMM.envFlag: "off"
            ]))
    }

    func testAffineQMMGeometryMatchesFusionVariant8() {
        XCTAssertEqual(Qwen4ExpAffineQMM.blockN(for: 2560), 64)
        XCTAssertEqual(Qwen4ExpAffineQMM.blockN(for: 640), 64)
        XCTAssertEqual(Qwen4ExpAffineQMM.blockN(for: 10240), 64)
        XCTAssertEqual(Qwen4ExpAffineQMM.blockN(for: 6144), 64)
        XCTAssertEqual(Qwen4ExpAffineQMM.blockN(for: 16480), 32)
        XCTAssertNil(Qwen4ExpAffineQMM.blockN(for: 48))
        XCTAssertEqual(Qwen4ExpAffineQMM.padTarget(for: 4), 32)
        XCTAssertNil(Qwen4ExpAffineQMM.padTarget(for: 48))
        XCTAssertNil(Qwen4ExpAffineQMM.padTarget(for: 1))
        XCTAssertFalse(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 2560, outputDim: 48, bits: 5, groupSize: 128,
                environment: [:]))
        XCTAssertNil(Qwen4ExpAffineQMM.blockN(for: 4))
        XCTAssertEqual(Qwen4ExpAffineQMM.blockN(for: 512), 64)
        XCTAssertTrue(Qwen4ExpAffineQMM.injectPadEnabled(environment: [:]))
        XCTAssertFalse(
            Qwen4ExpAffineQMM.injectPadEnabled(environment: [
                Qwen4ExpAffineQMM.injectPadEnvFlag: "0"
            ]))
        XCTAssertTrue(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 10240, outputDim: 4, bits: 5, groupSize: 64,
                environment: [:]))
        XCTAssertFalse(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 10240, outputDim: 4, bits: 5, groupSize: 64,
                environment: [Qwen4ExpAffineQMM.injectPadEnvFlag: "0"]))
        XCTAssertEqual(Qwen4ExpAffineQMM.tokenFloor(bits: 4, environment: [:]), 2048)
        XCTAssertEqual(Qwen4ExpAffineQMM.tokenFloor(bits: 5, environment: [:]), 2048)
        XCTAssertEqual(Qwen4ExpAffineQMM.tokenFloor(bits: 6, environment: [:]), 2048)
        XCTAssertEqual(Qwen4ExpAffineQMM.tokenFloor(bits: 8, environment: [:]), 16384)
        XCTAssertEqual(Qwen4ExpAffineQMM.tileM(bits: 4, environment: [:]), 64)
        XCTAssertEqual(Qwen4ExpAffineQMM.tileM(bits: 8, environment: [:]), 64)
        XCTAssertEqual(
            Qwen4ExpAffineQMM.tileM(
                bits: 8, environment: [Qwen4ExpAffineQMM.q8BlockMEnv: "128"]),
            128)
        XCTAssertEqual(
            Qwen4ExpAffineQMM.tileM(
                bits: 5, environment: [Qwen4ExpAffineQMM.q8BlockMEnv: "128"]),
            64)
        XCTAssertEqual(
            Qwen4ExpAffineQMM.tokenFloor(
                bits: 8, environment: [Qwen4ExpAffineQMM.q8BlockMEnv: "128"]),
            8192)
        XCTAssertEqual(
            Qwen4ExpAffineQMM.tokenFloor(
                bits: 8,
                environment: [
                    Qwen4ExpAffineQMM.q8BlockMEnv: "128",
                    Qwen4ExpAffineQMM.q8MinTokensEnv: "16384",
                ]),
            16384)
        XCTAssertEqual(
            Qwen4ExpAffineQMM.tokenFloor(
                bits: 8, environment: [Qwen4ExpAffineQMM.q8MinTokensEnv: "8192"]),
            8192)
        XCTAssertTrue(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 2560, outputDim: 640, bits: 8, groupSize: 128,
                environment: [Qwen4ExpAffineQMM.q8BlockMEnv: "128"]))
        XCTAssertFalse(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 4096, inputDim: 2560, outputDim: 640, bits: 8, groupSize: 128,
                environment: [Qwen4ExpAffineQMM.q8BlockMEnv: "128"]))
        XCTAssertTrue(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 2560, outputDim: 512, bits: 4, groupSize: 64,
                environment: [:]))
        XCTAssertTrue(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 2560, outputDim: 2560, bits: 4, groupSize: 64,
                environment: [:]))
        XCTAssertTrue(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 2560, outputDim: 16480, bits: 6, groupSize: 64,
                environment: [:]))
        XCTAssertFalse(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 2560, outputDim: 2560, bits: 8, groupSize: 128,
                environment: [:]))
        XCTAssertTrue(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 8192, inputDim: 2560, outputDim: 640, bits: 8, groupSize: 128,
                environment: [Qwen4ExpAffineQMM.q8MinTokensEnv: "8192"]))
        XCTAssertTrue(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 16384, inputDim: 2560, outputDim: 640, bits: 8, groupSize: 128,
                environment: [:]))
        XCTAssertFalse(
            Qwen4ExpAffineQMM.matchesGeometry(
                tokens: 100, inputDim: 2560, outputDim: 2560, bits: 4, groupSize: 64,
                environment: [:]))
    }

    func testBf16HiddenEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpActivation.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpActivation.isEnabled(environment: [
                Qwen4ExpActivation.envFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpActivation.isEnabled(environment: [
                Qwen4ExpActivation.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpActivation.isEnabled(environment: [
                Qwen4ExpActivation.envFlag: "off"
            ]))
        XCTAssertEqual(Qwen4ExpActivation.dtype, .bfloat16)
        XCTAssertEqual(Qwen4ExpGDNBlockedSeq.blockT(for: Qwen4ExpActivation.dtype), 32)
    }

    func testGatherQMMEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpGatherQMM.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpGatherQMM.isEnabled(environment: [
                Qwen4ExpGatherQMM.envFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpGatherQMM.isEnabled(environment: [
                Qwen4ExpGatherQMM.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpGatherQMM.isEnabled(environment: [
                Qwen4ExpGatherQMM.envFlag: "off"
            ]))
    }

    func testGDNStackedVerifyEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpGDNStackedVerify.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpGDNStackedVerify.isEnabled(environment: [
                Qwen4ExpGDNStackedVerify.envFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpGDNStackedVerify.isEnabled(environment: [
                Qwen4ExpGDNStackedVerify.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpGDNStackedVerify.isEnabled(environment: [
                Qwen4ExpGDNStackedVerify.envFlag: "off"
            ]))
    }

    func testAffineQMVEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpAffineQMV.isEnabled(environment: [:]))
        XCTAssertFalse(
            Qwen4ExpAffineQMV.isEnabled(environment: [
                Qwen4ExpAffineQMV.envFlag: "0"
            ]))
        XCTAssertTrue(
            Qwen4ExpAffineQMV.matchesDecodeGeometry(
                tokens: 1, inputDim: 2560, outputDim: 2560, bits: 4, groupSize: 64))
        XCTAssertTrue(
            Qwen4ExpAffineQMV.matchesDecodeGeometry(
                tokens: 8, inputDim: 640, outputDim: 2560, bits: 4, groupSize: 64))
        // N tails are eligible (block_inject N=4, shared_expert_gate N=1):
        // the kernel guards rows past N, so verify columns and decode share
        // the exact path for those banks too.
        XCTAssertTrue(
            Qwen4ExpAffineQMV.matchesDecodeGeometry(
                tokens: 1, inputDim: 2560, outputDim: 4, bits: 4, groupSize: 64))
        XCTAssertTrue(
            Qwen4ExpAffineQMV.matchesDecodeGeometry(
                tokens: 5, inputDim: 2560, outputDim: 1, bits: 8, groupSize: 64))
        XCTAssertFalse(
            Qwen4ExpAffineQMV.matchesDecodeGeometry(
                tokens: 32, inputDim: 2560, outputDim: 2560, bits: 4, groupSize: 64))
    }

    func testGatherQMMGeometryMatchesFlashNextMoE() {
        XCTAssertTrue(
            Qwen4ExpGatherQMM.matchesGeometry(
                assignments: 81920, inputDim: 2560, outputDim: 1280, experts: 512))
        XCTAssertTrue(
            Qwen4ExpGatherQMM.matchesGeometry(
                assignments: 81920, inputDim: 2560, outputDim: 640, experts: 512))
        XCTAssertTrue(
            Qwen4ExpGatherQMM.matchesGeometry(
                assignments: 81920, inputDim: 640, outputDim: 2560, experts: 512))
        XCTAssertFalse(
            Qwen4ExpGatherQMM.matchesGeometry(
                assignments: 81920, inputDim: 2048, outputDim: 1024, experts: 256))
        XCTAssertFalse(
            Qwen4ExpGatherQMM.matchesGeometry(
                assignments: 1000, inputDim: 2560, outputDim: 1280, experts: 512))
        XCTAssertFalse(
            Qwen4ExpGatherQMM.matchesGeometry(
                assignments: 8192, inputDim: 2560, outputDim: 1280, experts: 128))
    }

    static let embeddedFlashNextConfig = """
        {
          "model_type": "qwen4_exp",
          "image_token_id": 248056,
          "video_token_id": 248057,
          "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "output_gate_type": "sigmoid",
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "shared_expert_intermediate_size": 640,
            "moe_intermediate_size": 640,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "eos_token_id": 248044,
            "rope_parameters": {
              "type": "default",
              "mrope_section": [11, 11, 10],
              "rope_theta": 10000000,
              "partial_rotary_factor": 0.25
            }
          }
        }
        """
}
