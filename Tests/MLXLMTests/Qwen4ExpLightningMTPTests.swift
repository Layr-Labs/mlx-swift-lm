import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Lightning MTP (`Qwen4ExpMTPModule`) — Fusion fuse + in-tree probe, not
/// Qwen 3.5 concat-FC. Tiny random modules only; never load the 99 GB card.
@Suite
struct Qwen4ExpLightningMTPTests {
    private func tinyTextConfig() -> Qwen4ExpTextConfiguration {
        var args = Qwen4ExpTextConfiguration()
        args.hiddenSize = 32
        args.hiddenLayers = 1
        args.attentionHeads = 2
        args.kvHeads = 1
        args.headDim = 16
        args.linearNumValueHeads = 2
        args.linearNumKeyHeads = 1
        args.linearKeyHeadDim = 8
        args.linearValueHeadDim = 8
        args.vocabularySize = 64
        args.maxPositionEmbeddings = 128
        args.fullAttentionInterval = 1
        args.layerTypes = ["qwen_sparse_attention"]
        args.hcCount = 2
        args.hcLowrank = 8
        args.pleLayerIds = []
        args.pleEmbedDim = 32
        args.indexerNHeads = 2
        args.indexerKVHeads = 1
        args.indexerHeadDim = 8
        args.indexerBudget = 16
        args.indexerCompressRatio = 4
        args.numExperts = 1
        args.numExpertsPerTok = 1
        args.sharedExpertIntermediateSize = 16
        args.moeIntermediateSize = 16
        args.mropeSection = [2, 1, 1]
        args.partialRotaryFactor = 0.25
        args.mtpNumHiddenLayers = 1
        return args
    }

    private func writeFusionDirectory(
        prefix: String = "mtp.",
        layers: Int = 1,
        includeTensors: Bool = true
    ) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-lightning-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data(
            """
            {
              "model_type": "qwen4_exp",
              "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "hc_count": 2,
                "hc_lowrank": 8,
                "num_experts": 1,
                "num_experts_per_tok": 1,
                "vocab_size": 64,
                "mtp_num_hidden_layers": \(layers)
              },
              "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
              }
            }
            """.utf8
        ).write(to: root.appendingPathComponent("config.json"))
        let key = includeTensors ? "\(prefix)fc_embedding.weight" : "language_model.lm_head.weight"
        try JSONSerialization.data(withJSONObject: [
            "weight_map": [key: "model-00001-of-00001.safetensors"]
        ]).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        return root
    }

    @Test("Fusion in-tree mtp.* metadata loads without mtplx_mtp.included")
    func fusionInTreeMetadata() throws {
        let directory = try writeFusionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadata = try Qwen4ExpInlineMTPAssistant.loadMetadata(from: directory)
        #expect(metadata.prefix == "mtp.")
        #expect(metadata.blockSize == 3)
        #expect(metadata.textConfiguration.mtpNumHiddenLayers == 1)
        #expect(metadata.textConfiguration.hcCount == 2)
        #expect(metadata.quantization != nil)
    }

    @Test("language_model.mtp.* prefix is accepted")
    func languageModelPrefix() throws {
        let directory = try writeFusionDirectory(prefix: "language_model.mtp.")
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadata = try Qwen4ExpInlineMTPAssistant.loadMetadata(from: directory)
        #expect(metadata.prefix == "language_model.mtp.")
    }

    @Test("missing in-tree tensors and mtplx_mtp is rejected")
    func missingLightningDeclaration() throws {
        let directory = try writeFusionDirectory(includeTensors: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: Qwen4ExpInlineMTPError.self) {
            _ = try Qwen4ExpInlineMTPAssistant.loadMetadata(from: directory)
        }
    }

    @Test("mtp_num_hidden_layers outside 1...4 is rejected")
    func layerCountRejected() throws {
        let directory = try writeFusionDirectory(layers: 0)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: Qwen4ExpInlineMTPError.self) {
            _ = try Qwen4ExpInlineMTPAssistant.loadMetadata(from: directory)
        }
    }

    @Test("fuseInputs is embed + per-stream fc_hidden, residual width hc·H")
    func fuseInputsShape() throws {
        let args = tinyTextConfig()
        let module = try Qwen4ExpMTPModule(args)
        let batch = 1
        let tokens = 3
        let embed = MLXArray.ones([batch, tokens, args.hiddenSize])
        let residual = MLXArray.ones([batch, tokens, args.hcCount * args.hiddenSize])
        let fused = module.fuseInputs(tokenEmbeddings: embed, hiddenStates: residual)
        #expect(fused.shape == residual.shape)
        let streamed = MLXArray.ones([batch, tokens, args.hcCount, args.hiddenSize])
        let from4D = module.fuseInputs(tokenEmbeddings: embed, hiddenStates: streamed)
        #expect(from4D.shape == residual.shape)
    }

    @Test("serial env forces serialTarget; default is rectangular")
    func verificationMode() {
        #expect(
            Qwen4ExpInlineMTPAssistant.resolvedVerificationMode(
                requested: .rectangular, forceSerialEnvironment: true)
                == .serialTarget)
        #expect(
            Qwen4ExpInlineMTPAssistant.resolvedVerificationMode(
                requested: nil, forceSerialEnvironment: false)
                == .rectangular)
        #expect(
            Qwen4ExpInlineMTPAssistant.resolvedVerificationMode(
                requested: .automatic, forceSerialEnvironment: false)
                == .automatic)
    }

    @Test("Qwen4 chain supports five but production defaults to four")
    func fiveDraftChainPolicy() throws {
        let key = "DARKBLOOM_QWEN_MTP_MAX_DRAFT"
        #expect(Qwen4ExpInlineMTPAssistant.maximumChainedDraftTokens == 5)
        #expect(Qwen4ExpInlineMTPAssistant.defaultMaximumDraftTokens == 4)
        #expect(
            Qwen4ExpInlineMTPAssistant.maximumChainedDraftTokens
                <= CBv2MTPConfig.testedMaxDraftTokens)
        #expect(
            Qwen4ExpInlineMTPAssistant.resolvedMaximumDraftTokens(
                forceDoubleForward: false, maxDraftOverride: nil) == 4)
        #expect(
            Qwen4ExpInlineMTPAssistant.resolvedMaximumDraftTokens(
                forceDoubleForward: false, maxDraftOverride: 4) == 4)
        #expect(
            Qwen4ExpInlineMTPAssistant.resolvedMaximumDraftTokens(
                forceDoubleForward: false, maxDraftOverride: 5) == 5)
        #expect(
            Qwen4ExpInlineMTPAssistant.resolvedMaximumDraftTokens(
                forceDoubleForward: true, maxDraftOverride: 5) == 1)
        #expect(
            Qwen4ExpInlineMTPAssistant.validatedMaxDraftOverride(
                environment: [key: " 5 "]) == 5)
        #expect(
            Qwen4ExpInlineMTPAssistant.validatedMaxDraftOverride(
                environment: [key: "4"]) == 4)
        for raw in ["", "0", "6", "five"] {
            #expect(
                Qwen4ExpInlineMTPAssistant.validatedMaxDraftOverride(
                    environment: [key: raw]) == nil)
        }

        let args = tinyTextConfig()
        let assistant = try Qwen4ExpInlineMTPAssistant(
            configuration: args, blockSize: 3,
            target: Qwen4ExpTextModel(args), verificationMode: .rectangular,
            skipColdPromptReplay: false)
        #expect(assistant.requestStateTokenAllocationPadding == 5)
    }

    @Test("Qwen4 assistant executes five drafts and rolls speculative cache back")
    func fiveDraftChainRollback() throws {
        let args = tinyTextConfig()
        let target = Qwen4ExpTextModel(args)
        let assistant = try Qwen4ExpInlineMTPAssistant(
            configuration: args, blockSize: 3, target: target,
            verificationMode: .rectangular, skipColdPromptReplay: false)
        let state = assistant.makeRequestState()
        let qwenState = try #require(
            state as? Qwen4ExpInlineMTPAssistant.RequestState)
        defer { assistant.releaseRequestState(state) }
        var token = MLXArray([Int32(7)]).reshaped([1, 1])
        var hidden = MLXArray.ones([1, 1, args.hcCount * args.hiddenSize])
        for _ in 0..<5 {
            let output = assistant.draftStep(
                tokens: token, hidden: hidden, shortlist: nil,
                requestState: state)
            token = output.tokens.reshaped([1, 1])
            hidden = output.hidden
        }
        eval([token, hidden] + assistant.evaluationTargets(for: state))
        #expect(state.stagedInputCount == 4)
        #expect(!qwenState.qwen4Sidecars.isEmpty)
        #expect(
            assistant.evaluationTargets(for: state).count
                >= qwenState.caches.flatMap { $0.innerState() }.count
                    + qwenState.qwen4Sidecars.count)
        #expect(
            state.materializedBytes
                >= qwenState.qwen4Sidecars.reduce(0) { $0 + $1.nbytes })
        assistant.finalizeRound(
            requestState: state, confirmedInputTokens: 1,
            committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
            committedTargetHidden: MLXArray.zeros(
                [1, 0, args.hcCount * args.hiddenSize]))
        let settled = try assistant.snapshotRequestState(state)
        #expect(settled.cacheLayers[0].offset == 1)
        #expect(settled.cacheLayers[0].qwen4Indexer?.tokenCount == 1)
        #expect(settled.committedInputCount == 1)
    }

    @Test("request-state bytes stay finite on the tiny geometry")
    func stateBytesFinite() {
        let args = tinyTextConfig()
        let bytes = Qwen4ExpInlineMTPAssistant.stateBytesPerToken(
            configuration: args, layerCount: 1, cacheElementBytes: 2, hiddenElementBytes: 2)
        #expect(bytes > 0)
        #expect(bytes < Int.max)
    }

    @Test("settled Lightning request state snapshots and restores exactly")
    func requestStateSnapshotRoundTrip() throws {
        let args = tinyTextConfig()
        let target = Qwen4ExpTextModel(args)
        let assistant = try Qwen4ExpInlineMTPAssistant(
            configuration: args,
            blockSize: 3,
            target: target,
            verificationMode: .rectangularExact,
            skipColdPromptReplay: false)
        let original = assistant.makeRequestState()
        assistant.observeCommittedTarget(
            CBv2MTPCommittedTargetObservation(
                tokens: MLXArray([Int32(5), 6]).reshaped([1, 2]),
                hidden: MLXArray(
                    (0 ..< 2 * args.hcCount * args.hiddenSize).map {
                        Float($0) / 100
                    },
                    [1, 2, args.hcCount * args.hiddenSize])),
            requestState: original)

        _ = assistant.draftStep(
            tokens: MLXArray([Int32(7)]).reshaped([1, 1]),
            hidden: MLXArray.ones([1, 1, args.hcCount * args.hiddenSize]),
            shortlist: nil,
            requestState: original)
        eval(assistant.evaluationTargets(for: original))
        assistant.finalizeRound(
            requestState: original,
            confirmedInputTokens: 1,
            committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
            committedTargetHidden: MLXArray.zeros(
                [1, 0, args.hcCount * args.hiddenSize]))

        let snapshot = try assistant.snapshotRequestState(original)
        eval(snapshot.arrays)
        #expect(snapshot.cacheLayers.count == 1)
        #expect(snapshot.cacheLayers[0].offset == 2)
        #expect(snapshot.cacheLayers[0].qwen4Indexer?.tokenCount == 2)
        #expect(snapshot.committedInputCount == 2)
        #expect(snapshot.logicalInputBase == 0)

        let malformed = Qwen4ExpMTPStateSnapshot(
            cacheLayers: snapshot.cacheLayers,
            backlogHidden: snapshot.backlogHidden,
            backlogTokens: snapshot.backlogTokens,
            targetHiddenFrontier: snapshot.targetHiddenFrontier,
            committedInputCount: snapshot.committedInputCount + 1,
            logicalInputBase: snapshot.logicalInputBase)
        #expect(throws: Qwen4ExpMTPStateError.self) {
            _ = try assistant.restoreRequestState(from: malformed)
        }

        let restored = try assistant.restoreRequestState(from: snapshot)
        #expect(restored.committedInputCount == original.committedInputCount)

        let nextToken = MLXArray([Int32(8)]).reshaped([1, 1])
        let nextHidden = MLXArray.ones([1, 1, args.hcCount * args.hiddenSize]) * 0.25
        let expected = assistant.draftStep(
            tokens: nextToken, hidden: nextHidden, shortlist: nil,
            requestState: original)
        let actual = assistant.draftStep(
            tokens: nextToken, hidden: nextHidden, shortlist: nil,
            requestState: restored)
        eval([expected.tokens, expected.hidden, actual.tokens, actual.hidden])
        #expect(expected.tokens.asArray(Int32.self) == actual.tokens.asArray(Int32.self))
        #expect(expected.hidden.asArray(Float.self) == actual.hidden.asArray(Float.self))
        assistant.discardRound(requestState: original)
        assistant.discardRound(requestState: restored)
        assistant.releaseRequestState(original)
        assistant.releaseRequestState(restored)
    }

    @Test("in-flight Lightning round cannot be snapshotted")
    func requestStateSnapshotRejectsInFlightRound() throws {
        let args = tinyTextConfig()
        let target = Qwen4ExpTextModel(args)
        let assistant = try Qwen4ExpInlineMTPAssistant(
            configuration: args,
            blockSize: 3,
            target: target,
            verificationMode: .rectangularExact)
        let state = assistant.makeRequestState()
        _ = assistant.draftStep(
            tokens: MLXArray([Int32(7)]).reshaped([1, 1]),
            hidden: MLXArray.ones([1, 1, args.hcCount * args.hiddenSize]),
            shortlist: nil,
            requestState: state)
        #expect(throws: Qwen4ExpMTPStateError.self) {
            _ = try assistant.snapshotRequestState(state)
        }
        assistant.discardRound(requestState: state)
        assistant.releaseRequestState(state)
    }

    @Test("a Qwen 3.5 target is rejected by the Lightning loader")
    func refusesQwen35Target() throws {
        let directory = try writeFusionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let qwen35 = Qwen35TextModel(
            try JSONDecoder().decode(
                Qwen35TextConfiguration.self,
                from: Data(
                    """
                    {
                      "model_type": "qwen3_5_text",
                      "hidden_size": 32,
                      "num_hidden_layers": 1,
                      "num_attention_heads": 2,
                      "num_key_value_heads": 1,
                      "head_dim": 16,
                      "linear_num_value_heads": 2,
                      "linear_num_key_heads": 1,
                      "linear_key_head_dim": 8,
                      "linear_value_head_dim": 8,
                      "linear_conv_kernel_dim": 4,
                      "vocab_size": 64,
                      "num_experts": 0,
                      "num_experts_per_tok": 0
                    }
                    """.utf8)))
        #expect(throws: Qwen4ExpInlineMTPError.self) {
            _ = try Qwen4ExpInlineMTPAssistant.load(from: directory, target: qwen35)
        }
    }

    @Test("official Q4 stacked experts.gate_up_proj loads as inline Lightning MTP")
    func officialStackedExpertsInlineLoad() throws {
        let args = tinyTextConfig()
        let target = Qwen4ExpTextModel(args)
        let module = try Qwen4ExpMTPModule(args)
        var exported: [String: MLXArray] = [:]
        for (key, value) in module.parameters().flattened() {
            var stripped = key
            stripped = stripped.replacingOccurrences(
                of: ".switch_mlp.gate_up_proj", with: ".experts.gate_up_proj")
            stripped = stripped.replacingOccurrences(
                of: ".switch_mlp.down_proj", with: ".experts.down_proj")
            exported["mtp." + stripped] = value
        }
        #expect(exported.keys.contains { $0.contains(".experts.gate_up_proj") })
        #expect(!exported.keys.contains { $0.contains(".switch_mlp.gate_up_proj") })

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "qwen4-official-mtp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Custom encode(to:) omits mtp_num_hidden_layers and indexer fields.
        var text = try JSONSerialization.jsonObject(with: JSONEncoder().encode(args)) as! [String: Any]
        text["mtp_num_hidden_layers"] = args.mtpNumHiddenLayers
        text["indexer_n_heads"] = args.indexerNHeads
        text["indexer_kv_heads"] = args.indexerKVHeads
        text["indexer_head_dim"] = args.indexerHeadDim
        text["indexer_budget"] = args.indexerBudget
        text["indexer_compress_ratio"] = args.indexerCompressRatio
        text["output_gate_type"] = args.outputGateType
        let root: [String: Any] = [
            "model_type": "qwen4_exp",
            "text_config": text,
        ]
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("config.json"))
        let shard = "model-00001-of-00001.safetensors"
        try save(arrays: exported, url: directory.appendingPathComponent(shard))
        try JSONSerialization.data(
            withJSONObject: [
                "weight_map": Dictionary(
                    uniqueKeysWithValues: exported.keys.map { ($0, shard) })
            ]
        ).write(to: directory.appendingPathComponent("model.safetensors.index.json"))

        let loaded = try Qwen4ExpInlineMTPAssistant.load(from: directory, target: target)
        #expect(loaded.blockSize == 3)
    }
}
