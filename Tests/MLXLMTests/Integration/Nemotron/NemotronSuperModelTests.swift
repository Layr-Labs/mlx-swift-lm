import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.tags(.integration))
struct NemotronSuperModelTests {
    private func component() throws -> NemotronSuperModel {
        let configuration = NemotronHConfiguration(
            vocabSize: 32, hiddenSize: 64,
            numHiddenLayers: 2, numAttentionHeads: 2, numKeyValueHeads: 1,
            mambaNumHeads: 4, mambaHeadDim: 16, ssmStateSize: 16,
            convKernel: 4, nGroups: 2, intermediateSize: 128,
            moeIntermediateSize: 64, moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4, numExpertsPerTok: 2, hybridOverridePattern: "M*", headDim: 64)
        let dimensions = NemotronRadioDimensions(
            width: 2, layers: 1, heads: 1,
            channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)
        let images = NemotronSuperImageComposition(
            tower: NemotronRadio(dimensions),
            projector: NemotronSuperVisionProjector(
                visionWidth: 2, hiddenWidth: 3, outputWidth: 64),
            width: 64, imageTokenID: 18)
        return try NemotronSuperModel(
            languageModel: NemotronHModel(configuration), imageComposition: images)
    }

    private struct Result {
        let last: MLXArray, decoded: MLXArray, conv: MLXArray, ssm: MLXArray
        let keys: MLXArray, values: MLXArray
        let offset: Int
    }

    private func run(
        _ model: NemotronSuperModel, prepared: NemotronSuperModel.PreparedInput,
        chunks: [Int], useWrapper: Bool, textOracle: Bool = false, rejectInvalid: Bool = false
    ) throws -> Result {
        let kinds = model.cbv2LayerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 16 << 20, dtype: .float32, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 64))
        let rows = try backend.makeSequenceState(
            layerKinds: kinds,
            promptLength: prepared.prompt.count, maxLength: 64)
        defer { backend.release(rows) }
        let storage = backend.makeLayerCaches()
        for i in storage.indices { storage[i].setRows([rows[i]!]) }
        let caches: [KVCache] = storage.map { $0 as KVCache }
        let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        var start = 0
        var last = MLXArray.zeros([1])
        for count in chunks {
            let end = start + count
            let binding = try state.bind()
            let requirement: CBv2PrefillRequirement =
                end == prepared.prompt.count
                ? .lastPositionLogits : .evaluationOnly
            if useWrapper {
                last = try model.prefill(
                    prepared, range: start ..< end, cache: caches,
                    recurrentState: [binding], requirement: requirement)
            } else {
                let tokens = MLXArray(Array(prepared.prompt[start ..< end])).reshaped(1, count)
                last = model.languageModel.cbv2RecurrentPrefill(
                    tokens,
                    inputEmbedding: textOracle
                        ? nil : prepared.embeddings[0..., start ..< end, 0...],
                    cache: caches, recurrentState: [binding], positionIds: nil,
                    requirement: requirement)
            }
            eval([last] + (try binding.evaluate()))
            try binding.commit()
            start = end
        }
        #expect(start == prepared.prompt.count)
        let next = try state.bind()
        if rejectInvalid {
            let before = rows[0]!.snapshot()
            let keys = before.keys + 0
            let values = before.values + 0
            let committed = try #require(state.state(modelLayerIndex: 0))
            let conv = try #require(committed.conv) + 0
            let ssm = try #require(committed.ssm) + 0
            eval(keys, values, conv, ssm)
            #expect(throws: NemotronSuperModel.Failure.self) {
                try model.decode(token: -1, cache: caches, recurrentState: [next])
            }
            #expect(throws: NemotronSuperModel.Failure.self) {
                try model.prefill(
                    prepared, range: 0 ..< (prepared.prompt.count + 1),
                    cache: caches, recurrentState: [next], requirement: .lastPositionLogits)
            }
            #expect(throws: NemotronSuperImageComposition.Failure.self) {
                try model.prepare(prompt: [1, 18], imageFeatures: [])
            }
            let after = rows[0]!.snapshot()
            let recurrent = try #require(state.state(modelLayerIndex: 0))
            #expect(after.offset == before.offset)
            #expect(arrayEqual(after.keys, keys).item(Bool.self))
            #expect(arrayEqual(after.values, values).item(Bool.self))
            let afterConv = try #require(recurrent.conv)
            let afterSSM = try #require(recurrent.ssm)
            #expect(arrayEqual(afterConv, conv).item(Bool.self))
            #expect(arrayEqual(afterSSM, ssm).item(Bool.self))
        }
        let decoded: MLXArray
        if useWrapper {
            decoded = try model.decode(token: 7, cache: caches, recurrentState: [next])
        } else {
            decoded = model.languageModel.cbv2Forward(
                MLXArray([Int32(7)]).reshaped(1, 1),
                caches: caches, recurrentState: [next])
        }
        eval([decoded] + (try next.evaluate()))
        try next.commit()
        let recurrent = try #require(state.state(modelLayerIndex: 0))
        let snapshot = rows[0]!.snapshot()
        let keys = snapshot.keys + 0
        let values = snapshot.values + 0
        eval(keys, values)
        return Result(
            last: last, decoded: decoded, conv: try #require(recurrent.conv),
            ssm: try #require(recurrent.ssm), keys: keys, values: values, offset: snapshot.offset)
    }

    private func compare(_ a: Result, _ b: Result) {
        #expect(a.offset == b.offset)
        for (x, y) in [
            (a.last, b.last), (a.decoded, b.decoded), (a.conv, b.conv),
            (a.ssm, b.ssm), (a.keys, b.keys), (a.values, b.values),
        ] {
            #expect(x.shape == y.shape)
            #expect(abs(x - y).max().item(Float.self) < 0.0001)
        }
    }

    @Test func textPrefillAndDecodeMatchNativeTrunk() throws {
        let model = try component()
        let prepared = try model.prepare(prompt: [1, 2, 3, 4])
        compare(
            try run(model, prepared: prepared, chunks: [4], useWrapper: true, rejectInvalid: true),
            try run(model, prepared: prepared, chunks: [4], useWrapper: false, textOracle: true))
    }

    @Test func imageGraphPrefillAndSplitSpanMatchDirectEmbeddingTrunk() throws {
        let model = try component()
        let pixels = MLXArray((0 ..< 32).map { Float($0) / 32 }).reshaped(1, 1, 4, 8)
        let prepared = try model.prepare(prompt: [1, 18, 18, 2], normalizedImages: [pixels])
        let features = try model.imageComposition.imageFeatures(normalizedPixels: pixels)
        #expect(arrayEqual(prepared.embeddings[0..., 1 ..< 3, 0...], features).item(Bool.self))
        let oracle = try run(model, prepared: prepared, chunks: [4], useWrapper: false)
        compare(oracle, try run(model, prepared: prepared, chunks: [4], useWrapper: true))
        compare(oracle, try run(model, prepared: prepared, chunks: [2, 2], useWrapper: true))
    }

    @Test func validatesBeforeLookupAndRejectsWrongModel() throws {
        let model = try component()
        let invalidPrompts: [[Int32]] = [[], [-1], [32], [Int32.max]]
        for prompt in invalidPrompts {
            #expect(throws: NemotronSuperModel.Failure.self) { try model.prepare(prompt: prompt) }
        }
        #expect(throws: NemotronSuperImageComposition.Failure.self) {
            try model.prepare(prompt: [1, 18], imageFeatures: [])
        }
        let other = try component()
        let prepared = try other.prepare(prompt: [1, 2])
        #expect(throws: NemotronSuperModel.Failure.self) {
            try model.prefill(
                prepared, cache: [], recurrentState: [], requirement: .lastPositionLogits)
        }
    }

    // Pinned language fields only: parsing these tests creates no model or arrays.
    private var strictLanguageFixture: Data {
        Data(
            #"{"attention_bias":false,"chunk_size":128,"conv_kernel":4,"dtype":"bfloat16","head_dim":128,"hidden_size":4096,"intermediate_size":2688,"layer_norm_epsilon":1e-05,"layers_block_type":["mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","attention","moe","mamba","moe","mamba","moe","mamba","moe","mamba","moe"],"mamba_head_dim":64,"mamba_hidden_act":"silu","mamba_num_heads":128,"mamba_proj_bias":false,"mamba_ssm_cache_dtype":"float32","mlp_bias":false,"mlp_hidden_act":"relu2","model_type":"nemotron_h","moe_intermediate_size":2688,"moe_latent_size":1024,"moe_shared_expert_intermediate_size":5376,"mtp_layers_block_type":["attention","moe"],"n_group":1,"n_groups":8,"n_routed_experts":512,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":32,"num_experts_per_tok":22,"num_key_value_heads":2,"num_nextn_predict_layers":1,"routed_scaling_factor":5.0,"ssm_state_size":128,"tie_word_embeddings":false,"time_step_limit":[0.0,{"__float__":"Infinity"}],"topk_group":1,"use_bias":false,"use_conv_bias":true,"vocab_size":131072}"#
                .utf8)
    }

    @Test func strictLanguageDerivesOnlyMissingLayerCountWithoutAllocation() throws {
        let original = strictLanguageFixture
        let language = try NemotronSuperModel.validatedLanguageConfiguration(original)
        #expect(language.numHiddenLayers == 88)
        #expect(language.moeLatentSize == 1024)
        #expect(language.nRoutedExperts == 512 && language.numExpertsPerTok == 22)
        let raw = try #require(JSONSerialization.jsonObject(with: original) as? [String: Any])
        #expect(raw["num_hidden_layers"] == nil)
        var explicit = raw
        explicit["num_hidden_layers"] = 88
        #expect(
            try NemotronSuperModel.validatedLanguageConfiguration(
                JSONSerialization.data(withJSONObject: explicit)
            ).numHiddenLayers == 88)
    }

    @Test func strictLanguageRejectsConflictsAndDefaultsWithoutAllocation() throws {
        let raw = try #require(
            JSONSerialization.jsonObject(with: strictLanguageFixture) as? [String: Any])
        let changes: [(String, Any)] = [
            ("num_hidden_layers", 87), ("moe_latent_size", 512), ("dtype", "float32"),
            ("num_attention_heads", 16), ("num_experts_per_tok", 8),
        ]
        for (key, value) in changes {
            var changed = raw
            changed[key] = value
            #expect(throws: (any Error).self) {
                try NemotronSuperModel.validatedLanguageConfiguration(
                    JSONSerialization.data(withJSONObject: changed))
            }
        }
        for key in ["moe_latent_size", "dtype", "norm_topk_prob", "mlp_bias", "layers_block_type"] {
            var missing = raw
            missing.removeValue(forKey: key)
            #expect(throws: (any Error).self) {
                try NemotronSuperModel.validatedLanguageConfiguration(
                    JSONSerialization.data(withJSONObject: missing))
            }
        }
        var reordered = raw
        var blocks = try #require(raw["layers_block_type"] as? [String])
        blocks.swapAt(0, 1)
        reordered["layers_block_type"] = blocks
        #expect(throws: (any Error).self) {
            try NemotronSuperModel.validatedLanguageConfiguration(
                JSONSerialization.data(withJSONObject: reordered))
        }
    }
}
