import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A tiny latent-MoE hybrid trunk with installed ModelOpt modules, executed on
/// the default graph path and on the eager reference. Synthetic weights only.
@Suite(.tags(.integration))
struct NemotronSuperGraphExecutionTests {
    private let configuration = NemotronHConfiguration(
        vocabSize: 64, hiddenSize: 128, numHiddenLayers: 4,
        numAttentionHeads: 2, numKeyValueHeads: 1, mambaNumHeads: 4, mambaHeadDim: 32,
        ssmStateSize: 16, convKernel: 4, nGroups: 2, intermediateSize: 128,
        moeIntermediateSize: 96, moeSharedExpertIntermediateSize: 128,
        nRoutedExperts: 8, numExpertsPerTok: 3, hybridOverridePattern: "ME*E", headDim: 64,
        nSharedExperts: 1, routedScalingFactor: 5, moeLatentSize: 64)

    /// FP8 in every slot the checkpoint quantizes as FP8 when `fp8` is set;
    /// otherwise NVFP4 everywhere, so the whole trunk can be compared bitwise.
    private func model(
        fp8: Bool, seed: UInt64, dtype: DType = .bfloat16, powerOfTwoFP8: Bool = false
    ) throws -> NemotronHModel {
        let model = NemotronHModel(configuration)
        var fixtures = ModelOptGraphFixtures(seed: seed)
        var dense: [String: MLXArray] = [:]
        for (name, value) in model.parameters().flattened() {
            if name.hasSuffix("A_log") || name.hasSuffix("e_score_correction_bias") {
                dense[name] = value
                continue
            }
            let magnitude: Float = name.contains("norm") ? 0.5 : 0.08
            var random = MLXArray(fixtures.values(value.size, magnitude: magnitude))
                .reshaped(value.shape)
            if name.contains("norm") { random = random + 1 }
            dense[name] = random.asType(dtype)
        }
        try model.update(parameters: .unflattened(dense), verify: .all)
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var projections: [String: ModelOptCalibratedLinear] = [:]
        var experts: [String: ModelOptNVFP4SwitchLinear] = [:]
        for (path, module) in leaves.sorted(by: { $0.key < $1.key }) {
            let mixer = path.split(separator: ".").dropLast().last
            if let linear = module as? Linear, type(of: linear) == Linear.self {
                let (outputs, inputs) = linear.shape
                let isFP8Slot =
                    path == "lm_head"
                    || ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"].contains {
                        path.hasSuffix($0)
                    }
                let isShared = mixer == "shared_experts"
                guard isFP8Slot || isShared else { continue }
                projections[path] =
                    fp8 && isFP8Slot
                    ? try fixtures.fp8Projection(
                        outputs: outputs, inputs: inputs, magnitude: 64, dtype: dtype,
                        powerOfTwoScale: powerOfTwoFP8)
                    : try fixtures.nvfp4Projection(
                        outputs: outputs, inputs: inputs, magnitude: 64, dtype: dtype)
            }
            if module is SwitchLinear,
                let shape = Dictionary(uniqueKeysWithValues: module.parameters().flattened())[
                    "weight"]?.shape
            {
                experts[path] = try fixtures.expertBank(
                    experts: shape[0], outputs: shape[1], inputs: shape[2], magnitude: 64,
                    dtype: dtype)
            }
        }
        try model.installModelOptProjections(projections)
        try model.installModelOptExperts(experts)
        return model
    }

    private func tokens(_ count: Int, seed: Int) -> MLXArray {
        MLXArray((0 ..< count).map { Int32(($0 * 37 + seed * 11) % 64) }).reshaped(1, count)
    }

    @Test func nvfp4TrunkGraphMatchesEagerBitwiseAcrossSortedPrefill() throws {
        let model = try model(fp8: false, seed: 21)
        #expect(model.setModelOptExecution(.graph) == 15)
        // 24 tokens x 3 experts > 64 assignments: NemotronHSwitchMLP sorts rows.
        for count in [1, 5, 24] {
            let input = tokens(count, seed: count)
            let ledger = ModelOptGraphValidity()
            let graph = ModelOptGraphValidity.$current.withValue(ledger) {
                model(input, cache: nil)
            }
            eval([graph] + ledger.evaluationRoots)
            try ledger.check()
            model.setModelOptExecution(.eagerReference)
            let eager = model(input, cache: nil)
            model.setModelOptExecution(.graph)
            #expect(graph.shape == [1, count, 64])
            #expect(!isNaN(eager).any().item(Bool.self))
            #expect(modelOptBitwiseEqual(graph, eager))
        }
    }

    @Test func cbv2PrefillAndDecodeMatchEagerBitwise() throws {
        let model = try model(fp8: false, seed: 4)
        func run() throws -> [MLXArray] {
            let backend = try PagedKVBackend(
                layerKinds: model.cbv2LayerKinds,
                config: .init(
                    capacityBytes: 16 << 20, dtype: .bfloat16, maxPrefillChunk: 32,
                    nominalMaxSequenceLength: 64))
            let rows = try backend.makeSequenceState(
                layerKinds: model.cbv2LayerKinds, promptLength: 20, maxLength: 64)
            defer { backend.release(rows) }
            let storage = backend.makeLayerCaches()
            for i in storage.indices { storage[i].setRows([rows[i]!]) }
            let caches: [KVCache] = storage.map { $0 as KVCache }
            let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
            var outputs: [MLXArray] = []
            let prompt = tokens(20, seed: 3)
            for range in [0 ..< 12, 12 ..< 20] {
                let binding = try state.bind()
                let logits = model.cbv2RecurrentPrefill(
                    prompt[0..., range], inputEmbedding: nil, cache: caches,
                    recurrentState: [binding], positionIds: nil,
                    requirement: range.upperBound == 20 ? .lastPositionLogits : .evaluationOnly)
                eval([logits] + (try binding.evaluate()))
                try binding.commit()
                outputs.append(logits)
            }
            for token: Int32 in [7, 19, 42] {
                let binding = try state.bind()
                let logits = model.cbv2Forward(
                    MLXArray([token]).reshaped(1, 1), caches: caches, recurrentState: [binding])
                eval([logits] + (try binding.evaluate()))
                try binding.commit()
                outputs.append(logits)
            }
            return outputs
        }
        let graph = try run()
        model.setModelOptExecution(.eagerReference)
        let eager = try run()
        for (a, b) in zip(graph, eager) { #expect(modelOptBitwiseEqual(a, b)) }
    }

    /// Packed FP8 projections differ from the eager reference only by rounding each
    /// scaled weight to the weight dtype (bounded per projection in
    /// ModelOptGraphParityTests). In float32 that is 2^-24 relative, so the
    /// trunk stays within 1e-4 of the largest logit; so does bfloat16 with
    /// power-of-two weight scales, where that rounding vanishes. With general
    /// scales the bfloat16 differences can flip downstream NVFP4 activation
    /// codes and expert choices, as any two valid FP8 products could; that
    /// deviation is recorded, not bounded.
    @Test(arguments: [(DType.float32, false), (.bfloat16, true), (.bfloat16, false)])
    func mixedFP8TrunkGraphStaysWithinWeightRounding(dtype: DType, powerOfTwo: Bool) throws {
        do {
            let model = try model(fp8: true, seed: 9, dtype: dtype, powerOfTwoFP8: powerOfTwo)
            model.setModelOptFP8Product(.packed)
            let input = tokens(24, seed: 2)
            let graph = model(input, cache: nil).asType(.float32)
            model.setModelOptExecution(.eagerReference)
            let eager = model(input, cache: nil).asType(.float32)
            let difference = abs(graph - eager).max().item(Float.self)
            let scale = abs(eager).max().item(Float.self)
            print(
                "mixed FP8 trunk \(dtype) power-of-two \(powerOfTwo): max abs \(difference), "
                    + "max |logit| \(scale)")
            #expect(!isNaN(graph).any().item(Bool.self) && !isNaN(eager).any().item(Bool.self))
            if dtype == .float32 || powerOfTwo { #expect(difference <= 1e-4 * scale) }
        }
    }

    @Test func mixedTrunkWithExactFP8IsBitwise() throws {
        let model = try model(fp8: true, seed: 13)
        let input = tokens(24, seed: 5)
        let graph = model(input, cache: nil)
        model.setModelOptExecution(.eagerReference)
        let eager = model(input, cache: nil)
        #expect(!isNaN(eager).any().item(Bool.self))
        #expect(modelOptBitwiseEqual(graph, eager))
    }

    @Test func graphPrefillLimitAdmitsProductionChunks() throws {
        let model = try model(fp8: true, seed: 1)
        let limit = try #require(model.calibratedPrefillTokenLimit)
        let budget = NemotronHModel.modelOptGraphPrefillWorkspaceLimit
        let used = try #require(model.modelOptGraphPrefillWorkspaceBytes(tokens: limit))
        #expect(used <= budget)
        if limit < NemotronHModel.modelOptGraphPrefillTokenCap {
            #expect(
                try #require(model.modelOptGraphPrefillWorkspaceBytes(tokens: limit + 1)) > budget)
        }
        #expect(limit == NemotronHModel.modelOptGraphPrefillTokenCap)
        model.setModelOptExecution(.eagerReference)
        #expect(model.modelOptGraphPrefillWorkspaceBytes(tokens: 1) == nil)
        #expect(try #require(model.calibratedPrefillTokenLimit) <= 32)
    }
}
