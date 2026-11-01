import Foundation
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
    private func model(fp8: Bool, seed: UInt64, dtype: DType = .bfloat16) throws -> NemotronHModel {
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
                        outputs: outputs, inputs: inputs, magnitude: 64, dtype: dtype)
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
        // Bitwise eager parity holds for exact per-row products; tiled prefill
        // (the default) is covered with its bound in ModelOptGraphParityTests.
        model.setModelOptAccumulation(.rowwise)
        model.setModelOptFusedSSD(false)
        return model
    }

    private func tokens(_ count: Int, seed: Int) -> MLXArray {
        let ids: [Int32] = (0 ..< count).map { (i: Int) -> Int32 in Int32((i * 37 + seed * 11) % 64)
        }
        return MLXArray(ids).reshaped(1, count)
    }

    @Test func nvfp4TrunkGraphMatchesEagerBitwiseAcrossSortedPrefill() throws {
        let model = try model(fp8: false, seed: 21)
        let gates = model.namedModules().compactMap { ($0.1 as? NemotronHMoE)?.gate }
        func fused() -> [Bool] { gates.map { $0.fusedSuperSelection } }
        #expect(fused() == [true, true])
        #expect(model.setModelOptExecution(.eagerReference) == 15)
        #expect(fused() == [false, false])
        #expect(model.setModelOptExecution(.graph) == 15)
        #expect(fused() == [true, true])
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

    @Test func prefillProfileIsResultNeutralAndCoversEveryStage() throws {
        let model = try model(fp8: true, seed: 33)
        model.setModelOptAccumulation(.tiled)
        model.setModelOptFusedSSD(true)
        // 48 tokens: fused SSD (>= 32 tokens) and sorted routing (144 rows).
        let input = tokens(48, seed: 5)
        let plain = model(input, cache: nil)
        eval(plain)
        // A scoped recorder: other suites' concurrent forwards stay unprofiled.
        let recorder = NemotronSuperProfileRecorder()
        // Two calls: each layer's n-th call is the n-th chunk.
        let profiled = NemotronSuperPrefillProfile.$recorder.withValue(recorder) {
            (0 ..< 2).map { _ in
                let output = model(input, cache: nil)
                eval(output)
                return output
            }
        }
        let report = recorder.report()
        let counts = recorder.routingCounts()
        let perCall = recorder.routingCountsPerCall()
        let seconds = recorder.secondsPerCall()
        for output in profiled { #expect(modelOptBitwiseEqual(plain, output)) }
        #expect(perCall.map { $0.map { $0.reduce(0, +) } } == [[144, 144], [144, 144]])
        #expect(seconds["MoE expert fc2"]?.map(\.count) == [2, 2])
        #expect(seconds["Mamba SSD"]?.map(\.count) == [2])
        #expect(seconds["Mamba glue (between stages)"]?.map(\.count) == [2])
        #expect(seconds["attention o_proj"]?.map(\.count) == [2])
        let json = try #require(
            try JSONSerialization.jsonObject(with: recorder.json()) as? [String: Any])
        #expect(
            Set(json.keys) == [
                "stage_order", "seconds_per_call", "seconds_per_call_index", "routing_per_call",
            ])
        #expect((json["routing_per_call"] as? [[[Int]]]) == perCall)
        for stage in [
            "MoE router/top-k", "MoE fc1 latent", "MoE sort/gather", "MoE expert fc1+relu2",
            "MoE expert fc2", "MoE unsort", "MoE combine", "MoE fc2 latent", "MoE shared expert",
            "Mamba in_proj", "Mamba conv", "Mamba SSD", "Mamba norm/gate", "Mamba out_proj",
            "attention q/k/v", "attention KV update + attend", "attention o_proj",
            "MoE total", "Mamba total", "attention total", "MoE layer 1 (2 calls, 288 routed rows)",
            "ms per call index", "max per call",
        ] {
            #expect(report.contains(stage), "missing \(stage)")
        }
        #expect(counts.map { $0.reduce(0, +) } == [288, 288])
    }

    @Test func prefillProfileMemoryIsResultNeutralAndCoversEveryStage() throws {
        let model = try model(fp8: true, seed: 33)
        model.setModelOptAccumulation(.tiled)
        model.setModelOptFusedSSD(true)
        let input = tokens(48, seed: 5)
        let plain = model(input, cache: nil)
        eval(plain)
        let recorder = NemotronSuperProfileRecorder(recordsMemory: true)
        let profiled = NemotronSuperPrefillProfile.$recorder.withValue(recorder) {
            let output = model(input, cache: nil)
            eval(output)
            return output
        }
        #expect(modelOptBitwiseEqual(plain, profiled))
        let memory = recorder.memoryPerCall()
        // Every timed stage has one memory record per layer call.
        let seconds = recorder.secondsPerCall()
        for (stage, layers) in seconds where !stage.contains("glue") {
            #expect(memory[stage]?.map(\.count) == layers.map(\.count), "\(stage)")
        }
        // Each stage allocates at least its output: the routed fc1 rows
        // (144 rows of bfloat16 outputs) and the attention projections.
        let fc1 = memory["MoE expert fc1+relu2"]?.joined().map(\.transientBytes).max() ?? 0
        #expect(fc1 > 0)
        #expect(memory.values.joined().joined().allSatisfy { $0.transientBytes >= 0 })
        let json = try #require(
            try JSONSerialization.jsonObject(with: recorder.json()) as? [String: Any])
        #expect(json["memory_per_call"] != nil)
        #expect(recorder.report().contains("memory MoE expert fc1+relu2: max transient"))
    }

    /// Decode through the standard KV/Mamba caches: graph steps (fused
    /// router, combine and Mamba step kernels) against the eager reference.
    @Test(arguments: [false, true])
    func cachedDecodeMatchesEagerBitwise(fp8: Bool) throws {
        let model = try model(fp8: fp8, seed: 12)
        func run() throws -> [MLXArray] {
            let cache = try model.newCache(parameters: nil)
            var outputs = [model(tokens(20, seed: 2), cache: cache)]
            eval(outputs + cache.flatMap(\.state))
            for token: Int32 in [5, 33, 61] {
                outputs.append(model(MLXArray([token]).reshaped(1, 1), cache: cache))
                eval(outputs + cache.flatMap(\.state))
            }
            return outputs + cache.flatMap(\.state)
        }
        let graph = try run()
        model.setModelOptExecution(.eagerReference)
        let eager = try run()
        #expect(graph.count == eager.count)
        for (a, b) in zip(graph, eager) { #expect(modelOptBitwiseEqual(a, b)) }
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

    @Test(arguments: [DType.bfloat16, .float32])
    func mixedFP8TrunkIsBitwise(dtype: DType) throws {
        let model = try model(fp8: true, seed: 13, dtype: dtype)
        let input = tokens(24, seed: 5)
        let graph = model(input, cache: nil)
        model.setModelOptExecution(.eagerReference)
        let eager = model(input, cache: nil)
        #expect(!isNaN(eager).any().item(Bool.self))
        #expect(modelOptBitwiseEqual(graph, eager))
    }

    /// Batched decode of one latent-MoE layer: with row-exact batches every
    /// token row equals its one-row forward bit for bit (router, latent
    /// projections, routed and shared experts, weighted sum).
    @Test(arguments: [2, 5, 16])
    func rowExactBatchedMoEMatchesOneRowForward(rows: Int) throws {
        var configuration = self.configuration
        configuration.moeLatentSize = 96
        let moe = NemotronHMoE(configuration)
        var fixtures = ModelOptGraphFixtures(seed: UInt64(rows) + 40)
        var dense: [String: MLXArray] = [:]
        for (name, value) in moe.parameters().flattened() {
            dense[name] =
                name.hasSuffix("e_score_correction_bias")
                ? value
                : MLXArray(fixtures.values(value.size, magnitude: 0.08)).reshaped(value.shape)
                    .asType(.bfloat16)
        }
        try moe.update(parameters: .unflattened(dense), verify: .all)
        let experts = configuration.nRoutedExperts
        try moe.update(
            modules: .unflattened([
                (
                    "switch_mlp.fc1",
                    try fixtures.expertBank(
                        experts: experts, outputs: 96, inputs: 96, magnitude: 64, dtype: .bfloat16)
                        as Module
                ),
                (
                    "switch_mlp.fc2",
                    try fixtures.expertBank(
                        experts: experts, outputs: 96, inputs: 96, magnitude: 64, dtype: .bfloat16)
                ),
                (
                    "shared_experts.up_proj",
                    try fixtures.nvfp4Projection(
                        outputs: 128, inputs: 128, magnitude: 64, dtype: .bfloat16)
                ),
                (
                    "shared_experts.down_proj",
                    try fixtures.nvfp4Projection(
                        outputs: 128, inputs: 128, magnitude: 64, dtype: .bfloat16)
                ),
            ]))
        moe.rowExactBatches = true
        moe.gate.rowExactBatches = true
        let x = MLXArray(fixtures.values(rows * 128, magnitude: 1)).reshaped(rows, 1, 128)
            .asType(.bfloat16)
        let batched = moe(x)
        for r in 0 ..< rows {
            #expect(modelOptBitwiseEqual(batched[r ..< r + 1], moe(x[r ..< r + 1])))
        }
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
        #expect(NemotronHModel.prefillActivationBoundBytes(tokens: limit) <= budget)
        model.setModelOptExecution(.eagerReference)
        #expect(model.modelOptGraphPrefillWorkspaceBytes(tokens: 1) == nil)
        #expect(try #require(model.calibratedPrefillTokenLimit) <= 32)
    }
}
