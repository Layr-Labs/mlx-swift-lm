import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite struct GLM5NextTextStageTests {
        private func inventory(layers: Int = 8) throws -> GLM5NextWeightInventory {
            let c = try GLM5NextInventoryFixture.configuration(layers: layers)
            return try .validate(
                configuration: c,
                tensors: GLM5NextWeightInventory.expectedTensors(
                    configuration: c, storage: .float32, shard: "tiny.safetensors"),
                storage: .float32)
        }

        private func tensor(_ entry: GLM5NextWeightInventory.Entry) -> MLXArray {
            let name = entry.tensor.name
            let count = entry.tensor.shape.reduce(1, *)
            let values: [Float]
            if name.hasSuffix("_scale") || name.contains("norm.weight") {
                values = Array(repeating: 1, count: count)
            } else if name.hasSuffix("A_log") {
                values = Array(repeating: -0.7, count: count)
            } else if name.hasSuffix("dt_bias") {
                values = Array(repeating: -0.2, count: count)
            } else {
                let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 13 }
                values = (0 ..< count).map { Float(($0 + salt) % 11 - 5) * 0.025 }
            }
            return MLXArray(values, entry.tensor.shape)
        }

        private func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        private func close(_ actual: MLXArray, _ expected: MLXArray) {
            #expect(actual.shape == expected.shape)
            // The split materializes the same operations; this tolerance is
            // the full text model's existing FP32 cached-forward tolerance.
            #expect(abs(actual - expected).max().item(Float.self) < 0.0001)
        }

        @Test(arguments: Array(1 ..< 8))
        func everyCutPreservesFourStreamsGlobalIndicesAndCachedLogits(cut: Int) throws {
            try Device.withDefaultDevice(.cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let policy = GLM5NextExecutionPolicy(
                    stepEvaluation: cut == 5 ? .earlyAsync : .deferred)
                let full = try GLM5NextTextModel(
                    configuration: c, inventory: inv,
                    tensor: { tensor(try #require(inv.entries[$0])) }, executionPolicy: policy)
                var firstReads = Set<String>()
                var lastReads = Set<String>()
                let firstPlan = try GLM5NextTextStagePlan(configuration: c, layers: 0 ..< cut)
                let lastPlan = try GLM5NextTextStagePlan(
                    configuration: c, layers: cut ..< c.hiddenLayers)
                let first = try GLM5NextTextStage(
                    plan: firstPlan, inventory: inv,
                    tensor: {
                        firstReads.insert($0)
                        return tensor(try #require(inv.entries[$0]))
                    }, executionPolicy: policy)
                let last = try GLM5NextTextStage(
                    plan: lastPlan, inventory: inv,
                    tensor: {
                        lastReads.insert($0)
                        return tensor(try #require(inv.entries[$0]))
                    }, executionPolicy: policy)
                #expect(firstReads == (try firstPlan.tensorNames(in: inv)))
                #expect(lastReads == (try lastPlan.tensorNames(in: inv)))
                #expect(firstReads.isDisjoint(with: lastReads))
                #expect(
                    firstReads.union(lastReads)
                        == Set(inv.entries(in: .text).map { $0.tensor.name }))
                #expect(first.norm == nil && first.head == nil && last.embedding == nil)
                for (index, layer) in zip(lastPlan.layers, last.layers) {
                    if let sparse = layer.sparseAttention {
                        #expect(sparse.modelLayerIndex == index)
                    }
                }
                let fullCache = full.newTextCache()
                let firstCache = first.newCache()
                let lastCache = last.newCache()
                let chunks: [(MLXArray, MLXArray)] = [
                    (
                        MLXArray([Int32(2), 3, 4, 0, 0, 5], [2, 3]),
                        MLXArray([true, true, true, false, false, true], [2, 3])
                    ),
                    (MLXArray([Int32(6), 7, 8, 9], [2, 2]), MLXArray.ones([2, 2], dtype: .bool)),
                    (MLXArray([Int32(10), 11], [2, 1]), MLXArray.ones([2, 1], dtype: .bool)),
                ]
                for (tokens, validity) in chunks {
                    let expected = try full.prepareForward(
                        tokens: tokens, cache: fullCache, validity: validity
                    )
                    .commit(afterEvaluating: fence)
                    let before = firstCache.offset
                    let candidate = try first.prepareForward(
                        tokens: tokens, cache: firstCache, validity: validity)
                    guard case .streams(let boundary) = candidate.output else {
                        Issue.record("intermediate stage returned a final readout")
                        return
                    }
                    #expect(boundary.shape == [2, tokens.dim(1), c.hcMult, c.hiddenSize])
                    #expect(firstCache.offset == before)
                    let tail = try last.prepareForward(
                        streams: boundary, cache: lastCache, validity: validity)
                    var actual: GLM5NextTextOutput?
                    try candidate.commit { roots in
                        try fence(roots)
                        if case .final(let output) = try tail.commit(afterEvaluating: fence) {
                            actual = output
                        }
                    }
                    let output = try #require(actual)
                    close(output.hidden, expected.hidden)
                    close(output.logits, expected.logits)
                    #expect(
                        firstCache.offset == fullCache.offset
                            && lastCache.offset == fullCache.offset)
                    for (split, original) in zip(
                        firstCache.layers + lastCache.layers, fullCache.layers)
                    {
                        #expect(split.offset == original.offset)
                        #expect(split.state.count == original.state.count)
                        for (a, b) in zip(split.state, original.state) {
                            close(a.asType(.float32), b.asType(.float32))
                        }
                    }
                }
            }
        }

        @Test(arguments: [false, true])
        func admittedReadoutPreservesHiddenCacheAndFollowingDecode(gpu: Bool) throws {
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let first = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 0 ..< 4), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                let last = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 4 ..< 8), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                let firstCache = first.newCache()
                let fullCache = last.newCache()
                let selectedCache = last.newCache()
                let chunks: [(MLXArray, MLXArray, GLM5NextTextStageReadout)] = [
                    (
                        MLXArray([Int32(2), 3, 4, 5, 6, 0], [2, 3]),
                        MLXArray([true, true, true, true, true, false], [2, 3]), .none
                    ),
                    (
                        MLXArray([Int32(7), 8, 9, 0], [2, 2]),
                        MLXArray([true, true, true, false], [2, 2]), .lastToken
                    ),
                    (
                        MLXArray([Int32(10), 11], [2, 1]),
                        MLXArray.ones([2, 1], dtype: .bool), .lastToken
                    ),
                ]
                for (tokens, validity, readout) in chunks {
                    guard
                        case .streams(let boundary) = try first.prepareForward(
                            tokens: tokens, cache: firstCache, validity: validity
                        ).commit(afterEvaluating: fence)
                    else {
                        Issue.record("first stage lost its four-stream boundary")
                        return
                    }
                    guard
                        case .final(let expected) = try last.prepareForward(
                            streams: boundary, cache: fullCache, validity: validity
                        ).commit(afterEvaluating: fence)
                    else {
                        Issue.record("default must retain the full-logits API")
                        return
                    }
                    #expect(expected.logits.shape == [2, tokens.dim(1), c.vocabularySize])
                    let before = selectedCache.offset
                    // Even a hidden-only transaction must fence before publishing.
                    let failed = try last.prepareForward(
                        streams: boundary, cache: selectedCache, validity: validity,
                        readout: readout)
                    #expect(throws: GLM5NextLoadError.self) {
                        try failed.commit { roots in
                            try fence(roots)
                            throw GLM5NextLoadError.invalid("readout acknowledgement failed")
                        }
                    }
                    #expect(selectedCache.offset == before)
                    let selected = try last.prepareForward(
                        streams: boundary, cache: selectedCache, validity: validity,
                        readout: readout)
                    try selected.commit { roots in
                        // A full [B,T,V] projection must not be an evaluation root.
                        if readout == .none || tokens.dim(1) > 1 {
                            #expect(!roots.contains { $0.shape == expected.logits.shape })
                        }
                        try fence(roots)
                    }
                    switch selected.output {
                    case .hidden(let hidden):
                        #expect(readout == .none)
                        close(hidden, expected.hidden)
                    case .final(let value):
                        #expect(readout == .lastToken)
                        close(value.hidden, expected.hidden)
                        // Last physical column, including a masked column, is
                        // the existing stage API's selection semantics.
                        close(value.logits, expected.logits[0..., (-1)..., 0...])
                    case .streams:
                        Issue.record("final stage returned an intermediate boundary")
                    }
                    #expect(selectedCache.offset == fullCache.offset)
                    for (selected, full) in zip(selectedCache.layers, fullCache.layers) {
                        #expect(selected.state.count == full.state.count)
                        for (a, b) in zip(selected.state, full.state) {
                            close(a.asType(.float32), b.asType(.float32))
                        }
                    }
                }
            }
        }

        @Test func tokenAndEmbeddingEntryPointsHonorReadout() throws {
            try Device.withDefaultDevice(.gpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let stage = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 0 ..< 8), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                let tokens = MLXArray([Int32(2), 3, 4], [1, 3])
                let embedding = try #require(stage.embedding)
                let embedded = embedding(tokens)
                guard
                    case .final(let expected) = try stage.prepareForward(
                        tokens: tokens, cache: stage.newCache()
                    ).commit(afterEvaluating: fence),
                    case .final(let selected) = try stage.prepareForward(
                        tokens: tokens, cache: stage.newCache(), readout: .lastToken
                    ).commit(afterEvaluating: fence),
                    case .hidden(let hidden) = try stage.prepareForward(
                        embeddings: embedded,
                        cache: stage.newCache(), readout: .none
                    ).commit(afterEvaluating: fence)
                else {
                    Issue.record("token or embedding entry point ignored its readout")
                    return
                }
                close(selected.logits, expected.logits[0..., (-1)..., 0...])
                close(selected.hidden, expected.hidden)
                close(hidden, expected.hidden)
            }
        }

        @Test func failedAcknowledgementDiscardAndStaleCandidatesDoNotPublish() throws {
            try Device.withDefaultDevice(.cpu) {
                let inv = try inventory()
                let plan = try GLM5NextTextStagePlan(
                    configuration: inv.configuration.textConfig, layers: 0 ..< 4)
                let stage = try GLM5NextTextStage(plan: plan, inventory: inv) {
                    tensor(try #require(inv.entries[$0]))
                }
                let cache = stage.newCache()
                let tokens = MLXArray([Int32(2), 3], [1, 2])
                let failed = try stage.prepareForward(tokens: tokens, cache: cache)
                #expect(throws: GLM5NextLoadError.self) {
                    try failed.commit { roots in
                        try fence(roots)
                        throw GLM5NextLoadError.invalid("downstream acknowledgement failed")
                    }
                }
                #expect(cache.layers.allSatisfy { $0.offset == 0 && $0.state.isEmpty })
                #expect(throws: GLM5NextLoadError.self) {
                    try failed.commit(afterEvaluating: fence)
                }
                let discarded = try stage.prepareForward(tokens: tokens, cache: cache)
                discarded.discard()
                #expect(throws: GLM5NextLoadError.self) {
                    try discarded.commit(afterEvaluating: fence)
                }
                let stale = try stage.prepareForward(tokens: tokens, cache: cache)
                let winner = try stage.prepareForward(tokens: tokens, cache: cache)
                #expect(throws: GLM5NextLoadError.self) {
                    try stale.commit { roots in
                        try fence(roots)
                        try winner.commit(afterEvaluating: fence)
                    }
                }
                #expect(cache.offset == 2)
                let snapshot = cache.copy()
                let preserved = cache.layers.map {
                    $0.state.map { $0.asType(.float32).asArray(Float.self) }
                }
                let failedWithHistory = try stage.prepareForward(tokens: tokens, cache: cache)
                #expect(throws: GLM5NextLoadError.self) {
                    try failedWithHistory.commit { roots in
                        try fence(roots)
                        throw GLM5NextLoadError.invalid("downstream failed after cached prefill")
                    }
                }
                #expect(cache.offset == 2)
                #expect(
                    cache.layers.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }
                        == preserved)
                _ = try stage.prepareForward(tokens: tokens, cache: cache).commit(
                    afterEvaluating: fence)
                #expect(snapshot.offset == 2 && cache.offset == 4)
                let rewindCopy = cache.copy()
                try rewindCopy.truncate(to: 2)
                #expect(rewindCopy.offset == 2 && cache.offset == 4)
                #expect(
                    rewindCopy.layers.map {
                        $0.state.map { $0.asType(.float32).asArray(Float.self) }
                    } == preserved)
                #expect(throws: GLM5NextLoadError.self) { try cache.truncate(to: 3) }
                #expect(cache.layers.allSatisfy { $0.offset == 4 })
                try cache.truncate(to: 2)
                #expect(cache.offset == 2)
            }
        }

        @Test func rejectsCollapsedBoundaryForeignCacheAndWrongEntryPoint() throws {
            try Device.withDefaultDevice(.cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let plan = try GLM5NextTextStagePlan(configuration: c, layers: 4 ..< 8)
                func make() throws -> GLM5NextTextStage {
                    try .init(plan: plan, inventory: inv) { tensor(try #require(inv.entries[$0])) }
                }
                let stage = try make()
                let cache = stage.newCache()
                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(
                        streams: MLXArray.zeros([1, 1, c.hiddenSize]), cache: cache)
                }
                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(tokens: MLXArray([Int32(2)], [1, 1]), cache: cache)
                }
                let other = try make()
                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(
                        streams: MLXArray.zeros([1, 1, c.hcMult, c.hiddenSize]),
                        cache: other.newCache())
                }
                #expect(cache.offset == 0)
            }
        }

        @Test(arguments: [false, true])
        func artifactStageReadsOnlyOwnedPayloadAndAccountsForDuplicateEmbedding(q4: Bool) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture(q4: q4)
                let loader = try fixture.loader()
                let c = loader.configuration.textConfig
                let plan = try GLM5NextTextStagePlan(
                    configuration: c, layers: 2 ..< c.hiddenLayers, loadMTPEmbedding: true)
                let ordinary = try GLM5NextTextStagePlan(
                    configuration: c, layers: 2 ..< c.hiddenLayers)
                let expected = try plan.tensorNames(in: loader.inventory)
                var reads: [String: Int] = [:]
                let stage = try loader.loadTextStage(plan: plan) { name, bytes in
                    reads[name] = bytes
                }
                #expect(Set(reads.keys) == expected)
                #expect(reads.values.reduce(0, +) == (try plan.tensorBytes(in: loader.inventory)))
                #expect(stage.embedding != nil && stage.head != nil)
                let duplicate = loader.inventory.entries.values.filter {
                    $0.tensor.name.hasPrefix("model.language_model.embed_tokens.")
                }.reduce(0) { $0 + $1.bytes }
                #expect(
                    try plan.tensorBytes(in: loader.inventory)
                        - ordinary.tensorBytes(in: loader.inventory) == duplicate)
                #expect(
                    Set(loader.unresolved(in: .text))
                        == Set(loader.inventory.entries(in: .text).map { $0.tensor.name })
                        .subtracting(expected))
                #expect(
                    loader.unresolved(in: .mtp).count == loader.inventory.entries(in: .mtp).count)
                #expect(throws: GLM5NextArtifactError.self) { try loader.loadTextStage(plan: plan) }
            }
        }

        @Test func artifactMTPRequiresItsOwnFinalStageAndReadsOnlyNextN() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let loader = try fixture.loader()
                let plan = try GLM5NextTextStagePlan(
                    configuration: loader.configuration.textConfig, layers: 2 ..< 4,
                    loadMTPEmbedding: true)
                let stage = try loader.loadTextStage(plan: plan)
                let otherLoader = try fixture.loader()
                let otherStage = try otherLoader.loadTextStage(plan: plan)
                var reads: Set<String> = []
                #expect(throws: GLM5NextArtifactError.self) {
                    try loader.loadMTP(stage: otherStage) { name, _ in reads.insert(name) }
                }
                #expect(reads.isEmpty)
                let mtp = try loader.loadMTP(stage: stage) { name, _ in reads.insert(name) }
                #expect(mtp.targetIdentity == ObjectIdentifier(stage))
                #expect(reads == Set(loader.inventory.entries(in: .mtp).map { $0.tensor.name }))
                #expect(loader.unresolved(in: .mtp).isEmpty)
                #expect(
                    loader.unresolved(in: .vision).count
                        == loader.inventory.entries(in: .vision).count)
                #expect(!loader.unresolved(in: .text).isEmpty)
            }
        }

        @Test func admissionFailurePrecedesAnySelectedPayloadAllocation() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let loader = try fixture.loader()
                let plan = try GLM5NextTextStagePlan(
                    configuration: loader.configuration.textConfig, layers: 0 ..< 1)
                var admissions = 0
                #expect(throws: GLM5NextLoadError.self) {
                    try loader.loadTextStage(plan: plan) { _, _ in
                        admissions += 1
                        throw GLM5NextLoadError.invalid("load headroom exhausted")
                    }
                }
                #expect(admissions == 1)
                #expect(
                    loader.unresolved(in: .text).count == loader.inventory.entries(in: .text).count)
            }
        }
    }
}
