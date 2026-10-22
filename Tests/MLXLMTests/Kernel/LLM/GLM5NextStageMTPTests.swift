// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextStageMTPTests {
        private struct Fixture {
            let inventory: GLM5NextWeightInventory
            var configuration: GLM5NextTextConfiguration { inventory.configuration.textConfig }

            init(layers: Int = 4) throws {
                let c = try GLM5NextInventoryFixture.configuration(layers: layers)
                inventory = try .validate(
                    configuration: c,
                    tensors: GLM5NextWeightInventory.expectedTensors(
                        configuration: c, storage: .float32, shard: "tiny-stage-mtp.safetensors"),
                    storage: .float32)
            }

            func tensor(_ name: String) throws -> MLXArray {
                let entry = try #require(inventory.entries[name])
                let count = entry.tensor.shape.reduce(1, *)
                let values: [Float]
                if name.hasSuffix("_scale") || name.contains("norm.weight") {
                    values = Array(repeating: 1, count: count)
                } else if name.hasSuffix("A_log") {
                    values = Array(repeating: -0.7, count: count)
                } else if name.hasSuffix("dt_bias") {
                    values = Array(repeating: -0.2, count: count)
                } else if name.hasSuffix(".bias") {
                    values = Array(repeating: 0, count: count)
                } else {
                    let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 13 }
                    values = (0 ..< count).map { Float(($0 + salt) % 11 - 5) * 0.025 }
                }
                return MLXArray(values, entry.tensor.shape)
            }

            func stage(
                layers: Range<Int> = 3 ..< 4, loadMTPEmbedding: Bool = true,
                policy: GLM5NextExecutionPolicy = .reference
            ) throws -> GLM5NextTextStage {
                try .init(
                    plan: .init(
                        configuration: configuration, layers: layers,
                        loadMTPEmbedding: loadMTPEmbedding),
                    inventory: inventory, tensor: tensor, executionPolicy: policy)
            }

            func model() throws -> GLM5NextTextModel {
                try .init(configuration: configuration, inventory: inventory, tensor: tensor)
            }
        }

        private let invalidOwner = GLM5NextMTPError.invalid(
            "MTP target released or shared components replaced")

        private func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        private func close(_ actual: MLXArray, _ expected: MLXArray) {
            #expect(actual.shape == expected.shape)
            // The stage materializes the same FP32 target operations. Use the
            // existing tiny text model's cached-forward tolerance, not frozen bits.
            let error = abs(actual - expected).max().item(Float.self)
            #expect(error.isFinite && error < 0.0001)
        }

        @Test(arguments: [false, true])
        func sharesComponentsWithoutRetainingTheOwner(stageOwner: Bool) throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                var stage: GLM5NextTextStage?
                var model: GLM5NextTextModel?
                let mtp: GLM5NextMTP
                let embedding: Embedding
                let head: Linear
                var reads = Set<String>()
                var released = Set<String>()
                if stageOwner {
                    let policy = GLM5NextExecutionPolicy(sparseAttention: .absorbed)
                    stage = try f.stage(policy: policy)
                    mtp = try GLM5NextMTP(
                        stage: #require(stage),
                        tensor: {
                            reads.insert($0)
                            return try f.tensor($0)
                        },
                        releaseSourceTensors: { released.formUnion($0) })
                    embedding = try #require(stage?.embedding)
                    head = try #require(stage?.head)
                    #expect(mtp.targetIdentity == ObjectIdentifier(try #require(stage)))
                    #expect(mtp.executionPolicy == policy)
                } else {
                    model = try f.model()
                    mtp = try GLM5NextMTP(
                        target: #require(model),
                        tensor: {
                            reads.insert($0)
                            return try f.tensor($0)
                        },
                        releaseSourceTensors: { released.formUnion($0) })
                    embedding = try #require(model).model.embedTokens
                    head = try #require(model).lmHead
                    #expect(mtp.targetIdentity == ObjectIdentifier(try #require(model)))
                }
                let expected = Set(f.inventory.entries(in: .mtp).map { $0.tensor.name })
                #expect(reads == expected && released == expected)
                #expect(mtp.modules().contains { $0 === embedding })
                #expect(mtp.modules().contains { $0 === head })
                #expect(
                    !mtp.modules().contains { $0 is GLM5NextTextModel || $0 is GLM5NextTextStage })
                let state = try mtp.newRequestState()
                weak var weakStage = stage
                weak var weakModel = model
                stage = nil
                model = nil
                #expect(weakStage == nil && weakModel == nil)
                #expect(throws: invalidOwner) { try mtp.newRequestState() }
                #expect(throws: invalidOwner) { try mtp.reset(state: state) }
                #expect(throws: invalidOwner) { try mtp.validateExecutionTarget() }
                // Unloading the target must not prevent request cleanup.
                try mtp.release(state: state)
                #expect(state.committedInputCount == 0 && state.recycledHidden == nil)
            }
        }

        @Test func stateAndCheckpointBindingsAreUniqueEvenForTheSameStage() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let stage = try f.stage()
                let otherStage = try f.stage()
                let first = try GLM5NextMTP(stage: stage, tensor: f.tensor)
                let second = try GLM5NextMTP(stage: stage, tensor: f.tensor)
                let foreign = try GLM5NextMTP(stage: otherStage, tensor: f.tensor)
                #expect(first.targetIdentity == second.targetIdentity)
                #expect(first.targetIdentity != foreign.targetIdentity)
                let state = try first.newRequestState()
                let checkpoint = try first.capture(state: state)
                for other in [second, foreign] {
                    #expect(throws: GLM5NextMTPError.incompatibleState) {
                        try other.reset(state: state)
                    }
                    #expect(throws: GLM5NextMTPError.incompatibleState) {
                        try other.restore(checkpoint, into: other.newRequestState())
                    }
                }
                withExtendedLifetime((stage, otherStage)) {}
            }
        }

        @Test(arguments: [false, true])
        func refusesNonfinalOrMissingEmbeddingBeforeReadingMTP(isFinal: Bool) throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let stage = try f.stage(
                    layers: isFinal ? 3 ..< 4 : 0 ..< 3, loadMTPEmbedding: false)
                #expect(stage.plan.isFinal == isFinal)
                #expect((stage.embedding == nil) == isFinal)
                var reads = 0
                #expect(
                    throws: GLM5NextMTPError.invalid(
                        "MTP requires a final text stage with embedding and head")
                ) {
                    try GLM5NextMTP(stage: stage) { name in
                        reads += 1
                        return try f.tensor(name)
                    }
                }
                #expect(reads == 0)
            }
        }

        @Test func componentInitializerPreservesIdentityAndExactValidation() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let stage = try f.stage()
                let c = f.configuration
                let loader = GLM5NextTensorLoader(f.inventory, f.tensor)
                let p = "model.language_model.layers.\(c.hiddenLayers)"
                let projection = try loader.linear(p + ".eh_proj")
                let attention = try GLM5NextDSAAttention(
                    c, prefix: p + ".self_attn", loader: loader, executionPolicy: .reference)
                let moe = try loader.moe(c, prefix: p + ".mlp", materialize: nil)
                let norm = try loader.array(p + ".shared_head.norm.weight")
                func make(
                    projection: Linear, attention: GLM5NextDSAAttention, outputNorm: MLXArray
                ) throws -> GLM5NextMTP {
                    try .init(
                        stage: stage, embeddingNorm: loader.array(p + ".enorm.weight"),
                        hiddenNorm: loader.array(p + ".hnorm.weight"), projection: projection,
                        attention: attention,
                        inputNorm: loader.array(p + ".input_layernorm.weight"),
                        postNorm: loader.array(p + ".post_attention_layernorm.weight"),
                        mlp: moe, outputNorm: outputNorm)
                }
                let mtp = try make(projection: projection, attention: attention, outputNorm: norm)
                #expect(mtp.modules().contains { $0 === projection })
                #expect(mtp.modules().contains { $0 === attention })
                #expect(mtp.modules().contains { $0 === moe })
                #expect(throws: GLM5NextMTPError.invalid("trained MTP component geometry")) {
                    try make(
                        projection: Linear(weight: MLXArray.zeros([c.hiddenSize, c.hiddenSize])),
                        attention: attention, outputNorm: norm)
                }
                #expect(throws: GLM5NextMTPError.invalid("trained MTP normalization weights")) {
                    try make(
                        projection: projection, attention: attention,
                        outputNorm: MLXArray.ones([c.hiddenSize + 1]))
                }
                let foreignPolicy = try GLM5NextDSAAttention(
                    c, prefix: p + ".self_attn", loader: loader,
                    executionPolicy: .init(pagedMetadataReads: .deviceTable))
                #expect(
                    throws: GLM5NextMTPError.invalid("MTP execution policy must match the target")
                ) {
                    try make(projection: projection, attention: foreignPolicy, outputNorm: norm)
                }
                // Equal tensor dimensions are insufficient: the complete
                // attention configuration must belong to this target.
                let other = try Fixture(layers: 8)
                let foreignConfiguration = try GLM5NextDSAAttention(
                    other.configuration, prefix: "model.language_model.layers.8.self_attn",
                    loader: GLM5NextTensorLoader(other.inventory, other.tensor),
                    executionPolicy: .reference)
                #expect(throws: GLM5NextMTPError.invalid("trained MTP component geometry")) {
                    try make(
                        projection: projection, attention: foreignConfiguration, outputNorm: norm)
                }
            }
        }

        @Test(arguments: [false, true])
        func fullModelStillRejectsSameShapeSharedComponentReplacement(replaceHead: Bool) throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let model = try f.model()
                let mtp = try GLM5NextMTP(target: model, tensor: f.tensor)
                let state = try mtp.newRequestState()
                if replaceHead {
                    model.update(
                        modules: ModuleChildren.unflattened([
                            ("lm_head", Linear(weight: model.lmHead.weight) as Module)
                        ]))
                } else {
                    let replacement = Embedding(weight: model.model.embedTokens.weight)
                    model.model.update(
                        modules: ModuleChildren.unflattened([
                            ("embed_tokens", replacement as Module)
                        ]))
                }
                #expect(throws: invalidOwner) { try mtp.newRequestState() }
                #expect(throws: invalidOwner) { try mtp.reset(state: state) }
                try mtp.release(state: state)
            }
        }

        @Test func splitTargetMTPMatchesFullTargetPrimingAndDraftRounds() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let full = try f.model()
                let first = try f.stage(layers: 0 ..< 3, loadMTPEmbedding: false)
                let last = try f.stage()
                let fullMTP = try GLM5NextMTP(target: full, tensor: f.tensor)
                let stageMTP = try GLM5NextMTP(stage: last, tensor: f.tensor)
                let fullCache = full.newTextCache()
                let firstCache = first.newCache()
                let lastCache = last.newCache()
                func forward(_ tokens: MLXArray) throws
                    -> (GLM5NextTextOutput, GLM5NextTextOutput)
                {
                    let expected = try full.prepareForward(tokens: tokens, cache: fullCache)
                        .commit(afterEvaluating: fence)
                    let candidate = try first.prepareForward(tokens: tokens, cache: firstCache)
                    guard case .streams(let streams) = candidate.output else {
                        throw GLM5NextLoadError.invalid("expected intermediate streams")
                    }
                    let tail = try last.prepareForward(streams: streams, cache: lastCache)
                    guard case .final(let actual) = try tail.commit(afterEvaluating: fence) else {
                        throw GLM5NextLoadError.invalid("expected final stage output")
                    }
                    try candidate.commit(afterEvaluating: fence)
                    close(actual.hidden, expected.hidden)
                    close(actual.logits, expected.logits)
                    return (expected, actual)
                }
                let prompt = MLXArray([Int32(2), 5, 3, 9, 7], [1, 5])
                let carry = MLXArray([Int32(4)], [1, 1])
                let (fullPrompt, stagePrompt) = try forward(prompt)
                let fullState = try fullMTP.newRequestState()
                let stageState = try stageMTP.newRequestState()
                let expected = try fullMTP.primePrompt(
                    tokens: prompt, targetHidden: fullPrompt.hidden,
                    carryToken: carry, state: fullState)
                let actual = try stageMTP.primePrompt(
                    tokens: prompt, targetHidden: stagePrompt.hidden,
                    carryToken: carry, state: stageState)
                try fence([expected.hidden, expected.logits, actual.hidden, actual.logits])
                close(actual.hidden, expected.hidden)
                close(actual.logits, expected.logits)
                let (fullNext, stageNext) = try forward(carry)
                // Reject once, then repeat and accept. Both bindings must recycle
                // the same hidden state and preserve the same committed frontier.
                for acceptingInputs in [0, 1] {
                    try fullMTP.beginRound(state: fullState)
                    try stageMTP.beginRound(state: stageState)
                    let token = MLXArray([Int32(6)], [1, 1])
                    let expected = try fullMTP.draftStep(
                        shiftedToken: token, targetHidden: fullNext.hidden, state: fullState)
                    let actual = try stageMTP.draftStep(
                        shiftedToken: token, targetHidden: stageNext.hidden, state: stageState)
                    try fence([expected.hidden, expected.logits, actual.hidden, actual.logits])
                    close(actual.hidden, expected.hidden)
                    close(actual.logits, expected.logits)
                    try fullMTP.commitRound(state: fullState, acceptingInputs: acceptingInputs)
                    try stageMTP.commitRound(state: stageState, acceptingInputs: acceptingInputs)
                    #expect(stageState.committedInputCount == 5 + acceptingInputs)
                    #expect(stageState.committedInputCount == fullState.committedInputCount)
                    close(
                        try #require(stageState.recycledHidden),
                        try #require(fullState.recycledHidden))
                }
                try stageMTP.release(state: stageState)
                try fullMTP.release(state: fullState)
                withExtendedLifetime((full, first, last)) {}
            }
        }
    }
}
