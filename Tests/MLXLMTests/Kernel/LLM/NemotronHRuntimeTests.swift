import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// A subclass that the MTP graph cache does not know. The cache must then
/// use the reflective identity of the module tree.
private final class NemotronHRuntimeMoESubclass: NemotronHMoE {}

extension KernelTests {

    /// Runtime tests of `NemotronHModel` and `NemotronH35Model` with tiny
    /// random models.
    ///
    /// The tests drive the branches that the older XCTest suites do not run:
    /// configuration errors, the SSM padding mask, a convolution kernel of 1,
    /// the shortlist head, the CBv2 prefill entry points, and the MTP capture
    /// path through MLP and MoE layers.
    @Suite
    struct NemotronHRuntimeTests {

        static let vocabularySize = 64

        /// Hidden 64, 4 attention heads of 16, 4 Mamba heads of 16 with 2
        /// groups (Mamba inner width 64, convolution width 128), 4 experts
        /// with 1 shared expert.
        static var base: [String: Any] {
            [
                "model_type": "nemotron_h",
                "vocab_size": vocabularySize,
                "hidden_size": 64,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 64,
                "moe_intermediate_size": 32,
                "moe_shared_expert_intermediate_size": 32,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "layers_block_type": ["mamba", "mlp", "moe"],
                "mamba_ssm_cache_dtype": "float32",
            ]
        }

        /// A trunk with no attention layer. The CBv2 calls then need no
        /// attention caches.
        static let trunk = ["mamba", "mlp", "moe"]

        static func configuration(_ overrides: [String: Any] = [:]) throws
            -> NemotronHConfiguration
        {
            try SyntheticModel.configuration(
                NemotronHConfiguration.self, base, overrides: overrides)
        }

        static func blockOverrides(_ blocks: [String], _ overrides: [String: Any])
            -> [String: Any]
        {
            var merged = overrides
            merged["layers_block_type"] = blocks
            merged["num_hidden_layers"] = blocks.count
            return merged
        }

        static func makeModel(
            _ blocks: [String], overrides: [String: Any] = [:], seed: UInt64 = 1
        ) throws -> NemotronHModel {
            let model = NemotronHModel(
                try configuration(blockOverrides(blocks, overrides)))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func makeLightning(
            _ blocks: [String], overrides: [String: Any] = [:], seed: UInt64 = 1
        ) throws -> NemotronH35Model {
            let configuration = try SyntheticModel.configuration(
                NemotronH35Configuration.self, base,
                overrides: blockOverrides(blocks, overrides))
            let model = NemotronH35Model(configuration)
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func tokens(_ values: [Int]) -> MLXArray {
            MLXArray(values.map { Int32($0) }).reshaped(1, values.count)
        }

        /// A fixed float32 input with values in [-1, 1].
        static func signal(_ shape: [Int], phase: Float = 0) -> MLXArray {
            let count = shape.reduce(1, *)
            return sin(MLXArray(0 ..< count).asType(.float32) * 0.37 + phase).reshaped(shape)
        }

        /// Runs one committed CBv2 forward and returns the final hidden rows.
        static func committedStep(
            _ model: NemotronHModel, _ state: CBv2RecurrentRequestState, _ values: [Int]
        ) throws -> MLXArray {
            let binding = try state.bind()
            let hidden = model.cbv2Hidden(tokens(values), caches: [], recurrentState: [binding])
            eval([hidden] + (try binding.evaluate()))
            try binding.commit()
            return hidden
        }

        static func difference(_ a: MLXArray?, _ b: MLXArray?) -> Float {
            guard let a, let b, a.shape == b.shape else { return .infinity }
            return SyntheticModel.maxAbsDifference(a, b)
        }

        // Tolerance of the float32 comparisons: the two sides run the same
        // equations with other matrix shapes or kernels (one row against a
        // window, a compiled graph against eager calls), so sums run in
        // another order. The differences are near 1e-6 for values of size 1.
        // A wrong state or a wrong row gives differences above 1e-2.
        static let tolerance: Float = 1e-4

        // MARK: - Configuration

        @Test func configurationWithoutAValidBlockPatternThrows() throws {
            var noPattern = Self.base
            noPattern.removeValue(forKey: "layers_block_type")
            #expect(throws: DecodingError.self) {
                try SyntheticModel.configuration(NemotronHConfiguration.self, noPattern)
            }
            // Three block types for two layers.
            #expect(throws: DecodingError.self) {
                try Self.configuration(["num_hidden_layers": 2])
            }
        }

        @Test func timeStepLimitsDecodeFromOneValueOrFromScalars() throws {
            let single = try Self.configuration(["time_step_limit": [0.25]])
            #expect(single.timeStepLimitMin == 0.25)
            #expect(single.timeStepLimitMax == 0.25)

            let scalars = try Self.configuration([
                "time_step_limit_min": 0.5, "time_step_limit_max": 50,
            ])
            #expect(scalars.timeStepLimitMin == 0.5)
            #expect(scalars.timeStepLimitMax == 50)
        }

        @Test func lightningRejectsANarrowSSMCacheType() throws {
            let overrides: [String: Any] = ["mamba_ssm_cache_dtype": "bfloat16"]
            // The legacy configuration accepts the value.
            #expect(try Self.configuration(overrides).mambaSSMCacheDType == "bfloat16")
            #expect(throws: DecodingError.self) {
                try SyntheticModel.configuration(
                    NemotronH35Configuration.self, Self.base, overrides: overrides)
            }
        }

        @Test func lightningMinimumOnlyLimitRoundTripsWithAnOpenMaximum() throws {
            let configuration = try SyntheticModel.configuration(
                NemotronH35Configuration.self, Self.base,
                overrides: ["time_step_limit_min": 0.125])
            #expect(configuration.hasExplicitExecutionLimits)
            #expect(configuration.target.timeStepLimitMax == .infinity)

            // JSON has no infinity: the encoder writes null for the maximum.
            let data = try JSONEncoder().encode(configuration)
            let object = try #require(
                try JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["time_step_limit_max"] is NSNull)
            #expect(object["layers_block_type"] as? [String] == Self.trunk)

            let reopened = try JSONDecoder().decode(NemotronH35Configuration.self, from: data)
            #expect(reopened.hasExplicitExecutionLimits)
            #expect(reopened.target.timeStepLimitMin == 0.125)
            #expect(reopened.target.timeStepLimitMax == .infinity)

            // Explicit limits stay in force in the model.
            let model = NemotronH35Model(reopened)
            #expect(model.configuration.timeStepLimitMin == 0.125)
            // No MTP layers are declared, so the model does not offer MTP.
            #expect(!model.cbv2Capabilities.supportsMTP)
            #expect(model.cbv2Capabilities.supportsPagedKV)
        }

        // MARK: - Serial forward

        /// With no cache the backbone uses no attention mask, so attention
        /// sees later tokens. With a cache the mask is causal.
        @Test func attentionIsCausalOnlyWithACache() throws {
            let model = try Self.makeModel(["mamba", "attention", "mlp"])
            let row = SyntheticModel.tokens(
                count: 11, vocabularySize: Self.vocabularySize, seed: 1)
            var changed = row
            changed[6] = (row[6] + 1) % Self.vocabularySize

            let original = ForwardPassChecks.logits(model, [row])
            let modified = ForwardPassChecks.logits(model, [changed])
            let before = SyntheticModel.maxAbsDifference(
                original[0..., ..<6], modified[0..., ..<6])
            let after = SyntheticModel.maxAbsDifference(
                original[0..., 6...], modified[0..., 6...])
            withKnownIssue(
                """
                NemotronH.swift:837-843: NemotronHBackbone returns the attention mask \
                .none when the cache is nil, so a full forward pass without a cache \
                attends to later tokens. mlx-lm and createAttentionMask(h:cache:) give \
                a causal mask for a prompt of more than 1 token.
                """
            ) {
                #expect(before <= Self.tolerance, "positions before 6 changed by \(before)")
            } matching: {
                $0.isFailedExpectation(["positions before"])
            }
            #expect(after > 1e-3, "the change at 6 must change its own logits")

            // Control: the same pass with a new cache is causal.
            let cachedOriginal = ForwardPassChecks.logits(
                model, [row], cache: model.newCache(parameters: nil))
            let cachedModified = ForwardPassChecks.logits(
                model, [changed], cache: model.newCache(parameters: nil))
            let cachedBefore = SyntheticModel.maxAbsDifference(
                cachedOriginal[0..., ..<6], cachedModified[0..., ..<6])
            #expect(cachedBefore <= Self.tolerance, "cached positions before 6")
        }

        /// A left-padded row with the `MambaCache` padding mask gives the
        /// logits of the row without padding. Without the mask, the pad
        /// token changes the logits.
        @Test func ssmPaddingMaskRemovesTheLeftPad() throws {
            let model = try Self.makeModel(["mamba", "mlp"])
            let row = [12, 40, 7, 33]
            let padded = [5] + row
            let unpadded = ForwardPassChecks.logits(model, [row], cache: [MambaCache()])
            let masked = ForwardPassChecks.logits(
                model, [padded], cache: [MambaCache(leftPadding: [1])])
            let unmasked = ForwardPassChecks.logits(model, [padded], cache: [MambaCache()])

            #expect(masked.shape == [1, 5, Self.vocabularySize])
            let maskedDifference = SyntheticModel.maxAbsDifference(
                masked[0..., 1 ..< 5], unpadded)
            #expect(maskedDifference <= Self.tolerance, "masked rows differ by \(maskedDifference)")
            #expect(
                SyntheticModel.maxAbsDifference(unmasked[0..., 1 ..< 5], unpadded) > 1e-3,
                "without the mask the pad token must change the logits")
        }

        /// With a kernel of 1 the convolution keeps no history. The cache
        /// then holds an empty convolution state and the SSM state only.
        @Test func convolutionKernelOfOneKeepsNoHistory() throws {
            // No attention layer: the full pass without a cache is then
            // causal (see attentionIsCausalOnlyWithACache).
            let model = try Self.makeModel(["mamba", "mlp"], overrides: ["conv_kernel": 1])
            let row = SyntheticModel.tokens(
                count: 11, vocabularySize: Self.vocabularySize, seed: 2)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [row], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)

            let cache = model.newCache(parameters: nil)
            #expect(cache.count == 1)
            _ = ForwardPassChecks.logits(model, [Array(row.prefix(3))], cache: cache)
            let mamba = try #require(cache[0] as? MambaCache)
            #expect(mamba[0]?.shape == [1, 0, 128])
            #expect(mamba[1]?.shape == [1, 4, 16, 16])
            #expect(mamba[1]?.dtype == .float32)
        }

        /// The shortlist head scores only the given vocabulary rows. It must
        /// give the full head's logits at those rows, for a separate head and
        /// for tied embeddings, each plain and quantized.
        @Test func shortlistLogitsMatchTheFullHeadAtTheSelectedRows() throws {
            let ids = MLXArray([Int32(3), 17, 40, 63, 0, 9, 22, 51])
            let hidden = Self.signal([1, 3, 64])
            for tied in [false, true] {
                for quantized in [false, true] {
                    let model = try Self.makeModel(
                        ["mamba"], overrides: ["tie_word_embeddings": tied])
                    if quantized {
                        quantize(model: model, groupSize: 32, bits: 4)
                    }
                    #expect((model.lmHead == nil) == tied)
                    let shortlist = model.mtpShortlistLogits(hidden, ids: ids)
                    let full = model.logits(hidden).take(ids, axis: -1)
                    eval(shortlist, full)
                    let label = "tied=\(tied), quantized=\(quantized)"
                    #expect(shortlist.shape == [1, 3, 8], "\(label)")
                    #expect(
                        SyntheticModel.maxAbsDifference(shortlist, full) <= Self.tolerance,
                        "\(label)")
                }
            }
        }

        // MARK: - CBv2 entry points

        @Test func cbv2EntryPointsReturnTheRequestedSlices() throws {
            let model = try Self.makeLightning(Self.trunk)
            let spec = model.cbv2RecurrentStateSpec
            let prompt = [3, 1, 4, 1, 5]
            let input = Self.tokens(prompt)
            #expect(!model.cbv2SupportsPackedPrefill)
            #expect(model.cbv2LayerKinds.isEmpty)
            // No attention layer: the list of attention KV types is empty.
            #expect(model.cbv2CompleteCheckpointKVDTypes == [DType]())

            let hidden = try Self.committedStep(
                model, try CBv2RecurrentRequestState(spec: spec), prompt)
            let logits = model.logits(hidden)
            eval(logits)
            #expect(hidden.shape == [1, 5, 64])

            // The CBv2 path runs the MLP and MoE layers like the serial path.
            let serial = model(input, cache: model.newCache(parameters: nil))
            eval(serial)
            #expect(SyntheticModel.maxAbsDifference(serial, logits) <= Self.tolerance)

            // An evaluation holds its request state unowned, so the test keeps
            // each state alive to the end.
            var states: [CBv2RecurrentRequestState] = []
            func binding() throws -> CBv2RecurrentStateEvaluation {
                let state = try CBv2RecurrentRequestState(spec: spec)
                states.append(state)
                return try state.bind()
            }
            defer { withExtendedLifetime(states) {} }

            // Same ops on the same data: exact.
            let withHiddenBinding = try binding()
            let withHidden = model.cbv2ForwardWithHidden(
                input, caches: [], recurrentState: [withHiddenBinding], positionIds: nil)
            eval([withHidden.logits, withHidden.lastHidden] + (try withHiddenBinding.evaluate()))
            #expect(SyntheticModel.maxAbsDifference(withHidden.lastHidden, hidden) == 0)
            #expect(SyntheticModel.maxAbsDifference(withHidden.logits, logits) == 0)

            let lastLogitsBinding = try binding()
            let lastLogits = model.cbv2RecurrentPrefill(
                input, inputEmbedding: nil, cache: nil, recurrentState: [lastLogitsBinding],
                positionIds: nil, requirement: .lastPositionLogits)
            eval([lastLogits] + (try lastLogitsBinding.evaluate()))
            #expect(lastLogits.shape == [1, Self.vocabularySize])
            #expect(
                SyntheticModel.maxAbsDifference(lastLogits, logits[0..., -1, 0...])
                    <= Self.tolerance)

            let evaluationBinding = try binding()
            let evaluation = model.cbv2RecurrentPrefill(
                input, inputEmbedding: nil, cache: nil, recurrentState: [evaluationBinding],
                positionIds: nil, requirement: .evaluationOnly)
            eval([evaluation] + (try evaluationBinding.evaluate()))
            #expect(evaluation.shape == [1, 1])
            #expect(
                SyntheticModel.maxAbsDifference(evaluation, hidden[0..., -1, 0 ..< 1]) == 0)

            let prefillLogitsBinding = try binding()
            let prefillLogits = model.cbv2ForwardWithHiddenForPrefill(
                input, caches: [], recurrentState: [prefillLogitsBinding], positionIds: nil,
                requirement: .lastPositionLogits)
            eval(
                [prefillLogits.logits, prefillLogits.lastHidden]
                    + (try prefillLogitsBinding.evaluate()))
            #expect(prefillLogits.logits.shape == [1, 1, Self.vocabularySize])
            #expect(prefillLogits.lastHidden.shape == [1, 5, 64])
            #expect(
                SyntheticModel.maxAbsDifference(prefillLogits.logits, logits[0..., 4 ..< 5])
                    <= Self.tolerance)

            let prefillEvaluationBinding = try binding()
            let prefillEvaluation = model.cbv2ForwardWithHiddenForPrefill(
                input, caches: [], recurrentState: [prefillEvaluationBinding],
                positionIds: nil, requirement: .evaluationOnly)
            eval(
                [prefillEvaluation.logits, prefillEvaluation.lastHidden]
                    + (try prefillEvaluationBinding.evaluate()))
            #expect(prefillEvaluation.logits.shape == [1, 1, 1])
            #expect(
                SyntheticModel.maxAbsDifference(
                    prefillEvaluation.logits, hidden[0..., 4 ..< 5, 0 ..< 1]) == 0)
        }

        // MARK: - MTP capture

        /// A captured window gives the hidden rows of serial one-token steps.
        /// A commit that keeps `keep` positions leaves the recurrent state of
        /// the serial run after `keep` tokens.
        @Test(arguments: [1, 2, 4])
        func capturedWindowMatchesSerialStepsAndKeepsThePrefix(keep: Int) throws {
            let model = try Self.makeLightning(Self.trunk)
            let spec = model.cbv2RecurrentStateSpec
            let prompt = [3, 1, 4]
            let window = [9, 26, 5, 35]

            let serial = try CBv2RecurrentRequestState(spec: spec)
            _ = try Self.committedStep(model, serial, prompt)
            var serialHidden: [MLXArray] = []
            var serialStates: [CBv2RecurrentLayerState] = []
            for token in window {
                serialHidden.append(try Self.committedStep(model, serial, [token]))
                serialStates.append(try #require(serial.state(modelLayerIndex: 0)))
            }

            let captured = try CBv2RecurrentRequestState(spec: spec)
            _ = try Self.committedStep(model, captured, prompt)
            let binding = try captured.bind()
            let hidden = model.cbv2Hidden(
                Self.tokens(window), caches: [], recurrentState: [binding], captureMTP: true)
            eval([hidden] + (try binding.evaluate()))
            #expect(binding.isCaptured)
            #expect(hidden.shape == [1, window.count, 64])
            for position in window.indices {
                let row = hidden[0..., position ..< (position + 1), 0...]
                let difference = Self.difference(row, serialHidden[position])
                #expect(difference <= Self.tolerance, "row \(position) differs by \(difference)")
            }

            try binding.commit(keepPositions: keep)
            let kept = try #require(captured.state(modelLayerIndex: 0))
            let expected = serialStates[keep - 1]
            #expect(kept.conv?.shape == [1, 3, 128])
            #expect(Self.difference(kept.conv, expected.conv) <= Self.tolerance, "conv")
            #expect(kept.ssm?.dtype == .float32)
            #expect(Self.difference(kept.ssm, expected.ssm) <= Self.tolerance, "ssm")
        }

        /// One captured token from an empty state starts from zero state, like
        /// the serial path.
        @Test func capturedSingleTokenFromAnEmptyStateMatchesSerial() throws {
            let model = try Self.makeLightning(Self.trunk)
            let spec = model.cbv2RecurrentStateSpec

            let serial = try CBv2RecurrentRequestState(spec: spec)
            let expected = try Self.committedStep(model, serial, [11])
            let expectedState = try #require(serial.state(modelLayerIndex: 0))

            let captured = try CBv2RecurrentRequestState(spec: spec)
            let binding = try captured.bind()
            let hidden = model.cbv2Hidden(
                Self.tokens([11]), caches: [], recurrentState: [binding], captureMTP: true)
            eval([hidden] + (try binding.evaluate()))
            #expect(binding.isCaptured)
            #expect(Self.difference(hidden, expected) <= Self.tolerance)

            try binding.commit(keepPositions: 1)
            let kept = try #require(captured.state(modelLayerIndex: 0))
            #expect(Self.difference(kept.conv, expectedState.conv) <= Self.tolerance, "conv")
            #expect(Self.difference(kept.ssm, expectedState.ssm) <= Self.tolerance, "ssm")
        }

        /// The captured forward scores each row with the separate head or,
        /// for tied embeddings, with the embedding matrix.
        @Test(arguments: [false, true])
        func capturedLogitsUseTheHeadOrTheTiedEmbeddings(tied: Bool) throws {
            let model = try Self.makeLightning(
                Self.trunk, overrides: ["tie_word_embeddings": tied])
            #expect((model.lmHead == nil) == tied)
            let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
            let binding = try state.bind()
            let output = model.cbv2ForwardWithHiddenCaptured(
                Self.tokens([2, 7, 1]), caches: [], recurrentState: [binding], positionIds: nil)
            eval([output.logits, output.lastHidden] + (try binding.evaluate()))
            #expect(output.logits.shape == [1, 3, Self.vocabularySize])
            #expect(output.lastHidden.shape == [1, 3, 64])
            let full = model.logits(output.lastHidden)
            eval(full)
            #expect(SyntheticModel.maxAbsDifference(output.logits, full) <= Self.tolerance)

            // A rollback leaves no recurrent state.
            try binding.rollback()
            #expect(state.state(modelLayerIndex: 0) == nil)
        }
    }

    /// Tests of the row-by-row MTP forms of the NemotronH MLP and MoE layers
    /// and of the MoE graph cache.
    @Suite
    struct NemotronHMTPRowsTests {

        static func configuration() throws -> NemotronHConfiguration {
            try NemotronHRuntimeTests.configuration()
        }

        static let tolerance: Float = NemotronHRuntimeTests.tolerance

        /// The row form keeps a matrix height of 1 for each row. It must give
        /// the result of the ordinary forward, for plain and quantized
        /// projections.
        @Test(arguments: [false, true])
        func mlpRowsMatchTheOrdinaryForward(quantized: Bool) throws {
            let mlp = NemotronHMLP(try Self.configuration())
            SyntheticModel.randomize(mlp, seed: 3)
            if quantized {
                quantize(model: mlp, groupSize: 32, bits: 4)
            }
            #expect((mlp.upProj is QuantizedLinear) == quantized)
            let x = NemotronHRuntimeTests.signal([1, 4, 64])
            let rows = mlp.mtpForwardRows(x)
            let ordinary = mlp(x)
            eval(rows, ordinary)
            #expect(rows.shape == [1, 4, 64])
            #expect(SyntheticModel.maxAbsDifference(rows, ordinary) <= Self.tolerance)
        }

        /// More than 8 rows, or more than 1 batch row, is outside the shape of
        /// the compiled graph. The eager row form then runs.
        @Test func moeRowsOutsideTheCompiledShapeMatchTheOrdinaryForward() throws {
            let moe = NemotronHMoE(try Self.configuration())
            SyntheticModel.randomize(moe, seed: 4)
            for shape in [[1, 9, 64], [2, 3, 64]] {
                let x = NemotronHRuntimeTests.signal(shape, phase: 0.5)
                let rows = moe.mtpForwardRows(x)
                let ordinary = moe(x)
                eval(rows, ordinary)
                #expect(rows.shape == shape)
                #expect(
                    SyntheticModel.maxAbsDifference(rows, ordinary) <= Self.tolerance,
                    "shape \(shape)")
            }
            #expect(moe.mtpCompiledRowsCache.callCount == 0)
        }

        /// For an unknown MoE type the graph cache uses the reflective
        /// identity. A new expert module must build a new graph.
        @Test func graphCacheUsesTheReflectiveIdentityForASubclass() throws {
            let owner = NemotronHRuntimeMoESubclass(try Self.configuration())
            SyntheticModel.randomize(owner, seed: 5)
            let cache = owner.mtpCompiledRowsCache
            let x = NemotronHRuntimeTests.signal([1, 3, 64])

            let first = cache(x, owner: owner)
            let expected = owner.mtpForwardRowsUncompiled(x)
            eval(first, expected)
            #expect(SyntheticModel.maxAbsDifference(first, expected) <= Self.tolerance)
            let second = cache(x * 0.5, owner: owner)
            eval(second)
            #expect(cache.generationCount == 1)
            #expect(cache.callCount == 2)

            let replacement = NemotronHSwitchMLP(inputDims: 64, hiddenDims: 32, numExperts: 4)
            SyntheticModel.randomize(replacement, seed: 6)
            owner.update(modules: ModuleChildren.unflattened([("switch_mlp", replacement)]))
            let after = cache(x, owner: owner)
            let expectedAfter = owner.mtpForwardRowsUncompiled(x)
            eval(after, expectedAfter)
            #expect(cache.generationCount == 2, "a new expert module must build a new graph")
            #expect(SyntheticModel.maxAbsDifference(after, expectedAfter) <= Self.tolerance)
            #expect(
                SyntheticModel.maxAbsDifference(after, first) > 1e-3,
                "the new experts must change the output")
        }
    }
}
