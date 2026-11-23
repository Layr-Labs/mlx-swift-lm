import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Inkling 2 DSpark drafter against the CPU reference
/// (`tools/inkling2_dspark_reference.py tiny`): a small random drafter with
/// the checkpoint's structure (3 layers, 0 and 2 sliding, block 8, rnn_hidden
/// Markov head, four target features).
@Suite("Inkling2 DSpark drafter", .serialized)
struct Inkling2DSparkTests {
    struct Fixture {
        let model: Inkling2DSparkModel
        let target: Inkling2DSparkArrayTarget
        let head: Inkling2DSparkFoldedHead
        let arrays: [String: MLXArray]
        let cases: Int
    }

    static func rootConfig(mtp: [String: Any], vocabulary: Int, padded: Int, hidden: Int)
        -> [String: Any]
    {
        [
            "model_type": "inkling2_loop",
            "text_config": [
                "hidden_size": hidden, "num_hidden_layers": 24, "vocab_size": padded,
                "unpadded_vocab_size": vocabulary, "final_logit_softcapping": 30.0,
                "loop_config": ["core_start": 6, "core_end": 18, "num_passes": 3],
            ],
            "mtp_config": mtp,
        ]
    }

    static func fixture() throws -> Fixture {
        let url = try #require(
            Bundle.module.url(
                forResource: "inkling2-dspark-tiny-oracle", withExtension: "safetensors"))
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        let configText = try #require(metadata["config"])
        let mtp = try #require(
            JSONSerialization.jsonObject(with: Data(configText.utf8)) as? [String: Any])
        let vocabulary = Int(try #require(metadata["vocab"]))!
        let unembedding = try #require(arrays["target.unembed"])
        let root = rootConfig(
            mtp: mtp, vocabulary: vocabulary, padded: unembedding.dim(0),
            hidden: unembedding.dim(1))
        let configuration = try Inkling2DSparkConfiguration.parse(
            rootConfig: JSONSerialization.data(withJSONObject: root))
        let model = Inkling2DSparkModel(configuration)
        var weights: [String: MLXArray] = [:]
        for (key, value) in arrays where key.hasPrefix("weights.") {
            weights[String(key.dropFirst("weights.".count))] = value
        }
        try model.load(weights: weights)
        let target = Inkling2DSparkArrayTarget(
            embedding: try #require(arrays["target.embed"]),
            embeddingNorm: try #require(arrays["target.embed_norm"]),
            unembedding: unembedding, eps: configuration.rmsNormEps)
        let head = model.foldHead(unembedding: unembedding, chunk: 128)
        let cases = arrays.keys.filter { $0.hasSuffix(".anchor") }.count
        return Fixture(model: model, target: target, head: head, arrays: arrays, cases: cases)
    }

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).reshaped(-1)
        let y = b.asType(.float32).reshaped(-1)
        return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
    }

    static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Context of every layer over `features` [n, featureWidth], window-limited.
    static func contexts(_ model: Inkling2DSparkModel, features: MLXArray)
        -> [(keys: MLXArray, values: MLXArray, mask: MLXArray?)]
    {
        let c = model.configuration
        let n = features.dim(0)
        return (0 ..< c.numHiddenLayers).map { layer in
            var (keys, values) = model.contextKV(layer: layer, features: features, start: 0)
            if c.isLocal(layer: layer), n > c.slidingWindowSize {
                keys = keys[0..., 0..., (n - c.slidingWindowSize)..., 0...]
                values = values[0..., 0..., (n - c.slidingWindowSize)..., 0...]
            }
            return (keys: keys, values: values, mask: nil)
        }
    }

    static func directProposal(
        _ f: Fixture, features: MLXArray, anchor: Int32, depth: Int
    ) -> (drafts: [Int32], hidden: MLXArray) {
        let n = features.dim(0)
        let result = f.model.propose(
            anchor: MLXArray([anchor]), position: n, lastFeatures: features[(n - 1)...],
            contexts: contexts(f.model, features: features), target: f.target, head: f.head,
            depth: depth)
        eval(result.drafts, result.hidden)
        return (result.drafts.asArray(Int32.self), result.hidden)
    }

    @Test func parsesCheckpointConfiguration() throws {
        let mtp: [String: Any] = [
            "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24,
            "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9216,
            "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0, "local_layer_ids": [0, 2],
            "sliding_window_size": 1024, "block_size": 8, "markov_rank": 256,
            "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48],
            "q_bias": false, "o_bias": false,
        ]
        let root = Self.rootConfig(mtp: mtp, vocabulary: 200_058, padded: 201_024, hidden: 2304)
        let c = try Inkling2DSparkConfiguration.parse(
            rootConfig: JSONSerialization.data(withJSONObject: root))
        #expect(c.targetLayerVisits == 48)
        #expect(c.featureWidth == 4 * 2304)
        #expect(c.draftTokens == 7)
        #expect(c.vocabularySize == 200_058 && c.paddedVocabularySize == 201_024)
        #expect(c.isLocal(layer: 0) && !c.isLocal(layer: 1) && c.isLocal(layer: 2))

        var wrong = mtp
        wrong["target_layer_ids"] = [5, 23, 41, 47]
        #expect(throws: Inkling2DSparkError.self) {
            try Inkling2DSparkConfiguration.parse(
                rootConfig: JSONSerialization.data(
                    withJSONObject: Self.rootConfig(
                        mtp: wrong, vocabulary: 200_058, padded: 201_024, hidden: 2304)))
        }
        wrong = mtp
        wrong["markov_head_type"] = "mlp"
        #expect(throws: Inkling2DSparkError.self) {
            try Inkling2DSparkConfiguration.parse(
                rootConfig: JSONSerialization.data(
                    withJSONObject: Self.rootConfig(
                        mtp: wrong, vocabulary: 200_058, padded: 201_024, hidden: 2304)))
        }
    }

    @Test func foldedHeadMatchesReference() throws {
        let f = try Self.fixture()
        let referenceWeight = try #require(f.arrays["fold_w"])
        let referenceBias = try #require(f.arrays["fold_b"])
        // fp32 here, fp64 in the reference: at most a rare one-ulp bf16 difference.
        let weightDiffers = (f.head.foldWeight .!= referenceWeight).sum().item(Int.self)
        let biasDiffers = (f.head.foldBias .!= referenceBias).sum().item(Int.self)
        #expect(weightDiffers * 200 <= referenceWeight.size)
        #expect(biasDiffers * 50 <= referenceBias.size)
        #expect(Self.maxAbs(f.head.foldWeight, referenceWeight) < 0.02)
    }

    @Test func contextAndBlockMatchReference() throws {
        let f = try Self.fixture()
        let features = try #require(f.arrays["case0.features"])
        for layer in 0 ..< f.model.configuration.numHiddenLayers {
            let (keys, values) = f.model.contextKV(layer: layer, features: features, start: 0)
            let refKeys = try #require(f.arrays["case0.trace.ctx_k.\(layer)"])
            let refValues = try #require(f.arrays["case0.trace.ctx_v.\(layer)"])
            // Reference layout [T, kv, D]; ours [1, kv, T, D].
            let k = keys[0].transposed(1, 0, 2)
            let v = values[0].transposed(1, 0, 2)
            #expect(Self.cosine(k, refKeys) > 0.9999)
            #expect(Self.cosine(v, refValues) > 0.9999)
            #expect(Self.maxAbs(v, refValues) < 0.05)
        }
        var compared = 0
        var report: [String] = []
        for index in 0 ..< f.cases {
            let features = try #require(f.arrays["case\(index).features"])
            let anchor = try #require(f.arrays["case\(index).anchor"]).item(Int32.self)
            let expected = try #require(f.arrays["case\(index).drafts"]).asArray(Int32.self)
            let reference = try #require(f.arrays["case\(index).hidden"])
            let referenceLogits = try #require(f.arrays["case\(index).trace.logits"])
            let margins = try #require(f.arrays["case\(index).trace.margins"]).asArray(Float.self)
            let result = Self.directProposal(
                f, features: features, anchor: anchor, depth: expected.count)
            let cosine = Self.cosine(result.hidden[0], reference)
            #expect(cosine > 0.9995, "case \(index) hidden cosine \(cosine)")
            let logits = Self.firstStepLogits(
                f, hidden: result.hidden, features: features, anchor: anchor)
            let logitError = Self.maxAbs(logits, referenceLogits[0])
            #expect(logitError <= 0.125, "case \(index) first-step logits differ by \(logitError)")
            // A draft may differ only where the reference's top-two margin is
            // within bf16 rounding; the chains then diverge legitimately.
            var agreed = 0
            for (step, token) in expected.enumerated() {
                compared += 1
                guard result.drafts[step] == token else {
                    #expect(
                        margins[step] <= 0.25, "case \(index) step \(step) margin \(margins[step])")
                    break
                }
                agreed += 1
            }
            report.append(
                "case \(index): cosine \(cosine) logit error \(logitError) agreed \(agreed)/7")
        }
        print(report.joined(separator: "\n"))
        #expect(compared > 0)
    }

    /// Step-0 draft logits of a proposal, recomputed from its block hidden.
    static func firstStepLogits(
        _ f: Fixture, hidden: MLXArray, features: MLXArray, anchor: Int32
    ) -> MLXArray {
        let n = features.dim(0)
        let plan = f.model.markovPlan(
            rows: hidden[0..., 1..., 0...],
            markovFeatures: f.model.markovFeatures(features[(n - 1)...]), head: f.head)
        let step = f.model.markovStep(
            state: f.model.initialMarkovState(batch: 1), previous: MLXArray([anchor]),
            context: plan.context[0..., 0, 0...])
        return f.model.draftLogits(
            base: plan.baseLogits![0..., 0, 0...], latent: step.latent, head: f.head)[0]
    }

    // MARK: CBv2 seam

    static func makeDrafter(_ f: Fixture) -> Inkling2DSparkDrafter {
        Inkling2DSparkDrafter(model: f.model, target: f.target, head: f.head)
    }

    static func draft(
        _ drafter: Inkling2DSparkDrafter, state: any CBv2MTPRequestState, anchor: Int32,
        seed: MLXArray, depth: Int
    ) -> [Int32] {
        var token = MLXArray([anchor]).reshaped(1, 1)
        var hidden = seed
        var drafts: [Int32] = []
        for _ in 0 ..< depth {
            let step = drafter.draftStep(
                tokens: token, hidden: hidden, shortlist: nil, requestState: state)
            eval([step.tokens, step.hidden] + drafter.evaluationTargets(for: state))
            drafts.append(step.tokens.item(Int32.self))
            token = step.tokens.reshaped(1, 1)
            hidden = step.hidden
        }
        return drafts
    }

    @Test func pagedDrafterMatchesDirectProposal() throws {
        let f = try Self.fixture()
        let drafter = Self.makeDrafter(f)
        let depth = f.model.configuration.draftTokens
        for index in 0 ..< f.cases {
            let features = try #require(f.arrays["case\(index).features"])
            let anchor = try #require(f.arrays["case\(index).anchor"]).item(Int32.self)
            let n = features.dim(0)
            let state = drafter.makeRequestState()
            try drafter.configureRequestState(state, maximumSequenceLength: n + 64)
            drafter.observeCommittedTarget(
                .init(
                    tokens: MLXArray.zeros([1, n], dtype: .int32),
                    hidden: features.reshaped(1, n, -1)),
                requestState: state)
            let drafts = Self.draft(
                drafter, state: state, anchor: anchor,
                seed: features[(n - 1)...].reshaped(1, 1, -1), depth: depth)
            let direct = Self.directProposal(f, features: features, anchor: anchor, depth: depth)
            #expect(drafts == direct.drafts, "case \(index)")
            drafter.releaseRequestState(state)
        }
    }

    /// Rounds with partial acceptance, an ordinary decode with a history
    /// carry, and prefill chunks must leave the paged context identical to a
    /// full recompute over the committed features (positions and windows).
    @Test func incrementalRoundsMatchFullRecompute() throws {
        let f = try Self.fixture()
        let drafter = Self.makeDrafter(f)
        let c = f.model.configuration
        let features = try #require(f.arrays["case4.features"])
        let n = features.dim(0)
        #expect(n >= 36)
        let state = drafter.makeRequestState()
        try drafter.configureRequestState(state, maximumSequenceLength: n + 64)
        func rows(_ range: Range<Int>) -> MLXArray {
            features[range].reshaped(1, range.count, features.dim(1))
        }
        func observe(_ range: Range<Int>) {
            drafter.observeCommittedTarget(
                .init(
                    tokens: MLXArray.zeros([1, range.count], dtype: .int32), hidden: rows(range)),
                requestState: state)
        }
        func check(anchor: Int32, position: Int, depth: Int) -> [Int32] {
            let drafts = Self.draft(
                drafter, state: state, anchor: anchor, seed: rows((position - 1) ..< position),
                depth: depth)
            let direct = Self.directProposal(
                f, features: features[0 ..< position], anchor: anchor, depth: depth)
            #expect(drafts == direct.drafts, "position \(position)")
            return drafts
        }
        // Prompt in two chunks, then a round anchored at 12 accepting 3.
        observe(0 ..< 5)
        observe(5 ..< 12)
        var drafts = check(anchor: 7, position: 12, depth: c.draftTokens)
        drafter.finalizeRound(
            requestState: state, confirmedInputTokens: 4,
            committedDraftTokens: MLXArray(Array(drafts.prefix(3))).reshaped(1, 3),
            committedTargetHidden: rows(12 ..< 15))
        // Next round anchored at 16 (seed row 15) with depth 2, none accepted.
        drafts = check(anchor: 11, position: 16, depth: 2)
        drafter.finalizeRound(
            requestState: state, confirmedInputTokens: 1,
            committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
            committedTargetHidden: rows(16 ..< 16))
        // An ordinary decode replays the carry (row 16, token at 17), then
        // feeds token 17 with its own row 17; a round follows at 18.
        observe(16 ..< 17)
        observe(17 ..< 18)
        drafts = check(anchor: 3, position: 18, depth: c.draftTokens)
        drafter.finalizeRound(
            requestState: state, confirmedInputTokens: 8,
            committedDraftTokens: MLXArray(drafts).reshaped(1, -1),
            committedTargetHidden: rows(18 ..< 25))
        // Full acceptance: the seed is row 25; then a long gap of ordinary
        // decode crosses the sliding window.
        _ = check(anchor: 9, position: 26, depth: 1)
        drafter.finalizeRound(
            requestState: state, confirmedInputTokens: 2,
            committedDraftTokens: MLXArray([Int32(0)]).reshaped(1, 1),
            committedTargetHidden: rows(26 ..< 27))
        observe(27 ..< 28)
        for t in 28 ..< (n - 1) { observe(t ..< t + 1) }
        _ = check(anchor: 5, position: n - 1, depth: c.draftTokens)
        drafter.releaseRequestState(state)
    }

    /// The compiled Markov chain drafts exactly what the op-by-op chain does,
    /// one row and every fixture case batched, at every depth.
    @Test func compiledChainMatchesOpByOp() throws {
        let f = try Self.fixture()
        let drafter = Self.makeDrafter(f)
        func drafts(compiled: Bool, cases: [Int], depth: Int) throws -> [Int32] {
            var states: [any CBv2MTPRequestState] = []
            var tokens: [MLXArray] = []
            var seeds: [MLXArray] = []
            for index in cases {
                let features = try #require(f.arrays["case\(index).features"])
                let anchor = try #require(f.arrays["case\(index).anchor"]).item(Int32.self)
                let n = features.dim(0)
                let state = drafter.makeRequestState()
                try drafter.configureRequestState(state, maximumSequenceLength: n + 64)
                drafter.observeCommittedTarget(
                    .init(
                        tokens: MLXArray.zeros([1, n], dtype: .int32),
                        hidden: features.reshaped(1, n, -1)),
                    requestState: state)
                states.append(state)
                tokens.append(MLXArray([anchor]).reshaped(1, 1))
                seeds.append(features[(n - 1)...].reshaped(1, 1, -1))
            }
            let out = Inkling2DSparkDrafter.$compiledChainOverride.withValue(compiled) {
                drafter.draftBlock(
                    tokens: tokens, hidden: seeds, requestStates: states, depth: depth)
            }
            eval([out] + states.flatMap { drafter.evaluationTargets(for: $0) })
            for state in states { drafter.releaseRequestState(state) }
            return out.asArray(Int32.self)
        }
        let all = Array(0 ..< f.cases)
        for depth in 1 ... f.model.configuration.draftTokens {
            for cases in [[0], all] {
                let opByOp = try drafts(compiled: false, cases: cases, depth: depth)
                let compiled = try drafts(compiled: true, cases: cases, depth: depth)
                #expect(compiled == opByOp, "depth \(depth) cases \(cases)")
            }
        }
    }

    @Test func batchedBlockMatchesPerRowProposals() throws {
        let f = try Self.fixture()
        let drafter = Self.makeDrafter(f)
        let depth = f.model.configuration.draftTokens
        var states: [any CBv2MTPRequestState] = []
        var tokens: [MLXArray] = []
        var seeds: [MLXArray] = []
        var expected: [[Int32]] = []
        for index in 0 ..< f.cases {
            let features = try #require(f.arrays["case\(index).features"])
            let anchor = try #require(f.arrays["case\(index).anchor"]).item(Int32.self)
            let n = features.dim(0)
            let state = drafter.makeRequestState()
            try drafter.configureRequestState(state, maximumSequenceLength: n + 64)
            drafter.observeCommittedTarget(
                .init(
                    tokens: MLXArray.zeros([1, n], dtype: .int32),
                    hidden: features.reshaped(1, n, -1)),
                requestState: state)
            states.append(state)
            tokens.append(MLXArray([anchor]).reshaped(1, 1))
            seeds.append(features[(n - 1)...].reshaped(1, 1, -1))
            expected.append(
                Self.directProposal(f, features: features, anchor: anchor, depth: depth).drafts)
        }
        let drafts = drafter.draftBlock(
            tokens: tokens, hidden: seeds, requestStates: states, depth: depth)
        eval([drafts] + states.flatMap { drafter.evaluationTargets(for: $0) })
        // Padding and masking may move a draft only across a bf16 near-tie.
        var exactRows = 0
        for (index, row) in expected.enumerated() {
            let batched = drafts[index].asArray(Int32.self)
            let margins = try #require(f.arrays["case\(index).trace.margins"]).asArray(Float.self)
            if let step = row.indices.first(where: { batched[$0] != row[$0] }) {
                #expect(
                    margins[step] <= 0.25,
                    "row \(index) step \(step): batched \(batched) per-row \(row)")
            } else {
                exactRows += 1
            }
            #expect(states[index].stagedInputCount == depth)
        }
        #expect(exactRows * 2 > expected.count)
        // The staged block finalizes like a per-step round.
        for state in states {
            drafter.finalizeRound(
                requestState: state, confirmedInputTokens: 1,
                committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
                committedTargetHidden: MLXArray.zeros([1, 0, f.model.configuration.featureWidth]))
            drafter.releaseRequestState(state)
        }
    }

    @Test func prefixCheckpointRestoresColdDrafts() throws {
        let f = try Self.fixture()
        let drafter = Self.makeDrafter(f)
        let features = try #require(f.arrays["case4.features"])
        let n = features.dim(0)
        let prompt = (0 ..< 30).map { ($0 * 37 + 11) % 280 }
        func observe(_ state: any CBv2MTPRequestState, _ range: Range<Int>) {
            drafter.observeCommittedTarget(
                .init(
                    tokens: MLXArray(range.map { Int32(prompt[$0]) }).reshaped(1, range.count),
                    hidden: features[range].reshaped(1, range.count, features.dim(1))),
                requestState: state)
        }
        let seed = features[23 ..< 24].reshaped(1, 1, features.dim(1))
        let cold = drafter.makeRequestState()
        try drafter.configureRequestState(cold, maximumSequenceLength: n + 64)
        observe(cold, 0 ..< 24)
        let expected = Self.draft(drafter, state: cold, anchor: 5, seed: seed, depth: 7)

        // The donor's prefix (20 > the 16-position window) is checkpointed at
        // an interior prompt boundary, under its installed prompt context.
        let donor = drafter.makeRequestState()
        try drafter.configureRequestState(donor, maximumSequenceLength: n + 64)
        #expect(drafter.capturePrefixCheckpoint(requestState: donor, targetInputCount: 20) == nil)
        try drafter.installPrefixCaptureContext(requestState: donor, promptTokens: prompt)
        observe(donor, 0 ..< 10)
        observe(donor, 10 ..< 20)
        let checkpoint = try #require(
            drafter.capturePrefixCheckpoint(requestState: donor, targetInputCount: 20))
        eval(checkpoint.evaluationTargets)
        #expect(drafter.capturePrefixCheckpoint(requestState: donor, targetInputCount: 19) == nil)

        // In memory, and through the durable encode/decode round trip.
        let encoded = try #require(drafter.encodePrefixCheckpoint(checkpoint))
        let descriptors = try #require(
            drafter.prefixCheckpointTensorDescriptors(targetInputCount: 20))
        #expect(encoded.count == descriptors.count)
        let copies = encoded.map { $0 + MLXArray.zeros(like: $0) }
        eval(copies)
        #expect(
            drafter.decodePrefixCheckpoint(
                tensors: copies, prefixTokens: Array(prompt.prefix(19)) + [prompt[0] + 1]) == nil)
        let decoded = try #require(
            drafter.decodePrefixCheckpoint(tensors: copies, prefixTokens: Array(prompt.prefix(20))))
        for source in [checkpoint, decoded] {
            let restored = try #require(drafter.restorePrefixCheckpoint(source))
            #expect(restored.committedInputCount == 20)
            try drafter.configureRequestState(restored, maximumSequenceLength: n + 64)
            #expect(throws: Inkling2DSparkError.self) {
                try drafter.installPrefixCaptureContext(
                    requestState: restored, promptTokens: [prompt[0] + 1] + prompt.dropFirst())
            }
            try drafter.installPrefixCaptureContext(requestState: restored, promptTokens: prompt)
            observe(restored, 20 ..< 24)
            let drafts = Self.draft(drafter, state: restored, anchor: 5, seed: seed, depth: 7)
            #expect(drafts == expected)
            drafter.releaseRequestState(restored)
        }
        for state in [cold, donor] { drafter.releaseRequestState(state) }
    }

    /// Prompt identity evidence stops growing once no capture inside the
    /// installed prompt is possible, and is never kept without a context
    /// (an ordinary-decode stretch before the first round).
    @Test func observedTokensStopAtThePromptBoundary() throws {
        let f = try Self.fixture()
        let drafter = Self.makeDrafter(f)
        let features = try #require(f.arrays["case4.features"])
        let n = features.dim(0)
        let prompt = (0 ..< 24).map { ($0 * 37 + 11) % 280 }
        func observe(_ state: any CBv2MTPRequestState, _ range: Range<Int>) {
            drafter.observeCommittedTarget(
                .init(
                    tokens: MLXArray(range.map { Int32(prompt[$0]) }).reshaped(1, range.count),
                    hidden: features[range].reshaped(1, range.count, features.dim(1))),
                requestState: state)
        }
        func observed(_ state: any CBv2MTPRequestState) throws -> Int {
            try #require(state as? Inkling2DSparkDrafter.State).observedTokens
                .reduce(0) { $0 + $1.dim(1) }
        }
        let bare = drafter.makeRequestState()
        try drafter.configureRequestState(bare, maximumSequenceLength: n + 64)
        observe(bare, 0 ..< 10)
        #expect(try observed(bare) == 0)

        let state = drafter.makeRequestState()
        try drafter.configureRequestState(state, maximumSequenceLength: n + 64)
        try drafter.installPrefixCaptureContext(requestState: state, promptTokens: prompt)
        observe(state, 0 ..< 10)
        #expect(try observed(state) == 10)
        #expect(drafter.capturePrefixCheckpoint(requestState: state, targetInputCount: 10) != nil)
        observe(state, 10 ..< 24)
        #expect(try observed(state) == 0)
        #expect(drafter.capturePrefixCheckpoint(requestState: state, targetInputCount: 24) == nil)
        for s in [bare, state] { drafter.releaseRequestState(s) }
    }

    @Test func discardedRoundDoesNotDuplicateTheSeed() throws {
        let f = try Self.fixture()
        let drafter = Self.makeDrafter(f)
        let features = try #require(f.arrays["case3.features"])
        let n = features.dim(0)
        let state = drafter.makeRequestState()
        try drafter.configureRequestState(state, maximumSequenceLength: n + 64)
        drafter.observeCommittedTarget(
            .init(
                tokens: MLXArray.zeros([1, n], dtype: .int32), hidden: features.reshaped(1, n, -1)),
            requestState: state)
        let seed = features[(n - 1)...].reshaped(1, 1, -1)
        _ = Self.draft(drafter, state: state, anchor: 4, seed: seed, depth: 3)
        drafter.discardRound(requestState: state)
        let drafts = Self.draft(drafter, state: state, anchor: 4, seed: seed, depth: 7)
        let direct = Self.directProposal(f, features: features, anchor: 4, depth: 7)
        #expect(drafts == direct.drafts)
        drafter.releaseRequestState(state)
    }

    /// Real checkpoint drafter against the CPU reference
    /// (`B/fixtures/sdk-mtp/real-*.safetensors`). Runs only when
    /// `INKLING2_DSPARK_WEIGHTS` (checkpoint dir) and `INKLING2_DSPARK_FIXTURE`
    /// are set; reads the drafter plus the target's embed, embed_norm and
    /// unembed (about 2.5 GB).
    @Test func realCheckpointMatchesReference() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let weights = environment["INKLING2_DSPARK_WEIGHTS"],
            let fixturePath = environment["INKLING2_DSPARK_FIXTURE"]
        else { return }
        MLX.GPU.set(memoryLimit: 8 << 30)
        MLX.GPU.set(cacheLimit: 1 << 30)
        let directory = URL(fileURLWithPath: weights)
        let model = try Inkling2DSparkModel.load(directory: directory)
        let shards = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.lastPathComponent.hasPrefix("model-") && $0.pathExtension == "safetensors" }
        var shared: [String: MLXArray] = [:]
        for shard in shards {
            for (key, value) in try loadArrays(url: shard)
            where [
                "model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.unembed.weight",
            ]
            .contains(key) {
                shared[key] = value
            }
        }
        let target = Inkling2DSparkArrayTarget(
            embedding: try #require(shared["model.llm.embed.weight"]),
            embeddingNorm: try #require(shared["model.llm.embed_norm.weight"]),
            unembedding: try #require(shared["model.llm.unembed.weight"]),
            eps: model.configuration.rmsNormEps)
        let head = model.foldHead(unembedding: target.unembedding)
        let f = Fixture(model: model, target: target, head: head, arrays: [:], cases: 0)
        let fixture = try loadArrays(url: URL(fileURLWithPath: fixturePath))
        let cases = fixture.keys.filter { $0.hasSuffix(".anchor") }.count
        #expect(cases > 0)
        var report: [String] = []
        for index in 0 ..< cases {
            let features = try #require(fixture["case\(index).features"])
            let anchor = try #require(fixture["case\(index).anchor"]).item(Int32.self)
            let expected = try #require(fixture["case\(index).drafts"]).asArray(Int32.self)
            let reference = try #require(fixture["case\(index).hidden"])
            let result = Self.directProposal(
                f, features: features, anchor: anchor, depth: expected.count)
            let cosine = Self.cosine(result.hidden[0], reference)
            let agree = zip(result.drafts, expected).prefix { $0 == $1 }.count
            report.append(
                "case \(index) n=\(features.dim(0)) cosine=\(cosine) agree=\(agree)/\(expected.count) "
                    + "swift=\(result.drafts) reference=\(expected)")
            #expect(cosine > 0.999, "case \(index) hidden cosine \(cosine)")
        }
        print(report.joined(separator: "\n"))
    }

    /// Drafter-only cost per round on the GPU with the real drafter and a
    /// random bf16 unembedding of the checkpoint's size. Runs only with
    /// `INKLING2_DSPARK_BENCH=1` and `INKLING2_DSPARK_WEIGHTS`.
    @Test func draftRoundCost() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["INKLING2_DSPARK_BENCH"] == "1",
            let weights = environment["INKLING2_DSPARK_WEIGHTS"]
        else { return }
        MLX.GPU.set(memoryLimit: 12 << 30)
        MLX.GPU.set(cacheLimit: 2 << 30)
        let model = try Inkling2DSparkModel.load(directory: URL(fileURLWithPath: weights))
        let c = model.configuration
        let unembedding = (MLXRandom.normal([c.vocabularySize, c.hiddenSize]) * 0.02)
            .asType(.bfloat16)
        let embedding = (MLXRandom.normal([c.paddedVocabularySize, c.hiddenSize]) * 0.02)
            .asType(.bfloat16)
        eval(unembedding, embedding)
        let target = Inkling2DSparkArrayTarget(
            embedding: embedding, embeddingNorm: MLXArray.ones([c.hiddenSize], dtype: .bfloat16),
            unembedding: unembedding, eps: c.rmsNormEps)
        let drafter = Inkling2DSparkDrafter(model: model, target: target)
        let context = Int(environment["INKLING2_DSPARK_BENCH_CONTEXT"] ?? "") ?? 1024
        var lines: [String] = []
        for rows in [1, 2, 4, 8] {
            let states = (0 ..< rows).map { _ in drafter.makeRequestState() }
            for state in states {
                try drafter.configureRequestState(state, maximumSequenceLength: context + 512)
                let features = (MLXRandom.normal([1, context, c.featureWidth]) * 2)
                    .asType(.bfloat16)
                drafter.observeCommittedTarget(
                    .init(tokens: MLXArray.zeros([1, context], dtype: .int32), hidden: features),
                    requestState: state)
            }
            func seeds() -> ([MLXArray], [MLXArray]) {
                (
                    (0 ..< rows).map { MLXArray([Int32(13 + $0)]).reshaped(1, 1) },
                    (0 ..< rows).map { _ in
                        MLXRandom.normal([1, 1, c.featureWidth]).asType(.bfloat16)
                    }
                )
            }
            func finalize() {
                for state in states {
                    drafter.finalizeRound(
                        requestState: state, confirmedInputTokens: 1,
                        committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
                        committedTargetHidden: MLXArray.zeros([1, 0, c.featureWidth]))
                }
            }
            func batched() {
                let (tokens, hidden) = seeds()
                let drafts = drafter.draftBlock(
                    tokens: tokens, hidden: hidden, requestStates: states, depth: c.draftTokens)
                eval([drafts] + states.flatMap { drafter.evaluationTargets(for: $0) })
                finalize()
            }
            func perRow() {
                let (tokens, hidden) = seeds()
                var targets: [MLXArray] = []
                for (index, state) in states.enumerated() {
                    var token = tokens[index]
                    var seed = hidden[index]
                    for _ in 0 ..< c.draftTokens {
                        let step = drafter.draftStep(
                            tokens: token, hidden: seed, shortlist: nil, requestState: state)
                        token = step.tokens.reshaped(1, 1)
                        seed = step.hidden
                        targets.append(step.tokens)
                    }
                    targets += drafter.evaluationTargets(for: state)
                }
                eval(targets)
                finalize()
            }
            for (name, body) in [("batched", batched), ("per_row", perRow)] {
                for _ in 0 ..< 3 { body() }
                let iterations = 10
                let start = Date()
                for _ in 0 ..< iterations { body() }
                let ms = Date().timeIntervalSince(start) * 1000 / Double(iterations)
                lines.append(
                    "rows=\(rows) context=\(context) \(name) ms_per_round=\(String(format: "%.2f", ms))"
                )
            }
            for state in states { drafter.releaseRequestState(state) }
        }
        print(lines.joined(separator: "\n"))
    }
}
