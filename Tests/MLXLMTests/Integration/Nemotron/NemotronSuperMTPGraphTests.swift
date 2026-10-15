import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The graph-only head against the existing eager reference on synthetic
/// NVFP4/FP8 fixtures. Equality is bitwise unless a tolerance is stated.
@Suite(.tags(.integration))
struct NemotronSuperMTPGraphTests {
    private static func rows(_ count: Int, seed: UInt64, scale: Float = 1) -> MLXArray {
        MLXRandom.normal([1, count, NemotronSuperMTPFixture.hidden], key: MLXRandom.key(seed))
            * scale
    }

    private static func sharedUp(_ head: NemotronSuperMTPModule) throws -> ModelOptCalibratedLinear
    {
        try #require(head.expertBlock.mixer.sharedExperts?.upProj as? ModelOptCalibratedLinear)
    }

    @Test func lazySharedNVFP4MatchesEagerIncludingOverflowRows() throws {
        let model = try NemotronSuperMTPFixture.target()
        let head = try NemotronSuperMTPFixture.head(for: model.languageModel, kind: .random)
        let projection = try Self.sharedUp(head)
        let budget = try NemotronSuperMTPLazyNVFP4.arrayBytes(
            rows: 3, inputDimensions: projection.inputDimensions,
            outputDimensions: projection.outputDimensions)
        var input = Self.rows(3, seed: 5)
        // amax / (6 * 0.01) > 464: ModelOpt's eager E4M3 cast returns NaN.
        input[0..., 1, 0] = MLXArray(Float(100))
        let lazy = try NemotronSuperMTPLazyNVFP4.forward(
            projection, input, admittedArrayBytes: budget)
        let eager = try projection.checkedForward(input)
        eval(lazy.value, lazy.valid, eager)
        #expect(lazy.valid.item(Bool.self))
        #expect(lazy.value.shape == eager.shape && lazy.value.dtype == eager.dtype)
        #expect(arrayEqual(lazy.value, eager, equalNAN: true).item(Bool.self))
        #expect(isNaN(lazy.value[0..., 1, 0...]).all().item(Bool.self))
        #expect(!isNaN(lazy.value[0..., 0, 0...]).any().item(Bool.self))
        #expect(throws: (any Error).self) {
            try NemotronSuperMTPLazyNVFP4.forward(projection, input, admittedArrayBytes: budget - 1)
        }
    }

    @Test(arguments: [Float.nan, 0, -1, Float.infinity])
    func invalidRuntimeCalibrationIsReportedAndPoisons(scale: Float) throws {
        let model = try NemotronSuperMTPFixture.target()
        let head = try NemotronSuperMTPFixture.head(for: model.languageModel, kind: .random)
        let projection = try Self.sharedUp(head)
        let current = Dictionary(uniqueKeysWithValues: projection.parameters().flattened())
        let shape = try #require(current["input_scale"]).shape
        try projection.update(
            parameters: .unflattened([("input_scale", MLXArray(scale).reshaped(shape))]),
            verify: [.noUnusedKeys, .shapeMismatch])
        let input = Self.rows(1, seed: 6)
        let lazy = try NemotronSuperMTPLazyNVFP4.forward(
            projection, input, admittedArrayBytes: 1 << 20)
        eval(lazy.value, lazy.valid)
        #expect(!lazy.valid.item(Bool.self))
        #expect(isNaN(lazy.value).all().item(Bool.self))
    }

    @Test func graphHeadMatchesEagerHeadWithReplay() throws {
        let model = try NemotronSuperMTPFixture.target()
        let language = model.languageModel
        let head = try NemotronSuperMTPFixture.head(for: language, kind: .random)
        let quote = try head.graphQuote(maximumHistoryTokens: 8, historyChunkRows: 8)
        let eagerCache = try NemotronSuperFP8KVCache(
            heads: 1, headDimension: 16, capacityTokens: 8, dtype: .float32)
        let history = try NemotronSuperFP8KVHistory(
            heads: 1, headDimension: 16, capacityTokens: 8, dtype: .float32)
        let fp8Head = try #require(language.lmHead as? ModelOptCalibratedLinear)
        let headBytes = try ModelOptLazyFP8Projection.requiredArrayWorkspaceBytes(
            rows: 1, inputDimensions: NemotronSuperMTPFixture.hidden,
            outputDimensions: NemotronSuperMTPFixture.vocabulary)
        var maximumLogitError: Float = 0
        for step in 0 ..< 6 {
            let hidden = Self.rows(1, seed: 100 + UInt64(step))
            let embedding = language.mtpEmbedding(MLXArray([Int32(step + 3)]).reshaped(1, 1))
            let eager = try head.forward(
                postNormHidden: hidden, nextEmbedding: embedding, cache: eagerCache,
                policy: .fp8, maximumCacheTokens: 8)
            let graph = try head.graphForward(
                postNormHidden: hidden, nextEmbedding: embedding, history: history, quote: quote)
            let lazyLogits = try fp8Head.lazyFP8Forward(
                graph.hidden, admittedArrayWorkspaceBytes: headBytes)
            let eagerLogits = language.logits(eager)
            eval(graph.hidden, graph.valid, eager, lazyLogits.value, eagerLogits)
            #expect(graph.valid.item(Bool.self) && lazyLogits.calibrationValid.item(Bool.self))
            #expect(arrayEqual(graph.hidden, eager).item(Bool.self), "draft hidden step \(step)")
            #expect(
                arrayEqual(history.committedCodes()[0], eagerCache.state[0]).item(Bool.self)
                    && arrayEqual(history.committedCodes()[1], eagerCache.state[1]).item(Bool.self))
            let error = abs(lazyLogits.value - eagerLogits).max().item(Float.self)
            maximumLogitError = max(maximumLogitError, error)
            #expect(
                argMax(lazyLogits.value, axis: -1).asArray(Int32.self)
                    == argMax(eagerLogits, axis: -1).asArray(Int32.self))
            if step == 3 {
                // Rewind one row, replay it, and expect the identical result.
                #expect(eagerCache.trim(1) == 1)
                try history.rewind(to: history.offset - 1)
                let replay = try head.graphForward(
                    postNormHidden: hidden, nextEmbedding: embedding, history: history,
                    quote: quote)
                let eagerReplay = try head.forward(
                    postNormHidden: hidden, nextEmbedding: embedding, cache: eagerCache,
                    policy: .fp8, maximumCacheTokens: 8)
                eval(replay.hidden, eagerReplay)
                #expect(arrayEqual(replay.hidden, graph.hidden).item(Bool.self))
                #expect(arrayEqual(eagerReplay, eager).item(Bool.self))
            }
        }
        print("SYNTHETIC super-mtp lazy-vs-eager FP8 head max |logit delta| = \(maximumLogitError)")
        #expect(maximumLogitError <= 1e-5)
    }

    @Test func undersizedQuoteRefusesBeforeGraphConstruction() throws {
        let model = try NemotronSuperMTPFixture.target()
        let head = try NemotronSuperMTPFixture.head(for: model.languageModel, kind: .random)
        let quote = try head.graphQuote(maximumHistoryTokens: 4, historyChunkRows: 2)
        let short = NemotronSuperMTPModule.GraphQuote(
            routedArrayBytes: quote.routedArrayBytes / 3,
            routedFC1GraphUnits: quote.routedFC1GraphUnits,
            routedFC2GraphUnits: quote.routedFC2GraphUnits,
            sharedArrayBytes: quote.sharedArrayBytes,
            attentionArrayBytes: quote.attentionArrayBytes,
            mixerArrayBytes: quote.mixerArrayBytes,
            historyChunkArrayBytes: quote.historyChunkArrayBytes, historyChunkRows: 2)
        let history = try NemotronSuperFP8KVHistory(
            heads: 1, headDimension: 16, capacityTokens: 4, dtype: .float32)
        let hidden = Self.rows(1, seed: 1)
        #expect(throws: (any Error).self) {
            try head.graphForward(
                postNormHidden: hidden, nextEmbedding: hidden, history: history, quote: short)
        }
        #expect(history.offset == 0 && history.allocatedTokens == 0)
        #expect(throws: NemotronSuperMTPModule.Failure.self) {
            try head.appendTrustedHistory(
                hidden: Self.rows(3, seed: 2), embedding: Self.rows(3, seed: 3),
                history: history, quote: quote)
        }
        // A full history refuses before any KV append.
        let full = try NemotronSuperFP8KVHistory(
            heads: 1, headDimension: 16, capacityTokens: 1, dtype: .float32)
        _ = try full.checkedUpdate(
            keys: MLXArray.zeros([1, 1, 1, 16]), values: MLXArray.zeros([1, 1, 1, 16]))
        #expect(throws: NemotronSuperMTPModule.Failure.self) {
            try head.graphForward(
                postNormHidden: hidden, nextEmbedding: hidden, history: full, quote: quote)
        }
        #expect(full.offset == 1)
    }

    @Test func trustedHistoryEqualsDraftForwardKV() throws {
        let model = try NemotronSuperMTPFixture.target()
        let head = try NemotronSuperMTPFixture.head(for: model.languageModel, kind: .random)
        let quote = try head.graphQuote(maximumHistoryTokens: 8, historyChunkRows: 4)
        let drafted = try NemotronSuperFP8KVHistory(
            heads: 1, headDimension: 16, capacityTokens: 8, dtype: .float32)
        let trusted = try NemotronSuperFP8KVHistory(
            heads: 1, headDimension: 16, capacityTokens: 8, dtype: .float32)
        for step in 0 ..< 5 {
            let hidden = Self.rows(1, seed: 300 + UInt64(step))
            let embedding = Self.rows(1, seed: 400 + UInt64(step))
            let output = try head.graphForward(
                postNormHidden: hidden, nextEmbedding: embedding, history: drafted, quote: quote)
            try head.appendTrustedHistory(
                hidden: hidden, embedding: embedding, history: trusted, quote: quote)
            eval([output.hidden] + drafted.innerState() + trusted.innerState())
        }
        for (a, b) in zip(drafted.committedCodes(), trusted.committedCodes()) {
            #expect(arrayEqual(a, b).item(Bool.self))
        }
    }
}
