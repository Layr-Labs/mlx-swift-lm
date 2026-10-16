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

    @Test(arguments: [Float.nan, 0, -1, Float.infinity])
    func invalidRegisteredCalibrationRefusesBeforeHistoryMutation(scale: Float) throws {
        let model = try NemotronSuperMTPFixture.target()
        let head = try NemotronSuperMTPFixture.head(for: model.languageModel, kind: .random)
        let projection = try Self.sharedUp(head)
        let current = Dictionary(uniqueKeysWithValues: projection.parameters().flattened())
        let shape = try #require(current["input_scale"]).shape
        try projection.update(
            parameters: .unflattened([("input_scale", MLXArray(scale).reshaped(shape))]),
            verify: [.noUnusedKeys, .shapeMismatch])
        let quote = try head.graphQuote(maximumHistoryTokens: 4, historyChunkRows: 2)
        let history = try NemotronSuperFP8KVHistory(
            heads: 1, headDimension: 16, capacityTokens: 4, dtype: .float32)
        let hidden = Self.rows(1, seed: 6)
        #expect(throws: (any Error).self) {
            try head.graphForward(
                postNormHidden: hidden, nextEmbedding: hidden, history: history, quote: quote)
        }
        #expect(history.offset == 0 && history.allocatedTokens == 0)
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
        // Reference: the eager ModelOpt baselines. Graph: the fastpath wrappers.
        head.setModelOptExecution(.eagerReference)
        fp8Head.execution = .eagerReference
        for step in 0 ..< 6 {
            let hidden = Self.rows(1, seed: 100 + UInt64(step))
            let embedding = language.mtpEmbedding(MLXArray([Int32(step + 3)]).reshaped(1, 1))
            let eager = try head.forward(
                postNormHidden: hidden, nextEmbedding: embedding, cache: eagerCache,
                policy: .fp8, maximumCacheTokens: 8)
            let graph = try head.graphForward(
                postNormHidden: hidden, nextEmbedding: embedding, history: history, quote: quote)
            let exactLogits = try fp8Head.graphForward(graph.hidden)
            let eagerLogits = language.logits(eager)
            eval(graph.hidden, graph.valid, eager, exactLogits, eagerLogits)
            #expect(graph.valid.item(Bool.self))
            #expect(arrayEqual(exactLogits, eagerLogits).item(Bool.self))
            #expect(arrayEqual(graph.hidden, eager).item(Bool.self), "draft hidden step \(step)")
            #expect(
                arrayEqual(history.committedCodes()[0], eagerCache.state[0]).item(Bool.self)
                    && arrayEqual(history.committedCodes()[1], eagerCache.state[1]).item(Bool.self))
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
    }

    @Test func undersizedQuoteRefusesBeforeGraphConstruction() throws {
        let model = try NemotronSuperMTPFixture.target()
        let head = try NemotronSuperMTPFixture.head(for: model.languageModel, kind: .random)
        let quote = try head.graphQuote(maximumHistoryTokens: 4, historyChunkRows: 2)
        let short = NemotronSuperMTPModule.GraphQuote(
            routedArrayBytes: quote.routedArrayBytes / 3,
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
