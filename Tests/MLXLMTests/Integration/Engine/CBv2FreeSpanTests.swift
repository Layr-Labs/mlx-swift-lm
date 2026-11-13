import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

// A real native Gemma forward with a hand-specified output language. Tokens
// 9...22 form a protected span; the target prefers 10 but the grammar requires 20.
private final class SpanConstraint: CBv2FreeSpanTokenConstraint, @unchecked Sendable {
    let mode = CBv2TokenConstraintMode.auto
    let initialState = 0
    let maxTokens = 20
    let fallbackTokenID = 255

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
        guard remainingTokens > 0 else { return [] }
        switch state {
        case 0, 4: return Array(0 ..< 255)
        case 1: return [20]
        case 2: return [21]
        case 3: return [22]
        default: return []
        }
    }

    func nextState(state: Int, tokenID: Int) -> Int? {
        switch (state, tokenID) {
        case (0, 9): return 1
        case (1, 20): return 2
        case (2, 21): return 3
        case (3, 22): return 4
        case (0, 0 ..< 255), (4, 0 ..< 255): return state
        default: return nil
        }
    }

    func isFreeSpan(state: Int) -> Bool { state == 0 || state == 4 }
    func isFreeSpanContinuation(state: Int, tokenID: Int) -> Bool {
        isFreeSpan(state: state) && tokenID != 9 && (0 ..< 255).contains(tokenID)
    }
}

private final class SpanDrafter: CBv2MTPDrafter {
    let mtpTargetIdentity: ObjectIdentifier?
    private(set) var starts: [Int] = []
    private let tokens: [Int]

    init(target: Gemma4TextModel, tokens: [Int]) {
        mtpTargetIdentity = ObjectIdentifier(target)
        self.tokens = tokens
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        let offsets = rows.map { $0.anchor - 3 + 1 }
        starts += offsets
        return CBv2ParityScriptCursor(baseIndices: offsets)
    }

    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        let cursor = prepared as! CBv2ParityScriptCursor
        let proposed = cursor.baseIndices.map { offset in
            Int32(self.tokens[min(offset + cursor.step, self.tokens.count - 1)])
        }
        cursor.step += 1
        return (MLXArray(proposed), hidden)
    }
}

@Suite("CBv2 free-span drafting", .serialized, .tags(.integration))
struct CBv2FreeSpanTests {
    @Test func protectedSpanReturnsToDraftingInTheSameResponse() async throws {
        let json = #"""
            {"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":6,
             "intermediate_size":128,"num_attention_heads":2,"head_dim":32,
             "global_head_dim":32,"num_key_value_heads":1,"num_kv_shared_layers":2,
             "layer_types":["sliding_attention","full_attention","full_attention",
                            "sliding_attention","sliding_attention","full_attention"],
             "sliding_window":16,"final_logit_softcapping":30.0,
             "tie_word_embeddings":false,"vocab_size":256,"vocab_size_per_layer_input":256,
             "rms_norm_eps":1e-6,"hidden_size_per_layer_input":0,"use_double_wide_mlp":false}
            """#
        let target = Gemma4TextModel(
            try JSONDecoder().decode(
                Gemma4TextConfiguration.self, from: Data(json.utf8)))
        stabilizeCBv2MTPGreedyCycleTarget(target)
        eval(target)
        let expected = [4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        for enabled in [false, true] {
            let drafter = SpanDrafter(target: target, tokens: expected)
            let kinds = target.cbv2LayerKinds
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(target), layerKinds: kinds,
                backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 26)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
                mtpDrafter: drafter,
                mtpConfig: .init(
                    enabled: true, maxDraftTokens: 3, fixedDraftTokens: 3,
                    verificationMode: .serialTarget, freeSpanSpeculation: enabled))
            let result = await cbv2SchedCollect(
                try engine.submit(
                    .init(
                        id: .init(1), promptTokens: [1, 2, 3], sampling: .init(temperature: 0),
                        maxTokens: 20, stopTokens: [255], tokenConstraint: SpanConstraint())))
            let metrics = try #require(engine.mtpMetricsSnapshot())
            await engine.shutdown()
            #expect(result.tokens == expected)
            #expect(result.finishReason == .length)
            if enabled {
                #expect(metrics.acceptedTokens > 0)
                #expect(drafter.starts.contains { $0 < 5 })
                #expect(drafter.starts.contains { $0 >= 9 })
                #expect(!drafter.starts.contains { (6 ..< 9).contains($0) })
            } else {
                #expect(metrics.rounds == 0 && drafter.starts.isEmpty)
            }
        }
    }
}
