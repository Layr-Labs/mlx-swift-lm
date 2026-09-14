import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Qwen4 bounded assistant priming", .serialized)
struct Qwen4ExpMTPPrimingTests {
    @Test func policyIsOptInAndBounded() {
        let key = Qwen4ExpMTPPriming.environmentFlag
        for raw in ["", "0", "-1", "8193", "1.5", "true", "999999999999999999999"] {
            #expect(Qwen4ExpMTPPriming.environmentChunkTokens(environment: [key: raw]) == 0)
        }
        #expect(Qwen4ExpMTPPriming.environmentChunkTokens(environment: [:]) == 0)
        #expect(Qwen4ExpMTPPriming.environmentChunkTokens(environment: [key: " 2048 "]) == 2048)
        #expect(Qwen4ExpMTPPriming.environmentChunkTokens(environment: [key: "8192"]) == 8192)
        #expect(!Qwen4ExpMTPPriming.needsChunking(backlog: [], chunkTokens: 1))
        let backlog = [MLXArray.zeros([1, 7], dtype: .int32)]
        #expect(!Qwen4ExpMTPPriming.needsChunking(backlog: backlog, chunkTokens: 8))
        #expect(Qwen4ExpMTPPriming.needsChunking(backlog: backlog, chunkTokens: 7))
        #expect(!Qwen4ExpMTPPriming.needsChunking(backlog: backlog, chunkTokens: 0))

        let skipKey = Qwen4ExpMTPPriming.skipColdPromptReplayEnvironmentFlag
        #expect(!Qwen4ExpMTPPriming.skipsColdPromptReplay(environment: [:]))
        for raw in ["", "0", "false", "off", "invalid"] {
            #expect(!Qwen4ExpMTPPriming.skipsColdPromptReplay(
                environment: [skipKey: raw]))
        }
        for raw in ["1", "true", "YES", " on "] {
            #expect(Qwen4ExpMTPPriming.skipsColdPromptReplay(
                environment: [skipKey: raw]))
        }
    }

    @Test func fragmentsCrossChunksWithoutDroppingOrDuplicatingSeed() {
        let tokens = [MLXArray([Int32(0), 1, 2]).reshaped([1, 3]),
                      MLXArray([Int32(3)]).reshaped([1, 1]),
                      MLXArray([Int32(4), 5, 6, 7, 8]).reshaped([1, 5])]
        let hidden = tokens.map { $0.asType(.float32).expandedDimensions(axis: -1) }
        var seen: [Int32] = []
        var widths: [Int] = []
        var materialized = 0
        let result = Qwen4ExpMTPPriming.forward(hidden: hidden, tokens: tokens, chunkTokens: 4,
            forward: { h, t in
                #expect(materialized == widths.count)
                widths.append(t.dim(1))
                seen += t.asArray(Int32.self)
                #expect(h.asArray(Float.self) == t.asArray(Int32.self).map(Float.init))
                return (h, h)
            }, materialize: { output in
                eval(output.mixed, output.residual)
                materialized += 1
            })
        #expect(widths == [4, 4, 1])
        #expect(seen == Array(Int32(0)...Int32(8)))
        #expect(result.residual.asArray(Float.self) == [8])
    }

    private func configuration() -> Qwen4ExpTextConfiguration {
        var c = Qwen4ExpTextConfiguration()
        c.hiddenSize = 32; c.hiddenLayers = 1; c.attentionHeads = 2; c.kvHeads = 1; c.headDim = 16
        c.vocabularySize = 64; c.maxPositionEmbeddings = 128
        c.fullAttentionInterval = 1; c.layerTypes = ["qwen_sparse_attention"]
        c.hcCount = 2; c.hcLowrank = 8; c.pleLayerIds = []; c.pleEmbedDim = 32
        c.indexerNHeads = 2; c.indexerKVHeads = 1; c.indexerHeadDim = 8
        c.indexerBudget = 16; c.indexerCompressRatio = 4
        c.numExperts = 1; c.numExpertsPerTok = 1
        c.sharedExpertIntermediateSize = 16; c.moeIntermediateSize = 16
        c.mropeSection = [2, 1, 1]; c.partialRotaryFactor = 0.25; c.mtpNumHiddenLayers = 1
        return c
    }

    private func fixture() throws -> (Qwen4ExpInlineMTPAssistant, Qwen4ExpInlineMTPAssistant) {
        let c = configuration()
        MLXRandom.seed(8421)
        let target = Qwen4ExpTextModel(c)
        let whole = try Qwen4ExpInlineMTPAssistant(configuration: c, blockSize: 3,
            target: target, verificationMode: .rectangularExact, primeChunkTokens: 0)
        let chunked = try Qwen4ExpInlineMTPAssistant(configuration: c, blockSize: 3,
            target: target, verificationMode: .rectangularExact, primeChunkTokens: 8)
        chunked.update(parameters: whole.parameters())
        eval(target.parameters(), whole.parameters(), chunked.parameters())
        return (whole, chunked)
    }

    private func coldAssistant(skip: Bool) throws -> Qwen4ExpInlineMTPAssistant {
        let c = configuration()
        MLXRandom.seed(8421)
        let target = Qwen4ExpTextModel(c)
        let assistant = try Qwen4ExpInlineMTPAssistant(
            configuration: c, blockSize: 3, target: target,
            verificationMode: .rectangularExact, primeChunkTokens: 0,
            skipColdPromptReplay: skip)
        eval(target.parameters(), assistant.parameters())
        return assistant
    }

    private func observe(_ assistant: Qwen4ExpInlineMTPAssistant,
                         _ state: any CBv2MTPRequestState, range: Range<Int>) {
        let tokens = MLXArray(range.map { Int32($0 % 64) }).reshaped([1, range.count])
        let hidden = MLXArray((range.lowerBound * 64..<range.upperBound * 64).map {
            Float($0 % 37) / 100
        }, [1, range.count, 64]).asType(.bfloat16)
        assistant.observeCommittedTarget(.init(tokens: tokens, hidden: hidden), requestState: state)
    }

    private func draft(_ assistant: Qwen4ExpInlineMTPAssistant,
                       _ state: any CBv2MTPRequestState, seed: Int32) -> MLXArray {
        let result = assistant.draftStep(tokens: MLXArray([seed]).reshaped([1, 1]),
            hidden: MLXArray.ones([1, 1, 64], dtype: .bfloat16) * 0.25,
            shortlist: nil, requestState: state)
        eval([result.tokens, result.hidden] + assistant.evaluationTargets(for: state))
        #expect(result.tokens.item(Int32.self) >= 0 && result.tokens.item(Int32.self) < 64)
        return result.hidden
    }

    private func settle(_ assistant: Qwen4ExpInlineMTPAssistant,
                        _ state: any CBv2MTPRequestState) {
        assistant.finalizeRound(requestState: state, confirmedInputTokens: 1,
            committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
            committedTargetHidden: MLXArray.zeros([1, 0, 64], dtype: .bfloat16))
    }

    private func near(_ lhs: MLXArray, _ rhs: MLXArray) {
        #expect(lhs.shape == rhs.shape)
        #expect(allClose(lhs.asType(.float32), rhs.asType(.float32), rtol: 0.02, atol: 0.02).item(Bool.self))
    }

    @Test func tinyAssistantWholeVsChunkedRolloverRestoreAndSuffix() throws {
        let (whole, chunked) = try fixture()
        let reference = whole.makeRequestState(), candidate = chunked.makeRequestState()
        defer { whole.releaseRequestState(reference); chunked.releaseRequestState(candidate) }
        for range in [0..<7, 7..<19, 19..<25] {
            observe(whole, reference, range: range)
            observe(chunked, candidate, range: range)
        }
        // Crosses dense/sparse QSA and multiple compressed-index frontiers.
        // Assistant numerical closeness is not a target-token parity claim.
        near(draft(whole, reference, seed: 25), draft(chunked, candidate, seed: 25))
        settle(whole, reference); settle(chunked, candidate)
        let a = try whole.snapshotRequestState(reference)
        let b = try chunked.snapshotRequestState(candidate)
        #expect(a.committedInputCount == 25 && b.committedInputCount == 25)
        near(a.cacheLayers[0].keys, b.cacheLayers[0].keys)
        near(a.cacheLayers[0].values, b.cacheLayers[0].values)
        let ai = try #require(a.cacheLayers[0].qwen4Indexer)
        let bi = try #require(b.cacheLayers[0].qwen4Indexer)
        #expect(ai.tokenCount == bi.tokenCount && ai.pooledIndexBlocks == bi.pooledIndexBlocks)
        near(ai.indexKeys, bi.indexKeys)
        #expect(ai.positionIds.asArray(Int64.self) == bi.positionIds.asArray(Int64.self))
        if let ap = ai.pooledIndexKeys, let bp = bi.pooledIndexKeys { near(ap, bp) }
        let restored = try chunked.restoreRequestState(from: b)
        defer { chunked.releaseRequestState(restored) }
        for state in [candidate, restored] { observe(chunked, state, range: 26..<39) }
        let next = draft(chunked, candidate, seed: 39)
        let resumed = draft(chunked, restored, seed: 39)
        #expect(next.asArray(Float.self) == resumed.asArray(Float.self))
        chunked.discardRound(requestState: candidate)
        chunked.discardRound(requestState: restored)
        #expect(candidate.stagedInputCount == 0 && restored.stagedInputCount == 0)
        #expect(candidate.committedInputCount == restored.committedInputCount)
        // Discard restores trusted input ownership; retrying the same transition
        // from two restored copies must still agree after the cursor rollback.
        let retried = draft(chunked, candidate, seed: 40)
        let retriedCopy = draft(chunked, restored, seed: 40)
        #expect(retried.asArray(Float.self) == retriedCopy.asArray(Float.self))
        settle(chunked, candidate); settle(chunked, restored)
    }

    @Test func coldSkipUsesOnlyCarryAndPreservesWarmCheckpointReplay() throws {
        let assistant = try coldAssistant(skip: true)
        let cold = assistant.makeRequestState()
        defer { assistant.releaseRequestState(cold) }
        observe(assistant, cold, range: 0..<25)

        // Capturing remains legacy-compatible before the first cold round.
        let stateSnapshot = try assistant.snapshotRequestState(cold)
        let prefix = try #require(assistant.capturePrefixCheckpoint(
            requestState: cold, targetInputCount: 25))
        let warm = try #require(assistant.restorePrefixCheckpoint(prefix))
        let resumed = try assistant.restoreRequestState(from: stateSnapshot)
        defer {
            assistant.releaseRequestState(warm)
            assistant.releaseRequestState(resumed)
        }

        _ = draft(assistant, cold, seed: 25)
        _ = draft(assistant, warm, seed: 25)
        _ = draft(assistant, resumed, seed: 25)
        settle(assistant, cold)
        settle(assistant, warm)
        settle(assistant, resumed)

        let coldSettled = try assistant.snapshotRequestState(cold)
        let warmSettled = try assistant.snapshotRequestState(warm)
        let resumedSettled = try assistant.snapshotRequestState(resumed)
        #expect(coldSettled.cacheLayers[0].offset == 1)
        #expect(coldSettled.committedInputCount == 25)
        #expect(coldSettled.logicalInputBase == 24)
        // Restored prefix and settled-state snapshots deliberately retain the
        // established whole-history replay; this flag is cold-only, not a warm
        // suffix-replay equivalence claim.
        #expect(warmSettled.cacheLayers[0].offset == 25)
        #expect(resumedSettled.cacheLayers[0].offset == 25)
        #expect(warmSettled.logicalInputBase == 0)
        #expect(resumedSettled.logicalInputBase == 0)
        #expect(warmSettled.committedInputCount == coldSettled.committedInputCount)
        #expect(resumedSettled.committedInputCount == coldSettled.committedInputCount)
    }

    @Test func coldSkipDiscardRestoresTrustedHistoryForRetry() throws {
        let assistant = try coldAssistant(skip: true)
        let state = assistant.makeRequestState()
        defer { assistant.releaseRequestState(state) }
        observe(assistant, state, range: 0..<8)

        _ = draft(assistant, state, seed: 8)
        assistant.discardRound(requestState: state)
        let rolledBack = try assistant.snapshotRequestState(state)
        #expect(rolledBack.cacheLayers[0].offset == 0)
        #expect(rolledBack.backlogTokens.reduce(0) { $0 + $1.dim(1) } == 8)
        #expect(rolledBack.committedInputCount == 8)

        _ = draft(assistant, state, seed: 9)
        settle(assistant, state)
        let retried = try assistant.snapshotRequestState(state)
        #expect(retried.cacheLayers[0].offset == 1)
        #expect(retried.committedInputCount == 9)
        #expect(retried.logicalInputBase == 8)
    }

    @Test func coldSkipStateIsRequestLocal() throws {
        let assistant = try coldAssistant(skip: true)
        let first = assistant.makeRequestState()
        let second = assistant.makeRequestState()
        defer {
            assistant.releaseRequestState(first)
            assistant.releaseRequestState(second)
        }
        observe(assistant, first, range: 0..<12)
        observe(assistant, second, range: 0..<4)
        _ = draft(assistant, first, seed: 12)
        _ = draft(assistant, second, seed: 4)
        settle(assistant, first)
        settle(assistant, second)
        let firstSnapshot = try assistant.snapshotRequestState(first)
        let secondSnapshot = try assistant.snapshotRequestState(second)
        #expect(firstSnapshot.cacheLayers[0].offset == 1)
        #expect(secondSnapshot.cacheLayers[0].offset == 1)
        #expect(firstSnapshot.committedInputCount == 12)
        #expect(secondSnapshot.committedInputCount == 4)
    }
}
