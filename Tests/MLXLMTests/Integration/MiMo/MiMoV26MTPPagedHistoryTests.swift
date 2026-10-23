import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM

@Suite(.tags(.integration), .serialized)
struct MiMoV26MTPPagedHistoryTests {
    @Test func assistantPagesKeepEveryDraftAndCommittedReplayExact() throws {
        try Device.withDefaultDevice(.cpu) {
            for packed in [false, true] {
                let target: MiMoV26TextModel, predictor: MiMoV26MTP
                if packed {
                    target = try MiMoV26TextModel(MiMoV26MTPChecks.affineConfiguration())
                    try target.update(parameters: .unflattened(MiMoV26MTPChecks.fixtureWeights(target)
                        .mapValues { $0.asType(.bfloat16) }), verify: .all)
                    try packTarget(target)
                    predictor = try MiMoV26MTP(target: target)
                    try predictor.loadConvertedWeights(MiMoV26MTPChecks.packedFixture(predictor).stored)
                } else { (target, predictor) = try MiMoV26MTPChecks.fixture() }
                let assistants = try [MiMoV26MTPHistoryStorage.contiguous, .paged].map {
                    try MiMoV26MTPAssistant(target: target, predictor: predictor, historyStorage: $0)
                }
                #expect(try MiMoV26MTPAssistant(target: target, predictor: predictor).historyStorage == .contiguous)
                #expect(assistants[1].prefixCheckpointTensorDescriptors(targetInputCount: 8) == nil)
                for accepted in 0...3 {
                    let states = try assistants.map { assistant -> MiMoV26MTPState in
                        let state = assistant.makeRequestState() as! MiMoV26MTPState
                        try assistant.configureRequestState(state, maximumSequenceLength: 64)
                        return state
                    }
                    let targetCache = target.newCache(), prompt = (0..<19).map { 1 + $0 % 30 }
                    let output = try target.forward(inputIDs: ids(prompt), cache: targetCache)
                    for start in stride(from: 0, to: prompt.count, by: 7) {
                        let end = min(start + 7, prompt.count)
                        for (assistant, state) in zip(assistants, states) {
                            assistant.observeCommittedTarget(.init(tokens: ids(Array(prompt[start..<end])),
                                hidden: output.normalizedHiddenStates[0..., start..<end, 0...]), requestState: state)
                            try fence(assistant, state)
                        }
                        #expect(states[0].headInputCounts == states[1].headInputCounts)
                        #expect(states[1].materializedBytes > 0 && !states[1].hasUnmeasuredResidency)
                    }
                    let seed = Int(argMax(output.logits[0, -1], axis: -1).item(Int32.self))
                    let hidden = output.normalizedHiddenStates[0..., (-1)..., 0...]
                    let ordinary = try proposals(assistants[0], states[0], seed: seed, hidden: hidden)
                    let paged = try proposals(assistants[1], states[1], seed: seed, hidden: hidden)
                    #expect(ordinary == paged)
                    var confirmed: [MiMoV26TextOutput] = []
                    for token in [seed] + ordinary.tokens.prefix(accepted) {
                        confirmed.append(try target.forward(inputIDs: ids([token]), cache: targetCache))
                    }
                    let acceptedHidden = accepted == 0
                        ? MLXArray.zeros([1, 0, target.configuration.hiddenSize], dtype: target.activationDType)
                        : concatenated(confirmed.prefix(accepted).map(\.normalizedHiddenStates), axis: 1)
                    for (assistant, state) in zip(assistants, states) {
                        assistant.finalizeRound(requestState: state, confirmedInputTokens: accepted + 1,
                            committedDraftTokens: ids(Array(ordinary.tokens.prefix(accepted))), committedTargetHidden: acceptedHidden)
                        try fence(assistant, state)
                    }
                    let next = Int(argMax(confirmed.last!.logits[0, -1], axis: -1).item(Int32.self))
                    let nextOrdinary = try proposals(assistants[0], states[0], seed: next,
                        hidden: confirmed.last!.normalizedHiddenStates)
                    let nextPaged = try proposals(assistants[1], states[1], seed: next,
                        hidden: confirmed.last!.normalizedHiddenStates)
                    #expect(nextOrdinary == nextPaged)
                    #expect(states[0].headInputCounts == states[1].headInputCounts)
                    for (assistant, state) in zip(assistants, states) {
                        assistant.discardRound(requestState: state); try fence(assistant, state)
                        assistant.releaseRequestState(state)
                        #expect(state.isReleased && state.materializedBytes == 0)
                        #expect(assistant.evaluationTargets(for: state).isEmpty)
                    }
                }
            }
        }
    }

    @Test func pageAllocationDeclarationRetainsOldBoundsAndRoundsEveryPageSeparately() throws {
        let c = try MiMoV26MTPChecks.config(), limits = CBv2MTPAllocationLimits(maximumPrefillTokens: 64, maximumDraftTokens: 3)
        let original = try #require(MiMoV26MTPAssistant.boundedRequestAllocation(configuration: c, limits: limits))
        let pages = try #require(MiMoV26MTPAssistant.boundedRequestAllocation(configuration: c, limits: limits, historyStorage: .paged))
        #expect(pages.hostBytes == original.hostBytes)
        #expect(pages.resident.count == original.resident.count + 2)
        #expect(pages.working.count == original.working.count + 6)
        for (a, b) in zip(original.resident + original.working,
                          Array(pages.resident.prefix(original.resident.count)) + Array(pages.working.prefix(original.working.count - 1)) + Array(pages.working.suffix(1))) {
            #expect(a.logicalBytes == b.logicalBytes && a.allocationCount == b.allocationCount)
        }
        for buffer in pages.resident.suffix(2) { #expect(buffer.allocationCount == 12) }
        #expect(pages.working.dropLast().suffix(6).map(\.allocationCount) == [24, 24, 12, 12, 12, 12])
    }

    @Test func unchangedCPUCompatibleOrdinaryMTPComponentContractsStayGreen() throws {
        try Device.withDefaultDevice(.cpu) {
            // These two unchanged legacy cases execute a BF16 unquantized
            // routed target, whose GatherMM is GPU-only in the pinned MLX.
            // The new test above executes all three packed BF16 heads against
            // a correctly packed target; legacy GPU cases remain unchanged.
            let gpuTargetCases: Set<String> = [
                "actual affine4 group64 BF16 predictor format loads all90 tensors and executes all3 heads",
                "an explicitly unquantized first head does not break later affine head replacement",
            ]
            let cases = MiMoV26MTPChecks.cases()
            #expect(cases.filter { gpuTargetCases.contains($0.0) }.count == 2)
            for (name, check) in cases where !gpuTargetCases.contains(name) { try check() }
        }
    }

    private func packTarget(_ target: MiMoV26TextModel) throws {
        // Preserve the target's actual owning layer modules, as the stage
        // loader does. BF16 routed experts require the native packed path;
        // MLX's unquantized CPU gather-matmul supports Float32 only.
        var parents = Dictionary(uniqueKeysWithValues: target.namedModules())
        parents[""] = target
        for (path, leaf) in target.leafModules().flattened() {
            let expert = path.contains("switch_mlp")
            guard let packed = quantizeSingle(layer: leaf, groupSize: expert ? 32 : 64,
                bits: expert ? 4 : 8, mode: expert ? .mxfp4 : .affine) else { continue }
            var pieces = path.split(separator: ".").map(String.init); pieces.removeLast()
            while !pieces.isEmpty && parents[pieces.joined(separator: ".")] == nil { pieces.removeLast() }
            let parent = pieces.joined(separator: ".")
            let suffix = parent.isEmpty ? path : String(path.dropFirst(parent.count + 1))
            try parents[parent]!.update(modules: .unflattened([(suffix, packed)]), verify: [.noUnusedKeys])
        }
        target.freeze(); eval(target)
    }
    private func fence(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState) throws {
        eval(assistant.evaluationTargets(for: state))
        try assistant.requestStateDidFinishEvaluation(state)
    }
    private struct Proposal: Equatable { var tokens: [Int] = []; var bits: [Data] = [] }
    private func proposals(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState,
                           seed: Int, hidden: MLXArray) throws -> Proposal {
        var token = ids([seed]), feature = hidden, result = Proposal()
        for _ in 0..<3 {
            let next = assistant.draftStep(tokens: token, hidden: feature, shortlist: nil, requestState: state)
            eval(next.tokens, next.hidden); try fence(assistant, state)
            // Native token bits and returned target feature bits must both match.
            var bytes = next.tokens.asData(access: .copy).data
            bytes.append(next.hidden.asData(access: .copy).data); result.bits.append(bytes)
            result.tokens.append(Int(next.tokens.item(Int32.self)))
            token = next.tokens.reshaped(1, 1); feature = next.hidden
        }
        return result
    }
    private func ids(_ values: [Int]) -> MLXArray { MLXArray(values.map(Int32.init), [1, values.count]) }
}
