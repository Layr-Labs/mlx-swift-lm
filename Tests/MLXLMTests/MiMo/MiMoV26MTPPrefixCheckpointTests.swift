// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

/// Prepared native component regressions, UNRUN. Reuses the real tiny target
/// and all three trained-head module fixtures, never a scripted/fake drafter.
/// Common's complete-store/target-adoption tests are a separate integration gate.
final class MiMoV26MTPPrefixCheckpointTests: XCTestCase {
    private func fixture() throws -> (MiMoV26TextModel, MiMoV26MTP, MiMoV26MTPAssistant) {
        var object = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(MiMoV26MTPChecks.config())) as! [String: Any]
        object["dtype"] = "bfloat16"
        object["moe_router_dtype"] = "bfloat16"
        let config = try JSONDecoder().decode(MiMoV26Configuration.self,
            from: JSONSerialization.data(withJSONObject: object))
        let target = try MiMoV26TextModel(config)
        try target.update(parameters: .unflattened(
            MiMoV26MTPChecks.fixtureWeights(target).mapValues { $0.asType(.bfloat16) }), verify: .all)
        let predictor = try MiMoV26MTP(target: target)
        try predictor.loadConvertedWeights(
            MiMoV26MTPChecks.fixtureWeights(predictor, prefix: "mtp.").mapValues { $0.asType(.bfloat16) })
        return (target, predictor, try MiMoV26MTPAssistant(target: target, predictor: predictor))
    }
    private func prompt(_ count: Int = 17) -> [Int] { (0..<count).map { 1 + ($0 * 7) % 31 } }
    private func ids(_ values: [Int]) -> MLXArray { MLXArray(values.map(Int32.init), [1, values.count]) }
    private func state(_ assistant: MiMoV26MTPAssistant, prompt: [Int]? = nil) throws -> MiMoV26MTPState {
        let result = assistant.makeRequestState() as! MiMoV26MTPState
        try assistant.configureRequestState(result, maximumSequenceLength: 64)
        if let prompt { try assistant.installPrefixCaptureContext(requestState: result, promptTokens: prompt) }
        return result
    }
    private func finish(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState) throws {
        try withError { eval(assistant.evaluationTargets(for: state)) }
        try assistant.requestStateDidFinishEvaluation(state)
    }
    private func observe(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState,
                         tokens: [Int], hidden: MLXArray, range: Range<Int>, chunk: Int = 1) throws {
        for start in stride(from: range.lowerBound, to: range.upperBound, by: chunk) {
            let end = min(range.upperBound, start + chunk)
            assistant.observeCommittedTarget(.init(tokens: ids(Array(tokens[start..<end])),
                hidden: hidden[0..., start..<end, 0...]), requestState: state)
            try finish(assistant, state)
        }
    }
    private func capture(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState,
                         at position: Int = 11) throws -> any CBv2MTPPrefixCheckpoint {
        let value = try XCTUnwrap(assistant.capturePrefixCheckpoint(requestState: state, targetInputCount: position))
        try withError { eval(value.evaluationTargets) }
        return value
    }
    private func bytes(_ arrays: [MLXArray]) -> [Data] {
        arrays.map { $0.asData(access: .copy).data }
    }
    private func proposals(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState,
                           seed: Int, hidden: MLXArray, depth: Int) throws -> [Int] {
        var token = ids([seed]), feature = hidden, result: [Int] = []
        for _ in 0..<depth {
            let output = assistant.draftStep(tokens: token, hidden: feature, shortlist: nil, requestState: state)
            try withError { eval([output.tokens, output.hidden] + assistant.evaluationTargets(for: state)) }
            try assistant.requestStateDidFinishEvaluation(state)
            result.append(Int(output.tokens.item(Int32.self)))
            token = output.tokens.reshaped([1, 1]); feature = output.hidden
        }
        return result
    }

    func testDescriptorsAreAllThreeNativeHeadsTailTokensAndExactMetadata() throws {
        let (target, _, assistant) = try fixture()
        defer { withExtendedLifetime(target) {} }
        XCTAssertEqual(assistant.mtpTargetIdentity, ObjectIdentifier(target))
        let descriptors = try XCTUnwrap(assistant.prefixCheckpointTensorDescriptors(targetInputCount: 4))
        XCTAssertEqual(descriptors.count, 9)
        for depth in 0..<3 {
            XCTAssertEqual(descriptors[2 * depth].role, .assistantKeys)
            XCTAssertEqual(descriptors[2 * depth + 1].role, .assistantValues)
            XCTAssertEqual(descriptors[2 * depth].layer, depth)
            XCTAssertEqual(descriptors[2 * depth].shape, [1, 1, 3 - depth, 4])
            XCTAssertEqual(descriptors[2 * depth + 1].shape, [1, 1, 3 - depth, 2])
            XCTAssertEqual(descriptors[2 * depth].dtype, .bfloat16)
            XCTAssertEqual(descriptors[2 * depth + 1].dtype, .bfloat16)
        }
        XCTAssertEqual(descriptors[6].shape, [1, 3, 4])
        XCTAssertEqual(descriptors[7].shape, [1, 4])
        XCTAssertEqual(descriptors[8].shape, [3, 7])
        XCTAssertEqual(descriptors[8].dtype, .int64)
        XCTAssertNil(assistant.prefixCheckpointTensorDescriptors(targetInputCount: 3))
        XCTAssertNil(assistant.prefixCheckpointTensorDescriptors(targetInputCount: 64))
        XCTAssertNil(assistant.prefixCheckpointTensorDescriptors(targetInputCount: Int.max))
    }

    func testContextIsOneTimeOwnerBoundAndBeforeObservationWithoutCallerCOW() throws {
        let (target, _, assistant) = try fixture()
        defer { withExtendedLifetime(target) {} }
        var actual = prompt()
        let original = actual
        let output = try target.forward(inputIDs: ids(actual))
        let bound = try state(assistant, prompt: actual)
        actual[0] = 30
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: bound, promptTokens: original))
        try observe(assistant, bound, tokens: original, hidden: output.normalizedHiddenStates, range: 0..<11)
        XCTAssertNotNil(assistant.capturePrefixCheckpoint(requestState: bound, targetInputCount: 11))
        let late = try state(assistant)
        try observe(assistant, late, tokens: original, hidden: output.normalizedHiddenStates, range: 0..<1)
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: late, promptTokens: original))
        let (foreignTarget, _, foreign) = try fixture()
        defer { withExtendedLifetime(foreignTarget) {} }
        XCTAssertEqual(assistant.mtpTargetIdentity, ObjectIdentifier(target))
        XCTAssertEqual(foreign.mtpTargetIdentity, ObjectIdentifier(foreignTarget))
        XCTAssertNotEqual(assistant.mtpTargetIdentity, foreign.mtpTargetIdentity)
        XCTAssertThrowsError(try foreign.installPrefixCaptureContext(requestState: bound, promptTokens: original))
        let invalid = try state(assistant)
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: invalid, promptTokens: [0, 1, -1, 2, 3]))
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: invalid, promptTokens: [0, 1, 32, 2, 3]))
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: invalid, promptTokens: []))
        // A valid short request is not refused merely because no interior
        // boundary can contain all three trained heads yet.
        XCTAssertNoThrow(try assistant.installPrefixCaptureContext(requestState: invalid, promptTokens: [0, 1, 2, 3]))
        XCTAssertNil(assistant.capturePrefixCheckpoint(requestState: invalid, targetInputCount: 4))
        assistant.releaseRequestState(bound); assistant.releaseRequestState(late); assistant.releaseRequestState(invalid)
    }

    func testCaptureRequiresActualSettledInteriorBoundaryAndMatchingObservedTokens() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let request = try state(assistant, prompt: tokens)
        assistant.observeCommittedTarget(.init(tokens: ids(Array(tokens[0..<11])),
            hidden: output.normalizedHiddenStates[0..., 0..<11, 0...]), requestState: request)
        XCTAssertNil(assistant.capturePrefixCheckpoint(requestState: request, targetInputCount: 11))
        try finish(assistant, request)
        XCTAssertNotNil(assistant.capturePrefixCheckpoint(requestState: request, targetInputCount: 11))
        XCTAssertNil(assistant.capturePrefixCheckpoint(requestState: request, targetInputCount: 10))
        try observe(assistant, request, tokens: tokens, hidden: output.normalizedHiddenStates, range: 11..<17)
        XCTAssertNil(assistant.capturePrefixCheckpoint(requestState: request, targetInputCount: 17))
        let wrong = try state(assistant, prompt: tokens)
        var different = tokens; different[3] = different[3] == 1 ? 2 : 1
        let otherOutput = try target.forward(inputIDs: ids(different))
        try observe(assistant, wrong, tokens: different, hidden: otherOutput.normalizedHiddenStates, range: 0..<11)
        XCTAssertNil(assistant.capturePrefixCheckpoint(requestState: wrong, targetInputCount: 11))
        assistant.releaseRequestState(request); assistant.releaseRequestState(wrong)
    }

    func testNoDraftRoundCarryOrPendingStateCanBecomeHistoricalPrompt() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let request = try state(assistant, prompt: tokens)
        try observe(assistant, request, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
        _ = try proposals(assistant, request, seed: 7,
            hidden: output.normalizedHiddenStates[0..., 10..<11, 0...], depth: 1)
        XCTAssertNil(assistant.capturePrefixCheckpoint(requestState: request, targetInputCount: 11))
        assistant.finalizeRound(requestState: request, confirmedInputTokens: 1,
            committedDraftTokens: ids([]), committedTargetHidden: MLXArray.zeros([1, 0, 4], dtype: .bfloat16))
        try finish(assistant, request)
        XCTAssertNotNil(request.pendingLastToken)
        XCTAssertNil(assistant.capturePrefixCheckpoint(requestState: request, targetInputCount: 11))
        assistant.discardRound(requestState: request)
        try finish(assistant, request)
        XCTAssertNil(assistant.capturePrefixCheckpoint(requestState: request, targetInputCount: 11))
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: request, promptTokens: tokens))
        assistant.releaseRequestState(request)
    }

    func testWrappedPhysicalRingAndAllTensorBytesSurviveEncodeDecodeRestore() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        for chunk in [1, 2, 7] {
            let donor = try state(assistant, prompt: tokens)
            try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11, chunk: chunk)
            let checkpoint = try capture(assistant, donor)
            let arrays = try XCTUnwrap(assistant.encodePrefixCheckpoint(checkpoint))
            let expected = bytes(arrays) // independent host byte copies, before adoption
            let imported = try XCTUnwrap(assistant.decodePrefixCheckpoint(tensors: arrays, prefixTokens: Array(tokens.prefix(11))))
            let restored = try XCTUnwrap(assistant.restorePrefixCheckpoint(imported) as? MiMoV26MTPState)
            try assistant.configureRequestState(restored, maximumSequenceLength: 64)
            try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: tokens)
            try finish(assistant, restored)
            let recaptured = try capture(assistant, restored)
            XCTAssertEqual(bytes(recaptured.evaluationTargets), expected, "chunk \(chunk)")
            XCTAssertEqual(restored.headInputCounts, [10, 9, 8])
            XCTAssertEqual(restored.cache?.nextTokenPositions, [11, 11, 11])
            XCTAssertEqual(restored.retainedFeatureRows, 3)
            XCTAssertEqual(restored.committedInputCount, 11)
            XCTAssertEqual(restored.stagedInputCount, 0)
            XCTAssertFalse(restored.hasRequiredRestoredSuffix)
            XCTAssertNil(restored.pendingTokens); XCTAssertNil(restored.pendingHidden); XCTAssertNil(restored.pendingLastToken)
            assistant.releaseRequestState(restored); assistant.releaseRequestState(donor)
        }
    }

    func testImportRejectsWrongTokensDtypesShapesHeadCountAndRingMetadata() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let donor = try state(assistant, prompt: tokens)
        try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
        let checkpoint = try capture(assistant, donor), arrays = checkpoint.evaluationTargets
        let prefix = Array(tokens.prefix(11))
        var wrongTokens = prefix; wrongTokens[0] = 0
        XCTAssertNil(assistant.decodePrefixCheckpoint(tensors: arrays, prefixTokens: wrongTokens))
        XCTAssertNil(assistant.decodePrefixCheckpoint(tensors: Array(arrays.dropLast()), prefixTokens: prefix))
        for index in 0..<8 {
            var changed = arrays; changed[index] = arrays[index].asType(index == 7 ? .int64 : .float32)
            XCTAssertNil(assistant.decodePrefixCheckpoint(tensors: changed, prefixTokens: prefix))
            changed = arrays; changed[index] = arrays[index].reshaped([-1])
            XCTAssertNil(assistant.decodePrefixCheckpoint(tensors: changed, prefixTokens: prefix))
        }
        let metadata = arrays[8].asArray(Int64.self)
        let invalidFields: [(Int, Int64)] = [(0, 0), (1, 1), (2, 4), (3, 128), (4, 1), (5, 0), (6, 1)]
        for depth in 0..<3 {
            for (column, invalid) in invalidFields {
                var values = metadata; values[depth * 7 + column] = invalid
                var changed = arrays; changed[8] = MLXArray(values, [3, 7])
                XCTAssertNil(assistant.decodePrefixCheckpoint(tensors: changed, prefixTokens: prefix),
                             "depth \(depth), field \(column)")
            }
        }
        assistant.releaseRequestState(donor)
    }

    func testTokenAndMetadataValidationNeverEvaluatesLazyImportedWitnesses() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let donor = try state(assistant, prompt: tokens)
        try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
        let checkpoint = try capture(assistant, donor)
        let arrays = checkpoint.evaluationTargets, prefix = Array(tokens.prefix(11))
        var lazy = arrays
        lazy[7] = mimoV26MTPCopy(arrays[7])
        XCTAssertNil(try lazy[7].evaluatedBufferInfo())
        XCTAssertNil(assistant.decodePrefixCheckpoint(tensors: lazy, prefixTokens: prefix))
        XCTAssertNil(try lazy[7].evaluatedBufferInfo())
        try withError { eval(lazy) }
        XCTAssertNotNil(assistant.decodePrefixCheckpoint(tensors: lazy, prefixTokens: prefix))
        lazy = arrays; lazy[8] = mimoV26MTPCopy(arrays[8])
        XCTAssertNil(try lazy[8].evaluatedBufferInfo())
        XCTAssertNil(assistant.decodePrefixCheckpoint(tensors: lazy, prefixTokens: prefix))
        XCTAssertNil(try lazy[8].evaluatedBufferInfo())
        try withError { eval(lazy) }
        assistant.releaseRequestState(donor)
    }

    func testRestoreIsCurrentOwnerBoundAndReloadInvalidatesOldCheckpoint() throws {
        let (target, predictor, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let donor = try state(assistant, prompt: tokens)
        try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
        let checkpoint = try capture(assistant, donor)
        let foreign = try MiMoV26MTPAssistant(target: target, predictor: predictor)
        XCTAssertNil(foreign.encodePrefixCheckpoint(checkpoint))
        XCTAssertNil(foreign.restorePrefixCheckpoint(checkpoint))
        // Authentic import is separately rebound to the current owner; it is
        // not permission to restore another owner's in-memory checkpoint.
        let imported = try XCTUnwrap(foreign.decodePrefixCheckpoint(
            tensors: checkpoint.evaluationTargets, prefixTokens: Array(tokens.prefix(11))))
        let candidate = try XCTUnwrap(foreign.restorePrefixCheckpoint(imported) as? MiMoV26MTPState)
        try finish(foreign, candidate); foreign.releaseRequestState(candidate)
        assistant.releaseRequestState(donor)
        try predictor.loadConvertedWeights(
            MiMoV26MTPChecks.fixtureWeights(predictor, prefix: "mtp.").mapValues { $0.asType(.bfloat16) })
        XCTAssertNil(assistant.encodePrefixCheckpoint(checkpoint))
        XCTAssertNil(assistant.restorePrefixCheckpoint(checkpoint))
    }

    func testPristineRestoreContextRejectsChangedPrefixAndLateInstall() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let donor = try state(assistant, prompt: tokens)
        try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
        let checkpoint = try capture(assistant, donor)
        let restored = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint) as? MiMoV26MTPState)
        try assistant.configureRequestState(restored, maximumSequenceLength: 64)
        try finish(assistant, restored) // installer also works after actual copy completion
        var changed = tokens; changed[0] = 0
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: changed))
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: Array(tokens.prefix(11))))
        try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: tokens)
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: tokens))
        let late = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint) as? MiMoV26MTPState)
        try assistant.configureRequestState(late, maximumSequenceLength: 64)
        try finish(assistant, late)
        try observe(assistant, late, tokens: tokens, hidden: output.normalizedHiddenStates, range: 11..<12)
        XCTAssertThrowsError(try assistant.installPrefixCaptureContext(requestState: late, promptTokens: tokens))
        assistant.releaseRequestState(late); assistant.releaseRequestState(restored); assistant.releaseRequestState(donor)
    }

    func testActualSuffixAndDepthOneThroughThreeMatchUncachedNativeContinuation() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let donor = try state(assistant, prompt: tokens)
        try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
        let checkpoint = try capture(assistant, donor)
        for depth in 1...3 {
            let cold = try state(assistant)
            try observe(assistant, cold, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
            let restored = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint) as? MiMoV26MTPState)
            try assistant.configureRequestState(restored, maximumSequenceLength: 64)
            try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: tokens)
            try finish(assistant, restored)
            for request in [cold, restored] {
                try observe(assistant, request, tokens: tokens, hidden: output.normalizedHiddenStates, range: 11..<17)
            }
            XCTAssertTrue(restored.hasRequiredRestoredSuffix)
            XCTAssertEqual(bytes(restored.cache!.innerState()), bytes(cold.cache!.innerState()))
            let seed = Int(argMax(output.logits[0, -1], axis: -1).item(Int32.self))
            let hidden = output.normalizedHiddenStates[0..., (-1)..., 0...]
            XCTAssertEqual(try proposals(assistant, restored, seed: seed, hidden: hidden, depth: depth),
                           try proposals(assistant, cold, seed: seed, hidden: hidden, depth: depth))
            XCTAssertEqual(restored.headProposalCounts, cold.headProposalCounts)
            XCTAssertEqual(bytes(restored.round!.cache.innerState()), bytes(cold.round!.cache.innerState()))
            assistant.discardRound(requestState: restored); assistant.discardRound(requestState: cold)
            try finish(assistant, restored); try finish(assistant, cold)
            assistant.releaseRequestState(restored); assistant.releaseRequestState(cold)
        }
        assistant.releaseRequestState(donor)
    }


    func testFirstSuffixAndEveryDraftDepthPreservePartialAndWrappedPrefixState() throws {
        let (target, _, assistant) = try fixture()
        defer { withExtendedLifetime(target) {} }
        XCTAssertEqual(assistant.mtpTargetIdentity, ObjectIdentifier(target))
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        XCTAssertEqual(target.configuration.slidingWindow, 3)
        // P=4 has live head lengths [3,2,1]; P=11 has three wrapped rings.
        for boundary in [4, 11] {
            let donor = try state(assistant, prompt: tokens)
            try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates,
                        range: 0..<boundary)
            let checkpoint = try capture(assistant, donor, at: boundary)
            let published = bytes(try XCTUnwrap(assistant.encodePrefixCheckpoint(checkpoint)))
            XCTAssertEqual(published.count, 9)
            for depth in 1...3 {
                // No state/proposal from a preceding depth is reused.
                let cold = try state(assistant, prompt: tokens)
                try observe(assistant, cold, tokens: tokens, hidden: output.normalizedHiddenStates,
                            range: 0..<boundary)
                let restored = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint) as? MiMoV26MTPState)
                try assistant.configureRequestState(restored, maximumSequenceLength: 64)
                try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: tokens)
                try finish(assistant, restored)

                let coldBefore = bytes(try capture(assistant, cold, at: boundary).evaluationTargets)
                let restoredBefore = bytes(try capture(assistant, restored, at: boundary).evaluationTargets)
                XCTAssertEqual(coldBefore.count, 9); XCTAssertEqual(restoredBefore.count, 9)
                // Data(access: .copy) snapshots were taken independently, not
                // by comparing two aliases of the same checkpoint array.
                XCTAssertEqual(coldBefore, published, "cold boundary P=\(boundary), depth=\(depth)")
                XCTAssertEqual(restoredBefore, coldBefore, "restored boundary P=\(boundary), depth=\(depth)")
                for request in [cold, restored] {
                    XCTAssertEqual(request.headInputCounts, (0..<3).map { boundary - $0 - 1 })
                    XCTAssertEqual(request.cache?.nextTokenPositions, [boundary, boundary, boundary])
                    XCTAssertEqual(request.committedInputCount, boundary)
                    XCTAssertEqual(request.retainedFeatureRows, 3)
                    XCTAssertEqual(request.stagedInputCount, 0)
                    XCTAssertNil(request.round)
                    XCTAssertNil(request.pendingTokens); XCTAssertNil(request.pendingHidden)
                    XCTAssertNil(request.pendingLastToken)
                }
                XCTAssertFalse(restored.hasRequiredRestoredSuffix)

                // Exactly one real target suffix input; at W=3 this cannot
                // overwrite all restored history or the complete target tail.
                for request in [cold, restored] {
                    try observe(assistant, request, tokens: tokens, hidden: output.normalizedHiddenStates,
                                range: boundary..<boundary + 1)
                }
                let coldAfter = bytes(try capture(assistant, cold, at: boundary + 1).evaluationTargets)
                let restoredAfter = bytes(try capture(assistant, restored, at: boundary + 1).evaluationTargets)
                XCTAssertEqual(coldAfter.count, 9); XCTAssertEqual(restoredAfter.count, 9)
                XCTAssertEqual(restoredAfter, coldAfter, "first suffix P=\(boundary), depth=\(depth)")
                XCTAssertTrue(restored.hasRequiredRestoredSuffix)
                for request in [cold, restored] {
                    XCTAssertEqual(request.headInputCounts, (0..<3).map { boundary - $0 })
                    XCTAssertEqual(request.cache?.nextTokenPositions,
                                   [boundary + 1, boundary + 1, boundary + 1])
                    XCTAssertEqual(request.committedInputCount, boundary + 1)
                    XCTAssertNil(request.pendingTokens); XCTAssertNil(request.pendingHidden)
                    XCTAssertNil(request.pendingLastToken)
                }

                let seedArray = argMax(output.logits[0, boundary], axis: -1).asType(.int32)
                try withError { eval(seedArray) }
                let seed = Int(seedArray.item(Int32.self))
                let feature = output.normalizedHiddenStates[0..., boundary..<boundary + 1, 0...]
                let restoredProposals = try proposals(assistant, restored, seed: seed, hidden: feature, depth: depth)
                let coldProposals = try proposals(assistant, cold, seed: seed, hidden: feature, depth: depth)
                XCTAssertEqual(restoredProposals, coldProposals, "first drafts P=\(boundary), depth=\(depth)")
                XCTAssertEqual(restored.headProposalCounts, (0..<3).map { $0 < depth ? 1 : 0 })
                XCTAssertEqual(restored.headProposalCounts, cold.headProposalCounts)
                let restoredRound = try XCTUnwrap(restored.round), coldRound = try XCTUnwrap(cold.round)
                XCTAssertEqual(restoredRound.baseCount, boundary + 1)
                XCTAssertEqual(restoredRound.baseCount, coldRound.baseCount)
                XCTAssertEqual(restoredRound.cache.consumedTokenCounts, coldRound.cache.consumedTokenCounts)
                XCTAssertEqual(restoredRound.cache.nextTokenPositions, coldRound.cache.nextTokenPositions)
                XCTAssertEqual(bytes(restoredRound.cache.innerState()), bytes(coldRound.cache.innerState()))
                XCTAssertEqual(bytes(restoredRound.inputs), bytes(coldRound.inputs))
                XCTAssertEqual(restored.stagedInputCount, depth)
                XCTAssertEqual(cold.stagedInputCount, depth)
                assistant.discardRound(requestState: restored); assistant.discardRound(requestState: cold)
                try finish(assistant, restored); try finish(assistant, cold)
                assistant.releaseRequestState(restored); assistant.releaseRequestState(cold)
            }
            assistant.releaseRequestState(donor)
        }
    }

    func testRetainedPhysicalRowBitCorruptionSurvivesFirstSuffixAndDraft() throws {
        let (target, _, assistant) = try fixture()
        defer { withExtendedLifetime(target) {} }
        let tokens = prompt(), boundary = 11
        let output = try target.forward(inputIDs: ids(tokens))
        let donor = try state(assistant, prompt: tokens)
        try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates,
                    range: 0..<boundary)
        let checkpoint = try capture(assistant, donor, at: boundary)
        let arrays = try XCTUnwrap(assistant.encodePrefixCheckpoint(checkpoint))
        let original = bytes(arrays)
        XCTAssertEqual(arrays.count, 9)
        XCTAssertEqual(arrays[0].dtype, .bfloat16)
        XCTAssertEqual(arrays[0].shape, [1, 1, 3, 4])
        let metadata = arrays[8].asArray(Int64.self)
        let window = arrays[0].dim(2), index = Int(metadata[5])
        XCTAssertTrue((1...window).contains(index))
        let retainedRow = (index - 1) % window
        XCTAssertNotEqual(retainedRow, index % window) // next suffix write
        XCTAssertNotEqual(retainedRow, (index + 1) % window) // first depth-0 draft write
        let byteOffset = retainedRow * arrays[0].dim(3) * MemoryLayout<UInt16>.size
        var changedKeyBytes = original[0]
        let changedByte = try XCTUnwrap(changedKeyBytes.indices.contains(byteOffset) ? byteOffset : nil)
        changedKeyBytes[changedByte] ^= 1
        var changedArrays = arrays
        // A transport-oracle negative only, not a model-generated value or a
        // bypass of Common's authentication. Raw bytes avoid any BF16 cast.
        changedArrays[0] = try withError { MLXArray(changedKeyBytes, arrays[0].shape, dtype: .bfloat16) }
        try withError { eval(changedArrays) }
        let corruptedCheckpoint = try XCTUnwrap(assistant.decodePrefixCheckpoint(
            tensors: changedArrays, prefixTokens: Array(tokens.prefix(boundary))))
        let clean = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint) as? MiMoV26MTPState)
        let corrupted = try XCTUnwrap(assistant.restorePrefixCheckpoint(corruptedCheckpoint) as? MiMoV26MTPState)
        for request in [clean, corrupted] {
            try assistant.configureRequestState(request, maximumSequenceLength: 64)
            try assistant.installPrefixCaptureContext(requestState: request, promptTokens: tokens)
            try finish(assistant, request)
        }
        var changedBefore = original
        changedBefore[0] = changedKeyBytes
        XCTAssertEqual(bytes(try capture(assistant, clean, at: boundary).evaluationTargets), original)
        XCTAssertEqual(bytes(try capture(assistant, corrupted, at: boundary).evaluationTargets), changedBefore)
        XCTAssertNotEqual(changedBefore, original)

        for request in [clean, corrupted] {
            try observe(assistant, request, tokens: tokens, hidden: output.normalizedHiddenStates,
                        range: boundary..<boundary + 1)
        }
        let cleanAfter = bytes(try capture(assistant, clean, at: boundary + 1).evaluationTargets)
        let corruptedAfter = bytes(try capture(assistant, corrupted, at: boundary + 1).evaluationTargets)
        var expectedAfter = cleanAfter
        expectedAfter[0][changedByte] ^= 1
        XCTAssertEqual(corruptedAfter, expectedAfter) // exactly the retained bit, all nine tensors
        XCTAssertNotEqual(corruptedAfter, cleanAfter)

        let seedArray = argMax(output.logits[0, boundary], axis: -1).asType(.int32)
        try withError { eval(seedArray) }
        let seed = Int(seedArray.item(Int32.self))
        let feature = output.normalizedHiddenStates[0..., boundary..<boundary + 1, 0...]
        _ = try proposals(assistant, clean, seed: seed, hidden: feature, depth: 1)
        _ = try proposals(assistant, corrupted, seed: seed, hidden: feature, depth: 1)
        let cleanRound = try XCTUnwrap(clean.round), corruptedRound = try XCTUnwrap(corrupted.round)
        let cleanDraftBytes = bytes(cleanRound.cache.innerState())
        let corruptedDraftBytes = bytes(corruptedRound.cache.innerState())
        var expectedDraftBytes = cleanDraftBytes
        expectedDraftBytes[0][changedByte] ^= 1
        XCTAssertEqual(corruptedDraftBytes, expectedDraftBytes)
        XCTAssertNotEqual(corruptedDraftBytes, cleanDraftBytes)
        // No assertion that argmax must change: exact state is the discriminator.
        XCTAssertEqual(bytes(checkpoint.evaluationTargets), original)
        assistant.discardRound(requestState: clean); assistant.discardRound(requestState: corrupted)
        try finish(assistant, clean); try finish(assistant, corrupted)
        assistant.releaseRequestState(clean); assistant.releaseRequestState(corrupted)
        assistant.releaseRequestState(donor)
    }

    func testDonorAdvanceDoesNotMutatePartialOrWrappedPublishedPrefix() throws {
        let (target, _, assistant) = try fixture()
        defer { withExtendedLifetime(target) {} }
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        for boundary in [4, 11] {
            let donor = try state(assistant, prompt: tokens)
            try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates,
                        range: 0..<boundary)
            let checkpoint = try capture(assistant, donor, at: boundary)
            let published = bytes(try XCTUnwrap(assistant.encodePrefixCheckpoint(checkpoint)))
            XCTAssertEqual(published.count, 9)
            try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates,
                        range: boundary..<boundary + 1)
            XCTAssertEqual(bytes(checkpoint.evaluationTargets), published)
            try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates,
                        range: boundary + 1..<tokens.count)
            XCTAssertEqual(donor.committedInputCount, tokens.count)
            XCTAssertEqual(checkpoint.targetInputCount, boundary)
            XCTAssertEqual(bytes(checkpoint.evaluationTargets), published)

            let restored = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint) as? MiMoV26MTPState)
            try assistant.configureRequestState(restored, maximumSequenceLength: 64)
            try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: tokens)
            try finish(assistant, restored)
            XCTAssertEqual(bytes(try capture(assistant, restored, at: boundary).evaluationTargets), published)
            XCTAssertEqual(restored.headInputCounts, (0..<3).map { boundary - $0 - 1 })
            XCTAssertEqual(restored.cache?.nextTokenPositions, [boundary, boundary, boundary])
            XCTAssertFalse(restored.hasRequiredRestoredSuffix)
            assistant.releaseRequestState(restored); assistant.releaseRequestState(donor)
        }
    }

    func testCancelledOffToSideAdoptionLeavesDonorAndCheckpointUnchanged() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let donor = try state(assistant, prompt: tokens)
        try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
        let checkpoint = try capture(assistant, donor)
        let originalCache = bytes(donor.cache!.innerState()), originalCheckpoint = bytes(checkpoint.evaluationTargets)
        let candidate = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint) as? MiMoV26MTPState)
        XCTAssertTrue(candidate.hasUnmeasuredResidency)
        XCTAssertFalse(candidate === donor)
        // Component-side cancellation/target-veto rollback: no candidate is
        // published, and it is disposed ONLY after its actual native completion.
        try finish(assistant, candidate)
        assistant.releaseRequestState(candidate)
        XCTAssertTrue(candidate.isReleased)
        XCTAssertEqual(candidate.materializedBytes, 0) // owner bookkeeping, not physical-free proof
        XCTAssertEqual(bytes(donor.cache!.innerState()), originalCache)
        XCTAssertEqual(bytes(checkpoint.evaluationTargets), originalCheckpoint)
        XCTAssertEqual(donor.observedCount, 11)
        let retry = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint) as? MiMoV26MTPState)
        try finish(assistant, retry)
        XCTAssertEqual(retry.headInputCounts, [10, 9, 8])
        assistant.releaseRequestState(retry); assistant.releaseRequestState(donor)
    }

    func testCodecTransportPreservesBF16NonfiniteAndSubnormalPayloadBits() throws {
        let (target, _, assistant) = try fixture()
        let tokens = prompt(), output = try target.forward(inputIDs: ids(tokens))
        let donor = try state(assistant, prompt: tokens)
        try observe(assistant, donor, tokens: tokens, hidden: output.normalizedHiddenStates, range: 0..<11)
        let checkpoint = try capture(assistant, donor)
        var arrays = checkpoint.evaluationTargets
        let patterns: [UInt16] = [0x0001, 0x8000, 0x7fc3, 0xff80, 0x3f80, 0x8001]
        // Codec transport test, not a claim these artificial values were
        // generated by the model. The real model cases above remain separate.
        for index in 0...6 {
            arrays[index] = MLXArray((0..<arrays[index].size).map { patterns[$0 % patterns.count] },
                                     arrays[index].shape).view(dtype: .bfloat16)
        }
        try withError { eval(arrays) }
        let expected = bytes(arrays)
        let imported = try XCTUnwrap(assistant.decodePrefixCheckpoint(
            tensors: arrays, prefixTokens: Array(tokens.prefix(11))))
        let restored = try XCTUnwrap(assistant.restorePrefixCheckpoint(imported) as? MiMoV26MTPState)
        try assistant.configureRequestState(restored, maximumSequenceLength: 64)
        try assistant.installPrefixCaptureContext(requestState: restored, promptTokens: tokens)
        try finish(assistant, restored)
        let recaptured = try capture(assistant, restored)
        XCTAssertEqual(bytes(recaptured.evaluationTargets), expected)
        assistant.releaseRequestState(restored); assistant.releaseRequestState(donor)
    }

    func testBoundedPlanPricesContextWitnessCopiesAndOverflowRefuses() throws {
        let config = try MiMoV26MTPChecks.config()
        let spec = try XCTUnwrap(MiMoV26MTPAssistant.boundedRequestAllocation(configuration: config,
            limits: .init(maximumPrefillTokens: 16, maximumDraftTokens: 3)))
        XCTAssertGreaterThanOrEqual(spec.hostBytes, (64 << 10) + 32 * config.maxPositionEmbeddings)
        XCTAssertTrue(spec.resident.contains(.init(logicalBytes: 4 * config.maxPositionEmbeddings, allocationCount: 3)))
        XCTAssertTrue(spec.working.contains(.init(logicalBytes: 4 * config.maxPositionEmbeddings, allocationCount: 4)))
        XCTAssertNil(MiMoV26MTPAssistant.boundedRequestAllocation(configuration: config,
            limits: .init(maximumPrefillTokens: Int.max, maximumDraftTokens: 3)))
        let (target, predictor) = try MiMoV26MTPChecks.fixture() // FP32 remains valid MTP, not this BF16 codec
        let floatAssistant = try MiMoV26MTPAssistant(target: target, predictor: predictor)
        XCTAssertNil(floatAssistant.prefixCheckpointTensorDescriptors(targetInputCount: 11))
        let request = try state(floatAssistant)
        XCTAssertThrowsError(try floatAssistant.installPrefixCaptureContext(requestState: request, promptTokens: prompt()))
        floatAssistant.releaseRequestState(request)
    }
}
