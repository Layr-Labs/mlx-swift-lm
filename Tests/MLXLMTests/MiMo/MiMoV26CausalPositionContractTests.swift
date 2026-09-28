import Foundation
import MLX
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Direct native numerical/submit regressions only. These deliberately use the
/// existing untracked tiny media fixture, not a fabricated managed contract.
/// Real managed media issuance/preparation/retirement remains a separate gate.
final class MiMoV26CausalPositionContractTests: XCTestCase {
    private enum Failure: Error { case expectedPositionRefusal }

    private func prepared(_ fixture: MiMoMediaFixture.Models) throws -> MiMoV26PreparedMultimodal {
        let input = MiMoV26MultimodalInput(messages: [
            .init(role: .user, content: [.text("before"), .image(MiMoMediaFixture.image()),
                                        .text("01234567890123456789")])
        ], maximumOutputTokens: 18)
        return try fixture.processor.prepare(fixture.processor.plan(input), authorize: {
            MiMoMediaFixture.Reservation($0)
        })
    }

    /// Independent ordinary-cache reference: assemble exact native feature
    /// spans on the host and run native target prefill/decode. Does not invoke
    /// the engine's splice, position planner or changed submit validator.
    private func reference(_ fixture: MiMoMediaFixture.Models,
                           prepared: MiMoV26PreparedMultimodal) throws -> [Int] {
        let request = try prepared.makeRequest(binding: fixture.binding, id: .init(6000),
                                               sampling: .init(temperature: 0))
        let features = try XCTUnwrap(request.multimodal).embeddings()
        let text = fixture.target.model.embedTokens(
            MLXArray(request.promptTokens.map(Int32.init), [1, request.promptTokens.count]))
        var pieces: [MLXArray] = [], cursor = 0
        XCTAssertEqual(features.count, prepared.plan.spans.count)
        for (span, feature) in zip(prepared.plan.spans, features) {
            if cursor < span.tokenOffset { pieces.append(text[0..., cursor..<span.tokenOffset, 0...]) }
            pieces.append(feature.expandedDimensions(axis: 0))
            cursor = span.tokenOffset + span.length
        }
        if cursor < request.promptTokens.count { pieces.append(text[0..., cursor..<request.promptTokens.count, 0...]) }
        let cache = fixture.target.newCache()
        var logits = try fixture.target.forward(embeddings: concatenated(pieces, axis: 1), cache: cache).logits
        var tokens: [Int] = []
        for step in 0..<request.maxTokens {
            let next = argMax(logits[0..., -1, 0...], axis: -1)
            eval(next)
            let token = Int(next.item(Int32.self)); tokens.append(token)
            if step + 1 < request.maxTokens {
                logits = try fixture.target.forward(inputIDs: MLXArray([Int32(token)], [1, 1]), cache: cache).logits
            }
        }
        return tokens
    }

    func testRealProducedImageRequestUsesScalarPositionsAcrossChunksAndSWAWrap() async throws {
        for dtype in ["float32", "bfloat16"] {
            let fixture = try MiMoMediaFixture.models(dtype)
            let model: any CBv2MultimodalSteppableModel = fixture.adapter
            // Existential dispatch must use the MiMo extension's witness, not
            // accidentally inherit the protocol's required-position default.
            XCTAssertEqual(model.causalPositionRequirement, .scalarCacheOffset)
            let expectedPrepared = try prepared(fixture)
            let expected = try reference(fixture, prepared: expectedPrepared)
            let windows = fixture.adapter.layerKinds.compactMap { kind -> Int? in
                if case .slidingWindow(let value) = kind.attention { return value }; return nil
            }
            XCTAssertGreaterThan(expectedPrepared.plan.promptTokens.count, try XCTUnwrap(windows.max()))
            XCTAssertTrue(fixture.adapter.layerKinds.contains { if case .full = $0.attention { return true }; return false })
            for chunk in [1, 3] {
                let value = try prepared(fixture)
                XCTAssertEqual(value.plan.promptTokens, expectedPrepared.plan.promptTokens)
                let request = try value.makeRequest(binding: fixture.binding, id: .init(UInt64(6100 + chunk)),
                                                     sampling: .init(temperature: 0))
                XCTAssertEqual(request.multimodal?.attention, .causal)
                XCTAssertNil(request.positionState); XCTAssertNil(request.multimodal?.positionState)
                XCTAssertFalse(request.prefixCacheEnabled)
                let (actual, backend) = try MiMoMediaMTPFixture.engine(fixture.adapter, chunk: chunk)
                do {
                    XCTAssertNil(actual.nativeShutdownExecutionContractID, "direct fixture is not managed-media proof")
                    let result = await cbv2SchedCollect(try actual.submit(request))
                    XCTAssertEqual(result.finishReason, .length)
                    XCTAssertEqual(result.tokens, expected, "native scalar positions must preserve exact greedy tokens")
                    let usage = try XCTUnwrap(result.usage)
                    XCTAssertEqual(usage.promptTokens, request.promptTokens.count)
                    XCTAssertEqual(usage.completionTokens, request.maxTokens)
                    XCTAssertGreaterThan(usage.timing.prefillChunks, 1)
                    await actual.shutdown()
                    XCTAssertEqual(backend.bytesReserved, 0)
                } catch { await actual.shutdown(); throw error }
            }
        }
    }

    func testSuppliedPositionsRemainRejectedWithoutConsumingTheActualMiMoProducer() async throws {
        let fixture = try MiMoMediaFixture.models()
        for (axisCount, badLength, onMedia) in [(3, true, false), (3, false, true), (1, false, false)] {
            let value = try prepared(fixture)
            var request = try value.makeRequest(binding: fixture.binding, id: .init(6200),
                                                sampling: .init(temperature: 0))
            let positions = CBv2PositionState(promptPositionIds: MLXArray.zeros(
                [axisCount, 1, request.promptTokens.count - (badLength ? 1 : 0)], dtype: .int32), decodeDeltas: [0])
            if onMedia { request.multimodal?.positionState = positions }
            else { request.positionState = positions }
            let (actual, backend) = try MiMoMediaMTPFixture.engine(fixture.adapter, chunk: 3)
            do {
                do {
                    _ = try actual.submit(request)
                    XCTFail("MiMo must not silently ignore supplied positions")
                    throw Failure.expectedPositionRefusal
                } catch let error as CBv2MultimodalError {
                    if badLength {
                        guard case .invalidSpans(let detail) = error else { throw error }
                        XCTAssertTrue(detail.contains("position length"))
                    } else {
                        guard case .unsupportedModel(let detail) = error else { throw error }
                        XCTAssertTrue(detail.contains("positioned model forwarding"))
                    }
                }
                XCTAssertEqual(backend.bytesReserved, 0)
                XCTAssertEqual(actual.loopForTesting.onEngineQueueSync { actual.loopForTesting.stepCount }, 0)
                request.positionState = nil; request.multimodal?.positionState = nil
                // Reuse the SAME one-shot produced closure and numeric ID.
                // Successful actual submit proves the negative path neither
                // consumed the features nor registered a stale generation.
                let result = await cbv2SchedCollect(try actual.submit(request))
                XCTAssertEqual(result.finishReason, .length)
                XCTAssertEqual(result.tokens.count, request.maxTokens)
                await actual.shutdown()
                XCTAssertEqual(backend.bytesReserved, 0)
            } catch { await actual.shutdown(); throw error }
        }
    }
}
