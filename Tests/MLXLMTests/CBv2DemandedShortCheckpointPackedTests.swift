import XCTest

@testable import MLXLMCommon

final class CBv2DemandedShortCheckpointPackedTests: XCTestCase {
    private func run(blockerOutputs: Int, minimum: Int? = 1_024) async throws -> (
        RecurrentGeometryObservations, CompleteCheckpointFixtureStore, Bool, [[Int]]
    ) {
        let store = CompleteCheckpointFixtureStore()
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1, modelLayerIndex: 1)
        ]
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 3, maxBatchedTokensPerStep: 2_048,
            prefillChunkSize: 512, soloPrefillStripeTokens: 4_096,
            maxConcurrentPartialPrefills: 1, maxWaiting: 4, enablePrefixCache: true)
        config.demandedShortCheckpointMinimumTokens = minimum
        let engine = EngineV2(
            model: PackableCompleteCheckpointFixtureModel(), layerKinds: kinds,
            backend: backend, cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2GreedySampler(), schedulerConfig: config,
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store)
        let observed = RecurrentGeometryObservations()
        let targetID = CBv2RequestID(7_009)
        let streams = try engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = 3
            engine.loopForTesting.recurrentGeometryObserverForTesting = {
                id, range, cap, packed, phase, outcome in
                guard id == targetID, phase == "record", range.upperBound <= 2_800 else { return }
                observed.append(range: range, cap: cap ?? -1, packed: packed, outcome: outcome)

            }
            return [
                try engine.submit(
                    CBv2Request(id: .init(7_007), promptTokens: [1], maxTokens: blockerOutputs)),
                try engine.submit(
                    CBv2Request(
                        id: .init(7_008), promptTokens: Array(repeating: 2, count: 2_049),
                        maxTokens: 1, cacheSalt: "neighbor", prefixCacheReceiptID: .init(7_108),
                        prefixCheckpointTargetTokens: 2_048)),
                try engine.submit(
                    CBv2Request(
                        id: targetID, promptTokens: Array(repeating: 3, count: 2_800),
                        maxTokens: 1, cacheSalt: "tenant", prefixCacheReceiptID: .init(7_109),
                        prefixCheckpointTargetTokens: 2_311)),
            ]
        }
        if minimum != nil {
            try await waitUntilSuspended(engine, at: 3)
            engine.loopForTesting.onEngineQueueSync {
                if blockerOutputs == 4 {
                    // Before the ordinary packed group exists, projection has
                    // no model/cache packing fact. It conservatively prices
                    // one extra future boundary; it never underprices work.
                    let projection = engine.loopForTesting.scheduler.firstTokenWorkProjection(
                        for: targetID,
                        inFlightAssignments: [(.init(7_007), 1), (.init(7_008), 512)])
                    guard case .bounded(let work, _) = projection else {
                        return XCTFail("pre-pack projection must remain bounded")
                    }
                    XCTAssertEqual(work.scheduledSteps, 5)
                }
                engine.loopForTesting.suspendStepExecutionAtCountForTesting = 4
            }
            try await waitUntilSuspended(engine, at: 4)
            engine.loopForTesting.onEngineQueueSync {
                // The first target range is still in flight: its permanent
                // geometry has not been recorded, but admission must already
                // know whether the launched graph was actually packed.
                XCTAssertNil(engine.loopForTesting.recurrentCheckpointGeometry[targetID])
                let record = engine.loopForTesting.scheduler.record(for: targetID)
                XCTAssertEqual(record?.shortCheckpointCaptureDisarmed, blockerOutputs == 4)
                var assignments: [(id: CBv2RequestID, numTokens: Int)] = [
                    (.init(7_008), 512), (targetID, 512),
                ]
                if blockerOutputs == 4 { assignments.insert((.init(7_007), 1), at: 0) }
                let projection = engine.loopForTesting.scheduler.firstTokenWorkProjection(
                    for: targetID, inFlightAssignments: assignments)
                guard case .bounded(let work, _) = projection else {
                    return XCTFail("actual in-flight work must project once")
                }
                XCTAssertEqual(work.prefillTokens, 3_313)
                XCTAssertEqual(work.scheduledSteps, blockerOutputs == 4 ? 3 : 4)
            }
        }
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        var generated: [[Int]] = []
        for stream in streams {
            let result = await cbv2SchedCollect(stream)
            XCTAssertEqual(result.finishReason, .length)
            generated.append(result.tokens)
        }
        let packed = engine.packedPrefillActivity().didExecute
        store.finishPublicationCallbacks(engine: engine)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(backend.bytesReserved, 0)
        await engine.shutdown()
        return (observed, store, packed, generated)
    }

    private func waitUntilSuspended(_ engine: EngineV2, at step: Int) async throws {
        for _ in 0 ..< 200 {
            if engine.loopForTesting.onEngineQueueSync({ engine.loopForTesting.stepCount >= step })
            {
                return
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTFail("engine did not reach the bounded in-flight fixture gate")
    }

    func testActualPackedDisarmKeepsLaterSoloScheduleAndProjectionUnsplit() async throws {
        let (observed, store, packed, generated) = try await run(blockerOutputs: 4)
        let records = observed.snapshot
        XCTAssertTrue(packed, "ordinary equal-size chunks must retain their packed execution")
        XCTAssertEqual(records.first?.range, 0 ..< 512)
        XCTAssertEqual(records.first?.packed, true)
        XCTAssertEqual(records.first?.outcome, "disarm")
        XCTAssertEqual(records.map(\.range), [0 ..< 512, 512 ..< 1_024, 1_024 ..< 2_800])
        XCTAssertEqual(records.last?.cap, 4_096)
        XCTAssertTrue(records.allSatisfy { $0.outcome == "disarm" })
        XCTAssertFalse(store.saved.contains { $0.manifest.cacheSalt == "tenant" })
        let (_, _, controlPacked, control) = try await run(blockerOutputs: 4, minimum: nil)
        XCTAssertTrue(controlPacked)
        XCTAssertEqual(generated, control, "unchanged native token IDs for every cohort row")
    }

    func testNewDemandedInteriorBoundaryDoesNotPackAndDisarmItsOwnCapture() async throws {
        let (observed, store, packed, generated) = try await run(blockerOutputs: 3)
        let records = observed.snapshot
        XCTAssertFalse(
            packed, "only the range shortened for a new demanded boundary leaves this group")
        XCTAssertTrue(records.allSatisfy { !$0.packed && $0.outcome != "disarm" })
        XCTAssertEqual(
            records.map(\.range), [0 ..< 512, 512 ..< 1_024, 1_024 ..< 2_304, 2_304 ..< 2_800])
        XCTAssertTrue(
            store.saved.contains {
                $0.manifest.cacheSalt == "neighbor" && $0.manifest.position == 2_048
            })
        XCTAssertTrue(
            store.saved.contains {
                $0.manifest.cacheSalt == "tenant" && $0.manifest.position == 2_304
            })
        let (_, _, _, control) = try await run(blockerOutputs: 3, minimum: nil)
        XCTAssertEqual(generated, control, "unchanged native token IDs for every cohort row")
    }
}
