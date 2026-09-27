import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Both the earlier window and its dependent full layer affect every later
/// logit. Borrowed layers read their owner's cache, never a synthetic own row.
private final class HistoricalAttentionModel: CBv2SteppableModel, CBv2HistoricalAttentionCheckpointProviding {
    let cbv2SupportsHistoricalAttentionCheckpoint = true

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let batch = tokens.dim(0), length = tokens.dim(1)
        var hidden = tokens.asType(.float32).reshaped([batch, 1, length, 1]) / Float(32)
        for cache in caches {
            let q = MLXArray.zeros([batch, 2, length, 64], dtype: .float32)
            let result: MLXArray
            if let owner = cache.kind.sharesKVWithLayer {
                result = cache.attendBorrowing(source: caches[owner], queries: q, scale: 0.125, sinks: nil)
            } else {
                let kv = broadcast(hidden, to: [batch, 1, length, 64])
                result = cache.updateAndAttend(queries: q, keys: kv, values: kv, scale: 0.125, sinks: nil)
            }
            hidden = mean(result, axes: [1, 3], keepDims: true)
        }
        let target = MLX.round(hidden.reshaped([batch, length, 1]) * Float(128)).asType(.int32) % 31
        return MLX.where(MLXArray(Int32(0) ..< Int32(32)) .== target, Float(10), Float(-10))
    }
}

final class HistoricalWindowCheckpointEngineTests: XCTestCase {
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }

    /// `stripe` mirrors production: a solo text request prefills in
    /// `2 * chunk` stripes (the pool ring is sized to that stripe, as
    /// `EngineV2Factory` sizes it to `max(prefillChunkSize, soloStripe)`).
    /// The capture stride is lowered to one chunk so the tiny model exercises
    /// the same rule production applies at 1,024 tokens.
    private func engine(_ store: CompleteCheckpointFixtureStore, stripe: Bool = false)
        throws -> (EngineV2, PagedKVBackend)
    {
        let kinds = [
            CBv2LayerKind(attention: .slidingWindow(17), headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .slidingWindow(17), sharesKVWithLayer: 0,
                          headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .full, sharesKVWithLayer: 1,
                          headDim: 64, kvHeads: 1, queryHeads: 2),
        ]
        let largest = stripe ? 2 * chunk : chunk
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(capacityBytes: 256 << 20,
            maxPrefillChunk: largest, segmentSizeBytes: 64 << 10, layerDTypes: Array(repeating: .float32, count: 4)))
        let engine = EngineV2(model: HistoricalAttentionModel(), layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: largest,
                prefillChunkSize: chunk, soloPrefillStripeTokens: stripe ? largest : nil,
                maxWaiting: 4, enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store)
        XCTAssertEqual(engine.completeCheckpointCodec?.backendLayout,
                       CBv2CompleteCheckpointManifest.historicalAttentionLayout)
        engine.loopForTesting.onEngineQueueSync {
            engine.completeCheckpointCapture?.historicalCheckpointStrideTokens = chunk
        }
        return (engine, backend)
    }

    func testRestartUsesHistoricalWindowAfterDonorHasWrappedSeveralTimes() async throws {
        let store = CompleteCheckpointFixtureStore(segmentBytes: 258)
        let (first, firstBackend) = try engine(store)
        let tokens = (0 ..< 3 * chunk + 1).map { ($0 * 7) % 29 }
        let request = CBv2Request(id: .init(1), promptTokens: tokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
        let cold = await cbv2SchedCollect(try first.submit(request))
        XCTAssertEqual(cold.finishReason, .length)
        // Every chunk end under K is retained and published deepest first.
        XCTAssertEqual(store.saved.map(\.manifest.position), [3 * chunk, 2 * chunk, chunk])
        XCTAssertEqual(store.saved.map(\.manifest.chunkSize), [chunk, chunk, chunk])
        XCTAssertEqual(first.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(firstBackend.bytesWired, 0)
        await first.shutdown()

        // Reopen only the EARLIER boundary, whose donor ring was overwritten
        // twice before publication. Reusing the terminal window cannot pass.
        let reopened = CompleteCheckpointFixtureStore(archives: store.saved.filter { $0.manifest.position == chunk })
        let (second, secondBackend) = try engine(reopened)
        let warmRequest = CBv2Request(id: .init(2), promptTokens: tokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(2002))
        XCTAssertTrue(try reopened.stage(engine: second, request: warmRequest))
        let warm = await cbv2SchedCollect(try second.submit(warmRequest))
        XCTAssertEqual(warm.tokens, cold.tokens)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, chunk)
        XCTAssertEqual(warm.usage?.prefixCacheReplayTokens, 0)
        XCTAssertEqual(second.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(secondBackend.bytesWired, 0)
        await second.shutdown()
    }

    /// Production geometry in miniature: a solo request prefills in
    /// `2 * chunk` stripes and its `5 * chunk + 1` prompt ends in a ragged
    /// `[4c, 5c + 1)` range. Every stripe end below the prompt end is a
    /// checkpoint, the manifests carry the stride as their alignment, and the
    /// deepest one restores exactly.
    func testStripeBoundariesCaptureAndRestoreExactly() async throws {
        let store = CompleteCheckpointFixtureStore(segmentBytes: 258)
        let (first, firstBackend) = try engine(store, stripe: true)
        let tokens = (0 ..< 5 * chunk + 1).map { ($0 * 5) % 29 }
        let request = CBv2Request(id: .init(1), promptTokens: tokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
        let cold = await cbv2SchedCollect(try first.submit(request))
        XCTAssertEqual(cold.finishReason, .length)
        // Interior boundaries land too: c inside [0, 2c), 3c inside [2c, 4c)
        // and 5c inside the ragged tail [4c, 5c + 1).
        XCTAssertEqual(store.saved.map(\.manifest.position), [5 * chunk, 4 * chunk, 3 * chunk, 2 * chunk, chunk])
        XCTAssertTrue(store.saved.allSatisfy { $0.manifest.position % $0.manifest.chunkSize == 0 })
        XCTAssertEqual(first.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(firstBackend.bytesWired, 0)
        await first.shutdown()

        let reopened = CompleteCheckpointFixtureStore(archives: store.saved.filter { $0.manifest.position == 4 * chunk })
        let (second, secondBackend) = try engine(reopened, stripe: true)
        let warmRequest = CBv2Request(id: .init(2), promptTokens: tokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(2002))
        XCTAssertTrue(try reopened.stage(engine: second, request: warmRequest))
        let warm = await cbv2SchedCollect(try second.submit(warmRequest))
        XCTAssertEqual(warm.tokens, cold.tokens)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, 4 * chunk)
        XCTAssertEqual(warm.usage?.prefixCacheReplayTokens, 0)
        XCTAssertEqual(second.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(secondBackend.bytesWired, 0)
        await second.shutdown()

        // Interior boundaries were copied out of the ring after their stripe
        // had already advanced the frontier: 3c inside [2c, 4c) and 5c inside
        // the ragged tail. Both restore exactly.
        for (id, position) in [(3, 3 * chunk), (5, 5 * chunk)] {
            let interior = CompleteCheckpointFixtureStore(archives: store.saved.filter { $0.manifest.position == position })
            let (third, thirdBackend) = try engine(interior, stripe: true)
            let interiorRequest = CBv2Request(id: .init(UInt64(id)), promptTokens: tokens, maxTokens: 4,
                cacheSalt: "tenant", prefixCacheReceiptID: .init(UInt64(1000 + id)))
            XCTAssertTrue(try interior.stage(engine: third, request: interiorRequest))
            let fromInterior = await cbv2SchedCollect(try third.submit(interiorRequest))
            XCTAssertEqual(fromInterior.tokens, cold.tokens, "restore at \(position)")
            XCTAssertEqual(fromInterior.usage?.prefixCachePrefillTokensSaved, position)
            XCTAssertEqual(third.admissionForTesting.bytesReserved, 0)
            XCTAssertEqual(thirdBackend.bytesWired, 0)
            await third.shutdown()
        }
    }

    /// The prod failure in miniature: the donor's first range is a solo
    /// `2c` stripe; a second request arrives while that step is launching, so
    /// every later range is a plain `c` chunk. The uniform-cap rule disarmed
    /// at the cap change and kept only `2c`; the historical rule keeps
    /// capturing, and the deepest boundary restores exactly.
    func testCapChangeMidPromptKeepsCapturing() async throws {
        let store = CompleteCheckpointFixtureStore(segmentBytes: 258)
        let (engine, backend) = try engine(store, stripe: true)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let firstCapture = DispatchSemaphore(value: 0)
        firstCapture.signal()
        engine.loopForTesting.onEngineQueueSync {
            engine.completeCheckpointCapture?.makeHistoricalWindow = { row, position, admission in
                if firstCapture.wait(timeout: .now()) == .success {
                    entered.signal()
                    _ = release.wait(timeout: .now() + 10)
                }
                return try CBv2HistoricalWindow(row: row, position: position, admission: admission)
            }
        }
        let tokens = (0 ..< 5 * chunk + 1).map { ($0 * 5) % 29 }
        let request = CBv2Request(id: .init(1), promptTokens: tokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
        let stream = try engine.submit(request)
        let collected = Task { await cbv2SchedCollect(stream) }
        let blocked = await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                continuation.resume(returning: entered.wait(timeout: .now() + 5) == .success)
            }
        }
        XCTAssertTrue(blocked)
        // The first step [0, 2c) is launching under the solo stripe (the
        // capture seam runs before the launch is counted). Company queued now
        // is visible to every later plan.
        XCTAssertLessThanOrEqual(engine.stepCount, 1)
        let companyStream = try engine.submit(.init(id: .init(2), promptTokens: [3, 1, 4], maxTokens: 2))
        let company = Task { await cbv2SchedCollect(companyStream) }
        release.signal()
        let donor = await collected.value
        let companyResult = await company.value
        XCTAssertEqual(donor.finishReason, .length)
        XCTAssertEqual(companyResult.tokens.count, 2)
        let positions = store.saved.map(\.manifest.position)
        XCTAssertEqual(positions, positions.sorted(by: >), "published deepest first")
        XCTAssertTrue(positions.contains(2 * chunk), "the solo stripe's end: \(positions)")
        XCTAssertTrue(positions.contains(3 * chunk), "a plain-chunk end after the cap change: \(positions)")
        XCTAssertTrue(positions.allSatisfy { $0 % chunk == 0 && $0 < tokens.count })
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(backend.bytesWired, 0)
        await engine.shutdown()

        let deepest = try XCTUnwrap(positions.max())
        let reopened = CompleteCheckpointFixtureStore(archives: store.saved.filter { $0.manifest.position == deepest })
        let (second, secondBackend) = try self.engine(reopened, stripe: true)
        let warmRequest = CBv2Request(id: .init(3), promptTokens: tokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(3003))
        XCTAssertTrue(try reopened.stage(engine: second, request: warmRequest))
        let warm = await cbv2SchedCollect(try second.submit(warmRequest))
        XCTAssertEqual(warm.tokens, donor.tokens)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, deepest)
        XCTAssertEqual(second.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(secondBackend.bytesWired, 0)
        await second.shutdown()
    }

    /// A historical adopter is not held to the donor's chunk geometry: it
    /// resumes under ordinary scheduling (here the solo stripe) and donates
    /// its own deeper boundaries from the restored rows.
    func testHistoricalAdopterResumesWithOrdinaryChunkingAndDonatesDeeper() async throws {
        let store = CompleteCheckpointFixtureStore(segmentBytes: 258)
        let (first, _) = try engine(store, stripe: true)
        let donorTokens = (0 ..< 3 * chunk + 1).map { ($0 * 3) % 29 }
        let donor = await cbv2SchedCollect(try first.submit(.init(id: .init(1), promptTokens: donorTokens,
            maxTokens: 2, cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))))
        XCTAssertEqual(donor.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [3 * chunk, 2 * chunk, chunk])
        await first.shutdown()

        // The long prompt shares the donor's first 3c + 1 tokens exactly.
        let longTokens = donorTokens + (0 ..< 4 * chunk).map { ($0 * 13 + 1) % 29 }
        let (coldEngine, _) = try engine(CompleteCheckpointFixtureStore(), stripe: true)
        let cold = await cbv2SchedCollect(try coldEngine.submit(.init(id: .init(3), promptTokens: longTokens,
            maxTokens: 4, cacheSalt: "tenant", prefixCacheReceiptID: .init(3003))))
        await coldEngine.shutdown()

        let reopened = CompleteCheckpointFixtureStore(archives: store.saved.filter { $0.manifest.position == 2 * chunk })
        let (second, secondBackend) = try engine(reopened, stripe: true)
        let warmRequest = CBv2Request(id: .init(2), promptTokens: longTokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(2002))
        XCTAssertTrue(try reopened.stage(engine: second, request: warmRequest))
        let warm = await cbv2SchedCollect(try second.submit(warmRequest))
        XCTAssertEqual(warm.tokens, cold.tokens)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, 2 * chunk)
        XCTAssertEqual(warm.usage?.prefixCacheReplayTokens, 0)
        // Cold ran four 2c stripes over [0, 7c + 1) then three decodes (7
        // steps); the adopter resumes at 2c in three stripes (6). Under the
        // donor's forced stride it would take five c chunks plus the tail:
        // 9 steps, MORE than cold rather than fewer.
        XCTAssertLessThan(second.stepCount, coldEngine.stepCount,
                          "historical adoption must resume on the solo stripe, not the donor's chunk size")
        XCTAssertEqual(second.stepCount, coldEngine.stepCount - 1)
        // Resumed at 2c, the adopter's own stripes [2c, 4c), [4c, 6c) and the
        // tail [6c, 7c + 1) donate every boundary they cover. (The reopened
        // store's first archive is the donor's 2c it was seeded with.)
        XCTAssertEqual(Array(reopened.saved.map(\.manifest.position).dropFirst()),
                       [7 * chunk, 6 * chunk, 5 * chunk, 4 * chunk, 3 * chunk])
        XCTAssertEqual(second.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(secondBackend.bytesWired, 0)
        await second.shutdown()
    }

    func testExpiredStageClosesBeforeSubmissionAndFallsBackToCold() async throws {
        let store = CompleteCheckpointFixtureStore()
        let (first, _) = try engine(store)
        let tokens = (0 ..< 2 * chunk + 1).map { ($0 * 11) % 29 }
        let request = CBv2Request(id: .init(1), promptTokens: tokens, maxTokens: 2,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(3001))
        let cold = await cbv2SchedCollect(try first.submit(request))
        await first.shutdown()
        let reopened = CompleteCheckpointFixtureStore(archives: store.saved)
        let (second, backend) = try engine(reopened)
        XCTAssertTrue(try reopened.stage(engine: second, request: request))
        let stage = reopened.takeStaged(requestID: .init(3001), tokens: tokens, cacheSalt: "tenant",
                                       maximumSequenceLength: tokens.count + 2)
        XCTAssertNotNil(stage)
        stage?.close()
        let result = await cbv2SchedCollect(try second.submit(request))
        XCTAssertEqual(result.tokens, cold.tokens)
        XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved, 0)
        // The closed stage still owns its small manifest until this handle is
        // dropped; it never owns a page or native destination after expiry.
        XCTAssertEqual(backend.bytesWired, 0)
        await second.shutdown()
    }

    func testRawNativeCopyErrorTerminatesCohortWithoutSampleOrDonation() async throws {
        for duringConstruction in [true, false] {
            let store = CompleteCheckpointFixtureStore()
            let (engine, backend) = try engine(store)
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.makeHistoricalWindow = { row, position, admission in
                    if duringConstruction {
                        return try CBv2HistoricalWindow(row: row, position: position, admission: admission,
                            afterConstruction: { _ in throw MLXError.caught("injected native construction failure") })
                    }
                    return try CBv2HistoricalWindow(row: row, position: position, admission: admission,
                        evaluate: { array in
                            try withError { eval(array) }
                            throw MLXError.caught("injected native copy failure")
                        })
                }
            }
            let result = await cbv2SchedCollect(try engine.submit(.init(id: .init(1),
                promptTokens: Array(repeating: 7, count: 2 * chunk + 1), maxTokens: 3,
                prefixCacheReceiptID: .init(4001))))
            if case .error? = result.finishReason {} else { XCTFail("native error was swallowed as an optional cache miss") }
            XCTAssertTrue(result.tokens.isEmpty)
            XCTAssertTrue(store.saved.isEmpty)
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
            XCTAssertEqual(backend.bytesWired, 0)
            await engine.shutdown()
        }
    }

    func testCancelAndShutdownCannotLaunchSuccessorBeforePrivateCopyDrains() async throws {
        for shuttingDown in [false, true] {
            let store = CompleteCheckpointFixtureStore()
            let (engine, backend) = try engine(store)
            let entered = DispatchSemaphore(value: 0)
            let release = DispatchSemaphore(value: 0)
            let firstCopy = DispatchSemaphore(value: 0)
            firstCopy.signal()
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.makeHistoricalWindow = { row, position, admission in
                    try CBv2HistoricalWindow(row: row, position: position, admission: admission,
                        evaluate: { array in
                            if firstCopy.wait(timeout: .now()) == .success {
                                entered.signal()
                                _ = release.wait(timeout: .now() + 10)
                            }
                            try withError { eval(array) }
                        })
                }
            }
            let stream = try engine.submit(.init(id: .init(1),
                promptTokens: Array(repeating: 7, count: 2 * chunk + 1), maxTokens: 2,
                prefixCacheReceiptID: .init(5001)))
            let collected = Task { await cbv2SchedCollect(stream) }
            let blocked = await withCheckedContinuation { continuation in
                DispatchQueue.global().async {
                    continuation.resume(returning: entered.wait(timeout: .now() + 5) == .success)
                }
            }
            XCTAssertTrue(blocked)
            // The queue is inside first-step private readback. No later graph
            // can have run or advanced the ring while this owner is pending.
            XCTAssertEqual(engine.stepCount, 1)
            XCTAssertGreaterThan(engine.admissionForTesting.bytesReserved, 0)
            let shutdown = shuttingDown ? Task { await engine.shutdown() } : nil
            if !shuttingDown { engine.cancel(.init(1)) }
            release.signal()
            let result = await collected.value
            if shuttingDown {
                await shutdown?.value
            } else {
                XCTAssertEqual(result.finishReason, .cancelled)
                await engine.shutdown()
            }
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
            XCTAssertEqual(backend.bytesWired, 0)
        }
    }

}
