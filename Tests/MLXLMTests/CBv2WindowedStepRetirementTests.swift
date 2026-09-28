import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

// Source-only prepared native tests. Run serially in the exclusive synthetic
// lane: the active-memory delta below is an isolated retirement witness, never
// an allocation identity, process-reservation credit or a model qualification.
// Real helper dependencies: CBv2CompleteCheckpointEngineTests.swift supplies
// CompleteCheckpointFixtureStore; CBv2SchedulerTestSupport.swift supplies the
// actual stream collector/wait helpers. Neither helper file is modified.

private final class CBv2WindowRetirementWeakPair {
    weak var keys: MLXArray?
    weak var values: MLXArray?
    let positions: Int
    let logicalBytes: Int
    init(_ keys: MLXArray, _ values: MLXArray) {
        self.keys = keys; self.values = values
        positions = keys.dim(2); logicalBytes = keys.nbytes + values.nbytes
    }
}

/// Only weak native/row references. The observation cannot keep the buffers
/// alive and accidentally manufacture the retention the test is checking.
private final class CBv2WindowRetirementProbe: @unchecked Sendable {
    private let lock = NSLock()
    private weak var row: CBv2WindowedSequenceKV?
    private var pair: CBv2WindowRetirementWeakPair?
    private var count = 0
    func record(_ row: CBv2WindowedSequenceKV) {
        let views = row.borrowableViews()
        lock.withLock {
            self.row = row
            pair = .init(views.keys, views.values)
            count += 1
        }
    }
    func inspect(_ body: (CBv2WindowedSequenceKV?, CBv2WindowRetirementWeakPair?, Int) -> Void) {
        lock.withLock { body(row, pair, count) }
    }
}

/// Real BF16 asymmetric attention and shared-KV borrowing. Only the checkpoint
/// store/inputs are synthetic; this never substitutes a mock cache or backend.
private final class CBv2WindowRetirementModel: CBv2SteppableModel,
    CBv2HistoricalAttentionCheckpointProviding, CBv2CompleteCheckpointKVTypeProviding {
    let cbv2SupportsHistoricalAttentionCheckpoint = true
    let cbv2CompleteCheckpointKVDTypes: [DType]? = Array(repeating: .bfloat16, count: 4)
    let probe: CBv2WindowRetirementProbe?
    var pauseImportedChunk: (() -> Void)?
    init(probe: CBv2WindowRetirementProbe? = nil) { self.probe = probe }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let batch = tokens.dim(0), length = tokens.dim(1)
        var hidden = (tokens.asType(.float32).reshaped([batch, 1, length, 1]) / Float(32)).asType(.bfloat16)
        for cache in caches {
            let queries = MLXArray.zeros([batch, 2, length, 128], dtype: .bfloat16)
            let output: MLXArray
            if let owner = cache.kind.sharesKVWithLayer {
                output = cache.attendBorrowing(source: caches[owner], queries: queries, scale: 0.125, sinks: nil)
            } else {
                let keys = broadcast(hidden, to: [batch, 1, length, 128])
                let values = broadcast(hidden, to: [batch, 1, length, 192])
                output = cache.updateAndAttend(queries: queries, keys: keys, values: values, scale: 0.125, sinks: nil)
                if cache.layerIndex == 0, length == 256,
                    let row = (cache as? CBv2LayerCache)?.rows.first as? CBv2WindowedSequenceKV,
                    row.absoluteOffset == 512, row.checkpointBacking != nil {
                    probe?.record(row)
                    // Real backpressure scheduling API; enqueued on the engine
                    // queue before the next step can process the final suffix.
                    pauseImportedChunk?()
                }
            }
            hidden = mean(output.asType(.float32), axes: [1, 3], keepDims: true).asType(.bfloat16)
        }
        let target = MLX.round(hidden.asType(.float32).reshaped([batch, length, 1]) * Float(128)).asType(.int32) % 31
        return MLX.where(MLXArray(Int32(0)..<Int32(32)) .== target, Float(10), Float(-10))
    }
}

final class CBv2WindowedStepRetirementTests: XCTestCase {
    private func tensor(start: Int, count: Int, width: Int) -> MLXArray {
        let values = (0..<(count * width)).map { index in
            Float((start + index / width) % 31 + index % 7) / Float(16)
        }
        return MLXArray(values).reshaped([1, 1, count, width]).asType(.bfloat16)
    }

    private func fence(_ arrays: [MLXArray]) throws {
        try withError { fault in
            eval(arrays)
            StreamOrDevice.default.stream.synchronize()
            try fault.check()
        }
    }

    private func restoredRow() throws -> CBv2WindowedSequenceKV {
        let keys = tensor(start: 0, count: 257, width: 128)
        let values = tensor(start: 0, count: 257, width: 192)
        try fence([keys, values])
        return try .init(restoredKeys: keys, restoredValues: values, offset: 256,
                         window: 257, kvHeads: 1, headDim: 128, valueHeadDim: 192)
    }

    /// All strong references to the returned pair are scoped inside this
    /// helper. Only the row itself may retain them when the caller observes it.
    private func updateAndFence(_ row: CBv2WindowedSequenceKV, count: Int) throws -> CBv2WindowRetirementWeakPair {
        let start = row.absoluteOffset
        let pair = row.update(keys: tensor(start: start, count: count, width: 128),
                              values: tensor(start: start, count: count, width: 192))
        let witness = CBv2WindowRetirementWeakPair(pair.0, pair.1)
        try fence([pair.0, pair.1] + row.cbv2InnerState())
        XCTAssertNotNil(try pair.0.evaluatedBufferInfo())
        XCTAssertNotNil(try pair.1.evaluatedBufferInfo())
        return witness
    }

    private func ringValues(_ row: CBv2WindowedSequenceKV) throws -> [[Float]] {
        try autoreleasepool {
            let snapshot = row.snapshot()
            let keys = snapshot.keys.asType(.float32), values = snapshot.values.asType(.float32)
            try fence([keys, values])
            return [keys.asArray(Float.self), values.asArray(Float.self)]
        }
    }

    private func borrowedValues(_ row: CBv2WindowedSequenceKV) throws -> [[Float]] {
        try autoreleasepool {
            let views = row.borrowableViews()
            let keys = views.keys.asType(.float32), values = views.values.asType(.float32)
            try fence([keys, values])
            return [keys.asArray(Float.self), values.asArray(Float.self)]
        }
    }

    func testRestoredBF16ChunkViewsRetireAfterFenceWithoutRetiringRing() throws {
        let row = try restoredRow()
        let witness = try updateAndFence(row, count: 256)
        XCTAssertEqual(row.absoluteOffset, 512)
        XCTAssertEqual(witness.positions, 512)
        XCTAssertEqual(witness.logicalBytes, 512 * (128 + 192) * 2)
        XCTAssertNotNil(witness.keys); XCTAssertNotNil(witness.values)
        let expected = try ringValues(row)
        let ringBytes = row.byteCount
        XCTAssertEqual(ringBytes, 257 * (128 + 192) * 2)
        try fence(row.cbv2InnerState())
        let before = Memory.snapshot().activeMemory

        row.retireBorrowableChunkViews(afterFencedPosition: 512)
        try withError { StreamOrDevice.default.stream.synchronize() }
        let after = Memory.snapshot().activeMemory
        XCTAssertNil(witness.keys); XCTAssertNil(witness.values)
        XCTAssertGreaterThanOrEqual(before - after, witness.logicalBytes,
            "completed isolated chunk backing must leave native active accounting, not merely its Swift wrapper")
        XCTAssertEqual(row.absoluteOffset, 512)
        XCTAssertEqual(row.retainedCount, 257)
        XCTAssertEqual(row.byteCount, ringBytes, "temporary retirement must not refund/drop the ring")
        XCTAssertEqual(try ringValues(row), expected)
        XCTAssertEqual(row.borrowableViews().keys.dim(2), 257)
        row.retireBorrowableChunkViews(afterFencedPosition: 512) // idempotent
        XCTAssertEqual(try ringValues(row), expected)
    }

    func testStaleAndSpeculativeFencesPreserveCurrentBorrowedMath() throws {
        let row = try restoredRow(), reference = try restoredRow()
        let old = try updateAndFence(row, count: 256)
        _ = try updateAndFence(reference, count: 256)
        let expected = try borrowedValues(row)

        row.retireBorrowableChunkViews(afterFencedPosition: 256) // prior step generation
        XCTAssertNotNil(old.keys); XCTAssertNotNil(old.values)
        XCTAssertEqual(try borrowedValues(row), expected)
        row.beginSpeculativeWrite() // armed, but no staged write yet
        row.retireBorrowableChunkViews(afterFencedPosition: 512)
        XCTAssertNotNil(old.keys); XCTAssertNotNil(old.values)
        XCTAssertEqual(try borrowedValues(row), expected)
        row.commitSpeculativeWrite()
        row.retireBorrowableChunkViews(afterFencedPosition: 512)
        XCTAssertNil(old.keys); XCTAssertNil(old.values)

        row.beginSpeculativeWrite()
        let speculative = try updateAndFence(row, count: 3)
        let stagedMath = try borrowedValues(row)
        XCTAssertEqual(row.absoluteOffset, 515)
        row.retireBorrowableChunkViews(afterFencedPosition: 515) // exact offset but armed/staged
        XCTAssertNotNil(speculative.keys); XCTAssertNotNil(speculative.values)
        XCTAssertEqual(try borrowedValues(row), stagedMath)
        row.rollback(1)
        row.commitSpeculativeWrite()
        try fence(row.cbv2InnerState())
        _ = try updateAndFence(reference, count: 2)
        XCTAssertEqual(row.absoluteOffset, 514)
        XCTAssertEqual(try ringValues(row), try ringValues(reference),
                       "refused cleanup must not change accepted-prefix rollback/commit math")
        row.retireBorrowableChunkViews(afterFencedPosition: 514)
        XCTAssertEqual(try ringValues(row), try ringValues(reference))
    }

    private func engine(_ store: CompleteCheckpointFixtureStore,
                        model: CBv2WindowRetirementModel) -> (EngineV2, CBv2ContiguousKVBackend) {
        let kinds: [CBv2LayerKind] = [
            .init(attention: .slidingWindow(257), headDim: 128, valueHeadDim: 192, kvHeads: 1, queryHeads: 2),
            .init(attention: .full, headDim: 128, valueHeadDim: 192, kvHeads: 1, queryHeads: 2),
            .init(attention: .slidingWindow(257), sharesKVWithLayer: 0, headDim: 128, valueHeadDim: 192, kvHeads: 1, queryHeads: 2),
            .init(attention: .full, sharesKVWithLayer: 1, headDim: 128, valueHeadDim: 192, kvHeads: 1, queryHeads: 2),
        ]
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 128 << 20, kvDType: .bfloat16))
        let engine = EngineV2(model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds), sampler: CBv2DefaultSampler(fallbackSeed: 0),
            schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: 256,
                                   prefillChunkSize: 256, maxWaiting: 2, enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store)
        return (engine, backend)
    }

    func testRealImportedEnginePauseKeepsRowButRetiresCompletedChunkViews() async throws {
        // A 512-token request cannot donate a boundary at512: the native cache
        // reserves a final uncached token. Use the supported pause at512 before
        // the last1-token suffix of a513-token/maxTokens1 request instead. Do
        // not change prompt/export rules merely to force an I/O publication.
        XCTAssertEqual(256 % CBv2AttentionV1.queryBlockSize, 0)
        let prompt = (0..<513).map { ($0 * 7) % 29 }
        let seed = CompleteCheckpointFixtureStore(segmentBytes: 258)
        let (cold, _) = engine(seed, model: .init())
        let coldRequest = CBv2Request(id: .init(810), promptTokens: prompt,
            sampling: .init(temperature: 0, seed: 0), maxTokens: 1,
            cacheSalt: "window-retirement", prefixCacheReceiptID: .init(1810))
        let expected = await cbv2SchedCollect(try cold.submit(coldRequest))
        XCTAssertEqual(expected.finishReason, .length)
        await cold.shutdown()
        let earlier = seed.saved.filter { $0.manifest.position == 256 }
        XCTAssertEqual(earlier.count, 1)

        let store = CompleteCheckpointFixtureStore(archives: earlier, segmentBytes: 258)
        let probe = CBv2WindowRetirementProbe(), model = CBv2WindowRetirementModel(probe: probe)
        let (warm, backend) = engine(store, model: model)
        let request = CBv2Request(id: .init(811), promptTokens: prompt,
            sampling: .init(temperature: 0, seed: 0), maxTokens: 1,
            cacheSalt: "window-retirement", prefixCacheReceiptID: .init(1811))
        let loop = warm.loopForTesting
        model.pauseImportedChunk = { [weak loop] in loop?.setPaused(request.id, true) }
        XCTAssertTrue(try store.stage(engine: warm, request: request))
        // Counting is opt-in test instrumentation; it does not enable the
        // production fence. Keep the exact fence/retirement assertions below.
        let wasCounting = CBv2CoreInstrumentation.countingEnabled
        CBv2CoreInstrumentation.countingEnabled = true
        defer { CBv2CoreInstrumentation.countingEnabled = wasCounting }
        let hostSyncsBefore = CBv2CoreInstrumentation.hostSyncs
        let stream = try warm.submit(request)
        let collected = Task { await cbv2SchedCollect(stream) }
        let paused = await cbv2SchedWait { loop.pausedIDsSnapshot().contains(request.id) }
        XCTAssertTrue(paused)
        let retiredAtFence = await cbv2SchedWait(timeoutSeconds: 5) {
            loop.onEngineQueueSync {
                var ready = false
                probe.inspect { row, pair, count in
                    ready = CBv2CoreInstrumentation.hostSyncs > hostSyncsBefore
                        && count == 1 && row?.absoluteOffset == 512
                        && pair != nil && pair?.keys == nil && pair?.values == nil
                }
                return ready
            }
        }
        XCTAssertTrue(retiredAtFence, "the real finalize hook, not a later n1 update, must clear the weak views")
        loop.onEngineQueueSync {
            probe.inspect { row, pair, count in
                XCTAssertEqual(count, 1)
                XCTAssertNotNil(row, "actual imported/paused row must remain owned")
                XCTAssertNotNil(row?.checkpointBacking)
                XCTAssertEqual(row?.absoluteOffset, 512)
                XCTAssertEqual(row?.byteCount, 257 * (128 + 192) * 2)
                XCTAssertEqual(pair?.positions, 512)
                XCTAssertEqual(pair?.logicalBytes, 512 * (128 + 192) * 2)
                XCTAssertNil(pair?.keys); XCTAssertNil(pair?.values)
                XCTAssertEqual(row?.borrowableViews().keys.dim(2), 257)
                for array in row?.cbv2InnerState() ?? [] {
                    XCTAssertNotNil(try? array.evaluatedBufferInfo(), "ring is already available; observation must not eval")
                }
            }
        }
        XCTAssertEqual(warm.stepCount, 1, "no final1-token suffix or decode may mask the lifecycle defect")
        XCTAssertEqual(CBv2CoreInstrumentation.hostSyncs - hostSyncsBefore, 1,
                       "the exact imported256-token step completed its real host fence")
        XCTAssertEqual(store.releaseCount, 1)
        XCTAssertGreaterThan(backend.bytesReserved, 0)
        XCTAssertGreaterThan(warm.admissionForTesting.bytesReserved, 0)

        loop.onEngineQueueSync { model.pauseImportedChunk = nil }
        loop.setPaused(request.id, false)
        let result = await collected.value
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.tokens, expected.tokens)
        XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved, 256)
        await warm.shutdown()
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(warm.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(store.releaseCount, 1)
    }
}
