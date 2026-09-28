import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

/// One-shot state is not a semaphore permit: consuming a value-1 semaphore
/// forever violates libdispatch's destruction invariant.
private final class HistoricalCaptureOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var claimed = false
    func claim() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !claimed else { return false }
        claimed = true
        return true
    }
}

/// Synthetic inputs, real asymmetric attention/cache/EngineV2 paths. No model
/// payload or MiMo capability flag is involved in this component qualification.
private final class ContiguousHistoricalAttentionModel: CBv2SteppableModel,
    CBv2HistoricalAttentionCheckpointProviding, CBv2CompleteCheckpointKVTypeProviding {
    let cbv2SupportsHistoricalAttentionCheckpoint = true
    let cbv2CompleteCheckpointKVDTypes: [DType]? = Array(repeating: .float32, count: 4)
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let b=tokens.dim(0), n=tokens.dim(1)
        var hidden=tokens.asType(.float32).reshaped([b,1,n,1]) / Float(32)
        for cache in caches {
            let q=MLXArray.zeros([b,2,n,cache.kind.headDim],dtype: .float32)
            let result: MLXArray
            if let owner=cache.kind.sharesKVWithLayer {
                result=cache.attendBorrowing(source: caches[owner],queries: q,scale: 0.125,sinks: nil)
            } else {
                let k=broadcast(hidden,to: [b,1,n,cache.kind.headDim])
                let v=broadcast(hidden,to: [b,1,n,cache.kind.valueHeadDim])
                result=cache.updateAndAttend(queries: q,keys: k,values: v,scale: 0.125,sinks: nil)
            }
            hidden=mean(result,axes: [1,3],keepDims: true)
        }
        let target=MLX.round(hidden.reshaped([b,n,1])*Float(128)).asType(.int32)%31
        return MLX.where(MLXArray(Int32(0)..<Int32(32)) .== target,Float(10),Float(-10))
    }
}

final class CBv2ContiguousHistoricalCheckpointTests: XCTestCase {
    private var chunk: Int { max(32,CBv2AttentionV1.queryBlockSize) }
    private func engine(_ store: CompleteCheckpointFixtureStore, equalWidths: Bool = false) -> (EngineV2,CBv2ContiguousKVBackend) {
        let key = equalWidths ? 128 : 192
        let kinds: [CBv2LayerKind] = [
            .init(attention: .slidingWindow(17),headDim: key,valueHeadDim: 128,kvHeads: 1,queryHeads: 2),
            .init(attention: .full,headDim: key,valueHeadDim: 128,kvHeads: 1,queryHeads: 2),
            .init(attention: .slidingWindow(17),sharesKVWithLayer: 0,headDim: key,valueHeadDim: 128,kvHeads: 1,queryHeads: 2),
            .init(attention: .full,sharesKVWithLayer: 1,headDim: key,valueHeadDim: 128,kvHeads: 1,queryHeads: 2)]
        let backend=CBv2ContiguousKVBackend(config: .init(bytesCapacity: 128 << 20,kvDType: .float32))
        let engine=EngineV2(model: ContiguousHistoricalAttentionModel(),layerKinds: kinds,backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),sampler: CBv2DefaultSampler(fallbackSeed: 0),
            schedulerConfig: .init(maxConcurrentRequests: 2,maxBatchedTokensPerStep: chunk,prefillChunkSize: chunk,
                                   maxWaiting: 4,enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0),completePrefixCache: store)
        return (engine,backend)
    }
    private func request(_ id: UInt64) -> CBv2Request {
        .init(id: .init(id),promptTokens: (0..<3*chunk+1).map { ($0*7)%29 },
            sampling: .init(temperature: 0,seed: 0),maxTokens: 4,cacheSalt: "tenant",prefixCacheReceiptID: .init(id+1000))
    }

    func testRealEngineEarlierWindowRestartMatchesColdTokens() async throws {
        let store=CompleteCheckpointFixtureStore(segmentBytes: 258), (coldEngine,coldBackend)=engine(store)
        XCTAssertEqual(coldEngine.completeCheckpointCodec?.backendLayout,CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout)
        XCTAssertEqual(coldEngine.admissionForTesting.fullKVBytesPerToken,1280,
                       "observed FP32 must raise the default two-byte ledger")
        let cold=await cbv2SchedCollect(try coldEngine.submit(request(1)))
        XCTAssertEqual(cold.finishReason,.length)
        XCTAssertEqual(store.saved.map(\.manifest.position),[chunk,3*chunk])
        XCTAssertEqual(coldBackend.bytesReserved,0); XCTAssertEqual(coldEngine.admissionForTesting.bytesReserved,0)
        await coldEngine.shutdown()
        let reopened=CompleteCheckpointFixtureStore(archives: store.saved.filter { $0.manifest.position == chunk })
        let (warmEngine,warmBackend)=engine(reopened), warmRequest=request(2)
        XCTAssertTrue(try reopened.stage(engine: warmEngine,request: warmRequest))
        let warm=await cbv2SchedCollect(try warmEngine.submit(warmRequest))
        XCTAssertEqual(warm.tokens,cold.tokens)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved,chunk)
        XCTAssertEqual(warm.usage?.prefixCacheReplayTokens,0)
        XCTAssertEqual(warmBackend.bytesReserved,0); XCTAssertEqual(warmEngine.admissionForTesting.bytesReserved,0)
        await warmEngine.shutdown()
    }

    func testPostTransferBackendRefusalFallsBackToColdTokensWithoutOrphans() async throws {
        let seed = CompleteCheckpointFixtureStore(), (cold, _) = engine(seed)
        let expected = await cbv2SchedCollect(try cold.submit(request(40)))
        await cold.shutdown()
        let store = CompleteCheckpointFixtureStore(archives: seed.saved.filter { $0.manifest.position == chunk })
        let (warm, backend) = engine(store), input = request(41)
        XCTAssertTrue(try store.stage(engine: warm, request: input))
        backend.checkpointBeforeRegistration = { throw MLXError.caught("intentional imported registration refusal") }
        let actual = await cbv2SchedCollect(try warm.submit(input))
        XCTAssertEqual(actual.finishReason, .length)
        XCTAssertEqual(actual.tokens, expected.tokens)
        XCTAssertEqual(store.releaseCount, 1)
        await warm.shutdown()
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(warm.admissionForTesting.bytesReserved, 0)
    }

    func testImportedNaturalFinishRetainsChargeWhilePublicationIsBlocked() async throws {
        let seed = CompleteCheckpointFixtureStore(), (cold, _) = engine(seed)
        _ = await cbv2SchedCollect(try cold.submit(request(20)))
        await cold.shutdown()
        let gate = CheckpointPublicationGate()
        let store = CompleteCheckpointFixtureStore(archives: seed.saved.filter { $0.manifest.position == chunk }, gate: gate)
        let (warm, backend) = engine(store), input = request(21)
        XCTAssertTrue(try store.stage(engine: warm, request: input))
        let collection = Task { await cbv2SchedCollect(try warm.submit(input)) }
        let entered = await Task.detached { gate.waitUntilEntered() }.value
        XCTAssertTrue(entered)
        XCTAssertEqual(store.releaseCount, 1, "the staging callback transferred, not retained as a second reservation")
        XCTAssertGreaterThan(backend.bytesReserved, 0)
        XCTAssertGreaterThan(warm.admissionForTesting.bytesReserved, 0)
        warm.cancel(input.id) // late cancellation cannot refund publication's aliases
        XCTAssertThrowsError(try warm.submit(input))
        await warm.shutdown() // store close unblocks publication and drains owners
        let result = try await collection.value
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(warm.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(store.releaseCount, 1)
    }

    func testImportedCancellationAndUnconsumedStageShutdownReleaseOnce() async throws {
        let seed = CompleteCheckpointFixtureStore(), (cold, _) = engine(seed)
        _ = await cbv2SchedCollect(try cold.submit(request(30)))
        await cold.shutdown()
        for consume in [false, true] {
            let store = CompleteCheckpointFixtureStore(archives: seed.saved.filter { $0.manifest.position == chunk })
            let (warm, backend) = engine(store), input = request(31)
            XCTAssertTrue(try store.stage(engine: warm, request: input))
            XCTAssertGreaterThan(warm.admissionForTesting.bytesReserved, 0)
            if consume {
                warm.loopForTesting.onEngineQueueSync { warm.loopForTesting.suspendStepExecutionAtCountForTesting = 1 }
                let stream = try warm.submit(input)
                let entered = await cbv2SchedWait { warm.stepCount >= 1 }
                XCTAssertTrue(entered)
                warm.cancel(input.id)
                warm.loopForTesting.onEngineQueueSync { warm.loopForTesting.suspendStepExecutionAtCountForTesting = nil }
                let result = await cbv2SchedCollect(stream)
                XCTAssertEqual(result.finishReason, .cancelled)
            }
            await warm.shutdown()
            XCTAssertEqual(store.releaseCount, 1)
            XCTAssertEqual(backend.bytesReserved, 0)
            XCTAssertEqual(warm.admissionForTesting.bytesReserved, 0)
        }
    }

    func testLegacyEqualWidthContiguousAttentionOnlyRemainsUnactivated() async {
        let (value,_)=engine(CompleteCheckpointFixtureStore(),equalWidths: true)
        XCTAssertNil(value.completeCheckpointCodec)
        await value.shutdown()
    }

    func testNativeCopyFailureTerminatesWithoutDonation() async throws {
        for construction in [true,false] {
            let store=CompleteCheckpointFixtureStore(), (value,backend)=engine(store)
            value.loopForTesting.onEngineQueueSync {
                value.completeCheckpointCapture?.makeContiguousCheckpoint = { codec, position, chunk, state in
                    if construction { throw MLXError.caught("intentional contiguous construction failure") }
                    let candidate=try CBv2ContiguousHistoricalCheckpoint(codec: codec,position: position,chunkSize: chunk,state: state)
                    candidate.evaluate = { arrays in
                        try withError { eval(arrays) }
                        throw MLXError.caught("intentional contiguous copy failure")
                    }
                    return candidate
                }
            }
            let result=await cbv2SchedCollect(try value.submit(request(3)))
            if case .error? = result.finishReason {} else { XCTFail("copy error became an optional cache miss") }
            XCTAssertTrue(result.tokens.isEmpty); XCTAssertTrue(store.saved.isEmpty)
            await value.shutdown()
            XCTAssertEqual(backend.bytesReserved,0); XCTAssertEqual(value.admissionForTesting.bytesReserved,0)
        }
    }

    func testCancellationAndShutdownDrainCapturedWindowOwner() async throws {
        for shutdown in [false,true] {
            let store=CompleteCheckpointFixtureStore(), (value,backend)=engine(store)
            let entered=DispatchSemaphore(value: 0), release=DispatchSemaphore(value: 0)
            let first=HistoricalCaptureOnce()
            value.loopForTesting.onEngineQueueSync {
                value.completeCheckpointCapture?.makeContiguousCheckpoint = { codec, position, chunk, state in
                    let candidate=try CBv2ContiguousHistoricalCheckpoint(codec: codec,position: position,chunkSize: chunk,state: state)
                    candidate.evaluate = { arrays in
                        if first.claim() {
                            entered.signal(); _=release.wait(timeout: .now()+10)
                        }
                        try withError { eval(arrays) }
                    }
                    return candidate
                }
            }
            let stream=try value.submit(request(4)), collected=Task { await cbv2SchedCollect(stream) }
            let blocked=await withCheckedContinuation { continuation in
                DispatchQueue.global().async { continuation.resume(returning: entered.wait(timeout: .now()+5) == .success) }
            }
            XCTAssertTrue(blocked); XCTAssertEqual(value.stepCount,1)
            XCTAssertGreaterThan(value.admissionForTesting.bytesReserved,0)
            let drain=shutdown ? Task { await value.shutdown() } : nil
            if !shutdown { value.cancel(.init(4)) }
            release.signal()
            let result=await collected.value
            if let drain { await drain.value }
            else { XCTAssertEqual(result.finishReason,.cancelled); await value.shutdown() }
            XCTAssertEqual(backend.bytesReserved,0); XCTAssertEqual(value.admissionForTesting.bytesReserved,0)
        }
    }
}
