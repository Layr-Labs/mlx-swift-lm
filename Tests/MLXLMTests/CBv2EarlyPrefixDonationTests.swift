import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2EarlyPrefixDonationTests: XCTestCase {
    private final class RecordingCache: CBv2PrefixCache, @unchecked Sendable {
        struct Donation {
            let requestID: CBv2RequestID
            let tokens: [Int]
        }

        let inner = PrefixCacheV2(
            config: .init(blockSize: 8, modelName: "early-prefix-test"))
        private let lock = NSLock()
        private let donationGate = DispatchSemaphore(value: 0)
        private let blockedDonationOrdinals: Set<Int>
        private let storeDonations: Bool
        private var started: [CBv2RequestID] = []
        private var recorded: [Donation] = []

        init(
            blockFirstDonation: Bool = false,
            blockedDonationOrdinals: Set<Int> = [],
            storeDonations: Bool = true
        ) {
            self.blockedDonationOrdinals =
                blockedDonationOrdinals.union(blockFirstDonation ? [1] : [])
            self.storeDonations = storeDonations
        }

        var donations: [Donation] {
            lock.lock()
            defer { lock.unlock() }
            return recorded
        }

        var startedDonationIDs: [CBv2RequestID] {
            lock.lock()
            defer { lock.unlock() }
            return started
        }

        func unblockDonations() {
            donationGate.signal()
        }

        func lookup(tokens: [Int], layerKinds: [CBv2LayerKind])
            -> (matched: Int, prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?])?
        {
            inner.lookup(tokens: tokens, layerKinds: layerKinds)
        }

        func lookup(tokens: [Int], layerKinds: [CBv2LayerKind], cacheSalt: String?)
            -> (matched: Int, prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?])?
        {
            inner.lookup(tokens: tokens, layerKinds: layerKinds, cacheSalt: cacheSalt)
        }

        func donate(
            tokens: [Int], state: [CBv2SequenceKV?], layerKinds: [CBv2LayerKind]
        ) {
            inner.donate(tokens: tokens, state: state, layerKinds: layerKinds)
        }

        func donate(
            tokens: [Int],
            snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
            layerKinds: [CBv2LayerKind]
        ) {
            inner.donate(tokens: tokens, snapshots: snapshots, layerKinds: layerKinds)
        }

        func donate(
            requestID: CBv2RequestID, tokens: [Int],
            snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
            layerKinds: [CBv2LayerKind], cacheSalt: String?
        ) {
            lock.lock()
            started.append(requestID)
            let ordinal = started.count
            let shouldBlock = blockedDonationOrdinals.contains(ordinal)
            lock.unlock()
            if shouldBlock { donationGate.wait() }
            lock.lock()
            recorded.append(Donation(requestID: requestID, tokens: tokens))
            lock.unlock()
            if storeDonations {
                inner.donate(
                    tokens: tokens, snapshots: snapshots, layerKinds: layerKinds,
                    cacheSalt: cacheSalt)
            }
        }

        func endAdoption(tokens: [Int], matched: Int) {
            inner.endAdoption(tokens: tokens, matched: matched)
        }

        func endAdoption(tokens: [Int], matched: Int, cacheSalt: String?) {
            inner.endAdoption(tokens: tokens, matched: matched, cacheSalt: cacheSalt)
        }

        func evict(toFit byteBudget: Int) { inner.evict(toFit: byteBudget) }
        var bytesInUse: Int { inner.bytesInUse }
    }

    private final class DelayedModel: CBv2SteppableModel {
        let base: TinyTestModel
        let delay: TimeInterval

        init(base: TinyTestModel, delay: TimeInterval) {
            self.base = base
            self.delay = delay
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            Thread.sleep(forTimeInterval: delay)
            return base.forward(tokens: tokens, caches: caches)
        }
    }

    private func makeEngine(
        model: TinyTestModel, cache: RecordingCache,
        backend: CBv2KVBackend? = nil,
        early: Bool, requestTimeout: TimeInterval = 120,
        maxConcurrentRequests: Int = 2,
        maxPendingEarlyPrefixDonations: Int = 2,
        modelOverride: CBv2SteppableModel? = nil
    ) throws -> EngineV2 {
        let backend = backend
            ?? CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
        let cacheProvider: CBv2LayerCacheProvider
        if let paged = backend as? PagedKVBackend {
            cacheProvider = CBv2LayerCacheBank(caches: paged.makeLayerCaches())
        } else {
            cacheProvider = CBv2LayerCacheBank(layerKinds: model.layerKinds)
        }
        return EngineV2(
            model: modelOverride ?? model,
            layerKinds: model.layerKinds,
            backend: backend,
            cacheProvider: cacheProvider,
            sampler: CBv2DefaultSampler(fallbackSeed: 17),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrentRequests, maxBatchedTokensPerStep: 128,
                prefillChunkSize: 8, maxWaiting: 8, enablePrefixCache: true),
            loopConfig: CBv2EngineLoopConfig(
                requestTimeout: requestTimeout, enableEarlyPrefixDonation: early,
                maxPendingEarlyPrefixDonations: maxPendingEarlyPrefixDonations),
            prefixCache: cache)
    }

    private func request(
        _ id: UInt64, prompt: [Int], maxTokens: Int,
        prefixCacheEnabled: Bool = true, prefixCacheReceiptID: CBv2RequestID? = nil
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: .init(temperature: 0), maxTokens: maxTokens,
            prefixCacheEnabled: prefixCacheEnabled,
            prefixCacheReceiptID: prefixCacheReceiptID)
    }

    private func finishReason(
        from iterator: inout AsyncStream<CBv2Event>.AsyncIterator
    ) async -> CBv2FinishReason? {
        while let event = await iterator.next() {
            if case .finished(let reason, _) = event { return reason }
        }
        return nil
    }

    func testDefaultOffOnlyRunsTerminalDonation() async throws {
        let model = TinyTestModel.make(seed: 0xE411, fullAttentionOnly: true)
        let prompt = makePromptTokens(length: 41, seed: 1)
        let cache = RecordingCache()
        let engine = try makeEngine(model: model, cache: cache, early: false)

        let result = await cbv2SchedCollect(
            try engine.submit(request(1, prompt: prompt, maxTokens: 10)))
        XCTAssertEqual(result.finishReason, .length)
        let terminalDonated = await cbv2SchedWait { cache.donations.count == 1 }
        XCTAssertTrue(terminalDonated)
        await engine.shutdown()

        XCTAssertEqual(cache.donations.first?.requestID, CBv2RequestID(1))
        XCTAssertEqual(cache.donations.first?.tokens.count, prompt.count + 9)
    }

    func testContiguousEarlyDonationPrecedesTerminalAndDeduplicatesBlocks() async throws {
        let model = TinyTestModel.make(seed: 0xE412, fullAttentionOnly: true)
        let prompt = makePromptTokens(length: 41, seed: 2)
        let cache = RecordingCache()
        let engine = try makeEngine(model: model, cache: cache, early: true)

        let result = await cbv2SchedCollect(
            try engine.submit(request(7, prompt: prompt, maxTokens: 10)))
        XCTAssertEqual(result.finishReason, .length)
        let bothDonated = await cbv2SchedWait { cache.donations.count == 2 }
        XCTAssertTrue(bothDonated)
        await engine.shutdown()

        XCTAssertEqual(cache.donations.map(\.requestID), [CBv2RequestID(7), CBv2RequestID(7)])
        XCTAssertEqual(cache.donations.map(\.tokens.count), [prompt.count, prompt.count + 9])
        XCTAssertEqual(
            cache.inner.stats().entryCount, 1,
            "the longer terminal donation must replace/deduplicate the prompt entry")
    }

    func testReceiptIdentityCorrelatesEarlyAndTerminalDonations() async throws {
        let model = TinyTestModel.make(seed: 0xE418, fullAttentionOnly: true)
        let prompt = makePromptTokens(length: 41, seed: 8)
        let cache = RecordingCache()
        let engine = try makeEngine(model: model, cache: cache, early: true)
        let receiptID = CBv2RequestID(70_001)

        let result = await cbv2SchedCollect(
            try engine.submit(
                request(
                    70, prompt: prompt, maxTokens: 3,
                    prefixCacheReceiptID: receiptID)))
        XCTAssertEqual(result.finishReason, .length)
        let bothDonated = await cbv2SchedWait { cache.donations.count == 2 }
        XCTAssertTrue(bothDonated)
        await engine.shutdown()

        XCTAssertEqual(cache.donations.map(\.requestID), [receiptID, receiptID])
    }

    func testCancelledQueuedEarlyDonationRetainsBackendAccountingUntilMaterialized()
        async throws
    {
        let model = TinyTestModel.make(seed: 0xE419, fullAttentionOnly: true)
        let delayed = DelayedModel(base: model, delay: 0.02)
        let prompt = makePromptTokens(length: 41, seed: 9)
        // One request reserves 13,056 bytes. This ceiling admits exactly two
        // such requests through both AdmissionV2 and the backend, leaving too
        // little room for even a one-token third request.
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 28_000))
        let cache = RecordingCache(blockFirstDonation: true)
        let engine = try makeEngine(
            model: model, cache: cache, backend: backend, early: true,
            modelOverride: delayed)

        let streamA = try engine.submit(request(80, prompt: prompt, maxTokens: 10))
        let streamB = try engine.submit(request(81, prompt: prompt, maxTokens: 10))
        var iteratorA = streamA.makeAsyncIterator()
        var iteratorB = streamB.makeAsyncIterator()
        guard case .delta? = await iteratorA.next(), case .delta? = await iteratorB.next() else {
            cache.unblockDonations()
            await engine.shutdown()
            return XCTFail("both donors must sample before cancellation")
        }
        let firstDonationBlocked = await cbv2SchedWait {
            cache.startedDonationIDs == [CBv2RequestID(80)]
        }
        XCTAssertTrue(firstDonationBlocked)
        let retainedBytesInUse = backend.bytesInUse
        let retainedBytesReserved = backend.bytesReserved
        XCTAssertGreaterThan(retainedBytesInUse, 0)
        XCTAssertGreaterThan(retainedBytesReserved, 0)

        // B's early job is queued behind blocked A and has not started. Both
        // cancellations normally finish through an in-flight-step release.
        engine.cancel(CBv2RequestID(81))
        engine.cancel(CBv2RequestID(80))
        let finishB = await finishReason(from: &iteratorB)
        let finishA = await finishReason(from: &iteratorA)
        XCTAssertEqual(finishB, .cancelled)
        XCTAssertEqual(finishA, .cancelled)
        XCTAssertEqual(cache.startedDonationIDs, [CBv2RequestID(80)])
        XCTAssertEqual(backend.bytesInUse, retainedBytesInUse)
        XCTAssertEqual(backend.bytesReserved, retainedBytesReserved)

        XCTAssertThrowsError(
            try backend.makeSequenceState(
                layerKinds: model.layerKinds, promptLength: prompt.count,
                maxLength: prompt.count + 1)
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected retained accounting to block readmission, got \(error)")
            }
        }

        cache.unblockDonations()
        let bothMaterialized = await cbv2SchedWait { cache.donations.count == 2 }
        XCTAssertTrue(bothMaterialized)
        let released = await cbv2SchedWait { backend.bytesReserved == 0 }
        XCTAssertTrue(released, "backend release must follow queued donation materialization")

        let readmitted = try backend.makeSequenceState(
            layerKinds: model.layerKinds, promptLength: prompt.count,
            maxLength: prompt.count + 1)
        XCTAssertGreaterThan(backend.bytesReserved, 0)
        backend.release(readmitted)
        XCTAssertEqual(backend.bytesReserved, 0)
        await engine.shutdown()
    }

    func testCompletedEarlyDonationCancellationIgnoresUnrelatedBlockedDonation()
        async throws
    {
        let model = TinyTestModel.make(seed: 0xE421, fullAttentionOnly: true)
        let delayed = DelayedModel(base: model, delay: 0.005)
        let prompt = makePromptTokens(length: 41, seed: 11)
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
        let cache = RecordingCache(blockedDonationOrdinals: [2])
        let engine = try makeEngine(
            model: model, cache: cache, backend: backend, early: true,
            modelOverride: delayed)

        let completedID = CBv2RequestID(100)
        let blockedID = CBv2RequestID(101)
        let completedStream = try engine.submit(
            request(completedID.raw, prompt: prompt, maxTokens: 1_000))
        var completedIterator = completedStream.makeAsyncIterator()
        guard case .delta? = await completedIterator.next() else {
            return XCTFail("completed donor must reach its first sample")
        }

        let blockedStream = try engine.submit(
            request(blockedID.raw, prompt: prompt, maxTokens: 1_000))
        var blockedIterator = blockedStream.makeAsyncIterator()
        guard case .delta? = await blockedIterator.next() else {
            return XCTFail("unrelated donor must reach its first sample")
        }
        let unrelatedDonationBlocked = await cbv2SchedWait {
            cache.startedDonationIDs == [completedID, blockedID]
                && cache.donations.map(\.requestID) == [completedID]
        }
        XCTAssertTrue(
            unrelatedDonationBlocked,
            "the second donation starting proves the first donation completed")

        let bothReserved = backend.bytesReserved
        engine.cancel(completedID)
        let completedReason = await finishReason(from: &completedIterator)
        XCTAssertEqual(completedReason, .cancelled)
        let completedStateReleased = await cbv2SchedWait {
            backend.bytesReserved < bothReserved
        }
        XCTAssertTrue(
            completedStateReleased,
            "a completed early donation must not retain its state behind unrelated queue work")
        XCTAssertEqual(cache.donations.map(\.requestID), [completedID])

        let blockedReservation = backend.bytesReserved
        engine.cancel(blockedID)
        let blockedReason = await finishReason(from: &blockedIterator)
        XCTAssertEqual(blockedReason, .cancelled)
        XCTAssertEqual(
            backend.bytesReserved, blockedReservation,
            "the request whose own donation is pending must remain charged")
        cache.unblockDonations()
        let allReleased = await cbv2SchedWait { backend.bytesReserved == 0 }
        XCTAssertTrue(allReleased)
        await engine.shutdown()
    }

    func testReusedIDDoesNotInheritOldEarlyDonationCompletion() async throws {
        let model = TinyTestModel.make(seed: 0xE422, fullAttentionOnly: true)
        let delayed = DelayedModel(base: model, delay: 0.005)
        let prompt = makePromptTokens(length: 41, seed: 12)
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
        let cache = RecordingCache(blockedDonationOrdinals: [1, 2])
        let engine = try makeEngine(
            model: model, cache: cache, backend: backend, early: true,
            modelOverride: delayed)
        let reusedID = CBv2RequestID(120)

        let oldStream = try engine.submit(
            request(reusedID.raw, prompt: prompt, maxTokens: 1_000))
        var oldIterator = oldStream.makeAsyncIterator()
        guard case .delta? = await oldIterator.next() else {
            return XCTFail("old request must reach its first sample")
        }
        let oldDonationBlocked = await cbv2SchedWait {
            cache.startedDonationIDs == [reusedID]
        }
        XCTAssertTrue(oldDonationBlocked)
        engine.cancel(reusedID)
        let oldReason = await finishReason(from: &oldIterator)
        XCTAssertEqual(oldReason, .cancelled)

        let newStream = try engine.submit(
            request(reusedID.raw, prompt: prompt, maxTokens: 1_000))
        var newIterator = newStream.makeAsyncIterator()
        guard case .delta? = await newIterator.next() else {
            cache.unblockDonations()
            return XCTFail("reused id must reach its own first sample")
        }
        let bothStatesReserved = backend.bytesReserved

        cache.unblockDonations()
        let newDonationBlocked = await cbv2SchedWait {
            cache.startedDonationIDs == [reusedID, reusedID]
        }
        XCTAssertTrue(newDonationBlocked)
        _ = engine.loopForTesting.pausedIDsSnapshot()
        let oldStateReleased = await cbv2SchedWait {
            backend.bytesReserved < bothStatesReserved
        }
        XCTAssertTrue(oldStateReleased)
        let newReservation = backend.bytesReserved
        XCTAssertGreaterThan(newReservation, 0)

        engine.cancel(reusedID)
        let newReason = await finishReason(from: &newIterator)
        XCTAssertEqual(newReason, .cancelled)
        XCTAssertEqual(
            backend.bytesReserved, newReservation,
            "the old job's completion must not clear the reused id's pending job")

        cache.unblockDonations()
        let released = await cbv2SchedWait { backend.bytesReserved == 0 }
        XCTAssertTrue(released)
        await engine.shutdown()
    }

    func testPendingEarlyDonationCapSkipsExcessJobs() async throws {
        let model = TinyTestModel.make(seed: 0xE420, fullAttentionOnly: true)
        let delayed = DelayedModel(base: model, delay: 0.02)
        let prompt = makePromptTokens(length: 41, seed: 10)
        let cache = RecordingCache(blockFirstDonation: true)
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28))
        let engine = try makeEngine(
            model: model, cache: cache, backend: backend, early: true,
            maxConcurrentRequests: 3, maxPendingEarlyPrefixDonations: 2,
            modelOverride: delayed)

        let ids = [CBv2RequestID(90), CBv2RequestID(91), CBv2RequestID(92)]
        let streams = try ids.map { id in
            try engine.submit(request(id.raw, prompt: prompt, maxTokens: 10))
        }
        var iterators = streams.map { $0.makeAsyncIterator() }
        for index in iterators.indices {
            guard case .delta? = await iterators[index].next() else {
                cache.unblockDonations()
                await engine.shutdown()
                return XCTFail("request \(ids[index]) must reach first sample")
            }
        }
        let firstDonationBlocked = await cbv2SchedWait {
            cache.startedDonationIDs == [ids[0]]
        }
        XCTAssertTrue(firstDonationBlocked)

        let allReserved = backend.bytesReserved
        engine.cancel(ids[2])
        let capSkippedFinish = await finishReason(from: &iterators[2])
        XCTAssertEqual(capSkippedFinish, .cancelled)
        let skippedReleased = await cbv2SchedWait { backend.bytesReserved < allReserved }
        XCTAssertTrue(
            skippedReleased,
            "a request skipped by the early-donation cap must release immediately")

        let pendingReservations = backend.bytesReserved
        for id in ids.prefix(2) { engine.cancel(id) }
        for index in 0 ..< 2 {
            let finish = await finishReason(from: &iterators[index])
            XCTAssertEqual(finish, .cancelled)
        }
        XCTAssertEqual(
            backend.bytesReserved, pendingReservations,
            "requests with pending donations must remain charged until materialization")
        cache.unblockDonations()
        let cappedDonationsMaterialized = await cbv2SchedWait { cache.donations.count == 2 }
        XCTAssertTrue(cappedDonationsMaterialized)
        let allReleased = await cbv2SchedWait { backend.bytesReserved == 0 }
        XCTAssertTrue(allReleased)
        XCTAssertEqual(cache.donations.map(\.requestID), Array(ids.prefix(2)))

        let fourth = CBv2RequestID(93)
        let fourthResult = await cbv2SchedCollect(
            try engine.submit(request(fourth.raw, prompt: prompt, maxTokens: 2)))
        XCTAssertEqual(fourthResult.finishReason, .length)
        let fourthDonated = await cbv2SchedWait {
            cache.donations.map(\.requestID).contains(fourth)
        }
        XCTAssertTrue(fourthDonated, "completed jobs must return capacity to the global cap")
        await engine.shutdown()
    }

    func testEarlySnapshotSurvivesContinuedMutationAndCancellation() async throws {
        let model = TinyTestModel.make(seed: 0xE417, fullAttentionOnly: true)
        let delayed = DelayedModel(base: model, delay: 0.005)
        let prompt = makePromptTokens(length: 41, seed: 7)
        let cache = RecordingCache()
        let engine = try makeEngine(
            model: model, cache: cache, early: true, modelOverride: delayed)

        let stream = try engine.submit(request(8, prompt: prompt, maxTokens: 100))
        var iterator = stream.makeAsyncIterator()
        guard case .delta(_, let firstTokens, _)? = await iterator.next() else {
            return XCTFail("expected first sampled token")
        }
        let earlyDonated = await cbv2SchedWait { cache.donations.count == 1 }
        XCTAssertTrue(earlyDonated)
        XCTAssertEqual(cache.donations.first?.tokens, prompt)

        engine.cancel(CBv2RequestID(8))
        var finish: CBv2FinishReason?
        while let event = await iterator.next() {
            if case .finished(let reason, _) = event {
                finish = reason
                break
            }
        }
        XCTAssertEqual(finish, .cancelled)
        XCTAssertEqual(cache.donations.count, 1, "cancellation must not add terminal donation")

        let reused = await cbv2SchedCollect(
            try engine.submit(request(9, prompt: prompt, maxTokens: 1)))
        await engine.shutdown()
        XCTAssertEqual(reused.usage?.prefixCacheOutcome, .hit)
        XCTAssertEqual(reused.usage?.prefixCachePrefillTokensSaved, 40)
        XCTAssertEqual(reused.tokens.first, firstTokens.first)
    }

    func testCancellationAndDeadlineBeforeFirstSampleDoNotDonate() async throws {
        let model = TinyTestModel.make(seed: 0xE413, fullAttentionOnly: true)
        let delayed = DelayedModel(base: model, delay: 0.05)
        let prompt = makePromptTokens(length: 64, seed: 3)

        let cancelCache = RecordingCache()
        let cancelEngine = try makeEngine(
            model: model, cache: cancelCache, early: true, modelOverride: delayed)
        let cancelledStream = try cancelEngine.submit(request(11, prompt: prompt, maxTokens: 8))
        try? await Task.sleep(nanoseconds: 10_000_000)
        XCTAssertTrue(cancelCache.donations.isEmpty, "no prompt donation before first sample")
        cancelEngine.cancel(CBv2RequestID(11))
        let cancelled = await cbv2SchedCollect(cancelledStream)
        XCTAssertEqual(cancelled.finishReason, .cancelled)
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(cancelCache.donations.isEmpty)
        await cancelEngine.shutdown()

        let errorCache = RecordingCache()
        let errorEngine = try makeEngine(
            model: model, cache: errorCache, early: true, requestTimeout: 0.01,
            modelOverride: delayed)
        let failed = await cbv2SchedCollect(
            try errorEngine.submit(request(12, prompt: prompt, maxTokens: 8)))
        if case .error = failed.finishReason {
            // expected
        } else {
            XCTFail("expected deadline error, got \(String(describing: failed.finishReason))")
        }
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(errorCache.donations.isEmpty)
        await errorEngine.shutdown()
    }

    func testQuantizedAndRequestDisabledPoliciesDoNotDonate() async throws {
        let model = TinyTestModel.make(
            seed: 0xE414, headDim: 64, fullAttentionOnly: true)
        let prompt = makePromptTokens(length: 41, seed: 4)

        let quantizedCache = RecordingCache()
        let quantized = CBv2ContiguousKVBackend(
            config: .init(
                bytesCapacity: 1 << 28, quantization: (groupSize: 64, bits: 8)))
        let quantizedEngine = try makeEngine(
            model: model, cache: quantizedCache, backend: quantized, early: true)
        _ = await cbv2SchedCollect(
            try quantizedEngine.submit(request(20, prompt: prompt, maxTokens: 4)))
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(quantizedCache.donations.isEmpty)
        await quantizedEngine.shutdown()

        let disabledCache = RecordingCache()
        let disabledEngine = try makeEngine(model: model, cache: disabledCache, early: true)
        let disabled = await cbv2SchedCollect(
            try disabledEngine.submit(
                request(21, prompt: prompt, maxTokens: 4, prefixCacheEnabled: false)))
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(disabledCache.donations.isEmpty)
        XCTAssertEqual(disabled.usage?.prefixCacheOutcome, .skippedPolicy)
        await disabledEngine.shutdown()
    }

    func testPagedEarlyDonationIsExplicitlyDisabledButTerminalRemainsSafe() async throws {
        let model = TinyTestModel.make(
            seed: 0xE415, headDim: 64, fullAttentionOnly: true)
        let prompt = makePromptTokens(length: 41, seed: 5)
        let paged: PagedKVBackend
        do {
            paged = try PagedKVBackend(
                layerKinds: model.layerKinds,
                config: PagedKVPoolConfig(
                    capacityBytes: 64 << 20, maxPrefillChunk: 128,
                    nominalMaxSequenceLength: 512))
        } catch let error as CBv2KVError {
            throw XCTSkip("paged backend unavailable on this hardware: \(error)")
        }
        XCTAssertTrue(paged.requiresMaterializedSnapshots)
        let cache = RecordingCache()
        let engine = try makeEngine(
            model: model, cache: cache, backend: paged, early: true)

        _ = await cbv2SchedCollect(
            try engine.submit(request(30, prompt: prompt, maxTokens: 4)))
        let terminalDonated = await cbv2SchedWait { cache.donations.count == 1 }
        XCTAssertTrue(terminalDonated)
        await engine.shutdown()

        XCTAssertEqual(cache.donations.first?.tokens.count, prompt.count + 3)
    }

    func testUsageSeparatesMatchedTokensFromActualPrefillSavings() async throws {
        let model = TinyTestModel.make(seed: 0xE416, fullAttentionOnly: true)
        let prompt = makePromptTokens(length: 41, seed: 6)
        let cache = RecordingCache()
        let engine = try makeEngine(model: model, cache: cache, early: false)

        _ = await cbv2SchedCollect(try engine.submit(request(40, prompt: prompt, maxTokens: 4)))
        let firstDonated = await cbv2SchedWait { cache.donations.count == 1 }
        XCTAssertTrue(firstDonated)
        let hit = await cbv2SchedCollect(
            try engine.submit(request(41, prompt: prompt, maxTokens: 4)))
        await engine.shutdown()

        XCTAssertEqual(hit.usage?.prefixCacheOutcome, .hit)
        XCTAssertEqual(hit.usage?.prefixCacheMatchedTokens, 40)
        XCTAssertEqual(hit.usage?.prefixCachePrefillTokensSaved, 40)
        XCTAssertEqual(hit.usage?.prefixCacheHitTokens, 40)
    }
}
