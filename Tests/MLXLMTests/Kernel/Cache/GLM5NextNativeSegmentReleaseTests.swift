import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// The protected native MTP profile production serves GLM with: an issued
    /// pool, its process owner and complete-prefix store. Such a pool skipped
    /// `trimFreeSegments`, so it kept every segment it ever committed, free
    /// ones included, charged as the slot's physical floor until the pool
    /// retired; each complete-checkpoint import attached more. On a 256 GB Mac
    /// that floor ratcheted to the slot's grant, and then a lone request on an
    /// otherwise idle engine could not reserve its state and first step's
    /// scratch beside it and failed `kv_capacity_exhausted`.
    @Suite(.serialized)
    struct GLM5NextNativeSegmentReleaseTests {
        /// CI shards run suites side by side on one GPU; a step can take seconds.
        private static let collectTimeout: TimeInterval = 180

        private final class ProcessBudget: CBv2ProcessMemoryOwner, @unchecked Sendable {
            enum Failure: Error { case exceeded }
            private let lock = NSLock()
            private var charge: UInt64 = 0
            private var coverage: UInt64 = 0
            private var retired = false
            var charged: UInt64 { lock.withLock { charge } }
            func replaceCharge(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard bytes <= (512 << 20), bytes >= coverage,
                        !retired || bytes <= charge
                    else { throw Failure.exceeded }
                    charge = bytes
                }
            }
            func recordMaterialization(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard !retired, bytes >= coverage, bytes <= charge else {
                        throw Failure.exceeded
                    }
                    coverage = bytes
                }
            }
            func withdrawCoverage(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard bytes <= coverage else { throw Failure.exceeded }
                    coverage -= bytes
                }
            }
            func retire() { lock.withLock { retired = true } }
        }

        private final class Counter: @unchecked Sendable {
            private let lock = NSLock()
            private var count = 0
            var value: Int { lock.withLock { count } }
            func increment() { lock.withLock { count += 1 } }
        }

        /// Native store adapter over the shared fixture store.
        private final class NativeStore: CBv2NativeCompletePrefixCache, @unchecked Sendable {
            let base: CompleteCheckpointFixtureStore
            init(_ base: CompleteCheckpointFixtureStore) { self.base = base }
            var identity: CBv2CompleteCheckpointIdentity { base.identity }
            func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
                base.acceptsCheckpoint(position: position, packedBytes: packedBytes)
            }
            func takeStaged(
                requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
                maximumSequenceLength: Int
            ) -> CBv2StagedCompleteCheckpoint? {
                base.takeStaged(
                    requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                    maximumSequenceLength: maximumSequenceLength)
            }
            func donate(
                _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
                tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
            ) {
                base.donate(
                    source, requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                    completion: completion)
            }
            func close() { base.close() }
            func closeAndWait() async { base.close() }
        }

        private func model(layers: Int = 4) throws -> GLM5NextTextModel {
            let encoded = try JSONEncoder().encode(
                GLM5NextInventoryFixture.configuration(layers: layers))
            var root = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            text["kv_lora_rank"] = 64
            root["text_config"] = text
            let config = try JSONDecoder().decode(
                GLM5NextConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root))
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "segment-release-tiny.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .float32)
            return try GLM5NextTextModel(configuration: config.textConfig, inventory: inventory) {
                name in
                try Self.tensor(name, inventory: inventory)
            }
        }

        private static func tensor(_ name: String, inventory: GLM5NextWeightInventory) throws
            -> MLXArray
        {
            let shape = try #require(inventory.entries[name]?.tensor.shape)
            let count = shape.reduce(1, *)
            let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 17 }
            let values: [Float]
            if name.contains("norm.weight") || name.hasSuffix("_scale") {
                values = Array(repeating: 1, count: count)
            } else if name.hasSuffix("A_log") || name.hasSuffix("dt_bias") {
                values = Array(repeating: -0.5, count: count)
            } else if name.hasSuffix(".bias") || name.hasSuffix("_base") {
                values = Array(repeating: 0, count: count)
            } else {
                values = (0 ..< count).map { Float(($0 * 3 + salt) % 19 - 9) * 0.015 }
            }
            return MLXArray(values, shape)
        }

        /// The idle-release setting the engine reads at construction.
        private static func withRelease<T>(_ value: String?, _ body: () throws -> T) rethrows -> T {
            let key = "DARKBLOOM_NATIVE_SEGMENT_RELEASE"
            let previous = ProcessInfo.processInfo.environment[key]
            if let value { setenv(key, value, 1) } else { unsetenv(key) }
            defer {
                if let previous { setenv(key, previous, 1) } else { unsetenv(key) }
            }
            return try body()
        }

        /// What `trimSegments` would retire now: whole free segments beyond the
        /// pages the pool's rows reserve, read from the pool's own maps.
        private static func freeSegmentBytes(_ pool: PagedKVPool) -> Int {
            pool.groups.values.reduce(0) { total, group in
                var remaining = group.committedUsablePages
                var bytes = 0
                for index in group.segments.keys.sorted(by: >) {
                    guard let segment = group.segments[index] else { continue }
                    let usable = segment.pages.count - 1
                    guard remaining - usable >= group.pagesReserved,
                        segment.pages.dropFirst().allSatisfy({ group.refCounts[$0] == 0 })
                    else { continue }
                    remaining -= usable
                    bytes += segment.allocatedBytes
                }
                return total + bytes
            }
        }

        private struct Served {
            let engine: EngineV2
            let backend: PagedKVBackend
            let process: ProcessBudget
        }

        /// The production profile at test scale: rectangular MTP, an issued
        /// pool with four-page segments so one request spans several.
        private func engine(store: NativeStore?, release: String?) throws -> Served {
            let target = try model()
            let predictor = try GLM5NextMTP(target: target) { name in
                try Self.tensor(name, inventory: target.inventory)
            }
            let verification = GLM5NextMTPVerification.rectangular(depth: 1)
            let factory = try GLM5NextMTPExecution(
                target: target, predictor: predictor, maximumInputs: 128,
                storage: .contiguous, verification: verification)
            let assistant = try GLM5NextMTPAdapter(
                target: target, predictor: predictor, factory: factory)
            let limits = try CBv2PagedGatheredAttentionLimits(
                maximumBatchSize: 1, maximumQueryTokens: 32, maximumContextTokens: 128,
                maximumInFlightGraphs: 2, maximumScratchBytes: 512 << 20,
                admissionMode: .stepOwned(.pinnedMetal))
            let kind = try #require(target.cbv2LayerKinds.first)
            let pageBytes = CBv2PagedDefaults.pageSize * (kind.headDim + kind.valueHeadDim) * 4
            let process = ProcessBudget()
            let resources = try factory.makeNativeResources(
                assistant: assistant,
                config: .init(
                    capacityBytes: 512 << 20, dtype: .float32,
                    maxPrefillChunk: 32, nominalMaxSequenceLength: 128,
                    maxBufferLength: 16 << 20, segmentSizeBytes: 4 * pageBytes,
                    layerDTypes: target.cbv2CompleteCheckpointKVDTypes,
                    gatheredAttention: limits),
                processMemoryOwner: process, construction: NativeConstructionScope(),
                completePrefixCache: store)
            let engine = Self.withRelease(release) {
                EngineV2(
                    model: resources.model, layerKinds: target.cbv2LayerKinds,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    sampler: CBv2GreedySampler(),
                    schedulerConfig: .init(
                        maxConcurrentRequests: 1, maxBatchedTokensPerStep: 32,
                        prefillChunkSize: 32, maxConcurrentPartialPrefills: 1, maxWaiting: 4,
                        enablePrefixCache: store != nil),
                    admissionConfig: target.cbv2NativeAdmissionConfig,
                    completePrefixCache: store, mtpDrafter: assistant,
                    mtpConfig: .init(
                        enabled: true, maxDraftTokens: verification.depth,
                        maxSpeculativeBatch: 1, fixedDraftTokens: verification.depth,
                        verificationMode: verification.engineMode),
                    processMemoryOwner: process, nativeCompletionTracking: true,
                    nativeExecutionContract: resources.contract)
            }
            #expect(resources.backend.nativeModelBinding?.isGLM5Next == true)
            #expect(engine.mtpInactiveReason == nil)
            return Served(engine: engine, backend: resources.backend, process: process)
        }

        private func committed(_ served: Served) -> Int {
            served.engine.loopForTesting.onEngineQueueSync { served.backend.pool.bytesMaterialized }
        }

        private func isServed(_ reason: CBv2FinishReason?) -> Bool {
            reason == .length || reason == .stop
        }

        /// An idle engine returns its issued pool's free segments and their
        /// physical floor, and serves the next request on fresh segments.
        @Test func idleEngineReturnsIssuedPoolSegments() async throws {
            let served = try engine(store: nil, release: "0.05")
            let request = CBv2Request(
                id: .init(4101), promptTokens: (0 ..< 9).map { 4 + $0 % 11 },
                sampling: .init(temperature: 0), maxTokens: 100)
            let first = await cbv2SchedCollect(
                try served.engine.submit(request), timeoutSeconds: Self.collectTimeout)
            #expect(isServed(first.finishReason))
            #expect(first.tokens.count > 0)
            let returned = await cbv2SchedWait(timeoutSeconds: 60) { committed(served) == 0 }
            #expect(returned, "the idle engine kept \(committed(served)) committed bytes")
            #expect(served.backend.bytesReserved == 0 && served.backend.bytesInUse == 0)
            let again = await cbv2SchedCollect(
                try served.engine.submit(
                    .init(
                        id: .init(4102), promptTokens: request.promptTokens,
                        sampling: .init(temperature: 0), maxTokens: 100)),
                timeoutSeconds: Self.collectTimeout)
            #expect(again.finishReason == first.finishReason && again.tokens == first.tokens)
            let returnedAgain = await cbv2SchedWait(timeoutSeconds: 60) { committed(served) == 0 }
            #expect(returnedAgain)
            await served.engine.shutdown()
            #expect(served.process.charged == 0)
        }

        /// Production's ratchet: every restore attaches the staged checkpoint's
        /// segments, and the free ones stayed. Now each idle period returns
        /// them, so repeated restores leave no floor and stay exact.
        @Test func restoresDoNotRatchetIssuedPoolFloor() async throws {
            let request = CBv2Request(
                id: .init(4301), promptTokens: (0 ..< 73).map { 4 + $0 % 20 },
                sampling: .init(temperature: 0), maxTokens: 6,
                cacheSalt: "tenant-release", prefixCacheReceiptID: .init(14301))
            let producerStore = NativeStore(.init(segmentBytes: 28))
            let producer = try engine(store: producerStore, release: "0.05")
            let produced = await cbv2SchedCollect(
                try producer.engine.submit(request), timeoutSeconds: Self.collectTimeout)
            #expect(isServed(produced.finishReason))
            guard case .quiescent = await producer.engine.shutdownReportingNativeCompletion() else {
                Issue.record("producer did not shut down cleanly")
                return
            }
            #expect(!producerStore.base.saved.isEmpty)

            let consumerStore = NativeStore(
                .init(archives: producerStore.base.saved, segmentBytes: 28))
            let consumer = try engine(store: consumerStore, release: "0.05")
            var floors: [Int] = []
            for round in 0 ..< 3 {
                let restored = CBv2Request(
                    id: .init(UInt64(4302 + round)), promptTokens: request.promptTokens,
                    sampling: .init(temperature: 0), maxTokens: 6, cacheSalt: "tenant-release",
                    prefixCacheReceiptID: .init(UInt64(14302 + round)))
                #expect(try consumerStore.base.stage(engine: consumer.engine, request: restored))
                let resumed = await cbv2SchedCollect(
                    try consumer.engine.submit(restored), timeoutSeconds: Self.collectTimeout)
                #expect(resumed.tokens == produced.tokens)
                #expect((resumed.usage?.prefixCachePrefillTokensSaved ?? 0) > 0)
                _ = await cbv2SchedWait(timeoutSeconds: 60) { committed(consumer) == 0 }
                floors.append(committed(consumer))
            }
            #expect(floors == [0, 0, 0], "idle floor after each restore: \(floors)")
            guard case .quiescent = await consumer.engine.shutdownReportingNativeCompletion() else {
                Issue.record("consumer did not shut down cleanly")
                return
            }
            #expect(consumer.process.charged == 0)
        }

        /// The production failure: a lone request whose first step cannot
        /// reserve its scratch while free segments hold the bytes. It was
        /// finished `kv_capacity_exhausted`; now the next step returns those
        /// segments and the row is served, exactly.
        @Test func loneRowRefusedBesideFreeSegmentsIsServedAfterTheirRelease() async throws {
            let reference = try engine(store: nil, release: "3600")
            let small = CBv2Request(
                id: .init(4201), promptTokens: [9, 10, 11], sampling: .init(temperature: 0),
                maxTokens: 4)
            let expected = await cbv2SchedCollect(
                try reference.engine.submit(small), timeoutSeconds: Self.collectTimeout)
            await reference.engine.shutdown()
            #expect(isServed(expected.finishReason))

            // Idle release is an hour away: only the refusal path can return them.
            let served = try engine(store: nil, release: "3600")
            let loop = served.engine.loopForTesting
            let admission = served.engine.admissionForTesting
            let large = await cbv2SchedCollect(
                try served.engine.submit(
                    .init(
                        id: .init(4202), promptTokens: [4, 5, 6, 7, 8],
                        sampling: .init(temperature: 0), maxTokens: 100)),
                timeoutSeconds: Self.collectTimeout)
            #expect(isServed(large.finishReason))
            let free = loop.onEngineQueueSync { Self.freeSegmentBytes(served.backend.pool) }
            #expect(free > 0)

            let probe = CBv2StepScratchProbe()
            let refusals = Counter()
            let lone = CBv2RequestID(4203)
            loop.onEngineQueueSync {
                served.backend.pool.stepScratchReservationObserverForTesting = {
                    _, requests, _ in
                    probe.release()
                    guard requests.contains(lone), Self.freeSegmentBytes(served.backend.pool) > 0
                    else { return }
                    refusals.increment()
                    probe.exhaust(admission)
                }
            }
            let result = await cbv2SchedCollect(
                try served.engine.submit(
                    .init(
                        id: lone, promptTokens: small.promptTokens,
                        sampling: .init(temperature: 0), maxTokens: 4)),
                timeoutSeconds: Self.collectTimeout)
            loop.onEngineQueueSync {
                served.backend.pool.stepScratchReservationObserverForTesting = nil
                probe.release()
            }
            #expect(refusals.value == 1)
            #expect(
                result.finishReason == expected.finishReason && result.tokens == expected.tokens)
            #expect(served.backend.bytesReserved == 0 && served.backend.bytesInUse == 0)
            await served.engine.shutdown()
            #expect(served.process.charged == 0)
        }

        /// A finished request on a 256 GB Mac: idle long enough that the
        /// floor alone remains. Returns the engine and the requests' expected
        /// output for the small request on a fresh engine.
        private func floorBesideIdleEngine() async throws -> (Served, CBv2SchedCollected) {
            let reference = try engine(store: nil, release: "3600")
            let expected = await cbv2SchedCollect(
                try reference.engine.submit(
                    .init(
                        id: .init(4501), promptTokens: [9, 10, 11],
                        sampling: .init(temperature: 0), maxTokens: 4)),
                timeoutSeconds: Self.collectTimeout)
            await reference.engine.shutdown()
            #expect(isServed(expected.finishReason))
            let served = try engine(store: nil, release: "3600")
            let large = await cbv2SchedCollect(
                try served.engine.submit(
                    .init(
                        id: .init(4502), promptTokens: [4, 5, 6, 7, 8],
                        sampling: .init(temperature: 0), maxTokens: 100)),
                timeoutSeconds: Self.collectTimeout)
            #expect(isServed(large.finishReason))
            #expect(
                served.engine.loopForTesting.onEngineQueueSync {
                    Self.freeSegmentBytes(served.backend.pool)
                } > 0)
            return (served, expected)
        }

        /// Refuse a lone request's step scratch while the pool holds free
        /// segments: the production floor, at test scale.
        private func refuseBesideFreeSegments(
            _ served: Served, lone: CBv2RequestID, probe: CBv2StepScratchProbe, refusals: Counter
        ) {
            let admission = served.engine.admissionForTesting
            served.engine.loopForTesting.onEngineQueueSync {
                served.backend.pool.stepScratchReservationObserverForTesting = {
                    _, requests, _ in
                    probe.release()
                    guard requests.contains(lone), Self.freeSegmentBytes(served.backend.pool) > 0
                    else { return }
                    refusals.increment()
                    probe.exhaust(admission)
                }
            }
        }

        /// Production's timing: three in four failures came within 5 s of the
        /// previous request's end, while its checkpoint publication can still
        /// hold native work. The lone row waits for that work, then the release
        /// runs and it is served exactly.
        @Test func loneRowWaitsWhileOtherNativeWorkBlocksTheRelease() async throws {
            let (served, expected) = try await floorBesideIdleEngine()
            let loop = served.engine.loopForTesting
            let binding = try #require(served.backend.nativeModelBinding)
            // Stand-in for a finished request's publication: a live native loan.
            let publication = try loop.onEngineQueueSync {
                try binding.beginWork(nativeData: false)
            }
            let probe = CBv2StepScratchProbe()
            let refusals = Counter()
            let lone = CBv2RequestID(4503)
            refuseBesideFreeSegments(served, lone: lone, probe: probe, refusals: refusals)
            let stream = try served.engine.submit(
                .init(
                    id: lone, promptTokens: [9, 10, 11], sampling: .init(temperature: 0),
                    maxTokens: 4))
            let collected = Task {
                await cbv2SchedCollect(stream, timeoutSeconds: Self.collectTimeout)
            }
            let waited = await cbv2SchedWait(timeoutSeconds: 60) { refusals.value >= 3 }
            #expect(waited)
            try await Task.sleep(nanoseconds: 200_000_000)
            try loop.onEngineQueueSync {
                try publication.requiredDrain()
                #expect(publication.finish())
            }
            let result = await collected.value
            loop.onEngineQueueSync {
                served.backend.pool.stepScratchReservationObserverForTesting = nil
                probe.release()
            }
            #expect(
                result.finishReason == expected.finishReason && result.tokens == expected.tokens)
            #expect(served.backend.bytesReserved == 0 && served.backend.bytesInUse == 0)
            await served.engine.shutdown()
            #expect(served.process.charged == 0)
        }

        /// The wait is bounded: past `nativeSegmentReleaseWaitNanos` the row
        /// fails as it did before the release existed.
        @Test func loneRowWaitIsBounded() async throws {
            let (served, _) = try await floorBesideIdleEngine()
            let loop = served.engine.loopForTesting
            let binding = try #require(served.backend.nativeModelBinding)
            loop.onEngineQueueSync { loop.nativeSegmentReleaseWaitNanos = 200_000_000 }
            let publication = try loop.onEngineQueueSync {
                try binding.beginWork(nativeData: false)
            }
            let probe = CBv2StepScratchProbe()
            let refusals = Counter()
            let lone = CBv2RequestID(4504)
            refuseBesideFreeSegments(served, lone: lone, probe: probe, refusals: refusals)
            let started = Date()
            let result = await cbv2SchedCollect(
                try served.engine.submit(
                    .init(
                        id: lone, promptTokens: [9, 10, 11], sampling: .init(temperature: 0),
                        maxTokens: 4)), timeoutSeconds: Self.collectTimeout)
            let elapsed = Date().timeIntervalSince(started)
            try loop.onEngineQueueSync {
                served.backend.pool.stepScratchReservationObserverForTesting = nil
                probe.release()
                try publication.requiredDrain()
                #expect(publication.finish())
            }
            guard let reason = result.finishReason, case .error(let message) = reason else {
                Issue.record(
                    "expected the bounded capacity failure, got \(String(describing: result.finishReason))"
                )
                await served.engine.shutdown()
                return
            }
            #expect(message.hasPrefix(CBv2KVError.capacityExhaustedFinishPrefix))
            #expect(elapsed >= 0.2 && elapsed < 9, "waited \(elapsed) s (the default wait is 10 s)")
            #expect(refusals.value >= 2)
            await served.engine.shutdown()
        }

        /// `DARKBLOOM_NATIVE_SEGMENT_RELEASE=0` keeps the former behavior: the
        /// pool keeps its backing while idle and the refused lone row fails.
        @Test func killSwitchKeepsFormerBehavior() async throws {
            let served = try engine(store: nil, release: "0")
            let loop = served.engine.loopForTesting
            let admission = served.engine.admissionForTesting
            let large = await cbv2SchedCollect(
                try served.engine.submit(
                    .init(
                        id: .init(4401), promptTokens: [4, 5, 6, 7, 8],
                        sampling: .init(temperature: 0), maxTokens: 100)),
                timeoutSeconds: Self.collectTimeout)
            #expect(isServed(large.finishReason))
            let held = committed(served)
            try await Task.sleep(nanoseconds: 300_000_000)
            #expect(held > 0 && committed(served) == held)

            let probe = CBv2StepScratchProbe()
            let lone = CBv2RequestID(4402)
            loop.onEngineQueueSync {
                served.backend.pool.stepScratchReservationObserverForTesting = {
                    _, requests, _ in
                    probe.release()
                    guard requests.contains(lone), Self.freeSegmentBytes(served.backend.pool) > 0
                    else { return }
                    probe.exhaust(admission)
                }
            }
            let result = await cbv2SchedCollect(
                try served.engine.submit(
                    .init(
                        id: lone, promptTokens: [9, 10, 11], sampling: .init(temperature: 0),
                        maxTokens: 4)), timeoutSeconds: Self.collectTimeout)
            loop.onEngineQueueSync {
                served.backend.pool.stepScratchReservationObserverForTesting = nil
                probe.release()
            }
            guard let reason = result.finishReason, case .error(let message) = reason else {
                Issue.record(
                    "expected the former capacity failure, got \(String(describing: result.finishReason))"
                )
                await served.engine.shutdown()
                return
            }
            #expect(message.hasPrefix(CBv2KVError.capacityExhaustedFinishPrefix))
            await served.engine.shutdown()
        }
    }
}
