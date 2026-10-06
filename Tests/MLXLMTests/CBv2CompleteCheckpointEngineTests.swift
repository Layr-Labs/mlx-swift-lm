import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CheckpointPublicationGate: @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0)
    let resume = DispatchSemaphore(value: 0)
    func waitUntilEntered() -> Bool { entered.wait(timeout: .now() + 10) == .success }
    func block() {
        entered.signal()
        _ = resume.wait(timeout: .now() + 10)
    }
}

/// Encoded-byte stand-in for a durable store. It deliberately never keeps MLX
/// arrays; reopening copies only manifests/bytes to simulate provider restart.
final class CompleteCheckpointFixtureStore: CBv2CompletePrefixCache, @unchecked Sendable {
    struct Chunk: Sendable {
        let tensor: Int
        let offset: Int
        let bytes: Data
    }
    struct Archive: Sendable {
        let manifest: CBv2CompleteCheckpointManifest
        let chunks: [Chunk]
    }
    let identity = CBv2CompleteCheckpointIdentity(
        modelAggregateHash: "tiny-qwen", promptContractID: "same-template",
        buildID: "same-build", numericsFingerprint: "native-fp32")
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "test.complete-checkpoint-store")
    private var archives: [Archive]
    private var tickets: [CBv2RequestID: CBv2StagedCompleteCheckpoint] = [:]
    private var closed = false
    private var releases = 0
    private var disarmedAt: [Int] = []
    let maximumPosition: Int
    let segmentBytes: Int
    let gate: CheckpointPublicationGate?
    let completionGate: CheckpointPublicationGate?
    /// Blocks the engine queue inside the FIRST capture's policy probe, so a
    /// test can queue company before the donor's next range is planned.
    let admissionGate: CheckpointPublicationGate?
    private var admissionGateArmed: Bool

    init(
        archives: [Archive] = [], maximumPosition: Int = .max,
        gate: CheckpointPublicationGate? = nil,
        admissionGate: CheckpointPublicationGate? = nil, segmentBytes: Int = 64,
        completionGate: CheckpointPublicationGate? = nil
    ) {
        self.archives = archives
        self.maximumPosition = maximumPosition
        self.segmentBytes = segmentBytes
        self.gate = gate
        self.completionGate = completionGate
        self.admissionGate = admissionGate
        admissionGateArmed = admissionGate != nil
    }
    var saved: [Archive] {
        lock.lock()
        defer { lock.unlock() }
        return archives
    }
    var releaseCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return releases
    }
    /// Positions reported by `recordRecurrentCaptureDisarmed(packedAt:)`, in order.
    var recurrentCaptureDisarmedPackedAt: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return disarmedAt
    }
    func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
        lock.lock()
        let blocks = admissionGateArmed
        admissionGateArmed = false
        lock.unlock()
        if blocks { admissionGate?.block() }
        return position <= maximumPosition
    }
    func recordRecurrentCaptureDisarmed(packedAt position: Int) {
        lock.lock()
        disarmedAt.append(position)
        lock.unlock()
    }

    func takeStaged(
        requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?, maximumSequenceLength: Int
    ) -> CBv2StagedCompleteCheckpoint? {
        lock.lock()
        defer { lock.unlock() }
        return tickets.removeValue(forKey: requestID)
    }

    func stage(engine: EngineV2, request: CBv2Request) throws -> Bool {
        guard let receipt = request.prefixCacheReceiptID else { return false }
        let matching = saved.filter {
            $0.manifest.cacheSalt == request.checkpointCacheSalt
                && $0.manifest.position < request.promptTokens.count
                && $0.manifest.prefixTokens.elementsEqual(
                    request.promptTokens.prefix($0.manifest.position))
        }.max { $0.manifest.position < $1.manifest.position }
        guard let matching else { return false }
        let plan = try engine.planCompleteCheckpointImport(
            manifest: matching.manifest, request: request)
        let sink = try plan.allocate { [self] in
            lock.lock()
            releases += 1
            lock.unlock()
        }
        defer { sink.close() }
        for chunk in matching.chunks {
            try sink.appendSegment(
                tensorIndex: chunk.tensor, byteOffset: chunk.offset, data: chunk.bytes)
        }
        let ready = try sink.finish()
        lock.lock()
        tickets[receipt] = ready
        lock.unlock()
        return true
    }

    func donate(
        _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?, tokens: [Int],
        cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
    ) {
        queue.async { [self, source] in
            gate?.block()
            lock.lock()
            let cancelled = closed
            lock.unlock()
            guard !cancelled else {
                source.close()
                completion([])
                return
            }
            do {
                var chunks: [Chunk] = []
                for (index, tensor) in source.manifest.tensors.enumerated() {
                    var offset = 0
                    while offset < tensor.byteCount {
                        let data = try source.readSegment(
                            tensorIndex: index, byteOffset: offset, maximumBytes: segmentBytes)
                        chunks.append(.init(tensor: index, offset: offset, bytes: data))
                        offset += data.count
                    }
                }
                // The archive is a caller-owned encoded store, not a retained
                // native manifest value. Round-trip the wire form so reopening
                // cannot carry an old engine's host-metadata permit.
                let manifest = try JSONDecoder().decode(
                    CBv2CompleteCheckpointManifest.self,
                    from: JSONEncoder().encode(source.manifest))
                source.close()
                lock.lock()
                let commit = !closed
                if commit { archives.append(.init(manifest: manifest, chunks: chunks)) }
                lock.unlock()
                // Completion can deliver the consumer terminal while this
                // callback still owns the source's host manifest permit.
                withExtendedLifetime(source) {
                    completion(commit ? [source.manifest.position] : [])
                    completionGate?.block()
                }
            } catch {
                source.close()
                completion([])
            }
        }
    }

    /// Call only after the request terminal. GPU/request retirement precedes
    /// terminal delivery, while closed export manifests stay charged until the
    /// final callback aliases leave both publication queues.
    func finishPublicationCallbacks(engine: EngineV2) {
        queue.sync {}
        engine.completeCheckpointCapture?.queue.sync {}
    }

    func close() {
        lock.lock()
        closed = true
        let old = Array(tickets.values)
        tickets.removeAll()
        lock.unlock()
        old.forEach { $0.close() }
        gate?.resume.signal()
    }
}

/// Geometry records seen by the loop's test observer, engine-queue writes.
final class RecurrentGeometryObservations: @unchecked Sendable {
    struct Record {
        let range: Range<Int>
        let cap: Int
        let packed: Bool
        let outcome: String
    }
    private let lock = NSLock()
    private var records: [Record] = []
    func append(range: Range<Int>, cap: Int, packed: Bool = false, outcome: String) {
        lock.lock()
        records.append(.init(range: range, cap: cap, packed: packed, outcome: outcome))
        lock.unlock()
    }
    var snapshot: [Record] {
        lock.lock()
        defer { lock.unlock() }
        return records
    }
}

private final class CompleteCheckpointReceiptLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [(CBv2RequestID, [Int])] = []
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return entries.count
    }
    func append(_ id: CBv2RequestID, _ positions: [Int]) {
        lock.lock()
        entries.append((id, positions))
        lock.unlock()
    }
}

final class CBv2CompleteCheckpointEngineTests: XCTestCase {
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }
    /// `stripe` mirrors production: a solo text request prefills in
    /// `2 * chunk` stripes until company arrives, then in plain chunks.
    private func engine(
        _ store: CompleteCheckpointFixtureStore,
        model: CompleteCheckpointFixtureModel = CompleteCheckpointFixtureModel(),
        stripe: Bool = false, maxConcurrentRequests: Int = 1
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1, modelLayerIndex: 1)
        ]
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
        let largest = stripe ? 2 * chunk : chunk
        let engine = EngineV2(
            model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: maxConcurrentRequests, maxBatchedTokensPerStep: largest,
                prefillChunkSize: chunk, soloPrefillStripeTokens: stripe ? largest : nil,
                maxWaiting: 4, enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store)
        return (engine, backend)
    }

    private func positions(_ store: CompleteCheckpointFixtureStore, salt: String = "tenant")
        -> [Int]
    {
        store.saved.filter { $0.manifest.cacheSalt == salt }.map(\.manifest.position)
    }

    /// Recurrent donors run `CBv2CheckpointRetention` on their chunk ends: the
    /// first, the deepest chunk end at or below the coordinator's hint, and
    /// the rolling latest, published deepest first. A target already held by
    /// the first or deepest endpoint needs no additional copy.
    func testRecurrentDonorKeepsFirstForkTargetAndDeepestFromTheHint() async throws {
        let prompt = Array(repeating: 1, count: 6 * chunk + 1)
        for (hint, expected) in [
            (nil, [6 * chunk, chunk]),
            (0, [6 * chunk, chunk]),
            (4 * chunk + 5, [6 * chunk, 4 * chunk, chunk]),
            (5 * chunk + 1, [6 * chunk, 5 * chunk, chunk]),
            (chunk / 2, [6 * chunk, chunk]),
            (chunk, [6 * chunk, chunk]),
        ] as [(Int?, [Int])] {
            let store = CompleteCheckpointFixtureStore()
            let (engine, backend) = engine(store)
            let result = await cbv2SchedCollect(
                try engine.submit(
                    CBv2Request(
                        id: .init(21), promptTokens: prompt, maxTokens: 2, cacheSalt: "tenant",
                        prefixCacheReceiptID: .init(1021), prefixCheckpointTargetTokens: hint)))
            XCTAssertEqual(result.finishReason, .length)
            XCTAssertEqual(positions(store), expected, "hint \(String(describing: hint))")
            XCTAssertEqual(
                store.saved.map(\.manifest.chunkSize),
                Array(repeating: chunk, count: expected.count))
            XCTAssertEqual(backend.bytesReserved, 0)
            store.finishPublicationCallbacks(engine: engine)
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
            XCTAssertTrue(
                store.recurrentCaptureDisarmedPackedAt.isEmpty,
                "a solo prompt's ragged tail is a geometry disarm, never reported")
            await engine.shutdown()
        }
    }

    /// A prompt ending exactly on a boundary never stages that terminal
    /// checkpoint on the durable path: export refuses `position ==
    /// tokens.count`, so it could only have displaced the real deepest
    /// interior boundary (the loop skips it; the resident bank keeps its
    /// endpoint, see `CBv2RecurrentStateTests`). The final interior endpoint
    /// remains eligible even when it also serves the demanded fork.
    func testRecurrentPromptEndingOnABoundaryPublishesTheLastInteriorOne() async throws {
        for (length, hint, expected) in [
            (3 * chunk, 2 * chunk, [2 * chunk, chunk]),
            (3 * chunk, nil, [2 * chunk, chunk]),
            (6 * chunk, 4 * chunk, [5 * chunk, 4 * chunk, chunk]),
            (6 * chunk, 3 * chunk, [5 * chunk, 3 * chunk, chunk]),
        ] as [(Int, Int?, [Int])] {
            let store = CompleteCheckpointFixtureStore()
            let (engine, backend) = engine(store)
            let result = await cbv2SchedCollect(
                try engine.submit(
                    CBv2Request(
                        id: .init(23), promptTokens: Array(repeating: 1, count: length),
                        maxTokens: 2,
                        cacheSalt: "tenant", prefixCacheReceiptID: .init(1023),
                        prefixCheckpointTargetTokens: hint)))
            XCTAssertEqual(result.finishReason, .length)
            XCTAssertEqual(
                positions(store), expected, "length \(length) hint \(String(describing: hint))")
            XCTAssertFalse(
                store.saved.contains { $0.manifest.position == length },
                "the prompt end must never reach the store")
            XCTAssertEqual(backend.bytesReserved, 0)
            store.finishPublicationCallbacks(engine: engine)
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
            await engine.shutdown()
        }
    }

    func testAdjacentDemandedForkRestoresAfterRestartWithADivergentSuffix() async throws {
        let donorTokens = (0 ..< 6 * chunk + 1).map { ($0 * 5) % 7 }
        let shared = 5 * chunk + 3
        let forkTokens = Array(donorTokens.prefix(shared)) + Array(repeating: 11, count: chunk)
        let store = CompleteCheckpointFixtureStore()
        let (donor, _) = engine(store)
        _ = await cbv2SchedCollect(
            try donor.submit(
                .init(
                    id: .init(71), promptTokens: donorTokens,
                    maxTokens: 3, cacheSalt: "tenant", prefixCacheReceiptID: .init(1071),
                    prefixCheckpointTargetTokens: shared)))
        XCTAssertEqual(positions(store), [6 * chunk, 5 * chunk, chunk])
        store.finishPublicationCallbacks(engine: donor)
        await donor.shutdown()

        let reopened = CompleteCheckpointFixtureStore(archives: store.saved)
        let (warmEngine, warmBackend) = engine(reopened)
        let request = CBv2Request(
            id: .init(72), promptTokens: forkTokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1072))
        XCTAssertTrue(try reopened.stage(engine: warmEngine, request: request))
        let warm = await cbv2SchedCollect(try warmEngine.submit(request))
        XCTAssertEqual(
            warm.usage?.prefixCachePrefillTokensSaved, 5 * chunk,
            "the deeper donor endpoint differs after the shared fork and cannot replace it")
        XCTAssertEqual(warm.usage?.prefixCacheReplayTokens, 0)
        reopened.finishPublicationCallbacks(engine: warmEngine)
        XCTAssertEqual(warmBackend.bytesReserved, 0)
        XCTAssertEqual(warmEngine.admissionForTesting.bytesReserved, 0)
        await warmEngine.shutdown()

        let (coldEngine, _) = engine(CompleteCheckpointFixtureStore())
        let cold = await cbv2SchedCollect(try coldEngine.submit(request))
        XCTAssertEqual(warm.tokens, cold.tokens)
        await coldEngine.shutdown()
    }

    /// An adopter restored at `M` recaptures nothing at or below `M`: no
    /// first, and a fork target only when the hint names one above `M`.
    func testRecurrentAdopterCapturesOnlyAboveItsRestoredBoundary() async throws {
        let prompt = Array(repeating: 1, count: 8 * chunk + 1)
        let donorStore = CompleteCheckpointFixtureStore()
        let (donor, donorBackend) = engine(donorStore)
        let cold = await cbv2SchedCollect(
            try donor.submit(
                CBv2Request(
                    id: .init(31), promptTokens: prompt, maxTokens: 3, cacheSalt: "tenant",
                    prefixCacheReceiptID: .init(1031), prefixCheckpointTargetTokens: 4 * chunk)))
        XCTAssertEqual(cold.finishReason, .length)
        XCTAssertEqual(positions(donorStore), [8 * chunk, 4 * chunk, chunk])
        XCTAssertEqual(donorBackend.bytesReserved, 0)
        await donor.shutdown()

        for (hint, expected) in [
            (6 * chunk + 3, [8 * chunk, 6 * chunk]), (nil, [8 * chunk]),
            (4 * chunk, [8 * chunk]), (2 * chunk, [8 * chunk]),
        ] as [(Int?, [Int])] {
            let reopened = CompleteCheckpointFixtureStore(
                archives: donorStore.saved.filter { $0.manifest.position == 4 * chunk })
            let (adopter, adopterBackend) = engine(reopened)
            let warmRequest = CBv2Request(
                id: .init(32), promptTokens: prompt, maxTokens: 3, cacheSalt: "tenant",
                prefixCacheReceiptID: .init(1032), prefixCheckpointTargetTokens: hint)
            XCTAssertTrue(try reopened.stage(engine: adopter, request: warmRequest))
            let warm = await cbv2SchedCollect(try adopter.submit(warmRequest))
            XCTAssertEqual(warm.tokens, cold.tokens)
            XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, 4 * chunk)
            XCTAssertEqual(
                positions(reopened).filter { $0 != 4 * chunk }, expected,
                "hint \(String(describing: hint))")
            XCTAssertEqual(adopterBackend.bytesReserved, 0)
            reopened.finishPublicationCallbacks(engine: adopter)
            XCTAssertEqual(adopter.admissionForTesting.bytesReserved, 0)
            await adopter.shutdown()
        }
    }

    /// The donor's first range is a solo `2c` stripe; company arrives while
    /// that boundary is being captured, so the later ranges are plain `c`
    /// chunks. Capture is chunk-agnostic: every aligned chunk end after the
    /// switch is a boundary, the cap change disarms nothing, and the store
    /// hears of no disarm: the fixture cannot pack, so no range ran packed.
    func testCapChangeMidPromptKeepsCapturingAndNeverReportsADisarm() async throws {
        let gate = CheckpointPublicationGate()
        let store = CompleteCheckpointFixtureStore(admissionGate: gate)
        let (engine, backend) = engine(store, stripe: true, maxConcurrentRequests: 2)
        let prompt = (0 ..< 6 * chunk + 1).map { ($0 * 5) % 7 }
        let observed = RecurrentGeometryObservations()
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.recurrentGeometryObserverForTesting = {
                id, range, cap, _, phase, outcome in
                guard id == .init(41), phase == "record" else { return }
                observed.append(range: range, cap: cap ?? -1, outcome: outcome)
            }
        }
        let stream = try engine.submit(
            CBv2Request(
                id: .init(41), promptTokens: prompt, maxTokens: 3, cacheSalt: "tenant",
                prefixCacheReceiptID: .init(1041), prefixCheckpointTargetTokens: 4 * chunk + 1))
        let collected = Task { await cbv2SchedCollect(stream) }
        let entered = await Task.detached { gate.waitUntilEntered() }.value
        XCTAssertTrue(entered, "the first stripe boundary reached the store's policy probe")
        let companyStream = try engine.submit(
            CBv2Request(
                id: .init(42), promptTokens: [3, 1, 4], maxTokens: 2, cacheSalt: "other"))
        let company = Task { await cbv2SchedCollect(companyStream) }
        gate.resume.signal()
        let donor = await collected.value
        let companyResult = await company.value
        XCTAssertEqual(donor.finishReason, .length)
        XCTAssertEqual(companyResult.tokens.count, 2)
        let records = observed.snapshot.filter { $0.range.upperBound <= prompt.count }
        let caps = Set(records.map(\.cap))
        XCTAssertTrue(
            caps.contains(2 * chunk) && caps.contains(chunk),
            "the donor must have prefilled under both the stripe and plain chunks: \(records)")
        XCTAssertFalse(
            records.contains { $0.outcome == "disarm" }, "no prompt range disarmed: \(records)")
        // Every aligned range end is a boundary, whatever chunk produced it;
        // the chained successor of the first stripe may itself have been
        // planned as a stripe before the company was visible.
        let captured = records.filter { $0.outcome == "capture" }.map(\.range.upperBound)
        XCTAssertEqual(
            captured, records.map(\.range.upperBound).filter { $0 % chunk == 0 },
            "every aligned range end is a boundary: \(records)")
        XCTAssertTrue(captured.contains(6 * chunk) && captured.count >= 4, "\(captured)")
        XCTAssertTrue(
            store.recurrentCaptureDisarmedPackedAt.isEmpty,
            "a cap change is no disarm and nothing ran packed: \(store.recurrentCaptureDisarmedPackedAt)"
        )
        // Retention over the boundaries that landed, replayed: the first,
        // the deepest at or below the hint, the deepest; deepest first.
        var replay = CBv2CheckpointRetention(
            stride: nil, hintTokens: 4 * chunk + 1)
        for position in captured { _ = replay.commit(position) }
        XCTAssertEqual(positions(store), replay.publication)
        // 4c lands whether the second range was the chained stripe or two
        // plain chunks, and sits two chunks below the deepest.
        XCTAssertEqual(positions(store), [6 * chunk, 4 * chunk, 2 * chunk])
        let capByEnd = Dictionary(
            records.map { ($0.range.upperBound, $0.cap) }, uniquingKeysWith: { _, new in new })
        XCTAssertEqual(
            store.saved.map(\.manifest.chunkSize),
            store.saved.map { capByEnd[$0.manifest.position] ?? -1 },
            "each manifest records the chunk that ended at its boundary")
        XCTAssertEqual(backend.bytesReserved, 0)
        store.finishPublicationCallbacks(engine: engine)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        await engine.shutdown()

        // The deepest restores exactly on a fresh engine, under ordinary
        // scheduling (the solo stripe), not the donor's mixed geometry.
        let reopened = CompleteCheckpointFixtureStore(
            archives: store.saved.filter { $0.manifest.position == 6 * chunk })
        let (second, secondBackend) = self.engine(reopened, stripe: true)
        let warmRequest = CBv2Request(
            id: .init(43), promptTokens: prompt, maxTokens: 3,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1043))
        XCTAssertTrue(try reopened.stage(engine: second, request: warmRequest))
        let warm = await cbv2SchedCollect(try second.submit(warmRequest))
        XCTAssertEqual(warm.tokens, donor.tokens)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, 6 * chunk)
        XCTAssertEqual(secondBackend.bytesReserved, 0)
        reopened.finishPublicationCallbacks(engine: second)
        XCTAssertEqual(second.admissionForTesting.bytesReserved, 0)
        await second.shutdown()
    }

    /// The donor's first range is the solo `2c` stripe; company with three
    /// full chunks of its own arrives while that boundary is being captured,
    /// so a later donor chunk runs in a packed cohort beside a company chunk
    /// of the same length. Packing disarms the donor for the rest of its
    /// prompt and the store hears of it exactly once, at the packed range's
    /// start. The cap changes on either side of it (stripe to chunk before,
    /// chunk back to stripe once the company leaves) report nothing, and no
    /// boundary above the disarm is published. The company opts out of the
    /// prefix cache, so it is no donor and the report is the donor's alone.
    func testPackedCohortDisarmsRecurrentCaptureOnceAndCapChangesReportNothing() async throws {
        let gate = CheckpointPublicationGate()
        let store = CompleteCheckpointFixtureStore(admissionGate: gate)
        let (engine, backend) = engine(
            store, model: PackableCompleteCheckpointFixtureModel(), stripe: true,
            maxConcurrentRequests: 2)
        XCTAssertTrue(engine.packedPrefillActivity().isSupported)
        let prompt = (0 ..< 12 * chunk + 1).map { ($0 * 5) % 7 }
        let observed = RecurrentGeometryObservations()
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.recurrentGeometryObserverForTesting = {
                id, range, cap, packed, phase, outcome in
                guard id == .init(61), phase == "record" else { return }
                observed.append(range: range, cap: cap ?? -1, packed: packed, outcome: outcome)
            }
        }
        let stream = try engine.submit(
            CBv2Request(
                id: .init(61), promptTokens: prompt, maxTokens: 3, cacheSalt: "tenant",
                prefixCacheReceiptID: .init(1061), prefixCheckpointTargetTokens: 8 * chunk + 1))
        let collected = Task { await cbv2SchedCollect(stream) }
        let entered = await Task.detached { gate.waitUntilEntered() }.value
        XCTAssertTrue(entered, "the first stripe boundary reached the store's policy probe")
        let companyStream = try engine.submit(
            CBv2Request(
                id: .init(62), promptTokens: (0 ..< 3 * chunk + 1).map { ($0 * 3) % 5 },
                maxTokens: 2,
                cacheSalt: "other", prefixCacheEnabled: false))
        let company = Task { await cbv2SchedCollect(companyStream) }
        gate.resume.signal()
        let donor = await collected.value
        let companyResult = await company.value
        XCTAssertEqual(donor.finishReason, .length)
        XCTAssertEqual(companyResult.tokens.count, 2)
        XCTAssertTrue(
            engine.packedPrefillActivity().didExecute, "a company chunk packed beside the donor's")
        let records = observed.snapshot.filter { $0.range.upperBound <= prompt.count }
        guard let firstPacked = records.first(where: { $0.packed }) else {
            return XCTFail("no donor range ran packed: \(records)")
        }
        XCTAssertEqual(firstPacked.outcome, "disarm")
        XCTAssertEqual(
            store.recurrentCaptureDisarmedPackedAt, [firstPacked.range.lowerBound],
            "one report per request, at the packed range's start: \(records)")
        let before = records.filter { $0.range.upperBound <= firstPacked.range.lowerBound }
        let after = records.filter { $0.range.lowerBound >= firstPacked.range.upperBound }
        XCTAssertTrue(
            before.contains { $0.cap == 2 * chunk } && firstPacked.cap == chunk,
            "the donor left the stripe for plain chunks before packing: \(records)")
        XCTAssertTrue(
            after.contains { $0.cap == 2 * chunk },
            "the stripe resumed once the company left, a later cap change: \(records)")
        XCTAssertTrue(
            after.allSatisfy { $0.outcome == "disarm" },
            "the disarm holds for the rest of the prompt")
        XCTAssertFalse(
            positions(store).isEmpty, "the first stripe boundary was captured before the disarm")
        XCTAssertTrue(
            positions(store).allSatisfy { $0 <= firstPacked.range.lowerBound },
            "nothing above the disarm is captured: \(positions(store))")
        XCTAssertEqual(backend.bytesReserved, 0)
        store.finishPublicationCallbacks(engine: engine)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        await engine.shutdown()
    }

    /// Mixed partitions in miniature, without company: an adopter restored
    /// at `2c` resumes under the `2c` solo stripe over a prompt whose donor
    /// ran plain `c` chunks, is not held to the donor's chunk size, captures
    /// at its own aligned range ends above the restore point, and matches
    /// the cold run token for token.
    func testAdopterResumesUnderADifferentChunkAndCapturesAlignedEnds() async throws {
        let prompt = Array(repeating: 1, count: 6 * chunk + 1)
        let donorStore = CompleteCheckpointFixtureStore()
        let (donor, donorBackend) = engine(donorStore)
        let cold = await cbv2SchedCollect(
            try donor.submit(
                CBv2Request(
                    id: .init(51), promptTokens: prompt, maxTokens: 3, cacheSalt: "tenant",
                    prefixCacheReceiptID: .init(1051), prefixCheckpointTargetTokens: 2 * chunk)))
        XCTAssertEqual(positions(donorStore), [6 * chunk, 2 * chunk, chunk])
        XCTAssertEqual(donorStore.saved.map(\.manifest.chunkSize), [chunk, chunk, chunk])
        XCTAssertEqual(donorBackend.bytesReserved, 0)
        await donor.shutdown()

        let reopened = CompleteCheckpointFixtureStore(
            archives: donorStore.saved.filter { $0.manifest.position == 2 * chunk })
        let (adopter, adopterBackend) = engine(reopened, stripe: true)
        let observed = RecurrentGeometryObservations()
        var forcedChunk: Int?? = nil
        adopter.loopForTesting.onEngineQueueSync {
            adopter.loopForTesting.recurrentGeometryObserverForTesting = {
                [unowned adopter] id, range, cap, _, phase, outcome in
                guard phase == "record" else { return }
                observed.append(range: range, cap: cap ?? -1, outcome: outcome)
                if forcedChunk == nil {
                    forcedChunk = .some(
                        adopter.loopForTesting.scheduler.record(for: id)?.prefixReusePlan?
                            .recurrentChunkSize)
                }
            }
        }
        let warmRequest = CBv2Request(
            id: .init(52), promptTokens: prompt, maxTokens: 3, cacheSalt: "tenant",
            prefixCacheReceiptID: .init(1052), prefixCheckpointTargetTokens: 4 * chunk + 3)
        XCTAssertTrue(try reopened.stage(engine: adopter, request: warmRequest))
        let warm = await cbv2SchedCollect(try adopter.submit(warmRequest))
        XCTAssertEqual(warm.tokens, cold.tokens)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, 2 * chunk)
        XCTAssertEqual(
            forcedChunk, .some(nil),
            "a complete-checkpoint adopter is not held to the donor's chunk size")
        let prefill = observed.snapshot.filter { $0.range.upperBound <= prompt.count }
        XCTAssertTrue(
            prefill.contains { $0.cap == 2 * chunk },
            "the adopter resumed on the solo stripe: \(prefill)")
        XCTAssertEqual(
            prefill.filter { $0.outcome == "capture" }.map(\.range.upperBound),
            [4 * chunk, 6 * chunk])
        // Above the restore point only: the deepest at or below the hint (4c)
        // and the deepest, each recording the chunk that ended there.
        XCTAssertEqual(positions(reopened).filter { $0 != 2 * chunk }, [6 * chunk, 4 * chunk])
        XCTAssertEqual(
            reopened.saved.filter { $0.manifest.position > 2 * chunk }.map(\.manifest.chunkSize),
            [2 * chunk, 2 * chunk])
        XCTAssertEqual(adopterBackend.bytesReserved, 0)
        reopened.finishPublicationCallbacks(engine: adopter)
        XCTAssertEqual(adopter.admissionForTesting.bytesReserved, 0)
        await adopter.shutdown()
    }

    func testCompleteCacheSkipsRecurrentSpecReadsUntilEligibleCapture() async throws {
        let store = CompleteCheckpointFixtureStore()
        let model = CompleteCheckpointFixtureModel()
        let (engine, _) = engine(store, model: model)

        let decodeOnly = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: .init(5), promptTokens: [1, 2, 3], maxTokens: 6,
                    cacheSalt: "tenant", prefixCacheReceiptID: .init(1005))))
        XCTAssertEqual(decodeOnly.finishReason, .length)
        XCTAssertTrue(store.saved.isEmpty)
        let readsAfterDecode = model.recurrentSpecReads
        let emptyFinalize = CBv2InFlightStep(
            assignments: [], participants: [], sampledRows: [], sampledTokens: nil,
            evalTargets: [], wallStartedNanos: 0)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.captureRecurrentCheckpoints(emptyFinalize)
        }
        XCTAssertEqual(
            model.recurrentSpecReads, readsAfterDecode,
            "finalization without eligible checkpoint geometry must not probe recurrent spec")

        let capture = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: .init(6), promptTokens: Array(repeating: 1, count: chunk + 1), maxTokens: 2,
                    cacheSalt: "tenant", prefixCacheReceiptID: .init(1006))))
        XCTAssertEqual(capture.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [chunk])
        await engine.shutdown()
    }

    func testTerminalLeavesRetainedExportMetadataChargedUntilCallbackReturns() async throws {
        let gate = CheckpointPublicationGate()
        let store = CompleteCheckpointFixtureStore(completionGate: gate)
        let (engine, backend) = engine(store)
        defer { gate.resume.signal() }
        let request = CBv2Request(
            id: .init(12), promptTokens: Array(repeating: 1, count: chunk + 1), maxTokens: 1,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1012))
        let stream = try engine.submit(request)
        let collection = Task { await cbv2SchedCollect(stream) }
        let entered = await Task.detached { gate.waitUntilEntered() }.value
        XCTAssertTrue(entered, "the completed store callback still retains its closed source")
        let result = await collection.value
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(positions(store), [chunk])
        XCTAssertEqual(backend.bytesReserved, 0, "terminal fences GPU/request retirement")
        let metadataBytes = try CBv2CheckpointManifestMemory.reservationBytes(position: chunk)
        XCTAssertEqual(
            engine.admissionForTesting.bytesReserved, metadataBytes,
            "a live export manifest must retain its host-memory permit after close")
        gate.resume.signal()
        store.finishPublicationCallbacks(engine: engine)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        await engine.shutdown()
    }

    func testInitialReadScratchUsesSlotCeilingAndReleasesExactlyOnce() async throws {
        let (engine, _) = engine(CompleteCheckpointFixtureStore())
        let scratch = CBv2CompleteCheckpointManifest.maximumProviderScratchBytes
        engine.updateKVBytesCapacity(scratch - 1)
        XCTAssertThrowsError(try engine.reserveCompleteCheckpointReadScratch())
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)

        engine.updateKVBytesCapacity(scratch)
        var lease: CBv2CompleteCheckpointIOLease? =
            try engine.reserveCompleteCheckpointReadScratch()
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, scratch)
        XCTAssertEqual(engine.admissionForTesting.transientBytesReserved, scratch)
        XCTAssertThrowsError(try engine.reserveCompleteCheckpointReadScratch())
        lease?.close()
        lease?.close()
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        lease = try engine.reserveCompleteCheckpointReadScratch()
        withExtendedLifetime(lease) {
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, scratch)
        }
        lease = nil
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        await engine.shutdown()
    }

    func testEncodedCheckpointSurvivesEngineRestartAndLeavesZeroIdleCacheArrays() async throws {
        let store = CompleteCheckpointFixtureStore()
        let (first, firstBackend) = engine(store)
        let request = CBv2Request(
            id: .init(7), promptTokens: Array(repeating: 1, count: 2 * chunk + 1), maxTokens: 3,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1007))
        let cold = await cbv2SchedCollect(try first.submit(request))
        XCTAssertEqual(cold.finishReason, .length)
        // Deepest first, then the first boundary: the same order as a
        // historical donor.
        XCTAssertEqual(store.saved.map(\.manifest.position), [2 * chunk, chunk])
        XCTAssertNil(first.hybridPrefixCache)
        XCTAssertEqual(firstBackend.bytesReserved, 0)
        store.finishPublicationCallbacks(engine: first)
        XCTAssertEqual(first.admissionForTesting.bytesReserved, 0)
        await first.shutdown()

        let reopened = CompleteCheckpointFixtureStore(archives: store.saved)
        let (second, secondBackend) = engine(reopened)
        let warmRequest = CBv2Request(
            id: .init(7), promptTokens: request.promptTokens, maxTokens: 3,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(2007))
        XCTAssertTrue(try reopened.stage(engine: second, request: warmRequest))
        let warm = await cbv2SchedCollect(try second.submit(warmRequest))
        XCTAssertEqual(warm.tokens, cold.tokens, "restored recurrent state affects the sampled IDs")
        XCTAssertEqual(warm.usage?.prefixCacheTier, .snapshot)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, 2 * chunk)
        XCTAssertEqual(reopened.saved.count, 2, "inherited-only repeats perform no new writes")
        XCTAssertEqual(reopened.releaseCount, 1)
        XCTAssertEqual(secondBackend.bytesReserved, 0)
        reopened.finishPublicationCallbacks(engine: second)
        XCTAssertEqual(second.admissionForTesting.bytesReserved, 0)
        let foreign = CBv2Request(
            id: .init(8), promptTokens: request.promptTokens, maxTokens: 3,
            cacheSalt: "another-tenant", prefixCacheReceiptID: .init(2008))
        XCTAssertFalse(try reopened.stage(engine: second, request: foreign))
        await second.shutdown()
    }

    func testUnreadableLatestEndpointPreservesDeepestEligibleCheckpoint() async throws {
        let store = CompleteCheckpointFixtureStore(maximumPosition: 2 * chunk)
        let (engine, backend) = engine(store)
        let result = await cbv2SchedCollect(
            try engine.submit(
                .init(
                    id: .init(9), promptTokens: Array(repeating: 1, count: 4 * chunk + 1),
                    maxTokens: 1,
                    prefixCacheReceiptID: .init(1009))))
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [2 * chunk, chunk])
        XCTAssertEqual(backend.bytesReserved, 0)
        store.finishPublicationCallbacks(engine: engine)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        await engine.shutdown()
    }

    func testCancellationAfterCaptureDropsAllPayloadBeforeTerminal() async throws {
        let store = CompleteCheckpointFixtureStore()
        let (engine, backend) = engine(store)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = 2
        }
        let request = CBv2Request(
            id: .init(10), promptTokens: Array(repeating: 1, count: 3 * chunk + 1), maxTokens: 5,
            prefixCacheReceiptID: .init(1010))
        let stream = try engine.submit(request)
        let captured = await cbv2SchedWait {
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.hasCheckpoints(requestID: request.id) == true
            }
        }
        XCTAssertTrue(captured)
        engine.cancel(request.id)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        let result = await cbv2SchedCollect(stream)
        XCTAssertEqual(result.finishReason, .cancelled)
        XCTAssertTrue(store.saved.isEmpty)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(backend.bytesReserved, 0)
        await engine.shutdown()
    }

    func testShutdownDrainsBlockedExportAndLateCancellationPreservesGenerationFence() async throws {
        let gate = CheckpointPublicationGate()
        let store = CompleteCheckpointFixtureStore(gate: gate)
        let (engine, backend) = engine(store)
        let receipts = CompleteCheckpointReceiptLog()
        engine.setCompletePrefixPublicationHandler { receipts.append($0, $1) }
        let request = CBv2Request(
            id: .init(11), promptTokens: Array(repeating: 1, count: chunk + 1), maxTokens: 1,
            prefixCacheReceiptID: .init(1011))
        let stream = try engine.submit(request)
        let collection = Task { await cbv2SchedCollect(stream) }
        let entered = await Task.detached { gate.waitUntilEntered() }.value
        XCTAssertTrue(entered)
        XCTAssertGreaterThan(backend.bytesReserved, 0)
        XCTAssertGreaterThan(engine.admissionForTesting.bytesReserved, 0)
        engine.cancel(request.id)
        XCTAssertThrowsError(
            try engine.submit(
                .init(
                    id: request.id, promptTokens: request.promptTokens, maxTokens: 1,
                    prefixCacheReceiptID: .init(2011))))
        await engine.shutdown()
        let result = await collection.value
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(receipts.count, 0)
        XCTAssertTrue(store.saved.isEmpty)
        XCTAssertEqual(backend.bytesReserved, 0)
        store.finishPublicationCallbacks(engine: engine)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        XCTAssertTrue(engine.loopForTesting.recurrentStates.isEmpty)
        store.close()
    }
}
