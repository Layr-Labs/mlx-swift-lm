import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

final class NativeCheckpointMemoryTransport: CBv2NativeBlockPrefixCache, @unchecked Sendable {
    let identity = CBv2CompleteCheckpointIdentity(modelAggregateHash: String(repeating: "a", count: 64),
        promptContractID: String(repeating: "b", count: 64), buildID: String(repeating: "c", count: 64),
        numericsFingerprint: String(repeating: "d", count: 64))
    private let lock = NSLock()
    private var serialized: (Data, [[Data]])?
    private var staged: [CBv2RequestID: CBv2NativeBlockCheckpoint] = [:]
    private var pending: [(CBv2CompleteCheckpointExport, @Sendable ([Int]) -> Void)] = []
    private var closed = false
    private var failures = 0
    private var writes = 0
    let deferred: Bool
    let nativeBoundaries: Bool
    init(deferred: Bool = false, nativeBoundaries: Bool = false) {
        self.deferred = deferred
        self.nativeBoundaries = nativeBoundaries
    }
    var counts: (writes: Int, failures: Int, pending: Int) { lock.withLock { (writes, failures, pending.count) } }
    var isClosed: Bool { lock.withLock { closed } }
    func manifest() throws -> CBv2CompleteCheckpointManifest {
        let wire = try #require(lock.withLock { serialized?.0 })
        return try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: wire)
    }
    func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
        position > 0 && (nativeBoundaries || position % 4 == 0)
    }
    func takeNativeStaged(requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?, maximumSequenceLength: Int) -> CBv2NativeBlockCheckpoint? {
        lock.withLock { staged.removeValue(forKey: requestID) }
    }
    func donate(_ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?, tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void) {
        let accepted = lock.withLock {
            guard !closed else { return false }
            if deferred { pending.append((source, completion)) }
            return true
        }
        guard accepted else { source.close(); completion([]); return }
        if deferred { return }
        do {
            let wire = try JSONEncoder().encode(source.manifest)
            let payload = try source.manifest.tensors.enumerated().map { index, tensor in
                var parts = [Data](), offset = 0
                while offset < tensor.byteCount {
                    let bytes = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: 4096)
                    parts.append(bytes); offset += bytes.count
                }
                return parts
            }
            lock.withLock { serialized = (wire, payload); writes += 1 }
            let position = source.manifest.position
            source.close(); completion([position])
        } catch {
            lock.withLock { failures += 1 }
            source.close(); completion([])
        }
    }
    func stage(engine: CBv2NativeBlockEngine, request: CBv2Request) throws {
        let (wire, chunks) = try #require(lock.withLock { serialized })
        let manifest = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: wire)
        let plan = try engine.planNativeCheckpointImport(manifest: manifest, request: request)
        let importer = try plan.allocate()
        defer { importer.close() }
        for (index, parts) in chunks.enumerated() {
            var offset = 0
            for bytes in parts { try importer.appendSegment(tensorIndex: index, byteOffset: offset, data: bytes); offset += bytes.count }
        }
        let value = try importer.finish()
        lock.withLock { staged[request.prefixCacheReceiptID!] = value }
    }
    func close() {
        let retiring = lock.withLock {
            closed = true
            let result = (Array(staged.values), pending)
            staged.removeAll(); pending.removeAll(); serialized = nil
            return result
        }
        for value in retiring.0 { value.close() }
        for (source, completion) in retiring.1 { source.close(); completion([]) }
    }
}

@Suite("Native diffusion engine persistence binding", .serialized)
struct DiffusionGemmaPersistentEngineTests {
    @Test func directEngineShutdownClosesItsStoreWithoutResidentResources() async throws {
        let store = NativeCheckpointMemoryTransport()
        let engine = try CBv2NativeBlockEngine(tokenizer: TestTokenizer(vocabularySize: 128), kvBytesCapacity: 32 << 20,
            completeNativePrefixCache: store,
            checkpointPlanner: { _, _, _ in throw CBv2CompleteCheckpointError.incompatibleCheckpoint },
            reservationForRequest: { _ in 1024 },
            makeSession: { _, _ in throw CBv2NativeBlockError.unsupportedRequest("unused") })
        await engine.shutdown()
        #expect(store.isClosed && engine.capacity().kvBytesReserved == 0)
    }

    private func configuration(_ store: NativeCheckpointMemoryTransport) throws -> DiffusionGemmaResidentPrefixConfiguration {
        try .init(maximumBytes: 1 << 20, artifactIdentity: store.identity.modelAggregateHash,
            templateIdentity: store.identity.promptContractID, numericalProfile: store.identity.numericsFingerprint)
    }
    private func request(_ id: UInt64, tokens: [Int]) -> CBv2Request {
        var result = CBv2Request(id: .init(id), promptTokens: tokens, sampling: .init(seed: 341), maxTokens: 4, cacheSalt: "tenant")
        result.prefixCacheReceiptID = .init(id + 100)
        return result
    }
    private func run(_ engine: CBv2NativeBlockEngine, _ request: CBv2Request) async throws -> ([Int], CBv2Usage) {
        var tokens = [Int](), usage: CBv2Usage?
        for await event in try engine.submit(request) {
            switch event {
            case .delta(_, let ids, _): tokens += ids
            case .finished(let reason, let result):
                #expect(reason == .stop || reason == .length)
                usage = result
            }
        }
        return (tokens, try #require(usage))
    }
    @Test func exactAndAppendAreTokenExactWithMemoryRetentionOff() async throws {
        let (directory, _) = try DiffusionGemmaFactoryTests().fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: DiffusionGemmaFactoryTests.Loader(fail: false))
        let store = NativeCheckpointMemoryTransport()
        let engine = try context.makeNativeEngine(kvBytesCapacity: 128 << 20, prefillChunkSize: 4,
            prefixCache: configuration(store), completePrefixCache: store, retainMemoryPrefixes: false)
        let control = try context.makeNativeEngine(kvBytesCapacity: 128 << 20, prefillChunkSize: 4)
        let tokens = [2, 5, 7, 9, 11, 13, 17, 19]
        let cold = try await run(engine, request(1, tokens: tokens))
        #expect(cold.1.prefixCachePrefillTokensSaved == 0 && store.counts.writes == 1)
        let repeated = request(2, tokens: tokens)
        try store.stage(engine: engine, request: repeated)
        let hit = try await run(engine, repeated)
        #expect(hit.0 == cold.0 && hit.1.prefixCachePrefillTokensSaved == 8)
        #expect(hit.1.prefixCacheTier == .snapshot)
        let appended = request(3, tokens: tokens + [23, 29])
        let reference = try await run(control, appended)
        try store.stage(engine: engine, request: appended)
        let suffix = try await run(engine, appended)
        #expect(suffix.0 == reference.0 && suffix.1.prefixCachePrefillTokensSaved == 8)
        #expect(suffix.1.prefixCacheTier == .snapshot && store.counts.failures == 0)
        // No pre-staged transfer and RAM retention OFF must execute cold.
        let unstaged = try await run(engine, request(4, tokens: tokens))
        #expect(unstaged.0 == cold.0 && unstaged.1.prefixCachePrefillTokensSaved == 0)
        await engine.shutdown(); await control.shutdown()
        #expect(engine.capacity().kvBytesReserved == 0)
    }

    @Test func pendingDonationSurvivesEvictionUnderItsOwnLeaseAndShutdownReleasesWeights() async throws {
        let (directory, _) = try DiffusionGemmaFactoryTests().fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        var context: DiffusionGemmaContext? = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: DiffusionGemmaFactoryTests.Loader(fail: false))
        weak var model = context?.model
        let store = NativeCheckpointMemoryTransport(deferred: true)
        let engine = try context!.makeNativeEngine(kvBytesCapacity: 128 << 20, prefillChunkSize: 4,
            prefixCache: configuration(store), completePrefixCache: store, retainMemoryPrefixes: false)
        _ = try await run(engine, request(1, tokens: [2, 3, 5, 7, 11, 13, 17, 19]))
        #expect(store.counts.pending == 1)
        engine.updateKVBytesCapacity(64 << 10)
        try await Task.sleep(for: .milliseconds(20))
        #expect(engine.capacity().kvBytesReserved > 64 << 10, "Live transfer ownership cannot be refunded by a cache trim")
        context = nil
        await engine.shutdown()
        #expect(store.counts.pending == 0 && engine.capacity().kvBytesReserved == 0)
        #expect(model == nil, "Store/codec/donation handles must not retain model weights")
    }
}
