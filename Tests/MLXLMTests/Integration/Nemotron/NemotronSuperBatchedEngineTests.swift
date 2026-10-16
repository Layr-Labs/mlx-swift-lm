import Foundation
import MLX
import Testing

@testable import MLXLLM
@_spi(Benchmarking) @testable import MLXLMCommon
@testable import MLXVLM

/// Real EngineV2 serving of concurrent Super text requests through the
/// rectangular adapter: decode rows share one `[B, 1]` forward and prompts
/// stay serial. With row-exact products greedy output equals each request
/// served alone; a cancelled request emits a prefix of its own output and a
/// request submitted mid-stream joins the running batch. Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperBatchedEngineTests {
    typealias Fixture = NemotronSuperBatchFixture
    typealias Collected = NemotronSuperMTPEngineTests.Collected

    static func engine(
        _ model: NemotronSuperModel, rows: Int,
        products: NemotronSuperRecurrentAdapter.DecodeProducts = .rowExact
    ) throws -> EngineV2 {
        let adapter = try NemotronSuperRecurrentAdapter(
            model, maximumDecodeRows: rows, decodeProducts: products)
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(adapter), layerKinds: adapter.cbv2LayerKinds,
            backend: CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 64 << 20, kvDType: .float32)),
            cacheProvider: CBv2LayerCacheBank(
                caches: adapter.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: rows, maxBatchedTokensPerStep: 32,
                prefillChunkSize: 8, maxWaiting: 8),
            admissionConfig: .init(watermarkFraction: 0))
    }

    static let requests: [(prompt: [Int], maxTokens: Int)] = [
        (Fixture.tokens(5, seed: 1).map(Int.init), 12),
        (Fixture.tokens(13, seed: 2).map(Int.init), 20),
        (Fixture.tokens(3, seed: 3).map(Int.init), 16),
        (Fixture.tokens(9, seed: 4).map(Int.init), 10),
        (Fixture.tokens(7, seed: 5).map(Int.init), 30),
    ]
    /// Submitted once request 0 has emitted three tokens.
    static let lateRequest = 3
    /// Cancelled once it has emitted two tokens.
    static let cancelledRequest = 4

    static func request(_ index: Int, base: UInt64) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(base + UInt64(index)), promptTokens: requests[index].prompt,
            sampling: .init(temperature: 0), maxTokens: requests[index].maxTokens)
    }

    static func collect(
        _ stream: AsyncStream<CBv2Event>, onTokens: (Int) throws -> Void
    ) async throws -> Collected {
        var result = Collected()
        for await event in stream {
            switch event {
            case .delta(_, let tokens, _):
                result.tokens += tokens
                try onTokens(result.tokens.count)
            case .finished(let reason, _): result.finish = reason
            }
        }
        return result
    }

    static func solo(_ model: NemotronSuperModel) async throws -> [Collected] {
        let engine = try engine(model, rows: 1)
        var results: [Collected] = []
        for index in requests.indices {
            results.append(
                try await collect(try engine.submit(request(index, base: 100))) { _ in })
        }
        await engine.shutdown()
        return results
    }

    /// The late request's stream, started at most once from any collector.
    final class LateJoin: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<Collected, Error>?

        func start(_ submit: @escaping @Sendable () throws -> AsyncStream<CBv2Event>) {
            lock.withLock {
                guard task == nil else { return }
                task = Task {
                    try await NemotronSuperBatchedEngineTests.collect(try submit()) { _ in }
                }
            }
        }

        func value() async throws -> Collected? { try await lock.withLock { task }?.value }
    }

    static func concurrent(
        _ model: NemotronSuperModel, products: NemotronSuperRecurrentAdapter.DecodeProducts
    ) async throws -> ([Collected], CBv2ForwardShapeSnapshot) {
        let engine = try engine(model, rows: 4, products: products)
        _ = try engine.beginForwardShapeObservation()
        let late = LateJoin()
        let initial = requests.indices.filter { $0 != lateRequest }
        let streams = try initial.map { try engine.submit(request($0, base: 200)) }
        var results: [Int: Collected] = [:]
        try await withThrowingTaskGroup(of: (Int, Collected).self) { group in
            for (stream, index) in zip(streams, initial) {
                group.addTask {
                    var cancelSent = false
                    let collected = try await Self.collect(stream) { count in
                        if index == cancelledRequest, count >= 2, !cancelSent {
                            cancelSent = true
                            engine.cancel(CBv2RequestID(200 + UInt64(index)))
                        }
                        if index == 0, count >= 3 {
                            late.start { try engine.submit(request(lateRequest, base: 200)) }
                        }
                    }
                    return (index, collected)
                }
            }
            for try await (index, collected) in group { results[index] = collected }
        }
        results[lateRequest] = try await late.value()
        let snapshot = engine.forwardShapeSnapshot()
        await engine.shutdown()
        return (requests.indices.map { results[$0] ?? Collected() }, snapshot)
    }

    static func widestDecode(_ snapshot: CBv2ForwardShapeSnapshot) -> Int {
        snapshot.entries.filter { $0.axes.phase == .decode || $0.axes.phase == .mixedFrontier }
            .map(\.axes.liveBatchRows).max() ?? 0
    }

    @Test(.timeLimit(.minutes(5)))
    func rowExactConcurrentServingMatchesSoloServing() async throws {
        let model = try Fixture.model(seed: 17, dtype: .bfloat16, products: .rowwise)
        let solo = try await Self.solo(model)
        let (batched, shapes) = try await Self.concurrent(model, products: .rowExact)
        for index in Self.requests.indices where index != Self.cancelledRequest {
            #expect(batched[index].tokens == solo[index].tokens, "request \(index)")
            #expect(batched[index].finish == .length)
        }
        let cancelled = batched[Self.cancelledRequest]
        #expect(cancelled.finish == .cancelled)
        #expect(cancelled.tokens.count >= 2)
        #expect(cancelled.tokens.count < solo[Self.cancelledRequest].tokens.count)
        #expect(
            Array(solo[Self.cancelledRequest].tokens.prefix(cancelled.tokens.count))
                == cancelled.tokens)
        let widest = Self.widestDecode(shapes)
        print("BATCH engine widest decode rows \(widest)")
        #expect(widest >= 3)
    }

    /// Shared products: greedy agreement with solo serving is recorded as
    /// the common prefix length of each request.
    @Test(.timeLimit(.minutes(5)), arguments: [DType.float32, .bfloat16])
    func sharedProductServingAgreementIsRecorded(dtype: DType) async throws {
        let model = try Fixture.model(seed: 17, dtype: dtype, products: .tiled)
        let solo = try await Self.solo(model)
        let (batched, shapes) = try await Self.concurrent(model, products: .shared)
        var agreement: [String] = []
        for index in Self.requests.indices where index != Self.cancelledRequest {
            let shared = zip(batched[index].tokens, solo[index].tokens).prefix { $0 == $1 }.count
            agreement.append("\(shared)/\(solo[index].tokens.count)")
            #expect(batched[index].tokens.count == solo[index].tokens.count)
        }
        print("BATCH engine shared \(dtype) greedy common prefix per request: \(agreement)")
        #expect(batched[Self.cancelledRequest].finish == .cancelled)
        #expect(Self.widestDecode(shapes) >= 3)
    }
}
