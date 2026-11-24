import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Inkling 2 rows that cannot speculate (a stop string, a penalty, a logit
/// bias, logprobs, a token constraint) beside rows that decode. The DSpark
/// drafter does not record their prompt history in this release. The real
/// Inkling 2 Mini attention geometry at a tiny hidden size, BF16, synthetic
/// weights, the production sampler.
@Suite("Inkling 2 rows that cannot speculate save checkpoints", .serialized, .tags(.integration))
struct Inkling2DSparkNonSpeculatingRowCheckpointTests {
    typealias Control = CBv2NonSpeculatingControl
    typealias Store = CompleteCheckpointFixtureStore
    typealias Geometry = Inkling2PrefixRealGeometryTests

    static let chunk = 64
    static let segmentBytes = 1 << 20
    /// Inkling's sampled logits cover the unpadded vocabulary only.
    static let vocabulary = 60
    static var dtype: DType { .bfloat16 }

    static func engine(
        model: Inkling2Model, drafter: Inkling2DSparkDrafter?, store: Store?,
        concurrent: Int = 1, stepTokens: Int? = nil,
        sampler: CBv2DefaultSampler = CBv2DefaultSampler()
    ) throws -> (EngineV2, PagedKVBackend) {
        let kinds = model.cbv2LayerKinds
        let types = Array(repeating: dtype, count: kinds.count)
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 4 << 30, dtype: dtype, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: 1024, segmentSizeBytes: 16 << 20,
                layerDTypes: types))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: sampler,
            schedulerConfig: .init(
                maxConcurrentRequests: concurrent, maxBatchedTokensPerStep: stepTokens ?? chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            completeCheckpointKVDTypes: store == nil ? nil : types,
            mtpDrafter: drafter,
            mtpConfig: drafter == nil ? .init() : Inkling2DSparkDrafter.defaultMTPConfig())
        return (engine, paged)
    }

    /// Every charge and every drafter request state is back.
    static func released(_ engine: EngineV2, _ backend: PagedKVBackend) {
        let state = engine.loopForTesting.onEngineQueueSync {
            (
                engine.admissionForTesting.bytesReserved, backend.bytesReserved,
                backend.bytesWired, engine.loopForTesting.recurrentStates.isEmpty,
                engine.loopForTesting.mtp?.requestStateCountForTesting ?? 0
            )
        }
        #expect(state.0 == 0 && state.1 == 0 && state.2 == 0)
        #expect(state.3)
        #expect(state.4 == 0)
    }

    static func request(
        _ id: UInt64, _ prompt: [Int], _ control: Control?, maxTokens: Int = 8,
        cached: Bool = true
    ) -> CBv2Request {
        var request = CBv2Request(
            id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0),
            maxTokens: maxTokens, cacheSalt: "tenant", prefixCacheEnabled: cached,
            prefixCacheReceiptID: cached ? .init(id + 1000) : nil)
        control?.apply(to: &request, vocabulary: vocabulary)
        return request
    }

    static func manifestBytes(_ manifest: CBv2CompleteCheckpointManifest) throws -> Int {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(manifest).count
    }

    static func sameTensors(_ a: Store.Archive, _ b: Store.Archive) -> Bool {
        func parts(_ archive: Store.Archive) -> [Store.Chunk] {
            archive.chunks.sorted { ($0.tensor, $0.offset) < ($1.tensor, $1.offset) }
        }
        let (left, right) = (parts(a), parts(b))
        return left.count == right.count
            && zip(left, right).allSatisfy {
                $0.tensor == $1.tensor && $0.offset == $1.offset && $0.bytes == $1.bytes
            }
    }

    @Test("the DSpark drafter does not opt in for 0.10.19")
    func drafterDoesNotOptIn() throws {
        let model = try Geometry.target(dtype: Self.dtype)
        let drafter = try Geometry.drafter(target: model, dtype: Self.dtype)
        #expect(!drafter.checkpointsNonSpeculatingRows)
    }

    /// Counts steps whose sample call holds a decoding row while the
    /// non-speculating row's prompt advanced in the same step. The scheduler
    /// advances `numComputedTokens` at planning and schedules running decode
    /// rows in every step, so a prompt advance seen at a decode row's sample
    /// call is that step's prompt chunk.
    final class MixedSteps: @unchecked Sendable {
        private let lock = NSLock()
        private var last: Int?
        private var count = 0
        func observe(prompt: Int, promptCount: Int, decoding: Bool) {
            lock.withLock {
                if decoding, let last, prompt > last, prompt < promptCount { count += 1 }
                last = prompt
            }
        }
        var steps: Int { lock.withLock { count } }
    }

    static func absorb(_ event: CBv2Event, into served: inout CBv2ControlledServed) {
        switch event {
        case .delta(_, let tokens, let logprobs):
            served.tokens += tokens
            for entry in logprobs ?? [] {
                served.logprobBits += [UInt32(entry.token), entry.logprob.bitPattern]
                for top in entry.topLogprobs {
                    served.logprobBits += [UInt32(top.token), top.logprob.bitPattern]
                }
            }
        case .finished(let reason, let usage):
            served.finish = reason
            served.usage = usage
        }
    }

    /// A non-speculating row's prompt chunks prefill in the same steps as a
    /// greedy and a seeded sampled row that are already decoding (neither
    /// drafts: a logit bias, logprobs). Every row's token ids, finish and
    /// logprob bits equal the same staggered schedule on the engine without a
    /// prefix cache and on the engine without the drafter. Two chunk rows, one
    /// with logprobs and one with a logit bias. Fails as not exercised unless
    /// the engine shows a step holding both the prompt chunk and decode rows.
    @Test(.timeLimit(.minutes(10)), arguments: [Control.logprobs, .logitBias])
    func promptChunkBesideDecodingRowsIsExact(_ chunkControl: Control) async throws {
        let model = try Geometry.target(dtype: Self.dtype)
        let drafter = try Geometry.drafter(target: model, dtype: Self.dtype)
        let long = (0 ..< 4 * Self.chunk + 7).map { 1 + ($0 * 5) % 59 }
        let ids = (greedy: CBv2RequestID(21), sampled: CBv2RequestID(22), chunk: CBv2RequestID(23))
        func requests() -> [CBv2Request] {
            var greedy = CBv2Request(
                id: ids.greedy, promptTokens: (0 ..< 9).map { 2 + ($0 * 3) % 50 },
                sampling: .init(temperature: 0), maxTokens: 64, cacheSalt: "tenant",
                prefixCacheEnabled: true, prefixCacheReceiptID: .init(1021))
            Control.logitBias.apply(to: &greedy, vocabulary: Self.vocabulary)
            var sampled = CBv2Request(
                id: ids.sampled, promptTokens: (0 ..< 11).map { 3 + ($0 * 7) % 50 },
                sampling: .init(temperature: 0.7, topP: 0.95, topK: 20, seed: 4301),
                maxTokens: 64, cacheSalt: "tenant", prefixCacheEnabled: true,
                prefixCacheReceiptID: .init(1022))
            Control.logprobs.apply(to: &sampled, vocabulary: Self.vocabulary)
            return [greedy, sampled, Self.request(23, long, chunkControl, maxTokens: 8)]
        }
        func serve(_ drafting: Bool, _ store: Store?) async throws -> (
            [CBv2ControlledServed], Int
        ) {
            let sampler = CBv2DefaultSampler()
            let (engine, backend) = try Self.engine(
                model: model, drafter: drafting ? drafter : nil, store: store, concurrent: 3,
                stepTokens: Self.chunk + 8, sampler: sampler)
            let mixed = MixedSteps()
            let loop = engine.loopForTesting
            sampler.sampleObserverForTesting = { rows, _ in
                let decoders = [ids.greedy, ids.sampled].filter { rows.contains($0) }
                guard !decoders.isEmpty, let chunkRow = loop.scheduler.record(for: ids.chunk)
                else { return }
                let decoding = decoders.allSatisfy {
                    (loop.scheduler.record(for: $0)?.generatedTokenCount ?? 0) > 0
                }
                mixed.observe(
                    prompt: chunkRow.numComputedTokens,
                    promptCount: chunkRow.request.promptTokens.count, decoding: decoding)
            }
            let all = requests()
            var greedyIt = try engine.submit(all[0]).makeAsyncIterator()
            var sampledIt = try engine.submit(all[1]).makeAsyncIterator()
            var greedy = CBv2ControlledServed()
            var sampled = CBv2ControlledServed()
            while greedy.tokens.count < 2, let event = await greedyIt.next() {
                Self.absorb(event, into: &greedy)
            }
            while sampled.tokens.count < 2, let event = await sampledIt.next() {
                Self.absorb(event, into: &sampled)
            }
            let chunkStream = try engine.submit(all[2])
            async let chunked = cbv2CollectControlled(chunkStream)
            while let event = await greedyIt.next() { Self.absorb(event, into: &greedy) }
            while let event = await sampledIt.next() { Self.absorb(event, into: &sampled) }
            let served = [greedy, sampled, await chunked]
            store?.finishPublicationCallbacks(engine: engine)
            Self.released(engine, backend)
            await engine.shutdown()
            return (served, mixed.steps)
        }
        let (on, onMixed) = try await serve(true, Store(segmentBytes: Self.segmentBytes))
        let (noCache, _) = try await serve(true, nil)
        let (off, _) = try await serve(false, nil)
        print("INKLING2_MIXED_STEP \(chunkControl) mixed steps=\(onMixed)")
        #expect(onMixed > 0, "not exercised: no step held the prompt chunk and decode rows")
        for (row, name) in ["greedy", "sampled", "chunk"].enumerated() {
            #expect(on[row].tokens.count > 0, "\(name)")
            for (other, label) in [(noCache, "no prefix cache"), (off, "drafter off")] {
                #expect(on[row].tokens == other[row].tokens, "\(name) vs \(label)")
                #expect(on[row].logprobBits == other[row].logprobBits, "\(name) vs \(label)")
                #expect(on[row].finish == other[row].finish, "\(name) vs \(label)")
            }
        }
    }
}
