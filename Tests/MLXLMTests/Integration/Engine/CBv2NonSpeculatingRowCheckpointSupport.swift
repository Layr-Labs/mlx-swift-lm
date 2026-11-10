import Foundation
import Testing

@testable import MLXLMCommon

/// Request controls that keep a row from speculating. None of them changes
/// the prompt state a prefix checkpoint saves. The fixtures that use these
/// all have a 64-token vocabulary.
enum CBv2NonSpeculatingControl: String, CaseIterable, Sendable, CustomTestStringConvertible {
    case stopString, presencePenalty, frequencyPenalty, repetitionPenalty, logitBias, logprobs
    case tokenConstraint

    var testDescription: String { rawValue }

    func apply(to request: inout CBv2Request) {
        switch self {
        case .stopString: request.stopStrings = ["\u{1}never matches\u{1}"]
        case .presencePenalty: request.sampling.presencePenalty = 1.5
        case .frequencyPenalty: request.sampling.frequencyPenalty = 1.5
        case .repetitionPenalty: request.sampling.repetitionPenalty = 1.3
        case .logitBias: request.sampling.logitBias = [5: -40]
        case .logprobs: request.sampling.topLogprobs = 64
        case .tokenConstraint:
            request.tokenConstraint = CBv2OddTokenConstraint(maxTokens: request.maxTokens)
        }
    }
}

/// Allows only odd token ids.
private final class CBv2OddTokenConstraint: CBv2TokenConstraint, @unchecked Sendable {
    let mode: CBv2TokenConstraintMode = .required
    let maxTokens: Int
    let fallbackTokenID = 1
    let initialState = 0

    init(maxTokens: Int) { self.maxTokens = maxTokens }

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
        remainingTokens > 0 ? Array(stride(from: 1, to: 64, by: 2)) : []
    }

    func nextState(state: Int, tokenID: Int) -> Int? {
        tokenID > 0 && tokenID < 64 && tokenID % 2 == 1 ? 0 : nil
    }
}

struct CBv2ControlledServed {
    var tokens: [Int] = []
    /// Token id and Float32 bits of every reported logprob, in emission order.
    var logprobBits: [UInt32] = []
    var finish: CBv2FinishReason?
    var usage: CBv2Usage?
    var saved: Int { usage?.prefixCachePrefillTokensSaved ?? 0 }
}

func cbv2ServeControlled(_ engine: EngineV2, _ request: CBv2Request) async throws
    -> CBv2ControlledServed
{
    var served = CBv2ControlledServed()
    for await event in try engine.submit(request) {
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
    return served
}

/// A request whose controls keep it from speculating must leave the durable
/// checkpoints an unconstrained request leaves, without changing a token.
/// Serves `control`'s request as the donor of a complete-checkpoint store with
/// capture on and off, compares what it saved with an unconstrained donor's
/// archives, then restores both request shapes from it. `make(nil)` is the
/// engine without a prefix cache: the reference for every token.
func cbv2ExpectNonSpeculatingDonorCheckpoints(
    _ control: CBv2NonSpeculatingControl, chunk: Int, maxTokens: Int, segmentBytes: Int,
    make: (CompleteCheckpointFixtureStore?) throws -> (EngineV2, PagedKVBackend),
    released: (EngineV2, PagedKVBackend) -> Void
) async throws {
    typealias Store = CompleteCheckpointFixtureStore
    let prompt = (0 ..< 2 * chunk + 7).map { 1 + ($0 * 7) % 61 }
    let positions = [2 * chunk, chunk]
    func request(_ id: UInt64, controlled: Bool, cached: Bool = true) -> CBv2Request {
        var request = CBv2Request(
            id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0),
            maxTokens: maxTokens, cacheSalt: "tenant", prefixCacheEnabled: cached,
            prefixCacheReceiptID: cached ? .init(id + 1000) : nil)
        if controlled { control.apply(to: &request) }
        return request
    }
    func archives(_ store: Store) async -> [Store.Archive] {
        _ = await cbv2SchedWait(timeoutSeconds: 3) { store.saved.count >= positions.count }
        return store.saved
    }
    func sameTensors(_ a: Store.Archive, _ b: Store.Archive) -> Bool {
        func parts(_ archive: Store.Archive) -> [Store.Chunk] {
            archive.chunks.sorted { ($0.tensor, $0.offset) < ($1.tensor, $1.offset) }
        }
        let (left, right) = (parts(a), parts(b))
        return left.count == right.count
            && zip(left, right).allSatisfy {
                $0.tensor == $1.tensor && $0.offset == $1.offset && $0.bytes == $1.bytes
            }
    }

    let (cold, coldBackend) = try make(nil)
    let expected = try await cbv2ServeControlled(cold, request(1, controlled: true))
    let unconstrained = try await cbv2ServeControlled(cold, request(2, controlled: false))
    #expect(expected.finish == .length)
    #expect(expected.tokens.count == maxTokens)
    released(cold, coldBackend)
    await cold.shutdown()

    let referenceStore = Store(segmentBytes: segmentBytes)
    let (eligible, eligibleBackend) = try make(referenceStore)
    let eligibleServed = try await cbv2ServeControlled(eligible, request(3, controlled: false))
    let reference = await archives(referenceStore)
    #expect(eligibleServed.tokens == unconstrained.tokens)
    #expect(reference.map(\.manifest.position) == positions)
    released(eligible, eligibleBackend)
    await eligible.shutdown()

    let store = Store(segmentBytes: segmentBytes)
    let (donor, donorBackend) = try make(store)
    let on = try await cbv2ServeControlled(donor, request(4, controlled: true))
    let saved = await archives(store)
    released(donor, donorBackend)
    let off = try await cbv2ServeControlled(donor, request(5, controlled: true, cached: false))
    #expect(on.tokens == expected.tokens, "capture on")
    #expect(off.tokens == expected.tokens, "capture off")
    #expect(on.logprobBits == off.logprobBits, "capture on and off")
    #expect(on.logprobBits == expected.logprobBits, "capture on and no cache")
    #expect(on.finish == expected.finish && off.finish == expected.finish)
    #expect(on.saved == 0 && off.saved == 0)
    #expect(saved.map(\.manifest.position) == positions, "the donor's checkpoints")
    #expect(store.saved.count == saved.count, "capture off saves nothing")
    #expect(saved.count == reference.count)
    for (ours, theirs) in zip(saved, reference) {
        #expect(ours.manifest == theirs.manifest, "position \(ours.manifest.position)")
        #expect(ours.manifest.assistantCodecID != nil)
        #expect(sameTensors(ours, theirs), "position \(ours.manifest.position)")
    }
    released(donor, donorBackend)
    await donor.shutdown()

    let reopened = Store(archives: saved, segmentBytes: segmentBytes)
    let (warm, warmBackend) = try make(reopened)
    for (id, controlled) in [(UInt64(6), true), (UInt64(7), false)] {
        let next = request(id, controlled: controlled)
        let uncached = controlled ? expected : unconstrained
        let staged = try reopened.stage(engine: warm, request: next)
        #expect(staged, "request \(id)")
        let served = try await cbv2ServeControlled(warm, next)
        #expect(served.tokens == uncached.tokens, "request \(id)")
        #expect(served.finish == uncached.finish, "request \(id)")
        #expect(served.saved == 2 * chunk, "request \(id)")
        #expect(served.usage?.prefixCacheReplayTokens == 0, "request \(id)")
        if controlled { #expect(served.logprobBits == expected.logprobBits, "restored") }
        released(warm, warmBackend)
    }
    #expect((warm.mtpMetricsSnapshot()?.draftedTokens ?? 0) > 0)
    await warm.shutdown()
}
