import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Super requests whose logits turn non-finite fail closed. One token's
/// embedding row is spiked beyond layer 0's tight calibration, so under the
/// eager overflow rule every forward that consumes that token gives NaN
/// logits. The faulty request emits its solo tokens up to the trigger, then
/// finishes with the generation-failure terminal; a healthy request served
/// beside it keeps its solo tokens; every reservation returns. Plain row-exact
/// batched decode and the speculating owner's seed decode and verify window
/// are covered. Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperNonFiniteLogitsTests {
    typealias Overflow = NemotronSuperActivationOverflowTests
    typealias Batched = NemotronSuperBatchedEngineTests
    typealias Speculative = NemotronSuperMTPBatchedEngineTests
    typealias CoScheduled = NemotronSuperCoScheduledServingTests
    typealias Media = NemotronSuperMTPMediaFallbackTests
    typealias Collected = NemotronSuperMTPEngineTests.Collected

    /// The faulty request's output at `position` is the trigger; the healthy
    /// request never consumes it, and neither does the faulty one earlier.
    struct Plan {
        let faulty: Int
        let healthy: Int
        let position: Int
        let trigger: Int
        let solo: [Collected]
        var faultyPrefix: [Int] { Array(solo[faulty].tokens[...position]) }
    }

    static func plan(firstToken: Bool) async throws -> Plan? {
        let reference = try Overflow.stack(.eagerNaN)
        let solo = try await Speculative.solo(reference.model, kvDType: .bfloat16)
        let requests = Speculative.requests
        for faulty in requests.indices {
            let tokens = solo[faulty].tokens
            // A verify trigger leaves rounds before it and after it.
            let positions = firstToken ? [0] : Array(stride(from: 3, to: tokens.count - 6, by: 1))
            for position in positions {
                let trigger = tokens[position]
                guard trigger != 18, !requests[faulty].prompt.contains(trigger),
                    !tokens[..<position].contains(trigger)
                else { continue }
                for healthy in requests.indices where healthy != faulty {
                    if !requests[healthy].prompt.contains(trigger),
                        !solo[healthy].tokens.contains(trigger)
                    {
                        return Plan(
                            faulty: faulty, healthy: healthy, position: position,
                            trigger: trigger, solo: solo)
                    }
                }
            }
        }
        return nil
    }

    static func serve(_ engine: EngineV2, _ plan: Plan) async throws
        -> (faulty: Collected, healthy: Collected, sampled: Int, verify: Int)
    {
        let faulty = try engine.submit(Speculative.request(plan.faulty, base: 500))
        let healthy = try engine.submit(Speculative.request(plan.healthy, base: 500))
        async let faultyResult = Batched.collect(faulty) { _ in }
        async let healthyResult = Batched.collect(healthy) { _ in }
        let (failed, kept) = try await (faultyResult, healthyResult)
        let loop = engine.loopForTesting
        let counts = loop.onEngineQueueSync {
            (loop.nonFiniteSampledFailures, loop.nonFiniteVerifyFailures)
        }
        #expect(failed.finish == CBv2NonFiniteLogits.finishReason)
        #expect(kept.tokens == plan.solo[plan.healthy].tokens, "healthy request \(plan.healthy)")
        #expect(kept.finish == .length)
        return (failed, kept, counts.0, counts.1)
    }

    @Test(.timeLimit(.minutes(3)))
    func plainBatchedDecodeFailsTheFaultyRowOnly() async throws {
        let plan = try #require(await Self.plan(firstToken: false))
        let stack = try Overflow.stack(.eagerNaN, spikedToken: plan.trigger)
        let adapter = try NemotronSuperRecurrentAdapter(
            stack.model, maximumDecodeRows: CoScheduled.rows, decodeProducts: .rowExact)
        let (engine, backend) = CoScheduled.engine(
            adapter, kinds: adapter.cbv2LayerKinds,
            caches: adapter.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) },
            kvDType: .bfloat16, config: CoScheduled.policy(depth: 0, decodeSteps: 0),
            drafter: nil, depth: 0)
        let result = try await Self.serve(engine, plan)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(result.faulty.tokens == plan.faultyPrefix)
        #expect(result.sampled == 1 && result.verify == 0)
    }

    /// The trigger is the first token: the speculating owner's seed decode
    /// consumes it and fails before any verify round.
    @Test(.timeLimit(.minutes(3)))
    func speculatingSeedDecodeFailsTheFaultyRowOnly() async throws {
        let plan = try #require(await Self.plan(firstToken: true))
        let stack = try Overflow.stack(.eagerNaN, spikedToken: plan.trigger)
        let (engine, backend) = Media.speculativeEngine(stack, kvDType: .bfloat16)
        let result = try await Self.serve(engine, plan)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(result.faulty.tokens == [plan.trigger])
        #expect(result.sampled == 1 && result.verify == 0)
    }

    /// The trigger is committed mid-stream, so a verify window consumes it:
    /// either as the next round's carry or as an accepted draft whose
    /// following position is still committed. The row commits nothing from
    /// that round, never a token decided by the NaN logits.
    @Test(.timeLimit(.minutes(3)))
    func speculatingVerifyWindowFailsTheFaultyRowOnly() async throws {
        let plan = try #require(await Self.plan(firstToken: false))
        let stack = try Overflow.stack(.eagerNaN, spikedToken: plan.trigger)
        let (engine, backend) = Media.speculativeEngine(stack, kvDType: .bfloat16)
        let result = try await Self.serve(engine, plan)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(plan.faultyPrefix.starts(with: result.faulty.tokens))
        #expect(result.faulty.tokens.count > 1)
        #expect(result.sampled == 0 && result.verify == 1)
        #expect(metrics.rectangularVerificationRounds > 0)
    }
}
