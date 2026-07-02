// Copyright © 2026 Eigen Labs.
//
// WS-B: engine-loop tests with a scripted fake model (NO model weights).
// The scripted model emits next-token = (input + 1) % vocab per row, so
// expected outputs are exact and batch-composition invariant.
//
// Spec coverage: chained decode emits correct tokens one step late; stop
// token honored with ≤1 wasted step (KV tail rolled back); backpressure
// pauses exactly the slow stream; watchdog fires; deadlines error-finish;
// preemption end-to-end; cancel; shutdown drain.

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBv2SchedulerLoopTests: XCTestCase {

    /// PR#62 review (paged admission alignment): when several same-step
    /// admissions race for the last capacity, the loser of the backend's
    /// atomic charge must be REQUEUED to waiting — an accepted request waits
    /// for room; it never error-finishes on capacity.
    func testCapacityLoserIsRequeuedNotErrored() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 8))
        harness.backend.maxLiveStates = 1
        async let a = cbv2SchedCollect(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 6), maxTokens: 4)))
        async let b = cbv2SchedCollect(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(6 ..< 12), maxTokens: 4)))
        let (ra, rb) = try await (a, b)
        await harness.engine.shutdown()
        XCTAssertEqual(ra.finishReason, .length, "first request must complete")
        XCTAssertEqual(
            rb.finishReason, .length,
            "capacity loser must complete after room frees, not error")
        XCTAssertGreaterThanOrEqual(
            harness.engine.loopForTesting.capacityRequeueCount, 1,
            "the loser must have gone through requeue-on-capacity")
    }


    /// Expected scripted-model generation: prompt's last token + 1, +2, ...
    private func expectedTokens(prompt: [Int], count: Int, vocab: Int = 64) -> [Int] {
        var current = prompt.last!
        return (0 ..< count).map { _ in
            current = (current + 1) % vocab
            return current
        }
    }

    // MARK: Basic generation

    func testSingleRequestGeneratesExactSequence() async throws {
        let harness = CBv2SchedHarness()
        let request = CBv2SchedFixtures.request(prompt: [3], maxTokens: 5)
        let events = try harness.engine.submit(request)
        let collected = await cbv2SchedCollect(events)

        XCTAssertEqual(collected.tokens, [4, 5, 6, 7, 8])
        XCTAssertEqual(collected.text, "<4><5><6><7><8>")
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(collected.usage?.promptTokens, 1)
        XCTAssertEqual(collected.usage?.completionTokens, 5)

        // KV cleaned up after finish.
        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released)
    }

    func testMultiChunkPrefillThenDecode() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64, prefillChunkSize: 16,
                maxWaiting: 8))
        let prompt = Array(0 ..< 40)  // 3 chunks: 16 + 16 + 8
        let request = CBv2SchedFixtures.request(prompt: prompt, maxTokens: 3)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        XCTAssertEqual(collected.tokens, [40, 41, 42])
        XCTAssertEqual(collected.finishReason, .length)
        // Prefill ran as [1, chunk] forwards: 16, 16, then the sampling 8.
        let shapes = harness.model.forwardShapes
        XCTAssertTrue(shapes.contains([1, 16]), "chunked prefill must run [1, chunk]: \(shapes)")
        XCTAssertTrue(shapes.contains([1, 8]), "final partial chunk: \(shapes)")
    }

    // MARK: Chained decode

    func testChainedDecodeProducesCorrectTokensOneStepLate() async throws {
        let harness = CBv2SchedHarness()
        let request = CBv2SchedFixtures.request(prompt: [10], maxTokens: 12)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        XCTAssertEqual(collected.tokens, expectedTokens(prompt: [10], count: 12))
        XCTAssertEqual(collected.finishReason, .length)
        // A single decoding request must chain nearly every decode step
        // (tokens are inspected one step late while N+1 runs).
        XCTAssertGreaterThanOrEqual(
            harness.engine.chainedStepCount, 8,
            "chained overlap must engage for steady decode")
    }

    func testStopTokenHonoredWithAtMostOneWastedStepAndKVTailRollback() async throws {
        let harness = CBv2SchedHarness()
        // Scripted tokens: 4, 5, 6, ... — stop at 6.
        let request = CBv2SchedFixtures.request(prompt: [3], maxTokens: 100, stopTokens: [6])
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        XCTAssertEqual(collected.tokens, [4, 5, 6], "nothing past the stop token")
        XCTAssertEqual(collected.finishReason, .stop)
        XCTAssertEqual(collected.usage?.completionTokens, 3)

        // The chained step launched before the stop was inspected computed
        // one extra token; its KV tail must have been rolled back exactly 1.
        let released = await cbv2SchedWait { harness.backend.releasedStates.count == 1 }
        XCTAssertTrue(released)
        let state = harness.backend.releasedStates[0]
        let sequence = state[0] as! CBv2SchedMockSequenceKV
        XCTAssertEqual(
            sequence.rollbackCalls, [1],
            "wasted chained-step KV write must be rolled back exactly once")
    }

    func testStopStringViaDetokenizerHoldback() async throws {
        // The scripted detokenizer flags a stop-string match when it sees
        // token 20 (stands in for WS-E's StopHoldback).
        let harness = CBv2SchedHarness(stopTrigger: 20)
        let request = CBv2SchedFixtures.request(
            prompt: [17], maxTokens: 100, stopStrings: ["<stop>"])
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        XCTAssertEqual(collected.tokens, [18, 19, 20])
        XCTAssertEqual(collected.finishReason, .stop)
    }

    // MARK: Batch composition

    func testBatchmatesDoNotChangeEachOthersOutput() async throws {
        // Solo run.
        let solo = CBv2SchedHarness()
        let soloOut = await cbv2SchedCollect(
            try solo.engine.submit(CBv2SchedFixtures.request(prompt: [3], maxTokens: 6)))

        // Same request logic with two batchmates of different lengths.
        let batched = CBv2SchedHarness()
        let a = CBv2SchedFixtures.request(prompt: [3], maxTokens: 6)
        let b = CBv2SchedFixtures.request(prompt: [30, 31], maxTokens: 9)
        let c = CBv2SchedFixtures.request(prompt: [50], maxTokens: 2)
        let streamA = try batched.engine.submit(a)
        let streamB = try batched.engine.submit(b)
        let streamC = try batched.engine.submit(c)
        async let outA = cbv2SchedCollect(streamA)
        async let outB = cbv2SchedCollect(streamB)
        async let outC = cbv2SchedCollect(streamC)
        let (collectedA, collectedB, collectedC) = await (outA, outB, outC)

        XCTAssertEqual(collectedA.tokens, soloOut.tokens, "batch-composition invariance")
        XCTAssertEqual(collectedA.tokens, [4, 5, 6, 7, 8, 9])
        XCTAssertEqual(collectedB.tokens, expectedTokens(prompt: [30, 31], count: 9))
        XCTAssertEqual(collectedC.tokens, [51, 52])
        XCTAssertEqual(collectedA.finishReason, .length)
        XCTAssertEqual(collectedB.finishReason, .length)
        XCTAssertEqual(collectedC.finishReason, .length)
    }

    // MARK: Cancellation

    func testCancelRunningRequestFinishesPromptlyAndFreesKV() async throws {
        let harness = CBv2SchedHarness()
        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 1_000_000)
        let stream = try harness.engine.submit(request)

        let collector = Task { await cbv2SchedCollect(stream) }
        // Let it decode a bit, then cancel.
        _ = await cbv2SchedWait { harness.engine.stepCount > 5 }
        harness.engine.cancel(request.id)
        let collected = await collector.value

        XCTAssertEqual(collected.finishReason, .cancelled)
        // Output up to the cancel is the correct prefix.
        XCTAssertEqual(
            collected.tokens, expectedTokens(prompt: [1], count: collected.tokens.count))
        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released, "cancelled request's KV must be freed")
    }

    func testCancelImmediatelyAfterSubmitIsHonored() async throws {
        // Race: a cancel arrives after `submit` registered the stream but
        // BEFORE the engine's enqueue block runs (the scheduler has no record
        // yet). Artificially delay enqueue to make the window deterministic;
        // the request must never start (PR#62 review). Without the fix the
        // cancel is dropped and the request generates its full budget.
        let harness = CBv2SchedHarness()
        harness.engine.loopForTesting.enqueueStartDelayForTesting = 0.2

        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 50)
        let stream = try harness.engine.submit(request)
        // Cancel while the enqueue block is still sleeping (pre-scheduler).
        harness.engine.cancel(request.id)

        let collected = await cbv2SchedCollect(stream)
        XCTAssertEqual(
            collected.finishReason, .cancelled,
            "an early cancel must abort the request before it starts")
        XCTAssertTrue(
            collected.tokens.isEmpty, "a request cancelled before start must emit no tokens")

        // Nothing was ever admitted / allocated.
        XCTAssertEqual(harness.backend.makeCalls, 0, "KV must never be allocated")
        let idle = await cbv2SchedWait { harness.engine.capacity().activeRequests == 0 }
        XCTAssertTrue(idle)
    }

    // MARK: Backpressure

    func testBackpressurePausesExactlyTheSlowStream() async throws {
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(eventBufferCapacity: 4))
        let slow = CBv2SchedFixtures.request(prompt: [1], maxTokens: 30)
        let fast = CBv2SchedFixtures.request(prompt: [20], maxTokens: 30)
        let slowStream = try harness.engine.submit(slow)
        let fastStream = try harness.engine.submit(fast)

        // Consume fast eagerly; leave slow unread.
        let fastOut = await cbv2SchedCollect(fastStream)
        XCTAssertEqual(fastOut.tokens, expectedTokens(prompt: [20], count: 30))
        XCTAssertEqual(fastOut.finishReason, .length)

        // Exactly the slow request is paused, slot retained.
        let paused = await cbv2SchedWait {
            harness.engine.loopForTesting.pausedIDsSnapshot() == [slow.id]
        }
        XCTAssertTrue(paused, "slow stream must be paused while fast one finished")
        XCTAssertEqual(harness.engine.capacity().activeRequests, 1)

        // Draining the slow stream resumes it; the full sequence arrives in
        // order with nothing lost.
        let slowOut = await cbv2SchedCollect(slowStream)
        XCTAssertEqual(slowOut.tokens, expectedTokens(prompt: [1], count: 30))
        XCTAssertEqual(slowOut.finishReason, .length)
    }

    // MARK: Preemption end-to-end

    func testPreemptionRequeuesVictimWhichStillCompletesCorrectly() async throws {
        // 16 B/token (1 layer, kv1, hd4, fp16). Capacity 200 B ⇒ 12 tokens
        // of soft ledger. Two requests of prompt 4 + 6 generated = 10 tokens
        // each can never coexist ⇒ the younger is preempted mid-decode and
        // must resume (re-prefill with KEPT generated tokens) and finish
        // with a seamless token stream.
        let harness = CBv2SchedHarness(backendCapacity: 200)
        let older = CBv2SchedFixtures.request(prompt: [1, 2, 3, 4], maxTokens: 6)
        let younger = CBv2SchedFixtures.request(prompt: [40, 41, 42, 43], maxTokens: 6)
        let olderStream = try harness.engine.submit(older)
        let youngerStream = try harness.engine.submit(younger)
        async let olderOut = cbv2SchedCollect(olderStream)
        async let youngerOut = cbv2SchedCollect(youngerStream)
        let (olderCollected, youngerCollected) = await (olderOut, youngerOut)

        XCTAssertEqual(olderCollected.tokens, [5, 6, 7, 8, 9, 10])
        XCTAssertEqual(olderCollected.finishReason, .length)
        XCTAssertEqual(
            youngerCollected.tokens, [44, 45, 46, 47, 48, 49],
            "preempted request must keep generated tokens and continue seamlessly")
        XCTAssertEqual(youngerCollected.finishReason, .length)
        XCTAssertGreaterThan(harness.engine.preemptionCount, 0, "capacity forces preemption")
    }

    /// Regression: `prefixHitTokens` was not cleared when an ADOPTED request
    /// was preempted. Preemption discards the adopted KV and recomputes the
    /// full prompt from scratch, so a stale entry over-credited
    /// usage.prefixCacheHitTokens at finish. Both preemption paths now drop
    /// the entry; a non-preempted adopted batchmate keeps its true credit.
    func testPreemptedAdoptedRequestReportsZeroPrefixHitTokens() async throws {
        // Same capacity shape as the preemption test above (12-token soft
        // ledger), plus a scripted prefix cache that claims a 2-token hit
        // for every prompt. Victim = youngest ⇒ the second request is
        // preempted mid-decode and recomputes everything.
        let matched = 2
        let prefixCache = CBv2SchedScriptedPrefixCache(matched: matched)
        let harness = CBv2SchedHarness(
            backendCapacity: 200,
            schedulerConfig: CBv2SchedulerConfig(enablePrefixCache: true),
            prefixCache: prefixCache)
        let older = CBv2SchedFixtures.request(prompt: [1, 2, 3, 4], maxTokens: 6)
        let younger = CBv2SchedFixtures.request(prompt: [40, 41, 42, 43], maxTokens: 6)
        let olderStream = try harness.engine.submit(older)
        let youngerStream = try harness.engine.submit(younger)
        async let olderOut = cbv2SchedCollect(olderStream)
        async let youngerOut = cbv2SchedCollect(youngerStream)
        let (olderCollected, youngerCollected) = await (olderOut, youngerOut)

        XCTAssertGreaterThan(harness.engine.preemptionCount, 0, "capacity forces preemption")
        // Token streams stay seamless either way.
        XCTAssertEqual(olderCollected.tokens, [5, 6, 7, 8, 9, 10])
        XCTAssertEqual(youngerCollected.tokens, [44, 45, 46, 47, 48, 49])

        // Control: the non-preempted adopted request keeps its true credit.
        XCTAssertEqual(
            olderCollected.usage?.prefixCacheHitTokens, matched,
            "adopted, never preempted: usage reports the skipped tokens")
        // Regression: preemption + full recompute means NOTHING was skipped.
        XCTAssertEqual(
            youngerCollected.usage?.prefixCacheHitTokens, 0,
            "preempted-adopted request recomputed everything; stale credit must be dropped")

        // Every lookup pin was balanced by exactly one endAdoption.
        XCTAssertEqual(prefixCache.lookups, 2)
        XCTAssertEqual(prefixCache.endAdoptions, prefixCache.lookups)
    }

    // MARK: Deadlines & watchdog

    func testRequestDeadlineErrorFinishes() async throws {
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(requestTimeout: 0.05))
        harness.model.forwardDelay = 0.02  // a few slow steps blow the deadline
        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 1_000_000)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        guard case .error(let message)? = collected.finishReason else {
            return XCTFail("expected deadline error, got \(String(describing: collected.finishReason))")
        }
        XCTAssertTrue(message.contains("deadline"), message)
        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released)
    }

    func testWatchdogFiresOnWedgedStepAndErrorsStreams() async throws {
        final class WedgeFlag: @unchecked Sendable {
            private let lock = NSLock()
            private var _count = 0
            func fire() {
                lock.lock()
                _count += 1
                lock.unlock()
            }
            var count: Int {
                lock.lock()
                defer { lock.unlock() }
                return _count
            }
        }
        let flag = WedgeFlag()
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(
                stepTimeout: 0.05, watchdogInterval: 0.01))
        harness.engine.onStepWedge = { _ in flag.fire() }
        harness.model.forwardDelay = 0.5  // wedge the first step

        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 100)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))

        guard case .error(let message)? = collected.finishReason else {
            return XCTFail("expected watchdog error, got \(String(describing: collected.finishReason))")
        }
        XCTAssertTrue(message.contains("watchdog"), message)
        XCTAssertGreaterThanOrEqual(flag.count, 1, "wedge signal must fire")
        // The loop resumes after the wedge and cleans up the row.
        let cleaned = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(cleaned)
        XCTAssertTrue(harness.engine.isHealthy, "health recovers once the step completes")
    }

    // MARK: Degenerate submissions

    func testDegenerateSubmissionsFinishImmediately() async throws {
        let harness = CBv2SchedHarness()
        let zeroBudget = await cbv2SchedCollect(
            try harness.engine.submit(CBv2SchedFixtures.request(prompt: [1], maxTokens: 0)))
        XCTAssertEqual(zeroBudget.finishReason, .length)
        XCTAssertTrue(zeroBudget.tokens.isEmpty)

        let empty = await cbv2SchedCollect(
            try harness.engine.submit(CBv2SchedFixtures.request(prompt: [], maxTokens: 5)))
        guard case .error? = empty.finishReason else {
            return XCTFail("empty prompt must error-finish")
        }
    }

    func testOversizedRequestThrowsCapacityExhausted() throws {
        // 16 B/token, capacity 160 B ⇒ 10 tokens worst case.
        let harness = CBv2SchedHarness(backendCapacity: 160)
        XCTAssertThrowsError(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 8), maxTokens: 8))
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
        }
    }

    /// Regression (livelock): a request in `(capacity - watermark, capacity]`
    /// used to pass `canEverFit` (full capacity) but could never complete a
    /// reservation (`reserve` enforces capacity - watermark) — it hit the
    /// wall, self-preempted, restarted, and looped until its deadline. It
    /// must be rejected at submit with capacityExhausted instead.
    func testRequestInsideWatermarkBandRejectedAtSubmit() throws {
        // 16 B/token, capacity 1600 B, watermark 5 % (80 B) ⇒ 95 tokens is
        // the true ceiling; 96..100 tokens sit in the livelock band.
        let harness = CBv2SchedHarness(
            backendCapacity: 1600,
            admissionConfig: .init(watermarkFraction: 0.05))
        XCTAssertThrowsError(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 48), maxTokens: 50))
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
        }
        // Just under the watermark-adjusted ceiling still admits.
        XCTAssertNoThrow(
            try harness.engine.submit(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 45), maxTokens: 50)))
    }

    // MARK: Shutdown

    func testShutdownDrainsRunningCancelsWaitingAndRejectsNew() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2048, prefillChunkSize: 512,
                maxWaiting: 8))
        let running = CBv2SchedFixtures.request(prompt: [1], maxTokens: 40)
        let queued = CBv2SchedFixtures.request(prompt: [2], maxTokens: 5)
        let runningStream = try harness.engine.submit(running)
        // Ensure `running` occupies the single slot before `queued` arrives.
        _ = await cbv2SchedWait { harness.engine.capacity().activeRequests == 1 }
        let queuedStream = try harness.engine.submit(queued)
        _ = await cbv2SchedWait { harness.engine.capacity().waitingRequests == 1 }

        async let runningOut = cbv2SchedCollect(runningStream)
        async let queuedOut = cbv2SchedCollect(queuedStream)
        await harness.engine.shutdown()
        let (runningCollected, queuedCollected) = await (runningOut, queuedOut)

        XCTAssertEqual(
            runningCollected.tokens, expectedTokens(prompt: [1], count: 40),
            "running request finishes naturally during drain")
        XCTAssertEqual(runningCollected.finishReason, .length)
        XCTAssertEqual(queuedCollected.finishReason, .cancelled, "waiting request cancelled")

        XCTAssertThrowsError(
            try harness.engine.submit(CBv2SchedFixtures.request(prompt: [9], maxTokens: 1))
        ) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted after shutdown, got \(error)")
            }
        }
    }

    /// Regression: `shutdown()` waits for the drain on the ENGINE queue, so
    /// a wedged queue (a step blocked inside eval) hung it forever. It is
    /// now bounded by `shutdownTimeout`: live streams are force-finished
    /// with `.error` and shutdown returns while the wedged step is still
    /// blocked.
    func testShutdownTimesOutWhenEngineQueueIsWedged() async throws {
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(
                requestTimeout: 60,
                stepTimeout: 60,  // keep the step watchdog out of the way
                shutdownTimeout: 0.3))
        harness.model.forwardDelay = 3.0  // wedge the engine queue

        let request = CBv2SchedFixtures.request(prompt: [1], maxTokens: 100)
        let stream = try harness.engine.submit(request)
        async let collectedOut = cbv2SchedCollect(stream)

        // Give the engine a moment to enter the wedged step.
        try await Task.sleep(nanoseconds: 200_000_000)

        let started = Date()
        await harness.engine.shutdown()
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertLessThan(
            elapsed, 2.5,
            "shutdown must return at the timeout, not wait out the wedged step")

        let collected = await collectedOut
        guard case .error(let message)? = collected.finishReason else {
            return XCTFail(
                "expected shutdown-timeout error, got \(String(describing: collected.finishReason))"
            )
        }
        XCTAssertTrue(message.contains("shutdown"), message)
    }
}
