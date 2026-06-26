// Tests for Scheduler — §7 of ContinuousBatchingTestPlan.md

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBSchedulerTests: XCTestCase {

    // MARK: - Helpers

    private func runUntilFinished(
        scheduler: Scheduler,
        maxSteps: Int = 30
    ) -> [RequestOutput] {
        var outputs: [RequestOutput] = []
        for _ in 0 ..< maxSteps {
            let result = scheduler.step()
            outputs.append(contentsOf: result.outputs)
            if !scheduler.hasRequests() { break }
        }
        return outputs
    }

    // MARK: - EOS stop

    func testSchedulerSingleRequestStopsOnEOS() {
        // Prompt [3]: model generates 4, then 5 (EOS).
        // finishReason == "stop"; token 5 absent from outputTokenIds.
        let s = makeTestScheduler(eosTokenIds: [5])
        let req = makeIntRequest(tokens: [3], maxTokens: 20)
        s.addRequest(req)

        let outputs = runUntilFinished(scheduler: s)
        let final_ = outputs.last { $0.finished }

        XCTAssertNotNil(final_)
        XCTAssertEqual(final_?.finishReason, "stop")
        XCTAssertFalse(final_?.outputTokenIds.contains(5) ?? true,
                       "EOS token must not appear in output")
    }

    // MARK: - Length finish includes final token

    func testSchedulerLengthFinishIncludesFinalToken() {
        // maxTokens: 3, no EOS → outputTokenIds.count == 3.
        let s = makeTestScheduler(eosTokenIds: [])
        let req = makeIntRequest(tokens: [1], maxTokens: 3)
        s.addRequest(req)

        let outputs = runUntilFinished(scheduler: s)
        let final_ = outputs.last { $0.finished }

        XCTAssertEqual(final_?.finishReason, "length")
        XCTAssertEqual(final_?.outputTokenIds.count, 3,
                       "length-capped finish must include the final token")
    }

    // MARK: - Stop string

    func testSchedulerStopStringHaltsGeneration() {
        // Prompt [1]: model generates 2, 3, 4, ...
        // Stop string "stop" encodes → [3] → generation halts when token 3 is produced.
        let s = makeTestScheduler(
            eosTokenIds: [],
            encodingMap: ["stop": [3]]
        )
        let req = Request(
            requestId: "r1",
            prompt: [1] as AnyHashable,
            samplingParams: SamplingParams(maxTokens: 20, stop: ["stop"])
        )
        s.addRequest(req)

        let outputs = runUntilFinished(scheduler: s)
        let final_ = outputs.last { $0.finished }

        XCTAssertNotNil(final_)
        XCTAssertEqual(final_?.finishReason, "stop")
        // Token 3 (stop token) must not appear in outputTokenIds.
        XCTAssertFalse(final_?.outputTokenIds.contains(3) ?? true)
    }

    // MARK: - Concurrent requests batch together

    func testSchedulerConcurrentRequestsBatchCorrectly() {
        let s = makeTestScheduler(eosTokenIds: [])
        s.addRequest(makeIntRequest(tokens: [1], maxTokens: 10))
        s.addRequest(makeIntRequest(tokens: [6], maxTokens: 10))

        _ = s.step()   // admits both into pendingPrefill
        _ = s.step()   // advances chunked prefill, promotes to genBatch

        XCTAssertEqual(s.getNumRunning(), 2)
    }

    // MARK: - Abort removes request

    func testSchedulerAbortRemovesRequest() {
        let s = makeTestScheduler(eosTokenIds: [])
        let req1 = makeIntRequest(id: "r1", tokens: [1], maxTokens: 20)
        let req2 = makeIntRequest(id: "r2", tokens: [6], maxTokens: 20)
        s.addRequest(req1)
        s.addRequest(req2)

        _ = s.step()  // both admitted and running

        s.abortRequest("r1")

        // Drain remaining steps; r1 must not appear in any output.
        var seenR1 = false
        for _ in 0 ..< 30 {
            let out = s.step()
            for ro in out.outputs where ro.requestId == "r1" { seenR1 = true }
            if !s.hasRequests() { break }
        }

        XCTAssertFalse(seenR1, "aborted request must not produce outputs")
        XCTAssertEqual(s.getNumRunning(), 0, "both requests should be done")
    }

    // MARK: - Repetition penalty doesn't break generation

    func testSchedulerRepetitionPenaltyDoesNotBreakGeneration() {
        let s = makeTestScheduler(eosTokenIds: [5])
        let req = Request(
            requestId: "r1",
            prompt: [3] as AnyHashable,
            samplingParams: SamplingParams(maxTokens: 10, repetitionPenalty: 2.0)
        )
        s.addRequest(req)

        let outputs = runUntilFinished(scheduler: s)
        let final_ = outputs.last { $0.finished }

        XCTAssertNotNil(final_, "generation must complete even with repetition penalty")
    }

    // MARK: - Prefill-start admission marker

    /// A newly admitted request must produce a token-less prefill-start
    /// `RequestOutput` AT the admit step — BEFORE any decode token — so a consumer
    /// can stamp engine-admission at real prefill start (the provider's
    /// observed_prefill_tps window). Previously the FIRST output a cold request
    /// produced was its first decode token, collapsing that window to ~0.
    func testSchedulerEmitsPrefillStartAdmissionMarker() {
        let s = makeTestScheduler(eosTokenIds: [])
        let req = makeIntRequest(id: "r1", tokens: [1, 2, 3, 4], maxTokens: 5)
        s.addRequest(req)

        // Admit step: emits exactly one token-less marker, no decode token yet.
        let admit = s.step()
        let markers = admit.outputs.filter { $0.requestId == "r1" }
        XCTAssertEqual(markers.count, 1, "exactly one admission marker at admit")
        let marker = markers[0]
        XCTAssertTrue(marker.newTokenIds.isEmpty, "marker carries no decoded token")
        XCTAssertEqual(marker.newText, "", "marker carries no text")
        XCTAssertFalse(marker.finished, "marker is not terminal")
        XCTAssertGreaterThan(marker.promptTokens, 0, "marker reports the prompt size")
        XCTAssertFalse(marker.coalesceable,
                       "marker must be non-coalesceable so the collector keeps it discrete")
        XCTAssertEqual(marker.cachedTokens, 0, "a cold prompt has no cached tokens")

        // The first DECODE token arrives on a LATER step, never collapsed into the
        // admit step — proving the admission→first-token window is real (> one step).
        var sawFirstToken = false
        for _ in 0 ..< 10 {
            let out = s.step()
            if out.outputs.contains(where: { $0.requestId == "r1" && !$0.newTokenIds.isEmpty }) {
                sawFirstToken = true
                break
            }
            if !s.hasRequests() { break }
        }
        XCTAssertTrue(sawFirstToken, "request must decode a first token after admission")
    }

    /// Real downstream path: feed each step's outputs into the SAME
    /// `RequestOutputCollector(aggregate: true)` that `EngineCore` uses, WITHOUT
    /// draining between puts (worst-case slow/backpressured consumer). The
    /// admission marker must STILL be delivered as a discrete, token-less output
    /// BEFORE the first decode token — otherwise `mergeOutputs` collapses admission
    /// and first-token and the prefill window is lost. FAILS on the pre-fix
    /// single-slot collector (the first token merges into the buffered marker).
    func testAdmissionMarkerSurvivesCollectorBackpressureBeforeFirstToken() {
        let s = makeTestScheduler(eosTokenIds: [])
        let req = makeIntRequest(id: "r1", tokens: [1, 2, 3, 4], maxTokens: 5)
        s.addRequest(req)

        let col = RequestOutputCollector(aggregate: true)
        var bufferedFirstToken = false
        for _ in 0 ..< 12 {
            let out = s.step()
            for ro in out.outputs where ro.requestId == "r1" {
                col.put(ro)  // never drained between puts (backpressure)
                if !ro.newTokenIds.isEmpty { bufferedFirstToken = true }
            }
            if bufferedFirstToken { break }
            if !s.hasRequests() { break }
        }
        XCTAssertTrue(bufferedFirstToken, "request must have produced a first token")

        // Drain: the FIRST output must be the token-less admission marker.
        let first = col.getNowait()
        XCTAssertNotNil(first)
        XCTAssertTrue(first!.newTokenIds.isEmpty, "admission marker must arrive first (token-less)")
        XCTAssertGreaterThan(first!.promptTokens, 0)
        // A later drained output carries the first token (survived as its own unit).
        var sawTokenOut = false
        while let o = col.getNowait() { if !o.newTokenIds.isEmpty { sawTokenOut = true } }
        XCTAssertTrue(sawTokenOut, "first token must survive as its own output after the marker")
    }
}
