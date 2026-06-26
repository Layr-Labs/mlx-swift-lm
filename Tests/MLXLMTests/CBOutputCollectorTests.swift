// Tests for OutputCollector.swift — §2 of ContinuousBatchingTestPlan.md

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBOutputCollectorTests: XCTestCase {

    private func makeOutput(id: String = "r1", tokens: [Int] = [1], finished: Bool = false) -> RequestOutput {
        RequestOutput(
            requestId: id,
            newTokenIds: tokens,
            newText: tokens.map { String($0) }.joined(),
            finished: finished
        )
    }

    // MARK: - getNowait

    func testCollectorGetNowaitReturnsPutOutput() {
        let col = RequestOutputCollector(aggregate: false)
        let out = makeOutput(tokens: [42])
        col.put(out)

        let got = col.getNowait()
        XCTAssertEqual(got?.newTokenIds, [42])
        XCTAssertNil(col.getNowait(), "second getNowait must return nil")
    }

    // MARK: - Aggregation

    func testCollectorAggregationMergesTokenLists() {
        let col = RequestOutputCollector(aggregate: true)
        let a = makeOutput(tokens: [1, 2])
        let b = makeOutput(tokens: [3, 4])
        col.put(a)
        col.put(b)

        let merged = col.getNowait()
        XCTAssertEqual(merged?.newTokenIds, [1, 2, 3, 4])
        XCTAssertNil(col.getNowait())
    }

    func testCollectorNoAggregationOverwritesOutput() {
        let col = RequestOutputCollector(aggregate: false)
        col.put(makeOutput(tokens: [1]))
        col.put(makeOutput(tokens: [2]))

        XCTAssertEqual(col.getNowait()?.newTokenIds, [2])
    }

    // MARK: - Non-coalesceable markers (prefill-start admission)

    /// A non-coalesceable admission marker MUST survive as its own discrete output
    /// even when a token output is put before the consumer drains the buffer.
    /// FAILS on the pre-fix single-slot collector: `mergeOutputs` coalesces the
    /// token into the buffered marker, so the first output the consumer sees
    /// already carries `newTokenIds` and admission/first-token timestamps collapse.
    func testCollectorPreservesNonCoalesceableMarkerBeforeToken() {
        let col = RequestOutputCollector(aggregate: true)
        let marker = RequestOutput(requestId: "r1", promptTokens: 100, coalesceable: false)
        let token = makeOutput(tokens: [7])

        col.put(marker)   // buffered: no consumer waiting yet
        col.put(token)    // must NOT merge into the marker

        let first = col.getNowait()
        XCTAssertEqual(first?.newTokenIds, [], "admission marker must arrive first, token-less")
        XCTAssertEqual(first?.promptTokens, 100)
        let second = col.getNowait()
        XCTAssertEqual(second?.newTokenIds, [7], "first token arrives as a separate output")
        XCTAssertNil(col.getNowait(), "exactly two discrete outputs")
    }

    /// Tokens that follow the marker still aggregate AMONG THEMSELVES (the marker
    /// only blocks coalescing across itself), so streaming throughput is unchanged.
    func testCollectorAggregatesTokensAfterMarker() {
        let col = RequestOutputCollector(aggregate: true)
        col.put(RequestOutput(requestId: "r1", promptTokens: 100, coalesceable: false))
        col.put(makeOutput(tokens: [1]))
        col.put(makeOutput(tokens: [2]))

        XCTAssertEqual(col.getNowait()?.newTokenIds, [], "marker first")
        XCTAssertEqual(col.getNowait()?.newTokenIds, [1, 2], "post-marker tokens still merge")
        XCTAssertNil(col.getNowait())
    }

    // MARK: - Async get

    func testCollectorGetBlocksUntilPut() async {
        let col = RequestOutputCollector(aggregate: false)
        let expected = makeOutput(tokens: [99])

        let getTask = Task { await col.get() }

        // Brief yield so getTask reaches its suspension point.
        await Task.yield()

        col.put(expected)

        let result = await getTask.value
        XCTAssertEqual(result.newTokenIds, [99])
    }

    func testCollectorNoDuplicateDeliveryAfterContinuationResume() async {
        // Regression: put() must NOT buffer AND resume continuation.
        let col = RequestOutputCollector(aggregate: false)
        let out = makeOutput(tokens: [7])

        let getTask = Task { await col.get() }
        await Task.yield()

        col.put(out)

        _ = await getTask.value

        // After direct delivery, buffer must be empty.
        XCTAssertNil(col.getNowait())
    }

    // MARK: - clear

    func testCollectorClearDiscardsPendingOutput() {
        let col = RequestOutputCollector(aggregate: false)
        col.put(makeOutput(tokens: [5]))
        col.clear()
        XCTAssertNil(col.getNowait())
    }

    func testCollectorGetAfterClearResumesClearedContinuation() async {
        // put then clear — a subsequent get should block until a new put arrives.
        let col = RequestOutputCollector(aggregate: false)
        col.put(makeOutput(tokens: [1]))
        col.clear()

        let getTask = Task { await col.get() }
        await Task.yield()

        let fresh = makeOutput(tokens: [2])
        col.put(fresh)

        let result = await getTask.value
        XCTAssertEqual(result.newTokenIds, [2])
    }
}
