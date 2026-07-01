// Copyright © 2026 Eigen Labs.
//
// WS-B: CBv2OutputStream tests — ordering, bounded buffering with
// backpressure pause/resume, terminal idempotence, abandonment.

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBv2SchedulerStreamTests: XCTestCase {

    private func delta(_ token: Int) -> CBv2Event {
        .delta(text: "<\(token)>", tokens: [token], logprobs: nil)
    }

    private func usage(_ completion: Int = 0) -> CBv2Usage {
        CBv2Usage(promptTokens: 1, completionTokens: completion)
    }

    func testEventsDeliveredInOrderAndStreamEndsAfterFinish() async {
        let stream = CBv2OutputStream(id: CBv2RequestID(1), capacity: 16)
        for token in 0 ..< 5 { stream.emit(delta(token)) }
        stream.finish(reason: .stop, usage: usage(5))

        var tokens: [Int] = []
        var reason: CBv2FinishReason?
        for await event in stream.makeStream() {
            switch event {
            case .delta(_, let t, _): tokens.append(contentsOf: t)
            case .finished(let r, _): reason = r
            }
        }
        XCTAssertEqual(tokens, [0, 1, 2, 3, 4])
        XCTAssertEqual(reason, .stop)
    }

    func testBackpressureFiresOncePerTransition() async {
        // Track transitions without touching an engine.
        final class Transitions: @unchecked Sendable {
            private let lock = NSLock()
            private var _events: [Bool] = []
            func append(_ paused: Bool) {
                lock.lock()
                _events.append(paused)
                lock.unlock()
            }
            var events: [Bool] {
                lock.lock()
                defer { lock.unlock() }
                return _events
            }
        }
        let transitions = Transitions()
        let stream = CBv2OutputStream(
            id: CBv2RequestID(2), capacity: 4,
            onBackpressure: { _, paused in transitions.append(paused) })

        // Fill to capacity: exactly one pause.
        for token in 0 ..< 4 { stream.emit(delta(token)) }
        XCTAssertEqual(transitions.events, [true])
        // Producer may run slightly past capacity (in-flight steps) without
        // re-firing.
        stream.emit(delta(4))
        stream.emit(delta(5))
        XCTAssertEqual(transitions.events, [true])
        XCTAssertEqual(stream.bufferedCount, 6)

        // Drain to the low watermark (capacity/2 = 2): exactly one resume.
        var drained: [Int] = []
        while drained.count < 4, let event = await stream.next() {
            if case .delta(_, let t, _) = event { drained.append(contentsOf: t) }
        }
        XCTAssertEqual(drained, [0, 1, 2, 3])
        XCTAssertEqual(transitions.events, [true, false])

        // Refill: pause fires again on the next crossing.
        for token in 6 ..< 9 { stream.emit(delta(token)) }
        XCTAssertEqual(transitions.events, [true, false, true])
    }

    func testFinishIsIdempotentAndLaterEmitsAreDropped() async {
        let stream = CBv2OutputStream(id: CBv2RequestID(3), capacity: 8)
        stream.emit(delta(1))
        stream.finish(reason: .stop, usage: usage(1))
        // Watchdog/loop double-finish and post-terminal emits must no-op.
        stream.finish(reason: .error("late"), usage: usage(0))
        stream.emit(delta(99))

        var events: [CBv2Event] = []
        for await event in stream.makeStream() { events.append(event) }
        XCTAssertEqual(events.count, 2)
        guard case .delta(_, let tokens, _) = events[0], tokens == [1] else {
            return XCTFail("expected the pre-finish delta first")
        }
        guard case .finished(let reason, _) = events[1], reason == .stop else {
            return XCTFail("expected the FIRST terminal reason to win")
        }
    }

    func testParkedConsumerWakesOnEmit() async {
        let stream = CBv2OutputStream(id: CBv2RequestID(4), capacity: 8)
        let consumer = Task { () -> [Int] in
            var tokens: [Int] = []
            for await event in stream.makeStream() {
                if case .delta(_, let t, _) = event { tokens.append(contentsOf: t) }
            }
            return tokens
        }
        // Give the consumer time to park, then produce.
        try? await Task.sleep(nanoseconds: 20_000_000)
        stream.emit(delta(7))
        stream.finish(reason: .length, usage: usage(1))
        let tokens = await consumer.value
        XCTAssertEqual(tokens, [7])
    }

    func testAbandonmentFiresOnConsumerCancellation() async {
        final class Flag: @unchecked Sendable {
            private let lock = NSLock()
            private var _abandoned: [CBv2RequestID] = []
            func mark(_ id: CBv2RequestID) {
                lock.lock()
                _abandoned.append(id)
                lock.unlock()
            }
            var abandoned: [CBv2RequestID] {
                lock.lock()
                defer { lock.unlock() }
                return _abandoned
            }
        }
        let flag = Flag()
        let id = CBv2RequestID(5)
        let stream = CBv2OutputStream(
            id: id, capacity: 8, onAbandoned: { flag.mark($0) })

        let consumer = Task {
            for await _ in stream.makeStream() {}
        }
        try? await Task.sleep(nanoseconds: 20_000_000)  // let it park
        consumer.cancel()
        _ = await consumer.value
        let ok = await cbv2SchedWait { flag.abandoned == [id] }
        XCTAssertTrue(ok, "cancelling the consumer must signal abandonment")
    }
}
