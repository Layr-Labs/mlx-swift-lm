import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Native block leases, watchdog and actual retirement", .serialized)
struct NativeBlockDeadlineTests {
    private final class Clock: @unchecked Sendable {
        private let lock = NSLock()
        private var time = ContinuousClock.now
        var clock: CBv2Clock { .init { self.lock.withLock { self.time } } }
        func advance(_ seconds: Double) { lock.withLock { time = time.advanced(by: .seconds(seconds)) } }
    }
    private final class Gate: @unchecked Sendable {
        private let lock = NSLock()
        private let semaphore = DispatchSemaphore(value: 0)
        private var entered = false
        func block() { lock.withLock { entered = true }; semaphore.wait() }
        var isEntered: Bool { lock.withLock { entered } }
        func release() { semaphore.signal() }
    }
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        @discardableResult func increment() -> Int { lock.withLock { count += 1; return count } }
        var value: Int { lock.withLock { count } }
    }
    private struct Tokens: Tokenizer {
        var bosToken: String? { nil }; var eosToken: String? { nil }; var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
        }
        func convertTokenToId(_ token: String) -> Int? { token.utf8.first.map(Int.init) }
        func convertIdToToken(_ id: Int) -> String? { decode(tokenIds: [id], skipSpecialTokens: false) }
        func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?) throws -> [Int] { [1] }
    }
    private final class Session: CBv2NativeBlockSession {
        let gate: Gate?
        let cancellation: CBv2NativeBlockCancellation
        var index = 0
        var generatedTokenCount = 0
        var retainedBytes: Int { 64 }
        var activeTokenCount: Int { 1 + generatedTokenCount }
        init(gate: Gate?, cancellation: CBv2NativeBlockCancellation) {
            self.gate = gate; self.cancellation = cancellation
        }
        func cancel() {}
        func advanceNative() throws -> CBv2NativeBlockStep {
            defer { index += 1 }
            if index == 0 { return .prefill(computedTokens: 1, complete: true) }
            if index == 2 { gate?.block() }
            if cancellation.isCancelled { throw CancellationError() }
            generatedTokenCount += 1
            return .committed(tokens: [64 + index], stopToken: nil, finishReason: index == 2 ? .length : nil)
        }
    }
    private func eventually(_ name: String = "condition", _ predicate: @escaping @Sendable () -> Bool) async throws {
        let until = ContinuousClock.now.advanced(by: .seconds(2))
        while !predicate(), ContinuousClock.now < until { try await Task.sleep(for: .milliseconds(2)) }
        try #require(predicate(), "Timed out waiting for \(name)")
    }
    private func collect(_ stream: AsyncStream<CBv2Event>) async -> ([Int], CBv2FinishReason?, CBv2Usage?) {
        var ids = [Int](), reason: CBv2FinishReason?, usage: CBv2Usage?
        for await event in stream {
            switch event {
            case .delta(_, let tokens, _): ids += tokens
            case .finished(let end, let counts):
                #expect(reason == nil); reason = end; usage = counts
            }
        }
        return (ids, reason, usage)
    }

    @Test func watchdogTerminatesButRetirementAndBudgetWaitForBlockedQuantum() async throws {
        let clock = Clock(), gate = Gate(), starts = Counter(), wedges = Counter(), retired = Counter()
        let engine = try CBv2NativeBlockEngine(tokenizer: Tokens(), kvBytesCapacity: 200,
            outputBufferCapacity: 8, shutdownGraceSeconds: 0,
            loopConfig: .init(stepTimeout: 1, watchdogInterval: 0.01, clock: clock.clock),
            reservationForRequest: { _ in 100 }, makeSession: { _, cancellation in
                Session(gate: starts.increment() == 1 ? gate : nil, cancellation: cancellation)
            })
        engine.onStepWedge = { _ in wedges.increment() }
        defer { gate.release() }
        do {
            let submitted = try engine.submitWithRetirement(.init(id: .init(1), promptTokens: [1], maxTokens: 2))
            let consumer = Task { await collect(submitted.events) }
            let retirement = Task { await submitted.retirement.wait(); retired.increment() }
            try await eventually { gate.isEntered }
            clock.advance(2)
            try await eventually { !engine.isHealthy }
            let result = await consumer.value
            #expect(result.0 == [65] && result.2?.completionTokens == 1)
            #expect(result.1 == .terminal(cause: .watchdog, message: CBv2TerminalCause.watchdog.diagnostic))
            #expect(engine.capacity().kvBytesReserved == 100 && retired.value == 0)
            #expect(wedges.value == 1)
            #expect(throws: CBv2NativeBlockError.shuttingDown) {
                try engine.submit(.init(id: .init(2), promptTokens: [1], maxTokens: 2))
            }
            gate.release()
            await retirement.value
            #expect(engine.capacity().kvBytesReserved == 0 && engine.isHealthy)
            let readmitted = await collect(try engine.submit(.init(id: .init(1), promptTokens: [1], maxTokens: 2)))
            #expect(readmitted.0 == [65, 66] && readmitted.1 == .length)
            #expect(wedges.value == 1)
            await engine.shutdown()
        } catch { gate.release(); await engine.shutdown(); throw error }
    }

    @Test func pausedConsumerAndQueuedAdmissionKeepDistinctTypedLeases() async throws {
        let clock = Clock()
        let engine = try CBv2NativeBlockEngine(tokenizer: Tokens(), kvBytesCapacity: 300,
            maxConcurrentRequests: 1, outputBufferCapacity: 1, shutdownGraceSeconds: 0,
            loopConfig: .init(stepTimeout: 100, watchdogInterval: 0.01,
                admissionLease: 1, prefillProgressLease: 1, decodeProgressLease: 1,
                backpressureLease: 5, clock: clock.clock),
            reservationForRequest: { _ in 100 }, makeSession: { _, cancellation in Session(gate: nil, cancellation: cancellation) })
        do {
            let paused = try engine.submit(.init(id: .init(1), promptTokens: [1], maxTokens: 2))
            try await eventually("first request paused") { engine.pausedRequestCountForTesting == 1 }
            let waiting = try engine.submit(.init(id: .init(2), promptTokens: [1], maxTokens: 2))
            clock.advance(2)
            try await eventually("queued admission expired") { engine.capacity().waitingRequests == 0 }
            let queuedResult = await collect(waiting)
            #expect(queuedResult.1 == .terminal(cause: .admissionTimeout, message: CBv2TerminalCause.admissionTimeout.diagnostic))
            #expect(queuedResult.0.isEmpty && queuedResult.2?.completionTokens == 0)
            #expect(engine.capacity().activeRequests == 1 && engine.isHealthy)
            clock.advance(4)
            try await eventually("all-paused backpressure expired") { engine.capacity().activeRequests == 0 }
            let pausedResult = await collect(paused)
            #expect(pausedResult.0 == [65] && pausedResult.2?.completionTokens == 1)
            #expect(pausedResult.1 == .terminal(cause: .backpressureTimeout, message: CBv2TerminalCause.backpressureTimeout.diagnostic))
            #expect(engine.capacity().kvBytesReserved == 0)
            await engine.shutdown()
        } catch { engine.cancel(.init(1)); engine.cancel(.init(2)); await engine.shutdown(); throw error }
    }

    @Test func confirmedNativeQuantaRefreshWithoutInventingOutputOrRemovingSafetyCeiling() {
        let t = ContinuousClock.now
        var lease = CBv2RequestLeaseState(now: t, admissionLease: 10, prefillLease: 10,
            decodeLease: 10, backpressureLease: 10, safety: .seconds(100), computedTokens: 0, generatedTokens: 0)
        lease.recordNativeProgress(now: t.advanced(by: .seconds(1)), phase: .decode, completedWork: 1)
        lease.recordNativeProgress(now: t.advanced(by: .seconds(9)), phase: .decode, completedWork: 1)
        #expect(lease.expiredCause(now: t.advanced(by: .seconds(12)), isRunning: true, isPaused: false) == .decodeStall)
        lease.recordNativeProgress(now: t.advanced(by: .seconds(99)), phase: .decode, completedWork: 2)
        #expect(lease.expiredCause(now: t.advanced(by: .seconds(101)), isRunning: true, isPaused: false) == .safetyDeadline)
    }

    @Test func aWedgedFactoryDoesNotConstructTheCancelledWaitingCohort() async throws {
        let clock = Clock(), gate = Gate(), constructed = Counter()
        let engine = try CBv2NativeBlockEngine(tokenizer: Tokens(), kvBytesCapacity: 300,
            maxConcurrentRequests: 2, shutdownGraceSeconds: 0,
            loopConfig: .init(stepTimeout: 1, watchdogInterval: 0.01, clock: clock.clock),
            reservationForRequest: { _ in 100 }, makeSession: { _, cancellation in
                if constructed.increment() == 1 { gate.block() }
                return Session(gate: nil, cancellation: cancellation)
            })
        defer { gate.release() }
        do {
            let a = try engine.submitWithRetirement(.init(id: .init(1), promptTokens: [1], maxTokens: 2))
            try await eventually("factory entered") { gate.isEntered }
            let b = try engine.submitWithRetirement(.init(id: .init(2), promptTokens: [1], maxTokens: 2))
            clock.advance(2)
            try await eventually("factory watchdog") { !engine.isHealthy }
            let first = await collect(a.events), second = await collect(b.events)
            #expect(first.0.isEmpty && second.0.isEmpty)
            #expect(first.1 == .terminal(cause: .watchdog, message: CBv2TerminalCause.watchdog.diagnostic))
            #expect(second.1 == first.1 && engine.capacity().kvBytesReserved == 200)
            gate.release()
            await a.retirement.wait(); await b.retirement.wait()
            #expect(constructed.value == 1 && engine.capacity().kvBytesReserved == 0)
            await engine.shutdown()
        } catch { gate.release(); await engine.shutdown(); throw error }
    }
}
