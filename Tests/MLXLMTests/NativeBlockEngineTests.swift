import Foundation
import MLXLMCommon
import Testing

@Suite("Native block engine lifecycle and streaming", .serialized)
struct NativeBlockEngineTests {
    private struct BytesTokenizer: Tokenizer {
        var cleanupWhitespace = false
        var rewrite = false
        var pieces: [Int: String]?
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            if let pieces { return tokenIds.map { pieces[$0] ?? "" }.joined() }
            if rewrite { return tokenIds.count <= 1 ? "ab" : "xz" }
            let text = String(
                decoding: tokenIds.map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
            return cleanupWhitespace ? text.replacingOccurrences(of: "\n  ,", with: "\n ,") : text
        }
        func convertTokenToId(_ token: String) -> Int? { token.utf8.first.map(Int.init) }
        func convertIdToToken(_ id: Int) -> String? {
            decode(tokenIds: [id], skipSpecialTokens: false)
        }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [65] }
    }

    private final class Gate: @unchecked Sendable {
        private let lock = NSLock()
        private let releaseSignal = DispatchSemaphore(value: 0)
        private var entered = false
        private var waiter: CheckedContinuation<Void, Never>?
        func enterAndWait() {
            let continuation = lock.withLock {
                entered = true
                let value = waiter
                waiter = nil
                return value
            }
            continuation?.resume()
            releaseSignal.wait()
        }
        func waitUntilEntered() async {
            await withCheckedContinuation { continuation in
                let immediate = lock.withLock {
                    if entered { return true }
                    waiter = continuation
                    return false
                }
                if immediate { continuation.resume() }
            }
        }
        func release() { releaseSignal.signal() }
    }

    private final class Scripted: CBv2NativeBlockSession {
        let request: CBv2Request
        let cancellation: CBv2NativeBlockCancellation
        let blocks: [[Int]]
        var gate: Gate?
        var afterFirstBlockGate: Gate?
        var eos: Int?
        var index = -1
        var generatedTokenCount = 0
        var retainedBytes: Int { index < 0 ? 0 : 64 }
        var activeTokenCount: Int { request.promptTokens.count + generatedTokenCount }
        init(
            _ request: CBv2Request, _ cancellation: CBv2NativeBlockCancellation,
            blocks: [[Int]], gate: Gate? = nil, afterFirstBlockGate: Gate? = nil,
            eos: Int? = nil
        ) {
            self.request = request
            self.cancellation = cancellation
            self.blocks = blocks
            self.gate = gate
            self.afterFirstBlockGate = afterFirstBlockGate
            self.eos = eos
        }
        func cancel() { index = -2 }
        func advanceNative() throws -> CBv2NativeBlockStep {
            if index == 1, let afterFirstBlockGate {
                self.afterFirstBlockGate = nil
                afterFirstBlockGate.enterAndWait()
            }
            if let gate {
                self.gate = nil
                gate.enterAndWait()
            }
            if cancellation.isCancelled { throw CancellationError() }
            if index == -1 {
                index = 0
                return .prefill(computedTokens: request.promptTokens.count, complete: true)
            }
            let block = blocks[index]
            index += 1
            let stop = index == blocks.count ? eos : nil
            generatedTokenCount += block.count + (stop == nil ? 0 : 1)
            return .committed(
                tokens: block, stopToken: stop,
                finishReason: index == blocks.count ? (stop == nil ? .length : .stop) : nil)
        }
    }

    private func collect(_ stream: AsyncStream<CBv2Event>) async -> (
        text: String, tokens: [Int], usage: CBv2Usage?, reason: CBv2FinishReason?
    ) {
        var text = ""
        var tokens = [Int]()
        var usage: CBv2Usage?
        var reason: CBv2FinishReason?
        for await event in stream {
            switch event {
            case .delta(let next, let raw, _):
                text += next
                tokens += raw
            case .finished(let value, let counts):
                #expect(usage == nil, "Exactly one terminal event")
                usage = counts
                reason = value
            }
        }
        return (text, tokens, usage, reason)
    }

    @Test func completeBlockDecodingPreservesWhitespaceUTF8AndStopBoundaries() throws {
        let cleanup = CBv2NativeBlockTextDecoder(
            tokenizer: BytesTokenizer(cleanupWhitespace: true), stopStrings: [])
        #expect(try cleanup.append([10, 32, 32]) == "")
        #expect(try cleanup.append([44], terminal: true) == "\n ,")
        let unicode = CBv2NativeBlockTextDecoder(tokenizer: BytesTokenizer(), stopStrings: [])
        #expect(try unicode.append([0xe2]) == "")
        #expect(try unicode.append([0x82]) == "")
        #expect(try unicode.append([0xac], terminal: true) == "€")
        let stop = CBv2NativeBlockTextDecoder(tokenizer: BytesTokenizer(), stopStrings: ["<STOP>"])
        #expect(try stop.append(Array("hello<ST".utf8).map(Int.init)) == "hello")
        #expect(try stop.append(Array("OP>not visible".utf8).map(Int.init)) == "")
        #expect(stop.matchedStopString)
        let rewrite = CBv2NativeBlockTextDecoder(
            tokenizer: BytesTokenizer(rewrite: true), stopStrings: [])
        #expect(try rewrite.append([1]) == "ab")
        #expect(throws: CBv2NativeBlockError.tokenizerRewroteCommittedText) {
            try rewrite.append([2])
        }
    }

    @Test func committedEventsAccountExactlyAndEngineReleasesCapacity() async throws {
        let engine = try CBv2NativeBlockEngine(
            tokenizer: BytesTokenizer(), kvBytesCapacity: 200,
            reservationForRequest: { _ in 100 },
            makeSession: { request, cancellation in
                Scripted(request, cancellation, blocks: [[65, 32], [66]])
            })
        let result = await collect(
            try engine.submit(.init(id: .init(1), promptTokens: [1], maxTokens: 3)))
        #expect(result.text == "A B" && result.tokens == [65, 32, 66])
        #expect(result.reason == .length && result.usage?.completionTokens == 3)
        #expect(result.usage?.promptTokens == 1 && result.usage?.timing.prefillChunks == 1)
        #expect(result.usage?.timing.batchRowsMax == 1, "Round-robin is not rectangular batching")
        #expect(engine.capacity().kvBytesReserved == 0 && engine.capacity().activeRequests == 0)
        await engine.shutdown()
        #expect(throws: CBv2NativeBlockError.shuttingDown) {
            try engine.submit(.init(id: .init(2), promptTokens: [1], maxTokens: 3))
        }
    }

    @Test func stopStringDoesNotExposeOrChargeTheRemainingCanvas() async throws {
        let cases: [(pieces: [String], stop: String, visible: String, through: String)] = [
            (["hello<STOP>ignored canvas"], "<STOP>", "hello", "hello<STOP>"),
            (["hello<ST", "OP>ignored canvas"], "<STOP>", "hello", "hello<STOP>"),
            (["€<ST", "OP>ignored canvas"], "<STOP>", "€", "€<STOP>"),
            (["A ", " tail"], " ", "A", "A "),
        ]
        for (pieces, stop, visible, through) in cases {
            let blocks = pieces.map { Array($0.utf8).map(Int.init) }
            let engine = try CBv2NativeBlockEngine(
                tokenizer: BytesTokenizer(), kvBytesCapacity: 200,
                reservationForRequest: { _ in 100 },
                makeSession: { request, cancellation in
                    Scripted(request, cancellation, blocks: blocks)
                })
            let result = await collect(try engine.submit(.init(
                id: .init(1), promptTokens: [1], maxTokens: 64, stopStrings: [stop])))
            #expect(result.text == visible && result.reason == .stop)
            let throughStop = Array(through.utf8).map(Int.init)
            #expect(result.tokens == throughStop,
                "Native raw-token events must terminate at the token completing the requested stop")
            #expect(result.usage?.completionTokens == throughStop.count,
                "Do not charge trailing canvas tokens beyond the requested stop")
            await engine.shutdown()
            #expect(engine.capacity().activeRequests == 0 && engine.capacity().kvBytesReserved == 0)
        }
    }

    @Test func stopBoundaryUsesOriginalTokensCleanupAndShortestOverlappingDelimiter() throws {
        let token = CBv2NativeBlockTextDecoder(
            tokenizer: BytesTokenizer(pieces: [7: "hello<STOP>extra"]), stopStrings: ["<STOP>"])
        #expect(try token.append([7], terminal: true) == "hello")
        #expect(token.stopTokenCount == 1)
        let overlapping = CBv2NativeBlockTextDecoder(
            tokenizer: BytesTokenizer(), stopStrings: ["<S>long", "<S>"])
        #expect(try overlapping.append(Array("x<S>long ignored".utf8).map(Int.init)) == "x")
        #expect(overlapping.stopTokenCount == 4)
        let cleanup = CBv2NativeBlockTextDecoder(
            tokenizer: BytesTokenizer(cleanupWhitespace: true), stopStrings: [","])
        #expect(try cleanup.append([10, 32, 32]) == "")
        #expect(try cleanup.append([44, 88], terminal: true) == "\n ")
        #expect(cleanup.stopTokenCount == 4)
        let exact = CBv2NativeBlockTextDecoder(tokenizer: BytesTokenizer(), stopStrings: ["é"])
        #expect(try exact.append(Array("e\u{301}".utf8).map(Int.init), terminal: true) == "e\u{301}")
        #expect(!exact.matchedStopString && exact.stopTokenCount == nil)
    }

    @Test func unmatchedStopFlushesOriginalIDsAtEOSAndCancellation() async throws {
        let eosEngine = try CBv2NativeBlockEngine(tokenizer: BytesTokenizer(), kvBytesCapacity: 200,
            reservationForRequest: { _ in 100 }, makeSession: { request, cancellation in
                Scripted(request, cancellation, blocks: [[65], [66]], eos: 1)
            })
        let eosResult = await collect(try eosEngine.submit(.init(id: .init(1), promptTokens: [1],
            maxTokens: 8, stopStrings: ["NOT PRESENT"])))
        #expect(eosResult.text == "AB" && eosResult.tokens == [65, 66, 1])
        #expect(eosResult.usage?.completionTokens == 3 && eosResult.reason == .stop)
        await eosEngine.shutdown()

        let gate = Gate()
        defer { gate.release() }
        let engine = try CBv2NativeBlockEngine(tokenizer: BytesTokenizer(), kvBytesCapacity: 200,
            reservationForRequest: { _ in 100 }, makeSession: { request, cancellation in
                Scripted(request, cancellation, blocks: [[65], [66]], afterFirstBlockGate: gate)
            })
        let stream = try engine.submit(.init(id: .init(1), promptTokens: [1],
            maxTokens: 8, stopStrings: ["NOT PRESENT"]))
        let collector = Task { await collect(stream) }
        await gate.waitUntilEntered()
        engine.cancel(.init(1))
        gate.release()
        let result = await collector.value
        #expect(result.text == "A" && result.tokens == [65])
        #expect(result.usage?.completionTokens == 1 && result.reason == .cancelled)
        await engine.shutdown()
        #expect(engine.capacity().activeRequests == 0 && engine.capacity().kvBytesReserved == 0)
    }

    @Test func duplicateCapacityShrinkQueuedCancellationAndReadmissionAreAtomic() async throws {
        let gate = Gate()
        defer { gate.release() }
        let engine = try CBv2NativeBlockEngine(
            tokenizer: BytesTokenizer(), kvBytesCapacity: 200, maxConcurrentRequests: 1,
            maxWaiting: 1,
            reservationForRequest: { _ in 100 },
            makeSession: { request, cancellation in
                Scripted(
                    request, cancellation, blocks: [[65]], gate: request.id.raw == 1 ? gate : nil)
            })
        let first = try engine.submit(.init(id: .init(1), promptTokens: [1], maxTokens: 1))
        await gate.waitUntilEntered()
        #expect(throws: CBv2NativeBlockError.duplicateRequest) {
            try engine.submit(.init(id: .init(1), promptTokens: [1], maxTokens: 1))
        }
        let queued = try engine.submit(.init(id: .init(2), promptTokens: [1], maxTokens: 1))
        #expect(throws: CBv2KVError.self) {
            try engine.submit(.init(id: .init(3), promptTokens: [1], maxTokens: 1))
        }
        engine.updateKVBytesCapacity(100)
        #expect(
            engine.capacity().kvBytesReserved == 200 && engine.capacity().kvBytesCapacity == 100)
        engine.cancel(.init(2))
        gate.release()
        #expect(await collect(first).text == "A")
        let cancelled = await collect(queued)
        #expect(
            cancelled.reason == .cancelled && cancelled.tokens.isEmpty
                && cancelled.usage?.completionTokens == 0)
        #expect(
            await collect(try engine.submit(.init(id: .init(3), promptTokens: [1], maxTokens: 1)))
                .text == "A")
        await engine.shutdown()
        #expect(engine.capacity().kvBytesReserved == 0)
    }

    @Test func quietInFlightCancellationDoesNotEmitOrLoseTheSurvivor() async throws {
        let gate = Gate()
        defer { gate.release() }
        let engine = try CBv2NativeBlockEngine(
            tokenizer: BytesTokenizer(), kvBytesCapacity: 200, maxConcurrentRequests: 2,
            reservationForRequest: { _ in 100 },
            makeSession: { request, cancellation in
                Scripted(
                    request, cancellation, blocks: [[66]], gate: request.id.raw == 1 ? gate : nil)
            })
        let first = try engine.submit(.init(id: .init(1), promptTokens: [1], maxTokens: 1))
        await gate.waitUntilEntered()
        let survivor = try engine.submit(.init(id: .init(2), promptTokens: [1], maxTokens: 1))
        engine.cancel(.init(1))
        gate.release()
        let cancelled = await collect(first)
        #expect(cancelled.reason == .cancelled && cancelled.tokens.isEmpty)
        #expect(await collect(survivor).text == "B")
        await engine.shutdown()
        #expect(engine.capacity().activeRequests == 0 && engine.capacity().kvBytesReserved == 0)
    }
}
