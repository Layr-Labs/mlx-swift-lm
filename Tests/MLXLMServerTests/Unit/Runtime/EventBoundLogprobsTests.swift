import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {
    @Suite struct EventBoundLogprobsTests {
        private static func entry(_ token: String) -> OpenAITokenLogprob {
            .init(
                token: token, logprob: -0.25, bytes: Array(token.utf8).map(Int.init),
                topLogprobs: [
                    .init(token: token, logprob: -0.25, bytes: Array(token.utf8).map(Int.init))
                ])
        }
        private static func request(_ parser: ReasoningParserFormat = .none)
            -> OpenAIChatCompletionRequest
        {
            .init(
                model: "scripted", messages: [.init(role: .user, content: .text("fixture"))],
                reasoningParser: parser, stream: true)
        }
        private static func choices(_ frames: [String]) throws -> [OpenAIChatCompletionChunk.Choice]
        {
            try frames.filter { $0 != ServerSentEventEncoder.done }.flatMap { frame in
                let line = try #require(
                    frame.split(separator: "\n").first(where: { $0.hasPrefix("data: ") }))
                return try JSONDecoder().decode(
                    OpenAIChatCompletionChunk.self, from: Data(line.dropFirst(6).utf8)
                ).choices
            }
        }
        private static func frames(
            _ events: [MLXServerGenerationEvent], parser: ReasoningParserFormat = .none
        ) async throws -> [String] {
            let service = MLXOpenAIService(engine: BufferedLogprobEngine(events: events))
            var output: [String] = []
            for try await frame in try await service.streamChatCompletionFrames(
                request: request(parser))
            { output.append(frame) }
            return output
        }

        @Test func producerAheadCannotMoveFutureMetadataIntoEarlierContent() async throws {
            // All events are queued synchronously before the service receives
            // the stream. This deterministically exercises producer-ahead.
            let a = Self.entry("A")
            let b = Self.entry("B")
            let c = Self.entry("C")
            let choices = try Self.choices(
                try await Self.frames([
                    .logprobs([a]), .content("A"), .logprobs([b]), .content("B"),
                    .logprobs([c]), .content("C"),
                ]))
            let content = choices.filter { $0.delta.content != nil }
            #expect(content.map(\.delta.content) == ["A", "B", "C"])
            #expect(content.map { $0.logprobs?.content } == [[a], [b], [c]])
            #expect(choices.filter { $0.delta.content == nil }.allSatisfy { $0.logprobs == nil })
        }

        @Test func emptyAndHeldDeltaStayOrderedWithoutTokenCountGuessing() async throws {
            let a = Self.entry("partial-byte")
            let b = Self.entry("rest")
            let c = Self.entry("next")
            let choices = try Self.choices(
                try await Self.frames([
                    .logprobs([a]), .content(""), .logprobs([b]), .content("é"),
                    .logprobs([c]), .content("x"),
                ]))
            let content = choices.filter { $0.delta.content != nil }
            #expect(content.map(\.delta.content) == ["é", "x"])
            #expect(content.map { $0.logprobs?.content } == [[a, b], [c]])
        }

        @Test func reasoningAndToolsRetainExistingHoldbackClassification() async throws {
            let reasoning = Self.entry("reason")
            let tool = Self.entry("tool")
            let visible = Self.entry("answer")
            let choices = try Self.choices(
                try await Self.frames([
                    .logprobs([reasoning]), .parsed(.init(content: "", reasoningContent: "reason")),
                    .logprobs([tool]),
                    .toolCall(.init(function: .init(name: "lookup", arguments: [:]))),
                    .logprobs([visible]), .parsed(.init(content: "answer", reasoningContent: nil)),
                ]))
            #expect(
                choices.contains { $0.delta.reasoningContent == "reason" && $0.logprobs == nil })
            #expect(choices.contains { $0.delta.toolCalls != nil && $0.logprobs == nil })
            // This deliberately pins the pre-existing limitation: source-span
            // classification is unavailable; this is not OpenAI reasoning-token parity.
            #expect(
                choices.first { $0.delta.content == "answer" }?.logprobs?.content == [
                    reasoning, tool, visible,
                ])
        }

        @Test func parserFinishUsesTheSameMetadataAttachmentBoundary() async throws {
            let held = Self.entry("<thi")
            let choices = try Self.choices(
                try await Self.frames([.logprobs([held]), .content("<thi")], parser: .deepseekR1))
            #expect(choices.first { $0.delta.content == "<thi" }?.logprobs?.content == [held])
        }

        @Test func trailingNonContentEntriesDoNotLeakIntoTerminalOrNextRequest() async throws {
            let service = MLXOpenAIService(
                engine: BufferedLogprobEngine(events: [.logprobs([Self.entry("stop")])]))
            for _ in 0 ..< 2 {
                var frames: [String] = []
                for try await frame in try await service.streamChatCompletionFrames(
                    request: Self.request())
                { frames.append(frame) }
                #expect(try Self.choices(frames).allSatisfy { $0.logprobs == nil })
                #expect(frames.last == ServerSentEventEncoder.done)
            }
        }

        @Test func collectedCompletionRetainsOrderedCapturedEntries() async throws {
            let entries = [Self.entry("a"), Self.entry("b"), Self.entry("c")]
            let service = MLXOpenAIService(
                engine: BufferedLogprobEngine(events: [
                    .logprobs([entries[0]]), .content("a"), .logprobs(Array(entries.dropFirst())),
                    .content("bc"),
                ]))
            let result = try await service.createChatCompletion(request: Self.request())
            #expect(result.choices.first?.message.content == .text("abc"))
            #expect(result.choices.first?.logprobs?.content == entries)
        }

        @Test func holdbackIsBoundedAndTakeReleasesEntries() {
            var buffer = ChatLogprobsBuffer()
            let entries = (0 ..< ChatLogprobsBuffer.maximumEntries + 3).map {
                Self.entry(String($0))
            }
            buffer.append(Array(entries.prefix(2)))
            buffer.append(Array(entries.dropFirst(2)))
            #expect(buffer.entries == Array(entries.suffix(ChatLogprobsBuffer.maximumEntries)))
            #expect(
                buffer.take()?.content == Array(entries.suffix(ChatLogprobsBuffer.maximumEntries)))
            #expect(buffer.entries.isEmpty)
            #expect(buffer.take() == nil)
        }

        @Test func consumerCancellationEndsTheOwnedProducerWithoutPublishingPendingMetadata()
            async throws
        {
            let engine = CancellableLogprobEngine()
            let service = MLXOpenAIService(engine: engine)
            let frames = try await service.streamChatCompletionFrames(request: Self.request())
            let started = LogprobTestSignal()
            let consumer = Task {
                for try await frame in frames {
                    if frame.contains("visible") { await started.signal() }
                }
            }
            await started.wait()
            consumer.cancel()
            _ = try? await consumer.value
            await engine.terminated.wait()
            #expect(await engine.terminated.isSignalled)
        }

        @Test func failureDoesNotAttachPendingEntriesToErrorOrSuccessFrame() async throws {
            let service = MLXOpenAIService(
                engine: BufferedLogprobEngine(events: [.logprobs([Self.entry("held")])], fail: true)
            )
            var frames: [String] = []
            for try await frame in try await service.streamChatCompletionFrames(
                request: Self.request(), frameGenerationErrors: true)
            { frames.append(frame) }
            #expect(frames.contains { $0.contains("\"error\"") })
            #expect(
                !frames.contains {
                    $0.contains("\"logprobs\"") || $0.contains("\"finish_reason\":\"stop\"")
                })
        }
    }
}

private struct BufferedLogprobEngine: MLXServerEngine {
    let events: [MLXServerGenerationEvent]
    var fail = false
    enum Failure: Error { case scripted }
    func availableModels() async throws -> [MLXServerModel] { [.init(id: "scripted")] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws
        -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    {
        AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            if fail {
                continuation.finish(throwing: Failure.scripted)
            } else {
                continuation.finish()
            }
        }
    }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: []) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        .init(text: "")
    }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        .init(tokens: [])
    }
}

private actor LogprobTestSignal {
    private var signalled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    var isSignalled: Bool { signalled }
    func signal() {
        signalled = true
        let pending = waiters
        waiters = []
        for waiter in pending { waiter.resume() }
    }
    func wait() async {
        if signalled { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}

private struct CancellableLogprobEngine: MLXServerEngine {
    let terminated = LogprobTestSignal()
    func availableModels() async throws -> [MLXServerModel] { [] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws
        -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    {
        AsyncThrowingStream { continuation in
            continuation.onTermination = { _ in Task { await terminated.signal() } }
            continuation.yield(.content("visible"))
            continuation.yield(
                .logprobs([.init(token: "held", logprob: -1, bytes: nil, topLogprobs: [])]))
            // The consumer must cancel the owned stream; no fabricated finish.
        }
    }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: []) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        .init(text: "")
    }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        .init(tokens: [])
    }
}
