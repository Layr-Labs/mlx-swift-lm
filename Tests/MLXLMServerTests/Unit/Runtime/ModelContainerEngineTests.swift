import Foundation
import MLXEmbedders
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {

    /// Tests of `MLXModelContainerEngine` with a stub model: the model
    /// list, the token utilities, the template rendering and the request
    /// checks that come before the engine makes MLX arrays. The streaming
    /// tests are in `Kernel/Runtime/ModelContainerEngineStreamTests.swift`.
    @Suite
    struct ModelContainerEngineTests {

        private func makeEngine(
            model: any LanguageModel = StubLanguageModel(),
            configuration: ModelConfiguration = ModelConfiguration(id: "test/stub"),
            recorder: Recorder = Recorder(), modelType: String? = nil
        ) -> MLXModelContainerEngine {
            let container = ModelContainer(
                context: ModelContext(
                    configuration: configuration, model: model,
                    processor: RecordingProcessor(recorder: recorder),
                    tokenizer: ScalarTokenizer(recorder: recorder)))
            return MLXModelContainerEngine(
                modelID: "served-model", model: container, modelType: modelType)
        }

        private func request(
            _ messages: [OpenAIChatMessage] = [.init(role: .user, content: .text("hi"))],
            tools: [OpenAITool]? = nil, toolCallParser: String? = nil
        ) -> OpenAIChatCompletionRequest {
            OpenAIChatCompletionRequest(
                model: "served-model", messages: messages, tools: tools,
                toolCallParser: toolCallParser)
        }

        /// Runs the request and returns the error. A request that passes
        /// every check reaches the processor, which throws
        /// `ProcessorReached`.
        private func streamError(
            _ engine: MLXModelContainerEngine, _ request: OpenAIChatCompletionRequest
        ) async -> (any Error)? {
            do {
                _ = try await engine.streamChatCompletion(request: request)
                return nil
            } catch {
                return error
            }
        }

        // MARK: - Models and token utilities

        @Test func availableModelsListsTheModelID() async throws {
            let models = try await makeEngine().availableModels()
            #expect(models == [MLXServerModel(id: "served-model")])
            #expect(models.first?.ownedBy == "local")
        }

        @Test func tokenizeAddsSpecialTokensByDefault() async throws {
            let engine = makeEngine()
            #expect(try await engine.tokenize(.init(prompt: "hi")).tokens == [1, 104, 105])
            #expect(
                try await engine.tokenize(.init(prompt: "hi", addSpecialTokens: false)).tokens
                    == [104, 105])
        }

        @Test func detokenizeKeepsSpecialTokensByDefault() async throws {
            let engine = makeEngine()
            #expect(try await engine.detokenize(.init(tokens: [1, 104, 105])).text == "<1>hi")
            #expect(
                try await engine.detokenize(.init(tokens: [1, 104, 105], skipSpecialTokens: true))
                    .text == "hi")
        }

        /// `/apply-template` gives the template the full messages: name,
        /// tool call id and tool calls with decoded arguments.
        @Test func applyTemplateRendersTheFullMessages() async throws {
            let recorder = Recorder()
            let engine = makeEngine(recorder: recorder)
            let tools = [
                OpenAITool(function: .init(name: "lookup", description: "find a value"))
            ]
            let response = try await engine.applyTemplate(
                .init(
                    messages: [
                        .init(role: .system, content: .text("s")),
                        .init(
                            role: .assistant, content: .null,
                            toolCalls: [
                                .init(
                                    id: "call_1",
                                    function: .init(name: "lookup", arguments: #"{"q": "x"}"#))
                            ]),
                        .init(role: .tool, content: .text("v"), name: "lookup", toolCallID: "call_1"),
                    ], tools: tools))

            // "system:s|assistant:|tool:v" without special tokens, then 1 tool.
            let text = "system:s|assistant:|tool:v".unicodeScalars.map { Int($0.value) }
            #expect(response.tokens == text + [1])

            let messages = recorder.messages
            #expect(messages.count == 3)
            #expect(messages[2]["name"] as? String == "lookup")
            #expect(messages[2]["tool_call_id"] as? String == "call_1")
            let calls = messages[1]["tool_calls"] as? [[String: any Sendable]]
            #expect(calls?.count == 1)
            #expect(calls?.first?["id"] as? String == "call_1")
            let function = calls?.first?["function"] as? [String: any Sendable]
            #expect(function?["name"] as? String == "lookup")
            #expect(function?["arguments"] as? [String: any Sendable] != nil)

            let function0 = recorder.tools?.first?["function"] as? [String: any Sendable]
            #expect(function0?["name"] as? String == "lookup")
            #expect(function0?["description"] as? String == "find a value")
        }

        // MARK: - Request checks

        @Test func mediaIsRejectedBeforeAnyModelWork() async {
            let recorder = Recorder()
            let engine = makeEngine(recorder: recorder)
            let error = await streamError(
                engine,
                request([.init(role: .user, content: .parts([.text("a"), .imageURL("data:,")]))]))
            #expect(error as? MLXModelContainerEngineError == .mediaUnsupported)
            #expect(recorder.input == nil)
        }

        @Test func samplingAndReasoningControlsAreRejected() async {
            let engine = makeEngine()
            var seeded = request()
            seeded.seed = 4
            #expect(
                await streamError(engine, seeded) as? MLXModelContainerEngineError
                    == .unsupportedSamplingControl("seed"))

            var reasoning = request()
            reasoning.reasoning = .init(effort: "low")
            #expect(
                await streamError(engine, reasoning) as? MLXModelContainerEngineError
                    == .unsupportedReasoningControl("reasoning.effort"))
        }

        @Test func modelThatNeedsNativeGenerationIsRejected() async {
            let engine = makeEngine(model: NativeOnlyStubModel())
            let reason = GenericGenerationError.nativeCBv2Required(modelType: "stub_native")
                .localizedDescription
            #expect(
                await streamError(engine, request()) as? MLXModelContainerEngineError
                    == .nativeGenerationRequired(reason))
            #expect(reason.hasPrefix("stub_native requires native CBv2 generation"))
        }

        /// With no format in the configuration the stream parses JSON, so
        /// the pinned format is `.json`.
        @Test func conflictingToolParserOverrideIsRejected() async {
            let engine = makeEngine()
            #expect(
                await streamError(engine, request(toolCallParser: "mistral"))
                    as? MLXModelContainerEngineError
                    == .unsupportedToolCallParser(pinned: .json, requested: .mistral))

            let gemma = makeEngine(
                configuration: ModelConfiguration(id: "test/stub", toolCallFormat: .gemma))
            #expect(
                await streamError(gemma, request(toolCallParser: "json"))
                    as? MLXModelContainerEngineError
                    == .unsupportedToolCallParser(pinned: .gemma, requested: .json))
        }

        @Test func unknownToolParserOverrideIsRejected() async {
            let error = await streamError(makeEngine(), request(toolCallParser: "nope"))
            #expect(error as? ServerToolParserError == .unsupported("nope"))
        }

        /// A request without an override, with `auto`, or with an override
        /// that resolves to the pinned format passes the checks.
        @Test(arguments: [nil, "auto", "AUTO", "json", "qwen3", "default"] as [String?])
        func matchingToolParserOverrideIsAccepted(_ parser: String?) async {
            let recorder = Recorder()
            let error = await streamError(
                makeEngine(recorder: recorder), request(toolCallParser: parser))
            #expect(error is ProcessorReached)
            #expect(recorder.input == "chat 1")
        }

        /// A plain chat goes to the processor as structured chat messages,
        /// with the request tools.
        @Test func plainChatUsesChatMessages() async {
            let recorder = Recorder()
            let tools = [OpenAITool(function: .init(name: "lookup"))]
            let error = await streamError(
                makeEngine(recorder: recorder),
                request(
                    [
                        .init(role: .system, content: .text("s")),
                        .init(role: .user, content: .text("u")),
                    ], tools: tools))
            #expect(error is ProcessorReached)
            #expect(recorder.input == "chat 2")
            #expect(recorder.inputToolNames == ["lookup"])
        }

        /// Tool history goes to the processor as template messages, so that
        /// the template sees the tool fields.
        @Test func toolHistoryUsesTemplateMessages() async {
            let recorder = Recorder()
            let error = await streamError(
                makeEngine(recorder: recorder),
                request([
                    .init(role: .user, content: .text("u")),
                    .init(
                        role: .assistant, content: .null,
                        toolCalls: [
                            .init(id: "c", function: .init(name: "lookup", arguments: "{}"))
                        ]),
                    .init(role: .tool, content: .text("v"), toolCallID: "c"),
                ]))
            #expect(error is ProcessorReached)
            #expect(recorder.input == "messages 3")
            #expect(recorder.inputToolNames == nil)
        }

        @Test func templateMessagesAreNeededForFieldsThatChatDrops() {
            let user = OpenAIChatMessage(role: .user, content: .text("u"))
            #expect(!MLXModelContainerEngine.requiresTemplateMessages([user]))
            #expect(
                !MLXModelContainerEngine.requiresTemplateMessages([
                    .init(role: .assistant, content: .text("a"), toolCalls: [])
                ]))
            #expect(
                MLXModelContainerEngine.requiresTemplateMessages([
                    user, .init(role: .user, content: .text("n"), name: "bob"),
                ]))
            #expect(
                MLXModelContainerEngine.requiresTemplateMessages([
                    .init(role: .tool, content: .text("t"), toolCallID: "c")
                ]))
            #expect(
                MLXModelContainerEngine.requiresTemplateMessages([
                    .init(role: .assistant, content: .text("a"), reasoningContent: "r")
                ]))
        }

        @Test func toolParserValidatorRules() throws {
            try MLXModelContainerEngine.validateToolParserOverride(
                requested: nil, pinned: .harmony, modelType: nil)
            try MLXModelContainerEngine.validateToolParserOverride(
                requested: "Auto", pinned: .harmony, modelType: nil)
            try MLXModelContainerEngine.validateToolParserOverride(
                requested: "gpt-oss", pinned: .harmony, modelType: nil)
            #expect(throws: MLXModelContainerEngineError.unsupportedToolCallParser(
                pinned: .harmony, requested: .llama3)
            ) {
                try MLXModelContainerEngine.validateToolParserOverride(
                    requested: "llama3", pinned: .harmony, modelType: "gpt_oss")
            }
        }

        // MARK: - Embedding engine

        @Test func embeddingEngineListsTheModelID() async throws {
            let container = EmbedderModelContainer(
                context: EmbedderModelContext(
                    configuration: ModelConfiguration(id: "test/embedder"),
                    model: StubEmbeddingModel(), tokenizer: ScalarTokenizer(),
                    pooling: Pooling(strategy: .mean)))
            let engine = MLXEmbedderContainerEngine(modelID: "embedder", model: container)
            #expect(try await engine.availableModels() == [MLXServerModel(id: "embedder")])
        }
    }
}
