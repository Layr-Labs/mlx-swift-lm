import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {
    @Suite
    struct OpenAIResponseShapingTests {

        private func encodedString<T: Encodable>(_ value: T) throws -> String {
            String(decoding: try JSONEncoder.openAIServer.encode(value), as: UTF8.self)
        }

        @Test
        func usageAddsTheTotalAndClampsCachedTokens() throws {
            let usage = OpenAIUsage(promptTokens: 10, completionTokens: 5, cachedPromptTokens: 20)
            #expect(usage.totalTokens == 15)
            #expect(usage.promptTokensDetails?.cachedTokens == 10)
            #expect(
                OpenAIUsage(promptTokens: 3, completionTokens: 0, cachedPromptTokens: -4)
                    .promptTokensDetails?.cachedTokens == 0)
            #expect(OpenAIUsage(promptTokens: 1, completionTokens: 1).promptTokensDetails == nil)
            #expect(
                try encodedString(
                    OpenAIUsage(promptTokens: 2, completionTokens: 1, cachedPromptTokens: 1))
                    == #"{"completion_tokens":1,"prompt_tokens":2,"prompt_tokens_details":{"cached_tokens":1},"total_tokens":3}"#
            )
        }

        @Test
        func generationInfoClampsCachedTokens() {
            #expect(
                ServerGenerationInfo(
                    promptTokens: 4, completionTokens: 2, promptTime: 0, generationTime: 0,
                    stopReason: "stop", cachedPromptTokens: 9
                ).cachedPromptTokens == 4)
            #expect(
                ServerGenerationInfo(
                    promptTokens: 4, completionTokens: 2, promptTime: 0, generationTime: 0,
                    stopReason: "stop"
                ).cachedPromptTokens == nil)
        }

        @Test
        func stopReasonsMapToOpenAIFinishReasons() {
            #expect(GenerateStopReason.stop.openAIFinishReason == "stop")
            #expect(GenerateStopReason.length.openAIFinishReason == "length")
            #expect(GenerateStopReason.cancelled.openAIFinishReason == "stop")
        }

        @Test
        func chunkDeltaCarriesReasoningInBothFields() throws {
            let chunk = OpenAIChatCompletionChunk(
                id: "chatcmpl-1", model: "m",
                choices: [
                    .init(
                        index: 0,
                        delta: .init(
                            role: nil, content: nil, reasoningContent: "why", toolCalls: nil),
                        finishReason: nil)
                ],
                usage: nil, created: 5)
            #expect(chunk.object == "chat.completion.chunk")
            #expect(
                try encodedString(chunk)
                    == #"{"choices":[{"delta":{"reasoning":"why","reasoning_content":"why"},"index":0}],"created":5,"id":"chatcmpl-1","model":"m","object":"chat.completion.chunk"}"#
            )
        }

        @Test
        func completionResponseIsBuiltFromAChatResponse() {
            let chat = OpenAIChatCompletionResponse(
                id: "chatcmpl-abc", model: "m",
                choices: [
                    .init(
                        index: 0,
                        message: .init(
                            role: .assistant,
                            content: .parts([.text("hel"), .imageURL("u"), .text("lo")])),
                        finishReason: "length")
                ],
                usage: .init(promptTokens: 3, completionTokens: 2), created: 11)
            let completion = OpenAICompletionResponse(from: chat)
            #expect(completion.id == "cmpl-abc")
            #expect(completion.object == "text_completion")
            #expect(completion.created == 11)
            #expect(completion.model == "m")
            #expect(completion.choices == [.init(text: "hello", index: 0, finishReason: "length")])
            #expect(completion.usage.totalTokens == 5)
        }

        @Test
        func embeddingResponseHasTheListShape() throws {
            let response = OpenAIEmbeddingResponse(
                data: [.init(embedding: [0.5, 1], index: 0)], model: "e",
                usage: .init(promptTokens: 2, completionTokens: 0))
            #expect(response.object == "list")
            #expect(response.data.first?.object == "embedding")
            #expect(
                try encodedString(response)
                    == #"{"data":[{"embedding":[0.5,1],"index":0,"object":"embedding"}],"model":"e","object":"list","usage":{"completion_tokens":0,"prompt_tokens":2,"total_tokens":2}}"#
            )
        }

        @Test
        func modelEncodesTheContextLengthTwice() throws {
            #expect(
                try encodedString(MLXServerModel(id: "m", created: 1, contextLength: 4096))
                    == #"{"context_length":4096,"created":1,"id":"m","max_model_len":4096,"object":"model","owned_by":"local"}"#
            )
            #expect(
                try encodedString(MLXServerModel(id: "m"))
                    == #"{"id":"m","object":"model","owned_by":"local"}"#)
            #expect(
                try encodedString(OpenAIModelListResponse(data: [MLXServerModel(id: "a")]))
                    == #"{"data":[{"id":"a","object":"model","owned_by":"local"}],"object":"list"}"#
            )
        }

        @Test
        func modelDecodesDefaultsAndTheVLLMContextKey() throws {
            let minimal = try JSONDecoder().decode(
                MLXServerModel.self, from: Data(#"{"id":"m"}"#.utf8))
            #expect(minimal == MLXServerModel(id: "m"))
            let vllm = try JSONDecoder().decode(
                MLXServerModel.self,
                from: Data(
                    #"{"id":"m","object":"model","owned_by":"org","max_model_len":8192}"#.utf8))
            #expect(vllm.ownedBy == "org")
            #expect(vllm.contextLength == 8192)
        }

        @Test
        func serverSentEventFramesEndWithABlankLine() throws {
            #expect(try ServerSentEventEncoder.encode(["k": "a/b"]) == "data: {\"k\":\"a/b\"}\n\n")
            #expect(ServerSentEventEncoder.done == "data: [DONE]\n\n")
        }

        @Test
        func serviceErrorsDescribeTheProblem() {
            #expect(
                MLXOpenAIServiceError.responseNotFound("r1").errorDescription
                    == "Response 'r1' was not found")
            #expect(
                MLXOpenAIServiceError.invalidResponseFormatOutput("bad").errorDescription
                    == "Generated output did not satisfy response_format: bad")
            #expect(
                MLXOpenAIServiceError.embeddingsNotConfigured.errorDescription?.hasPrefix(
                    "Embeddings require") == true)
            #expect(
                MLXOpenAIServiceError.multipleToolCallsNotAllowed.errorDescription?.contains(
                    "parallel_tool_calls was false") == true)
        }
    }
}
