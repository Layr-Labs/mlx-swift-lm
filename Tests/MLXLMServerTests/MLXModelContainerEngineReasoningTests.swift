// Copyright © 2026 Eigen Labs Inc.

import Testing

@testable import MLXLMServer

struct MLXModelContainerEngineReasoningTests {
    private func request(_ reasoning: OpenAIReasoningConfig? = nil) -> OpenAIChatCompletionRequest {
        .init(model: "fixture", messages: [.init(role: .user, content: .text("hello"))],
              reasoningParser: .qwen3, reasoning: reasoning)
    }

    @Test func absentEmptyAndParserOnlyReasoningRemainSupported() throws {
        try MLXModelContainerEngine.validateReasoningControls(request())
        try MLXModelContainerEngine.validateReasoningControls(request(.init()))
    }

    @Test(arguments: ["none", "low", "medium", "high", "", "unknown"])
    func explicitEffortCannotSucceedWithoutBeingApplied(_ effort: String) {
        #expect(throws: MLXModelContainerEngineError.unsupportedReasoningControl("reasoning.effort")) {
            try MLXModelContainerEngine.validateReasoningControls(request(.init(effort: effort)))
        }
    }

    @Test(arguments: [false, true])
    func explicitEnabledCannotSucceedWithoutBeingApplied(_ enabled: Bool) {
        #expect(throws: MLXModelContainerEngineError.unsupportedReasoningControl("reasoning.enabled")) {
            try MLXModelContainerEngine.validateReasoningControls(request(.init(enabled: enabled)))
        }
    }

    @Test func responsesEffortReachesTheSameEngineGuard() {
        let response = OpenAIResponseRequest(model: "fixture", input: .text("hello"),
            reasoning: .init(effort: "none"))
        #expect(throws: MLXModelContainerEngineError.unsupportedReasoningControl("reasoning.effort")) {
            try MLXModelContainerEngine.validateReasoningControls(response.chatCompletionRequest)
        }
        #expect(MLXModelContainerEngineError.unsupportedReasoningControl("reasoning.effort").status.code == 400)
    }
}
