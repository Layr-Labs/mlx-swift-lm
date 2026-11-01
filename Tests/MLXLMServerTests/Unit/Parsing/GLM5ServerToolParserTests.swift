// Copyright © 2026 Eigen Labs.

import MLXLMCommon
import MLXLMServer
import Testing

extension UnitTests {
    @Suite
    struct GLM5ServerToolParserTests {
        @Test func exactGLM5TypesAndExplicitSelectorsResolve() throws {
            for modelType in ["glm5_next", "glm5_next_text", "glm5_next_mtp"] {
                #expect(try ServerToolParser.resolve(requested: nil, modelType: modelType) == .glm5)
                #expect(
                    try ServerToolParser.resolve(requested: "auto", modelType: modelType) == .glm5)
            }
            for name in ["glm5", "glm_5", "glm-5"] {
                #expect(try ServerToolParser.resolve(requested: name, modelType: nil) == .glm5)
            }
            #expect(
                try ServerToolParser.resolve(requested: "glm4", modelType: "glm5_next") == .glm4)
        }

        @Test func unknownGLMNamesDoNotGrantGLM5Parsing() throws {
            #expect(
                try ServerToolParser.resolve(requested: nil, modelType: "glm5_next_vision") == .json
            )
            #expect(throws: ServerToolParserError.unsupported("glm53")) {
                try ServerToolParser.resolve(requested: "glm53", modelType: "glm5_next")
            }
        }

        @Test func providerStreamHandlerCommitsOnlyAtEndOfResponse() {
            let tools: [[String: any Sendable]] = [
                [
                    "type": "function",
                    "function": ["name": "f"] as [String: any Sendable],
                ]
            ]
            let handler = BatchedToolStreamHandler(format: .glm5, tools: tools)
            #expect(handler.processChunk("<tool_call>f</tool_call>") == nil)
            #expect(handler.processChunk("<tool_call>f</tool_call>") == nil)
            #expect(handler.finish().map(\.function.name) == ["f", "f"])
            #expect(handler.takeResidualText() == nil)
            #expect(handler.parseFailureCount == 0)
        }

        @Test func providerStreamHandlerReturnsWholeMalformedBatchOnce() {
            let tools: [[String: any Sendable]] = [
                [
                    "type": "function",
                    "function": ["name": "f"] as [String: any Sendable],
                ]
            ]
            let handler = BatchedToolStreamHandler(format: .glm5, tools: tools)
            let raw = "<tool_call>f</tool_call><tool_call>f"
            #expect(handler.processChunk(raw) == nil)
            #expect(handler.finish().isEmpty)
            #expect(handler.takeResidualText() == raw)
            #expect(handler.takeResidualText() == nil)
            #expect(handler.parseFailureCount == 1)
        }
    }
}
