import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct ToolCallExecuteTests {

        private struct Input: Codable, Equatable {
            let city: String
            let days: Int
        }

        private struct Output: Codable, Equatable {
            let summary: String
        }

        private let tool = Tool<Input, Output>(
            name: "forecast",
            description: "Gives a forecast.",
            parameters: [
                .required("city", type: .string, description: "The city."),
                .required("days", type: .int, description: "The number of days."),
            ]
        ) { input in
            Output(summary: "\(input.city):\(input.days)")
        }

        @Test
        func executeDecodesTheArgumentsAndCallsTheHandler() async throws {
            let call = ToolCall(
                function: .init(
                    name: "forecast",
                    arguments: ["city": JSONValue.string("Oslo"), "days": JSONValue.int(3)]))
            let output = try await call.execute(with: tool)
            #expect(output == Output(summary: "Oslo:3"))
        }

        @Test
        func executeRejectsAnotherFunctionName() async throws {
            let call = ToolCall(function: .init(name: "weather", arguments: [String: JSONValue]()))
            let error = await #expect(throws: ToolError.self) {
                try await call.execute(with: tool)
            }
            guard case .nameMismatch(let toolName, let functionName) = error else {
                Issue.record("Expected ToolError.nameMismatch, got \(String(describing: error))")
                return
            }
            #expect(toolName == "forecast")
            #expect(functionName == "weather")
            #expect(
                error?.errorDescription
                    == "Tool name mismatch: expected 'forecast' but got 'weather'")
        }

        @Test
        func executeFailsWhenAnArgumentHasTheWrongType() async {
            let call = ToolCall(
                function: .init(
                    name: "forecast",
                    arguments: [
                        "city": JSONValue.string("Oslo"), "days": JSONValue.string("three"),
                    ]
                ))
            await #expect(throws: DecodingError.self) {
                try await call.execute(with: tool)
            }
        }

        @Test
        func functionInitConvertsSendableArguments() {
            let function = ToolCall.Function(
                name: "f", arguments: ["n": 2, "s": "x"] as [String: any Sendable])
            #expect(function.arguments == ["n": .int(2), "s": .string("x")])
        }
    }
}
