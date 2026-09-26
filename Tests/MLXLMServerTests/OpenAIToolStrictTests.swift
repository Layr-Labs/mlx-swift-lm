// Copyright © 2026 Eigen Labs Inc.

import Foundation
@testable import MLXLMServer
import Testing

struct OpenAIToolStrictTests {
    @Test(arguments: [true, false])
    func nestedFunctionStrictRoundTripsAndReachesTemplate(_ strict: Bool) throws {
        let data = Data(
            """
            {"type":"function","function":{"name":"lookup_record","strict":\(strict),"parameters":{"type":"object"}}}
            """.utf8)
        let tool = try JSONDecoder().decode(OpenAITool.self, from: data)

        #expect(tool.function.strict == strict)
        let function = try #require(tool.toolSpec()["function"] as? [String: any Sendable])
        #expect(function["strict"] as? Bool == strict)
        #expect(try JSONDecoder().decode(OpenAITool.self, from: JSONEncoder().encode(tool)) == tool)
    }

    @Test(arguments: [true, false])
    func flatFunctionStrictReachesTemplate(_ strict: Bool) throws {
        let data = Data(
            """
            {"type":"function","name":"lookup_record","strict":\(strict),"parameters":{"type":"object"}}
            """.utf8)
        let tool = try JSONDecoder().decode(OpenAITool.self, from: data)

        #expect(tool.function.strict == strict)
        let function = try #require(tool.toolSpec()["function"] as? [String: any Sendable])
        #expect(function["strict"] as? Bool == strict)
    }

    @Test func omittedStrictStaysOmitted() throws {
        let data = Data(
            #"{"type":"function","function":{"name":"lookup_record","parameters":{"type":"object"}}}"#.utf8)
        let tool = try JSONDecoder().decode(OpenAITool.self, from: data)

        #expect(tool.function.strict == nil)
        let function = try #require(tool.toolSpec()["function"] as? [String: any Sendable])
        #expect(function["strict"] == nil)
        let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(tool)) as? [String: Any]
        #expect((encoded?["function"] as? [String: Any])?["strict"] == nil)
    }
}
