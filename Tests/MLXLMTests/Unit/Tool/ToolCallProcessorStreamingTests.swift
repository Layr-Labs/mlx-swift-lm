import Foundation
import MLXLMCommon
import Testing

extension UnitTests {

    /// Tests of the streaming state machine of `ToolCallProcessor`. The
    /// expected text and calls follow the rules in `ToolCallProcessor.swift`:
    /// inline formats (no start tag) collect from the first `{` until the
    /// braces outside strings balance; tagged formats collect from the
    /// start tag to the end tag and give back the text around the call.
    @Suite
    struct ToolCallProcessorStreamingTests {

        private static let call = #"{"name": "f", "arguments": {"x": 1}}"#

        // MARK: - Inline format (Llama 3)

        @Test func inlineTextWithoutABraceIsPassedThrough() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk("plain text") == "plain text")
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.processEOS() == nil)
        }

        @Test func inlineCallInOneChunkKeepsTheLeadingText() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk("Calling: " + Self.call) == "Calling: ")
            let calls = processor.drainToolCalls()
            #expect(calls.count == 1)
            #expect(calls.first?.function.name == "f")
            #expect(calls.first?.function.arguments["x"] == .int(1))
            #expect(processor.drainToolCalls().isEmpty)
            #expect(processor.parseFailureCount == 0)
        }

        @Test func inlineCallOverSeveralChunksIsBuffered() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk(#"{"name": "f", "#) == nil)
            #expect(processor.processChunk(#""arguments": {"x": "#) == nil)
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.processChunk("1}}") == nil)
            #expect(processor.toolCalls.map(\.function.name) == ["f"])
            // After the call the processor passes text through again.
            #expect(processor.processChunk(" done") == " done")
        }

        @Test func inlineLeadingTextIsReturnedWhileTheCallIsBuffered() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk(#"ok {"name": "#) == "ok ")
            #expect(processor.processChunk(#""f", "arguments": {}}"#) == nil)
            #expect(processor.toolCalls.count == 1)
        }

        @Test func inlineBalancedObjectThatIsNotACallIsFlushed() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk(#"see {"a": 1}"#) == #"see {"a": 1}"#)
            #expect(processor.parseFailureCount == 1)
            #expect(processor.toolCalls.isEmpty)
        }

        @Test func inlineObjectThatIsNotACallIsFlushedWhenItCloses() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk(#"{"a": "#) == nil)
            #expect(processor.processChunk(#"{"b": 2}}"#) == #"{"a": {"b": 2}}"#)
            #expect(processor.parseFailureCount == 1)
            #expect(processor.processChunk("next") == "next")
        }

        /// A `}` inside a JSON string does not close the object.
        @Test func inlineBraceInsideAStringDoesNotCloseTheObject() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk(#"{"name": "f", "arguments": {"p": "a}b"#) == nil)
            #expect(processor.processChunk(#""}}"#) == nil)
            let calls = processor.drainToolCalls()
            #expect(calls.count == 1)
            #expect(calls.first?.function.arguments["p"] == .string("a}b"))
        }

        /// An escaped quote does not end the string, so the `}` after it is
        /// still inside the string.
        @Test func inlineEscapedQuoteKeepsTheStringOpen() {
            let processor = ToolCallProcessor(format: .llama3)
            let text = #"{"text": "a \" } b"}"#
            #expect(processor.processChunk(text) == text)
            #expect(processor.parseFailureCount == 1)
        }

        @Test func inlineEndOfSequenceReturnsAnIncompleteBuffer() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk(#"{"name": "f""#) == nil)
            #expect(processor.processEOS(returnBufferedText: true) == #"{"name": "f""#)
            #expect(processor.parseFailureCount == 1)
            #expect(processor.toolCalls.isEmpty)
            // The state is normal again.
            #expect(processor.processChunk("after") == "after")
        }

        @Test func inlineEndOfSequenceDropsTheBufferByDefault() {
            let processor = ToolCallProcessor(format: .llama3)
            #expect(processor.processChunk(#"{"name": "#) == nil)
            #expect(processor.processEOS() == nil)
            #expect(processor.parseFailureCount == 1)
        }

        // MARK: - Tagged format (JSON in <tool_call>)

        @Test func taggedTextWithoutAStartCharacterIsPassedThrough() {
            let processor = ToolCallProcessor(format: .json)
            #expect(processor.processChunk("hello") == "hello")
        }

        @Test func taggedFalseStartIsReturned() {
            let processor = ToolCallProcessor(format: .json)
            #expect(processor.processChunk("a < b") == "a < b")
            #expect(processor.processChunk("<b>bold</b>") == "<b>bold</b>")
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.parseFailureCount == 0)
        }

        @Test func taggedCallKeepsTheTextAroundIt() {
            let processor = ToolCallProcessor(format: .json)
            let text = "Hi <tool_call>" + Self.call + "</tool_call> bye"
            #expect(processor.processChunk(text) == "Hi  bye")
            #expect(processor.drainToolCalls().map(\.function.name) == ["f"])
        }

        @Test func taggedCallWithoutSurroundingTextReturnsNil() {
            let processor = ToolCallProcessor(format: .json)
            #expect(processor.processChunk("<tool_call>" + Self.call + "</tool_call>") == nil)
            #expect(processor.toolCalls.count == 1)
        }

        @Test func taggedCallsInOneChunkAreParsedInOrder() {
            let processor = ToolCallProcessor(format: .json)
            let first = #"<tool_call>{"name": "first", "arguments": {}}</tool_call>"#
            let second = #"<tool_call>{"name": "second", "arguments": {}}</tool_call>"#
            #expect(processor.processChunk(first + second) == nil)
            #expect(processor.drainToolCalls().map(\.function.name) == ["first", "second"])
        }

        @Test func taggedStartTagSplitOverChunks() {
            let processor = ToolCallProcessor(format: .json)
            #expect(processor.processChunk("<tool") == nil)
            #expect(processor.processChunk("_call>" + Self.call) == nil)
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.processChunk("</tool_call>") == nil)
            #expect(processor.toolCalls.map(\.function.name) == ["f"])
        }

        @Test func taggedMalformedCallIsReturnedAsText() {
            let processor = ToolCallProcessor(format: .json)
            #expect(
                processor.processChunk("<tool_call>not json</tool_call>tail")
                    == "<tool_call>not json</tool_call>tail")
            #expect(processor.parseFailureCount == 1)
            #expect(processor.toolCalls.isEmpty)
        }

        @Test func taggedMalformedCallBeforeAGoodCall() {
            let processor = ToolCallProcessor(format: .json)
            let bad = "<tool_call>bad</tool_call>"
            let good = "<tool_call>" + Self.call + "</tool_call>"
            #expect(processor.processChunk("x" + bad + good) == "x" + bad)
            #expect(processor.parseFailureCount == 1)
            #expect(processor.toolCalls.map(\.function.name) == ["f"])
        }

        @Test func taggedCallToAnUndeclaredToolIsReturnedAsText() {
            let tools: [[String: any Sendable]] = [
                ["type": "function", "function": ["name": "g"] as [String: any Sendable]]
            ]
            let processor = ToolCallProcessor(format: .json, tools: tools)
            let text = "<tool_call>" + Self.call + "</tool_call>"
            #expect(processor.processChunk(text) == text)
            #expect(processor.parseFailureCount == 1)
            #expect(processor.toolCalls.isEmpty)
        }

        @Test func taggedEndOfSequenceParsesAnOpenCall() {
            let processor = ToolCallProcessor(format: .json)
            #expect(processor.processChunk("<tool_call>" + Self.call) == nil)
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.processEOS(returnBufferedText: true) == nil)
            #expect(processor.drainToolCalls().map(\.function.name) == ["f"])
            #expect(processor.parseFailureCount == 0)
        }

        @Test func taggedEndOfSequenceReturnsAnOpenMalformedCall() {
            let processor = ToolCallProcessor(format: .json)
            #expect(processor.processChunk("<tool_call>oops") == nil)
            #expect(processor.processEOS(returnBufferedText: true) == "<tool_call>oops")
            #expect(processor.parseFailureCount == 1)
            #expect(processor.processChunk("again") == "again")
        }

        @Test func taggedEndOfSequenceWithAPartialStartTag() {
            let processor = ToolCallProcessor(format: .json)
            #expect(processor.processChunk("<tool") == nil)
            #expect(processor.processEOS(returnBufferedText: true) == "<tool")
            #expect(processor.parseFailureCount == 1)
        }

        @Test func endOfSequenceInTheNormalStateDoesNothing() {
            let processor = ToolCallProcessor(format: .json)
            #expect(processor.processEOS(returnBufferedText: true) == nil)
            #expect(processor.parseFailureCount == 0)
        }

        /// `strictGemma` applies only to the Gemma format.
        @Test func strictGemmaIsIgnoredForOtherFormats() {
            let processor = ToolCallProcessor(format: .json, strictGemma: true)
            #expect(processor.processChunk("<tool_call>" + Self.call + "</tool_call>") == nil)
            #expect(processor.toolCalls.count == 1)
        }
    }
}
