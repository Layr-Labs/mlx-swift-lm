// Copyright © 2026 Eigen Labs.

import MLXLMCommon

/// Token-loop handler for a framed response protocol that keeps reasoning
/// and content apart. `Generation` has no reasoning event, so the container
/// engine reads the protocol decoder itself and reports each channel as
/// `.parsed`, which the service does not parse again.
struct ChannelTokenLoopHandler: TokenLoopHandler {
    typealias Output = MLXServerGenerationEvent

    private var decoder: any TokenStreamDecoder

    /// - Returns: `nil` when `format` has no token protocol, or the tokenizer
    ///   lacks its control tokens.
    init?(
        format: ToolCallFormat, tokenizer: any Tokenizer, stopStrings: Set<String>,
        tools: [[String: any Sendable]]?, toolCallPolicy: ToolCallPolicy
    ) {
        guard
            let decoder = format.makeProtocolTokenStreamDecoder(
                tokenizer: tokenizer, tools: tools, stopStrings: stopStrings,
                toolCallPolicy: toolCallPolicy)
        else { return nil }
        self.decoder = decoder
    }

    var additionalStopTokenIDs: Set<Int> { decoder.additionalStopTokenIDs }
    var receivesStopTokens: Bool { decoder.receivesStopTokens }

    mutating func onToken(
        _ token: Int, logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending MLXServerGenerationEvent) -> Bool
    ) -> TokenLoopDisposition {
        push(token, emit: emit)
    }

    mutating func onStopToken(
        _ token: Int, logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending MLXServerGenerationEvent) -> Bool
    ) -> TokenLoopDisposition {
        guard decoder.receivesStopTokens else { return .more }
        return push(token, emit: emit)
    }

    mutating func onGenerationEnd(
        emit: (sending MLXServerGenerationEvent) -> Bool
    ) -> TokenLoopDisposition {
        var decoder = self.decoder
        var disposition = TokenLoopDisposition.more
        _ = decoder.finish { event in
            disposition = Self.forward(event, emit: emit)
            return disposition.isMore
        }
        self.decoder = decoder
        return disposition
    }

    func infoEvent(_ info: GenerateCompletionInfo) -> MLXServerGenerationEvent {
        .info(.init(info))
    }

    private mutating func push(
        _ token: Int, emit: (sending MLXServerGenerationEvent) -> Bool
    ) -> TokenLoopDisposition {
        var decoder = self.decoder
        var disposition = TokenLoopDisposition.more
        let completed = decoder.push(token) { event in
            disposition = Self.forward(event, emit: emit)
            return disposition.isMore
        }
        self.decoder = decoder
        if disposition.isMore { return completed ? .more : .cancelled }
        return disposition
    }

    private static func forward(
        _ event: TokenStreamEvent, emit: (sending MLXServerGenerationEvent) -> Bool
    ) -> TokenLoopDisposition {
        let output: MLXServerGenerationEvent
        switch event {
        case .reasoning(let text):
            output = .parsed(.init(content: "", reasoningContent: text))
        case .response(let text):
            output = .parsed(.init(content: text, reasoningContent: nil))
        case .toolCall(let call):
            output = .toolCall(call)
        case .rejectedToolCall, .protocolError:
            // The OpenAI wire has no field for either. Their text is never content.
            return .more
        case .stop:
            return .stop
        }
        return emit(output) ? .more : .cancelled
    }
}

extension TokenLoopDisposition {
    fileprivate var isMore: Bool {
        if case .more = self { return true }
        return false
    }
}
