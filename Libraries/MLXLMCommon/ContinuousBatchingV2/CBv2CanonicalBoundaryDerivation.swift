// Copyright © 2026 Eigen Labs.

import Foundation

/// One conversation the boundary-rule derivation asks its caller to render
/// with the model's chat template, on the path the caller serves requests with.
public struct CBv2BoundaryProbe: Sendable {
    public var messages: [Message]
    public var tools: [ToolSpec]?
    /// False renders the conversation without the trailing generation prompt.
    public var addGenerationPrompt: Bool
    /// The template's thinking switch. Nil leaves the template's default.
    public var enableThinking: Bool?

    public init(
        messages: [Message], tools: [ToolSpec]? = nil, addGenerationPrompt: Bool,
        enableThinking: Bool? = nil
    ) {
        self.messages = messages
        self.tools = tools
        self.addGenerationPrompt = addGenerationPrompt
        self.enableThinking = enableThinking
    }
}

/// A model's message-boundary rule as its own chat template implies it, and
/// whether that rule held on rendered two-turn prompts.
public struct CBv2BoundaryRuleDerivation: Sendable, Equatable {
    public enum Scenario: String, Sendable, CaseIterable {
        case plain, noSystem, thinkingOff, replyHadReasoning, toolCall
    }

    public enum Verdict: String, Sendable, Equatable {
        case held
        /// The template refused the scenario, for example it has no tool support.
        case notRendered
        /// The first prompt's assistant-start boundary is not a boundary of
        /// the second prompt inside their common part.
        case openBoundaryNotShared
        /// No boundary ends the past reply with the whole history in front of
        /// it and nothing behind it that depends on the reply.
        case replyBoundaryMissing
    }

    /// The rule read from the template. Nil when the template's assistant
    /// messages share no opening tokens or a probe the rule needs did not render.
    public let candidate: CBv2CanonicalBoundaryRule?
    public let verdicts: [Scenario: Verdict]

    /// The candidate, when `plain` held and no rendered scenario failed.
    public var rule: CBv2CanonicalBoundaryRule? {
        guard verdicts[.plain] == .held,
            verdicts.values.allSatisfy({ $0 == .held || $0 == .notRendered })
        else { return nil }
        return candidate
    }
}

extension CBv2CanonicalBoundaryRule {
    /// Derive the rule from the model's own chat template, then check it on
    /// two-turn prompts the same template renders.
    ///
    /// `render` returns the prompt tokens of one probe. `isSpecial` says
    /// whether a token id is one of the tokenizer's special tokens; it places
    /// the close of a reply that ends in a tool call.
    ///
    /// The check is about reuse only. For any rule the scan gives two prompts
    /// the same chunks inside the tokens they share. A rule that fails the
    /// check would put chunk ends where a later turn cannot resume.
    public static func derive(
        isSpecial: (Int) -> Bool, render: (CBv2BoundaryProbe) throws -> [Int]
    ) -> CBv2BoundaryRuleDerivation {
        guard let rule = try? candidate(isSpecial: isSpecial, render: render) else {
            return CBv2BoundaryRuleDerivation(candidate: nil, verdicts: [:])
        }
        var verdicts: [CBv2BoundaryRuleDerivation.Scenario: CBv2BoundaryRuleDerivation.Verdict] =
            [:]
        for scenario in CBv2BoundaryRuleDerivation.Scenario.allCases {
            verdicts[scenario] = verdict(of: rule, in: scenario, render: render)
        }
        return CBv2BoundaryRuleDerivation(candidate: rule, verdicts: verdicts)
    }

    private static func candidate(
        isSpecial: (Int) -> Bool, render: (CBv2BoundaryProbe) throws -> [Int]
    ) throws -> CBv2CanonicalBoundaryRule? {
        let user = [message("user", ProbeText.question)]
        let reply = message("assistant", ProbeText.reply)
        let longerReply = message("assistant", ProbeText.longerReply)
        let followUp = message("user", ProbeText.followUp)

        // open: the tokens every assistant message starts with, as the
        // generation prompt and as a past message, under each thinking switch.
        func start(
            after: [Message], generationPrompt: Bool, thinking: Bool? = nil
        ) throws -> [Int] {
            let before = try render(
                CBv2BoundaryProbe(
                    messages: user, addGenerationPrompt: false, enableThinking: thinking))
            let whole = try render(
                CBv2BoundaryProbe(
                    messages: user + after, addGenerationPrompt: generationPrompt,
                    enableThinking: thinking))
            return Array(whole[commonPrefixCount(before, whole)...])
        }
        var starts = [
            try start(after: [], generationPrompt: true),
            try start(after: [reply, followUp], generationPrompt: true),
        ]
        starts += [
            try? start(after: [], generationPrompt: true, thinking: true),
            try? start(after: [], generationPrompt: true, thinking: false),
            try? start(after: [reply], generationPrompt: false),
            try? start(after: [reasoned(reply), followUp], generationPrompt: true),
        ].compactMap { $0 }
        let shared = starts.map { commonPrefixCount(starts[0], $0) }.min() ?? 0
        let open = Array(starts[0].prefix(shared))
        guard !open.isEmpty else { return nil }

        // close: what the template writes right after a past reply, when the
        // reply is the last message and when a user message follows it.
        func tail(
            _ one: [Message], _ other: [Message], generationPrompt: Bool,
            tools: [ToolSpec]? = nil
        ) throws -> [Int] {
            let first = try render(
                CBv2BoundaryProbe(
                    messages: one, tools: tools, addGenerationPrompt: generationPrompt))
            let second = try render(
                CBv2BoundaryProbe(
                    messages: other, tools: tools, addGenerationPrompt: generationPrompt))
            return Array(first.suffix(commonSuffixCount(first, second)))
        }
        let afterLastReply = try tail(user + [reply], user + [longerReply], generationPrompt: false)
        let fromReplyEnd = try tail(
            user + [reply, followUp], user + [longerReply, followUp], generationPrompt: true)
        let afterReply = Array(
            afterLastReply.prefix(commonPrefixCount(afterLastReply, fromReplyEnd)))
        // A template that writes nothing after a reply: the first token of the
        // next message closes it.
        let closesAtNextMessage = afterReply.isEmpty
        var close: [[Int]]
        if closesAtNextMessage {
            guard let next = fromReplyEnd.first else { return nil }
            close = [[next]]
        } else {
            close = [afterReply]
        }

        // A reply that ends in a tool call may end differently. A template
        // that renders no tool calls keeps the rule of its text replies.
        let asker = [message("user", ProbeText.weatherQuestion)]
        let result = toolResult(ProbeText.weatherResult)
        if let afterLastCall = try? tail(
            asker + [ProbeText.parisCall], asker + [ProbeText.osloCall],
            generationPrompt: false, tools: [ProbeText.weatherTool]),
            let fromCallEnd = try? tail(
                asker + [ProbeText.parisCall, result], asker + [ProbeText.osloCall, result],
                generationPrompt: true, tools: [ProbeText.weatherTool])
        {
            let afterCall = Array(
                afterLastCall.prefix(commonPrefixCount(afterLastCall, fromCallEnd)))
            if closesAtNextMessage {
                if afterCall.count < fromCallEnd.count {
                    let next = [fromCallEnd[afterCall.count]]
                    if !close.contains(next) { close.append(next) }
                }
            } else if !close.contains(where: { afterCall.suffix($0.count).elementsEqual($0) }),
                let cut = afterCall.lastIndex(where: isSpecial)
            {
                close.append(Array(afterCall[cut...]))
            }
        }
        return CBv2CanonicalBoundaryRule(open: open, close: close)
    }

    /// One two-turn conversation: the first prompt, the second prompt, the
    /// second prompt with another follow-up and with another reply.
    private static func verdict(
        of rule: CBv2CanonicalBoundaryRule, in scenario: CBv2BoundaryRuleDerivation.Scenario,
        render: (CBv2BoundaryProbe) throws -> [Int]
    ) -> CBv2BoundaryRuleDerivation.Verdict {
        var head = [message("system", ProbeText.system), message("user", ProbeText.question)]
        var reply = message("assistant", ProbeText.reply)
        var otherReply = message("assistant", ProbeText.longerReply)
        var followUp = message("user", ProbeText.followUp)
        var otherFollowUp = message("user", ProbeText.otherFollowUp)
        var tools: [ToolSpec]? = nil
        var thinking: Bool? = nil
        switch scenario {
        case .plain:
            break
        case .noSystem:
            head.removeFirst()
        case .thinkingOff:
            thinking = false
        case .replyHadReasoning:
            reply = reasoned(reply)
            otherReply = reasoned(otherReply)
        case .toolCall:
            head[1] = message("user", ProbeText.weatherQuestion)
            reply = ProbeText.parisCall
            otherReply = ProbeText.osloCall
            followUp = toolResult(ProbeText.weatherResult)
            otherFollowUp = toolResult(ProbeText.weatherFailure)
            tools = [ProbeText.weatherTool]
        }
        func prompt(_ messages: [Message]) throws -> [Int] {
            try render(
                CBv2BoundaryProbe(
                    messages: messages, tools: tools, addGenerationPrompt: true,
                    enableThinking: thinking))
        }
        guard let first = try? prompt(head),
            let second = try? prompt(head + [reply, followUp]),
            let withOtherFollowUp = try? prompt(head + [reply, otherFollowUp]),
            let withOtherReply = try? prompt(head + [otherReply, followUp])
        else { return .notRendered }

        let secondBoundaries = rule.boundaries(in: second)
        guard let openBoundary = rule.boundaries(in: first).last,
            openBoundary <= commonPrefixCount(first, second),
            secondBoundaries.contains(openBoundary)
        else { return .openBoundaryNotShared }
        let history = commonPrefixCount(second, withOtherFollowUp)
        guard let replyBoundary = secondBoundaries.last(where: { $0 <= history }),
            replyBoundary > openBoundary,
            second.count - replyBoundary <= commonSuffixCount(second, withOtherReply)
        else { return .replyBoundaryMissing }
        return .held
    }
}

/// The probe conversation. The reply has line breaks so that a close made of
/// ordinary text tokens fails the check instead of ending a real reply early.
private enum ProbeText {
    static let system = "You are a careful assistant. Answer briefly."
    static let question = "What is 17 times 23? Reply with the number and one short sentence."
    static let reply =
        "391.\nSeventeen times twenty-three is three hundred ninety-one.\n\nThat is the product."
    static let longerReply = reply + " Shall I show the long multiplication as well?"
    static let followUp = "Now divide that by 17 and tell me the result."
    static let otherFollowUp = "Please translate your last answer into French."
    static let reasoning = "Multiply 17 by 23."
    static let weatherQuestion = "What is the weather in Paris?"
    static let weatherResult = "{\"temp_c\": 18}"
    static let weatherFailure = "{\"error\": \"station offline\"}"
    static let parisCall = toolCall(["city": "Paris"])
    static let osloCall = toolCall(["city": "Oslo", "unit": "c"])
    static let weatherTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": [
                "type": "object",
                "properties": [
                    "city": ["type": "string"] as [String: any Sendable],
                    "unit": ["type": "string"] as [String: any Sendable],
                ] as [String: any Sendable],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
}

private func message(_ role: String, _ content: String) -> Message {
    ["role": role, "content": content]
}

private func reasoned(_ reply: Message) -> Message {
    var reply = reply
    reply["reasoning_content"] = ProbeText.reasoning
    return reply
}

private func toolCall(_ arguments: [String: any Sendable]) -> Message {
    let function: [String: any Sendable] = ["name": "get_weather", "arguments": arguments]
    let call: [String: any Sendable] = ["id": "call_1", "type": "function", "function": function]
    return ["role": "assistant", "content": "", "tool_calls": [call]]
}

private func toolResult(_ content: String) -> Message {
    ["role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": content]
}

private func commonPrefixCount(_ one: [Int], _ other: [Int]) -> Int {
    var count = 0
    while count < min(one.count, other.count), one[count] == other[count] { count += 1 }
    return count
}

private func commonSuffixCount(_ one: [Int], _ other: [Int]) -> Int {
    var count = 0
    while count < min(one.count, other.count),
        one[one.count - 1 - count] == other[other.count - 1 - count]
    {
        count += 1
    }
    return count
}
