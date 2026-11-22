// Copyright © 2026 Eigen Labs.

import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// `CBv2CanonicalBoundaryRule.derive(isSpecial:render:)`: the rule read
    /// from a chat template and the check that it holds on two-turn prompts.
    /// The templates are toy renderers over a toy tokenizer; no model, no arrays.
    @Suite
    struct CBv2CanonicalBoundaryDerivationTests {

        private typealias Rule = CBv2CanonicalBoundaryRule
        private typealias Scenario = CBv2BoundaryRuleDerivation.Scenario
        private typealias Verdict = CBv2BoundaryRuleDerivation.Verdict

        private static let allHeld = Dictionary(
            uniqueKeysWithValues: Scenario.allCases.map { ($0, Verdict.held) })

        private func derive(
            _ render: (CBv2BoundaryProbe) throws -> [Int]
        ) -> CBv2BoundaryRuleDerivation {
            Rule.derive(isSpecial: ToyChat.isSpecial, render: render)
        }

        @Test func aChatMLTemplateGivesItsAssistantHeaderAndEndOfMessage() {
            let derivation = derive(ToyChat.ChatML.render)
            #expect(
                derivation.rule == Rule(open: ToyChat.ChatML.open, close: [ToyChat.ChatML.close]))
            #expect(derivation.verdicts == Self.allHeld)
        }

        /// The open stops before the tokens the thinking switch changes.
        @Test func theOpenHoldsUnderEitherThinkingSwitch() throws {
            let rule = try #require(derive(ToyChat.ChatML.render).rule)
            for thinking in [true, false, nil] as [Bool?] {
                let prompt = ToyChat.ChatML.render(
                    CBv2BoundaryProbe(
                        messages: [["role": "user", "content": "Hello there"]],
                        addGenerationPrompt: true, enableThinking: thinking))
                let afterOpen = ToyChat.ChatML.afterOpen(thinking: thinking).count
                #expect(rule.boundaries(in: prompt) == [prompt.count - afterOpen])
            }
        }

        @Test func aToolCallThatEndsInItsOwnTokenAddsAClose() {
            let derivation = derive(ToyChat.Turns.render)
            #expect(
                derivation.rule
                    == Rule(
                        open: ToyChat.Turns.open,
                        close: [ToyChat.Turns.close, [ToyChat.Turns.callEnd]]))
            #expect(derivation.verdicts == Self.allHeld)
        }

        @Test func aTemplateThatWritesNothingAfterAReplyClosesAtTheNextMessage() {
            let derivation = derive(ToyChat.RoleTokens.render)
            #expect(
                derivation.rule
                    == Rule(
                        open: ToyChat.RoleTokens.open,
                        close: [[ToyChat.RoleTokens.user], [ToyChat.RoleTokens.observation]]))
            #expect(derivation.verdicts == Self.allHeld)
        }

        @Test func aTemplateWithoutToolsKeepsTheRuleOfItsTextReplies() {
            let derivation = derive { probe in
                if probe.tools != nil { throw ToyChat.Refusal.tools }
                return ToyChat.ChatML.render(probe)
            }
            #expect(
                derivation.rule == Rule(open: ToyChat.ChatML.open, close: [ToyChat.ChatML.close]))
            #expect(derivation.verdicts[.toolCall] == .notRendered)
            #expect(derivation.verdicts.filter { $0.value == .held }.count == 4)
        }

        @Test func aTemplateThatCannotBeProbedGivesNoRule() {
            let alwaysPrompts = derive { probe in
                var probe = probe
                probe.addGenerationPrompt = true
                return ToyChat.ChatML.render(probe)
            }
            #expect(alwaysPrompts.candidate == nil && alwaysPrompts.rule == nil)
            #expect(alwaysPrompts.verdicts.isEmpty)

            let refuses = derive { _ in throw ToyChat.Refusal.everything }
            #expect(refuses.candidate == nil && refuses.rule == nil)
            #expect(refuses.verdicts.isEmpty)
        }

        /// A close made of ordinary text also ends inside a reply.
        @Test func aCloseThatAlsoOccursInsideAReplyIsRejected() {
            let derivation = derive(ToyChat.Labels.render)
            #expect(
                derivation.candidate
                    == Rule(open: ToyChat.Labels.open, close: [[ToyChat.newline]]))
            #expect(derivation.verdicts[.plain] == .replyBoundaryMissing)
            #expect(derivation.rule == nil)
        }

        @Test func anOpenThatDependsOnEarlierMessagesIsRejected() {
            let derivation = derive(ToyChat.chatMLWithSystemDependentPrompt)
            #expect(derivation.candidate == derive(ToyChat.ChatML.render).candidate)
            #expect(derivation.verdicts[.plain] == .openBoundaryNotShared)
            #expect(derivation.verdicts[.noSystem] == .held)
            #expect(derivation.rule == nil)
        }

        @Test func onlyAHeldPlainScenarioAcceptsTheCandidate() {
            let candidate = Rule(open: [1], close: [[2]])
            func rule(_ verdicts: [Scenario: Verdict]) -> Rule? {
                CBv2BoundaryRuleDerivation(candidate: candidate, verdicts: verdicts).rule
            }
            #expect(rule(Self.allHeld) == candidate)
            #expect(rule([.plain: .held, .toolCall: .notRendered]) == candidate)
            #expect(rule([.plain: .notRendered, .noSystem: .held]) == nil)
            #expect(rule([.plain: .held, .thinkingOff: .replyBoundaryMissing]) == nil)
            #expect(rule([:]) == nil)
        }
    }
}

/// A toy tokenizer and toy chat templates: enough structure to stand in for a
/// model's own template. Ids below 100 are special tokens.
private enum ToyChat {
    enum Refusal: Error { case tools, everything }

    static let newline = 100
    static let paragraph = 101

    static func isSpecial(_ id: Int) -> Bool { id < 100 }

    /// Words, "\n" and "\n\n" are one token each.
    static func text(_ string: String) -> [Int] {
        var ids: [Int] = []
        var word = ""
        var breaks = 0
        func flush() {
            if !word.isEmpty {
                var hash: UInt32 = 2_166_136_261
                for byte in word.utf8 { hash = (hash ^ UInt32(byte)) &* 16_777_619 }
                ids.append(1000 + Int(hash % 900_000))
                word = ""
            }
            ids += [Int](repeating: paragraph, count: breaks / 2)
            if breaks % 2 == 1 { ids.append(newline) }
            breaks = 0
        }
        for character in string {
            if character == "\n" {
                if !word.isEmpty { flush() }
                breaks += 1
            } else if character == " " {
                flush()
            } else {
                if breaks > 0 { flush() }
                word.append(character)
            }
        }
        flush()
        return ids
    }

    private static func role(_ message: Message) -> String { message["role"] as? String ?? "" }
    private static func content(_ message: Message) -> String {
        message["content"] as? String ?? ""
    }

    /// Each tool call of an assistant message as text, arguments in key order.
    private static func calls(_ message: Message) -> [String] {
        let calls = message["tool_calls"] as? [[String: any Sendable]] ?? []
        return calls.map { call in
            let function = call["function"] as? [String: any Sendable] ?? [:]
            let arguments = function["arguments"] as? [String: any Sendable] ?? [:]
            let pairs = arguments.keys.sorted().map { "\($0)=\(arguments[$0] ?? "")" }
            return "\(function["name"] ?? "") { \(pairs.joined(separator: " ")) }"
        }
    }

    /// ChatML with a thinking block: `<|im_start|>assistant\n<think>` opens
    /// every assistant message and `<|im_end|>\n` closes it.
    enum ChatML {
        static let imStart = 1, imEnd = 2, think = 3, thinkEnd = 4
        static let call = 5, callEnd = 6, response = 7, responseEnd = 8
        static let open = [imStart] + text("assistant\n") + [think]
        static let close = [imEnd, newline]

        static func afterOpen(thinking: Bool?) -> [Int] {
            thinking == false ? [paragraph, thinkEnd, paragraph] : [newline]
        }

        static func render(_ probe: CBv2BoundaryProbe) -> [Int] {
            var ids: [Int] = []
            if probe.tools != nil {
                ids += [imStart] + text("system\n# Tools get_weather") + close
            }
            for message in probe.messages {
                switch role(message) {
                case "assistant":
                    ids += open
                    if let reasoning = message["reasoning_content"] as? String {
                        ids += [newline] + text(reasoning) + [newline]
                    } else {
                        ids += [paragraph]
                    }
                    ids += [thinkEnd, paragraph] + text(content(message))
                    for callText in calls(message) {
                        ids += [newline, call, newline] + text(callText) + [newline, callEnd]
                    }
                    ids += close
                case "tool":
                    ids +=
                        [imStart] + text("user\n") + [response, newline]
                        + text(content(message)) + [newline, responseEnd] + close
                default:
                    ids += [imStart] + text(role(message) + "\n" + content(message)) + close
                }
            }
            if probe.addGenerationPrompt {
                ids += open + afterOpen(thinking: probe.enableThinking)
            }
            return ids
        }
    }

    /// Turn markers. A tool call ends in a special token and its response
    /// stays inside the model turn, so no generation prompt follows it.
    enum Turns {
        static let bos = 10, startOfTurn = 11, endOfTurn = 12
        static let call = 13, callEnd = 14, response = 15, responseEnd = 16
        static let open = [startOfTurn] + text("model\n")
        static let close = [endOfTurn, newline]

        static func render(_ probe: CBv2BoundaryProbe) -> [Int] {
            var ids = [bos]
            var modelTurnIsOpen = false
            for message in probe.messages {
                switch role(message) {
                case "assistant":
                    ids += open + text(content(message))
                    let toolCalls = calls(message)
                    for callText in toolCalls { ids += [call] + text(callText) + [callEnd] }
                    modelTurnIsOpen = !toolCalls.isEmpty
                    if toolCalls.isEmpty { ids += close }
                case "tool":
                    ids += [response] + text(content(message)) + [responseEnd]
                default:
                    if modelTurnIsOpen { ids += close }
                    modelTurnIsOpen = false
                    ids += [startOfTurn] + text(role(message) + "\n" + content(message)) + close
                }
            }
            if probe.addGenerationPrompt, !modelTurnIsOpen { ids += open }
            return ids
        }
    }

    /// One role token per message and nothing after a reply: the next
    /// message's role token is what ends it.
    enum RoleTokens {
        static let system = 20, user = 21, assistant = 22, observation = 23
        static let open = [assistant, newline]

        static func render(_ probe: CBv2BoundaryProbe) -> [Int] {
            var ids: [Int] = []
            for message in probe.messages {
                switch role(message) {
                case "system": ids += [system, newline]
                case "assistant": ids += open
                case "tool": ids += [observation, newline]
                default: ids += [user, newline]
                }
                ids += text(content(message)) + calls(message).flatMap(text)
            }
            if probe.addGenerationPrompt { ids += open }
            return ids
        }
    }

    /// Plain text labels: a reply ends at a line break, which a reply also contains.
    enum Labels {
        static let open = text("Assistant:")

        static func render(_ probe: CBv2BoundaryProbe) -> [Int] {
            var ids: [Int] = []
            for message in probe.messages {
                ids +=
                    text(role(message).capitalized + ":") + text(content(message))
                    + calls(message).flatMap(text) + [newline]
            }
            if probe.addGenerationPrompt { ids += open }
            return ids
        }
    }

    /// ChatML whose generation prompt names another role once a system
    /// message is present.
    static func chatMLWithSystemDependentPrompt(_ probe: CBv2BoundaryProbe) -> [Int] {
        var ids = ChatML.render(probe)
        if probe.addGenerationPrompt, probe.messages.first.map(role) == "system" {
            ids.removeLast(
                ChatML.open.count + ChatML.afterOpen(thinking: probe.enableThinking).count)
            ids += [ChatML.imStart] + text("model\n")
        }
        return ids
    }
}
