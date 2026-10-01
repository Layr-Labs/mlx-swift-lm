import Foundation
import MLXLMCommon
import Testing

extension UnitTests {

    /// Tests of the stop-sequence state machine. The expected values come
    /// from the trie rules in `SequenceStateMachine.swift`: a token extends
    /// the pending match, a mismatch drops tokens from the start of the
    /// candidate until a prefix matches, and a terminal node moves to the
    /// next state or stops the row.
    @Suite
    struct SequenceStateMachineTests {

        /// Feeds `tokens` to the machine and returns the result of each step.
        private func run(
            _ machine: SequenceStateMachine, _ tokens: [Int]
        ) -> (
            state: SequenceStateMachineState,
            steps: [(matched: [Int]?, current: String?)]
        ) {
            var state = machine.makeState()
            var steps: [(matched: [Int]?, current: String?)] = []
            for token in tokens {
                let result = machine.match(state, token)
                state = result.next
                steps.append((matched: result.matchedSequence, current: result.currentState))
            }
            return (state, steps)
        }

        @Test func emptyMachineNeverMatches() {
            let machine = SequenceStateMachine()
            #expect(machine.states.isEmpty)
            #expect(machine.initial == "normal")

            let state = machine.makeState()
            #expect(state.currentState == nil)
            #expect(state.trieNode == nil)
            #expect(state.pendingMatch.isEmpty)

            let result = machine.match(state, 7)
            #expect(result.matchedSequence == nil)
            #expect(result.currentState == nil)
            #expect(result.next.pendingMatch.isEmpty)
        }

        @Test func initBuildsOneTriePerState() throws {
            let machine = SequenceStateMachine(states: [
                "normal": [(sequence: [1, 2, 3], next: nil), (sequence: [4], next: "think")],
                "think": [(sequence: [5, 6], next: "normal")],
            ])
            #expect(Set(machine.states.keys) == ["normal", "think"])

            let normal = try #require(machine.states["normal"])
            #expect(normal.transition == nil)
            #expect(Set(normal.children.keys) == [1, 4])

            let one = try #require(normal.children[1])
            #expect(one.transition == nil)
            let two = try #require(one.children[2])
            let three = try #require(two.children[3])
            #expect(three.transition?.matchedSequence == [1, 2, 3])
            #expect(three.transition?.next == nil)

            let four = try #require(normal.children[4])
            #expect(four.transition?.matchedSequence == [4])
            #expect(four.transition?.next == "think")

            let state = machine.makeState()
            #expect(state.currentState == "normal")
            #expect(state.trieNode?.children.count == 2)
            #expect(Set(state.allStates.keys) == ["normal", "think"])
        }

        @Test func multiTokenSequenceTerminatesTheRow() {
            let machine = SequenceStateMachine(states: ["normal": [(sequence: [1, 2, 3], next: nil)]])
            let (state, steps) = run(machine, [1, 2, 3])

            #expect(steps[0].matched == nil)
            #expect(steps[0].current == "normal")
            #expect(steps[1].matched == nil)
            #expect(steps[1].current == "normal")
            #expect(steps[2].matched == [1, 2, 3])
            #expect(steps[2].current == nil)

            #expect(state.currentState == nil)
            #expect(state.trieNode == nil)
            #expect(state.pendingMatch.isEmpty)
        }

        @Test func pendingMatchGrowsWhileThePrefixMatches() {
            let machine = SequenceStateMachine(states: ["normal": [(sequence: [1, 2, 3], next: nil)]])
            var state = machine.makeState()
            state = machine.match(state, 1).next
            #expect(state.pendingMatch == [1])
            state = machine.match(state, 2).next
            #expect(state.pendingMatch == [1, 2])
            #expect(state.trieNode?.children.keys.first == 3)
        }

        @Test func aTerminatedRowStaysTerminated() {
            let machine = SequenceStateMachine(states: ["normal": [(sequence: [9], next: nil)]])
            let (state, steps) = run(machine, [9, 9, 1])
            #expect(steps.map { $0.matched } == [[9], nil, nil])
            #expect(steps.map { $0.current } == [nil, nil, nil])
            #expect(state.currentState == nil)
        }

        @Test func mismatchResetsToTheRoot() {
            let machine = SequenceStateMachine(states: ["normal": [(sequence: [1, 2, 3], next: nil)]])
            let (state, steps) = run(machine, [1, 2, 9])
            #expect(steps.allSatisfy { $0.matched == nil })
            #expect(steps.allSatisfy { $0.current == "normal" })
            #expect(state.pendingMatch.isEmpty)
            #expect(state.trieNode?.children.keys.first == 1)

            // The machine still matches after the reset.
            var next = state
            for token in [1, 2] {
                next = machine.match(next, token).next
            }
            #expect(machine.match(next, 3).matchedSequence == [1, 2, 3])
        }

        /// The sequence [1, 1, 2] in the stream 1, 1, 1, 2: the third 1
        /// breaks [1, 1, 1], the machine drops the first token and keeps
        /// [1, 1] as the pending match, and the 2 completes the sequence.
        @Test func mismatchKeepsTheLongestMatchingSuffix() {
            let machine = SequenceStateMachine(states: ["normal": [(sequence: [1, 1, 2], next: nil)]])
            var state = machine.makeState()
            state = machine.match(state, 1).next
            state = machine.match(state, 1).next
            state = machine.match(state, 1).next
            #expect(state.pendingMatch == [1, 1])
            let last = machine.match(state, 2)
            #expect(last.matchedSequence == [1, 1, 2])
            #expect(last.currentState == nil)
        }

        /// A mismatch whose last token starts the sequence again keeps that
        /// token as the pending match.
        @Test func mismatchKeepsTheNewStartToken() {
            let machine = SequenceStateMachine(states: ["normal": [(sequence: [1, 2], next: nil)]])
            var state = machine.makeState()
            state = machine.match(state, 1).next
            state = machine.match(state, 1).next
            #expect(state.pendingMatch == [1])
            #expect(machine.match(state, 2).matchedSequence == [1, 2])
        }

        @Test func transitionsMoveBetweenStates() {
            let machine = SequenceStateMachine(states: [
                "normal": [(sequence: [1, 2, 3], next: nil), (sequence: [4], next: "think")],
                "think": [(sequence: [5, 6], next: "normal")],
            ])
            // 4 enters "think". In "think" the sequence [1, 2, 3] is not
            // active, so it does not stop the row. 5, 6 go back to "normal",
            // where 1, 2, 3 stops the row.
            let (state, steps) = run(machine, [4, 1, 2, 3, 5, 6, 1, 2, 3])
            #expect(
                steps.map { $0.current } == [
                    "think", "think", "think", "think", "think", "normal", "normal", "normal", nil,
                ])
            #expect(steps[0].matched == [4])
            #expect(steps[5].matched == [5, 6])
            #expect(steps[8].matched == [1, 2, 3])
            #expect(steps.filter { $0.matched != nil }.count == 3)
            #expect(state.currentState == nil)
        }

        @Test func transitionStateUsesTheTargetTrie() {
            let machine = SequenceStateMachine(states: [
                "normal": [(sequence: [4], next: "think")],
                "think": [(sequence: [5, 6], next: "normal")],
            ])
            let result = machine.match(machine.makeState(), 4)
            #expect(result.next.currentState == "think")
            #expect(result.next.trieNode?.children.keys.first == 5)
            #expect(result.next.pendingMatch.isEmpty)
        }

        /// When one sequence is a prefix of another in the same state, the
        /// node of the shorter sequence is terminal, so the shorter one
        /// matches first.
        @Test func aShorterSequenceThatIsAPrefixMatchesFirst() {
            let machine = SequenceStateMachine(states: [
                "normal": [(sequence: [7, 8], next: nil), (sequence: [7], next: "other")],
                "other": [],
            ])
            let result = machine.match(machine.makeState(), 7)
            #expect(result.matchedSequence == [7])
            #expect(result.currentState == "other")
        }

        @Test func customInitialState() {
            let machine = SequenceStateMachine(
                states: ["start": [(sequence: [2], next: nil)]], initial: "start")
            #expect(machine.initial == "start")
            let state = machine.makeState()
            #expect(state.currentState == "start")
            #expect(machine.match(state, 2).matchedSequence == [2])
        }

        /// An initial state without a trie gives a state that never matches.
        @Test func unknownInitialStateNeverMatches() {
            let machine = SequenceStateMachine(
                states: ["normal": [(sequence: [2], next: nil)]], initial: "missing")
            let state = machine.makeState()
            #expect(state.currentState == "missing")
            #expect(state.trieNode == nil)
            let result = machine.match(state, 2)
            #expect(result.matchedSequence == nil)
            #expect(result.currentState == "missing")
        }

        /// A state whose `allStates` lacks the current state does not match.
        @Test func stateWithoutItsTrieDoesNotMatch() {
            let machine = SequenceStateMachine(states: ["normal": [(sequence: [2], next: nil)]])
            let state = SequenceStateMachineState(
                currentState: "normal", trieNode: StateMachineTrieNode(), allStates: [:])
            #expect(state.pendingMatch.isEmpty)
            let result = machine.match(state, 2)
            #expect(result.matchedSequence == nil)
            #expect(result.currentState == "normal")
        }
    }
}
