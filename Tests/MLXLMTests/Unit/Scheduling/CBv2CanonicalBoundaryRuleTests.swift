// Copyright © 2026 Eigen Labs.

import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// `CBv2CanonicalBoundaryRule.boundaries(in:)` and `CBv2CanonicalPartition`:
    /// where the canonical cells of a prompt end. A boundary is decided by the
    /// tokens before it, so two prompts share every boundary and every cell
    /// inside the tokens they share. Plain integers; no arrays.
    @Suite
    struct CBv2CanonicalBoundaryRuleTests {

        private typealias Rule = CBv2CanonicalBoundaryRule

        /// Rules read from the chat templates of served model families.
        private static let familyRules: [(family: String, rule: Rule)] = [
            ("Qwen3.8 27B", Rule(open: [248045, 74455, 198, 248068], close: [[248046, 198]])),
            ("Bonsai 2 9B", Rule(open: [248045, 74455, 198], close: [[248046, 198]])),
            ("Gemma 4", Rule(open: [105, 4368, 107], close: [[106, 107], [50]])),
            ("MiMo V2.6 Flash", Rule(open: [151644, 77091, 198], close: [[151645]])),
            ("GLM-5.3 Flash", Rule(open: [154828, 154841], close: [[154827], [154829]])),
        ]

        private static func fill(_ count: Int) -> [Int] { [Int](repeating: 7, count: count) }

        /// Random short patterns over a small alphabet, so matches, overlaps
        /// and patterns inside other patterns are all common.
        private struct RandomCases {
            var generator = CBv2SchedSplitMix64(seed: 20_261_109)

            mutating func tokens(count: ClosedRange<Int>, alphabet: Int) -> [Int] {
                var tokens: [Int] = []
                for _ in 0 ..< Int.random(in: count, using: &generator) {
                    tokens.append(Int.random(in: 0 ..< alphabet, using: &generator))
                }
                return tokens
            }

            mutating func next() -> (rule: Rule, tokens: [Int]) {
                let alphabet = Int.random(in: 2 ... 4, using: &generator)
                let open = tokens(count: 1 ... 3, alphabet: alphabet)
                var close: [[Int]] = []
                for _ in 0 ..< Int.random(in: 1 ... 3, using: &generator) {
                    close.append(tokens(count: 1 ... 3, alphabet: alphabet))
                }
                return (Rule(open: open, close: close), tokens(count: 0 ... 40, alphabet: alphabet))
            }
        }

        /// A scan anchored at the start of each match: it takes the first
        /// pattern that begins at the cursor, so it reads past the boundary it
        /// reports.
        private static func startAnchoredBoundaries(_ rule: Rule, in tokens: [Int]) -> [Int] {
            guard !rule.open.isEmpty else { return [] }
            var result: [Int] = []
            var index = 0
            var inAssistant = false
            func matches(_ pattern: [Int], at start: Int) -> Bool {
                start + pattern.count <= tokens.count
                    && tokens[start ..< start + pattern.count].elementsEqual(pattern)
            }
            while index < tokens.count {
                if matches(rule.open, at: index) {
                    index += rule.open.count
                    inAssistant = true
                } else if inAssistant,
                    let pattern = rule.close.first(where: { matches($0, at: index) })
                {
                    index += pattern.count
                    inAssistant = false
                } else {
                    index += 1
                    continue
                }
                result.append(index)
            }
            return result
        }

        private static func contains(_ whole: [Int], _ part: [Int]) -> Bool {
            guard part.count <= whole.count else { return false }
            return (0 ... whole.count - part.count).contains {
                whole[$0 ..< $0 + part.count].elementsEqual(part)
            }
        }

        // MARK: The rule

        @Test func familyRulesEndAChunkAtBothEdgesOfAnAssistantMessage() {
            for (family, rule) in Self.familyRules {
                for close in rule.close {
                    // A close before any assistant message, then the reply,
                    // a follow-up and the generation prompt.
                    var tokens = Self.fill(5) + rule.close[0] + rule.open
                    let replyStart = tokens.count
                    tokens += Self.fill(4) + close
                    let replyEnd = tokens.count
                    tokens += Self.fill(3) + rule.open
                    #expect(
                        rule.boundaries(in: tokens) == [replyStart, replyEnd, tokens.count],
                        "\(family), close \(close)")
                }
            }
        }

        @Test func anEmptyRuleOrPromptHasNoBoundaries() {
            #expect(Rule(open: [], close: [[2]]).boundaries(in: [1, 2, 3]).isEmpty)
            #expect(Rule(open: [1], close: [[2]]).boundaries(in: []).isEmpty)
            #expect(Rule(open: [1], close: [[], [2], []]).close == [[2]])
        }

        @Test func aMatchStartsAtOrAfterThePreviousBoundary() {
            let rule = Rule(open: [1, 1], close: [[1, 2]])
            #expect(rule.boundaries(in: [1, 1, 1]) == [2])
            #expect(rule.boundaries(in: [1, 1, 1, 1]) == [2, 4])
            #expect(rule.boundaries(in: [1, 1, 2]) == [2], "the open already used the close's 1")
            #expect(rule.boundaries(in: [1, 1, 1, 2]) == [2, 4])
        }

        @Test func everyBoundaryIsDecidedByTheTokensBeforeIt() {
            var cases = RandomCases()
            for _ in 0 ..< 2000 {
                let (rule, tokens) = cases.next()
                let whole = rule.boundaries(in: tokens)
                #expect(whole == Array(Set(whole)).sorted(), "\(rule) \(tokens)")
                #expect(whole.allSatisfy { $0 > 0 && $0 <= tokens.count }, "\(rule) \(tokens)")
                for cut in 0 ... tokens.count {
                    #expect(
                        rule.boundaries(in: Array(tokens[..<cut])) == whole.filter { $0 <= cut },
                        "\(rule) \(tokens) cut at \(cut)")
                }
            }
        }

        /// The family rules above are all of this kind.
        @Test func patternsThatDoNotContainOneAnotherMatchTheStartAnchoredScan() {
            var cases = RandomCases()
            var compared = 0
            var withBoundaries = 0
            for _ in 0 ..< 4000 {
                let (rule, tokens) = cases.next()
                let patterns = [rule.open] + rule.close
                let independent = patterns.indices.allSatisfy { one in
                    patterns.indices.allSatisfy { other in
                        one == other || !Self.contains(patterns[one], patterns[other])
                    }
                }
                guard independent else { continue }
                let boundaries = rule.boundaries(in: tokens)
                #expect(
                    boundaries == Self.startAnchoredBoundaries(rule, in: tokens),
                    "\(rule) \(tokens)")
                compared += 1
                if !boundaries.isEmpty { withBoundaries += 1 }
            }
            #expect(compared > 1000 && withBoundaries > 500, "the sample exercises the scan")
            for (family, rule) in Self.familyRules {
                let patterns = [rule.open] + rule.close
                for (one, other) in patterns.indices.flatMap({ one in
                    patterns.indices.filter { $0 != one }.map { (one, $0) }
                }) {
                    #expect(!Self.contains(patterns[one], patterns[other]), "\(family)")
                }
            }
        }

        @Test func aLongerCloseCannotMoveABoundaryAShorterOneEnded() {
            let rule = Rule(open: [9], close: [[1, 2, 3], [2]])
            // The start-anchored scan moves the boundary once the next token arrives.
            #expect(Self.startAnchoredBoundaries(rule, in: [9, 1, 2]) == [1, 3])
            #expect(Self.startAnchoredBoundaries(rule, in: [9, 1, 2, 3]) == [1, 4])
            #expect(rule.boundaries(in: [9, 1, 2]) == [1, 3])
            #expect(rule.boundaries(in: [9, 1, 2, 3]) == [1, 3])
        }

        // MARK: The partition

        @Test func aCellEndsAtTheNextGridMultipleOrMessageBoundary() {
            let partition = CBv2CanonicalPartition(grid: 8, boundaries: [5, 19])
            let starts = [0, 5, 8, 16, 19, 24]
            #expect(starts.map { partition.cellTokens(start: $0) } == [5, 3, 8, 3, 5, 8])
            #expect(partition.cellTokens(start: 2) == 3, "a start inside a cell gets its rest")
            #expect(partition.cellEnd(from: 24, limit: 30) == 30)
            #expect((0 ... 24).filter { partition.isCanonical($0) } == [5, 8, 16, 19, 24])
        }

        @Test func theDeepestCanonicalPositionRespectsTheAlignment() {
            let partition = CBv2CanonicalPartition(grid: 8, boundaries: [5, 19])
            #expect(partition.deepestCanonical(atOrBelow: 23, multipleOf: 1) == 19)
            #expect(partition.deepestCanonical(atOrBelow: 23, multipleOf: 4) == 16)
            #expect(partition.deepestCanonical(atOrBelow: 4, multipleOf: 1) == nil)
            #expect(partition.deepestCanonical(atOrBelow: 0, multipleOf: 1) == nil)
            #expect(partition.deepestCanonical(atOrBelow: 23, multipleOf: 0) == nil)
        }

        @Test func aPromptWalksTheCellsOfEveryLongerPromptItStarts() throws {
            var cases = RandomCases()
            for _ in 0 ..< 1000 {
                let (rule, tokens) = cases.next()
                var config = CBv2SchedulerConfig()
                config.canonicalPrefillGridTokens = Int.random(in: 1 ... 6, using: &cases.generator)
                config.canonicalBoundaryRule = rule
                func cellEnds(_ prompt: [Int]) throws -> [Int] {
                    let partition = try #require(config.canonicalPartition(for: prompt))
                    var ends: [Int] = []
                    while (ends.last ?? 0) < prompt.count {
                        ends.append(partition.cellEnd(from: ends.last ?? 0, limit: prompt.count))
                    }
                    return ends
                }
                let whole = try cellEnds(tokens)
                for cut in tokens.indices.dropFirst() {
                    #expect(
                        try cellEnds(Array(tokens[..<cut])) == whole.filter { $0 < cut } + [cut],
                        "\(rule) \(tokens) cut at \(cut)")
                }
            }
        }

        // MARK: Checkpoint capture

        @Test func captureAcceptsWholeCanonicalCellsOnly() {
            let partition = CBv2CanonicalPartition(grid: 8, boundaries: [5, 19])
            var geometry = CBv2RecurrentCheckpointGeometry()
            var atBoundary: [Bool] = []
            for range in [0 ..< 5, 5 ..< 8, 8 ..< 16, 16 ..< 19, 19 ..< 24, 24 ..< 30] {
                atBoundary.append(
                    geometry.record(
                        range: range, cap: 8, promptLength: 30, packed: false,
                        partition: partition))
            }
            // The prompt end, 30, is no canonical position.
            #expect(atBoundary == [true, true, true, true, true, false])
            #expect(geometry.isArmed && geometry.position == 30)

            for split in [0 ..< 4, 0 ..< 8] {
                var geometry = CBv2RecurrentCheckpointGeometry()
                let captured = geometry.record(
                    range: split, cap: 8, promptLength: 30, packed: false, partition: partition)
                #expect(!captured && !geometry.isArmed && geometry.disarmReason == .geometry)
            }
        }
    }
}
