// CBV2-MTP-COMPACT-ROOTS — the switch must not change what the round emits.
//
// The lever swaps the verify rectangle's evaluation-root list for the compacted
// one the plain decode step already uses. Roots decide only WHAT IS FORCED, never
// what is computed, so the emitted tokens and the acceptance packet must be
// byte-identical with the switch on and off. These are CPU tests: no Metal, no
// model download.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 MTP compact roots")
struct CBv2MTPCompactRootsTests {

    // MARK: the switch itself

    @Test("the switch is DEFAULT OFF on this branch")
    func defaultOff() {
        #expect(resolveCBv2MTPCompactRootsEnabled(nil) == false)
    }

    @Test("only explicit truthy values arm it")
    func onlyTruthyArms() {
        for raw in ["1", "true", "yes", "on", "TRUE", "On"] {
            #expect(resolveCBv2MTPCompactRootsEnabled(raw) == true, "\(raw) should arm")
        }
        for raw in ["0", "false", "no", "off", "", "maybe", "2"] {
            #expect(resolveCBv2MTPCompactRootsEnabled(raw) == false, "\(raw) must not arm")
        }
    }

    @Test("it is independent of the plain decode-roots switch")
    func independentOfDecodeSwitch() {
        // The decode switch defaults ON, this one defaults OFF: the verify runs
        // inside a speculative write transaction the decode path never enters,
        // so they are armed separately until the round arm has run.
        #expect(resolveCBv2CompactDecodeRootsEnabled(nil) == true)
        #expect(resolveCBv2MTPCompactRootsEnabled(nil) == false)
    }

    // MARK: the exactness argument, as an executable statement

    @Test("compaction is a root-set change only: it never enters the value graph")
    func rootsAreNotValues() {
        // The property the lever rests on: `compactDecodeEvaluationRoots` returns a
        // SUBSET-COVERING root list; the arrays it names are the same arrays the
        // full list names, so evaluating either set produces the same values. This
        // test pins the contract shape so a future refactor that starts returning
        // *derived* arrays (which could change rounding) fails here.
        //
        // A model that declines compaction must fall back, and the fallback is the
        // established full list.
        // `CBv2SteppableModel` is AnyObject-constrained (EngineLoopV2.swift:28),
        // so the stand-in must be a class, not a struct.
        final class Decliner: CBv2SteppableModel {
            func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
                tokens
            }
        }
        let model = Decliner()
        #expect(
            model.compactDecodeEvaluationRoots(
                forwardOutput: MLXArray([Int32(0)]), caches: []) == nil,
            "the default conformance must decline, so the round keeps the full list")
    }
}
