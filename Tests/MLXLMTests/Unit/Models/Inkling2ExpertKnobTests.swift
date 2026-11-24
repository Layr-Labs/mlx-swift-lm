// Copyright © 2026 Eigen Labs.

import Foundation
import Testing

@testable import MLXLLM

extension UnitTests {

    /// The grouped expert GEMVs' row and pair knobs take values in their
    /// ranges only; any other value keeps the default.
    @Suite
    struct Inkling2ExpertKnobTests {
        static let knobs: [(name: String, fallback: Int, low: Int, high: Int)] = [
            ("DARKBLOOM_INKLING2_EXPERT_GROUP_ROWS", 32, 8, 64),
            ("DARKBLOOM_INKLING2_EXPERT_GROUP_FROM", 9, 2, 64),
            ("DARKBLOOM_INKLING2_EXPERT_GROUP_TILE_AT", 4, 0, 64),
            ("DARKBLOOM_INKLING2_EXPERT_GROUP_UP_ROWS", 1, 1, 2),
        ]

        private func knob(_ name: String, _ fallback: Int, _ value: String?) -> Int {
            Inkling2Kernels.expertGroupKnob(
                name, default: fallback, environment: value.map { [name: $0] } ?? [:])
        }

        @Test func valuesInRangeAreTaken() {
            for (name, fallback, low, high) in Self.knobs {
                #expect(knob(name, fallback, "\(low)") == low, "\(name)")
                #expect(knob(name, fallback, "\(high)") == high, "\(name)")
            }
        }

        @Test func valuesOutOfRangeKeepTheDefault() {
            for (name, fallback, low, high) in Self.knobs {
                for value in [low - 1, high + 1, -1, 1_000_000] where value < low || value > high {
                    #expect(knob(name, fallback, "\(value)") == fallback, "\(name)=\(value)")
                }
            }
        }

        @Test func unsetOrUnparsedValuesKeepTheDefault() {
            for (name, fallback, _, _) in Self.knobs {
                #expect(knob(name, fallback, nil) == fallback, "\(name)")
                #expect(knob(name, fallback, "") == fallback, "\(name)")
                #expect(knob(name, fallback, "eight") == fallback, "\(name)")
            }
        }
    }
}
