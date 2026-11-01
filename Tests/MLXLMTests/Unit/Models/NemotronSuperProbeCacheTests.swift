import Foundation
import Testing

@testable import MLXVLM

extension UnitTests {
    /// The Super adapter resolves its probe values (the recurrent-state spec,
    /// the attention KV types) once: the first result, nil included, is
    /// returned for every later read without probing again. Swift logic
    /// only; no arrays are created.
    struct NemotronSuperProbeCacheTests {
        /// Reads `cache` with a probe that returns `value` and counts its calls.
        static func read<Value>(
            _ cache: NemotronSuperProbeCache<Value>, probing value: Value, probes: inout Int
        ) -> Value {
            cache.value {
                probes += 1
                return value
            }
        }

        @Test func theFirstValueIsComputedOnceAndKept() {
            let cache = NemotronSuperProbeCache<Int>()
            var probes = 0
            #expect(Self.read(cache, probing: 7, probes: &probes) == 7)
            #expect(Self.read(cache, probing: 8, probes: &probes) == 7)
            #expect(probes == 1)
        }

        @Test func aNilResultIsKeptToo() {
            let cache = NemotronSuperProbeCache<[Int]?>()
            var probes = 0
            #expect(Self.read(cache, probing: nil, probes: &probes) == nil)
            #expect(Self.read(cache, probing: [1], probes: &probes) == nil)
            #expect(probes == 1)
        }
    }
}
