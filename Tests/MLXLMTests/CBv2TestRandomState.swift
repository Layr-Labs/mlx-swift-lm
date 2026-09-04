// CBv2TestRandomState.swift
//
// Private PRNG state for random-initialised test fixtures.
//
// ## The bug this removes
//
// `MLXRandom.seed(_:)` writes `MLXRandom.globalState`, which is process-wide.
// swift-testing runs SUITES in parallel — `.serialized` orders tests inside one
// suite, not between suites — and eight suites in the MTP filter seed globally
// to build random-init Gemma 4 / Qwen 3.5 fixtures:
//
//   CBv2MTPRoundSmokeTests        CBv2MTPCaptureVerifyTests
//   CBv2MTPEngineParityTests      CBv2MTPKVStagingTests
//   CBv2MTPEngineMixedTests       CBv2MTPRectangularDegradeTests
//   CBv2MTPModelSeamTests         Gemma4MTPStochasticTests
//
// So one suite's `seed()` could land between another's `seed()` and its weight
// draw, and that suite would silently build a DIFFERENT model than its seed
// names. On a 6-layer random model over a 256-token vocabulary near-ties in the
// argmax are routine, and under greedy decoding one flipped token cascades —
// which is how `soloGreedyTokenExactWithWindowWrap`, an MTP-on vs MTP-off
// token-exactness test, failed 1 run in 8 with suites in parallel, 0 in 5 with
// `--no-parallel`, and 0 in 10 run alone.
//
// ## The fix
//
// `withRandomState` binds a `@TaskLocal` `MLXRandom.RandomState` that MLX's
// `resolve(key:)` prefers over the global one, so every nested `MLXRandom` call
// that does not pass an explicit key draws from THAT state. The
// `.fixtureRandomState` trait binds one per test, and `fixtureSeed(_:)` re-seeds
// the bound state instead of the global one.
//
// Nothing is serialised and no key is threaded through production model
// initialisers; the suites simply stop sharing a mutable global.
//
// Rule for this target: fixtures call `fixtureSeed(_:)`, never
// `MLXRandom.seed(_:)`. The latter still writes the process-global state, which
// under `.fixtureRandomState` would be BOTH racy and ignored — the worst of
// both, since the draws would no longer follow the seed the test names.

import MLX
import Testing

enum CBv2FixtureRandom {
    /// Starting state for a test that never calls `fixtureSeed`. A constant, so
    /// such a test is reproducible rather than inheriting whatever the global
    /// stream happened to hold.
    static let baseSeed: UInt64 = 0xCB_2F_5E_ED

    @TaskLocal static var current: MLXRandom.RandomState?
}

/// Seed the PRNG this test is using.
///
/// Under `.fixtureRandomState` that is the test's private state; without it,
/// the global one, so the helper is safe to use from a suite that has not been
/// annotated yet.
func fixtureSeed(_ seed: UInt64) {
    if let state = CBv2FixtureRandom.current {
        state.seed(seed)
    } else {
        MLXRandom.seed(seed)
    }
}

/// Give every test in the annotated suite its own `MLXRandom.RandomState`.
struct CBv2FixtureRandomStateTrait: TestTrait, SuiteTrait, TestScoping {
    /// Applies to the tests inside an annotated suite, not just the suite.
    var isRecursive: Bool { true }

    func provideScope(
        for test: Test,
        testCase: Test.Case?,
        performing function: @Sendable () async throws -> Void
    ) async throws {
        let state = MLXRandom.RandomState(seed: CBv2FixtureRandom.baseSeed)
        try await CBv2FixtureRandom.$current.withValue(state) {
            try await withRandomState(state) {
                try await function()
            }
        }
    }
}

extension Trait where Self == CBv2FixtureRandomStateTrait {
    /// Fixtures in this suite draw from a PRNG private to each test.
    static var fixtureRandomState: Self { .init() }
}
