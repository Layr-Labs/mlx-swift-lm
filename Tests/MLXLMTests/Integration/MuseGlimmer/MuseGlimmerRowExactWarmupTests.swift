// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The verify-width warm-up of `setRowExactWindows(true)`: it compiles every
/// row-exact kernel a verify window of 1 to 8 rows dispatches, changes no
/// bytes, and the environment can turn it off.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerRowExactWarmupTests {
    typealias Fixture = MuseGlimmerExactWindowTests
    static let key = MuseGlimmerTextModel.rowExactWarmupEnvironmentKey
    static let off = [key: "0"]

    /// The row-exact kernels (name and template values) `body` dispatches.
    static func pipelines(_ body: () throws -> Void) rethrows -> Set<String> {
        let log = ModelOptRowExactPipelines.Log()
        try ModelOptRowExactPipelines.$log.withValue(log) { try body() }
        return log.recorded
    }

    /// Every kernel a window of 1 to 8 tokens, a decode step and a batch of
    /// 8 rows dispatch was compiled when exact windows were turned on.
    @Test(arguments: [DType.bfloat16, .float32])
    func warmupCompilesEveryKernelAWindowUses(dtype: DType) throws {
        let m = try Fixture.mixed(dtype: dtype)
        let loaded = try Self.pipelines { try m.setRowExactWindows(true, environment: [:]) }
        #expect(!loaded.isEmpty)
        let prompts = [31, 9].map { MuseGlimmerFixture.tokens($0, vocabulary: 96, salt: $0) }
        var calls: [(String, [[Int32]], Bool)] = (1 ... 8).map { width in
            (
                "window \(width)", [MuseGlimmerFixture.tokens(width, vocabulary: 96, salt: width)],
                true
            )
        }
        calls.append(("decode step", [[5]], false))
        calls.append(("2 x 4 window", [[1, 2, 3, 4], [5, 6, 7, 8]], true))
        for (name, tokens, rectangular) in calls {
            let paged = try Fixture.Paged(m)
            for prompt in prompts.prefix(tokens.count) { try paged.add(prompt, extra: 8) }
            paged.select(Array(tokens.indices))
            let used = Self.pipelines {
                _ = paged.forward(tokens, rectangular: rectangular, columnsAsRows: rectangular)
            }
            #expect(!used.isEmpty, "\(name)")
            #expect(used.isSubset(of: loaded), "\(name): \(used.subtracting(loaded).sorted())")
        }
    }

    /// The same logits and K/V with the warm-up off, after it on the same
    /// tower, and on a tower that warmed up when exact windows were turned on.
    @Test(arguments: [DType.bfloat16, .float32])
    func warmupLeavesTheBytesUnchanged(dtype: DType) throws {
        func run(_ m: MuseGlimmerTextModel) throws -> (logits: [MLXArray], kv: [Data]) {
            let paged = try Fixture.Paged(m)
            try paged.add(MuseGlimmerFixture.tokens(70, vocabulary: 96, salt: 70), extra: 48)
            paged.select([0])
            var logits = (1 ... 8).map { width in
                paged.forward(
                    [MuseGlimmerFixture.tokens(width, vocabulary: 96, salt: 300 + width)],
                    rectangular: true, columnsAsRows: true)
            }
            logits.append(paged.forward([[5]], rectangular: false))
            return (logits, paged.kv(0))
        }
        let m = try Fixture.mixed(dtype: dtype)
        try m.setRowExactWindows(true, environment: Self.off)
        let cold = try run(m)
        m.warmRowExactWindows()
        let after = try run(m)
        let warmed = try Fixture.mixed(dtype: dtype)
        try warmed.setRowExactWindows(true, environment: [:])
        let warm = try run(warmed)
        for other in [after, warm] {
            #expect(other.logits.count == cold.logits.count)
            for (a, b) in zip(cold.logits, other.logits) {
                #expect(MuseGlimmerFixture.sameBits(a, b))
            }
            #expect(other.kv == cold.kv)
        }
    }

    /// `0` turns the warm-up off; anything else, or nothing, leaves it on.
    /// Turning exact windows off, or a refused switch, never warms up.
    @Test func killSwitchTurnsTheWarmupOff() throws {
        #expect(MuseGlimmerTextModel.rowExactWarmupEnabled([:]))
        #expect(MuseGlimmerTextModel.rowExactWarmupEnabled([Self.key: "1"]))
        #expect(!MuseGlimmerTextModel.rowExactWarmupEnabled(Self.off))
        #expect(Self.key == "DARKBLOOM_MUSE_GLIMMER_ROW_EXACT_WARMUP")
        let m = try Fixture.mixed()
        #expect(
            try Self.pipelines { try m.setRowExactWindows(true, environment: Self.off) }.isEmpty)
        #expect(m.verifiesWindowsExactly)
        #expect(try Self.pipelines { try m.setRowExactWindows(false, environment: [:]) }.isEmpty)
        #expect(!m.verifiesWindowsExactly)
        // 64 inputs have no row-exact product: the refused switch compiles
        // nothing.
        let refused = try MuseGlimmerFixture.model(
            MuseGlimmerFixture.pagedText(window: Fixture.window, layers: 2, hidden: 64))
        #expect(
            Self.pipelines {
                #expect(throws: MuseGlimmerRowExactError.self) {
                    try refused.setRowExactWindows(true, environment: [:])
                }
            }.isEmpty)
        #expect(!refused.verifiesWindowsExactly)
    }
}
