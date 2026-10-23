// TEMPORARY diagnostic (branch diag/pr339-flakes only; never merged).
// Splits the VNKV 32K smoke into CPU graph build and eval time per step, repeats
// the decode, and records memory at each step.

import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import XCTest

final class DiagVNKVTimingTests: XCTestCase {
    private func now() -> Double { Date.timeIntervalSinceReferenceDate }
    private func mb(_ bytes: Int) -> String { String(format: "%.1fMB", Double(bytes) / 1e6) }

    private func line(_ label: String, build: Double, eval: Double, baseline: Int) {
        print(
            "DIAG \(label): build=\(String(format: "%.4f", build))s "
                + "eval=\(String(format: "%.4f", eval))s "
                + "active=\(mb(Memory.activeMemory)) cache=\(mb(Memory.cacheMemory)) "
                + "peak=\(mb(Memory.peakMemory)) peakOverBaseline=\(Memory.peakMemory - baseline)")
    }

    private func run(round: Int, synchronizeBeforeBaseline: Bool) {
        let tokenCount = 32_768
        let cache = VarianceNormalizedKVCache(
            tileSize: 128, keyBits: 4, valueBits: 2, sinkhornIterations: 8)
        let queries = MLXRandom.normal([1, 32, 1, 128]).asType(.float16)
        let keys = MLXRandom.normal([1, 8, tokenCount, 128]).asType(.float16)
        let values = MLXRandom.normal([1, 8, tokenCount, 128]).asType(.float16)
        var t = now()
        eval(queries, keys, values)
        print("DIAG round\(round) inputs eval=\(String(format: "%.4f", now() - t))s")
        let activeBeforeSync = Memory.activeMemory
        if synchronizeBeforeBaseline { Stream.gpu.synchronize() }
        Memory.clearCache()
        let baseline = Memory.activeMemory
        print(
            "DIAG round\(round) sync=\(synchronizeBeforeBaseline) activeBeforeSync=\(activeBeforeSync) "
                + "baseline=\(baseline)")
        Memory.peakMemory = 0

        t = now()
        let output = cache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: 1 / sqrt(Float(128)))
        let build = now() - t
        t = now()
        eval(output)
        line("round\(round) ingest", build: build, eval: now() - t, baseline: baseline)

        for step in 0 ..< 4 {
            let q = MLXRandom.normal([1, 32, 1, 128]).asType(.float16)
            let k = MLXRandom.normal([1, 8, 1, 128]).asType(.float16)
            let v = MLXRandom.normal([1, 8, 1, 128]).asType(.float16)
            eval(q, k, v)
            Stream.gpu.synchronize()
            t = now()
            let out = cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: 1 / sqrt(Float(128)))
            let b = now() - t
            t = now()
            eval(out)
            let e = now() - t
            line("round\(round) decode\(step)", build: b, eval: e, baseline: baseline)
        }
    }

    func testVNKVTimingDiagnostics() throws {
        #if os(macOS) && arch(arm64)
        print("DIAG device=\(GPU.deviceInfo())")
        run(round: 0, synchronizeBeforeBaseline: false)
        run(round: 1, synchronizeBeforeBaseline: false)
        run(round: 2, synchronizeBeforeBaseline: true)
        #endif
    }
}
