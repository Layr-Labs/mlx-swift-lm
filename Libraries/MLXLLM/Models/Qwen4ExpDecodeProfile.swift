// Copyright © 2026 Apple Inc. and the MLX Swift LM contributors.
//
// Qwen4-Exp decode profiler — port of Fusion `_Qwen4DecodeProfileSample`
// and `_Qwen4CoarseProfileSample` (language.py). Diagnostic only; every
// hook is a no-op unless the matching environment flag is set.
//
//   DARKBLOOM_QWEN4_DECODE_PROFILE=1
//       Synchronized per-stage sample. An eval + synchronize boundary is
//       placed after every major model stage so the attribution is
//       trustworthy; the sampled token is therefore NOT a throughput
//       measurement. Stage names match Fusion's log line exactly.
//   DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE=1
//       Ordinary lazy path: time graph construction separately from the
//       final logits eval. Ignored when the fine profile is on.
//   DARKBLOOM_QWEN4_DECODE_PROFILE_WARMUP    (default 8)
//   DARKBLOOM_QWEN4_DECODE_PROFILE_INTERVAL  (default 16)
//
// Only ordinary B1/T1 text decode is sampled (same predicate as Fusion:
// token ids of shape [1, 1], no input embeddings, context > 0).

import Foundation
import MLX
import MLXLMCommon
import os

private let log = Logger(subsystem: "darkbloom", category: "Qwen4DecodeProfile")

enum Qwen4ExpDecodeProfile {
    /// Fusion stage keys, in log order.
    static let stageOrder = [
        "embed+mask", "ple", "attn_hc", "gdn", "qsa", "attn_residual",
        "mlp_hc", "moe", "mlp_residual", "final_hc",
    ]

    enum Mode {
        /// Fusion `_Qwen4DecodeProfileSample`: eval + synchronize after every stage.
        case synchronized
        /// Darkbloom addition: time only the lazy construction of each stage
        /// (no forced eval). Exposes host-side graph-build cost and any
        /// hidden syncs a stage performs internally.
        case buildOnly
    }

    final class Sample {
        let callIndex: Int
        let contextTokens: Int
        let width: Int
        let mode: Mode
        var stageNs: [String: UInt64] = [:]
        var modelNs: UInt64 = 0

        init(callIndex: Int, contextTokens: Int, width: Int = 1, mode: Mode) {
            self.callIndex = callIndex
            self.contextTokens = contextTokens
            self.width = width
            self.mode = mode
        }

        func add(_ stage: String, _ elapsed: UInt64) {
            stageNs[stage, default: 0] += elapsed
        }

        func milliseconds(_ stage: String) -> Double {
            Double(stageNs[stage] ?? 0) / 1_000_000
        }
    }

    struct CoarseSample {
        let callIndex: Int
        let contextTokens: Int
    }

    /// `1` = Fusion synchronized stages; `build` = construction-only stages.
    static func fineMode(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Mode? {
        switch environment["DARKBLOOM_QWEN4_DECODE_PROFILE"] {
        case "1": return .synchronized
        case "build": return .buildOnly
        default: return nil
        }
    }

    static func fineEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        fineMode(environment: environment) != nil
    }

    static func coarseEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        environment["DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE"] == "1" && !fineEnabled(environment: environment)
    }

    static func profileInt(
        _ name: String, default defaultValue: Int, minimum: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        guard let raw = environment[name], let value = Int(raw) else { return defaultValue }
        return max(minimum, value)
    }

    /// Fusion `call_index <= warmup or (call_index - warmup - 1) % interval` gate.
    static func shouldSample(callIndex: Int, warmup: Int, interval: Int) -> Bool {
        if callIndex <= warmup { return false }
        return (callIndex - warmup - 1) % max(1, interval) == 0
    }

    /// `DARKBLOOM_QWEN4_DECODE_PROFILE_MAX_WIDTH` (default 1): widest B1
    /// text call the profiler samples. Raising it to the Lightning verify
    /// width (1+k) attributes the rectangular target forward per stage.
    static func maxWidth(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        profileInt(
            "DARKBLOOM_QWEN4_DECODE_PROFILE_MAX_WIDTH", default: 1, minimum: 1,
            environment: environment)
    }

    /// Fusion `_new_decode_profile_sample` predicate: ordinary B1/T1 text decode
    /// (widened to `maxWidth` when the operator asks for verify-width samples).
    static func isOrdinaryDecode(
        inputs: MLXArray, inputEmbeddings: MLXArray?, contextTokens: Int,
        maxWidth: Int = Qwen4ExpDecodeProfile.maxWidth()
    ) -> Bool {
        inputEmbeddings == nil && inputs.ndim == 2 && inputs.dim(0) == 1
            && (1 ... maxWidth).contains(inputs.dim(1))
            && contextTokens > 0
    }

    // MARK: - Fine (synchronized) sampling

    // All three are guarded exclusively by `lock`.
    private static let lock = NSLock()
    nonisolated(unsafe) private static var fineCalls = 0
    nonisolated(unsafe) private static var coarseCalls = 0
    /// The sample owned by the decode call currently inside the model. Qwen4
    /// decode is serialized per model (CBv2 engine), matching Fusion's
    /// thread-local; a lock guards it anyway.
    nonisolated(unsafe) private static var activeSample: Sample?

    static var current: Sample? {
        lock.lock()
        defer { lock.unlock() }
        return activeSample
    }

    static func beginFine(inputs: MLXArray, inputEmbeddings: MLXArray?, contextTokens: Int)
        -> Sample?
    {
        guard let mode = fineMode(),
            isOrdinaryDecode(
                inputs: inputs, inputEmbeddings: inputEmbeddings, contextTokens: contextTokens)
        else { return nil }
        lock.lock()
        fineCalls += 1
        let callIndex = fineCalls
        lock.unlock()
        let warmup = profileInt("DARKBLOOM_QWEN4_DECODE_PROFILE_WARMUP", default: 8, minimum: 0)
        let interval = profileInt(
            "DARKBLOOM_QWEN4_DECODE_PROFILE_INTERVAL", default: 16, minimum: 1)
        guard shouldSample(callIndex: callIndex, warmup: warmup, interval: interval) else {
            return nil
        }
        let sample = Sample(
            callIndex: callIndex, contextTokens: contextTokens, width: inputs.dim(1), mode: mode)
        lock.lock()
        activeSample = sample
        lock.unlock()
        return sample
    }

    /// In build-only mode `graphBuildNs` is the un-forced construction time
    /// of the whole call; the final eval is then timed separately so the
    /// line reads like `[qwen4-decode-coarse]` with a per-stage breakdown.
    static func endFine(_ sample: Sample, logits: MLXArray, totalStart: UInt64) {
        let built = now()
        CBv2DeferredHostFill.resolveBeforeEvaluation()
        eval(logits)
        Stream().synchronize()
        let finished = now()
        lock.lock()
        if activeSample === sample { activeSample = nil }
        lock.unlock()
        logFine(sample, totalNs: finished &- totalStart, buildNs: built &- totalStart,
            evalNs: finished &- built)
    }

    /// Fusion `_profile_stage`: run, force the result, synchronize, attribute.
    /// Build-only mode records the body alone.
    @inline(__always)
    static func stage(_ name: String, _ body: () -> MLXArray) -> MLXArray {
        guard let sample = current else { return body() }
        let started = now()
        let result = body()
        if sample.mode == .synchronized {
            CBv2DeferredHostFill.resolveBeforeEvaluation()
            eval(result)
            Stream().synchronize()
        }
        sample.add(name, now() &- started)
        return result
    }

    @inline(__always)
    static func stageKV(_ name: String, _ body: () -> [(keys: MLXArray, values: MLXArray)])
        -> [(keys: MLXArray, values: MLXArray)]
    {
        guard let sample = current else { return body() }
        let started = now()
        let result = body()
        if sample.mode == .synchronized {
            CBv2DeferredHostFill.resolveBeforeEvaluation()
            eval(result.flatMap { [$0.keys, $0.values] })
            Stream().synchronize()
        }
        sample.add(name, now() &- started)
        return result
    }

    @inline(__always)
    static func stage3(_ name: String, _ body: () -> (MLXArray, MLXArray, MLXArray))
        -> (MLXArray, MLXArray, MLXArray)
    {
        guard let sample = current else { return body() }
        let started = now()
        let result = body()
        if sample.mode == .synchronized {
            CBv2DeferredHostFill.resolveBeforeEvaluation()
            eval(result.0, result.1, result.2)
            Stream().synchronize()
        }
        sample.add(name, now() &- started)
        return result
    }

    private static func logFine(_ sample: Sample, totalNs: UInt64, buildNs: UInt64, evalNs: UInt64)
    {
        let totalMs = Double(totalNs) / 1_000_000
        let modelMs = Double(sample.modelNs) / 1_000_000
        let outerMs = max(0, totalMs - modelMs)
        // Substages are included in qsa; never double-count them as model work.
        let accountedMs = Double(sample.stageNs.filter { !$0.key.hasPrefix("qsa.") }.values.reduce(0, +)) / 1_000_000
        let unaccountedMs = max(0, modelMs - accountedMs)
        let stages = stageOrder.map { "\($0)=\(fmt(sample.milliseconds($0)))ms" }
            .joined(separator: " ")
        let line: String
        switch sample.mode {
        case .synchronized:
            line =
                "[qwen4-decode-profile] synchronized diagnostic call=\(sample.callIndex) "
                + "context=\(sample.contextTokens) width=\(sample.width) "
                + "total=\(fmt(totalMs))ms model=\(fmt(modelMs))ms "
                + "outer+logits=\(fmt(outerMs))ms \(stages) model_other=\(fmt(unaccountedMs))ms"
        case .buildOnly:
            line =
                "[qwen4-decode-build] construction-only call=\(sample.callIndex) "
                + "context=\(sample.contextTokens) width=\(sample.width) "
                + "graph_build=\(fmt(Double(buildNs) / 1_000_000))ms "
                + "final_eval=\(fmt(Double(evalNs) / 1_000_000))ms total=\(fmt(totalMs))ms "
                + "model_build=\(fmt(modelMs))ms \(stages) model_other=\(fmt(unaccountedMs))ms"
        }
        let details = ["qsa.kv_read", "qsa.pool", "qsa.attend"]
            .filter { sample.stageNs[$0] != nil }
            .map { "\($0)=\(fmt(sample.milliseconds($0)))ms" }.joined(separator: " ")
        emit(line + (details.isEmpty ? "" : " included_in_qsa: " + details))
    }

    // MARK: - Coarse (lazy build vs final eval) sampling

    static func beginCoarse(inputs: MLXArray, inputEmbeddings: MLXArray?, contextTokens: Int)
        -> CoarseSample?
    {
        guard coarseEnabled(),
            isOrdinaryDecode(
                inputs: inputs, inputEmbeddings: inputEmbeddings, contextTokens: contextTokens)
        else { return nil }
        lock.lock()
        coarseCalls += 1
        let callIndex = coarseCalls
        lock.unlock()
        let warmup = profileInt(
            "DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE_WARMUP", default: 8, minimum: 0)
        let interval = profileInt(
            "DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE_INTERVAL", default: 16, minimum: 1)
        guard shouldSample(callIndex: callIndex, warmup: warmup, interval: interval) else {
            return nil
        }
        return CoarseSample(callIndex: callIndex, contextTokens: contextTokens)
    }

    static func endCoarse(_ sample: CoarseSample, logits: MLXArray, buildStart: UInt64) {
        let built = now()
        CBv2DeferredHostFill.resolveBeforeEvaluation()
        eval(logits)
        Stream().synchronize()
        let evaluated = now()
        let buildMs = Double(built &- buildStart) / 1_000_000
        let evalMs = Double(evaluated &- built) / 1_000_000
        emit(
            "[qwen4-decode-coarse] ordinary lazy path call=\(sample.callIndex) "
                + "context=\(sample.contextTokens) graph_build=\(fmt(buildMs))ms "
                + "final_eval=\(fmt(evalMs))ms total=\(fmt(buildMs + evalMs))ms")
    }

    // MARK: - Logits / stage dump (parity forensics)

    /// `DARKBLOOM_QWEN4_DUMP_DIR=<dir>`: write each B1 text call's logits per
    /// absolute position as raw bf16 (`pos<N>.logits.bf16`) so a serial run
    /// and a Lightning verify run can be diffed position by position. When
    /// `DARKBLOOM_QWEN4_DUMP_STAGES=<pos>` names a position, every decoder
    /// stage output at that position is written too
    /// (`pos<N>.L<layer>.<stage>.bf16`). Synchronizes; DEBUG-only. Release
    /// ignores the env vars so a hostile operator cannot arm raw dumps.
    static let dumpDirectory: URL? = {
        #if DEBUG
            guard let raw = Qwen4ExpEnvironment.snapshot["DARKBLOOM_QWEN4_DUMP_DIR"],
                !raw.isEmpty
            else { return nil }
            let url = URL(fileURLWithPath: raw)
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        #else
            return nil
        #endif
    }()

    static let dumpStagesPosition: Int? = {
        #if DEBUG
            return Qwen4ExpEnvironment.snapshot["DARKBLOOM_QWEN4_DUMP_STAGES"].flatMap(Int.init)
        #else
            return nil
        #endif
    }()

    /// Absolute positions of the columns of the call currently inside the model.
    nonisolated(unsafe) private static var dumpColumnBase: Int?
    nonisolated(unsafe) private static var dumpKind: String = "call"

    static func beginDump(
        inputs: MLXArray, inputEmbeddings: MLXArray?, contextTokens: Int, kind: String = "call"
    ) {
        guard dumpDirectory != nil, inputEmbeddings == nil, inputs.ndim == 2, inputs.dim(0) == 1
        else { return }
        lock.lock()
        dumpColumnBase = contextTokens
        dumpKind = "\(kind).w\(inputs.dim(1))"
        lock.unlock()
    }

    static func endDump(logits: MLXArray) {
        guard let dir = dumpDirectory else { return }
        lock.lock()
        let base = dumpColumnBase
        let kind = dumpKind
        dumpColumnBase = nil
        lock.unlock()
        guard let base, logits.ndim == 3, logits.dim(0) == 1 else { return }
        let width = logits.dim(1)
        // Never force the graph mid-build: a Lightning round fills its
        // host-resident PLE rows only after the whole round graph exists, so
        // an eager eval here would read stale slots. Run the write as the
        // last deferred fill instead (they run just before submit, in order).
        deferredWrite {
            eval(logits)
            for column in 0 ..< width {
                let url = dir.appendingPathComponent(
                    "pos\(base + column).\(kind).c\(column).logits.bf16")
                if FileManager.default.fileExists(atPath: url.path) { continue }
                let row = logits[0, column].asType(.bfloat16)
                eval(row)
                try? row.asData(access: .copy).data.write(to: url)
            }
        }
    }

    private static func deferredWrite(_ body: @escaping () -> Void) {
        if let scope = CBv2DeferredHostFill.current {
            scope.register(body)
        } else {
            body()
        }
    }

    /// Clears a dump column base left by a call that returned without logits.
    static func endDumpIfPending() {
        lock.lock()
        dumpColumnBase = nil
        lock.unlock()
    }

    /// Stage hook: writes `[1, T, ...]` stage outputs for the dump position.
    static func dumpStage(_ name: String, layer: Int, _ value: MLXArray) {
        guard let dir = dumpDirectory, let target = dumpStagesPosition else { return }
        lock.lock()
        let base = dumpColumnBase
        let kind = dumpKind
        lock.unlock()
        guard let base, value.ndim >= 2, value.dim(0) == 1 else { return }
        let width = value.dim(1)
        guard target >= base, target < base + width else { return }
        let url = dir.appendingPathComponent(
            "pos\(target).\(kind).c\(target - base).L\(layer).\(name).bf16")
        if FileManager.default.fileExists(atPath: url.path) { return }
        let column = target - base
        deferredWrite {
            let row = value[0, column].asType(.bfloat16)
            eval(row)
            try? row.asData(access: .copy).data.write(to: url)
        }
    }

    /// Whole-tensor dump (any dtype, written as float32) for small state
    /// tensors such as PLE history/conv rows at the dump position's call.
    static func dumpTensor(_ name: String, layer: Int, _ value: MLXArray) {
        guard let dir = dumpDirectory, let target = dumpStagesPosition else { return }
        lock.lock()
        let base = dumpColumnBase
        let kind = dumpKind
        lock.unlock()
        guard let base else { return }
        let width = Int(kind.split(separator: "w").last.map(String.init) ?? "1") ?? 1
        guard target >= base, target < base + width else { return }
        let url = dir.appendingPathComponent("pos\(target).\(kind).b\(base).L\(layer).\(name).f32")
        if FileManager.default.fileExists(atPath: url.path) { return }
        deferredWrite {
            let flat = value.asType(.float32)
            eval(flat)
            try? flat.asData(access: .copy).data.write(to: url)
        }
    }

    // MARK: - Helpers

    @inline(__always)
    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    private static func fmt(_ ms: Double) -> String { String(format: "%.3f", ms) }

    private static func emit(_ line: String) {
        log.info("\(line, privacy: .public)")
        FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
    }
}
