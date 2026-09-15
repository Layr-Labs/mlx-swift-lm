import Foundation
import MLX
import MLXLMCommon
import os

/// Opt-in decode/verify experiment. Independent QK tiles run in parallel;
/// FP32 scores feed the unchanged ordered online-softmax/PV recurrence. No split-K
/// reduction, selector change or lower precision. Full-KV mode consumes the
/// caller's existing materialized arrays; it never changes page ownership.
enum Qwen4ExpParallelQSA {
    static let flag = "DARKBLOOM_QWEN4_QSA_PARALLEL_SCORES"
    static let fullKVFlag = "DARKBLOOM_QWEN4_QSA_PARALLEL_FULL_KV"
    static func enabled(environment: [String: String] = Qwen4ExpEnvironment.snapshot) -> Bool {
        environment[flag] == "1"
    }

    static func fullKVEnabled(environment: [String: String] = Qwen4ExpEnvironment.snapshot) -> Bool {
        environment[fullKVFlag] == "1"
    }

    static func valuePartitions(requested: Int?, fallback: Int,
                                environment: [String: String] = Qwen4ExpEnvironment.snapshot) -> Int {
        let value = requested ?? Int(environment["DARKBLOOM_QWEN4_QSA_PARALLEL_VALUE_PARTITIONS"] ?? "")
        return value.flatMap { [1, 2, 4, 8, 16, 32].contains($0) ? $0 : nil } ?? fallback
    }

    private static let scoreStore = """
            const int padded_tokens = n_tiles * BK;
            device float* dst = output
                + ((size_t(q_pos) * 2 + size_t(kv_head)) * H_PAD + size_t(tm + sm))
                    * size_t(padded_tokens)
                + size_t(topk_off + sn);
            Stile.template store_safe<float, 1, 1>(
                dst, padded_tokens, short2(BK - sn, H_PAD - (tm + sm)));
        }
        """

    private static let scoreLoad = """
            const int padded_tokens = n_tiles * BK;
            const device float* src = score_bank
                + ((size_t(q_pos) * 2 + size_t(kv_head)) * H_PAD + size_t(tm + sm))
                    * size_t(padded_tokens)
                + size_t(topk_off + sn);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TK; ++j) {
                Frag::load(Stile.frag_at(0, j), src + j * kFragSize, padded_tokens, 1);
            }
        """

    private static let scores = MLXFast.metalKernel(
        name: "qwen4_parallel_qk_scores",
        inputNames: ["queries", "keys", "values", "selected", "q_offset"],
        outputNames: ["output"],
        source: qwen4SparseGQASteelPrefix(qkOnly: true) + qwen4SparseGQASteelDotSource + scoreStore,
        header: qwen4SparseGQASteelHeader + "\n" + qwen4SparseGQAStridedLoadHeader,
        ensureRowContiguous: true)

    private static let values = MLXFast.metalKernel(
        name: "qwen4_ordered_pv_from_scores",
        inputNames: ["queries", "keys", "values", "selected", "q_offset", "score_bank"],
        outputNames: ["output"],
        source: qwen4SparseGQASteelPrefix() + scoreLoad + qwen4SparseGQASteelValueSource,
        header: qwen4SparseGQASteelHeader + "\n" + qwen4SparseGQAStridedLoadHeader,
        ensureRowContiguous: true)

    static func attend(inputs: [MLXArray], queryTokens: Int, outputPartitions: Int,
                       valuePartitions requestedPartitions: Int?,
                       compactKV: Bool,
                       outputShape: [Int], dtype: DType) -> MLXArray {
        precondition(
            (1...Qwen4ExpGatheredQSA.optimizedVerifyMaxQueryTokens).contains(queryTokens)
                && [1, 2, 4].contains(outputPartitions))
        let partitions = valuePartitions(requested: requestedPartitions, fallback: outputPartitions)
        let valueTile = min(64, 256 / partitions)
        let keyTiles = (Qwen4ExpCompactQSA.slotsPerQuery + 63) / 64
        let simdWidth = 32, warps = 2, kvHeads = 2
        let common: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("BK", 64),
            ("COMPACT_KV", compactKV), ("STRIDED_KV", false)]
        let bank = scores(inputs, template: common + [("DC", 64), ("OPARTS", 1)],
            grid: (queryTokens * simdWidth, kvHeads * warps, keyTiles),
            threadGroup: (simdWidth, warps, 1),
            outputShapes: [[queryTokens, 2, 16, keyTiles * 64]], outputDTypes: [.float32])[0]
        // Each partition owns disjoint output columns. Key-tile traversal,
        // softmax and every column's MMA sequence remain unchanged.
        let output = values(inputs + [bank], template: common + [("DC", valueTile), ("OPARTS", partitions)],
            grid: (queryTokens * simdWidth, kvHeads * warps, partitions),
            threadGroup: (simdWidth, warps, 1),
            outputShapes: [outputShape], outputDTypes: [dtype])[0]
        Qwen4ExpParallelQSAInvocation.record(width: queryTokens, partitions: partitions)
        return output
    }
}

public enum Qwen4ExpParallelQSAInvocation {
    private static let lock = NSLock()
    private static let logger = Logger(subsystem: "darkbloom", category: "Qwen4ParallelQSA")
    private static let diagnoseFirstPlan = Qwen4ExpEnvironment.snapshot[
        "DARKBLOOM_QWEN4_QSA_PARALLEL_SCORES_DIAGNOSTICS"] == "1"
    nonisolated(unsafe) private static var calls = 0
    static func record(width: Int, partitions: Int) {
        let first = lock.withLock {
            calls += 1
            return calls == 1
        }
        if first && diagnoseFirstPlan {
            // Graph-path evidence only; completed outputs are verified separately.
            logger.info("qwen4_parallel_scores first_planned_call=1 width=\(width, privacy: .public) output_partitions=\(partitions, privacy: .public)")
        }
    }
    public static func snapshot() -> Int { lock.withLock { calls } }
}
