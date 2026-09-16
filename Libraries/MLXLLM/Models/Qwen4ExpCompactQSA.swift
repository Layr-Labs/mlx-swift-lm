import Foundation
import MLX
import MLXLMCommon
import os

/// Native selected-page read profile. Sparse selection and Steel attention arithmetic
/// remain the existing implementation; only KV row materialization/addressing
/// changes. No alternate SDPA and no split-K reduction are used.
enum Qwen4ExpCompactQSA {
    static let envFlag = "DARKBLOOM_QWEN4_QSA_PAGED_SELECTED"
    static let slotsPerQuery = 512 * 4 + 3

    static func enabled(environment: [String: String] = Qwen4ExpEnvironment.snapshot) -> Bool {
        ["1", "true", "yes", "on"].contains(
            environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "1")
    }

    static func eligible(offset: Int, width: Int) -> Bool {
        (1...Qwen4ExpGatheredQSA.optimizedVerifyMaxQueryTokens).contains(width)
            && Qwen4ExpGatheredQSA.decodeCrossesBudget(
            offset: offset, compressRatio: 4, tokenBudget: 2048)
    }

    static func isSparsityWin(offset: Int, width: Int) -> Bool {
        eligible(offset: offset, width: width) && width * slotsPerQuery < offset + width
    }

    /// Exact physical rows for Steel's fixed 2,051-slot loop, one packed bank
    /// per query. Preserve duplicate/order semantics; invalid tail slots are
    /// zero-filled, while the attention kernel still masks ABSOLUTE positions.
    static func tokenIndices(selected: MLXArray, offset: Int, keyTokens: Int) -> MLXArray {
        let width = selected.dim(1)
        precondition(selected.shape == [1, width, 512] && offset >= 0 && offset + width <= keyTokens)
        let positions = Qwen4ExpGatheredQSA.int32Range(offset, offset + width).reshaped([1, width, 1])
        let counts = floorDivide(positions + 1, 4)
        let expanded = (selected.asType(.int32).reshaped([1, width, 512, 1]) * Int32(4)
            + Qwen4ExpGatheredQSA.int32Range(4).reshaped([1, 1, 1, 4]))
            .reshaped([1, width, 2048])
        let validSlots = Qwen4ExpGatheredQSA.int32Range(2048).reshaped([1, 1, 2048])
            .< minimum(counts, MLXArray(Int32(512))) * Int32(4)
        let validBlocks = validSlots .&& (expanded .>= 0) .&& (expanded .< Int32(keyTokens))
            .&& (expanded .<= positions)
        let tail = counts * Int32(4) + Qwen4ExpGatheredQSA.int32Range(3).reshaped([1, 1, 3])
        let validTail = (tail .< Int32(keyTokens)) .&& (tail .<= positions)
        return concatenated([
            MLX.where(validBlocks, expanded, MLXArray(Int32(-1))),
            MLX.where(validTail, tail, MLXArray(Int32(-1)))], axis: -1).reshaped([-1])
    }

    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        indexQueries: MLXArray, cache: any CBv2Qwen4SelectedKVCache,
        offset: Int, indexerHeadDim: Int,
        indexKeyNorm: (MLXArray) -> MLXArray,
        applyIndexRope: (MLXArray, MLXArray) -> MLXArray,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray? {
        let width = queries.dim(2), keyTokens = offset + width
        guard enabled(environment: environment), isSparsityWin(offset: offset, width: width),
              Qwen4ExpNativeSparseGQA.steelEnabled(), !Qwen4ExpGatheredQSA.decodeGatherEnabled(),
              Qwen4ExpNativeSparseGQA.canAttend(
                queries: queries, keys: keys, values: values, selectedWidth: 512),
              cache.qwen4CanGatherSelectedKV(keys: keys, values: values)
        else { return nil }
        let pooled = Qwen4ExpPooledIndex.reuseOrCompute(
            cache: cache, compressRatio: 4, logicalTokens: keyTokens,
            indexKeyNorm: indexKeyNorm, applyIndexRope: applyIndexRope)
        let selected: MLXArray
        if width == 1 {
            selected = Qwen4ExpGatheredQSA.decodeSelectedBlocks(
                indexQueries: indexQueries, pooledIndexKeys: pooled, keyTokens: keyTokens,
                indexerHeadDim: indexerHeadDim, maxBlocks: keyTokens / 4, blockBudget: 512)
        } else {
            selected = Qwen4ExpGatheredQSA.verifySelectedBlocks(
                indexQueries: indexQueries, pooledIndexKeys: pooled, keyTokens: keyTokens,
                indexerHeadDim: indexerHeadDim, maxBlocks: keyTokens / 4, blockBudget: 512, ratio: 4)
        }
        let indices = tokenIndices(selected: selected, offset: offset, keyTokens: keyTokens)
        let compact = cache.qwen4UpdateAndGatherSelectedKV(keys: keys, values: values, tokenIndices: indices)
        // All optional checks preceded the write. A nil here is a programming
        // error, never permission to write/advance this row a second time.
        guard let output = Qwen4ExpNativeSparseGQA.attend(
            queries: queries, keys: compact.keys, values: compact.values,
            selectedBlocks: selected, qOffset: offset,
            outputPartitions: width == 1 ? Qwen4ExpGatheredQSA.decodeOutputPartitions()
                : Qwen4ExpGatheredQSA.verifyOutputPartitions(queryTokens: width),
            compactLogicalKeyTokens: keyTokens) else {
            preconditionFailure("Qwen4 compact KV violated its prevalidated native geometry")
        }
        if width == 1 { Qwen4ExpQSAInvocation.recordDecode() }
        else { Qwen4ExpQSAInvocation.recordVerify() }
        Qwen4ExpSelectedKVInvocation.record(width: width, logicalTokens: keyTokens)
        return output
    }
}

public enum Qwen4ExpSelectedKVInvocation {
    private static let lock = NSLock()
    private static let logger = Logger(subsystem: "darkbloom", category: "Qwen4SelectedKV")
    private static let diagnoseFirstPlan = Qwen4ExpEnvironment.snapshot[
        "DARKBLOOM_QWEN4_QSA_PAGED_SELECTED_DIAGNOSTICS"] == "1"
    nonisolated(unsafe) private static var calls = 0
    nonisolated(unsafe) private static var gatheredRows = 0
    nonisolated(unsafe) private static var logicalRows = 0

    static func record(width: Int, logicalTokens: Int) {
        let first = lock.withLock {
            calls += 1
            gatheredRows += width * Qwen4ExpCompactQSA.slotsPerQuery
            logicalRows += logicalTokens
            return calls == 1
        }
        if first && diagnoseFirstPlan {
            // Graph-path evidence, not a completed GPU dispatch counter.
            logger.info("qwen4_selected_kv first_planned_call=1 width=\(width, privacy: .public) logical_tokens=\(logicalTokens, privacy: .public)")
        }
    }

    public static func snapshot() -> (calls: Int, gatheredRows: Int, logicalRows: Int) {
        lock.withLock { (calls, gatheredRows, logicalRows) }
    }
}
