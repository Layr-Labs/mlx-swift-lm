// Copyright © 2026 Eigen Labs.
import Cmlx
import Foundation
import MLX
import MLXLMCommon

/// No caller-owned Array/COW capacity is retained. The private compact witness
/// is bounded by maxPositionEmbeddings and charged by boundedRequestAllocation.
final class MiMoV26MTPPrefixContext {
    private let tokens: ContiguousArray<Int32>
    var count: Int { tokens.count }

    init(_ prompt: [Int]) {
        tokens = ContiguousArray(unsafeUninitializedCapacity: prompt.count) { buffer, count in
            for (index, token) in prompt.enumerated() { buffer[index] = Int32(token) }
            count = prompt.count
        }
    }

    func matches(_ array: MLXArray, count: Int) -> Bool {
        guard count <= tokens.count else { return false }
        return MiMoV26MTPPrefixValidation.tokensMatch(array, count: count) { tokens[$0] }
    }
}

struct MiMoV26MTPPrefixHead {
    let keys: MLXArray
    let values: MLXArray
    // firstPosition, keep, window, step, localOffset, physicalRingIndex, liveLength
    let metadata: [Int64]
}

enum MiMoV26MTPPrefixValidation {
    static func metadataIsValid(
        _ row: [Int64], depth: Int, count: Int,
        configuration c: MiMoV26Configuration
    ) -> Bool {
        guard (0 ..< 3).contains(depth), count >= 4, count < c.maxPositionEmbeddings,
            row.count == 7, c.slidingWindow > 0
        else { return false }
        let offset = count - depth - 1
        let window = c.slidingWindow
        guard row[0] == Int64(depth + 1), row[1] == 0,
            row[2] == Int64(window), row[3] == 256, row[4] == Int64(offset),
            row[6] == Int64(min(offset, window))
        else { return false }
        return offset < window
            ? row[5] == Int64(offset)
            : row[5] > 0 && row[5] <= Int64(window)
    }

    static func headIsValid(
        _ head: MiMoV26MTPPrefixHead, depth: Int, count: Int,
        configuration c: MiMoV26Configuration,
        allowsUnusedCapacity: Bool = false
    ) -> Bool {
        guard metadataIsValid(head.metadata, depth: depth, count: count, configuration: c),
            head.keys.ndim == 4, head.values.ndim == 4,
            head.keys.dtype == .bfloat16, head.values.dtype == .bfloat16
        else { return false }
        let g = c.slidingAttention
        let live = min(count - depth - 1, c.slidingWindow)
        let length = head.keys.dim(2)
        guard head.keys.shape == [1, g.keyValueHeads, length, g.headDim],
            head.values.shape == [1, g.keyValueHeads, length, g.valueHeadDim]
        else { return false }
        return allowsUnusedCapacity
            ? length >= live && length <= c.slidingWindow : length == live
    }

    /// A non-evaluating readback. Lazy/noncontiguous imports fail cold; complete
    /// checkpoint staging must already have completed its real native read.
    static func tokensMatch(
        _ array: MLXArray, count: Int,
        expected: (Int) -> Int32
    ) -> Bool {
        guard array.dtype == .int32, array.shape == [1, count],
            let info = try? array.evaluatedBufferInfo(), info.isRowContiguous,
            info.dataElements == array.size,
            let pointer = mlx_array_data_int32(array.ctx)
        else { return false }
        return withExtendedLifetime(array) {
            for index in 0 ..< count where pointer[index] != expected(index) { return false }
            return true
        }
    }

    static func copyTokens(_ array: MLXArray, into output: inout ContiguousArray<Int32>) -> Bool {
        guard array.dtype == .int32, array.shape == [1, output.count],
            let info = try? array.evaluatedBufferInfo(), info.isRowContiguous,
            info.dataElements == array.size,
            let pointer = mlx_array_data_int32(array.ctx)
        else { return false }
        return withExtendedLifetime(array) {
            for index in output.indices { output[index] = pointer[index] }
            return true
        }
    }

    static func metadata(_ array: MLXArray) -> [[Int64]]? {
        guard array.dtype == .int64, array.shape == [3, 7],
            let info = try? array.evaluatedBufferInfo(), info.isRowContiguous,
            info.dataElements == array.size,
            let pointer = mlx_array_data_int64(array.ctx)
        else { return nil }
        return withExtendedLifetime(array) {
            (0 ..< 3).map { depth in (0 ..< 7).map { pointer[depth * 7 + $0] } }
        }
    }
}

extension MiMoV26MTPState {
    func recordPrefixObservation(tokens: MLXArray) {
        guard prefixCapturePromptOnly, let context = prefixCaptureContext else { return }
        let count = tokens.dim(1)
        guard observedCount <= context.count, count <= context.count - observedCount else {
            prefixCapturePromptOnly = false
            return
        }
        // uint32 inputs have the same bytes; out-of-vocabulary/high-bit values
        // cannot pass the later exact signed witness comparison. Never recast
        // floating features or head caches in this path.
        let incoming = tokens.dtype == .int32 ? tokens : tokens.view(dtype: .int32)
        let joined = prefixObservedTokens.map { concatenated([$0, incoming], axis: 1) } ?? incoming
        prefixObservedTokens = mimoV26MTPCopy(joined)
    }
}
