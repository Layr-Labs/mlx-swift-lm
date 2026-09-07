import Foundation
import MLX
import MLXFast

struct PagedQuantizedPrefillDispatch {
    let prepared: PagedSegmentPreparedDispatch
    let output: MLXArray
}

/// All blocks write disjoint query columns of one native output allocation.
/// The fence also protects the reused FP32 partials from successor overwrites.
enum PagedQuantizedPrefillMerge {
    private static let body = """
        cbv2::paged_attention_merge_impl<T, D, PTOK, HAS_SINKS>(
            partials, meta, seqinfo, sinks, partials_shape[1], partials_shape[2],
            const_cast<device T*>(output), threadgroup_position_in_grid,
            thread_index_in_simdgroup, output_shape[2]);
        if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
            fence[0] = previous[0] + 1;
        }
        """
    private static let complete = MLXFast.metalKernel(
        name: "cbv2_quant_prefill_visible", inputNames: ["previous"], outputNames: ["witness"],
        source: "witness[0] = previous[0] + 1;", ensureRowContiguous: true)
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    static func dispatch(partials: MLXArray, meta: MLXArray, seqinfo: MLXArray,
                         sinks: MLXArray, previous: MLXArray, output: MLXArray,
                         queryCount: Int, headDim: Int, queryHeads: Int,
                         partitionTokens: Int, hasSinks: Bool, source: String) -> MLXArray {
        let name = "cbv2_quant_prefill_merge_\(output.dtype)_d\(headDim)_p\(partitionTokens)_s\(hasSinks ? 1 : 0)"
        let kernel = lock.withLock {
            if let value = kernels[name] { return value }
            let value = MLXFast.metalKernel(
                name: name, inputNames: ["partials", "meta", "seqinfo", "sinks", "previous", "output"],
                outputNames: ["fence"], source: body, header: source, ensureRowContiguous: true)
            kernels[name] = value
            return value
        }
        return kernel([partials, meta, seqinfo, sinks, previous, output],
                      template: [("T", output.dtype), ("D", headDim), ("PTOK", partitionTokens), ("HAS_SINKS", hasSinks)],
                      grid: (queryHeads * 32, queryCount, 1), threadGroup: (32, 1, 1),
                      outputShapes: [[1]], outputDTypes: [.int32])[0]
    }

    static func finish(output: MLXArray, group: PagedKVGroup,
                       workspace: PagedQuantizedAttentionWorkspace) -> MLXArray {
        // A real consumer of the last merge fence provides the Metal barrier
        // for hidden output writes before a later Depends/Slice can copy them.
        group.writeFence = complete([group.writeFence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                                    outputShapes: [[1]], outputDTypes: [.int32])[0]
        workspace.recordCompletion(group.writeFence)
        return depends(input: output, dependencies: [group.writeFence])
    }
}
