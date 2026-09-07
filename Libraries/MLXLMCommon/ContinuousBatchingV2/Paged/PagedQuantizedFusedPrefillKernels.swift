import Foundation
import MLX
import MLXFast

/// Hidden writes are always followed by a real fence consumer before an MLX
/// alias enters SDPA or the model. All tensor inputs retain their true strides.
enum PagedQuantizedFusedPrefillKernels {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]
    private static let joinKernel = MLXFast.metalKernel(
        name: "cbv2_quant_sdpa_acquire", inputNames: ["previous", "last_use"],
        outputNames: ["fence"], source: "fence[0] = max(previous[0], last_use[0]) + 1;",
        ensureRowContiguous: false)
    private static let witnessKernel = MLXFast.metalKernel(
        name: "cbv2_quant_sdpa_visible", inputNames: ["previous"], outputNames: ["fence"],
        source: "fence[0] = previous[0] + 1;", ensureRowContiguous: false)

    static func acquire(previous: MLXArray, lastUse: MLXArray?) -> MLXArray {
        guard let lastUse else { return previous }
        return joinKernel([previous, lastUse], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                          outputShapes: [[1]], outputDTypes: [.int32])[0]
    }
    static func witness(_ previous: MLXArray) -> MLXArray {
        witnessKernel([previous], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                      outputShapes: [[1]], outputDTypes: [.int32])[0]
    }

    private static func kernel(name: String, inputs: [String], body: String,
                               header: String = "") -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let result = kernels[name] { return result }
            let result = MLXFast.metalKernel(name: name, inputNames: inputs,
                                            outputNames: ["fence"], source: body,
                                            header: header, ensureRowContiguous: false)
            kernels[name] = result
            return result
        }
    }

    static func dequantize(group: PagedKVGroup, prepared: PagedSegmentPreparedDispatch,
                           destination: MLXArray, parameters: MLXArray) {
        let quant = group.key.quantization!
        let d = group.key.headDim, h = group.key.kvHeads
        for (bucket, metadata) in zip(prepared.plan.buckets, prepared.metadata) {
            let bindings = bucket.bindingClass
            let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
            let body = """
                const int column = thread_position_in_threadgroup.x;
                const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
                const int h = threadgroup_position_in_grid.z % H;
                const int token = record[1] * PTOK + threadgroup_position_in_grid.z / H;
                if (token >= parameters[0]) return;
                cbv2::PagedQuantizedSegmentAccessor<float, SEGMENTS, D, S, G, KB, VB> cache{
                    {\(pointers)}, value_offsets, record, H};
                const int position = parameters[1] + token;
                const auto page = cache.read_page(records, position / S, 0);
                constexpr int KROW = D * KB / 8 + 8 * (D / G);
                constexpr int VROW = D * VB / 8 + 8 * (D / G);
                const size_t row = (size_t)h * S + position % S;
                const float key = cbv2::quant_load<KB, D, G>(page.key + row * KROW, column);
                const float value = cbv2::quant_load<VB, D, G>(page.value + row * VROW, column);
                const size_t count = destination_shape[3];
                const size_t target = ((size_t)h * count + token) * D + column;
                device float* out = const_cast<device float*>(destination);
                out[target] = key;
                out[(size_t)H * count * D + target] = value;
                if (column == 0 && threadgroup_position_in_grid.y == 0 && threadgroup_position_in_grid.z == 0) {
                    fence[0] = previous[0] + 1;
                }
                """
            let name = "cbv2_quant_sdpa_dequant_d\(d)_s\(group.pageSize)_k\(quant.keyBits)v\(quant.valueBits)g\(quant.groupSize)_b\(bindings)"
            var storage = bucket.segmentIDs.map { group.segments[$0]!.storage }
            storage.append(contentsOf: repeatElement(storage[0], count: bindings - storage.count))
            group.writeFence = kernel(
                name: name, inputs: (0 ..< bindings).map { "segment\($0)" }
                    + ["records", "value_offsets", "destination", "parameters", "previous"],
                body: body, header: PagedQuantizedMetal.header)(
                storage + [metadata.records, metadata.valueOffsets, destination, parameters, group.writeFence],
                template: [("H", h), ("D", d), ("S", group.pageSize), ("G", quant.groupSize),
                           ("KB", quant.keyBits), ("VB", quant.valueBits), ("SEGMENTS", bindings),
                           ("STRIDE", PagedSegmentDispatchPlan.recordStride), ("PTOK", prepared.plan.partitionTokens)],
                grid: (d, bucket.workCount, prepared.plan.partitionTokens * h), threadGroup: (d, 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
    }

    private static let queryBody = """
        const int d = thread_position_in_threadgroup.x;
        const int h = threadgroup_position_in_grid.y;
        const int row = threadgroup_position_in_grid.z;
        const int64_t source = h * input_strides[1] + (parameters[3] + row) * input_strides[2] + d * input_strides[3];
        threadgroup float values[D];
        const float value = float(input[source]);
        values[d] = R == 0 ? value : value * cbv2::quant_sign(d % max(R, 1));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < R; stride <<= 1) {
            const float a = values[d], b = values[d ^ stride];
            const float next = (d & stride) ? b - a : a + b;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            values[d] = next;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float result = R == 0 ? values[d] : values[d] * rsqrt(float(R));
        const_cast<device float*>(destination)[((size_t)h * MAXQ + row) * D + d] = result;
        if (d == 0 && h == 0 && row == 0) fence[0] = previous[0] + 1;
        """
    private static let maskBody = """
        const int key = thread_position_in_grid.x, row = thread_position_in_grid.y;
        if (key >= parameters[0] || row >= parameters[4]) return;
        const bool allowed = parameters[1] + key <= parameters[2] + parameters[3] + row;
        const_cast<device bool*>(destination)[(size_t)row * destination_shape[1] + key] = allowed;
        if (key == 0 && row == 0) fence[0] = previous[0] + 1;
        """

    static func prepareBlock(queries: MLXArray, arena: PagedQuantizedFusedPrefillArena,
                              parameters: MLXArray, count: Int, attendLength: Int,
                              quantization: PagedKVQuantizationConfig, previous: MLXArray) -> MLXArray {
        let d = queries.dim(3), rotation = quantization.resolvedRotationBlockSize(headDim: d)
        let qFence = kernel(name: "cbv2_quant_sdpa_query_\(queries.dtype)_d\(d)_r\(rotation)",
                            inputs: ["input", "destination", "parameters", "previous"],
                            body: queryBody, header: PagedQuantizedMetal.header)(
            [queries, arena.queries, parameters, previous],
            template: [("D", d), ("R", rotation), ("MAXQ", PagedQuantizedFusedPrefillPlan.maximumQueries)],
            grid: (d, queries.dim(1), count), threadGroup: (d, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
        let maskFence = kernel(name: "cbv2_quant_sdpa_mask", inputs: ["destination", "parameters", "previous"],
                               body: maskBody)(
            [arena.mask, parameters, qFence], grid: (attendLength, count, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
        return witness(maskFence)
    }

    private static let copyBody = """
        const int index = thread_position_in_grid.x;
        const int count = parameters[4];
        if (index >= H * count * D) return;
        const int d = index % D, row = (index / D) % count, h = index / (D * count);
        const int64_t source = h * attention_strides[1] + row * attention_strides[2] + d * attention_strides[3];
        const size_t target = ((size_t)h * parameters[5] + parameters[3] + row) * D + d;
        const_cast<device T*>(destination)[target] = T(attention[source]);
        if (index == 0) fence[0] = previous[0] + 1;
        """
    static func copyOutput(_ attention: MLXArray, output: MLXArray, parameters: MLXArray,
                            count: Int, previous: MLXArray) -> MLXArray {
        let h = output.dim(1), d = output.dim(3)
        return kernel(name: "cbv2_quant_sdpa_output_\(output.dtype)_d\(d)",
                      inputs: ["attention", "destination", "parameters", "previous"], body: copyBody)(
            [attention, output, parameters, previous], template: [("T", output.dtype), ("H", h), ("D", d)],
            grid: (h * count * d, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
    }
}
