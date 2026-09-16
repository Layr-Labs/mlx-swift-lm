import Foundation
import MLX
import MLXFast
import os

/// Selected-row copies only: no attention math, selector readback, or backing
/// slab concatenation. Segmented storage follows PagedSegmentTransfers' exact
/// write-fence/completion-witness contract for hidden destination mutations.
/// `DARKBLOOM_QWEN4_PAGED_SELECTED_GATHER_BOUND=1` groups up to 17 current
/// segment buffers per pass; it is the default. Explicit zero retains the
/// previous per-segment read path. Only Qwen4 selected-row reads use this gate.
enum PagedSelectedGather {
    static let boundEnvFlag = "DARKBLOOM_QWEN4_PAGED_SELECTED_GATHER_BOUND"

    struct Plan {
        let table: MLXArray
        let segmentIDs: [Int]
    }

    static let maximumBindings = PagedSegmentDispatchPlan.maximumBindings
    static let bindingClasses = PagedSegmentDispatchPlan.bindingClasses

    static func boundEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        environment[boundEnvFlag, default: "1"] == "1"
    }

    private static let monolithic = MLXFast.metalKernel(
        name: "cbv2_paged_selected_gather", inputNames: ["keys", "values", "table", "indices", "length", "previous"],
        outputNames: ["output", "fence"], source: """
        const int d = int(thread_position_in_grid.x);
        const int h = int(thread_position_in_grid.y);
        const int r = int(thread_position_in_grid.z);
        const int token = indices[r];
        const size_t target = ((size_t)h * R + r) * D + d;
        T k = T(0), v = T(0);
        if (token >= 0 && token < length[0]) {
            const int page = table[token / S];
            const size_t source = (((size_t)page * H + h) * S + token % S) * D + d;
            k = keys[source]; v = values[source];
        }
        output[target] = k;
        output[(size_t)H * R * D + target] = v;
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """, ensureRowContiguous: true)

    private static let kernelLock = NSLock()
    nonisolated(unsafe) private static var segmentedKernels: [Int: MLXFast.MLXFastKernel] = [:]

    private static let segmentedLegacy = MLXFast.metalKernel(
        name: "cbv2_segment_selected_gather",
        inputNames: ["storage", "table", "indices", "length", "output", "previous"],
        outputNames: ["fence"], source: """
        const int d = int(thread_position_in_grid.x);
        const int h = int(thread_position_in_grid.y);
        const int r = int(thread_position_in_grid.z);
        const int token = indices[r];
        if (token >= 0 && token < length[0]) {
            const int page = table[token / S];
            if (page > FIRST && page < END) {
                const int local = page - FIRST;
                const size_t source = (((size_t)local * H + h) * S + token % S) * D + d;
                const size_t target = ((size_t)h * R + r) * D + d;
                device T* destination = output;
                destination[target] = storage[source];
                destination[(size_t)H * R * D + target] = storage[VBASE + source];
            }
        }
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """, ensureRowContiguous: true, mutableInputs: ["output"])

    static func bindingBatches(segmentIDs: [Int]) -> [[Int]] {
        precondition(segmentIDs == Array(Set(segmentIDs)).sorted())
        return stride(from: 0, to: segmentIDs.count, by: maximumBindings).map {
            Array(segmentIDs[$0 ..< min($0 + maximumBindings, segmentIDs.count)])
        }
    }

    private static func bindingClass(for count: Int) -> Int {
        precondition(count > 0 && count <= maximumBindings)
        return bindingClasses.first { $0 >= count }!
    }

    private static func segmentedInputNames(bindings: Int) -> [String] {
        (0 ..< bindings).map { "segment\($0)" }
            + ["table", "indices", "length", "output", "previous", "bounds", "value_offsets"]
    }

    private static func segmentedSource(bindings: Int) -> String {
        let copies = (0 ..< bindings).map { index in
            let branch = index == 0 ? "if" : "else if"
            return """
                    \(branch) (page > bounds[\(2 * index)] && page < bounds[\(2 * index + 1)]) {
                        const int local = page - bounds[\(2 * index)];
                        const size_t source = (((size_t)local * H + h) * S + token % S) * D + d;
                        destination[target] = segment\(index)[source];
                        destination[(size_t)H * R * D + target] =
                            segment\(index)[size_t(value_offsets[\(index)]) + source];
                    }
                """
        }.joined(separator: "\n")
        return """
        const int d = int(thread_position_in_grid.x);
        const int h = int(thread_position_in_grid.y);
        const int r = int(thread_position_in_grid.z);
        const int token = indices[r];
        const size_t target = ((size_t)h * R + r) * D + d;
        device T* destination = output;
        if (token >= 0 && token < length[0]) {
            const int page = table[token / S];
        \(copies)
        }
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """
    }

    private static func segmentedKernel(bindings: Int) -> MLXFast.MLXFastKernel {
        kernelLock.withLock {
            if let cached = segmentedKernels[bindings] { return cached }
            let kernel = MLXFast.metalKernel(
                name: "cbv2_segment_selected_gather_bound\(bindings)",
                inputNames: segmentedInputNames(bindings: bindings),
                outputNames: ["fence"], source: segmentedSource(bindings: bindings),
                ensureRowContiguous: true, mutableInputs: ["output"])
            segmentedKernels[bindings] = kernel
            return kernel
        }
    }

    private static let complete = MLXFast.metalKernel(
        name: "cbv2_selected_gather_complete", inputNames: ["previous"], outputNames: ["witness"],
        source: "witness[0] = previous[0] + 1;", ensureRowContiguous: true)

    static func prepare(group: PagedKVGroup, pages: [Int32]) -> Plan {
        let segmentIDs = group.segmentLayout.map { layout in
            Array(Set(pages.map { layout.segmentIndex(page: $0) })).sorted()
        } ?? []
        return Plan(table: MLXArray(pages), segmentIDs: segmentIDs)
    }

    static func gather(group: PagedKVGroup, plan: Plan, indices: MLXArray, length: Int)
        -> (keys: MLXArray, values: MLXArray) {
        precondition(indices.ndim == 1 && indices.dtype == .int32 && indices.size > 0)
        precondition(length > 0 && plan.table.size * group.pageSize >= length)
        let h = group.key.kvHeads, d = group.key.headDim, count = indices.size
        let logicalLength = MLXArray([Int32(length)])
        let common: [(String, any KernelTemplateArg)] = [
            ("T", group.dtype), ("H", h), ("D", d), ("S", group.pageSize),
            ("R", count)]
        let output: MLXArray
        if group.segmentLayout == nil {
            let results = monolithic(
                [group.kSlab, group.vSlab, plan.table, indices, logicalLength, group.writeFence], template: common,
                grid: (d, h, count), threadGroup: (min(256, d), 1, 1),
                outputShapes: [[2, 1, h, count, d], [1]], outputDTypes: [group.dtype, .int32])
            output = results[0]
            // Both outputs belong to the same primitive. Its fence carries the
            // read back-edge to subsequent writes, including rollback/rewrite.
            group.writeFence = results[1]
        } else {
            let destination = MLXArray.zeros([2, 1, h, count, d], dtype: group.dtype)
            var fence = group.writeFence
            if boundEnabled() {
                let batches = bindingBatches(segmentIDs: plan.segmentIDs)
                PagedSelectedGatherInvocation.record(
                    pages: plan.table.size, segments: plan.segmentIDs.count,
                    passes: batches.count, selectedRows: count)
                // Selected indices stay on-device. Each bounded pass binds up
                // to 17 current buffers and copies only its matching rows.
                for batch in batches {
                    let bindingCount = bindingClass(for: batch.count)
                    let segments = batch.map { index -> PagedKVSegment in
                        guard let segment = group.segments[index] else {
                            preconditionFailure("selected gather names an uncommitted segment")
                        }
                        return segment
                    }
                    var storage = segments.map(\.storage)
                    storage.append(contentsOf: repeatElement(
                        storage[0], count: bindingCount - storage.count))
                    var bounds = segments.flatMap {
                        [Int32($0.pages.lowerBound), Int32($0.pages.upperBound)]
                    }
                    bounds.append(contentsOf: repeatElement(
                        Int32(0), count: 2 * (bindingCount - segments.count)))
                    var valueOffsets = segments.map { Int64($0.valueOffset) }
                    valueOffsets.append(contentsOf: repeatElement(
                        Int64(0), count: bindingCount - segments.count))
                    fence = segmentedKernel(bindings: bindingCount)(
                        storage + [plan.table, indices, logicalLength, destination, fence,
                            MLXArray(bounds), MLXArray(valueOffsets)], template: common,
                        grid: (d, h, count), threadGroup: (min(256, d), 1, 1),
                        outputShapes: [[1]], outputDTypes: [.int32])[0]
                }
            } else {
                for index in plan.segmentIDs {
                    guard let segment = group.segments[index] else {
                        preconditionFailure("selected gather names an uncommitted segment")
                    }
                    fence = segmentedLegacy(
                        [segment.storage, plan.table, indices, logicalLength, destination, fence],
                        template: common + [("FIRST", segment.pages.lowerBound),
                            ("END", segment.pages.upperBound), ("VBASE", segment.valueOffset)],
                        grid: (d, h, count), threadGroup: (min(256, d), 1, 1),
                        outputShapes: [[1]], outputDTypes: [.int32])[0]
                }
            }
            fence = complete([fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
            group.writeFence = fence
            output = depends(input: destination, dependencies: [fence])
        }
        return (output[0], output[1])
    }
}

public enum PagedSelectedGatherInvocation: Sendable {
    private static let lock = NSLock()
    private static let logger = Logger(subsystem: "darkbloom", category: "PagedSelectedGather")
    private static let diagnoseFirstPlan = Qwen4ExpEnvironment.snapshot[
        "DARKBLOOM_QWEN4_PAGED_SELECTED_GATHER_DIAGNOSTICS"] == "1"
    nonisolated(unsafe) private static var didRecord = false
    nonisolated(unsafe) private static var calls = 0

    static func record(pages: Int, segments: Int, passes: Int, selectedRows: Int) {
        guard diagnoseFirstPlan else { return }
        let first = lock.withLock {
            calls += 1
            if didRecord { return false }
            didRecord = true
            return true
        }
        if first {
            logger.info(
                "paged_selected_gather first_planned_call=1 pages=\(pages, privacy: .public) segments=\(segments, privacy: .public) passes=\(passes, privacy: .public) selected_rows=\(selectedRows, privacy: .public) max_bindings=\(PagedSelectedGather.maximumBindings, privacy: .public)")
        }
    }

    public static func snapshot() -> Int { lock.withLock { calls } }
}
