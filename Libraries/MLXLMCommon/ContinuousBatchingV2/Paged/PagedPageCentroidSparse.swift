// PagedPageCentroidSparse.swift
//
// Zero-copy filtered-block-table Page-Centroid Sparse Full-Layer Decode for
// ContinuousBatchingV2 `PagedLayerCache`.
//
// Exploits proportional-RoPE / NoPE full-attention layers (e.g. Gemma 4
// L5/L11/L17/L23/L29, 75% NoPE semantic key dimensions):
//   - Lazily summarizes each completed 16-token full-attention page into a
//     single mean key vector `mu_p` (`6.25%` of K bytes, cached per row and
//     refreshed only once every 16 decode steps).
//   - On decode steps where any row exceeds `minActivatePages * pageSize`,
//     scores complete pages on GPU via `max_{Hkv, G}(q_g . mu_p)`, selects
//     `[sinkPages (4), topInteriorPages (default 64), recentPages (8), tail]`
//     entirely on device (zero host syncs, chaining-safe), and passes the
//     compact filtered physical page table straight into the existing
//     `pagedattention.metal` fused write+attend kernels (`HAS_WRITE = true`)
//     for both fixed and segmented pools — zero KV copying and zero changes
//     to `pagedattention.metal`.

import Foundation
import MLX
import MLXFast

/// Opt-in configuration for full-attention Page-Centroid Sparse Decode.
public struct PagedPageCentroidSparseConfig: Sendable, Equatable {
    public static let envKey = "DARKBLOOM_CBV2_PAGE_SPARSE_DECODE"

    /// Number of leading sink pages always retained.
    public var sinkPages: Int
    /// Number of trailing complete recent pages always retained (in addition to
    /// any partially filled tail page).
    public var recentPages: Int
    /// Number of highest-scoring interior pages selected per decode step (`0` = disabled).
    public var topInteriorPages: Int

    public var enabled: Bool { topInteriorPages > 0 }

    /// Minimum complete pages before sparse selection activates for a row.
    public var minActivatePages: Int {
        sinkPages + recentPages + topInteriorPages + 1
    }

    public init(sinkPages: Int = 4, recentPages: Int = 8, topInteriorPages: Int = 0) {
        self.sinkPages = max(0, sinkPages)
        self.recentPages = max(0, recentPages)
        self.topInteriorPages = max(0, topInteriorPages)
    }

    public static let off = PagedPageCentroidSparseConfig(topInteriorPages: 0)

    public static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> PagedPageCentroidSparseConfig {
        guard
            let raw = environment[envKey]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased(),
            !raw.isEmpty
        else { return .off }
        switch raw {
        case "0", "false", "no", "off":
            return .off
        case "1", "true", "yes", "on":
            return PagedPageCentroidSparseConfig(sinkPages: 4, recentPages: 8, topInteriorPages: 64)
        default:
            if let k = Int(raw), k >= 16 {
                return PagedPageCentroidSparseConfig(
                    sinkPages: 4, recentPages: 8, topInteriorPages: min(k, 256))
            }
            return .off
        }
    }

    /// Process-cached environment configuration (read once per process).
    public static let current: PagedPageCentroidSparseConfig = resolve()
}

enum PagedPageCentroidSparse {
    /// Minimum element count for Int32 buffers passed to custom Metal kernels so MLX
    /// keeps a single `device` address-space binding across all batch sizes and budgets.
    private static let minMetalDeviceBufferElements = 8

    private static func paddedRightZeros(_ array: MLXArray, to length: Int) -> MLXArray {
        let shortfall = length - array.dim(0)
        guard shortfall > 0 else { return array }
        return concatenated([array, MLXArray.zeros([shortfall], dtype: .int32)], axis: 0)
    }

    /// Attempt zero-copy filtered-block-table sparse full-layer decode.
    /// Returns `nil` to fall through cleanly to exact dense paged decode whenever
    /// disabled, ineligible, short-context, or exceeding the segmented binding cap.
    static func tryDecode(
        config: PagedPageCentroidSparseConfig,
        queries: MLXArray,
        newKeys: MLXArray?,
        newValues: MLXArray?,
        writeTargets: [(page: Int32, slot: Int)]?,
        rows: [PagedSequenceKV],
        group: PagedKVGroup,
        qPosFromEnd: Int,
        sinks: MLXArray?,
        params: MLXArray,
        softcap: Bool,
        kernelSource: String,
        writeValidation: CBv2PagedKVWriteValidation
    ) -> MLXArray? {
        guard config.enabled,
            qPosFromEnd == 0,
            group.key.quantization == nil,
            !group.key.isAsymmetric,
            !rows.isEmpty,
            rows.allSatisfy({ $0.windowSize == nil && $0.baseOffset == 0 && !$0.isReleased })
        else { return nil }

        let pageSize = group.pageSize
        let attendRanges = rows.map(\.decodeAttendRange)
        guard attendRanges.allSatisfy({ $0.start == 0 && $0.length > 0 }) else { return nil }
        let attendLengths = attendRanges.map(\.length)
        let minPages = config.minActivatePages
        guard attendLengths.contains(where: { (($0 - 1) / pageSize) >= minPages }) else {
            return nil
        }

        let b = rows.count
        let q3D = queries.ndim == 4 ? queries.squeezed(axis: 2) : queries
        let qh = q3D.dim(1)
        let d = q3D.dim(2)
        let kvh = group.key.kvHeads
        guard d == group.key.headDim,
            qh % kvh == 0,
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let gqa = qh / kvh
        guard PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: gqa) != nil else {
            return nil
        }

        let sinkN = config.sinkPages
        let recentN = config.recentPages
        let topK = config.topInteriorPages
        let keptComplete = sinkN + topK + recentN

        var sparseAttendLengths: [Int] = []
        sparseAttendLengths.reserveCapacity(b)
        var maxSparsePages = 0
        for attendLen in attendLengths {
            let completePages = (attendLen - 1) / pageSize
            if completePages >= minPages {
                let tailTokens = attendLen - completePages * pageSize  // always 1...pageSize
                sparseAttendLengths.append(keptComplete * pageSize + tailTokens)
                maxSparsePages = max(maxSparsePages, keptComplete + 1)
            } else {
                sparseAttendLengths.append(attendLen)
                maxSparsePages = max(maxSparsePages, completePages + 1)
            }
        }
        guard maxSparsePages > 0 else { return nil }
        let maxSparseAttendLength = max(1, sparseAttendLengths.max() ?? 1)

        // Pre-validate and prepare segmented pool metadata BEFORE touching `ensureCentroids`
        // or `group.writeFence`, so an unsupported segment count or partition width never
        // leaves behind lazy centroid chains or fence advances.
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        let segmentedPrep: SegmentedDispatchPrep?
        if let layout = group.segmentLayout {
            guard
                let prep = prepareSegmentedDispatch(
                    layout: layout, rows: rows, writeTargets: writeTargets, group: group,
                    headDim: d, gqa: gqa, maxAttendLength: maxSparseAttendLength)
            else { return nil }
            segmentedPrep = prep
        } else {
            segmentedPrep = nil
        }
        #else
        if group.segmentLayout != nil { return nil }
        #endif

        // Float16 dot products over headDim=512 can overflow 65504 without scaling;
        // promote float16 to float32 for centroid scoring while keeping bfloat16 native.
        let scoreInFloat32 = group.dtype == .float16
        let qTyped: MLXArray =
            if scoreInFloat32 {
                q3D.asType(.float32)
            } else {
                q3D.dtype == group.dtype ? q3D : q3D.asType(group.dtype)
            }

        // 1. Refresh per-row cached page centroids lazily and select logical pages on GPU.
        var perRowSelectedLogical: [MLXArray] = []
        perRowSelectedLogical.reserveCapacity(b)
        let sinkIdx = sinkN > 0 ? MLXArray.arange(sinkN, dtype: .int32) : nil

        for i in 0 ..< b {
            let row = rows[i]
            let attendLen = attendLengths[i]
            // Exclude the tail page containing `attendLen - 1` (which this HAS_WRITE
            // dispatch may still be writing) so centroids only read frozen complete pages.
            let completePages = (attendLen - 1) / pageSize

            if completePages >= minPages {
                guard
                    let centroids = ensureCentroids(
                        for: row, completePages: completePages, group: group)
                else { return nil }
                let qRow = qTyped[i].reshaped(kvh, gqa, d)  // [Hkv, G, D]
                let cScore = scoreInFloat32 ? centroids.asType(.float32) : centroids
                let scores = matmul(qRow, cScore.transposed(0, 2, 1)).max(axes: [0, 1])  // [P]

                let interiorStart = sinkN
                let interiorEnd = completePages - recentN
                let interiorScores = scores[interiorStart ..< interiorEnd]
                let topRel =
                    argPartition(-interiorScores, kth: topK - 1)[0 ..< topK].asType(.int32)
                    + Int32(interiorStart)

                var parts: [MLXArray] = []
                parts.reserveCapacity(4)
                if let sinkIdx { parts.append(sinkIdx) }
                parts.append(topRel)
                if recentN > 0 {
                    parts.append(MLXArray.arange(interiorEnd, completePages, dtype: .int32))
                }
                parts.append(MLXArray([Int32(completePages)]))
                perRowSelectedLogical.append(concatenated(parts, axis: 0))
            } else {
                perRowSelectedLogical.append(MLXArray.arange(completePages + 1, dtype: .int32))
            }
        }

        // 2. Dispatch into the existing fused write+attend kernels with zero KV copy.
        let out: MLXArray
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        if let segmentedPrep {
            out = dispatchSegmentedSparse(
                queries: q3D, newKeys: newKeys, newValues: newValues,
                writeTargets: writeTargets, group: group, prep: segmentedPrep,
                perRowSelectedLogical: perRowSelectedLogical,
                sparseAttendLengths: sparseAttendLengths,
                maxSparsePages: maxSparsePages,
                sinks: sinks, params: params, softcap: softcap,
                kernelSource: kernelSource)
        } else {
            out = dispatchFixedSparse(
                queries: q3D, newKeys: newKeys, newValues: newValues,
                writeTargets: writeTargets, rows: rows, group: group,
                perRowSelectedLogical: perRowSelectedLogical,
                sparseAttendLengths: sparseAttendLengths,
                maxSparsePages: maxSparsePages,
                sinks: sinks, params: params, softcap: softcap,
                kernelSource: kernelSource, writeValidation: writeValidation)
        }
        #else
        out = dispatchFixedSparse(
            queries: q3D, newKeys: newKeys, newValues: newValues,
            writeTargets: writeTargets, rows: rows, group: group,
            perRowSelectedLogical: perRowSelectedLogical,
            sparseAttendLengths: sparseAttendLengths,
            maxSparsePages: maxSparsePages,
            sinks: sinks, params: params, softcap: softcap,
            kernelSource: kernelSource, writeValidation: writeValidation)
        #endif

        var result = out.expandedDimensions(axis: 2)
        if result.dtype != queries.dtype {
            result = result.asType(queries.dtype)
        }
        return result
    }

    // MARK: - Lazy Per-Row Centroid Maintenance

    /// Returns `[kvHeads, completePages, headDim]` mean key centroids in native KV
    /// dtype for `row`, reading only pages newly completed since `row.sparseCentroidPages`.
    /// Publishes the fence back-edge onto `group.writeFence` to stay strictly ordered
    /// with in-place slab writes.
    private static func ensureCentroids(
        for row: PagedSequenceKV,
        completePages: Int,
        group: PagedKVGroup
    ) -> MLXArray? {
        guard completePages > 0, completePages <= row.table.count else { return nil }
        if row.sparseCentroidPages >= completePages, let cached = row.sparsePageCentroids {
            return cached.dim(1) == completePages ? cached : cached[0..., 0 ..< completePages, 0...]
        }

        let startPage = row.sparsePageCentroids != nil ? row.sparseCentroidPages : 0
        guard startPage < completePages else { return nil }
        let newPages = Array(row.table[startPage ..< completePages])

        func gatherPageMeans(from kStorage: MLXArray, pages: [Int32]) -> MLXArray {
            let idx = MLXArray(pages) + group.writeFence * 0
            return take(kStorage, idx, axis: 0).asType(.float32).mean(axis: 2).asType(group.dtype)
        }

        let newMeansPHD: MLXArray
        if let layout = group.segmentLayout {
            // Segmented pool: each segment stores `[2, localPages, kvHeads, pageSize, headDim]`
            // with K at index 0. Group new pages by segment, mean along slot axis, restore order.
            var perSegment: [Int: [(order: Int, local: Int32)]] = [:]
            for (order, phys) in newPages.enumerated() {
                let segID = layout.segmentIndex(page: phys)
                guard let seg = group.segments[segID] else { return nil }
                let local = Int32(Int(phys) - seg.pages.lowerBound)
                perSegment[segID, default: []].append((order, local))
            }
            if perSegment.count == 1, let (segID, items) = perSegment.first,
                let seg = group.segments[segID]
            {
                newMeansPHD = gatherPageMeans(from: seg.storage[0], pages: items.map(\.local))
            } else {
                var meanChunks: [MLXArray] = []
                var orderIndices: [Int32] = []
                meanChunks.reserveCapacity(perSegment.count)
                orderIndices.reserveCapacity(newPages.count)
                for (segID, items) in perSegment {
                    guard let seg = group.segments[segID] else { return nil }
                    meanChunks.append(
                        gatherPageMeans(from: seg.storage[0], pages: items.map(\.local)))
                    orderIndices.append(contentsOf: items.map { Int32($0.order) })
                }
                let unsorted = concatenated(meanChunks, axis: 0)
                var inverseOrder = [Int32](repeating: 0, count: newPages.count)
                for (pos, origOrder) in orderIndices.enumerated() {
                    inverseOrder[Int(origOrder)] = Int32(pos)
                }
                newMeansPHD = take(unsorted, MLXArray(inverseOrder), axis: 0)
            }
        } else {
            newMeansPHD = gatherPageMeans(from: group.kSlab, pages: newPages)
        }
        let newMeans = newMeansPHD.transposed(1, 0, 2)  // [Hkv, deltaP, D]

        // Publish read back-edge onto `group.writeFence` so later in-place writes
        // cannot race with this lazy centroid read.
        let edge = newMeans.asType(.float32).sum()
        group.writeFence = group.writeFence + edge.asType(group.writeFence.dtype) * 0

        let combined: MLXArray
        if startPage > 0, let existing = row.sparsePageCentroids {
            let prefix =
                existing.dim(1) == startPage ? existing : existing[0..., 0 ..< startPage, 0...]
            combined = concatenated([prefix, newMeans], axis: 1)
        } else {
            combined = newMeans
        }
        row.sparsePageCentroids = combined
        row.sparseCentroidPages = completePages
        return combined
    }

    private static func ensureFixedTableGPU(for row: PagedSequenceKV) -> MLXArray {
        if row.sparseTableVersion == row.tableVersion, let cached = row.sparsePhysTableGPU {
            return cached
        }
        let arr = MLXArray(row.table)
        row.sparsePhysTableGPU = arr
        row.sparseTableVersion = row.tableVersion
        return arr
    }

    private static func ensureSegmentedTableGPU(
        for row: PagedSequenceKV,
        layout: PagedKVSegmentLayout,
        group: PagedKVGroup
    ) -> (segIDs: Set<Int>, rawSegsGPU: MLXArray, localsGPU: MLXArray)? {
        if row.sparseTableVersion == row.tableVersion,
            let rawSegsGPU = row.sparseSegRawIDsGPU,
            let localsGPU = row.sparseSegLocalsGPU
        {
            return (row.sparseSegIDs, rawSegsGPU, localsGPU)
        }
        let table = row.table
        var segSet = Set<Int>()
        var rawSegs: [Int32] = []
        var locals: [Int32] = []
        rawSegs.reserveCapacity(table.count)
        locals.reserveCapacity(table.count)
        for phys in table {
            let segID = layout.segmentIndex(page: phys)
            guard let seg = group.segments[segID] else { return nil }
            segSet.insert(segID)
            rawSegs.append(Int32(segID))
            locals.append(Int32(Int(phys) - seg.pages.lowerBound))
        }
        let rawSegsGPU = MLXArray(rawSegs)
        let localsGPU = MLXArray(locals)
        row.sparseSegIDs = segSet
        row.sparseSegRawIDsGPU = rawSegsGPU
        row.sparseSegLocalsGPU = localsGPU
        row.sparseTableVersion = row.tableVersion
        return (segSet, rawSegsGPU, localsGPU)
    }

    // MARK: - Fixed Pool Dispatch (Zero-Copy Filtered Block Table)

    private static func dispatchFixedSparse(
        queries: MLXArray,
        newKeys: MLXArray?,
        newValues: MLXArray?,
        writeTargets: [(page: Int32, slot: Int)]?,
        rows: [PagedSequenceKV],
        group: PagedKVGroup,
        perRowSelectedLogical: [MLXArray],
        sparseAttendLengths: [Int],
        maxSparsePages: Int,
        sinks: MLXArray?,
        params: MLXArray,
        softcap: Bool,
        kernelSource: String,
        writeValidation: CBv2PagedKVWriteValidation
    ) -> MLXArray {
        // Pad table columns to >= 8 so MLX binds `tables` in Metal `device` address space
        // (matching PagedAttentionKernel.decode's `tables.dim(1) >= 8` contract).
        let paddedTableCols = max(minMetalDeviceBufferElements, maxSparsePages)
        let perRowPhysTables = (0 ..< rows.count).map { i in
            let rowTableGPU = ensureFixedTableGPU(for: rows[i])
            let phys = take(rowTableGPU, perRowSelectedLogical[i], axis: 0)
            return paddedRightZeros(phys, to: paddedTableCols)
        }
        let sparseTables = stacked(perRowPhysTables, axis: 0)  // [B, paddedTableCols]

        let seqRows = (0 ..< rows.count).map { i in
            PagedAttentionKernel.SeqInfoRow(
                attendStart: 0,
                attendLength: sparseAttendLengths[i],
                tableLength: paddedTableCols,
                writePage: writeTargets?[i].page ?? 0,
                writeSlot: writeTargets?[i].slot ?? 0)
        }
        let (seqinfo, maxAttendLength) = PagedAttentionKernel.seqinfo(seqRows)

        do {
            let (out, nextFence) = try PagedAttentionKernel.decode(
                queries: queries,
                newKeys: newKeys,
                newValues: newValues,
                kSlab: group.kSlab,
                vSlab: group.vSlab,
                tables: sparseTables,
                seqinfo: seqinfo,
                maxAttendLength: maxAttendLength,
                sinks: sinks,
                params: params,
                softcap: softcap,
                pageSize: group.pageSize,
                writeFence: group.writeFence,
                kernelSource: kernelSource)
            if let nextFence {
                group.writeFence = nextFence
            }
            return out
        } catch {
            writeValidation.record(error)
            return queries
        }
    }

    // MARK: - Segmented Pool Dispatch (Pure-GPU Record Packing, Zero Host Sync)

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    private struct SegmentedDispatchPrep {
        let cachedRows: [(rawSegsGPU: MLXArray, localsGPU: MLXArray)]
        let segments: [PagedKVSegment]
        let bindingClass: Int
        let segLUT: MLXArray
        let writeMeta: [(binding: Int32, local: Int32)]
        let partitionTokens: Int
        let pagesPerPart: Int
        let maxPartitions: Int
    }

    private static func prepareSegmentedDispatch(
        layout: PagedKVSegmentLayout,
        rows: [PagedSequenceKV],
        writeTargets: [(page: Int32, slot: Int)]?,
        group: PagedKVGroup,
        headDim d: Int,
        gqa: Int,
        maxAttendLength: Int
    ) -> SegmentedDispatchPrep? {
        var segSet = Set<Int>()
        var cachedRows: [(rawSegsGPU: MLXArray, localsGPU: MLXArray)] = []
        cachedRows.reserveCapacity(rows.count)
        for row in rows {
            guard let cached = ensureSegmentedTableGPU(for: row, layout: layout, group: group)
            else { return nil }
            segSet.formUnion(cached.segIDs)
            cachedRows.append((cached.rawSegsGPU, cached.localsGPU))
        }
        if let writeTargets {
            for target in writeTargets {
                segSet.insert(layout.segmentIndex(page: target.page))
            }
        }
        let uniqueSegIDs = segSet.sorted()
        guard !uniqueSegIDs.isEmpty,
            uniqueSegIDs.count <= PagedSegmentDispatchPlan.maximumBindings,
            let bindingClass = PagedSegmentDispatchPlan.bindingClasses.first(where: {
                $0 >= uniqueSegIDs.count
            })
        else { return nil }

        let maxSegID = uniqueSegIDs.last ?? 0
        var segLUTHost = [Int32](repeating: 0, count: maxSegID + 1)
        var segToBinding: [Int: Int32] = [:]
        segToBinding.reserveCapacity(uniqueSegIDs.count)
        var segments: [PagedKVSegment] = []
        segments.reserveCapacity(uniqueSegIDs.count)
        for (bindingIdx, segID) in uniqueSegIDs.enumerated() {
            guard let seg = group.segments[segID] else { return nil }
            segToBinding[segID] = Int32(bindingIdx)
            segLUTHost[segID] = Int32(bindingIdx)
            segments.append(seg)
        }

        var writeMeta: [(binding: Int32, local: Int32)] = []
        writeMeta.reserveCapacity(rows.count)
        for i in 0 ..< rows.count {
            if let wt = writeTargets?[i] {
                let wSegID = layout.segmentIndex(page: wt.page)
                guard let wb = segToBinding[wSegID], let wSeg = group.segments[wSegID] else {
                    return nil
                }
                writeMeta.append((wb, Int32(Int(wt.page) - wSeg.pages.lowerBound)))
            } else {
                writeMeta.append((0, 0))
            }
        }

        let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: d, gqa: gqa)
        let splits = gqa / hpt
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: maxAttendLength, batch: rows.count, kvHeads: group.key.kvHeads,
                headSplits: splits, pageSize: group.pageSize),
            pageSize: group.pageSize)
        let pagesPerPart = ptok / group.pageSize
        guard pagesPerPart >= 1,
            pagesPerPart <= PagedSegmentDispatchPlan.maximumBindings
        else { return nil }
        let maxParts = (maxAttendLength + ptok - 1) / ptok

        return SegmentedDispatchPrep(
            cachedRows: cachedRows,
            segments: segments,
            bindingClass: bindingClass,
            segLUT: MLXArray(segLUTHost),
            writeMeta: writeMeta,
            partitionTokens: ptok,
            pagesPerPart: pagesPerPart,
            maxPartitions: maxParts)
    }

    private static let packRecordsKernel = MLXFast.metalKernel(
        name: "cbv2_paged_segment_pack_sparse_records",
        inputNames: ["sel_bindings", "sel_locals", "row_meta", "cfg"],
        outputNames: ["records"],
        source: """
            const uint wid = thread_position_in_grid.x;
            const uint B = uint(cfg[0]);
            const uint max_parts = uint(cfg[1]);
            const uint pages_per_part = uint(cfg[2]);
            const uint max_sparse_pages = uint(cfg[3]);
            const uint stride = uint(cfg[4]);
            if (wid >= B * max_parts) { return; }

            const uint b = wid / max_parts;
            const uint part = wid % max_parts;
            const uint row_pages = uint(row_meta[b * 4 + 0]);
            const int write_binding = row_meta[b * 4 + 1];
            const int write_local = row_meta[b * 4 + 2];

            const uint first_logical = part * pages_per_part;
            const uint remaining = first_logical < row_pages ? (row_pages - first_logical) : 0u;
            const uint count = remaining < pages_per_part ? remaining : pages_per_part;

            const size_t base = size_t(wid) * size_t(stride);
            records[base + 0] = int(b);
            records[base + 1] = int(part);
            records[base + 2] = int(first_logical);
            records[base + 3] = int(count);
            records[base + 4] = write_binding;
            records[base + 5] = write_local;
            records[base + 6] = int(max_parts);
            records[base + 7] = 0;

            const size_t sel_base = size_t(b) * size_t(max_sparse_pages) + size_t(first_logical);
            const uint max_pairs = (stride - 8u) / 2u;
            for (uint j = 0u; j < max_pairs; ++j) {
                if (j < count) {
                    records[base + 8u + 2u * j] = sel_bindings[sel_base + j];
                    records[base + 8u + 2u * j + 1u] = sel_locals[sel_base + j];
                } else {
                    records[base + 8u + 2u * j] = 0;
                    records[base + 8u + 2u * j + 1u] = 0;
                }
            }
            """,
        ensureRowContiguous: true
    )

    private static func dispatchSegmentedSparse(
        queries: MLXArray,
        newKeys: MLXArray?,
        newValues: MLXArray?,
        writeTargets: [(page: Int32, slot: Int)]?,
        group: PagedKVGroup,
        prep: SegmentedDispatchPrep,
        perRowSelectedLogical: [MLXArray],
        sparseAttendLengths: [Int],
        maxSparsePages: Int,
        sinks: MLXArray?,
        params: MLXArray,
        softcap: Bool,
        kernelSource: String
    ) -> MLXArray {
        let b = prep.cachedRows.count
        var selBindingsRows: [MLXArray] = []
        var selLocalsRows: [MLXArray] = []
        var rowMetaHost: [Int32] = []
        selBindingsRows.reserveCapacity(b)
        selLocalsRows.reserveCapacity(b)
        rowMetaHost.reserveCapacity(max(minMetalDeviceBufferElements, b * 4))

        // Pad selection columns, `rowMeta`, and `cfg` to >= 8 elements so
        // `packRecordsKernel` keeps a single Metal `device` binding across `B = 1` and `B >= 2`.
        let paddedCols = max(minMetalDeviceBufferElements, maxSparsePages)
        for i in 0 ..< b {
            let selLogical = perRowSelectedLogical[i]
            let rawSegs = take(prep.cachedRows[i].rawSegsGPU, selLogical, axis: 0)
            let sb = take(prep.segLUT, rawSegs, axis: 0)
            let sl = take(prep.cachedRows[i].localsGPU, selLogical, axis: 0)
            let rowPages = sb.dim(0)
            selBindingsRows.append(paddedRightZeros(sb, to: paddedCols))
            selLocalsRows.append(paddedRightZeros(sl, to: paddedCols))
            let wm = prep.writeMeta[i]
            rowMetaHost.append(contentsOf: [Int32(rowPages), wm.binding, wm.local, 0])
        }
        if rowMetaHost.count < minMetalDeviceBufferElements {
            rowMetaHost.append(
                contentsOf: repeatElement(
                    Int32(0), count: minMetalDeviceBufferElements - rowMetaHost.count))
        }

        let selBindings = stacked(selBindingsRows, axis: 0)  // [B, paddedCols]
        let selLocals = stacked(selLocalsRows, axis: 0)  // [B, paddedCols]
        let rowMeta = MLXArray(rowMetaHost)
        let stride = PagedSegmentDispatchPlan.recordStride
        let cfg = MLXArray([
            Int32(b), Int32(prep.maxPartitions), Int32(prep.pagesPerPart),
            Int32(paddedCols), Int32(stride), 0, 0, 0,
        ])
        let workCount = b * prep.maxPartitions
        let records = packRecordsKernel(
            [selBindings, selLocals, rowMeta, cfg],
            grid: (workCount, 1, 1),
            threadGroup: (min(64, workCount), 1, 1),
            outputShapes: [[workCount * stride]],
            outputDTypes: [.int32])[0]

        let seqRows = (0 ..< b).map { i in
            PagedAttentionKernel.SeqInfoRow(
                attendStart: 0,
                attendLength: sparseAttendLengths[i],
                tableLength: paddedCols,
                writePage: 0,
                writeSlot: writeTargets?[i].slot ?? 0)
        }
        let (seqinfo, _) = PagedAttentionKernel.seqinfo(seqRows)

        return PagedSegmentAttention.decodeWithCustomRecords(
            queries: queries,
            newKeys: newKeys,
            newValues: newValues,
            group: group,
            segments: prep.segments,
            bindingClass: prep.bindingClass,
            records: records,
            workCount: workCount,
            maxPartitions: prep.maxPartitions,
            partitionTokens: prep.partitionTokens,
            seqinfo: seqinfo,
            sinks: sinks,
            params: params,
            softcap: softcap,
            source: kernelSource)
    }
    #endif
}
