import MLX
import MLXFast

/// One submitted step owns these buffers. Dimensions are fixed before any
/// forward is built, so an increasing context cannot retain a series of grown
/// arenas under a reservation for only the last allocation.
final class PagedQuantizedStepScratch {
    struct Geometry: Hashable {
        let queryHeads: Int
        let headDim: Int
        let partitionTokens: Int
    }

    let maximumQueries: Int
    let maximumAttendLength: Int
    private(set) var arenas: [Geometry: PagedQuantizedStepArena] = [:]
    var fusedPrefillArenas: [PagedQuantizedFusedPrefillGeometry: PagedQuantizedFusedPrefillArena] = [:]

    init(maximumQueries: Int, maximumAttendLength: Int) throws {
        guard maximumQueries > 0, maximumAttendLength > 0 else {
            throw CBv2KVError.backendIneligible(reason: "invalid packed step workspace dimensions")
        }
        self.maximumQueries = maximumQueries
        self.maximumAttendLength = maximumAttendLength
    }

    func arena(pool: PagedKVPool, group: PagedKVGroup, queryHeads: Int,
               blockSize: Int, maxAttendLength: Int) throws -> PagedQuantizedStepArena {
        guard blockSize <= maximumQueries, maxAttendLength <= maximumAttendLength else {
            throw CBv2KVError.backendIneligible(reason: "packed attention exceeds its prepared step workspace")
        }
        let geometry = Geometry(queryHeads: queryHeads, headDim: group.key.headDim,
                                partitionTokens: PagedSegmentDispatchPlan.boundedPartitionTokens(
                                    PagedAttentionKernel.partitionTokens, pageSize: group.pageSize))
        if let arena = arenas[geometry] { return arena }
        let bytes = try PagedQuantizedAttentionWorkspace.sharedArenaBytes(
            blockSize: maximumQueries, queryHeads: queryHeads, headDim: geometry.headDim,
            pageSize: group.pageSize, maxAttendLength: maximumAttendLength)
        let permit = try pool.memoryAdmission?.reserveWorkspace(bytes: bytes)
        let lease = PagedQuantizedScratchLease(reservation: permit, bytes: bytes)
        let arena = PagedQuantizedStepArena(
            geometry: geometry, maximumQueries: maximumQueries,
            maximumAttendLength: maximumAttendLength, lease: lease)
        arenas[geometry] = arena
        pool.appendQuantizedScratch(lease)
        return arena
    }
}

/// Every user waits for the last merge before overwriting shared partials.
/// This dependency is independent of KV group fences, since distinct layer
/// groups and serial verification calls can share the same compute geometry.
final class PagedQuantizedStepArena {
    private static let join = MLXFast.metalKernel(
        name: "cbv2_quantized_arena_acquire", inputNames: ["previous", "last_use"],
        outputNames: ["fence"], source: "fence[0] = max(previous[0], last_use[0]) + 1;",
        ensureRowContiguous: true)

    let partials: MLXArray
    let meta: MLXArray
    let maximumQueries: Int
    let maximumPartitions: Int
    let partitionTokens: Int
    let lease: PagedQuantizedScratchLease
    private var lastUse: MLXArray?

    init(geometry: PagedQuantizedStepScratch.Geometry, maximumQueries: Int,
         maximumAttendLength: Int, lease: PagedQuantizedScratchLease) {
        self.maximumQueries = maximumQueries
        partitionTokens = geometry.partitionTokens
        maximumPartitions = (maximumAttendLength - 1) / partitionTokens + 1
        self.lease = lease
        // sharedArenaBytes checked these products before reservation/allocation.
        partials = MLXArray.zeros(
            [maximumQueries, geometry.queryHeads, maximumPartitions, geometry.headDim], dtype: .float32)
        meta = MLXArray.zeros(
            [max(8, maximumQueries * geometry.queryHeads * maximumPartitions * 2)], dtype: .float32)
        lease.retain([partials, meta])
    }

    func acquire(after previous: MLXArray) -> MLXArray {
        guard let lastUse else { return previous }
        // A real custom-kernel consumer, rather than an alias-only Depends,
        // registers both fences with Metal and inserts the required barrier.
        return Self.join([previous, lastUse], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                         outputShapes: [[1]], outputDTypes: [.int32])[0]
    }

    func recordCompletion(_ fence: MLXArray) {
        lastUse = fence
        lease.retainCompletion(fence)
    }
}

extension PagedKVPool {
    /// Engine builders declare the complete step envelope before any forward.
    /// Direct backend callers may omit this and retain per-call workspaces.
    func beginQuantizedScratch(maximumQueries: Int, maximumAttendLength: Int) throws {
        guard config.quantization != nil else { return }
        guard quantizedScratchScope == nil, pendingQuantizedScratch.isEmpty else {
            throw CBv2KVError.backendIneligible(reason: "previous packed step workspace was not detached")
        }
        quantizedScratchScope = try PagedQuantizedStepScratch(
            maximumQueries: maximumQueries, maximumAttendLength: maximumAttendLength)
    }
}
