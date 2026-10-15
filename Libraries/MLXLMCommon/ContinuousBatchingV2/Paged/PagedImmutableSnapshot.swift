// Copyright © 2026 Eigen Labs.

import MLX

enum PagedImmutableSnapshotError: Error, Equatable {
    case ineligible
    case stale
}

/// Engine-queue-owned page references. Only row issuance can create one; adopted
/// rows retain their own references and enable COW before exposing any write.
final class PagedImmutableSnapshot {
    let pool: PagedKVPool
    let group: PagedKVGroupKey
    let handles: [PagedKVPageHandle]
    let tokenCount: Int
    private let metadata: CBv2CheckpointReservation

    init(
        pool: PagedKVPool, group: PagedKVGroupKey,
        handles: [PagedKVPageHandle], tokenCount: Int
    ) throws {
        guard let admission = pool.memoryAdmission else {
            throw PagedImmutableSnapshotError.ineligible
        }
        let metadata = try admission.reserveTransient(bytes: (64 << 10) + handles.count * 64)
        try pool.reserve([group: handles.count])
        do {
            try pool.materializeReservedSegments()
            guard pool.retainPages(handles) else { throw PagedImmutableSnapshotError.stale }
        } catch {
            pool.unreserve([group: handles.count])
            throw error
        }
        self.pool = pool
        self.group = group
        self.handles = handles
        self.tokenCount = tokenCount
        self.metadata = metadata
        pool.noteImmutableSnapshotPages(handles.count)
    }

    deinit {
        // Readers retain this handle until their fence; no page can be evicted
        // while its refcount is positive. COW copies finish before replacing it.
        pool.freePages(group: group, pages: handles.map(\.page))
        pool.unreserve([group: handles.count])
        pool.noteImmutableSnapshotPages(-handles.count)
        metadata.release()
    }
}

extension PagedKVPool {
    func immutableCopyScratchBytes(group key: PagedKVGroupKey) throws -> Int {
        // Native gather destinations, contiguous writer inputs, and metadata.
        let page = try CBv2CheckpointAllocationFootprint.bound(group(key).pageBytes)
        let pair = try CBv2CheckpointAllocationFootprint.add(page, page)
        return try CBv2CheckpointAllocationFootprint.add(
            64 << 10, CBv2CheckpointAllocationFootprint.add(pair, pair))
    }

    func cloneImmutablePage(_ source: PagedKVPageHandle, initializedTokens: Int) throws -> Int32 {
        guard isValid(source), (refCount(for: source) ?? 0) > 1,
            (0 ... config.pageSize).contains(initializedTokens)
        else { throw PagedImmutableSnapshotError.stale }
        let page = allocatePage(group: source.group)
        do {
            if initializedTokens > 0 {
                let copied = gather(
                    group: source.group, pages: [source.page], firstSlot: 0,
                    count: initializedTokens)
                let slots = (0 ..< initializedTokens).map {
                    page * Int32(config.pageSize) + Int32($0)
                }
                writeTokens(
                    group: source.group, slots: slots,
                    keys: copied.keys.squeezed(axis: 0), values: copied.values.squeezed(axis: 0))
                // The row's lifetime scratch charge covers this bounded copy.
                // Complete it before old-page release or another page clone.
                try finishImmutablePageWork([
                    copied.keys, copied.values, group(source.group).writeFence,
                ])
                try writeValidation.check()
            }
            return page
        } catch {
            freePages(group: source.group, pages: [page])
            throw error
        }
    }

    func finishImmutablePageWork(_ roots: [MLXArray]) throws {
        try withError { fault in
            eval(roots)
            StreamOrDevice.default.stream.synchronize()
            try fault.check()
        }
    }
}
