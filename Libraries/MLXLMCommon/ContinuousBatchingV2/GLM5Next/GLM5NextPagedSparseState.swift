// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// A shared native segmented store, not an attention-model capability. Creation,
/// use and retirement are engine-queue confined. Duplicate latent V is charged.
public final class GLM5NextPagedSparseStore {
    public let geometry: GLM5NextSparseStateGeometry
    let pool: PagedKVPool
    let admission: AdmissionV2

    /// Constructs one complete pool for a previously unbound admission owner.
    /// This cannot attach a second pool to an already constructed EngineV2:
    /// that integration must borrow its single backend pool/floor instead.
    public init(
        geometry: GLM5NextSparseStateGeometry, admission: AdmissionV2,
        capacityBytes: Int, segmentBytes: Int, maximumBufferBytes: Int
    ) throws {
        self.geometry = geometry
        self.admission = admission
        let kinds = [geometry.latentDimension, geometry.indexDimension].map {
            CBv2LayerKind(attention: .full, headDim: $0, kvHeads: 1, queryHeads: 1)
        }
        let constructed = try admission.constructWithBackendPhysicalFloor {
            try PagedKVPool(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: capacityBytes, dtype: geometry.latentDType,
                    maxPrefillChunk: geometry.maximumStagedTokens,
                    nominalMaxSequenceLength: geometry.maximumTokens,
                    maxBufferLength: maximumBufferBytes, segmentSizeBytes: segmentBytes,
                    layerDTypes: [geometry.latentDType, geometry.indexDType]))
        }
        pool = constructed.value
        pool.memoryAdmission = admission
        pool.physicalLease = constructed.lease
    }

    public var bytesMaterialized: Int { pool.bytesMaterialized }
    public var bytesInUse: Int { pool.bytesInUse }
    public var bytesReserved: Int { pool.bytesReserved }

    public func makeState(binding: GLM5NextSparseStateBinding) throws -> GLM5NextPagedSparseState {
        try GLM5NextPagedSparseState(store: self, binding: binding)
    }

    func makeRows() throws -> [PagedSequenceKV] {
        try pool.writeValidation.check()
        var needs: [PagedKVGroupKey: Int] = [:]
        let pages = (geometry.maximumTokens - 1) / pool.config.pageSize + 1
        for index in 0 ..< 2 { needs[pool.groupKey(forLayer: index), default: 0] += pages }
        try pool.reserve(needs)
        do { try pool.materializeReservedSegments() } catch {
            pool.unreserve(needs)
            throw error
        }
        return (0 ..< 2).map { index in
            PagedSequenceKV(
                pool: pool, kind: pool.layerKinds[index], groupKey: pool.groupKey(forLayer: index),
                maxLength: geometry.maximumTokens, reservedPages: pages)
        }
    }

    func finish(_ roots: [MLXArray] = []) throws {
        try pool.finishImmutablePageWork(roots + pool.groupKeys.map { pool.group($0).writeFence })
        try pool.writeValidation.check()
    }

    func readScratchBytes(count: Int, width: Int, dtype: DType, copies: Int, metadata: Bool) throws
        -> Int
    {
        guard count > 0, count <= max(geometry.maximumTokens, 2051) + pool.config.pageSize,
            width > 0, copies > 0
        else {
            throw GLM5NextSparseStateError.invalidGeometry
        }
        let (elements, overflow) = count.multipliedReportingOverflow(by: width)
        let (logical, byteOverflow) = elements.multipliedReportingOverflow(by: dtype.size)
        guard !overflow, !byteOverflow else { throw GLM5NextSparseStateError.invalidGeometry }
        var bytes = 64 << 10
        let allocation = try CBv2CheckpointAllocationFootprint.bound(logical)
        for _ in 0 ..< copies {
            bytes = try CBv2CheckpointAllocationFootprint.add(bytes, allocation)
        }
        if metadata {
            for size in [DType.bool.size, DType.int32.size] {
                bytes = try CBv2CheckpointAllocationFootprint.add(
                    bytes, CBv2CheckpointAllocationFootprint.bound(count * size))
            }
        }
        // Device/host page tables and segment bindings, including the copy selector.
        return try CBv2CheckpointAllocationFootprint.add(
            bytes, (geometry.maximumTokens / pool.config.pageSize + 1) * 16)
    }

    func metadataAllocationBytes(count: Int) throws -> Int {
        guard count >= 0, count <= geometry.maximumTokens else {
            throw GLM5NextSparseStateError.invalidPosition
        }
        var bytes = (64 << 10) + count * MemoryLayout<Int32>.stride
        for size in [DType.bool.size, DType.int32.size] where count > 0 {
            bytes = try CBv2CheckpointAllocationFootprint.add(
                bytes, CBv2CheckpointAllocationFootprint.bound(count * size))
        }
        return try CBv2CheckpointAllocationFootprint.add(
            bytes, CBv2CheckpointAllocationFootprint.bound(1))
    }
}

/// One request/layer. Page writes and bounded COW copies finish at each public
/// operation boundary. This correctness-first owner never expands historical KV.
public final class GLM5NextPagedSparseState {
    public struct Transaction: Equatable {
        fileprivate let owner: UUID
        fileprivate let serial: UInt64
    }

    fileprivate final class Arrays {
        var values: [MLXArray]
        let reservation: CBv2CheckpointReservation
        init(_ values: [MLXArray], reservation: CBv2CheckpointReservation) {
            self.values = values
            self.reservation = reservation
        }
        deinit {
            values.removeAll()
            reservation.release()
        }
    }

    public final class IndexView {
        public let tokenCount: Int
        private let storage: Arrays
        fileprivate init(_ storage: Arrays, count: Int) {
            self.storage = storage
            tokenCount = count
        }
        public var indexKeys: MLXArray { storage.values[0] }
        public var poolGates: MLXArray { storage.values[1] }
        public var validity: MLXArray { storage.values[2] }
        public var positions: MLXArray { storage.values[3] }
    }

    public final class LatentView {
        private let storage: Arrays
        fileprivate init(_ storage: Arrays) { self.storage = storage }
        public var latents: MLXArray { storage.values[0] }
    }

    public final class Checkpoint {
        public let tokenCount: Int
        fileprivate let store: GLM5NextPagedSparseStore
        fileprivate let binding: GLM5NextSparseStateBinding
        fileprivate let pages: [PagedImmutableSnapshot]
        fileprivate let metadata: Arrays?
        fileprivate init(_ state: GLM5NextPagedSparseState, pages: [PagedImmutableSnapshot]) {
            tokenCount = state.committedTokenCount
            store = state.store
            binding = state.binding
            metadata = state.metadata
            self.pages = pages
        }
    }

    public let binding: GLM5NextSparseStateBinding
    public private(set) var committedTokenCount = 0
    public private(set) var stagedTokenCount = 0
    public private(set) var isClosed = false
    private let store: GLM5NextPagedSparseStore
    private var rows: [PagedSequenceKV]
    private var metadata: Arrays?
    private var transactionMetadata: Arrays?
    private let id = UUID()
    private var serial: UInt64 = 0
    private var active: Transaction?

    fileprivate init(store: GLM5NextPagedSparseStore, binding: GLM5NextSparseStateBinding) throws {
        self.store = store
        self.binding = binding
        rows = try store.makeRows()
    }

    public func begin(at position: Int) throws -> Transaction {
        try requireOpen()
        guard active == nil else { throw GLM5NextSparseStateError.invalidTransaction }
        guard position == committedTokenCount else {
            throw GLM5NextSparseStateError.invalidPosition
        }
        guard serial < UInt64.max else { throw GLM5NextSparseStateError.invalidTransaction }
        serial += 1
        let transaction = Transaction(owner: id, serial: serial)
        rows.forEach { $0.beginSpeculativeWrite() }
        active = transaction
        transactionMetadata = metadata
        return transaction
    }

    public func stage(
        _ transaction: Transaction, at position: Int,
        latents: MLXArray, indexKeys: MLXArray, poolGates: MLXArray, validity: MLXArray
    ) throws {
        try require(transaction)
        let geometry = store.geometry
        guard latents.ndim == 3 else { throw GLM5NextSparseStateError.invalidPayload("latents") }
        let count = latents.dim(1)
        guard position == committedTokenCount + stagedTokenCount, count > 0,
            count <= geometry.maximumTokens - position,
            count <= geometry.maximumStagedTokens - stagedTokenCount
        else { throw GLM5NextSparseStateError.invalidPosition }
        guard latents.shape == [1, count, geometry.latentDimension],
            latents.dtype == geometry.latentDType,
            indexKeys.shape == [1, count, geometry.indexDimension],
            indexKeys.dtype == geometry.indexDType,
            poolGates.shape == indexKeys.shape, poolGates.dtype == geometry.indexDType,
            validity.shape == [1, count], validity.dtype == .bool
        else { throw GLM5NextSparseStateError.invalidPayload("paged append") }
        let scratch = try store.admission.reserveTransient(
            bytes: geometry.allocationBytes(tokenCount: count))
        defer { scratch.release() }
        let next = try appendedMetadata(validity, through: position + count)
        do {
            let latent = latents.squeezed(axis: 0).expandedDimensions(axis: 0)
            rows[0].write(keys: latent, values: latent)
            rows[1].write(
                keys: indexKeys.squeezed(axis: 0).expandedDimensions(axis: 0),
                values: poolGates.squeezed(axis: 0).expandedDimensions(axis: 0))
            try store.finish(next.values)
            guard rows.allSatisfy({ $0.absoluteOffset == position + count }) else {
                throw GLM5NextSparseStateError.invalidPosition
            }
            transactionMetadata = next
            stagedTokenCount += count
        } catch {
            try? close()
            throw error
        }
    }

    public func commit(_ transaction: Transaction, accepting count: Int) throws {
        try require(transaction)
        guard (0 ... stagedTokenCount).contains(count) else {
            throw GLM5NextSparseStateError.invalidPosition
        }
        try store.finish()
        let rejected = stagedTokenCount - count
        rows.forEach {
            $0.rollback(rejected)
            $0.commitSpeculativeWrite()
        }
        committedTokenCount += count
        if count > 0 { metadata = transactionMetadata }
        transactionMetadata = nil
        stagedTokenCount = 0
        active = nil
    }

    public func rollback(_ transaction: Transaction) throws {
        try commit(transaction, accepting: 0)
    }

    public func indexView(_ transaction: Transaction? = nil) throws -> IndexView {
        let count = try visibleCount(transaction)
        guard count > 0 else { throw GLM5NextSparseStateError.invalidPosition }
        let owner = transaction == nil ? metadata : transactionMetadata
        guard let owner else { throw GLM5NextSparseStateError.invalidPosition }
        let pageSize = store.pool.config.pageSize
        let pages = (count - 1) / pageSize + 1
        // Reserve the small host planning map before constructing it. The
        // full transfer buffers below are reserved before native gather starts.
        let planning = try store.admission.reserveTransient(
            bytes: store.geometry.metadataBytes(entryCount: pages))
        defer { planning.release() }
        let group = store.pool.group(rows[1].groupKey)
        guard let layout = group.segmentLayout else {
            throw GLM5NextSparseStateError.invalidGeometry
        }
        var counts: [Int: Int] = [:]
        for (index, page) in rows[1].table.prefix(pages).enumerated() {
            let segment = layout.segmentIndex(page: page)
            counts[segment, default: 0] += min(pageSize, count - index * pageSize)
        }
        let footprint = try GLM5NextPagedIndexReadFootprint.plan(
            tokens: count, indexDimension: store.geometry.indexDimension,
            dtype: store.geometry.indexDType,
            pages: pages, tokensPerSegment: counts.keys.sorted().map { counts[$0]! })
        let reservation = try store.admission.reserveTransient(bytes: footprint.totalBytes)
        let gathered = rows[1].gatherRange(start: 0, count: count)
        let arrays = [
            gathered.keys.squeezed(axis: 1), gathered.values.squeezed(axis: 1),
            owner.values[0][0..., ..<count], owner.values[1][0..., ..<count],
        ]
        try store.finish(arrays)
        // Retain metadata through an independent exact copy, not an uncharged alias.
        let selector = MLXArray(true)
        let copied = arrays.map { MLX.where(selector, $0, $0) }
        try store.finish(copied)
        return IndexView(Arrays(copied, reservation: reservation), count: count)
    }

    /// Device indices remain device-resident. Invalid/-1 indices produce zero
    /// rows; there is no materialization of the complete latent history.
    public func gatherLatents(_ indices: MLXArray, transaction: Transaction? = nil) throws
        -> LatentView
    {
        let count = try visibleCount(transaction)
        guard count > 0, indices.ndim == 1, indices.dtype == .int32,
            (1 ... 2051).contains(indices.size)
        else {
            throw GLM5NextSparseStateError.invalidPayload("selected indices")
        }
        let bytes = try store.readScratchBytes(
            count: indices.size, width: store.geometry.latentDimension,
            dtype: store.geometry.latentDType, copies: 3, metadata: false)
        let reservation = try store.admission.reserveTransient(bytes: bytes)
        let row = rows[0]
        let group = store.pool.group(row.groupKey)
        let plan = PagedSelectedGather.prepare(group: group, pages: row.table)
        let result = PagedSelectedGather.gather(
            group: group, plan: plan, indices: indices, length: count)
        try store.finish([result.keys, result.values])
        return LatentView(Arrays([result.keys.squeezed(axis: 1)], reservation: reservation))
    }

    public func checkpoint() throws -> Checkpoint {
        try requireOpen()
        try store.finish()
        let pages = try rows.map { try $0.immutableSnapshot(through: committedTokenCount) }
        return Checkpoint(self, pages: pages)
    }

    public func restore(_ checkpoint: Checkpoint) throws {
        try requireOpen()
        guard active == nil else { throw GLM5NextSparseStateError.invalidTransaction }
        guard checkpoint.store === store, checkpoint.binding == binding else {
            throw GLM5NextSparseStateError.incompatibleCheckpoint
        }
        let replacement = try store.makeRows()
        do {
            for (row, snapshot) in zip(replacement, checkpoint.pages) {
                try row.adoptImmutableSnapshot(snapshot)
            }
            try store.finish()
        } catch {
            replacement.forEach { $0.releaseStorage() }
            throw error
        }
        rows.forEach { $0.releaseStorage() }
        rows = replacement
        metadata = checkpoint.metadata
        committedTokenCount = checkpoint.tokenCount
    }

    public func fork() throws -> GLM5NextPagedSparseState {
        let checkpoint = try checkpoint()
        let fork = try store.makeState(binding: binding)
        // The fresh rows already carry the required reservation.
        for (row, snapshot) in zip(fork.rows, checkpoint.pages) {
            try row.adoptImmutableSnapshot(snapshot)
        }
        fork.metadata = checkpoint.metadata
        fork.committedTokenCount = checkpoint.tokenCount
        return fork
    }

    public func trim(to count: Int) throws {
        try requireOpen()
        guard active == nil else { throw GLM5NextSparseStateError.invalidTransaction }
        guard (0 ... committedTokenCount).contains(count) else {
            throw GLM5NextSparseStateError.invalidPosition
        }
        try store.finish()
        rows.forEach { $0.rollback(committedTokenCount - count) }
        committedTokenCount = count
        if count == 0 { metadata = nil }
    }

    public func close() throws {
        guard !isClosed else { return }
        isClosed = true
        // finish() always joins the stream before reporting a recorded fault.
        var failure: (any Error)?
        do { try store.finish() } catch { failure = error }
        rows.forEach { $0.releaseStorage() }
        rows.removeAll()
        metadata = nil
        transactionMetadata = nil
        active = nil
        if let failure { throw failure }
    }

    deinit { try? close() }

    private func appendedMetadata(_ validity: MLXArray, through count: Int) throws -> Arrays {
        let bytes = try store.metadataAllocationBytes(count: count)
        let reservation = try store.admission.reserveTransient(bytes: bytes)
        let prefix = transactionMetadata?.values[0][0..., ..<(count - validity.dim(1))]
        let all = prefix.map { concatenated([$0, validity], axis: 1) } ?? validity
        let selector = MLXArray(true)
        return Arrays(
            [
                MLX.where(selector, all, all),
                MLXArray((0 ..< count).map(Int32.init)).reshaped(1, count),
            ],
            reservation: reservation)
    }

    private func visibleCount(_ transaction: Transaction?) throws -> Int {
        try requireOpen()
        if let transaction { try require(transaction) }
        return committedTokenCount + (transaction == nil ? 0 : stagedTokenCount)
    }
    private func requireOpen() throws { if isClosed { throw GLM5NextSparseStateError.closed } }
    private func require(_ transaction: Transaction) throws {
        try requireOpen()
        guard transaction.owner == id, transaction == active else {
            throw GLM5NextSparseStateError.invalidTransaction
        }
    }
}
