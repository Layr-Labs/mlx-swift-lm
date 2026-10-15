// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Engine-thread-confined sparse state. KDA convolution/SSM state belongs to
/// CBv2RecurrentRequestState, not this owner. No paging/prefix capability is issued.
public final class GLM5NextSparseState {
    public struct Transaction: Equatable {
        fileprivate let owner: UUID
        fileprivate let generation: UInt64
    }

    /// Keep this lease through the last native reader's evaluation fence.
    /// Naked array aliases must not outlive it. Reads never mutate shared chunks.
    public final class View {
        public let tokenCount: Int
        public let allocationBytes: Int
        public var latents: MLXArray { payload!.latents }
        public var indexKeys: MLXArray { payload!.keys }
        public var poolGates: MLXArray { payload!.gates }
        public var validity: MLXArray { payload!.validity }
        public var positions: MLXArray { payload!.positions }
        public var evaluationTargets: [MLXArray] { payload!.arrays }

        private var payload: Payload?
        private var sources: [Slice]
        private let reservation: CBv2CheckpointReservation

        fileprivate init(
            payload: Payload, sources: [Slice], tokenCount: Int,
            allocationBytes: Int, reservation: CBv2CheckpointReservation
        ) {
            self.payload = payload
            self.sources = sources
            self.tokenCount = tokenCount
            self.allocationBytes = allocationBytes
            self.reservation = reservation
        }

        deinit {
            payload = nil
            sources.removeAll()
            reservation.release()
        }
    }

    /// An in-process, same-admission immutable prefix. This is not an SSD codec
    /// or proof that matching KDA/assistant state was captured at this frontier.
    public final class Checkpoint {
        public let tokenCount: Int
        public let binding: GLM5NextSparseStateBinding
        public let geometry: GLM5NextSparseStateGeometry
        fileprivate let admission: AdmissionV2
        fileprivate var slices: [Slice]
        private let reservation: CBv2CheckpointReservation

        fileprivate init(
            tokenCount: Int, binding: GLM5NextSparseStateBinding,
            geometry: GLM5NextSparseStateGeometry, admission: AdmissionV2,
            slices: [Slice], reservation: CBv2CheckpointReservation
        ) {
            self.tokenCount = tokenCount
            self.binding = binding
            self.geometry = geometry
            self.admission = admission
            self.slices = slices
            self.reservation = reservation
        }

        deinit {
            slices.removeAll()
            reservation.release()
        }
    }

    fileprivate struct Payload {
        let latents: MLXArray
        let keys: MLXArray
        let gates: MLXArray
        let validity: MLXArray
        let positions: MLXArray
        var arrays: [MLXArray] { [latents, keys, gates, validity, positions] }

        func prefix(_ count: Int) -> Payload {
            Payload(
                latents: latents[0..., ..<count, 0...],
                keys: keys[0..., ..<count, 0...], gates: gates[0..., ..<count, 0...],
                validity: validity[0..., ..<count], positions: positions[0..., ..<count])
        }
    }

    fileprivate final class Chunk {
        var payload: Payload?
        let bytes: Int
        private let reservation: CBv2CheckpointReservation
        init(payload: Payload, bytes: Int, reservation: CBv2CheckpointReservation) {
            self.payload = payload
            self.bytes = bytes
            self.reservation = reservation
        }
        deinit {
            payload = nil
            reservation.release()
        }
    }

    fileprivate struct Slice {
        let chunk: Chunk
        let count: Int
    }

    public let geometry: GLM5NextSparseStateGeometry
    public let binding: GLM5NextSparseStateBinding
    public private(set) var committedTokenCount = 0
    public private(set) var stagedTokenCount = 0
    public private(set) var isClosed = false
    private let admission: AdmissionV2
    private let owner = UUID()
    private var generation: UInt64 = 0
    private var active: Transaction?
    private var committed: [Slice] = []
    private var staged: [Slice] = []
    private var inheritedMetadata: CBv2CheckpointReservation?

    /// Physical chunk charges, not logical prefix bytes. Partial acceptance can
    /// keep a larger immutable backing charged until every borrower retires.
    public var retainedAllocationBytes: Int {
        var seen = Set<ObjectIdentifier>()
        return (committed + staged).reduce(0) { total, slice in
            seen.insert(ObjectIdentifier(slice.chunk)).inserted ? total + slice.chunk.bytes : total
        }
    }

    public init(
        geometry: GLM5NextSparseStateGeometry, binding: GLM5NextSparseStateBinding,
        admission: AdmissionV2
    ) {
        self.geometry = geometry
        self.binding = binding
        self.admission = admission
    }

    public func begin(at position: Int) throws -> Transaction {
        try requireOpen()
        guard active == nil else { throw GLM5NextSparseStateError.invalidTransaction }
        guard position == committedTokenCount else { throw GLM5NextSparseStateError.invalidPosition }
        guard generation < UInt64.max else { throw GLM5NextSparseStateError.invalidTransaction }
        generation += 1
        let transaction = Transaction(owner: owner, generation: generation)
        active = transaction
        return transaction
    }

    /// Inputs stay caller-owned/charged and immutable until the copy graph is
    /// evaluated. A View of this transaction supplies all evaluation roots.
    public func stage(
        _ transaction: Transaction, at position: Int,
        latents: MLXArray, indexKeys: MLXArray, poolGates: MLXArray, validity: MLXArray
    ) throws {
        try require(transaction)
        guard position == committedTokenCount + stagedTokenCount else {
            throw GLM5NextSparseStateError.invalidPosition
        }
        guard latents.ndim == 3 else { throw GLM5NextSparseStateError.invalidPayload("latents") }
        let count = latents.dim(1)
        guard count > 0, count <= geometry.maximumTokens - position,
            count <= geometry.maximumStagedTokens - stagedTokenCount
        else { throw GLM5NextSparseStateError.invalidPosition }
        try validate(latents, shape: [1, count, geometry.latentDimension], dtype: geometry.latentDType, name: "latents")
        try validate(indexKeys, shape: [1, count, geometry.indexDimension], dtype: geometry.indexDType, name: "index keys")
        try validate(poolGates, shape: indexKeys.shape, dtype: geometry.indexDType, name: "pool gates")
        try validate(validity, shape: [1, count], dtype: .bool, name: "validity")
        let bytes = try geometry.allocationBytes(tokenCount: count)
        let reservation = try admission.reserveTransient(bytes: bytes)
        let selector = MLXArray(true)
        let positions = MLXArray((position..<(position + count)).map(Int32.init)).reshaped(1, count)
        let payload = Payload(
            latents: MLX.where(selector, latents, latents),
            keys: MLX.where(selector, indexKeys, indexKeys),
            gates: MLX.where(selector, poolGates, poolGates),
            validity: MLX.where(selector, validity, validity), positions: positions)
        staged.append(Slice(chunk: Chunk(payload: payload, bytes: bytes, reservation: reservation), count: count))
        stagedTokenCount += count
    }

    /// Without a transaction this exposes only the committed prefix, even if a
    /// speculative suffix exists. The five arrays always have the same frontier.
    public func view(_ transaction: Transaction? = nil) throws -> View {
        try requireOpen()
        if let transaction { try require(transaction) }
        let slices = transaction == nil ? committed : committed + staged
        let count = transaction == nil ? committedTokenCount : committedTokenCount + stagedTokenCount
        let bytes = try geometry.allocationBytes(tokenCount: count)
        let reservation = try admission.reserveTransient(bytes: bytes)
        let payload: Payload
        if slices.isEmpty {
            payload = Payload(
                latents: .zeros([1, 0, geometry.latentDimension], dtype: geometry.latentDType),
                keys: .zeros([1, 0, geometry.indexDimension], dtype: geometry.indexDType),
                gates: .zeros([1, 0, geometry.indexDimension], dtype: geometry.indexDType),
                validity: .zeros([1, 0], dtype: .bool), positions: .zeros([1, 0], dtype: .int32))
        } else {
            let parts = slices.map { $0.chunk.payload!.prefix($0.count) }
            let selector = MLXArray(true)
            func copy(_ arrays: [MLXArray]) -> MLXArray {
                let joined = arrays.count == 1 ? arrays[0] : concatenated(arrays, axis: 1)
                // Concatenate already produces the independently charged output.
                return arrays.count == 1 ? MLX.where(selector, joined, joined) : joined
            }
            payload = Payload(
                latents: copy(parts.map(\.latents)), keys: copy(parts.map(\.keys)),
                gates: copy(parts.map(\.gates)), validity: copy(parts.map(\.validity)),
                positions: copy(parts.map(\.positions)))
        }
        return View(payload: payload, sources: slices, tokenCount: count, allocationBytes: bytes, reservation: reservation)
    }

    /// Metadata-only after the caller's evaluation/acceptance fence. This never
    /// invents a recurrent checkpoint or commits a different target frontier.
    public func commit(_ transaction: Transaction, accepting tokens: Int) throws {
        try require(transaction)
        guard (0...stagedTokenCount).contains(tokens) else { throw GLM5NextSparseStateError.invalidPosition }
        committed.append(contentsOf: prefix(staged, count: tokens))
        committedTokenCount += tokens
        discardActive()
    }

    public func rollback(_ transaction: Transaction) throws {
        try require(transaction)
        discardActive()
    }

    public func checkpoint() throws -> Checkpoint {
        try requireOpen()
        let reservation = try admission.reserveTransient(bytes: geometry.metadataBytes(entryCount: committed.count))
        return Checkpoint(
            tokenCount: committedTokenCount, binding: binding, geometry: geometry,
            admission: admission, slices: committed, reservation: reservation)
    }

    public func restore(_ checkpoint: Checkpoint) throws {
        try requireOpen()
        guard active == nil else { throw GLM5NextSparseStateError.invalidTransaction }
        guard checkpoint.binding == binding, checkpoint.geometry == geometry,
            checkpoint.admission === admission else { throw GLM5NextSparseStateError.incompatibleCheckpoint }
        let metadata = try admission.reserveTransient(bytes: geometry.metadataBytes(entryCount: checkpoint.slices.count))
        committed = checkpoint.slices
        committedTokenCount = checkpoint.tokenCount
        inheritedMetadata = metadata
    }

    public func fork() throws -> GLM5NextSparseState {
        let checkpoint = try checkpoint()
        let result = GLM5NextSparseState(geometry: geometry, binding: binding, admission: admission)
        try result.restore(checkpoint)
        return result
    }

    public func trim(to position: Int) throws {
        try requireOpen()
        guard active == nil else { throw GLM5NextSparseStateError.invalidTransaction }
        guard (0...committedTokenCount).contains(position) else { throw GLM5NextSparseStateError.invalidPosition }
        committed = prefix(committed, count: position)
        committedTokenCount = position
        if position == 0 { inheritedMetadata = nil }
    }

    public func close() {
        guard !isClosed else { return }
        isClosed = true
        discardActive()
        committed.removeAll()
        committedTokenCount = 0
        inheritedMetadata = nil
    }

    deinit { close() }

    private func discardActive() {
        staged.removeAll()
        stagedTokenCount = 0
        active = nil
    }

    private func prefix(_ slices: [Slice], count: Int) -> [Slice] {
        var remaining = count
        var result: [Slice] = []
        for slice in slices where remaining > 0 {
            let kept = min(slice.count, remaining)
            result.append(Slice(chunk: slice.chunk, count: kept))
            remaining -= kept
        }
        return result
    }

    private func requireOpen() throws {
        guard !isClosed else { throw GLM5NextSparseStateError.closed }
    }

    private func require(_ transaction: Transaction) throws {
        try requireOpen()
        guard transaction.owner == owner, active == transaction else {
            throw GLM5NextSparseStateError.invalidTransaction
        }
    }

    private func validate(_ array: MLXArray, shape: [Int], dtype: DType, name: String) throws {
        guard array.shape == shape, array.dtype == dtype else {
            throw GLM5NextSparseStateError.invalidPayload(name)
        }
    }
}
