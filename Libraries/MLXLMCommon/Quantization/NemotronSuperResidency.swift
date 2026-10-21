import Foundation

/// Experimental host accounting inside an explicit caller-owned grant. This
/// neither probes available RAM nor replaces WiredMemoryManager/CBv2 admission.
/// The caller must prevent double grants and price all retained arrays, backend
/// scratch and concurrent lazy work. No model availability is implied.
public final class NemotronSuperResidency: @unchecked Sendable {
    public enum Kind: Sendable { case weights, staging, request }
    public enum Failure: Error, Equatable {
        case invalidBudget, invalidBytes, overflow, capacity, concurrency, reservationLimit,
            draining
    }
    public struct Snapshot: Sendable, Equatable {
        public let capacityBytes: Int
        public let weightBytes: Int
        public let stagingBytes: Int
        public let requestBytes: Int
        public let requestCount: Int
        public let reservationCount: Int
        public let isDraining: Bool
        public var usedBytes: Int { weightBytes + stagingBytes + requestBytes }
        public var availableBytes: Int { capacityBytes - usedBytes }
        /// Full drain includes weights, not merely the end of request work.
        public var isDrained: Bool { isDraining && reservationCount == 0 }
    }

    /// Explicit release only, deliberately no deinit release: a dropped handle
    /// must not make still-live GPU allocations appear available. Retain this
    /// handle until execution completes AND its charged buffers are retired.
    /// Repeated release is harmless. Losing the handle leaks accounting safely.
    public final class Reservation: @unchecked Sendable {
        public let kind: Kind
        public let bytes: Int
        private let owner: NemotronSuperResidency
        private let id: UUID
        fileprivate init(owner: NemotronSuperResidency, id: UUID, kind: Kind, bytes: Int) {
            self.owner = owner
            self.id = id
            self.kind = kind
            self.bytes = bytes
        }
        @discardableResult public func release() -> Bool { owner.release(id) }
    }

    private let lock = NSLock()
    private let capacity: Int
    private let maximumRequests: Int
    private let maximumReservations: Int
    private var entries: [UUID: (Kind, Int)] = [:]
    private var weightBytes = 0, stagingBytes = 0, requestBytes = 0, requestCount = 0
    private var draining = false

    public init(capacityBytes: Int, maximumRequests: Int, maximumReservations: Int = 4096) throws {
        guard capacityBytes > 0, maximumRequests > 0, maximumReservations > 0,
            maximumReservations <= 1_000_000, maximumRequests <= maximumReservations
        else {
            throw Failure.invalidBudget
        }
        capacity = capacityBytes
        self.maximumRequests = maximumRequests
        self.maximumReservations = maximumReservations
    }

    /// Sum separately estimated live components without wrapping. A request is
    /// one reservation regardless of its component count. Failure is atomic.
    public func reserve(_ kind: Kind, components: [Int]) throws -> Reservation {
        guard !components.isEmpty, components.count <= 1024 else { throw Failure.invalidBytes }
        var total = 0
        for bytes in components {
            guard bytes >= 0 else { throw Failure.invalidBytes }
            let (sum, overflow) = total.addingReportingOverflow(bytes)
            guard !overflow else { throw Failure.overflow }
            total = sum
        }
        guard total > 0 else { throw Failure.invalidBytes }
        return try lock.withLock {
            guard !draining else { throw Failure.draining }
            guard entries.count < maximumReservations else { throw Failure.reservationLimit }
            if kind == .request, requestCount >= maximumRequests { throw Failure.concurrency }
            let used = weightBytes + stagingBytes + requestBytes
            guard total <= capacity - used else { throw Failure.capacity }
            var id = UUID()
            while entries[id] != nil { id = UUID() }
            entries[id] = (kind, total)
            switch kind {
            case .weights: weightBytes += total
            case .staging: stagingBytes += total
            case .request:
                requestBytes += total
                requestCount += 1
            }
            return Reservation(owner: self, id: id, kind: kind, bytes: total)
        }
    }

    public func reserve(_ kind: Kind, bytes: Int) throws -> Reservation {
        try reserve(kind, components: [bytes])
    }

    private func release(_ id: UUID) -> Bool {
        lock.withLock {
            guard let (kind, bytes) = entries.removeValue(forKey: id) else { return false }
            switch kind {
            case .weights: weightBytes -= bytes
            case .staging: stagingBytes -= bytes
            case .request:
                requestBytes -= bytes
                requestCount -= 1
            }
            return true
        }
    }

    public var snapshot: Snapshot {
        lock.withLock { snapshotLocked() }
    }

    /// Terminal, nonblocking admission fence. Does not cancel work, release
    /// reservations or wait for GPU completion. Construct a new ledger only
    /// after the external owner has finished draining and acquired a new grant.
    @discardableResult public func beginDrain() -> Snapshot {
        lock.withLock {
            draining = true
            return snapshotLocked()
        }
    }

    private func snapshotLocked() -> Snapshot {
        Snapshot(
            capacityBytes: capacity, weightBytes: weightBytes, stagingBytes: stagingBytes,
            requestBytes: requestBytes, requestCount: requestCount,
            reservationCount: entries.count, isDraining: draining)
    }
}
