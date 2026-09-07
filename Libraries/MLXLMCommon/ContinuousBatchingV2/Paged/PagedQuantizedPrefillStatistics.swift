import Foundation

public enum PagedQuantizedPrefillMode: String, Codable, Sendable {
    case direct
    case opportunisticSDPA
}

/// Cumulative graph-built layer/row route decisions, not proof that a graph
/// completed on the GPU. Reservation bytes include accepted extra permits
/// until the owning step retires, independently of route decision counters.
public struct PagedQuantizedPrefillStatistics: Codable, Sendable {
    public let mode: PagedQuantizedPrefillMode
    public internal(set) var directCallCount = 0
    public internal(set) var fusedCallCount = 0
    public internal(set) var directQueryTokenCount = 0
    public internal(set) var fusedQueryTokenCount = 0
    public internal(set) var budgetFallbackCount = 0
    public internal(set) var ineligibleFallbackCount = 0
    public internal(set) var currentAdditionalWorkspaceBytes = 0
    public internal(set) var peakAdditionalWorkspaceBytes = 0
}

/// The engine mutates on its queue; benchmark/report readers may run elsewhere.
final class PagedQuantizedPrefillCounters: @unchecked Sendable {
    private let lock = NSLock()
    private var value: PagedQuantizedPrefillStatistics

    init(mode: PagedQuantizedPrefillMode) { value = .init(mode: mode) }
    var snapshot: PagedQuantizedPrefillStatistics { lock.withLock { value } }

    func direct(tokens: Int) {
        lock.withLock { value.directCallCount += 1; value.directQueryTokenCount += tokens }
    }
    func fused(tokens: Int) {
        lock.withLock { value.fusedCallCount += 1; value.fusedQueryTokenCount += tokens }
    }
    func fallback(budget: Bool) {
        lock.withLock {
            if budget { value.budgetFallbackCount += 1 }
            else { value.ineligibleFallbackCount += 1 }
        }
    }
    func reserved(bytes: Int) {
        lock.withLock {
            value.currentAdditionalWorkspaceBytes += bytes
            value.peakAdditionalWorkspaceBytes = max(
                value.peakAdditionalWorkspaceBytes, value.currentAdditionalWorkspaceBytes)
        }
    }
    func retired(bytes: Int) {
        lock.withLock { value.currentAdditionalWorkspaceBytes -= bytes }
    }
}

extension PagedKVPool {
    public var quantizedPrefillStatistics: PagedQuantizedPrefillStatistics {
        quantizedPrefillCounters.snapshot
    }
}
