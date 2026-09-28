import Foundation

/// Opt-in qualification counters; no tensor readback or retained model state.
@_spi(DiffusionGemmaDiagnostics)
public enum DiffusionGemmaSoftEmbeddingDiagnostics {
    public struct Snapshot: Equatable, Sendable {
        public let armed: Bool
        public let calls: UInt64
    }

    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var armed = false
        var calls: UInt64 = 0
    }
    private static let state = State()

    public static func clearAndArm() {
        state.lock.withLock { state.calls = 0; state.armed = true }
    }

    public static func snapshot() -> Snapshot {
        state.lock.withLock { Snapshot(armed: state.armed, calls: state.calls) }
    }

    public static func snapshotAndDisarm() -> Snapshot {
        state.lock.withLock {
            state.armed = false
            return Snapshot(armed: false, calls: state.calls)
        }
    }

    static func recordDispatch() {
        state.lock.withLock { if state.armed { state.calls += 1 } }
    }
}
