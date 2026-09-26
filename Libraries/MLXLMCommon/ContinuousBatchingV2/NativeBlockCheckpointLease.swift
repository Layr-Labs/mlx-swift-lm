import Foundation

/// A real native-engine capacity reservation, acquired before checkpoint
/// allocation. Storage owners drop their arrays before closing this lease.
public final class CBv2NativeBlockCheckpointLease: @unchecked Sendable {
    public let bytes: Int
    private let lock = NSLock()
    private var release: (@Sendable () -> Void)?

    init(bytes: Int, release: @escaping @Sendable () -> Void) {
        self.bytes = bytes
        self.release = release
    }

    public func close() {
        let callback = lock.withLock { let value = release; release = nil; return value }
        callback?()
    }
    deinit { close() }
}
