import Foundation

/// Model-owned process binding. No prompt data, tensors, or open file handles.
public final class Qwen4ExpPLEDirectoryLease: @unchecked Sendable {
    private let directory: URL
    private let lock = NSLock()
    private var released = false

    init(directory: URL) { self.directory = directory }

    public func release() {
        lock.lock()
        let shouldRelease = !released
        released = true
        lock.unlock()
        if shouldRelease { Qwen4ExpPLEResidency.release(directory: directory) }
    }

    deinit { release() }
}
