import Foundation

/// Coordinates test scopes that use the process-wide PLE directory. Unlike
/// a suite's `.serialized` trait, this also covers other suites and XCTest.
/// A semaphore permits async XCTest setup/teardown on different threads.
final class Qwen4ExpPLETestBinding {
    private static let semaphore = DispatchSemaphore(value: 1)
    private let lock = NSLock()
    private var released = false

    init() {
        Self.semaphore.wait()
    }

    func release() {
        lock.withLock {
            if !released {
                released = true
                Self.semaphore.signal()
            }
        }
    }

    deinit {
        release()
    }
}
