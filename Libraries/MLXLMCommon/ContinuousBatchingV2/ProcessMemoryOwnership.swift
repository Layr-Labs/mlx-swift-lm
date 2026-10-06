import Foundation

/// One engine's synchronous connection to process-wide memory admission.
/// Admission supplies its complete native charge and exact evaluated coverage.
/// Implementations must not call an engine, await, allocate GPU buffers or do I/O.
public protocol CBv2ProcessMemoryOwner: AnyObject, Sendable {
    /// Replace this owner's complete charge C, never an incremental delta.
    func replaceCharge(_ bytes: UInt64) throws
    /// Record the complete evaluated coverage M. This is an absolute total,
    /// nondecreasing until an explicit withdrawal, and cannot exceed C.
    func recordMaterialization(_ bytes: UInt64) throws
    /// Remove this many bytes of coverage; this delta does not refund charge.
    func withdrawCoverage(_ bytes: UInt64) throws
    /// Close to new charge growth. Retain existing C/M until actual consumers
    /// withdraw coverage and reduce charge; retirement itself is not a refund.
    func retire()
}

/// Removing materialization credit is conservative; it never refunds charge.
/// The native caller refunds charge only after the actual buffer owners drain.
final class CBv2MemoryCoverage: @unchecked Sendable {
    private let lock = NSLock()
    private var withdraw: (@Sendable () -> Void)?

    init(withdraw: @escaping @Sendable () -> Void) { self.withdraw = withdraw }

    func invalidate() {
        lock.lock()
        defer { lock.unlock() }
        let callback = withdraw
        withdraw = nil
        callback?()
    }

    deinit { invalidate() }
}
