// Copyright © 2026 Eigen Labs.
// MiMo-only target scratch; neither a new ledger nor a native completion owner.
import Foundation
import MLX

package struct MiMoV26RectangularDenseScratchSpec: Sendable {
    package let buffers: [CBv2MTPFixedBufferSpec]
    package let hostBytes: Int
    package init(buffers: [CBv2MTPFixedBufferSpec], hostBytes: Int) {
        self.buffers = buffers
        self.hostBytes = hostBytes
    }
}

package protocol MiMoV26RectangularDenseAllocatingModel: AnyObject {
    /// Nil means the optional candidate is not requested/supported.
    /// Metadata only: no allocation, native query, eval or model alias.
    var cbv2MiMoRectangularDenseScratch: MiMoV26RectangularDenseScratchSpec? { get }
}

/// Constructed only by EngineV2, after pricing all independent allocations.
/// The same fixed charge is added to actual Admission before requests exist.
/// Scalar identities do not retain a model/backend/cache alias.
// Lock protects the sole mutable diagnostic counter; all authority is immutable.
package final class MiMoV26RectangularDenseBudget: @unchecked Sendable {
    package let engineID: UUID
    package let modelIdentity, backendIdentity, cacheProviderIdentity: ObjectIdentifier
    package let fixedRequestBytes: Int
    private let lock = NSLock()
    private var activationCount = 0
    package var submittedCalls: Int { lock.withLock { activationCount } }
    fileprivate func didSubmit() { lock.withLock { activationCount += 1 } }

    init(engineID: UUID, model: AnyObject, backend: AnyObject, cacheProvider: AnyObject,
         spec: MiMoV26RectangularDenseScratchSpec, policy: AllocationFootprintPolicy) throws {
        guard let bytes = Self.resolve(spec, upperBound: policy.upperBound(byteCount:)) else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        self.engineID = engineID
        modelIdentity = ObjectIdentifier(model)
        backendIdentity = ObjectIdentifier(backend)
        cacheProviderIdentity = ObjectIdentifier(cacheProvider)
        fixedRequestBytes = bytes
    }

    /// Pure arithmetic injection is internal for bound tests, not authority.
    static func resolve(_ spec: MiMoV26RectangularDenseScratchSpec,
                        upperBound: (Int) -> Int?) -> Int? {
        guard !spec.buffers.isEmpty, spec.hostBytes >= 0 else { return nil }
        var total = spec.hostBytes
        for buffer in spec.buffers {
            guard buffer.logicalBytes > 0, buffer.allocationCount > 0,
                  let bound = upperBound(buffer.logicalBytes),
                  bound >= buffer.logicalBytes else { return nil }
            let (all, overflow) = bound.multipliedReportingOverflow(by: buffer.allocationCount)
            let (next, sumOverflow) = total.addingReportingOverflow(all)
            guard !overflow, !sumOverflow else { return nil }
            total = next
        }
        return total > 0 ? total : nil
    }
}

/// Synchronous, engine-only graph-build scope. A captured TaskLocal cannot arm
/// work after return or on a different thread. It owns metadata, not arrays.
package enum MiMoV26RectangularDenseAdmission {
    private final class Frame: @unchecked Sendable {
        let budget: MiMoV26RectangularDenseBudget
        let thread = ObjectIdentifier(Thread.current)
        private let lock = NSLock()
        private var open = true
        init(_ budget: MiMoV26RectangularDenseBudget) { self.budget = budget }
        func close() { lock.withLock { open = false } }
        func matches(_ model: AnyObject) -> Bool {
            lock.withLock {
                open && thread == ObjectIdentifier(Thread.current)
                    && budget.modelIdentity == ObjectIdentifier(model)
            }
        }
    }
    @TaskLocal private static var current: Frame?
    static func withBudget<Result>(_ budget: MiMoV26RectangularDenseBudget?,
                                   _ body: () throws -> Result) rethrows -> Result {
        guard let budget else { return try $current.withValue(nil, operation: body) }
        let frame = Frame(budget)
        defer { frame.close() }
        return try $current.withValue(frame, operation: body)
    }
    package static func isActive(for model: AnyObject) -> Bool {
        current?.matches(model) == true
    }
    /// Activation only, not GPU completion or numerical qualification.
    package static func recordSubmission(for model: AnyObject) {
        guard let current, current.matches(model) else { return }
        current.budget.didSubmit()
    }
}
