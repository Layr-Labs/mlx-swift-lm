#if canImport(Metal) && canImport(ObjectiveC)
import Foundation
import Metal
import ObjectiveC

@testable import MLXLMCommon

/// Live `MTLSharedEvent`s that the system Metal device created after the
/// first use of `live`. The first use wraps the device class's
/// `newSharedEvent` once per process: each new event gets an associated
/// tag whose deinit runs when the event is freed. MLX creates every eval's
/// completion event (`metal::EventImpl`) through `newSharedEvent`.
enum MetalSharedEventCounter {
    private final class Tag {
        deinit { MetalSharedEventCounter.add(-1) }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var count = 0
    nonisolated(unsafe) private static let tagKey = UnsafeMutableRawPointer.allocate(
        byteCount: 1, alignment: 1)

    fileprivate static func add(_ delta: Int) {
        lock.lock()
        count += delta
        lock.unlock()
    }

    /// nil when the process has no Metal device.
    static var live: Int? {
        guard installed else { return nil }
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    private static let installed: Bool = {
        guard let device = MTLCreateSystemDefaultDevice(),
            let deviceClass = object_getClass(device)
        else { return false }
        let selector = NSSelectorFromString("newSharedEvent")
        guard let method = class_getInstanceMethod(deviceClass, selector) else { return false }
        typealias Make =
            @convention(c) (UnsafeMutableRawPointer, Selector) -> UnsafeMutableRawPointer?
        let original = unsafeBitCast(method_getImplementation(method), to: Make.self)
        // `new` returns a +1 object; the raw pointers pass it through unretained.
        let wrapper: @convention(block) (UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? = {
            device in
            guard let event = original(device, selector) else { return nil }
            let object = Unmanaged<AnyObject>.fromOpaque(event).takeUnretainedValue()
            objc_setAssociatedObject(object, tagKey, Tag(), .OBJC_ASSOCIATION_RETAIN)
            add(1)
            return event
        }
        let implementation = imp_implementationWithBlock(wrapper)
        if !class_addMethod(
            deviceClass, selector, implementation, method_getTypeEncoding(method))
        {
            method_setImplementation(method, implementation)
        }
        return true
    }()
}

/// Live shared events at two holds of an engine whose rows decode together.
struct BatchGatedEventSample {
    let early: Int
    let late: Int
    let outputs: [[Int]]
    let metrics: CBv2MTPMetrics?
}

extension MetalSharedEventCounter {
    /// Admits `requests` together, holds the engine at both ends of `steps` to
    /// read the live count, then lets the requests finish and shuts the engine
    /// down. nil without Metal, or when the engine never reached a hold.
    static func sampleBatchGatedStretch(
        _ engine: EngineV2, requests: [CBv2Request], steps: ClosedRange<Int>
    ) async throws -> BatchGatedEventSample? {
        guard live != nil else { return nil }
        let loop = engine.loopForTesting
        let streams = try loop.onEngineQueueSync {
            loop.suspendStepExecutionAtCountForTesting = steps.lowerBound
            return try requests.map { try engine.submit($0) }
        }
        let early = try await live(whenHeldAt: steps.lowerBound, engine)
        loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = steps.upperBound }
        let late = try await live(whenHeldAt: steps.upperBound, engine)
        loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = nil }
        var outputs: [[Int]] = []
        for stream in streams { outputs.append(await cbv2SchedCollect(stream).tokens) }
        let metrics = engine.mtpMetricsSnapshot()
        await engine.shutdown()
        guard let early, let late else { return nil }
        return BatchGatedEventSample(early: early, late: late, outputs: outputs, metrics: metrics)
    }

    /// The live count once the engine holds at `step` and the in-flight
    /// step's GPU work has had time to retire.
    private static func live(whenHeldAt step: Int, _ engine: EngineV2) async throws -> Int? {
        for _ in 0 ..< 1_000 {
            let loop = engine.loopForTesting
            if loop.onEngineQueueSync({ loop.stepCount >= step }) {
                try await Task.sleep(nanoseconds: 200_000_000)
                return live
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        return nil
    }
}
#endif
