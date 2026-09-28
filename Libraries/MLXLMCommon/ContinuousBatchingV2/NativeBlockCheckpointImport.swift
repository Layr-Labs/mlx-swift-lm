import Cmlx
import Foundation
import MLX

/// Explicit native-block import; never constructs an AR/recurrent checkpoint.
/// The model codec validates identities and exact tensor geometry first. The
/// provider authenticates the full encrypted stream before calling finish.
public final class CBv2NativeBlockCheckpointImportPlan: @unchecked Sendable {
    public let manifest: CBv2CompleteCheckpointManifest
    public let nativeDestinationBytes: Int
    public let scratchBytes: Int
    public let maximumSequenceLength: Int
    public let usesProcessMemoryOwner: Bool
    let codecIdentity: UUID
    private let engine: CBv2NativeBlockEngine

    package init(
        manifest: CBv2CompleteCheckpointManifest, engine: CBv2NativeBlockEngine,
        codecIdentity: UUID,
        maximumSequenceLength: Int
    ) throws {
        _ = try manifest.validateStructure()
        guard manifest.backendLayout == CBv2CompleteCheckpointManifest.diffusionBlockLayout else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        self.engine = engine
        self.codecIdentity = codecIdentity
        self.maximumSequenceLength = maximumSequenceLength
        self.usesProcessMemoryOwner = engine.usesProcessMemoryOwner
        self.manifest = try manifest.owningNativeMetadata(engine: engine)
        var destination = 0
        var initialization = 0
        for descriptor in manifest.tensors {
            destination = try CBv2CheckpointAllocationFootprint.add(
                destination,
                CBv2CheckpointAllocationFootprint.bound(descriptor.byteCount))
            initialization = try CBv2CheckpointAllocationFootprint.add(
                initialization,
                CBv2CheckpointAllocationFootprint.bound(descriptor.dtype.mlxDType.size))
        }
        nativeDestinationBytes = destination
        scratchBytes = try CBv2CheckpointAllocationFootprint.add(
            initialization,
            usesProcessMemoryOwner ? 0 : CBv2CompleteCheckpointManifest.maximumProviderScratchBytes)
    }

    func coverDestinations(bytes: Int) throws -> CBv2MemoryCoverage? {
        try engine.coverNativeCheckpointAllocation(bytes: bytes)
    }

    public func allocate(onRelease: @escaping @Sendable () -> Void = {}) throws
        -> CBv2NativeBlockCheckpointImport
    {
        let external = CBv2CheckpointReservation(onRelease: onRelease)
        do {
            let lease = try engine.reserveNativeCheckpoint(
                bytes:
                    CBv2CheckpointAllocationFootprint.add(nativeDestinationBytes, scratchBytes))
            do { return try .init(plan: self, lease: lease, external: external) } catch {
                lease.close()
                throw error
            }
        } catch {
            external.release()
            throw error
        }
    }
}

public final class CBv2NativeBlockCheckpointImport: @unchecked Sendable {
    private let lock = NSLock()
    private let manifest: CBv2CompleteCheckpointManifest
    private let codecIdentity: UUID
    private let maximumSequenceLength: Int
    private var storage: CBv2NativeBlockCheckpointStorage?
    private var tensorIndex = 0
    private var byteOffset = 0

    fileprivate init(
        plan: CBv2NativeBlockCheckpointImportPlan, lease: CBv2NativeBlockCheckpointLease,
        external: CBv2CheckpointReservation
    ) throws {
        manifest = plan.manifest
        codecIdentity = plan.codecIdentity
        maximumSequenceLength = plan.maximumSequenceLength
        let stream = StreamOrDevice.default
        let arrays = try withError { errors in
            let arrays = plan.manifest.tensors.map {
                MLXArray.zeros($0.shape, dtype: $0.dtype.mlxDType, stream: stream)
            }
            do {
                try errors.check()
                eval(arrays)
                stream.stream.synchronize()
                try errors.check()
            } catch {
                stream.stream.synchronize()
                throw error
            }
            return arrays
        }
        let footprint = try CBv2CheckpointAllocationFootprint.freshBytes(arrays)
        guard footprint.actual <= plan.nativeDestinationBytes else {
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        storage = .init(
            arrays: arrays, lease: lease, external: external,
            nativeDestinationBytes: footprint.actual,
            coverage: try plan.coverDestinations(bytes: footprint.actual))
    }

    public func appendSegment(tensorIndex: Int, byteOffset: Int, data: Data) throws {
        try lock.withLock {
            guard let storage else { throw CBv2CompleteCheckpointError.closed }
            guard tensorIndex == self.tensorIndex, byteOffset == self.byteOffset,
                manifest.tensors.indices.contains(tensorIndex), !data.isEmpty,
                data.count <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
            else { throw CBv2CompleteCheckpointError.invalidSegment }
            let descriptor = manifest.tensors[tensorIndex]
            guard data.count % descriptor.dtype.mlxDType.size == 0,
                data.count <= descriptor.byteCount - byteOffset,
                let pointer = mlx_array_data_uint8(storage.arrays[tensorIndex].ctx)
            else { throw CBv2CompleteCheckpointError.invalidSegment }
            // Destinations are fresh, evaluated, contiguous and unpublished.
            data.withUnsafeBytes { bytes in
                UnsafeMutableRawPointer(mutating: pointer).advanced(by: byteOffset)
                    .copyMemory(from: bytes.baseAddress!, byteCount: data.count)
            }
            self.byteOffset += data.count
            if self.byteOffset == descriptor.byteCount {
                self.tensorIndex += 1
                self.byteOffset = 0
            }
        }
    }

    public func finish() throws -> CBv2NativeBlockCheckpoint {
        try lock.withLock {
            guard let storage else { throw CBv2CompleteCheckpointError.closed }
            guard tensorIndex == manifest.tensors.count, byteOffset == 0 else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
            self.storage = nil
            return .init(
                manifest: manifest, storage: storage, codecIdentity: codecIdentity,
                maximumSequenceLength: maximumSequenceLength)
        }
    }

    public func close() { lock.withLock { storage = nil } }
    deinit { close() }
}

/// Single-use, opaque authenticated-transfer destination. Only a native model
/// codec can bind its arrays to a new loaded model owner.
public final class CBv2NativeBlockCheckpoint: @unchecked Sendable {
    public let manifest: CBv2CompleteCheckpointManifest
    public let maximumSequenceLength: Int
    public let nativeDestinationBytes: Int
    package let codecIdentity: UUID
    private let lock = NSLock()
    private var storage: CBv2NativeBlockCheckpointStorage?
    fileprivate init(
        manifest: CBv2CompleteCheckpointManifest, storage: CBv2NativeBlockCheckpointStorage,
        codecIdentity: UUID,
        maximumSequenceLength: Int
    ) {
        self.manifest = manifest
        self.storage = storage
        self.codecIdentity = codecIdentity
        self.maximumSequenceLength = maximumSequenceLength
        self.nativeDestinationBytes = storage.nativeDestinationBytes
    }
    package func consume() throws -> CBv2NativeBlockCheckpointStorage {
        try lock.withLock {
            guard let storage else { throw CBv2CompleteCheckpointError.closed }
            self.storage = nil
            return storage
        }
    }
    public func close() { lock.withLock { storage = nil } }
    deinit { close() }
}

package final class CBv2NativeBlockCheckpointStorage {
    package private(set) var arrays: [MLXArray]
    private let lease: CBv2NativeBlockCheckpointLease
    private let external: CBv2CheckpointReservation
    private let coverage: CBv2MemoryCoverage?
    package let nativeDestinationBytes: Int
    fileprivate init(
        arrays: [MLXArray], lease: CBv2NativeBlockCheckpointLease,
        external: CBv2CheckpointReservation, nativeDestinationBytes: Int,
        coverage: CBv2MemoryCoverage?
    ) {
        self.arrays = arrays
        self.lease = lease
        self.external = external
        self.coverage = coverage
        self.nativeDestinationBytes = nativeDestinationBytes
    }
    deinit {
        arrays.removeAll(keepingCapacity: false)
        coverage?.invalidate()
        lease.close()
        external.release()
    }
}

extension CBv2CompleteCheckpointExport {
    package convenience init(
        nativeBlockManifest manifest: CBv2CompleteCheckpointManifest,
        arrays: [MLXArray], engine: CBv2NativeBlockEngine
    ) throws {
        _ = try manifest.validateStructure()
        guard manifest.backendLayout == CBv2CompleteCheckpointManifest.diffusionBlockLayout,
            arrays.count == manifest.tensors.count,
            zip(arrays, manifest.tensors).allSatisfy({
                $0.shape == $1.shape && $0.dtype == $1.dtype.mlxDType
            })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.init(manifest: try manifest.owningNativeMetadata(engine: engine), arrays: arrays)
    }
}
