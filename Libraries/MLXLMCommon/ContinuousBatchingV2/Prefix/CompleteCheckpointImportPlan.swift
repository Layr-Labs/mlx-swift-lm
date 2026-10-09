import Foundation
import MLX

/// Mutable only at explicit transfer/retirement boundaries. A native loan
/// owns this box before a public plan can retain its codec/assistant; clearing
/// a public handle therefore never relies on incidental ARC release order.
final class CBv2CompleteCheckpointCodecOwner: @unchecked Sendable {
    private let lock = NSLock()
    private var value: CBv2CompleteCheckpointCodec?
    init(_ codec: CBv2CompleteCheckpointCodec) { value = codec }
    func borrow() throws -> CBv2CompleteCheckpointCodec {
        try lock.withLock {
            guard let value else { throw CBv2CompleteCheckpointError.closed }
            return value
        }
    }
    func dropNativeReferences() {
        let retired = lock.withLock {
            let old = value
            value = nil
            return old
        }
        withExtendedLifetime(retired) {}  // release outside the box lock
    }
}

/// A native plan owns a counted loan from public creation, then moves the
/// same work into Import. Retained consumed/closed native plan aliases contain
/// metadata only. Untracked plans preserve their prior multi-allocation API.
public final class CBv2CompleteCheckpointImportPlan: @unchecked Sendable {
    public let manifest: CBv2CompleteCheckpointManifest
    public let maximumSequenceLength: Int
    /// Native target plus native auxiliary destinations ONLY.
    public let nativeDestinationBytes: Int
    public let nativeTargetBytes: Int
    public let nativeAuxiliaryBytes: Int
    public let checkpointHostBytes: Int
    public let usesProcessMemoryOwner: Bool
    let pagedStoragePlan: CBv2PagedCheckpointStoragePlan?
    let auxiliaryBytes: Int
    public let scratchBytes: Int
    private let stateLock = NSLock()
    private var legacyCodec: CBv2CompleteCheckpointCodec?
    private var nativeCodecOwner: CBv2CompleteCheckpointCodecOwner?
    private var nativeWork: CBv2NativeCompletePrefixWork?
    private var nativeBound = false
    private var nativeMovedOrClosed = false
    // Legacy internal consumers only. Native consumers must take the actual
    // one-shot owner/work pair before borrowing a codec.
    var codec: CBv2CompleteCheckpointCodec {
        precondition(!nativeBound, "native plan codec requires its actual counted owner")
        return legacyCodec!
    }
    private(set) var destinationShapes: [[Int]]
    /// Native destinations differ from wire descriptors only for storage-only
    /// quantization. The compressed byte count never prices serving allocations.
    let destinationDescriptors: [CBv2CheckpointTensorDescriptor]
    var evaluateDestinations: ([MLXArray]) throws -> Void = { arrays in
        try withError { eval(arrays) }
    }

    init(
        codec: CBv2CompleteCheckpointCodec, manifest: CBv2CompleteCheckpointManifest,
        maximumSequenceLength: Int
    ) throws {
        legacyCodec = codec
        self.manifest = try manifest.owningMetadata(admission: codec.admission)
        self.maximumSequenceLength = maximumSequenceLength
        usesProcessMemoryOwner = codec.admission.hasProcessMemoryOwner
        let paged = try codec.pagedConfig.map {
            try CBv2PagedCheckpointStoragePlan(
                layerKinds: codec.layerKinds, config: $0, position: manifest.position,
                historicalLayout: codec.historicalLayout,
                maximumSequenceLength: maximumSequenceLength)
        }
        pagedStoragePlan = paged
        let nativeDescriptors = try codec.checkpointQuantization == nil
            ? manifest.tensors
            : codec.nativeTargetDescriptors(position: manifest.position)
                + Array(manifest.tensors.dropFirst(codec.targetTensorCount))
        destinationDescriptors = nativeDescriptors
        let destinations =
            paged == nil
            ? nativeDescriptors
            : Array(nativeDescriptors.dropFirst(codec.targetTensorCount))
        var shapes: [[Int]] = []
        var target = 0
        var auxiliary = 0
        var initializationScratch = 0
        for tensor in destinations {
            let isTarget = tensor.role == .keys || tensor.role == .values
            var shape = tensor.shape
            if isTarget {
                shape[2] =
                    codec.contiguousLayout?.layers.first(where: {
                        $0.modelLayer == tensor.layer
                    })?.window ?? maximumSequenceLength
            }
            let count = try CBv2CheckpointTensorDescriptor.checkedByteCount(
                shape: shape, dtype: tensor.dtype.mlxDType)
            let bound = try CBv2CheckpointAllocationFootprint.bound(count)
            if isTarget {
                target = try CBv2CheckpointAllocationFootprint.add(target, bound)
            } else {
                auxiliary = try CBv2CheckpointAllocationFootprint.add(auxiliary, bound)
            }
            initializationScratch = try CBv2CheckpointAllocationFootprint.add(
                initializationScratch,
                CBv2CheckpointAllocationFootprint.bound(tensor.dtype.mlxDType.size))
            shapes.append(shape)
        }
        destinationShapes = shapes
        var pageInitializationScratch = 0
        for group in paged?.groups ?? [] {
            pageInitializationScratch = max(
                pageInitializationScratch,
                try CBv2CheckpointAllocationFootprint.bound(group.key.dtype.size))
        }
        // Imported original-precision bands are private input DTOs. They
        // remain charged as scratch through the new active row's fenced copy,
        // rather than being mistaken for physical page growth at handoff.
        scratchBytes = try CBv2CheckpointAllocationFootprint.add(
            CBv2CheckpointAllocationFootprint.add(
                max(initializationScratch, pageInitializationScratch), paged?.nativeRecentBytes ?? 0),
            codec.checkpointQuantization == nil ? 0 : CBv2NativeCheckpointRowCodec.scratchBytes)
        let totalTarget = try CBv2CheckpointAllocationFootprint.add(
            target, paged?.pageNativeBytes ?? 0)
        nativeTargetBytes = totalTarget
        nativeAuxiliaryBytes = auxiliary
        nativeDestinationBytes = try CBv2CheckpointAllocationFootprint.add(totalTarget, auxiliary)
        // The native complete-MTP schema is distinct from target-only v2.
        // This is retained host witness capacity, never measured native bytes.
        var retainedHostBytes = 0
        if manifest.backendLayout == CBv2CompleteCheckpointManifest.contiguousAsymmetricMTPLayout
            || manifest.backendLayout == CBv2CompleteCheckpointManifest.pagedAsymmetricMTPLayout
        {
            let (tokens, overflow) = manifest.position.multipliedReportingOverflow(by: 8)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            retainedHostBytes = try CBv2CheckpointAllocationFootprint.add(tokens, 64 << 10)
        }
        if codec.isNativePagedHistorical, let paged {
            // Private segment/page maps and owner/control entries. This
            // conservative host allowance creates no map, native credit or
            // second owner; it stays on the original stage through retirement.
            let pages = try paged.layers.reduce(0) {
                try CBv2CheckpointAllocationFootprint.add($0, $1.pageCount)
            }
            guard let pageBytes = CBv2KVGeometry.multiply(pages, 512) else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            retainedHostBytes = try CBv2CheckpointAllocationFootprint.add(
                retainedHostBytes,
                CBv2CheckpointAllocationFootprint.add(
                    64 << 10,
                    CBv2CheckpointAllocationFootprint.add(pageBytes, paged.layers.count * 512)))
        }
        checkpointHostBytes = retainedHostBytes
        auxiliaryBytes = auxiliary
    }

    /// Engine starts the real loan BEFORE codec.plan(), then binds here before
    /// returning the public plan. No closure supplied by a provider can mint it.
    func bindNativeCompletePrefixWork(
        _ work: CBv2NativeCompletePrefixWork,
        store: any CBv2CompletePrefixCache, request: CBv2Request, engineID: UUID
    ) throws {
        guard let codec = stateLock.withLock({ legacyCodec }),
            work.purpose == .importing, work.codecIdentity == manifest.identity,
            (codec.contiguousLayout != nil && pagedStoragePlan == nil)
                || (codec.isNativePagedHistorical && pagedStoragePlan != nil)
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try work.validate(store: store, codec: codec, request: request, engineID: engineID)
        let owner = CBv2CompleteCheckpointCodecOwner(codec)
        try work.retain(owners: [owner])
        try stateLock.withLock {
            guard !nativeBound, !nativeMovedOrClosed, legacyCodec === codec else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            nativeBound = true
            nativeCodecOwner = owner
            nativeWork = work
            legacyCodec = nil
        }
    }

    public func allocate(onRelease: @escaping @Sendable () -> Void) throws
        -> CBv2CompleteCheckpointImport
    {
        let providerReservation = CBv2CheckpointReservation(onRelease: onRelease)
        let native:
            (
                CBv2CompleteCheckpointCodecOwner, CBv2NativeCompletePrefixWork,
                ([MLXArray]) throws -> Void
            )?
        do {
            native = try stateLock.withLock {
                guard nativeBound else { return nil }
                guard !nativeMovedOrClosed, let owner = nativeCodecOwner, let work = nativeWork
                else {
                    throw CBv2CompleteCheckpointError.closed
                }
                let evaluate = evaluateDestinations
                nativeMovedOrClosed = true
                nativeCodecOwner = nil
                nativeWork = nil
                // A retained consumed plan must not keep a test evaluator's
                // captures or any model/codecs behind the transferred loan.
                evaluateDestinations = { arrays in try withError { eval(arrays) } }
                return (owner, work, evaluate)
            }
        } catch {
            providerReservation.release()
            throw error
        }
        if let native {
            // This initializer registers an actual owner box on work BEFORE
            // reserve/allocate/eval, including every late-returned root.
            return try CBv2CompleteCheckpointImport(
                plan: self, nativeCodecOwner: native.0,
                nativeWork: native.1, evaluate: native.2, reservation: providerReservation)
        }

        // Original untracked allocation path. Native handling never falls
        // through to this destructor/refund behavior.
        do {
            let codec = self.codec
            let scratch = try CBv2CheckpointAllocationFootprint.add(
                scratchBytes,
                usesProcessMemoryOwner
                    ? 0 : CBv2CompleteCheckpointManifest.maximumProviderScratchBytes)
            if codec.contiguousLayout != nil {
                let aux = try CBv2CheckpointAllocationFootprint.add(
                    nativeAuxiliaryBytes, checkpointHostBytes)
                let stage = try codec.admission.reserveCheckpointStage(
                    targetBytes: nativeTargetBytes,
                    auxiliaryBytes: aux, scratchBytes: scratch)
                do {
                    return try .init(
                        plan: self, reservation: providerReservation, stageLease: stage)
                } catch {
                    stage.closeAfterDroppingOwners()
                    throw error
                }
            }
            if let pagedStoragePlan {
                let stage = try codec.admission.reserveCheckpointStage(
                    targetBytes: pagedStoragePlan.pageNativeBytes, auxiliaryBytes: auxiliaryBytes,
                    scratchBytes: scratch)
                do {
                    return try .init(
                        plan: self, reservation: providerReservation, stageLease: stage)
                } catch {
                    stage.closeAfterDroppingOwners()
                    throw error
                }
            }
            let bytes = try CBv2CheckpointAllocationFootprint.add(nativeDestinationBytes, scratch)
            let engineReservation = try codec.admission.reserveTransient(bytes: bytes)
            let reservation = CBv2CheckpointReservation {
                engineReservation.release()
                providerReservation.release()
            }
            return try .init(plan: self, reservation: reservation)
        } catch {
            providerReservation.release()
            throw error
        }
    }

    /// Explicitly abandon a deferred native plan. Untracked behavior is
    /// unchanged. A closed alias remains readable metadata, never a live codec.
    public func close() {
        let retired = stateLock.withLock {
            () -> (CBv2CompleteCheckpointCodecOwner, CBv2NativeCompletePrefixWork)? in
            guard nativeBound, !nativeMovedOrClosed, let owner = nativeCodecOwner,
                let work = nativeWork
            else { return nil }
            nativeMovedOrClosed = true
            nativeCodecOwner = nil
            nativeWork = nil
            evaluateDestinations = { arrays in try withError { eval(arrays) } }
            return (owner, work)
        }
        if let (owner, work) = retired {
            work.finishAfterDroppingConsumers { owner.dropNativeReferences() }
        }
    }

    deinit {
        close()
        destinationShapes.removeAll(keepingCapacity: false)
    }
}
