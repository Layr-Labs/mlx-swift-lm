import Cmlx
import Foundation
import MLX

/// No-copy source views stay private and immutable until close. Packing reads
/// only one bounded segment, including when KV has strided donor-capacity gaps.
public final class CBv2CompleteCheckpointExport: @unchecked Sendable {
    public let manifest: CBv2CompleteCheckpointManifest
    /// Shared process mode: the provider owns only its host I/O buffers;
    /// Admission already owns native packing scratch and donor backing.
    public let usesProcessMemoryOwner: Bool
    private let lock = NSLock()
    private var sources: [CBv2CompleteCheckpointTensorSource]?
    private var retainedOwners: [AnyObject]
    private weak var nativeWork: CBv2NativeCompletePrefixWork?
    private var nativeBound = false

    convenience init(
        manifest: CBv2CompleteCheckpointManifest, arrays: [MLXArray],
        usesProcessMemoryOwner: Bool = false, retainedOwners: [AnyObject] = []
    ) {
        self.init(
            manifest: manifest, sources: arrays.map { .array($0) },
            usesProcessMemoryOwner: usesProcessMemoryOwner, retainedOwners: retainedOwners)
    }

    init(
        manifest: CBv2CompleteCheckpointManifest, sources: [CBv2CompleteCheckpointTensorSource],
        usesProcessMemoryOwner: Bool = false, retainedOwners: [AnyObject] = []
    ) {
        self.manifest = manifest
        self.sources = sources
        self.usesProcessMemoryOwner = usesProcessMemoryOwner
        self.retainedOwners = retainedOwners
    }

    func bindNativeCompletePrefixWork(_ work: CBv2NativeCompletePrefixWork) throws {
        try lock.withLock {
            guard !nativeBound, let sources, work.purpose == .publication,
                manifest.identity == work.codecIdentity
            else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            // Non-array sources need the actual issued page-native codec.
            // Their immutable map/window owners are retained independently of
            // this public closeable wrapper before any readback can run.
            var arrays: [MLXArray] = []
            for source in sources {
                switch source {
                case .array(let array): arrays.append(array)
                case .paged(let value): try value.retainForNativeExport(work)
                case .historicalWindow(let value): try value.retainForNativeExport(work)
                }
            }
            try work.retain(arrays: arrays, owners: retainedOwners)
            nativeWork = work
            nativeBound = true
        }
    }

    public func readSegment(tensorIndex: Int, byteOffset: Int, maximumBytes: Int) throws -> Data {
        try lock.withLock {
            guard let sources else { throw CBv2CompleteCheckpointError.closed }
            guard sources.indices.contains(tensorIndex) else {
                throw CBv2CompleteCheckpointError.invalidSegment
            }
            if nativeBound {
                guard let nativeWork else { throw CBv2CompleteCheckpointError.closed }
                return try sources[tensorIndex].readSegment(
                    descriptor: manifest.tensors[tensorIndex],
                    byteOffset: byteOffset, maximumBytes: maximumBytes, nativeWork: nativeWork)
            }
            return try sources[tensorIndex].readSegment(
                descriptor: manifest.tensors[tensorIndex], byteOffset: byteOffset,
                maximumBytes: maximumBytes)
        }
    }

    public func close() {
        lock.withLock {
            sources?.forEach { $0.close() }
            sources = nil
            retainedOwners.removeAll()
            nativeWork = nil  // shared batch work is finished by Capture, never Export.close
        }
    }
    deinit { close() }
}

/// One actual native transfer payload. Work retains this box BEFORE any
/// reservation/allocation. Neither a failed initializer nor public close can
/// deinitialize its native roots or callbacks ahead of required completion.
final class CBv2NativeCompleteCheckpointImportOwner: @unchecked Sendable {
    let plan: CBv2CompleteCheckpointImportPlan
    let codecOwner: CBv2CompleteCheckpointCodecOwner
    var arrays: [MLXArray] = []
    var backing: CBv2ContiguousCheckpointBacking?
    var pagedStorage: CBv2PagedCheckpointStorage?
    var stageLease: CBv2CheckpointStageLease?
    var prepared: CBv2PreparedCompleteCheckpoint?
    var reservation: CBv2CheckpointReservation?
    var evaluate: (([MLXArray]) throws -> Void)?
    var tensorIndex = 0
    var byteOffset = 0
    var nativeDestinationBytes = 0

    init(
        plan: CBv2CompleteCheckpointImportPlan, codecOwner: CBv2CompleteCheckpointCodecOwner,
        reservation: CBv2CheckpointReservation, evaluate: @escaping ([MLXArray]) throws -> Void
    ) {
        self.plan = plan
        self.codecOwner = codecOwner
        self.reservation = reservation
        self.evaluate = evaluate
    }

    func allocate(work: CBv2NativeCompletePrefixWork) throws {
        try work.captureCurrentStreams()
        let codec = try codecOwner.borrow()
        guard
            (codec.contiguousLayout != nil && plan.pagedStoragePlan == nil)
                || (codec.isNativePagedHistorical && plan.pagedStoragePlan != nil)
        else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let scratch = try CBv2CheckpointAllocationFootprint.add(
            plan.scratchBytes,
            plan.usesProcessMemoryOwner
                ? 0 : CBv2CompleteCheckpointManifest.maximumProviderScratchBytes)
        let auxiliary = try CBv2CheckpointAllocationFootprint.add(
            plan.nativeAuxiliaryBytes, plan.checkpointHostBytes)
        let lease = try codec.admission.reserveCheckpointStage(
            targetBytes: plan.nativeTargetBytes,
            auxiliaryBytes: auxiliary, scratchBytes: scratch)
        stageLease = lease
        try work.retain(owners: [lease])
        let stream = StreamOrDevice.default
        do {
            try withError { fault in
                if let storagePlan = plan.pagedStoragePlan {
                    guard stream.stream == MLX.Stream.gpu else {
                        throw CBv2NativeShutdownError.unsupportedConsumer
                    }
                    pagedStorage = try .init(
                        plan: storagePlan,
                        evaluate: { [self] array in
                            try work.retain(arrays: [array])
                            guard let evaluate else { throw CBv2CompleteCheckpointError.closed }
                            try evaluate([array])
                        }, admission: codec.admission, nativeWork: work)
                    if let pagedStorage { try work.retain(owners: [pagedStorage]) }
                }
                let descriptors =
                    plan.pagedStoragePlan == nil
                    ? plan.manifest.tensors
                    : Array(plan.manifest.tensors.dropFirst(codec.targetTensorCount))
                for (shape, descriptor) in zip(plan.destinationShapes, descriptors) {
                    let value = MLXArray.zeros(
                        shape, dtype: descriptor.dtype.mlxDType, stream: stream)
                    arrays.append(value)
                    try work.retain(arrays: [value])  // includes a late value after first-winner failure
                    try fault.check()
                }
                guard let evaluate else { throw CBv2CompleteCheckpointError.closed }
                try evaluate(arrays)
                try fault.check()
            }
            try work.fenceForProtectedPromotion()
        } catch {
            // This is required native construction/evaluation/completion, not
            // a later metadata/accounting refusal. Never retry into health.
            work.requiredCompletionFailed()
            throw error
        }
        guard arrays.allSatisfy({ mlx_array_data_uint8($0.ctx) != nil }) else {
            work.requiredCompletionFailed()
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        let footprint = try CBv2CheckpointAllocationFootprint.freshBytes(arrays)
        nativeDestinationBytes = try CBv2CheckpointAllocationFootprint.add(
            footprint.actual, pagedStorage?.pageAllocatedBytes ?? 0)  // persistent destinations only
        if plan.pagedStoragePlan == nil {
            backing = try .init(
                lease: lease, codec: codec,
                arrays: Array(arrays.prefix(codec.targetTensorCount)),
                expectedBound: plan.nativeTargetBytes,
                auxiliaryArrays: Array(arrays.dropFirst(codec.targetTensorCount)),
                hostBytes: plan.checkpointHostBytes, position: plan.manifest.position)
            if let backing { try work.retain(owners: [backing]) }
        }
    }

    func append(
        tensorIndex: Int, byteOffset: Int, data: Data,
        work: CBv2NativeCompletePrefixWork
    ) throws {
        try work.captureCurrentStreams()
        guard tensorIndex == self.tensorIndex, byteOffset == self.byteOffset,
            plan.manifest.tensors.indices.contains(tensorIndex), !data.isEmpty,
            data.count <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let descriptor = plan.manifest.tensors[tensorIndex]
        let size = descriptor.dtype.mlxDType.size
        guard data.count % size == 0, data.count <= descriptor.byteCount - byteOffset else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let codec = try codecOwner.borrow()
        let isTarget = descriptor.role == .keys || descriptor.role == .values
        if let pagedStorage, isTarget {
            try pagedStorage.append(
                layerIndex: tensorIndex / 2, values: descriptor.role == .values,
                byteOffset: byteOffset, data: data)
            self.byteOffset += data.count
            if self.byteOffset == descriptor.byteCount {
                self.tensorIndex += 1
                self.byteOffset = 0
            }
            return
        }
        let arrayIndex = pagedStorage == nil ? tensorIndex : tensorIndex - codec.targetTensorCount
        guard arrays.indices.contains(arrayIndex),
            let pointer = mlx_array_data_uint8(arrays[arrayIndex].ctx)
        else {
            work.requiredCompletionFailed()
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        let ring =
            isTarget
            ? codec.contiguousLayout?.layers.first {
                $0.modelLayer == descriptor.layer && $0.window != nil
            } : nil
        let destination = UnsafeMutableRawPointer(mutating: pointer)
        let strides = CBv2CheckpointByteLayout.contiguousStrides(plan.destinationShapes[arrayIndex])
        data.withUnsafeBytes { source in
            if let ring, let window = ring.window {
                CBv2CheckpointByteLayout.copyRing(
                    shape: descriptor.shape, window: window,
                    firstPosition: ring.tokenStart(at: plan.manifest.position), itemSize: size,
                    byteOffset: byteOffset, count: data.count
                ) { physical, packed, length in
                    destination.advanced(by: physical).copyMemory(
                        from: source.baseAddress!.advanced(by: packed), byteCount: length)
                }
            } else {
                CBv2CheckpointByteLayout.copy(
                    shape: descriptor.shape, strides: strides, itemSize: size,
                    byteOffset: byteOffset, count: data.count
                ) { physical, packed, length in
                    destination.advanced(by: physical).copyMemory(
                        from: source.baseAddress!.advanced(by: packed), byteCount: length)
                }
            }
        }
        self.byteOffset += data.count
        if self.byteOffset == descriptor.byteCount {
            self.tensorIndex += 1
            self.byteOffset = 0
        }
    }

    func prepare(work: CBv2NativeCompletePrefixWork) throws {
        guard tensorIndex == plan.manifest.tensors.count, byteOffset == 0 else {
            throw CBv2CompleteCheckpointError.incompleteTransfer
        }
        try work.captureCurrentStreams()
        do {
            let codec = try codecOwner.borrow()
            try withError { fault in
                if let pagedStorage, let stageLease {
                    let frame = try CBv2PagedCheckpointFrame(
                        storage: pagedStorage, auxiliary: arrays,
                        lease: stageLease, hostBytes: plan.checkpointHostBytes)
                    prepared = .init(pagedFrame: frame)
                    if codec.assistant == nil {
                        prepared?.checkpoint = .init(
                            position: plan.manifest.position,
                            chunkSize: plan.manifest.chunkSize, layers: [:], byteCount: 0)
                    } else {
                        prepared?.checkpoint = try codec.recurrentCheckpoint(
                            manifest: plan.manifest, auxiliary: arrays)
                    }
                } else {
                    prepared = try codec.preparedState(
                        manifest: plan.manifest, arrays: arrays,
                        maximumSequenceLength: plan.maximumSequenceLength,
                        contiguousBacking: backing)
                }
                if let prepared { try work.retain(owners: [prepared]) }
                try fault.check()
            }
            try work.fenceForProtectedPromotion()
        } catch {
            if error is MLXError { work.requiredCompletionFailed() }
            throw error
        }
    }

    /// Invoked ONLY by work's queued post-completion callback. Mutate the
    /// actual shared box, so retained closed public wrappers keep no model or
    /// native state merely because ARC chose a later destruction point.
    func retireAfterCompletion() throws {
        // The paged suffix operation has its own actual loan/coverage roots.
        // Failure here keeps this whole callback/owner retained by prefix work.
        try prepared?.nativePagedPreparation?.retireAfterCompletion()
        arrays.removeAll(keepingCapacity: false)
        pagedStorage?.close()
        pagedStorage = nil
        backing = nil
        prepared?.clear()
        prepared = nil
        evaluate = nil
        codecOwner.dropNativeReferences()
        let lease = stageLease
        stageLease = nil
        let provider = reservation
        reservation = nil
        lease?.closeAfterDroppingOwners()
        provider?.release()
    }
}

/// Filled on one staging queue, before any array is exposed to GPU consumers.
/// Each zeroed MLX destination is exclusively owned here. Raw writes cannot
/// race lazy graphs: allocation is evaluated once, then only authenticated CPU
/// segments touch the buffers until finish transfers their ownership.
public final class CBv2CompleteCheckpointImport: @unchecked Sendable {
    private let lock = NSLock()
    private let plan: CBv2CompleteCheckpointImportPlan
    private var arrays: [MLXArray]?
    private var pagedStorage: CBv2PagedCheckpointStorage?
    private var stageLease: CBv2CheckpointStageLease?
    private var reservation: CBv2CheckpointReservation?
    private var tensorIndex = 0
    private var byteOffset = 0
    private var nativeDestinationBytes = 0
    private var contiguousBacking: CBv2ContiguousCheckpointBacking?
    private var nativeOwner: CBv2NativeCompleteCheckpointImportOwner?
    private var nativeWork: CBv2NativeCompletePrefixWork?

    init(
        plan: CBv2CompleteCheckpointImportPlan, nativeCodecOwner: CBv2CompleteCheckpointCodecOwner,
        nativeWork: CBv2NativeCompletePrefixWork, evaluate: @escaping ([MLXArray]) throws -> Void,
        reservation: CBv2CheckpointReservation
    ) throws {
        self.plan = plan
        let owner = CBv2NativeCompleteCheckpointImportOwner(
            plan: plan, codecOwner: nativeCodecOwner,
            reservation: reservation, evaluate: evaluate)
        self.nativeOwner = owner
        self.nativeWork = nativeWork
        do {
            try nativeWork.retain(owners: [owner])
            try owner.allocate(work: nativeWork)
        } catch {
            self.nativeOwner = nil
            self.nativeWork = nil
            nativeWork.finishAfterDroppingConsumers { try owner.retireAfterCompletion() }
            throw error
        }
    }

    init(
        plan: CBv2CompleteCheckpointImportPlan, reservation: CBv2CheckpointReservation,
        stageLease: CBv2CheckpointStageLease? = nil
    ) throws {
        self.plan = plan
        self.reservation = reservation
        self.stageLease = stageLease
        if let storagePlan = plan.pagedStoragePlan {
            pagedStorage = try CBv2PagedCheckpointStorage(
                plan: storagePlan, evaluate: { try plan.evaluateDestinations([$0]) },
                admission: plan.codec.admission)
        }
        let destinationDescriptors =
            plan.pagedStoragePlan == nil
            ? plan.manifest.tensors
            : Array(plan.manifest.tensors.dropFirst(plan.codec.targetTensorCount))
        let allocationStream = StreamOrDevice.default
        let destinations = try withError { fault in
            let values = zip(plan.destinationShapes, destinationDescriptors).map { shape, tensor in
                MLXArray.zeros(shape, dtype: tensor.dtype.mlxDType, stream: allocationStream)
            }
            do {
                try fault.check()
                try plan.evaluateDestinations(values)
            } catch {
                // Drain partial work before its stage charge can be returned.
                allocationStream.stream.synchronize()
                if error is MLXError { throw error }
                try fault.check()
                throw error
            }
            // Evaluation can signal before Metal completion drops temporary
            // Data references. Fence once before checking exclusive ownership.
            allocationStream.stream.synchronize()
            try fault.check()
            return values
        }
        guard destinations.allSatisfy({ mlx_array_data_uint8($0.ctx) != nil }) else {
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        let footprint = try CBv2CheckpointAllocationFootprint.freshBytes(destinations)
        nativeDestinationBytes = try CBv2CheckpointAllocationFootprint.add(
            footprint.actual, pagedStorage?.pageAllocatedBytes ?? 0)
        if plan.codec.contiguousLayout != nil {
            guard let stageLease else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            contiguousBacking = try .init(
                lease: stageLease, codec: plan.codec,
                arrays: Array(destinations.prefix(plan.codec.targetTensorCount)),
                expectedBound: plan.nativeTargetBytes,
                auxiliaryArrays: Array(destinations.dropFirst(plan.codec.targetTensorCount)),
                hostBytes: plan.checkpointHostBytes, position: plan.manifest.position)
        }
        arrays = destinations
    }

    public func appendSegment(tensorIndex: Int, byteOffset: Int, data: Data) throws {
        lock.lock()
        defer { lock.unlock() }
        if let nativeOwner, let nativeWork {
            try nativeOwner.append(
                tensorIndex: tensorIndex, byteOffset: byteOffset, data: data, work: nativeWork)
            return
        }
        guard let arrays else { throw CBv2CompleteCheckpointError.closed }
        guard tensorIndex == self.tensorIndex, byteOffset == self.byteOffset,
            plan.manifest.tensors.indices.contains(tensorIndex), !data.isEmpty,
            data.count <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        let descriptor = plan.manifest.tensors[tensorIndex]
        let itemSize = descriptor.dtype.mlxDType.size
        guard data.count % itemSize == 0, data.count <= descriptor.byteCount - byteOffset else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let kvTensorCount = plan.codec.targetTensorCount
        if let pagedStorage, tensorIndex < kvTensorCount {
            try pagedStorage.append(
                layerIndex: tensorIndex / 2, values: tensorIndex % 2 == 1,
                byteOffset: byteOffset, data: data)
        } else {
            let index = pagedStorage == nil ? tensorIndex : tensorIndex - kvTensorCount
            guard let pointer = mlx_array_data_uint8(arrays[index].ctx) else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            // Fresh, evaluated, unpublished destinations. This writable
            // pointer never escapes or survives the move performed by finish.
            let destination = UnsafeMutableRawPointer(mutating: pointer)
            let strides = CBv2CheckpointByteLayout.contiguousStrides(plan.destinationShapes[index])
            let isTarget = descriptor.role == .keys || descriptor.role == .values
            let ring =
                isTarget
                ? plan.codec.contiguousLayout?.layers.first {
                    $0.modelLayer == descriptor.layer && $0.window != nil
                } : nil
            data.withUnsafeBytes { source in
                if let ring, let window = ring.window {
                    CBv2CheckpointByteLayout.copyRing(
                        shape: descriptor.shape, window: window,
                        firstPosition: ring.tokenStart(at: plan.manifest.position),
                        itemSize: itemSize,
                        byteOffset: byteOffset, count: data.count
                    ) { physicalOffset, packedOffset, length in
                        destination.advanced(by: physicalOffset).copyMemory(
                            from: source.baseAddress!.advanced(by: packedOffset), byteCount: length)
                    }
                    return
                }
                CBv2CheckpointByteLayout.copy(
                    shape: descriptor.shape, strides: strides, itemSize: itemSize,
                    byteOffset: byteOffset, count: data.count
                ) { physicalOffset, packedOffset, length in
                    destination.advanced(by: physicalOffset).copyMemory(
                        from: source.baseAddress!.advanced(by: packedOffset), byteCount: length)
                }
            }
        }
        self.byteOffset += data.count
        if self.byteOffset == descriptor.byteCount {
            self.tensorIndex += 1
            self.byteOffset = 0
        }
    }

    public func finish() throws -> CBv2StagedCompleteCheckpoint {
        lock.lock()
        defer { lock.unlock() }
        if let nativeOwner, let nativeWork {
            try nativeOwner.prepare(work: nativeWork)
            let result = CBv2StagedCompleteCheckpoint(
                nativeOwner: nativeOwner, nativeWork: nativeWork)
            self.nativeOwner = nil
            self.nativeWork = nil
            return result
        }
        guard let arrays, let reservation else { throw CBv2CompleteCheckpointError.closed }
        guard tensorIndex == plan.manifest.tensors.count, byteOffset == 0 else {
            throw CBv2CompleteCheckpointError.incompleteTransfer
        }
        let prepared: CBv2PreparedCompleteCheckpoint
        if let pagedStorage, let stageLease {
            prepared = .init(
                pagedFrame: try .init(storage: pagedStorage, auxiliary: arrays, lease: stageLease))
        } else {
            prepared = try plan.codec.preparedState(
                manifest: plan.manifest, arrays: arrays,
                maximumSequenceLength: plan.maximumSequenceLength,
                contiguousBacking: contiguousBacking)
        }
        let result = CBv2StagedCompleteCheckpoint(
            plan: plan, prepared: prepared,
            nativeDestinationBytes: nativeDestinationBytes, reservation: reservation)
        self.arrays = nil
        self.contiguousBacking = nil
        self.pagedStorage = nil
        self.stageLease = nil
        self.reservation = nil
        return result
    }

    public func close() {
        lock.lock()
        if let nativeOwner, let nativeWork {
            self.nativeOwner = nil
            self.nativeWork = nil
            lock.unlock()
            nativeWork.finishAfterDroppingConsumers { try nativeOwner.retireAfterCompletion() }
            return
        }
        arrays = nil
        contiguousBacking = nil
        pagedStorage?.close()
        pagedStorage = nil
        let stageLease = self.stageLease
        self.stageLease = nil
        let reservation = self.reservation
        self.reservation = nil
        lock.unlock()
        stageLease?.closeAfterDroppingOwners()
        reservation?.release()
    }

    deinit { close() }
}

/// Single-use request state, not a cache entry. close drops buffers before its
/// provider reservation. Successful engine adoption releases that reservation
/// only after both active admission and the backend have charged ownership.
public final class CBv2StagedCompleteCheckpoint: @unchecked Sendable {
    public let manifest: CBv2CompleteCheckpointManifest
    public let maximumSequenceLength: Int
    public let nativeDestinationBytes: Int
    private let legacyCodec: CBv2CompleteCheckpointCodec?
    /// Native code must borrow the codec together with validated work below.
    /// The existing paged/untracked adoption API retains its original access.
    var codec: CBv2CompleteCheckpointCodec {
        precondition(!hasNativeTracking, "native stage requires its counted work")
        return legacyCodec!
    }
    let usesPagedBacking: Bool
    let usesDestinationTransfer: Bool
    let hasNativeTracking: Bool
    private let lock = NSLock()
    private var prepared: CBv2PreparedCompleteCheckpoint?
    private var reservation: CBv2CheckpointReservation?
    private var nativeOwner: CBv2NativeCompleteCheckpointImportOwner?
    private var nativeWork: CBv2NativeCompletePrefixWork?

    init(
        plan: CBv2CompleteCheckpointImportPlan, prepared: CBv2PreparedCompleteCheckpoint,
        nativeDestinationBytes: Int, reservation: CBv2CheckpointReservation
    ) {
        manifest = plan.manifest
        maximumSequenceLength = plan.maximumSequenceLength
        self.nativeDestinationBytes = nativeDestinationBytes
        legacyCodec = plan.codec
        let paged = plan.codec.pagedConfig != nil
        usesPagedBacking = paged
        usesDestinationTransfer = paged || plan.codec.contiguousLayout != nil
        hasNativeTracking = false
        self.prepared = prepared
        self.reservation = reservation
    }

    init(
        nativeOwner: CBv2NativeCompleteCheckpointImportOwner,
        nativeWork: CBv2NativeCompletePrefixWork
    ) {
        manifest = nativeOwner.plan.manifest
        maximumSequenceLength = nativeOwner.plan.maximumSequenceLength
        nativeDestinationBytes = nativeOwner.nativeDestinationBytes
        legacyCodec = nil
        usesPagedBacking = nativeOwner.plan.pagedStoragePlan != nil
        usesDestinationTransfer = true
        hasNativeTracking = true
        self.nativeOwner = nativeOwner
        self.nativeWork = nativeWork
    }

    /// Lookup-only native preparation, before enqueue/deadline beginCommit.
    /// Serialize with close, but never hold an outcome/commit lock here.
    func prepareNativeHistoricalAssistant(
        store: any CBv2CompletePrefixCache,
        request: CBv2Request, engineID: UUID, expectedCodec: CBv2CompleteCheckpointCodec
    ) throws {
        try lock.withLock {
            guard hasNativeTracking, let nativeOwner, let nativeWork,
                let prepared = nativeOwner.prepared
            else {
                throw CBv2CompleteCheckpointError.closed
            }
            let codec = try nativeOwner.codecOwner.borrow()
            guard codec === expectedCodec else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            try nativeWork.validate(
                store: store, codec: codec, request: request, engineID: engineID)
            // No restoration or extra fence for target-only imports.
            guard codec.assistant != nil else { return }
            if prepared.historicalAssistantRestoration?.settled == true { return }
            try nativeWork.retain(owners: [prepared])
            try prepared.prepareHistoricalAssistant(
                codec: codec, request: request, work: nativeWork)
            guard nativeWork.hasProtectedPromotionCompletion else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
        }
    }

    /// Metadata-only validated borrow. close cannot detach this owner while
    /// lookup is using it. The callback must not return native/model aliases.
    func withValidatedNativeCodec<Result>(
        store: any CBv2CompletePrefixCache, request: CBv2Request, engineID: UUID,
        expectedCodec: CBv2CompleteCheckpointCodec,
        _ body: (CBv2CompleteCheckpointCodec) throws -> Result
    ) throws -> Result {
        try lock.withLock {
            guard hasNativeTracking, let nativeOwner, let nativeWork else {
                throw CBv2CompleteCheckpointError.closed
            }
            let codec = try nativeOwner.codecOwner.borrow()
            guard codec === expectedCodec else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            try nativeWork.validate(
                store: store, codec: codec, request: request, engineID: engineID)
            guard nativeWork.hasProtectedPromotionCompletion else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
            return try body(codec)
        }
    }

    /// Engine queue, outside the outcome commit. It MUST be followed by
    /// publication/refusal in this same turn, without letting another request
    /// mutate the pool's replacement-map inputs.
    func prepareNativePagedTargets(
        store: any CBv2CompletePrefixCache,
        request: CBv2Request, engineID: UUID, expectedCodec: CBv2CompleteCheckpointCodec,
        backend: PagedKVBackend, streamGeneration: UInt64
    ) throws -> CBv2PreparedNativePagedCheckpoint {
        try lock.withLock {
            guard hasNativeTracking, usesPagedBacking, let nativeOwner, let nativeWork,
                let prepared = nativeOwner.prepared, let frame = prepared.pagedFrame,
                prepared.nativePagedPreparation == nil
            else {
                throw CBv2CompleteCheckpointError.closed
            }
            let codec = try nativeOwner.codecOwner.borrow()
            guard codec === expectedCodec, codec.isNativePagedHistorical else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            try nativeWork.validate(
                store: store, codec: codec, request: request, engineID: engineID)
            let candidate = try CBv2PreparedNativePagedCheckpoint(
                backend: backend, codec: codec,
                work: nativeWork, request: request, streamGeneration: streamGeneration,
                maximumTokens: maximumSequenceLength)
            prepared.nativePagedPreparation = candidate  // owns failed/late partial preparation
            try nativeWork.retain(owners: [candidate])
            try candidate.prepare(frame: frame, admission: codec.admission)
            prepared.pagedFrame = nil  // consumed frame has no arrays/lease to refund
            return candidate
        }
    }

    /// Move the actual import owner/work exactly once. The callback performs
    /// the engine target+assistant adoption transaction. It receives the
    /// same loan so any new native restoration roots are retained BEFORE eval.
    func consumeNativePreparedState<Result>(
        store: any CBv2CompletePrefixCache, request: CBv2Request, engineID: UUID,
        expectedCodec: CBv2CompleteCheckpointCodec,
        _ adopt: (
            CBv2PreparedCompleteCheckpoint, CBv2CompleteCheckpointCodec,
            CBv2NativeCompletePrefixWork
        ) throws -> Result
    ) throws -> Result {
        // Refuse a foreign/repeated caller BEFORE moving the original owner.
        // Validation is metadata-only; no eval/fence runs under this lock.
        let moved = try lock.withLock {
            () -> (
                CBv2NativeCompleteCheckpointImportOwner,
                CBv2NativeCompletePrefixWork
            ) in
            guard hasNativeTracking, let owner = nativeOwner, let work = nativeWork else {
                throw CBv2CompleteCheckpointError.closed
            }
            let codec = try owner.codecOwner.borrow()
            guard codec === expectedCodec else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            try work.validate(store: store, codec: codec, request: request, engineID: engineID)
            guard work.hasProtectedPromotionCompletion, owner.prepared != nil else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
            nativeOwner = nil
            nativeWork = nil
            return (owner, work)
        }
        let (owner, work) = moved
        defer { work.finishAfterDroppingConsumers { try owner.retireAfterCompletion() } }
        let codec = try owner.codecOwner.borrow()
        guard let prepared = owner.prepared else {
            throw CBv2CompleteCheckpointError.incompleteTransfer
        }
        // This callback is metadata-only under the native commit. A backend
        // admission/registration veto is ordinary even if a test wraps it as
        // MLXError; all real native completion failures were sealed in the
        // off-side preparation phase, before this point.
        return try adopt(prepared, codec, work)
    }

    func consumePreparedState<Result>(_ adopt: (CBv2PreparedCompleteCheckpoint) throws -> Result)
        throws -> Result
    {
        guard !hasNativeTracking else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        lock.lock()
        guard let prepared else {
            lock.unlock()
            throw CBv2CompleteCheckpointError.closed
        }
        self.prepared = nil
        let reservation = self.reservation
        self.reservation = nil
        lock.unlock()
        defer {
            prepared.clear()
            reservation?.release()
        }
        return try adopt(prepared)
    }

    public func close() {
        lock.lock()
        if let nativeOwner, let nativeWork {
            self.nativeOwner = nil
            self.nativeWork = nil
            lock.unlock()
            nativeWork.finishAfterDroppingConsumers { try nativeOwner.retireAfterCompletion() }
            return
        }
        prepared?.clear()
        prepared = nil
        let reservation = self.reservation
        self.reservation = nil
        lock.unlock()
        reservation?.release()
    }
    deinit { close() }
}

/// Slot-owned scratch for the initial encrypted manifest read. Import-plan
/// allocation later reserves its own native destinations and transfer scratch;
/// callers release this lease only after that reservation succeeds, or after
/// abandoning the read. There are no tensor allocations in this owner.
public final class CBv2CompleteCheckpointIOLease: @unchecked Sendable {
    /// A bound native owner excludes provider IO; the caller must supply its
    /// own process host-buffer reservation before reading the manifest.
    public let usesProcessMemoryOwner: Bool
    private let reservation: CBv2CheckpointReservation

    init(reservation: CBv2CheckpointReservation, usesProcessMemoryOwner: Bool = false) {
        self.reservation = reservation
        self.usesProcessMemoryOwner = usesProcessMemoryOwner
    }

    public func close() { reservation.release() }
    deinit { close() }
}

final class CBv2CheckpointReservation: @unchecked Sendable {
    private let lock = NSLock()
    private var onRelease: (@Sendable () -> Void)?
    init(onRelease: @escaping @Sendable () -> Void) { self.onRelease = onRelease }
    func release() {
        lock.lock()
        let callback = onRelease
        onRelease = nil
        lock.unlock()
        callback?()
    }
    deinit { release() }
}

/// Map a logical packed span onto contiguous runs in strided native storage.
/// All products were checked while validating the descriptor/allocation plan.
enum CBv2CheckpointByteLayout {
    /// Checked rank-four destination geometry belongs to the immutable import
    /// plan. Split at feature boundaries so head/temporal wrap cannot alias.
    static func copyRing(
        shape: [Int], window: Int, firstPosition: Int, itemSize: Int,
        byteOffset: Int, count: Int, run: (Int, Int, Int) -> Void
    ) {
        let width = shape[3]
        let tokens = shape[2]
        var element = byteOffset / itemSize
        var copied = 0
        while copied < count {
            let feature = element % width
            let token = (element / width) % tokens
            let head = element / (width * tokens)
            let slot = (firstPosition + token) % window
            let length = min((width - feature) * itemSize, count - copied)
            run(((head * window + slot) * width + feature) * itemSize, copied, length)
            copied += length
            element += length / itemSize
        }
    }
    static func contiguousStrides(_ shape: [Int]) -> [Int] {
        var result = Array(repeating: 1, count: shape.count)
        for index in stride(from: shape.count - 2, through: 0, by: -1) {
            result[index] = result[index + 1] * shape[index + 1]
        }
        return result
    }

    static func copy(
        shape: [Int], strides: [Int], itemSize: Int, byteOffset: Int, count: Int,
        run: (Int, Int, Int) -> Void
    ) {
        var contiguousElements = 1
        for index in shape.indices.reversed() {
            if shape[index] == 1 { continue }
            guard strides[index] == contiguousElements else { break }
            contiguousElements *= shape[index]
        }
        var logicalElement = byteOffset / itemSize
        var copied = 0
        while copied < count {
            var remainder = logicalElement
            var physicalElement = 0
            for index in shape.indices.reversed() {
                physicalElement += (remainder % shape[index]) * strides[index]
                remainder /= shape[index]
            }
            let available = (contiguousElements - logicalElement % contiguousElements) * itemSize
            let bytes = min(available, count - copied)
            run(physicalElement * itemSize, copied, bytes)
            copied += bytes
            logicalElement += bytes / itemSize
        }
    }
}
