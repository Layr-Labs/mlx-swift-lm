// Copyright © 2026 Eigen Labs.
import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum MiMoV26SerialLoadError: Error, Equatable, Sendable {
    case alreadyConsumed, reservationMismatch, insufficientReservation
    case cancelled, invalidSourceHandles, unmaterializedSource, unsupportedScope
}

/// Root-bundle identity only. External audio-tokenizer/DFlash/original-head
/// assets are not authorized by this request and have no loading entrypoint here.
public struct MiMoV26SerialLoadBinding: Equatable, Encodable, Sendable {
    public struct Shard: Equatable, Encodable, Sendable {
        public let name: String
        public let object: MiMoV26FilesystemObjectState
        public let headerSHA256: String
        public let tensorBytes: Int
    }
    public let contract, scope, canonicalRoot: String
    public let configurationObject, indexObject: MiMoV26FilesystemObjectState
    public let configSHA256, indexSHA256, descriptorSHA256: String
    public let sourceRepository, sourceRevision, conversionManifestSHA256: String
    public let payloadVerificationReceiptSHA256: String?
    public let shards: [Shard]
    public let rootFileBytes, tensorBytes, metadataBytesRead: Int
    public let estimate: MiMoV26LoadFootprint.Estimate

    fileprivate init(plan: MiMoV26FilesystemLoadPlan, estimate: MiMoV26LoadFootprint.Estimate) {
        contract = MiMoV26LoadFootprint.contract; scope = "root-bundle-only"
        canonicalRoot = plan.canonicalRoot.path
        configurationObject = plan.configurationObject; indexObject = plan.indexObject
        configSHA256 = plan.bundlePlan.configSHA256; indexSHA256 = plan.bundlePlan.indexSHA256
        descriptorSHA256 = plan.bundlePlan.descriptorSHA256
        let provenance = plan.bundlePlan.provenance
        sourceRepository = provenance.sourceRepository; sourceRevision = provenance.sourceRevision
        conversionManifestSHA256 = provenance.conversionManifestSHA256
        payloadVerificationReceiptSHA256 = provenance.payloadVerificationReceiptSHA256
        shards = plan.shards.map { .init(name: $0.name, object: $0.objectState,
            headerSHA256: $0.headerSHA256, tensorBytes: $0.tensorBytes) }
        rootFileBytes = plan.totalFileBytes; tensorBytes = plan.bundlePlan.tensorBytes
        metadataBytesRead = plan.metadataBytesRead; self.estimate = estimate
    }

    public func fingerprint() throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return SHA256.hash(data: try encoder.encode(self)).map { String(format: "%02x", $0) }.joined()
    }
}

/// Created only by the serial transaction, never from an asserted policy name.
/// The nonce prevents a reservation for another session—even the same files—
/// from being silently reused. Hashes are prior evidence references, not a fresh
/// payload readback or a filesystem immutability guarantee.
public struct MiMoV26SerialLoadRequest: Equatable, Sendable {
    public let sessionID: UUID
    public let binding: MiMoV26SerialLoadBinding
    public var requiredLoadBytes: UInt64 { binding.estimate.totalBytes }
    fileprivate init(binding: MiMoV26SerialLoadBinding) {
        sessionID = UUID(); self.binding = binding
    }
}

public enum MiMoV26SerialLoadPhase: String, Codable, Sendable {
    case admitted, sourceHandles, sourceMaterialization, parameterMaterialization, complete
}
public struct MiMoV26SerialLoadProgress: Sendable {
    public let phase: MiMoV26SerialLoadPhase
    public let sourceTensorsCompleted, sourceTensorCount: Int
    /// Owner-proven source payload already synchronously evaluated. This is
    /// NOT process memory, allocator peak or an inferred global-memory delta.
    public let materializedSourcePayloadBytes: UInt64
    public let parametersCompleted, parameterCount: Int
    public let currentShard, currentTensor: String?
}

/// Host-owned real allocation permit. The provider implementation must bind an
/// actual pending-load owner, current policy and activation/KV/other commitments;
/// a diagnostic implementation must label its narrower scope. Metadata alone
/// is never a permit. Validation is synchronous at safe native boundaries.
///
/// The host retains/reconciles this owner through publication or actual resource
/// retirement. This SDK never releases host credit automatically. In particular
/// it must not infer materialization credit from unrelated MLX usage changes.
public protocol MiMoV26SerialLoadReservation: AnyObject {
    var request: MiMoV26SerialLoadRequest { get }
    var reservedLoadBytes: UInt64 { get }
    func validateActive(progress: MiMoV26SerialLoadProgress) throws
}

public struct MiMoV26SerialLoadReceipt: Encodable, Sendable {
    public let sessionID: UUID
    public let binding: MiMoV26SerialLoadBinding
    public let sourceTensorCount, parameterCount: Int
    public let materializedSourcePayloadBytes: UInt64
    public let payloadHashesRecomputed = false
    public let externalComponentsLoaded = false
    public let materializationPolicy = "serial-source-then-serial-verified-parameters"
}

/// Transfer this owner intact into the factory/container until the host has
/// reconciled actual resident resources and published the slot. The models are
/// not Sendable; use the normal serial ModelContainer access rules. Arbitrary
/// subsequent Module mutation invalidates the loaded-generation contract.
public struct MiMoV26SerialLoadResult {
    public let bundle: MiMoV26ConvertedBundle
    public let receipt: MiMoV26SerialLoadReceipt
    public let reservation: any MiMoV26SerialLoadReservation
}

/// One-shot strict loader. No injected evaluator, dtype conversion, alternate
/// backend, parallel/bulk mode or external-sidecar loading is exposed. Models and
/// native handles remain private until all checked synchronous work completes.
/// Source files must remain exclusively immutable through load/retirement; stat
/// revalidation detects ordinary mutation, not privileged timestamp tampering.
public final class MiMoV26SerialLoadSession {
    public let request: MiMoV26SerialLoadRequest
    private let plan: MiMoV26FilesystemLoadPlan
    private let lock = NSLock()
    private var consumed = false

    public init(plan: MiMoV26FilesystemLoadPlan) throws {
        let estimate = try MiMoV26LoadFootprint.estimate(plan: plan)
        self.plan = plan
        request = .init(binding: .init(plan: plan, estimate: estimate))
    }

    public func load(
        reservation: any MiMoV26SerialLoadReservation,
        retaining work: NativeConstructionScope,
        isCancelled: () -> Bool = { false },
        progress: (MiMoV26SerialLoadProgress) throws -> Void = { _ in }
    ) throws -> MiMoV26SerialLoadResult {
        try work.withPhase(.serialLoad) {
            try work.retainOwner(reservation)
            return try loadInScope(reservation: reservation, work: work,
                                   isCancelled: isCancelled, progress: progress)
        }
    }

    private func loadInScope(
        reservation: any MiMoV26SerialLoadReservation, work: NativeConstructionScope,
        isCancelled: () -> Bool,
        progress: (MiMoV26SerialLoadProgress) throws -> Void
    ) throws -> MiMoV26SerialLoadResult {
        try lock.withLock {
            guard !consumed else { throw MiMoV26SerialLoadError.alreadyConsumed }
            consumed = true
        }
        guard reservation.request == request else { throw MiMoV26SerialLoadError.reservationMismatch }
        guard reservation.reservedLoadBytes >= request.requiredLoadBytes else {
            throw MiMoV26SerialLoadError.insufficientReservation
        }
        var sourceCount = 0, sourceBytes: UInt64 = 0, parameterCount = 0
        var completedParameters = 0
        func checkpoint(_ phase: MiMoV26SerialLoadPhase, shard: String? = nil, tensor: String? = nil) throws {
            guard !isCancelled() else { throw MiMoV26SerialLoadError.cancelled }
            let value = MiMoV26SerialLoadProgress(phase: phase,
                sourceTensorsCompleted: sourceCount, sourceTensorCount: plan.bundlePlan.descriptors.count,
                materializedSourcePayloadBytes: sourceBytes,
                parametersCompleted: completedParameters, parameterCount: parameterCount,
                currentShard: shard, currentTensor: tensor)
            // Admission may have awaited or a callback may replace/revoke state.
            // Recheck binding and permit on both sides of each host callback.
            try revalidate(reservation, isCancelled: isCancelled,
                recomputeEstimate: phase == .admitted || phase == .complete)
            try reservation.validateActive(progress: value)
            try revalidate(reservation, isCancelled: isCancelled)
            try progress(value)
            try revalidate(reservation, isCancelled: isCancelled)
            try reservation.validateActive(progress: value)
            try revalidate(reservation, isCancelled: isCancelled)
        }
        try checkpoint(.admitted)
        var loaded = try MiMoV26FilesystemWeights.load(plan: plan, retaining: work, isCancelled: isCancelled) { value in
            try checkpoint(.sourceHandles, shard: value.currentFile)
        }
        var handles = loaded.takeMaterializationHandles()
        defer { handles.removeAll() }
        guard loaded.materializationHandleCount == 0,
            handles.count == plan.bundlePlan.descriptors.count,
            Set(handles.map(\.name)) == Set(plan.bundlePlan.descriptors.keys) else {
            throw MiMoV26SerialLoadError.invalidSourceHandles
        }
        var expectedBytes: UInt64 = 0
        for handle in handles {
            guard handle.byteCount > 0, handle.byteCount == handle.array.nbytes else {
                throw MiMoV26SerialLoadError.invalidSourceHandles
            }
            let next = expectedBytes.addingReportingOverflow(UInt64(handle.byteCount))
            guard !next.overflow else { throw MiMoV26SerialLoadError.invalidSourceHandles }
            expectedBytes = next.partialValue
        }
        guard expectedBytes == UInt64(plan.bundlePlan.tensorBytes) else {
            throw MiMoV26SerialLoadError.invalidSourceHandles
        }
        for (left, right) in zip(handles, handles.dropFirst()) {
            guard left.shard < right.shard || (left.shard == right.shard && left.dataOffset < right.dataOffset) else {
                throw MiMoV26SerialLoadError.invalidSourceHandles
            }
        }
        var previousShard: String?
        for (index, handle) in handles.enumerated() {
            guard !isCancelled() else { throw MiMoV26SerialLoadError.cancelled }
            if index.isMultiple(of: 64) || previousShard != handle.shard {
                try checkpoint(.sourceMaterialization, shard: handle.shard, tensor: handle.name)
            }
            // Only exact native Load roots, never constructor/random expressions.
            try work.retain(handle.array)
            try work.capture(StreamOrDevice.default.stream)
            try work.willSubmit()
            try withError { error in eval(handle.array); try error.check() }
            try work.checkpoint("serial.sourceMaterialized")
            guard let info = try handle.array.evaluatedBufferInfo(),
                info.allocatedBytes >= handle.byteCount, info.dataOffset == 0,
                info.isRowContiguous else { throw MiMoV26SerialLoadError.unmaterializedSource }
            sourceCount += 1; sourceBytes += UInt64(handle.byteCount)
            previousShard = handle.shard
        }
        // Keep the original roots distinct from sanitized parameter views. Eval
        // verified final parameters serially while the load/copy permit is still
        // held; do not defer their transforms until after releasing that permit.
        let modules: [(String, Module)] = [("target", loaded.bundle.target),
            ("vision", loaded.bundle.vision), ("audioPatch", loaded.bundle.audioPatch),
            ("mtp", loaded.bundle.mtp)]
        let parameters = modules.flatMap { component, module in
            module.parameters().flattened().map { (component + "." + $0.0, $0.1) }
        }.sorted { $0.0 < $1.0 }
        parameterCount = parameters.count
        guard parameterCount == handles.count else { throw MiMoV26SerialLoadError.invalidSourceHandles }
        for (index, item) in parameters.enumerated() {
            guard !isCancelled() else { throw MiMoV26SerialLoadError.cancelled }
            let (name, parameter) = item
            if index.isMultiple(of: 64) {
                try checkpoint(.parameterMaterialization, tensor: name)
            }
            try work.retain(parameter)
            try work.capture(StreamOrDevice.default.stream)
            try work.willSubmit()
            try withError { error in eval(parameter); try error.check() }
            try work.checkpoint("serial.parameterMaterialized")
            guard let info = try parameter.evaluatedBufferInfo(), info.allocatedBytes >= parameter.nbytes else {
                throw MiMoV26SerialLoadError.unmaterializedSource
            }
            completedParameters += 1
        }
        handles.removeAll()
        try checkpoint(.complete)
        return .init(bundle: loaded.bundle, receipt: .init(sessionID: request.sessionID,
            binding: request.binding, sourceTensorCount: sourceCount, parameterCount: parameterCount,
            materializedSourcePayloadBytes: sourceBytes), reservation: reservation)
    }

    private func revalidate(_ reservation: any MiMoV26SerialLoadReservation,
                            isCancelled: () -> Bool, recomputeEstimate: Bool = false) throws {
        guard !isCancelled() else { throw MiMoV26SerialLoadError.cancelled }
        guard reservation.request == request else { throw MiMoV26SerialLoadError.reservationMismatch }
        guard reservation.reservedLoadBytes >= request.requiredLoadBytes else {
            throw MiMoV26SerialLoadError.insufficientReservation
        }
        try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan, isCancelled: isCancelled)
        if recomputeEstimate {
            let estimate = try MiMoV26LoadFootprint.estimate(plan: plan)
            guard MiMoV26SerialLoadBinding(plan: plan, estimate: estimate) == request.binding else {
                throw MiMoV26SerialLoadError.reservationMismatch
            }
        }
    }
}
