// Copyright © 2026 Eigen Labs. Authenticated input-codec load, not serving activation.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

public struct MiMoV26AudioSidecarLoadRequest: Equatable, Encodable, Sendable {
    public let sessionID: UUID
    public let canonicalRoot, mainConfigurationSHA256, configurationSHA256, headerSHA256: String
    public let payloadSHA256: String
    public let configurationObject, payloadObject: MiMoV26FilesystemObjectState
    public let fileBytes, headerBytes, inputStoredBytes, unusedStoredBytes, largestInputBytes: Int
    public let inputTensorCount, unusedTensorCount: Int
    public let requiredLoadBytes: UInt64
}

public protocol MiMoV26AudioSidecarLoadReservation: AnyObject {
    var request: MiMoV26AudioSidecarLoadRequest { get }
    var reservedLoadBytes: UInt64 { get }
    /// Host implementation checks its actual registered lifecycle/ledger claim.
    func validateActive() throws
}

public enum MiMoV26AudioSidecarLoadPhase: String, Codable, Sendable {
    case admitted, authenticating, authenticated, inputMaterialization, runtimeMaterialization,
        complete
}

public struct MiMoV26AudioSidecarLoadProgress: Encodable, Sendable {
    public let phase: MiMoV26AudioSidecarLoadPhase
    public let authenticatedFileBytes, inputTensorsCompleted, inputBytesCompleted,
        runtimeRootsCompleted: Int
}

public struct MiMoV26AudioSidecarLoadReceipt: Encodable, Sendable {
    public let request: MiMoV26AudioSidecarLoadRequest
    public let sourceIdentity: String
    public let codecGeneration: UUID
    public let authenticatedFileBytes, materializedInputTensors, materializedInputBytes,
        runtimeRoots: Int
    /// Serial eval completed, but the enclosing protected setup still owes its
    /// actual construction fence before it can publish any serving contract.
    public let requiresOuterConstructionCompletion = true
}

private final class MiMoV26AudioSidecarLifetime {
    let source: MiMoV26AudioSidecarSource
    let reservation: any MiMoV26AudioSidecarLoadReservation
    var valid = true
    init(source: MiMoV26AudioSidecarSource, reservation: any MiMoV26AudioSidecarLoadReservation) {
        self.source = source
        self.reservation = reservation
    }
    func validate() throws {
        guard valid else { throw MiMoV26AudioSidecarError.invalidatedOwner }
        try source.validate()
        try reservation.validateActive()
    }
}

/// Actual owned codec, never a Module or a second reflected target tree.
/// The codec retains its lifetime box; that box does not retain the codec.
public final class MiMoV26AudioSidecarLoaded {
    public let receipt: MiMoV26AudioSidecarLoadReceipt
    let codec: MiMoV26OwnedAudioCodec
    private let lifetime: MiMoV26AudioSidecarLifetime
    fileprivate init(
        weights: MiMoV26AudioInputWeights, source: MiMoV26AudioSidecarSource,
        reservation: any MiMoV26AudioSidecarLoadReservation,
        mainConfiguration: MiMoV26Configuration,
        receipt: MiMoV26AudioSidecarLoadReceipt
    ) throws {
        self.receipt = receipt
        let lifetime = MiMoV26AudioSidecarLifetime(source: source, reservation: reservation)
        let codec = try MiMoV26OwnedAudioCodec(
            input: MiMoV26AudioInput(weights: weights), retaining: lifetime,
            expectedSourceIdentity: receipt.sourceIdentity,
            expectedGeneration: receipt.codecGeneration,
            mainConfiguration: mainConfiguration)
        self.lifetime = lifetime
        self.codec = codec
    }
    public func validate() throws {
        try lifetime.validate()
        try codec.validate()
    }
    package func invalidate() { lifetime.valid = false }
}

/// One-shot, non-Sendable source session. Async host transfer must use the same
/// consuming SendableBox pattern as the existing MiMo root serial session.
public final class MiMoV26AudioSidecarLoadSession {
    public let request: MiMoV26AudioSidecarLoadRequest
    let mainConfiguration: MiMoV26Configuration
    private let source: MiMoV26AudioSidecarSource
    private let lock = NSLock()
    private var consumed = false

    public init(
        root: URL, mainConfiguration: MiMoV26Configuration,
        mainConfigurationSHA256: String, isCancelled: () -> Bool = { false }
    ) throws {
        guard mainConfigurationSHA256.utf8.count == 64,
            mainConfigurationSHA256.utf8.allSatisfy({
                (48 ... 57).contains($0) || (97 ... 102).contains($0)
            })
        else {
            throw MiMoV26AudioSidecarError.invalidBinding
        }
        let source = try MiMoV26AudioSidecarSource(
            root: root, mainConfiguration: mainConfiguration,
            isCancelled: isCancelled)
        let selected = source.plan.requiredInputNames.compactMap { source.plan.descriptors[$0] }
        guard selected.count == 389, let largest = selected.map(\.byteCount).max() else {
            throw MiMoV26AudioSidecarError.invalidHeader
        }
        // Three whole selected-subset copy families conservatively cover the
        // retained read buffers, source arrays and transformed runtime roots.
        // Also price one whole source file for IO/page-cache coexistence, the
        // explicit BF16 table intermediate, largest read, per-object16-KiB slack
        // and64MiB metadata. Unused weights still are never native tensors.
        let input = UInt64(source.plan.inputStoredBytes)
        let codebookBytes = source.plan.requiredInputNames.filter {
            $0.hasPrefix("encoder.quantizer.")
        }
        .reduce(UInt64(0)) { $0 + UInt64(source.plan.descriptors[$1]!.byteCount) }
        var total = try mimoAudioAdd(input, input)
        for amount in [
            input, UInt64(source.payload.state.bytes), codebookBytes / 2,
            UInt64(largest), UInt64(3 * 389 * 16_384), UInt64(64 << 20),
        ] {
            total = try mimoAudioAdd(total, amount)
        }
        self.source = source
        self.mainConfiguration = mainConfiguration
        request = .init(
            sessionID: UUID(), canonicalRoot: source.root.path,
            mainConfigurationSHA256: mainConfigurationSHA256,
            configurationSHA256: MiMoV26AudioSidecarSource.configurationSHA256,
            headerSHA256: source.headerSHA256,
            payloadSHA256: MiMoV26AudioTokenizerWeights.selectedPayloadSHA256,
            configurationObject: source.configurationFile.state,
            payloadObject: source.payload.state,
            fileBytes: source.payload.state.bytes, headerBytes: source.headerBytes + 8,
            inputStoredBytes: source.plan.inputStoredBytes,
            unusedStoredBytes: source.plan.retainedUnusedStoredBytes,
            largestInputBytes: largest, inputTensorCount: selected.count,
            unusedTensorCount: source.plan.descriptors.count - selected.count,
            requiredLoadBytes: total)
    }

    public func validateSource() throws { try source.validate() }

    public func load(
        reservation: any MiMoV26AudioSidecarLoadReservation,
        retaining work: NativeConstructionScope,
        isCancelled: () -> Bool = { false },
        progress: (MiMoV26AudioSidecarLoadProgress) throws -> Void = { _ in }
    ) throws
        -> MiMoV26AudioSidecarLoaded
    {
        try work.withPhase(.serialLoad) {
            try work.retainOwner(self)
            try work.retainOwner(reservation)
            try lock.withLock {
                guard !consumed else { throw MiMoV26AudioSidecarError.alreadyConsumed }
                consumed = true
            }
            var authenticated = 0
            var count = 0
            var bytes = 0
            var runtimeCount = 0
            func validate() throws {
                guard !isCancelled(), !Task.isCancelled else {
                    throw MiMoV26AudioSidecarError.cancelled
                }
                guard reservation.request == request else {
                    throw MiMoV26AudioSidecarError.invalidBinding
                }
                guard reservation.reservedLoadBytes >= request.requiredLoadBytes else {
                    throw MiMoV26AudioSidecarError.insufficientReservation
                }
                try source.validate()
                try reservation.validateActive()
            }
            func checkpoint(_ phase: MiMoV26AudioSidecarLoadPhase) throws {
                try validate()
                try progress(
                    .init(
                        phase: phase, authenticatedFileBytes: authenticated,
                        inputTensorsCompleted: count, inputBytesCompleted: bytes,
                        runtimeRootsCompleted: runtimeCount))
                try validate()
            }
            try checkpoint(.admitted)
            let digest = try source.payload.authenticate(
                expected: request.payloadSHA256,
                isCancelled: { isCancelled() || Task.isCancelled }
            ) { value in
                authenticated = value
                try checkpoint(.authenticating)
            }
            try checkpoint(.authenticated)
            // Full-byte authentication is complete before constructing any
            // tensor. Decoder/vocoder/training payloads are never instantiated.
            var inputs: [String: MLXArray] = [:]
            let ordered = source.plan.requiredInputNames.sorted {
                source.header.tensors[$0]!.data_offsets[0]
                    < source.header.tensors[$1]!.data_offsets[0]
            }
            for name in ordered {
                try checkpoint(.inputMaterialization)
                let tensor = source.header.tensors[name]!
                let size = tensor.data_offsets[1] - tensor.data_offsets[0]
                let data = try source.payload.read(
                    offset: source.payloadOffset + tensor.data_offsets[0],
                    count: size, maximum: request.largestInputBytes,
                    isCancelled: { isCancelled() || Task.isCancelled })
                try validate()
                // Retain raw Data until the enclosing actual construction fence
                // as well as the native root. The three-copy charge includes it.
                try work.retainValue(data)
                try work.capture(Stream.cpu)
                try work.capture(StreamOrDevice.default.stream)
                try work.willSubmit()
                let array = try withError { error in
                    // Explicit raw-byte initializer preserves BF16 storage bits;
                    // a numeric UInt16 conversion would corrupt the checkpoint.
                    let result = MLXArray(data, tensor.shape, dtype: tensor.dtype.dtype)
                    try work.retain(result)
                    try error.check()
                    eval(result)
                    try error.check()
                    return result
                }
                guard let info = try array.evaluatedBufferInfo(), info.allocatedBytes >= size,
                    info.isRowContiguous, array.nbytes == size
                else {
                    throw MiMoV26AudioSidecarError.unmaterializedTensor
                }
                inputs[name] = array
                count += 1
                bytes += size
                try work.checkpoint("audioSidecar.sourceMaterialized")
            }
            guard count == request.inputTensorCount, bytes == request.inputStoredBytes else {
                throw MiMoV26AudioSidecarError.invalidHeader
            }
            try validate()
            let weights = try withError { error in
                let result = try MiMoV26AudioTokenizerWeights.load(
                    plan: source.plan, inputWeights: inputs)
                try work.retainValue(result)
                try error.check()
                return result
            }
            let roots = weights.materializationRoots
            guard roots.count == request.inputTensorCount else {
                throw MiMoV26AudioSidecarError.invalidBinding
            }
            for root in roots {
                try checkpoint(.runtimeMaterialization)
                try work.retain(root)
                try work.capture(StreamOrDevice.default.stream)
                try work.willSubmit()
                try withError { error in
                    eval(root)
                    try error.check()
                }
                guard let info = try root.evaluatedBufferInfo(), info.allocatedBytes >= root.nbytes
                else {
                    throw MiMoV26AudioSidecarError.unmaterializedTensor
                }
                runtimeCount += 1
                try work.checkpoint("audioSidecar.runtimeMaterialized")
            }
            try checkpoint(.complete)
            let receipt = MiMoV26AudioSidecarLoadReceipt(
                request: request, sourceIdentity: digest,
                codecGeneration: weights.generation, authenticatedFileBytes: authenticated,
                materializedInputTensors: count, materializedInputBytes: bytes,
                runtimeRoots: runtimeCount)
            let loaded = try MiMoV26AudioSidecarLoaded(
                weights: weights, source: source,
                reservation: reservation, mainConfiguration: mainConfiguration, receipt: receipt)
            try work.retainOwner(loaded)
            try work.invalidateOnFailedCompletion(loaded) { loaded.invalidate() }
            try loaded.validate()
            return loaded
        }
    }
}
