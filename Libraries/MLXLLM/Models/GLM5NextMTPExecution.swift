// Copyright © 2026 Eigen Labs.

import Cmlx
import Foundation
import MLX
import MLXLMCommon

/// A deliberately bounded contiguous assistant. The target remains independently
/// paged. This is never represented as a paged assistant or a qualified default.
public final class GLM5NextMTPExecution: GLM5NextMTPAdmittedExecutionFactory {
    public enum Storage: Sendable, Equatable { case contiguous }
    public static let layoutID = "glm5next-mtp-contiguous-v1"
    static let prefixCodecID = "glm5next-mtp-contiguous-prefix-v1"
    public let storage: Storage
    public let target: GLM5NextTextModel
    public let predictor: GLM5NextMTP
    public let maximumInputs: Int
    private weak var drafter: AnyObject?
    private var nativeProfileIssued = false
    private var nativeProfileValid = false
    private let prefixSchema: PrefixSchema?

    public struct NativeResources {
        public let model: any CBv2RecurrentMTPSteppableModel
        public let backend: PagedKVBackend
        public let cacheProvider: CBv2LayerCacheBank
        public let contract: CBv2NativeExecutionContract
        public let assistant: GLM5NextMTPAdapter
        public let processMemoryOwner: any CBv2ProcessMemoryOwner
        public let completePrefixCache: (any CBv2NativeCompletePrefixCache)?
    }

    /// Explicit protected construction, never a provider/model-registry default.
    /// The loaded tuple is immutable for this resource lifetime; replacing any
    /// component requires a fresh factory, construction scope and engine.
    public func makeNativeResources(
        assistant: GLM5NextMTPAdapter, config: PagedKVPoolConfig,
        processMemoryOwner: any CBv2ProcessMemoryOwner,
        construction: NativeConstructionScope,
        completePrefixCache: (any CBv2NativeCompletePrefixCache)? = nil
    ) throws -> NativeResources {
        try construction.withPhase(.nativeSetup) {
            guard !nativeProfileIssued, drafter === assistant,
                StreamOrDevice.default.stream == Stream.gpu,
                config.layerDTypes
                    == Array(repeating: .float32, count: target.cbv2LayerKinds.count),
                config.gatheredAttention?.admissionMode == .stepOwned(.pinnedMetal),
                config.gatheredAttention?.maximumContextTokens == maximumInputs,
                config.segmentSizeBytes != nil, config.quantization == nil,
                config.prefixSharingBlockSize == nil
            else { throw GLM5NextMTPError.invalid("unqualified native GLM resource tuple") }
            // All metadata refusals precede the first parameter evaluation.
            try predictor.validateExecutionTarget()
            nativeProfileIssued = true
            try construction.retainOwner(self)
            try construction.retainOwner(assistant)
            try construction.retainOwner(processMemoryOwner)
            try construction.invalidateOnFailedCompletion(self) { [weak self] in
                self?.nativeProfileValid = false
            }
            try construction.capture(StreamOrDevice.default.stream)
            try construction.capture(Stream.cpu)
            var seen = Set<ObjectIdentifier>()
            let weights = (target.parameters().flattened() + predictor.parameters().flattened())
                .map(\.1).filter { seen.insert(ObjectIdentifier($0)).inserted }
            guard weights.allSatisfy({ $0.dtype.size <= 4 }) else {
                throw GLM5NextMTPError.invalid("native GLM weight dtype exceeds admitted precision")
            }
            try construction.retain(arrays: weights)
            try construction.willSubmit()
            try withError { error in
                eval(weights)
                try error.check()
            }
            try construction.fence(cause: "GLM loaded target and trained MTP parameters")
            nativeProfileValid = true
            try construction.authorizeImmutableLoadedOwner(self)
            let model = NativeModel(target: target)
            try construction.retainOwner(model)
            let profile = try GLM5NextNativeProfileBinding(
                model: model, source: self, assistant: assistant,
                maximumRequests: config.gatheredAttention!.maximumBatchSize,
                completePrefixCache: completePrefixCache)
            try construction.retainOwner(profile)
            let binding = try CBv2NativePagedModelBinding(
                glm5Next: profile, processMemoryOwner: processMemoryOwner,
                construction: construction)
            try construction.retainOwner(binding)
            let backend = try PagedKVBackend(
                glm5NextTarget: target, config: config, nativeModelBinding: binding)
            try construction.retainOwner(backend)
            let caches = backend.makeLayerCaches()
            guard let paged = caches as? [PagedLayerCache] else {
                throw GLM5NextMTPError.invalid("native GLM cache bank did not contain paged rows")
            }
            let bank = CBv2LayerCacheBank(caches: caches)
            try construction.retainOwner(bank)
            try binding.seal(bank: bank, caches: paged)
            let contract = try CBv2NativeExecutionContract(
                model: model, backend: backend, cacheProvider: bank, assistant: assistant,
                construction: construction, loadedOwner: self,
                completePrefixCache: completePrefixCache,
                completePrefixValidator: completePrefixCache == nil ? nil : profile,
                prefixProcessMemoryOwner: completePrefixCache == nil ? nil : processMemoryOwner,
                nativePagedBinding: binding, nativePagedProcessMemoryOwner: processMemoryOwner)
            return NativeResources(
                model: model, backend: backend, cacheProvider: bank, contract: contract,
                assistant: assistant, processMemoryOwner: processMemoryOwner,
                completePrefixCache: completePrefixCache)
        }
    }

    /// The MTP capability exists only on this exact protected resource adapter.
    /// Ordinary target construction remains unchanged and does not enable MTP.
    private final class NativeModel: CBv2RecurrentMTPSteppableModel,
        CBv2TargetAuxiliaryAllocationProviding, CBv2CompleteCheckpointKVTypeProviding
    {
        let target: GLM5NextTextModel
        init(target: GLM5NextTextModel) { self.target = target }
        var cbv2Capabilities: CBv2ModelCapabilities {
            var value = target.cbv2Capabilities
            value.supportsMTP = true
            return value
        }
        var cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
            target.cbv2TargetAuxiliaryAllocationSpecs
        }
        var recurrentStateSpec: CBv2RecurrentStateSpec? { target.cbv2RecurrentStateSpec }
        var cbv2CompleteCheckpointKVDTypes: [DType]? { target.cbv2CompleteCheckpointKVDTypes }
        var mtpCaptureLayers: CBv2MTPCaptureLayers? { nil }
        var supportsRequestStatefulMTP: Bool { true }
        var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }
        func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache]) -> MLXArray {
            preconditionFailure("native GLM requires request-owned KDA state")
        }
        func forwardWithHidden(tokens: MLXArray, caches: [any CBv2AttendingLayerCache])
            -> (logits: MLXArray, lastHidden: MLXArray)
        {
            preconditionFailure("native GLM MTP requires request-owned KDA state")
        }
        func forward(
            tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation]
        ) -> MLXArray {
            forwardWithHidden(
                tokens: tokens, caches: caches, recurrentState: recurrentState, positionIds: nil
            ).logits
        }
        func forwardWithHidden(
            tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
        )
            -> (logits: MLXArray, lastHidden: MLXArray)
        {
            call(
                tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds,
                requirement: nil)
        }
        func forwardWithHiddenForPrefill(
            tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
            requirement: CBv2PrefillRequirement
        )
            -> (logits: MLXArray, lastHidden: MLXArray)
        {
            call(
                tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds,
                requirement: requirement)
        }
        private func call(
            _ tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
            requirement: CBv2PrefillRequirement?
        ) -> (logits: MLXArray, lastHidden: MLXArray) {
            let kv = caches.map { cache -> any KVCache in
                guard let value = cache as? any KVCache else {
                    preconditionFailure("foreign native GLM cache")
                }
                return value
            }
            if let requirement {
                return target.cbv2ForwardWithHiddenForPrefill(
                    tokens, caches: kv,
                    recurrentState: recurrentState, positionIds: positionIds,
                    requirement: requirement)
            }
            return target.cbv2ForwardWithHidden(
                tokens, caches: kv,
                recurrentState: recurrentState, positionIds: positionIds)
        }
    }

    public struct MemoryPlan {
        public let maximumInputs: Int
        public let maximumObservationTokens: Int
        public let maximumDraftQueryTokens: Int
        public let cacheBuffers: [CBv2MTPFixedBufferSpec]
        public let historyWorkBuffers: [CBv2MTPFixedBufferSpec]
        public let draftWorkBuffers: [CBv2MTPFixedBufferSpec]
        public let specification: CBv2MTPBoundedAllocationSpec
    }

    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }
    public var predictorIdentity: ObjectIdentifier { ObjectIdentifier(predictor) }

    /// maximumInputs is an enforced execution envelope, not a claimed model
    /// context. The declaration prices that ENTIRE bound even on the first token.
    public init(
        target: GLM5NextTextModel, predictor: GLM5NextMTP, maximumInputs: Int,
        storage: Storage
    ) throws {
        guard predictor.targetIdentity == ObjectIdentifier(target), maximumInputs > 0,
            maximumInputs <= Int(Int32.max),
            maximumInputs <= target.configuration.maxPositionEmbeddings
        else {
            throw GLM5NextMTPError.invalid("MTP execution identity or context envelope")
        }
        self.target = target
        self.predictor = predictor
        self.maximumInputs = maximumInputs
        self.storage = storage
        prefixSchema = try? PrefixSchema(target: target, predictor: predictor)
    }

    /// Loaded parameter metadata only. Each emitted tensor is also checked
    /// against the actual fenced bank; no projection or numeric cast runs here.
    private struct PrefixSchema {
        let hiddenWidth, latentWidth, indexWidth: Int
        let hiddenType, latentType, indexType, gateType: DType

        init(target: GLM5NextTextModel, predictor: GLM5NextMTP) throws {
            let parameters = Dictionary(uniqueKeysWithValues: predictor.parameters().flattened())
            func floating(_ name: String) throws -> DType {
                guard let value = parameters[name],
                    [.float16, .bfloat16, .float32].contains(value.dtype)
                else { throw GLM5NextMTPError.invalid("unresolved prefix dtype: \(name)") }
                return value.dtype
            }
            func promote(_ a: DType, _ b: DType) -> DType { a == b ? a : .float32 }
            func linear(_ name: String, input: DType) throws -> DType {
                guard let weight = parameters[name + ".weight"] else {
                    throw GLM5NextMTPError.invalid("missing prefix projection: \(name)")
                }
                let weightType =
                    try weight.dtype == .uint32
                    ? floating(name + ".scales") : floating(name + ".weight")
                var output = promote(input, weightType)
                if parameters[name + ".biases"] != nil {
                    output = promote(output, try floating(name + ".biases"))
                }
                if parameters[name + ".bias"] != nil {
                    output = promote(output, try floating(name + ".bias"))
                }
                return output
            }
            hiddenWidth = predictor.configuration.hiddenSize
            latentWidth = predictor.configuration.kvLoraRank
            indexWidth = predictor.configuration.indexHeadDim
            hiddenType = target.cbv2NativeActivationDType
            guard [.float16, .bfloat16, .float32].contains(hiddenType),
                target.model.norm.weight.dtype == hiddenType
            else { throw GLM5NextMTPError.invalid("unsupported normalized prefix hidden dtype") }
            let projected = try linear("eh_proj", input: hiddenType)
            latentType = promote(
                try linear("self_attn.kv_a_proj_with_mqa", input: projected),
                try floating("self_attn.kv_a_layernorm.weight"))
            indexType = try linear("self_attn.indexer.wk", input: projected)
            gateType = promote(
                projected, try floating("self_attn.indexer.index_kpool_compress_gate"))
        }

        func descriptors(_ count: Int) throws -> [CBv2CheckpointTensorDescriptor] {
            guard count > 1, count <= Int(Int32.max),
                let latent = CBv2CheckpointDType(latentType),
                let index = CBv2CheckpointDType(indexType),
                let gate = CBv2CheckpointDType(gateType),
                let hidden = CBv2CheckpointDType(hiddenType)
            else { throw GLM5NextMTPError.incompatibleState }
            return try [
                .init(
                    role: .assistantKeys, layer: 0, shape: [1, count - 1, latentWidth],
                    dtype: latent
                ),
                .init(
                    role: .assistantValues, layer: 0, shape: [1, count - 1, indexWidth],
                    dtype: index
                ),
                .init(
                    role: .assistantValues, layer: 1, shape: [1, count - 1, indexWidth], dtype: gate
                ),
                .init(
                    role: .assistantCacheMetadata, layer: 0, shape: [1, count - 1], dtype: .uint8),
                .init(role: .assistantHidden, layer: 0, shape: [1, 1, hiddenWidth], dtype: hidden),
                .init(role: .assistantFrontier, layer: 0, shape: [2], dtype: .int32),
            ]
        }

        func validate(_ tensors: [MLXArray], count: Int) throws -> Int {
            let expected = try descriptors(count)
            guard tensors.count == expected.count else { throw GLM5NextMTPError.incompatibleState }
            var bytes = 0
            for (tensor, descriptor) in zip(tensors, expected) {
                guard tensor.shape == descriptor.shape, tensor.dtype == descriptor.dtype.mlxDType,
                    let info = try tensor.evaluatedBufferInfo(), info.isRowContiguous,
                    info.dataOffset == 0, info.dataElements == tensor.size
                else { throw GLM5NextMTPError.incompatibleState }
                let next = bytes.addingReportingOverflow(info.allocatedBytes)
                guard !next.overflow else { throw CBv2MTPBindingError.capacityExceeded }
                bytes = next.partialValue
            }
            guard let validity = mlx_array_data_uint8(tensors[3].ctx),
                let frontiers = mlx_array_data_int32(tensors[5].ctx)
            else { throw GLM5NextMTPError.incompatibleState }
            try withExtendedLifetime(tensors) {
                guard frontiers[0] == Int32(count), frontiers[1] == Int32(count - 1) else {
                    throw GLM5NextMTPError.invalid("prefix target/assistant frontier mismatch")
                }
                for position in 0 ..< count - 1 where validity[position] > 1 {
                    throw GLM5NextMTPError.invalid("prefix validity is not exact 0/1")
                }
            }
            return bytes
        }
    }

    /// Immutable stage payload only. No request cache or new allocation owner
    /// exists until restoreDecodedPrefix receives the real admitted owner.
    final class DecodedPrefix {
        fileprivate let factory: GLM5NextMTPExecution
        let tensors: [MLXArray]
        let targetInputCount: Int
        let materializedBytes: Int
        fileprivate init(
            factory: GLM5NextMTPExecution, tensors: [MLXArray], count: Int, bytes: Int
        ) {
            self.factory = factory
            self.tensors = tensors
            targetInputCount = count
            materializedBytes = bytes
        }
    }

    func prefixDescriptors(targetInputCount: Int) -> [CBv2CheckpointTensorDescriptor]? {
        guard targetInputCount <= maximumInputs else { return nil }
        guard (try? predictor.validateExecutionTarget()) != nil else { return nil }
        return try? prefixSchema?.descriptors(targetInputCount)
    }

    func encodePrefix(
        checkpoint: any GLM5NextMTPExecutionCheckpoint,
        hidden: any GLM5NextMTPArrayLease, frontiers: any GLM5NextMTPArrayLease,
        targetInputCount: Int
    ) throws -> [MLXArray] {
        guard let checkpoint = checkpoint as? Checkpoint,
            checkpoint.target === target, checkpoint.predictor === predictor,
            checkpoint.inputCount == targetInputCount - 1,
            let bank = checkpoint.bank, bank.roots.arrays.count == 5,
            !bank.roots.hasUnmeasuredResidency,
            hidden.arrays.count == 1, frontiers.arrays.count == 1,
            !hidden.hasUnmeasuredResidency, !frontiers.hasUnmeasuredResidency,
            let prefixSchema, targetInputCount <= maximumInputs
        else { throw GLM5NextMTPError.incompatibleState }
        try predictor.validateExecutionTarget()
        let arrays =
            Array(bank.roots.arrays.prefix(3)) + [bank.roots.arrays[4]]
            + hidden.arrays + frontiers.arrays
        _ = try prefixSchema.validate(arrays, count: targetInputCount)
        return arrays
    }

    func decodePrefix(tensors: [MLXArray], targetInputCount: Int) throws -> DecodedPrefix {
        guard targetInputCount <= maximumInputs, let prefixSchema else {
            throw GLM5NextMTPError.incompatibleState
        }
        try predictor.validateExecutionTarget()
        let bytes = try prefixSchema.validate(tensors, count: targetInputCount)
        return DecodedPrefix(factory: self, tensors: tensors, count: targetInputCount, bytes: bytes)
    }

    func holdPrefixFrontiers(owner: any GLM5NextMTPExecutionOwner, targetInputCount: Int) throws
        -> any GLM5NextMTPArrayLease
    {
        guard let owner = owner as? Owner, owner.factory === self else {
            throw GLM5NextMTPError.incompatibleState
        }
        return try owner.holdPrefixFrontiers(targetInputCount)
    }

    func restoreDecodedPrefix(_ prefix: DecodedPrefix, owner: any GLM5NextMTPExecutionOwner) throws
        -> (hidden: any GLM5NextMTPArrayLease, frontiers: any GLM5NextMTPArrayLease)
    {
        guard let owner = owner as? Owner, owner.factory === self, prefix.factory === self else {
            throw GLM5NextMTPError.incompatibleState
        }
        return try owner.restoreDecodedPrefix(prefix)
    }

    /// Pure metadata reproduction. N prices retained history; Q is the actual
    /// Common prefill bound. Drafting is deliberately one query, never N queries.
    public static func memoryPlan(
        configuration c: GLM5NextTextConfiguration, maximumInputs: Int,
        limits: CBv2MTPAllocationLimits
    ) throws -> MemoryPlan {
        guard maximumInputs > 0, maximumInputs <= Int(Int32.max),
            maximumInputs <= c.maxPositionEmbeddings, limits.maximumPrefillTokens > 0,
            limits.maximumDraftTokens == 1, c.numNextNPredictLayers == 1,
            c.indexTopK > 0, c.indexTopK <= 2048
        else { throw GLM5NextMTPError.invalid("MTP allocation/priming envelope") }
        let n = maximumInputs
        let q = min(n, limits.maximumPrefillTokens)
        func bytes(_ factors: [Int], element: Int = 4) throws -> Int {
            try factors.reduce(element) { value, factor in
                let product = value.multipliedReportingOverflow(by: factor)
                guard factor > 0, !product.overflow else {
                    throw CBv2MTPBindingError.capacityExceeded
                }
                return product.partialValue
            }
        }
        let cacheBuffers: [CBv2MTPFixedBufferSpec] = try [
            .init(logicalBytes: bytes([n, c.kvLoraRank])),
            .init(logicalBytes: bytes([n, c.indexHeadDim]), allocationCount: 2),
            // Independent bool execution and exact uint8 checkpoint transport.
            .init(logicalBytes: bytes([n], element: 1), allocationCount: 2),
        ]
        let rawHistory = cacheBuffers.map {
            CBv2MTPFixedBufferSpec(
                logicalBytes: $0.logicalBytes, allocationCount: 2 * $0.allocationCount)
        }
        // History-only priming has no attention scores, selected-KV expansion,
        // MoE or vocabulary projection. It still pays all retained-history copies.
        var historyWork: [CBv2MTPFixedBufferSpec] = try [
            .init(logicalBytes: bytes([q, c.hiddenSize]), allocationCount: 32),
            .init(logicalBytes: bytes([q, c.kvLoraRank]), allocationCount: 8),
            .init(logicalBytes: bytes([q, c.indexHeadDim]), allocationCount: 12),
            .init(logicalBytes: 256, allocationCount: bytes([q, 32], element: 1)),
        ]
        historyWork += rawHistory
        // Only the real proposal query reaches the full layer. In particular,
        // the index score is [1,N,indexHeads], not [N,N,indexHeads].
        let selected = c.indexTopK + 3
        var draftWork: [CBv2MTPFixedBufferSpec] = try [
            // Five norms need <=6 H-wide intermediates each; 64 also covers
            // embedding, concatenation, residual, readout and independent copies.
            .init(logicalBytes: bytes([c.hiddenSize]), allocationCount: 64),
            .init(logicalBytes: bytes([c.qLoraRank]), allocationCount: 8),
            .init(logicalBytes: bytes([c.attentionHeads, c.qkNopeHeadDim]), allocationCount: 4),
            .init(logicalBytes: bytes([c.kvLoraRank]), allocationCount: 12),
            .init(logicalBytes: bytes([n, c.indexHeadDim]), allocationCount: 20),
            .init(logicalBytes: bytes([n, c.indexNHeads]), allocationCount: 8),
            .init(logicalBytes: bytes([selected, c.kvLoraRank]), allocationCount: 4),
            .init(
                logicalBytes: bytes([selected, c.attentionHeads, c.qkNopeHeadDim + c.vHeadDim]),
                allocationCount: 8),
            .init(logicalBytes: bytes([selected, c.attentionHeads]), allocationCount: 12),
            .init(
                logicalBytes: bytes([c.numExpertsPerTok, c.moeIntermediateSize]),
                allocationCount: 8),
            .init(logicalBytes: bytes([c.numExpertsPerTok, c.hiddenSize]), allocationCount: 6),
            .init(
                logicalBytes: bytes([c.moeIntermediateSize, c.nSharedExperts]),
                allocationCount: 8),
            .init(logicalBytes: bytes([c.nRoutedExperts]), allocationCount: 12),
            .init(logicalBytes: bytes([c.nRoutedExperts, c.hiddenSize]), allocationCount: 2),
            .init(logicalBytes: bytes([c.vocabularySize]), allocationCount: 3),
            // Tiny index/reduction/shape/fence allocations can be per query.
            .init(logicalBytes: 256, allocationCount: 64),
        ]
        draftWork += rawHistory
        let fragments = try bytes([q, 4], element: 1) + 16
        var resident = cacheBuffers.map {
            CBv2MTPFixedBufferSpec(
                logicalBytes: $0.logicalBytes, allocationCount: $0.allocationCount * 6)
        }
        resident += try [
            .init(logicalBytes: bytes([c.hiddenSize]), allocationCount: fragments),
            .init(logicalBytes: 4, allocationCount: fragments),
            .init(logicalBytes: 1, allocationCount: 2 * fragments),
            .init(logicalBytes: 8, allocationCount: 4),
        ]
        let working =
            draftWork.map {
                CBv2MTPFixedBufferSpec(
                    logicalBytes: $0.logicalBytes, allocationCount: $0.allocationCount * 2)
            }
            + historyWork.map {
                CBv2MTPFixedBufferSpec(
                    logicalBytes: $0.logicalBytes, allocationCount: $0.allocationCount * 3)
            }
        let host = try bytes([q + 1, 64 << 10], element: 1) + bytes([n, 64], element: 1)
        return MemoryPlan(
            maximumInputs: n, maximumObservationTokens: q, maximumDraftQueryTokens: 1,
            cacheBuffers: cacheBuffers, historyWorkBuffers: historyWork,
            draftWorkBuffers: draftWork,
            specification: .init(resident: resident, working: working, hostBytes: host))
    }

    public func bind(drafter: AnyObject) throws {
        guard self.drafter == nil || self.drafter === drafter else {
            throw GLM5NextMTPError.incompatibleState
        }
        self.drafter = drafter
    }
    public func allocation(limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec? {
        try? Self.memoryPlan(
            configuration: predictor.configuration, maximumInputs: maximumInputs,
            limits: limits
        ).specification
    }
    public func makeOwner(maximumInputs: Int) throws -> any GLM5NextMTPExecutionOwner {
        throw CBv2MTPBindingError.missingAdmission
    }
    public func makeOwner(admission: CBv2AdmittedMTPContext) throws -> any GLM5NextMTPExecutionOwner
    {
        guard let drafter, admission.maximumInputs <= maximumInputs else {
            throw GLM5NextMTPError.invalid("MTP owner exceeds installed envelope")
        }
        try admission.validate(
            target: targetIdentity, drafter: ObjectIdentifier(drafter),
            maximumInputs: admission.maximumInputs)
        return try Owner(factory: self, admission: admission, drafter: ObjectIdentifier(drafter))
    }

    private final class Roots: GLM5NextMTPArrayLease {
        let arrays: [MLXArray]
        let lease: CBv2MTPAllocationLease
        let metadata: CBv2MTPAllocationLease
        private(set) var materializedBytes = 0
        private(set) var hasUnmeasuredResidency = true
        init(_ arrays: [MLXArray], lease: CBv2MTPAllocationLease, metadata: CBv2MTPAllocationLease)
        {
            self.arrays = arrays
            self.lease = lease
            self.metadata = metadata
        }
        func measure() throws {
            var bytes = 0
            var unique = true
            for array in arrays where array.size > 0 {
                guard let info = try array.evaluatedBufferInfo(), info.isRowContiguous,
                    info.dataOffset == 0, info.dataElements == array.size
                else {
                    throw GLM5NextMTPError.invalid("MTP root was not compact and evaluated")
                }
                let sum = bytes.addingReportingOverflow(info.allocatedBytes)
                guard !sum.overflow else { throw CBv2MTPBindingError.capacityExceeded }
                bytes = sum.partialValue
                unique = unique && info.isUnique
            }
            guard bytes <= lease.bytes else { throw CBv2MTPBindingError.capacityExceeded }
            materializedBytes = unique ? bytes : 0
            hasUnmeasuredResidency = !unique
        }
    }

    private final class Bank {
        let cache: GLM5NextDSACache
        let roots: Roots
        init(cache: GLM5NextDSACache, roots: Roots) {
            self.cache = cache
            self.roots = roots
        }
    }
    private final class History {
        let bank: Bank
        let source: Bank?
        let work: CBv2MTPAllocationLease
        init(bank: Bank, source: Bank?, work: CBv2MTPAllocationLease) {
            self.bank = bank
            self.source = source
            self.work = work
        }
    }
    private final class Restoration {
        let source: Roots
        let bank: Bank
        let work: CBv2MTPAllocationLease
        let generation: UInt64
        init(source: Roots, bank: Bank, work: CBv2MTPAllocationLease, generation: UInt64) {
            self.source = source
            self.bank = bank
            self.work = work
            self.generation = generation
        }
    }
    private final class Checkpoint: GLM5NextMTPExecutionCheckpoint {
        let target: GLM5NextTextModel
        let predictor: GLM5NextMTP
        let bank: Bank?
        let metadata: CBv2MTPAllocationLease
        let layout = GLM5NextMTPExecution.layoutID
        var inputCount: Int { bank?.cache.offset ?? 0 }
        var materializedBytes: Int { bank?.roots.materializedBytes ?? 0 }
        var evaluationTargets: [MLXArray] { bank?.roots.arrays ?? [] }
        init(_ owner: Owner, metadata: CBv2MTPAllocationLease) {
            target = owner.factory.target
            predictor = owner.factory.predictor
            bank = owner.current
            self.metadata = metadata
        }
    }

    private final class WeakRoots {
        weak var value: Roots?
        init(_ value: Roots) { self.value = value }
    }

    private final class Owner: GLM5NextMTPIncrementalExecutionOwner,
        GLM5NextMTPNativeCompletionOwner
    {
        let factory: GLM5NextMTPExecution
        let admission: CBv2AdmittedMTPContext
        let drafter: ObjectIdentifier
        let plan: MemoryPlan
        var host: CBv2MTPAllocationLease?
        var current: Bank?
        var active: Prepared?
        var history: [History] = []
        var historyGeneration: UInt64?
        var restoration: Restoration?
        var retired: [Prepared] = []
        var held: [WeakRoots] = []
        var pendingHeld: [Roots] = []
        var stream: StreamOrDevice?
        var generation: UInt64 = 0
        var closed = false
        var nativeFenceCompleted = false
        var targetIdentity: ObjectIdentifier { factory.targetIdentity }
        var predictorIdentity: ObjectIdentifier { factory.predictorIdentity }
        var maximumInputs: Int { admission.maximumInputs }
        var maximumObservationTokens: Int { plan.maximumObservationTokens }
        var committedInputCount: Int { current?.cache.offset ?? 0 }
        var materializedBytes: Int {
            reachableRoots.reduce(0) { sum, roots in
                let next = sum.addingReportingOverflow(roots.materializedBytes)
                return next.overflow ? Int.max : next.partialValue
            }
        }
        func accountsForResidency(of lease: any GLM5NextMTPArrayLease) -> Bool {
            held.contains { $0.value === lease }
        }
        var hasUnmeasuredResidency: Bool {
            stream != nil || active != nil || !history.isEmpty || restoration != nil
                || reachableRoots.contains { $0.hasUnmeasuredResidency }
        }
        private var reachableRoots: [Roots] {
            var roots = held.compactMap(\.value) + pendingHeld
            if let restoration { roots += [restoration.source, restoration.bank.roots] }
            if let current { roots.append(current.roots) }
            for stage in history {
                roots.append(stage.bank.roots)
                if let source = stage.source { roots.append(source.roots) }
            }
            for candidate in retired + [active].compactMap({ $0 }) {
                roots += [candidate.bank.roots, candidate.outputs]
                if let source = candidate.source { roots.append(source.roots) }
            }
            var seen = Set<ObjectIdentifier>()
            return roots.filter { seen.insert(ObjectIdentifier($0)).inserted }
        }
        var evaluationTargets: [MLXArray] { reachableRoots.flatMap(\.arrays) }

        init(
            factory: GLM5NextMTPExecution, admission: CBv2AdmittedMTPContext,
            drafter: ObjectIdentifier
        ) throws {
            self.factory = factory
            self.admission = admission
            self.drafter = drafter
            plan = try GLM5NextMTPExecution.memoryPlan(
                configuration: factory.predictor.configuration,
                maximumInputs: factory.maximumInputs, limits: admission.resolved.limits)
            host = try admission.reserve(.host, buffers: [.init(logicalBytes: 4096)])
        }
        func validate() throws {
            guard !closed else { throw GLM5NextMTPError.releasedState }
            try admission.validate(
                target: targetIdentity, drafter: drafter, maximumInputs: maximumInputs)
        }
        func willConstruct() throws {
            try validate()
            let next = StreamOrDevice.default
            guard stream == nil || stream == next else {
                throw GLM5NextMTPError.invalid("MTP construction stream changed")
            }
            stream = next
            nativeFenceCompleted = false
        }
        func hold(_ arrays: [MLXArray]) throws -> any GLM5NextMTPArrayLease {
            try willConstruct()
            guard (1 ... 2).contains(arrays.count) else {
                throw GLM5NextMTPError.invalid("MTP lease arity")
            }
            for array in arrays {
                guard (2 ... 3).contains(array.ndim), array.dim(0) == 1,
                    array.dim(1) <= maximumObservationTokens, array.dim(1) <= maximumInputs,
                    array.dtype.size <= 4,
                    array.ndim == 2 || array.dim(2) == factory.predictor.configuration.hiddenSize
                else {
                    throw GLM5NextMTPError.invalid("MTP lease geometry")
                }
            }
            // Known shape extents debit the already full-context-priced pool;
            // they do not replace the token promise or report measured residency.
            let copied = arrays.filter { $0.size > 0 }
            var specs = copied.map {
                CBv2MTPFixedBufferSpec(logicalBytes: $0.nbytes)
            }
            if !copied.isEmpty {
                specs.append(.init(logicalBytes: 1, allocationCount: copied.count))
            }
            let lease = try admission.reserve(.resident, buffers: specs)
            let metadata = try admission.reserve(.host, buffers: [.init(logicalBytes: 512)])
            let copy = Roots(arrays.map(Self.copy), lease: lease, metadata: metadata)
            held.append(WeakRoots(copy))
            pendingHeld.append(copy)
            return copy
        }
        func holdPrefixFrontiers(_ targetInputCount: Int) throws -> Roots {
            guard targetInputCount > 0, targetInputCount <= maximumInputs else {
                throw GLM5NextMTPError.incompatibleState
            }
            try willConstruct()
            let lease = try admission.reserve(.resident, buffers: [.init(logicalBytes: 8)])
            let metadata = try admission.reserve(.host, buffers: [.init(logicalBytes: 512)])
            let roots = Roots(
                [MLXArray([Int32(targetInputCount), Int32(targetInputCount - 1)])],
                lease: lease, metadata: metadata)
            held.append(WeakRoots(roots))
            pendingHeld.append(roots)
            return roots
        }

        func restoreDecodedPrefix(_ prefix: DecodedPrefix) throws
            -> (hidden: Roots, frontiers: Roots)
        {
            try validate()
            guard current == nil, active == nil, history.isEmpty, restoration == nil,
                stream == nil, prefix.factory === factory,
                prefix.targetInputCount <= maximumInputs, generation < UInt64.max,
                let schema = factory.prefixSchema
            else { throw GLM5NextMTPError.incompatibleState }
            let bytes = try schema.validate(prefix.tensors, count: prefix.targetInputCount)
            // The active request now carries BOTH the immutable input aliases
            // and its private copies. No transient/parallel Admission is used.
            let sourceLease = try admission.reserve(
                .resident, buffers: prefix.tensors.map { .init(logicalBytes: $0.nbytes) })
            guard bytes <= sourceLease.bytes else { throw CBv2MTPBindingError.capacityExceeded }
            let sourceMetadata = try admission.reserve(.host, buffers: [.init(logicalBytes: 2048)])
            let bankLease = try admission.reserve(.resident, buffers: plan.cacheBuffers)
            let bankMetadata = try admission.reserve(.host, buffers: [.init(logicalBytes: 2048)])
            let hiddenLease = try admission.reserve(
                .resident,
                buffers: [.init(logicalBytes: prefix.tensors[4].nbytes), .init(logicalBytes: 1)])
            let hiddenMetadata = try admission.reserve(.host, buffers: [.init(logicalBytes: 512)])
            let frontiersLease = try admission.reserve(
                .resident, buffers: [.init(logicalBytes: 8), .init(logicalBytes: 1)])
            let frontiersMetadata = try admission.reserve(
                .host, buffers: [.init(logicalBytes: 512)])
            let work = try admission.reserve(.working, buffers: plan.historyWorkBuffers)
            try willConstruct()
            let source = Roots(prefix.tensors, lease: sourceLease, metadata: sourceMetadata)
            let input = prefix.tensors
            let bankRoots = Roots(
                Array(input.prefix(3)).map(Self.copy)
                    + [Self.copy(input[3].asType(.bool)), Self.copy(input[3])],
                lease: bankLease, metadata: bankMetadata)
            let cache = GLM5NextDSACache()
            cache.state = Array(bankRoots.arrays.prefix(4))
            let hidden = Roots([Self.copy(input[4])], lease: hiddenLease, metadata: hiddenMetadata)
            let frontiers = Roots(
                [Self.copy(input[5])], lease: frontiersLease, metadata: frontiersMetadata)
            held += [WeakRoots(hidden), WeakRoots(frontiers)]
            pendingHeld += [hidden, frontiers]
            restoration = Restoration(
                source: source, bank: Bank(cache: cache, roots: bankRoots), work: work,
                generation: generation)
            return (hidden, frontiers)
        }
        static func copy(_ array: MLXArray) -> MLXArray {
            guard array.size > 0 else { return MLXArray.zeros(array.shape, dtype: array.dtype) }
            return MLX.where(MLXArray(true), array, array)
        }
        func stageHistory(_ chunks: [GLM5NextMTPHistoryChunk]) throws {
            try validate()
            guard active == nil, restoration == nil, history.count + chunks.count <= 3,
                historyGeneration == nil || historyGeneration == generation
            else {
                throw GLM5NextMTPError.invalid(
                    "MTP history overlaps a draft or exceeds the observation fence envelope")
            }
            var position = history.last?.bank.cache.offset ?? committedInputCount
            for chunk in chunks {
                guard chunk.shiftedTokens.ndim == 2, chunk.shiftedTokens.dim(0) == 1,
                    [.int32, .uint32].contains(chunk.shiftedTokens.dtype),
                    chunk.shiftedTokens.dim(1) > 0,
                    chunk.shiftedTokens.dim(1) <= maximumObservationTokens,
                    chunk.shiftedTokens.dim(1) <= maximumInputs - position,
                    chunk.targetHidden.shape == [
                        1, chunk.shiftedTokens.dim(1), factory.predictor.configuration.hiddenSize,
                    ],
                    chunk.targetHidden.dtype.isFloatingPoint
                else {
                    throw GLM5NextMTPError.invalid(
                        "MTP history chunk exceeds the admitted query/context bound")
                }
                position += chunk.shiftedTokens.dim(1)
            }
            guard !chunks.isEmpty else { return }
            try willConstruct()
            var pending = history
            for chunk in chunks {
                let work = try admission.reserve(.working, buffers: plan.historyWorkBuffers)
                let bankLease = try admission.reserve(.resident, buffers: plan.cacheBuffers)
                let metadata = try admission.reserve(.host, buffers: [.init(logicalBytes: 2048)])
                let source = pending.last?.bank ?? current
                let result = try factory.predictor.prepareHistory(
                    shiftedTokens: chunk.shiftedTokens, hidden: chunk.targetHidden,
                    cache: source?.cache ?? GLM5NextDSACache())
                let cache = GLM5NextDSACache()
                let copied = Roots(
                    (result.state + [result.state[3].asType(.uint8)]).map(Self.copy),
                    lease: bankLease, metadata: metadata)
                cache.state = Array(copied.arrays.prefix(4))
                pending.append(
                    History(bank: Bank(cache: cache, roots: copied), source: source, work: work))
            }
            history = pending
            historyGeneration = generation
        }
        func prepare(shiftedTokens: MLXArray, targetHidden: MLXArray) throws
            -> any GLM5NextMTPPreparedExecution
        {
            try willConstruct()
            let base = history.last?.bank ?? current
            let baseCount = base?.cache.offset ?? 0
            guard active == nil, restoration == nil,
                shiftedTokens.ndim == 2, shiftedTokens.dim(0) == 1,
                [.int32, .uint32].contains(shiftedTokens.dtype),
                shiftedTokens.dim(1) == plan.maximumDraftQueryTokens,
                shiftedTokens.dim(1) <= maximumInputs - baseCount,
                history.isEmpty || generation < UInt64.max
            else {
                throw GLM5NextMTPError.invalid(
                    "MTP requires one draft query after fenced bounded history")
            }
            let work = try admission.reserve(.working, buffers: plan.draftWorkBuffers)
            let bankLease = try admission.reserve(.resident, buffers: plan.cacheBuffers)
            let metadata = try admission.reserve(.host, buffers: [.init(logicalBytes: 2048)])
            let result = try factory.predictor.prepareExecution(
                shiftedTokens: shiftedTokens,
                hidden: targetHidden, cache: base?.cache ?? GLM5NextDSACache())
            let cache = GLM5NextDSACache()
            let cached = Roots(
                (result.cache.state + [result.cache.state[3].asType(.uint8)]).map(Self.copy),
                lease: bankLease, metadata: metadata)
            cache.state = Array(cached.arrays.prefix(4))
            let outputs = Roots(
                [Self.copy(result.output.hidden), Self.copy(result.output.logits)],
                lease: work, metadata: metadata)
            let prepared = Prepared(
                owner: self, bank: Bank(cache: cache, roots: cached), outputs: outputs,
                count: shiftedTokens.dim(1), generation: generation + (history.isEmpty ? 0 : 1),
                source: base)
            active = prepared
            return prepared
        }
        func didFinishEvaluation() throws {
            try fenceNativeCompletion()
            try commitNativeCompletion()
        }
        func fenceNativeCompletion() throws {
            guard !closed else { throw GLM5NextMTPError.releasedState }
            if let stream {
                try withError { error in
                    stream.stream.synchronize()
                    try error.check()
                }
            }
            nativeFenceCompleted = true
        }
        func commitNativeCompletion() throws {
            guard nativeFenceCompleted else {
                throw GLM5NextMTPError.invalid("native MTP fence did not complete")
            }
            guard admission.isActive else {
                // Cancellation retired this request while its graph was live.
                // Completion succeeds, but no candidate/history becomes canonical.
                stream = nil
                nativeFenceCompleted = false
                return
            }
            try validate()
            for value in held.compactMap(\.value) { try value.measure() }
            for value in history { try value.bank.roots.measure() }
            if let restoration {
                guard restoration.generation == generation, generation < UInt64.max else {
                    throw GLM5NextMTPError.incompatibleState
                }
                try restoration.bank.roots.measure()
                current = restoration.bank
                generation += 1
                self.restoration = nil
            }
            if let active {
                try active.bank.roots.measure()
                try active.outputs.measure()
            }
            held.removeAll { $0.value == nil }
            retired.removeAll()
            pendingHeld.removeAll()
            if let last = history.last {
                guard historyGeneration == generation, generation < UInt64.max else {
                    throw GLM5NextMTPError.incompatibleState
                }
                current = last.bank
                generation += 1
                history.removeAll()
                historyGeneration = nil
            }
            stream = nil
            nativeFenceCompleted = false
        }
        func checkpoint() throws -> any GLM5NextMTPExecutionCheckpoint {
            try validate()
            guard active == nil, history.isEmpty, restoration == nil, stream == nil,
                !hasUnmeasuredResidency
            else {
                throw GLM5NextMTPError.invalid("checkpoint before completed MTP fence")
            }
            return Checkpoint(
                self, metadata: try admission.reserve(.host, buffers: [.init(logicalBytes: 512)]))
        }
        func restore(_ checkpoint: any GLM5NextMTPExecutionCheckpoint) throws {
            try validate()
            guard active == nil, history.isEmpty, restoration == nil, stream == nil,
                let checkpoint = checkpoint as? Checkpoint,
                checkpoint.target === factory.target, checkpoint.predictor === factory.predictor,
                checkpoint.layout == GLM5NextMTPExecution.layoutID,
                checkpoint.inputCount <= maximumInputs
            else {
                throw GLM5NextMTPError.incompatibleState
            }
            guard generation < UInt64.max else { throw GLM5NextMTPError.incompatibleState }
            generation += 1
            current = checkpoint.bank
        }
        func close() throws {
            guard !closed else { return }
            if let stream {
                try withError { error in
                    eval(evaluationTargets)
                    stream.stream.synchronize()
                    try error.check()
                }
            }
            closed = true
            active = nil
            history.removeAll()
            historyGeneration = nil
            restoration = nil
            retired.removeAll()
            current = nil
            held.removeAll()
            pendingHeld.removeAll()
            host = nil
            stream = nil
        }
    }

    private final class Prepared: GLM5NextMTPPreparedExecution {
        weak var owner: Owner?
        let bank: Bank
        let outputs: Roots
        let inputCount: Int
        let generation: UInt64
        let source: Bank?
        var evaluated = false
        var finished = false
        var output: GLM5NextMTPOutput {
            .init(hidden: outputs.arrays[0], logits: outputs.arrays[1])
        }
        var evaluationTargets: [MLXArray] { bank.roots.arrays + outputs.arrays }
        init(
            owner: Owner, bank: Bank, outputs: Roots, count: Int, generation: UInt64, source: Bank?
        ) {
            self.owner = owner
            self.bank = bank
            self.outputs = outputs
            inputCount = count
            self.generation = generation
            self.source = source
        }
        func didFinishEvaluation() throws {
            guard let owner, owner.stream == nil else {
                throw GLM5NextMTPError.invalid("MTP owner was not fenced")
            }
            try owner.validate()
            guard owner.active === self, owner.generation == generation, !finished else {
                throw GLM5NextMTPError.incompatibleState
            }
            evaluated = true
        }
        func commit() throws {
            guard let owner else { throw GLM5NextMTPError.releasedState }
            try owner.validate()
            guard evaluated, !finished, owner.active === self, owner.generation == generation,
                owner.generation < UInt64.max
            else {
                throw GLM5NextMTPError.incompatibleState
            }
            owner.current = bank
            owner.active = nil
            owner.generation += 1
            owner.retired.append(self)
            finished = true
        }
        func discard() throws {
            guard let owner else { throw GLM5NextMTPError.releasedState }
            guard !finished, owner.active === self, owner.generation < UInt64.max else {
                throw GLM5NextMTPError.incompatibleState
            }
            owner.active = nil
            // Discarding a query must not invalidate trusted priming staged
            // before it. That history still publishes at its own checked fence.
            if owner.history.isEmpty { owner.generation += 1 }
            owner.retired.append(self)
            finished = true
        }
    }
}

extension GLM5NextMTPExecution: GLM5NextNativeProfileSource {
    package var nativeGLMTargetIdentity: ObjectIdentifier { targetIdentity }
    package var nativeGLMPredictorIdentity: ObjectIdentifier { predictorIdentity }
    package var nativeGLMLayerKinds: [CBv2LayerKind] { target.cbv2LayerKinds }
    package var nativeGLMMaximumInputs: Int { maximumInputs }
    package func validateNativeGLMSource() throws {
        guard nativeProfileIssued, nativeProfileValid,
            predictor.targetIdentity == ObjectIdentifier(target), storage == .contiguous,
            target.configuration.numNextNPredictLayers == 1,
            !target.cbv2LayerKinds.isEmpty
        else {
            throw GLM5NextMTPError.invalid("inactive or mismatched native GLM loaded tuple")
        }
        try predictor.validateExecutionTarget()
    }
}
