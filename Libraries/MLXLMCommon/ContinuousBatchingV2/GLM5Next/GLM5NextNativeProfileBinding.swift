// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Implemented by the SDK's concrete loaded GLM target/predictor owner, not by
/// a profile string or a provider capability flag.
package protocol GLM5NextNativeProfileSource: AnyObject {
    var nativeGLMTargetIdentity: ObjectIdentifier { get }
    var nativeGLMPredictorIdentity: ObjectIdentifier { get }
    var nativeGLMLayerKinds: [CBv2LayerKind] { get }
    var nativeGLMMaximumInputs: Int { get }
    func validateNativeGLMSource() throws
}

package protocol GLM5NextNativeAssistantValidating: AnyObject {
    func validateNativeGLMAssistant(source: any GLM5NextNativeProfileSource) throws
}

/// Separate GLM issuer using the existing native operation/failure machinery.
/// It issues target paging + one serial trained proposal, never media, rectangular
/// verification, or complete-prefix/SSD permission.
package final class GLM5NextNativeProfileBinding: CBv2NativePagedModelValidating,
    CBv2NativeCompletePrefixBindingValidating {
    let model: any CBv2MTPSteppableModel
    let source: any GLM5NextNativeProfileSource
    let assistant: any CBv2MTPRequestStatefulDrafter
    let maximumRequests: Int
    let completePrefixCache: (any CBv2NativeCompletePrefixCache)?
    private let prefixIdentity: CBv2CompleteCheckpointIdentity?
    private let assistantCodecID: String?
    private struct Row {
        weak var value: PagedSequenceKV?
        let layer: Int
        let cohort: UUID
    }
    private var rows: [UInt64: Row] = [:]
    private var bindingID: UUID?

    package init(
        model: any CBv2MTPSteppableModel, source: any GLM5NextNativeProfileSource,
        assistant: any CBv2MTPRequestStatefulDrafter, maximumRequests: Int,
        completePrefixCache: (any CBv2NativeCompletePrefixCache)? = nil
    ) throws {
        self.model = model
        self.source = source
        self.assistant = assistant
        self.maximumRequests = maximumRequests
        self.completePrefixCache = completePrefixCache
        prefixIdentity = completePrefixCache?.identity
        assistantCodecID = (assistant as? any CBv2MTPPrefixCheckpointCoding)?.prefixCheckpointCodecID
        guard (1 ... 1024).contains(maximumRequests) else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        try validateNativePagedModel()
        if completePrefixCache != nil { try validateNativeCompletePrefixBinding() }
    }

    package func validateNativeCompletePrefixBinding() throws {
        try validateNativePagedModel()
        guard let completePrefixCache, completePrefixCache.identity == prefixIdentity,
            completePrefixCache.identity.isValid,
            let typed = model as? any CBv2CompleteCheckpointKVTypeProviding,
            typed.cbv2CompleteCheckpointKVDTypes == Array(repeating: .float32, count: source.nativeGLMLayerKinds.count),
            let recurrent = model as? any CBv2RecurrentSteppableModel,
            recurrent.recurrentStateSpec?.layers.isEmpty == false,
            let coding = assistant as? any CBv2MTPPrefixCheckpointCoding,
            coding.prefixCheckpointCodecID == assistantCodecID,
            assistantCodecID == "glm5next-mtp-contiguous-prefix-v1"
        else { throw CBv2NativeShutdownError.unsupportedConsumer }
    }

    package func validateNativePagedModel() throws {
        try source.validateNativeGLMSource()
        let kinds = source.nativeGLMLayerKinds
        guard model.mtpTargetIdentity == source.nativeGLMTargetIdentity,
            model.supportsRequestStatefulMTP,
            !kinds.isEmpty, kinds.count <= 64,
            kinds.allSatisfy({
                $0.attention == .full && $0.kvHeads == 1
                    && $0.sharesKVWithLayer == nil && $0.modelLayerIndex != nil
                    && (1 ... 512).contains($0.headDim) && (1 ... 512).contains($0.valueHeadDim)
                    && $0.valueHeadDim > 1 && ($0.valueHeadDim - 1).isMultiple(of: 2)
            }),
            Set(kinds.compactMap(\.modelLayerIndex)).count == kinds.count,
            assistant.mtpTargetIdentity == source.nativeGLMTargetIdentity,
            assistant.requiredVerificationMode == .serialTarget,
            assistant.maximumDraftTokens == 1, assistant.maximumSpeculativeBatch == 1,
            assistant is any CBv2AdmittedMTPRequestStatefulDrafter,
            assistant is any CBv2NativeMTPCompletionSplitting,
            let checked = assistant as? any GLM5NextNativeAssistantValidating
        else { throw CBv2NativeShutdownError.unsupportedConsumer }
        try checked.validateNativeGLMAssistant(source: source)
    }

    func register(_ values: [CBv2SequenceKV?], binding: UUID) throws {
        try validateNativePagedModel()
        let kinds = source.nativeGLMLayerKinds
        guard values.count == kinds.count, bindingID == nil || bindingID == binding,
            rows.count <= (maximumRequests - 1) * kinds.count
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        var pending: [UInt64: Row] = [:]
        let cohort = UUID()
        for (index, item) in values.enumerated() {
            guard let row = item as? PagedSequenceKV, !row.isReleased,
                row.groupKey.dtype == .float32, row.groupKey.headDim == kinds[index].headDim,
                row.groupKey.valueHeadDim == kinds[index].valueHeadDim,
                rows[row.serial] == nil, pending[row.serial] == nil
            else {
                throw CBv2NativeShutdownError.unsupportedConsumer
            }
            pending[row.serial] = Row(value: row, layer: index, cohort: cohort)
        }
        bindingID = binding
        rows.merge(pending) { _, new in new }
    }

    func request(_ value: any CBv2SequenceKV, layer: Int) throws -> UUID {
        guard let value = value as? PagedSequenceKV, let row = rows[value.serial],
            row.value === value, row.layer == layer, !value.isReleased
        else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        return row.cohort
    }

    func remove(_ values: [CBv2SequenceKV?], binding: UUID) throws {
        guard bindingID == binding, values.count == source.nativeGLMLayerKinds.count else {
            throw CBv2NativeShutdownError.unsupportedConsumer
        }
        var cohort: UUID?
        var serials: [UInt64] = []
        for (index, value) in values.enumerated() {
            guard let value = value as? PagedSequenceKV else {
                throw CBv2NativeShutdownError.unsupportedConsumer
            }
            let actual = try request(value, layer: index)
            guard cohort == nil || cohort == actual else {
                throw CBv2NativeShutdownError.unsupportedConsumer
            }
            cohort = actual
            serials.append(value.serial)
        }
        for serial in serials { rows.removeValue(forKey: serial) }
    }
}
