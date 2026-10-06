import MLX

/// Sequence-growing model state outside the backend's K/V pages and the
/// fixed recurrent rows. Nil refuses admission; an empty list owns no extras.
public protocol CBv2TargetAuxiliaryAllocationProviding: AnyObject {
    var cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? { get }
}

enum CBv2TargetAuxiliaryAdmission {
    static func apply(
        model: any CBv2SteppableModel, config: inout AdmissionV2.Config,
        policy: AllocationFootprintPolicy?, draftSpecs: [CBv2AuxiliaryAllocationSpec]?
    ) {
        guard let owner = model as? any CBv2TargetAuxiliaryAllocationProviding else { return }
        guard let target = owner.cbv2TargetAuxiliaryAllocationSpecs else {
            config.fixedBytesPerRequest = Int.max
            return
        }
        guard !target.isEmpty else { return }
        guard let policy = policy ?? Memory.allocationFootprintPolicy() else {
            config.fixedBytesPerRequest = Int.max
            return
        }
        var targetRate = 0
        for spec in target {
            guard spec.bytesPerToken > 0, spec.allocationCount > 0 else {
                config.fixedBytesPerRequest = Int.max
                return
            }
            let (bytes, overflow) = spec.bytesPerToken.multipliedReportingOverflow(by: spec.allocationCount)
            let (sum, sumOverflow) = targetRate.addingReportingOverflow(bytes)
            guard !overflow && !sumOverflow else {
                config.fixedBytesPerRequest = Int.max
                return
            }
            targetRate = sum
        }
        let draftRate = config.auxiliaryBytesPerToken
        guard draftRate >= 0 else {
            config.fixedBytesPerRequest = Int.max
            return
        }
        let drafts = draftRate == 0 ? [] : (draftSpecs ?? [CBv2AuxiliaryAllocationSpec(
            bytesPerToken: 1, allocationCount: draftRate,
            tokenGranularity: config.auxiliaryTokenGranularity,
            tokenPadding: config.auxiliaryTokenAllocationPadding, partitioned: true)])
        guard draftRate == 0 || !drafts.isEmpty else {
            config.fixedBytesPerRequest = Int.max
            return
        }
        let (total, overflow) = targetRate.addingReportingOverflow(draftRate)
        guard !overflow, let projection = CBv2AuxiliaryAllocationProjection(
            policy: policy, buffers: target + drafts) else {
            config.fixedBytesPerRequest = Int.max
            return
        }
        config.auxiliaryBytesPerToken = total
        // Each buffer's actual granularity/padding is already priced by the
        // combined projection. Do not round all buffers by one model's stride.
        config.auxiliaryTokenGranularity = 1
        config.auxiliaryTokenAllocationPadding = 0
        config.auxiliaryAllocationProjection = projection
    }
}
