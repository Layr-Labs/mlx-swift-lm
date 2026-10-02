/// Conservative additional ownership for an historical assistant checkpoint.
/// This is an admission promise, never measured backing or refundable credit.
enum CBv2HistoricalMTPCheckpointFootprint {
    static func retainedHostBytes(position: Int) throws -> Int {
        guard position >= 4 else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let value = position.multipliedReportingOverflow(by: 8)
        guard !value.overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        return try CBv2CheckpointAllocationFootprint.add(value.partialValue, 64 << 10)
    }

    static func nativeDestinationBound(_ descriptors: [CBv2CheckpointTensorDescriptor]) throws
        -> Int
    {
        guard !descriptors.isEmpty, descriptors.count <= 4096 else {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        return try descriptors.reduce(0) { total, descriptor in
            try descriptor.validate()
            return try CBv2CheckpointAllocationFootprint.add(
                total,
                CBv2CheckpointAllocationFootprint.bound(descriptor.byteCount))
        }
    }

    /// Source views, compact destination and a temporary copy generation may
    /// coexist. The donor's request reservation separately retains its complete
    /// native backing until this copy's real completion; no max-layer discount.
    static func captureBytes(position: Int, descriptors: [CBv2CheckpointTensorDescriptor]) throws
        -> Int
    {
        let native = try nativeDestinationBound(descriptors)
        let generations = native.multipliedReportingOverflow(by: 3)
        guard !generations.overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        let scalar = try CBv2CheckpointAllocationFootprint.bound(1)
        let controls = scalar.multipliedReportingOverflow(by: descriptors.count * 2)
        guard !controls.overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        return try CBv2CheckpointAllocationFootprint.add(
            CBv2CheckpointAllocationFootprint.add(generations.partialValue, controls.partialValue),
            retainedHostBytes(position: position))
    }
}
