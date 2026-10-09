import MLX

extension PagedKVPool {
    public func usesQuantization(layerIndex: Int) -> Bool {
        layerKinds.indices.contains(layerIndex)
            && groupKey(forLayer: layerIndex).quantization != nil
    }

    /// Exact packed/native physical rows. Native recent generations and scratch
    /// are dynamically reserved before allocation and never credited as pages.
    /// Feasibility nevertheless includes one compact native-tail owner so a
    /// request whose pages consume its entire ceiling cannot retry forever.
    public func admissionStorageConfig(_ base: AdmissionV2.Config) throws -> AdmissionV2.Config {
        var result = base
        result.layerBytesPerToken = try layerKinds.enumerated().map { index, kind in
            kind.sharesKVWithLayer == nil ? try groupKey(forLayer: index).bytesPerToken() : 0
        }
        var tail = 0
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let key = groupKey(forLayer: index)
            guard let format = key.quantization else { continue }
            guard let nativeTokenBytes = key.geometry?.bytesPerToken(elementBytes: key.dtype.size)
            else {
                throw CBv2KVError.backendIneligible(reason: "invalid native recent geometry")
            }
            let logical = try PagedKVQuantizationConfig.multiply(
                nativeTokenBytes, format.recentTokenCount)
            // Separate K/V allocator upper bounds; a single combined bound can
            // underquote small allocations with two independent size classes.
            let k = try PagedKVQuantizationConfig.multiply(
                try PagedKVQuantizationConfig.multiply(key.kvHeads, format.recentTokenCount),
                try PagedKVQuantizationConfig.multiply(key.headDim, key.dtype.size))
            let v = try PagedKVQuantizationConfig.multiply(
                try PagedKVQuantizationConfig.multiply(key.kvHeads, format.recentTokenCount),
                try PagedKVQuantizationConfig.multiply(key.valueHeadDim, key.dtype.size))
            let arrays = try CBv2CheckpointAllocationFootprint.add(
                CBv2CheckpointAllocationFootprint.bound(k),
                CBv2CheckpointAllocationFootprint.bound(v))
            let bound = try CBv2CheckpointAllocationFootprint.add(
                arrays,
                PagedKVQuantizationConfig.multiply(CBv2CheckpointAllocationFootprint.bound(1), 2))
            tail = try CBv2CheckpointAllocationFootprint.add(tail, max(logical, bound))
        }
        guard result.pagedKVTransientFeasibilityBytes >= 0,
            result.minimumRequestTransientBytes >= result.pagedKVTransientFeasibilityBytes
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        let callerMinimum =
            result.minimumRequestTransientBytes - result.pagedKVTransientFeasibilityBytes
        result.minimumRequestTransientBytes = try CBv2CheckpointAllocationFootprint.add(
            callerMinimum, tail)
        result.pagedKVTransientFeasibilityBytes = tail
        return result
    }
}
