import MLX

extension PagedKVPool {
    public func usesQuantization(layerIndex: Int) -> Bool {
        layerKinds.indices.contains(layerIndex)
            && groupKey(forLayer: layerIndex).quantization != nil
    }

    /// Exact packed/native physical rows. Native recent generations and scratch
    /// are dynamically reserved before allocation and never credited as pages.
    /// The numeric minimum retains a full-cap quotation; AdmissionV2 substitutes
    /// a request-sized native-owner peak. Attention workspaces remain separate.
    public func admissionStorageConfig(_ base: AdmissionV2.Config) throws -> AdmissionV2.Config {
        var result = base
        result.layerBytesPerToken = try layerKinds.enumerated().map { index, kind in
            kind.sharesKVWithLayer == nil ? try groupKey(forLayer: index).bytesPerToken() : 0
        }
        var owners: [PagedKVNativeRecentFeasibility.Owner] = []
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let key = groupKey(forLayer: index)
            guard let format = key.quantization else { continue }
            guard format.recentTokenCount >= 0,
                key.geometry?.bytesPerToken(elementBytes: key.dtype.size) != nil
            else {
                throw CBv2KVError.backendIneligible(reason: "invalid native recent geometry")
            }
            let keyTokenBytes = try PagedKVQuantizationConfig.multiply(
                key.kvHeads, PagedKVQuantizationConfig.multiply(key.headDim, key.dtype.size))
            let valueTokenBytes = try PagedKVQuantizationConfig.multiply(
                key.kvHeads, PagedKVQuantizationConfig.multiply(key.valueHeadDim, key.dtype.size))
            owners.append(
                .init(
                    recentTokenCount: format.recentTokenCount,
                    keyTokenBytes: keyTokenBytes, valueTokenBytes: valueTokenBytes))
        }
        let policy: PagedKVNativeRecentFeasibility?
        if owners.isEmpty {
            policy = nil
        } else {
            guard let allocation = Memory.allocationFootprintPolicy() else {
                throw CBv2KVError.backendIneligible(
                    reason: "native recent allocator bounds unavailable")
            }
            policy = .init(
                owners: owners, maxPrefillChunk: config.maxPrefillChunk,
                pageSize: config.pageSize, allocation: allocation)
        }
        let nativeOwnerPeak = try policy?.maximumBytes() ?? 0
        guard result.pagedKVTransientFeasibilityBytes >= 0,
            result.minimumRequestTransientBytes >= result.pagedKVTransientFeasibilityBytes
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        let callerMinimum =
            result.minimumRequestTransientBytes - result.pagedKVTransientFeasibilityBytes
        result.minimumRequestTransientBytes = try CBv2CheckpointAllocationFootprint.add(
            callerMinimum, nativeOwnerPeak)
        result.pagedKVTransientFeasibilityBytes = nativeOwnerPeak
        result.pagedKVNativeRecentFeasibility = policy
        return result
    }
}

/// Immutable allocator and owner geometry only: no pool, request or device work.
struct PagedKVNativeRecentFeasibility: Sendable {
    struct Owner: Sendable {
        let recentTokenCount: Int
        let keyTokenBytes: Int
        let valueTokenBytes: Int
    }

    let owners: [Owner]
    let maxPrefillChunk: Int
    let pageSize: Int
    let allocation: AllocationFootprintPolicy

    func maximumBytes() throws -> Int {
        let recent = owners.map(\.recentTokenCount).max() ?? 0
        return try peakBytes(
            promptTokens: CBv2CheckpointAllocationFootprint.add(recent, maxPrefillChunk),
            maxTokens: CBv2PagedSpeculation.maxSpeculativeSpan)
    }

    func peakBytes(promptTokens: Int, maxTokens: Int) throws -> Int {
        guard promptTokens >= 0, maxTokens >= 0, maxPrefillChunk > 0, pageSize > 0 else {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        let total = try CBv2CheckpointAllocationFootprint.add(promptTokens, maxTokens)
        guard total > 0 else { return 0 }
        let chunk = min(maxPrefillChunk, promptTokens)
        // mtpDepthWithinOutputBudget reserves one output slot for the target:
        // a depth-k round writes k+1 columns, never more than remaining output.
        let columns = min(CBv2PagedSpeculation.maxSpeculativeSpan, maxTokens)
        let metadata = try PagedKVQuantizationConfig.multiply(bound(1), 2)
        let offsetBytes = try bound(MemoryLayout<Int64>.stride)
        func transferOffsets(_ tokens: Int) throws -> Int {
            guard tokens > 0 else { return 0 }
            // Any initial page alignment; each touched page may be in a
            // different segment. No allocator rounding is shared across K/V.
            let lastSlot = try CBv2CheckpointAllocationFootprint.add(tokens - 1, pageSize - 1)
            let pages = try CBv2CheckpointAllocationFootprint.add(lastSlot / pageSize, 1)
            return try PagedKVQuantizationConfig.multiply(pages, offsetBytes)
        }
        var initialPrefill = 0
        var laterPrefill = 0
        var serialPeaks = Array(repeating: 0, count: columns + 1)
        for owner in owners {
            func arrays(_ count: Int) throws -> Int {
                try CBv2CheckpointAllocationFootprint.add(
                    bound(PagedKVQuantizationConfig.multiply(count, owner.keyTokenBytes)),
                    bound(PagedKVQuantizationConfig.multiply(count, owner.valueTokenBytes)))
            }
            func oldOwner(history: Int, maximumWrite: Int) throws -> Int {
                guard history > 0 else { return 0 }
                let count = min(owner.recentTokenCount, history)
                // An uncompacted early owner retains its copy metadata and
                // transfer offsets, even when count has not reached R yet.
                return try CBv2CheckpointAllocationFootprint.add(
                    CBv2CheckpointAllocationFootprint.add(arrays(count), metadata),
                    transferOffsets(min(count, maximumWrite)))
            }
            if chunk > 0 {
                var first = try CBv2CheckpointAllocationFootprint.add(
                    CBv2CheckpointAllocationFootprint.add(arrays(chunk), metadata),
                    transferOffsets(chunk))
                if chunk > owner.recentTokenCount {
                    first = try CBv2CheckpointAllocationFootprint.add(
                        first,
                        CBv2CheckpointAllocationFootprint.add(
                            arrays(owner.recentTokenCount), metadata))
                }
                initialPrefill = try CBv2CheckpointAllocationFootprint.add(initialPrefill, first)
            }
            if promptTokens > 1 {
                // Scheduling may split even a short prompt. Bound old history
                // and the new generation independently, but never beyond the
                // actual prompt, and omit retirement when all rows fit in R.
                let oldCount = min(owner.recentTokenCount, promptTokens - 1)
                let appended = min(chunk, promptTokens - 1)
                let newCount = try CBv2CheckpointAllocationFootprint.add(
                    oldCount, min(appended, promptTokens - oldCount))
                var later = try CBv2CheckpointAllocationFootprint.add(
                    oldOwner(history: promptTokens - 1, maximumWrite: chunk),
                    CBv2CheckpointAllocationFootprint.add(
                        arrays(newCount), transferOffsets(appended)))
                if newCount > owner.recentTokenCount {
                    later = try CBv2CheckpointAllocationFootprint.add(
                        later,
                        CBv2CheckpointAllocationFootprint.add(
                            arrays(owner.recentTokenCount), metadata))
                }
                laterPrefill = try CBv2CheckpointAllocationFootprint.add(laterPrefill, later)
            }
            guard columns > 0 else { continue }
            for width in 1 ... columns {
                let history = total - width
                let oldCount = min(owner.recentTokenCount, history)
                var serial = try oldOwner(history: history, maximumWrite: max(chunk, columns))
                for column in 1 ... width {
                    serial = try CBv2CheckpointAllocationFootprint.add(
                        serial,
                        CBv2CheckpointAllocationFootprint.add(
                            arrays(CBv2CheckpointAllocationFootprint.add(oldCount, column)),
                            transferOffsets(1)))
                }
                if history == 0 {
                    serial = try CBv2CheckpointAllocationFootprint.add(serial, metadata)
                }
                let newCount = try CBv2CheckpointAllocationFootprint.add(oldCount, width)
                // Rollback can require a compact copy even below R.
                if newCount > owner.recentTokenCount || width > 1 {
                    serial = try CBv2CheckpointAllocationFootprint.add(
                        serial,
                        CBv2CheckpointAllocationFootprint.add(
                            arrays(min(owner.recentTokenCount, newCount)), metadata))
                }
                serialPeaks[width] = try CBv2CheckpointAllocationFootprint.add(
                    serialPeaks[width], serial)
            }
        }
        return max(initialPrefill, max(laterPrefill, serialPeaks.max() ?? 0))
    }

    private func bound(_ bytes: Int) throws -> Int {
        guard let result = allocation.upperBound(byteCount: bytes) else {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        return result
    }
}
