extension CBv2CompleteCheckpointManifest {
    /// Structural validation only. The loaded assistant codec still must match
    /// every descriptor, decode the exact metadata/token witness, and authorize
    /// restoration into its current request owner. No arrays are allocated here.
    func validateHistoricalAssistantStructure(targetTensorCount: Int) throws {
        let auxiliary = Array(tensors.dropFirst(targetTensorCount))
        let allowed: Set<CBv2CheckpointTensorRole> = [
            .assistantKeys, .assistantValues, .assistantCacheMetadata,
            .assistantHidden, .assistantTokens, .assistantFrontier,
        ]
        guard !auxiliary.isEmpty, auxiliary.allSatisfy({ allowed.contains($0.role) }),
              auxiliary.contains(where: { $0.role == .assistantHidden && $0.dtype.isFloatingPoint }),
              auxiliary.contains(where: { $0.role == .assistantTokens && $0.dtype == .int32
                  && $0.layer == nil && $0.shape == [1, position] }),
              auxiliary.contains(where: { $0.role == .assistantCacheMetadata
                  && ($0.dtype == .int32 || $0.dtype == .int64) }) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let keys = auxiliary.filter { $0.role == .assistantKeys }
        let values = auxiliary.filter { $0.role == .assistantValues }
        guard !keys.isEmpty, keys.count == values.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        for key in keys {
            guard let head = key.layer, key.shape.count == 4, key.shape[0] == 1,
                  key.dtype.isFloatingPoint,
                  let value = values.first(where: { $0.layer == head }),
                  value.shape.count == 4, value.dtype == key.dtype,
                  Array(value.shape.prefix(3)) == Array(key.shape.prefix(3)) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
        }
    }
}
