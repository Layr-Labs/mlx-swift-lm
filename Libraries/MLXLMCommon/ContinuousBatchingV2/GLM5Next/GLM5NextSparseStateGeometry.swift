// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

public enum GLM5NextSparseStateError: Error, Equatable {
    case invalidGeometry
    case invalidPayload(String)
    case invalidPosition
    case invalidTransaction
    case incompatibleCheckpoint
    case closed
}

/// One loaded model layer and authenticated cache scope, never a routing hint.
public struct GLM5NextSparseStateBinding: Equatable {
    public let modelIdentity: ObjectIdentifier
    public let cacheSalt: String
    public let modelLayerIndex: Int
    // A saved ObjectIdentifier alone could alias a later loaded object at the
    // same address. Authorized adopters copy this binding value, not its fields.
    private let lifetime = UUID()

    public init(modelIdentity: ObjectIdentifier, cacheSalt: String, modelLayerIndex: Int) {
        self.modelIdentity = modelIdentity
        self.cacheSalt = cacheSalt
        self.modelLayerIndex = modelLayerIndex
    }
}

/// Explicit bounds for one request row. Learned pool compression stays in the model.
public struct GLM5NextSparseStateGeometry: Equatable {
    public let maximumTokens: Int
    public let maximumStagedTokens: Int
    public let indexDimension: Int
    public let latentDimension: Int
    public let indexDType: DType
    public let latentDType: DType
    public let poolSize: Int

    public init(
        maximumTokens: Int, maximumStagedTokens: Int,
        indexDimension: Int, latentDimension: Int,
        indexDType: DType, latentDType: DType, poolSize: Int = 4
    ) throws {
        guard (1 ... 1_048_576).contains(maximumTokens),
            (1 ... maximumTokens).contains(maximumStagedTokens),
            (1 ... 4096).contains(indexDimension), (1 ... 16384).contains(latentDimension),
            Self.isFloating(indexDType), Self.isFloating(latentDType), poolSize == 4
        else { throw GLM5NextSparseStateError.invalidGeometry }
        self.maximumTokens = maximumTokens
        self.maximumStagedTokens = maximumStagedTokens
        self.indexDimension = indexDimension
        self.latentDimension = latentDimension
        self.indexDType = indexDType
        self.latentDType = latentDType
        self.poolSize = poolSize
        _ = try logicalBytes(tokenCount: maximumTokens)
    }

    public func logicalBytes(tokenCount: Int) throws -> Int {
        try tensorBytes(tokenCount: tokenCount).reduce(0) {
            try CBv2CheckpointAllocationFootprint.add($0, $1)
        }
    }

    /// Fresh destination allocations, the copy selector and bounded host work.
    /// The caller still owns/charges the source tensors until the copy is fenced.
    public func allocationBytes(tokenCount: Int) throws -> Int {
        var bytes = try metadataBytes(entryCount: 1)
        for logical in try tensorBytes(tokenCount: tokenCount) where logical > 0 {
            bytes = try CBv2CheckpointAllocationFootprint.add(
                bytes, CBv2CheckpointAllocationFootprint.bound(logical))
        }
        bytes = try CBv2CheckpointAllocationFootprint.add(
            bytes, CBv2CheckpointAllocationFootprint.bound(1))
        return try CBv2CheckpointAllocationFootprint.add(
            bytes, checkedProduct(tokenCount, MemoryLayout<Int32>.stride))
    }

    func metadataBytes(entryCount: Int) throws -> Int {
        guard entryCount >= 0, entryCount <= maximumTokens else {
            throw GLM5NextSparseStateError.invalidGeometry
        }
        return try CBv2CheckpointAllocationFootprint.add(64 << 10, checkedProduct(entryCount, 128))
    }

    private func tensorBytes(tokenCount: Int) throws -> [Int] {
        guard (0 ... maximumTokens).contains(tokenCount) else {
            throw GLM5NextSparseStateError.invalidPosition
        }
        return try [
            checkedProduct(tokenCount, checkedProduct(latentDimension, latentDType.size)),
            checkedProduct(tokenCount, checkedProduct(indexDimension, indexDType.size)),
            checkedProduct(tokenCount, checkedProduct(indexDimension, indexDType.size)),
            checkedProduct(tokenCount, DType.bool.size),
            checkedProduct(tokenCount, DType.int32.size),
        ]
    }

    private func checkedProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard lhs >= 0, rhs >= 0, !overflow else { throw GLM5NextSparseStateError.invalidGeometry }
        return value
    }

    private static func isFloating(_ dtype: DType) -> Bool {
        [.float16, .bfloat16, .float32].contains(dtype)
    }
}
