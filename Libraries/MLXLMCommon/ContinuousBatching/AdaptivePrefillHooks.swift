import Foundation

public struct PrefillChunkContext: Sendable, Equatable {
    public let configuredStepSize: Int
    public let maxNumBatchedTokens: Int
    public let decodeBatchSize: Int
    public let maxRemaining: Int
    public let defaultChunkSize: Int

    public init(
        configuredStepSize: Int,
        maxNumBatchedTokens: Int,
        decodeBatchSize: Int,
        maxRemaining: Int,
        defaultChunkSize: Int
    ) {
        self.configuredStepSize = configuredStepSize
        self.maxNumBatchedTokens = maxNumBatchedTokens
        self.decodeBatchSize = decodeBatchSize
        self.maxRemaining = maxRemaining
        self.defaultChunkSize = defaultChunkSize
    }
}

public struct ColdPrefillChunkSample: Sendable, Equatable {
    public let requestedChunkSize: Int
    public let actualChunkSize: Int
    public let batchSize: Int
    public let totalTokens: Int
    public let durationSeconds: Double
    public let decodeBatchSize: Int
    public let cappedByBudget: Bool
    public let cappedByCheckpoint: Bool
    public let cappedByRemaining: Bool

    public init(
        requestedChunkSize: Int,
        actualChunkSize: Int,
        batchSize: Int,
        totalTokens: Int,
        durationSeconds: Double,
        decodeBatchSize: Int,
        cappedByBudget: Bool,
        cappedByCheckpoint: Bool,
        cappedByRemaining: Bool
    ) {
        self.requestedChunkSize = requestedChunkSize
        self.actualChunkSize = actualChunkSize
        self.batchSize = batchSize
        self.totalTokens = totalTokens
        self.durationSeconds = durationSeconds
        self.decodeBatchSize = decodeBatchSize
        self.cappedByBudget = cappedByBudget
        self.cappedByCheckpoint = cappedByCheckpoint
        self.cappedByRemaining = cappedByRemaining
    }
}
