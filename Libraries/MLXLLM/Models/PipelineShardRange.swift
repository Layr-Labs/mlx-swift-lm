// Copyright © 2026 Eigen Labs.
//
// PipelineShardRange -- the contiguous half-open layer interval [start, end) a
// single rank owns in a pipeline-parallel cluster, plus head/tail predicates.
// Architecture-agnostic: shared by every per-rank shard (GPT-OSS, Gemma 4, …).

import Foundation

public struct PipelineShardRange: Sendable, Equatable {
    public let start: Int
    public let end: Int
    public let totalLayers: Int
    public init(start: Int, end: Int, totalLayers: Int) {
        self.start = start
        self.end = end
        self.totalLayers = totalLayers
    }
    public var isHead: Bool { start == 0 }
    public var isTail: Bool { end == totalLayers }
    public var count: Int { end - start }
}
