import Foundation

/// Pure native cold-prefill geometry. Whole bidirectional media blocks may
/// move a boundary away from the nominal chunk grid; restored state must land
/// on the same boundary the current request would compute from cold.
public struct DiffusionGemmaPrefillGeometry: Sendable {
    public let promptCount: Int
    public let chunkSize: Int
    public let boundaries: [Int]
    public let capturePositions: Set<Int>
    public let lastStableBoundary: Int?
    public let lastAlignedStableBoundary: Int?

    public init(promptCount: Int, chunkSize: Int, spans: [CBv2ImageSpan] = []) throws {
        guard promptCount > 0, chunkSize > 0, promptCount <= Int.max - chunkSize else {
            throw CBv2NativeBlockError.unsupportedRequest("native prefill geometry")
        }
        var blocks = [CBv2ImageSpan]()
        var previousEnd = 0
        for span in spans {
            guard span.tokenOffset >= previousEnd, span.tokenOffset < promptCount,
                span.length > 0, span.length <= 1120,
                span.length <= promptCount - span.tokenOffset else {
                throw CBv2NativeBlockError.unsupportedRequest("native media prefix geometry")
            }
            if let index = blocks.indices.last,
                blocks[index].tokenOffset + blocks[index].length == span.tokenOffset {
                guard blocks[index].length <= 1120 - span.length else {
                    throw CBv2NativeBlockError.unsupportedRequest("native visual block budget")
                }
                blocks[index].length += span.length
            } else { blocks.append(span) }
            previousEnd = span.tokenOffset + span.length
        }
        var ends = [Int](), stable = [Int](), start = 0
        while start < promptCount {
            var naturalEnd = start + chunkSize
            for block in blocks where block.tokenOffset < naturalEnd
                && block.tokenOffset + block.length > naturalEnd {
                naturalEnd = block.tokenOffset > start ? block.tokenOffset : block.tokenOffset + block.length
            }
            let end = min(naturalEnd, promptCount)
            guard end > start else { throw CBv2NativeBlockError.unsupportedRequest("native prefill progress") }
            ends.append(end)
            if naturalEnd <= promptCount { stable.append(end) }
            start = end
        }
        self.promptCount = promptCount
        self.chunkSize = chunkSize
        self.boundaries = ends
        self.lastStableBoundary = stable.last
        self.lastAlignedStableBoundary = stable.last { $0 % chunkSize == 0 }
        self.capturePositions = Set([ends.first, stable.last, lastAlignedStableBoundary, promptCount].compactMap { $0 })
    }

    public func permitsRestore(position: Int) -> Bool { boundaries.contains(position) }

    /// Native media storage may address exact cold boundaries rather than an AR
    /// token grid. A truncated final chunk is still not a stable branch point
    /// for appended input, even when its position coincidentally aligns.
    public func permitsPersistentCapture(position: Int) -> Bool {
        position > 0
            && position <= (lastStableBoundary ?? 0) && boundaries.contains(position)
    }

    public func chunkLength(start: Int) throws -> Int {
        if start == promptCount { return 0 }
        guard start >= 0, start == 0 || boundaries.contains(start),
            let end = boundaries.first(where: { $0 > start }) else {
            throw CBv2NativeBlockError.unsupportedRequest("native restore splits cold quantum")
        }
        return end - start
    }
}
