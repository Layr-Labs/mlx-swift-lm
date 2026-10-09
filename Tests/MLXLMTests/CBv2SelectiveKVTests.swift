import MLX
import Testing
@testable import MLXLMCommon

@Suite("CBv2 selective full-attention storage", .serialized)
struct CBv2SelectiveKVTests {
    private let policy = CBv2SelectiveKVPolicy(olderHistoryFraction: 0.5,
        recentTokens: 16, minimumTokens: 64, chunkTokens: 4, pruneInterval: 16)

    private func values(_ range: Range<Int>) -> MLXArray {
        MLXArray(range.map(Float.init)).reshaped([1, 1, range.count, 1])
    }

    @Test func compactedKeysKeepTheirOriginalPositionsThroughRejection() {
        let row = CBv2SelectiveSequenceKV(promptLength: 128, maxLength: 1024,
            kvHeads: 1, headDim: 1, valueHeadDim: 1, policy: policy)
        _ = row.update(keys: values(0 ..< 128), values: values(0 ..< 128))
        row.prepareForAttention(queries: MLXArray.ones([1, 2, 1, 1]),
            scale: 0.1, sinks: MLXArray([Float(0), 1]), softcap: nil)
        let kept = row.snapshot().keys.asArray(Float.self)
        #expect(kept.count < 128)
        #expect(kept == kept.sorted())
        #expect(Array(kept.prefix(4)) == [0, 1, 2, 3])
        #expect(Array(kept.suffix(16)) == (112 ..< 128).map(Float.init))
        #expect(row.absoluteOffset == 128)
        row.beginSpeculativeWrite()
        _ = row.update(keys: values(128 ..< 132), values: values(128 ..< 132))
        row.rollback(3)
        row.commitSpeculativeWrite()
        #expect(row.snapshot().keys.asArray(Float.self) == kept + [128])
        #expect(row.absoluteOffset == 129)
    }

    @Test func releasedWorkingSetsRetainEvidenceWithoutRetainingRows() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 2)
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20,
            selectiveRetention: policy))
        let rows = try backend.makeSequenceState(layerKinds: [kind], promptLength: 128, maxLength: 1024)
        let row = try #require(rows[0] as? CBv2SelectiveSequenceKV)
        _ = row.update(keys: values(0 ..< 128), values: values(0 ..< 128))
        row.prepareForAttention(queries: MLXArray.ones([1, 2, 1, 1]),
            scale: 0.1, sinks: nil, softcap: nil)
        backend.release(rows)
        backend.release(rows)
        #expect(backend.bytesReserved == 0)
        #expect(backend.prefixReuseBackend == .unknown)
        #expect(backend.selectiveKVStatistics?.sequenceLayers == 1)
        #expect(backend.selectiveKVStatistics?.pruningEvents == 1)
        #expect((backend.selectiveKVStatistics?.tokenEntriesRemoved ?? 0) > 0)
    }
}
