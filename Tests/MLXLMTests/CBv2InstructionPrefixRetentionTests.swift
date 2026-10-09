import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 protected instruction prefix", .serialized)
struct CBv2InstructionPrefixRetentionTests {
    private func row(prefix: Int) -> CBv2SelectiveSequenceKV {
        CBv2SelectiveSequenceKV(
            promptLength: 2048, maxLength: 4096,
            kvHeads: 1, headDim: 1, valueHeadDim: 1,
            policy: .init(minimumTokens: 1024, protectedPrefixTokens: prefix))
    }

    private func positions(_ range: Range<Int>) -> MLXArray {
        MLXArray(range.map(Float.init)).reshaped([1, 1, range.count, 1])
    }

    @Test func prefixSurvivesRankedPruningRepeatedUpdatesAndRejection() {
        let ordinary = row(prefix: 0)
        let protected = row(prefix: 128)
        let query = MLXArray.ones([1, 1, 1, 1])
        for state in [ordinary, protected] {
            _ = state.update(keys: positions(0 ..< 2048), values: positions(0 ..< 2048))
            state.prepareForAttention(queries: query, scale: 0.01, sinks: nil, softcap: nil)
        }
        let original = ordinary.snapshot().keys.asArray(Float.self).map(Int.init)
        #expect(!Set(0 ..< 128).isSubset(of: Set(original)))
        var kept = protected.snapshot().keys.asArray(Float.self).map(Int.init)
        #expect(Array(kept.prefix(128)) == Array(0 ..< 128))
        #expect(kept == kept.sorted() && Set(kept).count == kept.count)
        #expect(Array(kept.suffix(512)) == Array(1536 ..< 2048))

        _ = protected.update(keys: positions(2048 ..< 2304), values: positions(2048 ..< 2304))
        protected.prepareForAttention(queries: query, scale: 0.01, sinks: nil, softcap: nil)
        #expect(protected.statistics.pruningEvents == 2)
        #expect(protected.absoluteOffset == 2304)
        kept = protected.snapshot().keys.asArray(Float.self).map(Int.init)
        #expect(Array(kept.prefix(128)) == Array(0 ..< 128))
        #expect(Array(kept.suffix(512)) == Array(1792 ..< 2304))
        let confirmed = protected.snapshot().keys.asArray(Float.self)
        protected.beginSpeculativeWrite()
        _ = protected.update(keys: positions(2304 ..< 2307), values: positions(2304 ..< 2307))
        protected.prepareForAttention(queries: query, scale: 0.01, sinks: nil, softcap: nil)
        protected.rollback(3)
        protected.commitSpeculativeWrite()
        #expect(protected.statistics.pruningEvents == 2)
        #expect(protected.absoluteOffset == 2304)
        #expect(protected.snapshot().keys.asArray(Float.self) == confirmed)
    }

    @Test func boundedProtectionKeepsNativeAdmissionReservation() throws {
        #expect(CBv2SelectiveKVPolicy().protectedPrefixTokens == 0)
        #expect(CBv2SelectiveKVPolicy.maximumProtectedPrefixTokens == 256)
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        let dense = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20))
        let protected = CBv2ContiguousKVBackend(
            config: .init(
                bytesCapacity: 1 << 20,
                selectiveRetention: .init(protectedPrefixTokens: 256)))
        let denseRows = try dense.makeSequenceState(
            layerKinds: [kind], promptLength: 2048, maxLength: 4096)
        let protectedRows = try protected.makeSequenceState(
            layerKinds: [kind], promptLength: 2048, maxLength: 4096)
        #expect(dense.bytesReserved == protected.bytesReserved)
        #expect(protected.prefixReuseBackend == .unknown)
        dense.release(denseRows)
        protected.release(protectedRows)
        #expect(protected.bytesReserved == 0)
    }
}
