import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// One Super attention layer (32 query heads, 2 KV heads of 128, BF16 KV)
/// attended from segmented native paged storage and from contiguous storage
/// with the same queries, keys and values: chunked prompts, then decode steps
/// up to long contexts. The two backends reduce differently (blocked SDPA
/// over gathered pages and the paged decode kernel against SDPA), so outputs
/// are held to a stated rounding tolerance; stored keys and values are equal.
@Suite(.tags(.integration))
struct NemotronSuperPagedAttentionTests {
    static let kind = CBv2LayerKind(
        attention: .full, headDim: 128, kvHeads: 2, queryHeads: 32, modelLayerIndex: 0)
    /// Largest accepted output deviation relative to the largest output
    /// magnitude of the call; one BF16 ulp is 2^-8 of a value's magnitude.
    static let tolerance: Float = 1.0 / 64

    struct Pair {
        let contiguous: CBv2LayerCache
        let paged: PagedLayerCache
        let rows: (contiguous: CBv2SequenceKV, paged: CBv2SequenceKV)
        let backends: (contiguous: CBv2ContiguousKVBackend, paged: PagedKVBackend)
    }

    static func pair(maximumLength: Int) throws -> Pair {
        let contiguous = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 256 << 20, kvDType: .bfloat16))
        let paged = try PagedKVBackend(
            layerKinds: [kind],
            config: .init(
                capacityBytes: 256 << 20, maxPrefillChunk: 512,
                nominalMaxSequenceLength: maximumLength, segmentSizeBytes: 64 << 10,
                layerDTypes: [.bfloat16]))
        let contiguousRow = try #require(
            try contiguous.makeSequenceState(
                layerKinds: [kind], promptLength: 1, maxLength: maximumLength)[0])
        let pagedRow = try #require(
            try paged.makeSequenceState(
                layerKinds: [kind], promptLength: 1, maxLength: maximumLength)[0])
        let contiguousCache = CBv2LayerCache(layerIndex: 0, kind: kind)
        contiguousCache.setRows([contiguousRow])
        let pagedCache = try #require(paged.makeLayerCaches().first)
        pagedCache.setRows([pagedRow])
        return Pair(
            contiguous: contiguousCache, paged: pagedCache, rows: (contiguousRow, pagedRow),
            backends: (contiguous, paged))
    }

    /// Returns the largest relative output deviation of one call.
    static func attend(_ pair: Pair, length: Int, key: UInt64) throws -> Float {
        func tensor(_ heads: Int, _ salt: UInt64) -> MLXArray {
            MLXRandom.normal([1, heads, length, 128], key: MLXRandom.key(key &* 31 &+ salt))
                .asType(.bfloat16)
        }
        let queries = tensor(32, 1)
        let keys = tensor(2, 2)
        let values = tensor(2, 3)
        let scale = 1 / Float(128).squareRoot()
        let expected = pair.contiguous.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        let actual = pair.paged.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        eval(expected, actual)
        try (pair.backends.paged.pool.writeValidation).check()
        #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
        let difference = abs(actual.asType(.float32) - expected.asType(.float32)).max()
        let magnitude = abs(expected.asType(.float32)).max()
        return difference.item(Float.self) / max(magnitude.item(Float.self), .leastNormalMagnitude)
    }

    @Test(arguments: [(600, [37, 475], 40), (16_640, [512, 512, 512, 512, 7], 24)])
    func pagedAttentionMatchesContiguousWithinRounding(
        maximumLength: Int, chunks: [Int], decodes: Int
    ) throws {
        let pair = try Self.pair(maximumLength: maximumLength)
        var prefill: Float = 0
        var decode: Float = 0
        var key: UInt64 = 1
        // A long context reaches 16K tokens through repeated 512-token chunks.
        let schedule =
            maximumLength > 8_192 ? Array(repeating: chunks, count: 8).flatMap { $0 } : chunks
        for length in schedule {
            prefill = max(prefill, try Self.attend(pair, length: length, key: key))
            key += 1
        }
        for _ in 0 ..< decodes {
            decode = max(decode, try Self.attend(pair, length: 1, key: key))
            key += 1
        }
        let context = schedule.reduce(0, +) + decodes
        #expect(pair.rows.contiguous.snapshot().offset == context)
        #expect(pair.rows.paged.snapshot().offset == context)
        let stored = (pair.rows.contiguous.snapshot(), pair.rows.paged.snapshot())
        #expect(
            modelOptBitwiseEqual(
                stored.0.keys[0..., 0..., ..<context], stored.1.keys[0..., 0..., ..<context]))
        #expect(
            modelOptBitwiseEqual(
                stored.0.values[0..., 0..., ..<context], stored.1.values[0..., 0..., ..<context]))
        print(
            "PAGED attention context \(context): max relative deviation prefill \(prefill), "
                + "decode \(decode)")
        #expect(prefill <= Self.tolerance && decode <= Self.tolerance)
        pair.contiguous.setRows([])
        pair.paged.setRows([])
        pair.backends.contiguous.release([pair.rows.contiguous])
        pair.backends.paged.release([pair.rows.paged])
        #expect(pair.backends.paged.bytesReserved == 0)
    }
}
