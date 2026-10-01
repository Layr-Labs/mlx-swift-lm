import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Regression tests for the host write into a deferred PLE slot
/// (`Qwen4ExpPLE.swift`, `DeferredSlot.withMutableBytes`).
///
/// The write went through `Data(bytesNoCopy:)`. Foundation keeps a `Data`
/// of 14 bytes or less inline, as a copy, so a write to a buffer of 14 bytes
/// or less did not reach the MLX array. A one-token slot of this tiny table
/// has 4 rows, so its scales and biases buffers are 8 bytes each. The slot
/// then read zero scales and gave zero rows.
///
/// The test is copied from `deferredSlotsMatchTheEagerGather` of PR #241
/// (`Tests/MLXLMTests/Kernel/Qwen4Exp/Qwen4ExpPLEMmapTests.swift`), without
/// the known issue and without the slot-cache metrics checks. The checkpoint
/// writer and the directory binding are private copies of that file.
///
/// The PLE table reads its directory from `Qwen4ExpPLEResidency`, a
/// process-wide binding. The suite is serialized, and each test binds the
/// directory only for its own body.
@Suite(.serialized)
struct Qwen4ExpPLEDeferredSlotTests {

    /// The deferred slot is built before its bytes exist. After the fill it
    /// holds the same rows as the eager gather, for slots of 4 rows (8-byte
    /// scales and biases), 8 rows and 12 rows.
    @Test func deferredSlotsMatchTheEagerGather() throws {
        let c = Self.configuration()
        let directory = try Self.writeCheckpoint(c)
        try Self.bound(directory) {
            let embedding = Qwen4ExpNGramEmbedding(
                c, layerIndex: 0, pleIndex: 0, mmap: true, deferredByteBudget: Int.max)

            let single = try #require(embedding.deferredGather(tokenRows: 1))
            #expect(single.values.shape == [1, 128])
            single.fill([3, 50, 50, 87])
            let eagerSingle = embedding.gather(ids: [[3, 50, 50, 87]])
            eval(single.values, eagerSingle)
            #expect(Self.isEqual(single.values, eagerSingle), "one-token deferred slot")
            #expect(Self.maxAbs(single.values) > 0, "one-token deferred slot is not zero")

            for rows in [
                [[3, 50, 50, 87], [44, 0, 1, 2]],
                [[9, 9, 9, 9], [87, 86, 85, 0]],
                [[1, 2, 3, 4], [60, 61, 62, 63]],
            ] {
                let deferred = try #require(embedding.deferredGather(tokenRows: 2))
                #expect(deferred.values.shape == [2, 128])
                deferred.fill(rows.flatMap { $0 })
                let eager = embedding.gather(ids: rows)
                eval(deferred.values, eager)
                #expect(Self.isEqual(deferred.values, eager), "rows \(rows)")
            }
            let rows = [[1, 2, 3, 4], [60, 61, 62, 63], [44, 45, 46, 47]]
            let window = try #require(embedding.deferredGather(tokenRows: 3))
            window.fill(rows.flatMap { $0 })
            let eagerWindow = embedding.gather(ids: rows)
            eval(window.values, eagerWindow)
            #expect(Self.isEqual(window.values, eagerWindow))

            embedding.releaseExternalResources()
        }
    }

    /// A second fill of the same one-token slot replaces its rows. Each
    /// width keeps two slots, so the third one-token gather reuses the slot
    /// of the first.
    @Test func oneTokenSlotIsRefilled() throws {
        let c = Self.configuration()
        let directory = try Self.writeCheckpoint(c)
        try Self.bound(directory) {
            let embedding = Qwen4ExpNGramEmbedding(
                c, layerIndex: 0, pleIndex: 0, mmap: true, deferredByteBudget: Int.max)
            for ids in [[3, 50, 50, 87], [44, 0, 1, 2], [9, 9, 9, 9]] {
                let deferred = try #require(embedding.deferredGather(tokenRows: 1))
                deferred.fill(ids)
                let eager = embedding.gather(ids: [ids])
                eval(deferred.values, eager)
                #expect(Self.isEqual(deferred.values, eager), "ids \(ids)")
            }
            embedding.releaseExternalResources()
        }
    }

    // MARK: - Helpers (copied from PR #241)

    /// Copied from `Qwen4ExpPLEMmapTests.configuration()` of PR #241.
    private static func configuration() -> Qwen4ExpTextConfiguration {
        var c = Qwen4ExpTextConfiguration()
        c.hiddenSize = 16
        c.hcCount = 2
        c.hiddenLayers = 1
        c.layerTypes = ["qwen_sparse_attention"]
        c.pleLayerIds = [1]
        c.pleEmbedDim = 128
        c.ngramSize = 3
        c.headsPerNgram = 2
        c.ngramVocabSizeBase = 17
        c.makeNgramVocabSizeDivisibleBy = 8
        c.splitNgramParts = 2
        c.vocabularySize = 64
        c.eosTokenId = [0]
        c.pleConvKernelSize = 3
        return c
    }

    /// Writes a checkpoint of two shards of 44 rows. Each row has 32 values,
    /// packed with 4 bits and group size 32. Copied from the `.standard`
    /// layout of `Qwen4ExpPLEMmapTests.write(_:layout:)` of PR #241.
    private static func writeCheckpoint(_ c: Qwen4ExpTextConfiguration) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-ple-slot-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let tables = Qwen4ExpNGramTables(c, pleIndex: 0)
        var main: [String: MLXArray] = [:]
        for (index, count) in tables.shardSizes.enumerated() {
            let values = MLXRandom.normal(
                [count, tables.headEmbedDim], key: MLXRandom.key(UInt64(9100 + index))
            ).asType(.bfloat16)
            let (weight, scales, biases) = MLX.quantized(
                values, groupSize: 32, bits: 4, mode: .affine)
            let base =
                "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards.\(index)"
            main[base + ".weight"] = weight
            main[base + ".scales"] = scales
            main[base + ".biases"] = biases ?? MLXArray.zeros(like: scales)
        }
        try save(arrays: main, url: directory.appendingPathComponent("model.safetensors"))
        var map: [String: String] = [:]
        for key in main.keys { map[key] = "model.safetensors" }
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// Runs `body` with the PLE directory bound to `directory`, then restores
    /// the old binding and deletes the directory. Copied from
    /// `Qwen4ExpPLEMmapTests.bound(_:_:)` of PR #241.
    private static func bound<T>(_ directory: URL, _ body: () throws -> T) rethrows -> T {
        let previous = Qwen4ExpPLEResidency.modelDirectory
        Qwen4ExpPLEResidency.modelDirectory = directory
        defer {
            Qwen4ExpPLEResidency.modelDirectory = previous
            try? FileManager.default.removeItem(at: directory)
        }
        return try body()
    }

    /// Copied from `Qwen4ExpKernelSupport.isEqual` of PR #241.
    private static func isEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && all(a .== b).item(Bool.self)
    }

    private static func maxAbs(_ a: MLXArray) -> Float {
        abs(a.asType(.float32)).max().item(Float.self)
    }
}
