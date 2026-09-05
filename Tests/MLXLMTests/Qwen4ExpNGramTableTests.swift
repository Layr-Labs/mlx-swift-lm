// Qwen4ExpNGramTableTests.swift
//
// The n-gram table's cache must never change a value.
//
// The whole point of the disk-resident table is that the ceiling is a MEMORY
// decision and nothing else. A run with the cache off, a run with a cache
// that evicts on almost every row, and a run that holds the whole table must
// return the same rows in the same order.
//
// Ported from
// `Layr-Labs/mlxfast-qwen38-125b-a6b-engine-dev@b0b8d28:Tests/MLXFastTests/Qwen4ExpNGramCacheExactnessTests.swift`.
// Byte level only, so nothing here needs the MLX runtime or a GPU: the rows
// are compared as the raw checkpoint bytes the cache actually holds, which is
// where the invariant lives.

import Foundation
import XCTest

@testable import MLXLLM

final class Qwen4ExpNGramTableTests: XCTestCase {

    // A small table with the checkpoint's quantization: 4-bit affine, group 32.
    private let layout = Qwen4ExpNGramTableLayout(
        shardCount: 4, rowsPerShard: 8, rowDimensions: 64)
    private let tensorPrefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"

    /// Ids chosen to cross every shard, repeat rows, and revisit an id after
    /// enough traffic to have evicted it from a small cache.
    private let requestedIds: [Int] = [
        0, 1, 8, 9, 17, 31, 0, 24, 25, 7, 16, 1, 30, 2, 8, 31, 0, 15, 23, 9,
    ]

    // MARK: Synthetic table directory

    /// Write one safetensors file holding every shard of the small table.
    ///
    /// The values are deterministic and distinct per (shard, row, byte), so a
    /// row served from the wrong place cannot pass by accident.
    private func writeShardFile(at url: URL) throws {
        var tensors: [String: [String: Any]] = [:]
        var payload = Data()
        var offset = 0

        func append(_ name: String, dtype: String, shape: [Int], bytes: Data) {
            tensors[name] = [
                "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.count],
            ]
            payload.append(bytes)
            offset += bytes.count
        }

        for shard in 0 ..< layout.shardCount {
            var weight = Data(count: layout.rowsPerShard * layout.weightBytesPerRow)
            for index in 0 ..< weight.count {
                weight[index] = UInt8((shard &* 37 &+ index &* 11 &+ 3) % 251)
            }
            var scales = Data(count: layout.rowsPerShard * layout.groupBytesPerRow)
            var biases = Data(count: layout.rowsPerShard * layout.groupBytesPerRow)
            for index in 0 ..< scales.count {
                // bfloat16 bit patterns that stay small and finite.
                scales[index] = UInt8((shard &* 5 &+ index &* 3 &+ 60) % 64)
                biases[index] = UInt8((shard &* 7 &+ index &* 13 &+ 32) % 64)
            }
            let base = "\(tensorPrefix).shard_\(shard)"
            append(
                "\(base).weight", dtype: "U32",
                shape: [layout.rowsPerShard, layout.weightWordsPerRow], bytes: weight)
            append(
                "\(base).scales", dtype: "BF16",
                shape: [layout.rowsPerShard, layout.groupsPerRow], bytes: scales)
            append(
                "\(base).biases", dtype: "BF16",
                shape: [layout.rowsPerShard, layout.groupsPerRow], bytes: biases)
        }

        let header = try JSONSerialization.data(withJSONObject: tensors, options: [.sortedKeys])
        var file = Data()
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { file.append(contentsOf: $0) }
        file.append(header)
        file.append(payload)
        try file.write(to: url)
    }

    private func makeTable(ceilingBytes: Int, directory: URL) throws -> Qwen4ExpNGramTable {
        try Qwen4ExpNGramTable(
            shardFiles: try Qwen4ExpNGramTable.shardFiles(in: directory),
            tensorPrefix: tensorPrefix,
            layout: layout,
            ceilingBytes: ceilingBytes
        )
    }

    private func makeDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-ngram-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try writeShardFile(at: directory.appendingPathComponent("shards.safetensors"))
        return directory
    }

    // MARK: Geometry

    func testLayoutMatchesTheCheckpointRowCost() {
        // The pinned checkpoint: 128 shards, 2,500,012 rows, 160 values, 4-bit
        // affine group 32. That is 80 weight bytes plus two 10-byte group
        // vectors, so 100 bytes a row and 29.80 GiB in total.
        let pinned = Qwen4ExpNGramTableLayout(
            shardCount: 128, rowsPerShard: 2_500_012, rowDimensions: 160)
        XCTAssertEqual(pinned.weightWordsPerRow, 20)
        XCTAssertEqual(pinned.weightBytesPerRow, 80)
        XCTAssertEqual(pinned.groupsPerRow, 5)
        XCTAssertEqual(pinned.groupBytesPerRow, 10)
        XCTAssertEqual(pinned.bytesPerRow, 100)
        XCTAssertEqual(pinned.rowCount, 320_001_536)
        XCTAssertEqual(pinned.rowCount * pinned.bytesPerRow, 32_000_153_600)
    }

    // MARK: Ceiling resolution

    func testCeilingDefaultsToOneGibibyte() throws {
        XCTAssertEqual(try Qwen4ExpNGramCacheLimit.resolve(environment: [:]), 1 << 30)
    }

    func testCeilingReadsTheEnvironmentAndTheFlag() throws {
        XCTAssertEqual(
            try Qwen4ExpNGramCacheLimit.resolve(
                environment: [Qwen4ExpNGramCacheLimit.environmentName: "65536"]),
            65536)
        XCTAssertEqual(
            try Qwen4ExpNGramCacheLimit.resolve(
                flagValue: "0",
                environment: [Qwen4ExpNGramCacheLimit.environmentName: "65536"]),
            0)
    }

    func testCeilingRefusesANegativeOrUnparseableValue() {
        XCTAssertThrowsError(try Qwen4ExpNGramCacheLimit.parse("-1"))
        XCTAssertThrowsError(try Qwen4ExpNGramCacheLimit.parse("1GiB"))
    }

    // MARK: Exactness

    func testRowBytesAreIdenticalAtEveryCeiling() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Off, room for two rows, and room for the whole table.
        let ceilings = [0, 2 * layout.bytesPerRow, layout.rowCount * layout.bytesPerRow]
        var results: [[[UInt8]]] = []
        for ceiling in ceilings {
            let table = try makeTable(ceilingBytes: ceiling, directory: directory)
            results.append(requestedIds.map { table.readRow($0) })
        }

        for index in 1 ..< results.count {
            XCTAssertEqual(
                results[0], results[index],
                "ceiling \(ceilings[index]) returned different bytes than a cache-free read")
        }
        // The synthetic data must actually distinguish rows, or the assertion
        // above would pass on an all-zero table.
        XCTAssertEqual(Set(results[0].map { Data($0) }).count, Set(requestedIds).count)
    }

    func testTheCacheIsUsedAndStillEvicts() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let off = try makeTable(ceilingBytes: 0, directory: directory)
        for id in requestedIds { _ = off.readRow(id) }
        XCTAssertEqual(off.hitCount, 0, "a zero ceiling must never serve a cached row")
        XCTAssertEqual(off.missCount, requestedIds.count)

        let whole = try makeTable(
            ceilingBytes: layout.rowCount * layout.bytesPerRow, directory: directory)
        for id in requestedIds { _ = whole.readRow(id) }
        XCTAssertEqual(whole.missCount, Set(requestedIds).count, "each row is read once")
        XCTAssertEqual(whole.hitCount, requestedIds.count - Set(requestedIds).count)

        let tiny = try makeTable(ceilingBytes: 2 * layout.bytesPerRow, directory: directory)
        for id in requestedIds { _ = tiny.readRow(id) }
        XCTAssertGreaterThan(tiny.missCount, whole.missCount, "a tiny ceiling must evict")
        XCTAssertLessThanOrEqual(tiny.missCount, requestedIds.count)
    }

    func testRowBytesMatchTheFileDirectly() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("shards.safetensors")
        let header = try SafetensorsHeader.read(url)
        let file = try Data(contentsOf: url)
        let table = try makeTable(ceilingBytes: 1 << 20, directory: directory)

        for id in requestedIds {
            let shard = id / layout.rowsPerShard
            let row = id % layout.rowsPerShard
            let base = "\(tensorPrefix).shard_\(shard)"
            let dataBase = header.dataBaseOffset
            let weight = header.tensors["\(base).weight"]!
            let scales = header.tensors["\(base).scales"]!
            let biases = header.tensors["\(base).biases"]!

            var expected = Data()
            let weightStart = dataBase + weight.dataStart + row * layout.weightBytesPerRow
            expected.append(file[weightStart ..< (weightStart + layout.weightBytesPerRow)])
            let scaleStart = dataBase + scales.dataStart + row * layout.groupBytesPerRow
            expected.append(file[scaleStart ..< (scaleStart + layout.groupBytesPerRow)])
            let biasStart = dataBase + biases.dataStart + row * layout.groupBytesPerRow
            expected.append(file[biasStart ..< (biasStart + layout.groupBytesPerRow)])

            XCTAssertEqual(Data(table.readRow(id)), expected, "row \(id) does not match the file")
        }
    }

    // MARK: Refusals

    func testAMissingShardIsRefusedByName() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertThrowsError(
            try Qwen4ExpNGramTable(
                shardFiles: try Qwen4ExpNGramTable.shardFiles(in: directory),
                tensorPrefix: tensorPrefix,
                layout: Qwen4ExpNGramTableLayout(
                    shardCount: 8, rowsPerShard: 8, rowDimensions: 64),
                ceilingBytes: 0
            )
        ) { error in
            XCTAssertTrue(
                "\(error)".contains("n-gram shards"), "unexpected refusal: \(error)")
        }
    }

    func testASingleFileIsRefusedByName() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("shards.safetensors")
        XCTAssertThrowsError(try Qwen4ExpNGramTable.shardFiles(in: file)) { error in
            XCTAssertTrue("\(error)".contains("is a file"), "unexpected refusal: \(error)")
            XCTAssertTrue("\(error)".contains("DIRECTORY"), "unexpected refusal: \(error)")
        }
    }

    func testAnEmptyDirectoryIsRefusedByName() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-ngram-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertThrowsError(try Qwen4ExpNGramTable.shardFiles(in: directory)) { error in
            XCTAssertTrue(
                "\(error)".contains("no .safetensors file"), "unexpected refusal: \(error)")
        }
    }
}
