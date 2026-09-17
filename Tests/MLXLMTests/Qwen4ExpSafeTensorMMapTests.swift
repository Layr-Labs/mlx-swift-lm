// Copyright © 2026 Eigen Labs Inc.

import Foundation
import XCTest

@testable import MLXLLM

final class Qwen4ExpSafeTensorMMapTests: XCTestCase {
    func testBF16RowsPreserveEveryBitAcrossReopenAndRepeatedLookups() throws {
        // Include signed zero, infinities and NaN payload bits. SSD lookup
        // transports checkpoint bits; it must not round-trip through Float.
        var words: [UInt16] = [0x0000, 0x8000, 0x3f80, 0x7f80, 0xff80, 0x7fc1]
        let payload = words.withUnsafeMutableBytes { Data($0) }
        let url = try makeSafeTensor(entries: ["weight": [
            "dtype": "BF16", "shape": [3, 2], "data_offsets": [0, 12]
        ]], payload: payload)
        let expected: [UInt16] = [0xff80, 0x7fc1, 0x0000, 0x8000, 0xff80, 0x7fc1]
        for _ in 0..<3 {
            let mapped = try Qwen4ExpSafeTensorMMap(url: url)
            XCTAssertEqual(try mapped.rowsUInt16(key: "weight", rows: [2, 0, 2]), expected)
        }
    }

    func testCopiesExactUInt32Rows() throws {
        let url = try makeSafeTensor(
            entries: [
                "weight": [
                    "dtype": "U32",
                    "shape": [2, 2],
                    "data_offsets": [0, 16],
                ]
            ],
            payload: uint32Data([1, 2, 3, 4]))
        let mapped = try Qwen4ExpSafeTensorMMap(url: url)

        XCTAssertEqual(try mapped.rowsUInt32(key: "weight", rows: [1]), [3, 4])
    }

    func testRejectsMalformedTensorEntryAtOpen() throws {
        let url = try makeSafeTensor(
            entries: [
                "weight": [
                    "shape": [1, 1],
                    "data_offsets": [0, 4],
                ]
            ],
            payload: uint32Data([1]))

        XCTAssertThrowsError(try Qwen4ExpSafeTensorMMap(url: url)) { error in
            XCTAssertTrue(String(describing: error).contains("malformed tensor entry weight"))
        }
    }

    func testRejectsHeaderBeyondBoundBeforeJSONParse() throws {
        var raw = UInt64(Qwen4ExpSafeTensorMMap.maxHeaderBytes + 1).littleEndian
        let data = withUnsafeBytes(of: &raw) { Data($0) }
        let url = try writeTemporary(data)

        XCTAssertThrowsError(try Qwen4ExpSafeTensorMMap(url: url)) { error in
            XCTAssertTrue(String(describing: error).contains("header exceeds"))
        }
    }

    func testRejectsTensorByteCountMismatch() throws {
        let url = try makeSafeTensor(
            entries: [
                "weight": [
                    "dtype": "U32",
                    "shape": [2, 2],
                    "data_offsets": [0, 12],
                ]
            ],
            payload: uint32Data([1, 2, 3]))
        let mapped = try Qwen4ExpSafeTensorMMap(url: url)

        XCTAssertThrowsError(try mapped.rowsUInt32(key: "weight", rows: [0])) { error in
            XCTAssertTrue(String(describing: error).contains("byte count"))
        }
    }

    func testRejectsSourceAndDestinationRowsOutOfRange() throws {
        let url = try makeSafeTensor(
            entries: [
                "weight": [
                    "dtype": "U32",
                    "shape": [1, 2],
                    "data_offsets": [0, 8],
                ]
            ],
            payload: uint32Data([1, 2]))
        let mapped = try Qwen4ExpSafeTensorMMap(url: url)
        XCTAssertThrowsError(try mapped.rowsUInt32(key: "weight", rows: [1])) { error in
            XCTAssertTrue(String(describing: error).contains("row 1 out of range"))
        }

        var destination = [UInt32](repeating: 0, count: 2)
        XCTAssertThrowsError(
            try destination.withUnsafeMutableBufferPointer { buffer in
                try mapped.copyIndexedRows(
                    key: "weight",
                    sourceRows: [0],
                    destRows: [1],
                    destinationRowCount: 1,
                    destination: XCTUnwrap(buffer.baseAddress))
            }
        ) { error in
            XCTAssertTrue(String(describing: error).contains("destination row 1 out of range"))
        }
    }

    func testRejectsBooleanTensorDimensions() throws {
        let url = try makeSafeTensor(entries: ["weight": [
            "dtype": "U32", "shape": [true, 1], "data_offsets": [0, 4]
        ]], payload: uint32Data([1]))
        XCTAssertThrowsError(try Qwen4ExpSafeTensorMMap(url: url))
    }

    func testRejectsUnrepresentableAndFractionalMetadataWithoutTrapping() throws {
        for invalid in [NSNumber(value: UInt64.max), NSNumber(value: UInt64(1) << 63),
                        NSNumber(value: -1), NSNumber(value: 0.5)] {
            for dimension in [true, false] {
                let shape: [Any] = dimension ? [invalid, 1] : [1, 1]
                let offsets: [Any] = dimension ? [0, 4] : [0, invalid]
                let url = try makeSafeTensor(entries: ["weight": [
                    "dtype": "U32",
                    "shape": shape, "data_offsets": offsets
                ]], payload: uint32Data([1]))
                XCTAssertThrowsError(try Qwen4ExpSafeTensorMMap(url: url))
            }
        }
    }

    func testRejectsOverflowingRowGeometryBeforeAllocatingOutput() throws {
        let url = try makeSafeTensor(entries: ["weight": [
            "dtype": "U32", "shape": [2, Int.max], "data_offsets": [0, 4]
        ]], payload: uint32Data([1]))
        let mapped = try Qwen4ExpSafeTensorMMap(url: url)
        XCTAssertThrowsError(try mapped.rowsUInt32(key: "weight", rows: [0, 1]))
    }

    private func makeSafeTensor(
        entries: [String: Any],
        payload: Data
    ) throws -> URL {
        let header = try JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
        var headerSize = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &headerSize) { Data($0) }
        data.append(header)
        data.append(payload)
        return try writeTemporary(data)
    }

    private func uint32Data(_ values: [UInt32]) -> Data {
        var littleEndian = values.map(\.littleEndian)
        return littleEndian.withUnsafeMutableBytes { Data($0) }
    }

    private func writeTemporary(_ data: Data) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-ple-mmap-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("model.safetensors")
        try data.write(to: url)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return url
    }
}
