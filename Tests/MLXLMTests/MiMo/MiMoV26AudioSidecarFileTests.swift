import Foundation
import XCTest

@testable import MLXVLM

final class MiMoV26AudioSidecarFileTests: XCTestCase {
    private func temporary(_ body: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "mimo-audio-sidecar-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    func testHeaderClosureRejectsHolesOverlapsTrailingAndDuplicateKeys() throws {
        func parse(_ value: String, _ count: Int = 8) throws -> MiMoV26AudioSidecarHeader {
            try .parse(Data(value.utf8), tensorBytes: count)
        }
        let good =
            "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}"
        XCTAssertEqual(try parse(good).tensors.count, 2)
        XCTAssertThrowsError(try parse(good.replacingOccurrences(of: "[4,8]", with: "[3,7]")))
        XCTAssertThrowsError(try parse(good.replacingOccurrences(of: "[4,8]", with: "[5,9]")))
        XCTAssertThrowsError(try parse(good, 9))
        XCTAssertThrowsError(try parse(good.replacingOccurrences(of: "\"b\":", with: "\"a\":")))
        XCTAssertThrowsError(try parse(good.replacingOccurrences(of: "\"BF16\"", with: "\"F64\"")))
        XCTAssertThrowsError(try parse(good.replacingOccurrences(of: "[2]", with: "[0]")))
    }

    func testFullByteDigestIncludesUnusedSuffixAndReadsRemainBounded() throws {
        try temporary { root in
            let path = root.appendingPathComponent("payload")
            let original = Data([1, 2, 3, 4, 91, 92, 93, 94])
            try original.write(to: path)
            let file = try MiMoV26AudioSidecarDescriptor(url: path, directory: false)
            var completed: [Int] = []
            XCTAssertEqual(
                try file.authenticate(
                    expected: mimoAudioDigest(original), isCancelled: { false },
                    checkpoint: { completed.append($0) }), mimoAudioDigest(original))
            XCTAssertEqual(completed, [0, 8])
            XCTAssertEqual(
                try file.read(offset: 2, count: 3, maximum: 3, isCancelled: { false }),
                Data([3, 4, 91]))
            XCTAssertThrowsError(
                try file.read(offset: 0, count: 8, maximum: 4, isCancelled: { false }))
            XCTAssertThrowsError(
                try file.read(offset: 7, count: 2, maximum: 2, isCancelled: { false }))
            // A fresh source with changed unused bytes must fail the old full hash.
            try Data([1, 2, 3, 4, 91, 92, 93, 95]).write(to: path)
            let replaced = try MiMoV26AudioSidecarDescriptor(url: path, directory: false)
            XCTAssertThrowsError(
                try replaced.authenticate(
                    expected: mimoAudioDigest(original),
                    isCancelled: { false }, checkpoint: { _ in })
            ) {
                XCTAssertEqual($0 as? MiMoV26AudioSidecarError, .wrongPayload)
            }
            XCTAssertThrowsError(try file.validate())
        }
    }

    func testSymlinkAndReplacedInodeRefuseWithoutReopeningAuthenticatedSource() throws {
        try temporary { root in
            let payload = root.appendingPathComponent("payload")
            try Data([1, 2, 3, 4]).write(to: payload)
            let link = root.appendingPathComponent("link")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: payload)
            XCTAssertThrowsError(try MiMoV26AudioSidecarDescriptor(url: link, directory: false))
            let held = try MiMoV26AudioSidecarDescriptor(url: payload, directory: false)
            try FileManager.default.moveItem(at: payload, to: root.appendingPathComponent("old"))
            try Data([1, 2, 3, 4]).write(to: payload)
            XCTAssertThrowsError(
                try held.read(offset: 0, count: 4, maximum: 4, isCancelled: { false })
            ) {
                XCTAssertEqual($0 as? MiMoV26AudioSidecarError, .changedObject)
            }
        }
    }

    func testCancellationCannotPublishDigest() throws {
        try temporary { root in
            let path = root.appendingPathComponent("payload")
            let data = Data(repeating: 7, count: 2 << 20)
            try data.write(to: path)
            let held = try MiMoV26AudioSidecarDescriptor(url: path, directory: false)
            XCTAssertThrowsError(
                try held.authenticate(
                    expected: mimoAudioDigest(data),
                    isCancelled: { true },
                    checkpoint: { _ in XCTFail("cancelled source reached checkpoint") })
            ) {
                XCTAssertEqual($0 as? MiMoV26AudioSidecarError, .cancelled)
            }
        }
    }
}
