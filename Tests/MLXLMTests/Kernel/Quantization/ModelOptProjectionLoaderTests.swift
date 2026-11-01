import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import Testing

extension KernelTests {
    struct ModelOptProjectionLoaderTests {
        private func fixture(
            _ header: [String: Any], bytes: [UInt8],
            _ body: (URL, String) throws -> Void
        ) throws {
            let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            var length = UInt64(json.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(json)
            data.append(contentsOf: bytes)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            try data.write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            try body(url, hash)
        }
        private func tensor(_ dtype: String, _ shape: [Int], _ low: Int, _ high: Int) -> [String:
            Any]
        {
            ["dtype": dtype, "shape": shape, "data_offsets": [low, high]]
        }
        @Test func verifiedFP8LoadsAndExecutes() throws {
            try fixture(
                [
                    "p.weight": tensor("F8_E4M3", [2, 2], 0, 4),
                    "p.weight_scale": tensor("F32", [], 4, 8),
                    "p.input_scale": tensor("F32", [1], 8, 12),
                ],
                bytes: [56, 0, 0, 56, 0, 0, 128, 63, 0, 0, 128, 63]
            ) { url, hash in
                let file = try ModelOptTensorFile(url: url, sha256: hash)
                let layer = try ModelOptProjectionLoader.load(
                    file: file, prefix: "p", inputDimensions: 2,
                    outputDimensions: 2, weightDType: .float32)
                let result = try layer.checkedForward(MLXArray([Float(2), 3]).reshaped(1, 2))
                #expect(arrayEqual(result, MLXArray([Float(2), 3]).reshaped(1, 2)).item(Bool.self))
                #expect(throws: (any Error).self) {
                    try ModelOptTensorFile(url: url, sha256: String(repeating: "0", count: 64))
                }
                #expect(throws: (any Error).self) {
                    try ModelOptProjectionLoader.load(
                        file: file, prefix: "p", inputDimensions: 2,
                        outputDimensions: 2, weightDType: .float32, maximumBytes: 11)
                }
            }
        }
        @Test func verifiedNVFP4LoadsWithoutDenseExpansion() throws {
            try fixture(
                [
                    "p.weight": tensor("U8", [1, 8], 0, 8),
                    "p.weight_scale": tensor("F8_E4M3", [1, 1], 8, 9),
                    "p.weight_scale_2": tensor("F32", [], 9, 13),
                    "p.input_scale": tensor("F32", [], 13, 17),
                ],
                bytes: Array(repeating: 0x22, count: 8) + [56, 0, 0, 128, 63, 0, 0, 128, 63]
            ) { url, hash in
                let file = try ModelOptTensorFile(url: url, sha256: hash)
                let layer = try ModelOptProjectionLoader.load(
                    file: file, prefix: "p", inputDimensions: 16,
                    outputDimensions: 1, weightDType: .float32)
                let output = try layer.checkedForward(MLXArray.ones([1, 16]))
                // E4M3 rounds the activation block scale 1/6 to 0.171875.
                #expect(abs(output.item(Float.self) - 16.5) < 0.001)
                let expert = try ModelOptExpertLoader.load(
                    file: file, prefixes: ["p"], expectedExperts: 1,
                    inputDimensions: 16, outputDimensions: 1, weightDType: .float32)
                let routed = try expert.checkedForward(
                    MLXArray.ones([1, 1, 16]), MLXArray([Int32(0)]))
                #expect(abs(routed.item(Float.self) - 16.5) < 0.001)
                #expect(throws: (any Error).self) {
                    try expert.checkedForward(
                        MLXArray.ones([1, 1, 16], dtype: .bfloat16), MLXArray([Int32(0)]))
                }
                #expect(throws: (any Error).self) {
                    try ModelOptExpertLoader.load(
                        file: file, prefixes: ["p", "p"], expectedExperts: 2,
                        inputDimensions: 16, outputDimensions: 1, weightDType: .float32)
                }
                #expect(throws: (any Error).self) {
                    try ModelOptExpertLoader.load(
                        file: file, prefixes: ["p"], expectedExperts: 1,
                        inputDimensions: 16, outputDimensions: 1, weightDType: .float32,
                        maximumBytes: 16)
                }
            }
        }
        @Test func rejectsOverlapTruncationUnsupportedAndLimits() throws {
            let cases: [[String: Any]] = [
                ["a": tensor("U8", [2], 0, 2), "b": tensor("U8", [2], 1, 3)],
                ["a": tensor("U8", [4], 0, 4)],
                ["a": tensor("I64", [3], 0, 3)],
                ["a": tensor("U8", [Int.max, 3], 0, 3)],
            ]
            for header in cases {
                try fixture(header, bytes: [1, 2, 3]) { url, hash in
                    #expect(throws: (any Error).self) {
                        try ModelOptTensorFile(url: url, sha256: hash)
                    }
                }
            }
            try fixture(["a": tensor("U8", [3], 0, 3)], bytes: [1, 2, 3]) { url, hash in
                #expect(throws: (any Error).self) {
                    try ModelOptTensorFile(url: url, sha256: hash, headerLimit: 2)
                }
                let file = try ModelOptTensorFile(url: url, sha256: hash, payloadLimit: 2)
                #expect(throws: (any Error).self) { try file.read("a") }
                #expect(throws: (any Error).self) { try file.read("absent") }
            }
        }
        /// A header entry that is not an object is malformed. Serializing it
        /// back to JSON would raise an Objective-C exception and end the process.
        @Test func rejectsNonObjectHeaderEntries() throws {
            let entries: [Any] = [1, "U8", [1, 2], NSNull(), true]
            for entry in entries {
                try fixture(["t": entry], bytes: [1]) { url, hash in
                    #expect(throws: ModelOptTensorFile.Failure.malformed) {
                        try ModelOptTensorFile(url: url, sha256: hash)
                    }
                }
            }
        }
        @Test func rejectsChangedFileAndSymlink() throws {
            try fixture(["a": tensor("U8", [3], 0, 3)], bytes: [1, 2, 3]) { url, hash in
                let file = try ModelOptTensorFile(url: url, sha256: hash)
                let link = url.appendingPathExtension("link")
                try FileManager.default.createSymbolicLink(at: link, withDestinationURL: url)
                defer { try? FileManager.default.removeItem(at: link) }
                #expect(throws: (any Error).self) {
                    try ModelOptTensorFile(url: link, sha256: hash)
                }
                let writer = try FileHandle(forWritingTo: url)
                try writer.seekToEnd()
                try writer.write(contentsOf: Data([4]))
                try writer.close()
                #expect(throws: (any Error).self) { try file.read("a") }
            }
        }

        @Test func crossShardExpertOrderAndCalibrationArePreserved() throws {
            let header: [String: Any] = [
                "p.weight": tensor("U8", [1, 8], 0, 8),
                "p.weight_scale": tensor("F8_E4M3", [1, 1], 8, 9),
                "p.weight_scale_2": tensor("F32", [], 9, 13),
                "p.input_scale": tensor("F32", [], 13, 17),
            ]
            let first =
                Array(repeating: UInt8(0x22), count: 8) + [56, 0, 0, 128, 63, 0, 0, 128, 63]
            let second = Array(repeating: UInt8(0x22), count: 8) + [56, 0, 0, 0, 64, 0, 0, 128, 63]
            try fixture(header, bytes: first) { firstURL, firstHash in
                try fixture(header, bytes: second) { secondURL, secondHash in
                    let a = try ModelOptTensorFile(url: firstURL, sha256: firstHash)
                    let b = try ModelOptTensorFile(url: secondURL, sha256: secondHash)
                    let layer = try ModelOptExpertLoader.load(
                        sources: [.init(file: b, prefix: "p"), .init(file: a, prefix: "p")],
                        expectedExperts: 2, inputDimensions: 16, outputDimensions: 1,
                        weightDType: .float32)
                    let output = try layer.checkedForward(
                        MLXArray.ones([1, 1, 16]), MLXArray([Int32(1), 0, 1]))
                    #expect(
                        abs(output.reshaped(3) - MLXArray([Float(16.5), 33, 16.5])).max().item(
                            Float.self) < 0.001)
                }
            }
        }

    }
}
