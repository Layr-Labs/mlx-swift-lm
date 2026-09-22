import Foundation

/// Native encoder ordering state. Diffusion canvases, RNG and provisional
/// logits are never checkpoint payloads. Model-specific geometry is checked by
/// the loaded native codec before allocation or adoption.
public struct CBv2NativeBlockCheckpointState: Codable, Sendable, Equatable {
    public let windowPhysicalLength: Int
    public let windowCursor: Int

    public init(windowPhysicalLength: Int, windowCursor: Int) {
        self.windowPhysicalLength = windowPhysicalLength
        self.windowCursor = windowCursor
    }

    func validate() throws {
        guard windowPhysicalLength > 0, windowPhysicalLength <= Int(Int32.max),
            windowCursor > 0, windowCursor <= windowPhysicalLength
        else { throw CBv2CompleteCheckpointError.invalidManifest }
    }
}

/// Canonical little-endian bit packing, only for the native-block layout.
/// A 262144-token, 19-bit vocabulary fits the existing 1 MiB encrypted manifest
/// bound including base64. Legacy checkpoint encodings are unchanged. This is
/// integer compression, not tensor quantization or a tokenizer transformation.
struct CBv2NativeBlockPackedTokens: Codable {
    let bitWidth: Int
    let bytes: Data

    init(tokens: [Int]) throws {
        guard !tokens.isEmpty, tokens.count <= CBv2CompleteCheckpointManifest.maximumEncodedBytes / 2,
            tokens.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) })
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        let largest = tokens.max()!
        bitWidth = max(1, Int.bitWidth - largest.leadingZeroBitCount)
        var output = Data()
        output.reserveCapacity((tokens.count * bitWidth + 7) / 8)
        var buffer: UInt64 = 0
        var available = 0
        for token in tokens {
            buffer |= UInt64(token) << available
            available += bitWidth
            while available >= 8 {
                output.append(UInt8(truncatingIfNeeded: buffer))
                buffer >>= 8
                available -= 8
            }
        }
        if available > 0 { output.append(UInt8(truncatingIfNeeded: buffer)) }
        bytes = output
    }

    func unpack(count: Int) throws -> [Int] {
        guard count > 1, count <= CBv2CompleteCheckpointManifest.maximumEncodedBytes / 2,
            (1...31).contains(bitWidth), bytes.count == (count * bitWidth + 7) / 8
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        var result = [Int]()
        result.reserveCapacity(count)
        var buffer: UInt64 = 0
        var available = 0
        let mask = (UInt64(1) << bitWidth) - 1
        for byte in bytes {
            buffer |= UInt64(byte) << available
            available += 8
            while available >= bitWidth && result.count < count {
                result.append(Int(buffer & mask))
                buffer >>= bitWidth
                available -= bitWidth
            }
        }
        // Reject alternate encodings and nonzero padding. The encrypted bytes
        // and the decoded token sequence have one canonical representation.
        guard result.count == count, buffer == 0,
            bitWidth == max(1, Int.bitWidth - result.max()!.leadingZeroBitCount)
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        return result
    }
}
