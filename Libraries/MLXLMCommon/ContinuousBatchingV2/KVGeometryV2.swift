import Foundation

/// Native attention storage geometry. K and V share a dtype and head count,
/// but their feature widths are independent. No allocation or MLX dependency.
public struct CBv2KVGeometry: Sendable, Equatable {
    public let kvHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int

    public init?(kvHeads: Int, keyHeadDim: Int, valueHeadDim: Int) {
        guard [kvHeads, keyHeadDim, valueHeadDim].allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) })
        else {
            return nil
        }
        self.kvHeads = kvHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
    }

    /// Returns nil for invalid element widths or overflowing arithmetic.
    /// Extra storage is already bytes; it is not multiplied by tensor dtype.
    public func bytesPerToken(elementBytes: Int, extraBytes: Int = 0) -> Int? {
        guard elementBytes > 0, extraBytes >= 0,
            let width = Self.add(keyHeadDim, valueHeadDim),
            let elements = Self.multiply(kvHeads, width),
            let bytes = Self.multiply(elements, elementBytes)
        else { return nil }
        return Self.add(bytes, extraBytes)
    }

    public func storageBytes(tokens: Int, elementBytes: Int, extraBytesPerToken: Int = 0) -> Int? {
        guard tokens >= 0,
            let bytes = bytesPerToken(elementBytes: elementBytes, extraBytes: extraBytesPerToken)
        else { return nil }
        return Self.multiply(tokens, bytes)
    }

    static func add(_ a: Int, _ b: Int) -> Int? {
        let result = a.addingReportingOverflow(b)
        return result.overflow ? nil : result.partialValue
    }
    static func multiply(_ a: Int, _ b: Int) -> Int? {
        let result = a.multipliedReportingOverflow(by: b)
        return result.overflow ? nil : result.partialValue
    }
}
