import Foundation

/// Physical components of one canonical, head-major checkpoint role stream.
/// The affine mirror includes the rows that also have an original native band.
public enum CBv2CheckpointTokenByteComponentKind: String, Codable, Sendable {
    case native, affineMirror, nativeRecent
}

public struct CBv2CheckpointTokenByteComponent: Sendable, Equatable {
    public let kind: CBv2CheckpointTokenByteComponentKind
    public let absoluteTokenStart: Int
    public let tokenCount: Int
    public let byteOffsetInHead: Int
    public let tokenStrideBytes: Int
    public let elementWidth: Int
    public let encodingIdentity: String
}

/// A bounded byte span in the original descriptor stream, never a re-encoding.
public struct CBv2CheckpointTokenByteSpan: Sendable, Equatable {
    public let tensorIndex: Int
    public let head: Int
    public let kind: CBv2CheckpointTokenByteComponentKind
    public let absoluteTokenStart: Int
    public let tokenCount: Int
    public let byteOffset: Int
    public let byteCount: Int
    public let elementWidth: Int
    public let encodingIdentity: String
}

/// The code and FP32 metadata locations inside each affine mirror row.
public struct CBv2CheckpointAffineRoleBytes: Sendable, Equatable {
    public let codeBytes: Int
    public let scaleByteOffset: Int
    public let offsetByteOffset: Int
    public let scaleCount: Int
    public let metadataElementWidth: Int
    public let rowStrideBytes: Int
}

/// Authenticated token-to-byte geometry supplied by the loaded checkpoint codec.
/// Values are fixed-size metadata; component iteration never builds a page list.
public struct CBv2CheckpointTokenByteTopology: Codable, Sendable, Equatable {
    public let tensorIndex: Int
    public let layer: Int
    public let role: CBv2CheckpointTensorRole
    public let headCount: Int
    public let roleWidth: Int
    public let nativeDType: CBv2CheckpointDType
    public let absoluteTokenStart: Int
    public let tokenCount: Int
    public let attentionWindow: Int?
    public var isFullAttentionHistory: Bool { attentionWindow == nil }
    public let headStrideBytes: Int
    public let quantization: PagedKVQuantizationConfig?
    public let nativeExempt: Bool
    public let nativeBandStart: Int
    public let nativeBandCount: Int

    public var affineRoleBytes: CBv2CheckpointAffineRoleBytes? {
        get throws {
            try validateGeometry()
            guard let quantization else { return nil }
            let layout = try quantization.rowLayout(headDim: roleWidth)
            return .init(
                codeBytes: role == .keys ? layout.keyDataBytes : layout.valueDataBytes,
                scaleByteOffset: role == .keys ? layout.keyScaleOffset : layout.valueScaleOffset,
                offsetByteOffset: role == .keys ? layout.keyOffsetOffset : layout.valueOffsetOffset,
                scaleCount: roleWidth / quantization.groupSize, metadataElementWidth: 4,
                rowStrideBytes: role == .keys ? layout.keyRowBytes : layout.valueRowBytes)
        }
    }

    init(
        tensorIndex: Int, descriptor: CBv2CheckpointTensorDescriptor,
        nativeDType: CBv2CheckpointDType, roleWidth: Int,
        absoluteTokenStart: Int, position: Int, attentionWindow: Int? = nil,
        quantization: PagedKVQuantizationConfig?, nativeExempt: Bool = false
    ) throws {
        guard let layer = descriptor.layer, descriptor.shape.count >= 2,
            position > absoluteTokenStart, absoluteTokenStart >= 0
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        self.tensorIndex = tensorIndex
        self.layer = layer
        self.role = descriptor.role
        self.headCount = descriptor.shape[1]
        self.roleWidth = roleWidth
        self.nativeDType = nativeDType
        self.absoluteTokenStart = absoluteTokenStart
        let retainedCount = position - absoluteTokenStart
        self.tokenCount = retainedCount
        self.attentionWindow = attentionWindow
        self.quantization = quantization
        self.nativeExempt = nativeExempt
        nativeBandCount = quantization.map { min($0.recentTokenCount, retainedCount) } ?? 0
        nativeBandStart = position - nativeBandCount
        let nativeBytes = try Self.multiply(roleWidth, nativeDType.mlxDType.size)
        let rowBytes = try Self.rowBytes(
            role: role, width: roleWidth, nativeBytes: nativeBytes, quantization: quantization)
        headStrideBytes = try Self.add(
            Self.multiply(tokenCount, rowBytes), Self.multiply(nativeBandCount, nativeBytes))
        try validate(descriptor: descriptor, position: position)
    }

    /// At most two entries in exact physical order within each head.
    public var components: [CBv2CheckpointTokenByteComponent] {
        get throws {
            try validateGeometry()
            let nativeBytes = try Self.multiply(roleWidth, nativeDType.mlxDType.size)
            let rowBytes = try Self.rowBytes(
                role: role, width: roleWidth, nativeBytes: nativeBytes, quantization: quantization)
            let identity = quantization?.identity ?? "native-v1"
            let base = "\(identity)-\(role.rawValue)-w\(roleWidth)-\(nativeDType.rawValue)"
            var result = [
                CBv2CheckpointTokenByteComponent(
                    kind: quantization == nil ? .native : .affineMirror,
                    absoluteTokenStart: absoluteTokenStart, tokenCount: tokenCount,
                    byteOffsetInHead: 0, tokenStrideBytes: rowBytes,
                    elementWidth: quantization == nil ? nativeDType.mlxDType.size : 1,
                    encodingIdentity: base)
            ]
            if nativeBandCount > 0 {
                result.append(
                    .init(
                        kind: .nativeRecent, absoluteTokenStart: nativeBandStart,
                        tokenCount: nativeBandCount,
                        byteOffsetInHead: try Self.multiply(tokenCount, rowBytes),
                        tokenStrideBytes: nativeBytes, elementWidth: nativeDType.mlxDType.size,
                        encodingIdentity: "\(base)-original-band"))
            }
            return result
        }
    }

    public func byteSpan(
        head: Int, component: CBv2CheckpointTokenByteComponentKind,
        absoluteTokenStart start: Int, tokenCount count: Int
    ) throws -> CBv2CheckpointTokenByteSpan {
        guard head >= 0, head < headCount,
            let part = try components.first(where: { $0.kind == component }),
            count > 0, count <= part.tokenCount, start >= part.absoluteTokenStart,
            start - part.absoluteTokenStart <= part.tokenCount - count
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        let offset = try Self.add(
            Self.add(Self.multiply(head, headStrideBytes), part.byteOffsetInHead),
            Self.multiply(start - part.absoluteTokenStart, part.tokenStrideBytes))
        return .init(
            tensorIndex: tensorIndex, head: head, kind: component,
            absoluteTokenStart: start, tokenCount: count, byteOffset: offset,
            byteCount: try Self.multiply(count, part.tokenStrideBytes),
            elementWidth: part.elementWidth, encodingIdentity: part.encodingIdentity)
    }

    func validate(descriptor: CBv2CheckpointTensorDescriptor, position: Int) throws {
        try validateGeometry()
        guard tensorIndex >= 0, layer == descriptor.layer, role == descriptor.role,
            absoluteTokenStart <= position, tokenCount == position - absoluteTokenStart,
            descriptor.shape.first == 1,
            descriptor.byteCount == (try Self.multiply(headCount, headStrideBytes))
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        if quantization != nil {
            guard descriptor.dtype == .uint8,
                descriptor.shape == [1, headCount, headStrideBytes], !nativeExempt
            else { throw CBv2CompleteCheckpointError.invalidManifest }
        } else {
            guard descriptor.dtype == nativeDType,
                descriptor.shape == [1, headCount, tokenCount, roleWidth]
            else { throw CBv2CompleteCheckpointError.invalidManifest }
        }
    }

    private func validateGeometry() throws {
        guard tensorIndex >= 0, layer >= 0, role == .keys || role == .values,
            headCount > 0, headCount <= Int(Int32.max), roleWidth > 0,
            roleWidth <= Int(Int32.max), nativeDType.isFloatingPoint,
            absoluteTokenStart >= 0, tokenCount > 0, tokenCount <= Int(Int32.max),
            nativeBandCount >= 0, nativeBandCount <= tokenCount
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        let end = try Self.add(absoluteTokenStart, tokenCount)
        if let attentionWindow {
            guard attentionWindow > 0, attentionWindow <= Int(Int32.max),
                absoluteTokenStart == max(0, end - attentionWindow)
            else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
        } else if absoluteTokenStart != 0 {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        let expectedBand = quantization.map { min($0.recentTokenCount, tokenCount) } ?? 0
        guard nativeBandCount == expectedBand, nativeBandStart == end - nativeBandCount,
            quantization == nil || !nativeExempt
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        let nativeBytes = try Self.multiply(roleWidth, nativeDType.mlxDType.size)
        let rowBytes = try Self.rowBytes(
            role: role, width: roleWidth, nativeBytes: nativeBytes, quantization: quantization)
        guard
            headStrideBytes
                == (try Self.add(
                    Self.multiply(tokenCount, rowBytes), Self.multiply(nativeBandCount, nativeBytes)
                ))
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        _ = try Self.multiply(headCount, headStrideBytes)
    }

    private static func rowBytes(
        role: CBv2CheckpointTensorRole, width: Int, nativeBytes: Int,
        quantization: PagedKVQuantizationConfig?
    ) throws -> Int {
        guard let quantization else { return nativeBytes }
        do {
            let layout = try quantization.rowLayout(headDim: width)
            return role == .keys ? layout.keyRowBytes : layout.valueRowBytes
        } catch { throw CBv2CompleteCheckpointError.invalidManifest }
    }

    static func multiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow, lhs >= 0, rhs >= 0 else {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        return result
    }

    static func add(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow, lhs >= 0, rhs >= 0 else {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        return result
    }
}
