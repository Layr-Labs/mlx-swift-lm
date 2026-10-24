import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// Lossless half-precision checkpoint wire encoding: the shared narrowing
    /// and widening helper on host bytes, and the GLM manifest that records
    /// each payload role's wire dtype. Swift logic only; no arrays are created.
    @Suite("Half-precision checkpoint wire encoding")
    struct CheckpointHalfPrecisionTests {
        private func words(_ bytes: [UInt8]) -> [UInt32] {
            stride(from: 0, to: bytes.count, by: 4).map { index in
                bytes[index ..< index + 4].enumerated().reduce(UInt32(0)) {
                    $0 | UInt32($1.element) << (8 * UInt32($1.offset))
                }
            }
        }

        private func roundTrip(_ values: [UInt32], wire: CBv2CheckpointDType, offset: Int = 0)
            -> (exact: Bool, restored: [UInt32])
        {
            // Unaligned buffers on both sides: the helper promises byte alignment only.
            var source = [UInt8](repeating: 0, count: values.count * 4 + offset)
            for (index, value) in values.enumerated() {
                withUnsafeBytes(of: value.littleEndian) {
                    source.replaceSubrange(offset + index * 4 ..< offset + index * 4 + 4, with: $0)
                }
            }
            var narrow = [UInt8](repeating: 0, count: values.count * 2 + offset)
            let exact = source.withUnsafeBytes { input in
                narrow.withUnsafeMutableBytes { output in
                    CBv2CheckpointHalfPrecision.narrow(
                        float32: input.baseAddress!.advanced(by: offset), count: values.count,
                        to: wire, destination: output.baseAddress!.advanced(by: offset))
                }
            }
            var wide = [UInt8](repeating: 0xA5, count: values.count * 4 + offset)
            narrow.withUnsafeBytes { input in
                wide.withUnsafeMutableBytes { output in
                    CBv2CheckpointHalfPrecision.widen(
                        wire, source: input.baseAddress!.advanced(by: offset),
                        count: values.count, float32: output.baseAddress!.advanced(by: offset))
                }
            }
            return (exact, words(Array(wide.dropFirst(offset))))
        }

        @Test func bfloat16RoundTripsEveryRepresentableBitPattern() {
            // Signed zero, subnormals, infinities and NaN payloads lead, so the
            // short prefixes cover them too; then all 65,536 BF16 bit patterns.
            let patterns: [UInt16] =
                [0x0000, 0x8000, 0x0001, 0x007F, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC1, 0xFFFF]
                + Array(0 ... UInt16.max)
            let values = patterns.map { UInt32($0) << 16 }
            for count in [0, 1, 7, 8, 9, 31, values.count] {
                for offset in [0, 1, 3] {
                    let result = roundTrip(
                        Array(values.prefix(count)), wire: .bfloat16, offset: offset)
                    #expect(result.exact)
                    #expect(result.restored == Array(values.prefix(count)))
                }
            }
        }

        @Test func bfloat16RefusesAnyDroppedLowBit() {
            for position in [0, 5, 8, 15, 16, 40] {
                var values = (0 ..< 41).map { UInt32($0) << 16 }
                values[position] |= 1
                #expect(!roundTrip(values, wire: .bfloat16).exact)
                values[position] = Float(1.0 / 3.0).bitPattern
                #expect(!roundTrip(values, wire: .bfloat16).exact)
            }
        }

        @Test func float16RoundTripsRepresentableValuesAndRefusesTheRest() {
            let representable: [Float] =
                [
                    0, -0.0, 1, -2, 0.5, 65504, -65504, 6.103515625e-05,
                    5.960464477539063e-08, .infinity, -.infinity,
                ] + (0 ..< 100).map { Float($0) * 0.125 }
            for offset in [0, 2] {
                let result = roundTrip(
                    representable.map(\.bitPattern), wire: .float16, offset: offset)
                #expect(result.exact)
                #expect(result.restored == representable.map(\.bitPattern))
            }
            // The canonical quiet NaN survives FP16 bit for bit; a payload in the
            // low mantissa bits does not.
            for refused: Float in [1.0 / 3.0, 70000, 1e-9, Float(bitPattern: 0x7FC0_0001)] {
                var values = representable
                values[9] = refused
                #expect(!roundTrip(values.map(\.bitPattern), wire: .float16).exact)
            }
        }

        // MARK: GLM manifest

        private func glmState(_ storage: CBv2GLM5NextCheckpointState.Storage)
            -> CBv2GLM5NextCheckpointState
        {
            .init(
                physicalPosition: 4, acceptedFrontier: 4, latentWidth: 64, indexWidth: 2,
                recurrentLayers: (0 ..< 3).map {
                    .init(
                        modelLayer: $0, convolutionShape: [1, 6, 4], recurrentShape: [1, 1, 2, 2])
                }, storage: storage)
        }

        private let attention = [
            CBv2CheckpointAttentionLayer(
                modelLayer: 3, owner: 0, window: nil, kvHeads: 1, headDim: 64, valueHeadDim: 5,
                queryHeads: 1, hasSinks: false, dtype: .float32)
        ]

        private func manifest(
            _ state: CBv2GLM5NextCheckpointState,
            layout: String = CBv2CompleteCheckpointManifest.glm5NextPagedLayout
        ) throws -> CBv2CompleteCheckpointManifest {
            CBv2CompleteCheckpointManifest(
                identity: .init(
                    modelAggregateHash: "glm-tiny-artifact", promptContractID: "glm-tiny-template",
                    buildID: "glm-tiny-native-build", numericsFingerprint: "glm-exact"),
                position: 4, chunkSize: 4, prefixTokens: [4, 5, 6, 7], cacheSalt: "tenant-a",
                assistantCodecID: nil,
                tensors: try state.targetDescriptors(attention: attention, position: 4),
                backendLayout: layout, attentionLayers: attention, glm5NextState: state)
        }

        @Test func glmStorageNarrowsOnlyActivationRolesAndKeepsRecurrentFP32() throws {
            let storage = CBv2GLM5NextCheckpointState.Storage(
                latent: .bfloat16, index: .bfloat16, convolution: .bfloat16)
            let state = glmState(storage)
            let wire = try state.targetDescriptors(attention: attention, position: 4)
            let native = try state.nativeTargetDescriptors(attention: attention, position: 4)
            #expect(
                wire.map(\.dtype) == [.bfloat16, .bfloat16]
                    + Array(repeating: [.bfloat16, .float32], count: 3).flatMap { $0 })
            #expect(native.allSatisfy { $0.dtype == .float32 })
            #expect(zip(wire, native).allSatisfy { $0.shape == $1.shape && $0.role == $1.role })
            #expect(
                zip(wire, native).allSatisfy {
                    $0.role == .recurrent
                        ? $0.byteCount == $1.byteCount : $0.byteCount * 2 == $1.byteCount
                })
            let encoded = try manifest(state)
            _ = try encoded.validateStructure()
            let decoded = try JSONDecoder().decode(
                CBv2CompleteCheckpointManifest.self, from: JSONEncoder().encode(encoded))
            #expect(decoded == encoded && decoded.glm5NextState?.storage == storage)
        }

        @Test func glmValidityCheckCoversEveryValidityElementOfASegment() throws {
            let manifest = try manifest(
                glmState(.init(latent: .bfloat16, index: .bfloat16, convolution: .bfloat16)))
            // Each token's values row is [index keys (2), pool gates (2), validity]:
            // 4 tokens x 5 BF16 elements. Only the validity column must be 0 or 1.
            let width = 5
            func validate(_ elements: Range<Int>, writing two: Int? = nil) throws {
                var data = Data()
                for element in elements {
                    let bits: UInt16 =
                        element == two
                        ? 0x4000
                        : element % width == width - 1 ? (element % 2 == 0 ? 0x3F80 : 0) : 0x3F00
                    withUnsafeBytes(of: bits.littleEndian) { data.append(contentsOf: $0) }
                }
                try CBv2GLM5NextCheckpointPayload.validate(
                    manifest: manifest, tensorIndex: 1, byteOffset: elements.lowerBound * 2,
                    data: data)
            }
            for elements in [0 ..< 20, 3 ..< 17, 4 ..< 5, 9 ..< 20, 5 ..< 9] {
                try validate(elements)
            }
            try validate(3 ..< 17, writing: 15)
            try validate(5 ..< 9, writing: 6)
            for (elements, two) in [
                (0 ..< 20, 19), (3 ..< 17, 14), (3 ..< 17, 4), (4 ..< 5, 4), (9 ..< 20, 9),
            ] {
                #expect(throws: CBv2CompleteCheckpointError.invalidSegment) {
                    try validate(elements, writing: two)
                }
            }
        }

        @Test func glmRefusesTheRetiredLayoutAndNonFloatingStorage() throws {
            #expect(
                CBv2CompleteCheckpointManifest.glm5NextPagedLayout
                    != CBv2CompleteCheckpointManifest.retiredGLM5NextPagedLayout)
            let valid = try manifest(glmState(.float32))
            _ = try valid.validateStructure()
            #expect(throws: CBv2CompleteCheckpointError.self) {
                try manifest(
                    glmState(.float32),
                    layout: CBv2CompleteCheckpointManifest.retiredGLM5NextPagedLayout
                ).validateStructure()
            }
            #expect(throws: CBv2CompleteCheckpointError.self) {
                _ = try glmState(.init(latent: .int32, index: .bfloat16, convolution: .bfloat16))
                    .targetDescriptors(attention: attention, position: 4)
            }
            // A v1 manifest has no storage record and cannot decode as v2 state.
            var object = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(valid)) as? [String: Any])
            var state = try #require(object["glm5NextState"] as? [String: Any])
            state.removeValue(forKey: "storage")
            object["glm5NextState"] = state
            #expect(throws: (any Error).self) {
                _ = try JSONDecoder().decode(
                    CBv2CompleteCheckpointManifest.self,
                    from: JSONSerialization.data(withJSONObject: object))
            }
        }
    }
}
