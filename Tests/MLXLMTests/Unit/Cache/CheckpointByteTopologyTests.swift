import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension UnitTests {
    @Suite("Authenticated checkpoint byte topology", .serialized)
    struct CheckpointByteTopologyTests {
        private let identity = CBv2CompleteCheckpointIdentity(
            modelAggregateHash: "model", promptContractID: "prompt", buildID: "build",
            numericsFingerprint: "quantized-native")

        private func topology(
            position: Int = 2_048, window: Int? = nil, dtype: DType = .bfloat16,
            quantization: PagedKVQuantizationConfig? = .init(), nativeExempt: Bool = false,
            widths: (Int, Int) = (64, 128)
        ) throws -> [CBv2CheckpointTokenByteTopology] {
            let kind = CBv2LayerKind(
                attention: window.map { .slidingWindow($0) } ?? .full,
                headDim: widths.0, valueHeadDim: widths.1, kvHeads: 2, queryHeads: 4)
            let key = PagedKVGroupKey(
                kind, dtype: dtype, separateWindow: true, quantization: quantization)
            return try [false, true].enumerated().map { index, values in
                let layout = try CBv2CheckpointPagedRoleLayout(
                    key: key, position: position,
                    tokenStart: window.map { max(0, position - $0) } ?? 0,
                    values: values)
                return try .init(
                    tensorIndex: index, descriptor: layout.descriptor(layer: 0),
                    nativeDType: CBv2CheckpointDType(dtype)!,
                    roleWidth: values ? widths.1 : widths.0,
                    absoluteTokenStart: layout.tokenStart, position: position,
                    attentionWindow: window,
                    quantization: quantization, nativeExempt: nativeExempt)
            }
        }

        @Test func allProfilesAndNativeDTypesCoverIndependentRolesExactly() throws {
            for dtype in [DType.float16, .bfloat16, .float32] {
                for bits in [(4, 4), (8, 4), (8, 8)] {
                    let profile = PagedKVQuantizationConfig(keyBits: bits.0, valueBits: bits.1)
                    for layout in try topology(dtype: dtype, quantization: profile) {
                        let components = try layout.components
                        #expect(components.map(\.kind) == [.affineMirror, .nativeRecent])
                        #expect(components[0].tokenCount == 2_048)
                        #expect(components[1].absoluteTokenStart == 1_920)
                        #expect(components[1].tokenCount == 128)
                        let affine = try #require(try layout.affineRoleBytes)
                        #expect(affine.metadataElementWidth == 4)
                        #expect(
                            affine.offsetByteOffset + affine.scaleCount * 4 == affine.rowStrideBytes
                        )
                        #expect(affine.scaleByteOffset == affine.codeBytes)
                        var offset = 0
                        for head in 0 ..< layout.headCount {
                            for part in components {
                                let span = try layout.byteSpan(
                                    head: head, component: part.kind,
                                    absoluteTokenStart: part.absoluteTokenStart,
                                    tokenCount: part.tokenCount)
                                #expect(span.byteOffset == offset)
                                #expect(
                                    span.elementWidth
                                        == (part.kind == .affineMirror ? 1 : dtype.size))
                                offset += span.byteCount
                            }
                        }
                        #expect(offset == layout.headCount * layout.headStrideBytes)
                        #expect(layout.isFullAttentionHistory)
                        #expect(throws: CBv2CompleteCheckpointError.self) {
                            try layout.byteSpan(
                                head: 2, component: .affineMirror, absoluteTokenStart: 0,
                                tokenCount: 1)
                        }
                        #expect(throws: CBv2CompleteCheckpointError.self) {
                            try layout.byteSpan(
                                head: 0, component: .nativeRecent, absoluteTokenStart: 0,
                                tokenCount: 1)
                        }
                    }
                }
            }
        }

        @Test func earlyAndWrappedWindowsRemainExplicitlyEndpointOwned() throws {
            for position in [128, 2_048] {
                for layout in try topology(position: position, window: 256) {
                    #expect(!layout.isFullAttentionHistory)
                    #expect(layout.attentionWindow == 256)
                    #expect(layout.absoluteTokenStart == max(0, position - 256))
                    #expect(try layout.components[0].tokenCount == min(position, 256))
                }
            }
            for layout in try topology(quantization: nil, nativeExempt: true) {
                #expect(layout.nativeExempt)
                #expect(try layout.components.count == 1)
                #expect(try layout.components[0].kind == .native)
                #expect(try layout.affineRoleBytes == nil)
            }
        }

        private func manifest() throws -> (
            CBv2CompleteCheckpointManifest, CBv2CompleteCheckpointCodec
        ) {
            let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)]
            let admission = AdmissionV2(
                layerKinds: kinds, bytesCapacity: 128 << 20, config: .init(watermarkFraction: 0))
            let config = PagedKVPoolConfig(
                capacityBytes: 128 << 20, segmentSizeBytes: 64 << 10,
                layerDTypes: [.bfloat16], quantization: .init())
            let codec = CBv2CompleteCheckpointCodec(
                identity: identity, layerKinds: kinds, recurrentSpec: nil, kvDTypes: [.bfloat16],
                assistant: nil, admission: admission, pagedConfig: config)
            let descriptors = try codec.tensorDescriptors(position: 2_048)
            return (
                .init(
                    identity: identity, position: 2_048, chunkSize: 256,
                    prefixTokens: Array(repeating: 7, count: 2_048), cacheSalt: "tenant",
                    assistantCodecID: nil, tensors: descriptors, backendLayout: codec.backendLayout,
                    attentionLayers: codec.historicalLayout?.layers,
                    tokenByteTopologies: try codec.checkpointTokenByteTopologies(
                        descriptors: descriptors, position: 2_048)), codec
            )
        }

        private func corrupt(
            _ manifest: CBv2CompleteCheckpointManifest,
            _ change: (inout [String: Any]) -> Void
        ) throws -> CBv2CompleteCheckpointManifest {
            var object = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest)) as? [String: Any]
            )
            change(&object)
            return try JSONDecoder().decode(
                CBv2CompleteCheckpointManifest.self,
                from: JSONSerialization.data(withJSONObject: object))
        }

        @Test func malformedGeometryAndFreshCodecMismatchRefuseBeforeReservations() throws {
            let (manifest, codec) = try manifest()
            let request = CBv2Request(
                id: .init(1), promptTokens: Array(repeating: 7, count: 2_049), maxTokens: 32,
                cacheSalt: "tenant")
            let fields: [(String, Any)] = [
                ("headStrideBytes", Int.max), ("headCount", Int.max), ("roleWidth", Int.max),
                ("absoluteTokenStart", 1), ("tokenCount", Int.max), ("nativeBandStart", 0),
                ("nativeBandCount", 0), ("nativeExempt", true), ("attentionWindow", Int.min),
                ("tensorIndex", 1), ("layer", 42),
            ]
            for (field, value) in fields {
                let bad = try corrupt(manifest) { object in
                    var topologies = object["tokenByteTopologies"] as! [[String: Any]]
                    topologies[0][field] = value
                    object["tokenByteTopologies"] = topologies
                }
                #expect(throws: (any Error).self) {
                    try codec.plan(manifest: bad, request: request)
                }
                #expect(codec.admission.bytesReserved == 0)
            }
            // A rotation change does not change descriptor byte counts. The
            // loaded codec still refuses its different numerical meaning.
            let wrongProfile = try corrupt(manifest) { object in
                var topologies = object["tokenByteTopologies"] as! [[String: Any]]
                var profile = topologies[0]["quantization"] as! [String: Any]
                profile["rotationBlockSize"] = 64
                topologies[0]["quantization"] = profile
                object["tokenByteTopologies"] = topologies
            }
            _ = try wrongProfile.validateStructure()
            #expect(throws: CBv2CompleteCheckpointError.self) {
                try codec.plan(manifest: wrongProfile, request: request)
            }
            #expect(codec.admission.bytesReserved == 0)
        }

        @Test func closedNativeAsymmetricAndAuxiliaryRolesKeepIndependentGeometry() throws {
            let kinds = [
                CBv2LayerKind(
                    attention: .full, headDim: 64, valueHeadDim: 128, kvHeads: 2, queryHeads: 4)
            ]
            let layers = try CBv2CheckpointAttentionLayer.resolveContiguousAsymmetric(
                layerKinds: kinds, dtypes: [.bfloat16])
            let descriptors = try CBv2HistoricalAttentionLayout(
                layerKinds: kinds, dtypes: [.bfloat16], allowAsymmetric: true
            ).tensorDescriptors(position: 256)
            let native = CBv2CompleteCheckpointManifest(
                identity: identity, position: 256, chunkSize: 256,
                prefixTokens: Array(repeating: 7, count: 256), cacheSalt: "tenant",
                assistantCodecID: nil,
                tensors: descriptors,
                backendLayout: CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout,
                attentionLayers: layers)
            let topology = try native.validatedTokenByteTopologies()
            #expect(topology.map(\.roleWidth) == [64, 128])
            #expect(topology[1].headStrideBytes == topology[0].headStrideBytes * 2)
            let auxiliary = CBv2CompleteCheckpointManifest(
                identity: identity, position: 256, chunkSize: 256,
                prefixTokens: Array(repeating: 7, count: 256), cacheSalt: "tenant",
                assistantCodecID: "fixture",
                tensors: [
                    try .init(role: .keys, layer: 0, shape: [1, 2, 256, 64], dtype: .bfloat16),
                    try .init(role: .values, layer: 0, shape: [1, 2, 256, 64], dtype: .bfloat16),
                    try .init(role: .convolution, layer: 3, shape: [1, 2, 2], dtype: .bfloat16),
                    try .init(role: .recurrent, layer: 3, shape: [1, 2, 2], dtype: .float32),
                    try .init(role: .assistantHidden, shape: [1, 64], dtype: .bfloat16),
                    try .init(
                        role: .assistantKeys, layer: 4, shape: [1, 2, 256, 64], dtype: .bfloat16),
                    try .init(role: .indexPositions, layer: 5, shape: [256], dtype: .int32),
                ], backendLayout: CBv2CompleteCheckpointManifest.pagedLayout)
            #expect(try auxiliary.validatedTokenByteTopologies().count == 2)
            for index in 2 ..< auxiliary.tensors.count {
                #expect(try auxiliary.validatedTokenByteTopology(tensorIndex: index) == nil)
            }
        }

        @Test func legacyOpaqueStreamsStayImportableWithoutInventedTopology() throws {
            let (manifest, codec) = try manifest()
            let legacy = try corrupt(manifest) { $0.removeValue(forKey: "tokenByteTopologies") }
            _ = try legacy.validateStructure()
            #expect(try legacy.validatedTokenByteTopology(tensorIndex: 0) == nil)
            let request = CBv2Request(
                id: .init(1), promptTokens: Array(repeating: 7, count: 2_049), maxTokens: 32,
                cacheSalt: "tenant")
            _ = try codec.plan(manifest: legacy, request: request)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let original = try encoder.encode(legacy)
            let decoded = try JSONDecoder().decode(
                CBv2CompleteCheckpointManifest.self, from: original)
            #expect(try encoder.encode(decoded) == original)
            #expect(!String(decoding: original, as: UTF8.self).contains("tokenByteTopologies"))
            let oldBytes = try CBv2CheckpointManifestMemory.reservationBytes(position: 2_048)
            let typedBytes = try CBv2CheckpointManifestMemory.reservationBytes(
                position: 2_048, includeTokenByteTopologies: true)
            #expect(typedBytes > oldBytes)
        }

        @Test func auxiliaryOnlyCompleteCheckpointsPreserveLegacyOmissionAndPlanning() throws {
            let admission = AdmissionV2(
                layerKinds: [], bytesCapacity: 64 << 20,
                config: .init(watermarkFraction: 0))
            let codec = CBv2CompleteCheckpointCodec(
                identity: identity, layerKinds: [],
                recurrentSpec: .init(layers: [
                    .init(
                        modelLayerIndex: 0, convShape: [1, 2, 2], convDType: .float16,
                        ssmShape: [1, 1, 2, 2], ssmDType: .float32)
                ]), kvDTypes: [], assistant: nil, admission: admission)
            let descriptors = try codec.tensorDescriptors(position: 256)
            #expect(descriptors.map(\.role) == [.convolution, .recurrent])
            let topology = try codec.checkpointTokenByteTopologies(
                descriptors: descriptors, position: 256)
            #expect(topology == nil)
            let manifest = CBv2CompleteCheckpointManifest(
                identity: identity, position: 256, chunkSize: 256,
                prefixTokens: Array(repeating: 7, count: 256), cacheSalt: "tenant",
                assistantCodecID: nil, tensors: descriptors, backendLayout: codec.backendLayout,
                tokenByteTopologies: topology)
            #expect(try manifest.validatedTokenByteTopologies().isEmpty)
            let request = CBv2Request(
                id: .init(1), promptTokens: Array(repeating: 7, count: 257), maxTokens: 32,
                cacheSalt: "tenant")
            _ = try codec.plan(manifest: manifest, request: request)
            let object = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
                    as? [String: Any])
            #expect(object["tokenByteTopologies"] == nil)
            #expect(admission.bytesReserved == 0)
        }

        @Test func invalidLegacyOwnerMapsRefuseBeforeAnyRangeDerivation() throws {
            let kinds = [
                CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4),
                CBv2LayerKind(
                    attention: .full, sharesKVWithLayer: 0, headDim: 64,
                    kvHeads: 2, queryHeads: 4),
            ]
            let layout = try CBv2HistoricalAttentionLayout(
                layerKinds: kinds, dtypes: [.bfloat16, .bfloat16])
            let native = CBv2CompleteCheckpointManifest(
                identity: identity, position: 256, chunkSize: 256,
                prefixTokens: Array(repeating: 7, count: 256), cacheSalt: "tenant",
                assistantCodecID: nil, tensors: try layout.tensorDescriptors(position: 256),
                backendLayout: CBv2CompleteCheckpointManifest.historicalAttentionLayout,
                attentionLayers: layout.layers)
            #expect(try native.validatedTokenByteTopologies().count == 2)
            for (field, value) in [
                ("modelLayer", 0), ("owner", -1), ("owner", 2),
                ("window", Int.min), ("kvHeads", 4), ("queryHeads", 3),
            ] {
                let bad = try corrupt(native) { object in
                    var owners = object["attentionLayers"] as! [[String: Any]]
                    owners[1][field] = value
                    object["attentionLayers"] = owners
                }
                #expect(throws: CBv2CompleteCheckpointError.invalidManifest) {
                    try bad.validatedTokenByteTopologies()
                }
                #expect(throws: CBv2CompleteCheckpointError.invalidManifest) {
                    try bad.validatedTokenByteTopology(tensorIndex: 0)
                }
            }
        }

        @Test func oversizedTopologyArraysRefuseBeforeDecodingTheirMembers() throws {
            let (manifest, _) = try manifest()
            #expect(throws: CBv2CompleteCheckpointError.invalidManifest) {
                try corrupt(manifest) { object in
                    // Empty members would otherwise fail on a missing field;
                    // the count envelope must win before member decoding.
                    object["tokenByteTopologies"] = Array(
                        repeating: [String: Any](), count: 4_097)
                }
            }
        }

        @Test func typedMetadataAliasesRetainTheirPositiveHostPermit() throws {
            let (manifest, codec) = try manifest()
            let legacyBytes = try CBv2CheckpointManifestMemory.reservationBytes(
                position: manifest.position)
            let typedBytes = try CBv2CheckpointManifestMemory.reservationBytes(
                position: manifest.position, includeTokenByteTopologies: true)
            #expect(
                typedBytes - legacyBytes
                    == CBv2CompleteCheckpointManifest.maximumTokenByteTopologyHostBytes)
            #expect(
                CBv2CompleteCheckpointManifest.maximumTokenByteTopologyHostBytes
                    >= 6 * 4096 * MemoryLayout<CBv2CheckpointTokenByteTopology>.stride)
            var retained: CBv2CompleteCheckpointManifest?
            var source: CBv2CompleteCheckpointExport?
            func construct() throws {
                let owned = try manifest.owningMetadata(admission: codec.admission)
                let replanned = try owned.owningMetadata(admission: codec.admission)
                #expect(owned.metadata === replanned.metadata)
                #expect(codec.admission.bytesReserved == typedBytes)
                source = .init(manifest: owned, arrays: [])
                retained = replanned
            }
            try construct()
            source?.close()
            source = nil
            #expect(codec.admission.bytesReserved == typedBytes)
            #expect(retained?.tokenByteTopologies == manifest.tokenByteTopologies)
            retained = nil
            #expect(codec.admission.bytesReserved == 0)
        }
    }
}
