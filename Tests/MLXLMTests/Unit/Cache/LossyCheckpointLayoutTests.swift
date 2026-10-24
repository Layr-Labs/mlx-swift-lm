import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// Eligibility and manifest identity of the lossy layouts that keep exact
    /// assistant history: the paged recurrent MTP layout and GLM's latent-only
    /// full-state layout. Swift logic only; no arrays are created.
    @Suite("Lossy checkpoint layouts beside exact assistant state")
    struct LossyCheckpointLayoutTests {
        private let profile = PagedKVQuantizationConfig(recentTokenCount: 2)

        private func glmKinds(headDim: Int = 64, kvHeads: Int = 1) -> [CBv2LayerKind] {
            [
                .init(
                    attention: .full, headDim: headDim, valueHeadDim: 5, kvHeads: kvHeads,
                    queryHeads: 2, modelLayerIndex: 3)
            ]
        }

        private func pagedConfig(quantization: PagedKVQuantizationConfig? = nil)
            -> PagedKVPoolConfig
        {
            // An explicit buffer bound: the default queries the Metal device.
            PagedKVPoolConfig(
                capacityBytes: 8 << 20, maxPrefillChunk: 32, maxBufferLength: 16 << 20,
                segmentSizeBytes: 64 << 10, layerDTypes: [.float32], quantization: quantization)
        }

        @Test func glmResolvesOnlyEligibleLatentRows() {
            let config = pagedConfig()
            #expect(
                CBv2CompleteCheckpointStorageQuantization.resolveGLM5Next(
                    profile, layerKinds: glmKinds(), pagedConfig: config, assistant: nil)
                    == profile)
            // The odd value width (2 x index + validity) never enters eligibility,
            // which the generic resolver refuses.
            #expect(
                CBv2CompleteCheckpointStorageQuantization.resolve(
                    profile, layerKinds: glmKinds(), layerDTypes: [.float32],
                    pagedConfig: config, hasAssistantState: false) == nil)
            for refused in [
                CBv2CompleteCheckpointStorageQuantization.resolveGLM5Next(
                    nil, layerKinds: glmKinds(), pagedConfig: config, assistant: nil),
                CBv2CompleteCheckpointStorageQuantization.resolveGLM5Next(
                    profile, layerKinds: glmKinds(), pagedConfig: nil, assistant: nil),
                CBv2CompleteCheckpointStorageQuantization.resolveGLM5Next(
                    profile, layerKinds: glmKinds(),
                    pagedConfig: pagedConfig(quantization: profile), assistant: nil),
                CBv2CompleteCheckpointStorageQuantization.resolveGLM5Next(
                    profile, layerKinds: glmKinds(headDim: 96), pagedConfig: config,
                    assistant: nil),
                CBv2CompleteCheckpointStorageQuantization.resolveGLM5Next(
                    profile, layerKinds: glmKinds(kvHeads: 2), pagedConfig: config,
                    assistant: nil),
                CBv2CompleteCheckpointStorageQuantization.resolveGLM5Next(
                    profile, layerKinds: [], pagedConfig: config, assistant: nil),
                // A persistent assistant without the exact-history contract.
                CBv2CompleteCheckpointStorageQuantization.resolveGLM5Next(
                    profile, layerKinds: glmKinds(), pagedConfig: config,
                    assistant: NSObject()),
            ] {
                #expect(refused == nil)
            }
        }

        @Test func onlyExactHistoryAssistantsBesidePagedRecurrentTargetsPermitLossyRows() {
            typealias Storage = CBv2CompleteCheckpointStorageQuantization
            #expect(Storage.permitsLossyTarget(besideAssistant: nil, pagedRecurrent: false))
            #expect(Storage.permitsLossyTarget(besideAssistant: nil, pagedRecurrent: true))
            #expect(!Storage.permitsLossyTarget(besideAssistant: NSObject(), pagedRecurrent: true))
            #expect(!Storage.permitsLossyTarget(besideAssistant: NSObject(), pagedRecurrent: false))
        }

        private func glmState(_ storage: CBv2GLM5NextCheckpointState.Storage)
            -> CBv2GLM5NextCheckpointState
        {
            .init(
                physicalPosition: 4, acceptedFrontier: 4, latentWidth: 64, indexWidth: 2,
                recurrentLayers: [
                    .init(
                        modelLayer: 0, convolutionShape: [1, 6, 4], recurrentShape: [1, 1, 2, 2])
                ], storage: storage)
        }

        private let attention = [
            CBv2CheckpointAttentionLayer(
                modelLayer: 1, owner: 0, window: nil, kvHeads: 1, headDim: 64, valueHeadDim: 5,
                queryHeads: 1, hasSinks: false, dtype: .float32)
        ]

        private func glmManifest(
            _ state: CBv2GLM5NextCheckpointState, layout: String,
            profile: PagedKVQuantizationConfig?, coded: PagedKVQuantizationConfig?
        ) throws -> CBv2CompleteCheckpointManifest {
            CBv2CompleteCheckpointManifest(
                identity: .init(
                    modelAggregateHash: "glm-tiny-artifact", promptContractID: "glm-tiny-template",
                    buildID: "glm-tiny-native-build", numericsFingerprint: "glm-lossy"),
                position: 4, chunkSize: 4, prefixTokens: [4, 5, 6, 7], cacheSalt: "tenant-a",
                assistantCodecID: nil,
                tensors: try state.targetDescriptors(
                    attention: attention, position: 4, latentQuantization: coded),
                backendLayout: layout, attentionLayers: attention, glm5NextState: state,
                checkpointQuantization: profile,
                checkpointNativeDTypes: profile == nil ? nil : [.float32])
        }

        @Test func glmLossyLayoutCodesOnlyLatentRowsAndRequiresItsProfile() throws {
            let state = glmState(.init(latent: .bfloat16, index: .bfloat16, convolution: .bfloat16))
            let wire = try state.targetDescriptors(
                attention: attention, position: 4, latentQuantization: profile)
            // Two coded rows (64 values K4: 32 code bytes, one FP32 scale and
            // offset), then the exact two-row BF16 band.
            #expect(wire[0].role == .keys && wire[0].dtype == .uint8)
            #expect(wire[0].shape == [1, 1, 2 * (32 + 8) + 2 * 64 * 2])
            #expect(wire.dropFirst().map(\.dtype) == [.bfloat16, .bfloat16, .float32])
            let lossy = try glmManifest(
                state, layout: CBv2CompleteCheckpointManifest.glm5NextQuantizedPagedLayout,
                profile: profile, coded: profile)
            #expect(lossy.usesLossyNativeCheckpoint && lossy.usesGLM5NextLayout)
            _ = try lossy.validateStructure()
            let decoded = try JSONDecoder().decode(
                CBv2CompleteCheckpointManifest.self, from: JSONEncoder().encode(lossy))
            #expect(decoded == lossy)
            for invalid in [
                // The lossy layout without its profile, or with native rows.
                try glmManifest(
                    state, layout: CBv2CompleteCheckpointManifest.glm5NextQuantizedPagedLayout,
                    profile: nil, coded: profile),
                try glmManifest(
                    state, layout: CBv2CompleteCheckpointManifest.glm5NextQuantizedPagedLayout,
                    profile: profile, coded: nil),
                // The exact layout never carries a profile or coded rows.
                try glmManifest(
                    state, layout: CBv2CompleteCheckpointManifest.glm5NextPagedLayout,
                    profile: profile, coded: profile),
                try glmManifest(
                    state, layout: CBv2CompleteCheckpointManifest.glm5NextPagedLayout,
                    profile: nil, coded: profile),
            ] {
                #expect(throws: CBv2CompleteCheckpointError.self) {
                    try invalid.validateStructure()
                }
            }
        }

        private func recurrentManifest(layout: String, assistantCodecID: String?) throws
            -> CBv2CompleteCheckpointManifest
        {
            CBv2CompleteCheckpointManifest(
                identity: .init(
                    modelAggregateHash: "recurrent", promptContractID: "causal",
                    buildID: "test", numericsFingerprint: "lossy-mtp"),
                position: 256, chunkSize: 256, prefixTokens: Array(repeating: 1, count: 256),
                cacheSalt: "tenant", assistantCodecID: assistantCodecID,
                tensors: [
                    try .init(role: .keys, layer: 1, shape: [1, 2, 4096], dtype: .uint8),
                    try .init(role: .values, layer: 1, shape: [1, 2, 4096], dtype: .uint8),
                    try .init(role: .convolution, layer: 0, shape: [1, 2], dtype: .bfloat16),
                    try .init(role: .recurrent, layer: 0, shape: [1, 1, 2], dtype: .float32),
                    try .init(role: .assistantHidden, layer: 0, shape: [1, 8], dtype: .float16),
                ],
                backendLayout: layout, checkpointQuantization: profile,
                checkpointNativeDTypes: [.bfloat16])
        }

        @Test func onlyTheMTPLayoutCarriesAssistantHistoryBesideLossyRows() throws {
            let mtp = CBv2CompleteCheckpointManifest.nativeQuantizedPagedMTPLayout
            let plain = CBv2CompleteCheckpointManifest.nativeQuantizedPagedLayout
            #expect(mtp != plain && mtp != CBv2CompleteCheckpointManifest.pagedLayout)
            let valid = try recurrentManifest(layout: mtp, assistantCodecID: "assistant-v1")
            #expect(valid.usesLossyNativeCheckpoint && !valid.usesGLM5NextLayout)
            _ = try valid.validateStructure()
            for invalid in [
                try recurrentManifest(layout: mtp, assistantCodecID: nil),
                try recurrentManifest(layout: plain, assistantCodecID: "assistant-v1"),
            ] {
                #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                    try invalid.validateStructure()
                }
            }
        }
    }
}
