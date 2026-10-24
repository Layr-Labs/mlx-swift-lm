import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The Super assistant's exact prompt history beside lossy older target rows,
/// on synthetic weights. A paged recurrent codec with the speculating owner's
/// assistant resolves the checkpoint profile to the MTP layout; a restore
/// decodes the coded rows to the profile's reference reconstruction while the
/// recent band, the recurrent state and the assistant history come back bit
/// for bit. Off paged recurrent storage the same assistant keeps native.
@Suite(.tags(.integration))
struct NemotronSuperLossyCompleteCheckpointTests {
    typealias Tests = NemotronSuperMTPAssistantTests

    private let position = 64
    private let profile = PagedKVQuantizationConfig(recentTokenCount: 8)
    private let kinds = [
        CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2, modelLayerIndex: 1)
    ]
    private let spec = CBv2RecurrentStateSpec(layers: [
        .init(
            modelLayerIndex: 0,
            convShape: [1, 2], convDType: .float32, ssmShape: [1, 1, 2], ssmDType: .float32)
    ])
    private let identity = CBv2CompleteCheckpointIdentity(
        modelAggregateHash: "super-synthetic", promptContractID: "causal", buildID: "test",
        numericsFingerprint: "lossy-mtp")

    private func admission(_ config: PagedKVPoolConfig? = nil) throws -> AdmissionV2 {
        let admissionConfig = AdmissionV2.Config(
            watermarkFraction: 0, elementBytes: 4,
            fixedBytesPerRequest: try spec.fixedBytesPerRequest())
        guard let config else {
            return AdmissionV2(layerKinds: kinds, bytesCapacity: 64 << 20, config: admissionConfig)
        }
        return AdmissionV2(
            layerKinds: kinds, bytesCapacity: config.capacityBytes, config: admissionConfig,
            residency: CBv2PagedKVResidency(config: config))
    }

    private func reference(_ row: [Float], values: Bool) throws -> [Float] {
        try PagedKVQuantizationReference.roundTrip(row, config: profile, isKey: !values)
    }

    @Test func assistantHistoryRestoresExactlyBesideLossyTargetRows() throws {
        let stack = try NemotronSuperMTPFixture.stack(kind: .random, depth: 2)
        let assistant = stack.assistant
        let config = PagedKVPoolConfig(
            capacityBytes: 64 << 20, maxPrefillChunk: 64,
            segmentSizeBytes: 32 << 10, layerDTypes: [.float32])
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = try admission(config)
        backend.pool.bindAdmission(admission)
        let codec = CBv2CompleteCheckpointCodec(
            identity: identity, layerKinds: kinds, recurrentSpec: spec, kvDTypes: [.float32],
            assistant: assistant, admission: admission, pagedConfig: config,
            checkpointQuantization: profile)
        #expect(codec.checkpointQuantization == profile)
        #expect(codec.backendLayout == CBv2CompleteCheckpointManifest.nativeQuantizedPagedMTPLayout)
        // Without paged recurrent storage the assistant keeps native storage.
        let contiguous = CBv2CompleteCheckpointCodec(
            identity: identity, layerKinds: kinds, recurrentSpec: spec, kvDTypes: [.float32],
            assistant: assistant, admission: try self.admission(),
            checkpointQuantization: profile)
        #expect(contiguous.checkpointQuantization == nil)
        #expect(contiguous.backendLayout == CBv2CompleteCheckpointManifest.layout)

        let prompt = (0 ... position).map { ($0 * 5 + 3) % 60 + 1 }
        let state = assistant.makeRequestState()
        defer { assistant.releaseRequestState(state) }
        try assistant.configureRequestState(state, maximumSequenceLength: 128)
        assistant.observeCommittedTarget(
            .init(
                tokens: Tests.tokens(Array(prompt.prefix(position))),
                hidden: Tests.rows(position, seed: 71)),
            requestState: state)
        let history = try #require(
            assistant.capturePrefixCheckpoint(requestState: state, targetInputCount: position)
                as? NemotronSuperMTPAssistant.PrefixCheckpoint)
        eval(history.evaluationTargets)

        let request = CBv2Request(
            id: .init(1), promptTokens: prompt, maxTokens: 31, cacheSalt: "tenant")
        let donorID = CBv2RequestID(9001)
        try admission.reserve(id: donorID, additionalTokens: 96)
        let donor = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: 96)
        defer {
            backend.release(donor)
            admission.releaseAll(id: donorID)
        }
        let row = try #require(donor[0] as? PagedSequenceKV)
        let keys = MLXArray((0 ..< position * 64).map { sin(Float($0) * 0.013) })
            .reshaped([1, position, 64])
        let values = MLXArray((0 ..< position * 64).map { cos(Float($0) * 0.007) * 0.5 })
            .reshaped([1, position, 64])
        row.write(keys: keys, values: values)
        eval(backend.pool.group(row.groupKey).writeFence)
        let conv = MLXArray([UInt32(0x8000_0000), 0x7fc0_1234], [1, 2]).view(dtype: .float32)
        let ssm = MLXArray([UInt32(0x3f80_0000), 0xff80_0000], [1, 1, 2]).view(dtype: .float32)
        let exported = try codec.export(
            checkpoint: .init(
                position: position, chunkSize: 32, layers: [0: .init(conv: conv, ssm: ssm)],
                byteCount: 16, assistant: history),
            state: donor, tokens: prompt, cacheSalt: request.cacheSalt)
        defer { exported.close() }
        let manifest = exported.manifest
        #expect(manifest.backendLayout == CBv2CompleteCheckpointManifest.nativeQuantizedPagedMTPLayout)
        #expect(manifest.assistantCodecID == assistant.prefixCheckpointCodecID)
        let assistantDescriptors = try #require(
            assistant.prefixCheckpointTensorDescriptors(targetInputCount: position))
        #expect(
            manifest.tensors
                == Array(manifest.tensors.prefix(4)) + assistantDescriptors)
        #expect(manifest.tensors.prefix(4).map(\.dtype) == [.uint8, .uint8, .float32, .float32])
        _ = try manifest.validateStructure()
        let auxiliaryBytes = try (2 ..< manifest.tensors.count).map {
            try exported.readSegment(tensorIndex: $0, byteOffset: 0, maximumBytes: 4 << 20)
        }

        let plan = try codec.plan(manifest: manifest, request: request)
        #expect(plan.scratchBytes >= CBv2NativeCheckpointRowCodec.scratchBytes)
        let sink = try plan.allocate(onRelease: {})
        defer { sink.close() }
        for (index, descriptor) in manifest.tensors.enumerated() {
            var offset = 0
            while offset < descriptor.byteCount {
                let bytes = try exported.readSegment(
                    tensorIndex: index, byteOffset: offset, maximumBytes: 257)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: bytes)
                offset += bytes.count
            }
        }
        let stage = try sink.finish()
        defer { stage.close() }
        let restored = try stage.consumePreparedState { prepared in
            let frame = try #require(prepared.pagedFrame)
            prepared.pagedFrame = nil
            let adopted = try backend.pool.importCheckpoint(
                frame, admission: admission, requestID: request.id, layerKinds: kinds,
                maximumTokens: plan.maximumSequenceLength)
            return try adopted.moveToActiveRequest { auxiliary in
                #expect(
                    auxiliary.map { $0.asData(access: .copy).data } == auxiliaryBytes)
                let checkpoint = try codec.recurrentCheckpoint(
                    manifest: manifest, auxiliary: auxiliary)
                let decoded = try #require(
                    checkpoint.assistant as? NemotronSuperMTPAssistant.PrefixCheckpoint)
                #expect(decoded.targetInputCount == position)
                Tests.exact(decoded.evaluationTargets, history.evaluationTargets)
            }
        }
        defer {
            backend.release(restored)
            admission.releaseAll(id: request.id)
        }
        let adopted = try #require(restored[0] as? PagedSequenceKV)
        #expect(adopted.groupKey.quantization == nil)
        let snapshot = adopted.gatherRange(start: 0, count: position)
        for (isValue, actual, original) in [
            (false, snapshot.keys, keys), (true, snapshot.values, values),
        ] {
            let result = actual.asArray(Float.self)
            let source = original.asArray(Float.self)
            for token in 0 ..< position {
                let row = Array(source[token * 64 ..< (token + 1) * 64])
                let got = Array(result[token * 64 ..< (token + 1) * 64])
                if token >= position - profile.recentTokenCount {
                    #expect(got.map(\.bitPattern) == row.map(\.bitPattern), "recent row \(token)")
                } else {
                    let expected = try reference(row, values: isValue)
                    let error = zip(got, expected).map { abs($0 - $1) }.max()!
                    #expect(error <= 0.00001, "row \(token) values \(isValue)")
                }
            }
        }
        #expect(!backend.usesQuantizedStorage && backend.supportsOrdinaryDecodeChaining)
    }
}
