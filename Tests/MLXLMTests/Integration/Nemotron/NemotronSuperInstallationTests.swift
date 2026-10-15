import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

struct NemotronSuperInstallationTests {
    private struct Tensor {
        let dtype: String
        let shape: [Int]
        let data: Data
    }
    private func model() throws -> NemotronSuperModel {
        let language = NemotronHModel(
            NemotronHConfiguration(
                vocabSize: 32, hiddenSize: 16,
                numHiddenLayers: 3, numAttentionHeads: 1, numKeyValueHeads: 1,
                mambaNumHeads: 1, mambaHeadDim: 16, ssmStateSize: 4, convKernel: 4, nGroups: 1,
                intermediateSize: 16, moeIntermediateSize: 16, moeSharedExpertIntermediateSize: 16,
                nRoutedExperts: 2, numExpertsPerTok: 1, hybridOverridePattern: "M*E", headDim: 16))
        let tower = NemotronRadio(
            NemotronRadioDimensions(
                width: 2, layers: 1, heads: 1,
                channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3))
        let images = NemotronSuperImageComposition(
            tower: tower,
            projector: NemotronSuperVisionProjector(
                visionWidth: 2, hiddenWidth: 3, outputWidth: 16),
            width: 16, imageTokenID: 18)
        return try NemotronSuperModel(languageModel: language, imageComposition: images)
    }
    private func scalar(_ value: Float) -> Tensor {
        var bits = value.bitPattern.littleEndian
        return Tensor(dtype: "F32", shape: [], data: withUnsafeBytes(of: &bits) { Data($0) })
    }
    private func tensors() throws -> [String: Tensor] {
        let candidate = try model()
        var result: [String: Tensor] = [:]
        for (name, value) in candidate.parameters().flattened() {
            let source = name.replacingOccurrences(of: "image_composition.", with: "")
            let data = value.asData()
            result[source] = Tensor(dtype: "F32", shape: data.shape, data: data.data)
        }
        for prefix in ["language_model.backbone.layers.0.mixer.in_proj", "language_model.lm_head"] {
            let old = result[prefix + ".weight"]!
            result[prefix + ".weight"] = Tensor(
                dtype: "F8_E4M3", shape: old.shape,
                data: Data(repeating: 0, count: old.shape.reduce(1, *)))
            result[prefix + ".input_scale"] = scalar(1)
            result[prefix + ".weight_scale"] = scalar(1)
        }
        let root = "language_model.backbone.layers.2.mixer"
        for (native, raw) in [("fc1", "up_proj"), ("fc2", "down_proj")] {
            let old = result.removeValue(forKey: root + ".switch_mlp." + native + ".weight")!
            for expert in 0 ..< old.shape[0] {
                let p = root + ".experts.\(expert)." + raw
                result[p + ".weight"] = Tensor(
                    dtype: "U8", shape: [16, 8], data: Data(repeating: 0, count: 128))
                result[p + ".weight_scale"] = Tensor(
                    dtype: "F8_E4M3", shape: [16, 1], data: Data(repeating: 56, count: 16))
                result[p + ".weight_scale_2"] = scalar(1)
                result[p + ".input_scale"] = scalar(1)
            }
        }
        // Retained MTP is accounted separately; it is never executed or dropped invisibly.
        result["language_model.mtp.fixture.weight"] = scalar(1)
        result["vision_model.summary_idxs"] = Tensor(
            dtype: "I64", shape: [2], data: Data(repeating: 0, count: 16))
        return result
    }
    private func withSource(
        _ tensors: [String: Tensor], _ body: (ModelOptCheckpointSource) throws -> Void
    ) throws {
        var files: [String: ModelOptTensorFile] = [:]
        var map: [String: String] = [:]
        var urls: [URL] = []
        defer { for url in urls { try? FileManager.default.removeItem(at: url) } }
        // Split consecutive tensor/scale keys across shards, deliberately.
        let keys = tensors.keys.sorted()
        for shard in 0 ..< 2 {
            let name = "fixture-\(shard).safetensors"
            var header: [String: Any] = [:]
            var bytes = Data()
            for (i, key) in keys.enumerated() where i % 2 == shard {
                let tensor = tensors[key]!
                let low = bytes.count
                bytes.append(tensor.data)
                header[key] = [
                    "dtype": tensor.dtype, "shape": tensor.shape,
                    "data_offsets": [low, bytes.count],
                ]
                map[key] = name
            }
            let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            var count = UInt64(json.count).littleEndian
            var data = withUnsafeBytes(of: &count) { Data($0) }
            data.append(json)
            data.append(bytes)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            urls.append(url)
            try data.write(to: url)
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            files[name] = try ModelOptTensorFile(url: url, sha256: hash)
        }
        let index = try JSONSerialization.data(
            withJSONObject: ["weight_map": map], options: [.sortedKeys])
        let hash = SHA256.hash(data: index).map { String(format: "%02x", $0) }.joined()
        try body(ModelOptCheckpointSource(indexData: index, indexSHA256: hash, files: files))
    }
    private func forward(_ model: NemotronSuperModel, image: Bool) throws -> MLXArray {
        let prompt: [Int32] = image ? [1, 18, 2] : [1, 2, 3]
        let prepared =
            image
            ? try model.prepare(prompt: prompt, normalizedImages: [MLXArray.ones([1, 1, 4, 4])])
            : try model.prepare(prompt: prompt)
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 20, kvDType: .float32))
        let bank = CBv2LayerCacheBank(
            caches: model.languageModel.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let rows = try backend.makeSequenceState(
            layerKinds: model.cbv2LayerKinds, promptLength: 3, maxLength: 8)
        defer { backend.release(rows) }
        let caches = try bank.layerCaches(rowStates: [rows]).map { try #require($0 as? KVCache) }
        let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        let binding = try state.bind()
        let output = try model.prefill(
            prepared, cache: caches, recurrentState: [binding], requirement: .lastPositionLogits)
        eval([output] + (try binding.evaluate()))
        try binding.commit()
        let next = try state.bind()
        let decoded = try model.decode(token: 4, cache: caches, recurrentState: [next])
        eval([decoded] + (try next.evaluate()))
        try next.commit()
        #expect(isFinite(output).all().item(Bool.self))
        #expect(isFinite(decoded).all().item(Bool.self))
        return decoded
    }
    @Test func completeMixedCheckpointLoadsTextAndImageAndRetires() throws {
        try withSource(tensors()) { source in
            let quote = try NemotronSuperInstallation.requiredLoadBytes(
                source: source, backendAllowanceBytes: 1 << 20)
            let ledger = try NemotronSuperResidency(capacityBytes: quote, maximumRequests: 2)
            let scope = NativeConstructionScope()
            let installed = try NemotronSuperInstallation.stage(
                source: source, residency: ledger, dtype: .float32,
                backendAllowanceBytes: 1 << 20, construction: scope, makeModel: model)
            #expect(scope.snapshot.epoch > 2)
            #expect(
                scope.snapshot.retainedArrayCount == 0 && scope.snapshot.retainedOwnerCount == 0)
            #expect(installed.receipt.retainedMTPCount == 1)
            #expect(installed.receipt.loadedTensorCount == source.tensors.count - 2)
            #expect(ledger.snapshot.stagingBytes == 0)
            #expect(ledger.snapshot.weightBytes == installed.receipt.payloadBytes)
            let value = try #require(installed.model)
            // Graph path (default): the largest chunk whose ModelOpt workspace
            // fits the admitted budget; this tiny fixture reaches the cap. The
            // eager reference keeps its 64 MiB, 32-token bound.
            let language = value.languageModel
            let limit = try #require(language.calibratedPrefillTokenLimit)
            #expect(limit == NemotronHModel.modelOptGraphPrefillTokenCap)
            #expect(
                try #require(language.modelOptGraphPrefillWorkspaceBytes(tokens: limit))
                    <= NemotronHModel.modelOptGraphPrefillWorkspaceLimit)
            language.setModelOptExecution(.eagerReference)
            #expect(language.calibratedPrefillTokenLimit == 32)
            language.setModelOptExecution(.graph)
            #expect(try forward(value, image: false).dim(-1) == 32)
            #expect(try forward(value, image: true).dim(-1) == 32)
            installed.closeAfterRetirement()
            #expect(installed.model == nil && ledger.snapshot.usedBytes == 0)
        }
    }
    /// The real checkpoint stores A_log as BF16; the model keeps it in float32.
    @Test func bf16ALogWidensExactlyAndOtherDTypeMismatchesFail() throws {
        let aLog = "language_model.backbone.layers.0.mixer.A_log"
        let bf16 = Tensor(dtype: "BF16", shape: [1], data: Data([0x31, 0x3F]))  // 0.69140625
        var widened = try tensors()
        widened[aLog] = bf16
        try withSource(widened) { source in
            let ledger = try NemotronSuperResidency(capacityBytes: 16 << 20, maximumRequests: 1)
            let installed = try NemotronSuperInstallation.stage(
                source: source, residency: ledger, dtype: .float32,
                backendAllowanceBytes: 1 << 20, construction: NativeConstructionScope(),
                makeModel: model)
            let value = try #require(
                installed.model?.languageModel.parameters().flattened().first {
                    $0.0 == "backbone.layers.0.mixer.A_log"
                }?.1)
            #expect(value.dtype == .float32)
            #expect(value.asArray(Float.self) == [0.69140625])
            installed.closeAfterRetirement()
        }
        var mismatched = try tensors()
        mismatched["language_model.backbone.layers.0.mixer.D"] = bf16
        try withSource(mismatched) { source in
            let ledger = try NemotronSuperResidency(capacityBytes: 16 << 20, maximumRequests: 1)
            #expect(throws: (any Error).self) {
                try NemotronSuperInstallation.stage(
                    source: source, residency: ledger, dtype: .float32,
                    backendAllowanceBytes: 1 << 20, construction: NativeConstructionScope(),
                    makeModel: model)
            }
            #expect(ledger.snapshot.usedBytes == 0)
        }
    }
    @Test func failureDoesNotPublishOrLeakOrdinaryAdmission() throws {
        var missing = try tensors()
        missing.removeValue(forKey: "language_model.backbone.embeddings.weight")
        try withSource(missing) { source in
            let ledger = try NemotronSuperResidency(capacityBytes: 16 << 20, maximumRequests: 1)
            #expect(throws: (any Error).self) {
                try NemotronSuperInstallation.stage(
                    source: source, residency: ledger, dtype: .float32,
                    backendAllowanceBytes: 1 << 20, construction: NativeConstructionScope(),
                    makeModel: model)
            }
            #expect(ledger.snapshot.usedBytes == 0)
        }
        try withSource(tensors()) { source in
            let ledger = try NemotronSuperResidency(capacityBytes: 1, maximumRequests: 1)
            var built = false
            #expect(throws: (any Error).self) {
                try NemotronSuperInstallation.stage(
                    source: source, residency: ledger, dtype: .float32,
                    backendAllowanceBytes: 1, construction: NativeConstructionScope()
                ) {
                    built = true
                    return try model()
                }
            }
            #expect(!built && ledger.snapshot.usedBytes == 0)
        }
    }
    @Test func failedNativeFenceRetainsOwnersAndCharges() throws {
        try withSource(tensors()) { source in
            let ledger = try NemotronSuperResidency(capacityBytes: 16 << 20, maximumRequests: 1)
            let scope = NativeConstructionScope()
            scope.testingBeforeFence = { _ in throw NemotronSuperInstallation.Failure.cancelled }
            #expect(throws: (any Error).self) {
                try NemotronSuperInstallation.stage(
                    source: source, residency: ledger, dtype: .float32,
                    backendAllowanceBytes: 1 << 20, construction: scope, makeModel: model)
            }
            #expect(scope.snapshot.isRetainedFault)
            #expect(
                scope.snapshot.retainedOwnerCount > 0 && ledger.snapshot.weightBytes > 0
                    && ledger.snapshot.stagingBytes > 0)
        }
    }
    @Test func nestedConstructionCannotRefundAnUnfencedEpoch() throws {
        try withSource(tensors()) { source in
            let ledger = try NemotronSuperResidency(capacityBytes: 16 << 20, maximumRequests: 1)
            let scope = NativeConstructionScope()
            try scope.withPhase(.modelFactory) {
                #expect(throws: (any Error).self) {
                    try NemotronSuperInstallation.stage(
                        source: source, residency: ledger, dtype: .float32,
                        backendAllowanceBytes: 1 << 20, construction: scope, makeModel: model)
                }
                #expect(ledger.snapshot.usedBytes == 0)
            }
        }
    }

    @Test func laterEpochFenceFailureKeepsEarlierResidentOutputsAndFullCharges() throws {
        try withSource(tensors()) { source in
            let quote = try NemotronSuperInstallation.requiredLoadBytes(
                source: source, backendAllowanceBytes: 1 << 20)
            let ledger = try NemotronSuperResidency(capacityBytes: quote, maximumRequests: 1)
            let scope = NativeConstructionScope()
            var reached = 0
            #expect(throws: (any Error).self) {
                try NemotronSuperInstallation.stage(
                    source: source, residency: ledger, dtype: .float32,
                    backendAllowanceBytes: 1 << 20, construction: scope,
                    testingBeforeEpoch: { number, scope in
                        reached = number
                        if number == 3 {
                            #expect(scope.snapshot.retainedArrayCount == 0)
                            #expect(ledger.snapshot.usedBytes == quote)
                            scope.testingBeforeFence = { _ in
                                throw NemotronSuperInstallation.Failure.cancelled
                            }
                        }
                    }, makeModel: model)
            }
            #expect(reached == 3 && scope.snapshot.isRetainedFault)
            #expect(scope.snapshot.retainedOwnerCount > 0)
            #expect(ledger.snapshot.usedBytes == quote)
        }
    }

}
