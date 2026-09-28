import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXLLM
@testable import MLXVLM

@Suite("Native diffusion portable committed prefix", .serialized)
struct DiffusionGemmaPersistentPrefixTests {
    private let diskIdentity = CBv2CompleteCheckpointIdentity(modelAggregateHash: String(repeating: "a", count: 64),
        promptContractID: String(repeating: "b", count: 64), buildID: String(repeating: "c", count: 64), numericsFingerprint: String(repeating: "d", count: 64))
    private func prefixIdentity(epoch: String) throws -> DiffusionGemmaPrefixIdentity {
        try .init(tenantScope: "fixture-tenant", artifact: diskIdentity.modelAggregateHash,
                  template: diskIdentity.promptContractID, media: "text-only",
                  numericalProfile: diskIdentity.numericsFingerprint, epoch: epoch)
    }
    private func exact(_ actual: MLXArray, _ expected: MLXArray) {
        eval(actual, expected)
        #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
        #expect(actual.asArray(Float.self).map(\.bitPattern) == expected.asArray(Float.self).map(\.bitPattern))
    }

    @Test func reloadRestoresEveryTensorAndVirtualRingOrderWithoutForwardReplay() async throws {
        let (directory, _) = try DiffusionGemmaFactoryTests().fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let donor = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: DiffusionGemmaFactoryTests.Loader(fail: false))
        let recipient = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: DiffusionGemmaFactoryTests.Loader(fail: false))
        let engine = try recipient.makeNativeEngine(kvBytesCapacity: 128 << 20)
        for chunks in [[4, 4], Array(repeating: 1, count: 10), [3, 3]] {
            try roundTrip(chunks: chunks, donor: donor, recipient: recipient, engine: engine)
            #expect(engine.capacity().kvBytesReserved == 0, "Every export/import/manifest owner must retire")
        }
        await engine.shutdown()
    }

    private func roundTrip(chunks: [Int], donor: DiffusionGemmaContext, recipient: DiffusionGemmaContext,
                           engine: CBv2NativeBlockEngine) throws {
        let count = chunks.reduce(0, +)
        let tokens = (0..<count).map { Int32($0 + 2) }
        let original = try donor.model.makeCache(expectedPromptLength: count + 1)
        var offset = 0
        for chunk in chunks {
            _ = try donor.model.encode(tokenIds: MLXArray(Array(tokens[offset..<offset + chunk])).reshaped(1, chunk), cache: original)
            eval(original.stateArrays()); offset += chunk
        }
        let oldIdentity = try prefixIdentity(epoch: "old-load")
        let newIdentity = try prefixIdentity(epoch: "new-load")
        let checkpoint = try donor.model.model.decoder.checkpoint(cache: original, identity: oldIdentity, compact: true)
        let oldCodec = try donor.model.model.decoder.makePersistentPrefixCodec(verifiedIdentity: diskIdentity, kvDType: .float32)
        let newCodec = try recipient.model.model.decoder.makePersistentPrefixCodec(verifiedIdentity: diskIdentity, kvDType: .float32)
        let source = try oldCodec.export(checkpoint, chunkSize: chunks[0], engine: engine)
        defer { source.close() }
        let decoded = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: JSONEncoder().encode(source.manifest))
        let prompt = tokens.map(Int.init) + [37]
        #expect(throws: DiffusionGemmaModelError.self) {
            try recipient.model.model.decoder.restorePrefix(checkpoint, identity: oldIdentity, promptTokenIds: MLXArray(prompt).asType(.int32).reshaped(1, prompt.count))
        }
        let plan = try newCodec.importPlan(decoded, prefixIdentity: newIdentity, promptTokens: prompt,
                                           chunkSize: chunks[0], engine: engine)
        let importer = try plan.allocate()
        defer { importer.close() }
        #expect(throws: CBv2CompleteCheckpointError.incompleteTransfer) { try importer.finish() }
        for (index, descriptor) in decoded.tensors.enumerated() {
            var offset = 0
            while offset < descriptor.byteCount {
                let bytes = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: 28)
                try importer.appendSegment(tensorIndex: index, byteOffset: offset, data: bytes)
                offset += bytes.count
            }
        }
        let staged = try importer.finish()
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try oldCodec.adopt(staged, prefixIdentity: oldIdentity)
        }
        var imported: DiffusionGemmaPrefixCheckpoint? = try newCodec.adopt(staged, prefixIdentity: newIdentity)
        #expect(throws: CBv2CompleteCheckpointError.closed) { try newCodec.adopt(staged, prefixIdentity: newIdentity) }
        let restored = try recipient.model.model.decoder.restorePrefix(imported!, identity: newIdentity,
            promptTokenIds: MLXArray(prompt).asType(.int32).reshaped(1, prompt.count))
        imported = nil
        #expect(engine.capacity().kvBytesReserved >= plan.nativeDestinationBytes,
            "Request cache must keep destination permit alive after its public checkpoint is dropped")
        #expect(restored.windowOrder == original.windowOrder)
        for (lhs, rhs) in zip(restored.snapshots(), original.snapshots()) {
            exact(lhs.keys, rhs.keys); exact(lhs.values, rhs.values)
        }
        let next = MLXArray([Int32(37)]).reshaped(1, 1)
        exact(try recipient.model.encode(tokenIds: next, cache: restored), try donor.model.encode(tokenIds: next, cache: original))
        let canvas = MLXArray([Int32(2), 3, 5, 7]).reshaped(1, 4)
        exact(try recipient.model.denoise(canvasIds: canvas, cache: restored), try donor.model.denoise(canvasIds: canvas, cache: original))
    }

    @Test func identityGeometryAndCapacityRefuseBeforeImportAndCloseRefunds() async throws {
        let (directory, _) = try DiffusionGemmaFactoryTests().fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: DiffusionGemmaFactoryTests.Loader(fail: false))
        let engine = try context.makeNativeEngine(kvBytesCapacity: 128 << 20)
        try validateRefusals(context: context, engine: engine)
        #expect(engine.capacity().kvBytesReserved == 0)
        let reservation = try engine.reserveNativeCheckpoint(bytes: 120 << 20)
        #expect(throws: CBv2KVError.self) { try engine.reserveNativeCheckpoint(bytes: 16 << 20) }
        reservation.close(); reservation.close()
        #expect(engine.capacity().kvBytesReserved == 0)
        await engine.shutdown()
        #expect(throws: CBv2NativeBlockError.shuttingDown) { try engine.reserveNativeCheckpoint(bytes: 1) }
    }

    private func validateRefusals(context: DiffusionGemmaContext, engine: CBv2NativeBlockEngine) throws {
        let identity = try prefixIdentity(epoch: "current")
        let cache = try context.model.makeCache(expectedPromptLength: 4)
        _ = try context.model.encode(tokenIds: MLXArray([Int32(2), 3, 5, 7]).reshaped(1, 4), cache: cache)
        let checkpoint = try context.model.model.decoder.checkpoint(cache: cache, identity: identity, compact: true)
        let codec = try context.model.model.decoder.makePersistentPrefixCodec(verifiedIdentity: diskIdentity, kvDType: .float32)
        let source = try codec.export(checkpoint, chunkSize: 4, engine: engine)
        defer { source.close() }
        let baseline = engine.capacity().kvBytesReserved
        let root = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(source.manifest)) as? [String: Any])
        for field in ["modelAggregateHash", "promptContractID", "buildID", "numericsFingerprint", "scope", "window"] {
            var changed = root
            if field == "scope" { changed["cacheSalt"] = "other-tenant" }
            else if field == "window" { changed["nativeBlockState"] = ["windowPhysicalLength": 100, "windowCursor": 1] }
            else {
                var identity = try #require(root["identity"] as? [String: Any]); identity[field] = String(repeating: "e", count: 64)
                changed["identity"] = identity
            }
            let manifest = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: JSONSerialization.data(withJSONObject: changed))
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try codec.importPlan(manifest, prefixIdentity: identity, promptTokens: [2, 3, 5, 7, 11], chunkSize: 4, engine: engine)
            }
            #expect(engine.capacity().kvBytesReserved == baseline)
        }
        let plan = try codec.importPlan(source.manifest, prefixIdentity: identity, promptTokens: [2, 3, 5, 7, 11], chunkSize: 4, engine: engine)
        let before = engine.capacity().kvBytesReserved
        let importer = try plan.allocate()
        #expect(engine.capacity().kvBytesReserved > before)
        #expect(throws: CBv2CompleteCheckpointError.invalidSegment) {
            try importer.appendSegment(tensorIndex: 1, byteOffset: 0, data: Data([0, 0, 0, 0]))
        }
        importer.close()
        #expect(engine.capacity().kvBytesReserved == before)
    }
}
