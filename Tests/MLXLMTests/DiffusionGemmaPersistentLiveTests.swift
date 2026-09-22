import CryptoKit
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

private final class DiffusionPersistentTestAnchor: NSObject {}

/// Full-weight transfer/reload numerical gate. This does not claim encrypted
/// filesystem integration, paging, HTTP restore or a throughput benchmark.
@Suite("DiffusionGemma full artifact portable state", .serialized)
struct DiffusionGemmaPersistentLiveTests {
    private struct Loader: TokenizerLoader {
        let tokenizer: any MLXLMCommon.Tokenizer
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { tokenizer }
    }
    private final class Witness { weak var model: DiffusionGemma? }
    private func sha(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    private func bits(_ create: @autoclosure () throws -> MLXArray) throws -> Data {
        try MLX.withError { errors in
            let array = try create()
            try errors.check()
            eval(array)
            try errors.check()
            return array.asArray(Float.self).map(\.bitPattern).withUnsafeBytes { Data($0) }
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_PERSISTENT_LIVE"] == "1"))
    func selectedQuantReloadPreservesSingletonAndDenoiserBits() async throws {
        let selectedPath = ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_MODEL_DIR"]
        let directory = URL(fileURLWithPath: try #require(selectedPath))
        let config = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        try #require(sha(config) == "b41320c97651075363f2895e2cbb3d1580670ee11edb653a14290a35bbf7cac5")
        let artifactHash = sha(try Data(contentsOf: directory.appendingPathComponent("local-verified-manifest.json")))
        try #require(artifactHash == "33eb488387819e31d1a848e7d0f20465ae6ad0ae1ed31dac7aa4cb5e0461d53d")
        // Register the source-bound test resource bundle before Metal use.
        _ = try #require(Bundle.module.url(forResource: "diffusiongemma-text-config", withExtension: "json"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let loader = Loader(tokenizer: #adaptHuggingFaceTokenizer(tokenizer))
        let executable = try #require(Bundle(for: DiffusionPersistentTestAnchor.self).executableURL)
        let buildHash = sha(try Data(contentsOf: executable))
        let resources = try #require(Bundle(for: DiffusionPersistentTestAnchor.self).resourceURL)
        let metal = resources.appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        let metalHash = sha(try Data(contentsOf: metal))
        try #require(metalHash == "38ceb8a1113b373ccaa89355d895fc8ded3bafc81076dc4ffdb426e2eee03a93")
        let nativeIdentity = CBv2CompleteCheckpointIdentity(modelAggregateHash: artifactHash,
            promptContractID: sha(try Data(contentsOf: directory.appendingPathComponent("chat_template.jinja"))),
            buildID: buildHash,
            numericsFingerprint: sha(Data((buildHash + ProcessInfo.processInfo.operatingSystemVersionString + metalHash + "canonical-strict-metal").utf8)))
        func identity(_ epoch: String) throws -> DiffusionGemmaPrefixIdentity {
            try .init(tenantScope: "native-portable-test", artifact: nativeIdentity.modelAggregateHash,
                      template: nativeIdentity.promptContractID, media: "text-only",
                      numericalProfile: nativeIdentity.numericsFingerprint, epoch: epoch)
        }
        let witness = Witness()
        let prompt = Array(repeating: [2, 3, 5, 7], count: 256).flatMap { $0 }
        let donor = try await makeDonor(directory: directory, loader: loader, identity: identity("before"),
                                        disk: nativeIdentity, prompt: prompt, witness: witness)
        defer { donor.source.close(); donor.permit.close(); withExtendedLifetime(donor.engine) {} }
        try #require(witness.model == nil, "Export and codec must not retain old model weights")
        Memory.clearCache()
        let start = ContinuousClock.now
        let context = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: loader)
        let loadSeconds = duration(start.duration(to: .now))
        let engine = try context.makeNativeEngine(kvBytesCapacity: 8 << 30)
        let manifest = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: JSONEncoder().encode(donor.source.manifest))
        let codec = try context.model.model.decoder.makePersistentPrefixCodec(verifiedIdentity: nativeIdentity, kvDType: .bfloat16)
        try transferAndCompare(context: context, codec: codec, engine: engine, manifest: manifest,
                               prefixIdentity: identity("after"), source: donor.source, prompt: prompt,
                               expectedEncoder: donor.encoder, expectedLogits: donor.logits)
        #expect(engine.capacity().kvBytesReserved == 0)
        await engine.shutdown()
        let row: [String: Any] = ["position": manifest.position, "layers": manifest.tensors.count / 2,
            "dtype": "bfloat16", "exactEncoder": true, "exactDenoiser": true, "oldModelReleased": witness.model == nil,
            "reloadSeconds": loadSeconds, "mlxActiveBytes": Memory.activeMemory, "mlxCacheBytes": Memory.cacheMemory,
            "scope": "full quant bounded portable transfer/reload; not encrypted persistence/HTTP/paging or benchmark"]
        print("DIFFUSION_PORTABLE_LIVE " + String(decoding: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), as: UTF8.self))
    }

    private func duration(_ value: Duration) -> Double {
        Double(value.components.seconds) + Double(value.components.attoseconds) / 1e18
    }

    private func makeDonor(directory: URL, loader: Loader, identity: DiffusionGemmaPrefixIdentity,
                           disk: CBv2CompleteCheckpointIdentity, prompt: [Int], witness: Witness) async throws
        -> (source: CBv2CompleteCheckpointExport, permit: CBv2NativeBlockCheckpointLease, engine: CBv2NativeBlockEngine, encoder: Data, logits: Data) {
        let context = try await DiffusionGemmaModelFactory.shared.load(from: directory, using: loader)
        witness.model = context.model
        let cache = try context.model.makeCache(expectedPromptLength: prompt.count + 1)
        for start in stride(from: 0, to: prompt.count, by: 512) {
            try MLX.withError { errors in
                _ = try context.model.encode(tokenIds: MLXArray(Array(prompt[start..<start + 512])).asType(.int32).reshaped(1, 512), cache: cache)
                try errors.check()
                eval(cache.stateArrays())
                try errors.check()
            }
        }
        try #require(cache.snapshots().allSatisfy { $0.keys.dtype == .bfloat16 && $0.values.dtype == .bfloat16 })
        let engine = try context.makeNativeEngine(kvBytesCapacity: 8 << 30)
        // Standalone capture owner, reserved before its compact copies and
        // boolean operands. No provider process-budget qualification here.
        let booleanBytes = try Memory.allocationFootprintUpperBound(byteCount: 1)
        let captureBytes = try cache.stateArrays().reduce(0) { try $0 + Memory.allocationFootprintUpperBound(byteCount: $1.nbytes) + booleanBytes }
        let permit = try engine.reserveNativeCheckpoint(bytes: captureBytes)
        let checkpoint = try context.model.model.decoder.checkpoint(cache: cache, identity: identity, compact: true)
        let codec = try context.model.model.decoder.makePersistentPrefixCodec(verifiedIdentity: disk, kvDType: .bfloat16)
        let source = try codec.export(checkpoint, chunkSize: 512, engine: engine)
        let encoder = try bits(try context.model.encode(tokenIds: MLXArray([Int32(37)]).reshaped(1, 1), cache: cache))
        let logits = try bits(try context.model.denoise(canvasIds: MLXArray([Int32(2), 3, 5, 7]).reshaped(1, 4), cache: cache))
        await engine.shutdown()
        return (source, permit, engine, encoder, logits)
    }

    private func transferAndCompare(context: DiffusionGemmaContext, codec: DiffusionGemmaPersistentPrefixCodec,
                                    engine: CBv2NativeBlockEngine, manifest: CBv2CompleteCheckpointManifest,
                                    prefixIdentity: DiffusionGemmaPrefixIdentity, source: CBv2CompleteCheckpointExport,
                                    prompt: [Int], expectedEncoder: Data, expectedLogits: Data) throws {
        let plan = try codec.importPlan(manifest, prefixIdentity: prefixIdentity, promptTokens: prompt + [37], chunkSize: 512, engine: engine)
        let importer = try plan.allocate()
        defer { importer.close() }
        for (index, tensor) in manifest.tensors.enumerated() {
            var offset = 0
            while offset < tensor.byteCount {
                let bytes = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: CBv2CompleteCheckpointManifest.maximumSegmentBytes)
                try importer.appendSegment(tensorIndex: index, byteOffset: offset, data: bytes)
                offset += bytes.count
            }
        }
        let checkpoint = try codec.adopt(importer.finish(), prefixIdentity: prefixIdentity)
        let cache = try context.model.model.decoder.restorePrefix(checkpoint, identity: prefixIdentity,
            promptTokenIds: MLXArray(prompt + [37]).asType(.int32).reshaped(1, prompt.count + 1))
        try #require(try bits(try context.model.encode(tokenIds: MLXArray([Int32(37)]).reshaped(1, 1), cache: cache)) == expectedEncoder)
        try #require(try bits(try context.model.denoise(canvasIds: MLXArray([Int32(2), 3, 5, 7]).reshaped(1, 4), cache: cache)) == expectedLogits)
    }
}
