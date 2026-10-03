import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Audio input codec composition with the internal tiny codec geometry
/// (`MiMoV26AudioInputConfiguration.fixture()`: hidden 8, two heads, four
/// layers, two mel bands, two codebooks of four) and deterministic synthetic
/// weights. It also checks the metadata-only preflight of the selected
/// sidecar against the shipped MiMo configuration files, with descriptors
/// that the test builds. No real weights are read.
final class MiMoV26TinyAudioInputTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint
    private final class Log {
        var acquired = 0
        var released = 0
        var failed: MiMoV26FailedAudioWork?
    }
    private final class Owner: MiMoV26AudioWorkReservation {
        let log: Log
        let plan: MiMoV26AudioInputPlan
        let sourceIdentity: String
        let generation: UUID
        init(_ log: Log, plan: MiMoV26AudioInputPlan, weights: MiMoV26AudioInputWeights) {
            self.log = log
            self.plan = plan
            sourceIdentity = weights.sourceIdentity
            generation = weights.generation
            log.acquired += 1
        }
        func validate(plan: MiMoV26AudioInputPlan, sourceIdentity: String, generation: UUID)
            throws
        {
            guard plan == self.plan, sourceIdentity == self.sourceIdentity,
                generation == self.generation
            else { throw MiMoV26AudioInputError.input("test reservation binding") }
        }
        func retainAfterFailedDrain(_ work: MiMoV26FailedAudioWork) { log.failed = work }
        deinit { log.released += 1 }
    }

    private func sourceWeights(_ c: MiMoV26AudioInputConfiguration) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (name, shape) in MiMoV26AudioTokenizerWeights.inputSourceShapes(configuration: c) {
            let array = MLXArray(Fixture.floats(name, count: shape.reduce(1, *)), shape)
            result[name] = name.hasPrefix("encoder.quantizer.") ? array : array.asType(.bfloat16)
        }
        return result
    }
    private func bundle() throws -> MiMoV26AudioInputWeights {
        let c = try MiMoV26AudioInputConfiguration.fixture()
        return try MiMoV26AudioTokenizerWeights.fixtureBundle(
            configuration: c, weights: sourceWeights(c))
    }
    private func clip(_ name: String, frames: Int) throws -> MiMoV26DecodedPCM {
        try .init(
            samples: (0 ..< frames).map { 0.25 * Float(sin(Double($0) * 0.1)) },
            descriptor: .init(
                sourceIdentity: name, channels: 1, frameCount: frames, sampleRate: 24000))
    }

    // MARK: - Tiny codec

    func testEncodeComposesClipsInOrderAndRetiresTheWorkOwner() throws {
        let weights = try bundle()
        XCTAssertEqual(weights.sourceIdentity, "synthetic-hf5711-fixture")
        XCTAssertEqual(weights.generation, weights.encoder.loadedGeneration)
        XCTAssertFalse(weights.materializationRoots.isEmpty)
        let input = MiMoV26AudioInput(weights: weights)
        XCTAssertEqual(input.configuration, weights.configuration)
        let log = Log()
        let clips = [try clip("first", frames: 481), try clip("second", frames: 1200)]
        var admitted: MiMoV26AudioInputPlan?
        let codes = try input.encode(
            clips: clips, limits: mimoAudioInputTestLimits(), retaining: input,
            authorize: { plan in
                admitted = plan
                return Owner(log, plan: plan, weights: weights)
            })
        let plan = try XCTUnwrap(admitted)
        XCTAssertEqual(plan.sourceIdentities, ["first", "second"])
        XCTAssertEqual(codes.map(\.frameCount), plan.codeFrameCounts)
        XCTAssertEqual(codes.count, 2)
        for code in codes {
            XCTAssertGreaterThan(code.frameCount, 0)
            XCTAssertEqual(code.codes.count, code.frameCount * 2)
            XCTAssertTrue(code.codes.allSatisfy { $0 >= 0 && $0 < 4 })
        }
        XCTAssertEqual(log.acquired, 1)
        XCTAssertEqual(log.released, 1)
        // Integer codes of the same input and weights are repeatable.
        let again = try input.encode(
            clips: clips, limits: mimoAudioInputTestLimits(), retaining: input,
            authorize: { Owner(log, plan: $0, weights: weights) })
        XCTAssertEqual(again.map(\.codes), codes.map(\.codes))
        XCTAssertEqual(log.released, 2)
    }

    func testAuthorizationAndOwnerRefusalsStopBeforeCodes() throws {
        let weights = try bundle()
        let input = MiMoV26AudioInput(weights: weights)
        let clips = [try clip("denied", frames: 481)]
        XCTAssertThrowsError(
            try input.encode(
                clips: clips, limits: mimoAudioInputTestLimits(), retaining: input,
                authorize: { _ in throw MiMoV26AudioInputError.limit("intentional refusal") })
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .limit("intentional refusal")) }
        let log = Log()
        let otherPlan = try MiMoV26AudioInputPlan.make(
            clips: [try clip("other", frames: 960)].map(\.descriptor),
            configuration: weights.configuration, limits: mimoAudioInputTestLimits())
        XCTAssertThrowsError(
            try input.encode(
                clips: clips, limits: mimoAudioInputTestLimits(), retaining: input,
                authorize: { _ in Owner(log, plan: otherPlan, weights: weights) })
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .input("test reservation binding")) }
        XCTAssertEqual(log.released, 1)
        XCTAssertThrowsError(
            try input.encode(
                clips: clips, limits: mimoAudioInputTestLimits(), retaining: input,
                authorize: { Owner(log, plan: $0, weights: weights) }, isCancelled: { true })
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .cancelled) }
        XCTAssertEqual(log.acquired, 1)
    }

    func testCancellationAfterTheFrontendDrainsAndAllowsRetry() throws {
        let input = MiMoV26AudioInput(weights: try bundle())
        let clips = [try clip("cancel", frames: 1200)]
        let log = Log()
        var polls = 0
        XCTAssertThrowsError(
            try input.encode(
                clips: clips, limits: mimoAudioInputTestLimits(), retaining: input,
                authorize: { Owner(log, plan: $0, weights: input.weights) },
                isCancelled: {
                    polls += 1
                    return polls > 8
                })
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .cancelled) }
        XCTAssertEqual(log.acquired, 1)
        XCTAssertEqual(log.released, 1)
        let retry = try input.encode(
            clips: clips, limits: mimoAudioInputTestLimits(), retaining: input,
            authorize: { Owner(log, plan: $0, weights: input.weights) })
        XCTAssertEqual(retry.count, 1)
        XCTAssertEqual(log.acquired, 2)
        XCTAssertEqual(log.released, 2)
    }

    func testNetworkReloadInvalidatesComposedWeights() throws {
        let weights = try bundle()
        let input = MiMoV26AudioInput(weights: weights)
        let old = weights.generation
        try weights.encoder.loadNativeNetworkWeights(
            sourceWeights(weights.configuration).filter { !$0.key.hasPrefix("encoder.quantizer.") })
        XCTAssertNotEqual(weights.encoder.loadedGeneration, old)
        var authorizations = 0
        XCTAssertThrowsError(
            try input.encode(
                clips: [try clip("stale", frames: 481)], limits: mimoAudioInputTestLimits(),
                retaining: input,
                authorize: { plan in
                    authorizations += 1
                    return Owner(Log(), plan: plan, weights: weights)
                })
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .weightsNotLoaded) }
        XCTAssertEqual(authorizations, 0)
    }

    func testFailedDrainKeepsRootsSourceAndLeaseUntilRecovery() throws {
        enum Injected: Error { case failure }
        let weights = try bundle()
        let clips = [try clip("quarantine", frames: 481)]
        let plan = try MiMoV26AudioInputPlan.make(
            clips: clips.map(\.descriptor), configuration: weights.configuration,
            limits: mimoAudioInputTestLimits())
        let log = Log()
        var source: NSObject? = NSObject()
        weak var weakSource = source
        var root: MLXArray? = MLXArray([Float(1), 2])
        weak var weakRoot = root
        var lease: Owner? = .init(log, plan: plan, weights: weights)
        var work: MiMoV26FailedAudioWork? = try .init(
            weights: weights, clips: clips, plan: plan, sourceOwner: source!, reservation: lease!)
        XCTAssertEqual(work?.sourceIdentity, weights.sourceIdentity)
        XCTAssertEqual(work?.inputPlanIdentity, try plan.preparationIdentityData())
        XCTAssertThrowsError(try work!.recoverAndReleaseAfterDrain()) {
            XCTAssertEqual($0 as? MiMoV26AudioWorkFailure, .noFailedWork)
        }
        work!.track([root!])
        source = nil
        root = nil
        lease = nil
        var invalidated = false
        XCTAssertThrowsError(
            try work!.drain(
                invalidate: { invalidated = true }, synchronize: { throw Injected.failure })
        ) { XCTAssertEqual($0 as? MiMoV26AudioWorkFailure, .drainFailed) }
        work = nil
        XCTAssertTrue(invalidated)
        XCTAssertEqual(log.failed?.retainedRootCount, 1)
        XCTAssertEqual(log.failed?.failedDrain, true)
        XCTAssertNotNil(weakSource)
        XCTAssertNotNil(weakRoot)
        XCTAssertEqual(log.released, 0)
        XCTAssertThrowsError(
            try log.failed!.verifyRecoveryDrain(synchronize: { throw Injected.failure }))
        XCTAssertNotNil(weakRoot)
        try log.failed!.recoverAndReleaseAfterDrain()
        XCTAssertNil(weakSource)
        XCTAssertNil(weakRoot)
        XCTAssertEqual(log.released, 1)
        XCTAssertThrowsError(try log.failed!.recoverAndReleaseAfterDrain())
        log.failed = nil
    }

    func testOwnedCodecRefusesAnyIdentityOtherThanTheSelectedSidecar() throws {
        let weights = try bundle()
        let input = MiMoV26AudioInput(weights: weights)
        let main = try Fixture.configuration(Fixture.baseFields())
        XCTAssertThrowsError(
            try MiMoV26OwnedAudioCodec(
                input: input, retaining: input, expectedSourceIdentity: weights.sourceIdentity,
                expectedGeneration: weights.generation, mainConfiguration: main)
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatibleOwner) }
        XCTAssertThrowsError(
            try MiMoV26OwnedAudioCodec(
                input: input, retaining: input,
                expectedSourceIdentity: MiMoV26AudioTokenizerWeights.selectedPayloadSHA256,
                expectedGeneration: weights.generation, mainConfiguration: main)
        ) { XCTAssertEqual($0 as? MiMoV26MultimodalError, .incompatibleOwner) }
    }

    // MARK: - Strict weight maps

    func testNetworkPreparationTransposesConvolutionsAndChecksClosure() throws {
        let c = try MiMoV26AudioInputConfiguration.fixture()
        let network = MiMoV26AudioTokenizerWeights.networkSourceShapes(configuration: c)
        // Nine stem tensors and 15 per encoder layer.
        XCTAssertEqual(network.count, 9 + 4 * 15)
        let input = MiMoV26AudioTokenizerWeights.inputSourceShapes(configuration: c)
        XCTAssertEqual(input.count, network.count + 2)
        XCTAssertEqual(input["encoder.quantizer.vq.layers.1._codebook.embed"], [4, 8])
        let source = sourceWeights(c).filter { !$0.key.hasPrefix("encoder.quantizer.") }
        let prepared = try MiMoV26AudioTokenizerWeights.prepareNetwork(source, configuration: c)
        XCTAssertEqual(prepared["encoder.conv1.weight"]?.shape, [8, 3, 2])
        XCTAssertEqual(prepared["encoder.conv2.weight"]?.shape, [8, 3, 8])
        XCTAssertEqual(prepared["encoder.down_sample_layer.0.weight"]?.shape, [8, 2, 8])
        XCTAssertEqual(prepared["encoder.layers.0.fc1.weight"]?.shape, [16, 8])
        let original = try XCTUnwrap(source["encoder.conv1.weight"])
        let moved = try XCTUnwrap(prepared["encoder.conv1.weight"])
        XCTAssertEqual(
            moved.asType(.float32).asArray(Float.self),
            original.transposed(0, 2, 1).asType(.float32).asArray(Float.self))
        var missing = source
        missing.removeValue(forKey: "encoder.conv2.bias")
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.prepareNetwork(missing, configuration: c)
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26AudioInputError, .weights("network tensor closure"))
        }
        var float = source
        float["encoder.conv2.bias"] = MLXArray.zeros([8])
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.prepareNetwork(float, configuration: c)
        ) {
            XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights("encoder.conv2.bias"))
        }
        var all = sourceWeights(c)
        all.removeValue(forKey: "encoder.quantizer.vq.layers.0._codebook.embed")
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.fixtureBundle(configuration: c, weights: all)
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights("input tensor keys")) }
        var halfTable = sourceWeights(c)
        let table = "encoder.quantizer.vq.layers.1._codebook.embed"
        halfTable[table] = halfTable[table]!.asType(.bfloat16)
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.fixtureBundle(configuration: c, weights: halfTable)
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights(table)) }
    }

    // MARK: - Selected sidecar metadata

    private func selectedConfiguration() throws -> MiMoV26AudioInputConfiguration {
        func resource(_ name: String) throws -> URL {
            try XCTUnwrap(
                Bundle.module.url(
                    forResource: name, withExtension: nil, subdirectory: "MiMoOpenRouter"))
        }
        let main = try JSONDecoder().decode(
            MiMoV26Configuration.self, from: Data(contentsOf: resource("config.json")))
        return try MiMoV26AudioInputConfiguration(
            sidecarJSON: Data(contentsOf: resource("audio-config.json")), mainConfiguration: main)
    }
    /// The 439 retained decoder and codebook-statistics tensors of the
    /// selected sidecar, written out from the published layout.
    private func retainedShapes(_ c: MiMoV26AudioInputConfiguration) -> [String: [Int]] {
        var shapes: [String: [Int]] = [
            "decoder.dconv1.conv.bias": [1024], "decoder.dconv1.conv.weight": [1024, 1024, 2],
            "decoder.dconv1.norm.bias": [1024], "decoder.dconv1.norm.weight": [1024],
            "decoder.dconv2.conv.bias": [128], "decoder.dconv2.conv.weight": [1024, 128, 3],
            "decoder.dconv2.norm.bias": [128], "decoder.dconv2.norm.weight": [128],
            "decoder.layer_norm.bias": [1024], "decoder.layer_norm.weight": [1024],
            "decoder.vocoder.istft.window": [960], "decoder.vocoder.out.bias": [962],
            "decoder.vocoder.out.weight": [962, 128],
        ]
        for (i, pair) in [(481, 128), (241, 64), (121, 32)].enumerated() {
            shapes["decoder.mel_loss_fn.mel_transforms.\(i).mel_scale.fb"] = [pair.0, pair.1]
            shapes["decoder.mel_loss_fn.mel_transforms.\(i).spectrogram.window"] = [
                (pair.0 - 1) * 2
            ]
        }
        for i in 0 ..< 24 {
            let p = "decoder.layers.\(i)"
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                shapes[p + ".self_attn." + name + ".weight"] = [1024, 1024]
                if name != "k_proj" { shapes[p + ".self_attn." + name + ".bias"] = [1024] }
            }
            for name in ["self_attn_layer_norm", "final_layer_norm"] {
                shapes[p + "." + name + ".weight"] = [1024]
                shapes[p + "." + name + ".bias"] = [1024]
            }
            shapes[p + ".fc1.weight"] = [4096, 1024]
            shapes[p + ".fc1.bias"] = [4096]
            shapes[p + ".fc2.weight"] = [1024, 4096]
            shapes[p + ".fc2.bias"] = [1024]
        }
        for (i, bins) in c.codebookSizes.enumerated() {
            let p = "encoder.quantizer.vq.layers.\(i)._codebook."
            shapes[p + "embed_avg"] = [bins, 1024]
            shapes[p + "cluster_size"] = [bins]
            shapes[p + "inited"] = [1]
        }
        return shapes
    }
    private func descriptors(_ c: MiMoV26AudioInputConfiguration) -> [String:
        MiMoV26AudioTensorDescriptor]
    {
        let all = MiMoV26AudioTokenizerWeights.inputSourceShapes(configuration: c)
            .merging(retainedShapes(c)) { first, _ in first }
        var result: [String: MiMoV26AudioTensorDescriptor] = [:]
        for (name, shape) in all {
            let type: MiMoV26AudioTensorType =
                name.hasPrefix("encoder.") && !name.hasPrefix("encoder.quantizer.")
                ? .bfloat16 : .float32
            result[name] = .init(shape: shape, dtype: type, byteCount: shape.reduce(type.bytes, *))
        }
        return result
    }

    func testSelectedSidecarPreflightClassifiesEveryTensorWithoutArrays() throws {
        let c = try selectedConfiguration()
        let all = descriptors(c)
        XCTAssertEqual(all.count, 828)
        let digest = MiMoV26AudioTokenizerWeights.selectedPayloadSHA256
        let plan = try MiMoV26AudioTokenizerWeights.preflight(
            descriptors: all, configuration: c, sourcePayloadSHA256: digest)
        XCTAssertEqual(plan.requiredInputNames.count, 389)
        XCTAssertEqual(plan.inputStoredBytes, 634_204_160)
        XCTAssertEqual(plan.retainedUnusedStoredBytes, 1_238_321_176)
        XCTAssertEqual(plan.sourcePayloadSHA256, digest)
        XCTAssertTrue(plan.requiredInputNames.allSatisfy { !$0.hasPrefix("decoder.") })
        XCTAssertThrowsError(try MiMoV26AudioTokenizerWeights.load(plan: plan, inputWeights: [:])) {
            XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights("input subset closure"))
        }
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.preflight(
                descriptors: all, configuration: c,
                sourcePayloadSHA256: String(repeating: "0", count: 64))
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26AudioInputError, .weights("selected sidecar/profile identity"))
        }
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.preflight(
                descriptors: all, configuration: MiMoV26AudioInputConfiguration.fixture(),
                sourcePayloadSHA256: digest)
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26AudioInputError, .weights("selected sidecar/profile identity"))
        }
        var missing = all
        missing.removeValue(forKey: "decoder.vocoder.istft.window")
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.preflight(
                descriptors: missing, configuration: c, sourcePayloadSHA256: digest)
        ) {
            XCTAssertEqual(
                $0 as? MiMoV26AudioInputError, .weights("missing/unclassified sidecar tensors"))
        }
        var wrongType = all
        wrongType["encoder.conv1.bias"] = .init(shape: [1024], dtype: .float32, byteCount: 4096)
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.preflight(
                descriptors: wrongType, configuration: c, sourcePayloadSHA256: digest)
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights("encoder.conv1.bias")) }
        var wrongBytes = all
        wrongBytes["decoder.vocoder.out.bias"] = .init(shape: [962], dtype: .float32, byteCount: 1)
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.preflight(
                descriptors: wrongBytes, configuration: c, sourcePayloadSHA256: digest)
        ) { XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights("decoder.vocoder.out.bias")) }
    }
}
