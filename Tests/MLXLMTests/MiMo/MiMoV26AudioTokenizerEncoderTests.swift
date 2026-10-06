import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

struct MiMoAudioOracleTensor: Decodable {
    let shape: [Int]
    let dtype: String
    let data: [Float]
    func array() -> MLXArray {
        MLXArray(data).reshaped(shape).asType(dtype == "BF16" ? .bfloat16 : .float32)
    }
}
struct MiMoAudioCodecOracle: Decodable {
    struct Case: Decodable {
        let name: String
        let mels: [MiMoAudioOracleTensor]
        let codes: [[Int32]]
        let traces: [String: MiMoAudioOracleTensor]
    }
    let sourceSHA256: String
    let weights: [String: MiMoAudioOracleTensor]
    let cases: [Case]
}
func mimoAudioCodecOracle() throws -> MiMoAudioCodecOracle {
    let root = try mimoAudioInputFixtureRoot()
    let value = try JSONDecoder().decode(
        MiMoAudioCodecOracle.self,
        from: Data(contentsOf: root.appendingPathComponent("codec-oracle.json")))
    XCTAssertEqual(
        value.sourceSHA256, "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad")
    return value
}

final class MiMoV26AudioTokenizerEncoderTests: XCTestCase {
    func testStrictCheckpointIndexedPoolLoadsRealArraysWithoutExtraParameters() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try MiMoV26AudioInputConfiguration.fixture()
            var source = MiMoV26AudioTokenizerWeights.networkSourceShapes(configuration: c)
                .mapValues { MLXArray.zeros($0, dtype: .bfloat16, stream: .cpu) }
            let name = "encoder.down_sample_layer.0.weight"
            let width = c.hiddenSize
            let values = (0 ..< width * width * 2).map { Float($0) }
            source[name] = MLXArray(values).reshaped(width, width, 2, stream: .cpu)
                .asType(.bfloat16, stream: .cpu)
            let encoder = MiMoV26AudioTokenizerEncoder(configuration: c)
            // This is the normal strict unflatten/update path that rejected
            // the old module-wrapped numeric key as incompatibleItems.
            try encoder.loadNativeNetworkWeights(source)
            XCTAssertNotNil(encoder.loadedGeneration)
            let installed = Dictionary(uniqueKeysWithValues: encoder.parameters().flattened())
            XCTAssertEqual(Set(installed.keys), Set(source.keys))
            XCTAssertEqual(installed.count, 69)
            XCTAssertEqual(
                installed.keys.filter { $0.hasPrefix("encoder.down_sample_layer.") }, [name])
            let pool = try XCTUnwrap(installed[name])
            XCTAssertEqual(pool.shape, [width, 2, width])
            XCTAssertEqual(pool.dtype, .bfloat16)
            // Independent source-layout oracle: [out, in, kernel] -> [out, kernel, in].
            let expected = (0 ..< width).flatMap { output in
                (0 ..< 2).flatMap { kernel in
                    (0 ..< width).map { input in values[(output * width + input) * 2 + kernel] }
                }
            }
            XCTAssertEqual(pool.asType(.float32, stream: .cpu).asArray(Float.self), expected)
        }
    }

    func testStrictCheckpointPoolRejectsMissingExtraWrongShapeAndWrongDType() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = try MiMoV26AudioInputConfiguration.fixture()
            let source = MiMoV26AudioTokenizerWeights.networkSourceShapes(configuration: c)
                .mapValues { MLXArray.zeros($0, dtype: .bfloat16, stream: .cpu) }
            let name = "encoder.down_sample_layer.0.weight"
            let encoder = MiMoV26AudioTokenizerEncoder(configuration: c)
            try encoder.loadNativeNetworkWeights(source)
            var missing = source
            missing.removeValue(forKey: name)
            XCTAssertThrowsError(try encoder.loadNativeNetworkWeights(missing)) {
                XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights("network tensor closure"))
            }
            XCTAssertNil(encoder.loadedGeneration)
            var extra = source
            extra["encoder.down_sample_layer.1.weight"] = source[name]
            XCTAssertThrowsError(try encoder.loadNativeNetworkWeights(extra)) {
                XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights("network tensor closure"))
            }
            var wrongShape = source
            wrongShape[name] = MLXArray.zeros(
                [c.hiddenSize, 2, c.hiddenSize], dtype: .bfloat16, stream: .cpu)
            XCTAssertThrowsError(try encoder.loadNativeNetworkWeights(wrongShape)) {
                XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights(name))
            }
            var wrongDType = source
            wrongDType[name] = source[name]!.asType(.float32, stream: .cpu)
            XCTAssertThrowsError(try encoder.loadNativeNetworkWeights(wrongDType)) {
                XCTAssertEqual($0 as? MiMoV26AudioInputError, .weights(name))
            }
            XCTAssertNil(encoder.loadedGeneration)
            try encoder.loadNativeNetworkWeights(source)
            XCTAssertNotNil(encoder.loadedGeneration)
        }
    }

    func testCompleteSidecarMetadataClosureAndNoInventedKeyBias() throws {
        let root = try mimoAudioInputFixtureRoot()
        let main = try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: Data(contentsOf: root.appendingPathComponent("main-config.json")))
        let c = try MiMoV26AudioInputConfiguration(
            sidecarJSON: Data(contentsOf: root.appendingPathComponent("sidecar-config.json")),
            mainConfiguration: main)
        let descriptors = try JSONDecoder().decode(
            [String: MiMoV26AudioTensorDescriptor].self,
            from: Data(contentsOf: root.appendingPathComponent("sidecar-descriptors.json")))
        let plan = try MiMoV26AudioTokenizerWeights.preflight(
            descriptors: descriptors, configuration: c,
            sourcePayloadSHA256: MiMoV26AudioTokenizerWeights.selectedPayloadSHA256)
        XCTAssertEqual(plan.requiredInputNames.count, 389)
        XCTAssertEqual(plan.inputStoredBytes, 634_204_160)
        XCTAssertEqual(plan.retainedUnusedStoredBytes, 1_238_321_176)
        XCTAssertFalse(plan.requiredInputNames.contains { $0.hasPrefix("decoder.") })
        XCTAssertFalse(
            plan.requiredInputNames.contains {
                $0.hasSuffix("k_proj.bias") || $0.contains("inv_freq")
            })
        var incomplete = descriptors
        incomplete.removeValue(forKey: "encoder.quantizer.vq.layers.0._codebook.embed")
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.preflight(
                descriptors: incomplete, configuration: c,
                sourcePayloadSHA256: plan.sourcePayloadSHA256))
        var wrong = descriptors
        wrong["encoder.conv1.weight"] = .init(
            shape: [1024, 128, 3], dtype: .float32, byteCount: 1_572_864)
        XCTAssertThrowsError(
            try MiMoV26AudioTokenizerWeights.preflight(
                descriptors: wrong, configuration: c, sourcePayloadSHA256: plan.sourcePayloadSHA256)
        )
    }

    func testUnchangedHFOracleAllIntermediatesAndOriginalBatchGeometry() throws {
        try mimoAudioInputRequireNative()
        let oracle = try mimoAudioCodecOracle()
        let c = try MiMoV26AudioInputConfiguration.fixture()
        let bundle = try MiMoV26AudioTokenizerWeights.fixtureBundle(
            configuration: c, weights: oracle.weights.mapValues { $0.array() })
        XCTAssertEqual(bundle.encoder.parameters().flattened().count, 69)
        XCTAssertGreaterThanOrEqual(oracle.cases.count, 5)
        var capturedConv2: [String: [Float]] = [:]
        for item in oracle.cases {
            let plan = try MiMoV26AudioInputPlan.makeMelInputs(
                frameCounts: item.mels.map { $0.shape[0] },
                sourceIdentities: item.mels.indices.map { "clip-\($0)" }, configuration: c,
                limits: mimoAudioInputTestLimits())
            var traces: [String: MLXArray] = [:]
            let output = try bundle.encoder.encode(
                item.mels.map { $0.array() }, plan: plan, isCancelled: { false }
            ) { traces[$0] = $1 }
            eval(Array(traces.values) + [output.features])
            XCTAssertEqual(Set(traces.keys), Set(item.traces.keys), item.name)
            for (name, expected) in item.traces {
                let actual = try XCTUnwrap(traces[name])
                XCTAssertEqual(actual.shape, expected.shape, item.name + ":" + name)
                let values = actual.asType(.float32).asArray(Float.self)
                XCTAssertEqual(values.count, expected.data.count)
                for i in values.indices {
                    XCTAssertEqual(
                        values[i], expected.data[i], accuracy: 0.04, item.name + ":" + name)
                }
            }
            let quantized = try bundle.quantizer.quantize(
                features: output.features, frameCounts: plan.codeFrameCounts, tileFrames: 2)
            eval(quantized.codes, quantized.allFinite)
            XCTAssertTrue(quantized.allFinite.item(Bool.self))
            XCTAssertEqual(
                quantized.codes.asArray(Int32.self), item.codes.flatMap { $0 }, item.name)
            capturedConv2[item.name] = traces["group.0.conv2"]!.asType(.float32).asArray(Float.self)
        }
        // Same short input, different padded peer extent; do not turn the
        // unpadded serial variant into the oracle for the selected profile.
        let alone = try XCTUnwrap(capturedConv2["alone5"])
        let mixed = try XCTUnwrap(capturedConv2["mixed5_6"])
        XCTAssertNotEqual(alone, Array(mixed.prefix(alone.count)))
    }

    func testRoundedRotaryAndNoLearnedConstantParameter() throws {
        try mimoAudioInputRequireNative()
        let c = try MiMoV26AudioInputConfiguration.fixture(hiddenSize: 64, heads: 1)
        let rope = MiMoV26AudioTokenizerEncoder.rotary(configuration: c, positions: [0, 128, 2999])
        eval(rope.inverse, rope.cosine, rope.sine)
        XCTAssertEqual(rope.inverse.asArray(Float.self)[1], 0.75)
        XCTAssertEqual(rope.cosine.dtype, .bfloat16)
        XCTAssertEqual(rope.sine.dtype, .bfloat16)
        let ordinary = Float(1) / pow(Float(10000), Float(2) / 64)
        XCTAssertNotEqual(ordinary * 2999, Float(0.75) * 2999)
        let encoder = MiMoV26AudioTokenizerEncoder(configuration: c)
        XCTAssertFalse(encoder.parameters().flattened().contains { $0.0.contains("inv_freq") })
    }

    func testMissingWeightsBadShapeCancellationAndSupportedReloadGeneration() throws {
        try mimoAudioInputRequireNative()
        let c = try MiMoV26AudioInputConfiguration.fixture()
        let oracle = try mimoAudioCodecOracle()
        let encoder = MiMoV26AudioTokenizerEncoder(configuration: c)
        let p = try MiMoV26AudioInputPlan.makeMelInputs(
            frameCounts: [3], sourceIdentities: ["fixture"], configuration: c,
            limits: mimoAudioInputTestLimits())
        let mel = MLXArray.zeros([3, 2], dtype: .float32)
        XCTAssertThrowsError(try encoder.encodeMelFeatures(mels: [mel], plan: p))
        let network = oracle.weights.filter { !$0.key.hasPrefix("encoder.quantizer.") }.mapValues {
            $0.array()
        }
        try encoder.loadNativeNetworkWeights(network)
        let first = try XCTUnwrap(encoder.loadedGeneration)
        XCTAssertThrowsError(
            try encoder.encodeMelFeatures(mels: [MLXArray.zeros([4, 2])], plan: p))
        XCTAssertThrowsError(
            try encoder.encodeMelFeatures(mels: [mel], plan: p, isCancelled: { true }))
        try encoder.loadNativeNetworkWeights(network)
        XCTAssertNotEqual(encoder.loadedGeneration, first)
        var missing = network
        missing.removeValue(forKey: "encoder.layers.0.self_attn.k_proj.weight")
        XCTAssertThrowsError(try encoder.loadNativeNetworkWeights(missing))
        XCTAssertNil(encoder.loadedGeneration)
        XCTAssertThrowsError(try encoder.encodeMelFeatures(mels: [mel], plan: p))
    }
}
