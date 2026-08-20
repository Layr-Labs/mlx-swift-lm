import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen35CBv2ConfigurationTests: XCTestCase {
    private final class PagedIdentityBackend: CBv2KVBackend {
        var prefixReuseBackend: CBv2PrefixReuseBackend { .pagedFP16 }
        var bytesInUse: Int { 0 }
        var bytesCapacity: Int { 1 << 20 }
        func makeSequenceState(
            layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
        ) throws -> [CBv2SequenceKV?] { [] }
        func release(_ state: [CBv2SequenceKV?]) {}
    }

    private func configuration(
        hiddenLayers: Int = 40,
        fullAttentionInterval: Int = 4,
        valueHeads: Int = 32,
        keyHeads: Int = 16,
        keyHeadDim: Int = 128,
        valueHeadDim: Int = 128,
        convKernel: Int = 4
    ) throws -> Qwen35TextConfiguration {
        let json = """
            {
              "model_type": "qwen3_5_moe",
              "hidden_size": 64,
              "num_hidden_layers": \(hiddenLayers),
              "intermediate_size": 128,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 16,
              "linear_num_value_heads": \(valueHeads),
              "linear_num_key_heads": \(keyHeads),
              "linear_key_head_dim": \(keyHeadDim),
              "linear_value_head_dim": \(valueHeadDim),
              "linear_conv_kernel_dim": \(convKernel),
              "vocab_size": 64,
              "full_attention_interval": \(fullAttentionInterval)
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    func testConcreteArtifactFixedBytesAndLayerMapping() throws {
        let config = try configuration()
        let spec = config.cbv2RecurrentStateSpec(activationDType: .bfloat16)

        XCTAssertEqual(spec.layers.count, 30)
        XCTAssertEqual(try spec.fixedBytesPerRequest(), 64_389_120)
        XCTAssertEqual(try spec.peakBytesPerRequest(), 193_167_360)
        XCTAssertTrue(spec.layers.allSatisfy { $0.convShape == [1, 3, 8192] })
        XCTAssertTrue(spec.layers.allSatisfy { $0.convDType == .bfloat16 })
        XCTAssertTrue(spec.layers.allSatisfy { $0.ssmShape == [1, 32, 128, 128] })
        XCTAssertTrue(spec.layers.allSatisfy { $0.ssmDType == .float32 })

        let kinds = config.cbv2LayerKinds
        XCTAssertEqual(kinds.count, 10)
        XCTAssertEqual(kinds.compactMap(\.modelLayerIndex), Array(stride(from: 3, to: 40, by: 4)))
        XCTAssertTrue(kinds.allSatisfy { $0.attention == .full })
    }

    func testCacheConstructionUsesCompactStorageAndOriginalLayerIndices() throws {
        let config = try configuration(hiddenLayers: 8, valueHeads: 2, keyHeads: 1,
            keyHeadDim: 4, valueHeadDim: 4)
        let model = Qwen35TextModel(config)
        var originalIndices: [Int] = []
        let caches = model.newCacheV2 { index, kind in
            originalIndices.append(index)
            return CBv2LayerCache(layerIndex: index, kind: kind)
        }
        XCTAssertEqual(caches.count, 2)
        XCTAssertEqual(originalIndices, [3, 7])
        XCTAssertEqual(caches.map(\.layerIndex), [3, 7])
        XCTAssertEqual(caches.map { $0.kind.modelLayerIndex }, [3, 7])
        XCTAssertEqual(
            CBv2SteppableLanguageModelAdapter(model).cbv2PositionAxisCount, 3)
    }

    func testInitialAdapterCapabilitiesFailClosed() throws {
        let config = try configuration()
        let capability = config.cbv2Capabilities
        XCTAssertFalse(capability.supportsPrefixReuse)
        XCTAssertFalse(capability.supportsPagedKV)
        XCTAssertFalse(capability.supportsCompiledDecode)
        // Packed prefill is now a deliberate Qwen claim (rectangular [B, L]
        // cohorts, one recurrent state row per batch row) — no longer part
        // of the fail-closed initial-target set.
        XCTAssertTrue(capability.supportsPackedPrefill)
        XCTAssertTrue(capability.supportsMTP)

        let prefix = CBv2PrefixReuseCapability.derive(
            layerKinds: config.cbv2LayerKinds,
            backend: .contiguousUnquantized,
            modelSupportsPrefixReuse: capability.supportsPrefixReuse)
        XCTAssertFalse(prefix.isSupported)
        XCTAssertEqual(prefix.unsupportedReason, .modelRequestStateUnsupported)
        XCTAssertNotNil(
            EngineV2.backendCapabilityViolation(
                capabilities: capability, backend: PagedIdentityBackend()))
    }

    func testOuterConfigurationDelegatesCapabilitiesToTextConfig() throws {
        let text = try configuration()
        let textObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(text)) as? [String: Any])
        let root: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": textObject,
        ]
        let outer = try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: JSONSerialization.data(withJSONObject: root))

        XCTAssertEqual(outer.cbv2Capabilities, outer.textConfig.cbv2Capabilities)
        XCTAssertTrue(outer.cbv2Capabilities.supportsMTP)
    }

    func testMRoPEUsesConfiguredLinearScaling() {
        let scaling: [String: StringOrNumber] = [
            "type": .string("linear"),
            "factor": .float(2),
        ]
        let rope = initializeRope(
            dims: 8, base: 10_000, traditional: false,
            scalingConfig: scaling, maxPositionEmbeddings: 128)
        let mrope = Qwen35MRoPE(
            rope: rope, dim: 8, base: 10_000, scalingConfig: scaling,
            sections: [2, 1, 1])
        let value = MLXArray(Array(0 ..< 16).map(Float.init)).reshaped([1, 1, 2, 8])
        let scalarPositions = MLXArray([Int32(3), 7]).reshaped([1, 2])
        let positions = broadcast(scalarPositions[.newAxis, 0..., 0...], to: [3, 1, 2])

        let expected = rope(
            value.transposed(0, 2, 1, 3).reshaped([2, 1, 1, 8]),
            offset: scalarPositions.flattened()
        ).reshaped([1, 2, 1, 8]).transposed(0, 2, 1, 3)
        let actual = mrope.apply(
            queries: value, keys: value, positionIds: positions).0
        eval(expected, actual)
        XCTAssertTrue(allClose(expected, actual, rtol: 0, atol: 0).item(Bool.self))
    }

    func testDefaultMRoPEMatchesInterleavedReference() {
        let rope = initializeRope(
            dims: 8, base: 10_000, traditional: false,
            scalingConfig: ["type": .string("default")], maxPositionEmbeddings: 128)
        let sections = [2, 1, 1]
        let mrope = Qwen35MRoPE(
            rope: rope, dim: 8, base: 10_000,
            scalingConfig: ["type": .string("default")], sections: sections)
        let value = MLXArray(Array(0 ..< 16).map(Float.init)).reshaped([1, 1, 2, 8])
        let positions = MLXArray([
            Int32(3), 7,
            5, 11,
            9, 13,
        ]).reshaped([3, 1, 2])

        let exponents = MLXArray(stride(from: 0, to: 8, by: 2)).asType(.float32) / 8
        let invFreq = 1 / pow(MLXArray(Float(10_000)), exponents)
        let all = positions.asType(.float32)[0..., 0..., 0..., .newAxis]
            * invFreq[.newAxis, .newAxis, .newAxis, 0...]
        var selected: [MLXArray] = []
        for index in 0 ..< 4 {
            var axis = 0
            for (candidate, offset) in [(1, 1), (2, 2)] {
                if index >= offset && index < min(sections[candidate] * 3, 4)
                    && (index - offset) % 3 == 0
                {
                    axis = candidate
                    break
                }
            }
            selected.append(all[axis, 0..., 0..., index])
        }
        let angles = concatenated([
            stacked(selected, axis: -1), stacked(selected, axis: -1),
        ], axis: -1)
        let cosine = cos(angles).expandedDimensions(axis: 1)
        let sine = sin(angles).expandedDimensions(axis: 1)
        let rotatedHalf = concatenated(
            [-value[.ellipsis, 4...], value[.ellipsis, ..<4]], axis: -1)
        let expected = value * cosine + rotatedHalf * sine
        let actual = mrope.apply(queries: value, keys: value, positionIds: positions).0
        eval(expected, actual)
        XCTAssertTrue(allClose(expected, actual, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    }

    func testOverflowingRecurrentShapeFailsClosed() {
        let spec = CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [Int.max, 2], convDType: .float32,
                ssmShape: [1], ssmDType: .float32)
        ])
        XCTAssertThrowsError(try spec.fixedBytesPerRequest()) { error in
            XCTAssertEqual(error as? CBv2RecurrentStateError, .byteCountOverflow)
        }
    }
}
