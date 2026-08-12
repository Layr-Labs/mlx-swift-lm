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
    }

    func testInitialAdapterCapabilitiesFailClosed() throws {
        let config = try configuration()
        let capability = config.cbv2Capabilities
        XCTAssertFalse(capability.supportsPrefixReuse)
        XCTAssertFalse(capability.supportsPagedKV)
        XCTAssertFalse(capability.supportsCompiledDecode)
        XCTAssertFalse(capability.supportsPackedPrefill)
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
