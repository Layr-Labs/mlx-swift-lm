import Foundation
import XCTest

@testable import MLXLMCommon

/// Metadata-only format tests. These do not certify native assistant capture,
/// allocation ownership, atomic adoption or model-level MTP/cache exactness.
final class CBv2HistoricalMTPManifestTests: XCTestCase {
    private let position = 128
    private func fixture() throws -> (
        target: [CBv2CheckpointTensorDescriptor],
        assistant: [CBv2CheckpointTensorDescriptor],
        layers: [CBv2CheckpointAttentionLayer]
    ) {
        let kinds: [CBv2LayerKind] = [
            .init(attention: .full, headDim: 192, valueHeadDim: 128, kvHeads: 4, queryHeads: 64),
            .init(
                attention: .slidingWindow(17), headDim: 192, valueHeadDim: 128, kvHeads: 8,
                queryHeads: 64),
        ]
        let layout = try CBv2HistoricalAttentionLayout(
            layerKinds: kinds,
            dtypes: [.bfloat16, .bfloat16], allowAsymmetric: true)
        var assistant: [CBv2CheckpointTensorDescriptor] = []
        for head in 0 ..< 3 {
            assistant.append(
                try .init(
                    role: .assistantKeys, layer: head,
                    shape: [1, 4, position - head - 1, 192], dtype: .bfloat16))
            assistant.append(
                try .init(
                    role: .assistantValues, layer: head,
                    shape: [1, 4, position - head - 1, 128], dtype: .bfloat16))
        }
        assistant.append(try .init(role: .assistantHidden, shape: [1, 3, 4096], dtype: .bfloat16))
        assistant.append(try .init(role: .assistantTokens, shape: [1, position], dtype: .int32))
        assistant.append(try .init(role: .assistantCacheMetadata, shape: [3, 7], dtype: .int64))
        return (try layout.tensorDescriptors(position: position), assistant, layout.layers)
    }
    private func manifest(
        _ tensors: [CBv2CheckpointTensorDescriptor],
        layers: [CBv2CheckpointAttentionLayer]?,
        layout: String = CBv2CompleteCheckpointManifest.contiguousAsymmetricMTPLayout,
        assistant: String? = "mimo-v26-three-head-swa-bf16-target-tail-v1"
    ) -> CBv2CompleteCheckpointManifest {
        .init(
            identity: .init(
                modelAggregateHash: "artifact", promptContractID: "template",
                buildID: "test", numericsFingerprint: "native-bf16"),
            position: position, chunkSize: position,
            prefixTokens: Array(repeating: 7, count: position),
            cacheSalt: "tenant", assistantCodecID: assistant, tensors: tensors,
            backendLayout: layout, attentionLayers: layers)
    }

    func testCompleteHistoricalAssistantHasDistinctRoundTrippableFormat() throws {
        let f = try fixture()
        let value = manifest(f.target + f.assistant, layers: f.layers)
        XCTAssertEqual(
            try value.validateStructure(), (f.target + f.assistant).reduce(0) { $0 + $1.byteCount })
        let restored = try JSONDecoder().decode(
            CBv2CompleteCheckpointManifest.self,
            from: JSONEncoder().encode(value))
        XCTAssertEqual(restored, value)
        XCTAssertEqual(try restored.validateStructure(), try value.validateStructure())
        XCTAssertNotEqual(
            value.backendLayout, CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout)
    }

    func testTargetOnlyFormatStillRejectsAssistantStateAndIdentity() throws {
        let f = try fixture()
        let old = CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout
        XCTAssertNoThrow(
            try manifest(f.target, layers: f.layers, layout: old, assistant: nil)
                .validateStructure())
        XCTAssertThrowsError(
            try manifest(f.target, layers: f.layers, layout: old).validateStructure())
        XCTAssertThrowsError(
            try manifest(
                f.target + f.assistant, layers: f.layers,
                layout: old, assistant: nil
            ).validateStructure())
        XCTAssertThrowsError(try manifest(f.target, layers: f.layers).validateStructure())
        XCTAssertThrowsError(
            try manifest(f.target + f.assistant, layers: f.layers, assistant: nil)
                .validateStructure())
        XCTAssertThrowsError(
            try manifest(f.target + f.assistant, layers: f.layers, assistant: "")
                .validateStructure())
    }

    func testAssistantKVPairAndExactTokenWitnessAreMandatory() throws {
        let f = try fixture()
        for removed in [0, 1, 6, 7, 8] {
            var assistant = f.assistant
            assistant.remove(at: removed)
            XCTAssertThrowsError(
                try manifest(f.target + assistant, layers: f.layers).validateStructure())
        }
        for wrong in [
            try CBv2CheckpointTensorDescriptor(
                role: .assistantTokens, shape: [1, position - 1], dtype: .int32),
            try CBv2CheckpointTensorDescriptor(
                role: .assistantTokens, shape: [1, position], dtype: .int64),
            try CBv2CheckpointTensorDescriptor(
                role: .assistantTokens, layer: 0, shape: [1, position], dtype: .int32),
        ] {
            var assistant = f.assistant
            assistant[7] = wrong
            XCTAssertThrowsError(
                try manifest(f.target + assistant, layers: f.layers).validateStructure())
        }
    }

    func testMismatchedHeadsDtypesAndTargetReorderingAreRejected() throws {
        let f = try fixture()
        for wrong in [
            try CBv2CheckpointTensorDescriptor(
                role: .assistantValues, layer: 0, shape: [1, 4, position - 2, 128], dtype: .bfloat16
            ),
            try CBv2CheckpointTensorDescriptor(
                role: .assistantValues, layer: 0, shape: [1, 4, position - 1, 128], dtype: .float16),
            try CBv2CheckpointTensorDescriptor(
                role: .assistantValues, layer: 7, shape: [1, 4, position - 1, 128], dtype: .bfloat16
            ),
            try CBv2CheckpointTensorDescriptor(
                role: .values, layer: 7, shape: [1, 4, position - 1, 128], dtype: .bfloat16),
        ] {
            var assistant = f.assistant
            assistant[1] = wrong
            XCTAssertThrowsError(
                try manifest(f.target + assistant, layers: f.layers).validateStructure())
        }
        var tensors = f.target + f.assistant
        tensors.swapAt(0, 1)
        XCTAssertThrowsError(try manifest(tensors, layers: f.layers).validateStructure())
        XCTAssertThrowsError(
            try manifest(f.target + f.assistant + [f.assistant[0]], layers: f.layers)
                .validateStructure())
    }
}
