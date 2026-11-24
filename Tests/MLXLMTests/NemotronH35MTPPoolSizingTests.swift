import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The Nemotron 3.5 MTP assistant's private pool holds one row for the
/// request's whole bound. Its segmented pool charges the grant whole segments
/// (one poison page each, power-of-two sizes, allocation footprint), so a
/// grant of the pages alone plus a fixed slack fell short past about 165K
/// tokens and the request's first step failed with `capacityExhausted`.
@Suite("Nemotron 3.5 MTP pool sizing", .serialized)
struct NemotronH35MTPPoolSizingTests {
    /// The attention layer of the Lightning MTP head (2 KV heads of 128, BF16).
    static let kind = CBv2LayerKind(
        attention: .full, headDim: 128, kvHeads: 2, queryHeads: 32, modelLayerIndex: 0)

    /// Bounds are prompt + max(max_tokens, 1) + the draft padding (8): 165,856
    /// fit before; 165,857, 166,881, 176,129 and up to the 262,144-token
    /// window did not.
    @Test(arguments: [165_856, 165_857, 166_881, 176_129, 200_008, 262_144, 262_152, 327_688])
    func aRowIsAdmittedForItsWholeBound(bound: Int) throws {
        let state = try NemotronH35MTPAssistant.PagedState(
            kind: Self.kind, dtype: .bfloat16, maximumSequenceLength: bound)
        #expect(state.maximumSequenceLength == bound)
        state.release()
    }

    /// The row's charge stays within the assistant's admission floor of
    /// 20 KiB per token, so the larger grant is already paid for.
    @Test(arguments: [165_857, 262_152, 327_688])
    func theRowChargeFitsThePerTokenFloor(bound: Int) throws {
        let config = PagedKVPoolConfig(
            capacityBytes: 1 << 30, dtype: .bfloat16, maxPrefillChunk: 512,
            nominalMaxSequenceLength: bound, segmentSizeBytes: 4 * 1024 * 1024,
            layerDTypes: [.bfloat16])
        let bytes = try #require(
            PagedKVPool.segmentedRowReservationBytes(
                layerKinds: [Self.kind], maxLength: bound, config: config))
        #expect(bytes <= 20 * 1024 * bound)
    }
}
