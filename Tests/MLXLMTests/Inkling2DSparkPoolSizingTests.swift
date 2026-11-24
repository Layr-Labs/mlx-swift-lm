import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The DSpark drafter's private pool holds one row for the request's whole
/// bound. Its segmented pool charges the grant whole segments (one poison page
/// each, power-of-two sizes, allocation footprint), so a grant of the pages
/// alone plus a fixed slack fell short past about 41K tokens: the request's
/// first step failed with `capacityExhausted` (for example needed 210,730,909,
/// available 209,780,736 at a 46,990-token bound).
@Suite("Inkling2 DSpark pool sizing", .serialized)
struct Inkling2DSparkPoolSizingTests {
    /// The drafter geometry of `thinkingmachines/Inkling-2-Mini` (`mtp_config`).
    static func configuration() throws -> Inkling2DSparkConfiguration {
        let c = Inkling2DSparkConfiguration(
            hiddenSize: 2304, numHiddenLayers: 3, numAttentionHeads: 24, numKeyValueHeads: 8,
            headDim: 128, intermediateSize: 9216, localLayerIds: [0, 2], slidingWindowSize: 1024,
            blockSize: 8, markovRank: 256, targetLayerIds: [5, 23, 41, 48],
            vocabularySize: 200_058, paddedVocabularySize: 201_024, finalLogitSoftcapping: 30,
            targetLayerVisits: 48)
        try c.validate()
        return c
    }

    /// Bounds are prompt + max(max_tokens, 1) + the block size (8): 41,440 fit
    /// before; 41,441, 46,056, 46,990 and 87,037 did not.
    @Test(arguments: [41_440, 41_441, 43_241, 46_056, 46_990, 87_037, 131_080])
    func aRowIsAdmittedForItsWholeBound(bound: Int) throws {
        let state = try Inkling2DSparkDrafter.PagedState(
            configuration: Self.configuration(), maximumSequenceLength: bound)
        #expect(state.maximumSequenceLength == bound)
        state.release()
    }
}
