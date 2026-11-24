import MLX
import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// What the engine does when the settled assistant copy of a prefix
    /// checkpoint throws (`EngineLoopV2.settledAssistantCopyFailure`). Only an
    /// assistant that opts in (Inkling 2's DSpark drafter) splits device errors
    /// from refusals. Every other one (MiMo V2.6) discards the steps of the
    /// rows that had checkpoints, whatever the error.
    struct CBv2SettledAssistantCopyFailureTests {
        struct Refusal: Error {}

        @Test func optedInAssistantSplitsDeviceErrorsFromRefusals() {
            #expect(
                EngineLoopV2.settledAssistantCopyFailure(
                    MLXError.caught("device"), keepsDonorStep: true) == .failCohort)
            #expect(
                EngineLoopV2.settledAssistantCopyFailure(Refusal(), keepsDonorStep: true)
                    == .dropCheckpoints)
        }

        @Test func otherAssistantsDiscardTheRowsWithCheckpoints() {
            #expect(
                EngineLoopV2.settledAssistantCopyFailure(
                    MLXError.caught("device"), keepsDonorStep: false) == .discardCheckpointRows)
            #expect(
                EngineLoopV2.settledAssistantCopyFailure(Refusal(), keepsDonorStep: false)
                    == .discardCheckpointRows)
        }
    }
}
