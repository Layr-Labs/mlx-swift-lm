import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension UnitTests {
    /// Coupled drafting (`CBv2MTPConfig.coupledProposals`) is off unless a
    /// configuration opts in, and only sampled rows outside any grammar couple
    /// (`EngineLoopV2.mtpCouplesProposals`). Only block drafters that adopt
    /// `CBv2MTPCoupledBlockDrafter` (Inkling 2's DSpark drafter) receive a
    /// selection, so every other family's rounds are unchanged.
    struct CBv2MTPCoupledProposalsConfigTests {
        @Test func offUnlessAConfigurationOptsIn() {
            #expect(!CBv2MTPConfig().coupledProposals)
            #expect(!CBv2MTPConfig(enabled: true).coupledProposals)
            #expect(!Inkling2DSparkDrafter.defaultMTPConfig().coupledProposals)
            #expect(!Inkling2DSparkDrafter.defaultMTPConfig(maxDraftTokens: 3).coupledProposals)
            let on = Inkling2DSparkDrafter.defaultMTPConfig(coupledProposals: true)
            #expect(on.coupledProposals)
            // Nothing else in the shipped profile moves with it.
            let off = Inkling2DSparkDrafter.defaultMTPConfig()
            #expect(on.maxDraftTokens == off.maxDraftTokens)
            #expect(on.maxSpeculativeBatch == off.maxSpeculativeBatch)
            #expect(on.fixedDraftTokens == off.fixedDraftTokens)
            #expect(on.acceptance == off.acceptance)
            #expect(on.costCurveDepthSelection == off.costCurveDepthSelection)
        }

        @Test func onlySampledRowsOutsideAGrammarCouple() {
            let sampled = CBv2SamplingParams(temperature: 0.5, seed: 3)
            #expect(EngineLoopV2.mtpCouplesProposals(sampled, freeSpan: false, constrained: false))
            #expect(!EngineLoopV2.mtpCouplesProposals(sampled, freeSpan: true, constrained: false))
            #expect(!EngineLoopV2.mtpCouplesProposals(sampled, freeSpan: false, constrained: true))
            for greedy in [
                CBv2SamplingParams(temperature: 0), CBv2SamplingParams(temperature: 1e-9),
            ] {
                #expect(
                    !EngineLoopV2.mtpCouplesProposals(greedy, freeSpan: false, constrained: false))
            }
        }
    }
}
