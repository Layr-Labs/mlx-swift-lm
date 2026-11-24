import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// `DARKBLOOM_CBV2_MTP_COST_CURVE` can only turn the cost-curve depth
    /// choice off (`CBv2MTPRoundDriver.costCurveDepthSelection`). A family whose
    /// configuration has not opted in never gets it.
    struct CBv2MTPCostCurveSwitchTests {
        @Test(arguments: [nil, "1", "0", "true", ""] as [String?])
        func familyWithoutTheOptInNeverUsesTheCostCurve(environment: String?) {
            #expect(
                !CBv2MTPRoundDriver.costCurveDepthSelection(
                    configured: false, environment: environment))
        }

        @Test func optedInFamilyUsesItUnlessTheSwitchIsZero() {
            for environment in [nil, "1", "true", ""] as [String?] {
                #expect(
                    CBv2MTPRoundDriver.costCurveDepthSelection(
                        configured: true, environment: environment))
            }
            #expect(!CBv2MTPRoundDriver.costCurveDepthSelection(configured: true, environment: "0"))
        }
    }
}
