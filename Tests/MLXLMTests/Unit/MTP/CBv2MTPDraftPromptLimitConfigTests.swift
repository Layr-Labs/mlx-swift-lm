import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// `CBv2MTPConfig.maxDraftPromptTokens`: nil by default, a value below
    /// one means no limit, and a prompt is over the limit only above it.
    struct CBv2MTPDraftPromptLimitConfigTests {
        @Test func noLimitByDefault() {
            let config = CBv2MTPConfig()
            #expect(config.maxDraftPromptTokens == nil)
            #expect(!config.promptExceedsDraftLimit(Int.max))
        }

        @Test(arguments: [0, -1, Int.min])
        func aValueBelowOneMeansNoLimit(_ value: Int) {
            let config = CBv2MTPConfig(maxDraftPromptTokens: value)
            #expect(config.maxDraftPromptTokens == nil)
            #expect(!config.promptExceedsDraftLimit(Int.max))
            var assigned = CBv2MTPConfig()
            assigned.maxDraftPromptTokens = value
            #expect(!assigned.promptExceedsDraftLimit(Int.max))
        }

        @Test func onlyAPromptAboveTheLimitIsOver() {
            let config = CBv2MTPConfig(maxDraftPromptTokens: 4096)
            #expect(config.maxDraftPromptTokens == 4096)
            #expect(!config.promptExceedsDraftLimit(0))
            #expect(!config.promptExceedsDraftLimit(4096))
            #expect(config.promptExceedsDraftLimit(4097))
        }

        /// The plan-time gate holds a plan at depth zero only when it has
        /// rows and every one is over the limit.
        @Test func aCohortIsOverOnlyWhenEveryRowIsOver() {
            let config = CBv2MTPConfig(maxDraftPromptTokens: 4096)
            #expect(!config.cohortExceedsDraftLimit(promptTokenCounts: []))
            #expect(config.cohortExceedsDraftLimit(promptTokenCounts: [4097]))
            #expect(config.cohortExceedsDraftLimit(promptTokenCounts: [4097, 30_000, 5000]))
            #expect(!config.cohortExceedsDraftLimit(promptTokenCounts: [4097, 4096]))
            #expect(!config.cohortExceedsDraftLimit(promptTokenCounts: [100, 30_000, 5000]))
            #expect(!config.cohortExceedsDraftLimit(promptTokenCounts: [4096]))
        }

        @Test(arguments: [nil, 0, -1] as [Int?])
        func noLimitNeverHoldsACohort(_ value: Int?) {
            let config = CBv2MTPConfig(maxDraftPromptTokens: value)
            #expect(!config.cohortExceedsDraftLimit(promptTokenCounts: [Int.max]))
            #expect(!config.cohortExceedsDraftLimit(promptTokenCounts: [Int.max, Int.max]))
        }
    }
}
