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
    }
}
