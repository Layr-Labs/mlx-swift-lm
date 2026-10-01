import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// Tests of the `ModelContainer` accessors and convenience methods with
    /// a stub model. The tests that prepare input or generate tokens need
    /// MLX arrays and are in `Kernel/Common/ModelContainerGenerationTests.swift`.
    @Suite
    struct ModelContainerAccessTests {

        private func makeContainer(
            configuration: ModelConfiguration = ModelConfiguration(id: "test/stub")
        ) -> ModelContainer {
            ModelContainer(
                context: ModelContext(
                    configuration: configuration, model: StubLanguageModel(),
                    processor: StandInUserInputProcessor(), tokenizer: ScalarTokenizer()))
        }

        @Test func accessorsReturnTheContextValues() async {
            let container = makeContainer()
            #expect(await container.configuration.name == "test/stub")
            #expect(await container.processor is StandInUserInputProcessor)
            #expect(await container.tokenizer.eosTokenId == 2)
        }

        @Test func performRunsTheActionOnTheContext() async throws {
            let container = makeContainer()
            let name = await container.perform { context in context.configuration.name }
            #expect(name == "test/stub")

            let isStub = await container.perform { context in
                context.model is StubLanguageModel
            }
            #expect(isStub)

            await #expect(throws: StubModelError.self) {
                try await container.perform { _ -> Int in throw StubModelError() }
            }
        }

        @Test func performPassesTheValues() async {
            let container = makeContainer()
            let tokens = await container.perform(values: "ab") { context, text in
                context.tokenizer.encode(text: text, addSpecialTokens: false)
            }
            #expect(tokens == [97, 98])

            final class NonSendableBox {
                let text = "c"
            }
            let more = await container.perform(nonSendable: NonSendableBox()) { context, box in
                context.tokenizer.encode(text: box.text, addSpecialTokens: true)
            }
            #expect(more == [1, 99])
        }

        /// The model-and-tokenizer forms of `perform` are deprecated. The
        /// test keeps them covered until they are removed.
        @Test func deprecatedPerformFormsStillWork() async {
            let container = makeContainer()
            let count = await container.perform { (model: any LanguageModel, tokenizer: any Tokenizer) in
                tokenizer.encode(text: "xyz", addSpecialTokens: false).count
            }
            #expect(count == 3)

            let decoded = await container.perform(values: [100, 101]) {
                (model: any LanguageModel, tokenizer: any Tokenizer, ids: [Int]) in
                tokenizer.decode(tokenIds: ids)
            }
            #expect(decoded == "de")
        }

        @Test func updateChangesTheContext() async {
            let container = makeContainer()
            await container.update { context in
                context.configuration.defaultPrompt = "changed"
                context.configuration.toolCallFormat = .gemma
            }
            #expect(await container.configuration.defaultPrompt == "changed")
            #expect(await container.configuration.toolCallFormat == .gemma)
        }

        @Test func directoriesOfALocalModel() async throws {
            let model = URL(fileURLWithPath: "/models/tiny")
            let tokenizer = URL(fileURLWithPath: "/tokenizers/tiny")

            let sameFolder = makeContainer(configuration: ModelConfiguration(directory: model))
            #expect(try await sameFolder.modelDirectory == model)
            #expect(try await sameFolder.tokenizerDirectory == model)

            let otherFolder = makeContainer(
                configuration: ModelConfiguration(
                    directory: model, tokenizerSource: .directory(tokenizer)))
            #expect(try await otherFolder.tokenizerDirectory == tokenizer)
        }

        @Test func directoriesOfAnUnresolvedModelThrow() async {
            let container = makeContainer()
            await #expect(
                throws: ModelConfiguration.DirectoryError.unresolvedModelDirectory("test/stub")
            ) {
                _ = try await container.modelDirectory
            }
            await #expect(
                throws: ModelConfiguration.DirectoryError.unresolvedModelDirectory("test/stub")
            ) {
                _ = try await container.tokenizerDirectory
            }

            let remoteTokenizer = makeContainer(
                configuration: ModelConfiguration(
                    directory: URL(fileURLWithPath: "/models/tiny"),
                    tokenizerSource: .id("org/tokenizer")))
            await #expect(
                throws: ModelConfiguration.DirectoryError.unresolvedTokenizerDirectory(
                    "org/tokenizer")
            ) {
                _ = try await remoteTokenizer.tokenizerDirectory
            }
        }

        @Test func encodeAndDecodeUseTheTokenizer() async {
            let container = makeContainer()
            #expect(await container.encode("hi") == [1, 104, 105])
            #expect(await container.decode(tokenIds: [1, 104, 105]) == "<1>hi")
        }

        /// `decode(tokens:)` and `applyChatTemplate(messages:)` are
        /// deprecated. The test keeps them covered until they are removed.
        @Test func deprecatedConvenienceMethodsStillWork() async throws {
            let container = makeContainer()
            #expect(await container.decode(tokens: [104, 105]) == "hi")
            let tokens = try await container.applyChatTemplate(messages: [
                ["role": "user", "content": "a"]
            ])
            // "user:a" without special tokens, then 0 tools.
            #expect(tokens == [117, 115, 101, 114, 58, 97, 0])
        }

        @Test func prepareForwardsTheProcessorError() async {
            let container = makeContainer()
            do {
                _ = try await container.prepare(input: UserInput(prompt: "x"))
                Issue.record("expected an error")
            } catch UserInputError.notImplemented {
                // Expected: the stand-in processor does not prepare input.
            } catch {
                Issue.record("unexpected error \(error)")
            }
        }
    }
}
