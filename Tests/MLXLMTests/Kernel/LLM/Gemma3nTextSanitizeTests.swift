import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression test for issue #195, defect 4: `sanitize(weights:)` must cut
/// a padded embedding table.
///
/// Copied from `Gemma3nTextForwardPassTests` in PR #184 without the
/// `withKnownIssue` block.
@Suite
struct Gemma3nTextSanitizeTests {

    typealias Tiny = Gemma3nTextTinyModel

    /// A Hugging Face checkpoint keys the text model as
    /// `model.language_model.*`. `sanitize(weights:)` cuts an embedding
    /// table with more rows than the vocabulary down to the vocabulary, and
    /// keeps the first rows.
    @Test func sanitizeCutsAPaddedEmbeddingTable() throws {
        let model = try Tiny.make()
        let padded = MLXArray(0 ..< Int32(70 * 32)).reshaped(70, 32).asType(.float32)
        let sanitized = model.sanitize(weights: [
            "model.language_model.embed_tokens.weight": padded
        ])
        let table = try #require(sanitized["language_model.embed_tokens.weight"])
        try #require(table.dim(0) == Tiny.vocabularySize, "embedding rows")
        #expect(Tiny.maxAbsDifference(table, padded[0 ..< Tiny.vocabularySize]) == 0)
    }
}
