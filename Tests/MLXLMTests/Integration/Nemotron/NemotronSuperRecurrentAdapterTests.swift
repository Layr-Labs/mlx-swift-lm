import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The adapter's single-request `LMInput` path reads only the tokens, so it
/// refuses images and video instead of dropping them.
@Suite(.tags(.integration))
struct NemotronSuperRecurrentAdapterTests {
    @Test func preparedInputRefusesMediaItWouldDrop() throws {
        let adapter = NemotronSuperRecurrentAdapter(try NemotronSuperMTPFixture.target())
        let text = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3]))
        let pixels = MLXArray.zeros([1, 1, 4, 4])
        for input in [
            LMInput(text: text, image: .init(pixels: pixels)),
            LMInput(text: text, video: .init(pixels: pixels)),
        ] {
            #expect(throws: NemotronHModel.SuperMTPTextError.media) {
                try adapter.prepare(input, cache: [], state: nil, prefill: PrefillParameters())
            }
        }
        // Text alone still leaves its tokens to the iterator.
        guard
            case .tokens(let remaining) = try adapter.prepare(
                LMInput(text: text), cache: [], state: nil, prefill: PrefillParameters())
        else {
            Issue.record("a three-token prompt prepares to tokens")
            return
        }
        #expect(remaining.tokens.asArray(Int32.self) == [1, 2, 3])
    }
}
