import MLX
import MLXLMCommon
import MLXNN
import Testing

private let iteratorStateKey = LMOutput.Key<Int>("tests.tokenIterator.prefillState")

private final class PrefillStateModel: Module, LanguageModel {
    private(set) var decodeStates: [Int?] = []

    private func output() -> LMOutput {
        var state = LMOutput.State()
        state[iteratorStateKey] = 23
        return .init(logits: MLXArray([Float(0), 1]).reshaped([1, 1, 2]), state: state)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .logits(output())
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        decodeStates.append(state?[iteratorStateKey])
        return output()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        output().logits
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

@Test("TokenIterator forwards state returned by logits prefill into first decode")
func tokenIteratorCarriesPrefillStateIntoDecode() throws {
    let model = PrefillStateModel()
    var iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray([Int32(0)])), model: model,
        parameters: GenerateParameters(maxTokens: 1, temperature: 0))

    _ = iterator.next()
    #expect(model.decodeStates == [23])
}
