import Foundation
@testable import MLXLMServer
import Testing

@Suite("OpenAI Chat sampling wire fields")
struct OpenAIChatSamplingFieldsTests {
    @Test func seedAndLogitBiasDecodeAndRoundTrip() throws {
        let input = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}],"seed":18446744073709551614,"logit_bias":{"17":-100,"42":1.5}}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatCompletionRequest.self, from: input)
        #expect(request.seed == UInt64.max - 1)
        #expect(request.logitBias == ["17": -100, "42": 1.5])

        let roundTrip = try JSONDecoder().decode(
            OpenAIChatCompletionRequest.self, from: JSONEncoder().encode(request))
        #expect(roundTrip.seed == request.seed)
        #expect(roundTrip.logitBias == request.logitBias)
    }

    @Test func omittedFieldsRemainNil() throws {
        let input = Data(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatCompletionRequest.self, from: input)
        #expect(request.seed == nil)
        #expect(request.logitBias == nil)
    }
}
