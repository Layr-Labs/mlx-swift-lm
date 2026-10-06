import Foundation
import MLXLMCommon
import Testing

@Suite("DiffusionGemma native generation configuration")
struct DiffusionGemmaGenerationConfigurationTests {
    private func decode(_ json: String) throws -> DiffusionGemmaGenerationConfiguration {
        try JSONDecoder().decode(DiffusionGemmaGenerationConfiguration.self, from: Data(json.utf8))
    }

    @Test func referenceDefaultsAndDescendingTemperature() throws {
        let config = try decode("{}")
        #expect(config.maxNewTokens == 256)
        #expect(config.maxDenoisingSteps == 48)
        #expect(config.sampler.entropyBound == 0.1)
        #expect(try config.temperature(remainingStep: 48) == 0.8)
        #expect(try config.temperature(remainingStep: 1) > 0.4)
        #expect(throws: DiffusionGemmaGenerationError.self) {
            try config.temperature(remainingStep: 0)
        }
        #expect(throws: DiffusionGemmaGenerationError.self) {
            try config.temperature(remainingStep: 49)
        }
    }

    @Test func everyNondefaultSemanticFieldSurvivesRoundTrip() throws {
        let config = try decode(
            """
            {"max_new_tokens":519,"max_length":9021,"max_denoising_steps":17,
             "sampler_config":{"_cls_name":"EntropyBoundSamplerConfig","entropy_bound":0.23},
             "t_min":0.15,"t_max":0.91,"stability_threshold":3,"confidence_threshold":0.02,
             "bos_token_id":4,"pad_token_id":5,"eos_token_id":[0,6,106]}
            """)
        let roundTrip = try JSONDecoder().decode(
            DiffusionGemmaGenerationConfiguration.self, from: JSONEncoder().encode(config))
        #expect(roundTrip == config)
        #expect(roundTrip.maxNewTokens == 519)
        #expect(roundTrip.maxLength == 9021)
        #expect(roundTrip.eosTokenIds == [0, 6, 106])
        #expect(roundTrip.sampler.entropyBound == 0.23)
    }

    @Test func scalarZeroAndAbsentEOSRemainDistinct() throws {
        #expect(try decode(#"{"eos_token_id":0}"#).eosTokenIds == [0])
        #expect(try decode(#"{"eos_token_id":[]}"#).eosTokenIds == [])
        #expect(try decode(#"{"eos_token_id":null}"#).eosTokenIds == nil)
        #expect(try decode("{}").eosTokenIds == nil)
    }

    @Test(arguments: [
        #"{"max_new_tokens":0}"#, #"{"max_denoising_steps":0}"#,
        #"{"max_length":-1}"#, #"{"t_min":0.8,"t_max":0.4}"#,
        #"{"stability_threshold":-1}"#, #"{"confidence_threshold":0}"#,
        #"{"eos_token_id":-1}"#,
        #"{"sampler_config":{"_cls_name":"EntropyBoundSamplerConfig","entropy_bound":0}}"#,
        #"{"sampler_config":{"_cls_name":"ConfidenceThresholdSampler","entropy_bound":0.1}}"#,
        #"{"sampler_config":{"_cls_name":"EntropyBoundSamplerConfig","entropy_bound":0.1,"temperature":0.5}}"#,
        #"{"top_p":0.9}"#, #"{"cache_implementation":"static"}"#,
    ])
    func rejectsInvalidOrUnimplementedControls(_ json: String) throws {
        #expect(throws: (any Error).self) { try decode(json) }
    }

    @Test func rejectsNonfiniteProgrammaticValues() throws {
        #expect(throws: DiffusionGemmaGenerationError.self) {
            try DiffusionGemmaGenerationConfiguration(minimumTemperature: .nan)
        }
        #expect(throws: DiffusionGemmaGenerationError.self) {
            try DiffusionGemmaGenerationConfiguration.EntropyBoundSampler(entropyBound: .infinity)
        }
    }

    @Test func nativeOutputLengthPrecedenceIsActuallyApplied() throws {
        #expect(try decode(#"{"max_length":1000}"#).outputTokenLimit(promptTokenCount: 100) == 900)
        #expect(
            try decode(#"{"max_new_tokens":256,"max_length":1000}"#).outputTokenLimit(
                promptTokenCount: 100) == 900)
        #expect(
            try decode(#"{"max_new_tokens":64,"max_length":1000}"#).outputTokenLimit(
                promptTokenCount: 100) == 64)
        #expect(throws: DiffusionGemmaGenerationError.self) {
            try decode(#"{"max_length":100}"#).outputTokenLimit(promptTokenCount: 100)
        }
    }
}
