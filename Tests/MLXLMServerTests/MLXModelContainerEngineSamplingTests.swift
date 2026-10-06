// Copyright © 2026 Eigen Labs Inc.

import Foundation
import Testing

@testable import MLXLMServer

struct MLXModelContainerEngineSamplingTests {
    private func request(seed: UInt64? = nil, bias: [String: Float]? = nil)
        -> OpenAIChatCompletionRequest
    {
        .init(model: "fixture", messages: [.init(role: .user, content: .text("hello"))],
              seed: seed, logitBias: bias)
    }

    @Test func absentControlsAndEmptyBiasAreNoOps() throws {
        try MLXModelContainerEngine.validateSamplingControls(request())
        try MLXModelContainerEngine.validateSamplingControls(request(bias: [:]))
    }

    @Test(arguments: [UInt64(0), 1, UInt64.max])
    func seedIsRejectedRatherThanSilentlyIgnored(_ seed: UInt64) {
        #expect(throws: MLXModelContainerEngineError.unsupportedSamplingControl("seed")) {
            try MLXModelContainerEngine.validateSamplingControls(request(seed: seed))
        }
    }

    @Test func populatedBiasIsRejectedIncludingZeroBias() {
        for bias: [String: Float] in [["0": 100], ["23": -100], ["9": 0]] {
            #expect(throws: MLXModelContainerEngineError.unsupportedSamplingControl("logit_bias")) {
                try MLXModelContainerEngine.validateSamplingControls(request(bias: bias))
            }
        }
    }

    @Test func sharedWireContractStillCarriesNativeControls() throws {
        let original = request(seed: 17, bias: ["42": -12.5])
        let decoded = try JSONDecoder().decode(OpenAIChatCompletionRequest.self,
            from: JSONEncoder().encode(original))
        #expect(decoded.seed == 17)
        #expect(decoded.logitBias == ["42": -12.5])
        #expect(MLXModelContainerEngineError.unsupportedSamplingControl("seed").status.code == 400)
        #expect(MLXModelContainerEngineError.nativeGenerationRequired("native required").status.code == 501)
    }
}
