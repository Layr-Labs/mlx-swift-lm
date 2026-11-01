// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Nemotron 3.5 decode command buffers")
struct NemotronH35DecodeCommandBufferTests {
    @Test func declaresDecodeLimitsOnlyThroughTheSteppableAdapter() throws {
        let adapter = CBv2SteppableLanguageModelAdapter(
            try NemotronH35BackendOracle.tinyModel(dtype: .float32))
        #expect(adapter.decodeCommandBufferLimits == NemotronH35DecodeCommandBuffers.limits)
        // MTP rounds keep MLX's defaults.
        #expect(adapter.speculativeCommandBufferLimits == nil)
        #expect(NemotronH35DecodeCommandBuffers.limits.operations == 100)
        #expect(NemotronH35DecodeCommandBuffers.limits.megabytes == 1 << 20)
    }

    @Test func decodeLimitsKeepExactBytes() throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .float32)
        let tokens = MLXArray((0 ..< 24).map { Int32($0 % 32) }).reshaped(1, -1)
        let expected = model(tokens, cache: nil)
        eval(expected)
        let actual = GPU.withCommandBufferLimits(NemotronH35DecodeCommandBuffers.limits) {
            let logits = model(tokens, cache: nil)
            eval(logits)
            return logits
        }
        #expect(expected.asData(access: .copy).data == actual.asData(access: .copy).data)
    }
}
