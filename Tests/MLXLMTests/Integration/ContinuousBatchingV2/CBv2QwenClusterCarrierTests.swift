import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Qwen cluster carrier local refusal", .tags(.integration))
struct CBv2QwenClusterCarrierTests {
    @Test func encodedCarrierCannotFallThroughToLocalModelOrAllocateKV() async throws {
        let harness = CBv2SchedHarness()
        var request = CBv2Request(id: .init(1), promptTokens: [1], maxTokens: 2)
        #expect(request.clusterQwenMedia == nil)
        for bytes in [Data(), Data([1, 2, 3])] {
            request.clusterQwenMedia = bytes
            #expect(
                throws: CBv2MultimodalError.unsupportedBackend(
                    "encoded cluster media requires the native cluster worker")
            ) {
                try harness.engine.submit(request)
            }
            #expect(harness.backend.makeCalls == 0 && harness.backend.liveStates == 0)
            #expect(harness.model.forwardShapes.isEmpty)
        }
        await harness.engine.shutdown()
    }
}
