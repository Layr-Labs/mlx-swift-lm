import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Plain Super decode must keep the engine's chained fast path: step N + 1 is
/// built on step N's lazy tokens while N runs, so host graph construction
/// overlaps GPU execution. Synthetic weights; only step kinds are checked.
@Suite(.tags(.integration))
struct NemotronSuperDecodePipelineTests {
    static func usage(_ stream: AsyncStream<CBv2Event>) async -> CBv2Usage? {
        var usage: CBv2Usage?
        for await event in stream {
            if case .finished(_, let finished) = event { usage = finished }
        }
        return usage
    }

    @Test(.timeLimit(.minutes(2)), arguments: [1, 3])
    func plainDecodeChainsSuccessiveSteps(rows: Int) async throws {
        let model = try NemotronSuperBatchFixture.model(seed: 9, dtype: .bfloat16, products: .tiled)
        let engine = try NemotronSuperBatchedEngineTests.engine(model, rows: rows)
        let streams = try (0 ..< rows).map { index in
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(UInt64(index + 1)),
                    promptTokens: NemotronSuperBatchFixture.tokens(5, seed: index).map(Int.init),
                    sampling: .init(temperature: 0), maxTokens: 24))
        }
        for stream in streams {
            let timing = try #require(await Self.usage(stream)).timing
            print(
                "PIPELINE super rows=\(rows) decodeSteps=\(timing.decodeSteps) "
                    + "chained=\(timing.chainedDecodeSteps)")
            // Joins and the final step may break the chain; steady decode may not.
            #expect(timing.decodeSteps >= 20)
            #expect(timing.chainedDecodeSteps + 4 >= timing.decodeSteps)
        }
        await engine.shutdown()
    }
}
