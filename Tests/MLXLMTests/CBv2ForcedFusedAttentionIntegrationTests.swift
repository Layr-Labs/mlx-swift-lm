import MLX
import MLXRandom
import XCTest

#if canImport(Metal)
    import Metal
#endif

@testable import MLXLMCommon

final class CBv2ForcedFusedAttentionIntegrationTests: XCTestCase {
    private let headDim = 256
    private let queryHeads = 16
    private let kvHeads = 2
    private let historyLength = 3

    private var scale: Float { 1.0 / Float(headDim).squareRoot() }

    private var qualifiedKind: CBv2LayerKind {
        CBv2LayerKind(
            attention: .full,
            headDim: headDim,
            kvHeads: kvHeads,
            queryHeads: queryHeads,
            attentionExecutionQualification: CBv2AttentionExecutionQualification(
                architecture: .qwenLike))
    }

    private struct Fixture {
        var queries: MLXArray
        var historyKeys: MLXArray
        var historyValues: MLXArray
        var newKeys: MLXArray
        var newValues: MLXArray
    }

    private struct Run {
        var output: MLXArray
        var observations: [CBv2AttentionExecutionObservation]
        var forcedFusedExecutions: Int
    }

    private func requireMetal() throws {
        #if canImport(Metal)
            guard MTLCreateSystemDefaultDevice() != nil else {
                throw XCTSkip("Metal device unavailable")
            }
        #else
            throw XCTSkip("Metal framework unavailable")
        #endif
    }

    private func fixture(queryLength: Int, seed: UInt64) -> Fixture {
        MLXRandom.seed(seed)
        return Fixture(
            queries: MLXRandom.normal(
                [1, queryHeads, queryLength, headDim], dtype: .bfloat16),
            historyKeys: MLXRandom.normal(
                [1, kvHeads, historyLength, headDim], dtype: .bfloat16),
            historyValues: MLXRandom.normal(
                [1, kvHeads, historyLength, headDim], dtype: .bfloat16),
            newKeys: MLXRandom.normal(
                [1, kvHeads, queryLength, headDim], dtype: .bfloat16),
            newValues: MLXRandom.normal(
                [1, kvHeads, queryLength, headDim], dtype: .bfloat16))
    }

    private func run(
        _ fixture: Fixture,
        policy: CBv2AttentionExecutionPolicy
    ) -> Run {
        let queryLength = fixture.queries.dim(2)
        let row = CBv2FullSequenceKV(
            promptLength: historyLength + queryLength,
            maxLength: historyLength + queryLength + 8,
            kvHeads: kvHeads,
            headDim: headDim)
        _ = row.update(keys: fixture.historyKeys, values: fixture.historyValues)

        let cache = CBv2LayerCache(
            layerIndex: 0,
            kind: qualifiedKind,
            rows: [row],
            attentionExecutionPolicy: policy)
        var observations: [CBv2AttentionExecutionObservation] = []
        var forcedFusedExecutions = 0
        cache.attentionExecutionObserver = { observations.append($0) }
        cache.forcedFusedExecutionObserver = { forcedFusedExecutions += 1 }

        let output = cache.updateAndAttend(
            queries: fixture.queries,
            keys: fixture.newKeys,
            values: fixture.newValues,
            scale: scale,
            sinks: nil)
        return Run(
            output: output,
            observations: observations,
            forcedFusedExecutions: forcedFusedExecutions)
    }

    private func assertWithinBF16Tolerance(
        _ actual: MLXArray,
        _ expected: MLXArray,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let a = actual.asType(.float32)
        let b = expected.asType(.float32)
        let delta = abs(a - b)
        let maxDelta = delta.max().item(Float.self)
        let meanDelta = delta.mean().item(Float.self)
        XCTAssertLessThanOrEqual(
            maxDelta,
            0.02,
            "max |delta| = \(maxDelta), mean |delta| = \(meanDelta)",
            file: file,
            line: line)
        XCTAssertLessThanOrEqual(
            meanDelta,
            0.001,
            "max |delta| = \(maxDelta), mean |delta| = \(meanDelta)",
            file: file,
            line: line)
    }

    func testMetalBF16ForcedFusedWithHistoryMatchesFallbackAndRemainsCausal() throws {
        try requireMetal()

        Device.withDefaultDevice(.gpu) {
            let fusedPolicy = CBv2AttentionExecutionPolicy(
                control: .fused,
                fallbackQueryBlockSize: 512)
            let fallbackPolicy = CBv2AttentionExecutionPolicy(
                control: .fallback,
                fallbackQueryBlockSize: 512)

            let vectorFixture = fixture(queryLength: 4, seed: 0xD256)
            let fused = run(vectorFixture, policy: fusedPolicy)
            let fallback = run(vectorFixture, policy: fallbackPolicy)
            eval(fused.output, fallback.output)

            XCTAssertEqual(fused.observations.first?.route, .forcedFused)
            XCTAssertEqual(fused.forcedFusedExecutions, 1)
            XCTAssertGreaterThan(vectorFixture.historyKeys.dim(2) + 4, 4)
            assertWithinBF16Tolerance(fused.output, fallback.output)

            let fullFixture = fixture(queryLength: 9, seed: 0xD900)
            let fullFused = run(fullFixture, policy: fusedPolicy)
            let fullFallback = run(fullFixture, policy: fallbackPolicy)
            eval(fullFused.output, fullFallback.output)
            XCTAssertEqual(fullFused.observations.first?.route, .forcedFused)
            XCTAssertEqual(fullFused.forcedFusedExecutions, 1)
            XCTAssertGreaterThan(fullFixture.historyKeys.dim(2) + 9, 9)
            assertWithinBF16Tolerance(fullFused.output, fullFallback.output)

            let futureIndex = 3
            var perturbed = vectorFixture
            perturbed.newKeys = concatenated(
                [
                    vectorFixture.newKeys[0..., 0..., 0 ..< futureIndex, 0...],
                    vectorFixture.newKeys[0..., 0..., futureIndex..., 0...] + 32,
                ],
                axis: 2)
            perturbed.newValues = concatenated(
                [
                    vectorFixture.newValues[0..., 0..., 0 ..< futureIndex, 0...],
                    vectorFixture.newValues[0..., 0..., futureIndex..., 0...] - 17,
                ],
                axis: 2)
            let perturbedRun = run(perturbed, policy: fusedPolicy)
            eval(perturbedRun.output)
            XCTAssertEqual(perturbedRun.forcedFusedExecutions, 1)
            XCTAssertTrue(
                arrayEqual(
                    fused.output[0..., 0..., 0 ..< futureIndex, 0...],
                    perturbedRun.output[0..., 0..., 0 ..< futureIndex, 0...]
                ).item(Bool.self),
                "future K/V must not affect earlier causal query rows")

            for queryLength in [5, 8] {
                let unsupported = fixture(
                    queryLength: queryLength,
                    seed: UInt64(0xD500 + queryLength))
                let requestedFused = run(unsupported, policy: fusedPolicy)
                let explicitFallback = run(unsupported, policy: fallbackPolicy)
                eval(requestedFused.output, explicitFallback.output)

                XCTAssertEqual(requestedFused.observations.first?.route, .fallback)
                XCTAssertEqual(requestedFused.forcedFusedExecutions, 0)
                assertWithinBF16Tolerance(requestedFused.output, explicitFallback.output)
            }
        }
    }
}
