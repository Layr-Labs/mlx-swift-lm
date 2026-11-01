import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private final class GreedyMetadataConstraint: CBv2TokenConstraint, @unchecked Sendable {
    let mode: CBv2TokenConstraintMode = .required
    let maxTokens = 1
    let fallbackTokenID = 1
    let initialState = 0

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
        state == 0 && remainingTokens > 0 ? [1] : []
    }

    func nextState(state: Int, tokenID: Int) -> Int? {
        state == 0 && tokenID == 1 ? 1 : nil
    }
}

extension KernelTests {
    @Suite(.serialized)
    struct CBv2GreedyMetadataTests {
        private func row(
            _ id: UInt64 = 1,
            params: CBv2SamplingParams = .init(temperature: 0),
            output: [Int] = []
        ) -> CBv2SamplerRow {
            CBv2SamplerRow(
                id: CBv2RequestID(id), params: params, promptTokens: [0, 1],
                outputTokens: output)
        }

        private func sameBits(_ actual: MLXArray, _ expected: MLXArray) -> Bool {
            actual.shape == expected.shape && actual.dtype == expected.dtype
                && actual.asData(access: .copy).data == expected.asData(access: .copy).data
        }

        private func isSet<T>(_ value: T?) -> Bool {
            if case .some = value { return true }
            return false
        }

        @Test func testIdentityGreedyRetainsExactFloatConversion() {
            let values: [Float] = [-0.0, 0.0, 1.25, -2.5, .infinity, -.infinity, .nan, 7]
            for dtype in [DType.float16, .bfloat16, .float32] {
                let pipeline = LogitsPipelineV2(vocabSize: values.count)
                pipeline.setRows([row()])
                let logits = MLXArray(values, [1, values.count]).asType(dtype)
                let result = pipeline.process(logits)
                #expect(sameBits(result.sampling, logits.asType(.float32)))
                #expect(!isSet(result.rawLogprobs))
                #expect(pipeline.logprobBuildCount == 0)
            }
        }

        @Test func testGreedyReconfigurationClearsEarlierPenaltyAndLogprobState() {
            let pipeline = LogitsPipelineV2(vocabSize: 4)
            let previous = row(
                params: .init(
                    temperature: 0.5, repetitionPenalty: 2, repetitionContextSize: 2,
                    frequencyPenalty: 0.25, presencePenalty: 0.5, logitBias: [1: 4], topLogprobs: 2),
                output: [1, 2])
            pipeline.setRows([previous])
            let prior = pipeline.process(MLXArray([Float(1), 2, 3, 4], [1, 4]))
            #expect(isSet(prior.rawLogprobs))
            pipeline.commit(sampledTokens: MLXArray([Int32(2)]))

            let ignoredFilters = CBv2SamplingParams(
                temperature: 0, topP: 0, topK: 1, minP: 1,
                repetitionPenalty: 2, repetitionContextSize: 0)
            pipeline.setRows([row(2, params: ignoredFilters), row(3, output: [1, 1, 1])])
            let logits = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8], [2, 4])
            pipeline.commit(sampledTokens: MLXArray([Int32(1), 2]))
            let result = pipeline.process(logits)
            #expect(sameBits(result.sampling, logits))
            #expect(!isSet(result.rawLogprobs))
            #expect(pipeline.allGreedy)
            #expect(!pipeline.wantsLogprobs)
            #expect(pipeline.logprobBuildCount == 1)
        }

        @Test func testReturningToTransformsMatchesFreshStateBitForBit() throws {
            let reused = LogitsPipelineV2(vocabSize: 4)
            reused.setRows([row()])
            let transformed = row(
                params: .init(
                    temperature: 0.5, repetitionPenalty: 2, repetitionContextSize: 2,
                    frequencyPenalty: 0.25, presencePenalty: 0.5, logitBias: [1: 1], topLogprobs: 2),
                output: [1, 2])
            let fresh = LogitsPipelineV2(vocabSize: 4)
            reused.setRows([transformed])
            fresh.setRows([transformed])
            let logits = MLXArray([Float(1), 2, -3, 4], [1, 4])
            for token in [Int32(2), 3, 2] {
                let a = reused.process(logits)
                let b = fresh.process(logits)
                #expect(sameBits(a.sampling, b.sampling))
                let aLogprobs = try #require(a.rawLogprobs)
                let bLogprobs = try #require(b.rawLogprobs)
                #expect(sameBits(aLogprobs, bLogprobs))
                reused.commit(sampledTokens: MLXArray([token]))
                fresh.commit(sampledTokens: MLXArray([token]))
            }
            #expect(!reused.allGreedy)
            #expect(reused.wantsLogprobs)
        }

        @Test func testOneNonIdentityRowStillAppliesEveryRequestedTransform() {
            let rows = [
                row(1),
                row(2, params: .init(temperature: 0, logitBias: [1: 2])),
                row(3, params: .init(temperature: 0, repetitionPenalty: 2), output: [1]),
                row(4, params: .init(temperature: 0, frequencyPenalty: 0.5), output: [1, 1]),
                row(5, params: .init(temperature: 0, presencePenalty: 0.5), output: [1, 1]),
                row(6, params: .init(temperature: 2)),
            ]
            let pipeline = LogitsPipelineV2(vocabSize: 4)
            pipeline.setRows(rows)
            let perRow: [Float] = [0, 4, 2, -2]
            let values = Array(repeating: perRow, count: rows.count).flatMap { $0 }
            let logits = MLXArray(values, [rows.count, 4])
            let result = pipeline.process(logits).sampling.asArray(Float.self)
            #expect([1, 5, 9, 13, 17, 21].map { result[$0] } == [4, 6, 2, 3, 3.5, 2])
        }

        @Test func testIdentityGreedyStillEnforcesRequiredTokenConstraint() {
            let sampler = CBv2DefaultSampler(fallbackSeed: 7)
            var constrained = row(77)
            constrained.tokenConstraint = GreedyMetadataConstraint()
            constrained.maxTokens = 1
            let token = sampler.sample(
                logits: MLXArray([Float(0), 1, 2, 100], [1, 4]),
                params: [constrained.params], requestIDs: [constrained.id],
                stepIndex: 0, pendingSampledTokens: nil, rowContext: { [constrained] })
            #expect(token.asArray(Int32.self) == [1])
        }

        @Test func testGreedyToMixedSamplingPreservesHistoryAndRNGSteps() {
            let reused = SamplerV2(fallbackSeed: 3)
            reused.setRows([row(10, output: [1, 2])])
            #expect(reused.stepIndices == [2])
            let greedyLogits = MLXArray([Float(0), 1, 2, 3], [1, 4])
            #expect(reused.sample(from: greedyLogits).asArray(Int32.self) == [3])
            reused.commit()
            #expect(reused.stepIndices == [3])

            let mixed = [
                row(10, output: [1, 2, 3]),
                row(11, params: .init(temperature: 0.7, seed: 1729), output: [1, 2, 1, 2]),
            ]
            let fresh = SamplerV2(fallbackSeed: 3)
            reused.setRows(mixed)
            fresh.setRows(mixed)
            let logits = MLXArray([Float(0), 1, 2, 3, 2, 1, 4, 3], [2, 4])
            for _ in 0 ..< 8 {
                #expect(reused.stepIndices == fresh.stepIndices)
                let a = reused.sample(from: logits).asArray(Int32.self)
                let b = fresh.sample(from: logits).asArray(Int32.self)
                #expect(a == b)
                reused.commit()
                fresh.commit()
            }
        }

        @Test func testMTPCommitStillRebuildsConfirmedRowContext() {
            let sampler = CBv2DefaultSampler(fallbackSeed: 1)
            var current = row(9)
            var rebuilds = 0
            let logits = MLXArray([Float(0), 1, 2, 3], [1, 4])
            func sample() -> [Int32] {
                sampler.sample(
                    logits: logits, params: [current.params], requestIDs: [current.id],
                    stepIndex: current.outputTokens.count, pendingSampledTokens: nil,
                    rowContext: {
                        rebuilds += 1
                        return [current]
                    }
                ).asArray(Int32.self)
            }
            #expect(sample() == [3])
            #expect(sample() == [3])
            #expect(rebuilds == 1)
            current.outputTokens = [3, 3]
            sampler.mtpRoundDidCommit(requestIDs: [current.id])
            #expect(sample() == [3])
            #expect(rebuilds == 2)
        }

        @Test(
            .enabled(
                if: ProcessInfo.processInfo.environment["DARKBLOOM_GREEDY_METADATA_BENCH"] == "1"))
        func identityGreedyReconfigurationBenchmark() {
            let pipeline = LogitsPipelineV2(vocabSize: 262_144)
            let sampler = SamplerV2(fallbackSeed: 1)
            let rows = (1 ... 4).map { row(UInt64($0), output: [1, 2, 3]) }
            for _ in 0 ..< 100 {
                pipeline.setRows(rows)
                sampler.setRows(rows)
            }
            var seconds: [Double] = []
            for _ in 0 ..< 5 {
                let start = ContinuousClock.now
                for _ in 0 ..< 10_000 {
                    pipeline.setRows(rows)
                    sampler.setRows(rows)
                }
                let elapsed = start.duration(to: .now).components
                seconds.append(Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18)
            }
            print(
                "greedy-metadata-config-benchmark B=4 reconfigurations=10000 "
                    + "seconds=\(seconds) median=\(seconds.sorted()[2])")
            #expect(sampler.stepIndices == [3, 3, 3, 3])
        }
    }
}
