// Copyright © 2026 Eigen Labs.
//
// Constrained chained decode: device-mask caching and speculative chaining
// must reproduce the unchained constrained token stream exactly (greedy and
// keyed stochastic sampling, penalties, budget-dependent masks, impossible
// states, stops, mixed constrained/ordinary batches, cancellation).

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLMCommon

/// Phase machine over a small vocabulary. Tokens divisible by 5 advance the
/// phase (state changes mid-stream); every other token is a self-loop. Each
/// phase forbids a different residue class, so a stale mask admits tokens
/// the unchained path cannot sample. The final `closing` budget only admits
/// multiples of 7, and `failAtRemaining` makes that one budget impossible.
final class CBv2ChainPhaseConstraint: CBv2TokenConstraint, @unchecked Sendable {
    let mode: CBv2TokenConstraintMode = .auto
    let maxTokens: Int
    let fallbackTokenID: Int
    let initialState = 0
    let allowsChainedDecode: Bool
    let vocab: Int
    let cacheKeys: Bool
    let closing: Int
    let failAtRemaining: Int?

    init(
        maxTokens: Int, vocab: Int, chained: Bool, cacheKeys: Bool = true,
        closing: Int = 2, failAtRemaining: Int? = nil, fallbackTokenID: Int = 0
    ) {
        self.maxTokens = maxTokens
        self.vocab = vocab
        self.allowsChainedDecode = chained
        self.cacheKeys = cacheKeys
        self.closing = closing
        self.failAtRemaining = failAtRemaining
        self.fallbackTokenID = fallbackTokenID
    }

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
        guard remainingTokens > 0, remainingTokens != failAtRemaining else { return [] }
        if remainingTokens <= closing {
            return stride(from: 0, to: vocab, by: 7).map { $0 }
        }
        return (0 ..< vocab).filter { $0 % 3 != state }
    }

    func nextState(state: Int, tokenID: Int) -> Int? {
        guard (0 ..< vocab).contains(tokenID) else { return nil }
        return tokenID % 5 == 0 ? (state + 1) % 3 : state
    }

    func maskCacheKey(state: Int, remainingTokens: Int) -> Int? {
        guard cacheKeys, remainingTokens > closing, remainingTokens != failAtRemaining
        else { return nil }
        return state
    }
}

/// TinyTestModel (bf16, paged-kernel head dim) recording forward shapes, so a
/// test can prove two runs saw identical batch compositions.
private final class CBv2ChainModel: CBv2SteppableModel {
    let inner = TinyTestModel.make(seed: 0x5EED, headDim: 64, fullAttentionOnly: true)
    private let lock = NSLock()
    private var _shapes: [[Int]] = []
    var shapes: [[Int]] { lock.withLock { _shapes } }

    init() {
        inner.update(
            parameters: ModuleParameters.unflattened(
                inner.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(inner)
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        lock.withLock { _shapes.append(tokens.shape) }
        return inner.forward(tokens: tokens, caches: caches)
    }
}

private struct CBv2ChainRow {
    var prompt: [Int]
    var sampling: CBv2SamplingParams
    var maxTokens: Int
    var constrained: Bool
    var stopTokens: Set<Int> = []
    var failAtRemaining: Int? = nil
}

private struct CBv2ChainRun {
    var outputs: [CBv2SchedCollected]
    var shapes: [[Int]]
    var chainedSteps: Int
    var redraws: Int
    var bytesWiredAfterShutdown: Int
}

final class CBv2ConstrainedChainingTests: XCTestCase {
    private static let vocab = 128

    private func run(
        _ rows: [CBv2ChainRow], chained: Bool, cancelRow: Int? = nil,
        cancelAfterTokens: Int = 0
    ) async throws -> CBv2ChainRun {
        let model = CBv2ChainModel()
        let kinds = model.inner.layerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 1 << 24, dtype: .bfloat16, maxPrefillChunk: 8,
                segmentSizeBytes: 1 << 18))
        let engine = EngineV2(
            model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2DefaultSampler(fallbackSeed: 0xABCD),
            schedulerConfig: .init(
                maxConcurrentRequests: 8, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 8, maxWaiting: 8, enablePrefixCache: false),
            loopConfig: CBv2EngineLoopConfig(idleRecheckInterval: 0.02))
        let requests = rows.enumerated().map { index, row in
            CBv2Request(
                id: CBv2RequestID(UInt64(900 + index)), promptTokens: row.prompt,
                sampling: row.sampling, maxTokens: row.maxTokens,
                stopTokens: row.stopTokens, prefixCacheEnabled: false,
                tokenConstraint: row.constrained
                    ? CBv2ChainPhaseConstraint(
                        maxTokens: row.maxTokens, vocab: Self.vocab, chained: chained,
                        failAtRemaining: row.failAtRemaining)
                    : nil)
        }
        // Submit every request inside one engine-queue frame so all of them
        // reach the scheduler before any step plans (identical compositions).
        var streams: [AsyncStream<CBv2Event>] = []
        try engine.loopForTesting.onEngineQueueSync {
            for request in requests { streams.append(try engine.submit(request)) }
        }
        var outputs = [CBv2SchedCollected](repeating: .init(), count: rows.count)
        await withTaskGroup(of: (Int, CBv2SchedCollected).self) { group in
            for (index, stream) in streams.enumerated() {
                if index == cancelRow {
                    let id = requests[index].id
                    group.addTask {
                        var collected = CBv2SchedCollected()
                        var cancelled = false
                        for await event in stream {
                            switch event {
                            case .delta(let text, let tokens, _):
                                collected.text += text
                                collected.tokens.append(contentsOf: tokens)
                                if !cancelled, collected.tokens.count >= cancelAfterTokens {
                                    cancelled = true
                                    engine.cancel(id)
                                }
                            case .finished(let reason, let usage):
                                collected.finishReason = reason
                                collected.usage = usage
                                return (index, collected)
                            }
                        }
                        return (index, collected)
                    }
                } else {
                    group.addTask { (index, await cbv2SchedCollect(stream)) }
                }
            }
            for await (index, collected) in group { outputs[index] = collected }
        }
        let chainedSteps = engine.chainedStepCount
        let redraws = engine.loopForTesting.constrainedChainRedrawCount
        await engine.shutdown()
        return CBv2ChainRun(
            outputs: outputs, shapes: model.shapes, chainedSteps: chainedSteps,
            redraws: redraws, bytesWiredAfterShutdown: backend.bytesWired)
    }

    /// Runs the unchained reference and the chained candidate, retrying the
    /// pair if (rarely) an idle recheck split admission differently.
    private func compare(
        _ rows: [CBv2ChainRow], expectRedraws: Bool = true,
        file: StaticString = #filePath, line: UInt = #line
    ) async throws -> (reference: CBv2ChainRun, chained: CBv2ChainRun) {
        var last: (CBv2ChainRun, CBv2ChainRun)?
        for _ in 0 ..< 3 {
            let reference = try await run(rows, chained: false)
            let candidate = try await run(rows, chained: true)
            last = (reference, candidate)
            // The chained run may launch one extra (discarded) successor
            // after the last confirmed token; every other forward matches.
            if candidate.shapes.starts(with: reference.shapes)
                || reference.shapes.starts(with: candidate.shapes)
            {
                break
            }
        }
        let (reference, candidate) = last!
        print(
            "[constrained-chain] tokens=\(reference.outputs.map(\.tokens)) "
                + "chained_steps=\(candidate.chainedSteps) redraws=\(candidate.redraws)")
        XCTAssertTrue(
            candidate.shapes.starts(with: reference.shapes),
            "batch compositions diverged: \(reference.shapes) vs \(candidate.shapes)",
            file: file, line: line)
        for (index, (expected, actual)) in zip(reference.outputs, candidate.outputs).enumerated() {
            XCTAssertFalse(
                expected.tokens.isEmpty, "row \(index) produced nothing", file: file, line: line)
            XCTAssertEqual(
                actual.tokens, expected.tokens, "row \(index) tokens", file: file, line: line)
            XCTAssertEqual(
                actual.finishReason, expected.finishReason, "row \(index) finish",
                file: file, line: line)
        }
        let anyConstrained = rows.contains(where: \.constrained)
        XCTAssertEqual(
            reference.chainedSteps, anyConstrained ? 0 : reference.chainedSteps,
            "unchained reference must not chain constrained batches", file: file, line: line)
        XCTAssertGreaterThan(
            candidate.chainedSteps, 0, "constrained rows must chain", file: file, line: line)
        if expectRedraws {
            XCTAssertGreaterThan(
                candidate.redraws, 0, "state changes must redraw", file: file, line: line)
        }
        XCTAssertEqual(reference.redraws, 0, file: file, line: line)
        XCTAssertEqual(candidate.bytesWiredAfterShutdown, 0, file: file, line: line)
        return (reference, candidate)
    }

    func testGreedyConstrainedChainingMatchesUnchained() async throws {
        let (reference, candidate) = try await compare([
            CBv2ChainRow(
                prompt: makePromptTokens(length: 6, seed: 21), sampling: .init(temperature: 0),
                maxTokens: 40, constrained: true)
        ])
        XCTAssertEqual(reference.outputs[0].tokens.count, 40)
        XCTAssertEqual(reference.outputs[0].finishReason, .length)
        // The closing budget really constrained the tail.
        XCTAssertTrue(reference.outputs[0].tokens.suffix(2).allSatisfy { $0 % 7 == 0 })
        // Most chained steps keep their speculative mask.
        XCTAssertLessThan(candidate.redraws, candidate.chainedSteps)
    }

    func testStochasticPenalizedChainingKeepsRNGStream() async throws {
        let sampling = CBv2SamplingParams(
            temperature: 0.9, topP: 0.95, topK: 40, repetitionPenalty: 1.2,
            repetitionContextSize: 16, frequencyPenalty: 0.3, presencePenalty: 0.2,
            seed: 1234, topLogprobs: 2)
        _ = try await compare([
            CBv2ChainRow(
                prompt: makePromptTokens(length: 6, seed: 5), sampling: sampling,
                maxTokens: 48, constrained: true)
        ])
    }

    func testMixedBatchConstrainedAndOrdinaryRowsMatchUnchained() async throws {
        let stochastic = CBv2SamplingParams(temperature: 1.0, topK: 50, seed: 77)
        _ = try await compare([
            CBv2ChainRow(
                prompt: makePromptTokens(length: 5, seed: 11), sampling: .init(temperature: 0),
                maxTokens: 32, constrained: true),
            CBv2ChainRow(
                prompt: makePromptTokens(length: 5, seed: 12), sampling: stochastic,
                maxTokens: 32, constrained: false),
            CBv2ChainRow(
                prompt: makePromptTokens(length: 5, seed: 13),
                sampling: .init(temperature: 0.7, frequencyPenalty: 0.5, seed: 9),
                maxTokens: 32, constrained: true),
        ])
    }

    func testStopTokenFinishRollsBackDiscardedSuccessor() async throws {
        let base = CBv2ChainRow(
            prompt: makePromptTokens(length: 6, seed: 21), sampling: .init(temperature: 0),
            maxTokens: 40, constrained: true)
        let probe = try await run([base], chained: false)
        let tokens = probe.outputs[0].tokens
        // Stop on the first token that first appears at index >= 8.
        let index = try XCTUnwrap(
            tokens.indices.first { $0 >= 8 && !tokens[..<$0].contains(tokens[$0]) })
        var stopped = base
        stopped.stopTokens = [tokens[index]]
        let (reference, _) = try await compare([stopped], expectRedraws: false)
        XCTAssertEqual(reference.outputs[0].finishReason, .stop)
        XCTAssertEqual(reference.outputs[0].tokens, Array(tokens[...index]))
    }

    func testImpossibleStateFailsAtTheSameTokenWhenChained() async throws {
        let (reference, _) = try await compare(
            [
                CBv2ChainRow(
                    prompt: makePromptTokens(length: 6, seed: 31),
                    sampling: .init(temperature: 0), maxTokens: 30, constrained: true,
                    failAtRemaining: 9)
            ], expectRedraws: false)
        XCTAssertEqual(
            reference.outputs[0].finishReason,
            .error(CBv2TokenConstraintFailure.impossibleState))
        XCTAssertEqual(reference.outputs[0].tokens.count, 30 - 9)
    }

    func testCancelledConstrainedChainEmitsReferencePrefixAndFreesKV() async throws {
        let rows = [
            CBv2ChainRow(
                prompt: makePromptTokens(length: 6, seed: 41), sampling: .init(temperature: 0),
                maxTokens: 200, constrained: true)
        ]
        let reference = try await run(rows, chained: false)
        let cancelled = try await run(rows, chained: true, cancelRow: 0, cancelAfterTokens: 12)
        let output = cancelled.outputs[0]
        XCTAssertEqual(output.finishReason, .cancelled)
        XCTAssertGreaterThanOrEqual(output.tokens.count, 12)
        XCTAssertEqual(
            output.tokens, Array(reference.outputs[0].tokens.prefix(output.tokens.count)))
        XCTAssertGreaterThan(cancelled.chainedSteps, 0)
        XCTAssertEqual(cancelled.bytesWiredAfterShutdown, 0)
    }
}

// MARK: - Sampler-level exactness

final class CBv2ConstrainedChainingSamplerTests: XCTestCase {
    /// The pre-cache dense mask construction, kept verbatim as the bitwise
    /// reference for the cached `which` mask.
    private func referenceMask(_ rowLogits: MLXArray, allowed: [Int]) -> MLXArray {
        let vocab = rowLogits.dim(-1)
        if allowed.count <= vocab / 2 {
            let indices = MLXArray(allowed.map(Int32.init)).reshaped([1, allowed.count])
            let values = takeAlong(rowLogits, indices, axis: -1)
            let base = MLXArray.full([1, vocab], values: MLXArray(-Float.infinity))
            return putAlong(base, indices, values: values, axis: -1)
        }
        let allowedSet = Set(allowed)
        let disallowed = (0 ..< vocab).filter { !allowedSet.contains($0) }.map(Int32.init)
        guard !disallowed.isEmpty else { return rowLogits }
        let indices = MLXArray(disallowed).reshaped([1, disallowed.count])
        let values = MLXArray.full([1, disallowed.count], values: MLXArray(-Float.infinity))
        return putAlong(rowLogits, indices, values: values, axis: -1)
    }

    func testCachedDeviceMaskIsBitwiseTheReferenceMask() throws {
        let vocab = 96
        let constraint = CBv2ChainPhaseConstraint(maxTokens: 64, vocab: vocab, chained: true)
        let sampler = CBv2TokenConstraintSampler()
        let ids = [CBv2RequestID(1), CBv2RequestID(2)]
        sampler.configure(
            ids.map {
                CBv2SamplerRow(
                    id: $0, params: .init(temperature: 0), promptTokens: [1],
                    tokenConstraint: constraint, maxTokens: 64)
            })
        var tokens: [Int] = []
        var state = 0
        for step in 0 ..< 24 {
            let logits = MLXRandom.normal([2, vocab], key: MLXRandom.key(UInt64(step)))
            let masked = sampler.mask(logits, requestIDs: ids)
            let allowed = constraint.allowedTokenIDs(state: state, remainingTokens: 64 - step)
            let expected = concatenated(
                [
                    referenceMask(logits[0 ..< 1, 0...], allowed: allowed),
                    referenceMask(logits[1 ..< 2, 0...], allowed: allowed),
                ], axis: 0)
            XCTAssertEqual(
                masked.asData(access: .copy).data, expected.asData(access: .copy).data,
                "step \(step)")
            let token = step.isMultiple(of: 4) ? 5 : allowed.first { $0 % 5 != 0 }!
            tokens.append(token)
            sampler.confirm(tokens: [token, token], requestIDs: ids)
            state = constraint.nextState(state: state, tokenID: token)!
        }
        // Three phases, two rows: the remaining steps reuse uploaded masks.
        XCTAssertLessThanOrEqual(sampler.maskBuildCount, 6)
        XCTAssertGreaterThan(sampler.maskCacheHitCount, 30)
    }

    /// Drives `CBv2DefaultSampler` through the exact chained protocol the
    /// engine uses (sample N+1 with N pending, confirm N, redraw on
    /// invalidation) and compares against the unchained protocol on the same
    /// logits, with stochastic rows, penalties and a mixed ordinary row.
    func testSpeculativeSampleAndRedrawMatchUnchainedDraws() throws {
        let vocab = 64
        let steps = 40
        let ids = [CBv2RequestID(11), CBv2RequestID(12), CBv2RequestID(13)]
        let params = [
            CBv2SamplingParams(temperature: 0),
            CBv2SamplingParams(
                temperature: 0.8, topP: 0.9, topK: 20, repetitionPenalty: 1.3,
                repetitionContextSize: 8, frequencyPenalty: 0.4, presencePenalty: 0.1,
                seed: 5, topLogprobs: 3),
            CBv2SamplingParams(temperature: 1.1, seed: 6),
        ]
        let logits = (0 ..< steps).map {
            MLXRandom.normal([3, vocab], key: MLXRandom.key(UInt64(1000 + $0))) * 3
        }

        func rows(_ constraints: [CBv2ChainPhaseConstraint?], _ outputs: [[Int]])
            -> [CBv2SamplerRow]
        {
            ids.indices.map {
                CBv2SamplerRow(
                    id: ids[$0], params: params[$0], promptTokens: [1, 2, 3],
                    outputTokens: outputs[$0], tokenConstraint: constraints[$0],
                    maxTokens: steps)
            }
        }
        func constraints() -> [CBv2ChainPhaseConstraint?] {
            [
                CBv2ChainPhaseConstraint(maxTokens: steps, vocab: vocab, chained: true, closing: 3),
                CBv2ChainPhaseConstraint(maxTokens: steps, vocab: vocab, chained: true, closing: 3),
                nil,
            ]
        }

        // Unchained reference.
        let referenceConstraints = constraints()
        let reference = CBv2DefaultSampler(fallbackSeed: 42)
        var referenceOut = [[Int]](repeating: [], count: 3)
        var referenceLogprobs: [[Int]] = []
        for step in 0 ..< steps {
            let snapshot = referenceOut
            let host = reference.sample(
                logits: logits[step], params: params, requestIDs: ids, stepIndex: step,
                pendingSampledTokens: nil,
                rowContext: { rows(referenceConstraints, snapshot) }
            ).asArray(Int32.self).map(Int.init)
            let gathered = try XCTUnwrap(reference.takeStepLogprobs())
            eval(gathered.evalTargets)
            referenceLogprobs.append(
                CBv2Logprobs.assemble(
                    gathered.gathered, sampledTokens: host,
                    topLogprobsPerRow: gathered.topLogprobsPerRow
                ).map { $0?.topLogprobs.map(\.token) ?? [] }.flatMap { $0 })
            reference.confirmSampledTokens(host, requestIDs: ids)
            for row in 0 ..< 3 { referenceOut[row].append(host[row]) }
        }

        // Chained protocol.
        let chainedConstraints = constraints()
        let chained = CBv2DefaultSampler(fallbackSeed: 42)
        var chainedOut = [[Int]](repeating: [], count: 3)
        var chainedLogprobs: [[Int]] = []
        var redraws = 0
        var inFlight = chained.sample(
            logits: logits[0], params: params, requestIDs: ids, stepIndex: 0,
            pendingSampledTokens: nil, rowContext: { rows(chainedConstraints, chainedOut) })
        var inFlightLogprobs = try XCTUnwrap(chained.takeStepLogprobs())
        for step in 0 ..< steps {
            var successor: MLXArray?
            var successorLogprobs: CBv2StepLogprobs?
            if step + 1 < steps {
                let snapshot = chainedOut
                successor = chained.sample(
                    logits: logits[step + 1], params: params, requestIDs: ids,
                    stepIndex: step + 1, pendingSampledTokens: inFlight,
                    rowContext: { rows(chainedConstraints, snapshot) })
                successorLogprobs = chained.takeStepLogprobs()
            }
            let host = inFlight.asArray(Int32.self).map(Int.init)
            eval(inFlightLogprobs.evalTargets)
            chainedLogprobs.append(
                CBv2Logprobs.assemble(
                    inFlightLogprobs.gathered, sampledTokens: host,
                    topLogprobsPerRow: inFlightLogprobs.topLogprobsPerRow
                ).map { $0?.topLogprobs.map(\.token) ?? [] }.flatMap { $0 })
            chained.confirmSampledTokens(host, requestIDs: ids)
            for row in 0 ..< 3 { chainedOut[row].append(host[row]) }
            let invalidated = chained.takeInvalidatedChainedRows()
            if let pending = successor {
                if !invalidated.isEmpty {
                    redraws += 1
                    successor = try XCTUnwrap(
                        chained.redrawChainedSuccessor(
                            logits: logits[step + 1], requestIDs: ids))
                    successorLogprobs = chained.takeStepLogprobs()
                } else {
                    successor = pending
                }
                inFlight = successor!
                inFlightLogprobs = try XCTUnwrap(successorLogprobs)
            }
        }

        XCTAssertEqual(chainedOut, referenceOut)
        XCTAssertEqual(chainedLogprobs, referenceLogprobs)
        XCTAssertGreaterThan(redraws, 0)
        XCTAssertLessThan(redraws, steps - 1)
        // Constrained rows honored their final closing budget.
        XCTAssertTrue(referenceOut[0].suffix(3).allSatisfy { $0 % 7 == 0 })
        XCTAssertTrue(referenceOut[1].suffix(3).allSatisfy { $0 % 7 == 0 })
        XCTAssertGreaterThan(chained.constraintMaskCacheHitCount, 0)
    }
}
