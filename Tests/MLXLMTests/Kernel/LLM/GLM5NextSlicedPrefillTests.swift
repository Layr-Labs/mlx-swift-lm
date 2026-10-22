// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Opt-in (`GLM53_SLICED_PREFILL=1`, GPU, the row-invariance suite's
    /// rank-shaped serving stage, about 0.9 GB of synthetic weights; 5.3 GB
    /// with `GLM53_ROW_INVARIANCE_FULL=1`): a tensor-parallel rank's merged
    /// prefill frame built in layer slices (`GLM5NextTextStage.beginForward`),
    /// with another request's decode steps on its own cache between the
    /// slices, is the whole frame bit for bit: every segment's hidden rows,
    /// the logits, the cache and the in-graph snapshot; from tokens, and from
    /// input embeddings with media rows spliced in (a media lane's frame,
    /// `beginForward(embeddings:)`); with contiguous and with paged DSA
    /// histories (`dsaPagePool`, the cluster's). The serving policy leaves
    /// each layer's last mHC expand pending into the next layer's collapse,
    /// so a slice edge holds that pending expand's inputs, never a
    /// materialized form.
    @Suite(.serialized)
    struct GLM5NextSlicedPrefillTests {
        static let enabled = ProcessInfo.processInfo.environment["GLM53_SLICED_PREFILL"] == "1"

        static func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        static func same(_ a: [MLXArray], _ b: [MLXArray]) throws -> Bool {
            try fence(a + b)
            func bytes(_ x: MLXArray) -> Data { x.asData(access: .copy).data }
            return a.count == b.count
                && zip(a, b).allSatisfy { $0.shape == $1.shape && $0.dtype == $1.dtype && bytes($0) == bytes($1) }
        }

        /// A frame's input: its tokens, or its input embeddings.
        enum Input {
            case tokens(MLXArray)
            case embeddings(MLXArray)
        }

        static func begin(
            _ stage: GLM5NextTextStage, _ input: Input, cache: GLM5NextTextStageCache,
            segments: [GLM5NextPrefillSegment]
        ) throws -> GLM5NextSlicedPrefill {
            switch input {
            case .tokens(let tokens):
                return try stage.beginForward(
                    tokens: tokens, cache: cache, segments: segments, readout: .lastToken, snapshotAfter: 2)
            case .embeddings(let embeddings):
                return try stage.beginForward(
                    embeddings: embeddings, cache: cache, segments: segments, readout: .lastToken, snapshotAfter: 2)
            }
        }

        /// The whole frame against layer slices (all 1-layer; 2 then the
        /// rest), a decode step of another request after every slice.
        static func check(
            _ stage: GLM5NextTextStage, _ input: Input, segments: [GLM5NextPrefillSegment], label: String,
            account: GLM5NextDSAPageBook.Account? = nil
        ) throws {
            let vocabulary = stage.plan.configuration.vocabularySize
            let whole = stage.newCache(account: account)
            let wholeCandidate: GLM5NextTextStageCandidate
            switch input {
            case .tokens(let tokens):
                wholeCandidate = try stage.prepareForward(
                    tokens: tokens, cache: whole, segments: segments, readout: .lastToken, snapshotAfter: 2)
            case .embeddings(let embeddings):
                wholeCandidate = try stage.prepareForward(
                    embeddings: embeddings, cache: whole, segments: segments, readout: .lastToken, snapshotAfter: 2)
            }
            try wholeCandidate.commit(afterEvaluating: fence)
            guard case .final(let expected) = wholeCandidate.output, let kept = wholeCandidate.snapshot else {
                Issue.record("merged frame returned no logits or snapshot")
                return
            }
            let layers = stage.layers.count
            for slices in [Array(repeating: 1, count: layers), [2, layers - 2]] {
                // A decoding request beside it: a short prompt, then one step a slice.
                let other = stage.newCache(account: account)
                let prompt = MLXArray((0 ..< 40).map { Int32(($0 * 104_729) % vocabulary) }, [1, 40])
                try stage.prepareForward(tokens: prompt, cache: other, readout: .none).commit(afterEvaluating: fence)
                let sliced = stage.newCache(account: account)
                let prefill = try begin(stage, input, cache: sliced, segments: segments)
                for (index, count) in slices.enumerated() {
                    try prefill.advance(layers: count)
                    try fence(prefill.frontier)
                    let step = try stage.prepareForward(
                        tokens: MLXArray([Int32(17 + index)], [1, 1]), cache: other, readout: .lastToken)
                    try step.commit(afterEvaluating: fence)
                }
                let candidate = try prefill.finish()
                try candidate.commit(afterEvaluating: fence)
                guard case .final(let output) = candidate.output, let snapshot = candidate.snapshot else {
                    Issue.record("sliced frame returned no logits or snapshot")
                    return
                }
                let name = "\(label) slices \(slices)"
                #expect(try same(candidate.segmentHidden, wholeCandidate.segmentHidden), "\(name)")
                #expect(try same([output.logits], [expected.logits]), "\(name)")
                #expect(try same(sliced.layers.flatMap(\.state), whole.layers.flatMap(\.state)), "\(name)")
                #expect(try same(snapshot.layers.flatMap(\.state), kept.layers.flatMap(\.state)), "\(name)")
            }
        }

        @Test(.enabled(if: enabled))
        func slicedFrameIsTheWholeFrameWithDecodeStepsBetween() throws {
            try Device.withDefaultDevice(.gpu) {
                // Its own stage: the page pool below is set on it.
                let stage = try GLM5NextBatchInvarianceTests.makeStage(layers: 5)
                let vocabulary = stage.plan.configuration.vocabularySize
                // A merged frame: a bulk chunk, a boundary chunk, the tail.
                let segments: [GLM5NextPrefillSegment] = [
                    .init(microBatchRows: [4096, 4096]), .init(microBatchRows: [2048, 2048]),
                    .init(microBatchRows: [1414, 1414]),
                ]
                let count = segments.reduce(0) { $0 + $1.tokens }
                let tokens = MLXArray((0 ..< count).map { Int32((($0 + 1) * 7919) % vocabulary) }, [1, count])
                try Self.check(stage, .tokens(tokens), segments: segments, label: "tokens")
                // Media rows across the first micro-batch boundary and inside the last segment.
                let embedding = try #require(stage.embedding)
                let text = embedding(tokens)
                let hidden = stage.plan.configuration.hiddenSize
                let noise = MLXRandom.normal([1, count, hidden], key: MLXRandom.key(29)) * 0.02
                let features = noise.asType(text.dtype)
                let pieces: [MLXArray] = [
                    text[0..., ..<3_000], features[0..., 3_000 ..< 5_500], text[0..., 5_500 ..< 12_700],
                    features[0..., 12_700 ..< 13_400], text[0..., 13_400...],
                ]
                let media = concatenated(pieces, axis: 1)
                try Self.fence([media])
                try Self.check(stage, .embeddings(media), segments: segments, label: "media embeddings")
                // Paged DSA histories (every cache this stage makes from here on).
                let pool = try stage.makeDSAPagePool(pageSize: 64)
                let account = try pool.reserve(pages: 4096)
                stage.dsaPagePool = pool
                try Self.check(stage, .tokens(tokens), segments: segments, label: "paged tokens", account: account)
                try Self.check(
                    stage, .embeddings(media), segments: segments, label: "paged media embeddings", account: account)
            }
        }
    }
}
