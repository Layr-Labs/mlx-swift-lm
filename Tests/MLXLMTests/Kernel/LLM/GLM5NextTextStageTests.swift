import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite struct GLM5NextTextStageTests {
        private func inventory(layers: Int = 8) throws -> GLM5NextWeightInventory {
            let c = try GLM5NextInventoryFixture.configuration(layers: layers)
            return try .validate(
                configuration: c,
                tensors: GLM5NextWeightInventory.expectedTensors(
                    configuration: c, storage: .float32, shard: "tiny.safetensors"),
                storage: .float32)
        }

        private func tensor(_ entry: GLM5NextWeightInventory.Entry) -> MLXArray {
            let name = entry.tensor.name
            let count = entry.tensor.shape.reduce(1, *)
            let values: [Float]
            if name.hasSuffix("_scale") || name.contains("norm.weight") {
                values = Array(repeating: 1, count: count)
            } else if name.hasSuffix("A_log") {
                values = Array(repeating: -0.7, count: count)
            } else if name.hasSuffix("dt_bias") {
                values = Array(repeating: -0.2, count: count)
            } else {
                let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 13 }
                values = (0 ..< count).map { Float(($0 + salt) % 11 - 5) * 0.025 }
            }
            return MLXArray(values, entry.tensor.shape)
        }

        private func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        private func close(_ actual: MLXArray, _ expected: MLXArray) {
            #expect(actual.shape == expected.shape)
            // The split materializes the same operations; this tolerance is
            // the full text model's existing FP32 cached-forward tolerance.
            #expect(abs(actual - expected).max().item(Float.self) < 0.0001)
        }

        @Test(arguments: Array(1 ..< 8))
        func everyCutPreservesFourStreamsGlobalIndicesAndCachedLogits(cut: Int) throws {
            try Device.withDefaultDevice(.cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let policy = GLM5NextExecutionPolicy(
                    stepEvaluation: cut == 5 ? .earlyAsync : .deferred)
                let full = try GLM5NextTextModel(
                    configuration: c, inventory: inv,
                    tensor: { tensor(try #require(inv.entries[$0])) }, executionPolicy: policy)
                var firstReads = Set<String>()
                var lastReads = Set<String>()
                let firstPlan = try GLM5NextTextStagePlan(configuration: c, layers: 0 ..< cut)
                let lastPlan = try GLM5NextTextStagePlan(
                    configuration: c, layers: cut ..< c.hiddenLayers)
                let first = try GLM5NextTextStage(
                    plan: firstPlan, inventory: inv,
                    tensor: {
                        firstReads.insert($0)
                        return tensor(try #require(inv.entries[$0]))
                    }, executionPolicy: policy)
                let last = try GLM5NextTextStage(
                    plan: lastPlan, inventory: inv,
                    tensor: {
                        lastReads.insert($0)
                        return tensor(try #require(inv.entries[$0]))
                    }, executionPolicy: policy)
                #expect(firstReads == (try firstPlan.tensorNames(in: inv)))
                #expect(lastReads == (try lastPlan.tensorNames(in: inv)))
                #expect(firstReads.isDisjoint(with: lastReads))
                #expect(
                    firstReads.union(lastReads)
                        == Set(inv.entries(in: .text).map { $0.tensor.name }))
                #expect(first.norm == nil && first.head == nil && last.embedding == nil)
                for (index, layer) in zip(lastPlan.layers, last.layers) {
                    if let sparse = layer.sparseAttention {
                        #expect(sparse.modelLayerIndex == index)
                    }
                }
                let fullCache = full.newTextCache()
                let firstCache = first.newCache()
                let lastCache = last.newCache()
                let chunks: [(MLXArray, MLXArray)] = [
                    (
                        MLXArray([Int32(2), 3, 4, 0, 0, 5], [2, 3]),
                        MLXArray([true, true, true, false, false, true], [2, 3])
                    ),
                    (MLXArray([Int32(6), 7, 8, 9], [2, 2]), MLXArray.ones([2, 2], dtype: .bool)),
                    (MLXArray([Int32(10), 11], [2, 1]), MLXArray.ones([2, 1], dtype: .bool)),
                ]
                for (tokens, validity) in chunks {
                    let expected = try full.prepareForward(
                        tokens: tokens, cache: fullCache, validity: validity
                    )
                    .commit(afterEvaluating: fence)
                    let before = firstCache.offset
                    let candidate = try first.prepareForward(
                        tokens: tokens, cache: firstCache, validity: validity)
                    guard case .streams(let boundary) = candidate.output else {
                        Issue.record("intermediate stage returned a final readout")
                        return
                    }
                    #expect(boundary.shape == [2, tokens.dim(1), c.hcMult, c.hiddenSize])
                    #expect(firstCache.offset == before)
                    let tail = try last.prepareForward(
                        streams: boundary, cache: lastCache, validity: validity)
                    var actual: GLM5NextTextOutput?
                    try candidate.commit { roots in
                        try fence(roots)
                        if case .final(let output) = try tail.commit(afterEvaluating: fence) {
                            actual = output
                        }
                    }
                    let output = try #require(actual)
                    close(output.hidden, expected.hidden)
                    close(output.logits, expected.logits)
                    #expect(
                        firstCache.offset == fullCache.offset
                            && lastCache.offset == fullCache.offset)
                    for (split, original) in zip(
                        firstCache.layers + lastCache.layers, fullCache.layers)
                    {
                        #expect(split.offset == original.offset)
                        #expect(split.state.count == original.state.count)
                        for (a, b) in zip(split.state, original.state) {
                            close(a.asType(.float32), b.asType(.float32))
                        }
                    }
                }
            }
        }

        @Test(arguments: [false, true])
        func admittedReadoutPreservesHiddenCacheAndFollowingDecode(gpu: Bool) throws {
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let first = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 0 ..< 4), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                let last = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 4 ..< 8), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                let firstCache = first.newCache()
                let fullCache = last.newCache()
                let selectedCache = last.newCache()
                let chunks: [(MLXArray, MLXArray, GLM5NextTextStageReadout)] = [
                    (
                        MLXArray([Int32(2), 3, 4, 5, 6, 0], [2, 3]),
                        MLXArray([true, true, true, true, true, false], [2, 3]), .none
                    ),
                    (
                        MLXArray([Int32(7), 8, 9, 0], [2, 2]),
                        MLXArray([true, true, true, false], [2, 2]), .lastToken
                    ),
                    (
                        MLXArray([Int32(10), 11], [2, 1]),
                        MLXArray.ones([2, 1], dtype: .bool), .lastToken
                    ),
                ]
                for (tokens, validity, readout) in chunks {
                    guard
                        case .streams(let boundary) = try first.prepareForward(
                            tokens: tokens, cache: firstCache, validity: validity
                        ).commit(afterEvaluating: fence)
                    else {
                        Issue.record("first stage lost its four-stream boundary")
                        return
                    }
                    guard
                        case .final(let expected) = try last.prepareForward(
                            streams: boundary, cache: fullCache, validity: validity
                        ).commit(afterEvaluating: fence)
                    else {
                        Issue.record("default must retain the full-logits API")
                        return
                    }
                    #expect(expected.logits.shape == [2, tokens.dim(1), c.vocabularySize])
                    let before = selectedCache.offset
                    // Even a hidden-only transaction must fence before publishing.
                    let failed = try last.prepareForward(
                        streams: boundary, cache: selectedCache, validity: validity,
                        readout: readout)
                    #expect(throws: GLM5NextLoadError.self) {
                        try failed.commit { roots in
                            try fence(roots)
                            throw GLM5NextLoadError.invalid("readout acknowledgement failed")
                        }
                    }
                    #expect(selectedCache.offset == before)
                    let selected = try last.prepareForward(
                        streams: boundary, cache: selectedCache, validity: validity,
                        readout: readout)
                    try selected.commit { roots in
                        // A full [B,T,V] projection must not be an evaluation root.
                        if readout == .none || tokens.dim(1) > 1 {
                            #expect(!roots.contains { $0.shape == expected.logits.shape })
                        }
                        try fence(roots)
                    }
                    switch selected.output {
                    case .hidden(let hidden):
                        #expect(readout == .none)
                        close(hidden, expected.hidden)
                    case .final(let value):
                        #expect(readout == .lastToken)
                        close(value.hidden, expected.hidden)
                        // Last physical column, including a masked column, is
                        // the existing stage API's selection semantics.
                        close(value.logits, expected.logits[0..., (-1)..., 0...])
                    case .streams:
                        Issue.record("final stage returned an intermediate boundary")
                    }
                    #expect(selectedCache.offset == fullCache.offset)
                    for (selected, full) in zip(selectedCache.layers, fullCache.layers) {
                        #expect(selected.state.count == full.state.count)
                        for (a, b) in zip(selected.state, full.state) {
                            close(a.asType(.float32), b.asType(.float32))
                        }
                    }
                }
            }
        }

        /// A segmented prefill is the per-segment prefills it merges, bit for
        /// bit: every segment's hidden rows, the last row's logits, the cache,
        /// and the snapshot at a segment boundary. A micro-batch computes what
        /// its rows compute as a chunk of their own, and a one-segment,
        /// one-batch call is the ordinary `prepareForward(tokens:)`.
        @Test(arguments: [false, true])
        func segmentedPrefillIsItsSegmentsBitForBit(gpu: Bool) throws {
            func bits(_ array: MLXArray) -> Data { array.asData(access: .copy).data }
            func same(_ a: [MLXArray], _ b: [MLXArray]) throws -> Bool {
                try fence(a + b)
                return a.count == b.count
                    && zip(a, b).allSatisfy { $0.shape == $1.shape && bits($0) == bits($1) }
            }
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let stage = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 0 ..< 8), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                let prompt: [Int32] = (0 ..< 23).map { Int32(4 + ($0 * 7) % 25) }
                let segments: [GLM5NextPrefillSegment] = [
                    .init(microBatchRows: [3, 5]), .init(microBatchRows: [4]),
                    .init(microBatchRows: [2, 2, 1]), .init(microBatchRows: [6]),
                ]
                func tokens(_ range: Range<Int>) -> MLXArray {
                    MLXArray(Array(prompt[range]), [1, range.count])
                }
                // One call per segment, and the cache after the first two.
                let reference = stage.newCache()
                var hidden: [MLXArray] = []
                var logits: MLXArray?
                var kept: [MLXArray] = []
                var start = 0
                for (index, segment) in segments.enumerated() {
                    let last = index == segments.count - 1
                    let candidate = try stage.prepareForward(
                        tokens: tokens(start ..< start + segment.tokens), cache: reference,
                        segments: [segment], readout: last ? .lastToken : .none)
                    try candidate.commit(afterEvaluating: fence)
                    hidden.append(candidate.segmentHidden[0])
                    if last, case .final(let output) = candidate.output { logits = output.logits }
                    if index == 1 { kept = reference.copy().layers.flatMap(\.state) }
                    start += segment.tokens
                }
                let merged = stage.newCache()
                let candidate = try stage.prepareForward(
                    tokens: tokens(0 ..< prompt.count), cache: merged, segments: segments,
                    readout: .lastToken, snapshotAfter: 2)
                try candidate.commit(afterEvaluating: fence)
                guard case .final(let output) = candidate.output else {
                    Issue.record("segmented prefill returned no logits")
                    return
                }
                #expect(try same(candidate.segmentHidden, hidden))
                #expect(try same([output.logits], [try #require(logits)]))
                #expect(try same([output.hidden], [concatenated(hidden, axis: 1)]))
                #expect(merged.offset == prompt.count)
                #expect(try same(merged.layers.flatMap(\.state), reference.layers.flatMap(\.state)))
                let snapshot = try #require(candidate.snapshot)
                #expect(snapshot.offset == 12)
                #expect(try same(snapshot.layers.flatMap(\.state), kept))

                // A micro-batch's layers compute what its rows compute as a
                // chunk of their own (the readout over a different row count
                // need not: CPU reductions vectorize by size, so each segment
                // reads out its own rows); one batch is the ordinary call.
                let chunked = stage.newCache()
                let plain = stage.newCache()
                start = 0
                for count in segments[0].microBatchRows + segments[1].microBatchRows {
                    let candidate = try stage.prepareForward(
                        tokens: tokens(start ..< start + count), cache: chunked,
                        segments: [.init(microBatchRows: [count])])
                    try candidate.commit(afterEvaluating: fence)
                    let ordinary = try stage.prepareForward(
                        tokens: tokens(start ..< start + count), cache: plain, readout: .none)
                    try ordinary.commit(afterEvaluating: fence)
                    guard case .hidden(let value) = ordinary.output else {
                        Issue.record("hidden readout returned no hidden rows")
                        return
                    }
                    #expect(try same([candidate.segmentHidden[0]], [value]))
                    start += count
                }
                #expect(try same(chunked.layers.flatMap(\.state), kept))
                #expect(try same(plain.layers.flatMap(\.state), kept))

                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(
                        tokens: tokens(0 ..< 4), cache: stage.newCache(),
                        segments: [.init(microBatchRows: [3])])
                }
                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(
                        tokens: tokens(0 ..< 4), cache: stage.newCache(),
                        segments: [.init(microBatchRows: [4])], snapshotAfter: 1)
                }
            }
        }

        /// The segmented prefill from input embeddings (a multimodal prompt's
        /// rows) is the token path's when the embeddings are the tokens', and a
        /// merged call with media rows spliced in is its per-segment calls bit
        /// for bit: every segment's hidden rows, the logits, the cache and the
        /// snapshot. One-batch segments are the ordinary embeddings call.
        @Test(arguments: [false, true])
        func segmentedPrefillFromEmbeddingsKeepsMediaRowsBitwise(gpu: Bool) throws {
            func bits(_ array: MLXArray) -> Data { array.asData(access: .copy).data }
            func same(_ a: [MLXArray], _ b: [MLXArray]) throws -> Bool {
                try fence(a + b)
                return a.count == b.count
                    && zip(a, b).allSatisfy { $0.shape == $1.shape && bits($0) == bits($1) }
            }
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let stage = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 0 ..< 8), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                let embedding = try #require(stage.embedding)
                let prompt: [Int32] = (0 ..< 23).map { Int32(4 + ($0 * 7) % 25) }
                let segments: [GLM5NextPrefillSegment] = [
                    .init(microBatchRows: [3, 5]), .init(microBatchRows: [4]),
                    .init(microBatchRows: [2, 2, 1]), .init(microBatchRows: [6]),
                ]
                let tokens = MLXArray(prompt, [1, prompt.count])
                let text = embedding(tokens)
                // Media rows 5..<10 and 14..<17 (across segment boundaries).
                let features = MLXRandom.normal(
                    [1, prompt.count, c.hiddenSize], key: MLXRandom.key(7)
                )
                .asType(text.dtype)
                let media = concatenated(
                    [
                        text[0..., ..<5], features[0..., 5 ..< 10], text[0..., 10 ..< 14],
                        features[0..., 14 ..< 17],
                        text[0..., 17...],
                    ], axis: 1)
                try fence([text, media])

                // The token path and the same rows as embeddings.
                let byTokens = stage.newCache()
                let byRows = stage.newCache()
                let a = try stage.prepareForward(
                    tokens: tokens, cache: byTokens, segments: segments, readout: .lastToken,
                    snapshotAfter: 2)
                try a.commit(afterEvaluating: fence)
                let b = try stage.prepareForward(
                    embeddings: text, cache: byRows, segments: segments, readout: .lastToken,
                    snapshotAfter: 2)
                try b.commit(afterEvaluating: fence)
                guard case .final(let ta) = a.output, case .final(let tb) = b.output else {
                    Issue.record("segmented prefill returned no logits")
                    return
                }
                #expect(try same(a.segmentHidden, b.segmentHidden))
                #expect(try same([ta.logits], [tb.logits]))
                #expect(try same(byTokens.layers.flatMap(\.state), byRows.layers.flatMap(\.state)))

                // Media rows: one call per segment against the merged call.
                let reference = stage.newCache()
                var hidden: [MLXArray] = []
                var logits: MLXArray?
                var kept: [MLXArray] = []
                var start = 0
                for (index, segment) in segments.enumerated() {
                    let last = index == segments.count - 1
                    let candidate = try stage.prepareForward(
                        embeddings: media[0..., start ..< start + segment.tokens], cache: reference,
                        segments: [segment], readout: last ? .lastToken : .none)
                    try candidate.commit(afterEvaluating: fence)
                    hidden.append(candidate.segmentHidden[0])
                    if last, case .final(let output) = candidate.output { logits = output.logits }
                    if index == 1 { kept = reference.copy().layers.flatMap(\.state) }
                    start += segment.tokens
                }
                let merged = stage.newCache()
                let candidate = try stage.prepareForward(
                    embeddings: media, cache: merged, segments: segments, readout: .lastToken,
                    snapshotAfter: 2)
                try candidate.commit(afterEvaluating: fence)
                guard case .final(let output) = candidate.output else {
                    Issue.record("segmented prefill returned no logits")
                    return
                }
                #expect(try same(candidate.segmentHidden, hidden))
                #expect(try same([output.logits], [try #require(logits)]))
                #expect(try same(merged.layers.flatMap(\.state), reference.layers.flatMap(\.state)))
                #expect(try same(try #require(candidate.snapshot).layers.flatMap(\.state), kept))
                // The media rows changed the result (they are not the tokens').
                #expect(!(try same([output.logits], [ta.logits])))

                // One-batch segments are the ordinary embeddings call.
                let segmented = stage.newCache()
                let plain = stage.newCache()
                start = 0
                for count in [8, 4, 5, 6] {
                    let rows = media[0..., start ..< start + count]
                    let one = try stage.prepareForward(
                        embeddings: rows, cache: segmented,
                        segments: [.init(microBatchRows: [count])])
                    try one.commit(afterEvaluating: fence)
                    let ordinary = try stage.prepareForward(
                        embeddings: rows, cache: plain, readout: .none)
                    try ordinary.commit(afterEvaluating: fence)
                    guard case .hidden(let value) = ordinary.output else {
                        Issue.record("hidden readout returned no hidden rows")
                        return
                    }
                    #expect(try same([one.segmentHidden[0]], [value]))
                    start += count
                }
                #expect(try same(segmented.layers.flatMap(\.state), plain.layers.flatMap(\.state)))

                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(
                        embeddings: media[0..., ..<4], cache: stage.newCache(),
                        segments: [.init(microBatchRows: [3])])
                }
            }
        }

        /// A segmented prefill built and evaluated in layer slices, with
        /// another request's forwards on its own cache between the slices, is
        /// the whole call bit for bit (hidden rows, logits, cache, snapshot),
        /// from tokens and from embeddings with media rows.
        @Test(arguments: [false, true])
        func slicedPrefillIsTheWholeCallBitForBit(gpu: Bool) throws {
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let stage = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 0 ..< 8), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                try slicedPrefillChecks(stage)
            }
        }

        private enum SlicedInput {
            case tokens(MLXArray)
            case embeddings(MLXArray)
        }

        private func slicedPrefillChecks(_ stage: GLM5NextTextStage) throws {
            let c = stage.plan.configuration
            let prompt: [Int32] = (0 ..< 23).map { Int32(4 + ($0 * 7) % 25) }
            let tokens = MLXArray(prompt, [1, prompt.count])
            let embedding = try #require(stage.embedding)
            let text = embedding(tokens)
            let noise = MLXRandom.normal([1, prompt.count, c.hiddenSize], key: MLXRandom.key(7))
            let features = noise.asType(text.dtype)
            let pieces: [MLXArray] = [
                text[0..., ..<5], features[0..., 5 ..< 10], text[0..., 10...],
            ]
            let media = concatenated(pieces, axis: 1)
            try fence([media])
            let segments: [GLM5NextPrefillSegment] = [
                .init(microBatchRows: [3, 5]), .init(microBatchRows: [4]),
                .init(microBatchRows: [2, 2, 1]), .init(microBatchRows: [6]),
            ]
            for input in [SlicedInput.tokens(tokens), .embeddings(media)] {
                for slices in [[8], [1, 1, 1, 1, 1, 1, 1, 1], [3, 5], [5, 2, 1]] {
                    try slicedPrefillCheck(stage, input, segments: segments, slices: slices)
                }
            }
            // Reading out before the last layer is refused.
            let early = try stage.beginForward(
                tokens: tokens, cache: stage.newCache(), segments: segments)
            try early.advance(layers: 7)
            #expect(throws: GLM5NextLoadError.self) { try early.finish() }
        }

        private func slicedPrefillCheck(
            _ stage: GLM5NextTextStage, _ input: SlicedInput, segments: [GLM5NextPrefillSegment],
            slices: [Int]
        ) throws {
            func bits(_ array: MLXArray) -> Data { array.asData(access: .copy).data }
            func same(_ a: [MLXArray], _ b: [MLXArray]) throws -> Bool {
                try fence(a + b)
                return a.count == b.count
                    && zip(a, b).allSatisfy { $0.shape == $1.shape && bits($0) == bits($1) }
            }
            let whole = stage.newCache()
            let sliced = stage.newCache()
            let other = stage.newCache()
            let reference: GLM5NextTextStageCandidate
            let prefill: GLM5NextSlicedPrefill
            switch input {
            case .tokens(let tokens):
                reference = try stage.prepareForward(
                    tokens: tokens, cache: whole, segments: segments, readout: .lastToken,
                    snapshotAfter: 2)
                prefill = try stage.beginForward(
                    tokens: tokens, cache: sliced, segments: segments, readout: .lastToken,
                    snapshotAfter: 2)
            case .embeddings(let embeddings):
                reference = try stage.prepareForward(
                    embeddings: embeddings, cache: whole, segments: segments, readout: .lastToken,
                    snapshotAfter: 2)
                prefill = try stage.beginForward(
                    embeddings: embeddings, cache: sliced, segments: segments, readout: .lastToken,
                    snapshotAfter: 2)
            }
            try reference.commit(afterEvaluating: fence)
            for (index, count) in slices.enumerated() {
                try prefill.advance(layers: count)
                try fence(prefill.frontier)
                // Another request's forward between the slices.
                let step = try stage.prepareForward(
                    tokens: MLXArray([Int32(5 + index)], [1, 1]), cache: other, readout: .lastToken)
                try step.commit(afterEvaluating: fence)
            }
            #expect(throws: GLM5NextLoadError.self) { try prefill.advance(layers: 1) }
            let candidate = try prefill.finish()
            try candidate.commit(afterEvaluating: fence)
            guard case .final(let expected) = reference.output,
                case .final(let output) = candidate.output,
                let snapshot = candidate.snapshot, let kept = reference.snapshot
            else {
                Issue.record("segmented prefill returned no logits or snapshot")
                return
            }
            let label = "\(input) slices \(slices)"
            #expect(try same(candidate.segmentHidden, reference.segmentHidden), "\(label)")
            #expect(
                try same([output.logits, output.hidden], [expected.logits, expected.hidden]),
                "\(label)")
            #expect(sliced.offset == whole.offset)
            #expect(
                try same(sliced.layers.flatMap(\.state), whole.layers.flatMap(\.state)), "\(label)")
            #expect(snapshot.offset == 12)
            #expect(
                try same(snapshot.layers.flatMap(\.state), kept.layers.flatMap(\.state)), "\(label)"
            )
        }

        @Test func segmentsOfAChunkAreItsMicroBatches() {
            func rows(_ tokens: Int, _ batches: Int, sharded: Bool = true) -> [Int] {
                GLM5NextPrefillSegment.chunk(
                    tokens: tokens, microBatches: batches, sharded: sharded
                ).microBatchRows
            }
            #expect(rows(8_192, 2) == [4_096, 4_096])
            #expect(rows(2_828, 2) == [1_414, 1_414])
            #expect(rows(2_829, 2) == [1_414, 1_415])
            #expect(rows(511, 2) == [511])
            #expect(rows(512, 2) == [256, 256])
            #expect(rows(1_000, 4) == [333, 333, 334])
            #expect(rows(8_192, 2, sharded: false) == [8_192])
            #expect(rows(8_192, 1) == [8_192])
        }

        @Test func tokenAndEmbeddingEntryPointsHonorReadout() throws {
            try Device.withDefaultDevice(.gpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let stage = try GLM5NextTextStage(
                    plan: .init(configuration: c, layers: 0 ..< 8), inventory: inv
                ) { tensor(try #require(inv.entries[$0])) }
                let tokens = MLXArray([Int32(2), 3, 4], [1, 3])
                let embedding = try #require(stage.embedding)
                let embedded = embedding(tokens)
                guard
                    case .final(let expected) = try stage.prepareForward(
                        tokens: tokens, cache: stage.newCache()
                    ).commit(afterEvaluating: fence),
                    case .final(let selected) = try stage.prepareForward(
                        tokens: tokens, cache: stage.newCache(), readout: .lastToken
                    ).commit(afterEvaluating: fence),
                    case .hidden(let hidden) = try stage.prepareForward(
                        embeddings: embedded,
                        cache: stage.newCache(), readout: .none
                    ).commit(afterEvaluating: fence)
                else {
                    Issue.record("token or embedding entry point ignored its readout")
                    return
                }
                close(selected.logits, expected.logits[0..., (-1)..., 0...])
                close(selected.hidden, expected.hidden)
                close(hidden, expected.hidden)
            }
        }

        @Test func failedAcknowledgementDiscardAndStaleCandidatesDoNotPublish() throws {
            try Device.withDefaultDevice(.cpu) {
                let inv = try inventory()
                let plan = try GLM5NextTextStagePlan(
                    configuration: inv.configuration.textConfig, layers: 0 ..< 4)
                let stage = try GLM5NextTextStage(plan: plan, inventory: inv) {
                    tensor(try #require(inv.entries[$0]))
                }
                let cache = stage.newCache()
                let tokens = MLXArray([Int32(2), 3], [1, 2])
                let failed = try stage.prepareForward(tokens: tokens, cache: cache)
                #expect(throws: GLM5NextLoadError.self) {
                    try failed.commit { roots in
                        try fence(roots)
                        throw GLM5NextLoadError.invalid("downstream acknowledgement failed")
                    }
                }
                #expect(cache.layers.allSatisfy { $0.offset == 0 && $0.state.isEmpty })
                #expect(throws: GLM5NextLoadError.self) {
                    try failed.commit(afterEvaluating: fence)
                }
                let discarded = try stage.prepareForward(tokens: tokens, cache: cache)
                discarded.discard()
                #expect(throws: GLM5NextLoadError.self) {
                    try discarded.commit(afterEvaluating: fence)
                }
                let stale = try stage.prepareForward(tokens: tokens, cache: cache)
                let winner = try stage.prepareForward(tokens: tokens, cache: cache)
                #expect(throws: GLM5NextLoadError.self) {
                    try stale.commit { roots in
                        try fence(roots)
                        try winner.commit(afterEvaluating: fence)
                    }
                }
                #expect(cache.offset == 2)
                let snapshot = cache.copy()
                let preserved = cache.layers.map {
                    $0.state.map { $0.asType(.float32).asArray(Float.self) }
                }
                let failedWithHistory = try stage.prepareForward(tokens: tokens, cache: cache)
                #expect(throws: GLM5NextLoadError.self) {
                    try failedWithHistory.commit { roots in
                        try fence(roots)
                        throw GLM5NextLoadError.invalid("downstream failed after cached prefill")
                    }
                }
                #expect(cache.offset == 2)
                #expect(
                    cache.layers.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }
                        == preserved)
                _ = try stage.prepareForward(tokens: tokens, cache: cache).commit(
                    afterEvaluating: fence)
                #expect(snapshot.offset == 2 && cache.offset == 4)
                let rewindCopy = cache.copy()
                try rewindCopy.truncate(to: 2)
                #expect(rewindCopy.offset == 2 && cache.offset == 4)
                #expect(
                    rewindCopy.layers.map {
                        $0.state.map { $0.asType(.float32).asArray(Float.self) }
                    } == preserved)
                #expect(throws: GLM5NextLoadError.self) { try cache.truncate(to: 3) }
                #expect(cache.layers.allSatisfy { $0.offset == 4 })
                try cache.truncate(to: 2)
                #expect(cache.offset == 2)
            }
        }

        @Test func rejectsCollapsedBoundaryForeignCacheAndWrongEntryPoint() throws {
            try Device.withDefaultDevice(.cpu) {
                let inv = try inventory()
                let c = inv.configuration.textConfig
                let plan = try GLM5NextTextStagePlan(configuration: c, layers: 4 ..< 8)
                func make() throws -> GLM5NextTextStage {
                    try .init(plan: plan, inventory: inv) { tensor(try #require(inv.entries[$0])) }
                }
                let stage = try make()
                let cache = stage.newCache()
                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(
                        streams: MLXArray.zeros([1, 1, c.hiddenSize]), cache: cache)
                }
                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(tokens: MLXArray([Int32(2)], [1, 1]), cache: cache)
                }
                let other = try make()
                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareForward(
                        streams: MLXArray.zeros([1, 1, c.hcMult, c.hiddenSize]),
                        cache: other.newCache())
                }
                #expect(cache.offset == 0)
            }
        }

        @Test(arguments: [false, true])
        func artifactStageReadsOnlyOwnedPayloadAndAccountsForDuplicateEmbedding(q4: Bool) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture(q4: q4)
                let loader = try fixture.loader()
                let c = loader.configuration.textConfig
                let plan = try GLM5NextTextStagePlan(
                    configuration: c, layers: 2 ..< c.hiddenLayers, loadMTPEmbedding: true)
                let ordinary = try GLM5NextTextStagePlan(
                    configuration: c, layers: 2 ..< c.hiddenLayers)
                let expected = try plan.tensorNames(in: loader.inventory)
                var reads: [String: Int] = [:]
                let stage = try loader.loadTextStage(plan: plan) { name, bytes in
                    reads[name] = bytes
                }
                #expect(Set(reads.keys) == expected)
                #expect(reads.values.reduce(0, +) == (try plan.tensorBytes(in: loader.inventory)))
                #expect(stage.embedding != nil && stage.head != nil)
                let duplicate = loader.inventory.entries.values.filter {
                    $0.tensor.name.hasPrefix("model.language_model.embed_tokens.")
                }.reduce(0) { $0 + $1.bytes }
                #expect(
                    try plan.tensorBytes(in: loader.inventory)
                        - ordinary.tensorBytes(in: loader.inventory) == duplicate)
                #expect(
                    Set(loader.unresolved(in: .text))
                        == Set(loader.inventory.entries(in: .text).map { $0.tensor.name })
                        .subtracting(expected))
                #expect(
                    loader.unresolved(in: .mtp).count == loader.inventory.entries(in: .mtp).count)
                #expect(throws: GLM5NextArtifactError.self) { try loader.loadTextStage(plan: plan) }
            }
        }

        @Test func artifactMTPRequiresItsOwnFinalStageAndReadsOnlyNextN() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let loader = try fixture.loader()
                let plan = try GLM5NextTextStagePlan(
                    configuration: loader.configuration.textConfig, layers: 2 ..< 4,
                    loadMTPEmbedding: true)
                let stage = try loader.loadTextStage(plan: plan)
                let otherLoader = try fixture.loader()
                let otherStage = try otherLoader.loadTextStage(plan: plan)
                var reads: Set<String> = []
                #expect(throws: GLM5NextArtifactError.self) {
                    try loader.loadMTP(stage: otherStage) { name, _ in reads.insert(name) }
                }
                #expect(reads.isEmpty)
                let mtp = try loader.loadMTP(stage: stage) { name, _ in reads.insert(name) }
                #expect(mtp.targetIdentity == ObjectIdentifier(stage))
                #expect(reads == Set(loader.inventory.entries(in: .mtp).map { $0.tensor.name }))
                #expect(loader.unresolved(in: .mtp).isEmpty)
                #expect(
                    loader.unresolved(in: .vision).count
                        == loader.inventory.entries(in: .vision).count)
                #expect(!loader.unresolved(in: .text).isEmpty)
            }
        }

        @Test func admissionFailurePrecedesAnySelectedPayloadAllocation() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture()
                let loader = try fixture.loader()
                let plan = try GLM5NextTextStagePlan(
                    configuration: loader.configuration.textConfig, layers: 0 ..< 1)
                var admissions = 0
                #expect(throws: GLM5NextLoadError.self) {
                    try loader.loadTextStage(plan: plan) { _, _ in
                        admissions += 1
                        throw GLM5NextLoadError.invalid("load headroom exhausted")
                    }
                }
                #expect(admissions == 1)
                #expect(
                    loader.unresolved(in: .text).count == loader.inventory.entries(in: .text).count)
            }
        }
    }
}
