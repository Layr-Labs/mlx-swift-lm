// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Speculative verification on a whole-model stage (`prepareVerification`,
    /// `commit(afterEvaluating:keeping:)`) and the stage-bound NextN proposal
    /// (`primeHistory`, `primeAndPropose`), on the tiny FP32 fixture: three
    /// KDA layers and one DSA layer whose four-token budget makes later
    /// queries sparse.
    @Suite(.serialized)
    struct GLM5NextStageVerificationTests {
        private struct Fixture {
            let inventory: GLM5NextWeightInventory
            var configuration: GLM5NextTextConfiguration { inventory.configuration.textConfig }

            init() throws {
                let c = try GLM5NextInventoryFixture.configuration(layers: 4)
                inventory = try .validate(
                    configuration: c,
                    tensors: GLM5NextWeightInventory.expectedTensors(
                        configuration: c, storage: .float32, shard: "tiny-verify.safetensors"),
                    storage: .float32)
            }

            func tensor(_ name: String) throws -> MLXArray {
                let entry = try #require(inventory.entries[name])
                let count = entry.tensor.shape.reduce(1, *)
                let values: [Float]
                if name.hasSuffix("_scale") || name.contains("norm.weight") {
                    values = Array(repeating: 1, count: count)
                } else if name.hasSuffix("A_log") {
                    values = Array(repeating: -0.7, count: count)
                } else if name.hasSuffix("dt_bias") {
                    values = Array(repeating: -0.2, count: count)
                } else if name.hasSuffix(".bias") {
                    values = Array(repeating: 0, count: count)
                } else {
                    let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 13 }
                    values = (0 ..< count).map { Float(($0 + salt) % 11 - 5) * 0.025 }
                }
                return MLXArray(values, entry.tensor.shape)
            }

            /// The whole model as one stage, as a single device or a
            /// tensor-parallel rank holds it.
            func stage() throws -> GLM5NextTextStage {
                try .init(
                    plan: .init(configuration: configuration, layers: 0 ..< configuration.hiddenLayers),
                    inventory: inventory, tensor: tensor, executionPolicy: .reference)
            }
        }

        private func fence(_ roots: [MLXArray]) throws {
            try MLX.withError { error in
                eval(roots)
                try error.check()
            }
        }

        private func close(_ actual: MLXArray, _ expected: MLXArray) {
            #expect(actual.shape == expected.shape)
            // The window materializes the same FP32 operations; the text
            // model's existing cached-forward tolerance, not frozen bits.
            let error = abs(actual - expected).max().item(Float.self)
            #expect(error.isFinite && error < 0.0001)
        }

        private func tokens(_ values: [Int32]) -> MLXArray { MLXArray(values, [1, values.count]) }

        private func prefilled(_ stage: GLM5NextTextStage) throws -> GLM5NextTextStageCache {
            let cache = stage.newCache()
            try stage.prepareForward(tokens: tokens([4, 5, 6, 7, 8, 9, 3]), cache: cache)
                .commit(afterEvaluating: fence)
            return cache
        }

        private func logits(_ output: GLM5NextTextStageOutput) throws -> MLXArray {
            guard case .final(let value) = output else {
                Issue.record("a whole-model stage reads out logits")
                return MLXArray(0)
            }
            return value.logits
        }

        /// Keeping `kept` window tokens leaves the cache an ordinary `kept`-token
        /// forward leaves: the same offset and the same next logits.
        @Test(arguments: [1, 2, 3])
        func keepingAPrefixMatchesAForwardOfThatPrefix(kept: Int) throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let stage = try f.stage()
                let window: [Int32] = [9, 3, 2]
                let verified = try prefilled(stage)
                let reference = try prefilled(stage)
                let candidate = try stage.prepareVerification(tokens: tokens(window), cache: verified)
                var seen: GLM5NextTextStageOutput?
                try candidate.commit(afterEvaluating: fence) { output in
                    seen = output
                    return kept
                }
                let whole = try stage.prepareForward(
                    tokens: tokens(window), cache: try prefilled(stage), readout: .allTokens)
                try whole.commit(afterEvaluating: fence)
                close(try logits(#require(seen)), try logits(whole.output))
                try stage.prepareForward(tokens: tokens(Array(window.prefix(kept))), cache: reference)
                    .commit(afterEvaluating: fence)
                #expect(verified.offset == reference.offset && verified.offset == 7 + kept)
                // The KDA layers adopted the captured state of the last kept
                // token and dropped the window; DSA dropped the rejected columns.
                let next = tokens([5])
                let a = try stage.prepareForward(tokens: next, cache: verified)
                try a.commit(afterEvaluating: fence)
                let b = try stage.prepareForward(tokens: next, cache: reference)
                try b.commit(afterEvaluating: fence)
                close(try logits(a.output), try logits(b.output))
                // No layer retains a captured window: at most its state and one
                // rollback checkpoint (KDA), or its four histories (DSA).
                #expect(verified.layers.allSatisfy { $0.innerState().count <= 4 })
            }
        }

        /// Every layer's logical state, in a comparable form (validity as FP32).
        private func states(_ cache: GLM5NextTextStageCache) -> [MLXArray] {
            cache.layers.flatMap { $0.state }.map { $0.dtype == .bool ? $0.asType(.float32) : $0 }
        }

        private func next(_ stage: GLM5NextTextStage, _ cache: GLM5NextTextStageCache) throws
            -> MLXArray
        {
            let candidate = try stage.prepareForward(tokens: tokens([5]), cache: cache)
            try candidate.commit(afterEvaluating: fence)
            return try logits(candidate.output)
        }

        /// Speculative rounds that reject proposals leave exactly the cache of
        /// one-token steps over the kept tokens: every KDA layer's convolution
        /// window and recurrent state, every DSA history and the next logits.
        /// Catches a convolution window not rolled back, a captured state one
        /// token off, and DSA columns kept or dropped wrongly.
        @Test func rejectingRoundsMatchOneTokenSteps() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let stage = try f.stage()
                let speculative = try prefilled(stage)
                let plain = try prefilled(stage)
                let windows: [[Int32]] = [
                    [9, 3, 2], [4, 11, 6], [7, 7, 1], [2, 8, 5], [6, 1, 9], [3, 3, 3], [10, 4, 8],
                ]
                let keeps = [1, 2, 3, 1, 3, 2, 1]
                for (window, kept) in zip(windows, keeps) {
                    try stage.prepareVerification(tokens: tokens(window), cache: speculative)
                        .commit(afterEvaluating: fence) { _ in kept }
                    for token in window.prefix(kept) {
                        try stage.prepareForward(tokens: tokens([token]), cache: plain)
                            .commit(afterEvaluating: fence)
                    }
                    #expect(speculative.offset == plain.offset)
                }
                #expect(speculative.offset == 7 + keeps.reduce(0, +))
                let left = states(speculative), right = states(plain)
                #expect(left.count == right.count)
                for (a, b) in zip(left, right) { close(a, b) }
                close(try next(stage, speculative), try next(stage, plain))
            }
        }

        /// A one-token window is an ordinary step: the same operations, so the
        /// same bits in every state and in the logits.
        @Test func oneTokenVerificationIsAnOrdinaryStepBitwise() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let stage = try f.stage()
                let speculative = try prefilled(stage)
                let plain = try prefilled(stage)
                for token: Int32 in [9, 3, 2, 4] {
                    let window = try stage.prepareVerification(tokens: tokens([token]), cache: speculative)
                    try window.commit(afterEvaluating: fence) { _ in 1 }
                    let step = try stage.prepareForward(tokens: tokens([token]), cache: plain)
                    try step.commit(afterEvaluating: fence)
                    #expect(
                        abs(try logits(window.output) - logits(step.output)).max().item(Float.self) == 0)
                }
                for (a, b) in zip(states(speculative), states(plain)) {
                    #expect(a.shape == b.shape && abs(a - b).max().item(Float.self) == 0)
                }
            }
        }

        /// Truncating into a captured window adopts exactly that token's
        /// captured convolution window and recurrent state (no off-by-one), the
        /// window start restores the pre-window state, and every truncation
        /// releases the window.
        @Test(arguments: [0, 1, 2, 3])
        func windowTruncationAdoptsTheKeptTokensState(kept: Int) throws {
            try Device.withDefaultDevice(.cpu) {
                func state(_ value: Float) -> GLM5NextLinearAttention.State {
                    .init(
                        convolution: MLXArray.full(
                            [1, 6, 4], values: MLXArray(value), type: Float.self),
                        recurrent: MLXArray.full(
                            [1, 2, 3, 3], values: MLXArray(value + 0.5), type: Float.self))
                }
                let cache = GLM5NextKDACache()
                cache.advance(state(100), count: 5)
                // Entry t is the state after window token t: values 1, 2, 3.
                let captured = GLM5NextKDASublayer.CapturedStates(
                    convolution: stacked((1 ... 3).map { state(Float($0)).convolution }, axis: 1),
                    recurrent: stacked((1 ... 3).map { state(Float($0)).recurrent }, axis: 1))
                cache.advance(state(3), count: 3, window: captured)
                #expect(cache.innerState().count == 6)
                try cache.truncate(to: 5 + kept)
                #expect(cache.offset == 5 + kept)
                let expected = state(kept == 0 ? 100 : Float(kept))
                #expect(cache.state.count == 2)
                #expect(abs(cache.state[0] - expected.convolution).max().item(Float.self) == 0)
                #expect(abs(cache.state[1] - expected.recurrent).max().item(Float.self) == 0)
                // No window survives a truncation: only ordinary rewinds remain.
                #expect(cache.innerState().count <= 4)
                if kept > 0 && kept < 3 {
                    #expect(throws: GLM5NextLoadError.self) { try cache.truncate(to: 5 + kept - 1) }
                }
            }
        }

        @Test func keepingIsOnlyForVerificationWindowsAndStaysInRange() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let stage = try f.stage()
                let cache = try prefilled(stage)
                let ordinary = try stage.prepareForward(tokens: tokens([9, 3]), cache: cache)
                #expect(throws: GLM5NextLoadError.self) {
                    try ordinary.commit(afterEvaluating: fence) { _ in 1 }
                }
                ordinary.discard()
                for kept in [0, 3] {
                    let candidate = try stage.prepareVerification(tokens: tokens([9, 3]), cache: cache)
                    #expect(throws: GLM5NextLoadError.self) {
                        try candidate.commit(afterEvaluating: fence) { _ in kept }
                    }
                    #expect(cache.offset == 7)
                }
                struct Refused: Error {}
                let refused = try stage.prepareVerification(tokens: tokens([9, 3]), cache: cache)
                #expect(throws: Refused.self) {
                    try refused.commit(afterEvaluating: fence) { _ in throw Refused() }
                }
                #expect(cache.offset == 7)
                #expect(throws: GLM5NextLoadError.self) {
                    try stage.prepareVerification(
                        tokens: tokens(
                            Array(repeating: 3, count: GLM5NextTextStage.maximumVerificationTokens + 1)),
                        cache: cache)
                }
                // A plain commit of a window keeps every token and its cache
                // still rewinds only to ordinary boundaries afterwards.
                try stage.prepareVerification(tokens: tokens([9, 3]), cache: cache)
                    .commit(afterEvaluating: fence)
                #expect(cache.offset == 9)
                #expect(throws: GLM5NextLoadError.self) { try cache.truncate(to: 8) }
            }
        }

        @Test func historyPrimingAndProposalsFollowPrime() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let stage = try f.stage()
                let mtp = try GLM5NextMTP(stage: stage, tensor: f.tensor)
                let prompt: [Int32] = [4, 5, 6, 7, 8, 9, 3]
                let cache = stage.newCache()
                let forward = try stage.prepareForward(tokens: tokens(prompt), cache: cache)
                try forward.commit(afterEvaluating: fence)
                guard case .final(let output) = forward.output else {
                    Issue.record("a whole-model stage reads out its hidden rows")
                    return
                }
                let shifted = tokens(Array(prompt.dropFirst()) + [2])
                // History-only priming stores the rows a full prime stores.
                let primed = try mtp.newRequestState()
                let full = try mtp.prime(
                    shiftedTokens: shifted, targetHidden: output.hidden, at: 0, state: primed)
                let historic = try mtp.newRequestState()
                try mtp.primeHistory(
                    shiftedTokens: shifted[0..., ..<5], targetHidden: output.hidden[0..., ..<5],
                    at: 0, state: historic)
                #expect(throws: GLM5NextMTPError.self) {
                    try mtp.primeHistory(
                        shiftedTokens: shifted[0..., 5...], targetHidden: output.hidden[0..., 5...],
                        at: 4, state: historic)
                }
                let twin = try mtp.newRequestState()
                try mtp.primeHistory(
                    shiftedTokens: shifted[0..., ..<5], targetHidden: output.hidden[0..., ..<5],
                    at: 0, state: twin)
                let proposed = try mtp.primeAndPropose(
                    shiftedTokens: shifted[0..., 5...], targetHidden: output.hidden[0..., 5...],
                    at: 5, state: historic, depth: 2)
                let step0 = try mtp.prime(
                    shiftedTokens: shifted[0..., 5...], targetHidden: output.hidden[0..., 5...],
                    at: 5, state: twin)
                #expect(proposed.shape == [1, 2] && proposed.dtype == .int32)
                try fence(
                    [full.logits, proposed, step0.logits] + (try mtp.evaluationTargets(state: primed))
                        + (try mtp.evaluationTargets(state: historic)))
                #expect(primed.committedInputCount == 7 && historic.committedInputCount == 7)
                for (left, right) in zip(
                    try mtp.evaluationTargets(state: historic).prefix(3),
                    try mtp.evaluationTargets(state: primed).prefix(3))
                {
                    close(left, right)
                }
                // Step 0 is the primed row's greedy token: the same graph as `prime`.
                #expect(
                    proposed[0, 0].item(Int32.self)
                        == argMax(step0.logits[0, -1], axis: -1).item(Int32.self))
                // Continuations append nothing.
                let checkpoint = try mtp.capture(state: historic)
                #expect(checkpoint.inputCount == 7 && !checkpoint.arrays.isEmpty)
                #expect(throws: GLM5NextMTPError.unsupportedDraftDepth(4)) {
                    try mtp.primeAndPropose(
                        shiftedTokens: tokens([5]), targetHidden: output.hidden[0..., ..<1],
                        at: 7, state: historic, depth: 4)
                }
            }
        }
    }
}
