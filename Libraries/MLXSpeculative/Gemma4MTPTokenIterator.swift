// Copyright © 2026 Apple Inc.
//
// `Gemma4MTPTokenIterator` — conforms to `TokenIteratorProtocol` so
// Gemma 4 MTP plugs into the existing `generate(iterator:...)` surface
// instead of requiring a separate `generateGemma4MTP(...)` entry point.
//
// Drives the same round-loop algorithm as `runGemma4MTPRounds` but as
// a synchronous iterator: each `next()` returns one token, rounds are
// filled into a pending buffer and drained one token at a time.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom

/// Single-batch (B=1) greedy MTP token iterator. Conforms to
/// ``TokenIteratorProtocol`` so callers can use the existing
/// ``generateTask(promptTokenCount:modelConfiguration:tokenizer:iterator:wiredMemoryTicket:)``
/// entry point.
///
/// Usage:
/// ```swift
/// let iter = try Gemma4MTPTokenIterator(
///     input: lmInput,
///     target: targetGemma4TextModel,
///     drafter: drafter,
///     parameters: GenerateParameters(maxTokens: 512, temperature: 0),
///     blockSize: 4)
/// let (stream, _) = generateTask(
///     promptTokenCount: lmInput.text.tokens.size,
///     modelConfiguration: modelContext.configuration,
///     tokenizer: modelContext.tokenizer,
///     iterator: iter)
/// for await gen in stream { ... }
/// ```
///
/// Greedy (`temperature=0`) only. Stochastic sampling would require
/// rejection-based speculative sampling which is deferred to a follow-up.
public struct Gemma4MTPTokenIterator: TokenIteratorProtocol {

    // Kept alive by the iterator so the GPU ops finish cleanly; not
    // inspected from the outside.
    private let target: Gemma4TextModel
    private let drafter: Gemma4AssistantDraftModel
    private var cache: [KVCache]
    private let blockSize: Int
    private let temperature: Float
    private let topP: Float
    private let topK: Int
    private let minP: Float
    private var rngKey: MLXArray

    // MTP state — mutated each round.
    private var bonus: Int
    private var hidden: MLXArray
    private var sharedKV: Gemma4SharedKV

    // Token stream state.
    private var pendingTokens: [Int] = []
    private var pendingIndex: Int = 0

    // TokenIteratorProtocol conformance.
    public var tokenCount: Int = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0

    /// Initialize — runs the prompt prefill and stages the first bonus.
    /// The first token emitted by the iterator is the bonus sampled from
    /// the last prefill position.
    ///
    /// - Parameters:
    ///   - input: prepared LM input. `input.text.tokens` is used for
    ///     prefill.
    ///   - target: the Gemma 4 target model.
    ///   - drafter: the MTP drafter. Will be bound to `target` on first
    ///     call (idempotent if already bound).
    ///   - cache: optional pre-allocated cache. If nil, allocated via
    ///     `target.newCache(parameters:)`.
    ///   - parameters: generation parameters. `maxTokens`, `temperature`,
    ///     `topP`, `topK`, `minP` are honored. At `temperature == 0` the
    ///     iterator uses the greedy walker (exact match with baseline).
    ///     At `temperature > 0` it uses rejection-based speculative
    ///     sampling with the same filter masks applied to both drafter
    ///     and target distributions. Emitted tokens match the distribution
    ///     that pure target sampling with the same filters would produce.
    ///   - blockSize: speculative block size (2–16). Default 4.
    ///   - rngSeed: seed for rejection sampling (only used when
    ///     `temperature > 0`). Default 0 which seeds from the current MLX
    ///     random state.
    /// - Throws: `Gemma4MTPError.invalidBlockSize` / bind errors.
    public init(
        input: LMInput,
        target: Gemma4TextModel,
        drafter: Gemma4AssistantDraftModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int = 4,
        rngSeed: UInt64 = 0
    ) throws {
        guard blockSize >= 2 && blockSize <= 16 else {
            throw Gemma4MTPError.invalidBlockSize(blockSize)
        }
        try drafter.bind(target: target)

        self.target = target
        self.drafter = drafter
        self.cache = cache ?? target.newCache(parameters: parameters)
        self.blockSize = blockSize
        self.maxTokens = parameters.maxTokens
        self.temperature = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        self.minP = parameters.minP
        self.rngKey = rngSeed == 0
            ? MLXRandom.key(UInt64(Date().timeIntervalSince1970 * 1e6))
            : MLXRandom.key(rngSeed)

        // Prefill.
        let prefillStart = Date()
        var promptTokens = input.text.tokens
        if promptTokens.ndim == 1 {
            promptTokens = promptTokens[.newAxis, .ellipsis]
        }
        let prefillOut = target.forwardForMTP(promptTokens, cache: self.cache)
        let lastLogits = prefillOut.logits[0..., -1, 0...]
        let firstBonus: Int
        if parameters.temperature == 0 {
            let firstBonusArr = lastLogits.asType(.float32).argMax(axis: -1)
            eval(firstBonusArr)
            firstBonus = Int(firstBonusArr.item(Int32.self))
        } else {
            // Sample the first bonus from the target's temperature-scaled
            // softmax with the same filters the rest of the round loop
            // will use. Split the rng key so the round loop gets fresh
            // keys each call.
            let keys = MLXRandom.split(key: self.rngKey, into: 2)
            let idx = sampleLogitsWithFilters(
                lastLogits,
                temperature: parameters.temperature,
                topP: parameters.topP, topK: parameters.topK, minP: parameters.minP,
                rngKey: keys[0])
            eval(idx)
            firstBonus = Int(idx.item(Int32.self))
            self.rngKey = keys[1]
        }
        self.bonus = firstBonus
        self.hidden = prefillOut.lastHidden[
            0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
        self.sharedKV = prefillOut.capturedSharedKV
        self.promptPrefillTime = -prefillStart.timeIntervalSinceNow

        // First emitted token is the prefill bonus.
        self.pendingTokens.append(self.bonus)
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain pending tokens before the next round.
        if pendingIndex < pendingTokens.count {
            let t = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return t
        }

        // All drained — run one MTP round to refill.
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        let remaining = (maxTokens.map { $0 - tokenCount }) ?? blockSize
        guard remaining > 0 else { return nil }
        let bs = Swift.min(blockSize, remaining + 1)
        if bs <= 1 { return nil }
        let k = bs - 1

        // Draft k tokens, GPU-resident.
        // At temperature > 0 we also need to retain per-step drafter
        // logits so the walker can compute q(c_i) for rejection sampling.
        let driveOffset = cache[0].offset
        var tok = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
        var h = hidden
        var draftPerStep: [MLXArray] = []
        var draftLogitsPerStep: [MLXArray] = []  // only populated when temperature > 0
        draftPerStep.reserveCapacity(k)
        if temperature > 0 {
            draftLogitsPerStep.reserveCapacity(k)
        }
        for _ in 0 ..< k {
            let tokEmbed = target.embedTokensForDrafter(tok)
            let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
            let (newH, logits) = drafter(
                inputsEmbeds: inputsEmbeds,
                sharedKV: sharedKV,
                positionOffset: .scalar(driveOffset))
            let stepLogits = logits.squeezed(axis: 1)  // [1, vocab]
            let sampled: MLXArray
            if temperature > 0 {
                draftLogitsPerStep.append(stepLogits)
                let keys = MLXRandom.split(key: rngKey, into: 2)
                rngKey = keys[0]
                sampled = sampleLogitsWithFilters(
                    stepLogits,
                    temperature: temperature,
                    topP: topP, topK: topK, minP: minP,
                    rngKey: keys[1])
            } else {
                sampled = stepLogits.argMax(axis: -1)
            }
            let sampled2d = sampled[.newAxis, .ellipsis]
            draftPerStep.append(sampled2d)
            tok = sampled2d
            h = newH
        }

        // Verify.
        let bonusCol = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
        let verifyInput: MLXArray =
            draftPerStep.isEmpty
            ? bonusCol
            : concatenated([bonusCol] + draftPerStep, axis: 1)
        let verifyOut = target.forwardForMTP(verifyInput, cache: cache)
        // fp32 argmax at verify to avoid bf16 near-uniform-tail argmax
        // flips that would otherwise diverge from baseline.
        let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
        let draftConcat: MLXArray =
            draftPerStep.isEmpty
            ? MLXArray.zeros([1, 0], dtype: .int32)
            : concatenated(draftPerStep, axis: 1)
        eval(mainTokens, draftConcat)
        let mainInts = mainTokens.squeezed(axis: 0).asArray(Int.self)
        let draftTokens = draftConcat.squeezed(axis: 0).asArray(Int32.self)
                                     .map { Int($0) }

        // Walk: greedy or rejection-based depending on temperature.
        let accepted: Int
        let newTokens: [Int]
        if temperature > 0 && k > 0 {
            // Gather verify logits at positions 0..k (bs=k+1 positions).
            let verifyLogits = verifyOut.logits.squeezed(axis: 0)  // [bs, vocab]
            // Stack drafter logits into [k, vocab].
            let draftLogits = concatenated(draftLogitsPerStep, axis: 0)
            let keys = MLXRandom.split(key: rngKey, into: 2)
            rngKey = keys[0]
            let result = speculativeSampleRound(
                draftLogits: draftLogits,
                draftTokens: draftTokens,
                verifyLogits: verifyLogits,
                temperature: temperature,
                topP: topP, topK: topK, minP: minP,
                rngKey: keys[1])
            accepted = result.accepted
            newTokens = result.emitted
        } else {
            var a = 0
            while a < k && draftTokens[a] == mainInts[a] {
                a += 1
            }
            accepted = a
            newTokens = Array(mainInts[0 ..< a + 1])
        }

        // Rewind.
        if accepted < k {
            target.rollbackSpeculativeCache(
                cache, accepted: .scalar(accepted), blockSize: bs)
        }

        // Update state.
        hidden = verifyOut.lastHidden[0..., accepted ..< accepted + 1, 0...]
        bonus = newTokens.last!
        let rejected = k - accepted
        sharedKV = Gemma4SharedKV.sliceTail(
            from: verifyOut.capturedSharedKV, rejected: rejected)

        pendingTokens.append(contentsOf: newTokens)
        if pendingTokens.isEmpty { return nil }
        let t = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return t
    }
}
