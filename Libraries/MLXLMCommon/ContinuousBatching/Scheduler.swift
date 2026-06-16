// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/scheduler.py — Continuous batching scheduler.
// https://github.com/jundot/omlx/blob/main/omlx/scheduler.py
//
import Foundation
import MLX

final class TokenHistoryHolder: @unchecked Sendable {
    var tokens: [Int]
    init(tokens: [Int]) { self.tokens = tokens }
}

func makeRepetitionSampler(
    base: @escaping RowSampler,
    history: TokenHistoryHolder,
    repetitionPenalty: Float,
    presencePenalty: Float,
    frequencyPenalty: Float
) -> RowSampler {
    return { @Sendable logits in
        let tokens = history.tokens
        guard !tokens.isEmpty else { return base(logits) }
        let vocab = logits.dim(-1)

        var counts: [Int: Int] = [:]
        for t in tokens where t >= 0 && t < vocab {
            counts[t, default: 0] += 1
        }
        guard !counts.isEmpty else { return base(logits) }

        eval(logits)
        var flat = logits.reshaped(-1).asArray(Float.self)
        for (tokenId, count) in counts {
            var v = flat[tokenId]
            if repetitionPenalty != 1.0 {
                v = v > 0 ? v / repetitionPenalty : v * repetitionPenalty
            }
            v -= presencePenalty
            v -= frequencyPenalty * Float(count)
            flat[tokenId] = v
        }
        return base(MLXArray(flat).reshaped(logits.shape))
    }
}

/// Which quantized batched-cache implementation to use for full-attention
/// layers when KV quantization is enabled.
public enum KVQuantCacheKind: Sendable, Equatable {
    /// Use ``QuantizedBatchKVCache`` and the native quantized attention kernel.
    case kernel
    /// Use ``DequantBatchKVCache``: stores quantized, but returns dequantized
    /// fp16 so the model's regular attention path (including sinks) is used.
    case dequant
}

/// Configuration for per-layer KV-cache quantization in the continuous-batching
/// scheduler. When set on ``SchedulerConfig/kvQuantization``, full-attention
/// layers use ``QuantizedBatchKVCache`` (or ``DequantBatchKVCache`` when
/// ``cacheKind`` is ``KVQuantCacheKind/dequant``); rotating / sliding-window
/// layers keep ``BatchRotatingKVCache``.
public struct KVQuantizationConfig: Sendable {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let cacheKind: KVQuantCacheKind

    public init(
        groupSize: Int = 128,
        bits: Int = 8,
        mode: QuantizationMode = .affine,
        cacheKind: KVQuantCacheKind = .kernel
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.cacheKind = cacheKind
    }
}

public struct SchedulerConfig: Sendable {
    public var maxNumSeqs: Int
    public var maxNumBatchedTokens: Int
    /// Tokens processed per prefill chunk. Smaller values give finer interleaving
    /// with decode at the cost of slightly higher per-step overhead.
    public var prefillStepSize: Int
    public var streamInterval: Int
    /// Maximum total KV-cache tokens across all running requests (prompt + output).
    /// When a new admission would exceed this, the running request with the
    /// largest KV footprint is preempted and re-queued. 0 = unlimited.
    public var maxKVCacheTokens: Int
    /// Optional KV-cache quantization configuration. When non-nil, full-attention
    /// layers are backed by ``QuantizedBatchKVCache``; sliding-window layers
    /// remain ``BatchRotatingKVCache``.
    public var kvQuantization: KVQuantizationConfig?

    public init(
        maxNumSeqs: Int = 64,
        maxNumBatchedTokens: Int = 8192,
        prefillStepSize: Int = 512,
        streamInterval: Int = 1,
        maxKVCacheTokens: Int = 0,
        kvQuantization: KVQuantizationConfig? = nil
    ) {
        self.maxNumSeqs = maxNumSeqs
        self.maxNumBatchedTokens = maxNumBatchedTokens
        self.prefillStepSize = prefillStepSize
        self.streamInterval = streamInterval
        self.maxKVCacheTokens = maxKVCacheTokens
        self.kvQuantization = kvQuantization
    }
}

public struct SchedulerOutput: Sendable {
    public var scheduledRequestIds: [String]
    public var numScheduledTokens: Int
    public var finishedRequestIds: Set<String>
    public var outputs: [RequestOutput]
    public var hasWork: Bool

    public init(
        scheduledRequestIds: [String] = [],
        numScheduledTokens: Int = 0,
        finishedRequestIds: Set<String> = [],
        outputs: [RequestOutput] = [],
        hasWork: Bool = false
    ) {
        self.scheduledRequestIds = scheduledRequestIds
        self.numScheduledTokens = numScheduledTokens
        self.finishedRequestIds = finishedRequestIds
        self.outputs = outputs
        self.hasWork = hasWork
    }
}

// One admitted request plus the routing info computed at admission.
private struct AdmittedEntry {
    let request: Request
    let promptTokens: [Int]
    let tokensToPrefill: [Int]
    let existingCache: [KVCache]?
    /// Per-layer single-stream caches restored from a checkpoint hit (mixed
    /// simple + rotating). Non-nil routes the entry through the dedicated
    /// B==1 restore path, NOT the KVCacheSimple-only warm path.
    /// `tokensToPrefill` is then the suffix after the restored prefix.
    let restoredCaches: [any KVCache]?
    let uid: Int
    let sampler: RowSampler?
    let machine: SequenceStateMachine
}

// Holds a batch of cold sequences being prefilled incrementally across steps.
private struct PendingPrefill {
    // Accumulates KV state across prompt() calls.
    var ppBatch: PromptProcessingBatch
    // Tokens yet to be forwarded for each sequence (excludes the seed token).
    var remaining: [[Int]]
    // Last token of each sequence; used as the seed for GenerationBatch.
    var seeds: [Int]
}

public final class Scheduler: @unchecked Sendable {
    public let config: SchedulerConfig
    /// Runtime-adjustable cap on concurrently running sequences. Initialised
    /// from `config.maxNumSeqs` and mutated via `setMaxNumSeqs(_:)`. The
    /// admission path consults this rather than `config.maxNumSeqs` so the
    /// engine can be reshaped at runtime (e.g. by an adaptive concurrency
    /// policy in the embedder). Must be mutated from the engine queue or
    /// before `EngineCore.start()`; see `setMaxNumSeqs` below.
    private var runtimeMaxNumSeqs: Int
    public let model: any LanguageModel
    private let tokenizer: any Tokenizer
    private let eosTokenIds: Set<Int>

    /// Optional drafter-based MTP runtime (Gemma 4). Set by the provider after
    /// loading a compatible drafter + flag; passed into every `GenerationBatch`
    /// the scheduler builds. Mutate before `EngineCore.start()` (same contract
    /// as `runtimeMaxNumSeqs`).
    public var mtpRuntime: (any BatchedMTPRuntime)?

    private var waiting: [Request] = []
    private var requests: [String: Request] = [:]
    public private(set) var finishedReqIds: Set<String> = []

    private var activeRids: [String] = []
    private var activeSamplers: [String: @Sendable (MLXArray) -> MLXArray] = [:]
    private var activeDetokenizers: [String: NaiveStreamingDetokenizer] = [:]
    private var activeStreamStates: [String: RequestStreamState] = [:]
    private var tokenHistories: [String: TokenHistoryHolder] = [:]

    private var uidCounter: Int = 0
    private var uidToRid: [Int: String] = [:]
    private var ridToUid: [String: Int] = [:]

    private var genBatch: GenerationBatch?

    // In-progress chunked cold prefill.
    private var pendingPrefill: PendingPrefill?

    private var pendingAbortIds: Set<String> = []

    public var prefixCache: PrefixCache?

    /// Checkpoint KV-cache capture hook (hybrid sliding-window models).
    /// When set, cold single-row prefill is boundary-aligned (see
    /// `CheckpointPrefillPlanner`) and this closure is invoked at each
    /// checkpoint boundary with the prompt-prefix tokens, the boundary
    /// length, and the per-layer single-row caches extracted at that point.
    /// Default nil ⇒ no capture and `advancePendingPrefill` is byte-identical
    /// to the pre-hook behavior. The closure must be cheap/non-blocking
    /// (the provider boxes the caches and stores via a detached Task).
    ///
    /// Note: prefill covers only the first `promptLength - 1` tokens (the
    /// final token is the decode seed, forwarded later inside `generate()`).
    /// So a boundary equal to the full prompt length never fires — reachable
    /// boundaries are those `<= promptLength - 1`. This is correct for a
    /// prefix checkpoint (the cache genuinely covers `checkpointLength`
    /// tokens at fire time); callers just shouldn't expect a capture at
    /// exactly `promptLength`.
    public var onCheckpointCapture: (@Sendable (_ prefixTokens: [Int], _ checkpointLength: Int, _ caches: [any KVCache]) -> Void)?
    /// Checkpoint boundaries for capture, derived from the model's sliding
    /// window. Empty ⇒ capture disabled even if the hook is set.
    public var checkpointBoundaries: [Int] = []

    public private(set) var numRequestsProcessed: Int = 0
    public private(set) var totalPromptTokens: Int = 0
    public private(set) var totalCompletionTokens: Int = 0

    public init(
        model: any LanguageModel,
        tokenizer: any Tokenizer,
        config: SchedulerConfig = SchedulerConfig(),
        eosTokenIds: Set<Int> = [],
        prefixCache: PrefixCache? = nil
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.runtimeMaxNumSeqs = config.maxNumSeqs
        self.eosTokenIds = eosTokenIds
        self.prefixCache = prefixCache
    }

    /// True when checkpoint capture should run for the current pending
    /// prefill: hook installed, boundaries present, and a single cold row
    /// (B==1 — extract is row-isolated for any B, but B==1 sidesteps the
    /// untested left-padded cross-row restore, per the design).
    private func checkpointCaptureActive(_ pp: PendingPrefill) -> Bool {
        onCheckpointCapture != nil && !checkpointBoundaries.isEmpty
            && pp.ppBatch.tokens.count == 1
    }

    /// Update the runtime concurrency cap. `value < 1` is clamped to 1 to
    /// prevent permanent admission deadlock (a cap of 0 would block every
    /// future request forever via the runtime setter).
    ///
    /// Thread-safety: like the rest of `Scheduler`'s mutating API, this must
    /// be invoked from the engine queue (i.e. via
    /// `EngineCore.setMaxNumSeqs(_:)`) or before `EngineCore.start()`. The
    /// new value is observed by the next `admitWaiting()` call.
    public func setMaxNumSeqs(_ value: Int) {
        runtimeMaxNumSeqs = max(1, value)
    }

    public func addRequest(_ request: Request) {
        requests[request.requestId] = request
        request.status = .waiting
        waiting.append(request)
    }

    public func abortRequest(_ requestId: String) -> Bool {
        guard requests[requestId] != nil else { return false }
        pendingAbortIds.insert(requestId)
        return true
    }

    /// Remove the decode-batch row for `uid`, if present. No-op when the row has
    /// already left the batch (e.g. it finished on the normal path), so it is safe
    /// to call from the abort, cleanup, and preemption paths in any order.
    private func dropRowFromBatch(uid: Int) {
        guard let batch = genBatch else { return }
        let keep = (0..<batch.uids.count).filter { batch.uids[$0] != uid }
        if keep.count != batch.uids.count {
            batch.filter(keep: keep)
        }
    }

    public func removeFinishedRequest(_ requestId: String) {
        requests.removeValue(forKey: requestId)
        activeSamplers.removeValue(forKey: requestId)
        activeDetokenizers.removeValue(forKey: requestId)
        activeStreamStates.removeValue(forKey: requestId)
        tokenHistories.removeValue(forKey: requestId)
        activeRids.removeAll { $0 == requestId }
        if let uid = ridToUid[requestId] {
            // Drop the row from the live batch too. Otherwise a removeFinishedRequest
            // that wins the race against the deferred doAbortRequest leaves an
            // orphaned, still-decoding row in genBatch whose outputs map to no request.
            dropRowFromBatch(uid: uid)
            uidToRid.removeValue(forKey: uid)
        }
        ridToUid.removeValue(forKey: requestId)
        // Mirror doAbortRequest's prefix-cache release so the abort-vs-cleanup race
        // can't permanently pin borrowed prefix blocks (refCount never reaches 0 →
        // never evictable). Idempotent: a no-op once the request's blocks have
        // already been released on the normal path.
        prefixCache?.releaseRequest(requestId)
    }

    public func hasRequests() -> Bool {
        !waiting.isEmpty || pendingPrefill != nil || (genBatch != nil && !genBatch!.isEmpty)
    }

    public func getNumWaiting() -> Int { waiting.count }
    public func getNumRunning() -> Int { genBatch?.batchSize ?? 0 }

    // MARK: - Main step

    public func step() -> SchedulerOutput {
        var output = SchedulerOutput()

        processPendingAborts()

        // 1. DECODE — advance all in-flight sequences by one token first.
        //    This ensures running requests are never starved by incoming prefill work.
        if let batch = genBatch, !batch.isEmpty {
            let responses = batch.next()
            output.hasWork = true

            let stepOutputs = processGenResponses(responses)
            output.outputs = stepOutputs

            for r in responses where r.finishReason != nil {
                let rid = uidToRid[r.uid] ?? ""
                output.finishedRequestIds.insert(rid)
            }
            cleanupFinished(output.finishedRequestIds)
        }

        // 2. PREFILL — advance the pending cold batch by one chunk, capped by
        //    whatever token budget remains after decode.  This is the vLLM-style
        //    shared max_num_batched_tokens budget: decode is served first, prefill
        //    gets the remainder up to prefillStepSize.
        let decodedThisStep = genBatch?.batchSize ?? 0
        advancePendingPrefill(decodeBatchSize: decodedThisStep)
        if pendingPrefill != nil { output.hasWork = true }

        // 3. ADMIT — warm (prefix-cached) sequences are promoted to genBatch immediately.
        //    Cold sequences start a new pending-prefill batch; only one cold batch is
        //    in flight at a time so all staggered arrivals are grouped efficiently.
        let newScheduled = admitWaiting()
        for r in newScheduled {
            output.scheduledRequestIds.append(r.requestId)
            output.numScheduledTokens += r.numPromptTokens
        }
        if !newScheduled.isEmpty || genBatch != nil { output.hasWork = true }

        return output
    }

    // MARK: - Abort handling

    private func processPendingAborts() {
        for rid in pendingAbortIds { doAbortRequest(rid) }
        pendingAbortIds.removeAll()
    }

    private func doAbortRequest(_ requestId: String) {
        guard let request = requests[requestId] else { return }

        if request.status == .waiting {
            waiting.removeAll { $0.requestId == requestId }
        }

        if let uid = ridToUid[requestId] {
            dropRowFromBatch(uid: uid)
            activeRids.removeAll { $0 == requestId }
            removeFinishedRequest(requestId)
        }

        prefixCache?.releaseRequest(requestId)
        request.setFinished(.finishedAborted)
        finishedReqIds.insert(requestId)
        requests.removeValue(forKey: requestId)
    }

    // MARK: - Chunked prefill

    /// Advance the pending cold-prefill batch by one chunk.
    ///
    /// `decodeBatchSize` is the number of tokens already consumed by the decode
    /// step this cycle.  The prefill chunk is capped at
    /// `max(1, maxNumBatchedTokens - decodeBatchSize)` so that decode and prefill
    /// together never exceed the configured token budget.  This mirrors vLLM's
    /// shared `max_num_batched_tokens` budget.
    private func advancePendingPrefill(decodeBatchSize: Int = 0) {
        guard var pp = pendingPrefill else { return }

        let maxRemaining = pp.remaining.map { $0.count }.max() ?? 0

        if maxRemaining == 0 {
            // Hold the completed prefill while the live genBatch is mid-MTP
            // session: `extend()` would grow `uids`/caches without growing the
            // session's per-row carry, tripping the round precondition. Merge
            // once the session ends (the batch empties → mtpSession cleared in
            // GenerationBatch.filter). pendingPrefill is unmutated here, so a
            // bare return safely retains it for a later step.
            if genBatch?.hasActiveMTPSession == true { return }
            // All sequences prefilled — transition to decode.
            let gen = pp.ppBatch.generate(lastTokensOf: pp.seeds.map { [$0] }, mtpRuntime: mtpRuntime)
            mergeIntoGenBatch(gen)
            pendingPrefill = nil
            return
        }

        // Budget: how many prefill tokens we're allowed this step.
        let prefillBudget = max(1, config.maxNumBatchedTokens - decodeBatchSize)
        // Chunk size: never exceed the per-step budget or the remaining work.
        var chunkSize = min(config.prefillStepSize, prefillBudget, maxRemaining)

        // Checkpoint capture (hybrid models, B==1, hook installed): cap the
        // chunk so the prefilled count lands exactly on a checkpoint boundary,
        // then snapshot the per-layer caches there. Pure planner decides the
        // chunk; default path below is untouched when inactive.
        var captureAt: Int? = nil
        let captureActive = checkpointCaptureActive(pp)
        if captureActive {
            // The cache covers (full prompt length - remaining) tokens so far.
            // For B==1 the single row's prompt length is its seed + its
            // already-consumed remaining.
            let prefilled = (pp.ppBatch.tokens.first?.count ?? 0)
            let step = CheckpointPrefillPlanner.plan(
                prefilled: prefilled,
                defaultChunk: chunkSize,
                remaining: pp.remaining.first?.count ?? 0,
                boundaries: checkpointBoundaries
            )
            chunkSize = step.chunk
            captureAt = step.captureAt
        }

        let chunks = pp.remaining.map { Array($0.prefix(chunkSize)) }

        pp.ppBatch.prompt(chunks)

        pp.remaining = zip(pp.remaining, chunks).map { rem, chunk in
            Array(rem.dropFirst(chunk.count))
        }

        // Fire the capture hook AFTER the chunk is prefilled (so the cache
        // covers exactly `captureAt` tokens) but BEFORE generate()'s in-init
        // decode step would slide a rotating window.
        if captureActive, let boundary = captureAt, let hook = onCheckpointCapture {
            let prefixTokens = Array((pp.ppBatch.tokens.first ?? []).prefix(boundary))
            if prefixTokens.count == boundary {
                let caches = pp.ppBatch.promptCache.map { $0.extractBatched(0) }
                hook(prefixTokens, boundary, caches)
            }
        }

        if pp.remaining.allSatisfy({ $0.isEmpty }) {
            // Hold (don't merge) while a live MTP session owns the genBatch;
            // the merge will fire from the maxRemaining==0 branch on a later
            // step once the session ends. Store the now-complete prefill back.
            if genBatch?.hasActiveMTPSession == true {
                pendingPrefill = pp
                return
            }
            let gen = pp.ppBatch.generate(lastTokensOf: pp.seeds.map { [$0] }, mtpRuntime: mtpRuntime)
            mergeIntoGenBatch(gen)
            pendingPrefill = nil
        } else {
            pendingPrefill = pp
        }
    }

    private func mergeIntoGenBatch(_ gen: GenerationBatch) {
        if let existing = genBatch, !existing.isEmpty {
            existing.extend(gen)
        } else {
            genBatch = gen
        }
    }

    // MARK: - KV-cache budget & preemption

    /// Total non-cached KV tokens currently held by all running requests.
    private var currentKVTokens: Int {
        activeRids.compactMap { requests[$0] }
            .reduce(0) { $0 + $1.numPromptTokens - $1.cachedTokens + $1.numOutputTokens }
    }

    /// Evict the running request with the largest KV footprint back to the
    /// front of the waiting queue so its KV cache is freed.
    /// Returns the preempted request ID, or nil if nothing is running.
    @discardableResult
    private func preemptOne() -> String? {
        guard !activeRids.isEmpty else { return nil }

        // Pick the request with the most KV tokens (frees the most memory).
        let victim = activeRids
            .compactMap { rid -> (String, Int)? in
                guard let r = requests[rid] else { return nil }
                return (rid, r.numPromptTokens - r.cachedTokens + r.numOutputTokens)
            }
            .max(by: { $0.1 < $1.1 })
            .map { $0.0 }

        guard let rid = victim, let request = requests[rid] else { return nil }

        // Remove from genBatch.
        if let uid = ridToUid[rid] {
            dropRowFromBatch(uid: uid)
        }

        // Tear down per-request scheduler state.
        activeRids.removeAll { $0 == rid }
        activeSamplers.removeValue(forKey: rid)
        activeDetokenizers.removeValue(forKey: rid)
        activeStreamStates.removeValue(forKey: rid)
        tokenHistories.removeValue(forKey: rid)
        if let uid = ridToUid[rid] { uidToRid.removeValue(forKey: uid) }
        ridToUid.removeValue(forKey: rid)

        // Release any prefix-cache slot so re-admission gets a fresh lookup.
        prefixCache?.releaseRequest(rid)

        // Reset generation state — output so far is discarded.
        request.outputTokenIds = []
        request.outputText = ""
        request.numComputedTokens = 0
        request.cachedTokens = 0
        request.status = .preempted

        // Re-queue at the front so it's the next request admitted.
        waiting.insert(request, at: 0)

        return rid
    }

    // MARK: - Admission

    /// Admit waiting requests.
    /// - Warm sequences (prefix-cache hits) run `doExternalPrefill` and go straight
    ///   to `genBatch`.
    /// - Cold sequences are queued into `pendingPrefill` for chunked advancement.
    ///   Only one cold batch is in flight at a time so staggered arrivals are grouped
    ///   into the next available batch for efficient GPU utilization.
    @discardableResult
    private func admitWaiting() -> [Request] {
        guard !waiting.isEmpty else { return [] }
        // Defer admission while a drafter-MTP session is live: it carries
        // ragged per-row pending tokens that the shared-index batched cache
        // cannot absorb via `extend` mid-stream. The batch runs to completion
        // (rows evict as they finish); waiting requests are admitted into a
        // fresh batch once it empties. (Mid-session growth is a follow-up.)
        if genBatch?.hasActiveMTPSession == true { return [] }
        let availableSlots = max(0, runtimeMaxNumSeqs - activeRids.count)
        guard availableSlots > 0 else { return [] }

        var admitted: [AdmittedEntry] = []
        var newScheduled: [Request] = []
        let admitCold = pendingPrefill == nil

        for _ in 0 ..< min(availableSlots, waiting.count) {
            let request = waiting.removeFirst()

            if request.promptTokenIds == nil {
                let tokens: [Int]
                if let stringPrompt = request.prompt as? String {
                    tokens = tokenizer.encode(text: stringPrompt)
                } else if let tokenArray = request.prompt as? [Int] {
                    tokens = tokenArray
                } else {
                    continue
                }
                request.promptTokenIds = tokens
                request.numPromptTokens = tokens.count
            }
            guard let promptTokens = request.promptTokenIds, !promptTokens.isEmpty else {
                continue
            }

            var tokensToPrefill = promptTokens
            var existingCache: [KVCache]? = nil
            var restoredCaches: [any KVCache]? = nil

            // Checkpoint restore (hybrid models) takes precedence: the
            // provider attached a per-layer mixed cache covering the first
            // `tokenCount` prompt tokens. Only valid for a real prefix
            // (1 <= tokenCount < promptTokens.count) so there is a suffix to
            // decode from; otherwise ignore and fall through to cold.
            if let rc = request.restoredCheckpoint,
               rc.tokenCount >= 1, rc.tokenCount < promptTokens.count,
               rc.caches.count == model.newCache(parameters: nil).count
            {
                restoredCaches = rc.caches
                tokensToPrefill = Array(promptTokens[rc.tokenCount...])
                request.cachedTokens = rc.tokenCount
            } else if let pc = prefixCache {
                let (cached, remaining) = pc.fetchPrefix(
                    requestId: request.requestId, tokens: promptTokens)
                if let cached {
                    tokensToPrefill = remaining
                    existingCache = cached.map { $0 as KVCache }
                    request.cachedTokens = promptTokens.count - remaining.count
                }
            }

            // A restored-checkpoint request is "warm" (not cold) for batch
            // gating, but goes through its own admit path below.
            let isCold = existingCache == nil && restoredCaches == nil
            if isCold && !admitCold {
                // A cold batch is already in flight; push back and wait for it to finish.
                waiting.insert(request, at: 0)
                break
            }

            // KV-cache budget check: if admitting this request would push total
            // in-flight KV tokens over the limit, preempt the heaviest runner first.
            if config.maxKVCacheTokens > 0 {
                let newTokens = request.numPromptTokens - request.cachedTokens
                while config.maxKVCacheTokens > 0
                    && currentKVTokens + newTokens > config.maxKVCacheTokens
                    && !activeRids.isEmpty
                {
                    preemptOne()
                }
            }

            let uid = uidCounter
            uidCounter += 1
            uidToRid[uid] = request.requestId
            ridToUid[request.requestId] = uid

            request.status = .running
            activeRids.append(request.requestId)

            let params = request.samplingParams
            let baseSampler = makeRowSampler(
                temperature: params.temperature,
                topP: params.topP,
                minP: params.minP,
                topK: params.topK
            )
            let needsPenalty = params.repetitionPenalty != 1.0
                || params.presencePenalty != 0.0
                || params.frequencyPenalty != 0.0
            let history = TokenHistoryHolder(tokens: promptTokens)
            tokenHistories[request.requestId] = history
            // Greedy (temperature 0, no penalty) uses a nil sampler so the
            // GenerationBatch takes its argMax fast path. `makeRowSampler(0)`
            // is exactly argMax, so this is behavior-preserving — and it's the
            // condition the drafter-MTP path keys on (it speculates greedily,
            // so it only engages when every row is nil-sampler greedy). With a
            // non-nil baseSampler here, MTP would never become eligible.
            let isGreedy = params.temperature == 0 && !needsPenalty
            let sampler: RowSampler? = needsPenalty
                ? makeRepetitionSampler(
                    base: baseSampler, history: history,
                    repetitionPenalty: params.repetitionPenalty,
                    presencePenalty: params.presencePenalty,
                    frequencyPenalty: params.frequencyPenalty)
                : (isGreedy ? nil : baseSampler)

            activeSamplers[request.requestId] = sampler
            activeDetokenizers[request.requestId] = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            activeStreamStates[request.requestId] = RequestStreamState(
                streamInterval: config.streamInterval
            )

            totalPromptTokens += request.numPromptTokens
            newScheduled.append(request)
            admitted.append(AdmittedEntry(
                request: request,
                promptTokens: promptTokens,
                tokensToPrefill: tokensToPrefill,
                existingCache: existingCache,
                restoredCaches: restoredCaches,
                uid: uid,
                sampler: sampler,
                machine: makeStateMachine(for: request)
            ))
        }

        guard !admitted.isEmpty else { return newScheduled }

        // Checkpoint-restored entries run their own B==1 mixed-cache path;
        // they are neither "cold" nor the KVCacheSimple-only "warm" path.
        let restored = admitted.filter { $0.restoredCaches != nil }
        let cold = admitted.filter { $0.existingCache == nil && $0.restoredCaches == nil }
        let warm = admitted.filter { $0.existingCache != nil && $0.restoredCaches == nil }

        // Restore: one GenerationBatch per restored request (B==1 each), via
        // a PromptProcessingBatch seeded with the restored batched caches.
        // A failed rebuild re-queues the request (admitColdFallback) — drop
        // it from newScheduled so step() doesn't report it as scheduled.
        for entry in restored where !admitRestoredCheckpoint(entry) {
            newScheduled.removeAll { $0.requestId == entry.request.requestId }
        }

        // Warm: run doExternalPrefill immediately and merge into genBatch.
        if !warm.isEmpty {
            var warmRowCaches: [[KVCacheSimple]] = []
            var warmUids: [Int] = []
            var warmSeedTokens: [Int] = []
            var warmMaxTokens: [Int] = []
            var warmSamplers: [RowSampler?] = []
            var warmMachines: [SequenceStateMachine] = []
            var warmTokenLists: [[Int]] = []

            for entry in warm {
                let (simpleCaches, seedTokens) = doExternalPrefill(
                    tokens: entry.tokensToPrefill,
                    existingCache: entry.existingCache
                )
                warmRowCaches.append(simpleCaches)
                warmUids.append(entry.uid)
                warmSeedTokens.append(seedTokens.last ?? entry.promptTokens.last ?? 0)
                warmMaxTokens.append(entry.request.maxTokens)
                warmSamplers.append(entry.sampler)
                warmMachines.append(entry.machine)
                warmTokenLists.append(entry.promptTokens)
            }

            let numLayers = warmRowCaches[0].count
            let warmPerLayer: [any BatchedCache] = (0 ..< numLayers).map { layer in
                BatchKVCache.merge(warmRowCaches.map { $0[layer] })
            }
            let warmGen = GenerationBatch(
                model: model,
                uids: warmUids,
                seedTokens: MLXArray(warmSeedTokens),
                promptCache: warmPerLayer,
                tokens: warmTokenLists,
                maxTokens: warmMaxTokens,
                samplers: warmSamplers,
                fallbackSampler: greedySampler,
                stateMachines: warmMachines,
                mtpRuntime: mtpRuntime
            )
            mergeIntoGenBatch(warmGen)
        }

        // Cold: set up pending prefill for chunked advancement.
        if !cold.isEmpty {
            let batchCache = makeBatchedCache(batchSize: cold.count)
            let ppBatch = PromptProcessingBatch(
                model: model,
                uids: cold.map { $0.uid },
                promptCache: batchCache,
                tokens: Array(repeating: [], count: cold.count),
                maxTokens: cold.map { $0.request.maxTokens },
                prefillStepSize: config.prefillStepSize,
                samplers: cold.map { $0.sampler },
                fallbackSampler: greedySampler,
                stateMachines: cold.map { $0.machine }
            )
            let remaining = cold.map { entry -> [Int] in
                let toks = entry.tokensToPrefill
                return toks.count > 1 ? Array(toks.dropLast()) : []
            }
            let seeds = cold.map { $0.tokensToPrefill.last ?? 0 }
            pendingPrefill = PendingPrefill(ppBatch: ppBatch, remaining: remaining, seeds: seeds)
        }

        return newScheduled
    }

    /// Admit a checkpoint-restored request (hybrid sliding-window models).
    /// Builds a B==1 batched cache from the restored per-layer single-stream
    /// caches (the inverse of `extractBatched`: `merge([c])` for full layers,
    /// `fromSingleRow` for sliding layers), seeds a `PromptProcessingBatch`
    /// that already holds the restored prefix, and prefills + decodes ONLY
    /// the suffix. This reuses the exact prefill→decode machinery a cold
    /// request uses (so the resulting GenerationBatch is shape-identical),
    /// but seeded from the restored cache instead of an empty one. Does NOT
    /// touch `doExternalPrefill` or `BatchKVCache.merge`'s KVCacheSimple-only
    /// warm path — it is a separate, additive branch.
    /// Returns true if the restore succeeded, false if it fell back to cold
    /// (caller then drops the entry from newScheduled).
    @discardableResult
    private func admitRestoredCheckpoint(_ entry: AdmittedEntry) -> Bool {
        guard let restored = entry.restoredCaches else { return true }
        // Defense in depth: the admission guard checked the layer COUNT, but
        // each layer's attention semantics are defined by its concrete cache
        // CLASS (sliding RotatingKVCache vs full KVCacheSimple). A
        // same-count-but-wrong-per-position-type checkpoint (a provider/
        // serializer bug) would otherwise rebuild the wrong batched class and
        // silently run wrong attention. Verify each restored layer's class
        // matches the model's own layer class at that position; any mismatch
        // aborts the restore and falls back to a clean cold prefill.
        let expected = model.newCache(parameters: nil)
        guard expected.count == restored.count else {
            admitColdFallback(entry)
            return false
        }
        // Inverse of extract, per layer, requiring the restored class to
        // match the model's class at that position. RotatingKVCache is NOT a
        // KVCacheSimple subclass; ChunkedKVCache IS, and is unsupported, so
        // it's rejected on either side.
        var batched: [any BatchedCache] = []
        batched.reserveCapacity(restored.count)
        for (layer, exp) in zip(restored, expected) {
            if layer is ChunkedKVCache || exp is ChunkedKVCache {
                admitColdFallback(entry)  // unsupported subclass
                return false
            } else if let rot = layer as? RotatingKVCache, exp is RotatingKVCache {
                batched.append(BatchRotatingKVCache.fromSingleRow(rot))
            } else if let simple = layer as? KVCacheSimple, exp is KVCacheSimple {
                batched.append(BatchKVCache.merge([simple]))
            } else {
                // Type mismatch (e.g. Rotating where the model wants Simple),
                // recurrent, or unknown — restore is unsound; cold-prefill.
                admitColdFallback(entry)
                return false
            }
        }

        // The suffix to process: everything after the restored prefix. The
        // last token is the decode seed; the rest is prefilled on top of the
        // restored cache. `generate(lastTokensOf:)` runs prompt(dropLast) then
        // seeds with last — exactly the cold transition, but warm-seeded.
        // suffix is non-empty (guard: 1 <= tokenCount < promptLen).
        let suffix = entry.tokensToPrefill
        let ppBatch = PromptProcessingBatch(
            model: model,
            uids: [entry.uid],
            promptCache: batched,
            tokens: [Array(entry.promptTokens.prefix(entry.promptTokens.count - suffix.count))],
            maxTokens: [entry.request.maxTokens],
            prefillStepSize: config.prefillStepSize,
            samplers: [entry.sampler],
            fallbackSampler: greedySampler,
            stateMachines: [entry.machine]
        )
        let gen = ppBatch.generate(lastTokensOf: [suffix], mtpRuntime: mtpRuntime)
        mergeIntoGenBatch(gen)
        return true
    }

    /// Fallback when a restored cache can't be rebuilt (an unsupported layer
    /// type slipped past the provider's capability gate). Fully unwinds the
    /// admission state this entry already took (mirroring `preemptOne`),
    /// clears `restoredCheckpoint`, and re-queues at the front so the next
    /// admit cycle processes it as a normal COLD full-prompt prefill. We do
    /// NOT start a parallel prefill here — that could clobber a cold batch
    /// being set up in the same `admitWaiting` cycle. Correctness over reuse.
    private func admitColdFallback(_ entry: AdmittedEntry) {
        let rid = entry.request.requestId
        // Undo the admission bookkeeping done earlier in admitWaiting.
        activeRids.removeAll { $0 == rid }
        activeSamplers.removeValue(forKey: rid)
        activeDetokenizers.removeValue(forKey: rid)
        activeStreamStates.removeValue(forKey: rid)
        tokenHistories.removeValue(forKey: rid)
        if let uid = ridToUid[rid] { uidToRid.removeValue(forKey: uid) }
        ridToUid.removeValue(forKey: rid)
        totalPromptTokens -= entry.request.numPromptTokens

        // Reset so the retry is a clean cold prefill of the full prompt.
        entry.request.restoredCheckpoint = nil
        entry.request.cachedTokens = 0
        entry.request.status = RequestStatus.waiting
        waiting.insert(entry.request, at: 0)
    }

    private func makeStateMachine(for request: Request) -> SequenceStateMachine {
        var seqs: [(sequence: [Int], next: String?)] = []
        for id in eosTokenIds { seqs.append((sequence: [id], next: nil)) }
        for id in request.samplingParams.stopTokenIds where !eosTokenIds.contains(id) {
            seqs.append((sequence: [id], next: nil))
        }
        for s in request.samplingParams.stop where !s.isEmpty {
            let toks = tokenizer.encode(text: s)
            if !toks.isEmpty { seqs.append((sequence: toks, next: nil)) }
        }
        guard !seqs.isEmpty else { return SequenceStateMachine() }
        return SequenceStateMachine(states: ["normal": seqs])
    }

    private lazy var cacheFactories: [(Int) -> any BatchedCache] = {
        let quantConfig = config.kvQuantization
        return model.newCache(parameters: nil).map { layer -> (Int) -> any BatchedCache in
            Self.makeCacheFactory(for: layer, quantConfig: quantConfig)
        }
    }()

    private func makeBatchedCache(batchSize B: Int) -> [any BatchedCache] {
        cacheFactories.map { $0(B) }
    }

    /// Internal decision of which batched cache type to build for a layer.
    /// Kept in one place so the production factory and the test-facing
    /// type-name helper cannot drift.
    private enum CacheFactoryKind: Sendable {
        case mamba
        case arrays(size: Int)
        case rotating(maxSize: Int)
        case quantized(config: KVQuantizationConfig)
        case fp16

        var typeName: String {
            switch self {
            case .mamba: return String(describing: MambaCache.self)
            case .arrays: return String(describing: ArraysCache.self)
            case .rotating: return String(describing: BatchRotatingKVCache.self)
            case .quantized(let config):
                switch config.cacheKind {
                case .kernel: return String(describing: QuantizedBatchKVCache.self)
                case .dequant: return String(describing: DequantBatchKVCache.self)
                }
            case .fp16: return String(describing: BatchKVCache.self)
            }
        }
    }

    private static func cacheFactoryKind(
        for layer: any KVCache,
        quantConfig: KVQuantizationConfig?
    ) -> CacheFactoryKind {
        if layer is MambaCache {
            return .mamba
        }
        if let arrays = layer as? ArraysCache {
            return .arrays(size: arrays.slotCount)
        }
        if let rotating = layer as? RotatingKVCache, let maxSize = rotating.maxSize {
            return .rotating(maxSize: maxSize)
        }
        if let quantConfig {
            return .quantized(config: quantConfig)
        }
        return .fp16
    }

    /// Build a single cache factory for a layer, optionally using quantized
    /// storage for full-attention layers. Exposed publicly so provider tests
    /// can verify per-layer factory selection without constructing a live model.
    public static func makeCacheFactory(
        for layer: any KVCache,
        quantConfig: KVQuantizationConfig?
    ) -> (Int) -> any BatchedCache {
        switch cacheFactoryKind(for: layer, quantConfig: quantConfig) {
        case .mamba:
            return { MambaCache(leftPadding: Array(repeating: 0, count: $0)) }
        case .arrays(let size):
            return { ArraysCache(size: size, leftPadding: Array(repeating: 0, count: $0)) }
        case .rotating(let maxSize):
            return { BatchRotatingKVCache(maxSize: maxSize, leftPadding: Array(repeating: 0, count: $0)) }
        case .quantized(let config):
            return {
                switch config.cacheKind {
                case .kernel:
                    QuantizedBatchKVCache(
                        leftPadding: Array(repeating: 0, count: $0),
                        groupSize: config.groupSize,
                        bits: config.bits,
                        mode: config.mode
                    )
                case .dequant:
                    DequantBatchKVCache(
                        leftPadding: Array(repeating: 0, count: $0),
                        groupSize: config.groupSize,
                        bits: config.bits,
                        mode: config.mode
                    )
                }
            }
        case .fp16:
            return { BatchKVCache(leftPadding: Array(repeating: 0, count: $0)) }
        }
    }

    /// Test-facing type-name helper. Avoids instantiating caches (which would
    /// require a Metal device) while still pinning the exact per-layer selection.
    public static func cacheFactoryTypeName(
        for layer: any KVCache,
        quantConfig: KVQuantizationConfig?
    ) -> String {
        cacheFactoryKind(for: layer, quantConfig: quantConfig).typeName
    }

    private func doExternalPrefill(
        tokens: [Int],
        existingCache: [KVCache]?
    ) -> ([KVCacheSimple], [Int]) {
        let n = tokens.count
        if n <= 1 {
            let cache: [KVCache] = existingCache ?? model.newCache(parameters: nil)
            let simples = cache.compactMap { $0 as? KVCacheSimple }
            precondition(
                simples.count == cache.count,
                "doExternalPrefill: all cache layers must be KVCacheSimple; "
                    + "hybrid models (Mamba, Jamba) are not supported in the warm prefix path"
            )
            return (simples, tokens)
        }

        let cache: [KVCache] = existingCache ?? model.newCache(parameters: nil)
        let prefillTokens = tokens[0..<(n - 1)]
        let lastTokens = [tokens[n - 1]]

        let prefillStep = config.prefillStepSize
        var processed = 0

        while processed < prefillTokens.count {
            let chunkEnd = min(processed + prefillStep, prefillTokens.count)
            let chunk = Array(prefillTokens[processed..<chunkEnd])
            let input = LMInput.Text(tokens: MLXArray(chunk).reshaped(1, -1))
            _ = model(input, cache: cache, state: nil)
            processed = chunkEnd
        }

        let simples = cache.compactMap { $0 as? KVCacheSimple }
        precondition(
            simples.count == cache.count,
            "doExternalPrefill: all cache layers must be KVCacheSimple; "
                + "hybrid models (Mamba, Jamba) are not supported in the warm prefix path"
        )
        return (simples, lastTokens)
    }
}

extension Scheduler {

    private func processGenResponses(
        _ responses: [GenerationBatchResponse]
    ) -> [RequestOutput] {
        var outputs: [RequestOutput] = []

        for resp in responses {
            let rid = uidToRid[resp.uid] ?? ""
            guard let request = requests[rid] else { continue }
            let tokenId = resp.token
            let isFinished = resp.finishReason != nil

            let newText: String
            if !isFinished {
                request.appendOutputToken(tokenId)
                tokenHistories[rid]?.tokens.append(tokenId)
                var detok = activeDetokenizers[rid]!
                detok.append(token: tokenId)
                newText = detok.next() ?? ""
                activeDetokenizers[rid] = detok
            } else if resp.finishReason == "length" {
                request.appendOutputToken(tokenId)
                tokenHistories[rid]?.tokens.append(tokenId)
                newText = ""
            } else {
                newText = ""
            }

            var output: RequestOutput

            if isFinished {
                output = RequestOutput(
                    requestId: rid,
                    newTokenIds: [],
                    newText: newText,
                    outputTokenIds: request.outputTokenIds,
                    outputText: tokenizer.decode(tokenIds: request.outputTokenIds),
                    finished: true,
                    finishReason: resp.finishReason,
                    promptTokens: request.numPromptTokens,
                    completionTokens: request.numOutputTokens,
                    cachedTokens: request.cachedTokens
                )
                request.outputText = output.outputText
                if resp.finishReason == "stop" {
                    request.setFinished(.finishedStopped)
                } else {
                    request.setFinished(.finishedLengthCapped)
                }
                totalCompletionTokens += request.numOutputTokens
                numRequestsProcessed += 1

                if let pc = prefixCache,
                   let promptTokens = request.promptTokenIds,
                   let rawCache = resp.promptCache,
                   rawCache.allSatisfy({ $0 is KVCacheSimple })
                {
                    let simpleCaches = rawCache.map { $0 as! KVCacheSimple }
                    let allTokens = promptTokens + request.outputTokenIds
                    pc.storePrefix(requestId: rid, tokens: allTokens, layerCaches: simpleCaches)
                }
                prefixCache?.releaseRequest(rid)
            } else {
                var streamState = activeStreamStates[rid]!
                let shouldSend = streamState.shouldSend(
                    totalTokens: request.numOutputTokens,
                    finished: false
                )
                if !shouldSend { continue }

                output = RequestOutput(
                    requestId: rid,
                    newTokenIds: [tokenId],
                    newText: newText,
                    outputTokenIds: request.outputTokenIds,
                    outputText: request.outputText,
                    finished: false,
                    promptTokens: request.numPromptTokens,
                    completionTokens: request.numOutputTokens,
                    cachedTokens: request.cachedTokens
                )
                streamState.markSent(totalTokens: request.numOutputTokens)
                activeStreamStates[rid] = streamState
            }

            outputs.append(output)
        }

        return outputs
    }

    private func cleanupFinished(_ finishedIds: Set<String>) {
        for rid in finishedIds {
            finishedReqIds.insert(rid)
        }
    }
}

extension Scheduler {

    public func reset() {
        pendingAbortIds.removeAll()
        for rid in activeRids {
            requests[rid]?.setFinished(.finishedAborted)
        }
        waiting.removeAll()
        activeRids.removeAll()
        activeSamplers.removeAll()
        activeDetokenizers.removeAll()
        activeStreamStates.removeAll()
        tokenHistories.removeAll()
        requests.removeAll()
        finishedReqIds.removeAll()
        genBatch = nil
        pendingPrefill = nil
        uidCounter = 0
        uidToRid.removeAll()
        ridToUid.removeAll()
    }

    public func deepReset() { reset() }

    public func getStats() -> [String: Any] {
        [
            "num_waiting": waiting.count,
            "num_running": activeRids.count,
            "num_requests_processed": numRequestsProcessed,
            "total_prompt_tokens": totalPromptTokens,
            "total_completion_tokens": totalCompletionTokens,
            "current_kv_tokens": currentKVTokens,
            "max_num_seqs": runtimeMaxNumSeqs,
        ]
    }
}
