// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/scheduler.py — Continuous batching scheduler.
// https://github.com/jundot/omlx/blob/main/omlx/scheduler.py
//
// Request lifecycle:
//   waiting → prefill (external, per-request KVCacheSimple) →
//   merging into shared BatchKVCache → batched decode via GenerationBatch
//   → finished extraction.
//
// The model's existing callAsFunction auto-batches with BatchKVCache:
// one [B, 1] forward pass replaces per-request sequential decode.

import Foundation
import MLX

// MARK: - Repetition-penalty helpers

/// Mutable token-history shared between a request and its penalty sampler.
final class TokenHistoryHolder: @unchecked Sendable {
    var tokens: [Int]
    init(tokens: [Int]) { self.tokens = tokens }
}

/// Wrap `base` with multiplicative repetition penalty and additive
/// presence/frequency penalties applied to previously-seen tokens.
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

        // Materialize once, modify in-place, then reconstruct.
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

// MARK: - SchedulerConfig

public struct SchedulerConfig: Sendable {
    public var maxNumSeqs: Int
    public var maxNumBatchedTokens: Int
    public var prefillStepSize: Int
    public var streamInterval: Int

    public init(
        maxNumSeqs: Int = 64,
        maxNumBatchedTokens: Int = 8192,
        prefillStepSize: Int = 2048,
        streamInterval: Int = 1
    ) {
        self.maxNumSeqs = maxNumSeqs
        self.maxNumBatchedTokens = maxNumBatchedTokens
        self.prefillStepSize = prefillStepSize
        self.streamInterval = streamInterval
    }
}

// MARK: - SchedulerOutput

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

// MARK: - Scheduler

public final class Scheduler: @unchecked Sendable {
    public let config: SchedulerConfig
    public let model: any LanguageModel
    private let tokenizer: any Tokenizer
    private let eosTokenIds: Set<Int>

    // Queues
    private var waiting: [Request] = []
    private var requests: [String: Request] = [:]
    public private(set) var finishedReqIds: Set<String> = []

    // Active request tracking
    private var activeRids: [String] = []
    private var activeSamplers: [String: @Sendable (MLXArray) -> MLXArray] = [:]
    private var activeDetokenizers: [String: NaiveStreamingDetokenizer] = [:]
    private var activeStreamStates: [String: RequestStreamState] = [:]
    private var tokenHistories: [String: TokenHistoryHolder] = [:]

    // UID management for GenerationBatch
    private var uidCounter: Int = 0
    private var uidToRid: [Int: String] = [:]
    private var ridToUid: [String: Int] = [:]

    // Active generation batch (nil when no requests are running)
    // Its promptCache holds the shared per-layer batched caches.
    private var genBatch: GenerationBatch?

    // Aborts
    private var pendingAbortIds: Set<String> = []

    // Optional prefix cache (in-memory block-level KV reuse).
    public var prefixCache: PrefixCache?

    // Stats
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
        self.eosTokenIds = eosTokenIds
        self.prefixCache = prefixCache
    }

    // MARK: - Request Lifecycle

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

    public func removeFinishedRequest(_ requestId: String) {
        requests.removeValue(forKey: requestId)
        activeSamplers.removeValue(forKey: requestId)
        activeDetokenizers.removeValue(forKey: requestId)
        activeStreamStates.removeValue(forKey: requestId)
        tokenHistories.removeValue(forKey: requestId)
        activeRids.removeAll { $0 == requestId }
        if let uid = ridToUid[requestId] {
            uidToRid.removeValue(forKey: uid)
        }
        ridToUid.removeValue(forKey: requestId)
    }

    public func hasRequests() -> Bool {
        !waiting.isEmpty || (genBatch != nil && !genBatch!.isEmpty)
    }

    public func getNumWaiting() -> Int { waiting.count }
    public func getNumRunning() -> Int { genBatch?.batchSize ?? 0 }

    // MARK: - Step

    public func step() -> SchedulerOutput {
        var output = SchedulerOutput()

        // 1. Pending aborts
        processPendingAborts()

        // 2. Prefill waiting requests and merge into active batch
        let newScheduled = scheduleAndPrefill()
        for r in newScheduled {
            output.scheduledRequestIds.append(r.requestId)
            output.numScheduledTokens += r.numPromptTokens
        }
        output.hasWork = !newScheduled.isEmpty

        // 3. Batched decode via GenerationBatch
        if let batch = genBatch, !batch.isEmpty {
            let responses = batch.next()
            output.hasWork = true

            // 4. Convert GenerationBatchResponse → RequestOutput
            let stepOutputs = processGenResponses(responses)
            output.outputs = stepOutputs

            // 5. Track finished
            for r in responses where r.finishReason != nil {
                let rid = uidToRid[r.uid] ?? ""
                output.finishedRequestIds.insert(rid)
            }
            cleanupFinished(output.finishedRequestIds)
        }

        return output
    }

    // MARK: - Aborts

    private func processPendingAborts() {
        for rid in pendingAbortIds {
            doAbortRequest(rid)
        }
        pendingAbortIds.removeAll()
    }

    private func doAbortRequest(_ requestId: String) {
        guard let request = requests[requestId] else { return }

        if request.status == .waiting {
            waiting.removeAll { $0.requestId == requestId }
        }

        if let uid = ridToUid[requestId] {
            // Remove from GenerationBatch
            if let batch = genBatch {
                let keep = (0..<batch.uids.count).filter { batch.uids[$0] != uid }
                batch.filter(keep: keep)
                activeRids.removeAll { $0 == requestId }
            }
            removeFinishedRequest(requestId)
        }

        prefixCache?.releaseRequest(requestId)
        request.setFinished(.finishedAborted)
        finishedReqIds.insert(requestId)
        requests.removeValue(forKey: requestId)
    }

    // MARK: - Schedule + Prefill

    /// Prefill waiting requests and merge into active batch.
    ///
    /// Cold requests (no prefix-cache hit) are prefilled as a batch via
    /// `PromptProcessingBatch`; warm requests (prefix-cache hit) are prefilled
    /// sequentially and then merged. Returns the newly scheduled requests.
    private func scheduleAndPrefill() -> [Request] {
        guard !waiting.isEmpty else { return [] }
        let availableSlots = max(0, config.maxNumSeqs - activeRids.count)
        guard availableSlots > 0 else { return [] }

        struct AdmittedEntry {
            let request: Request
            let promptTokens: [Int]
            let tokensToPrefill: [Int]
            let existingCache: [KVCache]?   // nil = cold, non-nil = warm (prefix hit)
            let uid: Int
            let sampler: RowSampler?
            let machine: SequenceStateMachine
        }

        var admitted: [AdmittedEntry] = []
        var newScheduled: [Request] = []

        for _ in 0 ..< min(availableSlots, waiting.count) {
            let request = waiting.removeFirst()

            // Tokenize
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
            guard let promptTokens = request.promptTokenIds, !promptTokens.isEmpty else { continue }

            // Prefix cache lookup: skip re-prefilling already-cached blocks.
            var tokensToPrefill = promptTokens
            var existingCache: [KVCache]? = nil
            if let pc = prefixCache {
                let (cached, remaining) = pc.fetchPrefix(
                    requestId: request.requestId, tokens: promptTokens)
                if let cached {
                    tokensToPrefill = remaining
                    existingCache = cached.map { $0 as KVCache }
                    request.cachedTokens = promptTokens.count - remaining.count
                }
            }

            let uid = uidCounter
            uidCounter += 1
            uidToRid[uid] = request.requestId
            ridToUid[request.requestId] = uid

            request.status = .running
            activeRids.append(request.requestId)

            // Build sampler, wrapping with repetition penalty if needed.
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
            let sampler: RowSampler? = needsPenalty
                ? makeRepetitionSampler(
                    base: baseSampler, history: history,
                    repetitionPenalty: params.repetitionPenalty,
                    presencePenalty: params.presencePenalty,
                    frequencyPenalty: params.frequencyPenalty)
                : baseSampler

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
                uid: uid,
                sampler: sampler,
                machine: makeStateMachine(for: request)
            ))
        }

        guard !admitted.isEmpty else { return newScheduled }

        // Split into cold (no prefix hit) and warm (prefix hit) groups.
        let cold = admitted.filter { $0.existingCache == nil }
        let warm = admitted.filter { $0.existingCache != nil }

        var newGenBatches: [GenerationBatch] = []

        // Cold: batched prefill via PromptProcessingBatch.
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
            let coldGen = ppBatch.generate(lastTokensOf: cold.map { $0.tokensToPrefill })
            if !coldGen.isEmpty { newGenBatches.append(coldGen) }
        }

        // Warm: sequential prefill for prefix-cache hits.
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
                stateMachines: warmMachines
            )
            if !warmGen.isEmpty { newGenBatches.append(warmGen) }
        }

        guard !newGenBatches.isEmpty else { return newScheduled }

        // Merge all new sub-batches, then extend (or initialize) genBatch.
        let first = newGenBatches[0]
        for other in newGenBatches.dropFirst() { first.extend(other) }

        if let existing = genBatch {
            existing.extend(first)
            genBatch = existing
        } else {
            genBatch = first
        }

        return newScheduled
    }

    // MARK: - scheduleAndPrefill helpers

    /// Build a `SequenceStateMachine` that terminates on EOS tokens and
    /// any per-request stop strings / stop-token IDs.
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

    /// Per-layer factory closures, initialised once from a model probe.
    /// Avoids re-allocating a full set of KV caches on every cold batch.
    private lazy var cacheFactories: [(Int) -> any BatchedCache] = {
        model.newCache(parameters: nil).map { layer -> (Int) -> any BatchedCache in
            if layer is MambaCache {
                return { MambaCache(leftPadding: Array(repeating: 0, count: $0)) }
            }
            if let arrays = layer as? ArraysCache {
                let size = arrays.slotCount
                return { ArraysCache(size: size, leftPadding: Array(repeating: 0, count: $0)) }
            }
            if let rotating = layer as? RotatingKVCache, let maxSize = rotating.maxSize {
                return { BatchRotatingKVCache(maxSize: maxSize, leftPadding: Array(repeating: 0, count: $0)) }
            }
            return { BatchKVCache(leftPadding: Array(repeating: 0, count: $0)) }
        }
    }()

    /// Allocate one batched cache per layer, matching `BatchGenerator.makeBatchedCache`.
    private func makeBatchedCache(batchSize B: Int) -> [any BatchedCache] {
        cacheFactories.map { $0(B) }
    }

    // MARK: - External Prefill

    /// Run prompt tokens through the model with per-request KVCacheSimple caches.
    /// Returns (per-layer KVCacheSimple array, leftover tokens list).
    /// `remaining` tokens includes the last token(s) used as decode seed.
    private func doExternalPrefill(
        tokens: [Int],
        existingCache: [KVCache]?
    ) -> ([KVCacheSimple], [Int]) {
        let n = tokens.count
        if n <= 1 {
            let cache: [KVCache] = existingCache ?? model.newCache(parameters: nil)
            let simples = cache.map { $0 as! KVCacheSimple }
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

        let simples = cache.map { $0 as! KVCacheSimple }
        return (simples, lastTokens)
    }
}

// MARK: - Response Processing

extension Scheduler {

    /// Convert GenerationBatch responses to RequestOutputs.
    /// Handles streaming detokenization, finish detection, and stream intervals.
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
                // omlx: append the final token for length-capped generation
                // (scheduler.py: `elif not is_stop: request.append_output_token(response.token)`).
                // For stop/EOS finish the stop token is intentionally excluded.
                request.appendOutputToken(tokenId)
                tokenHistories[rid]?.tokens.append(tokenId)
                newText = ""
            } else {
                newText = ""
            }

            // Handle finish
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

                // Store the completed KV state in the prefix cache for future reuse.
                // Only works for pure-attention models where all cache layers are KVCacheSimple.
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
                // Stream interval check
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

// MARK: - Reset & Stats

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
        ]
    }
}