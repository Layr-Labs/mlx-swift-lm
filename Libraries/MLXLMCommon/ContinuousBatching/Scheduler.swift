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

    public init(
        maxNumSeqs: Int = 64,
        maxNumBatchedTokens: Int = 8192,
        prefillStepSize: Int = 512,
        streamInterval: Int = 1,
        maxKVCacheTokens: Int = 0
    ) {
        self.maxNumSeqs = maxNumSeqs
        self.maxNumBatchedTokens = maxNumBatchedTokens
        self.prefillStepSize = prefillStepSize
        self.streamInterval = streamInterval
        self.maxKVCacheTokens = maxKVCacheTokens
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
        self.hasWork = false
    }
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
    public let model: any LanguageModel
    private let tokenizer: any Tokenizer
    private let eosTokenIds: Set<Int>

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
            // All sequences prefilled — transition to decode.
            let gen = pp.ppBatch.generate(lastTokensOf: pp.seeds.map { [$0] })
            mergeIntoGenBatch(gen)
            pendingPrefill = nil
            return
        }

        // Budget: how many prefill tokens we're allowed this step.
        let prefillBudget = max(1, config.maxNumBatchedTokens - decodeBatchSize)
        // Chunk size: never exceed the per-step budget or the remaining work.
        let chunkSize = min(config.prefillStepSize, prefillBudget, maxRemaining)
        let chunks = pp.remaining.map { Array($0.prefix(chunkSize)) }

        pp.ppBatch.prompt(chunks)

        pp.remaining = zip(pp.remaining, chunks).map { rem, chunk in
            Array(rem.dropFirst(chunk.count))
        }

        if pp.remaining.allSatisfy({ $0.isEmpty }) {
            let gen = pp.ppBatch.generate(lastTokensOf: pp.seeds.map { [$0] })
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
        if let uid = ridToUid[rid], let batch = genBatch {
            let keep = (0..<batch.uids.count).filter { batch.uids[$0] != uid }
            batch.filter(keep: keep)
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
        let availableSlots = max(0, config.maxNumSeqs - activeRids.count)
        guard availableSlots > 0 else { return [] }

        struct AdmittedEntry {
            let request: Request
            let promptTokens: [Int]
            let tokensToPrefill: [Int]
            let existingCache: [KVCache]?
            let uid: Int
            let sampler: RowSampler?
            let machine: SequenceStateMachine
        }

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
            if let pc = prefixCache {
                let (cached, remaining) = pc.fetchPrefix(
                    requestId: request.requestId, tokens: promptTokens)
                if let cached {
                    tokensToPrefill = remaining
                    existingCache = cached.map { $0 as KVCache }
                    request.cachedTokens = promptTokens.count - remaining.count
                }
            }

            let isCold = existingCache == nil
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

        let cold = admitted.filter { $0.existingCache == nil }
        let warm = admitted.filter { $0.existingCache != nil }

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
                stateMachines: warmMachines
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

    private func makeBatchedCache(batchSize B: Int) -> [any BatchedCache] {
        cacheFactories.map { $0(B) }
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
        ]
    }
}
