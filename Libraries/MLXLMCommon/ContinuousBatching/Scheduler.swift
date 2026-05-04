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

    // UID management for GenerationBatch
    private var uidCounter: Int = 0
    private var uidToRid: [Int: String] = [:]
    private var ridToUid: [String: Int] = [:]

    // Active generation batch (nil when no requests are running)
    // Its promptCache holds the shared per-layer batched caches.
    private var genBatch: GenerationBatch?

    // Aborts
    private var pendingAbortIds: Set<String> = []

    // Stats
    public private(set) var numRequestsProcessed: Int = 0
    public private(set) var totalPromptTokens: Int = 0
    public private(set) var totalCompletionTokens: Int = 0

    public init(
        model: any LanguageModel,
        tokenizer: any Tokenizer,
        config: SchedulerConfig = SchedulerConfig(),
        eosTokenIds: Set<Int> = []
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.eosTokenIds = eosTokenIds
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

        request.setFinished(.finishedAborted)
        finishedReqIds.insert(requestId)
        requests.removeValue(forKey: requestId)
    }

    // MARK: - Schedule + Prefill

    /// Prefill waiting requests one-by-one and merge into shared batched caches.
    /// Returns the newly scheduled requests.
    private func scheduleAndPrefill() -> [Request] {
        guard !waiting.isEmpty else { return [] }
        let slotCount = activeRids.count
        let availableSlots = max(0, config.maxNumSeqs - slotCount)
        guard availableSlots > 0 else { return [] }

        var newScheduled: [Request] = []
        var newUids: [Int] = []
        var newSeedTokens: [Int] = []
        var newMaxTokens: [Int] = []
        var newSamplers: [RowSampler?] = []
        var newMachines: [SequenceStateMachine] = []
        var newTokenLists: [[Int]] = []

        // Batch of per-row KVCacheSimple arrays (one per new request, each [KVCacheSimple] per layer)
        var newRowCaches: [[KVCacheSimple]] = []

        for _ in 0..<min(availableSlots, waiting.count) {
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

            // External prefill
            let (simpleCaches, seedTokens) = doExternalPrefill(
                tokens: promptTokens,
                existingCache: request.promptCache
            )

            let uid = uidCounter
            uidCounter += 1
            uidToRid[uid] = request.requestId
            ridToUid[request.requestId] = uid

            request.status = .running
            activeRids.append(request.requestId)

            let sampler = makeRowSampler(
                temperature: request.samplingParams.temperature,
                topP: request.samplingParams.topP,
                minP: request.samplingParams.minP,
                topK: request.samplingParams.topK
            )
            activeSamplers[request.requestId] = sampler
            activeDetokenizers[request.requestId] = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            activeStreamStates[request.requestId] = RequestStreamState(
                streamInterval: config.streamInterval
            )

            totalPromptTokens += request.numPromptTokens
            newScheduled.append(request)
            newUids.append(uid)
            newSeedTokens.append(seedTokens.last ?? promptTokens.last ?? 0)
            newMaxTokens.append(request.maxTokens)
            newSamplers.append(sampler)
            newMachines.append(SequenceStateMachine())
            newTokenLists.append(promptTokens)  // includes all tokens including last
            newRowCaches.append(simpleCaches)
        }

        guard !newUids.isEmpty else { return newScheduled }

        // Build seed tokens MLXArray
        let seedArr = MLXArray(newSeedTokens)

        // Build per-layer BatchKVCache for just the new rows
        let numLayers = newRowCaches[0].count
        let newPerLayer: [any BatchedCache] = (0..<numLayers).map { layer in
            let layerCaches = newRowCaches.map { $0[layer] }
            return BatchKVCache.merge(layerCaches)
        }

        if let existing = genBatch {
            // Extend existing generation batch: creates a temp batch with new
            // rows and extends the existing batch's caches via extendBatched.
            let tempBatch = GenerationBatch(
                model: model,
                uids: newUids,
                seedTokens: seedArr,
                promptCache: newPerLayer,
                tokens: newTokenLists,
                maxTokens: newMaxTokens,
                samplers: newSamplers,
                fallbackSampler: greedySampler,
                stateMachines: newMachines
            )
            existing.extend(tempBatch)
            genBatch = existing
        } else {
            // Create initial GenerationBatch with the merged per-layer caches
            genBatch = GenerationBatch(
                model: model,
                uids: newUids,
                seedTokens: seedArr,
                promptCache: newPerLayer,
                tokens: newTokenLists,
                maxTokens: newMaxTokens,
                samplers: newSamplers,
                fallbackSampler: greedySampler,
                stateMachines: newMachines
            )
        }

        return newScheduled
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
                var detok = activeDetokenizers[rid]!
                detok.append(token: tokenId)
                newText = detok.next() ?? ""
                activeDetokenizers[rid] = detok
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