import Foundation
import MLX
import MLXFast
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Gemma 4 Block-SDCM Prefill & Paged Last-Query", .serialized)
struct Gemma4BlockSDCMTests {

    // MARK: - Config Parsing

    @Test("Config resolve parses presets and overrides cleanly")
    func configResolve() {
        #expect(Gemma4BlockSDCMConfig.resolve(environment: [:]).mode == .off)
        #expect(
            Gemma4BlockSDCMConfig.resolve(environment: [Gemma4BlockSDCMConfig.envKey: "0"]).mode
                == .off)

        let lc = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "lightcone",
            Gemma4BlockSDCMConfig.minTokensEnvKey: "2048",
            Gemma4BlockSDCMConfig.earlyMoEKEnvKey: "6",
        ])
        #expect(lc.mode == .lightcone)
        #expect(lc.minPromptTokens == 2048)
        #expect(lc.earlyMoEK == 6)
        #expect(lc.keepSchedule[12] == 0.40)

        let bal = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "balanced"
        ])
        #expect(bal.mode == .balanced)
        #expect(bal.earlyMoEK == 4)
        #expect(bal.lateMoEK == 4)

        let fast = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "fast"
        ])
        #expect(fast.mode == .fast)
        #expect(fast.keepSchedule[6] == 0.45)
        #expect(fast.earlyMoEK == 8)
    }

    // MARK: - Arbitrary-Position RoPE Exact Parity

    @Test("applyArbitraryPositionsRoPE matches standard RoPE bit-identically on sliding and full ProportionalRoPE")
    func arbitraryPositionRoPEParity() {
        MLXRandom.seed(42)
        let L = 64
        let H = 4
        let slidingRope = initializeRope(
            dims: 64, base: 10_000, traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: 4096)
        let fullRope = initializeRope(
            dims: 128, base: 1_000_000, traditional: false,
            scalingConfig: ["type": .string("proportional"), "partial_rotary_factor": .float(0.25)],
            maxPositionEmbeddings: 4096)

        // 1. Contiguous [0 ..< L] parity
        let posContig = MLXArray((0 ..< L).map { Int32($0) })
        let xSlide = MLXRandom.normal([1, H, L, 64]).asType(.float32)
        let refSlide = slidingRope(xSlide, offset: 0)
        let arbSlide = Gemma4BlockSDCM.applyArbitraryPositionsRoPE(
            slidingRope, to: xSlide, positions: posContig)
        eval(refSlide, arbSlide)
        let diffSlide = MLX.abs(refSlide - arbSlide).max().item(Float.self)
        #expect(diffSlide < 1e-5, "Sliding RoPE contiguous mismatch: \(diffSlide)")

        let xFull = MLXRandom.normal([1, H, L, 128]).asType(.float32)
        let refFull = fullRope(xFull, offset: 0)
        let arbFull = Gemma4BlockSDCM.applyArbitraryPositionsRoPE(
            fullRope, to: xFull, positions: posContig)
        eval(refFull, arbFull)
        let diffFull = MLX.abs(refFull - arbFull).max().item(Float.self)
        #expect(diffFull < 1e-5, "ProportionalRoPE contiguous mismatch: \(diffFull)")

        // 2. Non-contiguous positions parity vs per-token scalar RoPE
        let sparsePosHost: [Int32] = [0, 5, 19, 127, 513, 1029, 2047]
        let xSparse = MLXRandom.normal([1, H, sparsePosHost.count, 128]).asType(.float32)
        let arbSparse = Gemma4BlockSDCM.applyArbitraryPositionsRoPE(
            fullRope, to: xSparse, positions: MLXArray(sparsePosHost))
        let refParts = sparsePosHost.enumerated().map { idx, p in
            fullRope(xSparse[0..., 0..., idx ..< (idx + 1), 0...], offset: Int(p))
        }
        let refSparse = concatenated(refParts, axis: 2)
        eval(arbSparse, refSparse)
        let diffSparse = MLX.abs(arbSparse - refSparse).max().item(Float.self)
        #expect(diffSparse < 1e-5, "ProportionalRoPE non-contiguous mismatch: \(diffSparse)")
    }

    // MARK: - Block-Atomic Selection Invariants

    @Test("selectCompactedIndices preserves prefix, tail, atomic blocks, symmetric halos, and macro strata")
    func blockAtomicSelectionInvariants() {
        let T = 1024  // 16 blocks of 64
        let positions = (0 ..< T).map { Int32($0) }
        var scores = [Float](repeating: 0.01, count: T)
        // Place a single hot peak inside block 6 (positions 384 ..< 448).
        scores[400] = 10.0

        var cfg = Gemma4BlockSDCMConfig.lightcone
        cfg.blockSize = 64
        cfg.prefixKeep = 64  // block 0
        cfg.tailKeep = 64    // block 15
        cfg.strataCount = 4
        cfg.forwardHalo = 0.85
        cfg.backwardHalo = 0.85

        let kept = Gemma4BlockSDCM.selectCompactedIndices(
            scores: scores, positions: positions, originalLength: T,
            targetFraction: 0.50, config: cfg)

        // Strictly ascending and bounded
        for i in 1 ..< kept.count {
            #expect(kept[i] > kept[i - 1])
        }
        // Prefix block 0 and tail block 15 (including frontier T-1) always kept
        #expect(kept.first == 0)
        #expect(kept.last == Int32(T - 1))

        // Atomic blocks: every kept block has ALL 64 tokens present
        var blockCounts: [Int: Int] = [:]
        for idx in kept {
            blockCounts[Int(idx) / 64, default: 0] += 1
        }
        for (_, count) in blockCounts {
            #expect(count == 64)
        }
        // Hot block 6 AND both symmetric halo neighbors (block 5 and block 7) must be retained!
        #expect(blockCounts[5] == 64)
        #expect(blockCounts[6] == 64)
        #expect(blockCounts[7] == 64)
        // All 4 macro-strata (blocks 0..3, 4..7, 8..11, 12..15) represented
        for stratum in 0 ..< 4 {
            let hasBlock = (stratum * 4 ..< (stratum + 1) * 4).contains { blockCounts[$0] == 64 }
            #expect(hasBlock, "Missing macro stratum \(stratum)")
        }
    }

    // MARK: - Chunked Banded Original-Position Sliding Attention Parity

    @Test("slidingAttentionWithPositions chunked O(L*W) matches dense original-position mask")
    func slidingAttentionParity() {
        MLXRandom.seed(7)
        // Simulate compacted non-contiguous positions spanning beyond chunkSize=64 and window=96.
        var posHost: [Int32] = []
        for b in [0, 1, 3, 5, 8, 9, 12] {
            for j in 0 ..< 32 {
                posHost.append(Int32(b * 32 + j))
            }
        }
        let L = posHost.count  // 224 tokens
        let window = 96
        let posMx = MLXArray(posHost)
        let q = MLXRandom.normal([1, 4, L, 32]).asType(.float32)
        let k = MLXRandom.normal([1, 2, L, 32]).asType(.float32)
        let v = MLXRandom.normal([1, 2, L, 32]).asType(.float32)

        let chunked = Gemma4BlockSDCM.slidingAttentionWithPositions(
            queries: q, keys: k, values: v, positions: posMx, positionsHost: posHost,
            slidingWindow: window, scale: 1.0 / sqrt(32.0), chunkSize: 64)
        let dense = Gemma4BlockSDCM.slidingAttentionWithPositions(
            queries: q, keys: k, values: v, positions: posMx, positionsHost: posHost,
            slidingWindow: window, scale: 1.0 / sqrt(32.0), chunkSize: 1024)
        eval(chunked, dense)
        let maxDiff = MLX.abs(chunked - dense).max().item(Float.self)
        #expect(maxDiff < 1e-5, "Chunked vs dense original-position sliding SDPA diff: \(maxDiff)")
    }

    // MARK: - PagedLayerCache Last-Query Prefill Exact Parity

    @Test("PagedLayerCache updateAndAttendLastQuery matches full-chunk updateAndAttend on frontier row")
    func pagedLastQueryPrefillParity() throws {
        MLXRandom.seed(123)
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let poolConfig = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 8 * 1024 * 1024, dtype: .float32, maxPrefillChunk: 128)

        let backendFull = try PagedKVBackend(layerKinds: [kind], config: poolConfig)
        let backendLast = try PagedKVBackend(layerKinds: [kind], config: poolConfig)
        let stateFull = try backendFull.makeSequenceState(
            layerKinds: [kind], promptLength: 64, maxLength: 256)
        let stateLast = try backendLast.makeSequenceState(
            layerKinds: [kind], promptLength: 64, maxLength: 256)
        defer {
            backendFull.release(stateFull)
            backendLast.release(stateLast)
        }
        let rowFull = stateFull[0] as! PagedSequenceKV
        let rowLast = stateLast[0] as! PagedSequenceKV

        let cacheFull = backendFull.makeLayerCaches()[0]
        let cacheLast = backendLast.makeLayerCaches()[0]
        cacheFull.setRows([rowFull])
        cacheLast.setRows([rowLast])
        #expect(cacheLast.supportsLastQueryPrefill)

        let L = 40
        let q = MLXRandom.normal([1, 4, L, 64]).asType(.float32)
        let k = MLXRandom.normal([1, 2, L, 64]).asType(.float32)
        let v = MLXRandom.normal([1, 2, L, 64]).asType(.float32)

        let outFull = cacheFull.updateAndAttend(
            queries: q, keys: k, values: v, scale: 1.0, sinks: nil)
        let outLast = cacheLast.updateAndAttendLastQuery(
            queries: q[0..., 0..., (L - 1)..., 0...], keys: k, values: v, scale: 1.0, sinks: nil)
        eval(outFull, outLast, cacheFull.positionOffsets, cacheLast.positionOffsets)

        let diff = MLX.abs(outFull[0..., 0..., (L - 1)..., 0...] - outLast).max().item(Float.self)
        #expect(diff < 1e-4, "Paged last-query vs full attention frontier mismatch: \(diff)")
        #expect(rowLast.absoluteOffset == L)
        #expect(cacheLast.hostPositionOffsets == [Int32(L)])

        // Verify subsequent decode step also matches identically!
        let qDec = MLXRandom.normal([1, 4, 1, 64]).asType(.float32)
        let kDec = MLXRandom.normal([1, 2, 1, 64]).asType(.float32)
        let vDec = MLXRandom.normal([1, 2, 1, 64]).asType(.float32)
        let decFull = cacheFull.updateAndAttend(
            queries: qDec, keys: kDec, values: vDec, scale: 1.0, sinks: nil)
        let decLast = cacheLast.updateAndAttend(
            queries: qDec, keys: kDec, values: vDec, scale: 1.0, sinks: nil)
        eval(decFull, decLast)
        let decDiff = MLX.abs(decFull - decLast).max().item(Float.self)
        #expect(decDiff < 1e-4, "Post-last-query decode mismatch: \(decDiff)")
    }

    // MARK: - End-to-End Synthetic Gemma 4 MoE Block-SDCM (Paged, Contiguous, Standalone)

    private static func makeSyntheticGemma4MoE() throws -> Gemma4TextModel {
        // 12-layer 5:1 Gemma 4 MoE fixture: full attention at layers 5 and 11, sliding elsewhere.
        let layerTypes = (0 ..< 12).map {
            ($0 + 1) % 6 == 0 ? "\"full_attention\"" : "\"sliding_attention\""
        }.joined(separator: ",")
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 12,
                "intermediate_size": 96,
                "num_attention_heads": 4,
                "head_dim": 64,
                "global_head_dim": 128,
                "vocab_size": 256,
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 2,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 48,
                "attention_k_eq_v": true,
                "layer_types": [\(layerTypes)],
                "tie_word_embeddings": true,
                "enable_moe_block": true,
                "num_experts": 8,
                "top_k_experts": 4,
                "moe_intermediate_size": 32,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                    "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
                }
            }
            """
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
        return Gemma4TextModel(config)
    }

    @Test("End-to-end Block-SDCM prefills & decodes cleanly on Paged, Contiguous, and Standalone KV caches")
    func endToEndBlockSDCMAllBackends() throws {
        MLXRandom.seed(99)
        let model = try Self.makeSyntheticGemma4MoE()
        let sdcm = Gemma4BlockSDCMConfig(
            mode: .fast,
            minPromptTokens: 96,
            keepSchedule: [6: 0.50],
            earlyMoEK: 2,
            lateMoEK: 4,
            earlyMoEUntilLayer: 6,
            blockSize: 16,
            prefixKeep: 16,
            tailKeep: 16,
            forwardHalo: 0.85,
            backwardHalo: 0.50,
            strataCount: 2,
            twoHopBlockPeaks: 4,
            slidingChunkSize: 32
        )
        model.blockSDCMConfig = sdcm

        let promptLen = 128
        let prompt = MLXArray((0 ..< promptLen).map { Int32($0 % 200) }).expandedDimensions(axis: 0)
        let nextTok = MLXArray([Int32(42)]).reshaped([1, 1])

        // 1. Paged CBv2 Backend
        do {
            let kinds = model.cbv2LayerKinds
            let backend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 64 * 1024 * 1024,
                    dtype: .float32, maxPrefillChunk: 128))
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: promptLen, maxLength: 256)
            defer { backend.release(state) }
            let rows = state.map { $0 as! PagedSequenceKV }
            let pagedCaches = backend.makeLayerCaches()
            let caches: [KVCache] = pagedCaches.enumerated().map { idx, c in
                c.setRows([rows[idx]])
                return c
            }

            let prefillLogits = model.cbv2Prefill(
                prompt, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
            eval(prefillLogits)
            #expect(prefillLogits.shape == [1, model.vocabularySize])
            #expect(!MLX.isNaN(prefillLogits).any().item(Bool.self))

            // Layer 5 (pre-compaction full) stored all 128 tokens; Layer 11 (post-compaction full)
            // compacted to < 128 physical tokens while keeping logical ropePositionOffset == 128!
            #expect(rows[5].absoluteOffset == promptLen)
            #expect(rows[5].ropePositionOffset == promptLen)
            #expect(!rows[5].isCompacted)

            #expect(rows[11].absoluteOffset < promptLen)
            #expect(rows[11].ropePositionOffset == promptLen)
            #expect(rows[11].isCompacted)
            #expect(!rows[11].supportsSpeculativeWrites)

            // Sliding layer 0 stored only trailing `slidingWindow = 48` tokens, with logical RoPE == 128!
            #expect(rows[0].absoluteOffset == 48)
            #expect(rows[0].ropePositionOffset == promptLen)

            // Subsequent decode step must advance all layers' logical RoPE offset to 129 cleanly!
            let decLogits = model(nextTok, cache: caches)
            eval(decLogits)
            #expect(decLogits.shape == [1, 1, model.vocabularySize])
            #expect(!MLX.isNaN(decLogits).any().item(Bool.self))
            #expect(rows[11].ropePositionOffset == promptLen + 1)
            #expect((caches[11] as! PagedLayerCache).hostPositionOffsets == [Int32(promptLen + 1)])
        }

        // 2. Contiguous CBv2 Backend
        do {
            let kinds = model.cbv2LayerKinds
            let rows: [CBv2SequenceKV] = kinds.map { kind in
                switch kind.attention {
                case .full:
                    CBv2FullSequenceKV(
                        promptLength: promptLen, maxLength: 256,
                        kvHeads: kind.kvHeads, headDim: kind.headDim)
                case .slidingWindow(let w):
                    CBv2WindowedSequenceKV(
                        window: w, kvHeads: kind.kvHeads, headDim: kind.headDim)
                }
            }
            let caches: [KVCache] = kinds.enumerated().map { idx, kind in
                let c = CBv2LayerCache(layerIndex: idx, kind: kind)
                c.setRows([rows[idx]])
                return c
            }

            let prefillLogits = model.cbv2Prefill(
                prompt, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
            eval(prefillLogits)
            #expect(prefillLogits.shape == [1, model.vocabularySize])
            #expect(rows[11].absoluteOffset < promptLen)
            #expect(rows[11].ropePositionOffset == promptLen)
            #expect(rows[11].isCompacted)
            #expect(!rows[11].supportsSpeculativeWrites)

            let decLogits = model(nextTok, cache: caches)
            eval(decLogits)
            #expect(!MLX.isNaN(decLogits).any().item(Bool.self))
            #expect(rows[11].ropePositionOffset == promptLen + 1)
        }

        // 3. Standalone KVCache (`prepare` + decode) & Short-Prompt Exact Bypass
        do {
            let caches = model.newCache(parameters: nil)
            let input = LMInput(tokens: prompt.squeezed(axis: 0))
            let prep = try model.prepare(input, cache: caches, state: nil, prefill: PrefillParameters())
            guard case .tokens(let remaining) = prep else {
                Issue.record("Expected .tokens from prepare")
                return
            }
            #expect(remaining.tokens.size == 1)
            if case .scalar(let ropePos) = caches[11].ropeOffset {
                #expect(ropePos == promptLen - 1)
            } else {
                Issue.record("Expected scalar ropeOffset")
            }
            let decOut = model(remaining.tokens.expandedDimensions(axis: 0), cache: caches)
            eval(decOut)
            #expect(!MLX.isNaN(decOut).any().item(Bool.self))

            // Short prompt (< minPromptTokens=96) must bypass Block-SDCM bit-identically!
            let shortPrompt = prompt[0..., 0 ..< 32]
            let offModel = try Self.makeSyntheticGemma4MoE()
            try offModel.update(parameters: model.parameters(), verify: .all)
            offModel.blockSDCMConfig = .off
            let lgOn = model.cbv2Prefill(
                shortPrompt, inputEmbedding: nil, cache: nil, requirement: .lastPositionLogits)
            let lgOff = offModel.cbv2Prefill(
                shortPrompt, inputEmbedding: nil, cache: nil, requirement: .lastPositionLogits)
            eval(lgOn, lgOff)
            #expect(MLX.abs(lgOn - lgOff).max().item(Float.self) < 1e-5)
        }
    }

    // MARK: - Lazy Multi-Turn Checkpoint Wake-Up

    @Test("Gemma4SDCMCheckpoint wakes up newly attended blocks while excluding already-kept positions")
    func lazyCheckpointWakeUp() {
        let T = 256  // 4 blocks of 64
        let positions = (0 ..< T).map { Int32($0) }
        let ckpt = Gemma4SDCMCheckpoint(
            hiddenStates: MLXArray.zeros([1, T, 16], dtype: .float32),
            positions: positions,
            checkpointLayer: 11,
            originalLength: T)

        // Construct keys where block 2 (positions 128..<192) strongly aligns with queryQ!
        var kData = [Float](repeating: 0.0, count: 1 * 2 * T * 8)
        for h in 0 ..< 2 {
            for p in 140 ..< 148 {
                let base = (h * T + p) * 8
                kData[base] = 10.0
            }
        }
        var qData = [Float](repeating: 0.0, count: 1 * 4 * 4 * 8)
        for h in 0 ..< 4 {
            for q in 0 ..< 4 {
                qData[(h * 4 + q) * 8] = 1.0
            }
        }
        let k = MLXArray(kData).reshaped([1, 2, T, 8])
        let q = MLXArray(qData).reshaped([1, 4, 4, 8])

        // Suppose Turn 1 only kept block 0 (0..<64) and block 3 (192..<256).
        let alreadyKept = Set((0 ..< 64).map { Int32($0) } + (192 ..< 256).map { Int32($0) })
        let wake = ckpt.selectWakeUpIndices(
            queryQ: q, keyK: k, scale: 1.0, nHeads: 4, nKvHeads: 2,
            alreadyKeptPositions: alreadyKept, wakeFraction: 0.25, blockSize: 64)

        // Must wake block 2 (128..<192) and must NOT duplicate any already-kept position!
        #expect(wake.contains(140))
        #expect(wake.allSatisfy { !alreadyKept.contains($0) })
    }

    // MARK: - Scheduler Solo Whole-Prompt Compaction

    @Test("SchedulerV2 arms solo whole-prompt prefill for cold text requests when soloCompactionMinPromptTokens is set")
    func schedulerSoloCompactionArmsOnlyForColdSoloText() throws {
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 4,
            maxBatchedTokensPerStep: 4096,
            prefillChunkSize: 2048,
            soloPrefillStripeTokens: nil)
        config.soloCompactionMinPromptTokens = 1536
        let scheduler = SchedulerV2(config: config)

        let longPrompt = Array(repeating: 1, count: 6000)
        let req = CBv2Request(id: CBv2RequestID(1), promptTokens: longPrompt, maxTokens: 16)
        try scheduler.enqueue(req)
        let plan = scheduler.plan()
        // Solo cold text request >= 1536 receives the entire 6000-token prompt in one step!
        #expect(plan.assignments.count == 1)
        #expect(plan.assignments[0].id == req.id)
        #expect(plan.assignments[0].numTokens == 6000)
    }

    @Test("SchedulerV2 does not split solo compaction prompts at demanded checkpoint boundaries")
    func schedulerSoloCompactionIgnoresDemandedCheckpoints() throws {
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 4,
            maxBatchedTokensPerStep: 4096,
            prefillChunkSize: 2048,
            soloPrefillStripeTokens: 4096,
            enablePrefixCache: true)
        config.soloCompactionMinPromptTokens = 1536
        config.demandedShortCheckpointMinimumTokens = 1024
        config.demandedCheckpointPartitionIncludesLongPrompts = true
        config.demandedLatestCheckpointMinimumTokens = 1024

        // Prompt wider than prefillChunkSize with a demanded fork hint: must dispatch whole.
        do {
            let scheduler = SchedulerV2(config: config)
            var req = CBv2Request(
                id: CBv2RequestID(10), promptTokens: Array(repeating: 1, count: 6000),
                maxTokens: 16, cacheSalt: "salt-a")
            req.prefixCheckpointTargetTokens = 2048
            try scheduler.enqueue(req)
            let plan = scheduler.plan()
            #expect(plan.assignments.count == 1)
            #expect(plan.assignments[0].numTokens == 6000)
            #expect(plan.demandedShortCheckpointRows.isEmpty)
        }

        // Prompt between minCompaction (1536) and prefillChunkSize (2048): also must not split.
        do {
            let scheduler = SchedulerV2(config: config)
            var req = CBv2Request(
                id: CBv2RequestID(11), promptTokens: Array(repeating: 1, count: 1800),
                maxTokens: 16, cacheSalt: "salt-b")
            req.prefixCheckpointTargetTokens = 1280
            try scheduler.enqueue(req)
            let plan = scheduler.plan()
            #expect(plan.assignments.count == 1)
            #expect(plan.assignments[0].numTokens == 1800)
            #expect(plan.demandedShortCheckpointRows.isEmpty)
        }
    }

    // MARK: - Edge Cases & Prefix-Veto Regressions

    @Test("Block-SDCM math handles L=1, L<=prefix+tail, gaps > window, empty wake-up, and KVCacheSimple.copy")
    func blockSDCMEdgeCases() {
        let cfg = Gemma4BlockSDCMConfig.lightcone

        // L == 1 boundary scoring & selection
        let q1 = MLXRandom.normal([1, 4, 1, 16]).asType(.float32)
        let k1 = MLXRandom.normal([1, 2, 1, 16]).asType(.float32)
        let v1 = MLXRandom.normal([1, 2, 1, 16]).asType(.float32)
        let pos1Host: [Int32] = [0]
        let pos1 = MLXArray(pos1Host)
        let scores1 = Gemma4BlockSDCM.computeBoundaryScores(
            queries: q1, keys: k1, positions: pos1, positionsHost: pos1Host,
            originalLength: 1, scale: 0.25, nHeads: 4, nKvHeads: 2, config: cfg)
        #expect(scores1.count == 1)
        #expect(!scores1[0].isNaN)
        let kept1 = Gemma4BlockSDCM.selectCompactedIndices(
            scores: scores1, positions: pos1Host, originalLength: 1,
            targetFraction: 0.2, config: cfg)
        #expect(kept1 == [0])

        // L <= prefixKeep + tailKeep retains every index
        let smallLen = cfg.prefixKeep + cfg.tailKeep
        let smallPos = (0 ..< smallLen).map { Int32($0) }
        let keptSmall = Gemma4BlockSDCM.selectCompactedIndices(
            scores: [Float](repeating: 0.1, count: smallLen),
            positions: smallPos, originalLength: smallLen,
            targetFraction: 0.1, config: cfg)
        #expect(keptSmall.count == smallLen)

        // Sliding attention binary search with gap > window and chunkSize == 0 safety
        let gapPosHost: [Int32] = [0, 1, 2, 500, 501, 502]
        let gapPos = MLXArray(gapPosHost)
        let qGap = MLXRandom.normal([1, 4, gapPosHost.count, 16]).asType(.float32)
        let kGap = MLXRandom.normal([1, 2, gapPosHost.count, 16]).asType(.float32)
        let vGap = MLXRandom.normal([1, 2, gapPosHost.count, 16]).asType(.float32)
        let chunkedGap = Gemma4BlockSDCM.slidingAttentionWithPositions(
            queries: qGap, keys: kGap, values: vGap, positions: gapPos,
            positionsHost: gapPosHost, slidingWindow: 16, scale: 0.25, chunkSize: 2)
        let denseGap = Gemma4BlockSDCM.slidingAttentionWithPositions(
            queries: qGap, keys: kGap, values: vGap, positions: gapPos,
            positionsHost: gapPosHost, slidingWindow: 16, scale: 0.25, chunkSize: 1024)
        let zeroChunkGap = Gemma4BlockSDCM.slidingAttentionWithPositions(
            queries: qGap, keys: kGap, values: vGap, positions: gapPos,
            positionsHost: gapPosHost, slidingWindow: 16, scale: 0.25, chunkSize: 0)
        eval(chunkedGap, denseGap, zeroChunkGap)
        #expect(MLX.abs(chunkedGap - denseGap).max().item(Float.self) < 1e-5)
        #expect(MLX.abs(zeroChunkGap - denseGap).max().item(Float.self) < 1e-5)

        // Empty checkpoint wake-up guards
        let emptyCkpt = Gemma4SDCMCheckpoint(
            hiddenStates: MLXArray.zeros([1, 0, 8], dtype: .float32),
            positions: [], checkpointLayer: 11, originalLength: 0)
        #expect(
            emptyCkpt.selectWakeUpIndices(
                queryQ: q1, keyK: k1, scale: 1.0, nHeads: 4, nKvHeads: 2,
                alreadyKeptPositions: []).isEmpty)

        // KVCacheSimple.copy preserves positionDelta and ropeOffset
        let simple = KVCacheSimple()
        _ = simple.update(keys: k1, values: v1)
        simple.positionDelta = 77
        let copied = simple.copy() as! KVCacheSimple
        #expect(copied.positionDelta == 77)
        if case .scalar(let ropePos) = copied.ropeOffset {
            #expect(ropePos == 78)
        } else {
            Issue.record("Expected scalar ropeOffset on copied KVCacheSimple")
        }
    }

    @Test("PagedLayerCache writeCompactedPromptKV handles stored=0, chunked sliding writes, and resident prefix veto")
    func pagedWriteCompactedEdgeCasesAndPrefixVeto() throws {
        MLXRandom.seed(314)
        let fullKind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let slidingKind = CBv2LayerKind(
            attention: .slidingWindow(64), headDim: 64, kvHeads: 2, queryHeads: 4)
        // Deliberately configure maxPrefillChunk (32) < slidingWindow (64) to exercise chunked writes.
        let poolConfig = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 16 * 1024 * 1024, dtype: .float32, maxPrefillChunk: 32)
        let backend = try PagedKVBackend(layerKinds: [fullKind, slidingKind], config: poolConfig)
        let state = try backend.makeSequenceState(
            layerKinds: [fullKind, slidingKind], promptLength: 128, maxLength: 256)
        defer { backend.release(state) }

        let fullRow = state[0] as! PagedSequenceKV
        let slidingRow = state[1] as! PagedSequenceKV
        let caches = backend.makeLayerCaches()
        caches[0].setRows([fullRow])
        caches[1].setRows([slidingRow])

        // 1. stored == 0 with logicalDelta > 0 advances logical offsets without writing pages
        let emptyK = MLXArray.zeros([1, 2, 0, 64], dtype: .float32)
        let emptyV = MLXArray.zeros([1, 2, 0, 64], dtype: .float32)
        caches[0].writeCompactedPromptKV(keys: emptyK, values: emptyV, logicalDelta: 16)
        #expect(fullRow.absoluteOffset == 0)
        #expect(fullRow.ropePositionOffset == 16)
        #expect(fullRow.isCompacted)
        #expect(caches[0].hostPositionOffsets == [16])

        // 2. Write 32 physical tokens + 16 more logicalDelta -> fullRow is compacted and MUST veto
        // resident prefix-page handles even for page-aligned [0..<16]!
        let k32 = MLXRandom.normal([1, 2, 32, 64]).asType(.float32)
        let v32 = MLXRandom.normal([1, 2, 32, 64]).asType(.float32)
        caches[0].writeCompactedPromptKV(keys: k32, values: v32, logicalDelta: 16)
        #expect(fullRow.absoluteOffset == 32)
        #expect(fullRow.ropePositionOffset == 64)
        #expect(fullRow.isCompacted)
        #expect(!fullRow.supportsSpeculativeWrites)
        #expect(fullRow.prefixPageHandles(tokens: 0 ..< 16) == nil)

        // 3. Sliding layer write of 64 tokens (> maxPrefillChunk=32) chunks safely without trapping
        let k64 = MLXRandom.normal([1, 2, 64, 64]).asType(.float32)
        let v64 = MLXRandom.normal([1, 2, 64, 64]).asType(.float32)
        caches[1].writeCompactedPromptKV(keys: k64, values: v64, logicalDelta: 64)
        #expect(slidingRow.absoluteOffset == 64)
        #expect(slidingRow.ropePositionOffset == 128)
        #expect(caches[1].hostPositionOffsets == [128])
        #expect(!backend.pool.writeValidation.isFaulted)

        // Subsequent decode on the sliding layer works cleanly
        let qDec = MLXRandom.normal([1, 4, 1, 64]).asType(.float32)
        let kDec = MLXRandom.normal([1, 2, 1, 64]).asType(.float32)
        let vDec = MLXRandom.normal([1, 2, 1, 64]).asType(.float32)
        let decOut = caches[1].updateAndAttend(
            queries: qDec, keys: kDec, values: vDec, scale: 1.0, sinks: nil)
        eval(decOut)
        #expect(!MLX.isNaN(decOut).any().item(Bool.self))
        #expect(slidingRow.ropePositionOffset == 129)
    }

    @Test("PagedLayerCache updateAndAttendLastQuery matches updateAndAttend for B>1 with history and gates sliding off")
    func pagedLastQueryPrefillBatchedAndHistoryParity() throws {
        MLXRandom.seed(271)
        let fullKind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let slidingKind = CBv2LayerKind(
            attention: .slidingWindow(32), headDim: 64, kvHeads: 2, queryHeads: 4)
        let poolConfig = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 16 * 1024 * 1024, dtype: .float32, maxPrefillChunk: 128)

        let backendFull = try PagedKVBackend(layerKinds: [fullKind, slidingKind], config: poolConfig)
        let backendLast = try PagedKVBackend(layerKinds: [fullKind, slidingKind], config: poolConfig)
        let sFull0 = try backendFull.makeSequenceState(
            layerKinds: [fullKind, slidingKind], promptLength: 64, maxLength: 256)
        let sFull1 = try backendFull.makeSequenceState(
            layerKinds: [fullKind, slidingKind], promptLength: 64, maxLength: 256)
        let sLast0 = try backendLast.makeSequenceState(
            layerKinds: [fullKind, slidingKind], promptLength: 64, maxLength: 256)
        let sLast1 = try backendLast.makeSequenceState(
            layerKinds: [fullKind, slidingKind], promptLength: 64, maxLength: 256)
        defer {
            backendFull.release(sFull0)
            backendFull.release(sFull1)
            backendLast.release(sLast0)
            backendLast.release(sLast1)
        }

        let cachesFull = backendFull.makeLayerCaches()
        let cachesLast = backendLast.makeLayerCaches()
        cachesFull[0].setRows([sFull0[0] as! PagedSequenceKV, sFull1[0] as! PagedSequenceKV])
        cachesLast[0].setRows([sLast0[0] as! PagedSequenceKV, sLast1[0] as! PagedSequenceKV])
        cachesLast[1].setRows([sLast0[1] as! PagedSequenceKV, sLast1[1] as! PagedSequenceKV])

        #expect(cachesLast[0].supportsLastQueryPrefill)
        #expect(!cachesLast[1].supportsLastQueryPrefill)

        // Seed history chunk (offset > 0) identically on both backends
        let qHist = MLXRandom.normal([2, 4, 19, 64]).asType(.float32)
        let kHist = MLXRandom.normal([2, 2, 19, 64]).asType(.float32)
        let vHist = MLXRandom.normal([2, 2, 19, 64]).asType(.float32)
        _ = cachesFull[0].updateAndAttend(queries: qHist, keys: kHist, values: vHist, scale: 0.25, sinks: nil)
        _ = cachesLast[0].updateAndAttend(queries: qHist, keys: kHist, values: vHist, scale: 0.25, sinks: nil)

        // Second chunk: full vs last-query for B=2 over existing history
        let L = 23
        let q2 = MLXRandom.normal([2, 4, L, 64]).asType(.float32)
        let k2 = MLXRandom.normal([2, 2, L, 64]).asType(.float32)
        let v2 = MLXRandom.normal([2, 2, L, 64]).asType(.float32)
        let outFull = cachesFull[0].updateAndAttend(
            queries: q2, keys: k2, values: v2, scale: 0.25, sinks: nil)
        let outLast = cachesLast[0].updateAndAttendLastQuery(
            queries: q2[0..., 0..., (L - 1)..., 0...], keys: k2, values: v2, scale: 0.25, sinks: nil)
        eval(outFull, outLast)
        let diff = MLX.abs(outFull[0..., 0..., (L - 1)..., 0...] - outLast).max().item(Float.self)
        #expect(diff < 1e-4, "Batched + history paged last-query mismatch: \(diff)")
        #expect(cachesLast[0].hostPositionOffsets == [Int32(19 + L), Int32(19 + L)])
    }

    @Test("Standalone multi-turn prompt and decode after Block-SDCM remain finite across compacted sliding layers")
    func standaloneMultiTurnAfterBlockSDCM() throws {
        MLXRandom.seed(55)
        let model = try Self.makeSyntheticGemma4MoE()
        model.blockSDCMConfig = Gemma4BlockSDCMConfig(
            mode: .fast,
            minPromptTokens: 96,
            keepSchedule: [6: 0.25],
            earlyMoEK: 2,
            lateMoEK: 4,
            earlyMoEUntilLayer: 6,
            blockSize: 8,
            prefixKeep: 8,
            tailKeep: 8,
            forwardHalo: 0.5,
            backwardHalo: 0.5,
            strataCount: 1,
            twoHopBlockPeaks: 2,
            slidingChunkSize: 32
        )

        let promptLen = 128
        let prompt = MLXArray((0 ..< promptLen).map { Int32($0 % 200) })
        let caches = model.newCache(parameters: nil)
        let prep = try model.prepare(
            LMInput(tokens: prompt), cache: caches, state: nil, prefill: PrefillParameters())
        guard case .tokens(let remaining) = prep else {
            Issue.record("Expected .tokens from prepare")
            return
        }
        // Finish turn 1 frontier token
        let t1Logits = model(remaining.tokens.expandedDimensions(axis: 0), cache: caches)
        eval(t1Logits)
        #expect(!MLX.isNaN(t1Logits).any().item(Bool.self))

        // Turn 2 multi-token prompt (n > 1) when layer 0 has 48 live tokens and compacted layer 6
        // has fewer live tokens (< slidingWindow=48) — must produce finite logits without mask mismatch.
        let followUp = MLXArray([Int32(11), 12, 13, 14, 15, 16]).reshaped([1, 6])
        let t2Logits = model(followUp, cache: caches)
        eval(t2Logits)
        #expect(t2Logits.shape == [1, 6, model.vocabularySize])
        #expect(!MLX.isNaN(t2Logits).any().item(Bool.self))
    }
}
