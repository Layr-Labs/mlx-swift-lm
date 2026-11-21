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
        #expect(lc.interiorMoEK(forLayer: 2) == 6)
        #expect(lc.keepSchedule[12] == 0.40)

        let lcDefault = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "lightcone"
        ])
        #expect(lcDefault.interiorMoEK(forLayer: 2) == 2)
        #expect(lcDefault.interiorMoEK(forLayer: 6) == 5)
        #expect(lcDefault.interiorMoEK(forLayer: 12) == 8)
        #expect(lcDefault.earlySparseFullLayer == 5)

        let bal = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "balanced"
        ])
        #expect(bal.mode == .balanced)
        #expect(bal.keepSchedule[6] == 0.46)
        #expect(bal.blockSize == 64)
        #expect(bal.interiorMoEK(forLayer: 2) == 2)
        #expect(bal.interiorMoEK(forLayer: 6) == 6)
        #expect(bal.lateMoEK == 8)
        #expect(bal.reversePrefixQuota[6] == 0.30)

        let fast = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "fast"
        ])
        #expect(fast.mode == .fast)
        #expect(fast.keepSchedule[6] == 0.36)
        #expect(fast.blockSize == 32)
        #expect(fast.interiorMoEK(forLayer: 2) == 2)
        #expect(fast.interiorMoEK(forLayer: 6) == 6)
        #expect(fast.reversePrefixQuota[6] == 0.32)
        #expect(fast.earlyPreCompactAfterLayer == nil)

        for alias in ["turbo", "v3", "l1", "ultra"] {
            let turbo = Gemma4BlockSDCMConfig.resolve(environment: [
                Gemma4BlockSDCMConfig.envKey: alias
            ])
            #expect(turbo.mode == .turbo)
            #expect(turbo.earlyPreCompactAfterLayer == 0)
            #expect(turbo.earlyPreCompactRatio == 0.55)
            #expect(turbo.earlyPreCompactBlockSize == 64)
            #expect(turbo.earlyPreCompactTopHaloK == 8)
            #expect(turbo.blockSize == 32)
            #expect(turbo.keepSchedule[6] == 0.36)
            #expect(turbo.reversePrefixQuota[24] == 0.20)
        }
    }

    // MARK: - Arbitrary-Position RoPE Exact Parity

    @Test(
        "applyArbitraryPositionsRoPE matches standard RoPE bit-identically on sliding and full ProportionalRoPE"
    )
    func arbitraryPositionRoPEParity() {
        MLXRandom.seed(42)
        let L = 64
        let H = 4
        let slidingRope = initializeRope(
            dims: 64, base: 10_000, traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: 4096)
        let fullRope = initializeRope(
            dims: 128, base: 1_000_000, traditional: false,
            scalingConfig: [
                "type": .string("proportional"), "partial_rotary_factor": .float(0.25),
            ],
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

    @Test(
        "selectCompactedIndices preserves prefix, tail, atomic blocks, symmetric halos, and macro strata"
    )
    func blockAtomicSelectionInvariants() {
        let T = 1024  // 16 blocks of 64
        let positions = (0 ..< T).map { Int32($0) }
        var scores = [Float](repeating: 0.01, count: T)
        // Place a single hot peak inside block 6 (positions 384 ..< 448).
        scores[400] = 10.0

        var cfg = Gemma4BlockSDCMConfig.lightcone
        cfg.blockSize = 64
        cfg.prefixKeep = 64  // block 0
        cfg.tailKeep = 64  // block 15
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

    @Test(
        "PagedLayerCache updateAndAttendLastQuery matches full-chunk updateAndAttend on frontier row"
    )
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

    @Test(
        "End-to-end Block-SDCM prefills & decodes cleanly on Paged, Contiguous, and Standalone KV caches"
    )
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
            let prep = try model.prepare(
                input, cache: caches, state: nil, prefill: PrefillParameters())
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

    @Test(
        "Gemma4SDCMCheckpoint wakes up newly attended blocks while excluding already-kept positions"
    )
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

    @Test(
        "SchedulerV2 arms solo whole-prompt prefill for cold text requests when soloCompactionMinPromptTokens is set"
    )
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

    @Test(
        "Block-SDCM math handles L=1, L<=prefix+tail, gaps > window, empty wake-up, and KVCacheSimple.copy"
    )
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
                alreadyKeptPositions: []
            ).isEmpty)

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

    @Test(
        "PagedLayerCache writeCompactedPromptKV handles stored=0, chunked sliding writes, and resident prefix veto"
    )
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

    @Test(
        "PagedLayerCache updateAndAttendLastQuery matches updateAndAttend for B>1 with history and gates sliding off"
    )
    func pagedLastQueryPrefillBatchedAndHistoryParity() throws {
        MLXRandom.seed(271)
        let fullKind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let slidingKind = CBv2LayerKind(
            attention: .slidingWindow(32), headDim: 64, kvHeads: 2, queryHeads: 4)
        let poolConfig = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 16 * 1024 * 1024, dtype: .float32, maxPrefillChunk: 128)

        let backendFull = try PagedKVBackend(
            layerKinds: [fullKind, slidingKind], config: poolConfig)
        let backendLast = try PagedKVBackend(
            layerKinds: [fullKind, slidingKind], config: poolConfig)
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
        _ = cachesFull[0].updateAndAttend(
            queries: qHist, keys: kHist, values: vHist, scale: 0.25, sinks: nil)
        _ = cachesLast[0].updateAndAttend(
            queries: qHist, keys: kHist, values: vHist, scale: 0.25, sinks: nil)

        // Second chunk: full vs last-query for B=2 over existing history
        let L = 23
        let q2 = MLXRandom.normal([2, 4, L, 64]).asType(.float32)
        let k2 = MLXRandom.normal([2, 2, L, 64]).asType(.float32)
        let v2 = MLXRandom.normal([2, 2, L, 64]).asType(.float32)
        let outFull = cachesFull[0].updateAndAttend(
            queries: q2, keys: k2, values: v2, scale: 0.25, sinks: nil)
        let outLast = cachesLast[0].updateAndAttendLastQuery(
            queries: q2[0..., 0..., (L - 1)..., 0...], keys: k2, values: v2, scale: 0.25, sinks: nil
        )
        eval(outFull, outLast)
        let diff = MLX.abs(outFull[0..., 0..., (L - 1)..., 0...] - outLast).max().item(Float.self)
        #expect(diff < 1e-4, "Batched + history paged last-query mismatch: \(diff)")
        #expect(cachesLast[0].hostPositionOffsets == [Int32(19 + L), Int32(19 + L)])
    }

    @Test(
        "Standalone multi-turn prompt and decode after Block-SDCM remain finite across compacted sliding layers"
    )
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

    // MARK: - Block-SDCM v2 Invariants

    @Test(
        "trueSlidingWindowStart excludes out-of-window compacted keys and preserves uncompacted tail"
    )
    func trueSlidingWindowStartInvariants() {
        // Uncompacted prompt: keeps exact trailing `W` tokens.
        let contig = (0 ..< 2048).map { Int32($0) }
        #expect(
            Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: contig, originalLength: 2048, slidingWindow: 1024) == 1024)

        // Compacted prompt (originalLength = 9600, W = 1024): every kept key must satisfy
        // origPos >= 9600 - 1024 + 1 = 8577 so decode at pos 9600 stays within sliding window W.
        let compacted: [Int32] = [0, 100, 1200, 4500, 8576, 8577, 9000, 9599]
        let start = Gemma4BlockSDCM.trueSlidingWindowStart(
            positionsHost: compacted, originalLength: 9600, slidingWindow: 1024)
        #expect(start == 5)
        for p in compacted[start...] {
            #expect(Int(p) >= 9600 - 1024 + 1)
        }
        #expect(compacted.last == 9599)

        // Edge cases: empty positions, no tokens within window, and non-positive slidingWindow.
        #expect(
            Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: [], originalLength: 0, slidingWindow: 1024) == 0)
        let outOfWindow: [Int32] = [0, 100, 1200, 4500, 8576]
        #expect(
            Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: outOfWindow, originalLength: 9600, slidingWindow: 1024)
                == outOfWindow.count)
        #expect(
            Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: compacted, originalLength: 9600, slidingWindow: 1)
                == compacted.count)
        #expect(
            Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: contig, originalLength: 2048, slidingWindow: 0) == contig.count)
        #expect(
            Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: contig, originalLength: 2048, slidingWindow: -16) == contig.count)
        #expect(
            Gemma4BlockSDCM.trueSlidingWindowStart(
                positionsHost: compacted, originalLength: 9600, slidingWindow: 0)
                == compacted.count)
    }

    @Test("prefixLocalFullTailAttention matches exact explicit prefix+local+full-tail causal mask")
    func prefixLocalFullTailParity() {
        MLXRandom.seed(314)
        let scale: Float = 1.0 / sqrt(32.0)

        // Check multiple boundary geometries around `pos <= P + W` vs `pos > P + W`, including `P == 0`.
        let cases: [(L: Int, P: Int, W: Int, tail: Int)] = [
            (256, 32, 64, 32),
            (320, 64, 64, 32),  // hits exact pos == P + W (128) and pos > P + W (192)
            (320, 63, 64, 32),  // hits exact pos == P + W + 1 (128)
            (256, 0, 64, 32),  // P == 0 skips prefix concatenation
        ]
        for c in cases {
            let q = MLXRandom.normal([1, 4, c.L, 32]).asType(.float32)
            let k = MLXRandom.normal([1, 2, c.L, 32]).asType(.float32)
            let v = MLXRandom.normal([1, 2, c.L, 32]).asType(.float32)

            let sparse = Gemma4BlockSDCM.prefixLocalFullTailAttention(
                queries: q, keys: k, values: v,
                prefixTokens: c.P, localWindow: c.W, tailTokens: c.tail, scale: scale)

            let qPos = MLXArray((0 ..< c.L).map { Int32($0) }).expandedDimensions(axis: 1)
            let kPos = MLXArray((0 ..< c.L).map { Int32($0) }).expandedDimensions(axis: 0)
            let cut = Int32(c.L - c.tail)
            let refMask =
                (kPos .<= qPos)
                .&& ((kPos .< Int32(c.P)) .|| ((qPos - kPos) .< Int32(c.W)) .|| (qPos .>= cut))
            let ref = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .array(refMask))
            eval(sparse, ref)
            let diff = MLX.abs(sparse - ref).max().item(Float.self)
            #expect(diff < 1e-5, "Prefix+Local+FullTail mismatch for \(c): \(diff)")
        }

        // L == 0 returns empty zeros cleanly; non-positive localWindow falls back safely to causal.
        let emptyQ = MLXArray.zeros([1, 4, 0, 32], dtype: .float32)
        let emptyK = MLXArray.zeros([1, 2, 0, 32], dtype: .float32)
        let emptyOut = Gemma4BlockSDCM.prefixLocalFullTailAttention(
            queries: emptyQ, keys: emptyK, values: emptyK,
            prefixTokens: 32, localWindow: 64, tailTokens: 32, scale: scale)
        #expect(emptyOut.shape == [1, 4, 0, 32])

        let smallQ = MLXRandom.normal([1, 4, 16, 32]).asType(.float32)
        let smallK = MLXRandom.normal([1, 2, 16, 32]).asType(.float32)
        let fallbackZeroW = Gemma4BlockSDCM.prefixLocalFullTailAttention(
            queries: smallQ, keys: smallK, values: smallK,
            prefixTokens: -4, localWindow: 0, tailTokens: -4, scale: scale)
        let exactCausal = MLXFast.scaledDotProductAttention(
            queries: smallQ, keys: smallK, values: smallK, scale: scale, mask: .causal)
        eval(fallbackZeroW, exactCausal)
        #expect(MLX.abs(fallbackZeroW - exactCausal).max().item(Float.self) < 1e-5)
    }

    @Test(
        "computeBilateralBlockScores GQA reverse resonance matches per-head reference and handles edge cases"
    )
    func bilateralBlockScoresGQAAndEdgeCases() {
        MLXRandom.seed(777)
        let nHeads = 8
        let nKvHeads = 2
        let g = nHeads / nKvHeads
        let D = 16
        let scale: Float = 1.0 / sqrt(Float(D))
        let blockSize = 32

        var cfg = Gemma4BlockSDCMConfig.fast
        cfg.blockSize = blockSize
        cfg.prefixKeep = 64
        cfg.tailKeep = 64
        cfg.reversePrefixEnd = 64
        cfg.twoHopBlockPeaks = 4

        // 1. L == 0 returns zero forward scores and nil reverse scores.
        let zeroScores = Gemma4BlockSDCM.computeBilateralBlockScores(
            queries: MLXArray.zeros([1, nHeads, 0, D], dtype: .float32),
            keys: MLXArray.zeros([1, nKvHeads, 0, D], dtype: .float32),
            positions: MLXArray([Int32]()),
            positionsHost: [],
            originalLength: 128,
            scale: scale, nHeads: nHeads, nKvHeads: nKvHeads,
            reverseQuotaFraction: 0.3, config: cfg)
        #expect(zeroScores.forwardBlockScores == [0, 0, 0, 0])
        #expect(zeroScores.reverseBlockScores == nil)
        #expect(zeroScores.reverseQuotaFraction == 0)

        // 2. Non-contiguous compacted positions (with dead blocks!) + GQA reverse log-odds parity.
        // Original length 512 (16 blocks of 32), drop blocks 4 and 9 to simulate post-compaction gaps.
        let origLen = 512
        let nBlocks = origLen / blockSize
        var nonContigPos: [Int32] = []
        for b in 0 ..< nBlocks where b != 4 && b != 9 {
            for j in 0 ..< blockSize {
                nonContigPos.append(Int32(b * blockSize + j))
            }
        }
        let L = nonContigPos.count  // 448 > reversePrefixEnd(64) + 256
        let q = MLXRandom.normal([1, nHeads, L, D]).asType(.float32)
        let k = MLXRandom.normal([1, nKvHeads, L, D]).asType(.float32)

        let scores = Gemma4BlockSDCM.computeBilateralBlockScores(
            queries: q, keys: k, positions: MLXArray(nonContigPos),
            positionsHost: nonContigPos, originalLength: origLen,
            scale: scale, nHeads: nHeads, nKvHeads: nKvHeads,
            reverseQuotaFraction: 0.30, config: cfg)
        #expect(scores.forwardBlockScores.count == nBlocks)
        #expect(scores.forwardBlockScores.allSatisfy { $0.isFinite })
        guard let revActual = scores.reverseBlockScores else {
            Issue.record("Expected non-nil reverseBlockScores")
            return
        }
        #expect(revActual.count == nBlocks)
        #expect(revActual.allSatisfy { $0.isFinite })

        // Naive unflattened per-head GQA reference for scale-free reciprocal-rank reverse log-odds.
        let sinkEnd = 4
        let prefEnd = 64
        let tokenBlocks = nonContigPos.map { Int($0) / blockSize }
        var blockCounts = [Int](repeating: 0, count: nBlocks)
        for b in tokenBlocks { blockCounts[b] += 1 }
        let interiorBlocks = (2 ..< (nBlocks - 2)).filter { blockCounts[$0] > 0 }
        let interiorSet = Set(interiorBlocks)
        var rrMaxRef = [Float](repeating: 0, count: nBlocks)
        var rrSumRef = [Float](repeating: 0, count: nBlocks)
        for h in 0 ..< nHeads {
            let kv = h / g
            let qH = q[0, h, 0..., 0...]  // [L, D]
            let kSinkH = k[0, kv, ..<sinkEnd, 0...]  // [4, D]
            let kPrefH = k[0, kv, sinkEnd ..< prefEnd, 0...]  // [60, D]
            let dotP = matmul(qH, kPrefH.transposed(1, 0)) * scale
            let dotS = matmul(qH, kSinkH.transposed(1, 0)) * scale
            let resH = sigmoid(logSumExp(dotP, axis: -1) - logSumExp(dotS, axis: -1))
            eval(resH)
            let vals = resH.asArray(Float.self)
            var bh = [Float](repeating: 0, count: nBlocks)
            for i in 0 ..< L { bh[tokenBlocks[i]] += vals[i] }
            for b in interiorBlocks { bh[b] /= Float(blockCounts[b]) }
            // blockSize == 32 applies immediate alive-interior [0.25, 0.50, 0.25] smoothing
            var sm = [Float](repeating: 0, count: nBlocks)
            for b in interiorBlocks {
                var acc = Float(0.50) * bh[b]
                var w: Float = 0.50
                if interiorSet.contains(b - 1) {
                    acc += Float(0.25) * bh[b - 1]
                    w += 0.25
                }
                if interiorSet.contains(b + 1) {
                    acc += Float(0.25) * bh[b + 1]
                    w += 0.25
                }
                sm[b] = acc / w
            }
            let ranked = interiorBlocks.sorted { lhs, rhs in
                sm[lhs] != sm[rhs] ? sm[lhs] > sm[rhs] : lhs < rhs
            }
            for (rk, b) in ranked.enumerated() {
                let rr = Float(1.0) / Float(4 + rk)
                if rr > rrMaxRef[b] { rrMaxRef[b] = rr }
                rrSumRef[b] += rr
            }
        }
        var revRef = [Float](repeating: 0, count: nBlocks)
        for b in interiorBlocks { revRef[b] = rrMaxRef[b] + Float(0.15) * rrSumRef[b] }
        let maxRevDiff = zip(revActual, revRef).map { abs($0 - $1) }.max() ?? 0
        #expect(maxRevDiff < 1e-4, "GQA qFlat/kPref reverse resonance mismatch: \(maxRevDiff)")

        // 3. sinkEnd == 0 (all positions >= 4) and prefEnd <= sinkEnd safely disable reverse scores.
        let shiftedPos = (0 ..< L).map { Int32($0 + 10) }
        let noSinkScores = Gemma4BlockSDCM.computeBilateralBlockScores(
            queries: q, keys: k, positions: MLXArray(shiftedPos),
            positionsHost: shiftedPos, originalLength: L + 10,
            scale: scale, nHeads: nHeads, nKvHeads: nKvHeads,
            reverseQuotaFraction: 0.30, config: cfg)
        #expect(noSinkScores.reverseBlockScores == nil)
        #expect(noSinkScores.forwardBlockScores.allSatisfy { $0.isFinite })

        var tinyRevCfg = cfg
        tinyRevCfg.reversePrefixEnd = 4  // prefEnd <= sinkEnd
        let emptyPrefScores = Gemma4BlockSDCM.computeBilateralBlockScores(
            queries: q, keys: k, positions: MLXArray(nonContigPos),
            positionsHost: nonContigPos, originalLength: origLen,
            scale: scale, nHeads: nHeads, nKvHeads: nKvHeads,
            reverseQuotaFraction: 0.30, config: tinyRevCfg)
        #expect(emptyPrefScores.reverseBlockScores == nil)

        // 4. Constant queries/keys (zero interior variance) stay finite with zero z-score blowup.
        let constQ = MLXArray.ones([1, nHeads, L, D], dtype: .float32)
        let constK = MLXArray.ones([1, nKvHeads, L, D], dtype: .float32)
        let constScores = Gemma4BlockSDCM.computeBilateralBlockScores(
            queries: constQ, keys: constK, positions: MLXArray(nonContigPos),
            positionsHost: nonContigPos, originalLength: origLen,
            scale: scale, nHeads: nHeads, nKvHeads: nKvHeads,
            reverseQuotaFraction: 0.30, config: cfg)
        #expect(constScores.forwardBlockScores.allSatisfy { $0.isFinite })
        if let revConst = constScores.reverseBlockScores {
            #expect(revConst.allSatisfy { $0 == 0 })
        }

        // 5. selectCompactedIndices stays deterministic and crash-free on NaN / -Inf scores with zero halo.
        var degenerateFwd = [Float](repeating: 0, count: nBlocks)
        degenerateFwd[5] = .nan
        degenerateFwd[6] = -.infinity
        degenerateFwd[7] = 5.0
        var degenerateRev = [Float](repeating: -.infinity, count: nBlocks)
        degenerateRev[10] = .nan
        degenerateRev[11] = 4.0
        var zeroHaloCfg = cfg
        zeroHaloCfg.forwardHalo = 0
        zeroHaloCfg.backwardHalo = 0
        let keptDegenerate = Gemma4BlockSDCM.selectCompactedIndices(
            boundaryScores: Gemma4SDCMBoundaryScores(
                forwardBlockScores: degenerateFwd,
                reverseBlockScores: degenerateRev,
                reverseQuotaFraction: 0.25),
            positions: nonContigPos,
            originalLength: origLen,
            targetFraction: 0.50,
            config: zeroHaloCfg)
        let keptDegenerateBlocks = Set(
            keptDegenerate.map { Int(nonContigPos[Int($0)]) / blockSize })
        #expect(keptDegenerateBlocks.contains(7))
        #expect(keptDegenerateBlocks.contains(11))
    }

    @Test(
        "Quota-partitioned bilateral selection guarantees both top forward and top reverse prefix blocks"
    )
    func bilateralQuotaSelectionGuarantees() {
        let blockSize = 32
        let nBlocks = 32
        let T = nBlocks * blockSize  // 1024 tokens
        let positions = (0 ..< T).map { Int32($0) }

        var fwd = [Float](repeating: 0.0, count: nBlocks)
        var rev = [Float](repeating: 0.0, count: nBlocks)
        // Pinned prefix block 1 has huge sink score; interior-only halo must NOT bleed into block 2!
        fwd[1] = 100.0
        rev[1] = 100.0
        // Forward-only peak at block 8 (bottom-question hit)
        fwd[8] = 10.0
        // Reverse-only peak at block 18 (top-of-prompt instruction hit, invisible to forward!)
        rev[18] = 10.0
        // Ensure non-adjacent interior block 12 beats block 2 (score 0) when sink halo is blocked.
        fwd[12] = 5.0
        rev[12] = 5.0

        var cfg = Gemma4BlockSDCMConfig.fast
        cfg.blockSize = blockSize
        cfg.prefixKeep = 64  // blocks 0..1
        cfg.tailKeep = 96  // blocks 29..31
        cfg.strataCount = 4
        cfg.forwardHalo = 0.85
        cfg.backwardHalo = 0.85

        let boundary = Gemma4SDCMBoundaryScores(
            forwardBlockScores: fwd,
            reverseBlockScores: rev,
            reverseQuotaFraction: 0.30)
        let kept = Gemma4BlockSDCM.selectCompactedIndices(
            boundaryScores: boundary, positions: positions,
            originalLength: T, targetFraction: 0.45, config: cfg)
        let keptBlocks = Set(kept.map { Int($0) / blockSize })
        // Both forward peak (block 8) AND reverse peak (block 18) + halos must survive!
        #expect(keptBlocks.contains(8))
        #expect(keptBlocks.contains(18))
        #expect(keptBlocks.contains(17))
        #expect(keptBlocks.contains(19))
        #expect(keptBlocks.contains(12))
        // Block 2 (adjacent to pinned prefix block 1) must not inherit block 1's +100 sink score.
        #expect(!keptBlocks.contains(2))
    }

    @Test(
        "Gemma4Router.routeTopK matches callAsFunction at full topK and renormalizes softmax at smaller k"
    )
    func routerRouteTopKParityAndRenorm() throws {
        MLXRandom.seed(99)
        let model = try Self.makeSyntheticGemma4MoE()
        let router = Gemma4Router(model.configuration)
        let x = MLXRandom.normal([1, 16, 64]).asType(.float32)

        let (fullIdx, fullW) = router(x)
        let (sameIdx, sameW) = router.routeTopK(x, topK: router.topK)
        eval(fullIdx, fullW, sameIdx, sameW)
        #expect(MLX.abs(fullIdx - sameIdx).max().item(Int32.self) == 0)
        #expect(MLX.abs(fullW - sameW).max().item(Float.self) == 0)

        // When perExpertScale == 1, routeTopK(k=2) softmax weights sum to 1.0 per token.
        let (subIdx, subW) = router.routeTopK(x, topK: 2)
        eval(subIdx, subW)
        #expect(subIdx.shape == [1, 16, 2])
        #expect(subW.shape == [1, 16, 2])
        let rowSums = subW.sum(axis: -1)
        eval(rowSums)
        let sumErr = MLX.abs(rowSums - Float(1.0)).max().item(Float.self)
        #expect(sumErr < 1e-5, "routeTopK softmax renormalization error: \(sumErr)")

        // Boundary clamping: requestedK < 1 clamps to 1; requestedK > numExperts clamps to numExperts.
        let (minIdx, minW) = router.routeTopK(x, topK: 0)
        let (maxIdx, maxW) = router.routeTopK(x, topK: 99)
        eval(minIdx, minW, maxIdx, maxW)
        #expect(minIdx.shape == [1, 16, 1])
        #expect(maxIdx.shape == [1, 16, model.configuration.numExperts ?? 8])
    }

    // MARK: - Block-SDCM v3 (Turbo Early Tri-Signal + Position-Aware Sparse L5)

    @Test(
        "prefixLocalFullTailAttention with non-contiguous positions matches exact explicit mask and edge cases"
    )
    func prefixLocalFullTailWithPositionsParity() {
        MLXRandom.seed(2026)
        let scale: Float = 1.0 / sqrt(32.0)

        func verifyAgainstReference(
            posHost: [Int32], P: Int, W: Int, tail: Int, chunkSize: Int, label: String
        ) {
            let L = posHost.count
            let posArr = MLXArray(posHost)
            let q = MLXRandom.normal([1, 4, L, 32]).asType(.float32)
            let k = MLXRandom.normal([1, 2, L, 32]).asType(.float32)
            let v = MLXRandom.normal([1, 2, L, 32]).asType(.float32)

            let sparse = Gemma4BlockSDCM.prefixLocalFullTailAttention(
                queries: q, keys: k, values: v,
                positions: posArr, positionsHost: posHost,
                prefixTokens: P, localWindow: W, tailTokens: tail,
                scale: scale, chunkSize: chunkSize)

            let ref: MLXArray
            if W <= 0 || tail >= L {
                ref = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: .causal)
            } else {
                let qPos = posArr.expandedDimensions(axis: 1)
                let kPos = posArr.expandedDimensions(axis: 0)
                let qIdx = MLXArray((0 ..< L).map { Int32($0) }).expandedDimensions(axis: 1)
                let tailCut = Int32(max(0, L - max(0, tail)))
                let refMask =
                    (kPos .<= qPos)
                    .&& ((kPos .< Int32(max(0, P))) .|| ((qPos - kPos) .< Int32(W))
                        .|| (qIdx .>= tailCut))
                ref = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: .array(refMask))
            }
            eval(sparse, ref)
            let diff = MLX.abs(sparse - ref).max().item(Float.self)
            #expect(diff < 1e-5, "Position-aware sparse L5 mismatch (\(label)): \(diff)")
        }

        // Simulate 55% post-L0 compacted positions: keep prefix [0..<64], tail [576..<640], and alternating interior blocks.
        let origLen = 640
        var compactedPos: [Int32] = []
        for b in 0 ..< (origLen / 32) {
            if b < 2 || b >= 18 || b.isMultiple(of: 2) {
                for j in 0 ..< 32 { compactedPos.append(Int32(b * 32 + j)) }
            }
        }
        // Test across small chunk, medium chunk, default chunkSize (1024 > L), and non-positive chunkSize.
        for chunk in [16, 64, 1024, 0, -8] {
            verifyAgainstReference(
                posHost: compactedPos, P: 64, W: 96, tail: 64, chunkSize: chunk,
                label: "compacted chunk=\(chunk)")
        }
        // P == 0, T == 0, T >= L (cut <= 0), W <= 0.
        verifyAgainstReference(
            posHost: compactedPos, P: 0, W: 96, tail: 64, chunkSize: 32, label: "P=0")
        verifyAgainstReference(
            posHost: compactedPos, P: 64, W: 96, tail: 0, chunkSize: 32, label: "T=0")
        verifyAgainstReference(
            posHost: compactedPos, P: 0, W: 96, tail: 0, chunkSize: 32, label: "P=0,T=0")
        verifyAgainstReference(
            posHost: compactedPos, P: 64, W: 96, tail: compactedPos.count, chunkSize: 32,
            label: "T==L")
        verifyAgainstReference(
            posHost: compactedPos, P: 64, W: 0, tail: 64, chunkSize: 32, label: "W=0")
        verifyAgainstReference(
            posHost: compactedPos, P: 64, W: -16, tail: 64, chunkSize: 32, label: "W<0")

        // Arbitrary large gaps >> W where k0 transitions across pEnd (and where pEnd == 0).
        let hugeGapPos: [Int32] = [0, 1, 2, 3, 500, 501, 502, 4000, 4001, 9000, 9001, 9002]
        verifyAgainstReference(
            posHost: hugeGapPos, P: 3, W: 16, tail: 2, chunkSize: 2, label: "huge-gaps-prefix")
        verifyAgainstReference(
            posHost: hugeGapPos, P: 3, W: 16, tail: 0, chunkSize: 1, label: "huge-gaps-chunk1")
        let shiftedPos: [Int32] = [100, 101, 250, 251, 600, 601, 900]
        verifyAgainstReference(
            posHost: shiftedPos, P: 50, W: 32, tail: 1, chunkSize: 2, label: "pEnd==0")

        // Exact window boundary: pos[cut - 1] - pos[pEnd] == W - 1 (exact causal) vs == W (sparse).
        let boundaryPos: [Int32] = [0, 1, 10, 15, 25, 26, 40]
        verifyAgainstReference(
            posHost: boundaryPos, P: 10, W: 17, tail: 1, chunkSize: 1024, label: "span==W-1")
        verifyAgainstReference(
            posHost: boundaryPos, P: 10, W: 16, tail: 1, chunkSize: 1024, label: "span==W")

        // Contiguous overload with small W < 64 where L - P - T > W (must NOT fall back to causal).
        let qSmallW = MLXRandom.normal([1, 4, 128, 32]).asType(.float32)
        let kSmallW = MLXRandom.normal([1, 2, 128, 32]).asType(.float32)
        let vSmallW = MLXRandom.normal([1, 2, 128, 32]).asType(.float32)
        let sparseSmallW = Gemma4BlockSDCM.prefixLocalFullTailAttention(
            queries: qSmallW, keys: kSmallW, values: vSmallW,
            prefixTokens: 32, localWindow: 32, tailTokens: 32, scale: scale)
        let qPos128 = MLXArray((0 ..< 128).map { Int32($0) }).expandedDimensions(axis: 1)
        let kPos128 = MLXArray((0 ..< 128).map { Int32($0) }).expandedDimensions(axis: 0)
        let refMask128 =
            (kPos128 .<= qPos128)
            .&& ((kPos128 .< Int32(32)) .|| ((qPos128 - kPos128) .< Int32(32))
                .|| (qPos128 .>= Int32(96)))
        let refSmallW = MLXFast.scaledDotProductAttention(
            queries: qSmallW, keys: kSmallW, values: vSmallW, scale: scale, mask: .array(refMask128)
        )
        eval(sparseSmallW, refSmallW)
        #expect(MLX.abs(sparseSmallW - refSmallW).max().item(Float.self) < 1e-5)

        // L == 0 returns empty zeros cleanly.
        let emptyOut = Gemma4BlockSDCM.prefixLocalFullTailAttention(
            queries: MLXArray.zeros([1, 4, 0, 32], dtype: .float32),
            keys: MLXArray.zeros([1, 2, 0, 32], dtype: .float32),
            values: MLXArray.zeros([1, 2, 0, 32], dtype: .float32),
            positions: MLXArray([Int32]()),
            positionsHost: [],
            prefixTokens: 32, localWindow: 64, tailTokens: 32, scale: scale, chunkSize: 0)
        #expect(emptyOut.shape == [1, 4, 0, 32])
    }

    @Test(
        "selectEarlyTriSignalIndices retains Query MaxSim, Outlier Novelty, and Token-IDF blocks with targeted halo"
    )
    func earlyTriSignalSelectionGuarantees() {
        let bs = 32
        let nBlocks = 24
        let T = nBlocks * bs  // 768 tokens
        let D = 16

        // Build synthetic embeddings [1, T, D]:
        //  - Boilerplate interior blocks point along axis 0.
        //  - Prefix tokens [4..<32] point along axis 1; block 6 matches prefix (Query MaxSim hit) -> should keep 6 + halo 5,7!
        //  - Post-L0 layerHidden has block 12 pointing along axis 2 (zero lexical overlap Outlier Novelty hit!).
        //  - Token IDs are uniform boilerplate (id=10) except block 16 has unique rare IDs (5000..5031 -> Token-IDF hit!).
        var embBuf = [Float](repeating: 0, count: T * D)
        var layBuf = [Float](repeating: 0, count: T * D)
        var tokIDs = [Int32](repeating: 10, count: T)

        for i in 0 ..< T {
            embBuf[i * D + 0] = 1.0
            layBuf[i * D + 0] = 1.0
        }
        // Prefix query along axis 1
        for i in 4 ..< 32 {
            embBuf[i * D + 0] = 0.0
            embBuf[i * D + 1] = 1.0
        }
        // Block 6: matches prefix query in embedding space (Query MaxSim hit)
        for i in (6 * bs) ..< (7 * bs) {
            embBuf[i * D + 0] = 0.0
            embBuf[i * D + 1] = 1.0
        }
        // Block 12: orthogonal outlier in post-L0 hidden state (C @ C^T Novelty hit)
        for i in (12 * bs) ..< (13 * bs) {
            layBuf[i * D + 0] = 0.0
            layBuf[i * D + 2] = 1.0
        }
        // Block 16: unique rare token IDs (Token-IDF Rarity hit)
        for j in 0 ..< bs {
            tokIDs[16 * bs + j] = Int32(5000 + j)
        }

        let emb = MLXArray(embBuf, [1, T, D])
        let lay = MLXArray(layBuf, [1, T, D])
        var cfg = Gemma4BlockSDCMConfig.turbo
        cfg.prefixKeep = 64  // blocks 0..1
        cfg.tailKeep = 64  // blocks 22..23
        cfg.strataCount = 4

        let kept = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: tokIDs,
            embeddingHidden: emb,
            layerHidden: lay,
            targetFraction: 0.50,
            blockSize: bs,
            topHaloK: 1,
            config: cfg)
        // Strictly ascending and compacted below T
        #expect(kept.count < T)
        for i in 1 ..< kept.count { #expect(kept[i] > kept[i - 1]) }
        let keptBlocks = Set(kept.map { Int($0) / bs })
        // Prefix & tail pinned
        #expect(keptBlocks.contains(0) && keptBlocks.contains(1))
        #expect(keptBlocks.contains(22) && keptBlocks.contains(23))
        // Signal 1 (Query MaxSim) + targeted ±1 halo
        #expect(keptBlocks.contains(6))
        #expect(keptBlocks.contains(5) && keptBlocks.contains(7))
        // Signal 2 (C @ C^T Outlier Novelty)
        #expect(keptBlocks.contains(12))
        // Signal 3 (CPU Token-IDF Rarity)
        #expect(keptBlocks.contains(16))
    }

    @Test(
        "selectEarlyTriSignalIndices handles T=0/1, p1<=p0, partial tail, nil/wrong tokenIDs, zero variance, large topHaloK, NaN/Inf, and FP16"
    )
    func earlyTriSignalEdgeCases() {
        let D = 16
        var cfg = Gemma4BlockSDCMConfig.turbo
        cfg.prefixKeep = 32
        cfg.tailKeep = 32
        cfg.strataCount = 4

        // 1. T = 0, T = 1, and D = 0
        #expect(
            Gemma4BlockSDCM.selectEarlyTriSignalIndices(
                tokenIDs: [],
                embeddingHidden: MLXArray.zeros([1, 0, D], dtype: .float32),
                layerHidden: MLXArray.zeros([1, 0, D], dtype: .float32),
                targetFraction: 0.50, blockSize: 16, topHaloK: 4, config: cfg
            ).isEmpty)
        #expect(
            Gemma4BlockSDCM.selectEarlyTriSignalIndices(
                tokenIDs: [42],
                embeddingHidden: MLXArray.ones([1, 1, D], dtype: .float32),
                layerHidden: MLXArray.ones([1, 1, D], dtype: .float32),
                targetFraction: 0.50, blockSize: 16, topHaloK: 4, config: cfg
            ) == [0])
        #expect(
            Gemma4BlockSDCM.selectEarlyTriSignalIndices(
                tokenIDs: [1, 2, 3, 4],
                embeddingHidden: MLXArray.zeros([1, 4, 0], dtype: .float32),
                layerHidden: MLXArray.zeros([1, 4, 0], dtype: .float32),
                targetFraction: 0.50, blockSize: 2, topHaloK: 1, config: cfg
            ) == [0, 1, 2, 3])

        // 2. p1 <= p0 (T = 16 -> p0 = 4, T/4 = 4 => p1 == p0) with > 4 interior blocks
        var tinyCfg = cfg
        tinyCfg.prefixKeep = 2
        tinyCfg.tailKeep = 2
        tinyCfg.strataCount = 2
        MLXRandom.seed(11)
        let emb16 = MLXRandom.normal([1, 16, D]).asType(.float32)
        let lay16 = MLXRandom.normal([1, 16, D]).asType(.float32)
        let kept16 = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: nil, embeddingHidden: emb16, layerHidden: lay16,
            targetFraction: 0.50, blockSize: 2, topHaloK: 1, config: tinyCfg)
        #expect(kept16.first == 0 && kept16.last == 15)
        #expect(kept16.count < 16)

        // 3. Partial last block (T % bs != 0), nil & wrong-length tokenIDs, and constant zero-variance inputs
        let T = 373  // 12 blocks of 32 (last block has 21 tokens)
        let bs = 32
        let embRand = MLXRandom.normal([1, T, D]).asType(.float32)
        let layRand = MLXRandom.normal([1, T, D]).asType(.float32)
        let keptNilTok = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: nil, embeddingHidden: embRand, layerHidden: layRand,
            targetFraction: 0.50, blockSize: bs, topHaloK: 2, config: cfg)
        let keptBadTok = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: [1, 2, 3], embeddingHidden: embRand, layerHidden: layRand,
            targetFraction: 0.50, blockSize: bs, topHaloK: 2, config: cfg)
        #expect(keptNilTok == keptBadTok)
        #expect(keptNilTok.first == 0 && keptNilTok.last == Int32(T - 1))
        #expect(keptNilTok.count < T)

        let constEmb = MLXArray.ones([1, T, D], dtype: .float32)
        let constLay = MLXArray.ones([1, T, D], dtype: .float32)
        let constTok = [Int32](repeating: 7, count: T)
        let keptConst = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: constTok, embeddingHidden: constEmb, layerHidden: constLay,
            targetFraction: 0.50, blockSize: bs, topHaloK: 2, config: cfg)
        #expect(keptConst.first == 0 && keptConst.last == Int32(T - 1))
        #expect(keptConst.count < T)

        // 4. topHaloK far exceeding needInteriorBlocks must stay bounded by needInteriorBlocks and compact < T.
        let keptHugeHalo = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: constTok, embeddingHidden: embRand, layerHidden: layRand,
            targetFraction: 0.50, blockSize: bs, topHaloK: 100, config: cfg)
        let keptHugeHaloBlocks = Set(keptHugeHalo.map { Int($0) / bs })
        // Pinned: block 0 (32) + block 11 (21) = 53 tokens; desired = 187 -> needInteriorBlocks = ceil(134/32) = 5.
        #expect(keptHugeHaloBlocks.count == 2 + 5)
        #expect(keptHugeHalo.count < T)

        // 5. Non-finite inputs (NaN / +Inf / -Inf) remain crash-free, strictly ascending, and pinned.
        var badEmb = [Float](repeating: 1.0, count: T * D)
        var badLay = [Float](repeating: 1.0, count: T * D)
        badEmb[5 * D] = .nan
        badEmb[100 * D] = .infinity
        badLay[120 * D] = -.infinity
        badLay[200 * D] = .nan
        let keptNonFinite = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: constTok,
            embeddingHidden: MLXArray(badEmb, [1, T, D]),
            layerHidden: MLXArray(badLay, [1, T, D]),
            targetFraction: 0.50, blockSize: bs, topHaloK: 2, config: cfg)
        #expect(keptNonFinite.first == 0 && keptNonFinite.last == Int32(T - 1))
        for i in 1 ..< keptNonFinite.count { #expect(keptNonFinite[i] > keptNonFinite[i - 1]) }

        // 6. FP16 inputs with large magnitude (30,000 -> would overflow FP16 when squared) stay safe.
        let fp16Emb = (embRand * Float(10_000.0)).asType(.float16)
        let fp16Lay = (layRand * Float(10_000.0)).asType(.float16)
        let keptFP16 = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: (0 ..< T).map { Int32($0 % 64) },
            embeddingHidden: fp16Emb,
            layerHidden: fp16Lay,
            targetFraction: 0.50, blockSize: bs, topHaloK: 2, config: cfg)
        #expect(keptFP16.first == 0 && keptFP16.last == Int32(T - 1))
        #expect(keptFP16.count < T)
    }

    @Test(
        "End-to-end turbo Block-SDCM v3 prefills and decodes cleanly across Paged and Standalone KV caches"
    )
    func endToEndTurboPrefillAndDecode() throws {
        MLXRandom.seed(404)
        let model = try Self.makeSyntheticGemma4MoE()
        var turboCfg = Gemma4BlockSDCMConfig.turbo
        turboCfg.minPromptTokens = 128
        turboCfg.prefixKeep = 16
        turboCfg.tailKeep = 16
        turboCfg.earlyPreCompactAfterLayer = 0
        turboCfg.earlyPreCompactRatio = 0.60
        turboCfg.earlyPreCompactBlockSize = 16
        turboCfg.earlyPreCompactTopHaloK = 2
        turboCfg.blockSize = 16
        // Synthetic 12-layer 5:1 model has full attention at layers 5 and 11.
        turboCfg.keepSchedule = [6: 0.35]
        turboCfg.reversePrefixQuota = [6: 0.25]
        turboCfg.reversePrefixEnd = 24
        turboCfg.earlySparseFullLayer = 5
        turboCfg.sparsePrefixTokens = 16
        turboCfg.sparseLocalWindow = 32
        turboCfg.sparseTailTokens = 16
        turboCfg.slidingChunkSize = 64
        model.blockSDCMConfig = turboCfg

        let T = 256
        let promptIDs = (0 ..< T).map { Int32(($0 * 11 + 5) % model.vocabularySize) }
        let prompt1D = MLXArray(promptIDs)
        let prompt = prompt1D.expandedDimensions(axis: 0)
        let nextTok = MLXArray([Int32(7)]).reshaped([1, 1])

        // 1. Standalone KVCache (`prepare` prefills [0 ..< T - 1], returns final token)
        do {
            let simpleCaches = model.newCache(parameters: nil)
            let prepared = try model.prepare(
                LMInput(tokens: prompt1D), cache: simpleCaches, state: nil,
                prefill: PrefillParameters())
            guard case .tokens(let remaining) = prepared else {
                Issue.record("Expected .tokens from turbo Block-SDCM prepare")
                return
            }
            #expect(remaining.tokens.size == 1)
            for c in simpleCaches {
                if case .scalar(let off) = c.ropeOffset {
                    #expect(off == T - 1)
                } else {
                    Issue.record("Expected scalar ropeOffset on standalone cache")
                }
            }
            let frontierLogits = model(
                remaining.tokens.expandedDimensions(axis: 0), cache: simpleCaches)
            eval(frontierLogits)
            #expect(frontierLogits.shape == [1, 1, model.vocabularySize])
            #expect(!MLX.isNaN(frontierLogits).any().item(Bool.self))

            let decLogits = model(nextTok, cache: simpleCaches)
            eval(decLogits)
            #expect(decLogits.shape == [1, 1, model.vocabularySize])
            #expect(!MLX.isNaN(decLogits).any().item(Bool.self))
        }

        // 2. PagedLayerCache via CBv2
        do {
            let kinds = model.cbv2LayerKinds
            let backend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 64 * 1024 * 1024,
                    dtype: .float32, maxPrefillChunk: 256))
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: T, maxLength: T + 32)
            defer { backend.release(state) }
            let rows = state.map { $0 as! PagedSequenceKV }
            let pagedCaches = backend.makeLayerCaches()
            let caches: [KVCache] = pagedCaches.enumerated().map { idx, c in
                c.setRows([rows[idx]])
                return c
            }

            let pagedOut = model.cbv2Prefill(
                prompt, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
            eval(pagedOut)
            #expect(pagedOut.shape == [1, model.vocabularySize])
            #expect(!MLX.isNaN(pagedOut).any().item(Bool.self))

            // Layer 5 runs AFTER L0 early pre-compaction (< T); Layer 11 runs AFTER L6 bilateral compaction (< L5).
            #expect(rows[5].absoluteOffset < T)
            #expect(rows[11].absoluteOffset < rows[5].absoluteOffset)
            for layer in 0 ..< kinds.count {
                #expect(rows[layer].ropePositionOffset == T)
                #expect(rows[layer].isCompacted)
            }

            let pagedDec = model(nextTok, cache: caches)
            eval(pagedDec)
            #expect(pagedDec.shape == [1, 1, model.vocabularySize])
            #expect(!MLX.isNaN(pagedDec).any().item(Bool.self))
            for layer in 0 ..< kinds.count {
                #expect(rows[layer].ropePositionOffset == T + 1)
            }
        }
    }
}
