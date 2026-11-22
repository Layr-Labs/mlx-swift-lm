import Foundation
import MLX
import MLXFast
import MLXNN
import Testing

@_spi(GemmaEncoder) @testable import MLXLLM
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
        #expect(lcDefault.minPromptTokens == 3072)
        #expect(lcDefault.densityGuardEnabled)
        #expect(lcDefault.maxPackedBatch == 8)
        #expect(lcDefault.maxPackedTokens == 32_768)
        #expect(lcDefault.interiorMoEK(forLayer: 2) == 2)
        #expect(lcDefault.interiorMoEK(forLayer: 6) == 5)
        #expect(lcDefault.interiorMoEK(forLayer: 12) == 8)
        #expect(lcDefault.earlySparseFullLayer == 5)

        let bal = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "balanced"
        ])
        #expect(bal.mode == .balanced)
        #expect(bal.minPromptTokens == 3072)
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
        #expect(fast.minPromptTokens == 3072)
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
            #expect(turbo.minPromptTokens == 3072)
            #expect(turbo.densityGuardEnabled)
            #expect(turbo.maxPackedBatch == 8)
            #expect(turbo.maxPackedTokens == 32_768)
            #expect(turbo.earlyPreCompactAfterLayer == 0)
            #expect(turbo.earlyPreCompactRatio == 0.55)
            #expect(turbo.earlyPreCompactBlockSize == 64)
            #expect(turbo.earlyPreCompactTopHaloK == 8)
            #expect(turbo.blockSize == 32)
            #expect(turbo.keepSchedule[6] == 0.36)
            #expect(turbo.reversePrefixQuota[24] == 0.20)
        }

        let overridden = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "fast",
            Gemma4BlockSDCMConfig.densityGuardEnvKey: "0",
            Gemma4BlockSDCMConfig.maxBatchEnvKey: "4",
            Gemma4BlockSDCMConfig.maxTokensEnvKey: "65536",
        ])
        #expect(!overridden.densityGuardEnabled)
        #expect(overridden.maxPackedBatch == 4)
        #expect(overridden.maxPackedTokens == 65_536)

        let clampedLow = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "fast",
            Gemma4BlockSDCMConfig.maxBatchEnvKey: "0",
            Gemma4BlockSDCMConfig.maxTokensEnvKey: "1024",
        ])
        #expect(clampedLow.maxPackedBatch == 1)
        #expect(clampedLow.maxPackedTokens == 3072)

        let clampedHigh = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "fast",
            Gemma4BlockSDCMConfig.maxBatchEnvKey: "64",
        ])
        #expect(clampedHigh.maxPackedBatch == 32)
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

    // MARK: - Checkpoint A: Density Guard, Batched B=4 Block-SDCM, Mode A & Mode B

    @Test(
        "passesDensityGuard vetoes 20-AA protein and dense short prose while passing 4K-16K code/logs and tiny tests"
    )
    func densityGuardSafetyGate() throws {
        var rng = CBv2SchedSplitMix64(seed: 0xC0DE_5AFE)
        let cfg = Gemma4BlockSDCMConfig.fast

        // 1. 20-AA protein sequence (4096 tokens, H1 ~ 4.32 bits < 5.0, repeated4GramRatio < 0.20) -> VETO
        let proteinTokens: [Int] = (0 ..< 4096).map { _ in Int(100 + rng.next() % 20) }
        #expect(!Gemma4BlockSDCM.passesDensityGuard(tokens: proteinTokens, config: cfg))

        // Disabling density guard overrides veto cleanly.
        var guardOffCfg = cfg
        guardOffCfg.densityGuardEnabled = false
        #expect(Gemma4BlockSDCM.passesDensityGuard(tokens: proteinTokens, config: guardOffCfg))

        // 2. Short ultra-dense prose/proof (n = 3584 < 4096, uniqueRatio > 0.50, repeated4GramRatio < 0.05) -> VETO
        let denseShortProse: [Int] = (0 ..< 3584).map { i in 10_000 + i }
        #expect(!Gemma4BlockSDCM.passesDensityGuard(tokens: denseShortProse, config: cfg))

        // 3. 4K and 16K code/logs (H1 >= 5.0 with recurring code/log templates) -> PASS
        let code4K: [Int] = (0 ..< 4096).map { i in
            (i % 16 < 8) ? (500 + (i % 8)) : Int(600 + rng.next() % 256)
        }
        let logs16K: [Int] = (0 ..< 16_384).map { i in
            (i % 24 < 10) ? (900 + (i % 10)) : Int(1_000 + rng.next() % 512)
        }
        #expect(Gemma4BlockSDCM.passesDensityGuard(tokens: code4K, config: cfg))
        #expect(Gemma4BlockSDCM.passesDensityGuard(tokens: logs16K, config: cfg))

        // 4. Tiny synthetic prompt (interior N < 512) always passes.
        let tinyProtein = Array(proteinTokens.prefix(500))
        #expect(Gemma4BlockSDCM.passesDensityGuard(tokens: tinyProtein, config: cfg))

        // 5. RandomAccessCollection slice with non-zero startIndex works identically.
        let sliceBase: [Int32] = Array(repeating: 0, count: 17) + proteinTokens.map(Int32.init)
        let slicedProtein = sliceBase[17...]
        #expect(!Gemma4BlockSDCM.passesDensityGuard(tokens: slicedProtein, config: cfg))

        // 6. canRun enforces maxPackedBatch and passesDensityGuard.
        let model = try Self.makeSyntheticGemma4MoE()
        var batchCfg = Gemma4BlockSDCMConfig.fast
        batchCfg.minPromptTokens = 3072
        batchCfg.maxPackedBatch = 8
        #expect(
            Gemma4BlockSDCM.canRun(
                sdcm: batchCfg, modelConfig: model.configuration, schedulePrefill: true,
                batchSize: 4, sequenceLength: 4096, hasInputEmbedding: false,
                hasImageMask: false, hasCaptureHook: false, initialOffset: 0,
                passesDensityGuard: true))
        #expect(
            !Gemma4BlockSDCM.canRun(
                sdcm: batchCfg, modelConfig: model.configuration, schedulePrefill: true,
                batchSize: 4, sequenceLength: 4096, hasInputEmbedding: false,
                hasImageMask: false, hasCaptureHook: false, initialOffset: 0,
                passesDensityGuard: false))
        #expect(
            !Gemma4BlockSDCM.canRun(
                sdcm: batchCfg, modelConfig: model.configuration, schedulePrefill: true,
                batchSize: 9, sequenceLength: 4096, hasInputEmbedding: false,
                hasImageMask: false, hasCaptureHook: false, initialOffset: 0,
                passesDensityGuard: true))
    }

    @Test(
        "Batched B=4 applyArbitraryPositionsRoPE matches B=1 per-row and scalar RoPE bit-identically"
    )
    func batchedArbitraryPositionRoPEParity() {
        MLXRandom.seed(202604)
        let B = 4
        let H = 4
        let L = 24
        let slidingRope = initializeRope(
            dims: 64, base: 10_000, traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: 4096)
        let fullRope = initializeRope(
            dims: 128, base: 1_000_000, traditional: false,
            scalingConfig: [
                "type": .string("proportional"), "partial_rotary_factor": .float(0.25),
            ],
            maxPositionEmbeddings: 4096)

        // Distinct non-contiguous ascending positions per batch row.
        var posRows: [[Int32]] = []
        var flatPos: [Int32] = []
        for b in 0 ..< B {
            let row = (0 ..< L).map { j in Int32(j * (b + 2) + b * 19) }
            posRows.append(row)
            flatPos.append(contentsOf: row)
        }
        let pos2D = MLXArray(flatPos, [B, L])

        for (label, rope, dim) in [
            ("sliding", slidingRope, 64), ("full-proportional", fullRope, 128),
        ] {
            let x = MLXRandom.normal([B, H, L, dim]).asType(.float32)
            let batched = Gemma4BlockSDCM.applyArbitraryPositionsRoPE(
                rope, to: x, positions: pos2D)
            let perRow = concatenated(
                (0 ..< B).map { b in
                    Gemma4BlockSDCM.applyArbitraryPositionsRoPE(
                        rope, to: x[b ..< (b + 1)], positions: MLXArray(posRows[b]))
                }, axis: 0)
            let scalarRef = concatenated(
                (0 ..< B).map { b in
                    concatenated(
                        posRows[b].enumerated().map { idx, p in
                            rope(
                                x[b ..< (b + 1), 0..., idx ..< (idx + 1), 0...],
                                offset: Int(p))
                        }, axis: 2)
                }, axis: 0)
            eval(batched, perRow, scalarRef)
            let diffRow = MLX.abs(batched - perRow).max().item(Float.self)
            let diffScalar = MLX.abs(batched - scalarRef).max().item(Float.self)
            #expect(diffRow == 0, "\(label) B=4 vs B=1 RoPE diff: \(diffRow)")
            #expect(diffScalar < 1e-5, "\(label) B=4 vs scalar RoPE diff: \(diffScalar)")
        }
    }

    @Test(
        "PagedLayerCache and CBv2LayerCache per-row writeCompactedPromptKV handles distinct lengths, chunked sliding, prefix veto, and B=4 vs B=1 decode parity"
    )
    func batchedWriteCompactedPromptKVPerRowBothBackends() throws {
        MLXRandom.seed(808)
        let B = 4
        let fullKind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        // maxPrefillChunk (32) < slidingWindow (64) exercises chunked per-row sliding writes.
        let slidingKind = CBv2LayerKind(
            attention: .slidingWindow(64), headDim: 64, kvHeads: 2, queryHeads: 4)
        let kinds = [fullKind, slidingKind]
        let poolConfig = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 32 * 1024 * 1024, dtype: .float32, maxPrefillChunk: 32)

        let storedCounts = [0, 17, 48, 64]
        let uniformDeltas = storedCounts.map { 128 - $0 }  // all advance by 128
        let heteroDeltas = [128, 113, 92, 86]  // advances: [128, 130, 140, 150]

        let perRowK3D = storedCounts.map { n in
            MLXRandom.normal([2, n, 64]).asType(.float32)
        }
        let perRowV3D = storedCounts.map { n in
            MLXRandom.normal([2, n, 64]).asType(.float32)
        }
        let perRowK4D = perRowK3D.map { $0.expandedDimensions(axis: 0) }
        let perRowV4D = perRowV3D.map { $0.expandedDimensions(axis: 0) }

        let qDec = MLXRandom.normal([B, 4, 1, 64]).asType(.float32)
        let kDec = MLXRandom.normal([B, 2, 1, 64]).asType(.float32)
        let vDec = MLXRandom.normal([B, 2, 1, 64]).asType(.float32)

        // 1. PagedLayerCache: B=4 per-row (3D uniform + 4D heterogeneous) vs B=1 parity.
        do {
            let pagedB4 = try PagedKVBackend(layerKinds: kinds, config: poolConfig)
            let pagedB1 = try PagedKVBackend(layerKinds: kinds, config: poolConfig)
            let statesB4 = try (0 ..< B).map { _ in
                try pagedB4.makeSequenceState(layerKinds: kinds, promptLength: 128, maxLength: 256)
            }
            let statesB1 = try (0 ..< B).map { _ in
                try pagedB1.makeSequenceState(layerKinds: kinds, promptLength: 128, maxLength: 256)
            }
            defer {
                for s in statesB4 { pagedB4.release(s) }
                for s in statesB1 { pagedB1.release(s) }
            }

            let cachesB4 = pagedB4.makeLayerCaches()
            let cachesB1 = pagedB1.makeLayerCaches()
            let fullRowsB4 = statesB4.map { $0[0] as! PagedSequenceKV }
            let slideRowsB4 = statesB4.map { $0[1] as! PagedSequenceKV }
            cachesB4[0].setRows(fullRowsB4)
            cachesB4[1].setRows(slideRowsB4)

            // Full layer with 4D heterogeneous advances, sliding layer with 3D uniform advances.
            cachesB4[0].writeCompactedPromptKV(
                perRowKeys: perRowK4D, perRowValues: perRowV4D, logicalDeltas: heteroDeltas)
            cachesB4[1].writeCompactedPromptKV(
                perRowKeys: perRowK3D, perRowValues: perRowV3D, logicalDeltas: uniformDeltas)
            #expect(!pagedB4.pool.writeValidation.isFaulted)

            let expectedFullOff = zip(storedCounts, heteroDeltas).map { Int32($0 + $1) }
            let expectedSlideOff = [Int32](repeating: 128, count: B)
            eval(cachesB4[0].positionOffsets, cachesB4[1].positionOffsets)
            #expect(cachesB4[0].hostPositionOffsets == expectedFullOff)
            #expect(cachesB4[0].positionOffsets.asArray(Int32.self) == expectedFullOff)
            #expect(cachesB4[1].hostPositionOffsets == expectedSlideOff)
            #expect(cachesB4[1].positionOffsets.asArray(Int32.self) == expectedSlideOff)
            for b in 0 ..< B {
                #expect(fullRowsB4[b].absoluteOffset == storedCounts[b])
                #expect(fullRowsB4[b].ropePositionOffset == Int(expectedFullOff[b]))
                #expect(fullRowsB4[b].isCompacted)
                #expect(fullRowsB4[b].prefixPageHandles(tokens: 0 ..< 16) == nil)
                #expect(slideRowsB4[b].absoluteOffset == storedCounts[b])
                #expect(slideRowsB4[b].ropePositionOffset == 128)
                #expect(slideRowsB4[b].isCompacted)
            }

            // Batched decode vs independent B=1 decode on sliding layer.
            let decB4 = cachesB4[1].updateAndAttend(
                queries: qDec, keys: kDec, values: vDec, scale: 0.25, sinks: nil)
            var decB1Rows: [MLXArray] = []
            for b in 0 ..< B {
                let row1 = statesB1[b][1] as! PagedSequenceKV
                cachesB1[1].setRows([row1])
                cachesB1[1].writeCompactedPromptKV(
                    keys: perRowK4D[b], values: perRowV4D[b], logicalDelta: uniformDeltas[b])
                decB1Rows.append(
                    cachesB1[1].updateAndAttend(
                        queries: qDec[b ..< (b + 1)],
                        keys: kDec[b ..< (b + 1)],
                        values: vDec[b ..< (b + 1)],
                        scale: 0.25, sinks: nil))
            }
            let decB1 = concatenated(decB1Rows, axis: 0)
            eval(decB4, decB1)
            let diff = MLX.abs(decB4 - decB1).max().item(Float.self)
            #expect(diff < 1e-4, "Paged B=4 vs B=1 post-compacted sliding decode diff: \(diff)")
            #expect(cachesB4[1].hostPositionOffsets == [129, 129, 129, 129])
        }

        // 2. Contiguous CBv2LayerCache: B=4 per-row vs B=1 parity.
        do {
            let fullRowsB4: [CBv2SequenceKV] = (0 ..< B).map { _ in
                CBv2FullSequenceKV(promptLength: 128, maxLength: 256, kvHeads: 2, headDim: 64)
            }
            let slideRowsB4: [CBv2SequenceKV] = (0 ..< B).map { _ in
                CBv2WindowedSequenceKV(window: 64, kvHeads: 2, headDim: 64)
            }
            let fullCacheB4 = CBv2LayerCache(layerIndex: 0, kind: fullKind, rows: fullRowsB4)
            let slideCacheB4 = CBv2LayerCache(layerIndex: 1, kind: slidingKind, rows: slideRowsB4)

            fullCacheB4.writeCompactedPromptKV(
                perRowKeys: perRowK4D, perRowValues: perRowV4D, logicalDeltas: heteroDeltas)
            slideCacheB4.writeCompactedPromptKV(
                perRowKeys: perRowK3D, perRowValues: perRowV3D, logicalDeltas: uniformDeltas)

            let expectedFullOff = zip(storedCounts, heteroDeltas).map { Int32($0 + $1) }
            eval(fullCacheB4.positionOffsets, slideCacheB4.positionOffsets)
            #expect(fullCacheB4.positionOffsets.asArray(Int32.self) == expectedFullOff)
            #expect(slideCacheB4.positionOffsets.asArray(Int32.self) == [128, 128, 128, 128])
            for b in 0 ..< B {
                #expect(slideRowsB4[b].absoluteOffset == storedCounts[b])
                #expect(slideRowsB4[b].ropePositionOffset == 128)
                #expect(slideRowsB4[b].isCompacted)
            }

            let decB4 = slideCacheB4.updateAndAttend(
                queries: qDec, keys: kDec, values: vDec, scale: 0.25, sinks: nil)
            var decB1Rows: [MLXArray] = []
            for b in 0 ..< B {
                let row1 = CBv2WindowedSequenceKV(window: 64, kvHeads: 2, headDim: 64)
                let c1 = CBv2LayerCache(layerIndex: 1, kind: slidingKind, rows: [row1])
                c1.writeCompactedPromptKV(
                    keys: perRowK4D[b], values: perRowV4D[b], logicalDelta: uniformDeltas[b])
                decB1Rows.append(
                    c1.updateAndAttend(
                        queries: qDec[b ..< (b + 1)],
                        keys: kDec[b ..< (b + 1)],
                        values: vDec[b ..< (b + 1)],
                        scale: 0.25, sinks: nil))
            }
            let decB1 = concatenated(decB1Rows, axis: 0)
            eval(decB4, decB1)
            let diff = MLX.abs(decB4 - decB1).max().item(Float.self)
            #expect(
                diff < 1e-4,
                "Contiguous B=4 vs B=1 post-compacted sliding decode diff: \(diff)")
            eval(slideCacheB4.positionOffsets)
            #expect(slideCacheB4.positionOffsets.asArray(Int32.self) == [129, 129, 129, 129])
        }
    }

    @Test(
        "Batched B=4 forwardBlockSDCMTrunk + Paged/Contiguous KV caches matches B=1 row-for-row across fast and turbo modes"
    )
    func batchedBlockSDCMTrunkParityB4VsB1() throws {
        MLXRandom.seed(9001)
        let model = try Self.makeSyntheticGemma4MoE()
        let B = 4
        let T = 160
        let kinds = model.cbv2LayerKinds

        var rng = CBv2SchedSplitMix64(seed: 0xBA7C_4004)
        let promptRows: [[Int32]] = (0 ..< B).map { b in
            (0 ..< T).map { j in Int32((Int(rng.next() % 200) + b * 17 + j) % model.vocabularySize)
            }
        }
        let flatPrompt = promptRows.flatMap { $0 }
        let tokensB4 = MLXArray(flatPrompt, [B, T])
        let hiddenB4 = model.model.embedTokens(tokensB4) * model.model.embedScale
        let nextTokRows: [Int32] = [13, 27, 41, 55]
        let nextTokB4 = MLXArray(nextTokRows, [B, 1])

        // Test both fast (bilateral L6 compaction) and turbo (L0 early Tri-Signal + L6 bilateral + sparse L5).
        var fastCfg = Gemma4BlockSDCMConfig.fast
        fastCfg.minPromptTokens = 96
        fastCfg.prefixKeep = 16
        fastCfg.tailKeep = 16
        fastCfg.blockSize = 16
        fastCfg.keepSchedule = [6: 0.50]
        fastCfg.reversePrefixQuota = [6: 0.25]
        fastCfg.reversePrefixEnd = 24
        fastCfg.earlySparseFullLayer = 5
        fastCfg.sparsePrefixTokens = 16
        fastCfg.sparseLocalWindow = 32
        fastCfg.sparseTailTokens = 16
        fastCfg.slidingChunkSize = 32
        fastCfg.earlyMoEK = 2
        fastCfg.lateMoEK = 4
        fastCfg.earlyMoEUntilLayer = 6
        fastCfg.layerMoEK = [0: 2, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3]

        var turboCfg = fastCfg
        turboCfg.mode = .turbo
        turboCfg.earlyPreCompactAfterLayer = 0
        turboCfg.earlyPreCompactRatio = 0.65
        turboCfg.earlyPreCompactBlockSize = 16
        turboCfg.earlyPreCompactTopHaloK = 2

        for (label, cfg) in [("fast", fastCfg), ("turbo", turboCfg)] {
            let poolCfg = PagedKVPoolConfig(
                pageSize: 16, capacityBytes: 64 * 1024 * 1024, dtype: .float32,
                maxPrefillChunk: 160, nominalMaxSequenceLength: 256)
            let pagedB4 = try PagedKVBackend(layerKinds: kinds, config: poolCfg)
            var statesB4: [[CBv2SequenceKV?]] = []
            defer { for s in statesB4 { pagedB4.release(s) } }
            for _ in 0 ..< B {
                statesB4.append(
                    try pagedB4.makeSequenceState(
                        layerKinds: kinds, promptLength: T, maxLength: T + 32))
            }
            let pagedCachesB4 = pagedB4.makeLayerCaches()
            let fullCacheB4: [KVCache?] = pagedCachesB4.enumerated().map { layerIdx, cache in
                cache.setRows(statesB4.map { $0[layerIdx]! })
                return cache
            }

            let (postNormB4, ckptB4) = model.model.forwardBlockSDCMTrunk(
                hiddenB4,
                inputTokenIDRows: promptRows,
                fullCache: fullCacheB4,
                sdcm: cfg,
                isExpertPrefill: false)
            #expect(ckptB4 == nil)
            #expect(postNormB4.shape == [B, 1, model.configuration.hiddenSize])
            #expect(!pagedB4.pool.writeValidation.isFaulted)

            let decLogitsB4 = model(nextTokB4, cache: pagedCachesB4)
            eval(postNormB4, decLogitsB4)

            // Run each row independently with B=1 on a fresh PagedKVBackend and assert parity.
            for b in 0 ..< B {
                let pagedB1 = try PagedKVBackend(layerKinds: kinds, config: poolCfg)
                let stateB1 = try pagedB1.makeSequenceState(
                    layerKinds: kinds, promptLength: T, maxLength: T + 32)
                defer { pagedB1.release(stateB1) }
                let pagedCachesB1 = pagedB1.makeLayerCaches()
                let fullCacheB1: [KVCache?] = pagedCachesB1.enumerated().map { layerIdx, cache in
                    cache.setRows([stateB1[layerIdx]!])
                    return cache
                }

                let (postNormB1, _) = model.model.forwardBlockSDCMTrunk(
                    hiddenB4[b ..< (b + 1)],
                    inputTokenIDs: promptRows[b],
                    fullCache: fullCacheB1,
                    sdcm: cfg,
                    isExpertPrefill: false)
                let decLogitsB1 = model(
                    MLXArray([nextTokRows[b]], [1, 1]), cache: pagedCachesB1)
                eval(postNormB1, decLogitsB1)

                let prefillDiff = MLX.abs(postNormB4[b ..< (b + 1)] - postNormB1).max().item(
                    Float.self)
                let decodeDiff = MLX.abs(decLogitsB4[b ..< (b + 1)] - decLogitsB1).max().item(
                    Float.self)
                #expect(
                    prefillDiff < 5e-4,
                    "\(label) row \(b) B=4 vs B=1 prefill postNorm diff: \(prefillDiff)")
                #expect(
                    decodeDiff < 5e-4,
                    "\(label) row \(b) B=4 vs B=1 post-compaction decode diff: \(decodeDiff)")

                for layerIdx in 0 ..< kinds.count {
                    let rowB4 = statesB4[b][layerIdx]!
                    let rowB1 = stateB1[layerIdx]!
                    #expect(rowB4.absoluteOffset == rowB1.absoluteOffset)
                    #expect(rowB4.ropePositionOffset == T + 1)
                    #expect(rowB4.ropePositionOffset == rowB1.ropePositionOffset)
                    #expect(rowB4.isCompacted == rowB1.isCompacted)
                }
            }
        }

        // Verify contiguous CBv2LayerCache matches PagedKVBackend on B=4 too.
        let contigStates: [[CBv2SequenceKV]] = (0 ..< B).map { _ in
            kinds.map { kind in
                switch kind.attention {
                case .full:
                    CBv2FullSequenceKV(
                        promptLength: T, maxLength: T + 32,
                        kvHeads: kind.kvHeads, headDim: kind.headDim)
                case .slidingWindow(let w):
                    CBv2WindowedSequenceKV(window: w, kvHeads: kind.kvHeads, headDim: kind.headDim)
                }
            }
        }
        let contigCaches: [CBv2LayerCache] = kinds.enumerated().map { layerIdx, kind in
            CBv2LayerCache(
                layerIndex: layerIdx, kind: kind, rows: contigStates.map { $0[layerIdx] })
        }
        let (contigPostNorm, _) = model.model.forwardBlockSDCMTrunk(
            hiddenB4,
            inputTokenIDRows: promptRows,
            fullCache: contigCaches.map { $0 as KVCache? },
            sdcm: fastCfg,
            isExpertPrefill: false)
        let contigDec = model(nextTokB4, cache: contigCaches)
        eval(contigPostNorm, contigDec)
        #expect(!MLX.isNaN(contigPostNorm).any().item(Bool.self))
        #expect(!MLX.isNaN(contigDec).any().item(Bool.self))
        for b in 0 ..< B {
            #expect(contigStates[b][11].absoluteOffset < T)
            #expect(contigStates[b][11].ropePositionOffset == T + 1)
            #expect(contigStates[b][11].isCompacted)
        }

        // Verify equalizeBatchedCompactedIndices reconciles unequal surviving lengths cleanly.
        let posHost = (0 ..< 128).map { Int32($0) }
        let unequal = Gemma4BlockSDCM.equalizeBatchedCompactedIndices(
            keepRows: [
                Array(0 ..< 16).map(Int32.init) + Array(112 ..< 128).map(Int32.init),
                Array(0 ..< 32).map(Int32.init) + Array(112 ..< 128).map(Int32.init),
            ],
            rowBlockScores: [
                [0, 1, 5, 2, 3, 4, 0, 0],
                [0, 1, 2, 3, 4, 5, 0, 0],
            ],
            positionsHostRows: [posHost, posHost],
            originalLength: 128,
            blockSize: 16,
            currentLength: 128)
        #expect(unequal[0].count == 48)
        #expect(unequal[1].count == 48)
        #expect(unequal[0].first == 0 && unequal[0].last == 127)
        for i in 1 ..< unequal[0].count { #expect(unequal[0][i] > unequal[0][i - 1]) }
        // Highest-scoring unkept interior block in row 0 is block 2 (positions 32..<48).
        #expect(unequal[0].contains(32) && unequal[0].contains(47))
    }

    @Test(
        "SchedulerV2 Mode A dispatches whole-prompt compaction beside active decode while isolating budget surplus"
    )
    func schedulerModeAConcurrentDecodeAndCompaction() throws {
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 4,
            maxBatchedTokensPerStep: 4096,
            prefillChunkSize: 2048,
            soloPrefillStripeTokens: nil,
            maxConcurrentPartialPrefills: 1,
            enablePrefixCache: true)
        config.soloCompactionMinPromptTokens = 3072
        config.allowConcurrentCompaction = true
        config.maxPackedCompactionBatch = 1
        config.demandedShortCheckpointMinimumTokens = 1024
        config.demandedCheckpointPartitionIncludesLongPrompts = true
        let scheduler = SchedulerV2(config: config)

        // Step 1: admit and finish prefill of short request R1 so it becomes decode-ready.
        let r1 = CBv2Request(
            id: CBv2RequestID(1), promptTokens: Array(repeating: 5, count: 64), maxTokens: 16)
        try scheduler.enqueue(r1)
        let p1 = scheduler.plan()
        #expect(p1.assignments.count == 1 && p1.assignments[0].numTokens == 64)
        scheduler.markPendingSamples(ids: [r1.id])
        scheduler.recordSampled(id: r1.id, token: 99)

        // Enqueue cold long compaction prompt R2 (with demanded fork hint) + short trailing waiter R3.
        var r2 = CBv2Request(
            id: CBv2RequestID(2), promptTokens: Array(repeating: 7, count: 6000),
            maxTokens: 16, cacheSalt: "mode-a")
        r2.prefixCheckpointTargetTokens = 2048
        let r3 = CBv2Request(
            id: CBv2RequestID(3), promptTokens: Array(repeating: 9, count: 500), maxTokens: 16)
        try scheduler.enqueue(r2)
        try scheduler.enqueue(r3)

        // Step 2 (Mode A): R1 decodes (1) AND R2 compacts whole-prompt (6000) bypassing demanded checkpoints;
        // R3 must stay waiting (cannot touch compaction budget surplus).
        let p2 = scheduler.plan()
        #expect(p2.assignments.count == 2)
        #expect(p2.assignments[0].id == r1.id && p2.assignments[0].numTokens == 1)
        #expect(p2.assignments[1].id == r2.id && p2.assignments[1].numTokens == 6000)
        #expect(p2.demandedShortCheckpointRows.isEmpty)
        #expect(scheduler.waiting.map(\.id) == [r3.id])

        // Legacy guard: with allowConcurrentCompaction = false, decode presence disarms compaction.
        var legacyConfig = config
        legacyConfig.allowConcurrentCompaction = false
        let legacyScheduler = SchedulerV2(config: legacyConfig)
        try legacyScheduler.enqueue(r1)
        _ = legacyScheduler.plan()
        legacyScheduler.markPendingSamples(ids: [r1.id])
        legacyScheduler.recordSampled(id: r1.id, token: 99)
        try legacyScheduler.enqueue(r2)
        let legacyPlan = legacyScheduler.plan()
        #expect(legacyPlan.assignments.count == 2)
        #expect(legacyPlan.assignments[0].id == r1.id && legacyPlan.assignments[0].numTokens == 1)
        #expect(
            legacyPlan.assignments[1].id == r2.id && legacyPlan.assignments[1].numTokens == 2048)
    }

    @Test(
        "SchedulerV2 Mode B packs equal-length compaction waves, enforces density guard veto, and preserves partial cohorts on KV exhaustion"
    )
    func schedulerModeBPackedCohortWavesDensityGuardAndKVReserve() throws {
        var rng = CBv2SchedSplitMix64(seed: 0x5C4E_D00B)
        let codePrompt = (0 ..< 4096).map { i in
            (i % 16 < 8) ? (500 + (i % 8)) : Int(600 + rng.next() % 256)
        }
        let proteinPrompt = (0 ..< 4096).map { _ in Int(100 + rng.next() % 20) }

        // 1. Packed cohort waves capped by min(maxPackedCompactionBatch, maxPackedCompactionTokens / targetLen).
        do {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 8,
                maxBatchedTokensPerStep: 4096,
                prefillChunkSize: 2048,
                soloPrefillStripeTokens: nil,
                maxConcurrentPartialPrefills: 1)
            config.soloCompactionMinPromptTokens = 3072
            config.allowConcurrentCompaction = true
            config.maxPackedCompactionBatch = 4
            config.maxPackedCompactionTokens = 16_384
            config.compactionEligibility = { tokens in
                Gemma4BlockSDCM.passesDensityGuard(tokens: tokens, config: .fast)
            }
            let scheduler = SchedulerV2(config: config)
            for i in 1 ... 6 {
                try scheduler.enqueue(
                    CBv2Request(
                        id: CBv2RequestID(UInt64(i)), promptTokens: codePrompt, maxTokens: 16))
            }

            // Wave 1: packs 4 x 4096 = 16,384; leaves 2 waiting.
            let wave1 = scheduler.plan()
            #expect(wave1.assignments.count == 4)
            #expect(wave1.assignments.allSatisfy { $0.numTokens == 4096 })
            #expect(scheduler.waitingCount == 2)

            let wave1IDs = wave1.assignments.map(\.id)
            scheduler.markPendingSamples(ids: wave1IDs)
            for id in wave1IDs { scheduler.recordSampled(id: id, token: 42) }

            // Wave 2: Wave 1 rows decode (4 x 1) concurrently with Wave 2 cohort (2 x 4096).
            let wave2 = scheduler.plan()
            #expect(wave2.assignments.count == 6)
            for i in 0 ..< 4 {
                #expect(wave2.assignments[i].numTokens == 1)
            }
            for i in 4 ..< 6 {
                #expect(wave2.assignments[i].numTokens == 4096)
            }
            #expect(scheduler.waitingCount == 0)
        }

        // 2. Density guard veto stops compaction cohort cleanly at vetoed prompt without jumping FCFS.
        do {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 8,
                maxBatchedTokensPerStep: 4096,
                prefillChunkSize: 2048,
                soloPrefillStripeTokens: nil,
                maxConcurrentPartialPrefills: 1)
            config.soloCompactionMinPromptTokens = 3072
            config.allowConcurrentCompaction = true
            config.maxPackedCompactionBatch = 4
            config.maxPackedCompactionTokens = 32_768
            config.compactionEligibility = { tokens in
                Gemma4BlockSDCM.passesDensityGuard(tokens: tokens, config: .fast)
            }
            let scheduler = SchedulerV2(config: config)
            try scheduler.enqueue(
                CBv2Request(id: CBv2RequestID(10), promptTokens: codePrompt, maxTokens: 8))
            try scheduler.enqueue(
                CBv2Request(id: CBv2RequestID(11), promptTokens: codePrompt, maxTokens: 8))
            try scheduler.enqueue(
                CBv2Request(id: CBv2RequestID(12), promptTokens: proteinPrompt, maxTokens: 8))
            try scheduler.enqueue(
                CBv2Request(id: CBv2RequestID(13), promptTokens: codePrompt, maxTokens: 8))

            // Packs only R10 & R11; stops before vetoed R12 (protein) and preserves FCFS.
            let p1 = scheduler.plan()
            #expect(p1.assignments.map(\.id) == [CBv2RequestID(10), CBv2RequestID(11)])
            #expect(p1.assignments.allSatisfy { $0.numTokens == 4096 })
            #expect(scheduler.waiting.map(\.id) == [CBv2RequestID(12), CBv2RequestID(13)])

            scheduler.markPendingSamples(ids: [CBv2RequestID(10), CBv2RequestID(11)])
            scheduler.recordSampled(id: CBv2RequestID(10), token: 1)
            scheduler.recordSampled(id: CBv2RequestID(11), token: 1)

            // Next step: vetoed R12 gets plain prefillChunkSize (2048), NOT whole-prompt 4096.
            let p2 = scheduler.plan()
            #expect(
                p2.assignments.map(\.id) == [
                    CBv2RequestID(10), CBv2RequestID(11), CBv2RequestID(12),
                ])
            #expect(p2.assignments[2].numTokens == 2048)
        }

        // 3. KV reservation failure: multi-row cohort breaks leaving already-reserved rows intact
        //    (never shrinks R3 even though prefillChunkSize=2048 would fit); 1-row cohort DOES shrink.
        do {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 8,
                maxBatchedTokensPerStep: 4096,
                prefillChunkSize: 2048,
                soloPrefillStripeTokens: nil,
                maxConcurrentPartialPrefills: 1)
            config.soloCompactionMinPromptTokens = 3072
            config.allowConcurrentCompaction = true
            config.maxPackedCompactionBatch = 4
            config.maxPackedCompactionTokens = 32_768

            // Capacity 10,500 holds R1(4096) + R2(4096) = 8192, plus 2308 spare (>= 2048).
            // R3 MUST NOT shrink to 2048 because cohort size is 4 > 1.
            let capMulti = CBv2SchedMockCapacity(tokenLimit: 10_500)
            let schedMulti = SchedulerV2(config: config, capacity: capMulti)
            for i in 1 ... 4 {
                try schedMulti.enqueue(
                    CBv2Request(
                        id: CBv2RequestID(UInt64(i)), promptTokens: codePrompt, maxTokens: 8))
            }
            let partial = schedMulti.plan()
            #expect(partial.assignments.map(\.id) == [CBv2RequestID(1), CBv2RequestID(2)])
            #expect(partial.assignments.allSatisfy { $0.numTokens == 4096 })
            #expect(schedMulti.waiting.map(\.id) == [CBv2RequestID(3), CBv2RequestID(4)])
            #expect(schedMulti.waiting.allSatisfy { $0.numComputedTokens == 0 })

            // Single-row cohort (1 waiter) with capacity 2500 DOES shrink to prefillChunkSize=2048.
            let capSolo = CBv2SchedMockCapacity(tokenLimit: 2500)
            let schedSolo = SchedulerV2(config: config, capacity: capSolo)
            try schedSolo.enqueue(
                CBv2Request(id: CBv2RequestID(20), promptTokens: codePrompt, maxTokens: 8))
            let shrunk = schedSolo.plan()
            #expect(shrunk.assignments.count == 1)
            #expect(shrunk.assignments[0].id == CBv2RequestID(20))
            #expect(shrunk.assignments[0].numTokens == 2048)

            // Multi-row cohort when ZERO cohort rows fit full targetLen (capacity 2500 < 4096):
            // row 0 shrinks once to prefillChunkSize=2048 instead of livelocking forever, and
            // stops the remaining cohort rows cleanly.
            let capTightMulti = CBv2SchedMockCapacity(tokenLimit: 2500)
            let schedTightMulti = SchedulerV2(config: config, capacity: capTightMulti)
            try schedTightMulti.enqueue(
                CBv2Request(id: CBv2RequestID(30), promptTokens: codePrompt, maxTokens: 8))
            try schedTightMulti.enqueue(
                CBv2Request(id: CBv2RequestID(31), promptTokens: codePrompt, maxTokens: 8))
            let noLivelock = schedTightMulti.plan()
            #expect(noLivelock.assignments.count == 1)
            #expect(noLivelock.assignments[0].id == CBv2RequestID(30))
            #expect(noLivelock.assignments[0].numTokens == 2048)
            #expect(schedTightMulti.waiting.map(\.id) == [CBv2RequestID(31)])
        }
    }
}
