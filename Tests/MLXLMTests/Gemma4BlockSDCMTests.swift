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
        #expect(lcDefault.sparseFullLayers.isEmpty)
        #expect(lcDefault.usesSparseFullAttention(forLayer: 5))
        #expect(!lcDefault.usesSparseFullAttention(forLayer: 11))
        #expect(lcDefault.adaptiveRetentionFloorWindows == 2.0)

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
        #expect(bal.adaptiveRetentionFloorWindows == 2.0)

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
        #expect(fast.adaptiveRetentionFloorWindows == 0)

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
            #expect(turbo.adaptiveRetentionFloorWindows == 0)
        }

        for alias in ["coherent", "fullkv", "full", " FULLKV "] {
            #expect(Gemma4BlockSDCMConfig.parseMode(alias) == .coherent)
            let coherent = Gemma4BlockSDCMConfig.resolve(environment: [
                Gemma4BlockSDCMConfig.envKey: alias
            ])
            #expect(coherent.mode == .coherent)
            #expect(coherent.minPromptTokens == 1536)
            #expect(coherent.keepSchedule.isEmpty)
            #expect(coherent.earlyMoEK == 2)
            #expect(coherent.lateMoEK == 2)
            #expect(coherent.earlyMoEUntilLayer == 29)
            #expect(coherent.layerMoEK.isEmpty)
            #expect(coherent.interiorMoEK(forLayer: 0) == 2)
            #expect(coherent.interiorMoEK(forLayer: 28) == 2)
            #expect(coherent.interiorMoEK(forLayer: 29) == 2)
            #expect(coherent.blockSize == 64)
            #expect(coherent.prefixKeep == 128)
            #expect(coherent.tailKeep == 1024)
            #expect(coherent.earlySparseFullLayer == nil)
            #expect(coherent.sparseFullLayers == [5, 11, 17])
            #expect(coherent.usesSparseFullAttention(forLayer: 5))
            #expect(coherent.usesSparseFullAttention(forLayer: 11))
            #expect(coherent.usesSparseFullAttention(forLayer: 17))
            #expect(!coherent.usesSparseFullAttention(forLayer: 23))
            #expect(!coherent.usesSparseFullAttention(forLayer: 29))
            #expect(coherent.sparsePrefixTokens == 256)
            #expect(coherent.sparseLocalWindow == 1024)
            #expect(coherent.sparseTailTokens == 1024)
            #expect(coherent.skipInteriorSlidingLayers == [24, 25, 26, 27])
            #expect(!coherent.skipInteriorSlidingLayers.contains(28))
            #expect(coherent.adaptiveRetentionFloorWindows == 0)
            #expect(coherent.hasActiveOptimizations)
        }

        // Clearing `earlySparseFullLayer` on presets like `.balanced` disables sparse L5.
        var balancedNoSparse = Gemma4BlockSDCMConfig.balanced
        #expect(balancedNoSparse.usesSparseFullAttention(forLayer: 5))
        balancedNoSparse.earlySparseFullLayer = nil
        #expect(!balancedNoSparse.usesSparseFullAttention(forLayer: 5))

        let coherentOverride = Gemma4BlockSDCMConfig.resolve(environment: [
            Gemma4BlockSDCMConfig.envKey: "coherent",
            Gemma4BlockSDCMConfig.earlyMoEKEnvKey: "4",
        ])
        #expect(coherentOverride.earlyMoEK == 4)
        #expect(coherentOverride.lateMoEK == 4)
        #expect(coherentOverride.interiorMoEK(forLayer: 2) == 4)
        #expect(coherentOverride.interiorMoEK(forLayer: 29) == 4)

        #expect(!Gemma4BlockSDCMConfig.off.hasActiveOptimizations)
        let noOpConfig = Gemma4BlockSDCMConfig(
            mode: .coherent,
            keepSchedule: [:],
            earlyMoEK: 8,
            lateMoEK: 8,
            earlySparseFullLayer: nil,
            sparseFullLayers: [],
            skipInteriorSlidingLayers: []
        )
        #expect(!noOpConfig.hasActiveOptimizations)
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

    private func compactedSlidingPrompt() -> (length: Int, window: Int, stored: [Int32]) {
        let length = 259  // Only three tokens in the final 8-token block.
        let window = 64
        var cfg = Gemma4BlockSDCMConfig.fast
        cfg.blockSize = 8
        cfg.prefixKeep = 8
        cfg.tailKeep = 8
        cfg.slidingWindowToRetain = window
        cfg.strataCount = 0
        cfg.forwardHalo = 0
        cfg.backwardHalo = 0

        var positions = (0 ..< length).map { Int32($0) }
        for fraction: Float in [0.55, 0.12] {
            // Favor 200..<208 and distant prefix blocks over the rest of the window.
            let scores: [Float] = positions.map { $0 / 8 == 25 ? 1000 : Float(length) - Float($0) }
            let kept = Gemma4BlockSDCM.selectCompactedIndices(
                scores: scores, positions: positions, originalLength: length,
                targetFraction: fraction, config: cfg)
            #expect(kept.count < positions.count)
            positions = kept.map { positions[Int($0)] }
            #expect(Array(positions.suffix(window)) == (195 ..< 259).map { Int32($0) })
        }

        let start = Gemma4BlockSDCM.trueSlidingWindowStart(
            positionsHost: positions, originalLength: length, slidingWindow: window)
        return (length, window, Array(positions[start...]))
    }

    @Test("Repeated pruning preserves exact sliding-ring visibility through decode and wraparound")
    func compactedSlidingRingVisibility() {
        let (length, window, stored) = compactedSlidingPrompt()
        let tags = MLXArray(stored.map { Float($0) }).reshaped(1, 1, -1, 1)
        let row = CBv2WindowedSequenceKV(window: window, kvHeads: 1, headDim: 1)
        _ = row.update(keys: tags, values: tags)
        row.addPositionDelta(length - tags.dim(2))
        for q in length ... length + window {
            #expect(row.ropePositionOffset == q)
            let token = MLXArray([Float(q)]).reshaped(1, 1, 1, 1)
            let (_, visible) = row.update(keys: token, values: token)
            #expect(visible.asArray(Float.self) == (q - window + 1 ... q).map { Float($0) })
        }
    }

    @Test("Paged decode sees exactly the original-position window after compaction and page reuse")
    func pagedCompactedSlidingRingVisibility() throws {
        let (length, window, stored) = compactedSlidingPrompt()
        let kind = CBv2LayerKind(
            attention: .slidingWindow(window), headDim: window, kvHeads: 1, queryHeads: 1)
        let config = PagedKVPoolConfig(
            pageSize: 16, capacityBytes: 8 << 20, dtype: .float32, maxPrefillChunk: 16)
        let backend = try PagedKVBackend(layerKinds: [kind], config: config)
        let ringTokens = PagedKVPool.ringPageCount(window: window, config: config) * config.pageSize
        let lastQuery = length + ringTokens
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: length, maxLength: lastQuery + 1)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([row])

        // Each dense window has one position per lane. The absolute tag distinguishes generations.
        func taggedKV(_ positions: [Int32]) -> MLXArray {
            var tags = [Float](repeating: 0, count: positions.count * window)
            for (index, position) in positions.enumerated() {
                tags[index * window + Int(position) % window] = Float(position)
            }
            return MLXArray(tags, [1, 1, positions.count, window])
        }

        let prompt = taggedKV(stored)
        cache.writeCompactedPromptKV(
            keys: prompt, values: prompt, logicalDelta: length - stored.count)
        try #require(!backend.pool.writeValidation.isFaulted)
        let zeroQuery = MLXArray.zeros([1, 1, 1, window], dtype: .float32)
        for q in length ... lastQuery {
            let token = taggedKV([Int32(q)])
            let output = cache.updateAndAttend(
                queries: zeroQuery, keys: token, values: token, scale: 1, sinks: nil)
            try #require(!backend.pool.writeValidation.isFaulted)
            try #require(output.shape == [1, 1, 1, window])

            // A zero query weights the dense window uniformly; each lane identifies its contributor.
            let dense = (q - window + 1) ... q
            var expected = [Float](repeating: 0, count: window)
            for p in dense { expected[p % window] = Float(p) / Float(window) }
            let error = MLX.abs(output - MLXArray(expected)).max().item(Float.self)
            #expect(error < 1e-5, "Paged attention read the wrong positions at query \(q)")
            let visible = row.snapshot().keys.sum(axis: -1).asArray(Float.self)
            #expect(visible == dense.map { Float($0) }, "Paged key order at query \(q)")
            #expect(row.ropePositionOffset == q + 1)
        }
        #expect(row.absoluteOffset > ringTokens)
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
            blockSize: 8,
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
            #expect(rows[6].absoluteOffset == 47)  // All W-1 prompt keys for the first decode.

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

        // Turn 2 multi-token prompt (n > 1) after filling the compacted sliding ring.
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
        var ringCfg = cfg
        ringCfg.slidingWindowToRetain = 96
        ringCfg.strataCount = 0
        let keptConst = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: constTok, embeddingHidden: constEmb, layerHidden: constLay,
            targetFraction: 0.10, blockSize: bs, topHaloK: 2, config: ringCfg)
        #expect(keptConst.first == 0 && keptConst.last == Int32(T - 1))
        #expect(keptConst.count < T)
        #expect(Array(keptConst.suffix(96)) == (277 ..< 373).map { Int32($0) })

        // 4. topHaloK far exceeding needInteriorBlocks must stay bounded by needInteriorBlocks and compact < T.
        let keptHugeHalo = Gemma4BlockSDCM.selectEarlyTriSignalIndices(
            tokenIDs: constTok, embeddingHidden: embRand, layerHidden: layRand,
            targetFraction: 0.50, blockSize: bs, topHaloK: 100, config: cfg)
        let keptHugeHaloBlocks = Set(keptHugeHalo.map { Int($0) / bs })
        // The last 32 tokens straddle blocks 10 and 11. With block 0 pinned, 4 interior blocks fit.
        #expect(Set([0, 10, 11]).isSubset(of: keptHugeHaloBlocks))
        #expect(keptHugeHaloBlocks.count == 3 + 4)
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

        let T = 259
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
            #expect(rows[1].absoluteOffset == 47)  // After turbo's early prune.
            #expect(rows[6].absoluteOffset == 47)  // After the subsequent bilateral prune.
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

    // MARK: - Hardening Regressions (Partial Chunks, Cache Roles, Capabilities, Request Veto)

    private static func makeHardeningSDCMConfig() -> Gemma4BlockSDCMConfig {
        Gemma4BlockSDCMConfig(
            mode: .fast,
            minPromptTokens: 96,
            keepSchedule: [6: 0.50],
            earlyMoEK: 2,
            lateMoEK: 4,
            earlyMoEUntilLayer: 6,
            blockSize: 8,
            prefixKeep: 16,
            tailKeep: 16,
            forwardHalo: 0.85,
            backwardHalo: 0.50,
            strataCount: 2,
            twoHopBlockPeaks: 4,
            slidingChunkSize: 32
        )
    }

    private static func makeContiguousCBv2Caches(
        kinds: [CBv2LayerKind], promptLength: Int, maxLength: Int = 256
    ) -> (rows: [CBv2SequenceKV], caches: [KVCache]) {
        let rows: [CBv2SequenceKV] = kinds.map { kind in
            switch kind.attention {
            case .full:
                CBv2FullSequenceKV(
                    promptLength: promptLength, maxLength: maxLength,
                    kvHeads: kind.kvHeads, headDim: kind.headDim)
            case .slidingWindow(let w):
                CBv2WindowedSequenceKV(
                    window: w, kvHeads: kind.kvHeads, headDim: kind.headDim)
            }
        }
        let caches: [KVCache] = kinds.enumerated().map { idx, kind in
            let cache = CBv2LayerCache(layerIndex: idx, kind: kind)
            cache.setRows([rows[idx]])
            return cache
        }
        return (rows, caches)
    }

    @Test("CBv2 .evaluationOnly intermediate prefill chunks bypass Block-SDCM")
    func evaluationOnlyPrefillBypassesBlockSDCM() throws {
        MLXRandom.seed(701)
        let model = try Self.makeSyntheticGemma4MoE()
        let sdcm = Self.makeHardeningSDCMConfig()
        model.blockSDCMConfig = sdcm

        let offModel = try Self.makeSyntheticGemma4MoE()
        try offModel.update(parameters: model.parameters(), verify: .all)
        offModel.blockSDCMConfig = .off

        #expect(
            !Gemma4BlockSDCM.canRun(
                sdcm: sdcm, modelConfig: model.configuration, schedulePrefill: true,
                allowsBlockSDCM: false, batchSize: 1, sequenceLength: 128,
                hasInputEmbedding: false, hasImageMask: false, hasCaptureHook: false,
                initialOffset: 0))
        #expect(
            Gemma4BlockSDCM.canRun(
                sdcm: sdcm, modelConfig: model.configuration, schedulePrefill: true,
                allowsBlockSDCM: true, batchSize: 1, sequenceLength: 128,
                hasInputEmbedding: false, hasImageMask: false, hasCaptureHook: false,
                initialOffset: 0))

        let promptLen = 160
        let firstChunkLen = 128  // >= minPromptTokens(96), cold offset 0
        let prompt = MLXArray((0 ..< promptLen).map { Int32(($0 * 7 + 3) % 200) })
            .expandedDimensions(axis: 0)
        let chunk1 = prompt[0..., 0 ..< firstChunkLen]
        let chunk2 = prompt[0..., firstChunkLen ..< promptLen]

        let kinds = model.cbv2LayerKinds
        let (onRows, onCaches) = Self.makeContiguousCBv2Caches(
            kinds: kinds, promptLength: promptLen)
        let (offRows, offCaches) = Self.makeContiguousCBv2Caches(
            kinds: kinds, promptLength: promptLen)

        let handleOn = model.cbv2Prefill(
            chunk1, inputEmbedding: nil, cache: onCaches, requirement: .evaluationOnly)
        let handleOff = offModel.cbv2Prefill(
            chunk1, inputEmbedding: nil, cache: offCaches, requirement: .evaluationOnly)
        eval(handleOn, handleOff)
        #expect(handleOn.shape == [1, 1])

        // Intermediate chunk must keep exact KV on every layer.
        for idx in 0 ..< kinds.count {
            #expect(!onRows[idx].isCompacted)
            #expect(!onRows[idx].usedApproximatePrefill)
            #expect(onRows[idx].ropePositionOffset == firstChunkLen)
            #expect(onRows[idx].absoluteOffset == offRows[idx].absoluteOffset)
        }
        #expect(onRows[5].absoluteOffset == firstChunkLen)
        #expect(onRows[11].absoluteOffset == firstChunkLen)

        let logitsOn = model.cbv2Prefill(
            chunk2, inputEmbedding: nil, cache: onCaches, requirement: .lastPositionLogits)
        let logitsOff = offModel.cbv2Prefill(
            chunk2, inputEmbedding: nil, cache: offCaches, requirement: .lastPositionLogits)
        eval(logitsOn, logitsOff)
        #expect(MLX.abs(logitsOn - logitsOff).max().item(Float.self) < 1e-4)
        #expect(onRows[11].absoluteOffset == promptLen)
        #expect(!onRows[11].isCompacted)
        #expect(!onRows[11].usedApproximatePrefill)
    }

    @Test("Standalone cache validation rejects capped full RotatingKVCache and mismatched sliding caches")
    func standaloneCappedFullCacheRejectedByBlockSDCM() throws {
        MLXRandom.seed(702)
        let model = try Self.makeSyntheticGemma4MoE()
        model.blockSDCMConfig = Self.makeHardeningSDCMConfig()

        let offModel = try Self.makeSyntheticGemma4MoE()
        try offModel.update(parameters: model.parameters(), verify: .all)
        offModel.blockSDCMConfig = .off

        let window = model.configuration.slidingWindow
        // Full-attention layers accept unbounded KVCacheSimple only.
        #expect(Gemma4BlockSDCM.supportsCache(StandardKVCache(), isSliding: false, slidingWindow: window))
        #expect(
            !Gemma4BlockSDCM.supportsCache(
                RotatingKVCache(maxSize: 64, keep: 4), isSliding: false, slidingWindow: window))
        #expect(
            !Gemma4BlockSDCM.supportsCache(
                RotatingKVCache(maxSize: window, keep: 0), isSliding: false, slidingWindow: window))
        #expect(
            !Gemma4BlockSDCM.supportsCache(
                ChunkedKVCache(chunkSize: 64), isSliding: false, slidingWindow: window))

        // Sliding-attention layers accept RotatingKVCache(maxSize == window, keep == 0) only.
        #expect(
            Gemma4BlockSDCM.supportsCache(
                RotatingKVCache(maxSize: window, keep: 0), isSliding: true, slidingWindow: window))
        #expect(
            !Gemma4BlockSDCM.supportsCache(
                RotatingKVCache(maxSize: window, keep: 4), isSliding: true, slidingWindow: window))
        #expect(
            !Gemma4BlockSDCM.supportsCache(
                RotatingKVCache(maxSize: window + 16, keep: 0), isSliding: true,
                slidingWindow: window))
        #expect(
            !Gemma4BlockSDCM.supportsCache(
                CompilableRotatingKVCache(maxSize: window, keep: 0), isSliding: true,
                slidingWindow: window))
        #expect(!Gemma4BlockSDCM.supportsCache(KVCacheSimple(), isSliding: true, slidingWindow: window))

        let cappedParams = GenerateParameters(maxKVSize: 64)
        let cappedCaches = model.newCache(parameters: cappedParams)
        #expect(
            !Gemma4BlockSDCM.supportsCaches(
                cappedCaches.map { $0 as KVCache? }, modelConfig: model.configuration))

        let promptLen = 128
        let prompt1D = MLXArray((0 ..< promptLen).map { Int32(($0 * 5 + 1) % 200) })
        let prompt = prompt1D.expandedDimensions(axis: 0)

        // prepare() must bypass Block-SDCM when full layers are capped with RotatingKVCache.
        let prepResult = try model.prepare(
            LMInput(tokens: prompt1D), cache: cappedCaches, state: nil,
            prefill: PrefillParameters(stepSize: 512))
        guard case .tokens(let remaining) = prepResult else {
            Issue.record("Expected .tokens from prepare on capped full cache")
            return
        }
        #expect(remaining.tokens.size == promptLen)
        #expect(cappedCaches.allSatisfy { $0.offset == 0 })

        // Direct cbv2Prefill on capped caches must also stay on the exact path.
        let cappedOffCaches = offModel.newCache(parameters: cappedParams)
        let logitsOn = model.cbv2Prefill(
            prompt, inputEmbedding: nil, cache: cappedCaches, requirement: .lastPositionLogits)
        let logitsOff = offModel.cbv2Prefill(
            prompt, inputEmbedding: nil, cache: cappedOffCaches, requirement: .lastPositionLogits)
        eval(logitsOn, logitsOff)
        #expect(MLX.abs(logitsOn - logitsOff).max().item(Float.self) < 1e-4)
    }

    @Test("Default CBv2AttendingLayerCache conformer and step-owned PagedLayerCache are rejected before mutating any layer")
    func incapableCBv2AndStepOwnedPagedCachesRejectedBeforeMutation() throws {
        MLXRandom.seed(703)
        let model = try Self.makeSyntheticGemma4MoE()
        model.blockSDCMConfig = Self.makeHardeningSDCMConfig()

        let offModel = try Self.makeSyntheticGemma4MoE()
        try offModel.update(parameters: model.parameters(), verify: .all)
        offModel.blockSDCMConfig = .off

        let kinds = model.cbv2LayerKinds
        let sharedKind = CBv2LayerKind(
            attention: .full, sharesKVWithLayer: 0, headDim: 128, kvHeads: 2, queryHeads: 4)
        let sharedKinds = [kinds[5], sharedKind]

        // Capability property checks across cache types and backends.
        #expect(CBv2LayerCache(layerIndex: 5, kind: kinds[5]).supportsCompactedPromptKVWrite)
        #expect(!CBv2LayerCache(layerIndex: 1, kind: sharedKind).supportsCompactedPromptKVWrite)
        #expect(
            CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 24, kvDType: .float32))
                .supportsCompactedPromptKVWrite)
        #expect(CBv2LayerCacheBank(layerKinds: kinds).supportsCompactedPromptKVWrite)
        #expect(!CBv2LayerCacheBank(layerKinds: sharedKinds).supportsCompactedPromptKVWrite)

        let normalPaged = try PagedKVBackend(
            layerKinds: sharedKinds,
            config: PagedKVPoolConfig(
                pageSize: 16, capacityBytes: 16 * 1024 * 1024, dtype: .float32,
                maxPrefillChunk: 128))
        let normalPagedCaches = normalPaged.makeLayerCaches()
        #expect(normalPaged.supportsCompactedPromptKVWrite)
        #expect(normalPagedCaches[0].supportsCompactedPromptKVWrite)
        #expect(!normalPagedCaches[1].supportsCompactedPromptKVWrite)
        #expect(!CBv2LayerCacheBank(caches: normalPagedCaches).supportsCompactedPromptKVWrite)

        let stepLimits = try CBv2PagedGatheredAttentionLimits(
            maximumBatchSize: 1, maximumQueryTokens: 128, maximumContextTokens: 256,
            maximumInFlightGraphs: 2, maximumScratchBytes: 64 * 1024 * 1024,
            admissionMode: .stepOwned(.pinnedMetal))
        let stepBackend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                pageSize: 16, capacityBytes: 64 * 1024 * 1024, dtype: .float32,
                maxPrefillChunk: 128, gatheredAttention: stepLimits))
        let stepCaches = stepBackend.makeLayerCaches()
        #expect(!stepBackend.supportsCompactedPromptKVWrite)
        #expect(stepCaches.allSatisfy { !$0.supportsCompactedPromptKVWrite })
        #expect(!CBv2LayerCacheBank(caches: stepCaches).supportsCompactedPromptKVWrite)
        #expect(
            !Gemma4BlockSDCM.supportsCaches(
                stepCaches.map { $0 as KVCache? }, modelConfig: model.configuration))

        // 1. Default CBv2AttendingLayerCache conformer at the final layer must reject SDCM
        // before mutating layers 0..<11 and fall back cleanly to exact attention.
        let promptLen = 128
        let prompt1D = MLXArray((0 ..< promptLen).map { Int32(($0 * 13 + 7) % 200) })
        let prompt = prompt1D.expandedDimensions(axis: 0)
        let (rows, cbv2Caches) = Self.makeContiguousCBv2Caches(
            kinds: kinds, promptLength: promptLen)
        let defaultLast = DefaultAttendingLayerCache(
            layerIndex: kinds.count - 1, kind: kinds[kinds.count - 1])
        defaultLast.setRows([rows[kinds.count - 1]])
        #expect(!defaultLast.supportsCompactedPromptKVWrite)

        var mixedCaches: [KVCache] = cbv2Caches
        mixedCaches[kinds.count - 1] = defaultLast
        #expect(
            !Gemma4BlockSDCM.supportsCaches(
                mixedCaches.map { $0 as KVCache? }, modelConfig: model.configuration))
        #expect(
            !CBv2LayerCacheBank(
                caches: mixedCaches.map { $0 as! any CBv2AttendingLayerCache }
            ).supportsCompactedPromptKVWrite)

        let (offRows, offCaches) = Self.makeContiguousCBv2Caches(
            kinds: kinds, promptLength: promptLen)
        let logitsMixed = model.cbv2Prefill(
            prompt, inputEmbedding: nil, cache: mixedCaches, requirement: .lastPositionLogits)
        let logitsOff = offModel.cbv2Prefill(
            prompt, inputEmbedding: nil, cache: offCaches, requirement: .lastPositionLogits)
        eval(logitsMixed, logitsOff)
        #expect(MLX.abs(logitsMixed - logitsOff).max().item(Float.self) < 1e-4)
        for idx in 0 ..< kinds.count {
            #expect(!rows[idx].isCompacted)
            #expect(!rows[idx].usedApproximatePrefill)
            #expect(rows[idx].absoluteOffset == offRows[idx].absoluteOffset)
            #expect(rows[idx].ropePositionOffset == promptLen)
        }

        // 2. Step-owned PagedLayerCache (all-step or final-layer only) must reject SDCM
        // cleanly before mutating any layer or faulting write validation.
        let (rows2, cbv2Caches2) = Self.makeContiguousCBv2Caches(
            kinds: kinds, promptLength: promptLen)
        let stepState = try stepBackend.makeSequenceState(
            layerKinds: kinds, promptLength: promptLen, maxLength: 256)
        defer { stepBackend.release(stepState) }
        for idx in 0 ..< kinds.count {
            stepCaches[idx].setRows([stepState[idx]!])
        }
        var mixedStepCaches: [KVCache] = cbv2Caches2
        mixedStepCaches[kinds.count - 1] = stepCaches[kinds.count - 1]
        #expect(
            !Gemma4BlockSDCM.supportsCaches(
                mixedStepCaches.map { $0 as KVCache? }, modelConfig: model.configuration))

        let prepAllStep = try model.prepare(
            LMInput(tokens: prompt1D), cache: stepCaches, state: nil,
            prefill: PrefillParameters(stepSize: 512))
        guard case .tokens(let remainingAllStep) = prepAllStep else {
            Issue.record("Expected .tokens from prepare on step-owned paged caches")
            return
        }
        #expect(remainingAllStep.tokens.size == promptLen)
        #expect(!stepBackend.pool.writeValidation.isFaulted)
        for idx in 0 ..< kinds.count {
            #expect(stepState[idx]!.absoluteOffset == 0)
            #expect(!stepState[idx]!.isCompacted)
            #expect(!stepState[idx]!.usedApproximatePrefill)
        }

        let prepMixedStep = try model.prepare(
            LMInput(tokens: prompt1D), cache: mixedStepCaches, state: nil,
            prefill: PrefillParameters(stepSize: 512))
        guard case .tokens(let remainingMixedStep) = prepMixedStep else {
            Issue.record("Expected .tokens from prepare on mixed step-owned caches")
            return
        }
        #expect(remainingMixedStep.tokens.size == promptLen)
        #expect(!stepBackend.pool.writeValidation.isFaulted)
        for idx in 0 ..< (kinds.count - 1) {
            #expect(rows2[idx].absoluteOffset == 0)
            #expect(!rows2[idx].isCompacted)
            #expect(!rows2[idx].usedApproximatePrefill)
        }
        #expect(stepState[kinds.count - 1]!.absoluteOffset == 0)
        #expect(!stepState[kinds.count - 1]!.isCompacted)
        #expect(!stepState[kinds.count - 1]!.usedApproximatePrefill)
    }

    @Test("Incapable backend or cache provider never widens solo compaction chunks past maxPrefillChunk")
    func incapableBackendOrCacheProviderNeverWidensSoloCompactionChunk() async throws {
        MLXRandom.seed(705)
        let sdcmModel = try Self.makeSyntheticGemma4MoE()
        sdcmModel.blockSDCMConfig = Self.makeHardeningSDCMConfig()

        let offModel = try Self.makeSyntheticGemma4MoE()
        try offModel.update(parameters: sdcmModel.parameters(), verify: .all)
        offModel.blockSDCMConfig = .off
        let kinds = sdcmModel.cbv2LayerKinds

        // 1. Direct SchedulerV2.plan(compactionSupported: false) refuses widening and keeps
        // demanded checkpoint boundaries intact.
        do {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 4,
                maxBatchedTokensPerStep: 4096,
                prefillChunkSize: 2048,
                soloPrefillStripeTokens: nil,
                enablePrefixCache: true)
            config.soloCompactionMinPromptTokens = 1536
            config.demandedLatestCheckpointMinimumTokens = 1024

            let schedulerLong = SchedulerV2(config: config)
            try schedulerLong.enqueue(
                CBv2Request(
                    id: CBv2RequestID(31), promptTokens: Array(repeating: 1, count: 6000),
                    maxTokens: 16))
            let planLong = schedulerLong.plan(compactionSupported: false)
            #expect(planLong.assignments.count == 1)
            #expect(planLong.assignments[0].numTokens == 2048)

            let schedulerLatest = SchedulerV2(config: config)
            try schedulerLatest.enqueue(
                CBv2Request(
                    id: CBv2RequestID(32), promptTokens: Array(repeating: 1, count: 1800),
                    maxTokens: 16, cacheSalt: "salt-incapable"))
            let planLatest = schedulerLatest.plan(compactionSupported: false)
            #expect(planLatest.assignments.count == 1)
            #expect(planLatest.assignments[0].numTokens == 1536)
            #expect(!planLatest.demandedShortCheckpointRows.isEmpty)
        }

        // 2. Step-owned PagedKVBackend clears soloCompactionMinPromptTokens in EngineV2 and
        // disarms soloCompactionSupported in direct EngineLoopV2 construction.
        do {
            let stepLimits = try CBv2PagedGatheredAttentionLimits(
                maximumBatchSize: 1, maximumQueryTokens: 64, maximumContextTokens: 256,
                maximumInFlightGraphs: 2, maximumScratchBytes: 64 * 1024 * 1024,
                admissionMode: .stepOwned(.pinnedMetal))
            let stepBackend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 64 * 1024 * 1024, dtype: .float32,
                    maxPrefillChunk: 64, gatheredAttention: stepLimits))
            let stepBank = CBv2LayerCacheBank(caches: stepBackend.makeLayerCaches())
            #expect(!stepBackend.supportsCompactedPromptKVWrite)
            #expect(!stepBank.supportsCompactedPromptKVWrite)

            var schedConfig = CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 64,
                prefillChunkSize: 64,
                soloPrefillStripeTokens: nil)
            schedConfig.soloCompactionMinPromptTokens = 96

            let stepEngine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(sdcmModel),
                layerKinds: kinds,
                backend: stepBackend,
                cacheProvider: stepBank,
                sampler: CBv2DefaultSampler(fallbackSeed: 23),
                schedulerConfig: schedConfig)
            #expect(stepEngine.schedulerConfig.soloCompactionMinPromptTokens == nil)
            await stepEngine.shutdown()

            // A capable backend with an `.off` model must also disarm solo compaction.
            let capableBackend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 64 * 1024 * 1024, dtype: .float32,
                    maxPrefillChunk: 64))
            let capableBank = CBv2LayerCacheBank(caches: capableBackend.makeLayerCaches())
            let offEngine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(offModel),
                layerKinds: kinds,
                backend: capableBackend,
                cacheProvider: capableBank,
                sampler: CBv2DefaultSampler(fallbackSeed: 23),
                schedulerConfig: schedConfig)
            #expect(offEngine.schedulerConfig.soloCompactionMinPromptTokens == nil)
            await offEngine.shutdown()

            // Direct EngineLoopV2 with a pre-built SchedulerV2 still carrying
            // soloCompactionMinPromptTokens must disarm soloCompactionSupported before plan().
            let directStepBackend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 64 * 1024 * 1024, dtype: .float32,
                    maxPrefillChunk: 64, gatheredAttention: stepLimits))
            let directStepBank = CBv2LayerCacheBank(caches: directStepBackend.makeLayerCaches())
            let directScheduler = SchedulerV2(config: schedConfig)
            let directLoop = EngineLoopV2(
                model: CBv2SteppableLanguageModelAdapter(sdcmModel),
                layerKinds: kinds,
                backend: directStepBackend,
                cacheProvider: directStepBank,
                sampler: CBv2DefaultSampler(fallbackSeed: 23),
                detokenizerFactory: CBv2NullDetokenizerFactory(),
                scheduler: directScheduler,
                capacity: nil,
                config: CBv2EngineLoopConfig(),
                gauges: CBv2EngineGauges(kvBytesCapacity: directStepBackend.bytesCapacity))
            _ = directLoop
            #expect(!directScheduler.soloCompactionSupported)
            try directScheduler.enqueue(
                CBv2Request(
                    id: CBv2RequestID(33), promptTokens: Array(repeating: 1, count: 160),
                    maxTokens: 4))
            let directPlan = directScheduler.plan()
            #expect(directPlan.assignments.count == 1)
            #expect(directPlan.assignments[0].numTokens == 64)
        }

        // 3. End-to-end EngineV2 with a real PagedKVBackend(maxPrefillChunk: 64) and an incapable
        // CBv2LayerCacheProvider: scheduler refuses widening a 160-token cold prompt (> 64, >= 96),
        // preventing PagedSequenceKV's windowed maxPrefillChunk trap and producing exact fallback.
        do {
            let pagedBackend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 64 * 1024 * 1024, dtype: .float32,
                    maxPrefillChunk: 64))
            let incapableProvider = IncapableCacheProvider(
                CBv2LayerCacheBank(caches: pagedBackend.makeLayerCaches()))
            #expect(pagedBackend.supportsCompactedPromptKVWrite)
            #expect(!incapableProvider.supportsCompactedPromptKVWrite)

            let spy = CompactionObservingSteppableModel(
                CBv2SteppableLanguageModelAdapter(sdcmModel), watchedLayer: kinds.count - 1)
            var schedConfig = CBv2SchedulerConfig(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 64,
                prefillChunkSize: 64,
                soloPrefillStripeTokens: nil)
            schedConfig.soloCompactionMinPromptTokens = 96

            let engine = EngineV2(
                model: spy,
                layerKinds: kinds,
                backend: pagedBackend,
                cacheProvider: incapableProvider,
                sampler: CBv2DefaultSampler(fallbackSeed: 29),
                schedulerConfig: schedConfig)
            #expect(engine.schedulerConfig.soloCompactionMinPromptTokens == nil)

            var refConfig = schedConfig
            refConfig.soloCompactionMinPromptTokens = nil
            let refBackend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 64 * 1024 * 1024, dtype: .float32,
                    maxPrefillChunk: 64))
            let refEngine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(offModel),
                layerKinds: kinds,
                backend: refBackend,
                cacheProvider: CBv2LayerCacheBank(caches: refBackend.makeLayerCaches()),
                sampler: CBv2DefaultSampler(fallbackSeed: 29),
                schedulerConfig: refConfig)

            let promptLen = 160  // >= soloCompactionMinPromptTokens(96) and > maxPrefillChunk(64)
            let promptTokens = (0 ..< promptLen).map { ($0 * 11 + 3) % 200 }
            let greedy = CBv2SamplingParams(temperature: 0)

            func collectTokens(_ engine: EngineV2, request: CBv2Request) async throws -> [Int] {
                var out: [Int] = []
                for await event in try engine.submit(request) {
                    if case .delta(_, let tokens, _) = event {
                        out.append(contentsOf: tokens)
                    }
                }
                return out
            }

            let actualTokens = try await collectTokens(
                engine,
                request: CBv2Request(
                    id: CBv2RequestID(130), promptTokens: promptTokens, sampling: greedy,
                    maxTokens: 3))
            let refTokens = try await collectTokens(
                refEngine,
                request: CBv2Request(
                    id: CBv2RequestID(131), promptTokens: promptTokens, sampling: greedy,
                    maxTokens: 3))

            await engine.shutdown()
            await refEngine.shutdown()

            #expect(actualTokens == refTokens)
            let observations = spy.observations
            #expect(observations.map(\.tokenCount) == [64, 64, 32])
            #expect(observations.allSatisfy { !$0.compacted && !$0.usedApproximatePrefill })
        }
    }

    @Test("Frozen-replay rows report supportsCompactedPositionDelta == false and fall back before mutating any layer")
    func frozenReplayRowRejectedBeforeMutatingAnyLayer() throws {
        MLXRandom.seed(706)
        let model = try Self.makeSyntheticGemma4MoE()
        model.blockSDCMConfig = Self.makeHardeningSDCMConfig()

        let offModel = try Self.makeSyntheticGemma4MoE()
        try offModel.update(parameters: model.parameters(), verify: .all)
        offModel.blockSDCMConfig = .off

        let kinds = model.cbv2LayerKinds
        let promptLen = 128
        let prompt = MLXArray((0 ..< promptLen).map { Int32(($0 * 3 + 17) % 200) })
            .expandedDimensions(axis: 0)

        // Compute exact reference K/V so the frozen-replay snapshot at layer 11 is exact.
        let (offRows, offCaches) = Self.makeContiguousCBv2Caches(
            kinds: kinds, promptLength: promptLen)
        let logitsOff = offModel.cbv2Prefill(
            prompt, inputEmbedding: nil, cache: offCaches, requirement: .lastPositionLogits)
        eval(logitsOff)
        let lastIdx = kinds.count - 1
        let exactLastSnapshot = offRows[lastIdx].snapshot()
        eval(exactLastSnapshot.keys, exactLastSnapshot.values)

        // 1. Contiguous CBv2FrozenReplayFullSequenceKV(replayStart: 0) starts at offset 0
        // (so initialOffset == 0), yet must reject Block-SDCM via supportsCompactedPositionDelta.
        let lastKind = kinds[lastIdx]
        let frozenRow = CBv2FrozenReplayFullSequenceKV(
            snapshot: exactLastSnapshot,
            replayStart: 0,
            maxLength: 256,
            kvHeads: lastKind.kvHeads,
            headDim: lastKind.headDim)
        #expect(frozenRow.absoluteOffset == 0)
        #expect(frozenRow.ropePositionOffset == 0)
        #expect(!frozenRow.supportsCompactedPositionDelta)

        let (rows, cbv2Caches) = Self.makeContiguousCBv2Caches(
            kinds: kinds, promptLength: promptLen)
        for idx in 0 ..< lastIdx {
            #expect(rows[idx].supportsCompactedPositionDelta)
        }
        let frozenCache = CBv2LayerCache(layerIndex: lastIdx, kind: lastKind)
        #expect(frozenCache.supportsCompactedPromptKVWrite)  // unbound: empty rows
        frozenCache.setRows([frozenRow])
        #expect(!frozenCache.supportsCompactedPromptKVWrite)

        var mixedCaches: [KVCache] = cbv2Caches
        mixedCaches[lastIdx] = frozenCache
        #expect(
            !Gemma4BlockSDCM.supportsCaches(
                mixedCaches.map { $0 as KVCache? }, modelConfig: model.configuration))

        let logitsFrozen = model.cbv2Prefill(
            prompt, inputEmbedding: nil, cache: mixedCaches, requirement: .lastPositionLogits)
        eval(logitsFrozen)
        #expect(MLX.abs(logitsFrozen - logitsOff).max().item(Float.self) < 1e-4)
        for idx in 0 ..< lastIdx {
            #expect(!rows[idx].isCompacted)
            #expect(!rows[idx].usedApproximatePrefill)
            #expect(rows[idx].absoluteOffset == promptLen)
            #expect(rows[idx].ropePositionOffset == promptLen)
        }
        #expect(!frozenRow.isCompacted)
        #expect(!frozenRow.usedApproximatePrefill)
        #expect(frozenRow.absoluteOffset == promptLen)
        #expect(frozenRow.ropePositionOffset == promptLen)

        // 2. PagedSequenceKV rejects compaction once frozen-adopted (even at replayStart: 0)
        // or fast-forwarded (baseOffset > 0).
        let pagedBackend = try PagedKVBackend(
            layerKinds: [kinds[0], lastKind],
            config: PagedKVPoolConfig(
                pageSize: 16, capacityBytes: 16 * 1024 * 1024, dtype: .float32,
                maxPrefillChunk: 128, nominalMaxSequenceLength: 256))
        let pagedState = try pagedBackend.makeSequenceState(
            layerKinds: [kinds[0], lastKind], promptLength: promptLen, maxLength: 256)
        defer { pagedBackend.release(pagedState) }
        let pagedSliding = pagedState[0] as! PagedSequenceKV
        let pagedFull = pagedState[1] as! PagedSequenceKV
        let pagedCaches = pagedBackend.makeLayerCaches()
        pagedCaches[0].setRows([pagedSliding])
        pagedCaches[1].setRows([pagedFull])
        #expect(pagedSliding.supportsCompactedPositionDelta)
        #expect(pagedFull.supportsCompactedPositionDelta)
        #expect(pagedCaches[0].supportsCompactedPromptKVWrite)
        #expect(pagedCaches[1].supportsCompactedPromptKVWrite)

        pagedFull.adoptFrozen(
            keys: exactLastSnapshot.keys.squeezed(axis: 0).asType(.float32),
            values: exactLastSnapshot.values.squeezed(axis: 0).asType(.float32),
            replayStart: 0)
        #expect(pagedFull.absoluteOffset == 0)
        #expect(pagedFull.ropePositionOffset == 0)
        #expect(!pagedFull.supportsCompactedPositionDelta)
        #expect(!pagedCaches[1].supportsCompactedPromptKVWrite)

        pagedSliding.fastForward(to: 16)
        #expect(!pagedSliding.supportsCompactedPositionDelta)
        #expect(!pagedCaches[0].supportsCompactedPromptKVWrite)
    }

    @Test("Zero-drop Block-SDCM marks sticky usedApproximatePrefill and blocks prefix-cache donation")
    func zeroDropBlockSDCMMarksStickyApproximatePrefillAndBlocksDonation() async throws {
        MLXRandom.seed(707)
        let model = try Self.makeSyntheticGemma4MoE()
        let zeroDropSDCM = Gemma4BlockSDCMConfig(
            mode: .fast,
            minPromptTokens: 32,
            keepSchedule: [6: 1.0],
            earlyMoEK: 2,
            lateMoEK: 4,
            earlyMoEUntilLayer: 6,
            blockSize: 8,
            prefixKeep: 8,
            tailKeep: 8,
            forwardHalo: 0.85,
            backwardHalo: 0.50,
            strataCount: 2,
            twoHopBlockPeaks: 4,
            slidingChunkSize: 32
        )
        model.blockSDCMConfig = zeroDropSDCM

        let kinds = model.cbv2LayerKinds
        // Keep promptLen <= slidingWindow (48) so sliding layers also write all tokens
        // with logicalDelta == 0 and isCompacted == false across every layer.
        let promptLen = 40
        let promptTokens = (0 ..< promptLen).map { ($0 * 7 + 9) % 200 }
        let prompt = MLXArray(promptTokens.map(Int32.init)).expandedDimensions(axis: 0)

        // 1. Direct contiguous CBv2 pass: zero positions dropped, yet every row has sticky
        // usedApproximatePrefill == true and PrefixCacheV2 refuses donation.
        do {
            let (rows, caches) = Self.makeContiguousCBv2Caches(
                kinds: kinds, promptLength: promptLen)
            let logits = model.cbv2Prefill(
                prompt, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
            eval(logits)
            for idx in 0 ..< kinds.count {
                #expect(!rows[idx].isCompacted)
                #expect(rows[idx].usedApproximatePrefill)
                #expect(rows[idx].absoluteOffset == promptLen)
                #expect(rows[idx].ropePositionOffset == promptLen)
            }
            let directPrefixCache = PrefixCacheV2(config: CBv2PrefixCacheConfig(blockSize: 16))
            directPrefixCache.donate(
                tokens: promptTokens, state: rows.map { $0 as CBv2SequenceKV? }, layerKinds: kinds)
            #expect(directPrefixCache.stats().entryCount == 0)
        }

        // 2. Direct paged CBv2 pass: zero positions dropped, yet usedApproximatePrefill == true
        // vetoes resident prefix page handle publication.
        do {
            let backend = try PagedKVBackend(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: 16, capacityBytes: 32 * 1024 * 1024, dtype: .float32,
                    maxPrefillChunk: 128, nominalMaxSequenceLength: 256))
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: promptLen, maxLength: 128)
            defer { backend.release(state) }
            let rows = state.map { $0 as! PagedSequenceKV }
            let caches = backend.makeLayerCaches()
            for idx in 0 ..< kinds.count {
                caches[idx].setRows([rows[idx]])
            }
            let logits = model.cbv2Prefill(
                prompt, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
            eval(logits)
            for idx in 0 ..< kinds.count {
                #expect(!rows[idx].isCompacted)
                #expect(rows[idx].usedApproximatePrefill)
                #expect(rows[idx].absoluteOffset == promptLen)
                #expect(rows[idx].ropePositionOffset == promptLen)
            }
            #expect(rows[5].prefixPageHandles(tokens: 0 ..< 16) == nil)
            #expect(rows[11].prefixPageHandles(tokens: 0 ..< 16) == nil)
        }

        // 3. EngineV2 with PrefixCacheV2: zero-drop Block-SDCM request never donates,
        // while disablePrefillCompaction: true donates normally on an identical engine.
        do {
            var schedConfig = CBv2SchedulerConfig(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 128,
                prefillChunkSize: 128,
                soloPrefillStripeTokens: nil,
                enablePrefixCache: true)
            schedConfig.soloCompactionMinPromptTokens = 32

            let spyApprox = CompactionObservingSteppableModel(
                CBv2SteppableLanguageModelAdapter(model), watchedLayer: kinds.count - 1)
            let approxPrefixCache = PrefixCacheV2(config: CBv2PrefixCacheConfig(blockSize: 16))
            let approxEngine = EngineV2(
                model: spyApprox,
                layerKinds: kinds,
                backend: CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 1 << 27, kvDType: .float32)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
                sampler: CBv2DefaultSampler(fallbackSeed: 31),
                schedulerConfig: schedConfig,
                prefixCache: approxPrefixCache)

            let spyExact = CompactionObservingSteppableModel(
                CBv2SteppableLanguageModelAdapter(model), watchedLayer: kinds.count - 1)
            let exactPrefixCache = PrefixCacheV2(config: CBv2PrefixCacheConfig(blockSize: 16))
            let exactEngine = EngineV2(
                model: spyExact,
                layerKinds: kinds,
                backend: CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 1 << 27, kvDType: .float32)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
                sampler: CBv2DefaultSampler(fallbackSeed: 31),
                schedulerConfig: schedConfig,
                prefixCache: exactPrefixCache)

            let greedy = CBv2SamplingParams(temperature: 0)
            func drain(_ engine: EngineV2, request: CBv2Request) async throws {
                for await _ in try engine.submit(request) {}
            }

            try await drain(
                approxEngine,
                request: CBv2Request(
                    id: CBv2RequestID(140), promptTokens: promptTokens, sampling: greedy,
                    maxTokens: 2))
            try await drain(
                exactEngine,
                request: CBv2Request(
                    id: CBv2RequestID(141), promptTokens: promptTokens, sampling: greedy,
                    maxTokens: 2, disablePrefillCompaction: true))

            await approxEngine.shutdown()
            await exactEngine.shutdown()

            #expect(spyApprox.observations.count == 1)
            #expect(!spyApprox.observations[0].compacted)
            #expect(spyApprox.observations[0].usedApproximatePrefill)
            #expect(approxPrefixCache.stats().entryCount == 0)

            #expect(spyExact.observations.count == 1)
            #expect(!spyExact.observations[0].compacted)
            #expect(!spyExact.observations[0].usedApproximatePrefill)
            #expect(exactPrefixCache.stats().entryCount == 1)
        }
    }

    @Test("disablePrefillCompaction vetoes SchedulerV2 solo compaction and EngineV2 single-chunk Block-SDCM")
    func disablePrefillCompactionBypassesSDCMOnColdSoloPrompt() async throws {
        // 1. SchedulerV2 must not arm isCompaction or suppress checkpoint splits when vetoed.
        do {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 4,
                maxBatchedTokensPerStep: 4096,
                prefillChunkSize: 2048,
                soloPrefillStripeTokens: nil,
                enablePrefixCache: true)
            config.soloCompactionMinPromptTokens = 1536
            config.demandedShortCheckpointMinimumTokens = 1024
            config.demandedCheckpointPartitionIncludesLongPrompts = true
            config.demandedLatestCheckpointMinimumTokens = 1024

            let schedulerLong = SchedulerV2(config: config)
            let vetoedLong = CBv2Request(
                id: CBv2RequestID(21), promptTokens: Array(repeating: 1, count: 6000),
                maxTokens: 16, disablePrefillCompaction: true)
            try schedulerLong.enqueue(vetoedLong)
            let planLong = schedulerLong.plan()
            #expect(planLong.assignments.count == 1)
            #expect(planLong.assignments[0].numTokens == 2048)

            // With soloPrefillStripeTokens == nil, the latest-boundary rule applies:
            // reach = 1800 - 1 - 64 = 1735 -> aligned down to 256-token boundary = 1536.
            let schedulerHint = SchedulerV2(config: config)
            var vetoedHint = CBv2Request(
                id: CBv2RequestID(22), promptTokens: Array(repeating: 1, count: 1800),
                maxTokens: 16, cacheSalt: "salt-veto", disablePrefillCompaction: true)
            vetoedHint.prefixCheckpointTargetTokens = 1280
            try schedulerHint.enqueue(vetoedHint)
            let planHint = schedulerHint.plan()
            #expect(planHint.assignments.count == 1)
            #expect(planHint.assignments[0].numTokens == 1536)
            #expect(!planHint.demandedShortCheckpointRows.isEmpty)

            // Without the veto, solo compaction arms and keeps the prompt whole (1800).
            let schedulerUnvetoed = SchedulerV2(config: config)
            var unvetoedHint = CBv2Request(
                id: CBv2RequestID(23), promptTokens: Array(repeating: 1, count: 1800),
                maxTokens: 16, cacheSalt: "salt-allow", disablePrefillCompaction: false)
            unvetoedHint.prefixCheckpointTargetTokens = 1280
            try schedulerUnvetoed.enqueue(unvetoedHint)
            let planUnvetoed = schedulerUnvetoed.plan()
            #expect(planUnvetoed.assignments.count == 1)
            #expect(planUnvetoed.assignments[0].numTokens == 1800)
            #expect(planUnvetoed.demandedShortCheckpointRows.isEmpty)
        }

        // 2. EngineV2 single-chunk cold solo prompt (L <= prefillChunkSize): veto must still
        // bypass Block-SDCM, while default requests on the same engine still compact.
        MLXRandom.seed(704)
        let sdcmModel = try Self.makeSyntheticGemma4MoE()
        sdcmModel.blockSDCMConfig = Self.makeHardeningSDCMConfig()

        let offModel = try Self.makeSyntheticGemma4MoE()
        try offModel.update(parameters: sdcmModel.parameters(), verify: .all)
        offModel.blockSDCMConfig = .off

        let kinds = sdcmModel.cbv2LayerKinds
        let spy = CompactionObservingSteppableModel(
            CBv2SteppableLanguageModelAdapter(sdcmModel), watchedLayer: kinds.count - 1)

        var schedConfig = CBv2SchedulerConfig(
            maxConcurrentRequests: 2,
            maxBatchedTokensPerStep: 256,
            prefillChunkSize: 256,
            soloPrefillStripeTokens: nil)
        schedConfig.soloCompactionMinPromptTokens = 96

        let engine = EngineV2(
            model: spy,
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27, kvDType: .float32)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 19),
            schedulerConfig: schedConfig)

        var refConfig = schedConfig
        refConfig.soloCompactionMinPromptTokens = nil
        let refEngine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(offModel),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 27, kvDType: .float32)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 19),
            schedulerConfig: refConfig)

        let promptLen = 128  // >= minPromptTokens(96) and <= prefillChunkSize(256)
        let promptTokens = (0 ..< promptLen).map { ($0 * 9 + 5) % 200 }
        let greedy = CBv2SamplingParams(temperature: 0)

        func collectTokens(_ engine: EngineV2, request: CBv2Request) async throws -> [Int] {
            var out: [Int] = []
            for await event in try engine.submit(request) {
                if case .delta(_, let tokens, _) = event {
                    out.append(contentsOf: tokens)
                }
            }
            return out
        }

        let vetoedTokens = try await collectTokens(
            engine,
            request: CBv2Request(
                id: CBv2RequestID(100), promptTokens: promptTokens, sampling: greedy,
                maxTokens: 3, disablePrefillCompaction: true))
        let refTokens = try await collectTokens(
            refEngine,
            request: CBv2Request(
                id: CBv2RequestID(101), promptTokens: promptTokens, sampling: greedy,
                maxTokens: 3))
        let compactedTokens = try await collectTokens(
            engine,
            request: CBv2Request(
                id: CBv2RequestID(102), promptTokens: promptTokens, sampling: greedy,
                maxTokens: 3))

        await engine.shutdown()
        await refEngine.shutdown()

        #expect(vetoedTokens == refTokens)
        #expect(compactedTokens.count == 3)

        let observations = spy.observations
        #expect(observations.count == 2)
        // First request (disablePrefillCompaction == true): exact single-chunk prefill.
        #expect(observations[0].scopeDisabled)
        #expect(!observations[0].compacted)
        #expect(!observations[0].usedApproximatePrefill)
        #expect(observations[0].absoluteOffset == promptLen)
        #expect(observations[0].ropeOffset == promptLen)
        // Second request (disablePrefillCompaction == false): Block-SDCM compacted layer 11.
        #expect(!observations[1].scopeDisabled)
        #expect(observations[1].compacted)
        #expect(observations[1].usedApproximatePrefill)
        #expect(observations[1].absoluteOffset < promptLen)
        #expect(observations[1].ropeOffset == promptLen)
    }

    // MARK: - Coherent Full-KV Mode & Adaptive 2W/L Retention Floor

    @Test(
        "effectiveTargetFraction enforces 2W/L floor across prompt lengths and guards degenerate inputs"
    )
    func effectiveTargetFractionAdaptiveFloor() {
        let W = 1024
        var bal = Gemma4BlockSDCMConfig.balanced
        bal.slidingWindowToRetain = W
        let schedule: [Float] = [0.46, 0.30, 0.24, 0.20]

        // 1W: floor = 2.0 -> clamped to 1.0 across all compaction boundaries.
        for planned in schedule {
            #expect(bal.effectiveTargetFraction(planned: planned, originalLength: W) == 1.0)
        }

        // 2.4W (~2458 tokens): floor = 2048 / 2458 ≈ 0.8332.
        let len24W = Int((2.4 * Float(W)).rounded())
        let expected24W = Float(2 * W) / Float(len24W)
        for planned in schedule {
            let actual = bal.effectiveTargetFraction(planned: planned, originalLength: len24W)
            #expect(abs(actual - expected24W) < 1e-5)
        }

        // 4W: floor = 0.50 lifts all four boundaries to 0.50.
        for planned in schedule {
            #expect(bal.effectiveTargetFraction(planned: planned, originalLength: 4 * W) == 0.50)
        }

        // 8W: floor = 0.25 -> 46 / 30 / 25 / 25%.
        let at8W = schedule.map {
            bal.effectiveTargetFraction(planned: $0, originalLength: 8 * W)
        }
        #expect(at8W == [0.46, 0.30, 0.25, 0.25])

        // 16W and 32W: floor <= 0.125 -> planned 46 / 30 / 24 / 20% unchanged.
        for mult in [16, 32] {
            let actual = schedule.map {
                bal.effectiveTargetFraction(planned: $0, originalLength: mult * W)
            }
            #expect(actual == schedule)
        }

        // Fast preset leaves adaptiveRetentionFloorWindows == 0.
        var fast = Gemma4BlockSDCMConfig.fast
        fast.slidingWindowToRetain = W
        #expect(fast.effectiveTargetFraction(planned: 0.12, originalLength: W) == 0.12)

        // Degenerate inputs stay finite and safe.
        #expect(bal.effectiveTargetFraction(planned: 0.20, originalLength: 0) == 0.20)
        #expect(bal.effectiveTargetFraction(planned: 0.20, originalLength: -64) == 0.20)
        #expect(bal.effectiveTargetFraction(planned: .nan, originalLength: 4 * W) == 1.0)
        #expect(bal.effectiveTargetFraction(planned: .infinity, originalLength: 4 * W) == 1.0)
        #expect(bal.effectiveTargetFraction(planned: -.infinity, originalLength: 4 * W) == 1.0)
        #expect(bal.effectiveTargetFraction(planned: -0.5, originalLength: 4 * W) == 0.50)
        #expect(bal.effectiveTargetFraction(planned: 1.5, originalLength: 4 * W) == 1.0)

        var noWindow = bal
        noWindow.slidingWindowToRetain = 0
        #expect(noWindow.effectiveTargetFraction(planned: 0.20, originalLength: W) == 0.20)
        noWindow.slidingWindowToRetain = -16
        #expect(noWindow.effectiveTargetFraction(planned: 0.20, originalLength: W) == 0.20)

        for badFloor: Float in [0, -2.0, .nan, .infinity, -.infinity] {
            var badCfg = bal
            badCfg.adaptiveRetentionFloorWindows = badFloor
            #expect(badCfg.effectiveTargetFraction(planned: 0.20, originalLength: W) == 0.20)
            let initCfg = Gemma4BlockSDCMConfig(
                mode: .balanced, keepSchedule: [6: 0.46], adaptiveRetentionFloorWindows: badFloor)
            #expect(initCfg.adaptiveRetentionFloorWindows == 0)
        }
    }

    @Test(
        "Skipped interior sliding layer matches exact tail hidden states, RMS-matched interior, and sliding KV commit"
    )
    func skippedInteriorSlidingLayerParity() throws {
        MLXRandom.seed(808)
        let model = try Self.makeSyntheticGemma4MoE()
        let layerIdx = 6  // sliding layer (W = 48)
        let layer = try #require(model.decoderLayers[layerIdx] as? Gemma4DecoderLayer)
        let window = model.configuration.slidingWindow
        let L = 128
        let cut = L - window

        let x = MLXRandom.normal([1, L, model.configuration.hiddenSize]).asType(.float32)
        let positionsHost = (0 ..< L).map { Int32($0) }
        let positions = MLXArray(positionsHost)

        let baseCfg = Gemma4BlockSDCMConfig(
            mode: .coherent,
            minPromptTokens: 64,
            keepSchedule: [:],
            earlyMoEK: 8,
            lateMoEK: 8,
            tailKeep: window,
            earlySparseFullLayer: nil
        )
        var skipCfg = baseCfg
        skipCfg.skipInteriorSlidingLayers = [layerIdx]

        let kind = model.cbv2LayerKinds[layerIdx]
        let exactRow = CBv2WindowedSequenceKV(
            window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
        let exactCache = CBv2LayerCache(layerIndex: layerIdx, kind: kind)
        exactCache.setRows([exactRow])

        let skipRow = CBv2WindowedSequenceKV(
            window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
        let skipCache = CBv2LayerCache(layerIndex: layerIdx, kind: kind)
        skipCache.setRows([skipRow])

        let (exactOut, _) = layer.forwardBlockSDCM(
            x, positions: positions, positionsHost: positionsHost, originalLength: L,
            cache: exactCache, sdcm: baseCfg, isFinalLayer: false,
            nextReverseQuota: nil, isExpertPrefill: false)
        let (skipOut, _) = layer.forwardBlockSDCM(
            x, positions: positions, positionsHost: positionsHost, originalLength: L,
            cache: skipCache, sdcm: skipCfg, isFinalLayer: false,
            nextReverseQuota: nil, isExpertPrefill: false)

        let exactSnap = exactRow.snapshot()
        let skipSnap = skipRow.snapshot()
        eval(exactOut, skipOut, exactSnap.keys, exactSnap.values, skipSnap.keys, skipSnap.values)

        let tailDiff = MLX.abs(
            exactOut[0..., cut..., 0...] - skipOut[0..., cut..., 0...]
        ).max().item(Float.self)
        #expect(tailDiff < 1e-4, "Skipped sliding layer tail mismatch: \(tailDiff)")

        let kDiff = MLX.abs(exactSnap.keys - skipSnap.keys).max().item(Float.self)
        let vDiff = MLX.abs(exactSnap.values - skipSnap.values).max().item(Float.self)
        #expect(kDiff < 1e-5, "Skipped sliding layer K cache mismatch: \(kDiff)")
        #expect(vDiff < 1e-5, "Skipped sliding layer V cache mismatch: \(vDiff)")
        #expect(skipRow.absoluteOffset == window)
        #expect(skipRow.ropePositionOffset == L)
        #expect(skipRow.usedApproximatePrefill)

        let xTail = x[0..., cut..., 0...]
        let hTail = skipOut[0..., cut..., 0...]
        let rmsIn = MLX.sqrt((xTail * xTail).mean())
        let rmsOut = MLX.sqrt((hTail * hTail).mean())
        let beta = MLX.clip(rmsOut / MLX.maximum(rmsIn, MLXArray(Float(1e-4))), min: Float(0.25), max: Float(4.0))
        let expectedInterior = x[0..., ..<cut, 0...] * beta
        let intDiff = MLX.abs(skipOut[0..., ..<cut, 0...] - expectedInterior).max().item(Float.self)
        #expect(intDiff < 1e-5, "Skipped sliding layer interior RMS-scale mismatch: \(intDiff)")

        // Larger `tailKeep > window` (and > `slidingChunkSize`) exercises chunked sliding attention
        // while still protecting all `tailKeep` tokens and committing trailing `window` KV.
        let wideTail = window + 32  // 80 > slidingChunkSize (64)
        let wideCut = L - wideTail
        var wideSkipCfg = skipCfg
        wideSkipCfg.tailKeep = wideTail
        wideSkipCfg.slidingChunkSize = 64
        let wideRow = CBv2WindowedSequenceKV(
            window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
        let wideCache = CBv2LayerCache(layerIndex: layerIdx, kind: kind)
        wideCache.setRows([wideRow])
        let (wideOut, _) = layer.forwardBlockSDCM(
            x, positions: positions, positionsHost: positionsHost, originalLength: L,
            cache: wideCache, sdcm: wideSkipCfg, isFinalLayer: false,
            nextReverseQuota: nil, isExpertPrefill: false)
        let wideSnap = wideRow.snapshot()
        eval(wideOut, wideSnap.keys, wideSnap.values)
        let wideTailDiff = MLX.abs(
            exactOut[0..., wideCut..., 0...] - wideOut[0..., wideCut..., 0...]
        ).max().item(Float.self)
        #expect(wideTailDiff < 1e-4, "Wide-tail skipped sliding layer mismatch: \(wideTailDiff)")
        #expect(wideRow.absoluteOffset == window)
        #expect(wideRow.ropePositionOffset == L)
        #expect(MLX.abs(exactSnap.keys - wideSnap.keys).max().item(Float.self) < 1e-5)

        // Zero-input tail in FP16 must stay finite (beta falls back to 1.0 instead of FP16 inf).
        let xZeroTailFP16 = concatenated(
            [
                MLXRandom.normal([1, cut, model.configuration.hiddenSize]).asType(.float16),
                MLXArray.zeros([1, window, model.configuration.hiddenSize], dtype: .float16),
            ],
            axis: 1)
        let (zeroTailOut, _) = layer.forwardBlockSDCM(
            xZeroTailFP16, positions: positions, positionsHost: positionsHost, originalLength: L,
            cache: nil, sdcm: skipCfg, isFinalLayer: false,
            nextReverseQuota: nil, isExpertPrefill: false)
        eval(zeroTailOut)
        #expect(!MLX.isNaN(zeroTailOut).any().item(Bool.self))
        #expect(!MLX.isInf(zeroTailOut).any().item(Bool.self))
        let zeroTailIntDiff = MLX.abs(
            zeroTailOut[0..., ..<cut, 0...].asType(.float32)
                - xZeroTailFP16[0..., ..<cut, 0...].asType(.float32)
        ).max().item(Float.self)
        #expect(zeroTailIntDiff < 1e-4, "Zero-tail FP16 interior must keep beta == 1.0")
    }

    @Test(
        "Coherent full-KV mode retains 100% full-attention KV, contiguous sliding tails, decodes cleanly, and blocks prefix donation"
    )
    func endToEndCoherentFullKVAllBackends() throws {
        MLXRandom.seed(809)
        let model = try Self.makeSyntheticGemma4MoE()
        let window = model.configuration.slidingWindow  // 48
        let fullLayers: Set<Int> = [5, 11]
        let skippedSlidingLayers: Set<Int> = [6, 7, 8, 9]

        let coherentCfg = Gemma4BlockSDCMConfig(
            mode: .coherent,
            minPromptTokens: 96,
            keepSchedule: [:],
            earlyMoEK: 2,
            lateMoEK: 2,
            earlyMoEUntilLayer: 11,
            layerMoEK: [:],
            blockSize: 16,
            prefixKeep: 16,
            tailKeep: window,
            slidingChunkSize: 32,
            earlySparseFullLayer: 5,
            sparsePrefixTokens: 16,
            sparseLocalWindow: 32,
            sparseTailTokens: 32,
            sparseFullLayers: [5],
            skipInteriorSlidingLayers: skippedSlidingLayers
        )
        model.blockSDCMConfig = coherentCfg

        let promptLen = 128
        #expect(
            Gemma4BlockSDCM.canRun(
                sdcm: coherentCfg, modelConfig: model.configuration, schedulePrefill: true,
                allowsBlockSDCM: true, batchSize: 1, sequenceLength: promptLen,
                hasInputEmbedding: false, hasImageMask: false, hasCaptureHook: false,
                initialOffset: 0))

        let promptTokens = (0 ..< promptLen).map { ($0 * 13 + 7) % 200 }
        let prompt1D = MLXArray(promptTokens.map(Int32.init))
        let prompt = prompt1D.expandedDimensions(axis: 0)
        let nextTok = MLXArray([Int32(42)]).reshaped([1, 1])
        let kinds = model.cbv2LayerKinds

        // 1. Paged CBv2 Backend
        do {
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
            #expect(!MLX.isInf(prefillLogits).any().item(Bool.self))
            #expect(!backend.pool.writeValidation.isFaulted)

            for idx in 0 ..< kinds.count {
                #expect(rows[idx].ropePositionOffset == promptLen)
                #expect(rows[idx].usedApproximatePrefill)
                #expect(pagedCaches[idx].hostPositionOffsets == [Int32(promptLen)])
                if fullLayers.contains(idx) {
                    #expect(rows[idx].absoluteOffset == promptLen)
                    #expect(!rows[idx].isCompacted)
                    #expect(rows[idx].prefixPageHandles(tokens: 0 ..< 16) == nil)
                } else {
                    #expect(rows[idx].absoluteOffset == window)
                }
            }

            let decLogits = model(nextTok, cache: caches)
            eval(decLogits)
            #expect(decLogits.shape == [1, 1, model.vocabularySize])
            #expect(!MLX.isNaN(decLogits).any().item(Bool.self))
            #expect(!MLX.isInf(decLogits).any().item(Bool.self))
            #expect(!backend.pool.writeValidation.isFaulted)
            for idx in 0 ..< kinds.count {
                #expect(rows[idx].ropePositionOffset == promptLen + 1)
                #expect(pagedCaches[idx].hostPositionOffsets == [Int32(promptLen + 1)])
            }
        }

        // 2. Contiguous CBv2 Backend
        do {
            let (rows, caches) = Self.makeContiguousCBv2Caches(
                kinds: kinds, promptLength: promptLen)
            let prefillLogits = model.cbv2Prefill(
                prompt, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
            eval(prefillLogits)
            #expect(prefillLogits.shape == [1, model.vocabularySize])
            #expect(!MLX.isNaN(prefillLogits).any().item(Bool.self))
            #expect(!MLX.isInf(prefillLogits).any().item(Bool.self))

            for idx in 0 ..< kinds.count {
                #expect(rows[idx].ropePositionOffset == promptLen)
                #expect(rows[idx].usedApproximatePrefill)
                if fullLayers.contains(idx) {
                    #expect(rows[idx].absoluteOffset == promptLen)
                    #expect(!rows[idx].isCompacted)
                } else {
                    #expect(rows[idx].absoluteOffset == window)
                }
            }

            let prefixCache = PrefixCacheV2(config: CBv2PrefixCacheConfig(blockSize: 16))
            prefixCache.donate(
                tokens: promptTokens, state: rows.map { $0 as CBv2SequenceKV? }, layerKinds: kinds)
            #expect(prefixCache.stats().entryCount == 0)

            let decLogits = model(nextTok, cache: caches)
            eval(decLogits)
            #expect(decLogits.shape == [1, 1, model.vocabularySize])
            #expect(!MLX.isNaN(decLogits).any().item(Bool.self))
            #expect(!MLX.isInf(decLogits).any().item(Bool.self))
            for idx in 0 ..< kinds.count {
                #expect(rows[idx].ropePositionOffset == promptLen + 1)
            }
        }

        // 3. Standalone KVCache (`prepare` + frontier + decode)
        do {
            let caches = model.newCache(parameters: nil)
            let prep = try model.prepare(
                LMInput(tokens: prompt1D), cache: caches, state: nil, prefill: PrefillParameters())
            guard case .tokens(let remaining) = prep else {
                Issue.record("Expected .tokens from coherent prepare")
                return
            }
            #expect(remaining.tokens.size == 1)

            for idx in 0 ..< kinds.count {
                if case .scalar(let ropePos) = caches[idx].ropeOffset {
                    #expect(ropePos == promptLen - 1)
                } else {
                    Issue.record("Expected scalar ropeOffset at layer \(idx)")
                }
                if fullLayers.contains(idx) {
                    let simple = try #require(caches[idx] as? KVCacheSimple)
                    #expect(simple.offset == promptLen - 1)
                    #expect(simple.positionDelta == 0)
                } else {
                    #expect(caches[idx].state[0].dim(2) == window)
                }
            }

            let frontierLogits = model(remaining.tokens.expandedDimensions(axis: 0), cache: caches)
            eval(frontierLogits)
            #expect(!MLX.isNaN(frontierLogits).any().item(Bool.self))
            #expect(!MLX.isInf(frontierLogits).any().item(Bool.self))

            let decLogits = model(nextTok, cache: caches)
            eval(decLogits)
            #expect(!MLX.isNaN(decLogits).any().item(Bool.self))
            #expect(!MLX.isInf(decLogits).any().item(Bool.self))
            for idx in 0 ..< kinds.count {
                if case .scalar(let ropePos) = caches[idx].ropeOffset {
                    #expect(ropePos == promptLen + 1)
                } else {
                    Issue.record("Expected scalar ropeOffset after decode at layer \(idx)")
                }
            }
        }

        // 4. Adaptive 2W/L floor inside trunk keeps 100% full-attention KV when L == 2 * W.
        do {
            let shortLen = 2 * window  // 96
            var flooredCfg = Self.makeHardeningSDCMConfig()
            flooredCfg.mode = .balanced
            flooredCfg.minPromptTokens = shortLen
            flooredCfg.adaptiveRetentionFloorWindows = 2.0
            model.blockSDCMConfig = flooredCfg

            let shortPrompt = prompt[0..., 0 ..< shortLen]
            let (rows, caches) = Self.makeContiguousCBv2Caches(
                kinds: kinds, promptLength: shortLen)
            let logits = model.cbv2Prefill(
                shortPrompt, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
            eval(logits)
            #expect(!MLX.isNaN(logits).any().item(Bool.self))
            #expect(rows[5].absoluteOffset == shortLen)
            #expect(rows[11].absoluteOffset == shortLen)
            #expect(!rows[11].isCompacted)
            #expect(rows[11].usedApproximatePrefill)
        }
    }

    @Test("willCompact skips no-op adaptive boundaries and matches selectCompactedIndices")
    func willCompactMatchesSelectCompactedIndices() {
        var bal = Gemma4BlockSDCMConfig.balanced
        bal.slidingWindowToRetain = 1024

        // Near 2W (T = 2049, 2100), adaptive 2W/L floor rounds up to all interior blocks.
        for noOpLength in [2049, 2100] {
            let pos = (0 ..< noOpLength).map { Int32($0) }
            let target = bal.effectiveTargetFraction(planned: 0.46, originalLength: noOpLength)
            #expect(target < 1.0)
            #expect(
                !Gemma4BlockSDCM.willCompact(
                    positions: pos, originalLength: noOpLength, targetFraction: target, config: bal)
            )
            let nBlocks = (noOpLength + bal.blockSize - 1) / bal.blockSize
            let kept = Gemma4BlockSDCM.selectCompactedIndices(
                boundaryScores: Gemma4SDCMBoundaryScores(
                    forwardBlockScores: [Float](repeating: 1.0, count: nBlocks)),
                positions: pos, originalLength: noOpLength, targetFraction: target, config: bal)
            #expect(kept.count == noOpLength)
        }

        // Above the 2W floor (T = 2427, 8825), compaction drops interior blocks.
        for activeLength in [2427, 8825] {
            let pos = (0 ..< activeLength).map { Int32($0) }
            let target = bal.effectiveTargetFraction(planned: 0.46, originalLength: activeLength)
            #expect(
                Gemma4BlockSDCM.willCompact(
                    positions: pos, originalLength: activeLength, targetFraction: target,
                    config: bal))
            let nBlocks = (activeLength + bal.blockSize - 1) / bal.blockSize
            let kept = Gemma4BlockSDCM.selectCompactedIndices(
                boundaryScores: Gemma4SDCMBoundaryScores(
                    forwardBlockScores: (0 ..< nBlocks).map { Float($0 + 1) }),
                positions: pos, originalLength: activeLength, targetFraction: target, config: bal)
            #expect(kept.count < activeLength)
        }

        // Strata + reverse quota floor retaining all 6 surviving interior blocks is also a no-op.
        var fast = Gemma4BlockSDCMConfig.fast
        fast.slidingWindowToRetain = 1024
        let fastPrefix = Array(0 ..< 128).map(Int32.init)
        let fastSixInterior = Array(128 ..< (128 + 6 * 32)).map(Int32.init)
        let fastTail = Array(512 ..< 1536).map(Int32.init)
        let fastAlive = fastPrefix + fastSixInterior + fastTail
        #expect(
            !Gemma4BlockSDCM.willCompact(
                positions: fastAlive, originalLength: 1536, targetFraction: 0.22,
                reverseQuotaFraction: 0.28, config: fast))
    }

    @Test(
        "Quantized paged KV refuses compacted writes fail-closed and falls back to exact prefill"
    )
    func quantizedPagedKVRefusesCompactedWrites() throws {
        MLXRandom.seed(810)
        let model = try Self.makeSyntheticGemma4MoE()
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        let kinds = model.cbv2LayerKinds
        let promptLen = 128

        let quantBackend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                pageSize: 16, capacityBytes: 64 * 1024 * 1024, dtype: .float16,
                maxPrefillChunk: 128, nominalMaxSequenceLength: 256,
                segmentSizeBytes: 4 * 1024 * 1024,
                quantization: PagedKVQuantizationConfig(recentTokenCount: 16)))
        #expect(quantBackend.usesQuantizedStorage)
        #expect(!quantBackend.supportsCompactedPromptKVWrite)

        let state = try quantBackend.makeSequenceState(
            layerKinds: kinds, promptLength: promptLen, maxLength: 256)
        defer { quantBackend.release(state) }
        let rows = state.map { $0 as! PagedSequenceKV }
        let pagedCaches = quantBackend.makeLayerCaches()
        let caches: [KVCache] = pagedCaches.enumerated().map { idx, c in
            c.setRows([rows[idx]])
            return c
        }

        for idx in 0 ..< kinds.count {
            #expect(!rows[idx].supportsCompactedPositionDelta)
            #expect(!pagedCaches[idx].supportsCompactedPromptKVWrite)
        }
        #expect(!Gemma4BlockSDCM.supportsCaches(caches, modelConfig: model.configuration))

        // Coherent/Block-SDCM falls back to exact prefill without faulting quantized storage.
        var cfg = Self.makeHardeningSDCMConfig()
        cfg.mode = .coherent
        cfg.keepSchedule = [:]
        cfg.skipInteriorSlidingLayers = [6, 7]
        cfg.tailKeep = model.configuration.slidingWindow
        model.blockSDCMConfig = cfg

        let prompt = MLXArray((0 ..< promptLen).map { Int32(($0 * 11 + 5) % 200) })
            .expandedDimensions(axis: 0)
        let logits = model.cbv2Prefill(
            prompt, inputEmbedding: nil, cache: caches, requirement: .lastPositionLogits)
        eval(logits)
        #expect(!quantBackend.pool.writeValidation.isFaulted)
        #expect(!MLX.isNaN(logits).any().item(Bool.self))
        for idx in 0 ..< kinds.count {
            #expect(!rows[idx].usedApproximatePrefill)
            #expect(rows[idx].ropePositionOffset == promptLen)
        }

        // Direct writeCompactedPromptKV call refuses cleanly via writeValidation without advancing offsets.
        let beforeOffset = rows[5].absoluteOffset
        let dummyK = MLXArray.zeros(
            [1, kinds[5].kvHeads, 4, kinds[5].headDim], dtype: .float16)
        pagedCaches[5].writeCompactedPromptKV(keys: dummyK, values: dummyK, logicalDelta: 8)
        #expect(quantBackend.pool.writeValidation.isFaulted)
        #expect(rows[5].absoluteOffset == beforeOffset)
        #expect(rows[5].ropePositionOffset == beforeOffset)
    }

    // MARK: - Lossless Mixed-K MoE & Query-Blocked Cached SDPA Parity

    @Test(
        "weightedExpertUnsortMixedK matches 2-slice scatter and weightedExpertSum in BF16"
    )
    func weightedExpertUnsortMixedKParity() {
        guard Device.defaultDevice().deviceType == .gpu else { return }
        let hidden = 128
        let kTail = 8
        let cut = 19
        let tailTokens = 12  // L = 31 (odd cut and odd L)
        let numExperts = 128

        for kInt in [1, 2, 4, 6] {
            MLXRandom.seed(UInt64(900 + kInt))
            let aInt = cut * kInt
            let aTail = tailTokens * kTail

            let intOutputs = MLXRandom.normal([cut, kInt, hidden]).asType(.bfloat16)
            let tailOutputs = MLXRandom.normal([tailTokens, kTail, hidden]).asType(.bfloat16)
            let intWeights = MLX.softmax(
                MLXRandom.normal([cut, kInt]), axis: -1, precise: true
            ).asType(.bfloat16)
            let tailWeights = MLX.softmax(
                MLXRandom.normal([tailTokens, kTail]), axis: -1, precise: true
            ).asType(.bfloat16)

            let intExperts = MLXArray(
                (0 ..< aInt).map { UInt32(($0 * 29 + kInt * 7) % numExperts) }
            ).reshaped(cut, kInt)
            let tailExperts = MLXArray(
                (0 ..< aTail).map { UInt32(($0 * 41 + 13) % numExperts) }
            ).reshaped(tailTokens, kTail)

            let flatExperts = concatenated(
                [intExperts.flattened(), tailExperts.flattened()], axis: 0)
            let flatWeights = concatenated(
                [intWeights.flattened(), tailWeights.flattened()], axis: 0)
            let allOriginal = concatenated(
                [
                    intOutputs.reshaped(aInt, hidden),
                    tailOutputs.reshaped(aTail, hidden),
                ], axis: 0)

            let order = argSort(flatExperts)
            let inverseOrder = argSort(order)
            let sortedOutputs = allOriginal[order]

            let actual = weightedExpertUnsortMixedK(
                sortedOutputs: sortedOutputs,
                inverseOrder: inverseOrder,
                weights: flatWeights,
                cut: cut,
                interiorK: kInt,
                tailTokens: tailTokens,
                tailK: kTail)

            let unsortAll = sortedOutputs[inverseOrder]
            let refInt = weightedExpertSum(
                unsortAll[..<aInt].reshaped(cut, kInt, hidden), intWeights)
            let refTail = weightedExpertSum(
                unsortAll[aInt...].reshaped(tailTokens, kTail, hidden), tailWeights)
            let expected = concatenated([refInt, refTail], axis: 0)

            eval(actual, expected)
            #expect(actual.shape == [cut + tailTokens, hidden])
            #expect(actual.dtype == .bfloat16)
            let maxDiff = MLX.abs(
                actual.asType(.float32) - expected.asType(.float32)
            ).max().item(Float.self)
            #expect(maxDiff == 0, "weightedExpertUnsortMixedK kInt=\(kInt) diff: \(maxDiff)")
        }

        // Degenerate zero-interior and zero-tail boundaries.
        let edgeTokens = 9
        let edgeA = edgeTokens * kTail
        let edgeOut = MLXRandom.normal([edgeA, hidden]).asType(.bfloat16)
        let edgeW = MLX.softmax(
            MLXRandom.normal([edgeTokens, kTail]), axis: -1, precise: true
        ).asType(.bfloat16)
        let edgeExperts = MLXArray((0 ..< edgeA).map { UInt32(($0 * 17) % numExperts) })
        let edgeOrder = argSort(edgeExperts)
        let edgeInv = argSort(edgeOrder)
        let edgeSorted = edgeOut[edgeOrder]

        let onlyTail = weightedExpertUnsortMixedK(
            sortedOutputs: edgeSorted, inverseOrder: edgeInv, weights: edgeW.flattened(),
            cut: 0, interiorK: 2, tailTokens: edgeTokens, tailK: kTail)
        let onlyInt = weightedExpertUnsortMixedK(
            sortedOutputs: edgeSorted, inverseOrder: edgeInv, weights: edgeW.flattened(),
            cut: edgeTokens, interiorK: kTail, tailTokens: 0, tailK: 2)
        let refEdge = weightedExpertSum(edgeOut.reshaped(edgeTokens, kTail, hidden), edgeW)
        eval(onlyTail, onlyInt, refEdge)
        #expect(
            MLX.abs(onlyTail.asType(.float32) - refEdge.asType(.float32)).max().item(Float.self)
                <= 0.01)
        #expect(
            MLX.abs(onlyInt.asType(.float32) - refEdge.asType(.float32)).max().item(Float.self)
                <= 0.01)
    }

    @Test(
        "Single-pass mixed-K router, SwitchGLU, and Gemma4Experts match separate calls"
    )
    func mixedKMoEDispatchParity() throws {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 96,
                "num_attention_heads": 4,
                "head_dim": 64,
                "global_head_dim": 64,
                "vocab_size": 128,
                "num_key_value_heads": 2,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 32,
                "layer_types": ["sliding_attention", "full_attention"],
                "enable_moe_block": true,
                "num_experts": 16,
                "top_k_experts": 8,
                "moe_intermediate_size": 32
            }
            """
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))

        func canonicalize(_ indices: MLXArray, _ weights: MLXArray) -> (MLXArray, MLXArray) {
            let perm = argSort(indices, axis: -1)
            return (MLX.takeAlong(indices, perm, axis: -1), MLX.takeAlong(weights, perm, axis: -1))
        }

        // 1. Gemma4Router.routeMixedTopK vs separate routeTopK + callAsFunction.
        MLXRandom.seed(911)
        let router = Gemma4Router(config)
        let cut = 17
        let tailLen = 10
        let totalL = cut + tailLen  // 27
        let xF32 = MLXRandom.normal([1, totalL, 64]).asType(.float32)
        let (mIntIdx, mIntW, mTailIdx, mTailW) = router.routeMixedTopK(
            xF32, cut: cut, interiorTopK: 2)
        let (rIntIdx, rIntW) = router.routeTopK(xF32[0..., ..<cut, 0...], topK: 2)
        let (rTailIdx, rTailW) = router(xF32[0..., cut..., 0...])

        let (cmIntIdx, cmIntW) = canonicalize(mIntIdx, mIntW)
        let (crIntIdx, crIntW) = canonicalize(rIntIdx, rIntW)
        let (cmTailIdx, cmTailW) = canonicalize(mTailIdx, mTailW)
        let (crTailIdx, crTailW) = canonicalize(rTailIdx, rTailW)
        eval(cmIntIdx, cmIntW, crIntIdx, crIntW, cmTailIdx, cmTailW, crTailIdx, crTailW)
        #expect((cmIntIdx .== crIntIdx).all().item(Bool.self))
        #expect((cmTailIdx .== crTailIdx).all().item(Bool.self))
        #expect(MLX.abs(cmIntW - crIntW).max().item(Float.self) < 1e-5)
        #expect(MLX.abs(cmTailW - crTailW).max().item(Float.self) < 1e-5)

        // 2. SwitchGLU.callAndMixedKWeightedReduce: FP32 sorted, BF16 fused GPU, fusedGateUp,
        //    A < 64 unsorted fallback, subclass hooks, and zero-slice fallbacks.
        let x2D = xF32.reshaped(totalL, 64)
        let iIdx = mIntIdx.reshaped(cut, 2)
        let iW = mIntW.reshaped(cut, 2)
        let tIdx = mTailIdx.reshaped(tailLen, 8)
        let tW = mTailW.reshaped(tailLen, 8)
        for fuseGateUp in [false, true] {
            let glu = SwitchGLU(
                inputDims: 64, hiddenDims: 32, numExperts: 16, bias: false, fuseGateUp: fuseGateUp)

            // FP32 sorted fallback (A = 17*2 + 10*8 = 114 >= 64).
            let mixedF32 = glu.callAndMixedKWeightedReduce(
                x2D, interiorIndices: iIdx, interiorWeights: iW,
                tailIndices: tIdx, tailWeights: tW, isProductionPrefill: true)
            let refIntF32 = weightedExpertSum(glu(x2D[..<cut], iIdx), iW)
            let refTailF32 = weightedExpertSum(glu(x2D[cut...], tIdx), tW)
            let refF32 = concatenated([refIntF32, refTailF32], axis: 0)
            eval(mixedF32, refF32)
            let diffF32 = MLX.abs(mixedF32 - refF32).max().item(Float.self)
            #expect(
                diffF32 < 1e-4,
                "SwitchGLU FP32 mixed-K (fused=\(fuseGateUp)) diff: \(diffF32)")

            // Unsorted fallback (A = 5*2 + 3*8 = 34 < 64).
            let smallCut = 5
            let smallTail = 3
            let xSmall = x2D[..<(smallCut + smallTail)]
            let siIdx = iIdx[..<smallCut]
            let siW = iW[..<smallCut]
            let stIdx = tIdx[..<smallTail]
            let stW = tW[..<smallTail]
            let mixedSmall = glu.callAndMixedKWeightedReduce(
                xSmall, interiorIndices: siIdx, interiorWeights: siW,
                tailIndices: stIdx, tailWeights: stW, isProductionPrefill: true)
            let refSmall = concatenated(
                [
                    weightedExpertSum(glu(xSmall[..<smallCut], siIdx), siW),
                    weightedExpertSum(glu(xSmall[smallCut...], stIdx), stW),
                ], axis: 0)
            eval(mixedSmall, refSmall)
            #expect(MLX.abs(mixedSmall - refSmall).max().item(Float.self) < 1e-4)

            // Zero-interior and zero-tail slice delegation.
            let emptyIdx = MLXArray.zeros([0, 2], dtype: iIdx.dtype)
            let emptyW = MLXArray.zeros([0, 2], dtype: iW.dtype)
            let tailOnly = glu.callAndMixedKWeightedReduce(
                x2D[cut...], interiorIndices: emptyIdx, interiorWeights: emptyW,
                tailIndices: tIdx, tailWeights: tW)
            let intOnly = glu.callAndMixedKWeightedReduce(
                x2D[..<cut], interiorIndices: iIdx, interiorWeights: iW,
                tailIndices: MLXArray.zeros([0, 8], dtype: tIdx.dtype),
                tailWeights: MLXArray.zeros([0, 8], dtype: tW.dtype))
            eval(tailOnly, intOnly)
            #expect(MLX.abs(tailOnly - refTailF32).max().item(Float.self) < 1e-4)
            #expect(MLX.abs(intOnly - refIntF32).max().item(Float.self) < 1e-4)

            // BF16 GPU fused path (fuseSortedReduction=true) vs BF16 fallback (false) vs separate.
            glu.train(false)
            glu.update(parameters: glu.parameters().mapValues { $0.asType(.bfloat16) })
            let xBF16 = x2D.asType(.bfloat16)
            let iWBF16 = iW.asType(.bfloat16)
            let tWBF16 = tW.asType(.bfloat16)
            let fusedBF16 = glu.callAndMixedKWeightedReduce(
                xBF16, interiorIndices: iIdx, interiorWeights: iWBF16,
                tailIndices: tIdx, tailWeights: tWBF16,
                fuseSortedReduction: true, isProductionPrefill: true)
            let unfusedBF16 = glu.callAndMixedKWeightedReduce(
                xBF16, interiorIndices: iIdx, interiorWeights: iWBF16,
                tailIndices: tIdx, tailWeights: tWBF16,
                fuseSortedReduction: false, isProductionPrefill: true)
            let sepBF16 = concatenated(
                [
                    weightedExpertSum(glu(xBF16[..<cut], iIdx), iWBF16),
                    weightedExpertSum(glu(xBF16[cut...], tIdx), tWBF16),
                ], axis: 0)
            eval(fusedBF16, unfusedBF16, sepBF16)
            let diffFusedVsUnfused = MLX.abs(
                fusedBF16.asType(.float32) - unfusedBF16.asType(.float32)
            ).max().item(Float.self)
            let diffFusedVsSep = MLX.abs(
                fusedBF16.asType(.float32) - sepBF16.asType(.float32)
            ).max().item(Float.self)
            #expect(
                diffFusedVsUnfused == 0,
                "BF16 fused vs unfused diff: \(diffFusedVsUnfused)")
            #expect(diffFusedVsSep <= 0.05, "BF16 fused vs separate diff: \(diffFusedVsSep)")
        }

        // Subclass transformInput / transformHidden hooks are preserved.
        let hookedGLU = ScalingSwitchGLU(inputDims: 64, hiddenDims: 32, numExperts: 16)
        let hookedMixed = hookedGLU.callAndMixedKWeightedReduce(
            x2D, interiorIndices: iIdx, interiorWeights: iW,
            tailIndices: tIdx, tailWeights: tW)
        let hookedRef = concatenated(
            [
                weightedExpertSum(hookedGLU(x2D[..<cut], iIdx), iW),
                weightedExpertSum(hookedGLU(x2D[cut...], tIdx), tW),
            ], axis: 0)
        eval(hookedMixed, hookedRef)
        #expect(MLX.abs(hookedMixed - hookedRef).max().item(Float.self) < 1e-4)

        // 3. Gemma4Experts.callMixedK vs separate calls (FP32, BF16, and PrefillProjection scope).
        let experts = Gemma4Experts(config)
        let expMixedF32 = experts.callMixedK(
            xF32, interiorIndices: mIntIdx, interiorWeights: mIntW,
            tailIndices: mTailIdx, tailWeights: mTailW, isExpertPrefill: true)
        let expRefF32 = concatenated(
            [
                experts(
                    xF32[0..., ..<cut, 0...], topKIndices: mIntIdx, topKWeights: mIntW,
                    isExpertPrefill: true),
                experts(
                    xF32[0..., cut..., 0...], topKIndices: mTailIdx, topKWeights: mTailW,
                    isExpertPrefill: true),
            ], axis: 1)
        eval(expMixedF32, expRefF32)
        #expect(MLX.abs(expMixedF32 - expRefF32).max().item(Float.self) < 1e-4)

        experts.update(parameters: experts.parameters().mapValues { $0.asType(.bfloat16) })
        let xBF163D = xF32.asType(.bfloat16)
        let mIntWBF16 = mIntW.asType(.bfloat16)
        let mTailWBF16 = mTailW.asType(.bfloat16)
        let expMixedBF16 = experts.callMixedK(
            xBF163D, interiorIndices: mIntIdx, interiorWeights: mIntWBF16,
            tailIndices: mTailIdx, tailWeights: mTailWBF16, isExpertPrefill: true)
        let expRefBF16 = concatenated(
            [
                experts(
                    xBF163D[0..., ..<cut, 0...], topKIndices: mIntIdx, topKWeights: mIntWBF16,
                    isExpertPrefill: true),
                experts(
                    xBF163D[0..., cut..., 0...], topKIndices: mTailIdx, topKWeights: mTailWBF16,
                    isExpertPrefill: true),
            ], axis: 1)
        eval(expMixedBF16, expRefBF16)
        #expect(
            MLX.abs(expMixedBF16.asType(.float32) - expRefBF16.asType(.float32))
                .max().item(Float.self) <= 0.05)

        let dummyTokens = MLXArray.zeros([1, totalL], dtype: .int32)
        let scopedMixed = Gemma4PrefillProjectionScope.withPrefillFrame(
            tokens: dummyTokens, enabled: true
        ) {
            experts.callMixedK(
                xBF163D, interiorIndices: mIntIdx, interiorWeights: mIntWBF16,
                tailIndices: mTailIdx, tailWeights: mTailWBF16, isExpertPrefill: true)
        }
        let scopedRef = Gemma4PrefillProjectionScope.withPrefillFrame(
            tokens: dummyTokens, enabled: true
        ) {
            let hInt = experts(
                xBF163D[0..., ..<cut, 0...], topKIndices: mIntIdx, topKWeights: mIntWBF16,
                isExpertPrefill: true)
            let hTail = experts(
                xBF163D[0..., cut..., 0...], topKIndices: mTailIdx, topKWeights: mTailWBF16,
                isExpertPrefill: true)
            return concatenated([hInt, hTail], axis: 1)
        }
        eval(scopedMixed, scopedRef)
        #expect(
            MLX.abs(scopedMixed.asType(.float32) - scopedRef.asType(.float32))
                .max().item(Float.self) <= 0.01)
    }

    @Test(
        "slidingAttentionWithPositions q=128 contiguous cached masks match dense mask"
    )
    func slidingAttentionBlockedCachedParity() {
        #expect(Gemma4BlockSDCMConfig.coherent.slidingChunkSize == 128)
        #expect(Gemma4BlockSDCMConfig.lightcone.slidingChunkSize == 128)
        #expect(Gemma4BlockSDCMConfig.balanced.slidingChunkSize == 128)
        #expect(Gemma4BlockSDCMConfig.fast.slidingChunkSize == 128)
        #expect(Gemma4BlockSDCMConfig.turbo.slidingChunkSize == 128)

        MLXRandom.seed(922)
        let scale: Float = 1.0 / sqrt(32.0)

        // Window 256 (multiple of 128) and window 200 (non-multiple) both cover cold `.causal`,
        // transition block, steady cached blocks, and odd remainder block.
        for (L, window) in [(683, 256), (555, 200)] {
            let posHost = (0 ..< L).map { Int32($0) }
            let pos = MLXArray(posHost)
            let q = MLXRandom.normal([1, 4, L, 32]).asType(.float32)
            let k = MLXRandom.normal([1, 2, L, 32]).asType(.float32)
            let v = MLXRandom.normal([1, 2, L, 32]).asType(.float32)

            let cache = Gemma4SDCMMaskCache()
            let blocked = Gemma4BlockSDCM.slidingAttentionWithPositions(
                queries: q, keys: k, values: v, positions: pos, positionsHost: posHost,
                slidingWindow: window, scale: scale, chunkSize: 128,
                isContiguous: true, maskCache: cache)
            let cachedAfterFirst = cache.cachedMaskCount
            #expect(cachedAfterFirst > 0)

            // Reusing the same cache across layers hits all cached masks without new entries.
            let blockedReuse = Gemma4BlockSDCM.slidingAttentionWithPositions(
                queries: q, keys: k, values: v, positions: pos, positionsHost: posHost,
                slidingWindow: window, scale: scale, chunkSize: 128,
                isContiguous: true, maskCache: cache)
            #expect(cache.cachedMaskCount == cachedAfterFirst)

            let diffMat = pos.expandedDimensions(axis: 1) - pos.expandedDimensions(axis: 0)
            let denseMask = (diffMat .>= Int32(0)) .&& (diffMat .< Int32(window))
            let ref = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .array(denseMask))
            eval(blocked, blockedReuse, ref)

            let diff1 = MLX.abs(blocked - ref).max().item(Float.self)
            let diff2 = MLX.abs(blockedReuse - ref).max().item(Float.self)
            #expect(diff1 < 1e-5, "Cached q=128 sliding mismatch (L=\(L), W=\(window)): \(diff1)")
            #expect(diff2 < 1e-5, "Cached reuse sliding mismatch (L=\(L), W=\(window)): \(diff2)")
        }
    }

    @Test(
        "prefixLocalFullTailAttention q=128 contiguous and compacted paths match dense mask"
    )
    func prefixLocalFullTailBlockedCachedParity() {
        MLXRandom.seed(933)
        let scale: Float = 1.0 / sqrt(32.0)

        func verify(
            posHost: [Int32], P: Int, W: Int, tail: Int, isContiguous: Bool, label: String
        ) {
            let L = posHost.count
            let pos = MLXArray(posHost)
            let q = MLXRandom.normal([1, 4, L, 32]).asType(.float32)
            let k = MLXRandom.normal([1, 2, L, 32]).asType(.float32)
            let v = MLXRandom.normal([1, 2, L, 32]).asType(.float32)

            let cache = Gemma4SDCMMaskCache()
            let actual = Gemma4BlockSDCM.prefixLocalFullTailAttention(
                queries: q, keys: k, values: v,
                positions: pos, positionsHost: posHost,
                prefixTokens: P, localWindow: W, tailTokens: tail,
                scale: scale, chunkSize: 128, isContiguous: isContiguous, maskCache: cache)
            if isContiguous {
                let firstCount = cache.cachedMaskCount
                let replay = Gemma4BlockSDCM.prefixLocalFullTailAttention(
                    queries: q, keys: k, values: v,
                    positions: pos, positionsHost: posHost,
                    prefixTokens: P, localWindow: W, tailTokens: tail,
                    scale: scale, chunkSize: 128, isContiguous: true, maskCache: cache)
                #expect(cache.cachedMaskCount == firstCount)
                eval(replay)
            }

            let qPos = pos.expandedDimensions(axis: 1)
            let kPos = pos.expandedDimensions(axis: 0)
            let qIdx = MLXArray((0 ..< L).map { Int32($0) }).expandedDimensions(axis: 1)
            let cut = Int32(max(0, L - max(0, tail)))
            let denseMask =
                (kPos .<= qPos)
                .&& ((kPos .< Int32(max(0, P))) .|| ((qPos - kPos) .< Int32(W))
                    .|| (qIdx .>= cut))
            let ref = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .array(denseMask))
            eval(actual, ref)
            let maxDiff = MLX.abs(actual - ref).max().item(Float.self)
            #expect(maxDiff < 1e-5, "prefixLocalFullTail q=128 mismatch (\(label)): \(maxDiff)")
        }

        // Contiguous: P+W=280 (non-multiple of 128), tail=175 > 128 (multi-block .causal tail).
        let contig700 = (0 ..< 700).map { Int32($0) }
        verify(
            posHost: contig700, P: 100, W: 180, tail: 175,
            isContiguous: true, label: "contig-280")
        verify(
            posHost: contig700, P: 0, W: 180, tail: 175,
            isContiguous: true, label: "contig-P0")

        // Contiguous with W < 128 so block 0 hits s0 < pEnd with s1 - 1 >= P + W (P+W = 104).
        let contig512 = (0 ..< 512).map { Int32($0) }
        verify(
            posHost: contig512, P: 40, W: 64, tail: 160, isContiguous: true, label: "contig-W64")

        // Compacted positions with P+W=280 and multi-block .causal tail (224 > 128).
        var compacted: [Int32] = []
        for b in 0 ..< 32 {
            if b < 4 || b >= 25 || b.isMultiple(of: 2) {
                for j in 0 ..< 32 { compacted.append(Int32(b * 32 + j)) }
            }
        }
        verify(
            posHost: compacted, P: 100, W: 180, tail: 224,
            isContiguous: false, label: "compacted-280")
    }

    @Test("causalAttentionBlocked matches unblocked causal SDPA for D=512 blocked and D=64 Flash")
    func causalAttentionBlockedParity() {
        MLXRandom.seed(944)
        let L = 307  // 2 full blocks of 128 + remainder 51

        // Gemma 4 global full-attention headDim D=512 takes the q=128 blocked path.
        let scale512: Float = 1.0 / sqrt(512.0)
        let q512 = MLXRandom.normal([1, 4, L, 512]).asType(.float32)
        let k512 = MLXRandom.normal([1, 2, L, 512]).asType(.float32)
        let v512 = MLXRandom.normal([1, 2, L, 512]).asType(.float32)
        let blocked512 = Gemma4BlockSDCM.causalAttentionBlocked(
            queries: q512, keys: k512, values: v512, scale: scale512, blockSize: 128)
        let ref512 = MLXFast.scaledDotProductAttention(
            queries: q512, keys: k512, values: v512, scale: scale512, mask: .causal)
        eval(blocked512, ref512)
        let diff512 = MLX.abs(blocked512 - ref512).max().item(Float.self)
        #expect(diff512 < 1e-4, "causalAttentionBlocked D=512 FP32 diff: \(diff512)")

        let blocked512BF16 = Gemma4BlockSDCM.causalAttentionBlocked(
            queries: q512.asType(.bfloat16), keys: k512.asType(.bfloat16),
            values: v512.asType(.bfloat16), scale: scale512, blockSize: 128)
        let ref512BF16 = MLXFast.scaledDotProductAttention(
            queries: q512.asType(.bfloat16), keys: k512.asType(.bfloat16),
            values: v512.asType(.bfloat16), scale: scale512, mask: .causal)
        eval(blocked512BF16, ref512BF16)
        let diff512BF16 = MLX.abs(
            blocked512BF16.asType(.float32) - ref512BF16.asType(.float32)
        ).max().item(Float.self)
        #expect(diff512BF16 <= 0.01, "causalAttentionBlocked D=512 BF16 diff: \(diff512BF16)")

        // D=64 is inside MLX's fused Flash set and passes straight through.
        let scale64: Float = 1.0 / sqrt(64.0)
        let q64 = MLXRandom.normal([1, 4, L, 64]).asType(.float32)
        let k64 = MLXRandom.normal([1, 2, L, 64]).asType(.float32)
        let v64 = MLXRandom.normal([1, 2, L, 64]).asType(.float32)
        let pass64 = Gemma4BlockSDCM.causalAttentionBlocked(
            queries: q64, keys: k64, values: v64, scale: scale64, blockSize: 128)
        let ref64 = MLXFast.scaledDotProductAttention(
            queries: q64, keys: k64, values: v64, scale: scale64, mask: .causal)
        eval(pass64, ref64)
        #expect(MLX.abs(pass64 - ref64).max().item(Float.self) == 0)

        // Empty L=0 returns zero-length output cleanly.
        let emptyOut = Gemma4BlockSDCM.causalAttentionBlocked(
            queries: MLXArray.zeros([1, 4, 0, 512], dtype: .float32),
            keys: MLXArray.zeros([1, 2, 0, 512], dtype: .float32),
            values: MLXArray.zeros([1, 2, 0, 512], dtype: .float32),
            scale: scale512, blockSize: 128)
        #expect(emptyOut.shape == [1, 4, 0, 512])
    }
}

/// Test-only SwitchGLU subclass verifying `transformInput` / `transformHidden` hooks
/// run identically on `callAndMixedKWeightedReduce`.
private final class ScalingSwitchGLU: SwitchGLU {
    override func transformInput(_ x: MLXArray) -> MLXArray { x * Float(2.0) }
    override func transformHidden(_ x: MLXArray) -> MLXArray { x * Float(0.5) }
}

/// Wrapper `CBv2LayerCacheProvider` that deliberately omits
/// `supportsCompactedPromptKVWrite` (inheriting the fail-closed default `false`).
private final class IncapableCacheProvider: CBv2LayerCacheProvider, CBv2CompositionInvalidating {
    private let inner: CBv2LayerCacheBank

    init(_ inner: CBv2LayerCacheBank) {
        self.inner = inner
    }

    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        inner.layerCaches(rowStates: rowStates)
    }

    func invalidateBoundComposition() {
        inner.invalidateBoundComposition()
    }

    func releaseBoundRows() {
        inner.releaseBoundRows()
    }
}

/// Minimal `CBv2AttendingLayerCache` conformer relying on the default fail-closed
/// `supportsCompactedPromptKVWrite == false` and trapping `writeCompactedPromptKV`.
private final class DefaultAttendingLayerCache: CBv2AttendingLayerCache, KVCache {
    let layerIndex: Int
    let kind: CBv2LayerKind
    private var storedRows: [CBv2SequenceKV] = []

    init(layerIndex: Int, kind: CBv2LayerKind) {
        self.layerIndex = layerIndex
        self.kind = kind
    }

    var rows: [CBv2SequenceKV] { storedRows }
    func setRows(_ rows: [CBv2SequenceKV]) { storedRows = rows }
    var positionOffsets: MLXArray { MLXArray(storedRows.map { Int32($0.ropePositionOffset) }) }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        CBv2AttentionV1.updateAndAttend(
            rows: storedRows, kind: kind, queries: queries, keys: keys, values: values,
            scale: scale, sinks: sinks)
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache, queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        CBv2AttentionV1.attendBorrowing(
            sourceRows: source.rows, sourceKind: source.kind, kind: kind,
            queries: queries, scale: scale, sinks: sinks)
    }

    var offset: Int { storedRows.map { $0.absoluteOffset }.max() ?? 0 }
    var maxSize: Int? { nil }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("v2 models must call updateAndAttend, not KVCache.update")
    }
    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [] }
        set {}
    }
    var isTrimmable: Bool { false }
    @discardableResult
    func trim(_ n: Int) -> Int { 0 }
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode { .none }
    func copy() -> any KVCache { fatalError("copy() unsupported in test cache") }
    func innerState() -> [MLXArray] { [] }
}

private final class CompactionObservingSteppableModel: CBv2PrefillSteppableModel,
    @unchecked Sendable
{
    struct Observation: Sendable {
        let tokenCount: Int
        let scopeDisabled: Bool
        let compacted: Bool
        let usedApproximatePrefill: Bool
        let absoluteOffset: Int
        let ropeOffset: Int
    }

    private let inner: CBv2SteppableLanguageModelAdapter
    private let watchedLayer: Int
    private let lock = NSLock()
    private var recorded: [Observation] = []

    init(_ inner: CBv2SteppableLanguageModelAdapter, watchedLayer: Int) {
        self.inner = inner
        self.watchedLayer = watchedLayer
    }

    var observations: [Observation] {
        lock.withLock { recorded }
    }

    var supportsCompactedPromptPrefill: Bool {
        inner.supportsCompactedPromptPrefill
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        inner.forward(tokens: tokens, caches: caches)
    }

    func prefill(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        let disabled = CBv2PrefillCompactionScope.isDisabled
        let output = inner.prefill(
            tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches,
            requirement: requirement)
        if let row = caches[watchedLayer].rows.first {
            let obs = Observation(
                tokenCount: tokens.dim(1),
                scopeDisabled: disabled,
                compacted: row.isCompacted,
                usedApproximatePrefill: row.usedApproximatePrefill,
                absoluteOffset: row.absoluteOffset,
                ropeOffset: row.ropePositionOffset)
            lock.withLock { recorded.append(obs) }
        }
        return output
    }
}
