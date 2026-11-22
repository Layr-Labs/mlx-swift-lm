import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@_spi(GemmaEncoder) @testable import MLXLLM

@Suite("Gemma 4 DFlash Block-Diffusion Speculative Drafter", .serialized)
struct Gemma4DFlashTests {

    private func makeTinyGemma4AndDFlash(
        numTargetLayers: Int = 6,
        blockSize: Int = 16,
        targetLayerIds: [Int] = [1, 3, 5]
    ) throws -> (Gemma4TextModel, Gemma4DFlashDraftModel, Gemma4CBv2DFlashDrafter) {
        MLXRandom.seed(42)
        let layerTypes = (0 ..< numTargetLayers).map {
            ($0 == numTargetLayers - 1 || $0 == 5) ? "full_attention" : "sliding_attention"
        }
        let layerTypesJSON = layerTypes.map { "\"\($0)\"" }.joined(separator: ", ")
        let targetJSON = """
            {
                "model_type": "gemma4_text",
                "vocab_size": 256,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": \(numTargetLayers),
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "global_head_dim": 64,
                "sliding_window": 32,
                "layer_types": [\(layerTypesJSON)],
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "vocab_size_per_layer_input": 0,
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true
            }
            """
        let targetCfg = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(targetJSON.utf8))
        let target = Gemma4TextModel(targetCfg)
        target.train(false)
        eval(target)

        let dflashCfg = Gemma4DFlashConfiguration(
            hiddenSize: 128,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 64,
            intermediateSize: 256,
            vocabSize: 256,
            rmsNormEps: 1e-6,
            ropeTheta: 1_000_000.0,
            maxPositionEmbeddings: 1024,
            numTargetLayers: numTargetLayers,
            blockSize: blockSize,
            layerTypes: ["sliding_attention", "full_attention"],
            slidingWindow: 32,
            finalLogitSoftcapping: 30.0,
            isCausal: nil,
            maskTokenId: 4,
            targetLayerIds: targetLayerIds)
        let dflash = try Gemma4DFlashDraftModel(dflashCfg)
        dflash.train(false)
        eval(dflash)

        let drafter = try Gemma4CBv2DFlashDrafter(
            drafter: dflash, target: target, maxDraftTokens: blockSize - 1, maxSpeculativeBatch: 4)
        return (target, dflash, drafter)
    }

    private func withDFlashEngine<T>(
        target: Gemma4TextModel,
        drafter: Gemma4CBv2DFlashDrafter?,
        mtpConfig: CBv2MTPConfig,
        maxConcurrentRequests: Int = 8,
        prefillChunkSize: Int = 512,
        capacityBytes: Int = 32 << 20,
        segmentSizeBytes: Int? = nil,
        fallbackSeed: UInt64 = 42,
        _ body: (EngineV2) async throws -> T
    ) async throws -> T {
        let kinds = target.cbv2LayerKinds
        let kvBackend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                pageSize: 16,
                capacityBytes: capacityBytes,
                dtype: .float32,
                maxPrefillChunk: prefillChunkSize,
                nominalMaxSequenceLength: 256,
                segmentSizeBytes: segmentSizeBytes))
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: kvBackend,
            cacheProvider: CBv2LayerCacheBank(caches: kvBackend.makeLayerCaches()),
            sampler: CBv2DefaultSampler(fallbackSeed: fallbackSeed),
            detokenizerFactory: CBv2NullDetokenizerFactory(),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrentRequests,
                maxBatchedTokensPerStep: prefillChunkSize,
                prefillChunkSize: prefillChunkSize,
                maxWaiting: max(4, maxConcurrentRequests * 2)),
            mtpDrafter: drafter,
            mtpConfig: mtpConfig)
        do {
            let result = try await body(engine)
            await engine.shutdown()
            return result
        } catch {
            await engine.shutdown()
            throw error
        }
    }

    private func collectGreedyBatch(
        target: Gemma4TextModel,
        drafter: Gemma4CBv2DFlashDrafter?,
        mtpConfig: CBv2MTPConfig,
        prompts: [[Int]],
        maxTokensPerRow: [Int],
        maxConcurrentRequests: Int = 8,
        expectMTPRounds: Bool = false
    ) async throws -> [[Int]] {
        try await withDFlashEngine(
            target: target,
            drafter: drafter,
            mtpConfig: mtpConfig,
            maxConcurrentRequests: maxConcurrentRequests
        ) { engine in
            var streams: [(Int, AsyncStream<CBv2Event>)] = []
            for (i, prompt) in prompts.enumerated() {
                let req = CBv2Request(
                    id: CBv2RequestID(UInt64(i + 1)),
                    promptTokens: prompt,
                    sampling: CBv2SamplingParams(temperature: 0.0),
                    maxTokens: maxTokensPerRow[i])
                streams.append((i, try engine.submit(req)))
            }
            var results = Array(repeating: [Int](), count: prompts.count)
            await withTaskGroup(of: (Int, [Int]).self) { group in
                for (i, stream) in streams {
                    group.addTask {
                        let collected = await cbv2SchedCollect(stream, timeoutSeconds: 30)
                        return (i, collected.tokens)
                    }
                }
                for await (idx, toks) in group {
                    results[idx] = toks
                }
            }
            if expectMTPRounds {
                let m = try #require(engine.mtpMetricsSnapshot())
                #expect(m.rounds > 0, "expected DFlash MTP rounds to execute")
            }
            return results
        }
    }

    @Test(
        "DFlash parallel block draft + CBv2 EngineLoopV2 matches target-only bit-identically at B=1 and B=4 (block_size=16)"
    )
    func cbv2EndToEndBitExactParity() async throws {
        let (target, _, drafter) = try makeTinyGemma4AndDFlash(blockSize: 16)
        #expect(target.cbv2SupportsRequestStatefulMTP)

        let prompts: [[Int]] = [
            [2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [2, 30, 31, 32, 33, 34, 35, 36, 37, 38],
            [2, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
            [2, 70, 71, 72, 73, 74, 75, 76, 77],
        ]

        func runEngine(useDFlash: Bool, batchPrompts: [[Int]]) async throws -> [[Int]] {
            let mtpCfg = CBv2MTPConfig(
                enabled: useDFlash,
                maxDraftTokens: 15,
                maxSpeculativeBatch: 8,
                fixedDraftTokens: 15,
                verificationMode: .automatic,
                maxAutomaticRectangularTokens: 32,
                acceptance: .exact,
                allowExtendedBlockDraftDepth: true)
            return try await collectGreedyBatch(
                target: target,
                drafter: useDFlash ? drafter : nil,
                mtpConfig: mtpCfg,
                prompts: batchPrompts,
                maxTokensPerRow: Array(repeating: 20, count: batchPrompts.count),
                expectMTPRounds: useDFlash)
        }

        // B = 1 bit-exact greedy parity with depth = 15 (block_size = 16).
        let refB1 = try await runEngine(useDFlash: false, batchPrompts: [prompts[0]])
        let dflashB1 = try await runEngine(useDFlash: true, batchPrompts: [prompts[0]])
        #expect(dflashB1[0].count == 20)
        #expect(dflashB1[0] == refB1[0])

        // B = 4 continuous-batching bit-exact greedy parity (maxAutomaticRectangularTokens=32 -> depth=7).
        let refB4 = try await runEngine(useDFlash: false, batchPrompts: prompts)
        let dflashB4 = try await runEngine(useDFlash: true, batchPrompts: prompts)
        for i in 0 ..< prompts.count {
            #expect(dflashB4[i].count == 20)
            #expect(dflashB4[i] == refB4[i], "B=4 row \(i) diverged from exact target-only")
        }
    }

    @Test(
        "DFlash composes cleanly with Block-SDCM compacted prefill and stays bit-exact vs Block-SDCM target-only"
    )
    func blockSDCMPlusDFlashBitExactParity() async throws {
        // 8 layers with full attention at layer 5 and 7 so Block-SDCM compacts at layer 6,
        // and DFlash taps [1, 4, 6] (all < final layer 7).
        let (target, _, drafter) = try makeTinyGemma4AndDFlash(
            numTargetLayers: 8, blockSize: 16, targetLayerIds: [1, 4, 6])
        var sdcm = Gemma4BlockSDCMConfig.preset(for: .fast)
        sdcm.minPromptTokens = 64
        sdcm.blockSize = 8
        sdcm.prefixKeep = 8
        sdcm.tailKeep = 16
        sdcm.earlyPreCompactAfterLayer = nil
        sdcm.densityGuardEnabled = false
        sdcm.keepSchedule = [6: 0.5]
        sdcm.earlySparseFullLayer = nil
        target.model.blockSDCMConfig = sdcm

        let longPrompt = [2] + (0 ..< 95).map { 10 + ($0 % 200) }

        func runOnce(useDFlash: Bool) async throws -> [Int] {
            let mtpCfg = CBv2MTPConfig(
                enabled: useDFlash,
                maxDraftTokens: 15,
                maxSpeculativeBatch: 8,
                fixedDraftTokens: 15,
                verificationMode: .automatic,
                maxAutomaticRectangularTokens: 32,
                acceptance: .exact,
                allowExtendedBlockDraftDepth: true)
            return try await collectGreedyBatch(
                target: target,
                drafter: useDFlash ? drafter : nil,
                mtpConfig: mtpCfg,
                prompts: [longPrompt],
                maxTokensPerRow: [20],
                maxConcurrentRequests: 4,
                expectMTPRounds: useDFlash)[0]
        }

        let ref = try await runOnce(useDFlash: false)
        let actual = try await runOnce(useDFlash: true)
        #expect(actual.count == 20)
        #expect(actual == ref)
    }

    @Test(
        "DFlash supports temperature>0 target-prefix acceptance and B>maxSpeculativeBatch transitions"
    )
    func temperatureAndBatchOverflowParity() async throws {
        let (target, _, drafter) = try makeTinyGemma4AndDFlash(blockSize: 16)
        let mtpCfg = CBv2MTPConfig(
            enabled: true,
            maxDraftTokens: 7,
            maxSpeculativeBatch: 4,
            fixedDraftTokens: 7,
            verificationMode: .automatic,
            maxAutomaticRectangularTokens: 32,
            acceptance: .exact,
            allowExtendedBlockDraftDepth: false)
        try await withDFlashEngine(
            target: target,
            drafter: drafter,
            mtpConfig: mtpCfg,
            fallbackSeed: 7
        ) { engine in
            // 6 concurrent requests (> maxSpeculativeBatch=4) with staggered maxTokens and temperature>0:
            // starts in ordinary decode history, then resumes DFlash rounds as early rows finish.
            var streams: [(Int, Int, AsyncStream<CBv2Event>)] = []
            for i in 0 ..< 6 {
                let maxTok = i < 3 ? 6 : 20
                let req = CBv2Request(
                    id: CBv2RequestID(UInt64(i + 1)),
                    promptTokens: [2, 10 + i, 20 + i, 30 + i, 40 + i, 50 + i],
                    sampling: CBv2SamplingParams(
                        temperature: 0.7, topP: 0.95, seed: UInt64(100 + i)),
                    maxTokens: maxTok)
                streams.append((i, maxTok, try engine.submit(req)))
            }
            await withTaskGroup(of: (Int, Int, Int).self) { group in
                for (i, expected, stream) in streams {
                    group.addTask {
                        let collected = await cbv2SchedCollect(stream, timeoutSeconds: 30)
                        return (i, expected, collected.tokens.count)
                    }
                }
                for await (idx, expected, got) in group {
                    #expect(got == expected, "row \(idx) expected \(expected) tokens, got \(got)")
                }
            }
            let m = try #require(engine.mtpMetricsSnapshot())
            #expect(
                m.rounds > 0,
                "expected DFlash rounds once batch dropped within maxSpeculativeBatch")
        }
    }

    @Test(
        "DFlash segmented-pool admission, multi-chunk prefill (L>=128 final-layer tap), and owner-scoped tap deinit"
    )
    func segmentedAdmissionMultiChunkAndOwnerScopedTap() async throws {
        // 1. Segmented paged pool (`segmentSizeBytes: 1 << 20`) exercises allocator-footprint
        //    projection via `requestStateAllocationSpecs` (must NOT overcharge 64 MiB/tok).
        let (target, dflash, drafterA) = try makeTinyGemma4AndDFlash(
            numTargetLayers: 6, blockSize: 16, targetLayerIds: [1, 3, 5])
        #expect(drafterA.requestStateAllocationSpecs != nil)

        // 2. Multi-chunk prefill (prompt=160 > prefillChunkSize=128 >= gemma4PrefillTailMinChunk)
        //    with final layer 5 tapped verifies both `.evaluationOnly` sync anchor and
        //    final-layer narrowing suppression.
        let prompt160 = [2] + (0 ..< 159).map { 10 + ($0 % 200) }
        let mtpCfg = CBv2MTPConfig(
            enabled: true,
            maxDraftTokens: 7,
            maxSpeculativeBatch: 2,
            fixedDraftTokens: 7,
            verificationMode: .automatic,
            maxAutomaticRectangularTokens: 32,
            acceptance: .exact)
        try await withDFlashEngine(
            target: target,
            drafter: drafterA,
            mtpConfig: mtpCfg,
            maxConcurrentRequests: 2,
            prefillChunkSize: 128,
            capacityBytes: 64 << 20,
            segmentSizeBytes: 1 << 20
        ) { engine in
            let collected = await cbv2SchedCollect(
                try engine.submit(
                    CBv2Request(
                        id: CBv2RequestID(1),
                        promptTokens: prompt160,
                        sampling: CBv2SamplingParams(temperature: 0.0),
                        maxTokens: 12)),
                timeoutSeconds: 30)
            #expect(collected.tokens.count == 12)
            let m = try #require(engine.mtpMetricsSnapshot())
            #expect(m.rounds > 0)
        }

        // 3. Cross-instance tap ownership: binding drafterB then releasing drafterA must NOT
        //    clobber drafterB's active context tap.
        var optionalA: Gemma4CBv2DFlashDrafter? = drafterA
        let drafterB = try Gemma4CBv2DFlashDrafter(drafter: dflash, target: target)
        #expect(target.cbv2SupportsRequestStatefulMTP)
        optionalA = nil
        _ = optionalA
        #expect(
            target.cbv2SupportsRequestStatefulMTP, "drafterA deinit must not clear drafterB tap")
        withExtendedLifetime(drafterB) {}
    }

    @Test("Block-SDCM hyper preset parses cleanly and configures Pre-L0 32-tok + L6 schedule")
    func blockSDCMHyperPreset() {
        #expect(Gemma4BlockSDCMConfig.parseMode("hyper") == .hyper)
        #expect(Gemma4BlockSDCMConfig.parseMode("v4") == .hyper)
        let cfg = Gemma4BlockSDCMConfig.preset(for: .hyper)
        #expect(cfg.mode == .hyper)
        #expect(cfg.earlyPreCompactAfterLayer == -1)
        #expect(cfg.earlyPreCompactBlockSize == 32)
        #expect(cfg.earlyPreCompactRatio <= 0.35)
        #expect(cfg.interiorMoEK(forLayer: 0) == 2)
        #expect(cfg.interiorMoEK(forLayer: 4) == 3)
        #expect(cfg.interiorMoEK(forLayer: 5) == 4)
        #expect(cfg.interiorMoEK(forLayer: 8) == 4)
        #expect(cfg.keepSchedule[6] == 0.16)
    }

    @Test(
        "Pre-L0 hyper compaction, fused DFlash QKV/GateUp + stepped cache, and Hi4 sparse ordinary history stay bit-exact"
    )
    func preLayer0HyperFusedDFlashAndSparseHistoryParity() async throws {
        // 1. Pre-Layer-0 (-1) Block-SDCM hyper + fused DFlash QKV/GateUp + stepped cache parity.
        let (target, dflash, drafter) = try makeTinyGemma4AndDFlash(
            numTargetLayers: 8, blockSize: 16, targetLayerIds: [1, 4, 6])
        dflash.prepareFusions()
        eval(dflash)

        var sdcm = Gemma4BlockSDCMConfig.preset(for: .hyper)
        sdcm.minPromptTokens = 64
        sdcm.blockSize = 8
        sdcm.earlyPreCompactBlockSize = 8
        sdcm.prefixKeep = 8
        sdcm.tailKeep = 16
        sdcm.densityGuardEnabled = false
        sdcm.keepSchedule = [6: 0.45]
        sdcm.earlySparseFullLayer = nil
        target.model.blockSDCMConfig = sdcm

        let longPrompt = [2] + (0 ..< 127).map { 10 + ($0 % 200) }
        func runPreL0(useDFlash: Bool) async throws -> [Int] {
            let mtpCfg = CBv2MTPConfig(
                enabled: useDFlash,
                maxDraftTokens: 7,
                maxSpeculativeBatch: 2,
                fixedDraftTokens: 7,
                verificationMode: .automatic,
                maxAutomaticRectangularTokens: 32,
                acceptance: .exact)
            return try await collectGreedyBatch(
                target: target,
                drafter: useDFlash ? drafter : nil,
                mtpConfig: mtpCfg,
                prompts: [longPrompt],
                maxTokensPerRow: [20],
                expectMTPRounds: useDFlash)[0]
        }
        let refPreL0 = try await runPreL0(useDFlash: false)
        let actualPreL0 = try await runPreL0(useDFlash: true)
        #expect(actualPreL0.count == 20)
        #expect(actualPreL0 == refPreL0)

        // 2. Quantized 8-bit Hi4/Lo4 sparse decode with ordinary MTP history (`B=4 > maxSpecBatch=2`).
        target.model.blockSDCMConfig = .off
        quantize(model: target) { path, module in
            guard path.hasSuffix("embed_tokens"), module is Embedding else { return nil }
            return (groupSize: 64, bits: 8, mode: .affine)
        }
        if let qEmb = target.model.embedTokens as? QuantizedEmbedding,
            let hi4 = Gemma4Hi4Lo4Embedding.from(qEmb, samplingStrategy: .exact)
        {
            target.model.update(modules: ModuleChildren.unflattened(["embed_tokens": hi4]))
        }
        eval(target)
        #expect(target.supportsSparseExactHeadSampling)
        #expect(target.supportsSparseHeadWithMTPHidden)

        let prompts = [
            [2, 10, 20, 30, 40, 50],
            [2, 15, 25, 35, 45, 55, 65],
            [2, 12, 22, 32, 42],
            [2, 19, 29, 39, 49, 59],
        ]
        func runB4(useDFlash: Bool) async throws -> [[Int]] {
            let mtpCfg = CBv2MTPConfig(
                enabled: useDFlash,
                maxDraftTokens: 5,
                maxSpeculativeBatch: 2,
                fixedDraftTokens: 5,
                verificationMode: .automatic,
                maxAutomaticRectangularTokens: 32,
                acceptance: .exact)
            return try await collectGreedyBatch(
                target: target,
                drafter: useDFlash ? drafter : nil,
                mtpConfig: mtpCfg,
                prompts: prompts,
                maxTokensPerRow: [8, 8, 20, 20],
                expectMTPRounds: useDFlash)
        }
        let refB4 = try await runB4(useDFlash: false)
        let actualB4 = try await runB4(useDFlash: true)
        for i in 0 ..< prompts.count {
            #expect(actualB4[i] == refB4[i], "Hi4 ordinary-history -> DFlash B=4 row \(i) diverged")
        }
    }

    @Test(
        "DFlash fusions match unfused numerically (Linear & 4-bit), stay clean on repeat/reload, and stepped cache discardRound rolls back cleanly across growthStep=256"
    )
    func dflashFusionParityInvalidationAndSteppedRollback() throws {
        // 1. Direct numerical parity (Linear + 4-bit QuantizedLinear), idempotency, and invalidation.
        for quant4 in [false, true] {
            let (_, dflash, _) = try makeTinyGemma4AndDFlash(
                numTargetLayers: 6, blockSize: 16, targetLayerIds: [1, 3, 5])
            if quant4 {
                quantize(model: dflash) { _, module in
                    guard let linear = module as? Linear, !(module is QuantizedLinear),
                        linear.weight.dim(1) % 64 == 0
                    else { return nil }
                    return (groupSize: 64, bits: 4, mode: .affine)
                }
                dflash.train(false)
                eval(dflash)
            }
            let x = MLXRandom.normal([1, 6, 128]).asType(.float32)
            let ctx = MLXRandom.normal([1, 20, 128]).asType(.float32)
            let refOut = dflash.forwardDraftBlock(
                embeddedBlock: x, projectedContext: ctx, caches: dflash.makeCaches())
            eval(refOut)

            let beforeKeys = Set(dflash.parameters().flattened().map(\.0))
            dflash.prepareFusions()
            dflash.prepareFusions()  // idempotent second call
            eval(dflash)
            let afterKeys = Set(dflash.parameters().flattened().map(\.0))
            #expect(
                afterKeys == beforeKeys, "prepareFusions must not pollute Module parameter tree")
            for layer in dflash.layers {
                #expect(!layer.selfAttn.qProj.training)
                #expect(!layer.mlp.gateProj.training)
            }

            let fusedOut = dflash.forwardDraftBlock(
                embeddedBlock: x, projectedContext: ctx, caches: dflash.makeCaches())
            eval(fusedOut)
            let maxDiff = abs(fusedOut - refOut).max().item(Float.self)
            #expect(
                maxDiff < 1e-4, "fused vs unfused DFlash diverged (quant4=\(quant4)): \(maxDiff)")

            // Reloading parameters with verify: .all must succeed and invalidate fused caches.
            let origParams = Dictionary(uniqueKeysWithValues: dflash.parameters().flattened())
            try dflash.update(parameters: ModuleParameters.unflattened(origParams), verify: .all)
            if !quant4 {
                // Re-prepare so each mutation tests invalidation of a live fused projection.
                dflash.prepareFusions()
                eval(dflash)
                let zeroGate = MLXArray.zeros(like: dflash.layers[0].mlp.gateProj.weight)
                try dflash.update(
                    parameters: ModuleParameters.unflattened([
                        "layers.0.mlp.gate_proj.weight": zeroGate
                    ]),
                    verify: .none)
                let afterGateMutate = dflash.forwardDraftBlock(
                    embeddedBlock: x, projectedContext: ctx, caches: dflash.makeCaches())
                eval(afterGateMutate)
                let gateDelta = abs(afterGateMutate - fusedOut).max().item(Float.self)
                #expect(
                    gateDelta > 1e-4, "post-fusion update(parameters:) must invalidate fusedGateUp")

                dflash.prepareFusions()
                eval(dflash)
                let zeroQ = MLXArray.zeros(like: dflash.layers[0].selfAttn.qProj.weight)
                try dflash.update(
                    parameters: ModuleParameters.unflattened([
                        "layers.0.self_attn.q_proj.weight": zeroQ
                    ]),
                    verify: .none)
                let afterQMutate = dflash.forwardDraftBlock(
                    embeddedBlock: x, projectedContext: ctx, caches: dflash.makeCaches())
                eval(afterQMutate)
                let qDelta = abs(afterQMutate - afterGateMutate).max().item(Float.self)
                #expect(qDelta > 1e-4, "post-fusion update(parameters:) must invalidate fusedQKV")
            }
        }

        // 2. Stepped full-attention cache across growthStep=256 + pre-eval discardRound rollback.
        let (target, _, drafter) = try makeTinyGemma4AndDFlash(
            numTargetLayers: 6, blockSize: 16, targetLayerIds: [1, 3, 5])
        try withExtendedLifetime(target) {
            let state = drafter.makeRequestState()
            let typed = try #require(state as? Gemma4DFlashRequestState)
            let initHidden = MLXRandom.normal([1, 250, 128]).asType(.float32)
            drafter.observeCommittedTarget(
                CBv2MTPCommittedTargetObservation(
                    tokens: MLXArray.zeros([1, 250], dtype: .int32), hidden: initHidden),
                requestState: state)
            let seed = MLXArray([Int32(10)]).reshaped([1, 1])
            let seedH = MLXRandom.normal([1, 1, 128]).asType(.float32)
            _ = try #require(
                drafter.draftBlock(
                    seedTokens: seed, seedHidden: seedH, depth: 5, requestState: state))
            eval(drafter.evaluationTargets(for: state))
            drafter.finalizeRound(
                requestState: state, confirmedInputTokens: 3,
                committedDraftTokens: MLXArray([Int32(1), 2]),
                committedTargetHidden: MLXRandom.normal([1, 12, 128]).asType(.float32))
            let fullCache = typed.caches[1]
            #expect(fullCache.used == 250)
            #expect(fullCache.offset == 250)
            let snapBeforeDiscard = fullCache.activeKeys!.asArray(Float.self)

            // Start round crossing 256 boundary (250 + 13 = 263), then discard BEFORE eval!
            _ = try #require(
                drafter.draftBlock(
                    seedTokens: seed, seedHidden: seedH, depth: 5, requestState: state))
            #expect(fullCache.stagedAdded == 13)
            drafter.discardRound(requestState: state)
            #expect(fullCache.used == 250)
            #expect(fullCache.offset == 250)
            #expect(fullCache.stagedKeys == nil)
            #expect(fullCache.activeKeys!.asArray(Float.self) == snapBeforeDiscard)

            // Now complete the round and commit past growthStep=256 (250 + 13 = 263 -> cap 512).
            _ = try #require(
                drafter.draftBlock(
                    seedTokens: seed, seedHidden: seedH, depth: 5, requestState: state))
            eval(drafter.evaluationTargets(for: state))
            drafter.finalizeRound(
                requestState: state, confirmedInputTokens: 2,
                committedDraftTokens: MLXArray([Int32(1)]),
                committedTargetHidden: MLXRandom.normal([1, 1, 128]).asType(.float32))
            #expect(fullCache.used == 263)
            #expect(fullCache.keys!.dim(2) == 512)
            drafter.releaseRequestState(state)
        }
    }

    @Test("Pre-L0 hyper compaction handles heterogeneous right-padded Mode B with sparse L5")
    func preLayer0HeterogeneousModeBParity() throws {
        #expect(
            Gemma4BlockSDCMConfig.preset(for: .hyper).resolvedEarlyCompactLayer(layerCount: 0)
                == nil)
        let (target, _, _) = try makeTinyGemma4AndDFlash(
            numTargetLayers: 8, blockSize: 16, targetLayerIds: [1, 4, 6])
        var sdcm = Gemma4BlockSDCMConfig.preset(for: .hyper)
        sdcm.minPromptTokens = 48
        sdcm.blockSize = 8
        sdcm.earlyPreCompactBlockSize = 8
        sdcm.prefixKeep = 8
        sdcm.tailKeep = 16
        sdcm.densityGuardEnabled = false
        sdcm.keepSchedule = [6: 0.45]
        sdcm.earlySparseFullLayer = 5
        sdcm.sparsePrefixTokens = 8
        sdcm.sparseLocalWindow = 16
        sdcm.sparseTailTokens = 16
        target.model.blockSDCMConfig = sdcm

        let lengths = [64, 72, 76, 80]
        let padded = 80
        var rows: [[Int32]] = []
        var flat: [Int32] = []
        for (b, len) in lengths.enumerated() {
            let row = (0 ..< len).map { Int32(10 + ((b * 37 + $0) % 200)) }
            rows.append(row)
            flat.append(contentsOf: row)
            flat.append(contentsOf: Array(repeating: Int32(0), count: padded - len))
        }
        let tokens = MLXArray(flat, [4, padded])
        let kinds = target.cbv2LayerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                pageSize: 16, capacityBytes: 32 << 20, dtype: .float32,
                maxPrefillChunk: 128, nominalMaxSequenceLength: 256))
        let seqStates = try lengths.map { len in
            try backend.makeSequenceState(layerKinds: kinds, promptLength: len, maxLength: 256)
        }
        defer { for s in seqStates { backend.release(s) } }
        let layerCaches = backend.makeLayerCaches()
        for (l, cache) in layerCaches.enumerated() {
            cache.setRows(seqStates.map { $0[l] as! PagedSequenceKV })
        }
        let caches: [KVCache?] = layerCaches.map { $0 as KVCache? }
        let (postNorm, _) = target.model.forwardBlockSDCMTrunk(
            target.model.embedTokens(tokens) * Float(128).squareRoot(),
            inputTokenIDRows: rows,
            originalLengths: lengths,
            fullCache: caches,
            sdcm: sdcm,
            isExpertPrefill: false)
        eval(postNorm)
        #expect(postNorm.shape == [4, 1, 128])
        #expect(!any(isNaN(postNorm)).item(Bool.self))
        for (b, len) in lengths.enumerated() {
            for layerState in seqStates[b] {
                #expect(layerState?.ropePositionOffset == len)
            }
        }
    }
}
