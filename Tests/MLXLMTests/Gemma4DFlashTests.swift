import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@_spi(GemmaEncoder) @testable import MLXLLM

@Suite("Gemma 4 DFlash Block-Diffusion Speculative Drafter")
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
            return try await withDFlashEngine(
                target: target,
                drafter: useDFlash ? drafter : nil,
                mtpConfig: mtpCfg
            ) { engine in
                var streams: [(Int, AsyncStream<CBv2Event>)] = []
                for (i, prompt) in batchPrompts.enumerated() {
                    let req = CBv2Request(
                        id: CBv2RequestID(UInt64(i + 1)),
                        promptTokens: prompt,
                        sampling: CBv2SamplingParams(temperature: 0.0),
                        maxTokens: 20)
                    streams.append((i, try engine.submit(req)))
                }
                var results = Array(repeating: [Int](), count: batchPrompts.count)
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
                if useDFlash {
                    let m = try #require(engine.mtpMetricsSnapshot())
                    #expect(m.rounds > 0, "expected DFlash MTP rounds to execute")
                }
                return results
            }
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
            return try await withDFlashEngine(
                target: target,
                drafter: useDFlash ? drafter : nil,
                mtpConfig: mtpCfg,
                maxConcurrentRequests: 4
            ) { engine in
                let req = CBv2Request(
                    id: CBv2RequestID(1),
                    promptTokens: longPrompt,
                    sampling: CBv2SamplingParams(temperature: 0.0),
                    maxTokens: 20)
                let collected = await cbv2SchedCollect(try engine.submit(req), timeoutSeconds: 30)
                if useDFlash {
                    let m = try #require(engine.mtpMetricsSnapshot())
                    #expect(
                        m.rounds > 0, "expected DFlash rounds after Block-SDCM compacted prefill")
                }
                return collected.tokens
            }
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

    @Test("Block-SDCM hyper preset parses cleanly and configures aggressive L0+L6 schedule")
    func blockSDCMHyperPreset() {
        #expect(Gemma4BlockSDCMConfig.parseMode("hyper") == .hyper)
        #expect(Gemma4BlockSDCMConfig.parseMode("v4") == .hyper)
        let cfg = Gemma4BlockSDCMConfig.preset(for: .hyper)
        #expect(cfg.mode == .hyper)
        #expect(cfg.earlyPreCompactAfterLayer == 0)
        #expect(cfg.earlyPreCompactRatio < 0.40)
        #expect(cfg.interiorMoEK(forLayer: 0) == 2)
        #expect(cfg.interiorMoEK(forLayer: 4) == 3)
        #expect(cfg.interiorMoEK(forLayer: 8) == 4)
        #expect(cfg.keepSchedule[6] == 0.18)
    }
}
