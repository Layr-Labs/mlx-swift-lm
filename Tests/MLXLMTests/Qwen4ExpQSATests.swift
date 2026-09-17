import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class Qwen4ExpQSATests: XCTestCase {
    func testTextPositionsAcceptAbsentIds() {
        XCTAssertTrue(Qwen4ExpTextPositions.isBatchOneText(nil, length: 8))
        XCTAssertTrue(Qwen4ExpTextPositions.isBatchOneText(nil, length: 2049))
    }

    func testSanitizeKeepsWeightScaleAndDropsComputedPLETables() {
        XCTAssertFalse(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
                mmapPLE: true))
        XCTAssertTrue(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "language_model.model.layers.1.ple.ple_embedding.layer_multipliers",
                mmapPLE: true))
        XCTAssertTrue(Qwen4ExpWeightSanitizer.shouldDrop("mtp.layers.0.self_attn.q_proj.weight", mmapPLE: true))
        XCTAssertTrue(Qwen4ExpWeightSanitizer.shouldDrop("vision_tower.pos_embed.weight", mmapPLE: true))
        XCTAssertFalse(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "vision_tower.pos_embed.weight", mmapPLE: true, keepVisionTower: true))
        XCTAssertTrue(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "mtp.fc_embedding.weight", mmapPLE: true, keepVisionTower: true))
        XCTAssertFalse(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "language_model.model.hyper_connection_mixer.hc_norm.weight",
                mmapPLE: true))
        XCTAssertFalse(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight",
                mmapPLE: true))
        XCTAssertTrue(Qwen4ExpWeightSanitizer.shouldDrop("mtp.hyper_connection_mixer.hc_norm.weight", mmapPLE: true))
        XCTAssertFalse(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                mmapPLE: true))
        XCTAssertFalse(
            Qwen4ExpWeightSanitizer.shouldDrop(
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales",
                mmapPLE: true))
    }

    func testNativeQSAEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpNativeSparseGQA.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpNativeSparseGQA.isEnabled(environment: [
                Qwen4ExpNativeSparseGQA.envFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpNativeSparseGQA.isEnabled(environment: [
                Qwen4ExpNativeSparseGQA.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpNativeSparseGQA.isEnabled(environment: [
                Qwen4ExpNativeSparseGQA.envFlag: "off"
            ]))
    }

    func testSteelQSAEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpNativeSparseGQA.steelEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpNativeSparseGQA.steelEnabled(environment: [
                Qwen4ExpNativeSparseGQA.steelEnvFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpNativeSparseGQA.steelEnabled(environment: [
                Qwen4ExpNativeSparseGQA.steelEnvFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpNativeSparseGQA.steelEnabled(environment: [
                Qwen4ExpNativeSparseGQA.steelEnvFlag: "off"
            ]))
        XCTAssertEqual(Qwen4ExpNativeSparseGQA.steelThreads, 64)
        XCTAssertEqual(
            Qwen4ExpNativeSparseGQA.resolvedSteelTiles(environment: [:]).keyTile, 64)
        XCTAssertEqual(
            Qwen4ExpNativeSparseGQA.resolvedSteelTiles(environment: [:]).dimensionTile, 64)
        let bk128 = Qwen4ExpNativeSparseGQA.resolvedSteelTiles(
            environment: [Qwen4ExpNativeSparseGQA.steelBK128EnvFlag: "1"])
        XCTAssertEqual(bk128.keyTile, 128)
        XCTAssertEqual(bk128.dimensionTile, 32)
        XCTAssertFalse(Qwen4ExpNativeSparseGQA.steelBK128Enabled(environment: [:]))
    }

    func testNativeQSAGeometryIsFusionContract() {
        XCTAssertTrue(
            Qwen4ExpNativeSparseGQA.matchesGeometry(
                queryHeads: 24, kvHeads: 2, headDim: 256, selectedWidth: 512))
        XCTAssertFalse(
            Qwen4ExpNativeSparseGQA.matchesGeometry(
                queryHeads: 16, kvHeads: 2, headDim: 256, selectedWidth: 512))
        XCTAssertFalse(
            Qwen4ExpNativeSparseGQA.matchesGeometry(
                queryHeads: 24, kvHeads: 2, headDim: 128, selectedWidth: 512))
        XCTAssertFalse(
            Qwen4ExpNativeSparseGQA.matchesGeometry(
                queryHeads: 24, kvHeads: 2, headDim: 256, selectedWidth: 256))
        XCTAssertTrue(Qwen4ExpNativeSparseGQA.matchesDtype(.float16))
        XCTAssertTrue(Qwen4ExpNativeSparseGQA.matchesDtype(.bfloat16))
        XCTAssertFalse(Qwen4ExpNativeSparseGQA.matchesDtype(.float32))
        XCTAssertFalse(Qwen4ExpNativeSparseGQA.matchesDtype(.int32))
        XCTAssertEqual(Qwen4ExpNativeSparseGQA.activationType(.float32), .bfloat16)
        XCTAssertEqual(Qwen4ExpNativeSparseGQA.activationType(.bfloat16), .bfloat16)
        XCTAssertEqual(Qwen4ExpGDNBlockedSeq.blockT(for: .bfloat16), 32)
        XCTAssertEqual(Qwen4ExpGDNBlockedSeq.blockT(for: .float32), 16)
    }

    func testNativeIndexerEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpNativeIndexer.scoresEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpNativeIndexer.scoresEnabled(environment: [
                Qwen4ExpNativeIndexer.scoreEnvFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpNativeIndexer.scoresEnabled(environment: [
                Qwen4ExpNativeIndexer.scoreEnvFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpNativeIndexer.scoresEnabled(environment: [
                Qwen4ExpNativeIndexer.scoreEnvFlag: "off"
            ]))
        XCTAssertTrue(Qwen4ExpNativeIndexer.topKEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpNativeIndexer.topKEnabled(environment: [
                Qwen4ExpNativeIndexer.topKEnvFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpNativeIndexer.topKEnabled(environment: [
                Qwen4ExpNativeIndexer.topKEnvFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpNativeIndexer.topKEnabled(environment: [
                Qwen4ExpNativeIndexer.topKEnvFlag: "off"
            ]))
    }

    func testSteelIndexerIsOptInAfter128KMiss() {
        XCTAssertFalse(Qwen4ExpSteelIndexer.isEnabled(environment: [:]))
        XCTAssertFalse(
            Qwen4ExpSteelIndexer.isEnabled(environment: [
                Qwen4ExpSteelIndexer.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpSteelIndexer.isEnabled(environment: [
                Qwen4ExpSteelIndexer.envFlag: "off"
            ]))
        XCTAssertTrue(
            Qwen4ExpSteelIndexer.isEnabled(environment: [
                Qwen4ExpSteelIndexer.envFlag: "1"
            ]))
        XCTAssertTrue(
            Qwen4ExpSteelIndexer.isEnabled(environment: [
                Qwen4ExpSteelIndexer.envFlag: "on"
            ]))
        XCTAssertEqual(Qwen4ExpNativeIndexer.steelEnvFlag, Qwen4ExpSteelIndexer.envFlag)
    }

    func testSteelIndexerTilePickMatchesFusion() {
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 1).blockM, 8)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 8).blockM, 8)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 16).blockM, 16)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 17).blockM, 64)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 8192).blockM, 64)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 1).warpsM, 1)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 16).warpsM, 1)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 17).warpsM, 2)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 1).threads, 64)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 16).threads, 64)
        XCTAssertEqual(Qwen4ExpSteelIndexer.tile(queryTokens: 17).threads, 128)
    }

    func testNativeIndexerGeometryIsFusionContract() {
        XCTAssertTrue(
            Qwen4ExpNativeIndexer.matchesScoreGeometry(queryHeads: 4, headDim: 128))
        XCTAssertFalse(
            Qwen4ExpNativeIndexer.matchesScoreGeometry(queryHeads: 64, headDim: 128))
        XCTAssertFalse(
            Qwen4ExpNativeIndexer.matchesScoreGeometry(queryHeads: 4, headDim: 256))
        XCTAssertEqual(Qwen4ExpNativeIndexer.indexerHeads, 4)
        XCTAssertEqual(Qwen4ExpNativeIndexer.headDim, 128)
        XCTAssertEqual(Qwen4ExpNativeIndexer.topK, 512)
        XCTAssertEqual(Qwen4ExpNativeIndexer.maskRatio, 4)
        XCTAssertEqual(Qwen4ExpNativeIndexer.tileM, 64)
        XCTAssertEqual(Qwen4ExpNativeIndexer.tileN, 64)
        XCTAssertEqual(Qwen4ExpNativeIndexer.scoreThreads, 128)
        XCTAssertEqual(Qwen4ExpNativeIndexer.topKThreads, 256)
    }

    func testGatheredQueryChunkBumpsForNative() {
        XCTAssertEqual(Qwen4ExpGatheredQSA.queryChunk(keyTokens: 2500, environment: [:]), 32)
        XCTAssertEqual(
            Qwen4ExpGatheredQSA.queryChunk(keyTokens: 2500, native: true, environment: [:]), 2048)
        XCTAssertEqual(Qwen4ExpGatheredQSA.queryChunk(keyTokens: 50_000, environment: [:]), 128)
        XCTAssertEqual(
            Qwen4ExpGatheredQSA.queryChunk(keyTokens: 50_000, native: true, environment: [:]),
            2048)
        XCTAssertEqual(
            Qwen4ExpGatheredQSA.queryChunk(
                keyTokens: 50_000, native: true,
                environment: [Qwen4ExpNativeSparseGQA.queryTileEnvFlag: "256"]),
            256)
        XCTAssertEqual(Qwen4ExpNativeSparseGQA.fusionQueryTile, 256)
        XCTAssertEqual(Qwen4ExpNativeSparseGQA.queryTile, 2048)
    }

    func testQSALayerSyncIsOptInAndClearsOnlyAt32K() {
        XCTAssertFalse(Qwen4ExpQSALayerSync.isEnabled(environment: [:]))
        XCTAssertFalse(
            Qwen4ExpQSALayerSync.isEnabled(environment: [
                Qwen4ExpQSALayerSync.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpQSALayerSync.isEnabled(environment: [
                Qwen4ExpQSALayerSync.envFlag: "off"
            ]))
        XCTAssertTrue(
            Qwen4ExpQSALayerSync.isEnabled(environment: [
                Qwen4ExpQSALayerSync.envFlag: "1"
            ]))
        XCTAssertTrue(
            Qwen4ExpQSALayerSync.isEnabled(environment: [
                Qwen4ExpQSALayerSync.envFlag: "on"
            ]))
        XCTAssertEqual(Qwen4ExpQSALayerSync.clearCacheKeyTokens, 32_768)
        XCTAssertFalse(
            Qwen4ExpQSALayerSync.shouldClearCache(keyTokens: 32_767, environment: [:]))
        XCTAssertFalse(
            Qwen4ExpQSALayerSync.shouldClearCache(
                keyTokens: 32_768, environment: [:]))
        XCTAssertFalse(
            Qwen4ExpQSALayerSync.shouldClearCache(
                keyTokens: 32_767,
                environment: [Qwen4ExpQSALayerSync.envFlag: "1"]))
        XCTAssertTrue(
            Qwen4ExpQSALayerSync.shouldClearCache(
                keyTokens: 32_768,
                environment: [Qwen4ExpQSALayerSync.envFlag: "1"]))
        XCTAssertTrue(
            Qwen4ExpQSALayerSync.shouldClearCache(
                keyTokens: 128_000,
                environment: [Qwen4ExpQSALayerSync.envFlag: "1"]))
    }

    func testQSAInvocationCountersReset() {
        Qwen4ExpQSAInvocation.reset()
        XCTAssertEqual(
            Qwen4ExpQSAInvocation.snapshot(),
            Qwen4ExpQSAInvocation.Snapshot(native: 0, portable: 0, dense: 0))
        Qwen4ExpQSAInvocation.recordDense()
        Qwen4ExpQSAInvocation.recordNative()
        Qwen4ExpQSAInvocation.recordPortable()
        Qwen4ExpQSAInvocation.recordDecode()
        Qwen4ExpQSAInvocation.recordVerify()
        XCTAssertEqual(
            Qwen4ExpQSAInvocation.snapshot().line,
            "qsa native=1 portable=1 dense=1 decode=1 verify=1")
        Qwen4ExpQSAInvocation.reset()
        XCTAssertEqual(Qwen4ExpQSAInvocation.snapshot().dense, 0)
        XCTAssertEqual(Qwen4ExpQSAInvocation.snapshot().decode, 0)
        XCTAssertEqual(Qwen4ExpQSAInvocation.snapshot().verify, 0)
    }

    func testDecodeCrossesBudgetAtFusionBoundary() {
        // 512 blocks of 4 = 2048 visible tokens is not yet sparse; 2052 is.
        XCTAssertFalse(
            Qwen4ExpGatheredQSA.decodeCrossesBudget(offset: 2047, compressRatio: 4, tokenBudget: 2048))
        XCTAssertFalse(
            Qwen4ExpGatheredQSA.decodeCrossesBudget(offset: 2050, compressRatio: 4, tokenBudget: 2048))
        XCTAssertTrue(
            Qwen4ExpGatheredQSA.decodeCrossesBudget(offset: 2051, compressRatio: 4, tokenBudget: 2048))
        XCTAssertTrue(
            Qwen4ExpGatheredQSA.decodeCrossesBudget(offset: 50_000, compressRatio: 4, tokenBudget: 2048))
        XCTAssertFalse(
            Qwen4ExpGatheredQSA.decodeCrossesBudget(offset: 50_000, compressRatio: 0, tokenBudget: 2048))
    }

    /// Fusion `contiguous_causal_gathered_qsa_decode` vs the official masked
    /// path: dense SDPA over the full KV with the QSA selector as a boolean
    /// mask must equal attention over only the gathered rows.
    func testGatheredDecodeMatchesMaskedDenseReference() {
        let queryHeads = 4
        let kvHeads = 2
        let headDim = 16
        let indexerHeads = 2
        let indexerDim = 8
        let ratio = 4
        let budget = 8
        let keyTokens = 43  // 10 complete blocks + 3 tail
        let maxBlocks = keyTokens / ratio
        let blockBudget = budget / ratio

        func seeded(_ shape: [Int], seed: UInt64) -> MLXArray {
            MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.float16)
        }
        let queries = seeded([1, queryHeads, 1, headDim], seed: 1)
        let keys = seeded([1, kvHeads, keyTokens, headDim], seed: 2)
        let values = seeded([1, kvHeads, keyTokens, headDim], seed: 3)
        let indexQueries = seeded([1, 1, indexerHeads, indexerDim], seed: 4)
        let pooled = seeded([1, maxBlocks, indexerDim], seed: 5)

        let gathered = Qwen4ExpGatheredQSA.attendDecode(
            queries: queries, keys: keys, values: values,
            indexQueries: indexQueries, pooledIndexKeys: pooled,
            queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim,
            indexerHeadDim: indexerDim, compressRatio: ratio, tokenBudget: budget)

        // Reference: same fp32 relu-sum scores, top-k blocks, tail, boolean mask.
        let scores = Qwen4ExpGatheredQSA.portableIndexerScores(
            queries: indexQueries, pooledKeys: pooled, headDim: indexerDim)
        let kth = maxBlocks - blockBudget
        let picked = argPartition(scores, kth: kth, axis: -1)[0..., 0..., kth...]
        let blockHits = putAlong(
            MLXArray.zeros([1, 1, maxBlocks], dtype: .bool), picked,
            values: MLXArray(true), axis: -1)
        var tokenMask = repeated(blockHits, count: ratio, axis: -1)
        let tail = MLXArray.ones([1, 1, keyTokens - maxBlocks * ratio], dtype: .bool)
        tokenMask = concatenated([tokenMask, tail], axis: -1)
        let bias = MLX.where(tokenMask, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(queries.dtype).reshaped([1, 1, 1, keyTokens])
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: pow(Float(headDim), -0.5), mask: .array(bias)
        ).transposed(0, 2, 1, 3)

        eval(gathered, reference)
        XCTAssertEqual(gathered.shape, [1, 1, queryHeads, headDim])
        let diff = abs(gathered.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(diff, 2e-3, "gathered decode diverged from masked dense reference")
    }

    func testVerifyGatherEnabledUnlessKillSwitch() {
        XCTAssertTrue(Qwen4ExpGatheredQSA.verifyGatherEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpGatheredQSA.verifyGatherEnabled(environment: [
                Qwen4ExpGatheredQSA.verifyGatherEnvFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpGatheredQSA.verifyGatherEnabled(environment: [
                Qwen4ExpGatheredQSA.verifyGatherEnvFlag: "0"
            ]))
        XCTAssertEqual(Qwen4ExpGatheredQSA.verifyMinKeyTokens(environment: [:]), 16_384)
        XCTAssertEqual(
            Qwen4ExpGatheredQSA.verifyMinKeyTokens(environment: [
                Qwen4ExpGatheredQSA.verifyMinKeyTokensEnvFlag: "4096"
            ]), 4096)
    }

    func testVerifyGeometryMatchesMlxServe() {
        // S=7, kv=20000, kb=128, ratio=4: union is 7*128 blocks, not the cache.
        let geom = Qwen4ExpGatheredQSA.verifyGeometry(
            queryTokens: 7, keyTokens: 20_000, compressRatio: 4, blockBudget: 128)
        XCTAssertNotNil(geom)
        XCTAssertEqual(geom?.completeBlocks, 5000)
        let offset = 20_000 - 7
        let tailStart = ((offset + 1) / 4) * 4
        XCTAssertEqual(geom?.tailStart, tailStart)
        XCTAssertEqual(geom?.blocksBelowTail, tailStart / 4)
        XCTAssertEqual(geom?.unionSlots, min(tailStart / 4, 7 * 128))
        XCTAssertEqual(
            geom?.gatheredRows,
            (geom?.unionSlots ?? 0) * 4 + (20_000 - tailStart))
        XCTAssertLessThan(geom?.gatheredRows ?? 20_000, 20_000)

        XCTAssertNil(
            Qwen4ExpGatheredQSA.verifyGeometry(
                queryTokens: 1, keyTokens: 20_000, compressRatio: 4, blockBudget: 128))
        XCTAssertNil(
            Qwen4ExpGatheredQSA.verifyGeometry(
                queryTokens: 16, keyTokens: 20_000, compressRatio: 4, blockBudget: 128))
        // Union fills the cache: caller must decline (rows >= kv).
        let filled = Qwen4ExpGatheredQSA.verifyGeometry(
            queryTokens: 4, keyTokens: 64, compressRatio: 4, blockBudget: 16)
        XCTAssertEqual(filled?.gatheredRows, 64)
    }

    func testVerifyHostPlanMatchesDenseMask() {
        // mlx-serve hermetic pin: every (row, token) the dense mask shows is
        // reachable through exactly one gathered slot.
        struct Case {
            var s: Int
            var kv: Int
            var kb: Int
            var ratio: Int
        }
        let cases = [
            Case(s: 2, kv: 256, kb: 8, ratio: 4),
            Case(s: 5, kv: 256, kb: 8, ratio: 4),
            Case(s: 7, kv: 1024, kb: 16, ratio: 4),
            Case(s: 15, kv: 1024, kb: 16, ratio: 4),
            Case(s: 3, kv: 61, kb: 3, ratio: 4),
            Case(s: 4, kv: 64, kb: 2, ratio: 4),
            Case(s: 8, kv: 257, kb: 5, ratio: 8),
        ]
        var seed: UInt64 = 0xbe1f7
        func nextUInt(_ bound: Int) -> Int {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1
            return Int(seed % UInt64(bound))
        }
        for c in cases {
            let nb = c.kv / c.ratio
            var blocks = [Int32](repeating: Int32.max, count: c.s * c.kb)
            for i in 0 ..< c.s {
                let p = c.kv - c.s + i
                let complete = min((p + 1) / c.ratio, nb)
                let count = min(complete, c.kb)
                var picked = [Bool](repeating: false, count: max(complete, 1))
                var n = 0
                while n < count {
                    let b = nextUInt(complete)
                    if picked[b] { continue }
                    picked[b] = true
                    n += 1
                }
                var w = 0
                for b in 0 ..< complete where picked[b] {
                    blocks[i * c.kb + w] = Int32(b)
                    w += 1
                }
            }
            let dense = Qwen4ExpGatheredQSA.verifyDenseMask(
                blocks: blocks, queryTokens: c.s, blockBudget: c.kb,
                keyTokens: c.kv, compressRatio: c.ratio)
            guard let plan = Qwen4ExpGatheredQSA.verifyHostPlan(
                blocks: blocks, queryTokens: c.s, blockBudget: c.kb,
                keyTokens: c.kv, compressRatio: c.ratio)
            else {
                XCTFail("verify plan declined S=\(c.s) kv=\(c.kv)")
                continue
            }
            for r in 1 ..< plan.rows.count {
                XCTAssertGreaterThan(plan.rows[r], plan.rows[r - 1])
            }
            var slotOf = [Int](repeating: -1, count: c.kv)
            for (slot, tok) in plan.rows.enumerated() {
                XCTAssertGreaterThanOrEqual(tok, 0)
                XCTAssertLessThan(tok, Int32(c.kv))
                slotOf[Int(tok)] = slot
            }
            for i in 0 ..< c.s {
                for t in 0 ..< c.kv {
                    let want = dense[i * c.kv + t]
                    let sl = slotOf[t]
                    let got = sl >= 0 && plan.mask[i * plan.rows.count + sl]
                    XCTAssertEqual(
                        want, got, "S=\(c.s) kv=\(c.kv) row \(i) token \(t)")
                }
            }
        }
    }

    func testVerifyUnionMatchesHostPlan() {
        let s = 5
        let kv = 256
        let kb = 8
        let ratio = 4
        let nb = kv / ratio
        var seed: UInt64 = 0xbe1f7
        func nextUInt(_ bound: Int) -> Int {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1
            return Int(seed % UInt64(bound))
        }
        var blocks = [Int32](repeating: Int32.max, count: s * kb)
        for i in 0 ..< s {
            let p = kv - s + i
            let complete = min((p + 1) / ratio, nb)
            let count = min(complete, kb)
            var picked = [Bool](repeating: false, count: max(complete, 1))
            var n = 0
            while n < count {
                let b = nextUInt(complete)
                if picked[b] { continue }
                picked[b] = true
                n += 1
            }
            var w = 0
            for b in 0 ..< complete where picked[b] {
                blocks[i * kb + w] = Int32(b)
                w += 1
            }
        }
        guard let geom = Qwen4ExpGatheredQSA.verifyGeometry(
            queryTokens: s, keyTokens: kv, compressRatio: ratio, blockBudget: kb),
            let plan = Qwen4ExpGatheredQSA.verifyHostPlan(
                blocks: blocks, queryTokens: s, blockBudget: kb,
                keyTokens: kv, compressRatio: ratio)
        else {
            XCTFail("verify plan declined")
            return
        }
        let selected = MLXArray(blocks).reshaped([1, s, kb])
        let (tok, vis) = Qwen4ExpGatheredQSA.verifyUnionTokensAndMask(
            selectedBlocks: selected, geometry: geom,
            queryTokens: s, keyTokens: kv, compressRatio: ratio)
        eval(tok, vis)
        XCTAssertLessThanOrEqual(plan.rows.count, geom.gatheredRows)
        XCTAssertEqual(tok.shape, [plan.rows.count])
        for (i, want) in plan.rows.enumerated() {
            XCTAssertEqual(tok[i].item(Int32.self), want, "token slot \(i)")
        }
        XCTAssertEqual(vis.shape, [s, plan.rows.count])
        for i in 0 ..< s {
            for r in 0 ..< plan.rows.count {
                XCTAssertEqual(
                    vis[i, r].item(Bool.self), plan.mask[i * plan.rows.count + r],
                    "vis row \(i) slot \(r)")
            }
        }
    }

    /// mlx-serve #352: subset SDPA over the union equals masked full SDPA
    /// for a known block selection (indexer top-k is pinned separately).
    func testGatheredVerifyMatchesMaskedDenseReference() {
        let queryHeads = 4
        let kvHeads = 2
        let headDim = 16
        let ratio = 4
        let queryTokens = 5
        let keyTokens = 200
        let blockBudget = 2
        let nb = keyTokens / ratio
        guard let geom = Qwen4ExpGatheredQSA.verifyGeometry(
            queryTokens: queryTokens, keyTokens: keyTokens,
            compressRatio: ratio, blockBudget: blockBudget)
        else {
            XCTFail("verify geometry declined")
            return
        }

        func seeded(_ shape: [Int], seed: UInt64) -> MLXArray {
            MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.float16)
        }
        let queries = seeded([1, queryHeads, queryTokens, headDim], seed: 11)
        let keys = seeded([1, kvHeads, keyTokens, headDim], seed: 12)
        let values = seeded([1, kvHeads, keyTokens, headDim], seed: 13)

        var hostBlocks = [Int32](repeating: Int32.max, count: queryTokens * blockBudget)
        var seed: UInt64 = 91
        func nextUInt(_ bound: Int) -> Int {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1
            return Int(seed % UInt64(bound))
        }
        for i in 0 ..< queryTokens {
            let p = keyTokens - queryTokens + i
            let complete = min((p + 1) / ratio, nb)
            let count = min(complete, blockBudget)
            var picked = [Bool](repeating: false, count: max(complete, 1))
            var n = 0
            while n < count {
                let b = nextUInt(complete)
                if picked[b] { continue }
                picked[b] = true
                n += 1
            }
            var w = 0
            for b in 0 ..< complete where picked[b] {
                hostBlocks[i * blockBudget + w] = Int32(b)
                w += 1
            }
        }
        let selected = MLXArray(hostBlocks).reshaped([1, queryTokens, blockBudget])
        let (tok, vis) = Qwen4ExpGatheredQSA.verifyUnionTokensAndMask(
            selectedBlocks: selected, geometry: geom,
            queryTokens: queryTokens, keyTokens: keyTokens, compressRatio: ratio)
        let gatheredRows = tok.dim(0)
        XCTAssertLessThanOrEqual(gatheredRows, geom.gatheredRows)
        let add = MLX.where(vis, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(queries.dtype)
            .reshaped([1, 1, queryTokens, gatheredRows])
        let tok2 = tok.reshaped([1, gatheredRows])
        let gatheredKeys = Qwen4ExpGatheredQSA.batchGatherTokens(
            keys.transposed(0, 2, 1, 3), indices: tok2
        ).transposed(0, 2, 1, 3)
        let gatheredValues = Qwen4ExpGatheredQSA.batchGatherTokens(
            values.transposed(0, 2, 1, 3), indices: tok2
        ).transposed(0, 2, 1, 3)
        let gathered = MLXFast.scaledDotProductAttention(
            queries: queries, keys: gatheredKeys, values: gatheredValues,
            scale: pow(Float(headDim), -0.5), mask: .array(add)
        ).transposed(0, 2, 1, 3)

        let dense = Qwen4ExpGatheredQSA.verifyDenseMask(
            blocks: hostBlocks, queryTokens: queryTokens, blockBudget: blockBudget,
            keyTokens: keyTokens, compressRatio: ratio)
        var addHost = [Float](repeating: -Float.infinity, count: queryTokens * keyTokens)
        for i in 0 ..< queryTokens {
            for t in 0 ..< keyTokens where dense[i * keyTokens + t] {
                addHost[i * keyTokens + t] = 0
            }
        }
        let bias = MLXArray(addHost).reshaped([1, 1, queryTokens, keyTokens]).asType(queries.dtype)
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: pow(Float(headDim), -0.5), mask: .array(bias)
        ).transposed(0, 2, 1, 3)

        eval(gathered, reference)
        XCTAssertEqual(gathered.shape, [1, queryTokens, queryHeads, headDim])
        let diff = abs(gathered.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(diff, 2e-3, "gathered verify diverged from masked dense reference")
    }

    func testAttendVerifyRecordsCounterAndShape() {
        let queryHeads = 4
        let kvHeads = 2
        let headDim = 16
        let indexerHeads = 2
        let indexerDim = 8
        let ratio = 4
        let budget = 8
        let queryTokens = 5
        let keyTokens = 200
        let maxBlocks = keyTokens / ratio
        func seeded(_ shape: [Int], seed: UInt64) -> MLXArray {
            MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.float16)
        }
        Qwen4ExpQSAInvocation.reset()
        let output = Qwen4ExpGatheredQSA.attendVerify(
            queries: seeded([1, queryHeads, queryTokens, headDim], seed: 21),
            keys: seeded([1, kvHeads, keyTokens, headDim], seed: 22),
            values: seeded([1, kvHeads, keyTokens, headDim], seed: 23),
            indexQueries: seeded([1, queryTokens, indexerHeads, indexerDim], seed: 24),
            pooledIndexKeys: seeded([1, maxBlocks, indexerDim], seed: 25),
            queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim,
            indexerHeadDim: indexerDim, compressRatio: ratio, tokenBudget: budget,
            minKeyTokens: 64)
        eval(output)
        XCTAssertEqual(output.shape, [1, queryTokens, queryHeads, headDim])
        XCTAssertEqual(Qwen4ExpQSAInvocation.snapshot().verify, 1)
        XCTAssertFalse(output.asType(.float32).sum().item(Float.self).isNaN)
    }

    func testPooledIndexIncrementalRemainsOptInForNoSyncQualification() {
        XCTAssertFalse(Qwen4ExpPooledIndex.isEnabled(environment: [:]))
        XCTAssertFalse(
            Qwen4ExpPooledIndex.isEnabled(environment: [
                Qwen4ExpPooledIndex.envFlag: "0"
            ]))
        XCTAssertFalse(
            Qwen4ExpPooledIndex.isEnabled(environment: [
                Qwen4ExpPooledIndex.envFlag: "off"
            ]))
        XCTAssertTrue(
            Qwen4ExpPooledIndex.isEnabled(environment: [
                Qwen4ExpPooledIndex.envFlag: "1"
            ]))
        XCTAssertTrue(
            Qwen4ExpPooledIndex.isEnabled(environment: [
                Qwen4ExpPooledIndex.envFlag: "on"
            ]))
    }

    func testPooledIndexSuffixMatchesFullSlice() {
        let ratio = 4
        let keys = MLXArray(
            (0 ..< 24 * 8).map { Float($0) * 0.01 }
        ).reshaped([1, 24, 8]).asType(.float16)
        let pos = MLXArray((0 ..< 24).map(Int32.init)).reshaped([1, 24])
        let identity: (MLXArray) -> MLXArray = { $0 }
        let rope: (MLXArray, MLXArray) -> MLXArray = { states, _ in states }
        let full = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: keys, indexPositionIds: pos, compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        let suffix = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: keys, indexPositionIds: pos, compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope,
            startBlock: 3, stopBlock: 6)
        let expected = full[0..., 3 ..< 6, 0...]
        eval(full, suffix, expected)
        XCTAssertEqual(suffix.shape, [1, 3, 8])
        XCTAssertTrue(all(equal(suffix, expected)).item(Bool.self))
    }

    func testPooledIndexGrowthCapacityMatchesFusion() {
        XCTAssertEqual(Qwen4ExpPooledIndex.tokenStep, 8192)
        XCTAssertEqual(
            Qwen4ExpPooledIndex.growthCapacity(current: 0, needed: 2048, step: 2048), 2048)
        XCTAssertEqual(
            Qwen4ExpPooledIndex.growthCapacity(current: 2048, needed: 4096, step: 2048), 4096)
        XCTAssertEqual(
            Qwen4ExpPooledIndex.growthCapacity(current: 4096, needed: 5000, step: 2048), 8192)
    }

    func testPooledIndexExtendMatchesFullRecompute() {
        let ratio = 4
        let keys = MLXArray(
            (0 ..< 24 * 8).map { Float(($0 % 17) - 8) * 0.05 }
        ).reshaped([1, 24, 8]).asType(.float16)
        let pos = MLXArray((10 ..< 34).map(Int32.init)).reshaped([1, 24])
        let identity: (MLXArray) -> MLXArray = { $0 }
        let rope: (MLXArray, MLXArray) -> MLXArray = { states, _ in states }
        let full = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: keys, indexPositionIds: pos, compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        let first = Qwen4ExpPooledIndex.extend(
            stored: nil, storedBlocks: 0,
            indexKeys: keys[0..., 0 ..< 8, 0...],
            indexPositionIds: pos[0..., 0 ..< 8],
            compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        let second = Qwen4ExpPooledIndex.extend(
            stored: first.keys, storedBlocks: first.blocks,
            indexKeys: keys[0..., 0 ..< 16, 0...],
            indexPositionIds: pos[0..., 0 ..< 16],
            compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        let third = Qwen4ExpPooledIndex.extend(
            stored: second.keys, storedBlocks: second.blocks,
            indexKeys: keys, indexPositionIds: pos,
            compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        let logical = Qwen4ExpPooledIndex.logicalKeys(third.keys, blocks: third.blocks)!
        eval(full, logical)
        XCTAssertEqual(first.blocks, 2)
        XCTAssertEqual(second.blocks, 4)
        XCTAssertEqual(third.blocks, 6)
        XCTAssertGreaterThanOrEqual(first.keys!.dim(1), 2)
        XCTAssertEqual(first.keys!.dim(1), third.keys!.dim(1))
        XCTAssertEqual(logical.shape, full.shape)
        XCTAssertTrue(all(equal(logical, full)).item(Bool.self))
    }

    func testPooledIndexUpdateDefersEvaluationToFinalConsumer() throws {
        let ratio = 4
        let keys = MLXArray(
            (0 ..< 24 * 8).map { Float(($0 % 23) - 11) * 0.03125 }
        ).reshaped([1, 24, 8]).asType(.float16)
        let positions = MLXArray((40 ..< 64).map(Int32.init)).reshaped([1, 24])
        let identity: (MLXArray) -> MLXArray = { $0 }
        let rope: (MLXArray, MLXArray) -> MLXArray = { values, _ in values }
        let cache = CBv2LayerCache(
            layerIndex: 0,
            kind: CBv2LayerKind(
                attention: .full, headDim: 8, kvHeads: 1, queryHeads: 1))
        cache.qwen4IndexKeys = keys
        cache.qwen4IndexPositionIds = positions

        let first = try XCTUnwrap(Qwen4ExpPooledIndex.update(
            cache: cache, compressRatio: ratio, logicalTokens: 16,
            indexKeyNorm: identity, applyIndexRope: rope,
            environment: [Qwen4ExpPooledIndex.envFlag: "1"]))
        XCTAssertNil(try cache.qwen4PooledIndexKeys?.evaluatedBufferInfo())
        let second = try XCTUnwrap(Qwen4ExpPooledIndex.update(
            cache: cache, compressRatio: ratio, logicalTokens: 24,
            indexKeyNorm: identity, applyIndexRope: rope,
            environment: [Qwen4ExpPooledIndex.envFlag: "1"]))
        XCTAssertNil(try cache.qwen4PooledIndexKeys?.evaluatedBufferInfo())

        let expectedFirst = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: keys[0..., 0 ..< 16, 0...],
            indexPositionIds: positions[0..., 0 ..< 16],
            compressRatio: ratio, indexKeyNorm: identity, applyIndexRope: rope)
        let expectedSecond = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: keys, indexPositionIds: positions, compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        eval(first, second, expectedFirst, expectedSecond)
        XCTAssertEqual(first.asData(access: .copy).data, expectedFirst.asData(access: .copy).data)
        XCTAssertEqual(second.asData(access: .copy).data, expectedSecond.asData(access: .copy).data)
    }

    func testPooledIndexStalePrefixRebuilds() {
        let ratio = 4
        let keys = MLXArray(
            (0 ..< 16 * 4).map { Float($0) }
        ).reshaped([1, 16, 4]).asType(.float16)
        let pos = MLXArray((0 ..< 16).map(Int32.init)).reshaped([1, 16])
        let identity: (MLXArray) -> MLXArray = { $0 }
        let rope: (MLXArray, MLXArray) -> MLXArray = { states, _ in states }
        let garbage = MLXArray(Array(repeating: Float(99), count: 8)).reshaped([1, 2, 4])
            .asType(.float16)
        let rebuilt = Qwen4ExpPooledIndex.extend(
            stored: garbage, storedBlocks: 9,
            indexKeys: keys, indexPositionIds: pos,
            compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        let full = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: keys, indexPositionIds: pos, compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        let logical = Qwen4ExpPooledIndex.logicalKeys(rebuilt.keys, blocks: rebuilt.blocks)!
        eval(logical, full)
        XCTAssertEqual(rebuilt.blocks, 4)
        XCTAssertTrue(all(equal(logical, full)).item(Bool.self))
    }

    func testPooledIndexRollbackFrontierCannotReuseRejectedDraftKeys() {
        let ratio = 4
        let original = MLXArray(
            (0 ..< 8 * 4).map { Float($0) }
        ).reshaped([1, 8, 4]).asType(.float16)
        let positions = MLXArray((0 ..< 8).map(Int32.init)).reshaped([1, 8])
        let identity: (MLXArray) -> MLXArray = { $0 }
        let rope: (MLXArray, MLXArray) -> MLXArray = { states, _ in states }
        let cache = CBv2LayerCache(
            layerIndex: 0,
            kind: CBv2LayerKind(
                attention: .full, headDim: 4, kvHeads: 1, queryHeads: 1))
        cache.qwen4IndexKeys = Qwen4ExpIndexCapacity.appendTokens(
            buffer: nil, offset: 0, rows: original)
        cache.qwen4IndexPositionIds = Qwen4ExpIndexCapacity.appendPositions(
            buffer: nil, offset: 0, rows: positions)
        cache.qwen4PooledIndexKeys = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: original,
            indexPositionIds: positions,
            compressRatio: ratio,
            indexKeyNorm: identity,
            applyIndexRope: rope)
        cache.qwen4PooledIndexBlocks = 2

        // Three rejected columns roll the committed frontier from 8 to 5.
        Qwen4ExpPooledIndex.reconcileCommittedFrontier(
            cache: cache, committedTokens: 5, compressRatio: ratio)
        XCTAssertEqual(cache.qwen4PooledIndexBlocks, 1)
        XCTAssertEqual(cache.qwen4PooledIndexKeys?.shape, [1, 1, 4])

        let replacement = MLXArray(
            (100 ..< 112).map { Float($0) }
        ).reshaped([1, 3, 4]).asType(.float16)
        cache.qwen4IndexKeys = Qwen4ExpIndexCapacity.appendTokens(
            buffer: cache.qwen4IndexKeys, offset: 5, rows: replacement)
        cache.qwen4IndexPositionIds = Qwen4ExpIndexCapacity.appendPositions(
            buffer: cache.qwen4IndexPositionIds,
            offset: 5,
            rows: positions[0..., 5 ..< 8])

        let expectedRaw = concatenated([original[0..., 0 ..< 5, 0...], replacement], axis: 1)
        let expected = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: expectedRaw,
            indexPositionIds: positions,
            compressRatio: ratio,
            indexKeyNorm: identity,
            applyIndexRope: rope)
        let actual = Qwen4ExpPooledIndex.reuseOrCompute(
            cache: cache,
            compressRatio: ratio,
            logicalTokens: 8,
            indexKeyNorm: identity,
            applyIndexRope: rope)
        eval(expected, actual)
        XCTAssertTrue(all(equal(expected, actual)).item(Bool.self))
    }

    func testIndexCapacityDefaultsOn() {
        XCTAssertTrue(Qwen4ExpIndexCapacity.isEnabled(environment: [:]))
        XCTAssertTrue(
            Qwen4ExpIndexCapacity.isEnabled(environment: [
                Qwen4ExpIndexCapacity.envFlag: "1"
            ]))
        XCTAssertFalse(
            Qwen4ExpIndexCapacity.isEnabled(environment: [
                Qwen4ExpIndexCapacity.envFlag: "off"
            ]))
        XCTAssertFalse(
            Qwen4ExpIndexCapacity.isEnabled(environment: [
                Qwen4ExpIndexCapacity.envFlag: "0"
            ]))
    }

    func testIndexCapacityBoundsLongContextSlackWithoutExtraGrowthEvents() {
        let step = Qwen4ExpPooledIndex.tokenStep
        XCTAssertEqual(Qwen4ExpIndexCapacity.growthCapacity(
            current: 0, needed: 8_000, step: step), 8_192)
        XCTAssertEqual(Qwen4ExpIndexCapacity.growthCapacity(
            current: 8_192, needed: 16_000, step: step), 16_384)
        XCTAssertEqual(Qwen4ExpIndexCapacity.growthCapacity(
            current: 16_384, needed: 24_000, step: step), 32_768)
        XCTAssertEqual(Qwen4ExpIndexCapacity.growthCapacity(
            current: 32_768, needed: 40_000, step: step), 49_152)
        XCTAssertEqual(Qwen4ExpIndexCapacity.growthCapacity(
            current: 49_152, needed: 56_000, step: step), 65_536)

        // The old pure-doubling path allocated 131,072 slots here.
        XCTAssertEqual(Qwen4ExpIndexCapacity.growthCapacity(
            current: 65_536, needed: 72_000, step: step), 81_920)
        XCTAssertEqual(Qwen4ExpIndexCapacity.growthCapacity(
            current: 81_920, needed: 80_073, step: step), 81_920)
        XCTAssertLessThan(
            Qwen4ExpIndexCapacity.growthCapacity(
                current: 65_536, needed: 80_073, step: step),
            2 * 80_073)
    }

    func testIndexCapacityAppendMatchesConcat() {
        Qwen4ExpQSAIndexGrowthMetrics.reset()
        let first = MLXArray((0 ..< 8 * 4).map { Float($0) }).reshaped([1, 8, 4]).asType(.float16)
        let second = MLXArray((100 ..< 132).map { Float($0) }).reshaped([1, 8, 4]).asType(.float16)
        let concat = concatenated([first, second], axis: 1)
        let grown = Qwen4ExpIndexCapacity.appendTokens(buffer: nil, offset: 0, rows: first)
        let both = Qwen4ExpIndexCapacity.appendTokens(buffer: grown, offset: 8, rows: second)
        let logical = Qwen4ExpIndexCapacity.logicalTokens(both, length: 16)!
        eval(concat, logical)
        XCTAssertGreaterThanOrEqual(grown.dim(1), 8)
        XCTAssertEqual(grown.dim(1), both.dim(1))
        XCTAssertEqual(logical.shape, concat.shape)
        XCTAssertTrue(all(equal(logical, concat)).item(Bool.self))
        XCTAssertEqual(
            Qwen4ExpQSAIndexGrowthMetrics.snapshot(),
            .init(
                growthEvents: 1,
                largestCapacityTokens: 8_192,
                neededAtLargestCapacityTokens: 8))
    }

    func testIndexCapacityPositionsGrowLastAxis() {
        let first = MLXArray((0 ..< 8).map(Int32.init)).reshaped([1, 8])
        let second = MLXArray((8 ..< 16).map(Int32.init)).reshaped([1, 8])
        let concat = concatenated([first, second], axis: -1)
        let grown = Qwen4ExpIndexCapacity.appendPositions(buffer: nil, offset: 0, rows: first)
        let both = Qwen4ExpIndexCapacity.appendPositions(buffer: grown, offset: 8, rows: second)
        let logical = Qwen4ExpIndexCapacity.logicalPositions(both, length: 16)!
        eval(concat, logical)
        XCTAssertTrue(all(equal(logical, concat)).item(Bool.self))
    }

    func testSynthesizedTextPositionsAreDeviceArange() {
        let pos = Qwen4ExpTextPositions.synthesized(offset: 10, length: 4)
        let expected = MLXArray([Int32(10), 11, 12, 13]).reshaped([1, 4])
        eval(pos, expected)
        XCTAssertEqual(pos.shape, [1, 4])
        XCTAssertEqual(pos.dtype, .int32)
        XCTAssertTrue(all(equal(pos, expected)).item(Bool.self))
    }

    func testPoolCompletedIndexKeysUsesCompressedBlockStarts() {
        let ratio = 4
        let keys = MLXArray(
            (0 ..< 24 * 8).map { Float($0) * 0.01 }
        ).reshaped([1, 24, 8]).asType(.float16)
        let pos = Qwen4ExpGatheredQSA.int32Range(24).reshaped([1, 24])
        let identity: (MLXArray) -> MLXArray = { $0 }
        var captured: [MLXArray] = []
        let rope: (MLXArray, MLXArray) -> MLXArray = { states, positions in
            captured.append(positions)
            return states
        }
        let full = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: keys, indexPositionIds: pos, compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope)
        let suffix = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: keys, indexPositionIds: pos, compressRatio: ratio,
            indexKeyNorm: identity, applyIndexRope: rope,
            startBlock: 3, stopBlock: 6)
        XCTAssertEqual(captured.count, 2)
        let fullStarts = Qwen4ExpGatheredQSA.int32Range(0, 24, step: ratio)
        let suffixStarts = Qwen4ExpGatheredQSA.int32Range(12, 24, step: ratio)
        eval(full, suffix, captured[0], captured[1], fullStarts, suffixStarts)
        XCTAssertEqual(captured[0].shape, [1, 6])
        XCTAssertEqual(captured[1].shape, [1, 3])
        XCTAssertTrue(all(equal(captured[0].reshaped([6]), fullStarts)).item(Bool.self))
        XCTAssertTrue(all(equal(captured[1].reshaped([3]), suffixStarts)).item(Bool.self))
    }

    func testInt32RangeMatchesFusionArange() {
        let stepped = Qwen4ExpGatheredQSA.int32Range(12, 24, step: 4)
        let prefix = Qwen4ExpGatheredQSA.int32Range(6)
        let expectedStepped = MLXArray([Int32(12), 16, 20])
        let expectedPrefix = MLXArray((0 ..< 6).map(Int32.init))
        eval(stepped, prefix, expectedStepped, expectedPrefix)
        XCTAssertEqual(stepped.dtype, .int32)
        XCTAssertEqual(prefix.dtype, .int32)
        XCTAssertTrue(all(equal(stepped, expectedStepped)).item(Bool.self))
        XCTAssertTrue(all(equal(prefix, expectedPrefix)).item(Bool.self))
    }
}
