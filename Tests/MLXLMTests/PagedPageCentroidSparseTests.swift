import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("PagedPageCentroidSparse Full-Layer Decode", .serialized)
struct PagedPageCentroidSparseTests {

    @Test("Config resolves off by default and parses on/numeric page budgets")
    func configResolution() {
        #expect(!PagedPageCentroidSparseConfig.resolve(environment: [:]).enabled)
        #expect(
            !PagedPageCentroidSparseConfig.resolve(
                environment: [PagedPageCentroidSparseConfig.envKey: "0"]
            ).enabled)
        let onCfg = PagedPageCentroidSparseConfig.resolve(
            environment: [PagedPageCentroidSparseConfig.envKey: "1"])
        #expect(onCfg.enabled && onCfg.topInteriorPages == 64)
        let c32 = PagedPageCentroidSparseConfig.resolve(
            environment: [PagedPageCentroidSparseConfig.envKey: "32"])
        #expect(c32.enabled && c32.topInteriorPages == 32)
        let clamped = PagedPageCentroidSparseConfig.resolve(
            environment: [PagedPageCentroidSparseConfig.envKey: "999"])
        #expect(clamped.enabled && clamped.topInteriorPages == 256)
    }

    @Test(
        "Fixed and segmented paged pools recover planted needle page, preserve fused writes across page boundaries, and support mixed short/long batches"
    )
    func fixedAndSegmentedNeedleRecoveryAndWriteParity() throws {
        MLXRandom.seed(2026)
        let pageSize = 16
        let kvHeads = 2
        let qHeads = 8
        let headDim = 64
        let scale = Float(1.0 / Double(headDim).squareRoot())
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: qHeads)
        ]

        // Small sparse budget so tests run fast: sink=2, recent=2, topInterior=4 -> minActivate=9 pages (144 tokens).
        let sparseCfg = PagedPageCentroidSparseConfig(
            sinkPages: 2, recentPages: 2, topInteriorPages: 4)

        for (label, segBytes) in [("fixed", nil as Int?), ("segmented", 256 * 1024 as Int?)] {
            let poolCfg = PagedKVPoolConfig(
                pageSize: pageSize,
                capacityBytes: 8 * 1024 * 1024,
                dtype: .bfloat16,
                maxPrefillChunk: 512,
                nominalMaxSequenceLength: 1024,
                segmentSizeBytes: segBytes)
            let backend = try PagedKVBackend(layerKinds: kinds, config: poolCfg)

            // Row 0: long (256 tokens = 16 pages), needle planted in interior page 7 (tokens 112..127).
            // Row 1: short (64 tokens = 4 pages, below minActivate -> exact full attention).
            let longLen = 256
            let shortLen = 64
            let stateLong = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: longLen, maxLength: longLen + 64)
            let stateShort = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: shortLen, maxLength: shortLen + 64)
            let stateShortRef = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: shortLen, maxLength: shortLen + 64)
            defer {
                backend.release(stateLong)
                backend.release(stateShort)
                backend.release(stateShortRef)
            }

            let rowLong = try #require(stateLong[0] as? PagedSequenceKV)
            let rowShort = try #require(stateShort[0] as? PagedSequenceKV)
            let rowShortRef = try #require(stateShortRef[0] as? PagedSequenceKV)

            // Build keys/values: weak background noise + strong semantic needle on page 7!
            let bgK = (MLXRandom.normal([kvHeads, longLen, headDim]) * Float(0.1)).asType(.bfloat16)
            let bgV = (MLXRandom.normal([kvHeads, longLen, headDim]) * Float(0.1)).asType(.bfloat16)
            let needleDir = MLXArray.ones([kvHeads, 1, headDim], type: Float.self).asType(.bfloat16)
            let needleVal =
                (MLXArray.ones([kvHeads, pageSize, headDim], type: Float.self)
                * Float(5.0)).asType(.bfloat16)
            let needleKey = broadcast(needleDir * Float(4.0), to: [kvHeads, pageSize, headDim])

            let kLong = concatenated(
                [
                    bgK[0..., 0 ..< (7 * pageSize), 0...], needleKey,
                    bgK[0..., (8 * pageSize)..., 0...],
                ],
                axis: 1)
            let vLong = concatenated(
                [
                    bgV[0..., 0 ..< (7 * pageSize), 0...], needleVal,
                    bgV[0..., (8 * pageSize)..., 0...],
                ],
                axis: 1)
            rowLong.write(keys: kLong, values: vLong)

            let kShort = (MLXRandom.normal([kvHeads, shortLen, headDim]) * Float(0.3)).asType(
                .bfloat16)
            let vShort = (MLXRandom.normal([kvHeads, shortLen, headDim]) * Float(0.3)).asType(
                .bfloat16)
            rowShort.write(keys: kShort, values: vShort)
            rowShortRef.write(keys: kShort, values: vShort)

            let cache = backend.makeLayerCaches()[0]
            let refCache = backend.makeLayerCaches()[0]
            cache.pageSparseDecodeConfig = sparseCfg
            refCache.pageSparseDecodeConfig = .off
            cache.setRows([rowLong, rowShort])
            refCache.setRows([rowShortRef])

            // Query aligned with needleDir for row 0, random for row 1.
            let qNeedle = broadcast(
                MLXArray.ones([1, qHeads, 1, headDim], type: Float.self) * Float(3.0),
                to: [1, qHeads, 1, headDim]
            ).asType(.bfloat16)
            let qShort = MLXRandom.normal([1, qHeads, 1, headDim]).asType(.bfloat16)
            let qBatch = concatenated([qNeedle, qShort], axis: 0)

            // Step 17 decode tokens across a full page boundary (`256 -> 273`, crossing 272 = 17*16)!
            for step in 0 ..< 17 {
                let kNew = (MLXRandom.normal([2, kvHeads, 1, headDim]) * Float(0.05)).asType(
                    .bfloat16)
                let vNew = (MLXRandom.normal([2, kvHeads, 1, headDim]) * Float(0.05)).asType(
                    .bfloat16)
                let out = cache.updateAndAttend(
                    queries: qBatch, keys: kNew, values: vNew, scale: scale, sinks: nil)
                let refOut = refCache.updateAndAttend(
                    queries: qShort, keys: kNew[1 ..< 2], values: vNew[1 ..< 2],
                    scale: scale, sinks: nil)
                eval(out, refOut)
                #expect(!MLX.isNaN(out).any().item(Bool.self), "\(label) step \(step) produced NaN")
                // Row 0 query points strongly at page 7's needle (V = 5.0), so output must stay close to 5.0!
                let row0Mean = out[0].asType(.float32).mean().item(Float.self)
                #expect(
                    row0Mean > 4.5 && row0Mean < 5.2,
                    "\(label) step \(step): expected needle value ~5.0, got \(row0Mean)")
                // Row 1 is below minActivate -> must match exact dense paged attention!
                let shortDiff = MLX.abs(
                    out[1 ..< 2].asType(.float32) - refOut.asType(.float32)
                ).max().item(Float.self)
                #expect(
                    shortDiff < 1e-2,
                    "\(label) step \(step): short row diverged from exact dense by \(shortDiff)")
            }

            // Verify centroids tracked up to completed page boundary, rollback clamps cleanly,
            // and subsequent B=1 <-> B=2 transitions + post-rollback decode succeed cleanly.
            #expect(rowLong.sparseCentroidPages == 17)
            #expect(rowShort.sparseCentroidPages == 0)
            rowLong.rollback(17)
            #expect(rowLong.sparseCentroidPages == 16)

            // Transition to B=1 on the same cache to exercise single-row kernel signature stability.
            cache.setRows([rowLong])
            let kOne = (MLXRandom.normal([1, kvHeads, 1, headDim]) * Float(0.05)).asType(.bfloat16)
            let vOne = (MLXRandom.normal([1, kvHeads, 1, headDim]) * Float(0.05)).asType(.bfloat16)
            let outOne = cache.updateAndAttend(
                queries: qNeedle, keys: kOne, values: vOne, scale: scale, sinks: nil)
            eval(outOne)
            let row0MeanAfterRollback = outOne[0].asType(.float32).mean().item(Float.self)
            #expect(
                row0MeanAfterRollback > 4.5 && row0MeanAfterRollback < 5.2,
                "\(label) B=1 post-rollback needle mean: \(row0MeanAfterRollback)")
            #expect(!backend.pool.writeValidation.isFaulted)
        }
    }

    @Test("Fixed pool pads maxSparsePages < 8 and float16 scoring avoids overflow")
    func smallBudgetPaddingAndFloat16Safety() throws {
        let pageSize = 16
        let kvHeads = 2
        let qHeads = 4
        let headDim = 64
        let scale = 1.0 / sqrt(Float(headDim))
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: qHeads)
        ]
        // sink=1, recent=1, topK=4 -> maxSparsePages=7 < 8 (must pad table cols to >= 8!).
        let smallCfg = PagedPageCentroidSparseConfig(
            sinkPages: 1, recentPages: 1, topInteriorPages: 4)
        let poolCfg = PagedKVPoolConfig(
            pageSize: pageSize,
            capacityBytes: 4 * 1024 * 1024,
            dtype: .float16,
            maxPrefillChunk: 512,
            nominalMaxSequenceLength: 1024,
            segmentSizeBytes: nil)
        let backend = try PagedKVBackend(layerKinds: kinds, config: poolCfg)
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 160, maxLength: 224)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        let k = (MLXRandom.normal([kvHeads, 160, headDim]) * Float(0.5)).asType(.float16)
        let v = (MLXRandom.normal([kvHeads, 160, headDim]) * Float(0.5)).asType(.float16)
        row.write(keys: k, values: v)

        let cache = backend.makeLayerCaches()[0]
        cache.pageSparseDecodeConfig = smallCfg
        cache.setRows([row])

        let q = (MLXRandom.normal([1, qHeads, 1, headDim]) * Float(2.0)).asType(.float16)
        let kNew = (MLXRandom.normal([1, kvHeads, 1, headDim]) * Float(0.1)).asType(.float16)
        let vNew = (MLXRandom.normal([1, kvHeads, 1, headDim]) * Float(0.1)).asType(.float16)
        let out = cache.updateAndAttend(
            queries: q, keys: kNew, values: vNew, scale: scale, sinks: nil)
        eval(out)
        #expect(!MLX.isNaN(out).any().item(Bool.self))
        #expect(!backend.pool.writeValidation.isFaulted)
    }
}
