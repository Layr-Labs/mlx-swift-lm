// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    /// Pins the engine-setup step price for the real GLM-5.3-Flash DSA
    /// geometry (11 layers, latent 512, index 32x128, top-k 2048, 64 heads)
    /// with the serving dense-history prefix. No weights or KV are allocated.
    @Suite(.serialized)
    struct GLM5NextStepPriceTests {
        private final class Target: GLM5NextPagedModelProviding {
            let glm5NextPagedKinds = (0 ..< 11).map {
                CBv2LayerKind(
                    attention: .full, headDim: 512, valueHeadDim: 257, kvHeads: 1,
                    queryHeads: 64, modelLayerIndex: 3 + 4 * $0)
            }
            let glm5NextPagedMaximumContext = 1 << 20
            let glm5NextPagedIndexHeads = 32
            let glm5NextPagedIndexDimension = 128
            let glm5NextPagedSelectionWidth = 2051
            let glm5NextPagedSelectionScoreBlockBytes = 64 << 20
            let glm5NextPagedAttentionBlockBytes = 64 << 20
            let glm5NextPagedDenseHistoryPrefix = true
            var glm5NextPagedNativeSparseDecodeQueries = 8
            var glm5NextPagedLatentHistoryMinimumQueries = 9
            let glm5NextPagedAbsorbedAttention = true
            let glm5NextPagedMetadataReads = GLM5NextPagedMetadataReads.hostRecords
        }

        /// Owns the target and backend the binding holds weakly.
        private struct Bound {
            let target: Target
            let backend: PagedKVBackend
            let binding: GLM5NextPagedModelBinding
        }

        private func bound(context: Int, step: Int, requests: Int) throws -> Bound {
            let target = Target()
            let backend = try PagedKVBackend(
                glm5NextTarget: target,
                config: .init(
                    capacityBytes: 64 << 30, dtype: .float32, maxPrefillChunk: step,
                    nominalMaxSequenceLength: context, maxBufferLength: 96 << 30,
                    segmentSizeBytes: 64 << 20,
                    layerDTypes: Array(repeating: .float32, count: 11),
                    gatheredAttention: try .init(
                        maximumBatchSize: requests, maximumQueryTokens: step,
                        maximumContextTokens: context, maximumInFlightGraphs: 2,
                        maximumScratchBytes: 1 << 50, admissionMode: .stepOwned(.pinnedMetal))))
            return .init(
                target: target, backend: backend,
                binding: try #require(backend.glm5NextModelBinding))
        }

        private func price(context: Int, step: Int, row: Int = 512, requests: Int = 4) throws
            -> Int
        {
            let owner = try bound(context: context, step: step, requests: requests)
            return try withExtendedLifetime(owner) {
                try owner.binding.worstCaseStepBytes(
                    maximumRequests: requests, maximumStepTokens: step, maximumRowTokens: row)
            }
        }

        /// Steps whose chunk equals the step (the provider's GLM geometry), at
        /// the provider's 8K/32K serving contexts and up to the 262,144-token
        /// coordinator context.
        @Test func realGeometryStepPriceIsBoundedByHistoryReads() throws {
            let contexts = [8192, 32768, 131_072, 262_144]
            var prices: [String: Int] = [:]
            for context in contexts {
                for step in [512, 2048] {
                    for requests in [1, 4] {
                        prices["ctx\(context).step\(step).rows\(requests)"] = try price(
                            context: context, step: step, row: step, requests: requests)
                    }
                }
            }
            print(
                "GLM53_STEP_PRICE "
                    + prices.keys.sorted().map {
                        "\($0)=\(prices[$0]!) (\(String(format: "%.2f", Double(prices[$0]!) / 1e9)) GB)"
                    }.joined(separator: " "))
            // A prefill-sized row reads its latent history once instead of
            // retaining 11 layers x 2051 x 512 FP32 per sparse query token.
            let perToken = 11 * 2051 * 512 * 4
            for context in [8192, 32768] {
                for requests in [1, 4] {
                    let small = prices["ctx\(context).step512.rows\(requests)"]!
                    let large = prices["ctx\(context).step2048.rows\(requests)"]!
                    #expect(large - small < (2048 - 512) * perToken / 4)
                }
            }
            // Pinned (allocator bounds of this host); +-3% absorbs
            // allocator-policy rounding differences.
            let pinned = [
                "ctx8192.step512.rows1": 2_158_268_942, "ctx8192.step512.rows4": 5_928_406_236,
                "ctx8192.step2048.rows1": 4_161_972_750, "ctx8192.step2048.rows4": 8_629_979_324,
                "ctx32768.step512.rows1": 3_540_882_190, "ctx32768.step512.rows4": 11_912_881_900,
                "ctx32768.step2048.rows1": 5_648_395_022,
                "ctx32768.step2048.rows4": 14_160_432_316,
                "ctx131072.step512.rows1": 9_033_807_198,
                "ctx131072.step512.rows4": 33_434_710_620,
                "ctx131072.step2048.rows1": 11_452_747_102,
                "ctx131072.step2048.rows4": 36_132_132_348,
                "ctx262144.step512.rows1": 16_356_112_734,
                "ctx262144.step512.rows4": 62_124_104_348,
                "ctx262144.step2048.rows1": 19_190_288_734,
                "ctx262144.step2048.rows4": 65_421_354_492,
            ]
            for (key, value) in pinned {
                let actual = Double(prices[key]!)
                #expect(abs(actual - Double(value)) <= 0.03 * Double(value), "\(key)=\(actual)")
            }
            // The provider's 2,048-token chunk and step fit the 256 GiB host's
            // post-load admission (about 56.8 GB) with 4 requests at 8K and
            // 32K, and one full 262,144-token request fits beside its ~30 KB
            // per token of FP32 sparse KV. The pooled index planes and index
            // metadata reads scale with context per row, so four full
            // 262,144-token rows do not fit at any chunk size.
            #expect(prices["ctx8192.step2048.rows4"]! < 56_000_000_000)
            #expect(prices["ctx32768.step2048.rows4"]! < 56_000_000_000)
            #expect(prices["ctx262144.step2048.rows1"]! + 262_144 * 30_000 < 56_000_000_000)
        }

        /// Decode and MTP verification steps keep their per-query reads;
        /// a prefill-sized row never prices one per query.
        @Test func onlyDecodeSizedRowsRetainSelectedReads() throws {
            let perQuery = 11 * 2051 * 512 * 4
            #expect(try price(context: 8192, step: 8, row: 8, requests: 1) > 8 * perQuery)
            #expect(try price(context: 8192, step: 2048, row: 2048, requests: 1) < 512 * perQuery)
        }

        /// Every row-layer's exact price is bounded by its split: the
        /// context-independent part plus the per-token rate times its history,
        /// for decode, threshold and prefill rows at every history length and
        /// start position up to the context limit.
        @Test func stepScratchSplitBoundsEveryExactRowPrice() throws {
            let context = 262_144
            let owner = try bound(context: context, step: 2048, requests: 4)
            try withExtendedLifetime(owner) {
                let binding = owner.binding
                let pool = owner.backend.pool
                let policy = try #require(Memory.allocationFootprintPolicy())
                let pageSize = pool.config.pageSize
                let usable = try #require(
                    pool.group(pool.groupKey(forLayer: 0)).segmentLayout?.maximumUsablePages)
                let classes = Int.bitWidth - usable.leadingZeroBitCount
                let histories = [
                    1, 15, 16, 17, 2048, 2051, 2052, 4096, 8192, 16_387, 32768, 65_536,
                    131_072, 200_000, 262_143, 262_144,
                ]
                for q in [1, 4, 8, 9, 16, 64, 127, 128, 200, 512, 2048] {
                    let split = try binding.rowScratchBound(
                        layer: 0, query: q, pool: pool, policy: policy)
                    for n in histories where n >= q {
                        for start in Set([n - q, 0, max(0, min(n - q, 2044))]) where start + q <= n
                        {
                            let pages = (n - 1) / pageSize + 1
                            let allocations = try binding.workAllocations(
                                kind: binding.kinds[0], range: start ..< start + q,
                                pageSize: pageSize,
                                segments: min(pages, pages / usable + classes + 1),
                                copyPageCount: 0, policy: policy)
                            let transient = allocations.filter {
                                GLM5NextStepBytes.transientRoles.contains($0.role)
                            }.reduce(0) { $0 + $1.upperBoundBytes }
                            let retained =
                                allocations.reduce(0) { $0 + $1.upperBoundBytes } - transient
                            // `workAllocations` prices the history through
                            // the range's end; longer histories only grow.
                            let history = start + q
                            #expect(
                                retained <= split.fixed + split.perToken * history,
                                "retained q=\(q) n=\(history) start=\(start)")
                            #expect(
                                transient
                                    <= split.transientFixed + split.transientPerToken * history,
                                "temporary q=\(q) n=\(history) start=\(start)")
                        }
                    }
                }
            }
        }

        /// The split's setup bytes and per-token rate bound the all-rows-at-
        /// context worst case, and show what one host's envelope admits.
        @Test func stepScratchSplitTable() throws {
            let context = 262_144
            var setups: [Int: Int] = [:]
            var rate = 0
            for requests in [1, 2, 4] {
                let owner = try bound(context: context, step: 2048, requests: requests)
                let scratch = try withExtendedLifetime(owner) {
                    try owner.binding.stepScratch(
                        maximumRequests: requests, maximumStepTokens: 2048, maximumRowTokens: 2048)
                }
                setups[requests] = scratch.setupBytes
                rate = scratch.bytesPerContextToken
                for history in [8192, 32768, 131_072, 262_144] {
                    let exact = try price(
                        context: history, step: 2048, row: 2048, requests: requests)
                    #expect(
                        exact <= scratch.setupBytes + scratch.bytesPerContextToken * requests
                            * history,
                        "requests=\(requests) context=\(history)")
                }
            }
            print(
                "GLM53_STEP_SCRATCH bytes_per_context_token=\(rate) "
                    + [1, 2, 4].map { "setup_rows\($0)=\(setups[$0]!)" }.joined(separator: " "))
            // Index metadata reads, pooled index planes and latent history
            // reads of the prefill row and its decode successor.
            #expect((40_000 ... 70_000).contains(rate))
            // With ~30 KB/token of FP32 sparse KV, the 256 GiB host's post-load
            // admission holds the setup plus one full 262,144-token request.
            #expect(setups[1]! + (rate + 30_000) * 262_144 < 56_800_000_000)
        }

        /// Peak memory over the resident model measured on the real NVFP4
        /// artifact (`GLM5NextRealConcurrencyTests`, 2026-10-16, logs
        /// glm53-paged-4/5/6; MTP off, 2048-token steps): 32K x1 4.55 GB and
        /// 32K x4 5.96 GB on a 4-row engine, 130K x1 10.09 GB and 261K x1
        /// 18.32 GB on 1-row engines. Each includes the requests' FP32 sparse
        /// KV (33,836 bytes per token). The split price plus that KV, and the
        /// split price alone against the peak minus KV, keep at least a 1.25x
        /// margin, as does the all-rows-at-context price.
        @Test func layerTemporariesPriceAboveMeasuredPeak() throws {
            let kvPerToken = 33836
            let measured: [(context: Int, rows: Int, prompts: [Int], peak: Int)] = [
                (32768, 4, [32002], 4_554_729_646),
                (32768, 4, Array(repeating: 32002, count: 4), 5_961_532_235),
                (131_072, 1, [130_002], 10_093_922_084),
                (262_144, 1, [261_002], 18_319_652_302),
            ]
            for run in measured {
                let owner = try bound(context: run.context, step: 2048, requests: run.rows)
                let scratch = try withExtendedLifetime(owner) {
                    try owner.binding.stepScratch(
                        maximumRequests: run.rows, maximumStepTokens: 2048,
                        maximumRowTokens: 2048)
                }
                let tokens = run.prompts.reduce(0, +)
                let split = scratch.setupBytes + scratch.bytesPerContextToken * tokens
                let kv = kvPerToken * tokens
                print(
                    "GLM53_STEP_MEASURED context=\(run.context) requests=\(run.prompts.count) "
                        + "peak=\(run.peak) split=\(split) kv=\(kv)")
                #expect(Double(split + kv) >= 1.25 * Double(run.peak), "\(run)")
                #expect(Double(split) >= 1.25 * Double(run.peak - kv), "\(run)")
                if run.prompts.count == 1 {
                    let priced = try price(
                        context: run.context, step: 2048, row: 2048, requests: 1)
                    #expect(Double(priced) >= 1.25 * Double(run.peak - kv), "\(run)")
                }
            }
        }

        @Test func oneRowsChunkBoundsItsQueryNotTheStepTotal() throws {
            let single = try price(context: 8192, step: 1024, row: 512, requests: 1)
            let full = try price(context: 8192, step: 1024, row: 1024, requests: 1)
            #expect(single < full)
        }
    }
}
