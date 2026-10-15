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
            var glm5NextPagedNativeSparseDecodeQueries = 0
            let glm5NextPagedMetadataReads = GLM5NextPagedMetadataReads.hostRecords
        }

        private func price(context: Int, step: Int, row: Int = 512, requests: Int = 4) throws
            -> Int
        {
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
            let binding = try #require(backend.glm5NextModelBinding)
            return try binding.worstCaseStepBytes(
                maximumRequests: requests, maximumStepTokens: step, maximumRowTokens: row)
        }

        @Test func realGeometryStepPriceIsBoundedByRetainedReads() throws {
            var prices: [String: Int] = [:]
            for context in [8192, 32768] {
                for step in [512, 1024] {
                    prices["ctx\(context).step\(step)"] = try price(context: context, step: step)
                }
            }
            let gib = Double(1 << 30)
            print(
                "GLM53_STEP_PRICE "
                    + prices.keys.sorted().map {
                        "\($0)=\(prices[$0]!) (\(String(format: "%.2f", Double(prices[$0]!) / gib)) GiB)"
                    }.joined(separator: " "))
            // The dominant term is the retained per-query selected-latent read:
            // 11 layers x 2051 x 512 FP32 per sparse query token.
            let perToken = 11 * 2051 * 512 * 4
            for context in [8192, 32768] {
                for step in [512, 1024] {
                    #expect(prices["ctx\(context).step\(step)"]! >= step * perToken)
                }
                #expect(prices["ctx\(context).step1024"]! < 2 * prices["ctx\(context).step512"]!)
            }
            // Pinned (4 requests, 512-token rows, allocator bounds of this
            // host); +-3% absorbs allocator-policy rounding differences.
            let pinned = [
                "ctx8192.step512": 33_577_415_956, "ctx8192.step1024": 62_906_308_340,
                "ctx32768.step512": 46_254_833_852, "ctx32768.step1024": 72_342_667_404,
            ]
            for (key, value) in pinned {
                let actual = Double(prices[key]!)
                #expect(abs(actual - Double(value)) <= 0.03 * Double(value), "\(key)=\(actual)")
            }
            // A 512-token step fits the 256 GiB host's post-load admission
            // (about 56.8 GB) at both contexts.
            #expect(prices["ctx8192.step512"]! < 56_000_000_000)
            #expect(prices["ctx32768.step512"]! < 56_000_000_000)
        }

        @Test func oneRowsChunkBoundsItsQueryNotTheStepTotal() throws {
            let single = try price(context: 8192, step: 1024, row: 512, requests: 1)
            let full = try price(context: 8192, step: 1024, row: 1024, requests: 1)
            #expect(single < full)
        }
    }
}
