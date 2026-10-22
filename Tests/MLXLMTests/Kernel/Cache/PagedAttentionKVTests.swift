// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Opt-in (`QWEN_PAGED_KV=1`, GPU): a Qwen 3.5 hybrid model (randomly
    /// initialized; Gated DeltaNet layers and full-attention layers, as one
    /// tensor-parallel rank holds them) whose attention keys and values live
    /// in a page pool (`PagedAttentionKVCache`) gives exactly the logits and
    /// the attention and recurrent state of the same model with
    /// `KVCacheSimple`, bit for bit: chunked prefill across page boundaries,
    /// one-token decode steps, rollback (`trim`) and prefixes shared by page
    /// (copies that diverge), in float32 and bfloat16.
    @Suite(.serialized)
    struct PagedAttentionKVTests {
        static let enabled = ProcessInfo.processInfo.environment["QWEN_PAGED_KV"] == "1"

        private static func configuration(headDim: Int) throws -> Qwen35TextConfiguration {
            let json = """
                {
                  "model_type": "qwen3_5",
                  "hidden_size": 256,
                  "num_hidden_layers": 8,
                  "intermediate_size": 512,
                  "num_attention_heads": 8,
                  "num_key_value_heads": 2,
                  "linear_num_value_heads": 4,
                  "linear_num_key_heads": 2,
                  "linear_key_head_dim": 32,
                  "linear_value_head_dim": 32,
                  "linear_conv_kernel_dim": 4,
                  "vocab_size": 512,
                  "head_dim": \(headDim),
                  "full_attention_interval": 4,
                  "num_experts": 0,
                  "num_experts_per_tok": 0
                }
                """
            return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
        }

        private static func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

        private static func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && a.dtype == b.dtype && bytes(a) == bytes(b)
        }

        /// One request's caches: paged attention beside `KVCacheSimple`.
        private struct Pair {
            let paged: [KVCache]
            let contiguous: [KVCache]
            func copy() -> Pair { Pair(paged: paged.map { $0.copy() }, contiguous: contiguous.map { $0.copy() }) }
        }

        private static func expectSame(_ pair: Pair, _ label: String) {
            for (index, (a, b)) in zip(pair.paged, pair.contiguous).enumerated() {
                #expect(a.offset == b.offset, "\(label) layer \(index) offset")
                let left = a.state, right = b.state
                eval(left + right)
                #expect(left.count == right.count, "\(label) layer \(index) state count")
                for (x, y) in zip(left, right) {
                    #expect(identical(x, y), "\(label) layer \(index) state")
                }
            }
        }

        /// The same tokens through both caches; logits must be the same bits.
        private static func forward(
            _ model: Qwen35TextModel, _ pair: Pair, _ tokens: [Int32], _ label: String
        ) {
            let input = MLXArray(tokens, [1, tokens.count])
            let a = model(input, cache: pair.paged)
            eval([a] + pair.paged.flatMap { $0.innerState() })
            let b = model(input, cache: pair.contiguous)
            eval([b] + pair.contiguous.flatMap { $0.innerState() })
            #expect(identical(a, b), "\(label) logits")
            expectSame(pair, label)
        }

        @Test(.enabled(if: enabled), arguments: [(DType.float32, 128), (.bfloat16, 256)])
        func pagedAttentionIsKVCacheSimple(dtype: DType, headDim: Int) throws {
            try Device.withDefaultDevice(.gpu) {
                let c = try Self.configuration(headDim: headDim)
                MLXRandom.seed(2026)
                let model = Qwen35TextModel(c)
                if dtype != .float32 {
                    model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
                }
                eval(model)
                let pool = try PagedAttentionKVPool(
                    pageSize: 64, kvHeads: 2, keyHeadDim: headDim, valueHeadDim: headDim, dtype: dtype)
                let attentionLayers = 2
                let account = try pool.reserve(pages: 6 * attentionLayers * pool.pages(tokens: 1_200))
                func open() -> Pair {
                    let contiguous = model.newCache(parameters: nil)
                    let paged = model.newCache(parameters: nil).map { cache -> KVCache in
                        cache is KVCacheSimple ? PagedAttentionKVCache(pool: pool, account: account) : cache
                    }
                    #expect(paged.filter { $0 is PagedAttentionKVCache }.count == attentionLayers)
                    return Pair(paged: paged, contiguous: contiguous)
                }
                func prompt(_ length: Int, _ seed: Int) -> [Int32] {
                    (0 ..< length).map { Int32((($0 + 1) * 7919 + seed * 104_729) % c.vocabularySize) }
                }
                let pair = open()
                let values = prompt(1_000, 1)
                // Chunks of 300 (page boundaries fall inside them), then decode.
                for start in stride(from: 0, to: values.count, by: 300) {
                    let end = min(values.count, start + 300)
                    Self.forward(model, pair, Array(values[start ..< end]), "\(dtype) prefill \(start)")
                }
                for step in 0 ..< 12 {
                    Self.forward(model, pair, [Int32((step * 31 + 7) % c.vocabularySize)], "\(dtype) step \(step)")
                }
                // Rollback inside a page and across a page boundary, then go on.
                for (round, n) in [3, 70].enumerated() {
                    for cache in pair.paged + pair.contiguous where cache.isTrimmable { cache.trim(n) }
                    Self.forward(
                        model, pair, [Int32((round * 13 + 5) % c.vocabularySize)],
                        "\(dtype) after trim \(n)")
                }
                // A prefix shared by page: the original and two copies go on
                // differently; the entry itself is unchanged after them.
                let entry = pair.copy()
                let branches = [pair, entry.copy(), entry.copy()]
                for (index, branch) in branches.enumerated() {
                    Self.forward(model, branch, prompt(90 + index * 7, 40 + index), "\(dtype) branch \(index)")
                    for step in 0 ..< 3 {
                        Self.forward(
                            model, branch, [Int32((index * 17 + step * 3 + 1) % c.vocabularySize)],
                            "\(dtype) branch \(index) step \(step)")
                    }
                }
                Self.expectSame(entry, "\(dtype) entry after its branches")
                #expect(pool.allocatedPages > 0)
                _ = (branches, entry)
            }
        }

        /// Pages come back when the histories go, and the pool shrinks to nothing.
        @Test(.enabled(if: enabled))
        func pagesComeBackAndThePoolShrinks() throws {
            try Device.withDefaultDevice(.gpu) {
                let pool = try PagedAttentionKVPool(
                    pageSize: 16, kvHeads: 2, keyHeadDim: 8, valueHeadDim: 8, dtype: .float32)
                let account = try pool.reserve(pages: 32)
                do {
                    let cache = PagedAttentionKVCache(pool: pool, account: account)
                    let keys = MLXRandom.normal([1, 2, 40, 8]), values = MLXRandom.normal([1, 2, 40, 8])
                    let (k, v) = cache.update(keys: keys, values: values)
                    eval(k, v)
                    #expect(Self.identical(k, keys) && Self.identical(v, values))
                    let copy = cache.copy()
                    let (k2, _) = copy.update(keys: keys[0..., 0..., ..<1, 0...], values: values[0..., 0..., ..<1, 0...])
                    eval(k2)
                    #expect(Self.identical(k2[0..., 0..., ..<40, 0...], keys))
                    // The copy took the tail page's write token: it wrote in place.
                    #expect(pool.allocatedPages == 3)
                    // The original writes past what the copy reads: its tail page is copied first.
                    let (k3, v3) = cache.update(
                        keys: keys[0..., 0..., 1 ..< 2, 0...], values: values[0..., 0..., 1 ..< 2, 0...])
                    eval(k3, v3)
                    #expect(pool.allocatedPages == 4)
                    #expect(Self.identical(k3[0..., 0..., ..<40, 0...], keys))
                    #expect(Self.identical(k3[0..., 0..., 40..., 0...], keys[0..., 0..., 1 ..< 2, 0...]))
                    let again = copy.state
                    eval(again)
                    #expect(Self.identical(again[0][0..., 0..., 40..., 0...], keys[0..., 0..., ..<1, 0...]))
                }
                pool.release(account)
                #expect(try pool.trim() > 0)
                #expect(pool.capacityPages == 0 && pool.allocatedPages == 0)
            }
        }
    }
}
