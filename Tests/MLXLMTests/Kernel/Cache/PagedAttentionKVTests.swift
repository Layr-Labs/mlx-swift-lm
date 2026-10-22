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

        @Test(.enabled(if: enabled), arguments: [(DType.float32, 128, true), (.bfloat16, 256, true), (.bfloat16, 256, false)])
        func pagedAttentionIsKVCacheSimple(dtype: DType, headDim: Int, inPlace: Bool) throws {
            let saved = (PagedAttentionKVCache.decodeInPlace, PagedAttentionKVCache.decodeInPlaceMinimumTokens)
            PagedAttentionKVCache.decodeInPlace = inPlace
            PagedAttentionKVCache.decodeInPlaceMinimumTokens = 0
            defer { (PagedAttentionKVCache.decodeInPlace, PagedAttentionKVCache.decodeInPlaceMinimumTokens) = saved }
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

        /// A decode step attended in place (`PagedAttentionDecode`: MLX's
        /// `sdpa_vector` / `sdpa_vector_2pass` with paged rows) is bitwise
        /// `scaledDotProductAttention` over the same keys and values held
        /// contiguously, at histories on and across page boundaries and on
        /// both sides of every MLX routing threshold (one pass, two passes
        /// with each block count); where MLX would take its GQA-specialised
        /// pass the cache declines (nil) and the gather path runs.
        @Test(.enabled(if: enabled), arguments: [(DType.float32, 128, 8), (.bfloat16, 256, 12), (.float16, 128, 16)])
        func pagedDecodeIsMLXVectorAttention(dtype: DType, headDim: Int, queryHeads: Int) throws {
            let saved = (PagedAttentionKVCache.decodeInPlace, PagedAttentionKVCache.decodeInPlaceMinimumTokens)
            (PagedAttentionKVCache.decodeInPlace, PagedAttentionKVCache.decodeInPlaceMinimumTokens) = (true, 0)
            defer { (PagedAttentionKVCache.decodeInPlace, PagedAttentionKVCache.decodeInPlaceMinimumTokens) = saved }
            try Device.withDefaultDevice(.gpu) {
                let kvHeads = 2
                let pool = try PagedAttentionKVPool(
                    pageSize: 64, kvHeads: kvHeads, keyHeadDim: headDim, valueHeadDim: headDim, dtype: dtype)
                let scale = 1 / Float(headDim).squareRoot()
                print("PAGED_DECODE arch=\(GPU.deviceInfo().architecture) class=\(PagedAttentionDecode.architectureClass.map(String.init) ?? "-")")
                for N in [1, 2, 63, 64, 65, 129, 700, 1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 9000, 16384, 20_000, 33_001] {
                    let account = try pool.reserve(pages: pool.pages(tokens: N))
                    defer { pool.release(account) }
                    MLXRandom.seed(UInt64(N))
                    let keys = MLXRandom.normal([1, kvHeads, N, headDim]).asType(dtype)
                    let values = MLXRandom.normal([1, kvHeads, N, headDim]).asType(dtype)
                    let queries = MLXRandom.normal([1, queryHeads, 1, headDim]).asType(dtype)
                    eval(keys, values, queries)
                    let reference = MLXFast.scaledDotProductAttention(
                        queries: queries, keys: keys, values: values, scale: scale, mask: .none)
                    let cache = PagedAttentionKVCache(pool: pool, account: account)
                    if N > 1 {
                        let (k, v) = cache.update(
                            keys: keys[0..., 0..., ..<(N - 1), 0...], values: values[0..., 0..., ..<(N - 1), 0...])
                        eval(k, v)
                    }
                    let plan = PagedAttentionDecode.plan(
                        queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim, valueHeadDim: headDim, N: N,
                        dtype: dtype)
                    let output = cache.updateAndAttendStep(
                        queries: queries, keys: keys[0..., 0..., (N - 1)..., 0...],
                        values: values[0..., 0..., (N - 1)..., 0...], scale: scale, mask: .none)
                    let taken = output.map { _ in true } ?? false
                    #expect(taken == (plan != nil), "N=\(N) plan and step disagree")
                    guard let output else {
                        #expect(queryHeads == 8 * kvHeads && N >= 8192 && headDim == 128, "N=\(N) declined")
                        #expect(cache.offset == N - 1, "N=\(N) a declined step changed the cache")
                        continue
                    }
                    eval(output, reference)
                    #expect(cache.offset == N)
                    #expect(Self.identical(output, reference), "\(dtype) hd\(headDim) q\(queryHeads) N=\(N) plan \(String(describing: plan))")
                    // The gather path of the same cache (state) is the same history.
                    let state = cache.state
                    eval(state)
                    #expect(Self.identical(state[0], keys) && Self.identical(state[1], values), "N=\(N) state")
                }
            }
        }

        /// Opt-in timing (`PAGED_DECODE_BENCH=1`, release build): one decode
        /// step's attention over 16 layers of 27B-under-TP2 shape (12 query
        /// heads, 2 kv heads of 256, bf16) at 2K and 32K context: KVCacheSimple
        /// + MLX SDPA, paged with the gather, paged in place. Median ms.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["PAGED_DECODE_BENCH"] == "1"))
        func pagedDecodeStepTime() throws {
            try Device.withDefaultDevice(.gpu) {
                let layers = 16, kvHeads = 2, queryHeads = 12, headDim = 256, dtype = DType.bfloat16
                let scale = 1 / Float(headDim).squareRoot()
                let pool = try PagedAttentionKVPool(
                    pageSize: 64, kvHeads: kvHeads, keyHeadDim: headDim, valueHeadDim: headDim, dtype: dtype)
                func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
                let saved = (PagedAttentionKVCache.decodeInPlace, PagedAttentionKVCache.decodeInPlaceMinimumTokens)
                defer { (PagedAttentionKVCache.decodeInPlace, PagedAttentionKVCache.decodeInPlaceMinimumTokens) = saved }
                PagedAttentionKVCache.decodeInPlaceMinimumTokens = 0
                for context in [2048, 4096, 8192, 16384, 32768] {
                    let prompt = MLXRandom.normal([1, kvHeads, context, headDim]).asType(dtype)
                    eval(prompt)
                    for mode in ["contiguous", "paged-gather", "paged-inplace"] {
                        let account = try pool.reserve(pages: layers * pool.pages(tokens: context + 64))
                        defer { pool.release(account) }
                        PagedAttentionKVCache.decodeInPlace = mode == "paged-inplace"
                        let caches: [KVCache] = (0 ..< layers).map { _ in
                            mode == "contiguous"
                                ? KVCacheSimple() as KVCache : PagedAttentionKVCache(pool: pool, account: account) as KVCache
                        }
                        for cache in caches {
                            let (k, v) = cache.update(keys: prompt, values: prompt)
                            eval(k, v)
                        }
                        var times: [Double] = []
                        for step in 0 ..< 40 {
                            let q = MLXRandom.normal([1, queryHeads, 1, headDim]).asType(dtype)
                            let kv = MLXRandom.normal([1, kvHeads, 1, headDim]).asType(dtype)
                            eval(q, kv)
                            let started = now()
                            let outputs = caches.map {
                                attentionWithCacheUpdate(
                                    queries: q, keys: kv, values: kv, cache: $0, scale: scale, mask: .none)
                            }
                            eval(outputs + caches.flatMap { $0.innerState() })
                            if step >= 8 { times.append(now() - started) }
                        }
                        print(String(format: "PAGED_DECODE_BENCH ctx=%d mode=%@ layers=%d step_ms=%.3f",
                            context, mode, layers, times.sorted()[times.count / 2]))
                    }
                }
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
