// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The paged attention kernel core through EngineV2 on the segmented pool: a
/// tiny BF16 Inkling 2 (`Inkling2PagedEngineTests.model`) prefills and decodes
/// with the kernel core, the default paged core.
@Suite(.serialized, .tags(.integration))
struct Inkling2PagedAttentionEngineTests {
    /// Fresh requests on a fresh segmented pool: every first prefill chunk
    /// starts with no pages and no segment. Prefill chunks above a verification
    /// window give the same tokens however the prompt is chunked, and a request
    /// decodes the same alone and beside another.
    @Test func kernelCorePrefillsAndDecodesOnTheSegmentedPool() async throws {
        let model = try Inkling2PagedEngineTests.model(dtype: .bfloat16)
        #expect(model.pagedAttentionCore is Inkling2KernelAttentionCore)
        let prompt = Inkling2PagedEngineTests.prompt(7, count: 96)
        var runs: [[Int]] = []
        // Chunks of 32 and 48 tokens: every chunk is longer than a verification
        // window, so every prompt token takes the prefill arithmetic.
        for chunk in [32, 48] {
            let (engine, backend) = try Inkling2PagedEngineTests.engine(
                model, dtype: .bfloat16, chunk: chunk)
            let tokens = try await Inkling2PagedEngineTests.generate(
                engine, id: 1, prompt: prompt, tokens: 12)
            await engine.shutdown()
            #expect(tokens.count == 12)
            #expect(backend.bytesInUse == 0)
            runs.append(tokens)
        }
        #expect(runs[0] == runs[1], "chunked prefill: \(runs)")

        // Two requests at once: the first one's tokens are unchanged.
        let (engine, backend) = try Inkling2PagedEngineTests.engine(
            model, dtype: .bfloat16, chunk: 32, concurrent: 2)
        async let first = Inkling2PagedEngineTests.generate(
            engine, id: 2, prompt: prompt, tokens: 12)
        async let second = Inkling2PagedEngineTests.generate(
            engine, id: 3, prompt: Inkling2PagedEngineTests.prompt(9, count: 50), tokens: 12)
        let (a, b) = try await (first, second)
        #expect(a == runs[0])
        #expect(b.count == 12)
        // After both finished and released their pages: requests with smaller
        // reservations start on a pool whose earlier segments are retired.
        for (index, count) in [20, 17, 40].enumerated() {
            let tokens = try await Inkling2PagedEngineTests.generate(
                engine, id: UInt64(10 + index),
                prompt: Inkling2PagedEngineTests.prompt(11 + index, count: count), tokens: 4)
            #expect(tokens.count == 4)
        }
        await engine.shutdown()
        #expect(backend.bytesInUse == 0)
    }
}
