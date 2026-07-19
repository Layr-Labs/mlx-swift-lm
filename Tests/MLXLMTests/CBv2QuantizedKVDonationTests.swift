// CBv2QuantizedKVDonationTests.swift — quantized-KV prefix-cache donation
// coverage, end to end through the real engine.
//
// FINDING (the reason donation is SKIPPED for quantized rows): MLX affine
// quantization is not idempotent. The kernel snaps each group's scale to its
// dominant edge (`scale = edge / round(edge / scale)`), so re-quantizing a
// DEQUANTIZED snapshot lands on a different grid — values drift by up to
// ~one quantization step per donate→adopt generation, compounding. A
// quantized row's `snapshot()` therefore reports `snapshotIsLossless ==
// false` and the engine never donates such state (documented in the
// contract); adopting FULL-PRECISION donations into a quantized backend
// stays legal (single quantization, same as a cold prefill).
//
// Covered:
//  (i)   the re-quantization drift itself (4-bit and 8-bit) — a CANARY: if
//        MLX affine quantization ever becomes idempotent, these start
//        failing and quantized donation can be reconsidered;
//  (ii)  quantized engines never donate — cache stays empty, a repeat
//        prompt is a miss, and greedy output stays token-exact vs a
//        no-cache run (all-full-attention AND mixed windowed variants);
//  (iii) control: the same all-full-attention model with FULL-PRECISION KV
//        donates and adopts token-exact with the expected hit size (the
//        mixed-model control lives in CBv2EndToEndTests).
//
// No model downloads; weights are seeded random.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class CBv2QuantizedKVDonationTests: XCTestCase {

    /// Small hash blocks so 41-token prompts produce multi-block hits.
    private let blockSize = 8
    private let window = TinyTestModelConfig().windowSize  // 16

    private struct Stack {
        let engine: EngineV2
        let prefixCache: PrefixCacheV2
    }

    private func makeStack(
        model: TinyTestModel, quantizedBits: Int?, enablePrefixCache: Bool
    ) -> Stack {
        let backend = CBv2ContiguousKVBackend(
            config: .init(
                bytesCapacity: 1 << 28,
                quantization: quantizedBits.map { (groupSize: 64, bits: $0) }))
        let prefixCache = PrefixCacheV2(
            config: .init(blockSize: blockSize, promptContractID: "cbv2-qkv"))
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 7),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16,
                enablePrefixCache: enablePrefixCache),
            prefixCache: prefixCache)
        return Stack(engine: engine, prefixCache: prefixCache)
    }

    private func greedyRequest(id: UInt64, prompt: [Int], maxTokens: Int) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0), maxTokens: maxTokens)
    }

    // MARK: (i) Re-quantization drift canary

    func testQuantizedBackendCapabilityFailsCold() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let backend = CBv2ContiguousKVBackend(
            config: .init(
                bytesCapacity: 1 << 24,
                quantization: (groupSize: 64, bits: 4)))
        XCTAssertEqual(backend.prefixReuseBackend, .contiguousQuantized)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: [kind],
            backend: backend.prefixReuseBackend)
        XCTAssertFalse(capability.isSupported)
        XCTAssertEqual(capability.unsupportedReason, .quantizedRowsUnsupported)
        XCTAssertNil(capability.plan(matchedBoundary: 16))
    }

    func testQuantizedSnapshotsRemainLossyAcrossRequantization() {
        for bits in [4, 8] {
            let donor = CBv2QuantizedSequenceKV(
                promptLength: 16,
                maxLength: 64,
                kvHeads: 2,
                headDim: 64,
                groupSize: 64,
                bits: bits)
            MLXRandom.seed(UInt64(0xD0_0D + bits))
            let keys = MLXRandom.normal([1, 2, 16, 64]).asType(.float16)
            let values = MLXRandom.normal([1, 2, 16, 64]).asType(.float16)
            _ = donor.update(keys: keys, values: values)
            let snapshot = donor.snapshot()

            let resumed = CBv2QuantizedSequenceKV(
                promptLength: 16,
                maxLength: 64,
                kvHeads: 2,
                headDim: 64,
                groupSize: 64,
                bits: bits)
            _ = resumed.update(keys: snapshot.keys, values: snapshot.values)
            let requantized = resumed.snapshot()
            let drift = max(
                abs(requantized.keys - snapshot.keys).max().item(Float.self),
                abs(requantized.values - snapshot.values).max().item(Float.self))
            XCTAssertGreaterThan(
                drift,
                1e-6,
                "\(bits)-bit snapshots must remain fail-cold until adoption is lossless")
        }
    }

    // MARK: (ii) Quantized engines skip donation, stay token-exact

    /// Quantized-KV engine with prefix caching enabled: natural finishes
    /// must NOT donate (the cache stays empty), repeat prompts are misses,
    /// and greedy output is token-exact vs a no-cache run under the same
    /// quantization config.
    private func runQuantizedDonationSkip(model: TinyTestModel, bits: Int) async throws {
        let prompt = makePromptTokens(length: 41, seed: 123)
        let budget = 8

        // No-cache baseline under the SAME quantized-KV config.
        let soloStack = makeStack(model: model, quantizedBits: bits, enablePrefixCache: false)
        let solo = await cbv2SchedCollect(
            try soloStack.engine.submit(greedyRequest(id: 900, prompt: prompt, maxTokens: budget)))
        await soloStack.engine.shutdown()
        XCTAssertEqual(solo.finishReason, .length)
        XCTAssertEqual(solo.tokens.count, budget)

        let stack = makeStack(model: model, quantizedBits: bits, enablePrefixCache: true)
        let first = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 1, prompt: prompt, maxTokens: budget)))
        XCTAssertEqual(first.finishReason, .length)
        XCTAssertEqual(first.tokens, solo.tokens, "caching must not perturb the cold run")

        // Donation runs on an async queue — give a wrong implementation a
        // real window to donate before asserting the skip.
        let donated = await cbv2SchedWait(timeoutSeconds: 0.5) {
            stack.prefixCache.stats().entryCount >= 1
        }
        XCTAssertFalse(
            donated,
            "quantized-KV state must never be donated (lossy snapshot — see "
                + "CBv2SequenceKV.snapshotIsLossless)")

        let second = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 2, prompt: prompt, maxTokens: budget)))
        await stack.engine.shutdown()

        XCTAssertEqual(second.finishReason, .length)
        XCTAssertEqual(
            second.usage?.prefixCacheHitTokens, 0,
            "with donation skipped, a repeat prompt must be a clean miss")
        XCTAssertEqual(
            second.tokens, solo.tokens,
            "quantized decode must stay token-exact vs the no-cache run")
        XCTAssertEqual(stack.prefixCache.stats().entryCount, 0)
    }

    /// ALL layers full attention ⇒ every layer's KV is quantized (the
    /// donation skip covers the whole state).
    func testQuantizedEngineSkipsDonation_FullAttentionOnly() async throws {
        let model = TinyTestModel.make(seed: 0xFEED, headDim: 64, fullAttentionOnly: true)
        try await runQuantizedDonationSkip(model: model, bits: 8)
    }

    /// Mixed model: quantized full layer + unquantized sliding-window(16)
    /// layer. The windowed layer never donates anyway; the quantized full
    /// layer's lossy snapshot must veto the donation as a whole.
    func testQuantizedEngineSkipsDonation_MixedWindowed() async throws {
        let model = TinyTestModel.make(seed: 0xFEED, headDim: 64)
        try await runQuantizedDonationSkip(model: model, bits: 4)
    }

    // MARK: (iii) Full-precision control on the all-full-attention fixture

    /// The SAME all-full-attention model with full-precision KV donates and
    /// adopts token-exact (proves the skip above is specifically about
    /// quantization, not the fixture or the donation path).
    func testFullPrecisionDonationRoundTrip_FullAttentionOnly() async throws {
        let model = TinyTestModel.make(seed: 0xFEED, headDim: 64, fullAttentionOnly: true)
        let prompt = makePromptTokens(length: 41, seed: 123)
        let budget = 8
        // 41 tokens, blockSize 8 ⇒ 5 whole lookup blocks; no windowed layers
        // ⇒ zero trailing recompute ⇒ all 40 matched tokens are adopted.
        let expectedHit = 5 * blockSize

        let stack = makeStack(model: model, quantizedBits: nil, enablePrefixCache: true)
        let first = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 1, prompt: prompt, maxTokens: budget)))
        XCTAssertEqual(first.finishReason, .length)
        XCTAssertEqual(first.usage?.prefixCacheHitTokens, 0, "first run is a miss")

        let donated = await cbv2SchedWait {
            stack.prefixCache.stats().entryCount >= 1
        }
        XCTAssertTrue(donated, "full-precision request must donate its prefix")

        let second = await cbv2SchedCollect(
            try stack.engine.submit(greedyRequest(id: 2, prompt: prompt, maxTokens: budget)))
        await stack.engine.shutdown()

        XCTAssertEqual(second.finishReason, .length)
        XCTAssertEqual(second.usage?.prefixCacheHitTokens, expectedHit)
        XCTAssertEqual(
            second.tokens, first.tokens,
            "full-precision adoption must be token-exact vs the cold run")
    }
}
