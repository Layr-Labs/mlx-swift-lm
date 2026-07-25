// CBv2PagedPrefixReuseChunkRatioTests.swift
//
// Paged frozen-full replay when the pool's prefill chunk is LARGER than the
// model's sliding window.
//
// `CBv2PrefixReusePagedFrozenFullTests` covers the opposite ratio (window 16
// against chunk 8) and every other paged reuse suite inherits that fixture,
// so the chunk > window regime had no coverage at all. It is not a corner:
// gpt-oss-20b is 12 sliding layers of window 128 against the default
// 512-token chunk, and it is the model where prefix reuse matters most,
// because its frozen-replay bound is 1,536 tokens rather than gemma-4's
// 25,600 — reuse fires at a prompt length an operator actually sends.
//
// What the gap cost, measured by the Gate G2 parity harness on real weights:
// paged returned `adoption_failed` with saved = 0 on a 28,672-token prompt
// where contiguous saved 26,880, after BOTH arms matched 28,416 tokens. The
// lookup hit and the plan was derived; only the adoption refused.
//
// The refusal itself was CORRECT. `CBv2PrefixReuseCapability.derive` cannot
// see pool config, so it padded its grant by `maxWindow` as a proxy for the
// chunk, and `PagedKVBackend.requiredFrozenReplayTokens` re-checked the plan
// against the pool's REAL chunk and demanded
// `windowCount * maxWindow + maxPrefillChunk`. On gemma-4 the proxy is
// generous (1,024 against 512) and the check passes; on gpt-oss it is short
// (128 against 512), so the backend was handed 1,664 tokens of replay where
// it needed 2,048 and refused rather than serve sliding rows that would come
// back inexact. Short windows do not fail loudly — they attend fewer keys and
// return a plausible wrong answer — so refusing was the only safe response to
// an under-provisioned plan.
//
// The defect was that the plan was under-provisioned at all. These tests
// therefore pin BOTH halves, because either one alone can be satisfied by a
// wrong fix:
//
//   1. adoption must be ACCEPTED in this ratio — a `saved = 0` on a hit is a
//      capability regression against contiguous;
//   2. the adopted row must be TOKEN-EXACT against a cold twin — relaxing the
//      bound to buy (1) would trade the original frozen-full defect back for
//      a throughput number, and nothing downstream could see it.
//
// (2) is the one with teeth. A test that only asserts (1) passes for a fix
// that simply lowers the bound, which is precisely the unsafe change.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2PrefixReuse: paged frozen replay with chunk > window")
struct CBv2PagedPrefixReuseChunkRatioTests {

    // MARK: - Fixture

    /// Window 8 against a 32-token chunk: the same 1:4 ratio gpt-oss-20b runs
    /// (128 against 512), small enough to decode end to end.
    ///
    /// `TinyTestModel.make(stackedSlidingFull:)` yields
    /// `[full, sliding, sliding, full]` — an owning full layer downstream of
    /// the windowed ones, which is what selects `.frozenFullReplay`.
    private static let window = 8
    private static let chunkSize = 32
    private static let maxLength = 256
    private static let matched = 64
    private static let slidingLayers = 2

    /// What the planner grants: `windowCount * maxWindow + maxWindow`.
    private static let grantedReplay = slidingLayers * window + window  // 24

    private func makeModel() -> TinyTestModel {
        TinyTestModel.make(
            seed: 0x9F17_C0DE, headDim: 64, stackedSlidingFull: true,
            windowSize: Self.window)
    }

    private func makeBackend(_ kinds: [CBv2LayerKind]) throws -> PagedKVBackend {
        try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 64 << 20,
                maxPrefillChunk: Self.chunkSize,
                nominalMaxSequenceLength: Self.maxLength))
    }

    /// Prefill `prompt[from ..< upTo]`, honouring the plan's chunk clamp
    /// exactly as `SchedulerV2` does. The clamp is load-bearing here: it is
    /// what keeps the replay leg inside the slack the planner bought.
    @discardableResult
    private func prefill(
        model: TinyTestModel, backend: PagedKVBackend, state: [CBv2SequenceKV?],
        prompt: [Int], from: Int, upTo: Int, plan: CBv2PrefixReusePlan? = nil
    ) -> [CBv2AttendingLayerCache] {
        let caches: [CBv2AttendingLayerCache] = backend.makeLayerCaches()
        for (i, kind) in model.layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            caches[i].setRows([state[i]!])
        }
        var index = from
        while index < upTo {
            var count = min(Self.chunkSize, upTo - index)
            if let plan { count = plan.clampedChunk(start: index, proposed: count) }
            let slice = Array(prompt[index ..< (index + count)])
            _ = model.forward(
                tokens: MLXArray(slice.map(Int32.init)).reshaped(1, slice.count),
                caches: caches)
            index += slice.count
        }
        return caches
    }

    /// Prefill then greedy-decode. Greedy is the amplifier: a window short by
    /// one key perturbs the logits, one argmax flips, and the trajectories
    /// separate permanently.
    private func run(
        model: TinyTestModel, backend: PagedKVBackend, state: [CBv2SequenceKV?],
        prompt: [Int], from: Int, steps: Int, plan: CBv2PrefixReusePlan? = nil
    ) -> [Int] {
        let caches = prefill(
            model: model, backend: backend, state: state, prompt: prompt,
            from: from, upTo: prompt.count - 1, plan: plan)
        var current = prompt.last!
        var generated: [Int] = []
        for _ in 0 ..< steps {
            let logits = model.forward(
                tokens: MLXArray([Int32(current)]).reshaped(1, 1), caches: caches)
            current = Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0])
            generated.append(current)
        }
        return generated
    }

    /// Donor arm: a cold prefill of `prompt[0 ..< matched)` snapshotted per
    /// layer. The replay form recomputes windowed layers, so only the full
    /// rows are donated — exactly what `PrefixCacheV2` hands over.
    private func donateFullRows(
        model: TinyTestModel, backend: PagedKVBackend, prompt: [Int], matched: Int
    ) throws -> [(keys: MLXArray, values: MLXArray, offset: Int)?] {
        let kinds = model.layerKinds
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: matched, maxLength: Self.maxLength)
        prefill(
            model: model, backend: backend, state: state, prompt: prompt,
            from: 0, upTo: matched)
        let prefix = kinds.enumerated().map {
            index, kind -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard kind.sharesKVWithLayer == nil, let row = state[index],
                case .full = kind.attention
            else { return nil }
            let snapshot = row.snapshot()
            // Paged snapshots are lazy views over the SHARED slabs; the
            // donor's pages are recycled the instant its state is released.
            eval(snapshot.keys, snapshot.values)
            return snapshot
        }
        backend.release(state)
        return prefix
    }

    private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }

    // MARK: - 1. The planner's grant, at real gpt-oss-20b geometry

    /// gpt-oss-20b as `config.json` declares it: 24 layers alternating
    /// `sliding_attention` (window 128) and `full_attention`, head_dim 64,
    /// 8 KV heads.
    private func gptOssLayerKinds() -> [CBv2LayerKind] {
        (0 ..< 24).map { index in
            CBv2LayerKind(
                attention: index.isMultiple(of: 2) ? .slidingWindow(128) : .full,
                headDim: 64,
                kvHeads: 8,
                queryHeads: 64)
        }
    }

    /// The arithmetic that produced `saved = 0`, pinned at the real geometry.
    ///
    /// The planner must grant at least what a 512-token-chunk pool will
    /// demand. It buys that by capping the replay chunk rather than by
    /// assuming a relation between the chunk and the window — the assumption
    /// that gemma-4 satisfied and gpt-oss did not.
    @Test func gptOssGeometryGrantsEnoughReplayForTheDefaultChunk() throws {
        let kinds = gptOssLayerKinds()
        #expect(kinds.filter { $0.attention == .slidingWindow(128) }.count == 12)

        let contiguous = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .contiguousUnquantized)
        #expect(contiguous.strategy == .frozenFullReplay)
        #expect(contiguous.conservativeReplayBoundTokens == 1536, "12 sliding layers x 128")

        let paged = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        #expect(paged.strategy == .frozenFullReplay)
        #expect(paged.conservativeReplayBoundTokens == 1664, "1536 + one window of slack")

        // The boundary the G2 parity harness actually matched on both arms.
        let plan = try #require(
            paged.plan(matchedBoundary: 28416, maximumSequenceLength: 28672))
        #expect(plan.replayTokens == 1664)
        #expect(plan.replayStart == 26752)
        #expect(plan.prefillTokensSaved == 26752, "a hit must save, not return zero")

        // The defect, in one line of arithmetic. Left uncapped, a 512-token
        // pool chunk demands more replay than the planner ever grants, so
        // EVERY gpt-oss adoption was refused after a confirmed hit.
        let uncapped = PagedKVBackend.requiredFrozenReplayTokens(
            layerKinds: kinds, replayChunkTokens: 512)
        #expect(uncapped == 2048, "12 x 128 + one 512-token chunk")
        #expect(
            uncapped > plan.replayTokens,
            "regression guard: this inequality is what returned saved = 0")

        // The fix caps the REPLAY chunk at one window, so the grant covers
        // the demand by construction rather than by luck of configuration.
        #expect(plan.replayChunkCeiling == 128)
        let demanded = PagedKVBackend.requiredFrozenReplayTokens(
            layerKinds: kinds,
            replayChunkTokens: min(512, plan.replayChunkCeiling))
        #expect(
            demanded <= plan.replayTokens,
            """
            planner granted \(plan.replayTokens) replay tokens but the pool \
            demands \(demanded) — PagedKVBackend.makeSequenceState(adopting:) refuses \
            this plan and the request cold-prefills, throwing away \
            \(plan.prefillTokensSaved) tokens of a confirmed cache hit
            """)
    }

    /// The clamp must bind BELOW M and release above it.
    ///
    /// Below M it is what makes the grant sufficient. Above M the request is
    /// doing an ordinary prefill of `[M, promptLength)` — 26k+ tokens on the
    /// harness prompt — and holding that to one window would trade a prefill
    /// throughput regression for the reuse win.
    @Test func theReplayChunkClampBindsOnlyBelowTheMatchedBoundary() throws {
        let paged = CBv2PrefixReuseCapability.derive(
            layerKinds: gptOssLayerKinds(), backend: .pagedFP16)
        let plan = try #require(
            paged.plan(matchedBoundary: 28416, maximumSequenceLength: 28672))

        #expect(plan.clampedChunk(start: plan.replayStart, proposed: 512) == 128)
        #expect(plan.clampedChunk(start: plan.matchedBoundary - 1, proposed: 512) == 1)
        #expect(
            plan.clampedChunk(start: plan.matchedBoundary, proposed: 512) == 512,
            "the prefill above M keeps the pool's full chunk")

        // gemma-4's ratio is the other way round (window 1,024 against a 512
        // chunk), so the clamp must be inert there — its replay chunk is
        // unchanged and this fix cannot regress the model that worked.
        let gemma = CBv2PrefixReuseCapability.derive(
            layerKinds: (0 ..< 30).map { index in
                CBv2LayerKind(
                    attention: index < 25 ? .slidingWindow(1024) : .full,
                    headDim: index < 25 ? 256 : 512,
                    kvHeads: 4,
                    queryHeads: 8)
            },
            backend: .pagedFP16)
        let gemmaPlan = try #require(
            gemma.plan(matchedBoundary: 28416, maximumSequenceLength: 28672))
        #expect(gemmaPlan.replayTokens == 26624, "25 x 1024 + 1024")
        #expect(gemmaPlan.clampedChunk(start: 0, proposed: 512) == 512)
    }

    // MARK: - 2. Adoption is accepted in this ratio

    /// The regression. Before the chunk cap, `derive` granted
    /// `2 * 8 + 8 == 24` replay tokens and the backend demanded
    /// `2 * 8 + 32 == 48`, so this threw `backendIneligible` and the engine
    /// cold-prefilled — the unit-scope shape of gpt-oss's `adoption_failed`.
    @Test func adoptionIsAcceptedWhenTheChunkExceedsTheWindow() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 96, seed: 0x1CE_B00D)

        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let plan = try #require(
            capability.plan(
                matchedBoundary: Self.matched, maximumSequenceLength: Self.maxLength))
        #expect(plan.replayTokens == Self.grantedReplay)
        #expect(plan.replayStart == Self.matched - Self.grantedReplay)

        let backend = try makeBackend(kinds)
        let prefix = try donateFullRows(
            model: model, backend: backend, prompt: prompt, matched: Self.matched)

        let state = try backend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)

        // Every row on the same logical cursor C, full rows frozen through M.
        for (index, kind) in kinds.enumerated() {
            let row = try #require(state[index] as? PagedSequenceKV)
            #expect(row.absoluteOffset == plan.replayStart, "layer \(index)")
            switch kind.attention {
            case .full:
                #expect(row.frozenHighWater == Self.matched)
            case .slidingWindow:
                #expect(row.frozenHighWater == 0)
                #expect(row.retainedCount == 0, "sliding rows rebuild from C")
            }
        }
        backend.release(state)
    }

    // MARK: - 3. And it is exact

    /// The safety half. Accepting the adoption is only correct if the sliding
    /// rows come back bit-exact; a fix that widens acceptance by lowering the
    /// bound reintroduces the frozen-full defect, where the windows are
    /// full-length but built from too few keys.
    ///
    /// Decoded over three full window turnovers so a single missing key
    /// cannot hide inside the prompt.
    @Test func frozenReplayIsTokenExactWhenTheChunkExceedsTheWindow() throws {
        let model = makeModel()
        let kinds = model.layerKinds
        let prompt = makePromptTokens(length: 96, seed: 0x1CE_B00D)
        let steps = 3 * Self.window

        let coldBackend = try makeBackend(kinds)
        let coldState = try coldBackend.makeSequenceState(
            layerKinds: kinds, promptLength: prompt.count, maxLength: Self.maxLength)
        let cold = run(
            model: model, backend: coldBackend, state: coldState, prompt: prompt,
            from: 0, steps: steps)
        coldBackend.release(coldState)

        let backend = try makeBackend(kinds)
        let prefix = try donateFullRows(
            model: model, backend: backend, prompt: prompt, matched: Self.matched)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: kinds, backend: .pagedFP16)
        let plan = try #require(
            capability.plan(
                matchedBoundary: Self.matched, maximumSequenceLength: Self.maxLength))
        let state = try backend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: Self.maxLength)

        let adopted = run(
            model: model, backend: backend, state: state, prompt: prompt,
            from: plan.replayStart, steps: steps, plan: plan)

        // The frozen rows must still hold the donated bytes at [0, M) after
        // the replay wrote straight through them.
        for (index, kind) in kinds.enumerated() {
            guard case .full = kind.attention else { continue }
            let row = try #require(state[index] as? PagedSequenceKV)
            let original = try #require(prefix[index])
            let (keys, values) = row.gatherRange(start: 0, count: Self.matched)
            #expect(
                maxAbsDiff(keys, original.keys) == 0,
                "layer \(index) frozen keys were overwritten by the replay")
            #expect(maxAbsDiff(values, original.values) == 0)
        }
        backend.release(state)

        let agreed = zip(adopted, cold).prefix(while: { $0.0 == $0.1 }).count
        #expect(
            adopted == cold,
            """
            frozen replay with chunk \(Self.chunkSize) > window \(Self.window) diverged \
            from a cold twin (\(agreed) of \(steps) tokens matched before divergence) — \
            the replay bound is too short for this ratio and the sliding rows came back \
            inexact
            """)
    }

    // MARK: - 4. The grant and the demand, across every ratio

    /// The invariant the gpt-oss defect violated, swept over the whole
    /// (windowCount, window, chunk) space rather than the two shapes that
    /// happen to ship.
    ///
    /// `CBv2PrefixReuseCapability.derive` grants
    /// `windowCount * maxWindow + maxWindow`; the pool demands
    /// `windowCount * maxWindow + replayChunkTokens`. The grant covers the
    /// demand iff the replay chunk never exceeds one window — which is what
    /// `replayChunkCeiling` guarantees, and what the old `maxPrefillChunk <=
    /// maxWindow` assumption merely hoped for.
    ///
    /// Written against the CAPPED chunk on purpose. Asserted against the raw
    /// pool chunk this reproduces the defect instead of the invariant, so a
    /// failure here means the grant and the cap have drifted apart — the two
    /// live in different files and nothing else couples them.
    @Test func theGrantCoversThePoolDemandForEveryWindowChunkRatio() throws {
        for windowCount in [1, 2, 5, 12, 25] {
            for window in [8, 16, 128, 512, 1024] {
                for chunk in [8, 16, 128, 512, 1024, 2048] {
                    var kinds = (0 ..< windowCount).map { _ in
                        CBv2LayerKind(
                            attention: .slidingWindow(window),
                            headDim: 64, kvHeads: 2, queryHeads: 4)
                    }
                    // One owning full layer downstream: what selects
                    // `.frozenFullReplay` over `.tailReplay`.
                    kinds.append(
                        CBv2LayerKind(
                            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4))

                    let capability = CBv2PrefixReuseCapability.derive(
                        layerKinds: kinds, backend: .pagedFP16)
                    #expect(capability.strategy == .frozenFullReplay)

                    let granted = window * (windowCount + 1)
                    #expect(capability.conservativeReplayBoundTokens == granted)

                    // A boundary comfortably past the bound, so `replayTokens`
                    // is the full grant rather than a clamp against M.
                    let plan = try #require(
                        capability.plan(
                            matchedBoundary: 2 * granted,
                            maximumSequenceLength: 4 * granted))
                    #expect(plan.replayTokens == granted)

                    let replayChunk =
                        plan.replayChunkCeiling > 0
                        ? min(chunk, plan.replayChunkCeiling) : chunk
                    let demanded = PagedKVBackend.requiredFrozenReplayTokens(
                        layerKinds: kinds, replayChunkTokens: replayChunk)
                    #expect(
                        plan.replayTokens >= demanded,
                        """
                        \(windowCount) sliding layers x window \(window), pool chunk \
                        \(chunk): granted \(plan.replayTokens) but the pool demands \
                        \(demanded) (replay chunk \(replayChunk)) — adoption refuses \
                        and the hit saves nothing
                        """)
                }
            }
        }
    }
}
