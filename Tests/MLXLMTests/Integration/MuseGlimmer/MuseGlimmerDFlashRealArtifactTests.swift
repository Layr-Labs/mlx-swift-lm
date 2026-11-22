// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

/// The published drafter on its real weights, against the independent oracle
/// (`MUSE_GLIMMER_ORACLE_DIR`) and oracle traces (`MUSE_GLIMMER_DFLASH_TRACE_DIR`).
/// Every test is skipped unless `MUSE_GLIMMER_NVFP4_DIR` and
/// `MUSE_GLIMMER_ASSISTANT_DIR` name the verified checkpoints. Run them one
/// process at a time while holding the GPU lane.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerDFlashRealArtifactTests {
    typealias Real = MuseGlimmerDFlashRealFixture

    /// A draft may differ from the oracle's only where the oracle's own
    /// top-two raw head values are closer than this. Raw head values are
    /// about five times the target's logit scale (`output_multiplier` 0.196).
    static let nearTieRaw: Float = 0.75

    private func ints(_ a: MLXArray) -> [Int] {
        a.reshaped([-1]).asType(.int32).asArray(Int32.self).map(Int.init)
    }

    /// Compares one block's drafts with the oracle's top five per position.
    /// Returns the mismatches that are not near ties and the largest
    /// difference of raw head values at the oracle's top-five ids.
    private func compare(
        logits: MLXArray, oracleTopIDs: [Int], oracleTopRaw: [Float]
    ) -> (drafts: [Int], mismatches: [[String: Any]], hard: Int, maxRawDifference: Float) {
        let drafts = ints(argMax(logits, axis: -1))
        let rows = drafts.count
        let ids = MLXArray(oracleTopIDs.prefix(rows * 5).map(Int32.init), [rows, 5])
        let ours = takeAlong(logits, ids, axis: -1).asArray(Float.self)
        var maxDifference: Float = 0
        var mismatches: [[String: Any]] = []
        var hard = 0
        for row in 0 ..< rows {
            for k in 0 ..< 5 {
                maxDifference = max(
                    maxDifference, abs(ours[row * 5 + k] - oracleTopRaw[row * 5 + k]))
            }
            guard drafts[row] != oracleTopIDs[row * 5] else { continue }
            let top = Array(oracleTopIDs[(row * 5) ..< (row * 5 + 5)])
            let rank = top.firstIndex(of: drafts[row])
            let gap = rank.map { oracleTopRaw[row * 5] - oracleTopRaw[row * 5 + $0] }
            let nearTie = gap.map { $0 <= Self.nearTieRaw } ?? false
            if !nearTie { hard += 1 }
            mismatches.append([
                "row": row + 1, "ours": drafts[row], "oracle": top[0],
                "oracle_rank_of_ours": rank ?? -1, "oracle_raw_gap": gap.map(Double.init) ?? -1,
                "near_tie": nearTie,
            ])
        }
        return (drafts, mismatches, hard, maxDifference)
    }

    @Test(.enabled(if: Real.hasWeights))
    func loaderAcceptsThePublishedArtifactAndTheHeadKernelMatchesTheFormula() throws {
        let real = try Real.shared.get()
        let c = real.configuration
        #expect(c == .museGlimmer30B)
        #expect(real.drafter.parameters().flattened().count == 58)
        #expect(real.drafter.artifactPayloadBytes == 5_111_970_304)
        #expect(real.drafter.projectionFormats.values.allSatisfy { $0 == .dense })
        #expect(real.embedding.shape == [202_048, 6656] && real.embedding.dtype == .bfloat16)
        // MLX's NVFP4 product against the dequantized rows of the same bytes.
        let hidden = MLXRandom.normal([3, c.hiddenSize], key: MLXRandom.key(5))
        let rows = 1000 ..< 1256
        let kernel = real.head(hidden)[0..., rows]
        let dense = matmul(hidden, try real.denseHeadRows(rows).T)
        let difference = abs(kernel - dense).max().item(Float.self)
        let scale = abs(dense).max().item(Float.self)
        #expect(difference <= 2e-4 * scale, "head kernel \(difference) of \(scale)")
        try Real.writeResult(
            "real-load",
            [
                "load_seconds": real.loadSeconds, "tensors": 58,
                "payload_bytes": real.drafter.artifactPayloadBytes,
                "head_kernel_max_abs_difference": Double(difference),
                "head_dense_max_abs": Double(scale),
            ])
    }

    /// Round 0 of the oracle's P1 fixture: the prompt's 71 context rows come
    /// from `P1_residual_stream.npz`, the expected drafts from `P1_dflash.npz`.
    @Test(.enabled(if: Real.hasWeights && Real.oracleDirectory != nil))
    func firstBlockOfTheOracleFixtureMatches() throws {
        let real = try Real.shared.get()
        let c = real.configuration
        let directory = URL(fileURLWithPath: Real.oracleDirectory!)
        let stream = try MuseGlimmerDFlashNPZ(
            directory.appendingPathComponent("P1_residual_stream.npz"))
        let fixture = try MuseGlimmerDFlashNPZ(directory.appendingPathComponent("P1_dflash.npz"))
        let prompt = try fixture.ints("input_ids")
        #expect(try stream.ints("input_ids") == prompt)
        #expect(try fixture.ints("mask_token_id") == [c.maskTokenID])
        #expect(try fixture.ints("r0_anchor_position") == [prompt.count])
        let outputs = try stream.array("layer_outputs")
        #expect(outputs.shape == [52, prompt.count, c.hiddenSize])
        // Output of 0-based layers 1, 13, 25, 37, 49, in that order.
        let features = concatenated(c.targetLayerIDs.map { outputs[$0] }, axis: -1)
        let anchor = try fixture.ints("r0_anchor")[0]
        let expected = try fixture.ints("r0_draft_ids")
        let topIDs = try fixture.ints("r0_draft_top5_ids")
        let topRaw = try fixture.floats("r0_draft_top5_raw_lm_head")

        var report: [String: Any] = ["oracle_drafts": expected, "anchor": anchor]
        for (name, dtype) in [("float32", DType.float32), ("bfloat16", .bfloat16)] {
            let model = try real.drafterCopy(dtype: dtype)
            let state = model.makeContextState()
            model.commit(features: features, at: 0, to: state)
            let drafted = real.drafts(model: model, state: state, anchor: anchor)
            let result = compare(logits: drafted.logits, oracleTopIDs: topIDs, oracleTopRaw: topRaw)
            report[name] = [
                "drafts": result.drafts, "mismatches": result.mismatches,
                "max_raw_difference_at_oracle_top5": Double(result.maxRawDifference),
            ]
            #expect(result.hard == 0, "\(name): \(result.mismatches)")
            if dtype == .float32 {
                // Same arithmetic as the oracle up to summation order.
                #expect(result.drafts == expected)
                #expect(result.maxRawDifference < 0.05)
            }
        }
        // The wrong tap layers are a different drafter: acceptance collapses.
        let shifted = concatenated(c.targetLayerIDs.map { outputs[$0 - 1] }, axis: -1)
        let state = real.drafter.makeContextState()
        real.drafter.commit(features: shifted, at: 0, to: state)
        let wrong = ints(real.drafts(model: real.drafter, state: state, anchor: anchor).ids)
        report["drafts_with_taps_one_layer_early"] = wrong
        #expect(wrong != expected)
        try Real.writeResult("real-oracle-p1-round0", report)
    }

    /// The pinned quantized artifact loads through the strict loader, is what
    /// the pinned plan gives when applied to the published drafter, and
    /// drafts the oracle's first block.
    @Test(
        .enabled(
            if: Real.hasWeights && Real.oracleDirectory != nil && Real.quantizedDirectory != nil))
    func pinnedQuantizedArtifactLoadsAndDrafts() throws {
        let real = try Real.shared.get()
        let c = real.configuration
        let pin = MuseGlimmerDFlashArtifactPin.museGlimmer30BQuantized
        let loaded = try MuseGlimmerDFlashDraftModel.load(
            directory: URL(fileURLWithPath: Real.quantizedDirectory!), pin: pin,
            maximumBytes: 2 << 30)
        #expect(loaded.artifactPayloadBytes < 2 << 30)
        #expect(loaded.projectionFormats == c.museGlimmer30BQuantizedPlan)
        let converted = try real.drafterCopy()
        try converted.quantize(c.museGlimmer30BQuantizedPlan)
        let expected = Dictionary(uniqueKeysWithValues: converted.parameters().flattened())
        for (name, value) in loaded.parameters().flattened() {
            #expect(value.asData().data == expected[name]?.asData().data, "\(name)")
        }
        // The published pin does not accept the quantized artifact.
        #expect(throws: MuseGlimmerDFlashError.self) {
            try MuseGlimmerDFlashDraftModel.load(
                directory: URL(fileURLWithPath: Real.quantizedDirectory!),
                pin: .museGlimmer30BBF16, maximumBytes: 6 << 30)
        }

        let directory = URL(fileURLWithPath: Real.oracleDirectory!)
        let stream = try MuseGlimmerDFlashNPZ(
            directory.appendingPathComponent("P1_residual_stream.npz"))
        let fixture = try MuseGlimmerDFlashNPZ(directory.appendingPathComponent("P1_dflash.npz"))
        let outputs = try stream.array("layer_outputs")
        let features = concatenated(c.targetLayerIDs.map { outputs[$0] }, axis: -1)
        let state = loaded.makeContextState()
        loaded.commit(features: features, at: 0, to: state)
        let drafted = real.drafts(
            model: loaded, state: state, anchor: try fixture.ints("r0_anchor")[0])
        let result = compare(
            logits: drafted.logits, oracleTopIDs: try fixture.ints("r0_draft_top5_ids"),
            oracleTopRaw: try fixture.floats("r0_draft_top5_raw_lm_head"))
        // The first 7 are what the engine uses; the oracle accepts 2 of them.
        #expect(Array(result.drafts.prefix(2)) == Array(try fixture.ints("r0_draft_ids").prefix(2)))
        try Real.writeResult(
            "real-quantized-artifact",
            [
                "payload_bytes": loaded.artifactPayloadBytes, "drafts": result.drafts,
                "oracle_drafts": try fixture.ints("r0_draft_ids"),
                "mismatches_vs_bf16_oracle": result.mismatches,
                "max_raw_difference_at_oracle_top5": Double(result.maxRawDifference),
            ])
    }

    /// Every round of every oracle trace, with the context committed in the
    /// increments the engine would deliver.
    @Test(.enabled(if: Real.hasWeights && Real.traceDirectory != nil))
    func everyTraceRoundMatchesTheOracleDrafts() throws {
        let real = try Real.shared.get()
        var report: [String: Any] = [:]
        for trace in try MuseGlimmerDFlashTrace.all() {
            let model = real.drafter
            let state = model.makeContextState()
            var rounds = 0
            var equalRounds = 0
            var tokenMismatches = 0
            var hard = 0
            var maxRaw: Float = 0
            var details: [[String: Any]] = []
            for (round, position) in trace.anchorPositions.enumerated() {
                model.commit(
                    features: try trace.features(state.nextPosition ..< position),
                    at: state.nextPosition, to: state)
                let drafted = real.drafts(
                    model: model, state: state, anchor: trace.tokens[position])
                let result = compare(
                    logits: drafted.logits, oracleTopIDs: trace.oracleTopIDs[round],
                    oracleTopRaw: trace.oracleTopRaw[round])
                eval(state.arrays)
                rounds += 1
                equalRounds += result.mismatches.isEmpty ? 1 : 0
                tokenMismatches += result.mismatches.count
                hard += result.hard
                maxRaw = max(maxRaw, result.maxRawDifference)
                for mismatch in result.mismatches {
                    details.append(mismatch.merging(["round": round]) { a, _ in a })
                }
            }
            #expect(hard == 0, "\(trace.name): \(details)")
            report[trace.name] = [
                "prompt_tokens": trace.promptLength, "rounds": rounds,
                "rounds_with_identical_drafts": equalRounds,
                "draft_tokens": rounds * 15, "token_mismatches": tokenMismatches,
                "mismatches_beyond_near_tie": hard,
                "max_raw_difference_at_oracle_top5": Double(maxRaw), "mismatches": details,
                "oracle_accepted_per_round": trace.oracleAccepted,
            ]
        }
        #expect(!report.isEmpty)
        try Real.writeResult("real-oracle-traces-bf16", report)
    }
}
