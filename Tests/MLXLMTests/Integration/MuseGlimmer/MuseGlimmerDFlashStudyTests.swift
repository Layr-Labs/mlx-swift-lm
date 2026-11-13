// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

/// Measurements on the published drafter over oracle traces: which
/// quantization keeps the drafts, and how much context a block needs. They
/// write JSON results and assert only what must hold for any weights. Skipped
/// without the real checkpoints and traces; run while holding the GPU lane.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerDFlashStudyTests {
    typealias Real = MuseGlimmerDFlashRealFixture
    typealias Format = MuseGlimmerDFlashProjectionFormat

    static let enabled = Real.hasWeights && Real.traceDirectory != nil
    /// Draft depth the engine verifies (`CBv2MTPConfig.testedMaxDraftTokens`).
    static let engineDepth = 7

    /// A named quantization plan; nil for the dense BF16 drafter.
    static func plan(
        _ name: String, _ c: MuseGlimmerDFlashConfiguration
    ) -> [String: Format]? {
        let paths = MuseGlimmerDFlashInventory(c).projections.map(\.path)
        func all(_ format: (String) -> Format) -> [String: Format] {
            Dictionary(uniqueKeysWithValues: paths.map { ($0, format($0)) })
        }
        switch name {
        case "bf16": return nil
        case "q4g64": return all { _ in .affine(bits: 4, groupSize: 64) }
        case "q4g32": return all { _ in .affine(bits: 4, groupSize: 32) }
        case "q5g64": return all { _ in .affine(bits: 5, groupSize: 64) }
        case "q6g64": return all { _ in .affine(bits: 6, groupSize: 64) }
        case "q8g64": return all { _ in .affine(bits: 8, groupSize: 64) }
        case "q3g64": return all { _ in .affine(bits: 3, groupSize: 64) }
        case "q4g64-fc8":
            return all { .affine(bits: $0 == "encoder.fc" ? 8 : 4, groupSize: 64) }
        case "q4mlp-q8rest":
            return all { .affine(bits: $0.contains(".mlp.") ? 4 : 8, groupSize: 64) }
        case "q5mlp-q8rest": return c.museGlimmer30BQuantizedPlan
        case "q4mlp-q6rest":
            return all { .affine(bits: $0.contains(".mlp.") ? 4 : 6, groupSize: 64) }
        default: return [:]
        }
    }

    static var variantNames: [String] {
        (Real.environment["MUSE_GLIMMER_DFLASH_VARIANTS"]
            ?? "bf16,q8g64,q6g64,q5g64,q5mlp-q8rest,q4mlp-q8rest,q4mlp-q6rest,q4g64-fc8,q4g32,q4g64,q3g64")
            .split(separator: ",").map(String.init)
    }

    /// Anchor positions of a trace: every `stride`-th generated position that
    /// has a full block of continuation behind it.
    static func anchors(_ trace: MuseGlimmerDFlashTrace, stride step: Int) -> [Int] {
        Array(stride(from: trace.promptLength, to: trace.tokens.count - 16, by: step))
    }

    private func ints(_ a: MLXArray) -> [Int] {
        a.reshaped([-1]).asType(.int32).asArray(Int32.self).map(Int.init)
    }

    private func mean(_ values: [Int]) -> Double {
        values.isEmpty ? 0 : Double(values.reduce(0, +)) / Double(values.count)
    }

    private func median(_ values: [Double]) -> Double {
        values.isEmpty ? 0 : values.sorted()[values.count / 2]
    }

    /// Drafts of `model` at every anchor of every trace, with the context
    /// committed incrementally, and the wall time of each block.
    private func drafts(
        _ model: MuseGlimmerDFlashDraftModel, real: Real, traces: [MuseGlimmerDFlashTrace],
        stride step: Int
    ) throws -> (drafts: [String: [Int: [Int]]], seconds: [Double], commitSeconds: [Double]) {
        var result: [String: [Int: [Int]]] = [:]
        var seconds: [Double] = []
        var commitSeconds: [Double] = []
        for trace in traces {
            let state = model.makeContextState()
            var perAnchor: [Int: [Int]] = [:]
            for position in Self.anchors(trace, stride: step) {
                let commitStart = Date()
                model.commit(
                    features: try trace.features(state.nextPosition ..< position),
                    at: state.nextPosition, to: state)
                eval(state.arrays)
                commitSeconds.append(Date().timeIntervalSince(commitStart))
                let start = Date()
                let ids = real.drafts(model: model, state: state, anchor: trace.tokens[position])
                    .ids
                eval(ids)
                seconds.append(Date().timeIntervalSince(start))
                perAnchor[position] = ints(ids)
            }
            result[trace.name] = perAnchor
        }
        return (result, seconds, commitSeconds)
    }

    @Test(.enabled(if: MuseGlimmerDFlashStudyTests.enabled))
    func quantizedDraftersAgainstTheDenseDrafter() throws {
        let real = try Real.shared.get()
        let c = real.configuration
        let traces = try MuseGlimmerDFlashTrace.all()
        try #require(!traces.isEmpty)
        let step = Int(Real.environment["MUSE_GLIMMER_DFLASH_ANCHOR_STRIDE"] ?? "2") ?? 2
        var reference: [String: [Int: [Int]]] = [:]
        var rows: [[String: Any]] = []
        for name in ["bf16"] + Self.variantNames.filter({ $0 != "bf16" }) {
            let model = try real.drafterCopy()
            let quantizeStart = Date()
            if let plan = Self.plan(name, c) {
                try #require(!plan.isEmpty, "unknown variant \(name)")
                try model.quantize(plan)
            }
            let quantizeSeconds = Date().timeIntervalSince(quantizeStart)
            let measured = try drafts(model, real: real, traces: traces, stride: step)
            if name == "bf16" { reference = measured.drafts }
            var tokens = 0
            var equal = 0
            var equalFirst = 0
            var identicalFirst = 0
            var anchors = 0
            var accepted7: [Int] = []
            var accepted15: [Int] = []
            var perTrace: [String: Any] = [:]
            for trace in traces {
                var traceAccepted7: [Int] = []
                for (position, ids) in measured.drafts[trace.name] ?? [:] {
                    let expected = reference[trace.name]![position]!
                    anchors += 1
                    tokens += ids.count
                    equal += zip(ids, expected).filter { $0 == $1 }.count
                    let first = zip(ids, expected).prefix(Self.engineDepth)
                    equalFirst += first.filter { $0 == $1 }.count
                    identicalFirst += first.allSatisfy { $0 == $1 } ? 1 : 0
                    if let n = trace.accepted(ids, after: position, limit: Self.engineDepth) {
                        accepted7.append(n)
                        traceAccepted7.append(n)
                    }
                    if let n = trace.accepted(ids, after: position, limit: 15) {
                        accepted15.append(n)
                    }
                }
                perTrace[trace.name] = ["mean_accepted_k7": mean(traceAccepted7)]
            }
            let bytes = model.artifactPayloadBytes
            rows.append([
                "variant": name, "payload_bytes": bytes,
                "payload_gib": Double(bytes) / Double(1 << 30), "under_2_gib": bytes < 2 << 30,
                "anchors": anchors, "draft_tokens": tokens,
                "agreement_15": Double(equal) / Double(max(tokens, 1)),
                "agreement_first_7": Double(equalFirst)
                    / Double(max(anchors * Self.engineDepth, 1)),
                "blocks_with_identical_first_7": Double(identicalFirst) / Double(max(anchors, 1)),
                "mean_accepted_k7": mean(accepted7), "mean_accepted_k15": mean(accepted15),
                "median_block_ms": median(measured.seconds) * 1000,
                "median_commit_ms": median(measured.commitSeconds) * 1000,
                "quantize_seconds": quantizeSeconds, "per_trace": perTrace,
            ])
            if name == "bf16" {
                #expect(equal == tokens)
            }
        }
        try Real.writeResult(
            "study-quantization",
            [
                "anchor_stride": step, "engine_depth": Self.engineDepth,
                "traces": traces.map {
                    [
                        "name": $0.name, "prompt_tokens": $0.promptLength,
                        "committed": $0.committed,
                    ] as [String: Any]
                },
                "variants": rows,
                "note": "debug build, M3 Ultra; times include the 7..15-row head product",
            ])
    }

    /// A block with only the last N context rows, as after a restore that
    /// leaves the drafter without older context.
    @Test(.enabled(if: MuseGlimmerDFlashStudyTests.enabled))
    func draftsWithTruncatedContext() throws {
        let real = try Real.shared.get()
        let c = real.configuration
        let traces = try MuseGlimmerDFlashTrace.all()
        try #require(!traces.isEmpty)
        let step = Int(Real.environment["MUSE_GLIMMER_DFLASH_ANCHOR_STRIDE"] ?? "2") ?? 2
        let variant = Real.environment["MUSE_GLIMMER_DFLASH_TRUNCATION_VARIANT"] ?? "bf16"
        let model = try real.drafterCopy()
        if let plan = Self.plan(variant, c) { try model.quantize(plan) }
        let lengths = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        struct Tally {
            var anchors = 0
            var equal15 = 0
            var equal7 = 0
            var accepted7: [Int] = []
            var accepted15: [Int] = []
            var fullAccepted7: [Int] = []
            var fullAccepted15: [Int] = []
        }
        var tallies = [Int: Tally](uniqueKeysWithValues: lengths.map { ($0, Tally()) })
        for trace in traces {
            let state = model.makeContextState()
            for position in Self.anchors(trace, stride: step) {
                model.commit(
                    features: try trace.features(state.nextPosition ..< position),
                    at: state.nextPosition, to: state)
                eval(state.arrays)
                let embeddings = real.blockEmbeddings(anchor: trace.tokens[position], model: model)
                let visible = try #require(state.visibleContext())
                let held = state.heldPositions
                func draft(last rows: Int?) -> [Int] {
                    var context: (keys: MLXArray, values: MLXArray)?
                    var mask: MLXArray?
                    if let rows {
                        let kept = held.enumerated().filter { $0.element >= position - rows }
                        if !kept.isEmpty {
                            let index = MLXArray(kept.map { Int32($0.offset) })
                            context = (
                                take(visible.keys, index, axis: 2),
                                take(visible.values, index, axis: 2)
                            )
                            mask = MuseGlimmerDFlashContextState.visibility(
                                anchor: position, blockSize: c.blockSize, window: c.slidingWindow,
                                contextPositions: kept.map(\.element)
                            ).map { MLXArray($0, [c.blockSize, kept.count + c.blockSize]) }
                        }
                    } else {
                        context = (visible.keys, visible.values)
                        mask = visible.mask
                    }
                    let hidden = model.blockForward(
                        embeddings: embeddings, at: position, context: context, mask: mask)
                    return ints(argMax(real.head(hidden[1..., 0...]), axis: -1))
                }
                let full = draft(last: nil)
                for rows in lengths where rows <= held.count {
                    let truncated = rows == held.count ? full : draft(last: rows)
                    var tally = tallies[rows]!
                    tally.anchors += 1
                    tally.equal15 += zip(truncated, full).filter { $0 == $1 }.count
                    tally.equal7 +=
                        zip(truncated, full).prefix(Self.engineDepth).filter { $0 == $1 }.count
                    for (limit, isSeven) in [(Self.engineDepth, true), (15, false)] {
                        guard let n = trace.accepted(truncated, after: position, limit: limit),
                            let m = trace.accepted(full, after: position, limit: limit)
                        else { continue }
                        if isSeven {
                            tally.accepted7.append(n)
                            tally.fullAccepted7.append(m)
                        } else {
                            tally.accepted15.append(n)
                            tally.fullAccepted15.append(m)
                        }
                    }
                    tallies[rows] = tally
                }
            }
        }
        let rows: [[String: Any]] = lengths.compactMap { length in
            let t = tallies[length]!
            guard t.anchors > 0 else { return nil }
            return [
                "context_rows": length, "anchors": t.anchors,
                "agreement_15_with_full_context": Double(t.equal15) / Double(t.anchors * 15),
                "agreement_first_7_with_full_context":
                    Double(t.equal7) / Double(t.anchors * Self.engineDepth),
                "mean_accepted_k7": mean(t.accepted7),
                "mean_accepted_k7_full_context": mean(t.fullAccepted7),
                "mean_accepted_k15": mean(t.accepted15),
                "mean_accepted_k15_full_context": mean(t.fullAccepted15),
            ]
        }
        try Real.writeResult(
            "study-context-truncation-\(variant)",
            [
                "variant": variant, "anchor_stride": step, "rows": rows,
                "note": "context_rows = most recent committed positions visible to the block; "
                    + "each row compares with the same anchors' drafts over all held rows "
                    + "(at most 2,048)",
            ])
    }

    /// Wall time of the drafter's parts on a full context ring: one block
    /// forward, the seven head rows the engine uses, and context commits.
    @Test(.enabled(if: MuseGlimmerDFlashStudyTests.enabled))
    func blockHeadAndCommitTimes() throws {
        let real = try Real.shared.get()
        let c = real.configuration
        let trace = try #require(
            try MuseGlimmerDFlashTrace.all().max { $0.committed < $1.committed })
        let position = trace.committed
        let features = try trace.features(max(0, position - c.slidingWindow) ..< position)
        eval(features)
        func time(_ runs: Int = 15, _ body: () -> [MLXArray]) -> Double {
            for _ in 0 ..< 3 { eval(body()) }
            var samples: [Double] = []
            for _ in 0 ..< runs {
                let start = Date()
                eval(body())
                samples.append(Date().timeIntervalSince(start) * 1000)
            }
            return median(samples)
        }
        var rows: [[String: Any]] = []
        let names =
            (Real.environment["MUSE_GLIMMER_DFLASH_TIMING_VARIANTS"]
            ?? "bf16,q6g64,q5mlp-q8rest,q4g64").split(separator: ",").map(String.init)
        for name in names {
            let model = try real.drafterCopy()
            if let plan = Self.plan(name, c) { try model.quantize(plan) }
            let state = model.makeContextState()
            state.restart(at: position - features.dim(0))
            model.commit(features: features, at: state.nextPosition, to: state)
            eval(state.arrays)
            let embeddings = real.blockEmbeddings(anchor: trace.tokens[position], model: model)
            eval(embeddings)
            let block = model.draftBlock(embeddings: embeddings, state: state)
            eval(block)
            let seven = block[1 ..< 8, 0...]
            // Commits go to scratch states so the timed ring stays full.
            func commit(_ count: Int) -> [MLXArray] {
                let scratch = model.makeContextState()
                model.commit(features: features[0 ..< count, 0...], at: 0, to: scratch)
                return scratch.arrays
            }
            rows.append([
                "variant": name, "context_rows": state.count,
                "block_forward_ms": time {
                    [model.draftBlock(embeddings: embeddings, state: state)]
                },
                "head_7_rows_ms": time { [argMax(real.head(seven), axis: -1)] },
                "block_and_head_7_ms": time {
                    let hidden = model.draftBlock(embeddings: embeddings, state: state)
                    return [argMax(real.head(hidden[1 ..< 8, 0...]), axis: -1)]
                },
                "block_and_head_15_ms": time {
                    let hidden = model.draftBlock(embeddings: embeddings, state: state)
                    return [argMax(real.head(hidden[1..., 0...]), axis: -1)]
                },
                "commit_1_row_ms": time { commit(1) },
                "commit_8_rows_ms": time { commit(8) },
                "commit_512_rows_ms": time(5) { commit(min(512, features.dim(0))) },
                "commit_2048_rows_ms": time(5) { commit(min(2048, features.dim(0))) },
            ])
        }
        try Real.writeResult(
            "study-timings",
            [
                "trace": trace.name, "rows": rows,
                "note": "median wall ms, debug build, M3 Ultra, GPU lane held; a commit "
                    + "includes allocating its scratch ring",
            ])
    }

    /// Converts the dense drafter to the named variant, writes the artifact,
    /// loads it back through the strict loader and checks the drafts.
    @Test(
        .enabled(
            if: MuseGlimmerDFlashStudyTests.enabled && Real.artifactDirectory != nil
                && Real.environment["MUSE_GLIMMER_DFLASH_ARTIFACT_VARIANT"] != nil))
    func writesTheQuantizedArtifact() throws {
        let real = try Real.shared.get()
        let c = real.configuration
        let variant = Real.environment["MUSE_GLIMMER_DFLASH_ARTIFACT_VARIANT"]!
        let plan = try #require(Self.plan(variant, c))
        try #require(!plan.isEmpty)
        let model = try real.drafterCopy()
        try model.quantize(plan)
        let source =
            "meta-models/Muse-Glimmer-30B-assistant@c676cce91658f9568aaf264a416211e2782735b7 "
            + "model.safetensors sha256 \(MuseGlimmerDFlashArtifactPin.museGlimmer30BBF16.weightsSHA256); "
            + "MLX affine quantization, variant \(variant)"
        let root = URL(fileURLWithPath: Real.artifactDirectory!)
        let directory = root.appendingPathComponent(variant)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let written = try model.writeArtifact(to: directory, source: source)
        #expect(written.manifest.totalBytes < 2 << 30)
        // The same conversion again gives the same bytes.
        let second = root.appendingPathComponent(".determinism-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: second, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: second) }
        let again = try real.drafterCopy()
        try again.quantize(plan)
        let rewritten = try again.writeArtifact(to: second, source: source)
        #expect(rewritten.pin == written.pin && rewritten.manifest == written.manifest)

        let loaded = try MuseGlimmerDFlashDraftModel.load(
            directory: directory, pin: written.pin, maximumBytes: written.manifest.totalBytes)
        #expect(loaded.projectionFormats == model.projectionFormats)
        let traces = try MuseGlimmerDFlashTrace.all()
        let memory = try drafts(model, real: real, traces: traces, stride: 16)
        let disk = try drafts(loaded, real: real, traces: traces, stride: 16)
        #expect(memory.drafts == disk.drafts)
        try Real.writeResult(
            "artifact-\(variant)",
            [
                "variant": variant, "directory": directory.path,
                "total_bytes": written.manifest.totalBytes,
                "files": written.manifest.files.mapValues {
                    ["bytes": $0.bytes, "sha256": $0.sha256] as [String: Any]
                },
                "tensor_count": written.manifest.tensorCount,
                "deterministic_rewrite_identical": rewritten.pin == written.pin,
                "reloaded_drafts_identical": memory.drafts == disk.drafts,
                "anchors_checked": disk.drafts.values.reduce(0) { $0 + $1.count },
            ])
    }
}
