// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Arrays of an uncompressed `.npz` (numpy `savez`): a ZIP of stored `.npy`
/// members. Only what the oracle fixtures use: little-endian int32, float32
/// and float64 in C order.
struct MuseGlimmerNPZ {
    enum Failure: Error { case malformed(String) }
    struct Entry {
        let descr: String
        let shape: [Int]
        let data: Data
    }
    let entries: [String: Entry]

    init(url: URL) throws {
        let file = try Data(contentsOf: url)
        func u16(_ at: Int) -> Int { Int(file[at]) | Int(file[at + 1]) << 8 }
        func u32(_ at: Int) -> Int { u16(at) | u16(at + 2) << 16 }
        guard file.count >= 22 else { throw Failure.malformed("size") }
        var end = file.count - 22
        while end >= 0, u32(end) != 0x0605_4b50 { end -= 1 }
        guard end >= 0 else { throw Failure.malformed("end record") }
        let count = u16(end + 10)
        var cursor = u32(end + 16)
        var entries = [String: Entry]()
        for _ in 0 ..< count {
            guard u32(cursor) == 0x0201_4b50 else { throw Failure.malformed("directory") }
            guard u16(cursor + 10) == 0 else { throw Failure.malformed("compressed member") }
            let size = u32(cursor + 24)
            let nameLength = u16(cursor + 28)
            let local = u32(cursor + 42)
            let name = String(
                decoding: file[(cursor + 46) ..< (cursor + 46 + nameLength)], as: UTF8.self)
            cursor += 46 + nameLength + u16(cursor + 30) + u16(cursor + 32)
            guard u32(local) == 0x0403_4b50, size < 0xFFFF_FFFF else {
                throw Failure.malformed("local header")
            }
            let start = local + 30 + u16(local + 26) + u16(local + 28)
            let member = file.subdata(in: start ..< start + size)
            guard member.count > 12, member[0] == 0x93, member[1 ... 5].elementsEqual("NUMPY".utf8)
            else { throw Failure.malformed(name) }
            let headerStart = member[6] == 1 ? 10 : 12
            let headerLength =
                member[6] == 1
                ? Int(member[8]) | Int(member[9]) << 8
                : Int(member[8]) | Int(member[9]) << 8 | Int(member[10]) << 16
                    | Int(member[11]) << 24
            let header = String(
                decoding: member[headerStart ..< headerStart + headerLength], as: UTF8.self)
            guard let descr = header.firstMatch(of: /'descr': '([^']+)'/)?.1,
                let shape = header.firstMatch(of: /'shape': \(([^)]*)\)/)?.1,
                header.contains("'fortran_order': False")
            else { throw Failure.malformed(name) }
            entries[name.replacingOccurrences(of: ".npy", with: "")] = Entry(
                descr: String(descr),
                shape: shape.split(separator: ",").compactMap {
                    Int($0.trimmingCharacters(in: .whitespaces))
                },
                data: member.subdata(in: (headerStart + headerLength) ..< member.count))
        }
        self.entries = entries
    }

    private func values<T>(_ name: String, _ descr: String, as type: T.Type) throws -> [T] {
        guard let entry = entries[name], entry.descr == descr,
            entry.data.count == entry.shape.reduce(1, *) * MemoryLayout<T>.size
        else { throw Failure.malformed(name) }
        return entry.data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
    }

    func ints(_ name: String) throws -> [Int32] { try values(name, "<i4", as: Int32.self) }
    func floats(_ name: String) throws -> [Float] { try values(name, "<f4", as: Float.self) }
    func doubles(_ name: String) throws -> [Double] { try values(name, "<f8", as: Double.self) }
    func shape(_ name: String) -> [Int] { entries[name]?.shape ?? [] }
}

/// The pinned NVIDIA checkpoint, staged once for every test of the process.
enum MuseGlimmerRealArtifact {
    static var directory: URL? {
        ProcessInfo.processInfo.environment["MUSE_GLIMMER_NVFP4_DIR"].map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
    }
    static var oracleDirectory: URL? {
        ProcessInfo.processInfo.environment["MUSE_GLIMMER_ORACLE_DIR"].map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
    }
    static var hasOracle: Bool { directory != nil && oracleDirectory != nil }

    struct Loaded {
        let installed: MuseGlimmerInstallation.Installed
        let loadSeconds: Double
        let activeBytesAfterLoad: Int
        let peakBytesDuringLoad: Int
        let footprintAfterLoad: Int
    }

    static func footprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? Int(info.phys_footprint) : -1
    }

    nonisolated(unsafe) static let loaded: Result<Loaded, Error> = Result {
        guard let directory else { throw MuseGlimmerNPZ.Failure.malformed("no directory") }
        Memory.peakMemory = 0
        let start = Date()
        let installed = try MuseGlimmerInstallation.stage(directory: directory)
        let seconds = Date().timeIntervalSince(start)
        return Loaded(
            installed: installed, loadSeconds: seconds, activeBytesAfterLoad: Memory.activeMemory,
            peakBytesDuringLoad: Memory.peakMemory, footprintAfterLoad: footprint())
    }

    /// One line per measurement, greppable in the test log, and appended to
    /// `MUSE_GLIMMER_EVIDENCE_FILE` when that is set.
    static func record(_ line: String) {
        let text = "MUSE_GLIMMER_EVIDENCE " + line
        print(text)
        guard let path = ProcessInfo.processInfo.environment["MUSE_GLIMMER_EVIDENCE_FILE"],
            let data = (text + "\n").data(using: .utf8)
        else { return }
        if let handle = FileHandle(forWritingAtPath: path) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: URL(fileURLWithPath: path))
        }
    }
}

/// The real checkpoint against the independent numpy oracle (float32, FP8
/// activations saturating). Skipped unless `MUSE_GLIMMER_NVFP4_DIR` (and, for
/// the comparisons, `MUSE_GLIMMER_ORACLE_DIR`) are set. Run under the GPU lane.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerRealArtifactTests {
    private func gib(_ bytes: Int) -> String {
        String(format: "%.2f", Double(bytes) / Double(1 << 30))
    }

    @Test(.enabled(if: MuseGlimmerRealArtifact.directory != nil))
    func pinnedArtifactLoadsWithThePublishedCensus() throws {
        let loaded = try MuseGlimmerRealArtifact.loaded.get()
        let receipt = loaded.installed.receipt
        #expect(receipt.indexSHA256 == MuseGlimmerCheckpointPin.index)
        #expect(receipt.shardSHA256 == MuseGlimmerCheckpointPin.shards)
        #expect(receipt.plan.count(.fp8) == 180 && receipt.plan.count(.nvfp4) == 213)
        #expect(receipt.plan.count(.dense) == 24)
        #expect(receipt.plan.projections["lm_head"] == .nvfp4)
        #expect(receipt.plan.visionTensors.count == 809)
        #expect(receipt.plan.textPayloadBytes == 20_825_654_516)
        #expect(receipt.plan.visionPayloadBytes == 3_843_691_520)
        #expect(receipt.loadedTensorCount == 2222 - 809)
        #expect(receipt.activationOverflow == .saturating)
        let model = loaded.installed.model
        #expect(model.configuration == MuseGlimmerInstallation.pinnedConfiguration)
        #expect(model.activationDType == .bfloat16 && model.cbv2LayerKinds.count == 52)
        // The dense projections are the attention projections of layers 7 to 12.
        let dense = receipt.plan.projections.filter { $0.value == .dense }.keys.sorted()
        #expect(dense.allSatisfy { $0.contains(".self_attn.") && !$0.hasSuffix("o_proj") })
        #expect(
            Set(dense.map { $0.split(separator: ".")[2] }) == ["7", "8", "9", "10", "11", "12"])
        MuseGlimmerRealArtifact.record(
            "load seconds=\(String(format: "%.1f", loaded.loadSeconds)) "
                + "mlx_active_gib=\(gib(loaded.activeBytesAfterLoad)) "
                + "mlx_peak_gib=\(gib(loaded.peakBytesDuringLoad)) "
                + "footprint_gib=\(gib(loaded.footprintAfterLoad)) "
                + "text_payload_gib=\(gib(receipt.plan.textPayloadBytes))")
    }

    private func relative(_ ours: [Float], _ oracle: [Float]) -> Float {
        var difference: Float = 0
        var norm: Float = 0
        for (a, b) in zip(ours, oracle) {
            difference += (a - b) * (a - b)
            norm += b * b
        }
        return (difference / max(norm, 1e-30)).squareRoot()
    }

    /// One uncached pass over the whole prompt: the residual stream after every
    /// layer at the last position, the drafter layers' last four rows, the
    /// final hidden and the last-row logits, each against the oracle.
    ///
    /// Tolerances. The oracle is float32; this tower runs BF16 (8 significant
    /// bits). The FP8 layers round their input activations to an 8-bit grid, so
    /// a last-bit difference before a rounding boundary becomes a 6% step in
    /// that element and spreads from there: the oracle's own single-pass and
    /// chunked float32 runs differ by 2 to 11% in relative L2 over layers 13 to
    /// 49 and by up to 1.2 in a logit. Measured here on 2026-11-13 (M3 Ultra):
    /// 1.2% at layer 1, 3% at 13, 3 to 4% at 25, 5 to 8% at 37, 8 to 9% at 49,
    /// 10 to 14% in the final hidden, at most 1.1 in a top-50 logit. The
    /// bounds are about twice that; the choice of token is the firm criterion.
    @Test(
        .enabled(if: MuseGlimmerRealArtifact.hasOracle), arguments: ["P1", "P2", "P3"])
    func uncachedForwardTracksTheOracle(name: String) throws {
        let model = try MuseGlimmerRealArtifact.loaded.get().installed.model
        let oracle = try MuseGlimmerNPZ(
            url: MuseGlimmerRealArtifact.oracleDirectory!.appendingPathComponent(
                "\(name)_target.npz"))
        let ids = try oracle.ints("input_ids")
        let layers = model.configuration.hiddenLayers
        let embeddings = model.scaledTokenEmbeddings(MLXArray(ids).reshaped(1, -1))
        let (residual, captured) = model.model.residual(
            embeddings, cache: nil, capture: Array(0 ..< layers))
        let normed = model.model.norm(residual)
        let logits = model.outputLogits(normed[0..., -1, 0...]).asType(.float32)
        let probeIndex = MLXArray(try oracle.ints("probe_index"))
        let lastRows = captured.map { $0[0, -1].asType(.float32) }
        let samples = stacked(lastRows.map { $0[probeIndex] })
        let norms = stacked(lastRows.map { sqrt(($0 * $0).sum()) })
        let drafterLayers = try oracle.ints("hidden_layer_ids").map(Int.init)
        let lastFour = drafterLayers.map { captured[$0][0, (-4)..., 0...].asType(.float32) }
        let embedLast = embeddings[0, -1].asType(.float32)
        let finalLast = normed[0, -1].asType(.float32)
        try MLX.withError { errors in
            eval([logits, samples, norms, embedLast, finalLast] + lastFour)
            try errors.check()
        }

        // First-layer input: only BF16 rounding separates the two.
        let embedError = relative(embedLast.asArray(Float.self), try oracle.floats("embed_last"))
        #expect(embedError < 0.004)

        // Per-layer residual at the last position: norm and 16 samples.
        let oracleNorms = try oracle.doubles("probe_l2").map(Float.init)
        let oracleSamples = try oracle.floats("probe_sample")
        let ourNorms = norms.asArray(Float.self)
        let ourSamples = samples.asArray(Float.self)
        var sampleErrors = [Float]()
        var normErrors = [Float]()
        for layer in 0 ..< layers {
            let range = (layer * 16) ..< ((layer + 1) * 16)
            // 16 samples of 6,656: scale by the row norm, not by the samples' own.
            var difference: Float = 0
            for index in range {
                difference +=
                    (ourSamples[index] - oracleSamples[index])
                    * (ourSamples[index] - oracleSamples[index])
            }
            sampleErrors.append(
                (difference / 16).squareRoot() / (oracleNorms[layer] / Float(6656).squareRoot()))
            normErrors.append(abs(ourNorms[layer] - oracleNorms[layer]) / oracleNorms[layer])
        }
        let firstOver = sampleErrors.firstIndex { $0 > 0.05 } ?? -1
        MuseGlimmerRealArtifact.record(
            "\(name) tokens=\(ids.count) embed_rel=\(embedError) "
                + "first_layer_sample_rms_over_5pct=\(firstOver) "
                + "sample_rms_by_layer=\(sampleErrors.map { String(format: "%.4f", $0) }.joined(separator: ",")) "
                + "norm_rel_by_layer=\(normErrors.map { String(format: "%.4f", $0) }.joined(separator: ","))"
        )
        #expect(sampleErrors[0] < 0.03 && sampleErrors[1] < 0.05)
        #expect(sampleErrors.allSatisfy { $0 < 0.30 })
        #expect(normErrors.allSatisfy { $0 < 0.05 })

        // Drafter hidden rows.
        var hiddenErrors = [String]()
        for (slot, layer) in drafterLayers.enumerated() {
            let key = String(format: "hidden_L%02d_last4", layer)
            let error = relative(lastFour[slot].asArray(Float.self), try oracle.floats(key))
            hiddenErrors.append("L\(layer)=\(String(format: "%.4f", error))")
            #expect(error < (layer <= 1 ? 0.03 : 0.20))
        }
        let finalError = relative(
            finalLast.asArray(Float.self), try oracle.floats("final_hidden_last"))
        #expect(finalError < 0.25)

        // Logits: the choice, the shortlist and the values.
        let ours = logits[0].asArray(Float.self)
        let topIDs = try oracle.ints("top50_ids").map(Int.init)
        let topLogits = try oracle.floats("top50_logits")
        let argmax = ours.indices.max { ours[$0] < ours[$1] }!
        #expect(argmax == topIDs[0])
        let ourTop50 = Set(ours.indices.sorted { ours[$0] > ours[$1] }.prefix(50))
        let overlap = ourTop50.intersection(topIDs).count
        var worst: Float = 0
        for (id, value) in zip(topIDs, topLogits) { worst = max(worst, abs(ours[id] - value)) }
        let margin = topLogits[0] - topLogits[1]
        MuseGlimmerRealArtifact.record(
            "\(name) hidden_rel[\(hiddenErrors.joined(separator: " "))] final_hidden_rel=\(finalError) "
                + "argmax=\(argmax) oracle_argmax=\(topIDs[0]) oracle_margin=\(margin) "
                + "top50_overlap=\(overlap) top50_max_abs_logit_diff=\(worst) "
                + "our_top1_logit=\(ours[argmax])")
        #expect(overlap >= 38)
        #expect(worst < 2.0)
    }

    /// Greedy continuation through the production path: the generic adapter
    /// on the segmented paged backend, prompt in 1,024-token chunks (P3 wraps
    /// the 2,048-token rings). Tokens must equal the oracle's wherever its
    /// top-1 margin is not a near-tie. The oracle's own chunking noise in a
    /// logit is about 1.2, so a margin under 1.5 is a near-tie. Measured: P3
    /// equal for all 16 tokens (smallest margin 6.7); P1 and P2 part at their
    /// smallest-margin steps (0.26 at step 18, 0.15 at step 21).
    @Test(
        .enabled(if: MuseGlimmerRealArtifact.hasOracle), arguments: ["P1", "P2", "P3"])
    func pagedGreedyContinuationEqualsTheOracle(name: String) async throws {
        let model = try MuseGlimmerRealArtifact.loaded.get().installed.model
        let oracle = try MuseGlimmerNPZ(
            url: MuseGlimmerRealArtifact.oracleDirectory!.appendingPathComponent(
                "\(name)_greedy.npz"))
        let prompt = try oracle.ints("input_ids").map(Int.init)
        let expected = try oracle.ints("new_ids").map(Int.init)
        let margins = try oracle.floats("top1_margin")
        let backend = try PagedKVBackend(
            layerKinds: model.cbv2LayerKinds,
            config: .init(
                capacityBytes: 1 << 30, dtype: .bfloat16, maxPrefillChunk: 1024,
                nominalMaxSequenceLength: 8192, segmentSizeBytes: 8 << 20,
                layerDTypes: model.kvStorageDTypes))
        #expect(
            EngineV2.backendCapabilityViolation(
                capabilities: model.cbv2Capabilities, backend: backend) == nil)
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: model.cbv2LayerKinds,
            backend: backend, cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 1024, prefillChunkSize: 1024,
                soloPrefillStripeTokens: 1024, maxWaiting: 2, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0))
        let start = Date()
        let result = await cbv2SchedCollect(
            try engine.submit(
                .init(
                    id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0, seed: 0),
                    maxTokens: expected.count)), timeoutSeconds: 900)
        let seconds = Date().timeIntervalSince(start)
        await engine.shutdown()
        #expect(result.finishReason == .length && result.tokens.count == expected.count)
        let firstMismatch = zip(result.tokens, expected).enumerated().first {
            $0.element.0 != $0.element.1
        }
        MuseGlimmerRealArtifact.record(
            "\(name) paged_greedy prompt=\(prompt.count) new=\(expected.count) seconds=\(String(format: "%.1f", seconds)) "
                + "first_mismatch=\(firstMismatch?.offset ?? -1) "
                + "oracle_margin_at_mismatch=\(firstMismatch.map { margins[$0.offset] } ?? -1) "
                + "min_oracle_margin=\(margins.min() ?? -1) tokens=\(result.tokens)")
        if let firstMismatch {
            // A divergence is acceptable only at an oracle near-tie.
            #expect(margins[firstMismatch.offset] < 1.5)
        }
        #expect(!backend.pool.writeValidation.isFaulted)
    }

    /// Plain decode and prefill speed on the paged backend, one row. A
    /// measurement, not a gate: set `MUSE_GLIMMER_BENCH=1` and hold the GPU lane.
    /// The prompt is the oracle's P3 token ids repeated to `MUSE_GLIMMER_BENCH_PROMPT`
    /// tokens (default 4096), or synthetic ids when no oracle directory is set.
    @Test(
        .enabled(
            if: MuseGlimmerRealArtifact.directory != nil
                && ProcessInfo.processInfo.environment["MUSE_GLIMMER_BENCH"] != nil))
    func pagedDecodeAndPrefillSpeed() async throws {
        let environment = ProcessInfo.processInfo.environment
        let model = try MuseGlimmerRealArtifact.loaded.get().installed.model
        let promptLength = environment["MUSE_GLIMMER_BENCH_PROMPT"].flatMap(Int.init) ?? 4096
        let chunk = environment["MUSE_GLIMMER_BENCH_CHUNK"].flatMap(Int.init) ?? 1024
        let newTokens = environment["MUSE_GLIMMER_BENCH_TOKENS"].flatMap(Int.init) ?? 264
        let warmup = 32
        var source = (0 ..< 4096).map { 1000 + ($0 * 7919) % 150_000 }
        if let directory = MuseGlimmerRealArtifact.oracleDirectory,
            let oracle = try? MuseGlimmerNPZ(url: directory.appendingPathComponent("P3_greedy.npz"))
        {
            source = try oracle.ints("input_ids").map(Int.init)
        }
        // Keep the template head once; repeat the body to the requested length.
        var prompt = source
        while prompt.count < promptLength { prompt += source.dropFirst(4) }
        prompt = Array(prompt.prefix(promptLength - 4)) + source.suffix(4)

        func run(_ id: UInt64, prompt: [Int], maxTokens: Int) async throws
            -> (ttft: Double, stamps: [Double], tokens: [Int])
        {
            let backend = try PagedKVBackend(
                layerKinds: model.cbv2LayerKinds,
                config: .init(
                    capacityBytes: 2 << 30, dtype: .bfloat16, maxPrefillChunk: chunk,
                    nominalMaxSequenceLength: 16384, segmentSizeBytes: 8 << 20,
                    layerDTypes: model.kvStorageDTypes))
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model), layerKinds: model.cbv2LayerKinds,
                backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                    prefillChunkSize: chunk, soloPrefillStripeTokens: chunk, maxWaiting: 2,
                    enablePrefixCache: false),
                admissionConfig: .init(watermarkFraction: 0))
            let start = Date()
            var stamps = [Double]()
            var tokens = [Int]()
            let stream = try engine.submit(
                .init(
                    id: .init(id), promptTokens: prompt,
                    sampling: .init(temperature: 0, seed: 0), maxTokens: maxTokens))
            for await event in stream {
                if case .delta(_, let new, _) = event {
                    let now = Date().timeIntervalSince(start)
                    for token in new {
                        stamps.append(now)
                        tokens.append(token)
                    }
                }
            }
            await engine.shutdown()
            #expect(!backend.pool.writeValidation.isFaulted)
            return (stamps.first ?? -1, stamps, tokens)
        }

        // Short prompt: decode speed after warm-up, twice.
        let short = Array(source.prefix(min(source.count, 256)))
        for attempt in 0 ..< 2 {
            Memory.peakMemory = 0
            let result = try await run(UInt64(10 + attempt), prompt: short, maxTokens: newTokens)
            try #require(result.stamps.count == newTokens && newTokens > warmup + 200)
            let measured = result.stamps.count - 1 - warmup
            let seconds = result.stamps[result.stamps.count - 1] - result.stamps[warmup]
            MuseGlimmerRealArtifact.record(
                "bench decode attempt=\(attempt) prompt=\(short.count) tokens=\(measured) "
                    + "seconds=\(String(format: "%.3f", seconds)) "
                    + "tok_per_s=\(String(format: "%.2f", Double(measured) / seconds)) "
                    + "ttft_s=\(String(format: "%.2f", result.ttft)) "
                    + "mlx_active_gib=\(gib(Memory.activeMemory)) mlx_peak_gib=\(gib(Memory.peakMemory)) "
                    + "footprint_gib=\(gib(MuseGlimmerRealArtifact.footprint())) "
                    + "distinct_tokens=\(Set(result.tokens).count)")
        }
        // Long prompt: prefill speed from submit to the first token, twice.
        for attempt in 0 ..< 2 {
            Memory.peakMemory = 0
            let result = try await run(UInt64(20 + attempt), prompt: prompt, maxTokens: 4)
            try #require(result.ttft > 0)
            MuseGlimmerRealArtifact.record(
                "bench prefill attempt=\(attempt) prompt=\(prompt.count) chunk=\(chunk) "
                    + "ttft_s=\(String(format: "%.2f", result.ttft)) "
                    + "tok_per_s=\(String(format: "%.1f", Double(prompt.count) / result.ttft)) "
                    + "mlx_active_gib=\(gib(Memory.activeMemory)) mlx_peak_gib=\(gib(Memory.peakMemory)) "
                    + "footprint_gib=\(gib(MuseGlimmerRealArtifact.footprint()))")
        }
    }
}
