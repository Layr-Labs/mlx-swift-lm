#if canImport(Metal)
import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Ordinary prefill and one-token continuation only. No sampler, drafter, HTTP or persistent KV cache.
@Suite(
    "Gemma 4 row-mapped NAX full-model native state", .tags(.integration), .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["GEMMA4_ROW_MAP_STATE_CHECKPOINT"] != nil))
struct Gemma4RowMapCheckpointStateTests {
    private static let environment = ProcessInfo.processInfo.environment

    private struct Case {
        let prefix: Int
        let batch: Int
        var name: String { "p\(prefix)-b\(batch)" }
    }

    private static let cases: [Case] = [
        Case(prefix: 64, batch: 1),
        Case(prefix: 2048, batch: 1),
        Case(prefix: 2049, batch: 1),
        Case(prefix: 64, batch: 4),
    ]

    private final class Report {
        let directory: URL
        let reference: URL?
        let expected: [String: [String: Any]]
        var metadata: [String: Any]
        var entries: [String: [String: Any]] = [:]
        var routes: [[String: Any]] = []
        var rowMapCalls: [[String: Any]] = []
        var failed = 0
        var completedCases = 0
        private var firstFailures: Set<String> = []
        private var storedBytes = 0

        static func hash(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }

        static func json(_ value: Any) throws -> Data {
            try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        }

        init(directory: URL, reference: URL?, metadata: [String: Any]) throws {
            self.directory = directory
            self.reference = reference
            self.metadata = metadata
            if let reference {
                let data = try Data(contentsOf: reference.appendingPathComponent("report.json"))
                let root = try #require(
                    try JSONSerialization.jsonObject(with: data) as? [String: Any])
                try #require(root["completed"] as? Bool == true)
                try #require(root["failed_checks"] as? Int == 0)
                try #require(root["feature_flag"] as? String == "0")
                try #require(root["capture_mode"] as? Bool == true)
                try #require(root["completed_cases"] as? Int == Self.expectedCaseCount)
                let oldSignature = try #require(root["signature"] as? [String: Any])
                let newSignature = try #require(metadata["signature"] as? [String: Any])
                let oldSignatureBytes = try Self.json(oldSignature)
                let newSignatureBytes = try Self.json(newSignature)
                try #require(oldSignatureBytes == newSignatureBytes)
                expected = try #require(root["entries"] as? [String: [String: Any]])
            } else {
                expected = [:]
            }
            try #require(!FileManager.default.fileExists(atPath: directory.path))
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700])
        }

        private static let expectedCaseCount = 4

        func record(_ data: Data, shape: [Int], dtype: String, kind: String, label: String) throws {
            try #require(entries[label] == nil, "Duplicate oracle label")
            let digest = Self.hash(data)
            var entry: [String: Any] = [
                "shape": shape, "dtype": dtype, "kind": kind,
                "bytes": data.count, "sha256": digest,
            ]
            if let reference {
                let old = try #require(expected[label], "Missing baseline tensor \(label)")
                let filename = try #require(old["file"] as? String)
                try #require(filename.count == 68 && filename.hasSuffix(".bin"))
                try #require(filename.dropLast(4).allSatisfy { "0123456789abcdef".contains($0) })
                let original = try Data(
                    contentsOf: reference.appendingPathComponent(filename), options: .mappedIfSafe)
                try #require(Self.hash(original) == old["sha256"] as? String)
                let same =
                    old["shape"] as? [Int] == shape && old["dtype"] as? String == dtype
                    && old["kind"] as? String == kind && old["bytes"] as? Int == data.count
                    && original == data
                entry["equal"] = same
                if !same {
                    failed += 1
                    entry["first_differing_byte"] =
                        zip(original, data).enumerated()
                        .first(where: { $0.element.0 != $0.element.1 })?.offset
                        ?? min(original.count, data.count)
                    if firstFailures.insert(kind).inserted {
                        let name = "first-" + kind + "-candidate.bin"
                        try data.write(to: directory.appendingPathComponent(name), options: .atomic)
                        entry["first_candidate_file"] = name
                        entry["baseline_file"] = filename
                    }
                }
                #expect(same, "\(label): native bytes or metadata differ")
            } else {
                let filename = digest + ".bin"
                let path = directory.appendingPathComponent(filename)
                if !FileManager.default.fileExists(atPath: path.path) {
                    // Bound diagnostic evidence; this never changes inference/cache policy.
                    try #require(storedBytes + data.count <= 3 * 1024 * 1024 * 1024)
                    try data.write(to: path, options: .atomic)
                    try FileManager.default.setAttributes(
                        [.posixPermissions: 0o600], ofItemAtPath: path.path)
                    storedBytes += data.count
                }
                entry["file"] = filename
            }
            entries[label] = entry
        }

        func routeCalls(label: String, phase: String, tokens: Int, calls: Int) throws {
            let expectedOn =
                metadata["feature_flag"] as? String == "1" && phase == "prefill" && tokens >= 64
            rowMapCalls.append([
                "label": label, "phase": phase, "tokens": tokens, "calls": calls,
                "scope":
                    "Constructed row-mapped gate/up graph pairs, followed by completed native evaluation",
            ])
            try #require(expectedOn ? (calls > 0 && calls <= 30) : calls == 0)
        }

        func graph(_ outputs: [MLXArray], label: String, phase: String, tokens: Int) throws {
            let description = graphDescription(outputs)
            let data = Data(description.utf8)
            let name = Self.hash(data) + ".graph.txt"
            try #require(data.count <= 8 * 1024 * 1024)
            try data.write(to: directory.appendingPathComponent(name), options: .atomic)
            let lines = description.split(separator: "\n")
            routes.append([
                "label": label, "phase": phase, "tokens": tokens, "file": name,
                "gather_qmm_nodes": lines.filter { $0.hasPrefix("GatherQMM ") }.count,
                "custom_kernel_nodes": lines.filter { $0.hasPrefix("CustomKernel ") }.count,
                "scope":
                    "Remaining graph at observation; internal async eval may prune nodes; native output/state checks establish parity",
            ])
        }

        func tensor(_ array: MLXArray, kind: String, label: String) throws {
            try record(
                array.asData(access: .copy).data, shape: array.shape,
                dtype: String(describing: array.dtype), kind: kind, label: label)
        }

        func frontier(_ value: [String: Any], label: String) throws {
            try record(Self.json(value), shape: [], dtype: "json", kind: "frontier", label: label)
        }

        func flush(completed: Bool = false) throws {
            metadata["completed"] = completed
            metadata["completed_cases"] = completedCases
            metadata["failed_checks"] = failed
            metadata["entries"] = entries
            metadata["graph_routes"] = routes
            metadata["row_map_calls"] = rowMapCalls
            metadata["stored_raw_bytes"] = storedBytes
            try JSONSerialization.data(
                withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]
            )
            .write(to: directory.appendingPathComponent("report.json"), options: .atomic)
        }

        func finish() throws {
            try #require(completedCases == Self.expectedCaseCount)
            try #require(entries.count == 1915 && routes.count == 16 && rowMapCalls.count == 16)
            if reference != nil { try #require(Set(entries.keys) == Set(expected.keys)) }
            try #require(failed == 0)
            try flush(completed: true)
        }
    }

    private final class Runner {
        let text: Gemma4TextModel
        let adapter: CBv2SteppableLanguageModelAdapter
        let backend: PagedKVBackend
        let bank: CBv2LayerCacheBank
        let caches: [PagedLayerCache]

        init(text: Gemma4TextModel, types: [DType]) throws {
            self.text = text
            adapter = CBv2SteppableLanguageModelAdapter(text)
            backend = try PagedKVBackend(
                layerKinds: text.cbv2LayerKinds,
                config: .init(
                    capacityBytes: 1 << 30, dtype: types[0], maxPrefillChunk: 2048,
                    nominalMaxSequenceLength: 2064, segmentSizeBytes: 64 << 20,
                    layerDTypes: types))
            caches = backend.makeLayerCaches()
            bank = CBv2LayerCacheBank(caches: caches)
        }

        func withPrefilled(_ item: Case, report: Report, body: ([[CBv2SequenceKV?]]) throws -> Void)
            throws
        {
            var rows: [[CBv2SequenceKV?]] = []
            defer {
                bank.releaseBoundRows()
                for row in rows { backend.release(row) }
            }
            for rowIndex in 0 ..< item.batch {
                let row = try backend.makeSequenceState(
                    layerKinds: text.cbv2LayerKinds, promptLength: item.prefix,
                    maxLength: item.prefix + 16)
                rows.append(row)
                let prompt = (0 ..< item.prefix).map {
                    Int32(3 + ($0 * 37 + 11 + rowIndex * 53) % 1000)
                }
                for start in stride(from: 0, to: item.prefix, by: 2048) {
                    let end = min(item.prefix, start + 2048)
                    let bound = bank.layerCaches(rowStates: [row])
                    CBv2StepProfiler.armEvents()
                    let output = adapter.prefill(
                        tokens: MLXArray(Array(prompt[start ..< end])).reshaped([1, end - start]),
                        inputEmbeddings: nil, caches: bound,
                        requirement: end == item.prefix ? .lastPositionLogits : .evaluationOnly)
                    let outputs = [output] + caches.flatMap { $0.innerState() }
                    try report.graph(
                        outputs, label: item.name + ".row\(rowIndex).chunk\(start)",
                        phase: "prefill", tokens: end - start)
                    eval(outputs)
                    let calls = CBv2StepProfiler.snapshotAndDisarmEvents()[
                        "v2.gemma4.nax.row_map", default: 0]
                    try report.routeCalls(
                        label: item.name + ".row\(rowIndex).chunk\(start)",
                        phase: "prefill", tokens: end - start, calls: calls)
                    try report.tensor(
                        output, kind: "prefill",
                        label:
                            item.name + ".row\(rowIndex).chunk\(start).prefill")
                }
            }
            try body(rows)
        }

        func snapshot(
            _ rows: [[CBv2SequenceKV?]], expectedOffset: Int, label: String, report: Report
        ) throws {
            for (rowIndex, layers) in rows.enumerated() {
                try #require(layers.count == 30)
                for (layerIndex, item) in layers.enumerated() {
                    let row = try #require(item as? PagedSequenceKV)
                    try #require(row.absoluteOffset == expectedOffset)
                    let state = row.snapshot()
                    eval(state.keys, state.values)
                    try #require(state.offset == row.absoluteOffset)
                    try #require(
                        state.keys.dim(2) == row.retainedCount
                            && state.values.dim(2) == row.retainedCount)
                    let key = label + ".row\(rowIndex).layer\(layerIndex)"
                    try report.frontier(
                        [
                            "offset": state.offset, "retained_count": row.retainedCount,
                            "retained_start": state.offset - row.retainedCount,
                            "window": row.windowSize.map { $0 as Any } ?? NSNull(),
                        ], label: key + ".frontier")
                    try report.tensor(state.keys, kind: "keys", label: key + ".keys")
                    try report.tensor(state.values, kind: "values", label: key + ".values")
                }
            }
        }

        func forward(
            _ tokens: [Int32], rows: [[CBv2SequenceKV?]], label: String, report: Report
        ) throws {
            let batch = tokens.count
            try #require(rows.count == batch)
            let bound = bank.layerCaches(rowStates: rows)
            CBv2StepProfiler.armEvents()
            let work = {
                let output = self.adapter.forwardWithHidden(
                    tokens: MLXArray(tokens).reshaped([batch, 1]), caches: bound)
                let outputs =
                    [output.logits, output.lastHidden] + self.caches.flatMap { $0.innerState() }
                try report.graph(outputs, label: label, phase: "decode", tokens: 1)
                eval(outputs)
                let calls = CBv2StepProfiler.snapshotAndDisarmEvents()[
                    "v2.gemma4.nax.row_map", default: 0]
                try report.routeCalls(label: label, phase: "decode", tokens: 1, calls: calls)
                return output
            }
            let output =
                try adapter.decodeCommandBufferLimits.map {
                    try GPU.withCommandBufferLimits($0, work)
                } ?? work()
            try #require(output.logits.shape == [batch, 1, 262144])
            try #require(output.lastHidden.shape == [batch, 1, 2816])
            try #require(MLX.all(isFinite(output.logits)).item(Bool.self))
            try #require(MLX.all(isFinite(output.lastHidden)).item(Bool.self))
            try report.tensor(output.logits, kind: "logits", label: label + ".logits")
            try report.tensor(output.lastHidden, kind: "hidden", label: label + ".hidden")
        }

        func run(_ item: Case, report: Report) throws {
            let before = report.failed
            try withPrefilled(item, report: report) { rows in
                try snapshot(
                    rows, expectedOffset: item.prefix,
                    label: item.name + ".prefilled", report: report)
                for step in 0 ..< 2 {
                    let tokens = (0 ..< item.batch).map { Int32(17 + $0 * 7 + step * 80) }
                    try forward(
                        tokens, rows: rows,
                        label: item.name + ".decode\(step)", report: report)
                    try snapshot(
                        rows, expectedOffset: item.prefix + step + 1,
                        label: item.name + ".decode\(step).state", report: report)
                }
            }
            try #require(backend.bytesReserved == 0)
            report.completedCases += 1
            try report.flush()
            try #require(
                report.failed == before, "Native state mismatch; preserve failure and stop")
        }
    }

    @Test func ordinaryPrefillAndContinuationStateOracle() throws {
        let env = Self.environment
        defer { _ = CBv2StepProfiler.snapshotAndDisarmEvents() }
        let feature = try #require(env["DARKBLOOM_GEMMA4_NAX_ROW_MAP"])
        try #require(feature == "0" || feature == "1")
        try #require(env["MLX_COMPILED_DECODE"] == "1" && env["MLX_ENABLE_TF32"] == "1")
        try #require(env["DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD"] == "1")
        try #require(env["DARKBLOOM_BF16_WEIGHTS"] == nil || env["DARKBLOOM_BF16_WEIGHTS"] == "1")
        let mode = try #require(env["GEMMA4_ROW_MAP_STATE_MODE"])
        try #require(mode == "capture" || mode == "compare")
        try #require(mode != "capture" || feature == "0")
        let directory = URL(
            fileURLWithPath: try #require(env["GEMMA4_ROW_MAP_STATE_CHECKPOINT"]))
        let output = URL(fileURLWithPath: try #require(env["GEMMA4_ROW_MAP_STATE_OUTPUT"]))
        let reference: URL?
        if mode == "compare" {
            reference = URL(fileURLWithPath: try #require(env["GEMMA4_ROW_MAP_STATE_REFERENCE"]))
        } else {
            reference = nil
        }
        try #require(
            !output.resolvingSymlinksInPath().path.hasPrefix(
                directory.resolvingSymlinksInPath().path + "/"))
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let indexData = try Data(
            contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        try #require(
            Report.hash(configData)
                == "29910322dd085f45c8f95c6c0f1611b20f722d6f6c8394321b34817e98a972fa")
        try #require(
            Report.hash(indexData)
                == "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80")
        let provenanceURL = URL(
            fileURLWithPath: try #require(env["GEMMA4_ROW_MAP_STATE_PROVENANCE"]))
        let provenance = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: provenanceURL))
                as? [String: Any])
        for key in [
            "sdk_commit", "sdk_diff_sha256", "swift_commit", "core_commit",
            "test_binary_sha256", "test_source_sha256", "metallib_sha256", "weight_manifest_sha256",
        ] {
            try #require(
                provenance[key] as? String != nil, "Missing immutable build/model provenance")
        }
        var signature = provenance
        signature["config_sha256"] = Report.hash(configData)
        signature["index_sha256"] = Report.hash(indexData)
        signature["oracle_version"] = 1
        let report = try Report(
            directory: output, reference: reference,
            metadata: [
                "signature": signature, "feature_flag": feature, "capture_mode": mode == "capture",
                "scope":
                    "Ordinary prefill and one-token continuation OFF/ON; native prefill outputs, logits, hidden and retained paged KV; no sampler/drafter/throughput claim",
            ])
        var finished = false
        defer {
            if !finished {
                do { try report.flush() } catch {
                    Issue.record("Could not preserve partial oracle: \(error)")
                }
            }
        }
        try report.flush()
        let config = try JSONDecoder().decode(MLXVLM.Gemma4Configuration.self, from: configData)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let model = MLXVLM.Gemma4(config)
        try loadWeights(
            modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization
        )
        model.train(false)
        let text = model.textModel
        let adapter = CBv2SteppableLanguageModelAdapter(text)
        let kinds = text.cbv2LayerKinds
        try #require(kinds.count == 30 && kinds.allSatisfy { $0.sharesKVWithLayer == nil })
        let probe = try CBv2NativeKVTypeProbe.run(
            model: adapter, layerKinds: kinds,
            caches: try text.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        try #require(probe.layerDTypes.count == 30)
        let runner = try Runner(text: text, types: probe.layerDTypes)
        try #require(64.isMultiple(of: runner.backend.pool.config.pageSize))
        try report.frontier(
            [
                "native_kv_dtypes": probe.layerDTypes.map { String(describing: $0) },
                "page_size": runner.backend.pool.config.pageSize,
            ], label: "model.native-state-contract")
        for item in Self.cases { try runner.run(item, report: report) }
        try report.finish()
        finished = true
        print(
            "GEMMA4_ROW_MAP_STATE cases=\(report.completedCases) entries=\(report.entries.count) failures=\(report.failed) completed=true"
        )
    }
}
#endif
