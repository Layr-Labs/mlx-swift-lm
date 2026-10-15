// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Explicit offline real-checkpoint concurrency benchmark of the provider's
/// default GLM serving shape: `GLM5NextExecutionPolicy.serving`, EngineV2 on
/// the native paged backend (`makeCBv2PagedBackend`, provider step/chunk
/// bounds), a complete-prefix store and the default sampler, MTP off. Runs
/// only when `GLM53_CONC_BENCH=1` and `GLM53_REAL_PROFILE_DIR` names the
/// prepared directory with `identity.json` and `prompts.json`; downloads
/// nothing and asserts no performance threshold.
///
/// Every concurrent greedy stream is compared with the same prompt run alone
/// on an engine of the same configuration (`GLM53_CONC_VERIFY=0` skips).
///
/// Environment (all optional):
/// - `GLM53_CONC_N` request counts (default 1,2,4,8,16)
/// - `GLM53_CONC_PROMPTS` prompt lengths (default 32,2500)
/// - `GLM53_CONC_OUTPUT` output tokens per request (default 128)
/// - `GLM53_CONC_PARTIAL` partial-prefill caps; 0 = unlimited (default 0)
/// - `GLM53_CONC_CONTEXT` engine context bound (default 4096)
/// - `GLM53_CONC_KV_GB` paged KV capacity, which also bounds step scratch
///   (default 72: the worst-case step price at 16 rows and 1024 step tokens)
/// - `GLM53_CONC_STEP` / `GLM53_CONC_CHUNK` step and prefill-chunk tokens
///   (default 1024 / 512, the provider's GLM bounds)
/// - `GLM53_CONC_CAPTURE=1` lets the matrix store accept checkpoints
/// - `GLM53_CONC_PREFIX` prefix-hit wave sizes (default 8; empty skips)
/// - `GLM53_CONC_PREFIX_TOKENS` shared system-prompt length (default 2048)
/// - `GLM53_CONC_STRICT=1` fails on any concurrent/solo greedy divergence
extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextRealConcurrencyTests {
        private struct Identity: Decodable {
            struct File: Decodable {
                let bytes: Int
                let sha256: String
            }
            let repository: String
            let revision: String
            let files: [String: File]
        }

        private struct Prompt: Decodable {
            let name: String
            let ids: [Int32]
        }

        private static var environment: [String: String] { ProcessInfo.processInfo.environment }
        private static var profileDirectory: URL? {
            guard environment["GLM53_CONC_BENCH"] == "1" else { return nil }
            return environment["GLM53_REAL_PROFILE_DIR"].map {
                URL(fileURLWithPath: $0, isDirectory: true)
            }
        }

        private static func list(_ name: String, _ fallback: [Int]) -> [Int] {
            environment[name].map { $0.split(separator: ",").compactMap { Int($0) } } ?? fallback
        }

        private static func integer(_ name: String, _ fallback: Int) -> Int {
            environment[name].flatMap(Int.init) ?? fallback
        }

        private static func seconds(_ from: ContinuousClock.Instant, _ to: ContinuousClock.Instant)
            -> Double
        {
            let d = from.duration(to: to)
            return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
        }

        private static func median(_ values: [Double]) -> Double {
            values.isEmpty ? .nan : values.sorted()[values.count / 2]
        }

        /// The provider's engine budget is the machine's process ledger; this
        /// bench bounds it at 192 GiB with exact coverage accounting.
        private final class ProcessBudget: CBv2ProcessMemoryOwner, @unchecked Sendable {
            enum Failure: Error { case exhausted, invalidCoverage }
            private let lock = NSLock()
            private var charge: UInt64 = 0
            private var coverage: UInt64 = 0
            func replaceCharge(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard bytes <= (192 << 30) else { throw Failure.exhausted }
                    guard bytes >= coverage else { throw Failure.invalidCoverage }
                    charge = bytes
                }
            }
            func recordMaterialization(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard bytes >= coverage && bytes <= charge else {
                        throw Failure.invalidCoverage
                    }
                    coverage = bytes
                }
            }
            func withdrawCoverage(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard bytes <= coverage else { throw Failure.invalidCoverage }
                    coverage -= bytes
                }
            }
            func retire() {}
        }

        private struct Collected {
            var tokens: [Int] = []
            var stamps: [(time: ContinuousClock.Instant, count: Int)] = []
            var usage: CBv2Usage?
            var first: ContinuousClock.Instant? { stamps.first?.time }
            var last: ContinuousClock.Instant? { stamps.last?.time }
        }

        private struct Configuration {
            let rows: Int
            let context: Int
            let kvBytes: Int
            let partialPrefills: Int?
            let stepTokens: Int
            let chunkTokens: Int
        }

        private static func makeEngine(
            _ model: GLM5NextTextModel, _ configuration: Configuration,
            store: CompleteCheckpointFixtureStore
        ) throws -> EngineV2 {
            let maximumBuffer = MLX.GPU.deviceInfo().maxBufferSize
            // The provider's GLM bounds (EngineV2Factory.glm5NextServingScheduler).
            let chunk = min(configuration.context, configuration.chunkTokens)
            let stepTokens = min(configuration.context, configuration.stepTokens)
            let backend = try model.makeCBv2PagedBackend(
                kvBytesCapacity: configuration.kvBytes, maximumBatchSize: configuration.rows,
                maximumQueryTokens: max(stepTokens, chunk),
                maximumContextTokens: configuration.context,
                maximumScratchBytes: min(configuration.kvBytes, maximumBuffer),
                segmentBytes: 64 << 20, maximumBufferBytes: maximumBuffer)
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds, backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: configuration.rows,
                    maxBatchedTokensPerStep: stepTokens, prefillChunkSize: chunk,
                    maxConcurrentPartialPrefills: configuration.partialPrefills,
                    maxWaiting: 4 * configuration.rows, enablePrefixCache: true),
                admissionConfig: model.cbv2NativeAdmissionConfig, completePrefixCache: store,
                processMemoryOwner: ProcessBudget())
            if let reason = engine.pagedAttentionWorkInactiveReason {
                print("GLM53_CONC engine_refused reason=\(reason)")
                throw CBv2KVError.backendIneligible(reason: reason)
            }
            return engine
        }

        /// Submits every request at once and collects each stream's tokens
        /// with arrival times.
        private static func run(_ engine: EngineV2, _ requests: [CBv2Request]) async throws
            -> (start: ContinuousClock.Instant, streams: [Collected])
        {
            let start = ContinuousClock.now
            let sources = try requests.map { try engine.submit($0) }
            let streams = await withTaskGroup(of: (Int, Collected).self) { group in
                for (index, source) in sources.enumerated() {
                    group.addTask {
                        var stream = Collected()
                        for await event in source {
                            switch event {
                            case .delta(_, let tokens, _):
                                guard !tokens.isEmpty else { continue }
                                stream.tokens += tokens
                                stream.stamps.append((ContinuousClock.now, tokens.count))
                            case .finished(_, let usage):
                                stream.usage = usage
                            }
                        }
                        return (index, stream)
                    }
                }
                var result = [Collected](repeating: Collected(), count: sources.count)
                for await (index, stream) in group { result[index] = stream }
                return result
            }
            return (start, streams)
        }

        private static func phase(_ snapshot: [String: [Double]], _ name: String) -> String {
            guard let values = snapshot[name], !values.isEmpty else { return "n/a" }
            return String(format: "%.2f", median(values) * 1000)
        }

        /// One line of aggregate, per-stream, TTFT and step-split numbers.
        private static func report(
            _ label: String, start: ContinuousClock.Instant, streams: [Collected],
            profile: [String: [Double]]
        ) {
            let firsts = streams.compactMap(\.first)
            let lasts = streams.compactMap(\.last)
            guard firsts.count == streams.count, let end = lasts.max() else {
                print("GLM53_CONC \(label) incomplete")
                return
            }
            let total = streams.reduce(0) { $0 + $1.tokens.count }
            let ttft = firsts.map { seconds(start, $0) * 1000 }
            // The window in which every stream is decoding.
            let open = firsts.max()!
            let close = lasts.min()!
            var windowed = 0
            for stream in streams {
                for stamp in stream.stamps where stamp.time > open && stamp.time <= close {
                    windowed += stamp.count
                }
            }
            let window = seconds(open, close)
            let aggregate = window > 0.05 ? String(format: "%.1f", Double(windowed) / window) : "n/a"
            let perStream = streams.map { stream -> Double in
                let span = seconds(stream.first!, stream.last!)
                return span > 0 ? Double(stream.tokens.count - 1) / span : .nan
            }
            let saved = streams.reduce(0) { $0 + ($1.usage?.prefixCachePrefillTokensSaved ?? 0) }
            print(
                String(
                    format:
                        "GLM53_CONC %@ e2e_tok_s=%.1f decode_agg_tok_s=%@ per_stream_tok_s=%.1f "
                        + "ttft_ms mean=%.0f p50=%.0f max=%.0f makespan_s=%.2f "
                        + "step_ms wall=%@ launch=%@ build=%@ sampler=%@ submit=%@ readback_wait=%@ "
                        + "prefix_saved=%d",
                    label, Double(total) / seconds(start, end), aggregate,
                    perStream.reduce(0, +) / Double(perStream.count),
                    ttft.reduce(0, +) / Double(ttft.count), median(ttft), ttft.max()!,
                    seconds(start, end),
                    phase(profile, "v2.step.wall"), phase(profile, "v2.launch.total"),
                    phase(profile, "v2.forward.build"), phase(profile, "v2.sampler.build"),
                    phase(profile, "v2.asyncEval.submit"),
                    phase(profile, "v2.readback.wait"), saved))
        }

        /// Prompt `index` of `length` tokens: distinct leading tokens and a
        /// distinct rotation of the real prompt corpus.
        private static func prompt(_ corpus: [Int], length: Int, index: Int, salt: Int) -> [Int] {
            let head = [corpus[(index * 13 + salt) % corpus.count], 4 + (index * 31 + salt) % 9000]
            let body = (0 ..< max(0, length - head.count)).map {
                corpus[($0 + index * 7 + salt * 3) % corpus.count]
            }
            return Array((head + body).prefix(length))
        }

        private static func compare(
            _ label: String, _ streams: [Collected], _ expected: [[Int]]
        ) -> Int {
            var mismatches = 0
            for (index, stream) in streams.enumerated() {
                let solo = expected[index]
                guard stream.tokens != solo else { continue }
                mismatches += 1
                let divergence =
                    zip(stream.tokens, solo).enumerated().first { $0.element.0 != $0.element.1 }?
                    .offset ?? min(stream.tokens.count, solo.count)
                print(
                    "GLM53_CONC_MISMATCH \(label) row=\(index) first_divergence=\(divergence) "
                        + "got=\(stream.tokens.count) solo=\(solo.count)")
            }
            return mismatches
        }

        @Test(.enabled(if: profileDirectory != nil))
        func realConcurrentServingThroughput() async throws {
            let root = try #require(Self.profileDirectory)
            let modelURL = URL(
                fileURLWithPath: try #require(Self.environment["GLM53_REAL_MODEL_DIR"]),
                isDirectory: true)
            let identityFile = try JSONDecoder().decode(
                Identity.self, from: Data(contentsOf: root.appendingPathComponent("identity.json")))
            let prompts = try JSONDecoder().decode(
                [Prompt].self, from: Data(contentsOf: root.appendingPathComponent("prompts.json")))
            let corpus = prompts.flatMap(\.ids).map(Int.init)
            let identity = GLM5NextArtifactIdentity(
                repository: identityFile.repository, revision: identityFile.revision,
                files: identityFile.files.mapValues { .init(bytes: $0.bytes, sha256: $0.sha256) })
            let counts = Self.list("GLM53_CONC_N", [1, 2, 4, 8, 16])
            let lengths = Self.list("GLM53_CONC_PROMPTS", [32, 2500])
            let output = Self.integer("GLM53_CONC_OUTPUT", 128)
            let partials = Self.list("GLM53_CONC_PARTIAL", [0])
            let context = Self.integer("GLM53_CONC_CONTEXT", 4096)
            let kvBytes = Self.integer("GLM53_CONC_KV_GB", 72) << 30
            let stepTokens = Self.integer("GLM53_CONC_STEP", 1024)
            let chunkTokens = Self.integer("GLM53_CONC_CHUNK", 512)
            let verify = Self.environment["GLM53_CONC_VERIFY"] != "0"
            let capture = Self.environment["GLM53_CONC_CAPTURE"] == "1"
            let prefixWaves = Self.list("GLM53_CONC_PREFIX", [8])
            let prefixTokens = Self.integer("GLM53_CONC_PREFIX_TOKENS", 2048)
            let rows = max(counts.max() ?? 1, prefixWaves.max() ?? 1)
            print(
                "GLM53_CONC policy=serving rows=\(rows) counts=\(counts) prompts=\(lengths) "
                    + "output=\(output) partial=\(partials) context=\(context) "
                    + "kv_gb=\(kvBytes >> 30) step=\(stepTokens) chunk=\(chunkTokens) capture=\(capture)")

            let loadStart = ContinuousClock.now
            let roots = GLM5NextArtifactLoadRoots()
            let artifact = try GLM5NextArtifactLoader(
                directory: modelURL, expectedIdentity: identity,
                storage: GLM5NextWeightInventory.packedStorage(
                    declaredBy: JSONDecoder().decode(
                        GLM5NextConfiguration.self,
                        from: Data(contentsOf: modelURL.appendingPathComponent("config.json")))),
                constructionRoots: roots, executionPolicy: .serving)
            let glm = try GLM5NextModel(artifact: artifact, loadMTP: false)
            try roots.evaluate(glm.parameters().flattened().map(\.1))
            let model = glm.textTower
            print(
                "GLM53_CONC load_s=\(Self.seconds(loadStart, .now)) "
                    + "active_gb=\(Double(Memory.activeMemory) / 1e9)")

            let profiling = CBv2StepProfiler.enabled
            CBv2StepProfiler.enabled = true
            defer { CBv2StepProfiler.enabled = profiling }
            var nextID: UInt64 = 1
            func request(_ tokens: [Int], target: Int? = nil, receipt: Bool = false) -> CBv2Request
            {
                defer { nextID += 1 }
                return CBv2Request(
                    id: .init(nextID), promptTokens: tokens, sampling: .init(temperature: 0),
                    maxTokens: output, cacheSalt: "glm53-concurrency",
                    prefixCacheReceiptID: receipt ? .init(1_000_000 + nextID) : nil,
                    prefixCheckpointTargetTokens: target)
            }
            func configuration(partial: Int) -> Configuration {
                .init(
                    rows: rows, context: context, kvBytes: kvBytes,
                    partialPrefills: partial > 0 ? partial : nil, stepTokens: stepTokens,
                    chunkTokens: chunkTokens)
            }
            func matrixStore() -> CompleteCheckpointFixtureStore {
                CompleteCheckpointFixtureStore(
                    maximumPosition: capture ? .max : 0, segmentBytes: 64 << 20)
            }

            var mismatches = 0
            for length in lengths {
                let maximum = counts.max() ?? 1
                let promptSet = (0 ..< maximum).map {
                    Self.prompt(corpus, length: length, index: $0, salt: length)
                }
                // Solo references: each prompt alone, sequentially, on one engine.
                var solo: [[Int]] = []
                if verify {
                    let engine = try Self.makeEngine(
                        model, configuration(partial: partials.first ?? 0), store: matrixStore())
                    for (index, tokens) in promptSet.enumerated() {
                        CBv2StepProfiler.reset()
                        let (start, streams) = try await Self.run(engine, [request(tokens)])
                        solo.append(streams[0].tokens)
                        if index == 0 {
                            Self.report(
                                "solo prompt=\(length)", start: start, streams: streams,
                                profile: CBv2StepProfiler.snapshot())
                        }
                    }
                    await engine.shutdown()
                }
                for partial in partials {
                    for count in counts {
                        let engine = try Self.makeEngine(
                            model, configuration(partial: partial), store: matrixStore())
                        CBv2StepProfiler.reset()
                        let (start, streams) = try await Self.run(
                            engine, promptSet.prefix(count).map { request($0) })
                        let label = "n=\(count) prompt=\(length) partial=\(partial)"
                        Self.report(
                            label, start: start, streams: streams,
                            profile: CBv2StepProfiler.snapshot())
                        await engine.shutdown()
                        #expect(streams.allSatisfy { $0.tokens.count == output })
                        if verify {
                            mismatches += Self.compare(label, streams, Array(solo.prefix(count)))
                        }
                    }
                }
            }

            // Prefix-hit wave: one producer captures the shared system prompt,
            // then `count` requests sharing it are staged and run together,
            // against the same prompts run cold.
            for count in prefixWaves where count > 0 {
                let system = Self.prompt(corpus, length: prefixTokens, index: 97, salt: 5)
                let suffix = { (index: Int) in
                    Self.prompt(corpus, length: 64, index: index + 200, salt: 11)
                }
                let store = CompleteCheckpointFixtureStore(segmentBytes: 64 << 20)
                let engine = try Self.makeEngine(
                    model, configuration(partial: partials.first ?? 0), store: store)
                _ = try await Self.run(
                    engine, [request(system + suffix(-1), target: system.count, receipt: true)])
                store.finishPublicationCallbacks(engine: engine)
                let positions = store.saved.map { $0.manifest.position }
                print("GLM53_CONC prefix_capture positions=\(positions)")
                let wave = (0 ..< count).map { system + suffix($0) }
                let warm = wave.map { request($0, receipt: true) }
                var staged = 0
                for item in warm {
                    if try store.stage(engine: engine, request: item) { staged += 1 }
                }
                CBv2StepProfiler.reset()
                let hit = try await Self.run(engine, warm)
                Self.report(
                    "prefix_hit n=\(count) shared=\(system.count) staged=\(staged)",
                    start: hit.start, streams: hit.streams, profile: CBv2StepProfiler.snapshot())
                await engine.shutdown()
                let coldEngine = try Self.makeEngine(
                    model, configuration(partial: partials.first ?? 0), store: matrixStore())
                CBv2StepProfiler.reset()
                let cold = try await Self.run(coldEngine, wave.map { request($0) })
                Self.report(
                    "prefix_cold n=\(count) shared=\(system.count)", start: cold.start,
                    streams: cold.streams, profile: CBv2StepProfiler.snapshot())
                await coldEngine.shutdown()
                if verify {
                    mismatches += Self.compare(
                        "prefix n=\(count)", hit.streams, cold.streams.map { $0.tokens })
                }
            }
            print("GLM53_CONC mismatches=\(mismatches)")
            // Non-attention kernels (MoE, projections) pick batch-dependent
            // reduction orders, so greedy streams can diverge from solo runs at
            // near ties; `GLM53_CONC_STRICT=1` makes divergence a failure.
            if Self.environment["GLM53_CONC_STRICT"] == "1" { #expect(mismatches == 0) }
        }
    }
}
