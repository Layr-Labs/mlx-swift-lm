// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Explicit offline native-engine MTP throughput probe. It runs only when
/// `GLM53_REAL_MTP_PROBE=1` and `GLM53_REAL_SMOKE_DIR`/`GLM53_REAL_MODEL_DIR`
/// name a prepared artifact with its trained MTP layer; it downloads nothing.
/// Every mode decodes the smoke prompts through EngineV2's native paged path
/// and is compared with the target-only paged engine under the same policy.
extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextRealMTPTests {
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

        /// `0` is the target-only engine; `serial` the depth-one serial
        /// profile; `1`...`3` fixed rectangular depths; `adaptive` the
        /// rectangular profile's marginal depth policy up to depth 3.
        private enum Mode: Equatable, CustomStringConvertible {
            case targetOnly
            case serial
            case rectangular(Int)
            case adaptive

            init?(_ name: String) {
                switch name {
                case "0": self = .targetOnly
                case "serial": self = .serial
                case "adaptive": self = .adaptive
                default:
                    guard let depth = Int(name), (1 ... 3).contains(depth) else { return nil }
                    self = .rectangular(depth)
                }
            }

            var description: String {
                switch self {
                case .targetOnly: "target_only"
                case .serial: "serial_d1"
                case .rectangular(let depth): "rect_d\(depth)"
                case .adaptive: "rect_adaptive"
                }
            }
        }

        private struct Run {
            var tokens: [Int] = []
            var decodeSeconds = 0.0
            var decodeSteps = 0
            var chainedSteps = 0
        }

        private static var environment: [String: String] { ProcessInfo.processInfo.environment }

        private static var enabled: Bool {
            environment["GLM53_REAL_MTP_PROBE"] == "1" && environment["GLM53_REAL_SMOKE_DIR"] != nil
        }

        private static func policy(named name: String) -> GLM5NextExecutionPolicy {
            switch name {
            case "serving": return .serving
            case "fastest":
                return GLM5NextExecutionPolicy(
                    sparseAttention: .absorbedDenseHistory,
                    sparseSelection: .init(
                        selection: .parallelThreshold, deviceRanges: true, queryBlocks: true),
                    pagedMetadataReads: .deviceTable, hyperConnection: .nativeNormFused,
                    routedExperts: .fusedDecode, linearAttention: .nativeRecurrenceChunkedPrefill,
                    normalization: .fusedKernel, linearDecodeStep: .fusedEveryStep,
                    routing: .nativeKernel, projections: .fusedInputs, stepEvaluation: .earlyAsync,
                    sparseDecode: .nativeKernels)
            default: return .reference
            }
        }

        private static func seconds(_ duration: Duration) -> Double {
            Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
        }

        private final class ProcessBudget: CBv2ProcessMemoryOwner, @unchecked Sendable {
            private let lock = NSLock()
            private var charged: UInt64 = 0
            func replaceCharge(_ value: UInt64) throws { lock.withLock { charged = value } }
            func recordMaterialization(_ value: UInt64) throws {}
            func withdrawCoverage(_ value: UInt64) throws {}
            func retire() {}
        }

        /// Synchronous per-step decode time of the native text path and of the
        /// CBv2 forward over contiguous rows (no paging, no engine), same tokens.
        private static func forwardBench(glm: GLM5NextModel, prompt: [Int]) throws {
            let steps = 64
            func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] * 1000 }
            var nativeTimes: [Double] = []
            var nativeTokens: [Int] = []
            try MLX.withError { error in
                let cache = glm.newCache(parameters: nil)
                var logits = glm(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), cache: cache)
                var next = argMax(logits[0, -1], axis: -1).asType(.int32)
                eval(next)
                try error.check()
                for _ in 0 ..< steps {
                    let start = ContinuousClock.now
                    logits = glm(next.reshaped(1, 1), cache: cache)
                    next = argMax(logits[0, -1], axis: -1).asType(.int32)
                    eval(next)
                    nativeTimes.append(seconds(start.duration(to: .now)))
                    nativeTokens.append(next.item(Int.self))
                }
                try error.check()
            }
            let target = glm.textTower
            let kinds = target.cbv2LayerKinds
            let rows = kinds.map {
                CBv2FullSequenceKV(
                    promptLength: 0, maxLength: prompt.count + steps + 8, kvHeads: $0.kvHeads,
                    headDim: $0.headDim, valueHeadDim: $0.valueHeadDim)
            }
            let caches = kinds.indices.map {
                CBv2LayerCache(layerIndex: $0, kind: kinds[$0], rows: [rows[$0]])
            }
            let recurrent = try CBv2RecurrentRequestState(spec: target.cbv2RecurrentStateSpec)
            func step(_ ids: [Int]) throws -> MLXArray {
                let evaluation = try recurrent.bind()
                let output = try target.prepareCBv2Forward(
                    tokens: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count), caches: caches,
                    recurrentState: [evaluation])
                let next = argMax(output.logits[0, -1], axis: -1).asType(.int32)
                eval([next] + (try evaluation.evaluate()) + caches.flatMap { $0.innerState() })
                try evaluation.commit()
                return next
            }
            var cbv2Times: [Double] = []
            var cbv2Tokens: [Int] = []
            var next = try step(prompt)
            for _ in 0 ..< steps {
                let start = ContinuousClock.now
                next = try step([next.item(Int.self)])
                cbv2Times.append(seconds(start.duration(to: .now)))
                cbv2Tokens.append(next.item(Int.self))
            }
            for cache in caches { cache.setRows([]) }
            try recurrent.release()
            print(
                "GLM53_MTP forward_bench sync_step_ms native=\(median(nativeTimes)) "
                    + "cbv2_contiguous=\(median(cbv2Times)) tokens_equal=\(nativeTokens == cbv2Tokens)"
            )
        }

        @Test(.enabled(if: enabled))
        func realArtifactNativeMTPThroughput() async throws {
            let root = URL(
                fileURLWithPath: try #require(Self.environment["GLM53_REAL_SMOKE_DIR"]),
                isDirectory: true)
            let modelDirectory = URL(
                fileURLWithPath: try #require(Self.environment["GLM53_REAL_MODEL_DIR"]),
                isDirectory: true)
            let identityFile = try JSONDecoder().decode(
                Identity.self, from: Data(contentsOf: root.appendingPathComponent("identity.json")))
            var prompts = try JSONDecoder().decode(
                [Prompt].self, from: Data(contentsOf: root.appendingPathComponent("prompts.json")))
            // Optional long-context case: the first smoke prompt's ids repeated
            // to this many prompt tokens.
            if let long = Self.environment["GLM53_MTP_LONG_PROMPT_TOKENS"].flatMap(Int.init),
                long > 0, let first = prompts.first
            {
                let ids = (0 ..< long).map { first.ids[$0 % first.ids.count] }
                prompts.append(Prompt(name: "long\(long)", ids: ids))
            }
            let outputTokens = Self.environment["GLM53_REAL_OUTPUT_TOKENS"].flatMap(Int.init) ?? 128
            let modes = (Self.environment["GLM53_MTP_MODES"] ?? "0,1,2,3,adaptive")
                .split(separator: ",").compactMap { Mode(String($0)) }
            let temperature =
                Self.environment["GLM53_MTP_TEMPERATURE"].flatMap(Float.init) ?? 0
            let policyName = Self.environment["GLM53_REAL_POLICY"] ?? "fastest"
            let context = 4096
            // The engine admits its worst-case step scratch (512-token rows at
            // this context) inside the KV capacity.
            let kvBytes = 48 << 30
            let identity = GLM5NextArtifactIdentity(
                repository: identityFile.repository, revision: identityFile.revision,
                files: identityFile.files.mapValues { .init(bytes: $0.bytes, sha256: $0.sha256) })
            let roots = GLM5NextArtifactLoadRoots()
            let artifact = try GLM5NextArtifactLoader(
                directory: modelDirectory, expectedIdentity: identity,
                storage: Self.environment["GLM53_REAL_STORAGE"] == "vontraQ4"
                    ? .vontraQ4 : .nvfp4Hybrid,
                constructionRoots: roots, executionPolicy: Self.policy(named: policyName))
            let glm = try GLM5NextModel(artifact: artifact, loadMTP: true)
            let predictor = try #require(glm.mtp)
            try roots.evaluate(
                glm.parameters().flattened().map(\.1) + predictor.parameters().flattened().map(\.1))
            GLM5NextRealResidency.applyIfRequested(ProcessInfo.processInfo.environment)
            let target = glm.textTower
            print(
                "GLM53_MTP policy=\(policyName) temperature=\(temperature) "
                    + "output_tokens=\(outputTokens) active_gb=\(Double(Memory.activeMemory) / 1e9)"
            )

            let maximumBuffer = GPU.deviceInfo().maxBufferSize
            func pagedConfig() throws -> PagedKVPoolConfig {
                try target.makeCBv2PagedBackend(
                    kvBytesCapacity: kvBytes, maximumBatchSize: 1, maximumQueryTokens: 512,
                    maximumContextTokens: context, maximumScratchBytes: min(kvBytes, maximumBuffer),
                    segmentBytes: 64 << 20, maximumBufferBytes: maximumBuffer
                ).pool.config
            }
            let scheduler = CBv2SchedulerConfig(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 512, prefillChunkSize: 512,
                maxWaiting: 4, enablePrefixCache: false)

            func generate(_ engine: EngineV2, prompt: Prompt, id: UInt64) async throws -> Run {
                let stream = try engine.submit(
                    .init(
                        id: .init(id), promptTokens: prompt.ids.map(Int.init),
                        sampling: .init(temperature: temperature, seed: 7),
                        maxTokens: outputTokens))
                var run = Run()
                var first: ContinuousClock.Instant?
                var last = ContinuousClock.now
                for await event in stream {
                    if case .finished(_, let usage) = event {
                        let timing = usage.timing
                        run.decodeSteps = Int(timing.decodeSteps)
                        run.chainedSteps = Int(timing.chainedDecodeSteps)
                    }
                    guard case .delta(_, let values, _) = event, !values.isEmpty else { continue }
                    if first == nil { first = .now }
                    last = .now
                    run.tokens += values
                }
                // The first delta carries the prefill token; decode spans the rest.
                run.decodeSeconds = first.map { Self.seconds($0.duration(to: last)) } ?? 0
                return run
            }

            if Self.environment["GLM53_MTP_FORWARD_BENCH"] == "1", let prompt = prompts.first {
                try Self.forwardBench(glm: glm, prompt: prompt.ids.map(Int.init))
            }
            CBv2StepProfiler.enabled = Self.environment["GLM53_MTP_STEP_PROFILE"] == "1"
            CBv2StepProfiler.reset()
            var baseline: [String: [Int]] = [:]
            var nextID: UInt64 = 5300
            for mode in modes {
                let process = ProcessBudget()
                let engine: EngineV2
                switch mode {
                case .targetOnly:
                    let backend = try target.makeCBv2PagedBackend(
                        kvBytesCapacity: kvBytes, maximumBatchSize: 1, maximumQueryTokens: 512,
                        maximumContextTokens: context,
                        maximumScratchBytes: min(kvBytes, maximumBuffer),
                        segmentBytes: 64 << 20, maximumBufferBytes: maximumBuffer)
                    engine = EngineV2(
                        model: CBv2SteppableLanguageModelAdapter(target),
                        layerKinds: target.cbv2LayerKinds, backend: backend,
                        cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                        sampler: CBv2DefaultSampler(), schedulerConfig: scheduler,
                        admissionConfig: target.cbv2NativeAdmissionConfig,
                        processMemoryOwner: process)
                case .serial, .rectangular, .adaptive:
                    let verification: GLM5NextMTPVerification
                    switch mode {
                    case .rectangular(let depth): verification = .rectangular(depth: depth)
                    case .adaptive: verification = .rectangular(depth: 3)
                    default: verification = .serialTarget
                    }
                    let factory = try GLM5NextMTPExecution(
                        target: target, predictor: predictor, maximumInputs: context,
                        storage: .contiguous, verification: verification)
                    let assistant = try GLM5NextMTPAdapter(
                        target: target, predictor: predictor, factory: factory)
                    let resources = try factory.makeNativeResources(
                        assistant: assistant, config: pagedConfig(), processMemoryOwner: process,
                        construction: NativeConstructionScope())
                    engine = EngineV2(
                        model: resources.model, layerKinds: target.cbv2LayerKinds,
                        backend: resources.backend, cacheProvider: resources.cacheProvider,
                        sampler: CBv2DefaultSampler(), schedulerConfig: scheduler,
                        admissionConfig: target.cbv2NativeAdmissionConfig,
                        mtpDrafter: assistant,
                        mtpConfig: .init(
                            enabled: true, maxDraftTokens: verification.depth,
                            maxSpeculativeBatch: 1,
                            fixedDraftTokens: mode == .adaptive ? nil : verification.depth,
                            verificationMode: verification.engineMode),
                        processMemoryOwner: process, nativeCompletionTracking: true,
                        nativeExecutionContract: resources.contract)
                    if let reason = engine.mtpInactiveReason {
                        Issue.record("GLM53_MTP \(mode) inactive: \(reason)")
                    }
                }
                var tokens = 0
                var seconds = 0.0
                var matches = 0
                for prompt in prompts {
                    nextID += 1
                    let run = try await generate(engine, prompt: prompt, id: nextID)
                    tokens += max(0, run.tokens.count - 1)
                    seconds += run.decodeSeconds
                    if mode == .targetOnly { baseline[prompt.name] = run.tokens }
                    let reference = baseline[prompt.name]
                    let match = reference == run.tokens
                    if match { matches += 1 }
                    let divergence =
                        reference.map { reference in
                            zip(reference, run.tokens).enumerated()
                                .first { $0.element.0 != $0.element.1 }?.offset
                                ?? min(reference.count, run.tokens.count)
                        } ?? -1
                    print(
                        "GLM53_MTP \(mode) \(prompt.name) tokens=\(run.tokens.count) "
                            + "decode_tok_s=\(Double(max(0, run.tokens.count - 1)) / max(run.decodeSeconds, 1e-9)) "
                            + "matches_target_only=\(match) first_divergence=\(divergence) "
                            + "decode_steps=\(run.decodeSteps) chained=\(run.chainedSteps)")
                }
                var summary =
                    "GLM53_MTP \(mode) total decode_tok_s=\(Double(tokens) / max(seconds, 1e-9)) "
                    + "matches=\(matches)/\(prompts.count)"
                if let metrics = engine.mtpMetricsSnapshot() {
                    let acceptance =
                        metrics.proposedTokens > 0
                        ? Double(metrics.acceptedTokens) / Double(metrics.proposedTokens) : 0
                    summary +=
                        " proposed=\(metrics.proposedTokens) accepted=\(metrics.acceptedTokens)"
                        + " acceptance=\(acceptance) rounds=\(metrics.rounds)"
                        + " rect_rounds=\(metrics.rectangularVerificationRounds)"
                        + " serial_rounds=\(metrics.serialVerificationRounds)"
                        + " per_position=\(metrics.perPositionAccepted)"
                        + " depths=\(metrics.depthSelections.sorted { $0.key < $1.key })"
                        + " fallbacks=\(metrics.controllerFallbacks)"
                }
                print(summary)
                if CBv2StepProfiler.enabled {
                    print("GLM53_MTP \(mode) step_profile\n" + CBv2StepProfiler.summaryTable())
                    CBv2StepProfiler.reset()
                }
                if mode == .targetOnly {
                    await engine.shutdown()
                } else {
                    let outcome = await engine.shutdownReportingNativeCompletion()
                    guard case .quiescent = outcome else {
                        Issue.record("GLM53_MTP \(mode) native shutdown: \(outcome)")
                        return
                    }
                }
                Memory.clearCache()
            }
        }
    }
}
