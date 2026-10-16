// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXVLM

/// Explicit offline full-checkpoint smoke. It loads a real local artifact only
/// when `GLM53_REAL_SMOKE_DIR` names a prepared directory with `model`,
/// `identity.json` and `prompts.json`; it downloads nothing.
extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextRealSmokeTests {
        private struct Identity: Decodable {
            struct File: Decodable {
                let bytes: Int
                let sha256: String
            }
            let repository: String
            let revision: String
            let files: [String: File]
        }

        private struct Prompt: Codable {
            let name: String
            let ids: [Int32]
        }

        private struct Result: Codable {
            let name: String
            let promptTokens: Int
            let generated: [Int32]
            let prefillSeconds: Double
            let decodeTokensPerSecond: Double
        }

        private static var smokeDirectory: URL? {
            ProcessInfo.processInfo.environment["GLM53_REAL_SMOKE_DIR"].map {
                URL(fileURLWithPath: $0, isDirectory: true)
            }
        }

        private static func seconds(_ duration: Duration) -> Double {
            Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
        }

        @Test(.enabled(if: smokeDirectory != nil))
        func realArtifactGeneratesGreedyTokens() throws {
            let root = try #require(Self.smokeDirectory)
            let model = URL(
                fileURLWithPath: try #require(
                    ProcessInfo.processInfo.environment["GLM53_REAL_MODEL_DIR"]),
                isDirectory: true)
            let identityFile = try JSONDecoder().decode(
                Identity.self, from: Data(contentsOf: root.appendingPathComponent("identity.json")))
            let prompts = try JSONDecoder().decode(
                [Prompt].self, from: Data(contentsOf: root.appendingPathComponent("prompts.json")))
            let outputTokens =
                ProcessInfo.processInfo.environment["GLM53_REAL_OUTPUT_TOKENS"].flatMap(Int.init)
                ?? 48
            let identity = GLM5NextArtifactIdentity(
                repository: identityFile.repository, revision: identityFile.revision,
                files: identityFile.files.mapValues { .init(bytes: $0.bytes, sha256: $0.sha256) })

            // `fast` and `fastest` are diagnostic throughput probes over
            // non-reference policy members, not qualified serving policies.
            let policyName = ProcessInfo.processInfo.environment["GLM53_REAL_POLICY"] ?? "reference"
            let fast = policyName == "fast" || policyName == "fastest"
            let policy =
                fast
                ? GLM5NextExecutionPolicy(
                    sparseAttention: policyName == "fastest" ? .absorbedDenseHistory : .absorbed,
                    sparseSelection: .init(
                        selection: .parallelThreshold, deviceRanges: true,
                        queryBlocks: policyName == "fastest"),
                    pagedMetadataReads: .deviceTable,
                    hyperConnection: policyName == "fastest" ? .nativeNormFused : .nativeRequired,
                    routedExperts: policyName == "fastest"
                        ? .fusedDecode : .nativeCombineSortedPrefill,
                    linearAttention: policyName == "fastest"
                        ? .nativeRecurrenceChunkedPrefill : .nativeRecurrence,
                    normalization: .fusedKernel,
                    linearDecodeStep: policyName == "fastest" ? .fusedEveryStep : .composed,
                    routing: policyName == "fastest" ? .nativeKernel : .reference,
                    projections: policyName == "fastest" ? .fusedInputs : .separate,
                    stepEvaluation: policyName == "fastest" ? .earlyAsync : .deferred,
                    sparseDecode: policyName == "fastest" ? .nativeKernels : .reference)
                : .reference
            print("GLM53_SMOKE policy=\(fast ? policyName : "reference")")
            let loadStart = ContinuousClock.now
            let roots = GLM5NextArtifactLoadRoots()
            let artifact = try GLM5NextArtifactLoader(
                directory: model, expectedIdentity: identity,
                storage: GLM5NextWeightInventory.packedStorage(
                    declaredBy: JSONDecoder().decode(
                        GLM5NextConfiguration.self,
                        from: Data(contentsOf: model.appendingPathComponent("config.json")))),
                constructionRoots: roots, executionPolicy: policy)
            let glm = try GLM5NextModel(artifact: artifact, loadMTP: false)
            try roots.evaluate(glm.parameters().flattened().map(\.1))
            GLM5NextRealResidency.applyIfRequested(ProcessInfo.processInfo.environment)
            print(
                "GLM53_SMOKE load_seconds=\(Self.seconds(loadStart.duration(to: .now))) "
                    + "active_gb=\(Double(Memory.activeMemory) / 1e9)")

            var results: [Result] = []
            for prompt in prompts {
                try MLX.withError { error in
                    let cache = glm.newCache(parameters: nil)
                    let start = ContinuousClock.now
                    var logits = glm(MLXArray(prompt.ids).reshaped(1, -1), cache: cache)
                    var next = argMax(logits[0, -1], axis: -1).asType(.int32)
                    eval(next)
                    try error.check()
                    let prefill = start.duration(to: .now)
                    var generated: [Int32] = [next.item(Int32.self)]
                    let decodeStart = ContinuousClock.now
                    var stepSeconds: [Double] = []
                    var buildSeconds: [Double] = []
                    while generated.count < outputTokens {
                        let stepStart = ContinuousClock.now
                        logits = glm(next.reshaped(1, 1), cache: cache)
                        next = argMax(logits[0, -1], axis: -1).asType(.int32)
                        // Host graph construction only; GPU work starts at eval.
                        buildSeconds.append(Self.seconds(stepStart.duration(to: .now)))
                        eval(next)
                        try error.check()
                        generated.append(next.item(Int32.self))
                        stepSeconds.append(Self.seconds(stepStart.duration(to: .now)))
                    }
                    let sorted = stepSeconds.sorted()
                    if !sorted.isEmpty {
                        print(
                            "GLM53_SMOKE \(prompt.name) step_ms first=\(stepSeconds[0] * 1000) "
                                + "median=\(sorted[sorted.count / 2] * 1000) "
                                + "min=\(sorted[0] * 1000) max=\(sorted[sorted.count - 1] * 1000) "
                                + "build_median=\(buildSeconds.sorted()[buildSeconds.count / 2] * 1000)"
                        )
                    }
                    let decode = Self.seconds(decodeStart.duration(to: .now))
                    let result = Result(
                        name: prompt.name, promptTokens: prompt.ids.count, generated: generated,
                        prefillSeconds: Self.seconds(prefill),
                        decodeTokensPerSecond: Double(generated.count - 1) / max(decode, 1e-9))
                    print(
                        "GLM53_SMOKE \(prompt.name) prompt=\(prompt.ids.count) "
                            + "prefill_s=\(result.prefillSeconds) "
                            + "decode_tok_s=\(result.decodeTokensPerSecond) ids=\(generated)")
                    results.append(result)
                }
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(results).write(to: root.appendingPathComponent("results.json"))
            #expect(results.count == prompts.count)
            #expect(results.allSatisfy { $0.generated.count == outputTokens })
        }
    }
}
