// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXVLM

/// Explicit offline full-checkpoint perplexity probe for comparing local
/// quantization variants. It runs only when `GLM53_PPL_DIR` names a prepared
/// directory with `tokens.json` and `identities/<GLM53_PPL_LABEL>.json`, and
/// `GLM53_REAL_MODEL_DIR` names the artifact; it downloads nothing and asserts
/// no quality threshold. Per-token NLL and argmax are written to
/// `results/<label>.json` so variants can be compared offline.
extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextRealPerplexityTests {
        private struct Identity: Decodable {
            struct File: Decodable {
                let bytes: Int
                let sha256: String
            }
            let repository: String
            let revision: String
            let files: [String: File]
        }

        private struct Document: Decodable {
            let name: String
            let kind: String
            /// Leading template tokens whose own prediction is not scored.
            let prefixTokens: Int
            let ids: [Int32]

            enum CodingKeys: String, CodingKey {
                case name, kind, ids
                case prefixTokens = "prefix_tokens"
            }
        }

        private struct DocumentResult: Encodable {
            let name: String
            let kind: String
            let tokens: Int
            let meanNLL: Double
            /// Mean NLL of the predictions made by each prefill chunk.
            let chunkMeanNLL: [Double]
            /// NLL of the token at position i + 1 given positions 0...i, for
            /// every scored position i (prefixTokens - 1 ..< tokens - 1).
            let nll: [Float]
            let argmax: [Int32]
            let seconds: Double
        }

        private struct Result: Encodable {
            let label: String
            let repository: String
            let revision: String
            let policy: String
            let storage: String
            let chunk: Int
            let predicted: Int
            let meanNLL: Double
            let perplexity: Double
            let loadSeconds: Double
            let documents: [DocumentResult]
        }

        private static var environment: [String: String] { ProcessInfo.processInfo.environment }
        private static var pplDirectory: URL? {
            environment["GLM53_PPL_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        }

        private static func seconds(_ start: ContinuousClock.Instant) -> Double {
            let d = start.duration(to: .now)
            return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
        }

        /// The same policy must be used for every compared variant.
        private static func policy(_ name: String) throws -> GLM5NextExecutionPolicy {
            switch name {
            case "reference": return .reference
            case "fast":
                return GLM5NextExecutionPolicy(
                    sparseAttention: .absorbed,
                    sparseSelection: .init(selection: .parallelThreshold, deviceRanges: true),
                    pagedMetadataReads: .deviceTable, hyperConnection: .nativeRequired,
                    routedExperts: .nativeCombineSortedPrefill, linearAttention: .nativeRecurrence)
            case "fastest":
                return GLM5NextExecutionPolicy(
                    sparseAttention: .absorbedDenseHistory,
                    sparseSelection: .init(
                        selection: .parallelThreshold, deviceRanges: true, queryBlocks: true),
                    pagedMetadataReads: .deviceTable, hyperConnection: .nativeNormFused,
                    routedExperts: .fusedDecode,
                    linearAttention: .nativeRecurrenceChunkedPrefill, normalization: .fusedKernel,
                    linearDecodeStep: .fusedEveryStep, routing: .nativeKernel,
                    projections: .fusedInputs, stepEvaluation: .earlyAsync,
                    sparseDecode: .nativeKernels)
            default: throw GLM5NextLoadError.invalid("unknown GLM53_REAL_POLICY \(name)")
            }
        }

        @Test(.enabled(if: pplDirectory != nil))
        func realArtifactPerplexity() throws {
            let root = try #require(Self.pplDirectory)
            let label = try #require(Self.environment["GLM53_PPL_LABEL"])
            let model = URL(
                fileURLWithPath: try #require(Self.environment["GLM53_REAL_MODEL_DIR"]),
                isDirectory: true)
            let chunk = Self.environment["GLM53_PPL_CHUNK"].flatMap(Int.init) ?? 512
            let policyName = Self.environment["GLM53_REAL_POLICY"] ?? "reference"
            let identityFile = try JSONDecoder().decode(
                Identity.self,
                from: Data(
                    contentsOf: root.appendingPathComponent("identities/\(label).json")))
            let documents = try JSONDecoder().decode(
                [Document].self, from: Data(contentsOf: root.appendingPathComponent("tokens.json")))
            let identity = GLM5NextArtifactIdentity(
                repository: identityFile.repository, revision: identityFile.revision,
                files: identityFile.files.mapValues { .init(bytes: $0.bytes, sha256: $0.sha256) })
            // Q4 or the closed mixed policy, exactly as the artifact declares it.
            let declared = try JSONDecoder().decode(
                GLM5NextConfiguration.self,
                from: Data(contentsOf: model.appendingPathComponent("config.json")))
            let storage = try GLM5NextWeightInventory.packedStorage(declaredBy: declared)
            print("GLM53_PPL label=\(label) policy=\(policyName) storage=\(storage) chunk=\(chunk)")

            let loadStart = ContinuousClock.now
            let roots = GLM5NextArtifactLoadRoots()
            let artifact = try GLM5NextArtifactLoader(
                directory: model, expectedIdentity: identity, storage: storage,
                constructionRoots: roots, executionPolicy: try Self.policy(policyName))
            let glm = try GLM5NextModel(artifact: artifact, loadMTP: false)
            try roots.evaluate(glm.parameters().flattened().map(\.1))
            let loadSeconds = Self.seconds(loadStart)
            print(
                "GLM53_PPL load_seconds=\(loadSeconds) active_gb=\(Double(Memory.activeMemory) / 1e9)"
            )

            var results: [DocumentResult] = []
            for document in documents {
                let ids = document.ids
                let first = document.prefixTokens - 1
                try #require(first >= 0 && ids.count > document.prefixTokens)
                let start = ContinuousClock.now
                var nll: [Float] = []
                var argmax: [Int32] = []
                var chunkMeans: [Double] = []
                try MLX.withError { error in
                    let cache = glm.newTextCache()
                    for offset in stride(from: 0, to: ids.count, by: chunk) {
                        let end = min(ids.count, offset + chunk)
                        let output = try glm.textTower.forward(
                            tokens: MLXArray(Array(ids[offset ..< end])).reshaped(1, -1),
                            cache: cache)
                        // Scored positions i predict ids[i + 1].
                        let lower = max(offset, first)
                        let upper = min(end, ids.count - 1)
                        guard lower < upper else {
                            eval(cache.layers.flatMap { $0.state })
                            try error.check()
                            continue
                        }
                        let logits = output.logits[0, (lower - offset) ..< (upper - offset)]
                            .asType(.float32)
                        let targets = MLXArray(Array(ids[(lower + 1) ... upper]))
                        let picked = takeAlong(
                            logits, targets.reshaped(-1, 1).asType(.int32), axis: -1
                        ).reshaped(-1)
                        let tokenNLL = logSumExp(logits, axis: -1) - picked
                        let top = argMax(logits, axis: -1).asType(.int32)
                        eval([tokenNLL, top] + cache.layers.flatMap { $0.state })
                        try error.check()
                        let values = tokenNLL.asArray(Float.self)
                        nll += values
                        argmax += top.asArray(Int32.self)
                        chunkMeans.append(values.reduce(0.0) { $0 + Double($1) } / Double(values.count))
                    }
                }
                let mean = nll.reduce(0.0) { $0 + Double($1) } / Double(nll.count)
                let result = DocumentResult(
                    name: document.name, kind: document.kind, tokens: ids.count, meanNLL: mean,
                    chunkMeanNLL: chunkMeans, nll: nll, argmax: argmax,
                    seconds: Self.seconds(start))
                print(
                    "GLM53_PPL doc=\(document.name) kind=\(document.kind) predicted=\(nll.count) "
                        + "mean_nll=\(mean) ppl=\(exp(mean)) chunks=\(chunkMeans) "
                        + "seconds=\(result.seconds)")
                results.append(result)
                Memory.clearCache()
            }
            let predicted = results.reduce(0) { $0 + $1.nll.count }
            let total = results.reduce(0.0) { $0 + $1.meanNLL * Double($1.nll.count) }
            let mean = total / Double(predicted)
            let result = Result(
                label: label, repository: identity.repository, revision: identity.revision,
                policy: policyName, storage: "\(storage)", chunk: chunk, predicted: predicted,
                meanNLL: mean, perplexity: exp(mean), loadSeconds: loadSeconds,
                documents: results)
            print("GLM53_PPL total label=\(label) predicted=\(predicted) mean_nll=\(mean) ppl=\(exp(mean))")
            let directory = root.appendingPathComponent("results", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(result).write(
                to: directory.appendingPathComponent("\(label)-\(policyName).json"))
            #expect(results.count == documents.count)
            #expect(mean.isFinite)
        }
    }
}
