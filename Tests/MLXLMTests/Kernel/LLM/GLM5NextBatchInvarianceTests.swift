// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Opt-in (`GLM53_BATCH_INVARIANCE=1`) diagnostic: the same prompt in every
    /// row of a B-row batch should give row 0 exactly the B=1 logits. Prints
    /// max |Δ| of B=2/B=16 row 0 vs B=1 for the serving policy and for the
    /// serving policy with one member reset to its reference value, to locate
    /// batch-dependent arithmetic. Tiny synthetic FP32 weights; asserts nothing.
    @Suite(.serialized)
    struct GLM5NextBatchInvarianceTests {
        private func model(_ policy: GLM5NextExecutionPolicy) throws -> GLM5NextTextModel {
            let configuration = try GLM5NextInventoryFixture.configuration(layers: 4)
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: configuration,
                tensors: GLM5NextWeightInventory.expectedTensors(
                    configuration: configuration, storage: .float32, shard: "invariance.safetensors"
                ),
                storage: .float32)
            return try GLM5NextTextModel(
                configuration: configuration.textConfig, inventory: inventory,
                tensor: { name in
                    let header = try #require(inventory.entries[name]?.tensor)
                    let count = header.shape.reduce(1, *)
                    let salt = name.utf8.reduce(0) { ($0 * 31 + Int($1)) % 1009 }
                    let values: [Float] = (0 ..< count).map { index in
                        let wave = Float(sin(Double(index) * 0.731 + Double(salt)))
                        if name.contains("norm.weight") || name.hasSuffix("_scale") {
                            return 1 + 0.25 * wave
                        }
                        if name.hasSuffix("A_log") { return -0.7 }
                        if name.hasSuffix("dt_bias") { return -0.2 }
                        if name.hasSuffix("_base") || name.hasSuffix("e_score_correction_bias") {
                            return 0
                        }
                        if name.hasSuffix(".bias") { return 0.05 * wave }
                        return 0.25 * wave
                    }
                    return MLXArray(values, header.shape)
                }, executionPolicy: policy)
        }

        /// Row 0 logits of a prefill then three decode steps with `batch`
        /// identical rows.
        private func rowZero(_ model: GLM5NextTextModel, batch: Int) throws -> [MLXArray] {
            let prompt: [Int32] = [4, 5, 6, 7, 8, 9, 3, 2, 11, 7]
            let cache = model.newTextCache()
            var outputs: [MLXArray] = []
            var logits = try model.forward(
                tokens: MLXArray(
                    Array(repeating: prompt, count: batch).flatMap { $0 },
                    [batch, prompt.count]), cache: cache
            ).logits
            outputs.append(logits[0, -1])
            for step in 0 ..< 3 {
                let token = Int32(3 + step)
                logits = try model.forward(
                    tokens: MLXArray(Array(repeating: token, count: batch), [batch, 1]),
                    cache: cache
                ).logits
                outputs.append(logits[0, -1])
            }
            eval(outputs)
            return outputs
        }

        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_BATCH_INVARIANCE"] == "1"))
        func rowZeroDependsOnBatchSize() throws {
            try Device.withDefaultDevice(.gpu) {
                var variants: [(String, GLM5NextExecutionPolicy)] = [("serving", .serving)]
                func reset(_ name: String, _ body: (inout GLM5NextExecutionPolicy) -> Void) {
                    var policy = GLM5NextExecutionPolicy.serving
                    body(&policy)
                    variants.append((name, policy))
                }
                reset("hc=reference") { $0.hyperConnection = .reference }
                reset("routed=reference") { $0.routedExperts = .reference }
                reset("routing=reference") { $0.routing = .reference }
                reset("projections=separate") { $0.projections = .separate }
                reset("norm=reference") { $0.normalization = .reference }
                reset("kdaStep=composed") { $0.linearDecodeStep = .composed }
                reset("sparseDecode=reference") { $0.sparseDecode = .reference }
                variants.append(("reference", .reference))
                for (name, policy) in variants {
                    let target = try model(policy)
                    let single = try rowZero(target, batch: 1)
                    var line = "GLM53_BATCH_INVARIANCE \(name)"
                    for batch in [2, 9, 16] {
                        let rows = try rowZero(target, batch: batch)
                        let deltas = zip(rows, single).map {
                            abs($0.asType(.float32) - $1.asType(.float32)).max().item(Float.self)
                        }
                        line +=
                            " B\(batch)="
                            + deltas.map { String(format: "%.2e", $0) }
                            .joined(separator: ",")
                    }
                    print(line)
                }
            }
        }
    }
}
