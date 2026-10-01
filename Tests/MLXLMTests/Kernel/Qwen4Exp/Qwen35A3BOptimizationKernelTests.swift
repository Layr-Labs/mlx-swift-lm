import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Tests of the load-time checks and the decode routes in
    /// `Qwen35A3BOptimization.swift` that need MLX arrays: the exact
    /// projection check, the loaded-target check on a tiny Qwen3.5 model,
    /// and the router and combine closures under a decode installation.
    @Suite
    struct Qwen35A3BOptimizationKernelTests {
        typealias Support = Qwen4ExpKernelSupport

        static func mismatch(_ error: any Error) -> (field: String, expected: String)? {
            guard case .mismatch(let field, let expected, _)? = error as? Qwen35A3BArtifactError
            else { return nil }
            return (field, expected)
        }

        static func projection(
            outputs: Int = 8, bits: Int = 4, groupSize: Int = 64, dtype: DType = .bfloat16
        ) -> QuantizedLinear {
            let weight = MLXRandom.normal([outputs, 64], key: MLXRandom.key(9300)).asType(dtype)
            return QuantizedLinear(weight: weight, bias: nil, groupSize: groupSize, bits: bits)
        }

        @Test func exactProjectionChecksEachPackingField() throws {
            let valid = Self.projection()
            let returned = try qwen35A3BValidateExactProjection(valid, path: "p")
            #expect(returned === valid)

            let noBiases = QuantizedLinear(
                weight: valid.weight, scales: valid.scales, biases: nil, groupSize: 64, bits: 4)
            let cases: [(QuantizedLinear, String, String)] = [
                (Self.projection(bits: 8), "loaded.p.bits", "4"),
                (Self.projection(groupSize: 32), "loaded.p.group_size", "64"),
                (noBiases, "loaded.p.mode", "affine with biases"),
                (Self.projection(dtype: .float16), "loaded.p.scale_dtype", "bfloat16"),
                (Self.projection(outputs: 6), "loaded.p.output_tiling", "multiple of 4"),
            ]
            for (layer, field, expected) in cases {
                do {
                    _ = try qwen35A3BValidateExactProjection(layer, path: "p")
                    Issue.record("no error for \(field)")
                } catch {
                    let found = Self.mismatch(error)
                    #expect(found?.field == field, "\(error)")
                    #expect(found?.expected == expected, "\(error)")
                }
            }
        }

        static func target() throws -> Qwen35TextModel {
            let json = """
                {"model_type":"qwen3_5_moe_text","hidden_size":64,"num_hidden_layers":4,
                 "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,
                 "linear_num_value_heads":4,"linear_num_key_heads":1,"linear_key_head_dim":64,
                 "linear_value_head_dim":64,"linear_conv_kernel_dim":4,"full_attention_interval":2,
                 "vocab_size":64,"num_experts":0,"num_experts_per_tok":0,
                 "moe_intermediate_size":64,"shared_expert_intermediate_size":64}
                """
            MLXRandom.seed(9310)
            return Qwen35TextModel(
                try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8)))
        }

        /// The loaded-target check: the embedding dtype first, then each
        /// exact projection, then the projection count of the contract.
        @Test func loadedTargetCheckRunsInOrder() throws {
            let contract = try Qwen35A3BArtifactContract.inspect(fixture: .eigenLabsRouter8)
            let model = try Self.target()

            #expect(model.model.embedTokens.weight.dtype == .float32)
            do {
                try qwen35A3BValidateLoadedExactTarget(model, contract: contract)
                Issue.record("float32 embedding accepted")
            } catch {
                #expect(Self.mismatch(error)?.field == "loaded.embed_tokens.dtype", "\(error)")
            }

            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
            eval(model)
            do {
                try qwen35A3BValidateLoadedExactTarget(model, contract: contract)
                Issue.record("unquantized projections accepted")
            } catch {
                let found = Self.mismatch(error)
                #expect(found?.field.hasSuffix(".type") == true, "\(error)")
                #expect(found?.expected == "QuantizedLinear", "\(error)")
            }

            quantize(
                model: model, groupSize: 64, bits: 4, mode: .affine,
                filter: { _, module in
                    guard let linear = module as? Linear else { return false }
                    return linear.weight.dim(-1).isMultiple(of: 64)
                })
            eval(model)
            do {
                try qwen35A3BValidateLoadedExactTarget(model, contract: contract)
                Issue.record("a 4-layer model matched the 40-layer contract")
            } catch {
                let found = Self.mismatch(error)
                #expect(found?.field == "loaded.exact_projection_count", "\(error)")
                // 30 recurrent layers * 5 + 10 attention layers * 4
                // + 40 layers * 3 shared-expert projections + lm_head.
                #expect(found?.expected == "311", "\(error)")
            }
        }

        static func probabilities(rows: Int) -> MLXArray {
            let logits = MLXArray(0 ..< rows * 256).asType(.float32).reshaped([rows, 256])
            return softmax(sin(logits * Float(0.173)) + cos(logits * Float(0.071)), axis: -1)
                .asType(.bfloat16)
        }

        /// Under a decode installation the router closure uses the
        /// row-owned kernel for 1 to 16 rows and the stock top-k for more
        /// rows. Without normalization it is always the stock top-k.
        @Test func installedRouterSelectsTheRowOwnedKernelByWidth() throws {
            let contract = try Qwen35A3BArtifactContract.inspect(fixture: .eigenLabsRouter8)
            let installation = try Qwen35A3BConstructionInstallation.install(
                contract: contract, profile: .full)
            let (routed, plain) = Qwen35A3BConstructionContext.withInstallation(installation) {
                (
                    qwen35A3BRouterFinalizer(hidden: 2_048, experts: 256, topK: 8, normalize: true),
                    qwen35A3BRouterFinalizer(
                        hidden: 2_048, experts: 256, topK: 8, normalize: false)
                )
            }

            let narrow = Self.probabilities(rows: 2)
            let (ids, scores) = routed(narrow)
            let (kernelIDs, kernelScores) = qwen35A3BRowOwnedRoute(narrow, rows: 2)
            #expect(Support.isEqual(ids, kernelIDs))
            #expect(Support.isEqual(scores, kernelScores))
            #expect(ids.dtype == .uint32)

            let wide = Self.probabilities(rows: 17)
            let (wideIDs, wideScores) = routed(wide)
            #expect(wideIDs.shape == [17, 8])
            // The stock route normalizes the 8 scores of each row.
            let sums = wideScores.asType(.float32).sum(axis: -1)
            #expect(Support.isClose(sums, MLXArray.ones([17]), atol: 1e-2, rtol: 0))

            let (plainIDs, plainScores) = plain(narrow)
            #expect(plainIDs.shape == [2, 8])
            #expect(
                Support.isEqual(
                    plainScores, takeAlong(narrow, plainIDs, axis: -1)),
                "no normalization")
        }

        /// Under a decode installation the combine closure uses the kernel
        /// for 1 or 2 rows only. Three rows use the stock weighted sum.
        ///
        /// Tolerance: `1e-2 + 1e-2 * |reference|`. The stock sum and the
        /// float32 reference differ by bfloat16 rounding of 8 products.
        @Test func installedCombineFallsBackForThreeRows() throws {
            let contract = try Qwen35A3BArtifactContract.inspect(fixture: .eigenLabsRouter8)
            let installation = try Qwen35A3BConstructionInstallation.install(
                contract: contract, profile: .decode)
            let (combine, other) = Qwen35A3BConstructionContext.withInstallation(installation) {
                (
                    qwen35A3BExpertCombiner(hidden: 2_048, topK: 8),
                    qwen35A3BExpertCombiner(hidden: 1_024, topK: 8)
                )
            }
            let routed = sin(MLXArray(0 ..< 3 * 8 * 2_048).asType(.float32) * Float(0.013))
                .asType(.bfloat16).reshaped([3, 8, 2_048])
            let scores = softmax(
                MLXArray(0 ..< 3 * 8).asType(.float32).reshaped([3, 8]), axis: -1
            ).asType(.bfloat16)
            let expected =
                (routed.asType(.float32)
                * expandedDimensions(scores.asType(.float32), axis: -1)).sum(axis: -2)
            let actual = combine(routed, scores)
            #expect(actual.shape == [3, 2_048])
            #expect(
                Support.isClose(actual, expected, atol: 1e-2, rtol: 1e-2),
                "max difference \(Support.maxAbsDifference(actual, expected))")

            let small = routed[0..., 0..., 0 ..< 1_024]
            let otherActual = other(small, scores)
            let otherExpected =
                (small.asType(.float32)
                * expandedDimensions(scores.asType(.float32), axis: -1)).sum(axis: -2)
            #expect(
                Support.isClose(otherActual, otherExpected, atol: 1e-2, rtol: 1e-2),
                "max difference \(Support.maxAbsDifference(otherActual, otherExpected))")
        }
    }
}
