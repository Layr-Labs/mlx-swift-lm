import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Whether this device and the default stream run the NAX dense kernel.
private var naxDenseLane: Bool {
    ModelOptNAXDenseLinear.naxAvailable && MiMoV26NAXGatherQMM.gpuStream(.default)
}

extension KernelTests {
    /// The fused-decode NAX dense product against the graph path it replaces
    /// (FP8: decoded BF16 weight and MLX's NAX GEMM; NVFP4: MLX's NAX product
    /// and `scaledCast`). The GPU cases need Neural Accelerators and are
    /// skipped elsewhere.
    @Suite(.serialized)
    struct ModelOptNAXDenseLinearTests {
        typealias Plan = ModelOptNAXDenseLinear.TilePlan

        @Test func tilePlansParseAndAdmit() {
            for plan in Plan.sweep {
                #expect(Plan(plan.description) == plan)
                #expect(plan.admits(inputs: 4096) && plan.admits(inputs: 8192))
                #expect(plan.admits(inputs: 5376))
            }
            #expect(Plan.plan(inputs: 4096).description == "seg+pf 128x64 bk128 pad8192")
            #expect(Plan.plan(inputs: 320).description == "db 128x64 bk64")
            #expect(Plan("db 128x64 bk128").map { $0.admits(inputs: 4096) } == false)
            #expect(Plan("seg 128x64 bk128").map { $0.admits(inputs: 4096) } == false)
            #expect(Plan("db 96x64 bk64") == nil)
            #expect(Plan("fast 128x64 bk64") == nil)
            #expect(!Plan.sweep[2].admits(inputs: 320))

            let saved = ModelOptNAXDenseLinear.planOverrides
            defer { ModelOptNAXDenseLinear.planOverrides = saved }
            ModelOptNAXDenseLinear.planOverrides = [4096: Plan.sweep[0], 320: Plan.sweep[2]]
            #expect(ModelOptNAXDenseLinear.plan(inputs: 4096) == Plan.sweep[0])
            #expect(ModelOptNAXDenseLinear.plan(inputs: 8192) == Plan.plan(inputs: 8192))
            // A pinned plan that does not admit the width falls back.
            #expect(ModelOptNAXDenseLinear.plan(inputs: 320) == Plan.plan(inputs: 320))
        }

        /// The routes mirror MLX's dispatch: only where the graph path runs
        /// MLX's regular NAX kernels, only for prefill-sized chunks.
        @Test func routesTakeOnlyPrefillChunksOnRegularNAXProducts() {
            #expect(
                ModelOptNAXDenseLinear.mlxRegularNAXGEMM(rows: 2048, inputs: 4096, outputs: 8192))
            #expect(
                ModelOptNAXDenseLinear.mlxRegularNAXGEMM(rows: 2048, inputs: 4096, outputs: 256))
            #expect(
                ModelOptNAXDenseLinear.mlxRegularNAXGEMM(rows: 2048, inputs: 8192, outputs: 4096))
            #expect(
                !ModelOptNAXDenseLinear.mlxRegularNAXGEMM(rows: 1024, inputs: 4096, outputs: 256))
            #expect(!ModelOptNAXDenseLinear.mlxRegularNAXGEMM(rows: 128, inputs: 8192, outputs: 64))

            let saved = ModelOptNAXDenseLinear.isEnabled
            defer { ModelOptNAXDenseLinear.isEnabled = saved }
            func fp8(_ rows: Int, _ tiles: [Int] = [4096], dtype: DType = .bfloat16) -> Bool {
                ModelOptNAXDenseLinear.takesFP8(
                    rows: rows, inputs: 4096, outputs: 4096, tileOutputs: tiles, dtype: dtype)
            }
            func nvfp4(_ rows: Int, outputs: Int = 4096) -> Bool {
                ModelOptNAXDenseLinear.takesNVFP4(
                    rows: rows, inputs: 5376, outputs: outputs, dtype: .bfloat16)
            }
            ModelOptNAXDenseLinear.isEnabled = false
            #expect(!fp8(2048) && !nvfp4(2048))
            ModelOptNAXDenseLinear.isEnabled = true
            #expect(fp8(2048) == naxDenseLane)
            #expect(nvfp4(2048) == (naxDenseLane && ModelOptNAXDenseLinear.mlxTF32Enabled))
            #expect(!fp8(ModelOptNAXDenseLinear.minimumRows - 1))
            #expect(!fp8(1) && !nvfp4(1) && !fp8(16) && !nvfp4(16))
            #expect(!fp8(2048, dtype: .float16) && !fp8(2048, dtype: .float32))
            // One eager tile on MLX's split-K route keeps the whole call on MLX.
            #expect(!fp8(256, [4096, 64]))
        }

        /// Super's dense prefill projections at a 2,048-token chunk, then
        /// partial row tiles.
        @Test(
            .enabled(if: naxDenseLane),
            arguments: [
                (2048, 4096, 18560, false, false),  // Mamba in_proj
                (2048, 8192, 4096, false, false),  // Mamba out_proj
                (2048, 4096, 4096, false, false),  // attention q / o
                (2048, 4096, 256, false, false),  // attention k / v
                (2048, 4096, 5376, true, true),  // shared expert up + relu2
                (2048, 5376, 4096, true, false),  // shared expert down
                (300, 4096, 4096, false, false),
                (300, 5376, 4096, true, false),
            ])
        func matchesCurrentGraphPath(
            rows: Int, inputs: Int, outputs: Int, nvfp4: Bool, relu2: Bool
        ) throws {
            let result = try ModelOptNAXDenseLinear.compareWithReference(
                rows: rows, inputs: inputs, outputs: outputs, format: nvfp4 ? .nvfp4 : .fp8,
                seed: UInt64(rows + inputs + outputs), squaredReLU: relu2, iterations: 3)
            print(
                "NAX dense \(nvfp4 ? "NVFP4" : "FP8") \(rows)x\(inputs)->\(outputs) "
                    + "relu2 \(relu2): \(result)")
            #expect(result.maxRel.isFinite && result.maxRel <= 0x1p-7)
            #expect(result.bitIdentical, "not bit-identical: \(result)")
        }

        /// Every swept plan issues the same tensor ops in the same order.
        @Test(.enabled(if: naxDenseLane), arguments: Plan.sweep)
        func everyPlanMatchesCurrentGraphPath(plan: Plan) throws {
            for nvfp4 in [false, true] {
                let result = try ModelOptNAXDenseLinear.compareWithReference(
                    rows: 520, inputs: 1024, outputs: 640, format: nvfp4 ? .nvfp4 : .fp8,
                    seed: 5, squaredReLU: nvfp4, plan: plan, iterations: 1)
                #expect(result.bitIdentical, "\(nvfp4 ? "NVFP4" : "FP8") \(result)")
            }
        }

        /// The graph path takes the kernel for a tiled prefill chunk only; a
        /// decode row and a row-exact batch keep their own products.
        @Test(.enabled(if: naxDenseLane), arguments: [false, true])
        func graphForwardHookTakesTheKernelOnlyForPrefill(nvfp4: Bool) throws {
            var fixtures = ModelOptGraphFixtures(seed: nvfp4 ? 21 : 22)
            // 1,088 outputs: MLX's NVFP4 product at 256 rows is not split-K.
            let (inputs, outputs) = (512, 1088)
            let layer =
                nvfp4
                ? try fixtures.nvfp4Projection(
                    outputs: outputs, inputs: inputs, magnitude: 3, dtype: .bfloat16)
                : try fixtures.fp8Projection(
                    outputs: outputs, inputs: inputs, magnitude: 3, dtype: .bfloat16)
            layer.accumulation = .tiled
            layer.rowExactBatches = true
            let rows = MLXArray(fixtures.values(256 * inputs, magnitude: 3))
                .reshaped(2, 128, inputs).asType(.bfloat16)

            let saved = ModelOptNAXDenseLinear.isEnabled
            defer { ModelOptNAXDenseLinear.isEnabled = saved }
            ModelOptNAXDenseLinear.isEnabled = false
            let reference = try layer.graphForward(rows, squaredReLU: nvfp4)
            let decode = [layer(rows[0, 0 ..< 1]), layer(rows[0, 0 ..< 8])]
            ModelOptNAXDenseLinear.isEnabled = true
            let taken =
                nvfp4
                ? ModelOptNAXDenseLinear.takesNVFP4(
                    rows: 256, inputs: inputs, outputs: outputs, dtype: .bfloat16)
                : ModelOptNAXDenseLinear.takesFP8(
                    rows: 256, inputs: inputs, outputs: outputs,
                    tileOutputs: layer.fp8EagerTiles(rows: 256).map(\.count), dtype: .bfloat16)
            #expect(taken == (!nvfp4 || ModelOptNAXDenseLinear.mlxTF32Enabled))
            let hooked = try layer.graphForward(rows, squaredReLU: nvfp4)
            #expect(hooked.shape == [2, 128, outputs])
            #expect(modelOptBitwiseEqual(hooked, reference))
            #expect(modelOptBitwiseEqual(layer(rows[0, 0 ..< 1]), decode[0]))
            #expect(modelOptBitwiseEqual(layer(rows[0, 0 ..< 8]), decode[1]))
        }
    }
}
