import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Whether this device and the default stream run the NAX expert kernel.
private var naxExpertLane: Bool {
    ModelOptNVFP4NAXExperts.naxAvailable && MiMoV26NAXGatherQMM.gpuStream(.default)
}

extension KernelTests {
    /// The NAX NVFP4 expert product against MLX's sorted gathered product (the
    /// tiled route it replaces). The GPU cases need Neural Accelerators and
    /// are skipped elsewhere.
    @Suite(.serialized)
    struct ModelOptNVFP4NAXExpertsTests {
        /// The tile plans to sweep on Super's shapes.
        static let sweep: [ModelOptNVFP4NAXExperts.TilePlan] = [
            .init(rows: 64, depth: 64, doubleBuffered: true, groupX: 0, pad: 0),
            .init(rows: 64, depth: 64, doubleBuffered: true, groupX: 32, pad: 0),
            .init(rows: 96, depth: 64, doubleBuffered: true, groupX: 32, pad: 0),
            .init(rows: 128, depth: 64, doubleBuffered: true, groupX: 32, pad: 0),
            .init(rows: 96, depth: 128, doubleBuffered: false, groupX: 32, pad: 0),
            .init(
                rows: 128, depth: 128, doubleBuffered: false, groupX: 32, pad: 8192,
                prefetch: true),
        ]

        @Test func tilePlansParseAndAdmitSuperShapes() {
            typealias Plan = ModelOptNVFP4NAXExperts.TilePlan
            // 2,048 tokens x 22 experts over 512 experts: 88 rows per expert.
            for input in [1024, 2688] {
                #expect(
                    Plan.plan(rows: 45_056, experts: 512, input: input).description
                        == "db 96x64 bk64 gx32")
            }
            for plan in Self.sweep {
                #expect(Plan(plan.description) == plan)
                #expect(plan.admits(input: 1024) && plan.admits(input: 2688))
            }
            #expect(Plan("db 80x64 bk64") == nil)
            #expect(Plan("db 96x64") == nil)
            #expect(Plan("fast 96x64 bk64") == nil)
            #expect(
                !Plan(rows: 64, depth: 128, doubleBuffered: false, groupX: 0, pad: 0)
                    .admits(input: 1024))
            #expect(
                !Plan(rows: 96, depth: 128, doubleBuffered: true, groupX: 0, pad: 0)
                    .admits(input: 1024))

            let pinned = Self.sweep[5]
            let defaults = ModelOptNVFP4NAXExperts.planOverrides
            ModelOptNVFP4NAXExperts.planOverrides = [2688: pinned, 1024: Self.sweep[4]]
            defer { ModelOptNVFP4NAXExperts.planOverrides = defaults }
            #expect(
                ModelOptNVFP4NAXExperts.plan(rows: 45_056, experts: 512, input: 2688) == pinned)
            #expect(
                ModelOptNVFP4NAXExperts.plan(rows: 45_056, experts: 512, input: 1024)
                    == Self.sweep[4])
            // A pinned plan that does not admit the width falls back.
            ModelOptNVFP4NAXExperts.planOverrides = [192: pinned]
            #expect(
                ModelOptNVFP4NAXExperts.plan(rows: 4096, experts: 16, input: 192)
                    == Plan.plan(rows: 4096, experts: 16, input: 192))
        }

        @Test func unsupportedGeometryAndDTypesDecline() {
            func supports(
                rows: Int = 45_056, inputs: Int = 1024, outputs: Int = 2688, experts: Int = 512,
                dtype: DType = .bfloat16
            ) -> Bool {
                ModelOptNVFP4NAXExperts.supports(
                    rows: rows, inputs: inputs, outputs: outputs, experts: experts, dtype: dtype)
            }
            #expect(supports() == naxExpertLane)
            #expect(!supports(dtype: .float16))
            #expect(!supports(dtype: .float32))
            #expect(!supports(inputs: 1000))
            #expect(!supports(outputs: 2600))
            #expect(!supports(experts: 1025))
            #expect(!supports(rows: 8))
            #expect(!supports(rows: ModelOptNVFP4SwitchLinear.maximumGraphRows + 1))
        }

        /// Super's fc1 (relu2) and fc2 at a 2,048-token chunk with log-normal
        /// routing skew, then small banks with heavier skew (empty and one-row
        /// experts) and the bare sums.
        @Test(
            .enabled(if: naxExpertLane),
            arguments: [
                (45_056, 1024, 2688, 512, true, true, 1.0),
                (45_056, 2688, 1024, 512, false, true, 1.0),
                (90_112, 1024, 2688, 512, true, true, 1.0),
                (600, 256, 128, 16, true, true, 2.0),
                (600, 256, 128, 16, false, false, 2.0),
                (4_100, 256, 64, 1024, false, true, 0.5),
            ])
        func matchesMLXSortedGather(
            rows: Int, k: Int, n: Int, experts: Int, relu2: Bool, fused: Bool, skew: Double
        ) throws {
            let result = try ModelOptNVFP4NAXExperts.compareWithMLX(
                rows: rows, k: k, n: n, experts: experts, seed: UInt64(rows + k),
                squaredReLU: relu2, fused: fused, skew: skew, iterations: 3)
            print(
                "NAX NVFP4 experts rows \(rows) k \(k) n \(n) E \(experts) relu2 \(relu2) "
                    + "fused \(fused): maxAbs \(result.maxAbs) maxRel \(result.maxRel) "
                    + "nax \(result.seconds.nax * 1e3) ms mlx \(result.seconds.mlx * 1e3) ms")
            #expect(result.maxRel.isFinite && result.maxRel <= 0x1p-7)
        }

        /// Every plan of the sweep on uneven, partly empty expert runs.
        @Test(.enabled(if: naxExpertLane), arguments: ModelOptNVFP4NAXExpertsTests.sweep)
        func everyPlanMatchesMLX(plan: ModelOptNVFP4NAXExperts.TilePlan) throws {
            let result = try ModelOptNVFP4NAXExperts.compareWithMLX(
                rows: 3_000, k: 1024, n: 256, experts: 24, seed: 7, squaredReLU: true,
                skew: 1.5, plan: plan, iterations: 1)
            #expect(result.maxRel.isFinite && result.maxRel <= 0x1p-7, "\(plan)")
        }

        /// The fused route (route kernel, activation through the order, fc2's
        /// rounding in fc1's epilogue or as its own pass, combine through the
        /// inverse) against gatherSort + both graphForward + scatterUnsort +
        /// the weighted sum: bitwise, NaN positions equal. Super's chunk,
        /// a NaN token, 4,096 tokens, and a small bank with heavy skew.
        @Test(
            .enabled(if: naxExpertLane),
            arguments: [
                (2048, 512, 1.0, false), (2048, 512, 1.0, true), (4096, 512, 1.0, false),
                (100, 24, 3.0, false),
            ])
        func fusedRouteMatchesTheSortedRoute(
            tokens: Int, experts: Int, skew: Double, nanToken: Bool
        ) throws {
            let result = try ModelOptNVFP4NAXExperts.compareRoute(
                tokens: tokens, experts: experts, seed: UInt64(tokens + experts), skew: skew,
                nanToken: nanToken, iterations: 3)
            print("tokens \(tokens) E \(experts) skew \(skew) nan \(nanToken): \(result)")
            #expect(result.maxAbs.epilogue == 0 && result.bitMismatches.epilogue == 0)
            #expect(result.maxAbs.separate == 0 && result.bitMismatches.separate == 0)
        }

        @Test func fusedRouteIsOptIn() throws {
            var fixtures = ModelOptGraphFixtures(seed: 21)
            let fc1 = try fixtures.expertBank(
                experts: 8, outputs: 128, inputs: 64, magnitude: 3, dtype: .bfloat16)
            let fc2 = try fixtures.expertBank(
                experts: 8, outputs: 64, inputs: 128, magnitude: 3, dtype: .bfloat16)
            let fused = ModelOptNVFP4NAXExperts.fusedRoute
            defer { ModelOptNVFP4NAXExperts.fusedRoute = fused }
            ModelOptNVFP4NAXExperts.fusedRoute = false
            #expect(
                !ModelOptNVFP4NAXExperts.admitsRoute(
                    fc1: fc1, fc2: fc2, tokens: 64, topK: 4, dtype: .bfloat16))
            ModelOptNVFP4NAXExperts.fusedRoute = true
            #expect(
                ModelOptNVFP4NAXExperts.admitsRoute(
                    fc1: fc1, fc2: fc2, tokens: 64, topK: 4, dtype: .bfloat16)
                    == (naxExpertLane && ModelOptNVFP4NAXExperts.isEnabled))
            // Below graphForward's tiled threshold (4 rows per expert), and
            // a top-k past the combine's small strided reduction.
            #expect(
                !ModelOptNVFP4NAXExperts.admitsRoute(
                    fc1: fc1, fc2: fc2, tokens: 7, topK: 4, dtype: .bfloat16))
            #expect(
                !ModelOptNVFP4NAXExperts.admitsRoute(
                    fc1: fc1, fc2: fc2, tokens: 64, topK: 32, dtype: .bfloat16))
            #expect(
                !ModelOptNVFP4NAXExperts.admitsRoute(
                    fc1: fc1, fc2: fc2, tokens: 64, topK: 4, dtype: .float16))
        }

        /// The tiled graph route takes the kernel when enabled: the same bits
        /// as a direct product, within the tiled bound of MLX's route.
        @Test(.enabled(if: naxExpertLane), arguments: [false, true])
        func graphForwardHookUsesTheKernel(squaredReLU: Bool) throws {
            var fixtures = ModelOptGraphFixtures(seed: squaredReLU ? 11 : 12)
            let bank = try fixtures.expertBank(
                experts: 16, outputs: 128, inputs: 256, magnitude: 3, dtype: .bfloat16)
            let counts = [0, 1, 7, 33, 64, 0, 95, 2, 31, 32, 140, 5, 0, 66, 17, 120]
            let ids = counts.enumerated().flatMap {
                Array(repeating: Int32($0.offset), count: $0.element)
            }
            let routed = MLXArray(ids)
            let rows = MLXArray(fixtures.values(ids.count * 256, magnitude: 3))
                .reshaped(ids.count, 1, 256).asType(.bfloat16)

            let enabled = ModelOptNVFP4NAXExperts.isEnabled
            let grouped = ModelOptNVFP4GroupedGEMM.isEnabled
            defer {
                ModelOptNVFP4NAXExperts.isEnabled = enabled
                ModelOptNVFP4GroupedGEMM.isEnabled = grouped
            }
            ModelOptNVFP4GroupedGEMM.isEnabled = false
            ModelOptNVFP4NAXExperts.isEnabled = false
            let reference = try bank.graphForward(
                rows, routed, sortedIndices: true, squaredReLU: squaredReLU)
            ModelOptNVFP4NAXExperts.isEnabled = true
            let hooked = try bank.graphForward(
                rows, routed, sortedIndices: true, squaredReLU: squaredReLU)
            #expect(hooked.validIndices.item(Bool.self))

            let quantized = ModelOptActivationKernels.nvfp4Routed(
                rows.reshaped(ids.count, 256),
                sources: MLXArray((0 ..< ids.count).map { Int32($0) }), experts: routed,
                globalScale: bank.routeTables.globalScale,
                overflowDivisor: bank.routeTables.overflowDivisor,
                inverseWeightScale: bank.routeTables.inverseWeightScale,
                activationDType: .bfloat16)
            let direct = ModelOptNVFP4NAXExperts.product(
                quantized.activation, weight: bank.weight, scales: bank.blockScales,
                experts: quantized.safeExperts,
                expertScale: bank.routeTables.inverseWeightScale, squaredReLU: squaredReLU)
            #expect(modelOptBitwiseEqual(hooked.value.reshaped(direct.shape), direct))

            let a = hooked.value.asType(.float32)
            let b = reference.value.asType(.float32)
            #expect(arrayEqual(isNaN(a), isNaN(b)).item(Bool.self))
            let rowMax = abs(b).max(axis: -1, keepDims: true)
            #expect(!((abs(a - b) - rowMax * 0x1p-7) .> 0).any().item(Bool.self))
        }
    }
}
