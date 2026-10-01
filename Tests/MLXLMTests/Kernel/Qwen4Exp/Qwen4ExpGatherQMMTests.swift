import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the Qwen4 routed-expert dispatcher in
    /// `Qwen4ExpGatherQMM.swift`.
    ///
    /// The dispatcher accepts only the Flash-Next expert bank: 512 experts,
    /// 4 bits, group size 64, bfloat16 scales. Up to 256 assignments go to
    /// the expert-indexed QMV kernel. 2048 or more sorted assignments with
    /// one of three projection shapes go to the expert-tile kernel. All
    /// other inputs return nil, so the caller keeps `gatherQuantizedMM`.
    @Suite(.serialized)
    struct Qwen4ExpGatherQMMTests {
        typealias Support = Qwen4ExpKernelSupport

        static let experts = Qwen4ExpGatherQMM.expertCount

        static func call(
            _ x: MLXArray, _ indices: MLXArray,
            _ packed: (weight: MLXArray, scales: MLXArray, biases: MLXArray),
            sorted: Bool, bits: Int = 4, groupSize: Int = 64
        ) -> MLXArray? {
            Qwen4ExpGatherQMM.tryMatmul(
                x: x, indices: indices, weight: packed.weight, scales: packed.scales,
                affineBiases: packed.biases, sorted: sorted, bits: bits, groupSize: groupSize,
                mode: .affine)
        }

        /// Decode: 10 unsorted assignments over a 512-expert bank go to the
        /// expert-indexed QMV kernel.
        ///
        /// Tolerance: `1e-2 + 1e-2 * |reference|`, one bfloat16 rounding of
        /// outputs near 1 (see `Qwen4ExpAffineQMVTests`).
        @Test func decodeAssignmentsUseTheIndexedQMVKernel() throws {
            let packed = Support.packed(
                [Self.experts, 16, 128], bits: 4, groupSize: 64, seed: 6100)
            let indices = MLXArray([Int32(511), 3, 3, 0, 200, 17, 64, 511, 1, 300], [10])
            let x = Support.input([10, 1, 128], seed: 6101)
            let before = Qwen4ExpGatherQMMInvocation.snapshot()
            let y = try #require(Self.call(x, indices, packed, sorted: false))
            let w = Support.dequantizedFloat32(
                packed.weight, scales: packed.scales, biases: packed.biases, bits: 4,
                groupSize: 64)
            let expected = (w[indices] * x.asType(.float32)).sum(axis: -1)
                .reshaped([10, 1, 16])
            eval(y, expected)
            #expect(Qwen4ExpGatherQMMInvocation.snapshot().native == before.native + 1)
            #expect(y.shape == [10, 1, 16])
            #expect(y.dtype == .bfloat16)
            #expect(
                Support.isClose(y, expected, atol: 1e-2, rtol: 1e-2),
                "max difference \(Support.maxAbsDifference(y, expected))")
        }

        /// The misses are counted by kind and return nil.
        @Test func ineligibleInputsAreCountedAndReturnNil() {
            let packed = Support.packed(
                [Self.experts, 16, 96], bits: 4, groupSize: 32, seed: 6110)
            let indices = MLXArray([Int32(1), 2], [2])
            let x = Support.input([2, 1, 96], seed: 6111)
            let before = Qwen4ExpGatherQMMInvocation.snapshot()

            // Group size 32 is not the Flash-Next bank.
            #expect(Support.isNil(Self.call(x, indices, packed, sorted: false, groupSize: 32)))
            // A bank without 512 experts is not checked further.
            let small = Support.packed([4, 16, 128], bits: 4, groupSize: 64, seed: 6112)
            #expect(
                Support.isNil(
                    Self.call(Support.input([2, 1, 128], seed: 6113), indices, small, sorted: true))
            )

            let bank = Support.packed(
                [Self.experts, 16, 128], bits: 4, groupSize: 64, seed: 6114)
            // Float16 biases are not the Flash-Next bank.
            let odd = (
                weight: bank.weight, scales: bank.scales, biases: bank.biases.asType(.float16)
            )
            #expect(
                Support.isNil(
                    Self.call(Support.input([2, 1, 128], seed: 6115), indices, odd, sorted: false)))
            // Unsorted rows that miss the QMV kernel stay on the stock path.
            let wide = Support.input([300, 1, 128], seed: 6116)
            let wideIndices = MLXArray.zeros([300], dtype: .int32)
            #expect(Support.isNil(Self.call(wide, wideIndices, bank, sorted: false)))
            // Sorted rows below 2048 miss the tile geometry.
            #expect(Support.isNil(Self.call(wide, wideIndices, bank, sorted: true)))

            let after = Qwen4ExpGatherQMMInvocation.snapshot()
            #expect(after.missDtype == before.missDtype + 2)
            #expect(after.missSorted == before.missSorted + 1)
            #expect(after.missGeometry == before.missGeometry + 1)
            #expect(after.native == before.native)
            #expect(after.line.hasPrefix("gatherQmm native="))
        }

        @Test func geometryRules() {
            #expect(Qwen4ExpGatherQMM.isEnabled(environment: [:]))
            #expect(!Qwen4ExpGatherQMM.isEnabled(environment: [Qwen4ExpGatherQMM.envFlag: "no"]))
            func matches(
                assignments: Int = 2048, inputDim: Int = 2560, outputDim: Int = 640,
                experts: Int = 512
            ) -> Bool {
                Qwen4ExpGatherQMM.matchesGeometry(
                    assignments: assignments, inputDim: inputDim, outputDim: outputDim,
                    experts: experts)
            }
            #expect(matches())
            #expect(matches(outputDim: 1280), "fused gate and up")
            #expect(matches(inputDim: 640, outputDim: 2560), "down")
            #expect(!matches(experts: 256))
            #expect(!matches(assignments: 2047))
            #expect(!matches(inputDim: 0))
            #expect(!matches(inputDim: 2592), "K is not a multiple of 64")
            #expect(!matches(outputDim: 650), "N is not a multiple of 32")
            #expect(!matches(inputDim: 1280, outputDim: 640), "not a Flash-Next shape")
        }

        /// Prefill: 2048 sorted assignments in the Flash-Next down shape
        /// (K=640, N=2560) go to the expert-tile kernel. The bank holds all
        /// 512 experts (420 MB of packed weight). The rows are in 64 expert
        /// segments of 40 or 24 rows, so the kernel runs full 32-row tiles,
        /// partial tiles of more than 16 rows and tiles of 8 rows.
        ///
        /// The reference is the stock `gatherQuantizedMM` with sorted
        /// indices. Both round the output to bfloat16 once, from a float32
        /// accumulator. The outputs have a standard deviation near 2, so
        /// the tolerance is `5e-2 + 2e-2 * |reference|` (two rounding steps
        /// and the order of the sums).
        @Test func sortedPrefillUsesTheExpertTileKernel() throws {
            let inputs = 640
            let outputs = 2560
            let packedInputs = inputs * 4 / 32
            // Unique random packed rows for 8 experts, repeated to fill the
            // bank. The scales differ for each of the 512 experts.
            let base = MLXRandom.randInt(
                0 ..< Int32(1 << 30), [8, outputs, packedInputs], key: MLXRandom.key(6120)
            ).asType(.uint32)
            let weight = tiled(base, repetitions: [Self.experts / 8, 1, 1])
            let scales =
                (abs(
                    MLXRandom.normal(
                        [Self.experts, outputs, inputs / 64], key: MLXRandom.key(6121)))
                * Float(0.004) + Float(0.008)).asType(.bfloat16)
            let biases = (scales.asType(.float32) * Float(-8)).asType(.bfloat16)
            eval(weight, scales, biases)

            var rows: [Int32] = []
            for segment in 0 ..< 64 {
                let count = segment.isMultiple(of: 2) ? 40 : 24
                rows += Array(repeating: Int32(segment * 8), count: count)
            }
            #expect(rows.count == 2048)
            let indices = MLXArray(rows, [2048])
            let x = Support.input([2048, 1, inputs], seed: 6122)

            let before = Qwen4ExpGatherQMMInvocation.snapshot()
            let y = try #require(
                Self.call(x, indices, (weight, scales, biases), sorted: true))
            let expected = gatherQuantizedMM(
                x, weight, scales: scales, biases: biases, rhsIndices: indices,
                transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: true)
            eval(y, expected)
            #expect(Qwen4ExpGatherQMMInvocation.snapshot().native == before.native + 1)
            #expect(y.shape == [2048, 1, outputs])
            #expect(y.dtype == .bfloat16)
            #expect(
                Support.isClose(y, expected, atol: 5e-2, rtol: 2e-2),
                "max difference \(Support.maxAbsDifference(y, expected))")
        }
    }
}
