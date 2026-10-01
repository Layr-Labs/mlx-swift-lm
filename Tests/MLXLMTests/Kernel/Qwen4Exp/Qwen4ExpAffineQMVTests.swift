import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the Qwen4 affine quantized matrix-vector kernels in
    /// `Qwen4ExpAffineQMV.swift`: `tryMatmul` (dense decode and verify) and
    /// `tryGather` (expert decode).
    ///
    /// Each case compares the kernel output with a float32 reference: the
    /// packed weight dequantized to float32, times the input in float32.
    ///
    /// Tolerance: `1e-2 + 1e-2 * |reference|`. The kernel keeps a float32
    /// accumulator and rounds the output once to bfloat16 (relative step
    /// 2^-8). The outputs have a standard deviation near 1, so one rounding
    /// step is below 1e-2.
    @Suite(.serialized)
    struct Qwen4ExpAffineQMVTests {
        typealias Support = Qwen4ExpKernelSupport

        static let atol: Float = 1e-2
        static let rtol: Float = 1e-2

        struct Case: CustomStringConvertible, Sendable {
            let bits: Int
            let groupSize: Int
            let tokens: Int
            let inputs: Int
            let outputs: Int
            /// The value of `packsPerThread` for this shape: 2 is the fast
            /// layout, 1 is the layout for shapes that are not aligned.
            let packs: Int

            var description: String {
                "bits=\(bits) gs=\(groupSize) T=\(tokens) K=\(inputs) N=\(outputs)"
            }
        }

        /// Each bit width and group size, in two shapes:
        /// - K=512, N=16: the fast layout (two packs for each lane).
        /// - K=640, N=4: the layout with one pack for each lane. N=4 is less
        ///   than one 8-row tile, and K=640 leaves a K tail for 4 and 5 bits.
        static let cases: [Case] = {
            var cases: [Case] = []
            for bits in [4, 5, 6, 8] {
                for groupSize in [64, 128] {
                    cases.append(
                        Case(
                            bits: bits, groupSize: groupSize, tokens: 1, inputs: 512,
                            outputs: 16, packs: 2))
                    cases.append(
                        Case(
                            bits: bits, groupSize: groupSize, tokens: 3, inputs: 640,
                            outputs: 4, packs: 1))
                }
            }
            return cases
        }()

        static func reference(
            _ x: MLXArray, _ packed: (weight: MLXArray, scales: MLXArray, biases: MLXArray),
            bits: Int, groupSize: Int
        ) -> MLXArray {
            let w = Support.dequantizedFloat32(
                packed.weight, scales: packed.scales, biases: packed.biases, bits: bits,
                groupSize: groupSize)
            return matmul(x.asType(.float32), w.T)
        }

        @Test(arguments: cases)
        func denseMatchesDequantizedReference(_ c: Case) throws {
            #expect(
                Qwen4ExpAffineQMV.packsPerThread(
                    inputDim: c.inputs, outputDim: c.outputs, bits: c.bits) == c.packs)
            #expect(
                Qwen4ExpAffineQMV.matchesDecodeGeometry(
                    tokens: c.tokens, inputDim: c.inputs, outputDim: c.outputs, bits: c.bits,
                    groupSize: c.groupSize))
            let packed = Support.packed(
                [c.outputs, c.inputs], bits: c.bits, groupSize: c.groupSize, seed: 4100)
            let x = Support.input([c.tokens, c.inputs], seed: 4101)
            let before = Qwen4ExpAffineQMVInvocation.snapshot().native
            let y = try #require(
                Qwen4ExpAffineQMV.tryMatmul(
                    x: x, weight: packed.weight, scales: packed.scales, biases: packed.biases,
                    bits: c.bits, groupSize: c.groupSize, environment: [:]),
                "\(c)")
            let expected = Self.reference(x, packed, bits: c.bits, groupSize: c.groupSize)
            eval(y, expected)
            #expect(Qwen4ExpAffineQMVInvocation.snapshot().native == before + 1)
            #expect(y.shape == [c.tokens, c.outputs])
            #expect(y.dtype == .bfloat16)
            #expect(
                Support.isClose(y, expected, atol: Self.atol, rtol: Self.rtol),
                "\(c): max difference \(Support.maxAbsDifference(y, expected))")
        }

        /// A 3-D input keeps its leading axes. K=128 is shorter than one
        /// 256-value block, so only the bounded tail loop runs. A float16
        /// input is cast to the bfloat16 compute dtype.
        @Test func leadingAxesTailOnlyBlockAndFloat16Input() throws {
            let packed = Support.packed([20, 128], bits: 4, groupSize: 64, seed: 4110)
            let x = Support.input([1, 2, 128], seed: 4111)
            let y = try #require(
                Qwen4ExpAffineQMV.tryMatmul(
                    x: x, weight: packed.weight, scales: packed.scales, biases: packed.biases,
                    bits: 4, groupSize: 64, environment: [:]))
            let expected = Self.reference(x, packed, bits: 4, groupSize: 64)
            #expect(y.shape == [1, 2, 20])
            #expect(
                Support.isClose(y, expected, atol: Self.atol, rtol: Self.rtol),
                "max difference \(Support.maxAbsDifference(y, expected))")

            let half = x.asType(.float16)
            let fromHalf = try #require(
                Qwen4ExpAffineQMV.tryMatmul(
                    x: half, weight: packed.weight, scales: packed.scales,
                    biases: packed.biases, bits: 4, groupSize: 64, environment: [:]))
            #expect(fromHalf.dtype == .bfloat16)
            #expect(
                Support.isClose(fromHalf, expected, atol: Self.atol, rtol: Self.rtol),
                "max difference \(Support.maxAbsDifference(fromHalf, expected))")
        }

        /// Float16 scales select float16 compute and a float16 output.
        @Test func float16ScalesComputeInFloat16() throws {
            let packed = Support.packed(
                [16, 512], bits: 4, groupSize: 64, dtype: .float16, seed: 4120)
            #expect(packed.scales.dtype == .float16)
            let x = Support.input([2, 512], dtype: .float16, seed: 4121)
            let y = try #require(
                Qwen4ExpAffineQMV.tryMatmul(
                    x: x, weight: packed.weight, scales: packed.scales, biases: packed.biases,
                    bits: 4, groupSize: 64, environment: [:]))
            let expected = Self.reference(x, packed, bits: 4, groupSize: 64)
            #expect(y.dtype == .float16)
            #expect(
                Support.isClose(y, expected, atol: Self.atol, rtol: Self.rtol),
                "max difference \(Support.maxAbsDifference(y, expected))")
        }

        /// Shapes and dtypes outside the kernel contract return nil, so the
        /// caller keeps the stock path.
        @Test func ineligibleInputsReturnNil() {
            let packed = Support.packed([16, 512], bits: 4, groupSize: 64, seed: 4130)
            let x = Support.input([1, 512], seed: 4131)
            func run(
                _ x: MLXArray, weight: MLXArray? = nil, scales: MLXArray? = nil,
                biases: MLXArray? = nil, bits: Int = 4,
                environment: [String: String] = [:]
            ) -> MLXArray? {
                Qwen4ExpAffineQMV.tryMatmul(
                    x: x, weight: weight ?? packed.weight, scales: scales ?? packed.scales,
                    biases: biases ?? packed.biases, bits: bits, groupSize: 64,
                    environment: environment)
            }
            #expect(Support.isNil(run(x, environment: [Qwen4ExpAffineQMV.envFlag: "0"])))
            #expect(Support.isNil(run(x.asType(.float32))))
            #expect(Support.isNil(run(Support.input([17, 512], seed: 4132))))
            #expect(Support.isNil(run(Support.input([512], seed: 4133))))
            #expect(
                Support.isNil(
                    run(
                        x, scales: packed.scales.asType(.float32),
                        biases: packed.biases.asType(.float32))))
            #expect(Support.isNil(run(x, biases: packed.biases.asType(.float16))))
            #expect(Support.isNil(run(x, weight: packed.weight.expandedDimensions(axis: 0))))
            #expect(Support.isNil(run(x, bits: 8)), "packed width does not match 8 bits")
        }

        @Test func geometryAndLayoutRules() {
            #expect(Qwen4ExpAffineQMV.isEnabled(environment: [:]))
            #expect(Qwen4ExpAffineQMV.isEnabled(environment: [Qwen4ExpAffineQMV.envFlag: "1"]))
            for off in ["0", "false", "no", "off", " OFF "] {
                #expect(
                    !Qwen4ExpAffineQMV.isEnabled(environment: [Qwen4ExpAffineQMV.envFlag: off]))
            }
            #expect(Qwen4ExpAffineQMV.packFactor(4) == 8)
            #expect(Qwen4ExpAffineQMV.packFactor(5) == 8)
            #expect(Qwen4ExpAffineQMV.packFactor(6) == 4)
            #expect(Qwen4ExpAffineQMV.packFactor(8) == 4)
            #expect(Qwen4ExpAffineQMV.bytesPerPack(4) == 4)
            #expect(Qwen4ExpAffineQMV.bytesPerPack(5) == 5)
            #expect(Qwen4ExpAffineQMV.bytesPerPack(6) == 3)
            #expect(Qwen4ExpAffineQMV.bytesPerPack(8) == 4)
            // The fast layout needs N % 8 == 0 and K % (packFactor * 64) == 0.
            #expect(Qwen4ExpAffineQMV.packsPerThread(inputDim: 512, outputDim: 8, bits: 4) == 2)
            #expect(Qwen4ExpAffineQMV.packsPerThread(inputDim: 256, outputDim: 8, bits: 4) == 1)
            #expect(Qwen4ExpAffineQMV.packsPerThread(inputDim: 256, outputDim: 8, bits: 8) == 2)
            #expect(Qwen4ExpAffineQMV.packsPerThread(inputDim: 512, outputDim: 12, bits: 4) == 1)

            func matches(
                tokens: Int = 1, inputDim: Int = 512, outputDim: Int = 16, bits: Int = 4,
                groupSize: Int = 64
            ) -> Bool {
                Qwen4ExpAffineQMV.matchesDecodeGeometry(
                    tokens: tokens, inputDim: inputDim, outputDim: outputDim, bits: bits,
                    groupSize: groupSize)
            }
            #expect(matches())
            #expect(matches(tokens: Qwen4ExpAffineQMV.maxTokens))
            #expect(!matches(tokens: 0))
            #expect(!matches(tokens: Qwen4ExpAffineQMV.maxTokens + 1))
            #expect(!matches(bits: 3))
            #expect(!matches(bits: 2))
            #expect(!matches(groupSize: 32))
            #expect(!matches(inputDim: 0))
            #expect(!matches(outputDim: 0))
            #expect(!matches(inputDim: 96, groupSize: 64), "K is not a multiple of the group")
            #expect(matches(outputDim: 1), "an N tail is allowed")
        }

        struct GatherCase: CustomStringConvertible, Sendable {
            let bits: Int
            let groupSize: Int
            let inputs: Int
            let outputs: Int

            var description: String { "bits=\(bits) gs=\(groupSize) K=\(inputs) N=\(outputs)" }
        }

        static let gatherCases: [GatherCase] = [
            GatherCase(bits: 4, groupSize: 64, inputs: 512, outputs: 16),
            GatherCase(bits: 4, groupSize: 128, inputs: 640, outputs: 4),
            GatherCase(bits: 5, groupSize: 64, inputs: 512, outputs: 16),
            GatherCase(bits: 6, groupSize: 64, inputs: 640, outputs: 4),
            GatherCase(bits: 8, groupSize: 128, inputs: 512, outputs: 16),
        ]

        /// Each assignment row uses the weight of its own expert. The
        /// reference takes the dequantized expert weights by index and
        /// multiplies each row in float32.
        @Test(arguments: gatherCases)
        func gatherMatchesPerExpertReference(_ c: GatherCase) throws {
            let experts = 4
            let packed = Support.packed(
                [experts, c.outputs, c.inputs], bits: c.bits, groupSize: c.groupSize,
                seed: 4200)
            let indices = MLXArray([Int32(3), 0, 2, 3, 1], [5])
            let x = Support.input([5, c.inputs], seed: 4201)
            let before = Qwen4ExpAffineQMVInvocation.snapshot().gatherNative
            let y = try #require(
                Qwen4ExpAffineQMV.tryGather(
                    x: x, indices: indices, weight: packed.weight, scales: packed.scales,
                    biases: packed.biases, bits: c.bits, groupSize: c.groupSize,
                    environment: [:]),
                "\(c)")
            let w = Support.dequantizedFloat32(
                packed.weight, scales: packed.scales, biases: packed.biases, bits: c.bits,
                groupSize: c.groupSize)
            let selected = w[indices]
            let expected = (selected * expandedDimensions(x.asType(.float32), axis: 1))
                .sum(axis: -1)
            eval(y, expected)
            #expect(Qwen4ExpAffineQMVInvocation.snapshot().gatherNative == before + 1)
            #expect(y.shape == [5, c.outputs])
            #expect(
                Support.isClose(y, expected, atol: Self.atol, rtol: Self.rtol),
                "\(c): max difference \(Support.maxAbsDifference(y, expected))")
        }

        /// Unsigned indices and a 3-D input with leading axes.
        @Test func gatherKeepsLeadingAxesWithUnsignedIndices() throws {
            let packed = Support.packed([3, 16, 512], bits: 4, groupSize: 64, seed: 4210)
            let indices = MLXArray([UInt32(2), 1], [2, 1])
            let x = Support.input([2, 1, 512], seed: 4211)
            let y = try #require(
                Qwen4ExpAffineQMV.tryGather(
                    x: x, indices: indices, weight: packed.weight, scales: packed.scales,
                    biases: packed.biases, bits: 4, groupSize: 64, environment: [:]))
            let w = Support.dequantizedFloat32(
                packed.weight, scales: packed.scales, biases: packed.biases, bits: 4,
                groupSize: 64)
            let selected = w[indices.reshaped([2]).asType(.int32)]
            let expected = (selected * x.asType(.float32)).sum(axis: -1).reshaped([2, 1, 16])
            #expect(y.shape == [2, 1, 16])
            #expect(
                Support.isClose(y, expected, atol: Self.atol, rtol: Self.rtol),
                "max difference \(Support.maxAbsDifference(y, expected))")
        }

        @Test func gatherRejectsIneligibleInputs() {
            let packed = Support.packed([2, 16, 512], bits: 4, groupSize: 64, seed: 4220)
            let indices = MLXArray([Int32(0), 1], [2])
            let x = Support.input([2, 512], seed: 4221)
            func run(
                _ x: MLXArray, indices: MLXArray, weight: MLXArray? = nil,
                scales: MLXArray? = nil, biases: MLXArray? = nil,
                environment: [String: String] = [:]
            ) -> MLXArray? {
                Qwen4ExpAffineQMV.tryGather(
                    x: x, indices: indices, weight: weight ?? packed.weight,
                    scales: scales ?? packed.scales, biases: biases ?? packed.biases, bits: 4,
                    groupSize: 64, environment: environment)
            }
            #expect(
                Support.isNil(
                    run(x, indices: indices, environment: [Qwen4ExpAffineQMV.envFlag: "off"])))
            #expect(
                Support.isNil(run(x, indices: MLXArray([Int32(0)], [1]))), "index count differs")
            #expect(Support.isNil(run(x, indices: indices, weight: packed.weight[0])))
            #expect(
                Support.isNil(
                    run(
                        x, indices: indices, scales: packed.scales.asType(.float32),
                        biases: packed.biases.asType(.float32))))
            let many = Qwen4ExpAffineQMV.maxAssignments + 1
            #expect(
                Support.isNil(
                    run(
                        Support.input([many, 512], seed: 4222),
                        indices: MLXArray.zeros([many], dtype: .int32))))
            #expect(Support.isNil(run(Support.input([512], seed: 4223), indices: indices)))
        }
    }
}
