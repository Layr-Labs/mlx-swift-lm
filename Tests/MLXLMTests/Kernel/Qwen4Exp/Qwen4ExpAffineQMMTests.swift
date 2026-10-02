import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the Qwen4 affine quantized prefill tile in
    /// `Qwen4ExpAffineQMM.swift` (`tryMatmul`, `apply`, `applyMLP`).
    ///
    /// Each case compares the kernel output with a float32 reference: the
    /// packed weight dequantized to float32, times the input in float32.
    ///
    /// Tolerance: `2e-2 + 1e-2 * |reference|`. The tile multiplies bfloat16
    /// operands with a float32 accumulator and rounds the output once to
    /// bfloat16 (relative step 2^-8). The outputs have a standard deviation
    /// near 1.
    @Suite(.serialized)
    struct Qwen4ExpAffineQMMTests {
        typealias Support = Qwen4ExpKernelSupport

        static let atol: Float = 2e-2
        static let rtol: Float = 1e-2

        static func layer(
            outputs: Int, inputs: Int, bits: Int, groupSize: Int, dtype: DType = .bfloat16,
            seed: UInt64
        ) -> QuantizedLinear {
            let weight =
                (MLXRandom.normal([outputs, inputs], key: MLXRandom.key(seed))
                * Float(1 / Double(inputs).squareRoot())).asType(dtype)
            let layer = QuantizedLinear(
                weight: weight, bias: nil, groupSize: groupSize, bits: bits)
            eval(layer)
            return layer
        }

        static func reference(_ x: MLXArray, _ layer: QuantizedLinear) -> MLXArray {
            let w = Support.dequantizedFloat32(
                layer.weight, scales: layer.scales, biases: layer.biases!, bits: layer.bits,
                groupSize: layer.groupSize)
            return matmul(x.asType(.float32), w.T)
        }

        struct Case: CustomStringConvertible, Sendable {
            let bits: Int
            let groupSize: Int
            let tokens: Int
            let inputs: Int
            let outputs: Int
            let environment: [String: String]
            /// Tile rows (BM) and tile columns (BN) the dispatcher selects.
            let blockM: Int
            let blockN: Int?

            var description: String {
                "bits=\(bits) gs=\(groupSize) M=\(tokens) K=\(inputs) N=\(outputs) env=\(environment)"
            }
        }

        /// One case for each bit width, both group sizes, both column tiles
        /// (BN=64 when N % 64 == 0, else BN=32), a row count that leaves a
        /// partial last tile (M=2050), and the two 8-bit floors.
        static let cases: [Case] = [
            Case(
                bits: 4, groupSize: 64, tokens: 2050, inputs: 128, outputs: 64,
                environment: [:], blockM: 64, blockN: 64),
            Case(
                bits: 4, groupSize: 128, tokens: 2048, inputs: 128, outputs: 96,
                environment: [:], blockM: 64, blockN: 32),
            Case(
                bits: 5, groupSize: 64, tokens: 2048, inputs: 64, outputs: 64,
                environment: [:], blockM: 64, blockN: 64),
            Case(
                bits: 6, groupSize: 64, tokens: 2048, inputs: 128, outputs: 32,
                environment: [:], blockM: 64, blockN: 32),
            Case(
                bits: 8, groupSize: 64, tokens: 2048, inputs: 64, outputs: 64,
                environment: [Qwen4ExpAffineQMM.q8MinTokensEnv: "2048"], blockM: 64,
                blockN: 64),
            Case(
                bits: 8, groupSize: 128, tokens: 8192, inputs: 128, outputs: 64,
                environment: [Qwen4ExpAffineQMM.q8BlockMEnv: "128"], blockM: 128,
                blockN: 64),
        ]

        @Test(arguments: cases)
        func prefillTileMatchesDequantizedReference(_ c: Case) throws {
            #expect(Qwen4ExpAffineQMM.tileM(bits: c.bits, environment: c.environment) == c.blockM)
            #expect(Qwen4ExpAffineQMM.blockN(for: c.outputs) == c.blockN)
            #expect(
                Qwen4ExpAffineQMM.matchesGeometry(
                    tokens: c.tokens, inputDim: c.inputs, outputDim: c.outputs, bits: c.bits,
                    groupSize: c.groupSize, environment: c.environment))
            let layer = Self.layer(
                outputs: c.outputs, inputs: c.inputs, bits: c.bits, groupSize: c.groupSize,
                seed: 5100)
            let x = Support.input([c.tokens, c.inputs], seed: 5101)
            let y = try #require(
                Qwen4ExpAffineQMM.tryMatmul(x, layer, environment: c.environment), "\(c)")
            let expected = Self.reference(x, layer)
            eval(y, expected)
            #expect(y.shape == [c.tokens, c.outputs])
            #expect(y.dtype == .bfloat16)
            #expect(
                Support.isClose(y, expected, atol: Self.atol, rtol: Self.rtol),
                "\(c): max difference \(Support.maxAbsDifference(y, expected))")
        }

        /// The HC inject bank has N=4. The dispatcher pads it to 32 rows,
        /// runs the BN=32 tile and slices the first 4 columns. The kill
        /// switch of the padding makes the shape ineligible.
        @Test func injectBankIsPaddedAndSliced() throws {
            let layer = Self.layer(outputs: 4, inputs: 64, bits: 5, groupSize: 64, seed: 5110)
            let x = Support.input([1, 2048, 64], seed: 5111)
            let y = try #require(Qwen4ExpAffineQMM.tryMatmul(x, layer, environment: [:]))
            let expected = Self.reference(x, layer)
            eval(y, expected)
            #expect(y.shape == [1, 2048, 4])
            #expect(
                Support.isClose(y, expected, atol: Self.atol, rtol: Self.rtol),
                "max difference \(Support.maxAbsDifference(y, expected))")

            let bank = try #require(
                Qwen4ExpAffineQMM.paddedInjectBank(layer, compute: .bfloat16, targetRows: 32))
            #expect(bank.weight.shape == [32, layer.weight.dim(1)])
            #expect(bank.scales.shape == [32, 1])
            #expect(Support.isEqual(bank.weight[0 ..< 4], layer.weight))
            #expect(
                Support.isNil(
                    Qwen4ExpAffineQMM.tryMatmul(
                        x, layer, environment: [Qwen4ExpAffineQMM.injectPadEnvFlag: "0"])))
            #expect(
                Support.isNil(
                    Qwen4ExpAffineQMM.paddedInjectBank(layer, compute: .bfloat16, targetRows: 4)),
                "no padding when the bank is not narrower than the target")
        }

        /// Float16 scales select float16 compute.
        @Test func float16LayerComputesInFloat16() throws {
            let layer = Self.layer(
                outputs: 64, inputs: 64, bits: 4, groupSize: 64, dtype: .float16, seed: 5120)
            let x = Support.input([2048, 64], dtype: .float16, seed: 5121)
            let y = try #require(Qwen4ExpAffineQMM.tryMatmul(x, layer, environment: [:]))
            let expected = Self.reference(x, layer)
            #expect(y.dtype == .float16)
            #expect(
                Support.isClose(y, expected, atol: Self.atol, rtol: Self.rtol),
                "max difference \(Support.maxAbsDifference(y, expected))")
        }

        /// Below the floor, a decode width goes to the QMV kernel, and a
        /// width between 17 and the floor returns nil. Layers that are not
        /// a plain affine `QuantizedLinear` without bias return nil.
        @Test func routingAndMisses() throws {
            let layer = Self.layer(outputs: 64, inputs: 64, bits: 4, groupSize: 64, seed: 5130)
            let decode = Support.input([3, 64], seed: 5131)
            let viaQMV = try #require(
                Qwen4ExpAffineQMM.tryMatmul(decode, layer, environment: [:]))
            #expect(
                Support.isClose(
                    viaQMV, Self.reference(decode, layer), atol: Self.atol, rtol: Self.rtol))

            #expect(
                Support.isNil(
                    Qwen4ExpAffineQMM.tryMatmul(
                        Support.input([100, 64], seed: 5132), layer, environment: [:])))
            #expect(
                Support.isNil(
                    Qwen4ExpAffineQMM.tryMatmul(
                        Support.input([2048, 64], seed: 5133), layer,
                        environment: [Qwen4ExpAffineQMM.envFlag: "0"])))

            let w8 = Self.layer(outputs: 64, inputs: 64, bits: 8, groupSize: 64, seed: 5134)
            #expect(
                Support.isNil(
                    Qwen4ExpAffineQMM.tryMatmul(
                        Support.input([2048, 64], seed: 5135), w8, environment: [:])),
                "8 bits stays below its 16384 floor")

            let before = Qwen4ExpAffineQMMInvocation.snapshot()
            let plain = Linear(64, 64, bias: false)
            #expect(Support.isNil(Qwen4ExpAffineQMM.tryMatmul(decode, plain, environment: [:])))
            let withBias = QuantizedLinear(
                weight: MLXRandom.normal([64, 64], key: MLXRandom.key(5136)).asType(.bfloat16),
                bias: MLXArray.zeros([64], dtype: .bfloat16), groupSize: 64, bits: 4)
            #expect(Support.isNil(Qwen4ExpAffineQMM.tryMatmul(decode, withBias, environment: [:])))
            let float32Layer = Self.layer(
                outputs: 64, inputs: 64, bits: 4, groupSize: 64, dtype: .float32, seed: 5137)
            #expect(
                Support.isNil(
                    Qwen4ExpAffineQMM.tryMatmul(
                        Support.input([2048, 64], seed: 5138), float32Layer, environment: [:])))
            let after = Qwen4ExpAffineQMMInvocation.snapshot()
            #expect(after.missType == before.missType + 2)
            #expect(after.missDtype == before.missDtype + 1)
            #expect(after.line.hasPrefix("affineQmm native="))
        }

        /// `apply` uses the native result when it exists and the stock
        /// layer otherwise. A stock result is kept in bfloat16.
        @Test func applyFallsBackToTheStockLayer() {
            let layer = Self.layer(outputs: 64, inputs: 64, bits: 4, groupSize: 64, seed: 5140)
            let x = Support.input([2048, 64], seed: 5141)
            let before = Qwen4ExpAffineQMMInvocation.snapshot()
            let native = Qwen4ExpAffineQMM.apply(layer, x, environment: [:])
            let stock = Qwen4ExpAffineQMM.apply(
                layer, x, environment: [Qwen4ExpAffineQMM.envFlag: "0"])
            eval(native, stock)
            let after = Qwen4ExpAffineQMMInvocation.snapshot()
            #expect(after.native == before.native + 1)
            #expect(after.fallback == before.fallback + 1)
            #expect(stock.dtype == .bfloat16)
            #expect(
                Support.isClose(native, stock, atol: Self.atol, rtol: Self.rtol),
                "max difference \(Support.maxAbsDifference(native, stock))")

            let plain = Linear(64, 64, bias: false)
            let float32Input = MLXRandom.normal([2, 64], key: MLXRandom.key(5142))
            let kept = Qwen4ExpAffineQMM.apply(plain, float32Input, environment: [:])
            #expect(kept.dtype == .bfloat16)
            #expect(
                Support.isClose(kept, plain(float32Input), atol: 1e-2, rtol: 1e-2),
                "bfloat16 rounding of the stock float32 result")
        }

        /// `applyMLP` is `down(silu(gate(x)) * up(x))`. The reference uses
        /// the dequantized weights in float32. The tolerance is wider than
        /// for one product because the hidden activation is rounded to
        /// bfloat16 before the down projection.
        @Test func applyMLPMatchesFloat32Reference() {
            let gate = Self.layer(outputs: 128, inputs: 64, bits: 4, groupSize: 64, seed: 5150)
            let up = Self.layer(outputs: 128, inputs: 64, bits: 4, groupSize: 64, seed: 5151)
            let down = Self.layer(outputs: 64, inputs: 128, bits: 4, groupSize: 64, seed: 5152)
            let x = Support.input([1, 3, 64], seed: 5153)
            let y = Qwen4ExpAffineQMM.applyMLP(gate: gate, up: up, down: down, x: x)
            let hidden = silu(Self.reference(x, gate)) * Self.reference(x, up)
            let expected = Self.reference(hidden, down)
            #expect(y.shape == [1, 3, 64])
            #expect(
                Support.isClose(y, expected, atol: 5e-2, rtol: 2e-2),
                "max difference \(Support.maxAbsDifference(y, expected))")
        }

        @Test func floorsAndTileSelection() {
            #expect(Qwen4ExpAffineQMM.q8Floor(environment: [:]) == 16384)
            #expect(
                Qwen4ExpAffineQMM.q8Floor(environment: [Qwen4ExpAffineQMM.q8BlockMEnv: "128"])
                    == 8192)
            #expect(
                Qwen4ExpAffineQMM.q8Floor(environment: [Qwen4ExpAffineQMM.q8MinTokensEnv: " 4096 "])
                    == 4096)
            #expect(
                Qwen4ExpAffineQMM.q8Floor(environment: [Qwen4ExpAffineQMM.q8MinTokensEnv: "0"])
                    == 16384, "a floor that is not positive is ignored")
            #expect(
                Qwen4ExpAffineQMM.q8TileM(environment: [Qwen4ExpAffineQMM.q8BlockMEnv: "96"]) == 64)
            #expect(Qwen4ExpAffineQMM.shouldPad(outputDim: 4, environment: [:]))
            #expect(!Qwen4ExpAffineQMM.shouldPad(outputDim: 8, environment: [:]))
            func matches(
                tokens: Int = 2048, inputDim: Int = 128, outputDim: Int = 64, bits: Int = 4,
                groupSize: Int = 64
            ) -> Bool {
                Qwen4ExpAffineQMM.matchesGeometry(
                    tokens: tokens, inputDim: inputDim, outputDim: outputDim, bits: bits,
                    groupSize: groupSize, environment: [:])
            }
            #expect(matches())
            #expect(!matches(tokens: 2047))
            #expect(!matches(bits: 3))
            #expect(!matches(groupSize: 32))
            #expect(!matches(inputDim: 0))
            #expect(!matches(inputDim: 96))
            #expect(!matches(outputDim: 48))
        }
    }
}
