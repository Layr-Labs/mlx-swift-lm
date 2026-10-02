import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Tests of the blocked gated-delta prefill kernel in
    /// `Qwen4ExpGDNBlocked.swift` (`gatedDeltaBlockedSeq`).
    ///
    /// The reference is `gatedDeltaOps`, the step-by-step recurrence in
    /// MLX operations, with float32 inputs and a float32 state.
    @Suite(.serialized)
    struct Qwen4ExpGDNBlockedTests {
        typealias Support = Qwen4ExpKernelSupport

        struct Case: CustomStringConvertible, Sendable {
            let dtype: DType
            let batch: Int
            let tokens: Int
            let valueHeadDim: Int

            var description: String {
                "dtype=\(dtype) B=\(batch) T=\(tokens) Dv=\(valueHeadDim)"
            }
        }

        /// Float32 uses 16-token blocks and bfloat16 uses 32-token blocks.
        /// T=70 leaves a partial last block for both. Dv=64 runs two
        /// 32-column value blocks. Two value heads share one key head.
        static let cases: [Case] = [
            Case(dtype: .float32, batch: 1, tokens: 70, valueHeadDim: 64),
            Case(dtype: .float32, batch: 2, tokens: 64, valueHeadDim: 32),
            Case(dtype: .bfloat16, batch: 1, tokens: 70, valueHeadDim: 64),
            Case(dtype: .float16, batch: 2, tokens: 64, valueHeadDim: 32),
        ]

        /// Output tolerance: float32 `1e-4 + 1e-4 * |reference|` (order of
        /// the 128-term sums). For a 16-bit output, one rounding step of the
        /// output dtype: `1e-2 + 1e-2 * |reference|`. The state is float32
        /// in both: `1e-4 + 1e-4 * |reference|`.
        @Test(arguments: cases)
        func blockedRecurrenceMatchesStepReference(_ c: Case) {
            let keyHeads = 1
            let valueHeads = 2
            let keyDim = 128
            func values(_ shape: [Int], _ seed: UInt64) -> MLXArray {
                MLXRandom.normal(shape, key: MLXRandom.key(seed))
            }
            let q = (values([c.batch, c.tokens, keyHeads, keyDim], 7100) * Float(0.0625))
                .asType(c.dtype)
            let k = (values([c.batch, c.tokens, keyHeads, keyDim], 7101) * Float(0.0625))
                .asType(c.dtype)
            let v = values([c.batch, c.tokens, valueHeads, c.valueHeadDim], 7102)
                .asType(c.dtype)
            let gateShape = [c.batch, c.tokens, valueHeads]
            let g = exp(-abs(values(gateShape, 7103)))
            let beta = sigmoid(values(gateShape, 7104))
            let state =
                values([c.batch, valueHeads, c.valueHeadDim, keyDim], 7105)
                * Float(0.03125)
            eval(q, k, v, g, beta, state)

            #expect(
                Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: c.tokens, keyHeadDim: keyDim, valueHeadDim: c.valueHeadDim,
                    hasMask: false, environment: [:]))
            let (y, finalState) = gatedDeltaBlockedSeq(
                q: q, k: k, v: v, g: g, beta: beta, state: state)
            let (expectedY, expectedState) = gatedDeltaOps(
                q: q.asType(.float32), k: k.asType(.float32), v: v.asType(.float32), g: g,
                beta: beta, state: state)
            eval(y, finalState, expectedY, expectedState)

            #expect(y.shape == [c.batch, c.tokens, valueHeads, c.valueHeadDim])
            #expect(y.dtype == c.dtype)
            #expect(finalState.shape == state.shape)
            #expect(finalState.dtype == .float32)
            let outputTolerance: Float = c.dtype == .float32 ? 1e-4 : 1e-2
            #expect(
                Support.isClose(y, expectedY, atol: outputTolerance, rtol: outputTolerance),
                "\(c): output max difference \(Support.maxAbsDifference(y, expectedY))")
            #expect(
                Support.isClose(finalState, expectedState, atol: 1e-4, rtol: 1e-4),
                "\(c): state max difference \(Support.maxAbsDifference(finalState, expectedState))")
        }

        /// A second call that starts from the state of the first call gives
        /// the same result as one call over both windows.
        @Test func stateCarriesAcrossCalls() {
            let shapeQ = [1, 128, 1, 128]
            let q = (MLXRandom.normal(shapeQ, key: MLXRandom.key(7110)) * Float(0.0625))
            let k = (MLXRandom.normal(shapeQ, key: MLXRandom.key(7111)) * Float(0.0625))
            let v = MLXRandom.normal([1, 128, 2, 32], key: MLXRandom.key(7112))
            let g = exp(-abs(MLXRandom.normal([1, 128, 2], key: MLXRandom.key(7113))))
            let beta = sigmoid(MLXRandom.normal([1, 128, 2], key: MLXRandom.key(7114)))
            let zero = MLXArray.zeros([1, 2, 32, 128], dtype: .float32)
            let (whole, wholeState) = gatedDeltaBlockedSeq(
                q: q, k: k, v: v, g: g, beta: beta, state: zero)
            let first = 0 ..< 64
            let second = 64 ..< 128
            let (head, headState) = gatedDeltaBlockedSeq(
                q: q[0..., first], k: k[0..., first], v: v[0..., first], g: g[0..., first],
                beta: beta[0..., first], state: zero)
            let (tail, tailState) = gatedDeltaBlockedSeq(
                q: q[0..., second], k: k[0..., second], v: v[0..., second],
                g: g[0..., second], beta: beta[0..., second], state: headState)
            let joined = concatenated([head, tail], axis: 1)
            eval(whole, wholeState, joined, tailState)
            // Same kernel, same block boundaries (64 is a multiple of 16):
            // the results are equal bit for bit.
            #expect(Support.isEqual(joined, whole))
            #expect(Support.isEqual(tailState, wholeState))
        }

        @Test func dispatchRules() {
            #expect(Qwen4ExpGDNBlockedSeq.blockT(for: .float32) == 16)
            #expect(Qwen4ExpGDNBlockedSeq.blockT(for: .bfloat16) == 32)
            #expect(Qwen4ExpGDNBlockedSeq.blockT(for: .float16) == 32)
            #expect(
                !Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: 64, keyHeadDim: 128, valueHeadDim: 32, hasMask: true, environment: [:]))
            #expect(
                !Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: 63, keyHeadDim: 128, valueHeadDim: 32, hasMask: false, environment: [:]))
            #expect(
                !Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: 64, keyHeadDim: 64, valueHeadDim: 32, hasMask: false, environment: [:]))
            #expect(
                !Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: 64, keyHeadDim: 128, valueHeadDim: 48, hasMask: false, environment: [:]))
            #expect(
                !Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: 64, keyHeadDim: 128, valueHeadDim: 32, hasMask: false,
                    environment: [Qwen4ExpGDNBlockedSeq.envFlag: "false"]))
        }
    }
}
