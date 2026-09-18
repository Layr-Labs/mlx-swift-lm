import MLX
import XCTest

@testable import MLXLLM

/// Compare actual kernels with precomputed gates, isolating recurrence and
/// captured-state arithmetic from model projection or assistant behavior.
final class Qwen4GDNStackedKernelParityTests: XCTestCase {
    func testEveryCapturedStateMatchesChainedCanonicalKernelBits() throws {
        for dtype: DType in [.float16, .bfloat16] {
            for (batch, keyHeads, valueHeads, keyDim, valueDim) in [
                (1, 2, 4, 128, 128), (2, 1, 2, 64, 64)
            ] {
                for width in 1...6 {
                    func values(_ shape: [Int], _ seed: UInt64) -> MLXArray {
                        MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(dtype)
                    }
                    let q = (values([batch, width, keyHeads, keyDim], 8350) * 0.0625).asType(dtype)
                    let k = (values([batch, width, keyHeads, keyDim], 8351) * 0.0625).asType(dtype)
                    let v = values([batch, width, valueHeads, valueDim], 8352)
                    let gateShape = [batch, width, valueHeads]
                    let g = exp(-abs(MLXRandom.normal(gateShape, key: MLXRandom.key(8353))))
                    let beta = sigmoid(MLXRandom.normal(gateShape, key: MLXRandom.key(8354)))
                    let initial = MLXRandom.normal([batch, valueHeads, valueDim, keyDim],
                        key: MLXRandom.key(8355)) * 0.03125
                    let captured = try XCTUnwrap(gatedDeltaKernelStacked(
                        q: q, k: k, v: v, g: g, beta: beta, state: initial))
                    var state = initial
                    var outputs: [MLXArray] = []
                    var states: [MLXArray] = []
                    for position in 0..<width {
                        let slice = position..<position + 1
                        let (output, next) = gatedDeltaKernel(
                            q: q[0..., slice, 0..., 0...],
                            k: k[0..., slice, 0..., 0...],
                            v: v[0..., slice, 0..., 0...],
                            g: g[0..., slice, 0...], beta: beta[0..., slice, 0...],
                            state: state)
                        outputs.append(output)
                        states.append(next.expandedDimensions(axis: 1))
                        state = next
                    }
                    let expectedOutput = concatenated(outputs, axis: 1)
                    let expectedStates = concatenated(states, axis: 1)
                    eval(captured.y, captured.state, captured.stack, expectedOutput, expectedStates, state)
                    XCTAssertEqual(captured.y.dtype, dtype)
                    XCTAssertEqual(captured.state.dtype, .float32)
                    XCTAssertEqual(captured.stack.dtype, .float32)
                    XCTAssertEqual(captured.y.asData().data, expectedOutput.asData().data,
                        "dtype=\(dtype), B=\(batch), Dk=\(keyDim), width=\(width), output")
                    XCTAssertEqual(captured.state.asData().data, state.asData().data,
                        "dtype=\(dtype), B=\(batch), Dk=\(keyDim), width=\(width), final state")
                    XCTAssertEqual(captured.stack.asData().data, expectedStates.asData().data,
                        "dtype=\(dtype), B=\(batch), Dk=\(keyDim), width=\(width), every captured state")
                }
            }
        }
    }
}
