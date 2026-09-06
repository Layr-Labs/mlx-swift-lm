import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class ZZScratchMoEKernelTests: XCTestCase {
    func testRouterAndTailVsChain() throws {
        let (E, k, D, rows) = (512, 10, 2560, 1)
        MLXRandom.seed(4)
        let logits = MLXRandom.normal([1, rows, E]) * 3
        let experts = MLXRandom.normal([1, rows, k, D]).asType(.bfloat16)
        let gate = MLXRandom.normal([1, rows, 1]).asType(.bfloat16)
        let shared = MLXRandom.normal([1, rows, D]).asType(.bfloat16)
        eval(logits, experts, gate, shared)
        // reference
        let idx = argPartition(-logits, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let w = MLX.softmax(takeAlong(logits, idx, axis: -1), axis: -1, precise: true)
        let r = Qwen4ExpMoEScratchKernels.router(logits: logits, topK: k)
        eval(idx, w, r.indices, r.weights)
        let refSet = Set(idx.asArray(Int32.self)), gotSet = Set(r.indices.asArray(UInt32.self).map { Int32($0) })
        XCTAssertEqual(refSet, gotSet, "router picks the same experts")
        let refW = Dictionary(uniqueKeysWithValues: zip(idx.asArray(Int32.self), w.asArray(Float.self)))
        for (i, wt) in zip(r.indices.asArray(UInt32.self), r.weights.asArray(Float.self)) {
            XCTAssertEqual(Double(refW[Int32(i)]!), Double(wt), accuracy: 1e-5)
        }
        // tail: experts are gathered in the SAME order as indices, so build the reference with the kernel's order
        let wk = r.weights
        let ref = (experts * wk[.ellipsis, .newAxis].asType(.bfloat16)).sum(axis: -2).asType(.bfloat16) + sigmoid(gate) * shared
        let got = Qwen4ExpMoEScratchKernels.tail(experts: experts, weights: wk, gate: gate, shared: shared)
        eval(ref, got)
        let delta = MLX.abs(ref.asType(.float32) - got.asType(.float32)).max().item(Float.self)
        print(String(format: "MOEK tail max|delta| %.4f", delta))
        XCTAssertLessThan(delta, 0.1)
        func bench(_ name: String, _ reps: Int, _ body: () -> [MLXArray]) {
            for _ in 0 ..< 5 { eval(body()) }
            var t: [Double] = []
            for _ in 0 ..< 30 {
                let s = DispatchTime.now().uptimeNanoseconds
                var outs: [MLXArray] = []
                for _ in 0 ..< reps { outs += body() }
                eval(outs)
                t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
            }
            t.sort(); print(String(format: "MOEK %-40@ x%d min %.3f ms (%.1f us per rep)", name, reps, t[0], t[0] / Double(reps) * 1000))
        }
        for reps in [1, 20] {
            bench("chain: router (neg+argPartition+take+softmax)", reps) {
                let i = argPartition(-logits, kth: k - 1, axis: -1)[.ellipsis, ..<k]
                return [i, MLX.softmax(takeAlong(logits, i, axis: -1), axis: -1, precise: true)]
            }
            bench("kernel: router", reps) { let r = Qwen4ExpMoEScratchKernels.router(logits: logits, topK: k); return [r.indices, r.weights] }
            bench("chain: tail (mul+sum+cast+sigmoid+mul+add)", reps) {
                [(experts * wk[.ellipsis, .newAxis].asType(.bfloat16)).sum(axis: -2).asType(.bfloat16) + sigmoid(gate) * shared]
            }
            bench("kernel: tail", reps) { [Qwen4ExpMoEScratchKernels.tail(experts: experts, weights: wk, gate: gate, shared: shared)] }
        }
    }
}
