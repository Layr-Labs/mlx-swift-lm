import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest

@testable import MLXLLM

/// Opt-in: times the PLE gate section at the 125B's shapes, op by op.
final class ZZScratchGateCostTests: XCTestCase {
    func testGateSectionOpByOp() throws {
        let hidden = 2560, hc = 4, wide = hidden * hc
        let stream = MLXRandom.normal([1, 1, wide]).asType(.bfloat16)
        let keyFlat = MLXRandom.normal([1, 1, wide]).asType(.bfloat16)
        let value = MLXRandom.normal([1, 1, hidden]).asType(.bfloat16)
        let scale = (MLXArray(Float(1), dtype: .bfloat16) + MLXRandom.normal([wide]).asType(.bfloat16))
        eval(stream, keyFlat, value, scale)
        func normQ(_ x: MLXArray) -> MLXArray {
            let g = x.reshaped([1, 1, hc, hidden])
            return MLXFast.rmsNorm(g, weight: MLXArray.mlxNone, eps: 1e-6).reshaped([1, 1, wide]) * scale
        }
        let key = keyFlat.reshaped([1, 1, hc, hidden])
        func gateOf(_ q: MLXArray) -> MLXArray {
            (key * q.reshaped([1, 1, hc, hidden])).sum(axis: -1, keepDims: true) / Foundation.sqrt(Float(hidden))
        }
        func sqrtSign(_ g: MLXArray) -> MLXArray {
            MLX.sqrt(maximum(MLX.abs(g), MLXArray(Float(1e-6)))) * MLX.sign(g)
        }
        func sig(_ g: MLXArray) -> MLXArray {
            let gated = sigmoid(g) * value[.ellipsis, .newAxis, 0...]
            return gated.reshaped([1, 1, wide])
        }
        func bench(_ name: String, _ f: () -> MLXArray) {
            for _ in 0 ..< 5 { eval(f()) }
            var t: [Double] = []
            for _ in 0 ..< 30 {
                let s = DispatchTime.now().uptimeNanoseconds
                eval(f())
                t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
            }
            t.sort()
            print(String(format: "ZZGATE %-28@ min %.3f p50 %.3f ms", name, t[0], t[15]))
        }
        let q = normQ(stream); eval(q)
        let g = gateOf(q); eval(g)
        let ss = sqrtSign(g); eval(ss)
        bench("normQuery(stream)") { normQ(stream) }
        bench("dot+sum") { gateOf(q) }
        bench("sqrt/max/abs/sign") { sqrtSign(g) }
        bench("sigmoid*value") { sig(ss) }
        bench("whole gate section") { sig(sqrtSign(gateOf(normQ(stream)))) }
        bench("whole, fresh inputs") {
            let s2 = stream + MLXArray(Float(0), dtype: .bfloat16)
            return sig(sqrtSign(gateOf(normQ(s2))))
        }
    }
}
