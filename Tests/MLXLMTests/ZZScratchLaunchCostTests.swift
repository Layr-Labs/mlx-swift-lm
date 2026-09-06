import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest

@testable import MLXLLM

final class ZZScratchLaunchCostTests: XCTestCase {
    func testCustomKernelVsBuiltinPerLaunch() throws {
        let kernel = MLXFast.metalKernel(
            name: "zz_add", inputNames: ["a", "b"], outputNames: ["y"],
            source: """
                    auto i = thread_position_in_grid.x;
                    y[i] = a[i] + b[i];
                """)
        let n = 2560 * 4
        let a = MLXRandom.normal([n]).asType(.bfloat16)
        let b = MLXRandom.normal([n]).asType(.bfloat16)
        eval(a, b)
        func custom(_ x: MLXArray) -> MLXArray {
            kernel([x, b], template: [("T", DType.bfloat16)], grid: (n, 1, 1), threadGroup: (256, 1, 1),
                   outputShapes: [[n]], outputDTypes: [.bfloat16])[0]
        }
        func bench(_ name: String, _ chain: Int, _ step: (MLXArray) -> MLXArray) -> Double {
            for _ in 0 ..< 5 { var x = a; for _ in 0 ..< chain { x = step(x) }; eval(x) }
            var t: [Double] = []
            for _ in 0 ..< 30 {
                var x = a
                let s = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< chain { x = step(x) }
                eval(x)
                t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
            }
            t.sort(); return t[0]
        }
        let c1 = bench("custom x1", 1, custom); let c100 = bench("custom x100", 100, custom)
        let b1 = bench("builtin add x1", 1) { $0 + b }; let b100 = bench("builtin add x100", 100) { $0 + b }
        let r1 = bench("builtin rmsNorm x1", 1) { MLXFast.rmsNorm($0, weight: MLXArray.mlxNone, eps: 1e-6) }
        let r100 = bench("builtin rmsNorm x100", 100) { MLXFast.rmsNorm($0, weight: MLXArray.mlxNone, eps: 1e-6) }
        print(String(format: "LAUNCH custom-kernel per launch %.1f us (x1 %.3f ms, x100 %.3f ms)", (c100 - c1) / 99 * 1000, c1, c100))
        print(String(format: "LAUNCH builtin add per launch %.1f us (x1 %.3f ms, x100 %.3f ms)", (b100 - b1) / 99 * 1000, b1, b100))
        print(String(format: "LAUNCH builtin rmsNorm per launch %.1f us (x1 %.3f ms, x100 %.3f ms)", (r100 - r1) / 99 * 1000, r1, r100))
    }
}
