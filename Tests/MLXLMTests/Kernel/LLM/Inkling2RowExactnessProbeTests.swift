import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Which matmul formulations keep a row's bits independent of how many rows
/// run together, at Inkling 2 projection shapes. Prints a table; asserts
/// nothing until the decode projection design settles.
extension KernelTests {
    @Suite(.serialized)
    struct Inkling2RowExactnessProbeTests {
        static let shapes: [(String, Int, Int, DType)] = [
            ("wq", 2304, 3072, .bfloat16), ("wo", 3072, 2304, .bfloat16),
            ("wk", 2304, 1024, .bfloat16), ("wr", 2304, 384, .bfloat16),
            ("sink", 2304, 24, .bfloat16), ("router", 2304, 128, .float32),
            ("down", 4608, 64, .bfloat16),
        ]

        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            arrayEqual(a.asType(.float32), b.asType(.float32)).item(Bool.self)
        }

        @Test func reportRowIndependence() throws {
            guard ProcessInfo.processInfo.environment["INKLING2_ROW_PROBE"] == "1" else { return }
            var lines: [String] = []
            for (name, k, n, dtype) in Self.shapes {
                let w = (MLXRandom.normal([n, k], key: MLXRandom.key(1)) * 0.02).asType(dtype)
                let x = MLXRandom.normal([64, k], key: MLXRandom.key(2)).asType(dtype)
                let single = matmul(x[0 ..< 1], w.T)
                let pair = matmul(x[0 ..< 2], w.T)[0 ..< 1]
                var plain: [String] = []
                var broadcast: [String] = []
                for m in [1, 2, 3, 4, 8, 16, 17, 32, 64] {
                    let y = matmul(x[0 ..< m], w.T)[0 ..< 1]
                    plain.append(
                        "\(m):\(Self.bitsEqual(y, single) ? "=1" : "")\(Self.bitsEqual(y, pair) ? "=2" : "")"
                    )
                    let yb = matmul(
                        x[0 ..< m].reshaped(m, 1, k), w.T.expandedDimensions(axis: 0))[0]
                    broadcast.append("\(m):\(Self.bitsEqual(yb, single) ? "=1" : "x")")
                }
                lines.append(
                    "\(name) plain[\(plain.joined(separator: " "))] bcast[\(broadcast.joined(separator: " "))]"
                )
            }
            print("INKLING2_ROW_PROBE\n" + lines.joined(separator: "\n"))
        }
    }
}
