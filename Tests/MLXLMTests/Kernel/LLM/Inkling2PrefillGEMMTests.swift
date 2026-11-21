import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// The fixed-tile prefill GEMM (M5 neural accelerators): every row of every
/// Inkling 2 prefill product shape is bit-identical for any row count and any
/// company, and as accurate as MLX's own GEMM against an FP32 reference.
/// Skipped without the neural accelerators.
extension KernelTests {
    @Suite(.enabled(if: Inkling2PagedAttention.neuralAcceleratorsAvailable))
    struct Inkling2PrefillGEMMTests {
        /// (outputs, inputs) of the prefill products: sliding [wq; wr; sink;
        /// gate; wk; wv], full [wq; wr; sink; gate], [wk; k weight_proj], wk,
        /// wr, sink, wo, message-gate down.
        static let shapes: [(Int, Int)] = [
            (8600, 2304), (6552, 2304), (1056, 2304), (1024, 2304), (384, 2304), (24, 2304),
            (2304, 3072), (64, 4608),
        ]

        static let lengths = [
            1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256,
            257, 511, 512, 513, 1000, 1024, 1500, 2047,
        ]

        static func operands(_ n: Int, _ k: Int, rows: Int = 2048) -> (MLXArray, MLXArray) {
            let x = (MLXRandom.normal([rows, k], key: MLXRandom.key(UInt64(n + k))) * 0.5)
                .asType(.bfloat16)
            let w = (MLXRandom.normal([n, k], key: MLXRandom.key(UInt64(n * 7 + 1))) * 0.02)
                .asType(.bfloat16)
            eval(x, w)
            return (x, w)
        }

        static func product(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
            let y = Inkling2PrefillGEMM.matmul(x, transposedWeight: w)!
            eval(y)
            return y
        }

        static func equal(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && arrayEqual(a, b).item(Bool.self)
        }

        /// The first M rows of a 2,048-row call equal an M-row call for every M,
        /// the last rows alone equal theirs, a shuffled call equals the shuffled
        /// rows, and a repeat is identical.
        @Test(arguments: shapes.map { [$0.0, $0.1] })
        func rowsDoNotDependOnRowCountOrCompany(shape: [Int]) throws {
            let (n, k) = (shape[0], shape[1])
            let (x, w) = Self.operands(n, k)
            let full = Self.product(x, w)
            #expect(Self.equal(Self.product(x, w), full), "repeat")
            for m in Self.lengths {
                #expect(Self.equal(Self.product(x[0 ..< m], w), full[0 ..< m]), "M = \(m)")
            }
            for m in [1, 3, 77] {
                #expect(
                    Self.equal(Self.product(x[(2048 - m)...], w), full[(2048 - m)...]),
                    "last \(m)")
            }
            var rng = KernelTests.Inkling2ComponentTests.Lcg(state: 5)
            var permutation = Array(0 ..< 2048)
            for i in stride(from: 2047, to: 0, by: -1) {
                permutation.swapAt(i, Int(rng.next() % UInt64(i + 1)))
            }
            let p = MLXArray(permutation.map(Int32.init))
            #expect(Self.equal(Self.product(x[p], w), full[p]), "shuffled")
        }

        /// Against an FP32 product on the CPU: as accurate as MLX's GEMM (both
        /// round an FP32 sum to BF16).
        @Test(arguments: shapes.map { [$0.0, $0.1] })
        func matchesTheFP32ProductLikeMLX(shape: [Int]) throws {
            let (n, k) = (shape[0], shape[1])
            let (x, w) = Self.operands(n, k, rows: 96)
            let reference = matmul(x.asType(.float32), w.asType(.float32).T, stream: .cpu)
            let ours = Self.product(x, w).asType(.float32)
            let mlx = matmul(x, w.T).asType(.float32)
            let scale = abs(reference).max().item(Float.self)
            let oursError = abs(ours - reference).max().item(Float.self) / scale
            let mlxError = abs(mlx - reference).max().item(Float.self) / scale
            let cosine =
                (ours * reference).sum().item(Float.self)
                / (sqrt((ours * ours).sum()) * sqrt((reference * reference).sum())).item(
                    Float.self)
            print(
                "inkling2-prefill-gemm accuracy \(n)x\(k): max rel \(oursError) (MLX \(mlxError)), cosine \(cosine)"
            )
            #expect(cosine >= 0.99999)
            #expect(oursError <= mlxError * 1.25 + 1e-6)
        }

        /// `prefillMatmul` takes the kernel (any row count, no padding) and the
        /// off switch restores the padded calls; row-exact steps do not use it.
        @Test func prefillMatmulUsesTheKernel() throws {
            // Invariant prefill is a per-model switch (off by default) since
            // 0daf9d16: turn it on with 512-row calls for this check.
            try Inkling2RowExact.withPrefill(blockRows: 512) { try check() }
        }

        func check() throws {
            let (x, w) = Self.operands(1024, 2304, rows: 70)
            precondition(Inkling2RowExact.prefillBlockRows > 70)
            let direct = Self.product(x, w)
            let on = Inkling2PrefillGEMM.$override.withValue(true) {
                Inkling2RowExact.prefillMatmul(x, transposedWeight: w)
            }
            let off = Inkling2PrefillGEMM.$override.withValue(false) {
                Inkling2RowExact.prefillMatmul(x, transposedWeight: w)
            }
            let block = Inkling2RowExact.prefillBlockRows
            let padded = matmul(
                concatenated([x, MLXArray.zeros([block - 70, 2304], dtype: .bfloat16)]), w.T
            )[0 ..< 70]
            #expect(Self.equal(on, direct))
            #expect(Self.equal(off, padded))
        }
    }
}
