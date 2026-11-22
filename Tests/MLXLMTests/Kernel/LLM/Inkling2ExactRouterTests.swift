import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// The exact FP32 product gives the same bits on every tile, and the prefill
/// router's select kernel gives the bits of the module's graph (indices in
/// order and weights), ties and edge values included.
extension KernelTests {
    @Suite
    struct Inkling2ExactRouterTests {
        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            Inkling2ProjectionFusionTests.bitsEqual(a, b)
        }

        /// FP32 operands with signed zeros, a denormal row and a large row.
        static func operands(rows: Int, k: Int, n: Int, seed: UInt64) -> (MLXArray, MLXArray) {
            var x = MLXRandom.normal([max(rows, 3), k], key: MLXRandom.key(seed))
            x[0, 0] = MLXArray(Float(0))
            x[0, 1] = MLXArray(-Float(0))
            x[1] = x[1] * 1e-30
            x[2] = x[2] * 1e18
            let w = MLXRandom.normal([n, k], key: MLXRandom.key(seed + 1)) * 0.05
            let rowsOnly = x[0 ..< rows]
            eval(rowsOnly, w)
            return (rowsOnly, w)
        }

        static let tiles: [Inkling2ExactFP32.Tile] = [
            .wide, .tall, Inkling2ExactFP32.Tile("64x64:16:4x4")!,
            Inkling2ExactFP32.Tile("128x128:16:4x8")!, Inkling2ExactFP32.Tile("32x32:8:2x2")!,
            Inkling2ExactFP32.Tile("96x64:16:6x4")!,
        ]

        /// Router, message gate and loop-message shapes, and shapes that leave
        /// partial tiles in every direction.
        @Test(arguments: [(2304, 128), (64, 4608), (2304, 48), (100, 130), (37, 16)])
        func everyTileGivesTheShippedBits(k: Int, n: Int) throws {
            for rows in [1, 7, 63, 64, 65, 129, 300] {
                let (x, w) = Self.operands(rows: rows, k: k, n: n, seed: UInt64(k + n + rows))
                let shipped = Inkling2ExactFP32.$tilesOverride.withValue(false) {
                    Inkling2ExactFP32.matmul(x, transposedWeight: w)
                }
                for tile in Self.tiles {
                    #expect(tile.valid)
                    #expect(
                        Self.bitsEqual(
                            Inkling2ExactFP32.matmul(x, transposedWeight: w, tile: tile), shipped),
                        "\(k)->\(n) rows \(rows) tile \(tile)")
                }
                let chosen = Inkling2ExactFP32.$tilesOverride.withValue(true) {
                    Inkling2ExactFP32.matmul(x, transposedWeight: w)
                }
                #expect(Self.bitsEqual(chosen, shipped), "\(k)->\(n) rows \(rows) default tile")
            }
            // A row does not depend on its company.
            let (x, w) = Self.operands(rows: 200, k: k, n: n, seed: 9)
            let all = Inkling2ExactFP32.matmul(x, transposedWeight: w)
            #expect(
                Self.bitsEqual(
                    Inkling2ExactFP32.matmul(x[150 ..< 151], transposedWeight: w), all[150 ..< 151])
            )
        }

        @Test func tilesParseAndValidate() {
            #expect(Inkling2ExactFP32.Tile("32x64:16:2x4") == .wide)
            #expect(Inkling2ExactFP32.Tile("64x64:16:8x4") == .tall)
            #expect(Inkling2ExactFP32.Tile("128x128:16:8x8") == nil)
            #expect(Inkling2ExactFP32.Tile("64x64:16:3x4") == nil)
            #expect(Inkling2ExactFP32.Tile("64x64") == nil)
        }

        static let experts = 128
        static let topK = 8

        /// Logits `[rows, experts]` and a bias for each case: random values,
        /// saturated scores (ties at 1 and 0), runs of equal logits across the
        /// k-th place, all equal, infinities and signed zeros.
        static func selectCases(rows: Int) -> [(String, MLXArray, MLXArray)] {
            let normal = MLXRandom.normal([rows, experts], key: MLXRandom.key(3)) * 3
            let bias = MLXRandom.normal([experts], key: MLXRandom.key(4)) * 0.05
            let zero = MLXArray.zeros([experts])
            let quantized = MLX.round(normal * 2) / 2
            var special = normal + 0
            special[0, 5] = MLXArray(Float.infinity)
            special[1, 7] = MLXArray(-Float.infinity)
            special[2] = MLXArray(-Float(0))
            return [
                ("random logits, random bias", normal, bias),
                ("random logits, zero bias", normal, zero),
                ("saturated scores", normal * 40, zero),
                ("quantized logits, zero bias", quantized, zero),
                ("quantized logits, random bias", quantized, bias),
                ("all equal", MLXArray.zeros([rows, experts]), zero),
                ("infinities and signed zeros", special, bias),
            ]
        }

        static func router(bias: MLXArray, hidden: Int, seed: UInt64?) throws -> Inkling2Router {
            let router = Inkling2Router(hidden: hidden, experts: experts, topK: topK, routeScale: 8)
            // An identity weight makes the logits the input itself (finite inputs).
            let identity = (0 ..< experts * hidden).map {
                Float($0 / hidden == $0 % hidden ? 1 : 0)
            }
            let weight =
                seed.map { MLXRandom.normal([experts, hidden], key: MLXRandom.key($0)) * 0.2 }
                ?? MLXArray(identity, [experts, hidden])
            try router.update(
                parameters: ModuleParameters.unflattened([("weight", weight), ("bias", bias)]),
                verify: [.all])
            return router
        }

        /// The select kernel against the module's graph on the same logits.
        @Test func selectKernelMatchesTheGraphOnEveryCase() throws {
            for rows in [1, 7, 300] {
                for (name, logits, bias) in Self.selectCases(rows: max(rows, 3)) {
                    let router = try Self.router(bias: bias, hidden: Self.experts, seed: nil)
                    let own = logits[0 ..< rows]
                    let (indices, weights, _) = Inkling2Kernels.select(router, logits: own)
                    // The module's graph after its logits.
                    let scores = sigmoid(own)
                    let selection = scores + bias
                    let picks = argPartition(-selection, kth: Self.topK - 1, axis: -1)[
                        .ellipsis, ..<Self.topK]
                    let selected = takeAlong(scores, picks, axis: -1)
                    let graphWeights =
                        selected / (selected.sum(axis: -1, keepDims: true) + 1e-20) * 8
                    #expect(
                        Self.bitsEqual(indices, picks.asType(.uint32)),
                        "indices: \(name), \(rows) rows")
                    #expect(Self.bitsEqual(weights, graphWeights), "weights: \(name), \(rows) rows")
                }
            }
        }

        /// The module with the kernel on and off: the finite cases through an
        /// identity weight, and a random router on BF16 states.
        @Test func routerMatchesWithAndWithoutTheKernel() throws {
            func both(_ router: Inkling2Router, _ x: MLXArray) -> Bool {
                let on = Inkling2Router.$selectKernelOverride.withValue(true) { router(x) }
                let off = Inkling2Router.$selectKernelOverride.withValue(false) { router(x) }
                return on.0.dtype == .uint32 && Self.bitsEqual(on.0, off.0)
                    && Self.bitsEqual(on.1, off.1)
            }
            for (name, logits, bias) in Self.selectCases(rows: 65).dropLast() {
                let router = try Self.router(bias: bias, hidden: Self.experts, seed: nil)
                #expect(both(router, logits), "\(name)")
            }
            let bias = MLXRandom.normal([Self.experts], key: MLXRandom.key(6)) * 0.05
            let router = try Self.router(bias: bias, hidden: 256, seed: 7)
            for rows in [1, 61, 300] {
                let x = MLXRandom.normal([rows, 256], key: MLXRandom.key(8)).asType(.bfloat16)
                #expect(both(router, x), "random router, \(rows) rows")
            }
        }

        /// A random looped model: the forward has the same bits with the
        /// tiles and the select kernel as without them.
        @Test func forwardsMatchBitForBit() throws {
            let model = try Inkling2ModelTests.makeModel(seed: 43)
            try model.prepare()
            let tokens = Inkling2ModelTests.tokens(5, count: 47)
            func logits(_ on: Bool) -> MLXArray {
                Inkling2ExactFP32.$tilesOverride.withValue(on) {
                    Inkling2Router.$selectKernelOverride.withValue(on) {
                        model.forwardNoCache(tokens).logits
                    }
                }
            }
            #expect(Self.bitsEqual(logits(false), logits(true)))
        }
    }
}
