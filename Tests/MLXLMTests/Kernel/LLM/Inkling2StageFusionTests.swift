import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// The stage kernels give the bits of the chains and of the plain graphs: the
/// short convolution with each tail (FP32 value, added to the layer input,
/// residual add and cast), chunks shorter than the kernel, strided inputs and
/// carried states; a message gate's products with their tail; and a forward of
/// a random model, one-shot and in chunks.
extension KernelTests {
    @Suite
    struct Inkling2StageFusionTests {
        typealias ConvState = Inkling2ProjectionFusionTests.ConvState

        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            Inkling2ProjectionFusionTests.bitsEqual(a, b)
        }

        static func fused<R>(_ on: Bool, _ body: () throws -> R) rethrows -> R {
            try Inkling2StageFusion.$override.withValue(on) { try body() }
        }

        /// `[b, l, c]` of `dtype` with a zero row and a large one; `layout` 1
        /// takes the rows from inside a wider buffer, 2 every second element.
        static func values(
            _ b: Int, _ l: Int, _ c: Int, dtype: DType, layout: Int, sigma: Float, seed: UInt64
        ) -> MLXArray {
            let width = layout == 1 ? c + 40 : (layout == 2 ? 2 * c : c)
            var x = MLXRandom.normal([b, l, width], key: MLXRandom.key(seed)) * sigma
            x[0, 0] = MLXArray.zeros([width])
            x[0, l - 1] = x[0, l - 1] * 50
            x = x.asType(dtype)
            eval(x)
            switch layout {
            case 1: return x[0..., 0..., 24 ..< 24 + c]
            case 2: return x[0..., 0..., .stride(from: 0, by: 2)]
            default: return x
            }
        }

        static func module(channels: Int, dtype: DType, seed: UInt64) throws
            -> Inkling2ShortConvolution
        {
            let module = Inkling2ShortConvolution(channels: channels, kernel: 4)
            let weight = (MLXRandom.normal([channels, 1, 4], key: MLXRandom.key(seed)) * 0.5)
                .asType(dtype)
            try module.update(
                parameters: ModuleParameters.unflattened([("weight", weight)]), verify: [.all])
            return module
        }

        /// The layer's chain: the module, then the tail's casts and add.
        static func reference(
            _ module: Inkling2ShortConvolution, _ x: MLXArray, previous: MLXArray?,
            tail: Inkling2StageFusion.Tail, compiled: Bool
        ) -> (MLXArray, MLXArray) {
            Inkling2Fused.$override.withValue(compiled) {
                let (y, next) = module(x, state: previous)
                switch tail {
                case .widened:
                    return (y.asType(.float32), next)
                case .added(let other):
                    return (other.asType(.float32) + y.asType(.float32), next)
                case .residual(let residual, let dtype):
                    return ((residual + y.asType(.float32)).asType(dtype), next)
                }
            }
        }

        @Test(arguments: [0, 1, 2])
        func convolutionsMatchTheChains(tailKind: Int) throws {
            for dtype in [DType.bfloat16, .float32] {
                for channels in [2304, 70] {
                    let module = try Self.module(channels: channels, dtype: dtype, seed: 11)
                    for (b, l) in [(1, 1), (1, 2), (1, 3), (1, 4), (1, 37), (2, 9), (1, 300)] {
                        for layout in 0 ..< 3 {
                            for carried in [false, true] {
                                let seed = UInt64(b * 1000 + l * 10 + layout)
                                let x = Self.values(
                                    b, l, channels, dtype: dtype, layout: layout, sigma: 1.2,
                                    seed: seed)
                                let tail: Inkling2StageFusion.Tail =
                                    switch tailKind {
                                    case 0: .widened
                                    case 1:
                                        .added(
                                            Self.values(
                                                b, l, channels, dtype: dtype, layout: layout,
                                                sigma: 2, seed: seed + 1))
                                    default:
                                        .residual(
                                            Self.values(
                                                b, l, channels, dtype: .float32, layout: layout,
                                                sigma: 3, seed: seed + 2), dtype)
                                    }
                                let previous: MLXArray? =
                                    carried
                                    ? Self.values(
                                        b, 3, channels, dtype: dtype, layout: 1, sigma: 1,
                                        seed: seed + 3)
                                    : nil
                                let state = ConvState()
                                if let previous { state.states[5 * 4 + 3] = previous }
                                let got = try #require(
                                    Self.fused(true) {
                                        Inkling2StageFusion.convolution(
                                            module, x, tail: tail, slot: 5, kind: .postMLP,
                                            state: state)
                                    })
                                let label =
                                    "tail \(tailKind) \(dtype) c\(channels) b\(b) l\(l) layout \(layout) carried \(carried)"
                                for compiled in [true, false] {
                                    let (y, next) = Self.reference(
                                        module, x, previous: previous, tail: tail,
                                        compiled: compiled)
                                    #expect(Self.bitsEqual(got.output, y), "\(compiled) \(label)")
                                    #expect(
                                        Self.bitsEqual(got.next, next), "next \(compiled) \(label)")
                                    #expect(
                                        Self.bitsEqual(state.states[23]!, next),
                                        "stored \(compiled) \(label)")
                                }
                                #expect(state.inputs[23]! === x, "input \(label)")
                            }
                        }
                    }
                }
            }
        }

        /// Switched off, with the plain graphs, or with a state of another
        /// dtype: the chains run and nothing is stored.
        @Test func uncoveredCallsReturnNil() throws {
            let module = try Self.module(channels: 64, dtype: .bfloat16, seed: 3)
            let x = Self.values(1, 5, 64, dtype: .bfloat16, layout: 0, sigma: 1, seed: 4)
            func call(_ state: ConvState) -> Bool {
                Inkling2StageFusion.convolution(
                    module, x, tail: .widened, slot: 0, kind: .postAttention, state: state) != nil
            }
            #expect(Self.fused(true) { call(ConvState()) })
            let off = ConvState()
            #expect(!Self.fused(false) { call(off) })
            #expect(off.states.isEmpty)
            #expect(
                !Inkling2Fused.$override.withValue(false) { Self.fused(true) { call(ConvState()) } }
            )
            let other = ConvState()
            other.states[2] = MLXArray.zeros([1, 3, 64], dtype: .float32)
            #expect(!Self.fused(true) { call(other) })
        }

        static func gate(hidden: Int, rank: Int, dtype: DType, seed: UInt64) throws
            -> Inkling2MessageGate
        {
            let gate = Inkling2MessageGate(hidden: hidden, rank: rank)
            let parameters: [(String, MLXArray)] = [
                (
                    "down_weight",
                    (MLXRandom.normal([rank, 2 * hidden], key: MLXRandom.key(seed)) * 0.1)
                        .asType(dtype)
                ),
                (
                    "forget_weight",
                    MLXRandom.normal([hidden, rank], key: MLXRandom.key(seed + 1)) * 0.6
                ),
                (
                    "gain_weight",
                    MLXRandom.normal([hidden, rank], key: MLXRandom.key(seed + 2)) * 0.6
                ),
            ]
            try gate.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
            gate.prepareFusedGates()
            return gate
        }

        /// The module with the stage kernel against the product and the
        /// compiled tail, and against the plain graph of the tail.
        @Test(arguments: [(96, 8), (2304, 64)])
        func messageGatesMatchTheChains(hidden: Int, rank: Int) throws {
            for dtype in [DType.bfloat16, .float32] {
                let gate = try Self.gate(hidden: hidden, rank: rank, dtype: dtype, seed: 21)
                #expect(gate.hasFusedGates)
                for (b, l) in [(1, 1), (1, 7), (1, 65), (2, 9), (1, 300)] {
                    let seed = UInt64(b * 100 + l)
                    // Scales from the linear part of tanh to saturation.
                    let h = Self.values(
                        b, l, hidden, dtype: .float32, layout: 0, sigma: 3, seed: seed)
                    let m = Self.values(
                        b, l, hidden, dtype: .float32, layout: 1, sigma: 3, seed: seed + 1)
                    let on = Self.fused(true) { gate(h, message: m) }
                    let off = Self.fused(false) { gate(h, message: m) }
                    let label = "\(hidden)x\(rank) \(dtype) b\(b) l\(l)"
                    #expect(on.shape == h.shape && on.dtype == .float32)
                    #expect(Self.bitsEqual(on, off), "\(label)")
                }
            }
        }

        /// The kernel itself, on `g` values chosen to reach the tails of tanh.
        @Test func messageGateKernelMatchesProductAndTail() throws {
            let (hidden, rank, rows) = (160, 16, 77)
            let fused = MLXRandom.normal([2 * hidden, rank], key: MLXRandom.key(31)) * 0.5
            let h = Self.values(1, rows, hidden, dtype: .float32, layout: 0, sigma: 2, seed: 32)
            let m = Self.values(1, rows, hidden, dtype: .float32, layout: 0, sigma: 2, seed: 33)
            for scale: Float in [0.01, 1, 30] {
                var g = MLXRandom.normal([1, rows, rank], key: MLXRandom.key(34)) * scale
                g[0, 0] = MLXArray.zeros([rank])
                eval(g, fused)
                let got = try #require(
                    Self.fused(true) {
                        Inkling2StageFusion.messageGate(g, fused: fused, h: h, m: m)
                    })
                let both = Inkling2RowExact.fp32Rows(g, fused)
                let (f, gain) = (both[.ellipsis, ..<hidden], both[.ellipsis, hidden...])
                #expect(
                    Self.bitsEqual(got, Inkling2Fused.messageGateTail(h, m, f: f, g: gain)),
                    "compiled, scale \(scale)")
                #expect(
                    Self.bitsEqual(got, (1 - tanh(f)) * h + tanh(gain) * m),
                    "plain, scale \(scale)")
            }
            // Row-exact steps take other products: not covered.
            let g = MLXRandom.normal([1, rows, rank], key: MLXRandom.key(35))
            let rowExact = Inkling2RowExact.$active.withValue(true) {
                Self.fused(true) { Inkling2StageFusion.messageGate(g, fused: fused, h: h, m: m) }
            }
            #expect(rowExact == nil)
        }

        /// A random looped model: the whole forward, one-shot and in chunks
        /// (some shorter than the convolution kernel) through carried states,
        /// has the same bits with the stage kernels as with the chains.
        @Test(arguments: [DType.float32, .bfloat16])
        func forwardsMatchBitForBit(dtype: DType) throws {
            let model = try Inkling2ModelTests.makeModel(seed: 41)
            if dtype != .float32 {
                model.update(
                    parameters: ModuleParameters.unflattened(
                        model.parameters().flattened().map {
                            (
                                $0.0,
                                $0.1.dtype == .float32 && !Inkling2Checkpoint.keepsFP32($0.0)
                                    ? $0.1.asType(dtype) : $0.1
                            )
                        }))
            }
            try model.prepare()
            let tokens = Inkling2ModelTests.tokens(9, count: 61)

            func visits(_ on: Bool) -> (MLXArray, [MLXArray]) {
                var outputs: [MLXArray] = []
                let logits = Self.fused(on) {
                    model.forwardNoCache(tokens) { _, output in outputs.append(output) }.logits
                }
                return (logits, outputs)
            }
            let (plain, plainVisits) = visits(false)
            let (fused, fusedVisits) = visits(true)
            #expect(Self.bitsEqual(plain, fused))
            #expect(plainVisits.count == fusedVisits.count)
            for (i, (a, b)) in zip(plainVisits, fusedVisits).enumerated() {
                #expect(Self.bitsEqual(a, b), "visit \(i)")
            }

            func chunked(_ on: Bool, _ lengths: [Int]) -> (MLXArray, [Int: MLXArray]) {
                let state = Inkling2ModelTests.IncrementalState()
                var pieces: [MLXArray] = []
                for length in lengths {
                    state.begin(length)
                    pieces.append(
                        Self.fused(on) {
                            model.forward(
                                tokens[0..., state.offset ..< state.offset + length], state: state
                            ).logits
                        })
                    state.end(length)
                }
                let logits = concatenated(pieces, axis: 1)
                eval([logits] + Array(state.conv.values))
                return (logits, state.conv)
            }
            for lengths in [[29, 32], [1, 2, 3, 1, 54], [60, 1]] {
                let (a, aStates) = chunked(false, lengths)
                let (b, bStates) = chunked(true, lengths)
                #expect(Self.bitsEqual(a, b), "chunks \(lengths)")
                #expect(aStates.count == bStates.count)
                for (key, value) in aStates {
                    #expect(Self.bitsEqual(value, bStates[key]!), "state \(key) chunks \(lengths)")
                }
            }
        }
    }
}
