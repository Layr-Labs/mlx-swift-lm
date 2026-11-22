import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// The fused projection stage gives the bits of the compiled chains and of the
/// plain graphs: the scaled clips (128- and 16-wide rows, with and without the
/// log scaling, contiguous and strided products), the static and dynamic
/// convolutions with the L2 clip and the RMS norm (with and without a carried
/// state, given and computed dynamic weights), and a forward of a random model
/// with 128-wide heads, one-shot and in chunks.
extension KernelTests {
    @Suite
    struct Inkling2ProjectionFusionTests {
        /// The same shape, dtype and bits (signed zeros and NaN payloads included).
        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            guard a.shape == b.shape, a.dtype == b.dtype else { return false }
            let view: DType = a.dtype.size == 4 ? .uint32 : .uint16
            let (x, y) = (contiguous(a).view(dtype: view), contiguous(b).view(dtype: view))
            return all(x .== y).item(Bool.self)
        }

        static func fused<R>(_ on: Bool, _ body: () throws -> R) rethrows -> R {
            try Inkling2ProjectionFusion.$override.withValue(on) { try body() }
        }

        /// `[B, L, n]` products of `dtype`, strided rows inside a wider buffer
        /// when `strided`, scaled so rows fall on both sides of the clip.
        static func products(
            _ b: Int, _ l: Int, _ n: Int, dtype: DType, strided: Bool, sigma: Float, seed: UInt64
        ) -> MLXArray {
            let pad = strided ? 40 : 0
            var x = MLXRandom.normal([b, l, n + pad], key: MLXRandom.key(seed)) * sigma
            // A zero row and a large one exercise the edges of the clip.
            x[0, 0] = MLXArray.zeros([n + pad])
            x[0, l - 1] = x[0, l - 1] * 50
            x = x.asType(dtype)
            eval(x)
            return strided ? x[0..., 0..., 24 ..< 24 + n] : x
        }

        static func clipModule(heads: Int, dimensions: Int, threshold: Float, seed: UInt64)
            throws -> Inkling2ScaledL2Clip
        {
            let module = Inkling2ScaledL2Clip(
                heads: heads, dimensions: dimensions, threshold: threshold)
            let scale = MLXRandom.uniform(
                low: 0.5, high: 2.5, [heads, dimensions], key: MLXRandom.key(seed))
            try module.update(
                parameters: ModuleParameters.unflattened([("scale", scale)]), verify: [.all])
            return module
        }

        static func tau(_ b: Int, _ l: Int, perRow: Bool, seed: UInt64) -> MLXArray {
            let positions = MLXRandom.randInt(
                low: 0, high: 40_000, [perRow ? b : 1, l], key: MLXRandom.key(seed)
            ).asType(.int32)
            return Inkling2Math.logScale(positions: positions, floor: 8192, alpha: 0.1)
                .reshaped(-1, l, 1, 1)
        }

        /// `logScaled(module(products))` as Inkling2Attention traces it.
        static func referenceClip(
            _ products: MLXArray, _ module: Inkling2ScaledL2Clip, tau: MLXArray?, compiled: Bool
        ) -> MLXArray {
            let (b, l) = (products.dim(0), products.dim(1))
            let v = Inkling2Fused.$override.withValue(compiled) {
                module(products.reshaped(b, l, module.scale.dim(0), -1))
            }
            guard let tau else { return v }
            return (v.asType(.float32) * tau).asType(v.dtype)
        }

        @Test(arguments: [
            (24, 128, Float(22.627416997969522), Float(1.5)), (24, 16, Float(8), Float(2)),
            (3, 64, Float(4), Float(0.5)), (5, 12, Float(2), Float(0.7)),
        ])
        func clipsMatchTheChains(heads: Int, d: Int, threshold: Float, sigma: Float) throws {
            let module = try Self.clipModule(
                heads: heads, dimensions: d, threshold: threshold, seed: 3)
            for dtype in [DType.bfloat16, .float16, .float32] {
                for (b, l) in [(1, 37), (2, 9), (1, 300)] {
                    for strided in [false, true] {
                        for tauKind in 0 ..< 3 {
                            let x = Self.products(
                                b, l, heads * d, dtype: dtype, strided: strided, sigma: sigma,
                                seed: UInt64(7 + b * 1000 + l))
                            let tau =
                                tauKind == 0
                                ? nil : Self.tau(b, l, perRow: tauKind == 2, seed: UInt64(l))
                            let got = try #require(
                                Self.fused(true) {
                                    Inkling2ProjectionFusion.clip(x, module, tau: tau)
                                })
                            let label = "\(heads)x\(d) \(dtype) b\(b) l\(l) s\(strided) t\(tauKind)"
                            #expect(
                                Self.bitsEqual(
                                    got, Self.referenceClip(x, module, tau: tau, compiled: true)),
                                "compiled \(label)")
                            #expect(
                                Self.bitsEqual(
                                    got, Self.referenceClip(x, module, tau: tau, compiled: false)),
                                "plain \(label)")
                        }
                    }
                }
            }
        }

        /// Fewer than 32 rows of 128: MLX sums them with another kernel, so
        /// the chains run; switched off: the chains run.
        @Test func uncoveredCallsReturnNil() throws {
            let module = try Self.clipModule(heads: 24, dimensions: 128, threshold: 22, seed: 4)
            let x = Self.products(
                1, 1, 24 * 128, dtype: .bfloat16, strided: false, sigma: 1, seed: 5)
            #expect(Self.fused(true) { Inkling2ProjectionFusion.clip(x, module, tau: nil) } == nil)
            let y = Self.products(
                1, 4, 24 * 128, dtype: .bfloat16, strided: false, sigma: 1, seed: 6)
            #expect(Self.fused(true) { Inkling2ProjectionFusion.clip(y, module, tau: nil) } != nil)
            #expect(Self.fused(false) { Inkling2ProjectionFusion.clip(y, module, tau: nil) } == nil)
            let plain = Inkling2Fused.$override.withValue(false) {
                Self.fused(true) { Inkling2ProjectionFusion.clip(y, module, tau: nil) }
            }
            #expect(plain == nil)
        }

        /// Conv states held per (slot, kind), with the inputs the stage passes.
        final class ConvState: Inkling2StepState {
            var positions = MLXArray.zeros([1, 0], dtype: .int32)
            var states: [Int: MLXArray] = [:]
            var inputs: [Int: MLXArray] = [:]

            func convState(slot: Int, _ kind: Inkling2ConvKind) -> MLXArray? {
                states[slot * 4 + kind.rawValue]
            }
            func setConvState(
                slot: Int, _ kind: Inkling2ConvKind, _ state: MLXArray, input: MLXArray
            ) {
                states[slot * 4 + kind.rawValue] = state
                inputs[slot * 4 + kind.rawValue] = input
            }
            func attend(slot: Int, _ inputs: Inkling2AttentionInputs) -> MLXArray {
                fatalError("not used")
            }
        }

        static func convolutionModule(
            dynamic: Bool, channels: Int, heads: Int, hidden: Int, dtype: DType, seed: UInt64
        ) throws -> Module {
            let kernel = 4
            if dynamic {
                let module = Inkling2DynamicShortConvolution(
                    channels: channels, inputDimensions: hidden, heads: heads, kernel: kernel)
                let parameters: [(String, MLXArray)] = [
                    (
                        "static_w",
                        (MLXRandom.normal([kernel, channels], key: MLXRandom.key(seed)) * 0.5)
                            .asType(dtype)
                    ),
                    (
                        "weight_proj",
                        (MLXRandom.normal(
                            [heads * kernel, hidden], key: MLXRandom.key(seed + 1)) * 0.05)
                            .asType(dtype)
                    ),
                ]
                try module.update(
                    parameters: ModuleParameters.unflattened(parameters), verify: [.all])
                return module
            }
            let module = Inkling2ShortConvolution(channels: channels, kernel: kernel)
            let weight = (MLXRandom.normal([channels, 1, kernel], key: MLXRandom.key(seed)) * 0.5)
                .asType(dtype)
            try module.update(
                parameters: ModuleParameters.unflattened([("weight", weight)]), verify: [.all])
            return module
        }

        /// What Inkling2Attention's chains return for the key (clip) or value
        /// (norm) convolution, and the next state.
        static func referenceConvolution(
            _ module: Module, _ x: MLXArray, input: MLXArray, projected: MLXArray?,
            previous: MLXArray?, heads: Int, tail: Inkling2ProjectionFusion.Tail, compiled: Bool
        ) -> (MLXArray, MLXArray) {
            let (b, l, dtype) = (x.dim(0), x.dim(1), x.dtype)
            return Inkling2Fused.$override.withValue(compiled) {
                let y: MLXArray
                let next: MLXArray
                if let dynamic = module as? Inkling2DynamicShortConvolution {
                    (y, next) = dynamic(
                        x, dynamicInput: input, state: previous, projected: projected)
                } else {
                    (y, next) = (module as! Inkling2ShortConvolution)(x, state: previous)
                }
                let rows = y.reshaped(b, l, heads, -1)
                let out =
                    switch tail {
                    case .clip(let threshold): Inkling2Math.l2Clip(rows, threshold: threshold)
                    case .norm(let eps): Inkling2Math.unweightedRMSNorm(rows, eps: eps)
                    }
                return (out.asType(dtype), next)
            }
        }

        @Test(arguments: [false, true])
        func convolutionsMatchTheChains(dynamic: Bool) throws {
            let (heads, hidden) = (8, 96)
            let channels = heads * 128
            for dtype in [DType.bfloat16, .float32] {
                let module = try Self.convolutionModule(
                    dynamic: dynamic, channels: channels, heads: heads, hidden: hidden,
                    dtype: dtype, seed: 11)
                for (b, l) in [(1, 4), (1, 37), (2, 9), (1, 300)] {
                    for strided in [false, true] {
                        for carried in [false, true] {
                            for tail in [
                                Inkling2ProjectionFusion.Tail.clip(22.627416997969522),
                                .norm(1e-6),
                            ] {
                                let seed = UInt64(b * 100 + l)
                                let x = Self.products(
                                    b, l, channels, dtype: dtype, strided: strided, sigma: 1.2,
                                    seed: seed)
                                let input =
                                    (MLXRandom.normal(
                                        [b, l, hidden], key: MLXRandom.key(seed + 1))).asType(dtype)
                                let wide =
                                    (MLXRandom.normal(
                                        [b, l, heads * 4 + 8], key: MLXRandom.key(seed + 2)) * 0.3)
                                    .asType(dtype)
                                let projections: [MLXArray?] =
                                    dynamic ? [nil, wide[0..., 0..., 8...]] : [nil]
                                for projected in projections {
                                    let previous: MLXArray? =
                                        carried
                                        ? Self.products(
                                            b, 3, channels + 64, dtype: dtype, strided: false,
                                            sigma: 1, seed: seed + 3)[
                                                0..., 0..., 32 ..< 32 + channels]
                                        : nil
                                    let state = ConvState()
                                    if let previous { state.states[5 * 4 + 1] = previous }
                                    let got = try #require(
                                        Self.fused(true) {
                                            Inkling2ProjectionFusion.convolution(
                                                module, x, input: input, projected: projected,
                                                heads: heads, tail: tail, dtype: dtype, slot: 5,
                                                kind: .value, state: state)
                                        })
                                    let label =
                                        "dyn \(dynamic) \(dtype) b\(b) l\(l) s\(strided) c\(carried) \(tail) p\(projected != nil)"
                                    for compiled in [true, false] {
                                        let (y, next) = Self.referenceConvolution(
                                            module, x, input: input, projected: projected,
                                            previous: previous, heads: heads, tail: tail,
                                            compiled: compiled)
                                        #expect(Self.bitsEqual(got, y), "\(compiled) \(label)")
                                        #expect(
                                            Self.bitsEqual(state.states[21]!, next),
                                            "next \(compiled) \(label)")
                                    }
                                    #expect(state.inputs[21]! === x, "input \(label)")
                                }
                            }
                        }
                    }
                }
            }
        }

        /// A random model with 128-wide heads: the whole forward, one-shot
        /// and in chunks through carried states, has the same bits with the
        /// fused stage as with the chains.
        @Test func forwardsMatchBitForBit() throws {
            var base = Inkling2ModelTests.base
            var text = base["text_config"] as! [String: Any]
            text["hidden_size"] = 128
            text["head_dim"] = 128
            text["num_attention_heads"] = 4
            text["num_key_value_heads"] = 2
            text["d_rel"] = 16
            text["sliding_window_size"] = 24
            text["rel_extent"] = 32
            text["log_scaling_n_floor"] = 16
            base["text_config"] = text
            let model = Inkling2Model(
                try SyntheticModel.configuration(Inkling2Configuration.self, base))
            SyntheticModel.randomize(model, seed: 31)
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

            func chunked(_ on: Bool) -> MLXArray {
                let state = Inkling2ModelTests.IncrementalState()
                var pieces: [MLXArray] = []
                for length in [29, 32] {
                    state.begin(length)
                    pieces.append(
                        Self.fused(on) {
                            model.forward(
                                tokens[0..., state.offset ..< state.offset + length], state: state
                            ).logits
                        })
                    state.end(length)
                }
                return concatenated(pieces, axis: 1)
            }
            #expect(Self.bitsEqual(chunked(false), chunked(true)))
        }
    }
}
