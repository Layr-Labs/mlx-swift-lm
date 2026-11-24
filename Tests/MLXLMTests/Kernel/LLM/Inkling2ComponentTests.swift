import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Inkling 2 building blocks against small hand-written references with random
/// weights: the native NVFP4 / MXFP8 expert layout (decoded on the host from
/// the checkpoint bytes), the short convolutions, the biased attention with
/// sinks and the routed MLP with interleaved gate/up rows.
extension KernelTests {
    @Suite
    struct Inkling2ComponentTests {

        // MARK: Host decoders of the checkpoint formats

        static let e2m1: [Float] = [
            0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6,
        ]

        static func e4m3(_ byte: UInt8) -> Float {
            let sign: Float = byte & 0x80 != 0 ? -1 : 1
            let exponent = Int((byte >> 3) & 0xF)
            let mantissa = Float(byte & 0x7)
            if exponent == 0 { return sign * mantissa / 8 * pow(2, -6) }
            return sign * (1 + mantissa / 8) * pow(2, Float(exponent - 7))
        }

        struct Lcg {
            var state: UInt64
            mutating func next() -> UInt64 {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return state >> 33
            }
            mutating func byte() -> UInt8 { UInt8(truncatingIfNeeded: next()) }
            mutating func unit() -> Float { Float(next() % 10_000) / 10_000 }
        }

        /// Checkpoint bytes for one NVFP4 bank and its host-decoded dense weights.
        static func nvfp4Bank(experts: Int, rows: Int, columns: Int, seed: UInt64)
            -> (tensors: [String: MLXArray], dense: [Float])
        {
            var rng = Lcg(state: seed)
            let codes = (0 ..< experts * rows * columns / 2).map { _ in rng.byte() }
            // Normal positive E4M3 scales between 2^-3 and 2^2.
            let scales = (0 ..< experts * rows * columns / 16).map { _ in
                UInt8((4 + Int(rng.next() % 6)) << 3) | UInt8(rng.next() % 8)
            }
            let amax = (0 ..< experts).map { _ in 0.5 + 3 * rng.unit() }
            var dense = [Float](repeating: 0, count: experts * rows * columns)
            for e in 0 ..< experts {
                for r in 0 ..< rows {
                    for c in 0 ..< columns {
                        let byte = codes[(e * rows + r) * columns / 2 + c / 2]
                        let code = c % 2 == 0 ? byte & 0xF : byte >> 4
                        let scale = e4m3(scales[(e * rows + r) * columns / 16 + c / 16])
                        dense[(e * rows + r) * columns + c] =
                            e2m1[Int(code)] * scale * (amax[e] / 2688)
                    }
                }
            }
            let tensors: [String: MLXArray] = [
                "weight": MLXArray(codes, [experts, rows, columns / 2]),
                "weight_scale": MLXArray(scales, [experts, rows, columns / 16]),
                "weight_amax": MLXArray(amax),
                "weight_outer_scale_denominator": MLXArray([Float(2688)]),
            ]
            return (tensors, dense)
        }

        static func mxfp8Bank(experts: Int, rows: Int, columns: Int, seed: UInt64)
            -> (tensors: [String: MLXArray], dense: [Float])
        {
            var rng = Lcg(state: seed)
            let values = (0 ..< experts * rows * columns).map { _ -> UInt8 in
                let b = rng.byte()
                return b & 0x7F == 0x7F ? b & 0xF0 : b  // no NaN codes
            }
            let scales = (0 ..< experts * rows * columns / 32).map { _ in
                UInt8(122 + rng.next() % 8)
            }
            var dense = [Float](repeating: 0, count: values.count)
            for i in 0 ..< values.count {
                let row = i / columns
                let scale = pow(
                    2, Float(Int(scales[row * columns / 32 + (i % columns) / 32]) - 127))
                dense[i] = e4m3(values[i]) * scale
            }
            let tensors: [String: MLXArray] = [
                "weight": MLXArray(values, [experts, rows, columns]),
                "weight_scale": MLXArray(scales, [experts, rows, columns / 32]),
            ]
            return (tensors, dense)
        }

        static func checkpoint(_ bank: String, _ tensors: [String: MLXArray]) -> [String: MLXArray]
        {
            Dictionary(uniqueKeysWithValues: tensors.map { ("\(bank)_\($0.key)", $0.value) })
        }

        static func projectReference(
            _ x: MLXArray, dense: [Float], experts: Int, rows: Int, columns: Int, ids: [Int]
        )
            -> MLXArray
        {
            let w = MLXArray(dense, [experts, rows, columns])
            return concatenated(
                ids.enumerated().map { i, e in matmul(x[i ..< i + 1], w[e].T) }, axis: 0)
        }

        static func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
            let diff = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            return diff / max(abs(b.asType(.float32)).max().item(Float.self), 1e-12)
        }

        // MARK: FP32 products

        /// The FP32 row products (router, message gates, residual mixes) are
        /// batched gemvs: each row equals the same row computed alone, at every
        /// row count. With `INKLING2_FP32ROWS_OUT` set, the results are written
        /// there so runs with TF32 on and off can be compared byte for byte.
        @Test func fp32RowProductsAreRowInvariant() throws {
            let shapes: [(String, Int, Int)] = [
                ("router", 2304, 128), ("gate", 64, 2304), ("mix", 2304, 3),
            ]
            var bytes = Data()
            for (index, (_, k, n)) in shapes.enumerated() {
                let w = MLXRandom.normal([n, k], key: MLXRandom.key(UInt64(10 + index))) * 0.05
                let x = MLXRandom.normal([37, k], key: MLXRandom.key(UInt64(20 + index)))
                let all = Inkling2RowExact.fp32Rows(x, w)
                #expect(all.dtype == .float32)
                for r in [0, 5, 36] {
                    let alone = Inkling2RowExact.fp32Rows(x[r ..< r + 1], w)
                    #expect(arrayEqual(all[r ..< r + 1], alone).item(Bool.self))
                }
                bytes.append(all.asData(access: .copy).data)
            }
            if let path = ProcessInfo.processInfo.environment["INKLING2_FP32ROWS_OUT"] {
                try bytes.write(to: URL(fileURLWithPath: path))
            }
        }

        // MARK: Expert banks

        @Test(arguments: [Inkling2ExpertFormat.nvfp4, .mxfp8])
        func quantizedBankMatchesTheHostDecodedCheckpoint(format: Inkling2ExpertFormat) throws {
            let (experts, rows, columns) = (4, 32, 64)
            let made =
                format == .nvfp4
                ? Self.nvfp4Bank(experts: experts, rows: rows, columns: columns, seed: 7)
                : Self.mxfp8Bank(experts: experts, rows: rows, columns: columns, seed: 7)
            let converted = try Inkling2Checkpoint.convertExperts(
                Self.checkpoint("w2", made.tensors), configuration: Inkling2ModelTests.config(),
                layer: "test")
            let bank = Inkling2QuantizedExpertBank(
                format: format, experts: experts, inputDimensions: columns, outputDimensions: rows)
            let parameters = converted.compactMap { key, value -> (String, MLXArray)? in
                guard key.hasPrefix("w2.") else { return nil }
                return (String(key.dropFirst(3)), value)
            }
            try bank.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])

            let ids = [0, 3, 1, 2, 3, 0]
            let x = MLXRandom.normal([ids.count, columns], key: MLXRandom.key(3))
            let indices = MLXArray(ids.map(UInt32.init))
            let out = bank(x.reshaped(ids.count, 1, columns), indices, sorted: false)
                .reshaped(ids.count, rows)
            let reference = Self.projectReference(
                x, dense: made.dense, experts: experts, rows: rows, columns: columns, ids: ids)
            // FP32 activations: only the summation order differs.
            #expect(Self.relativeError(out, reference) < 1e-5)
        }

        @Test func nvfp4GlobalScaleUsesTheStoredAmaxExactly() throws {
            let made = Self.nvfp4Bank(experts: 2, rows: 16, columns: 32, seed: 1)
            let converted = try Inkling2Checkpoint.convertExperts(
                Self.checkpoint("w13", made.tensors), configuration: Inkling2ModelTests.config(),
                layer: "test")
            let global = converted["w13.global_scale"]!
            #expect(global.dtype == .float32)
            #expect(arrayEqual(global, made.tensors["weight_amax"]!).item(Bool.self))
            #expect(converted["w13.weight"]!.dtype == .uint32)
            #expect(converted["w13.weight"]!.shape == [2, 16, 4])
            #expect(converted["w13.scales"]!.shape == [2, 16, 2])
        }

        // MARK: Routed MLP

        @Test func moeMatchesTheInterleavedReference() throws {
            let c = Inkling2ModelTests.config()
            let moe = Inkling2SparseMoE(c, format: .dense)
            SyntheticModel.randomize(moe, seed: 5)
            let tokens = 5
            let x = MLXRandom.normal([1, tokens, c.hiddenSize], key: MLXRandom.key(9))
            let out = moe(x)

            // Reference: transformers Interleave(dim=1) + chunk(2), per token and expert.
            let logits = matmul(x.reshaped(tokens, -1), moe.router.weight.T)
            let scores = sigmoid(logits)
            let selection = (scores + moe.router.bias).asArray(Float.self)
            let scoreValues = scores.asArray(Float.self)
            let w13 = (moe.experts.gateUp as! Inkling2DenseExpertBank).weight
            let w2 = (moe.experts.down as! Inkling2DenseExpertBank).weight
            let i = c.moeIntermediateSize
            let stored = w13.reshaped(c.experts, i, 2, c.hiddenSize)
            let gateRows = stored[0..., 0..., 0, 0...]
            let upRows = stored[0..., 0..., 1, 0...]
            var rows: [MLXArray] = []
            for t in 0 ..< tokens {
                let s = Array(selection[(t * c.experts) ..< ((t + 1) * c.experts)])
                let top = s.indices.sorted { s[$0] > s[$1] }.prefix(c.expertsPerToken)
                let picked = top.map { scoreValues[t * c.experts + $0] }
                let total = picked.reduce(0, +) + 1e-20
                var acc = MLXArray.zeros([c.hiddenSize])
                for (e, score) in zip(top, picked) {
                    let xt = x[0, t]
                    let gate = matmul(gateRows[e], xt)
                    let up = matmul(upRows[e], xt)
                    let act = tanh(gate / 4) * 4 * sigmoid(gate) * tanh(up / 25) * 25
                    acc = acc + matmul(w2[e], act) * (score / total * c.routeScale)
                }
                rows.append(acc)
            }
            let reference = stacked(rows).reshaped(1, tokens, -1)
            #expect(Self.relativeError(out, reference) < 1e-5)
        }

        // MARK: Short convolutions

        @Test func shortConvolutionMatchesConv1dAndCarriesState() throws {
            let (channels, kernel, length) = (12, 4, 9)
            let conv = Inkling2ShortConvolution(channels: channels, kernel: kernel)
            SyntheticModel.randomize(conv, seed: 2)
            let x = MLXRandom.normal([2, length, channels], key: MLXRandom.key(4))
            let (whole, _) = conv(x, state: nil)
            let weight = conv.weight.transposed(0, 2, 1)
            let reference =
                (conv1d(x, weight, padding: kernel - 1, groups: channels)[0..., 0 ..< length, 0...]
                    + x)
                / sqrt(Float(kernel + 1))
            #expect(Self.relativeError(whole, reference) < 1e-6)

            var state: MLXArray? = nil
            var pieces: [MLXArray] = []
            for range in [0 ..< 1, 1 ..< 4, 4 ..< 5, 5 ..< 9] {
                let (y, next) = conv(x[0..., range, 0...], state: state)
                pieces.append(y)
                state = next
            }
            #expect(arrayEqual(concatenated(pieces, axis: 1), whole).item(Bool.self))
        }

        @Test func dynamicConvolutionMatchesTheUnfoldFormula() throws {
            let (b, l, heads, perHead, input, kernel) = (2, 6, 2, 3, 5, 4)
            let channels = heads * perHead
            let conv = Inkling2DynamicShortConvolution(
                channels: channels, inputDimensions: input, heads: heads, kernel: kernel)
            SyntheticModel.randomize(conv, seed: 8)
            let x = MLXRandom.normal([b, l, channels], key: MLXRandom.key(11))
            let d = MLXRandom.normal([b, l, input], key: MLXRandom.key(12))
            let (whole, _) = conv(x, dynamicInput: d, state: nil)

            let xs = x.asArray(Float.self)
            let staticW = conv.staticWeight.asArray(Float.self)
            let dyn = matmul(d, conv.weightProjection.T).asArray(Float.self)
            var expected = [Float](repeating: 0, count: b * l * channels)
            for bi in 0 ..< b {
                for t in 0 ..< l {
                    for ch in 0 ..< channels {
                        var acc: Float = 0
                        for lag in 0 ..< kernel where t - lag >= 0 {
                            let tap =
                                staticW[lag * channels + ch]
                                + dyn[(bi * l + t) * heads * kernel + (ch / perHead) * kernel + lag]
                            acc += tap * xs[(bi * l + t - lag) * channels + ch]
                        }
                        let i = (bi * l + t) * channels + ch
                        expected[i] = (acc + xs[i]) / sqrt(Float(2 * kernel + 1))
                    }
                }
            }
            #expect(Self.relativeError(whole, MLXArray(expected, [b, l, channels])) < 1e-5)

            let (first, state) = conv(
                x[0..., 0 ..< 2, 0...], dynamicInput: d[0..., 0 ..< 2, 0...], state: nil)
            let (second, _) = conv(
                x[0..., 2..., 0...], dynamicInput: d[0..., 2..., 0...], state: state)
            #expect(
                Self.relativeError(concatenated([first, second], axis: 1), whole) < 1e-6)
        }

        // MARK: Attention

        @Test(arguments: [nil, 3] as [Int?])
        func referenceAttentionMatchesTheHostLoop(window: Int?) throws {
            let (h, kvh, l, d, extent) = (4, 2, 6, 8, 4)
            let q = MLXRandom.normal([1, h, l, d], key: MLXRandom.key(1))
            let k = MLXRandom.normal([1, kvh, l, d], key: MLXRandom.key(2))
            let v = MLXRandom.normal([1, kvh, l, d], key: MLXRandom.key(3))
            let rel = MLXRandom.normal([1, h, l, extent], key: MLXRandom.key(4))
            let sinks = MLXRandom.normal([1, h, l], key: MLXRandom.key(5))
            let positions = MLXArray((10 ..< 10 + l).map { Int32($0) }).reshaped(1, l)
            let scale: Float = 0.3
            let inputs = Inkling2AttentionInputs(
                queries: q, keys: k, values: v, relativeLogits: rel,
                relativeStates: MLXArray.zeros([1, l, h, 2]),
                relativeProjection: MLXArray.zeros([2, extent]),
                sinks: sinks, positions: positions, window: window, scale: scale)
            let out = Inkling2ReferenceAttention.attend(
                inputs, keys: k, values: v, keyPositions: positions)

            let qs = q.asArray(Float.self)
            let ks = k.asArray(Float.self)
            let vs = v.asArray(Float.self)
            let rs = rel.asArray(Float.self)
            let ss = sinks.asArray(Float.self)
            var expected = [Float](repeating: 0, count: l * h * d)
            for head in 0 ..< h {
                let kvHead = head / (h / kvh)
                for i in 0 ..< l {
                    var logits: [Float] = []
                    for j in 0 ... i {
                        let distance = i - j
                        if let window, distance >= window {
                            logits.append(-.infinity)
                            continue
                        }
                        var dot: Float = 0
                        for e in 0 ..< d {
                            dot += qs[(head * l + i) * d + e] * ks[(kvHead * l + j) * d + e]
                        }
                        let bias = distance < extent ? rs[(head * l + i) * extent + distance] : 0
                        logits.append(dot * scale + bias)
                    }
                    let sink = ss[head * l + i]
                    let m = max(logits.max()!, sink)
                    let exps = logits.map { exp($0 - m) }
                    let total = exps.reduce(0, +) + exp(sink - m)
                    for e in 0 ..< d {
                        var acc: Float = 0
                        for (j, w) in exps.enumerated() {
                            acc += w / total * vs[(kvHead * l + j) * d + e]
                        }
                        expected[(i * h + head) * d + e] = acc
                    }
                }
            }
            #expect(Self.relativeError(out, MLXArray(expected, [1, l, h * d])) < 1e-5)
        }
    }
}
