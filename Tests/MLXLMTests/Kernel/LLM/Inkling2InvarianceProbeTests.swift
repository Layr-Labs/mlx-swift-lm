import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

/// Chunk invariance of Inkling 2's prefill ops on this device: for each op,
/// is a token's result the same when it is computed in a chunk of M tokens
/// (the first M of 2,048) as in the full 2,048-token chunk, and with other
/// tokens beside it? Prints a table with INKLING2_INVARIANCE_PROBE=1; asserts
/// nothing (the invariant implementations are asserted by their own tests).
extension KernelTests {
    @Suite(.serialized)
    struct Inkling2InvarianceProbeTests {
        static let lengths = [
            1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256,
            257, 511, 512, 513, 1000, 1024, 1500, 2047, 2048,
        ]

        static func bitsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && arrayEqual(a, b).item(Bool.self)
        }

        /// Lengths whose first rows differ from the full chunk's, and whether
        /// replacing the other rows (at M = 512) changes the first rows.
        static func probe(
            _ name: String, rows: Int = 2048, _ op: (MLXArray) -> MLXArray, input: MLXArray
        )
            -> String
        {
            let full = op(input)
            eval(full)
            var variant: [Int] = []
            for m in lengths where m <= rows {
                let part = op(input[0 ..< m])
                if !bitsEqual(part, full[0 ..< m]) { variant.append(m) }
            }
            let other = concatenated([input[0 ..< 256], input[1024 ..< 1280]], axis: 0)
            let company = bitsEqual(op(other)[0 ..< 256], op(input[0 ..< 512])[0 ..< 256])
            return
                "\(name): M-variant at \(variant.isEmpty ? "none" : "\(variant)"); company-invariant \(company)"
        }

        @Test func reportChunkInvariance() throws {
            guard ProcessInfo.processInfo.environment["INKLING2_INVARIANCE_PROBE"] == "1" else {
                return
            }
            // Fixed-row prefill products on (as an invariant model's steps run).
            try Inkling2RowExact.withPrefill(blockRows: 64) { try report() }
        }

        func report() throws {
            var lines: [String] = []
            let x = (MLXRandom.normal([2048, 2304], key: MLXRandom.key(1)) * 2).asType(.bfloat16)
            let x3 = (MLXRandom.normal([2048, 3072], key: MLXRandom.key(2))).asType(.bfloat16)
            let x46 = (MLXRandom.normal([2048, 4608], key: MLXRandom.key(3))).asType(.bfloat16)
            eval(x, x3, x46)
            // Dense BF16 products as prefill runs them (Inkling2RowExact inactive).
            for (name, n, input) in [
                ("dense q|r|sink|gate 2304->6552", 6552, x), ("dense shared 2304->8600", 8600, x),
                ("dense k|taps 2304->1056", 1056, x), ("dense wr 2304->384", 384, x),
                ("dense wk 2304->1024", 1024, x), ("dense wo 3072->2304", 2304, x3),
                ("dense message down 4608->64", 64, x46),
            ] {
                let w =
                    (MLXRandom.normal([n, input.dim(1)], key: MLXRandom.key(UInt64(n)))
                    * 0.02).asType(.bfloat16)
                eval(w)
                lines.append(
                    Self.probe(name + " (MLX as called)", { MLX.matmul($0, w.T) }, input: input))
                for block in [64, 512] {
                    lines.append(
                        Self.probe(
                            name + " (\(block)-row calls)",
                            {
                                Inkling2RowExact.prefillMatmul(
                                    $0, transposedWeight: w, blockRows: block)
                            }, input: input))
                }
                // Equal bits from 64- and 512-row calls would let short chunks
                // take the small calls without losing invariance.
                let blocks = [64, 512].map {
                    Inkling2RowExact.prefillMatmul(input, transposedWeight: w, blockRows: $0)
                }
                lines.append(
                    "\(name): 64-row calls == 512-row calls \(Self.bitsEqual(blocks[0], blocks[1]))"
                )
            }
            // Row reductions.
            let xf = x.asType(.float32)
            eval(xf)
            let weight = MLXRandom.normal([2304], key: MLXRandom.key(9)) * 0.1 + 1
            eval(weight)
            for fused in [false, true] {
                Inkling2Fused.$override.withValue(fused) {
                    lines.append(
                        Self.probe(
                            "unweighted RMS norm (fused \(fused))",
                            { Inkling2Math.unweightedRMSNorm($0) }, input: xf))
                    lines.append(
                        Self.probe(
                            "l2 clip (fused \(fused))",
                            { Inkling2Math.l2Clip($0.reshaped($0.dim(0), 18, 128), threshold: 4) },
                            input: xf))
                    let norm = Inkling2RMSNorm(dimensions: 2304, eps: 1e-6)
                    norm.update(parameters: ModuleParameters.unflattened(["weight": weight]))
                    lines.append(
                        Self.probe("weighted RMS norm (fused \(fused))", { norm($0) }, input: x))
                }
            }
            // Routed experts with Inkling 2 Mini's shapes and random NVFP4 banks.
            var base = Inkling2ModelTests.base
            var text = base["text_config"] as! [String: Any]
            text["hidden_size"] = 2304
            text["n_routed_experts"] = 128
            text["num_experts_per_tok"] = 8
            text["intermediate_size"] = 1152
            base["text_config"] = text
            let config = try SyntheticModel.configuration(Inkling2Configuration.self, base).text
            let moe = Inkling2SparseMoE(config, format: .nvfp4)
            SyntheticModel.randomize(moe, seed: 4)
            var banks: [(String, MLXArray)] = []
            for (bank, k, n) in [("w13", 2304, 2304), ("w2", 1152, 2304)] {
                banks.append(
                    (
                        "experts.\(bank).weight",
                        (MLXRandom.uniform(
                            Float(0) ..< Float(1), [128, n, k / 8], key: MLXRandom.key(5))
                            * 4_294_967_000).asType(.uint32)
                    ))
                banks.append(
                    (
                        "experts.\(bank).scales",
                        (MLXRandom.uniform(
                            Float(0) ..< Float(1), [128, n, k / 16], key: MLXRandom.key(6))
                            * 8 + 48).asType(.uint8)
                    ))
            }
            moe.update(parameters: ModuleParameters.unflattened(banks))
            eval(moe)
            let tokens = (MLXRandom.normal([2048, 2304], key: MLXRandom.key(7)) * 0.5).asType(
                .bfloat16)
            eval(tokens)
            lines.append(
                Self.probe(
                    "routed experts, MLX routes as called (NVFP4)",
                    { moe($0.expandedDimensions(axis: 0))[0] }, input: tokens))
            lines.append(
                Self.probe(
                    "routed experts, prefill entry (short chunks padded)",
                    {
                        Inkling2DecoderLayer.invariantPrefillExperts(
                            moe, $0.expandedDimensions(axis: 0))[0]
                    }, input: tokens))
            print("INKLING2_INVARIANCE_PROBE\n" + lines.joined(separator: "\n"))
        }
    }
}
