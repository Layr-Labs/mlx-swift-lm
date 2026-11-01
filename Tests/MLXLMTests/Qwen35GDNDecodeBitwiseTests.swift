// Copyright © 2026 Apple Inc.
//
// Pins `decodeConv` bit-for-bit against `generalConv` (MLX's `Convolution`
// kernel); a mismatch means compiled decode silently diverges from prefill.
//
// Fork: MLXLLM `Qwen35GatedDeltaNet.decodeConv` is the conv step that
// `processChunk` runs for every chunk, including the one-token decode step
// (fork.yaml section "Qwen 3.5 / 3.6 / 3.8 (hybrid DeltaNet MoE, MTP)").
// Upstream's library `generalConv` reference is defined below in this file.
// `testCachedDecodeStepConvStateMatchesConvolutionKernelBitwise` also checks
// the conv state that a cached one-token forward writes to its `MambaCache`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35GDNDecodeBitwiseTests: XCTestCase {

    /// The f32 upcast is injective on finite f16/bf16/f32 values, so
    /// bit-comparing the upcast compares the originals.
    private func assertBitIdentical(
        _ got: MLXArray, _ want: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.dtype, want.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(got.shape, want.shape, "\(label): shape", file: file, line: line)
        let a = got.asType(.float32).asArray(Float.self)
        let b = want.asType(.float32).asArray(Float.self)
        let mismatches = zip(a, b).lazy.filter { $0.bitPattern != $1.bitPattern }.count
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches)/\(a.count) elements differ bitwise",
            file: file, line: line)
    }

    /// 256 conv channels — wide enough that an accumulation-order change in
    /// MLX's Convolution kernel cannot pass by coincidence (native-dtype
    /// accumulation diverges in ~47% of channels).
    private func tinyGDNConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 16,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    func testDecodeConvMatchesConvolutionKernelBitwise() throws {
        let config = try tinyGDNConfiguration()
        for dtype in [DType.float16, DType.bfloat16] {
            withRandomState(MLXRandom.RandomState(seed: 11)) {
                let gdn = Qwen35GatedDeltaNet(config)
                gdn.conv1d.update(
                    parameters: gdn.conv1d.parameters().mapValues { $0.asType(dtype) })

                let qkv = MLXRandom.normal([1, 1, gdn.convDim]).asType(dtype)
                let convState = MLXRandom.normal(
                    [1, gdn.convKernelSize - 1, gdn.convDim]
                ).asType(dtype)
                eval(qkv, convState)

                let (conv, state) = gdn.decodeConv(convState: convState, qkv: qkv)
                let (refConv, refState) = gdn.generalConv(convState: convState, qkv: qkv)
                eval(conv, state, refConv, refState)

                assertBitIdentical(conv, refConv, "conv output (\(dtype))")
                assertBitIdentical(state, refState, "conv state (\(dtype))")
            }
        }
    }

    func testCachedDecodeStepConvStateMatchesConvolutionKernelBitwise() throws {
        let config = try tinyGDNConfiguration()
        for dtype in [DType.float16, DType.bfloat16] {
            let (gdn, input, convState) = withRandomState(MLXRandom.RandomState(seed: 13)) {
                let gdn = Qwen35GatedDeltaNet(config)
                gdn.update(parameters: gdn.parameters().mapValues { $0.asType(dtype) })
                let input = MLXRandom.normal([1, 1, config.hiddenSize]).asType(dtype)
                let convState = MLXRandom.normal(
                    [1, gdn.convKernelSize - 1, gdn.convDim]
                ).asType(dtype)
                eval(input, convState)
                return (gdn, input, convState)
            }

            let qkv = gdn.projectInputs(input, B: 1, S: 1).qkv
            let (_, refState) = gdn.generalConv(convState: convState, qkv: qkv)
            eval(refState)

            // The one-token forward runs `processChunk` and stores its conv
            // state in slot 0 of the cache.
            let cache = MambaCache()
            cache[0] = convState
            eval(gdn(input, cache: cache))
            let state = try XCTUnwrap(cache[0])
            eval(state)

            assertBitIdentical(state, refState, "cached decode conv state (\(dtype))")
        }
    }
}

extension Qwen35GatedDeltaNet {
    /// The sliding-window conv via MLX's `Convolution` kernel: the reference
    /// that `decodeConv` is pinned against. Upstream keeps it in the library;
    /// the fork's GDN has one conv path, so the reference lives with the test.
    fileprivate func generalConv(
        convState: MLXArray, qkv: MLXArray
    ) -> (conv: MLXArray, state: MLXArray) {
        let convInput = concatenated([convState, qkv], axis: 1)
        return (
            conv1d(convInput),
            contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
        )
    }
}
