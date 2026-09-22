import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite("DiffusionGemma pinned block oracle", .serialized)
struct DiffusionGemmaBlockOracleTests {
    struct Fixture: Decodable {
        let reference: String
        let model_config: DiffusionGemmaTextConfiguration
        let cases: [Case]
        struct Case: Decodable {
            let name: String
            let layer: Int
            let decoder: Bool
            let length: Int
            let prefix: Int
            let encoder_scalar: Float?
        }
    }

    @Test func encoderAndCanvasBlocksMatchFrozenReferenceExactly() throws {
        let metadataURL = try #require(
            Bundle.module.url(
                forResource: "diffusiongemma-block-oracle", withExtension: "json"))
        let tensorURL = try #require(
            Bundle.module.url(
                forResource: "diffusiongemma-block-oracle", withExtension: "safetensors"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: metadataURL))
        #expect(fixture.reference == "e79b0e041677ec4ca5333ba750376bb4e8c434cb")
        let arrays = try loadArrays(url: tensorURL)
        #expect(fixture.cases.count == 8)
        let tracePath = ProcessInfo.processInfo.environment["MLX_DIFFUSION_ORACLE_TRACE"]
        if let tracePath { #expect(!FileManager.default.fileExists(atPath: tracePath)) }
        var traces = [String: MLXArray]()
        for item in fixture.cases {
            let block = DiffusionGemmaTextBlock(fixture.model_config, layer: item.layer)
            let prefix = "layer\(item.layer)."
            let parameters = arrays.filter { $0.key.hasPrefix(prefix) }.map {
                (String($0.key.dropFirst(prefix.count)), $0.value)
            }
            try block.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
            let input = try #require(arrays[item.name + ".input"])
            let expected = try #require(arrays[item.name + ".expected"])
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                arrays[item.name + ".mask"].map { .array($0) } ?? .none
            let scalar = item.encoder_scalar.map { MLXArray([$0]) }
            let keyValues: (MLXArray, MLXArray) -> (MLXArray, MLXArray) = {
                keys, values in
                guard item.decoder, item.prefix > 0 else { return (keys, values) }
                var prefixKeys = arrays[item.name + ".prefix_keys"]!
                var prefixValues = arrays[item.name + ".prefix_values"]!
                if item.layer == 0, item.prefix > fixture.model_config.slidingWindow - 1 {
                    let start = item.prefix - fixture.model_config.slidingWindow + 1
                    prefixKeys = prefixKeys[.ellipsis, start..., 0...]
                    prefixValues = prefixValues[.ellipsis, start..., 0...]
                }
                return (
                    concatenated([prefixKeys, keys], axis: 2),
                    concatenated([prefixValues, values], axis: 2)
                )
            }
            let actual = block(
                input, position: item.prefix, mask: mask, encoderScalar: scalar,
                keyValues: keyValues)
            if tracePath != nil {
                let normalized = block.inputNorm(input)
                let a = block.attention
                let qProjected = a.query(normalized).reshaped(
                    1, item.length, a.heads, a.headDimension)
                let qNormalized = a.queryNorm(qProjected).transposed(0, 2, 1, 3)
                let qRotated = a.rope(qNormalized, offset: item.prefix)
                let kProjected = a.key(normalized).reshaped(
                    1, item.length, a.kvHeads, a.headDimension)
                let kRotated = a.rope(
                    a.keyNorm(kProjected).transposed(0, 2, 1, 3), offset: item.prefix)
                let rawValues =
                    a.value.map {
                        $0(normalized).reshaped(1, item.length, a.kvHeads, a.headDimension)
                    } ?? kProjected
                let vNormalized = a.valueNorm(rawValues).transposed(0, 2, 1, 3)
                let (kAll, vAll) = keyValues(kRotated, vNormalized)
                let sdpa = MLXFast.scaledDotProductAttention(
                    queries: qRotated, keys: kAll, values: vAll,
                    scale: 1, mask: mask)
                for (name, value) in [
                    ("q_projected", qProjected), ("q_normalized", qNormalized),
                    ("q_rotated", qRotated), ("k_projected", kProjected),
                    ("k_rotated", kRotated), ("values", vNormalized), ("sdpa", sdpa),
                ] {
                    traces[item.name + "." + name] = value
                }
                let attention = block.attention(
                    normalized, position: item.prefix, mask: mask, keyValues: keyValues)
                let residual = input + block.postAttentionNorm(attention)
                let dense = block.denseNorm(block.mlp(block.preFFN(residual)))
                let flat = residual.reshaped(-1, residual.dim(-1))
                let routed = block.router(flat)
                let expertInput = block.preExpertNorm(flat)
                let expert = block.experts(
                    expertInput, indices: routed.indices, weights: routed.weights
                )
                .reshaped(residual.shape)
                let combined = block.postFFN(dense + block.expertNorm(expert))
                let output = (residual + combined) * (scalar ?? block.layerScalar)
                for (name, value) in [
                    ("input_norm", normalized), ("attention", attention), ("residual", residual),
                    ("dense", dense), ("routing_indices", routed.indices),
                    ("routing_weights", routed.weights),
                    ("expert_input", expertInput), ("expert", expert), ("combined", combined),
                    ("output", output),
                ] { traces[item.name + "." + name] = value }
                #expect(
                    output.asArray(Float.self).map(\.bitPattern)
                        == actual.asArray(Float.self).map(\.bitPattern),
                    "Diagnostic replay must preserve the native output")
            }
            eval(actual, expected)
            #expect(actual.shape == expected.shape)
            #expect(actual.dtype == expected.dtype)
            let lhs = actual.asArray(Float.self).map(\.bitPattern)
            let rhs = expected.asArray(Float.self).map(\.bitPattern)
            #expect(
                lhs == rhs,
                "\(item.name): raw FP32 bits differ; preserve the oracle and isolate the first stage"
            )
        }
        if let tracePath {
            let url = URL(fileURLWithPath: tracePath)
            guard !FileManager.default.fileExists(atPath: tracePath) else {
                throw CocoaError(.fileWriteFileExists)
            }
            try save(arrays: traces, url: url)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: tracePath)
        }
    }
}
