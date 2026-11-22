import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@_spi(GemmaEncoder) @testable import MLXLLM

final class Gemma4FusionTests: XCTestCase {
    private func assertIdentical(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self).map(\.bitPattern),
            expected.asType(.float32).asArray(Float.self).map(\.bitPattern),
            file: file, line: line)
    }

    func testSoftcapPreservesScalarPromotionAndRuntimeCaps() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let x = MLX.linspace(Float(-100), Float(100), count: 4096).reshaped(1, 4096)
                .asType(dtype)
            for value: Float in [30, 7.25, 1] {
                for cap in [MLXArray(value), value.asMLXArray(dtype: dtype)] {
                    assertIdentical(gemma4LogitSoftcap(x, cap), tanh(x / cap) * cap)
                    let strided = x[.ellipsis, .stride(by: 2)]
                    assertIdentical(
                        gemma4LogitSoftcap(strided, cap), tanh(strided / cap) * cap)
                }
            }
        }
    }

    func testGradientsMatchOriginal() {
        let cap = MLXArray(Float(30))
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLX.linspace(Float(-3), Float(3), count: 64)
                .reshaped(1, 1, 64).asType(dtype)
            assertIdentical(
                grad({ sum(gemma4LogitSoftcap($0, cap)) })(logits),
                grad({ sum(tanh($0 / cap) * cap) })(logits))
        }
    }

    func testScalarSoftcapPreservesDtypeAndRuntimeCaps() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLX.linspace(Float(-100), Float(100), count: 4096)
                .reshaped(2, 2, 1024).asType(dtype)
            for cap: Float in [30, 7.25, 30.1, 1, -30] {
                assertIdentical(gemma4LogitSoftcap(logits, cap), tanh(logits / cap) * cap)
                let strided = logits[.ellipsis, .stride(by: 2)]
                assertIdentical(gemma4LogitSoftcap(strided, cap), tanh(strided / cap) * cap)
            }
        }
    }

    func testScalarSoftcapPreservesEveryFiniteLowPrecisionValue() {
        for dtype in [DType.float16, .bfloat16] {
            let exponentMask: UInt16 = dtype == .float16 ? 0x7c00 : 0x7f80
            let bits = (0 ... 65535).map { UInt16($0) }.filter {
                $0 & exponentMask != exponentMask
            }
            let logits = MLXArray(bits).view(dtype: dtype)
            for cap: Float in [30, 7.25, 30.1, 1, -30] {
                assertIdentical(gemma4LogitSoftcap(logits, cap), tanh(logits / cap) * cap)
            }
        }
    }

    func testScalarSoftcapGradientsMatchOriginal() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLX.linspace(Float(-3), Float(3), count: 64)
                .reshaped(1, 1, 64).asType(dtype)
            let cap: Float = 30
            assertIdentical(
                grad({ sum(gemma4LogitSoftcap($0, cap)) })(logits),
                grad({ sum(tanh($0 / cap) * cap) })(logits))
        }
    }

    func testScalarSoftcapPreservesSpecialValues() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLXArray([-Float.infinity, -100, -0.0, 0.0, 100, .infinity, .nan])
                .asType(dtype)
            for cap: Float in [30, -30, 0] {
                let actual = gemma4LogitSoftcap(logits, cap)
                let expected = tanh(logits / cap) * cap
                XCTAssertEqual(actual.dtype, expected.dtype)
                for (a, b) in zip(
                    actual.asType(.float32).asArray(Float.self),
                    expected.asType(.float32).asArray(Float.self))
                {
                    if b.isNaN {
                        XCTAssertTrue(a.isNaN)
                    } else {
                        XCTAssertEqual(a.bitPattern, b.bitPattern)
                    }
                }
            }
        }
    }

    func testScalarSoftcapOnCPU() {
        Stream.withNewDefaultStream(device: .cpu) {
            for dtype in [DType.float16, .bfloat16, .float32] {
                let logits = MLX.linspace(Float(-100), Float(100), count: 128).asType(dtype)
                let cap: Float = 30
                assertIdentical(gemma4LogitSoftcap(logits, cap), tanh(logits / cap) * cap)
                let tensorCap = MLXArray(cap)
                assertIdentical(
                    gemma4LogitSoftcap(logits, tensorCap), tanh(logits / tensorCap) * tensorCap)
            }
        }
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLX.linspace(Float(-100), Float(100), count: 128).asType(dtype)
            let cap: Float = 30
            assertIdentical(gemma4LogitSoftcap(logits, cap), tanh(logits / cap) * cap)
            let tensorCap = MLXArray(cap)
            assertIdentical(
                gemma4LogitSoftcap(logits, tensorCap), tanh(logits / tensorCap) * tensorCap)
        }
    }

    func testFusionLatency() throws {
        guard ProcessInfo.processInfo.environment["MLX_BENCHMARK_GEMMA4_FUSIONS"] == "1" else {
            throw XCTSkip("Set MLX_BENCHMARK_GEMMA4_FUSIONS=1 to benchmark")
        }
        let logits = MLXRandom.normal([1, 1, 262144]).asType(.bfloat16)
        let cap = MLXArray(Float(30))
        eval(logits, cap)
        let cases: [(String, () -> MLXArray, () -> MLXArray)] = [
            ("VLM softcap", { tanh(logits / cap) * cap }, { gemma4LogitSoftcap(logits, cap) }),
            (
                "text softcap", { tanh(logits / Float(30)) * Float(30) },
                { gemma4LogitSoftcap(logits, Float(30)) }
            ),
        ]
        for (name, original, fused) in cases {
            assertIdentical(fused(), original())
            for _ in 0 ..< 10 { eval(original(), fused()) }
            var samples = [[Double]](repeating: [], count: 2)
            for round in 0 ..< 10 {
                for arm in (round.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                    let body = arm == 0 ? original : fused
                    let start = ProcessInfo.processInfo.systemUptime
                    for _ in 0 ..< 100 { eval(body()) }
                    samples[arm].append((ProcessInfo.processInfo.systemUptime - start) * 10000)
                }
            }
            let medians = samples.map { values in
                let sorted = values.sorted()
                return (sorted[4] + sorted[5]) / 2
            }
            print("[Gemma4 fusion] \(name): original \(medians[0]) µs, fused \(medians[1]) µs")
        }
    }

    func testFusedQKVParityZeroExtraBytesAndInvalidation() throws {
        MLXRandom.seed(2026)
        let json = """
            {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 2,
                "head_dim": 32,
                "global_head_dim": 64,
                "vocab_size": 256,
                "rms_norm_eps": 1e-6,
                "sliding_window": 64,
                "layer_types": ["sliding_attention", "full_attention"],
                "attention_k_eq_v": true,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "tie_word_embeddings": true
            }
            """
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: Data(json.utf8))
        let unfused = Gemma4TextModel(config)
        let fused = Gemma4TextModel(config)
        quantize(model: unfused, groupSize: 64, bits: 8)
        quantize(model: fused, groupSize: 64, bits: 8)
        // Mirror identical quantized weights into both models
        try fused.update(parameters: unfused.parameters(), verify: .all)
        eval(unfused, fused)

        let qkvBytesBefore = fused.parameters().flattened()
            .filter {
                $0.0.contains(".self_attn.")
                    && ($0.0.contains("q_proj") || $0.0.contains("k_proj")
                        || $0.0.contains("v_proj"))
            }
            .reduce(0) { $0 + $1.1.nbytes }
        let embedWeightBytesBefore =
            (fused.model.embedTokens as? QuantizedEmbedding)?.weight.nbytes ?? 0
        try fused.prepare()
        eval(fused)
        let qkvBytesAfter = fused.parameters().flattened()
            .filter {
                $0.0.contains(".self_attn.")
                    && ($0.0.contains("q_proj") || $0.0.contains("k_proj")
                        || $0.0.contains("v_proj"))
            }
            .reduce(0) { $0 + $1.1.nbytes }
        let hiLo = fused.model.embedTokens as? Gemma4Hi4Lo4Embedding

        // Both layer 0 (sliding 3-way Q+K+V) and layer 1 (full 2-way Q+K, k_eq_v) fuse cleanly.
        XCTAssertTrue(fused.model.layers[0].selfAttn.hasFusedQKV)
        XCTAssertTrue(fused.model.layers[1].selfAttn.hasFusedQKV)
        // Zero extra weight bytes for both Fused QKV and Hi4/Lo4 bit-plane embedding.
        XCTAssertEqual(qkvBytesAfter, qkvBytesBefore)
        if let hiLo {
            XCTAssertEqual(hiLo.hiWeight.nbytes + hiLo.loWeight.nbytes, embedWeightBytesBefore)
        }

        // Verify parity across B=1, 2, 4, 8, 16, 32 at L=1 (decode) and B=1 at L=16 (prefill).
        for (B, L) in [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (1, 16)] {
            let x = MLXRandom.normal([B, L, 128]).asType(.bfloat16)
            for li in 0 ..< 2 {
                let (qU, kU, vU) = unfused.model.layers[li].selfAttn.projectQKV(
                    x, useBatchDecodeProjection: true)
                let (qF, kF, vF) = fused.model.layers[li].selfAttn.projectQKV(
                    x, useBatchDecodeProjection: true)
                eval(qU, kU, vU, qF, kF, vF)
                let qErr = MLX.abs(qF.asType(.float32) - qU.asType(.float32)).max().item(Float.self)
                let kErr = MLX.abs(kF.asType(.float32) - kU.asType(.float32)).max().item(Float.self)
                let vErr = MLX.abs(vF.asType(.float32) - vU.asType(.float32)).max().item(Float.self)
                // B <= 8 is 0-ULP bit-exact; B=16/32 allows <= 1e-2 BF16 qmm split-K tolerance.
                let tol: Float = (B <= 8 && L == 1) ? 0.0 : 1e-2
                XCTAssertLessThanOrEqual(qErr, tol, "Layer \(li) B=\(B) L=\(L) Q mismatch: \(qErr)")
                XCTAssertLessThanOrEqual(kErr, tol, "Layer \(li) B=\(B) L=\(L) K mismatch: \(kErr)")
                XCTAssertLessThanOrEqual(vErr, tol, "Layer \(li) B=\(B) L=\(L) V mismatch: \(vErr)")
            }
        }

        // Verify parameter update invalidates fusedQKV and restores QuantizedEmbedding cleanly
        // both at model root and via direct Gemma4DecoderLayer update (LoRA / weight reload).
        try fused.update(parameters: unfused.parameters(), verify: .noUnusedKeys)
        XCTAssertFalse(fused.model.layers[0].selfAttn.hasFusedQKV)
        XCTAssertFalse(fused.model.layers[1].selfAttn.hasFusedQKV)
        XCTAssertTrue(fused.model.embedTokens is QuantizedEmbedding)

        try fused.prepare()
        XCTAssertTrue(fused.model.layers[0].selfAttn.hasFusedQKV)
        try fused.model.layers[0].update(
            parameters: unfused.model.layers[0].parameters(), verify: .all)
        XCTAssertFalse(fused.model.layers[0].selfAttn.hasFusedQKV)
        XCTAssertTrue(fused.model.layers[1].selfAttn.hasFusedQKV)
    }
}
