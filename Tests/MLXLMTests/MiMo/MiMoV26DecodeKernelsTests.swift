// Copyright © 2026 Eigen Labs.
// Native tests for the source-only #3990 residual/norm port. They are NOT run
// by preparing this packet and require the coordinator's exclusive native lane.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class MiMoV26DecodeEligibilityTests: XCTestCase {
    func testNarrowShapeDTypeAndDeviceContract() {
        for rows in 1 ... 7 {
            for type: DType in [.bfloat16, .float16] {
                XCTAssertTrue(
                    MiMoV26DecodeKernels.supports(
                        shape: [1, rows, 4096], dtype: type, device: .gpu))
            }
        }
        for shape in [
            [1, 0, 4096], [1, 8, 4096], [2, 1, 4096], [1, 1, 4097],
            [1, 1, 8192], [4096], [1, 1, 0], [1, 1, 3],
        ] {
            XCTAssertFalse(
                MiMoV26DecodeKernels.supports(
                    shape: shape, dtype: .bfloat16, device: .gpu))
        }
        XCTAssertFalse(
            MiMoV26DecodeKernels.supports(
                shape: [1, 1, 4096], dtype: .float32, device: .gpu))
        XCTAssertFalse(
            MiMoV26DecodeKernels.supports(
                shape: [1, 1, 4096], dtype: .bfloat16, device: .cpu))
    }
}

final class MiMoV26DecodeKernelsTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["MLX_TEST_MIMO_DECODE_KERNELS"] == "1" else {
            throw XCTSkip("Requires explicit native-lane run: MLX_TEST_MIMO_DECODE_KERNELS=1")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The Metal dispatch cannot be qualified by a CPU fallback")
        }
    }

    private func signal(_ shape: [Int], dtype: DType, salt: Int = 0) -> MLXArray {
        let size = shape.reduce(1, *)
        let values = (0 ..< size).map { i in
            sin(Float((i * 13 + salt * 17) % 251)) * Float(1 + i % 5)
        }
        return MLXArray(values, shape).asType(dtype)
    }

    private func norm(_ width: Int, dtype: DType, salt: Int = 1) throws -> RMSNorm {
        let result = RMSNorm(dimensions: width, eps: 1e-6)
        let gamma = (0 ..< width).map { i in 1 + Float((i + salt) % 17 - 8) / 32 }
        try result.update(
            parameters: .unflattened([
                ("weight", MLXArray(gamma).asType(dtype))
            ]), verify: .all)
        return result
    }

    /// Check actual storage bits, plus an explicit maximum-ULP report. This
    /// slice claims zero ULP for every row, so no tolerance is inherited from
    /// the upstream verify-router or attention tests.
    private func exact(
        _ actual: MLXArray, _ expected: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, label, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, label, file: file, line: line)
        eval(actual, expected)
        let values = actual.asType(.float32).asArray(Float.self)
        let reference = expected.asType(.float32).asArray(Float.self)
        XCTAssertTrue(values.allSatisfy(\.isFinite), label, file: file, line: line)
        XCTAssertTrue(reference.allSatisfy(\.isFinite), label, file: file, line: line)
        func ordered(_ bits: UInt32) -> UInt32 {
            bits & 0x8000_0000 == 0 ? bits | 0x8000_0000 : ~bits
        }
        let maxULP: UInt32 =
            zip(values, reference).map { a, b in
                let x = ordered(a.bitPattern)
                let y = ordered(b.bitPattern)
                return x > y ? x - y : y - x
            }.max() ?? 0
        XCTAssertEqual(maxULP, 0, "\(label) maxFloat32ULP=\(maxULP)", file: file, line: line)
        if actual.dtype == .bfloat16 || actual.dtype == .float16 {
            XCTAssertEqual(
                actual.view(dtype: .uint16).asArray(UInt16.self),
                expected.view(dtype: .uint16).asArray(UInt16.self),
                label + " storage bits", file: file, line: line)
        } else {
            XCTAssertEqual(
                values.map(\.bitPattern), reference.map(\.bitPattern),
                label + " storage bits", file: file, line: line)
        }
    }

    func testAddRMSMatchesPinnedPrimitiveIncludingTailAndStrides() throws {
        for type: DType in [.bfloat16, .float16] {
            for width in [4, 128, 4000, 4096] {
                let normalizer = try norm(width, dtype: type)
                for rows in 1 ... 7 {
                    let x = signal([1, width, rows], dtype: type).transposed(0, 2, 1)
                    let y = signal([1, rows, width], dtype: type, salt: 2)
                    let actual = try XCTUnwrap(MiMoV26DecodeKernels.addRMS(x, y, norm: normalizer))
                    let residual = x + y
                    exact(actual.residual, residual, "add \(type) rows\(rows) D\(width)")
                    exact(
                        actual.normalized, normalizer(residual),
                        "norm \(type) rows\(rows) D\(width)")
                }
            }
        }
    }

    func testCombineKeepsFP32ScoresProductsAndIntermediateRounding() throws {
        for type: DType in [.bfloat16, .float16] {
            for rows in 1 ... 7 {
                let width = 4096
                let normalizer = try norm(width, dtype: type)
                let hidden = signal([1, rows, width], dtype: type)
                let experts = signal([1, rows, 8, width], dtype: type, salt: 3)
                // Values deliberately distinguish FP32 routing from BF16.
                let scoreValues: [Float] = (0 ..< (rows * 8)).map { index in
                    let routed: Float = Float(index % 8 + 1) / 37
                    let roundingWitness: Float = 1.0 / 65536
                    return routed + roundingWitness
                }
                let weights = MLXArray(scoreValues, [1, rows, 8])
                let actual = try XCTUnwrap(
                    MiMoV26DecodeKernels.combineRMS(
                        hidden, experts: experts, weights: weights, norm: normalizer))
                let reduced = (experts * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(type)
                let residual = hidden + reduced
                exact(actual.residual, residual, "combine \(type) rows\(rows)")
                exact(actual.normalized, normalizer(residual), "combine norm \(type) rows\(rows)")
            }
        }
    }

    func testUnsupportedAndChangedNormWeightsUseTheReference() throws {
        let x = signal([1, 1, 128], dtype: .bfloat16)
        let normalizer = try norm(128, dtype: .bfloat16)
        let first = try XCTUnwrap(MiMoV26DecodeKernels.addRMS(x, x, norm: normalizer))
        let changed = (0 ..< 128).map { Float(1 + $0 % 9) / 8 }
        try normalizer.update(
            parameters: .unflattened([
                ("weight", MLXArray(changed).asType(.bfloat16))
            ]), verify: .all)
        let second = try XCTUnwrap(MiMoV26DecodeKernels.addRMS(x, x, norm: normalizer))
        exact(second.normalized, normalizer(x + x), "fresh norm weights")
        eval(first.normalized, second.normalized)
        XCTAssertFalse(arrayEqual(first.normalized, second.normalized).item(Bool.self))
        let fp32Norm = try norm(128, dtype: .float32)
        XCTAssertNil(MiMoV26DecodeKernels.addRMS(x, x, norm: fp32Norm))
        XCTAssertNil(
            MiMoV26DecodeKernels.addRMS(
                signal([1, 8, 128], dtype: .bfloat16), signal([1, 8, 128], dtype: .bfloat16),
                norm: normalizer))
        let experts = signal([1, 1, 8, 128], dtype: .bfloat16)
        XCTAssertNil(
            MiMoV26DecodeKernels.combineRMS(
                x, experts: experts, weights: MLXArray(Float(1)), norm: normalizer))
        XCTAssertNil(
            MiMoV26DecodeKernels.combineRMS(
                x, experts: experts, weights: signal([1, 1, 8], dtype: .bfloat16), norm: normalizer)
        )
    }

    private func configuration() throws -> MiMoV26Configuration {
        try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: Data(
                """
                {"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],
                 "hidden_size":128,"intermediate_size":256,"moe_intermediate_size":64,
                 "vocab_size":64,"num_hidden_layers":2,"max_position_embeddings":128,
                 "sliding_window_size":7,"sliding_window":7,"num_nextn_predict_layers":3,
                 "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
                 "partial_rotary_factor":0.334,"attention_value_scale":0.707,
                 "layernorm_epsilon":0.000001,"attention_projection_layout":"split",
                 "moe_router_dtype":"bfloat16","hidden_act":"silu","dtype":"bfloat16",
                 "attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0,
                 "scoring_func":"sigmoid","topk_method":"noaux_tc","n_routed_experts":16,
                 "num_experts_per_tok":8,"n_group":1,"topk_group":1,"norm_topk_prob":true,
                 "n_shared_experts":null,"routed_scaling_factor":null,
                 "num_attention_heads":8,"num_key_value_heads":2,"head_dim":48,"v_head_dim":32,
                 "swa_num_attention_heads":8,"swa_num_key_value_heads":4,"swa_head_dim":48,
                 "swa_v_head_dim":32,"rope_theta":10000000,"swa_rope_theta":10000,
                 "add_full_attention_sink_bias":false,"add_swa_attention_sink_bias":true,
                 "eos_token_id":3,"pad_token_id":0}
                """.utf8))
    }

    private func model(fused: Bool) throws -> MiMoV26TextModel {
        let result = try MiMoV26TextModel(configuration())
        let values = result.parameters().flattened().map { name, array in
            let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 997 }
            let data = (0 ..< array.size).map { i -> Float in
                if name.contains("norm.weight") { return 1 + Float((i + salt) % 11 - 5) / 32 }
                return sin(Float((i * 13 + salt) % 109)) * 0.12
            }
            let dtype: DType = name.hasSuffix("e_score_correction_bias") ? .float32 : .bfloat16
            return (name, MLXArray(data, array.shape).asType(dtype))
        }
        try result.update(parameters: .unflattened(values), verify: .all)
        result.model.useFusedDecodeNorms = fused
        return result
    }

    private func ids(_ start: Int, _ count: Int) -> MLXArray {
        MLXArray((start ..< start + count).map { Int32($0 % 63) }, [1, count])
    }

    private func completeOrdinaryState(_ actual: [KVCache], _ expected: [KVCache]) {
        XCTAssertEqual(actual.count, expected.count)
        for (index, pair) in zip(actual, expected).enumerated() {
            let (a, b) = pair
            XCTAssertEqual(a.offset, b.offset, "layer \(index) offset")
            XCTAssertEqual(a.metaState, b.metaState, "layer \(index) metadata")
            XCTAssertEqual(a.state.count, b.state.count)
            for (i, arrays) in zip(a.state, b.state).enumerated() {
                exact(arrays.0, arrays.1, "layer \(index) complete active state \(i)")
            }
        }
    }

    func testOrdinaryForwardExactGreedyHiddenFeaturesAndCompleteCache() throws {
        let reference = try model(fused: false)
        let candidate = try model(fused: true)
        let refCache = reference.newCache()
        let fastCache = candidate.newCache()
        var position = 0
        // Initial fallback prefill; every short row count; wrap and fallback
        // reentry. The baseline always uses the unmodified router/attention.
        for count in [9, 1, 2, 3, 4, 5, 6, 7, 8, 1] {
            let tokens = ids(position, count)
            let expected = try reference.forward(
                inputIDs: tokens, cache: refCache, captureLayers: [0, 1])
            let actual = try candidate.forward(
                inputIDs: tokens, cache: fastCache, captureLayers: [0, 1])
            exact(actual.logits, expected.logits, "ordinary logits rows\(count)")
            exact(
                actual.normalizedHiddenStates, expected.normalizedHiddenStates, "target post-norm")
            for layer in [0, 1] {
                exact(
                    try XCTUnwrap(actual.layerFeatures[layer]),
                    try XCTUnwrap(expected.layerFeatures[layer]), "raw layer feature \(layer)")
            }
            XCTAssertEqual(
                argMax(actual.logits, axis: -1).asArray(UInt32.self),
                argMax(expected.logits, axis: -1).asArray(UInt32.self))
            completeOrdinaryState(fastCache, refCache)
            position += count
        }
    }

    private struct ManagedFixture {
        let target: MiMoV26TextModel
        let adapter: MiMoV26CBv2Adapter
        let backend: MiMoV26CBv2Backend
        let caches: [any CBv2AttendingLayerCache]
        let rows: [CBv2SequenceKV?]
    }

    private func managed(fused: Bool) throws -> ManagedFixture {
        let target = try model(fused: fused)
        let adapter = try MiMoV26CBv2Adapter(target: target)
        let work = NativeConstructionScope()
        defer { if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) } }
        _ = try adapter.probeNativeKVTypes(retaining: work)
        let backend = try adapter.makeBackend(bytesCapacity: 16 << 20)
        let rows = try backend.makeSequenceState(
            layerKinds: adapter.layerKinds, promptLength: 9, maxLength: 128)
        let caches = adapter.makeCaches()
        try adapter.bindRows([rows], caches: caches)
        return ManagedFixture(
            target: target, adapter: adapter, backend: backend, caches: caches, rows: rows)
    }

    private func completeManagedState(_ actual: ManagedFixture, _ expected: ManagedFixture) throws {
        XCTAssertEqual(actual.rows.count, expected.rows.count)
        for (index, pair) in zip(actual.rows, expected.rows).enumerated() {
            let a = try XCTUnwrap(pair.0)
            let b = try XCTUnwrap(pair.1)
            XCTAssertEqual(a.absoluteOffset, b.absoluteOffset, "managed layer\(index) offset")
            XCTAssertEqual(a.retainedCount, b.retainedCount, "managed layer\(index) retained count")
            let left = a.snapshot()
            let right = b.snapshot()
            XCTAssertEqual(left.offset, right.offset)
            exact(left.keys, right.keys, "managed layer\(index) all retained keys")
            exact(left.values, right.values, "managed layer\(index) all retained values")
        }
    }

    private func step(_ fixture: ManagedFixture, _ tokens: MLXArray) throws -> MLXArray {
        try fixture.adapter.bindRows([fixture.rows], caches: fixture.caches)
        return try fixture.adapter.forwardValidated(tokens: tokens, caches: fixture.caches)
    }

    func testManagedTrunkExactGreedyAndCompleteCacheForEveryRowCount() throws {
        let reference = try managed(fused: false)
        let candidate = try managed(fused: true)
        defer {
            reference.backend.release(reference.rows)
            candidate.backend.release(candidate.rows)
        }
        var position = 0
        for count in [9, 1, 2, 3, 4, 5, 6, 7, 8, 1] {
            let expected = try step(reference, ids(position, count))
            let actual = try step(candidate, ids(position, count))
            exact(actual, expected, "managed logits rows\(count)")
            XCTAssertEqual(
                argMax(actual, axis: -1).asArray(UInt32.self),
                argMax(expected, axis: -1).asArray(UInt32.self))
            try completeManagedState(candidate, reference)
            position += count
        }
    }

    func testManagedRollbackEveryPrefixPreservesFullAndWindowState() throws {
        for accepted in 0 ... 7 {
            let reference = try managed(fused: false)
            let candidate = try managed(fused: true)
            defer {
                reference.backend.release(reference.rows)
                candidate.backend.release(candidate.rows)
            }
            exact(try step(candidate, ids(0, 9)), try step(reference, ids(0, 9)), "prompt")
            for fixture in [reference, candidate] {
                for row in fixture.rows { try XCTUnwrap(row).beginSpeculativeWrite() }
            }
            exact(try step(candidate, ids(9, 7)), try step(reference, ids(9, 7)), "verify rows")
            for fixture in [reference, candidate] {
                for item in fixture.rows {
                    let row = try XCTUnwrap(item)
                    row.rollback(7 - accepted)
                    row.commitSpeculativeWrite()
                }
            }
            try completeManagedState(candidate, reference)
            exact(
                try step(candidate, ids(9 + accepted, 1)),
                try step(reference, ids(9 + accepted, 1)), "post-rollback decode")
            try completeManagedState(candidate, reference)
        }
    }
}
