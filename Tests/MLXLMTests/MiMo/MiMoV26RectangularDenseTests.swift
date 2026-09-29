// Copyright © 2026 Eigen Labs.
// Prepared native numerical selectors; no downloaded/full-model fixture here.
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class MiMoV26RectangularDenseMetadataTests: XCTestCase {
    func testOnlyExplicitB1RectangularWidthsAreEligible() {
        for width in 2 ... 4 {
            XCTAssertTrue(
                MiMoV26RectangularDense.eligible(
                    shape: [1, width],
                    rectangularCacheFlags: [true, true], requested: true, fusedNorms: false))
        }
        for shape in [[1, 1], [1, 5], [2, 2], [4], []] {
            XCTAssertFalse(
                MiMoV26RectangularDense.eligible(
                    shape: shape,
                    rectangularCacheFlags: [true], requested: true, fusedNorms: false))
        }
        for (flags, requested, fused) in [
            ([], true, false), ([true, false], true, false),
            ([true], false, false), ([true], true, true),
        ] {
            XCTAssertFalse(
                MiMoV26RectangularDense.eligible(
                    shape: [1, 3],
                    rectangularCacheFlags: flags, requested: requested, fusedNorms: fused))
        }
    }

    func testIndependentAllocatorPaddingAndOverflowRefuseRatherThanUnderprice() {
        let spec = MiMoV26RectangularDenseScratchSpec(
            buffers: [
                .init(logicalBytes: 32, allocationCount: 4),
                .init(logicalBytes: 128, allocationCount: 2),
            ], hostBytes: 19)
        XCTAssertEqual(
            MiMoV26RectangularDenseBudget.resolve(spec, upperBound: { $0 + 64 }),
            19 + 4 * 96 + 2 * 192)
        XCTAssertNil(MiMoV26RectangularDenseBudget.resolve(spec, upperBound: { $0 - 1 }))
        XCTAssertNil(MiMoV26RectangularDenseBudget.resolve(spec, upperBound: { _ in nil }))
        for invalid in [
            MiMoV26RectangularDenseScratchSpec(buffers: [], hostBytes: 0),
            .init(buffers: [.init(logicalBytes: 1, allocationCount: 0)], hostBytes: 0),
            .init(buffers: [.init(logicalBytes: Int.max, allocationCount: 2)], hostBytes: 0),
            .init(buffers: [.init(logicalBytes: 1)], hostBytes: Int.max),
            .init(buffers: [.init(logicalBytes: 1)], hostBytes: -1),
        ] {
            XCTAssertNil(MiMoV26RectangularDenseBudget.resolve(invalid, upperBound: { $0 }))
        }
    }
}

final class MiMoV26RectangularDenseTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_RECTANGULAR_DENSE_NATIVE_TESTS"] == "1"
        else {
            throw XCTSkip("Requires the authorized exclusive native lane")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Numerical Metal tests must not be relabelled CPU passes")
        }
    }

    private func signal(_ shape: [Int], salt: Int, dtype: DType) -> MLXArray {
        MLXArray(
            (0 ..< shape.reduce(1, *)).map {
                sin(Float(($0 * 17 + salt * 13) % 4093)) * 0.07
            }, shape
        ).asType(dtype)
    }

    private func exact(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        // Byte comparison includes signed zero and avoids a tolerance oracle.
        XCTAssertEqual(
            actual.asData(access: .copy).data, expected.asData(access: .copy).data,
            file: file, line: line)
        if actual.dtype == expected.dtype && actual.shape == expected.shape {
            func ordered(_ bits: UInt32, sign: UInt32) -> UInt32 {
                bits & sign == 0 ? bits | sign : (~bits & (sign | (sign - 1)))
            }
            let a: [UInt32]
            let b: [UInt32]
            let sign: UInt32
            if actual.dtype == .bfloat16 || actual.dtype == .float16 {
                a = actual.view(dtype: .uint16).asArray(UInt16.self).map(UInt32.init)
                b = expected.view(dtype: .uint16).asArray(UInt16.self).map(UInt32.init)
                sign = 0x8000
            } else if actual.dtype == .float32 {
                a = actual.view(dtype: .uint32).asArray(UInt32.self)
                b = expected.view(dtype: .uint32).asArray(UInt32.self)
                sign = 0x8000_0000
            } else {
                return
            }
            let ulp = zip(a, b).reduce(UInt32(0)) { result, pair in
                let x = ordered(pair.0, sign: sign)
                let y = ordered(pair.1, sign: sign)
                return max(result, x >= y ? x - y : y - x)
            }
            print(
                "MIMO_SCALAR_DENSE dtype=\(actual.dtype) shape=\(actual.shape) maxNativeULP=\(ulp)")
        }
    }

    private func serial(_ x: MLXArray, _ body: (MLXArray) -> MLXArray) -> MLXArray {
        // Independent original scalar operation, no candidate helper.
        concatenated(
            (0 ..< x.dim(1)).map {
                body(x[0..., $0 ..< ($0 + 1), 0...].contiguous())
            }, axis: 1)
    }

    private func config(_ dtype: DType, tied: Bool = false) throws -> MiMoV26Configuration {
        var f =
            try JSONSerialization.jsonObject(
                with:
                    JSONEncoder().encode(MiMoV26MTPChecks.config())) as! [String: Any]
        f["hidden_size"] = 64
        f["intermediate_size"] = 128
        f["moe_intermediate_size"] = 64
        f["vocab_size"] = 128
        f["dtype"] = dtype == .bfloat16 ? "bfloat16" : "float16"
        f["moe_router_dtype"] = "bfloat16"
        f["n_routed_experts"] = 8
        f["num_experts_per_tok"] = 2
        f["n_group"] = 4
        f["topk_group"] = 2
        f["tie_word_embeddings"] = tied
        return try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: JSONSerialization.data(withJSONObject: f))
    }

    func testAffine4And8ProjectionBF16FP16StridedRowsAreExactOriginalScalarCalls() throws {
        // Actual implicated K shape and scalar quantized dispatch, bounded
        // synthetic weights. This is not the loaded checkpoint's drift witness.
        for bits in [4, 8] {
            for dtype: DType in [.bfloat16, .float16] {
                let layer = QuantizedLinear(
                    weight: signal([768, 4096], salt: 3, dtype: dtype),
                    bias: nil, groupSize: 64, bits: bits, mode: .affine)
                eval(layer)
                for width in 1 ... 4 {
                    let x = signal([1, 4096, width], salt: width, dtype: dtype).transposed(0, 2, 1)
                    let reference = serial(x) { layer($0) }
                    exact(MiMoV26RectangularDense.projection(layer, x, enabled: true), reference)
                }
            }
        }
    }

    func testDefaultUnsupportedAndScalarShapesUseOriginalWholeCall() throws {
        let layer = Linear(weight: signal([32, 64], salt: 5, dtype: .bfloat16))
        for shape in [[1, 1, 64], [1, 5, 64], [2, 2, 64], [3, 64]] {
            let x = signal(shape, salt: 7, dtype: .bfloat16)
            exact(MiMoV26RectangularDense.projection(layer, x, enabled: true), layer(x))
        }
        let x = signal([1, 3, 64], salt: 8, dtype: .bfloat16)
        exact(MiMoV26RectangularDense.projection(layer, x, enabled: false), layer(x))
        let fp32 = Linear(weight: signal([32, 64], salt: 9, dtype: .float32))
        let y = signal([1, 3, 64], salt: 10, dtype: .float32)
        exact(MiMoV26RectangularDense.projection(fp32, y, enabled: true), fp32(y))
    }

    func testDenseMLPAndTiedUntiedReadoutAreExactScalarOperations() throws {
        for dtype: DType in [.bfloat16, .float16] {
            let dense = MiMoV26DenseMLP(hiddenSize: 64, intermediateSize: 128)
            try dense.update(
                parameters: .unflattened(
                    MiMoV26MTPChecks.fixtureWeights(dense)
                        .mapValues { $0.asType(dtype) }), verify: .all)
            quantize(model: dense, groupSize: 32, bits: 4, mode: .mxfp4)
            for width in 2 ... 4 {
                let x = signal([1, 64, width], salt: 11, dtype: dtype).transposed(0, 2, 1)
                exact(MiMoV26RectangularDense.mlp(dense, x, enabled: true), serial(x) { dense($0) })
            }
            for tied in [false, true] {
                let target = try MiMoV26TextModel(config(dtype, tied: tied))
                try target.update(
                    parameters: .unflattened(
                        MiMoV26MTPChecks.fixtureWeights(target)
                            .mapValues { $0.asType(dtype) }), verify: .all)
                // Both arms use this same, once-prepared affine8 readout.
                if let head = target.lmHead {
                    target.model.useFusedDecodeNorms = false
                    target.update(
                        modules: .unflattened([
                            (
                                "lm_head",
                                QuantizedLinear(head, groupSize: 64, bits: 8, mode: .affine)
                            )
                        ]))
                } else {
                    target.model.update(
                        modules: .unflattened([
                            (
                                "embed_tokens",
                                QuantizedEmbedding(
                                    target.model.embedTokens,
                                    groupSize: 64, bits: 8, mode: .affine)
                            )
                        ]))
                }
                let x = signal([1, 3, 64], salt: 12, dtype: dtype)
                func original(_ row: MLXArray) -> MLXArray {
                    target.lmHead.map { $0(row) } ?? target.model.embedTokens.asLinear(row)
                }
                exact(
                    MiMoV26RectangularDense.readout(target, x, enabled: true), serial(x, original))
                exact(MiMoV26RectangularDense.readout(target, x, enabled: false), original(x))
            }
        }
    }

    func testSharedRouterMatrixRetainsExactPerRowSelectionNormalizationAndReload() throws {
        let c = try config(.bfloat16)
        let gate = MiMoV26Router(c)
        gate.useFusedDecodeRouter = false
        for salt in [13, 29] {
            try gate.update(
                parameters: .unflattened([
                    "weight": signal(
                        [c.routedExpertCount, c.hiddenSize], salt: salt, dtype: .bfloat16),
                    "e_score_correction_bias": signal(
                        [c.routedExpertCount], salt: salt + 1, dtype: .float32),
                ]), verify: .all)
            for width in 2 ... 4 {
                let x = signal([1, c.hiddenSize, width], salt: 15, dtype: .bfloat16).transposed(
                    0, 2, 1)
                let independent = (0 ..< width).map {
                    gate(x[0..., $0 ..< ($0 + 1), 0...].contiguous())
                }
                let actual = gate(x, rowLocal: true)
                exact(actual.indices, concatenated(independent.map(\.indices), axis: 1))
                exact(actual.weights, concatenated(independent.map(\.weights), axis: 1))
                let disabled = gate(x, rowLocal: false)
                let original = gate(x)
                exact(disabled.indices, original.indices)
                exact(disabled.weights, original.weights)
            }
        }
    }

    func testMoERetainsOriginalBulkExpertGatherAndReduction() throws {
        let c = try config(.bfloat16)
        let moe = MiMoV26MoE(c, fp32WeightedReduction: false)
        try moe.update(
            parameters: .unflattened(
                MiMoV26MTPChecks.fixtureWeights(moe)
                    .mapValues { $0.asType(.bfloat16) }), verify: .all)
        moe.gate.useFusedDecodeRouter = false
        let x = signal([1, 3, c.hiddenSize], salt: 17, dtype: .bfloat16)
        let rows = (0 ..< 3).map { moe.gate(x[0..., $0 ..< ($0 + 1), 0...].contiguous()) }
        let indices = concatenated(rows.map(\.indices), axis: 1)
        let weights = concatenated(rows.map(\.weights), axis: 1)
        let expertOutput = moe.switchMLP(x, indices)
        let expected = (expertOutput * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)
        exact(moe.forwardWithWeightedReductionRoute(x, rowLocalRouter: true).output, expected)
        exact(moe(x), moe.forwardWithWeightedReductionRoute(x, rowLocalRouter: false).output)
    }

    func testScratchSpecPricesTargetSeparatelyAndChecksActualShapes() throws {
        let target = try MiMoV26TextModel(config(.bfloat16))
        try target.update(
            parameters: .unflattened(
                MiMoV26MTPChecks.fixtureWeights(target)
                    .mapValues { $0.asType(.bfloat16) }), verify: .all)
        let spec = try XCTUnwrap(MiMoV26RectangularDense.scratchSpec(target))
        let actual = try XCTUnwrap(
            MiMoV26RectangularDenseBudget.resolve(spec, upperBound: { $0 + 4096 }))
        let independentlySummed = spec.buffers.reduce(spec.hostBytes) {
            $0 + ($1.logicalBytes + 4096) * $1.allocationCount
        }
        XCTAssertEqual(actual, independentlySummed)
        XCTAssertGreaterThan(spec.hostBytes, 0)
        XCTAssertTrue(spec.buffers.allSatisfy { $0.allocationCount.isMultiple(of: 2) })
        // Actual replacement geometry, not config-only admission.
        target.model.layers[0].selfAttention.update(
            modules: .unflattened([
                ("k_proj", Linear(64, 7, bias: false))
            ]))
        XCTAssertNil(MiMoV26RectangularDense.scratchSpec(target))
    }

    func testAdmittedScopeUsesActualModelIdentityAndExpires() throws {
        let target = try MiMoV26TextModel(config(.bfloat16))
        try target.update(
            parameters: .unflattened(
                MiMoV26MTPChecks.fixtureWeights(target)
                    .mapValues { $0.asType(.bfloat16) }), verify: .all)
        let first = try MiMoV26CBv2Adapter(target: target)
        let second = try MiMoV26CBv2Adapter(target: target)
        let spec = try XCTUnwrap(MiMoV26RectangularDense.scratchSpec(target))
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let construction = NativeConstructionScope()
        defer {
            if construction.snapshot.isRetainedFault { _ = Unmanaged.passRetained(construction) }
        }
        _ = try first.probeNativeKVTypes(retaining: construction)
        let backend = try first.makeBackend(bytesCapacity: 32 << 20)
        let bank = CBv2LayerCacheBank(caches: first.makeCaches())
        // Scope/identity test with REAL resources; not an execution contract,
        // admitted request or synthetic native completion. No forward below.
        let budget = try MiMoV26RectangularDenseBudget(
            engineID: UUID(), model: first,
            backend: backend, cacheProvider: bank, spec: spec, policy: policy)
        XCTAssertFalse(MiMoV26RectangularDenseAdmission.isActive(for: first))
        MiMoV26RectangularDenseAdmission.withBudget(budget) {
            XCTAssertTrue(MiMoV26RectangularDenseAdmission.isActive(for: first))
            XCTAssertFalse(MiMoV26RectangularDenseAdmission.isActive(for: second))
            MiMoV26RectangularDenseAdmission.withBudget(nil) {
                XCTAssertFalse(MiMoV26RectangularDenseAdmission.isActive(for: first))
            }
            XCTAssertTrue(MiMoV26RectangularDenseAdmission.isActive(for: first))
        }
        XCTAssertFalse(MiMoV26RectangularDenseAdmission.isActive(for: first))
        XCTAssertEqual(budget.submittedCalls, 0)
    }
}
