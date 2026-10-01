import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Optional text-path candidates of the tiny BF16 MiMo target: the fused
/// decode norms, the row-local rectangular dense projections, their scratch
/// pricing and the admission scope. Row-local results are compared with the
/// same layer called on one row; they use the same operation, so they match
/// exactly.
final class MiMoV26TinyTextPathTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint
    /// BF16 storage. The fused kernels round the normalized value at another
    /// point than the MLX operations; 0.05 allows a few BF16 steps near 1.0.
    private let tolerance: Float = 0.05

    private func values(_ name: String, _ shape: [Int]) -> MLXArray {
        MLXArray(Fixture.floats(name, count: shape.reduce(1, *)).map { $0 * 20 }, shape)
            .asType(.bfloat16)
    }
    private func difference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    private func rows(_ x: MLXArray, _ body: (MLXArray) -> MLXArray) -> MLXArray {
        concatenated(
            (0 ..< x.dim(1)).map { body(x[0..., $0 ..< ($0 + 1), 0...].contiguous()) }, axis: 1)
    }

    func testFusedDecodeNormsMatchTheUnfusedForward() throws {
        let target = try Fixture.mediaModels(dtype: "bfloat16").target
        for ids in [[Int32(5)], [5, 6, 7]] {
            let input = MLXArray(ids).reshaped(1, ids.count)
            target.model.useFusedDecodeNorms = false
            let plain = try target.forward(
                inputIDs: input, cache: target.newCache(), captureLayers: [0, 1])
            target.model.useFusedDecodeNorms = true
            let fused = try target.forward(
                inputIDs: input, cache: target.newCache(), captureLayers: [0, 1])
            target.model.useFusedDecodeNorms = false
            eval(plain.logits, fused.logits, plain.normalizedHiddenStates)
            XCTAssertEqual(fused.logits.shape, [1, ids.count, 128])
            XCTAssertEqual(Set(fused.layerFeatures.keys), [0, 1])
            XCTAssertLessThanOrEqual(difference(fused.logits, plain.logits), tolerance)
            XCTAssertLessThanOrEqual(
                difference(fused.normalizedHiddenStates, plain.normalizedHiddenStates), tolerance)
            XCTAssertLessThanOrEqual(
                difference(
                    try XCTUnwrap(fused.layerFeatures[1]), try XCTUnwrap(plain.layerFeatures[1])),
                tolerance)
        }
    }

    func testRowLocalProjectionsMatchOneRowCalls() throws {
        let target = try Fixture.mediaModels(dtype: "bfloat16").target
        let x = values("rows", [1, 3, 64])
        let attention = target.model.layers[0].selfAttention
        let projected = MiMoV26RectangularDense.projection(attention.qProj, x, enabled: true)
        let readout = MiMoV26RectangularDense.readout(target, x, enabled: true)
        let dense = target.model.layers[0].mlp
        let mlp = MiMoV26RectangularDense.mlp(dense, x, enabled: true)
        let moe = MiMoV26RectangularDense.mlp(target.model.layers[1].mlp, x, enabled: true)
        let lmHead = try XCTUnwrap(target.lmHead)
        let expected = [
            rows(x) { attention.qProj($0) }, rows(x) { lmHead($0) }, rows(x) { dense($0) },
        ]
        eval([projected, readout, mlp, moe] + expected)
        XCTAssertEqual(projected.shape, [1, 3, 128])
        XCTAssertEqual(readout.shape, [1, 3, 128])
        XCTAssertEqual(difference(projected, expected[0]), 0)
        XCTAssertEqual(difference(readout, expected[1]), 0)
        XCTAssertEqual(difference(mlp, expected[2]), 0)
        // The MoE keeps the bulk expert gather; only the router is row local.
        XCTAssertEqual(moe.shape, [1, 3, 64])
        XCTAssertTrue(moe.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertEqual(
            difference(
                MiMoV26RectangularDense.projection(attention.qProj, x, enabled: false),
                attention.qProj(x)), 0)
        XCTAssertEqual(
            difference(
                MiMoV26RectangularDense.projection(
                    attention.qProj, x.asType(.float32), enabled: true),
                attention.qProj(x.asType(.float32))), 0)
    }

    func testEligibilityNeedsOneShortRowAndRectangularCaches() {
        let eligible = MiMoV26RectangularDense.eligible(
            shape:rectangularCacheFlags:requested:fusedNorms:)
        XCTAssertTrue(eligible([1, 2], [true], true, false))
        XCTAssertTrue(eligible([1, 4], [true, true], true, false))
        XCTAssertFalse(eligible([1, 1], [true], true, false))
        XCTAssertFalse(eligible([1, 5], [true], true, false))
        XCTAssertFalse(eligible([2, 2], [true], true, false))
        XCTAssertFalse(eligible([1, 2, 1], [true], true, false))
        XCTAssertFalse(eligible([1, 2], [], true, false))
        XCTAssertFalse(eligible([1, 2], [true, false], true, false))
        XCTAssertFalse(eligible([1, 2], [true], false, false))
        XCTAssertFalse(eligible([1, 2], [true], true, true))
        XCTAssertTrue(
            MiMoV26RectangularDense.supports(MLXArray.zeros([1, 2, 8], dtype: .bfloat16)))
        XCTAssertTrue(MiMoV26RectangularDense.supports(MLXArray.zeros([1, 4, 8], dtype: .float16)))
        XCTAssertFalse(
            MiMoV26RectangularDense.supports(MLXArray.zeros([1, 1, 8], dtype: .bfloat16)))
        XCTAssertFalse(
            MiMoV26RectangularDense.supports(MLXArray.zeros([2, 2, 8], dtype: .bfloat16)))
        XCTAssertFalse(MiMoV26RectangularDense.supports(MLXArray.zeros([1, 2, 8])))
        XCTAssertFalse(MiMoV26RectangularDense.supports(MLXArray.zeros([2, 8], dtype: .bfloat16)))
    }

    func testScratchSpecPricesEveryLayerAndTheReadout() throws {
        let target = try Fixture.mediaModels(dtype: "bfloat16").target
        let spec = try XCTUnwrap(MiMoV26RectangularDense.scratchSpec(target))
        // Each layer: 4 projections x 3 buffers; dense MLP 4 more, MoE router
        // 5 more; readout 3. Host: 2 x (128 KiB + 2 layers x 4 x 2048).
        XCTAssertEqual(spec.buffers.count, 16 + 17 + 3)
        XCTAssertEqual(spec.hostBytes, 294_912)
        XCTAssertTrue(spec.buffers.allSatisfy { $0.allocationCount.isMultiple(of: 2) })
        XCTAssertEqual(
            MiMoV26RectangularDenseBudget.resolve(spec, upperBound: { $0 }),
            spec.buffers.reduce(spec.hostBytes) { $0 + $1.logicalBytes * $1.allocationCount })
        // A float32 target is outside the candidate.
        XCTAssertNil(MiMoV26RectangularDense.scratchSpec(try Fixture.mediaModels().target))
        // Actual module geometry is checked, not only the configuration.
        target.model.layers[1].selfAttention.update(
            modules: .unflattened([("v_proj", Linear(64, 7, bias: false))]))
        XCTAssertNil(MiMoV26RectangularDense.scratchSpec(target))
    }

    func testAdmissionScopeIsBoundToOneModelAndCountsSubmissions() throws {
        let target = try Fixture.mediaModels(dtype: "bfloat16").target
        let first = try MiMoV26CBv2Adapter(target: target)
        let second = try MiMoV26CBv2Adapter(target: target)
        let spec = try XCTUnwrap(MiMoV26RectangularDense.scratchSpec(target))
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let backend = try Fixture.withScope { work in
            _ = try first.probeNativeKVTypes(retaining: work)
            return try first.makeBackend(bytesCapacity: 32 << 20)
        }
        let bank = CBv2LayerCacheBank(caches: first.makeCaches())
        let budget = try MiMoV26RectangularDenseBudget(
            engineID: UUID(), model: first, backend: backend, cacheProvider: bank, spec: spec,
            policy: policy)
        XCTAssertGreaterThan(budget.fixedRequestBytes, 0)
        XCTAssertEqual(budget.modelIdentity, ObjectIdentifier(first))
        XCTAssertThrowsError(
            try MiMoV26RectangularDenseBudget(
                engineID: UUID(), model: first, backend: backend, cacheProvider: bank,
                spec: .init(buffers: [], hostBytes: 0), policy: policy))
        XCTAssertFalse(MiMoV26RectangularDenseAdmission.isActive(for: first))
        MiMoV26RectangularDenseAdmission.recordSubmission(for: first)
        XCTAssertEqual(budget.submittedCalls, 0)
        MiMoV26RectangularDenseAdmission.withBudget(budget) {
            XCTAssertTrue(MiMoV26RectangularDenseAdmission.isActive(for: first))
            XCTAssertFalse(MiMoV26RectangularDenseAdmission.isActive(for: second))
            MiMoV26RectangularDenseAdmission.recordSubmission(for: first)
            MiMoV26RectangularDenseAdmission.recordSubmission(for: second)
            MiMoV26RectangularDenseAdmission.withBudget(nil) {
                XCTAssertFalse(MiMoV26RectangularDenseAdmission.isActive(for: first))
            }
        }
        XCTAssertFalse(MiMoV26RectangularDenseAdmission.isActive(for: first))
        XCTAssertEqual(budget.submittedCalls, 1)
    }
}
