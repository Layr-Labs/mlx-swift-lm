import Foundation
import MLX
import MLXLLM
import XCTest

@testable import MLXVLM

final class MiMoV26VisionWorkingSetTests: XCTestCase {
    private func configuration(_ overrides: [String: Int] = [:]) throws
        -> MiMoV26VisionConfiguration
    {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "config.json", withExtension: nil, subdirectory: "MiMoOpenRouter"))
        var fields = try JSONDecoder().decode(
            [String: MiMoV26JSONValue].self, from: Data(contentsOf: url))
        guard case .object(var vision) = fields["vision_config"] else {
            throw MiMoV26VisionError.invalidConfiguration("fixture")
        }
        for (key, value) in overrides { vision[key] = .number(Decimal(value)) }
        fields["vision_config"] = .object(vision)
        return try XCTUnwrap(MiMoV26Configuration(rawFields: fields).vision)
    }

    private func geometry(height: Int = 864, width: Int = 1280) throws -> MiMoV26MediaGeometry.Plan
    {
        try MiMoV26MediaGeometry.image(
            height: height, width: width,
            settings: .init(
                patchSize: 16, mergeSize: 2, temporalPatchSize: 2, temporalCompressionRatio: 1,
                imageMinPixels: 8192, imageMaxPixels: 8_388_608,
                videoMinPixels: 8192, videoMaxPixels: 8_388_608, videoTotalMaxPixels: 268_435_456))
    }

    func testPublishedHeadGeometrySelectsLinearMetalWorkingSet() throws {
        let c = try configuration()
        let shape = try MiMoV26VisionShape(c)
        XCTAssertEqual(c.hiddenSize / c.queryHeads, 40)
        XCTAssertEqual(shape.headDim, 64)
        XCTAssertTrue(MiMoV26VisionWorkingSet.usesFusedAttention(shape, stream: .gpu))
        let g = try geometry()
        let value = try MiMoV26VisionWorkingSet.frameBytes(g, configuration: c, stream: .gpu)
        XCTAssertLessThan(value, 1 << 30)
        XCTAssertGreaterThan(value, 512 << 20)
        let doubled = try MiMoV26VisionWorkingSet.frameBytes(
            geometry(height: 1728), configuration: c, stream: .gpu)
        XCTAssertLessThanOrEqual(doubled, value * 2)
    }

    func testCPUCustomStreamsAndUnqualifiedShapesRetainFullScoreBound() throws {
        let c = try configuration()
        let g = try geometry()
        let fallback = try MiMoV26VisionWorkingSet.frameBytes(g, configuration: c, stream: .cpu)
        XCTAssertGreaterThan(fallback, 11 << 30)
        XCTAssertFalse(
            MiMoV26VisionWorkingSet.usesFusedAttention(try MiMoV26VisionShape(c), stream: .cpu))
        try Stream.withNewDefaultStream(device: .cpu) {
            XCTAssertEqual(try MiMoV26VisionWorkingSet.frameBytes(g, configuration: c), fallback)
        }
        try Stream.withNewDefaultStream(device: .gpu) {
            XCTAssertEqual(try MiMoV26VisionWorkingSet.frameBytes(g, configuration: c), fallback)
        }
        for values in [
            ["qk_channels": 40, "kv_channels": 40],
            ["visual_token_window_size": 128],
            ["num_key_value_heads": 4, "num_query_groups": 8],
        ] {
            let other = try configuration(values)
            XCTAssertFalse(
                MiMoV26VisionWorkingSet.usesFusedAttention(
                    try MiMoV26VisionShape(other), stream: .gpu))
            XCTAssertGreaterThan(
                try MiMoV26VisionWorkingSet.frameBytes(
                    g, configuration: other, stream: .gpu), 11 << 30)
        }
    }

    func testPinnedMetalKernelDoesNotMaterializePublishedFullScoreMatrix() throws {
        // The mandatory no-parallel media lane exercises actual SDPA with the
        // production head geometry. A backend change to the dense fallback
        // must fail this regression before a smaller quote can ship with it.
        for dtype in [DType.bfloat16, .float16, .float32] {
            let q = MLXArray.ones([1, 32, 4320, 64], dtype: dtype)
            let k = MLXArray.ones([1, 8, 4320, 64], dtype: dtype)
            let v = MLXArray.ones([1, 8, 4320, 64], dtype: dtype)
            try withError { eval(q, k, v) }
            Memory.clearCache()
            let baseline = Memory.snapshot().activeMemory
            Memory.peakMemory = 0
            let result = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 0.125, mask: .none)
            try withError { eval(result) }
            XCTAssertLessThan(Memory.snapshot().peakMemory - baseline, 256 << 20)
            XCTAssertEqual(result.shape, q.shape)
            XCTAssertLessThan(abs(result.asType(.float32) - 1).max().item(Float.self), 0.002)
        }
    }
}
