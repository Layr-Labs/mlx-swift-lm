import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

/// Prepared native fixtures. The contracts worker did not import/run this test
/// module. Model-array execution requires root's explicit native-test opt-in.
final class MiMoV26VisionTests: XCTestCase {
    private let smallLimits = MiMoV26VisionLimits(
        maximumPatches: 1024,
        maximumAttentionScoreElements: 1_000_000)

    func testMergeGroupPositionsAndFrameIsolation() throws {
        let l = try MiMoV26VisionLayout.make(
            grids: [.init(temporal: 2, height: 4, width: 6)], mergeSize: 2, queryHeads: 4,
            limits: smallLimits)
        XCTAssertEqual(l.patchCount, 48)
        XCTAssertEqual(l.frames, [0 ..< 24, 24 ..< 48])
        XCTAssertEqual(Array(l.rowPositions.prefix(8)), [0, 0, 1, 1, 0, 0, 1, 1])
        XCTAssertEqual(Array(l.columnPositions.prefix(8)), [0, 1, 0, 1, 2, 3, 2, 3])
        XCTAssertEqual(Array(l.columnPermutation.prefix(8)), [0, 1, 2, 3, 12, 13, 14, 15])
        XCTAssertEqual(
            Array(l.columnPermutation.suffix(24)),
            Array(l.columnPermutation.prefix(24)).map { $0 + 24 })
        XCTAssertEqual(l.inverseColumnPermutation.map { l.columnPermutation[$0] }, Array(0 ..< 48))
        XCTAssertEqual(Array(l.rowPositions.prefix(24)), Array(l.rowPositions.suffix(24)))
    }

    func testLayoutRefusesInvalidGeometryAndLimits() throws {
        for grids in [
            [MiMoV26VisionGrid](), [.init(temporal: 0, height: 4, width: 6)],
            [.init(temporal: 1, height: 3, width: 6)],
            [.init(temporal: Int.max, height: 4, width: 6)],
        ] {
            XCTAssertThrowsError(
                try MiMoV26VisionLayout.make(
                    grids: grids, mergeSize: 2,
                    queryHeads: 4, limits: smallLimits))
        }
        let grids = [MiMoV26VisionGrid(temporal: 2, height: 4, width: 6)]
        XCTAssertThrowsError(
            try MiMoV26VisionLayout.make(
                grids: grids, mergeSize: 2, queryHeads: 4,
                limits: .init(maximumPatches: 47, maximumAttentionScoreElements: 1_000_000)))
        XCTAssertThrowsError(
            try MiMoV26VisionLayout.make(
                grids: grids, mergeSize: 2, queryHeads: 4,
                limits: .init(maximumPatches: 100, maximumAttentionScoreElements: 2303)))
    }

    func testAll364NativeTensorShapesAgainstHeaderReceipt() throws {
        guard let configPath = ProcessInfo.processInfo.environment["MIMO_V26_OFFICIAL_CONFIG"],
            let coveragePath = ProcessInfo.processInfo.environment["MIMO_V26_VISION_COVERAGE"]
        else {
            throw XCTSkip(
                "Set official config and vision coverage paths for the artifact-bound 364-tensor gate"
            )
        }
        let target = try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: Data(contentsOf: URL(fileURLWithPath: configPath)))
        let vision = try XCTUnwrap(target.vision)
        let expected = try MiMoV26VisionTower.expectedTensorShapes(configuration: vision)
        let receipt = try JSONDecoder().decode(
            Coverage.self,
            from: Data(contentsOf: URL(fileURLWithPath: coveragePath)))
        XCTAssertEqual(expected.count, 364)
        XCTAssertEqual(expected, receipt.expectedShapes)
        XCTAssertEqual(expected["visual.blocks.0.attn.qkv.weight"], [3072, 1280])
        XCTAssertEqual(expected["visual.blocks.0.attn.proj.weight"], [1280, 2048])
        XCTAssertNil(expected["visual.blocks.0.attn.sinks"])
        XCTAssertEqual(expected["visual.blocks.1.attn.sinks"], [32])
        XCTAssertNil(expected["visual.merger.mlp.0.bias"])
        XCTAssertNil(expected["visual.merger.ln_q.bias"])
    }

    func testNativeTinyTowerAgainstIndependentScalarOracle() throws {
        let corpus = try nativeCorpus()
        let model = try makeModel(corpus)
        try model.loadNativeWeights(arrays(corpus), expectedDType: .float32)
        XCTAssertEqual(corpus.cases.count, 3)
        for fixture in corpus.cases {
            let grids = fixture.grids.map {
                MiMoV26VisionGrid(temporal: $0[0], height: $0[1], width: $0[2])
            }
            let layout = try MiMoV26VisionLayout.make(
                grids: grids, mergeSize: 2, queryHeads: 4, limits: smallLimits)
            XCTAssertEqual(layout.patchCount, fixture.layout.patchCount)
            XCTAssertEqual(layout.rowPositions, fixture.layout.rowPositions)
            XCTAssertEqual(layout.columnPositions, fixture.layout.columnPositions)
            XCTAssertEqual(layout.columnPermutation, fixture.layout.columnPermutation)
            XCTAssertEqual(layout.inverseColumnPermutation, fixture.layout.inverseColumnPermutation)
            XCTAssertEqual(
                layout.frames.map { [$0.lowerBound, $0.upperBound] }, fixture.layout.frames)
            let patches = MLXArray(fixture.patches).reshaped(layout.patchCount, -1)
            let result = try model.forward(patches: patches, grids: grids, limits: smallLimits)
            eval(result)
            XCTAssertEqual(result.shape, fixture.outputShape)
            let actual = result.asType(.float32).asArray(Float.self)
            XCTAssertEqual(actual.count, fixture.expected.count)
            // Tiny standard-library Double -> native FP32 diagnostic only.
            // This is not a BF16/artifact/full-model numerical acceptance bound.
            for (a, b) in zip(actual, fixture.expected) {
                XCTAssertTrue(a.isFinite)
                XCTAssertEqual(a, b, accuracy: 1e-4, fixture.id)
            }
        }
    }

    func testStrictLoadingAndInputFailuresAreRecoverable() throws {
        let corpus = try nativeCorpus()
        let model = try makeModel(corpus)
        let weights = arrays(corpus)
        let fixture = try XCTUnwrap(corpus.cases.first)
        let input = MLXArray(fixture.patches).reshaped(fixture.layout.patchCount, -1)
        let grids = [MiMoV26VisionGrid(temporal: 1, height: 2, width: 2)]
        XCTAssertThrowsError(try model.forward(patches: input, grids: grids, limits: smallLimits)) {
            XCTAssertEqual($0 as? MiMoV26VisionError, .weightsNotLoaded)
        }
        var missing = weights
        missing.removeValue(forKey: "visual.blocks.1.attn.sinks")
        XCTAssertThrowsError(try model.loadNativeWeights(missing, expectedDType: .float32))
        var unknown = weights
        unknown["visual.merger.mlp.0.bias"] = MLXArray.zeros([32])
        XCTAssertThrowsError(try model.loadNativeWeights(unknown, expectedDType: .float32))
        var malformed = weights
        malformed["visual.blocks.0.attn.qkv.weight"] = MLXArray.zeros([24, 8])
        XCTAssertThrowsError(try model.loadNativeWeights(malformed, expectedDType: .float32))
        var wrongType = weights
        wrongType["visual.blocks.0.norm1.weight"] = MLXArray.zeros([8], dtype: .int32)
        XCTAssertThrowsError(try model.loadNativeWeights(wrongType, expectedDType: .float32))
        try model.loadNativeWeights(weights, expectedDType: .float32)
        XCTAssertThrowsError(
            try model.forward(patches: input.asType(.int32), grids: grids, limits: smallLimits))
        XCTAssertThrowsError(
            try model.forward(
                patches: input, grids: [.init(temporal: 2, height: 2, width: 2)],
                limits: smallLimits))
        try model.update(parameters: model.parameters(), verify: .all)
        XCTAssertThrowsError(try model.forward(patches: input, grids: grids, limits: smallLimits)) {
            XCTAssertEqual($0 as? MiMoV26VisionError, .weightsNotLoaded)
        }
        try model.loadNativeWeights(weights, expectedDType: .float32)
        eval(try model.forward(patches: input, grids: grids, limits: smallLimits))
    }

    func testDenominatorOnlySinkPrimitive() throws {
        try requireNativeLane()
        let q = MLXArray.zeros([1, 4, 3, 8])
        let k = MLXArray.zeros([1, 2, 3, 8])
        let v = MLXArray.ones([1, 2, 3, 8])
        let sink = MLXArray.zeros([4])
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1 / Float(8).squareRoot(), mask: .none, sinks: sink)
        eval(output)
        for value in output.asArray(Float.self) { XCTAssertEqual(value, 0.75, accuracy: 1e-6) }
        // Adding zero bias to the first real key would return1; that is not the
        // selected native sink contract and must not be used as this oracle.
    }

    func testNoSinkReferenceUsesGlobalAttention() throws {
        let corpus = try nativeCorpus(variable: "MIMO_V26_VISION_NO_SINK_ORACLE")
        let model = try makeModel(corpus)
        XCTAssertFalse(model.configuration.usesSinks)
        XCTAssertEqual(model.configuration.windowAttentionTypes, [0])
        XCTAssertEqual(corpus.cases.count, 2)
        XCTAssertFalse(corpus.weights.keys.contains(where: { $0.hasSuffix(".sinks") }))
        try model.loadNativeWeights(arrays(corpus), expectedDType: .float32)
        for fixture in corpus.cases {
            let grids = fixture.grids.map {
                MiMoV26VisionGrid(temporal: $0[0], height: $0[1], width: $0[2])
            }
            let patches = MLXArray(fixture.patches).reshaped(fixture.layout.patchCount, -1)
            let output = try model.forward(patches: patches, grids: grids, limits: smallLimits)
            eval(output)
            let actual = output.asType(.float32).asArray(Float.self)
            let windowed = try XCTUnwrap(fixture.windowedCounterfactual)
            XCTAssertEqual(actual.count, fixture.expected.count)
            XCTAssertEqual(windowed.count, fixture.expected.count)
            for (a, expected) in zip(actual, fixture.expected) {
                XCTAssertTrue(a.isFinite)
                XCTAssertEqual(a, expected, accuracy: 1e-4, fixture.id)
            }
            // Pinned SGLang explicitly selects global attention without sinks.
            // Reject the proposed windowed alternative, including132tokens.
            let distinction = zip(actual, windowed).map { abs($0 - $1) }.max() ?? 0
            XCTAssertGreaterThan(distinction, 0.01, fixture.id)
        }
    }

    private func requireNativeLane() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_VISION_NATIVE_TESTS"] == "1" else {
            throw XCTSkip(
                "Requires an exclusive native execution lane and MIMO_V26_VISION_NATIVE_TESTS=1")
        }
    }
    private func nativeCorpus(variable: String = "MIMO_V26_VISION_ORACLE") throws -> Corpus {
        try requireNativeLane()
        guard let path = ProcessInfo.processInfo.environment[variable] else {
            throw XCTSkip("Set \(variable) to the independent tiny fixture")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let corpus = try JSONDecoder().decode(Corpus.self, from: data)
        XCTAssertEqual(corpus.schemaVersion, 1)
        return corpus
    }
    private func makeModel(_ corpus: Corpus) throws -> MiMoV26VisionTower {
        let data = try JSONEncoder().encode(corpus.configuration)
        let target = try JSONDecoder().decode(MiMoV26Configuration.self, from: data)
        return try MiMoV26VisionTower(configuration: XCTUnwrap(target.vision))
    }
    private func arrays(_ corpus: Corpus) -> [String: MLXArray] {
        corpus.weights.mapValues { MLXArray($0.values).reshaped($0.shape) }
    }

    private struct Coverage: Decodable {
        let expectedShapes: [String: [Int]]
        enum CodingKeys: String, CodingKey { case expectedShapes = "expected_shapes" }
    }
    private struct Tensor: Decodable {
        let shape: [Int]
        let values: [Float]
    }
    private struct Layout: Decodable {
        let patchCount: Int
        let frames: [[Int]]
        let rowPositions, columnPositions, columnPermutation, inverseColumnPermutation: [Int]
    }
    private struct Case: Decodable {
        let id: String
        let grids: [[Int]]
        let layout: Layout
        let patches, expected: [Float]
        let windowedCounterfactual: [Float]?
        let outputShape: [Int]
    }
    private struct Corpus: Decodable {
        let schemaVersion: Int
        let configuration: MiMoV26JSONValue
        let weights: [String: Tensor]
        let cases: [Case]
    }
}
