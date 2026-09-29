import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXVLM
import XCTest

/// Discrete-code component only. Native execution requires root's owned lane.
final class MiMoV26AudioPatchTests: XCTestCase {
    private let limits = MiMoV26AudioPatchLimits(
        maximumClips: 8, maximumFrames: 64,
        maximumPatches: 32, maximumWorkingElements: 1_000_000)

    func testCPUPlanMatchesPerItemPaddingAndOrder() throws {
        let corpus = try loadCorpus()
        let c = try configuration(corpus)
        XCTAssertEqual(c.rawFields["speech_vocab_size"], .string("7"))
        XCTAssertEqual(c.rawFields["speech_zeroemb_idx"], .string("6"))
        XCTAssertEqual(c.channels, 20)
        for fixture in corpus.cases {
            let p = try MiMoV26AudioPatchPlan.make(
                clips: clips(fixture), configuration: c, limits: limits)
            XCTAssertEqual(p.groupedCodes, fixture.groupedCodes, fixture.id)
            XCTAssertEqual(
                p.clipPatchRanges.map { [$0.lowerBound, $0.upperBound] }, fixture.clipPatchRanges,
                fixture.id)
            XCTAssertEqual(p.patchCount, fixture.outputShape[0], fixture.id)
            XCTAssertEqual(p.originalFrameCount, fixture.clips.reduce(0) { $0 + $1.frameCount })
        }
        let empty = try MiMoV26AudioPatchPlan.make(
            clips: [], configuration: c,
            limits: .init(
                maximumClips: 0, maximumFrames: 0, maximumPatches: 0, maximumWorkingElements: 0))
        XCTAssertEqual(empty.groupedCodes, [])
        XCTAssertEqual(empty.clipPatchRanges, [])
        XCTAssertEqual(empty.workingElements, 0)
    }

    func testCPUPreflightRejectsBadNamespacesShapesAndBounds() throws {
        let c = try configuration(loadCorpus())
        let valid = MiMoV26AudioCodeClip(codes: [Int32](repeating: 0, count: 20), frameCount: 1)
        let malformed = [
            MiMoV26AudioCodeClip(codes: [], frameCount: 0),
            .init(codes: [], frameCount: -1), .init(codes: [], frameCount: Int.max),
            .init(codes: [0], frameCount: 1),
            .init(codes: [Int32](repeating: -1, count: 20), frameCount: 1),
            .init(codes: [Int32](repeating: 7, count: 20), frameCount: 1),
            // A text audio-placeholder ID is not a speech codebook index.
            .init(codes: [Int32](repeating: 151669, count: 20), frameCount: 1),
        ]
        for clip in malformed {
            XCTAssertThrowsError(
                try MiMoV26AudioPatchPlan.make(
                    clips: [valid, clip], configuration: c, limits: limits))
        }
        for restricted in [
            MiMoV26AudioPatchLimits(
                maximumClips: 0, maximumFrames: 64, maximumPatches: 32,
                maximumWorkingElements: 1_000_000),
            .init(
                maximumClips: 8, maximumFrames: 0, maximumPatches: 32,
                maximumWorkingElements: 1_000_000),
            .init(
                maximumClips: 8, maximumFrames: 64, maximumPatches: 0,
                maximumWorkingElements: 1_000_000),
            .init(
                maximumClips: 8, maximumFrames: 64, maximumPatches: 32, maximumWorkingElements: 0),
            .init(
                maximumClips: -1, maximumFrames: 64, maximumPatches: 32,
                maximumWorkingElements: 1_000_000),
        ] {
            XCTAssertThrowsError(
                try MiMoV26AudioPatchPlan.make(clips: [valid], configuration: c, limits: restricted)
            )
        }
        let p = try MiMoV26AudioPatchPlan.make(clips: [valid], configuration: c, limits: limits)
        XCTAssertNoThrow(
            try MiMoV26AudioPatchPlan.make(
                clips: [valid], configuration: c,
                limits: .init(
                    maximumClips: 1, maximumFrames: 1, maximumPatches: 1,
                    maximumWorkingElements: p.workingElements)))
        XCTAssertThrowsError(
            try MiMoV26AudioPatchPlan.make(
                clips: [valid], configuration: c,
                limits: .init(
                    maximumClips: 1, maximumFrames: 1, maximumPatches: 1,
                    maximumWorkingElements: p.workingElements - 1)))
    }

    func testCPURefusesUnsupportedReferencePolicy() throws {
        let corpus = try loadCorpus()
        for (key, value) in [
            ("input_full_attention", MiMoV26JSONValue.bool(false)), ("add_post_norm", .bool(false)),
            ("projection_layers", .number(1)),
            ("partial_rotary_factor", .number(Decimal(string: "0.5")!)),
            ("input_local_hidden_dropout", .number(Decimal(string: "0.1")!)),
            ("input_local_layers", .number(7)),
        ] {
            var fields = try target(corpus).rawFields
            guard case .object(var audio) = fields["audio_config"] else {
                return XCTFail("missing audio")
            }
            audio[key] = value
            fields["audio_config"] = .object(audio)
            let changed = try MiMoV26Configuration(rawFields: fields)
            XCTAssertThrowsError(
                try MiMoV26AudioPatchEncoder.expectedTensorShapes(
                    configuration: XCTUnwrap(changed.audio)), key)
        }
    }

    func testAll95ArtifactTensorShapesAndDTypeReceipt() throws {
        guard let path = ProcessInfo.processInfo.environment["MIMO_V26_OFFICIAL_CONFIG"],
            let receiptPath = ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_PATCH_COVERAGE"]
        else {
            throw XCTSkip(
                "Set official config and audio-patch coverage paths for the95-tensor header gate")
        }
        let t = try JSONDecoder().decode(
            MiMoV26Configuration.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let c = try XCTUnwrap(t.audio)
        let expected = try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: c)
        let receipt = try JSONDecoder().decode(
            Coverage.self, from: Data(contentsOf: URL(fileURLWithPath: receiptPath)))
        XCTAssertEqual(expected.count, 95)
        XCTAssertEqual(expected, receipt.expectedShapes)
        XCTAssertEqual(expected.keys.filter { $0.hasPrefix("speech_embeddings.") }.count, 20)
        XCTAssertEqual(expected.keys.filter { $0.hasPrefix("audio_encoder.") }.count, 75)
        XCTAssertEqual(expected["audio_encoder.projection.mlp.0.weight"], [16384, 4096])
        XCTAssertEqual(expected["audio_encoder.projection.mlp.2.weight"], [4096, 16384])
        XCTAssertNil(expected["audio_encoder.input_local_transformer.embed_tokens.weight"])
        XCTAssertNil(expected["audio_encoder.projection.mlp.0.bias"])
        XCTAssertEqual(c.rawFields["speech_vocab_size"], .string("1280"))
        XCTAssertEqual(c.rawFields["speech_zeroemb_idx"], .string("1024"))
    }

    func testNativeTinyEncoderAgainstIndependentScalarOracle() throws {
        try requireNativeLane()
        let corpus = try loadCorpus()
        let model = try MiMoV26AudioPatchEncoder(configuration: configuration(corpus))
        try model.loadNativeWeights(arrays(corpus), expectedDType: .float32)
        XCTAssertEqual(corpus.cases.count, 6)
        for fixture in corpus.cases {
            let output = try model.forward(clips: clips(fixture), limits: limits)
            eval(output.features)
            XCTAssertEqual(output.features.shape, fixture.outputShape, fixture.id)
            XCTAssertEqual(
                output.clipPatchRanges.map { [$0.lowerBound, $0.upperBound] },
                fixture.clipPatchRanges)
            // The pinned Swift MLX asArray bridge computes physicalSize using
            // max(shape*stride), which is nonzero for[0,width], then wraps a
            // nil zero-allocation pointer. Empty model output has no bytes to
            // read: assert its logical size instead, without skipping the case.
            XCTAssertEqual(output.features.size, fixture.expected.count, fixture.id)
            let actual: [Float] =
                output.features.size == 0
                ? [] : output.features.asType(.float32).asArray(Float.self)
            XCTAssertEqual(actual.count, fixture.expected.count)
            for (a, b) in zip(actual, fixture.expected) {
                XCTAssertTrue(a.isFinite)
                XCTAssertEqual(a, b, accuracy: 1e-4, fixture.id)
            }
            if fixture.id == "three-frame" {
                XCTAssertGreaterThan(maxDifference(actual, fixture.causalCounterfactual), 0.01)
                XCTAssertGreaterThan(
                    maxDifference(actual, fixture.wrongChannelCounterfactual), 0.01)
            }
            if fixture.id == "padding-code" {
                XCTAssertGreaterThan(maxDifference(actual, fixture.zeroPaddingCounterfactual), 0.1)
            }
        }
        // Request state and repeat-last padding cannot leak between calls.
        let first = try XCTUnwrap(corpus.cases.first { $0.id == "three-frame" })
        let mixed = try XCTUnwrap(corpus.cases.first { $0.id == "two-segments" })
        let a = try model.forward(clips: clips(first), limits: limits).features
        let b = try model.forward(clips: clips(mixed), limits: limits).features
        eval(a, b)
        for (value, expected) in zip(a.asArray(Float.self), b.asArray(Float.self).prefix(8)) {
            XCTAssertEqual(value, expected, accuracy: 1e-5)
        }
    }

    func testNativeStrictLoadingAndMutationInvalidation() throws {
        try requireNativeLane()
        let corpus = try loadCorpus()
        let model = try MiMoV26AudioPatchEncoder(configuration: configuration(corpus))
        let weights = arrays(corpus)
        let fixture = try XCTUnwrap(corpus.cases.first)
        XCTAssertThrowsError(try model.forward(clips: clips(fixture), limits: limits)) {
            XCTAssertEqual($0 as? MiMoV26AudioPatchError, .weightsNotLoaded)
        }
        var missing = weights
        missing.removeValue(forKey: "speech_embeddings.19.weight")
        XCTAssertThrowsError(try model.loadNativeWeights(missing, expectedDType: .float32))
        for extra in [
            "audio_encoder.input_local_transformer.embed_tokens.weight",
            "audio_encoder.projection.mlp.0.bias",
            "speech_embeddings.20.weight", "audio_tokenizer.encoder.weight",
            "model.embed_tokens.weight",
        ] {
            var changed = weights
            changed[extra] = MLXArray.zeros([4])
            XCTAssertThrowsError(
                try model.loadNativeWeights(changed, expectedDType: .float32), extra)
        }
        var badShape = weights
        badShape["speech_embeddings.0.weight"] = MLXArray.zeros([7, 3])
        XCTAssertThrowsError(try model.loadNativeWeights(badShape, expectedDType: .float32))
        var badDType = weights
        badDType["speech_embeddings.0.weight"] = weights["speech_embeddings.0.weight"]!.asType(
            .int32)
        XCTAssertThrowsError(try model.loadNativeWeights(badDType, expectedDType: .float32))
        try model.loadNativeWeights(weights, expectedDType: .float32)
        XCTAssertThrowsError(try model.loadNativeWeights(missing, expectedDType: .float32))
        XCTAssertThrowsError(try model.forward(clips: clips(fixture), limits: limits)) {
            XCTAssertEqual($0 as? MiMoV26AudioPatchError, .weightsNotLoaded)
        }
        try model.loadNativeWeights(weights, expectedDType: .float32)
        try model.update(parameters: model.parameters(), verify: .all)
        XCTAssertThrowsError(try model.forward(clips: clips(fixture), limits: limits)) {
            XCTAssertEqual($0 as? MiMoV26AudioPatchError, .weightsNotLoaded)
        }
        try model.loadNativeWeights(weights, expectedDType: .float32)
        eval(try model.forward(clips: clips(fixture), limits: limits).features)
    }

    private func requireNativeLane() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_PATCH_NATIVE_TESTS"] == "1" else {
            throw XCTSkip(
                "Requires an exclusive native execution lane and MIMO_V26_AUDIO_PATCH_NATIVE_TESTS=1"
            )
        }
    }
    private func loadCorpus() throws -> Corpus {
        guard let path = ProcessInfo.processInfo.environment["MIMO_V26_AUDIO_PATCH_ORACLE"] else {
            throw XCTSkip("Set MIMO_V26_AUDIO_PATCH_ORACLE to the generated scalar fixture")
        }
        return try JSONDecoder().decode(
            Corpus.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    }
    private func target(_ corpus: Corpus) throws -> MiMoV26Configuration {
        try JSONDecoder().decode(
            MiMoV26Configuration.self, from: JSONEncoder().encode(corpus.configuration))
    }
    private func configuration(_ corpus: Corpus) throws -> MiMoV26AudioConfiguration {
        try XCTUnwrap(target(corpus).audio)
    }
    private func clips(_ fixture: Case) -> [MiMoV26AudioCodeClip] {
        fixture.clips.map { .init(codes: $0.codes, frameCount: $0.frameCount) }
    }
    private func arrays(_ corpus: Corpus) -> [String: MLXArray] {
        corpus.weights.mapValues { MLXArray($0.values).reshaped($0.shape) }
    }
    private func maxDifference(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }
    private struct Coverage: Decodable { let expectedShapes: [String: [Int]] }
    private struct Clip: Decodable {
        let frameCount: Int
        let codes: [Int32]
    }
    private struct Tensor: Decodable {
        let shape: [Int]
        let values: [Float]
    }
    private struct Case: Decodable {
        let id: String
        let clips: [Clip]
        let groupedCodes: [Int32]
        let clipPatchRanges: [[Int]]
        let outputShape: [Int]
        let expected, causalCounterfactual, zeroPaddingCounterfactual,
            wrongChannelCounterfactual: [Float]
    }
    private struct Corpus: Decodable {
        let schemaVersion: Int
        let configuration: MiMoV26JSONValue
        let weights: [String: Tensor]
        let cases: [Case]
    }
}
