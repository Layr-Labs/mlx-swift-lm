// Copyright © 2026 Eigen Labs.
//
// pplx-decider-v1-27b: the `decision` block contract, the option-wide output
// head on the dense Qwen 3.5/3.8 text and VLM wrappers, and the decision
// probabilities against the published reference math.
// `Resources/pplx-decider-reference-vectors.json` is copied from
// `vectors/pplx_decider_reference_vectors.json` (make_vectors.py, decision_config
// sha256 in `source`): the reference `decision` block and the `probabilities` cases.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class PplxDeciderConfigurationTests: XCTestCase {

    private struct ProbabilityCase: Decodable {
        let count: Int
        let temperature: Double
        let engine_logprobs: [Double]
        let probabilities: [Double]
    }

    private struct ProbabilityFixture: Decodable {
        let probabilities: [ProbabilityCase]
    }

    private func fixtureData() throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "pplx-decider-reference-vectors", withExtension: "json"))
        return try Data(contentsOf: url)
    }

    private func referenceDecision() throws -> [String: Any] {
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any])
        return try XCTUnwrap(root["decision"] as? [String: Any])
    }

    private func probabilityCases() throws -> [ProbabilityCase] {
        try JSONDecoder().decode(ProbabilityFixture.self, from: fixtureData()).probabilities
    }

    /// The reference block truncated to its first seven options.
    private func tinyDecision() throws -> [String: Any] {
        var block = try referenceDecision()
        let codes = try XCTUnwrap(block["codes"] as? [String])
        let tokenIDs = try XCTUnwrap(block["token_ids"] as? [Int])
        block["options"] = 7
        block["codes"] = Array(codes.prefix(7))
        block["token_ids"] = Array(tokenIDs.prefix(7))
        return block
    }

    /// Two layers (GDN, then full attention), hidden 32, vocabulary 64.
    /// GDN key head dim 32: the gated-delta Metal kernel needs Dk >= 32.
    private func tinyRoot(
        modelType: String = PplxDeciderConfiguration.modelType,
        decision: [String: Any]?,
        tieWordEmbeddings: Bool = false
    ) -> [String: Any] {
        var root: [String: Any] = [
            "model_type": modelType,
            "tie_word_embeddings": tieWordEmbeddings,
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 64,
                "full_attention_interval": 2,
                "mtp_num_hidden_layers": 1,
                "tie_word_embeddings": tieWordEmbeddings,
            ] as [String: Any],
            "vision_config": [
                "model_type": "qwen3_5",
                "depth": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "out_hidden_size": 32,
                "num_heads": 1,
                "patch_size": 16,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
                "num_position_embeddings": 8,
            ] as [String: Any],
        ]
        root["decision"] = decision
        return root
    }

    private func data(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object)
    }

    private func decodeLLM(_ root: [String: Any]) throws -> MLXLLM.Qwen35Configuration {
        try JSONDecoder.json5().decode(MLXLLM.Qwen35Configuration.self, from: data(root))
    }

    private func decodeVLM(_ root: [String: Any]) throws -> MLXVLM.Qwen35Configuration {
        try JSONDecoder.json5().decode(MLXVLM.Qwen35Configuration.self, from: data(root))
    }

    private func assertDeciderError(
        _ body: @autoclosure () throws -> Any, _ message: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(try body(), message, file: file, line: line) { error in
            XCTAssertTrue(
                error is PplxDeciderError, "\(message): \(error)", file: file, line: line)
        }
    }

    // MARK: - Configuration

    func testReferenceDecisionBlockDecodes() throws {
        let decision = try JSONDecoder().decode(
            PplxDeciderConfiguration.self, from: data(referenceDecision()))
        XCTAssertEqual(decision.formatVersion, 1)
        XCTAssertEqual(decision.options, 255)
        XCTAssertEqual(decision.codes.count, 255)
        XCTAssertEqual(decision.codes.prefix(3), ["A", "B", "C"])
        XCTAssertEqual(decision.codes[26], "AA")
        XCTAssertEqual(decision.codes.last, "JT")
        XCTAssertEqual(decision.tokenIDs.first, 32)
        XCTAssertEqual(decision.maxInputTokens, 8192)
        XCTAssertEqual(decision.temperature, Float(2.207568021892729))

        let again = try JSONDecoder().decode(
            PplxDeciderConfiguration.self, from: JSONEncoder().encode(decision))
        XCTAssertEqual(again, decision)
    }

    func testDeciderConfigurationAppliesTheOptionHeadSize() throws {
        let root = tinyRoot(decision: try tinyDecision())

        let llm = try decodeLLM(root)
        XCTAssertEqual(llm.modelType, PplxDeciderConfiguration.modelType)
        XCTAssertEqual(llm.decision?.options, 7)
        XCTAssertEqual(llm.textConfig.outputHeadSize, 7)
        XCTAssertEqual(llm.textConfig.vocabularySize, 64)
        XCTAssertFalse(llm.cbv2Capabilities.supportsMTP)

        let vlm = try decodeVLM(root)
        XCTAssertEqual(vlm.decision?.codes, ["A", "B", "C", "D", "E", "F", "G"])
        XCTAssertEqual(vlm.textConfiguration.outputHeadSize, 7)
        XCTAssertEqual(vlm.vocabSize, 64)
    }

    func testOrdinaryQwenConfigurationKeepsTheVocabularyHead() throws {
        let root = tinyRoot(modelType: "qwen3_5", decision: nil)
        let llm = try decodeLLM(root)
        XCTAssertNil(llm.decision)
        XCTAssertNil(llm.textConfig.outputHeadSize)
        XCTAssertTrue(llm.cbv2Capabilities.supportsMTP)
        XCTAssertNil(try decodeVLM(root).textConfiguration.outputHeadSize)
    }

    func testDecisionBlockUnderAnotherModelTypeIsRejected() throws {
        let root = tinyRoot(modelType: "qwen3_5", decision: try tinyDecision())
        assertDeciderError(try decodeLLM(root), "LLM qwen3_5 with a decision block")
        assertDeciderError(try decodeVLM(root), "VLM qwen3_5 with a decision block")
    }

    func testDeciderModelTypeRequiresAnUntiedDecisionHead() throws {
        let missing = tinyRoot(decision: nil)
        assertDeciderError(try decodeLLM(missing), "LLM decider without a decision block")
        assertDeciderError(try decodeVLM(missing), "VLM decider without a decision block")

        let tied = tinyRoot(decision: try tinyDecision(), tieWordEmbeddings: true)
        assertDeciderError(try decodeLLM(tied), "LLM tied decider head")
        assertDeciderError(try decodeVLM(tied), "VLM tied decider head")
    }

    func testInvalidDecisionBlocksAreRejected() throws {
        let valid = try tinyDecision()
        var cases: [(String, [String: Any])] = []
        func mutate(_ name: String, _ change: (inout [String: Any]) -> Void) {
            var block = valid
            change(&block)
            cases.append((name, block))
        }
        mutate("format_version 2") { $0["format_version"] = 2 }
        mutate("options 0") { $0["options"] = 0 }
        mutate("options 256") { $0["options"] = 256 }
        mutate("options/codes mismatch") { $0["options"] = 8 }
        mutate("token_ids count mismatch") { $0["token_ids"] = [32, 33, 34, 35, 36, 37] }
        mutate("duplicate codes") { $0["codes"] = ["A", "B", "C", "D", "E", "F", "A"] }
        mutate("lowercase code") { $0["codes"] = ["A", "B", "C", "D", "E", "F", "g"] }
        mutate("three-letter code") { $0["codes"] = ["A", "B", "C", "D", "E", "F", "AAA"] }
        mutate("empty code") { $0["codes"] = ["A", "B", "C", "D", "E", "F", ""] }
        mutate("duplicate token_ids") { $0["token_ids"] = [32, 33, 34, 35, 36, 37, 32] }
        mutate("negative token_id") { $0["token_ids"] = [32, 33, 34, 35, 36, 37, -1] }
        mutate("zero temperature") { $0["temperature"] = 0 }
        mutate("negative temperature") { $0["temperature"] = -1.5 }
        mutate("zero max_input_tokens") { $0["max_input_tokens"] = 0 }

        for (name, block) in cases {
            assertDeciderError(
                try JSONDecoder().decode(PplxDeciderConfiguration.self, from: data(block)),
                name)
            assertDeciderError(try decodeLLM(tinyRoot(decision: block)), "LLM \(name)")
            assertDeciderError(try decodeVLM(tinyRoot(decision: block)), "VLM \(name)")
        }

        let nonFinite = JSONDecoder()
        nonFinite.nonConformingFloatDecodingStrategy = .convertFromString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        for value in ["inf", "nan"] {
            var block = valid
            block["temperature"] = value
            assertDeciderError(
                try nonFinite.decode(PplxDeciderConfiguration.self, from: data(block)),
                "temperature \(value)")
        }
    }

    func testFactoriesMapTheDeciderModelType() async throws {
        XCTAssertEqual(
            VLMModelFactory.processorType(
                modelType: PplxDeciderConfiguration.modelType, declaredClass: "Other"),
            "Qwen3VLProcessor")

        let configuration = try data(tinyRoot(decision: try tinyDecision()))
        let llm = try await LLMTypeRegistry.shared.createModel(
            configuration: configuration, modelType: PplxDeciderConfiguration.modelType)
        let text = try XCTUnwrap(llm as? Qwen35Model)
        XCTAssertEqual(text.languageModel.lmHead?.weight.shape, [7, 32])

        let vlm = try await VLMTypeRegistry.shared.createModel(
            configuration: configuration, modelType: PplxDeciderConfiguration.modelType)
        XCTAssertTrue(vlm is MLXVLM.Qwen35)
    }

    // MARK: - Model shape

    func testTextModelHasTheOptionWideHead() throws {
        let config = try decodeLLM(tinyRoot(decision: try tinyDecision()))
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)

        let text = Qwen35TextModel(config.textConfig)
        XCTAssertEqual(text.lmHead?.weight.shape, [7, 32])
        XCTAssertEqual(text.vocabularySize, 64)
        XCTAssertNil(text.mtp)
        let logits = text(tokens, cache: text.newCache(parameters: nil))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 4, 7])

        let wrapper = Qwen35Model(config)
        XCTAssertEqual(wrapper.languageModel.lmHead?.weight.shape, [7, 32])
        XCTAssertEqual(wrapper.vocabularySize, 64)
        let wrapped = wrapper(tokens, cache: wrapper.newCache(parameters: nil))
        eval(wrapped)
        XCTAssertEqual(wrapped.shape, [1, 4, 7])
    }

    func testVLMHasTheOptionWideHead() throws {
        let config = try decodeVLM(tinyRoot(decision: try tinyDecision()))
        let model = MLXVLM.Qwen35(config)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(parameters["language_model.lm_head.weight"]?.shape, [7, 32])
        XCTAssertEqual(parameters["language_model.model.embed_tokens.weight"]?.shape, [64, 32])
        XCTAssertEqual(model.vocabularySize, 64)

        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let logits = model(tokens, cache: model.newCache(parameters: nil))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 4, 7])
    }

    // MARK: - Probabilities

    func testProbabilitiesMatchTheReferenceVectors() throws {
        let cases = try probabilityCases()
        XCTAssertEqual(cases.map(\.count), [1, 2, 3, 5, 26, 30, 255])
        for reference in cases {
            let logprobs = reference.engine_logprobs.prefix(reference.count).map { Float($0) }
            let probabilities = PplxDecision.probabilities(
                optionLogprobs: logprobs, temperature: Float(reference.temperature))
            XCTAssertEqual(probabilities.count, reference.count)
            for (actual, expected) in zip(probabilities, reference.probabilities) {
                XCTAssertEqual(Double(actual), expected, accuracy: 1e-5, "count \(reference.count)")
            }
            let total = probabilities.reduce(Double(0)) { $0 + Double($1) }
            XCTAssertEqual(total, 1, accuracy: 1e-5, "count \(reference.count)")
        }
        XCTAssertEqual(PplxDecision.probabilities(optionLogprobs: [], temperature: 1), [])
    }

    func testTopLogprobsBecomeOrderedOptionLogprobs() throws {
        let reference = try XCTUnwrap(try probabilityCases().first { $0.count == 26 })
        let all = reference.engine_logprobs.map { Float($0) }
        // Engine order: descending logprob over every head row.
        let top = all.enumerated()
            .sorted { $0.element > $1.element }
            .map { (token: $0.offset, logprob: $0.element) }

        let ordered = try PplxDecision.optionLogprobs(topLogprobs: top, count: 26)
        XCTAssertEqual(ordered, Array(all.prefix(26)))
        let probabilities = PplxDecision.probabilities(
            optionLogprobs: ordered, temperature: Float(reference.temperature))
        for (actual, expected) in zip(probabilities, reference.probabilities) {
            XCTAssertEqual(Double(actual), expected, accuracy: 1e-5)
        }

        // Rows at or above `count` are ignored; a missing row is an error.
        XCTAssertEqual(
            try PplxDecision.optionLogprobs(
                topLogprobs: [
                    (token: 300, logprob: 0), (token: 1, logprob: -2),
                    (token: 0, logprob: -1),
                ],
                count: 2),
            [-1, -2])
        let missing = top.filter { $0.token != 3 }
        assertDeciderError(
            try PplxDecision.optionLogprobs(topLogprobs: missing, count: 26), "missing row")
        assertDeciderError(
            try PplxDecision.optionLogprobs(topLogprobs: top + [top[0]], count: 255),
            "duplicate row")
        assertDeciderError(
            try PplxDecision.optionLogprobs(
                topLogprobs: [(token: 0, logprob: .nan)], count: 1),
            "non-finite row")
        assertDeciderError(
            try PplxDecision.optionLogprobs(topLogprobs: top, count: 0), "zero options")
        assertDeciderError(
            try PplxDecision.optionLogprobs(topLogprobs: top, count: 256), "too many options")
    }
}
