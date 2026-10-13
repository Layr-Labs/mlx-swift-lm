// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Real-artifact qualification for LiquidAI/LFM2.5-24B-A2B-MLX-4bit against
/// the mlx-lm 0.32.0 reference outputs. Env-gated; a skip is NOT a pass.
///
///   DARKBLOOM_LFM25_REAL_MODEL   verified snapshot directory (667408d)
///   DARKBLOOM_LFM25_REFERENCE    directory of reference JSON + .final.f32
///   DARKBLOOM_LFM25_EVIDENCE_OUT optional JSON path for measured numbers
///
/// Tolerances: both sides run bf16 activations over 4-bit/6-bit affine
/// weights, but on different MLX cores (mlx 0.32.3 wheel vs the pinned Swift
/// MLX fork) whose quantized matmul/gather and attention kernels reduce in
/// different orders. A bf16 ulp is 0.125 at |logit| in [16, 32), so
/// differences of a few ulps per position are expected after 40 layers; an
/// implementation error (routing, conv state, RoPE, masking) instead moves
/// logits by whole units and flips the argmax broadly.
final class LFM2MoERealCheckpointTests: XCTestCase {
    private struct TopK: Decodable { let ids: [Int]; let logits: [Float] }
    private struct Position: Decodable { let position: Int; let ids: [Int]; let logits: [Float] }
    private struct Final: Decodable { let argmax: Int; let max: Float }
    private struct Reference: Decodable {
        let name: String
        let prompt_tokens: [Int]
        let positions: [Position]
        let final: Final
        let final_logits_file: String
        let greedy_tokens: [Int]
        let greedy_step_topk: [TopK]
    }

    private struct Comparison: Encodable {
        var positions = 0
        var top1Agreements = 0
        var maxAbs: Float = 0
        var sumAbs: Double = 0
        var count = 0
        var perPosition: [[Double]] = []  // [index, max |d|, top-1 match]
        var meanAbs: Double { count == 0 ? 0 : sumAbs / Double(count) }
        var top1Fraction: Double { Double(top1Agreements) / Double(max(1, positions)) }

        mutating func add(_ row: MLXArray, ids: [Int], logits: [Float], index: Int) {
            let values = row.asType(.float32)[MLXArray(ids.map(Int32.init))].asArray(Float.self)
            var rowMax: Float = 0
            for (a, b) in zip(values, logits) {
                rowMax = max(rowMax, abs(a - b))
                sumAbs += Double(abs(a - b))
                count += 1
            }
            maxAbs = max(maxAbs, rowMax)
            positions += 1
            let hit = row.argMax().item(Int.self) == ids[0]
            if hit { top1Agreements += 1 }
            perPosition.append([Double(index), Double(rowMax), hit ? 1 : 0])
        }

        func summary() -> [String: Any] {
            ["positions": positions, "top1_agreements": top1Agreements,
             "max_abs": Double(maxAbs), "mean_abs": meanAbs, "per_position": perPosition]
        }
    }

    private func environment() throws -> (model: URL, reference: URL) {
        let env = ProcessInfo.processInfo.environment
        guard let model = env["DARKBLOOM_LFM25_REAL_MODEL"], !model.isEmpty,
            let reference = env["DARKBLOOM_LFM25_REFERENCE"], !reference.isEmpty
        else {
            throw XCTSkip("Set DARKBLOOM_LFM25_REAL_MODEL and DARKBLOOM_LFM25_REFERENCE")
        }
        return (URL(fileURLWithPath: model), URL(fileURLWithPath: reference))
    }

    private func loadReferences(_ directory: URL) throws -> [Reference] {
        try ["chat_short", "chat_1k", "chat_4k", "raw_text"].map {
            try JSONDecoder().decode(
                Reference.self,
                from: Data(contentsOf: directory.appendingPathComponent("\($0).json")))
        }
    }

    private func finalLogits(_ directory: URL, _ reference: Reference) throws -> MLXArray {
        let data = try Data(contentsOf: directory.appendingPathComponent(reference.final_logits_file))
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return MLXArray(floats)
    }

    private func productionLoad(_ directory: URL) async throws -> LFM2MoEModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)
        try loadWeights(
            modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
        return try XCTUnwrap(model as? LFM2MoEModel)
    }

    private func greedyMatch(_ a: [Int], _ b: [Int]) -> Int {
        zip(a, b).prefix { $0 == $1 }.count
    }

    func testNativeSerialAndCBv2PagedMatchMLXLMReference() async throws {
        let (modelURL, referenceURL) = try environment()
        let model = try await productionLoad(modelURL)
        eval(model.parameters().flattened().map(\.1))
        let references = try loadReferences(referenceURL)

        // Loaded artifact contract.
        let spec = model.cbv2RecurrentStateSpec
        XCTAssertEqual(spec.layers.count, 30)
        XCTAssertTrue(spec.layers.allSatisfy { $0.convDType == .bfloat16 && !$0.hasSSMState })
        XCTAssertEqual(try spec.fixedBytesPerRequest(), 245_760)
        XCTAssertEqual(model.cbv2CompleteCheckpointKVDTypes, Array(repeating: .bfloat16, count: 10))
        let embedding = try XCTUnwrap(model.model.embedTokens as? QuantizedEmbedding)
        XCTAssertEqual(embedding.bits, 6)
        var routerTensors = 0
        for layer in model.model.layers {
            guard let moe = layer.feedForward as? Lfm2MoeSparseMoeBlock else { continue }
            XCTAssertFalse(moe.gate is QuantizedLinear)
            XCTAssertEqual(moe.gate.weight.dtype, .float32)
            XCTAssertEqual(moe.expertBias?.dtype, .float32)
            routerTensors += 2
        }
        XCTAssertEqual(routerTensors, 76)

        var evidence: [String: Any] = ["model": modelURL.lastPathComponent]
        let decodeSteps = 64
        let chunk = 512

        for reference in references {
            let prompt = reference.prompt_tokens
            var record: [String: Any] = ["prompt_tokens": prompt.count]

            // 1. Native serial: prompt positions, final full vector, teacher-
            //    forced decode, free-running greedy.
            let caches = model.newCache(parameters: nil)
            let promptLogits = model(MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: caches)[0]
            eval(promptLogits)
            var serialPrompt = Comparison()
            for position in reference.positions {
                serialPrompt.add(
                    promptLogits[position.position], ids: position.ids, logits: position.logits,
                    index: position.position)
            }
            let expectedFinal = try finalLogits(referenceURL, reference)
            let serialFinal = promptLogits[-1].asType(.float32)
            let finalDiff = abs(serialFinal - expectedFinal)
            record["serial_final"] = [
                "max_abs": Double(finalDiff.max().item(Float.self)),
                "mean_abs": Double(finalDiff.mean().item(Float.self)),
                "argmax_match": serialFinal.argMax().item(Int.self) == reference.final.argmax ? 1.0 : 0.0,
            ]
            var serialForced = Comparison()
            for (step, token) in reference.greedy_tokens.prefix(decodeSteps).enumerated() {
                let row = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: caches)[0, 0]
                eval(row)
                serialForced.add(row, ids: reference.greedy_step_topk[step].ids,
                                 logits: reference.greedy_step_topk[step].logits, index: step)
            }
            let greedyCaches = model.newCache(parameters: nil)
            var logits = model(MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: greedyCaches)
            var greedy: [Int] = []
            for _ in 0 ..< decodeSteps {
                let token = logits[0, -1].argMax().item(Int.self)
                greedy.append(token)
                logits = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: greedyCaches)
            }
            record["serial_prompt_positions"] = serialPrompt.summary()
            record["serial_teacher_forced_decode"] = serialForced.summary()
            record["serial_greedy_match_length"] = greedyMatch(greedy, reference.greedy_tokens)

            // 2. CBv2 on native paged KV, chunked prefill, teacher-forced
            //    decode, then free-running greedy decode.
            let storage = try LFM2MoECBv2Storage(
                model: model, paged: true, maximumLength: prompt.count + decodeSteps + 8,
                capacity: 2 << 30, maxPrefillChunk: chunk)
            let request = try LFM2MoECBv2Request(
                model: model, storage: storage, promptLength: prompt.count,
                maximumLength: prompt.count + decodeSteps + 8)
            let cbPrompt = try LFM2MoECBv2Driver.prefill(
                model, storage: storage, request: request, prompt: prompt, chunk: chunk)
            var cbPromptCmp = Comparison()
            for position in reference.positions {
                cbPromptCmp.add(
                    cbPrompt[position.position], ids: position.ids, logits: position.logits,
                    index: position.position)
            }
            let cbFinal = cbPrompt[-1].asType(.float32)
            let cbDiff = abs(cbFinal - expectedFinal)
            let cbVsSerial = abs(cbFinal - serialFinal)
            record["cbv2_paged_final"] = [
                "max_abs": Double(cbDiff.max().item(Float.self)),
                "mean_abs": Double(cbDiff.mean().item(Float.self)),
                "argmax_match": cbFinal.argMax().item(Int.self) == reference.final.argmax ? 1.0 : 0.0,
                "max_abs_vs_native_serial": Double(cbVsSerial.max().item(Float.self)),
            ]
            var cbForced = Comparison()
            for (step, token) in reference.greedy_tokens.prefix(decodeSteps).enumerated() {
                let row = try LFM2MoECBv2Driver.forward(
                    model, storage: storage, requests: [request], tokens: [[token]]).logits[0, 0]
                cbForced.add(row, ids: reference.greedy_step_topk[step].ids,
                             logits: reference.greedy_step_topk[step].logits, index: step)
            }
            storage.bank.releaseBoundRows()
            request.close()
            let greedyRequest = try LFM2MoECBv2Request(
                model: model, storage: storage, promptLength: prompt.count,
                maximumLength: prompt.count + decodeSteps + 8)
            var cbLast = try LFM2MoECBv2Driver.prefill(
                model, storage: storage, request: greedyRequest, prompt: prompt, chunk: chunk)[-1]
            var cbGreedy: [Int] = []
            for _ in 0 ..< decodeSteps {
                let token = cbLast.argMax().item(Int.self)
                cbGreedy.append(token)
                cbLast = try LFM2MoECBv2Driver.forward(
                    model, storage: storage, requests: [greedyRequest], tokens: [[token]]).logits[0, 0]
            }
            storage.bank.releaseBoundRows()
            greedyRequest.close()
            XCTAssertEqual(storage.backend.bytesReserved, 0)
            record["cbv2_prompt_positions"] = cbPromptCmp.summary()
            record["cbv2_teacher_forced_decode"] = cbForced.summary()
            record["cbv2_greedy_match_length"] = greedyMatch(cbGreedy, reference.greedy_tokens)
            record["cbv2_greedy_vs_native_serial_match_length"] = greedyMatch(cbGreedy, greedy)
            evidence[reference.name] = record

            // Assertions. Bounds are calibrated against the reference's OWN
            // bf16 self-consistency (tools/reference/lfm2moe_selfcheck.py:
            // mlx-lm chunked vs one-shot prefill of the same artifact gives
            // prompt-position top-1 76-92%, mean |dlogit| 0.55-0.67 (raw text
            // 67%, 1.29) and teacher-forced decode top-1 98-100%, mean
            // 0.17-0.24 (raw text 0.74)). A routing, conv-state, RoPE or mask
            // error moves every position, not a few near-ties.
            let raw = reference.name == "raw_text"
            XCTAssertEqual(serialFinal.argMax().item(Int.self), reference.final.argmax, reference.name)
            XCTAssertEqual(cbFinal.argMax().item(Int.self), reference.final.argmax, reference.name)
            for (label, cmp) in [("serial prompt", serialPrompt), ("cbv2 prompt", cbPromptCmp)] {
                XCTAssertGreaterThanOrEqual(
                    cmp.top1Fraction, raw ? 0.6 : 0.75, "\(reference.name) \(label) top-1")
                XCTAssertLessThanOrEqual(cmp.meanAbs, raw ? 3.0 : 1.0, "\(reference.name) \(label) mean")
            }
            for (label, cmp) in [("serial decode", serialForced), ("cbv2 decode", cbForced)] {
                XCTAssertGreaterThanOrEqual(
                    cmp.top1Fraction, raw ? 0.85 : 0.95, "\(reference.name) \(label) top-1")
                XCTAssertLessThanOrEqual(cmp.meanAbs, raw ? 1.5 : 0.4, "\(reference.name) \(label) mean")
            }
            if !raw {
                XCTAssertGreaterThanOrEqual(
                    greedyMatch(cbGreedy, reference.greedy_tokens), 16,
                    "\(reference.name): CBv2 greedy diverged early")
            }
        }

        // 3. Rectangular CBv2 decode B=4 (all prompts together, different
        //    lengths), teacher-forced, against the reference rows.
        let storage = try LFM2MoECBv2Storage(
            model: model, paged: true, maximumLength: 3111 + decodeSteps + 8, capacity: 4 << 30,
            maxPrefillChunk: chunk)
        let requests = try references.map {
            try LFM2MoECBv2Request(
                model: model, storage: storage, promptLength: $0.prompt_tokens.count,
                maximumLength: $0.prompt_tokens.count + decodeSteps + 8)
        }
        for (reference, request) in zip(references, requests) {
            _ = try LFM2MoECBv2Driver.prefill(
                model, storage: storage, request: request, prompt: reference.prompt_tokens, chunk: chunk)
        }
        var batched = Array(repeating: Comparison(), count: references.count)
        for step in 0 ..< 32 {
            let order = step % 2 == 0 ? Array(0 ..< references.count) : Array((0 ..< references.count).reversed())
            let logits = try LFM2MoECBv2Driver.forward(
                model, storage: storage, requests: order.map { requests[$0] },
                tokens: order.map { [references[$0].greedy_tokens[step]] }).logits
            for (row, index) in order.enumerated() {
                batched[index].add(
                    logits[row, 0], ids: references[index].greedy_step_topk[step].ids,
                    logits: references[index].greedy_step_topk[step].logits, index: step)
            }
        }
        storage.bank.releaseBoundRows()
        for request in requests { request.close() }
        XCTAssertEqual(storage.backend.bytesReserved, 0)
        var batchedEvidence: [String: Any] = [:]
        for (index, reference) in references.enumerated() {
            batchedEvidence[reference.name] = batched[index].summary()
            let raw = reference.name == "raw_text"
            XCTAssertGreaterThanOrEqual(
                batched[index].top1Fraction, raw ? 0.8 : 0.95,
                "\(reference.name) batched B=4 top-1 agreement")
            XCTAssertLessThanOrEqual(
                batched[index].meanAbs, raw ? 1.5 : 0.4, "\(reference.name) batched B=4")
        }
        evidence["cbv2_paged_batched_b4_teacher_forced_32_steps"] = batchedEvidence

        // 4. Discriminating power on the real router weights: the legacy
        //    softmax + bias-in-weights router (mlx-lm <= 0.31.3) selects a
        //    different expert set for many token/layer pairs.
        let study = routerStudy(model, prompt: references[2].prompt_tokens)
        evidence["router_legacy_vs_reference"] = study
        XCTAssertGreaterThan(study["pairs_with_different_expert_set"] ?? 0, 0)

        if let out = ProcessInfo.processInfo.environment["DARKBLOOM_LFM25_EVIDENCE_OUT"] {
            let data = try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: out))
        }
        print("LFM25_REAL_EVIDENCE " + String(
            decoding: try JSONSerialization.data(withJSONObject: evidence, options: [.sortedKeys]),
            as: UTF8.self))
    }

    /// Runs the serial layers by hand to observe each MoE block's input, and
    /// compares the production route with the legacy router on it.
    private func routerStudy(_ model: LFM2MoEModel, prompt: [Int]) -> [String: Double] {
        let inner = model.model
        let caches = model.newCache(parameters: nil)
        var hidden = inner.embedTokens(MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count))
        let mask = createAttentionMask(h: hidden, cache: caches[inner.firstAttentionIndex!])
        var differing = 0
        var total = 0
        for (index, layer) in inner.layers.enumerated() {
            let normed = layer.operatorNorm(hidden)
            let r = layer.isAttentionLayer
                ? layer.attention!(normed, mask: mask, cache: caches[index])
                : layer.conv!(normed, mask: nil, cache: caches[index] as? MambaCache)
            let h = hidden + r
            let ffnInput = layer.ffnNorm(h)
            if let moe = layer.feedForward as? Lfm2MoeSparseMoeBlock {
                let current = MLX.sorted(moe.route(ffnInput).indices, axis: -1)
                var gates = softmax(moe.gate(ffnInput).asType(.float32), axis: -1, precise: true)
                gates = gates + moe.expertBias!.asType(.float32)
                let k = moe.topK
                let legacy = MLX.sorted(argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k], axis: -1)
                let differs = (current .!= legacy).any(axis: -1)
                eval(differs)
                differing += differs.asType(.int32).sum().item(Int.self)
                total += differs.size
            }
            hidden = h + layer.feedForward(ffnInput)
            eval(hidden)
        }
        return ["token_layer_pairs": Double(total), "pairs_with_different_expert_set": Double(differing),
                "fraction_different": Double(differing) / Double(max(1, total))]
    }
}
