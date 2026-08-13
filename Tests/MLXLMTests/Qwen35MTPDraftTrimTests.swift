// Qwen35MTPDraftTrimTests.swift
//
// Draft-step trim coverage for the inline Qwen3.5/3.6 MTP assistant:
//  - the single-forward round (deferred draft pair) produces the same
//    draft tokens as the legacy double-forward staging oracle,
//  - rejection drops the deferred pair (carry rollback) without touching
//    canonical assistant KV,
//  - shortlisted draft head evaluation matches the full head on both the
//    float and the quantized (gathered-row qmv) paths.
//
// The assistant is built through the production `load` path from a
// fabricated inline artifact (real safetensors + config), never via
// test-only constructors.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

private func draftTrimConfigJSON(hiddenSize: Int) -> Data {
    Data(
        """
        {
          "model_type": "qwen3_5_moe",
          "mtplx_mtp": {
            "included": true,
            "prefix": "mtp.",
            "block_size": 3
          },
          "mtplx_mtp_quantization": {},
          "text_config": {
            "model_type": "qwen3_5_moe",
            "hidden_size": \(hiddenSize),
            "num_hidden_layers": 4,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "linear_num_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32,
            "head_dim": 8,
            "full_attention_interval": 4,
            "num_experts": 0,
            "num_experts_per_tok": 0,
            "mtp_num_hidden_layers": 1
          }
        }
        """.utf8)
}

/// Build a REAL loadable assistant: decode the text config, materialize a
/// randomly initialized `Qwen35MTPModule`, save its exact parameter set as
/// an indexed `mtp.*` shard, and load through the production path.
private func withDraftTrimAssistant(
    hiddenSize: Int = 8,
    _ body: (Qwen35InlineMTPAssistant, Qwen35TextModel) throws -> Void
) throws {
    let configData = draftTrimConfigJSON(hiddenSize: hiddenSize)
    let root = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
    let textData = try JSONSerialization.data(withJSONObject: root["text_config"]!)
    let configuration = try JSONDecoder().decode(
        Qwen35TextConfiguration.self, from: textData)

    MLXRandom.seed(1913)
    let donor = Qwen35MTPModule(configuration)
    var weights: [String: MLXArray] = [:]
    for (key, value) in donor.parameters().flattened() {
        weights["mtp." + key] = value
    }

    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwen-mtp-draft-trim-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try configData.write(to: directory.appendingPathComponent("config.json"))
    let shardName = "model-00001-of-00001.safetensors"
    try save(arrays: weights, url: directory.appendingPathComponent(shardName))
    let index = [
        "weight_map": Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shardName) })
    ]
    try JSONSerialization.data(withJSONObject: index)
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))

    let target = Qwen35TextModel(configuration)
    eval(target)
    let assistant = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
    try body(assistant, target)
}

/// The removed double-forward staging path, reproduced verbatim as the
/// parity oracle: proposal forward + full staging re-forward, rejection
/// trims the draft column.
private struct LegacyDoubleForwardOracle {
    let assistant: Qwen35InlineMTPAssistant
    let caches: [any KVCache]

    init(_ assistant: Qwen35InlineMTPAssistant) {
        self.assistant = assistant
        self.caches = assistant.makeCache()
    }

    func draft(seedToken: Int32, carryHidden: MLXArray) -> Int32 {
        let tokens = MLXArray([seedToken]).reshaped([1, 1])
        let output = assistant.forward(hidden: carryHidden, tokens: tokens, cache: caches)
        let draft = argMax(output.logits[0..., -1, 0...], axis: -1).asType(.int32)
        eval([draft, output.hidden] + caches.flatMap { $0.innerState() })
        _ = assistant.forward(
            hidden: output.hidden, tokens: draft.reshaped([1, 1]), cache: caches)
        eval(caches.flatMap { $0.innerState() })
        return draft.item(Int32.self)
    }

    func finalize(confirmed: Int) {
        let rollback = 2 - confirmed
        if rollback > 0 {
            for cache in caches {
                precondition(cache.trim(rollback) == rollback)
            }
        }
    }

    var offset: Int { caches.first?.offset ?? 0 }
}

@Suite("Qwen inline MTP draft trim", .serialized)
struct Qwen35MTPDraftTrimTests {

    private func drafterDraft(
        _ assistant: Qwen35InlineMTPAssistant,
        state: any CBv2MTPRequestState,
        seedToken: Int32, carryHidden: MLXArray,
        shortlist: MLXArray? = nil
    ) -> Int32 {
        let tokens = MLXArray([seedToken]).reshaped([1, 1])
        let result = assistant.draftStep(
            tokens: tokens, hidden: carryHidden, shortlist: shortlist,
            requestState: state)
        eval([result.tokens] + assistant.evaluationTargets(for: state))
        return result.tokens.item(Int32.self)
    }

    @Test("single-forward rounds match the double-forward oracle across accept/reject")
    func singleForwardMatchesDoubleForwardOracle() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(7)
            let state = assistant.makeRequestState()
            let oracle = LegacyDoubleForwardOracle(assistant)

            // Scripted round outcomes; the target's committed tokens and
            // captured hiddens are deterministic stand-ins fed identically
            // to both variants. accept ⇒ confirmed 2 (draft + bonus),
            // reject ⇒ confirmed 1 (replacement token).
            let outcomes: [Bool] = [true, true, false, true, false, true, false]
            var seed: Int32 = 5
            for (round, accepted) in outcomes.enumerated() {
                let carryHidden = MLXRandom.normal([1, 1, 8]) * 0.3
                let draftNew = drafterDraft(
                    assistant, state: state, seedToken: seed, carryHidden: carryHidden)
                let draftOracle = oracle.draft(seedToken: seed, carryHidden: carryHidden)
                #expect(draftNew == draftOracle, "round \(round)")

                let confirmed = accepted ? 2 : 1
                assistant.finalizeRound(
                    requestState: state, confirmedInputTokens: confirmed)
                oracle.finalize(confirmed: confirmed)
                // Next seed: bonus after an accept, replacement after a
                // reject — arbitrary but deterministic and shared.
                seed = accepted ? (draftNew + 7) % 32 : (draftNew + 11) % 32
            }

            // Terminal reject above ⇒ no deferred pair remains: assistant
            // KV holds exactly the canonical pairs on both variants.
            #expect(state.committedInputCount == oracle.offset)
            #expect(state.stagedInputCount == 0)
        }
    }

    @Test("rejection drops the deferred pair and leaves canonical KV intact")
    func rejectionRestoresCarry() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(11)
            let state = assistant.makeRequestState()
            let oracle = LegacyDoubleForwardOracle(assistant)

            let hidden0 = MLXRandom.normal([1, 1, 8]) * 0.3
            let draft0 = drafterDraft(
                assistant, state: state, seedToken: 3, carryHidden: hidden0)
            let oracleDraft0 = oracle.draft(seedToken: 3, carryHidden: hidden0)
            #expect(draft0 == oracleDraft0)
            // Only the seed pair entered assistant KV; the draft pair is
            // deferred, nothing is staged.
            #expect(state.committedInputCount == 1)
            #expect(state.stagedInputCount == 0)

            // Reject: confirmed == 1 clears the pending pair; the oracle
            // trims its eagerly staged draft column.
            assistant.finalizeRound(requestState: state, confirmedInputTokens: 1)
            oracle.finalize(confirmed: 1)
            #expect(state.committedInputCount == 1)
            #expect(state.committedInputCount == oracle.offset)

            // The next round must be indistinguishable from one that never
            // proposed draft0: the independent double-forward oracle is the
            // ground truth for "no residue of the rejected pair".
            let replacement = (draft0 + 5) % 32
            let hidden1 = MLXRandom.normal([1, 1, 8]) * 0.3
            let next = drafterDraft(
                assistant, state: state, seedToken: replacement, carryHidden: hidden1)
            let control = oracle.draft(seedToken: replacement, carryHidden: hidden1)
            #expect(next == control)
        }
    }

    @Test("acceptance feeds the deferred pair into the next single forward")
    func acceptanceAppendsDeferredPair() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(13)
            let state = assistant.makeRequestState()
            let hidden0 = MLXRandom.normal([1, 1, 8]) * 0.3
            let draft0 = drafterDraft(
                assistant, state: state, seedToken: 9, carryHidden: hidden0)
            #expect(state.committedInputCount == 1)
            assistant.finalizeRound(requestState: state, confirmedInputTokens: 2)
            // Pair still deferred: KV grows only inside the next forward.
            #expect(state.committedInputCount == 1)

            let bonus = (draft0 + 7) % 32
            let hidden1 = MLXRandom.normal([1, 1, 8]) * 0.3
            _ = drafterDraft(
                assistant, state: state, seedToken: bonus, carryHidden: hidden1)
            // [deferred draft pair, seed pair] both appended by ONE forward.
            #expect(state.committedInputCount == 3)
        }
    }

    @Test("discard clears the deferred pair")
    func discardClearsDeferredPair() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(17)
            let state = assistant.makeRequestState()
            _ = drafterDraft(
                assistant, state: state, seedToken: 4,
                carryHidden: MLXRandom.normal([1, 1, 8]) * 0.3)
            assistant.discardRound(requestState: state)
            #expect(state.stagedInputCount == 0)
            // A discarded round leaves canonical history usable: the next
            // draft appends exactly one seed pair.
            _ = drafterDraft(
                assistant, state: state, seedToken: 6,
                carryHidden: MLXRandom.normal([1, 1, 8]) * 0.3)
            #expect(state.committedInputCount == 2)
        }
    }

    @Test("full-coverage shortlist reproduces the full-head draft (float path)")
    func shortlistMatchesFullHeadFloat() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(19)
            let hidden = MLXRandom.normal([1, 1, 8]) * 0.3
            let full = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 12, carryHidden: hidden)
            // A permutation of the whole vocabulary must select the exact
            // same token through the gathered-row path AND map the local
            // argmax back to the true id.
            let permutation = MLXArray((0 ..< 32).shuffled().map(Int32.init))
            let shortlisted = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 12, carryHidden: hidden, shortlist: permutation)
            #expect(shortlisted == full)

            // A shortlist that excludes the true argmax constrains the
            // draft to the shortlist (coverage-miss behavior: the draft is
            // simply a worse proposal, never an out-of-list token).
            let excluding = MLXArray(
                (0 ..< 32).map(Int32.init).filter { $0 != full }.prefix(5).map { $0 })
            let constrained = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 12, carryHidden: hidden, shortlist: excluding)
            #expect(excluding.asArray(Int32.self).contains(constrained))
            #expect(constrained != full)
        }
    }

    @Test("shortlisted rows equal the full quantized head rows (gathered qmv)")
    func shortlistMatchesFullHeadQuantized() throws {
        // hidden 64 so the 4-bit group-32 quantized lm_head is legal.
        try withDraftTrimAssistant(hiddenSize: 64) { assistant, target in
            // Quantize ONLY the shared output head — exactly the production
            // shape (Qwen3.6 serves a 4-bit lm_head; the MTP module keeps
            // its own dtype).
            quantize(model: target, groupSize: 32, bits: 4) { path, module in
                path == "lm_head" && module is Linear
            }
            #expect(target.lmHead is QuantizedLinear)

            MLXRandom.seed(23)
            let hidden = MLXRandom.normal([1, 1, 64]) * 0.3

            // Row-level check: gathered-row quantized logits must equal the
            // full quantized head at those ids.
            let ids = MLXArray([3, 31, 7, 19, 0, 27].map(Int32.init))
            let gathered = assistant.shortlistLogits(hidden: hidden, ids: ids)
            let full = assistant.headLogits(hidden)
            let reference = takeAlong(full, ids.reshaped([1, 1, -1]), axis: -1)
            #expect(allClose(gathered, reference, atol: 1e-5).item(Bool.self))

            // Token-level check through draftStep.
            let fullDraft = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 21, carryHidden: hidden)
            let permutation = MLXArray((0 ..< 32).shuffled().map(Int32.init))
            let shortlisted = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 21, carryHidden: hidden, shortlist: permutation)
            #expect(shortlisted == fullDraft)
        }
    }
}
