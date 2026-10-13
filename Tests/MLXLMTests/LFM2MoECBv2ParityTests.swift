// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// CBv2 adapter vs the serial LFM2-MoE oracle on tiny random-weight models.
/// The serial path (KVCacheSimple + MambaCache) is the independent oracle;
/// CBv2 runs through the production steppable adapter on contiguous and
/// native paged KV. Attention reductions differ between the backends, so
/// logits are compared with precision-specific tolerances (float32 tight,
/// bf16/quantized bounded by bf16 rounding of the activations), while
/// same-backend repeats are required to be bit-identical.
final class LFM2MoECBv2ParityTests: XCTestCase {
    private func tolerance(_ precision: LFM2MoEFixture.Precision) -> Float {
        switch precision {
        case .float32: return 2e-4
        // bf16 keeps 8 mantissa bits: a few ulps of O(1) logits after six
        // layers of differently ordered attention reductions.
        case .bfloat16, .quantizedMixed: return 0.08
        }
    }

    func testSurfaceDeclaresHybridLayoutAndCapabilities() throws {
        let model = try LFM2MoEFixture.model(.bfloat16)
        let kinds = model.cbv2LayerKinds
        XCTAssertEqual(kinds.map(\.modelLayerIndex), [1, 4])
        XCTAssertTrue(
            kinds.allSatisfy { $0.headDim == 64 && $0.kvHeads == 1 && $0.queryHeads == 4 })
        let spec = model.cbv2RecurrentStateSpec
        XCTAssertEqual(spec.modelLayerIndices, [0, 2, 3, 5])
        for layer in spec.layers {
            XCTAssertEqual(layer.convShape, [1, 2, 256])
            XCTAssertEqual(layer.convDType, .bfloat16, "conv history uses the activation dtype")
            XCTAssertFalse(layer.hasSSMState)
        }
        XCTAssertEqual(try spec.fixedBytesPerRequest(), 4 * 2 * 256 * 2)
        XCTAssertEqual(model.cbv2CompleteCheckpointKVDTypes, [.bfloat16, .bfloat16])
        XCTAssertEqual(
            try LFM2MoEFixture.model(.float32).cbv2RecurrentStateSpec.layers.first?.convDType,
            .float32)

        let caps = model.cbv2Capabilities
        XCTAssertTrue(caps.supportsRecurrentCheckpointReuse)
        XCTAssertTrue(caps.supportsPagedKV)
        XCTAssertTrue(caps.requiresNativePagedKV)
        XCTAssertFalse(caps.supportsMTP, "no LFM2-MoE drafter exists")
        XCTAssertFalse(caps.supportsCompactRecurrentMTPReplay)
        XCTAssertFalse(caps.supportsCompiledDecode)
        XCTAssertFalse(caps.supportsPackedPrefill)
        XCTAssertFalse(
            caps.supportsPrefixReuse, "attention-only prefix reuse would skip conv state")

        let adapter = CBv2SteppableLanguageModelAdapter(model)
        XCTAssertEqual(adapter.recurrentStateSpec, spec)
        XCTAssertFalse(adapter.supportsPackedPrefill)
        XCTAssertNil(adapter.mtpCaptureLayers, "no MTP capture surface")
        XCTAssertFalse(adapter.supportsRequestStatefulMTP)

        // The real artifact's geometry.
        let real = try JSONDecoder().decode(
            LFM2MoEConfiguration.self, from: Data(LFM2MoEArtifactConfig.json.utf8))
        XCTAssertEqual(real.ropeTheta, 1e7, "rope_parameters.rope_theta wins over the 1e6 default")
        XCTAssertEqual(real.headDimensions, 64)
        XCTAssertEqual(real.fullAttnIdxs, Array(stride(from: 2, to: 40, by: 4)))
        XCTAssertEqual(real.numDenseLayers, 2)
        XCTAssertEqual(real.convLCache, 3)
        XCTAssertEqual(
            real.cbv2LayerKinds.map(\.modelLayerIndex), Array(stride(from: 2, to: 40, by: 4)))
        XCTAssertTrue(
            real.cbv2LayerKinds.allSatisfy {
                $0.headDim == 64 && $0.kvHeads == 8 && $0.queryHeads == 32 && $0.attention == .full
            })
        let realSpec = real.cbv2RecurrentStateSpec()
        XCTAssertEqual(realSpec.layers.count, 30)
        XCTAssertTrue(
            realSpec.layers.allSatisfy { $0.convShape == [1, 2, 2048] && !$0.hasSSMState })
        XCTAssertEqual(try realSpec.fixedBytesPerRequest(), 30 * 2 * 2048 * 2)
    }

    func testFloat32SerialMatchesCBv2PrefillAndDecodeOnBothBackends() throws {
        try prefillAndDecodeParity(.float32)
    }

    func testBFloat16SerialMatchesCBv2PrefillAndDecodeOnBothBackends() throws {
        try prefillAndDecodeParity(.bfloat16)
    }

    func testQuantizedMixedSerialMatchesCBv2PrefillAndDecodeOnBothBackends() throws {
        try prefillAndDecodeParity(.quantizedMixed)
    }

    /// One-shot and chunked prompts (including chunks shorter than the conv
    /// history and a chunk that crosses the attention query block) followed by
    /// forced decode, against the serial oracle; repeats are bit-identical.
    private func prefillAndDecodeParity(_ precision: LFM2MoEFixture.Precision) throws {
        let model = try LFM2MoEFixture.model(precision)
        let prompt = LFM2MoEFixture.prompt(LFM2MoEFixture.chunk + 45)
        let decode = [5, 77, 13, 42, 8]
        let oracle = LFM2MoEFixture.serial(model, prompt: prompt, decode: decode)
        let tol = tolerance(precision)
        for paged in [false, true] {
            var firstRun: MLXArray?
            for chunk in [prompt.count, 1, 2, 61, LFM2MoEFixture.chunk] {
                if chunk == 1 && prompt.count > 64 {
                    // Token-by-token prompt on a short prefix is enough to
                    // cover chunk < L-1 and keeps the suite fast.
                    let short = Array(prompt.prefix(9))
                    let shortOracle = LFM2MoEFixture.serial(
                        model, prompt: short, decode: [],
                        prefillChunk: precision == .float32 ? nil : 1)
                    let storage = try LFM2MoECBv2Storage(
                        model: model, paged: paged, maximumLength: 32)
                    let request = try LFM2MoECBv2Request(
                        model: model, storage: storage, promptLength: short.count, maximumLength: 32
                    )
                    defer {
                        storage.bank.releaseBoundRows()
                        request.close()
                    }
                    let logits = try LFM2MoECBv2Driver.prefill(
                        model, storage: storage, request: request, prompt: short, chunk: 1)
                    let diff = LFM2MoEFixture.maxAbsDifference(logits, shortOracle.prompt)
                    XCTAssertLessThanOrEqual(
                        diff, tol, "\(precision) paged=\(paged) token-by-token prompt diff \(diff)")
                    continue
                }
                let storage = try LFM2MoECBv2Storage(
                    model: model, paged: paged, maximumLength: prompt.count + decode.count + 8)
                let request = try LFM2MoECBv2Request(
                    model: model, storage: storage, promptLength: prompt.count,
                    maximumLength: prompt.count + decode.count + 8)
                defer {
                    storage.bank.releaseBoundRows()
                    request.close()
                    XCTAssertEqual(storage.backend.bytesReserved, 0)
                }
                let logits = try LFM2MoECBv2Driver.prefill(
                    model, storage: storage, request: request, prompt: prompt, chunk: chunk)
                // float32: chunking is numerically invisible, so every CBv2
                // chunking must match the ONE-SHOT serial oracle. Reduced
                // precision: chunk boundaries legitimately change bf16
                // attention rounding, which discrete MoE routing can amplify
                // in a random tiny model, so the oracle is the serial path
                // with the SAME chunk boundaries.
                let chunkOracle =
                    precision == .float32 || chunk == prompt.count
                    ? oracle
                    : LFM2MoEFixture.serial(
                        model, prompt: prompt, decode: decode, prefillChunk: chunk)
                let diff = LFM2MoEFixture.maxAbsDifference(logits, chunkOracle.prompt)
                XCTAssertLessThanOrEqual(
                    diff, tol,
                    "\(precision) paged=\(paged) chunk=\(chunk) prompt logits diff \(diff)")
                XCTAssertEqual(
                    logits[-1].argMax().item(Int.self),
                    chunkOracle.prompt[-1].argMax().item(Int.self),
                    "\(precision) paged=\(paged) chunk=\(chunk) final argmax")
                var stepRows: [MLXArray] = []
                for (step, token) in decode.enumerated() {
                    let row = try LFM2MoECBv2Driver.forward(
                        model, storage: storage, requests: [request], tokens: [[token]]
                    ).logits[0, 0]
                    stepRows.append(row)
                    let stepDiff = LFM2MoEFixture.maxAbsDifference(row, chunkOracle.decode[step])
                    XCTAssertLessThanOrEqual(
                        stepDiff, tol,
                        "\(precision) paged=\(paged) chunk=\(chunk) decode \(step) diff \(stepDiff)"
                    )
                }
                // The request's conv history equals the last L-1 gated inputs:
                // offsets advanced exactly once per consumed token.
                for row in request.rows.compactMap({ $0 }) {
                    XCTAssertEqual(row.absoluteOffset, prompt.count + decode.count)
                }
                if chunk == prompt.count {
                    firstRun = concatenated(
                        [logits] + stepRows.map { $0.expandedDimensions(axis: 0) })
                } else if chunk == LFM2MoEFixture.chunk, let firstRun {
                    // Same backend, same chunk boundaries => bit-identical repeat.
                    let repeatStorage = try LFM2MoECBv2Storage(
                        model: model, paged: paged, maximumLength: prompt.count + decode.count + 8)
                    let again = try LFM2MoECBv2Request(
                        model: model, storage: repeatStorage, promptLength: prompt.count,
                        maximumLength: prompt.count + decode.count + 8)
                    defer {
                        repeatStorage.bank.releaseBoundRows()
                        again.close()
                    }
                    let repeated = try LFM2MoECBv2Driver.prefill(
                        model, storage: repeatStorage, request: again, prompt: prompt, chunk: chunk)
                    XCTAssertEqual(
                        repeated.asData(access: .copy).data, logits.asData(access: .copy).data,
                        "\(precision) paged=\(paged) repeated chunked prefill must be bit-identical"
                    )
                    _ = firstRun
                }
            }
        }
    }

    /// The engine's narrowed prompt output equals the full forward's last row.
    func testRecurrentPrefillNarrowingMatchesFullForward() throws {
        let model = try LFM2MoEFixture.model(.float32)
        let prompt = LFM2MoEFixture.prompt(40)
        let storage = try LFM2MoECBv2Storage(model: model, paged: false, maximumLength: 64)
        let full = try LFM2MoECBv2Request(
            model: model, storage: storage, promptLength: 40, maximumLength: 64)
        let reference = try LFM2MoECBv2Driver.forward(
            model, storage: storage, requests: [full], tokens: [prompt]
        ).logits[0, -1]
        storage.bank.releaseBoundRows()
        full.close()

        let narrowed = try LFM2MoECBv2Request(
            model: model, storage: storage, promptLength: 40, maximumLength: 64)
        defer {
            storage.bank.releaseBoundRows()
            narrowed.close()
        }
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let caches = storage.bank.layerCaches(rowStates: [narrowed.rows])
        let binding = try narrowed.recurrent.bind()
        let last = adapter.recurrentPrefill(
            tokens: MLXArray(prompt.map(Int32.init)).reshaped(1, 40), inputEmbeddings: nil,
            caches: caches, recurrentState: [binding], positionIds: nil,
            requirement: .lastPositionLogits)
        eval([last] + (try binding.evaluate()))
        try binding.commit()
        XCTAssertEqual(last.shape, [1, model.vocabularySize])
        // Only the head GEMM shape differs (one row instead of 40).
        XCTAssertLessThanOrEqual(LFM2MoEFixture.maxAbsDifference(last[0], reference), 1e-5)
        XCTAssertEqual(last[0].argMax().item(Int.self), reference.argMax().item(Int.self))
    }

    /// Rectangular decode with B = 1...4 rows of different lengths, rows
    /// reordered and removed between steps; every row must match its own
    /// serial oracle (no cross-row contamination of KV or conv state).
    func testBatchedDecodeVariableLengthsReorderRemovalAndIsolation() throws {
        for precision in [LFM2MoEFixture.Precision.float32, .quantizedMixed] {
            let model = try LFM2MoEFixture.model(precision)
            let lengths = [5, 17, 40, 93]
            let prompts = lengths.enumerated().map {
                LFM2MoEFixture.prompt($0.element, salt: $0.offset)
            }
            let schedule: [[Int]] = [
                [0], [0, 1], [0, 1, 2, 3], [2, 0, 3, 1], [3, 0, 2], [2], [1, 3], [3, 2, 1, 0],
            ]
            var forced: [[Int]] = Array(repeating: [], count: 4)
            for (step, rows) in schedule.enumerated() {
                for row in rows { forced[row].append(1 + (step * 13 + row * 29) % 90) }
            }
            let oracles = (0 ..< 4).map {
                LFM2MoEFixture.serial(model, prompt: prompts[$0], decode: forced[$0])
            }
            for paged in [false, true] {
                let storage = try LFM2MoECBv2Storage(model: model, paged: paged, maximumLength: 128)
                let requests = try lengths.map {
                    try LFM2MoECBv2Request(
                        model: model, storage: storage, promptLength: $0, maximumLength: 128)
                }
                defer {
                    storage.bank.releaseBoundRows()
                    for request in requests { request.close() }
                    XCTAssertEqual(storage.backend.bytesReserved, 0)
                }
                for (index, request) in requests.enumerated() {
                    _ = try LFM2MoECBv2Driver.prefill(
                        model, storage: storage, request: request, prompt: prompts[index],
                        chunk: LFM2MoEFixture.chunk)
                }
                var cursor = Array(repeating: 0, count: 4)
                for (step, rows) in schedule.enumerated() {
                    let tokens = rows.map { [forced[$0][cursor[$0]]] }
                    let logits = try LFM2MoECBv2Driver.forward(
                        model, storage: storage, requests: rows.map { requests[$0] }, tokens: tokens
                    ).logits
                    XCTAssertEqual(logits.shape, [rows.count, 1, model.vocabularySize])
                    for (position, row) in rows.enumerated() {
                        let diff = LFM2MoEFixture.maxAbsDifference(
                            logits[position, 0], oracles[row].decode[cursor[row]])
                        XCTAssertLessThanOrEqual(
                            diff, tolerance(precision),
                            "\(precision) paged=\(paged) step \(step) row \(row) (B=\(rows.count)) diff \(diff)"
                        )
                        cursor[row] += 1
                    }
                }
            }
        }
    }

    /// Clone = adopt another request's confirmed conv snapshot; reset = a
    /// fresh zero history. Rollback restores the exact prior state.
    func testRecurrentStateCloneResetAndRollback() throws {
        let model = try LFM2MoEFixture.model(.bfloat16)
        let prompt = LFM2MoEFixture.prompt(30)
        let storage = try LFM2MoECBv2Storage(model: model, paged: false, maximumLength: 64)
        let donor = try LFM2MoECBv2Request(
            model: model, storage: storage, promptLength: 30, maximumLength: 64)
        let clone = try LFM2MoECBv2Request(
            model: model, storage: storage, promptLength: 30, maximumLength: 64)
        defer {
            storage.bank.releaseBoundRows()
            donor.close()
            clone.close()
            XCTAssertEqual(storage.backend.bytesReserved, 0)
        }
        for request in [donor, clone] {
            _ = try LFM2MoECBv2Driver.prefill(
                model, storage: storage, request: request, prompt: prompt, chunk: 30)
        }
        let snapshot = try XCTUnwrap(donor.recurrent.confirmedStateSnapshot())
        XCTAssertEqual(Set(snapshot.keys), Set(model.cbv2RecurrentStateSpec.modelLayerIndices))
        XCTAssertTrue(
            snapshot.values.allSatisfy { $0.ssm == nil && $0.conv?.shape == [1, 2, 256] })

        // Reset: a fresh state is zero history and must change the output.
        try clone.recurrent.release()
        clone.recurrent = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        for row in (donor.rows + clone.rows).compactMap({ $0 }) {
            XCTAssertTrue(row.supportsSpeculativeWrites)
            row.beginSpeculativeWrite()
        }
        let donorNext = try LFM2MoECBv2Driver.forward(
            model, storage: storage, requests: [donor], tokens: [[7]], commit: false)
        let resetNext = try LFM2MoECBv2Driver.forward(
            model, storage: storage, requests: [clone], tokens: [[7]], commit: false)
        XCTAssertNotEqual(
            resetNext.logits.asData(access: .copy).data,
            donorNext.logits.asData(access: .copy).data,
            "conv history must influence the next token")
        for (request, tx) in [
            (donor, donorNext.transactions[0]), (clone, resetNext.transactions[0]),
        ] {
            try tx.rollback()
            for row in request.rows.compactMap({ $0 }) {
                row.rollback(1)
                row.commitSpeculativeWrite()
            }
        }
        storage.bank.invalidateBoundComposition()
        // Rollback restored the donor's exact confirmed snapshot.
        let afterRollback = try XCTUnwrap(donor.recurrent.confirmedStateSnapshot())
        for (index, state) in snapshot {
            XCTAssertEqual(
                afterRollback[index]?.conv?.asData(access: .copy).data,
                state.conv?.asData(access: .copy).data)
            XCTAssertNil(donor.recurrent.state(modelLayerIndex: index)?.ssm)
        }

        // Clone: adopting the donor snapshot reproduces the donor bit-exactly.
        try clone.recurrent.release()
        clone.recurrent = try CBv2RecurrentRequestState(
            spec: model.cbv2RecurrentStateSpec, adoptedCommitted: snapshot)
        let a = try LFM2MoECBv2Driver.forward(
            model, storage: storage, requests: [donor], tokens: [[7]])
        let b = try LFM2MoECBv2Driver.forward(
            model, storage: storage, requests: [clone], tokens: [[7]])
        XCTAssertEqual(a.logits.asData(access: .copy).data, b.logits.asData(access: .copy).data)
        for index in snapshot.keys {
            XCTAssertEqual(
                donor.recurrent.state(modelLayerIndex: index)?.conv?.asData(access: .copy).data,
                clone.recurrent.state(modelLayerIndex: index)?.conv?.asData(access: .copy).data)
        }
    }

    /// The paged native backend and the hybrid checkpoint paths need the
    /// model's KV dtypes and conv-only spec to agree with what it stages.
    func testStagedStateMatchesDeclaredSpecExactly() throws {
        for precision in [LFM2MoEFixture.Precision.float32, .bfloat16, .quantizedMixed] {
            let model = try LFM2MoEFixture.model(precision)
            let spec = model.cbv2RecurrentStateSpec
            let storage = try LFM2MoECBv2Storage(model: model, paged: true, maximumLength: 64)
            let request = try LFM2MoECBv2Request(
                model: model, storage: storage, promptLength: 12, maximumLength: 64)
            defer {
                storage.bank.releaseBoundRows()
                request.close()
            }
            _ = try LFM2MoECBv2Driver.prefill(
                model, storage: storage, request: request, prompt: LFM2MoEFixture.prompt(12),
                chunk: 12)
            let snapshot = try XCTUnwrap(request.recurrent.confirmedStateSnapshot())
            for layer in spec.layers {
                let state = try XCTUnwrap(snapshot[layer.modelLayerIndex])
                XCTAssertEqual(state.conv?.shape, layer.convShape)
                XCTAssertEqual(state.conv?.dtype, layer.convDType, "\(precision)")
                XCTAssertTrue(layer.admitsSSM(state.ssm))
            }
            // Adoption accepts the exact snapshot and refuses an SSM tensor
            // on a convolution-only layer.
            XCTAssertNoThrow(try CBv2RecurrentRequestState(spec: spec, adoptedCommitted: snapshot))
            var poisoned = snapshot
            let first = spec.layers[0].modelLayerIndex
            poisoned[first] = .init(conv: snapshot[first]!.conv, ssm: MLXArray.zeros([1, 1]))
            XCTAssertThrowsError(
                try CBv2RecurrentRequestState(spec: spec, adoptedCommitted: poisoned))
        }
    }
}

/// The published artifact's config.json (LiquidAI/LFM2.5-24B-A2B-MLX-4bit
/// @ 667408d), minus quantization metadata, for geometry assertions.
enum LFM2MoEArtifactConfig {
    static var json: String {
        let types = (0 ..< 40).map { $0 % 4 == 2 ? "\"full_attention\"" : "\"conv\"" }
            .joined(separator: ",")
        return """
            {"model_type":"lfm2_moe","vocab_size":128000,"hidden_size":2048,
             "intermediate_size":11776,"moe_intermediate_size":1536,"num_hidden_layers":40,
             "num_experts":64,"num_experts_per_tok":4,"norm_topk_prob":true,
             "num_attention_heads":32,"num_key_value_heads":8,"max_position_embeddings":131072,
             "use_expert_bias":true,"num_dense_layers":2,"norm_eps":1e-05,"conv_bias":false,
             "conv_L_cache":3,"rope_parameters":{"rope_theta":10000000.0,"rope_type":"default"},
             "routed_scaling_factor":1.0,"tie_word_embeddings":true,"layer_types":[\(types)]}
            """
    }
}
