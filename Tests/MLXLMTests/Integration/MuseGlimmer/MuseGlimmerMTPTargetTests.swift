// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The target side of block drafting: captured residuals, the last-row prompt
/// readout, raw embeddings, the head, and the adapter a request-stateful
/// drafter binds to. Tiny models only.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerMTPTargetTests {
    private let tokens = MLXArray(MuseGlimmerFixture.tokens(19, vocabulary: 64)).reshaped(1, -1)

    /// A tower made of the first `count` layers of `full`, with its other weights.
    private func truncated(_ full: MuseGlimmerTextModel, layers count: Int) throws
        -> MuseGlimmerTextModel
    {
        var text = MuseGlimmerFixture.eagerText
        text["num_hidden_layers"] = count
        text["layer_types"] = Array(full.configuration.layerTypes.prefix(count))
        text["layer_rope_theta"] = Array(full.configuration.layerRopeTheta.prefix(count))
        let model = MuseGlimmerTextModel(try MuseGlimmerFixture.configuration(text))
        let keep = Set(model.parameters().flattened().map(\.0))
        let weights = full.parameters().flattened().filter { keep.contains($0.0) }
        try model.update(parameters: .unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    @Test(arguments: [DType.bfloat16, DType.float32])
    func capturedHiddenIsTheResidualAfterEachNamedLayer(dtype: DType) throws {
        let full = try MuseGlimmerFixture.model(MuseGlimmerFixture.eagerText, dtype: dtype)
        #expect(full.mtpHiddenLayerIDs == [1])  // the published ids that exist in 6 layers
        full.mtpHiddenLayerIDs = [1, 3, 5]
        #expect(full.mtpHiddenWidth == 96)
        let plain = full(tokens, cache: nil)
        let (logits, hidden) = full.forwardWithMTPHidden(tokens, cache: nil)
        #expect(MuseGlimmerFixture.sameBits(logits, plain))
        #expect(hidden.shape == [1, 19, 96] && hidden.dtype == dtype)
        // Layer i's captured output is exactly what a tower ending at layer i
        // hands to its final norm and head.
        for (slot, layer) in [1, 3, 5].enumerated() {
            let prefix = try truncated(full, layers: layer + 1)
            let residual = hidden[.ellipsis, (slot * 32) ..< ((slot + 1) * 32)]
            let readout = prefix.outputLogits(prefix.model.norm(residual))
            #expect(MuseGlimmerFixture.sameBits(readout, prefix(tokens, cache: nil)))
        }
        // The same capture on incremental caches, one token at a time.
        let cache = try full.newCache(parameters: nil)
        var rows = [MLXArray]()
        for index in 0 ..< 19 {
            let step = full.forwardWithMTPHidden(tokens[0..., index ..< index + 1], cache: cache)
            eval([step.hidden] + cache.flatMap { $0.innerState() })
            rows.append(step.hidden)
        }
        let stepped = concatenated(rows, axis: 1).asType(.float32)
        #expect(abs(stepped - hidden.asType(.float32)).max().item(Float.self) < 0.05)
    }

    @Test func prefillReadoutKeepsEveryHiddenRowAndProjectsTheLastOnly() throws {
        let m = try MuseGlimmerFixture.model(MuseGlimmerFixture.eagerText, dtype: .bfloat16)
        m.mtpHiddenLayerIDs = [1, 4]
        let reference = m.forwardWithMTPHidden(tokens, cache: nil)
        let intermediate = m.prefillWithMTPHidden(tokens, cache: nil, requirement: .evaluationOnly)
        #expect(intermediate.logits.shape == [1, 1])
        #expect(MuseGlimmerFixture.sameBits(intermediate.hidden, reference.hidden))
        let frontier = m.prefillWithMTPHidden(
            tokens, cache: nil, requirement: .lastPositionLogits)
        #expect(frontier.logits.shape == [1, 64])
        #expect(MuseGlimmerFixture.sameBits(frontier.hidden, reference.hidden))
        let last = reference.logits[0..., -1, 0...]
        #expect(
            abs(frontier.logits.asType(.float32) - last.asType(.float32)).max().item(Float.self)
                <= 0.05)
        #expect(
            argMax(frontier.logits, axis: -1).item(Int.self)
                == argMax(last, axis: -1).item(Int.self))
    }

    @Test func rawEmbeddingsAndTheHeadAreTheTargetsOwn() throws {
        let m = try MuseGlimmerFixture.model(MuseGlimmerFixture.eagerText, dtype: .bfloat16)
        let ids = MLXArray([Int32(3), 17, 63]).reshaped(1, 3)
        let raw = m.rawTokenEmbeddings(ids)
        #expect(raw.shape == [1, 3, 32] && m.tokenEmbeddingTable.shape == [64, 32])
        #expect(MuseGlimmerFixture.sameBits(raw, m.tokenEmbeddingTable[ids]))
        let scaled = m.scaledTokenEmbeddings(ids)
        #expect(
            MuseGlimmerFixture.sameBits(
                scaled, MLXFast.rmsNorm(raw, weight: MLXArray.mlxNone, eps: 1e-5)))
        #expect(!MuseGlimmerFixture.sameBits(scaled, raw))
        let hidden = MuseGlimmerFixture.hostTensor([2, 5, 32], seed: 9).asType(.bfloat16)
        let product = m.headProduct(hidden)
        let multiplier = m.configuration.outputMultiplier
        let capped = tanh(product * multiplier / Float(20)) * Float(20)
        #expect(MuseGlimmerFixture.sameBits(m.outputLogits(hidden), capped))
        #expect(
            arrayEqual(argMax(product, axis: -1), argMax(m.outputLogits(hidden), axis: -1))
                .item(Bool.self))
    }

    // MARK: Adapter

    private final class State: CBv2MTPRequestState {
        var committedInputCount: Int { 0 }
        var stagedInputCount: Int { 0 }
    }
    private final class Capture: CBv2MTPPreparedCapture {}
    /// Metadata-only request-stateful drafter: driver admission never runs it.
    private final class Drafter: CBv2MTPRequestStatefulDrafter {
        let identity: ObjectIdentifier
        init(_ target: MuseGlimmerTextModel) { identity = ObjectIdentifier(target) }
        var mtpTargetIdentity: ObjectIdentifier? { identity }
        var requiredVerificationMode: CBv2MTPVerificationMode? { .serialTarget }
        var maximumDraftTokens: Int? { 7 }
        var maximumSpeculativeBatch: Int? { 1 }
        func prepare(rows: [CBv2MTPRowCapture]) -> any CBv2MTPPreparedCapture { Capture() }
        func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: any CBv2MTPPreparedCapture)
            -> (tokens: MLXArray, hidden: MLXArray)
        { preconditionFailure("admission must not draft") }
        func makeRequestState() -> any CBv2MTPRequestState { State() }
        func observeCommittedTarget(
            _ observation: CBv2MTPCommittedTargetObservation,
            requestState: any CBv2MTPRequestState
        ) {}
        func draftStep(
            tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
            requestState: any CBv2MTPRequestState
        ) -> (tokens: MLXArray, hidden: MLXArray) {
            preconditionFailure("admission must not draft")
        }
        func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { [] }
        func finalizeRound(
            requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
            committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
        ) {}
        func discardRound(requestState: any CBv2MTPRequestState) {}
        func releaseRequestState(_ requestState: any CBv2MTPRequestState) {}
    }

    @Test func adapterAdmitsARequestStatefulDrafterWhereTheGenericAdapterCannot() throws {
        let m = try MuseGlimmerFixture.model(MuseGlimmerFixture.eagerText, dtype: .bfloat16)
        let adapter = MuseGlimmerCBv2Adapter(m)
        #expect(adapter.supportsRequestStatefulMTP && adapter.mtpCaptureLayers == nil)
        #expect(adapter.mtpTargetIdentity == ObjectIdentifier(m))
        #expect(!((adapter as Any) is any CBv2RecurrentMTPSteppableModel))
        #expect(adapter.cbv2Capabilities == m.cbv2Capabilities)
        #expect(adapter.cbv2CompleteCheckpointKVDTypes == m.cbv2CompleteCheckpointKVDTypes)
        #expect(adapter.cbv2SupportsHistoricalAttentionCheckpoint)
        #expect(adapter.decodeCommandBufferLimits == MuseGlimmerDecodeCommandBuffers.limits)
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 7, maxSpeculativeBatch: 4, fixedDraftTokens: 7,
            verificationMode: .automatic)
        let driver = try #require(
            CBv2MTPRoundDriver.build(model: adapter, drafter: Drafter(m), config: config))
        #expect(driver.usesRequestStatefulDrafter && driver.config.maxDraftTokens == 7)
        // Another target instance, or the generic adapter, stays inactive.
        let other = try MuseGlimmerFixture.model(MuseGlimmerFixture.eagerText, dtype: .bfloat16)
        #expect(
            CBv2MTPRoundDriver.build(model: adapter, drafter: Drafter(other), config: config) == nil
        )
        #expect(
            CBv2MTPRoundDriver.build(
                model: CBv2SteppableLanguageModelAdapter(m), drafter: Drafter(m), config: config)
                == nil)
        // No drafter hidden configured: the target cannot serve a stateful drafter.
        m.mtpHiddenLayerIDs = []
        #expect(!adapter.supportsRequestStatefulMTP)
        #expect(
            CBv2MTPRoundDriver.build(model: adapter, drafter: Drafter(m), config: config) == nil)
    }

    /// The adapter's forwards on native paged caches: the hidden-returning
    /// forward and the plain one write the same K/V and return the same logits.
    @Test func adapterForwardsAgreeOnPagedCaches() throws {
        let m = try MuseGlimmerFixture.model(
            MuseGlimmerFixture.pagedText(window: 64), dtype: .bfloat16)
        m.mtpHiddenLayerIDs = [1, 4]
        let adapter = MuseGlimmerCBv2Adapter(m)
        let prompt = MuseGlimmerFixture.tokens(150, vocabulary: 96)
        func run(hidden: Bool) throws -> ([Data], [Data], [Int]) {
            let backend = try PagedKVBackend(
                layerKinds: m.cbv2LayerKinds,
                config: .init(
                    capacityBytes: 64 << 20, dtype: .bfloat16, maxPrefillChunk: 64,
                    nominalMaxSequenceLength: 512, segmentSizeBytes: 1 << 20,
                    layerDTypes: m.kvStorageDTypes))
            let caches = backend.makeLayerCaches()
            let state = try backend.makeSequenceState(
                layerKinds: m.cbv2LayerKinds, promptLength: prompt.count,
                maxLength: prompt.count + 16)
            defer {
                eval(caches.flatMap { $0.innerState() })
                for cache in caches { cache.setRows([]) }
                backend.release(state)
            }
            for (index, cache) in caches.enumerated() { cache.setRows([state[index]!]) }
            var outputs = [Data]()
            var widths = [Int]()
            var next = Int32(0)
            for start in stride(from: 0, to: prompt.count, by: 64) {
                let end = min(start + 64, prompt.count)
                let chunk = MLXArray(Array(prompt[start ..< end])).reshaped(1, -1)
                let requirement: CBv2PrefillRequirement =
                    end == prompt.count ? .lastPositionLogits : .evaluationOnly
                let logits: MLXArray
                if hidden {
                    let result = adapter.forwardWithHiddenForPrefill(
                        tokens: chunk, caches: caches, requirement: requirement)
                    widths.append(result.lastHidden.dim(-1))
                    #expect(result.lastHidden.dim(1) == end - start)
                    logits = result.logits
                    eval(result.lastHidden)
                } else {
                    logits = adapter.prefill(
                        tokens: chunk, inputEmbeddings: nil, caches: caches,
                        requirement: requirement)
                }
                eval([logits] + caches.flatMap { $0.innerState() })
                if end == prompt.count {
                    outputs.append(logits.asData(access: .copy).data)
                    next = argMax(logits[0], axis: -1).item(Int32.self)
                }
            }
            for _ in 0 ..< 10 {
                let token = MLXArray([next]).reshaped(1, 1)
                let logits: MLXArray
                if hidden {
                    let result = adapter.forwardWithHidden(tokens: token, caches: caches)
                    #expect(result.lastHidden.shape == [1, 1, 128])
                    logits = result.logits
                } else {
                    logits = adapter.forward(tokens: token, caches: caches)
                }
                eval([logits] + caches.flatMap { $0.innerState() })
                outputs.append(logits.asData(access: .copy).data)
                next = argMax(logits[0, -1], axis: -1).item(Int32.self)
            }
            var kv = [Data]()
            for row in state {
                let snapshot = try #require(row).snapshot()
                eval(snapshot.keys, snapshot.values)
                kv.append(snapshot.keys.asData(access: .copy).data)
                kv.append(snapshot.values.asData(access: .copy).data)
            }
            return (outputs, kv, widths)
        }
        let plain = try run(hidden: false)
        let captured = try run(hidden: true)
        #expect(captured.0 == plain.0)
        #expect(captured.1 == plain.1)
        #expect(captured.2 == [128, 128, 128])
    }
}
