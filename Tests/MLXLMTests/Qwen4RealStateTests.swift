import Foundation
import MLX
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

/// Opt-in full-weight regression, never part of the default GPU suite.
/// Run alone with DARKBLOOM_QWEN4_REAL_STATE_TEST=1, an explicit owned
/// DARKBLOOM_QWEN4_REAL_MODEL, cache+memory cache=0, and MAX_DRAFT=5.
/// This exercises native state boundaries; it is not signed SSD persistence,
/// BF16 source authentication, or an exhaustive natural rejection distribution.
final class Qwen4RealStateTests: XCTestCase {
    func testOwnedArtifactStateRollbackReload() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DARKBLOOM_QWEN4_REAL_STATE_TEST"] == "1",
            "Requires the owned full artifact and exclusive isolated GPU process")
        Qwen4RealStateFixture.log("phase=load begin")
        let fixture = try Qwen4RealStateFixture()
        defer { fixture.close() }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("qwen4-real-state-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }

        // Beyond the actual 2048-token QSA budget, crossing pooled frontiers.
        let prompt = (0..<2051).map { 100 + ($0 * 17 + 31) % 997 }
        let prefix = try Qwen4RealPagedState(fixture)
        defer { try? prefix.close() }
        var hidden: [MLXArray] = []
        var lastLogits: MLXArray?
        for start in stride(from: 0, to: prompt.count, by: fixture.chunk) {
            let end = min(start + fixture.chunk, prompt.count)
            let output = try prefix.forward(Array(prompt[start..<end]))
            hidden.append(output.hidden)
            lastLogits = output.logits
            Qwen4RealStateFixture.log("phase=prefill through=\(end) hidden=\(output.hidden.shape)/\(output.hidden.dtype)")
        }
        let residual = concatenated(hidden, axis: 1)
        eval(residual)
        let seed = argMax(try XCTUnwrap(lastLogits)[0..., -1, 0...], axis: -1).item(Int.self)
        let prefixArrays = try prefix.snapshot()
        XCTAssertTrue(prefixArrays.keys.contains { $0.hasPrefix("qsa.") && $0.hasSuffix(".pooled") })
        XCTAssertTrue(prefixArrays.keys.contains { $0.hasPrefix("recurrent.") })
        let prefixFile = root.appendingPathComponent("prefix.safetensors")
        try save(arrays: prefixArrays, url: prefixFile)
        let serializedPrefix = try loadArrays(url: prefixFile)
        compare(serializedPrefix, prefixArrays, label: "initial serialized prefix")
        try prefix.close()

        let drafts = try assistantRoundTrip(fixture, prompt: prompt, residual: residual, seed: seed, root: root)
        let inputs = [seed] + drafts
        XCTAssertEqual(inputs.count, 6)
        for width in 2...6 {
            for kept in 0...width {
                let label = "width=\(width) retained=\(kept)"
                Qwen4RealStateFixture.log("phase=target-rollback \(label) begin")
                let serial = try Qwen4RealPagedState(fixture, restoring: serializedPrefix)
                let verified = try Qwen4RealPagedState(fixture, restoring: serializedPrefix)
                defer { try? serial.close(); try? verified.close() }
                var serialLogits: [MLXArray] = []
                for token in inputs.prefix(kept) { serialLogits.append(try serial.forward([token]).logits) }
                let captured = try verified.forward(Array(inputs.prefix(width)), captured: true, keep: kept)
                for column in 0..<kept {
                    Qwen4RealStateFixture.equal(captured.logits[0..., column..<column + 1, 0...],
                        serialLogits[column], "\(label) logits column \(column)")
                }
                let expected = try serial.snapshot(), actual = try verified.snapshot()
                compare(actual, expected, label: label)
                // Reuse only this test-owned temporary destination so storage
                // stays bounded to one retained state, not the entire matrix.
                let file = root.appendingPathComponent("retained.safetensors")
                try save(arrays: actual, url: file)
                let payload = try loadArrays(url: file)
                compare(payload, expected, label: "\(label) serialized")
                try verified.close()
                let restored = try Qwen4RealPagedState(fixture, restoring: payload)
                defer { try? restored.close() }
                let suffix = 117
                let nextExpected = try serial.forward([suffix]), nextRestored = try restored.forward([suffix])
                Qwen4RealStateFixture.equal(nextRestored.logits, nextExpected.logits, "\(label) resumed logits")
                Qwen4RealStateFixture.equal(nextRestored.hidden, nextExpected.hidden, "\(label) resumed HC residual")
                compare(try restored.snapshot(), try serial.snapshot(), label: "\(label) resumed state")
                Qwen4RealStateFixture.log("\(label) target KV/QSA/GDN/PLE and serialized suffix assertions evaluated; see XCTest result")
            }
        }
        Qwen4RealStateFixture.log("state-only assertions evaluated; final output budgets run in testOwnedArtifactFinalOutputBudgets")
    }

    func testOwnedArtifactFinalOutputBudgets() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DARKBLOOM_QWEN4_REAL_STATE_TEST"] == "1",
            "Requires the owned full artifact and exclusive isolated GPU process")
        Qwen4RealStateFixture.log("phase=budget-only-load begin")
        let fixture = try Qwen4RealStateFixture()
        defer { fixture.close() }
        let prompt = (0..<37).map { 100 + ($0 * 17 + 31) % 997 }
        try await finalBudgets(fixture, prompt: prompt)
    }

    private func assistantRoundTrip(_ fixture: Qwen4RealStateFixture, prompt: [Int], residual: MLXArray,
        seed: Int, root: URL) throws -> [Int] {
        let assistant = fixture.assistant
        let donor = assistant.makeRequestState()
        defer { assistant.releaseRequestState(donor) }
        assistant.observeCommittedTarget(.init(tokens: MLXArray(prompt.map(Int32.init), [1, prompt.count]),
            hidden: residual), requestState: donor)
        Qwen4RealStateFixture.log("phase=assistant-capture begin residual=\(residual.shape)/\(residual.dtype)")
        let checkpoint = try XCTUnwrap(assistant.capturePrefixCheckpoint(requestState: donor, targetInputCount: prompt.count))
        let tensors = try XCTUnwrap(assistant.encodePrefixCheckpoint(checkpoint))
        eval(tensors)
        let file = root.appendingPathComponent("assistant-prefix.safetensors")
        try save(arrays: Dictionary(uniqueKeysWithValues: tensors.enumerated().map { ("tensor.\($0.offset)", $0.element) }), url: file)
        let loaded = try loadArrays(url: file)
        let imported = try tensors.indices.map { try XCTUnwrap(loaded["tensor.\($0)"]) }
        // SafeTensor reload is lazy. Production checkpoint staging evaluates
        // its owned buffers before import, and the codec deliberately refuses
        // unevaluated token storage instead of evaluating unreserved inputs.
        eval(imported)
        for (index, pair) in zip(imported, tensors).enumerated() {
            XCTAssertNotNil(try pair.0.evaluatedBufferInfo(), "assistant imported tensor \(index) must be materialized")
            Qwen4RealStateFixture.equal(pair.0, pair.1, "assistant serialized tensor \(index)")
        }
        let decoded = try XCTUnwrap(assistant.decodePrefixCheckpoint(
            tensors: imported, prefixTokens: prompt))
        Qwen4RealStateFixture.log("phase=assistant-import exact evaluated tensors decoded")
        let reference = try XCTUnwrap(assistant.restorePrefixCheckpoint(checkpoint))
        let restored = try XCTUnwrap(assistant.restorePrefixCheckpoint(decoded))
        defer { assistant.releaseRequestState(reference); assistant.releaseRequestState(restored) }
        var tokens = MLXArray([Int32(seed)], [1, 1])
        var carry = residual[0..., (prompt.count - 1)..<prompt.count, 0...]
        let before = try assistant.snapshotRequestState(reference)
        guard before.cacheLayers.allSatisfy({
            $0.offset == 0 && $0.qwen4Indexer == nil && $0.keys.size == 0 && $0.values.size == 0
        }) else {
            XCTFail("independent discard oracle requires genuinely unallocated empty head history")
            throw Qwen4RealStateFixture.Failure.invalidPrerequisite
        }
        let nativeDTypes = try headNativeDTypes(assistant, tokens: tokens, carry: carry)
        guard nativeDTypes.count == before.cacheLayers.count else {
            XCTFail("native head dtype probe and pre-round cache layer counts differ")
            throw Qwen4RealStateFixture.Failure.invalidTensorStorage
        }
        // Discard restores pre-round head caches and retains the canonical
        // seed/carry as trusted backlog. Freeze this independent expected
        // transition before either real head consumes a speculative token.
        func frozen(_ tensor: MLXArray) throws -> MLXArray {
            Qwen4RealStateFixture.log("phase=freeze-trusted shape=\(tensor.shape) dtype=\(tensor.dtype) size=\(tensor.size)")
            // Empty tensor metadata is already immutable and has no payload to
            // copy; no synthetic values or arithmetic replacement are needed.
            if tensor.size == 0 { return tensor }
            return MLXArray(try Qwen4RealStateFixture.bytes(tensor, label: "freeze trusted state"),
                tensor.shape, dtype: tensor.dtype)
        }
        let expectedDiscard = try Qwen4ExpMTPStateSnapshot(
            cacheLayers: before.cacheLayers.enumerated().map { index, layer in
                // FullSequenceKV.snapshot emits fp16 placeholders before its
                // first allocation. ensureCapacity chooses the actual native
                // projection dtype; rollback retains the allocated dtype at
                // offset zero. Only these proven empty expectations change.
                .init(layerIndex: layer.layerIndex,
                    keys: MLXArray.zeros(layer.keys.shape, dtype: nativeDTypes[index].0),
                    values: MLXArray.zeros(layer.values.shape, dtype: nativeDTypes[index].1), offset: 0)
            },
            backlogHidden: before.backlogHidden.map(frozen) + [frozen(carry)],
            backlogTokens: before.backlogTokens.map(frozen) + [frozen(tokens)],
            targetHiddenFrontier: nil, committedInputCount: before.committedInputCount + 1,
            logicalInputBase: before.logicalInputBase)
        eval(expectedDiscard.arrays)
        var proposals: [Int] = []
        for step in 0..<5 {
            Qwen4RealStateFixture.log("phase=assistant-draft step=\(step) begin")
            let a = assistant.draftStep(tokens: tokens, hidden: carry, shortlist: nil, requestState: reference)
            let b = assistant.draftStep(tokens: tokens, hidden: carry, shortlist: nil, requestState: restored)
            eval([a.tokens, a.hidden, b.tokens, b.hidden] + assistant.evaluationTargets(for: reference) + assistant.evaluationTargets(for: restored))
            Qwen4RealStateFixture.equal(a.tokens, b.tokens, "assistant restored draft \(step)")
            Qwen4RealStateFixture.equal(a.hidden, b.hidden, "assistant restored hidden \(step)")
            proposals.append(a.tokens.item(Int.self))
            tokens = a.tokens.reshaped([1, 1]); carry = a.hidden
        }
        assistant.discardRound(requestState: reference)
        assistant.discardRound(requestState: restored)
        let a = try assistant.snapshotRequestState(reference), b = try assistant.snapshotRequestState(restored)
        XCTAssertEqual(a.committedInputCount, b.committedInputCount)
        XCTAssertEqual(a.logicalInputBase, b.logicalInputBase)
        XCTAssertEqual(a.arrays.count, b.arrays.count)
        for (i, pair) in zip(a.arrays, b.arrays).enumerated() {
            Qwen4RealStateFixture.equal(pair.0, pair.1, "assistant discard state \(i)")
        }
        XCTAssertEqual(a.committedInputCount, expectedDiscard.committedInputCount)
        XCTAssertEqual(a.logicalInputBase, expectedDiscard.logicalInputBase)
        XCTAssertEqual(a.cacheLayers.map(\.offset), expectedDiscard.cacheLayers.map(\.offset))
        XCTAssertEqual(a.backlogHidden.count, expectedDiscard.backlogHidden.count)
        XCTAssertEqual(a.backlogTokens.count, expectedDiscard.backlogTokens.count)
        XCTAssertNil(a.targetHiddenFrontier)
        XCTAssertEqual(a.arrays.count, expectedDiscard.arrays.count)
        for (i, pair) in zip(a.arrays, expectedDiscard.arrays).enumerated() {
            Qwen4RealStateFixture.equal(pair.0, pair.1, "discard restores pre-round state plus canonical seed, tensor \(i)")
        }
        Qwen4RealStateFixture.log("actual embedded head draft/replay and independent pre-round+canonical-seed discard assertions evaluated; see XCTest result")
        return proposals
    }

    private func headNativeDTypes(_ assistant: Qwen4ExpInlineMTPAssistant,
        tokens: MLXArray, carry: MLXArray) throws -> [(DType, DType)] {
        let caches = assistant.makeCache()
        defer { for cache in caches { (cache as? CBv2LayerCache)?.setRows([]) } }
        let output = assistant.forward(hidden: carry, tokens: tokens, cache: caches)
        eval([output.logits, output.hidden] + caches.flatMap { $0.innerState() })
        return try caches.enumerated().map { index, cache in
            let layer = try XCTUnwrap(cache as? CBv2LayerCache)
            let row = try XCTUnwrap(layer.rows.first)
            let snapshot = row.snapshot()
            guard layer.rows.count == 1, snapshot.offset == 1,
                snapshot.keys.size > 0, snapshot.values.size > 0 else {
                throw Qwen4RealStateFixture.Failure.invalidTensorStorage
            }
            Qwen4RealStateFixture.log("independent native head dtype layer=\(index) keys=\(snapshot.keys.dtype) values=\(snapshot.values.dtype); empty-placeholder expectation only")
            return (snapshot.keys.dtype, snapshot.values.dtype)
        }
    }

    private func finalBudgets(_ fixture: Qwen4RealStateFixture, prompt: [Int]) async throws {
        for budget in 1...Qwen4RealStateFixture.maximumOutputBudget {
            Qwen4RealStateFixture.log("phase=engine-budget budget=\(budget) begin")
            Qwen4RealStateFixture.log("phase=engine-budget budget=\(budget) arm=serial construct")
            let (serial, serialBackend) = try await fixture.engine(mtp: false)
            let expected: CBv2SchedCollected
            do {
                Qwen4RealStateFixture.log("phase=engine-budget budget=\(budget) arm=serial submit")
                expected = await cbv2SchedCollect(try serial.submit(.init(id: .init(UInt64(900 + budget)),
                    promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: budget)), timeoutSeconds: 120)
            } catch { await serial.shutdown(); throw error }
            await serial.shutdown()
            XCTAssertEqual(serialBackend.bytesReserved, 0)
            Qwen4RealStateFixture.log("phase=engine-budget budget=\(budget) arm=mtp construct")
            let (mtp, mtpBackend) = try await fixture.engine(mtp: true)
            let actual: CBv2SchedCollected
            do {
                Qwen4RealStateFixture.log("phase=engine-budget budget=\(budget) arm=mtp submit")
                actual = await cbv2SchedCollect(try mtp.submit(.init(id: .init(UInt64(900 + budget)),
                    promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: budget)), timeoutSeconds: 120)
            } catch { await mtp.shutdown(); throw error }
            let observedMetrics = mtp.mtpMetricsSnapshot()
            await mtp.shutdown()
            let metrics = try XCTUnwrap(observedMetrics)
            XCTAssertEqual(mtpBackend.bytesReserved, 0)
            XCTAssertEqual(actual.tokens, expected.tokens, "actual engine final budget \(budget)")
            XCTAssertEqual(actual.tokens.count, budget)
            XCTAssertEqual(actual.finishReason, .length)
            XCTAssertEqual(expected.finishReason, .length)
            if budget == 1 { XCTAssertEqual(metrics.draftedTokens, 0) }
            if budget == Qwen4RealStateFixture.maximumOutputBudget {
                // Include enough output slots to exercise all five proposed
                // tokens in the actual engine, not just the manual state seam.
                XCTAssertGreaterThanOrEqual(metrics.draftedTokens, 5)
                XCTAssertGreaterThan(metrics.rectangularVerificationRounds, 0)
            }
            Qwen4RealStateFixture.log("engine budget=\(budget) tokens=\(actual.tokens.count) rounds=\(metrics.rounds) proposed=\(metrics.draftedTokens) accepted=\(metrics.acceptedTokens)")
        }
        Qwen4RealStateFixture.log("limits: forced retained-prefix target state, actual-head snapshots and natural engine output budgets; no claim every natural acceptance count occurred, no encrypted SSD restart or external BF16 comparison")
    }

    private func compare(_ actual: [String: MLXArray], _ expected: [String: MLXArray], label: String) {
        XCTAssertEqual(Set(actual.keys), Set(expected.keys), label)
        for name in expected.keys.sorted() {
            guard let a = actual[name], let b = expected[name] else { continue }
            Qwen4RealStateFixture.equal(a, b, "\(label) \(name)")
        }
    }
}
