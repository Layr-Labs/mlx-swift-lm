import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Historical paged window checkpoints", .serialized)
struct HistoricalWindowCheckpointTests {
    // Match production import alignment while keeping several wraps of window 17.
    private var chunkSize: Int { max(32, CBv2AttentionV1.queryBlockSize) }
    private var maximumLength: Int { 4 * chunkSize }

    private struct Fixture {
        let backend: PagedKVBackend
        let admission: AdmissionV2
        let codec: CBv2CompleteCheckpointCodec
        let request: CBv2Request
        let kinds: [CBv2LayerKind]
    }

    private func fixture(dtype: DType = .float16) throws -> Fixture {
        let kinds = [
            CBv2LayerKind(attention: .slidingWindow(17), headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .full, hasSinks: true, headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .slidingWindow(17), sharesKVWithLayer: 0,
                          headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .full, sharesKVWithLayer: 1, hasSinks: true,
                          headDim: 64, kvHeads: 1, queryHeads: 2),
        ]
        let config = PagedKVPoolConfig(capacityBytes: 256 << 20, maxPrefillChunk: chunkSize,
            segmentSizeBytes: 64 << 10, layerDTypes: Array(repeating: dtype, count: kinds.count))
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 256 << 20,
            config: .init(watermarkFraction: 0, elementBytes: dtype.size,
                          layerElementBytes: Array(repeating: dtype.size, count: kinds.count)),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        let identity = CBv2CompleteCheckpointIdentity(modelAggregateHash: "tiny-window-borrower",
            promptContractID: "causal-text", buildID: "history-test", numericsFingerprint: "native-\(dtype)")
        let codec = CBv2CompleteCheckpointCodec(identity: identity, layerKinds: kinds,
            recurrentSpec: nil, kvDTypes: Array(repeating: dtype, count: kinds.count),
            assistant: nil, admission: admission, pagedConfig: config)
        let request = CBv2Request(id: .init(1), promptTokens: Array(repeating: 1, count: 3 * chunkSize + 1),
            maxTokens: chunkSize - 1, cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
        return .init(backend: backend, admission: admission, codec: codec, request: request, kinds: kinds)
    }

    private func donor(_ fixture: Fixture, maxLength: Int? = nil) throws -> [CBv2SequenceKV?] {
        let maxLength = maxLength ?? maximumLength
        try fixture.admission.reserve(id: .init(9001), additionalTokens: maxLength)
        return try fixture.backend.makeSequenceState(layerKinds: fixture.kinds,
            promptLength: fixture.request.promptTokens.count, maxLength: maxLength)
    }

    private func write(_ state: [CBv2SequenceKV?], start: Int, count: Int, salt: Int = 0) {
        for (index, entry) in state.enumerated() {
            guard let row = entry as? PagedSequenceKV else { continue }
            let values = (0 ..< count).flatMap { Array(repeating: Float(start + $0 + index * 128 + salt), count: 64) }
            let array = MLXArray(values, [1, count, 64]).asType(row.groupKey.dtype)
            row.write(keys: array, values: array + MLXArray(1).asType(row.groupKey.dtype))
        }
    }

    private func checkpoint(_ fixture: Fixture, state: [CBv2SequenceKV?], position: Int)
        throws -> CBv2HistoricalCompleteCheckpoint
    {
        let row = try #require(state[0] as? PagedSequenceKV)
        let window = try CBv2HistoricalWindow(row: row,
            position: position, admission: fixture.admission)
        return .init(position: position, chunkSize: chunkSize, windows: [0: window])
    }

    private func stageSource(_ source: CBv2CompleteCheckpointExport, fixture: Fixture)
        throws -> CBv2StagedCompleteCheckpoint
    {
        let plan = try fixture.codec.plan(manifest: source.manifest, request: fixture.request,
            minimumChunkSize: chunkSize, maximumChunkSize: chunkSize)
        let sink = try plan.allocate(onRelease: {})
        defer { sink.close() }
        for (index, descriptor) in source.manifest.tensors.enumerated() {
            var offset = 0
            while offset < descriptor.byteCount {
                let data = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: 258)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: data)
                offset += data.count
            }
        }
        return try sink.finish()
    }

    private func importSource(_ source: CBv2CompleteCheckpointExport, fixture: Fixture, id: CBv2RequestID)
        throws -> [CBv2SequenceKV?]
    {
        let staged = try stageSource(source, fixture: fixture)
        defer { staged.close() }
        return try staged.consumePreparedState { prepared in
            let frame = try #require(prepared.pagedFrame)
            prepared.pagedFrame = nil
            let adopted = try fixture.backend.pool.importCheckpoint(frame, admission: fixture.admission,
                requestID: id, layerKinds: fixture.kinds, maximumTokens: staged.maximumSequenceLength)
            return try adopted.moveToActiveRequest { #expect($0.isEmpty) }
        }
    }

    private func values(_ row: PagedSequenceKV) -> [Float] {
        row.gatherRange(start: row.absoluteOffset - row.retainedCount, count: row.retainedCount)
            .keys.asType(.float32).asArray(Float.self)
    }

    @Test("Captured ring survives terminal overwrite; two restored branches own independent pages",
          arguments: [DType.float16, .bfloat16, .float32])
    func historicalBranches(dtype: DType) throws {
        let fixture = try fixture(dtype: dtype)
        func exercise() throws {
            var original = try donor(fixture)
            defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
            write(original, start: 0, count: chunkSize)
            let first = try checkpoint(fixture, state: original, position: chunkSize)
            // The engine forbids chained successors while this candidate is
            // pending. Complete that launch boundary before later ring writes.
            try first.finishEvaluation()
            write(original, start: chunkSize, count: chunkSize)
            let second = try checkpoint(fixture, state: original, position: 2 * chunkSize)
            try second.finishEvaluation()
            write(original, start: 2 * chunkSize, count: chunkSize)
            let old = try fixture.codec.exportHistorical(checkpoint: first, state: original,
                tokens: fixture.request.promptTokens, cacheSalt: fixture.request.cacheSalt)
            let recent = try fixture.codec.exportHistorical(checkpoint: second, state: original,
                tokens: fixture.request.promptTokens, cacheSalt: fixture.request.cacheSalt)
            defer { old.close(); recent.close() }
            #expect(old.manifest.tensors.count == 4 && old.manifest.attentionLayers?.map(\.owner) == [0, 1, 0, 1])
            #expect(old.manifest.tensors[0].shape[2] == 17 && old.manifest.tensors[2].shape[2] == chunkSize)
            var left = try importSource(old, fixture: fixture, id: .init(1))
            var right = try importSource(recent, fixture: fixture, id: .init(2))
            defer {
                fixture.backend.release(left); left.removeAll(); fixture.admission.releaseAll(id: .init(1))
                fixture.backend.release(right); right.removeAll(); fixture.admission.releaseAll(id: .init(2))
            }
            #expect(left.count == 4 && left[2] == nil && left[3] == nil)
            #expect(right.count == 4 && right[2] == nil && right[3] == nil)
            let leftFull = try #require(left[1] as? PagedSequenceKV)
            let rightFull = try #require(right[1] as? PagedSequenceKV)
            #expect(leftFull.absoluteOffset == chunkSize && leftFull.retainedCount == chunkSize)
            #expect(rightFull.absoluteOffset == 2 * chunkSize && rightFull.retainedCount == 2 * chunkSize)
            let l = try #require(left[0] as? PagedSequenceKV)
            let r = try #require(right[0] as? PagedSequenceKV)
            #expect(l.absoluteOffset == chunkSize && l.baseOffset == chunkSize - 17 && l.writtenHighWater == chunkSize)
            #expect(r.absoluteOffset == 2 * chunkSize && r.baseOffset == 2 * chunkSize - 17 && r.writtenHighWater == 2 * chunkSize)
            #expect(l.decodeTableLength == l.ringPages && r.decodeTableLength == r.ringPages)
            #expect(values(l) == ((chunkSize - 17) ..< chunkSize).flatMap { Array(repeating: Float($0), count: 64) })
            let before = values(r)
            write(left, start: chunkSize, count: 16, salt: 1000)
            #expect(values(r) == before, "different request positions do not share a mutable frontier")
            #expect(Set(l.table).isDisjoint(with: Set(r.table)))
            #expect(fixture.admission.bytesReserved >= 2 * fixture.admission.allocatedBytes(forTokens: maximumLength))
            let reuse = try fixture.codec.historicalReusePlan(position: chunkSize, maximumSequenceLength: maximumLength)
            #expect(reuse.strategy == .direct && reuse.replayTokens == 0 && reuse.prefillTokensSaved == chunkSize)
            #expect(reuse.capacityTokensForChunk(start: chunkSize, count: chunkSize) == 0)
        }
        try exercise()
        #expect(fixture.admission.bytesReserved == 0 && fixture.backend.bytesWired == 0)
    }

    @Test("Capture refusal occurs before page metadata or GPU graph construction")
    func refusalAndFailure() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        write(original, start: 0, count: chunkSize)
        let row = try #require(original[0] as? PagedSequenceKV)
        let tiny = AdmissionV2(layerKinds: [], bytesCapacity: 1, config: .init(watermarkFraction: 0))
        var allocations = 0
        #expect(throws: (any Error).self) {
            try CBv2HistoricalWindow(row: row, position: chunkSize, admission: tiny, beforeAllocation: { allocations += 1 })
        }
        #expect(allocations == 0 && tiny.bytesReserved == 0)
        let before = fixture.admission.bytesReserved
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            try CBv2HistoricalWindow(row: row, position: chunkSize, admission: fixture.admission,
                beforeAllocation: { allocations += 1; throw CBv2CompleteCheckpointError.allocationFailed })
        }
        #expect(allocations == 1 && fixture.admission.bytesReserved == before)
    }

    @Test("Last K/V source owns the capture charge through closure")
    func sourceLifetime() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        write(original, start: 0, count: chunkSize)
        let row = try #require(original[0] as? PagedSequenceKV)
        let before = fixture.admission.bytesReserved
        var capture: CBv2HistoricalWindow? = try .init(row: row, position: chunkSize, admission: fixture.admission)
        let charge = fixture.admission.bytesReserved - before
        let keys = CBv2HistoricalWindowTensorSource(window: try #require(capture), values: false)
        let values = CBv2HistoricalWindowTensorSource(window: try #require(capture), values: true)
        capture = nil
        keys.close()
        #expect(charge > 0 && fixture.admission.bytesReserved == before + charge)
        _ = try values.readSegment(byteOffset: 0, maximumBytes: 128)
        values.close()
        #expect(fixture.admission.bytesReserved == before)
        #expect(throws: CBv2CompleteCheckpointError.closed) { try values.readSegment(byteOffset: 0, maximumBytes: 128) }
    }

    @Test("Malformed owner/layout/window/token metadata fails before staging")
    func incompatibleHistory() throws {
        let fixture = try fixture()
        let layout = try #require(fixture.codec.historicalLayout)
        let descriptors = try fixture.codec.tensorDescriptors(position: chunkSize)
        let base = CBv2CompleteCheckpointManifest(identity: fixture.codec.identity, position: chunkSize, chunkSize: chunkSize,
            prefixTokens: Array(fixture.request.promptTokens.prefix(chunkSize)), cacheSalt: fixture.request.cacheSalt,
            assistantCodecID: nil, tensors: descriptors,
            backendLayout: CBv2CompleteCheckpointManifest.historicalAttentionLayout, attentionLayers: layout.layers)
        _ = try fixture.codec.plan(manifest: base, request: fixture.request,
            minimumChunkSize: chunkSize, maximumChunkSize: chunkSize)
        let encoded = try JSONEncoder().encode(base)
        for mutation in ["owner", "window", "dtype", "tokens"] {
            let decoded = try JSONSerialization.jsonObject(with: encoded)
            var object = try #require(decoded as? [String: Any])
            if mutation == "tokens" { object["prefixTokens"] = Array(repeating: 2, count: chunkSize) }
            else {
                var layers = try #require(object["attentionLayers"] as? [[String: Any]])
                if mutation == "owner" { layers[2]["owner"] = 1 }
                if mutation == "window" { layers[0]["window"] = 18 }
                if mutation == "dtype" { layers[0]["dtype"] = "float32" }
                object["attentionLayers"] = layers
            }
            let corrupt = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self,
                from: JSONSerialization.data(withJSONObject: object))
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try fixture.codec.plan(manifest: corrupt, request: fixture.request, minimumChunkSize: chunkSize, maximumChunkSize: chunkSize)
            }
        }
        #expect(fixture.admission.bytesReserved == 0 && fixture.backend.bytesWired == 0)
    }

    @Test("Cancelled or failed window adoption drains every private destination before refund")
    func closeDuringConsumeAndFailedRestore() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        write(original, start: 0, count: chunkSize)
        let captured = try checkpoint(fixture, state: original, position: chunkSize)
        try captured.finishEvaluation()
        let source = try fixture.codec.exportHistorical(checkpoint: captured, state: original,
            tokens: fixture.request.promptTokens, cacheSalt: fixture.request.cacheSalt)
        defer { source.close() }
        let before = fixture.admission.bytesReserved
        let cancelled = try stageSource(source, fixture: fixture)
        #expect(fixture.admission.bytesReserved > before)
        cancelled.close()
        #expect(fixture.admission.bytesReserved == before)
        #expect(throws: CBv2CompleteCheckpointError.closed) { try cancelled.consumePreparedState { _ in } }
        let staged = try stageSource(source, fixture: fixture)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try staged.consumePreparedState { prepared in
                // Close loses ownership to consume; it cannot reclaim buffers
                // while the active transfer/restore callback still uses them.
                DispatchQueue.global().sync { staged.close() }
                #expect(fixture.admission.bytesReserved > before)
                let frame = try #require(prepared.pagedFrame)
                prepared.pagedFrame = nil
                let adopted = try fixture.backend.pool.importCheckpoint(frame, admission: fixture.admission,
                    requestID: .init(1), layerKinds: fixture.kinds, maximumTokens: staged.maximumSequenceLength)
                return try adopted.moveToActiveRequest { auxiliary in
                    // Adoption owns two physical rows; successful move expands
                    // them into four model slots with two nil borrowers.
                    #expect(auxiliary.isEmpty && adopted.rows.count == 2)
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
            }
        }
        #expect(fixture.admission.bytesReserved == before)
        let originalWindow = try #require(original[0] as? PagedSequenceKV)
        #expect(originalWindow.absoluteOffset == chunkSize)
    }

    private func commit(_ capture: CBv2CompleteCheckpointCapture, state: [CBv2SequenceKV?],
                        positions: [Int], hint: Int? = nil, resumedAt: Int = 0,
                        id: CBv2RequestID = .init(9001)) throws -> Int {
        var bytesPerCheckpoint = 0
        for position in positions {
            write(state, start: position - chunkSize, count: chunkSize)
            let prepared = try capture.prepareHistorical(position: position, chunkSize: chunkSize, state: state)
            let candidate = try #require(prepared)
            try candidate.finishEvaluation()
            bytesPerCheckpoint = candidate.stagedHistoricalBytes
            capture.commitHistorical(candidate, requestID: id, hintTokens: hint, resumedAt: resumedAt)
            #expect((capture.staged[id]?.count ?? 0) <= CBv2HistoricalCheckpointRetention.maximumRetained)
            #expect(capture.staged[id]?.compactMap(\.position) == capture.historicalRetention[id]?.retained)
        }
        return bytesPerCheckpoint
    }

    private func retentionCapture(_ fixture: Fixture) -> CBv2CompleteCheckpointCapture {
        let capture = CBv2CompleteCheckpointCapture(codec: fixture.codec, store: CompleteCheckpointFixtureStore())
        capture.historicalCheckpointStrideTokens = chunkSize
        return capture
    }

    private func dropAndWait(_ capture: CBv2CompleteCheckpointCapture, id: CBv2RequestID = .init(9001)) {
        let done = DispatchSemaphore(value: 0)
        let dropped = capture.drop(requestID: id, completion: { done.signal() })
        #expect(dropped)
        #expect(done.wait(timeout: .now() + 10) == .success)
        capture.queue.sync {}
        #expect(capture.historicalRetention[id] == nil && capture.staged[id] == nil)
    }

    @Test("Without a hint a donor stages the first and the rolling latest; retired windows refund their charge")
    func retentionKeepsFirstAndDeepest() throws {
        let fixture = try fixture()
        var original = try donor(fixture, maxLength: 8 * chunkSize)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        let capture = retentionCapture(fixture)
        #expect(capture.historicalStagedByteBudget == fixture.admission.bytesCapacity / 16)
        let before = fixture.admission.bytesReserved
        let bytes = try commit(capture, state: original, positions: (1 ... 6).map { $0 * chunkSize })
        #expect(bytes > 0)
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 6 * chunkSize])
        capture.queue.sync {}
        #expect(fixture.admission.bytesReserved == before + 2 * bytes)
        // A duplicate boundary is refused without touching the retained set.
        let duplicate = try #require(try capture.prepareHistorical(position: 6 * chunkSize, chunkSize: chunkSize,
                                                                   state: original))
        try duplicate.finishEvaluation()
        capture.commitHistorical(duplicate, requestID: .init(9001))
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 6 * chunkSize])
        dropAndWait(capture)
        #expect(fixture.admission.bytesReserved == before, "retired interior windows refund their transient charge")
        capture.close()
    }

    @Test("A fork hint stages first, target and latest; an adopter stages nothing at or below its restore point")
    func retentionKeepsForkTarget() throws {
        let fixture = try fixture()
        var original = try donor(fixture, maxLength: 8 * chunkSize)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        let capture = retentionCapture(fixture)
        let before = fixture.admission.bytesReserved
        let bytes = try commit(capture, state: original, positions: (1 ... 6).map { $0 * chunkSize },
                               hint: 3 * chunkSize + 9)
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 3 * chunkSize, 6 * chunkSize])
        #expect(capture.historicalRetention[.init(9001)]?.publication.publish
            == [6 * chunkSize, 3 * chunkSize, chunkSize])
        capture.queue.sync {}
        #expect(fixture.admission.bytesReserved == before + 3 * bytes)
        dropAndWait(capture)
        #expect(fixture.admission.bytesReserved == before)

        // The same rows seen by an adopter that restored at 3c with the same
        // hint: the first and the target are already durable below it.
        var adopted = try fixture.backend.makeSequenceState(layerKinds: fixture.kinds,
            promptLength: fixture.request.promptTokens.count, maxLength: 8 * chunkSize)
        try fixture.admission.reserve(id: .init(9002), additionalTokens: 8 * chunkSize)
        defer { fixture.backend.release(adopted); adopted.removeAll(); fixture.admission.releaseAll(id: .init(9002)) }
        write(adopted, start: 0, count: chunkSize)
        write(adopted, start: chunkSize, count: chunkSize)
        write(adopted, start: 2 * chunkSize, count: chunkSize)
        _ = try commit(capture, state: adopted, positions: (4 ... 6).map { $0 * chunkSize },
                       hint: 3 * chunkSize + 9, resumedAt: 3 * chunkSize, id: .init(9002))
        #expect(capture.staged[.init(9002)]?.compactMap(\.position) == [6 * chunkSize])
        #expect(capture.historicalRetention[.init(9002)]?.first == nil)
        dropAndWait(capture, id: .init(9002))
        capture.close()
    }

    @Test("One range copies only the boundaries retention keeps: the latest, an open first and the target")
    func prepareCopiesOnlyRetainedBoundaries() throws {
        let fixture = try fixture()
        var original = try donor(fixture, maxLength: 8 * chunkSize)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        let capture = CBv2CompleteCheckpointCapture(codec: fixture.codec, store: CompleteCheckpointFixtureStore())
        // Four boundaries per chunk; the fixture ring holds one whole chunk.
        let stride = chunkSize / 4
        capture.historicalCheckpointStrideTokens = stride
        var copied: [Int] = []
        capture.makeHistoricalWindow = { row, position, admission in
            copied.append(position)
            return try CBv2HistoricalWindow(row: row, position: position, admission: admission)
        }
        func range(_ index: Int, hint: Int?) throws -> [Int] {
            write(original, start: index * chunkSize, count: chunkSize)
            copied.removeAll()
            let positions = (1 ... 4).map { index * chunkSize + $0 * stride }
            let retention = capture.historicalRetention(requestID: .init(9001), hintTokens: hint, resumedAt: 0)
            let prepared = try capture.prepareHistorical(positions: positions, retention: retention, state: original)
            #expect(prepared.compactMap(\.position) == prepared.compactMap(\.position).sorted())
            for candidate in prepared {
                try candidate.finishEvaluation()
                capture.commitHistorical(candidate, requestID: .init(9001), hintTokens: hint)
            }
            return copied
        }
        let hint = chunkSize + 2 * stride + 3
        // First range: the deepest boundary, then the open first below it.
        let firstRange = try range(0, hint: hint)
        #expect(firstRange == [4 * stride, stride])
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [stride, 4 * stride])
        // Second range: the deepest and the fork target; two interior
        // boundaries are never copied.
        let secondRange = try range(1, hint: hint)
        #expect(secondRange == [8 * stride, 6 * stride])
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [stride, 6 * stride, 8 * stride])
        // Third range: only the rolling latest.
        let thirdRange = try range(2, hint: hint)
        #expect(thirdRange == [12 * stride])
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [stride, 6 * stride, 12 * stride])
        // A refused deepest boundary hands the latest to the next one down.
        let bounded = CBv2CompleteCheckpointCapture(codec: fixture.codec,
            store: CompleteCheckpointFixtureStore(maximumPosition: 15 * stride))
        bounded.historicalCheckpointStrideTokens = stride
        bounded.makeHistoricalWindow = capture.makeHistoricalWindow
        write(original, start: 3 * chunkSize, count: chunkSize)
        copied.removeAll()
        let retention = bounded.historicalRetention(requestID: .init(9003), hintTokens: nil, resumedAt: 2 * chunkSize)
        let prepared = try bounded.prepareHistorical(positions: (13 ... 16).map { $0 * stride },
                                                     retention: retention, state: original)
        #expect(prepared.compactMap(\.position) == [15 * stride] && copied == [15 * stride])
        prepared.forEach { $0.finishEvaluationAndClose() }
        dropAndWait(capture)
        capture.close(); bounded.close()
    }

    @Test("The byte budget gives up the fork target but never the first/latest pair")
    func retentionByteBudget() throws {
        let fixture = try fixture()
        var original = try donor(fixture, maxLength: 8 * chunkSize)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        let capture = retentionCapture(fixture)
        let before = fixture.admission.bytesReserved
        let perCheckpoint = try commit(capture, state: original, positions: [chunkSize], hint: 3 * chunkSize)
        #expect(perCheckpoint > 0)
        #expect(fixture.admission.bytesReserved == before + perCheckpoint)
        capture.historicalStagedByteBudgetOverride = perCheckpoint * 5 / 2
        _ = try commit(capture, state: original, positions: [2 * chunkSize, 3 * chunkSize], hint: 3 * chunkSize)
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 3 * chunkSize])
        // The third staged boundary exceeds 2.5 checkpoints: the target goes.
        _ = try commit(capture, state: original, positions: [4 * chunkSize, 5 * chunkSize], hint: 3 * chunkSize)
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 5 * chunkSize])
        capture.queue.sync {}
        #expect(fixture.admission.bytesReserved == before + 2 * perCheckpoint)
        capture.historicalStagedByteBudgetOverride = 0
        _ = try commit(capture, state: original, positions: [6 * chunkSize], hint: 3 * chunkSize)
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 6 * chunkSize],
                "a zero budget still keeps the older first/latest pair")
        capture.historicalStagedByteBudgetOverride = nil
        #expect(capture.historicalStagedByteBudget == fixture.admission.bytesCapacity / 16,
                "production reads the admission capacity at every commit")
        dropAndWait(capture)
        #expect(fixture.admission.bytesReserved == before)
        capture.close()
    }

    @Test("Rolling retirement preserves first/latest and cancellation drops both generations")
    func rollingRetirement() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        let capture = retentionCapture(fixture)
        let before = fixture.admission.bytesReserved
        _ = try commit(capture, state: original, positions: [chunkSize, 2 * chunkSize, 3 * chunkSize])
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 3 * chunkSize])
        let done = DispatchSemaphore(value: 0)
        let retirementStarted = capture.drop(requestID: .init(9001), completion: { done.signal() })
        #expect(retirementStarted)
        #expect(done.wait(timeout: .now() + 10) == .success)
        #expect(!capture.hasCheckpoints(requestID: .init(9001)))
        #expect(fixture.admission.bytesReserved == before)
        capture.close()
        #expect(try capture.prepareHistorical(position: 3 * chunkSize, chunkSize: chunkSize, state: original) == nil)
    }


    @Test("Private construction/evaluation failures cannot replace a serving fence or refund a retained array")
    func failedPrivateGraphOwnership() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        write(original, start: 0, count: chunkSize)
        let row = try #require(original[0] as? PagedSequenceKV)
        let group = row.pool.group(row.groupKey)
        let fence = ObjectIdentifier(group.writeFence)
        let before = fixture.admission.bytesReserved
        weak var failedConstruction: MLXArray?
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            try CBv2HistoricalWindow(row: row, position: chunkSize, admission: fixture.admission,
                afterConstruction: { array in
                    failedConstruction = array
                    #expect(fixture.admission.bytesReserved > before)
                    throw CBv2CompleteCheckpointError.allocationFailed
                })
        }
        #expect(failedConstruction == nil && fixture.admission.bytesReserved == before)
        #expect(ObjectIdentifier(group.writeFence) == fence)
        weak var failedEvaluation: MLXArray?
        var evaluations = 0
        var owner: CBv2HistoricalWindow? = try .init(row: row, position: chunkSize, admission: fixture.admission,
            evaluate: { array in
                failedEvaluation = array
                evaluations += 1
                try withError { eval(array) }
                throw CBv2CompleteCheckpointError.allocationFailed
            })
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) { try owner?.finishEvaluation() }
        #expect(ObjectIdentifier(group.writeFence) == fence)
        #expect(failedEvaluation != nil && fixture.admission.bytesReserved > before)
        owner = nil
        #expect(evaluations == 1, "retirement never retries a failed graph")
        #expect(failedEvaluation == nil && fixture.admission.bytesReserved == before)
        write(original, start: chunkSize, count: 16)
        #expect(values(row).count == 17 * 64, "typed optional failure leaves serving target fence intact")
    }

    @Test("An interior position captures while its window is resident and is refused once the ring moved on")
    func interiorWindowResidency() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        let row = try #require(original[0] as? PagedSequenceKV)
        // The fixture ring holds exactly one chunk (maxPrefillChunk == chunkSize).
        #expect(row.ringPages! * row.pool.config.pageSize == chunkSize)
        write(original, start: 0, count: chunkSize)
        write(original, start: chunkSize, count: chunkSize)
        #expect(row.absoluteOffset == 2 * chunkSize && row.oldestValidPosition == chunkSize)
        // [c + 17 - 17, c + 17) is still resident; [c - 17, c) is not.
        let resident = try CBv2HistoricalWindow(row: row, position: chunkSize + 17, admission: fixture.admission)
        try resident.finishEvaluation()
        #expect(resident.start == chunkSize && resident.position == chunkSize + 17)
        let bytes = 17 * 64 * row.groupKey.dtype.size
        let keys = try resident.read(values: false, byteOffset: 0, maximumBytes: bytes)
        let expected = MLXArray((chunkSize ..< chunkSize + 17).flatMap { Array(repeating: Float($0), count: 64) },
                                [17, 64]).asType(row.groupKey.dtype)
        #expect(keys == expected.asData(),
                "an interior window copies the exact rows the frontier left behind in the ring")
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try CBv2HistoricalWindow(row: row, position: chunkSize, admission: fixture.admission)
        }
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try CBv2HistoricalWindow(row: row, position: 2 * chunkSize + 1, admission: fixture.admission)
        }
        let capture = CBv2CompleteCheckpointCapture(codec: fixture.codec, store: CompleteCheckpointFixtureStore())
        #expect(try capture.prepareHistorical(position: chunkSize, chunkSize: chunkSize, state: original) == nil,
                "an evicted interior window is skipped, never a stale copy")
        capture.close()
    }

    /// Real catalog window geometry, one owning sliding layer per model. The
    /// per-checkpoint charge is that figure times the model's owner count; it
    /// is what `historicalStagedByteBudget` and K = 8 are sized against.
    /// K/V storage type is what the loaded model's projections and RoPE
    /// produce, not the weight type: the live gpt-oss-20b manifest reports
    /// float32 for 23 of its 24 layers (one sliding owner is bfloat16);
    /// gemma-4-26b stores 16-bit K/V.
    @Test("Per-checkpoint window reservation for the historical catalog models",
          arguments: [("gpt-oss-20b", 128, 8, 64, 64, 12, 4), ("gemma-4-26b", 1024, 8, 256, 16, 25, 2)])
    func catalogWindowReservation(model: String, window: Int, kvHeads: Int, headDim: Int,
                                  queryHeads: Int, owners: Int, elementBytes: Int) throws {
        let kinds = [
            CBv2LayerKind(attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads),
            CBv2LayerKind(attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads),
        ]
        let dtype: DType = elementBytes == 4 ? .float32 : .bfloat16
        let stride = CBv2RecurrentCheckpointGeometry.historicalCheckpointStrideTokens
        let config = PagedKVPoolConfig(capacityBytes: 256 << 20, maxPrefillChunk: 2 * stride,
            segmentSizeBytes: 8 << 20, layerDTypes: Array(repeating: dtype, count: kinds.count))
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 256 << 20,
            config: .init(watermarkFraction: 0, elementBytes: dtype.size,
                          layerElementBytes: Array(repeating: dtype.size, count: kinds.count)),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        try admission.reserve(id: .init(1), additionalTokens: 3 * stride)
        var state = try backend.makeSequenceState(layerKinds: kinds, promptLength: 2 * stride + 1, maxLength: 3 * stride)
        defer { backend.release(state); state.removeAll(); admission.releaseAll(id: .init(1)) }
        for entry in state {
            guard let row = entry as? PagedSequenceKV else { continue }
            let array = MLXArray.zeros([kvHeads, 2 * stride, headDim], dtype: dtype)
            row.write(keys: array, values: array)
        }
        let row = try #require(state[0] as? PagedSequenceKV)
        #expect(row.ringPages == 2 * stride / row.pool.config.pageSize,
                "the ring holds max(window + speculative span, maxPrefillChunk) tokens")
        // Interior residency at production geometry: after the solo stripe
        // [0, 2048) both models can capture 1,024; after a company chunk
        // shifted the stripe to [512, 2560) only gpt-oss (W = 128) still
        // holds [1024 - W, 1024), gemma-4 (W = 1,024) has lost it.
        _ = try CBv2HistoricalWindow.reservationBytes(row: row, position: stride)
        for entry in state {
            guard let row = entry as? PagedSequenceKV else { continue }
            row.write(keys: MLXArray.zeros([kvHeads, stride / 2, headDim], dtype: dtype),
                      values: MLXArray.zeros([kvHeads, stride / 2, headDim], dtype: dtype))
        }
        #expect(row.absoluteOffset == 2 * stride + stride / 2)
        if window < stride / 2 {
            _ = try CBv2HistoricalWindow.reservationBytes(row: row, position: stride)
        } else {
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try CBv2HistoricalWindow.reservationBytes(row: row, position: stride)
            }
        }
        _ = try CBv2HistoricalWindow.reservationBytes(row: row, position: 2 * stride)
        let perLayer = try CBv2HistoricalWindow.reservationBytes(row: row, position: 2 * stride)
        let logical = 2 * kvHeads * window * headDim * dtype.size
        let perCheckpoint = perLayer * owners
        let retained = CBv2HistoricalCheckpointRetention.maximumRetained
        print("[historical-window-reservation] model=\(model) owners=\(owners) window=\(window) "
            + "dtype=\(dtype) logicalPerLayer=\(logical) reservedPerLayer=\(perLayer) "
            + "perCheckpoint=\(perCheckpoint) staged\(retained)=\(retained * perCheckpoint)")
        #expect(perLayer >= logical && perLayer < 2 * logical + (1 << 20))
        if model == "gpt-oss-20b" {
            #expect(logical == 524_288, "float32 K/V: 2 x 8 heads x 128 tokens x 64 x 4 bytes")
            #expect(perCheckpoint > 6 << 20 && perCheckpoint < 12 << 20,
                    "gpt-oss stages ~7 MB per checkpoint; three are ~22 MB")
        } else {
            #expect(perCheckpoint > 128 << 20 && perCheckpoint < 512 << 20,
                    "gemma-4 stages ~200 MB per checkpoint; the byte budget, not K, bounds it")
        }
    }

    @Test("Custom copy stream survives its task-local scope through failure, abandonment or successful close",
          arguments: ["failure", "abandonment", "success"])
    func customStreamRetirement(outcome: String) throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer { fixture.backend.release(original); original.removeAll(); fixture.admission.releaseAll(id: .init(9001)) }
        // Source writes belong to the normal stream; the copy must consume
        // their dependencies through MLX events without globally draining it.
        write(original, start: 0, count: chunkSize)
        let row = try #require(original[0] as? PagedSequenceKV)
        let fence = ObjectIdentifier(row.pool.group(row.groupKey).writeFence)
        let before = fixture.admission.bytesReserved
        let outerStream = StreamOrDevice.default
        var capturedStream: StreamOrDevice?
        var drains = 0
        weak var output: MLXArray?
        var owner: CBv2HistoricalWindow?
        try Stream.withNewDefaultStream(device: .gpu) {
            capturedStream = .default
            #expect(capturedStream != outerStream)
            owner = try CBv2HistoricalWindow(row: row, position: chunkSize, admission: fixture.admission,
                afterConstruction: { output = $0 },
                evaluate: { array in
                    if outcome == "success" { try withError { eval(array) } }
                    else {
                        try withError { asyncEval(array) }
                        throw CBv2CompleteCheckpointError.allocationFailed
                    }
                },
                synchronize: { stream in
                    #expect(stream == capturedStream)
                    #expect(StreamOrDevice.default == outerStream)
                    #expect(output != nil && fixture.admission.bytesReserved > before)
                    drains += 1
                    try withError { stream.stream.synchronize() }
                })
            #expect(owner?.copyStream == capturedStream)
            if outcome == "abandonment" {
                owner?.markSubmitted()
                let root = try #require(owner?.evaluationRoot)
                try withError { asyncEval(root) }
            }
        }
        if outcome == "failure" {
            #expect(throws: CBv2CompleteCheckpointError.allocationFailed) { try owner?.finishEvaluation() }
            #expect(drains == 1 && output != nil && fixture.admission.bytesReserved > before)
        } else if outcome == "success" {
            try owner?.finishEvaluation()
            #expect(drains == 0 && output != nil && fixture.admission.bytesReserved > before,
                    "successful evaluation keeps its charge until retirement drains completion handlers")
        }
        owner = nil
        #expect(drains == 1 && output == nil && fixture.admission.bytesReserved == before)
        #expect(ObjectIdentifier(row.pool.group(row.groupKey).writeFence) == fence)
        write(original, start: chunkSize, count: 16)
        #expect(values(row).count == 17 * 64)
    }

}
