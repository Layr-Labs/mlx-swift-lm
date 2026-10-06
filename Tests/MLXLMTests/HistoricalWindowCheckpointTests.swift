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
            CBv2LayerKind(
                attention: .slidingWindow(17), sharesKVWithLayer: 0,
                headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(
                attention: .full, sharesKVWithLayer: 1, hasSinks: true,
                headDim: 64, kvHeads: 1, queryHeads: 2),
        ]
        let config = PagedKVPoolConfig(
            capacityBytes: 256 << 20, maxPrefillChunk: chunkSize,
            segmentSizeBytes: 64 << 10, layerDTypes: Array(repeating: dtype, count: kinds.count))
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 256 << 20,
            config: .init(
                watermarkFraction: 0, elementBytes: dtype.size,
                layerElementBytes: Array(repeating: dtype.size, count: kinds.count)),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        let identity = CBv2CompleteCheckpointIdentity(
            modelAggregateHash: "tiny-window-borrower",
            promptContractID: "causal-text", buildID: "history-test",
            numericsFingerprint: "native-\(dtype)")
        let codec = CBv2CompleteCheckpointCodec(
            identity: identity, layerKinds: kinds,
            recurrentSpec: nil, kvDTypes: Array(repeating: dtype, count: kinds.count),
            assistant: nil, admission: admission, pagedConfig: config)
        let request = CBv2Request(
            id: .init(1), promptTokens: Array(repeating: 1, count: 3 * chunkSize + 1),
            maxTokens: chunkSize - 1, cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
        return .init(
            backend: backend, admission: admission, codec: codec, request: request, kinds: kinds)
    }

    private func donor(_ fixture: Fixture, maxLength: Int? = nil, id: CBv2RequestID = .init(9001))
        throws -> [CBv2SequenceKV?]
    {
        let maxLength = maxLength ?? maximumLength
        try fixture.admission.reserve(id: id, additionalTokens: maxLength)
        return try fixture.backend.makeSequenceState(
            layerKinds: fixture.kinds,
            promptLength: fixture.request.promptTokens.count, maxLength: maxLength)
    }

    private func write(_ state: [CBv2SequenceKV?], start: Int, count: Int, salt: Int = 0) {
        for (index, entry) in state.enumerated() {
            guard let row = entry as? PagedSequenceKV else { continue }
            let values = (0 ..< count).flatMap {
                Array(repeating: Float(start + $0 + index * 128 + salt), count: 64)
            }
            let array = MLXArray(values, [1, count, 64]).asType(row.groupKey.dtype)
            row.write(keys: array, values: array + MLXArray(1).asType(row.groupKey.dtype))
        }
    }

    private func checkpoint(_ fixture: Fixture, state: [CBv2SequenceKV?], position: Int)
        throws -> CBv2HistoricalCompleteCheckpoint
    {
        let row = try #require(state[0] as? PagedSequenceKV)
        let window = try CBv2HistoricalWindow(
            row: row,
            position: position, admission: fixture.admission)
        return .init(position: position, chunkSize: chunkSize, windows: [0: window])
    }

    private func stageSource(_ source: CBv2CompleteCheckpointExport, fixture: Fixture)
        throws -> CBv2StagedCompleteCheckpoint
    {
        let plan = try fixture.codec.plan(manifest: source.manifest, request: fixture.request)
        let sink = try plan.allocate(onRelease: {})
        defer { sink.close() }
        for (index, descriptor) in source.manifest.tensors.enumerated() {
            var offset = 0
            while offset < descriptor.byteCount {
                let data = try source.readSegment(
                    tensorIndex: index, byteOffset: offset, maximumBytes: 258)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: data)
                offset += data.count
            }
        }
        return try sink.finish()
    }

    private func importSource(
        _ source: CBv2CompleteCheckpointExport, fixture: Fixture, id: CBv2RequestID
    )
        throws -> [CBv2SequenceKV?]
    {
        let staged = try stageSource(source, fixture: fixture)
        defer { staged.close() }
        return try staged.consumePreparedState { prepared in
            let frame = try #require(prepared.pagedFrame)
            prepared.pagedFrame = nil
            let adopted = try fixture.backend.pool.importCheckpoint(
                frame, admission: fixture.admission,
                requestID: id, layerKinds: fixture.kinds,
                maximumTokens: staged.maximumSequenceLength)
            return try adopted.moveToActiveRequest { #expect($0.isEmpty) }
        }
    }

    private func values(_ row: PagedSequenceKV) -> [Float] {
        row.gatherRange(start: row.absoluteOffset - row.retainedCount, count: row.retainedCount)
            .keys.asType(.float32).asArray(Float.self)
    }

    @Test(
        "Captured ring survives terminal overwrite; two restored branches own independent pages",
        arguments: [DType.float16, .bfloat16, .float32])
    func historicalBranches(dtype: DType) throws {
        let fixture = try fixture(dtype: dtype)
        func exercise() throws {
            var original = try donor(fixture)
            defer {
                fixture.backend.release(original)
                original.removeAll()
                fixture.admission.releaseAll(id: .init(9001))
            }
            write(original, start: 0, count: chunkSize)
            let first = try checkpoint(fixture, state: original, position: chunkSize)
            // The engine forbids chained successors while this candidate is
            // pending. Complete that launch boundary before later ring writes.
            try first.finishEvaluation()
            write(original, start: chunkSize, count: chunkSize)
            let second = try checkpoint(fixture, state: original, position: 2 * chunkSize)
            try second.finishEvaluation()
            write(original, start: 2 * chunkSize, count: chunkSize)
            let old = try fixture.codec.exportHistorical(
                checkpoint: first, state: original,
                tokens: fixture.request.promptTokens, cacheSalt: fixture.request.cacheSalt)
            let recent = try fixture.codec.exportHistorical(
                checkpoint: second, state: original,
                tokens: fixture.request.promptTokens, cacheSalt: fixture.request.cacheSalt)
            defer {
                old.close()
                recent.close()
            }
            #expect(
                old.manifest.tensors.count == 4
                    && old.manifest.attentionLayers?.map(\.owner) == [0, 1, 0, 1])
            #expect(
                old.manifest.tensors[0].shape[2] == 17
                    && old.manifest.tensors[2].shape[2] == chunkSize)
            var left = try importSource(old, fixture: fixture, id: .init(1))
            var right = try importSource(recent, fixture: fixture, id: .init(2))
            defer {
                fixture.backend.release(left)
                left.removeAll()
                fixture.admission.releaseAll(id: .init(1))
                fixture.backend.release(right)
                right.removeAll()
                fixture.admission.releaseAll(id: .init(2))
            }
            #expect(left.count == 4 && left[2] == nil && left[3] == nil)
            #expect(right.count == 4 && right[2] == nil && right[3] == nil)
            let leftFull = try #require(left[1] as? PagedSequenceKV)
            let rightFull = try #require(right[1] as? PagedSequenceKV)
            #expect(leftFull.absoluteOffset == chunkSize && leftFull.retainedCount == chunkSize)
            #expect(
                rightFull.absoluteOffset == 2 * chunkSize
                    && rightFull.retainedCount == 2 * chunkSize)
            let l = try #require(left[0] as? PagedSequenceKV)
            let r = try #require(right[0] as? PagedSequenceKV)
            #expect(
                l.absoluteOffset == chunkSize && l.baseOffset == chunkSize - 17
                    && l.writtenHighWater == chunkSize)
            #expect(
                r.absoluteOffset == 2 * chunkSize && r.baseOffset == 2 * chunkSize - 17
                    && r.writtenHighWater == 2 * chunkSize)
            #expect(l.decodeTableLength == l.ringPages && r.decodeTableLength == r.ringPages)
            #expect(
                values(l)
                    == ((chunkSize - 17) ..< chunkSize).flatMap {
                        Array(repeating: Float($0), count: 64)
                    })
            let before = values(r)
            write(left, start: chunkSize, count: 16, salt: 1000)
            #expect(
                values(r) == before, "different request positions do not share a mutable frontier")
            #expect(Set(l.table).isDisjoint(with: Set(r.table)))
            #expect(
                fixture.admission.bytesReserved >= 2
                    * fixture.admission.allocatedBytes(forTokens: maximumLength))
            let reuse = try fixture.codec.historicalReusePlan(
                position: chunkSize, maximumSequenceLength: maximumLength)
            #expect(
                reuse.strategy == .direct && reuse.replayTokens == 0
                    && reuse.prefillTokensSaved == chunkSize)
            #expect(reuse.capacityTokensForChunk(start: chunkSize, count: chunkSize) == 0)
        }
        try exercise()
        #expect(fixture.admission.bytesReserved == 0 && fixture.backend.bytesWired == 0)
    }

    @Test("Capture refusal occurs before page metadata or GPU graph construction")
    func refusalAndFailure() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        write(original, start: 0, count: chunkSize)
        let row = try #require(original[0] as? PagedSequenceKV)
        let tiny = AdmissionV2(
            layerKinds: [], bytesCapacity: 1, config: .init(watermarkFraction: 0))
        var allocations = 0
        #expect(throws: (any Error).self) {
            try CBv2HistoricalWindow(
                row: row, position: chunkSize, admission: tiny,
                beforeAllocation: { allocations += 1 })
        }
        #expect(allocations == 0 && tiny.bytesReserved == 0)
        let before = fixture.admission.bytesReserved
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            try CBv2HistoricalWindow(
                row: row, position: chunkSize, admission: fixture.admission,
                beforeAllocation: {
                    allocations += 1
                    throw CBv2CompleteCheckpointError.allocationFailed
                })
        }
        #expect(allocations == 1 && fixture.admission.bytesReserved == before)
    }

    @Test("Last K/V source owns the capture charge through closure")
    func sourceLifetime() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        write(original, start: 0, count: chunkSize)
        let row = try #require(original[0] as? PagedSequenceKV)
        let before = fixture.admission.bytesReserved
        var capture: CBv2HistoricalWindow? = try .init(
            row: row, position: chunkSize, admission: fixture.admission)
        let charge = fixture.admission.bytesReserved - before
        let keys = CBv2HistoricalWindowTensorSource(window: try #require(capture), values: false)
        let values = CBv2HistoricalWindowTensorSource(window: try #require(capture), values: true)
        capture = nil
        keys.close()
        #expect(charge > 0 && fixture.admission.bytesReserved == before + charge)
        _ = try values.readSegment(byteOffset: 0, maximumBytes: 128)
        values.close()
        #expect(fixture.admission.bytesReserved == before)
        #expect(throws: CBv2CompleteCheckpointError.closed) {
            try values.readSegment(byteOffset: 0, maximumBytes: 128)
        }
    }

    @Test("Malformed owner/layout/window/token metadata fails before staging")
    func incompatibleHistory() throws {
        let fixture = try fixture()
        let layout = try #require(fixture.codec.historicalLayout)
        let descriptors = try fixture.codec.tensorDescriptors(position: chunkSize)
        let base = CBv2CompleteCheckpointManifest(
            identity: fixture.codec.identity, position: chunkSize, chunkSize: chunkSize,
            prefixTokens: Array(fixture.request.promptTokens.prefix(chunkSize)),
            cacheSalt: fixture.request.cacheSalt,
            assistantCodecID: nil, tensors: descriptors,
            backendLayout: CBv2CompleteCheckpointManifest.historicalAttentionLayout,
            attentionLayers: layout.layers)
        _ = try fixture.codec.plan(manifest: base, request: fixture.request)
        let encoded = try JSONEncoder().encode(base)
        for mutation in ["owner", "window", "dtype", "tokens"] {
            let decoded = try JSONSerialization.jsonObject(with: encoded)
            var object = try #require(decoded as? [String: Any])
            if mutation == "tokens" {
                object["prefixTokens"] = Array(repeating: 2, count: chunkSize)
            } else {
                var layers = try #require(object["attentionLayers"] as? [[String: Any]])
                if mutation == "owner" { layers[2]["owner"] = 1 }
                if mutation == "window" { layers[0]["window"] = 18 }
                if mutation == "dtype" { layers[0]["dtype"] = "float32" }
                object["attentionLayers"] = layers
            }
            let corrupt = try JSONDecoder().decode(
                CBv2CompleteCheckpointManifest.self,
                from: JSONSerialization.data(withJSONObject: object))
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try fixture.codec.plan(manifest: corrupt, request: fixture.request)
            }
        }
        #expect(fixture.admission.bytesReserved == 0 && fixture.backend.bytesWired == 0)
    }

    @Test("Cancelled or failed window adoption drains every private destination before refund")
    func closeDuringConsumeAndFailedRestore() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        write(original, start: 0, count: chunkSize)
        let captured = try checkpoint(fixture, state: original, position: chunkSize)
        try captured.finishEvaluation()
        let source = try fixture.codec.exportHistorical(
            checkpoint: captured, state: original,
            tokens: fixture.request.promptTokens, cacheSalt: fixture.request.cacheSalt)
        defer { source.close() }
        let before = fixture.admission.bytesReserved
        let cancelled = try stageSource(source, fixture: fixture)
        #expect(fixture.admission.bytesReserved > before)
        cancelled.close()
        #expect(fixture.admission.bytesReserved == before)
        #expect(throws: CBv2CompleteCheckpointError.closed) {
            try cancelled.consumePreparedState { _ in }
        }
        let staged = try stageSource(source, fixture: fixture)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try staged.consumePreparedState { prepared in
                // Close loses ownership to consume; it cannot reclaim buffers
                // while the active transfer/restore callback still uses them.
                DispatchQueue.global().sync { staged.close() }
                #expect(fixture.admission.bytesReserved > before)
                let frame = try #require(prepared.pagedFrame)
                prepared.pagedFrame = nil
                let adopted = try fixture.backend.pool.importCheckpoint(
                    frame, admission: fixture.admission,
                    requestID: .init(1), layerKinds: fixture.kinds,
                    maximumTokens: staged.maximumSequenceLength)
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

    private func commit(
        _ capture: CBv2CompleteCheckpointCapture, state: [CBv2SequenceKV?],
        positions: [Int], hint: Int? = nil, resumedAt: Int = 0,
        id: CBv2RequestID = .init(9001)
    ) throws -> Int {
        var bytesPerCheckpoint = 0
        for position in positions {
            write(state, start: position - chunkSize, count: chunkSize)
            let prepared = try capture.prepareHistorical(
                position: position, chunkSize: chunkSize, state: state)
            let candidate = try #require(prepared)
            try candidate.finishEvaluation()
            bytesPerCheckpoint = candidate.stagedHistoricalBytes
            capture.commitHistorical(
                candidate, requestID: id, hintTokens: hint, resumedAt: resumedAt)
            #expect((capture.staged[id]?.count ?? 0) <= CBv2CheckpointRetention.maximumRetained)
            #expect(capture.staged[id]?.compactMap(\.position) == capture.retentions[id]?.retained)
        }
        return bytesPerCheckpoint
    }

    private func retentionCapture(_ fixture: Fixture) -> CBv2CompleteCheckpointCapture {
        let capture = CBv2CompleteCheckpointCapture(
            codec: fixture.codec, store: CompleteCheckpointFixtureStore())
        capture.historicalCheckpointStrideTokens = chunkSize
        return capture
    }

    private func dropAndWait(
        _ capture: CBv2CompleteCheckpointCapture, id: CBv2RequestID = .init(9001)
    ) {
        let done = DispatchSemaphore(value: 0)
        let dropped = capture.drop(requestID: id, completion: { done.signal() })
        #expect(dropped)
        #expect(done.wait(timeout: .now() + 10) == .success)
        capture.queue.sync {}
        #expect(capture.retentions[id] == nil && capture.staged[id] == nil)
    }

    @Test(
        "Without a hint a donor stages the first and the rolling latest; retired windows refund their charge"
    )
    func retentionKeepsFirstAndDeepest() throws {
        let fixture = try fixture()
        var original = try donor(fixture, maxLength: 8 * chunkSize)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        let capture = retentionCapture(fixture)
        #expect(capture.historicalStagedByteBudget == fixture.admission.bytesCapacity / 16)
        let before = fixture.admission.bytesReserved
        let bytes = try commit(
            capture, state: original, positions: (1 ... 6).map { $0 * chunkSize })
        #expect(bytes > 0)
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 6 * chunkSize])
        capture.queue.sync {}
        #expect(fixture.admission.bytesReserved == before + 2 * bytes)
        // A duplicate boundary is refused without touching the retained set.
        let duplicate = try #require(
            try capture.prepareHistorical(
                position: 6 * chunkSize, chunkSize: chunkSize,
                state: original))
        try duplicate.finishEvaluation()
        capture.commitHistorical(duplicate, requestID: .init(9001))
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 6 * chunkSize])
        dropAndWait(capture)
        #expect(
            fixture.admission.bytesReserved == before,
            "retired interior windows refund their transient charge")
        capture.close()
    }

    @Test(
        "A fork hint stages first, target and latest; an adopter stages nothing at or below its restore point"
    )
    func retentionKeepsForkTarget() throws {
        let fixture = try fixture()
        var original = try donor(fixture, maxLength: 8 * chunkSize)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        let capture = retentionCapture(fixture)
        let before = fixture.admission.bytesReserved
        let bytes = try commit(
            capture, state: original, positions: (1 ... 6).map { $0 * chunkSize },
            hint: 3 * chunkSize + 9)
        #expect(
            capture.staged[.init(9001)]?.compactMap(\.position) == [
                chunkSize, 3 * chunkSize, 6 * chunkSize,
            ])
        #expect(
            capture.retentions[.init(9001)]?.publication
                == [6 * chunkSize, 3 * chunkSize, chunkSize])
        capture.queue.sync {}
        #expect(fixture.admission.bytesReserved == before + 3 * bytes)
        dropAndWait(capture)
        #expect(fixture.admission.bytesReserved == before)

        // The same rows seen by an adopter that restored at 3c with the same
        // hint: the first and the target are already durable below it.
        var adopted = try fixture.backend.makeSequenceState(
            layerKinds: fixture.kinds,
            promptLength: fixture.request.promptTokens.count, maxLength: 8 * chunkSize)
        try fixture.admission.reserve(id: .init(9002), additionalTokens: 8 * chunkSize)
        defer {
            fixture.backend.release(adopted)
            adopted.removeAll()
            fixture.admission.releaseAll(id: .init(9002))
        }
        write(adopted, start: 0, count: chunkSize)
        write(adopted, start: chunkSize, count: chunkSize)
        write(adopted, start: 2 * chunkSize, count: chunkSize)
        _ = try commit(
            capture, state: adopted, positions: (4 ... 6).map { $0 * chunkSize },
            hint: 3 * chunkSize + 9, resumedAt: 3 * chunkSize, id: .init(9002))
        #expect(capture.staged[.init(9002)]?.compactMap(\.position) == [6 * chunkSize])
        #expect(capture.retentions[.init(9002)]?.first == nil)
        dropAndWait(capture, id: .init(9002))
        capture.close()
    }

    @Test(
        "One range copies only the boundaries retention keeps: the latest, an open first and the target"
    )
    func prepareCopiesOnlyRetainedBoundaries() throws {
        let fixture = try fixture()
        var original = try donor(fixture, maxLength: 8 * chunkSize)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        let capture = CBv2CompleteCheckpointCapture(
            codec: fixture.codec, store: CompleteCheckpointFixtureStore())
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
            let retention = capture.retention(
                requestID: .init(9001), stride: stride, hintTokens: hint, resumedAt: 0)
            let prepared = try capture.prepareHistorical(
                positions: positions, retention: retention, state: original)
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
        #expect(
            capture.staged[.init(9001)]?.compactMap(\.position) == [stride, 6 * stride, 8 * stride])
        // Third range: only the rolling latest.
        let thirdRange = try range(2, hint: hint)
        #expect(thirdRange == [12 * stride])
        #expect(
            capture.staged[.init(9001)]?.compactMap(\.position) == [
                stride, 6 * stride, 12 * stride,
            ])
        // A refused deepest boundary hands the latest to the next one down.
        let bounded = CBv2CompleteCheckpointCapture(
            codec: fixture.codec,
            store: CompleteCheckpointFixtureStore(maximumPosition: 15 * stride))
        bounded.historicalCheckpointStrideTokens = stride
        bounded.makeHistoricalWindow = capture.makeHistoricalWindow
        write(original, start: 3 * chunkSize, count: chunkSize)
        copied.removeAll()
        let retention = bounded.retention(
            requestID: .init(9003), stride: bounded.historicalCheckpointStrideTokens,
            hintTokens: nil, resumedAt: 2 * chunkSize)
        let prepared = try bounded.prepareHistorical(
            positions: (13 ... 16).map { $0 * stride },
            retention: retention, state: original)
        #expect(prepared.compactMap(\.position) == [15 * stride] && copied == [15 * stride])
        prepared.forEach { $0.finishEvaluationAndClose() }
        dropAndWait(capture)
        capture.close()
        bounded.close()
    }

    /// One donor's next chunk, captured the way the engine does it: every
    /// stride boundary of the range offered, retention and the slot-wide cap
    /// deciding which are copied. Returns the positions that were copied.
    private func stageRange(
        _ capture: CBv2CompleteCheckpointCapture, state: [CBv2SequenceKV?],
        id: CBv2RequestID, through position: Int, hint: Int? = nil
    ) throws -> [Int] {
        write(state, start: position - chunkSize, count: chunkSize)
        let stride = capture.historicalCheckpointStrideTokens
        let positions = Array(
            Swift.stride(from: position - chunkSize + stride, through: position, by: stride))
        let retention = capture.retention(
            requestID: id, stride: capture.historicalCheckpointStrideTokens, hintTokens: hint,
            resumedAt: 0)
        let prepared = try capture.prepareHistorical(
            positions: positions, retention: retention, state: state, requestID: id)
        let copied = prepared.compactMap(\.position)
        for candidate in prepared {
            try candidate.finishEvaluation()
            capture.commitHistorical(candidate, requestID: id, hintTokens: hint)
        }
        #expect(
            capture.staged[id]?.compactMap(\.position) == capture.retentions[id]?.retained
                || (capture.staged[id] == nil && capture.retentions[id]?.retained.isEmpty != false))
        return copied
    }

    private func staged(_ capture: CBv2CompleteCheckpointCapture, _ id: CBv2RequestID) -> [Int] {
        capture.staged[id]?.compactMap(\.position) ?? []
    }

    @Test(
        "The slot-wide cap refuses the donor that would exceed it, admits it once another releases, and lets a donor at the cap roll its latest"
    )
    func slotCapAcrossDonors() throws {
        let fixture = try fixture()
        let ids: [CBv2RequestID] = [.init(9001), .init(9002), .init(9003)]
        var states = try ids.map { try donor(fixture, maxLength: 8 * chunkSize, id: $0) }
        defer {
            for (state, id) in zip(states, ids) {
                fixture.backend.release(state)
                fixture.admission.releaseAll(id: id)
            }
            states.removeAll()
        }
        let capture = retentionCapture(fixture)
        #expect(
            capture.historicalSlotStagedByteCap == fixture.admission.bytesCapacity / 8,
            "production reads the slot capacity at every capture")
        let before = fixture.admission.transientBytesReserved
        func stagedBytes() -> Int {
            capture.queue.sync {}
            return fixture.admission.transientBytesReserved - before
        }
        let first = try stageRange(capture, state: states[0], id: ids[0], through: chunkSize)
        #expect(first == [chunkSize])
        let window = stagedBytes()
        #expect(window > 0 && capture.stagedHistoricalBytes == window)
        capture.historicalSlotStagedByteCapOverride = 4 * window

        _ = try stageRange(capture, state: states[0], id: ids[0], through: 2 * chunkSize)
        _ = try stageRange(capture, state: states[1], id: ids[1], through: chunkSize)
        _ = try stageRange(capture, state: states[1], id: ids[1], through: 2 * chunkSize)
        #expect(staged(capture, ids[0]) == [chunkSize, 2 * chunkSize])
        #expect(staged(capture, ids[1]) == [chunkSize, 2 * chunkSize])
        #expect(
            stagedBytes() == 4 * window && capture.stagedHistoricalBytes == 4 * window,
            "the cap is full")

        // A donor at the cap still rolls its latest: the replacement does not raise the total.
        let rolled = try stageRange(capture, state: states[1], id: ids[1], through: 3 * chunkSize)
        #expect(rolled == [3 * chunkSize])
        #expect(staged(capture, ids[1]) == [chunkSize, 3 * chunkSize])
        #expect(stagedBytes() == 4 * window)

        // The third donor is refused: nothing copied, nothing reserved.
        var copies = 0
        capture.makeHistoricalWindow = { row, position, admission in
            copies += 1
            return try CBv2HistoricalWindow(row: row, position: position, admission: admission)
        }
        let refused = try stageRange(capture, state: states[2], id: ids[2], through: chunkSize)
        #expect(refused.isEmpty && copies == 0 && capture.staged[ids[2]] == nil)
        #expect(stagedBytes() == 4 * window)
        #expect(capture.inFlightHistoricalBytes == 0)

        // Releasing one donor admits the next.
        dropAndWait(capture, id: ids[0])
        #expect(stagedBytes() == 2 * window)
        let admitted = try stageRange(capture, state: states[2], id: ids[2], through: 2 * chunkSize)
        #expect(admitted == [2 * chunkSize] && copies == 1)
        _ = try stageRange(capture, state: states[2], id: ids[2], through: 3 * chunkSize)
        #expect(staged(capture, ids[2]) == [2 * chunkSize, 3 * chunkSize])
        #expect(stagedBytes() == 4 * window)
        let again = try stageRange(capture, state: states[2], id: ids[2], through: 4 * chunkSize)
        #expect(again == [4 * chunkSize] && stagedBytes() == 4 * window)

        dropAndWait(capture, id: ids[1])
        dropAndWait(capture, id: ids[2])
        #expect(stagedBytes() == 0 && capture.stagedHistoricalBytes == 0)
        capture.close()
    }

    @Test(
        "Checkpoints being written still count against the slot-wide cap until their batch closes")
    func slotCapCountsPublication() throws {
        let fixture = try fixture()
        let ids: [CBv2RequestID] = [.init(9001), .init(9002), .init(9003)]
        var states = try ids.map { try donor(fixture, maxLength: 8 * chunkSize, id: $0) }
        defer {
            for (state, id) in zip(states, ids) {
                fixture.backend.release(state)
                fixture.admission.releaseAll(id: id)
            }
            states.removeAll()
        }
        let gate = CheckpointPublicationGate()
        let store = CompleteCheckpointFixtureStore(gate: gate)
        let capture = CBv2CompleteCheckpointCapture(codec: fixture.codec, store: store)
        capture.historicalCheckpointStrideTokens = chunkSize
        let before = fixture.admission.transientBytesReserved
        func stagedBytes() -> Int {
            capture.queue.sync {}
            return fixture.admission.transientBytesReserved - before
        }
        // Donor A stages two boundaries, then finishes and publishes them.
        _ = try stageRange(capture, state: states[0], id: ids[0], through: chunkSize)
        _ = try stageRange(capture, state: states[0], id: ids[0], through: 2 * chunkSize)
        let window = stagedBytes() / 2
        try #require(window > 0 && capture.stagedHistoricalBytes == 2 * window)
        capture.historicalSlotStagedByteCapOverride = 3 * window
        let published = DispatchSemaphore(value: 0)
        final class Positions: @unchecked Sendable {
            private let lock = NSLock()
            private var values: [Int] = []
            var value: [Int] { lock.withLock { values } }
            func set(_ new: [Int]) { lock.withLock { values = new } }
        }
        let positions = Positions()
        capture.publish(
            intent: .init(
                requestID: ids[0], tokens: fixture.request.promptTokens,
                cacheSalt: fixture.request.cacheSalt),
            state: states[0]
        ) {
            positions.set($0)
            published.signal()
        }
        #expect(capture.staged[ids[0]] == nil && capture.retentions[ids[0]] == nil)
        #expect(gate.waitUntilEntered(), "the first file is being written")
        #expect(capture.stagedHistoricalBytes == 2 * window, "publication still holds two windows")
        #expect(capture.publishingHistoricalBytesTotal == 2 * window)
        // The windows are still charged, under the batch's export scratch,
        // page maps and manifest permits, which retire with the batch.
        #expect(stagedBytes() >= 2 * window, "their reservations are still charged")

        // Donor B fits one boundary beside the publication and rolls it,
        // giving up its first for the new latest; donor C finds no room
        // until the batch closes.
        #expect(
            try stageRange(capture, state: states[1], id: ids[1], through: chunkSize) == [chunkSize]
        )
        #expect(capture.stagedHistoricalBytes == 3 * window)
        #expect(
            try stageRange(capture, state: states[1], id: ids[1], through: 2 * chunkSize) == [
                2 * chunkSize
            ])
        #expect(staged(capture, ids[1]) == [2 * chunkSize])
        let stagedByB = staged(capture, ids[1])
        #expect(try stageRange(capture, state: states[2], id: ids[2], through: chunkSize).isEmpty)
        #expect(capture.staged[ids[2]] == nil)
        #expect(
            capture.stagedHistoricalBytes <= 3 * window,
            "a donor in publication plus new donors never exceed the cap")

        // The store gates every file; the donor writes two, deepest first.
        gate.resume.signal()
        #expect(gate.waitUntilEntered(), "the second file is being written")
        #expect(
            capture.publishingHistoricalBytesTotal == 2 * window,
            "still counted until the batch closes")
        gate.resume.signal()
        #expect(published.wait(timeout: .now() + 10) == .success)
        capture.queue.sync {}
        #expect(
            positions.value == [2 * chunkSize, chunkSize], "both files were written, deepest first")
        #expect(
            capture.publishingHistoricalBytesTotal == 0,
            "the batch closed: publication no longer counts")
        #expect(capture.stagedHistoricalBytes == stagedByB.count * window)
        #expect(stagedBytes() == stagedByB.count * window)
        #expect(
            try stageRange(capture, state: states[2], id: ids[2], through: 2 * chunkSize) == [
                2 * chunkSize
            ],
            "the room the publication held is available again")
        for id in ids where capture.staged[id] != nil { dropAndWait(capture, id: id) }
        #expect(stagedBytes() == 0 && capture.stagedHistoricalBytes == 0)
        capture.close()
    }

    @Test("A discarded candidate leaves nothing in flight and refunds its windows")
    func discardLeavesNothingInFlight() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        let capture = retentionCapture(fixture)
        let before = fixture.admission.transientBytesReserved
        write(original, start: 0, count: chunkSize)
        let retention = capture.retention(
            requestID: .init(9001), stride: capture.historicalCheckpointStrideTokens,
            hintTokens: nil, resumedAt: 0)
        let prepared = try capture.prepareHistorical(
            positions: [chunkSize], retention: retention,
            state: original, requestID: .init(9001))
        let candidate = try #require(prepared.first)
        #expect(
            capture.inFlightHistoricalBytes == candidate.stagedHistoricalBytes
                && capture.inFlightHistoricalBytes > 0)
        #expect(fixture.admission.transientBytesReserved > before)
        capture.discardHistorical(candidate)
        #expect(capture.inFlightHistoricalBytes == 0)
        #expect(capture.staged[.init(9001)] == nil)
        #expect(fixture.admission.transientBytesReserved == before)
        capture.close()
    }

    @Test("Under the slot-wide cap a donor keeps its latest over its target over its first")
    func slotCapPriorityWithinDonor() throws {
        let fixture = try fixture()
        let ids: [CBv2RequestID] = [.init(9001), .init(9002)]
        var states = try ids.map { try donor(fixture, maxLength: 8 * chunkSize, id: $0) }
        defer {
            for (state, id) in zip(states, ids) {
                fixture.backend.release(state)
                fixture.admission.releaseAll(id: id)
            }
            states.removeAll()
        }
        let capture = retentionCapture(fixture)
        let before = fixture.admission.transientBytesReserved
        func stagedBytes() -> Int {
            capture.queue.sync {}
            return fixture.admission.transientBytesReserved - before
        }
        let hint = 3 * chunkSize + 5
        _ = try stageRange(capture, state: states[0], id: ids[0], through: chunkSize, hint: hint)
        let window = stagedBytes()
        capture.historicalSlotStagedByteCapOverride = 2 * window
        _ = try stageRange(
            capture, state: states[0], id: ids[0], through: 2 * chunkSize, hint: hint)
        #expect(
            staged(capture, ids[0]) == [chunkSize, 2 * chunkSize] && stagedBytes() == 2 * window)
        // The fork target replaces the role-less latest.
        _ = try stageRange(
            capture, state: states[0], id: ids[0], through: 3 * chunkSize, hint: hint)
        #expect(staged(capture, ids[0]) == [chunkSize, 3 * chunkSize])
        // A new latest beside a full cap: the first is given up, the target kept.
        let latest = try stageRange(
            capture, state: states[0], id: ids[0], through: 4 * chunkSize, hint: hint)
        #expect(latest == [4 * chunkSize])
        #expect(
            staged(capture, ids[0]) == [3 * chunkSize, 4 * chunkSize],
            "latest over target over first")
        #expect(capture.retentions[ids[0]]?.first == chunkSize, "a given-up first is not reopened")
        #expect(stagedBytes() == 2 * window)
        _ = try stageRange(
            capture, state: states[0], id: ids[0], through: 5 * chunkSize, hint: hint)
        #expect(staged(capture, ids[0]) == [3 * chunkSize, 5 * chunkSize])
        // The slot shrinks to one window: the latest outlives the target.
        capture.historicalSlotStagedByteCapOverride = window
        _ = try stageRange(
            capture, state: states[0], id: ids[0], through: 6 * chunkSize, hint: hint)
        #expect(staged(capture, ids[0]) == [6 * chunkSize])
        #expect(stagedBytes() == window)
        // A first never displaces anything: another donor finds no room.
        let other = try stageRange(capture, state: states[1], id: ids[1], through: chunkSize)
        #expect(other.isEmpty && capture.staged[ids[1]] == nil)
        // With no room at all even the latest is refused and what is staged stays.
        capture.historicalSlotStagedByteCapOverride = window - 1
        let none = try stageRange(
            capture, state: states[0], id: ids[0], through: 7 * chunkSize, hint: hint)
        #expect(none.isEmpty && staged(capture, ids[0]) == [6 * chunkSize])
        dropAndWait(capture, id: ids[0])
        #expect(stagedBytes() == 0)
        capture.close()
    }

    @Test("With room for one more boundary in a range, the fork target is copied before the first")
    func slotCapTargetBeforeFirst() throws {
        let fixture = try fixture()
        var original = try donor(fixture, maxLength: 8 * chunkSize)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        let capture = CBv2CompleteCheckpointCapture(
            codec: fixture.codec, store: CompleteCheckpointFixtureStore())
        let stride = chunkSize / 4
        capture.historicalCheckpointStrideTokens = stride
        let before = fixture.admission.transientBytesReserved
        write(original, start: 0, count: chunkSize)
        let row = try #require(original[0] as? PagedSequenceKV)
        let window = try CBv2HistoricalWindow.reservationBytes(row: row, position: 4 * stride)
        capture.historicalSlotStagedByteCapOverride = 2 * window
        let hint = 2 * stride + 1
        let retention = capture.retention(
            requestID: .init(9001), stride: capture.historicalCheckpointStrideTokens,
            hintTokens: hint, resumedAt: 0)
        let prepared = try capture.prepareHistorical(
            positions: (1 ... 4).map { $0 * stride },
            retention: retention, state: original, requestID: .init(9001))
        #expect(
            prepared.compactMap(\.position) == [2 * stride, 4 * stride],
            "latest, then target; no room for the first")
        #expect(capture.inFlightHistoricalBytes == 2 * window)
        for candidate in prepared {
            try candidate.finishEvaluation()
            capture.commitHistorical(candidate, requestID: .init(9001), hintTokens: hint)
        }
        #expect(capture.inFlightHistoricalBytes == 0)
        #expect(staged(capture, .init(9001)) == [2 * stride, 4 * stride])
        capture.queue.sync {}
        #expect(fixture.admission.transientBytesReserved == before + 2 * window)
        dropAndWait(capture)
        #expect(fixture.admission.transientBytesReserved == before)
        capture.close()
    }

    /// The ledger the scheduler reserves chunks against is the one staged
    /// windows are charged to. Three donors that each want three staged
    /// boundaries: capped to three windows in all, a request's reservation
    /// fits; uncapped (nine windows), the SAME reservation exhausts the
    /// ledger, which in the scheduler is a preemption or a refused admission.
    @Test("Capped staged windows leave the ledger room a request's chunk reservation needs")
    func slotCapLeavesRoomForServing() throws {
        func arm(capped: Bool, tokens: Int?) throws -> (
            tokens: Int, window: Int, windows: Int, reserved: Bool
        ) {
            let fixture = try fixture()
            let ids: [CBv2RequestID] = [.init(9001), .init(9002), .init(9003)]
            var states = try ids.map { try donor(fixture, maxLength: 4 * chunkSize, id: $0) }
            defer {
                for (state, id) in zip(states, ids) {
                    fixture.backend.release(state)
                    fixture.admission.releaseAll(id: id)
                }
                states.removeAll()
            }
            let capture = retentionCapture(fixture)
            capture.historicalStagedByteBudgetOverride = .max
            let before = fixture.admission.transientBytesReserved
            capture.historicalSlotStagedByteCapOverride = .max
            _ = try stageRange(
                capture, state: states[0], id: ids[0], through: chunkSize, hint: 2 * chunkSize)
            capture.queue.sync {}
            let window = fixture.admission.transientBytesReserved - before
            try #require(window > 0)
            capture.historicalSlotStagedByteCapOverride = capped ? 3 * window : .max
            for (index, (state, id)) in zip(states, ids).enumerated() {
                for multiple in 1 ... 3 where index > 0 || multiple > 1 {
                    _ = try stageRange(
                        capture, state: state, id: id, through: multiple * chunkSize,
                        hint: 2 * chunkSize)
                }
            }
            capture.queue.sync {}
            let stagedBytes = fixture.admission.transientBytesReserved - before
            #expect(stagedBytes == capture.stagedHistoricalBytes)
            #expect(stagedBytes <= capture.historicalSlotStagedByteCap)
            // The largest request that fits beside the CAPPED set with half a
            // window to spare; the uncapped arm is handed the same request.
            // Size it from what the ledger accepts, not from `bytesReserved`.
            // `bytesReserved` includes the pool's physical-floor overhead,
            // which a request's target KV replaces. That overhead depends on
            // the buffer sizes the MLX allocator returns, and a buffer reused
            // from its process-wide cache can be larger than a new one.
            func largest(_ fits: (Int) -> Bool) -> Int {
                var low = 0
                var high = 1 << 24
                while low < high {
                    let middle = (low + high + 1) / 2
                    if fits(middle) { low = middle } else { high = middle - 1 }
                }
                return low
            }
            var need = tokens ?? 0
            if tokens == nil {
                let probe = CBv2RequestID(7778)
                let accepted = largest { candidate in
                    defer { fixture.admission.releaseAll(id: probe) }
                    do {
                        try fixture.admission.reserve(id: probe, additionalTokens: candidate)
                        return true
                    } catch {
                        return false
                    }
                }
                let limit = fixture.admission.allocatedBytes(forTokens: accepted) - window / 2
                need = largest { fixture.admission.allocatedBytes(forTokens: $0) <= limit }
            }
            var reserved = true
            do { try fixture.admission.reserve(id: .init(7777), additionalTokens: need) } catch {
                reserved = false
            }
            fixture.admission.releaseAll(id: .init(7777))
            for id in ids where capture.staged[id] != nil { dropAndWait(capture, id: id) }
            capture.close()
            return (need, window, stagedBytes / window, reserved)
        }
        let capped = try arm(capped: true, tokens: nil)
        #expect(capped.windows == 3 && capped.reserved && capped.tokens > 0)
        let uncapped = try arm(capped: false, tokens: capped.tokens)
        #expect(uncapped.windows == 9, "every donor staged its first, target and latest")
        #expect(
            !uncapped.reserved, "the same reservation no longer fits beside nine staged windows")
        print(
            "[slot-cap-ledger] window=\(capped.window) cappedWindows=\(capped.windows) "
                + "uncappedWindows=\(uncapped.windows) requestTokens=\(capped.tokens) "
                + "capped=\(capped.reserved) uncapped=\(uncapped.reserved)")
    }

    @Test("The byte budget sheds the first, then the target, and always keeps the latest")
    func retentionByteBudget() throws {
        let fixture = try fixture()
        let ids: [CBv2RequestID] = [.init(9001), .init(9002)]
        var states = try ids.map { try donor(fixture, maxLength: 8 * chunkSize, id: $0) }
        defer {
            for (state, id) in zip(states, ids) {
                fixture.backend.release(state)
                fixture.admission.releaseAll(id: id)
            }
            states.removeAll()
        }
        let capture = retentionCapture(fixture)
        let before = fixture.admission.transientBytesReserved
        func stagedBytes() -> Int {
            capture.queue.sync {}
            return fixture.admission.transientBytesReserved - before
        }
        let hint = 3 * chunkSize + 5
        let perCheckpoint = try commit(
            capture, state: states[0], positions: [chunkSize], hint: hint, id: ids[0])
        #expect(perCheckpoint > 0 && stagedBytes() == perCheckpoint)

        // Room for two: latest + target outlive the first.
        capture.historicalStagedByteBudgetOverride = perCheckpoint * 5 / 2
        _ = try commit(
            capture, state: states[0], positions: [2 * chunkSize, 3 * chunkSize], hint: hint,
            id: ids[0])
        #expect(staged(capture, ids[0]) == [chunkSize, 3 * chunkSize])
        _ = try commit(
            capture, state: states[0], positions: [4 * chunkSize], hint: hint, id: ids[0])
        #expect(
            staged(capture, ids[0]) == [3 * chunkSize, 4 * chunkSize],
            "room for two keeps latest + target")
        #expect(capture.retentions[ids[0]]?.first == chunkSize, "a shed first is not reopened")
        #expect(stagedBytes() == 2 * perCheckpoint)
        _ = try commit(
            capture, state: states[0], positions: [5 * chunkSize], hint: hint, id: ids[0])
        #expect(staged(capture, ids[0]) == [3 * chunkSize, 5 * chunkSize])

        // Room for one: the latest outlives the target.
        capture.historicalStagedByteBudgetOverride = perCheckpoint * 3 / 2
        _ = try commit(
            capture, state: states[0], positions: [6 * chunkSize], hint: hint, id: ids[0])
        #expect(staged(capture, ids[0]) == [6 * chunkSize], "room for one keeps the latest")
        #expect(stagedBytes() == perCheckpoint)
        // No room at all: the latest is still the one boundary kept.
        capture.historicalStagedByteBudgetOverride = 0
        _ = try commit(
            capture, state: states[0], positions: [7 * chunkSize], hint: hint, id: ids[0])
        #expect(staged(capture, ids[0]) == [7 * chunkSize])
        #expect(stagedBytes() == perCheckpoint)
        dropAndWait(capture, id: ids[0])
        #expect(stagedBytes() == 0)

        // Without a target the pair first + latest is kept exactly as before.
        capture.historicalStagedByteBudgetOverride = perCheckpoint * 5 / 2
        _ = try commit(
            capture, state: states[1], positions: (1 ... 4).map { $0 * chunkSize }, id: ids[1])
        #expect(staged(capture, ids[1]) == [chunkSize, 4 * chunkSize], "no target: first + latest")
        #expect(stagedBytes() == 2 * perCheckpoint)
        capture.historicalStagedByteBudgetOverride = perCheckpoint * 3 / 2
        _ = try commit(capture, state: states[1], positions: [5 * chunkSize], id: ids[1])
        #expect(
            staged(capture, ids[1]) == [5 * chunkSize],
            "room for one keeps the latest, not the first")
        capture.historicalStagedByteBudgetOverride = nil
        #expect(
            capture.historicalStagedByteBudget == fixture.admission.bytesCapacity / 16,
            "production reads the admission capacity at every commit")
        dropAndWait(capture, id: ids[1])
        #expect(stagedBytes() == 0)
        capture.close()
    }

    @Test("Rolling retirement preserves first/latest and cancellation drops both generations")
    func rollingRetirement() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        let capture = retentionCapture(fixture)
        let before = fixture.admission.bytesReserved
        _ = try commit(
            capture, state: original, positions: [chunkSize, 2 * chunkSize, 3 * chunkSize])
        #expect(capture.staged[.init(9001)]?.compactMap(\.position) == [chunkSize, 3 * chunkSize])
        let done = DispatchSemaphore(value: 0)
        let retirementStarted = capture.drop(requestID: .init(9001), completion: { done.signal() })
        #expect(retirementStarted)
        #expect(done.wait(timeout: .now() + 10) == .success)
        #expect(!capture.hasCheckpoints(requestID: .init(9001)))
        #expect(fixture.admission.bytesReserved == before)
        capture.close()
        #expect(
            try capture.prepareHistorical(
                position: 3 * chunkSize, chunkSize: chunkSize, state: original) == nil)
    }

    @Test(
        "Private construction/evaluation failures cannot replace a serving fence or refund a retained array"
    )
    func failedPrivateGraphOwnership() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        write(original, start: 0, count: chunkSize)
        let row = try #require(original[0] as? PagedSequenceKV)
        let group = row.pool.group(row.groupKey)
        let fence = ObjectIdentifier(group.writeFence)
        let before = fixture.admission.bytesReserved
        weak var failedConstruction: MLXArray?
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            try CBv2HistoricalWindow(
                row: row, position: chunkSize, admission: fixture.admission,
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
        var owner: CBv2HistoricalWindow? = try .init(
            row: row, position: chunkSize, admission: fixture.admission,
            evaluate: { array in
                failedEvaluation = array
                evaluations += 1
                try withError { eval(array) }
                throw CBv2CompleteCheckpointError.allocationFailed
            })
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            try owner?.finishEvaluation()
        }
        #expect(ObjectIdentifier(group.writeFence) == fence)
        #expect(failedEvaluation != nil && fixture.admission.bytesReserved > before)
        owner = nil
        #expect(evaluations == 1, "retirement never retries a failed graph")
        #expect(failedEvaluation == nil && fixture.admission.bytesReserved == before)
        write(original, start: chunkSize, count: 16)
        #expect(
            values(row).count == 17 * 64,
            "typed optional failure leaves serving target fence intact")
    }

    @Test(
        "An interior position captures while its window is resident and is refused once the ring moved on"
    )
    func interiorWindowResidency() throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
        let row = try #require(original[0] as? PagedSequenceKV)
        // The fixture ring holds exactly one chunk (maxPrefillChunk == chunkSize).
        #expect(row.ringPages! * row.pool.config.pageSize == chunkSize)
        write(original, start: 0, count: chunkSize)
        write(original, start: chunkSize, count: chunkSize)
        #expect(row.absoluteOffset == 2 * chunkSize && row.oldestValidPosition == chunkSize)
        // [c + 17 - 17, c + 17) is still resident; [c - 17, c) is not.
        let resident = try CBv2HistoricalWindow(
            row: row, position: chunkSize + 17, admission: fixture.admission)
        try resident.finishEvaluation()
        #expect(resident.start == chunkSize && resident.position == chunkSize + 17)
        let bytes = 17 * 64 * row.groupKey.dtype.size
        let keys = try resident.read(values: false, byteOffset: 0, maximumBytes: bytes)
        let expected = MLXArray(
            (chunkSize ..< chunkSize + 17).flatMap { Array(repeating: Float($0), count: 64) },
            [17, 64]
        ).asType(row.groupKey.dtype)
        #expect(
            keys == expected.asData(),
            "an interior window copies the exact rows the frontier left behind in the ring")
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try CBv2HistoricalWindow(row: row, position: chunkSize, admission: fixture.admission)
        }
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try CBv2HistoricalWindow(
                row: row, position: 2 * chunkSize + 1, admission: fixture.admission)
        }
        let capture = CBv2CompleteCheckpointCapture(
            codec: fixture.codec, store: CompleteCheckpointFixtureStore())
        #expect(
            try capture.prepareHistorical(
                position: chunkSize, chunkSize: chunkSize, state: original) == nil,
            "an evicted interior window is skipped, never a stale copy")
        capture.close()
    }

    /// Real catalog window geometry, one owning sliding layer per model. The
    /// per-checkpoint charge is that figure times the model's owner count; it
    /// is what the per-donor budget and the slot-wide cap are sized against.
    /// K/V storage type is what the loaded model's projections and RoPE
    /// produce, not the weight type: the live gpt-oss-20b manifest reports
    /// float32 for 23 of its 24 layers (one sliding owner is bfloat16);
    /// gemma-4-26b stores 16-bit K/V.
    @Test(
        "Per-checkpoint window reservation for the historical catalog models",
        arguments: [
            ("gpt-oss-20b", 128, 8, 64, 64, 12, 4), ("gemma-4-26b", 1024, 8, 256, 16, 25, 2),
        ])
    func catalogWindowReservation(
        model: String, window: Int, kvHeads: Int, headDim: Int,
        queryHeads: Int, owners: Int, elementBytes: Int
    ) throws {
        let kinds = [
            CBv2LayerKind(
                attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads,
                queryHeads: queryHeads),
            CBv2LayerKind(
                attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads),
        ]
        let dtype: DType = elementBytes == 4 ? .float32 : .bfloat16
        let stride = CBv2RecurrentCheckpointGeometry.historicalCheckpointStrideTokens
        let config = PagedKVPoolConfig(
            capacityBytes: 256 << 20, maxPrefillChunk: 2 * stride,
            segmentSizeBytes: 8 << 20, layerDTypes: Array(repeating: dtype, count: kinds.count))
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 256 << 20,
            config: .init(
                watermarkFraction: 0, elementBytes: dtype.size,
                layerElementBytes: Array(repeating: dtype.size, count: kinds.count)),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        try admission.reserve(id: .init(1), additionalTokens: 3 * stride)
        var state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 2 * stride + 1, maxLength: 3 * stride)
        defer {
            backend.release(state)
            state.removeAll()
            admission.releaseAll(id: .init(1))
        }
        for entry in state {
            guard let row = entry as? PagedSequenceKV else { continue }
            let array = MLXArray.zeros([kvHeads, 2 * stride, headDim], dtype: dtype)
            row.write(keys: array, values: array)
        }
        let row = try #require(state[0] as? PagedSequenceKV)
        #expect(
            row.ringPages == 2 * stride / row.pool.config.pageSize,
            "the ring holds max(window + speculative span, maxPrefillChunk) tokens")
        // Interior residency at production geometry: after the solo stripe
        // [0, 2048) both models can capture 1,024; after a company chunk
        // shifted the stripe to [512, 2560) only gpt-oss (W = 128) still
        // holds [1024 - W, 1024), gemma-4 (W = 1,024) has lost it.
        _ = try CBv2HistoricalWindow.reservationBytes(row: row, position: stride)
        for entry in state {
            guard let row = entry as? PagedSequenceKV else { continue }
            row.write(
                keys: MLXArray.zeros([kvHeads, stride / 2, headDim], dtype: dtype),
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
        let retained = CBv2CheckpointRetention.maximumRetained
        print(
            "[historical-window-reservation] model=\(model) owners=\(owners) window=\(window) "
                + "dtype=\(dtype) logicalPerLayer=\(logical) reservedPerLayer=\(perLayer) "
                + "perCheckpoint=\(perCheckpoint) staged\(retained)=\(retained * perCheckpoint)")
        #expect(perLayer >= logical && perLayer < 2 * logical + (1 << 20))
        if model == "gpt-oss-20b" {
            #expect(logical == 524_288, "float32 K/V: 2 x 8 heads x 128 tokens x 64 x 4 bytes")
            #expect(
                perCheckpoint > 6 << 20 && perCheckpoint < 12 << 20,
                "gpt-oss stages ~7 MB per checkpoint; three are ~22 MB")
        } else {
            #expect(
                perCheckpoint > 128 << 20 && perCheckpoint < 512 << 20,
                "gemma-4 stages ~200 MB per checkpoint; the byte budget, not K, bounds it")
        }
    }

    @Test(
        "Custom copy stream survives its task-local scope through failure, abandonment or successful close",
        arguments: ["failure", "abandonment", "success"])
    func customStreamRetirement(outcome: String) throws {
        let fixture = try fixture()
        var original = try donor(fixture)
        defer {
            fixture.backend.release(original)
            original.removeAll()
            fixture.admission.releaseAll(id: .init(9001))
        }
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
            owner = try CBv2HistoricalWindow(
                row: row, position: chunkSize, admission: fixture.admission,
                afterConstruction: { output = $0 },
                evaluate: { array in
                    if outcome == "success" {
                        try withError { eval(array) }
                    } else {
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
            #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
                try owner?.finishEvaluation()
            }
            #expect(drains == 1 && output != nil && fixture.admission.bytesReserved > before)
        } else if outcome == "success" {
            try owner?.finishEvaluation()
            #expect(
                drains == 0 && output != nil && fixture.admission.bytesReserved > before,
                "successful evaluation keeps its charge until retirement drains completion handlers"
            )
        }
        owner = nil
        #expect(drains == 1 && output == nil && fixture.admission.bytesReserved == before)
        #expect(ObjectIdentifier(row.pool.group(row.groupKey).writeFence) == fence)
        write(original, start: chunkSize, count: 16)
        #expect(values(row).count == 17 * 64)
    }

}
