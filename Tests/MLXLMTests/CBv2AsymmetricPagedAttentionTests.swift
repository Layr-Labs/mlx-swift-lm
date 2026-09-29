import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

private final class AsymPagedEngineModel: CBv2SteppableModel {
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let b = tokens.dim(0)
        let n = tokens.dim(1)
        var hidden = tokens.asType(.float32).reshaped([b, 1, n, 1]) / Float(32)
        for cache in caches {
            let q = MLXArray.zeros([b, 2, n, cache.kind.headDim], dtype: .float32)
            let k = broadcast(hidden, to: [b, 1, n, cache.kind.headDim])
            let v = broadcast(hidden, to: [b, 1, n, cache.kind.valueHeadDim])
            let output = cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
            hidden = mean(output, axes: [1, 3], keepDims: true)
        }
        let target = MLX.round(hidden.reshaped([b, n, 1]) * Float(128)).asType(.int32) % 31
        return MLX.where(MLXArray(Int32(0) ..< Int32(32)) .== target, Float(10), Float(-10))
    }
}

private func asymPagedCollect(_ stream: AsyncStream<CBv2Event>) async -> ([Int], CBv2FinishReason?)
{
    let task = Task { () -> ([Int], CBv2FinishReason?) in
        var tokens: [Int] = []
        for await event in stream {
            switch event {
            case .delta(_, let ids, _): tokens.append(contentsOf: ids)
            case .finished(let reason, _): return (tokens, reason)
            }
        }
        return (tokens, nil)
    }
    let watchdog = Task {
        try? await Task.sleep(nanoseconds: 20_000_000_000)
        task.cancel()
    }
    let result = await task.value
    watchdog.cancel()
    return result
}

final class CBv2AsymmetricPagedAttentionTests: XCTestCase {
    private func engine(paged: Bool) throws -> (EngineV2, CBv2KVBackend) {
        let kinds = [asymPagedKind(window: 17), asymPagedKind()]
        let backend: CBv2KVBackend
        let bank: CBv2LayerCacheBank
        if paged {
            let limits = try CBv2PagedGatheredAttentionLimits(
                maximumBatchSize: 2, maximumQueryTokens: 16,
                maximumContextTokens: 128, maximumInFlightGraphs: 2, maximumScratchBytes: 512 << 20)
            let value = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: 512 << 20, dtype: .float32,
                    maxPrefillChunk: 16, nominalMaxSequenceLength: 128, maxBufferLength: 4 << 20,
                    segmentSizeBytes: 128 << 10, layerDTypes: [.float32, .float32],
                    gatheredAttention: limits))
            backend = value
            bank = CBv2LayerCacheBank(caches: value.makeLayerCaches())
        } else {
            backend = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 512 << 20, kvDType: .float32))
            bank = CBv2LayerCacheBank(layerKinds: kinds)
        }
        return (
            EngineV2(
                model: AsymPagedEngineModel(), layerKinds: kinds, backend: backend,
                cacheProvider: bank, sampler: CBv2DefaultSampler(fallbackSeed: 0),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 16, prefillChunkSize: 16,
                    maxWaiting: 4, enablePrefixCache: false),
                admissionConfig: .init(watermarkFraction: 0, elementBytes: 4)), backend
        )
    }

    func testRealEngineSegmentedGatheredTokensMatchOrdinaryAndRetirePages() async throws {
        let request = CBv2Request(
            id: .init(1), promptTokens: (0 ..< 41).map { ($0 * 7) % 29 },
            sampling: .init(temperature: 0, seed: 0), maxTokens: 5)
        let (cold, _) = try engine(paged: false)
        let (paged, backend) = try engine(paged: true)
        let expected = await asymPagedCollect(try cold.submit(request))
        let actual = await asymPagedCollect(try paged.submit(request))
        XCTAssertEqual(expected.1, .length)
        XCTAssertEqual(actual.1, .length)
        XCTAssertEqual(actual.0, expected.0)
        await cold.shutdown()
        await paged.shutdown()
        XCTAssertEqual(backend.bytesReserved, 0)
        let pool = try XCTUnwrap((backend as? PagedKVBackend)?.pool)
        XCTAssertEqual(pool.bytesMaterialized, 0)
        XCTAssertEqual(paged.admissionForTesting.bytesReserved, pool.gatheredAttentionScratchBound)
    }

    func testRealEngineCancellationDrainsPageWritesAndKeepsOnlyPoolScratch() async throws {
        let (value, backend) = try engine(paged: true)
        value.loopForTesting.onEngineQueueSync {
            value.loopForTesting.suspendStepExecutionAtCountForTesting = 1
        }
        let stream = try value.submit(
            .init(id: .init(2), promptTokens: Array(repeating: 1, count: 64), maxTokens: 8))
        let deadline = Date().addingTimeInterval(5)
        while value.stepCount < 1 && Date() < deadline {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(value.stepCount, 1)
        value.cancel(.init(2))
        value.loopForTesting.onEngineQueueSync {
            value.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        let result = await asymPagedCollect(stream)
        XCTAssertEqual(result.1, .cancelled)
        await value.shutdown()
        XCTAssertEqual(backend.bytesReserved, 0)
        let pool = try XCTUnwrap((backend as? PagedKVBackend)?.pool)
        XCTAssertEqual(pool.bytesMaterialized, 0)
        XCTAssertEqual(value.admissionForTesting.bytesReserved, pool.gatheredAttentionScratchBound)
    }
    private func scalarOracle(
        query: [Float], start: Int, count: Int, keyWidth: Int, valueWidth: Int,
        dtype: DType, window: Int?, scale: Float, sinks: [Float]
    ) -> [Float] {
        var result = [Float](repeating: 0, count: 2 * count * valueWidth)
        for h in 0 ..< 2 {
            for t in 0 ..< count {
                let end = start + t + 1
                let begin = window.map { max(0, end - $0) } ?? 0
                var scores: [Double] = []
                for token in begin ..< end {
                    var dot = Double(0)
                    for d in 0 ..< keyWidth {
                        let k = asymPagedRounded(Float(token) / 32 + Float(d) / 1024, dtype: dtype)
                        dot += Double(query[(h * count + t) * keyWidth + d]) * Double(k)
                    }
                    scores.append(dot * Double(scale))
                }
                let sink = Double(asymPagedRounded(sinks[h], dtype: dtype))
                let maximum = max(scores.max()!, sink)
                let weights = scores.map { exp($0 - maximum) }
                let denominator = weights.reduce(0, +) + exp(sink - maximum)
                for d in 0 ..< valueWidth {
                    var sum = Double(0)
                    for (i, token) in (begin ..< end).enumerated() {
                        let v = asymPagedRounded(
                            -4 + Float(token) / 32 + Float(d) / 1024, dtype: dtype)
                        sum += weights[i] * Double(v)
                    }
                    result[(h * count + t) * valueWidth + d] = Float(sum / denominator)
                }
            }
        }
        return result
    }

    func testNativeGatheredAttentionMatchesIndependentScalarAndOrdinaryCache() throws {
        for (dk, dv) in [(192, 128), (128, 192)] {
            for dtype: DType in [.float16, .bfloat16, .float32] {
                for window: Int? in [nil, 17] {
                    let kind = asymPagedKind(dk, dv, window: window, sinks: true)
                    let f = try AsymPagedFixture([kind], dtype: dtype)
                    let rows = try f.rows()
                    let cache = f.backend.makeLayerCaches()[0]
                    cache.setRows(rows.compactMap { $0 })
                    XCTAssertEqual(cache.nativeAttentionPath, "paged_native_gathered")
                    let ordinary: CBv2SequenceKV =
                        window.map {
                            CBv2WindowedSequenceKV(
                                window: $0, kvHeads: 1, headDim: dk, valueHeadDim: dv)
                                as CBv2SequenceKV
                        }
                        ?? CBv2FullSequenceKV(
                            promptLength: 0, maxLength: 128, kvHeads: 1, headDim: dk,
                            valueHeadDim: dv)
                    let sinkValues: [Float] = [0.125, -0.25]
                    let sinks = MLXArray(sinkValues).asType(dtype)
                    var start = 0
                    for count in [16, 1, 16, 1] {
                        let q =
                            (asymPagedTensor(
                                heads: 2, start: start, count: count, width: dk, dtype: dtype,
                                bias: 0.25) * Float(0.125)).asType(dtype)
                        let k = asymPagedTensor(start: start, count: count, width: dk, dtype: dtype)
                        let v = asymPagedTensor(
                            start: start, count: count, width: dv, dtype: dtype, bias: -4)
                        let output = cache.updateAndAttend(
                            queries: q, keys: k, values: v, scale: 0.03125, sinks: sinks)
                        let direct = CBv2AttentionV1.updateAndAttend(
                            rows: [ordinary], kind: kind,
                            queries: q, keys: k, values: v, scale: 0.03125, sinks: sinks)
                        eval([output, direct] + cache.innerState())
                        try f.backend.pool.writeValidation.check()
                        XCTAssertEqual(output.shape, [1, 2, count, dv])
                        XCTAssertEqual(output.dtype, dtype)
                        let expected = scalarOracle(
                            query: q.asType(.float32).asArray(Float.self), start: start,
                            count: count,
                            keyWidth: dk, valueWidth: dv, dtype: dtype, window: window,
                            scale: 0.03125, sinks: sinkValues)
                        let actual = output.asType(.float32).asArray(Float.self)
                        let contiguous = direct.asType(.float32).asArray(Float.self)
                        let tolerance: Float =
                            dtype == .float32 ? 0.0002 : (dtype == .float16 ? 0.008 : 0.06)
                        XCTAssertEqual(actual.count, expected.count)
                        for i in actual.indices {
                            XCTAssertEqual(actual[i], expected[i], accuracy: tolerance)
                            XCTAssertEqual(actual[i], contiguous[i], accuracy: tolerance)
                        }
                        start += count
                    }
                }
            }
        }
    }

    func testPackedRowsAndBorrowersAreIndependent() throws {
        let kinds = [asymPagedKind(window: 17), asymPagedKind(window: 17, shares: 0)]
        let f = try AsymPagedFixture(kinds)
        let a = try f.rows()
        let b = try f.rows()
        let caches = f.backend.makeLayerCaches()
        let source = caches[0]
        let borrower = caches[1]
        source.setRows([a[0]!, b[0]!])
        borrower.setRows([])
        let q = MLXArray.ones([2, 2, 16, 192], dtype: .float32) * Float(0.01)
        let k0 = asymPagedTensor(start: 0, count: 16, width: 192, dtype: .float32)
        let k = concatenated([k0, k0 + Float(3)], axis: 0)
        let v0 = asymPagedTensor(start: 0, count: 16, width: 128, dtype: .float32, bias: -4)
        let v = concatenated([v0, v0 + Float(9)], axis: 0)
        let own = source.updateAndAttend(queries: q, keys: k, values: v, scale: 0.1, sinks: nil)
        let borrowed = borrower.attendBorrowing(source: source, queries: q, scale: 0.1, sinks: nil)
        eval([own, borrowed] + source.innerState() + borrower.innerState())
        XCTAssertEqual(own.asArray(Float.self), borrowed.asArray(Float.self))
        XCTAssertGreaterThan(
            borrowed[1].mean().item(Float.self) - borrowed[0].mean().item(Float.self), 8)
        XCTAssertNil(a[1])
        XCTAssertNil(b[1])
        XCTAssertEqual(a[0]!.absoluteOffset, 16)
        source.setRows([b[0]!])
        let q1 = MLXArray.ones([1, 2, 1, 192]) * Float(0.01)
        let next = source.updateAndAttend(
            queries: q1,
            keys: asymPagedTensor(start: 16, count: 1, width: 192, dtype: .float32, bias: 3),
            values: asymPagedTensor(start: 16, count: 1, width: 128, dtype: .float32, bias: 5),
            scale: 0.1, sinks: nil)
        eval([next] + source.innerState())
        XCTAssertEqual(a[0]!.absoluteOffset, 16)
        XCTAssertEqual(b[0]!.absoluteOffset, 17)
    }

    /// Both owning layers intentionally share a geometry group, but contain
    /// different values. Standalone caches retain chunk views by default, so
    /// the wrong-source prefill would otherwise read valid but foreign views.
    private func assertWrongOwnerRefusedBeforeRead(queryCount: Int) throws {
        let kinds = [asymPagedKind(), asymPagedKind(), asymPagedKind(shares: 0)]
        let f = try AsymPagedFixture(kinds, batch: 1, queries: 16)
        let states = try f.rows()
        let rows = try (0 ..< 2).map { try XCTUnwrap(states[$0] as? PagedSequenceKV) }
        let caches = f.backend.makeLayerCaches()
        let expectedSource = caches[0]
        let wrongSource = caches[1]
        let borrower = caches[2]
        expectedSource.setRows([rows[0]])
        wrongSource.setRows([rows[1]])
        borrower.setRows([])
        XCTAssertEqual(rows[0].groupKey, rows[1].groupKey)
        XCTAssertEqual(borrower.kind.sharesKVWithLayer, 0)
        XCTAssertNotEqual(wrongSource.layerIndex, borrower.kind.sharesKVWithLayer)
        let q = MLXArray.zeros([1, 2, queryCount, 192], dtype: .float32)
        let k = MLXArray.zeros([1, 1, queryCount, 192], dtype: .float32)
        let v0 = MLXArray.ones([1, 1, queryCount, 128], dtype: .float32) * Float(3)
        let v1 = MLXArray.ones([1, 1, queryCount, 128], dtype: .float32) * Float(9)
        let own0 = expectedSource.updateAndAttend(
            queries: q, keys: k, values: v0, scale: 0.125, sinks: nil)
        let own1 = wrongSource.updateAndAttend(
            queries: q, keys: k, values: v1, scale: 0.125, sinks: nil)
        let correct = borrower.attendBorrowing(
            source: expectedSource, queries: q, scale: 0.125, sinks: nil)
        eval([own0, own1, correct] + expectedSource.innerState() + wrongSource.innerState())
        StreamOrDevice.default.stream.synchronize()
        try f.backend.pool.writeValidation.check()
        XCTAssertEqual(correct.shape, [1, 2, queryCount, 128])
        for value in correct.asArray(Float.self) { XCTAssertEqual(value, 3, accuracy: 0.00001) }
        for value in own1.asArray(Float.self) { XCTAssertEqual(value, 9, accuracy: 0.00001) }

        let group = f.backend.pool.group(rows[0].groupKey)
        let fence = group.writeFence
        let offsets = rows.map(\.absoluteOffset)
        let tables = rows.map(\.table)
        let versions = rows.map(\.tableVersion)
        let bytes = rows.map(\.byteCount)
        let reserved = f.backend.bytesReserved
        let wired = f.backend.bytesWired
        let backingBytes = Dictionary(
            uniqueKeysWithValues: group.segments.map {
                ($0.key, $0.value.storage.asData(access: .copy).data)
            })
        let refused = borrower.attendBorrowing(
            source: wrongSource, queries: q, scale: 0.125, sinks: nil)
        XCTAssertThrowsError(try f.backend.pool.writeValidation.check())
        XCTAssertEqual(refused.shape, [1, 2, queryCount, 128])
        XCTAssertTrue(
            group.writeFence === fence, "wrong owner must be rejected before a gather/read fence")
        XCTAssertEqual(rows.map(\.absoluteOffset), offsets)
        XCTAssertEqual(rows.map(\.table), tables)
        XCTAssertEqual(rows.map(\.tableVersion), versions)
        XCTAssertEqual(rows.map(\.byteCount), bytes)
        XCTAssertEqual(f.backend.bytesReserved, reserved)
        XCTAssertEqual(f.backend.bytesWired, wired)
        XCTAssertEqual(Set(group.segments.keys), Set(backingBytes.keys))
        for (index, segment) in group.segments {
            XCTAssertEqual(segment.storage.asData(access: .copy).data, backingBytes[index])
        }
        // Never evaluate the refused output or clear its latch to keep serving.
        // Fixture retirement drains the already completed valid work only.
    }

    func testWrongLayerWithIdenticalGeometryRefusesBeforeDecodeGather() throws {
        try assertWrongOwnerRefusedBeforeRead(queryCount: 1)
    }

    func testWrongLayerWithIdenticalGeometryRefusesBeforeRetainedPrefillRead() throws {
        try assertWrongOwnerRefusedBeforeRead(queryCount: 16)
    }

    func testDTypeValueWidthQueryAndExecutionFaultsDoNotWrite() throws {
        for fault in ["dtype", "value", "query", "querydtype", "sink", "limit", "mtp"] {
            let dtype: DType = fault == "sink" ? .bfloat16 : .float32
            let f = try AsymPagedFixture(
                [asymPagedKind(sinks: fault == "sink")], dtype: dtype, queries: 16)
            let rows = try f.rows()
            let cache = f.backend.makeLayerCaches()[0]
            cache.setRows(rows.compactMap { $0 })
            let group = f.backend.pool.group(f.backend.pool.groupKey(forLayer: 0))
            let fence = group.writeFence
            let count = fault == "limit" ? 17 : 1
            let q = MLXArray.ones(
                [1, 2, count, fault == "query" ? 64 : 192],
                dtype: fault == "querydtype" ? .float16 : dtype)
            let k = MLXArray.ones([1, 1, count, 192], dtype: dtype)
            let v = MLXArray.ones(
                [1, 1, count, fault == "value" ? 192 : 128],
                dtype: fault == "dtype" ? .float16 : dtype)
            if fault == "mtp" { cache.mtpSerializesRectangularAttention = true }
            let output = cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: 0.1,
                sinks: fault == "sink" ? MLXArray.zeros([2], dtype: .float32) : nil)
            XCTAssertThrowsError(try f.backend.pool.writeValidation.check())
            XCTAssertEqual(output.shape, [1, 2, count, 128])
            XCTAssertEqual(rows[0]!.absoluteOffset, 0)
            XCTAssertEqual(rows[0]!.byteCount, 0)
            XCTAssertTrue(group.writeFence === fence)
        }
    }

    func testForeignDuplicateAndReleasedRowsRefuseWithoutMutation() throws {
        for scenario in ["foreign", "duplicate", "released"] {
            let f = try AsymPagedFixture([asymPagedKind()])
            let other = try AsymPagedFixture([asymPagedKind()])
            let rows = try f.rows()
            let foreign = try other.rows()
            let cache = f.backend.makeLayerCaches()[0]
            cache.setRows(rows.compactMap { $0 })
            let original = cache.rows.map(ObjectIdentifier.init)
            if scenario == "foreign" { cache.setRows(foreign.compactMap { $0 }) }
            if scenario == "duplicate" { cache.setRows([rows[0]!, rows[0]!]) }
            if scenario == "released" {
                f.backend.release(rows)
                cache.setRows(rows.compactMap { $0 })
            }
            XCTAssertThrowsError(try f.backend.pool.writeValidation.check())
            XCTAssertEqual(cache.rows.map(ObjectIdentifier.init), original)
            XCTAssertEqual(foreign[0]!.absoluteOffset, 0)
        }
    }

    func testNativeFullAndWindowRestorePreservesBoundaryAndContinues() throws {
        let kinds = [asymPagedKind(), asymPagedKind(window: 17)]
        let f = try AsymPagedFixture(kinds)
        let source = try f.rows()
        for start in stride(from: 0, to: 48, by: 16) {
            for row in source.compactMap({ $0 }) {
                _ = row.update(
                    keys: asymPagedTensor(start: start, count: 16, width: 192, dtype: .float32),
                    values: asymPagedTensor(
                        start: start, count: 16, width: 128, dtype: .float32, bias: -4))
            }
        }
        let snapshots = source.map { $0?.snapshot() }
        for s in snapshots.compactMap({ $0 }) { eval(s.keys, s.values) }
        let plan = CBv2PrefixReusePlan(
            backend: .pagedFP16, strategy: .frozenFullReplay, matchedBoundary: 48,
            replayStart: 48, replayTokens: 0, prefillTokensSaved: 48, restoredFullTokens: 48,
            capacityReservationTokens: 128, nominalFullKVBytesPerToken: 1280,
            fullKVBytesPerToken: 1280,
            additionalFullKVBytesPerToken: 0, initialAdditionalCapacityBytes: 0,
            fullCapacityTokensReserved: 128,
            stagedFullKVBytes: 48 * 1280, residentFullKVBytes: 48 * 1280)
        let restored = try f.backend.makeSequenceState(
            adopting: snapshots, plan: plan, layerKinds: kinds, maxLength: 128)
        f.owned.append(restored)
        for i in 0 ..< 2 {
            XCTAssertEqual(restored[i]!.absoluteOffset, 48)
            let current = restored[i]!.snapshot()
            eval(current.keys, current.values)
            XCTAssertEqual(
                current.keys.asData(access: .copy).data,
                snapshots[i]!.keys.asData(access: .copy).data)
            XCTAssertEqual(
                current.values.asData(access: .copy).data,
                snapshots[i]!.values.asData(access: .copy).data)
            let k = asymPagedTensor(start: 48, count: 1, width: 192, dtype: .float32)
            let v = asymPagedTensor(start: 48, count: 1, width: 128, dtype: .float32, bias: -4)
            let expected = source[i]!.update(keys: k, values: v)
            let actual = restored[i]!.update(keys: k, values: v)
            eval(expected.0, expected.1, actual.0, actual.1)
            XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
            XCTAssertEqual(actual.1.asArray(Float.self), expected.1.asArray(Float.self))
        }
        let before = f.backend.bytesReserved
        var wrong = snapshots
        wrong[1] = (snapshots[1]!.keys, MLXArray.zeros([1, 1, 17, 192]), 48)
        XCTAssertThrowsError(
            try f.backend.makeSequenceState(
                adopting: wrong, plan: plan, layerKinds: kinds, maxLength: 128))
        XCTAssertEqual(f.backend.bytesReserved, before)
    }

    func testUniformDefaultDispatchAndOutputRemainUnchanged() throws {
        let kind = asymPagedKind(64, 64)
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: .init(
                capacityBytes: 4 << 20,
                maxPrefillChunk: 16, segmentSizeBytes: 32 << 10))
        XCTAssertEqual(backend.pool.gatheredAttentionScratchBound, 0)
        let rows = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 32)
        defer {
            StreamOrDevice.default.stream.synchronize()
            backend.release(rows)
        }
        let cache = backend.makeLayerCaches()[0]
        cache.setRows(rows.compactMap { $0 })
        let q = MLXArray.ones([1, 2, 1, 64], dtype: .float16)
        let k = MLXArray.ones([1, 1, 1, 64], dtype: .float16)
        let output = cache.updateAndAttend(
            queries: q, keys: k, values: k * Float(3), scale: 0.125, sinks: nil)
        eval([output] + cache.innerState())
        XCTAssertEqual(output.shape, [1, 2, 1, 64])
        XCTAssertTrue(output.asType(.float32).asArray(Float.self).allSatisfy { abs($0 - 3) < 0.01 })
        XCTAssertNotEqual(cache.nativeAttentionPath, "paged_native_gathered")
    }

    func testFullFrozenReplayDoesNotOverwriteAdoptedAsymmetricValues() throws {
        let f = try AsymPagedFixture([asymPagedKind()])
        let source = try f.rows()
        _ = source[0]!.update(
            keys: asymPagedTensor(start: 0, count: 48, width: 192, dtype: .float32),
            values: asymPagedTensor(start: 0, count: 48, width: 128, dtype: .float32, bias: -4))
        let snapshot = source[0]!.snapshot()
        eval(snapshot.keys, snapshot.values)
        let plan = CBv2PrefixReusePlan(
            backend: .pagedFP16, strategy: .frozenFullReplay, matchedBoundary: 48,
            replayStart: 32, replayTokens: 16, prefillTokensSaved: 32, restoredFullTokens: 48,
            capacityReservationTokens: 128, nominalFullKVBytesPerToken: 1280,
            fullKVBytesPerToken: 1280,
            additionalFullKVBytesPerToken: 0, initialAdditionalCapacityBytes: 0,
            fullCapacityTokensReserved: 128,
            stagedFullKVBytes: 48 * 1280, residentFullKVBytes: 48 * 1280)
        let restored = try f.backend.makeSequenceState(
            adopting: [snapshot], plan: plan, layerKinds: f.kinds, maxLength: 128)
        f.owned.append(restored)
        XCTAssertEqual(restored[0]!.absoluteOffset, 32)
        _ = restored[0]!.update(
            keys: asymPagedTensor(start: 32, count: 16, width: 192, dtype: .float32, bias: 99),
            values: asymPagedTensor(start: 32, count: 16, width: 128, dtype: .float32, bias: 99))
        let final = restored[0]!.snapshot()
        eval(final.keys, final.values)
        XCTAssertEqual(
            final.values.asData(access: .copy).data, snapshot.values.asData(access: .copy).data)
    }
}
