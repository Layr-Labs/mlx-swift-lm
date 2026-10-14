import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Mixed packed/native attention", .serialized)
struct CBv2MixedQuantizedAttentionTests {
    private func setup(
        dim: Int, dtype: DType, window: Int? = nil, sharing: Bool = false,
        quantization: PagedKVQuantizationConfig = .init(), softcap: Float? = nil,
        bidirectional: Bool = false
    )
        throws -> (PagedKVBackend, [PagedSequenceKV], [PagedLayerCache])
    {
        var kinds = [
            CBv2LayerKind(
                attention: window.map { .slidingWindow($0) } ?? .full,
                hasSinks: true, headDim: dim, kvHeads: 1, queryHeads: 2)
        ]
        if sharing {
            kinds.append(
                .init(
                    attention: kinds[0].attention, sharesKVWithLayer: 0,
                    hasSinks: true, headDim: dim, kvHeads: 1, queryHeads: 2))
        }
        kinds[0].isBidirectional = bidirectional
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                pageSize: 16, capacityBytes: 16 << 20, dtype: dtype, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 1024, maxBufferLength: 16 << 20,
                segmentSizeBytes: 64 << 10, quantization: quantization))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 160, maxLength: 1024)
        let rows = state.compactMap { $0 as? PagedSequenceKV }
        var caches = backend.makeLayerCaches().map { $0 as! PagedLayerCache }
        if let softcap {
            caches[0] = PagedLayerCache(
                layerIndex: 0, kind: kinds[0], pool: backend.pool, attentionSoftcap: softcap)
        }
        caches[0].setRows([rows[0]])
        return (backend, rows, caches)
    }

    private func values(tokens: Int, dim: Int, heads: Int = 1, tag: Float, dtype: DType) -> MLXArray
    {
        let data = (0 ..< heads * tokens * dim).map { index -> Float in
            sin(Float(index + 3) * 0.027 + tag) * 0.7 + cos(Float(index % dim) * 0.043 + tag) * 0.2
        }
        return MLXArray(data, [1, heads, tokens, dim]).asType(dtype)
    }

    private func complete(_ pool: PagedKVPool, roots: [MLXArray]) throws {
        eval(roots)
        StreamOrDevice.default.stream.synchronize()
        try pool.finishQuantizedStorageStep()
    }

    private func reference(
        q: [Float], keys: [[Float]], values: [[Float]],
        dim: Int, queryCount: Int, queryStart: Int,
        window: Int? = nil, spans: [CBv2ImageSpan] = [],
        bidirectional: Bool = false, sinks: [Float], softcap: Float
    ) -> [Float] {
        var result = [Float](repeating: 0, count: 2 * queryCount * dim)
        for head in 0 ..< 2 {
            for query in 0 ..< queryCount {
                let pos = queryStart + query
                var low = window.map { max(0, pos - $0 + 1) } ?? 0
                var high =
                    bidirectional ? min(keys.count, window.map { pos + $0 } ?? keys.count) : pos + 1
                for span in spans where pos >= span.tokenOffset && pos < span.end {
                    low = min(low, span.tokenOffset)
                    high = max(high, span.end)
                }
                var logits: [Double] = []
                for token in low ..< high {
                    var dot: Float = 0
                    for d in 0 ..< dim {
                        dot += q[(head * queryCount + query) * dim + d] * keys[token][d]
                    }
                    let score = dot / Float(dim).squareRoot()
                    logits.append(Double(softcap * tanh(score / softcap)))
                }
                let maximum = max(Double(sinks[head]), logits.max()!)
                let weights = logits.map { exp($0 - maximum) }
                let denominator = weights.reduce(exp(Double(sinks[head]) - maximum), +)
                for (index, token) in (low ..< high).enumerated() {
                    for d in 0 ..< dim {
                        result[(head * queryCount + query) * dim + d] +=
                            Float(weights[index] / denominator) * values[token][d]
                    }
                }
            }
        }
        return result
    }

    @Test(arguments: [64, 128, 256, 512], [DType.float16, .bfloat16, .float32])
    func packedOldRowsAndOriginalNativeBandMatchScalarOracle(dim: Int, dtype: DType) throws {
        try verifyOracle(dim: dim, dtype: dtype, quantization: .init())
    }

    @Test(arguments: [4, 8])
    func key8ValueFallbackModesUseTheActualMixedShader(valueBits: Int) throws {
        try verifyOracle(
            dim: 256, dtype: .float32,
            quantization: .init(keyBits: 8, valueBits: valueBits))
    }

    @Test
    func softcapAndSinksApplyToTheCombinedPackedAndNativeWeights() throws {
        try verifyOracle(dim: 64, dtype: .float32, quantization: .init(), softcap: 0.75)
    }

    private func verifyOracle(
        dim: Int, dtype: DType, quantization: PagedKVQuantizationConfig,
        softcap: Float? = nil
    ) throws {
        let (backend, rows, caches) = try setup(
            dim: dim, dtype: dtype,
            quantization: quantization, softcap: softcap)
        defer { backend.release(rows.map { $0 as CBv2SequenceKV? }) }
        let row = rows[0]
        let cache = caches[0]
        cache.setRetainsChunkForBorrowers(false)
        let primeK = values(tokens: 160, dim: dim, tag: 0.3, dtype: dtype)
        let primeV = values(tokens: 160, dim: dim, tag: 0.9, dtype: dtype)
        row.write(keys: primeK[0], values: primeV[0])
        try complete(backend.pool, roots: cache.innerState())
        #expect(row.nativeRecentStart == 32)
        let newK = values(tokens: 3, dim: dim, tag: 1.3, dtype: dtype)
        let newV = values(tokens: 3, dim: dim, tag: 1.9, dtype: dtype)
        let q = values(tokens: 3, dim: dim, heads: 2, tag: 2.3, dtype: dtype)
        let sinks: [Float] = [0.2, -0.1]
        let output = cache.updateAndAttend(
            queries: q, keys: newK, values: newV,
            scale: 1 / Float(dim).squareRoot(), sinks: MLXArray(sinks))
        let originalK = concatenated([primeK, newK], axis: 2).asArray(Float.self)
        let originalV = concatenated([primeV, newV], axis: 2).asArray(Float.self)
        var refK: [[Float]] = []
        var refV: [[Float]] = []
        for token in 0 ..< 163 {
            let k = Array(originalK[token * dim ..< (token + 1) * dim])
            let v = Array(originalV[token * dim ..< (token + 1) * dim])
            refK.append(
                token < 32
                    ? try PagedKVQuantizationReference.roundTrip(
                        k, config: quantization, isKey: true) : k)
            refV.append(
                token < 32
                    ? try PagedKVQuantizationReference.roundTrip(
                        v, config: quantization, isKey: false) : v)
        }
        // This cache's layer softcap is nil; a large cap approximates the same
        // operator without hiding quantization in an original-input baseline.
        let expected = reference(
            q: q.asArray(Float.self), keys: refK, values: refV,
            dim: dim, queryCount: 3, queryStart: 160, sinks: sinks, softcap: softcap ?? 1_000_000)
        let actual = output.asArray(Float.self)
        let tolerance: Float = dtype == .bfloat16 ? 0.005 : (dtype == .float16 ? 0.001 : 0.0002)
        #expect(zip(actual, expected).map { abs($0 - $1) }.max()! < tolerance)
        #expect(!backend.pool.writeValidation.isFaulted)
        #expect(
            PagedQuantizedAttention.partThreadgroupBytes(
                headDim: dim, gqa: 2,
                simdgroups: PagedQuantizedAttention.simdgroups(headDim: dim, gqa: 2)!) <= 32768)
        try complete(backend.pool, roots: [output] + cache.innerState())
    }

    @Test
    func packedBatchRowsStayIsolatedAtDifferentNativeFrontiers() throws {
        let (backend, rows, caches) = try setup(dim: 64, dtype: .float32)
        let kind = caches[0].kind
        let extra = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 192, maxLength: 1024)
        let second = try #require(extra[0] as? PagedSequenceKV)
        defer {
            backend.release(rows.map { $0 as CBv2SequenceKV? })
            backend.release(extra)
        }
        let allRows = [rows[0], second]
        caches[0].setRows(allRows)
        var primeKeys: [MLXArray] = []
        var primeValues: [MLXArray] = []
        for (index, row) in allRows.enumerated() {
            let count = index == 0 ? 160 : 192
            let k = values(tokens: count, dim: 64, tag: Float(index) + 0.3, dtype: .float32)
            let v = values(tokens: count, dim: 64, tag: Float(index) + 0.9, dtype: .float32)
            row.write(keys: k[0], values: v[0])
            primeKeys.append(k)
            primeValues.append(v)
        }
        try complete(backend.pool, roots: caches[0].innerState())
        caches[0].setRows(allRows)
        let queries = (0 ..< 2).map {
            values(tokens: 3, dim: 64, heads: 2, tag: Float($0) + 2.3, dtype: .float32)
        }
        let newKeys = (0 ..< 2).map {
            values(tokens: 3, dim: 64, tag: Float($0) + 1.3, dtype: .float32)
        }
        let newValues = (0 ..< 2).map {
            values(tokens: 3, dim: 64, tag: Float($0) + 1.9, dtype: .float32)
        }
        let output = caches[0].updateAndAttend(
            queries: concatenated(queries, axis: 0),
            keys: concatenated(newKeys, axis: 0), values: concatenated(newValues, axis: 0),
            scale: 0.125, sinks: nil)
        for index in 0 ..< 2 {
            let count = index == 0 ? 160 : 192
            let keys = concatenated([primeKeys[index], newKeys[index]], axis: 2).asArray(Float.self)
            let vals = concatenated([primeValues[index], newValues[index]], axis: 2).asArray(
                Float.self)
            var k: [[Float]] = []
            var v: [[Float]] = []
            for token in 0 ..< count + 3 {
                let kr = Array(keys[token * 64 ..< (token + 1) * 64])
                let vr = Array(vals[token * 64 ..< (token + 1) * 64])
                k.append(
                    token < count - 128
                        ? try PagedKVQuantizationReference.roundTrip(
                            kr, config: .init(), isKey: true) : kr)
                v.append(
                    token < count - 128
                        ? try PagedKVQuantizationReference.roundTrip(
                            vr, config: .init(), isKey: false) : vr)
            }
            let expected = reference(
                q: queries[index].asArray(Float.self), keys: k, values: v,
                dim: 64, queryCount: 3, queryStart: count, sinks: [-1e30, -1e30], softcap: 1_000_000
            )
            #expect(
                zip(output[index].asArray(Float.self), expected).map { abs($0 - $1) }.max()!
                    < 0.0002)
        }
        #expect(caches[0].positionOffsets.asArray(Int32.self) == [163, 195])
        try complete(backend.pool, roots: [output] + caches[0].innerState())
    }

    @Test
    func packedSnapshotPreservesEveryCodeAndDoesNotPublishItsPrivateFence() throws {
        let (backend, rows, caches) = try setup(
            dim: 128, dtype: .float32,
            quantization: .init(keyBits: 8, valueBits: 4))
        defer { backend.release(rows.map { $0 as CBv2SequenceKV? }) }
        let row = rows[0]
        let group = backend.pool.group(row.groupKey)
        row.write(
            keys: values(tokens: 160, dim: 128, tag: 0.3, dtype: .float32)[0],
            values: values(tokens: 160, dim: 128, tag: 0.9, dtype: .float32)[0])
        eval(caches[0].innerState())
        let prior = ObjectIdentifier(group.writeFence)
        let packed = PagedQuantizedTransfers.gatherPacked(
            group: group,
            pages: row.table, firstSlot: 0, count: 160, publishReadFence: false)
        #expect(ObjectIdentifier(group.writeFence) == prior)
        let layout = try row.groupKey.quantization!.rowLayout(headDim: 128)
        let keyBytes = packed.keys.asArray(UInt8.self)
        let valueBytes = packed.values.asArray(UInt8.self)
        var expectedK: [UInt8] = []
        var expectedV: [UInt8] = []
        for token in 0 ..< 160 {
            let page = row.table[token / group.pageSize]
            let map = group.segmentLayout!
            let segment = group.segments[map.segmentIndex(page: page)]!
            let bytes = segment.storage.asArray(UInt8.self)
            let slot = map.localPage(page) * group.pageSize + token % group.pageSize
            expectedK.append(
                contentsOf: bytes[slot * layout.keyRowBytes ..< (slot + 1) * layout.keyRowBytes])
            let start = segment.valueOffset + slot * layout.valueRowBytes
            expectedV.append(contentsOf: bytes[start ..< start + layout.valueRowBytes])
        }
        #expect(keyBytes == expectedK && valueBytes == expectedV)
        #expect(packed.keys.shape == [1, 1, 160, layout.keyRowBytes])
        #expect(packed.values.shape == [1, 1, 160, layout.valueRowBytes])
        try complete(backend.pool, roots: [packed.keys, packed.values] + caches[0].innerState())
    }

    @Test
    func largeWindowChunkAndSharedBorrowerRetainThePreWriteView() throws {
        let (backend, rows, caches) = try setup(
            dim: 64, dtype: .float32, window: 256, sharing: true)
        defer { backend.release(rows.map { $0 as CBv2SequenceKV? }) }
        let primeK = values(tokens: 160, dim: 64, tag: 0.2, dtype: .float32)
        let primeV = values(tokens: 160, dim: 64, tag: 0.8, dtype: .float32)
        rows[0].write(keys: primeK[0], values: primeV[0])
        try complete(backend.pool, roots: caches[0].innerState())
        let q = values(tokens: 384, dim: 64, heads: 2, tag: 1.1, dtype: .float32)
        let k = values(tokens: 384, dim: 64, tag: 1.7, dtype: .float32)
        let v = values(tokens: 384, dim: 64, tag: 2.1, dtype: .float32)
        let owner = caches[0].updateAndAttend(
            queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
        let borrower = caches[1].attendBorrowing(
            source: caches[0], queries: q, scale: 0.125, sinks: nil)
        #expect(abs(owner - borrower).max().item(Float.self) < 1e-6)
        #expect(owner.shape == [1, 2, 384, 64])
        #expect(!backend.pool.writeValidation.isFaulted)
        try complete(backend.pool, roots: [owner, borrower] + caches[0].innerState())
    }

    @Test
    func bidirectionalWindowLimitsFutureKeysEvenForAChunkLargerThanTheRing() throws {
        let (backend, rows, caches) = try setup(
            dim: 64, dtype: .float32,
            window: 256, bidirectional: true)
        defer { backend.release(rows.map { $0 as CBv2SequenceKV? }) }
        let primeK = values(tokens: 256, dim: 64, tag: 0.2, dtype: .float32)
        let primeV = values(tokens: 256, dim: 64, tag: 0.8, dtype: .float32)
        rows[0].write(keys: primeK[0], values: primeV[0])
        try complete(backend.pool, roots: caches[0].innerState())
        let q = MLXArray.zeros([1, 2, 300, 64], dtype: .float32)
        let k = values(tokens: 300, dim: 64, tag: 1.7, dtype: .float32)
        let v = values(tokens: 300, dim: 64, tag: 2.1, dtype: .float32)
        let output = caches[0].updateAndAttend(
            queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
        let allK = concatenated([primeK, k], axis: 2).asArray(Float.self)
        let allV = concatenated([primeV, v], axis: 2).asArray(Float.self)
        var refK: [[Float]] = []
        var refV: [[Float]] = []
        for token in 0 ..< 556 {
            let kr = Array(allK[token * 64 ..< (token + 1) * 64])
            let vr = Array(allV[token * 64 ..< (token + 1) * 64])
            refK.append(
                token < 128
                    ? try PagedKVQuantizationReference.roundTrip(kr, config: .init(), isKey: true)
                    : kr)
            refV.append(
                token < 128
                    ? try PagedKVQuantizationReference.roundTrip(vr, config: .init(), isKey: false)
                    : vr)
        }
        let expected = reference(
            q: q.asArray(Float.self), keys: refK, values: refV,
            dim: 64, queryCount: 300, queryStart: 256, window: 256, bidirectional: true,
            sinks: [-1e30, -1e30], softcap: 1_000_000)
        #expect(zip(output.asArray(Float.self), expected).map { abs($0 - $1) }.max()! < 0.0002)
        #expect(!backend.pool.writeValidation.isFaulted)
        try complete(backend.pool, roots: [output] + caches[0].innerState())
    }

    @Test
    func visionSpanExtendsCausalBoundsInsideTheMixedNativeBand() throws {
        let (backend, rows, caches) = try setup(dim: 64, dtype: .float32)
        defer { backend.release(rows.map { $0 as CBv2SequenceKV? }) }
        let primeK = values(tokens: 160, dim: 64, tag: 0.2, dtype: .float32)
        let primeV = values(tokens: 160, dim: 64, tag: 0.8, dtype: .float32)
        rows[0].write(keys: primeK[0], values: primeV[0])
        try complete(backend.pool, roots: caches[0].innerState())
        let q = values(tokens: 12, dim: 64, heads: 2, tag: 1.1, dtype: .float32)
        let k = values(tokens: 12, dim: 64, tag: 1.7, dtype: .float32)
        let v = values(tokens: 12, dim: 64, tag: 2.1, dtype: .float32)
        let spans = [CBv2ImageSpan(tokenOffset: 150, length: 22)]
        caches[0].bindSpanContext(.init(chunkEnd: 172, blocks: spans))
        let output = caches[0].updateAndAttend(
            queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
        caches[0].bindSpanContext(nil)
        let allK = concatenated([primeK, k], axis: 2).asArray(Float.self)
        let allV = concatenated([primeV, v], axis: 2).asArray(Float.self)
        var refK: [[Float]] = []
        var refV: [[Float]] = []
        for token in 0 ..< 172 {
            let kr = Array(allK[token * 64 ..< (token + 1) * 64])
            let vr = Array(allV[token * 64 ..< (token + 1) * 64])
            refK.append(
                token < 32
                    ? try PagedKVQuantizationReference.roundTrip(kr, config: .init(), isKey: true)
                    : kr)
            refV.append(
                token < 32
                    ? try PagedKVQuantizationReference.roundTrip(vr, config: .init(), isKey: false)
                    : vr)
        }
        let expected = reference(
            q: q.asArray(Float.self), keys: refK, values: refV,
            dim: 64, queryCount: 12, queryStart: 160, spans: spans,
            sinks: [-1e30, -1e30], softcap: 1_000_000)
        let causal = reference(
            q: q.asArray(Float.self), keys: refK, values: refV,
            dim: 64, queryCount: 12, queryStart: 160,
            sinks: [-1e30, -1e30], softcap: 1_000_000)
        #expect(zip(expected, causal).map { abs($0 - $1) }.max()! > 0.001)
        #expect(zip(output.asArray(Float.self), expected).map { abs($0 - $1) }.max()! < 0.0002)
        #expect(!backend.pool.writeValidation.isFaulted)
        try complete(backend.pool, roots: [output] + caches[0].innerState())
    }

    @Test
    func aSingleVisionSpanContextCannotMutateTwoPhysicalRows() throws {
        let (backend, rows, caches) = try setup(dim: 64, dtype: .float32)
        let extra = try backend.makeSequenceState(
            layerKinds: [caches[0].kind], promptLength: 4, maxLength: 1024)
        let second = try #require(extra[0] as? PagedSequenceKV)
        defer {
            backend.release(rows.map { $0 as CBv2SequenceKV? })
            backend.release(extra)
        }
        caches[0].setRows([rows[0], second])
        caches[0].bindSpanContext(
            .init(
                chunkEnd: 4,
                blocks: [CBv2ImageSpan(tokenOffset: 0, length: 4)]))
        let q = concatenated(
            [
                values(tokens: 4, dim: 64, heads: 2, tag: 1.1, dtype: .float32),
                values(tokens: 4, dim: 64, heads: 2, tag: 2.1, dtype: .float32),
            ], axis: 0)
        let k = MLXArray.zeros([2, 1, 4, 64], dtype: .float32)
        _ = caches[0].updateAndAttend(queries: q, keys: k, values: k, scale: 0.125, sinks: nil)
        #expect(backend.pool.writeValidation.isFaulted)
        #expect(rows[0].absoluteOffset == 0 && second.absoluteOffset == 0)
        #expect(rows[0].nativeRecentKeys == nil && second.nativeRecentKeys == nil)
    }

    @Test
    func readOnlyWindowCanvasUsesBoundedRingTopologyAtALaterAbsoluteOffset() throws {
        let (backend, rows, caches) = try setup(dim: 64, dtype: .float32, window: 256)
        defer { backend.release(rows.map { $0 as CBv2SequenceKV? }) }
        let row = rows[0]
        let primeK = values(tokens: 512, dim: 64, tag: 0.2, dtype: .float32)
        let primeV = values(tokens: 512, dim: 64, tag: 0.8, dtype: .float32)
        row.write(keys: primeK[0], values: primeV[0])
        try complete(backend.pool, roots: caches[0].innerState())
        let q = MLXArray.zeros([1, 2, 4, 64], dtype: .float32)
        let k = values(tokens: 4, dim: 64, tag: 1.7, dtype: .float32)
        let v = values(tokens: 4, dim: 64, tag: 2.1, dtype: .float32)
        let tableVersion = row.tableVersion
        #expect(try row.quantizedReadOnlyPages(end: 516).count == row.decodeTableLength)
        let candidate = try row.attendQuantizedReadOnly(
            queries: q, currentKeys: k, currentValues: v,
            scale: 0.125, mask: .none, prefixCount: 255)
        let output = try #require(candidate)
        let originalV = primeV.asArray(Float.self)
        var expected = [Float](repeating: 0, count: 64)
        for token in 257 ..< 512 {
            let value = Array(originalV[token * 64 ..< (token + 1) * 64])
            let decoded =
                token < 384
                ? try PagedKVQuantizationReference.roundTrip(value, config: .init(), isKey: false)
                : value
            for d in 0 ..< 64 { expected[d] += decoded[d] / 259 }
        }
        let currentV = v.asArray(Float.self)
        for token in 0 ..< 4 {
            for d in 0 ..< 64 { expected[d] += currentV[token * 64 + d] / 259 }
        }
        let actual = output.asArray(Float.self)
        #expect(
            actual.enumerated().map { abs($0.element - expected[$0.offset % 64]) }.max()! < 0.0002)
        #expect(row.absoluteOffset == 512 && row.tableVersion == tableVersion)
        try complete(backend.pool, roots: [output] + row.quantizedEvaluationRoots())
    }

    @Test
    func readOnlyCanvasDoesNotInspectUnallocatedSlotsBeforeAnAdoptedWindowBase() throws {
        let (backend, rows, caches) = try setup(dim: 64, dtype: .float32, window: 256)
        defer { backend.release(rows.map { $0 as CBv2SequenceKV? }) }
        let row = rows[0]
        row.fastForward(to: 512)
        let prefix = values(tokens: 4, dim: 64, tag: 0.8, dtype: .float32)
        row.write(keys: prefix[0], values: prefix[0])
        try complete(backend.pool, roots: caches[0].innerState())
        #expect(row.table.count < row.decodeTableLength)
        let canvas = values(tokens: 4, dim: 64, tag: 2.1, dtype: .float32)
        let candidate = try row.attendQuantizedReadOnly(
            queries: .zeros([1, 2, 4, 64], dtype: .float32),
            currentKeys: canvas, currentValues: canvas,
            scale: 0.125, mask: .none, prefixCount: 4)
        let output = try #require(candidate)
        let expected = concatenated([prefix, canvas], axis: 2).mean(axis: 2).asArray(Float.self)
        #expect(
            output.asArray(Float.self).enumerated().map {
                abs($0.element - expected[$0.offset % 64])
            }.max()! < 0.0002)
        #expect(row.absoluteOffset == 516)
        try complete(backend.pool, roots: [output] + row.quantizedEvaluationRoots())
    }

    @Test
    func readOnlyCanvasKeepsThePrefixImmutableAndHonorsBooleanVisionMask() throws {
        let (backend, rows, caches) = try setup(dim: 512, dtype: .float32)
        defer { backend.release(rows.map { $0 as CBv2SequenceKV? }) }
        let row = rows[0]
        let primeK = values(tokens: 160, dim: 512, tag: 0.2, dtype: .float32)
        let primeV = values(tokens: 160, dim: 512, tag: 0.8, dtype: .float32)
        row.write(keys: primeK[0], values: primeV[0])
        try complete(backend.pool, roots: caches[0].innerState())
        let serial = row.tableVersion
        let position = row.absoluteOffset
        let q = values(tokens: 4, dim: 512, heads: 2, tag: 1.1, dtype: .float32)
        let k = values(tokens: 4, dim: 512, tag: 1.7, dtype: .float32)
        let v = values(tokens: 4, dim: 512, tag: 2.1, dtype: .float32)
        var mask = [Bool](repeating: true, count: 4 * 164)
        for query in 0 ..< 4 { mask[query * 164 + 10] = false }
        let maskedCandidate = try row.attendQuantizedReadOnly(
            queries: q,
            currentKeys: k, currentValues: v, scale: 1 / Float(512).squareRoot(),
            mask: .array(MLXArray(mask, [1, 1, 4, 164])), prefixCount: 160)
        let withMask = try #require(maskedCandidate)
        let unmaskedCandidate = try row.attendQuantizedReadOnly(
            queries: q,
            currentKeys: k, currentValues: v, scale: 1 / Float(512).squareRoot(), mask: .none,
            prefixCount: 160)
        let unmasked = try #require(unmaskedCandidate)
        #expect(abs(withMask - unmasked).max().item(Float.self) > 1e-7)
        #expect(row.absoluteOffset == position && row.tableVersion == serial)
        #expect(row.nativeRecentKeys!.dim(1) == 128)
        try complete(backend.pool, roots: [withMask, unmasked] + row.quantizedEvaluationRoots())
    }
}
