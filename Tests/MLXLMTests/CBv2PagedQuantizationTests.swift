import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("Packed paged KV quantization", .serialized)
struct CBv2PagedQuantizationTests {
    private func makeBackend(dim: Int, dtype: DType = .float32,
                             quantization: PagedKVQuantizationConfig,
                             hasSinks: Bool = false) throws -> (PagedKVBackend, CBv2LayerKind) {
        let kind = CBv2LayerKind(attention: .full, hasSinks: hasSinks, headDim: dim, kvHeads: 2,
                               queryHeads: 4)
        let page = try quantization.bytesPerToken(kvHeads: 2, headDim: dim) * 16
        return (try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 8 << 20, dtype: dtype, maxPrefillChunk: 512,
            nominalMaxSequenceLength: 512, maxBufferLength: 8 << 20,
            segmentSizeBytes: 3 * page, quantization: quantization)), kind)
    }

    @Test func checkedFormatIdentityAndAccounting() throws {
        let q = PagedKVQuantizationConfig()
        #expect(try q.bytesPerToken(kvHeads: 2, headDim: 256) == 640)
        #expect(try PagedKVQuantizationConfig(keyBits: 8).bytesPerToken(kvHeads: 2, headDim: 256) == 896)
        #expect(try PagedKVQuantizationConfig(keyBits: 8, valueBits: 8).bytesPerToken(kvHeads: 2, headDim: 256) == 1152)
        #expect(q.resolvedRotationBlockSize(headDim: 64) == 64)
        let encoded = try JSONEncoder().encode(q)
        #expect(try JSONDecoder().decode(PagedKVQuantizationConfig.self, from: encoded) == q)
        for bad in [PagedKVQuantizationConfig(keyBits: 3), .init(groupSize: 0), .init(rotationBlockSize: 3)] {
            #expect(throws: CBv2KVError.self) { try bad.validate(headDim: 256) }
        }
        #expect(throws: CBv2KVError.self) { try q.bytesPerToken(kvHeads: Int.max, headDim: 256) }
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2),
                     CBv2LayerKind(attention: .slidingWindow(64), headDim: 64, kvHeads: 1, queryHeads: 2)]
        let pool = try PagedKVPool(layerKinds: kinds, config: .init(
            capacityBytes: 1 << 20, segmentSizeBytes: 32768, quantization: q))
        #expect(pool.groupKey(forLayer: 0).quantization == q)
        #expect(pool.groupKey(forLayer: 1).quantization == nil)
        #expect(throws: CBv2KVError.self) {
            try PagedKVPool(layerKinds: kinds, config: .init(capacityBytes: 1 << 20, quantization: q))
        }
    }

    @Test(arguments: [64, 128, 256, 512])
    func rotationPreservesDotProductAndInverts(dim: Int) throws {
        let x = (0 ..< dim).map { sin(Float($0) * 0.13) }
        let y = (0 ..< dim).map { cos(Float($0) * 0.21) }
        let q = PagedKVQuantizationConfig()
        let r = q.resolvedRotationBlockSize(headDim: dim)
        let xr = PagedKVQuantizationReference.rotate(x, blockSize: r)
        let yr = PagedKVQuantizationReference.rotate(y, blockSize: r)
        let restored = PagedKVQuantizationReference.rotate(xr, blockSize: r, inverse: true)
        #expect(zip(x, restored).map { abs($0 - $1) }.max()! < 1e-5)
        #expect(abs(zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
            - zip(xr, yr).reduce(0) { $0 + $1.0 * $1.1 }) < 1e-4)
        let actual = PagedQuantizedMetal.rotate(MLXArray(x, [1, dim]), config: q).asArray(Float.self)
        #expect(zip(actual, xr).map { abs($0 - $1) }.max()! < 1e-5)
    }

    @Test(arguments: [DType.float16, .bfloat16, .float32], [64, 256, 512])
    func packedStorageGatherAndNewestDecode(dtype: DType, dim: Int) throws {
        for bits in [(4, 4), (8, 4), (8, 8)] {
            let quantization = PagedKVQuantizationConfig(keyBits: bits.0, valueBits: bits.1)
            let (backend, kind) = try makeBackend(dim: dim, dtype: dtype, quantization: quantization)
            let state = try backend.makeSequenceState(layerKinds: [kind], promptLength: 33, maxLength: 96)
            defer { backend.release(state) }
            let row = try #require(state[0] as? PagedSequenceKV)
            let group = backend.pool.group(row.groupKey)
            let k = MLXRandom.normal([2, 33, dim], key: MLXRandom.key(10)).asType(dtype)
            let v = MLXRandom.normal([2, 33, dim], key: MLXRandom.key(20)).asType(dtype)
            row.write(keys: k, values: v)
            let snapshot = row.snapshot()
            let originalK = k.asType(.float32).asArray(Float.self)
            let originalV = v.asType(.float32).asArray(Float.self)
            let gatheredK = snapshot.keys.asType(.float32).asArray(Float.self)
            let gatheredV = snapshot.values.asType(.float32).asArray(Float.self)
            for index in 0 ..< 66 {
                let range = index * dim ..< (index + 1) * dim
                let expectedK = try PagedKVQuantizationReference.roundTrip(
                    Array(originalK[range]), config: quantization, isKey: true)
                let expectedV = try PagedKVQuantizationReference.roundTrip(
                    Array(originalV[range]), config: quantization, isKey: false)
                let tolerance: Float = dtype == .bfloat16 ? 0.025 : (dtype == .float16 ? 0.003 : 1e-4)
                #expect(zip(gatheredK[range], expectedK).map { abs($0 - $1) }.max()! < tolerance)
                #expect(zip(gatheredV[range], expectedV).map { abs($0 - $1) }.max()! < tolerance)
            }
            let shape = try quantization.rowLayout(headDim: dim)
            for segment in group.segments.values {
                #expect(segment.storage.dtype == .uint8)
                #expect(segment.storage.nbytes == segment.pages.count * group.pageBytes)
                #expect(segment.valueOffset == segment.pages.count * 2 * 16 * shape.keyRowBytes)
                let raw = segment.storage.asArray(UInt8.self)
                #expect(raw[0 ..< 2 * 16 * shape.keyRowBytes].allSatisfy { $0 == 0 })
                #expect(raw[segment.valueOffset ..< segment.valueOffset + 2 * 16 * shape.valueRowBytes].allSatisfy { $0 == 0 })
            }
            // Use one real appended decode token and then read the same state
            // without a write. Both must attend its compressed representation.
            let q = MLXRandom.normal([1, 4, dim], key: MLXRandom.key(30)).asType(dtype)
            let nk = MLXRandom.normal([1, 2, dim], key: MLXRandom.key(40)).asType(dtype)
            let nv = MLXRandom.normal([1, 2, dim], key: MLXRandom.key(50)).asType(dtype)
            let target = row.prepareDecodeWrite()
            let info = row.seqInfoRow(attending: row.decodeAttendRange, writeTarget: target)
            let params = MLXArray([Float(2), 1 / Float(dim).squareRoot(), 0, 0, 0, 0, 0, 0])
            let sinks = MLXArray([Float(0.2), -0.1, 0.4, -0.3, 0, 0, 0, 0])
            let appended = PagedSegmentAttention.decode(
                queries: q, newKeys: nk, newValues: nv, group: group,
                rows: [.init(pages: row.table, info: info)], sinks: sinks,
                params: params, softcap: true, source: backend.pool.kernelSource)
            eval(appended)
            let reread = PagedSegmentAttention.decode(
                queries: q, newKeys: nil, newValues: nil, group: group,
                rows: [.init(pages: row.table, info: info)], sinks: sinks,
                params: params, softcap: true, source: backend.pool.kernelSource)
            #expect(appended.asData(access: .copy).data == reread.asData(access: .copy).data)
            #expect(appended.asArray(Float.self).allSatisfy { $0.isFinite })
            // Independent scalar attention over encoded rows checks address
            // interpretation, K/V bit asymmetry, GQA, softcap and learned sinks.
            let qCPU = q.asType(.float32).asArray(Float.self)
            let nkCPU = nk.asType(.float32).asArray(Float.self)
            let nvCPU = nv.asType(.float32).asArray(Float.self)
            var expected: [Float] = []
            for head in 0 ..< 4 {
                let kh = head / 2
                let qr = PagedKVQuantizationReference.rotate(
                    Array(qCPU[head * dim ..< (head + 1) * dim]),
                    blockSize: quantization.resolvedRotationBlockSize(headDim: dim))
                var scores: [Float] = [], cachedValues: [[Float]] = []
                for token in 0 ..< 34 {
                    let oldStart = (kh * 33 + min(token, 32)) * dim
                    let key = token == 33 ? Array(nkCPU[kh * dim ..< (kh + 1) * dim])
                        : Array(originalK[oldStart ..< oldStart + dim])
                    let value = token == 33 ? Array(nvCPU[kh * dim ..< (kh + 1) * dim])
                        : Array(originalV[oldStart ..< oldStart + dim])
                    let kr = try PagedKVQuantizationReference.encode(key, config: quantization, isKey: true).values
                    let vr = try PagedKVQuantizationReference.encode(value, config: quantization, isKey: false).values
                    let score = zip(qr, kr).reduce(Float(0)) { $0 + $1.0 * $1.1 } / Float(dim).squareRoot()
                    scores.append(2 * tanh(score / 2))
                    cachedValues.append(vr)
                }
                let sink: Float = [0.2, -0.1, 0.4, -0.3][head]
                let maximum = max(sink, scores.max()!)
                let weights = scores.map { exp($0 - maximum) }
                let denominator = weights.reduce(exp(sink - maximum), +)
                for column in 0 ..< dim {
                    var value: Float = 0
                    for token in 0 ..< 34 { value += weights[token] * cachedValues[token][column] }
                    expected.append(value / denominator)
                }
            }
            #expect(appended.dtype == dtype)
            let outputTolerance: Float = dtype == .float32 ? 0.0002 : (dtype == .float16 ? 0.001 : 0.01)
            #expect(zip(appended.asArray(Float.self), expected).map { abs($0 - $1) }.max()! < outputTolerance)
        }
    }

    @Test(arguments: [33, 273], [false, true])
    func directPackedPrefillEqualsSerialDecodeAcrossQueryBlocks(tokens: Int, negativeLogits: Bool) throws {
        let quant = PagedKVQuantizationConfig()
        let (prefillBackend, kind) = try makeBackend(dim: 64, quantization: quant, hasSinks: true)
        let (serialBackend, _) = try makeBackend(dim: 64, quantization: quant, hasSinks: true)
        let a = try prefillBackend.makeSequenceState(layerKinds: [kind], promptLength: tokens, maxLength: 512)
        let b = try serialBackend.makeSequenceState(layerKinds: [kind], promptLength: tokens, maxLength: 512)
        defer { prefillBackend.release(a); serialBackend.release(b) }
        let prefill = prefillBackend.makeLayerCaches()[0]
        let serial = serialBackend.makeLayerCaches()[0]
        prefill.setRows([a[0]!]); serial.setRows([b[0]!])
        let q = negativeLogits ? -20 * MLXArray.ones([1, 4, tokens, 64])
            : MLXRandom.normal([1, 4, tokens, 64], key: MLXRandom.key(101))
        let k = negativeLogits ? MLXArray.ones([1, 2, tokens, 64])
            : MLXRandom.normal([1, 2, tokens, 64], key: MLXRandom.key(102))
        let v = negativeLogits ? 3 * MLXArray.ones([1, 2, tokens, 64])
            : MLXRandom.normal([1, 2, tokens, 64], key: MLXRandom.key(103))
        let sinks: MLXArray? = negativeLogits ? nil : MLXArray([Float(0.2), -0.1, 0.4, -0.3])
        let actual = prefill.updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: sinks)
        eval(actual)
        var parts: [MLXArray] = []
        for token in 0 ..< tokens {
            let out = serial.updateAndAttend(
                queries: q[0..., 0..., token ..< token + 1, 0...],
                keys: k[0..., 0..., token ..< token + 1, 0...],
                values: v[0..., 0..., token ..< token + 1, 0...], scale: 0.125, sinks: sinks)
            eval(out)
            parts.append(out)
        }
        let reference = concatenated(parts, axis: 2)
        #expect(actual.asData(access: .copy).data == reference.asData(access: .copy).data)
        if negativeLogits {
            #expect(actual.asArray(Float.self).map { abs($0 - 3) }.max()! < 1e-5)
        }
        let leases = prefillBackend.pool.takePendingQuantizedScratch()
        #expect(leases.count == 1, "all query blocks share one charged workspace")
        for lease in leases { lease.finishAfterSynchronization() }
        for lease in serialBackend.pool.takePendingQuantizedScratch() { lease.finishAfterSynchronization() }
    }


    @Test func constantOutlierRowsAndAsymmetricDiagnosticFormat() throws {
        for quant in [PagedKVQuantizationConfig(keyBits: 4, valueBits: 8, groupSize: 32, rotationBlockSize: 0),
                      .init(keyBits: 8, valueBits: 8, groupSize: 128, rotationBlockSize: 256)] {
            let (backend, kind) = try makeBackend(dim: 256, quantization: quant)
            let state = try backend.makeSequenceState(layerKinds: [kind], promptLength: 2, maxLength: 16)
            defer { backend.release(state) }
            let row = try #require(state[0] as? PagedSequenceKV)
            var input = [Float](repeating: 1.25, count: 2 * 2 * 256)
            input[256] = 1e20; input[257] = -1e20
            input[768] = -3.75
            let k = MLXArray(input, [2, 2, 256])
            row.write(keys: k, values: k)
            let result = row.snapshot()
            for (isKey, actual) in [(true, result.keys), (false, result.values)] {
                let got = actual.asArray(Float.self)
                for index in 0 ..< 4 {
                    let range = index * 256 ..< (index + 1) * 256
                    let want = try PagedKVQuantizationReference.roundTrip(Array(input[range]), config: quant, isKey: isKey)
                    let tolerance = max(Float(1e-5), want.map { abs($0) }.max()! * 2e-5)
                    for (a, b) in zip(got[range], want) {
                        #expect(a.isFinite && b.isFinite)
                        #expect(abs(a - b) <= tolerance)
                    }
                }
            }
            let key = backend.pool.groupKey(forLayer: 0)
            let group = backend.pool.group(key)
            let expected = group.segmentLayout!.allocationBytes(addingUsablePages: 1)! - group.pageBytes
            #expect(backend.pool.minimumSegmentedOverhead(tokens: 16, layerKinds: [kind]) == expected)
        }
    }

    @Test func packedVisionSpanMaskMatchesIndependentDenseAttention() throws {
        let quant = PagedKVQuantizationConfig()
        let (backend, kind) = try makeBackend(dim: 64, quantization: quant)
        let state = try backend.makeSequenceState(layerKinds: [kind], promptLength: 33, maxLength: 64)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([row])
        cache.bindSpanContext(.init(chunkEnd: 33, blocks: [.init(tokenOffset: 5, length: 20)]))
        let q = MLXRandom.normal([1, 4, 33, 64], key: MLXRandom.key(111))
        let k = MLXRandom.normal([1, 2, 33, 64], key: MLXRandom.key(112))
        let v = MLXRandom.normal([1, 2, 33, 64], key: MLXRandom.key(113))
        let actual = cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
        eval(actual)
        cache.bindSpanContext(nil)
        let snapshot = row.snapshot()
        let mask = (0 ..< 33).flatMap { query in
            (0 ..< 33).map { key in key <= query || (5 ..< 25).contains(query) && (5 ..< 25).contains(key) }
        }
        let reference = PagedAttentionReference.composedAttention(
            queries: q, keys: snapshot.keys, values: snapshot.values, scale: 0.125,
            boolMask: MLXArray(mask, [33, 33]), sinks: nil, softcap: nil)
        #expect(allClose(actual, reference, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        for lease in backend.pool.takePendingQuantizedScratch() { lease.finishAfterSynchronization() }
    }


    @Test func stridedInputsAndSharedTopologyArenas() throws {
        let quant = PagedKVQuantizationConfig()
        let (backend, kind) = try makeBackend(dim: 64, quantization: quant)
        let (referenceBackend, _) = try makeBackend(dim: 64, quantization: quant)
        let a = try backend.makeSequenceState(layerKinds: [kind], promptLength: 273, maxLength: 512)
        let b = try referenceBackend.makeSequenceState(layerKinds: [kind], promptLength: 273, maxLength: 512)
        defer { backend.release(a); referenceBackend.release(b) }
        let cache = backend.makeLayerCaches()[0], reference = referenceBackend.makeLayerCaches()[0]
        cache.setRows([a[0]!]); reference.setRows([b[0]!])
        let q = MLXRandom.normal([1, 273, 4, 64], key: MLXRandom.key(211)).transposed(0, 2, 1, 3)
        let k = MLXRandom.normal([1, 273, 2, 64], key: MLXRandom.key(212)).transposed(0, 2, 1, 3)
        let v = MLXRandom.normal([1, 273, 2, 64], key: MLXRandom.key(213)).transposed(0, 2, 1, 3)
        func copy(_ x: MLXArray) -> MLXArray { MLXArray(x.asArray(Float.self), x.shape) }
        let want = reference.updateAndAttend(queries: copy(q), keys: copy(k), values: copy(v), scale: 0.125, sinks: nil)
        let got = cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
        #expect(got.asData(access: .copy).data == want.asData(access: .copy).data)
        for pool in [backend.pool, referenceBackend.pool] {
            for lease in pool.takePendingQuantizedScratch() { lease.finishAfterSynchronization() }
        }

        let (fragmented, fragmentedKind) = try makeBackend(dim: 64, quantization: quant)
        let states = try fragmented.makeSequenceState(layerKinds: [fragmentedKind], promptLength: 1025, maxLength: 1040)
        defer { fragmented.release(states) }
        let row = try #require(states[0] as? PagedSequenceKV)
        let kv = MLXArray.ones([2, 1025, 64])
        row.write(keys: kv, values: kv)
        let group = fragmented.pool.group(row.groupKey)
        let prepared = PagedSegmentPreparedDispatch(
            rows: [.init(pages: row.table, info: row.seqInfoRow(attending: row.decodeAttendRange))],
            group: group, partitionTokens: 256, hasWrite: false)
        #expect(prepared.plan.buckets.count > 1)
        #expect(prepared.allocationArrays.count == 2)
        eval(prepared.allocationArrays + prepared.metadata.flatMap { [$0.records, $0.valueOffsets] })
        let rootInfo = try prepared.allocationArrays[0].evaluatedBufferInfo()
        let lastInfo = try prepared.metadata.last!.records.evaluatedBufferInfo()
        let root = try #require(rootInfo)
        let last = try #require(lastInfo)
        #expect(last.isRowContiguous && last.dataOffset > 0)
        #expect(last.allocatedBytes == root.allocatedBytes)
    }

}
