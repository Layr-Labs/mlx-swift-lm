import Foundation
import MLX

@testable import MLXLMCommon

#if !ASYMMETRIC_CONTIGUOUS_PROBE
    import XCTest
#endif

private final class AsymmetricNativeProbeModel: CBv2SteppableModel {
    let malformedValue: Bool
    let mixedDType: Bool
    private(set) var forwardCount = 0
    init(malformedValue: Bool = false, mixedDType: Bool = false) {
        self.malformedValue = malformedValue
        self.mixedDType = mixedDType
    }
    func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache]) -> MLXArray {
        forwardCount += 1
        let n = tokens.dim(1)
        var result = MLXArray.zeros([1, n, 4])
        for cache in caches where cache.kind.sharesKVWithLayer == nil {
            let kind = cache.kind
            let q = MLXArray.ones([1, kind.queryHeads, n, kind.headDim])
            let k = MLXArray.ones([1, kind.kvHeads, n, kind.headDim])
            let v = MLXArray.ones(
                [1, kind.kvHeads, n, kind.valueHeadDim + (malformedValue ? 1 : 0)],
                dtype: mixedDType ? .float16 : .float32)
            result = cache.updateAndAttend(
                queries: q, keys: k, values: v,
                scale: 0.25, sinks: nil)
        }
        return result
    }
}

private enum AsymmetricContiguousChecks {
    struct Failure: Error { let message: String }
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw Failure(message: message) }
    }
    static func reject(_ body: () throws -> Void) throws {
        do {
            try body()
            throw Failure(message: "invalid operation accepted")
        } catch is CBv2KVError {} catch is CBv2CompleteCheckpointError {}
    }
    static func kind(window: Int? = nil, value: Int? = 4, shares: Int? = nil) -> CBv2LayerKind {
        .init(
            attention: window.map { .slidingWindow($0) } ?? .full,
            sharesKVWithLayer: shares, hasSinks: true,
            headDim: 8, valueHeadDim: value, kvHeads: 2, queryHeads: 4)
    }
    static func tile(_ range: Range<Int>, heads: Int = 2, width: Int, bias: Float = 0) -> MLXArray {
        var data: [Float] = []
        data.reserveCapacity(heads * range.count * width)
        for h in 0 ..< heads {
            for t in range {
                for d in 0 ..< width {
                    let position = Float(t) * 0.01
                    let head = Float(h) * 0.1
                    let dimension = Float(d) * 0.001
                    data.append(bias + position + head + dimension)
                }
            }
        }
        return MLXArray(data, [1, heads, range.count, width])
    }
    static func near(_ a: MLXArray, _ b: MLXArray, _ message: String) throws {
        eval(a, b)
        try require(a.shape == b.shape, message + " shape")
        if a.size > 0 {
            let error = abs(a - b).max().item(Float.self)
            try require(error.isFinite && error < 3e-5, "\(message): error \(error)")
        }
    }

    // Scalar oracle never invokes the implementation's SDPA, mask or cache.
    static func reference(
        q: MLXArray, k: MLXArray, v: MLXArray,
        firstPosition: Int, window: Int?, sinks: [Float], scale: Float
    ) -> MLXArray {
        let queries = q.asArray(Float.self)
        let keys = k.asArray(Float.self)
        let values = v.asArray(Float.self)
        let qh = q.dim(1)
        let kh = k.dim(1)
        let count = q.dim(2)
        let context = k.dim(2)
        let dk = k.dim(3)
        let dv = v.dim(3)
        var result = [Float](repeating: 0, count: qh * count * dv)
        for h in 0 ..< qh {
            for t in 0 ..< count {
                let position = firstPosition + t
                let kvh = h / (qh / kh)
                let start = window.map { max(0, position - $0 + 1) } ?? 0
                let end = min(position, context - 1)
                var scores: [Float] = []
                for row in start ... end {
                    var dot: Float = 0
                    for d in 0 ..< dk {
                        dot +=
                            queries[(h * count + t) * dk + d] * keys[(kvh * context + row) * dk + d]
                    }
                    scores.append(dot * scale)
                }
                let maximum = max(scores.max()!, sinks[h])
                let probabilities = scores.map { exp($0 - maximum) }
                let denominator = probabilities.reduce(0, +) + exp(sinks[h] - maximum)
                for (index, row) in (start ... end).enumerated() {
                    for d in 0 ..< dv {
                        result[(h * count + t) * dv + d] +=
                            probabilities[index] / denominator
                            * values[(kvh * context + row) * dv + d]
                    }
                }
            }
        }
        return MLXArray(result, [1, qh, count, dv])
    }

    static var identity: CBv2CompleteCheckpointIdentity {
        .init(
            modelAggregateHash: "fixture", promptContractID: "template", buildID: "probe",
            numericsFingerprint: "fp32")
    }
    static func manifest(valueWidth: Int = 8) throws -> CBv2CompleteCheckpointManifest {
        .init(
            identity: identity, position: 8, chunkSize: 8,
            prefixTokens: Array(repeating: 1, count: 8),
            cacheSalt: nil, assistantCodecID: nil,
            tensors: [
                try .init(role: .keys, layer: 0, shape: [1, 2, 8, 8], dtype: .float32),
                try .init(role: .values, layer: 0, shape: [1, 2, 8, valueWidth], dtype: .float32),
            ])
    }

    static func cases() -> [(String, () throws -> Void)] {
        [
            (
                "uniform rows retain legacy cast-on-append storage and byte accounting",
                {
                    let rows: [any CBv2SequenceKV] = [
                        CBv2FullSequenceKV(promptLength: 2, maxLength: 12, kvHeads: 2, headDim: 8),
                        CBv2WindowedSequenceKV(window: 3, kvHeads: 2, headDim: 8),
                    ]
                    for row in rows {
                        _ = row.update(
                            keys: tile(0 ..< 2, width: 8), values: tile(0 ..< 2, width: 8, bias: 1))
                        let allocated = row.byteCount
                        let k = tile(2 ..< 3, width: 8).asType(.float16)
                        let v = tile(2 ..< 3, width: 8, bias: 1).asType(.float16)
                        _ = row.update(keys: k, values: v)
                        let snapshot = row.snapshot()
                        try require(
                            snapshot.keys.dtype == .float32 && snapshot.values.dtype == .float32,
                            "legacy allocation dtype changed")
                        try require(
                            row.byteCount == allocated,
                            "same-capacity append changed reserved storage")
                        try near(
                            snapshot.keys[0..., 0..., 2 ..< 3, 0...], k.asType(.float32),
                            "legacy key cast")
                        try near(
                            snapshot.values[0..., 0..., 2 ..< 3, 0...], v.asType(.float32),
                            "legacy value cast")
                    }
                }
            ),
            (
                "asymmetric full growth keeps the original storage dtype in both cast directions",
                {
                    for (stored, incoming): (DType, DType) in [
                        (.float32, .float16), (.float16, .float32),
                    ] {
                        let row = CBv2FullSequenceKV(
                            promptLength: 1, maxLength: 400,
                            kvHeads: 2, headDim: 8, valueHeadDim: 4)
                        _ = row.update(
                            keys: tile(0 ..< 1, width: 8).asType(stored),
                            values: tile(0 ..< 1, width: 4).asType(stored))
                        let tailK = tile(1 ..< 258, width: 8).asType(incoming)
                        let tailV = tile(1 ..< 258, width: 4).asType(incoming)
                        _ = row.update(keys: tailK, values: tailV)
                        let snapshot = row.snapshot()
                        try require(
                            snapshot.keys.dtype == stored && snapshot.values.dtype == stored,
                            "growth changed the native allocation dtype")
                        try require(
                            row.byteCount == 400 * 2 * (8 + 4) * stored.size, "grown byte count")
                        try near(
                            snapshot.values[0..., 0..., 257 ..< 258, 0...].asType(.float32),
                            tailV[0..., 0..., 256 ..< 257, 0...].asType(stored).asType(.float32),
                            "grown cast values")
                    }
                }
            ),
            (
                "window temporary promotion preserves the ring and charges speculative incoming bytes",
                {
                    let row = CBv2WindowedSequenceKV(
                        window: 3, kvHeads: 2, headDim: 8, valueHeadDim: 4)
                    _ = row.update(
                        keys: tile(0 ..< 2, width: 8).asType(.float16),
                        values: tile(0 ..< 2, width: 4).asType(.float16))
                    let result = row.update(
                        keys: tile(2 ..< 4, width: 8), values: tile(2 ..< 4, width: 4))
                    try require(
                        result.0.dtype == .float32 && result.1.dtype == .float32,
                        "pre-eviction attention promotion changed")
                    try require(
                        row.snapshot().values.dtype == .float16 && row.byteCount == 3 * 2 * 12 * 2,
                        "temporary promotion changed ring residency")
                    row.beginSpeculativeWrite()
                    _ = row.update(keys: tile(4 ..< 6, width: 8), values: tile(4 ..< 6, width: 4))
                    try require(
                        row.byteCount == 3 * 2 * 12 * 2 + 2 * 2 * 12 * 4,
                        "staged native-byte accounting")
                    row.rollback(1)
                    row.commitSpeculativeWrite()
                    let snapshot = row.snapshot()
                    eval(snapshot.keys, snapshot.values)
                    try require(
                        snapshot.values.dtype == .float16 && row.byteCount == 3 * 2 * 12 * 2,
                        "committed ring dtype or storage changed")
                }
            ),
            (
                "implicit value width follows mutation and explicit equality is semantic",
                {
                    var implicit = kind(value: nil)
                    var explicit = kind(value: 8)
                    try require(implicit == explicit, "implicit/explicit equal geometry differs")
                    implicit.headDim = 16
                    explicit.headDim = 16
                    try require(
                        implicit.valueHeadDim == 16 && explicit.valueHeadDim == 8,
                        "value override semantics")
                    explicit.valueHeadDim = 16
                    try require(implicit == explicit, "resolved equality ignores value width")
                    try require(
                        kind().kvGeometry?.bytesPerToken(elementBytes: 4) == 96, "pair byte formula"
                    )
                    try require(
                        kind().kvGeometry?.bytesPerToken(elementBytes: 4, extraBytes: 7) == 103,
                        "extra bytes multiplied by dtype")
                }
            ),
            (
                "contiguous reservations and admission count both widths exactly once",
                {
                    var full = kind()
                    full.extraStorageBytesPerToken = 17
                    let kinds = [full, kind(window: 3)]
                    let backend = CBv2ContiguousKVBackend(
                        config: .init(bytesCapacity: 1 << 20, kvDType: .float32))
                    let rows = try backend.makeSequenceState(
                        layerKinds: kinds, promptLength: 1, maxLength: 400)
                    try require(
                        backend.bytesReserved == (257 + 3) * 96 && backend.bytesInUse == 0,
                        "initial reservations")
                    let admission = AdmissionV2(
                        layerKinds: kinds, bytesCapacity: 1 << 20,
                        config: .init(watermarkFraction: 0, elementBytes: 4))
                    try require(
                        admission.estimatedBytes(forTokens: 7) == (7 + 3) * 96,
                        "admission bytes or duplicate auxiliary charge")
                    backend.release(rows)
                    try require(
                        backend.bytesReserved == 0 && backend.bytesInUse == 0,
                        "retirement leaked reservation")
                }
            ),
            (
                "invalid geometry overflow and borrower widths reject before registration",
                {
                    let backend = CBv2ContiguousKVBackend(
                        config: .init(bytesCapacity: Int.max / 4, kvDType: .float32))
                    var invalid: [[CBv2LayerKind]] = []
                    for width in [0, -1, Int.max] {
                        var bad = kind()
                        bad.valueHeadDim = width
                        invalid.append([bad])
                    }
                    var badHeads = kind()
                    badHeads.queryHeads = 3
                    invalid.append([badHeads])
                    var overflow = kind()
                    overflow.headDim = Int(Int32.max)
                    overflow.valueHeadDim = Int(Int32.max)
                    overflow.kvHeads = Int(Int32.max)
                    overflow.queryHeads = Int(Int32.max)
                    invalid.append([overflow])
                    invalid.append([kind(), kind(value: 8, shares: 0)])
                    for kinds in invalid {
                        try reject {
                            _ = try backend.makeSequenceState(
                                layerKinds: kinds, promptLength: 1, maxLength: 8)
                        }
                        try require(
                            backend.bytesReserved == 0 && backend.bytesInUse == 0,
                            "invalid geometry registered rows")
                    }
                    try reject {
                        _ = try backend.makeSequenceState(
                            layerKinds: [kind()], promptLength: -1, maxLength: 8)
                    }
                    try reject {
                        _ = try backend.makeSequenceState(
                            layerKinds: [kind()], promptLength: 1, maxLength: Int.max)
                    }
                }
            ),
            (
                "full cache empty snapshots growth rollback and restored destinations keep value width",
                {
                    let row = CBv2FullSequenceKV(
                        promptLength: 1, maxLength: 400, kvHeads: 2, headDim: 8, valueHeadDim: 4)
                    try require(
                        row.snapshot().keys.shape == [1, 2, 0, 8]
                            && row.snapshot().values.shape == [1, 2, 0, 4], "empty shapes")
                    _ = row.update(
                        keys: tile(0 ..< 260, width: 8), values: tile(0 ..< 260, width: 4, bias: 1))
                    try near(
                        row.snapshot().values, tile(0 ..< 260, width: 4, bias: 1), "grown values")
                    try require(row.byteCount == 260 * 96, "first allocation adopts required rows")
                    _ = row.update(
                        keys: tile(260 ..< 300, width: 8),
                        values: tile(260 ..< 300, width: 4, bias: 1))
                    try require(row.byteCount == 400 * 96, "capped growth bytes")
                    row.rollback(3)
                    try near(
                        row.snapshot().values, tile(0 ..< 297, width: 4, bias: 1), "rollback values"
                    )
                    let restored = try CBv2FullSequenceKV(
                        restoredKeys: tile(0 ..< 12, width: 8),
                        restoredValues: tile(0 ..< 12, width: 4, bias: 1), offset: 7, maxLength: 12,
                        kvHeads: 2, headDim: 8, valueHeadDim: 4)
                    try near(
                        restored.snapshot().values, tile(0 ..< 7, width: 4, bias: 1),
                        "restored values")
                    try reject {
                        _ = try CBv2FullSequenceKV(
                            restoredKeys: tile(0 ..< 12, width: 8),
                            restoredValues: tile(0 ..< 12, width: 8), offset: 7, maxLength: 12,
                            kvHeads: 2, headDim: 8, valueHeadDim: 4)
                    }
                }
            ),
            (
                "window history retains unequal widths through wrap and speculative rollback",
                {
                    let row = CBv2WindowedSequenceKV(
                        window: 3, kvHeads: 2, headDim: 8, valueHeadDim: 4)
                    try require(row.snapshot().values.shape == [1, 2, 0, 4], "empty window shape")
                    _ = row.update(
                        keys: tile(0 ..< 5, width: 8), values: tile(0 ..< 5, width: 4, bias: 1))
                    try near(
                        row.snapshot().values, tile(2 ..< 5, width: 4, bias: 1),
                        "chunk greater than ring")
                    row.beginSpeculativeWrite()
                    _ = row.update(
                        keys: tile(5 ..< 8, width: 8), values: tile(5 ..< 8, width: 4, bias: 1))
                    row.rollback(2)
                    row.commitSpeculativeWrite()
                    try near(row.snapshot().keys, tile(3 ..< 6, width: 8), "committed ring keys")
                    try near(
                        row.snapshot().values, tile(3 ..< 6, width: 4, bias: 1),
                        "committed ring values")
                    try require(row.byteCount == 3 * 96, "window allocation bytes")
                }
            ),
            (
                "frozen replay preserves authoritative values then appends with independent widths",
                {
                    let row = CBv2FrozenReplayFullSequenceKV(
                        snapshot: (tile(0 ..< 5, width: 8), tile(0 ..< 5, width: 4, bias: 1), 5),
                        replayStart: 2, maxLength: 12, kvHeads: 2, headDim: 8, valueHeadDim: 4)
                    let replay = row.update(
                        keys: tile(2 ..< 5, width: 8, bias: 99),
                        values: tile(2 ..< 5, width: 4, bias: 99))
                    try near(
                        replay.1, tile(0 ..< 5, width: 4, bias: 1),
                        "poison replay did not alter stored values")
                    _ = row.update(
                        keys: tile(5 ..< 6, width: 8), values: tile(5 ..< 6, width: 4, bias: 1))
                    try require(row.byteCount == 12 * 96, "frozen append capacity bytes")
                    row.rollback(1)
                    try near(
                        row.snapshot().values, tile(0 ..< 5, width: 4, bias: 1),
                        "frozen append rollback")
                }
            ),
            (
                "in-memory prefix adoption validates role shapes and leaves no failed owners",
                {
                    let kinds = [kind()]
                    let capability = CBv2PrefixReuseCapability.derive(
                        layerKinds: kinds, backend: .contiguousUnquantized)
                    guard
                        let plan = capability.plan(
                            matchedBoundary: 5, exactStagedFullKVBytes: 5 * 96,
                            maximumSequenceLength: 12)
                    else { throw Failure(message: "missing reuse plan") }
                    let backend = CBv2ContiguousKVBackend(
                        config: .init(bytesCapacity: 1 << 20, kvDType: .float32))
                    for values in [
                        tile(0 ..< 5, width: 8), tile(0 ..< 5, width: 4).asType(.float16),
                        MLXArray.zeros([1]),
                    ] {
                        try reject {
                            _ = try backend.makeSequenceState(
                                adopting: [(tile(0 ..< 5, width: 8), values, 5)],
                                plan: plan, layerKinds: kinds, maxLength: 12)
                        }
                        try require(backend.bytesReserved == 0, "invalid snapshot registered owner")
                    }
                    let rows = try backend.makeSequenceState(
                        adopting: [(tile(0 ..< 5, width: 8), tile(0 ..< 5, width: 4, bias: 1), 5)],
                        plan: plan, layerKinds: kinds, maxLength: 12)
                    try require(backend.bytesReserved == 12 * 96, "adopted native bytes")
                    try near(
                        rows[0]!.snapshot().values, tile(0 ..< 5, width: 4, bias: 1),
                        "adopted values")
                    backend.release(rows)
                }
            ),
            (
                "full sliding and last-query attention output value width matches scalar oracle",
                {
                    let count = 7
                    let q = tile(0 ..< 7, heads: 4, width: 8, bias: 0.3)
                    let k = tile(0 ..< 7, width: 8)
                    let v = tile(0 ..< 7, width: 4, bias: 1)
                    let sinkValues: [Float] = [0, 0.5, -0.5, 1]
                    let sinks = MLXArray(sinkValues)
                    for window: Int? in [nil, 3] {
                        let kind = kind(window: window)
                        let backend = CBv2ContiguousKVBackend(
                            config: .init(bytesCapacity: 1 << 20, kvDType: .float32))
                        let rows = try backend.makeSequenceState(
                            layerKinds: [kind], promptLength: count, maxLength: 12)
                        let output = CBv2AttentionV1.updateAndAttend(
                            rows: [rows[0]!], kind: kind,
                            queries: q, keys: k, values: v, scale: 0.25, sinks: sinks)
                        try near(
                            output,
                            reference(
                                q: q, k: k, v: v, firstPosition: 0,
                                window: window, sinks: sinkValues, scale: 0.25),
                            "asymmetric attention")
                        backend.release(rows)
                    }
                    let row = CBv2FullSequenceKV(
                        promptLength: count, maxLength: 12, kvHeads: 2, headDim: 8, valueHeadDim: 4)
                    let last = CBv2AttentionV1.updateAndAttendLastQuery(
                        rows: [row], kind: kind(),
                        queries: q[0..., 0..., 6 ..< 7, 0...], keys: k, values: v, scale: 0.25,
                        sinks: sinks)
                    try near(
                        last,
                        reference(
                            q: q[0..., 0..., 6 ..< 7, 0...], k: k, v: v,
                            firstPosition: 6, window: nil, sinks: sinkValues, scale: 0.25),
                        "last-query output")
                    try require(
                        last.shape == [1, 4, 1, 4] && row.absoluteOffset == count,
                        "last query committed wrong geometry")
                }
            ),
            (
                "native probe observes separate widths and drains rejected shape or dtype state",
                {
                    let kinds = [kind()]
                    func caches() -> [any CBv2AttendingLayerCache] {
                        [CBv2LayerCache(layerIndex: 0, kind: kinds[0])]
                    }
                    let goodCaches = caches()
                    let result = try CBv2NativeKVTypeProbe.run(
                        model: AsymmetricNativeProbeModel(), layerKinds: kinds, caches: goodCaches)
                    try require(
                        result.observations.count == 2
                            && result.observations.allSatisfy {
                                $0.keysShape.last == 8 && $0.valuesShape.last == 4
                            }, "probe dimensions")
                    try require(goodCaches.allSatisfy { $0.rows.isEmpty }, "probe retained owner")
                    for model in [
                        AsymmetricNativeProbeModel(malformedValue: true),
                        AsymmetricNativeProbeModel(mixedDType: true),
                    ] {
                        let bound = caches()
                        try reject {
                            _ = try CBv2NativeKVTypeProbe.run(
                                model: model, layerKinds: kinds, caches: bound)
                        }
                        try require(
                            bound.allSatisfy { $0.rows.isEmpty }, "rejected probe retained owner")
                    }
                }
            ),
            (
                "native probe rejects invalid borrower geometry before any forward or row binding",
                {
                    let owner = kind()
                    for borrower in [
                        kind(value: 8, shares: 0), kind(window: 3, shares: 0),
                        kind(shares: -1), kind(shares: 2), kind(shares: 1),
                    ] {
                        let kinds = [owner, borrower]
                        let caches: [any CBv2AttendingLayerCache] = kinds.enumerated().map {
                            CBv2LayerCache(layerIndex: $0.offset, kind: $0.element)
                        }
                        let model = AsymmetricNativeProbeModel()
                        try reject {
                            _ = try CBv2NativeKVTypeProbe.run(
                                model: model, layerKinds: kinds, caches: caches)
                        }
                        try require(
                            model.forwardCount == 0 && caches.allSatisfy { $0.rows.isEmpty },
                            "invalid ownership reached model or retained probe rows")
                    }
                }
            ),
            (
                "asymmetric paging refuses before constructing physical storage",
                {
                    let asymmetric = CBv2LayerKind(
                        attention: .full, headDim: 192, valueHeadDim: 128, kvHeads: 4,
                        queryHeads: 64)
                    try reject {
                        _ = try PagedKVBackend(
                            layerKinds: [asymmetric], config: .init(capacityBytes: 1 << 20))
                    }
                }
            ),
            (
                "complete codec refuses every export import and historical route without reservation",
                {
                    let kinds = [
                        CBv2LayerKind(
                            attention: .full, headDim: 192, valueHeadDim: 128, kvHeads: 4,
                            queryHeads: 64)
                    ]
                    let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 1 << 20)
                    let checkpoint = CBv2RecurrentCheckpoint(
                        position: 8, chunkSize: 8, layers: [:], byteCount: 0)
                    let tokens = Array(repeating: 1, count: 9)
                    let request = CBv2Request(id: .init(1), promptTokens: tokens, maxTokens: 1)
                    let initial = admission.bytesReserved
                    for paged: PagedKVPoolConfig? in [nil, .init(capacityBytes: 1 << 20)] {
                        let codec = CBv2CompleteCheckpointCodec(
                            identity: identity, layerKinds: kinds, recurrentSpec: nil,
                            kvDTypes: [.float32], assistant: nil, admission: admission,
                            pagedConfig: paged)
                        try require(
                            codec.unsupportedAsymmetricGeometry,
                            "codec did not retain refusal state")
                        try reject { _ = try codec.tensorDescriptors(position: 8) }
                        try reject { _ = try codec.plan(manifest: manifest(), request: request) }
                        try reject {
                            _ = try codec.export(
                                checkpoint: checkpoint, kv: [nil], tokens: tokens, cacheSalt: nil)
                        }
                        try reject {
                            _ = try codec.export(
                                checkpoint: checkpoint, state: [nil], tokens: tokens, cacheSalt: nil
                            )
                        }
                        try reject {
                            _ = try codec.exportPaged(
                                checkpoint: checkpoint, state: [nil], tokens: tokens, cacheSalt: nil
                            )
                        }
                        try reject {
                            _ = try codec.historicalReusePlan(position: 8, maximumSequenceLength: 9)
                        }
                        try require(
                            admission.bytesReserved == initial
                                && admission.transientBytesReserved == 0
                                && codec.historicalLayout == nil,
                            "refused codec obtained a reservation/owner")
                    }
                }
            ),
            (
                "uniform manifest stays valid while asymmetric tensor descriptors are refused",
                {
                    let uniform = try manifest()
                    try require(
                        try uniform.validateStructure() == 2 * 2 * 8 * 8 * 4,
                        "uniform manifest bytes changed")
                    let decoded = try JSONDecoder().decode(
                        CBv2CompleteCheckpointManifest.self, from: JSONEncoder().encode(uniform))
                    try require(decoded == uniform, "uniform manifest roundtrip")
                    try reject { _ = try manifest(valueWidth: 4).validateStructure() }
                }
            ),
        ]
    }
}

#if ASYMMETRIC_CONTIGUOUS_PROBE
    @main
    private struct AsymmetricContiguousProbe {
        static func main() throws {
            let cases = AsymmetricContiguousChecks.cases()
            var failures = 0
            for (name, body) in cases {
                do {
                    try body()
                    print("PASS \(name)")
                } catch {
                    failures += 1
                    print("FAIL \(name): \(error)")
                }
            }
            print(
                "RESULT discovered=\(cases.count) passed=\(cases.count-failures) failed=\(failures) skipped=0"
            )
            if failures > 0 {
                throw AsymmetricContiguousChecks.Failure(
                    message: "asymmetric contiguous probe failed")
            }
        }
    }
#else
    final class CBv2AsymmetricContiguousTests: XCTestCase {
        func testAsymmetricContiguousContracts() throws {
            for (name, body) in AsymmetricContiguousChecks.cases() {
                do { try body() } catch { XCTFail("\(name): \(error)") }
            }
        }
    }
#endif
