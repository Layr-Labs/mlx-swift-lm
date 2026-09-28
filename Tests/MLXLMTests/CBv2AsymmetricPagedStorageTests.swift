import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

// Shared native fixtures, no model files and no replacement storage/backend.
func asymPagedKind(_ key: Int = 192, _ value: Int = 128, window: Int? = nil,
                   heads: Int = 1, queries: Int = 2, shares: Int? = nil, sinks: Bool = false) -> CBv2LayerKind {
    .init(attention: window.map { .slidingWindow($0) } ?? .full, sharesKVWithLayer: shares, hasSinks: sinks,
          headDim: key, valueHeadDim: value, kvHeads: heads, queryHeads: queries)
}
func asymPagedValues(heads: Int, start: Int, count: Int, width: Int, bias: Float) -> [Float] {
    (0..<heads*count*width).map { i in
        bias + Float(i / (count*width)) / 4 + Float(start + (i / width) % count) / 32 + Float(i % width) / 1024
    }
}
func asymPagedTensor(heads: Int = 1, start: Int, count: Int, width: Int, dtype: DType, bias: Float = 0) -> MLXArray {
    MLXArray(asymPagedValues(heads: heads,start: start,count: count,width: width,bias: bias))
        .reshaped([1,heads,count,width]).asType(dtype)
}
func asymPagedRounded(_ value: Float, dtype: DType) -> Float {
    if dtype == .float16 { return Float(Float16(value)) }
    if dtype == .bfloat16 {
        let bits = value.bitPattern
        return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff0000)
    }
    return value
}
final class AsymPagedFixture {
    let kinds: [CBv2LayerKind]
    let backend: PagedKVBackend
    let admission: AdmissionV2
    var owned: [[CBv2SequenceKV?]] = []
    init(_ kinds: [CBv2LayerKind], dtype: DType = .float32, batch: Int = 2,
         context: Int = 128, queries: Int = 32, capacity: Int = 512 << 20,
         processOwner: (any CBv2ProcessMemoryOwner)? = nil) throws {
        self.kinds = kinds
        let limits = try CBv2PagedGatheredAttentionLimits(maximumBatchSize: batch,maximumQueryTokens: queries,
            maximumContextTokens: context,maximumInFlightGraphs: 2,maximumScratchBytes: 512 << 20)
        let config = PagedKVPoolConfig(capacityBytes: capacity,dtype: dtype,maxPrefillChunk: 32,
            nominalMaxSequenceLength: context,maxBufferLength: 4 << 20,segmentSizeBytes: 128 << 10,
            layerDTypes: Array(repeating: dtype,count: kinds.count),gatheredAttention: limits)
        backend = try PagedKVBackend(layerKinds: kinds,config: config)
        admission = AdmissionV2(layerKinds: kinds,bytesCapacity: capacity,
            config: .init(watermarkFraction: 0,elementBytes: dtype.size,layerElementBytes: Array(repeating: dtype.size,count: kinds.count)),
            residency: backend.kvResidency,processMemoryOwner: processOwner)
        backend.pool.bindAdmission(admission)
    }
    func rows(maximum: Int = 128) throws -> [CBv2SequenceKV?] {
        let value = try backend.makeSequenceState(layerKinds: kinds,promptLength: 0,maxLength: maximum)
        owned.append(value); return value
    }
    deinit {
        StreamOrDevice.default.stream.synchronize()
        for rows in owned { backend.release(rows) }
        owned.removeAll()
    }
}

/// Records calls from real Admission/physical backing owners; it does not
/// replace allocation, page storage or the engine's ledger arithmetic.
private final class AsymPagedProcessReceipt: CBv2ProcessMemoryOwner, @unchecked Sendable {
    let maximum: UInt64
    private let lock = NSLock()
    private var c: UInt64 = 0, m: UInt64 = 0
    var charge: UInt64 { lock.withLock { c } }
    var coverage: UInt64 { lock.withLock { m } }
    init(maximum: UInt64) { self.maximum = maximum }
    func replaceCharge(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= maximum, bytes >= m else { throw MLXError.caught("intentional process reservation refusal") }
            c = bytes
        }
    }
    func recordMaterialization(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= c-m else { throw MLXError.caught("materialization exceeded real charge") }
            m += bytes
        }
    }
    func withdrawCoverage(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= m else { throw MLXError.caught("duplicate materialization withdrawal") }
            m -= bytes
        }
    }
    func retire() { lock.withLock { c = 0 } }
}

final class CBv2AsymmetricPagedStorageTests: XCTestCase {
    func testGroupIdentityBytesAndLegacyDescription() throws {
        let implicit = PagedKVGroupKey(kvHeads: 2,headDim: 64)
        let explicit = PagedKVGroupKey(kvHeads: 2,headDim: 64,valueHeadDim: 64)
        XCTAssertEqual(implicit,explicit); XCTAssertEqual(implicit.description,"kv2xd64-float16-full")
        let asymmetric = PagedKVGroupKey(kvHeads: 2,headDim: 192,dtype: .bfloat16,valueHeadDim: 128)
        XCTAssertNotEqual(asymmetric,PagedKVGroupKey(kvHeads: 2,headDim: 192,dtype: .bfloat16))
        XCTAssertEqual(asymmetric.geometry?.storageBytes(tokens: 16,elementBytes: 2),20480)
        XCTAssertTrue(asymmetric.description.contains("d192v128"))
        for width in [0,-1,Int.max] {
            XCTAssertThrowsError(try PagedKVPool(layerKinds: [asymPagedKind(192,width)],config: .init(capacityBytes: 1 << 20)))
        }
    }

    func testFixedAndSegmentedShuffledPageMappingBothWidthOrdersAndDTypes() throws {
        for segmented in [false,true] {
            for (dk,dv) in [(192,128),(128,192)] {
                for dtype: DType in [.float16,.bfloat16,.float32] {
                    let kind = asymPagedKind(dk,dv,heads: 2,queries: 4)
                    let pageBytes = 2*16*(dk+dv)*dtype.size
                    let pool = try PagedKVPool(layerKinds: [kind],config: .init(capacityBytes: 4 << 20,dtype: dtype,
                        maxPrefillChunk: 32,nominalMaxSequenceLength: 128,maxBufferLength: 4 << 20,
                        segmentSizeBytes: segmented ? pageBytes*3 : nil))
                    let key = pool.groupKey(forLayer: 0), group = pool.group(key)
                    try pool.reserve([key: 5])
                    if segmented { try pool.materializeReservedSegments() } else { try pool.materializeSlabs() }
                    let allocated = (0..<5).map { _ in pool.allocatePage(group: key) }, pages = Array(allocated.reversed())
                    defer { StreamOrDevice.default.stream.synchronize(); pool.freePages(group: key,pages: allocated);pool.unreserve([key:5]) }
                    let first = 3, count = 68
                    let keys = asymPagedTensor(heads: 2,start: 0,count: count,width: dk,dtype: dtype)
                    let values = asymPagedTensor(heads: 2,start: 0,count: count,width: dv,dtype: dtype,bias: -4)
                    let slots = (0..<count).map { i in pages[(first+i)/16]*16 + Int32((first+i)%16) }
                    pool.writeTokens(group: key,slots: slots,keys: keys.squeezed(axis: 0),values: values.squeezed(axis: 0))
                    let read = pool.gather(group: key,pages: pages,firstSlot: first,count: count)
                    eval(read.keys,read.values)
                    XCTAssertEqual(read.keys.shape,[1,2,count,dk]);XCTAssertEqual(read.values.shape,[1,2,count,dv])
                    XCTAssertEqual(read.keys.asData(access: .copy).data,keys.asData(access: .copy).data)
                    XCTAssertEqual(read.values.asData(access: .copy).data,values.asData(access: .copy).data)
                    // Independent host address oracle checks physical regions,
                    // not another invocation of the gather implementation.
                    let segmentData = Dictionary(uniqueKeysWithValues: group.segments.map {
                        ($0.key,$0.value.storage.asType(.float32).asArray(Float.self))
                    })
                    let fixedKeys = segmented ? [] : group.kSlab.asType(.float32).asArray(Float.self)
                    let fixedValues = segmented ? [] : group.vSlab.asType(.float32).asArray(Float.self)
                    for h in 0..<2 { for token in 0..<count {
                        let slot = Int(slots[token]), page = slot/16, within = slot%16
                        let kRaw: [Float], vRaw: [Float], base: Int, local: Int
                        if segmented {
                            let segment = group.segment(for: Int32(page))
                            kRaw = segmentData[segment.index]!;vRaw = kRaw
                            base = segment.valueOffset;local = page-segment.pages.lowerBound
                            XCTAssertEqual(segment.storage.ndim,1)
                            XCTAssertEqual(segment.byteCount,segment.pages.count*pageBytes)
                            XCTAssertGreaterThanOrEqual(segment.allocatedBytes,segment.byteCount)
                        } else {
                            kRaw = fixedKeys;vRaw = fixedValues;base = 0;local = page
                        }
                        for d in [0,dk-1] {
                            let expected = Float(h)/4 + Float(token)/32 + Float(d)/1024
                            XCTAssertEqual(kRaw[((local*2+h)*16+within)*dk+d],asymPagedRounded(expected,dtype: dtype))
                        }
                        for d in [0,dv-1] {
                            let expected = -Float(4)+Float(h)/4+Float(token)/32+Float(d)/1024
                            XCTAssertEqual(vRaw[base+((local*2+h)*16+within)*dv+d],asymPagedRounded(expected,dtype: dtype))
                        }
                    } }
                    if segmented {
                        XCTAssertGreaterThan(group.segments.count,1)
                        let firstSegment = group.segments[group.segments.keys.min()!]!
                        let rebaseLayout = try PagedKVSegmentLayout(pageCount: firstSegment.pages.count*2,
                            pageBytes: pageBytes,targetBytes: pageBytes*3,maximumBufferBytes: 4 << 20)
                        let rebased = PagedKVSegment(rebasing: firstSegment,index: 1,layout: rebaseLayout)
                        XCTAssertTrue(rebased.backing === firstSegment.backing)
                        XCTAssertEqual(rebased.valueOffset,firstSegment.valueOffset)
                        XCTAssertEqual(rebased.allocatedBytes,firstSegment.allocatedBytes)
                        for segment in group.segments.values {
                            let raw = segmentData[segment.index]!
                            XCTAssertTrue(raw[0..<2*16*dk].allSatisfy { $0 == 0 })
                            XCTAssertTrue(raw[segment.valueOffset..<segment.valueOffset+2*16*dv].allSatisfy { $0 == 0 })
                        }
                    } else {
                        XCTAssertEqual(group.keySlabBytes + group.valueSlabBytes,group.pageCount*pageBytes)
                    }
                    let empty = pool.gather(group: key,pages: [],firstSlot: 0,count: 0)
                    XCTAssertEqual(empty.keys.shape,[1,2,0,dk]);XCTAssertEqual(empty.values.shape,[1,2,0,dv])
                }
            }
        }
    }

    func testWindowWrapRollbackAndOverwriteFence() throws {
        let f = try AsymPagedFixture([asymPagedKind(window: 17)])
        let rows = try f.rows(), row = try XCTUnwrap(rows[0] as? PagedSequenceKV)
        for start in stride(from: 0,to: 80,by: 16) {
            _ = row.update(keys: asymPagedTensor(start: start,count: 16,width: 192,dtype: .float32),
                           values: asymPagedTensor(start: start,count: 16,width: 128,dtype: .float32,bias: -4))
            let s = row.snapshot();eval(s.keys,s.values)
            let count = min(start+16,17)
            XCTAssertEqual(s.values.asArray(Float.self),asymPagedValues(heads: 1,start: start+16-count,count: count,width: 128,bias: -4))
        }
        let preserved = row.snapshot();eval(preserved.keys,preserved.values)
        let expected = preserved.values.asArray(Float.self)
        row.beginSpeculativeWrite()
        _ = row.update(keys: asymPagedTensor(start: 80,count: 3,width: 192,dtype: .float32,bias: 5),
                       values: asymPagedTensor(start: 80,count: 3,width: 128,dtype: .float32,bias: 5))
        row.rollback(3);row.commitSpeculativeWrite()
        let restored = row.snapshot();eval(restored.keys,restored.values)
        XCTAssertEqual(row.absoluteOffset,80);XCTAssertEqual(restored.values.asArray(Float.self),expected)
        // The gather's back-edge must preserve old data even when constructed
        // lazily before a later page overwrite in the same evaluation graph.
        let before = row.snapshot()
        _ = row.update(keys: asymPagedTensor(start: 80,count: 32,width: 192,dtype: .float32,bias: 8),
                       values: asymPagedTensor(start: 80,count: 32,width: 128,dtype: .float32,bias: 8))
        let after = row.snapshot();eval(before.values,after.values)
        XCTAssertEqual(before.values.asArray(Float.self),expected)
    }

    func testWindowSnapshotExplicitValueWidthAndNativeBoundaryContract() throws {
        let k = asymPagedTensor(start: 16,count: 17,width: 192,dtype: .float32)
        let v = asymPagedTensor(start: 16,count: 17,width: 128,dtype: .float32)
        XCTAssertNil(CBv2PagedWindowSnapshot(keys: k,values: v,base: 16))
        XCTAssertNil(CBv2PagedWindowSnapshot(keys: k,values: v,base: 16,valueHeadDim: 192))
        XCTAssertNil(CBv2PagedWindowSnapshot(keys: k,values: v,base: Int.max,valueHeadDim: 128))
        let snapshot = try XCTUnwrap(CBv2PagedWindowSnapshot(keys: k,values: v,base: 16,valueHeadDim: 128))
        try snapshot.requireAdmissible(at: 33,window: 17)
        XCTAssertThrowsError(try snapshot.requireAdmissible(at: 32,window: 17))
        XCTAssertThrowsError(try snapshot.requireAdmissible(at: 33,window: 18))
    }

    func testFixedAttentionMissingPolicyAndUnboundOwnerRefuse() throws {
        let kind = asymPagedKind()
        XCTAssertThrowsError(try PagedKVBackend(layerKinds: [kind],config: .init(capacityBytes: 4 << 20)))
        let limits = try CBv2PagedGatheredAttentionLimits(maximumBatchSize: 1,maximumQueryTokens: 16,
            maximumContextTokens: 64,maximumInFlightGraphs: 2,maximumScratchBytes: 128 << 20)
        XCTAssertThrowsError(try PagedKVBackend(layerKinds: [kind],config: .init(capacityBytes: 128 << 20,gatheredAttention: limits)))
        let backend = try PagedKVBackend(layerKinds: [kind],config: .init(capacityBytes: 128 << 20,
            maxPrefillChunk: 16,segmentSizeBytes: 128 << 10,gatheredAttention: limits))
        XCTAssertThrowsError(try backend.makeSequenceState(layerKinds: [kind],promptLength: 0,maxLength: 64))
        XCTAssertEqual(backend.bytesReserved,0);XCTAssertEqual(backend.bytesWired,0)
        XCTAssertThrowsError(try CBv2PagedGatheredAttentionLimits(maximumBatchSize: 1,maximumQueryTokens: 16,
            maximumContextTokens: 64,maximumInFlightGraphs: 1,maximumScratchBytes: 128 << 20))
    }

    func testScratchRefusalLiveRequestBoundAndAllocationFailurePreserveOwners() throws {
        let kind = asymPagedKind()
        let tiny = try CBv2PagedGatheredAttentionLimits(maximumBatchSize: 1,maximumQueryTokens: 16,
            maximumContextTokens: 64,maximumInFlightGraphs: 2,maximumScratchBytes: 1)
        XCTAssertThrowsError(try tiny.scratchUpperBound(layerKinds: [kind],pageSize: 16))
        let f = try AsymPagedFixture([kind],batch: 1)
        XCTAssertThrowsError(try f.rows(maximum: 129))
        XCTAssertEqual(f.admission.bytesReserved,0)
        let rows = try f.rows(), bound = f.backend.pool.gatheredAttentionScratchBound
        XCTAssertGreaterThan(bound,0);XCTAssertGreaterThanOrEqual(f.admission.bytesReserved,bound + f.backend.bytesWired)
        let before = f.admission.bytesReserved
        XCTAssertThrowsError(try f.rows())
        XCTAssertEqual(f.admission.bytesReserved,before)
        f.backend.release(rows)
        XCTAssertEqual(f.backend.bytesReserved,0)
        XCTAssertEqual(f.admission.bytesReserved,bound, "pool-scoped scratch remains held, not reported as measured usage")
        f.backend.pool.slabEval = { _ in throw MLXError.caught("intentional asymmetric page allocation failure") }
        XCTAssertThrowsError(try f.rows())
        XCTAssertEqual(f.backend.bytesReserved,0);XCTAssertEqual(f.backend.bytesWired,0)
        XCTAssertEqual(f.admission.bytesReserved,bound)
    }

    func testValueOnlyBorrowerMismatchRefusesBeforeNativeAllocation() throws {
        let kinds = [asymPagedKind(),asymPagedKind(192,64,shares: 0)]
        let config = PagedKVPoolConfig(capacityBytes: 1 << 20,segmentSizeBytes: 128 << 10)
        XCTAssertThrowsError(try PagedKVPool(layerKinds: kinds,config: config))
        XCTAssertNil(PagedAttentionKernel.ineligibilityReason(headDim: 128,gqa: 2))
        XCTAssertNotNil(PagedAttentionKernel.ineligibilityReason(headDim: 192,gqa: 2), "direct fused asymmetric support is not claimed")
    }

    func testScratchLeaseDropsOnlyWithFinalPoolOwner() throws {
        var fixture: AsymPagedFixture? = try AsymPagedFixture([asymPagedKind()])
        let admission = fixture!.admission
        var rows = try fixture!.rows()
        let bound = fixture!.backend.pool.gatheredAttentionScratchBound
        fixture!.backend.release(rows)
        fixture!.owned.removeAll()
        fixture = nil
        XCTAssertEqual(admission.bytesReserved,bound, "released row aliases still retain their pool")
        rows.removeAll()
        XCTAssertEqual(admission.bytesReserved,0)
    }

    func testRealPhysicalReceiptsAndScratchShareProcessAdmissionWithoutInventedCoverage() throws {
        let denied = AsymPagedProcessReceipt(maximum: 0)
        do {
            let f = try AsymPagedFixture([asymPagedKind()],processOwner: denied)
            XCTAssertThrowsError(try f.rows())
            XCTAssertEqual(f.backend.bytesReserved,0);XCTAssertEqual(f.backend.bytesWired,0)
            XCTAssertEqual(denied.charge,0);XCTAssertEqual(denied.coverage,0)
        }
        let owner = AsymPagedProcessReceipt(maximum: 512 << 20)
        do {
            let f = try AsymPagedFixture([asymPagedKind()],processOwner: owner), rows = try f.rows()
            let actual = f.backend.pool.groupKeys.reduce(0) { $0 + f.backend.pool.group($1).segments.values.reduce(0) { $0+$1.allocatedBytes } }
            XCTAssertEqual(owner.coverage,UInt64(actual))
            XCTAssertEqual(owner.charge,UInt64(f.admission.bytesReserved))
            XCTAssertGreaterThanOrEqual(owner.charge,UInt64(actual + f.backend.pool.gatheredAttentionScratchBound))
            f.backend.release(rows)
            XCTAssertEqual(owner.coverage,0)
            XCTAssertEqual(owner.charge,UInt64(f.backend.pool.gatheredAttentionScratchBound))
        }
        XCTAssertEqual(owner.coverage,0);XCTAssertEqual(owner.charge,0)
    }
}
