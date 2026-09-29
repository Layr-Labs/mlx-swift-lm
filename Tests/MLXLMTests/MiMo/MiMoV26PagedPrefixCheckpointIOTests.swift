import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Physical IO/copy components only: genuine native buffers and Admission,
/// NOT a native-prefix issuance/adoption, MTP-state, store or model-run proof.
/// Reuses the existing asymmetric paging fixtures; no synthetic capability.
final class MiMoV26PagedPrefixCheckpointIOTests: XCTestCase {
    private enum Failure: Error { case injectedAfterRealEvaluation, injectedBeforeAllocation }

    // Stored words, not numerically cast UInt16 values. Includes signed zero,
    // infinities and quiet NaN payloads; no arithmetic oracle/tolerance.
    private func bytes(
        heads: Int, start: Int, count: Int, width: Int,
        dtype: DType, values: Bool
    ) -> Data {
        let words16: [UInt16] = [
            0, 0x8000, 1, 0x3f80, 0xbf81, 0x7c00, 0xfc00, 0x7e53, 0x7f80, 0xff80, 0x7fc5,
        ]
        let words32: [UInt32] = [
            0, 0x8000_0000, 1, 0x3f80_0001, 0xbf81_2345, 0x7f80_0000, 0xff80_0000, 0x7fc1_2345,
        ]
        var data = Data()
        data.reserveCapacity(heads * count * width * dtype.size)
        for h in 0 ..< heads {
            for token in start ..< (start + count) {
                for d in 0 ..< width {
                    let i = h * 31 + token * 7 + d * 3 + (values ? 5 : 0)
                    if dtype.size == 2 {
                        var word = words16[i % words16.count].littleEndian
                        withUnsafeBytes(of: &word) { data.append(contentsOf: $0) }
                    } else {
                        var word = words32[i % words32.count].littleEndian
                        withUnsafeBytes(of: &word) { data.append(contentsOf: $0) }
                    }
                }
            }
        }
        return data
    }

    private func config(_ kinds: [CBv2LayerKind], dtype: DType) -> PagedKVPoolConfig {
        let k = kinds[0]
        return .init(
            capacityBytes: 32 << 20, dtype: dtype, maxPrefillChunk: 32,
            nominalMaxSequenceLength: 128, maxBufferLength: 4 << 20,
            segmentSizeBytes: 3 * k.kvHeads * 16 * (k.headDim + k.valueHeadDim) * dtype.size,
            layerDTypes: Array(repeating: dtype, count: kinds.count))
    }

    private func write(_ storage: CBv2PagedCheckpointStorage, values: Bool, data: Data) throws {
        var offset = 0
        var turn = 0
        let widths = [2, 6, 14, 254, 258, 770]
        let item = storage.plan.layers[0].key.dtype.size
        while offset < data.count {
            let requested = max(item, widths[turn % widths.count] / item * item)
            let count = min(requested, data.count - offset)
            try storage.append(
                layerIndex: 0, values: values, byteOffset: offset,
                data: data.subdata(in: offset ..< offset + count))
            offset += count
            turn += 1
        }
    }

    private func read(byteCount: Int, _ segment: (Int, Int) throws -> Data) throws -> Data {
        var result = Data()
        while result.count < byteCount {
            let piece = try segment(result.count, 259)  // split features, page/head boundaries
            XCTAssertFalse(piece.isEmpty)
            guard !piece.isEmpty else { throw CBv2CompleteCheckpointError.invalidSegment }
            result.append(piece)
        }
        XCTAssertEqual(result.count, byteCount)
        return result
    }

    // Independent physical address oracle: no gather/read-source under test.
    private func assertPhysicalBytes(
        _ storage: CBv2PagedCheckpointStorage,
        keys: Data, values: Data
    ) throws {
        let layer = storage.plan.layers[0]
        let s = storage.plan.pageSize
        let group = try XCTUnwrap(storage.groups[layer.key])
        var raw: [Int: Data] = [:]
        for (id, segment) in group.segments {
            raw[id] = segment.storage.asData(access: .copy).data
            let keyPoison = layer.key.kvHeads * s * layer.key.headDim * layer.key.dtype.size
            let valuePoison = layer.key.kvHeads * s * layer.key.valueHeadDim * layer.key.dtype.size
            XCTAssertEqual(raw[id]!.subdata(in: 0 ..< keyPoison), Data(count: keyPoison))
            let v = segment.valueOffset * layer.key.dtype.size
            XCTAssertEqual(raw[id]!.subdata(in: v ..< v + valuePoison), Data(count: valuePoison))
        }
        for h in 0 ..< layer.key.kvHeads {
            for token in 0 ..< layer.tokenCount {
                let absolute = layer.tokenStart + token
                let logicalPage = layer.ringPages.map { (absolute / s) % $0 } ?? absolute / s
                let page = group.pages[layer.firstPage + logicalPage]
                let id = group.layout.segmentIndex(page: page)
                let local = group.layout.localPage(page)
                let segment = try XCTUnwrap(group.segments[id])
                for isValue in [false, true] {
                    let d = isValue ? layer.key.valueHeadDim : layer.key.headDim
                    let rowBytes = d * layer.key.dtype.size
                    let source = (h * layer.tokenCount + token) * rowBytes
                    let destination =
                        (((local * layer.key.kvHeads + h) * s + absolute % s) * d
                            + (isValue ? segment.valueOffset : 0)) * layer.key.dtype.size
                    let expected = (isValue ? values : keys).subdata(
                        in: source ..< source + rowBytes)
                    XCTAssertEqual(
                        raw[id]!.subdata(in: destination ..< destination + rowBytes), expected)
                }
            }
        }
    }

    func testRoleSizedFullStageSegmentReadsAndPhysicalPoisonBothWidthOrders() throws {
        for (dk, dv) in [(192, 128), (128, 192), (64, 64)] {
            for dtype: DType in [.float16, .bfloat16, .float32] {
                let kinds = [asymPagedKind(dk, dv, heads: 2, queries: 4)]
                let cfg = config(kinds, dtype: dtype)
                let plan = try CBv2PagedCheckpointStoragePlan(
                    layerKinds: kinds, config: cfg, position: 35)
                XCTAssertEqual(plan.groups[0].layout.pageBytes, 2 * 16 * (dk + dv) * dtype.size)
                let admission = AdmissionV2(
                    layerKinds: [], bytesCapacity: 32 << 20,
                    config: .init(watermarkFraction: 0))
                let scratch = try Memory.allocationFootprintUpperBound(byteCount: dtype.size)
                let lease = try admission.reserveCheckpointStage(
                    targetBytes: plan.nativeBytes,
                    auxiliaryBytes: 0, scratchBytes: scratch)
                func exercise() throws {
                    let storage = try CBv2PagedCheckpointStorage(
                        plan: plan,
                        evaluate: {
                            array in
                            try withError { fault in
                                eval(array)
                                try fault.check()
                            }
                        }, admission: admission)
                    defer { storage.close() }
                    let keys = bytes(
                        heads: 2, start: 0, count: 35, width: dk, dtype: dtype, values: false)
                    let values = bytes(
                        heads: 2, start: 0, count: 35, width: dv, dtype: dtype, values: true)
                    try write(storage, values: false, data: keys)
                    try write(storage, values: true, data: values)
                    try assertPhysicalBytes(storage, keys: keys, values: values)
                    XCTAssertGreaterThan(storage.allocatedBytes, 0)
                    XCTAssertLessThanOrEqual(storage.allocatedBytes, plan.nativeBytes)
                    for isValue in [false, true] {
                        let source = try CBv2PagedCheckpointTensorSource(
                            storage: storage,
                            layerIndex: 0, values: isValue, admission: admission)
                        defer { source.close() }
                        let expected = isValue ? values : keys
                        let descriptor = try CBv2CheckpointTensorDescriptor(
                            role: isValue ? .values : .keys, layer: 0,
                            shape: [1, 2, 35, isValue ? dv : dk],
                            dtype: XCTUnwrap(CBv2CheckpointDType(dtype)))
                        XCTAssertTrue(source.matches(descriptor))
                        XCTAssertEqual(source.byteCount, expected.count)
                        XCTAssertEqual(
                            try read(byteCount: source.byteCount, source.readSegment), expected)
                    }
                    XCTAssertThrowsError(
                        try storage.append(
                            layerIndex: 0, values: true,
                            byteOffset: values.count, data: Data(count: dtype.size)))
                    XCTAssertThrowsError(
                        try storage.append(
                            layerIndex: 0, values: true,
                            byteOffset: 1, data: Data(count: dtype.size)))
                }
                do { try exercise() } catch {
                    lease.closeAfterDroppingOwners()
                    throw error
                }
                lease.closeAfterDroppingOwners()
                XCTAssertEqual(admission.bytesReserved, 0)
            }
        }
    }

    func testRoleSizedWrappedRingStagingUsesAbsolutePhysicalSlots() throws {
        for (dk, dv) in [(192, 128), (128, 192)] {
            for position in [2, 17, 47, 65] {
                let dtype = DType.bfloat16
                let kinds = [asymPagedKind(dk, dv, window: 17, heads: 2, queries: 4)]
                let cfg = config(kinds, dtype: dtype)
                let layout = try CBv2HistoricalAttentionLayout(
                    layerKinds: kinds,
                    dtypes: [dtype], allowAsymmetric: true)
                let plan = try CBv2PagedCheckpointStoragePlan(
                    layerKinds: kinds, config: cfg,
                    position: position, historicalLayout: layout, maximumSequenceLength: 128)
                let admission = AdmissionV2(
                    layerKinds: [], bytesCapacity: 32 << 20,
                    config: .init(watermarkFraction: 0))
                let lease = try admission.reserveCheckpointStage(
                    targetBytes: plan.nativeBytes,
                    auxiliaryBytes: 0, scratchBytes: 64 << 10)
                func exercise() throws {
                    let storage = try CBv2PagedCheckpointStorage(
                        plan: plan,
                        evaluate: {
                            array in
                            try withError { fault in
                                eval(array)
                                try fault.check()
                            }
                        }, admission: admission)
                    defer { storage.close() }
                    let start = max(0, position - 17)
                    let count = min(17, position)
                    XCTAssertEqual(plan.layers[0].tokenStart, start)
                    XCTAssertNotNil(plan.layers[0].ringPages)
                    let keys = bytes(
                        heads: 2, start: start, count: count, width: dk, dtype: dtype, values: false
                    )
                    let values = bytes(
                        heads: 2, start: start, count: count, width: dv, dtype: dtype, values: true)
                    try write(storage, values: false, data: keys)
                    try write(storage, values: true, data: values)
                    try assertPhysicalBytes(storage, keys: keys, values: values)
                    // A full-page source must still refuse this ring; windows
                    // need the dedicated immutable historical copy.
                    XCTAssertThrowsError(
                        try CBv2PagedCheckpointTensorSource(
                            storage: storage,
                            layerIndex: 0, values: true, admission: admission))
                }
                do { try exercise() } catch {
                    lease.closeAfterDroppingOwners()
                    throw error
                }
                lease.closeAfterDroppingOwners()
                XCTAssertEqual(admission.bytesReserved, 0)
            }
        }
    }

    private func write(_ row: PagedSequenceKV, start: Int, count: Int) {
        let key = row.groupKey
        let k = bytes(
            heads: key.kvHeads, start: start, count: count,
            width: key.headDim, dtype: key.dtype, values: false)
        let v = bytes(
            heads: key.kvHeads, start: start, count: count,
            width: key.valueHeadDim, dtype: key.dtype, values: true)
        row.write(
            keys: MLXArray(k, [key.kvHeads, count, key.headDim], dtype: key.dtype),
            values: MLXArray(v, [key.kvHeads, count, key.valueHeadDim], dtype: key.dtype))
    }

    func testCapturedAsymmetricNativeBytesSurviveDonorWrapAndLastRoleOwnsCharge() throws {
        for (dk, dv) in [(192, 128), (128, 192), (64, 64)] {
            for dtype: DType in [.float16, .bfloat16, .float32] {
                for position in [2, 47] {
                    let f = try AsymPagedFixture(
                        [asymPagedKind(dk, dv, window: 17, heads: 2, queries: 4)],
                        dtype: dtype)
                    let row = try XCTUnwrap(try f.rows()[0] as? PagedSequenceKV)
                    for start in stride(from: 0, to: position, by: 16) {
                        write(row, start: start, count: min(16, position - start))
                    }
                    let group = row.pool.group(row.groupKey)
                    let previous = group.writeFence
                    let before = f.admission.bytesReserved
                    var window: CBv2HistoricalWindow? = try .init(
                        row: row, position: position, admission: f.admission)
                    weak var actual = window
                    let charge = try XCTUnwrap(window).reservedBytes
                    XCTAssertTrue(
                        group.writeFence === previous, "private copy cannot replace serving fence")
                    XCTAssertEqual(window?.evaluationRoots.count, dk == dv ? 1 : 2)
                    XCTAssertEqual(window?.evaluationRoot == nil, dk != dv)
                    let sources = [
                        CBv2HistoricalWindowTensorSource(
                            window: try XCTUnwrap(window), values: false),
                        CBv2HistoricalWindowTensorSource(
                            window: try XCTUnwrap(window), values: true),
                    ]
                    window?.markSubmitted()
                    try withError { fault in
                        asyncEval(try XCTUnwrap(window).evaluationRoots)
                        try fault.check()
                    }
                    try window?.finishEvaluation()
                    var actualCopyBytes = 0
                    for root in try XCTUnwrap(window).evaluationRoots {
                        let info = try XCTUnwrap(root.evaluatedBufferInfo())
                        XCTAssertTrue(info.isRowContiguous)
                        XCTAssertEqual(info.dataOffset, 0)
                        XCTAssertEqual(info.dataElements, root.size)
                        XCTAssertLessThanOrEqual(
                            info.allocatedBytes,
                            try Memory.allocationFootprintUpperBound(byteCount: root.nbytes))
                        actualCopyBytes += info.allocatedBytes
                    }
                    XCTAssertGreaterThan(actualCopyBytes, 0)
                    XCTAssertLessThan(actualCopyBytes, charge)
                    XCTAssertEqual(row.absoluteOffset, position)
                    for start in stride(from: position, to: position + 64, by: 16) {
                        write(row, start: start, count: 16)
                    }
                    try withError { fault in
                        eval(group.writeFence)
                        try fault.check()
                    }
                    XCTAssertEqual(row.absoluteOffset, position + 64)
                    window = nil
                    XCTAssertNotNil(actual)
                    XCTAssertEqual(f.admission.bytesReserved, before + charge)
                    // Value first exercises its separate evaluated Depends result.
                    for isValue in [true, false] {
                        let d = isValue ? dv : dk
                        let source = sources[isValue ? 1 : 0]
                        let count = min(17, position)
                        let expected = bytes(
                            heads: 2, start: position - count, count: count,
                            width: d, dtype: dtype, values: isValue)
                        let descriptor = try CBv2CheckpointTensorDescriptor(
                            role: isValue ? .values : .keys,
                            layer: 0, shape: [1, 2, count, d],
                            dtype: XCTUnwrap(CBv2CheckpointDType(dtype)))
                        XCTAssertTrue(source.matches(descriptor))
                        XCTAssertEqual(
                            try read(byteCount: expected.count, source.readSegment), expected)
                    }
                    sources[0].close()
                    XCTAssertNotNil(actual)
                    XCTAssertEqual(f.admission.bytesReserved, before + charge)
                    sources[1].close()
                    XCTAssertNil(actual)
                    XCTAssertEqual(f.admission.bytesReserved, before)
                }
            }
        }
    }

    func testCaptureRefusesBeforeAllocationAndKeepsRequiredCompletionFailure() throws {
        let f = try AsymPagedFixture([asymPagedKind(window: 17)], dtype: .bfloat16)
        let row = try XCTUnwrap(try f.rows()[0] as? PagedSequenceKV)
        write(row, start: 0, count: 16)
        write(row, start: 16, count: 16)
        let tiny = AdmissionV2(
            layerKinds: [], bytesCapacity: 1, config: .init(watermarkFraction: 0))
        var allocations = 0
        XCTAssertThrowsError(
            try CBv2HistoricalWindow(
                row: row, position: 32, admission: tiny,
                beforeAllocation: { allocations += 1 }))
        XCTAssertEqual(allocations, 0)
        XCTAssertEqual(tiny.bytesReserved, 0)
        let before = f.admission.bytesReserved
        XCTAssertThrowsError(
            try CBv2HistoricalWindow(
                row: row, position: 32, admission: f.admission,
                beforeAllocation: { throw Failure.injectedBeforeAllocation }))
        XCTAssertEqual(f.admission.bytesReserved, before)
        let group = row.pool.group(row.groupKey)
        let previous = group.writeFence
        var evaluations = 0
        var drains = 0
        var window: CBv2HistoricalWindow? = try .init(
            row: row, position: 32, admission: f.admission,
            evaluate: { array in
                try withError { fault in
                    eval(array)
                    try fault.check()
                }
                evaluations += 1
                throw Failure.injectedAfterRealEvaluation
            },
            synchronize: { stream in
                try withError { fault in
                    stream.stream.synchronize()
                    try fault.check()
                }
                drains += 1
            })
        let charge = try XCTUnwrap(window).reservedBytes
        XCTAssertThrowsError(try window?.finishEvaluation())
        XCTAssertEqual(evaluations, 1)
        XCTAssertEqual(drains, 1)
        XCTAssertTrue(group.writeFence === previous)
        XCTAssertThrowsError(try window?.finishEvaluation())
        XCTAssertEqual(evaluations, 1, "a failed required completion is not retried")
        XCTAssertEqual(f.admission.bytesReserved, before + charge)
        window = nil  // the actual successful drain above precedes this release
        XCTAssertEqual(drains, 1)
        XCTAssertEqual(f.admission.bytesReserved, before)
        // Failed drain / native first-winner quarantine belongs to the
        // still-closed native owner integration, not this untracked component.
    }

    func testAsymmetricCodecAdmissionRemainsClosedAndInvalidGeometryRefuses() throws {
        let f = try AsymPagedFixture([asymPagedKind(window: 17)])
        let codec = CBv2CompleteCheckpointCodec(
            identity: .init(
                modelAggregateHash: "io-component",
                promptContractID: "text", buildID: "unissued", numericsFingerprint: "native"),
            layerKinds: f.kinds, recurrentSpec: nil, kvDTypes: [.float32],
            assistant: nil, admission: f.admission, pagedConfig: f.backend.pool.config)
        XCTAssertNil(codec.historicalLayout)
        XCTAssertThrowsError(try codec.tensorDescriptors(position: 32))
        XCTAssertThrowsError(
            try CBv2PagedCheckpointStoragePlan(
                layerKinds: f.kinds,
                config: f.backend.pool.config, position: Int.max))
        let malformed = [asymPagedKind(192, Int.max)]
        XCTAssertThrowsError(
            try CBv2PagedCheckpointStoragePlan(
                layerKinds: malformed,
                config: config([asymPagedKind()], dtype: .float32), position: 32))
    }
}
