import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// The side-by-side pass A (`paged_attention_part_side_impl`) gives the
    /// plain pass A's output and slab bytes: same partitions, same per-head
    /// arithmetic and order.
    @Suite(.serialized)
    struct PagedSideBySideKernelTests {
        private func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

        /// Random pages in a shuffled physical order; one row per entry of
        /// `lengths` (cached tokens before this step).
        private struct DirectCase {
            let kSlab: MLXArray
            let vSlab: MLXArray
            let tables: MLXArray
            let rows: [PagedAttentionKernel.SeqInfoRow]
        }

        private func directCase(
            dtype: DType, headDim: Int, kvHeads: Int, lengths: [Int], seed: UInt64
        ) -> DirectCase {
            let pageSize = 16
            let pages = lengths.map { ($0 + pageSize) / pageSize }
            let total = pages.reduce(0, +)
            var physical = Array(0 ..< total)
            var state = seed
            for i in stride(from: total - 1, to: 0, by: -1) {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                physical.swapAt(i, Int((state >> 33) % UInt64(i + 1)))
            }
            let width = max(8, pages.max()!)
            var table: [Int32] = []
            var rows: [PagedAttentionKernel.SeqInfoRow] = []
            var next = 0
            for (row, length) in lengths.enumerated() {
                let mine = physical[next ..< next + pages[row]].map { Int32($0) }
                next += pages[row]
                table += mine + Array(repeating: 0, count: width - mine.count)
                rows.append(
                    .init(
                        attendStart: 0, attendLength: length + 1, tableLength: pages[row],
                        writePage: mine[length / pageSize], writeSlot: length % pageSize))
            }
            let shape = [total, kvHeads, pageSize, headDim]
            let k = MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(dtype)
            let v = MLXRandom.normal(shape, key: MLXRandom.key(seed + 1)).asType(dtype)
            return DirectCase(
                kSlab: k, vSlab: v, tables: MLXArray(table, [lengths.count, width]), rows: rows)
        }

        @Test(
            arguments: [DType.bfloat16, .float16],
            [[256, 2, 6], [512, 1, 8], [128, 1, 16]])
        func directDecodeKeepsThePlainBits(dtype: DType, shape: [Int]) throws {
            let (d, kvh, gqa) = (shape[0], shape[1], shape[2])
            let lengths = [700, 1537, 2050]
            let b = lengths.count
            let q = MLXRandom.normal([b, kvh * gqa, d], key: MLXRandom.key(11)).asType(dtype)
            let kNew = MLXRandom.normal([b, kvh, d], key: MLXRandom.key(12)).asType(dtype)
            let vNew = MLXRandom.normal([b, kvh, d], key: MLXRandom.key(13)).asType(dtype)
            let source = try PagedAttentionResources.loadSourceForCurrentProcess()
            for softcap in [false, true] {
                let params = MLXArray([
                    Float(softcap ? 30 : 0), 1 / Float(d).squareRoot(), 0, 0, 0, 0, 0, 0,
                ])
                for write in [false, true] {
                    var outs: [Data] = []
                    var slabs: [(Data, Data)] = []
                    for minimum in [0, 1] {
                        let c = directCase(
                            dtype: dtype, headDim: d, kvHeads: kvh, lengths: lengths, seed: 21)
                        let rows =
                            write
                            ? c.rows
                            : c.rows.map {
                                var r = $0
                                r.writePage = 0
                                r.writeSlot = 0
                                return r
                            }
                        let (info, maxLength) = PagedAttentionKernel.seqinfo(rows)
                        let result = try PagedAttentionKernel.decode(
                            queries: q, newKeys: write ? kNew : nil, newValues: write ? vNew : nil,
                            kSlab: c.kSlab, vSlab: c.vSlab, tables: c.tables, seqinfo: info,
                            maxAttendLength: maxLength, sinks: nil, params: params,
                            softcap: softcap,
                            pageSize: 16, writeFence: MLXArray([Int32(0)]),
                            kernelSource: source, sideBySideMinimumTokens: minimum)
                        eval(result.out, c.kSlab, c.vSlab)
                        outs.append(bytes(result.out))
                        slabs.append((bytes(c.kSlab), bytes(c.vSlab)))
                    }
                    #expect(
                        outs[0] == outs[1],
                        "d\(d) gqa\(gqa) \(dtype) softcap \(softcap) write \(write)"
                    )
                    #expect(slabs[0].0 == slabs[1].0 && slabs[0].1 == slabs[1].1)
                }
            }
        }

        @Test(arguments: [DType.bfloat16, .float16])
        func segmentedDecodeKeepsThePlainBits(dtype: DType) throws {
            let kind = CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 12)
            let pageBytes = 2 * 16 * 256 * 2 * dtype.size
            var config = PagedKVPoolConfig(
                capacityBytes: 16 << 20, dtype: dtype, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 4096, maxBufferLength: 16 << 20)
            config.segmentSizeBytes = pageBytes * 5
            let plain = try PagedKVBackend(layerKinds: [kind], config: config)
            let side = try PagedKVBackend(layerKinds: [kind], config: config)
            var plainStates: [[CBv2SequenceKV?]] = []
            var sideStates: [[CBv2SequenceKV?]] = []
            defer {
                for state in plainStates { plain.release(state) }
                for state in sideStates { side.release(state) }
            }
            var plainRows: [PagedSequenceKV] = []
            var sideRows: [PagedSequenceKV] = []
            for (index, count) in [613, 1100].enumerated() {
                let a = try plain.makeSequenceState(
                    layerKinds: [kind], promptLength: 0, maxLength: 4096)
                let b = try side.makeSequenceState(
                    layerKinds: [kind], promptLength: 0, maxLength: 4096)
                plainStates.append(a)
                sideStates.append(b)
                let rowA = try #require(a[0] as? PagedSequenceKV)
                let rowB = try #require(b[0] as? PagedSequenceKV)
                let k = MLXRandom.normal([2, count, 256], key: MLXRandom.key(UInt64(40 + index)))
                    .asType(dtype)
                let v = MLXRandom.normal([2, count, 256], key: MLXRandom.key(UInt64(50 + index)))
                    .asType(dtype)
                rowA.write(keys: k, values: v)
                rowB.write(keys: k, values: v)
                plainRows.append(rowA)
                sideRows.append(rowB)
            }
            let plainGroup = plain.pool.group(plainRows[0].groupKey)
            let sideGroup = side.pool.group(sideRows[0].groupKey)
            let params = MLXArray([Float(0), 0.0625, 0, 0, 0, 0, 0, 0])
            for step in 0 ..< 3 {
                let q = MLXRandom.normal([2, 12, 256], key: MLXRandom.key(UInt64(60 + step)))
                    .asType(
                        dtype)
                let k = MLXRandom.normal([2, 2, 256], key: MLXRandom.key(UInt64(70 + step))).asType(
                    dtype)
                let v = MLXRandom.normal([2, 2, 256], key: MLXRandom.key(UInt64(80 + step))).asType(
                    dtype)
                func rows(_ list: [PagedSequenceKV]) -> [PagedSegmentDispatchPlan.Row] {
                    list.map { row in
                        let target = row.prepareDecodeWrite()
                        return .init(
                            pages: row.table,
                            info: row.seqInfoRow(
                                attending: row.decodeAttendRange, writeTarget: target))
                    }
                }
                let expected = PagedSegmentAttention.decode(
                    queries: q, newKeys: k, newValues: v, group: plainGroup, rows: rows(plainRows),
                    sinks: nil, params: params, softcap: false, source: plain.pool.kernelSource,
                    sideBySideMinimumTokens: 0)
                let got = PagedSegmentAttention.decode(
                    queries: q, newKeys: k, newValues: v, group: sideGroup, rows: rows(sideRows),
                    sinks: nil, params: params, softcap: false, source: side.pool.kernelSource,
                    sideBySideMinimumTokens: 1)
                eval(expected, got)
                #expect(bytes(got) == bytes(expected), "step \(step)")
                for (a, b) in zip(plainRows, sideRows) {
                    #expect(bytes(b.snapshot().keys) == bytes(a.snapshot().keys))
                    #expect(bytes(b.snapshot().values) == bytes(a.snapshot().values))
                }
            }
        }
    }
}
