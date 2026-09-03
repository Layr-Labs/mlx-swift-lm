// CPU-only tests for the batch-1 adaptations of the Gemma 4 mlxfast port.
//
// Every test here exercises a PURE policy or index/tiling function — no MLX
// array is constructed, so the suite runs without a Metal device and never
// contends for the GPU.

import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Gemma4SinglePromptAdaptationTests: XCTestCase {

    // MARK: - B1-LADDER: decode async-eval submission boundaries

    /// The scored cohort geometry keeps its exact boundary set in both modes.
    func testLadderCohortGeometryUnchanged() {
        for anyBatch in [false, true] {
            for layer in 0 ..< 32 {
                let expected = (0 ... 3).contains(layer)
                XCTAssertEqual(
                    gemma4ShouldSubmitDecodeAsyncEvalLadder(
                        enabled: true, schedulePrefill: false, isCBv2: true,
                        batchSize: 8, inputLength: 1, layerIndex: layer,
                        anyBatch: anyBatch),
                    expected,
                    "layer \(layer) anyBatch=\(anyBatch)")
            }
        }
    }

    /// Batch one takes the same boundaries when admitted, and none when the
    /// kill switch restores the `batchSize == 8` pin.
    func testLadderBatchOneAdmission() {
        for layer in 0 ..< 32 {
            XCTAssertEqual(
                gemma4ShouldSubmitDecodeAsyncEvalLadder(
                    enabled: true, schedulePrefill: false, isCBv2: true,
                    batchSize: 1, inputLength: 1, layerIndex: layer,
                    anyBatch: true),
                (0 ... 3).contains(layer))
            XCTAssertFalse(
                gemma4ShouldSubmitDecodeAsyncEvalLadder(
                    enabled: true, schedulePrefill: false, isCBv2: true,
                    batchSize: 1, inputLength: 1, layerIndex: layer,
                    anyBatch: false))
        }
    }

    /// Every non-batch precondition still fails closed at batch one.
    func testLadderBatchOneStillFailsClosed() {
        let cases: [(Bool, Bool, Bool, Int)] = [
            (false, false, true, 1),  // master switch off
            (true, true, true, 1),  // prefill scheduled
            (true, false, false, 1),  // not CBv2
            (true, false, true, 2),  // multi-token input
        ]
        for (enabled, prefill, isCBv2, inputLength) in cases {
            XCTAssertFalse(
                gemma4ShouldSubmitDecodeAsyncEvalLadder(
                    enabled: enabled, schedulePrefill: prefill, isCBv2: isCBv2,
                    batchSize: 1, inputLength: inputLength, layerIndex: 0,
                    anyBatch: true))
        }
        XCTAssertFalse(
            gemma4ShouldSubmitDecodeAsyncEvalLadder(
                enabled: true, schedulePrefill: false, isCBv2: true,
                batchSize: 0, inputLength: 1, layerIndex: 0, anyBatch: true))
    }

    func testLadderAnyBatchSwitchParsing() {
        XCTAssertTrue(resolveGemma4DecodeLadderAnyBatchEnabled(nil))
        XCTAssertTrue(resolveGemma4DecodeLadderAnyBatchEnabled("1"))
        for off in ["0", "false", "no", "off", "OFF", "False"] {
            XCTAssertFalse(
                resolveGemma4DecodeLadderAnyBatchEnabled(off), "value \(off)")
        }
    }

    // MARK: - GLUE-B1: fused layer glue tiling arithmetic

    /// One 704-thread threadgroup per row, four values per thread, covers
    /// `rows * 2816` exactly once with nothing out of range -- for every row
    /// count, which is what makes `rows` a grid extent rather than a tiling
    /// constant.
    func testGluePlaneCoverageIsExactAtEveryRowCount() {
        for rows in [1, 2, 3, 7, 8, 16, 33] {
            let plan = gemma4GluePlaneCoverage(rows: rows)
            XCTAssertEqual(plan.covered, rows * 2816, "rows \(rows)")
            XCTAssertEqual(plan.outOfRange, 0, "rows \(rows)")
            XCTAssertEqual(plan.duplicates, 0, "rows \(rows)")
        }
    }

    /// Regression guard for the one index that is NOT row-generic. The
    /// activation-sum store is `xSums[lid * 8 + row]` with the cohort's
    /// eight frozen in the kernel text. Below eight rows it runs off the end
    /// of the buffer; above eight it stays in range but aliases, so the table
    /// would silently carry the wrong sums. Only eight is a bijection, which
    /// is why `attentionBranchPrefix` and `dualPreNorm`'s sums arm stay
    /// pinned and the no-sums body carries the fusion everywhere else.
    func testGlueXSumsLayoutOnlyFitsTheCohort() {
        let cohort = gemma4GlueXSumsHazards(rows: 8)
        XCTAssertEqual(cohort.outOfRange, 0)
        XCTAssertEqual(cohort.collisions, 0)

        for rows in [1, 2, 3, 4, 7] {
            XCTAssertGreaterThan(
                gemma4GlueXSumsHazards(rows: rows).outOfRange, 0,
                "rows \(rows) would store past the table")
        }
        for rows in [9, 16, 33] {
            XCTAssertGreaterThan(
                gemma4GlueXSumsHazards(rows: rows).collisions, 0,
                "rows \(rows) would alias inside the table")
        }
    }

    // MARK: - ROUTER-B1: finalists admission

    /// The scored prompt rectangle is admitted in both modes and keeps the
    /// same flattened row count.
    func testRouterCohortRectangleUnchanged() {
        for anyRows in [false, true] {
            XCTAssertEqual(
                gemma4RouterAdmittedRows(batch: 8, length: 1024, anyRows: anyRows),
                8192)
            XCTAssertEqual(
                gemma4RouterAdmittedRows(batch: 8, length: 2, anyRows: anyRows), 16)
        }
    }

    /// The batch-one decode cell is admitted only with the switch on, and
    /// carries exactly one row.
    func testRouterBatchOneDecodeCell() {
        XCTAssertEqual(
            gemma4RouterAdmittedRows(batch: 1, length: 1, anyRows: true), 1)
        XCTAssertNil(
            gemma4RouterAdmittedRows(batch: 1, length: 1, anyRows: false))
        // The eight-row decode cell was excluded by `L > 1` before; with the
        // switch on it flattens to eight rows like any other plane.
        XCTAssertEqual(
            gemma4RouterAdmittedRows(batch: 8, length: 1, anyRows: true), 8)
        XCTAssertNil(
            gemma4RouterAdmittedRows(batch: 8, length: 1, anyRows: false))
    }

    /// Degenerate planes never produce a launch in either mode.
    func testRouterEmptyPlanesFailClosed() {
        for anyRows in [false, true] {
            XCTAssertNil(
                gemma4RouterAdmittedRows(batch: 0, length: 4, anyRows: anyRows))
            XCTAssertNil(
                gemma4RouterAdmittedRows(batch: 4, length: 0, anyRows: anyRows))
            XCTAssertNil(
                gemma4RouterAdmittedRows(batch: -1, length: 1, anyRows: anyRows))
        }
    }

    // MARK: - QKVNORM-B1: fused decode Q/K/V norm + RoPE row plan

    /// The cohort plan is unchanged: 128 query rows, 64 key rows, 64 value
    /// rows for the sliding geometry, 64 threads per row at D = 256.
    func testQKVNormCohortRowPlanUnchanged() {
        let plan = gemma4FusedQKVNormRowPlan(
            batch: 8, queryHeads: 16, keyHeads: 8, dimension: 256,
            keyValueShared: false)
        XCTAssertEqual(plan.queryRows, 128)
        XCTAssertEqual(plan.keyRows, 64)
        XCTAssertEqual(plan.normRows, 256)
        XCTAssertEqual(plan.threadsPerRow, 64)

        let shared = gemma4FusedQKVNormRowPlan(
            batch: 8, queryHeads: 16, keyHeads: 2, dimension: 512,
            keyValueShared: true)
        XCTAssertEqual(shared.queryRows, 128)
        XCTAssertEqual(shared.keyRows, 16)
        XCTAssertEqual(shared.normRows, 144)
        XCTAssertEqual(shared.threadsPerRow, 128)
    }

    /// The kernel recovers a row's batch index as `local_row / heads`, so the
    /// plan must lay query, key and value rows out batch-major within each
    /// segment. Replay that inverse for every row of several batches and
    /// check it lands on the right (segment, batch, head).
    func testQKVNormRowPlanInvertsToBatchAndHead() {
        for batch in [1, 2, 3, 8] {
            let queryHeads = 16
            let keyHeads = 8
            let plan = gemma4FusedQKVNormRowPlan(
                batch: batch, queryHeads: queryHeads, keyHeads: keyHeads,
                dimension: 256, keyValueShared: false)
            XCTAssertEqual(
                plan.normRows, batch * (queryHeads + 2 * keyHeads),
                "batch \(batch)")

            var seenQuery = Set<Int>()
            var seenKey = Set<Int>()
            var seenValue = Set<Int>()
            for row in 0 ..< plan.normRows {
                if row < plan.queryRows {
                    let heads = queryHeads
                    let b = row / heads
                    XCTAssertLessThan(b, batch, "query row \(row)")
                    XCTAssertTrue(seenQuery.insert(row).inserted)
                } else if row < plan.queryRows + plan.keyRows {
                    let local = row - plan.queryRows
                    let b = local / keyHeads
                    XCTAssertLessThan(b, batch, "key row \(row)")
                    XCTAssertTrue(seenKey.insert(local).inserted)
                } else {
                    let local = row - plan.queryRows - plan.keyRows
                    let b = local / keyHeads
                    XCTAssertLessThan(b, batch, "value row \(row)")
                    XCTAssertTrue(seenValue.insert(local).inserted)
                }
            }
            XCTAssertEqual(seenQuery.count, batch * queryHeads)
            XCTAssertEqual(seenKey.count, batch * keyHeads)
            XCTAssertEqual(seenValue.count, batch * keyHeads)
        }
    }

    // MARK: - KVQ-CONSUMER-GATE: q4 KV mirror capacity

    /// An unpublished capacity keeps the mirror (previous behaviour), a
    /// capacity that cannot assemble the reader's eight-row batch drops it,
    /// and any capacity that can keeps it.
    func testKVQuantMirrorCapacityPredicate() {
        XCTAssertTrue(CBv2WindowedSequenceKV.mirrorReachable(capacity: nil))
        for capacity in [1, 2, 4, 7] {
            XCTAssertFalse(
                CBv2WindowedSequenceKV.mirrorReachable(capacity: capacity),
                "capacity \(capacity)")
        }
        for capacity in [8, 9, 16, 64] {
            XCTAssertTrue(
                CBv2WindowedSequenceKV.mirrorReachable(capacity: capacity),
                "capacity \(capacity)")
        }
    }

    /// The published capacity is monotonic, so a second, smaller engine in
    /// the same process cannot turn the mirror off underneath the first.
    func testKVQuantCapacityPublicationIsMonotonic() {
        let restore = CBv2WindowedSequenceKV.decodeBatchCapacity
        defer { CBv2WindowedSequenceKV.publishDecodeBatchCapacity(restore ?? 0) }

        CBv2WindowedSequenceKV.publishDecodeBatchCapacity(16)
        XCTAssertEqual(CBv2WindowedSequenceKV.decodeBatchCapacity, 16)
        CBv2WindowedSequenceKV.publishDecodeBatchCapacity(1)
        XCTAssertEqual(CBv2WindowedSequenceKV.decodeBatchCapacity, 16)
        CBv2WindowedSequenceKV.publishDecodeBatchCapacity(0)
        XCTAssertEqual(CBv2WindowedSequenceKV.decodeBatchCapacity, 16)
        CBv2WindowedSequenceKV.publishDecodeBatchCapacity(32)
        XCTAssertEqual(CBv2WindowedSequenceKV.decodeBatchCapacity, 32)
    }

    /// The reader's admitted batch is the one `canUseRaggedTwoPassDecode`
    /// pins; if that ever changes the gate must change with it.
    func testKVQuantMirrorReaderBatchMatchesTheRaggedPin() {
        XCTAssertEqual(CBv2WindowedSequenceKV.mirrorReaderBatch, 8)
    }

    // MARK: - ARGMAX-B1: parallel greedy argmax admission

    /// The cohort rows keep the decomposition; one row falls back to stock.
    func testParallelArgMaxDefaultAdmission() {
        XCTAssertFalse(CBv2ParallelArgMaxV1.admitsRows(1))
        for rows in 2 ... 8 {
            XCTAssertTrue(CBv2ParallelArgMaxV1.admitsRows(rows), "rows \(rows)")
        }
        XCTAssertFalse(CBv2ParallelArgMaxV1.admitsRows(9))
        XCTAssertFalse(CBv2ParallelArgMaxV1.admitsRows(0))
    }

    /// `MIN_ROWS=1` restores the previous admission exactly, so the arm can
    /// be re-measured without a rebuild.
    func testParallelArgMaxMinimumRowsOverride() {
        for rows in 1 ... 8 {
            XCTAssertTrue(
                CBv2ParallelArgMaxV1.admitsRows(rows, minimumRows: 1),
                "rows \(rows)")
        }
        XCTAssertFalse(CBv2ParallelArgMaxV1.admitsRows(9, minimumRows: 1))
        XCTAssertFalse(CBv2ParallelArgMaxV1.admitsRows(0, minimumRows: 1))
        // A narrower override drops the rows below it and nothing else.
        XCTAssertFalse(CBv2ParallelArgMaxV1.admitsRows(3, minimumRows: 4))
        XCTAssertTrue(CBv2ParallelArgMaxV1.admitsRows(4, minimumRows: 4))
    }
}
