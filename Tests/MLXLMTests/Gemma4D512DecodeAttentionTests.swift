// CPU-only tests for C1 rung 1 — the D=512 composed decode attention chain
// lifted to one row (COMPOSED-B1).
//
// Every test here exercises a PURE policy or index/tiling function, or a
// Swift re-implementation of a kernel's addressing, so the suite runs without
// a Metal device and never contends for the GPU.

import XCTest

@testable import MLXLMCommon

final class Gemma4D512DecodeAttentionTests: XCTestCase {

    typealias D512 = CBv2RaggedComposedD512DecodeAttentionV1

    // MARK: - Switch policy

    /// Both switches default ON and rung 2 wins, so an unset environment
    /// leaves the composed chain dark at one row.
    func testComposedStandsDownWhenTwoPassIsDefaultOn() {
        XCTAssertFalse(
            D512.composedBatchOneAdmitted(composedRaw: nil, twoPassRaw: nil))
        XCTAssertFalse(
            D512.composedBatchOneAdmitted(composedRaw: "1", twoPassRaw: nil))
        XCTAssertFalse(
            D512.composedBatchOneAdmitted(composedRaw: "1", twoPassRaw: "1"))
    }

    /// Turning rung 2 off hands the cell to rung 1, and rung 1's own switch
    /// still removes it.
    func testComposedAdmittedOnlyWithTwoPassOff() {
        for twoPassOff in ["0", "false", "no", "off", "OFF", "False"] {
            XCTAssertTrue(
                D512.composedBatchOneAdmitted(
                    composedRaw: nil, twoPassRaw: twoPassOff),
                "twoPass=\(twoPassOff)")
            for composedOff in ["0", "false", "no", "off"] {
                XCTAssertFalse(
                    D512.composedBatchOneAdmitted(
                        composedRaw: composedOff, twoPassRaw: twoPassOff),
                    "composed=\(composedOff) twoPass=\(twoPassOff)")
            }
        }
    }

    // MARK: - Row admission

    /// The scored cohort is admitted whatever the switch says; row counts
    /// between 1 and 8 never are.
    func testCohortAlwaysAdmittedAndMiddleRowCountsNever() {
        XCTAssertTrue(D512.admits(rowCount: 8))
        for rows in [0, 2, 3, 4, 5, 6, 7, 9, 16] {
            XCTAssertFalse(D512.admits(rowCount: rows), "rows \(rows)")
        }
    }

    /// The cohort keeps its transcribed 4095 ceiling; one row does not.
    func testKeyLengthCeilingIsPerRowCount() {
        XCTAssertEqual(D512.maxKeyLength(rowCount: 8), 4095)
        XCTAssertEqual(D512.maxKeyLength(rowCount: 1), Int.max)
        XCTAssertGreaterThan(D512.maxKeyLength(rowCount: 1), 17408)
    }

    // MARK: - Softmax body selection and launch geometry

    /// Reproduces softmax.cpp:55 (`axis_size > SOFTMAX_LOOPED_LIMIT`) and
    /// :64-68 (`32 * ceil(ceil(kL / 4) / 32)`).
    func testSoftmaxLaunchMatchesStockSelection() {
        // Below and at the limit: the block body, one thread per 4 elements
        // rounded up to whole simdgroups.
        for (kL, threads) in [(4, 32), (128, 32), (129, 64), (1024, 256),
                              (1100, 288), (4096, 1024)]
        {
            let launch = D512.softmaxLaunch(keyLength: kL, rows: 16)
            XCTAssertFalse(launch.looped, "kL \(kL)")
            XCTAssertEqual(launch.threads, threads, "kL \(kL)")
            XCTAssertEqual(launch.gridThreads, threads * 16, "kL \(kL)")
            XCTAssertLessThanOrEqual(launch.threads, 1024, "kL \(kL)")
        }
        // Above it: the looped body at a fixed 1024-wide threadgroup.
        for kL in [4097, 8192, 17408, 65536] {
            let launch = D512.softmaxLaunch(keyLength: kL, rows: 16)
            XCTAssertTrue(launch.looped, "kL \(kL)")
            XCTAssertEqual(launch.threads, 1024, "kL \(kL)")
            XCTAssertEqual(launch.gridThreads, 1024 * 16, "kL \(kL)")
        }
    }

    /// THE TEST's key length is exactly the case the old 4095 pin excluded,
    /// and the block body could not have run there: it needs one thread per
    /// four elements, which is 4352 threads — over the Metal maximum.
    func testTestShapeNeedsTheLoopedBody() {
        let kL = 17408
        XCTAssertGreaterThan((kL + 3) / 4, 1024)
        XCTAssertTrue(D512.softmaxLaunch(keyLength: kL, rows: 16).looped)
    }

    /// Every element of every row is covered exactly once per pass by the
    /// looped body's `offset = r * lsize * N_READS + lid * N_READS` walk, and
    /// nothing outside the row is ever stored.
    func testLoopedSoftmaxOffsetsCoverTheRowExactlyOnce() {
        let lsize = D512.loopedSoftmaxThreads
        let nReads = 4
        for axisSize in [4097, 8192, 17408, 17409, 65536, 100_003] {
            let rounds = (axisSize + nReads * lsize - 1) / (nReads * lsize)
            var touched = [Int](repeating: 0, count: axisSize)
            for r in 0 ..< rounds {
                for lid in 0 ..< lsize {
                    let offset = r * lsize * nReads + lid * nReads
                    for i in 0 ..< nReads where offset + i < axisSize {
                        touched[offset + i] += 1
                    }
                }
            }
            XCTAssertEqual(
                touched.filter { $0 != 1 }.count, 0,
                "axis \(axisSize): every element written exactly once")
        }
    }

    // MARK: - Dispatch-1 (QKᵀ) tiling at one row

    /// The QK kernel's z decomposition. Reproduces the kernel text:
    ///   n_chunks = (kL + 63) / 64; chunk = z % n_chunks;
    ///   row = (z / n_chunks) / 2; kv_head = (z / n_chunks) % 2.
    /// At one row every z maps to row 0, so the k1…k7 slots the host pads
    /// with plane 0 are selected by no thread.
    func testQKZDecompositionNeverSelectsAPaddedSlot() {
        for kL in [4, 17, 1024, 1100, 17408] {
            let chunks = (kL + 63) / 64
            for batch in [1, 8] {
                var seenRows = Set<Int>()
                for z in 0 ..< (batch * 2 * chunks) {
                    let rowKV = z / chunks
                    let row = rowKV / 2
                    let kvHead = rowKV % 2
                    XCTAssertTrue(
                        (0 ..< batch).contains(row),
                        "kL \(kL) batch \(batch) z \(z) row \(row)")
                    XCTAssertTrue((0 ... 1).contains(kvHead))
                    seenRows.insert(row)
                }
                XCTAssertEqual(seenRows, Set(0 ..< batch))
            }
        }
    }

    /// Dispatch 1 covers every score row, and the tail shift never addresses
    /// a key past kL — the property that makes the duplicate tail writes
    /// value-identical rather than out of bounds.
    func testQKScoreRowsFullyCoveredWithInBoundsTailShift() {
        for kL in [4, 5, 16, 17, 63, 64, 65, 1100, 4095, 4096, 17408] {
            let chunks = (kL + 63) / 64
            let virtualGroups = (kL + 15) / 16
            var covered = [Bool](repeating: false, count: kL)
            for chunk in 0 ..< chunks {
                let lo = chunk * 4
                let hi = min(lo + 4, virtualGroups)
                for vtg in lo ..< hi {
                    for sg in 0 ..< 4 {
                        var outRow = vtg * 16 + sg * 4
                        if outRow >= kL { continue }
                        outRow = outRow + 4 <= kL ? outRow : kL - 4
                        XCTAssertGreaterThanOrEqual(outRow, 0, "kL \(kL)")
                        XCTAssertLessThanOrEqual(outRow + 4, kL, "kL \(kL)")
                        for t in 0 ..< 4 { covered[outRow + t] = true }
                    }
                }
            }
            XCTAssertFalse(
                covered.contains(false), "kL \(kL): every score row written")
        }
    }

    // MARK: - Dispatch-3 (probs·V) tiling

    /// The AV kernel walks the key axis as `n_iter` full 32-key blocks
    /// (`bm = thrM * 4`, `bm += 32`) plus a `leftover` tail guarded by
    /// `bm + tm < key_length`. Model both loops over the eight `thrM` lanes:
    /// every key must be consumed exactly once and none past kL.
    func testAVKeyBlocksCoverTheKeyAxisExactlyOnce() {
        for kL in [4, 31, 32, 33, 63, 64, 65, 1100, 4095, 17408, 17411] {
            let nIter = kL / 32
            let leftover = kL - nIter * 32
            var touched = [Int](repeating: 0, count: kL)
            for thrM in 0 ..< 8 {
                var bm = thrM * 4
                for _ in 0 ..< nIter {
                    for tm in 0 ..< 4 {
                        XCTAssertLessThan(bm + tm, kL, "kL \(kL) main loop")
                        touched[bm + tm] += 1
                    }
                    bm += 32
                }
                if leftover > 0 {
                    var tm = 0
                    while tm < 4 && bm + tm < kL {
                        touched[bm + tm] += 1
                        tm += 1
                    }
                }
            }
            XCTAssertEqual(
                touched.filter { $0 != 1 }.count, 0,
                "kL \(kL): every key consumed exactly once")
        }
    }

    /// The AV column tiles partition the 512 output columns with no overlap
    /// and no gap at either admitted tiling.
    func testAVColumnTilesPartitionTheHeadDim() {
        for tiles in [8, 16] {
            let tileColumns = 512 / tiles
            let simdgroups = tileColumns / 16
            var covered = [Bool](repeating: false, count: 512)
            for tile in 0 ..< tiles {
                for sg in 0 ..< simdgroups {
                    for thrN in 0 ..< 4 {
                        let outCol = tile * tileColumns + (4 * sg + thrN) * 4
                        for c in 0 ..< 4 {
                            XCTAssertFalse(
                                covered[outCol + c],
                                "tiles \(tiles): column \(outCol + c) twice")
                            covered[outCol + c] = true
                        }
                    }
                }
            }
            XCTAssertFalse(covered.contains(false), "tiles \(tiles)")
        }
    }
}
